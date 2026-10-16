// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked Arrow IPC reader
//!
//! Reads a directory of `<base>-NNNNNNNNNN.arrow` files emitted by
//! `ChunkedArrowWriter` and presents them as an ordered iterator of
//! `Table`s. Each chunk file is a complete, independently readable Arrow
//! IPC file. The reader includes files in nested partition
//! directories and sorts files by partition directory and then by numeric
//! index. Consumers of an unpartitioned directory see batches in write
//! order.
//!
//! ## Concurrency model
//!
//! The default [`Iterator`] path is sync and serial: each `next()` opens
//! one chunk, reads the footer + record batches, and closes the file
//! before the next is opened. There is no benefit to making per-file
//! reads async - opening and reading a complete Arrow IPC file is
//! syscall-bound, not concurrency-bound, and `tokio::fs` would only add
//! `spawn_blocking` overhead with no I/O parallelism in return.
//!
//! Across files the picture changes: chunk files are independent, so
//! parallel reads scale roughly with disk queue depth and core count.
//! [`ChunkedTableReader::par_load_batched`] (inherited from the trait) uses
//! `std::thread::scope` to fan per-chunk work (open + footer + body +
//! decode) across worker threads and returns a `SuperTable` with batches
//! in write order.

use std::io;
use std::path::{Path, PathBuf};

use std::sync::Arc;

use minarrow::{Concatenate, SuperTable, Table};

use crate::models::readers::ipc::file_table::FileTableReader;
use crate::models::readers::object_storage::list_files;
use crate::traits::chunked_table_reader::ChunkedTableReader;

/// Iterator over chunk files in a directory written by
/// `ChunkedArrowWriter`. Yields one `Table` per chunk file in ascending
/// index order; the previous chunk's reader is dropped before the next
/// is opened.
pub struct ChunkedArrowReader {
    paths: Vec<PathBuf>,
    cursor: usize,
}

impl ChunkedTableReader for ChunkedArrowReader {
    type Error = io::Error;
    type Options = ();

    fn open<P: AsRef<Path>>(dir: P, base: &str, _options: ()) -> io::Result<Self> {
        let paths = Self::list_paths(dir, base)?;
        Ok(Self { paths, cursor: 0 })
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn list_paths<P: AsRef<Path>>(dir: P, base: &str) -> io::Result<Vec<PathBuf>> {
        let prefix = format!("{base}-");
        let mut indexed: Vec<(u64, PathBuf)> = Vec::new();
        for path in list_files(dir)? {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !name.starts_with(&prefix) || !name.ends_with(".arrow") {
                continue;
            }
            let index_str = &name[prefix.len()..name.len() - ".arrow".len()];
            let Ok(index) = index_str.parse::<u64>() else {
                continue;
            };
            indexed.push((index, path));
        }
        // Order by partition directory, then by index within it, which is
        // write order for an unpartitioned directory and time order for
        // `PartitionType::Auto` partitions.
        indexed.sort_by(|(a_index, a), (b_index, b)| {
            a.parent().cmp(&b.parent()).then(a_index.cmp(b_index))
        });
        Ok(indexed.into_iter().map(|(_, p)| p).collect())
    }

    fn read_chunk(&self, path: &Path) -> io::Result<Table> {
        let reader = FileTableReader::open(path)?;
        let n = reader.num_batches();
        if n == 0 {
            return Ok(Table::default());
        }
        let mut accumulated: Option<Table> = None;
        for i in 0..n {
            let batch = reader.read_batch(i)?;
            accumulated = Some(match accumulated {
                None => batch,
                Some(existing) => existing.concat(batch).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("concat: {e}"))
                })?,
            });
        }
        Ok(accumulated.unwrap_or_default())
    }

    fn read_chunk_cols(&self, path: &Path, columns: &[&str]) -> io::Result<Table> {
        // Push projection into the chunk's IPC decoder via
        // `FileTableReader::load_table_cols`, which feeds the column
        // set to `decode_record_batch` so non-selected columns'
        // buffers are skipped at decode time.
        FileTableReader::open(path)?.load_table_cols(columns)
    }

    fn load_batched_cols(self, columns: &[&str]) -> io::Result<SuperTable> {
        let mut batches: Vec<Arc<Table>> = Vec::new();
        let mut name: Option<String> = None;
        for path in &self.paths[self.cursor..] {
            let table = self.read_chunk_cols(path, columns)?;
            if name.is_none() {
                name = Some(table.name.clone());
            }
            batches.push(Arc::new(table));
        }
        Ok(SuperTable::from_batches(
            batches,
            name.or(Some("chunked".into())),
        ))
    }
}

impl Iterator for ChunkedArrowReader {
    type Item = io::Result<Table>;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.paths.get(self.cursor)?.clone();
        self.cursor += 1;
        Some(self.read_chunk(&path))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use std::sync::Arc;

    use crate::models::writers::chunked::arrow::ChunkedArrowWriter;
    use crate::models::writers::chunked::partition::PartitionType;
    use crate::traits::chunked_table_writer::ChunkedTableWriter;
    use minarrow::{Table, Vec64, fa_i32};

    /// 2026-02-09T07:00:00Z and 08:00:00Z in epoch nanoseconds.
    #[cfg(feature = "datetime")]
    const SEVEN_AM: u64 = 1_770_620_400_000_000_000;
    #[cfg(feature = "datetime")]
    const EIGHT_AM: u64 = SEVEN_AM + 3_600_000_000_000;

    /// Table of `n_rows` rows, identifying each chunk by its row count.
    fn rows(n_rows: i32) -> Table {
        let values: Vec64<i32> = (0..n_rows).collect();
        Table::new("b", Some(vec![fa_i32!("n", @vec64 values)]))
    }

    #[test]
    fn par_load_batched_returns_batches_in_write_order() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_par_reader");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedArrowWriter::new(&dir, "part", None).unwrap();
        for i in 0..16i32 {
            w.write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!("n", i, i + 100)]),
            ))
            .unwrap();
        }

        let st = ChunkedArrowReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert_eq!(st.batches.len(), 16);
        for (i, batch) in st.batches.iter().enumerate() {
            assert_eq!(batch.n_rows, 2);
            // First column's first row should be `i`, confirming write order
            // round-trips through parallel decode unchanged.
            let arr = &batch.cols[0].array;
            let s = format!("{arr:?}");
            assert!(
                s.contains(&format!("{i}")),
                "batch {i} first column did not contain {i}: {s}"
            );
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn par_load_batched_handles_empty_directory() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_par_reader_empty");
        let _ = fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let st = ChunkedArrowReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert!(st.batches.is_empty());
        assert_eq!(st.n_rows, 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn auto_partitioned_chunks_read_back_by_partition_then_index() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_auto_partition");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedArrowWriter::new(&dir, "part", Some(PartitionType::Auto)).unwrap();
        let p0 = w.write_chunk_for_time(&rows(1), EIGHT_AM).unwrap();
        let p1 = w.write_chunk_for_time(&rows(2), SEVEN_AM).unwrap();
        let p2 = w.write_chunk_for_time(&rows(3), EIGHT_AM).unwrap();
        assert_eq!(p0, dir.join("date=2026-02-09/hour=08/part-0000000000.arrow"));
        assert_eq!(p1, dir.join("date=2026-02-09/hour=07/part-0000000001.arrow"));
        assert_eq!(p2, dir.join("date=2026-02-09/hour=08/part-0000000002.arrow"));

        // The 07:00 chunk precedes both 08:00 chunks, which keep write order.
        let all = ChunkedArrowReader::open(&dir, "part", ()).unwrap().load_batched().unwrap();
        let n_rows: Vec<usize> = all.batches.iter().map(|batch| batch.n_rows).collect();
        assert_eq!(n_rows, vec![2, 1, 3]);

        // A partition directory reads that partition alone.
        let eight = ChunkedArrowReader::open(dir.join("date=2026-02-09/hour=08"), "part", ())
            .unwrap()
            .load_batched()
            .unwrap();
        let n_rows: Vec<usize> = eight.batches.iter().map(|batch| batch.n_rows).collect();
        assert_eq!(n_rows, vec![1, 3]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn custom_partitions_come_from_the_table() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_custom_partition");
        let _ = fs::remove_dir_all(&dir);

        let by_size = PartitionType::Custom(Arc::new(|table: &Table, _time: u64| {
            format!("rows={}", table.n_rows)
        }));
        let mut w = ChunkedArrowWriter::new(&dir, "part", Some(by_size)).unwrap();
        let path = w.write_chunk(&rows(4)).unwrap();
        assert_eq!(path, dir.join("rows=4/part-0000000000.arrow"));

        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn par_write_all_creates_partition_directories() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_par_partition");
        let _ = fs::remove_dir_all(&dir);

        let w = ChunkedArrowWriter::new(&dir, "part", Some(PartitionType::Auto)).unwrap();
        let tables: Vec<Table> = (1..=6).map(rows).collect();
        let refs: Vec<&Table> = tables.iter().collect();
        let paths = w.par_write_all(&refs, Some(3)).unwrap();

        // Every batch of one call shares one time and therefore one partition.
        let partition = paths[0].parent().unwrap();
        assert!(partition.starts_with(&dir) && partition != dir);
        assert!(paths.iter().all(|path| path.parent() == Some(partition)));

        let read = ChunkedArrowReader::par_load_batched(&dir, "part", (), None).unwrap();
        let n_rows: Vec<usize> = read.batches.iter().map(|batch| batch.n_rows).collect();
        assert_eq!(n_rows, vec![1, 2, 3, 4, 5, 6]);

        fs::remove_dir_all(&dir).ok();
    }
}
