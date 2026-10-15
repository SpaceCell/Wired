// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked CSV reader
//!
//! Reads a directory of `<base>-NNNNNNNNNN.csv` files emitted by
//! `ChunkedCsvWriter` and presents them as an ordered iterator of `Table`s.
//! Each chunk file is a complete, independently parseable CSV (header +
//! rows). The reader includes files in nested partition
//! directories and sorts files by partition directory and then by numeric
//! index. Consumers of an unpartitioned directory see batches in write
//! order.
//!
//! Inherits [`ChunkedTableReader::par_load_batched`] for sync parallel
//! decode across chunk files.

use std::io;
use std::path::{Path, PathBuf};

use std::sync::Arc;

use minarrow::{ColumnSelection, Concatenate, SuperTable, Table};

use crate::models::decoders::csv::CsvDecodeOptions;
use crate::models::readers::csv::CsvReader;
use crate::models::readers::object_storage::{StorageReader, list_files};
use crate::traits::chunked_table_reader::ChunkedTableReader;

/// Per-format `Options` for `ChunkedCsvReader::open`.
#[derive(Debug, Clone)]
pub struct ChunkedCsvReadOptions {
    pub decode: CsvDecodeOptions,
    pub batch_size: usize,
}

impl Default for ChunkedCsvReadOptions {
    fn default() -> Self {
        Self {
            decode: CsvDecodeOptions::default(),
            batch_size: usize::MAX,
        }
    }
}

/// Iterator over chunk files in a directory written by `ChunkedCsvWriter`.
///
/// Yields one `Table` per chunk file in ascending index order. `next` reads
/// the next file lazily; the previous file's reader is dropped before the
/// next is opened, so file handles do not accumulate.
pub struct ChunkedCsvReader {
    paths: Vec<PathBuf>,
    cursor: usize,
    options: ChunkedCsvReadOptions,
}

impl ChunkedTableReader for ChunkedCsvReader {
    type Error = io::Error;
    type Options = ChunkedCsvReadOptions;

    fn open<P: AsRef<Path>>(
        dir: P,
        base: &str,
        options: ChunkedCsvReadOptions,
    ) -> io::Result<Self> {
        let paths = Self::list_paths(dir, base)?;
        Ok(Self {
            paths,
            cursor: 0,
            options,
        })
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
            if !name.starts_with(&prefix) || !name.ends_with(".csv") {
                continue;
            }
            let index_str = &name[prefix.len()..name.len() - ".csv".len()];
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
        let reader = StorageReader::open(path)?;
        let mut csv = CsvReader::from_reader(
            reader,
            self.options.decode.clone(),
            self.options.batch_size,
        );
        // A chunk file is one complete CSV; pull its single batch (or
        // accumulate batches if the chunk is large enough that the reader
        // splits internally) into one Table.
        let mut accumulated: Option<Table> = None;
        while let Some(batch) = csv.next_batch()? {
            accumulated = Some(match accumulated {
                None => batch,
                Some(existing) => existing.concat(batch).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("concat: {e}"))
                })?,
            });
        }
        Ok(accumulated.unwrap_or_default())
    }

    /// CSV is row-oriented on disk; the decoder must read every row's
    /// bytes to find delimiters regardless of which columns are kept.
    /// Projection here is applied via minarrow's `ColumnSelection`
    /// after the chunk has been decoded - it saves Table-level storage
    /// for non-selected columns but does not save disk I/O. For
    /// column-oriented disk reads use chunked Parquet or chunked
    /// Arrow IPC where the format supports per-column access.
    fn read_chunk_cols(&self, path: &Path, columns: &[&str]) -> io::Result<Table> {
        let table = self.read_chunk(path)?;
        Ok(table.c(columns).to_table())
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

impl Iterator for ChunkedCsvReader {
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
    use crate::models::encoders::csv::CsvEncodeOptions;
    use crate::models::writers::chunked::csv::ChunkedCsvWriter;
    use crate::traits::chunked_table_writer::ChunkedTableWriter;
    use minarrow::{Table, fa_i32};

    #[test]
    fn read_all_unifies_in_write_order() {
        let dir = std::env::temp_dir().join("lightstream_chunked_csv_test_reader");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedCsvWriter::new(&dir, "part", CsvEncodeOptions::default()).unwrap();
        w.write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 1, 2, 3)])))
            .unwrap();
        w.write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 4, 5)])))
            .unwrap();
        w.write_chunk(&Table::new(
            "b",
            Some(vec![fa_i32!("n", 6, 7, 8, 9)]),
        ))
        .unwrap();

        let reader = ChunkedCsvReader::open(
            &dir,
            "part",
            ChunkedCsvReadOptions {
                decode: CsvDecodeOptions::default(),
                batch_size: 1024,
            },
        )
        .unwrap();
        let combined = reader.load_batched().unwrap();
        assert_eq!(combined.n_rows, 9);
        assert_eq!(combined.batches.len(), 3);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn par_load_batched_returns_batches_in_write_order() {
        let dir = std::env::temp_dir().join("lightstream_chunked_csv_par_reader");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedCsvWriter::new(&dir, "part", CsvEncodeOptions::default()).unwrap();
        for i in 0..12i32 {
            w.write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!["n", i, i + 100]]),
            ))
            .unwrap();
        }

        let st = ChunkedCsvReader::par_load_batched(
            &dir,
            "part",
            ChunkedCsvReadOptions {
                decode: CsvDecodeOptions::default(),
                batch_size: 1024,
            },
            // Force a non-default thread count so we exercise the override path.
            Some(2),
        )
        .unwrap();
        assert_eq!(st.batches.len(), 12);
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
    fn iterator_yields_each_chunk_in_order() {
        let dir = std::env::temp_dir().join("lightstream_chunked_csv_test_iter");
        let _ = fs::remove_dir_all(&dir);

        let mut w = ChunkedCsvWriter::new(&dir, "part", CsvEncodeOptions::default()).unwrap();
        w.write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 10)])))
            .unwrap();
        w.write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 20, 21)])))
            .unwrap();

        let reader = ChunkedCsvReader::open(
            &dir,
            "part",
            ChunkedCsvReadOptions {
                decode: CsvDecodeOptions::default(),
                batch_size: 1024,
            },
        )
        .unwrap();
        let lengths: Vec<usize> = reader.map(|t| t.unwrap().n_rows()).collect();
        assert_eq!(lengths, vec![1, 2]);

        fs::remove_dir_all(&dir).ok();
    }
}
