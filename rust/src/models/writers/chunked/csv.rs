// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked CSV writer
//!
//! Writes one CSV file per batch into a directory, naming files with a
//! zero-padded 10-digit incrementing index (`<base>-NNNNNNNNNN.csv`).
//! Each file is a complete, independently readable CSV including its
//! own header. The width covers any realistic streaming sink lifetime
//! (~10^10 chunks) without relying on lexicographic sort matching
//! numeric order.
//!
//! Intended as the streaming-sink companion to the single-file
//! `CsvWriter`: a streaming pipeline can write one chunk per emitted
//! batch without holding open file state across batches. The matching
//! `ChunkedCsvReader` globs the directory and presents the chunks back
//! as an ordered iterator of `Table`s.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;

use minarrow::Table;

use crate::models::encoders::csv::CsvEncodeOptions;
use crate::models::writers::object_storage::{StorageWriter, create_dir_all};
use crate::models::writers::csv::CsvWriter;
use crate::traits::chunked_table_writer::ChunkedTableWriter;

/// Streaming CSV sink that writes one file per batch.
///
/// The directory is created on construction. Each `write_chunk` call
/// reserves the next index off the shared counter and writes to
/// `<dir>/<base>-NNNNNNNNNN.csv`. The counter is fully managed
/// internally; callers do not pass an index.
#[derive(Debug)]
pub struct ChunkedCsvWriter {
    dir: PathBuf,
    base: String,
    options: CsvEncodeOptions,
    next_index: AtomicU64,
}

impl ChunkedCsvWriter {
    /// Create a new chunked writer rooted at `dir`. Each chunk filename
    /// will be `<base>-NNNNNNNNNN.csv` with a 10-digit zero-padded
    /// index. The directory is created if it does not exist.
    pub fn new<P: AsRef<Path>>(dir: P, base: &str, options: CsvEncodeOptions) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        create_dir_all(&dir)?;
        Ok(Self {
            dir,
            base: base.to_string(),
            options,
            next_index: AtomicU64::new(0),
        })
    }
}

impl ChunkedTableWriter for ChunkedCsvWriter {
    type Error = io::Error;

    /// Chunk file extension - each emitted file is a complete CSV with header.
    fn extension() -> &'static str {
        "csv"
    }

    /// Directory the chunks are being written into.
    fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename stem used for chunk files (`<base>-NNNNNNNNNN.csv`).
    fn base(&self) -> &str {
        &self.base
    }

    /// Shared chunk-index counter; both `write_chunk` and `par_write_all`
    /// reserve indices off this atom.
    fn counter(&self) -> &AtomicU64 {
        &self.next_index
    }

    /// Encode `table` as a complete CSV file at `path` using the writer's
    /// configured encode options. Shared per-format encode path used by
    /// both `write_chunk` and `par_write_all`.
    fn write_chunk_at(&self, path: &Path, table: &Table) -> io::Result<()> {
        let file = StorageWriter::create(path)?;
        let mut writer = CsvWriter::with_options(file, self.options.clone());
        writer.write_table(table)?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::models::readers::chunked::csv::{ChunkedCsvReadOptions, ChunkedCsvReader};
    use crate::traits::chunked_table_reader::ChunkedTableReader;
    use minarrow::{Table, fa_i32};

    #[test]
    fn writes_indexed_chunk_files() {
        let dir = std::env::temp_dir().join("lightstream_chunked_csv_test_writer");
        let _ = fs::remove_dir_all(&dir);
        let mut w = ChunkedCsvWriter::new(&dir, "part", CsvEncodeOptions::default()).unwrap();
        let p0 = w
            .write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 0, 1, 2)])))
            .unwrap();
        let p1 = w
            .write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!("n", 10, 11, 12)]),
            ))
            .unwrap();
        let p2 = w
            .write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!("n", 20, 21, 22)]),
            ))
            .unwrap();

        assert_eq!(p0.file_name().unwrap(), "part-0000000000.csv");
        assert_eq!(p1.file_name().unwrap(), "part-0000000001.csv");
        assert_eq!(p2.file_name().unwrap(), "part-0000000002.csv");
        assert_eq!(w.batches_written(), 3);

        let body = fs::read_to_string(&p1).unwrap();
        assert!(body.starts_with("n\n"));
        assert!(body.contains("10\n11\n12"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn par_write_all_emits_files_in_order_and_advances_counter() {
        let dir = std::env::temp_dir().join("lightstream_chunked_csv_par_writer");
        let _ = fs::remove_dir_all(&dir);

        let w = ChunkedCsvWriter::new(&dir, "part", CsvEncodeOptions::default()).unwrap();
        let tables: Vec<Table> = (0..10i32)
            .map(|i| Table::new("b", Some(vec![fa_i32!("n", i, i + 100)])))
            .collect();
        let refs: Vec<&Table> = tables.iter().collect();

        let paths = w.par_write_all(&refs, None).unwrap();
        assert_eq!(paths.len(), 10);
        for (i, p) in paths.iter().enumerate() {
            assert_eq!(
                p.file_name().unwrap(),
                std::ffi::OsString::from(format!("part-{i:010}.csv"))
            );
        }
        assert_eq!(w.batches_written(), 10);

        let st =
            ChunkedCsvReader::par_load_batched(&dir, "part", ChunkedCsvReadOptions::default(), None)
                .unwrap();
        assert_eq!(st.batches.len(), 10);

        fs::remove_dir_all(&dir).ok();
    }
}
