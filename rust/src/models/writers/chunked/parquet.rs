// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked Parquet writer
//!
//! Writes one Parquet file per batch into a directory, naming files with a
//! zero-padded 10-digit incrementing index (`<base>-NNNNNNNNNN.parquet`).
//! Each file is a complete, independently readable Parquet file with its
//! own schema and footer. The width covers any realistic streaming sink
//! lifetime (~10^10 chunks) without relying on lexicographic sort matching
//! numeric order.
//!
//! Companion to the single-file `write_parquet_table`: a streaming pipeline
//! can write one chunk per emitted batch without holding open file state
//! across batches. The matching `ChunkedParquetReader` globs the directory
//! and presents the chunks back as an ordered iterator of `Table`s.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;

use minarrow::Table;

use crate::compression::Compression;
use crate::error::IoError;
use crate::models::writers::object_storage::{StorageWriter, create_dir_all};
use crate::models::writers::parquet::write_parquet_table;
use crate::traits::chunked_table_writer::ChunkedTableWriter;

/// Streaming Parquet sink that writes one file per batch.
#[derive(Debug)]
pub struct ChunkedParquetWriter {
    dir: PathBuf,
    base: String,
    compression: Option<Compression>,
    next_index: AtomicU64,
}

impl ChunkedParquetWriter {
    /// Create a new chunked writer rooted at `dir`. Each chunk filename
    /// will be `<base>-NNNNNNNNNN.parquet` with a 10-digit zero-padded
    /// index. The directory is created if it does not exist.
    pub fn new<P: AsRef<Path>>(
        dir: P,
        base: &str,
        compression: Option<Compression>,
    ) -> Result<Self, IoError> {
        let dir = dir.as_ref().to_path_buf();
        create_dir_all(&dir)?;
        Ok(Self {
            dir,
            base: base.to_string(),
            compression,
            next_index: AtomicU64::new(0),
        })
    }
}

impl ChunkedTableWriter for ChunkedParquetWriter {
    type Error = IoError;

    /// Chunk file extension - each emitted file is a complete Parquet file.
    fn extension() -> &'static str {
        "parquet"
    }

    /// Directory the chunks are being written into.
    fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename stem used for chunk files (`<base>-NNNNNNNNNN.parquet`).
    fn base(&self) -> &str {
        &self.base
    }

    /// Shared chunk-index counter; both `write_chunk` and `par_write_all`
    /// reserve indices off this atom.
    fn counter(&self) -> &AtomicU64 {
        &self.next_index
    }

    /// Encode `table` as a complete Parquet file at `path` using the
    /// writer's configured compression. Shared per-format encode path
    /// used by both `write_chunk` and `par_write_all`.
    fn write_chunk_at(&self, path: &Path, table: &Table) -> Result<(), IoError> {
        let mut file = StorageWriter::create(path)?;
        write_parquet_table(table, &mut file, self.compression)?;
        // Flushing completes the object when `path` is an object URL.
        file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::models::readers::chunked::parquet::ChunkedParquetReader;
    use crate::traits::chunked_table_reader::ChunkedTableReader;
    use minarrow::{Table, fa_i32};

    #[test]
    fn writes_indexed_chunk_files_and_round_trips() {
        let dir = std::env::temp_dir().join("lightstream_chunked_parquet_test_writer");
        let _ = fs::remove_dir_all(&dir);
        let mut w = ChunkedParquetWriter::new(&dir, "part", None).unwrap();
        let p0 = w
            .write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 1, 2, 3)])))
            .unwrap();
        let p1 = w
            .write_chunk(&Table::new("b", Some(vec![fa_i32!("n", 4, 5)])))
            .unwrap();
        let p2 = w
            .write_chunk(&Table::new(
                "b",
                Some(vec![fa_i32!("n", 6, 7, 8, 9)]),
            ))
            .unwrap();

        assert_eq!(p0.file_name().unwrap(), "part-0000000000.parquet");
        assert_eq!(p1.file_name().unwrap(), "part-0000000001.parquet");
        assert_eq!(p2.file_name().unwrap(), "part-0000000002.parquet");
        assert_eq!(w.batches_written(), 3);

        let reader = ChunkedParquetReader::open(&dir, "part", ()).unwrap();
        let st = reader.load_batched().unwrap();
        assert_eq!(st.batches.len(), 3);
        assert_eq!(st.n_rows, 9);
        assert_eq!(st.batches[0].n_rows, 3);
        assert_eq!(st.batches[1].n_rows, 2);
        assert_eq!(st.batches[2].n_rows, 4);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn par_write_all_emits_files_in_order_and_advances_counter() {
        let dir = std::env::temp_dir().join("lightstream_chunked_parquet_par_writer");
        let _ = fs::remove_dir_all(&dir);

        let w = ChunkedParquetWriter::new(&dir, "part", None).unwrap();
        let tables: Vec<Table> = (0..6i32)
            .map(|i| Table::new("b", Some(vec![fa_i32!("n", i, i + 100)])))
            .collect();
        let refs: Vec<&Table> = tables.iter().collect();

        let paths = w.par_write_all(&refs, None).unwrap();
        assert_eq!(paths.len(), 6);
        for (i, p) in paths.iter().enumerate() {
            assert_eq!(
                p.file_name().unwrap(),
                std::ffi::OsString::from(format!("part-{i:010}.parquet"))
            );
        }
        assert_eq!(w.batches_written(), 6);

        let st = ChunkedParquetReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert_eq!(st.batches.len(), 6);

        fs::remove_dir_all(&dir).ok();
    }
}
