// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Chunked Arrow IPC writer
//!
//! Writes one Arrow IPC file per batch into a directory, naming files with
//! a zero-padded 10-digit incrementing index (`<base>-NNNNNNNNNN.arrow`).
//! Each file is a complete, independently readable Arrow IPC file with its
//! own footer. The width covers any realistic streaming sink lifetime
//! (~10^10 chunks) without relying on lexicographic sort matching numeric
//! order.
//!
//! Companion to the single-file `write_table_to_file`: a streaming pipeline
//! can write one chunk per emitted batch without holding open file state
//! across batches. The matching `ChunkedArrowReader` globs the directory
//! and presents the chunks back as an ordered iterator of `Table`s.
//!
//! ## Sync, no runtime
//!
//! Per-chunk writes go through `SyncTableWriter`, which drives the existing
//! sync IPC encoder into a `std::io::Write` sink. No tokio runtime is
//! created for chunk writing; callers can drive this directly from any
//! sync context (e.g. the engine's per-Block invocation path).

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;

use minarrow::{Field, Table};

use crate::enums::IPCMessageProtocol;
use crate::models::writers::chunked::partition::PartitionType;
use crate::models::writers::object_storage::{StorageWriter, create_dir_all};
use crate::models::writers::ipc::sync_table::SyncTableWriter;
use crate::traits::chunked_table_writer::ChunkedTableWriter;

/// Streaming Arrow IPC sink that writes one file per batch.
///
/// The directory is created on construction. Each `write_chunk` call
/// reserves the next index off the shared counter and writes to
/// `<dir>/<base>-NNNNNNNNNN.arrow`.
#[derive(Debug)]
pub struct ChunkedArrowWriter {
    dir: PathBuf,
    base: String,
    next_index: AtomicU64,
    partition: Option<PartitionType>,
}

impl ChunkedArrowWriter {
    /// Create a new chunked writer rooted at `dir`. Each chunk filename
    /// will be `<base>-NNNNNNNNNN.arrow` with a 10-digit zero-padded
    /// index. The directory is created if it does not exist. Arrow IPC
    /// has no per-chunk options at this layer (each file is a complete
    /// Arrow IPC File so external readers like pyarrow and arrow-rs can
    /// open it).
    ///
    /// `partition` places each chunk in a partition directory beneath `dir`,
    /// for example `<dir>/date=2026-10-14/hour=13/<base>-NNNNNNNNNN.arrow`.
    /// `None` writes every chunk into `dir`.
    pub fn new<P: AsRef<Path>>(
        dir: P,
        base: &str,
        partition: Option<PartitionType>,
    ) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        create_dir_all(&dir)?;
        Ok(Self {
            dir,
            base: base.to_string(),
            next_index: AtomicU64::new(0),
            partition,
        })
    }
}

impl ChunkedTableWriter for ChunkedArrowWriter {
    type Error = io::Error;

    /// Chunk file extension - each emitted file is a complete Arrow IPC File.
    fn extension() -> &'static str {
        "arrow"
    }

    /// Directory the chunks are being written into.
    fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename stem used for chunk files (`<base>-NNNNNNNNNN.arrow`).
    fn base(&self) -> &str {
        &self.base
    }

    /// Shared chunk-index counter; both `write_chunk` and `par_write_all`
    /// reserve indices off this atom.
    fn counter(&self) -> &AtomicU64 {
        &self.next_index
    }

    /// Partitioning applied to chunk paths, if configured.
    fn partition(&self) -> Option<&PartitionType> {
        self.partition.as_ref()
    }

    /// Encode `table` as a complete Arrow IPC File at `path`. Shared
    /// per-format encode path used by both `write_chunk` and
    /// `par_write_all`.
    fn write_chunk_at(&self, path: &Path, table: &Table) -> io::Result<()> {
        let schema: Vec<Field> = table.cols.iter().map(|col| (*col.field).clone()).collect();
        let file = StorageWriter::create(path)?;
        let mut writer: SyncTableWriter<_> =
            SyncTableWriter::new(file, schema, IPCMessageProtocol::File, None);
        writer.write_table(table.clone())?;
        writer.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::models::readers::chunked::arrow::ChunkedArrowReader;
    use crate::traits::chunked_table_reader::ChunkedTableReader;
    use minarrow::{Table, fa_i32};

    #[test]
    fn writes_indexed_chunk_files_and_round_trips() {
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_test_writer");
        let _ = fs::remove_dir_all(&dir);
        let mut w = ChunkedArrowWriter::new(&dir, "part", None).unwrap();
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

        assert_eq!(p0.file_name().unwrap(), "part-0000000000.arrow");
        assert_eq!(p1.file_name().unwrap(), "part-0000000001.arrow");
        assert_eq!(p2.file_name().unwrap(), "part-0000000002.arrow");
        assert_eq!(w.batches_written(), 3);

        let reader = ChunkedArrowReader::open(&dir, "part", ()).unwrap();
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
        let dir = std::env::temp_dir().join("lightstream_chunked_arrow_par_writer");
        let _ = fs::remove_dir_all(&dir);

        let w = ChunkedArrowWriter::new(&dir, "part", None).unwrap();
        let tables: Vec<Table> = (0..8i32)
            .map(|i| Table::new("b", Some(vec![fa_i32!("n", i, i + 100)])))
            .collect();
        let refs: Vec<&Table> = tables.iter().collect();

        let paths = w.par_write_all(&refs, None).unwrap();
        assert_eq!(paths.len(), 8);
        for (i, p) in paths.iter().enumerate() {
            assert_eq!(
                p.file_name().unwrap(),
                std::ffi::OsString::from(format!("part-{i:010}.arrow"))
            );
        }
        assert_eq!(w.batches_written(), 8);

        let st = ChunkedArrowReader::par_load_batched(&dir, "part", (), None).unwrap();
        assert_eq!(st.batches.len(), 8);

        fs::remove_dir_all(&dir).ok();
    }
}
