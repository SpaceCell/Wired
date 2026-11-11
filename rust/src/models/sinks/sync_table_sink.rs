// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Arrow IPC table sink
//!
//! Wraps any `std::io::Write` and writes [`minarrow::Table`] values as Arrow IPC frames.
//! Blocking counterpart of [`GTableSink`], for writers that run on a dedicated
//! thread with no async runtime.
//!
//! ## Overview:
//! - Supports Stream or File protocol
//! - Encodes through the same state as [`GTableSink`], and therefore writes the same bytes for the same tables.
//! - Under the Stream protocol, every table is encoded into one pooled buffer that is reused across writes.
//! - Each write blocks the calling thread until the destination has accepted the frame.
//!
//! [`GTableSink`]: crate::models::sinks::table_sink::GTableSink

use std::io;

use minarrow::{Field, TableV, Vec64};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::TableSinkCore;
use crate::traits::stream_buffer::StreamBuffer;

/// Blocking Arrow Table sink over any `std::io::Write` destination.
///
/// Handles the Arrow IPC protocol (framing, schema, dictionaries, record
/// batches, footers) through the encode state shared with [`GTableSink`].
/// The sync transport writers wrap this sink, as the async transport writers
/// wrap [`TableSink64`].
///
/// The `B` parameter controls frame alignment. The default `Vec64<u8>`
/// writes 64-byte SIMD aligned frames.
///
/// [`GTableSink`]: crate::models::sinks::table_sink::GTableSink
/// [`TableSink64`]: crate::models::sinks::table_sink::TableSink64
pub struct SyncTableSink<W, B = Vec64<u8>>
where
    W: io::Write,
    B: StreamBuffer + Unpin + 'static,
{
    core: TableSinkCore<B>,
    destination: W,
}

impl<W, B> SyncTableSink<W, B>
where
    W: io::Write,
    B: StreamBuffer + Unpin + 'static,
{
    /// Create a new sync Arrow Table sink. Pass `None` for `compression` to
    /// write uncompressed batches; `Some(codec)` compresses every
    /// record-batch body.
    pub fn new(
        destination: W,
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> Self {
        Self {
            core: TableSinkCore::new(schema, protocol, compression),
            destination,
        }
    }

    /// Get the schema used for this sink.
    pub fn schema(&self) -> &[Field] {
        &self.core.schema
    }

    /// Register a dictionary with the given id and values.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.core.register_dictionary(dict_id, values);
    }

    /// Expose a mutable reference to the inner sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.destination
    }

    /// Encode a table or table view and write its frames to the destination.
    ///
    /// Returns an error once [`Self::finish`] has been called.
    pub fn write_table(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        if self.core.finished {
            return Err(io::Error::other("sink already finished"));
        }
        self.core.encode_frame(table.into(), None)?;
        self.write_frames()
    }

    /// Flush the destination.
    pub fn flush(&mut self) -> io::Result<()> {
        self.destination.flush()
    }

    /// Write the end-of-stream marker or the File footer, then flush the
    /// destination. Later calls flush only.
    pub fn finish(&mut self) -> io::Result<()> {
        self.core.finish()?;
        self.write_frames()?;
        self.destination.flush()
    }

    /// Write every encoded frame to the destination, returning each buffer
    /// to the pool once written.
    fn write_frames(&mut self) -> io::Result<()> {
        while let Some(buf) = self.core.next_frame() {
            self.destination.write_all(buf.as_ref())?;
            self.core.reclaim(buf);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures_util::SinkExt;
    use minarrow::Table;

    use super::*;
    use crate::models::sinks::table_sink::TableSink64;
    use crate::test_helpers::{int32_col, int64_col};

    fn test_table() -> Table {
        Table::new("t".to_string(), vec![int32_col(), int64_col()].into())
    }

    fn schema_of(table: &Table) -> Vec<Field> {
        table
            .cols
            .iter()
            .map(|fa| fa.field.as_ref().clone())
            .collect()
    }

    fn sync_bytes(n_tables: usize, protocol: IPCMessageProtocol) -> Vec<u8> {
        let table = test_table();
        let mut sink: SyncTableSink<_> =
            SyncTableSink::new(Vec::new(), schema_of(&table), protocol, None);
        for _ in 0..n_tables {
            sink.write_table(table.clone()).unwrap();
        }
        sink.finish().unwrap();
        sink.destination
    }

    async fn async_bytes(n_tables: usize, protocol: IPCMessageProtocol) -> Vec<u8> {
        let table = test_table();
        let mut sink = TableSink64::new(Vec::new(), schema_of(&table), protocol, None).unwrap();
        for _ in 0..n_tables {
            SinkExt::send(&mut sink, table.clone().into()).await.unwrap();
        }
        SinkExt::close(&mut sink).await.unwrap();
        std::mem::take(&mut sink.destination)
    }

    /// The sync sink writes the same Stream protocol bytes as the async
    /// sink, including the stream with no tables.
    #[tokio::test]
    async fn stream_bytes_match_the_async_sink() {
        for n_tables in [0, 1, 3] {
            let expected = async_bytes(n_tables, IPCMessageProtocol::Stream).await;
            assert!(!expected.is_empty());
            assert_eq!(sync_bytes(n_tables, IPCMessageProtocol::Stream), expected);
        }
    }

    /// The sync sink writes the same File protocol bytes as the async sink.
    #[tokio::test]
    async fn file_bytes_match_the_async_sink() {
        for n_tables in [1, 3] {
            let expected = async_bytes(n_tables, IPCMessageProtocol::File).await;
            assert!(!expected.is_empty());
            assert_eq!(sync_bytes(n_tables, IPCMessageProtocol::File), expected);
        }
    }

    /// Stream protocol writes after the first reuse the allocation of the
    /// pooled encode buffer.
    #[test]
    fn stream_writes_reuse_the_pooled_encode_buffer() {
        let table = test_table();
        let mut sink: SyncTableSink<_> = SyncTableSink::new(
            Vec::new(),
            schema_of(&table),
            IPCMessageProtocol::Stream,
            None,
        );
        sink.write_table(table.clone()).unwrap();
        let pooled = sink.core.encode_buf.as_ref().as_ptr();
        for _ in 0..3 {
            sink.write_table(table.clone()).unwrap();
            assert_eq!(sink.core.encode_buf.as_ref().as_ptr(), pooled);
        }
    }

    #[test]
    fn write_after_finish_is_an_error() {
        let table = test_table();
        let mut sink: SyncTableSink<_> = SyncTableSink::new(
            Vec::new(),
            schema_of(&table),
            IPCMessageProtocol::Stream,
            None,
        );
        sink.write_table(table.clone()).unwrap();
        sink.finish().unwrap();
        let written = sink.destination.len();
        assert!(sink.write_table(table).is_err());
        assert_eq!(sink.destination.len(), written);
    }
}
