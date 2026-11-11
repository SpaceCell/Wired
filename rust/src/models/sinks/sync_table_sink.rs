// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Arrow IPC table sink
//!
//! Wraps any `std::io::Write` and writes [`minarrow::Table`] values as Arrow IPC frames.
//! Blocking counterpart of [`GTableSink`], for writers that run on a dedicated
//! thread without an async runtime.
//!
//! ## Overview:
//! - Supports Stream or File protocol
//! - Handles schema emission, optional compression, dictionary batches, record batches, and end-of-stream/footer generation.
//! - Encodes through [`ArrowIpcCodec`] and [`TableStreamWriter`], as [`GTableSink`] does, and therefore writes the same bytes for the same tables.
//! - Under the Stream protocol, every table is encoded into one pooled buffer that is reused across writes.
//! - Each write blocks the calling thread until the destination has accepted the frame.
//!
//! [`GTableSink`]: crate::models::sinks::table_sink::GTableSink

use std::io;

use minarrow::{Field, TableV, Vec64};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::codecs::ipc::ArrowIpcCodec;
use crate::models::writers::ipc::table_stream::TableStreamWriter;
use crate::traits::stream_buffer::StreamBuffer;

/// Blocking Arrow Table sink over any `std::io::Write` destination.
///
/// Handles the Arrow IPC protocol (framing, schema, dictionaries, record
/// batches, footers). The sync writers wrap this sink, as the async writers
/// wrap [`GTableSink`].
///
/// The `B` parameter controls frame alignment. The default `Vec64<u8>`
/// writes 64-byte SIMD aligned frames.
///
/// [`GTableSink`]: crate::models::sinks::table_sink::GTableSink
pub struct SyncTableSink<W, B = Vec64<u8>>
where
    W: io::Write,
    B: StreamBuffer + Unpin + 'static,
{
    schema: Vec<Field>,
    codec: ArrowIpcCodec<B>,
    destination: W,
    protocol: IPCMessageProtocol,
    finished: bool,
    /// Pooled encode buffer reused across write_table calls.
    encode_buf: B,
    /// Frame-by-frame writer for File protocol with footer tracking
    file_writer: Option<TableStreamWriter<B>>,
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
        let file_writer = if protocol == IPCMessageProtocol::File {
            Some(TableStreamWriter::new(schema.clone(), protocol, compression))
        } else {
            None
        };
        Self {
            codec: ArrowIpcCodec::new(schema.clone(), protocol, compression, None),
            schema,
            destination,
            protocol,
            finished: false,
            encode_buf: B::with_capacity(0),
            file_writer,
        }
    }

    /// Get the schema used for this sink.
    pub fn schema(&self) -> &[Field] {
        &self.schema
    }

    /// Return the protocol in use (Stream or File).
    pub fn protocol(&self) -> IPCMessageProtocol {
        self.protocol
    }

    /// Register a dictionary with the given id and values.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.codec.register_dictionary(dict_id, values.clone());
        if let Some(writer) = &mut self.file_writer {
            writer.register_dictionary(dict_id, values);
        }
    }

    /// Expose a mutable reference to the inner sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.destination
    }

    /// Consume the sink and return the destination. The stream is left as
    /// written, and [`Self::finish`] is called first when the end-of-stream
    /// marker or footer is required.
    pub fn into_inner(self) -> W {
        self.destination
    }

    /// Encode a table or table view and write its frames to the destination.
    ///
    /// Returns an error once [`Self::finish`] has been called.
    pub fn write_table(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        if self.finished {
            return Err(io::Error::other("sink already finished"));
        }
        let view: TableV = table.into();
        if let Some(writer) = &mut self.file_writer {
            // The File protocol routes through the frame-by-frame writer that
            // tracks footer blocks.
            writer.write(view)?;
            return self.write_file_frames();
        }
        // The Stream protocol encodes into a pooled buffer reused across
        // writes.
        self.clear_encode_buf();
        self.codec
            .encode_stream_batch(&view, &mut self.encode_buf, 0, None)?;
        self.destination.write_all(self.encode_buf.as_ref())
    }

    /// Flush the destination.
    pub fn flush(&mut self) -> io::Result<()> {
        self.destination.flush()
    }

    /// Write the end-of-stream marker or the File footer, then flush the
    /// destination. Later calls flush only.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.finished {
            if let Some(writer) = &mut self.file_writer {
                // File protocol: writer handles footer + EOS
                writer.finish()?;
                self.write_file_frames()?;
            } else {
                // Stream protocol: write EOS into the pooled buffer
                self.clear_encode_buf();
                self.codec.finish(&mut self.encode_buf)?;
                self.destination.write_all(self.encode_buf.as_ref())?;
            }
            self.finished = true;
        }
        self.destination.flush()
    }

    /// Write every frame queued by the File protocol writer to the
    /// destination.
    fn write_file_frames(&mut self) -> io::Result<()> {
        if let Some(writer) = &mut self.file_writer {
            while let Some(frame) = writer.next_frame() {
                let buf = frame?;
                self.destination.write_all(buf.as_ref())?;
            }
        }
        Ok(())
    }

    /// Empty the pooled encode buffer, keeping its allocation.
    fn clear_encode_buf(&mut self) {
        let len = self.encode_buf.len();
        if len > 0 {
            self.encode_buf.drain(0..len);
        }
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
    /// sink, including an empty stream.
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

    /// Writes after the first reuse the allocation of the pooled encode
    /// buffer.
    #[test]
    fn writes_reuse_the_pooled_encode_buffer() {
        let table = test_table();
        let mut sink: SyncTableSink<_> = SyncTableSink::new(
            Vec::new(),
            schema_of(&table),
            IPCMessageProtocol::Stream,
            None,
        );
        sink.write_table(table.clone()).unwrap();
        let pooled = sink.encode_buf.as_ref().as_ptr();
        for _ in 0..3 {
            sink.write_table(table.clone()).unwrap();
            assert_eq!(sink.encode_buf.as_ref().as_ptr(), pooled);
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
