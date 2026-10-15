// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Arrow IPC Table Writer
//!
//! Provides a streaming, synchronous encoder for writing [`minarrow::Table`] values
//! into Arrow IPC frames - File or Stream protocol
//!
//! ## Overview
//! - Wraps the generic `GTableStreamEncoder` for frame emission
//! - For pipes, custom network protocols, or synchronous contexts
//! - Emits schema, dictionaries, record batches, and end-of-stream/footer
//! - Frames can be pulled incrementally (`next_frame`) or drained all at once
//!
//! ## Async Helpers
//! - [`write_tables_to_stream`] - write a sequence of tables to an async sink.
//! - [`write_table_to_stream`] - write a single table to an async sink.
//!
//! ## Usage
//! ```ignore
//! let mut writer: TableStreamWriter = TableStreamWriter::new(schema, IPCMessageProtocol::Stream, None);
//! writer.register_dictionary(0, vec!["A".into(), "B".into()]);
//! writer.write(table.clone())?;
//! writer.finish()?;
//! while let Some(frame) = writer.next_frame() {
//!     let buf = frame?;
//!     sink.write_all(buf.as_ref()).await?;
//! }
//! ```

use std::collections::VecDeque;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use minarrow::{Field, Table, TableV, Vec64};
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;

use crate::arrow::message::org::apache::arrow::flatbuf as fbm;
use crate::compression::Compression;
use crate::enums::{IPCMessageProtocol, WriterState};
use crate::models::encoders::ipc::schema::{FooterBlockMeta, build_flatbuf_footer};
use crate::models::encoders::ipc::table_stream::TableStreamEncoder;
use crate::models::encoders::ipc::{IPCFrame, IPCFrameEncoder};
use crate::traits::frame_encoder::FrameEncoder;
use crate::traits::stream_buffer::StreamBuffer;
use crate::utils::dict_values;

/// # Synchronous Arrow IPC Table Writer
///
/// Encodes [`minarrow::Table`] values into Arrow File or Stream protocol IPC frames.
/// Great for pipes, custom transports, or synchronous contexts.
///
/// ## Example
/// ```ignore
/// let mut writer: TableStreamWriter = TableStreamWriter::new(schema, IPCMessageProtocol::Stream, None);
/// writer.register_dictionary(0, vec!["A".into(), "B".into()]);
/// writer.write(table.clone())?;
/// writer.finish()?;
/// while let Some(frame) = writer.next_frame() {
///     let buf = frame?;
///     sink.write_all(buf.as_ref()).await?;
/// }
/// ```
pub struct TableStreamWriter<B = Vec64<u8>>
where
    B: StreamBuffer + Unpin + 'static,
{
    /// The encoder produces (meta, body) pairs
    encoder: TableStreamEncoder<B>,
    /// Queue of framed IPC output buffers
    out_frames: VecDeque<B>,
    /// True when finish() has been called and all frames emitted
    finished: bool,
    /// Running byte offset for IPC frame alignment
    global_offset: usize,
    // ----- File format only -----
    /// Block metadata for record batches in the footer
    blocks_record_batches: Vec<FooterBlockMeta>,
    /// Block metadata for dictionary batches in the footer
    blocks_dictionaries: Vec<FooterBlockMeta>,
    /// Frame offsets for footer
    frame_offsets: Vec<u64>,
    /// Running total bytes for file offset tracking
    total_len_offset: u64,
}

impl<B> TableStreamWriter<B>
where
    B: StreamBuffer + Unpin + 'static,
{
    /// Create a new streaming Arrow Table writer. Pass `None` for
    /// `compression` to write uncompressed batches; `Some(codec)`
    /// compresses every record-batch body.
    pub fn new(
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> Self {
        Self {
            encoder: TableStreamEncoder::new(schema, protocol, compression),
            out_frames: VecDeque::new(),
            finished: false,
            global_offset: 0,
            blocks_record_batches: Vec::new(),
            blocks_dictionaries: Vec::new(),
            frame_offsets: Vec::new(),
            total_len_offset: 0,
        }
    }

    /// Register a dictionary for categorical columns.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.encoder.register_dictionary(dict_id, values);
    }

    /// Write a single table as a record batch frame, emitting the schema and
    /// any required dictionaries as needed.
    ///
    /// The parameter accepts an owned `Table` as well as a `TableV` window
    /// over one, so a caller holding a whole table passes it as it stands.
    /// Either conversion shares the source buffers, so no column data moves.
    pub fn write(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        let view = table.into();
        if self.encoder.state == WriterState::Closed {
            return Err(io::Error::other(
                "writer already finished",
            ));
        }
        if view.cols.len() != self.encoder.schema.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "table column count mismatch with writer schema",
            ));
        }

        // Emit schema on first write
        if self.encoder.state == WriterState::Fresh {
            let meta = self.encoder.encode_schema()?;
            let body = B::with_capacity(0);
            self.emit_frame(meta, body, fbm::MessageHeader::Schema);
        }

        // Emit any pending dictionaries
        let dict_ids = self.encoder.pending_dict_ids();
        for dict_id in dict_ids {
            if let Some((meta, body_vec)) = self.encoder.encode_dictionary(dict_id)? {
                let mut body = B::with_capacity(body_vec.len());
                body.extend_from_slice(&body_vec);
                self.emit_frame(meta, body, fbm::MessageHeader::DictionaryBatch);
            }
        }

        // Encode and emit the record batch
        let (meta, body) = self.encoder.encode_record_batch(&view)?;
        self.emit_frame(meta, body, fbm::MessageHeader::RecordBatch);
        Ok(())
    }

    /// Emit Arrow footer/EOS marker, finalising the stream.
    /// This must be called before draining all frames.
    pub fn finish(&mut self) -> io::Result<()> {
        if self.encoder.state == WriterState::Closed {
            return Ok(());
        }
        match self.encoder.protocol {
            IPCMessageProtocol::File => {
                let is_first = self.frame_offsets.is_empty();
                let footer_bytes = build_flatbuf_footer(
                    &mut self.encoder.fbb,
                    &self.encoder.schema,
                    &self.blocks_dictionaries,
                    &self.blocks_record_batches,
                )?;
                let frame = IPCFrame {
                    meta: &[],
                    body: &[],
                    protocol: IPCMessageProtocol::File,
                    is_first,
                    is_last: true,
                    footer_bytes: Some(&footer_bytes),
                };
                let (footer_frame, _) =
                    IPCFrameEncoder::encode::<B>(&mut self.global_offset, &frame)?;
                self.out_frames.push_back(footer_frame);
            }
            IPCMessageProtocol::Stream => {
                // Only emit EOS marker if we actually wrote something
                if self.encoder.state != WriterState::Fresh {
                    let frame = IPCFrame {
                        meta: &[],
                        body: &[],
                        protocol: IPCMessageProtocol::Stream,
                        is_first: false,
                        is_last: true,
                        footer_bytes: None,
                    };
                    let (eos_frame, _) =
                        IPCFrameEncoder::encode::<B>(&mut self.global_offset, &frame)?;
                    self.out_frames.push_back(eos_frame);
                }
            }
        }
        self.encoder.state = WriterState::Closed;
        self.finished = true;
        Ok(())
    }

    /// Poll the next encoded Arrow IPC frame as a buffer chunk.
    /// Returns None when all frames are emitted and finished.
    pub fn next_frame(&mut self) -> Option<io::Result<B>> {
        self.out_frames.pop_front().map(Ok)
    }

    /// Drain all remaining encoded frames to a Vec.
    pub fn drain_all_frames(&mut self) -> Vec<B> {
        self.out_frames.drain(..).collect()
    }

    /// Return true if the stream is finished and no more frames remain.
    pub fn is_finished(&self) -> bool {
        self.finished && self.out_frames.is_empty()
    }

    /// Access current writer schema.
    pub fn schema(&self) -> &[Field] {
        &self.encoder.schema
    }

    /// Frame a (meta, body) pair as an IPC frame and queue it.
    /// Tracks file protocol block metadata for footer generation.
    fn emit_frame(&mut self, meta: Vec<u8>, body: B, header_type: fbm::MessageHeader) {
        let is_first =
            self.encoder.protocol == IPCMessageProtocol::File && self.frame_offsets.is_empty();

        let frame = IPCFrame {
            meta: &meta,
            body: body.as_ref(),
            protocol: self.encoder.protocol,
            is_first,
            is_last: false,
            footer_bytes: None,
        };

        let (encoded, ipc_frame_metadata) =
            IPCFrameEncoder::encode::<B>(&mut self.global_offset, &frame)
                .expect("IPC frame encoding failed");

        if self.encoder.protocol == IPCMessageProtocol::File {
            let block = FooterBlockMeta {
                offset: self.total_len_offset,
                metadata_len: ipc_frame_metadata.metadata_total_len() as u32
                    + ipc_frame_metadata.header_len as u32,
                body_len: ipc_frame_metadata.body_total_len() as u64,
            };
            match header_type {
                fbm::MessageHeader::DictionaryBatch => self.blocks_dictionaries.push(block),
                fbm::MessageHeader::RecordBatch => self.blocks_record_batches.push(block),
                _ => {}
            }
            self.frame_offsets.push(self.total_len_offset);
            self.total_len_offset += ipc_frame_metadata.frame_len() as u64;
        }
        self.out_frames.push_back(encoded);
    }
}

impl<B> Stream for TableStreamWriter<B>
where
    B: StreamBuffer + Unpin + 'static,
{
    type Item = io::Result<B>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(frame) = this.out_frames.pop_front() {
            Poll::Ready(Some(Ok(frame)))
        } else if this.finished {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

/// Write a sequence of `Table`s to an arbitrary async stream (e.g., socket, pipe, etc.).
///
/// * `stream`      - the destination async byte sink
/// * `tables`      - the batches to write (each a `Table`, i.e., Arrow `RecordBatch`)
/// * `schema`      - the common schema (must match each `Table`)
/// * `protocol`    - IPC protocol to use (File or Stream)
pub async fn write_tables_to_stream<W, B>(
    mut stream: W,
    tables: &[Table],
    schema: Vec<Field>,
    protocol: IPCMessageProtocol,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Send,
    B: StreamBuffer + Unpin,
{
    let mut writer = TableStreamWriter::<B>::new(schema, protocol, None);

    for table in tables {
        for (col_idx, col) in table.cols.iter().enumerate() {
            if let Some(values) = dict_values(&col.array) {
                writer.register_dictionary(col_idx as i64, values);
            }
        }
        writer.write(table.clone())?;
    }
    writer.finish()?;

    while let Some(frame) = writer.next_frame() {
        let buf = frame?;
        stream.write_all(buf.as_ref()).await?;
    }
    stream.flush().await?;
    Ok(())
}

/// Write a single `Table` to an arbitrary async stream (i.e., socket, pipe, etc.).
///
/// * `stream`      - the destination async byte sink
/// * `table`       - the batch to write (a `Table`)
/// * `schema`      - the schema (must match the table)
/// * `protocol`    - IPC protocol to use (File or Stream)
pub async fn write_table_to_stream<W, B>(
    mut stream: W,
    table: &Table,
    schema: Vec<Field>,
    protocol: IPCMessageProtocol,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Send,
    B: StreamBuffer + Unpin,
{
    let mut writer = TableStreamWriter::<B>::new(schema, protocol, None);

    // Register dictionaries (if any categorical columns present)
    for (col_idx, col) in table.cols.iter().enumerate() {
        if let Some(values) = dict_values(&col.array) {
            writer.register_dictionary(col_idx as i64, values);
        }
    }
    writer.write(table.clone())?;
    writer.finish()?;

    while let Some(frame) = writer.next_frame() {
        let buf = frame?;
        stream.write_all(buf.as_ref()).await?;
    }
    stream.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::IPCMessageProtocol;
    use crate::test_helpers::*;
    use minarrow::{Field, Table, Vec64};
    use std::io;

    fn all_types_schema() -> Vec<Field> {
        make_schema_all_types()
    }

    fn test_table() -> Table {
        make_all_types_table()
    }

    #[test]
    fn test_table_stream_writer_schema_and_finish() {
        let schema = all_types_schema();
        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        assert_eq!(writer.schema(), &schema[..]);
        assert!(!writer.is_finished());
        writer.finish().unwrap();
        assert!(writer.is_finished());
    }

    #[test]
    fn test_write_and_drain_one_table() {
        let schema = all_types_schema();
        let table = test_table();

        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        // Register dictionaries for categorical columns
        for (col_idx, col) in table.cols.iter().enumerate() {
            if let Some(values) = dict_values(&col.array) {
                writer.register_dictionary(col_idx as i64, values);
            }
        }
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();

        let frames = writer.drain_all_frames();
        assert!(
            !frames.is_empty(),
            "No frames emitted after writing table and finish"
        );

        // The first frame is the schema, at least one record batch frame, and an EOS marker.
        assert!(frames.len() >= 2);
        let total_len: usize = frames.iter().map(|f| f.len()).sum();
        assert!(total_len > 0);
    }

    #[test]
    fn test_multiple_batches_emit_multiple_frames() {
        let schema = all_types_schema();
        let table1 = test_table();
        let mut table2 = test_table();
        table2.name = "another".into();

        let mut writer =
            TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
        // Register dictionaries for categorical columns
        for (col_idx, col) in table1.cols.iter().enumerate() {
            if let Some(values) = dict_values(&col.array) {
                writer.register_dictionary(col_idx as i64, values);
            }
        }
        writer.write(table1.clone()).unwrap();
        writer.write(table2.clone()).unwrap();
        writer.finish().unwrap();

        let frames = writer.drain_all_frames();
        // At least: 1 schema + 2 batches + 1 EOS
        assert!(
            frames.len() >= 4,
            "Expected at least 4 frames: schema, 2 batches, EOS"
        );
    }

    #[test]
    fn test_next_frame_returns_none_when_empty() {
        let schema = all_types_schema();
        let mut writer = TableStreamWriter::<Vec64<u8>>::new(schema, IPCMessageProtocol::Stream, None);
        assert!(writer.next_frame().is_none());
        writer.finish().unwrap();
        assert!(writer.next_frame().is_none());
    }

    #[test]
    fn test_error_on_schema_mismatch() {
        let schema = all_types_schema();
        let mut bad_table = test_table();
        bad_table.cols.pop(); // Now schema and columns mismatch
        let mut writer = TableStreamWriter::<Vec64<u8>>::new(schema, IPCMessageProtocol::Stream, None);
        let err = writer.write(bad_table.clone()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    // #[test]
    // fn test_dictionary_column() {
    //     let mut table = test_table();
    //     // Add a dictionary column if not present
    //     let mut schema = all_types_schema();
    //     let dict_col = dict32_col();
    //     schema.push(dict_col.field.as_ref().clone());
    //     table.cols.push(dict_col.clone());

    //     let mut writer = TableStreamWriter::<Vec64<u8>>::new(schema.clone(), IPCMessageProtocol::Stream, None);
    //     // Register dictionary explicitly
    //     writer.register_dictionary((table.cols.len() - 1) as i64, dict_col.array.as_dict_values().unwrap());
    //     writer.write(table.clone()).unwrap();
    //     writer.finish().unwrap();

    //     let frames = writer.drain_all_frames();
    //     assert!(frames.len() >= 3, "Should emit schema, dictionary, batch, EOS");
    // }

    #[test]
    fn test_stream_trait_polling() {
        let schema = all_types_schema();
        let table = test_table();

        let mut writer = TableStreamWriter::<Vec64<u8>>::new(schema, IPCMessageProtocol::Stream, None);
        // Register dictionaries for categorical columns
        for (col_idx, col) in table.cols.iter().enumerate() {
            if let Some(values) = dict_values(&col.array) {
                writer.register_dictionary(col_idx as i64, values);
            }
        }
        writer.write(table.clone()).unwrap();
        writer.finish().unwrap();

        let mut pin_writer = Box::pin(writer);
        let mut frames = Vec::new();
        let cx = futures_util::task::noop_waker_ref();

        // Manual poll
        loop {
            match Pin::new(&mut pin_writer)
                .as_mut()
                .poll_next(&mut Context::from_waker(cx))
            {
                Poll::Ready(Some(Ok(frame))) => frames.push(frame),
                Poll::Ready(None) => break,
                Poll::Ready(Some(Err(e))) => panic!("Unexpected error from poll_next: {e}"),
                Poll::Pending => continue,
            }
        }
        assert!(
            !frames.is_empty(),
            "Should emit at least some frames through poll_next"
        );
    }
}
