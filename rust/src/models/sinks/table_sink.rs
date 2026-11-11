// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Asynchronous Arrow IPC table sink
//!
//! Wraps any `AsyncWrite` and streams [`minarrow::Table`] values as Arrow IPC frames.
//!
//! ## Overview:
//! - Supports Stream or File protocol
//! - Handles schema emission, optional compression, dictionary batches, record batches, and end-of-stream/footer generation.
//! - Supports both 8-byte (`Vec<u8>`) and 64-byte SIMD-aligned (`Vec64<u8>`) buffers via [`TableSink`] and [`TableSink64`] type aliases.
//! - Supports backpressure-friendly, chunked writes with partial-write handling in async runtimes (e.g. Tokio).

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::codecs::ipc::ArrowIpcCodec;
use crate::models::writers::ipc::table_stream::TableStreamWriter;
use crate::traits::stream_buffer::StreamBuffer;
use minarrow::{Field, TableV, Vec64};
use std::io;
use tokio::io::AsyncWrite;

use futures_sink::Sink;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Async Arrow Table Sink for (`Vec<u8>`).
///
/// Use this writer to stream Arrow [`minarrow::Table`] values to any asynchronous byte sink,
/// automatically handling Arrow IPC framing, schema, dictionaries, and alignment.
/// It will write Arrow tables with 8-byte alignment.
///
/// When writing `Minarrow` objects, and high-performance/64-byte SIMD, use `TableWriter64`.
pub type TableSink<W> = GTableSink<W, Vec<u8>>;

/// Async Arrow Table Sink for (`Vec64<u8>`).
///
/// Prefer this when you plan to use the SIMD feature for the `Lightning` crate kernels + library,
/// or otherwise want to make sure that the arrow stream /file you are writing to has
/// 64 byte aligned Arrow buffers.
pub type TableSink64<W> = GTableSink<W, Vec64<u8>>;

/// Transport-independent encode state of the Arrow IPC table sinks.
///
/// Holds the schema, the codec, the pooled encode buffer and the File
/// protocol frame writer for one IPC session, and returns encoded frames for
/// the owning sink to write to its destination. The async [`GTableSink`] and
/// the blocking [`SyncTableSink`] encode through this state and differ only
/// in how bytes reach the destination.
///
/// [`SyncTableSink`]: crate::models::sinks::sync_table_sink::SyncTableSink
pub(crate) struct TableSinkCore<B>
where
    B: StreamBuffer + Unpin + 'static,
{
    pub(crate) schema: Vec<Field>,
    pub(crate) codec: ArrowIpcCodec<B>,
    pub(crate) protocol: IPCMessageProtocol,
    pub(crate) finished: bool,
    pub(crate) frame_buf: Option<B>, // Current frame being written
    pub(crate) frame_pos: usize,     // How many bytes have been written so far
    /// Pooled encode buffer reused across send_table calls.
    pub(crate) encode_buf: B,
    /// Frame-by-frame writer for File protocol with footer tracking
    pub(crate) file_writer: Option<TableStreamWriter<B>>,
}

impl<B> TableSinkCore<B>
where
    B: StreamBuffer + Unpin + 'static,
{
    /// Create the encode state for one IPC session. Pass `None` for
    /// `compression` to write uncompressed batches; `Some(codec)` compresses
    /// every record-batch body.
    pub(crate) fn new(
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
            protocol,
            finished: false,
            frame_buf: None,
            frame_pos: 0,
            encode_buf: B::with_capacity(0),
            file_writer,
        }
    }

    /// Register a dictionary with the given id and values.
    pub(crate) fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.codec.register_dictionary(dict_id, values.clone());
        if let Some(writer) = &mut self.file_writer {
            writer.register_dictionary(dict_id, values);
        }
    }

    /// Encode the table view into the pending frame buffer, attaching
    /// `custom_metadata` to its record batch message.
    pub(crate) fn encode_frame(
        &mut self,
        view: TableV,
        custom_metadata: Option<&[(String, String)]>,
    ) -> io::Result<()> {
        if self.protocol == IPCMessageProtocol::Stream {
            // The Stream protocol encodes into a pooled buffer reused across
            // sends.
            let mut buf = std::mem::replace(&mut self.encode_buf, B::with_capacity(0));
            let len = buf.len();
            if len > 0 {
                buf.drain(0..len);
            }
            self.codec
                .encode_stream_batch(&view, &mut buf, 0, custom_metadata)?;
            self.frame_buf = Some(buf);
            self.frame_pos = 0;
        } else if let Some(writer) = &mut self.file_writer {
            // The File protocol routes through the frame-by-frame writer that
            // tracks footer blocks.
            writer.write(view)?;
        }
        Ok(())
    }

    /// Return the next encoded frame for the destination, or `None` when no
    /// frame is waiting.
    ///
    /// The pending frame buffer is returned first, followed by the frames
    /// queued by the File protocol writer.
    pub(crate) fn next_frame(&mut self) -> Option<B> {
        if let Some(buf) = self.frame_buf.take() {
            return Some(buf);
        }
        // If no frame buffered, pull next from the file writer's queue.
        if let Some(writer) = &mut self.file_writer {
            if let Some(Ok(frame)) = writer.next_frame() {
                self.frame_pos = 0;
                return Some(frame);
            }
        }
        None
    }

    /// Return a fully written frame buffer to the pool for the next encode.
    pub(crate) fn reclaim(&mut self, buf: B) {
        self.encode_buf = buf;
        self.frame_pos = 0;
    }

    /// Queue the end-of-stream marker or the File footer. Later calls do
    /// nothing.
    pub(crate) fn finish(&mut self) -> io::Result<()> {
        if !self.finished {
            if let Some(writer) = &mut self.file_writer {
                // File protocol: writer handles footer + EOS
                writer.finish()?;
            } else {
                // Stream protocol: write EOS into a buffer and queue it
                let mut eos_buf = B::with_capacity(8);
                self.codec.finish(&mut eos_buf)?;
                self.frame_buf = Some(eos_buf);
                self.frame_pos = 0;
            }
            self.finished = true;
        }
        Ok(())
    }
}

/// Generic asynchronous Arrow Table sink for Tokio and other async runtimes.
///
/// This wraps any compatible async byte sink (`W: AsyncWrite`) and handles
/// Arrow IPC protocol (framing, schema, dictionaries, record batches, footers).
///
/// It yields whole `Table` objects and is the recommended sink for *Minarrow* data.
/// Tables it yields are analogous with *Apache Arrow* *`Record Batches`* once
/// written into the wider ecosystem.
pub struct GTableSink<W, B>
where
    W: AsyncWrite + Unpin + Send + 'static,
    B: StreamBuffer + Unpin + 'static,
{
    pub(crate) core: TableSinkCore<B>,
    pub(crate) destination: W,
}

impl<W, B> GTableSink<W, B>
where
    W: AsyncWrite + Unpin + Send + 'static,
    B: StreamBuffer + std::fmt::Debug + Unpin + 'static,
{
    /// Create a new generic Arrow Table writer. Pass `None` for
    /// `compression` to write uncompressed batches; `Some(codec)` compresses
    /// every record-batch body.
    pub fn new(
        sink: W,
        schema: Vec<Field>,
        protocol: IPCMessageProtocol,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        Ok(Self {
            core: TableSinkCore::new(schema, protocol, compression),
            destination: sink,
        })
    }

    /// Expose a mutable reference to the inner sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.destination
    }
}

impl<W, B> Sink<TableV> for GTableSink<W, B>
where
    W: AsyncWrite + Unpin + Send + 'static,
    B: StreamBuffer + std::fmt::Debug + Unpin + 'static,
{
    type Error = io::Error;

    /// Admit a new table once the pending frame has reached the sink.
    fn poll_ready(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        if self.core.frame_buf.is_some() {
            return self.as_mut().poll_flush(cx);
        }
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, view: TableV) -> Result<(), Self::Error> {
        self.get_mut().core.encode_frame(view, None)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // drain encoder into sink, honouring partial writes
        loop {
            // Attempt to write current frame.
            if let Some(buf) = self.core.next_frame() {
                let remaining = &buf.as_ref()[self.core.frame_pos..];
                // Limit chunk size to avoid blocking the async runtime
                const MAX_WRITE_CHUNK: usize = 1024 * 1024; // 1MB chunks
                let chunk = if remaining.len() > MAX_WRITE_CHUNK {
                    &remaining[..MAX_WRITE_CHUNK]
                } else {
                    remaining
                };
                match Pin::new(&mut self.destination).poll_write(cx, chunk) {
                    Poll::Pending => {
                        self.core.frame_buf = Some(buf);
                        return Poll::Pending;
                    }
                    Poll::Ready(Ok(0)) => return Poll::Ready(Err(io::ErrorKind::WriteZero.into())),
                    Poll::Ready(Ok(n)) => {
                        self.core.frame_pos += n;
                        if self.core.frame_pos < buf.as_ref().len() {
                            self.core.frame_buf = Some(buf);
                            cx.waker().wake_by_ref();
                            return Poll::Pending;
                        } else {
                            // Frame fully written - reclaim the buffer for reuse
                            self.core.reclaim(buf);
                        }
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                }
            } else {
                break; // nothing buffered
            }
        }

        // flush underlying AsyncWrite
        Pin::new(&mut self.destination).poll_flush(cx)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // Emit EOS / footer
        self.core.finish()?;

        // Drain all remaining frames
        match self.as_mut().poll_flush(cx)? {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(()) => { /* fall‑through */ }
        }

        // Shut down the underlying AsyncWrite
        Pin::new(&mut self.destination)
            .poll_shutdown(cx)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::SinkExt;
    use minarrow::Table;

    use super::*;
    use crate::models::decoders::ipc::table_stream_decoder::TableStreamDecoder;
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

    async fn decoded_batches(bytes: Vec<u8>) -> usize {
        let mut decoder = TableStreamDecoder::<Vec64<u8>>::new(
            std::io::Cursor::new(bytes),
            64 * 1024,
            IPCMessageProtocol::Stream,
            None,
        );
        let mut count = 0;
        while let Some(item) = decoder.read_keyed().await {
            item.expect("stream decoded cleanly");
            count += 1;
        }
        count
    }

    /// `feed` admits an item without flushing, so a caller may queue several
    /// tables against a single flush. Every one of them must still reach the
    /// wire, and the stream must stay decodable.
    #[tokio::test]
    async fn feed_without_flush_keeps_every_table() {
        let table = test_table();
        let mut sink =
            TableSink64::new(Vec::new(), schema_of(&table), IPCMessageProtocol::Stream, None)
                .unwrap();

        sink.feed(table.clone().into()).await.unwrap();
        sink.feed(table.clone().into()).await.unwrap();
        sink.feed(table.clone().into()).await.unwrap();
        SinkExt::flush(&mut sink).await.unwrap();
        SinkExt::close(&mut sink).await.unwrap();

        let bytes = std::mem::take(&mut sink.destination);
        assert_eq!(decoded_batches(bytes).await, 3);
    }

    /// `send` flushes per item, the path the transport writers take.
    #[tokio::test]
    async fn send_per_item_keeps_every_table() {
        let table = test_table();
        let mut sink =
            TableSink64::new(Vec::new(), schema_of(&table), IPCMessageProtocol::Stream, None)
                .unwrap();

        SinkExt::send(&mut sink, table.clone().into()).await.unwrap();
        SinkExt::send(&mut sink, table.clone().into()).await.unwrap();
        SinkExt::close(&mut sink).await.unwrap();

        let bytes = std::mem::take(&mut sink.destination);
        assert_eq!(decoded_batches(bytes).await, 2);
    }
}
