// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Parallel HTTP/2 table reader
//!
//! Accepts several concurrent HTTP/2 request streams on one server
//! connection and decodes them across cores, one task per stream. Each task
//! feeds its own channel, and the reader merges the channels into a single
//! table stream. Each table is paired with its sequence key - `Some` when the
//! peer used an ordered writer, `None` otherwise.
//!
//! The h2 server connection is the I/O driver for the in-flight request
//! bodies, so a background task keeps it polled while the accepted streams
//! decode. Under [`SortBehaviour::None`] and [`SortBehaviour::RequestKeys`]
//! tables surface in the order the streams produce them. Under
//! [`SortBehaviour::Ordered`] the reader pulls the streams in the writer's
//! round-robin rotation, so tables surface in global write order.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use http::Response;
use minarrow::{Table, Vec64};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::enums::{BufferChunkSize, IPCMessageProtocol};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::readers::ipc::table::TableReader;
use crate::models::streams::http::{H2RecvRead, HttpByteStream};
use crate::traits::parallel_transport_reader::{ParallelTransportReader, SortBehaviour};
use crate::traits::parallel_transport_writer::SEQ_ID_META_KEY;

/// Per-stream flow-control window the server advertises on the handshake.
/// HTTP/2 upload throughput is bounded by the receiver's window, and h2's
/// 64 KiB default forces a WINDOW_UPDATE round-trip every 64 KiB. At 8 MiB
/// a multi-MiB Arrow batch ships without that stall, which on a
/// few-millisecond cross-host link would otherwise dominate the transfer.
const STREAM_WINDOW_BYTES: u32 = 8 * 1024 * 1024;

/// Bounded depth of each per-stream channel. Lets a stream task decode a few
/// tables ahead of the consumer without unbounded buffering, and applies
/// backpressure to a stream that runs ahead of the rotation.
const STREAM_CHANNEL_DEPTH: usize = 8;

type StreamItem = io::Result<(Table, Option<u64>)>;

/// Async Arrow IPC reader that decodes several concurrent HTTP/2 request
/// streams on one server connection in parallel and merges them into a
/// single table stream.
pub struct HttpParallelTableReader {
    streams: Vec<mpsc::Receiver<StreamItem>>,
    tasks: Vec<JoinHandle<()>>,
    driver: JoinHandle<()>,
    stream_count: usize,
    sort: SortBehaviour,
    /// Next stream to pull. Under `Ordered` this walks the writer's rotation;
    /// otherwise it rotates the starting point so no stream is starved.
    cursor: usize,
    /// Tracks which streams have closed, used by the arrival-order merge to
    /// end once every stream is drained.
    closed: Vec<bool>,
}

impl HttpParallelTableReader {
    /// Accept `stream_count` request streams on an established h2 server
    /// `connection` and decode each on its own task. A headers-only 200 is
    /// returned on each stream so the client's response drain resolves while
    /// the request body keeps uploading. `sort` selects whether sequence keys
    /// are surfaced and whether tables are emitted in global write order.
    pub async fn accept<T>(
        mut connection: h2::server::Connection<T, Bytes>,
        stream_count: usize,
        sort: SortBehaviour,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self>
    where
        T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        assert!(stream_count >= 1, "stream_count must be at least 1");
        let mut streams = Vec::with_capacity(stream_count);
        let mut tasks = Vec::with_capacity(stream_count);
        for _ in 0..stream_count {
            let (request, mut respond) = connection
                .accept()
                .await
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "h2 connection closed before all streams were accepted",
                    )
                })?
                .map_err(io::Error::other)?;
            let response = Response::builder()
                .status(200)
                .body(())
                .map_err(io::Error::other)?;
            respond
                .send_response(response, true)
                .map_err(io::Error::other)?;
            let mut reader = TableReader::<Vec64<u8>>::new(
                HttpByteStream::new(H2RecvRead::new(request.into_body()), BufferChunkSize::Http),
                BufferChunkSize::Http.chunk_size(),
                IPCMessageProtocol::Stream,
                limits,
            );
            let (tx, rx) = mpsc::channel(STREAM_CHANNEL_DEPTH);
            let task = tokio::spawn(async move {
                loop {
                    match reader.read_keyed().await {
                        Some(Ok((table, kv))) => {
                            let seq = if sort == SortBehaviour::None {
                                None
                            } else {
                                kv.and_then(|pairs| {
                                    pairs
                                        .into_iter()
                                        .find(|k| k.key == SEQ_ID_META_KEY)
                                        .and_then(|k| k.value.parse::<u64>().ok())
                                })
                            };
                            if tx.send(Ok((table, seq))).await.is_err() {
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            let _ = tx.send(Err(e)).await;
                            break;
                        }
                        None => break,
                    }
                }
            });
            streams.push(rx);
            tasks.push(task);
        }
        // Keep polling the connection so the accepted request bodies receive
        // their data frames. The loop ends when the peer closes; Drop aborts
        // it otherwise.
        let driver = tokio::spawn(async move { while connection.accept().await.is_some() {} });
        Ok(Self {
            streams,
            tasks,
            driver,
            stream_count,
            sort,
            cursor: 0,
            closed: vec![false; stream_count],
        })
    }

    /// Run the h2 server handshake on an accepted TCP stream with
    /// upload-sized flow-control windows, then accept `stream_count` request
    /// streams. POST throughput is governed by the server's flow-control
    /// window, so this advertises `STREAM_WINDOW_BYTES` per stream and
    /// scales the connection window with `stream_count`.
    pub async fn from_tcp(
        tcp: TcpStream,
        stream_count: usize,
        sort: SortBehaviour,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let connection_window =
            (stream_count as u64 * STREAM_WINDOW_BYTES as u64).min(u32::MAX as u64) as u32;
        let connection = h2::server::Builder::new()
            .initial_window_size(STREAM_WINDOW_BYTES)
            .initial_connection_window_size(connection_window)
            .handshake::<_, Bytes>(tcp)
            .await
            .map_err(io::Error::other)?;
        Self::accept(connection, stream_count, sort, limits).await
    }
}

impl Stream for HttpParallelTableReader {
    type Item = StreamItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.sort == SortBehaviour::Ordered {
            // Pull the streams in the writer's rotation. The next sequence id
            // is always the head of stream `cursor % stream_count`, so a
            // single targeted recv yields the next table in global order. A
            // closed target means that sequence will never arrive, ending the
            // merged stream.
            let idx = this.cursor % this.stream_count;
            return match this.streams[idx].poll_recv(cx) {
                Poll::Ready(Some(item)) => {
                    this.cursor += 1;
                    Poll::Ready(Some(item))
                }
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            };
        }

        // Arrival-order merge. Scan the streams from a rotating start so a
        // single busy stream cannot starve the others, returning the first
        // ready table and ending once every stream has closed.
        let n = this.stream_count;
        let mut any_pending = false;
        for offset in 0..n {
            let idx = (this.cursor + offset) % n;
            if this.closed[idx] {
                continue;
            }
            match this.streams[idx].poll_recv(cx) {
                Poll::Ready(Some(item)) => {
                    this.cursor = (idx + 1) % n;
                    return Poll::Ready(Some(item));
                }
                Poll::Ready(None) => this.closed[idx] = true,
                Poll::Pending => any_pending = true,
            }
        }
        if any_pending {
            Poll::Pending
        } else {
            Poll::Ready(None)
        }
    }
}

impl ParallelTransportReader for HttpParallelTableReader {
    fn stream_count(&self) -> usize {
        self.stream_count
    }

    async fn read_all_tables(mut self) -> io::Result<Vec<(Table, Option<u64>)>> {
        let mut out = Vec::new();
        while let Some(item) = self.next().await {
            out.push(item?);
        }
        Ok(out)
    }
}

impl Drop for HttpParallelTableReader {
    fn drop(&mut self) {
        self.driver.abort();
        for task in &self.tasks {
            task.abort();
        }
    }
}
