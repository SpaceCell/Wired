// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebSocket table reader
//!
//! High-level async reader that connects to a WebSocket endpoint streaming
//! Arrow IPC data and decodes it into MinArrow tables.
//!
//! Extracts the raw TCP stream after the tungstenite handshake and uses
//! [`WsRead`] for zero-copy WebSocket frame parsing on the data path.
//!
//! ## Security
//!
//! The transport is whatever the URL scheme says: `ws://` is plaintext,
//! `wss://` runs the connection through tokio-tungstenite's bundled rustls
//! integration (webpki-roots verifier). Build with the `tls` feature so
//! that integration is compiled in.
//!
//! For pinned roots, a custom verifier, or client-auth keys, use
//! [`WebSocketTableReader::connect_tls`] - it takes an
//! `Arc<rustls::ClientConfig>` directly and bypasses the bundled
//! verifier. The library does not enforce a transport policy; if a
//! deployment requires TLS, that is the caller's deployment decision.
//!
//! ## Continuous streaming
//!
//! `WebSocketTableReader` implements `Stream<Item = io::Result<Table>>`, so it
//! can be used with `StreamExt` for infinite or long-lived streams:
//!
//! ```rust,no_run
//! use futures_util::StreamExt;
//! # async fn run() -> std::io::Result<()> {
//! # use lightstream::models::readers::websocket::WebSocketTableReader;
//! let mut reader = WebSocketTableReader::connect("ws://127.0.0.1:9000", None).await?;
//! while let Some(result) = reader.next().await {
//!     let table = result?;
//!     // process each batch as it arrives
//! }
//! # Ok(()) }
//! ```

use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_core::Stream;
use minarrow::{Field, SuperTable, Table, Vec64};

use crate::enums::{BufferChunkSize, IPCMessageProtocol};
use crate::models::readers::ipc::table::TableReader;
use crate::models::decoders::limits::DecodeLimits;
use crate::models::streams::websocket::{WsRead, WsWrite};
use crate::models::transports::websocket::{WebSocketListener, WebSocketTransport};
use crate::traits::transport_reader::IPCTransportReader;

/// Async Arrow IPC reader over a WebSocket connection.
///
/// Connects to a remote WebSocket endpoint, reads binary messages containing
/// Arrow IPC data, and decodes them into MinArrow tables.
///
/// Uses `WsRead` for zero-copy WebSocket frame parsing after the
/// tungstenite handshake completes.
///
/// Implements `Stream<Item = io::Result<Table>>` for continuous streaming.
pub struct WebSocketTableReader {
    inner: TableReader<Vec64<u8>>,
}

impl WebSocketTableReader {
    /// Connect to a WebSocket server streaming Arrow IPC and return a table reader.
    ///
    /// Uses `IPCMessageProtocol::Stream` and a 64 KiB initial decode capacity.
    /// The write half is dropped - use the Lightstream connection for
    /// bidirectional communication.
    pub async fn connect(url: &str, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let (read_half, write_half) = WebSocketTransport::connect(url).await?;
        let (shared_writer, _ws_write) = WsWrite::new_client(write_half);
        let ws_read = WsRead::new_client(read_half, shared_writer);
        let inner = TableReader::<Vec64<u8>>::new(
            ws_read,
            BufferChunkSize::WebSocket.chunk_size(),
            IPCMessageProtocol::Stream,
            limits,
        );
        Ok(Self { inner })
    }

    /// Accept the next inbound connection to the listener's path, run the
    /// server upgrade handshake, and return a table reader over it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `WebSocketTransport::bind`, and holds it across
    /// connections. Uses `IPCMessageProtocol::Stream`.
    pub async fn accept(
        listener: &WebSocketListener,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let (read_half, write_half) = WebSocketTransport::accept(listener).await?;
        Ok(Self::from_halves(
            read_half,
            write_half,
            IPCMessageProtocol::Stream,
            limits,
        ))
    }

    /// Wrap a raw, post-handshake TCP read half as a WebSocket table reader.
    ///
    /// The input must be an `AsyncRead` that delivers raw WebSocket frame
    /// bytes - typically the read half of the TCP/TLS stream returned by
    /// `tokio_tungstenite::accept_async(...).into_inner()`. Internally this
    /// constructor wraps the stream in a [`WsRead`] for WS frame parsing,
    /// so passing an already-built `WsRead` here would double-wrap and
    /// corrupt the parse; use [`Self::from_halves`] instead when the
    /// caller has the write half too and wants ping/pong support.
    ///
    /// Pong responses route through `tokio::io::sink()`, which silently
    /// discards them. Use [`Self::from_halves`] for full ping/pong.
    pub fn from_raw_stream(
        stream: impl tokio::io::AsyncRead + Unpin + Send + 'static,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        let shared_writer = Arc::new(Mutex::new(tokio::io::sink()));
        let ws_read = WsRead::new(stream, shared_writer);
        let inner = TableReader::<Vec64<u8>>::new(
            ws_read,
            BufferChunkSize::WebSocket.chunk_size(),
            protocol,
            limits,
        );
        Self { inner }
    }

    /// Build a table reader from the raw post-handshake read and write
    /// halves of a WebSocket-upgraded TCP/TLS stream. The write half is
    /// retained inside a shared [`WsWrite`] so the reader can answer
    /// inbound ping frames with pongs on the same socket.
    ///
    /// Typical use is the server side, where `tokio_tungstenite::accept_async`
    /// returns the upgraded `WebSocketStream`; calling `.into_inner()` and
    /// then `tokio::io::split` yields the two halves to hand here.
    pub fn from_halves<R, W>(
        read_half: R,
        write_half: W,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let (shared_writer, _ws_write) = WsWrite::new(write_half);
        let ws_read = WsRead::new(read_half, shared_writer);
        let inner = TableReader::<Vec64<u8>>::new(
            ws_read,
            BufferChunkSize::WebSocket.chunk_size(),
            protocol,
            limits,
        );
        Self { inner }
    }

    /// As [`Self::from_halves`], for the connecting peer, whose pong
    /// responses are masked as RFC 6455 requires of every
    /// client-to-server frame.
    pub fn from_client_halves<R, W>(
        read_half: R,
        write_half: W,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let (shared_writer, _ws_write) = WsWrite::new_client(write_half);
        let ws_read = WsRead::new_client(read_half, shared_writer);
        let inner = TableReader::<Vec64<u8>>::new(
            ws_read,
            BufferChunkSize::WebSocket.chunk_size(),
            protocol,
            limits,
        );
        Self { inner }
    }

    /// Connect to a `wss://` endpoint, performing the TLS handshake
    /// using the supplied `rustls::ClientConfig`, with the upgrade
    /// handshake read byte-precisely so early server frames stay in
    /// the stream for the frame parser. The caller controls verifier
    /// and root store through their config.
    #[cfg(feature = "tls")]
    pub async fn connect_tls(
        url: &str,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let (read_half, write_half) = WebSocketTransport::connect_tls(url, config).await?;
        Ok(Self::from_client_halves(
            read_half,
            write_half,
            IPCMessageProtocol::Stream,
            limits,
        ))
    }
}

impl IPCTransportReader for WebSocketTableReader {
    async fn read_all_tables(self) -> io::Result<Vec<Table>> {
        self.inner.read_all_tables().await
    }

    async fn read_tables(self, n: Option<usize>) -> io::Result<Vec<Table>> {
        self.inner.read_tables(n).await
    }

    async fn read_to_super_table(
        self,
        name: Option<String>,
        n: Option<usize>,
    ) -> io::Result<SuperTable> {
        self.inner.read_to_super_table(name, n).await
    }

    async fn combine_to_table(self, name: Option<String>) -> io::Result<Table> {
        self.inner.combine_to_table(name).await
    }

    fn schema(&self) -> Option<&[Field]> {
        self.inner.schema()
    }

    async fn read_next(&mut self) -> io::Result<Option<Table>> {
        self.inner.read_next().await
    }
}

impl Stream for WebSocketTableReader {
    type Item = io::Result<Table>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();
        Pin::new(&mut me.inner).poll_next(cx)
    }
}
