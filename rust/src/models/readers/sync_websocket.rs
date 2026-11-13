// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous WebSocket table reader
//!
//! Blocking counterpart of [`WebSocketTableReader`] over `std::net`, for
//! readers that run on a dedicated thread without an async runtime.
//!
//! Wraps a [`SyncTableReader`] over a [`SyncWsRead`]. Decoding shares the
//! decode path of the async reader, with the same [`DecodeLimits`], and
//! accepts the same wire format from either WebSocket writer.
//!
//! ## Security
//!
//! The sync transport is plaintext `ws://`. For `wss://` endpoints, see the
//! async `WebSocketTableReader::connect_tls`, available with the `tls`
//! feature.
//!
//! ## Continuous streaming
//!
//! `SyncWebSocketTableReader` implements `Iterator<Item = io::Result<Table>>`,
//! which suits infinite or long-lived streams:
//!
//! ```rust,no_run
//! # fn run() -> std::io::Result<()> {
//! # use lightstream::models::readers::sync_websocket::SyncWebSocketTableReader;
//! let reader = SyncWebSocketTableReader::connect("ws://127.0.0.1:9000", None)?;
//! for result in reader {
//!     let table = result?;
//!     // process each batch as it arrives
//! }
//! # Ok(()) }
//! ```
//!
//! [`WebSocketTableReader`]: crate::models::readers::websocket::WebSocketTableReader
//! [`SyncTableReader`]: crate::models::readers::ipc::sync_table::SyncTableReader
//! [`SyncWsRead`]: crate::models::streams::sync_websocket::SyncWsRead

use std::io::{self, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use minarrow::{Field, SuperTable, Table, Vec64};

use crate::enums::{BufferChunkSize, IPCMessageProtocol};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::readers::ipc::sync_table::SyncTableReader;
use crate::models::streams::sync_websocket::{SyncWsRead, SyncWsWrite};
use crate::models::transports::sync_websocket::SyncWebSocketTransport;

/// Read-side type of [`SyncWebSocketTableReader`]. Boxing the
/// [`SyncWsRead`] lets every constructor return the same struct type for
/// different read and pong-writer halves.
type SyncWsBoxedRead = Box<dyn Read + Send + 'static>;

/// Sync Arrow IPC reader over a WebSocket connection.
///
/// Reads binary messages containing Arrow IPC data and decodes them into
/// MinArrow tables. Reads block the calling thread until bytes arrive.
///
/// Implements `Iterator<Item = io::Result<Table>>` for continuous streaming.
pub struct SyncWebSocketTableReader {
    inner: SyncTableReader<SyncWsBoxedRead, Vec64<u8>>,
}

impl SyncWebSocketTableReader {
    /// Connect to a WebSocket server streaming Arrow IPC and return a table reader.
    ///
    /// Uses `IPCMessageProtocol::Stream` and a 64 KiB initial decode capacity.
    /// The write half answers pings with pongs.
    pub fn connect(url: &str, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let (read_half, write_half) = SyncWebSocketTransport::connect(url)?;
        Ok(Self::from_client_halves(
            read_half,
            write_half,
            IPCMessageProtocol::Stream,
            limits,
        ))
    }

    /// Accept the next inbound connection, run the server upgrade
    /// handshake, and return a table reader over it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `SyncWebSocketTransport::bind`, and holds it across
    /// connections. Uses `IPCMessageProtocol::Stream`.
    pub fn accept(listener: &TcpListener, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let (read_half, write_half) = SyncWebSocketTransport::accept(listener)?;
        Ok(Self::from_halves(
            read_half,
            write_half,
            IPCMessageProtocol::Stream,
            limits,
        ))
    }

    /// Wrap a raw, post-handshake read half as a WebSocket table reader.
    ///
    /// The input must deliver raw WebSocket frame bytes. This constructor
    /// wraps the stream in a [`SyncWsRead`] for frame parsing, therefore an
    /// already-built `SyncWsRead` must not be passed here.
    ///
    /// Pong responses route through `std::io::sink()`, which discards them.
    /// Use [`Self::from_halves`] for full ping and pong handling.
    pub fn from_raw_stream(
        stream: impl Read + Send + 'static,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        let shared_writer = Arc::new(Mutex::new(io::sink()));
        Self::from_ws_read(SyncWsRead::new(stream, shared_writer), protocol, limits)
    }

    /// Build a table reader from the raw post-handshake read and write
    /// halves of a WebSocket-upgraded stream, for the accepting peer. The
    /// write half answers inbound ping frames with pongs on the same socket.
    pub fn from_halves<R, W>(
        read_half: R,
        write_half: W,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let (shared_writer, _ws_write) = SyncWsWrite::new(write_half);
        Self::from_ws_read(SyncWsRead::new(read_half, shared_writer), protocol, limits)
    }

    /// As [`Self::from_halves`], for the connecting peer, with pong
    /// responses masked as RFC 6455 requires of every client-to-server frame.
    pub fn from_client_halves<R, W>(
        read_half: R,
        write_half: W,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let (shared_writer, _ws_write) = SyncWsWrite::new_client(write_half);
        Self::from_ws_read(
            SyncWsRead::new_client(read_half, shared_writer),
            protocol,
            limits,
        )
    }

    fn from_ws_read<R, W>(
        ws_read: SyncWsRead<R, W>,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let boxed: SyncWsBoxedRead = Box::new(ws_read);
        let inner = SyncTableReader::new(
            boxed,
            BufferChunkSize::WebSocket.chunk_size(),
            protocol,
            limits,
        );
        Self { inner }
    }

    /// Read all tables from the stream until it closes.
    pub fn read_all_tables(self) -> io::Result<Vec<Table>> {
        self.inner.read_all_tables()
    }

    /// Read up to `n` tables. If `n` is `None`, read until end of stream.
    pub fn read_tables(self, n: Option<usize>) -> io::Result<Vec<Table>> {
        self.inner.read_tables(n)
    }

    /// Read batches and assemble into a `SuperTable`.
    ///
    /// If `n` is `None`, read until end of stream.
    pub fn read_to_super_table(
        self,
        name: Option<String>,
        n: Option<usize>,
    ) -> io::Result<SuperTable> {
        self.inner.read_to_super_table(name, n)
    }

    /// Read all batches and concatenate into a single `Table`.
    pub fn combine_to_table(self, name: Option<String>) -> io::Result<Table> {
        self.inner.combine_to_table(name)
    }

    /// Return the decoded schema, if available after the first schema message.
    pub fn schema(&self) -> Option<&[Field]> {
        self.inner.schema()
    }

    /// Read the next table from the stream, or `None` on end of stream.
    pub fn read_next(&mut self) -> io::Result<Option<Table>> {
        self.inner.read_next()
    }
}

impl Iterator for SyncWebSocketTableReader {
    type Item = io::Result<Table>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
