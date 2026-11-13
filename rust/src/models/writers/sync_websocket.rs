// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous WebSocket table writer
//!
//! Blocking counterpart of [`WebSocketTableWriter`] over `std::net`, for
//! writers that run on a dedicated thread without an async runtime.
//!
//! Wraps a [`SyncTableSink`] over a [`SyncWsWrite`], as
//! [`WebSocketTableWriter`] wraps the async table sink over `WsWrite`. The
//! wire format is the Arrow IPC stream protocol with 64-byte aligned frames.
//! Each table is sent as one binary WebSocket message. The first message
//! holds the schema and dictionary frames followed by the record batch, later
//! messages hold the record batch, and [`SyncWebSocketTableWriter::finish`]
//! sends the end-of-stream marker as a final message. The messages are
//! identical to those of [`WebSocketTableWriter`]. Either WebSocket reader,
//! and browsers reading through the `apache-arrow` library, receive the same
//! stream.
//!
//! Both peer roles are supported. [`SyncWebSocketTableWriter::connect`]
//! reaches a listening `ws://` URL, and [`SyncWebSocketTableWriter::accept`]
//! serves the next inbound connection on a bound `TcpListener`.
//!
//! ## Security
//!
//! The sync transport is plaintext `ws://`. For `wss://` endpoints, see the
//! async `WebSocketTableWriter::connect_tls`, available with the `tls`
//! feature.
//!
//! [`WebSocketTableWriter`]: crate::models::writers::websocket::WebSocketTableWriter
//! [`SyncTableSink`]: crate::models::sinks::sync_table_sink::SyncTableSink
//! [`SyncWsWrite`]: crate::models::streams::sync_websocket::SyncWsWrite

use std::io::{self, Read, Write};
use std::net::TcpListener;

use minarrow::{Field, Table, TableV, Vec64};

use crate::compression::Compression;
use crate::models::sinks::sync_table_sink::SyncTableSink;
use crate::models::streams::sync_websocket::SyncWsWrite;
use crate::models::transports::sync_websocket::SyncWebSocketTransport;

/// Write half of [`SyncWebSocketTableWriter`]. Boxing the underlying
/// `std::io::Write` lets every constructor return the same struct type for
/// different write halves.
type WsSyncWrite = Box<dyn Write + Send + 'static>;

/// Sync Arrow IPC writer over a WebSocket connection.
///
/// Writes Arrow IPC stream protocol data as binary WebSocket messages, with
/// `Vec64<u8>` 64-byte SIMD aligned encoding. Writes block the calling thread
/// until the socket accepts the bytes.
pub struct SyncWebSocketTableWriter {
    sink: SyncTableSink<SyncWsWrite<WsSyncWrite>, Vec64<u8>>,
}

impl SyncWebSocketTableWriter {
    /// Connect to a WebSocket server and prepare to write Arrow IPC tables.
    /// Pass `None` for `compression` to write uncompressed batches.
    ///
    /// Uses `IPCMessageProtocol::Stream` - the unbounded protocol suited
    /// for network transport where the total number of batches is not
    /// known up front. Frames are masked, as RFC 6455 requires of the
    /// connecting peer.
    pub fn connect(
        url: &str,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (_read_half, write_half) = SyncWebSocketTransport::connect(url)?;
        let boxed: WsSyncWrite = Box::new(write_half);
        let (_shared, ws_write) = SyncWsWrite::new_client(boxed);
        let sink = SyncTableSink::new(ws_write, schema, compression);
        Ok(Self { sink })
    }

    /// Accept the next inbound connection, run the server upgrade
    /// handshake, and prepare to write Arrow IPC tables to it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `SyncWebSocketTransport::bind`, and holds it across
    /// connections. Pass `None` for `compression` to write
    /// uncompressed batches.
    pub fn accept(
        listener: &TcpListener,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (read_half, write_half) = SyncWebSocketTransport::accept(listener)?;
        Ok(Self::from_halves(read_half, write_half, schema, compression))
    }

    /// Build a table writer from the raw post-handshake read and write
    /// halves of a WebSocket-upgraded stream, for the accepting peer.
    ///
    /// Symmetric to
    /// [`SyncWebSocketTableReader::from_halves`](crate::models::readers::sync_websocket::SyncWebSocketTableReader::from_halves).
    pub fn from_halves<R, W>(
        _read_half: R,
        write_half: W,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> Self
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let boxed: WsSyncWrite = Box::new(write_half);
        let (_shared, ws_write) = SyncWsWrite::new(boxed);
        let sink = SyncTableSink::new(ws_write, schema, compression);
        Self { sink }
    }

    /// Get the schema used for this writer.
    pub fn schema(&self) -> &[Field] {
        self.sink.schema()
    }

    /// Register a dictionary for categorical columns.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.sink.register_dictionary(dict_id, values);
    }

    /// Write a single table and send its frames to the socket.
    pub fn write_table(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        self.sink.write_table(table)
    }

    /// Write all tables and close.
    pub fn write_all_tables<I>(&mut self, tables: I) -> io::Result<()>
    where
        I: IntoIterator<Item = Table>,
    {
        for table in tables {
            self.sink.write_table(table)?;
        }
        self.finish()
    }

    /// Finalise the stream. Must be called after writing all tables.
    ///
    /// Writes the end-of-stream marker, then the close frame. Later calls
    /// flush only. The connection closes when the writer is dropped.
    pub fn finish(&mut self) -> io::Result<()> {
        self.sink.finish()?;
        // The peer may already have closed the connection after reading the
        // end-of-stream marker.
        let _ = self.sink.sink_mut().close();
        Ok(())
    }
}
