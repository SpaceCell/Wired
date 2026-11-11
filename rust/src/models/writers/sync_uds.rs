// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Unix domain socket table writer
//!
//! Blocking counterpart of [`UdsTableWriter`] over `std::os::unix::net`,
//! for writers that run on a dedicated thread with no async runtime.
//!
//! Wraps a [`SyncTableWriter`] over a `UnixStream`. The wire format is the
//! Arrow IPC stream protocol with 64-byte aligned frames, identical to
//! [`UdsTableWriter`], so either UDS reader receives the tables.
//!
//! Both peer roles are supported. [`SyncUdsTableWriter::connect`] reaches a
//! listening socket path, and [`SyncUdsTableWriter::accept`] serves the next
//! inbound connection on a bound `UnixListener`.
//!
//! [`UdsTableWriter`]: crate::models::writers::uds::UdsTableWriter

use std::io;
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;

use minarrow::{Field, Table, TableV, Vec64};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::writers::ipc::sync_table::SyncTableWriter;

/// Sync Arrow IPC writer over a Unix domain socket connection.
///
/// Writes Arrow IPC stream protocol data with `Vec64<u8>` 64-byte SIMD
/// aligned encoding, matching the Arrow IPC frame decoder on the read side.
/// Writes block the calling thread until the socket accepts the bytes.
pub struct SyncUdsTableWriter {
    inner: SyncTableWriter<UnixStream, Vec64<u8>>,
}

impl SyncUdsTableWriter {
    /// Connect to a UDS server and prepare to write Arrow IPC tables.
    /// Pass `None` for `compression` to write uncompressed batches.
    ///
    /// Uses `IPCMessageProtocol::Stream` - the unbounded protocol suited
    /// for network transport where the total number of batches is not
    /// known up front.
    pub fn connect(
        path: impl AsRef<Path>,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        Ok(Self::from_stream(stream, schema, compression))
    }

    /// Accept the next inbound connection and prepare to write Arrow IPC
    /// tables to it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `UnixListener::bind`, and holds it across connections.
    /// Pass `None` for `compression` to write uncompressed batches.
    pub fn accept(
        listener: &UnixListener,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (stream, _) = listener.accept()?;
        Ok(Self::from_stream(stream, schema, compression))
    }

    /// Wrap an existing connected `UnixStream` as a table writer.
    pub fn from_stream(
        stream: UnixStream,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> Self {
        let inner =
            SyncTableWriter::new(stream, schema, IPCMessageProtocol::Stream, compression);
        Self { inner }
    }

    /// Get the schema used for this writer.
    pub fn schema(&self) -> &[Field] {
        self.inner.schema()
    }

    /// Register a dictionary for categorical columns.
    pub fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.inner.register_dictionary(dict_id, values);
    }

    /// Write a single table and send its frames to the socket.
    pub fn write_table(&mut self, table: impl Into<TableV>) -> io::Result<()> {
        self.inner.write_table(table)
    }

    /// Write all tables and close.
    pub fn write_all_tables<I>(&mut self, tables: I) -> io::Result<()>
    where
        I: IntoIterator<Item = Table>,
    {
        for table in tables {
            self.inner.write_table(table)?;
        }
        self.finish()
    }

    /// Finalise the stream. Must be called after writing all tables.
    ///
    /// Writes the end-of-stream marker and shuts down the write direction of
    /// the socket, as the async writer does on close.
    pub fn finish(&mut self) -> io::Result<()> {
        self.inner.finish()?;
        match self.inner.sink_mut().shutdown(Shutdown::Write) {
            // The peer may already have closed after reading the marker.
            Err(e) if e.kind() == io::ErrorKind::NotConnected => Ok(()),
            other => other,
        }
    }
}
