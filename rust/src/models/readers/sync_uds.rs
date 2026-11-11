// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous Unix domain socket table reader
//!
//! Blocking counterpart of [`UdsTableReader`] over `std::os::unix::net`,
//! for readers that run on a dedicated thread with no async runtime.
//!
//! Wraps a [`SyncTableReader`] over a `UnixStream`. Decoding shares the
//! decode path of the async reader, with the same [`DecodeLimits`], and
//! accepts the same wire format from either UDS writer.
//!
//! ## Continuous streaming
//!
//! `SyncUdsTableReader` implements `Iterator<Item = io::Result<Table>>`,
//! which suits infinite or long-lived streams:
//!
//! ```rust,no_run
//! # fn run() -> std::io::Result<()> {
//! # use lightstream::models::readers::sync_uds::SyncUdsTableReader;
//! let reader = SyncUdsTableReader::connect("/tmp/my.sock", None)?;
//! for result in reader {
//!     let table = result?;
//!     // process each batch as it arrives
//! }
//! # Ok(()) }
//! ```
//!
//! [`UdsTableReader`]: crate::models::readers::uds::UdsTableReader
//! [`SyncTableReader`]: crate::models::readers::ipc::sync_table::SyncTableReader

use std::io;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;

use minarrow::{Field, SuperTable, Table, Vec64};

use crate::enums::{BufferChunkSize, IPCMessageProtocol};
use crate::models::decoders::limits::DecodeLimits;
use crate::models::readers::ipc::sync_table::SyncTableReader;

/// Sync Arrow IPC reader over a Unix domain socket connection.
///
/// Reads an Arrow IPC stream from a local UDS connection and decodes it into
/// MinArrow tables. Reads block the calling thread until bytes arrive.
///
/// Implements `Iterator<Item = io::Result<Table>>` for continuous streaming.
pub struct SyncUdsTableReader {
    inner: SyncTableReader<UnixStream, Vec64<u8>>,
}

impl SyncUdsTableReader {
    /// Connect to a UDS server streaming Arrow IPC and return a table reader.
    ///
    /// Uses `BufferChunkSize::Http` (64 KiB) and protocol
    /// `IPCMessageProtocol::Stream`.
    pub fn connect(path: impl AsRef<Path>, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        Ok(Self::from_stream(stream, IPCMessageProtocol::Stream, limits))
    }

    /// Connect with chunk size and protocol control.
    pub fn connect_with(
        path: impl AsRef<Path>,
        chunk_size: BufferChunkSize,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        let inner = SyncTableReader::new(stream, chunk_size.chunk_size(), protocol, limits);
        Ok(Self { inner })
    }

    /// Accept the next inbound connection and return a table reader over it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `UnixListener::bind`, and holds it across connections.
    /// Uses `BufferChunkSize::Http` (64 KiB) and protocol
    /// `IPCMessageProtocol::Stream`.
    pub fn accept(listener: &UnixListener, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let (stream, _) = listener.accept()?;
        Ok(Self::from_stream(stream, IPCMessageProtocol::Stream, limits))
    }

    /// Wrap an existing connected `UnixStream` as a table reader.
    pub fn from_stream(
        stream: UnixStream,
        protocol: IPCMessageProtocol,
        limits: Option<DecodeLimits>,
    ) -> Self {
        let inner =
            SyncTableReader::new(stream, BufferChunkSize::Http.chunk_size(), protocol, limits);
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

impl Iterator for SyncUdsTableReader {
    type Item = io::Result<Table>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
