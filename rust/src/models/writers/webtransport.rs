// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebTransport table writer
//!
//! High-level async writer that sends Arrow IPC encoded tables over a
//! WebTransport send stream.
//!
//! Wraps a [`TableSink64`] over a [`wtransport::SendStream`], hiding the wiring
//! so callers get a one-liner API.
//!
//! Uses `Vec64<u8>` for 64-byte SIMD aligned encoding, matching the
//! alignment expected by the Arrow IPC frame decoder on the read side.
//!
//! ## Stability: unstable
//!
//! See the [reader docs](crate::models::readers::webtransport) - same
//! considerations apply on the write side.

use std::io;
use std::pin::Pin;

use futures_util::sink::SinkExt;
use minarrow::{Field, TableV};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::TableSink64;
use crate::traits::transport_writer::IPCTransportWriter;

/// Async Arrow IPC writer over a WebTransport send stream.
///
/// Wraps a WebTransport send stream and writes Arrow IPC stream protocol data
/// using the standard encoding pipeline.
///
/// Uses `Vec64<u8>` for 64-byte SIMD aligned encoding, matching the
/// Arrow IPC frame decoder on the read side.
pub struct WebTransportTableWriter {
    sink: TableSink64<wtransport::SendStream>,
}

impl WebTransportTableWriter {
    /// Wrap a WebTransport send stream and prepare to write Arrow IPC tables.
    /// Pass `None` for `compression` to write uncompressed batches.
    ///
    /// Uses `IPCMessageProtocol::Stream` - the unbounded protocol suited
    /// for network transport where the total number of batches is not
    /// known up front.
    pub fn new(
        send: wtransport::SendStream,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let sink = TableSink64::new(send, schema, IPCMessageProtocol::Stream, compression)?;
        Ok(Self { sink })
    }
}

impl IPCTransportWriter for WebTransportTableWriter {
    /// Get the schema used for this writer.
    fn schema(&self) -> &[Field] {
        &self.sink.core.schema
    }

    /// Register a dictionary for categorical columns.
    fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.sink.core.codec.register_dictionary(dict_id, values);
    }

    /// Write a single table and flush.
    async fn write_table(&mut self, table: impl Into<TableV> + Send) -> io::Result<()> {
        SinkExt::send(&mut self.sink, table.into()).await?;
        SinkExt::flush(&mut self.sink).await?;
        Ok(())
    }

    /// Write all tables and close.
    async fn write_all_tables(
        &mut self,
        tables: Vec<impl Into<TableV> + Send>,
    ) -> io::Result<()> {
        let mut sink = Pin::new(&mut self.sink);
        for table in tables {
            SinkExt::send(&mut sink, table.into()).await?;
        }
        SinkExt::close(&mut sink).await?;
        Ok(())
    }

    /// Finalise the stream. Must be called after writing all tables.
    async fn finish(&mut self) -> io::Result<()> {
        SinkExt::close(&mut self.sink).await
    }
}
