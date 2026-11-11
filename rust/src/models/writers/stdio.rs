// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Stdout table writer
//!
//! High-level async writer that writes Arrow IPC encoded tables to stdout.
//!
//! Wraps a [`TableSink64`] over `tokio::io::Stdout`, hiding the wiring
//! so callers get a simple API for CLI tools.
//!
//! Uses `Vec64<u8>` for 64-byte SIMD aligned encoding, matching the
//! alignment expected by the Arrow IPC frame decoder on the read side.
//!
//! ## CLI pipeline example
//!
//! ```bash
//! producer | my_tool | consumer
//! ```
//!
//! Where `my_tool` uses `StdinTableReader` to read tables and
//! `StdoutTableWriter` to write them.

use std::io;
use std::pin::Pin;

use futures_util::sink::SinkExt;
use minarrow::{Field, TableV};
use tokio::io::Stdout;

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::TableSink64;
use crate::traits::transport_writer::IPCTransportWriter;

/// Async Arrow IPC writer to stdout.
///
/// Writes Arrow IPC stream protocol data to stdout using the standard
/// encoding pipeline.
///
/// Uses `Vec64<u8>` for 64-byte SIMD aligned encoding, matching the
/// Arrow IPC frame decoder on the read side.
pub struct StdoutTableWriter {
    sink: TableSink64<Stdout>,
}

impl StdoutTableWriter {
    /// Create a new stdout table writer with the given schema.
    ///
    /// Pass `None` for `compression` to write uncompressed batches.
    ///
    /// Uses `IPCMessageProtocol::Stream` - the unbounded protocol suited
    /// for pipe-based transport where the total number of batches is not
    /// known up front.
    pub fn new(schema: Vec<Field>, compression: Option<Compression>) -> io::Result<Self> {
        let stdout = tokio::io::stdout();
        let sink = TableSink64::new(stdout, schema, IPCMessageProtocol::Stream, compression)?;
        Ok(Self { sink })
    }
}

impl IPCTransportWriter for StdoutTableWriter {
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
