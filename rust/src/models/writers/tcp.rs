// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # TCP table writer
//!
//! High-level async writer that connects to a TCP endpoint and sends
//! Arrow IPC encoded tables over the wire.
//!
//! Wraps a [`TableSink`](crate::models::sinks::table_sink::TableSink) over a TCP write half, hiding the wiring
//! so callers get a one-liner API.
//!
//! Uses `Vec64<u8>` for 64-byte SIMD aligned encoding.

use std::io;
use std::pin::Pin;

use futures_util::sink::SinkExt;
use minarrow::{Field, TableV};
use tokio::net::tcp::OwnedWriteHalf;
#[cfg(feature = "tls")]
use tokio::net::TcpStream;
use tokio::net::{TcpListener, ToSocketAddrs};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::TableSink64;
use crate::models::streams::tcp::TcpWriteHalf;
use crate::models::transports::tcp::TcpTransport;
use crate::traits::transport_writer::IPCTransportWriter;

/// Async Arrow IPC writer over a TCP connection.
///
/// Connects to a remote TCP endpoint and writes Arrow IPC stream
/// protocol data using the standard encoding pipeline. The underlying
/// transport may be plaintext or TLS-wrapped; both share the same wire
/// framing.
///
/// Uses 64-byte SIMD aligned buffers via Vec64.
pub struct TcpTableWriter {
    sink: TableSink64<TcpWriteHalf>,
}

impl TcpTableWriter {
    /// Connect to a TCP server and prepare to write Arrow IPC tables.
    /// Pass `None` for `compression` to write uncompressed batches.
    ///
    /// Uses `IPCMessageProtocol::Stream` - the unbounded protocol suited
    /// for network transport where the total number of batches is not
    /// known up front.
    pub async fn connect(
        addr: impl ToSocketAddrs,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (_read, write) = TcpTransport::connect(addr).await?;
        let sink = TableSink64::new(
            TcpWriteHalf::Plain(write),
            schema,
            IPCMessageProtocol::Stream,
            compression,
        )?;
        Ok(Self { sink })
    }

    /// Accept the next inbound connection and prepare to write Arrow IPC
    /// tables to it.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `TcpTransport::bind`, and holds it across connections.
    /// Pass `None` for `compression` to write uncompressed batches.
    pub async fn accept(
        listener: &TcpListener,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (_read, write) = TcpTransport::accept(listener).await?;
        Self::from_write_half(write, schema, compression)
    }

    /// Wrap an existing TCP write half as a table writer.
    pub fn from_write_half(
        write_half: OwnedWriteHalf,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let sink = TableSink64::new(
            TcpWriteHalf::Plain(write_half),
            schema,
            IPCMessageProtocol::Stream,
            compression,
        )?;
        Ok(Self { sink })
    }

    /// Write a single table with Arrow custom_metadata key/value pairs
    /// attached to its record batch message, then flush.
    pub async fn write_table_with_metadata(
        &mut self,
        table: impl Into<TableV> + Send,
        metadata: Vec<(String, String)>,
    ) -> io::Result<()> {
        self.sink.core.encode_frame(table.into(), Some(metadata.as_slice()))?;
        SinkExt::flush(&mut self.sink).await?;
        Ok(())
    }

    /// Connect to a TCP server, upgrade the channel to TLS via the supplied
    /// `rustls::ClientConfig`, and return a table writer over the encrypted
    /// channel. Pass `None` for `compression` to write uncompressed batches.
    ///
    /// No default root store is bundled - the caller supplies one through
    /// their `ClientConfig`.
    #[cfg(feature = "tls")]
    pub async fn connect_tls(
        addr: impl ToSocketAddrs,
        server_name: rustls_pki_types::ServerName<'static>,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let half = Self::tls_write_half(addr, server_name, config).await?;
        let sink = TableSink64::new(half, schema, IPCMessageProtocol::Stream, compression)?;
        Ok(Self { sink })
    }

    #[cfg(feature = "tls")]
    async fn tls_write_half(
        addr: impl ToSocketAddrs,
        server_name: rustls_pki_types::ServerName<'static>,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
    ) -> io::Result<TcpWriteHalf> {
        let tcp = TcpStream::connect(addr).await?;
        let connector = tokio_rustls::TlsConnector::from(config);
        let tls = connector.connect(server_name, tcp).await?;
        let (_read_half, write_half) = tokio::io::split(tls);
        Ok(TcpWriteHalf::Tls(Box::new(write_half)))
    }
}

impl IPCTransportWriter for TcpTableWriter {
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
