// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # HTTP/2 table writer
//!
//! POSTs an Arrow IPC stream to an HTTP/2 endpoint. Transport is `h2`
//! directly - hyper's Body / Service layer is not in the dep tree.
//!
//! Wraps a [`TableSink64`] over an [`H2SendWrite`] adapter, matching
//! the structural shape of every other lightstream transport writer
//! (`TcpTableWriter`, `QuicTableWriter`, ...). The encoder's
//! `encode_buf` is reused across frames as in `TableSink64`, and the
//! `H2SendWrite::poll_write` adapter feeds h2 with one chunk per
//! flow-control grant.
//!
//! Plug-and-play one-liner over `http://` is [`HttpTableWriter::post`];
//! over `https://` is [`HttpTableWriter::post_tls`]. Callers that need
//! custom headers pass a fully-built `http::Request<()>` via
//! [`HttpTableWriter::from_request`].

use std::io;
use std::pin::Pin;

use bytes::Bytes;
use futures_util::sink::SinkExt;
use http::{Method, Request, Uri};
use minarrow::{Field, TableV};
use tokio::net::{TcpListener, TcpStream};

use crate::compression::Compression;
use crate::enums::IPCMessageProtocol;
use crate::models::sinks::table_sink::TableSink64;
use crate::models::streams::http::{H2RecvRead, H2SendWrite};
use crate::models::transports::http::HttpTransport;
use crate::traits::transport_writer::IPCTransportWriter;

/// Async Arrow IPC writer over an HTTP/2 POST request body.
pub struct HttpTableWriter {
    sink: TableSink64<H2SendWrite>,
}

impl HttpTableWriter {
    /// POST an Arrow IPC stream to an `http://` URL over plaintext h2c.
    /// Pass `None` for `compression` to write uncompressed batches.
    pub async fn post(
        url: &str,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let req = parse_post(url)?;
        Self::from_request(req, schema, compression).await
    }

    /// POST an Arrow IPC stream to an `https://` URL over h2. ALPN must
    /// be set to `h2` on the supplied `rustls::ClientConfig`. No default
    /// root store is bundled. Pass `None` for `compression` to write
    /// uncompressed batches.
    #[cfg(feature = "tls")]
    pub async fn post_tls(
        url: &str,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let req = parse_post(url)?;
        Self::from_request_tls(req, config, schema, compression).await
    }

    /// Issue a fully-built request (typically POST) and stream Arrow
    /// IPC batches into its body. Use when the request needs custom
    /// headers like `Authorization` or `Content-Type`. Scheme must be
    /// `http`. Pass `None` for `compression` to write uncompressed batches.
    pub async fn from_request(
        req: Request<()>,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (host, port) = host_port(req.uri(), "http", 80)?;
        let tcp = TcpStream::connect((host.as_str(), port)).await?;
        let (send_stream, response_fut) = h2_send_post(tcp, req).await?;
        Self::from_send_stream(send_stream, response_fut, schema, compression)
    }

    /// As [`Self::from_request`], over HTTPS h2. Scheme must be `https`.
    #[cfg(feature = "tls")]
    pub async fn from_request_tls(
        req: Request<()>,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (host, port) = host_port(req.uri(), "https", 443)?;
        let server_name = rustls_pki_types::ServerName::try_from(host.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let tcp = TcpStream::connect((host.as_str(), port)).await?;
        let connector = tokio_rustls::TlsConnector::from(config);
        let tls = connector.connect(server_name, tcp).await?;
        if tls.get_ref().1.alpn_protocol() != Some(b"h2") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "TLS ALPN did not negotiate h2; set \
                 config.alpn_protocols = vec![b\"h2\".to_vec()] on ClientConfig",
            ));
        }
        let (send_stream, response_fut) = h2_send_post(tls, req).await?;
        Self::from_send_stream(send_stream, response_fut, schema, compression)
    }

    /// Accept the next inbound exchange and prepare to write Arrow IPC
    /// tables into its response body.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `HttpTransport::bind`, and holds it across connections.
    /// The request direction carries no data in this role, so a task
    /// drains it to its clean end while the response streams out. Pass
    /// `None` for `compression` to write uncompressed batches.
    pub async fn accept(
        listener: &TcpListener,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        let (recv_read, send_write) = HttpTransport::accept(listener).await?;
        Self::from_exchange(recv_read, send_write, schema, compression)
    }

    /// Build a writer over an accepted exchange's byte halves, streaming
    /// Arrow IPC batches into the response body. The request direction
    /// carries no data in this role, so a task drains it to its clean
    /// end while the response streams out. Pass `None` for `compression`
    /// to write uncompressed batches.
    pub fn from_exchange(
        mut recv_read: H2RecvRead,
        send_write: H2SendWrite,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        tokio::spawn(async move {
            let _ = tokio::io::copy(&mut recv_read, &mut tokio::io::sink()).await;
        });
        let sink = TableSink64::new(send_write, schema, IPCMessageProtocol::Stream, compression)?;
        Ok(Self { sink })
    }

    /// Build a writer over an already-open h2 request stream and its
    /// response future. The parallel writer uses this to place one writer
    /// on each request stream of a shared connection. Pass `None` for
    /// `compression` to write uncompressed batches.
    pub fn from_send_stream(
        send_stream: h2::SendStream<Bytes>,
        response_fut: h2::client::ResponseFuture,
        schema: Vec<Field>,
        compression: Option<Compression>,
    ) -> io::Result<Self> {
        // h2 expects the ResponseFuture to be polled; otherwise the
        // stream can be reset. Spawn a fire-and-forget task that awaits
        // it. The response body is not surfaced through the writer API;
        // a caller needing it should consume the response via h2 directly.
        tokio::spawn(async move {
            let _ = response_fut.await;
        });
        let write = H2SendWrite::new(send_stream);
        let sink = TableSink64::new(write, schema, IPCMessageProtocol::Stream, compression)?;
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
}

impl IPCTransportWriter for HttpTableWriter {
    fn schema(&self) -> &[Field] {
        &self.sink.core.schema
    }

    fn register_dictionary(&mut self, dict_id: i64, values: Vec<String>) {
        self.sink.core.codec.register_dictionary(dict_id, values);
    }

    async fn write_table(&mut self, table: impl Into<TableV> + Send) -> io::Result<()> {
        SinkExt::send(&mut self.sink, table.into()).await?;
        SinkExt::flush(&mut self.sink).await?;
        Ok(())
    }

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

    async fn finish(&mut self) -> io::Result<()> {
        SinkExt::close(&mut self.sink).await
    }
}

// ---------------------------------------------------------------------------
// Internals - URI parsing + h2 handshake + POST request dispatch
// ---------------------------------------------------------------------------

fn parse_post(url: &str) -> io::Result<Request<()>> {
    let uri: Uri = url
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .body(())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

fn host_port(uri: &Uri, expected_scheme: &str, default_port: u16) -> io::Result<(String, u16)> {
    let scheme = uri
        .scheme_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "uri missing scheme"))?;
    if scheme != expected_scheme {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected scheme {expected_scheme}, got {scheme}"),
        ));
    }
    let host = uri
        .host()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "uri missing host"))?
        .to_string();
    let port = uri.port_u16().unwrap_or(default_port);
    Ok((host, port))
}

/// Run the h2 handshake on `io`, send `req` with end_of_stream=false
/// (body comes via subsequent send_data calls), spawn both the
/// connection driver and a task that drains the server's response.
async fn h2_send_post<T>(
    io: T,
    req: Request<()>,
) -> io::Result<(h2::SendStream<Bytes>, h2::client::ResponseFuture)>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin + 'static,
{
    // For POST throughput it is the *server's* initial_window_size that
    // governs the client's send capacity, so this client-side handshake
    // takes h2's defaults. The receiving server is responsible for
    // advertising larger windows when it cares about bulk-upload
    // throughput; see the example and bench server setups.
    let (mut send_request, connection) =
        h2::client::handshake(io).await.map_err(io::Error::other)?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!("h2 connection driver exited: {e}");
        }
    });
    let (response_fut, send_stream) = send_request
        .send_request(req, false)
        .map_err(io::Error::other)?;
    Ok((send_stream, response_fut))
}
