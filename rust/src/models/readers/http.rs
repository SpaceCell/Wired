// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # HTTP/2 table reader
//!
//! Issues a GET to an HTTP/2 endpoint and decodes the response body as
//! an Arrow IPC stream. The transport is `h2` directly - hyper's Body /
//! Service layer is not in the dep tree.
//!
//! Plug-and-play one-liner over `http://` is [`HttpTableReader::get`](crate::models::readers::http::HttpTableReader::get);
//! over `https://` is [`HttpTableReader::get_tls`] (requires the `tls`
//! feature and a caller-supplied `rustls::ClientConfig` with ALPN set
//! to `h2`). Callers that need custom headers (auth tokens, API keys)
//! pass a fully-built `http::Request<()>` via
//! [`HttpTableReader::from_request`].
//!
//! ## Continuous streaming
//!
//! `HttpTableReader` implements `Stream<Item = io::Result<Table>>`:
//!
//! ```ignore
//! use futures_util::StreamExt;
//! use lightstream::models::readers::http::HttpTableReader;
//!
//! let mut reader = HttpTableReader::get("http://localhost:8080/feed", None).await?;
//! while let Some(result) = reader.next().await {
//!     let table = result?;
//!     // ...
//! }
//! ```

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use http::{Request, Uri};
use minarrow::{Field, SuperTable, Table, Vec64};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::enums::{BufferChunkSize, IPCMessageProtocol};
use crate::models::readers::ipc::table::TableReader;
use crate::models::decoders::limits::DecodeLimits;
use crate::models::streams::http::{H2RecvRead, H2SendWrite, HttpByteStream};
use crate::models::transports::http::{HttpListener, HttpTransport};
use crate::traits::transport_reader::IPCTransportReader;

/// Async Arrow IPC reader over an HTTP/2 GET response body.
pub struct HttpTableReader {
    inner: TableReader<Vec64<u8>>,
}

impl HttpTableReader {
    /// GET an `http://` URL over plaintext h2c and decode the response
    /// body as an Arrow IPC stream. Most public REST endpoints don't
    /// speak h2c; this exists for local testing and for deployments
    /// behind a TLS-terminating proxy that exposes h2c internally.
    pub async fn get(url: &str, limits: Option<DecodeLimits>) -> io::Result<Self> {
        let req = parse_get(url)?;
        Self::from_request(req, limits).await
    }

    /// GET an `https://` URL over h2, performing the TLS handshake
    /// using the supplied `rustls::ClientConfig`. ALPN must be set to
    /// `h2` on the config (the post-handshake check rejects anything
    /// else). No default root store is bundled.
    #[cfg(feature = "tls")]
    pub async fn get_tls(
        url: &str,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let req = parse_get(url)?;
        Self::from_request_tls(req, config, limits).await
    }

    /// Issue a fully-built request (typically GET) and stream the
    /// response body as an Arrow IPC stream. Use when the request needs
    /// custom headers like `Authorization` or `X-API-Key`. Scheme must
    /// be `http`.
    pub async fn from_request(
        req: Request<()>,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let (host, port) = host_port(req.uri(), "http", 80)?;
        let tcp = TcpStream::connect((host.as_str(), port)).await?;
        let recv = h2_send_get(tcp, req).await?;
        Ok(Self::from_recv(recv, limits))
    }

    /// As [`Self::from_request`], over HTTPS h2. Scheme must be `https`.
    #[cfg(feature = "tls")]
    pub async fn from_request_tls(
        req: Request<()>,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
        limits: Option<DecodeLimits>,
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
        let recv = h2_send_get(tls, req).await?;
        Ok(Self::from_recv(recv, limits))
    }

    /// Wrap a server-supplied [`h2::RecvStream`] (typically from
    /// `h2::server::handshake` → `accept` → `Request<RecvStream>`) as a
    /// table reader. Symmetric to
    /// [`QuicTableReader::from_recv`](crate::models::readers::quic::QuicTableReader::from_recv).
    pub fn from_recv(recv: h2::RecvStream, limits: Option<DecodeLimits>) -> Self {
        let stream =
            HttpByteStream::new(H2RecvRead::new(recv), crate::enums::BufferChunkSize::Http);
        let inner = TableReader::<Vec64<u8>>::new(
            stream,
            BufferChunkSize::Http.chunk_size(),
            IPCMessageProtocol::Stream,
            limits,
        );
        Self { inner }
    }

    /// Accept the next inbound exchange for the listener's path and return
    /// a table reader over its request body.
    ///
    /// Serves the accepting peer role. The caller binds the listener,
    /// e.g. via `HttpTransport::bind`, and holds it across connections.
    /// The response direction carries no data in this role, so it is
    /// ended with a clean h2 half-close and the exchange continues on
    /// the request body alone.
    pub async fn accept(
        listener: &HttpListener,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        let (recv_read, send_write) = HttpTransport::accept(listener).await?;
        Self::from_exchange(recv_read, send_write, limits).await
    }

    /// Build a table reader over an accepted exchange's byte halves.
    /// The response direction carries no data in this role, so it is
    /// ended with a clean h2 half-close and the exchange continues on
    /// the request body alone.
    pub async fn from_exchange(
        recv_read: H2RecvRead,
        mut send_write: H2SendWrite,
        limits: Option<DecodeLimits>,
    ) -> io::Result<Self> {
        send_write.shutdown().await?;
        let stream = HttpByteStream::new(recv_read, BufferChunkSize::Http);
        let inner = TableReader::<Vec64<u8>>::new(
            stream,
            BufferChunkSize::Http.chunk_size(),
            IPCMessageProtocol::Stream,
            limits,
        );
        Ok(Self { inner })
    }
}

impl IPCTransportReader for HttpTableReader {
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

impl Stream for HttpTableReader {
    type Item = io::Result<Table>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();
        Pin::new(&mut me.inner).poll_next(cx)
    }
}

// ---------------------------------------------------------------------------
// Internals - URI parsing + h2 handshake + GET request dispatch
// ---------------------------------------------------------------------------

fn parse_get(url: &str) -> io::Result<Request<()>> {
    let uri: Uri = url
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    Request::get(uri)
        .body(())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

pub(crate) fn host_port(
    uri: &Uri,
    expected_scheme: &str,
    default_port: u16,
) -> io::Result<(String, u16)> {
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

/// Initial flow-control window the client advertises during the h2
/// handshake. This governs the server's send capacity to us, so it
/// directly bounds GET response body throughput. h2's default is
/// 64 KiB which forces a WINDOW_UPDATE round-trip every 64 KiB and
/// collapses throughput on multi-MiB Arrow streams; 8 MiB lets a
/// typical record batch land in one round-trip.
const INITIAL_WINDOW: u32 = 8 * 1024 * 1024;

/// Run the h2 handshake on `io`, send `req` with end_of_stream=true
/// (GET has no body), spawn the connection driver, and return the
/// response body. The response future is awaited inline so the caller
/// gets headers before receiving any body bytes.
async fn h2_send_get<T>(io: T, req: Request<()>) -> io::Result<h2::RecvStream>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin + 'static,
{
    let (mut send_request, connection) = h2::client::Builder::new()
        .initial_window_size(INITIAL_WINDOW)
        .initial_connection_window_size(INITIAL_WINDOW)
        .handshake::<_, bytes::Bytes>(io)
        .await
        .map_err(io::Error::other)?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!("h2 connection driver exited: {e}");
        }
    });
    let (response_fut, _send_stream) = send_request
        .send_request(req, true)
        .map_err(io::Error::other)?;
    let response = response_fut.await.map_err(io::Error::other)?;
    Ok(response.into_body())
}
