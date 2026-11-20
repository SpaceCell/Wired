// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # HTTP/2 Transport
//!
//! Connection establishment for plaintext HTTP/2 (h2c) in both peer
//! roles. One connection carries one exchange. The connecting peer
//! issues a POST whose streaming request body is its write half and
//! whose streaming response body is its read half. The accepting peer
//! answers with response headers on accept, so the connecting peer's
//! wait for them returns before any body bytes flow, and the request
//! body it reads is its read half while the response body it sends is
//! its write half.
//!
//! GET semantics for cross-ecosystem endpoints stay on
//! [`HttpTableReader::get`](crate::models::readers::http::HttpTableReader::get).

use std::io;

use bytes::Bytes;
use http::{Request, Response, Uri};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};

use crate::models::readers::http::host_port;
use crate::models::streams::http::{H2RecvRead, H2SendWrite};
use crate::traits::transport::Transport;

/// Initial flow-control window each peer advertises during the h2
/// handshake. h2's 64 KiB default forces a WINDOW_UPDATE round-trip
/// every 64 KiB and collapses throughput on multi-MiB Arrow streams,
/// while 8 MiB lets a typical record batch land in one round-trip.
const INITIAL_WINDOW: u32 = 8 * 1024 * 1024;

/// HTTP/2 implementation of [`Transport`].
///
/// Establishes plaintext h2c exchanges. TLS entry points stay on the
/// table reader and writer types.
pub struct HttpTransport;

impl HttpTransport {
    /// Connect to a listening `http://` URL, open the exchange, and
    /// return its byte halves.
    pub async fn connect(url: &str) -> io::Result<(H2RecvRead, H2SendWrite)> {
        let uri: Uri = url
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let (host, port) = host_port(&uri, "http", 80)?;
        let tcp = TcpStream::connect((host.as_str(), port)).await?;
        connect_exchange(tcp, uri).await
    }

    /// Connect to a listening `https://` URL, upgrade the channel to
    /// TLS via the supplied `rustls::ClientConfig`, open the exchange,
    /// and return its byte halves. ALPN must be set to `h2` on the
    /// config. No default root store is bundled.
    #[cfg(feature = "tls")]
    pub async fn connect_tls(
        url: &str,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
    ) -> io::Result<(H2RecvRead, H2SendWrite)> {
        let uri: Uri = url
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let (host, port) = host_port(&uri, "https", 443)?;
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
        connect_exchange(tls, uri).await
    }

    /// Bind the URL's authority and return the listener.
    pub async fn bind(url: &str) -> io::Result<TcpListener> {
        let uri: Uri = url
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let scheme = uri.scheme_str().unwrap_or("http");
        let default_port = if scheme == "https" { 443 } else { 80 };
        let (host, port) = host_port(&uri, scheme, default_port)?;
        TcpListener::bind((host.as_str(), port)).await
    }

    /// Accept the next inbound connection, take its first request,
    /// answer with response headers, and return the request path with the
    /// exchange's byte halves.
    ///
    /// The path is returned as written in the request, for example `/feed`.
    pub async fn accept(listener: &TcpListener) -> io::Result<(String, H2RecvRead, H2SendWrite)> {
        let (tcp, _) = listener.accept().await?;
        accept_exchange(tcp).await
    }

    /// Accept the next inbound connection, terminate TLS with the
    /// supplied `rustls::ServerConfig`, take the exchange's first
    /// request, answer with response headers, and return the request path
    /// with its byte halves. The config advertises the `h2` application
    /// protocol.
    #[cfg(feature = "tls")]
    pub async fn accept_tls(
        listener: &TcpListener,
        config: std::sync::Arc<tokio_rustls::rustls::ServerConfig>,
    ) -> io::Result<(String, H2RecvRead, H2SendWrite)> {
        let (tcp, _) = listener.accept().await?;
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let tls = acceptor.accept(tcp).await?;
        accept_exchange(tls).await
    }
}

/// Run the h2 client handshake on an established stream, POST the
/// exchange's request, await response headers, and return the byte
/// halves.
async fn connect_exchange<T>(io_stream: T, uri: Uri) -> io::Result<(H2RecvRead, H2SendWrite)>
where
    T: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let (mut send_request, connection) = h2::client::Builder::new()
        .initial_window_size(INITIAL_WINDOW)
        .initial_connection_window_size(INITIAL_WINDOW)
        .handshake::<_, Bytes>(io_stream)
        .await
        .map_err(io::Error::other)?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!("h2 connection driver exited: {e}");
        }
    });

    let req = Request::post(uri)
        .body(())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let (response_fut, send_stream) = send_request
        .send_request(req, false)
        .map_err(io::Error::other)?;
    let response = response_fut.await.map_err(io::Error::other)?;
    let recv = response.into_body();
    Ok((H2RecvRead::new(recv), H2SendWrite::new(send_stream)))
}

/// Run the h2 server handshake on an established stream, take its
/// first request, answer with response headers, and return the request
/// path with the byte halves.
async fn accept_exchange<T>(io_stream: T) -> io::Result<(String, H2RecvRead, H2SendWrite)>
where
    T: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let mut conn = h2::server::Builder::new()
        .initial_window_size(INITIAL_WINDOW)
        .initial_connection_window_size(INITIAL_WINDOW)
        .handshake::<_, Bytes>(io_stream)
        .await
        .map_err(io::Error::other)?;

    let (req, mut respond) = conn
        .accept()
        .await
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before a request arrived",
            )
        })?
        .map_err(io::Error::other)?;
    let path = req.uri().path().to_string();
    let recv = req.into_body();

    let response = Response::builder()
        .status(200)
        .body(())
        .map_err(io::Error::other)?;
    let send_stream = respond
        .send_response(response, false)
        .map_err(io::Error::other)?;

    // The h2 Connection is the I/O driver for all in-flight
    // streams. Polling accept() in a loop drives the connection
    // until the peer closes.
    tokio::spawn(async move { while conn.accept().await.is_some() {} });

    Ok((path, H2RecvRead::new(recv), H2SendWrite::new(send_stream)))
}

impl Transport for HttpTransport {
    type Endpoint = str;
    type Listener = TcpListener;
    type Read = H2RecvRead;
    type Write = H2SendWrite;

    async fn connect(endpoint: &str) -> io::Result<(H2RecvRead, H2SendWrite)> {
        HttpTransport::connect(endpoint).await
    }

    async fn bind(endpoint: &str) -> io::Result<TcpListener> {
        HttpTransport::bind(endpoint).await
    }

    /// The `Transport` contract returns the halves only, because most
    /// transports don't have a request path. Callers that route by path use
    /// [`HttpTransport::accept`].
    async fn accept(listener: &TcpListener) -> io::Result<(H2RecvRead, H2SendWrite)> {
        let (_path, recv_read, send_write) = HttpTransport::accept(listener).await?;
        Ok((recv_read, send_write))
    }
}
