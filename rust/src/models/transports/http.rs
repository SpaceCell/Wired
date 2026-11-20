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
//!
//! ## Listeners
//!
//! Each [`HttpListener`] is the endpoint of one URL path.
//! [`HttpTransport::bind`] binds the URL's scheme and authority as a socket
//! held in the process, and listeners for other paths on the same scheme
//! and authority share that socket. `accept` returns only the exchanges
//! that open with a request for the listener's own path. Requests for a
//! path that no listener in the process has bound are answered with status
//! `404 Not Found`. URLs without a path refer to the path `/`.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use h2::server::SendResponse;
use http::{Request, Response, StatusCode, Uri};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};

use crate::models::readers::http::host_port;
use crate::models::streams::http::{H2RecvRead, H2SendWrite};
use crate::models::transports::routing::{AcceptRouter, SocketKey, SocketRegistry};
use crate::traits::transport::Transport;

/// Initial flow-control window each peer advertises during the h2
/// handshake. h2's 64 KiB default forces a WINDOW_UPDATE round-trip
/// every 64 KiB and collapses throughput on multi-MiB Arrow streams,
/// while 8 MiB lets a typical record batch land in one round-trip.
const INITIAL_WINDOW: u32 = 8 * 1024 * 1024;

/// First request of an accepted connection, with its response handle.
type PendingRequest = (Request<h2::RecvStream>, SendResponse<Bytes>);

/// Sockets bound by [`HttpTransport::bind`] in this process.
static SOCKETS: SocketRegistry<SharedSocket> = SocketRegistry::new();

/// Socket shared by the HTTP/2 listeners of one scheme and authority.
struct SharedSocket {
    listener: TcpListener,
    router: AcceptRouter<PendingRequest>,
}

/// # HttpListener
///
/// ## Description
/// - Accepting endpoint for one `http://` or `https://` URL, returned by
///   [`HttpTransport::bind`].
/// - Accepts only the exchanges that open with a request for the URL's
///   path. URLs without a path refer to the path `/`.
/// - Listeners for different paths on the same scheme and authority share
///   one socket in the process.
///
/// ## Structure
/// - `socket`: the socket shared by the listeners of the URL's scheme and
///   authority, with the paths bound on it.
/// - `path`: the URL path this listener accepts.
///
/// ## Usage
/// - Pass to [`HttpTransport::accept`] or [`HttpTransport::accept_tls`],
///   or to the `accept` constructors of [`HttpTableReader`] and
///   [`HttpTableWriter`].
/// - Hold the listener across connections to keep its path bound.
///
/// ## Notes
/// - Binding a URL that is already bound in the process returns
///   `io::ErrorKind::AddrInUse`.
/// - Dropping the listener unbinds its path. Dropping the last listener on
///   an authority closes the socket.
/// - Requests for a path that no listener has bound are answered with
///   status `404 Not Found`.
///
/// ## Example
/// ```rust,no_run
/// # async fn run() -> std::io::Result<()> {
/// use lightstream::models::readers::http::HttpTableReader;
/// use lightstream::models::transports::http::HttpTransport;
///
/// let prices = HttpTransport::bind("http://127.0.0.1:9910/prices").await?;
/// let trades = HttpTransport::bind("http://127.0.0.1:9910/trades").await?;
///
/// // Receives only the exchanges made to `/prices`.
/// let reader = HttpTableReader::accept(&prices, None).await?;
/// # drop((reader, trades));
/// # Ok(()) }
/// ```
///
/// [`HttpTableReader`]: crate::models::readers::http::HttpTableReader
/// [`HttpTableWriter`]: crate::models::writers::http::HttpTableWriter
pub struct HttpListener {
    socket: Arc<SharedSocket>,
    path: String,
}

impl HttpListener {
    /// Returns the local address of the bound socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.listener.local_addr()
    }
}

impl Drop for HttpListener {
    fn drop(&mut self) {
        self.socket.router.unbind(&self.path);
    }
}

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

    /// Bind the URL as an endpoint and return its listener.
    ///
    /// The first bind of a scheme and authority in the process binds its
    /// socket, and later binds of other paths on it share that socket.
    /// URLs with port 0 bind a new socket on a free port, reported by
    /// [`HttpListener::local_addr`]. Returns `io::ErrorKind::AddrInUse`
    /// for a URL that is already bound in the process.
    pub async fn bind(url: &str) -> io::Result<HttpListener> {
        let uri: Uri = url
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let scheme = uri.scheme_str().unwrap_or("http");
        let default_port = if scheme == "https" { 443 } else { 80 };
        let (host, port) = host_port(&uri, scheme, default_port)?;
        let path = match uri.path() {
            "" => "/",
            path => path,
        };
        let key = SocketKey::new(scheme, &host, port);
        let socket = SOCKETS.open(key, || {
            let std_listener = std::net::TcpListener::bind((host.as_str(), port))?;
            std_listener.set_nonblocking(true)?;
            let port = std_listener.local_addr()?.port();
            let listener = TcpListener::from_std(std_listener)?;
            Ok((
                SharedSocket {
                    listener,
                    router: AcceptRouter::new(),
                },
                port,
            ))
        })?;
        socket.router.bind(path)?;
        Ok(HttpListener {
            socket,
            path: path.to_string(),
        })
    }

    /// Accept the next inbound connection that opens with a request for the
    /// listener's path, answer with response headers, and return the
    /// exchange's byte halves.
    ///
    /// Requests that arrive for other bound paths on the socket are held,
    /// unanswered, for those paths' next `accept`.
    pub async fn accept(listener: &HttpListener) -> io::Result<(H2RecvRead, H2SendWrite)> {
        let socket = &*listener.socket;
        let pending = socket
            .router
            .accept(&listener.path, || async move {
                let (tcp, _) = socket.listener.accept().await?;
                accept_request(tcp, &socket.router).await
            })
            .await?;
        answer_request(pending)
    }

    /// Accept the next inbound connection, terminate TLS with the
    /// supplied `rustls::ServerConfig`, take the exchange's first
    /// request for the listener's path, answer with response headers,
    /// and return its byte halves. The config advertises the `h2`
    /// application protocol.
    ///
    /// TLS is terminated before the request path is known. Endpoints that
    /// share an authority therefore share its TLS identity, and requests
    /// held for other bound paths have completed TLS with `config`.
    #[cfg(feature = "tls")]
    pub async fn accept_tls(
        listener: &HttpListener,
        config: std::sync::Arc<tokio_rustls::rustls::ServerConfig>,
    ) -> io::Result<(H2RecvRead, H2SendWrite)> {
        let socket = &*listener.socket;
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let pending = socket
            .router
            .accept(&listener.path, || {
                let acceptor = acceptor.clone();
                async move {
                    let (tcp, _) = socket.listener.accept().await?;
                    let tls = acceptor.accept(tcp).await?;
                    accept_request(tls, &socket.router).await
                }
            })
            .await?;
        answer_request(pending)
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
/// first request, and return the request path with the request and its
/// response handle.
///
/// Returns `None` for a request to a path that no listener has bound,
/// after answering it with status `404 Not Found`.
async fn accept_request<T>(
    io_stream: T,
    router: &AcceptRouter<PendingRequest>,
) -> io::Result<Option<(String, PendingRequest)>>
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

    // The h2 Connection is the I/O driver for all in-flight
    // streams. Polling accept() in a loop drives the connection
    // until the peer closes.
    tokio::spawn(async move { while conn.accept().await.is_some() {} });

    let path = req.uri().path().to_string();
    if !router.is_bound(&path) {
        let refusal = Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(())
            .map_err(io::Error::other)?;
        respond
            .send_response(refusal, true)
            .map_err(io::Error::other)?;
        return Ok(None);
    }
    Ok(Some((path, (req, respond))))
}

/// Answer an accepted request with response headers and return the
/// exchange's byte halves.
fn answer_request(
    (req, mut respond): PendingRequest,
) -> io::Result<(H2RecvRead, H2SendWrite)> {
    let recv = req.into_body();
    let response = Response::builder()
        .status(200)
        .body(())
        .map_err(io::Error::other)?;
    let send_stream = respond
        .send_response(response, false)
        .map_err(io::Error::other)?;
    Ok((H2RecvRead::new(recv), H2SendWrite::new(send_stream)))
}

impl Transport for HttpTransport {
    type Endpoint = str;
    type Listener = HttpListener;
    type Read = H2RecvRead;
    type Write = H2SendWrite;

    async fn connect(endpoint: &str) -> io::Result<(H2RecvRead, H2SendWrite)> {
        HttpTransport::connect(endpoint).await
    }

    async fn bind(endpoint: &str) -> io::Result<HttpListener> {
        HttpTransport::bind(endpoint).await
    }

    async fn accept(listener: &HttpListener) -> io::Result<(H2RecvRead, H2SendWrite)> {
        HttpTransport::accept(listener).await
    }
}
