// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebSocket Transport
//!
//! Connection establishment for plaintext WebSocket in both peer
//! roles. After the HTTP upgrade handshake, the socket's raw halves
//! carry WebSocket frames for
//! [`WsRead`](crate::models::streams::websocket::WsRead) and
//! [`WsWrite`](crate::models::streams::websocket::WsWrite) to parse in
//! the layer above.
//!
//! The connecting peer runs the client upgrade itself and reads the
//! response headers byte-precisely, so frames a fast server sends
//! straight after its `101 Switching Protocols` response stay in the
//! socket for the frame parser. The accepting peer's upgrade has no
//! such hazard, as the connecting peer sends no frames until that
//! response reaches it.
//!
//! TLS-wrapped channels keep their `connect_tls` entry points on the
//! table reader and writer types.
//!
//! ## Listeners
//!
//! Each [`WebSocketListener`] is the endpoint of one URL path.
//! [`WebSocketTransport::bind`] binds the URL's scheme and authority as a
//! socket held in the process, and listeners for other paths on the same
//! scheme and authority share that socket. `accept` returns only the
//! upgrades requested for the listener's own path. Upgrade requests for a
//! path that no listener in the process has bound are refused with status
//! `404 Not Found`. URLs without a path refer to the path `/`.

use std::io;
use std::net::SocketAddr;
use std::str;
use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::Error as WsError;
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::handshake::server::{
    Callback, ErrorResponse, Request, Response,
};
use tokio_tungstenite::tungstenite::http::StatusCode;

use crate::models::transports::routing::{AcceptRouter, SocketKey, SocketRegistry};
use crate::traits::transport::Transport;

/// Cap on upgrade response header size before the handshake is
/// rejected as malformed.
const MAX_HANDSHAKE_RESPONSE: usize = 16 * 1024;

/// Sockets bound by [`WebSocketTransport::bind`] in this process.
static SOCKETS: SocketRegistry<SharedSocket> = SocketRegistry::new();

/// Socket shared by the WebSocket listeners of one scheme and authority.
struct SharedSocket {
    listener: TcpListener,
    router: AcceptRouter<Upgraded>,
}

/// Upgraded connection queued for the listener of its path.
enum Upgraded {
    Plain(TcpStream),
    #[cfg(feature = "tls")]
    Tls(Box<tokio_rustls::server::TlsStream<TcpStream>>),
}

/// # WebSocketListener
///
/// ## Description
/// - Accepting endpoint for one `ws://` or `wss://` URL, returned by
///   [`WebSocketTransport::bind`].
/// - Accepts only the upgrades requested for the URL's path. URLs without
///   a path refer to the path `/`.
/// - Listeners for different paths on the same scheme and authority share
///   one socket in the process.
///
/// ## Structure
/// - `socket`: the socket shared by the listeners of the URL's scheme and
///   authority, with the paths bound on it.
/// - `path`: the URL path this listener accepts.
///
/// ## Usage
/// - Pass to [`WebSocketTransport::accept`] or
///   [`WebSocketTransport::accept_tls`], or to the `accept` constructors of
///   [`WebSocketTableReader`] and [`WebSocketTableWriter`].
/// - Hold the listener across connections to keep its path bound.
///
/// ## Notes
/// - Binding a URL that is already bound in the process returns
///   `io::ErrorKind::AddrInUse`.
/// - Dropping the listener unbinds its path. Dropping the last listener on
///   an authority closes the socket.
/// - Upgrades for a path that no listener has bound are refused with
///   status `404 Not Found`.
///
/// ## Example
/// ```rust,no_run
/// # async fn run() -> std::io::Result<()> {
/// use lightstream::models::readers::websocket::WebSocketTableReader;
/// use lightstream::models::transports::websocket::WebSocketTransport;
///
/// let prices = WebSocketTransport::bind("ws://127.0.0.1:9910/prices").await?;
/// let trades = WebSocketTransport::bind("ws://127.0.0.1:9910/trades").await?;
///
/// // Receives only the connections made to `/prices`.
/// let reader = WebSocketTableReader::accept(&prices, None).await?;
/// # drop((reader, trades));
/// # Ok(()) }
/// ```
///
/// [`WebSocketTableReader`]: crate::models::readers::websocket::WebSocketTableReader
/// [`WebSocketTableWriter`]: crate::models::writers::websocket::WebSocketTableWriter
pub struct WebSocketListener {
    socket: Arc<SharedSocket>,
    path: String,
}

impl WebSocketListener {
    /// Returns the local address of the bound socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.listener.local_addr()
    }
}

impl Drop for WebSocketListener {
    fn drop(&mut self) {
        self.socket.router.unbind(&self.path);
    }
}

/// WebSocket implementation of [`Transport`].
pub struct WebSocketTransport;

impl WebSocketTransport {
    /// Connect to a listening `ws://` URL, run the client upgrade
    /// handshake, and return the post-handshake halves.
    pub async fn connect(
        url: &str,
    ) -> io::Result<(ReadHalf<TcpStream>, WriteHalf<TcpStream>)> {
        if url.starts_with("wss://") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wss endpoints go through connect_tls",
            ));
        }
        let (authority, path) = split_url(url);
        let mut tcp = TcpStream::connect(authority).await?;
        client_handshake(&mut tcp, authority, path).await?;
        Ok(tokio::io::split(tcp))
    }

    /// Connect to a listening `wss://` URL, upgrade the channel to TLS
    /// via the supplied `rustls::ClientConfig`, run the client upgrade
    /// handshake over it, and return the post-handshake halves. The
    /// caller controls verifier and root store through their config.
    #[cfg(feature = "tls")]
    pub async fn connect_tls(
        url: &str,
        config: std::sync::Arc<tokio_rustls::rustls::ClientConfig>,
    ) -> io::Result<(
        ReadHalf<tokio_rustls::client::TlsStream<TcpStream>>,
        WriteHalf<tokio_rustls::client::TlsStream<TcpStream>>,
    )> {
        let (authority, path) = split_url(url);
        let host = host_of(authority);
        let server_name = rustls_pki_types::ServerName::try_from(host.to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let tcp = TcpStream::connect(authority).await?;
        let connector = tokio_rustls::TlsConnector::from(config);
        let mut tls = connector.connect(server_name, tcp).await?;
        client_handshake(&mut tls, authority, path).await?;
        Ok(tokio::io::split(tls))
    }

    /// Bind the URL as an endpoint and return its listener.
    ///
    /// The first bind of a scheme and authority in the process binds its
    /// socket, and later binds of other paths on it share that socket.
    /// URLs with port 0 bind a new socket on a free port, reported by
    /// [`WebSocketListener::local_addr`]. Returns `io::ErrorKind::AddrInUse`
    /// for a URL that is already bound in the process.
    pub async fn bind(url: &str) -> io::Result<WebSocketListener> {
        let scheme = if url.starts_with("wss://") { "wss" } else { "ws" };
        let (authority, path) = split_url(url);
        let key = SocketKey::from_authority(scheme, authority)?;
        let socket = SOCKETS.open(key, || {
            let std_listener = std::net::TcpListener::bind(authority)?;
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
        Ok(WebSocketListener {
            socket,
            path: path.to_string(),
        })
    }

    /// Accept the next inbound connection to the listener's path, run the
    /// server upgrade handshake, and return the post-handshake halves.
    ///
    /// Connections that arrive for other bound paths on the socket are
    /// upgraded and held for those paths' next `accept`.
    pub async fn accept(
        listener: &WebSocketListener,
    ) -> io::Result<(ReadHalf<TcpStream>, WriteHalf<TcpStream>)> {
        let socket = &*listener.socket;
        let conn = socket
            .router
            .accept(&listener.path, || async move {
                let (tcp, _) = socket.listener.accept().await?;
                let upgraded = upgrade(tcp, &socket.router).await?;
                Ok(upgraded.map(|(path, tcp)| (path, Upgraded::Plain(tcp))))
            })
            .await?;
        match conn {
            Upgraded::Plain(tcp) => Ok(tokio::io::split(tcp)),
            #[cfg(feature = "tls")]
            Upgraded::Tls(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the connection for this path was accepted with TLS",
            )),
        }
    }

    /// Accept the next inbound connection to the listener's path, terminate
    /// TLS with the supplied `rustls::ServerConfig`, run the server upgrade
    /// handshake over it, and return the post-handshake halves.
    ///
    /// TLS is terminated before the request path is known. Endpoints that
    /// share an authority therefore share its TLS identity, and connections
    /// held for other bound paths have completed TLS with `config`.
    #[cfg(feature = "tls")]
    pub async fn accept_tls(
        listener: &WebSocketListener,
        config: std::sync::Arc<tokio_rustls::rustls::ServerConfig>,
    ) -> io::Result<(
        ReadHalf<tokio_rustls::server::TlsStream<TcpStream>>,
        WriteHalf<tokio_rustls::server::TlsStream<TcpStream>>,
    )> {
        let socket = &*listener.socket;
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let conn = socket
            .router
            .accept(&listener.path, || {
                let acceptor = acceptor.clone();
                async move {
                    let (tcp, _) = socket.listener.accept().await?;
                    let tls = acceptor.accept(tcp).await?;
                    let upgraded = upgrade(tls, &socket.router).await?;
                    Ok(upgraded.map(|(path, tls)| (path, Upgraded::Tls(Box::new(tls)))))
                }
            })
            .await?;
        match conn {
            Upgraded::Tls(tls) => Ok(tokio::io::split(*tls)),
            Upgraded::Plain(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the connection for this path was accepted without TLS",
            )),
        }
    }
}

/// Run the server upgrade handshake on an accepted stream, and return the
/// request path with the upgraded stream.
///
/// Returns `None` for a request to a path that no listener has bound,
/// after answering it with status `404 Not Found`.
async fn upgrade<S>(
    stream: S,
    router: &AcceptRouter<Upgraded>,
) -> io::Result<Option<(String, S)>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut requested = None;
    let callback = PathCallback::new(|path| router.is_bound(path), &mut requested);
    let result = accept_hdr_async(stream, callback).await;
    match result {
        Ok(ws_stream) => Ok(requested.map(|path| (path, ws_stream.into_inner()))),
        Err(WsError::Http(_)) => Ok(None),
        Err(e) => Err(io::Error::other(e)),
    }
}

/// Server upgrade callback that accepts a request for a bound path and
/// refuses any other path with status `404 Not Found`.
///
/// Records the path of an accepted request. Shared by the async and
/// blocking server handshakes.
pub(crate) struct PathCallback<'a, F> {
    is_bound: F,
    requested: &'a mut Option<String>,
}

impl<'a, F> PathCallback<'a, F>
where
    F: FnOnce(&str) -> bool,
{
    pub(crate) fn new(is_bound: F, requested: &'a mut Option<String>) -> Self {
        Self {
            is_bound,
            requested,
        }
    }
}

impl<F> Callback for PathCallback<'_, F>
where
    F: FnOnce(&str) -> bool,
{
    fn on_request(self, request: &Request, response: Response) -> Result<Response, ErrorResponse> {
        let path = request.uri().path();
        if !(self.is_bound)(path) {
            let mut refusal = ErrorResponse::new(None);
            *refusal.status_mut() = StatusCode::NOT_FOUND;
            return Err(refusal);
        }
        *self.requested = Some(path.to_string());
        Ok(response)
    }
}

/// Run the client upgrade handshake on an established stream, reading
/// the response byte-precisely so early frames stay in the stream for
/// the frame parser.
async fn client_handshake<S>(stream: &mut S, authority: &str, path: &str) -> io::Result<()>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    let (request, key) = handshake_request(authority, path);
    stream.write_all(request.as_bytes()).await?;

    // Read the response one byte at a time so nothing beyond the
    // header terminator is consumed. Frames a fast server sends
    // straight after its response stay in the stream for the frame
    // parser.
    let mut response = Vec::with_capacity(256);
    let mut byte = [0u8; 1];
    while !handshake_response_complete(&response)? {
        if stream.read(&mut byte).await? == 0 {
            return Err(handshake_closed());
        }
        response.push(byte[0]);
    }
    check_handshake_response(&response, &key)
}

/// Build the client upgrade request for `authority` and `path`, and return
/// it with the `Sec-WebSocket-Key` it sends.
///
/// Shared by the async and blocking client handshakes.
pub(crate) fn handshake_request(authority: &str, path: &str) -> (String, String) {
    let key = generate_key();
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {authority}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: {key}\r\n\
         Sec-WebSocket-Version: 13\r\n\r\n"
    );
    (request, key)
}

/// Report whether the upgrade response bytes read so far end with the
/// header terminator.
///
/// Returns `io::ErrorKind::InvalidData` once the response exceeds
/// [`MAX_HANDSHAKE_RESPONSE`] without a terminator.
pub(crate) fn handshake_response_complete(response: &[u8]) -> io::Result<bool> {
    if response.ends_with(b"\r\n\r\n") {
        return Ok(true);
    }
    if response.len() > MAX_HANDSHAKE_RESPONSE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "oversized WebSocket handshake response",
        ));
    }
    Ok(false)
}

/// Error for a connection closed before the upgrade response completed.
pub(crate) fn handshake_closed() -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "connection closed during the WebSocket handshake",
    )
}

/// Check the complete upgrade response against the request's `key`.
///
/// Returns `io::ErrorKind::ConnectionRefused` for a status other than
/// `101`, and `io::ErrorKind::InvalidData` for a missing or mismatched
/// `Sec-WebSocket-Accept` header.
pub(crate) fn check_handshake_response(response: &[u8], key: &str) -> io::Result<()> {
    let text =
        str::from_utf8(response).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let status_line = text.lines().next().unwrap_or_default();
    if !status_line.contains(" 101 ") {
        return Err(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            format!("WebSocket upgrade refused: {status_line}"),
        ));
    }
    let expected = derive_accept_key(key.as_bytes());
    let accepted = text.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.trim().eq_ignore_ascii_case("sec-websocket-accept") && value.trim() == expected
        })
    });
    if !accepted {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebSocket handshake response carried a mismatched Sec-WebSocket-Accept",
        ));
    }
    Ok(())
}

/// Split a `ws://` or `wss://` URL or bare authority string into its
/// authority and path sections.
pub(crate) fn split_url(url: &str) -> (&str, &str) {
    let rest = url
        .strip_prefix("ws://")
        .or_else(|| url.strip_prefix("wss://"))
        .unwrap_or(url);
    match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    }
}

/// The host section of an authority, without port or brackets.
#[cfg(feature = "tls")]
fn host_of(authority: &str) -> &str {
    let host = authority
        .rsplit_once(':')
        .map(|(host, _port)| host)
        .unwrap_or(authority);
    host.trim_start_matches('[').trim_end_matches(']')
}

impl Transport for WebSocketTransport {
    type Endpoint = str;
    type Listener = WebSocketListener;
    type Read = ReadHalf<TcpStream>;
    type Write = WriteHalf<TcpStream>;

    async fn connect(endpoint: &str) -> io::Result<(Self::Read, Self::Write)> {
        WebSocketTransport::connect(endpoint).await
    }

    async fn bind(endpoint: &str) -> io::Result<WebSocketListener> {
        WebSocketTransport::bind(endpoint).await
    }

    async fn accept(listener: &WebSocketListener) -> io::Result<(Self::Read, Self::Write)> {
        WebSocketTransport::accept(listener).await
    }
}
