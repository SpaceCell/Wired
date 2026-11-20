// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Synchronous WebSocket Transport
//!
//! Blocking counterpart of [`WebSocketTransport`] over `std::net`, for
//! readers and writers that run on a dedicated thread without an async runtime.
//!
//! Connection establishment for plaintext WebSocket in both peer roles.
//! After the HTTP upgrade handshake, the socket's two halves are
//! `TcpStream` handles to the same connection, obtained through
//! `TcpStream::try_clone`. They exchange WebSocket frames through
//! [`SyncWsRead`] and [`SyncWsWrite`] in the layer above.
//!
//! The connecting peer runs the client upgrade with the request text and
//! response checks of the async transport, reading the response
//! byte-precisely, which leaves frames a fast server sends straight after its
//! `101 Switching Protocols` response in the socket for the frame parser.
//! The accepting peer runs the sync server handshake of `tungstenite`.
//!
//! ## Listeners
//!
//! Each [`SyncWebSocketListener`] is the endpoint of one URL path.
//! [`SyncWebSocketTransport::bind`] binds the URL's scheme and authority as a
//! socket held in the process, and listeners for other paths on the same
//! scheme and authority share that socket. `accept` returns only the
//! upgrades requested for the listener's own path. Upgrade requests for a
//! path that no listener in the process has bound are refused with status
//! `404 Not Found`. URLs without a path refer to the path `/`.
//!
//! [`WebSocketTransport`]: crate::models::transports::websocket::WebSocketTransport
//! [`SyncWsRead`]: crate::models::streams::sync_websocket::SyncWsRead
//! [`SyncWsWrite`]: crate::models::streams::sync_websocket::SyncWsWrite

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

use tokio_tungstenite::tungstenite::Error as WsError;
use tokio_tungstenite::tungstenite::{HandshakeError, WebSocket, accept_hdr};

use crate::models::transports::routing::{SocketKey, SocketRegistry, SyncAcceptRouter};
use crate::models::transports::websocket::{
    PathCallback, check_handshake_response, handshake_closed, handshake_request,
    handshake_response_complete, split_url,
};

/// Sockets bound by [`SyncWebSocketTransport::bind`] in this process.
static SOCKETS: SocketRegistry<SharedSocket> = SocketRegistry::new();

/// Socket shared by the sync WebSocket listeners of one scheme and
/// authority.
struct SharedSocket {
    listener: TcpListener,
    router: SyncAcceptRouter<TcpStream>,
}

/// # SyncWebSocketListener
///
/// ## Description
/// - Blocking accepting endpoint for one `ws://` URL, returned by
///   [`SyncWebSocketTransport::bind`].
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
/// - Pass to [`SyncWebSocketTransport::accept`], or to the `accept`
///   constructors of [`SyncWebSocketTableReader`] and
///   [`SyncWebSocketTableWriter`].
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
/// # fn run() -> std::io::Result<()> {
/// use lightstream::models::readers::sync_websocket::SyncWebSocketTableReader;
/// use lightstream::models::transports::sync_websocket::SyncWebSocketTransport;
///
/// let prices = SyncWebSocketTransport::bind("ws://127.0.0.1:9910/prices")?;
/// let trades = SyncWebSocketTransport::bind("ws://127.0.0.1:9910/trades")?;
///
/// // Receives only the connections made to `/prices`.
/// let reader = SyncWebSocketTableReader::accept(&prices, None)?;
/// # drop((reader, trades));
/// # Ok(()) }
/// ```
///
/// [`SyncWebSocketTableReader`]: crate::models::readers::sync_websocket::SyncWebSocketTableReader
/// [`SyncWebSocketTableWriter`]: crate::models::writers::sync_websocket::SyncWebSocketTableWriter
pub struct SyncWebSocketListener {
    socket: Arc<SharedSocket>,
    path: String,
}

impl SyncWebSocketListener {
    /// Returns the local address of the bound socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.listener.local_addr()
    }
}

impl Drop for SyncWebSocketListener {
    fn drop(&mut self) {
        self.socket.router.unbind(&self.path);
    }
}

/// Blocking WebSocket transport over `std::net`.
pub struct SyncWebSocketTransport;

impl SyncWebSocketTransport {
    /// Connect to a listening `ws://` URL, run the client upgrade
    /// handshake, and return the post-handshake read and write halves.
    ///
    /// Returns `io::ErrorKind::InvalidInput` for `wss://` URLs.
    pub fn connect(url: &str) -> io::Result<(TcpStream, TcpStream)> {
        if url.starts_with("wss://") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the sync WebSocket transport doesn't support wss endpoints",
            ));
        }
        let (authority, path) = split_url(url);
        let mut tcp = TcpStream::connect(authority)?;
        client_handshake(&mut tcp, authority, path)?;
        let write_half = tcp.try_clone()?;
        Ok((tcp, write_half))
    }

    /// Bind the URL as an endpoint and return its listener.
    ///
    /// The first bind of a scheme and authority in the process binds its
    /// socket, and later binds of other paths on it share that socket.
    /// URLs with port 0 bind a new socket on a free port, reported by
    /// [`SyncWebSocketListener::local_addr`]. Returns `io::ErrorKind::AddrInUse`
    /// for a URL that is already bound in the process.
    pub fn bind(url: &str) -> io::Result<SyncWebSocketListener> {
        let scheme = if url.starts_with("wss://") { "wss" } else { "ws" };
        let (authority, path) = split_url(url);
        let key = SocketKey::from_authority(scheme, authority)?;
        let socket = SOCKETS.open(key, || {
            let listener = TcpListener::bind(authority)?;
            let port = listener.local_addr()?.port();
            Ok((
                SharedSocket {
                    listener,
                    router: SyncAcceptRouter::new(),
                },
                port,
            ))
        })?;
        socket.router.bind(path)?;
        Ok(SyncWebSocketListener {
            socket,
            path: path.to_string(),
        })
    }

    /// Accept the next inbound connection to the listener's path, run the
    /// server upgrade handshake, and return the post-handshake read and
    /// write halves.
    ///
    /// Connections that arrive for other bound paths on the socket are
    /// upgraded and held for those paths' next `accept`.
    pub fn accept(listener: &SyncWebSocketListener) -> io::Result<(TcpStream, TcpStream)> {
        let socket = &*listener.socket;
        let tcp = socket.router.accept(&listener.path, || {
            let (tcp, _) = socket.listener.accept()?;
            upgrade(tcp, &socket.router)
        })?;
        let write_half = tcp.try_clone()?;
        Ok((tcp, write_half))
    }
}

/// Run the server upgrade handshake on an accepted connection, and return
/// the request path with the upgraded connection.
///
/// Returns `None` for a request to a path that no listener has bound,
/// after answering it with status `404 Not Found`.
fn upgrade(
    tcp: TcpStream,
    router: &SyncAcceptRouter<TcpStream>,
) -> io::Result<Option<(String, TcpStream)>> {
    let mut requested = None;
    let callback = PathCallback::new(|path| router.is_bound(path), &mut requested);
    let result = accept_hdr(tcp, callback)
        .map(WebSocket::into_inner)
        .map_err(|e| match e {
            HandshakeError::Failure(WsError::Http(_)) => None,
            e => Some(io::Error::other(e.to_string())),
        });
    match result {
        Ok(tcp) => Ok(requested.map(|path| (path, tcp))),
        Err(None) => Ok(None),
        Err(Some(e)) => Err(e),
    }
}

/// Run the client upgrade handshake on an established stream, reading
/// the response byte-precisely so early frames stay in the stream for
/// the frame parser.
fn client_handshake(stream: &mut TcpStream, authority: &str, path: &str) -> io::Result<()> {
    let (request, key) = handshake_request(authority, path);
    stream.write_all(request.as_bytes())?;

    let mut response = Vec::with_capacity(256);
    let mut byte = [0u8; 1];
    while !handshake_response_complete(&response)? {
        match stream.read(&mut byte) {
            Ok(0) => return Err(handshake_closed()),
            Ok(_) => response.push(byte[0]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    check_handshake_response(&response, &key)
}
