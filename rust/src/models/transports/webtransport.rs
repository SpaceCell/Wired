// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebTransport Transport
//!
//! Connection establishment for WebTransport in both peer roles.
//! WebTransport requires TLS, so each role takes its wtransport
//! configuration directly. Each established session opens one
//! bidirectional stream whose receive and send sides are the returned
//! halves.
//!
//! wtransport configurations are consumed when an endpoint is built,
//! so this wire keeps the connect, bind, accept shape as inherent
//! methods rather than through the by-reference
//! [`Transport`](crate::traits::transport::Transport)
//! contract.
//!
//! WebTransport streams reach the peer on first use, so a connecting
//! peer that only reads should shut down its write half to open the
//! stream.
//!
//! ## Listeners
//!
//! Each [`WebTransportListener`] is the endpoint of one URL path.
//! [`WebTransport::bind`] binds the URL's scheme and authority as a
//! wtransport endpoint held in the process, and listeners for other paths
//! on the same scheme and authority share that endpoint. `accept` returns
//! only the sessions requested for the listener's own path. Session
//! requests for a path that no listener in the process has bound are
//! rejected with status `404 Not Found`. URLs without a path refer to the
//! path `/`.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use wtransport::endpoint::SessionRequest;
use wtransport::endpoint::endpoint_side::Server;
use wtransport::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig};

use crate::models::transports::routing::{AcceptRouter, SocketKey, SocketRegistry};

/// Endpoints bound by [`WebTransport::bind`] in this process.
static SOCKETS: SocketRegistry<SharedSocket> = SocketRegistry::new();

/// Endpoint shared by the WebTransport listeners of one scheme and
/// authority.
struct SharedSocket {
    endpoint: Endpoint<Server>,
    router: AcceptRouter<SessionRequest>,
}

/// # WebTransportListener
///
/// ## Description
/// - Accepting endpoint for one `https://` URL, returned by
///   [`WebTransport::bind`].
/// - Accepts only the sessions requested for the URL's path. URLs without
///   a path refer to the path `/`.
/// - Listeners for different paths on the same scheme and authority share
///   one wtransport endpoint in the process.
///
/// ## Structure
/// - `socket`: the endpoint shared by the listeners of the URL's scheme and
///   authority, with the paths bound on it.
/// - `path`: the URL path this listener accepts.
///
/// ## Usage
/// - Pass to [`WebTransport::accept`].
/// - Hold the listener across sessions to keep its path bound.
///
/// ## Notes
/// - Binding a URL that is already bound in the process returns
///   `io::ErrorKind::AddrInUse`.
/// - Dropping the listener unbinds its path. Dropping the last listener on
///   an authority closes the endpoint.
/// - Session requests for a path that no listener has bound are rejected
///   with status `404 Not Found`.
///
/// ## Example
/// ```rust,no_run
/// # async fn run(prices_config: wtransport::ServerConfig,
/// #              trades_config: wtransport::ServerConfig) -> std::io::Result<()> {
/// use lightstream::models::transports::webtransport::WebTransport;
///
/// let prices = WebTransport::bind("https://127.0.0.1:4433/prices", prices_config)?;
/// let trades = WebTransport::bind("https://127.0.0.1:4433/trades", trades_config)?;
///
/// // Receives only the sessions requested for `/prices`.
/// let (recv, send) = WebTransport::accept(&prices).await?;
/// # drop((recv, send, trades));
/// # Ok(()) }
/// ```
pub struct WebTransportListener {
    socket: Arc<SharedSocket>,
    path: String,
}

impl WebTransportListener {
    /// Returns the local address of the bound endpoint.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.endpoint.local_addr()
    }
}

impl Drop for WebTransportListener {
    fn drop(&mut self) {
        self.socket.router.unbind(&self.path);
    }
}

/// WebTransport wire in the connect, bind, accept shape.
pub struct WebTransport;

impl WebTransport {
    /// Connect to a listening `https://` URL, open a bidirectional
    /// stream, and return its halves.
    pub async fn connect(
        url: &str,
        config: ClientConfig,
    ) -> io::Result<(RecvStream, SendStream)> {
        let endpoint = Endpoint::client(config)?;
        let conn = endpoint.connect(url).await.map_err(io::Error::other)?;
        let (send, recv) = conn
            .open_bi()
            .await
            .map_err(io::Error::other)?
            .await
            .map_err(io::Error::other)?;
        // The session dies when every handle drops, so a task holds
        // one until the peers close it or it idles out.
        tokio::spawn(async move {
            let _endpoint = endpoint;
            conn.closed().await;
        });
        Ok((recv, send))
    }

    /// Bind an `https://` URL as an endpoint and return its listener.
    ///
    /// The first bind of a scheme and authority in the process builds the
    /// wtransport endpoint from `config`. Later binds of other paths on the
    /// same scheme and authority share that endpoint and its TLS identity,
    /// and their `config` is unused. The bind address of `config` is
    /// expected to match the URL's authority. For URLs with port 0, the
    /// endpoint binds the port `config` selects, reported by
    /// [`WebTransportListener::local_addr`]. Returns
    /// `io::ErrorKind::AddrInUse` for a URL that is already bound in the
    /// process.
    pub fn bind(url: &str, config: ServerConfig) -> io::Result<WebTransportListener> {
        let (scheme, rest) = url.split_once("://").unwrap_or(("https", url));
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let key = SocketKey::from_authority(scheme, authority)?;
        let socket = SOCKETS.open(key, || {
            let endpoint = Endpoint::server(config)?;
            let port = endpoint.local_addr()?.port();
            Ok((
                SharedSocket {
                    endpoint,
                    router: AcceptRouter::new(),
                },
                port,
            ))
        })?;
        socket.router.bind(path)?;
        Ok(WebTransportListener {
            socket,
            path: path.to_string(),
        })
    }

    /// Accept the next inbound session for the listener's path, take its
    /// first bidirectional stream, and return the stream's halves.
    ///
    /// Session requests that arrive for other bound paths on the endpoint
    /// are held for those paths' next `accept`.
    pub async fn accept(listener: &WebTransportListener) -> io::Result<(RecvStream, SendStream)> {
        let socket = &*listener.socket;
        let session_request = socket
            .router
            .accept(&listener.path, || async move {
                let request = socket
                    .endpoint
                    .accept()
                    .await
                    .await
                    .map_err(io::Error::other)?;
                let path = request.path().to_string();
                if !socket.router.is_bound(&path) {
                    request.not_found().await;
                    return Ok(None);
                }
                Ok(Some((path, request)))
            })
            .await?;
        let conn = session_request.accept().await.map_err(io::Error::other)?;
        let (send, recv) = conn.accept_bi().await.map_err(io::Error::other)?;
        // The session dies when every handle drops, so a task holds
        // one until the peers close it or it idles out.
        tokio::spawn(async move {
            conn.closed().await;
        });
        Ok((recv, send))
    }
}
