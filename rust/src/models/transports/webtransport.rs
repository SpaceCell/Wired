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

use std::io;

use wtransport::endpoint::endpoint_side::Server;
use wtransport::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig};

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

    /// Bind a WebTransport endpoint and return its listener.
    pub fn bind(config: ServerConfig) -> io::Result<Endpoint<Server>> {
        Endpoint::server(config)
    }

    /// Accept the next inbound session, take its first bidirectional
    /// stream, and return the session's request path with the stream's
    /// halves.
    ///
    /// The path is returned as written in the session's CONNECT request, for
    /// example `/feed`.
    pub async fn accept(listener: &Endpoint<Server>) -> io::Result<(String, RecvStream, SendStream)> {
        let session_request = listener.accept().await.await.map_err(io::Error::other)?;
        let path = session_request.path().to_string();
        let conn = session_request.accept().await.map_err(io::Error::other)?;
        let (send, recv) = conn.accept_bi().await.map_err(io::Error::other)?;
        // The session dies when every handle drops, so a task holds
        // one until the peers close it or it idles out.
        tokio::spawn(async move {
            conn.closed().await;
        });
        Ok((path, recv, send))
    }
}
