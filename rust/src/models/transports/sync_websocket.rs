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
//! [`WebSocketTransport`]: crate::models::transports::websocket::WebSocketTransport
//! [`SyncWsRead`]: crate::models::streams::sync_websocket::SyncWsRead
//! [`SyncWsWrite`]: crate::models::streams::sync_websocket::SyncWsWrite

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};

use tokio_tungstenite::tungstenite::accept;

use crate::models::transports::websocket::{
    check_handshake_response, handshake_closed, handshake_request, handshake_response_complete,
    split_url,
};

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

    /// Bind the URL's authority and return the listener.
    pub fn bind(url: &str) -> io::Result<TcpListener> {
        let (authority, _path) = split_url(url);
        TcpListener::bind(authority)
    }

    /// Accept the next inbound connection, run the server upgrade
    /// handshake, and return the post-handshake read and write halves.
    pub fn accept(listener: &TcpListener) -> io::Result<(TcpStream, TcpStream)> {
        let (tcp, _) = listener.accept()?;
        let ws_stream = accept(tcp).map_err(|e| io::Error::other(e.to_string()))?;
        let tcp = ws_stream.into_inner();
        let write_half = tcp.try_clone()?;
        Ok((tcp, write_half))
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
