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

use std::io;
use std::str;

use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;

use crate::traits::transport::Transport;

/// Cap on upgrade response header size before the handshake is
/// rejected as malformed.
const MAX_HANDSHAKE_RESPONSE: usize = 16 * 1024;

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

    /// Bind the URL's authority and return the listener.
    pub async fn bind(url: &str) -> io::Result<TcpListener> {
        let (authority, _path) = split_url(url);
        TcpListener::bind(authority).await
    }

    /// Accept the next inbound connection, run the server upgrade
    /// handshake, and return the post-handshake halves.
    pub async fn accept(
        listener: &TcpListener,
    ) -> io::Result<(ReadHalf<TcpStream>, WriteHalf<TcpStream>)> {
        let (tcp, _) = listener.accept().await?;
        let ws_stream = accept_async(tcp).await.map_err(io::Error::other)?;
        Ok(tokio::io::split(ws_stream.into_inner()))
    }

    /// Accept the next inbound connection, terminate TLS with the
    /// supplied `rustls::ServerConfig`, run the server upgrade
    /// handshake over it, and return the post-handshake halves.
    #[cfg(feature = "tls")]
    pub async fn accept_tls(
        listener: &TcpListener,
        config: std::sync::Arc<tokio_rustls::rustls::ServerConfig>,
    ) -> io::Result<(
        ReadHalf<tokio_rustls::server::TlsStream<TcpStream>>,
        WriteHalf<tokio_rustls::server::TlsStream<TcpStream>>,
    )> {
        let (tcp, _) = listener.accept().await?;
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let tls = acceptor.accept(tcp).await?;
        let ws_stream = accept_async(tls).await.map_err(io::Error::other)?;
        Ok(tokio::io::split(ws_stream.into_inner()))
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
    type Listener = TcpListener;
    type Read = ReadHalf<TcpStream>;
    type Write = WriteHalf<TcpStream>;

    async fn connect(endpoint: &str) -> io::Result<(Self::Read, Self::Write)> {
        WebSocketTransport::connect(endpoint).await
    }

    async fn bind(endpoint: &str) -> io::Result<TcpListener> {
        WebSocketTransport::bind(endpoint).await
    }

    async fn accept(listener: &TcpListener) -> io::Result<(Self::Read, Self::Write)> {
        WebSocketTransport::accept(listener).await
    }
}
