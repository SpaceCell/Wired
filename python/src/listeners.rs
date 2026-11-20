// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Bound listeners
//!
//! Accepting reads and writes serve from listeners held for the life
//! of the process. The first accepting call on an endpoint binds it,
//! and every later call draws from the same listener, so a serving
//! loop keeps its endpoint continuously bound and clients queue in
//! the listener's backlog while another connection is being served.
//!
//! The ws, wss, http, https and wt endpoints are URLs, and each path is
//! its own endpoint. Their listeners are the lightstream listeners for
//! those transports, which share one socket between the paths of a
//! scheme and authority and return to each listener only the
//! connections made to its path.
//!
//! Binds run under the registry lock, so threads opening the same
//! endpoint at once resolve to one bound listener. The QUIC and
//! WebTransport listeners carry the TLS identity given at first bind,
//! and later accepting calls on the endpoint reuse that identity.
//! WebTransport paths on one authority share the identity of the
//! first path bound there.

use std::collections::HashMap;
#[cfg(unix)]
use std::fs;
use std::io;
use std::net;
use std::net::ToSocketAddrs;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use lightstream::models::transports::http::{HttpListener, HttpTransport};
use lightstream::models::transports::webtransport::{WebTransport, WebTransportListener};
use lightstream::models::transports::websocket::{WebSocketListener, WebSocketTransport};
use pyo3::PyResult;
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;

use crate::errors::{TransportError, to_py_err};
use crate::tls;
use lightstream::error::IoError;

/// One bound endpoint per scheme and address. The address is the
/// authority for TCP and QUIC, the socket path for UDS, and the
/// authority with the URL path for the URL endpoints. The scheme stays
/// in the key so listeners for different wires on the same port remain
/// distinct bindings rather than sharing one accept queue.
#[derive(Hash, PartialEq, Eq)]
enum ListenerKey {
    Tcp(String),
    Ws(String),
    Wss(String),
    Http(String),
    Https(String),
    #[cfg(unix)]
    Uds(PathBuf),
    Quic(String),
    Wt(String),
}

enum BoundListener {
    Tcp(Arc<TcpListener>),
    #[cfg(unix)]
    Uds(Arc<UnixListener>),
    Ws(Arc<WebSocketListener>),
    Http(Arc<HttpListener>),
    Quic(Arc<quinn::Endpoint>),
    Wt(Arc<WebTransportListener>),
}

static LISTENERS: OnceLock<Mutex<HashMap<ListenerKey, BoundListener>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<ListenerKey, BoundListener>> {
    LISTENERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns the process-held TCP listener for the address, binding it
/// on the first call. Call within the runtime context.
pub fn tcp(addr: &str) -> PyResult<Arc<TcpListener>> {
    shared_tcp(ListenerKey::Tcp(addr.to_string()), addr)
}

/// Returns the process-held listener for a `ws://` URL, binding it on
/// the first call.
pub async fn ws(url: &str) -> PyResult<Arc<WebSocketListener>> {
    shared_ws(ListenerKey::Ws(endpoint(url, "ws://")), url).await
}

/// Returns the process-held listener for a `wss://` URL, binding it on
/// the first call.
pub async fn wss(url: &str) -> PyResult<Arc<WebSocketListener>> {
    shared_ws(ListenerKey::Wss(endpoint(url, "wss://")), url).await
}

/// Returns the process-held listener for an `http://` URL, binding it
/// on the first call.
pub async fn http(url: &str) -> PyResult<Arc<HttpListener>> {
    shared_http(ListenerKey::Http(endpoint(url, "http://")), url).await
}

/// Returns the process-held listener for an `https://` URL, binding it
/// on the first call.
pub async fn https(url: &str) -> PyResult<Arc<HttpListener>> {
    shared_http(ListenerKey::Https(endpoint(url, "https://")), url).await
}

/// Returns the process-held UDS listener for the socket path, binding
/// it on the first call. A stale socket file left by an earlier
/// process is removed before the bind. Call within the runtime
/// context.
#[cfg(unix)]
pub fn uds(path: &Path) -> PyResult<Arc<UnixListener>> {
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Uds(listener)) = map.get(&ListenerKey::Uds(path.to_path_buf())) {
        return Ok(listener.clone());
    }
    let _ = fs::remove_file(path);
    let listener =
        Arc::new(UnixListener::bind(path).map_err(|e| to_py_err(IoError::Io(e)))?);
    map.insert(
        ListenerKey::Uds(path.to_path_buf()),
        BoundListener::Uds(listener.clone()),
    );
    Ok(listener)
}

/// Returns the process-held QUIC endpoint for the authority, binding
/// it with the given TLS identity on the first call. Call within the
/// runtime context.
pub fn quic(authority: &str, cert: &Path, key: &Path) -> PyResult<Arc<quinn::Endpoint>> {
    let key_entry = ListenerKey::Quic(authority.to_string());
    {
        let map = registry().lock().expect("listener registry lock");
        if let Some(BoundListener::Quic(listener)) = map.get(&key_entry) {
            return Ok(listener.clone());
        }
    }
    let config = tls::quic_server(cert, key)
        .map_err(|e| TransportError::new_err(e.to_string()))?;
    let addr = socket_addr(authority)?;
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Quic(listener)) = map.get(&key_entry) {
        return Ok(listener.clone());
    }
    let listener = Arc::new(
        quinn::Endpoint::server(config, addr).map_err(|e| to_py_err(IoError::Io(e)))?,
    );
    map.insert(key_entry, BoundListener::Quic(listener.clone()));
    Ok(listener)
}

/// Returns the process-held WebTransport listener for a `wt://` URL
/// without its scheme, binding it with the given TLS identity on the
/// first call. Call within the runtime context.
pub async fn wt(address: &str, cert: &Path, key: &Path) -> PyResult<Arc<WebTransportListener>> {
    let key_entry = ListenerKey::Wt(endpoint(address, ""));
    {
        let map = registry().lock().expect("listener registry lock");
        if let Some(BoundListener::Wt(listener)) = map.get(&key_entry) {
            return Ok(listener.clone());
        }
    }
    let authority = address.split('/').next().unwrap_or(address);
    let config = tls::wt_server(socket_addr(authority)?, cert, key)
        .await
        .map_err(|e| TransportError::new_err(e.to_string()))?;
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Wt(listener)) = map.get(&key_entry) {
        return Ok(listener.clone());
    }
    let listener = Arc::new(
        WebTransport::bind(&format!("https://{address}"), config)
            .map_err(|e| to_py_err(IoError::Io(e)))?,
    );
    map.insert(key_entry, BoundListener::Wt(listener.clone()));
    Ok(listener)
}

/// Looks the key up or binds the WebSocket URL, storing the listener
/// for the life of the process.
async fn shared_ws(key: ListenerKey, url: &str) -> PyResult<Arc<WebSocketListener>> {
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Ws(listener)) = map.get(&key) {
        return Ok(listener.clone());
    }
    let listener = Arc::new(
        WebSocketTransport::bind(url)
            .await
            .map_err(|e| to_py_err(IoError::Io(e)))?,
    );
    map.insert(key, BoundListener::Ws(listener.clone()));
    Ok(listener)
}

/// Looks the key up or binds the HTTP URL, storing the listener for the
/// life of the process.
async fn shared_http(key: ListenerKey, url: &str) -> PyResult<Arc<HttpListener>> {
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Http(listener)) = map.get(&key) {
        return Ok(listener.clone());
    }
    let listener = Arc::new(
        HttpTransport::bind(url)
            .await
            .map_err(|e| to_py_err(IoError::Io(e)))?,
    );
    map.insert(key, BoundListener::Http(listener.clone()));
    Ok(listener)
}

/// Looks the key up or binds the authority, storing the listener for
/// the life of the process.
fn shared_tcp(key: ListenerKey, authority: &str) -> PyResult<Arc<TcpListener>> {
    let mut map = registry().lock().expect("listener registry lock");
    if let Some(BoundListener::Tcp(listener)) = map.get(&key) {
        return Ok(listener.clone());
    }
    let bind = || -> io::Result<TcpListener> {
        let std_listener = net::TcpListener::bind(authority)?;
        std_listener.set_nonblocking(true)?;
        TcpListener::from_std(std_listener)
    };
    let listener = Arc::new(bind().map_err(|e| to_py_err(IoError::Io(e)))?);
    map.insert(key, BoundListener::Tcp(listener.clone()));
    Ok(listener)
}

/// Resolves an authority string to its first socket address.
pub fn socket_addr(authority: &str) -> PyResult<net::SocketAddr> {
    authority
        .to_socket_addrs()
        .map_err(|e| to_py_err(IoError::Io(e)))?
        .next()
        .ok_or_else(|| {
            TransportError::new_err(format!("endpoint '{authority}' resolves to no address"))
        })
}

/// Strips the scheme from a URL, leaving the authority and path that
/// identify the endpoint. A URL without a path has the path `/`.
fn endpoint(url: &str, scheme: &str) -> String {
    let rest = url.strip_prefix(scheme).unwrap_or(url);
    if rest.contains('/') {
        rest.to_string()
    } else {
        format!("{rest}/")
    }
}
