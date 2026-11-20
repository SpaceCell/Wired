// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Path routing
//!
//! Shared listener state for the transports with URL endpoints, i.e.
//! WebSocket, HTTP/2 and WebTransport.
//!
//! ## Context
//!
//! An endpoint such as `ws://host:9910/prices` is one path on the socket
//! bound to the URL's scheme and authority. Listeners for different paths
//! on one authority share that socket. Each transport holds a
//! [`SocketRegistry`] of its bound sockets for the life of the process,
//! keyed by [`SocketKey`], and each socket records its bound paths in an
//! [`AcceptRouter`] or, for the blocking WebSocket transport, a
//! [`SyncAcceptRouter`].
//!
//! ## Details
//!
//! - Accepting calls drive the shared socket themselves, without a
//!   background task or thread. One call at a time accepts from the socket
//!   and reads the request path of each connection. Connections for the
//!   calling listener's path are returned, connections for another bound
//!   path are queued until that path's next accept, and connections for an
//!   unbound path are refused.
//! - Registry entries are weak references. Sockets stay open while at least
//!   one listener holds them, and close with the last one.
//! - The async router waits through `tokio::sync::Notify` and the blocking
//!   router through `std::sync::Condvar`. Both keep their paths and queued
//!   connections in the same `PathQueues` state.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::pin::pin;
#[cfg(feature = "websocket")]
use std::sync::Condvar;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use tokio::sync::Notify;

/// Identifies a bound socket by URL scheme, host and port.
///
/// The scheme separates sockets for different schemes on one port, for
/// example `ws` and `wss`, into distinct bindings.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SocketKey {
    scheme: String,
    host: String,
    port: u16,
}

impl SocketKey {
    /// Builds the key for a scheme, host and port.
    pub(crate) fn new(scheme: &str, host: &str, port: u16) -> Self {
        Self {
            scheme: scheme.to_string(),
            host: host.to_string(),
            port,
        }
    }

    /// Builds the key for a scheme and an authority in `host:port` form.
    ///
    /// Returns `io::ErrorKind::InvalidInput` for an authority without a
    /// numeric port.
    #[cfg(any(feature = "websocket", feature = "webtransport"))]
    pub(crate) fn from_authority(scheme: &str, authority: &str) -> io::Result<Self> {
        let (host, port) = authority
            .rsplit_once(':')
            .and_then(|(host, port)| Some((host, port.parse::<u16>().ok()?)))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("endpoint authority '{authority}' requires a port"),
                )
            })?;
        Ok(Self::new(scheme, host, port))
    }
}

/// Process-held record of the sockets one transport has bound.
///
/// Entries are weak references. A socket stays open while a listener
/// holds it and closes when the last listener is dropped.
pub(crate) struct SocketRegistry<S> {
    sockets: OnceLock<Mutex<HashMap<SocketKey, Weak<S>>>>,
}

impl<S> SocketRegistry<S> {
    /// Constructs an empty registry for use in a `static`.
    pub(crate) const fn new() -> Self {
        Self {
            sockets: OnceLock::new(),
        }
    }

    /// Returns the open socket for `key`, opening it with `open` when the
    /// process holds none.
    ///
    /// `open` returns the socket with the port it bound. That port replaces
    /// port 0 in the recorded key, ensuring that later binds of the bound
    /// port share the socket.
    pub(crate) fn open(
        &self,
        key: SocketKey,
        open: impl FnOnce() -> io::Result<(S, u16)>,
    ) -> io::Result<Arc<S>> {
        let mut sockets = self
            .sockets
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        sockets.retain(|_, socket| socket.strong_count() > 0);
        if key.port != 0 {
            if let Some(socket) = sockets.get(&key).and_then(Weak::upgrade) {
                return Ok(socket);
            }
        }
        let (socket, port) = open()?;
        let socket = Arc::new(socket);
        sockets.insert(SocketKey { port, ..key }, Arc::downgrade(&socket));
        Ok(socket)
    }
}

/// Bound paths of one socket, with the connections accepted for each path
/// and not yet returned.
struct PathQueues<C> {
    queues: HashMap<String, VecDeque<C>>,
    /// Set while an accepting call is accepting from the socket.
    accepting: bool,
}

impl<C> PathQueues<C> {
    fn new() -> Self {
        Self {
            queues: HashMap::new(),
            accepting: false,
        }
    }

    /// Binds `path`, returning `io::ErrorKind::AddrInUse` when it is
    /// already bound.
    fn bind(&mut self, path: &str) -> io::Result<()> {
        if self.queues.contains_key(path) {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!("endpoint path '{path}' is already bound"),
            ));
        }
        self.queues.insert(path.to_string(), VecDeque::new());
        Ok(())
    }

    /// Queues a connection for `path`, or drops it when `path` is no longer
    /// bound.
    fn deliver(&mut self, path: &str, conn: C) {
        if let Some(queue) = self.queues.get_mut(path) {
            queue.push_back(conn);
        }
    }

    fn take(&mut self, path: &str) -> Option<C> {
        self.queues.get_mut(path)?.pop_front()
    }
}

/// Path router for a socket shared by async listeners.
pub(crate) struct AcceptRouter<C> {
    state: Mutex<PathQueues<C>>,
    /// Notified when a connection is queued or the accepting call returns.
    changed: Notify,
}

impl<C> AcceptRouter<C> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PathQueues::new()),
            changed: Notify::new(),
        }
    }

    /// Binds `path`, returning `io::ErrorKind::AddrInUse` when it is
    /// already bound.
    pub(crate) fn bind(&self, path: &str) -> io::Result<()> {
        self.state().bind(path)
    }

    /// Unbinds `path` and drops its queued connections.
    pub(crate) fn unbind(&self, path: &str) {
        self.state().queues.remove(path);
    }

    /// Reports whether a listener has bound `path`.
    pub(crate) fn is_bound(&self, path: &str) -> bool {
        self.state().queues.contains_key(path)
    }

    /// Returns the next connection for `path`.
    ///
    /// Queued connections are returned first. Otherwise the call accepts
    /// from the socket when no other call is accepting, and waits for a
    /// queued connection or a free socket when another call is. `next`
    /// accepts one connection from the socket and returns it with its
    /// request path, or `None` for a refused connection. Connections for
    /// other paths are queued for those paths, and accepting continues
    /// until a connection for `path` arrives. Errors from `next` are
    /// returned to the caller.
    pub(crate) async fn accept<F, Fut>(&self, path: &str, mut next: F) -> io::Result<C>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = io::Result<Option<(String, C)>>>,
    {
        loop {
            // Registering for the notification before checking the queue
            // ensures that a connection queued in between wakes this call.
            let mut changed = pin!(self.changed.notified());
            changed.as_mut().enable();
            let accepting = {
                let mut state = self.state();
                if let Some(conn) = state.take(path) {
                    return Ok(conn);
                }
                !std::mem::replace(&mut state.accepting, true)
            };
            if !accepting {
                changed.await;
                continue;
            }
            let _role = AcceptingRole { router: self };
            loop {
                let Some((conn_path, conn)) = next().await? else {
                    continue;
                };
                if conn_path == path {
                    return Ok(conn);
                }
                self.state().deliver(&conn_path, conn);
                self.changed.notify_waiters();
            }
        }
    }

    fn state(&self) -> MutexGuard<'_, PathQueues<C>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Accepting role of one call to [`AcceptRouter::accept`]. Dropping the
/// role on return, error or cancellation releases the socket to the
/// waiting calls.
struct AcceptingRole<'a, C> {
    router: &'a AcceptRouter<C>,
}

impl<C> Drop for AcceptingRole<'_, C> {
    fn drop(&mut self) {
        self.router.state().accepting = false;
        self.router.changed.notify_waiters();
    }
}

/// Path router for a socket shared by blocking listeners.
#[cfg(feature = "websocket")]
pub(crate) struct SyncAcceptRouter<C> {
    state: Mutex<PathQueues<C>>,
    /// Signalled when a connection is queued or the accepting call returns.
    changed: Condvar,
}

#[cfg(feature = "websocket")]
impl<C> SyncAcceptRouter<C> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PathQueues::new()),
            changed: Condvar::new(),
        }
    }

    /// Binds `path`, returning `io::ErrorKind::AddrInUse` when it is
    /// already bound.
    pub(crate) fn bind(&self, path: &str) -> io::Result<()> {
        self.state().bind(path)
    }

    /// Unbinds `path` and drops its queued connections.
    pub(crate) fn unbind(&self, path: &str) {
        self.state().queues.remove(path);
    }

    /// Reports whether a listener has bound `path`.
    pub(crate) fn is_bound(&self, path: &str) -> bool {
        self.state().queues.contains_key(path)
    }

    /// Returns the next connection for `path`, blocking the calling thread.
    ///
    /// Follows [`AcceptRouter::accept`], with `next` blocking until it has
    /// accepted one connection.
    pub(crate) fn accept<F>(&self, path: &str, mut next: F) -> io::Result<C>
    where
        F: FnMut() -> io::Result<Option<(String, C)>>,
    {
        {
            let mut state = self.state();
            loop {
                if let Some(conn) = state.take(path) {
                    return Ok(conn);
                }
                if !state.accepting {
                    state.accepting = true;
                    break;
                }
                state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
        let _role = SyncAcceptingRole { router: self };
        loop {
            let Some((conn_path, conn)) = next()? else {
                continue;
            };
            if conn_path == path {
                return Ok(conn);
            }
            self.state().deliver(&conn_path, conn);
            self.changed.notify_all();
        }
    }

    fn state(&self) -> MutexGuard<'_, PathQueues<C>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Accepting role of one call to [`SyncAcceptRouter::accept`]. Dropping
/// the role on return or error releases the socket to the waiting calls.
#[cfg(feature = "websocket")]
struct SyncAcceptingRole<'a, C> {
    router: &'a SyncAcceptRouter<C>,
}

#[cfg(feature = "websocket")]
impl<C> Drop for SyncAcceptingRole<'_, C> {
    fn drop(&mut self) {
        self.router.state().accepting = false;
        self.router.changed.notify_all();
    }
}
