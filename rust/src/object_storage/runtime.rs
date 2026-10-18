// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage on the caller's runtime
//!
//! Object storage requests run as tasks on the caller's Tokio runtime. The
//! synchronous readers and writers spawn each request on the runtime and
//! wait for its result on a channel, in the same way as other synchronous
//! code waits for a worker thread. Multipart uploads spawn their part
//! uploads on the same handle.
//!
//! The synchronous readers and writers are called with object URLs from
//! threads outside the async executor, such as a thread that has entered
//! the runtime with `Handle::enter` or a `spawn_blocking` task. Waiting from
//! inside an async task holds one of the runtime's workers until the request
//! completes, and on a current-thread runtime the request never runs. Async
//! code uses the async adapters in `models::streams::object_store`.

use std::future::Future;
use std::io;
use std::sync::mpsc;

use tokio::runtime::Handle;

/// Handle to the caller's Tokio runtime.
///
/// ## Errors
/// Returns `ErrorKind::Other` when the calling thread is outside a Tokio
/// runtime context.
pub(crate) fn current_handle() -> io::Result<Handle> {
    Handle::try_current().map_err(|_| {
        io::Error::new(
            io::ErrorKind::Other,
            "object storage URLs require a Tokio runtime context on the calling thread",
        )
    })
}

/// Runs `future` as a task on `runtime` and returns its output to the
/// calling thread, which waits on a channel while the runtime's workers run
/// the task.
///
/// ## Errors
/// Returns `ErrorKind::Other` when the task ends without a result, for
/// example because it panicked or the runtime shut down.
pub(crate) fn run_task<F>(runtime: &Handle, future: F) -> io::Result<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    runtime.spawn(async move {
        let _ = result_tx.send(future.await);
    });
    result_rx.recv().map_err(|_| {
        io::Error::new(
            io::ErrorKind::Other,
            "object storage task ended without a result",
        )
    })
}
