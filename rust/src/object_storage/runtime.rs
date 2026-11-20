// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage on the IO runtime
//!
//! Object storage requests run as Tokio tasks on the runtime from
//! [`crate::runtime::handle`]: the runtime current on the calling thread, or
//! the process runtime otherwise. The synchronous readers and
//! writers spawn each request with [`run_task`] and wait for its result, and
//! multipart uploads spawn their part uploads on the same runtime.
//!
//! Async code uses the async adapters in `models::streams::object_store`,
//! because a synchronous call from async code holds one of the runtime's
//! workers until its request completes.

use std::future::Future;
use std::io;
use std::sync::mpsc;

use tokio::runtime::Handle;

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
