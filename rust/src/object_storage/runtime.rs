// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage on the caller's runtime
//!
//! Object storage requests run on the caller's Tokio runtime. The
//! synchronous readers and writers wait for their requests with
//! `Handle::block_on`, and multipart uploads spawn their part uploads on the
//! same handle.
//!
//! `Handle::block_on` blocks the calling thread, and it cannot be called
//! from inside an async task. The synchronous readers and writers are
//! therefore called with object URLs from outside the async executor: from a
//! `spawn_blocking` task, or from a thread that has entered the runtime with
//! `Handle::enter`. Called from inside an async task, they panic with Tokio's
//! "Cannot start a runtime from within a runtime" message.

use std::io;

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
