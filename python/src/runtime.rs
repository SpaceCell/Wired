// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Runtime
//!
//! The embedded Tokio runtime that drives the async readers and writers.
//! File-format paths use the synchronous readers and writers. With the
//! `object_storage` feature, file-format reads and writes enter the
//! runtime on the calling thread, where object URL requests wait on it
//! with `Handle::block_on`. These calls come from Python threads, which
//! are outside the runtime's executor.

use std::sync::OnceLock;

use pyo3::Python;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// The multi-threaded runtime, created on first use.
pub fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to initialise the Tokio runtime")
    })
}

/// Runs a future to completion on the embedded runtime, detached from the
/// Python interpreter so Python threads keep running while Rust waits on I/O.
pub fn block_on_py<F>(py: Python<'_>, fut: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    py.detach(|| runtime().block_on(fut))
}
