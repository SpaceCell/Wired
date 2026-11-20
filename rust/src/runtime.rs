// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Runtime
//!
//! The Tokio runtime that Lightstream's synchronous IO runs on.
//!
//! ## Context
//! Synchronous object storage reads and writes run their requests as Tokio
//! tasks and wait for the results. The tasks run on the runtime current on
//! the calling thread when there is one, and on the process runtime
//! otherwise. Lightstream and the application share the process runtime,
//! and one multi-thread runtime therefore performs the IO of both.
//!
//! ## Details
//! - Applications that build their own runtime, such as Lightning, install it
//!   with [`init_with`] before their first IO.
//! - Without an installed runtime, [`handle`] initialises a default
//!   multi-thread runtime on its first call, with its worker threads named
//!   `lightstream-io`.
//! - The process runtime lives for the lifetime of the process.
//! - WebAssembly targets have no default runtime, and their applications
//!   install one with [`init_with`].

use std::io;
use std::sync::{Mutex, OnceLock};

use tokio::runtime::{Handle, Runtime};

/// The process runtime, initialised once by `init_with` or `handle`.
static PROCESS_RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Serialises the initialisation of `PROCESS_RUNTIME` across threads,
/// ensuring that `init` runs once.
static INITIALISING: Mutex<()> = Mutex::new(());

/// Returns the handle of the process runtime, initialising it with `init`
/// when it does not exist.
///
/// `init` runs once in the process, and only when no runtime has been
/// installed. A runtime installed earlier, including the default runtime
/// from [`handle`], is returned unchanged and `init` does not run.
///
/// ## Errors
/// Returns the error from `init`.
pub fn init_with(init: impl FnOnce() -> io::Result<Runtime>) -> io::Result<Handle> {
    if let Some(runtime) = PROCESS_RUNTIME.get() {
        return Ok(runtime.handle().clone());
    }
    let _initialising = INITIALISING.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(runtime) = PROCESS_RUNTIME.get() {
        return Ok(runtime.handle().clone());
    }
    let runtime = init()?;
    let runtime = PROCESS_RUNTIME.get_or_init(|| runtime);
    Ok(runtime.handle().clone())
}

/// Returns the handle of the runtime that Lightstream's synchronous IO runs
/// on.
///
/// The runtime current on the calling thread is returned first, followed by
/// the process runtime. Without either, the default process runtime is
/// initialised.
///
/// ## Errors
/// Returns `ErrorKind::Other` when the default runtime fails to build, or on
/// WebAssembly targets when no runtime has been installed.
pub fn handle() -> io::Result<Handle> {
    if let Ok(current) = Handle::try_current() {
        return Ok(current);
    }
    init_with(default_runtime)
}

/// Builds the default multi-thread runtime.
#[cfg(not(target_family = "wasm"))]
fn default_runtime() -> io::Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .thread_name("lightstream-io")
        .enable_all()
        .build()
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn processes_install_one_runtime() {
        let first = init_with(default_runtime).unwrap();
        let second = init_with(|| panic!("the process runtime is already installed")).unwrap();
        assert_eq!(first.id(), second.id());
        assert_eq!(handle().unwrap().id(), first.id());
    }

    #[test]
    fn runtimes_current_on_the_calling_thread_come_first() {
        let current = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let _entered = current.enter();
        assert_eq!(handle().unwrap().id(), current.handle().id());
    }
}

/// WebAssembly targets have no default runtime.
#[cfg(target_family = "wasm")]
fn default_runtime() -> io::Result<Runtime> {
    Err(io::Error::new(
        io::ErrorKind::Other,
        "Lightstream IO on WebAssembly requires a runtime installed with `runtime::init_with`",
    ))
}
