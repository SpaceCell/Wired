// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage errors
//!
//! The file readers and writers return `io::Result`, and object storage
//! failures are reported through the same type. The `object_store` error
//! remains available as the source of the `io::Error`, and its category is
//! mapped onto the closest `io::ErrorKind`, keeping a missing object and a
//! missing file distinguishable from other failures in the same way.

use std::io;

/// Convert an `object_store` error into an `io::Error` with the matching
/// kind.
pub(crate) fn io_error(error: object_store::Error) -> io::Error {
    let kind = match &error {
        object_store::Error::NotFound { .. } => io::ErrorKind::NotFound,
        object_store::Error::AlreadyExists { .. } => io::ErrorKind::AlreadyExists,
        object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => io::ErrorKind::PermissionDenied,
        object_store::Error::NotSupported { .. } | object_store::Error::NotImplemented { .. } => {
            io::ErrorKind::Unsupported
        }
        object_store::Error::InvalidPath { .. }
        | object_store::Error::UnknownConfigurationKey { .. } => io::ErrorKind::InvalidInput,
        _ => io::ErrorKind::Other,
    };
    io::Error::new(kind, error)
}
