// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Storage writers
//!
//! Write handles for the file writers, over a local file or, with the
//! `object_storage` feature, an object in object storage. Each writer
//! creates its handle from the path it is given, and the encoders work
//! against the standard `Write` and `Seek` traits, or `AsyncWrite` for the
//! async Arrow IPC writers.
//!
//! ## Handles
//! - [`StorageWriter`]: sequential writes, for the CSV, JSON, Parquet and
//!   Arrow IPC writers. An object uploads as it is written and completes on
//!   `flush`. Object storage writes are append-only and complete once,
//!   matching how the writers use their sink.
//! - [`AsyncStorageWriter`]: the async equivalent, for the async Arrow IPC
//!   writers. An object completes on shutdown.
//! - [`create_dir_all`]: directory creation for the chunked writers. Object
//!   storage has no directories, and creating an object prefix does nothing.
//!
//! Object storage requests from `StorageWriter` block the calling thread on
//! the caller's Tokio runtime. See [`crate::object_storage`] for the runtime
//! requirement.

use std::fs::{self, File};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
#[cfg(not(target_family = "wasm"))]
use std::pin::Pin;
#[cfg(not(target_family = "wasm"))]
use std::task::{Context, Poll};

#[cfg(not(target_family = "wasm"))]
use tokio::io::AsyncWrite;

#[cfg(feature = "object_storage")]
use crate::models::streams::object_store::ObjectWrite;
#[cfg(feature = "object_storage")]
use crate::object_storage::{ObjectUpload, object_url, resolve};

/// Sequential writer to a local file or an object.
///
/// Implements `Write` and `Seek`. For an object, `flush` completes the
/// object, and `Seek` reports the current position without moving it,
/// which is the form the Parquet writer uses to record offsets.
pub enum StorageWriter {
    /// A local file.
    File(File),
    /// A streaming upload to an object.
    #[cfg(feature = "object_storage")]
    Object(ObjectUpload),
}

impl StorageWriter {
    /// Create the file or object at `path`, replacing any existing one.
    ///
    /// An object URL makes no request until the first part fills or the
    /// writer is flushed.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[cfg(feature = "object_storage")]
        if let Some(url) = object_url(path.as_ref()) {
            return Ok(Self::Object(ObjectUpload::new(resolve(&url)?)?));
        }
        Ok(Self::File(File::create(path)?))
    }
}

impl Write for StorageWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.write(buf),
            #[cfg(feature = "object_storage")]
            Self::Object(upload) => {
                upload.write(buf)?;
                Ok(buf.len())
            }
        }
    }

    /// Flush a local file, or complete an object.
    ///
    /// Completing an object uploads its remaining bytes and makes it
    /// visible in the store. Later flushes do nothing, and later writes
    /// return an error.
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::File(file) => file.flush(),
            #[cfg(feature = "object_storage")]
            Self::Object(upload) => upload.complete().map(|_| ()),
        }
    }
}

impl Seek for StorageWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::File(file) => file.seek(pos),
            #[cfg(feature = "object_storage")]
            Self::Object(upload) => {
                let written = upload.written();
                let target = match pos {
                    SeekFrom::Start(offset) => offset as i128,
                    SeekFrom::End(delta) | SeekFrom::Current(delta) => {
                        written as i128 + delta as i128
                    }
                };
                match target == written as i128 && !upload.is_closed() {
                    true => Ok(written),
                    false => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "object storage writes are append-only",
                    )),
                }
            }
        }
    }
}

/// Async writer to a local file or an object.
///
/// Implements `AsyncWrite`. For an object, `poll_shutdown` completes the
/// object.
#[cfg(not(target_family = "wasm"))]
pub enum AsyncStorageWriter {
    /// A local file.
    File(tokio::fs::File),
    /// A buffered upload to an object.
    #[cfg(feature = "object_storage")]
    Object(ObjectWrite),
}

#[cfg(not(target_family = "wasm"))]
impl AsyncStorageWriter {
    /// Create the file or object at `path`, replacing any existing one.
    pub async fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[cfg(feature = "object_storage")]
        if let Some(url) = object_url(path.as_ref()) {
            return Ok(Self::Object(ObjectWrite::create(&url)?));
        }
        Ok(Self::File(tokio::fs::File::create(path).await?))
    }
}

#[cfg(not(target_family = "wasm"))]
impl AsyncWrite for AsyncStorageWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::File(file) => Pin::new(file).poll_write(cx, buf),
            #[cfg(feature = "object_storage")]
            Self::Object(object) => Pin::new(object).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::File(file) => Pin::new(file).poll_flush(cx),
            #[cfg(feature = "object_storage")]
            Self::Object(object) => Pin::new(object).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::File(file) => Pin::new(file).poll_shutdown(cx),
            #[cfg(feature = "object_storage")]
            Self::Object(object) => Pin::new(object).poll_shutdown(cx),
        }
    }
}

/// Create the directory `dir` and its parents. Does nothing for an object
/// prefix, since object storage has no directories.
pub fn create_dir_all<P: AsRef<Path>>(dir: P) -> io::Result<()> {
    #[cfg(feature = "object_storage")]
    if object_url(dir.as_ref()).is_some() {
        return Ok(());
    }
    fs::create_dir_all(dir)
}
