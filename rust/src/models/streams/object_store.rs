// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage byte stream adapters
//!
//! Async read and write adapters over objects in object storage, for the
//! async Arrow IPC readers and writers.
//!
//! - [`ObjectByteStream`] reads an object as a chunked byte stream and an
//!   `AsyncRead`, ready for `TableReader`.
//! - [`ObjectWrite`] writes an object as an `AsyncWrite`, ready for
//!   `TableWriter`. Objects up to one part are uploaded with a single
//!   request, and larger objects stream as a multipart upload. The object
//!   is completed by `poll_shutdown`.
//!
//! Both adapters use the object storage HTTP client, which is separate
//! from the HTTP/2 transport in `streams::http`. See
//! [`crate::object_storage`] for URL forms and credentials.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use object_store::ObjectStoreExt;
use object_store::buffered::{BufReader, BufWriter};
use tokio::io::AsyncWrite;

use crate::enums::BufferChunkSize;
use crate::models::streams::async_read::AsyncReadByteStream;
use crate::object_storage::{MAX_CONCURRENT_PARTS, PART_BYTES, io_error, resolve};

/// Chunked byte stream over one object.
pub type ObjectByteStream = AsyncReadByteStream<BufReader>;

impl ObjectByteStream {
    /// Open the object at `url` for streaming reads.
    ///
    /// Requests the object's metadata, then reads its contents in ranged
    /// requests as the stream is polled.
    pub async fn open(url: &str, size: BufferChunkSize) -> io::Result<Self> {
        let location = resolve(url)?;
        let meta = location
            .store
            .head(&location.path)
            .await
            .map_err(io_error)?;
        Ok(AsyncReadByteStream::new(
            BufReader::new(location.store, &meta),
            size,
        ))
    }
}

/// Async writer for one object.
pub struct ObjectWrite {
    inner: BufWriter,
}

impl ObjectWrite {
    /// Create the object at `url`, replacing any existing one.
    ///
    /// No request is made until the first part fills or the writer shuts
    /// down.
    pub fn create(url: &str) -> io::Result<Self> {
        let location = resolve(url)?;
        let inner = BufWriter::with_capacity(location.store, location.path, PART_BYTES)
            .with_max_concurrency(MAX_CONCURRENT_PARTS);
        Ok(Self { inner })
    }
}

impl AsyncWrite for ObjectWrite {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    /// Complete the object, uploading any buffered bytes.
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}
