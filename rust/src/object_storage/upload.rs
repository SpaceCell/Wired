// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object uploads
//!
//! Streaming upload of one object from synchronous writes. Written bytes
//! are buffered up to one part, of the size registered for the object's URL
//! (see [`crate::object_storage::register_options`]). An object that never fills a part is
//! uploaded with a single request when it completes. Larger objects start a
//! multipart upload when the first part fills, and each part uploads on the
//! caller's Tokio runtime while writing continues, which bounds memory to
//! the parts in flight regardless of the object size.
//!
//! An object appears in the store only when [`ObjectUpload::complete`]
//! succeeds. A failed or abandoned upload is aborted, leaving no object.

use std::collections::VecDeque;
use std::io;
use std::mem;
use std::sync::Arc;

use object_store::{MultipartUpload, ObjectStoreExt, PutPayload, PutResult};
use tokio::runtime::Handle;
use tokio::task::JoinHandle;

use crate::object_storage::connection::ObjectLocation;
use crate::object_storage::error::io_error;
use crate::object_storage::runtime::{current_handle, run_task};

/// Upload of one object, driven by synchronous writes.
///
/// Held by `StorageWriter` for object URLs. See the module documentation
/// for the upload behaviour.
pub struct ObjectUpload {
    location: ObjectLocation,
    /// The caller's runtime, captured when the upload was created.
    runtime: Handle,
    /// Bytes not yet handed to a part upload.
    pending: Vec<u8>,
    /// Multipart upload, started when the first part fills.
    multipart: Option<Box<dyn MultipartUpload>>,
    /// Part uploads in flight, oldest first.
    parts: VecDeque<JoinHandle<object_store::Result<()>>>,
    /// Bytes written to the object so far.
    written: u64,
    /// Set once the object has been completed or the upload aborted.
    closed: bool,
}

impl ObjectUpload {
    /// Start an upload to `location` on the caller's runtime.
    ///
    /// ## Errors
    /// Returns an error when the calling thread is outside a Tokio runtime
    /// context.
    pub(crate) fn new(location: ObjectLocation) -> io::Result<Self> {
        Ok(Self {
            location,
            runtime: current_handle()?,
            pending: Vec::new(),
            multipart: None,
            parts: VecDeque::new(),
            written: 0,
            closed: false,
        })
    }

    /// Bytes written to the object so far.
    pub(crate) fn written(&self) -> u64 {
        self.written
    }

    /// Whether the object has been completed or the upload aborted.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed
    }

    /// Append `buf` to the object, uploading every part that fills.
    pub(crate) fn write(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "object '{}' is already complete. Object storage writes complete on the first flush",
                    self.location.path
                ),
            ));
        }
        self.pending.extend_from_slice(buf);
        self.written += buf.len() as u64;
        while self.pending.len() >= self.location.part_size {
            let rest = self.pending.split_off(self.location.part_size);
            let part = mem::replace(&mut self.pending, rest);
            if let Err(error) = self.send_part(part) {
                self.abort();
                return Err(error);
            }
        }
        Ok(())
    }

    /// Complete the object, uploading any remaining bytes.
    ///
    /// Completing an object that is already complete does nothing and
    /// returns `None`.
    pub(crate) fn complete(&mut self) -> io::Result<Option<PutResult>> {
        if self.closed {
            return Ok(None);
        }
        let result = match self.multipart.is_some() {
            false => {
                let payload = PutPayload::from(mem::take(&mut self.pending));
                let store = Arc::clone(&self.location.store);
                let path = self.location.path.clone();
                run_task(&self.runtime, async move { store.put(&path, payload).await })
                    .and_then(|result| result.map_err(io_error))
            }
            true => self.complete_multipart(),
        };
        match result {
            Ok(result) => {
                self.closed = true;
                Ok(Some(result))
            }
            Err(error) => {
                self.abort();
                Err(error)
            }
        }
    }

    /// Upload the remaining bytes as the last part, wait for every part,
    /// then complete the multipart upload.
    fn complete_multipart(&mut self) -> io::Result<PutResult> {
        if !self.pending.is_empty() {
            let last = mem::take(&mut self.pending);
            self.send_part(last)?;
        }
        while let Some(part) = self.parts.pop_front() {
            await_part(&self.runtime, part)?;
        }
        let mut upload = self.multipart.take().expect("multipart upload started");
        run_task(&self.runtime, async move {
            let result = upload.complete().await;
            // A failed completion leaves the uploaded parts in the store
            // until they are aborted.
            if result.is_err() {
                let _ = upload.abort().await;
            }
            result
        })
        .and_then(|result| result.map_err(io_error))
    }

    /// Hand `part` to the multipart upload, starting the upload first when
    /// this is the first part.
    fn send_part(&mut self, part: Vec<u8>) -> io::Result<()> {
        if self.multipart.is_none() {
            let store = Arc::clone(&self.location.store);
            let path = self.location.path.clone();
            let upload = run_task(&self.runtime, async move { store.put_multipart(&path).await })?
                .map_err(io_error)?;
            self.multipart = Some(upload);
        }
        // Wait for the oldest part once the registered number of parts is in
        // flight, which bounds memory and shares the upload bandwidth across
        // fewer requests.
        if self.parts.len() >= self.location.max_concurrent_parts {
            let oldest = self.parts.pop_front().expect("parts in flight");
            await_part(&self.runtime, oldest)?;
        }
        let upload = self
            .multipart
            .as_mut()
            .expect("multipart upload started")
            .put_part(PutPayload::from(part));
        self.parts.push_back(self.runtime.spawn(upload));
        Ok(())
    }

    /// Abort the upload, discarding any uploaded parts.
    ///
    /// The abort runs on the caller's runtime without blocking the caller,
    /// which is already reporting the failure that caused it.
    fn abort(&mut self) {
        self.closed = true;
        self.pending = Vec::new();
        for part in self.parts.drain(..) {
            part.abort();
        }
        if let Some(mut upload) = self.multipart.take() {
            self.runtime.spawn(async move {
                let _ = upload.abort().await;
            });
        }
    }
}

impl Drop for ObjectUpload {
    fn drop(&mut self) {
        if !self.closed {
            self.abort();
        }
    }
}

/// Wait for one part upload to finish.
fn await_part(runtime: &Handle, part: JoinHandle<object_store::Result<()>>) -> io::Result<()> {
    run_task(runtime, part)?
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
        .map_err(io_error)
}
