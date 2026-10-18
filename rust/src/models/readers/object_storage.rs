// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Storage readers
//!
//! Read handles for the file readers, over a local file or, with the
//! `object_storage` feature, an object in object storage. Each reader opens
//! its handle from the path it is given, and the decoders work against the
//! standard `Read`, `BufRead` and `Seek` traits the handles implement.
//!
//! ## Handles
//! - [`StorageReader`]: sequential reads with seeking, for the CSV, JSON and
//!   Parquet readers. An object is fetched whole when the reader opens.
//! - [`StorageSource`]: positional reads from a shared handle, for the
//!   Arrow IPC file reader. An object is read with one ranged request per
//!   read, which fetches the footer and then each record batch as it is
//!   read.
//! - [`list_dir`]: directory listing for the chunked readers. Object storage
//!   has no directories, and listing an object prefix returns the objects
//!   immediately beneath it.
//!
//! Object storage requests block the calling thread on the caller's Tokio
//! runtime. See [`crate::object_storage`] for the runtime requirement.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
#[cfg(feature = "arena")]
use std::mem::MaybeUninit;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
#[cfg(feature = "object_storage")]
use std::sync::Arc;

#[cfg(feature = "object_storage")]
use bytes::Bytes;
#[cfg(feature = "object_storage")]
use futures_util::TryStreamExt;
#[cfg(feature = "object_storage")]
use object_store::{ObjectMeta, ObjectStoreExt};

#[cfg(feature = "object_storage")]
use crate::object_storage::{
    ObjectLocation, current_handle, io_error, object_url, resolve, run_task,
};

/// Sequential reader over a local file or a fetched object.
///
/// Implements `Read`, `BufRead` and `Seek`.
pub enum StorageReader {
    /// A buffered local file.
    File(BufReader<File>),
    /// The contents of an object, fetched when the reader opened, with the
    /// current read position.
    #[cfg(feature = "object_storage")]
    Object { bytes: Bytes, position: usize },
}

impl StorageReader {
    /// Open the file or object URL at `path` for reading.
    ///
    /// An object URL fetches the whole object before returning.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[cfg(feature = "object_storage")]
        if let Some(url) = object_url(path.as_ref()) {
            let ObjectLocation { store, path, .. } = resolve(&url)?;
            let bytes = run_task(&current_handle()?, async move {
                store.get(&path).await?.bytes().await
            })?
            .map_err(io_error)?;
            return Ok(Self::Object { bytes, position: 0 });
        }
        Ok(Self::File(BufReader::new(File::open(path)?)))
    }
}

impl Read for StorageReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            #[cfg(feature = "object_storage")]
            Self::Object { .. } => {
                let available = self.fill_buf()?;
                let n = available.len().min(buf.len());
                buf[..n].copy_from_slice(&available[..n]);
                self.consume(n);
                Ok(n)
            }
        }
    }
}

impl BufRead for StorageReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            Self::File(file) => file.fill_buf(),
            #[cfg(feature = "object_storage")]
            Self::Object { bytes, position } => Ok(&bytes[(*position).min(bytes.len())..]),
        }
    }

    fn consume(&mut self, amount: usize) {
        match self {
            Self::File(file) => file.consume(amount),
            #[cfg(feature = "object_storage")]
            Self::Object { position, .. } => *position += amount,
        }
    }
}

impl Seek for StorageReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::File(file) => file.seek(pos),
            #[cfg(feature = "object_storage")]
            Self::Object { bytes, position } => {
                let target = match pos {
                    SeekFrom::Start(offset) => Some(offset as i128),
                    SeekFrom::End(delta) => Some(bytes.len() as i128 + delta as i128),
                    SeekFrom::Current(delta) => Some(*position as i128 + delta as i128),
                }
                .filter(|target| *target >= 0)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "seek before the start of the object",
                    )
                })?;
                *position = target as usize;
                Ok(target as u64)
            }
        }
    }
}

/// Positional reader over a local file or an object.
///
/// Positional reads take `&self` and leave no read position behind, which
/// lets many reads share one handle across threads. Local files use
/// `pread(2)` on Unix and `seek_read` on Windows. Objects issue one ranged
/// request per read.
pub enum StorageSource {
    /// A local file.
    File(File),
    /// An object and its size, recorded when the source opened.
    #[cfg(feature = "object_storage")]
    Object { location: ObjectLocation, len: u64 },
}

impl StorageSource {
    /// Open the file or object URL at `path` for positional reads.
    ///
    /// An object URL requests the object's metadata to record its size.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[cfg(feature = "object_storage")]
        if let Some(url) = object_url(path.as_ref()) {
            let location = resolve(&url)?;
            let store = Arc::clone(&location.store);
            let object = location.path.clone();
            let meta = run_task(&current_handle()?, async move { store.head(&object).await })?
                .map_err(io_error)?;
            return Ok(Self::Object {
                location,
                len: meta.size,
            });
        }
        Ok(Self::File(File::open(path)?))
    }

    /// Size of the file or object in bytes.
    pub fn len(&self) -> io::Result<u64> {
        match self {
            Self::File(file) => Ok(file.metadata()?.len()),
            #[cfg(feature = "object_storage")]
            Self::Object { len, .. } => Ok(*len),
        }
    }

    /// Whether the file or object is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Read exactly `buf.len()` bytes starting at `offset`.
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        match self {
            Self::File(file) => read_file_at(file, buf, offset),
            #[cfg(feature = "object_storage")]
            Self::Object { location, .. } => {
                let range = offset..offset + buf.len() as u64;
                let store = Arc::clone(&location.store);
                let object = location.path.clone();
                let bytes = run_task(&current_handle()?, async move {
                    store.get_range(&object, range).await
                })?
                .map_err(io_error)?;
                if bytes.len() != buf.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "object ended before requested offset/length",
                    ));
                }
                buf.copy_from_slice(&bytes);
                Ok(())
            }
        }
    }

    /// Read exactly `buf.len()` bytes starting at `offset` into spare
    /// capacity, without first initialising it where the platform allows.
    #[cfg(feature = "arena")]
    pub fn read_at_uninit(&self, buf: &mut [MaybeUninit<u8>], offset: u64) -> io::Result<()> {
        match self {
            Self::File(file) => read_file_at_uninit(file, buf, offset),
            #[cfg(feature = "object_storage")]
            Self::Object { .. } => self.read_at(initialise(buf), offset),
        }
    }
}

/// Paths of the entries immediately inside the directory or object prefix
/// `dir`.
///
/// Object entries are returned as object URLs beneath `dir`, which open
/// with the readers in the same way as local paths.
pub fn list_dir<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    #[cfg(feature = "object_storage")]
    if let Some(url) = object_url(dir.as_ref()) {
        let ObjectLocation { store, path, .. } = resolve(&url)?;
        let listing = run_task(&current_handle()?, async move {
            store.list_with_delimiter(Some(&path)).await
        })?
        .map_err(io_error)?;
        let base = url.trim_end_matches('/');
        return Ok(listing
            .objects
            .iter()
            .filter_map(|meta| meta.location.filename())
            .map(|name| PathBuf::from(format!("{base}/{name}")))
            .collect());
    }
    fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect()
}

/// Paths of every file beneath the directory or object prefix `dir`,
/// including files in nested directories, in no particular order.
///
/// Object entries are returned as object URLs beneath `dir`, as for
/// [`list_dir`]. On object storage this is one listing of every object under
/// the prefix.
pub fn list_files<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    #[cfg(feature = "object_storage")]
    if let Some(url) = object_url(dir.as_ref()) {
        let location = resolve(&url)?;
        let store = Arc::clone(&location.store);
        let prefix = location.path.clone();
        let objects: Vec<ObjectMeta> = run_task(&current_handle()?, async move {
            store.list(Some(&prefix)).try_collect().await
        })?
        .map_err(io_error)?;
        let base = url.trim_end_matches('/');
        return Ok(objects
            .iter()
            .filter_map(|meta| meta.location.prefix_match(&location.path))
            .map(|parts| {
                let relative: Vec<String> = parts.map(|part| part.as_ref().to_string()).collect();
                PathBuf::from(format!("{base}/{}", relative.join("/")))
            })
            .collect());
    }
    let mut files = Vec::new();
    let mut pending = vec![dir.as_ref().to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

/// Read exactly `buf.len()` bytes starting at `offset` without touching
/// the file's seek position. Backed by `pread(2)` on Unix and a
/// `seek_read` loop on Windows so callers can issue many positional
/// reads against the same shared `&File` without serialising through
/// `&mut self` or reopening the file. The wrapper exists only because
/// `seek_read` on Windows can return short, so callers can't share a
/// single-call shape with Unix's `read_exact_at`.
#[cfg(unix)]
fn read_file_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    file.read_exact_at(buf, offset)
}

/// Read directly into spare capacity without first claiming that its bytes
/// are initialised. Unix `pread` accepts a raw output pointer, preserving the
/// arena fast path without a full-buffer zeroing pass.
#[cfg(all(unix, feature = "arena"))]
fn read_file_at_uninit(file: &File, buf: &mut [MaybeUninit<u8>], offset: u64) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    let mut filled = 0usize;
    while filled < buf.len() {
        let file_offset: libc::off_t = (offset + filled as u64).try_into().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "file offset exceeds off_t")
        })?;
        let n = unsafe {
            libc::pread(
                file.as_raw_fd(),
                buf.as_mut_ptr().add(filled).cast(),
                buf.len() - filled,
                file_offset,
            )
        };
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file ended before requested offset/length",
            ));
        }
        if n < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        filled += n as usize;
    }
    Ok(())
}

#[cfg(all(windows, feature = "arena"))]
fn read_file_at_uninit(file: &File, buf: &mut [MaybeUninit<u8>], offset: u64) -> io::Result<()> {
    // Windows FileExt has no spare-capacity API. Initialise once before
    // passing the region through its safe `&mut [u8]` interface.
    read_file_at(file, initialise(buf), offset)
}

#[cfg(windows)]
fn read_file_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut total = 0;
    while total < buf.len() {
        let n = file.seek_read(&mut buf[total..], offset + total as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file ended before requested offset/length",
            ));
        }
        total += n;
    }
    Ok(())
}

/// Positional file reads are available on Unix and Windows. Other targets,
/// such as WebAssembly in the browser, report `ErrorKind::Unsupported`.
#[cfg(not(any(unix, windows)))]
fn read_file_at(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "positional file reads are not supported on this target",
    ))
}

#[cfg(all(not(any(unix, windows)), feature = "arena"))]
fn read_file_at_uninit(_file: &File, _buf: &mut [MaybeUninit<u8>], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "positional file reads are not supported on this target",
    ))
}

/// Zero `buf` and return it as an initialised byte slice.
#[cfg(all(feature = "arena", any(windows, feature = "object_storage")))]
fn initialise(buf: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    for byte in &mut *buf {
        byte.write(0);
    }
    unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), buf.len()) }
}
