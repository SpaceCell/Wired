// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage
//!
//! Remote object storage for the file readers and writers. With the
//! `object_storage` feature enabled, every reader and writer that takes a
//! file path also accepts an object URL in its place:
//!
//! ```rust,ignore
//! let mut writer = CsvWriter::to_path("s3://telemetry/trades/2026-10-14.csv")?;
//! writer.write_table(&table)?;
//! writer.flush()?;
//!
//! let reader = FileTableReader::open("gs://archive/trades.arrow")?;
//! ```
//!
//! ## Structure
//! - `connection.rs`: URL detection, store resolution and caching, and the
//!   per-prefix option registry.
//! - `runtime.rs`: access to the caller's Tokio runtime for object storage
//!   requests.
//! - `upload.rs`: the streaming upload behind `StorageWriter`.
//! - `error.rs`: conversion of store errors into `io::Error`.
//!
//! ## Backends
//!
//! | URL | Backend | Feature |
//! |---|---|---|
//! | `memory:///path` | In-process memory, shared across the process | `object_storage` |
//! | `s3://bucket/path`, `s3a://` | Amazon S3 and S3-compatible stores | `aws` |
//! | `gs://bucket/path` | Google Cloud Storage | `gcp` |
//! | `az://container/path`, `abfs://`, `abfss://`, `adl://`, `azure://` | Azure Blob Storage | `azure` |
//!
//! Paths without one of these schemes, including Windows drive paths, are
//! local file paths as before.
//!
//! ## Credentials
//! Each backend reads its standard environment variables (`AWS_*`,
//! `GOOGLE_*`, `AZURE_*`), falling back to instance or workload identity
//! where the platform provides one. Options for a bucket or prefix are set
//! with [`register_options`], and apply to every URL beneath that prefix.
//!
//! ## Runtime
//! Object storage I/O runs on the caller's Tokio runtime. The async adapters
//! in `models::streams::object_store` await their requests. The synchronous
//! readers and writers wait on their requests with `Handle::block_on`, and
//! are called with object URLs from outside the async executor: from a
//! `spawn_blocking` task, or from a thread that has entered the runtime with
//! `Handle::enter`. The parallel chunked reads and writes enter the caller's
//! runtime on each of their worker threads.
//!
//! ## HTTP versions
//! Requests use the `object_store` HTTP client, which is separate from the
//! HTTP/2 transport behind the `http` feature. The protocol is negotiated
//! per backend: S3 and Azure Blob serve HTTP/1.1, and GCS supports HTTP/2.
//! The `http1_only` and `http2_only` options force one protocol.

mod connection;
mod error;
mod runtime;
mod upload;

pub use connection::{ObjectLocation, register_options};
pub use upload::ObjectUpload;

pub(crate) use connection::{object_url, resolve};
pub(crate) use error::io_error;
pub(crate) use runtime::current_handle;
pub(crate) use upload::{MAX_CONCURRENT_PARTS, PART_BYTES};
