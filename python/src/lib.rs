// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # lightstream
//!
//! Streaming Arrow I/O for Python. Reads and writes tables across files,
//! sockets, and network transports, with results as minarrow objects that
//! interoperate with the Arrow ecosystem through the PyCapsule protocol.

mod errors;
mod input;
mod listeners;
mod message;
mod output;
mod runtime;
mod tls;
mod uri;

#[cfg(feature = "object_storage")]
use std::collections::HashMap;

#[cfg(feature = "object_storage")]
use lightstream::error::IoError;
#[cfg(feature = "object_storage")]
use lightstream::object_storage::register_options;
use pyo3::prelude::*;

pub use errors::{FormatError, LightstreamError, ProtocolError, TransportError, to_py_err};
pub use input::PyDataStreamReader;
pub use message::PyMessage;
pub use output::PyDataStreamWriter;
pub use runtime::{block_on_py, runtime};

/// Opens a source for reading and returns a `Reader` that yields
/// `minarrow.Table` batches.
///
/// The URI scheme selects the medium and `protocol` selects the wire
/// protocol. The format comes from `format` or the path extension.
///
/// ## Wire sources
/// - `accept=True` makes the reader the accepting peer. The endpoint stays
///   bound for the life of the process, and the call blocks until a writer
///   connects.
/// - `parallel` sets the stream count on TCP sources, where the reader
///   accepts that many connections.
/// - The quic, wt, wss and https schemes use TLS. The accepting peer takes
///   `tls_cert` and `tls_key` PEM paths, and the connecting peer takes a
///   `tls_ca` PEM path.
///
/// ## Object storage
/// Builds with the `object_storage` feature accept object URLs, such as
/// `s3://bucket/key.parquet` or `memory:///key.parquet`, in place of file
/// paths. An object prefix with chunk objects immediately beneath it reads
/// as a chunk directory.
#[pyfunction]
#[pyo3(signature = (uri, *, format=None, protocol=None, parallel=0, accept=false, tls_cert=None, tls_key=None, tls_ca=None, mmap=None, out_of_core=false, base=None, delimiter=None, header=None, batch_size=None))]
fn read(
    py: Python<'_>,
    uri: &str,
    format: Option<&str>,
    protocol: Option<&str>,
    parallel: usize,
    accept: bool,
    tls_cert: Option<&str>,
    tls_key: Option<&str>,
    tls_ca: Option<&str>,
    mmap: Option<bool>,
    out_of_core: bool,
    base: Option<&str>,
    delimiter: Option<&str>,
    header: Option<bool>,
    batch_size: Option<usize>,
) -> PyResult<PyDataStreamReader> {
    let resolved = py.detach(|| {
        uri::resolve_source(
            uri,
            format,
            protocol,
            parallel,
            accept,
            tls_cert,
            tls_key,
            tls_ca,
            mmap,
            out_of_core,
            base,
            delimiter,
            header,
            batch_size,
        )
    })?;
    Ok(PyDataStreamReader::new(resolved))
}

/// Opens a target for writing and returns a `Writer` that accepts any
/// Arrow-compatible object.
///
/// The URI scheme selects the medium and `protocol` selects the wire
/// protocol. The format comes from `format` or the path extension.
///
/// ## Wire targets
/// - `accept=True` makes the writer the accepting peer. The endpoint stays
///   bound for the life of the process, and the call blocks until a reader
///   connects.
/// - The quic, wt, wss and https schemes use TLS. The accepting peer takes
///   `tls_cert` and `tls_key` PEM paths, and the connecting peer takes a
///   `tls_ca` PEM path.
///
/// ## Object storage
/// Builds with the `object_storage` feature accept object URLs in place of
/// file paths. Each object becomes visible in its store when the writer
/// closes.
#[pyfunction]
#[pyo3(signature = (uri, *, format=None, protocol=None, parallel=0, accept=false, tls_cert=None, tls_key=None, tls_ca=None, compression=None, delimiter=None, header=None))]
fn write(
    py: Python<'_>,
    uri: &str,
    format: Option<&str>,
    protocol: Option<&str>,
    parallel: usize,
    accept: bool,
    tls_cert: Option<&str>,
    tls_key: Option<&str>,
    tls_ca: Option<&str>,
    compression: Option<&str>,
    delimiter: Option<&str>,
    header: Option<bool>,
) -> PyResult<PyDataStreamWriter> {
    let resolved = py.detach(|| {
        uri::resolve_target(
            uri, format, protocol, parallel, accept, tls_cert, tls_key, tls_ca, compression,
            delimiter, header,
        )
    })?;
    Ok(PyDataStreamWriter::new(resolved))
}

/// Registers object storage options for every object URL beneath `prefix`,
/// for example `s3://telemetry` or `s3://telemetry/trades`.
///
/// Settings not registered here come from the backend's standard
/// environment variables, such as `AWS_REGION`.
///
/// ## Options
/// Keys and values are strings. The accepted keys are:
/// - backend configuration keys, for example `aws_region`, `aws_endpoint`
///   or `google_service_account`;
/// - HTTP client keys, for example `timeout`;
/// - the upload settings `part_size` and `max_concurrent_parts`.
///
/// ## Behaviour
/// - Registering a prefix again replaces its earlier options.
/// - Each URL uses the options of the longest registered prefix that
///   contains it.
///
/// ## Errors
/// - Raises `LightstreamError` when `prefix` is not an object URL or an
///   upload setting is invalid.
/// - Unknown backend keys raise `LightstreamError` when a URL beneath the
///   prefix is first opened.
#[cfg(feature = "object_storage")]
#[pyfunction(name = "register_options")]
fn register_object_options(prefix: &str, options: HashMap<String, String>) -> PyResult<()> {
    let options: Vec<(&str, &str)> = options
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    register_options(prefix, &options).map_err(|e| to_py_err(IoError::Io(e)))
}

/// Streaming Arrow I/O for Python.
#[pymodule]
#[pyo3(name = "lightstream")]
fn lightstream_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Streaming Arrow I/O for Python - files, sockets, and network transports with zero-copy minarrow interop.",
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    #[cfg(feature = "object_storage")]
    m.add_function(wrap_pyfunction!(register_object_options, m)?)?;
    m.add_class::<PyDataStreamReader>()?;
    m.add_class::<PyDataStreamWriter>()?;
    m.add_class::<PyMessage>()?;

    m.add("LightstreamError", m.py().get_type::<LightstreamError>())?;
    m.add("FormatError", m.py().get_type::<FormatError>())?;
    m.add("TransportError", m.py().get_type::<TransportError>())?;
    m.add("ProtocolError", m.py().get_type::<ProtocolError>())?;

    Ok(())
}
