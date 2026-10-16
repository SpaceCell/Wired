// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Input
//!
//! Table sources and the `Reader` they feed. `Source` splits between
//! file-format storage reads and protocol-framed byte transports.
//! `Protocol` picks the wire framing, with both arms running over every
//! transport, and each leaf variant wraps one lightstream end reader.

use std::fs::File;
use std::future::poll_fn;
use std::io::{BufReader, Stdin};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_core::Stream;
use lightstream::error::IoError;
use lightstream::models::frames::lightstream_message::LightstreamMessage;
use lightstream::models::readers::chunked::arrow::ChunkedArrowReader;
use lightstream::models::readers::lightstream::LightstreamReader;
use lightstream::models::readers::chunked::csv::ChunkedCsvReader;
use lightstream::models::readers::chunked::parquet::ChunkedParquetReader;
use lightstream::models::readers::csv::CsvReader;
use lightstream::models::readers::ipc::file_table::FileTableReader;
#[cfg(unix)]
use lightstream::models::readers::ipc::mmap_table::MmapTableReader;
use lightstream::models::readers::http::HttpTableReader;
use lightstream::models::readers::parallel::tcp::TcpParallelTableReader;
use lightstream::models::readers::json::JsonReader;
use lightstream::models::readers::object_storage::StorageReader;
use lightstream::models::readers::parquet::load_parquet_table;
use lightstream::models::readers::quic::QuicTableReader;
use lightstream::models::readers::stdio::StdinTableReader;
use lightstream::models::readers::tcp::TcpTableReader;
#[cfg(unix)]
use lightstream::models::readers::uds::UdsTableReader;
use lightstream::models::readers::websocket::WebSocketTableReader;
use lightstream::models::readers::webtransport::WebTransportTableReader;
use lightstream::traits::transport_reader::IPCTransportReader;
use minarrow::ffi::arrow_c_ffi::{
    ArrowArrayStream, RecordBatchProducer, export_record_batch_producer_stream,
};
use minarrow::ffi::schema::Schema;
use minarrow::{Field, SuperTable, Table};
use minarrow_pyo3::ffi::to_py::super_table_to_stream_capsule;
use minarrow_pyo3::ffi::to_rust::record_batch_to_rust;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{FormatError, LightstreamError, ProtocolError, to_py_err};
use crate::message::PyMessage;
use crate::runtime::runtime;

/// A resolved table source. File formats read storage directly with no
/// wire protocol involved, and stream sources read a protocol-framed
/// byte transport.
pub enum Source {
    File(FileIO),
    Stream(Protocol),
}

/// Wire protocol, agnostic over the transport beneath it, so both arms
/// run over every wire. Arrow reads raw Arrow IPC streaming, and
/// Lightstream frames TLV messages that proxy Arrow IPC, Protobuf, and
/// MessagePack payloads over one connection.
pub enum Protocol {
    Arrow(ArrowIO),
    Lightstream(LightstreamIO),
}

/// File-format readers. Routing between the variants happens during
/// resolution, from the open options and the path shape. The chunked
/// variants read `<base>-<index>` file sets written by the chunked
/// writers, one batch per chunk file.
pub enum FileIO {
    Ipc {
        reader: FileTableReader,
        cursor: usize,
    },
    #[cfg(unix)]
    IpcMmap {
        reader: MmapTableReader,
        cursor: usize,
    },
    IpcChunked {
        reader: ChunkedArrowReader,
    },
    /// Parquet holds one table per file, loaded on the first read.
    Parquet {
        path: PathBuf,
        done: bool,
    },
    ParquetChunked {
        reader: ChunkedParquetReader,
    },
    Csv {
        reader: CsvReader<StorageReader>,
    },
    /// CSV text streamed over the stdio wire, one batch per
    /// `batch_size` rows.
    CsvStdio {
        reader: CsvReader<BufReader<Stdin>>,
    },
    CsvChunked {
        reader: ChunkedCsvReader,
    },
    Json {
        reader: JsonReader<StorageReader>,
    },
}

/// Raw Arrow IPC protocol readers, one pre-composed reader per wire.
pub enum ArrowIO {
    Tcp(TcpTableReader),
    /// Merges `parallel=` concurrent connections accepted on a bound
    /// listener into one batch stream.
    TcpParallel(TcpParallelTableReader),
    Ws(WebSocketTableReader),
    Http(HttpTableReader),
    #[cfg(unix)]
    Uds(UdsTableReader),
    Quic(QuicTableReader),
    Wt(WebTransportTableReader),
    Stdio(StdinTableReader),
}

impl ArrowIO {
    /// Pulls the next table off the wire, driving the async reader to
    /// completion on the embedded runtime.
    fn read(&mut self) -> Result<Option<Table>, IoError> {
        let table = match self {
            ArrowIO::Tcp(reader) => runtime().block_on(reader.read_next()),
            ArrowIO::TcpParallel(reader) => {
                return runtime()
                    .block_on(poll_fn(|cx| Pin::new(&mut *reader).poll_next(cx)))
                    .transpose()
                    .map(|item| item.map(|(table, _seq)| table))
                    .map_err(IoError::Io);
            }
            ArrowIO::Ws(reader) => runtime().block_on(reader.read_next()),
            ArrowIO::Http(reader) => runtime().block_on(reader.read_next()),
            #[cfg(unix)]
            ArrowIO::Uds(reader) => runtime().block_on(reader.read_next()),
            ArrowIO::Quic(reader) => runtime().block_on(reader.read_next()),
            ArrowIO::Wt(reader) => runtime().block_on(reader.read_next()),
            ArrowIO::Stdio(reader) => runtime().block_on(reader.read_next()),
        };
        table.map_err(IoError::Io)
    }
}

/// Lightstream TLV protocol readers. One reader serves every wire,
/// since it frames any transport's byte stream.
pub enum LightstreamIO {
    Stream(LightstreamReader),
}

impl Source {
    /// Pulls the next frame, dispatching on the resolved arm. File and
    /// Arrow sources yield table frames only.
    fn read(&mut self) -> Result<Option<LightstreamMessage>, IoError> {
        match self {
            Source::File(file) => Ok(file
                .read()?
                .map(|table| LightstreamMessage::Table { tag: 0, table: table.into() })),
            Source::Stream(protocol) => protocol.read(),
        }
    }

    /// Pulls the next table frame, discarding opaque message frames, for
    /// the tabular paths.
    fn read_table(&mut self) -> Result<Option<Table>, IoError> {
        loop {
            match self.read()? {
                Some(frame) => {
                    if let Some(table) = frame.into_table() {
                        return Ok(Some(table));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// True when the source speaks the Lightstream protocol, whose frames
    /// surface to Python as `Message` objects.
    fn is_lightstream(&self) -> bool {
        matches!(self, Source::Stream(Protocol::Lightstream(_)))
    }
}

impl Protocol {
    /// Pulls the next frame, dispatching on the protocol arm. Arrow
    /// sources yield table frames only.
    fn read(&mut self) -> Result<Option<LightstreamMessage>, IoError> {
        match self {
            Protocol::Arrow(io) => Ok(io
                .read()?
                .map(|table| LightstreamMessage::Table { tag: 0, table: table.into() })),
            Protocol::Lightstream(io) => io.read(),
        }
    }
}

impl LightstreamIO {
    /// Pulls the next TLV frame, driving the async reader to completion
    /// on the embedded runtime.
    fn read(&mut self) -> Result<Option<LightstreamMessage>, IoError> {
        match self {
            LightstreamIO::Stream(reader) => runtime()
                .block_on(poll_fn(|cx| Pin::new(&mut *reader).poll_next(cx)))
                .transpose()
                .map_err(IoError::Io),
        }
    }
}

impl FileIO {
    /// Reads the next batch from the file, advancing the batch cursor for
    /// the indexed IPC readers and the chunk iterator for directories.
    fn read(&mut self) -> Result<Option<Table>, IoError> {
        match self {
            FileIO::Ipc { reader, cursor } => {
                if *cursor >= reader.num_batches() {
                    return Ok(None);
                }
                let table = reader.read_batch(*cursor)?;
                *cursor += 1;
                Ok(Some(table))
            }
            #[cfg(unix)]
            FileIO::IpcMmap { reader, cursor } => {
                if *cursor >= reader.num_batches() {
                    return Ok(None);
                }
                let table = reader.read_batch(*cursor)?;
                *cursor += 1;
                Ok(Some(table))
            }
            FileIO::IpcChunked { reader } => Ok(reader.next().transpose()?),
            FileIO::Parquet { path, done } => {
                if *done {
                    return Ok(None);
                }
                *done = true;
                let file = File::open(&path).map_err(IoError::Io)?;
                Ok(Some(load_parquet_table(file)?))
            }
            FileIO::ParquetChunked { reader } => reader.next().transpose(),
            FileIO::Csv { reader } => Ok(reader.next_batch()?),
            FileIO::CsvStdio { reader } => Ok(reader.next_batch()?),
            FileIO::CsvChunked { reader } => Ok(reader.next().transpose()?),
            FileIO::Json { reader } => Ok(reader.next_batch()?),
        }
    }
}

/// One or more batches exposed through the Arrow PyCapsule stream
/// protocol, so the minarrow module constructs its own objects from
/// lightstream output with zero-copy exchange.
#[pyclass(name = "BatchCapsuleStream")]
pub struct PyBatchCapsuleStream {
    batches: SuperTable,
}

#[pymethods]
impl PyBatchCapsuleStream {
    /// Arrow PyCapsule stream protocol entry point.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python<'_>,
        requested_schema: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        super_table_to_stream_capsule(&self.batches, py)
    }
}

impl PyBatchCapsuleStream {
    /// Builds a `minarrow.Table` from one batch via the capsule exchange.
    pub fn into_table(py: Python<'_>, table: Table) -> PyResult<Py<PyAny>> {
        let name = Some(table.name.clone());
        let batches = SuperTable::from_batches(vec![Arc::new(table)], name);
        let carrier = Py::new(py, Self { batches })?;
        let table_cls = py.import("minarrow")?.getattr("Table")?;
        Ok(table_cls.call_method1("from_arrow", (carrier,))?.unbind())
    }

    /// Builds a `minarrow.ChunkedTable` from the batches via the capsule
    /// exchange.
    pub fn into_tables(py: Python<'_>, batches: SuperTable) -> PyResult<Py<PyAny>> {
        let carrier = Py::new(py, Self { batches })?;
        let chunked_cls = py.import("minarrow")?.getattr("ChunkedTable")?;
        Ok(chunked_cls.call_method1("from_arrow", (carrier,))?.unbind())
    }
}

/// Capsule destructor for an unconsumed ArrowArrayStream, releasing the
/// stream and its producer.
unsafe extern "C" fn arrow_stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_array_stream";
        let ptr =
            pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowArrayStream;
        if !ptr.is_null() {
            let stream = &mut *ptr;
            if let Some(release) = stream.release {
                release(stream);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Streams tables from a resolved source.
///
/// Iterating yields one `minarrow.Table` per batch. `read_all` drains the
/// remaining batches into a single `minarrow.Table` when the source
/// produced one batch, or a `minarrow.ChunkedTable` when it produced
/// several.
#[pyclass(name = "Reader")]
pub struct PyDataStreamReader {
    inner: Mutex<Option<Source>>,
}

impl PyDataStreamReader {
    /// Wraps a resolved source for handing to Python.
    pub fn new(source: Source) -> Self {
        Self {
            inner: Mutex::new(Some(source)),
        }
    }

    /// Takes the source out of the reader, erroring when it is closed.
    fn take(&self) -> PyResult<Source> {
        self.inner
            .lock()
            .map_err(|_| LightstreamError::new_err("reader lock poisoned"))?
            .take()
            .ok_or_else(|| LightstreamError::new_err("reader is closed"))
    }

    /// Returns the source after a read so the next call continues from
    /// the same position.
    fn restore(&self, source: Source) -> PyResult<()> {
        *self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("reader lock poisoned"))? = Some(source);
        Ok(())
    }
}

#[pymethods]
impl PyDataStreamReader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next item, detaching from the Python interpreter for the
    /// blocking read. File and Arrow sources yield `minarrow.Table`
    /// batches, and Lightstream sources yield `Message` frames.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut source = self.take()?;
        let lightstream = source.is_lightstream();
        let (source, result) = py.detach(move || {
            let result = source.read();
            (source, result)
        });
        self.restore(source)?;
        match result.map_err(to_py_err)? {
            Some(frame) if lightstream => Ok(Some(Py::new(py, PyMessage::new(frame))?.into_any())),
            Some(frame) => match frame.into_table() {
                Some(table) => Ok(Some(PyBatchCapsuleStream::into_table(py, table)?)),
                None => Err(LightstreamError::new_err(
                    "unexpected message frame from a table source",
                )),
            },
            None => Ok(None),
        }
    }

    /// Drains the remaining batches. One batch returns a `minarrow.Table`,
    /// several return a `minarrow.ChunkedTable`, and a drained source
    /// returns an empty `minarrow.Table`. Opaque message frames on
    /// Lightstream sources are discarded, since the result is tabular.
    fn read_all(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut source = self.take()?;
        let (source, result) = py.detach(move || {
            let mut batches: Vec<Arc<Table>> = Vec::new();
            let result = loop {
                match source.read_table() {
                    Ok(Some(table)) => batches.push(Arc::new(table)),
                    Ok(None) => break Ok(batches),
                    Err(e) => break Err(e),
                }
            };
            (source, result)
        });
        self.restore(source)?;
        let mut batches = result.map_err(to_py_err)?;
        match batches.len() {
            0 => {
                let table_cls = py.import("minarrow")?.getattr("Table")?;
                Ok(table_cls.call1((PyDict::new(py),))?.unbind())
            }
            1 => {
                let table = Arc::try_unwrap(batches.remove(0)).unwrap_or_else(|t| (*t).clone());
                PyBatchCapsuleStream::into_table(py, table)
            }
            _ => {
                let name = Some(batches[0].name.clone());
                let batches = SuperTable::from_batches(batches, name);
                PyBatchCapsuleStream::into_tables(py, batches)
            }
        }
    }

    /// Arrow PyCapsule stream protocol. Exporting hands the source to the
    /// consumer as a pull-driven stream, so batches decode only as the
    /// consumer requests them. The export consumes the reader, and further
    /// reads raise `LightstreamError`.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        &self,
        py: Python<'_>,
        requested_schema: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = requested_schema;
        let mut source = self.take()?;

        // The stream schema derives from the first batch, pulled ahead of
        // the export since Arrow stream consumers request the schema
        // before the first batch.
        let Some(first) = source.read_table().map_err(to_py_err)? else {
            return Err(LightstreamError::new_err(
                "cannot export an empty source through the Arrow stream protocol",
            ));
        };
        let fields: Vec<Field> = first.cols.iter().map(|fa| (*fa.field).clone()).collect();

        let mut pending = Some(first);
        let producer: RecordBatchProducer = Box::new(move || {
            let table = match pending.take() {
                Some(table) => Some(table),
                None => source.read_table().map_err(|e| e.to_string())?,
            };
            Ok(table.map(|table| {
                table
                    .cols
                    .iter()
                    .map(|fa| {
                        (
                            Arc::new(fa.array.clone()),
                            Schema::from(vec![(*fa.field).clone()]),
                        )
                    })
                    .collect()
            }))
        });

        let stream = export_record_batch_producer_stream(producer, fields, None);
        let stream_ptr = Box::into_raw(stream);

        let name = c"arrow_array_stream";
        let capsule = unsafe {
            let cap = pyo3::ffi::PyCapsule_New(
                stream_ptr as *mut std::ffi::c_void,
                name.as_ptr(),
                Some(arrow_stream_capsule_destructor),
            );
            if cap.is_null() {
                let stream = &mut *stream_ptr;
                if let Some(release) = stream.release {
                    release(stream_ptr);
                }
                let _ = Box::from_raw(stream_ptr);
                return Err(LightstreamError::new_err(
                    "failed to create the Arrow stream capsule",
                ));
            }
            Bound::from_owned_ptr(py, cap)
        };
        Ok(capsule.unbind())
    }

    /// Registers a table type on a Lightstream protocol source. The
    /// schema comes from any Arrow-compatible object carrying at least
    /// one batch, such as a representative table. Returns the assigned
    /// type tag. Register the same types in the same order as the
    /// producing writer, since tags assign in registration order.
    fn register_table(&self, name: &str, schema: &Bound<'_, PyAny>) -> PyResult<u8> {
        let table = record_batch_to_rust(schema)
            .map_err(|e| FormatError::new_err(format!("expected an Arrow-compatible object: {}", e)))?;
        if table.cols.is_empty() {
            return Err(FormatError::new_err(
                "the schema object carries no columns, pass an object with at least one batch",
            ));
        }
        let fields: Vec<Field> = table.cols.iter().map(|col| (*col.field).clone()).collect();
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("reader lock poisoned"))?;
        match guard.as_mut() {
            Some(Source::Stream(Protocol::Lightstream(LightstreamIO::Stream(reader)))) => {
                Ok(reader.register_table(name, fields))
            }
            Some(_) => Err(ProtocolError::new_err(
                "register_table applies to lightstream protocol sources",
            )),
            None => Err(LightstreamError::new_err("reader is closed")),
        }
    }

    /// Registers an opaque message type on a Lightstream protocol source.
    /// Returns the assigned type tag.
    fn register_message(&self, name: &str) -> PyResult<u8> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("reader lock poisoned"))?;
        match guard.as_mut() {
            Some(Source::Stream(Protocol::Lightstream(LightstreamIO::Stream(reader)))) => {
                Ok(reader.register_message(name))
            }
            Some(_) => Err(ProtocolError::new_err(
                "register_message applies to lightstream protocol sources",
            )),
            None => Err(LightstreamError::new_err("reader is closed")),
        }
    }

    /// Releases the source. Further reads raise `LightstreamError`.
    fn close(&self) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|_| LightstreamError::new_err("reader lock poisoned"))?
            .take();
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}
