// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Output
//!
//! Table targets and the `Writer` that feeds them. `Target` splits
//! between file-format storage writes and protocol-framed byte
//! transports. `Protocol` picks the wire framing, with both arms running
//! over every transport, and each leaf variant wraps one lightstream end
//! writer.

use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lightstream::compression::Compression;
use lightstream::enums::IPCMessageProtocol;
use lightstream::error::IoError;
use lightstream::models::encoders::csv::CsvEncodeOptions;
use lightstream::models::encoders::json::JsonEncodeOptions;
use lightstream::models::writers::csv::CsvWriter;
use lightstream::models::writers::ipc::sync_table::SyncTableWriter;
use lightstream::models::writers::json::JsonWriter;
use lightstream::models::writers::lightstream::LightstreamWriter;
use lightstream::models::writers::object_storage::StorageWriter;
use lightstream::models::writers::parallel::tcp::TcpParallelTableWriter;
use lightstream::models::writers::parquet::write_parquet_table;
use lightstream::models::writers::http::HttpTableWriter;
use lightstream::models::writers::stdio::StdoutTableWriter;
use lightstream::models::writers::tcp::TcpTableWriter;
#[cfg(unix)]
use lightstream::models::writers::uds::UdsTableWriter;
use lightstream::models::writers::websocket::WebSocketTableWriter;
use lightstream::traits::parallel_transport_writer::ParallelTransportWriter;
use lightstream::traits::transport_writer::IPCTransportWriter;
use lightstream::models::streams::http::{H2RecvRead, H2SendWrite};
use lightstream::models::transports::quic::QuicTransport;
use lightstream::models::transports::webtransport::WebTransport;
use lightstream::models::writers::quic::QuicTableWriter;
use lightstream::models::writers::webtransport::WebTransportTableWriter;
use minarrow::{Consolidate, Field, SuperTable, Table};
use minarrow_pyo3::ffi::to_rust::table_to_rust;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use tokio::io::{AsyncWrite, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedWriteHalf as TcpOwnedWriteHalf;
#[cfg(unix)]
use tokio::net::unix::OwnedWriteHalf as UdsOwnedWriteHalf;
use tokio_rustls::server::TlsStream as ServerTlsStream;

use crate::errors::{FormatError, LightstreamError, ProtocolError, to_py_err};
use crate::runtime::runtime;
use crate::tls;

/// The byte sink a Lightstream protocol writer frames, boxed so one
/// writer type serves every wire.
type ByteSink = Box<dyn AsyncWrite + Unpin + Send>;

/// A resolved table target. File formats write storage directly with no
/// wire protocol involved, and stream targets write a protocol-framed
/// byte transport.
pub enum Target {
    File(FileIO),
    Stream(Protocol),
}

/// Wire protocol, agnostic over the transport beneath it, so both arms
/// run over every wire. Arrow writes raw Arrow IPC streaming, and
/// Lightstream frames TLV messages that proxy Arrow IPC, Protobuf, and
/// MessagePack payloads over one connection.
pub enum Protocol {
    Arrow(ArrowIO),
    Lightstream(LightstreamIO),
}

/// File-format writers. The IPC writer streams, building its writer on
/// the first batch since the Arrow IPC schema header derives from that
/// batch's fields. Parquet, CSV, and JSON are document formats, so their
/// variants buffer batches and write the file once on `finish`.
pub enum FileIO {
    Ipc {
        path: PathBuf,
        compression: Option<Compression>,
        writer: Option<SyncTableWriter<StorageWriter>>,
    },
    Parquet {
        path: PathBuf,
        compression: Option<Compression>,
        batches: Vec<Arc<Table>>,
    },
    Csv {
        path: PathBuf,
        options: CsvEncodeOptions,
        batches: Vec<Arc<Table>>,
    },
    /// CSV text streamed over the stdio wire, one batch per write with
    /// the header on the first.
    CsvStdio {
        options: CsvEncodeOptions,
    },
    /// NDJSON text streamed over the stdio wire, one line per row.
    JsonStdio {
        options: JsonEncodeOptions,
    },
    Json {
        path: PathBuf,
        options: JsonEncodeOptions,
        batches: Vec<Arc<Table>>,
    },
}

/// How a wire target reaches its connection. A connecting target dials
/// the endpoint when the first batch arrives, and an accepting target
/// already holds the connection taken at open, whose halves the first
/// batch consumes to build the writer.
pub enum Link<E, H> {
    Connect(E),
    Accepted(Option<H>),
}

/// Connection details a QUIC target dials on the first batch. The
/// configuration is validated when the target resolves.
pub struct QuicConnect {
    pub addr: SocketAddr,
    pub server_name: String,
    pub config: quinn::ClientConfig,
}

/// Connection details a wss target dials on the first batch. The
/// configuration is validated when the target resolves.
pub struct WssConnect {
    pub url: String,
    pub config: Arc<rustls::ClientConfig>,
}

/// Connection details an https target dials on the first batch. The
/// configuration is validated when the target resolves.
pub struct HttpsConnect {
    pub url: String,
    pub config: Arc<rustls::ClientConfig>,
}

/// Connection details a WebTransport target dials on the first batch.
/// The root path stays here so each dial attempt rebuilds the
/// configuration, which wtransport consumes on use.
pub struct WtConnect {
    pub url: String,
    pub ca: PathBuf,
}

/// Raw Arrow IPC protocol writers, one pre-composed writer per wire.
/// Each wire builds its writer on the first batch, since the Arrow IPC
/// schema header derives from that batch's fields.
pub enum ArrowIO {
    Tcp {
        link: Link<String, TcpOwnedWriteHalf>,
        compression: Option<Compression>,
        writer: Option<TcpTableWriter>,
    },
    /// Fans batches out over `streams` concurrent connections to one
    /// accepting endpoint.
    TcpParallel {
        addr: String,
        streams: usize,
        compression: Option<Compression>,
        writer: Option<TcpParallelTableWriter>,
    },
    Ws {
        link: Link<String, (ReadHalf<TcpStream>, WriteHalf<TcpStream>)>,
        compression: Option<Compression>,
        writer: Option<WebSocketTableWriter>,
    },
    Wss {
        link: Link<WssConnect, (ReadHalf<ServerTlsStream<TcpStream>>, WriteHalf<ServerTlsStream<TcpStream>>)>,
        compression: Option<Compression>,
        writer: Option<WebSocketTableWriter>,
    },
    Http {
        link: Link<String, (H2RecvRead, H2SendWrite)>,
        compression: Option<Compression>,
        writer: Option<HttpTableWriter>,
    },
    Https {
        link: Link<HttpsConnect, (H2RecvRead, H2SendWrite)>,
        compression: Option<Compression>,
        writer: Option<HttpTableWriter>,
    },
    #[cfg(unix)]
    Uds {
        link: Link<PathBuf, UdsOwnedWriteHalf>,
        compression: Option<Compression>,
        writer: Option<UdsTableWriter>,
    },
    Quic {
        link: Link<QuicConnect, (quinn::RecvStream, quinn::SendStream)>,
        compression: Option<Compression>,
        writer: Option<QuicTableWriter>,
    },
    Wt {
        link: Link<WtConnect, (wtransport::RecvStream, wtransport::SendStream)>,
        compression: Option<Compression>,
        writer: Option<WebTransportTableWriter>,
    },
    Stdio {
        compression: Option<Compression>,
        writer: Option<StdoutTableWriter>,
    },
}

impl ArrowIO {
    /// Writes one table to the wire, connecting with the first batch's
    /// schema when no connection exists yet.
    fn write(&mut self, table: &Table) -> Result<(), IoError> {
        let fields: Vec<Field> = table.cols.iter().map(|col| (*col.field).clone()).collect();
        runtime()
            .block_on(async {
                match self {
                    ArrowIO::Tcp {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(addr) => {
                                    TcpTableWriter::connect(addr.as_str(), fields, *compression)
                                        .await?
                                }
                                Link::Accepted(half) => {
                                    let half =
                                        half.take().expect("accepted half present until first batch");
                                    TcpTableWriter::from_write_half(half, fields, *compression)?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::TcpParallel {
                        addr,
                        streams,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            let addr = addr
                                .parse()
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
                            *writer = Some(
                                TcpParallelTableWriter::connect(
                                    addr,
                                    *streams,
                                    fields,
                                    Vec::new(),
                                    *compression,
                                )
                                .await?,
                            );
                        }
                        writer
                            .as_mut()
                            .expect("connected above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Ws {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(url) => {
                                    WebSocketTableWriter::connect(
                                        url.as_str(),
                                        fields,
                                        *compression,
                                    )
                                    .await?
                                }
                                Link::Accepted(halves) => {
                                    let (read_half, write_half) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    WebSocketTableWriter::from_halves(
                                        read_half,
                                        write_half,
                                        fields,
                                        *compression,
                                    )?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Wss {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(spec) => {
                                    WebSocketTableWriter::connect_tls(
                                        &spec.url,
                                        spec.config.clone(),
                                        fields,
                                        *compression,
                                    )
                                    .await?
                                }
                                Link::Accepted(halves) => {
                                    let (read_half, write_half) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    WebSocketTableWriter::from_halves(
                                        read_half,
                                        write_half,
                                        fields,
                                        *compression,
                                    )?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Http {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(url) => {
                                    HttpTableWriter::post(url.as_str(), fields, *compression)
                                        .await?
                                }
                                Link::Accepted(halves) => {
                                    let (recv_read, send_write) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    HttpTableWriter::from_exchange(
                                        recv_read,
                                        send_write,
                                        fields,
                                        *compression,
                                    )?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Https {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(spec) => {
                                    HttpTableWriter::post_tls(
                                        &spec.url,
                                        spec.config.clone(),
                                        fields,
                                        *compression,
                                    )
                                    .await?
                                }
                                Link::Accepted(halves) => {
                                    let (recv_read, send_write) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    HttpTableWriter::from_exchange(
                                        recv_read,
                                        send_write,
                                        fields,
                                        *compression,
                                    )?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    #[cfg(unix)]
                    ArrowIO::Uds {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(match link {
                                Link::Connect(path) => {
                                    UdsTableWriter::connect(&path, fields, *compression).await?
                                }
                                Link::Accepted(half) => {
                                    let half =
                                        half.take().expect("accepted half present until first batch");
                                    UdsTableWriter::from_write_half(half, fields, *compression)?
                                }
                            });
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Quic {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            let send = match link {
                                Link::Connect(spec) => {
                                    let (_recv, send) = QuicTransport::connect(
                                        spec.addr,
                                        &spec.server_name,
                                        spec.config.clone(),
                                    )
                                    .await?;
                                    send
                                }
                                Link::Accepted(halves) => {
                                    let (_recv, send) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    send
                                }
                            };
                            *writer = Some(QuicTableWriter::new(send, fields, *compression)?);
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Wt {
                        link,
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            let send = match link {
                                Link::Connect(spec) => {
                                    let config = tls::wt_client(&spec.ca)?;
                                    let (_recv, send) =
                                        WebTransport::connect(&spec.url, config).await?;
                                    send
                                }
                                Link::Accepted(halves) => {
                                    let (_recv, send) = halves
                                        .take()
                                        .expect("accepted halves present until first batch");
                                    send
                                }
                            };
                            *writer =
                                Some(WebTransportTableWriter::new(send, fields, *compression)?);
                        }
                        writer
                            .as_mut()
                            .expect("built above")
                            .write_table(table.clone())
                            .await
                    }
                    ArrowIO::Stdio {
                        compression,
                        writer,
                    } => {
                        if writer.is_none() {
                            *writer = Some(StdoutTableWriter::new(fields, *compression)?);
                        }
                        writer
                            .as_mut()
                            .expect("initialised above")
                            .write_table(table.clone())
                            .await
                    }
                }
            })
            .map_err(IoError::Io)
    }

    /// Finalises the Arrow IPC stream on the wire. A target that received
    /// no batches never connected, so there is nothing to finalise.
    fn finish(&mut self) -> Result<(), IoError> {
        runtime()
            .block_on(async {
                match self {
                    ArrowIO::Tcp { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::TcpParallel { writer, .. } => match writer.take() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Ws { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Wss { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Http { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Https { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    #[cfg(unix)]
                    ArrowIO::Uds { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Quic { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Wt { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                    ArrowIO::Stdio { writer, .. } => match writer.as_mut() {
                        Some(writer) => writer.finish().await,
                        None => Ok(()),
                    },
                }
            })
            .map_err(IoError::Io)
    }
}

/// Lightstream TLV protocol writers. One writer serves every wire,
/// since it frames any transport's byte stream.
pub enum LightstreamIO {
    Stream(LightstreamWriter<ByteSink>),
}

impl LightstreamIO {
    /// Sends one table frame under a registered type name, driving the
    /// async writer to completion on the embedded runtime.
    fn send_table(&mut self, name: &str, table: &Table) -> Result<(), IoError> {
        match self {
            LightstreamIO::Stream(writer) => runtime()
                .block_on(writer.send_table(name, table.clone()))
                .map_err(IoError::Io),
        }
    }

    /// Sends one opaque message frame under a registered type name.
    fn send_message(&mut self, name: &str, payload: &[u8]) -> Result<(), IoError> {
        match self {
            LightstreamIO::Stream(writer) => runtime()
                .block_on(writer.send(name, payload))
                .map_err(IoError::Io),
        }
    }

    /// Flushes and shuts the byte sink down.
    fn finish(&mut self) -> Result<(), IoError> {
        match self {
            LightstreamIO::Stream(writer) => runtime()
                .block_on(async {
                    writer.flush().await?;
                    writer.shutdown().await
                })
                .map_err(IoError::Io),
        }
    }
}

impl Target {
    /// Writes one unnamed batch, dispatching on the resolved arm. File
    /// and Arrow targets take unnamed batches, since their schema is a
    /// property of the whole stream.
    fn write(&mut self, table: &Table) -> Result<(), IoError> {
        match self {
            Target::File(file) => file.write(table),
            Target::Stream(protocol) => protocol.write(table),
        }
    }

    /// Writes one table frame under a registered type name on a
    /// Lightstream protocol target.
    fn write_named(&mut self, name: &str, table: &Table) -> Result<(), IoError> {
        match self {
            Target::Stream(Protocol::Lightstream(io)) => io.send_table(name, table),
            _ => Err(IoError::Format(
                "named writes apply to lightstream protocol targets".to_string(),
            )),
        }
    }

    /// True when the target speaks the Lightstream protocol, whose
    /// writes carry a registered type name.
    fn is_lightstream(&self) -> bool {
        matches!(self, Target::Stream(Protocol::Lightstream(_)))
    }

    /// Finalises the target, writing footers where the format has them.
    fn finish(&mut self) -> Result<(), IoError> {
        match self {
            Target::File(file) => file.finish(),
            Target::Stream(protocol) => protocol.finish(),
        }
    }
}

impl Protocol {
    /// Writes one unnamed batch, dispatching on the protocol arm.
    fn write(&mut self, table: &Table) -> Result<(), IoError> {
        match self {
            Protocol::Arrow(io) => io.write(table),
            Protocol::Lightstream(_) => Err(IoError::Format(
                "lightstream protocol writes carry a registered type name".to_string(),
            )),
        }
    }

    /// Finalises the stream, dispatching on the protocol arm.
    fn finish(&mut self) -> Result<(), IoError> {
        match self {
            Protocol::Arrow(io) => io.finish(),
            Protocol::Lightstream(io) => io.finish(),
        }
    }
}

impl FileIO {
    /// Writes one batch. The IPC arm streams it through the writer built
    /// from the first batch's schema, and the document formats buffer it
    /// for `finish`.
    fn write(&mut self, table: &Table) -> Result<(), IoError> {
        // Object URL uploads wait on the embedded runtime from this thread.
        #[cfg(feature = "object_storage")]
        let _guard = runtime().enter();
        match self {
            FileIO::Ipc {
                path,
                compression,
                writer,
            } => {
                if writer.is_none() {
                    let fields: Vec<Field> =
                        table.cols.iter().map(|col| (*col.field).clone()).collect();
                    let file = StorageWriter::create(&path).map_err(IoError::Io)?;
                    *writer = Some(SyncTableWriter::new(
                        file,
                        fields,
                        IPCMessageProtocol::File,
                        *compression,
                    ));
                }
                Ok(writer
                    .as_mut()
                    .expect("writer initialised above")
                    .write_table(table.clone())?)
            }
            FileIO::CsvStdio { options } => {
                let mut writer = CsvWriter::with_options(io::stdout(), options.clone());
                writer.write_table(table)?;
                writer.flush()?;
                options.write_header = false;
                Ok(())
            }
            FileIO::JsonStdio { options } => {
                let mut writer = JsonWriter::new(io::stdout(), options.clone());
                writer.write_table(table)?;
                writer.flush()?;
                Ok(())
            }
            FileIO::Parquet { batches, .. }
            | FileIO::Csv { batches, .. }
            | FileIO::Json { batches, .. } => {
                batches.push(Arc::new(table.clone()));
                Ok(())
            }
        }
    }

    /// Finalises the target. The IPC arm writes its footer, and the
    /// document formats write their buffered batches as one file. A target
    /// that received no batches writes nothing.
    fn finish(&mut self) -> Result<(), IoError> {
        // Object URL uploads wait on the embedded runtime from this thread.
        #[cfg(feature = "object_storage")]
        let _guard = runtime().enter();
        match self {
            FileIO::Ipc { writer, .. } => match writer.as_mut() {
                Some(writer) => Ok(writer.finish()?),
                None => Ok(()),
            },
            FileIO::Parquet {
                path,
                compression,
                batches,
            } => {
                if batches.is_empty() {
                    return Ok(());
                }
                let tables: Vec<Table> = batches.drain(..).map(|t| (*t).clone()).collect();
                let table = tables.consolidate();
                let mut file = StorageWriter::create(&path).map_err(IoError::Io)?;
                write_parquet_table(&table, &mut file, *compression)?;
                // Flushing completes the object when `path` is an object URL.
                Ok(file.flush()?)
            }
            FileIO::CsvStdio { .. } | FileIO::JsonStdio { .. } => Ok(()),
            FileIO::Csv {
                path,
                options,
                batches,
            } => {
                if batches.is_empty() {
                    return Ok(());
                }
                let name = Some(batches[0].name.clone());
                let super_table = SuperTable::from_batches(batches.drain(..).collect(), name);
                let file = StorageWriter::create(&path).map_err(IoError::Io)?;
                let mut writer = CsvWriter::with_options(file, options.clone());
                writer.write_supertable(&super_table)?;
                Ok(writer.flush()?)
            }
            FileIO::Json {
                path,
                options,
                batches,
            } => {
                if batches.is_empty() {
                    return Ok(());
                }
                let name = Some(batches[0].name.clone());
                let super_table = SuperTable::from_batches(batches.drain(..).collect(), name);
                let file = StorageWriter::create(&path).map_err(IoError::Io)?;
                let mut writer = JsonWriter::new(file, options.clone());
                writer.write_supertable(&super_table)?;
                Ok(writer.flush()?)
            }
        }
    }
}

impl Drop for FileIO {
    /// Finalises the file when the writer is dropped unclosed, so the
    /// footer lands even without a `close` call. Errors surface through
    /// `close`, which is the accountable path.
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Writes tables to a resolved target.
///
/// Accepts `minarrow.Table` and any object implementing the Arrow
/// PyCapsule stream protocol, so pyarrow and polars objects write without
/// conversion. A multi-batch object writes one batch at a time.
#[pyclass(name = "Writer")]
pub struct PyDataStreamWriter {
    inner: Mutex<Option<Target>>,
}

impl PyDataStreamWriter {
    /// Wraps a resolved target for handing to Python.
    pub fn new(target: Target) -> Self {
        Self {
            inner: Mutex::new(Some(target)),
        }
    }

    /// Takes the target out of the writer, erroring when it is closed.
    fn take(&self) -> PyResult<Target> {
        self.inner
            .lock()
            .map_err(|_| LightstreamError::new_err("writer lock poisoned"))?
            .take()
            .ok_or_else(|| LightstreamError::new_err("writer is closed"))
    }

    /// Returns the target after a write so the next call continues on
    /// the same target.
    fn restore(&self, target: Target) -> PyResult<()> {
        *self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("writer lock poisoned"))? = Some(target);
        Ok(())
    }
}

#[pymethods]
impl PyDataStreamWriter {
    /// Writes one table, or every batch of a multi-batch Arrow object,
    /// detaching from the Python interpreter for the blocking writes.
    /// Lightstream protocol targets take `name`, the registered table
    /// type each batch frames under.
    #[pyo3(signature = (data, name=None))]
    fn write(&self, py: Python<'_>, data: &Bound<'_, PyAny>, name: Option<&str>) -> PyResult<()> {
        let batches = table_to_rust(data)
            .map_err(|e| FormatError::new_err(format!("expected an Arrow-compatible object: {}", e)))?
            .batches;
        let mut target = self.take()?;
        if target.is_lightstream() != name.is_some() {
            self.restore(target)?;
            return Err(PyValueError::new_err(if name.is_some() {
                "name applies to lightstream protocol targets only"
            } else {
                "lightstream protocol writes need name=..., the registered table type"
            }));
        }
        let (target, result) = py.detach(move || {
            let result = batches.iter().try_for_each(|batch| match name {
                Some(name) => target.write_named(name, batch),
                None => target.write(batch),
            });
            (target, result)
        });
        self.restore(target)?;
        result.map_err(to_py_err)
    }

    /// Sends one opaque message frame under a registered type name on a
    /// Lightstream protocol target.
    fn write_message(&self, py: Python<'_>, name: &str, payload: &[u8]) -> PyResult<()> {
        let mut target = self.take()?;
        if !target.is_lightstream() {
            self.restore(target)?;
            return Err(ProtocolError::new_err(
                "write_message applies to lightstream protocol targets",
            ));
        }
        let (target, result) = py.detach(move || {
            let result = match &mut target {
                Target::Stream(Protocol::Lightstream(io)) => io.send_message(name, payload),
                _ => unreachable!("guarded by is_lightstream above"),
            };
            (target, result)
        });
        self.restore(target)?;
        result.map_err(to_py_err)
    }

    /// Registers a table type on a Lightstream protocol target. The
    /// schema comes from any Arrow-compatible object carrying at least
    /// one batch, such as a representative table. Returns the assigned
    /// type tag.
    fn register_table(&self, name: &str, schema: &Bound<'_, PyAny>) -> PyResult<u8> {
        let table = table_to_rust(schema)
            .map_err(|e| FormatError::new_err(format!("expected an Arrow-compatible object: {}", e)))?;
        let fields: Vec<Field> = match table.batches.first() {
            Some(batch) if !batch.cols.is_empty() => {
                batch.cols.iter().map(|col| (*col.field).clone()).collect()
            }
            _ => {
                return Err(FormatError::new_err(
                    "the schema object carries no columns, pass an object with at least one batch",
                ));
            }
        };
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("writer lock poisoned"))?;
        match guard.as_mut() {
            Some(Target::Stream(Protocol::Lightstream(LightstreamIO::Stream(writer)))) => {
                Ok(writer.register_table(name, fields))
            }
            Some(_) => Err(ProtocolError::new_err(
                "register_table applies to lightstream protocol targets",
            )),
            None => Err(LightstreamError::new_err("writer is closed")),
        }
    }

    /// Registers an opaque message type on a Lightstream protocol target.
    /// Returns the assigned type tag.
    fn register_message(&self, name: &str) -> PyResult<u8> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| LightstreamError::new_err("writer lock poisoned"))?;
        match guard.as_mut() {
            Some(Target::Stream(Protocol::Lightstream(LightstreamIO::Stream(writer)))) => {
                Ok(writer.register_message(name))
            }
            Some(_) => Err(ProtocolError::new_err(
                "register_message applies to lightstream protocol targets",
            )),
            None => Err(LightstreamError::new_err("writer is closed")),
        }
    }

    /// File targets write through on each batch, so flush delegates to the
    /// operating system. Objects at object URLs are completed, and become
    /// visible in their store, on `close`.
    fn flush(&self) -> PyResult<()> {
        Ok(())
    }

    /// Finalises the target, writing footers where the format has them.
    /// Further writes raise `LightstreamError`.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut target = self.take()?;
        let result = py.detach(move || target.finish());
        result.map_err(to_py_err)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}
