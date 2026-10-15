// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # URI resolution
//!
//! Maps a URI plus the open options onto a constructed source or target.
//! The scheme picks the medium, the `protocol` argument picks the wire
//! protocol arm, and the format comes from the `format` argument or the
//! path extension.

#[cfg(unix)]
use std::fs;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use lightstream::compression::Compression;
#[cfg(feature = "object_storage")]
use lightstream::enums::BufferChunkSize;
use lightstream::error::IoError;
use lightstream::models::decoders::csv::CsvDecodeOptions;
use lightstream::models::decoders::json::JsonDecodeOptions;
use lightstream::models::encoders::csv::CsvEncodeOptions;
use lightstream::models::encoders::json::{JsonEncodeOptions, JsonFormat};
use lightstream::models::readers::chunked::arrow::ChunkedArrowReader;
use lightstream::models::readers::chunked::csv::{ChunkedCsvReadOptions, ChunkedCsvReader};
use lightstream::models::readers::chunked::parquet::ChunkedParquetReader;
use lightstream::models::readers::csv::CsvReader;
use lightstream::models::readers::ipc::file_table::FileTableReader;
#[cfg(unix)]
use lightstream::models::readers::ipc::mmap_table::MmapTableReader;
use lightstream::models::readers::http::HttpTableReader;
use lightstream::models::readers::parallel::tcp::TcpParallelTableReader;
use lightstream::models::readers::json::JsonReader;
use lightstream::models::readers::lightstream::LightstreamReader;
use lightstream::models::readers::object_storage::list_dir;
use lightstream::models::readers::stdio::StdinTableReader;
use lightstream::models::readers::tcp::TcpTableReader;
#[cfg(unix)]
use lightstream::models::readers::uds::UdsTableReader;
use lightstream::models::readers::quic::QuicTableReader;
use lightstream::enums::IPCMessageProtocol;
use lightstream::models::readers::websocket::WebSocketTableReader;
use lightstream::models::readers::webtransport::WebTransportTableReader;
#[cfg(feature = "object_storage")]
use lightstream::models::streams::object_store::ObjectByteStream;
use lightstream::models::streams::websocket::{WsRead, WsWrite};
use lightstream::models::transports::http::HttpTransport;
use lightstream::models::transports::quic::QuicTransport;
use lightstream::models::transports::tcp::TcpTransport;
#[cfg(unix)]
use lightstream::models::transports::uds::UdsTransport;
use lightstream::models::transports::webtransport::WebTransport;
use lightstream::models::transports::websocket::WebSocketTransport;
use lightstream::models::writers::lightstream::LightstreamWriter;
use lightstream::models::writers::object_storage::AsyncStorageWriter;
use lightstream::traits::chunked_table_reader::ChunkedTableReader;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use tokio::fs::File as TokioFile;
use tokio::io::{AsyncWriteExt, copy, sink, stdin, stdout};
use lightstream::traits::parallel_transport_reader::SortBehaviour;

use crate::errors::{FormatError, ProtocolError, TransportError, to_py_err};
use crate::listeners;
use crate::runtime::runtime;
use crate::tls;
use crate::{input, output};

/// Files at or above this size default to the mmap IPC reader. Smaller
/// files default to the buffered reader, where the mapping overhead
/// outweighs the zero-copy gain.
#[cfg(unix)]
const MMAP_DEFAULT_THRESHOLD: u64 = 64 * 1024 * 1024;

/// Default row count per decoded batch for the CSV and JSON readers.
const TEXT_BATCH_SIZE: usize = 65_536;

/// File formats resolvable from a name or a path extension.
#[derive(Clone, Copy, PartialEq)]
pub enum Format {
    Ipc,
    Parquet,
    Csv,
    Json,
}

impl Format {
    /// Resolves a format from the `format` argument.
    fn from_name(name: &str) -> PyResult<Format> {
        match name {
            "ipc" | "arrow" | "feather" => Ok(Format::Ipc),
            "parquet" => Ok(Format::Parquet),
            "csv" => Ok(Format::Csv),
            "json" => Ok(Format::Json),
            other => Err(PyValueError::new_err(format!(
                "unknown format '{}', expected one of ipc, parquet, csv, json",
                other
            ))),
        }
    }

    /// Resolves a format from a path extension.
    fn from_path(path: &Path) -> Option<Format> {
        match path.extension()?.to_str()? {
            "arrow" | "ipc" | "feather" => Some(Format::Ipc),
            "parquet" | "pq" => Some(Format::Parquet),
            "csv" => Some(Format::Csv),
            "json" | "jsonl" | "ndjson" => Some(Format::Json),
            _ => None,
        }
    }
}

/// The wire protocol arm the `protocol` argument selects. `None` means a
/// plain file-format read or write with no protocol involved.
enum WireProtocol {
    None,
    Arrow,
    Lightstream,
}

/// Validates the `protocol` argument.
fn resolve_protocol(protocol: Option<&str>) -> PyResult<WireProtocol> {
    match protocol {
        None => Ok(WireProtocol::None),
        Some("arrow") => Ok(WireProtocol::Arrow),
        Some("lightstream") => Ok(WireProtocol::Lightstream),
        Some(other) => Err(ProtocolError::new_err(format!(
            "unknown protocol '{}', expected 'arrow' or 'lightstream'",
            other
        ))),
    }
}

/// A URI resolved to its addressing form. Wire endpoints carry the
/// address their transport connects with.
enum Endpoint {
    /// A local file path, or an object URL, which the file readers and
    /// writers accept in place of a path.
    Path(PathBuf),
    Tcp(String),
    Ws(String),
    Wss(String),
    Http(String),
    Https(String),
    #[cfg(unix)]
    Uds(PathBuf),
    Quic(String),
    Wt(String),
    Stdio,
}

/// URL schemes that address object storage, matching the schemes of
/// lightstream's object storage backends.
const OBJECT_SCHEMES: [&str; 9] = [
    "s3", "s3a", "gs", "az", "adl", "azure", "abfs", "abfss", "memory",
];

/// True when `path` holds an object URL rather than a local path.
fn is_object_url(path: &Path) -> bool {
    path.to_str()
        .and_then(|text| text.split_once("://"))
        .is_some_and(|(scheme, _)| {
            OBJECT_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str())
        })
}

/// Splits a URI into its endpoint, raising for schemes whose transport
/// is not in this build. Object URLs resolve to path endpoints, keeping
/// the whole URL.
fn resolve_endpoint(uri: &str) -> PyResult<Endpoint> {
    if let Some((scheme, rest)) = uri.split_once("://") {
        return match scheme {
            "file" => Ok(Endpoint::Path(PathBuf::from(rest))),
            "tcp" => Ok(Endpoint::Tcp(rest.to_string())),
            "ws" => Ok(Endpoint::Ws(uri.to_string())),
            "http" => Ok(Endpoint::Http(uri.to_string())),
            #[cfg(unix)]
            "uds" => Ok(Endpoint::Uds(PathBuf::from(rest))),
            #[cfg(not(unix))]
            "uds" => Err(TransportError::new_err(
                "Unix-domain sockets are not supported on this platform",
            )),
            "quic" => Ok(Endpoint::Quic(rest.to_string())),
            "wt" => Ok(Endpoint::Wt(rest.to_string())),
            "wss" => Ok(Endpoint::Wss(uri.to_string())),
            "https" => Ok(Endpoint::Https(uri.to_string())),
            #[cfg(feature = "object_storage")]
            _ if is_object_url(Path::new(uri)) => Ok(Endpoint::Path(PathBuf::from(uri))),
            #[cfg(not(feature = "object_storage"))]
            _ if is_object_url(Path::new(uri)) => Err(TransportError::new_err(
                "object storage URLs require a build with the object_storage feature",
            )),
            other => Err(TransportError::new_err(format!(
                "unknown scheme '{}'",
                other
            ))),
        };
    }
    if uri == "stdio:" {
        return Ok(Endpoint::Stdio);
    }
    Ok(Endpoint::Path(PathBuf::from(uri)))
}

/// Validates the TLS arguments against the endpoint and peer role.
/// The quic, wt, wss, and https schemes carry TLS, so an accepting
/// peer presents `tls_cert` and `tls_key` while a connecting peer
/// verifies against `tls_ca`. The other wires carry no TLS surface.
fn resolve_tls_args(
    endpoint: &Endpoint,
    accept: bool,
    tls_cert: Option<&str>,
    tls_key: Option<&str>,
    tls_ca: Option<&str>,
) -> PyResult<()> {
    if !matches!(
        endpoint,
        Endpoint::Quic(_) | Endpoint::Wt(_) | Endpoint::Wss(_) | Endpoint::Https(_)
    ) {
        if tls_cert.is_some() || tls_key.is_some() || tls_ca.is_some() {
            return Err(TransportError::new_err(
                "tls arguments apply to the quic, wt, wss, and https endpoints",
            ));
        }
        return Ok(());
    }
    if accept {
        if tls_cert.is_none() || tls_key.is_none() {
            return Err(TransportError::new_err(
                "accepting on this endpoint requires tls_cert and tls_key",
            ));
        }
        if tls_ca.is_some() {
            return Err(TransportError::new_err(
                "tls_ca applies to the connecting peer",
            ));
        }
    } else {
        if tls_ca.is_none() {
            return Err(TransportError::new_err(
                "connecting to this endpoint requires tls_ca",
            ));
        }
        if tls_cert.is_some() || tls_key.is_some() {
            return Err(TransportError::new_err(
                "tls_cert and tls_key apply to the accepting peer",
            ));
        }
    }
    Ok(())
}

/// Extracts the name a connecting peer expects the accepting peer's
/// certificate to present, from the endpoint authority's host section.
fn resolve_server_name(authority: &str) -> String {
    let host = authority
        .rsplit_once(':')
        .map(|(host, _port)| host)
        .unwrap_or(authority);
    host.trim_start_matches('[').trim_end_matches(']').to_string()
}

/// Rejects file-format arguments on wire endpoints, so a misspelt intent
/// fails rather than silently ignoring the option.
fn reject_format_args_on_wire(
    format: Option<&str>,
    delimiter: Option<&str>,
    header: Option<bool>,
    batch_size: Option<usize>,
    mmap: Option<bool>,
    out_of_core: bool,
    base: Option<&str>,
) -> PyResult<()> {
    if format.is_some()
        || delimiter.is_some()
        || header.is_some()
        || batch_size.is_some()
        || mmap.is_some()
        || out_of_core
        || base.is_some()
    {
        return Err(PyValueError::new_err(
            "format arguments apply to file-format reads and writes, not wire endpoints",
        ));
    }
    Ok(())
}

/// Resolves the format for a path from the `format` argument or the
/// extension.
fn resolve_format(path: &Path, format: Option<&str>) -> PyResult<Format> {
    match format {
        Some(name) => Format::from_name(name),
        None => Format::from_path(path).ok_or_else(|| {
            FormatError::new_err(format!(
                "cannot infer a format from '{}', pass format=...",
                path.display()
            ))
        }),
    }
}

/// Chunk file extensions matched during directory inference, paired with
/// the format each one resolves to.
const CHUNK_EXTENSIONS: [(&str, Format); 3] = [
    (".arrow", Format::Ipc),
    (".parquet", Format::Parquet),
    (".csv", Format::Csv),
];

/// Infers the chunk base name and format inside a directory of
/// `<base>-<index>.<ext>` files written by the chunked writers. The
/// `format` and `base` arguments narrow the match when given.
fn infer_chunk_base(
    dir: &Path,
    format: Option<Format>,
    base: Option<&str>,
) -> PyResult<(String, Format)> {
    let mut bases: Vec<(String, Format)> = Vec::new();
    for entry in list_dir(dir).map_err(|e| FormatError::new_err(e.to_string()))? {
        let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((stem, ext_format)) = CHUNK_EXTENSIONS
            .iter()
            .find_map(|(ext, f)| name.strip_suffix(ext).map(|stem| (stem, *f)))
        else {
            continue;
        };
        if format.is_some_and(|f| f != ext_format) {
            continue;
        }
        let Some((found_base, index)) = stem.rsplit_once('-') else {
            continue;
        };
        if base.is_some_and(|b| b != found_base) {
            continue;
        }
        if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if !bases.iter().any(|(b, f)| b == found_base && *f == ext_format) {
            bases.push((found_base.to_string(), ext_format));
        }
    }
    match bases.len() {
        0 => Err(FormatError::new_err(format!(
            "no chunked table files found in '{}'",
            dir.display()
        ))),
        1 => Ok(bases.remove(0)),
        _ => Err(FormatError::new_err(format!(
            "multiple chunk sets found in '{}', pass base=... or format=...",
            dir.display()
        ))),
    }
}

/// Parses the `delimiter` argument into the single-byte form the CSV
/// codec takes.
fn resolve_delimiter(delimiter: Option<&str>) -> PyResult<Option<u8>> {
    match delimiter {
        None => Ok(None),
        Some(s) if s.len() == 1 && s.is_ascii() => Ok(Some(s.as_bytes()[0])),
        Some(other) => Err(PyValueError::new_err(format!(
            "delimiter must be a single ASCII character, got '{}'",
            other
        ))),
    }
}

/// Parses the `compression` argument into a codec.
fn resolve_compression(compression: Option<&str>) -> PyResult<Option<Compression>> {
    match compression {
        None => Ok(None),
        Some("zstd") => Ok(Some(Compression::Zstd)),
        Some("snappy") => Ok(Some(Compression::Snappy)),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown compression '{}', expected 'zstd' or 'snappy'",
            other
        ))),
    }
}

/// Rejects arguments that do not apply to the resolved format, so a
/// misspelt intent fails rather than silently ignoring the option.
fn reject_inapplicable(
    format: Format,
    delimiter: Option<&str>,
    header: Option<bool>,
    batch_size: Option<usize>,
) -> PyResult<()> {
    if format != Format::Csv && (delimiter.is_some() || header.is_some()) {
        return Err(PyValueError::new_err(
            "delimiter and header apply to the csv format only",
        ));
    }
    if !matches!(format, Format::Csv | Format::Json) && batch_size.is_some() {
        return Err(PyValueError::new_err(
            "batch_size applies to the csv and json formats only",
        ));
    }
    Ok(())
}

/// Resolves a URI and the read options into a constructed source.
pub fn resolve_source(
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
) -> PyResult<input::Source> {
    let endpoint = resolve_endpoint(uri)?;
    let wire_protocol = resolve_protocol(protocol)?;
    if accept {
        match endpoint {
            Endpoint::Path(_) => {
                return Err(TransportError::new_err(
                    "file sources have no accepting form",
                ));
            }
            Endpoint::Stdio => {
                return Err(TransportError::new_err("stdio has no accepting form"));
            }
            _ => {}
        }
    }
    resolve_tls_args(&endpoint, accept, tls_cert, tls_key, tls_ca)?;

    let path = match (endpoint, wire_protocol) {
        (Endpoint::Path(path), WireProtocol::None) => path,
        (Endpoint::Path(_), WireProtocol::Arrow) => {
            return Err(TransportError::new_err(
                "the arrow protocol does not frame the disk transport in this build",
            ));
        }
        (Endpoint::Path(path), WireProtocol::Lightstream) => {
            if parallel > 0 {
                return Err(TransportError::new_err(
                    "the parallel lightstream reader is not available in this build",
                ));
            }
            if format.is_some() || delimiter.is_some() || header.is_some() || batch_size.is_some()
            {
                return Err(PyValueError::new_err(
                    "format arguments apply to file-format reads, not protocol streams",
                ));
            }
            #[cfg(feature = "object_storage")]
            if is_object_url(&path) {
                let object = runtime()
                    .block_on(ObjectByteStream::open(uri, BufferChunkSize::FileIO))
                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                let reader = LightstreamReader::new(object, None);
                return Ok(input::Source::Stream(input::Protocol::Lightstream(
                    input::LightstreamIO::Stream(reader),
                )));
            }
            let file = runtime()
                .block_on(TokioFile::open(&path))
                .map_err(|e| to_py_err(IoError::Io(e)))?;
            let reader = LightstreamReader::new(file, None);
            return Ok(input::Source::Stream(input::Protocol::Lightstream(
                input::LightstreamIO::Stream(reader),
            )));
        }
        (Endpoint::Stdio, WireProtocol::None) if format.is_some() => {
            let text_format = Format::from_name(format.expect("checked above"))?;
            if text_format != Format::Csv {
                return Err(FormatError::new_err(
                    "text streaming over stdio supports the csv format",
                ));
            }
            if parallel > 0 {
                return Err(TransportError::new_err("stdio has no parallel form"));
            }
            if mmap.is_some() || out_of_core || base.is_some() {
                return Err(PyValueError::new_err(
                    "mmap, out_of_core, and base apply to file sources",
                ));
            }
            let mut options = CsvDecodeOptions::default();
            if let Some(d) = resolve_delimiter(delimiter)? {
                options.delimiter = d;
            }
            if let Some(h) = header {
                options.has_header = h;
            }
            let reader = CsvReader::from_reader(
                BufReader::new(io::stdin()),
                options,
                batch_size.unwrap_or(TEXT_BATCH_SIZE),
            );
            return Ok(input::Source::File(input::FileIO::CsvStdio { reader }));
        }
        (endpoint, wire_protocol) => {
            reject_format_args_on_wire(
                format,
                delimiter,
                header,
                batch_size,
                mmap,
                out_of_core,
                base,
            )?;
            if parallel > 0 {
                let Endpoint::Tcp(addr) = endpoint else {
                    return Err(TransportError::new_err(
                        "this transport has no parallel form in this build",
                    ));
                };
                if matches!(wire_protocol, WireProtocol::Lightstream) {
                    return Err(TransportError::new_err(
                        "the parallel lightstream reader is not available in this build",
                    ));
                }
                let listener = {
                    let _guard = runtime().enter();
                    listeners::tcp(addr.as_str())?
                };
                let reader = runtime()
                    .block_on(TcpParallelTableReader::accept(
                        &listener,
                        parallel,
                        SortBehaviour::None,
                        None,
                    ))
                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                return Ok(input::Source::Stream(input::Protocol::Arrow(
                    input::ArrowIO::TcpParallel(reader),
                )));
            }
            let protocol_arm = match wire_protocol {
                WireProtocol::Lightstream => {
                    let reader = match endpoint {
                        Endpoint::Tcp(addr) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::tcp(addr.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (read_half, _write_half) =
                                            TcpTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            read_half, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (read_half, _write_half) =
                                            TcpTransport::connect(addr.as_str()).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            read_half, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        #[cfg(unix)]
                        Endpoint::Uds(path) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::uds(&path)?
                                };
                                runtime()
                                    .block_on(async {
                                        let (read_half, _write_half) =
                                            UdsTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            read_half, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (read_half, _write_half) =
                                            UdsTransport::connect(&path).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            read_half, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Ws(url) => {
                            // Lightstream TLV bytes ride inside WebSocket
                            // binary frames, parsed by the WsRead adapter.
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::ws(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (read_half, write_half) =
                                            WebSocketTransport::accept(&listener).await?;
                                        let (shared_writer, _ws_write) =
                                            WsWrite::new(write_half);
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            WsRead::new(read_half, shared_writer),
                                            None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (read_half, write_half) =
                                            WebSocketTransport::connect(url.as_str()).await?;
                                        let (shared_writer, _ws_write) =
                                            WsWrite::new_client(write_half);
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            WsRead::new_client(read_half, shared_writer),
                                            None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Http(url) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::http(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (recv_read, mut send_write) =
                                            HttpTransport::accept(&listener).await?;
                                        send_write.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            recv_read, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (recv_read, mut send_write) =
                                            HttpTransport::connect(url.as_str()).await?;
                                        send_write.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            recv_read, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Quic(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::quic(&authority, Path::new(cert), Path::new(key))?
                                };
                                runtime()
                                    .block_on(async {
                                        let (recv, _send) =
                                            QuicTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(recv, None))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::quic_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let addr = listeners::socket_addr(&authority)?;
                                let server_name = resolve_server_name(&authority);
                                runtime()
                                    .block_on(async {
                                        let (recv, mut send) =
                                            QuicTransport::connect(addr, &server_name, config)
                                                .await?;
                                        // A reading-only connecting peer opens
                                        // the stream by shutting down its
                                        // write half.
                                        send.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(recv, None))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Wt(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = runtime().block_on(listeners::wt(
                                    &authority,
                                    Path::new(cert),
                                    Path::new(key),
                                ))?;
                                runtime()
                                    .block_on(async {
                                        let (recv, _send) =
                                            WebTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(recv, None))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wt_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let url = format!("https://{authority}");
                                runtime()
                                    .block_on(async {
                                        let (recv, mut send) =
                                            WebTransport::connect(&url, config).await?;
                                        // A reading-only connecting peer opens
                                        // the stream by shutting down its
                                        // write half.
                                        send.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(recv, None))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Wss(url) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::wss_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::wss(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (read_half, write_half) =
                                            WebSocketTransport::accept_tls(&listener, config)
                                                .await?;
                                        let (shared_writer, _ws_write) =
                                            WsWrite::new(write_half);
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            WsRead::new(read_half, shared_writer),
                                            None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wss_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime()
                                    .block_on(async {
                                        let (read_half, write_half) =
                                            WebSocketTransport::connect_tls(url.as_str(), config)
                                                .await?;
                                        let (shared_writer, _ws_write) =
                                            WsWrite::new_client(write_half);
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            WsRead::new_client(read_half, shared_writer),
                                            None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Https(url) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::https_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::https(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (recv_read, mut send_write) =
                                            HttpTransport::accept_tls(&listener, config).await?;
                                        send_write.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            recv_read, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::https_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime()
                                    .block_on(async {
                                        let (recv_read, mut send_write) =
                                            HttpTransport::connect_tls(url.as_str(), config)
                                                .await?;
                                        send_write.shutdown().await?;
                                        Ok::<_, io::Error>(LightstreamReader::new(
                                            recv_read, None,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Stdio => LightstreamReader::new(stdin(), None),
                        Endpoint::Path(_) => unreachable!("path endpoints resolve above"),
                    };
                    input::Protocol::Lightstream(input::LightstreamIO::Stream(reader))
                }
                WireProtocol::None | WireProtocol::Arrow => {
                    let arrow_io = match endpoint {
                        Endpoint::Tcp(addr) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::tcp(addr.as_str())?
                                };
                                runtime().block_on(TcpTableReader::accept(&listener, None))
                            } else {
                                runtime()
                                    .block_on(TcpTableReader::connect(addr.as_str(), None))
                            }
                            .map(input::ArrowIO::Tcp)
                            .map_err(|e| to_py_err(IoError::Io(e)))?
                        }
                        #[cfg(unix)]
                        Endpoint::Uds(path) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::uds(&path)?
                                };
                                runtime().block_on(UdsTableReader::accept(&listener, None))
                            } else {
                                runtime().block_on(UdsTableReader::connect(&path, None))
                            }
                            .map(input::ArrowIO::Uds)
                            .map_err(|e| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Ws(url) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::ws(url.as_str())?
                                };
                                runtime()
                                    .block_on(WebSocketTableReader::accept(&listener, None))
                            } else {
                                runtime()
                                    .block_on(WebSocketTableReader::connect(url.as_str(), None))
                            }
                            .map(input::ArrowIO::Ws)
                            .map_err(|e| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Wss(url) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::wss_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::wss(url.as_str())?
                                };
                                runtime().block_on(async {
                                    let (read_half, write_half) =
                                        WebSocketTransport::accept_tls(&listener, config).await?;
                                    Ok(WebSocketTableReader::from_halves(
                                        read_half,
                                        write_half,
                                        IPCMessageProtocol::Stream,
                                        None,
                                    ))
                                })
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wss_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime().block_on(WebSocketTableReader::connect_tls(
                                    url.as_str(),
                                    config,
                                    None,
                                ))
                            }
                            .map(input::ArrowIO::Ws)
                            .map_err(|e: io::Error| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Http(url) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::http(url.as_str())?
                                };
                                runtime().block_on(HttpTableReader::accept(&listener, None))
                            } else {
                                runtime().block_on(HttpTableReader::get(url.as_str(), None))
                            }
                            .map(input::ArrowIO::Http)
                            .map_err(|e| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Https(url) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::https_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::https(url.as_str())?
                                };
                                runtime().block_on(async {
                                    let (recv_read, send_write) =
                                        HttpTransport::accept_tls(&listener, config).await?;
                                    HttpTableReader::from_exchange(recv_read, send_write, None)
                                        .await
                                })
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::https_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime().block_on(HttpTableReader::get_tls(
                                    url.as_str(),
                                    config,
                                    None,
                                ))
                            }
                            .map(input::ArrowIO::Http)
                            .map_err(|e| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Quic(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::quic(&authority, Path::new(cert), Path::new(key))?
                                };
                                runtime().block_on(async {
                                    let (recv, _send) = QuicTransport::accept(&listener).await?;
                                    Ok(QuicTableReader::from_recv(recv, None))
                                })
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::quic_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let addr = listeners::socket_addr(&authority)?;
                                let server_name = resolve_server_name(&authority);
                                runtime().block_on(async {
                                    let (recv, mut send) =
                                        QuicTransport::connect(addr, &server_name, config)
                                            .await?;
                                    // A reading-only connecting peer opens the
                                    // stream by shutting down its write half.
                                    send.shutdown().await?;
                                    Ok(QuicTableReader::from_recv(recv, None))
                                })
                            }
                            .map(input::ArrowIO::Quic)
                            .map_err(|e: io::Error| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Wt(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = runtime().block_on(listeners::wt(
                                    &authority,
                                    Path::new(cert),
                                    Path::new(key),
                                ))?;
                                runtime().block_on(async {
                                    let (recv, _send) = WebTransport::accept(&listener).await?;
                                    Ok(WebTransportTableReader::from_recv(recv, None))
                                })
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wt_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let url = format!("https://{authority}");
                                runtime().block_on(async {
                                    let (recv, mut send) =
                                        WebTransport::connect(&url, config).await?;
                                    // A reading-only connecting peer opens the
                                    // stream by shutting down its write half.
                                    send.shutdown().await?;
                                    Ok(WebTransportTableReader::from_recv(recv, None))
                                })
                            }
                            .map(input::ArrowIO::Wt)
                            .map_err(|e: io::Error| to_py_err(IoError::Io(e)))?
                        }
                        Endpoint::Stdio => input::ArrowIO::Stdio(StdinTableReader::new(None)),
                        Endpoint::Path(_) => unreachable!("path endpoints resolve above"),
                    };
                    input::Protocol::Arrow(arrow_io)
                }
            };
            return Ok(input::Source::Stream(protocol_arm));
        }
    };

    if parallel > 0 {
        return Err(TransportError::new_err(
            "file sources have no parallel form",
        ));
    }
    if out_of_core && mmap == Some(true) {
        return Err(PyValueError::new_err(
            "out_of_core reads through the buffered reader, so it cannot combine with mmap=True",
        ));
    }
    #[cfg(not(unix))]
    if mmap == Some(true) {
        return Err(FormatError::new_err(
            "mmap is not supported on this platform; use buffered file reads",
        ));
    }
    let delimiter_byte = resolve_delimiter(delimiter)?;

    // Object URL requests wait on the embedded runtime from this thread.
    #[cfg(feature = "object_storage")]
    let _guard = runtime().enter();
    let object_url = is_object_url(&path);
    if object_url && mmap == Some(true) {
        return Err(FormatError::new_err(
            "mmap applies to local files only",
        ));
    }
    // Object storage has no directories. An object URL with objects
    // immediately beneath it addresses a chunk set, and any other object
    // URL addresses a single object.
    let chunk_set = if object_url {
        !list_dir(&path)
            .map_err(|e| FormatError::new_err(e.to_string()))?
            .is_empty()
    } else {
        path.is_dir()
    };
    if chunk_set {
        let named_format = format.map(Format::from_name).transpose()?;
        let (base, chunk_format) = match (base, named_format) {
            (Some(base), Some(format)) => (base.to_string(), format),
            _ => infer_chunk_base(&path, named_format, base)?,
        };
        reject_inapplicable(chunk_format, delimiter, header, batch_size)?;
        let file_io = match chunk_format {
            Format::Ipc => input::FileIO::IpcChunked {
                reader: ChunkedArrowReader::open(&path, &base, ())
                    .map_err(|e| FormatError::new_err(e.to_string()))?,
            },
            Format::Parquet => input::FileIO::ParquetChunked {
                reader: ChunkedParquetReader::open(&path, &base, ())
                    .map_err(|e| FormatError::new_err(e.to_string()))?,
            },
            Format::Csv => {
                let mut options = ChunkedCsvReadOptions::default();
                if let Some(d) = delimiter_byte {
                    options.decode.delimiter = d;
                }
                if let Some(h) = header {
                    options.decode.has_header = h;
                }
                if let Some(b) = batch_size {
                    options.batch_size = b;
                }
                input::FileIO::CsvChunked {
                    reader: ChunkedCsvReader::open(&path, &base, options)
                        .map_err(|e| FormatError::new_err(e.to_string()))?,
                }
            }
            Format::Json => {
                return Err(FormatError::new_err(
                    "json has no chunked directory reader",
                ));
            }
        };
        return Ok(input::Source::File(file_io));
    }

    let resolved_format = resolve_format(&path, format)?;
    reject_inapplicable(resolved_format, delimiter, header, batch_size)?;
    let file_io = match resolved_format {
        Format::Ipc => {
            #[cfg(unix)]
            if !object_url {
                let size = fs::metadata(&path)
                    .map_err(|e| FormatError::new_err(e.to_string()))?
                    .len();
                // Out-of-core reads use the buffered reader to keep pages reclaimable.
                let use_mmap = mmap.unwrap_or(size >= MMAP_DEFAULT_THRESHOLD) && !out_of_core;
                if use_mmap {
                    let reader = MmapTableReader::open(&path)
                        .map_err(|e| FormatError::new_err(e.to_string()))?;
                    return Ok(input::Source::File(input::FileIO::IpcMmap {
                        reader,
                        cursor: 0,
                    }));
                }
            }
            let reader = FileTableReader::open(&path)
                .map_err(|e| FormatError::new_err(e.to_string()))?;
            input::FileIO::Ipc { reader, cursor: 0 }
        }
        Format::Parquet => input::FileIO::Parquet { path, done: false },
        Format::Csv => {
            let mut options = CsvDecodeOptions::default();
            if let Some(d) = delimiter_byte {
                options.delimiter = d;
            }
            if let Some(h) = header {
                options.has_header = h;
            }
            let reader =
                CsvReader::from_path(&path, options, batch_size.unwrap_or(TEXT_BATCH_SIZE))
                    .map_err(|e| FormatError::new_err(e.to_string()))?;
            input::FileIO::Csv { reader }
        }
        Format::Json => {
            let json_format = resolve_json_format(&path);
            let reader = JsonReader::from_path(
                &path,
                json_format,
                JsonDecodeOptions::default(),
                batch_size.unwrap_or(TEXT_BATCH_SIZE),
            )
            .map_err(|e| FormatError::new_err(e.to_string()))?;
            input::FileIO::Json { reader }
        }
    };
    Ok(input::Source::File(file_io))
}

/// The JSON document layout for a path. Line-delimited extensions read
/// and write NDJSON, and everything else uses a single JSON array.
fn resolve_json_format(path: &Path) -> JsonFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") | Some("ndjson") => JsonFormat::Ndjson,
        _ => JsonFormat::Array { pretty: false },
    }
}

/// Resolves a URI and the write options into a constructed target.
pub fn resolve_target(
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
) -> PyResult<output::Target> {
    let endpoint = resolve_endpoint(uri)?;
    let wire_protocol = resolve_protocol(protocol)?;
    if accept {
        match endpoint {
            Endpoint::Path(_) => {
                return Err(TransportError::new_err(
                    "file targets have no accepting form",
                ));
            }
            Endpoint::Stdio => {
                return Err(TransportError::new_err("stdio has no accepting form"));
            }
            _ => {}
        }
        if parallel > 0 {
            return Err(TransportError::new_err(
                "the parallel writer connects, so it has no accepting form",
            ));
        }
    }
    resolve_tls_args(&endpoint, accept, tls_cert, tls_key, tls_ca)?;

    let path = match (endpoint, wire_protocol) {
        (Endpoint::Path(path), WireProtocol::None) => path,
        (Endpoint::Path(_), WireProtocol::Arrow) => {
            return Err(TransportError::new_err(
                "the arrow protocol does not frame the disk transport in this build",
            ));
        }
        (Endpoint::Path(path), WireProtocol::Lightstream) => {
            if parallel > 0 {
                return Err(TransportError::new_err(
                    "the parallel lightstream writer is not available in this build",
                ));
            }
            if format.is_some() || compression.is_some() || delimiter.is_some() || header.is_some()
            {
                return Err(PyValueError::new_err(
                    "format arguments apply to file-format writes, not protocol streams",
                ));
            }
            let file = runtime()
                .block_on(AsyncStorageWriter::create(&path))
                .map_err(|e| to_py_err(IoError::Io(e)))?;
            let writer = LightstreamWriter::new(Box::new(file) as _);
            return Ok(output::Target::Stream(output::Protocol::Lightstream(
                output::LightstreamIO::Stream(writer),
            )));
        }
        (Endpoint::Stdio, WireProtocol::None) if format.is_some() => {
            let text_format = Format::from_name(format.expect("checked above"))?;
            if parallel > 0 {
                return Err(TransportError::new_err("stdio has no parallel form"));
            }
            if compression.is_some() {
                return Err(PyValueError::new_err(
                    "compression applies to the ipc and parquet formats only",
                ));
            }
            match text_format {
                Format::Csv => {
                    let mut options = CsvEncodeOptions::default();
                    if let Some(d) = resolve_delimiter(delimiter)? {
                        options.delimiter = d;
                    }
                    if let Some(h) = header {
                        options.write_header = h;
                    }
                    return Ok(output::Target::File(output::FileIO::CsvStdio { options }));
                }
                Format::Json => {
                    if delimiter.is_some() || header.is_some() {
                        return Err(PyValueError::new_err(
                            "delimiter and header apply to the csv format only",
                        ));
                    }
                    let options = JsonEncodeOptions {
                        format: JsonFormat::Ndjson,
                        ..JsonEncodeOptions::default()
                    };
                    return Ok(output::Target::File(output::FileIO::JsonStdio { options }));
                }
                Format::Ipc | Format::Parquet => {
                    return Err(FormatError::new_err(
                        "text streaming over stdio supports the csv and json formats",
                    ));
                }
            }
        }
        (endpoint, wire_protocol) => {
            reject_format_args_on_wire(format, delimiter, header, None, None, false, None)?;
            let compression = resolve_compression(compression)?;
            if parallel > 0 {
                let Endpoint::Tcp(addr) = endpoint else {
                    return Err(TransportError::new_err(
                        "this transport has no parallel form in this build",
                    ));
                };
                if matches!(wire_protocol, WireProtocol::Lightstream) {
                    return Err(TransportError::new_err(
                        "the parallel lightstream writer is not available in this build",
                    ));
                }
                return Ok(output::Target::Stream(output::Protocol::Arrow(
                    output::ArrowIO::TcpParallel {
                        addr,
                        streams: parallel,
                        compression,
                        writer: None,
                    },
                )));
            }
            let protocol_arm = match wire_protocol {
                WireProtocol::Lightstream => {
                    if compression.is_some() {
                        return Err(PyValueError::new_err(
                            "compression applies to the arrow protocol and file formats",
                        ));
                    }
                    let writer = match endpoint {
                        Endpoint::Tcp(addr) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::tcp(addr.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            TcpTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(write_half) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            TcpTransport::connect(addr.as_str()).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(write_half) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        #[cfg(unix)]
                        Endpoint::Uds(path) => {
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::uds(&path)?
                                };
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            UdsTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(write_half) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            UdsTransport::connect(&path).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(write_half) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Ws(url) => {
                            // Lightstream TLV bytes ride inside WebSocket
                            // binary frames, built by the WsWrite adapter.
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::ws(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            WebSocketTransport::accept(&listener).await?;
                                        let (_shared_writer, ws_write) =
                                            WsWrite::new(write_half);
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(ws_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            WebSocketTransport::connect(url.as_str()).await?;
                                        let (_shared_writer, ws_write) =
                                            WsWrite::new_client(write_half);
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(ws_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Http(url) => {
                            // The unused request or response body drains in a
                            // task, reaching its clean end without resetting
                            // the h2 stream.
                            if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::http(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (mut recv_read, send_write) =
                                            HttpTransport::accept(&listener).await?;
                                        tokio::spawn(async move {
                                            let _ = copy(&mut recv_read, &mut sink()).await;
                                        });
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                runtime()
                                    .block_on(async {
                                        let (mut recv_read, send_write) =
                                            HttpTransport::connect(url.as_str()).await?;
                                        tokio::spawn(async move {
                                            let _ = copy(&mut recv_read, &mut sink()).await;
                                        });
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Quic(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::quic(&authority, Path::new(cert), Path::new(key))?
                                };
                                runtime()
                                    .block_on(async {
                                        let (_recv, send) =
                                            QuicTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::quic_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let addr = listeners::socket_addr(&authority)?;
                                let server_name = resolve_server_name(&authority);
                                runtime()
                                    .block_on(async {
                                        let (_recv, send) =
                                            QuicTransport::connect(addr, &server_name, config)
                                                .await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Wt(authority) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = runtime().block_on(listeners::wt(
                                    &authority,
                                    Path::new(cert),
                                    Path::new(key),
                                ))?;
                                runtime()
                                    .block_on(async {
                                        let (_recv, send) =
                                            WebTransport::accept(&listener).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wt_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let url = format!("https://{authority}");
                                runtime()
                                    .block_on(async {
                                        let (_recv, send) =
                                            WebTransport::connect(&url, config).await?;
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Wss(url) => {
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::wss_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::wss(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            WebSocketTransport::accept_tls(&listener, config)
                                                .await?;
                                        let (_shared_writer, ws_write) =
                                            WsWrite::new(write_half);
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(ws_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wss_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime()
                                    .block_on(async {
                                        let (_read_half, write_half) =
                                            WebSocketTransport::connect_tls(url.as_str(), config)
                                                .await?;
                                        let (_shared_writer, ws_write) =
                                            WsWrite::new_client(write_half);
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(ws_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Https(url) => {
                            // The unused response body drains in a task,
                            // reaching its clean end without resetting the h2
                            // stream.
                            if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::https_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::https(url.as_str())?
                                };
                                runtime()
                                    .block_on(async {
                                        let (mut recv_read, send_write) =
                                            HttpTransport::accept_tls(&listener, config).await?;
                                        tokio::spawn(async move {
                                            let _ = copy(&mut recv_read, &mut sink()).await;
                                        });
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::https_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                runtime()
                                    .block_on(async {
                                        let (mut recv_read, send_write) =
                                            HttpTransport::connect_tls(url.as_str(), config)
                                                .await?;
                                        tokio::spawn(async move {
                                            let _ = copy(&mut recv_read, &mut sink()).await;
                                        });
                                        Ok::<_, io::Error>(LightstreamWriter::new(
                                            Box::new(send_write) as _,
                                        ))
                                    })
                                    .map_err(|e| to_py_err(IoError::Io(e)))?
                            }
                        }
                        Endpoint::Stdio => LightstreamWriter::new(Box::new(stdout()) as _),
                        Endpoint::Path(_) => unreachable!("path endpoints resolve above"),
                    };
                    output::Protocol::Lightstream(output::LightstreamIO::Stream(writer))
                }
                WireProtocol::None | WireProtocol::Arrow => {
                    let arrow_io = match endpoint {
                        Endpoint::Tcp(addr) => {
                            let link = if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::tcp(addr.as_str())?
                                };
                                let (_read_half, write_half) = runtime()
                                    .block_on(TcpTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(write_half))
                            } else {
                                output::Link::Connect(addr)
                            };
                            output::ArrowIO::Tcp {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        #[cfg(unix)]
                        Endpoint::Uds(path) => {
                            let link = if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::uds(&path)?
                                };
                                let (_read_half, write_half) = runtime()
                                    .block_on(UdsTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(write_half))
                            } else {
                                output::Link::Connect(path)
                            };
                            output::ArrowIO::Uds {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Ws(url) => {
                            let link = if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::ws(url.as_str())?
                                };
                                let halves = runtime()
                                    .block_on(WebSocketTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                output::Link::Connect(url)
                            };
                            output::ArrowIO::Ws {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Wss(url) => {
                            let link = if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::wss_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::wss(url.as_str())?
                                };
                                let halves = runtime()
                                    .block_on(WebSocketTransport::accept_tls(&listener, config))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::wss_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                output::Link::Connect(output::WssConnect { url, config })
                            };
                            output::ArrowIO::Wss {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Http(url) => {
                            let link = if accept {
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::http(url.as_str())?
                                };
                                let halves = runtime()
                                    .block_on(HttpTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                output::Link::Connect(url)
                            };
                            output::ArrowIO::Http {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Https(url) => {
                            let link = if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let config = tls::https_server(Path::new(cert), Path::new(key))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::https(url.as_str())?
                                };
                                let halves = runtime()
                                    .block_on(HttpTransport::accept_tls(&listener, config))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::https_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                output::Link::Connect(output::HttpsConnect { url, config })
                            };
                            output::ArrowIO::Https {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Quic(authority) => {
                            let link = if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = {
                                    let _guard = runtime().enter();
                                    listeners::quic(&authority, Path::new(cert), Path::new(key))?
                                };
                                let halves = runtime()
                                    .block_on(QuicTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                let ca = tls_ca.expect("validated above");
                                let config = tls::quic_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                output::Link::Connect(output::QuicConnect {
                                    addr: listeners::socket_addr(&authority)?,
                                    server_name: resolve_server_name(&authority),
                                    config,
                                })
                            };
                            output::ArrowIO::Quic {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Wt(authority) => {
                            let link = if accept {
                                let cert = tls_cert.expect("validated above");
                                let key = tls_key.expect("validated above");
                                let listener = runtime().block_on(listeners::wt(
                                    &authority,
                                    Path::new(cert),
                                    Path::new(key),
                                ))?;
                                let halves = runtime()
                                    .block_on(WebTransport::accept(&listener))
                                    .map_err(|e| to_py_err(IoError::Io(e)))?;
                                output::Link::Accepted(Some(halves))
                            } else {
                                let ca = tls_ca.expect("validated above");
                                // Validates the roots at open. The dial rebuilds
                                // the configuration from the same path, since
                                // wtransport consumes it on use.
                                tls::wt_client(Path::new(ca))
                                    .map_err(|e| TransportError::new_err(e.to_string()))?;
                                output::Link::Connect(output::WtConnect {
                                    url: format!("https://{authority}"),
                                    ca: PathBuf::from(ca),
                                })
                            };
                            output::ArrowIO::Wt {
                                link,
                                compression,
                                writer: None,
                            }
                        }
                        Endpoint::Stdio => output::ArrowIO::Stdio {
                            compression,
                            writer: None,
                        },
                        Endpoint::Path(_) => unreachable!("path endpoints resolve above"),
                    };
                    output::Protocol::Arrow(arrow_io)
                }
            };
            return Ok(output::Target::Stream(protocol_arm));
        }
    };

    if parallel > 0 {
        return Err(TransportError::new_err(
            "file targets have no parallel form",
        ));
    }
    if !is_object_url(&path) && path.is_dir() {
        return Err(FormatError::new_err(
            "chunked directory targets are not available in this build",
        ));
    }
    let resolved_format = resolve_format(&path, format)?;
    reject_inapplicable(resolved_format, delimiter, header, None)?;
    let compression = resolve_compression(compression)?;
    if compression.is_some() && !matches!(resolved_format, Format::Ipc | Format::Parquet) {
        return Err(PyValueError::new_err(
            "compression applies to the ipc and parquet formats only",
        ));
    }

    let file_io = match resolved_format {
        Format::Ipc => output::FileIO::Ipc {
            path,
            compression,
            writer: None,
        },
        Format::Parquet => output::FileIO::Parquet {
            path,
            compression,
            batches: Vec::new(),
        },
        Format::Csv => {
            let mut options = CsvEncodeOptions::default();
            if let Some(d) = resolve_delimiter(delimiter)? {
                options.delimiter = d;
            }
            if let Some(h) = header {
                options.write_header = h;
            }
            output::FileIO::Csv {
                path,
                options,
                batches: Vec::new(),
            }
        }
        Format::Json => {
            let options = JsonEncodeOptions {
                format: resolve_json_format(&path),
                ..JsonEncodeOptions::default()
            };
            output::FileIO::Json {
                path,
                options,
                batches: Vec::new(),
            }
        }
    };
    Ok(output::Target::File(file_io))
}
