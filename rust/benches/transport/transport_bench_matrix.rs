// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Benchmarks throughput across the enabled transports.
//!
//! For each [`BenchMatrix`] cell selected by `LIGHTSTREAM_BENCH_MATRIX`, the
//! writer sends a fixed number of tables while the reader is timed. Connection
//! setup is completed before measurement begins.
//!
//! Each shape and scale pair creates a Criterion group named
//! `transport_<shape>_<scale>`. Enabled transport and compression combinations
//! are registered as separate benchmarks within the group.
//!
//! Supported benchmarks include TCP, Unix domain sockets and WebSocket in
//! async and sync forms, QUIC, WebTransport, HTTP/2 and the Lightstream protocol
//! over TCP. The `io_uring`
//! feature adds TCP and UDS variants. The `zstd` and `tls` features add their
//! respective compression and TLS variants.


#[path = "../common/bench_helpers.rs"]
mod bench_helpers;
use bench_helpers::{
    BenchMatrix, BenchScale, bench_schema, logical_payload_bytes_shape, make_bench_table_shape,
};

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

#[cfg(any(feature = "tcp", feature = "uds", feature = "websocket"))]
use lightstream::enums::{BufferChunkSize, IPCMessageProtocol};
#[cfg(any(feature = "tcp", feature = "uds", feature = "websocket"))]
use lightstream::models::readers::ipc::table::TableReader;
#[cfg(any(feature = "tcp", feature = "uds", feature = "websocket"))]
use lightstream::traits::transport_writer::IPCTransportWriter;
#[cfg(any(feature = "tcp", feature = "uds", feature = "websocket"))]
use minarrow::{Field, Table, Vec64};

#[cfg(feature = "tcp")]
use tokio::net::TcpListener;
#[cfg(feature = "uds")]
use tokio::net::UnixListener;
use tokio::runtime::Runtime;

#[cfg(feature = "zstd")]
use lightstream::compression::Compression;

#[cfg(feature = "io_uring")]
use lightstream::models::io_uring::{IoUringTcpConnection, IoUringUdsConnection};

#[cfg(any(feature = "quic", feature = "webtransport"))]
use std::net::SocketAddr;

// `rustls` is a dev-dependency available whenever the `tls` feature is on.
// The bench-only TLS paths below generate a self-signed cert per iteration
// with `rcgen` and pin it on the client side via a fresh `RootCertStore`.
#[cfg(all(feature = "tls", any(feature = "tcp", feature = "websocket")))]
use std::sync::Arc as TlsArc;

// ---------------------------------------------------------------------------
// Top-level matrix driver
// ---------------------------------------------------------------------------

#[cfg_attr(
    not(any(
        feature = "tcp",
        feature = "uds",
        feature = "websocket",
        feature = "quic",
        feature = "webtransport",
        feature = "http",
        feature = "io_uring",
        all(feature = "protocol", feature = "tcp"),
    )),
    allow(unused_variables)
)]
fn bench_transport(c: &mut Criterion) {
    // rustls needs a process-wide crypto provider before any TLS handshake.
    // The bench-only TLS variants below build a ServerConfig or ClientConfig
    // against it. Install the default ring provider once and ignore a
    // repeated install, which returns Err on the second attempt.
    #[cfg(any(
        all(feature = "tcp", feature = "tls"),
        all(feature = "websocket", feature = "tls"),
        feature = "quic",
    ))]
    {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    let rt = Runtime::new().unwrap();

    for (shape, scale) in BenchMatrix::from_env().cells() {
        let rows = scale.rows();
        let table = Arc::new(make_bench_table_shape(shape, rows));
        let schema = bench_schema(&table);
        let dict_regs = shape.dictionary_registrations();

        let group_name = format!("transport_{}_{}", shape.label(), scale.label());
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Bytes(logical_payload_bytes_shape(shape, rows, 1)));

        // Mid- and large-scale cells take long enough per iteration that
        // Criterion's default sample size runs too long, so cap it.
        if matches!(scale, BenchScale::Medium | BenchScale::Large) {
            group.sample_size(10);
        }

        // ---- TCP -------------------------------------------------------
        #[cfg(feature = "tcp")]
        bench_tcp(&mut group, &rt, &table, &schema, &dict_regs, "tcp", None);
        #[cfg(all(feature = "tcp", feature = "zstd"))]
        bench_tcp(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "tcp_zstd",
            Some(Compression::Zstd),
        );

        // ---- UDS -------------------------------------------------------
        #[cfg(feature = "uds")]
        bench_uds(&mut group, &rt, &table, &schema, &dict_regs, "uds", None);
        #[cfg(all(feature = "uds", feature = "zstd"))]
        bench_uds(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "uds_zstd",
            Some(Compression::Zstd),
        );
        #[cfg(feature = "uds")]
        bench_uds_sync(&mut group, &table, &schema, &dict_regs, "uds_sync", None);
        #[cfg(all(feature = "uds", feature = "zstd"))]
        bench_uds_sync(
            &mut group,
            &table,
            &schema,
            &dict_regs,
            "uds_sync_zstd",
            Some(Compression::Zstd),
        );

        // ---- WebSocket -------------------------------------------------
        #[cfg(feature = "websocket")]
        bench_websocket(&mut group, &rt, &table, &schema, &dict_regs, "websocket", None);
        #[cfg(all(feature = "websocket", feature = "zstd"))]
        bench_websocket(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "websocket_zstd",
            Some(Compression::Zstd),
        );
        #[cfg(feature = "websocket")]
        bench_websocket_sync(&mut group, &table, &schema, &dict_regs, "websocket_sync", None);
        #[cfg(all(feature = "websocket", feature = "zstd"))]
        bench_websocket_sync(
            &mut group,
            &table,
            &schema,
            &dict_regs,
            "websocket_sync_zstd",
            Some(Compression::Zstd),
        );

        // ---- TCP TLS ---------------------------------------------------
        #[cfg(all(feature = "tcp", feature = "tls"))]
        bench_tcp_tls(&mut group, &rt, &table, &schema, &dict_regs);

        // ---- WebSocket TLS ---------------------------------------------
        #[cfg(all(feature = "websocket", feature = "tls"))]
        bench_websocket_tls(&mut group, &rt, &table, &schema, &dict_regs);

        // ---- HTTP/2 (h2c) ----------------------------------------------
        #[cfg(feature = "http")]
        bench_http2(&mut group, &rt, &table, &schema, &dict_regs, "http2", None);
        #[cfg(all(feature = "http", feature = "zstd"))]
        bench_http2(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "http2_zstd",
            Some(Compression::Zstd),
        );

        // ---- QUIC ------------------------------------------------------
        #[cfg(feature = "quic")]
        bench_quic(&mut group, &rt, &table, &schema, &dict_regs, "quic", None);
        #[cfg(all(feature = "quic", feature = "zstd"))]
        bench_quic(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "quic_zstd",
            Some(Compression::Zstd),
        );

        // ---- WebTransport ----------------------------------------------
        #[cfg(feature = "webtransport")]
        bench_webtransport(&mut group, &rt, &table, &schema, &dict_regs, "webtransport", None);
        #[cfg(all(feature = "webtransport", feature = "zstd"))]
        bench_webtransport(
            &mut group,
            &rt,
            &table,
            &schema,
            &dict_regs,
            "webtransport_zstd",
            Some(Compression::Zstd),
        );

        // ---- Lightstream protocol over TCP -----------------------------
        #[cfg(all(feature = "protocol", feature = "tcp"))]
        bench_protocol_tcp(&mut group, &rt, &table, &schema);

        // ---- io_uring (Lightstream protocol over tokio-uring) ----------
        #[cfg(feature = "io_uring")]
        bench_uds_io_uring(&mut group, &table, &schema, "uds_io_uring");
        #[cfg(feature = "io_uring")]
        bench_tcp_io_uring(&mut group, &table, &schema, "tcp_io_uring");

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// TCP
// ---------------------------------------------------------------------------

#[cfg(feature = "tcp")]
fn bench_tcp(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let mut writer = lightstream::models::writers::tcp::TcpTableWriter::connect(
                        addr,
                        write_schema,
                        compression,
                    )
                    .await
                    .unwrap();
                    for (id, values) in write_dicts {
                        writer.register_dictionary(id, values);
                    }
                    for _ in 0..n {
                        writer.write_table((*write_table).clone()).await.unwrap();
                    }
                    writer.finish().await.unwrap();
                });

                let (socket, _) = listener.accept().await.unwrap();
                let (read_half, _write_half) = socket.into_split();
                let mut reader = TableReader::<Vec64<u8>>::new(
                    read_half,
                    BufferChunkSize::Http.chunk_size(),
                    IPCMessageProtocol::Stream,
                    None,
                );

                let start = std::time::Instant::now();
                let mut count = 0u64;
                while let Some(batch) = reader.read_next().await.unwrap() {
                    assert!(batch.n_rows > 0);
                    std::hint::black_box(&batch.cols);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                writer.await.unwrap();
                elapsed
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Unix domain sockets
// ---------------------------------------------------------------------------

#[cfg(feature = "uds")]
fn bench_uds(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let tempdir = tempfile::tempdir().unwrap();
                let socket_path = tempdir.path().join("transport_throughput.sock");
                let listener = UnixListener::bind(&socket_path).unwrap();

                let path = socket_path.clone();
                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let mut writer = lightstream::models::writers::uds::UdsTableWriter::connect(
                        &path,
                        write_schema,
                        compression,
                    )
                    .await
                    .unwrap();
                    for (id, values) in write_dicts {
                        writer.register_dictionary(id, values);
                    }
                    for _ in 0..n {
                        writer.write_table((*write_table).clone()).await.unwrap();
                    }
                    writer.finish().await.unwrap();
                });

                let (socket, _) = listener.accept().await.unwrap();
                let (read_half, _write_half) = socket.into_split();
                let mut reader = TableReader::<Vec64<u8>>::new(
                    read_half,
                    BufferChunkSize::Http.chunk_size(),
                    IPCMessageProtocol::Stream,
                    None,
                );

                let start = std::time::Instant::now();
                let mut count = 0u64;
                while let Some(batch) = reader.read_next().await.unwrap() {
                    assert!(batch.n_rows > 0);
                    std::hint::black_box(&batch.cols);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                writer.await.unwrap();
                elapsed
            }
        });
    });
}

/// Sync UDS throughput through `SyncUdsTableWriter` and `SyncUdsTableReader`.
#[cfg(feature = "uds")]
fn bench_uds_sync(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    use lightstream::models::readers::sync_uds::SyncUdsTableReader;
    use lightstream::models::writers::sync_uds::SyncUdsTableWriter;

    group.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let tempdir = tempfile::tempdir().unwrap();
            let socket_path = tempdir.path().join("transport_throughput_sync.sock");
            let listener = std::os::unix::net::UnixListener::bind(&socket_path).unwrap();

            let path = socket_path.clone();
            let write_table = Arc::clone(table);
            let write_schema = schema.to_vec();
            let write_dicts = dict_regs.to_vec();
            let n = iters;

            let writer = std::thread::spawn(move || {
                let mut writer =
                    SyncUdsTableWriter::connect(&path, write_schema, compression).unwrap();
                for (id, values) in write_dicts {
                    writer.register_dictionary(id, values);
                }
                for _ in 0..n {
                    writer.write_table((*write_table).clone()).unwrap();
                }
                writer.finish().unwrap();
            });

            let mut reader = SyncUdsTableReader::accept(&listener, None).unwrap();

            let start = std::time::Instant::now();
            let mut count = 0u64;
            while let Some(batch) = reader.read_next().unwrap() {
                assert!(batch.n_rows > 0);
                std::hint::black_box(&batch.cols);
                count += 1;
            }
            let elapsed = start.elapsed();
            assert_eq!(count, n);

            writer.join().unwrap();
            elapsed
        });
    });
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

#[cfg(feature = "websocket")]
fn bench_websocket(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let url = format!("ws://{addr}");

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let mut writer =
                        lightstream::models::writers::websocket::WebSocketTableWriter::connect(
                            &url,
                            write_schema,
                            compression,
                        )
                        .await
                        .unwrap();
                    for (id, values) in write_dicts {
                        writer.register_dictionary(id, values);
                    }
                    for _ in 0..n {
                        writer.write_table((*write_table).clone()).await.unwrap();
                    }
                    writer.finish().await.unwrap();
                });

                let (socket, _) = listener.accept().await.unwrap();
                let ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                let raw = ws.into_inner();
                let (read_half, write_half) = tokio::io::split(raw);
                let (shared_writer, _ws_write) =
                    lightstream::models::streams::websocket::WsWrite::new(write_half);
                let ws_read = lightstream::models::streams::websocket::WsRead::new(
                    read_half,
                    shared_writer,
                );
                let mut reader = TableReader::<Vec64<u8>>::new(
                    ws_read,
                    BufferChunkSize::WebSocket.chunk_size(),
                    IPCMessageProtocol::Stream,
                    None,
                );

                let start = std::time::Instant::now();
                let mut count = 0u64;
                while let Some(batch) = reader.read_next().await.unwrap() {
                    assert!(batch.n_rows > 0);
                    std::hint::black_box(&batch.cols);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                writer.await.unwrap();
                elapsed
            }
        });
    });
}

/// Sync WebSocket throughput through `SyncWebSocketTableWriter` and
/// `SyncWebSocketTableReader`.
#[cfg(feature = "websocket")]
fn bench_websocket_sync(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    use lightstream::models::readers::sync_websocket::SyncWebSocketTableReader;
    use lightstream::models::transports::sync_websocket::SyncWebSocketTransport;
    use lightstream::models::writers::sync_websocket::SyncWebSocketTableWriter;

    group.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let listener = SyncWebSocketTransport::bind("ws://127.0.0.1:0").unwrap();
            let url = format!("ws://{}", listener.local_addr().unwrap());

            let write_table = Arc::clone(table);
            let write_schema = schema.to_vec();
            let write_dicts = dict_regs.to_vec();
            let n = iters;

            let writer = std::thread::spawn(move || {
                let mut writer =
                    SyncWebSocketTableWriter::connect(&url, write_schema, compression).unwrap();
                for (id, values) in write_dicts {
                    writer.register_dictionary(id, values);
                }
                for _ in 0..n {
                    writer.write_table((*write_table).clone()).unwrap();
                }
                writer.finish().unwrap();
            });

            let mut reader = SyncWebSocketTableReader::accept(&listener, None).unwrap();

            let start = std::time::Instant::now();
            let mut count = 0u64;
            while let Some(batch) = reader.read_next().unwrap() {
                assert!(batch.n_rows > 0);
                std::hint::black_box(&batch.cols);
                count += 1;
            }
            let elapsed = start.elapsed();
            assert_eq!(count, n);

            writer.join().unwrap();
            elapsed
        });
    });
}

// ---------------------------------------------------------------------------
// Lightstream protocol over TCP
// ---------------------------------------------------------------------------

#[cfg(all(feature = "protocol", feature = "tcp"))]
fn bench_protocol_tcp(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
) {
    use lightstream::models::protocol::connection::TcpLightstreamConnection;

    group.bench_function("protocol_tcp", |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
                    let mut conn = TcpLightstreamConnection::from_tcp(stream, None);
                    conn.register_table("Data", write_schema);
                    for _ in 0..n {
                        conn.send_table("Data", Arc::clone(&write_table)).await.unwrap();
                    }
                    conn.flush().await.unwrap();
                    conn.shutdown().await.unwrap();
                });

                let (socket, _) = listener.accept().await.unwrap();
                let mut conn = TcpLightstreamConnection::from_tcp(socket, None);
                conn.register_table("Data", schema);

                let start = std::time::Instant::now();
                for _ in 0..n {
                    let msg = conn.recv().await.unwrap().unwrap();
                    assert!(msg.is_table());
                    std::hint::black_box(&msg);
                }
                let elapsed = start.elapsed();

                writer.await.unwrap();
                elapsed
            }
        });
    });
}

// ---------------------------------------------------------------------------
// QUIC
// ---------------------------------------------------------------------------

#[cfg(feature = "quic")]
fn bench_quic(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
                let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
                let key_der =
                    rustls::pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der())
                        .unwrap();

                let mut server_crypto = rustls::ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(vec![cert_der], key_der)
                    .unwrap();
                server_crypto.alpn_protocols = vec![b"ls".to_vec()];
                let server_config = quinn::ServerConfig::with_crypto(Arc::new(
                    quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).unwrap(),
                ));

                let endpoint =
                    quinn::Endpoint::server(server_config, "127.0.0.1:0".parse::<SocketAddr>().unwrap())
                        .unwrap();
                let addr = endpoint.local_addr().unwrap();

                let mut client_crypto = rustls::ClientConfig::builder()
                    .dangerous()
                    .with_custom_certificate_verifier(Arc::new(BenchSkipVerification))
                    .with_no_client_auth();
                client_crypto.alpn_protocols = vec![b"ls".to_vec()];
                let client_config = quinn::ClientConfig::new(Arc::new(
                    quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto).unwrap(),
                ));

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let mut client_ep =
                        quinn::Endpoint::client("0.0.0.0:0".parse::<SocketAddr>().unwrap()).unwrap();
                    client_ep.set_default_client_config(client_config);
                    let conn = client_ep.connect(addr, "localhost").unwrap().await.unwrap();
                    let send = conn.open_uni().await.unwrap();
                    let mut writer = lightstream::models::writers::quic::QuicTableWriter::new(
                        send,
                        write_schema,
                        compression,
                    )
                    .unwrap();
                    for (id, values) in write_dicts {
                        writer.register_dictionary(id, values);
                    }
                    for _ in 0..n {
                        writer.write_table((*write_table).clone()).await.unwrap();
                    }
                    writer.finish().await.unwrap();
                    // Keep the connection open until the reader drains every
                    // batch and closes from its side, so the stream delivers a
                    // clean finish rather than a connection-lost error.
                    conn.closed().await;
                });

                let incoming = endpoint.accept().await.unwrap();
                let conn = incoming.await.unwrap();
                let recv = conn.accept_uni().await.unwrap();
                let mut reader = TableReader::<Vec64<u8>>::new(
                    recv,
                    BufferChunkSize::WebTransport.chunk_size(),
                    IPCMessageProtocol::Stream,
                    None,
                );

                let start = std::time::Instant::now();
                let mut count = 0u64;
                while let Some(batch) = reader.read_next().await.unwrap() {
                    assert!(batch.n_rows > 0);
                    std::hint::black_box(&batch.cols);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                // Close from the reader side to release the writer's wait on
                // the connection, then join the writer task.
                conn.close(0u32.into(), b"done");
                writer.await.unwrap();
                elapsed
            }
        });
    });
}

// ---------------------------------------------------------------------------
// WebTransport
// ---------------------------------------------------------------------------

#[cfg(feature = "webtransport")]
fn bench_webtransport(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let identity =
                    wtransport::Identity::self_signed(["localhost", "127.0.0.1", "::1"]).unwrap();
                let cert_hash = identity.certificate_chain().as_slice()[0].hash();
                let server_config = wtransport::ServerConfig::builder()
                    .with_bind_default(0)
                    .with_identity(identity)
                    .build();
                let server = wtransport::Endpoint::server(server_config).unwrap();
                let addr: SocketAddr = server.local_addr().unwrap();

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                let writer = tokio::spawn(async move {
                    let client_config = wtransport::ClientConfig::builder()
                        .with_bind_default()
                        .with_server_certificate_hashes([cert_hash])
                        .build();
                    let client = wtransport::Endpoint::client(client_config).unwrap();
                    let conn = client
                        .connect(format!("https://127.0.0.1:{}", addr.port()))
                        .await
                        .unwrap();
                    let send = conn.open_uni().await.unwrap().await.unwrap();
                    let mut writer =
                        lightstream::models::writers::webtransport::WebTransportTableWriter::new(
                            send,
                            write_schema,
                            compression,
                        )
                        .unwrap();
                    for (id, values) in write_dicts {
                        writer.register_dictionary(id, values);
                    }
                    for _ in 0..n {
                        writer.write_table((*write_table).clone()).await.unwrap();
                    }
                    writer.finish().await.unwrap();
                    // Keep the connection open until the reader drains every
                    // batch and closes from its side, so the stream delivers a
                    // clean finish rather than a connection-lost error.
                    conn.closed().await;
                });

                let incoming = server.accept().await;
                let session_request = incoming.await.unwrap();
                let conn = session_request.accept().await.unwrap();
                let recv = conn.accept_uni().await.unwrap();
                let mut reader = TableReader::<Vec64<u8>>::new(
                    recv,
                    BufferChunkSize::WebTransport.chunk_size(),
                    IPCMessageProtocol::Stream,
                    None,
                );

                let start = std::time::Instant::now();
                let mut count = 0u64;
                while let Some(batch) = reader.read_next().await.unwrap() {
                    assert!(batch.n_rows > 0);
                    std::hint::black_box(&batch.cols);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                // Close from the reader side to release the writer's wait on
                // the connection, then join the writer task.
                conn.close(wtransport::VarInt::from_u32(0), b"done");
                writer.await.unwrap();
                elapsed
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Bench-only TLS verifier that skips certificate validation. QUIC needs it
// because each bench iteration generates a fresh self-signed cert, so the
// client has no trust root to validate against.
// ---------------------------------------------------------------------------

#[cfg(feature = "quic")]
#[derive(Debug)]
struct BenchSkipVerification;

#[cfg(feature = "quic")]
impl rustls::client::danger::ServerCertVerifier for BenchSkipVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

// ---------------------------------------------------------------------------
// TCP TLS - per-iteration self-signed cert, client trusts only that cert
// ---------------------------------------------------------------------------

#[cfg(all(feature = "tcp", feature = "tls"))]
fn bench_tcp_tls(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
) {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
    use rustls::{ClientConfig, RootCertStore, ServerConfig};
    use tokio_rustls::TlsAcceptor;

    group.bench_function("tcp_tls", |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
                    .unwrap();
                let cert_der: CertificateDer<'static> = cert.cert.der().clone();
                let key_der: PrivateKeyDer<'static> =
                    PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();

                let server_config = ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(vec![cert_der.clone()], key_der)
                    .unwrap();
                let acceptor = TlsAcceptor::from(TlsArc::new(server_config));

                let mut roots = RootCertStore::empty();
                roots.add(cert_der).unwrap();
                let client_config = TlsArc::new(
                    ClientConfig::builder()
                        .with_root_certificates(roots)
                        .with_no_client_auth(),
                );

                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let server_name = ServerName::try_from("localhost".to_string()).unwrap();

                let n = iters;

                // The server reader runs in the spawned task. The writer stays
                // in this task so its TLS write half is not dropped before the
                // server finishes reading. The opposite ordering races the TLS
                // close_notify against the server's read path and surfaces as
                // ConnectionReset on the reader side.
                let server = tokio::spawn(async move {
                    let (tcp, _peer) = listener.accept().await.unwrap();
                    let tls = acceptor.accept(tcp).await.unwrap();
                    let (read_half, _write_half) = tokio::io::split(tls);
                    let byte_stream =
                        lightstream::models::streams::tcp::TcpByteStream::from_tls_read_half(
                            read_half,
                            BufferChunkSize::Http,
                        );
                    let mut reader = TableReader::<Vec64<u8>>::new(
                        byte_stream,
                        BufferChunkSize::Http.chunk_size(),
                        IPCMessageProtocol::Stream,
                        None,
                    );

                    let start = std::time::Instant::now();
                    let mut count = 0u64;
                    while let Some(batch) = reader.read_next().await.unwrap() {
                        assert!(batch.n_rows > 0);
                        std::hint::black_box(&batch.cols);
                        count += 1;
                    }
                    let elapsed = start.elapsed();
                    assert_eq!(count, n);
                    elapsed
                });

                let mut writer = lightstream::models::writers::tcp::TcpTableWriter::connect_tls(
                    addr,
                    server_name,
                    client_config,
                    schema.clone(),
                    None,
                )
                .await
                .unwrap();
                for (id, values) in dict_regs.clone() {
                    writer.register_dictionary(id, values);
                }
                for _ in 0..n {
                    writer.write_table((*table).clone()).await.unwrap();
                }
                writer.finish().await.unwrap();

                server.await.unwrap()
            }
        });
    });
}

// ---------------------------------------------------------------------------
// WebSocket TLS (wss://)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "websocket", feature = "tls"))]
fn bench_websocket_tls(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
) {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    use rustls::{ClientConfig, RootCertStore, ServerConfig};
    use tokio_rustls::TlsAcceptor;

    group.bench_function("websocket_tls", |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
                    .unwrap();
                let cert_der: CertificateDer<'static> = cert.cert.der().clone();
                let key_der: PrivateKeyDer<'static> =
                    PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();

                let server_config = ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(vec![cert_der.clone()], key_der)
                    .unwrap();
                let acceptor = TlsAcceptor::from(TlsArc::new(server_config));

                let mut roots = RootCertStore::empty();
                roots.add(cert_der).unwrap();
                let client_config = TlsArc::new(
                    ClientConfig::builder()
                        .with_root_certificates(roots)
                        .with_no_client_auth(),
                );

                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let url = format!("wss://localhost:{}", addr.port());

                let n = iters;

                // As with `tcp_tls`, the writer stays in this task so the
                // TLS write half is not dropped before the server's reader
                // has consumed the stream.
                let server = tokio::spawn(async move {
                    let (tcp, _peer) = listener.accept().await.unwrap();
                    let tls = acceptor.accept(tcp).await.unwrap();
                    let ws = tokio_tungstenite::accept_async(tls).await.unwrap();
                    let raw = ws.into_inner();
                    let (read_half, write_half) = tokio::io::split(raw);
                    let (shared_writer, _ws_write) =
                        lightstream::models::streams::websocket::WsWrite::new(write_half);
                    let ws_read = lightstream::models::streams::websocket::WsRead::new(
                        read_half,
                        shared_writer,
                    );
                    let mut reader = TableReader::<Vec64<u8>>::new(
                        ws_read,
                        BufferChunkSize::WebSocket.chunk_size(),
                        IPCMessageProtocol::Stream,
                        None,
                    );

                    let start = std::time::Instant::now();
                    let mut count = 0u64;
                    while let Some(batch) = reader.read_next().await.unwrap() {
                        assert!(batch.n_rows > 0);
                        std::hint::black_box(&batch.cols);
                        count += 1;
                    }
                    let elapsed = start.elapsed();
                    assert_eq!(count, n);
                    elapsed
                });

                let mut writer =
                    lightstream::models::writers::websocket::WebSocketTableWriter::connect_tls(
                        &url,
                        client_config,
                        schema.clone(),
                        None,
                    )
                    .await
                    .unwrap();
                for (id, values) in dict_regs.clone() {
                    writer.register_dictionary(id, values);
                }
                for _ in 0..n {
                    writer.write_table((*table).clone()).await.unwrap();
                }
                writer.finish().await.unwrap();

                server.await.unwrap()
            }
        });
    });
}

// ---------------------------------------------------------------------------
// HTTP/2 (plaintext h2c). The server runs the h2 handshake, accepts the
// POST request, and decodes the request body via `HttpTableReader::from_recv`.
// The 8 MiB connection and stream windows let multi-MiB Arrow batches stream
// without a WINDOW_UPDATE round-trip every 64 KiB.
// ---------------------------------------------------------------------------

#[cfg(feature = "http")]
fn bench_http2(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    table: &Arc<Table>,
    schema: &[Field],
    dict_regs: &[(i64, Vec<String>)],
    name: &str,
    compression: Option<lightstream::compression::Compression>,
) {
    use lightstream::models::readers::http::HttpTableReader;
    use lightstream::models::writers::http::HttpTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;

    group.bench_function(name, |b| {
        b.to_async(rt).iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            let dict_regs = dict_regs.to_vec();
            async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let url = format!("http://{addr}/ingest");
                let ready = std::sync::Arc::new(tokio::sync::Notify::new());
                let ready_for_task = ready.clone();

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let write_dicts = dict_regs.clone();
                let n = iters;

                // Server side. Runs the h2 handshake, replies headers-only
                // (END_STREAM on the response so the client's drain task
                // returns), keeps the connection driver alive while the
                // request body streams in, then decodes it via the standard
                // HTTP/2 table reader.
                let server = tokio::spawn(async move {
                    ready_for_task.notify_one();
                    let (tcp, _peer) = listener.accept().await.unwrap();
                    let mut h2 = h2::server::Builder::new()
                        .initial_window_size(8 * 1024 * 1024)
                        .initial_connection_window_size(8 * 1024 * 1024)
                        .handshake::<_, bytes::Bytes>(tcp)
                        .await
                        .unwrap();
                    let (req, mut respond) = h2.accept().await.unwrap().unwrap();
                    let response = http::Response::builder().status(200).body(()).unwrap();
                    let _ = respond.send_response(response, true).unwrap();
                    let driver =
                        tokio::spawn(async move { while h2.accept().await.is_some() {} });
                    let reader = HttpTableReader::from_recv(req.into_body(), None);
                    let tables = reader.read_all_tables().await.unwrap();
                    driver.abort();
                    let _ = driver.await;
                    tables.len()
                });

                ready.notified().await;

                // Connection setup stays outside the timed region, as with the
                // other transports. The POST opens the request and runs the h2
                // handshake before the timer starts.
                let mut writer = HttpTableWriter::post(&url, write_schema, compression).await.unwrap();
                for (id, values) in write_dicts {
                    writer.register_dictionary(id, values);
                }

                let start = std::time::Instant::now();
                for _ in 0..n {
                    writer.write_table((*write_table).clone()).await.unwrap();
                }
                writer.finish().await.unwrap();
                let received = server.await.unwrap();
                let elapsed = start.elapsed();
                assert_eq!(received as u64, n);
                std::hint::black_box(received);
                elapsed
            }
        });
    });
}

// ---------------------------------------------------------------------------
// io_uring (Lightstream protocol over the tokio-uring completion driver)
// ---------------------------------------------------------------------------

#[cfg(feature = "io_uring")]
fn bench_uds_io_uring(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    table: &Arc<Table>,
    schema: &[Field],
    name: &str,
) {
    group.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            // io_uring runs on the tokio-uring runtime rather than the
            // shared tokio runtime the other transports drive.
            tokio_uring::start(async move {
                // socketpair sidesteps tokio-uring's AF_UNIX bind path.
                let (a, b) = std::os::unix::net::UnixStream::pair().unwrap();
                a.set_nonblocking(true).unwrap();
                b.set_nonblocking(true).unwrap();
                let stream_a = tokio_uring::net::UnixStream::from_std(a);
                let stream_b = tokio_uring::net::UnixStream::from_std(b);

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let n = iters;

                let writer = tokio_uring::spawn(async move {
                    let mut conn = IoUringUdsConnection::new(stream_a, None);
                    conn.register_table("Data", write_schema);
                    for _ in 0..n {
                        conn.send_table("Data", Arc::clone(&write_table)).await.unwrap();
                    }
                    conn.flush().await.unwrap();
                    conn.shutdown().await.unwrap();
                });

                let mut conn = IoUringUdsConnection::new(stream_b, None);
                conn.register_table("Data", schema);

                let start = std::time::Instant::now();
                let mut count = 0u64;
                for _ in 0..n {
                    let msg = conn.recv().await.unwrap().unwrap();
                    assert!(msg.is_table());
                    std::hint::black_box(&msg);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                writer.await.unwrap();
                elapsed
            })
        });
    });
}

#[cfg(feature = "io_uring")]
fn bench_tcp_io_uring(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    table: &Arc<Table>,
    schema: &[Field],
    name: &str,
) {
    group.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let table = Arc::clone(table);
            let schema = schema.to_vec();
            tokio_uring::start(async move {
                let listener =
                    tokio_uring::net::TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
                let addr = listener.local_addr().unwrap();

                let write_table = Arc::clone(&table);
                let write_schema = schema.clone();
                let n = iters;

                let writer = tokio_uring::spawn(async move {
                    let stream = tokio_uring::net::TcpStream::connect(addr).await.unwrap();
                    let mut conn = IoUringTcpConnection::new(stream, None);
                    conn.register_table("Data", write_schema);
                    for _ in 0..n {
                        conn.send_table("Data", Arc::clone(&write_table)).await.unwrap();
                    }
                    conn.flush().await.unwrap();
                    conn.shutdown().await.unwrap();
                });

                let (stream, _) = listener.accept().await.unwrap();
                let mut conn = IoUringTcpConnection::new(stream, None);
                conn.register_table("Data", schema);

                let start = std::time::Instant::now();
                let mut count = 0u64;
                for _ in 0..n {
                    let msg = conn.recv().await.unwrap().unwrap();
                    assert!(msg.is_table());
                    std::hint::black_box(&msg);
                    count += 1;
                }
                let elapsed = start.elapsed();
                assert_eq!(count, n);

                writer.await.unwrap();
                elapsed
            })
        });
    });
}

criterion_group!(benches, bench_transport);
criterion_main!(benches);
