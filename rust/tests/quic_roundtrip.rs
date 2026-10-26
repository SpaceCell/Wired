// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! QUIC roundtrip integration test.
//!
//! Spins up a local QUIC endpoint, writes Arrow IPC tables from one task,
//! reads them back from another, and verifies the data survives the trip.

#![cfg(feature = "quic")]

use std::sync::Arc;

use futures_util::StreamExt;
use lightstream::enums::{BufferChunkSize, IPCMessageProtocol};
use lightstream::models::readers::ipc::table::TableReader;
use lightstream::models::readers::quic::QuicTableReader;
use lightstream::models::readers::parallel::quic::QuicParallelTableReader;
use lightstream::models::streams::quic::QuicByteStream;
use lightstream::models::writers::quic::QuicTableWriter;
use lightstream::models::writers::parallel::quic::QuicParallelTableWriter;
use lightstream::traits::parallel_transport_reader::{ParallelTransportReader, SortBehaviour};
use lightstream::traits::parallel_transport_writer::ParallelTransportWriter;
use lightstream::traits::transport_reader::IPCTransportReader;
use lightstream::traits::transport_writer::IPCTransportWriter;
use minarrow::{
    Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, FloatArray,
    IntegerArray, NumericArray, StringArray, Table, TextArray, Vec64,
    ffi::arrow_dtype::CategoricalIndexType,
};

fn make_test_table() -> Table {
    let int_col = FieldArray::new(
        Field {
            name: "ids".into(),
            dtype: ArrowType::Int32,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[10, 20, 30, 40])),
            null_mask: None,
        }))),
    );

    let float_col = FieldArray::new(
        Field {
            name: "values".into(),
            dtype: ArrowType::Float64,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(Vec64::from_slice(&[1.1, 2.2, 3.3, 4.4])),
            null_mask: None,
        }))),
    );

    let str_col = FieldArray::new(
        Field {
            name: "labels".into(),
            dtype: ArrowType::String,
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::String32(Arc::new(StringArray::new(
            Buffer::from(Vec64::from_slice("helloworldtest".as_bytes())),
            Some(Bitmask::new_set_all(4, true)),
            Buffer::from(Vec64::from_slice(&[0u32, 5, 10, 14, 14])),
        )))),
    );

    #[cfg(not(feature = "default_categorical_8"))]
    let dict_col = FieldArray::new(
        Field {
            name: "category".into(),
            dtype: ArrowType::Dictionary(CategoricalIndexType::UInt32),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray {
            data: Buffer::from(Vec64::from_slice(&[0u32, 1, 2, 0])),
            unique_values: Vec64::from(vec![
                "red".to_string(),
                "green".to_string(),
                "blue".to_string(),
            ]),
            null_mask: Some(Bitmask::new_set_all(4, true)),
        }))),
    );
    #[cfg(feature = "default_categorical_8")]
    let dict_col = FieldArray::new(
        Field {
            name: "category".into(),
            dtype: ArrowType::Dictionary(CategoricalIndexType::UInt8),
            nullable: true,
            metadata: Default::default(),
        },
        Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray {
            data: Buffer::from(Vec64::from_slice(&[0u8, 1, 2, 0])),
            unique_values: Vec64::from(vec![
                "red".to_string(),
                "green".to_string(),
                "blue".to_string(),
            ]),
            null_mask: Some(Bitmask::new_set_all(4, true)),
        }))),
    );

    Table::new(
        "test_table".to_string(),
        Some(vec![int_col, float_col, str_col, dict_col]),
    )
}

fn make_schema(table: &Table) -> Vec<Field> {
    table
        .cols
        .iter()
        .map(|fa| fa.field.as_ref().clone())
        .collect()
}

/// Create a self-signed TLS server config for testing.
fn make_server_config() -> quinn::ServerConfig {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
    let key_der =
        rustls::pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der.clone()], key_der)
        .unwrap();
    server_crypto.alpn_protocols = vec![b"ls".to_vec()];

    quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).unwrap(),
    ))
}

/// Create a client config that skips certificate verification for local testing.
fn make_client_config() -> quinn::ClientConfig {
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipVerification))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![b"ls".to_vec()];

    quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).unwrap(),
    ))
}

/// Certificate verifier that accepts any certificate, for test use only.
#[derive(Debug)]
struct SkipVerification;

impl rustls::client::danger::ServerCertVerifier for SkipVerification {
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
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
        ]
    }
}

/// Basic roundtrip: write one table over QUIC, read it back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_quic_single_table_roundtrip() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let send = conn.open_uni().await.unwrap();
        let mut writer = QuicTableWriter::new(send, write_schema, None).unwrap();
        writer.register_dictionary(
            3,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        writer.write_table(write_table).await.unwrap();
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let recv = conn.accept_uni().await.unwrap();
    let stream = QuicByteStream::new(recv, BufferChunkSize::WebTransport);
    let reader = TableReader::<Vec64<u8>>::new(stream, 64 * 1024, IPCMessageProtocol::Stream, None);
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");

    writer_handle.await.unwrap();

    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].n_rows, 4);
    assert_eq!(tables[0].cols.len(), 4);
}

/// Write multiple tables, read them all back.
#[tokio::test]
async fn test_quic_multi_table_roundtrip() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let send = conn.open_uni().await.unwrap();
        let mut writer = QuicTableWriter::new(send, write_schema, None).unwrap();
        writer.register_dictionary(
            3,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        writer.write_table(write_table.clone()).await.unwrap();
        writer.write_table(write_table.clone()).await.unwrap();
        writer.write_table(write_table).await.unwrap();
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let recv = conn.accept_uni().await.unwrap();
    let stream = QuicByteStream::new(recv, BufferChunkSize::WebTransport);
    let reader = TableReader::<Vec64<u8>>::new(stream, 64 * 1024, IPCMessageProtocol::Stream, None);
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");

    writer_handle.await.unwrap();

    assert_eq!(tables.len(), 3);
    for t in &tables {
        assert_eq!(t.n_rows, 4);
        assert_eq!(t.cols.len(), 4);
    }
}

/// Use the high-level QuicTableReader with Stream trait for continuous reading.
#[tokio::test]
async fn test_quic_stream_trait() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let send = conn.open_uni().await.unwrap();
        let mut writer = QuicTableWriter::new(send, write_schema, None).unwrap();
        writer.register_dictionary(
            3,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        writer.write_table(write_table.clone()).await.unwrap();
        writer.write_table(write_table).await.unwrap();
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let recv = conn.accept_uni().await.unwrap();
    let stream = QuicByteStream::new(recv, BufferChunkSize::WebTransport);
    let mut reader = QuicTableReader::from_stream(stream, IPCMessageProtocol::Stream, None);

    let mut count = 0;
    while let Some(result) = reader.next().await {
        let t = result.unwrap();
        assert_eq!(t.n_rows, 4);
        count += 1;
    }
    conn.close(0u32.into(), b"done");

    writer_handle.await.unwrap();
    assert_eq!(count, 2);
}

/// Parallel roundtrip: fan tables across several concurrent QUIC
/// streams, merge them on the far side, and verify every table arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_roundtrip() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 4;
    const TABLES: usize = 12;

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let dictionaries = vec![(
            3i64,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        )];
        let mut writer =
            QuicParallelTableWriter::open_ordered(&conn, STREAMS, write_schema, dictionaries, None)
                .await
                .unwrap();
        for _ in 0..TABLES {
            writer.write_table(write_table.clone()).await.unwrap();
        }
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let reader = QuicParallelTableReader::accept(&conn, STREAMS, SortBehaviour::Ordered, None)
        .await
        .unwrap();
    assert_eq!(reader.stream_count(), STREAMS);
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");

    writer_handle.await.unwrap();

    assert_eq!(tables.len(), TABLES);
    // open_ordered tags each table with a monotonic key 0..TABLES; Ordered
    // emits them in ascending key order across the streams.
    for (i, (t, seq)) in tables.iter().enumerate() {
        assert_eq!(t.n_rows, 4);
        assert_eq!(t.cols.len(), 4);
        assert_eq!(*seq, Some(i as u64));
    }
}

/// Collect multiple QUIC batches into a SuperTable without re-allocation.
#[tokio::test]
async fn test_quic_read_to_super_table() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let send = conn.open_uni().await.unwrap();
        let mut writer = QuicTableWriter::new(send, write_schema, None).unwrap();
        writer.register_dictionary(
            3,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );
        writer.write_table(write_table.clone()).await.unwrap();
        writer.write_table(write_table).await.unwrap();
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let recv = conn.accept_uni().await.unwrap();
    let stream = QuicByteStream::new(recv, BufferChunkSize::WebTransport);
    let reader = QuicTableReader::from_stream(stream, IPCMessageProtocol::Stream, None);
    let super_table = reader
        .read_to_super_table(Some("merged".into()), None)
        .await
        .unwrap();
    conn.close(0u32.into(), b"done");

    writer_handle.await.unwrap();

    assert_eq!(super_table.n_rows, 8);
    assert_eq!(super_table.batches.len(), 2);
    assert_eq!(super_table.name, "merged");
    for batch in &super_table.batches {
        assert_eq!(batch.n_rows, 4);
        assert_eq!(batch.cols.len(), 4);
    }
}

// ---------------------------------------------------------------------------
// Parallel stream correctness
// ---------------------------------------------------------------------------

/// Single Int32 column carrying `marker`, used to track which table lands
/// on which parallel stream and in what order.
fn make_marked_table(marker: i32) -> Table {
    let col = FieldArray::new(
        Field {
            name: "marker".into(),
            dtype: ArrowType::Int32,
            nullable: false,
            metadata: Default::default(),
        },
        Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[marker])),
            null_mask: None,
        }))),
    );
    Table::new("marked".to_string(), Some(vec![col]))
}

/// Read the marker back out of a table built by `make_marked_table`.
fn marker_of(table: &Table) -> i32 {
    match &table.cols[0].array {
        Array::NumericArray(NumericArray::Int32(arr)) => arr.data[0],
        other => panic!("expected an Int32 marker column, found {other:?}"),
    }
}

/// Decode the category column into its string labels through the
/// dictionary indices.
fn category_labels(table: &Table) -> Vec<String> {
    match &table.cols[3].array {
        #[cfg(feature = "default_categorical_8")]
        Array::TextArray(TextArray::Categorical8(arr)) => arr
            .data
            .iter()
            .map(|&i| arr.unique_values[i as usize].clone())
            .collect(),
        Array::TextArray(TextArray::Categorical32(arr)) => arr
            .data
            .iter()
            .map(|&i| arr.unique_values[i as usize].clone())
            .collect(),
        other => panic!("expected a categorical column, found {other:?}"),
    }
}

/// Fan indexed tables across several streams and check round-robin
/// distribution and within-stream ordering. Table `i` routes to stream
/// `i % STREAMS`, so markers sharing a residue arrive in ascending order
/// even though the merge interleaves the streams.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_ordering_and_round_robin() {
    let schema = vec![Field {
        name: "marker".into(),
        dtype: ArrowType::Int32,
        nullable: false,
        metadata: Default::default(),
    }];

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 4;
    const TABLES: i32 = 40;

    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut writer = QuicParallelTableWriter::open(&conn, STREAMS, write_schema, Vec::new(), None)
            .await
            .unwrap();
        for i in 0..TABLES {
            writer.write_table(make_marked_table(i)).await.unwrap();
        }
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let reader = QuicParallelTableReader::accept(&conn, STREAMS, SortBehaviour::None, None)
        .await
        .unwrap();
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");
    writer_handle.await.unwrap();

    let markers: Vec<i32> = tables.iter().map(|(t, _)| marker_of(t)).collect();
    assert_eq!(markers.len(), TABLES as usize);

    // Every marker arrives once.
    let mut sorted = markers.clone();
    sorted.sort();
    assert_eq!(sorted, (0..TABLES).collect::<Vec<_>>());

    // Markers sharing a residue mod STREAMS came down one stream, so they
    // must stay in ascending order.
    for residue in 0..STREAMS as i32 {
        let stream_markers: Vec<i32> =
            markers.iter().copied().filter(|m| m % STREAMS as i32 == residue).collect();
        let mut ascending = stream_markers.clone();
        ascending.sort();
        assert_eq!(stream_markers, ascending, "stream {residue} arrived out of order");
    }
}

/// The Ordered merge emits tables in global write order even when the streams
/// carry uneven counts. With 42 tables over 4 streams the streams hold
/// 11/11/10/10 tables, so the rotation must terminate on the short streams
/// without dropping or reordering. Each table's surfaced key equals its write
/// index.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_ordered_uneven_streams() {
    let schema = vec![Field {
        name: "marker".into(),
        dtype: ArrowType::Int32,
        nullable: false,
        metadata: Default::default(),
    }];

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 4;
    const TABLES: i32 = 42;

    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut writer =
            QuicParallelTableWriter::open_ordered(&conn, STREAMS, write_schema, Vec::new(), None)
                .await
                .unwrap();
        for i in 0..TABLES {
            writer.write_table(make_marked_table(i)).await.unwrap();
        }
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let reader = QuicParallelTableReader::accept(&conn, STREAMS, SortBehaviour::Ordered, None)
        .await
        .unwrap();
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");
    writer_handle.await.unwrap();

    assert_eq!(tables.len(), TABLES as usize);
    // Ordered emits in global write order, and the surfaced key matches the
    // write index of each table.
    for (i, (table, seq)) in tables.iter().enumerate() {
        assert_eq!(marker_of(table), i as i32, "table {i} arrived out of order");
        assert_eq!(*seq, Some(i as u64));
    }
}

/// Every parallel stream registers the dictionary, so a table landing on
/// any stream decodes its categorical column to the same labels. TABLES
/// exceeds STREAMS, so every stream carries at least one table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_dictionary_per_stream() {
    let table = make_test_table();
    let schema = make_schema(&table);

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 4;
    const TABLES: usize = 8;

    let write_table = table.clone();
    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let dictionaries = vec![(
            3i64,
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        )];
        let mut writer = QuicParallelTableWriter::open(&conn, STREAMS, write_schema, dictionaries, None)
            .await
            .unwrap();
        for _ in 0..TABLES {
            writer.write_table(write_table.clone()).await.unwrap();
        }
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let reader = QuicParallelTableReader::accept(&conn, STREAMS, SortBehaviour::None, None)
        .await
        .unwrap();
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");
    writer_handle.await.unwrap();

    assert_eq!(tables.len(), TABLES);
    for (t, _) in &tables {
        assert_eq!(category_labels(t), vec!["red", "green", "blue", "red"]);
    }
}

/// Drive far more tables than the per-stream channel depth so the producer
/// blocks on send and resumes as the streams drain. Completion with every
/// table delivered shows backpressure holds without deadlock.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_backpressure_under_load() {
    let schema = vec![Field {
        name: "marker".into(),
        dtype: ArrowType::Int32,
        nullable: false,
        metadata: Default::default(),
    }];

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 3;
    const TABLES: i32 = 200;

    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut writer = QuicParallelTableWriter::open(&conn, STREAMS, write_schema, Vec::new(), None)
            .await
            .unwrap();
        for i in 0..TABLES {
            writer.write_table(make_marked_table(i)).await.unwrap();
        }
        writer.finish().await.unwrap();
        conn.closed().await;
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let reader = QuicParallelTableReader::accept(&conn, STREAMS, SortBehaviour::None, None)
        .await
        .unwrap();
    let tables = reader.read_all_tables().await.unwrap();
    conn.close(0u32.into(), b"done");
    writer_handle.await.unwrap();

    let mut markers: Vec<i32> = tables.iter().map(|(t, _)| marker_of(t)).collect();
    assert_eq!(markers.len(), TABLES as usize);
    markers.sort();
    assert_eq!(markers, (0..TABLES).collect::<Vec<_>>());
}

/// When the peer stops the receive streams mid-send, finish surfaces the
/// resulting stream error instead of reporting success.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_quic_parallel_finish_reports_error() {
    let schema = vec![Field {
        name: "marker".into(),
        dtype: ArrowType::Int32,
        nullable: false,
        metadata: Default::default(),
    }];

    let server_config = make_server_config();
    let endpoint = quinn::Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = endpoint.local_addr().unwrap();

    const STREAMS: usize = 4;

    let write_schema = schema.clone();
    let writer_handle = tokio::spawn(async move {
        let client_endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        let conn = client_endpoint
            .connect_with(make_client_config(), addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut writer = QuicParallelTableWriter::open(&conn, STREAMS, write_schema, Vec::new(), None)
            .await
            .unwrap();
        // Keep pushing until a stream error propagates back through the
        // producer, then report what finish makes of the failed streams.
        for i in 0..2000 {
            if writer.write_table(make_marked_table(i)).await.is_err() {
                break;
            }
        }
        writer.finish().await
    });

    let conn = endpoint.accept().await.unwrap().await.unwrap();
    let mut recvs = Vec::with_capacity(STREAMS);
    for _ in 0..STREAMS {
        recvs.push(conn.accept_uni().await.unwrap());
    }
    for recv in &mut recvs {
        let _ = recv.stop(7u32.into());
    }

    let result = writer_handle.await.unwrap();
    assert!(result.is_err(), "finish should surface the peer stop as an error");
    conn.close(0u32.into(), b"done");
}
