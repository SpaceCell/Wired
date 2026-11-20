// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Transport roundtrip integration tests.
//!
//! Exercises both peer roles of every Transport through the table
//! readers and writers. Each wire runs the roles both ways round, with
//! the accepting peer writing in one test and reading in the other,
//! and TCP additionally serving connections back-to-back on one
//! persistent listener.

#![cfg(any(
    feature = "tcp",
    feature = "uds",
    feature = "websocket",
    feature = "http",
    feature = "quic",
    feature = "webtransport"
))]

use minarrow::{FieldArray, Table, Vec64, arr_f64, arr_i64};

fn make_table(label: &str, n_rows: usize) -> Table {
    let ids: Vec64<i64> = (0..n_rows as i64).collect();
    let values: Vec64<f64> = (0..n_rows).map(|i| i as f64 * 0.5).collect();
    Table::new(
        label.to_string(),
        Some(vec![
            FieldArray::from_arr("id", arr_i64!(ids)),
            FieldArray::from_arr("value", arr_f64!(values)),
        ]),
    )
}

fn table_schema() -> Vec<minarrow::Field> {
    make_table("_schema", 0)
        .schema()
        .iter()
        .map(|f| (**f).clone())
        .collect()
}

fn assert_tables(tables: &[Table], expected_rows: &[usize]) {
    assert_eq!(tables.len(), expected_rows.len());
    for (table, rows) in tables.iter().zip(expected_rows) {
        assert_eq!(table.n_rows, *rows);
        assert_eq!(table.cols.len(), 2);
    }
}

#[cfg(feature = "tcp")]
mod tcp {
    use lightstream::models::readers::tcp::TcpTableReader;
    use lightstream::models::transports::tcp::TcpTransport;
    use lightstream::models::writers::tcp::TcpTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;

    use super::{assert_tables, make_table, table_schema};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tcp_accepting_writer_to_connecting_reader() {
        let listener = TcpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let mut writer = TcpTableWriter::accept(&listener, table_schema(), None)
                .await
                .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let reader = TcpTableReader::connect(addr, None).await.unwrap();
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tcp_connecting_writer_to_accepting_reader() {
        let listener = TcpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let reader = TcpTableReader::accept(&listener, None).await.unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let mut writer = TcpTableWriter::connect(addr, table_schema(), None)
            .await
            .unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tcp_listener_serves_connections_back_to_back() {
        let listener = TcpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            for rows in [2usize, 6] {
                let mut writer = TcpTableWriter::accept(&listener, table_schema(), None)
                    .await
                    .unwrap();
                writer.write_table(make_table("t", rows)).await.unwrap();
                writer.finish().await.unwrap();
            }
        });

        for rows in [2usize, 6] {
            let reader = TcpTableReader::connect(addr, None).await.unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[rows]);
        }
        server.await.unwrap();
    }
}

#[cfg(feature = "uds")]
mod uds {
    use lightstream::models::readers::uds::UdsTableReader;
    use lightstream::models::transports::uds::UdsTransport;
    use lightstream::models::writers::uds::UdsTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;

    use super::{assert_tables, make_table, table_schema};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_uds_accepting_writer_to_connecting_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("accepting_writer.sock");
        let listener = UdsTransport::bind(&path).unwrap();

        let server = tokio::spawn(async move {
            let mut writer = UdsTableWriter::accept(&listener, table_schema(), None)
                .await
                .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let reader = UdsTableReader::connect(&path, None).await.unwrap();
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_uds_connecting_writer_to_accepting_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("accepting_reader.sock");
        let listener = UdsTransport::bind(&path).unwrap();

        let server = tokio::spawn(async move {
            let reader = UdsTableReader::accept(&listener, None).await.unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let mut writer = UdsTableWriter::connect(&path, table_schema(), None)
            .await
            .unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }
}

#[cfg(feature = "websocket")]
mod ws {
    use lightstream::models::readers::websocket::WebSocketTableReader;
    use lightstream::models::transports::websocket::WebSocketTransport;
    use lightstream::models::writers::websocket::WebSocketTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;

    use super::{assert_tables, make_table, table_schema};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_accepting_writer_to_connecting_reader() {
        let listener = WebSocketTransport::bind("ws://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let mut writer = WebSocketTableWriter::accept(&listener, table_schema(), None)
                .await
                .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let reader = WebSocketTableReader::connect(&format!("ws://{addr}"), None)
            .await
            .unwrap();
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_connecting_writer_to_accepting_reader() {
        let listener = WebSocketTransport::bind("ws://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&listener, None).await.unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let mut writer =
            WebSocketTableWriter::connect(&format!("ws://{addr}"), table_schema(), None)
                .await
                .unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_accept_returns_the_request_path() {
        let listener = WebSocketTransport::bind("ws://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (path, read_half, write_half) = WebSocketTransport::accept(&listener).await.unwrap();
            assert_eq!(path, "/prices");
            let mut writer =
                WebSocketTableWriter::from_halves(read_half, write_half, table_schema(), None)
                    .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let reader = WebSocketTableReader::connect(&format!("ws://{addr}/prices"), None)
            .await
            .unwrap();
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3]);
        server.await.unwrap();
    }
}

#[cfg(feature = "http")]
mod http {
    use lightstream::models::readers::http::HttpTableReader;
    use lightstream::models::transports::http::HttpTransport;
    use lightstream::models::writers::http::HttpTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;

    use super::{assert_tables, make_table, table_schema};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_accepting_writer_to_connecting_reader() {
        let listener = HttpTransport::bind("http://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let mut writer = HttpTableWriter::accept(&listener, table_schema(), None)
                .await
                .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let reader = HttpTableReader::get(&format!("http://{addr}/feed"), None)
            .await
            .unwrap();
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_connecting_writer_to_accepting_reader() {
        let listener = HttpTransport::bind("http://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&listener, None).await.unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let mut writer =
            HttpTableWriter::post(&format!("http://{addr}/ingest"), table_schema(), None)
                .await
                .unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }
}

#[cfg(feature = "quic")]
mod quic {
    use std::sync::Arc;

    use lightstream::models::readers::quic::QuicTableReader;
    use lightstream::models::transports::quic::QuicTransport;
    use lightstream::models::writers::quic::QuicTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use tokio::io::AsyncWriteExt;

    use super::{assert_tables, make_table, table_schema};

    /// Create a self-signed TLS server config for testing.
    fn make_server_config() -> quinn::ServerConfig {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
        let key_der =
            rustls::pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();

        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key_der)
            .unwrap();
        server_crypto.alpn_protocols = vec![b"ls".to_vec()];

        quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).unwrap(),
        ))
    }

    /// Create a client config that skips certificate verification for
    /// local testing.
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
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_quic_connecting_writer_to_accepting_reader() {
        let listener = QuicTransport::bind(
            "127.0.0.1:0".parse().unwrap(),
            make_server_config(),
        )
        .unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (recv, _send) = QuicTransport::accept(&listener).await.unwrap();
            let reader = QuicTableReader::from_recv(recv, None);
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let (_recv, send) = QuicTransport::connect(addr, "localhost", make_client_config())
            .await
            .unwrap();
        let mut writer = QuicTableWriter::new(send, table_schema(), None).unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_quic_accepting_writer_to_connecting_reader() {
        let listener = QuicTransport::bind(
            "127.0.0.1:0".parse().unwrap(),
            make_server_config(),
        )
        .unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (_recv, send) = QuicTransport::accept(&listener).await.unwrap();
            let mut writer = QuicTableWriter::new(send, table_schema(), None).unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let (recv, mut send) = QuicTransport::connect(addr, "localhost", make_client_config())
            .await
            .unwrap();
        // A reading-only connecting peer opens the stream by shutting
        // down its unused write half.
        send.shutdown().await.unwrap();
        let reader = QuicTableReader::from_recv(recv, None);
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }
}

#[cfg(feature = "webtransport")]
mod wt {
    use lightstream::models::readers::webtransport::WebTransportTableReader;
    use lightstream::models::transports::webtransport::WebTransport;
    use lightstream::models::writers::webtransport::WebTransportTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use tokio::io::AsyncWriteExt;
    use wtransport::tls::Sha256Digest;
    use wtransport::{ClientConfig, Identity, ServerConfig};

    use super::{assert_tables, make_table, table_schema};

    /// Create a self-signed identity and its certificate hash so the
    /// client can verify the server by certificate pinning.
    fn make_test_identity() -> (Identity, Sha256Digest) {
        let identity = Identity::self_signed(["localhost", "127.0.0.1", "::1"]).unwrap();
        let hash = identity.certificate_chain().as_slice()[0].hash();
        (identity, hash)
    }

    fn make_server_config(identity: Identity) -> ServerConfig {
        ServerConfig::builder()
            .with_bind_default(0)
            .with_identity(identity)
            .build()
    }

    fn make_client_config(server_cert_hash: Sha256Digest) -> ClientConfig {
        ClientConfig::builder()
            .with_bind_default()
            .with_server_certificate_hashes([server_cert_hash])
            .build()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_connecting_writer_to_accepting_reader() {
        let (identity, cert_hash) = make_test_identity();
        let listener = WebTransport::bind(make_server_config(identity)).unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (path, recv, _send) = WebTransport::accept(&listener).await.unwrap();
            assert_eq!(path, "/feed");
            let reader = WebTransportTableReader::from_recv(recv, None);
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let (_recv, send) = WebTransport::connect(
            &format!("https://127.0.0.1:{port}/feed"),
            make_client_config(cert_hash),
        )
        .await
        .unwrap();
        let mut writer = WebTransportTableWriter::new(send, table_schema(), None).unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_accepting_writer_to_connecting_reader() {
        let (identity, cert_hash) = make_test_identity();
        let listener = WebTransport::bind(make_server_config(identity)).unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (_path, _recv, send) = WebTransport::accept(&listener).await.unwrap();
            let mut writer = WebTransportTableWriter::new(send, table_schema(), None).unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.write_table(make_table("b", 5)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let (recv, mut send) = WebTransport::connect(
            &format!("https://127.0.0.1:{port}"),
            make_client_config(cert_hash),
        )
        .await
        .unwrap();
        // A reading-only connecting peer opens the stream by shutting
        // down its unused write half.
        send.shutdown().await.unwrap();
        let reader = WebTransportTableReader::from_recv(recv, None);
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3, 5]);
        server.await.unwrap();
    }
}

#[cfg(all(feature = "websocket", feature = "http", feature = "tls"))]
mod tls_wires {
    use std::sync::Arc;

    use lightstream::models::readers::http::HttpTableReader;
    use lightstream::models::readers::websocket::WebSocketTableReader;
    use lightstream::models::transports::http::HttpTransport;
    use lightstream::models::transports::websocket::WebSocketTransport;
    use lightstream::models::writers::http::HttpTableWriter;
    use lightstream::models::writers::websocket::WebSocketTableWriter;
    use lightstream::enums::IPCMessageProtocol;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use tokio::net::TcpListener;

    use super::{assert_tables, make_table, table_schema};

    /// Self-signed server config plus a client config trusting it, for
    /// the given ALPN set.
    fn make_tls_pair(
        alpn: &[&[u8]],
    ) -> (Arc<rustls::ServerConfig>, Arc<rustls::ClientConfig>) {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
        let key_der =
            rustls::pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();

        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert_der.clone()).unwrap();

        let mut server = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key_der)
            .unwrap();
        server.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();

        let mut client = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        client.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();

        (Arc::new(server), Arc::new(client))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_wss_accepting_writer_to_connecting_reader() {
        let (server_config, client_config) = make_tls_pair(&[]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (_path, read_half, write_half) =
                WebSocketTransport::accept_tls(&listener, server_config)
                    .await
                    .unwrap();
            let mut writer =
                WebSocketTableWriter::from_halves(read_half, write_half, table_schema(), None)
                    .unwrap();
            writer.write_table(make_table("a", 3)).await.unwrap();
            writer.finish().await.unwrap();
        });

        let (read_half, write_half) =
            WebSocketTransport::connect_tls(&format!("wss://localhost:{}", addr.port()), client_config)
                .await
                .unwrap();
        let reader = WebSocketTableReader::from_client_halves(
            read_half,
            write_half,
            IPCMessageProtocol::Stream,
            None,
        );
        let tables = reader.read_all_tables().await.unwrap();
        assert_tables(&tables, &[3]);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_https_connecting_writer_to_accepting_reader() {
        let (server_config, client_config) = make_tls_pair(&[b"h2"]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (path, recv_read, send_write) = HttpTransport::accept_tls(&listener, server_config)
                .await
                .unwrap();
            assert_eq!(path, "/ingest");
            let reader = HttpTableReader::from_exchange(recv_read, send_write, None)
                .await
                .unwrap();
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let (recv_read, send_write) = HttpTransport::connect_tls(
            &format!("https://localhost:{}/ingest", addr.port()),
            client_config,
        )
        .await
        .unwrap();
        let mut writer =
            HttpTableWriter::from_exchange(recv_read, send_write, table_schema(), None).unwrap();
        writer.write_table(make_table("a", 4)).await.unwrap();
        writer.finish().await.unwrap();
        server.await.unwrap();
    }
}

#[cfg(all(feature = "tcp", feature = "uds"))]
mod generic {
    use lightstream::models::transports::tcp::TcpTransport;
    use lightstream::models::transports::uds::UdsTransport;
    use lightstream::traits::transport::Transport;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Push bytes through a transport generically, with the accepting
    /// peer echoing what the connecting peer sends.
    async fn roundtrip_bytes<T: Transport>(listener: &T::Listener, endpoint: &T::Endpoint) {
        let accepted = T::accept(listener);
        let connected = T::connect(endpoint);
        let ((mut a_read, mut a_write), (mut c_read, mut c_write)) =
            tokio::try_join!(accepted, connected).unwrap();

        let payload = b"Transport";
        c_write.write_all(payload).await.unwrap();
        c_write.shutdown().await.unwrap();

        let mut received = Vec::new();
        a_read.read_to_end(&mut received).await.unwrap();
        assert_eq!(received, payload);

        a_write.write_all(&received).await.unwrap();
        a_write.shutdown().await.unwrap();

        let mut echoed = Vec::new();
        c_read.read_to_end(&mut echoed).await.unwrap();
        assert_eq!(echoed, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_generic_roundtrip_over_both_wires() {
        let listener = TcpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        roundtrip_bytes::<TcpTransport>(&listener, &addr).await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generic.sock");
        let listener = UdsTransport::bind(&path).unwrap();
        roundtrip_bytes::<UdsTransport>(&listener, path.as_path()).await;
    }
}
