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
    use std::io;
    use std::net::SocketAddr;

    use lightstream::models::readers::websocket::WebSocketTableReader;
    use lightstream::models::transports::websocket::{WebSocketListener, WebSocketTransport};
    use lightstream::models::writers::websocket::WebSocketTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;

    use super::{assert_tables, make_table, table_schema};

    /// Bind `/prices` on a free port, then `/trades` on the same authority.
    async fn bind_two_paths(scheme: &str) -> (WebSocketListener, WebSocketListener, SocketAddr) {
        let prices = WebSocketTransport::bind(&format!("{scheme}://127.0.0.1:0/prices"))
            .await
            .unwrap();
        let addr = prices.local_addr().unwrap();
        let trades = WebSocketTransport::bind(&format!("{scheme}://{addr}/trades"))
            .await
            .unwrap();
        assert_eq!(trades.local_addr().unwrap(), addr);
        (prices, trades, addr)
    }

    /// Connect to `url`, write one table of `n_rows` rows and finish.
    async fn send_table(url: &str, n_rows: usize) {
        let mut writer = WebSocketTableWriter::connect(url, table_schema(), None)
            .await
            .unwrap();
        writer.write_table(make_table("t", n_rows)).await.unwrap();
        writer.finish().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_listeners_on_one_authority_receive_their_own_paths() {
        let (prices, trades, addr) = bind_two_paths("ws").await;

        // Only the `/prices` listener accepts at first, therefore it
        // upgrades the `/trades` connection and holds it for `/trades`.
        let prices_server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("ws://{addr}/trades"), 5).await;
        send_table(&format!("ws://{addr}/prices"), 3).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);

        let reader = WebSocketTableReader::accept(&trades, None).await.unwrap();
        assert_tables(&reader.read_all_tables().await.unwrap(), &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_listeners_on_one_authority_accept_concurrently() {
        let (prices, trades, addr) = bind_two_paths("ws").await;
        let prices_server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        let trades_server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&trades, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("ws://{addr}/prices"), 3).await;
        send_table(&format!("ws://{addr}/trades"), 5).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);
        assert_tables(&trades_server.await.unwrap(), &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_upgrades_to_unbound_paths_are_refused() {
        let listener = WebSocketTransport::bind("ws://127.0.0.1:0/prices").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&listener, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });

        let error = WebSocketTransport::connect(&format!("ws://{addr}/missing"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);

        send_table(&format!("ws://{addr}/prices"), 4).await;
        assert_tables(&server.await.unwrap(), &[4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_url_without_a_path_is_the_root_path() {
        let listener = WebSocketTransport::bind("ws://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let error = WebSocketTransport::bind(&format!("ws://{addr}/"))
            .await
            .err()
            .expect("the root path is bound");
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse);

        let server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&listener, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("ws://{addr}/"), 2).await;
        assert_tables(&server.await.unwrap(), &[2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_binding_bound_urls_returns_addr_in_use() {
        for scheme in ["ws", "wss"] {
            let (_prices, _trades, addr) = bind_two_paths(scheme).await;
            for path in ["/prices", "/trades"] {
                let error = WebSocketTransport::bind(&format!("{scheme}://{addr}{path}"))
                    .await
                    .err()
                    .expect("the path is bound");
                assert_eq!(error.kind(), io::ErrorKind::AddrInUse, "{scheme} {path}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_dropped_listeners_unbind_their_paths_and_the_last_frees_the_port() {
        for scheme in ["ws", "wss"] {
            let (prices, trades, addr) = bind_two_paths(scheme).await;
            drop(trades);
            let trades = WebSocketTransport::bind(&format!("{scheme}://{addr}/trades"))
                .await
                .unwrap();
            drop(trades);
            drop(prices);
            std::net::TcpListener::bind(addr).unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_websocket_upgrades_to_dropped_paths_are_refused() {
        let (prices, trades, addr) = bind_two_paths("ws").await;
        drop(trades);
        let server = tokio::spawn(async move {
            let reader = WebSocketTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });

        let error = WebSocketTransport::connect(&format!("ws://{addr}/trades"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);

        send_table(&format!("ws://{addr}/prices"), 6).await;
        assert_tables(&server.await.unwrap(), &[6]);
    }

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
}

#[cfg(feature = "http")]
mod http {
    use std::io;
    use std::net::SocketAddr;

    use ::http::{Request, StatusCode};
    use lightstream::models::readers::http::HttpTableReader;
    use lightstream::models::transports::http::{HttpListener, HttpTransport};
    use lightstream::models::writers::http::HttpTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use tokio::net::TcpStream;

    use super::{assert_tables, make_table, table_schema};

    /// Bind `/prices` on a free port, then `/trades` on the same authority.
    async fn bind_two_paths(scheme: &str) -> (HttpListener, HttpListener, SocketAddr) {
        let prices = HttpTransport::bind(&format!("{scheme}://127.0.0.1:0/prices"))
            .await
            .unwrap();
        let addr = prices.local_addr().unwrap();
        let trades = HttpTransport::bind(&format!("{scheme}://{addr}/trades"))
            .await
            .unwrap();
        assert_eq!(trades.local_addr().unwrap(), addr);
        (prices, trades, addr)
    }

    /// POST one table of `n_rows` rows to `url` and finish.
    async fn send_table(url: &str, n_rows: usize) {
        let mut writer = HttpTableWriter::post(url, table_schema(), None).await.unwrap();
        writer.write_table(make_table("t", n_rows)).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// POST an empty body to `url` over h2c and return the response status.
    async fn post_status(url: &str) -> StatusCode {
        let request = Request::post(url).body(()).unwrap();
        let authority = request.uri().authority().unwrap().to_string();
        let tcp = TcpStream::connect(authority).await.unwrap();
        let (mut send_request, connection) = h2::client::handshake(tcp).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let (response, _body) = send_request.send_request(request, true).unwrap();
        response.await.unwrap().status()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_listeners_on_one_authority_receive_their_own_paths() {
        let (prices, trades, addr) = bind_two_paths("http").await;

        // Only the `/prices` listener accepts at first, therefore it
        // holds the `/trades` request for `/trades`.
        let prices_server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("http://{addr}/trades"), 5).await;
        send_table(&format!("http://{addr}/prices"), 3).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);

        let reader = HttpTableReader::accept(&trades, None).await.unwrap();
        assert_tables(&reader.read_all_tables().await.unwrap(), &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_listeners_on_one_authority_accept_concurrently() {
        let (prices, trades, addr) = bind_two_paths("http").await;
        let prices_server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        let trades_server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&trades, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("http://{addr}/prices"), 3).await;
        send_table(&format!("http://{addr}/trades"), 5).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);
        assert_tables(&trades_server.await.unwrap(), &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_requests_to_unbound_paths_are_refused() {
        let listener = HttpTransport::bind("http://127.0.0.1:0/prices").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&listener, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });

        assert_eq!(
            post_status(&format!("http://{addr}/missing")).await,
            StatusCode::NOT_FOUND
        );

        send_table(&format!("http://{addr}/prices"), 4).await;
        assert_tables(&server.await.unwrap(), &[4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_url_without_a_path_is_the_root_path() {
        let listener = HttpTransport::bind("http://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let error = HttpTransport::bind(&format!("http://{addr}/"))
            .await
            .err()
            .expect("the root path is bound");
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse);

        let server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&listener, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });
        send_table(&format!("http://{addr}/"), 2).await;
        assert_tables(&server.await.unwrap(), &[2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_binding_bound_urls_returns_addr_in_use() {
        for scheme in ["http", "https"] {
            let (_prices, _trades, addr) = bind_two_paths(scheme).await;
            for path in ["/prices", "/trades"] {
                let error = HttpTransport::bind(&format!("{scheme}://{addr}{path}"))
                    .await
                    .err()
                    .expect("the path is bound");
                assert_eq!(error.kind(), io::ErrorKind::AddrInUse, "{scheme} {path}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_dropped_listeners_unbind_their_paths_and_the_last_frees_the_port() {
        for scheme in ["http", "https"] {
            let (prices, trades, addr) = bind_two_paths(scheme).await;
            drop(trades);
            let trades = HttpTransport::bind(&format!("{scheme}://{addr}/trades"))
                .await
                .unwrap();
            drop(trades);
            drop(prices);
            std::net::TcpListener::bind(addr).unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_requests_to_dropped_paths_are_refused() {
        let (prices, trades, addr) = bind_two_paths("http").await;
        drop(trades);
        let server = tokio::spawn(async move {
            let reader = HttpTableReader::accept(&prices, None).await.unwrap();
            reader.read_all_tables().await.unwrap()
        });

        assert_eq!(
            post_status(&format!("http://{addr}/trades")).await,
            StatusCode::NOT_FOUND
        );

        send_table(&format!("http://{addr}/prices"), 6).await;
        assert_tables(&server.await.unwrap(), &[6]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_http_accepting_writer_to_connecting_reader() {
        let listener = HttpTransport::bind("http://127.0.0.1:0/feed").await.unwrap();
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
        let listener = HttpTransport::bind("http://127.0.0.1:0/ingest").await.unwrap();
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
    use std::io;
    use std::net::{SocketAddr, UdpSocket};
    use std::time::{Duration, Instant};

    use lightstream::models::readers::webtransport::WebTransportTableReader;
    use lightstream::models::transports::webtransport::{WebTransport, WebTransportListener};
    use lightstream::models::writers::webtransport::WebTransportTableWriter;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use tokio::io::AsyncWriteExt;
    use wtransport::tls::Sha256Digest;
    use minarrow::Table;
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

    /// Server config bound to a free loopback port, matching the
    /// `https://127.0.0.1:0` URLs of the path tests.
    fn make_loopback_server_config(identity: Identity) -> ServerConfig {
        ServerConfig::builder()
            .with_bind_address(SocketAddr::from(([127, 0, 0, 1], 0)))
            .with_identity(identity)
            .build()
    }

    /// Bind `/prices` on a free port, then `/trades` on the same authority,
    /// and return the certificate hash clients pin.
    fn bind_two_paths() -> (WebTransportListener, WebTransportListener, SocketAddr, Sha256Digest) {
        let (identity, cert_hash) = make_test_identity();
        let prices = WebTransport::bind(
            "https://127.0.0.1:0/prices",
            make_loopback_server_config(identity),
        )
        .unwrap();
        let addr = prices.local_addr().unwrap();
        // The second bind shares the endpoint, and its config is unused.
        let (identity, _) = make_test_identity();
        let trades = WebTransport::bind(
            &format!("https://{addr}/trades"),
            make_loopback_server_config(identity),
        )
        .unwrap();
        assert_eq!(trades.local_addr().unwrap(), addr);
        (prices, trades, addr, cert_hash)
    }

    /// Open a session on `url`, write one table of `n_rows` rows and finish.
    async fn send_table(url: String, cert_hash: Sha256Digest, n_rows: usize) {
        let (_recv, send) = WebTransport::connect(&url, make_client_config(cert_hash))
            .await
            .unwrap();
        let mut writer = WebTransportTableWriter::new(send, table_schema(), None).unwrap();
        writer.write_table(make_table("t", n_rows)).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// Accept one session on `listener` and read its tables.
    async fn receive_tables(listener: &WebTransportListener) -> Vec<Table> {
        let (recv, _send) = WebTransport::accept(listener).await.unwrap();
        WebTransportTableReader::from_recv(recv, None)
            .read_all_tables()
            .await
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_listeners_on_one_authority_receive_their_own_paths() {
        let (prices, trades, addr, cert_hash) = bind_two_paths();

        // Only the `/prices` listener accepts at first, therefore it holds
        // the `/trades` session request for `/trades`. That session opens
        // once `/trades` accepts, so its client runs on a task.
        let prices_server = tokio::spawn(async move { receive_tables(&prices).await });
        let trades_client = tokio::spawn(send_table(format!("https://{addr}/trades"), cert_hash.clone(), 5));
        send_table(format!("https://{addr}/prices"), cert_hash, 3).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);

        assert_tables(&receive_tables(&trades).await, &[5]);
        trades_client.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_sessions_for_unbound_paths_are_rejected() {
        let (identity, cert_hash) = make_test_identity();
        let listener = WebTransport::bind(
            "https://127.0.0.1:0/prices",
            make_loopback_server_config(identity),
        )
        .unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { receive_tables(&listener).await });

        let rejected = WebTransport::connect(
            &format!("https://{addr}/missing"),
            make_client_config(cert_hash.clone()),
        )
        .await;
        let error = rejected.err().expect("the session is rejected");
        assert_eq!(error.to_string(), "server rejected WebTransport session request");

        send_table(format!("https://{addr}/prices"), cert_hash, 4).await;
        assert_tables(&server.await.unwrap(), &[4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_url_without_a_path_is_the_root_path() {
        let (identity, cert_hash) = make_test_identity();
        let listener =
            WebTransport::bind("https://127.0.0.1:0", make_loopback_server_config(identity))
                .unwrap();
        let addr = listener.local_addr().unwrap();
        let (identity, _) = make_test_identity();
        let error = WebTransport::bind(
            &format!("https://{addr}/"),
            make_loopback_server_config(identity),
        )
        .err()
        .expect("the root path is bound");
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse);

        let server = tokio::spawn(async move { receive_tables(&listener).await });
        send_table(format!("https://{addr}/"), cert_hash, 2).await;
        assert_tables(&server.await.unwrap(), &[2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_binding_bound_urls_returns_addr_in_use() {
        let (_prices, _trades, addr, _) = bind_two_paths();
        for path in ["/prices", "/trades"] {
            let (identity, _) = make_test_identity();
            let error = WebTransport::bind(
                &format!("https://{addr}{path}"),
                make_loopback_server_config(identity),
            )
            .err()
            .expect("the path is bound");
            assert_eq!(error.kind(), io::ErrorKind::AddrInUse, "{path}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_dropped_listeners_unbind_their_paths_and_the_last_frees_the_port() {
        let (prices, trades, addr, _) = bind_two_paths();
        drop(trades);
        let (identity, _) = make_test_identity();
        let trades = WebTransport::bind(
            &format!("https://{addr}/trades"),
            make_loopback_server_config(identity),
        )
        .unwrap();
        drop(trades);
        drop(prices);

        // The endpoint's driver closes the socket once it observes the drop.
        let deadline = Instant::now() + Duration::from_secs(5);
        while UdpSocket::bind(addr).is_err() {
            assert!(Instant::now() < deadline, "the port stays bound");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_sessions_for_dropped_paths_are_rejected() {
        let (prices, trades, addr, cert_hash) = bind_two_paths();
        drop(trades);
        let server = tokio::spawn(async move { receive_tables(&prices).await });

        let rejected = WebTransport::connect(
            &format!("https://{addr}/trades"),
            make_client_config(cert_hash.clone()),
        )
        .await;
        let error = rejected.err().expect("the session is rejected");
        assert_eq!(error.to_string(), "server rejected WebTransport session request");

        send_table(format!("https://{addr}/prices"), cert_hash, 6).await;
        assert_tables(&server.await.unwrap(), &[6]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_webtransport_connecting_writer_to_accepting_reader() {
        let (identity, cert_hash) = make_test_identity();
        let listener =
            WebTransport::bind("https://127.0.0.1:0", make_server_config(identity)).unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (recv, _send) = WebTransport::accept(&listener).await.unwrap();
            let reader = WebTransportTableReader::from_recv(recv, None);
            let tables = reader.read_all_tables().await.unwrap();
            assert_tables(&tables, &[4]);
        });

        let (_recv, send) = WebTransport::connect(
            &format!("https://127.0.0.1:{port}"),
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
        let listener =
            WebTransport::bind("https://127.0.0.1:0", make_server_config(identity)).unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (_recv, send) = WebTransport::accept(&listener).await.unwrap();
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
    use std::io;
    use std::sync::Arc;

    use ::http::{Request, StatusCode};
    use lightstream::models::readers::http::HttpTableReader;
    use lightstream::models::readers::websocket::WebSocketTableReader;
    use lightstream::models::transports::http::{HttpListener, HttpTransport};
    use lightstream::models::transports::websocket::{WebSocketListener, WebSocketTransport};
    use lightstream::models::writers::http::HttpTableWriter;
    use lightstream::models::writers::websocket::WebSocketTableWriter;
    use lightstream::enums::IPCMessageProtocol;
    use lightstream::traits::transport_reader::IPCTransportReader;
    use lightstream::traits::transport_writer::IPCTransportWriter;
    use minarrow::Table;
    use tokio::net::TcpStream;

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

    /// Accept one wss connection on `listener` and read its tables.
    async fn receive_wss_tables(
        listener: &WebSocketListener,
        config: Arc<rustls::ServerConfig>,
    ) -> Vec<Table> {
        let (read_half, write_half) = WebSocketTransport::accept_tls(listener, config)
            .await
            .unwrap();
        WebSocketTableReader::from_halves(read_half, write_half, IPCMessageProtocol::Stream, None)
            .read_all_tables()
            .await
            .unwrap()
    }

    /// Connect to a `wss://` URL, write one table of `n_rows` rows and finish.
    async fn send_wss_table(url: &str, config: Arc<rustls::ClientConfig>, n_rows: usize) {
        let mut writer = WebSocketTableWriter::connect_tls(url, config, table_schema(), None)
            .await
            .unwrap();
        writer.write_table(make_table("t", n_rows)).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// Accept one https exchange on `listener` and read its tables.
    async fn receive_https_tables(
        listener: &HttpListener,
        config: Arc<rustls::ServerConfig>,
    ) -> Vec<Table> {
        let (recv_read, send_write) = HttpTransport::accept_tls(listener, config).await.unwrap();
        HttpTableReader::from_exchange(recv_read, send_write, None)
            .await
            .unwrap()
            .read_all_tables()
            .await
            .unwrap()
    }

    /// POST one table of `n_rows` rows to an `https://` URL and finish.
    async fn send_https_table(url: &str, config: Arc<rustls::ClientConfig>, n_rows: usize) {
        let mut writer = HttpTableWriter::post_tls(url, config, table_schema(), None)
            .await
            .unwrap();
        writer.write_table(make_table("t", n_rows)).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// POST an empty body to an `https://localhost` URL and return the
    /// response status.
    async fn post_tls_status(url: &str, config: Arc<rustls::ClientConfig>) -> StatusCode {
        let request = Request::post(url).body(()).unwrap();
        let port = request.uri().port_u16().unwrap();
        let tcp = TcpStream::connect(("localhost", port)).await.unwrap();
        let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let tls = tokio_rustls::TlsConnector::from(config)
            .connect(server_name, tcp)
            .await
            .unwrap();
        let (mut send_request, connection) = h2::client::handshake(tls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let (response, _body) = send_request.send_request(request, true).unwrap();
        response.await.unwrap().status()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_wss_listeners_on_one_authority_receive_their_own_paths() {
        let (server_config, client_config) = make_tls_pair(&[]);
        let prices = WebSocketTransport::bind("wss://127.0.0.1:0/prices").await.unwrap();
        let port = prices.local_addr().unwrap().port();
        let trades = WebSocketTransport::bind(&format!("wss://127.0.0.1:{port}/trades"))
            .await
            .unwrap();

        // The `/prices` accept terminates TLS for the `/trades` connection
        // too, because both paths share the authority's TLS identity.
        let prices_config = server_config.clone();
        let prices_server =
            tokio::spawn(async move { receive_wss_tables(&prices, prices_config).await });
        let base = format!("wss://localhost:{port}");
        send_wss_table(&format!("{base}/trades"), client_config.clone(), 5).await;
        send_wss_table(&format!("{base}/prices"), client_config, 3).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);
        assert_tables(&receive_wss_tables(&trades, server_config).await, &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_wss_upgrades_to_unbound_paths_are_refused() {
        let (server_config, client_config) = make_tls_pair(&[]);
        let listener = WebSocketTransport::bind("wss://127.0.0.1:0/prices").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server =
            tokio::spawn(async move { receive_wss_tables(&listener, server_config).await });

        let base = format!("wss://localhost:{port}");
        let error = WebSocketTransport::connect_tls(&format!("{base}/missing"), client_config.clone())
            .await
            .err()
            .expect("the upgrade is refused");
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);

        send_wss_table(&format!("{base}/prices"), client_config, 4).await;
        assert_tables(&server.await.unwrap(), &[4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_wss_url_without_a_path_is_the_root_path() {
        let (server_config, client_config) = make_tls_pair(&[]);
        let listener = WebSocketTransport::bind("wss://127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server =
            tokio::spawn(async move { receive_wss_tables(&listener, server_config).await });
        send_wss_table(&format!("wss://localhost:{port}/"), client_config, 2).await;
        assert_tables(&server.await.unwrap(), &[2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_https_listeners_on_one_authority_receive_their_own_paths() {
        let (server_config, client_config) = make_tls_pair(&[b"h2"]);
        let prices = HttpTransport::bind("https://127.0.0.1:0/prices").await.unwrap();
        let port = prices.local_addr().unwrap().port();
        let trades = HttpTransport::bind(&format!("https://127.0.0.1:{port}/trades"))
            .await
            .unwrap();

        // The `/prices` accept terminates TLS for the `/trades` request
        // too, because both paths share the authority's TLS identity.
        let prices_config = server_config.clone();
        let prices_server =
            tokio::spawn(async move { receive_https_tables(&prices, prices_config).await });
        let base = format!("https://localhost:{port}");
        send_https_table(&format!("{base}/trades"), client_config.clone(), 5).await;
        send_https_table(&format!("{base}/prices"), client_config, 3).await;
        assert_tables(&prices_server.await.unwrap(), &[3]);
        assert_tables(&receive_https_tables(&trades, server_config).await, &[5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_https_requests_to_unbound_paths_are_refused() {
        let (server_config, client_config) = make_tls_pair(&[b"h2"]);
        let listener = HttpTransport::bind("https://127.0.0.1:0/prices").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server =
            tokio::spawn(async move { receive_https_tables(&listener, server_config).await });

        let base = format!("https://localhost:{port}");
        assert_eq!(
            post_tls_status(&format!("{base}/missing"), client_config.clone()).await,
            StatusCode::NOT_FOUND
        );

        send_https_table(&format!("{base}/prices"), client_config, 4).await;
        assert_tables(&server.await.unwrap(), &[4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_https_url_without_a_path_is_the_root_path() {
        let (server_config, client_config) = make_tls_pair(&[b"h2"]);
        let listener = HttpTransport::bind("https://127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server =
            tokio::spawn(async move { receive_https_tables(&listener, server_config).await });
        send_https_table(&format!("https://localhost:{port}/"), client_config, 2).await;
        assert_tables(&server.await.unwrap(), &[2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_wss_accepting_writer_to_connecting_reader() {
        let (server_config, client_config) = make_tls_pair(&[]);
        let listener = WebSocketTransport::bind("wss://127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (read_half, write_half) =
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
        let listener = HttpTransport::bind("https://127.0.0.1:0/ingest").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (recv_read, send_write) = HttpTransport::accept_tls(&listener, server_config)
                .await
                .unwrap();
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
