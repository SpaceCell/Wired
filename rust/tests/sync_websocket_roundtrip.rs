// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Synchronous WebSocket round-trip integration tests.
//!
//! Tests `SyncWebSocketTableReader` and `SyncWebSocketTableWriter` together
//! and with their asynchronous counterparts, `WebSocketTableReader` and
//! `WebSocketTableWriter`. Covers client and server roles, with and without
//! compression.
//!
//! Tests text and binary message reads through `SyncWebSocketTransport` and
//! `SyncWsRead` against synchronous `tungstenite` peers.
//!
//! Synchronous peers run on dedicated threads without an async runtime.
//! Asynchronous peers run on dedicated threads with current-thread runtimes.

#![cfg(feature = "websocket")]

use std::io::{self, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use futures_util::StreamExt;
use lightstream::compression::Compression;
use lightstream::models::decoders::limits::DecodeLimits;
use lightstream::models::frames::websocket::{
    MAX_HEADER_LEN, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_TEXT, parse_header,
    write_masked_binary_header,
};
use lightstream::models::readers::sync_websocket::SyncWebSocketTableReader;
use lightstream::models::readers::websocket::WebSocketTableReader;
use lightstream::models::streams::sync_websocket::{SyncWsRead, SyncWsWrite};
use lightstream::models::transports::sync_websocket::SyncWebSocketTransport;
use lightstream::models::writers::sync_websocket::SyncWebSocketTableWriter;
use lightstream::models::writers::websocket::WebSocketTableWriter;
use lightstream::traits::transport_reader::IPCTransportReader;
use lightstream::traits::transport_writer::IPCTransportWriter;
use minarrow::{
    Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, FloatArray,
    IntegerArray, NumericArray, StringArray, Table, TextArray, Vec64, arr_f64, arr_i64,
    ffi::arrow_dtype::CategoricalIndexType,
};
use tokio_tungstenite::tungstenite::{self, Message};

/// Dictionary id of the categorical column in `make_mixed_table`.
const DICT_ID: i64 = 3;

/// Message size cap used by the message-read tests.
const MAX_MESSAGE: usize = 1 << 20;

fn dictionary_values() -> Vec<String> {
    vec!["red".to_string(), "green".to_string(), "blue".to_string()]
}

/// Four-row table with integer, float, string and categorical columns.
fn make_mixed_table() -> Table {
    let int_col = FieldArray::new(
        Field::new("ids", ArrowType::Int32, false, None),
        Array::NumericArray(NumericArray::Int32(Arc::new(IntegerArray {
            data: Buffer::from(Vec64::from_slice(&[10, 20, 30, 40])),
            null_mask: None,
        }))),
    );
    let float_col = FieldArray::new(
        Field::new("values", ArrowType::Float64, false, None),
        Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
            data: Buffer::from(Vec64::from_slice(&[1.1, 2.2, 3.3, 4.4])),
            null_mask: None,
        }))),
    );
    let str_col = FieldArray::new(
        Field::new("labels", ArrowType::String, true, None),
        Array::TextArray(TextArray::String32(Arc::new(StringArray::new(
            Buffer::from(Vec64::from_slice("helloworldtest".as_bytes())),
            Some(Bitmask::new_set_all(4, true)),
            Buffer::from(Vec64::from_slice(&[0u32, 5, 10, 14, 14])),
        )))),
    );

    #[cfg(not(feature = "default_categorical_8"))]
    let dict_col = FieldArray::new(
        Field::new(
            "category",
            ArrowType::Dictionary(CategoricalIndexType::UInt32),
            true,
            None,
        ),
        Array::TextArray(TextArray::Categorical32(Arc::new(CategoricalArray {
            data: Buffer::from(Vec64::from_slice(&[0u32, 1, 2, 0])),
            unique_values: Vec64::from(dictionary_values()),
            null_mask: Some(Bitmask::new_set_all(4, true)),
        }))),
    );
    #[cfg(feature = "default_categorical_8")]
    let dict_col = FieldArray::new(
        Field::new(
            "category",
            ArrowType::Dictionary(CategoricalIndexType::UInt8),
            true,
            None,
        ),
        Array::TextArray(TextArray::Categorical8(Arc::new(CategoricalArray {
            data: Buffer::from(Vec64::from_slice(&[0u8, 1, 2, 0])),
            unique_values: Vec64::from(dictionary_values()),
            null_mask: Some(Bitmask::new_set_all(4, true)),
        }))),
    );

    Table::new(
        "mixed".to_string(),
        Some(vec![int_col, float_col, str_col, dict_col]),
    )
}

/// Numeric table large enough that record batch bodies span many frames and
/// socket reads.
fn make_large_table(n_rows: usize) -> Table {
    let ids: Vec64<i64> = (0..n_rows as i64).collect();
    let values: Vec64<f64> = (0..n_rows).map(|i| i as f64 * 0.25).collect();
    Table::new(
        "large".to_string(),
        Some(vec![
            FieldArray::from_arr("id", arr_i64!(ids)),
            FieldArray::from_arr("value", arr_f64!(values)),
        ]),
    )
}

fn schema_of(table: &Table) -> Vec<Field> {
    table.cols.iter().map(|fa| fa.field.as_ref().clone()).collect()
}

/// Compression settings covered by the roundtrips. Zstd is included when the
/// `zstd` feature is enabled.
fn compressions() -> Vec<Option<Compression>> {
    #[allow(unused_mut)]
    let mut out = vec![None];
    #[cfg(feature = "zstd")]
    out.push(Some(Compression::Zstd));
    out
}

#[derive(Clone, Copy, Debug)]
enum Side {
    Sync,
    Async,
}

/// Peer that binds the address and accepts the connection.
#[derive(Clone, Copy, Debug)]
enum Listening {
    Writer,
    Reader,
}

/// Connection role of one peer.
enum Endpoint {
    Connect(String),
    Accept(TcpListener),
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Bind a loopback listener and return it with its `ws://` URL.
fn bind_local() -> (TcpListener, String) {
    let listener = SyncWebSocketTransport::bind("ws://127.0.0.1:0").unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    (listener, url)
}

fn to_tokio_listener(listener: TcpListener) -> io::Result<tokio::net::TcpListener> {
    listener.set_nonblocking(true)?;
    tokio::net::TcpListener::from_std(listener)
}

fn write_tables(
    side: Side,
    endpoint: Endpoint,
    schema: Vec<Field>,
    tables: Vec<Table>,
    compression: Option<Compression>,
) {
    match side {
        Side::Sync => {
            // The sync writer registers dictionaries from categorical columns.
            let mut writer = match endpoint {
                Endpoint::Connect(url) => {
                    SyncWebSocketTableWriter::connect(&url, schema, compression).unwrap()
                }
                Endpoint::Accept(listener) => {
                    SyncWebSocketTableWriter::accept(&listener, schema, compression).unwrap()
                }
            };
            writer.write_all_tables(tables).unwrap();
        }
        Side::Async => current_thread_runtime().block_on(async move {
            let mut writer = match endpoint {
                Endpoint::Connect(url) => WebSocketTableWriter::connect(&url, schema, compression)
                    .await
                    .unwrap(),
                Endpoint::Accept(listener) => {
                    let listener = to_tokio_listener(listener).unwrap();
                    WebSocketTableWriter::accept(&listener, schema, compression)
                        .await
                        .unwrap()
                }
            };
            writer.register_dictionary(DICT_ID, dictionary_values());
            for table in tables {
                writer.write_table(table).await.unwrap();
            }
            writer.finish().await.unwrap();
        }),
    }
}

fn read_tables(
    side: Side,
    endpoint: Endpoint,
    limits: Option<DecodeLimits>,
) -> io::Result<Vec<Table>> {
    match side {
        Side::Sync => {
            let reader = match endpoint {
                Endpoint::Connect(url) => SyncWebSocketTableReader::connect(&url, limits)?,
                Endpoint::Accept(listener) => SyncWebSocketTableReader::accept(&listener, limits)?,
            };
            reader.read_all_tables()
        }
        Side::Async => current_thread_runtime().block_on(async move {
            let reader = match endpoint {
                Endpoint::Connect(url) => WebSocketTableReader::connect(&url, limits).await?,
                Endpoint::Accept(listener) => {
                    let listener = to_tokio_listener(listener)?;
                    WebSocketTableReader::accept(&listener, limits).await?
                }
            };
            reader.read_all_tables().await
        }),
    }
}

/// Send `tables` from `writer` to `reader` over one WebSocket connection and
/// return what the reader decoded.
fn transfer(
    writer: Side,
    reader: Side,
    listening: Listening,
    tables: Vec<Table>,
    compression: Option<Compression>,
    limits: Option<DecodeLimits>,
) -> io::Result<Vec<Table>> {
    let (listener, url) = bind_local();
    let schema = schema_of(&tables[0]);

    let (writer_end, reader_end) = match listening {
        Listening::Writer => (Endpoint::Accept(listener), Endpoint::Connect(url)),
        Listening::Reader => (Endpoint::Connect(url), Endpoint::Accept(listener)),
    };

    let writer_thread =
        thread::spawn(move || write_tables(writer, writer_end, schema, tables, compression));
    let result = read_tables(reader, reader_end, limits);
    // A rejecting reader closes the connection mid-stream, which may fail
    // the writer. Only a successful read requires a clean writer.
    let writer_outcome = writer_thread.join();
    if result.is_ok() {
        writer_outcome.unwrap();
    }
    result
}

fn assert_equal_contents(got: &[Table], expected: &[Table]) {
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected) {
        assert_eq!(g.n_rows, e.n_rows);
        assert_eq!(g.cols, e.cols);
    }
}

fn roundtrip(writer: Side, reader: Side, listening: Listening) {
    let tables = vec![
        make_mixed_table(),
        make_mixed_table(),
        make_mixed_table(),
    ];
    for compression in compressions() {
        let got = transfer(writer, reader, listening, tables.clone(), compression, None)
            .unwrap_or_else(|e| {
                panic!("{writer:?} -> {reader:?}, {listening:?} listening, {compression:?}: {e}")
            });
        assert_equal_contents(&got, &tables);
    }
}

#[test]
fn sync_writer_to_async_reader_with_writer_listening() {
    roundtrip(Side::Sync, Side::Async, Listening::Writer);
}

#[test]
fn sync_writer_to_async_reader_with_reader_listening() {
    roundtrip(Side::Sync, Side::Async, Listening::Reader);
}

#[test]
fn async_writer_to_sync_reader_with_writer_listening() {
    roundtrip(Side::Async, Side::Sync, Listening::Writer);
}

#[test]
fn async_writer_to_sync_reader_with_reader_listening() {
    roundtrip(Side::Async, Side::Sync, Listening::Reader);
}

#[test]
fn sync_writer_to_sync_reader_with_writer_listening() {
    roundtrip(Side::Sync, Side::Sync, Listening::Writer);
}

#[test]
fn sync_writer_to_sync_reader_with_reader_listening() {
    roundtrip(Side::Sync, Side::Sync, Listening::Reader);
}

#[test]
fn large_batches_cross_in_both_directions_and_roles() {
    let tables = vec![make_large_table(250_000), make_large_table(100_000)];
    for (writer, reader) in [(Side::Sync, Side::Async), (Side::Async, Side::Sync)] {
        for listening in [Listening::Writer, Listening::Reader] {
            for compression in compressions() {
                let got = transfer(
                    writer,
                    reader,
                    listening,
                    tables.clone(),
                    compression,
                    None,
                )
                .unwrap();
                assert_equal_contents(&got, &tables);
            }
        }
    }
}

#[test]
fn sync_reader_rejects_input_beyond_decode_limits_as_the_async_reader_does() {
    let tables = vec![make_mixed_table()];
    let limits = DecodeLimits {
        max_n_rows: 3,
        ..DecodeLimits::default()
    };

    let sync_err = transfer(
        Side::Sync,
        Side::Sync,
        Listening::Reader,
        tables.clone(),
        None,
        Some(limits),
    )
    .unwrap_err();
    let async_err = transfer(
        Side::Sync,
        Side::Async,
        Listening::Reader,
        tables,
        None,
        Some(limits),
    )
    .unwrap_err();

    assert_eq!(sync_err.kind(), io::ErrorKind::InvalidData);
    assert_eq!(sync_err.kind(), async_err.kind());
    assert_eq!(sync_err.to_string(), async_err.to_string());
}

#[test]
fn sync_reader_iterates_batches_and_reports_schema() {
    let (listener, url) = bind_local();
    let table = make_mixed_table();
    let schema = schema_of(&table);

    let write_table = table.clone();
    let writer_thread = thread::spawn(move || {
        let mut writer = SyncWebSocketTableWriter::accept(&listener, schema, None).unwrap();
        writer.write_table(write_table.clone()).unwrap();
        writer.write_table(write_table).unwrap();
        writer.finish().unwrap();
    });

    let mut reader = SyncWebSocketTableReader::connect(&url, None).unwrap();
    assert!(reader.schema().is_none());
    let first = reader.next().unwrap().unwrap();
    assert_eq!(first.cols, table.cols);
    assert_eq!(reader.schema().unwrap().len(), table.cols.len());
    let rest: Vec<Table> = reader.map(Result::unwrap).collect();
    assert_eq!(rest.len(), 1);
    writer_thread.join().unwrap();
}

#[test]
fn async_reader_stream_receives_sync_writer_batches() {
    let (listener, url) = bind_local();
    let table = make_mixed_table();
    let schema = schema_of(&table);

    let write_table = table.clone();
    let writer_thread = thread::spawn(move || {
        let mut writer = SyncWebSocketTableWriter::accept(&listener, schema, None).unwrap();
        writer.write_table(write_table.clone()).unwrap();
        writer.write_table(write_table).unwrap();
        writer.finish().unwrap();
    });

    let count = current_thread_runtime().block_on(async {
        let mut reader = WebSocketTableReader::connect(&url, None).await.unwrap();
        let mut count = 0;
        while let Some(batch) = reader.next().await {
            assert_eq!(batch.unwrap().cols, table.cols);
            count += 1;
        }
        count
    });
    assert_eq!(count, 2);
    writer_thread.join().unwrap();
}

/// Read every message a writer sends, through the sync transport and
/// `SyncWsRead::read_message`, in the reader role given by `listening`.
fn capture_messages(
    writer: Side,
    listening: Listening,
    tables: Vec<Table>,
    compression: Option<Compression>,
) -> Vec<(u8, Vec<u8>)> {
    let (listener, url) = bind_local();
    let schema = schema_of(&tables[0]);
    let (writer_end, reader_listener) = match listening {
        Listening::Writer => (Endpoint::Accept(listener), None),
        Listening::Reader => (Endpoint::Connect(url.clone()), Some(listener)),
    };
    let writer_thread =
        thread::spawn(move || write_tables(writer, writer_end, schema, tables, compression));

    let mut ws_read = match reader_listener {
        Some(listener) => {
            let (read_half, write_half) = SyncWebSocketTransport::accept(&listener).unwrap();
            let (shared, _ws_write) = SyncWsWrite::new(write_half);
            SyncWsRead::new(read_half, shared)
        }
        None => {
            let (read_half, write_half) = SyncWebSocketTransport::connect(&url).unwrap();
            let (shared, _ws_write) = SyncWsWrite::new_client(write_half);
            SyncWsRead::new_client(read_half, shared)
        }
    };
    let mut messages = Vec::new();
    let mut buf = Vec::new();
    while let Some(opcode) = ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap() {
        messages.push((opcode, buf.clone()));
    }
    writer_thread.join().unwrap();
    messages
}

#[test]
fn sync_and_async_writers_send_identical_messages_in_both_roles() {
    let cases = [
        vec![make_mixed_table(), make_mixed_table()],
        vec![make_large_table(10_000), make_large_table(3)],
    ];
    for tables in cases {
        for listening in [Listening::Writer, Listening::Reader] {
            for compression in compressions() {
                let sync_messages =
                    capture_messages(Side::Sync, listening, tables.clone(), compression);
                let async_messages =
                    capture_messages(Side::Async, listening, tables.clone(), compression);
                // One message per table, then one for the end-of-stream marker.
                assert_eq!(sync_messages.len(), tables.len() + 1);
                assert!(sync_messages.iter().all(|(op, _)| *op == OPCODE_BINARY));
                assert_eq!(sync_messages, async_messages, "{listening:?} {compression:?}");
            }
        }
    }
}

#[test]
fn sync_transport_reads_text_and_binary_messages_from_a_connecting_peer() {
    let (listener, url) = bind_local();
    let large: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let sent = large.clone();
    let peer = thread::spawn(move || {
        let (mut socket, _) = tungstenite::connect(url.as_str()).unwrap();
        socket.send(Message::Text("first quote".into())).unwrap();
        socket.send(Message::Ping(vec![7u8, 8, 9].into())).unwrap();
        socket.send(Message::Binary(sent.into())).unwrap();
        socket.send(Message::Text("".into())).unwrap();
        socket.close(None).unwrap();
        // The pong answering the ping arrives before the close reply.
        let mut pongs = Vec::new();
        loop {
            match socket.read() {
                Ok(Message::Pong(payload)) => pongs.push(payload.to_vec()),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        pongs
    });

    let (read_half, write_half) = SyncWebSocketTransport::accept(&listener).unwrap();
    let (shared, ws_write) = SyncWsWrite::new(write_half);
    let mut ws_read = SyncWsRead::new(read_half, shared);
    let mut buf = Vec::new();

    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_TEXT));
    assert_eq!(buf, b"first quote");
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_BINARY));
    assert_eq!(buf, large);
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_TEXT));
    assert!(buf.is_empty());
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), None);
    // Closing every handle to the socket ends the peer's read loop.
    drop(ws_read);
    drop(ws_write);

    assert_eq!(peer.join().unwrap(), vec![vec![7u8, 8, 9]]);
}

#[test]
fn sync_transport_reads_messages_from_an_accepting_peer() {
    let (listener, url) = bind_local();
    let peer = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut socket = tungstenite::accept(stream).unwrap();
        socket.send(Message::Binary(vec![1u8, 2, 3].into())).unwrap();
        socket.send(Message::Text("tick".into())).unwrap();
        socket.close(None).unwrap();
        while socket.read().is_ok() {}
    });

    let (read_half, write_half) = SyncWebSocketTransport::connect(&url).unwrap();
    let (shared, ws_write) = SyncWsWrite::new_client(write_half);
    let mut ws_read = SyncWsRead::new_client(read_half, shared);
    let mut buf = Vec::new();

    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_BINARY));
    assert_eq!(buf, [1, 2, 3]);
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_TEXT));
    assert_eq!(buf, b"tick");
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), None);
    // Closing every handle to the socket ends the peer's read loop.
    drop(ws_read);
    drop(ws_write);
    peer.join().unwrap();
}

/// Write one masked frame with the given first header byte.
fn write_masked_frame(stream: &mut impl Write, first_byte: u8, payload: &[u8]) {
    let key = [0x5Au8, 0xC3, 0x0F, 0x99];
    let mut header = [0u8; MAX_HEADER_LEN];
    let n = write_masked_binary_header(&mut header, payload.len(), key);
    header[0] = first_byte;
    stream.write_all(&header[..n]).unwrap();
    let masked: Vec<u8> = payload.iter().enumerate().map(|(i, b)| b ^ key[i % 4]).collect();
    stream.write_all(&masked).unwrap();
}

#[test]
fn fragmented_messages_are_joined_across_interleaved_pings() {
    let (listener, url) = bind_local();
    let peer = thread::spawn(move || {
        let (_read_half, mut write_half) = SyncWebSocketTransport::connect(&url).unwrap();
        // Text message in three fragments, with a ping between the first two.
        write_masked_frame(&mut write_half, 0x01, b"bid ");
        write_masked_frame(&mut write_half, 0x89, b"p");
        write_masked_frame(&mut write_half, 0x00, b"101.");
        write_masked_frame(&mut write_half, 0x80, b"25");
        // Binary message in two fragments.
        write_masked_frame(&mut write_half, 0x02, &[1, 2]);
        write_masked_frame(&mut write_half, 0x80, &[3]);
        write_masked_frame(&mut write_half, 0x88, &[0x03, 0xE8]);
        _read_half
    });

    let (read_half, write_half) = SyncWebSocketTransport::accept(&listener).unwrap();
    let (shared, _ws_write) = SyncWsWrite::new(write_half);
    let mut ws_read = SyncWsRead::new(read_half, shared);
    let mut buf = Vec::new();

    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_TEXT));
    assert_eq!(buf, b"bid 101.25");
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), Some(OPCODE_BINARY));
    assert_eq!(buf, [1, 2, 3]);
    assert_eq!(ws_read.read_message(&mut buf, MAX_MESSAGE).unwrap(), None);

    // The ping is answered with an unmasked pong on the peer's read half.
    let mut read_half = peer.join().unwrap();
    let mut pong = [0u8; 3];
    io::Read::read_exact(&mut read_half, &mut pong).unwrap();
    assert_eq!(pong, [0x8A, 1, b'p']);
}

#[test]
fn messages_beyond_max_len_are_rejected() {
    let (listener, url) = bind_local();
    let peer = thread::spawn(move || {
        let (_read_half, mut write_half) = SyncWebSocketTransport::connect(&url).unwrap();
        write_masked_frame(&mut write_half, 0x01, b"0123");
        write_masked_frame(&mut write_half, 0x80, b"4567");
        _read_half
    });

    let (read_half, write_half) = SyncWebSocketTransport::accept(&listener).unwrap();
    let (shared, _ws_write) = SyncWsWrite::new(write_half);
    let mut ws_read = SyncWsRead::new(read_half, shared);
    let mut buf = Vec::new();

    let error = ws_read.read_message(&mut buf, 6).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(buf.len() <= 6);
    drop(peer.join().unwrap());
}

#[test]
fn table_reader_rejects_text_frames() {
    let (listener, url) = bind_local();
    let peer = thread::spawn(move || {
        let (_read_half, mut write_half) = SyncWebSocketTransport::connect(&url).unwrap();
        write_masked_frame(&mut write_half, 0x81, b"not arrow");
        _read_half
    });

    let reader = SyncWebSocketTableReader::accept(&listener, None).unwrap();
    let error = reader.read_all_tables().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    drop(peer.join().unwrap());
}

#[test]
fn sync_transport_refuses_wss_urls() {
    let error = SyncWebSocketTransport::connect("wss://127.0.0.1:1").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn finish_called_twice_writes_one_close_frame() {
    let (listener, url) = bind_local();
    let table = make_mixed_table();
    let schema = schema_of(&table);
    let writer_thread = thread::spawn(move || {
        let mut writer = SyncWebSocketTableWriter::accept(&listener, schema, None).unwrap();
        writer.write_table(table).unwrap();
        writer.finish().unwrap();
        writer.finish().unwrap();
        let error = writer.write_table(make_mixed_table()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
    });

    let (mut read_half, _write_half) = SyncWebSocketTransport::connect(&url).unwrap();
    writer_thread.join().unwrap();
    drop(_write_half);
    let mut wire = Vec::new();
    io::Read::read_to_end(&mut read_half, &mut wire).unwrap();

    let mut close_frames = 0;
    let mut pos = 0;
    while pos < wire.len() {
        let header = parse_header(&wire[pos..]).unwrap();
        if header.opcode == OPCODE_CLOSE {
            close_frames += 1;
        }
        pos += header.header_len + header.payload_len as usize;
    }
    assert_eq!(pos, wire.len());
    assert_eq!(close_frames, 1);
}

#[test]
fn sync_ws_write_closes_once_and_rejects_later_writes() {
    let (shared, mut ws_write) = SyncWsWrite::new(Vec::new());
    ws_write.write_all(b"ok").unwrap();
    ws_write.close().unwrap();
    ws_write.close().unwrap();
    assert_eq!(ws_write.write(b"late").unwrap_err().kind(), io::ErrorKind::Other);
    assert_eq!(*shared.lock().unwrap(), [0x82, 2, b'o', b'k', 0x88, 2, 0x03, 0xE8]);
}
