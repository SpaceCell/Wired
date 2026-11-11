// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sync UDS roundtrip integration tests.
//!
//! Exercises `SyncUdsTableReader` and `SyncUdsTableWriter` against each other
//! and against the async `UdsTableReader` and `UdsTableWriter`, in both peer
//! roles, with and without compression. Sync peers run on their own threads
//! with no async runtime. Async peers run on a current-thread runtime on
//! their own threads.

#![cfg(feature = "uds")]

use std::io;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use futures_util::StreamExt;
use lightstream::compression::Compression;
use lightstream::models::decoders::limits::DecodeLimits;
use lightstream::models::readers::sync_uds::SyncUdsTableReader;
use lightstream::models::readers::uds::UdsTableReader;
use lightstream::models::writers::sync_uds::SyncUdsTableWriter;
use lightstream::models::writers::uds::UdsTableWriter;
use lightstream::traits::transport_reader::IPCTransportReader;
use lightstream::traits::transport_writer::IPCTransportWriter;
use minarrow::{
    Array, ArrowType, Bitmask, Buffer, CategoricalArray, Field, FieldArray, FloatArray,
    IntegerArray, NumericArray, StringArray, Table, TextArray, Vec64, arr_f64, arr_i64,
    ffi::arrow_dtype::CategoricalIndexType,
};

/// Dictionary id of the categorical column in `make_mixed_table`.
const DICT_ID: i64 = 3;

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

/// Numeric table large enough that record batch bodies span many socket reads.
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

/// Peer that binds the socket path and accepts the connection.
#[derive(Clone, Copy, Debug)]
enum Listening {
    Writer,
    Reader,
}

/// Connection role of one peer.
enum Endpoint {
    Connect(PathBuf),
    Accept(UnixListener),
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
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
                Endpoint::Connect(path) => {
                    SyncUdsTableWriter::connect(&path, schema, compression).unwrap()
                }
                Endpoint::Accept(listener) => {
                    SyncUdsTableWriter::accept(&listener, schema, compression).unwrap()
                }
            };
            writer.write_all_tables(tables).unwrap();
        }
        Side::Async => current_thread_runtime().block_on(async move {
            let mut writer = match endpoint {
                Endpoint::Connect(path) => {
                    UdsTableWriter::connect(&path, schema, compression).await.unwrap()
                }
                Endpoint::Accept(listener) => {
                    listener.set_nonblocking(true).unwrap();
                    let listener = tokio::net::UnixListener::from_std(listener).unwrap();
                    UdsTableWriter::accept(&listener, schema, compression)
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
                Endpoint::Connect(path) => SyncUdsTableReader::connect(&path, limits)?,
                Endpoint::Accept(listener) => SyncUdsTableReader::accept(&listener, limits)?,
            };
            reader.read_all_tables()
        }
        Side::Async => current_thread_runtime().block_on(async move {
            let reader = match endpoint {
                Endpoint::Connect(path) => UdsTableReader::connect(&path, limits).await?,
                Endpoint::Accept(listener) => {
                    listener.set_nonblocking(true)?;
                    let listener = tokio::net::UnixListener::from_std(listener)?;
                    UdsTableReader::accept(&listener, limits).await?
                }
            };
            reader.read_all_tables().await
        }),
    }
}

/// Send `tables` from `writer` to `reader` over one UDS connection and return
/// what the reader decoded.
fn transfer(
    writer: Side,
    reader: Side,
    listening: Listening,
    tables: Vec<Table>,
    compression: Option<Compression>,
    limits: Option<DecodeLimits>,
) -> io::Result<Vec<Table>> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sync_uds.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let schema = schema_of(&tables[0]);

    let (writer_end, reader_end) = match listening {
        Listening::Writer => (Endpoint::Accept(listener), Endpoint::Connect(path)),
        Listening::Reader => (Endpoint::Connect(path), Endpoint::Accept(listener)),
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
fn large_batches_cross_in_both_directions() {
    let tables = vec![make_large_table(250_000), make_large_table(100_000)];
    for (writer, reader) in [(Side::Sync, Side::Async), (Side::Async, Side::Sync)] {
        for compression in compressions() {
            let got = transfer(
                writer,
                reader,
                Listening::Reader,
                tables.clone(),
                compression,
                None,
            )
            .unwrap();
            assert_equal_contents(&got, &tables);
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("iterate.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let table = make_mixed_table();
    let schema = schema_of(&table);

    let write_table = table.clone();
    let writer_thread = thread::spawn(move || {
        let mut writer = SyncUdsTableWriter::accept(&listener, schema, None).unwrap();
        writer.write_table(write_table.clone()).unwrap();
        writer.write_table(write_table).unwrap();
        writer.finish().unwrap();
    });

    let mut reader = SyncUdsTableReader::connect(&path, None).unwrap();
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let table = make_mixed_table();
    let schema = schema_of(&table);

    let write_table = table.clone();
    let writer_thread = thread::spawn(move || {
        let mut writer = SyncUdsTableWriter::accept(&listener, schema, None).unwrap();
        writer.write_table(write_table.clone()).unwrap();
        writer.write_table(write_table).unwrap();
        writer.finish().unwrap();
    });

    let count = current_thread_runtime().block_on(async {
        let mut reader = UdsTableReader::connect(&path, None).await.unwrap();
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

/// Read every byte a writer sends to an accepted connection.
fn capture_wire_bytes(writer: Side, tables: Vec<Table>, compression: Option<Compression>) -> Vec<u8> {
    use std::io::Read;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wire.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let schema = schema_of(&tables[0]);

    let writer_thread = thread::spawn(move || {
        write_tables(writer, Endpoint::Connect(path), schema, tables, compression)
    });
    let (mut stream, _) = listener.accept().unwrap();
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).unwrap();
    writer_thread.join().unwrap();
    bytes
}

#[test]
fn sync_and_async_writers_send_identical_bytes() {
    let cases = [
        vec![make_mixed_table(), make_mixed_table()],
        vec![make_large_table(10_000), make_large_table(3)],
    ];
    for tables in cases {
        for compression in compressions() {
            let sync_bytes = capture_wire_bytes(Side::Sync, tables.clone(), compression);
            let async_bytes = capture_wire_bytes(Side::Async, tables.clone(), compression);
            assert!(!sync_bytes.is_empty());
            assert_eq!(sync_bytes, async_bytes, "{compression:?}");
        }
    }
}
