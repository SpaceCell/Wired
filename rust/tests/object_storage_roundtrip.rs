// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage roundtrip integration tests.
//!
//! Covers the standard file readers and writers with `memory://` object
//! URLs in place of file paths. Each object is read back through the entry
//! point that matches the writer that produced it.
//!
//! The in-memory store is shared across the process, and each test uses its
//! own prefix to keep its objects separate. Object storage requests run on
//! the caller's Tokio runtime. The synchronous tests enter a runtime on the
//! test thread, and the async tests call the synchronous readers through
//! `spawn_blocking`.

#![cfg(feature = "object_storage")]

use std::io::{Read, Write};
use std::path::Path;

use lightstream::enums::{BufferChunkSize, IPCMessageProtocol};
use lightstream::models::readers::ipc::file_table::FileTableReader;
use lightstream::models::readers::ipc::table::TableReader;
use lightstream::models::readers::object_storage::{StorageReader, list_dir};
use lightstream::models::streams::object_store::{ObjectByteStream, ObjectWrite};
use lightstream::models::writers::ipc::sync_table::write_tables_to_file_sync;
use lightstream::models::writers::ipc::table::{TableWriter, write_tables_to_file};
use lightstream::models::writers::object_storage::StorageWriter;
use minarrow::{Array, Field, FieldArray, IntegerArray, NumericArray, StringArray, Table};
use tokio::runtime::Runtime;
use tokio::task::spawn_blocking;

/// Table of `n` rows whose `id` column starts at `first`.
fn sample_table(first: i64, n: usize) -> Table {
    let ids: Vec<i64> = (first..first + n as i64).collect();
    let venues: Vec<&str> = ids
        .iter()
        .map(|id| if id % 2 == 0 { "XNAS" } else { "XNYS" })
        .collect();
    Table::new(
        "trades".to_string(),
        vec![
            FieldArray::from_arr("id", Array::from_int64(IntegerArray::from_slice(&ids))),
            FieldArray::from_arr(
                "venue",
                Array::from_string32(StringArray::from_slice(&venues)),
            ),
        ]
        .into(),
    )
}

/// Multi-threaded runtime for the synchronous tests. Each test enters it on
/// the test thread, which is outside the async executor.
fn runtime() -> Runtime {
    Runtime::new().expect("test runtime starts")
}

fn schema_of(table: &Table) -> Vec<Field> {
    table.cols.iter().map(|col| (*col.field).clone()).collect()
}

/// Concatenated `id` column values of `tables`.
fn ids(tables: &[Table]) -> Vec<i64> {
    tables
        .iter()
        .flat_map(|table| match &table.cols[0].array {
            Array::NumericArray(NumericArray::Int64(ids)) => ids.data.as_slice().to_vec(),
            // CSV type inference narrows small integers to Int32.
            Array::NumericArray(NumericArray::Int32(ids)) => {
                ids.data.as_slice().iter().map(|id| *id as i64).collect()
            }
            other => panic!("id column decoded as {other:?}"),
        })
        .collect()
}

#[test]
fn ipc_file_reader_reads_an_object_written_by_the_sync_writer() {
    let runtime = runtime();
    let _entered = runtime.enter();
    let url = "memory:///ipc-sync/trades.arrow";
    let first = sample_table(0, 10);
    let schema = schema_of(&first);
    write_tables_to_file_sync(Path::new(url), &[first, sample_table(10, 5)], schema).unwrap();

    let reader = FileTableReader::open(url).unwrap();
    assert_eq!(reader.num_batches(), 2);
    let tables: Vec<Table> = (0..2).map(|i| reader.read_batch(i).unwrap()).collect();
    assert_eq!(ids(&tables), (0..15).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread")]
async fn ipc_file_reader_reads_an_object_written_by_the_async_writer() {
    let url = "memory:///ipc-async/trades.arrow";
    let first = sample_table(0, 8);
    let schema = schema_of(&first);
    write_tables_to_file(url, &[first, sample_table(8, 8)], schema)
        .await
        .unwrap();

    // The file reader is synchronous and runs outside the async executor.
    let tables = spawn_blocking(move || {
        let reader = FileTableReader::open(url).unwrap();
        (0..reader.num_batches())
            .map(|i| reader.read_batch(i).unwrap())
            .collect::<Vec<Table>>()
    })
    .await
    .unwrap();
    assert_eq!(ids(&tables), (0..16).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread")]
async fn ipc_stream_round_trips_through_the_async_adapters() {
    let url = "memory:///ipc-stream/trades.arrows";
    let first = sample_table(0, 20);
    let schema = schema_of(&first);
    let mut writer = TableWriter::new(
        ObjectWrite::create(url).unwrap(),
        schema,
        IPCMessageProtocol::Stream,
        None,
    )
    .unwrap();
    writer
        .write_all_tables(vec![first, sample_table(20, 20)])
        .await
        .unwrap();

    let stream = ObjectByteStream::open(url, BufferChunkSize::Http)
        .await
        .unwrap();
    let reader: TableReader = TableReader::new(stream, 64 * 1024, IPCMessageProtocol::Stream, None);
    let tables = reader.read_all_tables().await.unwrap();
    assert_eq!(ids(&tables), (0..40).collect::<Vec<_>>());
}

#[test]
fn large_objects_upload_in_parts_and_read_back_whole() {
    let runtime = runtime();
    let _entered = runtime.enter();
    let url = "memory:///large/blob.bin";
    // Three and a half parts, written in uneven slices.
    let data: Vec<u8> = (0..(17 * 1024 * 1024 + 512 * 1024))
        .map(|i| (i % 251) as u8)
        .collect();
    let mut writer = StorageWriter::create(url).unwrap();
    for slice in data.chunks(1_000_003) {
        writer.write_all(slice).unwrap();
    }
    writer.flush().unwrap();

    let mut read = Vec::new();
    StorageReader::open(url)
        .unwrap()
        .read_to_end(&mut read)
        .unwrap();
    assert_eq!(read.len(), data.len());
    assert!(read == data, "object contents match what was written");
}

#[test]
fn writes_after_an_object_completes_are_rejected() {
    let runtime = runtime();
    let _entered = runtime.enter();
    let url = "memory:///complete/once.bin";
    let mut writer = StorageWriter::create(url).unwrap();
    writer.write_all(b"first").unwrap();
    writer.flush().unwrap();
    // A second flush with nothing new written does nothing.
    writer.flush().unwrap();
    assert!(writer.write_all(b"second").is_err());
}

#[test]
fn a_writer_dropped_before_flushing_leaves_no_object() {
    let runtime = runtime();
    let _entered = runtime.enter();
    let url = "memory:///dropped/never.bin";
    let mut writer = StorageWriter::create(url).unwrap();
    writer.write_all(b"unfinished").unwrap();
    drop(writer);

    let err = StorageReader::open(url)
        .err()
        .expect("no object was created");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn object_urls_outside_a_runtime_context_report_an_error() {
    let err = StorageReader::open("memory:///no-runtime/a.bin")
        .err()
        .expect("no runtime context");
    assert!(
        err.to_string().contains("Tokio runtime context"),
        "got {err}"
    );
}

#[test]
fn a_missing_object_reports_not_found() {
    let runtime = runtime();
    let _entered = runtime.enter();
    let err = FileTableReader::open("memory:///missing/none.arrow")
        .err()
        .expect("missing object");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn listing_returns_object_urls_immediately_beneath_the_prefix() {
    let runtime = runtime();
    let _entered = runtime.enter();
    for name in ["a.bin", "b.bin", "nested/c.bin"] {
        let mut writer = StorageWriter::create(format!("memory:///listing/{name}")).unwrap();
        writer.write_all(name.as_bytes()).unwrap();
        writer.flush().unwrap();
    }
    let mut listed = list_dir("memory:///listing").unwrap();
    listed.sort();
    assert_eq!(
        listed,
        vec![
            Path::new("memory:///listing/a.bin").to_path_buf(),
            Path::new("memory:///listing/b.bin").to_path_buf(),
        ]
    );
}

#[cfg(feature = "parquet")]
mod parquet {
    use super::*;
    use lightstream::models::readers::chunked::parquet::ChunkedParquetReader;
    use lightstream::models::readers::parquet::load_parquet_table;
    use lightstream::models::writers::chunked::parquet::ChunkedParquetWriter;
    use lightstream::models::writers::parquet::write_parquet_table;
    use lightstream::traits::chunked_table_reader::ChunkedTableReader;
    use lightstream::traits::chunked_table_writer::ChunkedTableWriter;

    #[test]
    fn parquet_round_trips_through_the_storage_handles() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let url = "memory:///parquet/trades.parquet";
        let mut writer = StorageWriter::create(url).unwrap();
        write_parquet_table(&sample_table(0, 100), &mut writer, None).unwrap();
        writer.flush().unwrap();

        let table = load_parquet_table(StorageReader::open(url).unwrap()).unwrap();
        assert_eq!(ids(&[table]), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn chunked_parquet_round_trips_beneath_an_object_prefix() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let dir = "memory:///chunked-parquet/run";
        let mut writer = ChunkedParquetWriter::new(dir, "trades", None).unwrap();
        for first in [0, 10, 20] {
            writer.write_chunk(&sample_table(first, 10)).unwrap();
        }

        let reader = ChunkedParquetReader::open(dir, "trades", ()).unwrap();
        let tables: Vec<Table> = reader.map(|chunk| chunk.unwrap()).collect();
        assert_eq!(ids(&tables), (0..30).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_chunk_writes_reach_the_object_prefix() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let dir = "memory:///chunked-parquet/parallel";
        let writer = ChunkedParquetWriter::new(dir, "trades", None).unwrap();
        let tables: Vec<Table> = (0..6).map(|i| sample_table(i * 5, 5)).collect();
        let refs: Vec<&Table> = tables.iter().collect();
        writer.par_write_all(&refs, Some(3)).unwrap();

        let reader = ChunkedParquetReader::open(dir, "trades", ()).unwrap();
        let tables: Vec<Table> = reader.map(|chunk| chunk.unwrap()).collect();
        assert_eq!(ids(&tables), (0..30).collect::<Vec<_>>());
    }
}

mod arrow_chunks {
    use super::*;
    use lightstream::models::readers::chunked::arrow::ChunkedArrowReader;
    use lightstream::models::writers::chunked::arrow::ChunkedArrowWriter;
    use lightstream::traits::chunked_table_reader::ChunkedTableReader;
    use lightstream::traits::chunked_table_writer::ChunkedTableWriter;

    #[test]
    fn chunked_arrow_round_trips_beneath_an_object_prefix() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let dir = "memory:///chunked-arrow/run";
        let mut writer = ChunkedArrowWriter::new(dir, "trades").unwrap();
        for first in [0, 7, 14] {
            writer.write_chunk(&sample_table(first, 7)).unwrap();
        }

        let reader = ChunkedArrowReader::open(dir, "trades", ()).unwrap();
        let tables: Vec<Table> = reader.map(|chunk| chunk.unwrap()).collect();
        assert_eq!(ids(&tables), (0..21).collect::<Vec<_>>());
    }
}

#[cfg(feature = "csv")]
mod csv {
    use super::*;
    use lightstream::models::readers::chunked::csv::{ChunkedCsvReadOptions, ChunkedCsvReader};
    use lightstream::models::readers::csv::CsvReader;
    use lightstream::models::writers::chunked::csv::ChunkedCsvWriter;
    use lightstream::models::writers::csv::CsvWriter;
    use lightstream::traits::chunked_table_reader::ChunkedTableReader;
    use lightstream::traits::chunked_table_writer::ChunkedTableWriter;

    #[test]
    fn csv_round_trips_through_the_path_entry_points() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let url = "memory:///csv/trades.csv";
        let mut writer = CsvWriter::to_path(url).unwrap();
        writer.write_table(&sample_table(0, 25)).unwrap();
        writer.flush().unwrap();

        let mut reader = CsvReader::from_path(url, Default::default(), 1_000).unwrap();
        let table = reader.next_batch().unwrap().expect("one batch");
        assert_eq!(table.n_rows, 25);
        assert_eq!(ids(&[table]), (0..25).collect::<Vec<_>>());
    }

    #[test]
    fn chunked_csv_round_trips_beneath_an_object_prefix() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let dir = "memory:///chunked-csv/run";
        let mut writer = ChunkedCsvWriter::new(dir, "trades", Default::default()).unwrap();
        for first in [0, 4] {
            writer.write_chunk(&sample_table(first, 4)).unwrap();
        }

        let reader =
            ChunkedCsvReader::open(dir, "trades", ChunkedCsvReadOptions::default()).unwrap();
        let tables: Vec<Table> = reader.map(|chunk| chunk.unwrap()).collect();
        assert_eq!(ids(&tables), (0..8).collect::<Vec<_>>());
    }
}

#[cfg(feature = "json")]
mod json {
    use super::*;
    use lightstream::models::decoders::json::JsonDecodeOptions;
    use lightstream::models::encoders::json::{JsonEncodeOptions, JsonFormat};
    use lightstream::models::readers::json::JsonReader;
    use lightstream::models::writers::json::JsonWriter;

    #[test]
    fn json_round_trips_through_the_path_entry_points() {
        let runtime = runtime();
        let _entered = runtime.enter();
        let url = "memory:///json/trades.ndjson";
        let options = JsonEncodeOptions {
            format: JsonFormat::Ndjson,
            ..Default::default()
        };
        let table = sample_table(0, 12);
        let decode = JsonDecodeOptions {
            schema: Some(schema_of(&table)),
            ..Default::default()
        };
        let mut writer = JsonWriter::to_path(url, options).unwrap();
        writer.write_table(&table).unwrap();
        writer.flush().unwrap();

        let mut reader = JsonReader::from_path(url, JsonFormat::Ndjson, decode, 1_000).unwrap();
        let table = reader.next_batch().unwrap().expect("one batch");
        assert_eq!(ids(&[table]), (0..12).collect::<Vec<_>>());
    }
}
