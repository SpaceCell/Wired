// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage roundtrip integration tests against the in-memory store.
//!
//! Runs the shared object storage checks with `memory://` object URLs in
//! place of file paths. The in-memory store is shared across the process,
//! and each check writes beneath its own path to keep its objects separate.

#![cfg(feature = "object_storage")]

mod object_storage_checks;

use lightstream::models::readers::object_storage::StorageReader;
use lightstream::object_storage::register_options;
use object_storage_checks as checks;

const BASE: &str = "memory:///checks";

/// Run a synchronous check on the test thread, inside an entered runtime.
fn run(check: fn(&str)) {
    let runtime = checks::runtime();
    let _entered = runtime.enter();
    check(BASE);
}

#[test]
fn object_urls_outside_a_runtime_context_report_an_error() {
    let err = StorageReader::open(format!("{BASE}/no-runtime/a.bin"))
        .err()
        .expect("no runtime context");
    assert!(
        err.to_string().contains("Tokio runtime context"),
        "got {err}"
    );
}

#[test]
fn ipc_file_reader_reads_an_object_written_by_the_sync_writer() {
    run(checks::ipc_file_reader_reads_an_object_written_by_the_sync_writer);
}

#[tokio::test(flavor = "multi_thread")]
async fn ipc_file_reader_reads_an_object_written_by_the_async_writer() {
    checks::ipc_file_reader_reads_an_object_written_by_the_async_writer(BASE).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn ipc_stream_round_trips_through_the_async_adapters() {
    checks::ipc_stream_round_trips_through_the_async_adapters(BASE).await;
}

#[test]
fn large_objects_upload_in_parts_and_read_back_whole() {
    run(checks::large_objects_upload_in_parts_and_read_back_whole);
}

#[test]
fn registered_upload_settings_apply_to_large_uploads() {
    let base = "memory:///checks-one-part-at-a-time";
    register_options(
        base,
        &[("max_concurrent_parts", "1"), ("part_size", "6291456")],
    )
    .unwrap();
    let runtime = checks::runtime();
    let _entered = runtime.enter();
    checks::large_objects_upload_in_parts_and_read_back_whole(base);
}

#[test]
fn writes_after_an_object_completes_are_rejected() {
    run(checks::writes_after_an_object_completes_are_rejected);
}

#[test]
fn a_writer_dropped_before_flushing_leaves_no_object() {
    run(checks::a_writer_dropped_before_flushing_leaves_no_object);
}

#[test]
fn a_missing_object_reports_not_found() {
    run(checks::a_missing_object_reports_not_found);
}

#[test]
fn listing_returns_object_urls_immediately_beneath_the_prefix() {
    run(checks::listing_returns_object_urls_immediately_beneath_the_prefix);
}

#[cfg(feature = "parquet")]
#[test]
fn parquet_round_trips_through_the_storage_handles() {
    run(checks::parquet::parquet_round_trips_through_the_storage_handles);
}

#[cfg(feature = "parquet")]
#[test]
fn chunked_parquet_round_trips_beneath_an_object_prefix() {
    run(checks::parquet::chunked_parquet_round_trips_beneath_an_object_prefix);
}

#[cfg(feature = "parquet")]
#[test]
fn parallel_chunk_writes_reach_the_object_prefix() {
    run(checks::parquet::parallel_chunk_writes_reach_the_object_prefix);
}

#[cfg(feature = "parquet")]
#[test]
fn chunked_parquet_loads_as_a_super_table() {
    run(checks::parquet::chunked_parquet_loads_as_a_super_table);
}

#[test]
fn chunked_arrow_round_trips_beneath_an_object_prefix() {
    run(checks::arrow_chunks::chunked_arrow_round_trips_beneath_an_object_prefix);
}

#[test]
fn auto_partitioned_chunks_round_trip_beneath_an_object_prefix() {
    run(checks::arrow_chunks::auto_partitioned_chunks_round_trip_beneath_an_object_prefix);
}

#[cfg(feature = "csv")]
#[test]
fn csv_round_trips_through_the_path_entry_points() {
    run(checks::csv::csv_round_trips_through_the_path_entry_points);
}

#[cfg(feature = "csv")]
#[test]
fn chunked_csv_round_trips_beneath_an_object_prefix() {
    run(checks::csv::chunked_csv_round_trips_beneath_an_object_prefix);
}

#[cfg(feature = "json")]
#[test]
fn json_round_trips_through_the_path_entry_points() {
    run(checks::json::json_round_trips_through_the_path_entry_points);
}
