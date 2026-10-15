// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage roundtrip integration tests against Google Cloud Storage.
//!
//! Runs the shared object storage checks against a real bucket. The tests
//! are ignored by default because they need a bucket and credentials. Run
//! them with `scripts/run-object-storage-gcs-tests.sh`, which uses the
//! gcloud Application Default Credentials and runs the ignored tests.
//!
//! ## Settings
//!
//! | Variable | Purpose |
//! |---|---|
//! | `LIGHTSTREAM_GCS_URL` | Bucket and optional prefix, e.g. `gs://bucket/ci` |
//! | `GOOGLE_APPLICATION_CREDENTIALS` | Optional credentials file, defaulting to `~/.config/gcloud/application_default_credentials.json` |
//! | `GOOGLE_SERVICE_ACCOUNT`, `GOOGLE_SERVICE_ACCOUNT_KEY` | Optional service account key, as a file path or as JSON |
//! | `GOOGLE_BASE_URL` | Optional endpoint for a GCS emulator |
//! | `LIGHTSTREAM_GCS_MAX_CONCURRENT_PARTS` | Optional `max_concurrent_parts` upload setting |
//! | `LIGHTSTREAM_GCS_PART_SIZE` | Optional `part_size` upload setting, in bytes |
//!
//! Each check writes beneath its own run prefix,
//! `<LIGHTSTREAM_GCS_URL>/lightstream-test-<process>-<time>/<check>`, and
//! the prefix is deleted when the check finishes, whether it passes or
//! fails.

#![cfg(feature = "gcp")]

mod object_storage_checks;

use std::future::Future;

use object_storage_checks as checks;
use object_storage_checks::cloud;
use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;

/// Prefix of the `LIGHTSTREAM_*` environment variables for these tests.
const ENV_PREFIX: &str = "LIGHTSTREAM_GCS";

/// Store for an object URL, configured from the `GOOGLE_*` environment
/// variables and the Application Default Credentials.
fn store(url: &str) -> Box<dyn ObjectStore> {
    Box::new(
        GoogleCloudStorageBuilder::from_env()
            .with_url(url)
            .build()
            .unwrap(),
    )
}

/// Run a synchronous check against the bucket, then delete its objects.
fn run(name: &str, check: fn(&str)) {
    cloud::run(ENV_PREFIX, store, name, check);
}

/// Run an async check against the bucket, then delete its objects.
async fn run_async<F, Fut>(name: &str, check: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    cloud::run_async(ENV_PREFIX, store, name, check).await;
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn ipc_file_reader_reads_an_object_written_by_the_sync_writer() {
    run(
        "ipc-sync",
        checks::ipc_file_reader_reads_an_object_written_by_the_sync_writer,
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
async fn ipc_file_reader_reads_an_object_written_by_the_async_writer() {
    run_async("ipc-async", |base| async move {
        checks::ipc_file_reader_reads_an_object_written_by_the_async_writer(&base).await
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
async fn ipc_stream_round_trips_through_the_async_adapters() {
    run_async("ipc-stream", |base| async move {
        checks::ipc_stream_round_trips_through_the_async_adapters(&base).await
    })
    .await;
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn large_objects_upload_in_parts_and_read_back_whole() {
    run(
        "large",
        checks::large_objects_upload_in_parts_and_read_back_whole,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn writes_after_an_object_completes_are_rejected() {
    run(
        "complete",
        checks::writes_after_an_object_completes_are_rejected,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn a_writer_dropped_before_flushing_leaves_no_object() {
    run(
        "dropped",
        checks::a_writer_dropped_before_flushing_leaves_no_object,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn a_missing_object_reports_not_found() {
    run("missing", checks::a_missing_object_reports_not_found);
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn listing_returns_object_urls_immediately_beneath_the_prefix() {
    run(
        "listing",
        checks::listing_returns_object_urls_immediately_beneath_the_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn parquet_round_trips_through_the_storage_handles() {
    run(
        "parquet",
        checks::parquet::parquet_round_trips_through_the_storage_handles,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn chunked_parquet_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-parquet",
        checks::parquet::chunked_parquet_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn parallel_chunk_writes_reach_the_object_prefix() {
    run(
        "parallel",
        checks::parquet::parallel_chunk_writes_reach_the_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn chunked_parquet_loads_as_a_super_table() {
    run(
        "super-table",
        checks::parquet::chunked_parquet_loads_as_a_super_table,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn chunked_arrow_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-arrow",
        checks::arrow_chunks::chunked_arrow_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn csv_round_trips_through_the_path_entry_points() {
    run(
        "csv",
        checks::csv::csv_round_trips_through_the_path_entry_points,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn chunked_csv_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-csv",
        checks::csv::chunked_csv_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "json")]
#[test]
#[ignore = "requires LIGHTSTREAM_GCS_URL and Google Cloud credentials"]
fn json_round_trips_through_the_path_entry_points() {
    run(
        "json",
        checks::json::json_round_trips_through_the_path_entry_points,
    );
}
