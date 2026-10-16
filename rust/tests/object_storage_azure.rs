// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage roundtrip integration tests against Azure Blob Storage.
//!
//! Runs the shared object storage checks against a real container. The
//! tests are ignored by default because they need a container and
//! credentials. Run them with `scripts/run-object-storage-azure-tests.sh`,
//! which uses the Azure CLI's signed-in credential and runs the ignored
//! tests.
//!
//! ## Settings
//!
//! | Variable | Purpose |
//! |---|---|
//! | `LIGHTSTREAM_AZURE_URL` | Container and optional prefix, e.g. `az://container/ci` |
//! | `AZURE_STORAGE_ACCOUNT_NAME` | Storage account, unless the URL includes it |
//! | `AZURE_USE_AZURE_CLI` | `true` to authenticate with the Azure CLI's signed-in credential |
//! | `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_SAS_KEY` | Optional account key or SAS token in place of the Azure CLI |
//! | `AZURE_STORAGE_USE_EMULATOR` | Optional, `true` to use Azurite |
//! | `LIGHTSTREAM_AZURE_MAX_CONCURRENT_PARTS` | Optional `max_concurrent_parts` upload setting |
//! | `LIGHTSTREAM_AZURE_PART_SIZE` | Optional `part_size` upload setting, in bytes |
//!
//! ## URL forms
//!
//! - `az://<container>/<prefix>`, or the same form with the `adl`, `azure`,
//!   `abfs` or `abfss` scheme. The account comes from
//!   `AZURE_STORAGE_ACCOUNT_NAME`.
//! - `abfss://<container>@<account>.dfs.core.windows.net/<prefix>`, which
//!   includes the account.
//!
//! Container names cannot contain dots.
//!
//! Each check writes beneath its own run prefix,
//! `<LIGHTSTREAM_AZURE_URL>/lightstream-test-<process>-<time>/<check>`, and
//! the prefix is deleted when the check finishes, whether it passes or
//! fails.

#![cfg(feature = "azure")]

mod object_storage_checks;

use std::future::Future;

use object_storage_checks as checks;
use object_storage_checks::cloud;

/// Prefix of the `LIGHTSTREAM_*` environment variables for these tests.
const ENV_PREFIX: &str = "LIGHTSTREAM_AZURE";

/// Run a synchronous check against the container, then delete its objects.
fn run(name: &str, check: fn(&str)) {
    cloud::run(ENV_PREFIX, name, check);
}

/// Run an async check against the container, then delete its objects.
async fn run_async<F, Fut>(name: &str, check: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    cloud::run_async(ENV_PREFIX, name, check).await;
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn ipc_file_reader_reads_an_object_written_by_the_sync_writer() {
    run(
        "ipc-sync",
        checks::ipc_file_reader_reads_an_object_written_by_the_sync_writer,
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
async fn ipc_file_reader_reads_an_object_written_by_the_async_writer() {
    run_async("ipc-async", |base| async move {
        checks::ipc_file_reader_reads_an_object_written_by_the_async_writer(&base).await
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
async fn ipc_stream_round_trips_through_the_async_adapters() {
    run_async("ipc-stream", |base| async move {
        checks::ipc_stream_round_trips_through_the_async_adapters(&base).await
    })
    .await;
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn large_objects_upload_in_parts_and_read_back_whole() {
    run(
        "large",
        checks::large_objects_upload_in_parts_and_read_back_whole,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn writes_after_an_object_completes_are_rejected() {
    run(
        "complete",
        checks::writes_after_an_object_completes_are_rejected,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn a_writer_dropped_before_flushing_leaves_no_object() {
    run(
        "dropped",
        checks::a_writer_dropped_before_flushing_leaves_no_object,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn a_missing_object_reports_not_found() {
    run("missing", checks::a_missing_object_reports_not_found);
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn listing_returns_object_urls_immediately_beneath_the_prefix() {
    run(
        "listing",
        checks::listing_returns_object_urls_immediately_beneath_the_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn parquet_round_trips_through_the_storage_handles() {
    run(
        "parquet",
        checks::parquet::parquet_round_trips_through_the_storage_handles,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn chunked_parquet_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-parquet",
        checks::parquet::chunked_parquet_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn parallel_chunk_writes_reach_the_object_prefix() {
    run(
        "parallel",
        checks::parquet::parallel_chunk_writes_reach_the_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn chunked_parquet_loads_as_a_super_table() {
    run(
        "super-table",
        checks::parquet::chunked_parquet_loads_as_a_super_table,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn chunked_arrow_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-arrow",
        checks::arrow_chunks::chunked_arrow_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "datetime")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn auto_partitioned_chunks_round_trip_beneath_an_object_prefix() {
    run(
        "partitioned-arrow",
        checks::arrow_chunks::auto_partitioned_chunks_round_trip_beneath_an_object_prefix,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn csv_round_trips_through_the_path_entry_points() {
    run(
        "csv",
        checks::csv::csv_round_trips_through_the_path_entry_points,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn chunked_csv_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-csv",
        checks::csv::chunked_csv_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "json")]
#[test]
#[ignore = "requires LIGHTSTREAM_AZURE_URL and Azure credentials"]
fn json_round_trips_through_the_path_entry_points() {
    run(
        "json",
        checks::json::json_round_trips_through_the_path_entry_points,
    );
}
