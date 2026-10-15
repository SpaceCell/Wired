// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage roundtrip integration tests against Amazon S3.
//!
//! Runs the shared object storage checks against a real bucket. The tests
//! are ignored by default because they need a bucket and credentials. Run
//! them with `scripts/run-object-storage-s3-tests.sh`, which exports the
//! AWS CLI's credentials and runs the ignored tests.
//!
//! ## Settings
//!
//! | Variable | Purpose |
//! |---|---|
//! | `LIGHTSTREAM_S3_URL` | Bucket and optional prefix, e.g. `s3://bucket/ci` |
//! | `AWS_REGION` | Bucket region |
//! | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | Credentials |
//! | `AWS_ENDPOINT` | Optional endpoint for S3-compatible stores |
//!
//! Each check writes beneath its own run prefix,
//! `<LIGHTSTREAM_S3_URL>/lightstream-test-<process>-<time>/<check>`, and
//! the prefix is deleted when the check finishes, whether it passes or
//! fails.

#![cfg(feature = "aws")]

mod object_storage_checks;

use std::env;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::{FutureExt, TryStreamExt};
use object_storage_checks as checks;
use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, ObjectStoreExt, ObjectStoreScheme};
use url::Url;

/// Base object URL for one check, unique to this process and start time.
fn check_base(check: &str) -> String {
    let url = env::var("LIGHTSTREAM_S3_URL")
        .expect("LIGHTSTREAM_S3_URL is not set, e.g. s3://bucket/prefix");
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "{}/lightstream-test-{}-{started}/{check}",
        url.trim_end_matches('/'),
        process::id()
    )
}

/// Delete every object beneath `base`.
async fn delete_prefix(base: &str) {
    let url = Url::parse(base).unwrap();
    let (_, prefix) = ObjectStoreScheme::parse(&url).unwrap();
    let store = AmazonS3Builder::from_env().with_url(base).build().unwrap();
    let objects: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    for meta in objects {
        store.delete(&meta.location).await.unwrap();
    }
}

/// Run a synchronous check on the test thread, inside an entered runtime,
/// then delete its objects.
fn run(name: &str, check: fn(&str)) {
    let base = check_base(name);
    let runtime = checks::runtime();
    let result = {
        let _entered = runtime.enter();
        catch_unwind(AssertUnwindSafe(|| check(&base)))
    };
    runtime.block_on(delete_prefix(&base));
    if let Err(panic) = result {
        resume_unwind(panic);
    }
}

/// Run an async check, then delete its objects.
async fn run_async<F, Fut>(name: &str, check: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    let base = check_base(name);
    let result = AssertUnwindSafe(check(base.clone())).catch_unwind().await;
    delete_prefix(&base).await;
    if let Err(panic) = result {
        resume_unwind(panic);
    }
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn ipc_file_reader_reads_an_object_written_by_the_sync_writer() {
    run(
        "ipc-sync",
        checks::ipc_file_reader_reads_an_object_written_by_the_sync_writer,
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
async fn ipc_file_reader_reads_an_object_written_by_the_async_writer() {
    run_async("ipc-async", |base| async move {
        checks::ipc_file_reader_reads_an_object_written_by_the_async_writer(&base).await
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
async fn ipc_stream_round_trips_through_the_async_adapters() {
    run_async("ipc-stream", |base| async move {
        checks::ipc_stream_round_trips_through_the_async_adapters(&base).await
    })
    .await;
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn large_objects_upload_in_parts_and_read_back_whole() {
    run(
        "large",
        checks::large_objects_upload_in_parts_and_read_back_whole,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn writes_after_an_object_completes_are_rejected() {
    run(
        "complete",
        checks::writes_after_an_object_completes_are_rejected,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn a_writer_dropped_before_flushing_leaves_no_object() {
    run(
        "dropped",
        checks::a_writer_dropped_before_flushing_leaves_no_object,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn a_missing_object_reports_not_found() {
    run("missing", checks::a_missing_object_reports_not_found);
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn listing_returns_object_urls_immediately_beneath_the_prefix() {
    run(
        "listing",
        checks::listing_returns_object_urls_immediately_beneath_the_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn parquet_round_trips_through_the_storage_handles() {
    run(
        "parquet",
        checks::parquet::parquet_round_trips_through_the_storage_handles,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn chunked_parquet_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-parquet",
        checks::parquet::chunked_parquet_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn parallel_chunk_writes_reach_the_object_prefix() {
    run(
        "parallel",
        checks::parquet::parallel_chunk_writes_reach_the_object_prefix,
    );
}

#[cfg(feature = "parquet")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn chunked_parquet_loads_as_a_super_table() {
    run(
        "super-table",
        checks::parquet::chunked_parquet_loads_as_a_super_table,
    );
}

#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn chunked_arrow_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-arrow",
        checks::arrow_chunks::chunked_arrow_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn csv_round_trips_through_the_path_entry_points() {
    run(
        "csv",
        checks::csv::csv_round_trips_through_the_path_entry_points,
    );
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn chunked_csv_round_trips_beneath_an_object_prefix() {
    run(
        "chunked-csv",
        checks::csv::chunked_csv_round_trips_beneath_an_object_prefix,
    );
}

#[cfg(feature = "json")]
#[test]
#[ignore = "requires LIGHTSTREAM_S3_URL and AWS credentials"]
fn json_round_trips_through_the_path_entry_points() {
    run(
        "json",
        checks::json::json_round_trips_through_the_path_entry_points,
    );
}
