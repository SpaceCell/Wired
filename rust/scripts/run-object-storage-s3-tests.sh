#!/usr/bin/env bash
# Run the object storage tests against Amazon S3.
#
# Usage:
#   LIGHTSTREAM_S3_URL=s3://bucket/prefix scripts/run-object-storage-s3-tests.sh [profile] [cargo test args]
#
# Credentials come from the AWS CLI, which resolves profiles and SSO sessions
# that the object_store client does not read itself. The region is taken
# from AWS_REGION when set, otherwise from the CLI's configured region, and
# must match the bucket's region.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LIGHTSTREAM_S3_URL:?set LIGHTSTREAM_S3_URL, for example s3://bucket/prefix}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    export AWS_PROFILE="$1"
    shift
fi

eval "$(aws configure export-credentials --format env)"
export AWS_REGION="${AWS_REGION:-$(aws configure get region || true)}"
: "${AWS_REGION:?set AWS_REGION to the region of the bucket}"

echo "Running object storage tests against ${LIGHTSTREAM_S3_URL} in ${AWS_REGION}"
cargo test --features "aws parquet csv json" --test object_storage_s3 -- --ignored "$@"
