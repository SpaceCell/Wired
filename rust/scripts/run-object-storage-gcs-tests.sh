#!/usr/bin/env bash
# Run the object storage tests against Google Cloud Storage.
#
# Usage:
#   LIGHTSTREAM_GCS_URL=gs://bucket/prefix scripts/run-object-storage-gcs-tests.sh [credentials-file] [cargo test args]
#
# Credentials come from the gcloud Application Default Credentials, created
# with `gcloud auth application-default login`. The credentials file is read
# from GOOGLE_APPLICATION_CREDENTIALS when set, otherwise from the gcloud
# configuration directory (CLOUDSDK_CONFIG, defaulting to ~/.config/gcloud).
# A credentials file given as the first argument, either Application Default
# Credentials or a service account key, replaces both. A service account key
# in GOOGLE_SERVICE_ACCOUNT, GOOGLE_SERVICE_ACCOUNT_PATH or
# GOOGLE_SERVICE_ACCOUNT_KEY takes precedence over the credentials file.
# The account needs permission to create, read, list and delete objects in
# the bucket, for example through the Storage Object Admin role.
#
# Emulator:
#   Point GOOGLE_BASE_URL at fake-gcs-server, started with `-scheme http`,
#   and disable request signing:
#     GOOGLE_BASE_URL=http://localhost:4443 GOOGLE_SKIP_SIGNATURE=true GOOGLE_ALLOW_HTTP=true
#   The bucket must already exist. fake-gcs-server does not implement the XML
#   multipart uploads used by the large object check. Pass
#   `--skip large_objects` to leave that check out.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LIGHTSTREAM_GCS_URL:?set LIGHTSTREAM_GCS_URL, for example gs://bucket/prefix}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$1"
    shift
fi

if [[ -n "${GOOGLE_BASE_URL:-}" ]]; then
    echo "Running object storage tests against ${LIGHTSTREAM_GCS_URL} on ${GOOGLE_BASE_URL}"
else
    credentials="${GOOGLE_SERVICE_ACCOUNT:-}${GOOGLE_SERVICE_ACCOUNT_PATH:-}${GOOGLE_SERVICE_ACCOUNT_KEY:-}${GOOGLE_BEARER_TOKEN:-}"
    if [[ -z "$credentials" ]]; then
        adc="${GOOGLE_APPLICATION_CREDENTIALS:-${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}/application_default_credentials.json}"
        if [[ ! -f "$adc" ]]; then
            echo "No credentials file at ${adc}. Run 'gcloud auth application-default login'." >&2
            exit 1
        fi
        export GOOGLE_APPLICATION_CREDENTIALS="$adc"
    fi
    echo "Running object storage tests against ${LIGHTSTREAM_GCS_URL}"
fi

cargo test --features "gcp parquet csv json" --test object_storage_gcs -- --ignored "$@"
