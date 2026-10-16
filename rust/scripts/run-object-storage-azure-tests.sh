#!/usr/bin/env bash
# Run the object storage tests against Azure Blob Storage.
#
# Usage:
#   LIGHTSTREAM_AZURE_URL=az://container/prefix scripts/run-object-storage-azure-tests.sh [storage-account] [cargo test args]
#
# The storage account is taken from the first argument, otherwise from
# AZURE_STORAGE_ACCOUNT_NAME. URLs of the form
# abfss://container@account.dfs.core.windows.net/prefix include the account
# and need neither.
#
# Credentials come from the Azure CLI after `az login`. The signed-in
# identity needs a data access role on the storage account, such as Storage
# Blob Data Contributor. An account key or SAS token in
# AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_SAS_KEY replaces the Azure CLI
# credential when set, for example:
#   export AZURE_STORAGE_ACCOUNT_KEY="$(az storage account keys list \
#       --account-name <account> --query '[0].value' --output tsv)"
#
# Emulator:
#   Set AZURE_STORAGE_USE_EMULATOR=true to run against Azurite at
#   http://127.0.0.1:10000, or at the endpoint in AZURITE_BLOB_STORAGE_URL.
#   Azurite's default account and key apply, and the container must already
#   exist.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LIGHTSTREAM_AZURE_URL:?set LIGHTSTREAM_AZURE_URL, for example az://container/prefix}"

if [[ $# -gt 0 && "$1" != -* ]]; then
    export AZURE_STORAGE_ACCOUNT_NAME="$1"
    shift
fi

emulator="$(printf '%s' "${AZURE_STORAGE_USE_EMULATOR:-}" | tr '[:upper:]' '[:lower:]')"
case "$emulator" in
    1 | true | on | yes | y)
        echo "Running object storage tests against ${LIGHTSTREAM_AZURE_URL} on Azurite"
        ;;
    *)
        if [[ "$LIGHTSTREAM_AZURE_URL" != *@* ]]; then
            : "${AZURE_STORAGE_ACCOUNT_NAME:?set AZURE_STORAGE_ACCOUNT_NAME or pass the storage account as the first argument}"
        fi
        credentials="${AZURE_STORAGE_ACCOUNT_KEY:-}${AZURE_STORAGE_ACCESS_KEY:-}${AZURE_STORAGE_SAS_KEY:-}${AZURE_STORAGE_SAS_TOKEN:-}${AZURE_STORAGE_TOKEN:-}"
        if [[ -z "$credentials" ]]; then
            # Confirm the Azure CLI has a signed-in account before the tests
            # request tokens through it.
            az account show --output none
            export AZURE_USE_AZURE_CLI=true
        fi
        echo "Running object storage tests against ${LIGHTSTREAM_AZURE_URL}"
        ;;
esac

cargo test --features "azure datetime parquet csv json" --test object_storage_azure -- --ignored "$@"
