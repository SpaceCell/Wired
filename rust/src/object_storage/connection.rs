// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage connections
//!
//! Resolves object URLs to a store and an object path. Stores are built
//! once per bucket, option set and Tokio runtime, and cached for the life of
//! the process. Readers and writers that address the same bucket from the
//! same runtime share one HTTP connection pool and one set of credentials.
//!
//! The runtime forms part of the cache key because an HTTP client's pooled
//! connections are driven by tasks on the runtime that opened them. A store
//! reused from another runtime fails its requests once the original runtime
//! has shut down.
//!
//! Options registered for a URL prefix apply to every URL beneath it. The
//! longest registered prefix that matches a URL supplies its options.
//! Registering options clears the store cache, which applies the new
//! options to later URLs.

use std::collections::HashMap;
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
use std::env;
use std::io;
use std::path::Path;
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
use std::str::FromStr;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
use object_store::{ClientConfigKey, ClientOptions};
use object_store::{ObjectStore, ObjectStoreScheme};
use tokio::runtime::Handle;
use url::{Position, Url};

use crate::object_storage::error::io_error;

/// Default size of each multipart upload part, in bytes. S3 requires every
/// part except the last to be at least this size, and it is the minimum
/// accepted for `part_size`.
pub const DEFAULT_PART_SIZE: usize = 5 * 1024 * 1024;

/// Default number of parts uploaded concurrently per object.
pub const DEFAULT_MAX_CONCURRENT_PARTS: usize = 8;

/// Registered option key for the multipart upload part size, in bytes.
const PART_SIZE_KEY: &str = "part_size";

/// Registered option key for the number of parts uploaded concurrently per
/// object.
const MAX_CONCURRENT_PARTS_KEY: &str = "max_concurrent_parts";

/// URL schemes routed to object storage. Every other path is a local path.
const OBJECT_SCHEMES: &[&str] = &[
    "s3", "s3a", "gs", "az", "adl", "azure", "abfs", "abfss", "memory",
];

/// Options registered per URL prefix, as `(prefix, options)`.
static OPTIONS: LazyLock<RwLock<Vec<(String, Vec<(String, String)>)>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

/// Stores built so far, keyed by the URL up to its path, matched option
/// prefix and Tokio runtime.
///
/// The URL up to its path includes the user part, which holds the container
/// in Azure URLs such as `abfss://container@account.dfs.core.windows.net`.
static STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The process-wide store behind `memory://` URLs. It is held outside the
/// store cache, which keeps its contents when options are registered.
static MEMORY: LazyLock<Arc<dyn ObjectStore>> = LazyLock::new(|| Arc::new(InMemory::new()));

/// One object or prefix within a store, resolved from an object URL, with
/// the upload settings registered for it.
#[derive(Debug, Clone)]
pub struct ObjectLocation {
    pub(crate) store: Arc<dyn ObjectStore>,
    pub(crate) path: ObjectPath,
    /// Size of each multipart upload part, in bytes.
    pub(crate) part_size: usize,
    /// Number of parts uploaded concurrently per object.
    pub(crate) max_concurrent_parts: usize,
}

/// Set the options used for every object URL beneath `prefix`.
///
/// `prefix` is an object URL such as `s3://telemetry` or
/// `s3://telemetry/lightning`. The options replace any registered earlier
/// for the same prefix, and are of two kinds:
///
/// - Upload settings, applied by lightstream to every backend:
///   - `part_size`: size of each multipart upload part in bytes. Defaults to
///     [`DEFAULT_PART_SIZE`], which is also the minimum.
///   - `max_concurrent_parts`: parts uploaded concurrently per object.
///     Defaults to [`DEFAULT_MAX_CONCURRENT_PARTS`]. Lower values suit links
///     with little upload bandwidth.
/// - Backend configuration keys, for example `aws_region`,
///   `aws_access_key_id`, `aws_endpoint` or `google_service_account`, and the
///   HTTP client keys such as `timeout`, `connect_timeout` and
///   `http1_only`. Options not given here are read by the backend from its
///   standard environment variables.
///
/// Requests have no limit on their total duration unless `timeout` is set,
/// which lets large parts upload over slow links. Connections that cannot
/// be established fail after the client's `connect_timeout`.
///
/// ## Errors
/// Returns `ErrorKind::InvalidInput` when `prefix` is not an object URL or an
/// upload setting is invalid. Unknown backend keys are reported when a URL
/// beneath the prefix is first opened.
pub fn register_options(prefix: &str, options: &[(&str, &str)]) -> io::Result<()> {
    if object_url(Path::new(prefix)).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("'{prefix}' is not an object storage URL"),
        ));
    }
    let prefix = prefix.trim_end_matches('/').to_string();
    let options: Vec<(String, String)> = options
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    upload_settings(&options)?;
    {
        let mut registered = OPTIONS.write().expect("object storage options lock");
        registered.retain(|(existing, _)| *existing != prefix);
        registered.push((prefix, options));
    }
    STORES
        .lock()
        .expect("object storage store cache lock")
        .clear();
    Ok(())
}

/// The object URL held by `path`, when its scheme is an object storage
/// scheme.
///
/// On Windows, `Path::join` inserts backslashes into joined URLs, and these
/// are normalised to `/`.
pub(crate) fn object_url(path: &Path) -> Option<String> {
    let text = path.to_str()?;
    let (scheme, _) = text.split_once("://")?;
    if !OBJECT_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str()) {
        return None;
    }
    #[cfg(windows)]
    let text = text.replace('\\', "/");
    Some(text.to_string())
}

/// Resolve an object URL to its store and object path.
///
/// ## Errors
/// - `ErrorKind::InvalidInput` for a malformed URL or an unknown option key.
/// - `ErrorKind::Unsupported` for a scheme whose backend feature is not
///   enabled.
pub(crate) fn resolve(url: &str) -> io::Result<ObjectLocation> {
    let parsed = Url::parse(url).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid object storage URL '{url}': {e}"),
        )
    })?;
    let (scheme, path) = ObjectStoreScheme::parse(&parsed).map_err(|e| io_error(e.into()))?;
    let (prefix, options) = registered_options(url);
    let (part_size, max_concurrent_parts) = upload_settings(&options)?;
    if scheme == ObjectStoreScheme::Memory {
        return Ok(ObjectLocation {
            store: Arc::clone(&MEMORY),
            path,
            part_size,
            max_concurrent_parts,
        });
    }

    let runtime = Handle::try_current()
        .map(|handle| handle.id().to_string())
        .unwrap_or_default();
    let key = format!("{}|{prefix}|{runtime}", &parsed[..Position::BeforePath]);

    let mut stores = STORES.lock().expect("object storage store cache lock");
    let store = match stores.get(&key) {
        Some(store) => Arc::clone(store),
        None => {
            let backend_options: Vec<(String, String)> = options
                .into_iter()
                .filter(|(key, _)| key != PART_SIZE_KEY && key != MAX_CONCURRENT_PARTS_KEY)
                .collect();
            let store = build(scheme, &parsed, &backend_options)?;
            stores.insert(key, Arc::clone(&store));
            store
        }
    };
    Ok(ObjectLocation {
        store,
        path,
        part_size,
        max_concurrent_parts,
    })
}

/// The `(part_size, max_concurrent_parts)` upload settings in `options`,
/// with the defaults for any not given.
///
/// ## Errors
/// Returns `ErrorKind::InvalidInput` for a value that is not a whole number,
/// a part size below [`DEFAULT_PART_SIZE`], or zero concurrent parts.
fn upload_settings(options: &[(String, String)]) -> io::Result<(usize, usize)> {
    let mut part_size = DEFAULT_PART_SIZE;
    let mut max_concurrent_parts = DEFAULT_MAX_CONCURRENT_PARTS;
    for (key, value) in options {
        let target = match key.as_str() {
            PART_SIZE_KEY => &mut part_size,
            MAX_CONCURRENT_PARTS_KEY => &mut max_concurrent_parts,
            _ => continue,
        };
        *target = value.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("object storage option '{key}' must be a whole number, got '{value}'"),
            )
        })?;
    }
    if part_size < DEFAULT_PART_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "object storage option '{PART_SIZE_KEY}' must be at least {DEFAULT_PART_SIZE} bytes"
            ),
        ));
    }
    if max_concurrent_parts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("object storage option '{MAX_CONCURRENT_PARTS_KEY}' must be at least 1"),
        ));
    }
    Ok((part_size, max_concurrent_parts))
}

/// Options of the longest registered prefix that `url` falls under, with
/// that prefix. Returns an empty prefix and no options when none match.
fn registered_options(url: &str) -> (String, Vec<(String, String)>) {
    let registered = OPTIONS.read().expect("object storage options lock");
    registered
        .iter()
        .filter(|(prefix, _)| {
            url == prefix
                || url
                    .strip_prefix(prefix.as_str())
                    .is_some_and(|rest| rest.starts_with('/'))
        })
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|(prefix, options)| (prefix.clone(), options.clone()))
        .unwrap_or_default()
}

/// Build the store for `scheme`, applying `options` over the backend's
/// environment configuration.
///
/// The HTTP client starts with no limit on total request time. The
/// backend's environment variables and then `options` are applied over that
/// default, and either can set a `timeout`.
#[cfg_attr(
    not(any(feature = "aws", feature = "gcp", feature = "azure")),
    allow(unused_variables)
)]
fn build(
    scheme: ObjectStoreScheme,
    url: &Url,
    options: &[(String, String)],
) -> io::Result<Arc<dyn ObjectStore>> {
    match scheme {
        #[cfg(feature = "aws")]
        ObjectStoreScheme::AmazonS3 => {
            use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
            let mut builder = AmazonS3Builder::from_env()
                .with_url(url.as_str())
                .with_client_options(client_options::<AmazonS3ConfigKey>(
                    "AWS_",
                    |key| match key {
                        AmazonS3ConfigKey::Client(key) => Some(key),
                        _ => None,
                    },
                ));
            for (key, value) in options {
                let key: AmazonS3ConfigKey = key.parse().map_err(io_error)?;
                builder = builder.with_config(key, value);
            }
            let store = builder.build().map_err(io_error)?;
            Ok(Arc::new(store))
        }
        #[cfg(feature = "gcp")]
        ObjectStoreScheme::GoogleCloudStorage => {
            use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
            let mut builder = GoogleCloudStorageBuilder::from_env()
                .with_url(url.as_str())
                .with_client_options(client_options::<GoogleConfigKey>(
                    "GOOGLE_",
                    |key| match key {
                        GoogleConfigKey::Client(key) => Some(key),
                        _ => None,
                    },
                ));
            for (key, value) in options {
                let key: GoogleConfigKey = key.parse().map_err(io_error)?;
                builder = builder.with_config(key, value);
            }
            let store = builder.build().map_err(io_error)?;
            Ok(Arc::new(store))
        }
        #[cfg(feature = "azure")]
        ObjectStoreScheme::MicrosoftAzure => {
            use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
            let mut builder = MicrosoftAzureBuilder::from_env()
                .with_url(url.as_str())
                .with_client_options(client_options::<AzureConfigKey>(
                    "AZURE_",
                    |key| match key {
                        AzureConfigKey::Client(key) => Some(key),
                        _ => None,
                    },
                ));
            for (key, value) in options {
                let key: AzureConfigKey = key.parse().map_err(io_error)?;
                builder = builder.with_config(key, value);
            }
            let store = builder.build().map_err(io_error)?;
            Ok(Arc::new(store))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "object storage URL '{url}' requires the lightstream `aws`, `gcp` or `azure` feature for its scheme"
            ),
        )),
    }
}

/// HTTP client options with no limit on total request time, followed by any
/// client settings in the environment variables beginning with `prefix`.
///
/// The backend builders' `from_env` reads the same variables. Setting the
/// client options afterwards replaces those, so the client settings among
/// them are applied again here.
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
fn client_options<K: FromStr>(
    prefix: &str,
    client_key: impl Fn(K) -> Option<ClientConfigKey>,
) -> ClientOptions {
    let mut options = ClientOptions::new().with_timeout_disabled();
    for (key, value) in env::vars() {
        if !key.starts_with(prefix) {
            continue;
        }
        if let Some(key) = key.to_ascii_lowercase().parse().ok().and_then(&client_key) {
            options = options.with_config(key, value);
        }
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_settings_default_and_validate() {
        assert_eq!(
            upload_settings(&[]).unwrap(),
            (DEFAULT_PART_SIZE, DEFAULT_MAX_CONCURRENT_PARTS)
        );
        let custom = [
            (PART_SIZE_KEY.to_string(), (8 * 1024 * 1024).to_string()),
            (MAX_CONCURRENT_PARTS_KEY.to_string(), "2".to_string()),
        ];
        assert_eq!(upload_settings(&custom).unwrap(), (8 * 1024 * 1024, 2));

        for (key, value) in [
            (PART_SIZE_KEY, "1024"),
            (PART_SIZE_KEY, "large"),
            (MAX_CONCURRENT_PARTS_KEY, "0"),
        ] {
            let err = upload_settings(&[(key.to_string(), value.to_string())]).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{key}={value}");
        }
    }

    #[test]
    fn registering_an_invalid_upload_setting_is_rejected() {
        let err = register_options(
            "memory:///invalid-settings",
            &[("max_concurrent_parts", "0")],
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn registered_upload_settings_reach_the_location() {
        register_options(
            "memory:///upload-settings",
            &[("max_concurrent_parts", "2"), ("part_size", "6291456")],
        )
        .unwrap();
        let location = resolve("memory:///upload-settings/a.bin").unwrap();
        assert_eq!(location.max_concurrent_parts, 2);
        assert_eq!(location.part_size, 6 * 1024 * 1024);
    }

    #[test]
    fn object_url_detects_object_schemes_only() {
        assert!(object_url(Path::new("s3://bucket/key.parquet")).is_some());
        assert!(object_url(Path::new("memory:///data/a.csv")).is_some());
        assert!(object_url(Path::new("/tmp/data/a.csv")).is_none());
        assert!(object_url(Path::new("relative/a.csv")).is_none());
        assert!(object_url(Path::new("C:\\data\\a.csv")).is_none());
        assert!(object_url(Path::new("https://host/a.csv")).is_none());
    }

    #[test]
    fn object_urls_joined_as_paths_keep_their_form() {
        let joined = Path::new("memory:///chunks").join("trades-0000000001.parquet");
        assert_eq!(
            object_url(&joined).as_deref(),
            Some("memory:///chunks/trades-0000000001.parquet")
        );
    }

    #[test]
    fn the_memory_store_is_shared_across_resolutions() {
        let first = resolve("memory:///shared/a").unwrap();
        let second = resolve("memory:///shared/b").unwrap();
        assert!(Arc::ptr_eq(&first.store, &second.store));
        assert_eq!(first.path.as_ref(), "shared/a");
    }

    #[test]
    fn the_longest_registered_prefix_supplies_the_options() {
        register_options("memory:///prefix-test", &[("a", "1")]).unwrap();
        register_options("memory:///prefix-test/inner", &[("b", "2")]).unwrap();

        let (prefix, options) = registered_options("memory:///prefix-test/inner/x");
        assert_eq!(prefix, "memory:///prefix-test/inner");
        assert_eq!(options, vec![("b".to_string(), "2".to_string())]);

        let (prefix, _) = registered_options("memory:///prefix-test/other");
        assert_eq!(prefix, "memory:///prefix-test");

        let (prefix, _) = registered_options("memory:///prefix-testing");
        assert_eq!(prefix, "");
    }

    #[test]
    fn registering_a_local_path_is_rejected() {
        let err = register_options("/tmp/data", &[]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(not(feature = "aws"))]
    #[test]
    fn a_backend_without_its_feature_is_unsupported() {
        let err = resolve("s3://bucket/key").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[cfg(feature = "azure")]
    #[test]
    fn containers_in_the_url_user_part_resolve_to_separate_stores() {
        let first = resolve("abfss://first@account.dfs.core.windows.net/key").unwrap();
        let second = resolve("abfss://second@account.dfs.core.windows.net/key").unwrap();
        assert!(first.store.to_string().contains("first"), "{}", first.store);
        assert!(second.store.to_string().contains("second"), "{}", second.store);
    }
}
