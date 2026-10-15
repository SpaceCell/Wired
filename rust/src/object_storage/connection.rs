// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # Object storage connections
//!
//! Resolves object URLs to a store and an object path. Stores are built
//! once per bucket and option set and cached for the life of the process,
//! keeping HTTP connection pools and credentials shared across every
//! reader and writer that addresses the same bucket.
//!
//! Options registered for a URL prefix apply to every URL beneath it. The
//! longest registered prefix that matches a URL supplies its options.
//! Registering options clears the store cache, which applies the new
//! options to later URLs.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreScheme};
use url::Url;

use crate::object_storage::error::io_error;

/// URL schemes routed to object storage. Every other path is a local path.
const OBJECT_SCHEMES: &[&str] = &[
    "s3", "s3a", "gs", "az", "adl", "azure", "abfs", "abfss", "memory",
];

/// Options registered per URL prefix, as `(prefix, options)`.
static OPTIONS: LazyLock<RwLock<Vec<(String, Vec<(String, String)>)>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

/// Stores built so far, keyed by scheme, bucket and matched option prefix.
static STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The process-wide store behind `memory://` URLs. It is held outside the
/// store cache, which keeps its contents when options are registered.
static MEMORY: LazyLock<Arc<dyn ObjectStore>> = LazyLock::new(|| Arc::new(InMemory::new()));

/// One object or prefix within a store, resolved from an object URL.
#[derive(Debug, Clone)]
pub struct ObjectLocation {
    pub(crate) store: Arc<dyn ObjectStore>,
    pub(crate) path: ObjectPath,
}

/// Set the options used for every object URL beneath `prefix`.
///
/// `prefix` is an object URL such as `s3://telemetry` or
/// `s3://telemetry/lightning`. `options` are backend configuration keys and
/// values, for example `aws_region`, `aws_access_key_id`, `aws_endpoint` or
/// `google_service_account`, and replace any options registered earlier for
/// the same prefix. Options not given here are read by the backend from its
/// standard environment variables.
///
/// ## Errors
/// Returns `ErrorKind::InvalidInput` when `prefix` is not an object URL.
/// Unknown option keys are reported when a URL beneath the prefix is first
/// opened.
pub fn register_options(prefix: &str, options: &[(&str, &str)]) -> io::Result<()> {
    if object_url(Path::new(prefix)).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("'{prefix}' is not an object storage URL"),
        ));
    }
    let prefix = prefix.trim_end_matches('/').to_string();
    let options = options
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
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
    if scheme == ObjectStoreScheme::Memory {
        return Ok(ObjectLocation {
            store: Arc::clone(&MEMORY),
            path,
        });
    }

    let (prefix, options) = registered_options(url);
    let key = format!(
        "{}://{}|{prefix}",
        parsed.scheme(),
        parsed.host_str().unwrap_or("")
    );

    let mut stores = STORES.lock().expect("object storage store cache lock");
    let store = match stores.get(&key) {
        Some(store) => Arc::clone(store),
        None => {
            let store = build(scheme, &parsed, &options)?;
            stores.insert(key, Arc::clone(&store));
            store
        }
    };
    Ok(ObjectLocation { store, path })
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
            let mut builder = AmazonS3Builder::from_env().with_url(url.as_str());
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
            let mut builder = GoogleCloudStorageBuilder::from_env().with_url(url.as_str());
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
            let mut builder = MicrosoftAzureBuilder::from_env().with_url(url.as_str());
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
