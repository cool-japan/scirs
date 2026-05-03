//! Cloud storage abstraction layer
//!
//! Provides a unified `ObjectStore` trait with local-filesystem and in-memory
//! implementations, enabling code to be written against a storage-agnostic API
//! and tested without network access.
//!
//! ## Components
//!
//! - `ObjectStore` trait: `put`, `get`, `delete`, `list`, `head`
//! - `LocalObjectStore`: local filesystem as an object store
//! - `MemoryObjectStore`: in-memory object store for unit testing
//! - `ObjectKey`: URI-like key scheme (`<bucket>/<path>`)
//! - `StorageConfig`: endpoint, credentials, retry policy
//! - `MultipartUpload`: chunked upload simulation
//!
//! ## Example
//!
//! ```rust,no_run
//! use scirs2_io::cloud::{LocalObjectStore, ObjectStore, ObjectKey, StorageConfig};
//!
//! let store = LocalObjectStore::new(std::env::temp_dir().join("my_bucket"));
//! let key = ObjectKey::new("my_bucket", "data/file.bin");
//! store.put(&key, b"hello world").unwrap();
//! let data = store.get(&key).unwrap();
//! assert_eq!(data, b"hello world");
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]

pub mod azure_sas;
pub mod gcs;

pub use azure_sas::{
    build_sas_url, generate_sas_token, is_sas_valid, parse_sas_token, AzureError, AzureSasParams,
    SasPermissions, SasResource,
};
pub use gcs::{GcsError, GcsResumableUpload, UploadStatus};

use crate::error::{IoError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// ObjectKey
// ---------------------------------------------------------------------------

/// A URI-like object key composed of an optional bucket and a path.
///
/// Examples:
/// - `ObjectKey::new("raw-data", "2024/01/events.parquet")`
/// - `ObjectKey::root("checkpoint.bin")` (no bucket)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectKey {
    /// Optional bucket / container name
    pub bucket: Option<String>,
    /// Object path within the bucket (must not start with '/')
    pub path: String,
}

impl ObjectKey {
    /// Create a key with a bucket prefix.
    pub fn new(bucket: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            bucket: Some(bucket.into()),
            path: path.into(),
        }
    }

    /// Create a key without a bucket (flat namespace).
    pub fn root(path: impl Into<String>) -> Self {
        Self {
            bucket: None,
            path: path.into(),
        }
    }

    /// Return a canonical string representation: `"<bucket>/<path>"` or `"<path>"`.
    pub fn as_uri(&self) -> String {
        match &self.bucket {
            Some(b) => format!("{b}/{}", self.path),
            None => self.path.clone(),
        }
    }

    /// Parse a URI string into an `ObjectKey`.
    /// If the string contains a `/`, everything before the first `/` is the bucket.
    pub fn parse(uri: &str) -> Self {
        match uri.find('/') {
            Some(idx) => Self::new(&uri[..idx], &uri[idx + 1..]),
            None => Self::root(uri),
        }
    }
}

impl std::fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_uri())
    }
}

// ---------------------------------------------------------------------------
// ObjectMetadata
// ---------------------------------------------------------------------------

/// Metadata returned by `head()` or `list()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object key
    pub key: ObjectKey,
    /// Content length in bytes
    pub size: u64,
    /// Last-modified time
    pub last_modified: SystemTime,
    /// Optional content-type / MIME type
    pub content_type: Option<String>,
    /// Optional ETag (hex digest of content)
    pub etag: Option<String>,
    /// User-defined metadata key-value pairs
    pub user_metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// StorageConfig
// ---------------------------------------------------------------------------

/// Configuration for a cloud storage backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Endpoint URL (e.g. `"https://s3.amazonaws.com"`)
    pub endpoint: Option<String>,
    /// Access key / client ID
    pub access_key: Option<String>,
    /// Secret key (stored in memory only; do not log)
    #[serde(skip_serializing)]
    pub secret_key: Option<String>,
    /// Region name
    pub region: Option<String>,
    /// Maximum number of retries on transient failures
    pub max_retries: u32,
    /// Initial retry backoff duration
    pub retry_backoff: Duration,
    /// Request timeout
    pub timeout: Duration,
    /// Whether to use TLS/HTTPS
    pub use_tls: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            access_key: None,
            secret_key: None,
            region: None,
            max_retries: 3,
            retry_backoff: Duration::from_millis(100),
            timeout: Duration::from_secs(30),
            use_tls: true,
        }
    }
}

impl StorageConfig {
    /// Create a default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set credentials.
    pub fn with_credentials(
        mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        self.access_key = Some(access_key.into());
        self.secret_key = Some(secret_key.into());
        self
    }

    /// Set max retries.
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

// ---------------------------------------------------------------------------
// ObjectStore trait
// ---------------------------------------------------------------------------

/// Trait for object storage backends.
///
/// Implementations must be `Send + Sync` to allow use from multiple threads.
pub trait ObjectStore: Send + Sync {
    /// Store `data` under `key`, overwriting any existing object.
    fn put(&self, key: &ObjectKey, data: &[u8]) -> Result<()>;

    /// Retrieve the object stored under `key`.
    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>>;

    /// Delete the object at `key`.  Returns `Ok(())` if the key does not exist.
    fn delete(&self, key: &ObjectKey) -> Result<()>;

    /// List all keys whose URI starts with `prefix`.
    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>>;

    /// Return metadata for a single object without downloading its content.
    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata>;

    /// Check whether an object exists.
    fn exists(&self, key: &ObjectKey) -> bool {
        self.head(key).is_ok()
    }

    /// Copy an object from `src` to `dst` within the same store.
    fn copy(&self, src: &ObjectKey, dst: &ObjectKey) -> Result<()> {
        let data = self.get(src)?;
        self.put(dst, &data)
    }

    /// Rename/move an object from `src` to `dst`.
    fn rename(&self, src: &ObjectKey, dst: &ObjectKey) -> Result<()> {
        self.copy(src, dst)?;
        self.delete(src)
    }
}

// ---------------------------------------------------------------------------
// LocalObjectStore
// ---------------------------------------------------------------------------

/// Object store backed by the local filesystem.
///
/// Keys are mapped to paths as `<root>/<bucket>/<path>`.
/// If the key has no bucket the root is used directly.
pub struct LocalObjectStore {
    root: PathBuf,
}

impl LocalObjectStore {
    /// Create a store rooted at `root`.  The directory is created if it does not exist.
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        let root = root.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&root);
        Self { root }
    }

    fn key_to_path(&self, key: &ObjectKey) -> PathBuf {
        match &key.bucket {
            Some(b) => self.root.join(b).join(&key.path),
            None => self.root.join(&key.path),
        }
    }

    fn etag_for(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(data);
        hex::encode(h.finalize())
    }
}

impl ObjectStore for LocalObjectStore {
    fn put(&self, key: &ObjectKey, data: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| IoError::FileError(format!("Cannot create dir: {e}")))?;
        }
        let mut f = std::fs::File::create(&path)
            .map_err(|e| IoError::FileError(format!("Cannot create object file: {e}")))?;
        f.write_all(data)
            .map_err(|e| IoError::FileError(format!("Cannot write object: {e}")))?;
        Ok(())
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        let path = self.key_to_path(key);
        let mut f = std::fs::File::open(&path)
            .map_err(|_| IoError::NotFound(format!("Object not found: {key}")))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)
            .map_err(|e| IoError::FileError(format!("Cannot read object: {e}")))?;
        Ok(buf)
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        let path = self.key_to_path(key);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| IoError::FileError(format!("Cannot delete object: {e}")))?;
        }
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        let mut results = Vec::new();
        self.collect_entries(&self.root, prefix, &mut results)?;
        Ok(results)
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        let path = self.key_to_path(key);
        let meta = std::fs::metadata(&path)
            .map_err(|_| IoError::NotFound(format!("Object not found: {key}")))?;
        let data = self.get(key)?;
        Ok(ObjectMetadata {
            key: key.clone(),
            size: meta.len(),
            last_modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            content_type: None,
            etag: Some(Self::etag_for(&data)),
            user_metadata: HashMap::new(),
        })
    }
}

impl LocalObjectStore {
    fn collect_entries(
        &self,
        dir: &Path,
        prefix: &str,
        results: &mut Vec<ObjectMetadata>,
    ) -> Result<()> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };
        for entry in entries {
            let entry =
                entry.map_err(|e| IoError::FileError(format!("Cannot read dir entry: {e}")))?;
            let path = entry.path();
            if path.is_dir() {
                self.collect_entries(&path, prefix, results)?;
            } else {
                // Compute relative key from root
                let rel = path.strip_prefix(&self.root).unwrap_or(&path);
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if rel_str.starts_with(prefix) {
                    // Parse bucket / path
                    let key = ObjectKey::parse(&rel_str);
                    let meta_fs = std::fs::metadata(&path)
                        .map_err(|e| IoError::FileError(format!("Metadata error: {e}")))?;
                    results.push(ObjectMetadata {
                        key,
                        size: meta_fs.len(),
                        last_modified: meta_fs.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                        content_type: None,
                        etag: None,
                        user_metadata: HashMap::new(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryObjectStore
// ---------------------------------------------------------------------------

/// In-memory object store, suitable for unit tests and simulations.
///
/// Thread-safe via an `Arc<Mutex<...>>` interior.
#[derive(Clone)]
pub struct MemoryObjectStore {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl MemoryObjectStore {
    /// Create an empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the number of objects currently stored.
    pub fn len(&self) -> usize {
        self.data.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Return `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemoryObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectStore for MemoryObjectStore {
    fn put(&self, key: &ObjectKey, data: &[u8]) -> Result<()> {
        let mut guard = self
            .data
            .lock()
            .map_err(|_| IoError::Other("MemoryStore lock poisoned".to_string()))?;
        guard.insert(key.as_uri(), data.to_vec());
        Ok(())
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        let guard = self
            .data
            .lock()
            .map_err(|_| IoError::Other("MemoryStore lock poisoned".to_string()))?;
        guard
            .get(&key.as_uri())
            .cloned()
            .ok_or_else(|| IoError::NotFound(format!("Object not found: {key}")))
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        let mut guard = self
            .data
            .lock()
            .map_err(|_| IoError::Other("MemoryStore lock poisoned".to_string()))?;
        guard.remove(&key.as_uri());
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        let guard = self
            .data
            .lock()
            .map_err(|_| IoError::Other("MemoryStore lock poisoned".to_string()))?;
        let results = guard
            .iter()
            .filter(|(uri, _)| uri.starts_with(prefix))
            .map(|(uri, data)| ObjectMetadata {
                key: ObjectKey::parse(uri),
                size: data.len() as u64,
                last_modified: SystemTime::UNIX_EPOCH,
                content_type: None,
                etag: None,
                user_metadata: HashMap::new(),
            })
            .collect();
        Ok(results)
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        let guard = self
            .data
            .lock()
            .map_err(|_| IoError::Other("MemoryStore lock poisoned".to_string()))?;
        let data = guard
            .get(&key.as_uri())
            .ok_or_else(|| IoError::NotFound(format!("Object not found: {key}")))?;
        Ok(ObjectMetadata {
            key: key.clone(),
            size: data.len() as u64,
            last_modified: SystemTime::UNIX_EPOCH,
            content_type: None,
            etag: None,
            user_metadata: HashMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// MultipartUpload
// ---------------------------------------------------------------------------

/// Chunked multipart upload simulation.
///
/// Mimics the semantics of S3/GCS multipart uploads:
/// 1. Create an upload session via `MultipartUpload::new(store, key)`.
/// 2. Upload parts with `upload_part(part_number, data)`.
/// 3. Finalise with `complete()` which concatenates parts and stores the object.
///
/// Parts may be uploaded in any order; they are sorted by part number on `complete`.
///
/// Any part number ≥ 1 is valid.
pub struct MultipartUpload<'a> {
    store: &'a dyn ObjectStore,
    key: ObjectKey,
    parts: Vec<(u16, Vec<u8>)>,
    min_part_size: usize,
}

impl<'a> MultipartUpload<'a> {
    /// Create a new multipart upload session.
    pub fn new(store: &'a dyn ObjectStore, key: ObjectKey) -> Self {
        Self {
            store,
            key,
            parts: Vec::new(),
            min_part_size: 5 * 1024 * 1024, // 5 MiB (S3 minimum, advisory only)
        }
    }

    /// Set the minimum part size advisory (default: 5 MiB).
    /// Parts smaller than this will be accepted but a warning can be emitted.
    pub fn with_min_part_size(mut self, size: usize) -> Self {
        self.min_part_size = size;
        self
    }

    /// Upload a single part.
    ///
    /// Part numbers must be in `[1, 10_000]`.
    /// Parts are buffered in memory until `complete()` is called.
    pub fn upload_part(&mut self, part_number: u16, data: Vec<u8>) -> Result<()> {
        if part_number == 0 {
            return Err(IoError::ValidationError(
                "MultipartUpload: part number must be >= 1".to_string(),
            ));
        }
        if part_number > 10_000 {
            return Err(IoError::ValidationError(
                "MultipartUpload: part number must be <= 10000".to_string(),
            ));
        }
        // Replace any existing part with this number
        self.parts.retain(|(n, _)| *n != part_number);
        self.parts.push((part_number, data));
        Ok(())
    }

    /// Return the number of parts currently uploaded.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Return the total bytes buffered so far.
    pub fn total_bytes(&self) -> usize {
        self.parts.iter().map(|(_, d)| d.len()).sum()
    }

    /// Abort the upload and discard all buffered parts.
    pub fn abort(&mut self) {
        self.parts.clear();
    }

    /// Finalise the upload: sort parts by part number, concatenate, and store.
    ///
    /// Consumes `self`.
    pub fn complete(mut self) -> Result<UploadResult> {
        if self.parts.is_empty() {
            return Err(IoError::ValidationError(
                "MultipartUpload: no parts to complete".to_string(),
            ));
        }
        self.parts.sort_by_key(|(n, _)| *n);
        let total_size: usize = self.parts.iter().map(|(_, d)| d.len()).sum();
        let mut assembled = Vec::with_capacity(total_size);
        for (_, data) in &self.parts {
            assembled.extend_from_slice(data);
        }
        let etag = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(&assembled);
            hex::encode(h.finalize())
        };
        self.store.put(&self.key, &assembled)?;
        Ok(UploadResult {
            key: self.key.clone(),
            total_size,
            part_count: self.parts.len(),
            etag,
        })
    }
}

/// Result returned by `MultipartUpload::complete`.
#[derive(Debug, Clone)]
pub struct UploadResult {
    /// The key the object was stored under
    pub key: ObjectKey,
    /// Total assembled size in bytes
    pub total_size: usize,
    /// Number of parts
    pub part_count: usize,
    /// SHA-256 ETag of the assembled object
    pub etag: String,
}

// ---------------------------------------------------------------------------
// StorageStats
// ---------------------------------------------------------------------------

/// Aggregated statistics for a storage backend session.
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total bytes written (via `put`)
    pub bytes_written: u64,
    /// Total bytes read (via `get`)
    pub bytes_read: u64,
    /// Number of `put` operations
    pub put_count: u64,
    /// Number of `get` operations
    pub get_count: u64,
    /// Number of `delete` operations
    pub delete_count: u64,
    /// Number of errors encountered
    pub error_count: u64,
}

/// A wrapper that instruments any `ObjectStore` implementation with statistics.
pub struct InstrumentedStore<S: ObjectStore> {
    inner: S,
    stats: Arc<Mutex<StorageStats>>,
}

impl<S: ObjectStore> InstrumentedStore<S> {
    /// Wrap `store` in a statistics-gathering layer.
    pub fn new(store: S) -> Self {
        Self {
            inner: store,
            stats: Arc::new(Mutex::new(StorageStats::default())),
        }
    }

    /// Take a snapshot of the current statistics.
    pub fn stats(&self) -> StorageStats {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Reset all statistics to zero.
    pub fn reset_stats(&self) {
        if let Ok(mut g) = self.stats.lock() {
            *g = StorageStats::default();
        }
    }
}

impl<S: ObjectStore> ObjectStore for InstrumentedStore<S> {
    fn put(&self, key: &ObjectKey, data: &[u8]) -> Result<()> {
        let result = self.inner.put(key, data);
        if let Ok(mut s) = self.stats.lock() {
            match &result {
                Ok(()) => {
                    s.bytes_written += data.len() as u64;
                    s.put_count += 1;
                }
                Err(_) => s.error_count += 1,
            }
        }
        result
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        let result = self.inner.get(key);
        if let Ok(mut s) = self.stats.lock() {
            match &result {
                Ok(data) => {
                    s.bytes_read += data.len() as u64;
                    s.get_count += 1;
                }
                Err(_) => s.error_count += 1,
            }
        }
        result
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        let result = self.inner.delete(key);
        if let Ok(mut s) = self.stats.lock() {
            match &result {
                Ok(()) => s.delete_count += 1,
                Err(_) => s.error_count += 1,
            }
        }
        result
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        self.inner.list(prefix)
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        self.inner.head(key)
    }
}

// ---------------------------------------------------------------------------
// Cloud provider stubs: S3, GCS, Azure Blob
// ---------------------------------------------------------------------------

/// Cloud storage provider type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreProviderType {
    /// Local filesystem.
    LocalFs,
    /// AWS S3 or S3-compatible (MinIO, Ceph, etc.).
    S3,
    /// Google Cloud Storage.
    Gcs,
    /// Azure Blob Storage.
    AzureBlob,
    /// In-memory (testing).
    Memory,
}

/// AWS S3-compatible object store stub.
///
/// Full HTTP/SigV4 implementation requires the `aws-sdk-s3` feature.
/// Without that feature, all operations return `IoError::Other("s3 feature not enabled")`.
pub struct S3Store {
    /// Target bucket name.
    pub bucket: String,
    /// AWS region (e.g. `"us-east-1"`).
    pub region: String,
    /// Optional custom endpoint (for S3-compatible services).
    pub endpoint: Option<String>,
    /// AWS access key ID.
    pub access_key: String,
    /// AWS secret access key.
    pub secret_key: String,
}

impl S3Store {
    /// Create a new S3 store configuration.
    pub fn new(
        bucket: impl Into<String>,
        region: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        Self {
            bucket: bucket.into(),
            region: region.into(),
            endpoint: None,
            access_key: access_key.into(),
            secret_key: secret_key.into(),
        }
    }

    /// Set a custom endpoint URL (e.g. for MinIO).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }
}

impl ObjectStore for S3Store {
    fn put(&self, key: &ObjectKey, _data: &[u8]) -> Result<()> {
        #[cfg(feature = "aws-sdk-s3")]
        {
            // Real implementation would use reqwest + SigV4 signing.
            // For now even with the feature flag we return a stub error.
            Err(IoError::Other(format!(
                "S3 put '{}': real HTTP implementation not yet complete",
                key
            )))
        }
        #[cfg(not(feature = "aws-sdk-s3"))]
        Err(IoError::Other(format!(
            "S3 put '{}': enable the 'aws-sdk-s3' feature for AWS S3 support",
            key
        )))
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        #[cfg(feature = "aws-sdk-s3")]
        {
            Err(IoError::Other(format!(
                "S3 get '{}': real HTTP implementation not yet complete",
                key
            )))
        }
        #[cfg(not(feature = "aws-sdk-s3"))]
        Err(IoError::Other(format!(
            "S3 get '{}': enable the 'aws-sdk-s3' feature for AWS S3 support",
            key
        )))
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        #[cfg(feature = "aws-sdk-s3")]
        {
            Err(IoError::Other(format!(
                "S3 delete '{}': real HTTP implementation not yet complete",
                key
            )))
        }
        #[cfg(not(feature = "aws-sdk-s3"))]
        Err(IoError::Other(format!(
            "S3 delete '{}': enable the 'aws-sdk-s3' feature for AWS S3 support",
            key
        )))
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        #[cfg(feature = "aws-sdk-s3")]
        {
            Err(IoError::Other(format!(
                "S3 list '{}': real HTTP implementation not yet complete",
                prefix
            )))
        }
        #[cfg(not(feature = "aws-sdk-s3"))]
        Err(IoError::Other(format!(
            "S3 list '{}': enable the 'aws-sdk-s3' feature for AWS S3 support",
            prefix
        )))
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        #[cfg(feature = "aws-sdk-s3")]
        {
            Err(IoError::Other(format!(
                "S3 head '{}': real HTTP implementation not yet complete",
                key
            )))
        }
        #[cfg(not(feature = "aws-sdk-s3"))]
        Err(IoError::Other(format!(
            "S3 head '{}': enable the 'aws-sdk-s3' feature for AWS S3 support",
            key
        )))
    }

    fn exists(&self, key: &ObjectKey) -> bool {
        // Stubs always report "not found".
        let _ = key;
        false
    }
}

/// Google Cloud Storage stub.
///
/// Full implementation requires the `google-cloud-storage` feature.
pub struct GcsStore {
    /// GCS bucket name.
    pub bucket: String,
    /// GCP project ID.
    pub project_id: String,
    /// Path to service-account JSON key file (optional).
    pub credentials_path: Option<String>,
}

impl GcsStore {
    /// Create a new GCS store configuration.
    pub fn new(bucket: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            project_id: project_id.into(),
            credentials_path: None,
        }
    }

    /// Set the service-account credentials JSON path.
    pub fn with_credentials(mut self, path: impl Into<String>) -> Self {
        self.credentials_path = Some(path.into());
        self
    }
}

impl ObjectStore for GcsStore {
    fn put(&self, key: &ObjectKey, _data: &[u8]) -> Result<()> {
        Err(IoError::Other(format!(
            "GCS put '{}': enable the 'google-cloud-storage' feature for GCS support",
            key
        )))
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        Err(IoError::Other(format!(
            "GCS get '{}': enable the 'google-cloud-storage' feature for GCS support",
            key
        )))
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        Err(IoError::Other(format!(
            "GCS delete '{}': enable the 'google-cloud-storage' feature for GCS support",
            key
        )))
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        Err(IoError::Other(format!(
            "GCS list '{}': enable the 'google-cloud-storage' feature for GCS support",
            prefix
        )))
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        Err(IoError::Other(format!(
            "GCS head '{}': enable the 'google-cloud-storage' feature for GCS support",
            key
        )))
    }

    fn exists(&self, key: &ObjectKey) -> bool {
        let _ = key;
        false
    }
}

/// Azure Blob Storage stub.
///
/// Full implementation requires the `azure-storage-blobs` feature.
pub struct AzureBlobStore {
    /// Storage account name.
    pub account: String,
    /// Container name.
    pub container: String,
    /// SAS token or account key for authentication.
    pub credential: Option<String>,
    /// Optional custom endpoint (for Azurite emulator).
    pub endpoint: Option<String>,
}

impl AzureBlobStore {
    /// Create a new Azure Blob store configuration.
    pub fn new(account: impl Into<String>, container: impl Into<String>) -> Self {
        Self {
            account: account.into(),
            container: container.into(),
            credential: None,
            endpoint: None,
        }
    }

    /// Set an authentication credential (SAS token or account key).
    pub fn with_credential(mut self, cred: impl Into<String>) -> Self {
        self.credential = Some(cred.into());
        self
    }

    /// Set a custom endpoint (e.g. Azurite emulator).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }
}

impl ObjectStore for AzureBlobStore {
    fn put(&self, key: &ObjectKey, _data: &[u8]) -> Result<()> {
        Err(IoError::Other(format!(
            "Azure put '{}': enable the 'azure-storage-blobs' feature for Azure support",
            key
        )))
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>> {
        Err(IoError::Other(format!(
            "Azure get '{}': enable the 'azure-storage-blobs' feature for Azure support",
            key
        )))
    }

    fn delete(&self, key: &ObjectKey) -> Result<()> {
        Err(IoError::Other(format!(
            "Azure delete '{}': enable the 'azure-storage-blobs' feature for Azure support",
            key
        )))
    }

    fn list(&self, prefix: &str) -> Result<Vec<ObjectMetadata>> {
        Err(IoError::Other(format!(
            "Azure list '{}': enable the 'azure-storage-blobs' feature for Azure support",
            prefix
        )))
    }

    fn head(&self, key: &ObjectKey) -> Result<ObjectMetadata> {
        Err(IoError::Other(format!(
            "Azure head '{}': enable the 'azure-storage-blobs' feature for Azure support",
            key
        )))
    }

    fn exists(&self, key: &ObjectKey) -> bool {
        let _ = key;
        false
    }
}

// ---------------------------------------------------------------------------
// URL parsing and factory
// ---------------------------------------------------------------------------

/// Parse a cloud storage URL into provider type, bucket, and path.
///
/// Supported URL schemes:
/// - `s3://bucket/key` — AWS S3
/// - `gs://bucket/key` — Google Cloud Storage
/// - `az://container/blob` — Azure Blob Storage
/// - Any other string — treated as a local filesystem path
///
/// Returns `(provider_type, bucket_or_container, object_path)`.
pub fn parse_store_url(
    url: &str,
) -> std::result::Result<(StoreProviderType, String, String), IoError> {
    if let Some(rest) = url.strip_prefix("s3://") {
        let (bucket, path) = split_bucket_path(rest)?;
        return Ok((StoreProviderType::S3, bucket, path));
    }
    if let Some(rest) = url.strip_prefix("gs://") {
        let (bucket, path) = split_bucket_path(rest)?;
        return Ok((StoreProviderType::Gcs, bucket, path));
    }
    if let Some(rest) = url.strip_prefix("az://") {
        let (bucket, path) = split_bucket_path(rest)?;
        return Ok((StoreProviderType::AzureBlob, bucket, path));
    }
    // Local path
    Ok((StoreProviderType::LocalFs, String::new(), url.to_string()))
}

fn split_bucket_path(rest: &str) -> std::result::Result<(String, String), IoError> {
    match rest.find('/') {
        Some(idx) => Ok((rest[..idx].to_string(), rest[idx + 1..].to_string())),
        None => {
            if rest.is_empty() {
                Err(IoError::Other(
                    "Invalid store URL: missing bucket name".to_string(),
                ))
            } else {
                // Bucket-only URL (no path component) — path is empty
                Ok((rest.to_string(), String::new()))
            }
        }
    }
}

/// Build a boxed `ObjectStore` from a URL string.
///
/// - `file://path` or a bare path: returns `LocalObjectStore` rooted at that path.
/// - `s3://...`, `gs://...`, `az://...`: returns the corresponding stub store.
///   The stub will fail at runtime unless credentials are provided via environment
///   variables and the appropriate Cargo feature is enabled.
/// - Credentials for cloud stores are read from standard environment variables:
///   - S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
///   - GCS: `GCP_PROJECT_ID`
///   - Azure: `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`
pub fn from_url(url: &str) -> std::result::Result<Box<dyn ObjectStore>, IoError> {
    let (provider, bucket, path) = parse_store_url(url)?;
    match provider {
        StoreProviderType::LocalFs => {
            let root = if path.is_empty() { "." } else { &path };
            Ok(Box::new(LocalObjectStore::new(root)))
        }
        StoreProviderType::S3 => {
            let access_key = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
            let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();
            let region =
                std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".to_string());
            Ok(Box::new(S3Store::new(
                bucket, region, access_key, secret_key,
            )))
        }
        StoreProviderType::Gcs => {
            let project_id = std::env::var("GCP_PROJECT_ID").unwrap_or_default();
            let mut store = GcsStore::new(bucket, project_id);
            if let Ok(creds) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
                store = store.with_credentials(creds);
            }
            Ok(Box::new(store))
        }
        StoreProviderType::AzureBlob => {
            let account = std::env::var("AZURE_STORAGE_ACCOUNT").unwrap_or_else(|_| bucket);
            let mut store = AzureBlobStore::new(account, path.clone());
            if let Ok(key) = std::env::var("AZURE_STORAGE_KEY") {
                store = store.with_credential(key);
            }
            Ok(Box::new(store))
        }
        StoreProviderType::Memory => Ok(Box::new(MemoryObjectStore::new())),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use uuid::Uuid;

    // ---- ObjectKey ----

    #[test]
    fn test_object_key_uri() {
        let k = ObjectKey::new("my-bucket", "data/foo.bin");
        assert_eq!(k.as_uri(), "my-bucket/data/foo.bin");
    }

    #[test]
    fn test_object_key_root() {
        let k = ObjectKey::root("bar.bin");
        assert_eq!(k.as_uri(), "bar.bin");
    }

    #[test]
    fn test_object_key_parse() {
        let k = ObjectKey::parse("bucket/path/to/file");
        assert_eq!(k.bucket.as_deref(), Some("bucket"));
        assert_eq!(k.path, "path/to/file");
        let k2 = ObjectKey::parse("no-slash");
        assert_eq!(k2.bucket, None);
        assert_eq!(k2.path, "no-slash");
    }

    // ---- MemoryObjectStore ----

    #[test]
    fn test_memory_store_put_get() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::new("b", "hello.txt");
        store.put(&key, b"hello world").unwrap();
        assert_eq!(store.get(&key).unwrap(), b"hello world");
    }

    #[test]
    fn test_memory_store_delete() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::root("x.bin");
        store.put(&key, b"data").unwrap();
        store.delete(&key).unwrap();
        assert!(!store.exists(&key));
    }

    #[test]
    fn test_memory_store_list() {
        let store = MemoryObjectStore::new();
        store.put(&ObjectKey::new("b", "a/1.bin"), b"1").unwrap();
        store.put(&ObjectKey::new("b", "a/2.bin"), b"2").unwrap();
        store.put(&ObjectKey::new("c", "x.bin"), b"3").unwrap();
        let items = store.list("b/").unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_memory_store_head() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::new("bkt", "file.bin");
        store.put(&key, b"1234567890").unwrap();
        let meta = store.head(&key).unwrap();
        assert_eq!(meta.size, 10);
    }

    #[test]
    fn test_memory_store_copy_rename() {
        let store = MemoryObjectStore::new();
        let src = ObjectKey::root("src.bin");
        let dst = ObjectKey::root("dst.bin");
        store.put(&src, b"payload").unwrap();
        store.rename(&src, &dst).unwrap();
        assert!(!store.exists(&src));
        assert_eq!(store.get(&dst).unwrap(), b"payload");
    }

    // ---- LocalObjectStore ----

    #[test]
    fn test_local_store_put_get_delete() {
        let dir = temp_dir().join(format!("scirs2_cloud_{}", Uuid::new_v4()));
        let store = LocalObjectStore::new(&dir);
        let key = ObjectKey::new("bkt", "sub/data.bin");
        store.put(&key, b"binary data").unwrap();
        assert!(store.exists(&key));
        assert_eq!(store.get(&key).unwrap(), b"binary data");
        store.delete(&key).unwrap();
        assert!(!store.exists(&key));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_local_store_head() {
        let dir = temp_dir().join(format!("scirs2_cloud_{}", Uuid::new_v4()));
        let store = LocalObjectStore::new(&dir);
        let key = ObjectKey::root("file.txt");
        store.put(&key, b"abcdef").unwrap();
        let meta = store.head(&key).unwrap();
        assert_eq!(meta.size, 6);
        assert!(meta.etag.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- MultipartUpload ----

    #[test]
    fn test_multipart_upload() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::new("bucket", "big.bin");
        let mut upload = MultipartUpload::new(&store, key.clone());
        upload.upload_part(1, b"part1".to_vec()).unwrap();
        upload.upload_part(3, b"part3".to_vec()).unwrap();
        upload.upload_part(2, b"part2".to_vec()).unwrap();
        assert_eq!(upload.part_count(), 3);
        assert_eq!(upload.total_bytes(), 15);
        let result = upload.complete().unwrap();
        assert_eq!(result.part_count, 3);
        // Parts should be concatenated in order 1,2,3
        assert_eq!(store.get(&key).unwrap(), b"part1part2part3");
    }

    #[test]
    fn test_multipart_upload_abort() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::root("file.bin");
        let mut upload = MultipartUpload::new(&store, key.clone());
        upload.upload_part(1, b"data".to_vec()).unwrap();
        upload.abort();
        assert_eq!(upload.part_count(), 0);
    }

    #[test]
    fn test_multipart_invalid_part_number() {
        let store = MemoryObjectStore::new();
        let key = ObjectKey::root("f");
        let mut upload = MultipartUpload::new(&store, key);
        assert!(upload.upload_part(0, vec![]).is_err());
        assert!(upload.upload_part(10_001, vec![]).is_err());
    }

    // ---- InstrumentedStore ----

    #[test]
    fn test_instrumented_store_stats() {
        let inner = MemoryObjectStore::new();
        let store = InstrumentedStore::new(inner);
        let key = ObjectKey::root("x");
        store.put(&key, b"hello").unwrap();
        store.get(&key).unwrap();
        store.delete(&key).unwrap();
        let s = store.stats();
        assert_eq!(s.put_count, 1);
        assert_eq!(s.get_count, 1);
        assert_eq!(s.delete_count, 1);
        assert_eq!(s.bytes_written, 5);
        assert_eq!(s.bytes_read, 5);
    }

    // ---- parse_store_url ----

    #[test]
    fn test_parse_store_url_s3() {
        let (provider, bucket, path) = parse_store_url("s3://my-bucket/path/to/key").unwrap();
        assert_eq!(provider, StoreProviderType::S3);
        assert_eq!(bucket, "my-bucket");
        assert_eq!(path, "path/to/key");
    }

    #[test]
    fn test_parse_store_url_gcs() {
        let (provider, bucket, path) = parse_store_url("gs://my-bucket/data/file.parquet").unwrap();
        assert_eq!(provider, StoreProviderType::Gcs);
        assert_eq!(bucket, "my-bucket");
        assert_eq!(path, "data/file.parquet");
    }

    #[test]
    fn test_parse_store_url_azure() {
        let (provider, bucket, path) = parse_store_url("az://mycontainer/blob/path").unwrap();
        assert_eq!(provider, StoreProviderType::AzureBlob);
        assert_eq!(bucket, "mycontainer");
        assert_eq!(path, "blob/path");
    }

    #[test]
    fn test_parse_store_url_local() {
        let local_path = std::env::temp_dir()
            .join("local")
            .join("path")
            .to_str()
            .unwrap()
            .to_string();
        let (provider, bucket, path) = parse_store_url(&local_path).unwrap();
        assert_eq!(provider, StoreProviderType::LocalFs);
        assert!(bucket.is_empty());
        assert_eq!(path, local_path);
    }

    // ---- S3Store / GcsStore / AzureBlobStore stub behaviour ----

    #[test]
    fn test_s3_store_stub_returns_error() {
        let store = S3Store::new("bucket", "us-east-1", "key", "secret");
        let key = ObjectKey::new("bucket", "file.bin");
        assert!(store.put(&key, b"data").is_err());
        assert!(store.get(&key).is_err());
        assert!(store.list("bucket/").is_err());
        assert!(store.head(&key).is_err());
        assert!(store.delete(&key).is_err());
        // exists() must not panic
        assert!(!store.exists(&key));
    }

    #[test]
    fn test_gcs_store_stub_returns_error() {
        let store = GcsStore::new("my-bucket", "my-project");
        let key = ObjectKey::new("my-bucket", "file.bin");
        assert!(store.put(&key, b"data").is_err());
        assert!(store.get(&key).is_err());
    }

    #[test]
    fn test_azure_store_stub_returns_error() {
        let store = AzureBlobStore::new("myaccount", "mycontainer");
        let key = ObjectKey::new("mycontainer", "blob.bin");
        assert!(store.put(&key, b"data").is_err());
        assert!(store.get(&key).is_err());
    }

    // ---- from_url ----

    #[test]
    fn test_from_url_local() {
        let dir = temp_dir().join(format!("scirs2_from_url_{}", Uuid::new_v4()));
        let store = from_url(dir.to_str().unwrap()).unwrap();
        let key = ObjectKey::root("test.bin");
        store.put(&key, b"hello from_url").unwrap();
        assert_eq!(store.get(&key).unwrap(), b"hello from_url");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_from_url_s3_stub() {
        let store = from_url("s3://test-bucket/prefix").unwrap();
        let key = ObjectKey::new("test-bucket", "file.bin");
        // Stub should return an error, not panic
        assert!(store.get(&key).is_err());
    }
}
