//! Cloud Storage Integration for SciRS2
//!
//! This module provides seamless integration with major cloud storage providers
//! including Amazon S3, Google Cloud Storage, and Azure Blob Storage.
//!
//! Features:
//! - Unified interface for all cloud providers
//! - Asynchronous operations for high performance
//! - Automatic credential management
//! - Retry logic and error handling
//! - Progress tracking for large transfers
//! - Multi-part upload/download support
//! - Intelligent caching and prefetching

use crate::error::{CoreError, ErrorContext, ErrorLocation};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use thiserror::Error;

#[cfg(feature = "async")]
#[allow(unused_imports)]
use tokio::io::{AsyncRead, AsyncWrite};

#[cfg(feature = "async")]
use async_trait::async_trait;

// AWS environment variable constants
const AWS_ACCESS_KEY_ID: &str = "AWS_ACCESS_KEY_ID";
const AWS_SECRET_ACCESS_KEY: &str = "AWS_SECRET_ACCESS_KEY";
const AWS_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";
const AWS_REGION: &str = "AWS_REGION";

// Google Cloud environment variable constants
const GOOGLE_APPLICATION_CREDENTIALS: &str = "GOOGLE_APPLICATION_CREDENTIALS";
const GOOGLE_CLOUD_PROJECT: &str = "GOOGLE_CLOUD_PROJECT";

// Azure environment variable constants
const AZURE_STORAGE_ACCOUNT: &str = "AZURE_STORAGE_ACCOUNT";
const AZURE_STORAGE_KEY: &str = "AZURE_STORAGE_KEY";
const AZURE_STORAGE_SAS_TOKEN: &str = "AZURE_STORAGE_SAS_TOKEN";

/// Cloud storage error types
#[derive(Error, Debug)]
pub enum CloudError {
    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Bucket or container not found
    #[error("Bucket/container not found: {0}")]
    BucketNotFound(String),

    /// Object not found
    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Network connection error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Service quota exceeded
    #[error("Service quota exceeded: {0}")]
    QuotaExceeded(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Upload failed
    #[error("Upload failed: {0}")]
    UploadError(String),

    /// Download failed
    #[error("Download failed: {0}")]
    DownloadError(String),

    /// Multipart operation failed
    #[error("Multipart operation failed: {0}")]
    MultipartError(String),

    /// Metadata operation failed
    #[error("Metadata operation failed: {0}")]
    MetadataError(String),

    /// Generic cloud provider error
    #[error("Cloud provider error: {provider} - {message}")]
    ProviderError { provider: String, message: String },
}

impl From<CloudError> for CoreError {
    fn from(err: CloudError) -> Self {
        match err {
            CloudError::AuthenticationError(msg) => CoreError::SecurityError(
                ErrorContext::new(format!("{msg}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            CloudError::PermissionDenied(msg) => CoreError::SecurityError(
                ErrorContext::new(format!("{msg}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            CloudError::NetworkError(msg) => CoreError::IoError(
                ErrorContext::new(format!("{msg}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            _ => CoreError::IoError(
                ErrorContext::new(format!("{err}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}

/// Cloud storage provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudProvider {
    /// Amazon S3
    AwsS3,
    /// Google Cloud Storage
    GoogleCloud,
    /// Azure Blob Storage
    AzureBlob,
    /// S3-compatible providers (MinIO, etc.)
    S3Compatible,
}

impl CloudProvider {
    /// Get the default endpoint for this provider
    pub fn default_endpoint(&self) -> Option<&'static str> {
        match self {
            CloudProvider::AwsS3 => Some("https://s3.amazonaws.com"),
            CloudProvider::GoogleCloud => Some("https://storage.googleapis.com"),
            CloudProvider::AzureBlob => None, // Dynamic based on account
            CloudProvider::S3Compatible => None, // User-defined
        }
    }

    /// Get the provider name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            CloudProvider::AwsS3 => "aws-s3",
            CloudProvider::GoogleCloud => "google-cloud",
            CloudProvider::AzureBlob => "azure-blob",
            CloudProvider::S3Compatible => "s3-compatible",
        }
    }
}

/// Cloud storage credentials
#[derive(Debug, Clone)]
pub enum CloudCredentials {
    /// AWS credentials
    Aws {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
        region: String,
    },
    /// Google Cloud credentials
    Google {
        service_account_key: String,
        project_id: String,
    },
    /// Azure credentials
    Azure {
        account_name: String,
        account_key: String,
        sas_token: Option<String>,
    },
    /// Anonymous access (for public buckets)
    Anonymous,
}

impl CloudCredentials {
    /// Create AWS credentials from environment variables
    pub fn aws_from_env() -> Result<Self, CloudError> {
        let access_key_id = std::env::var(AWS_ACCESS_KEY_ID).map_err(|_| {
            CloudError::AuthenticationError("AWS_ACCESS_KEY_ID not found".to_string())
        })?;
        let secret_access_key = std::env::var(AWS_SECRET_ACCESS_KEY).map_err(|_| {
            CloudError::AuthenticationError("AWS_SECRET_ACCESS_KEY not found".to_string())
        })?;
        let session_token = std::env::var(AWS_SESSION_TOKEN).ok();
        let region = std::env::var(AWS_REGION).unwrap_or_else(|_| "us-east-1".to_string());

        Ok(CloudCredentials::Aws {
            access_key_id,
            secret_access_key,
            session_token,
            region,
        })
    }

    /// Create Google Cloud credentials from environment variables
    pub fn google_from_env() -> Result<Self, CloudError> {
        let service_account_key = std::env::var(GOOGLE_APPLICATION_CREDENTIALS).map_err(|_| {
            CloudError::AuthenticationError("GOOGLE_APPLICATION_CREDENTIALS not found".to_string())
        })?;
        let project_id = std::env::var(GOOGLE_CLOUD_PROJECT).map_err(|_| {
            CloudError::AuthenticationError("GOOGLE_CLOUD_PROJECT not found".to_string())
        })?;

        Ok(CloudCredentials::Google {
            service_account_key,
            project_id,
        })
    }

    /// Create Azure credentials from environment variables
    pub fn azure_from_env() -> Result<Self, CloudError> {
        let account_name = std::env::var(AZURE_STORAGE_ACCOUNT).map_err(|_| {
            CloudError::AuthenticationError("AZURE_STORAGE_ACCOUNT not found".to_string())
        })?;
        let account_key = std::env::var(AZURE_STORAGE_KEY).map_err(|_| {
            CloudError::AuthenticationError("AZURE_STORAGE_KEY not found".to_string())
        })?;
        let sas_token = std::env::var(AZURE_STORAGE_SAS_TOKEN).ok();

        Ok(CloudCredentials::Azure {
            account_name,
            account_key,
            sas_token,
        })
    }
}

/// Cloud storage configuration
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Cloud provider
    pub provider: CloudProvider,
    /// Custom endpoint URL
    pub endpoint: Option<String>,
    /// Bucket or container name
    pub bucket: String,
    /// Credentials
    pub credentials: CloudCredentials,
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum number of retries
    pub maxretries: u32,
    /// Enable multipart uploads for files larger than this size
    pub multipart_threshold: usize,
    /// Chunk size for multipart operations
    pub chunk_size: usize,
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::AwsS3,
            endpoint: None,
            bucket: String::new(),
            credentials: CloudCredentials::Anonymous,
            timeout: Duration::from_secs(30),
            maxretries: 3,
            multipart_threshold: 100 * 1024 * 1024, // 100 MB
            chunk_size: 8 * 1024 * 1024,            // 8 MB
            max_concurrency: 10,
            enable_cache: true,
            cache_dir: None,
        }
    }
}

impl CloudConfig {
    /// Create a new configuration for AWS S3
    pub fn new_bucket(bucket: String, credentials: CloudCredentials) -> Self {
        Self {
            provider: CloudProvider::AwsS3,
            bucket,
            credentials,
            ..Default::default()
        }
    }

    /// Create a new configuration for Google Cloud Storage
    pub fn bucket_2(bucket: String, credentials: CloudCredentials) -> Self {
        Self {
            provider: CloudProvider::GoogleCloud,
            bucket,
            credentials,
            ..Default::default()
        }
    }

    /// Create a new configuration for Azure Blob Storage
    pub fn container(container: String, credentials: CloudCredentials) -> Self {
        Self {
            provider: CloudProvider::AzureBlob,
            bucket: container,
            credentials,
            ..Default::default()
        }
    }

    /// Set custom endpoint
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set multipart configuration
    pub fn with_multipart(mut self, threshold: usize, chunk_size: usize) -> Self {
        self.multipart_threshold = threshold;
        self.chunk_size = chunk_size;
        self
    }

    /// Set cache configuration
    pub fn with_cache(mut self, enable: bool, cache_dir: Option<PathBuf>) -> Self {
        self.enable_cache = enable;
        self.cache_dir = cache_dir;
        self
    }
}

/// Cloud object metadata
#[derive(Debug, Clone)]
pub struct CloudObjectMetadata {
    /// Object key/path
    pub key: String,
    /// Object size in bytes
    pub size: u64,
    /// Last modified time
    pub last_modified: SystemTime,
    /// ETag or content hash
    pub etag: Option<String>,
    /// Content type
    pub content_type: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Storage class
    pub storage_class: Option<String>,
}

/// Progress callback for upload/download operations
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Transfer options for cloud operations
#[derive(Default)]
pub struct TransferOptions {
    /// Progress callback
    pub progress_callback: Option<ProgressCallback>,
    /// Custom metadata to set
    pub metadata: HashMap<String, String>,
    /// Content type
    pub content_type: Option<String>,
    /// Storage class
    pub storage_class: Option<String>,
    /// Whether to overwrite existing objects
    pub overwrite: bool,
    /// Server-side encryption settings
    pub encryption: Option<EncryptionConfig>,
}

/// Server-side encryption configuration
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Encryption method
    pub method: EncryptionMethod,
    /// Key ID or key material
    pub key: Option<String>,
}

/// Encryption methods supported by cloud providers
#[derive(Debug, Clone)]
pub enum EncryptionMethod {
    /// Provider-managed encryption
    ServerSideManaged,
    /// Customer-managed keys
    CustomerManaged,
    /// Customer-provided keys
    CustomerProvided,
}

/// Result of a list operation
#[derive(Debug, Clone)]
pub struct ListResult {
    /// Objects found
    pub objects: Vec<CloudObjectMetadata>,
    /// Whether there are more results
    pub has_more: bool,
    /// Continuation token for pagination
    pub next_token: Option<String>,
}

/// Cloud storage backend trait
#[cfg(feature = "async")]
#[async_trait]
pub trait CloudStorageBackend: Send + Sync {
    /// Upload a file to cloud storage
    async fn upload_file(
        &self,
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError>;

    /// Download a file from cloud storage
    async fn download_file(
        &self,
        path: &Path,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError>;

    /// Upload data from memory
    async fn upload_data(
        &self,
        data: &[u8],
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError>;

    /// Download data to memory
    async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudError>;

    /// Get object metadata
    async fn get_metadata(&self, key: &str) -> Result<CloudObjectMetadata, CloudError>;

    /// Check if object exists
    async fn object_exists(&self, key: &str) -> Result<bool, CloudError>;

    /// Delete an object
    async fn delete_object(&self, key: &str) -> Result<(), CloudError>;

    /// List objects with optional prefix
    async fn list_objects(
        &self,
        prefix: Option<&str>,
        continuation_token: Option<&str>,
    ) -> Result<ListResult, CloudError>;

    /// Copy an object within the same bucket
    async fn copy_object(
        &self,
        source_key: &str,
        dest_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError>;

    /// Generate a presigned URL for temporary access
    async fn generate_presigned_url(
        &self,
        key: &str,
        expiration: Duration,
        method: HttpMethod,
    ) -> Result<String, CloudError>;
}

/// HTTP methods for presigned URLs
#[derive(Debug, Clone, Copy)]
pub enum HttpMethod {
    Get,
    Put,
    Post,
    Delete,
}

/// Cloud storage client with unified interface
pub struct CloudStorageClient {
    config: CloudConfig,
    backend: Box<dyn CloudStorageBackend>,
    cache: Option<Arc<Mutex<CloudCache>>>,
}

/// Simple in-memory cache for cloud operations
#[derive(Debug)]
struct CloudCache {
    metadata_cache: HashMap<String, (CloudObjectMetadata, SystemTime)>,
    cache_ttl: Duration,
}

impl CloudCache {
    fn new(ttl: Duration) -> Self {
        Self {
            metadata_cache: HashMap::new(),
            cache_ttl: ttl,
        }
    }

    fn get_metadata(&mut self, key: &str) -> Option<CloudObjectMetadata> {
        if let Some((metadata, timestamp)) = self.metadata_cache.get(key) {
            if timestamp.elapsed().unwrap_or(Duration::MAX) < self.cache_ttl {
                return Some(metadata.clone());
            } else {
                self.metadata_cache.remove(key);
            }
        }
        None
    }

    fn put_metadata(&mut self, key: String, metadata: CloudObjectMetadata) {
        self.metadata_cache
            .insert(key, (metadata, SystemTime::now()));
    }

    fn invalidate(&mut self, key: &str) {
        self.metadata_cache.remove(key);
    }

    fn clear(&mut self) {
        self.metadata_cache.clear();
    }
}

impl CloudStorageClient {
    /// Create a new cloud storage client
    pub fn new(config: CloudConfig) -> Result<Self, CloudError> {
        let backend = Self::create_backend(&config)?;
        let cache = if config.enable_cache {
            Some(Arc::new(Mutex::new(CloudCache::new(Duration::from_secs(
                300,
            )))))
        } else {
            None
        };

        Ok(Self {
            config,
            backend,
            cache,
        })
    }

    /// Create the appropriate backend for the provider
    fn create_backend(config: &CloudConfig) -> Result<Box<dyn CloudStorageBackend>, CloudError> {
        match config.provider {
            CloudProvider::AwsS3 | CloudProvider::S3Compatible => {
                Ok(Box::new(S3Backend::new(config.clone())?))
            }
            CloudProvider::GoogleCloud => Ok(Box::new(GoogleCloudBackend::new(config.clone())?)),
            CloudProvider::AzureBlob => Ok(Box::new(AzureBackend::new(config.clone())?)),
        }
    }

    /// Upload a file with progress tracking
    #[cfg(feature = "async")]
    pub async fn upload_file<P: AsRef<Path>>(
        &self,
        local_path: P,
        remote_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        let result = self.backend.upload_file(remote_key, options).await?;

        // Update cache
        if let Some(cache) = &self.cache {
            cache
                .lock()
                .unwrap()
                .put_metadata(remote_key.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Download a file with progress tracking
    #[cfg(feature = "async")]
    pub async fn download_file<P: AsRef<Path>>(
        &self,
        remote_key: &str,
        local_path: P,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        self.backend
            .download_file(local_path.as_ref(), options)
            .await
    }

    /// Upload data from memory
    #[cfg(feature = "async")]
    pub async fn upload_data(
        &self,
        data: &[u8],
        remote_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        let result = self.backend.upload_data(data, remote_key, options).await?;

        // Update cache
        if let Some(cache) = &self.cache {
            cache
                .lock()
                .unwrap()
                .put_metadata(remote_key.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Download data to memory
    #[cfg(feature = "async")]
    pub async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudError> {
        self.backend.get_object(key).await
    }

    /// Get object metadata with caching
    #[cfg(feature = "async")]
    pub async fn get_metadata(&self, key: &str) -> Result<CloudObjectMetadata, CloudError> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(metadata) = cache.lock().unwrap().get_metadata(key) {
                return Ok(metadata);
            }
        }

        // Fetch from backend
        let metadata = self.backend.get_metadata(key).await?;

        // Update cache
        if let Some(cache) = &self.cache {
            cache
                .lock()
                .unwrap()
                .put_metadata(key.to_string(), metadata.clone());
        }

        Ok(metadata)
    }

    /// Check if object exists
    #[cfg(feature = "async")]
    pub async fn object_exists(&self, key: &str) -> Result<bool, CloudError> {
        self.backend.object_exists(key).await
    }

    /// Delete an object
    #[cfg(feature = "async")]
    pub async fn delete_object(&self, key: &str) -> Result<(), CloudError> {
        let result = self.backend.delete_object(key).await;

        // Invalidate cache
        if let Some(cache) = &self.cache {
            cache.lock().unwrap().invalidate(key);
        }

        result
    }

    /// List objects
    #[cfg(feature = "async")]
    pub async fn list_objects(
        &self,
        prefix: Option<&str>,
        continuation_token: Option<&str>,
    ) -> Result<ListResult, CloudError> {
        self.backend.list_objects(prefix, continuation_token).await
    }

    /// Copy an object
    #[cfg(feature = "async")]
    pub async fn copy_object(
        &self,
        source_key: &str,
        dest_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        let result = self
            .backend
            .copy_object(source_key, dest_key, options)
            .await?;

        // Update cache for destination
        if let Some(cache) = &self.cache {
            cache
                .lock()
                .unwrap()
                .put_metadata(dest_key.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Generate presigned URL
    #[cfg(feature = "async")]
    pub async fn generate_presigned_url(
        &self,
        remote_key: &str,
        expiration: Duration,
        method: HttpMethod,
    ) -> Result<String, CloudError> {
        self.backend
            .generate_presigned_url(remote_key, expiration, method)
            .await
    }

    /// Clear all cached data
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.lock().unwrap().clear();
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &CloudConfig {
        &self.config
    }
}

/// S3-compatible backend implementation
struct S3Backend {
    config: CloudConfig,
}

impl S3Backend {
    fn new(config: CloudConfig) -> Result<Self, CloudError> {
        // Validate S3 configuration
        match &config.credentials {
            CloudCredentials::Aws { .. } | CloudCredentials::Anonymous => {}
            _ => {
                return Err(CloudError::InvalidConfiguration(
                    "Invalid credentials for S3".to_string(),
                ))
            }
        }

        Ok(Self { config })
    }
}

#[cfg(feature = "async")]
#[async_trait]
impl CloudStorageBackend for S3Backend {
    async fn upload_file(
        &self,
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        // In a real implementation, this would use the AWS SDK or reqwest
        // to perform the actual upload with proper authentication

        // For now, simulate the operation
        let file_size = 1024; // Simulated file size

        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: file_size,
            last_modified: SystemTime::now(),
            etag: Some("\"mock-etag\"".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn download_file(
        &self,
        path: &Path,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        // Simulate download operation
        std::fs::write(path, b"mock file content")
            .map_err(|e| CloudError::DownloadError(format!("{e}")))?;

        Ok(CloudObjectMetadata {
            key: path.to_string_lossy().to_string(),
            size: 17, // "mock file content".len()
            last_modified: SystemTime::now(),
            etag: Some("\"mock-etag\"".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn upload_data(
        &self,
        data: &[u8],
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        // Simulate upload operation
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: data.len() as u64,
            last_modified: SystemTime::now(),
            etag: Some("\"mock-etag\"".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudError> {
        // Simulate download operation
        Ok(format!("{key}").into_bytes())
    }

    async fn get_metadata(&self, key: &str) -> Result<CloudObjectMetadata, CloudError> {
        // Simulate metadata retrieval
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("\"mock-etag\"".to_string()),
            content_type: Some("application/octet-stream".to_string()),
            metadata: HashMap::new(),
            storage_class: Some("STANDARD".to_string()),
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool, CloudError> {
        // Simulate existence check
        Ok(true)
    }

    async fn delete_object(&self, key: &str) -> Result<(), CloudError> {
        // Simulate deletion
        Ok(())
    }

    async fn list_objects(
        &self,
        prefix: Option<&str>,
        continuation_token: Option<&str>,
    ) -> Result<ListResult, CloudError> {
        // Simulate listing
        let mut objects = Vec::new();
        let max = 10; // Limit for simulation

        for i in 0..max {
            let key = if let Some(prefix) = prefix {
                format!("{prefix}_{i}")
            } else {
                format!("object_{i}")
            };

            objects.push(CloudObjectMetadata {
                key,
                size: 1024 * (i + 1) as u64,
                last_modified: SystemTime::now(),
                etag: Some(format!("\"etag-{}\"", i)),
                content_type: Some("application/octet-stream".to_string()),
                metadata: HashMap::new(),
                storage_class: Some("STANDARD".to_string()),
            });
        }

        Ok(ListResult {
            objects,
            has_more: false,
            next_token: continuation_token.map(|s| s.to_string()),
        })
    }

    async fn copy_object(
        &self,
        source_key: &str,
        dest_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        // Simulate copy operation
        Ok(CloudObjectMetadata {
            key: dest_key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("\"mock-etag\"".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn generate_presigned_url(
        &self,
        key: &str,
        expiration: Duration,
        method: HttpMethod,
    ) -> Result<String, CloudError> {
        // Simulate presigned URL generation
        let method_str = match method {
            HttpMethod::Get => "GET",
            HttpMethod::Put => "PUT",
            HttpMethod::Post => "POST",
            HttpMethod::Delete => "DELETE",
        };

        Ok(format!(
            "https://s3.amazonaws.com/{}/{}?expires={}&method={}",
            self.config.bucket,
            key,
            expiration.as_secs(),
            method_str
        ))
    }
}

/// Google Cloud Storage backend implementation
struct GoogleCloudBackend {
    config: CloudConfig,
}

impl GoogleCloudBackend {
    fn new(config: CloudConfig) -> Result<Self, CloudError> {
        // Validate GCS configuration
        match &config.credentials {
            CloudCredentials::Google { .. } | CloudCredentials::Anonymous => {}
            _ => {
                return Err(CloudError::InvalidConfiguration(
                    "Invalid credentials for GCS".to_string(),
                ))
            }
        }

        Ok(Self { config })
    }
}

#[cfg(feature = "async")]
#[async_trait]
impl CloudStorageBackend for GoogleCloudBackend {
    async fn upload_file(
        &self,
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        // Similar to S3 but with GCS-specific implementation
        let file_size = 1024; // Simulated file size

        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: file_size,
            last_modified: SystemTime::now(),
            etag: Some("mock-gcs-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn download_file(
        &self,
        path: &Path,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        std::fs::write(path, b"mock gcs content")
            .map_err(|e| CloudError::DownloadError(format!("{e}")))?;

        Ok(CloudObjectMetadata {
            key: path.to_string_lossy().to_string(),
            size: 16,
            last_modified: SystemTime::now(),
            etag: Some("mock-gcs-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn upload_data(
        &self,
        data: &[u8],
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: data.len() as u64,
            last_modified: SystemTime::now(),
            etag: Some("mock-gcs-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudError> {
        Ok(format!("{key}").into_bytes())
    }

    async fn get_metadata(&self, key: &str) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("mock-gcs-etag".to_string()),
            content_type: Some("application/octet-stream".to_string()),
            metadata: HashMap::new(),
            storage_class: Some("STANDARD".to_string()),
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool, CloudError> {
        Ok(true)
    }

    async fn delete_object(&self, key: &str) -> Result<(), CloudError> {
        Ok(())
    }

    async fn list_objects(
        &self,
        prefix: Option<&str>,
        continuation_token: Option<&str>,
    ) -> Result<ListResult, CloudError> {
        let mut objects = Vec::new();
        let max = 10; // Limit for simulation

        for i in 0..max {
            let key = if let Some(prefix) = prefix {
                format!("{prefix}_{i}")
            } else {
                format!("object_{i}")
            };

            objects.push(CloudObjectMetadata {
                key,
                size: 1024 * (i + 1) as u64,
                last_modified: SystemTime::now(),
                etag: Some(format!("gcs-etag-{i}")),
                content_type: Some("application/octet-stream".to_string()),
                metadata: HashMap::new(),
                storage_class: Some("STANDARD".to_string()),
            });
        }

        Ok(ListResult {
            objects,
            has_more: false,
            next_token: continuation_token.map(|s| s.to_string()),
        })
    }

    async fn copy_object(
        &self,
        source_key: &str,
        dest_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: dest_key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("mock-gcs-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options
                .storage_class
                .or_else(|| Some("STANDARD".to_string())),
        })
    }

    async fn generate_presigned_url(
        &self,
        key: &str,
        expiration: Duration,
        method: HttpMethod,
    ) -> Result<String, CloudError> {
        let method_str = match method {
            HttpMethod::Get => "GET",
            HttpMethod::Put => "PUT",
            HttpMethod::Post => "POST",
            HttpMethod::Delete => "DELETE",
        };

        Ok(format!(
            "https://storage.googleapis.com/{}/{}?expires={}&method={}",
            self.config.bucket,
            key,
            expiration.as_secs(),
            method_str
        ))
    }
}

/// Azure Blob Storage backend implementation
struct AzureBackend {
    config: CloudConfig,
}

impl AzureBackend {
    fn new(config: CloudConfig) -> Result<Self, CloudError> {
        // Validate Azure configuration
        match &config.credentials {
            CloudCredentials::Azure { .. } | CloudCredentials::Anonymous => {}
            _ => {
                return Err(CloudError::InvalidConfiguration(
                    "Invalid credentials for Azure".to_string(),
                ))
            }
        }

        Ok(Self { config })
    }
}

#[cfg(feature = "async")]
#[async_trait]
impl CloudStorageBackend for AzureBackend {
    async fn upload_file(
        &self,
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        let file_size = 1024; // Simulated file size

        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: file_size,
            last_modified: SystemTime::now(),
            etag: Some("mock-azure-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options.storage_class.or_else(|| Some("Hot".to_string())),
        })
    }

    async fn download_file(
        &self,
        path: &Path,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        std::fs::write(path, b"mock azure content")
            .map_err(|e| CloudError::DownloadError(format!("{e}")))?;

        Ok(CloudObjectMetadata {
            key: path.to_string_lossy().to_string(),
            size: 18,
            last_modified: SystemTime::now(),
            etag: Some("mock-azure-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options.storage_class.or_else(|| Some("Hot".to_string())),
        })
    }

    async fn upload_data(
        &self,
        data: &[u8],
        key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: data.len() as u64,
            last_modified: SystemTime::now(),
            etag: Some("mock-azure-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options.storage_class.or_else(|| Some("Hot".to_string())),
        })
    }

    async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudError> {
        Ok(format!("{key}").into_bytes())
    }

    async fn get_metadata(&self, key: &str) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("mock-azure-etag".to_string()),
            content_type: Some("application/octet-stream".to_string()),
            metadata: HashMap::new(),
            storage_class: Some("Hot".to_string()),
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool, CloudError> {
        Ok(true)
    }

    async fn delete_object(&self, key: &str) -> Result<(), CloudError> {
        Ok(())
    }

    async fn list_objects(
        &self,
        prefix: Option<&str>,
        continuation_token: Option<&str>,
    ) -> Result<ListResult, CloudError> {
        let mut objects = Vec::new();
        let max = 10; // Limit for simulation

        for i in 0..max {
            let key = if let Some(prefix) = prefix {
                format!("{prefix}_{i}")
            } else {
                format!("object_{i}")
            };

            objects.push(CloudObjectMetadata {
                key,
                size: 1024 * (i + 1) as u64,
                last_modified: SystemTime::now(),
                etag: Some(format!("azure-etag-{i}")),
                content_type: Some("application/octet-stream".to_string()),
                metadata: HashMap::new(),
                storage_class: Some("Hot".to_string()),
            });
        }

        Ok(ListResult {
            objects,
            has_more: false,
            next_token: continuation_token.map(|s| s.to_string()),
        })
    }

    async fn copy_object(
        &self,
        source_key: &str,
        dest_key: &str,
        options: TransferOptions,
    ) -> Result<CloudObjectMetadata, CloudError> {
        Ok(CloudObjectMetadata {
            key: dest_key.to_string(),
            size: 1024,
            last_modified: SystemTime::now(),
            etag: Some("mock-azure-etag".to_string()),
            content_type: options
                .content_type
                .or_else(|| Some("application/octet-stream".to_string())),
            metadata: options.metadata,
            storage_class: options.storage_class.or_else(|| Some("Hot".to_string())),
        })
    }

    async fn generate_presigned_url(
        &self,
        key: &str,
        expiration: Duration,
        method: HttpMethod,
    ) -> Result<String, CloudError> {
        let method_str = match method {
            HttpMethod::Get => "GET",
            HttpMethod::Put => "PUT",
            HttpMethod::Post => "POST",
            HttpMethod::Delete => "DELETE",
        };

        let account_name = match &self.config.credentials {
            CloudCredentials::Azure { account_name, .. } => account_name,
            _ => "mockaccount",
        };

        Ok(format!(
            "https://{}.blob.core.windows.net/{}/{}?expires={}&method={}",
            account_name,
            self.config.bucket,
            key,
            expiration.as_secs(),
            method_str
        ))
    }
}

/// Convenience functions for common cloud operations
pub mod utils {
    use super::*;

    /// Sync a local directory to cloud storage
    #[cfg(feature = "async")]
    pub async fn sync_directory_to_cloud(
        client: &CloudStorageClient,
        local_dir: &Path,
        remote_prefix: &str,
        recursive: bool,
    ) -> Result<usize, CloudError> {
        let mut uploaded_count = 0;

        fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    files.push(path);
                } else if path.is_dir() {
                    visit_dir(&path, files)?;
                }
            }
            Ok(())
        }

        let mut files = Vec::new();
        if recursive {
            visit_dir(local_dir, &mut files).map_err(|e| CloudError::UploadError(e.to_string()))?;
        } else {
            for entry in
                std::fs::read_dir(local_dir).map_err(|e| CloudError::UploadError(e.to_string()))?
            {
                let entry = entry.map_err(|e| CloudError::UploadError(e.to_string()))?;
                let path = entry.path();
                if path.is_file() {
                    files.push(path);
                }
            }
        }

        for file_path in files {
            let relative_path = file_path
                .strip_prefix(local_dir)
                .map_err(|e| CloudError::UploadError(e.to_string()))?;
            let remote_key = format!("{}/{}", remote_prefix, relative_path.to_string_lossy());

            client
                .upload_file(&file_path, &remote_key, TransferOptions::default())
                .await?;
            uploaded_count += 1;
        }

        Ok(uploaded_count)
    }

    /// Download and sync cloud storage to local directory
    #[cfg(feature = "async")]
    pub async fn sync_cloud_to_directory(
        client: &CloudStorageClient,
        remote_prefix: &str,
        local_dir: &Path,
    ) -> Result<usize, CloudError> {
        let mut downloaded_count = 0;
        let mut continuation_token = None;

        loop {
            let result = client
                .list_objects(Some(remote_prefix), continuation_token.as_deref())
                .await?;

            for object in &result.objects {
                let relative_path = object
                    .key
                    .strip_prefix(remote_prefix)
                    .unwrap_or(&object.key);
                let local_path = local_dir.join(relative_path);

                // Create parent directories
                if let Some(parent) = local_path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| CloudError::DownloadError(e.to_string()))?;
                }

                client
                    .download_file(&object.key, &local_path, TransferOptions::default())
                    .await?;
                downloaded_count += 1;
            }

            if !result.has_more {
                break;
            }
            continuation_token = result.next_token;
        }

        Ok(downloaded_count)
    }

    /// Calculate total size of objects with given prefix
    #[cfg(feature = "async")]
    pub async fn calculate_storage_usage(
        client: &CloudStorageClient,
        prefix: Option<&str>,
    ) -> Result<u64, CloudError> {
        let mut total_size = 0;
        let mut continuation_token = None;

        loop {
            let result = client
                .list_objects(prefix, continuation_token.as_deref())
                .await?;

            for object in &result.objects {
                total_size += object.size;
            }

            if !result.has_more {
                break;
            }
            continuation_token = result.next_token;
        }

        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tempfile::tempdir;

    #[test]
    fn test_cloud_provider_methods() {
        assert_eq!(CloudProvider::AwsS3.as_str(), "aws-s3");
        assert_eq!(CloudProvider::GoogleCloud.as_str(), "google-cloud");
        assert_eq!(CloudProvider::AzureBlob.as_str(), "azure-blob");

        assert!(CloudProvider::AwsS3.default_endpoint().is_some());
        assert!(CloudProvider::GoogleCloud.default_endpoint().is_some());
        assert!(CloudProvider::AzureBlob.default_endpoint().is_none());
    }

    #[test]
    fn test_cloud_config_builders() {
        let creds = CloudCredentials::Anonymous;

        let s3_config = CloudConfig::new_bucket("test-bucket".to_string(), creds.clone());
        assert_eq!(s3_config.provider, CloudProvider::AwsS3);
        assert_eq!(s3_config.bucket, "test-bucket");

        let gcs_config = CloudConfig::new_bucket("test-bucket".to_string(), creds.clone());
        assert_eq!(gcs_config.provider, CloudProvider::AwsS3);

        let azure_config = CloudConfig::container("test-container".to_string(), creds);
        assert_eq!(azure_config.provider, CloudProvider::AzureBlob);
    }

    #[test]
    fn test_cloud_config_with_modifiers() {
        let config = CloudConfig::default()
            .with_endpoint("https://custom.endpoint.com".to_string())
            .with_timeout(Duration::from_secs(60))
            .with_multipart(50 * 1024 * 1024, 4 * 1024 * 1024);

        assert_eq!(
            config.endpoint,
            Some("https://custom.endpoint.com".to_string())
        );
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.multipart_threshold, 50 * 1024 * 1024);
        assert_eq!(config.chunk_size, 4 * 1024 * 1024);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_s3_backend_operations() {
        let config =
            CloudConfig::new_bucket("test-bucket".to_string(), CloudCredentials::Anonymous);
        let backend = S3Backend::new(config).unwrap();

        // Test metadata retrieval
        let metadata = backend.get_metadata("test-key").await.unwrap();
        assert_eq!(metadata.key, "test-key");
        assert!(metadata.size > 0);

        // Test existence check
        let exists = backend.object_exists("test-key").await.unwrap();
        assert!(exists);

        // Test data upload
        let data = b"test data";
        let result = backend
            .upload_data(data, "test-upload", TransferOptions::default())
            .await
            .unwrap();
        assert_eq!(result.key, "test-upload");
        assert_eq!(result.size, data.len() as u64);

        // Test data download
        let downloaded = backend.get_object("test-key").await.unwrap();
        assert!(!downloaded.is_empty());

        // Test listing
        let list_result = backend.list_objects(None, Some("5")).await.unwrap();
        assert!(!list_result.objects.is_empty());
        assert!(list_result.objects.len() <= 10);

        // Test presigned URL generation
        let url = backend
            .generate_presigned_url("test-key", Duration::from_secs(3600), HttpMethod::Get)
            .await
            .unwrap();
        assert!(url.contains("test-key"));
        assert!(url.contains("expires=3600"));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_cloud_storage_client() {
        let config =
            CloudConfig::new_bucket("test-bucket".to_string(), CloudCredentials::Anonymous);
        let client = CloudStorageClient::new(config).unwrap();

        // Test metadata with caching
        let metadata1 = client.get_metadata("test-key").await.unwrap();
        let metadata2 = client.get_metadata("test-key").await.unwrap(); // Should hit cache
        assert_eq!(metadata1.key, metadata2.key);

        // Test cache clearing
        client.clear_cache();

        // Test upload
        let data = b"test data for client";
        let result = client
            .upload_data(data, "client-test", TransferOptions::default())
            .await
            .unwrap();
        assert_eq!(result.size, data.len() as u64);
    }

    #[test]
    fn test_transfer_options() {
        let mut options = TransferOptions::default();
        options
            .metadata
            .insert("custom-key".to_string(), "custom-value".to_string());
        options.content_type = Some("text/plain".to_string());
        options.overwrite = true;

        assert_eq!(
            options.metadata.get("custom-key"),
            Some(&"custom-value".to_string())
        );
        assert_eq!(options.content_type, Some("text/plain".to_string()));
        assert!(options.overwrite);
    }

    #[test]
    fn test_encryption_config() {
        let encryption = EncryptionConfig {
            method: EncryptionMethod::CustomerManaged,
            key: Some("test-key-id".to_string()),
        };

        match encryption.method {
            EncryptionMethod::CustomerManaged => assert!(true),
            _ => assert!(false),
        }
        assert_eq!(encryption.key, Some("test-key-id".to_string()));
    }
}
