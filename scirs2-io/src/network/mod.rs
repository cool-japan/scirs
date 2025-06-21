//! Network I/O and cloud storage integration
//!
//! This module provides functionality for reading and writing files over network protocols
//! and integrating with cloud storage services. It supports efficient streaming, caching,
//! and secure authentication for various cloud providers.
//!
//! ## Features
//!
//! - **HTTP/HTTPS I/O**: Download and upload files via HTTP protocols
//! - **Cloud Storage**: Integration with AWS S3, Google Cloud Storage, Azure Blob Storage
//! - **Streaming**: Efficient handling of large files with minimal memory usage
//! - **Caching**: Local caching of remote files for offline access
//! - **Authentication**: Secure handling of credentials and API keys
//! - **Retry Logic**: Robust error handling with automatic retry mechanisms
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::network::NetworkClient;
//!
//! // Create a network client
//! let client = NetworkClient::new();
//! println!("Network client created for file operations");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{IoError, Result};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Cloud storage integration
pub mod cloud;
/// HTTP client functionality  
pub mod http;
/// Streaming I/O operations
pub mod streaming;

/// Network client configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Connection timeout in seconds
    pub connect_timeout: Duration,
    /// Read timeout in seconds
    pub read_timeout: Duration,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// User agent string for HTTP requests
    pub user_agent: String,
    /// Custom HTTP headers
    pub headers: HashMap<String, String>,
    /// Enable response compression
    pub compression: bool,
    /// Local cache directory for downloaded files
    pub cache_dir: Option<String>,
    /// Maximum cache size in MB
    pub max_cache_size: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        let mut headers = HashMap::new();
        headers.insert("Accept".to_string(), "*/*".to_string());

        Self {
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(300),
            max_retries: 3,
            user_agent: "scirs2-io/0.1.0".to_string(),
            headers,
            compression: true,
            cache_dir: None,
            max_cache_size: 1024, // 1GB default
        }
    }
}

/// Network client for remote I/O operations
#[derive(Debug)]
pub struct NetworkClient {
    config: NetworkConfig,
    #[cfg(feature = "reqwest")]
    http_client: Option<reqwest::Client>,
    cloud_provider: Option<cloud::CloudProvider>,
}

impl Default for NetworkClient {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkClient {
    /// Create a new network client with default configuration
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            #[cfg(feature = "reqwest")]
            http_client: None,
            cloud_provider: None,
        }
    }

    /// Create a new network client with custom configuration
    pub fn with_config(config: NetworkConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "reqwest")]
            http_client: None,
            cloud_provider: None,
        }
    }

    /// Set cloud provider for cloud storage operations
    pub fn with_cloud_provider(mut self, provider: cloud::CloudProvider) -> Self {
        self.cloud_provider = Some(provider);
        self
    }

    /// Set cache directory for downloaded files
    pub fn with_cache_dir<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.config.cache_dir = Some(cache_dir.as_ref().to_string_lossy().to_string());
        self
    }

    /// Download a file from URL to local path
    #[cfg(feature = "reqwest")]
    pub async fn download<P: AsRef<Path>>(&self, url: &str, local_path: P) -> Result<()> {
        self.http_client.download(url, local_path).await
    }

    /// Upload a file from local path to URL
    #[cfg(feature = "reqwest")]
    pub async fn upload<P: AsRef<Path>>(&self, local_path: P, url: &str) -> Result<()> {
        self.http_client.upload(local_path, url).await
    }

    /// Download a file to cloud storage
    pub async fn upload_to_cloud<P: AsRef<Path>>(
        &self,
        local_path: P,
        remote_path: &str,
    ) -> Result<()> {
        if let Some(ref provider) = self.cloud_provider {
            provider.upload_file(local_path, remote_path).await
        } else {
            Err(IoError::ConfigError(
                "No cloud provider configured".to_string(),
            ))
        }
    }

    /// Download a file from cloud storage
    pub async fn download_from_cloud<P: AsRef<Path>>(
        &self,
        remote_path: &str,
        local_path: P,
    ) -> Result<()> {
        if let Some(ref provider) = self.cloud_provider {
            provider.download_file(remote_path, local_path).await
        } else {
            Err(IoError::ConfigError(
                "No cloud provider configured".to_string(),
            ))
        }
    }

    /// List files in cloud storage path
    pub async fn list_cloud_files(&self, path: &str) -> Result<Vec<String>> {
        if let Some(ref provider) = self.cloud_provider {
            provider.list_files(path).await
        } else {
            Err(IoError::ConfigError(
                "No cloud provider configured".to_string(),
            ))
        }
    }

    /// Check if a file exists in cloud storage
    pub async fn cloud_file_exists(&self, path: &str) -> Result<bool> {
        if let Some(ref provider) = self.cloud_provider {
            provider.file_exists(path).await
        } else {
            Err(IoError::ConfigError(
                "No cloud provider configured".to_string(),
            ))
        }
    }

    /// Get file metadata from cloud storage
    pub async fn get_cloud_file_metadata(&self, path: &str) -> Result<cloud::FileMetadata> {
        if let Some(ref provider) = self.cloud_provider {
            provider.get_metadata(path).await
        } else {
            Err(IoError::ConfigError(
                "No cloud provider configured".to_string(),
            ))
        }
    }

    /// Clear local cache
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(ref cache_dir) = self.config.cache_dir {
            let cache_path = Path::new(cache_dir);
            if cache_path.exists() {
                std::fs::remove_dir_all(cache_path)
                    .map_err(|e| IoError::FileError(format!("Failed to clear cache: {}", e)))?;
                std::fs::create_dir_all(cache_path).map_err(|e| {
                    IoError::FileError(format!("Failed to recreate cache dir: {}", e))
                })?;
            }
        }
        Ok(())
    }

    /// Get cache usage information
    pub fn get_cache_info(&self) -> Result<(u64, u64)> {
        if let Some(ref cache_dir) = self.config.cache_dir {
            let cache_path = Path::new(cache_dir);
            if cache_path.exists() {
                let mut total_size = 0u64;
                let mut file_count = 0u64;

                for entry in (std::fs::read_dir(cache_path)
                    .map_err(|e| IoError::FileError(format!("Failed to read cache dir: {}", e)))?)
                .flatten()
                {
                    if let Ok(metadata) = entry.metadata() {
                        if metadata.is_file() {
                            total_size += metadata.len();
                            file_count += 1;
                        }
                    }
                }
                return Ok((total_size, file_count));
            }
        }
        Ok((0, 0))
    }
}

/// Convenience functions for common network operations

/// Download a file from URL using default client
#[cfg(feature = "reqwest")]
pub async fn download_file<P: AsRef<Path>>(url: &str, local_path: P) -> Result<()> {
    let client = NetworkClient::new();
    client.download(url, local_path).await
}

/// Upload a file to URL using default client
#[cfg(feature = "reqwest")]
pub async fn upload_file<P: AsRef<Path>>(local_path: P, url: &str) -> Result<()> {
    let client = NetworkClient::new();
    client.upload(local_path, url).await
}

/// Download a file with caching support
#[cfg(feature = "reqwest")]
pub async fn download_with_cache<P: AsRef<Path>>(
    url: &str,
    local_path: P,
    cache_dir: Option<&str>,
) -> Result<()> {
    let mut client = NetworkClient::new();
    if let Some(cache) = cache_dir {
        client = client.with_cache_dir(cache);
    }
    client.download(url, local_path).await
}

/// Create a network client with cloud provider
pub fn create_cloud_client(provider: cloud::CloudProvider) -> NetworkClient {
    NetworkClient::new().with_cloud_provider(provider)
}

/// Batch download multiple files
#[cfg(feature = "reqwest")]
pub async fn batch_download(downloads: Vec<(&str, &str)>) -> Result<Vec<Result<()>>> {
    let client = NetworkClient::new();
    let mut results = Vec::new();

    for (url, local_path) in downloads {
        let result = client.download(url, local_path).await;
        results.push(result);
    }

    Ok(results)
}

/// Batch upload multiple files to cloud storage
pub async fn batch_upload_to_cloud(
    client: &NetworkClient,
    uploads: Vec<(&str, &str)>,
) -> Result<Vec<Result<()>>> {
    let mut results = Vec::new();

    for (local_path, remote_path) in uploads {
        let result = client.upload_to_cloud(local_path, remote_path).await;
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(30));
        assert_eq!(config.read_timeout, Duration::from_secs(300));
        assert_eq!(config.max_retries, 3);
        assert!(config.compression);
        assert_eq!(config.max_cache_size, 1024);
    }

    #[test]
    fn test_network_client_creation() {
        let client = NetworkClient::new();
        assert!(client.cloud_provider.is_none());

        let client_with_cache = NetworkClient::new().with_cache_dir("/tmp/cache");
        assert_eq!(
            client_with_cache.config.cache_dir,
            Some("/tmp/cache".to_string())
        );
    }

    #[test]
    fn test_network_config_custom() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token".to_string());

        let config = NetworkConfig {
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(60),
            max_retries: 5,
            user_agent: "custom-agent/1.0".to_string(),
            headers,
            compression: false,
            cache_dir: Some("/custom/cache".to_string()),
            max_cache_size: 512,
        };

        let client = NetworkClient::with_config(config.clone());
        assert_eq!(client.config.connect_timeout, Duration::from_secs(10));
        assert_eq!(client.config.max_retries, 5);
        assert!(!client.config.compression);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_cache_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_path = temp_dir.path().to_str().unwrap();

        let client = NetworkClient::new().with_cache_dir(cache_path);

        // Test cache info on empty cache
        let (size, count) = client.get_cache_info().unwrap();
        assert_eq!(size, 0);
        assert_eq!(count, 0);

        // Test cache clearing
        client.clear_cache().unwrap();
    }
}
