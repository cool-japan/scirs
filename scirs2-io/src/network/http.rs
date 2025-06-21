//! HTTP client functionality for network I/O operations
//!
//! This module provides HTTP/HTTPS capabilities for downloading and uploading files,
//! with support for streaming, authentication, retries, and caching.

use crate::error::{IoError, Result};
use crate::network::NetworkConfig;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
#[cfg(feature = "reqwest")]
use std::time::Instant;

/// HTTP request method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HttpMethod {
    /// GET method
    GET,
    /// POST method
    POST,
    /// PUT method
    PUT,
    /// DELETE method
    DELETE,
    /// HEAD method
    HEAD,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::GET => write!(f, "GET"),
            HttpMethod::POST => write!(f, "POST"),
            HttpMethod::PUT => write!(f, "PUT"),
            HttpMethod::DELETE => write!(f, "DELETE"),
            HttpMethod::HEAD => write!(f, "HEAD"),
        }
    }
}

/// HTTP response information
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Content length if available
    pub content_length: Option<u64>,
    /// Content type if available
    pub content_type: Option<String>,
    /// Response body as bytes
    pub body: Vec<u8>,
}

/// Progress callback for long-running operations
pub type ProgressCallback = Box<dyn Fn(u64, Option<u64>) + Send + Sync>;

/// HTTP client for network operations
#[derive(Debug)]
pub struct HttpClient {
    #[allow(dead_code)]
    config: NetworkConfig,
    #[cfg(feature = "reqwest")]
    client: Option<reqwest::Client>,
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "reqwest")]
            client: None,
        }
    }

    /// Initialize the HTTP client (creates underlying reqwest client)
    #[cfg(feature = "reqwest")]
    pub fn init(&mut self) -> Result<()> {
        let mut client_builder = reqwest::Client::builder()
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.read_timeout)
            .user_agent(&self.config.user_agent);

        // Add default headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in &self.config.headers {
            if let (Ok(header_name), Ok(header_value)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }
        client_builder = client_builder.default_headers(headers);

        self.client =
            Some(client_builder.build().map_err(|e| {
                IoError::NetworkError(format!("Failed to create HTTP client: {}", e))
            })?);

        Ok(())
    }

    /// Download a file from URL to local path
    #[cfg(feature = "reqwest")]
    pub async fn download<P: AsRef<Path>>(&self, url: &str, local_path: P) -> Result<()> {
        let client = self.get_client()?;
        let local_path = local_path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| IoError::FileError(format!("Failed to create directory: {}", e)))?;
        }

        let mut retries = 0;
        loop {
            let start_time = Instant::now();

            match self.download_with_retry(client, url, local_path).await {
                Ok(_) => {
                    let duration = start_time.elapsed();
                    log::info!("Downloaded {} in {:.2}s", url, duration.as_secs_f64());
                    return Ok(());
                }
                Err(e) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err(e);
                    }

                    let delay = Duration::from_millis(100 * 2_u64.pow(retries - 1));
                    log::warn!(
                        "Download failed, retrying in {}ms: {}",
                        delay.as_millis(),
                        e
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    #[cfg(feature = "reqwest")]
    async fn download_with_retry(
        &self,
        client: &reqwest::Client,
        url: &str,
        local_path: &Path,
    ) -> Result<()> {
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| IoError::NetworkError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(IoError::NetworkError(format!(
                "HTTP error {}: {}",
                response.status().as_u16(),
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
            )));
        }

        let content_length = response.content_length();
        let mut downloaded = 0u64;

        let mut file = std::fs::File::create(local_path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;

        use std::io::Write;

        let bytes = response
            .bytes()
            .await
            .map_err(|e| IoError::NetworkError(format!("Failed to read response body: {}", e)))?;

        file.write_all(&bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write file: {}", e)))?;

        downloaded = bytes.len() as u64;

        // Progress reporting could be added here
        if let Some(total) = content_length {
            let progress = (downloaded as f64 / total as f64 * 100.0) as u8;
            log::debug!("Download progress: {}%", progress);
        }

        Ok(())
    }

    /// Upload a file from local path to URL
    #[cfg(feature = "reqwest")]
    pub async fn upload<P: AsRef<Path>>(&self, local_path: P, url: &str) -> Result<()> {
        let client = self.get_client()?;
        let local_path = local_path.as_ref();

        if !local_path.exists() {
            return Err(IoError::FileError(format!(
                "File does not exist: {}",
                local_path.display()
            )));
        }

        let file_content = std::fs::read(local_path)
            .map_err(|e| IoError::FileError(format!("Failed to read file: {}", e)))?;

        let mut retries = 0;
        loop {
            let start_time = Instant::now();

            match self.upload_with_retry(client, &file_content, url).await {
                Ok(_) => {
                    let duration = start_time.elapsed();
                    log::info!(
                        "Uploaded {} in {:.2}s",
                        local_path.display(),
                        duration.as_secs_f64()
                    );
                    return Ok(());
                }
                Err(e) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err(e);
                    }

                    let delay = Duration::from_millis(100 * 2_u64.pow(retries - 1));
                    log::warn!("Upload failed, retrying in {}ms: {}", delay.as_millis(), e);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    #[cfg(feature = "reqwest")]
    async fn upload_with_retry(
        &self,
        client: &reqwest::Client,
        content: &[u8],
        url: &str,
    ) -> Result<()> {
        let response = client
            .put(url)
            .body(content.to_vec())
            .send()
            .await
            .map_err(|e| IoError::NetworkError(format!("HTTP upload failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(IoError::NetworkError(format!(
                "HTTP upload error {}: {}",
                response.status().as_u16(),
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
            )));
        }

        Ok(())
    }

    /// Make a custom HTTP request
    #[cfg(feature = "reqwest")]
    pub async fn request(
        &self,
        method: HttpMethod,
        url: &str,
        body: Option<&[u8]>,
    ) -> Result<HttpResponse> {
        let client = self.get_client()?;

        let mut request_builder = match method {
            HttpMethod::GET => client.get(url),
            HttpMethod::POST => client.post(url),
            HttpMethod::PUT => client.put(url),
            HttpMethod::DELETE => client.delete(url),
            HttpMethod::HEAD => client.head(url),
        };

        if let Some(body_data) = body {
            request_builder = request_builder.body(body_data.to_vec());
        }

        let response = request_builder
            .send()
            .await
            .map_err(|e| IoError::NetworkError(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let content_length = response.content_length();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let body = response
            .bytes()
            .await
            .map_err(|e| IoError::NetworkError(format!("Failed to read response body: {}", e)))?
            .to_vec();

        Ok(HttpResponse {
            status,
            headers,
            content_length,
            content_type,
            body,
        })
    }

    /// Check if a URL is accessible (HEAD request)
    #[cfg(feature = "reqwest")]
    pub async fn check_url(&self, url: &str) -> Result<bool> {
        match self.request(HttpMethod::HEAD, url, None).await {
            Ok(response) => Ok(response.status >= 200 && response.status < 300),
            Err(_) => Ok(false),
        }
    }

    /// Get file size from URL without downloading (HEAD request)
    #[cfg(feature = "reqwest")]
    pub async fn get_remote_file_size(&self, url: &str) -> Result<Option<u64>> {
        let response = self.request(HttpMethod::HEAD, url, None).await?;
        Ok(response.content_length)
    }

    #[cfg(feature = "reqwest")]
    fn get_client(&self) -> Result<&reqwest::Client> {
        self.client
            .as_ref()
            .ok_or_else(|| IoError::ConfigError("HTTP client not initialized".to_string()))
    }

    // Fallback implementations when reqwest feature is not enabled
    #[cfg(not(feature = "reqwest"))]
    /// Download a file (fallback implementation when reqwest feature is disabled)
    pub async fn download<P: AsRef<Path>>(&self, _url: &str, _local_path: P) -> Result<()> {
        Err(IoError::ConfigError(
            "HTTP support requires 'reqwest' feature".to_string(),
        ))
    }

    #[cfg(not(feature = "reqwest"))]
    /// Upload a file (fallback implementation when reqwest feature is disabled)
    pub async fn upload<P: AsRef<Path>>(&self, _local_path: P, _url: &str) -> Result<()> {
        Err(IoError::ConfigError(
            "HTTP support requires 'reqwest' feature".to_string(),
        ))
    }

    #[cfg(not(feature = "reqwest"))]
    /// Make an HTTP request (fallback implementation when reqwest feature is disabled)
    pub async fn request(
        &self,
        _method: HttpMethod,
        _url: &str,
        _body: Option<&[u8]>,
    ) -> Result<HttpResponse> {
        Err(IoError::ConfigError(
            "HTTP support requires 'reqwest' feature".to_string(),
        ))
    }

    #[cfg(not(feature = "reqwest"))]
    /// Check if URL is reachable (fallback implementation when reqwest feature is disabled)
    pub async fn check_url(&self, _url: &str) -> Result<bool> {
        Err(IoError::ConfigError(
            "HTTP support requires 'reqwest' feature".to_string(),
        ))
    }

    #[cfg(not(feature = "reqwest"))]
    /// Get remote file size (fallback implementation when reqwest feature is disabled)
    pub async fn get_remote_file_size(&self, _url: &str) -> Result<Option<u64>> {
        Err(IoError::ConfigError(
            "HTTP support requires 'reqwest' feature".to_string(),
        ))
    }
}

/// Utility functions

/// Download multiple files concurrently
#[cfg(feature = "reqwest")]
pub async fn download_concurrent(
    downloads: Vec<(String, String)>,
    max_concurrent: usize,
) -> Result<Vec<Result<()>>> {
    use futures_util::stream::{FuturesUnordered, StreamExt};

    let client = HttpClient::new(NetworkConfig::default());

    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let mut futures = FuturesUnordered::new();

    for (url, local_path) in downloads {
        let client_clone = &client;
        let semaphore_clone = semaphore.clone();

        futures.push(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();
            client_clone.download(&url, &local_path).await
        });
    }

    let mut results = Vec::new();
    while let Some(result) = futures.next().await {
        results.push(result);
    }

    Ok(results)
}

/// Calculate download speed
pub fn calculate_speed(bytes: u64, duration: Duration) -> f64 {
    if duration.as_secs_f64() > 0.0 {
        bytes as f64 / duration.as_secs_f64()
    } else {
        0.0
    }
}

/// Format file size in human-readable format
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Format download speed in human-readable format
pub fn format_speed(bytes_per_second: f64) -> String {
    format!("{}/s", format_file_size(bytes_per_second as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::GET.to_string(), "GET");
        assert_eq!(HttpMethod::POST.to_string(), "POST");
        assert_eq!(HttpMethod::PUT.to_string(), "PUT");
        assert_eq!(HttpMethod::DELETE.to_string(), "DELETE");
        assert_eq!(HttpMethod::HEAD.to_string(), "HEAD");
    }

    #[test]
    fn test_http_response_creation() {
        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());

        let response = HttpResponse {
            status: 200,
            headers,
            content_length: Some(1024),
            content_type: Some("application/json".to_string()),
            body: b"test data".to_vec(),
        };

        assert_eq!(response.status, 200);
        assert_eq!(response.content_length, Some(1024));
        assert_eq!(response.content_type, Some("application/json".to_string()));
        assert_eq!(response.body, b"test data");
    }

    #[test]
    fn test_http_client_creation() {
        let config = NetworkConfig::default();
        let _client = HttpClient::new(config);

        // Client should be created successfully
        assert!(true); // Basic creation test
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(512), "512.0 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_calculate_speed() {
        let duration = Duration::from_secs(1);
        assert_eq!(calculate_speed(1024, duration), 1024.0);

        let duration = Duration::from_secs(2);
        assert_eq!(calculate_speed(2048, duration), 1024.0);

        let duration = Duration::from_secs(0);
        assert_eq!(calculate_speed(1024, duration), 0.0);
    }

    #[test]
    fn test_format_speed() {
        assert_eq!(format_speed(1024.0), "1.0 KB/s");
        assert_eq!(format_speed(1024.0 * 1024.0), "1.0 MB/s");
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_http_client_without_reqwest_feature() {
        let config = NetworkConfig::default();
        let client = HttpClient::new(config);

        // These should return feature errors when reqwest is not enabled
        #[cfg(not(feature = "reqwest"))]
        {
            let download_result = client.download("http://example.com", "test.txt").await;
            assert!(download_result.is_err());

            let upload_result = client.upload("test.txt", "http://example.com").await;
            assert!(upload_result.is_err());

            let request_result = client
                .request(HttpMethod::GET, "http://example.com", None)
                .await;
            assert!(request_result.is_err());

            let check_result = client.check_url("http://example.com").await;
            assert!(check_result.is_err());

            let size_result = client.get_remote_file_size("http://example.com").await;
            assert!(size_result.is_err());
        }
    }
}
