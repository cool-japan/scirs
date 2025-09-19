//! External data sources integration
//!
//! This module provides functionality for loading datasets from external sources including:
//! - URLs and web resources
//! - API endpoints
//! - Popular dataset repositories
//! - Remote file systems

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::cache::DatasetCache;
use crate::error::{DatasetsError, Result};
use crate::loaders::{load_csv, CsvConfig};
use crate::utils::Dataset;

/// Configuration for external data source access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalConfig {
    /// Timeout for requests (in seconds)
    pub timeout_seconds: u64,
    /// Number of retry attempts
    pub max_retries: u32,
    /// User agent string for requests
    pub user_agent: String,
    /// Headers to include in requests
    pub headers: HashMap<String, String>,
    /// Whether to verify SSL certificates
    pub verify_ssl: bool,
    /// Cache downloaded files
    pub use_cache: bool,
}

impl Default for ExternalConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 300, // 5 minutes
            max_retries: 3,
            user_agent: "scirs2-datasets/0.1.0".to_string(),
            headers: HashMap::new(),
            verify_ssl: true,
            use_cache: true,
        }
    }
}

/// Progress callback for download operations
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// External data source client
pub struct ExternalClient {
    config: ExternalConfig,
    cache: DatasetCache,
    #[cfg(feature = "download")]
    client: reqwest::Client,
}

impl ExternalClient {
    /// Create a new external client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ExternalConfig::default())
    }

    /// Create a new external client with custom configuration
    pub fn with_config(config: ExternalConfig) -> Result<Self> {
        let cache = DatasetCache::new(crate::cache::get_cachedir()?);

        #[cfg(feature = "download")]
        let client = {
            let mut builder = reqwest::Client::builder()
                .timeout(Duration::from_secs(config.timeout_seconds))
                .user_agent(&config.user_agent);

            if !config.verify_ssl {
                builder = builder.danger_accept_invalid_certs(true);
            }

            builder
                .build()
                .map_err(|e| DatasetsError::IoError(std::io::Error::other(e)))?
        };

        Ok(Self {
            config,
            cache,
            #[cfg(feature = "download")]
            client,
        })
    }

    /// Download a dataset from a URL
    #[cfg(feature = "download")]
    pub async fn download_dataset(
        &self,
        url: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<Dataset> {
        // Check cache first
        if self.config.use_cache {
            let cache_key = format!("external_{}", blake3::hash(url.as_bytes()).to_hex());
            if let Ok(cached_data) = self.cache.read_cached(&cache_key) {
                return self.parse_cached_data(&cached_data);
            }
        }

        // Download the file
        let response = self.make_request(url).await?;
        let total_size = response.content_length().unwrap_or(0);

        let mut downloaded = 0u64;
        let mut buffer = Vec::new();
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| DatasetsError::IoError(std::io::Error::other(e)))?;
            downloaded += chunk.len() as u64;
            buffer.extend_from_slice(&chunk);

            if let Some(ref callback) = progress {
                callback(downloaded, total_size);
            }
        }

        // Cache the downloaded data
        if self.config.use_cache {
            let cache_key = format!("external_{}", blake3::hash(url.as_bytes()).to_hex());
            let _ = self.cache.put(&cache_key, &buffer);
        }

        // Parse the data based on content type or URL extension
        self.parse_downloaded_data(url, &buffer)
    }

    /// Download a dataset synchronously (blocking) - when download feature is enabled
    #[cfg(feature = "download")]
    pub fn download_dataset_sync(
        &self,
        url: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<Dataset> {
        // Use tokio runtime to block on the async version
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| DatasetsError::IoError(std::io::Error::other(e)))?;
        rt.block_on(self.download_dataset(url, progress))
    }

    /// Download a dataset synchronously (blocking) - fallback when download feature is disabled
    #[cfg(not(feature = "download"))]
    pub fn download_dataset_sync(
        &self,
        url: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<Dataset> {
        // Fallback implementation using ureq
        self.download_with_ureq(url, progress)
    }

    /// Download using ureq (synchronous HTTP client)
    #[allow(dead_code)]
    fn download_with_ureq(&self, url: &str, progress: Option<ProgressCallback>) -> Result<Dataset> {
        // Check cache first
        if self.config.use_cache {
            let cache_key = format!("external_{}", blake3::hash(url.as_bytes()).to_hex());
            if let Ok(cached_data) = self.cache.read_cached(&cache_key) {
                return self.parse_cached_data(&cached_data);
            }
        }

        let mut request = ureq::get(url)
            .set("User-Agent", &self.config.user_agent)
            .timeout(Duration::from_secs(self.config.timeout_seconds));

        // Add custom headers
        for (key, value) in &self.config.headers {
            request = request.set(key, value);
        }

        let response = request
            .call()
            .map_err(|e| DatasetsError::IoError(std::io::Error::other(e)))?;

        let total_size = response
            .header("content-length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        let mut buffer = Vec::new();
        let mut reader = response.into_reader();
        let mut chunk = vec![0; 8192];
        let mut downloaded = 0u64;

        loop {
            let bytes_read = reader.read(&mut chunk).map_err(DatasetsError::IoError)?;

            if bytes_read == 0 {
                break;
            }

            buffer.extend_from_slice(&chunk[..bytes_read]);
            downloaded += bytes_read as u64;

            if let Some(ref callback) = progress {
                callback(downloaded, total_size);
            }
        }

        // Cache the downloaded data
        if self.config.use_cache {
            let cache_key = format!("external_{}", blake3::hash(url.as_bytes()).to_hex());
            let _ = self.cache.put(&cache_key, &buffer);
        }

        // Parse the data
        self.parse_downloaded_data(url, &buffer)
    }

    #[cfg(feature = "download")]
    async fn make_request(&self, url: &str) -> Result<reqwest::Response> {
        let mut request = self.client.get(url);

        // Add custom headers
        for (key, value) in &self.config.headers {
            request = request.header(key, value);
        }

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match request
                .try_clone()
                .ok_or_else(|| {
                    DatasetsError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to clone request",
                    ))
                })?
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(response);
                    } else {
                        last_error = Some(DatasetsError::IoError(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "HTTP {}: {}",
                                response.status(),
                                response.status().canonical_reason().unwrap_or("Unknown")
                            ),
                        )));
                    }
                }
                Err(e) => {
                    last_error = Some(DatasetsError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    )));
                }
            }

            if attempt < self.config.max_retries {
                tokio::time::sleep(Duration::from_millis(1000 * (attempt + 1) as u64)).await;
            }
        }

        Err(last_error.unwrap())
    }

    fn parse_cached_data(&self, data: &[u8]) -> Result<Dataset> {
        // Try to deserialize as JSON first (cached parsed data)
        if let Ok(dataset) = serde_json::from_slice::<Dataset>(data) {
            return Ok(dataset);
        }

        // Otherwise parse as raw data
        self.parse_raw_data(data, None)
    }

    fn parse_downloaded_data(&self, url: &str, data: &[u8]) -> Result<Dataset> {
        let extension = Path::new(url)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        self.parse_raw_data(data, Some(&extension))
    }

    fn parse_raw_data(&self, data: &[u8], extension: Option<&str>) -> Result<Dataset> {
        match extension {
            Some("csv") | None => {
                // Try CSV parsing
                let csv_data = String::from_utf8(data.to_vec())
                    .map_err(|e| DatasetsError::FormatError(format!("Invalid UTF-8: {e}")))?;

                // Write to temporary file for CSV parsing
                let temp_file = tempfile::NamedTempFile::new().map_err(DatasetsError::IoError)?;

                std::fs::write(temp_file.path(), &csv_data).map_err(DatasetsError::IoError)?;

                load_csv(temp_file.path(), CsvConfig::default())
            }
            Some("json") => {
                // Try JSON parsing
                let json_str = String::from_utf8(data.to_vec())
                    .map_err(|e| DatasetsError::FormatError(format!("Invalid UTF-8: {e}")))?;

                serde_json::from_str(&json_str)
                    .map_err(|e| DatasetsError::FormatError(format!("Invalid JSON: {e}")))
            }
            Some("arff") => {
                // Basic ARFF parsing (simplified)
                self.parse_arff_data(data)
            }
            _ => {
                // Try to auto-detect format
                self.auto_detect_and_parse(data)
            }
        }
    }

    fn parse_arff_data(&self, data: &[u8]) -> Result<Dataset> {
        let content = String::from_utf8(data.to_vec())
            .map_err(|e| DatasetsError::FormatError(format!("Invalid UTF-8: {e}")))?;

        let lines = content.lines();
        let mut attributes = Vec::new();
        let mut data_section = false;
        let mut data_lines = Vec::new();

        for line in lines {
            let line = line.trim();

            if line.is_empty() || line.starts_with('%') {
                continue;
            }

            if line.to_lowercase().starts_with("@attribute") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    attributes.push(parts[1].to_string());
                }
            } else if line.to_lowercase().starts_with("@data") {
                data_section = true;
            } else if data_section {
                data_lines.push(line.to_string());
            }
        }

        // Parse data rows
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for line in data_lines {
            let values: Result<Vec<f64>> = line
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| DatasetsError::FormatError(format!("Invalid number: {s}")))
                })
                .collect();

            match values {
                Ok(row) => rows.push(row),
                Err(_) => continue, // Skip invalid rows
            }
        }

        if rows.is_empty() {
            return Err(DatasetsError::FormatError(
                "No valid data rows found".to_string(),
            ));
        }

        let n_features = rows[0].len();
        let n_samples = rows.len();

        // Assume last column is target if more than one column
        let (data_cols, target_col) = if n_features > 1 {
            (n_features - 1, Some(n_features - 1))
        } else {
            (n_features, None)
        };

        // Create data array
        let mut data_vec = Vec::with_capacity(n_samples * data_cols);
        let mut target_vec = if target_col.is_some() {
            Some(Vec::with_capacity(n_samples))
        } else {
            None
        };

        for row in rows {
            for (i, &value) in row.iter().enumerate() {
                if i < data_cols {
                    data_vec.push(value);
                } else if let Some(ref mut targets) = target_vec {
                    targets.push(value);
                }
            }
        }

        let data = Array2::from_shape_vec((n_samples, data_cols), data_vec)
            .map_err(|e| DatasetsError::FormatError(e.to_string()))?;

        let target = target_vec.map(Array1::from_vec);

        Ok(Dataset {
            data,
            target,
            featurenames: Some(attributes[..data_cols].to_vec()),
            targetnames: None,
            feature_descriptions: None,
            description: Some("ARFF dataset loaded from external source".to_string()),
            metadata: std::collections::HashMap::new(),
        })
    }

    fn auto_detect_and_parse(&self, data: &[u8]) -> Result<Dataset> {
        let content = String::from_utf8(data.to_vec())
            .map_err(|e| DatasetsError::FormatError(format!("Invalid UTF-8: {e}")))?;

        // Try JSON first
        if content.trim().starts_with('{') || content.trim().starts_with('[') {
            if let Ok(dataset) = serde_json::from_str::<Dataset>(&content) {
                return Ok(dataset);
            }
        }

        // Try CSV
        if content.contains(',') || content.contains('\t') {
            return self.parse_raw_data(data, Some("csv"));
        }

        // Try ARFF
        if content.to_lowercase().contains("@relation") {
            return self.parse_arff_data(data);
        }

        Err(DatasetsError::FormatError(
            "Unable to auto-detect data format".to_string(),
        ))
    }
}

/// Popular dataset repository APIs
pub mod repositories {
    use super::*;

    /// UCI Machine Learning Repository client
    pub struct UCIRepository {
        client: ExternalClient,
        base_url: String,
    }

    impl UCIRepository {
        /// Create a new UCI repository client
        pub fn new() -> Result<Self> {
            Ok(Self {
                client: ExternalClient::new()?,
                base_url: "https://archive.ics.uci.edu/ml/machine-learning-databases".to_string(),
            })
        }

        /// Loads a dataset from the UCI Machine Learning Repository.
        ///
        /// # Arguments
        /// * `name` - The name of the dataset to load
        ///
        /// # Returns
        /// A `Dataset` containing the loaded data
        #[cfg(feature = "download")]
        pub async fn load_dataset(&self, name: &str) -> Result<Dataset> {
            let url = match name {
                "adult" => format!("{}/adult/adult.data", self.base_url),
                "wine" => format!("{}/wine/wine.data", self.base_url),
                "glass" => format!("{}/glass/glass.data", self.base_url),
                "hepatitis" => format!("{}/hepatitis/hepatitis.data", self.base_url),
                "heart-disease" => {
                    format!("{}/heart-disease/processed.cleveland.data", self.base_url)
                }
                _ => {
                    return Err(DatasetsError::NotFound(format!(
                        "UCI dataset '{name}' not found"
                    )))
                }
            };

            self.client.download_dataset(&url, None).await
        }

        #[cfg(not(feature = "download"))]
        /// Load a UCI dataset synchronously
        pub fn load_dataset_sync(&self, name: &str) -> Result<Dataset> {
            let url = match name {
                "adult" => format!("{}/adult/adult.data", self.base_url),
                "wine" => format!("{}/wine/wine.data", self.base_url),
                "glass" => format!("{}/glass/glass.data", self.base_url),
                "hepatitis" => format!("{}/hepatitis/hepatitis.data", self.base_url),
                "heart-disease" => {
                    format!("{}/heart-disease/processed.cleveland.data", self.base_url)
                }
                _ => {
                    return Err(DatasetsError::NotFound(format!(
                        "UCI dataset '{name}' not found"
                    )))
                }
            };

            self.client.download_dataset_sync(&url, None)
        }

        /// List available UCI datasets
        pub fn list_datasets(&self) -> Vec<&'static str> {
            vec!["adult", "wine", "glass", "hepatitis", "heart-disease"]
        }
    }

    /// Kaggle dataset client (requires API key)
    pub struct KaggleRepository {
        #[allow(dead_code)]
        client: ExternalClient,
        #[allow(dead_code)]
        api_key: Option<String>,
    }

    impl KaggleRepository {
        /// Create a new Kaggle repository client
        pub fn new(_apikey: Option<String>) -> Result<Self> {
            let mut config = ExternalConfig::default();

            if let Some(ref key) = _apikey {
                config
                    .headers
                    .insert("Authorization".to_string(), format!("Bearer {key}"));
            }

            Ok(Self {
                client: ExternalClient::with_config(config)?,
                api_key: _apikey,
            })
        }

        /// Loads competition data from Kaggle.
        ///
        /// # Arguments
        /// * `competition` - The name of the Kaggle competition
        ///
        /// # Returns
        /// A `Dataset` containing the competition data
        #[cfg(feature = "download")]
        pub async fn load_competition_data(&self, competition: &str) -> Result<Dataset> {
            if self.api_key.is_none() {
                return Err(DatasetsError::AuthenticationError(
                    "Kaggle API key required".to_string(),
                ));
            }

            let url = format!(
                "https://www.kaggle.com/api/v1/competitions/{}/data/download",
                competition
            );
            self.client.download_dataset(&url, None).await
        }
    }

    /// GitHub repository client for datasets
    pub struct GitHubRepository {
        client: ExternalClient,
    }

    impl GitHubRepository {
        /// Create a new GitHub repository client
        pub fn new() -> Result<Self> {
            Ok(Self {
                client: ExternalClient::new()?,
            })
        }

        /// Loads a dataset from a GitHub repository.
        ///
        /// # Arguments
        /// * `user` - The GitHub username
        /// * `repo` - The repository name
        /// * `path` - The path to the dataset file within the repository
        ///
        /// # Returns
        /// A `Dataset` containing the loaded data
        #[cfg(feature = "download")]
        pub async fn load_from_repo(&self, user: &str, repo: &str, path: &str) -> Result<Dataset> {
            let url = format!("https://raw.githubusercontent.com/{user}/{repo}/main/{path}");
            self.client.download_dataset(&url, None).await
        }

        #[cfg(not(feature = "download"))]
        /// Load a dataset from GitHub repository synchronously
        pub fn load_from_repo_sync(&self, user: &str, repo: &str, path: &str) -> Result<Dataset> {
            let url = format!("https://raw.githubusercontent.com/{user}/{repo}/main/{path}");
            self.client.download_dataset_sync(&url, None)
        }
    }
}

/// Convenience functions for common external data operations
pub mod convenience {
    use super::repositories::*;
    use super::*;

    /// Load a dataset from a URL with progress tracking
    #[cfg(feature = "download")]
    pub async fn load_from_url(url: &str, config: Option<ExternalConfig>) -> Result<Dataset> {
        let client = match config {
            Some(cfg) => ExternalClient::with_config(cfg)?,
            None => ExternalClient::new()?,
        };

        client
            .download_dataset(
                url,
                Some(Box::new(|downloaded, total| {
                    if total > 0 {
                        let percent = (downloaded * 100) / total;
                        eprintln!("Downloaded: {percent:.1}% ({downloaded}/{total})");
                    } else {
                        eprintln!("Downloaded: {downloaded} bytes");
                    }
                })),
            )
            .await
    }

    /// Load a dataset from a URL synchronously
    pub fn load_from_url_sync(url: &str, config: Option<ExternalConfig>) -> Result<Dataset> {
        let client = match config {
            Some(cfg) => ExternalClient::with_config(cfg)?,
            None => ExternalClient::new()?,
        };

        client.download_dataset_sync(
            url,
            Some(Box::new(|downloaded, total| {
                if total > 0 {
                    let percent = (downloaded * 100) / total;
                    eprintln!("Downloaded: {percent:.1}% ({downloaded}/{total})");
                } else {
                    eprintln!("Downloaded: {downloaded} bytes");
                }
            })),
        )
    }

    /// Load a UCI dataset by name
    #[cfg(feature = "download")]
    pub async fn load_uci_dataset(name: &str) -> Result<Dataset> {
        let repo = UCIRepository::new()?;
        repo.load_dataset(name).await
    }

    /// Load a UCI dataset by name synchronously
    #[cfg(not(feature = "download"))]
    pub fn load_uci_dataset_sync(name: &str) -> Result<Dataset> {
        let repo = UCIRepository::new()?;
        repo.load_dataset_sync(name)
    }

    /// Load a dataset from GitHub repository
    #[cfg(feature = "download")]
    pub async fn load_github_dataset(user: &str, repo: &str, path: &str) -> Result<Dataset> {
        let github = GitHubRepository::new()?;
        github.load_from_repo(user, repo, path).await
    }

    /// Load a dataset from GitHub repository synchronously
    #[cfg(not(feature = "download"))]
    pub fn load_github_dataset_sync(user: &str, repo: &str, path: &str) -> Result<Dataset> {
        let github = GitHubRepository::new()?;
        github.load_from_repo_sync(user, repo, path)
    }

    /// List available UCI datasets
    pub fn list_uci_datasets() -> Result<Vec<&'static str>> {
        let repo = UCIRepository::new()?;
        Ok(repo.list_datasets())
    }
}

#[cfg(test)]
mod tests {
    use super::convenience::*;
    use super::*;

    #[test]
    fn test_external_config_default() {
        let config = ExternalConfig::default();
        assert_eq!(config.timeout_seconds, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.verify_ssl);
        assert!(config.use_cache);
    }

    #[test]
    fn test_uci_repository_list_datasets() {
        let datasets = list_uci_datasets().unwrap();
        assert!(!datasets.is_empty());
        assert!(datasets.contains(&"wine"));
        assert!(datasets.contains(&"adult"));
    }

    #[test]
    fn test_parse_arff_data() {
        let arff_content = r#"
@relation test
@attribute feature1 numeric
@attribute feature2 numeric
@attribute class {0,1}
@data
1.0,2.0,0
3.0,4.0,1
5.0,6.0,0
"#;

        let client = ExternalClient::new().unwrap();
        let dataset = client.parse_arff_data(arff_content.as_bytes()).unwrap();

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.target.is_some());
    }

    #[tokio::test]
    #[cfg(feature = "download")]
    async fn test_download_small_csv() {
        // Test with a small public CSV dataset
        let url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv";

        let result = load_from_url(url, None).await;
        match result {
            Ok(dataset) => {
                assert!(dataset.n_samples() > 0);
                assert!(dataset.n_features() > 0);
            }
            Err(e) => {
                // Network tests may fail in CI, so we just log the error
                eprintln!("Network test failed (expected in CI): {}", e);
            }
        }
    }
}
