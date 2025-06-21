//! Dataset caching functionality

use crate::error::{DatasetsError, Result};
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};
use std::cell::RefCell;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The base directory name for caching datasets
const CACHE_DIR_NAME: &str = "scirs2-datasets";

/// Default cache size for in-memory caching
const DEFAULT_CACHE_SIZE: usize = 100;

/// Default TTL for in-memory cache (in seconds)
const DEFAULT_CACHE_TTL: u64 = 3600; // 1 hour

/// Default maximum cache size on disk (in bytes) - 500 MB
const DEFAULT_MAX_CACHE_SIZE: u64 = 500 * 1024 * 1024;

/// Cache directory environment variable
const CACHE_DIR_ENV: &str = "SCIRS2_CACHE_DIR";

/// Compute SHA256 hash of a file
fn sha256_hash_file(path: &Path) -> std::result::Result<String, String> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Registry entry for dataset files
pub struct RegistryEntry {
    /// SHA256 hash of the file
    pub sha256: &'static str,
    /// URL to download the file from
    pub url: &'static str,
}

/// Get the platform-specific cache directory for downloading and storing datasets
///
/// The cache directory is determined in the following order:
/// 1. Environment variable `SCIRS2_CACHE_DIR` if set
/// 2. Platform-specific cache directory:
///    - Windows: `%LOCALAPPDATA%\scirs2-datasets`
///    - macOS: `~/Library/Caches/scirs2-datasets`
///    - Linux/Unix: `~/.cache/scirs2-datasets` (respects XDG_CACHE_HOME)
/// 3. Fallback to `~/.scirs2-datasets` if platform-specific directory fails
pub fn get_cache_dir() -> Result<PathBuf> {
    // Check environment variable first
    if let Ok(cache_dir) = std::env::var(CACHE_DIR_ENV) {
        let cache_path = PathBuf::from(cache_dir);
        ensure_directory_exists(&cache_path)?;
        return Ok(cache_path);
    }

    // Try platform-specific cache directory
    if let Some(cache_dir) = get_platform_cache_dir() {
        ensure_directory_exists(&cache_dir)?;
        return Ok(cache_dir);
    }

    // Fallback to home directory
    let home_dir = dirs::home_dir()
        .ok_or_else(|| DatasetsError::CacheError("Could not find home directory".to_string()))?;
    let cache_dir = home_dir.join(format!(".{}", CACHE_DIR_NAME));
    ensure_directory_exists(&cache_dir)?;

    Ok(cache_dir)
}

/// Get platform-specific cache directory
fn get_platform_cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir().map(|dir| dir.join(CACHE_DIR_NAME))
    }
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir().map(|dir| dir.join("Library").join("Caches").join(CACHE_DIR_NAME))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        // Linux/Unix: Use XDG cache directory
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            Some(PathBuf::from(xdg_cache).join(CACHE_DIR_NAME))
        } else {
            dirs::home_dir().map(|home| home.join(".cache").join(CACHE_DIR_NAME))
        }
    }
}

/// Ensure a directory exists, creating it if necessary
fn ensure_directory_exists(dir: &Path) -> Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| {
            DatasetsError::CacheError(format!("Failed to create cache directory: {}", e))
        })?;
    }
    Ok(())
}

/// Fetch a dataset file from either cache or download it from the URL
///
/// This function will:
/// 1. Check if the file exists in the cache directory
/// 2. If not, download it from the URL in the registry entry
/// 3. Store it in the cache directory
/// 4. Return the path to the cached file
///
/// # Arguments
///
/// * `filename` - The name of the file to fetch
/// * `registry_entry` - Optional registry entry containing URL and SHA256 hash
///
/// # Returns
///
/// * `Ok(PathBuf)` - Path to the cached file
/// * `Err(String)` - Error message if fetching fails
pub fn fetch_data(
    filename: &str,
    registry_entry: Option<&RegistryEntry>,
) -> std::result::Result<PathBuf, String> {
    // Get the cache directory
    let cache_dir = match get_cache_dir() {
        Ok(dir) => dir,
        Err(e) => return Err(format!("Failed to get cache directory: {}", e)),
    };

    // Check if file exists in cache
    let cache_path = cache_dir.join(filename);
    if cache_path.exists() {
        return Ok(cache_path);
    }

    // If not in cache, fetch from the URL
    let entry = match registry_entry {
        Some(entry) => entry,
        None => return Err(format!("No registry entry found for {}", filename)),
    };

    // Create a temporary file to download to
    let temp_dir = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let temp_file = temp_dir.path().join(filename);

    // Download the file
    let response = ureq::get(entry.url)
        .call()
        .map_err(|e| format!("Failed to download {}: {}", filename, e))?;

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(&temp_file)
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    std::io::copy(&mut reader, &mut file).map_err(|e| format!("Failed to download file: {}", e))?;

    // Verify the SHA256 hash of the downloaded file if provided
    if !entry.sha256.is_empty() {
        let computed_hash = sha256_hash_file(&temp_file)?;
        if computed_hash != entry.sha256 {
            return Err(format!(
                "SHA256 hash mismatch for {}: expected {}, got {}",
                filename, entry.sha256, computed_hash
            ));
        }
    }

    // Move the file to the cache
    fs::create_dir_all(&cache_dir).map_err(|e| format!("Failed to create cache dir: {}", e))?;
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create cache dir: {}", e))?;
    }

    fs::copy(&temp_file, &cache_path).map_err(|e| format!("Failed to copy to cache: {}", e))?;

    Ok(cache_path)
}

/// File path wrapper for hashing
#[derive(Clone, Debug, Eq, PartialEq)]
struct FileCacheKey(String);

impl Hash for FileCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Manages caching of downloaded datasets, using both file-based and in-memory caching
///
/// This implementation uses scirs2-core::cache's TTLSizedCache for in-memory caching,
/// while maintaining the file-based persistence for long-term storage.
pub struct DatasetCache {
    /// Directory for file-based caching
    cache_dir: PathBuf,
    /// In-memory cache for frequently accessed datasets
    mem_cache: RefCell<TTLSizedCache<FileCacheKey, Vec<u8>>>,
    /// Maximum cache size in bytes (0 means unlimited)
    max_cache_size: u64,
    /// Whether to operate in offline mode (no downloads)
    offline_mode: bool,
}

impl Default for DatasetCache {
    fn default() -> Self {
        let cache_dir = get_cache_dir().expect("Could not get cache directory");

        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(DEFAULT_CACHE_SIZE)
                .with_ttl(DEFAULT_CACHE_TTL)
                .build_sized_cache(),
        );

        // Check if offline mode is enabled via environment variable
        let offline_mode = std::env::var("SCIRS2_OFFLINE")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        DatasetCache {
            cache_dir,
            mem_cache,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
            offline_mode,
        }
    }
}

impl DatasetCache {
    /// Create a new dataset cache with the given cache directory and default memory cache
    pub fn new(cache_dir: PathBuf) -> Self {
        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(DEFAULT_CACHE_SIZE)
                .with_ttl(DEFAULT_CACHE_TTL)
                .build_sized_cache(),
        );

        let offline_mode = std::env::var("SCIRS2_OFFLINE")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        DatasetCache {
            cache_dir,
            mem_cache,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
            offline_mode,
        }
    }

    /// Create a new dataset cache with custom settings
    pub fn with_config(cache_dir: PathBuf, cache_size: usize, ttl_seconds: u64) -> Self {
        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(cache_size)
                .with_ttl(ttl_seconds)
                .build_sized_cache(),
        );

        let offline_mode = std::env::var("SCIRS2_OFFLINE")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        DatasetCache {
            cache_dir,
            mem_cache,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
            offline_mode,
        }
    }

    /// Create a new dataset cache with comprehensive configuration
    pub fn with_full_config(
        cache_dir: PathBuf,
        cache_size: usize,
        ttl_seconds: u64,
        max_cache_size: u64,
        offline_mode: bool,
    ) -> Self {
        let mem_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(cache_size)
                .with_ttl(ttl_seconds)
                .build_sized_cache(),
        );

        DatasetCache {
            cache_dir,
            mem_cache,
            max_cache_size,
            offline_mode,
        }
    }

    /// Create the cache directory if it doesn't exist
    pub fn ensure_cache_dir(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to create cache directory: {}", e))
            })?;
        }
        Ok(())
    }

    /// Get the path to a cached file
    pub fn get_cached_path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(name)
    }

    /// Check if a file is already cached (either in memory or on disk)
    pub fn is_cached(&self, name: &str) -> bool {
        // Check memory cache first
        let key = FileCacheKey(name.to_string());
        if self.mem_cache.borrow_mut().get(&key).is_some() {
            return true;
        }

        // Then check file system
        self.get_cached_path(name).exists()
    }

    /// Read a cached file as bytes
    ///
    /// This method checks the in-memory cache first, and falls back to the file system if needed.
    /// When reading from the file system, the result is also stored in the in-memory cache.
    pub fn read_cached(&self, name: &str) -> Result<Vec<u8>> {
        // Try memory cache first
        let key = FileCacheKey(name.to_string());
        if let Some(data) = self.mem_cache.borrow_mut().get(&key) {
            return Ok(data);
        }

        // Fall back to file system cache
        let path = self.get_cached_path(name);
        if !path.exists() {
            return Err(DatasetsError::CacheError(format!(
                "Cached file does not exist: {}",
                name
            )));
        }

        let mut file = File::open(path)
            .map_err(|e| DatasetsError::CacheError(format!("Failed to open cached file: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| DatasetsError::CacheError(format!("Failed to read cached file: {}", e)))?;

        // Update memory cache
        self.mem_cache.borrow_mut().insert(key, buffer.clone());

        Ok(buffer)
    }

    /// Write data to both the file cache and memory cache
    pub fn write_cached(&self, name: &str, data: &[u8]) -> Result<()> {
        self.ensure_cache_dir()?;

        // Check if writing this file would exceed cache size limit
        if self.max_cache_size > 0 {
            let current_size = self.get_cache_size_bytes()?;
            let new_file_size = data.len() as u64;

            if current_size + new_file_size > self.max_cache_size {
                self.cleanup_cache_to_fit(new_file_size)?;
            }
        }

        // Write to file system cache
        let path = self.get_cached_path(name);
        let mut file = File::create(path).map_err(|e| {
            DatasetsError::CacheError(format!("Failed to create cache file: {}", e))
        })?;

        file.write_all(data).map_err(|e| {
            DatasetsError::CacheError(format!("Failed to write to cache file: {}", e))
        })?;

        // Update memory cache
        let key = FileCacheKey(name.to_string());
        self.mem_cache.borrow_mut().insert(key, data.to_vec());

        Ok(())
    }

    /// Clear the entire cache (both memory and file-based)
    pub fn clear_cache(&self) -> Result<()> {
        // Clear file system cache
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| DatasetsError::CacheError(format!("Failed to clear cache: {}", e)))?;
        }

        // Clear memory cache
        self.mem_cache.borrow_mut().clear();

        Ok(())
    }

    /// Remove a specific cached file (from both memory and file system)
    pub fn remove_cached(&self, name: &str) -> Result<()> {
        // Remove from file system
        let path = self.get_cached_path(name);
        if path.exists() {
            fs::remove_file(path).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to remove cached file: {}", e))
            })?;
        }

        // Remove from memory cache
        let key = FileCacheKey(name.to_string());
        self.mem_cache.borrow_mut().remove(&key);

        Ok(())
    }

    /// Compute a hash for a filename or URL
    pub fn hash_filename(name: &str) -> String {
        let hash = blake3::hash(name.as_bytes());
        hash.to_hex().to_string()
    }

    /// Get the total size of the cache in bytes
    pub fn get_cache_size_bytes(&self) -> Result<u64> {
        let mut total_size = 0u64;

        if self.cache_dir.exists() {
            let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to read cache directory: {}", e))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    DatasetsError::CacheError(format!("Failed to read directory entry: {}", e))
                })?;

                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        total_size += metadata.len();
                    }
                }
            }
        }

        Ok(total_size)
    }

    /// Clean up cache to fit a new file of specified size
    ///
    /// This method removes the oldest files first until there's enough space
    /// for the new file plus some buffer space.
    fn cleanup_cache_to_fit(&self, needed_size: u64) -> Result<()> {
        if self.max_cache_size == 0 {
            return Ok(()); // No size limit
        }

        let current_size = self.get_cache_size_bytes()?;
        let target_size = (self.max_cache_size as f64 * 0.8) as u64; // Leave 20% buffer
        let total_needed = current_size + needed_size;

        if total_needed <= target_size {
            return Ok(()); // No cleanup needed
        }

        let size_to_free = total_needed - target_size;

        // Get all files with their modification times
        let mut files_with_times = Vec::new();

        if self.cache_dir.exists() {
            let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to read cache directory: {}", e))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    DatasetsError::CacheError(format!("Failed to read directory entry: {}", e))
                })?;

                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        if let Ok(modified) = metadata.modified() {
                            files_with_times.push((entry.path(), metadata.len(), modified));
                        }
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        files_with_times.sort_by_key(|(_, _, modified)| *modified);

        // Remove files until we've freed enough space
        let mut freed_size = 0u64;
        for (path, size, _) in files_with_times {
            if freed_size >= size_to_free {
                break;
            }

            // Remove from memory cache first
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                let key = FileCacheKey(filename.to_string());
                self.mem_cache.borrow_mut().remove(&key);
            }

            // Remove file
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("Warning: Failed to remove cache file {:?}: {}", path, e);
            } else {
                freed_size += size;
            }
        }

        Ok(())
    }

    /// Set offline mode
    pub fn set_offline_mode(&mut self, offline: bool) {
        self.offline_mode = offline;
    }

    /// Check if cache is in offline mode
    pub fn is_offline(&self) -> bool {
        self.offline_mode
    }

    /// Set maximum cache size in bytes (0 for unlimited)
    pub fn set_max_cache_size(&mut self, max_size: u64) {
        self.max_cache_size = max_size;
    }

    /// Get maximum cache size in bytes
    pub fn max_cache_size(&self) -> u64 {
        self.max_cache_size
    }

    /// Get detailed cache information
    pub fn get_detailed_stats(&self) -> Result<DetailedCacheStats> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;
        let mut files = Vec::new();

        if self.cache_dir.exists() {
            let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to read cache directory: {}", e))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    DatasetsError::CacheError(format!("Failed to read directory entry: {}", e))
                })?;

                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        let size = metadata.len();
                        total_size += size;
                        file_count += 1;

                        if let Some(filename) = entry.file_name().to_str() {
                            files.push(CacheFileInfo {
                                name: filename.to_string(),
                                size_bytes: size,
                                modified: metadata.modified().ok(),
                            });
                        }
                    }
                }
            }
        }

        // Sort files by size (largest first)
        files.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

        Ok(DetailedCacheStats {
            total_size_bytes: total_size,
            file_count,
            cache_dir: self.cache_dir.clone(),
            max_cache_size: self.max_cache_size,
            offline_mode: self.offline_mode,
            files,
        })
    }
}

/// Downloads data from a URL and returns it as bytes, using the cache when possible
#[cfg(feature = "download")]
pub fn download_data(url: &str, force_download: bool) -> Result<Vec<u8>> {
    let cache = DatasetCache::default();
    let cache_key = DatasetCache::hash_filename(url);

    // Check if the data is already cached
    if !force_download && cache.is_cached(&cache_key) {
        return cache.read_cached(&cache_key);
    }

    // Download the data
    let response = reqwest::blocking::get(url).map_err(|e| {
        DatasetsError::DownloadError(format!("Failed to download from {}: {}", url, e))
    })?;

    if !response.status().is_success() {
        return Err(DatasetsError::DownloadError(format!(
            "Failed to download from {}: HTTP status {}",
            url,
            response.status()
        )));
    }

    let data = response.bytes().map_err(|e| {
        DatasetsError::DownloadError(format!("Failed to read response data: {}", e))
    })?;

    let data_vec = data.to_vec();

    // Cache the data
    cache.write_cached(&cache_key, &data_vec)?;

    Ok(data_vec)
}

// Stub for when download feature is not enabled
#[cfg(not(feature = "download"))]
/// Downloads data from a URL or retrieves it from cache
///
/// This is a stub implementation when the download feature is not enabled.
/// It returns an error informing the user to enable the download feature.
///
/// # Arguments
///
/// * `_url` - The URL to download from
/// * `_force_download` - If true, force a new download instead of using cache
///
/// # Returns
///
/// * An error indicating that the download feature is not enabled
pub fn download_data(_url: &str, _force_download: bool) -> Result<Vec<u8>> {
    Err(DatasetsError::Other(
        "Download feature is not enabled. Recompile with --features download".to_string(),
    ))
}

/// Cache management utilities
#[derive(Default)]
pub struct CacheManager {
    cache: DatasetCache,
}

impl CacheManager {
    /// Create a new cache manager with custom settings
    pub fn new(cache_dir: PathBuf, cache_size: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: DatasetCache::with_config(cache_dir, cache_size, ttl_seconds),
        }
    }

    /// Create a cache manager with comprehensive configuration
    pub fn with_full_config(
        cache_dir: PathBuf,
        cache_size: usize,
        ttl_seconds: u64,
        max_cache_size: u64,
        offline_mode: bool,
    ) -> Self {
        Self {
            cache: DatasetCache::with_full_config(
                cache_dir,
                cache_size,
                ttl_seconds,
                max_cache_size,
                offline_mode,
            ),
        }
    }

    /// Get basic cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let cache_dir = &self.cache.cache_dir;
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        if cache_dir.exists() {
            if let Ok(entries) = fs::read_dir(cache_dir) {
                for entry in entries.flatten() {
                    if let Ok(metadata) = entry.metadata() {
                        if metadata.is_file() {
                            total_size += metadata.len();
                            file_count += 1;
                        }
                    }
                }
            }
        }

        CacheStats {
            total_size_bytes: total_size,
            file_count,
            cache_dir: cache_dir.clone(),
        }
    }

    /// Get detailed cache statistics
    pub fn get_detailed_stats(&self) -> Result<DetailedCacheStats> {
        self.cache.get_detailed_stats()
    }

    /// Set offline mode
    pub fn set_offline_mode(&mut self, offline: bool) {
        self.cache.set_offline_mode(offline);
    }

    /// Check if in offline mode
    pub fn is_offline(&self) -> bool {
        self.cache.is_offline()
    }

    /// Set maximum cache size in bytes (0 for unlimited)
    pub fn set_max_cache_size(&mut self, max_size: u64) {
        self.cache.set_max_cache_size(max_size);
    }

    /// Get maximum cache size in bytes
    pub fn max_cache_size(&self) -> u64 {
        self.cache.max_cache_size()
    }

    /// Clear all cached data
    pub fn clear_all(&self) -> Result<()> {
        self.cache.clear_cache()
    }

    /// Remove specific cached file
    pub fn remove(&self, name: &str) -> Result<()> {
        self.cache.remove_cached(name)
    }

    /// Remove old files to free up space
    pub fn cleanup_old_files(&self, target_size: u64) -> Result<()> {
        self.cache.cleanup_cache_to_fit(target_size)
    }

    /// List all cached files
    pub fn list_cached_files(&self) -> Result<Vec<String>> {
        let cache_dir = &self.cache.cache_dir;
        let mut files = Vec::new();

        if cache_dir.exists() {
            let entries = fs::read_dir(cache_dir).map_err(|e| {
                DatasetsError::CacheError(format!("Failed to read cache directory: {}", e))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    DatasetsError::CacheError(format!("Failed to read directory entry: {}", e))
                })?;

                if let Some(filename) = entry.file_name().to_str() {
                    files.push(filename.to_string());
                }
            }
        }

        files.sort();
        Ok(files)
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache.cache_dir
    }

    /// Check if a file is cached
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.is_cached(name)
    }

    /// Print detailed cache report
    pub fn print_cache_report(&self) -> Result<()> {
        let stats = self.get_detailed_stats()?;

        println!("=== Cache Report ===");
        println!("Cache Directory: {}", stats.cache_dir.display());
        println!(
            "Total Size: {} ({} files)",
            stats.formatted_size(),
            stats.file_count
        );
        println!("Max Size: {}", stats.formatted_max_size());

        if stats.max_cache_size > 0 {
            println!("Usage: {:.1}%", stats.usage_percentage() * 100.0);
        }

        println!(
            "Offline Mode: {}",
            if stats.offline_mode {
                "Enabled"
            } else {
                "Disabled"
            }
        );

        if !stats.files.is_empty() {
            println!("\nCached Files:");
            for file in &stats.files {
                println!(
                    "  {} - {} ({})",
                    file.name,
                    file.formatted_size(),
                    file.formatted_modified()
                );
            }
        }

        Ok(())
    }
}

/// Cache statistics
pub struct CacheStats {
    /// Total size of all cached files in bytes
    pub total_size_bytes: u64,
    /// Number of cached files
    pub file_count: usize,
    /// Cache directory path
    pub cache_dir: PathBuf,
}

/// Detailed cache statistics with file-level information
pub struct DetailedCacheStats {
    /// Total size of all cached files in bytes
    pub total_size_bytes: u64,
    /// Number of cached files
    pub file_count: usize,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Maximum cache size (0 = unlimited)
    pub max_cache_size: u64,
    /// Whether cache is in offline mode
    pub offline_mode: bool,
    /// Information about individual cached files
    pub files: Vec<CacheFileInfo>,
}

/// Information about a cached file
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    /// Name of the cached file
    pub name: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last modified time
    pub modified: Option<std::time::SystemTime>,
}

impl CacheStats {
    /// Get total size formatted as human-readable string
    pub fn formatted_size(&self) -> String {
        format_bytes(self.total_size_bytes)
    }
}

impl DetailedCacheStats {
    /// Get total size formatted as human-readable string
    pub fn formatted_size(&self) -> String {
        format_bytes(self.total_size_bytes)
    }

    /// Get max cache size formatted as human-readable string
    pub fn formatted_max_size(&self) -> String {
        if self.max_cache_size == 0 {
            "Unlimited".to_string()
        } else {
            format_bytes(self.max_cache_size)
        }
    }

    /// Get cache usage percentage (0.0-1.0)
    pub fn usage_percentage(&self) -> f64 {
        if self.max_cache_size == 0 {
            0.0
        } else {
            self.total_size_bytes as f64 / self.max_cache_size as f64
        }
    }
}

impl CacheFileInfo {
    /// Get file size formatted as human-readable string
    pub fn formatted_size(&self) -> String {
        format_bytes(self.size_bytes)
    }

    /// Get formatted modification time
    pub fn formatted_modified(&self) -> String {
        match &self.modified {
            Some(time) => {
                if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                {
                    if let Ok(modified) = time.duration_since(std::time::UNIX_EPOCH) {
                        let diff_secs = now.as_secs().saturating_sub(modified.as_secs());
                        let days = diff_secs / 86400;
                        let hours = (diff_secs % 86400) / 3600;
                        let mins = (diff_secs % 3600) / 60;

                        if days > 0 {
                            format!("{} days ago", days)
                        } else if hours > 0 {
                            format!("{} hours ago", hours)
                        } else if mins > 0 {
                            format!("{} minutes ago", mins)
                        } else {
                            "Just now".to_string()
                        }
                    } else {
                        "Unknown".to_string()
                    }
                } else {
                    "Unknown".to_string()
                }
            }
            None => "Unknown".to_string(),
        }
    }
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    let size = bytes as f64;
    if size < 1024.0 {
        format!("{} B", size)
    } else if size < 1024.0 * 1024.0 {
        format!("{:.1} KB", size / 1024.0)
    } else if size < 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} MB", size / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", size / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Batch operation result containing success/failure information
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of successful operations
    pub success_count: usize,
    /// Number of failed operations
    pub failure_count: usize,
    /// List of failed items with error messages
    pub failures: Vec<(String, String)>,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Total time taken for the batch operation
    pub elapsed_time: std::time::Duration,
}

impl BatchResult {
    /// Create a new empty batch result
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            failures: Vec::new(),
            total_bytes: 0,
            elapsed_time: std::time::Duration::ZERO,
        }
    }

    /// Check if all operations were successful
    pub fn is_all_success(&self) -> bool {
        self.failure_count == 0
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            (self.success_count as f64 / total as f64) * 100.0
        }
    }

    /// Get formatted summary
    pub fn summary(&self) -> String {
        format!(
            "Batch completed: {}/{} successful ({:.1}%), {} bytes processed in {:.2}s",
            self.success_count,
            self.success_count + self.failure_count,
            self.success_rate(),
            format_bytes(self.total_bytes),
            self.elapsed_time.as_secs_f64()
        )
    }
}

impl Default for BatchResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch operations manager for dataset caching
pub struct BatchOperations {
    cache: CacheManager,
    parallel: bool,
    max_retries: usize,
    retry_delay: std::time::Duration,
}

impl BatchOperations {
    /// Create a new batch operations manager
    pub fn new(cache: CacheManager) -> Self {
        Self {
            cache,
            parallel: true,
            max_retries: 3,
            retry_delay: std::time::Duration::from_millis(1000),
        }
    }

    /// Configure parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Configure retry settings
    pub fn with_retry_config(
        mut self,
        max_retries: usize,
        retry_delay: std::time::Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = retry_delay;
        self
    }

    /// Download multiple datasets in batch
    #[cfg(feature = "download")]
    pub fn batch_download(&self, urls_and_names: &[(&str, &str)]) -> BatchResult {
        let start_time = std::time::Instant::now();
        let mut result = BatchResult::new();

        if self.parallel {
            self.batch_download_parallel(urls_and_names, &mut result)
        } else {
            self.batch_download_sequential(urls_and_names, &mut result)
        }

        result.elapsed_time = start_time.elapsed();
        result
    }

    #[cfg(feature = "download")]
    fn batch_download_parallel(&self, urls_and_names: &[(&str, &str)], result: &mut BatchResult) {
        use std::fs::File;
        use std::io::Write;
        use std::sync::{Arc, Mutex};
        use std::thread;

        // Ensure cache directory exists before spawning threads
        if let Err(e) = self.cache.cache.ensure_cache_dir() {
            result.failure_count += urls_and_names.len();
            for &(_, name) in urls_and_names {
                result
                    .failures
                    .push((name.to_string(), format!("Cache setup failed: {}", e)));
            }
            return;
        }

        let result = Arc::new(Mutex::new(result));
        let cache_dir = self.cache.cache.cache_dir.clone();
        let max_retries = self.max_retries;
        let retry_delay = self.retry_delay;

        let handles: Vec<_> = urls_and_names
            .iter()
            .map(|&(url, name)| {
                let result_clone = Arc::clone(&result);
                let url = url.to_string();
                let name = name.to_string();
                let cache_dir = cache_dir.clone();

                thread::spawn(move || {
                    let mut success = false;
                    let mut last_error = String::new();
                    let mut downloaded_data = Vec::new();

                    for attempt in 0..=max_retries {
                        match download_data(&url, false) {
                            Ok(data) => {
                                // Write directly to filesystem (bypassing RefCell memory cache)
                                let path = cache_dir.join(&name);
                                match File::create(&path) {
                                    Ok(mut file) => match file.write_all(&data) {
                                        Ok(_) => {
                                            let mut r = result_clone.lock().unwrap();
                                            r.success_count += 1;
                                            r.total_bytes += data.len() as u64;
                                            downloaded_data = data;
                                            success = true;
                                            break;
                                        }
                                        Err(e) => {
                                            last_error =
                                                format!("Failed to write cache file: {}", e);
                                        }
                                    },
                                    Err(e) => {
                                        last_error = format!("Failed to create cache file: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                last_error = format!("Download failed: {}", e);
                                if attempt < max_retries {
                                    thread::sleep(retry_delay);
                                }
                            }
                        }
                    }

                    if !success {
                        let mut r = result_clone.lock().unwrap();
                        r.failure_count += 1;
                        r.failures.push((name, last_error));
                    }

                    (name, success, downloaded_data)
                })
            })
            .collect();

        // Collect results and update memory cache for successful downloads
        let mut successful_downloads = Vec::new();
        for handle in handles {
            if let Ok((name, success, data)) = handle.join() {
                if success && !data.is_empty() {
                    successful_downloads.push((name, data));
                }
            }
        }

        // Update memory cache after all threads complete
        for (name, data) in successful_downloads {
            let key = FileCacheKey(name);
            self.cache.cache.mem_cache.borrow_mut().insert(key, data);
        }
    }

    #[cfg(feature = "download")]
    fn batch_download_sequential(&self, urls_and_names: &[(&str, &str)], result: &mut BatchResult) {
        for &(url, name) in urls_and_names {
            let mut success = false;
            let mut last_error = String::new();

            for attempt in 0..=self.max_retries {
                match download_data(url, false) {
                    Ok(data) => match self.cache.cache.write_cached(name, &data) {
                        Ok(_) => {
                            result.success_count += 1;
                            result.total_bytes += data.len() as u64;
                            success = true;
                            break;
                        }
                        Err(e) => {
                            last_error = format!("Cache write failed: {}", e);
                        }
                    },
                    Err(e) => {
                        last_error = format!("Download failed: {}", e);
                        if attempt < self.max_retries {
                            std::thread::sleep(self.retry_delay);
                        }
                    }
                }
            }

            if !success {
                result.failure_count += 1;
                result.failures.push((name.to_string(), last_error));
            }
        }
    }

    /// Verify integrity of multiple cached files
    pub fn batch_verify_integrity(&self, files_and_hashes: &[(&str, &str)]) -> BatchResult {
        let start_time = std::time::Instant::now();
        let mut result = BatchResult::new();

        for &(filename, expected_hash) in files_and_hashes {
            match self.cache.cache.get_cached_path(filename).exists() {
                true => match sha256_hash_file(&self.cache.cache.get_cached_path(filename)) {
                    Ok(actual_hash) => {
                        if actual_hash == expected_hash {
                            result.success_count += 1;
                            if let Ok(metadata) =
                                std::fs::metadata(self.cache.cache.get_cached_path(filename))
                            {
                                result.total_bytes += metadata.len();
                            }
                        } else {
                            result.failure_count += 1;
                            result.failures.push((
                                filename.to_string(),
                                format!(
                                    "Hash mismatch: expected {}, got {}",
                                    expected_hash, actual_hash
                                ),
                            ));
                        }
                    }
                    Err(e) => {
                        result.failure_count += 1;
                        result.failures.push((
                            filename.to_string(),
                            format!("Hash computation failed: {}", e),
                        ));
                    }
                },
                false => {
                    result.failure_count += 1;
                    result
                        .failures
                        .push((filename.to_string(), "File not found in cache".to_string()));
                }
            }
        }

        result.elapsed_time = start_time.elapsed();
        result
    }

    /// Clean up cache selectively based on patterns
    pub fn selective_cleanup(
        &self,
        patterns: &[&str],
        max_age_days: Option<u32>,
    ) -> Result<BatchResult> {
        let start_time = std::time::Instant::now();
        let mut result = BatchResult::new();

        let cached_files = self.cache.list_cached_files()?;
        let now = std::time::SystemTime::now();

        for filename in cached_files {
            let should_remove = patterns.iter().any(|pattern| {
                filename.contains(pattern) || matches_glob_pattern(&filename, pattern)
            });

            if should_remove {
                let file_path = self.cache.cache.get_cached_path(&filename);

                // Check age if max_age_days is specified
                let remove_due_to_age = if let Some(max_age) = max_age_days {
                    if let Ok(metadata) = std::fs::metadata(&file_path) {
                        if let Ok(modified) = metadata.modified() {
                            if let Ok(age) = now.duration_since(modified) {
                                age.as_secs() > (max_age as u64 * 24 * 3600)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    true // Remove regardless of age if no age limit specified
                };

                if remove_due_to_age {
                    match self.cache.remove(&filename) {
                        Ok(_) => {
                            result.success_count += 1;
                            if let Ok(metadata) = std::fs::metadata(&file_path) {
                                result.total_bytes += metadata.len();
                            }
                        }
                        Err(e) => {
                            result.failure_count += 1;
                            result
                                .failures
                                .push((filename, format!("Removal failed: {}", e)));
                        }
                    }
                }
            }
        }

        result.elapsed_time = start_time.elapsed();
        Ok(result)
    }

    /// Process multiple datasets with a given function
    pub fn batch_process<F, T, E>(&self, names: &[String], processor: F) -> BatchResult
    where
        F: Fn(&str, &[u8]) -> std::result::Result<T, E> + Sync + Send + 'static,
        E: std::fmt::Display,
        T: Send,
    {
        let start_time = std::time::Instant::now();
        let mut result = BatchResult::new();

        if self.parallel {
            self.batch_process_parallel(names, processor, &mut result)
        } else {
            self.batch_process_sequential(names, processor, &mut result)
        }

        result.elapsed_time = start_time.elapsed();
        result
    }

    fn batch_process_parallel<F, T, E>(
        &self,
        names: &[String],
        processor: F,
        result: &mut BatchResult,
    ) where
        F: Fn(&str, &[u8]) -> std::result::Result<T, E> + Sync + Send + 'static,
        E: std::fmt::Display,
        T: Send,
    {
        // For thread safety with the current cache implementation,
        // we need to read all data first, then process in parallel
        let mut data_pairs = Vec::new();

        // Sequential read phase
        for name in names {
            match self.cache.cache.read_cached(name) {
                Ok(data) => data_pairs.push((name.clone(), data)),
                Err(e) => {
                    result.failure_count += 1;
                    result
                        .failures
                        .push((name.clone(), format!("Cache read failed: {}", e)));
                }
            }
        }

        // Parallel processing phase
        if !data_pairs.is_empty() {
            use std::sync::{Arc, Mutex};
            use std::thread;

            let parallel_result = Arc::new(Mutex::new(BatchResult::new()));
            let processor = Arc::new(processor);

            let handles: Vec<_> = data_pairs
                .into_iter()
                .map(|(name, data)| {
                    let result_clone = Arc::clone(&parallel_result);
                    let processor_clone = Arc::clone(&processor);

                    thread::spawn(move || match processor_clone(&name, &data) {
                        Ok(_) => {
                            let mut r = result_clone.lock().unwrap();
                            r.success_count += 1;
                            r.total_bytes += data.len() as u64;
                        }
                        Err(e) => {
                            let mut r = result_clone.lock().unwrap();
                            r.failure_count += 1;
                            r.failures.push((name, format!("Processing failed: {}", e)));
                        }
                    })
                })
                .collect();

            for handle in handles {
                let _ = handle.join();
            }

            // Merge parallel results into main result
            let parallel_result = parallel_result.lock().unwrap();
            result.success_count += parallel_result.success_count;
            result.failure_count += parallel_result.failure_count;
            result.total_bytes += parallel_result.total_bytes;
            result.failures.extend(parallel_result.failures.clone());
        }
    }

    fn batch_process_sequential<F, T, E>(
        &self,
        names: &[String],
        processor: F,
        result: &mut BatchResult,
    ) where
        F: Fn(&str, &[u8]) -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        for name in names {
            match self.cache.cache.read_cached(name) {
                Ok(data) => match processor(name, &data) {
                    Ok(_) => {
                        result.success_count += 1;
                        result.total_bytes += data.len() as u64;
                    }
                    Err(e) => {
                        result.failure_count += 1;
                        result
                            .failures
                            .push((name.clone(), format!("Processing failed: {}", e)));
                    }
                },
                Err(e) => {
                    result.failure_count += 1;
                    result
                        .failures
                        .push((name.clone(), format!("Cache read failed: {}", e)));
                }
            }
        }
    }

    /// Get access to the underlying cache manager
    pub fn cache_manager(&self) -> &CacheManager {
        &self.cache
    }

    /// Write data to cache
    pub fn write_cached(&self, name: &str, data: &[u8]) -> Result<()> {
        self.cache.cache.write_cached(name, data)
    }

    /// Read data from cache
    pub fn read_cached(&self, name: &str) -> Result<Vec<u8>> {
        self.cache.cache.read_cached(name)
    }

    /// List cached files
    pub fn list_cached_files(&self) -> Result<Vec<String>> {
        self.cache.list_cached_files()
    }

    /// Print cache report
    pub fn print_cache_report(&self) -> Result<()> {
        self.cache.print_cache_report()
    }

    /// Get statistics about cached datasets
    pub fn get_cache_statistics(&self) -> Result<BatchResult> {
        let start_time = std::time::Instant::now();
        let mut result = BatchResult::new();

        let cached_files = self.cache.list_cached_files()?;

        for filename in cached_files {
            let file_path = self.cache.cache.get_cached_path(&filename);
            match std::fs::metadata(&file_path) {
                Ok(metadata) => {
                    result.success_count += 1;
                    result.total_bytes += metadata.len();
                }
                Err(e) => {
                    result.failure_count += 1;
                    result
                        .failures
                        .push((filename, format!("Metadata read failed: {}", e)));
                }
            }
        }

        result.elapsed_time = start_time.elapsed();
        Ok(result)
    }
}

/// Simple glob pattern matching for filenames
fn matches_glob_pattern(filename: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return filename.starts_with(prefix) && filename.ends_with(suffix);
        }
    }

    filename == pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_batch_result() {
        let mut result = BatchResult::new();
        assert_eq!(result.success_count, 0);
        assert_eq!(result.failure_count, 0);
        assert!(result.is_all_success());
        assert_eq!(result.success_rate(), 0.0);

        result.success_count = 8;
        result.failure_count = 2;
        result.total_bytes = 1024;

        assert!(!result.is_all_success());
        assert_eq!(result.success_rate(), 80.0);
        assert!(result.summary().contains("8/10 successful"));
        assert!(result.summary().contains("80.0%"));
    }

    #[test]
    fn test_batch_operations_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache_manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);
        let batch_ops = BatchOperations::new(cache_manager)
            .with_parallel(false)
            .with_retry_config(2, std::time::Duration::from_millis(500));

        assert!(!batch_ops.parallel);
        assert_eq!(batch_ops.max_retries, 2);
    }

    #[test]
    fn test_selective_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let cache_manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);
        let batch_ops = BatchOperations::new(cache_manager);

        // Create some test files
        let test_data = vec![0u8; 100];
        batch_ops
            .cache
            .cache
            .write_cached("test1.csv", &test_data)
            .unwrap();
        batch_ops
            .cache
            .cache
            .write_cached("test2.csv", &test_data)
            .unwrap();
        batch_ops
            .cache
            .cache
            .write_cached("data.json", &test_data)
            .unwrap();

        // Clean up files matching pattern
        let result = batch_ops.selective_cleanup(&["*.csv"], None).unwrap();

        assert_eq!(result.success_count, 2); // Should remove test1.csv and test2.csv
        assert!(!batch_ops.cache.is_cached("test1.csv"));
        assert!(!batch_ops.cache.is_cached("test2.csv"));
        assert!(batch_ops.cache.is_cached("data.json")); // Should remain
    }

    #[test]
    fn test_batch_process() {
        let temp_dir = TempDir::new().unwrap();
        let cache_manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);
        let batch_ops = BatchOperations::new(cache_manager).with_parallel(false);

        // Create test files
        let test_data1 = vec![1u8; 100];
        let test_data2 = vec![2u8; 200];
        batch_ops
            .cache
            .cache
            .write_cached("file1.dat", &test_data1)
            .unwrap();
        batch_ops
            .cache
            .cache
            .write_cached("file2.dat", &test_data2)
            .unwrap();

        let files = vec!["file1.dat".to_string(), "file2.dat".to_string()];

        // Process files (verify they're non-empty)
        let result = batch_ops.batch_process(&files, |_name, data| {
            if data.is_empty() {
                Err("Empty file")
            } else {
                Ok(data.len())
            }
        });

        assert_eq!(result.success_count, 2);
        assert_eq!(result.failure_count, 0);
        assert_eq!(result.total_bytes, 300); // 100 + 200
    }

    #[test]
    fn test_get_cache_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let cache_manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);
        let batch_ops = BatchOperations::new(cache_manager);

        // Start with empty cache
        let result = batch_ops.get_cache_statistics().unwrap();
        assert_eq!(result.success_count, 0);

        // Add some files
        let test_data = vec![0u8; 500];
        batch_ops
            .cache
            .cache
            .write_cached("test1.dat", &test_data)
            .unwrap();
        batch_ops
            .cache
            .cache
            .write_cached("test2.dat", &test_data)
            .unwrap();

        let result = batch_ops.get_cache_statistics().unwrap();
        assert_eq!(result.success_count, 2);
        assert_eq!(result.total_bytes, 1000);
    }

    #[test]
    fn test_matches_glob_pattern() {
        assert!(matches_glob_pattern("test.csv", "*"));
        assert!(matches_glob_pattern("test.csv", "*.csv"));
        assert!(matches_glob_pattern("test.csv", "test.*"));
        assert!(matches_glob_pattern("test.csv", "test.csv"));

        assert!(!matches_glob_pattern("test.json", "*.csv"));
        assert!(!matches_glob_pattern("other.csv", "test.*"));
    }

    #[test]
    fn test_cache_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);
        let stats = manager.get_stats();
        assert_eq!(stats.file_count, 0);
    }

    #[test]
    fn test_cache_stats_formatting() {
        let temp_dir = TempDir::new().unwrap();
        let stats = CacheStats {
            total_size_bytes: 1024,
            file_count: 1,
            cache_dir: temp_dir.path().to_path_buf(),
        };

        assert_eq!(stats.formatted_size(), "1.0 KB");

        let stats_large = CacheStats {
            total_size_bytes: 1024 * 1024 * 1024,
            file_count: 1,
            cache_dir: temp_dir.path().to_path_buf(),
        };

        assert_eq!(stats_large.formatted_size(), "1.0 GB");
    }

    #[test]
    fn test_hash_filename() {
        let hash1 = DatasetCache::hash_filename("test.csv");
        let hash2 = DatasetCache::hash_filename("test.csv");
        let hash3 = DatasetCache::hash_filename("different.csv");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // Blake3 produces 32-byte hashes = 64 hex chars
    }

    #[test]
    fn test_platform_cache_dir() {
        let cache_dir = get_platform_cache_dir();
        // Should work on any platform
        assert!(cache_dir.is_some() || cfg!(target_os = "unknown"));

        if let Some(dir) = cache_dir {
            assert!(dir.to_string_lossy().contains("scirs2-datasets"));
        }
    }

    #[test]
    fn test_cache_size_management() {
        let temp_dir = TempDir::new().unwrap();
        let cache = DatasetCache::with_full_config(
            temp_dir.path().to_path_buf(),
            10,
            3600,
            2048, // 2KB limit
            false,
        );

        // Write multiple small files to approach the limit
        let small_data1 = vec![0u8; 400];
        cache.write_cached("small1.dat", &small_data1).unwrap();

        let small_data2 = vec![0u8; 400];
        cache.write_cached("small2.dat", &small_data2).unwrap();

        let small_data3 = vec![0u8; 400];
        cache.write_cached("small3.dat", &small_data3).unwrap();

        // Now write a file that should trigger cleanup
        let medium_data = vec![0u8; 800];
        cache.write_cached("medium.dat", &medium_data).unwrap();

        // The cache should have cleaned up to stay under the limit
        let stats = cache.get_detailed_stats().unwrap();
        assert!(stats.total_size_bytes <= cache.max_cache_size());

        // The most recent file should still be cached
        assert!(cache.is_cached("medium.dat"));
    }

    #[test]
    fn test_offline_mode() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = DatasetCache::new(temp_dir.path().to_path_buf());

        assert!(!cache.is_offline());
        cache.set_offline_mode(true);
        assert!(cache.is_offline());
    }

    #[test]
    fn test_detailed_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache = DatasetCache::new(temp_dir.path().to_path_buf());

        let test_data = vec![1, 2, 3, 4, 5];
        cache.write_cached("test.dat", &test_data).unwrap();

        let stats = cache.get_detailed_stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.total_size_bytes, test_data.len() as u64);
        assert_eq!(stats.files.len(), 1);
        assert_eq!(stats.files[0].name, "test.dat");
        assert_eq!(stats.files[0].size_bytes, test_data.len() as u64);
    }

    #[test]
    fn test_cache_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CacheManager::new(temp_dir.path().to_path_buf(), 10, 3600);

        let stats = manager.get_stats();
        assert_eq!(stats.file_count, 0);
        assert_eq!(stats.total_size_bytes, 0);

        assert_eq!(manager.cache_dir(), &temp_dir.path().to_path_buf());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }
}
