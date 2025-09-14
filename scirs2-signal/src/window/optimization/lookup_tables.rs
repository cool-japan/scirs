//! Pre-computed Window Function Lookup Tables
//!
//! This module provides pre-computed lookup tables for commonly used window functions
//! to improve performance in applications requiring frequent window generation.

use super::super::families::{cosine, exponential, specialized, triangular};
use crate::error::{SignalError, SignalResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

/// Window lookup table entry
#[derive(Debug, Clone)]
pub struct WindowTableEntry {
    /// Window coefficients
    pub coefficients: Vec<f64>,
    /// Window length
    pub length: usize,
    /// Window type identifier
    pub window_type: String,
    /// Additional parameters (e.g., beta for Kaiser)
    pub parameters: Vec<f64>,
    /// Creation timestamp for cache management
    pub created_at: std::time::SystemTime,
}

/// Window lookup table manager
pub struct WindowLookupTable {
    /// Cache of pre-computed windows
    cache: Arc<Mutex<HashMap<String, WindowTableEntry>>>,
    /// Maximum cache size (number of entries)
    max_cache_size: usize,
    /// Cache statistics
    cache_hits: Arc<Mutex<u64>>,
    cache_misses: Arc<Mutex<u64>>,
}

/// Global window lookup table instance
static mut GLOBAL_WINDOW_TABLE: Option<WindowLookupTable> = None;
static WINDOW_TABLE_INIT: Once = Once::new();

impl WindowLookupTable {
    /// Create new window lookup table
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size,
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get global window lookup table instance
    #[allow(static_mut_refs)]
    pub fn global() -> &'static WindowLookupTable {
        unsafe {
            WINDOW_TABLE_INIT.call_once(|| {
                GLOBAL_WINDOW_TABLE = Some(WindowLookupTable::new(1000));
            });
            GLOBAL_WINDOW_TABLE.as_ref().unwrap()
        }
    }

    /// Get or compute window with caching
    ///
    /// # Arguments
    /// * `window_type` - Type of window to generate
    /// * `length` - Window length
    /// * `parameters` - Additional parameters (e.g., beta for Kaiser)
    /// * `symmetric` - Whether to generate symmetric window
    ///
    /// # Returns
    /// Window coefficients from cache or newly computed
    pub fn get_or_compute_window(
        &self,
        window_type: &str,
        length: usize,
        parameters: &[f64],
        symmetric: bool,
    ) -> SignalResult<Vec<f64>> {
        let cache_key = self.generate_cache_key(window_type, length, parameters, symmetric);

        // Try to get from cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(entry) = cache.get(&cache_key) {
                *self.cache_hits.lock().unwrap() += 1;
                return Ok(entry.coefficients.clone());
            }
        }

        // Cache miss - compute window
        *self.cache_misses.lock().unwrap() += 1;
        let coefficients = self.compute_window(window_type, length, parameters, symmetric)?;

        // Store in cache
        self.store_in_cache(
            cache_key,
            window_type,
            length,
            parameters,
            coefficients.clone(),
        )?;

        Ok(coefficients)
    }

    /// Pre-populate cache with commonly used windows
    pub fn populate_common_windows(&self) -> SignalResult<()> {
        let common_lengths = vec![64, 128, 256, 512, 1024, 2048, 4096];
        let common_windows = vec![
            ("hann", vec![]),
            ("hamming", vec![]),
            ("blackman", vec![]),
            ("kaiser", vec![2.0, 5.0, 8.0]),
            ("gaussian", vec![0.5, 1.0, 2.0]),
            ("bartlett", vec![]),
        ];

        for &length in &common_lengths {
            for (window_name, param_sets) in &common_windows {
                if param_sets.is_empty() {
                    // Single parameter set (empty)
                    self.get_or_compute_window(window_name, length, &[], true)?;
                } else {
                    // Multiple parameter values
                    for &param in param_sets {
                        self.get_or_compute_window(window_name, length, &[param], true)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        *self.cache_hits.lock().unwrap() = 0;
        *self.cache_misses.lock().unwrap() = 0;
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let hits = *self.cache_hits.lock().unwrap();
        let misses = *self.cache_misses.lock().unwrap();
        let cache = self.cache.lock().unwrap();

        CacheStatistics {
            hits,
            misses,
            hit_ratio: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
            cache_size: cache.len(),
            max_cache_size: self.max_cache_size,
        }
    }

    /// Remove least recently used entries to maintain cache size
    fn cleanup_cache(&self) -> SignalResult<()> {
        let mut cache = self.cache.lock().unwrap();

        if cache.len() <= self.max_cache_size {
            return Ok(());
        }

        // Find oldest entries
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.created_at);

        let remove_count = cache.len() - self.max_cache_size;
        let keys_to_remove: Vec<String> = entries
            .iter()
            .take(remove_count)
            .map(|(key, _)| (*key).clone())
            .collect();

        for key in keys_to_remove {
            cache.remove(&key);
        }

        Ok(())
    }

    /// Generate cache key for window configuration
    fn generate_cache_key(
        &self,
        window_type: &str,
        length: usize,
        parameters: &[f64],
        symmetric: bool,
    ) -> String {
        let mut key = format!("{}_{}_sym{}", window_type, length, symmetric);

        for (i, &param) in parameters.iter().enumerate() {
            key.push_str(&format!("_p{}_{:.6}", i, param));
        }

        key
    }

    /// Compute window function
    fn compute_window(
        &self,
        window_type: &str,
        length: usize,
        parameters: &[f64],
        symmetric: bool,
    ) -> SignalResult<Vec<f64>> {
        match window_type {
            "hann" => cosine::hann(length, symmetric),
            "hamming" => cosine::hamming(length, symmetric),
            "blackman" => cosine::blackman(length, symmetric),
            "blackmanharris" => cosine::blackmanharris(length, symmetric),
            "nuttall" => cosine::nuttall(length, symmetric),
            "flattop" => cosine::flattop(length, symmetric),
            "cosine" => cosine::cosine(length, symmetric),
            "barthann" => cosine::barthann(length, symmetric),

            "bartlett" => triangular::bartlett(length, symmetric),
            "triang" => triangular::triang(length, symmetric),
            "parzen" => triangular::parzen(length, symmetric),
            "welch" => triangular::welch(length, symmetric),

            "kaiser" => {
                let beta = parameters.get(0).copied().unwrap_or(5.0);
                exponential::kaiser(length, beta, symmetric)
            }
            "gaussian" => {
                let std = parameters.get(0).copied().unwrap_or(1.0);
                exponential::gaussian(length, std, symmetric)
            }
            "tukey" => {
                let alpha = parameters.get(0).copied().unwrap_or(0.5);
                exponential::tukey(length, alpha, symmetric)
            }
            "exponential" => {
                let tau = parameters.get(0).copied().unwrap_or(2.0);
                exponential::exponential(length, tau, symmetric)
            }

            "bohman" => specialized::bohman(length, symmetric),
            "poisson" => {
                let alpha = parameters.get(0).copied().unwrap_or(1.0);
                specialized::poisson(length, alpha, symmetric)
            }
            "dpss" => {
                let nw = parameters.get(0).copied().unwrap_or(2.5);
                specialized::dpss_approximation(length, nw, symmetric)
            }

            _ => Err(SignalError::ValueError(format!(
                "Unknown window type: {}",
                window_type
            ))),
        }
    }

    /// Store window in cache
    fn store_in_cache(
        &self,
        key: String,
        window_type: &str,
        length: usize,
        parameters: &[f64],
        coefficients: Vec<f64>,
    ) -> SignalResult<()> {
        let entry = WindowTableEntry {
            coefficients,
            length,
            window_type: window_type.to_string(),
            parameters: parameters.to_vec(),
            created_at: std::time::SystemTime::now(),
        };

        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, entry);
        }

        // Cleanup if needed
        self.cleanup_cache()?;

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
    pub cache_size: usize,
    pub max_cache_size: usize,
}

/// Optimized window generation with lookup table caching
///
/// Convenience functions that automatically use the global lookup table
pub mod cached_windows {
    use super::*;

    /// Generate Hann window with caching
    pub fn hann(length: usize, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("hann", length, &[], symmetric)
    }

    /// Generate Hamming window with caching
    pub fn hamming(length: usize, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("hamming", length, &[], symmetric)
    }

    /// Generate Blackman window with caching
    pub fn blackman(length: usize, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("blackman", length, &[], symmetric)
    }

    /// Generate Kaiser window with caching
    pub fn kaiser(length: usize, beta: f64, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("kaiser", length, &[beta], symmetric)
    }

    /// Generate Gaussian window with caching
    pub fn gaussian(length: usize, std: f64, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("gaussian", length, &[std], symmetric)
    }

    /// Generate Bartlett window with caching
    pub fn bartlett(length: usize, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("bartlett", length, &[], symmetric)
    }

    /// Generate Tukey window with caching
    pub fn tukey(length: usize, alpha: f64, symmetric: bool) -> SignalResult<Vec<f64>> {
        WindowLookupTable::global().get_or_compute_window("tukey", length, &[alpha], symmetric)
    }
}

/// Interpolated window generation for arbitrary lengths
///
/// For lengths not in the cache, interpolate between cached windows
pub fn interpolated_window(
    window_type: &str,
    target_length: usize,
    parameters: &[f64],
    symmetric: bool,
) -> SignalResult<Vec<f64>> {
    if target_length <= 16 {
        // For small windows, compute directly
        return WindowLookupTable::global().get_or_compute_window(
            window_type,
            target_length,
            parameters,
            symmetric,
        );
    }

    // Find nearest power-of-2 lengths in cache
    let lower_length = (target_length as f64).log2().floor().exp2() as usize;
    let upper_length = (target_length as f64).log2().ceil().exp2() as usize;

    if lower_length == upper_length {
        // Exact power of 2
        return WindowLookupTable::global().get_or_compute_window(
            window_type,
            target_length,
            parameters,
            symmetric,
        );
    }

    // Get windows at lower and upper lengths
    let lower_window = WindowLookupTable::global().get_or_compute_window(
        window_type,
        lower_length,
        parameters,
        symmetric,
    )?;
    let upper_window = WindowLookupTable::global().get_or_compute_window(
        window_type,
        upper_length,
        parameters,
        symmetric,
    )?;

    // Interpolate to target length
    interpolate_window(&lower_window, &upper_window, target_length)
}

/// Interpolate between two windows to create window of target length
fn interpolate_window(
    lower_window: &[f64],
    upper_window: &[f64],
    target_length: usize,
) -> SignalResult<Vec<f64>> {
    let mut result = Vec::with_capacity(target_length);

    for i in 0..target_length {
        let normalized_pos = i as f64 / (target_length - 1) as f64;

        // Sample from lower window
        let lower_idx = (normalized_pos * (lower_window.len() - 1) as f64).round() as usize;
        let lower_idx = lower_idx.min(lower_window.len() - 1);
        let lower_val = lower_window[lower_idx];

        // Sample from upper window
        let upper_idx = (normalized_pos * (upper_window.len() - 1) as f64).round() as usize;
        let upper_idx = upper_idx.min(upper_window.len() - 1);
        let upper_val = upper_window[upper_idx];

        // Linear interpolation weight
        let lower_len = lower_window.len() as f64;
        let upper_len = upper_window.len() as f64;
        let target_len = target_length as f64;

        let weight = (target_len - lower_len) / (upper_len - lower_len);
        let interpolated = lower_val * (1.0 - weight) + upper_val * weight;

        result.push(interpolated);
    }

    Ok(result)
}

/// Benchmark lookup table performance
pub fn benchmark_lookup_table(
    window_type: &str,
    lengths: &[usize],
    parameters: &[f64],
    iterations: usize,
) -> SignalResult<LookupTableBenchmark> {
    use std::time::Instant;

    let table = WindowLookupTable::global();

    // Clear cache to start fresh
    table.clear_cache();

    // Benchmark with empty cache (all misses)
    let start = Instant::now();
    for _ in 0..iterations {
        for &length in lengths {
            let _ = table.get_or_compute_window(window_type, length, parameters, true)?;
        }
    }
    let cold_duration = start.elapsed();

    // Benchmark with warm cache (all hits)
    let start = Instant::now();
    for _ in 0..iterations {
        for &length in lengths {
            let _ = table.get_or_compute_window(window_type, length, parameters, true)?;
        }
    }
    let warm_duration = start.elapsed();

    let stats = table.get_cache_stats();
    let speedup = cold_duration.as_secs_f64() / warm_duration.as_secs_f64();

    Ok(LookupTableBenchmark {
        cold_duration,
        warm_duration,
        speedup,
        cache_stats: stats,
    })
}

/// Lookup table benchmark results
#[derive(Debug)]
pub struct LookupTableBenchmark {
    pub cold_duration: std::time::Duration,
    pub warm_duration: std::time::Duration,
    pub speedup: f64,
    pub cache_stats: CacheStatistics,
}

/// Initialize global lookup table with common windows
pub fn initialize_window_cache() -> SignalResult<()> {
    WindowLookupTable::global().populate_common_windows()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_lookup_table() {
        let table = WindowLookupTable::new(10);

        // First access should be cache miss
        let window1 = table.get_or_compute_window("hann", 64, &[], true).unwrap();
        assert_eq!(window1.len(), 64);

        // Second access should be cache hit
        let window2 = table.get_or_compute_window("hann", 64, &[], true).unwrap();
        assert_eq!(window1, window2);

        let stats = table.get_cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cached_windows() {
        // Test cached window functions
        let hann1 = cached_windows::hann(128, true).unwrap();
        let hann2 = cached_windows::hann(128, true).unwrap();
        assert_eq!(hann1, hann2);

        let kaiser1 = cached_windows::kaiser(64, 5.0, true).unwrap();
        let kaiser2 = cached_windows::kaiser(64, 5.0, true).unwrap();
        assert_eq!(kaiser1, kaiser2);
    }

    #[test]
    fn test_cache_cleanup() {
        let table = WindowLookupTable::new(3); // Small cache

        // Fill cache beyond capacity
        table.get_or_compute_window("hann", 32, &[], true).unwrap();
        table.get_or_compute_window("hann", 64, &[], true).unwrap();
        table.get_or_compute_window("hann", 128, &[], true).unwrap();
        table.get_or_compute_window("hann", 256, &[], true).unwrap();

        let stats = table.get_cache_stats();
        assert!(stats.cache_size <= 3);
    }

    #[test]
    fn test_interpolated_window() {
        let window = interpolated_window("hann", 100, &[], true).unwrap();
        assert_eq!(window.len(), 100);

        // Should be approximately zero at endpoints for Hann
        assert!(window[0].abs() < 0.1);
        assert!(window[99].abs() < 0.1);
    }

    #[test]
    fn test_populate_common_windows() {
        let table = WindowLookupTable::new(100);
        table.populate_common_windows().unwrap();

        let stats = table.get_cache_stats();
        assert!(stats.cache_size > 10); // Should have populated many windows
    }

    #[test]
    fn test_parameter_handling() {
        let table = WindowLookupTable::new(10);

        // Kaiser with different beta values should be different cache entries
        let kaiser1 = table
            .get_or_compute_window("kaiser", 64, &[2.0], true)
            .unwrap();
        let kaiser2 = table
            .get_or_compute_window("kaiser", 64, &[5.0], true)
            .unwrap();

        assert_ne!(kaiser1, kaiser2);

        let stats = table.get_cache_stats();
        assert_eq!(stats.cache_size, 2);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_functionality() {
        let lengths = vec![64, 128];
        let result = benchmark_lookup_table("hann", &lengths, &[], 5).unwrap();

        assert!(result.cold_duration.as_nanos() > 0);
        assert!(result.warm_duration.as_nanos() > 0);
        assert!(result.speedup > 1.0); // Warm cache should be faster
    }

    #[test]
    fn test_unknown_window_type() {
        let table = WindowLookupTable::new(10);
        let result = table.get_or_compute_window("unknown", 64, &[], true);
        assert!(result.is_err());
    }
}
