//! FFT Plan Caching Module
//!
//! This module provides a caching mechanism for FFT plans to improve performance
//! when performing repeated transforms of the same size.
//! Uses OxiFFT as the backend (COOLJAPAN Pure Rust policy).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Cache key for storing FFT plans
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlanKey {
    size: usize,
    forward: bool,
}

/// Cached FFT plan metadata (OxiFFT backend)
///
/// With OxiFFT, plans are managed globally via oxifft_plan_cache.
/// This struct just tracks metadata for statistics.
#[derive(Clone)]
struct CachedPlan {
    size: usize,
    forward: bool,
    last_used: Instant,
    usage_count: usize,
}

/// FFT Plan Cache with configurable size limits and TTL
pub struct PlanCache {
    cache: Arc<Mutex<HashMap<PlanKey, CachedPlan>>>,
    max_entries: usize,
    max_age: Duration,
    enabled: Arc<Mutex<bool>>,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

impl PlanCache {
    /// Create a new plan cache with default settings
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_entries: 128,
            max_age: Duration::from_secs(3600), // 1 hour
            enabled: Arc::new(Mutex::new(true)),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new plan cache with custom settings
    pub fn with_config(max_entries: usize, max_age: Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_entries,
            max_age,
            enabled: Arc::new(Mutex::new(true)),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Enable or disable the cache
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.lock().expect("Operation failed") = enabled;
    }

    /// Check if the cache is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().expect("Operation failed")
    }

    /// Clear all cached plans
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get statistics about cache usage
    pub fn get_stats(&self) -> CacheStats {
        let hit_count = *self.hit_count.lock().expect("Operation failed");
        let miss_count = *self.miss_count.lock().expect("Operation failed");
        let total_requests = hit_count + miss_count;
        let hit_rate = if total_requests > 0 {
            hit_count as f64 / total_requests as f64
        } else {
            0.0
        };

        let size = self.cache.lock().map(|c| c.len()).unwrap_or(0);

        CacheStats {
            hit_count,
            miss_count,
            hit_rate,
            size,
            max_size: self.max_entries,
        }
    }

    /// Track FFT plan usage in the cache (OxiFFT backend)
    ///
    /// Note: OxiFFT plans are managed globally via oxifft_plan_cache.
    /// This method provides tracking and statistics for plan usage.
    pub fn track_plan_usage(&self, size: usize, forward: bool) {
        if !*self.enabled.lock().expect("Operation failed") {
            return;
        }

        let key = PlanKey { size, forward };

        // Try to get from cache first
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(cached) = cache.get_mut(&key) {
                // Check if the plan is still valid (not too old)
                if cached.last_used.elapsed() <= self.max_age {
                    cached.last_used = Instant::now();
                    cached.usage_count += 1;
                    *self.hit_count.lock().expect("Operation failed") += 1;
                    return;
                } else {
                    // Remove stale entry
                    cache.remove(&key);
                }
            }
        }

        // Cache miss - track new plan
        *self.miss_count.lock().expect("Operation failed") += 1;

        // Store metadata in cache if enabled
        if let Ok(mut cache) = self.cache.lock() {
            // Clean up old entries if we're at capacity
            if cache.len() >= self.max_entries {
                self.evict_old_entries(&mut cache);
            }

            cache.insert(
                key,
                CachedPlan {
                    size,
                    forward,
                    last_used: Instant::now(),
                    usage_count: 1,
                },
            );
        }
    }

    /// Evict old entries from the cache (LRU-style)
    fn evict_old_entries(&self, cache: &mut HashMap<PlanKey, CachedPlan>) {
        // Remove entries older than max_age
        cache.retain(|_, v| v.last_used.elapsed() <= self.max_age);

        // If still over capacity, remove least recently used
        while cache.len() >= self.max_entries {
            if let Some((key_to_remove_, _)) = cache
                .iter()
                .min_by_key(|(_, v)| (v.last_used, v.usage_count))
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                cache.remove(&key_to_remove_);
            } else {
                break;
            }
        }
    }

    /// Pre-populate cache with common sizes (OxiFFT backend)
    ///
    /// Note: With OxiFFT, plans are created lazily and cached globally.
    /// This method tracks sizes for statistics purposes.
    pub fn precompute_common_sizes(&self, sizes: &[usize]) {
        for &size in sizes {
            // Track both forward and inverse plans
            self.track_plan_usage(size, true);
            self.track_plan_usage(size, false);
        }
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate: f64,
    pub size: usize,
    pub max_size: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache Stats: {} hits, {} misses ({:.1}% hit rate), {}/{} entries",
            self.hit_count,
            self.miss_count,
            self.hit_rate * 100.0,
            self.size,
            self.max_size
        )
    }
}

/// Global plan cache instance
static GLOBAL_PLAN_CACHE: std::sync::OnceLock<PlanCache> = std::sync::OnceLock::new();

/// Get the global plan cache instance
#[allow(dead_code)]
pub fn get_global_cache() -> &'static PlanCache {
    GLOBAL_PLAN_CACHE.get_or_init(PlanCache::new)
}

/// Initialize the global plan cache with custom settings
#[allow(dead_code)]
pub fn init_global_cache(max_entries: usize, max_age: Duration) -> Result<(), &'static str> {
    GLOBAL_PLAN_CACHE
        .set(PlanCache::with_config(max_entries, max_age))
        .map_err(|_| "Global plan cache already initialized")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_cache_basic() {
        let cache = PlanCache::new();

        // Track the same plan twice
        cache.track_plan_usage(128, true);
        cache.track_plan_usage(128, true);

        // Second request should be a cache hit
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PlanCache::with_config(2, Duration::from_secs(3600));

        // Fill cache with 2 entries
        cache.track_plan_usage(64, true);
        cache.track_plan_usage(128, true);

        // Add a third entry, which should evict the oldest
        cache.track_plan_usage(256, true);

        let stats = cache.get_stats();
        assert_eq!(stats.size, 2);
    }

    #[test]
    fn test_cache_disabled() {
        let cache = PlanCache::new();
        cache.set_enabled(false);

        // Track the same plan twice with cache disabled
        cache.track_plan_usage(128, true);
        cache.track_plan_usage(128, true);

        // Both should be misses
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 0); // No tracking when disabled
    }
}
