//! FFT Plan Caching Module
//!
//! This module provides a caching mechanism for FFT plans to improve performance
//! when performing repeated transforms of the same size.

use rustfft::FftPlanner;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Cache key for storing FFT plans
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlanKey {
    size: usize,
    forward: bool,
    // Future: Add backend identifier when we support multiple backends
}

/// Cached FFT plan with metadata
#[derive(Clone)]
struct CachedPlan {
    plan: Arc<dyn rustfft::Fft<f64>>,
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
        *self.enabled.lock().unwrap() = enabled;
    }

    /// Check if the cache is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap()
    }

    /// Clear all cached plans
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get statistics about cache usage
    pub fn get_stats(&self) -> CacheStats {
        let hit_count = *self.hit_count.lock().unwrap();
        let miss_count = *self.miss_count.lock().unwrap();
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

    /// Get or create an FFT plan for the given size and direction
    pub fn get_or_create_plan(
        &self,
        size: usize,
        forward: bool,
        planner: &mut FftPlanner<f64>,
    ) -> Arc<dyn rustfft::Fft<f64>> {
        if !*self.enabled.lock().unwrap() {
            return if forward {
                planner.plan_fft_forward(size)
            } else {
                planner.plan_fft_inverse(size)
            };
        }

        let key = PlanKey { size, forward };

        // Try to get from cache first
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(cached) = cache.get_mut(&key) {
                // Check if the plan is still valid (not too old)
                if cached.last_used.elapsed() <= self.max_age {
                    cached.last_used = Instant::now();
                    cached.usage_count += 1;
                    *self.hit_count.lock().unwrap() += 1;
                    return cached.plan.clone();
                } else {
                    // Remove stale entry
                    cache.remove(&key);
                }
            }
        }

        // Cache miss - create new plan
        *self.miss_count.lock().unwrap() += 1;

        let plan: Arc<dyn rustfft::Fft<f64>> = if forward {
            planner.plan_fft_forward(size)
        } else {
            planner.plan_fft_inverse(size)
        };

        // Store in cache if enabled
        if let Ok(mut cache) = self.cache.lock() {
            // Clean up old entries if we're at capacity
            if cache.len() >= self.max_entries {
                self.evict_old_entries(&mut cache);
            }

            cache.insert(
                key,
                CachedPlan {
                    plan: plan.clone(),
                    last_used: Instant::now(),
                    usage_count: 1,
                },
            );
        }

        plan
    }

    /// Evict old entries from the cache (LRU-style)
    fn evict_old_entries(&self, cache: &mut HashMap<PlanKey, CachedPlan>) {
        // Remove entries older than max_age
        cache.retain(|_, v| v.last_used.elapsed() <= self.max_age);

        // If still over capacity, remove least recently used
        while cache.len() >= self.max_entries {
            if let Some((key_to_remove, _)) = cache
                .iter()
                .min_by_key(|(_, v)| (v.last_used, v.usage_count))
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                cache.remove(&key_to_remove);
            } else {
                break;
            }
        }
    }

    /// Pre-populate cache with common sizes
    pub fn precompute_common_sizes(&self, sizes: &[usize], planner: &mut FftPlanner<f64>) {
        for &size in sizes {
            // Pre-compute both forward and inverse plans
            self.get_or_create_plan(size, true, planner);
            self.get_or_create_plan(size, false, planner);
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
pub fn get_global_cache() -> &'static PlanCache {
    GLOBAL_PLAN_CACHE.get_or_init(PlanCache::new)
}

/// Initialize the global plan cache with custom settings
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
        let mut planner = FftPlanner::new();

        // Get the same plan twice
        let _plan1 = cache.get_or_create_plan(128, true, &mut planner);
        let _plan2 = cache.get_or_create_plan(128, true, &mut planner);

        // Second request should be a cache hit
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PlanCache::with_config(2, Duration::from_secs(3600));
        let mut planner = FftPlanner::new();

        // Fill cache with 2 entries
        cache.get_or_create_plan(64, true, &mut planner);
        cache.get_or_create_plan(128, true, &mut planner);

        // Add a third entry, which should evict the oldest
        cache.get_or_create_plan(256, true, &mut planner);

        let stats = cache.get_stats();
        assert_eq!(stats.size, 2);
    }

    #[test]
    fn test_cache_disabled() {
        let cache = PlanCache::new();
        cache.set_enabled(false);

        let mut planner = FftPlanner::new();

        // Get the same plan twice with cache disabled
        cache.get_or_create_plan(128, true, &mut planner);
        cache.get_or_create_plan(128, true, &mut planner);

        // Both should be misses
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 0); // No tracking when disabled
    }
}
