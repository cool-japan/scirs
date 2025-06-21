//! Cache and memoization utilities for improved performance
//!
//! This module provides caching and memoization utilities to improve performance
//! by avoiding redundant computations.

use cached::{proc_macro::cached, Cached, SizedCache};
use std::hash::Hash;
use std::time::{Duration, Instant};

/// Cache configuration for the library
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Default cache size (number of items)
    pub default_size: usize,
    /// Default time-to-live for cached items (seconds)
    pub default_ttl: u64,
    /// Whether to enable caching by default
    pub enable_caching: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_size: 1024,
            default_ttl: 3600, // 1 hour
            enable_caching: true,
        }
    }
}

/// A sized cache with time-to-live (TTL) functionality
pub struct TTLSizedCache<K, V> {
    /// Internal cache
    cache: SizedCache<K, (V, Instant)>,
    /// Time-to-live for cache entries
    ttl: Duration,
}

impl<K, V> TTLSizedCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new TTL cache with specified size and TTL
    #[must_use]
    pub fn new(size: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: SizedCache::with_size(size),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Insert a key-value pair into the cache
    pub fn insert(&mut self, key: K, value: V) {
        let now = Instant::now();
        self.cache.cache_set(key, (value, now));
    }

    /// Get a value from the cache if it exists and hasn't expired
    #[must_use]
    pub fn get(&mut self, key: &K) -> Option<V> {
        match self.cache.cache_get(key) {
            Some((value, timestamp)) if timestamp.elapsed() < self.ttl => Some(value.clone()),
            Some(_) => {
                // Value has expired, remove it from cache
                self.cache.cache_remove(key);
                None
            }
            None => None,
        }
    }

    /// Remove a key-value pair from the cache
    pub fn remove(&mut self, key: &K) {
        self.cache.cache_remove(key);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.cache_clear();
    }

    /// Get the number of items in the cache
    #[must_use]
    pub fn len(&self) -> usize {
        (self.cache.cache_misses().unwrap_or(0) + self.cache.cache_hits().unwrap_or(0)) as usize
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A thread-safe cache builder
///
/// This struct provides a fluent interface for configuring and building
/// various types of caches.
pub struct CacheBuilder {
    /// Cache size
    size: Option<usize>,
    /// Time-to-live in seconds
    ttl: Option<u64>,
    /// Whether to make the cache thread-safe
    thread_safe: bool,
}

impl Default for CacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheBuilder {
    /// Create a new cache builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            size: None,
            ttl: None,
            thread_safe: false,
        }
    }

    /// Set the cache size
    #[must_use]
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the time-to-live in seconds
    #[must_use]
    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Make the cache thread-safe
    #[must_use]
    pub fn thread_safe(mut self) -> Self {
        self.thread_safe = true;
        self
    }

    /// Build a sized cache
    #[must_use]
    pub fn build_sized_cache<K, V>(self) -> TTLSizedCache<K, V>
    where
        K: Hash + Eq + Clone,
        V: Clone,
    {
        let config = CacheConfig::default();
        let size = self.size.unwrap_or(config.default_size);
        let ttl = self.ttl.unwrap_or(config.default_ttl);

        TTLSizedCache::new(size, ttl)
    }
}

/// Example of how to use the cached attribute
///
/// ```rust
/// use cached::proc_macro::cached;
///
/// #[cached(size = 100)]
/// pub fn expensive_calculation(x: u64) -> u64 {
///     // Expensive computation here
///     x * x
/// }
/// ```
/// Example of using TTL cache for memoization
///
/// This example shows how to use the TTL cache for memoizing expensive operations.
///
/// ```rust
/// use scirs2_core::cache::TTLSizedCache;
/// use std::time::Duration;
///
/// // Create a TTL cache
/// let mut cache = TTLSizedCache::<String, String>::new(100, 60);
///
/// // Cache a value
/// cache.insert("key".to_string(), "value".to_string());
///
/// // Retrieve a value
/// let value = cache.get(&"key".to_string());
/// ```
pub fn memoize_example() {
    // Create a TTL cache
    let mut cache = TTLSizedCache::<String, String>::new(100, 60);

    // Cache a value
    cache.insert("key".to_string(), "value".to_string());

    // Retrieve a value
    let value = cache.get(&"key".to_string());

    println!("Value from cache: {:?}", value);
}

/// Compute Fibonacci numbers with memoization
///
/// This function demonstrates the use of memoization for computing
/// Fibonacci numbers efficiently.
///
/// # Arguments
///
/// * `n` - The Fibonacci number to compute
///
/// # Returns
///
/// * The nth Fibonacci number
#[cached]
#[must_use]
pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_ttl_sized_cache() {
        let mut cache = TTLSizedCache::<i32, &str>::new(5, 1);

        // Test insertion and retrieval
        cache.insert(1, "one");
        cache.insert(2, "two");

        assert_eq!(cache.get(&1), Some("one"));
        assert_eq!(cache.get(&2), Some("two"));
        assert_eq!(cache.get(&3), None);

        // Test TTL expiration
        thread::sleep(Duration::from_secs(2));

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);

        // Test size limit
        for i in 0..10 {
            cache.insert(i, "value");
        }

        // Only the last 5 should be in the cache due to size limit
        for i in 0..5 {
            assert_eq!(cache.get(&i), None);
        }

        for i in 5..10 {
            assert_eq!(cache.get(&i), Some("value"));
        }
    }

    #[test]
    fn test_cache_builder() {
        let cache = CacheBuilder::new()
            .with_size(10)
            .with_ttl(60)
            .build_sized_cache::<String, i32>();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_fibonacci() {
        // Compute fibonacci numbers with memoization
        let fib10 = fibonacci(10);
        let fib20 = fibonacci(20);

        assert_eq!(fib10, 55);
        assert_eq!(fib20, 6765);

        // The second call should be much faster due to memoization
        let start = Instant::now();
        let fib20_again = fibonacci(20);
        let duration = start.elapsed();

        assert_eq!(fib20_again, 6765);

        // The second call should be very fast (less than 1ms)
        assert!(duration.as_millis() < 10);
    }
}
