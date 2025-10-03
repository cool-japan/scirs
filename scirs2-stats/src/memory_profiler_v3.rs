//! Advanced memory profiling and optimization for statistical operations
//!
//! This module provides comprehensive memory profiling tools and adaptive
//! memory management strategies for optimal performance.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive memory profiler for statistical operations
pub struct MemoryProfiler {
    allocations: Arc<Mutex<HashMap<String, AllocationStats>>>,
    peak_memory: Arc<Mutex<usize>>,
    current_memory: Arc<Mutex<usize>>,
    enabled: bool,
}

/// Statistics for memory allocations
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub total_allocations: usize,
    pub total_bytes: usize,
    pub peak_bytes: usize,
    pub averagesize: f64,
    pub allocation_times: Vec<Duration>,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            peak_memory: Arc::new(Mutex::new(0)),
            current_memory: Arc::new(Mutex::new(0)),
            enabled: true,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, category: &str, size: usize, duration: Duration) {
        if !self.enabled {
            return;
        }

        let mut allocations = self.allocations.lock().unwrap();
        let stats = allocations.entry(category.to_string()).or_default();

        stats.total_allocations += 1;
        stats.total_bytes += size;
        stats.peak_bytes = stats.peak_bytes.max(size);
        stats.averagesize = stats.total_bytes as f64 / stats.total_allocations as f64;
        stats.allocation_times.push(duration);

        // Update global memory tracking
        let mut current = self.current_memory.lock().unwrap();
        *current += size;

        let mut peak = self.peak_memory.lock().unwrap();
        *peak = (*peak).max(*current);
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        if !self.enabled {
            return;
        }

        let mut current = self.current_memory.lock().unwrap();
        *current = current.saturating_sub(size);
    }

    /// Get memory usage report
    pub fn get_report(&self) -> MemoryReport {
        let allocations = self.allocations.lock().unwrap().clone();
        let peak_memory = *self.peak_memory.lock().unwrap();
        let current_memory = *self.current_memory.lock().unwrap();

        let recommendations = self.generate_recommendations(&allocations);
        MemoryReport {
            allocations,
            peak_memory,
            current_memory,
            recommendations,
        }
    }

    /// Generate memory optimization recommendations
    fn generate_recommendations(
        &self,
        allocations: &HashMap<String, AllocationStats>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (category, stats) in allocations {
            // Check for frequent small allocations
            if stats.total_allocations > 1000 && stats.averagesize < 1024.0 {
                recommendations.push(format!(
                    "Consider memory pooling for '{}' category (many small allocations: {} allocations, avg size: {:.1} bytes)",
                    category, stats.total_allocations, stats.averagesize
                ));
            }

            // Check for large allocations
            if stats.peak_bytes > 10 * 1024 * 1024 {
                // 10MB
                recommendations.push(format!(
                    "Consider streaming processing for '{}' category (large allocation: {:.1} MB)",
                    category,
                    stats.peak_bytes as f64 / 1024.0 / 1024.0
                ));
            }

            // Check for slow allocations
            if let Some(&max_time) = stats.allocation_times.iter().max() {
                if max_time > Duration::from_millis(10) {
                    recommendations.push(format!(
                        "Consider pre-allocation for '{}' category (slow allocation: {:?})",
                        category, max_time
                    ));
                }
            }
        }

        recommendations
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Reset all profiling data
    pub fn reset(&self) {
        self.allocations.lock().unwrap().clear();
        *self.peak_memory.lock().unwrap() = 0;
        *self.current_memory.lock().unwrap() = 0;
    }
}

/// Memory usage report
#[derive(Debug)]
pub struct MemoryReport {
    pub allocations: HashMap<String, AllocationStats>,
    pub peak_memory: usize,
    pub current_memory: usize,
    pub recommendations: Vec<String>,
}

impl MemoryReport {
    /// Print formatted report
    pub fn print_report(&self) {
        println!("=== Memory Usage Report ===");
        println!(
            "Peak Memory Usage: {:.2} MB",
            self.peak_memory as f64 / 1024.0 / 1024.0
        );
        println!(
            "Current Memory Usage: {:.2} MB",
            self.current_memory as f64 / 1024.0 / 1024.0
        );
        println!();

        println!("Allocation Statistics by Category:");
        for (category, stats) in &self.allocations {
            println!("  {}:", category);
            println!("    Total Allocations: {}", stats.total_allocations);
            println!(
                "    Total Bytes: {:.2} MB",
                stats.total_bytes as f64 / 1024.0 / 1024.0
            );
            println!(
                "    Peak Allocation: {:.2} KB",
                stats.peak_bytes as f64 / 1024.0
            );
            println!("    Average Size: {:.1} bytes", stats.averagesize);

            if !stats.allocation_times.is_empty() {
                let avg_time = stats.allocation_times.iter().sum::<Duration>().as_micros() as f64
                    / stats.allocation_times.len() as f64;
                println!("    Average Allocation Time: {:.1} µs", avg_time);
            }
            println!();
        }

        if !self.recommendations.is_empty() {
            println!("Optimization Recommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }
    }
}

/// Memory-efficient cache for statistical computations
pub struct StatisticsCache<F> {
    cache: HashMap<String, CachedResult<F>>,
    max_entries: usize,
    max_memory: usize,
    current_memory: usize,
    profiler: Option<Arc<MemoryProfiler>>,
}

#[derive(Clone)]
struct CachedResult<F> {
    value: F,
    timestamp: Instant,
    memorysize: usize,
    access_count: usize,
}

impl<F: Float + Clone + std::fmt::Display> StatisticsCache<F> {
    pub fn new(_max_entries: usize, maxmemory: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries: _max_entries,
            max_memory: maxmemory,
            current_memory: 0,
            profiler: None,
        }
    }

    pub fn with_profiler(mut self, profiler: Arc<MemoryProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }

    /// Cache a computed result
    pub fn put(&mut self, key: String, value: F) {
        let memorysize = std::mem::size_of::<F>() + key.len();

        // Check if we need to evict entries
        self.maybe_evict(memorysize);

        let cached_result = CachedResult {
            value,
            timestamp: Instant::now(),
            memorysize,
            access_count: 0,
        };

        if let Some(old_result) = self.cache.insert(key.clone(), cached_result) {
            self.current_memory -= old_result.memorysize;
        }

        self.current_memory += memorysize;

        if let Some(profiler) = &self.profiler {
            profiler.record_allocation("statistics_cache", memorysize, Duration::from_nanos(0));
        }
    }

    /// Retrieve a cached result
    pub fn get(&mut self, key: &str) -> Option<F> {
        if let Some(entry) = self.cache.get_mut(key) {
            entry.access_count += 1;
            Some(entry.value)
        } else {
            None
        }
    }

    /// Evict entries to make room for new ones
    fn maybe_evict(&mut self, neededsize: usize) {
        // Check memory limit
        while self.current_memory + neededsize > self.max_memory && !self.cache.is_empty() {
            self.evict_lru();
        }

        // Check entry count limit
        while self.cache.len() >= self.max_entries && !self.cache.is_empty() {
            self.evict_lru();
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((key_to_remove, entry_to_remove)) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| (entry.access_count, entry.timestamp))
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.cache.remove(&key_to_remove);
            self.current_memory -= entry_to_remove.memorysize;

            if let Some(profiler) = &self.profiler {
                profiler.record_deallocation(entry_to_remove.memorysize);
            }
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            memory_usage: self.current_memory,
            hit_rate: self.calculate_hit_rate(),
        }
    }

    fn calculate_hit_rate(&self) -> f64 {
        let total_accesses: usize = self.cache.values().map(|entry| entry.access_count).sum();
        if total_accesses == 0 {
            0.0
        } else {
            total_accesses as f64 / (total_accesses + self.cache.len()) as f64
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        if let Some(profiler) = &self.profiler {
            for entry in self.cache.values() {
                profiler.record_deallocation(entry.memorysize);
            }
        }

        self.cache.clear();
        self.current_memory = 0;
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub entries: usize,
    pub memory_usage: usize,
    pub hit_rate: f64,
}

/// Adaptive memory manager that adjusts algorithms based on available memory
pub struct AdaptiveMemoryManager {
    memory_threshold_low: usize,
    memory_threshold_high: usize,
    profiler: Arc<MemoryProfiler>,
}

impl AdaptiveMemoryManager {
    pub fn new(profiler: Arc<MemoryProfiler>) -> Self {
        Self {
            memory_threshold_low: 100 * 1024 * 1024,   // 100MB
            memory_threshold_high: 1024 * 1024 * 1024, // 1GB
            profiler,
        }
    }

    /// Choose optimal algorithm based on current memory usage
    pub fn choose_algorithm(&self, datasize: usize) -> AlgorithmChoice {
        let current_memory = *self.profiler.current_memory.lock().unwrap();

        if current_memory > self.memory_threshold_high {
            // High memory usage - use most memory-efficient algorithms
            if datasize > 1_000_000 {
                AlgorithmChoice::Streaming
            } else {
                AlgorithmChoice::InPlace
            }
        } else if current_memory > self.memory_threshold_low {
            // Medium memory usage - balance speed and memory
            if datasize > 100_000 {
                AlgorithmChoice::Chunked
            } else {
                AlgorithmChoice::Standard
            }
        } else {
            // Low memory usage - prioritize speed
            if datasize > 10_000 {
                AlgorithmChoice::Parallel
            } else {
                AlgorithmChoice::Standard
            }
        }
    }

    /// Suggest chunk size based on available memory
    pub fn suggest_chunksize(&self, datasize: usize, elementsize: usize) -> usize {
        let current_memory = *self.profiler.current_memory.lock().unwrap();
        let available_memory = self.memory_threshold_high.saturating_sub(current_memory);

        // Use at most 10% of available memory for chunking
        let max_chunk_memory = available_memory / 10;
        let max_chunk_elements = max_chunk_memory / elementsize;

        // Clamp to reasonable bounds
        max_chunk_elements.clamp(1000, datasize / 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgorithmChoice {
    Standard,  // Normal algorithms
    InPlace,   // In-place algorithms to save memory
    Chunked,   // Process data in chunks
    Streaming, // Stream processing for minimal memory
    Parallel,  // Parallel algorithms for speed
}

/// Memory-efficient statistical operations with profiling
pub struct ProfiledStatistics<F> {
    profiler: Arc<MemoryProfiler>,
    cache: StatisticsCache<F>,
    adaptive_manager: AdaptiveMemoryManager,
}

impl<F> ProfiledStatistics<F>
where
    F: Float + NumCast + Clone + Send + Sync + std::fmt::Display,
{
    pub fn new(profiler: Arc<MemoryProfiler>) -> Self {
        let cache = StatisticsCache::new(1000, 50 * 1024 * 1024) // 50MB cache
            .with_profiler(profiler.clone());
        let adaptive_manager = AdaptiveMemoryManager::new(profiler.clone());

        Self {
            profiler: profiler.clone(),
            cache,
            adaptive_manager,
        }
    }

    /// Compute mean with memory profiling and caching
    pub fn mean_profiled<D>(&mut self, data: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F>,
    {
        let start_time = Instant::now();

        // Generate cache key
        let cache_key = format!("mean_{}", data.len());

        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result);
        }

        // Choose algorithm based on memory situation
        let algorithm = self.adaptive_manager.choose_algorithm(data.len());

        let result = match algorithm {
            AlgorithmChoice::Streaming => self.compute_mean_streaming(data),
            AlgorithmChoice::Chunked => self.compute_mean_chunked(data),
            _ => self.compute_mean_standard(data),
        }?;

        // Record allocation timing
        let duration = start_time.elapsed();
        self.profiler.record_allocation(
            "mean_computation",
            data.len() * std::mem::size_of::<F>(),
            duration,
        );

        // Cache result
        self.cache.put(cache_key, result);

        Ok(result)
    }

    fn compute_mean_streaming<D>(&self, data: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F>,
    {
        // Streaming mean computation (minimal memory)
        let mut sum = F::zero();
        let mut count = 0;

        for &value in data.iter() {
            sum = sum + value;
            count += 1;
        }

        if count == 0 {
            return Err(StatsError::invalid_argument(
                "Cannot compute mean of empty array",
            ));
        }

        Ok(sum / F::from(count).unwrap())
    }

    fn compute_mean_chunked<D>(&self, data: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F>,
    {
        // Chunked processing
        let chunksize = self
            .adaptive_manager
            .suggest_chunksize(data.len(), std::mem::size_of::<F>());
        let mut total_sum = F::zero();
        let mut total_count = 0;

        for chunk_start in (0..data.len()).step_by(chunksize) {
            let chunk_end = (chunk_start + chunksize).min(data.len());
            let chunk = data.slice(scirs2_core::ndarray::s![chunk_start..chunk_end]);

            let chunk_sum = chunk.iter().fold(F::zero(), |acc, &x| acc + x);
            total_sum = total_sum + chunk_sum;
            total_count += chunk.len();
        }

        if total_count == 0 {
            return Err(StatsError::invalid_argument(
                "Cannot compute mean of empty array",
            ));
        }

        Ok(total_sum / F::from(total_count).unwrap())
    }

    fn compute_mean_standard<D>(&self, data: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        D: Data<Elem = F>,
    {
        // Standard computation
        let sum = data.iter().fold(F::zero(), |acc, &x| acc + x);
        let count = data.len();

        if count == 0 {
            return Err(StatsError::invalid_argument(
                "Cannot compute mean of empty array",
            ));
        }

        Ok(sum / F::from(count).unwrap())
    }

    /// Get memory report
    pub fn get_memory_report(&self) -> MemoryReport {
        self.profiler.get_report()
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();

        profiler.record_allocation("test", 1024, Duration::from_millis(5));
        profiler.record_allocation("test", 2048, Duration::from_millis(10));

        let report = profiler.get_report();

        assert_eq!(report.allocations["test"].total_allocations, 2);
        assert_eq!(report.allocations["test"].total_bytes, 3072);
        assert_eq!(report.allocations["test"].peak_bytes, 2048);
    }

    #[test]
    fn test_statistics_cache() {
        let mut cache = StatisticsCache::new(2, 1024);

        cache.put("key1".to_string(), 42.0);
        cache.put("key2".to_string(), 24.0);

        assert_eq!(cache.get("key1"), Some(42.0));
        assert_eq!(cache.get("key2"), Some(24.0));
        assert_eq!(cache.get("key3"), None);

        // Test eviction
        cache.put("key3".to_string(), 12.0);
        assert_eq!(cache.cache.len(), 2); // Should evict least recently used
    }

    #[test]
    fn test_profiled_statistics() {
        let profiler = Arc::new(MemoryProfiler::new());
        let mut stats = ProfiledStatistics::new(profiler.clone());

        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = stats.mean_profiled(&data.view()).unwrap();

        assert_relative_eq!(mean, 3.0, epsilon = 1e-10);

        // Test caching
        let mean2 = stats.mean_profiled(&data.view()).unwrap();
        assert_relative_eq!(mean2, 3.0, epsilon = 1e-10);

        let report = stats.get_memory_report();
        assert!(!report.allocations.is_empty());
    }

    #[test]
    fn test_adaptive_memory_manager() {
        let profiler = Arc::new(MemoryProfiler::new());
        let manager = AdaptiveMemoryManager::new(profiler);

        // Test algorithm choice for different data sizes
        let choice_small = manager.choose_algorithm(1000);
        let choice_large = manager.choose_algorithm(1_000_000);

        // Should choose different algorithms based on size
        assert_ne!(choice_small, choice_large);

        // Test chunk size suggestion
        let chunksize = manager.suggest_chunksize(100_000, 8);
        assert!(chunksize > 0);
        assert!(chunksize <= 25_000); // Should be reasonable fraction of data size
    }
}
