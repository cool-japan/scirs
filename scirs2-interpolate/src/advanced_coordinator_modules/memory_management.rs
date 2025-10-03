//! Memory management and caching system for interpolation operations
//!
//! This module provides sophisticated memory management capabilities including
//! memory tracking, adaptive caching, and performance monitoring to optimize
//! interpolation operations for both memory usage and execution speed.

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use crate::advanced_coordinator_modules::types::{
    DataProfile, InterpolationMethodType, PerformanceMetrics,
};
use crate::error::{InterpolateError, InterpolateResult};

/// Memory management for interpolation
#[derive(Debug)]
pub struct InterpolationMemoryManager {
    /// Memory usage tracking
    memory_tracker: MemoryTracker,
    /// Cache management
    cache_manager: CacheManager,
    /// Memory allocation strategy
    allocation_strategy: MemoryAllocationStrategy,
}

impl InterpolationMemoryManager {
    /// Create new memory manager
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            memory_tracker: MemoryTracker::default(),
            cache_manager: CacheManager::new()?,
            allocation_strategy: MemoryAllocationStrategy::Adaptive,
        })
    }

    /// Track memory usage for operation
    pub fn track_memory_usage(&mut self, usage: usize, operation: String) -> InterpolateResult<()> {
        self.memory_tracker.record_usage(usage, operation)?;

        // Check if we need to adjust allocation strategy
        if self.memory_tracker.should_adjust_strategy() {
            self.adjust_allocation_strategy()?;
        }

        Ok(())
    }

    /// Get current memory usage
    pub fn get_current_usage(&self) -> usize {
        self.memory_tracker.current_usage
    }

    /// Get peak memory usage
    pub fn get_peak_usage(&self) -> usize {
        self.memory_tracker.peak_usage
    }

    /// Get memory usage statistics
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        MemoryStatistics {
            current_usage: self.memory_tracker.current_usage,
            peak_usage: self.memory_tracker.peak_usage,
            average_usage: self.memory_tracker.calculate_average_usage(),
            usage_trend: self.memory_tracker.calculate_usage_trend(),
            allocation_strategy: self.allocation_strategy.clone(),
        }
    }

    /// Optimize memory usage
    pub fn optimize_memory_usage(&mut self) -> InterpolateResult<MemoryOptimizationResult> {
        let initial_usage = self.memory_tracker.current_usage;

        // Trigger cache cleanup
        let cache_freed = self.cache_manager.cleanup_cache()?;

        // Adjust allocation strategy if needed
        self.adjust_allocation_strategy()?;

        let final_usage = self.memory_tracker.current_usage;
        let memory_freed = initial_usage.saturating_sub(final_usage);

        Ok(MemoryOptimizationResult {
            memory_freed: memory_freed + cache_freed,
            cache_freed,
            new_allocation_strategy: self.allocation_strategy.clone(),
            optimization_effectiveness: if initial_usage > 0 {
                memory_freed as f64 / initial_usage as f64
            } else {
                0.0
            },
        })
    }

    /// Set allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: MemoryAllocationStrategy) {
        self.allocation_strategy = strategy;
    }

    /// Adjust allocation strategy based on usage patterns
    fn adjust_allocation_strategy(&mut self) -> InterpolateResult<()> {
        let avg_usage = self.memory_tracker.calculate_average_usage();
        let peak_usage = self.memory_tracker.peak_usage;

        if peak_usage > 0 {
            let usage_ratio = avg_usage as f64 / peak_usage as f64;

            self.allocation_strategy = if usage_ratio > 0.8 {
                MemoryAllocationStrategy::Aggressive // High consistent usage
            } else if usage_ratio < 0.3 {
                MemoryAllocationStrategy::Conservative // Low average usage
            } else {
                MemoryAllocationStrategy::Adaptive // Balanced usage
            };
        }

        Ok(())
    }
}

impl Default for InterpolationMemoryManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            memory_tracker: MemoryTracker::default(),
            cache_manager: CacheManager::default(),
            allocation_strategy: MemoryAllocationStrategy::Adaptive,
        })
    }
}

/// Memory usage tracking
#[derive(Debug, Default)]
pub struct MemoryTracker {
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Memory usage history
    pub usage_history: VecDeque<MemoryUsageRecord>,
}

impl MemoryTracker {
    /// Record memory usage
    pub fn record_usage(&mut self, usage: usize, operation: String) -> InterpolateResult<()> {
        self.current_usage = usage;
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }

        let record = MemoryUsageRecord {
            usage,
            timestamp: Instant::now(),
            operation,
        };

        self.usage_history.push_back(record);

        // Keep only last 1000 records
        if self.usage_history.len() > 1000 {
            self.usage_history.pop_front();
        }

        Ok(())
    }

    /// Calculate average memory usage
    pub fn calculate_average_usage(&self) -> usize {
        if self.usage_history.is_empty() {
            return 0;
        }

        let sum: usize = self.usage_history.iter().map(|record| record.usage).sum();
        sum / self.usage_history.len()
    }

    /// Calculate usage trend
    pub fn calculate_usage_trend(&self) -> f64 {
        if self.usage_history.len() < 2 {
            return 0.0;
        }

        let recent_count = 10.min(self.usage_history.len());
        let recent_usage: Vec<usize> = self
            .usage_history
            .iter()
            .rev()
            .take(recent_count)
            .map(|record| record.usage)
            .collect();

        if recent_usage.len() < 2 {
            return 0.0;
        }

        let first_half = &recent_usage[recent_usage.len() / 2..];
        let second_half = &recent_usage[..recent_usage.len() / 2];

        let first_avg = first_half.iter().sum::<usize>() as f64 / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<usize>() as f64 / second_half.len() as f64;

        if first_avg > 0.0 {
            (second_avg - first_avg) / first_avg
        } else {
            0.0
        }
    }

    /// Check if allocation strategy should be adjusted
    pub fn should_adjust_strategy(&self) -> bool {
        self.usage_history.len() >= 10 && self.usage_history.len().is_multiple_of(50)
        // Check every 50 records
    }
}

/// Memory usage record
#[derive(Debug, Clone)]
pub struct MemoryUsageRecord {
    /// Memory usage (bytes)
    pub usage: usize,
    /// Timestamp
    pub timestamp: Instant,
    /// Operation type that caused the usage
    pub operation: String,
}

/// Cache management system
#[derive(Debug)]
pub struct CacheManager {
    /// Cache hit ratio
    hit_ratio: f64,
    /// Cache size (bytes)
    cache_size: usize,
    /// Eviction policy
    eviction_policy: CacheEvictionPolicy,
    /// Cache statistics
    statistics: CacheStatistics,
}

impl CacheManager {
    /// Create new cache manager
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            hit_ratio: 0.0,
            cache_size: 0,
            eviction_policy: CacheEvictionPolicy::Adaptive,
            statistics: CacheStatistics::default(),
        })
    }

    /// Record cache hit
    pub fn record_hit(&mut self) {
        self.statistics.hit_count += 1;
        self.update_hit_ratio();
    }

    /// Record cache miss
    pub fn record_miss(&mut self) {
        self.statistics.miss_count += 1;
        self.update_hit_ratio();
    }

    /// Record cache eviction
    pub fn record_eviction(&mut self, freed_bytes: usize) {
        self.statistics.eviction_count += 1;
        self.cache_size = self.cache_size.saturating_sub(freed_bytes);
        self.statistics.total_cache_size = self.cache_size;
    }

    /// Update cache size
    pub fn update_cache_size(&mut self, new_size: usize) {
        self.cache_size = new_size;
        self.statistics.total_cache_size = new_size;
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> &CacheStatistics {
        &self.statistics
    }

    /// Get hit ratio
    pub fn get_hit_ratio(&self) -> f64 {
        self.hit_ratio
    }

    /// Cleanup cache
    pub fn cleanup_cache(&mut self) -> InterpolateResult<usize> {
        let initial_size = self.cache_size;

        // Simulate cache cleanup based on eviction policy
        let cleanup_percentage = match self.eviction_policy {
            CacheEvictionPolicy::LRU => 0.3,
            CacheEvictionPolicy::LFU => 0.25,
            CacheEvictionPolicy::TimeBasedExpiration { .. } => 0.4,
            CacheEvictionPolicy::SizeBasedEviction { .. } => 0.5,
            CacheEvictionPolicy::Adaptive => {
                if self.hit_ratio < 0.3 {
                    0.6 // Aggressive cleanup for poor hit ratio
                } else if self.hit_ratio < 0.7 {
                    0.3 // Moderate cleanup
                } else {
                    0.1 // Conservative cleanup for good hit ratio
                }
            }
        };

        let bytes_freed = (self.cache_size as f64 * cleanup_percentage) as usize;
        self.cache_size = self.cache_size.saturating_sub(bytes_freed);
        self.statistics.total_cache_size = self.cache_size;
        self.statistics.eviction_count += 1;

        Ok(bytes_freed)
    }

    /// Update hit ratio
    fn update_hit_ratio(&mut self) {
        let total_requests = self.statistics.hit_count + self.statistics.miss_count;
        if total_requests > 0 {
            self.hit_ratio = self.statistics.hit_count as f64 / total_requests as f64;
        }
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            hit_ratio: 0.0,
            cache_size: 0,
            eviction_policy: CacheEvictionPolicy::Adaptive,
            statistics: CacheStatistics::default(),
        })
    }
}

/// Cache eviction policy
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TimeBasedExpiration { ttl: Duration },
    /// Size-based eviction
    SizeBasedEviction { max_size: usize },
    /// Adaptive policy
    Adaptive,
}

/// Memory allocation strategy
#[derive(Debug, Clone)]
pub enum MemoryAllocationStrategy {
    /// Conservative allocation
    Conservative,
    /// Aggressive pre-allocation
    Aggressive,
    /// Adaptive based on usage patterns
    Adaptive,
    /// Custom allocation strategy
    Custom { strategy: String },
}

/// Performance tracking for interpolation
#[derive(Debug, Default)]
pub struct InterpolationPerformanceTracker {
    /// Execution time history
    pub execution_times: VecDeque<f64>,
    /// Memory usage history
    pub memory_usage: VecDeque<usize>,
    /// Accuracy measurements
    pub accuracy_measurements: VecDeque<f64>,
    /// Method usage statistics
    pub method_usage: HashMap<InterpolationMethodType, MethodStats>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}

impl InterpolationPerformanceTracker {
    /// Track performance for method
    pub fn track_performance(
        &mut self,
        method: InterpolationMethodType,
        performance: &PerformanceMetrics,
    ) -> InterpolateResult<()> {
        // Update execution times
        self.execution_times
            .push_back(performance.execution_time_ms);
        if self.execution_times.len() > 1000 {
            self.execution_times.pop_front();
        }

        // Update memory usage
        self.memory_usage.push_back(performance.memory_usage_bytes);
        if self.memory_usage.len() > 1000 {
            self.memory_usage.pop_front();
        }

        // Update accuracy measurements
        self.accuracy_measurements.push_back(performance.accuracy);
        if self.accuracy_measurements.len() > 1000 {
            self.accuracy_measurements.pop_front();
        }

        // Update method statistics
        let stats = self.method_usage.entry(method).or_default();
        stats.update_with_performance(performance);

        // Update trends
        self.update_performance_trends()?;

        Ok(())
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            average_execution_time: self.calculate_average_execution_time(),
            average_memory_usage: self.calculate_average_memory_usage(),
            average_accuracy: self.calculate_average_accuracy(),
            best_performing_method: self.get_best_performing_method(),
            performance_trends: self.performance_trends.clone(),
        }
    }

    /// Calculate average execution time
    fn calculate_average_execution_time(&self) -> f64 {
        if self.execution_times.is_empty() {
            return 0.0;
        }
        self.execution_times.iter().sum::<f64>() / self.execution_times.len() as f64
    }

    /// Calculate average memory usage
    fn calculate_average_memory_usage(&self) -> usize {
        if self.memory_usage.is_empty() {
            return 0;
        }
        self.memory_usage.iter().sum::<usize>() / self.memory_usage.len()
    }

    /// Calculate average accuracy
    fn calculate_average_accuracy(&self) -> f64 {
        if self.accuracy_measurements.is_empty() {
            return 0.0;
        }
        self.accuracy_measurements.iter().sum::<f64>() / self.accuracy_measurements.len() as f64
    }

    /// Get best performing method
    fn get_best_performing_method(&self) -> Option<InterpolationMethodType> {
        self.method_usage
            .iter()
            .max_by(|a, b| {
                let score_a = a.1.calculate_performance_score();
                let score_b = b.1.calculate_performance_score();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(method, _)| *method)
    }

    /// Update performance trends
    fn update_performance_trends(&mut self) -> InterpolateResult<()> {
        self.performance_trends.execution_time_trend = self.calculate_trend(&self.execution_times);
        self.performance_trends.memory_usage_trend = self.calculate_usage_trend(&self.memory_usage);
        self.performance_trends.accuracy_trend = self.calculate_trend(&self.accuracy_measurements);

        // Calculate overall performance score
        self.performance_trends.overall_performance_score =
            (self.performance_trends.accuracy_trend * 0.5)
                + (-self.performance_trends.execution_time_trend * 0.3)
                + (-self.performance_trends.memory_usage_trend * 0.2);

        Ok(())
    }

    /// Calculate trend for f64 values
    fn calculate_trend(&self, values: &VecDeque<f64>) -> f64 {
        if values.len() < 10 {
            return 0.0;
        }

        let recent_count = 10.min(values.len());
        let recent_values: Vec<f64> = values.iter().rev().take(recent_count).copied().collect();

        let mid_point = recent_values.len() / 2;
        let first_half = &recent_values[mid_point..];
        let second_half = &recent_values[..mid_point];

        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

        if first_avg != 0.0 {
            (second_avg - first_avg) / first_avg
        } else {
            0.0
        }
    }

    /// Calculate trend for usize values
    fn calculate_usage_trend(&self, values: &VecDeque<usize>) -> f64 {
        if values.len() < 10 {
            return 0.0;
        }

        let recent_count = 10.min(values.len());
        let recent_values: Vec<usize> = values.iter().rev().take(recent_count).copied().collect();

        let mid_point = recent_values.len() / 2;
        let first_half = &recent_values[mid_point..];
        let second_half = &recent_values[..mid_point];

        let first_avg = first_half.iter().sum::<usize>() as f64 / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<usize>() as f64 / second_half.len() as f64;

        if first_avg != 0.0 {
            (second_avg - first_avg) / first_avg
        } else {
            0.0
        }
    }
}

/// Method usage statistics
#[derive(Debug, Clone, Default)]
pub struct MethodStats {
    /// Usage count
    pub usage_count: usize,
    /// Average execution time
    pub avg_execution_time: f64,
    /// Average memory usage
    pub avg_memory_usage: usize,
    /// Average accuracy
    pub avg_accuracy: f64,
    /// Success rate
    pub success_rate: f64,
    /// Total executions
    total_executions: usize,
    /// Successful executions
    successful_executions: usize,
}

impl MethodStats {
    /// Update statistics with performance data
    pub fn update_with_performance(&mut self, performance: &PerformanceMetrics) {
        self.usage_count += 1;
        self.total_executions += 1;

        if performance.accuracy > 0.7 {
            self.successful_executions += 1;
        }

        // Update running averages
        let count = self.usage_count as f64;
        self.avg_execution_time =
            (self.avg_execution_time * (count - 1.0) + performance.execution_time_ms) / count;
        self.avg_memory_usage = ((self.avg_memory_usage as f64 * (count - 1.0))
            + performance.memory_usage_bytes as f64) as usize
            / self.usage_count;
        self.avg_accuracy = (self.avg_accuracy * (count - 1.0) + performance.accuracy) / count;

        // Update success rate
        self.success_rate = self.successful_executions as f64 / self.total_executions as f64;
    }

    /// Calculate overall performance score
    pub fn calculate_performance_score(&self) -> f64 {
        if self.usage_count == 0 {
            return 0.0;
        }

        // Weighted score: accuracy (50%), success rate (30%), efficiency (20%)
        let efficiency_score = if self.avg_execution_time > 0.0 {
            1.0 / (1.0 + self.avg_execution_time / 1000.0) // Normalize execution time
        } else {
            1.0
        };

        self.avg_accuracy * 0.5 + self.success_rate * 0.3 + efficiency_score * 0.2
    }
}

/// Performance trends
#[derive(Debug, Default, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend (positive = getting slower)
    pub execution_time_trend: f64,
    /// Memory usage trend (positive = using more memory)
    pub memory_usage_trend: f64,
    /// Accuracy trend (positive = getting more accurate)
    pub accuracy_trend: f64,
    /// Overall performance score
    pub overall_performance_score: f64,
}

/// Adaptive interpolation cache system
#[derive(Debug)]
pub struct AdaptiveInterpolationCache<F: Float + Debug> {
    /// Cached interpolants
    interpolant_cache: HashMap<InterpolantCacheKey, CachedInterpolant<F>>,
    /// Cache statistics
    cache_stats: CacheStatistics,
    /// Cache policy
    cache_policy: AdaptiveCachePolicy,
}

impl<F: Float + Debug> AdaptiveInterpolationCache<F> {
    /// Create new adaptive cache
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            interpolant_cache: HashMap::new(),
            cache_stats: CacheStatistics::default(),
            cache_policy: AdaptiveCachePolicy::new(),
        })
    }

    /// Get cached interpolant
    pub fn get(&mut self, key: &InterpolantCacheKey) -> Option<&CachedInterpolant<F>> {
        if let Some(interpolant) = self.interpolant_cache.get_mut(key) {
            interpolant.access_count += 1;
            interpolant.last_access = Instant::now();
            self.cache_stats.hit_count += 1;
            Some(interpolant)
        } else {
            self.cache_stats.miss_count += 1;
            None
        }
    }

    /// Store interpolant in cache
    pub fn store(
        &mut self,
        key: InterpolantCacheKey,
        interpolant: CachedInterpolant<F>,
    ) -> InterpolateResult<()> {
        // Check if we need to evict
        if self.should_evict() {
            self.evict_entries()?;
        }

        let interpolant_size = interpolant.interpolant_data.len();
        self.interpolant_cache.insert(key, interpolant);
        self.cache_stats.total_cache_size += interpolant_size;

        Ok(())
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.interpolant_cache.clear();
        self.cache_stats.total_cache_size = 0;
        self.cache_stats.eviction_count += 1;
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> &CacheStatistics {
        &self.cache_stats
    }

    /// Check if eviction is needed
    fn should_evict(&self) -> bool {
        match &self.cache_policy.base_policy {
            CacheEvictionPolicy::SizeBasedEviction { max_size } => {
                self.cache_stats.total_cache_size > *max_size
            }
            _ => self.interpolant_cache.len() > 1000, // Default size limit
        }
    }

    /// Evict cache entries
    fn evict_entries(&mut self) -> InterpolateResult<()> {
        match &self.cache_policy.base_policy {
            CacheEvictionPolicy::LRU => {
                self.evict_lru();
            }
            CacheEvictionPolicy::LFU => {
                self.evict_lfu();
            }
            CacheEvictionPolicy::TimeBasedExpiration { ttl } => {
                self.evict_expired(*ttl);
            }
            _ => {
                self.evict_lru(); // Default to LRU
            }
        }

        self.cache_stats.eviction_count += 1;
        Ok(())
    }

    /// Evict least recently used entries
    fn evict_lru(&mut self) {
        if let Some((key_to_remove, size)) = self
            .interpolant_cache
            .iter()
            .min_by_key(|(_, interpolant)| interpolant.last_access)
            .map(|(key, interpolant)| (key.clone(), interpolant.interpolant_data.len()))
        {
            self.interpolant_cache.remove(&key_to_remove);
            self.cache_stats.total_cache_size =
                self.cache_stats.total_cache_size.saturating_sub(size);
        }
    }

    /// Evict least frequently used entries
    fn evict_lfu(&mut self) {
        if let Some((key_to_remove, size)) = self
            .interpolant_cache
            .iter()
            .min_by_key(|(_, interpolant)| interpolant.access_count)
            .map(|(key, interpolant)| (key.clone(), interpolant.interpolant_data.len()))
        {
            self.interpolant_cache.remove(&key_to_remove);
            self.cache_stats.total_cache_size =
                self.cache_stats.total_cache_size.saturating_sub(size);
        }
    }

    /// Evict expired entries
    fn evict_expired(&mut self, ttl: Duration) {
        let now = Instant::now();
        let mut keys_to_remove = Vec::new();
        let mut total_size_removed = 0;

        for (key, interpolant) in &self.interpolant_cache {
            if now.duration_since(interpolant.creation_time) > ttl {
                keys_to_remove.push(key.clone());
                total_size_removed += interpolant.interpolant_data.len();
            }
        }

        for key in keys_to_remove {
            self.interpolant_cache.remove(&key);
        }

        self.cache_stats.total_cache_size = self
            .cache_stats
            .total_cache_size
            .saturating_sub(total_size_removed);
    }
}

impl<F: Float + Debug> Default for AdaptiveInterpolationCache<F> {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            interpolant_cache: HashMap::new(),
            cache_stats: CacheStatistics::default(),
            cache_policy: AdaptiveCachePolicy::default(),
        })
    }
}

/// Key for interpolant cache
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct InterpolantCacheKey {
    /// Data signature (hash of input data)
    pub data_signature: String,
    /// Method used
    pub method: InterpolationMethodType,
    /// Parameters used
    pub parameters: String, // Serialized parameters
}

/// Cached interpolant
#[derive(Debug, Clone)]
pub struct CachedInterpolant<F: Float> {
    /// Interpolant data
    pub interpolant_data: Vec<u8>,
    /// Creation time
    pub creation_time: Instant,
    /// Access count
    pub access_count: usize,
    /// Last access time
    pub last_access: Instant,
    /// Performance metrics
    pub performance_metrics: CachedInterpolantMetrics<F>,
}

/// Performance metrics for cached interpolants
#[derive(Debug, Clone)]
pub struct CachedInterpolantMetrics<F: Float> {
    /// Creation time
    pub creation_time: F,
    /// Memory usage
    pub memory_usage: usize,
    /// Accuracy score
    pub accuracy_score: F,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Cache hit count
    pub hit_count: usize,
    /// Cache miss count
    pub miss_count: usize,
    /// Cache eviction count
    pub eviction_count: usize,
    /// Total cache size (bytes)
    pub total_cache_size: usize,
}

impl CacheStatistics {
    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total > 0 {
            self.hit_count as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Adaptive cache policy
#[derive(Debug)]
pub struct AdaptiveCachePolicy {
    /// Base eviction policy
    base_policy: CacheEvictionPolicy,
    /// Adaptive parameters
    adaptive_params: CacheAdaptiveParams,
}

impl AdaptiveCachePolicy {
    /// Create new adaptive cache policy
    pub fn new() -> Self {
        Self {
            base_policy: CacheEvictionPolicy::LRU,
            adaptive_params: CacheAdaptiveParams::default(),
        }
    }

    /// Adapt policy based on statistics
    pub fn adapt_policy(&mut self, stats: &CacheStatistics) {
        let hit_ratio = stats.hit_ratio();

        if hit_ratio < self.adaptive_params.hit_ratio_threshold {
            // Poor hit ratio - try different policy
            self.base_policy = match &self.base_policy {
                CacheEvictionPolicy::LRU => CacheEvictionPolicy::LFU,
                CacheEvictionPolicy::LFU => CacheEvictionPolicy::TimeBasedExpiration {
                    ttl: Duration::from_secs(300),
                },
                _ => CacheEvictionPolicy::LRU,
            };
        }
    }
}

impl Default for AdaptiveCachePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive parameters for cache
#[derive(Debug, Clone)]
pub struct CacheAdaptiveParams {
    /// Hit ratio threshold for policy adaptation
    pub hit_ratio_threshold: f64,
    /// Memory pressure threshold
    pub memory_pressure_threshold: f64,
    /// Access pattern weight
    pub access_pattern_weight: f64,
    /// Temporal locality weight
    pub temporal_locality_weight: f64,
}

impl Default for CacheAdaptiveParams {
    fn default() -> Self {
        Self {
            hit_ratio_threshold: 0.7,
            memory_pressure_threshold: 0.8,
            access_pattern_weight: 0.6,
            temporal_locality_weight: 0.4,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average memory usage
    pub average_usage: usize,
    /// Usage trend
    pub usage_trend: f64,
    /// Current allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
}

/// Memory optimization result
#[derive(Debug, Clone)]
pub struct MemoryOptimizationResult {
    /// Memory freed (bytes)
    pub memory_freed: usize,
    /// Memory freed from cache (bytes)
    pub cache_freed: usize,
    /// New allocation strategy
    pub new_allocation_strategy: MemoryAllocationStrategy,
    /// Optimization effectiveness (0.0 to 1.0)
    pub optimization_effectiveness: f64,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Average execution time
    pub average_execution_time: f64,
    /// Average memory usage
    pub average_memory_usage: usize,
    /// Average accuracy
    pub average_accuracy: f64,
    /// Best performing method
    pub best_performing_method: Option<InterpolationMethodType>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}
