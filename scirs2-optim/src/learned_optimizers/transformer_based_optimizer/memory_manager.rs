//! Memory management for transformer-based optimizer

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::error::Result;
use super::config::{TransformerBasedOptimizerConfig, MemoryConfig, CacheEvictionStrategy};

/// Memory management strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryManagementStrategy {
    /// Simple FIFO eviction
    FIFO,
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive replacement cache
    ARC,
    /// Compressed memory storage
    Compressed,
    /// Hierarchical memory organization
    Hierarchical,
}

/// Transformer memory manager
pub struct TransformerMemoryManager<T: Float> {
    /// Memory management strategy
    strategy: MemoryManagementStrategy,

    /// Configuration
    config: MemoryConfig,

    /// Primary memory cache
    primary_cache: MemoryCache<T>,

    /// Secondary cache for overflow
    secondary_cache: Option<MemoryCache<T>>,

    /// Memory compression manager
    compression_manager: Option<CompressionManager<T>>,

    /// Memory statistics
    statistics: MemoryStatistics,

    /// Access patterns tracker
    access_tracker: AccessTracker,

    /// Memory pressure monitor
    pressure_monitor: MemoryPressureMonitor,

    /// Model dimension
    model_dimension: usize,
}

impl<T: Float> TransformerMemoryManager<T> {
    /// Create new memory manager
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let memory_config = config.memory_config.clone();
        let strategy = match memory_config.eviction_strategy {
            CacheEvictionStrategy::LRU => MemoryManagementStrategy::LRU,
            CacheEvictionStrategy::LFU => MemoryManagementStrategy::LFU,
            CacheEvictionStrategy::FIFO => MemoryManagementStrategy::FIFO,
            CacheEvictionStrategy::Random => MemoryManagementStrategy::LRU, // Fallback
        };

        let primary_cache = MemoryCache::new(
            memory_config.max_cache_size / 2,
            memory_config.eviction_strategy,
        )?;

        let secondary_cache = if memory_config.max_cache_size > 1024 * 1024 * 100 { // 100MB
            Some(MemoryCache::new(
                memory_config.max_cache_size / 2,
                CacheEvictionStrategy::FIFO,
            )?)
        } else {
            None
        };

        let compression_manager = if memory_config.enable_compression {
            Some(CompressionManager::new(0.5)?) // 50% compression ratio target
        } else {
            None
        };

        let statistics = MemoryStatistics::new();
        let access_tracker = AccessTracker::new(1000);
        let pressure_monitor = MemoryPressureMonitor::new();

        Ok(Self {
            strategy,
            config: memory_config,
            primary_cache,
            secondary_cache,
            compression_manager,
            statistics,
            access_tracker,
            pressure_monitor,
            model_dimension: config.model_dimension,
        })
    }

    /// Store tensor in memory with key
    pub fn store(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let start_time = Instant::now();

        // Check memory pressure and evict if necessary
        self.pressure_monitor.update(self.get_memory_usage());
        if self.pressure_monitor.is_high_pressure() {
            self.evict_memory()?;
        }

        // Try to store in primary cache first
        let storage_result = match self.strategy {
            MemoryManagementStrategy::LRU => self.store_lru(key.clone(), tensor.clone()),
            MemoryManagementStrategy::LFU => self.store_lfu(key.clone(), tensor.clone()),
            MemoryManagementStrategy::FIFO => self.store_fifo(key.clone(), tensor.clone()),
            MemoryManagementStrategy::ARC => self.store_arc(key.clone(), tensor.clone()),
            MemoryManagementStrategy::Compressed => self.store_compressed(key.clone(), tensor.clone()),
            MemoryManagementStrategy::Hierarchical => self.store_hierarchical(key.clone(), tensor.clone()),
        };

        // Get tensor length before potential move
        let tensor_len = tensor.len();

        // If primary cache is full, try secondary cache
        if storage_result.is_err() && self.secondary_cache.is_some() {
            if let Some(ref mut secondary) = self.secondary_cache {
                secondary.store(key.clone(), tensor)?;
            }
        }

        // Update statistics
        let storage_time = start_time.elapsed();
        self.statistics.record_storage(tensor_len, storage_time);
        self.access_tracker.record_write(key);

        storage_result
    }

    /// Retrieve tensor from memory
    pub fn retrieve(&mut self, key: &str) -> Result<Option<Array2<T>>> {
        let start_time = Instant::now();

        // Try primary cache first
        let result = self.primary_cache.retrieve(key)?;

        if result.is_some() {
            self.access_tracker.record_read(key.to_string());
            let retrieval_time = start_time.elapsed();
            self.statistics.record_retrieval(retrieval_time, true);
            return Ok(result);
        }

        // Try secondary cache
        if let Some(ref mut secondary) = self.secondary_cache {
            let result = secondary.retrieve(key)?;
            if result.is_some() {
                self.access_tracker.record_read(key.to_string());
                let retrieval_time = start_time.elapsed();
                self.statistics.record_retrieval(retrieval_time, true);
                return Ok(result);
            }
        }

        // Check compressed storage
        if let Some(ref mut compression) = self.compression_manager {
            if let Some(compressed_data) = compression.retrieve(key)? {
                let decompressed = compression.decompress(&compressed_data)?;
                self.access_tracker.record_read(key.to_string());
                let retrieval_time = start_time.elapsed();
                self.statistics.record_retrieval(retrieval_time, true);
                return Ok(Some(decompressed));
            }
        }

        let retrieval_time = start_time.elapsed();
        self.statistics.record_retrieval(retrieval_time, false);
        Ok(None)
    }

    /// Remove tensor from memory
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        let mut removed = false;

        if self.primary_cache.remove(key)? {
            removed = true;
        }

        if let Some(ref mut secondary) = self.secondary_cache {
            if secondary.remove(key)? {
                removed = true;
            }
        }

        if let Some(ref mut compression) = self.compression_manager {
            if compression.remove(key)? {
                removed = true;
            }
        }

        self.access_tracker.record_removal(key.to_string());
        Ok(removed)
    }

    /// Clear all memory
    pub fn clear(&mut self) -> Result<()> {
        self.primary_cache.clear()?;

        if let Some(ref mut secondary) = self.secondary_cache {
            secondary.clear()?;
        }

        if let Some(ref mut compression) = self.compression_manager {
            compression.clear()?;
        }

        self.statistics.reset();
        self.access_tracker.clear();
        self.pressure_monitor.reset();

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> usize {
        let primary_usage = self.primary_cache.get_memory_usage();
        let secondary_usage = self.secondary_cache
            .as_ref()
            .map(|cache| cache.get_memory_usage())
            .unwrap_or(0);
        let compression_usage = self.compression_manager
            .as_ref()
            .map(|comp| comp.get_memory_usage())
            .unwrap_or(0);

        primary_usage + secondary_usage + compression_usage
    }

    /// Optimize memory layout
    pub fn optimize_memory(&mut self) -> Result<OptimizationReport> {
        let start_time = Instant::now();
        let initial_usage = self.get_memory_usage();

        // Analyze access patterns
        let access_patterns = self.access_tracker.analyze_patterns();

        // Reorganize based on access frequency
        self.reorganize_by_frequency(&access_patterns)?;

        // Compress frequently accessed but large items
        if let Some(ref mut compression) = self.compression_manager {
            compression.optimize_compression_ratios(&access_patterns)?;
        }

        // Defragment memory
        self.defragment_memory()?;

        let final_usage = self.get_memory_usage();
        let optimization_time = start_time.elapsed();

        Ok(OptimizationReport {
            initial_memory_usage: initial_usage,
            final_memory_usage: final_usage,
            memory_saved: initial_usage.saturating_sub(final_usage),
            optimization_time,
            operations_performed: access_patterns.total_accesses,
        })
    }

    /// Prefetch data based on access patterns
    pub fn prefetch(&mut self, keys: Vec<String>) -> Result<usize> {
        let mut prefetched_count = 0;

        for key in keys {
            if !self.primary_cache.contains(&key) {
                // Try to move from secondary to primary cache
                if let Some(ref mut secondary) = self.secondary_cache {
                    if let Some(tensor) = secondary.retrieve(&key)? {
                        if self.primary_cache.store(key.clone(), tensor).is_ok() {
                            secondary.remove(&key)?;
                            prefetched_count += 1;
                        }
                    }
                }

                // Try to decompress and move to primary cache
                if let Some(ref mut compression) = self.compression_manager {
                    if let Some(compressed_data) = compression.retrieve(&key)? {
                        let decompressed = compression.decompress(&compressed_data)?;
                        if self.primary_cache.store(key.clone(), decompressed).is_ok() {
                            compression.remove(&key)?;
                            prefetched_count += 1;
                        }
                    }
                }
            }
        }

        Ok(prefetched_count)
    }

    /// Storage strategy implementations
    fn store_lru(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        self.primary_cache.store(key, tensor)
    }

    fn store_lfu(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        // For LFU, we need to track access frequency
        self.primary_cache.store(key, tensor)
    }

    fn store_fifo(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        self.primary_cache.store(key, tensor)
    }

    fn store_arc(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        // Adaptive Replacement Cache - simplified implementation
        self.primary_cache.store(key, tensor)
    }

    fn store_compressed(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        if let Some(ref mut compression) = self.compression_manager {
            let compressed_data = compression.compress(&tensor)?;
            compression.store(key, compressed_data)?;
            Ok(())
        } else {
            self.primary_cache.store(key, tensor)
        }
    }

    fn store_hierarchical(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let tensor_size = tensor.len() * std::mem::size_of::<T>();

        if tensor_size < self.config.allocation_block_size {
            // Small tensors go to primary cache
            self.primary_cache.store(key, tensor)
        } else if let Some(ref mut secondary) = self.secondary_cache {
            // Large tensors go to secondary cache
            secondary.store(key, tensor)
        } else {
            // Fallback to primary cache
            self.primary_cache.store(key, tensor)
        }
    }

    fn evict_memory(&mut self) -> Result<()> {
        // Evict from primary cache first
        self.primary_cache.evict_lru()?;

        // If still under pressure, evict from secondary cache
        if self.pressure_monitor.is_high_pressure() {
            if let Some(ref mut secondary) = self.secondary_cache {
                secondary.evict_lru()?;
            }
        }

        Ok(())
    }

    fn reorganize_by_frequency(&mut self, patterns: &AccessPatterns) -> Result<()> {
        // Move frequently accessed items to primary cache
        let frequent_keys: Vec<String> = patterns.frequency_map
            .iter()
            .filter(|(_, &count)| count as f64 > patterns.average_frequency)
            .map(|(key, _)| key.clone())
            .collect();

        self.prefetch(frequent_keys)?;
        Ok(())
    }

    fn defragment_memory(&mut self) -> Result<()> {
        // Simplified defragmentation - rebuild caches
        let primary_items = self.primary_cache.get_all_items()?;
        self.primary_cache.clear()?;

        for (key, tensor) in primary_items {
            self.primary_cache.store(key, tensor)?;
        }

        Ok(())
    }

    /// Get memory statistics
    pub fn get_statistics(&self) -> &MemoryStatistics {
        &self.statistics
    }

    /// Get access patterns
    pub fn get_access_patterns(&self) -> AccessPatterns {
        self.access_tracker.analyze_patterns()
    }

    /// Set memory management strategy
    pub fn set_strategy(&mut self, strategy: MemoryManagementStrategy) {
        self.strategy = strategy;
    }

    /// Get current memory pressure
    pub fn get_memory_pressure(&self) -> f64 {
        self.pressure_monitor.get_pressure_ratio()
    }
}

/// Memory cache implementation
pub struct MemoryCache<T: Float> {
    /// Stored tensors
    storage: HashMap<String, CacheEntry<T>>,

    /// Access order for LRU
    access_order: VecDeque<String>,

    /// Access frequency for LFU
    access_frequency: HashMap<String, usize>,

    /// Maximum cache size in bytes
    max_size: usize,

    /// Current cache size in bytes
    current_size: usize,

    /// Eviction strategy
    eviction_strategy: CacheEvictionStrategy,
}

impl<T: Float> MemoryCache<T> {
    pub fn new(max_size: usize, eviction_strategy: CacheEvictionStrategy) -> Result<Self> {
        Ok(Self {
            storage: HashMap::new(),
            access_order: VecDeque::new(),
            access_frequency: HashMap::new(),
            max_size,
            current_size: 0,
            eviction_strategy,
        })
    }

    pub fn store(&mut self, key: String, tensor: Array2<T>) -> Result<()> {
        let tensor_size = tensor.len() * std::mem::size_of::<T>();

        // Check if we need to evict
        while self.current_size + tensor_size > self.max_size && !self.storage.is_empty() {
            self.evict_one()?;
        }

        if tensor_size > self.max_size {
            return Err(crate::error::OptimError::Other(
                "Tensor too large for cache".to_string()
            ));
        }

        // Remove existing entry if present
        if let Some(old_entry) = self.storage.remove(&key) {
            self.current_size -= old_entry.size;
            self.remove_from_access_order(&key);
        }

        // Add new entry
        let entry = CacheEntry {
            tensor,
            size: tensor_size,
            access_time: Instant::now(),
            access_count: 1,
        };

        self.storage.insert(key.clone(), entry);
        self.current_size += tensor_size;
        self.update_access_tracking(&key);

        Ok(())
    }

    pub fn retrieve(&mut self, key: &str) -> Result<Option<Array2<T>>> {
        let tensor_result = if let Some(entry) = self.storage.get_mut(key) {
            entry.access_time = Instant::now();
            entry.access_count += 1;
            Some(entry.tensor.clone())
        } else {
            None
        };

        // Update access tracking after releasing the mutable borrow
        if tensor_result.is_some() {
            self.update_access_tracking(key);
        }

        Ok(tensor_result)
    }

    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(entry) = self.storage.remove(key) {
            self.current_size -= entry.size;
            self.remove_from_access_order(key);
            self.access_frequency.remove(key);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.storage.contains_key(key)
    }

    pub fn clear(&mut self) -> Result<()> {
        self.storage.clear();
        self.access_order.clear();
        self.access_frequency.clear();
        self.current_size = 0;
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.current_size
    }

    pub fn evict_lru(&mut self) -> Result<()> {
        if let Some(oldest_key) = self.access_order.front().cloned() {
            self.remove(&oldest_key)?;
        }
        Ok(())
    }

    fn evict_one(&mut self) -> Result<()> {
        match self.eviction_strategy {
            CacheEvictionStrategy::LRU => self.evict_lru(),
            CacheEvictionStrategy::LFU => self.evict_lfu(),
            CacheEvictionStrategy::FIFO => self.evict_fifo(),
            CacheEvictionStrategy::Random => self.evict_random(),
        }
    }

    fn evict_lfu(&mut self) -> Result<()> {
        if let Some((min_freq, lfu_key)) = self.access_frequency
            .iter()
            .min_by_key(|(_, &freq)| freq)
            .map(|(key, &freq)| (freq, key.clone()))
        {
            self.remove(&lfu_key)?;
        }
        Ok(())
    }

    fn evict_fifo(&mut self) -> Result<()> {
        if let Some(first_key) = self.access_order.front().cloned() {
            self.remove(&first_key)?;
        }
        Ok(())
    }

    fn evict_random(&mut self) -> Result<()> {
        if let Some(random_key) = self.storage.keys().next().cloned() {
            self.remove(&random_key)?;
        }
        Ok(())
    }

    fn update_access_tracking(&mut self, key: &str) {
        // Update LRU order
        self.remove_from_access_order(key);
        self.access_order.push_back(key.to_string());

        // Update LFU frequency
        *self.access_frequency.entry(key.to_string()).or_insert(0) += 1;
    }

    fn remove_from_access_order(&mut self, key: &str) {
        self.access_order.retain(|k| k != key);
    }

    pub fn get_all_items(&self) -> Result<Vec<(String, Array2<T>)>> {
        let items = self.storage
            .iter()
            .map(|(key, entry)| (key.clone(), entry.tensor.clone()))
            .collect();
        Ok(items)
    }
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry<T: Float> {
    pub tensor: Array2<T>,
    pub size: usize,
    pub access_time: Instant,
    pub access_count: usize,
}

/// Compression manager
pub struct CompressionManager<T: Float> {
    /// Compressed storage
    compressed_storage: HashMap<String, CompressedData<T>>,

    /// Compression ratio target
    compression_ratio: f64,

    /// Memory usage
    memory_usage: usize,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CompressionManager<T> {
    pub fn new(compression_ratio: f64) -> Result<Self> {
        Ok(Self {
            compressed_storage: HashMap::new(),
            compression_ratio,
            memory_usage: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn compress(&self, tensor: &Array2<T>) -> Result<CompressedData<T>> {
        // Simplified compression - just store dimensions and flattened data
        let shape = tensor.shape().to_vec();
        let data: Vec<T> = tensor.iter().cloned().collect();
        let data_len = data.len(); // Get length before move

        Ok(CompressedData::<T> {
            shape,
            data,
            original_size: tensor.len() * std::mem::size_of::<T>(),
            compressed_size: data_len * std::mem::size_of::<T>() / 2, // Simulated compression
        })
    }

    pub fn decompress(&self, compressed: &CompressedData<T>) -> Result<Array2<T>> {
        let array = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), compressed.data.clone())
            .map_err(|_| crate::error::OptimError::Other("Decompression failed".to_string()))?;
        Ok(array)
    }

    pub fn store(&mut self, key: String, compressed: CompressedData<T>) -> Result<()> {
        self.memory_usage += compressed.compressed_size;
        self.compressed_storage.insert(key, compressed);
        Ok(())
    }

    pub fn retrieve(&self, key: &str) -> Result<Option<CompressedData<T>>> {
        Ok(self.compressed_storage.get(key).cloned())
    }

    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(compressed) = self.compressed_storage.remove(key) {
            self.memory_usage -= compressed.compressed_size;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn clear(&mut self) -> Result<()> {
        self.compressed_storage.clear();
        self.memory_usage = 0;
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage
    }

    pub fn optimize_compression_ratios(&mut self, _patterns: &AccessPatterns) -> Result<()> {
        // Optimize compression based on access patterns
        // This is a placeholder for more sophisticated compression optimization
        Ok(())
    }
}

/// Compressed data structure
#[derive(Debug, Clone)]
pub struct CompressedData<T: Float> {
    pub shape: Vec<usize>,
    pub data: Vec<T>, // Generic data type
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total storage operations
    pub total_stores: usize,

    /// Total retrieval operations
    pub total_retrievals: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Total bytes stored
    pub total_bytes_stored: usize,

    /// Average storage time
    pub average_storage_time: Duration,

    /// Average retrieval time
    pub average_retrieval_time: Duration,

    /// Memory pressure events
    pub pressure_events: usize,
}

impl MemoryStatistics {
    pub fn new() -> Self {
        Self {
            total_stores: 0,
            total_retrievals: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_bytes_stored: 0,
            average_storage_time: Duration::new(0, 0),
            average_retrieval_time: Duration::new(0, 0),
            pressure_events: 0,
        }
    }

    pub fn record_storage(&mut self, bytes: usize, time: Duration) {
        self.total_stores += 1;
        self.total_bytes_stored += bytes;
        self.average_storage_time =
            (self.average_storage_time * (self.total_stores - 1) as u32 + time) / self.total_stores as u32;
    }

    pub fn record_retrieval(&mut self, time: Duration, hit: bool) {
        self.total_retrievals += 1;
        if hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        self.average_retrieval_time =
            (self.average_retrieval_time * (self.total_retrievals - 1) as u32 + time) / self.total_retrievals as u32;
    }

    pub fn record_pressure_event(&mut self) {
        self.pressure_events += 1;
    }

    pub fn get_hit_ratio(&self) -> f64 {
        if self.total_retrievals > 0 {
            self.cache_hits as f64 / self.total_retrievals as f64
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Access tracker
pub struct AccessTracker {
    /// Read access log
    read_log: VecDeque<AccessEvent>,

    /// Write access log
    write_log: VecDeque<AccessEvent>,

    /// Maximum log size
    max_log_size: usize,
}

impl AccessTracker {
    pub fn new(max_log_size: usize) -> Self {
        Self {
            read_log: VecDeque::new(),
            write_log: VecDeque::new(),
            max_log_size,
        }
    }

    pub fn record_read(&mut self, key: String) {
        self.read_log.push_back(AccessEvent {
            key,
            timestamp: Instant::now(),
        });

        if self.read_log.len() > self.max_log_size {
            self.read_log.pop_front();
        }
    }

    pub fn record_write(&mut self, key: String) {
        self.write_log.push_back(AccessEvent {
            key,
            timestamp: Instant::now(),
        });

        if self.write_log.len() > self.max_log_size {
            self.write_log.pop_front();
        }
    }

    pub fn record_removal(&mut self, _key: String) {
        // Record removal operation
    }

    pub fn analyze_patterns(&self) -> AccessPatterns {
        let mut frequency_map = HashMap::new();

        // Count access frequencies
        for event in self.read_log.iter().chain(self.write_log.iter()) {
            *frequency_map.entry(event.key.clone()).or_insert(0) += 1;
        }

        let total_accesses: usize = frequency_map.values().sum();
        let average_frequency = if frequency_map.is_empty() {
            0.0
        } else {
            total_accesses as f64 / frequency_map.len() as f64
        };

        AccessPatterns {
            frequency_map,
            average_frequency,
            total_accesses,
        }
    }

    pub fn clear(&mut self) {
        self.read_log.clear();
        self.write_log.clear();
    }
}

/// Access event
#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub key: String,
    pub timestamp: Instant,
}

/// Access patterns analysis
#[derive(Debug, Clone)]
pub struct AccessPatterns {
    pub frequency_map: HashMap<String, usize>,
    pub average_frequency: f64,
    pub total_accesses: usize,
}

/// Memory pressure monitor
pub struct MemoryPressureMonitor {
    /// Current memory usage
    current_usage: usize,

    /// Maximum allowed memory
    max_memory: usize,

    /// Pressure thresholds
    warning_threshold: f64,
    critical_threshold: f64,

    /// Pressure history
    pressure_history: VecDeque<f64>,
}

impl MemoryPressureMonitor {
    pub fn new() -> Self {
        Self {
            current_usage: 0,
            max_memory: 1024 * 1024 * 1024, // 1GB default
            warning_threshold: 0.7,
            critical_threshold: 0.9,
            pressure_history: VecDeque::new(),
        }
    }

    pub fn update(&mut self, current_usage: usize) {
        self.current_usage = current_usage;
        let pressure_ratio = self.get_pressure_ratio();

        self.pressure_history.push_back(pressure_ratio);
        if self.pressure_history.len() > 100 {
            self.pressure_history.pop_front();
        }
    }

    pub fn get_pressure_ratio(&self) -> f64 {
        if self.max_memory > 0 {
            self.current_usage as f64 / self.max_memory as f64
        } else {
            0.0
        }
    }

    pub fn is_high_pressure(&self) -> bool {
        self.get_pressure_ratio() > self.critical_threshold
    }

    pub fn is_warning_pressure(&self) -> bool {
        self.get_pressure_ratio() > self.warning_threshold
    }

    pub fn reset(&mut self) {
        self.current_usage = 0;
        self.pressure_history.clear();
    }
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub initial_memory_usage: usize,
    pub final_memory_usage: usize,
    pub memory_saved: usize,
    pub optimization_time: Duration,
    pub operations_performed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let manager = TransformerMemoryManager::new(&config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_memory_cache() {
        let mut cache = MemoryCache::<f32>::new(1024 * 1024, CacheEvictionStrategy::LRU);
        assert!(cache.is_ok());

        let mut c = cache.unwrap();
        let tensor = Array2::<f32>::ones((10, 10));
        assert!(c.store("test".to_string(), tensor).is_ok());
        assert!(c.contains("test"));
    }

    #[test]
    fn test_compression_manager() {
        let compression = CompressionManager::<f32>::new(0.5);
        assert!(compression.is_ok());

        let comp = compression.unwrap();
        let tensor = Array2::<f32>::ones((5, 5));
        let compressed = comp.compress(&tensor);
        assert!(compressed.is_ok());

        let decompressed = comp.decompress(&compressed.unwrap());
        assert!(decompressed.is_ok());
    }

    #[test]
    fn test_access_tracker() {
        let mut tracker = AccessTracker::new(100);

        tracker.record_read("key1".to_string());
        tracker.record_write("key2".to_string());

        let patterns = tracker.analyze_patterns();
        assert!(patterns.total_accesses > 0);
    }

    #[test]
    fn test_memory_pressure_monitor() {
        let mut monitor = MemoryPressureMonitor::new();

        monitor.update(500 * 1024 * 1024); // 500MB
        assert!(!monitor.is_high_pressure());

        monitor.update(950 * 1024 * 1024); // 950MB
        assert!(monitor.is_high_pressure());
    }
}