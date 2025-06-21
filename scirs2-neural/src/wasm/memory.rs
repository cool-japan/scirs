//! WebAssembly memory management and configuration
//!
//! This module provides comprehensive memory management functionality for WebAssembly neural networks including:
//! - Memory configuration and growth strategies
//! - Memory alignment and optimization
//! - Shared memory support for multi-threading
//! - Progressive loading and streaming for large models
//! - Caching strategies and storage management

/// WebAssembly memory configuration
#[derive(Debug, Clone)]
pub struct WasmMemoryConfig {
    /// Initial memory pages (64KB each)
    pub initial_pages: u32,
    /// Maximum memory pages
    pub maximum_pages: Option<u32>,
    /// Shared memory (for threading)
    pub shared: bool,
    /// Memory growth strategy
    pub growth_strategy: MemoryGrowthStrategy,
    /// Memory alignment
    pub alignment: MemoryAlignment,
}

/// Memory growth strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryGrowthStrategy {
    /// Fixed size - no growth allowed
    Fixed,
    /// On-demand growth
    OnDemand,
    /// Pre-allocated growth
    PreAllocated,
    /// Streaming growth for large models
    Streaming,
}

/// Memory alignment configuration
#[derive(Debug, Clone)]
pub struct MemoryAlignment {
    /// Data alignment (bytes)
    pub data_alignment: u32,
    /// Function alignment (bytes)
    pub function_alignment: u32,
    /// SIMD alignment (bytes)
    pub simd_alignment: u32,
}

/// Progressive loading configuration
#[derive(Debug, Clone)]
pub struct ProgressiveLoadingConfig {
    /// Enable progressive loading
    pub enable: bool,
    /// Loading strategy
    pub strategy: LoadingStrategy,
    /// Chunk size in bytes
    pub chunk_size: usize,
    /// Preloading configuration
    pub preloading: PreloadingConfig,
    /// Enable streaming
    pub streaming: bool,
}

/// Loading strategy for progressive loading
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingStrategy {
    /// Load all at once
    Eager,
    /// Load on demand
    Lazy,
    /// Load in chunks
    Chunked,
    /// Stream continuously
    Streaming,
}

/// Preloading configuration
#[derive(Debug, Clone)]
pub struct PreloadingConfig {
    /// Enable preloading
    pub enable: bool,
    /// Preload percentage (0.0 to 1.0)
    pub percentage: f64,
    /// Preload on idle
    pub on_idle: bool,
    /// Preload based on user interaction
    pub on_interaction: bool,
}

/// Caching configuration
#[derive(Debug, Clone)]
pub struct CachingConfig {
    /// Enable caching
    pub enable: bool,
    /// Cache strategy
    pub strategy: CacheStrategy,
    /// Storage backend
    pub storage: CacheStorage,
    /// Time to live in seconds
    pub ttl_seconds: Option<u64>,
    /// Versioning strategy
    pub versioning: VersioningStrategy,
}

/// Cache strategy
#[derive(Debug, Clone, PartialEq)]
pub enum CacheStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Time-based expiration
    TTL,
    /// Custom strategy
    Custom,
}

/// Cache storage backend
#[derive(Debug, Clone, PartialEq)]
pub enum CacheStorage {
    /// Browser Cache API
    CacheAPI,
    /// IndexedDB
    IndexedDB,
    /// Local Storage
    LocalStorage,
    /// Session Storage
    SessionStorage,
    /// In-memory only
    Memory,
}

/// Versioning strategy for cache
#[derive(Debug, Clone, PartialEq)]
pub enum VersioningStrategy {
    /// Use content hash
    Hash,
    /// Use timestamp
    Timestamp,
    /// Use semantic version
    Semantic,
    /// Custom versioning
    Custom(String),
}

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable Web Workers
    pub web_workers: bool,
    /// Maximum number of workers
    pub max_workers: Option<usize>,
    /// Shared memory support
    pub shared_memory: bool,
    /// Work stealing algorithm
    pub work_stealing: bool,
}

/// Memory export specification
#[derive(Debug, Clone)]
pub struct WasmMemoryExport {
    /// Export name
    pub name: String,
    /// Memory configuration
    pub config: WasmMemoryConfig,
}

/// Memory import specification
#[derive(Debug, Clone)]
pub struct WasmMemoryImport {
    /// Module name
    pub module: String,
    /// Memory name
    pub name: String,
    /// Memory configuration
    pub config: WasmMemoryConfig,
}

impl Default for WasmMemoryConfig {
    fn default() -> Self {
        Self {
            initial_pages: 256,        // 16MB initial
            maximum_pages: Some(1024), // 64MB maximum
            shared: false,
            growth_strategy: MemoryGrowthStrategy::OnDemand,
            alignment: MemoryAlignment::default(),
        }
    }
}

impl Default for MemoryAlignment {
    fn default() -> Self {
        Self {
            data_alignment: 8,      // 8-byte alignment for f64
            function_alignment: 16, // 16-byte alignment for functions
            simd_alignment: 16,     // 16-byte alignment for SIMD
        }
    }
}

impl Default for ProgressiveLoadingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            strategy: LoadingStrategy::Lazy,
            chunk_size: 1024 * 1024, // 1MB chunks
            preloading: PreloadingConfig::default(),
            streaming: true,
        }
    }
}

impl Default for PreloadingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            percentage: 0.1, // Preload 10%
            on_idle: true,
            on_interaction: false,
        }
    }
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            strategy: CacheStrategy::LRU,
            storage: CacheStorage::CacheAPI,
            ttl_seconds: Some(3600), // 1 hour
            versioning: VersioningStrategy::Hash,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            web_workers: true,
            max_workers: Some(4),
            shared_memory: false,
            work_stealing: false,
        }
    }
}

impl WasmMemoryConfig {
    /// Create a new memory configuration
    pub fn new(initial_pages: u32, maximum_pages: Option<u32>) -> Self {
        Self {
            initial_pages,
            maximum_pages,
            shared: false,
            growth_strategy: MemoryGrowthStrategy::OnDemand,
            alignment: MemoryAlignment::default(),
        }
    }

    /// Create a configuration for small models
    pub fn small() -> Self {
        Self {
            initial_pages: 64,        // 4MB initial
            maximum_pages: Some(256), // 16MB maximum
            shared: false,
            growth_strategy: MemoryGrowthStrategy::Fixed,
            alignment: MemoryAlignment::default(),
        }
    }

    /// Create a configuration for large models
    pub fn large() -> Self {
        Self {
            initial_pages: 512,        // 32MB initial
            maximum_pages: Some(4096), // 256MB maximum
            shared: false,
            growth_strategy: MemoryGrowthStrategy::Streaming,
            alignment: MemoryAlignment::high_performance(),
        }
    }

    /// Create a configuration for multi-threaded execution
    pub fn multithreaded() -> Self {
        Self {
            initial_pages: 256,        // 16MB initial
            maximum_pages: Some(2048), // 128MB maximum
            shared: true,
            growth_strategy: MemoryGrowthStrategy::PreAllocated,
            alignment: MemoryAlignment::default(),
        }
    }

    /// Get total initial memory size in bytes
    pub fn initial_size_bytes(&self) -> usize {
        self.initial_pages as usize * 65536 // 64KB per page
    }

    /// Get maximum memory size in bytes
    pub fn max_size_bytes(&self) -> Option<usize> {
        self.maximum_pages.map(|pages| pages as usize * 65536)
    }

    /// Check if configuration supports growth
    pub fn supports_growth(&self) -> bool {
        self.growth_strategy != MemoryGrowthStrategy::Fixed
    }

    /// Check if configuration is suitable for large models
    pub fn is_large_model_config(&self) -> bool {
        self.initial_size_bytes() >= 32 * 1024 * 1024 // 32MB or more
    }
}

impl MemoryAlignment {
    /// Create alignment configuration optimized for performance
    pub fn high_performance() -> Self {
        Self {
            data_alignment: 32,     // Cache line alignment
            function_alignment: 32, // Optimal function alignment
            simd_alignment: 32,     // AVX alignment
        }
    }

    /// Create alignment configuration optimized for size
    pub fn compact() -> Self {
        Self {
            data_alignment: 4,     // Minimal alignment
            function_alignment: 8, // Minimal function alignment
            simd_alignment: 16,    // Standard SIMD alignment
        }
    }

    /// Check if alignment is compatible with SIMD operations
    pub fn is_simd_compatible(&self) -> bool {
        self.simd_alignment >= 16
    }

    /// Check if alignment is optimized for cache performance
    pub fn is_cache_optimized(&self) -> bool {
        self.data_alignment >= 32 && self.function_alignment >= 32
    }
}

impl ProgressiveLoadingConfig {
    /// Create configuration for fast initial loading
    pub fn fast_start() -> Self {
        Self {
            enable: true,
            strategy: LoadingStrategy::Lazy,
            chunk_size: 512 * 1024, // 512KB chunks
            preloading: PreloadingConfig {
                enable: true,
                percentage: 0.05, // Preload 5%
                on_idle: true,
                on_interaction: true,
            },
            streaming: true,
        }
    }

    /// Create configuration for bandwidth-constrained environments
    pub fn low_bandwidth() -> Self {
        Self {
            enable: true,
            strategy: LoadingStrategy::Chunked,
            chunk_size: 128 * 1024, // 128KB chunks
            preloading: PreloadingConfig {
                enable: false,
                percentage: 0.0,
                on_idle: false,
                on_interaction: false,
            },
            streaming: true,
        }
    }

    /// Check if preloading is enabled
    pub fn has_preloading(&self) -> bool {
        self.preloading.enable
    }

    /// Get estimated memory overhead for preloading
    pub fn preload_memory_overhead(&self, total_size: usize) -> usize {
        if self.has_preloading() {
            (total_size as f64 * self.preloading.percentage) as usize
        } else {
            0
        }
    }
}

impl CachingConfig {
    /// Create configuration for aggressive caching
    pub fn aggressive() -> Self {
        Self {
            enable: true,
            strategy: CacheStrategy::LRU,
            storage: CacheStorage::IndexedDB,
            ttl_seconds: Some(7 * 24 * 3600), // 1 week
            versioning: VersioningStrategy::Hash,
        }
    }

    /// Create configuration for minimal caching
    pub fn minimal() -> Self {
        Self {
            enable: true,
            strategy: CacheStrategy::TTL,
            storage: CacheStorage::Memory,
            ttl_seconds: Some(300), // 5 minutes
            versioning: VersioningStrategy::Timestamp,
        }
    }

    /// Check if persistent storage is used
    pub fn uses_persistent_storage(&self) -> bool {
        matches!(
            self.storage,
            CacheStorage::CacheAPI | CacheStorage::IndexedDB | CacheStorage::LocalStorage
        )
    }

    /// Get estimated cache lifetime in seconds
    pub fn cache_lifetime(&self) -> Option<u64> {
        self.ttl_seconds
    }
}

impl ParallelConfig {
    /// Create configuration for maximum parallelism
    pub fn max_parallel() -> Self {
        Self {
            web_workers: true,
            max_workers: Some(navigator_hardware_concurrency().unwrap_or(8)),
            shared_memory: true,
            work_stealing: true,
        }
    }

    /// Create configuration for single-threaded execution
    pub fn single_threaded() -> Self {
        Self {
            web_workers: false,
            max_workers: Some(1),
            shared_memory: false,
            work_stealing: false,
        }
    }

    /// Get effective number of workers
    pub fn effective_workers(&self) -> usize {
        if self.web_workers {
            self.max_workers.unwrap_or(1)
        } else {
            1
        }
    }

    /// Check if configuration supports multi-threading
    pub fn supports_multithreading(&self) -> bool {
        self.web_workers && self.effective_workers() > 1
    }
}

/// Utility function to get navigator hardware concurrency (mock for server-side)
fn navigator_hardware_concurrency() -> Option<usize> {
    // In real implementation, this would access navigator.hardwareConcurrency
    // For now, return a reasonable default
    Some(4)
}

/// Memory manager for WebAssembly models
pub struct MemoryManager {
    config: WasmMemoryConfig,
    progressive_config: ProgressiveLoadingConfig,
    cache_config: CachingConfig,
    parallel_config: ParallelConfig,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        config: WasmMemoryConfig,
        progressive_config: ProgressiveLoadingConfig,
        cache_config: CachingConfig,
        parallel_config: ParallelConfig,
    ) -> Self {
        Self {
            config,
            progressive_config,
            cache_config,
            parallel_config,
        }
    }

    /// Create a memory manager optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            config: WasmMemoryConfig::large(),
            progressive_config: ProgressiveLoadingConfig::fast_start(),
            cache_config: CachingConfig::aggressive(),
            parallel_config: ParallelConfig::max_parallel(),
        }
    }

    /// Create a memory manager optimized for low resource usage
    pub fn resource_constrained() -> Self {
        Self {
            config: WasmMemoryConfig::small(),
            progressive_config: ProgressiveLoadingConfig::low_bandwidth(),
            cache_config: CachingConfig::minimal(),
            parallel_config: ParallelConfig::single_threaded(),
        }
    }

    /// Get memory configuration
    pub fn memory_config(&self) -> &WasmMemoryConfig {
        &self.config
    }

    /// Get progressive loading configuration
    pub fn progressive_config(&self) -> &ProgressiveLoadingConfig {
        &self.progressive_config
    }

    /// Get caching configuration
    pub fn cache_config(&self) -> &CachingConfig {
        &self.cache_config
    }

    /// Get parallel configuration
    pub fn parallel_config(&self) -> &ParallelConfig {
        &self.parallel_config
    }

    /// Calculate total memory requirements
    pub fn calculate_memory_requirements(&self, model_size: usize) -> MemoryRequirements {
        let base_memory = self.config.initial_size_bytes();
        let model_memory = model_size;
        let cache_overhead = if self.cache_config.enable {
            model_size / 10 // 10% overhead for caching
        } else {
            0
        };
        let preload_overhead = self.progressive_config.preload_memory_overhead(model_size);
        let worker_overhead = if self.parallel_config.supports_multithreading() {
            model_size * self.parallel_config.effective_workers() / 4 // 25% per worker
        } else {
            0
        };

        MemoryRequirements {
            base_memory,
            model_memory,
            cache_overhead,
            preload_overhead,
            worker_overhead,
            total: base_memory + model_memory + cache_overhead + preload_overhead + worker_overhead,
        }
    }

    /// Check if configuration is suitable for given model size
    pub fn is_suitable_for_model(&self, model_size: usize) -> bool {
        let requirements = self.calculate_memory_requirements(model_size);

        if let Some(max_size) = self.config.max_size_bytes() {
            requirements.total <= max_size
        } else {
            true // No limit
        }
    }

    /// Get recommended chunk size for streaming
    pub fn recommended_chunk_size(&self, model_size: usize) -> usize {
        let base_chunk = self.progressive_config.chunk_size;

        // Adjust chunk size based on model size
        if model_size < 1024 * 1024 {
            // Small models: use smaller chunks
            base_chunk / 4
        } else if model_size > 100 * 1024 * 1024 {
            // Large models: use larger chunks
            base_chunk * 2
        } else {
            base_chunk
        }
    }
}

/// Memory requirements calculation result
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Base WebAssembly memory
    pub base_memory: usize,
    /// Memory for model data
    pub model_memory: usize,
    /// Cache overhead
    pub cache_overhead: usize,
    /// Preloading overhead
    pub preload_overhead: usize,
    /// Worker overhead
    pub worker_overhead: usize,
    /// Total memory requirement
    pub total: usize,
}

impl MemoryRequirements {
    /// Get memory usage breakdown as percentages
    pub fn breakdown_percentages(&self) -> MemoryBreakdown {
        let total_f = self.total as f64;

        MemoryBreakdown {
            base_percent: (self.base_memory as f64 / total_f) * 100.0,
            model_percent: (self.model_memory as f64 / total_f) * 100.0,
            cache_percent: (self.cache_overhead as f64 / total_f) * 100.0,
            preload_percent: (self.preload_overhead as f64 / total_f) * 100.0,
            worker_percent: (self.worker_overhead as f64 / total_f) * 100.0,
        }
    }

    /// Format memory size as human-readable string
    pub fn format_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

        if bytes == 0 {
            return "0 B".to_string();
        }

        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }

    /// Get formatted total memory requirement
    pub fn total_formatted(&self) -> String {
        Self::format_size(self.total)
    }
}

/// Memory usage breakdown in percentages
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Percentage of memory for base operations
    pub base_percent: f64,
    /// Percentage of memory for model storage
    pub model_percent: f64,
    /// Percentage of memory for caching
    pub cache_percent: f64,
    /// Percentage of memory for preloading
    pub preload_percent: f64,
    /// Percentage of memory for worker threads
    pub worker_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_memory_config_default() {
        let config = WasmMemoryConfig::default();
        assert_eq!(config.initial_pages, 256);
        assert_eq!(config.maximum_pages, Some(1024));
        assert!(!config.shared);
        assert_eq!(config.growth_strategy, MemoryGrowthStrategy::OnDemand);
    }

    #[test]
    fn test_memory_config_sizes() {
        let config = WasmMemoryConfig::new(128, Some(512));
        assert_eq!(config.initial_size_bytes(), 128 * 65536);
        assert_eq!(config.max_size_bytes(), Some(512 * 65536));
        assert!(config.supports_growth());
    }

    #[test]
    fn test_memory_config_presets() {
        let small = WasmMemoryConfig::small();
        assert_eq!(small.initial_pages, 64);
        assert_eq!(small.growth_strategy, MemoryGrowthStrategy::Fixed);
        assert!(!small.supports_growth());

        let large = WasmMemoryConfig::large();
        assert_eq!(large.initial_pages, 512);
        assert_eq!(large.growth_strategy, MemoryGrowthStrategy::Streaming);
        assert!(large.is_large_model_config());

        let mt = WasmMemoryConfig::multithreaded();
        assert!(mt.shared);
        assert_eq!(mt.growth_strategy, MemoryGrowthStrategy::PreAllocated);
    }

    #[test]
    fn test_memory_alignment() {
        let default_align = MemoryAlignment::default();
        assert!(default_align.is_simd_compatible());
        assert!(!default_align.is_cache_optimized());

        let perf_align = MemoryAlignment::high_performance();
        assert!(perf_align.is_simd_compatible());
        assert!(perf_align.is_cache_optimized());

        let compact_align = MemoryAlignment::compact();
        assert!(compact_align.is_simd_compatible());
        assert!(!compact_align.is_cache_optimized());
    }

    #[test]
    fn test_progressive_loading_config() {
        let config = ProgressiveLoadingConfig::default();
        assert!(config.enable);
        assert!(config.has_preloading());

        let fast = ProgressiveLoadingConfig::fast_start();
        assert_eq!(fast.chunk_size, 512 * 1024);
        assert!(fast.preloading.on_interaction);

        let low_bw = ProgressiveLoadingConfig::low_bandwidth();
        assert_eq!(low_bw.chunk_size, 128 * 1024);
        assert!(!low_bw.has_preloading());
    }

    #[test]
    fn test_caching_config() {
        let default_cache = CachingConfig::default();
        assert!(default_cache.enable);
        assert!(default_cache.uses_persistent_storage());
        assert_eq!(default_cache.cache_lifetime(), Some(3600));

        let aggressive = CachingConfig::aggressive();
        assert_eq!(aggressive.strategy, CacheStrategy::LRU);
        assert_eq!(aggressive.cache_lifetime(), Some(7 * 24 * 3600));

        let minimal = CachingConfig::minimal();
        assert_eq!(minimal.strategy, CacheStrategy::TTL);
        assert!(!minimal.uses_persistent_storage());
    }

    #[test]
    fn test_parallel_config() {
        let default_parallel = ParallelConfig::default();
        assert!(default_parallel.web_workers);
        assert!(default_parallel.supports_multithreading());

        let max_parallel = ParallelConfig::max_parallel();
        assert!(max_parallel.work_stealing);
        assert!(max_parallel.shared_memory);

        let single = ParallelConfig::single_threaded();
        assert!(!single.web_workers);
        assert!(!single.supports_multithreading());
        assert_eq!(single.effective_workers(), 1);
    }

    #[test]
    fn test_memory_manager() {
        let manager = MemoryManager::performance_optimized();
        assert!(manager.memory_config().is_large_model_config());
        assert!(manager.parallel_config().supports_multithreading());

        let constrained = MemoryManager::resource_constrained();
        assert!(!constrained.memory_config().is_large_model_config());
        assert!(!constrained.parallel_config().supports_multithreading());
    }

    #[test]
    fn test_memory_requirements_calculation() {
        let manager = MemoryManager::performance_optimized();
        let model_size = 10 * 1024 * 1024; // 10MB model

        let requirements = manager.calculate_memory_requirements(model_size);
        assert!(requirements.total > model_size);
        assert!(requirements.model_memory == model_size);

        let breakdown = requirements.breakdown_percentages();
        assert!(
            (breakdown.base_percent
                + breakdown.model_percent
                + breakdown.cache_percent
                + breakdown.preload_percent
                + breakdown.worker_percent
                - 100.0)
                .abs()
                < 0.1
        );
    }

    #[test]
    fn test_memory_requirements_formatting() {
        assert_eq!(MemoryRequirements::format_size(0), "0 B");
        assert_eq!(MemoryRequirements::format_size(1024), "1.0 KB");
        assert_eq!(MemoryRequirements::format_size(1024 * 1024), "1.0 MB");
        assert_eq!(MemoryRequirements::format_size(1536), "1.5 KB");
    }

    #[test]
    fn test_recommended_chunk_size() {
        let manager = MemoryManager::performance_optimized();

        // Small model should get smaller chunks
        let small_chunk = manager.recommended_chunk_size(512 * 1024);
        assert!(small_chunk < manager.progressive_config().chunk_size);

        // Large model should get larger chunks
        let large_chunk = manager.recommended_chunk_size(200 * 1024 * 1024);
        assert!(large_chunk > manager.progressive_config().chunk_size);

        // Medium model should get default chunks
        let medium_chunk = manager.recommended_chunk_size(10 * 1024 * 1024);
        assert_eq!(medium_chunk, manager.progressive_config().chunk_size);
    }
}
