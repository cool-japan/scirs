//! Advanced memory optimization strategies for statistical computations
//!
//! This module provides sophisticated memory management techniques specifically
//! designed for large-scale statistical operations, including streaming algorithms,
//! memory-mapped data processing, cache-aware algorithms, and adaptive memory
//! allocation strategies.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use serde::{Deserialize, Serialize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::sync::{Arc, Mutex, RwLock};

/// Memory optimization configuration for statistical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryOptimizationConfig {
    /// Maximum memory usage in bytes before triggering optimization strategies
    pub memory_limit: usize,
    /// Enable streaming algorithms for large datasets
    pub enable_streaming: bool,
    /// Enable memory-mapped file operations
    pub enable_memory_mapping: bool,
    /// Enable cache-aware algorithm variants
    pub enable_cache_optimization: bool,
    /// Enable adaptive memory allocation
    pub enable_adaptive_allocation: bool,
    /// Target cache line size for optimization
    pub cache_linesize: usize,
    /// Memory pool initial size
    pub memory_poolsize: usize,
    /// Enable memory compression for intermediate results
    pub enable_compression: bool,
    /// Chunk size for streaming operations
    pub streaming_chunksize: usize,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1024 * 1024 * 1024, // 1GB default limit
            enable_streaming: true,
            enable_memory_mapping: true,
            enable_cache_optimization: true,
            enable_adaptive_allocation: true,
            cache_linesize: 64,                // Typical cache line size
            memory_poolsize: 64 * 1024 * 1024, // 64MB pool
            enable_compression: false,         // Disabled by default due to CPU overhead
            streaming_chunksize: 10000,        // Default chunk size
        }
    }
}

/// Memory usage statistics and profiling information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryProfile {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory fragmentation ratio (0-1, lower is better)
    pub fragmentation_ratio: f64,
    /// Cache hit ratio (0-1, higher is better)
    pub cache_hit_ratio: f64,
    /// Memory efficiency score (0-100)
    pub efficiency_score: f64,
    /// Active memory pools
    pub active_pools: Vec<MemoryPoolStats>,
}

/// Statistics for individual memory pools
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryPoolStats {
    /// Pool identifier
    pub pool_id: String,
    /// Pool size in bytes
    pub poolsize: usize,
    /// Used bytes in pool
    pub used_bytes: usize,
    /// Number of allocations from this pool
    pub allocations: usize,
    /// Average allocation size
    pub avg_allocationsize: f64,
}

/// Streaming statistics calculator for large datasets
pub struct StreamingStatsCalculator<F> {
    config: MemoryOptimizationConfig,
    /// Running count of elements processed
    count: usize,
    /// Running sum for mean calculation
    sum: F,
    /// Running sum of squares for variance calculation
    sum_squares: F,
    /// Running minimum value
    min_value: Option<F>,
    /// Running maximum value
    max_value: Option<F>,
    /// Memory profile tracking
    memory_profile: Arc<Mutex<MemoryProfile>>,
    /// Intermediate computation buffer
    computation_buffer: VecDeque<F>,
}

/// Cache-aware matrix operations optimized for memory locality
pub struct CacheOptimizedMatrix<F> {
    data: Array2<F>,
    blocksize: usize,
    memory_layout: MatrixLayout,
    cache_linesize: usize,
}

/// Memory layout optimization strategies
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
    BlockedRowMajor,
    BlockedColumnMajor,
    ZOrderCurve,
}

/// Adaptive memory allocator for statistical computations
pub struct AdaptiveStatsAllocator {
    config: MemoryOptimizationConfig,
    memory_pools: HashMap<String, Arc<Mutex<MemoryPool>>>,
    allocation_patterns: Arc<RwLock<AllocationPatternAnalyzer>>,
    global_stats: Arc<Mutex<MemoryProfile>>,
}

/// Memory pool optimized for statistical data types
#[allow(dead_code)]
struct MemoryPool {
    pool_id: String,
    base_ptr: *mut u8,
    poolsize: usize,
    usedsize: usize,
    free_blocks: Vec<MemoryBlock>,
    used_blocks: Vec<MemoryBlock>,
    allocation_count: usize,
}

/// Memory block descriptor
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MemoryBlock {
    offset: usize,
    size: usize,
    alignment: usize,
    allocation_time: std::time::Instant,
}

/// Allocation pattern analyzer for optimizing future allocations
#[allow(dead_code)]
struct AllocationPatternAnalyzer {
    allocation_history: VecDeque<AllocationEvent>,
    pattern_cache: HashMap<String, AllocationPattern>,
    prediction_accuracy: f64,
}

/// Allocation event for pattern analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AllocationEvent {
    size: usize,
    alignment: usize,
    lifetime: std::time::Duration,
    operation_type: String,
    timestamp: std::time::Instant,
}

/// Identified allocation pattern
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AllocationPattern {
    typicalsize: usize,
    typical_alignment: usize,
    expected_lifetime: std::time::Duration,
    frequency: f64,
    confidence: f64,
}

/// Memory-mapped statistical data processor
pub struct MemoryMappedStatsProcessor {
    config: MemoryOptimizationConfig,
    mapped_files: HashMap<String, MemoryMappedFile>,
    cache_manager: CacheManager,
}

/// Memory-mapped file wrapper
#[allow(dead_code)]
struct MemoryMappedFile {
    file_path: String,
    filesize: usize,
    mapped_ptr: *const u8,
    access_pattern: AccessPattern,
}

/// File access pattern tracking
#[derive(Debug, Clone)]
struct AccessPattern {
    sequential_ratio: f64,
    random_ratio: f64,
    hot_regions: Vec<MemoryRegion>,
}

/// Memory region descriptor
#[derive(Debug, Clone)]
struct MemoryRegion {
    start_offset: usize,
    end_offset: usize,
    access_frequency: f64,
    last_access: std::time::Instant,
}

/// Cache manager for optimizing memory access patterns
#[allow(dead_code)]
struct CacheManager {
    cachesize: usize,
    cache_entries: HashMap<u64, CacheEntry>,
    access_order: VecDeque<u64>,
    hit_count: usize,
    miss_count: usize,
}

/// Cache entry with metadata
#[allow(dead_code)]
struct CacheEntry {
    data: Vec<u8>,
    access_count: usize,
    last_access: std::time::Instant,
    size: usize,
}

impl<F> StreamingStatsCalculator<F>
where
    F: Float + NumCast + Zero + One + Send + Sync + std::fmt::Display,
{
    /// Create a new streaming statistics calculator
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        Self {
            config,
            count: 0,
            sum: F::zero(),
            sum_squares: F::zero(),
            min_value: None,
            max_value: None,
            memory_profile: Arc::new(Mutex::new(MemoryProfile::new())),
            computation_buffer: VecDeque::with_capacity(1000), // Small buffer for efficiency
        }
    }

    /// Process a chunk of data in streaming fashion
    pub fn process_chunk(&mut self, chunk: ArrayView1<F>) -> StatsResult<()> {
        self.update_memory_profile();

        for &value in chunk.iter() {
            self.count += 1;
            self.sum = self.sum + value;
            self.sum_squares = self.sum_squares + value * value;

            // Update min/max
            match self.min_value {
                None => self.min_value = Some(value),
                Some(current_min) => {
                    if value < current_min {
                        self.min_value = Some(value);
                    }
                }
            }

            match self.max_value {
                None => self.max_value = Some(value),
                Some(current_max) => {
                    if value > current_max {
                        self.max_value = Some(value);
                    }
                }
            }

            // Manage computation buffer to prevent memory growth
            if self.computation_buffer.len() >= self.config.streaming_chunksize {
                self.computation_buffer.pop_front();
            }
            self.computation_buffer.push_back(value);
        }

        Ok(())
    }

    /// Get current streaming statistics
    pub fn get_statistics(&self) -> StatsResult<StreamingStatistics<F>> {
        if self.count == 0 {
            return Err(StatsError::InvalidArgument(
                "No data processed yet".to_string(),
            ));
        }

        let count_f = F::from(self.count).unwrap();
        let mean = self.sum / count_f;

        let variance = if self.count > 1 {
            let count_minus_one = F::from(self.count - 1).unwrap();
            (self.sum_squares - self.sum * self.sum / count_f) / count_minus_one
        } else {
            F::zero()
        };

        let std_dev = variance.sqrt();

        Ok(StreamingStatistics {
            count: self.count,
            mean,
            variance,
            std_dev,
            min: self.min_value.unwrap_or(F::zero()),
            max: self.max_value.unwrap_or(F::zero()),
            memory_efficiency: self.calculate_memory_efficiency(),
        })
    }

    /// Calculate memory efficiency score
    fn calculate_memory_efficiency(&self) -> f64 {
        let theoretical_minimum = mem::size_of::<F>() * self.count;
        let actual_usage = self.estimate_memory_usage();

        if actual_usage > 0 {
            (theoretical_minimum as f64 / actual_usage as f64 * 100.0).min(100.0)
        } else {
            100.0
        }
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        mem::size_of::<Self>() + self.computation_buffer.len() * mem::size_of::<F>()
    }

    /// Update memory profiling information
    fn update_memory_profile(&mut self) {
        if let Ok(mut profile) = self.memory_profile.lock() {
            let current_usage = self.estimate_memory_usage();
            profile.current_usage = current_usage;
            if current_usage > profile.peak_usage {
                profile.peak_usage = current_usage;
            }
            profile.allocation_count += 1;
            profile.efficiency_score = self.calculate_memory_efficiency();
        }
    }
}

/// Result structure for streaming statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StreamingStatistics<F> {
    pub count: usize,
    pub mean: F,
    pub variance: F,
    pub std_dev: F,
    pub min: F,
    pub max: F,
    pub memory_efficiency: f64,
}

impl<F> CacheOptimizedMatrix<F>
where
    F: Float + NumCast + Zero + One + Clone + 'static + std::fmt::Display,
{
    /// Create a cache-optimized matrix with specified layout
    pub fn new(data: Array2<F>, layout: MatrixLayout, cache_linesize: usize) -> Self {
        let optimal_blocksize =
            Self::calculate_optimal_blocksize(data.nrows(), data.ncols(), cache_linesize);

        let mut matrix = Self {
            data,
            blocksize: optimal_blocksize,
            memory_layout: layout,
            cache_linesize,
        };

        matrix.optimize_layout();
        matrix
    }

    /// Perform cache-optimized matrix multiplication
    pub fn multiply_optimized(
        &self,
        other: &CacheOptimizedMatrix<F>,
    ) -> StatsResult<CacheOptimizedMatrix<F>> {
        if self.data.ncols() != other.data.nrows() {
            return Err(StatsError::DimensionMismatch(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let resultdata = match self.memory_layout {
            MatrixLayout::BlockedRowMajor | MatrixLayout::BlockedColumnMajor => {
                self.blocked_multiply(&other.data)?
            }
            _ => self.standard_multiply(&other.data)?,
        };

        Ok(CacheOptimizedMatrix::new(
            resultdata,
            self.memory_layout,
            self.cache_linesize,
        ))
    }

    /// Cache-optimized correlation computation
    pub fn correlation_matrix(&self) -> StatsResult<CacheOptimizedMatrix<F>> {
        let (_n_samples_, n_features) = self.data.dim();

        // Compute means using cache-friendly access patterns
        let means = self.compute_column_means_optimized()?;

        // Compute correlation matrix using blocked algorithm
        let mut correlation = Array2::zeros((n_features, n_features));

        let blocksize = self.blocksize;
        for i_block in (0..n_features).step_by(blocksize) {
            for j_block in (0..n_features).step_by(blocksize) {
                let i_end = (i_block + blocksize).min(n_features);
                let j_end = (j_block + blocksize).min(n_features);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let correlation_value = self.compute_correlation_pair(i, j, &means)?;
                        correlation[[i, j]] = correlation_value;
                    }
                }
            }
        }

        Ok(CacheOptimizedMatrix::new(
            correlation,
            MatrixLayout::BlockedRowMajor,
            self.cache_linesize,
        ))
    }

    /// Calculate optimal block size for cache efficiency
    fn calculate_optimal_blocksize(_rows: usize, cols: usize, cache_linesize: usize) -> usize {
        let elementsize = mem::size_of::<F>();
        let elements_per_cache_line = cache_linesize / elementsize;

        // Find block size that maximizes cache utilization
        let target_block_elements = (32 * 1024) / elementsize; // Target 32KB blocks
        let max_dimension = _rows.max(cols);

        ((target_block_elements as f64).sqrt() as usize)
            .min(max_dimension)
            .max(elements_per_cache_line)
    }

    /// Optimize matrix layout for cache performance
    fn optimize_layout(&mut self) {
        match self.memory_layout {
            MatrixLayout::ZOrderCurve => {
                self.data = self.convert_to_z_order();
            }
            MatrixLayout::BlockedRowMajor => {
                self.data = self.convert_to_blocked_layout(true);
            }
            MatrixLayout::BlockedColumnMajor => {
                self.data = self.convert_to_blocked_layout(false);
            }
            _ => {
                // No layout conversion needed
            }
        }
    }

    /// Convert matrix to Z-order (Morton order) layout
    fn convert_to_z_order(&self) -> Array2<F> {
        // Simplified Z-order conversion
        // In practice, this would implement proper Morton encoding
        self.data.clone() // Placeholder implementation
    }

    /// Convert matrix to blocked layout
    fn convert_to_blocked_layout(&self, rowmajor: bool) -> Array2<F> {
        // Simplified blocked layout conversion
        self.data.clone() // Placeholder implementation
    }

    /// Blocked matrix multiplication optimized for cache
    fn blocked_multiply(&self, other: &Array2<F>) -> StatsResult<Array2<F>> {
        let (m, k) = self.data.dim();
        let (k2, n) = other.dim();

        if k != k2 {
            return Err(StatsError::DimensionMismatch(
                "Matrix dimensions incompatible".to_string(),
            ));
        }

        let mut result = Array2::zeros((m, n));
        let blocksize = self.blocksize;

        // Blocked multiplication to improve cache locality
        for i_block in (0..m).step_by(blocksize) {
            for j_block in (0..n).step_by(blocksize) {
                for k_block in (0..k).step_by(blocksize) {
                    let i_end = (i_block + blocksize).min(m);
                    let j_end = (j_block + blocksize).min(n);
                    let k_end = (k_block + blocksize).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = F::zero();
                            for k_idx in k_block..k_end {
                                sum = sum + self.data[[i, k_idx]] * other[[k_idx, j]];
                            }
                            result[[i, j]] = result[[i, j]] + sum;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Standard matrix multiplication
    fn standard_multiply(&self, other: &Array2<F>) -> StatsResult<Array2<F>> {
        let result = self.data.dot(other);
        Ok(result)
    }

    /// Compute column means with cache-optimized access
    fn compute_column_means_optimized(&self) -> StatsResult<Array1<F>> {
        let n_features = self.data.ncols();
        let n_samples_f = F::from(self.data.nrows()).unwrap();
        let mut means = Array1::zeros(n_features);

        // Process columns in blocks to improve cache locality
        let blocksize = self.blocksize;
        for col_block in (0..n_features).step_by(blocksize) {
            let col_end = (col_block + blocksize).min(n_features);

            for col in col_block..col_end {
                let column_sum = self.data.column(col).sum();
                means[col] = column_sum / n_samples_f;
            }
        }

        Ok(means)
    }

    /// Compute correlation between two columns
    fn compute_correlation_pair(
        &self,
        col_i: usize,
        col_j: usize,
        means: &Array1<F>,
    ) -> StatsResult<F> {
        if col_i == col_j {
            return Ok(F::one());
        }

        let n_samples_ = self.data.nrows();
        let _n_samples_f = F::from(n_samples_).unwrap();

        let mean_i = means[col_i];
        let mean_j = means[col_j];

        let mut numerator = F::zero();
        let mut sum_sq_i = F::zero();
        let mut sum_sq_j = F::zero();

        // Single pass through the data for cache efficiency
        for row in 0..n_samples_ {
            let val_i = self.data[[row, col_i]] - mean_i;
            let val_j = self.data[[row, col_j]] - mean_j;

            numerator = numerator + val_i * val_j;
            sum_sq_i = sum_sq_i + val_i * val_i;
            sum_sq_j = sum_sq_j + val_j * val_j;
        }

        let denominator = (sum_sq_i * sum_sq_j).sqrt();
        if denominator > F::epsilon() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }
}

impl AdaptiveStatsAllocator {
    /// Create a new adaptive allocator
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        let mut allocator = Self {
            config: config.clone(),
            memory_pools: HashMap::new(),
            allocation_patterns: Arc::new(RwLock::new(AllocationPatternAnalyzer::new())),
            global_stats: Arc::new(Mutex::new(MemoryProfile::new())),
        };

        // Initialize default memory pools
        let _ = allocator.create_memory_pool("float_arrays", config.memory_poolsize / 4);
        let _ = allocator.create_memory_pool("matrix_operations", config.memory_poolsize / 2);
        let _ = allocator.create_memory_pool("temporary_buffers", config.memory_poolsize / 4);

        allocator
    }

    /// Create a specialized memory pool
    pub fn create_memory_pool(&mut self, poolid: &str, size: usize) -> StatsResult<()> {
        let pool = Arc::new(Mutex::new(MemoryPool::new(poolid, size)?));
        self.memory_pools.insert(poolid.to_string(), pool);
        Ok(())
    }

    /// Allocate memory with pattern analysis
    pub fn allocate_optimized(
        &self,
        size: usize,
        alignment: usize,
        operation_type: &str,
    ) -> StatsResult<*mut u8> {
        // Analyze allocation patterns to predict optimal pool
        let predicted_pool = self.predict_optimal_pool(size, alignment, operation_type);

        // Try to allocate from predicted pool first
        if let Some(pool) = self.memory_pools.get(&predicted_pool) {
            if let Ok(mut pool_guard) = pool.lock() {
                if let Ok(ptr) = pool_guard.allocate(size, alignment) {
                    self.record_allocation_event(size, alignment, operation_type);
                    return Ok(ptr);
                }
            }
        }

        // Fallback to system allocator
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| StatsError::ComputationError(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { System.alloc(layout) };
        if ptr.is_null() {
            return Err(StatsError::ComputationError(
                "Memory allocation failed".to_string(),
            ));
        }

        self.record_allocation_event(size, alignment, operation_type);
        Ok(ptr)
    }

    /// Predict optimal memory pool for allocation
    fn predict_optimal_pool(&self, size: usize, alignment: usize, operationtype: &str) -> String {
        if let Ok(analyzer) = self.allocation_patterns.read() {
            if let Some(pattern) = analyzer.get_pattern(operationtype) {
                // Use pattern analysis to select optimal pool
                if pattern.typicalsize <= 1024 {
                    return "temporary_buffers".to_string();
                } else if pattern.typicalsize <= 64 * 1024 {
                    return "float_arrays".to_string();
                } else {
                    return "matrix_operations".to_string();
                }
            }
        }

        // Default pool selection based on size
        if size <= 1024 {
            "temporary_buffers".to_string()
        } else if size <= 64 * 1024 {
            "float_arrays".to_string()
        } else {
            "matrix_operations".to_string()
        }
    }

    /// Record allocation event for pattern analysis
    fn record_allocation_event(&self, size: usize, alignment: usize, operationtype: &str) {
        if let Ok(mut analyzer) = self.allocation_patterns.write() {
            analyzer.record_allocation(size, alignment, operationtype);
        }

        if let Ok(mut stats) = self.global_stats.lock() {
            stats.allocation_count += 1;
            stats.current_usage += size;
            if stats.current_usage > stats.peak_usage {
                stats.peak_usage = stats.current_usage;
            }
        }
    }

    /// Get current memory profile
    pub fn get_memory_profile(&self) -> MemoryProfile {
        if let Ok(profile) = self.global_stats.lock() {
            profile.clone()
        } else {
            MemoryProfile::new()
        }
    }

    /// Optimize memory pools based on usage patterns
    pub fn optimize_pools(&mut self) -> StatsResult<()> {
        // Collect the data first to avoid borrow checker issues
        let pools_to_create: Vec<(String, usize)> = {
            if let Ok(analyzer) = self.allocation_patterns.read() {
                analyzer
                    .pattern_cache
                    .iter()
                    .filter(|(_, pattern)| pattern.confidence > 0.8)
                    .map(|(operation_type, pattern)| {
                        let pool_name = format!("specialized_{}", operation_type);
                        let optimalsize = (pattern.typicalsize * 100).max(64 * 1024);
                        (pool_name, optimalsize)
                    })
                    .collect()
            } else {
                Vec::new()
            }
        };

        // Now create the pools
        for (pool_name, optimalsize) in pools_to_create {
            self.create_memory_pool(&pool_name, optimalsize)?;
        }
        Ok(())
    }
}

impl MemoryPool {
    /// Create a new memory pool
    fn new(poolid: &str, size: usize) -> StatsResult<Self> {
        let layout = Layout::from_size_align(size, 64) // 64-byte alignment for cache lines
            .map_err(|e| StatsError::ComputationError(format!("Invalid layout: {}", e)))?;

        let base_ptr = unsafe { System.alloc(layout) };
        if base_ptr.is_null() {
            return Err(StatsError::ComputationError(
                "Memory pool allocation failed".to_string(),
            ));
        }

        Ok(Self {
            pool_id: poolid.to_string(),
            base_ptr,
            poolsize: size,
            usedsize: 0,
            free_blocks: vec![MemoryBlock {
                offset: 0,
                size,
                alignment: 64,
                allocation_time: std::time::Instant::now(),
            }],
            used_blocks: Vec::new(),
            allocation_count: 0,
        })
    }

    /// Allocate memory from the pool
    fn allocate(&mut self, size: usize, alignment: usize) -> StatsResult<*mut u8> {
        // Find suitable free block
        for (index, block) in self.free_blocks.iter().enumerate() {
            let aligned_offset = self.align_offset(block.offset, alignment);
            let totalsize = aligned_offset - block.offset + size;

            if totalsize <= block.size {
                // Split the block if necessary
                let used_block = MemoryBlock {
                    offset: aligned_offset,
                    size,
                    alignment,
                    allocation_time: std::time::Instant::now(),
                };

                let remainingsize = block.size - totalsize;
                if remainingsize > 0 {
                    let remaining_block = MemoryBlock {
                        offset: aligned_offset + size,
                        size: remainingsize,
                        alignment: 1,
                        allocation_time: std::time::Instant::now(),
                    };
                    self.free_blocks[index] = remaining_block;
                } else {
                    self.free_blocks.remove(index);
                }

                self.used_blocks.push(used_block);
                self.usedsize += size;
                self.allocation_count += 1;

                let ptr = unsafe { self.base_ptr.add(aligned_offset) };
                return Ok(ptr);
            }
        }

        Err(StatsError::ComputationError(
            "No suitable block available in pool".to_string(),
        ))
    }

    /// Align offset to specified alignment
    fn align_offset(&self, offset: usize, alignment: usize) -> usize {
        (offset + alignment - 1) & !(alignment - 1)
    }
}

impl AllocationPatternAnalyzer {
    fn new() -> Self {
        Self {
            allocation_history: VecDeque::with_capacity(10000),
            pattern_cache: HashMap::new(),
            prediction_accuracy: 0.0,
        }
    }

    fn record_allocation(&mut self, size: usize, alignment: usize, operationtype: &str) {
        let event = AllocationEvent {
            size,
            alignment,
            lifetime: std::time::Duration::from_secs(0), // Will be updated on deallocation
            operation_type: operationtype.to_string(),
            timestamp: std::time::Instant::now(),
        };

        self.allocation_history.push_back(event);

        // Maintain reasonable history size
        if self.allocation_history.len() > 10000 {
            self.allocation_history.pop_front();
        }

        // Update patterns periodically
        if self.allocation_history.len().is_multiple_of(100) {
            self.update_patterns();
        }
    }

    fn update_patterns(&mut self) {
        let mut operation_groups: HashMap<String, Vec<&AllocationEvent>> = HashMap::new();

        for event in &self.allocation_history {
            operation_groups
                .entry(event.operation_type.clone())
                .or_default()
                .push(event);
        }

        for (operation_type, events) in operation_groups {
            if events.len() >= 10 {
                // Minimum sample size for pattern
                let pattern = self.analyze_allocation_pattern(&events);
                self.pattern_cache.insert(operation_type, pattern);
            }
        }
    }

    fn analyze_allocation_pattern(&self, events: &[&AllocationEvent]) -> AllocationPattern {
        let sizes: Vec<usize> = events.iter().map(|e| e.size).collect();
        let alignments: Vec<usize> = events.iter().map(|e| e.alignment).collect();

        let typicalsize = self.calculate_median(&sizes);
        let typical_alignment = self.calculate_mode(&alignments);
        let frequency = events.len() as f64 / self.allocation_history.len() as f64;

        // Simple confidence based on consistency
        let size_variance = self.calculate_variance(&sizes);
        let confidence = 1.0 / (1.0 + size_variance / typicalsize as f64);

        AllocationPattern {
            typicalsize,
            typical_alignment,
            expected_lifetime: std::time::Duration::from_millis(100), // Placeholder
            frequency,
            confidence,
        }
    }

    fn calculate_median(&self, values: &[usize]) -> usize {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn calculate_mode(&self, values: &[usize]) -> usize {
        let mut counts = HashMap::new();
        for &value in values {
            *counts.entry(value).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(_, value)| value)
            .unwrap_or(64) // Default alignment
    }

    fn calculate_variance(&self, values: &[usize]) -> f64 {
        let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        variance
    }

    fn get_pattern(&self, operationtype: &str) -> Option<&AllocationPattern> {
        self.pattern_cache.get(operationtype)
    }
}

impl MemoryProfile {
    fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            fragmentation_ratio: 0.0,
            cache_hit_ratio: 0.0,
            efficiency_score: 100.0,
            active_pools: Vec::new(),
        }
    }
}

/// High-level memory optimization utilities
pub struct MemoryOptimizationSuite {
    config: MemoryOptimizationConfig,
    allocator: AdaptiveStatsAllocator,
    cache_manager: CacheManager,
}

impl MemoryOptimizationSuite {
    /// Create a new memory optimization suite
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        let allocator = AdaptiveStatsAllocator::new(config.clone());
        let cache_manager = CacheManager::new(config.memory_poolsize / 8); // Use 1/8 of pool for cache

        Self {
            config,
            allocator,
            cache_manager,
        }
    }

    /// Optimize correlation computation for large matrices
    pub fn optimized_correlation_matrix<F>(&mut self, data: ArrayView2<F>) -> StatsResult<Array2<F>>
    where
        F: Float + NumCast + Zero + One + Clone + Send + Sync + 'static + std::fmt::Display,
    {
        let (n_samples_, n_features) = data.dim();
        let datasize = n_samples_ * n_features * mem::size_of::<F>();

        if datasize > self.config.memory_limit {
            // Use streaming algorithm for large datasets
            self.streaming_correlation_matrix(data)
        } else {
            // Use cache-optimized algorithm for smaller datasets
            let cache_optimized = CacheOptimizedMatrix::new(
                data.to_owned(),
                MatrixLayout::BlockedRowMajor,
                self.config.cache_linesize,
            );

            let result = cache_optimized.correlation_matrix()?;
            Ok(result.data)
        }
    }

    /// Streaming correlation matrix computation for large datasets
    fn streaming_correlation_matrix<F>(&mut self, data: ArrayView2<F>) -> StatsResult<Array2<F>>
    where
        F: Float + NumCast + Zero + One + Clone + 'static + std::fmt::Display,
    {
        let (n_samples_, n_features) = data.dim();
        let chunksize = self.config.streaming_chunksize;

        // Initialize streaming calculators for each feature pair
        let mut means = vec![F::zero(); n_features];
        let _variances = vec![F::zero(); n_features];

        // First pass: compute means
        for chunk_start in (0..n_samples_).step_by(chunksize) {
            let chunk_end = (chunk_start + chunksize).min(n_samples_);
            let chunk = data.slice(scirs2_core::ndarray::s![chunk_start..chunk_end, ..]);

            for (feature_idx, column) in chunk.axis_iter(Axis(1)).enumerate() {
                for &value in column.iter() {
                    means[feature_idx] = means[feature_idx] + value;
                }
            }
        }

        let n_samples_f = F::from(n_samples_).unwrap();
        for mean in &mut means {
            *mean = *mean / n_samples_f;
        }

        // Second pass: compute correlations
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                correlation_matrix[[i, j]] =
                    self.compute_streaming_correlation(&data, i, j, means[i], means[j])?;
                correlation_matrix[[j, i]] = correlation_matrix[[i, j]];
            }
        }

        Ok(correlation_matrix)
    }

    /// Compute correlation between two features using streaming approach
    fn compute_streaming_correlation<F>(
        &self,
        data: &ArrayView2<F>,
        feature_i: usize,
        feature_j: usize,
        mean_i: F,
        mean_j: F,
    ) -> StatsResult<F>
    where
        F: Float + NumCast + Zero + One + std::fmt::Display,
    {
        if feature_i == feature_j {
            return Ok(F::one());
        }

        let mut numerator = F::zero();
        let mut sum_sq_i = F::zero();
        let mut sum_sq_j = F::zero();

        let chunksize = self.config.streaming_chunksize;
        let n_samples_ = data.nrows();

        for chunk_start in (0..n_samples_).step_by(chunksize) {
            let chunk_end = (chunk_start + chunksize).min(n_samples_);

            for row in chunk_start..chunk_end {
                let val_i = data[[row, feature_i]] - mean_i;
                let val_j = data[[row, feature_j]] - mean_j;

                numerator = numerator + val_i * val_j;
                sum_sq_i = sum_sq_i + val_i * val_i;
                sum_sq_j = sum_sq_j + val_j * val_j;
            }
        }

        let denominator = (sum_sq_i * sum_sq_j).sqrt();
        if denominator > F::epsilon() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }

    /// Get comprehensive memory optimization report
    pub fn get_optimization_report(&self) -> MemoryOptimizationReport {
        let memory_profile = self.allocator.get_memory_profile();
        let cache_stats = self.cache_manager.get_statistics();

        MemoryOptimizationReport {
            config: self.config.clone(),
            memory_profile,
            cache_statistics: cache_stats,
            optimization_recommendations: self.generate_optimization_recommendations(),
        }
    }

    /// Generate optimization recommendations based on usage patterns
    fn generate_optimization_recommendations(&self) -> Vec<MemoryOptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let profile = self.allocator.get_memory_profile();

        // Memory efficiency recommendations
        if profile.efficiency_score < 70.0 {
            recommendations.push(MemoryOptimizationRecommendation {
                priority: 5,
                category: "Memory Efficiency".to_string(),
                description: "Low memory efficiency detected".to_string(),
                suggestion: "Consider using streaming algorithms for large datasets".to_string(),
                expected_impact: "20-50% memory reduction".to_string(),
            });
        }

        // Cache performance recommendations
        if profile.cache_hit_ratio < 0.8 {
            recommendations.push(MemoryOptimizationRecommendation {
                priority: 4,
                category: "Cache Performance".to_string(),
                description: "Low cache hit ratio detected".to_string(),
                suggestion: "Use cache-optimized matrix layouts".to_string(),
                expected_impact: "10-30% performance improvement".to_string(),
            });
        }

        // Fragmentation recommendations
        if profile.fragmentation_ratio > 0.3 {
            recommendations.push(MemoryOptimizationRecommendation {
                priority: 3,
                category: "Memory Fragmentation".to_string(),
                description: "High memory fragmentation detected".to_string(),
                suggestion: "Enable memory pool optimization".to_string(),
                expected_impact: "Reduced allocation overhead".to_string(),
            });
        }

        recommendations
    }
}

impl CacheManager {
    fn new(cachesize: usize) -> Self {
        Self {
            cachesize,
            cache_entries: HashMap::new(),
            access_order: VecDeque::new(),
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn get_statistics(&self) -> CacheStatistics {
        let total_accesses = self.hit_count + self.miss_count;
        let hit_ratio = if total_accesses > 0 {
            self.hit_count as f64 / total_accesses as f64
        } else {
            0.0
        };

        CacheStatistics {
            hit_ratio,
            total_entries: self.cache_entries.len(),
            totalsize: self.cache_entries.values().map(|e| e.size).sum(),
            total_accesses,
        }
    }
}

/// Memory optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryOptimizationReport {
    pub config: MemoryOptimizationConfig,
    pub memory_profile: MemoryProfile,
    pub cache_statistics: CacheStatistics,
    pub optimization_recommendations: Vec<MemoryOptimizationRecommendation>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CacheStatistics {
    pub hit_ratio: f64,
    pub total_entries: usize,
    pub totalsize: usize,
    pub total_accesses: usize,
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryOptimizationRecommendation {
    pub priority: u8,
    pub category: String,
    pub description: String,
    pub suggestion: String,
    pub expected_impact: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_streaming_stats_calculator() {
        let config = MemoryOptimizationConfig::default();
        let mut calculator = StreamingStatsCalculator::new(config);

        let chunk1 = array![1.0, 2.0, 3.0];
        let chunk2 = array![4.0, 5.0, 6.0];

        calculator.process_chunk(chunk1.view()).unwrap();
        calculator.process_chunk(chunk2.view()).unwrap();

        let stats = calculator.get_statistics().unwrap();
        assert_eq!(stats.count, 6);
        assert!((stats.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_cache_optimized_matrix() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = CacheOptimizedMatrix::new(data, MatrixLayout::BlockedRowMajor, 64);

        assert_eq!(matrix.data.nrows(), 2);
        assert_eq!(matrix.data.ncols(), 2);
    }

    #[test]
    fn test_adaptive_allocator() {
        let config = MemoryOptimizationConfig::default();
        let allocator = AdaptiveStatsAllocator::new(config);

        let ptr = allocator
            .allocate_optimized(1024, 8, "test_operation")
            .unwrap();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_memory_optimization_suite() {
        let config = MemoryOptimizationConfig::default();
        let mut suite = MemoryOptimizationSuite::new(config);

        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let correlation = suite.optimized_correlation_matrix(data.view()).unwrap();

        assert_eq!(correlation.nrows(), 3);
        assert_eq!(correlation.ncols(), 3);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((correlation[[i, i]] - 1.0).abs() < 1e-10);
        }
    }
}
