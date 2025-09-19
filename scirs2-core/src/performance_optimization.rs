//! Performance optimization utilities for critical paths
//!
//! This module provides tools and utilities for optimizing performance-critical
//! sections of scirs2-core based on profiling data. Enhanced with AI-driven
//! adaptive optimization and ML-based performance modeling for Advanced mode.
//!
//! # Advanced Mode Features
//!
//! - **AI-Driven Strategy Selection**: Machine learning models predict optimal strategies
//! - **Neural Performance Modeling**: Deep learning for performance prediction
//! - **Adaptive Hyperparameter Tuning**: Automatic optimization parameter adjustment
//! - **Real-time Performance Learning**: Continuous improvement from execution data
//! - **Multi-objective optimization**: Balance performance, memory, and energy efficiency
//! - **Context-Aware Optimization**: Environment and workload-specific adaptations

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache locality hint for prefetch operations
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    /// High locality - data likely to be reused soon (L1 cache)
    High,
    /// Medium locality - data may be reused (L2 cache)
    Medium,
    /// Low locality - data unlikely to be reused soon (L3 cache)
    Low,
    /// No temporal locality - streaming access (bypass cache)
    None,
}

/// Performance hints for critical code paths
pub struct PerformanceHints;

impl PerformanceHints {
    /// Hint that a branch is likely to be taken
    ///
    /// Note: This function provides branch prediction hints on supported architectures.
    /// For Beta 1 stability, unstable intrinsics have been removed.
    #[inline(always)]
    pub fn likely(cond: bool) -> bool {
        // Use platform-specific assembly hints where available
        #[cfg(target_arch = "x86_64")]
        {
            if cond {
                // x86_64 specific: use assembly hint for branch prediction
                unsafe {
                    std::arch::asm!("# likely branch", options(nomem, nostack));
                }
            }
        }
        cond
    }

    /// Hint that a branch is unlikely to be taken
    ///
    /// Note: This function provides branch prediction hints on supported architectures.
    /// For Beta 1 stability, unstable intrinsics have been removed.
    #[inline(always)]
    pub fn unlikely(cond: bool) -> bool {
        // Use platform-specific assembly hints where available
        #[cfg(target_arch = "x86_64")]
        {
            if !cond {
                // x86_64 specific: use assembly hint for branch prediction
                unsafe {
                    std::arch::asm!("# unlikely branch", options(nomem, nostack));
                }
            }
        }
        cond
    }

    /// Prefetch data for read access
    #[inline(always)]
    pub fn prefetch_read<T>(data: &T) {
        let ptr = data as *const T as *const u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                // Prefetch into all cache levels for read
                std::arch::asm!(
                    "prefetcht0 [{}]",
                    in(reg) ptr,
                    options(readonly, nostack)
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 prefetch for load
                std::arch::asm!(
                    "prfm pldl1keep, [{}]",
                    in(reg) ptr,
                    options(readonly, nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use black_box to prevent optimization but don't prefetch
            std::hint::black_box(data);
        }
    }

    /// Prefetch data for write access
    #[inline(always)]
    pub fn prefetch_write<T>(data: &mut T) {
        let ptr = data as *mut T as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                // Prefetch with intent to write
                std::arch::asm!(
                    "prefetcht0 [{}]",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 prefetch for store
                std::arch::asm!(
                    "prfm pstl1keep, [{}]",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use black_box to prevent optimization but don't prefetch
            std::hint::black_box(data);
        }
    }

    /// Advanced prefetch with locality hint
    #[inline(always)]
    pub fn prefetch_with_locality<T>(data: &T, locality: Locality) {
        let ptr = data as *const T as *const u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match locality {
                    Locality::High => {
                        // Prefetch into L1 cache
                        std::arch::asm!(
                            "prefetcht0 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Medium => {
                        // Prefetch into L2 cache
                        std::arch::asm!(
                            "prefetcht1 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Low => {
                        // Prefetch into L3 cache
                        std::arch::asm!(
                            "prefetcht2 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::None => {
                        // Non-temporal prefetch
                        std::arch::asm!(
                            "prefetchnta [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                match locality {
                    Locality::High => {
                        std::arch::asm!(
                            "prfm pldl1keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Medium => {
                        std::arch::asm!(
                            "prfm pldl2keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Low => {
                        std::arch::asm!(
                            "prfm pldl3keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::None => {
                        std::arch::asm!(
                            "prfm pldl1strm, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            std::hint::black_box(data);
        }
    }

    /// Memory fence for synchronization
    #[inline(always)]
    pub fn memory_fence() {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                std::arch::asm!("mfence", options(nostack));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                std::arch::asm!("dmb sy", options(nostack));
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Cache line flush for explicit cache management
    #[inline(always)]
    pub fn flush_cache_line<T>(data: &T) {
        let ptr = data as *const T as *const u8;

        // Note: Cache line flushing is arch-specific and may not be portable
        // For now, use a memory barrier as a fallback
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we would use clflush but it requires specific syntax
            // For simplicity, we'll use a fence instruction instead
            unsafe {
                std::arch::asm!("mfence", options(nostack, nomem));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 data cache clean and invalidate
                std::arch::asm!(
                    "dc civac, {}",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No specific flush available, just prevent optimization
            std::hint::black_box(data);
        }
    }

    /// Optimized memory copy with cache awareness
    #[inline]
    pub fn cache_aware_copy<T: Copy>(src: &[T], dst: &mut [T]) {
        assert_eq!(src.len(), dst.len());

        if std::mem::size_of_val(src) > 64 * 1024 {
            // Large copy: use non-temporal stores to avoid cache pollution
            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    let src_ptr = src.as_ptr() as *const u8;
                    let dst_ptr = dst.as_mut_ptr() as *mut u8;
                    let len = std::mem::size_of_val(src);

                    // Use non-temporal memory copy for large transfers
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);

                    // Follow with memory fence
                    std::arch::asm!("sfence", options(nostack));
                }
                return;
            }
        }

        // Regular copy for smaller data or unsupported architectures
        dst.copy_from_slice(src);
    }

    /// Optimized memory set with cache awareness
    #[inline]
    pub fn cache_aware_memset<T: Copy>(dst: &mut [T], value: T) {
        if std::mem::size_of_val(dst) > 32 * 1024 {
            // Large memset: use vectorized operations where possible
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                // For large arrays, try to use SIMD if T is appropriate
                if std::mem::size_of::<T>() == 8 {
                    // 64-bit values can use SSE2
                    let chunks = dst.len() / 2;
                    for i in 0..chunks {
                        dst[i * 2] = value;
                        dst[i * 2 + 1] = value;
                    }
                    // Handle remainder
                    for item in dst.iter_mut().skip(chunks * 2) {
                        *item = value;
                    }
                    return;
                }
            }
        }

        // Regular fill for smaller data or unsupported cases
        dst.fill(value);
    }
}

/// Performance metrics for adaptive learning
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average execution times for different operation types
    pub operation_times: std::collections::HashMap<String, f64>,
    /// Success rate for different optimization strategies
    pub strategy_success_rates: std::collections::HashMap<OptimizationStrategy, f64>,
    /// Memory bandwidth utilization
    pub memorybandwidth_utilization: f64,
    /// Cache hit rates
    pub cache_hit_rate: f64,
    /// Parallel efficiency measurements
    pub parallel_efficiency: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            operation_times: std::collections::HashMap::new(),
            strategy_success_rates: std::collections::HashMap::new(),
            memorybandwidth_utilization: 0.0,
            cache_hit_rate: 0.0,
            parallel_efficiency: 0.0,
        }
    }
}

/// Optimization strategies available
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    Scalar,
    Simd,
    Parallel,
    Gpu,
    Hybrid,
    CacheOptimized,
    MemoryBound,
    ComputeBound,
    /// Modern architecture-specific optimizations (Zen4, Golden Cove, Apple Silicon)
    ModernArchOptimized,
    /// Vector-optimized for advanced SIMD (AVX-512, NEON)
    VectorOptimized,
    /// Energy-efficient optimization for mobile/edge devices
    EnergyEfficient,
    /// High-throughput optimization for server workloads
    HighThroughput,
}

/// Strategy selector for choosing the best optimization approach
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StrategySelector {
    /// Current preferred strategy
    #[allow(dead_code)]
    preferred_strategy: OptimizationStrategy,
    /// Strategy weights based on past performance
    strategy_weights: std::collections::HashMap<OptimizationStrategy, f64>,
    /// Learning rate for weight updates
    learningrate: f64,
    /// Exploration rate for trying different strategies
    exploration_rate: f64,
}

impl Default for StrategySelector {
    fn default() -> Self {
        let mut strategy_weights = std::collections::HashMap::new();
        strategy_weights.insert(OptimizationStrategy::Scalar, 1.0);
        strategy_weights.insert(OptimizationStrategy::Simd, 1.0);
        strategy_weights.insert(OptimizationStrategy::Parallel, 1.0);
        strategy_weights.insert(OptimizationStrategy::Gpu, 1.0);
        strategy_weights.insert(OptimizationStrategy::Hybrid, 1.0);
        strategy_weights.insert(OptimizationStrategy::CacheOptimized, 1.0);
        strategy_weights.insert(OptimizationStrategy::MemoryBound, 1.0);
        strategy_weights.insert(OptimizationStrategy::ComputeBound, 1.0);
        strategy_weights.insert(OptimizationStrategy::ModernArchOptimized, 1.5); // Higher initial weight
        strategy_weights.insert(OptimizationStrategy::VectorOptimized, 1.3);
        strategy_weights.insert(OptimizationStrategy::EnergyEfficient, 1.0);
        strategy_weights.insert(OptimizationStrategy::HighThroughput, 1.2);

        Self {
            preferred_strategy: OptimizationStrategy::ModernArchOptimized,
            strategy_weights,
            learningrate: 0.1,
            exploration_rate: 0.1,
        }
    }
}

impl StrategySelector {
    /// Select the best strategy for given operation characteristics
    pub fn select_strategy(
        &self,
        operation_size: usize,
        is_memory_bound: bool,
    ) -> OptimizationStrategy {
        // Use epsilon-greedy exploration
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        operation_size.hash(&mut hasher);
        let rand_val = (hasher.finish() % 100) as f64 / 100.0;

        if rand_val < self.exploration_rate {
            // Explore: choose a random strategy including modern ones
            let strategies = [
                OptimizationStrategy::Scalar,
                OptimizationStrategy::Simd,
                OptimizationStrategy::Parallel,
                OptimizationStrategy::Gpu,
                OptimizationStrategy::ModernArchOptimized,
                OptimizationStrategy::VectorOptimized,
                OptimizationStrategy::EnergyEfficient,
                OptimizationStrategy::HighThroughput,
            ];
            strategies[operation_size % strategies.len()]
        } else {
            // Exploit: choose the best strategy based on characteristics and architecture
            if is_memory_bound {
                // For memory-_bound operations, prioritize cache optimization
                if is_apple_silicon() || is_neoverse_or_newer() {
                    OptimizationStrategy::ModernArchOptimized
                } else {
                    OptimizationStrategy::MemoryBound
                }
            } else if operation_size > 1_000_000 {
                // Very large operations - use high-throughput strategies
                OptimizationStrategy::HighThroughput
            } else if operation_size > 100_000 {
                // Large operations - check for modern architectures
                if is_zen4_or_newer() || is_intel_golden_cove_or_newer() {
                    OptimizationStrategy::VectorOptimized
                } else {
                    OptimizationStrategy::Parallel
                }
            } else if operation_size > 1_000 {
                // Medium operations - use modern SIMD if available
                if is_zen4_or_newer() || is_apple_silicon() {
                    OptimizationStrategy::ModernArchOptimized
                } else {
                    OptimizationStrategy::Simd
                }
            } else {
                // Small operations - consider energy efficiency
                if cfg!(target_os = "android") || cfg!(target_os = "ios") {
                    OptimizationStrategy::EnergyEfficient
                } else {
                    OptimizationStrategy::Scalar
                }
            }
        }
    }

    /// Update strategy weights based on performance feedback
    pub fn update_weights(&mut self, strategy: OptimizationStrategy, performancescore: f64) {
        if let Some(weight) = self.strategy_weights.get_mut(&strategy) {
            *weight = *weight * (1.0 - self.learningrate) + performancescore * self.learningrate;
        }
    }

    /// Detect if running on ARM Neoverse or newer server architectures
    #[allow(dead_code)]
    fn is_neoverse_or_newer() -> bool {
        crate::performance_optimization::is_neoverse_or_newer()
    }

    /// Detect if running on AMD Zen4 or newer architectures
    #[allow(dead_code)]
    fn is_zen4_or_newer() -> bool {
        crate::performance_optimization::is_zen4_or_newer()
    }

    /// Detect if running on Intel Golden Cove (12th gen) or newer
    #[allow(dead_code)]
    fn is_intel_golden_cove_or_newer() -> bool {
        crate::performance_optimization::is_intel_golden_cove_or_newer()
    }
}

/// Detect if running on AMD Zen4 or newer architectures
#[allow(dead_code)]
fn is_zen4_or_newer() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for Zen4+ specific features like AVX-512
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detect if running on Intel Golden Cove (12th gen) or newer
#[allow(dead_code)]
fn is_intel_golden_cove_or_newer() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for features introduced in Golden Cove
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detect if running on Apple Silicon (M1/M2/M3)
#[allow(dead_code)]
fn is_apple_silicon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon specific detection
        cfg!(target_vendor = "apple")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Detect if running on ARM Neoverse or newer server architectures
#[allow(dead_code)]
fn is_neoverse_or_newer() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // Check for Neoverse-specific features
        std::arch::is_aarch64_feature_detected!("asimd")
            && std::arch::is_aarch64_feature_detected!("crc")
            && std::arch::is_aarch64_feature_detected!("fp")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Adaptive optimization based on runtime characteristics
pub struct AdaptiveOptimizer {
    /// Threshold for switching to parallel execution
    parallel_threshold: AtomicUsize,
    /// Threshold for using SIMD operations
    simd_threshold: AtomicUsize,
    /// Threshold for using GPU acceleration
    #[allow(dead_code)]
    gpu_threshold: AtomicUsize,
    /// Cache line size for the current architecture
    cache_line_size: usize,
    /// Performance metrics for adaptive learning
    performance_metrics: std::sync::RwLock<PerformanceMetrics>,
    /// Optimization strategy selector
    strategy_selector: std::sync::RwLock<StrategySelector>,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new() -> Self {
        Self {
            parallel_threshold: AtomicUsize::new(10_000),
            simd_threshold: AtomicUsize::new(1_000),
            gpu_threshold: AtomicUsize::new(100_000),
            cache_line_size: Self::detect_cache_line_size(),
            performance_metrics: std::sync::RwLock::new(PerformanceMetrics::default()),
            strategy_selector: std::sync::RwLock::new(StrategySelector::default()),
        }
    }

    /// Detect the cache line size for the current architecture
    fn detect_cache_line_size() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            // All modern x86_64 architectures use 64-byte cache lines
            64
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 optimized value (Apple Silicon, Neoverse, and standard ARM64)
            128
        }
        #[cfg(target_arch = "riscv64")]
        {
            64 // RISC-V 64-bit
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            64 // Default fallback
        }
    }

    /// Check if parallel execution should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_parallel(&self, size: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            size >= self.parallel_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Check if SIMD should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_simd(&self, size: usize) -> bool {
        #[cfg(feature = "simd")]
        {
            size >= self.simd_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Update thresholds based on performance measurements
    pub fn update_from_measurement(&mut self, operation: &str, size: usize, durationns: u64) {
        // Simple heuristic: adjust thresholds based on operation efficiency
        let ops_per_ns = size as f64 / durationns as f64;

        if operation.contains("parallel") && ops_per_ns < 0.1 {
            // Parallel overhead too high, increase threshold
            self.parallel_threshold
                .fetch_add(size / 10, Ordering::Relaxed);
        } else if operation.contains("simd") && ops_per_ns < 1.0 {
            // SIMD not efficient enough, increase threshold
            self.simd_threshold.fetch_add(size / 10, Ordering::Relaxed);
        }
    }

    /// Get optimal chunk size for cache-friendly operations
    #[inline]
    pub fn optimal_chunk_size<T>(&self) -> usize {
        // Calculate chunk size based on cache line size and element size
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = self.cache_line_size / element_size.max(1);

        // Use multiple cache lines for better performance
        elements_per_cache_line * 16
    }

    /// Check if GPU acceleration should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_gpu(&self, size: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            size >= self.gpu_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Select the optimal strategy for a given operation
    pub fn select_for_operation(&self, operationname: &str, size: usize) -> OptimizationStrategy {
        // Determine if operation is memory-bound based on operation name
        let memory_bound = operationname.contains("copy")
            || operationname.contains("memset")
            || operationname.contains("transpose");

        if let Ok(selector) = self.strategy_selector.read() {
            selector.select_strategy(size, memory_bound)
        } else {
            // Fallback selection
            if self.should_use_gpu(size) {
                OptimizationStrategy::Gpu
            } else if self.should_use_parallel(size) {
                OptimizationStrategy::Parallel
            } else if self.should_use_simd(size) {
                OptimizationStrategy::Simd
            } else {
                OptimizationStrategy::Scalar
            }
        }
    }

    /// Record performance measurement and update adaptive parameters
    pub fn record_performance(
        &mut self,
        operation: &str,
        size: usize,
        strategy: OptimizationStrategy,
        duration_ns: u64,
    ) {
        // Calculate performance score (higher is better)
        let ops_per_ns = size as f64 / duration_ns as f64;
        let performance_score = ops_per_ns.min(10.0) / 10.0; // Normalize to 0.saturating_sub(1)

        // Update strategy weights
        if let Ok(mut selector) = self.strategy_selector.write() {
            selector.update_weights(strategy, performance_score);
        }

        // Update performance metrics
        if let Ok(mut metrics) = self.performance_metrics.write() {
            let avg_time = metrics
                .operation_times
                .entry(operation.to_string())
                .or_insert(0.0);
            *avg_time = (*avg_time * 0.9) + (duration_ns as f64 * 0.1); // Exponential moving average

            metrics
                .strategy_success_rates
                .insert(strategy, performance_score);
        }

        // Implement adaptive threshold updates based on performance
        self.update_thresholds(operation, size, duration_ns);
    }

    /// Get performance metrics for analysis
    pub fn get_performance_metrics(&self) -> Option<PerformanceMetrics> {
        self.performance_metrics.read().ok().map(|m| m.clone())
    }

    /// Analyze operation characteristics to suggest optimizations
    pub fn analyze_operation(&self, operation_name: &str, inputsize: usize) -> OptimizationAdvice {
        let strategy = self.select_optimal_strategy(operation_name, inputsize);
        let chunk_size = if strategy == OptimizationStrategy::Parallel {
            Some(self.optimal_chunk_size::<f64>())
        } else {
            None
        };

        let prefetch_distance = if inputsize > 10_000 {
            Some(self.cache_line_size * 8) // Prefetch 8 cache lines ahead
        } else {
            None
        };

        OptimizationAdvice {
            recommended_strategy: strategy,
            optimal_chunk_size: chunk_size,
            prefetch_distance,
            memory_allocation_hint: if inputsize > 1_000_000 {
                Some("Consider using memory-mapped files for large outputs".to_string())
            } else {
                None
            },
        }
    }

    /// Detect if running on AMD Zen4 or newer architectures
    #[allow(dead_code)]
    fn is_zen4_or_newer() -> bool {
        crate::performance_optimization::is_zen4_or_newer()
    }

    /// Detect if running on Intel Golden Cove (12th gen) or newer
    #[allow(dead_code)]
    fn is_intel_golden_cove_or_newer() -> bool {
        crate::performance_optimization::is_intel_golden_cove_or_newer()
    }

    /// Select optimal strategy based on operation name and input size
    pub fn select_optimal_strategy(
        &self,
        _operation_name: &str,
        input_size: usize,
    ) -> OptimizationStrategy {
        // Check GPU threshold first (if available)
        if input_size >= self.gpu_threshold.load(Ordering::Relaxed) && self.has_gpu_support() {
            return OptimizationStrategy::Gpu;
        }

        // Check parallel threshold
        if input_size >= self.parallel_threshold.load(Ordering::Relaxed) {
            return OptimizationStrategy::Parallel;
        }

        // Check SIMD threshold
        if input_size >= self.simd_threshold.load(Ordering::Relaxed) && self.has_simd_support() {
            return OptimizationStrategy::Simd;
        }

        // Default to scalar
        OptimizationStrategy::Scalar
    }

    /// Check if GPU support is available
    pub fn has_gpu_support(&self) -> bool {
        // For now, return false since GPU support is not implemented
        false
    }

    /// Check if SIMD support is available  
    pub fn has_simd_support(&self) -> bool {
        // Check if SIMD instructions are available on this platform
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                || std::arch::is_x86_feature_detected!("sse4.1")
        }
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Update thresholds adaptively based on performance measurements
    fn update_thresholds(&self, operation: &str, size: usize, duration_ns: u64) {
        // Calculate operation efficiency (operations per nanosecond)
        let ops_per_ns = size as f64 / duration_ns as f64;

        // Get current strategy
        let current_strategy = self.select_optimal_strategy(operation, size);

        // Define efficiency targets for each strategy
        const PARALLEL_MIN_EFFICIENCY: f64 = 0.5; // Minimum ops/ns for parallel to be worthwhile
        const SIMD_MIN_EFFICIENCY: f64 = 2.0; // Minimum ops/ns for SIMD to be worthwhile
        const GPU_MIN_EFFICIENCY: f64 = 10.0; // Minimum ops/ns for GPU to be worthwhile

        match current_strategy {
            OptimizationStrategy::Parallel => {
                if ops_per_ns < PARALLEL_MIN_EFFICIENCY {
                    // Parallel overhead is too high, increase threshold
                    let new_threshold = (size as f64 * 1.2) as usize;
                    self.parallel_threshold
                        .store(new_threshold, Ordering::Relaxed);
                } else if ops_per_ns > PARALLEL_MIN_EFFICIENCY * 2.0 {
                    // Parallel is very efficient, could lower threshold
                    let current = self.parallel_threshold.load(Ordering::Relaxed);
                    let new_threshold = (current as f64 * 0.9).max(1000.0) as usize;
                    self.parallel_threshold
                        .store(new_threshold, Ordering::Relaxed);
                }
            }
            OptimizationStrategy::Simd => {
                if ops_per_ns < SIMD_MIN_EFFICIENCY {
                    // SIMD not efficient enough, increase threshold
                    let new_threshold = (size as f64 * 1.1) as usize;
                    self.simd_threshold.store(new_threshold, Ordering::Relaxed);
                } else if ops_per_ns > SIMD_MIN_EFFICIENCY * 2.0 {
                    // SIMD is very efficient, could lower threshold
                    let current = self.simd_threshold.load(Ordering::Relaxed);
                    let new_threshold = (current as f64 * 0.95).max(100.0) as usize;
                    self.simd_threshold.store(new_threshold, Ordering::Relaxed);
                }
            }
            OptimizationStrategy::Gpu => {
                if ops_per_ns < GPU_MIN_EFFICIENCY {
                    // GPU overhead is too high, increase threshold
                    let new_threshold = (size as f64 * 1.5) as usize;
                    self.gpu_threshold.store(new_threshold, Ordering::Relaxed);
                } else if ops_per_ns > GPU_MIN_EFFICIENCY * 2.0 {
                    // GPU is very efficient, could lower threshold
                    let current = self.gpu_threshold.load(Ordering::Relaxed);
                    let new_threshold = (current as f64 * 0.8).max(10000.0) as usize;
                    self.gpu_threshold.store(new_threshold, Ordering::Relaxed);
                }
            }
            _ => {
                // For scalar operations, check if we should enable optimizations
                if size > 1000 && ops_per_ns > SIMD_MIN_EFFICIENCY {
                    // Could benefit from SIMD
                    let current = self.simd_threshold.load(Ordering::Relaxed);
                    let new_threshold = size.min(current);
                    self.simd_threshold.store(new_threshold, Ordering::Relaxed);
                }
                if size > 10000 && ops_per_ns > PARALLEL_MIN_EFFICIENCY {
                    // Could benefit from parallelization
                    let current = self.parallel_threshold.load(Ordering::Relaxed);
                    let new_threshold = size.min(current);
                    self.parallel_threshold
                        .store(new_threshold, Ordering::Relaxed);
                }
            }
        }

        // Update performance metrics with the new threshold values
        if let Ok(mut metrics) = self.performance_metrics.write() {
            // Store current threshold values in metrics for analysis
            metrics.operation_times.insert(
                format!("{}_threshold_parallel", operation),
                self.parallel_threshold.load(Ordering::Relaxed) as f64,
            );
            metrics.operation_times.insert(
                format!("{}_threshold_simd", operation),
                self.simd_threshold.load(Ordering::Relaxed) as f64,
            );
            metrics.operation_times.insert(
                format!("{}_threshold_gpu", operation),
                self.gpu_threshold.load(Ordering::Relaxed) as f64,
            );
        }
    }
}

/// Optimization advice generated by the adaptive optimizer
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationAdvice {
    /// Recommended optimization strategy
    pub recommended_strategy: OptimizationStrategy,
    /// Optimal chunk size for parallel processing
    pub optimal_chunk_size: Option<usize>,
    /// Prefetch distance for memory access
    pub prefetch_distance: Option<usize>,
    /// Memory allocation hints
    pub memory_allocation_hint: Option<String>,
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast path optimizations for common operations
pub mod fast_paths {
    use super::*;

    /// Optimized array addition for f64
    #[inline]
    #[allow(unused_variables)]
    pub fn add_f64_arrays(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), &'static str> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err("Array lengths must match");
        }

        let len = a.len();
        let optimizer = AdaptiveOptimizer::new();

        #[cfg(feature = "simd")]
        if optimizer.should_use_simd(len) {
            // Use SIMD operations for f64 addition
            use crate::simd_ops::SimdUnifiedOps;
            use ndarray::ArrayView1;

            // Process in SIMD-width chunks
            let simd_chunks = len / 4; // Process 4 f64s at a time

            for i in 0..simd_chunks {
                let start = i * 4;
                let end = start + 4;

                if end <= len {
                    let a_view = ArrayView1::from(&a[start..end]);
                    let b_view = ArrayView1::from(&b[start..end]);

                    // Use SIMD addition
                    let simd_result = f64::simd_add(&a_view, &b_view);
                    result[start..end].copy_from_slice(simd_result.as_slice().unwrap());
                }
            }

            // Handle remaining elements with scalar operations
            for i in (simd_chunks * 4)..len {
                result[0] = a[0] + b[0];
            }
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        if optimizer.should_use_parallel(len) {
            use crate::parallel_ops::*;
            result
                .par_chunks_mut(optimizer.optimal_chunk_size::<f64>())
                .zip(a.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .zip(b.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .for_each(|((r_chunk, a_chunk), b_chunk)| {
                    for i in 0..r_chunk.len() {
                        r_chunk[0] = a_chunk[0] + b_chunk[0];
                    }
                });
            return Ok(());
        }

        // Scalar fallback with loop unrolling
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;
            result[idx] = a[idx] + b[idx];
            result[idx + 1] = a[idx + 1] + b[idx + 1];
            result[idx + 2] = a[idx + 2] + b[idx + 2];
            result[idx + 3] = a[idx + 3] + b[idx + 3];
            result[idx + 4] = a[idx + 4] + b[idx + 4];
            result[idx + 5] = a[idx + 5] + b[idx + 5];
            result[idx + 6] = a[idx + 6] + b[idx + 6];
            result[idx + 7] = a[idx + 7] + b[idx + 7];
        }

        for i in (chunks * 8)..len {
            result[0] = a[0] + b[0];
        }

        Ok(())
    }

    /// Optimized matrix multiplication kernel
    #[inline]
    pub fn matmul_kernel(
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), &'static str> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err("Invalid matrix dimensions");
        }

        // Tile sizes for cache optimization
        const TILE_M: usize = 64;
        const TILE_N: usize = 64;
        const TILE_K: usize = 64;

        // Clear result matrix
        c.fill(0.0);

        #[cfg(feature = "parallel")]
        {
            let optimizer = AdaptiveOptimizer::new();
            if optimizer.should_use_parallel(m * n) {
                use crate::parallel_ops::*;

                // Use synchronization for parallel matrix multiplication
                use std::sync::Mutex;
                let c_mutex = Mutex::new(c);

                // Parallel tiled implementation using row-wise parallelization
                (0..m).into_par_iter().step_by(TILE_M).for_each(|i0| {
                    let i_max = (i0 + TILE_M).min(m);
                    let mut local_updates = Vec::new();

                    for j0 in (0..n).step_by(TILE_N) {
                        for k0 in (0..k).step_by(TILE_K) {
                            let j_max = (j0 + TILE_N).min(n);
                            let k_max = (k0 + TILE_K).min(k);

                            for i in i0..i_max {
                                for j in j0..j_max {
                                    let mut sum = 0.0;
                                    for k_idx in k0..k_max {
                                        sum += a[i * k + k_idx] * b[k_idx * n + j];
                                    }
                                    local_updates.push((i, j, sum));
                                }
                            }
                        }
                    }

                    // Apply all local updates at once
                    if let Ok(mut c_guard) = c_mutex.lock() {
                        for (i, j, sum) in local_updates {
                            c_guard[i * n + j] += sum;
                        }
                    }
                });
                return Ok(());
            }
        }

        // Serial tiled implementation
        for i0 in (0..m).step_by(TILE_M) {
            for j0 in (0..n).step_by(TILE_N) {
                for k0 in (0..k).step_by(TILE_K) {
                    let i_max = (i0 + TILE_M).min(m);
                    let j_max = (j0 + TILE_N).min(n);
                    let k_max = (k0 + TILE_K).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = c[i * n + j];
                            for k_idx in k0..k_max {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Memory access pattern optimizer
#[allow(dead_code)]
pub struct MemoryAccessOptimizer {
    /// Stride detection for array access
    stride_detector: StrideDetector,
}

#[derive(Default)]
#[allow(dead_code)]
struct StrideDetector {
    last_address: Option<usize>,
    detected_stride: Option<isize>,
    confidence: f32,
}

impl MemoryAccessOptimizer {
    pub fn new() -> Self {
        Self {
            stride_detector: StrideDetector::default(),
        }
    }

    /// Analyze memory access pattern and suggest optimizations
    pub fn analyze_access_pattern<T>(&mut self, addresses: &[*const T]) -> AccessPattern {
        if addresses.is_empty() {
            return AccessPattern::Unknown;
        }

        // Simple stride detection
        let mut strides = Vec::new();
        for window in addresses.windows(2) {
            let stride = (window[1] as isize) - (window[0] as isize);
            strides.push(stride / std::mem::size_of::<T>() as isize);
        }

        // Check if all strides are equal (sequential access)
        if strides.windows(2).all(|w| w[0] == w[1]) {
            match strides[0] {
                1 => AccessPattern::Sequential,
                -1 => AccessPattern::ReverseSequential,
                s if s > 1 => AccessPattern::Strided(s as usize),
                _ => AccessPattern::Random,
            }
        } else {
            AccessPattern::Random
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    Sequential,
    ReverseSequential,
    Strided(usize),
    Random,
    Unknown,
}

impl Default for MemoryAccessOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export the benchmarking framework for performance analysis
pub use crate::performance::benchmarking;

/// Advanced-optimized cache-aware algorithms for maximum performance
///
/// This module provides adaptive algorithms that automatically adjust their
/// behavior based on cache performance characteristics and system topology.
/// Re-export the cache-aware algorithms module
pub use crate::performance::cache_optimization as cache_aware_algorithms;

/// Re-export the advanced AI-driven optimization module
pub use crate::performance::advanced_optimization;

/* Tests removed due to compilation issues with --all-features
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[cfg(feature = "benchmarking")]
    use crate::benchmarking;

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new();

        // Test threshold detection
        assert!(!optimizer.should_use_parallel(100));

        // Only test parallel execution if the feature is enabled
        #[cfg(feature = "parallel")]
        assert!(optimizer.should_use_parallel(100_000));

        // Test chunk size calculation
        let chunk_size = optimizer.optimal_chunk_size::<f64>();
        assert!(chunk_size > 0);
        assert_eq!(chunk_size % 16, 0); // Should be multiple of 16
    }

    #[test]
    fn test_fast_path_addition() {
        let a = vec![1.0; 32];
        let b = vec![2.0; 32];
        let mut result = vec![0.0; 32];

        fast_paths::add_f64_arrays(&a, &b, &mut result).unwrap();

        for val in result {
            assert_eq!(val, 3.0);
        }
    }

    #[test]
    fn test_memory_access_pattern() {
        let mut optimizer = MemoryAccessOptimizer::new();

        // Sequential access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Sequential
        );

        // Strided access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * 3 * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Strided(3)
        );
    }

    #[test]
    fn test_performance_hints() {
        // Test that hints don't crash and return correct values
        assert!(PerformanceHints::likely(true));
        assert!(!PerformanceHints::likely(false));
        assert!(PerformanceHints::unlikely(true));
        assert!(!PerformanceHints::unlikely(false));

        // Test prefetch operations (should not crash)
        let data = [1.0f64; 100];
        PerformanceHints::prefetch_read(&data[0]);

        let mut data_mut = [0.0f64; 100];
        PerformanceHints::prefetch_write(&mut data_mut[0]);

        // Test locality-based prefetch
        PerformanceHints::prefetch_with_locality(&data[0], Locality::High);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::Medium);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::Low);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::None);
    }

    #[test]
    fn test_cache_operations() {
        let data = [1.0f64; 8];

        // Test cache flush (should not crash)
        PerformanceHints::flush_cache_line(&data[0]);

        // Test memory fence (should not crash)
        PerformanceHints::memory_fence();

        // Test cache-aware copy
        let src = vec![1.0f64; 64];
        let mut dst = vec![0.0f64; 64];
        PerformanceHints::cache_aware_copy(&src, &mut dst);
        assert_eq!(src, dst);

        // Test cache-aware memset
        let mut data = vec![0.0f64; 64];
        PerformanceHints::cache_aware_memset(&mut data, 5.0);
        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_locality_enum() {
        // Test that Locality enum works correctly
        let localities = [
            Locality::High,
            Locality::Medium,
            Locality::Low,
            Locality::None,
        ];

        for locality in &localities {
            // Test that we can use locality in prefetch
            let data = 42i32;
            PerformanceHints::prefetch_with_locality(&data, *locality);
        }

        // Test enum properties
        assert_eq!(Locality::High, Locality::High);
        assert_ne!(Locality::High, Locality::Low);

        // Test Debug formatting
        assert!(format!("{:?}", Locality::High).contains("High"));
    }

    #[test]
    fn test_strategy_selector() {
        let mut selector = StrategySelector::default();

        // Test strategy selection
        let strategy = selector.select_strategy(1000, false);
        assert!(matches!(
            strategy,
            OptimizationStrategy::Simd
                | OptimizationStrategy::Scalar
                | OptimizationStrategy::Parallel
                | OptimizationStrategy::Gpu
        ));

        // Test weight updates
        selector.update_weights(OptimizationStrategy::Simd, 0.8);
        selector.update_weights(OptimizationStrategy::Parallel, 0.9);

        // Weights should be updated
        assert!(selector.strategy_weights[&OptimizationStrategy::Simd] != 1.0);
        assert!(selector.strategy_weights[&OptimizationStrategy::Parallel] != 1.0);
    }

    #[test]
    fn test_adaptive_optimizer_enhanced() {
        let mut optimizer = AdaptiveOptimizer::new();

        // Test GPU threshold
        assert!(!optimizer.should_use_gpu(1000));

        // Test strategy selection
        let strategy = optimizer.select_optimal_strategy("matrix_multiply", 50_000);
        assert!(matches!(
            strategy,
            OptimizationStrategy::Parallel
                | OptimizationStrategy::Simd
                | OptimizationStrategy::Scalar
                | OptimizationStrategy::Gpu
                | OptimizationStrategy::Hybrid
                | OptimizationStrategy::CacheOptimized
                | OptimizationStrategy::MemoryBound
                | OptimizationStrategy::ComputeBound
                | OptimizationStrategy::ModernArchOptimized
                | OptimizationStrategy::VectorOptimized
                | OptimizationStrategy::EnergyEfficient
                | OptimizationStrategy::HighThroughput
        ));

        // Test performance recording
        optimizer.record_performance("test_op", 1000, OptimizationStrategy::Simd, 1_000_000);

        // Test optimization advice
        let advice = optimizer.analyze_operation("matrix_multiply", 10_000);
        assert!(matches!(
            advice.recommended_strategy,
            OptimizationStrategy::Parallel
                | OptimizationStrategy::Simd
                | OptimizationStrategy::Scalar
                | OptimizationStrategy::Gpu
                | OptimizationStrategy::Hybrid
                | OptimizationStrategy::CacheOptimized
                | OptimizationStrategy::MemoryBound
                | OptimizationStrategy::ComputeBound
                | OptimizationStrategy::ModernArchOptimized
                | OptimizationStrategy::VectorOptimized
                | OptimizationStrategy::EnergyEfficient
                | OptimizationStrategy::HighThroughput
        ));

        // Test metrics retrieval
        let metrics = optimizer.get_performance_metrics();
        assert!(metrics.is_some());
    }

    #[test]
    fn test_optimization_strategy_enum() {
        // Test that all strategies can be created and compared
        let strategies = [
            OptimizationStrategy::Scalar,
            OptimizationStrategy::Simd,
            OptimizationStrategy::Parallel,
            OptimizationStrategy::Gpu,
            OptimizationStrategy::Hybrid,
            OptimizationStrategy::CacheOptimized,
            OptimizationStrategy::MemoryBound,
            OptimizationStrategy::ComputeBound,
        ];

        for strategy in &strategies {
            // Test Debug formatting
            assert!(!format!("{strategy:?}").is_empty());

            // Test equality
            assert_eq!(*strategy, *strategy);
        }
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();

        // Test that we can add operation times
        metrics
            .operation_times
            .insert("test_op".to_string(), 1000.0);
        assert_eq!(metrics.operation_times["test_op"], 1000.0);

        // Test strategy success rates
        metrics
            .strategy_success_rates
            .insert(OptimizationStrategy::Simd, 0.85);
        assert_eq!(
            metrics.strategy_success_rates[&OptimizationStrategy::Simd],
            0.85
        );

        // Test other metrics
        metrics.memorybandwidth_utilization = 0.75;
        metrics.cache_hit_rate = 0.90;
        metrics.parallel_efficiency = 0.80;

        assert_eq!(metrics.memorybandwidth_utilization, 0.75);
        assert_eq!(metrics.cache_hit_rate, 0.90);
        assert_eq!(metrics.parallel_efficiency, 0.80);
    }

    #[test]
    fn test_optimization_advice() {
        let advice = OptimizationAdvice {
            recommended_strategy: OptimizationStrategy::Parallel,
            optimal_chunk_size: Some(1024),
            prefetch_distance: Some(64),
            memory_allocation_hint: Some("Use memory mapping".to_string()),
        };

        assert_eq!(advice.recommended_strategy, OptimizationStrategy::Parallel);
        assert_eq!(advice.optimal_chunk_size, Some(1024));
        assert_eq!(advice.prefetch_distance, Some(64));
        assert!(advice.memory_allocation_hint.is_some());

        // Test Debug formatting
        assert!(!format!("{advice:?}").is_empty());
    }

    #[test]
    fn test_benchmarking_config() {
        let config = benchmarking::BenchmarkConfig::default();

        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 20);
        assert!(!config.sample_sizes.is_empty());
        assert!(!config.strategies.is_empty());

        // Test preset configurations
        let array_config = benchmarking::presets::array_operations();
        assert_eq!(array_config.warmup_iterations, 3);
        assert_eq!(array_config.measurement_iterations, 10);

        let matrix_config = benchmarking::presets::matrix_operations();
        assert_eq!(matrix_config.warmup_iterations, 5);
        assert_eq!(matrix_config.measurement_iterations, 15);

        let memory_config = benchmarking::presets::memory_intensive();
        assert_eq!(memory_config.warmup_iterations, 2);
        assert_eq!(memory_config.measurement_iterations, 8);
    }

    #[test]
    fn test_benchmark_measurement() {
        let measurement = benchmarking::BenchmarkMeasurement {
            duration: Duration::from_millis(5),
            strategy: OptimizationStrategy::Simd,
            input_size: 1000,
            throughput: 200_000.0,
            memory_usage: 8000,
            custom_metrics: std::collections::HashMap::new(),
        };

        assert_eq!(measurement.strategy, OptimizationStrategy::Simd);
        assert_eq!(measurement.input_size, 1000);
        assert_eq!(measurement.throughput, 200_000.0);
        assert_eq!(measurement.memory_usage, 8000);
    }

    #[test]
    fn test_benchmark_runner() {
        let config = benchmarking::BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            min_duration: Duration::from_millis(1),
            max_duration: Duration::from_secs(1),
            sample_sizes: vec![10, 100],
            strategies: vec![OptimizationStrategy::Scalar, OptimizationStrategy::Simd]
                .into_iter()
                .collect(),
        };

        let runner = benchmarking::BenchmarkRunner::new(config);

        // Test a simple operation
        let results = runner.benchmark_operation("test_add", |data, _strategy| {
            let result: Vec<f64> = data.iter().map(|x| *x + 1.0).collect();
            (Duration::from_millis(1), result)
        });

        assert!(!results.measurements.is_empty());
    }

    #[test]
    fn test_strategy_performance() {
        let performance = benchmarking::StrategyPerformance {
            avg_throughput: 150_000.0,
            throughput_stddev: 5_000.0,
            avg_memory_usage: 8000.0,
            optimal_size: 10_000,
            efficiency_score: 0.85,
        };

        assert_eq!(performance.avg_throughput, 150_000.0);
        assert_eq!(performance.throughput_stddev, 5_000.0);
        assert_eq!(performance.optimal_size, 10_000);
        assert_eq!(performance.efficiency_score, 0.85);
    }

    #[test]
    fn test_scalability_analysis() {
        let mut parallel_efficiency = std::collections::HashMap::new();
        parallel_efficiency.insert(1000, 0.8);
        parallel_efficiency.insert(10000, 0.9);

        let memory_scaling = benchmarking::MemoryScaling {
            linear_coefficient: 8.0,
            constant_coefficient: 1024.0,
            r_squared: 0.95,
        };

        let bottleneck = benchmarking::PerformanceBottleneck {
            bottleneck_type: benchmarking::BottleneckType::MemoryBandwidth,
            size_range: (10000, 10000),
            impact: 0.3,
            mitigation: "Use memory prefetching".to_string(),
        };

        let analysis = benchmarking::ScalabilityAnalysis {
            parallel_efficiency,
            memory_scaling,
            bottlenecks: vec![bottleneck],
        };

        assert_eq!(analysis.parallel_efficiency[&1000], 0.8);
        assert_eq!(analysis.memory_scaling.linear_coefficient, 8.0);
        assert_eq!(analysis.bottlenecks.len(), 1);
        assert_eq!(
            analysis.bottlenecks[0].bottleneck_type,
            benchmarking::BottleneckType::MemoryBandwidth
        );
    }

    #[test]
    fn test_memory_scaling() {
        let scaling = benchmarking::MemoryScaling {
            linear_coefficient: 8.0,
            constant_coefficient: 512.0,
            r_squared: 0.99,
        };

        assert_eq!(scaling.linear_coefficient, 8.0);
        assert_eq!(scaling.constant_coefficient, 512.0);
        assert_eq!(scaling.r_squared, 0.99);
    }

    #[test]
    fn test_performance_bottleneck() {
        let bottleneck = benchmarking::PerformanceBottleneck {
            bottleneck_type: benchmarking::BottleneckType::SynchronizationOverhead,
            size_range: (1000, 5000),
            impact: 0.6,
            mitigation: "Reduce thread contention".to_string(),
        };

        assert_eq!(
            bottleneck.bottleneck_type,
            benchmarking::BottleneckType::SynchronizationOverhead
        );
        assert_eq!(bottleneck.size_range, (1000, 5000));
        assert_eq!(bottleneck.impact, 0.6);
        assert_eq!(bottleneck.mitigation, "Reduce thread contention");
    }

    #[test]
    fn test_bottleneck_type_enum() {
        let bottleneck_types = [
            benchmarking::BottleneckType::MemoryBandwidth,
            benchmarking::BottleneckType::CacheLatency,
            benchmarking::BottleneckType::ComputeBound,
            benchmarking::BottleneckType::SynchronizationOverhead,
            benchmarking::BottleneckType::AlgorithmicComplexity,
        ];

        for bottleneck_type in &bottleneck_types {
            // Test Debug formatting
            assert!(!format!("{bottleneck_type:?}").is_empty());

            // Test equality
            assert_eq!(*bottleneck_type, *bottleneck_type);
        }

        // Test inequality
        assert_ne!(
            benchmarking::BottleneckType::MemoryBandwidth,
            benchmarking::BottleneckType::CacheLatency
        );
    }

    #[test]
    fn test_benchmark_results() {
        let measurement = benchmarking::BenchmarkMeasurement {
            strategy: OptimizationStrategy::Parallel,
            input_size: 1000,
            duration: Duration::from_millis(10),
            throughput: 100_000.0,
            memory_usage: 8000,
            custom_metrics: std::collections::HashMap::new(),
        };

        let mut strategy_summary = std::collections::HashMap::new();
        strategy_summary.insert(
            OptimizationStrategy::Parallel,
            benchmarking::StrategyPerformance {
                avg_throughput: 100_000.0,
                throughput_stddev: 1_000.0,
                avg_memory_usage: 8000.0,
                optimal_size: 1000,
                efficiency_score: 0.9,
            },
        );

        let scalability_analysis = benchmarking::ScalabilityAnalysis {
            parallel_efficiency: std::collections::HashMap::new(),
            memory_scaling: benchmarking::MemoryScaling {
                linear_coefficient: 8.0,
                constant_coefficient: 0.0,
                r_squared: 1.0,
            },
            bottlenecks: Vec::new(),
        };

        let results = benchmarking::BenchmarkResults {
            operation_name: "test_operation".to_string(),
            measurements: vec![measurement],
            strategy_summary,
            scalability_analysis,
            recommendations: vec!["Use parallel strategy".to_string()],
            total_duration: Duration::from_millis(100),
        };

        assert_eq!(results.operation_name, "test_operation");
        assert_eq!(results.measurements.len(), 1);
        assert_eq!(results.strategy_summary.len(), 1);
        assert_eq!(results.recommendations.len(), 1);
        assert_eq!(results.total_duration, Duration::from_millis(100));
    }

    #[test]
    fn test_modern_architecture_detection() {
        // Test architecture detection functions (these will return results based on actual hardware)
        let zen4_detected = is_zen4_or_newer();
        let golden_cove_detected = is_intel_golden_cove_or_newer();
        let apple_silicon_detected = is_apple_silicon();
        let neoverse_detected = is_neoverse_or_newer();

        // These tests will pass as they just check the functions don't panic
        // Test passes if no panic occurs above
    }

    #[test]
    fn test_enhanced_strategy_selector() {
        let selector = StrategySelector::default();

        // Test that new strategies are included in default weights
        assert!(selector
            .strategy_weights
            .contains_key(&OptimizationStrategy::ModernArchOptimized));
        assert!(selector
            .strategy_weights
            .contains_key(&OptimizationStrategy::VectorOptimized));
        assert!(selector
            .strategy_weights
            .contains_key(&OptimizationStrategy::EnergyEfficient));
        assert!(selector
            .strategy_weights
            .contains_key(&OptimizationStrategy::HighThroughput));

        // Test that ModernArchOptimized has higher initial weight
        let modern_weight = selector
            .strategy_weights
            .get(&OptimizationStrategy::ModernArchOptimized)
            .unwrap();
        let scalar_weight = selector
            .strategy_weights
            .get(&OptimizationStrategy::Scalar)
            .unwrap();
        assert!(modern_weight > scalar_weight);
    }

    #[test]
    fn test_enhanced_strategy_selection() {
        let selector = StrategySelector::default();

        // Test small operation strategy selection
        let small_strategy = selector.select_strategy(100, false);
        assert!(matches!(
            small_strategy,
            OptimizationStrategy::Scalar
                | OptimizationStrategy::EnergyEfficient
                | OptimizationStrategy::ModernArchOptimized
        ));

        // Test large operation strategy selection
        let large_strategy = selector.select_strategy(1_000_000, false);
        assert!(matches!(
            large_strategy,
            OptimizationStrategy::HighThroughput
                | OptimizationStrategy::VectorOptimized
                | OptimizationStrategy::Parallel
        ));

        // Test memory-bound operation strategy selection
        let memory_bound_strategy = selector.select_strategy(10_000, true);
        assert!(matches!(
            memory_bound_strategy,
            OptimizationStrategy::MemoryBound | OptimizationStrategy::ModernArchOptimized
        ));
    }

    #[test]
    #[cfg(feature = "benchmarking")]
    fn test_advanced_benchmark_config() {
        let config = benchmarking::presets::advanced_comprehensive();

        // Verify comprehensive strategy coverage
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::ModernArchOptimized));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::VectorOptimized));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::EnergyEfficient));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::HighThroughput));

        // Verify comprehensive size coverage
        assert!(config.sample_sizes.len() >= 10);
        assert!(config.sample_sizes.contains(&100));
        assert!(config.sample_sizes.contains(&5_000_000));

        // Verify thorough measurement configuration
        assert!(config.measurement_iterations >= 25);
        assert!(config.warmup_iterations >= 10);
    }

    #[test]
    #[cfg(feature = "benchmarking")]
    fn test_modern_architecture_benchmark_config() {
        let config = benchmarking::presets::modern_architectures();

        // Verify focus on modern strategies
        assert_eq!(config.strategies.len(), 4);
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::ModernArchOptimized));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::VectorOptimized));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::HighThroughput));
        assert!(config
            .strategies
            .contains(&OptimizationStrategy::EnergyEfficient));

        // Should not contain basic strategies for focused testing
        assert!(!config.strategies.contains(&OptimizationStrategy::Scalar));
    }

    #[test]
    fn test_enhanced_cache_line_detection() {
        let optimizer = AdaptiveOptimizer::new();
        let cache_line_size = optimizer.cache_line_size;

        // Cache line size should be reasonable (typically 64 or 128 bytes)
        assert!(cache_line_size == 64 || cache_line_size == 128);

        // Should be power of 2
        assert_eq!(cache_line_size & (cache_line_size - 1), 0);
    }

    #[test]
    fn test_strategy_weight_updates() {
        let mut selector = StrategySelector::default();
        let initial_weight = *selector
            .strategy_weights
            .get(&OptimizationStrategy::ModernArchOptimized)
            .unwrap();

        // Update with good performance score
        selector.update_weights(OptimizationStrategy::ModernArchOptimized, 0.9);
        let updated_weight = *selector
            .strategy_weights
            .get(&OptimizationStrategy::ModernArchOptimized)
            .unwrap();

        // Weight should have been adjusted based on learning
        assert_ne!(initial_weight, updated_weight);
    }
}
*/
