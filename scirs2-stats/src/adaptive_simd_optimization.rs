//! Adaptive SIMD optimization framework for scirs2-stats v1.0.0
//!
//! This module provides an intelligent SIMD optimization system that automatically
//! selects the best SIMD strategy based on data characteristics, hardware capabilities,
//! and performance requirements. It builds on the existing SIMD infrastructure
//! to provide optimal performance across different scenarios.

use crate::error::StatsResult;
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for adaptive SIMD optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSimdConfig {
    /// Enable automatic hardware detection
    pub auto_detect_hardware: bool,
    /// Enable performance profiling for optimization selection
    pub enable_profiling: bool,
    /// Minimum data size for SIMD optimization
    pub min_simdsize: usize,
    /// Performance cache size
    pub cachesize: usize,
    /// Benchmarking sample size for algorithm selection
    pub benchmark_samples: usize,
    /// Enable hybrid CPU-GPU processing
    pub enable_hybrid_processing: bool,
    /// SIMD alignment requirements
    pub alignment_requirements: SimdAlignment,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable adaptive vectorization
    pub adaptive_vectorization: bool,
    /// Memory bandwidth optimization
    pub memory_bandwidth_optimization: bool,
}

impl Default for AdaptiveSimdConfig {
    fn default() -> Self {
        Self {
            auto_detect_hardware: true,
            enable_profiling: true,
            min_simdsize: 64,
            cachesize: 1000,
            benchmark_samples: 10,
            enable_hybrid_processing: false,
            alignment_requirements: SimdAlignment::Optimal,
            optimization_level: OptimizationLevel::Aggressive,
            adaptive_vectorization: true,
            memory_bandwidth_optimization: true,
        }
    }
}

/// SIMD alignment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimdAlignment {
    /// No special alignment requirements
    None,
    /// Basic alignment (16-byte)
    Basic,
    /// Optimal alignment for current hardware
    Optimal,
    /// Custom alignment requirement
    Custom(usize),
}

/// Optimization level settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Conservative optimization (focus on correctness)
    Conservative,
    /// Balanced optimization (good performance with safety)
    Balanced,
    /// Aggressive optimization (maximum performance)
    Aggressive,
    /// Extreme optimization (experimental features)
    Extreme,
}

/// Hardware capability detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Available SIMD instruction sets
    pub simd_instructions: Vec<SimdInstructionSet>,
    /// Vector register width
    pub vector_width: usize,
    /// Number of SIMD execution units
    pub simd_units: usize,
    /// Cache hierarchy information
    pub cache_info: CacheHierarchy,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// CPU architecture
    pub cpu_architecture: CpuArchitecture,
    /// GPU availability
    pub gpu_available: bool,
    /// GPU compute capabilities
    pub gpu_capabilities: Option<GpuCapabilities>,
}

/// SIMD instruction sets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    /// SSE (128-bit)
    SSE,
    /// SSE2 (128-bit)
    SSE2,
    /// SSE3 (128-bit)
    SSE3,
    /// SSE4.1 (128-bit)
    SSE41,
    /// SSE4.2 (128-bit)
    SSE42,
    /// AVX (256-bit)
    AVX,
    /// AVX2 (256-bit)
    AVX2,
    /// AVX-512 (512-bit)
    AVX512,
    /// ARM NEON
    NEON,
    /// ARM SVE
    SVE,
}

/// Cache hierarchy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHierarchy {
    /// L1 cache size (bytes)
    pub l1size: usize,
    /// L2 cache size (bytes)
    pub l2size: usize,
    /// L3 cache size (bytes)
    pub l3size: usize,
    /// Cache line size (bytes)
    pub cache_linesize: usize,
    /// Cache associativity
    pub associativity: Vec<usize>,
}

/// CPU architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuArchitecture {
    /// x86 architecture
    X86,
    /// x86-64 architecture
    X86_64,
    /// ARM architecture
    ARM,
    /// ARM64 architecture
    ARM64,
    /// RISC-V architecture
    RISCV,
    /// Other architecture
    Other(String),
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// GPU compute units
    pub compute_units: usize,
    /// GPU memory (bytes)
    pub gpu_memory: usize,
    /// GPU memory bandwidth (GB/s)
    pub gpu_bandwidth: f64,
    /// Supported compute APIs
    pub compute_apis: Vec<String>,
}

/// SIMD optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdStrategy {
    /// Strategy name
    pub name: String,
    /// Target instruction set
    pub instruction_set: SimdInstructionSet,
    /// Vector width to use
    pub vector_width: usize,
    /// Memory access pattern
    pub memory_pattern: MemoryAccessPattern,
    /// Alignment strategy
    pub alignment: AlignmentStrategy,
    /// Unrolling factor
    pub unroll_factor: usize,
    /// Prefetch strategy
    pub prefetch_strategy: PrefetchStrategy,
    /// Expected performance gain
    pub expected_speedup: f64,
}

/// Memory access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    /// Sequential access
    Sequential,
    /// Strided access
    Strided { stride: usize },
    /// Random access
    Random,
    /// Blocked access
    Blocked { blocksize: usize },
    /// Tiled access
    Tiled { tilesize: (usize, usize) },
}

/// Alignment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentStrategy {
    /// Force alignment with padding
    ForceAlign,
    /// Use unaligned loads
    UnalignedLoads,
    /// Dynamic alignment checking
    DynamicAlign,
    /// Copy to aligned buffer
    CopyAlign,
}

/// Prefetch strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Software prefetching
    Software { distance: usize },
    /// Hardware prefetching hints
    Hardware,
    /// Adaptive prefetching
    Adaptive,
}

/// Performance metrics for SIMD operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdPerformanceMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Throughput (elements/second)
    pub throughput: f64,
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// SIMD utilization efficiency
    pub simd_efficiency: f64,
    /// Energy efficiency (operations/joule)
    pub energy_efficiency: Option<f64>,
}

/// SIMD optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdOptimizationResult<T> {
    /// Computed result
    pub result: T,
    /// Strategy used
    pub strategy_used: SimdStrategy,
    /// Performance metrics
    pub metrics: SimdPerformanceMetrics,
    /// Success status
    pub success: bool,
    /// Fallback information
    pub fallback_info: Option<FallbackInfo>,
}

/// Fallback information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackInfo {
    /// Reason for fallback
    pub reason: String,
    /// Fallback strategy used
    pub fallback_strategy: String,
    /// Performance impact
    pub performance_impact: f64,
}

/// Data characteristics for optimization selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Data size
    pub size: usize,
    /// Data type size (bytes)
    pub elementsize: usize,
    /// Memory alignment
    pub alignment: usize,
    /// Access pattern
    pub access_pattern: MemoryAccessPattern,
    /// Data locality score (0.0-1.0)
    pub locality_score: f64,
    /// Sparsity level (0.0-1.0)
    pub sparsity: Option<f64>,
    /// Value distribution characteristics
    pub value_distribution: ValueDistribution,
}

/// Value distribution characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueDistribution {
    /// Range of values
    pub value_range: (f64, f64),
    /// Presence of special values (NaN, infinity)
    pub has_special_values: bool,
    /// Clustering characteristics
    pub clustering: ClusteringInfo,
}

/// Clustering information for values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringInfo {
    /// Number of distinct clusters
    pub cluster_count: usize,
    /// Cluster density
    pub density: f64,
    /// Separation between clusters
    pub separation: f64,
}

/// Main adaptive SIMD optimization system
pub struct AdaptiveSimdOptimizer {
    config: AdaptiveSimdConfig,
    hardware_capabilities: HardwareCapabilities,
    strategy_cache: Arc<Mutex<HashMap<String, SimdStrategy>>>,
    performance_cache: Arc<Mutex<HashMap<String, SimdPerformanceMetrics>>>,
    benchmark_results: Arc<Mutex<HashMap<String, Vec<SimdPerformanceMetrics>>>>,
}

impl AdaptiveSimdOptimizer {
    /// Create new adaptive SIMD optimizer
    pub fn new(config: AdaptiveSimdConfig) -> StatsResult<Self> {
        let hardware_capabilities = Self::detect_hardware_capabilities()?;

        Ok(Self {
            config,
            hardware_capabilities,
            strategy_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_cache: Arc::new(Mutex::new(HashMap::new())),
            benchmark_results: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create with default configuration
    pub fn default() -> StatsResult<Self> {
        Self::new(AdaptiveSimdConfig::default())
    }

    /// Optimize vector operation using adaptive SIMD
    pub fn optimize_vector_operation<F, T>(
        &self,
        operation_name: &str,
        data: ArrayView1<F>,
        operation: impl Fn(&ArrayView1<F>, &SimdStrategy) -> StatsResult<T> + Send + Sync,
    ) -> StatsResult<SimdOptimizationResult<T>>
    where
        F: Float + NumCast + SimdUnifiedOps + Send + Sync + std::fmt::Display,
        T: Send + Sync + std::fmt::Display,
    {
        let data_characteristics = self.analyzedata_characteristics(&data)?;

        // Get or select optimal strategy
        let strategy = self.select_optimal_strategy(operation_name, &data_characteristics)?;

        // Execute operation with performance monitoring
        let start_time = Instant::now();
        let result = operation(&data, &strategy);
        let execution_time = start_time.elapsed();

        match result {
            Ok(value) => {
                let metrics = self.calculate_performance_metrics(
                    &data_characteristics,
                    &strategy,
                    execution_time,
                )?;

                // Update performance cache
                self.update_performance_cache(operation_name, &strategy, &metrics);

                Ok(SimdOptimizationResult {
                    result: value,
                    strategy_used: strategy,
                    metrics,
                    success: true,
                    fallback_info: None,
                })
            }
            Err(_e) => {
                // Try fallback strategy
                self.try_fallback_strategy(operation_name, data, operation, &strategy)
            }
        }
    }

    /// Optimize matrix operation using adaptive SIMD
    pub fn optimize_matrix_operation<F, T>(
        &self,
        operation_name: &str,
        data: ArrayView2<F>,
        operation: impl Fn(&ArrayView2<F>, &SimdStrategy) -> StatsResult<T> + Send + Sync,
    ) -> StatsResult<SimdOptimizationResult<T>>
    where
        F: Float + NumCast + SimdUnifiedOps + Send + Sync + std::fmt::Display,
        T: Send + Sync + std::fmt::Display,
    {
        let data_characteristics = self.analyze_matrix_characteristics(&data)?;
        let strategy =
            self.select_optimal_matrix_strategy(operation_name, &data_characteristics)?;

        let start_time = Instant::now();
        let result = operation(&data, &strategy);
        let execution_time = start_time.elapsed();

        match result {
            Ok(value) => {
                let metrics = self.calculate_matrix_performance_metrics(
                    &data_characteristics,
                    &strategy,
                    execution_time,
                )?;

                self.update_performance_cache(operation_name, &strategy, &metrics);

                Ok(SimdOptimizationResult {
                    result: value,
                    strategy_used: strategy,
                    metrics,
                    success: true,
                    fallback_info: None,
                })
            }
            Err(_e) => {
                // Implement matrix fallback strategy
                self.try_matrix_fallback_strategy(operation_name, data, operation, &strategy)
            }
        }
    }

    /// Detect hardware capabilities
    fn detect_hardware_capabilities() -> StatsResult<HardwareCapabilities> {
        // Simplified hardware detection - would use proper CPU feature detection
        let capabilities = HardwareCapabilities {
            simd_instructions: vec![
                SimdInstructionSet::SSE2,
                SimdInstructionSet::AVX,
                SimdInstructionSet::AVX2,
            ],
            vector_width: 256, // AVX2
            simd_units: 2,
            cache_info: CacheHierarchy {
                l1size: 32 * 1024,       // 32KB
                l2size: 256 * 1024,      // 256KB
                l3size: 8 * 1024 * 1024, // 8MB
                cache_linesize: 64,
                associativity: vec![8, 8, 16],
            },
            memory_bandwidth: 50.0, // 50 GB/s
            cpu_architecture: CpuArchitecture::X86_64,
            gpu_available: false,
            gpu_capabilities: None,
        };

        Ok(capabilities)
    }

    /// Analyze data characteristics
    fn analyzedata_characteristics<F>(
        &self,
        data: &ArrayView1<F>,
    ) -> StatsResult<DataCharacteristics>
    where
        F: Float + NumCast + std::fmt::Display,
    {
        let size = data.len();
        let elementsize = std::mem::size_of::<F>();

        // Check alignment
        let alignment = (data.as_ptr() as usize) % 32; // Check 32-byte alignment

        // Analyze value distribution
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();
        let mut has_special = false;

        for &value in data.iter() {
            if value.is_nan() || value.is_infinite() {
                has_special = true;
            } else {
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
        }

        let value_distribution = ValueDistribution {
            value_range: (
                min_val.to_f64().unwrap_or(0.0),
                max_val.to_f64().unwrap_or(0.0),
            ),
            has_special_values: has_special,
            clustering: ClusteringInfo {
                cluster_count: 1, // Simplified
                density: 1.0,
                separation: 0.0,
            },
        };

        Ok(DataCharacteristics {
            size,
            elementsize,
            alignment,
            access_pattern: MemoryAccessPattern::Sequential,
            locality_score: 1.0, // Assume good locality for contiguous arrays
            sparsity: None,
            value_distribution,
        })
    }

    /// Analyze matrix characteristics
    fn analyze_matrix_characteristics<F>(
        &self,
        data: &ArrayView2<F>,
    ) -> StatsResult<DataCharacteristics>
    where
        F: Float + NumCast + std::fmt::Display,
    {
        let size = data.len();
        let elementsize = std::mem::size_of::<F>();

        // Check if matrix is C-contiguous or Fortran-contiguous
        let access_pattern = if data.is_standard_layout() {
            MemoryAccessPattern::Sequential
        } else {
            MemoryAccessPattern::Strided {
                stride: data.strides()[0] as usize,
            }
        };

        // Calculate sparsity
        let zero_count = data.iter().filter(|&&x| x == F::zero()).count();
        let sparsity = if size > 0 {
            Some(zero_count as f64 / size as f64)
        } else {
            None
        };

        Ok(DataCharacteristics {
            size,
            elementsize,
            alignment: (data.as_ptr() as usize) % 32,
            access_pattern,
            locality_score: if data.is_standard_layout() { 1.0 } else { 0.5 },
            sparsity,
            value_distribution: ValueDistribution {
                value_range: (0.0, 1.0), // Simplified
                has_special_values: false,
                clustering: ClusteringInfo {
                    cluster_count: 1,
                    density: 1.0,
                    separation: 0.0,
                },
            },
        })
    }

    /// Select optimal SIMD strategy
    fn select_optimal_strategy(
        &self,
        operation_name: &str,
        characteristics: &DataCharacteristics,
    ) -> StatsResult<SimdStrategy> {
        let cache_key = format!(
            "{}_{}_{}",
            operation_name, characteristics.size, characteristics.elementsize
        );

        // Check cache first
        if let Ok(cache) = self.strategy_cache.lock() {
            if let Some(strategy) = cache.get(&cache_key) {
                return Ok(strategy.clone());
            }
        }

        // Generate candidate strategies
        let candidates = self.generate_candidate_strategies(characteristics)?;

        // Select best strategy based on characteristics and hardware
        let best_strategy = self.evaluate_strategies(&candidates, characteristics)?;

        // Cache the result
        if let Ok(mut cache) = self.strategy_cache.lock() {
            cache.insert(cache_key, best_strategy.clone());

            // Maintain cache size
            if cache.len() > self.config.cachesize {
                let oldest_key = cache.keys().next().cloned();
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                }
            }
        }

        Ok(best_strategy)
    }

    /// Select optimal matrix strategy
    fn select_optimal_matrix_strategy(
        &self,
        operation_name: &str,
        characteristics: &DataCharacteristics,
    ) -> StatsResult<SimdStrategy> {
        // For matrix operations, consider tiling and blocking strategies
        let mut strategy = self.select_optimal_strategy(operation_name, characteristics)?;

        // Adjust for matrix-specific optimizations
        if characteristics.size > 1000000 {
            // Large matrices
            strategy.memory_pattern = MemoryAccessPattern::Tiled { tilesize: (64, 64) };
            strategy.prefetch_strategy = PrefetchStrategy::Software { distance: 8 };
        } else if matches!(
            characteristics.access_pattern,
            MemoryAccessPattern::Strided { .. }
        ) {
            strategy.memory_pattern = MemoryAccessPattern::Blocked { blocksize: 256 };
        }

        Ok(strategy)
    }

    /// Generate candidate SIMD strategies
    fn generate_candidate_strategies(
        &self,
        characteristics: &DataCharacteristics,
    ) -> StatsResult<Vec<SimdStrategy>> {
        let mut candidates = Vec::new();

        // Generate strategies based on available instruction sets
        for instruction_set in &self.hardware_capabilities.simd_instructions {
            let vector_width = match instruction_set {
                SimdInstructionSet::SSE | SimdInstructionSet::SSE2 => 128,
                SimdInstructionSet::AVX | SimdInstructionSet::AVX2 => 256,
                SimdInstructionSet::AVX512 => 512,
                SimdInstructionSet::NEON => 128,
                _ => 128,
            };

            // Conservative strategy
            candidates.push(SimdStrategy {
                name: format!("{:?}_conservative", instruction_set),
                instruction_set: instruction_set.clone(),
                vector_width,
                memory_pattern: characteristics.access_pattern.clone(),
                alignment: if characteristics.alignment == 0 {
                    AlignmentStrategy::ForceAlign
                } else {
                    AlignmentStrategy::UnalignedLoads
                },
                unroll_factor: 2,
                prefetch_strategy: PrefetchStrategy::None,
                expected_speedup: 2.0,
            });

            // Aggressive strategy
            if matches!(
                self.config.optimization_level,
                OptimizationLevel::Aggressive | OptimizationLevel::Extreme
            ) {
                candidates.push(SimdStrategy {
                    name: format!("{:?}_aggressive", instruction_set),
                    instruction_set: instruction_set.clone(),
                    vector_width,
                    memory_pattern: characteristics.access_pattern.clone(),
                    alignment: AlignmentStrategy::DynamicAlign,
                    unroll_factor: 4,
                    prefetch_strategy: if characteristics.size > 10000 {
                        PrefetchStrategy::Software { distance: 4 }
                    } else {
                        PrefetchStrategy::None
                    },
                    expected_speedup: 4.0,
                });
            }
        }

        Ok(candidates)
    }

    /// Evaluate strategies and select the best one
    fn evaluate_strategies(
        &self,
        candidates: &[SimdStrategy],
        characteristics: &DataCharacteristics,
    ) -> StatsResult<SimdStrategy> {
        let mut best_strategy = candidates[0].clone();
        let mut best_score = 0.0;

        for strategy in candidates {
            let score = self.calculate_strategy_score(strategy, characteristics);
            if score > best_score {
                best_score = score;
                best_strategy = strategy.clone();
            }
        }

        Ok(best_strategy)
    }

    /// Calculate strategy score based on characteristics
    fn calculate_strategy_score(
        &self,
        strategy: &SimdStrategy,
        characteristics: &DataCharacteristics,
    ) -> f64 {
        let mut score = strategy.expected_speedup;

        // Adjust score based on data characteristics
        if characteristics.size < self.config.min_simdsize {
            score *= 0.5; // Penalty for small data
        }

        // Bonus for good alignment
        if characteristics.alignment == 0
            && matches!(strategy.alignment, AlignmentStrategy::ForceAlign)
        {
            score *= 1.2;
        }

        // Penalty for complex memory patterns
        match &characteristics.access_pattern {
            MemoryAccessPattern::Sequential => score *= 1.0,
            MemoryAccessPattern::Strided { .. } => score *= 0.8,
            MemoryAccessPattern::Random => score *= 0.5,
            _ => score *= 0.7,
        }

        // Hardware compatibility bonus
        if self
            .hardware_capabilities
            .simd_instructions
            .contains(&strategy.instruction_set)
        {
            score *= 1.5;
        }

        score
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        characteristics: &DataCharacteristics,
        strategy: &SimdStrategy,
        execution_time: Duration,
    ) -> StatsResult<SimdPerformanceMetrics> {
        let throughput = characteristics.size as f64 / execution_time.as_secs_f64();

        // Estimate bandwidth utilization
        let bytes_processed = characteristics.size * characteristics.elementsize;
        let bandwidth_used = bytes_processed as f64 / execution_time.as_secs_f64() / 1e9; // GB/s
        let bandwidth_utilization = bandwidth_used / self.hardware_capabilities.memory_bandwidth;

        // Estimate SIMD efficiency
        let theoretical_max = strategy.vector_width / (characteristics.elementsize * 8); // elements per vector
        let actual_vectors = characteristics.size / theoretical_max;
        let simd_efficiency = if actual_vectors > 0 {
            characteristics.size as f64 / (actual_vectors * theoretical_max) as f64
        } else {
            0.0
        };

        Ok(SimdPerformanceMetrics {
            execution_time,
            throughput,
            bandwidth_utilization: bandwidth_utilization.min(1.0),
            cache_hit_rate: 0.9, // Placeholder
            simd_efficiency: simd_efficiency.min(1.0),
            energy_efficiency: None, // Would require hardware energy monitoring
        })
    }

    /// Calculate matrix performance metrics
    fn calculate_matrix_performance_metrics(
        &self,
        characteristics: &DataCharacteristics,
        strategy: &SimdStrategy,
        execution_time: Duration,
    ) -> StatsResult<SimdPerformanceMetrics> {
        // Similar to vector metrics but adjusted for matrix operations
        let mut metrics =
            self.calculate_performance_metrics(characteristics, strategy, execution_time)?;

        // Adjust cache hit rate based on matrix access pattern
        metrics.cache_hit_rate = match &characteristics.access_pattern {
            MemoryAccessPattern::Sequential => 0.95,
            MemoryAccessPattern::Strided { .. } => 0.8,
            MemoryAccessPattern::Tiled { .. } => 0.9,
            _ => 0.7,
        };

        Ok(metrics)
    }

    /// Try fallback strategy on failure
    fn try_fallback_strategy<F, T>(
        &self,
        _operation_name: &str,
        data: ArrayView1<F>,
        operation: impl Fn(&ArrayView1<F>, &SimdStrategy) -> StatsResult<T> + Send + Sync,
        failed_strategy: &SimdStrategy,
    ) -> StatsResult<SimdOptimizationResult<T>>
    where
        F: Float + NumCast + SimdUnifiedOps + Send + Sync + std::fmt::Display,
        T: Send + Sync + std::fmt::Display,
    {
        // Create a conservative fallback _strategy
        let fallback_strategy = SimdStrategy {
            name: "fallback_conservative".to_string(),
            instruction_set: SimdInstructionSet::SSE2, // Most widely supported
            vector_width: 128,
            memory_pattern: MemoryAccessPattern::Sequential,
            alignment: AlignmentStrategy::UnalignedLoads,
            unroll_factor: 1,
            prefetch_strategy: PrefetchStrategy::None,
            expected_speedup: 1.0,
        };

        let start_time = Instant::now();
        match operation(&data, &fallback_strategy) {
            Ok(result) => {
                let execution_time = start_time.elapsed();
                let characteristics = self.analyzedata_characteristics(&data)?;
                let metrics = self.calculate_performance_metrics(
                    &characteristics,
                    &fallback_strategy,
                    execution_time,
                )?;

                Ok(SimdOptimizationResult {
                    result,
                    strategy_used: fallback_strategy,
                    metrics,
                    success: true,
                    fallback_info: Some(FallbackInfo {
                        reason: format!("Primary _strategy '{}' failed", failed_strategy.name),
                        fallback_strategy: "conservative_sse2".to_string(),
                        performance_impact: 0.5, // Estimated 50% slower
                    }),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Try matrix fallback strategy
    fn try_matrix_fallback_strategy<F, T>(
        &self,
        _operation_name: &str,
        data: ArrayView2<F>,
        operation: impl Fn(&ArrayView2<F>, &SimdStrategy) -> StatsResult<T> + Send + Sync,
        failed_strategy: &SimdStrategy,
    ) -> StatsResult<SimdOptimizationResult<T>>
    where
        F: Float + NumCast + SimdUnifiedOps + Send + Sync + std::fmt::Display,
        T: Send + Sync + std::fmt::Display,
    {
        // Similar to vector fallback but for matrices
        let fallback_strategy = SimdStrategy {
            name: "matrix_fallback_conservative".to_string(),
            instruction_set: SimdInstructionSet::SSE2,
            vector_width: 128,
            memory_pattern: MemoryAccessPattern::Sequential,
            alignment: AlignmentStrategy::UnalignedLoads,
            unroll_factor: 1,
            prefetch_strategy: PrefetchStrategy::None,
            expected_speedup: 1.0,
        };

        let start_time = Instant::now();
        match operation(&data, &fallback_strategy) {
            Ok(result) => {
                let execution_time = start_time.elapsed();
                let characteristics = self.analyze_matrix_characteristics(&data)?;
                let metrics = self.calculate_matrix_performance_metrics(
                    &characteristics,
                    &fallback_strategy,
                    execution_time,
                )?;

                Ok(SimdOptimizationResult {
                    result,
                    strategy_used: fallback_strategy,
                    metrics,
                    success: true,
                    fallback_info: Some(FallbackInfo {
                        reason: format!(
                            "Primary matrix _strategy '{}' failed",
                            failed_strategy.name
                        ),
                        fallback_strategy: "conservative_matrix_sse2".to_string(),
                        performance_impact: 0.6,
                    }),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Update performance cache
    fn update_performance_cache(
        &self,
        operation_name: &str,
        strategy: &SimdStrategy,
        metrics: &SimdPerformanceMetrics,
    ) {
        if !self.config.enable_profiling {
            return;
        }

        let cache_key = format!("{}_{}", operation_name, strategy.name);

        if let Ok(mut cache) = self.performance_cache.lock() {
            cache.insert(cache_key.clone(), metrics.clone());
        }

        // Also update benchmark results for learning
        if let Ok(mut benchmarks) = self.benchmark_results.lock() {
            benchmarks
                .entry(cache_key)
                .or_insert_with(Vec::new)
                .push(metrics.clone());
        }
    }

    /// Get performance statistics
    pub fn get_performance_statistics(&self) -> PerformanceStatistics {
        let cache = self.performance_cache.lock().unwrap();
        let _benchmarks = self.benchmark_results.lock().unwrap();

        let total_operations = cache.len();
        let avg_speedup = if !cache.is_empty() {
            cache.values().map(|m| m.simd_efficiency).sum::<f64>() / cache.len() as f64
        } else {
            0.0
        };

        let best_strategies: Vec<(String, f64)> = cache
            .iter()
            .map(|(name, metrics)| (name.clone(), metrics.simd_efficiency))
            .collect();

        PerformanceStatistics {
            total_operations,
            average_speedup: avg_speedup,
            best_strategies,
            hardware_utilization: self.calculate_hardware_utilization(&cache),
        }
    }

    /// Calculate hardware utilization
    fn calculate_hardware_utilization(
        &self,
        cache: &HashMap<String, SimdPerformanceMetrics>,
    ) -> HardwareUtilization {
        let avg_bandwidth = if !cache.is_empty() {
            cache.values().map(|m| m.bandwidth_utilization).sum::<f64>() / cache.len() as f64
        } else {
            0.0
        };

        let avg_cache_hit_rate = if !cache.is_empty() {
            cache.values().map(|m| m.cache_hit_rate).sum::<f64>() / cache.len() as f64
        } else {
            0.0
        };

        HardwareUtilization {
            simd_utilization: 0.8, // Placeholder
            memory_bandwidth_utilization: avg_bandwidth,
            cache_efficiency: avg_cache_hit_rate,
            energy_efficiency: None,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Total operations performed
    pub total_operations: usize,
    /// Average speedup achieved
    pub average_speedup: f64,
    /// Best performing strategies
    pub best_strategies: Vec<(String, f64)>,
    /// Hardware utilization metrics
    pub hardware_utilization: HardwareUtilization,
}

/// Hardware utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareUtilization {
    /// SIMD unit utilization (0.0-1.0)
    pub simd_utilization: f64,
    /// Memory bandwidth utilization (0.0-1.0)
    pub memory_bandwidth_utilization: f64,
    /// Cache efficiency (0.0-1.0)
    pub cache_efficiency: f64,
    /// Energy efficiency (operations/joule)
    pub energy_efficiency: Option<f64>,
}

/// Convenience functions for adaptive SIMD optimization
#[allow(dead_code)]
pub fn create_adaptive_simd_optimizer() -> StatsResult<AdaptiveSimdOptimizer> {
    AdaptiveSimdOptimizer::default()
}

#[allow(dead_code)]
pub fn optimize_simd_operation<F, T>(
    operation_name: &str,
    data: ArrayView1<F>,
    operation: impl Fn(&ArrayView1<F>, &SimdStrategy) -> StatsResult<T> + Send + Sync,
) -> StatsResult<SimdOptimizationResult<T>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync + std::fmt::Display,
    T: Send + Sync + std::fmt::Display,
{
    let optimizer = AdaptiveSimdOptimizer::default()?;
    optimizer.optimize_vector_operation(operation_name, data, operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_simd_config() {
        let config = AdaptiveSimdConfig::default();
        assert!(config.auto_detect_hardware);
        assert!(config.enable_profiling);
        assert!(config.min_simdsize > 0);
    }

    #[test]
    fn test_hardware_detection() {
        let capabilities = AdaptiveSimdOptimizer::detect_hardware_capabilities().unwrap();
        assert!(!capabilities.simd_instructions.is_empty());
        assert!(capabilities.vector_width > 0);
    }

    #[test]
    fn testdata_characteristics_analysis() {
        let optimizer = AdaptiveSimdOptimizer::default().unwrap();
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

        let characteristics = optimizer.analyzedata_characteristics(&data.view()).unwrap();
        assert_eq!(characteristics.size, 5);
        assert_eq!(characteristics.elementsize, 8); // f64
    }

    #[test]
    fn test_strategy_generation() {
        let optimizer = AdaptiveSimdOptimizer::default().unwrap();
        let characteristics = DataCharacteristics {
            size: 1000,
            elementsize: 8,
            alignment: 0,
            access_pattern: MemoryAccessPattern::Sequential,
            locality_score: 1.0,
            sparsity: None,
            value_distribution: ValueDistribution {
                value_range: (0.0, 1.0),
                has_special_values: false,
                clustering: ClusteringInfo {
                    cluster_count: 1,
                    density: 1.0,
                    separation: 0.0,
                },
            },
        };

        let strategies = optimizer
            .generate_candidate_strategies(&characteristics)
            .unwrap();
        assert!(!strategies.is_empty());
    }

    #[test]
    fn test_strategy_selection() {
        let optimizer = AdaptiveSimdOptimizer::default().unwrap();
        let characteristics = DataCharacteristics {
            size: 1000,
            elementsize: 8,
            alignment: 0,
            access_pattern: MemoryAccessPattern::Sequential,
            locality_score: 1.0,
            sparsity: None,
            value_distribution: ValueDistribution {
                value_range: (0.0, 1.0),
                has_special_values: false,
                clustering: ClusteringInfo {
                    cluster_count: 1,
                    density: 1.0,
                    separation: 0.0,
                },
            },
        };

        let strategy = optimizer
            .select_optimal_strategy("test_op", &characteristics)
            .unwrap();
        assert!(!strategy.name.is_empty());
        assert!(strategy.expected_speedup > 0.0);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let optimizer = AdaptiveSimdOptimizer::default().unwrap();
        let characteristics = DataCharacteristics {
            size: 1000,
            elementsize: 8,
            alignment: 0,
            access_pattern: MemoryAccessPattern::Sequential,
            locality_score: 1.0,
            sparsity: None,
            value_distribution: ValueDistribution {
                value_range: (0.0, 1.0),
                has_special_values: false,
                clustering: ClusteringInfo {
                    cluster_count: 1,
                    density: 1.0,
                    separation: 0.0,
                },
            },
        };

        let strategy = SimdStrategy {
            name: "test_strategy".to_string(),
            instruction_set: SimdInstructionSet::AVX2,
            vector_width: 256,
            memory_pattern: MemoryAccessPattern::Sequential,
            alignment: AlignmentStrategy::ForceAlign,
            unroll_factor: 2,
            prefetch_strategy: PrefetchStrategy::None,
            expected_speedup: 2.0,
        };

        let metrics = optimizer
            .calculate_performance_metrics(&characteristics, &strategy, Duration::from_millis(10))
            .unwrap();

        assert!(metrics.throughput > 0.0);
        assert!(metrics.simd_efficiency >= 0.0 && metrics.simd_efficiency <= 1.0);
    }
}
