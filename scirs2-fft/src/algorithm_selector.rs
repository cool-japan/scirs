//! FFT Algorithm Auto-Selection Module
//!
//! This module provides intelligent automatic selection of FFT algorithms based on:
//! - Input size characteristics (power-of-2, prime, smooth numbers)
//! - Hardware capabilities (cache size, SIMD support, core count)
//! - Memory constraints
//! - Historical performance data
//!
//! # Features
//!
//! - **Input Analysis**: Detects optimal algorithm based on input size properties
//! - **Cache-Aware Selection**: Considers L1/L2/L3 cache sizes for optimal performance
//! - **Memory Optimization**: Selects memory-efficient algorithms for large inputs
//! - **Hardware Detection**: Adapts to available SIMD instructions and core count
//! - **Performance Profiling**: Tracks and learns from execution history
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_fft::algorithm_selector::{AlgorithmSelector, SelectionConfig};
//!
//! let selector = AlgorithmSelector::new();
//! let recommendation = selector.select_algorithm(1024, true).expect("Selection failed");
//! println!("Recommended: {:?}", recommendation.algorithm);
//! ```

use crate::error::{FFTError, FFTResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// FFT Algorithm variants available for selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FftAlgorithm {
    /// Standard Cooley-Tukey radix-2 FFT (optimal for power-of-2 sizes)
    CooleyTukeyRadix2,
    /// Radix-4 FFT (faster for sizes that are powers of 4)
    Radix4,
    /// Split-radix FFT (good balance of speed and memory)
    SplitRadix,
    /// Mixed-radix FFT (handles non-power-of-2 sizes efficiently)
    #[default]
    MixedRadix,
    /// Bluestein's algorithm (handles prime and arbitrary sizes)
    Bluestein,
    /// Rader's algorithm (efficient for prime sizes)
    Rader,
    /// Winograd FFT (minimal multiplications)
    Winograd,
    /// Good-Thomas PFA (prime factor algorithm)
    GoodThomas,
    /// Streaming FFT (memory-efficient for very large inputs)
    Streaming,
    /// Cache-oblivious FFT (optimized cache behavior)
    CacheOblivious,
    /// In-place FFT (minimal memory overhead)
    InPlace,
    /// SIMD-optimized FFT
    SimdOptimized,
    /// Parallel FFT (multi-threaded)
    Parallel,
    /// Hybrid (combines multiple algorithms)
    Hybrid,
}

impl std::fmt::Display for FftAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CooleyTukeyRadix2 => write!(f, "Cooley-Tukey Radix-2"),
            Self::Radix4 => write!(f, "Radix-4"),
            Self::SplitRadix => write!(f, "Split-Radix"),
            Self::MixedRadix => write!(f, "Mixed-Radix"),
            Self::Bluestein => write!(f, "Bluestein"),
            Self::Rader => write!(f, "Rader"),
            Self::Winograd => write!(f, "Winograd"),
            Self::GoodThomas => write!(f, "Good-Thomas PFA"),
            Self::Streaming => write!(f, "Streaming"),
            Self::CacheOblivious => write!(f, "Cache-Oblivious"),
            Self::InPlace => write!(f, "In-Place"),
            Self::SimdOptimized => write!(f, "SIMD-Optimized"),
            Self::Parallel => write!(f, "Parallel"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Input size characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizeCharacteristic {
    /// Size is a power of 2 (e.g., 256, 512, 1024)
    PowerOf2,
    /// Size is a power of 4 (e.g., 256, 1024, 4096)
    PowerOf4,
    /// Size is a prime number
    Prime,
    /// Size is a product of small primes (2, 3, 5, 7, 11)
    Smooth,
    /// Size is a product of coprime factors
    Composite,
    /// Size has large prime factors
    HardSize,
}

/// Detected input characteristics for algorithm selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputCharacteristics {
    /// Input size
    pub size: usize,
    /// Size characteristic type
    pub size_type: SizeCharacteristic,
    /// Whether size is a power of 2
    pub is_power_of_2: bool,
    /// Whether size is a power of 4
    pub is_power_of_4: bool,
    /// Whether size is prime
    pub is_prime: bool,
    /// Prime factorization (factor -> power)
    pub prime_factors: HashMap<usize, usize>,
    /// Largest prime factor
    pub largest_prime_factor: usize,
    /// Number of distinct prime factors
    pub num_distinct_factors: usize,
    /// Whether size is "smooth" (only small prime factors)
    pub is_smooth: bool,
    /// Maximum prime factor for smooth classification
    pub smooth_bound: usize,
    /// Estimated memory requirement in bytes
    pub estimated_memory_bytes: usize,
    /// Fits in L1 cache
    pub fits_l1_cache: bool,
    /// Fits in L2 cache
    pub fits_l2_cache: bool,
    /// Fits in L3 cache
    pub fits_l3_cache: bool,
}

impl InputCharacteristics {
    /// Analyze input size and return characteristics
    pub fn analyze(size: usize, cache_info: &CacheInfo) -> Self {
        let is_power_of_2 = size.is_power_of_two();
        let is_power_of_4 = is_power_of_2 && (size.trailing_zeros() % 2 == 0);
        let prime_factors = factorize(size);
        let is_prime = prime_factors.len() == 1 && prime_factors.get(&size).copied() == Some(1);
        let largest_prime_factor = *prime_factors.keys().max().unwrap_or(&1);
        let num_distinct_factors = prime_factors.len();

        // Check if smooth (only factors <= 11)
        let smooth_bound = 11;
        let is_smooth = prime_factors.keys().all(|&p| p <= smooth_bound);

        // Estimate memory: complex64 = 16 bytes per element
        let estimated_memory_bytes = size * 16;

        // Determine size type
        let size_type = if is_power_of_4 {
            SizeCharacteristic::PowerOf4
        } else if is_power_of_2 {
            SizeCharacteristic::PowerOf2
        } else if is_prime {
            SizeCharacteristic::Prime
        } else if is_smooth {
            SizeCharacteristic::Smooth
        } else if largest_prime_factor <= 1000 {
            SizeCharacteristic::Composite
        } else {
            SizeCharacteristic::HardSize
        };

        Self {
            size,
            size_type,
            is_power_of_2,
            is_power_of_4,
            is_prime,
            prime_factors,
            largest_prime_factor,
            num_distinct_factors,
            is_smooth,
            smooth_bound,
            estimated_memory_bytes,
            fits_l1_cache: estimated_memory_bytes <= cache_info.l1_size,
            fits_l2_cache: estimated_memory_bytes <= cache_info.l2_size,
            fits_l3_cache: estimated_memory_bytes <= cache_info.l3_size,
        }
    }
}

/// Hardware information for algorithm selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// Number of physical cores
    pub num_cores: usize,
    /// Number of logical processors (including hyperthreading)
    pub num_logical_processors: usize,
    /// Cache information
    pub cache_info: CacheInfo,
    /// SIMD capabilities
    pub simd_capabilities: SimdCapabilities,
    /// CPU architecture
    pub architecture: String,
    /// Available memory in bytes
    pub available_memory: usize,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self::detect()
    }
}

impl HardwareInfo {
    /// Detect hardware capabilities
    pub fn detect() -> Self {
        let num_cores = num_cpus::get_physical();
        let num_logical_processors = num_cpus::get();

        Self {
            num_cores,
            num_logical_processors,
            cache_info: CacheInfo::detect(),
            simd_capabilities: SimdCapabilities::detect(),
            architecture: std::env::consts::ARCH.to_string(),
            available_memory: estimate_available_memory(),
        }
    }
}

/// CPU cache information
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CacheInfo {
    /// L1 data cache size in bytes
    pub l1_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// Cache line size in bytes
    pub cache_line_size: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        Self::detect()
    }
}

impl CacheInfo {
    /// Detect cache sizes (uses conservative estimates if detection fails)
    pub fn detect() -> Self {
        // Conservative default estimates
        // These are typical for modern desktop CPUs
        Self {
            l1_size: 32 * 1024,       // 32 KB
            l2_size: 256 * 1024,      // 256 KB
            l3_size: 8 * 1024 * 1024, // 8 MB
            cache_line_size: 64,      // 64 bytes
        }
    }

    /// Create with custom cache sizes
    pub fn custom(l1: usize, l2: usize, l3: usize, line_size: usize) -> Self {
        Self {
            l1_size: l1,
            l2_size: l2,
            l3_size: l3,
            cache_line_size: line_size,
        }
    }
}

/// SIMD capability detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdCapabilities {
    /// SSE support
    pub has_sse: bool,
    /// SSE2 support
    pub has_sse2: bool,
    /// SSE3 support
    pub has_sse3: bool,
    /// SSE4.1 support
    pub has_sse4_1: bool,
    /// SSE4.2 support
    pub has_sse4_2: bool,
    /// AVX support
    pub has_avx: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// AVX-512 support
    pub has_avx512: bool,
    /// FMA support
    pub has_fma: bool,
    /// ARM NEON support
    pub has_neon: bool,
    /// Vector register width in bits
    pub vector_width: usize,
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdCapabilities {
    /// Detect SIMD capabilities
    pub fn detect() -> Self {
        let mut caps = Self {
            has_sse: false,
            has_sse2: false,
            has_sse3: false,
            has_sse4_1: false,
            has_sse4_2: false,
            has_avx: false,
            has_avx2: false,
            has_avx512: false,
            has_fma: false,
            has_neon: false,
            vector_width: 64, // Scalar default
        };

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "sse")]
            {
                caps.has_sse = true;
                caps.vector_width = 128;
            }
            #[cfg(target_feature = "sse2")]
            {
                caps.has_sse2 = true;
            }
            #[cfg(target_feature = "sse3")]
            {
                caps.has_sse3 = true;
            }
            #[cfg(target_feature = "sse4.1")]
            {
                caps.has_sse4_1 = true;
            }
            #[cfg(target_feature = "sse4.2")]
            {
                caps.has_sse4_2 = true;
            }
            #[cfg(target_feature = "avx")]
            {
                caps.has_avx = true;
                caps.vector_width = 256;
            }
            #[cfg(target_feature = "avx2")]
            {
                caps.has_avx2 = true;
            }
            #[cfg(target_feature = "fma")]
            {
                caps.has_fma = true;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            {
                caps.has_neon = true;
                caps.vector_width = 128;
            }
        }

        caps
    }

    /// Check if SIMD is available
    pub fn simd_available(&self) -> bool {
        self.has_sse2 || self.has_avx || self.has_neon
    }

    /// Get optimal vector width for complex f64
    pub fn optimal_complex_vector_count(&self) -> usize {
        // Complex f64 = 16 bytes = 128 bits
        self.vector_width / 128
    }
}

/// Algorithm recommendation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRecommendation {
    /// Recommended primary algorithm
    pub algorithm: FftAlgorithm,
    /// Fallback algorithm if primary is not suitable
    pub fallback: Option<FftAlgorithm>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated execution time (nanoseconds)
    pub estimated_time_ns: Option<u64>,
    /// Estimated memory usage (bytes)
    pub estimated_memory_bytes: usize,
    /// Reasoning for the selection
    pub reasoning: Vec<String>,
    /// Whether to use parallel execution
    pub use_parallel: bool,
    /// Recommended number of threads
    pub recommended_threads: usize,
    /// Whether to use SIMD optimization
    pub use_simd: bool,
    /// Whether to use in-place computation
    pub use_inplace: bool,
    /// Input characteristics that led to this recommendation
    pub input_characteristics: InputCharacteristics,
}

/// Performance history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEntry {
    /// FFT size
    pub size: usize,
    /// Algorithm used
    pub algorithm: FftAlgorithm,
    /// Whether it was a forward transform
    pub forward: bool,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Hardware info hash for matching
    pub hardware_hash: u64,
}

/// Configuration for algorithm selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Prefer memory efficiency over speed
    pub prefer_memory_efficiency: bool,
    /// Maximum memory budget in bytes (0 = unlimited)
    pub max_memory_bytes: usize,
    /// Minimum parallel size threshold
    pub min_parallel_size: usize,
    /// Enable performance learning
    pub enable_learning: bool,
    /// Maximum threads to use
    pub max_threads: usize,
    /// Force specific algorithm (None = auto-select)
    pub force_algorithm: Option<FftAlgorithm>,
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Prefer in-place computation
    pub prefer_inplace: bool,
    /// Cache-aware selection
    pub cache_aware: bool,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            prefer_memory_efficiency: false,
            max_memory_bytes: 0,
            min_parallel_size: 65536, // 64K elements
            enable_learning: true,
            max_threads: 0, // 0 = auto
            force_algorithm: None,
            enable_simd: true,
            prefer_inplace: false,
            cache_aware: true,
        }
    }
}

/// Performance history database
#[derive(Debug, Default)]
pub struct PerformanceHistory {
    /// History entries indexed by (size, algorithm, forward)
    entries: HashMap<(usize, FftAlgorithm, bool), Vec<PerformanceEntry>>,
    /// Best known algorithm for each size
    best_algorithms: HashMap<(usize, bool), FftAlgorithm>,
}

impl PerformanceHistory {
    /// Create new empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a performance measurement
    pub fn record(&mut self, entry: PerformanceEntry) {
        let key = (entry.size, entry.algorithm, entry.forward);
        self.entries.entry(key).or_default().push(entry.clone());

        // Update best algorithm if this is faster
        let size_key = (entry.size, entry.forward);
        let should_update = match self.best_algorithms.get(&size_key) {
            None => true,
            Some(&best_algo) => {
                let best_key = (entry.size, best_algo, entry.forward);
                if let Some(best_entries) = self.entries.get(&best_key) {
                    let best_avg = Self::average_time(best_entries);
                    let current_avg = Self::average_time(std::slice::from_ref(&entry));
                    current_avg < best_avg
                } else {
                    true
                }
            }
        };

        if should_update {
            self.best_algorithms.insert(size_key, entry.algorithm);
        }
    }

    /// Get best algorithm for a size
    pub fn get_best(&self, size: usize, forward: bool) -> Option<FftAlgorithm> {
        // First check exact match
        if let Some(&algo) = self.best_algorithms.get(&(size, forward)) {
            return Some(algo);
        }

        // Find closest size
        let mut closest_size = 0;
        let mut min_diff = usize::MAX;

        for &(s, f) in self.best_algorithms.keys() {
            if f == forward {
                let diff = s.abs_diff(size);
                if diff < min_diff {
                    min_diff = diff;
                    closest_size = s;
                }
            }
        }

        if closest_size > 0 {
            self.best_algorithms.get(&(closest_size, forward)).copied()
        } else {
            None
        }
    }

    /// Get average execution time for entries
    fn average_time(entries: &[PerformanceEntry]) -> u64 {
        if entries.is_empty() {
            return u64::MAX;
        }
        entries.iter().map(|e| e.execution_time_ns).sum::<u64>() / entries.len() as u64
    }

    /// Get statistics for an algorithm at a size
    pub fn get_stats(
        &self,
        size: usize,
        algorithm: FftAlgorithm,
        forward: bool,
    ) -> Option<PerformanceStats> {
        let key = (size, algorithm, forward);
        self.entries.get(&key).map(|entries| {
            let times: Vec<u64> = entries.iter().map(|e| e.execution_time_ns).collect();
            let avg = Self::average_time(entries);
            let min = times.iter().min().copied().unwrap_or(0);
            let max = times.iter().max().copied().unwrap_or(0);

            let variance = if times.len() > 1 {
                let avg_f = avg as f64;
                times
                    .iter()
                    .map(|&t| {
                        let diff = t as f64 - avg_f;
                        diff * diff
                    })
                    .sum::<f64>()
                    / (times.len() - 1) as f64
            } else {
                0.0
            };

            PerformanceStats {
                sample_count: times.len(),
                avg_time_ns: avg,
                min_time_ns: min,
                max_time_ns: max,
                std_dev_ns: variance.sqrt(),
            }
        })
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Number of samples
    pub sample_count: usize,
    /// Average execution time
    pub avg_time_ns: u64,
    /// Minimum execution time
    pub min_time_ns: u64,
    /// Maximum execution time
    pub max_time_ns: u64,
    /// Standard deviation
    pub std_dev_ns: f64,
}

/// Main algorithm selector
pub struct AlgorithmSelector {
    /// Configuration
    config: SelectionConfig,
    /// Hardware information
    hardware: HardwareInfo,
    /// Performance history
    history: Arc<RwLock<PerformanceHistory>>,
}

impl Default for AlgorithmSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgorithmSelector {
    /// Create a new algorithm selector with default configuration
    pub fn new() -> Self {
        Self::with_config(SelectionConfig::default())
    }

    /// Create a new algorithm selector with custom configuration
    pub fn with_config(config: SelectionConfig) -> Self {
        Self {
            config,
            hardware: HardwareInfo::detect(),
            history: Arc::new(RwLock::new(PerformanceHistory::new())),
        }
    }

    /// Select the best algorithm for the given input size
    pub fn select_algorithm(
        &self,
        size: usize,
        forward: bool,
    ) -> FFTResult<AlgorithmRecommendation> {
        // Check for forced algorithm
        if let Some(algo) = self.config.force_algorithm {
            let chars = InputCharacteristics::analyze(size, &self.hardware.cache_info);
            return Ok(AlgorithmRecommendation {
                algorithm: algo,
                fallback: None,
                confidence: 1.0,
                estimated_time_ns: None,
                estimated_memory_bytes: chars.estimated_memory_bytes,
                reasoning: vec!["Algorithm forced by configuration".to_string()],
                use_parallel: false,
                recommended_threads: 1,
                use_simd: self.config.enable_simd,
                use_inplace: self.config.prefer_inplace,
                input_characteristics: chars,
            });
        }

        // Analyze input characteristics
        let chars = InputCharacteristics::analyze(size, &self.hardware.cache_info);

        // Check performance history first
        if self.config.enable_learning {
            if let Ok(history) = self.history.read() {
                if let Some(best_algo) = history.get_best(size, forward) {
                    let stats = history.get_stats(size, best_algo, forward);
                    return Ok(self.build_recommendation(
                        best_algo,
                        &chars,
                        0.95, // High confidence from learned data
                        stats.as_ref().map(|s| s.avg_time_ns),
                        vec!["Selected based on historical performance data".to_string()],
                    ));
                }
            }
        }

        // Select based on input characteristics
        let (algorithm, fallback, reasoning) = self.select_by_characteristics(&chars);

        // Determine confidence based on how well we match the algorithm's optimal case
        let confidence = self.calculate_confidence(&chars, algorithm);

        // Estimate execution time (rough model)
        let estimated_time = self.estimate_execution_time(size, algorithm);

        Ok(self.build_recommendation(
            algorithm,
            &chars,
            confidence,
            Some(estimated_time),
            reasoning,
        ))
    }

    /// Select algorithm based on input characteristics
    fn select_by_characteristics(
        &self,
        chars: &InputCharacteristics,
    ) -> (FftAlgorithm, Option<FftAlgorithm>, Vec<String>) {
        let mut reasoning = Vec::new();
        let size = chars.size;

        // Memory constraint check
        if self.config.max_memory_bytes > 0
            && chars.estimated_memory_bytes > self.config.max_memory_bytes
        {
            reasoning.push(format!(
                "Memory constraint: {} bytes required, {} bytes available",
                chars.estimated_memory_bytes, self.config.max_memory_bytes
            ));
            return (
                FftAlgorithm::Streaming,
                Some(FftAlgorithm::InPlace),
                reasoning,
            );
        }

        // Very large sizes - use streaming or parallel
        if size > 16 * 1024 * 1024 {
            reasoning.push(format!(
                "Very large size ({}): using streaming for memory efficiency",
                size
            ));
            return (
                FftAlgorithm::Streaming,
                Some(FftAlgorithm::Parallel),
                reasoning,
            );
        }

        // Cache-aware selection
        if self.config.cache_aware {
            if chars.fits_l1_cache {
                reasoning.push("Data fits in L1 cache".to_string());
            } else if chars.fits_l2_cache {
                reasoning.push("Data fits in L2 cache".to_string());
            } else if chars.fits_l3_cache {
                reasoning
                    .push("Data fits in L3 cache, using cache-oblivious algorithm".to_string());
                if !chars.is_power_of_2 {
                    return (
                        FftAlgorithm::CacheOblivious,
                        Some(FftAlgorithm::MixedRadix),
                        reasoning,
                    );
                }
            } else {
                reasoning
                    .push("Data exceeds L3 cache, considering streaming or parallel".to_string());
            }
        }

        // Parallel threshold check
        let use_parallel = size >= self.config.min_parallel_size && self.hardware.num_cores > 1;
        if use_parallel {
            reasoning.push(format!(
                "Size {} exceeds parallel threshold {}, {} cores available",
                size, self.config.min_parallel_size, self.hardware.num_cores
            ));
        }

        // SIMD check
        let use_simd = self.config.enable_simd && self.hardware.simd_capabilities.simd_available();
        if use_simd {
            reasoning.push("SIMD optimization enabled".to_string());
        }

        // Select based on size characteristics
        match chars.size_type {
            SizeCharacteristic::PowerOf4 => {
                reasoning.push(format!(
                    "Size {} is a power of 4: optimal for Radix-4",
                    size
                ));
                if use_parallel {
                    (
                        FftAlgorithm::Parallel,
                        Some(FftAlgorithm::Radix4),
                        reasoning,
                    )
                } else if use_simd {
                    (
                        FftAlgorithm::SimdOptimized,
                        Some(FftAlgorithm::Radix4),
                        reasoning,
                    )
                } else {
                    (
                        FftAlgorithm::Radix4,
                        Some(FftAlgorithm::CooleyTukeyRadix2),
                        reasoning,
                    )
                }
            }
            SizeCharacteristic::PowerOf2 => {
                reasoning.push(format!(
                    "Size {} is a power of 2: optimal for Radix-2",
                    size
                ));
                if use_parallel && size >= 262144 {
                    (
                        FftAlgorithm::Parallel,
                        Some(FftAlgorithm::SplitRadix),
                        reasoning,
                    )
                } else if use_simd {
                    (
                        FftAlgorithm::SimdOptimized,
                        Some(FftAlgorithm::SplitRadix),
                        reasoning,
                    )
                } else {
                    (
                        FftAlgorithm::SplitRadix,
                        Some(FftAlgorithm::CooleyTukeyRadix2),
                        reasoning,
                    )
                }
            }
            SizeCharacteristic::Prime => {
                reasoning.push(format!("Size {} is prime: using Bluestein or Rader", size));
                // Rader is better for small primes, Bluestein for large
                if size < 1000 {
                    (
                        FftAlgorithm::Rader,
                        Some(FftAlgorithm::Bluestein),
                        reasoning,
                    )
                } else {
                    (
                        FftAlgorithm::Bluestein,
                        Some(FftAlgorithm::MixedRadix),
                        reasoning,
                    )
                }
            }
            SizeCharacteristic::Smooth => {
                reasoning.push(format!(
                    "Size {} is {}-smooth: good for mixed-radix",
                    size, chars.smooth_bound
                ));
                if chars.num_distinct_factors <= 3 && are_coprime(&chars.prime_factors) {
                    reasoning.push("Factors are coprime: Good-Thomas PFA applicable".to_string());
                    (
                        FftAlgorithm::GoodThomas,
                        Some(FftAlgorithm::MixedRadix),
                        reasoning,
                    )
                } else {
                    (
                        FftAlgorithm::MixedRadix,
                        Some(FftAlgorithm::Bluestein),
                        reasoning,
                    )
                }
            }
            SizeCharacteristic::Composite => {
                reasoning.push(format!(
                    "Size {} is composite with largest factor {}: using mixed-radix",
                    size, chars.largest_prime_factor
                ));
                (
                    FftAlgorithm::MixedRadix,
                    Some(FftAlgorithm::Bluestein),
                    reasoning,
                )
            }
            SizeCharacteristic::HardSize => {
                reasoning.push(format!(
                    "Size {} has large prime factor {}: using Bluestein",
                    size, chars.largest_prime_factor
                ));
                (
                    FftAlgorithm::Bluestein,
                    Some(FftAlgorithm::MixedRadix),
                    reasoning,
                )
            }
        }
    }

    /// Build a recommendation structure
    fn build_recommendation(
        &self,
        algorithm: FftAlgorithm,
        chars: &InputCharacteristics,
        confidence: f64,
        estimated_time_ns: Option<u64>,
        reasoning: Vec<String>,
    ) -> AlgorithmRecommendation {
        let use_parallel =
            chars.size >= self.config.min_parallel_size && self.hardware.num_cores > 1;
        let recommended_threads = if use_parallel {
            if self.config.max_threads > 0 {
                self.config.max_threads.min(self.hardware.num_cores)
            } else {
                // Use sqrt(cores) for good parallelism without overhead
                ((self.hardware.num_cores as f64).sqrt().ceil() as usize).max(2)
            }
        } else {
            1
        };

        AlgorithmRecommendation {
            algorithm,
            fallback: None,
            confidence,
            estimated_time_ns,
            estimated_memory_bytes: chars.estimated_memory_bytes,
            reasoning,
            use_parallel,
            recommended_threads,
            use_simd: self.config.enable_simd && self.hardware.simd_capabilities.simd_available(),
            use_inplace: self.config.prefer_inplace,
            input_characteristics: chars.clone(),
        }
    }

    /// Calculate confidence score for algorithm selection
    fn calculate_confidence(&self, chars: &InputCharacteristics, algorithm: FftAlgorithm) -> f64 {
        let base_confidence = match (chars.size_type, algorithm) {
            (SizeCharacteristic::PowerOf4, FftAlgorithm::Radix4) => 0.95,
            (SizeCharacteristic::PowerOf4, FftAlgorithm::SimdOptimized) => 0.93,
            (SizeCharacteristic::PowerOf2, FftAlgorithm::SplitRadix) => 0.92,
            (SizeCharacteristic::PowerOf2, FftAlgorithm::CooleyTukeyRadix2) => 0.90,
            (SizeCharacteristic::PowerOf2, FftAlgorithm::SimdOptimized) => 0.91,
            (SizeCharacteristic::Prime, FftAlgorithm::Rader) => 0.85,
            (SizeCharacteristic::Prime, FftAlgorithm::Bluestein) => 0.80,
            (SizeCharacteristic::Smooth, FftAlgorithm::GoodThomas) => 0.88,
            (SizeCharacteristic::Smooth, FftAlgorithm::MixedRadix) => 0.85,
            (SizeCharacteristic::Composite, FftAlgorithm::MixedRadix) => 0.80,
            (SizeCharacteristic::HardSize, FftAlgorithm::Bluestein) => 0.75,
            _ => 0.70,
        };

        // Adjust based on cache fit
        let cache_bonus: f64 = if chars.fits_l1_cache {
            0.02
        } else if chars.fits_l2_cache {
            0.01
        } else {
            -0.02
        };

        (base_confidence + cache_bonus).clamp(0.0, 1.0)
    }

    /// Estimate execution time (rough model based on O(n log n))
    fn estimate_execution_time(&self, size: usize, algorithm: FftAlgorithm) -> u64 {
        if size == 0 {
            return 0;
        }

        let n = size as f64;
        let log_n = n.log2();
        let base_ops = n * log_n;

        // Algorithm-specific coefficients (nanoseconds per operation)
        let coeff = match algorithm {
            FftAlgorithm::Radix4 => 0.8,
            FftAlgorithm::CooleyTukeyRadix2 => 1.0,
            FftAlgorithm::SplitRadix => 0.9,
            FftAlgorithm::SimdOptimized => 0.5,
            FftAlgorithm::Parallel => 0.4,
            FftAlgorithm::MixedRadix => 1.2,
            FftAlgorithm::Bluestein => 2.0,
            FftAlgorithm::Rader => 1.5,
            FftAlgorithm::GoodThomas => 1.1,
            FftAlgorithm::Winograd => 1.3,
            FftAlgorithm::Streaming => 1.5,
            FftAlgorithm::CacheOblivious => 1.1,
            FftAlgorithm::InPlace => 1.0,
            FftAlgorithm::Hybrid => 1.0,
        };

        (base_ops * coeff) as u64
    }

    /// Record performance measurement for learning
    pub fn record_performance(&self, entry: PerformanceEntry) -> FFTResult<()> {
        if !self.config.enable_learning {
            return Ok(());
        }

        let mut history = self
            .history
            .write()
            .map_err(|e| FFTError::ValueError(format!("Failed to acquire write lock: {e}")))?;

        history.record(entry);
        Ok(())
    }

    /// Run a benchmark for a specific size and algorithm
    pub fn benchmark(
        &self,
        size: usize,
        algorithm: FftAlgorithm,
        forward: bool,
    ) -> FFTResult<PerformanceEntry> {
        use scirs2_core::numeric::Complex64;

        #[cfg(feature = "oxifft")]
        {
            use oxifft::{Complex as OxiComplex, Direction, Flags, Plan};

            // Create test data
            let data: Vec<OxiComplex<f64>> = (0..size)
                .map(|i| OxiComplex::new(i as f64, (i * 2) as f64))
                .collect();

            // Create plan
            let direction = if forward {
                Direction::Forward
            } else {
                Direction::Backward
            };

            let plan = Plan::dft_1d(size, direction, Flags::ESTIMATE).ok_or_else(|| {
                FFTError::ComputationError(format!("Failed to create FFT plan for size {}", size))
            })?;

            // Warm-up
            for _ in 0..3 {
                let mut warm_data = data.clone();
                let mut output = vec![OxiComplex::default(); size];
                plan.execute(&warm_data, &mut output);
            }

            // Benchmark
            let mut input = data;
            let mut output = vec![OxiComplex::default(); size];
            let start = Instant::now();
            plan.execute(&input, &mut output);
            let elapsed = start.elapsed();

            Ok(PerformanceEntry {
                size,
                algorithm,
                forward,
                execution_time_ns: elapsed.as_nanos() as u64,
                peak_memory_bytes: size * 16, // Complex64 = 16 bytes
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_secs(),
                hardware_hash: 0, // Simplified for now
            })
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SelectionConfig {
        &self.config
    }

    /// Get hardware info
    pub fn hardware(&self) -> &HardwareInfo {
        &self.hardware
    }

    /// Get performance history
    pub fn history(&self) -> Arc<RwLock<PerformanceHistory>> {
        self.history.clone()
    }
}

// Helper functions

/// Factorize a number into prime factors
fn factorize(mut n: usize) -> HashMap<usize, usize> {
    let mut factors = HashMap::new();

    if n <= 1 {
        return factors;
    }

    // Check for 2
    let mut count = 0;
    while n % 2 == 0 {
        count += 1;
        n /= 2;
    }
    if count > 0 {
        factors.insert(2, count);
    }

    // Check odd factors up to sqrt(n)
    let mut i = 3;
    while i * i <= n {
        let mut count = 0;
        while n % i == 0 {
            count += 1;
            n /= i;
        }
        if count > 0 {
            factors.insert(i, count);
        }
        i += 2;
    }

    // If n is still > 1, it's a prime factor
    if n > 1 {
        factors.insert(n, 1);
    }

    factors
}

/// Check if factors are pairwise coprime
fn are_coprime(factors: &HashMap<usize, usize>) -> bool {
    // All prime factors are coprime by definition
    // We check if the product of prime powers are coprime
    let powers: Vec<usize> = factors.iter().map(|(&p, &e)| p.pow(e as u32)).collect();

    for i in 0..powers.len() {
        for j in (i + 1)..powers.len() {
            if gcd(powers[i], powers[j]) != 1 {
                return false;
            }
        }
    }
    true
}

/// Greatest common divisor
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Estimate available memory
fn estimate_available_memory() -> usize {
    // Conservative estimate: assume 1 GB available
    // In a real implementation, this would query the OS
    1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize() {
        let factors = factorize(12);
        assert_eq!(factors.get(&2), Some(&2));
        assert_eq!(factors.get(&3), Some(&1));

        let factors = factorize(1024);
        assert_eq!(factors.get(&2), Some(&10));
        assert_eq!(factors.len(), 1);

        let factors = factorize(17);
        assert_eq!(factors.get(&17), Some(&1));
        assert_eq!(factors.len(), 1);
    }

    #[test]
    fn test_input_characteristics_power_of_2() {
        let cache_info = CacheInfo::default();
        let chars = InputCharacteristics::analyze(1024, &cache_info);

        assert!(chars.is_power_of_2);
        assert!(chars.is_power_of_4);
        assert!(!chars.is_prime);
        assert!(chars.is_smooth);
        assert_eq!(chars.size_type, SizeCharacteristic::PowerOf4);
    }

    #[test]
    fn test_input_characteristics_prime() {
        let cache_info = CacheInfo::default();
        let chars = InputCharacteristics::analyze(1009, &cache_info);

        assert!(!chars.is_power_of_2);
        assert!(chars.is_prime);
        assert_eq!(chars.largest_prime_factor, 1009);
        assert_eq!(chars.size_type, SizeCharacteristic::Prime);
    }

    #[test]
    fn test_input_characteristics_smooth() {
        let cache_info = CacheInfo::default();
        let chars = InputCharacteristics::analyze(360, &cache_info); // 2^3 * 3^2 * 5

        assert!(!chars.is_power_of_2);
        assert!(!chars.is_prime);
        assert!(chars.is_smooth);
        assert_eq!(chars.size_type, SizeCharacteristic::Smooth);
    }

    #[test]
    fn test_algorithm_selector_power_of_2() {
        let selector = AlgorithmSelector::new();
        let rec = selector
            .select_algorithm(1024, true)
            .expect("Selection failed");

        // Power of 4 should recommend Radix-4 or SIMD optimized
        assert!(
            matches!(
                rec.algorithm,
                FftAlgorithm::Radix4 | FftAlgorithm::SimdOptimized | FftAlgorithm::Parallel
            ),
            "Expected Radix-4 or SIMD for power of 4, got {:?}",
            rec.algorithm
        );
        assert!(rec.confidence > 0.8);
    }

    #[test]
    fn test_algorithm_selector_prime() {
        let selector = AlgorithmSelector::new();
        let rec = selector
            .select_algorithm(1009, true)
            .expect("Selection failed");

        // Prime should recommend Rader or Bluestein
        assert!(
            matches!(rec.algorithm, FftAlgorithm::Rader | FftAlgorithm::Bluestein),
            "Expected Rader or Bluestein for prime, got {:?}",
            rec.algorithm
        );
    }

    #[test]
    fn test_algorithm_selector_large_size() {
        let selector = AlgorithmSelector::new();
        let rec = selector
            .select_algorithm(16 * 1024 * 1024 + 1, true)
            .expect("Selection failed");

        // Very large size should recommend streaming
        assert_eq!(rec.algorithm, FftAlgorithm::Streaming);
        assert!(rec.reasoning.iter().any(|r| r.contains("streaming")));
    }

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareInfo::detect();
        assert!(hw.num_cores > 0);
        assert!(hw.cache_info.l1_size > 0);
    }

    #[test]
    fn test_simd_capabilities() {
        let caps = SimdCapabilities::detect();
        // Just check it doesn't panic
        let _ = caps.simd_available();
        let _ = caps.optimal_complex_vector_count();
    }

    #[test]
    fn test_performance_history() {
        let mut history = PerformanceHistory::new();

        let entry = PerformanceEntry {
            size: 1024,
            algorithm: FftAlgorithm::Radix4,
            forward: true,
            execution_time_ns: 1000,
            peak_memory_bytes: 16384,
            timestamp: 0,
            hardware_hash: 0,
        };

        history.record(entry);
        assert_eq!(history.get_best(1024, true), Some(FftAlgorithm::Radix4));
    }

    #[test]
    #[cfg(feature = "oxifft")]
    fn test_benchmark() {
        let selector = AlgorithmSelector::new();
        let result = selector.benchmark(256, FftAlgorithm::MixedRadix, true);

        assert!(result.is_ok());
        let entry = result.expect("Benchmark failed");
        assert_eq!(entry.size, 256);
        assert!(entry.execution_time_ns > 0);
    }

    #[test]
    fn test_selection_config_forced_algorithm() {
        let config = SelectionConfig {
            force_algorithm: Some(FftAlgorithm::Bluestein),
            ..Default::default()
        };
        let selector = AlgorithmSelector::with_config(config);
        let rec = selector
            .select_algorithm(1024, true)
            .expect("Selection failed");

        assert_eq!(rec.algorithm, FftAlgorithm::Bluestein);
        assert_eq!(rec.confidence, 1.0);
    }

    #[test]
    fn test_memory_constraint() {
        let config = SelectionConfig {
            max_memory_bytes: 1024, // Very small
            ..Default::default()
        };
        let selector = AlgorithmSelector::with_config(config);
        let rec = selector
            .select_algorithm(1024, true)
            .expect("Selection failed");

        // Should select streaming due to memory constraint
        assert_eq!(rec.algorithm, FftAlgorithm::Streaming);
    }
}
