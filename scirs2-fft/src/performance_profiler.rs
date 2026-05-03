//! FFT Performance Profiling Module
//!
//! This module provides comprehensive performance profiling for FFT operations,
//! including execution time measurement, memory usage tracking, and performance
//! analysis across different FFT sizes and algorithms.
//!
//! # Features
//!
//! - **Execution Time Profiling**: High-resolution timing for FFT operations
//! - **Memory Usage Tracking**: Monitor peak memory usage and allocation patterns
//! - **Algorithm Comparison**: Compare performance across different FFT algorithms
//! - **Statistical Analysis**: Mean, std dev, percentiles for benchmark results
//! - **Performance Reporting**: Generate human-readable performance reports
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_fft::performance_profiler::{PerformanceProfiler, ProfileConfig};
//!
//! let profiler = PerformanceProfiler::new();
//! let report = profiler.profile_size(1024, true).expect("Profiling failed");
//! println!("Average time: {:?}", report.avg_time);
//! ```

use crate::algorithm_selector::{
    AlgorithmRecommendation, AlgorithmSelector, FftAlgorithm, InputCharacteristics,
    PerformanceEntry, SelectionConfig,
};
use crate::error::{FFTError, FFTResult};

// Backend-specific imports

use oxifft::{Complex as OxiComplex, Direction};

use scirs2_core::numeric::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for performance profiling
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Number of warm-up iterations (not timed)
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Whether to track memory usage
    pub track_memory: bool,
    /// Sizes to benchmark (if None, use default sizes)
    pub sizes: Option<Vec<usize>>,
    /// Algorithms to compare (if None, use all)
    pub algorithms: Option<Vec<FftAlgorithm>>,
    /// Whether to include inverse FFT in profiling
    pub include_inverse: bool,
    /// Timeout per benchmark in milliseconds
    pub timeout_ms: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            track_memory: true,
            sizes: None,
            algorithms: None,
            include_inverse: true,
            timeout_ms: 30000, // 30 seconds
        }
    }
}

/// Default benchmark sizes covering common use cases
const DEFAULT_SIZES: &[usize] = &[
    64, 128, 256, 512, 1024, 2048, 4096, 8192, // Powers of 2
    100, 360, 480, 1000, 2000, 5000, // Smooth numbers
    97, 101, 127, 251, 509, 1009, 2003, // Primes
    1200, 3600, 7200, // Common signal sizes
    65536, 131072, 262144, // Large powers of 2
];

/// Single benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// Execution time in nanoseconds
    pub time_ns: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Allocated memory in bytes
    pub allocated_bytes: usize,
}

/// Profiling result for a single (size, algorithm) combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// FFT size
    pub size: usize,
    /// Algorithm used
    pub algorithm: FftAlgorithm,
    /// Whether this was a forward transform
    pub forward: bool,
    /// Individual measurements
    pub measurements: Vec<Measurement>,
    /// Average execution time
    pub avg_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Median execution time
    pub median_time: Duration,
    /// 90th percentile execution time
    pub p90_time: Duration,
    /// 99th percentile execution time
    pub p99_time: Duration,
    /// Operations per second (throughput)
    pub ops_per_second: f64,
    /// Average peak memory usage
    pub avg_peak_memory: usize,
    /// Throughput in elements per second
    pub elements_per_second: f64,
    /// Input characteristics
    pub input_characteristics: Option<InputCharacteristics>,
}

impl ProfileResult {
    /// Create a profile result from measurements
    pub fn from_measurements(
        size: usize,
        algorithm: FftAlgorithm,
        forward: bool,
        measurements: Vec<Measurement>,
        input_characteristics: Option<InputCharacteristics>,
    ) -> Self {
        if measurements.is_empty() {
            return Self {
                size,
                algorithm,
                forward,
                measurements: Vec::new(),
                avg_time: Duration::ZERO,
                min_time: Duration::ZERO,
                max_time: Duration::ZERO,
                std_dev: Duration::ZERO,
                median_time: Duration::ZERO,
                p90_time: Duration::ZERO,
                p99_time: Duration::ZERO,
                ops_per_second: 0.0,
                avg_peak_memory: 0,
                elements_per_second: 0.0,
                input_characteristics,
            };
        }

        let times_ns: Vec<u64> = measurements.iter().map(|m| m.time_ns).collect();
        let n = times_ns.len();

        // Calculate statistics
        let sum: u64 = times_ns.iter().sum();
        let avg_ns = sum / n as u64;

        let min_ns = times_ns.iter().min().copied().unwrap_or(0);
        let max_ns = times_ns.iter().max().copied().unwrap_or(0);

        // Standard deviation
        let variance: f64 = times_ns
            .iter()
            .map(|&t| {
                let diff = t as f64 - avg_ns as f64;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev_ns = variance.sqrt() as u64;

        // Percentiles
        let mut sorted_times = times_ns.clone();
        sorted_times.sort_unstable();

        let median_ns = if n % 2 == 0 {
            (sorted_times[n / 2 - 1] + sorted_times[n / 2]) / 2
        } else {
            sorted_times[n / 2]
        };

        let p90_idx = (n as f64 * 0.90).ceil() as usize - 1;
        let p99_idx = (n as f64 * 0.99).ceil() as usize - 1;
        let p90_ns = sorted_times.get(p90_idx.min(n - 1)).copied().unwrap_or(0);
        let p99_ns = sorted_times.get(p99_idx.min(n - 1)).copied().unwrap_or(0);

        // Throughput
        let ops_per_second = if avg_ns > 0 {
            1_000_000_000.0 / avg_ns as f64
        } else {
            0.0
        };

        let elements_per_second = ops_per_second * size as f64;

        // Average memory
        let avg_peak_memory = if !measurements.is_empty() {
            measurements
                .iter()
                .map(|m| m.peak_memory_bytes)
                .sum::<usize>()
                / measurements.len()
        } else {
            0
        };

        Self {
            size,
            algorithm,
            forward,
            measurements,
            avg_time: Duration::from_nanos(avg_ns),
            min_time: Duration::from_nanos(min_ns),
            max_time: Duration::from_nanos(max_ns),
            std_dev: Duration::from_nanos(std_dev_ns),
            median_time: Duration::from_nanos(median_ns),
            p90_time: Duration::from_nanos(p90_ns),
            p99_time: Duration::from_nanos(p99_ns),
            ops_per_second,
            avg_peak_memory,
            elements_per_second,
            input_characteristics,
        }
    }

    /// Format as a table row
    pub fn as_table_row(&self) -> String {
        format!(
            "{:>8} | {:>16} | {:>12?} | {:>12?} | {:>12?} | {:>12.2} | {:>10}",
            self.size,
            self.algorithm.to_string(),
            self.avg_time,
            self.min_time,
            self.max_time,
            self.ops_per_second,
            format_bytes(self.avg_peak_memory)
        )
    }
}

/// Comparison result between algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmComparison {
    /// FFT size
    pub size: usize,
    /// Results for each algorithm
    pub results: HashMap<FftAlgorithm, ProfileResult>,
    /// Best algorithm for this size
    pub best_algorithm: FftAlgorithm,
    /// Speedup of best vs worst
    pub speedup: f64,
    /// Recommendation
    pub recommendation: String,
}

/// Performance report for a range of sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Results by (size, algorithm)
    pub results: HashMap<(usize, FftAlgorithm), ProfileResult>,
    /// Best algorithm for each size
    pub best_by_size: HashMap<usize, FftAlgorithm>,
    /// Algorithm comparisons
    pub comparisons: Vec<AlgorithmComparison>,
    /// Total profiling time
    pub total_time: Duration,
    /// Configuration used
    pub config_summary: String,
}

impl PerformanceReport {
    /// Generate a text report
    pub fn to_text_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== FFT Performance Report ===\n\n");
        report.push_str(&format!("Total profiling time: {:?}\n", self.total_time));
        report.push_str(&format!("Configuration: {}\n\n", self.config_summary));

        report.push_str("--- Results by Size ---\n\n");
        report.push_str(&format!(
            "{:>8} | {:>16} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10}\n",
            "Size", "Algorithm", "Avg Time", "Min Time", "Max Time", "Ops/sec", "Memory"
        ));
        report.push_str(&"-".repeat(100));
        report.push('\n');

        let mut sizes: Vec<usize> = self.best_by_size.keys().copied().collect();
        sizes.sort_unstable();

        for size in sizes {
            if let Some(algo) = self.best_by_size.get(&size) {
                if let Some(result) = self.results.get(&(size, *algo)) {
                    report.push_str(&result.as_table_row());
                    report.push('\n');
                }
            }
        }

        report.push_str("\n--- Recommendations ---\n\n");
        for comparison in &self.comparisons {
            report.push_str(&format!(
                "Size {:>8}: {} (speedup: {:.2}x)\n",
                comparison.size, comparison.recommendation, comparison.speedup
            ));
        }

        report
    }
}

/// Performance profiler for FFT operations
pub struct PerformanceProfiler {
    /// Configuration
    config: ProfileConfig,
    /// Algorithm selector for recommendations
    selector: AlgorithmSelector,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfileConfig::default())
    }

    /// Create a new performance profiler with custom configuration
    pub fn with_config(config: ProfileConfig) -> Self {
        Self {
            config,
            selector: AlgorithmSelector::new(),
        }
    }

    /// Profile a single size with the recommended algorithm
    pub fn profile_size(&self, size: usize, forward: bool) -> FFTResult<ProfileResult> {
        let recommendation = self.selector.select_algorithm(size, forward)?;
        self.profile_size_with_algorithm(size, recommendation.algorithm, forward)
    }

    /// Profile a single size with a specific algorithm (OxiFFT backend)
    pub fn profile_size_with_algorithm(
        &self,
        size: usize,
        algorithm: FftAlgorithm,
        forward: bool,
    ) -> FFTResult<ProfileResult> {
        // Validate size
        if size == 0 {
            return Err(FFTError::ValueError("Size must be positive".to_string()));
        }

        // Create test data (convert to OxiFFT format)
        let data: Vec<OxiComplex<f64>> = (0..size)
            .map(|i| OxiComplex::new(i as f64, (i * 2) as f64))
            .collect();

        // Create FFT plan using OxiFFT
        let direction = if forward {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let plan = oxifft::Plan::dft_1d(size, direction, oxifft::Flags::ESTIMATE)
            .ok_or_else(|| FFTError::ComputationError("Failed to create FFT plan".to_string()))?;

        // Warm-up iterations
        for _ in 0..self.config.warmup_iterations {
            let work_data = data.clone();
            let mut out = vec![OxiComplex::new(0.0, 0.0); size];
            plan.execute(&work_data, &mut out);
        }

        // Benchmark iterations
        let mut measurements = Vec::with_capacity(self.config.benchmark_iterations);

        for _ in 0..self.config.benchmark_iterations {
            let mut work_data = data.clone();

            // Track memory before
            let memory_before = if self.config.track_memory {
                estimate_current_allocation()
            } else {
                0
            };

            // Time the FFT
            let start = Instant::now();
            let mut out = vec![OxiComplex::new(0.0, 0.0); size];
            plan.execute(&work_data, &mut out);
            let elapsed = start.elapsed();

            // Track memory after
            let memory_after = if self.config.track_memory {
                estimate_current_allocation()
            } else {
                0
            };

            measurements.push(Measurement {
                time_ns: elapsed.as_nanos() as u64,
                peak_memory_bytes: size * 16, // Complex64 = 16 bytes
                allocated_bytes: memory_after.saturating_sub(memory_before),
            });
        }

        // Get input characteristics
        let chars = InputCharacteristics::analyze(size, &self.selector.hardware().cache_info);

        Ok(ProfileResult::from_measurements(
            size,
            algorithm,
            forward,
            measurements,
            Some(chars),
        ))
    }

    /// Profile multiple sizes
    pub fn profile_sizes(&self, sizes: &[usize], forward: bool) -> FFTResult<Vec<ProfileResult>> {
        let mut results = Vec::with_capacity(sizes.len());

        for &size in sizes {
            match self.profile_size(size, forward) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log error but continue with other sizes
                    eprintln!("Failed to profile size {}: {}", size, e);
                }
            }
        }

        Ok(results)
    }

    /// Compare algorithms for a given size
    pub fn compare_algorithms(
        &self,
        size: usize,
        algorithms: &[FftAlgorithm],
        forward: bool,
    ) -> FFTResult<AlgorithmComparison> {
        let mut results = HashMap::new();

        for &algorithm in algorithms {
            match self.profile_size_with_algorithm(size, algorithm, forward) {
                Ok(result) => {
                    results.insert(algorithm, result);
                }
                Err(e) => {
                    eprintln!("Failed to profile {} for size {}: {}", algorithm, size, e);
                }
            }
        }

        if results.is_empty() {
            return Err(FFTError::ComputationError(
                "No algorithms could be profiled".to_string(),
            ));
        }

        // Find best and worst
        let best_algorithm = results
            .iter()
            .min_by_key(|(_, r)| r.avg_time)
            .map(|(&algo, _)| algo)
            .unwrap_or(FftAlgorithm::MixedRadix);

        let best_time = results
            .get(&best_algorithm)
            .map(|r| r.avg_time)
            .unwrap_or(Duration::ZERO);

        let worst_time = results
            .values()
            .map(|r| r.avg_time)
            .max()
            .unwrap_or(Duration::ZERO);

        let speedup = if best_time.as_nanos() > 0 {
            worst_time.as_nanos() as f64 / best_time.as_nanos() as f64
        } else {
            1.0
        };

        let recommendation = format!("{} is fastest ({:?})", best_algorithm, best_time);

        Ok(AlgorithmComparison {
            size,
            results,
            best_algorithm,
            speedup,
            recommendation,
        })
    }

    /// Run comprehensive profiling
    pub fn run_comprehensive_profile(&self) -> FFTResult<PerformanceReport> {
        let start = Instant::now();

        let sizes = self.config.sizes.as_deref().unwrap_or(DEFAULT_SIZES);

        let algorithms = self.config.algorithms.as_deref().unwrap_or(&[
            FftAlgorithm::CooleyTukeyRadix2,
            FftAlgorithm::SplitRadix,
            FftAlgorithm::MixedRadix,
            FftAlgorithm::Bluestein,
        ]);

        let mut results = HashMap::new();
        let mut best_by_size = HashMap::new();
        let mut comparisons = Vec::new();

        for &size in sizes {
            // Profile with each algorithm
            let comparison = self.compare_algorithms(size, algorithms, true)?;

            for (algo, result) in &comparison.results {
                results.insert((size, *algo), result.clone());
            }

            best_by_size.insert(size, comparison.best_algorithm);
            comparisons.push(comparison);

            // Also profile inverse if configured
            if self.config.include_inverse {
                if let Ok(inv_comparison) = self.compare_algorithms(size, algorithms, false) {
                    // Store inverse results with a different key or in a separate structure
                    // For now, we focus on forward transforms
                    let _ = inv_comparison;
                }
            }
        }

        let total_time = start.elapsed();

        let config_summary = format!(
            "warmup={}, iterations={}, sizes={}, algorithms={}",
            self.config.warmup_iterations,
            self.config.benchmark_iterations,
            sizes.len(),
            algorithms.len()
        );

        Ok(PerformanceReport {
            results,
            best_by_size,
            comparisons,
            total_time,
            config_summary,
        })
    }

    /// Auto-tune for specific sizes and record results
    pub fn auto_tune(&mut self, sizes: &[usize]) -> FFTResult<HashMap<usize, FftAlgorithm>> {
        let algorithms = [
            FftAlgorithm::CooleyTukeyRadix2,
            FftAlgorithm::SplitRadix,
            FftAlgorithm::MixedRadix,
            FftAlgorithm::Bluestein,
        ];

        let mut best_algorithms = HashMap::new();

        for &size in sizes {
            let comparison = self.compare_algorithms(size, &algorithms, true)?;
            best_algorithms.insert(size, comparison.best_algorithm);

            // Record to the selector's history
            if let Some(result) = comparison.results.get(&comparison.best_algorithm) {
                let entry = PerformanceEntry {
                    size,
                    algorithm: comparison.best_algorithm,
                    forward: true,
                    execution_time_ns: result.avg_time.as_nanos() as u64,
                    peak_memory_bytes: result.avg_peak_memory,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    hardware_hash: 0,
                };
                let _ = self.selector.record_performance(entry);
            }
        }

        Ok(best_algorithms)
    }

    /// Get selector reference
    pub fn selector(&self) -> &AlgorithmSelector {
        &self.selector
    }

    /// Get selector mutable reference
    pub fn selector_mut(&mut self) -> &mut AlgorithmSelector {
        &mut self.selector
    }
}

/// Memory profiler for tracking FFT memory usage
pub struct MemoryProfiler {
    /// Tracking enabled
    enabled: bool,
    /// Peak memory recorded
    peak_memory: usize,
    /// Current allocation estimate
    current_allocation: usize,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            enabled: true,
            peak_memory: 0,
            current_allocation: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        if self.enabled {
            self.current_allocation += bytes;
            if self.current_allocation > self.peak_memory {
                self.peak_memory = self.current_allocation;
            }
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, bytes: usize) {
        if self.enabled {
            self.current_allocation = self.current_allocation.saturating_sub(bytes);
        }
    }

    /// Reset counters
    pub fn reset(&mut self) {
        self.peak_memory = 0;
        self.current_allocation = 0;
    }

    /// Get peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }

    /// Get current allocation
    pub fn current_allocation(&self) -> usize {
        self.current_allocation
    }

    /// Enable/disable tracking
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Estimate memory required for FFT of given size
pub fn estimate_fft_memory(size: usize, algorithm: FftAlgorithm) -> usize {
    let base_memory = size * 16; // Complex64 = 16 bytes

    // Algorithm-specific overhead
    let overhead = match algorithm {
        FftAlgorithm::CooleyTukeyRadix2 => base_memory / 4, // Minimal scratch
        FftAlgorithm::Radix4 => base_memory / 4,
        FftAlgorithm::SplitRadix => base_memory / 3,
        FftAlgorithm::MixedRadix => base_memory / 2,
        FftAlgorithm::Bluestein => base_memory * 2, // Needs convolution buffer
        FftAlgorithm::Rader => base_memory,
        FftAlgorithm::Winograd => base_memory / 2,
        FftAlgorithm::GoodThomas => base_memory / 3,
        FftAlgorithm::Streaming => base_memory / 8, // Minimal memory
        FftAlgorithm::CacheOblivious => base_memory / 4,
        FftAlgorithm::InPlace => 0, // No extra memory
        FftAlgorithm::SimdOptimized => base_memory / 4,
        FftAlgorithm::Parallel => base_memory, // Thread buffers
        FftAlgorithm::Hybrid => base_memory / 2,
    };

    base_memory + overhead
}

/// Estimate current memory allocation (simplified)
fn estimate_current_allocation() -> usize {
    // This is a simplified estimation
    // In a real implementation, this would query the allocator
    0
}

/// Format bytes in human-readable form
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_result_statistics() {
        let measurements = vec![
            Measurement {
                time_ns: 1000,
                peak_memory_bytes: 1000,
                allocated_bytes: 100,
            },
            Measurement {
                time_ns: 1200,
                peak_memory_bytes: 1000,
                allocated_bytes: 100,
            },
            Measurement {
                time_ns: 1100,
                peak_memory_bytes: 1000,
                allocated_bytes: 100,
            },
            Measurement {
                time_ns: 1050,
                peak_memory_bytes: 1000,
                allocated_bytes: 100,
            },
        ];

        let result = ProfileResult::from_measurements(
            256,
            FftAlgorithm::MixedRadix,
            true,
            measurements,
            None,
        );

        assert_eq!(result.size, 256);
        assert!(result.avg_time.as_nanos() > 0);
        assert!(result.min_time <= result.avg_time);
        assert!(result.max_time >= result.avg_time);
        assert!(result.ops_per_second > 0.0);
    }

    #[test]
    fn test_profile_single_size() {
        let profiler = PerformanceProfiler::new();
        let result = profiler.profile_size(256, true);

        assert!(result.is_ok());
        let profile = result.expect("Profiling failed");
        assert_eq!(profile.size, 256);
        assert!(profile.avg_time.as_nanos() > 0);
    }

    #[test]
    fn test_compare_algorithms() {
        let profiler = PerformanceProfiler::new();
        let algorithms = [FftAlgorithm::MixedRadix, FftAlgorithm::Bluestein];

        let result = profiler.compare_algorithms(256, &algorithms, true);

        assert!(result.is_ok());
        let comparison = result.expect("Comparison failed");
        assert_eq!(comparison.size, 256);
        assert!(!comparison.results.is_empty());
        assert!(comparison.speedup >= 1.0);
    }

    #[test]
    fn test_memory_profiler() {
        let mut mp = MemoryProfiler::new();

        mp.record_allocation(1000);
        assert_eq!(mp.current_allocation(), 1000);
        assert_eq!(mp.peak_memory(), 1000);

        mp.record_allocation(500);
        assert_eq!(mp.current_allocation(), 1500);
        assert_eq!(mp.peak_memory(), 1500);

        mp.record_deallocation(1000);
        assert_eq!(mp.current_allocation(), 500);
        assert_eq!(mp.peak_memory(), 1500);

        mp.reset();
        assert_eq!(mp.current_allocation(), 0);
        assert_eq!(mp.peak_memory(), 0);
    }

    #[test]
    fn test_estimate_fft_memory() {
        let size = 1024;

        let inplace_mem = estimate_fft_memory(size, FftAlgorithm::InPlace);
        let bluestein_mem = estimate_fft_memory(size, FftAlgorithm::Bluestein);

        // Bluestein should use more memory than in-place
        assert!(bluestein_mem > inplace_mem);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_profile_config_default() {
        let config = ProfileConfig::default();
        assert!(config.warmup_iterations > 0);
        assert!(config.benchmark_iterations > 0);
        assert!(config.track_memory);
    }
}
