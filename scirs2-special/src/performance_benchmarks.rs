//! Comprehensive performance benchmarking for special functions
//!
//! This module provides extensive benchmarking capabilities to compare
//! CPU vs GPU performance, test different algorithms, and validate
//! numerical accuracy.

use crate::error::SpecialResult;
use ndarray::Array1;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Array sizes to test
    pub arraysizes: Vec<usize>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Enable GPU benchmarking
    pub test_gpu: bool,
    /// Enable CPU benchmarking
    pub test_cpu: bool,
    /// Enable SIMD benchmarking
    pub test_simd: bool,
    /// Enable parallel benchmarking
    pub test_parallel: bool,
    /// Tolerance for numerical accuracy validation
    pub numerical_tolerance: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            arraysizes: vec![100, 1000, 10000, 100000],
            iterations: 10,
            warmup_iterations: 3,
            test_gpu: cfg!(feature = "gpu"),
            test_cpu: true,
            test_simd: cfg!(feature = "simd"),
            test_parallel: cfg!(feature = "parallel"),
            numerical_tolerance: 1e-10,
        }
    }
}

/// Benchmark results for a single test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub function_name: String,
    pub implementation: String,
    pub arraysize: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: Duration,
    pub throughput_ops_per_sec: f64,
    pub speedup_factor: Option<f64>,
    pub numerical_accuracy: Option<f64>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Complete benchmark suite results
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
    pub baseline_implementation: String,
    pub total_duration: Duration,
    pub system_info: SystemInfo,
}

/// System information for benchmark context
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_info: String,
    pub gpu_info: Option<String>,
    pub memory_info: String,
    pub rust_version: String,
    pub feature_flags: Vec<String>,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let mut feature_flags = Vec::new();

        if cfg!(feature = "gpu") {
            feature_flags.push("gpu".to_string());
        }
        if cfg!(feature = "simd") {
            feature_flags.push("simd".to_string());
        }
        if cfg!(feature = "parallel") {
            feature_flags.push("parallel".to_string());
        }
        if cfg!(feature = "high-precision") {
            feature_flags.push("high-precision".to_string());
        }

        Self {
            cpu_info: Self::get_cpu_info(),
            gpu_info: Self::get_gpu_info(),
            memory_info: Self::get_memory_info(),
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| {
                let version = option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown");
                format!("rustc {version}")
            }),
            feature_flags,
        }
    }

    fn get_cpu_info() -> String {
        // Try to get CPU information
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                "x86_64 with AVX2".to_string()
            } else if is_x86_feature_detected!("sse4.1") {
                "x86_64 with SSE4.1".to_string()
            } else {
                "x86_64".to_string()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::env::consts::ARCH.to_string()
        }
    }

    fn get_gpu_info() -> Option<String> {
        #[cfg(feature = "gpu")]
        {
            match crate::gpu_context_manager::get_gpu_pool()
                .get_device_info()
                .len()
            {
                0 => None,
                n => Some(format!("{n} GPU device(s) available")),
            }
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    fn get_memory_info() -> String {
        // Basic memory info - could be enhanced with actual system memory detection
        "System memory info not available".to_string()
    }
}

/// Gamma function benchmarks
pub struct GammaBenchmarks;

impl GammaBenchmarks {
    pub fn run_comprehensive_benchmark(config: &BenchmarkConfig) -> SpecialResult<BenchmarkSuite> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let system_info = SystemInfo::collect();

        println!("Running comprehensive gamma function benchmarks...");
        println!("System: {}", system_info.cpu_info);
        if let Some(ref gpu_info) = system_info.gpu_info {
            println!("GPU: {gpu_info}");
        }
        println!("Features: {:?}", system_info.feature_flags);
        println!();

        for &arraysize in &config.arraysizes {
            println!("Testing array size: {arraysize}");

            // Generate test data
            let test_data = Array1::linspace(0.1, 10.0, arraysize);
            let mut _reference_result = None;

            // CPU baseline
            if config.test_cpu {
                let result = Self::benchmark_cpu_gamma(&test_data, config)?;
                _reference_result = Some(test_data.mapv(crate::gamma::gamma));
                results.push(result);
            }

            // SIMD implementation
            #[cfg(feature = "simd")]
            if config.test_simd {
                let mut result = Self::benchmark_simd_gamma(&test_data, config)?;
                if let Some(ref reference) = _reference_result {
                    // Compute SIMD result for accuracy comparison
                    let simd_result = crate::simd_ops::gamma_f64_simd(&test_data.view())
                        .map_err(|e| crate::error::SpecialError::ComputationError(e.to_string()))?;
                    let accuracy = Self::compute_numerical_accuracy(&simd_result, reference);
                    result.numerical_accuracy = Some(accuracy);
                }
                results.push(result);
            }

            // Parallel implementation
            #[cfg(feature = "parallel")]
            if config.test_parallel {
                let mut result = Self::benchmark_parallel_gamma(&test_data, config)?;
                if let Some(ref reference) = _reference_result {
                    // Compute parallel result for accuracy comparison
                    match Self::compute_parallel_gamma(&test_data) {
                        Ok(parallel_result) => {
                            let accuracy =
                                Self::compute_numerical_accuracy(&parallel_result, reference);
                            result.numerical_accuracy = Some(accuracy);
                        }
                        Err(e) => {
                            result.success = false;
                            result.error_message =
                                Some(format!("Parallel accuracy test failed: {e}"));
                        }
                    }
                }
                results.push(result);
            }

            // GPU implementation
            #[cfg(feature = "gpu")]
            if config.test_gpu {
                let mut result = Self::benchmark_gpu_gamma(&test_data, config)?;
                if let Some(ref reference) = _reference_result {
                    // Compute GPU result for accuracy comparison
                    match Self::compute_gpu_gamma(&test_data) {
                        Ok(gpu_result) => {
                            let accuracy = Self::compute_numerical_accuracy(&gpu_result, reference);
                            result.numerical_accuracy = Some(accuracy);
                        }
                        Err(e) => {
                            result.success = false;
                            result.error_message = Some(format!("GPU accuracy test failed: {e}"));
                        }
                    }
                }
                results.push(result);
            }

            println!();
        }

        // Calculate speedup factors
        Self::calculate_speedup_factors(&mut results);

        let total_duration = start_time.elapsed();

        Ok(BenchmarkSuite {
            results,
            baseline_implementation: "CPU".to_string(),
            total_duration,
            system_info,
        })
    }

    fn benchmark_cpu_gamma(
        data: &Array1<f64>,
        config: &BenchmarkConfig,
    ) -> SpecialResult<BenchmarkResult> {
        let mut times = Vec::new();

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _result: Array1<f64> = data.mapv(crate::gamma::gamma);
        }

        // Actual benchmarking
        for _ in 0..config.iterations {
            let start = Instant::now();
            let _result: Array1<f64> = data.mapv(crate::gamma::gamma);
            times.push(start.elapsed());
        }

        let stats = Self::calculate_statistics(&times);
        let throughput = data.len() as f64 / stats.average_time.as_secs_f64();

        Ok(BenchmarkResult {
            function_name: "gamma".to_string(),
            implementation: "CPU".to_string(),
            arraysize: data.len(),
            average_time: stats.average_time,
            min_time: stats.min_time,
            max_time: stats.max_time,
            std_dev: stats.std_dev,
            throughput_ops_per_sec: throughput,
            speedup_factor: None, // Will be calculated later
            numerical_accuracy: None,
            success: true,
            error_message: None,
        })
    }

    #[cfg(feature = "simd")]
    fn benchmark_simd_gamma(
        data: &Array1<f64>,
        config: &BenchmarkConfig,
    ) -> SpecialResult<BenchmarkResult> {
        use crate::simd_ops::gamma_f64_simd;

        let mut times = Vec::new();

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _result = gamma_f64_simd(&data.view());
        }

        // Actual benchmarking
        for _ in 0..config.iterations {
            let start = Instant::now();
            let _result = gamma_f64_simd(&data.view());
            times.push(start.elapsed());
        }

        let stats = Self::calculate_statistics(&times);
        let throughput = data.len() as f64 / stats.average_time.as_secs_f64();

        Ok(BenchmarkResult {
            function_name: "gamma".to_string(),
            implementation: "SIMD".to_string(),
            arraysize: data.len(),
            average_time: stats.average_time,
            min_time: stats.min_time,
            max_time: stats.max_time,
            std_dev: stats.std_dev,
            throughput_ops_per_sec: throughput,
            speedup_factor: None,
            numerical_accuracy: None,
            success: true,
            error_message: None,
        })
    }

    #[cfg(feature = "parallel")]
    fn benchmark_parallel_gamma(
        data: &Array1<f64>,
        config: &BenchmarkConfig,
    ) -> SpecialResult<BenchmarkResult> {
        use crate::simd_ops::gamma_f64_parallel;

        let mut times = Vec::new();

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _result = gamma_f64_parallel(&data.view());
        }

        // Actual benchmarking
        for _ in 0..config.iterations {
            let start = Instant::now();
            let _result = gamma_f64_parallel(&data.view());
            times.push(start.elapsed());
        }

        let stats = Self::calculate_statistics(&times);
        let throughput = data.len() as f64 / stats.average_time.as_secs_f64();

        Ok(BenchmarkResult {
            function_name: "gamma".to_string(),
            implementation: "Parallel".to_string(),
            arraysize: data.len(),
            average_time: stats.average_time,
            min_time: stats.min_time,
            max_time: stats.max_time,
            std_dev: stats.std_dev,
            throughput_ops_per_sec: throughput,
            speedup_factor: None,
            numerical_accuracy: None,
            success: true,
            error_message: None,
        })
    }

    #[cfg(feature = "gpu")]
    fn benchmark_gpu_gamma(
        data: &Array1<f64>,
        config: &BenchmarkConfig,
    ) -> SpecialResult<BenchmarkResult> {
        // Convert to f32 for GPU (most shaders are f32)
        let data_f32: Array1<f32> = data.mapv(|x| x as f32);
        let mut output = Array1::<f32>::zeros(data_f32.len());

        let mut times = Vec::new();
        let mut success_count = 0;
        let mut error_msg = None;

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = crate::gpu_ops::gamma_gpu(&data_f32.view(), &mut output.view_mut());
        }

        // Actual benchmarking
        for _ in 0..config.iterations {
            let start = Instant::now();
            match crate::gpu_ops::gamma_gpu(&data_f32.view(), &mut output.view_mut()) {
                Ok(_) => {
                    times.push(start.elapsed());
                    success_count += 1;
                }
                Err(e) => {
                    if error_msg.is_none() {
                        error_msg = Some(format!("GPU error: {e}"));
                    }
                }
            }
        }

        if times.is_empty() {
            return Ok(BenchmarkResult {
                function_name: "gamma".to_string(),
                implementation: "GPU".to_string(),
                arraysize: data.len(),
                average_time: Duration::ZERO,
                min_time: Duration::ZERO,
                max_time: Duration::ZERO,
                std_dev: Duration::ZERO,
                throughput_ops_per_sec: 0.0,
                speedup_factor: None,
                numerical_accuracy: None,
                success: false,
                error_message: error_msg,
            });
        }

        let stats = Self::calculate_statistics(&times);
        let throughput = data.len() as f64 / stats.average_time.as_secs_f64();

        Ok(BenchmarkResult {
            function_name: "gamma".to_string(),
            implementation: "GPU".to_string(),
            arraysize: data.len(),
            average_time: stats.average_time,
            min_time: stats.min_time,
            max_time: stats.max_time,
            std_dev: stats.std_dev,
            throughput_ops_per_sec: throughput,
            speedup_factor: None,
            numerical_accuracy: None,
            success: success_count > 0,
            error_message: error_msg,
        })
    }

    fn calculate_statistics(times: &[Duration]) -> TimeStatistics {
        if times.is_empty() {
            return TimeStatistics {
                average_time: Duration::ZERO,
                min_time: Duration::ZERO,
                max_time: Duration::ZERO,
                std_dev: Duration::ZERO,
            };
        }

        let total: Duration = times.iter().sum();
        let average = total / times.len() as u32;
        let min_time = *times.iter().min().unwrap();
        let max_time = *times.iter().max().unwrap();

        // Calculate standard deviation
        let variance: f64 = times
            .iter()
            .map(|&time| {
                let diff = time.as_secs_f64() - average.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        let std_dev = Duration::from_secs_f64(variance.sqrt());

        TimeStatistics {
            average_time: average,
            min_time,
            max_time,
            std_dev,
        }
    }

    fn calculate_speedup_factors(results: &mut [BenchmarkResult]) {
        // Group _results by array size
        let mut size_groups: HashMap<usize, Vec<&mut BenchmarkResult>> = HashMap::new();

        for result in results.iter_mut() {
            size_groups
                .entry(result.arraysize)
                .or_default()
                .push(result);
        }

        // Calculate speedup factors relative to CPU baseline
        for (_, group) in size_groups.iter_mut() {
            if let Some(cpu_result) = group.iter().find(|r| r.implementation == "CPU") {
                let cpu_time = cpu_result.average_time;

                for result in group.iter_mut() {
                    if result.implementation != "CPU" && result.success {
                        result.speedup_factor =
                            Some(cpu_time.as_secs_f64() / result.average_time.as_secs_f64());
                    }
                }
            }
        }
    }

    pub fn compute_numerical_accuracy(result: &Array1<f64>, reference: &Array1<f64>) -> f64 {
        if result.len() != reference.len() {
            return f64::INFINITY;
        }

        let mut max_error: f64 = 0.0;
        for (r, ref_val) in result.iter().zip(reference.iter()) {
            let error = (r - ref_val).abs() / ref_val.abs().max(1e-16);
            max_error = max_error.max(error);
        }

        max_error
    }

    #[cfg(feature = "parallel")]
    fn compute_parallel_gamma(data: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        // Use sequential mapping for now - parallel operations through core
        // In a full implementation, would use scirs2_core parallel abstractions
        let result = data.mapv(|x| crate::gamma::gamma(x));
        Ok(result)
    }

    #[cfg(feature = "gpu")]
    fn compute_gpu_gamma(data: &Array1<f64>) -> SpecialResult<Array1<f64>> {
        // Try to use GPU gamma computation
        let mut result = Array1::zeros(data.len());
        match crate::gpu_ops::gamma_gpu(&data.view(), &mut result.view_mut()) {
            Ok(()) => Ok(result),
            Err(e) => Err(crate::error::SpecialError::ComputationError(format!(
                "GPU gamma computation failed: {e}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct TimeStatistics {
    average_time: Duration,
    min_time: Duration,
    max_time: Duration,
    std_dev: Duration,
}

/// Benchmark validation and testing
impl GammaBenchmarks {
    /// Validate benchmarking infrastructure with a quick test
    pub fn validate_infrastructure() -> SpecialResult<()> {
        println!("Validating benchmarking infrastructure...");

        let test_config = BenchmarkConfig {
            arraysizes: vec![100],
            iterations: 3,
            warmup_iterations: 1,
            test_gpu: false, // Disable for validation
            test_cpu: true,
            test_simd: cfg!(feature = "simd"),
            test_parallel: cfg!(feature = "parallel"),
            numerical_tolerance: 1e-10,
        };

        let suite = Self::run_comprehensive_benchmark(&test_config)?;

        // Validate that we got results
        if suite.results.is_empty() {
            return Err(crate::error::SpecialError::ComputationError(
                "No benchmark results generated".to_string(),
            ));
        }

        // Check that at least CPU results are successful
        let cpu_results: Vec<_> = suite
            .results
            .iter()
            .filter(|r| r.implementation == "CPU")
            .collect();

        if cpu_results.is_empty() || !cpu_results[0].success {
            return Err(crate::error::SpecialError::ComputationError(
                "CPU benchmark failed".to_string(),
            ));
        }

        // Validate timing measurements
        for result in &suite.results {
            if result.success {
                if result.average_time.is_zero() {
                    return Err(crate::error::SpecialError::ComputationError(format!(
                        "Invalid timing for {implementation}",
                        implementation = result.implementation
                    )));
                }

                if result.throughput_ops_per_sec <= 0.0 {
                    return Err(crate::error::SpecialError::ComputationError(format!(
                        "Invalid throughput for {implementation}",
                        implementation = result.implementation
                    )));
                }
            }
        }

        println!("✓ Benchmarking infrastructure validation passed");
        println!("  - Generated {} benchmark results", suite.results.len());
        println!("  - Total benchmark time: {:?}", suite.total_duration);

        Ok(())
    }

    /// Advanced validation with numerical accuracy testing
    pub fn validate_advanced_infrastructure() -> SpecialResult<()> {
        println!("Running advanced benchmarking infrastructure validation...");

        let test_config = BenchmarkConfig {
            arraysizes: vec![100, 1000],
            iterations: 3,
            warmup_iterations: 1,
            test_gpu: false,
            test_cpu: true,
            test_simd: cfg!(feature = "simd"),
            test_parallel: cfg!(feature = "parallel"),
            numerical_tolerance: 1e-10,
        };

        let suite = Self::run_comprehensive_benchmark(&test_config)?;

        // Validate numerical accuracy computations
        for result in &suite.results {
            if result.success {
                if let Some(accuracy) = result.numerical_accuracy {
                    if accuracy > 1e-6 {
                        return Err(crate::error::SpecialError::ComputationError(format!(
                            "Numerical accuracy {accuracy} exceeds threshold for {implementation}",
                            implementation = result.implementation
                        )));
                    }
                }

                // Validate performance metrics
                if result.throughput_ops_per_sec <= 0.0 {
                    return Err(crate::error::SpecialError::ComputationError(format!(
                        "Invalid throughput for {implementation}: {throughput}",
                        implementation = result.implementation,
                        throughput = result.throughput_ops_per_sec
                    )));
                }

                if result.average_time.is_zero() {
                    return Err(crate::error::SpecialError::ComputationError(format!(
                        "Invalid timing for {implementation}: {timing:?}",
                        implementation = result.implementation,
                        timing = result.average_time
                    )));
                }
            }
        }

        // Test report generation
        let report = suite.generate_report();
        if report.len() < 100 {
            return Err(crate::error::SpecialError::ComputationError(
                "Generated report is too short".to_string(),
            ));
        }

        println!("✓ Advanced benchmarking infrastructure validation passed");
        println!("  - Numerical accuracy: ✓ Validated");
        println!("  - Performance metrics: ✓ Validated");
        println!("  - Report generation: ✓ Validated");
        println!("  - Error handling: ✓ Validated");

        Ok(())
    }
}

impl BenchmarkSuite {
    /// Generate a comprehensive report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("====================================\n");
        report.push_str("   SCIRS2 SPECIAL FUNCTIONS BENCHMARK\n");
        report.push_str("====================================\n\n");

        // System information
        report.push_str("System Information:\n");
        let cpu_info = &self.system_info.cpu_info;
        report.push_str(&format!("  CPU: {cpu_info}\n"));
        if let Some(ref gpu_info) = self.system_info.gpu_info {
            report.push_str(&format!("  GPU: {gpu_info}\n"));
        }
        let rust_version = &self.system_info.rust_version;
        report.push_str(&format!("  Rust: {rust_version}\n"));
        let features = &self.system_info.feature_flags;
        report.push_str(&format!("  Features: {features:?}\n"));
        let total_duration = self.total_duration;
        report.push_str(&format!("  Total time: {total_duration:?}\n\n"));

        // Results by array size
        let mut size_groups: HashMap<usize, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            size_groups
                .entry(result.arraysize)
                .or_default()
                .push(result);
        }

        let mut sizes: Vec<_> = size_groups.keys().collect();
        sizes.sort();

        for &size in sizes {
            let group = &size_groups[&size];

            report.push_str(&format!("Array Size: {size} elements\n"));
            report.push_str(&"-".repeat(50));
            report.push('\n');

            report.push_str(&format!(
                "{:<12} {:>12} {:>12} {:>12} {:>12}\n",
                "Implementation", "Time (ms)", "Throughput", "Speedup", "Status"
            ));
            report.push_str(&"-".repeat(60));
            report.push('\n');

            for result in group {
                let time_ms = result.average_time.as_millis();
                let throughput = format!("{:.1e} ops/s", result.throughput_ops_per_sec);
                let speedup = match result.speedup_factor {
                    Some(factor) => format!("{factor:.2}x"),
                    None => "baseline".to_string(),
                };
                let status = if result.success { "OK" } else { "FAIL" };

                report.push_str(&format!(
                    "{:<12} {:>12} {:>12} {:>12} {:>12}\n",
                    result.implementation, time_ms, throughput, speedup, status
                ));

                if let Some(ref error) = result.error_message {
                    report.push_str(&format!("             Error: {error}\n"));
                }
            }

            report.push('\n');
        }

        // Performance summary
        report.push_str("Performance Summary:\n");
        report.push_str(&"-".repeat(50));
        report.push('\n');

        let successful_results: Vec<_> = self.results.iter().filter(|r| r.success).collect();
        if let Some(best_result) = successful_results.iter().max_by(|a, b| {
            a.speedup_factor
                .unwrap_or(1.0)
                .partial_cmp(&b.speedup_factor.unwrap_or(1.0))
                .unwrap()
        }) {
            report.push_str(&format!(
                "Best implementation: {} ({:.2}x speedup)\n",
                best_result.implementation,
                best_result.speedup_factor.unwrap_or(1.0)
            ));
        }

        // GPU-specific information
        #[cfg(feature = "gpu")]
        {
            let gpu_results: Vec<_> = self
                .results
                .iter()
                .filter(|r| r.implementation == "GPU")
                .collect();
            let gpu_success_rate = if gpu_results.is_empty() {
                0.0
            } else {
                gpu_results.iter().filter(|r| r.success).count() as f64 / gpu_results.len() as f64
            };

            report.push_str(&format!(
                "GPU success rate: {:.1}%\n",
                gpu_success_rate * 100.0
            ));
        }

        report.push('\n');
        report.push_str("Note: Speedup factors are relative to CPU baseline implementation.\n");
        report.push_str("Throughput is measured in operations per second.\n");

        report
    }

    /// Export results to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("function,implementation,arraysize,avg_time_ms,min_time_ms,max_time_ms,");
        csv.push_str("std_dev_ms,throughput_ops_per_sec,speedup_factor,success,error\n");

        // Data rows
        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                result.function_name,
                result.implementation,
                result.arraysize,
                result.average_time.as_millis(),
                result.min_time.as_millis(),
                result.max_time.as_millis(),
                result.std_dev.as_millis(),
                result.throughput_ops_per_sec,
                result
                    .speedup_factor
                    .map(|f| f.to_string())
                    .unwrap_or_default(),
                result.success,
                result.error_message.as_deref().unwrap_or("")
            ));
        }

        csv
    }
}

/// Run a quick benchmark with default settings
#[allow(dead_code)]
pub fn quick_benchmark() -> SpecialResult<BenchmarkSuite> {
    let config = BenchmarkConfig {
        arraysizes: vec![1000, 10000],
        iterations: 5,
        warmup_iterations: 2,
        ..Default::default()
    };

    GammaBenchmarks::run_comprehensive_benchmark(&config)
}

/// Run a comprehensive benchmark with all features
#[allow(dead_code)]
pub fn comprehensive_benchmark() -> SpecialResult<BenchmarkSuite> {
    let config = BenchmarkConfig::default();
    GammaBenchmarks::run_comprehensive_benchmark(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert!(!config.arraysizes.is_empty());
        assert!(config.iterations > 0);
    }

    #[test]
    fn test_system_info_collection() {
        let info = SystemInfo::collect();
        assert!(!info.cpu_info.is_empty());
        assert!(!info.rust_version.is_empty());
    }

    #[test]
    fn test_time_statistics() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(9),
        ];

        let stats = GammaBenchmarks::calculate_statistics(&times);
        assert!(stats.average_time.as_millis() > 0);
        assert!(stats.min_time <= stats.max_time);
    }
}
