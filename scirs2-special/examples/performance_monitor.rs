//! Performance Monitor for CI/CD Pipeline
//!
//! This example provides comprehensive performance monitoring for special functions
//! used in the CI/CD pipeline to detect performance regressions.
//!
//! Run with: cargo run --release --example performancemonitor --all-features

use ndarray::Array1;
use scirs2_special::*;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
struct BenchmarkResult {
    function_name: String,
    mean_time_ns: f64,
    std_dev_ns: f64,
    min_time_ns: f64,
    max_time_ns: f64,
    throughput_ops_per_sec: f64,
    memory_usage_bytes: usize,
    samples: usize,
}

struct PerformanceMonitor {
    results: HashMap<String, BenchmarkResult>,
    warmup_iterations: usize,
    measurement_iterations: usize,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
            warmup_iterations: 100,
            measurement_iterations: 1000,
        }
    }

    fn benchmark_function<F>(&mut self, name: &str, inputsize: usize, mut f: F)
    where
        F: FnMut() -> f64,
    {
        println!(
            "Benchmarking {}: {} operations...",
            name, self.measurement_iterations
        );

        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = f();
        }

        // Actual measurements
        let mut times = Vec::with_capacity(self.measurement_iterations);
        let start_memory = self.get_memory_usage();

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            let _ = f();
            let elapsed = start.elapsed();
            times.push(elapsed.as_nanos() as f64);
        }

        let end_memory = self.get_memory_usage();
        let memory_usage = end_memory.saturating_sub(start_memory);

        // Calculate statistics
        let mean_time_ns = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times
            .iter()
            .map(|&t| (t - mean_time_ns).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let std_dev_ns = variance.sqrt();
        let min_time_ns = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time_ns = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let throughput_ops_per_sec = 1_000_000_000.0 / mean_time_ns;

        let result = BenchmarkResult {
            function_name: name.to_string(),
            mean_time_ns,
            std_dev_ns,
            min_time_ns,
            max_time_ns,
            throughput_ops_per_sec,
            memory_usage_bytes: memory_usage,
            samples: self.measurement_iterations,
        };

        println!(
            "  Mean: {:.2} ns, Std Dev: {:.2} ns",
            mean_time_ns, std_dev_ns
        );
        println!("  Throughput: {:.0} ops/sec", throughput_ops_per_sec);

        self.results.insert(name.to_string(), result);
    }

    fn benchmark_array_function<F>(&mut self, name: &str, arraysize: usize, mut f: F)
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        println!(
            "Benchmarking {} with array size {}: {} iterations...",
            name, arraysize, self.measurement_iterations
        );

        let input_array = Array1::linspace(0.1, 10.0, arraysize);

        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = f(&input_array);
        }

        // Actual measurements
        let mut times = Vec::with_capacity(self.measurement_iterations);
        let start_memory = self.get_memory_usage();

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            let _ = f(&input_array);
            let elapsed = start.elapsed();
            times.push(elapsed.as_nanos() as f64);
        }

        let end_memory = self.get_memory_usage();
        let memory_usage = end_memory.saturating_sub(start_memory);

        // Calculate statistics
        let mean_time_ns = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times
            .iter()
            .map(|&t| (t - mean_time_ns).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let std_dev_ns = variance.sqrt();
        let min_time_ns = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time_ns = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let throughput_elements_per_sec = (arraysize as f64) * 1_000_000_000.0 / mean_time_ns;

        let result = BenchmarkResult {
            function_name: format!("{}_{}_elements", name, arraysize),
            mean_time_ns,
            std_dev_ns,
            min_time_ns,
            max_time_ns,
            throughput_ops_per_sec: throughput_elements_per_sec,
            memory_usage_bytes: memory_usage,
            samples: self.measurement_iterations,
        };

        println!(
            "  Mean: {:.2} ns, Std Dev: {:.2} ns",
            mean_time_ns, std_dev_ns
        );
        println!(
            "  Throughput: {:.0} elements/sec",
            throughput_elements_per_sec
        );

        self.results.insert(result.function_name.clone(), result);
    }

    fn get_memory_usage(&self) -> usize {
        // Simplified memory usage estimation
        // In a real implementation, this would use more sophisticated memory tracking

        // For demonstration, we'll return a basic estimate
        // Real implementation would use tools like jemalloc or tcmalloc statistics

        // Get current process memory usage from /proc/self/status if available
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback: return 0 if we can't measure memory
        0
    }

    fn export_results(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut json_results = serde_json::Map::new();

        for (name, result) in &self.results {
            let result_json = json!({
                "mean_time_ns": result.mean_time_ns,
                "std_dev_ns": result.std_dev_ns,
                "min_time_ns": result.min_time_ns,
                "max_time_ns": result.max_time_ns,
                "throughput_ops_per_sec": result.throughput_ops_per_sec,
                "memory_usage_bytes": result.memory_usage_bytes,
                "samples": result.samples
            });
            json_results.insert(name.clone(), result_json);
        }

        let output = serde_json::to_string_pretty(&json_results)?;
        std::fs::write(filename, output)?;

        println!("Results exported to {}", filename);
        Ok(())
    }

    fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("PERFORMANCE MONITORING SUMMARY");
        println!("{}", "=".repeat(80));

        println!(
            "{:<30} {:>12} {:>12} {:>15} {:>10}",
            "Function", "Mean (ns)", "Std Dev", "Throughput/s", "Memory (KB)"
        );
        println!("{}", "-".repeat(80));

        let mut sorted_results: Vec<_> = self.results.iter().collect();
        sorted_results.sort_by_key(|(name_, _)| name_.as_str());

        for (name, result) in sorted_results {
            println!(
                "{:<30} {:>12.2} {:>12.2} {:>15.0} {:>10}",
                name,
                result.mean_time_ns,
                result.std_dev_ns,
                result.throughput_ops_per_sec,
                result.memory_usage_bytes / 1024
            );
        }

        println!("{}", "=".repeat(80));

        // Performance warnings
        let slow_functions: Vec<_> = self
            .results
            .iter()
            .filter(|(_, result)| result.mean_time_ns > 1000.0)
            .collect();

        if !slow_functions.is_empty() {
            println!("\n⚠️  Functions with mean time > 1μs:");
            for (name, result) in slow_functions {
                println!("  - {}: {:.2} ns", name, result.mean_time_ns);
            }
        }

        // High variability warnings
        let variable_functions: Vec<_> = self
            .results
            .iter()
            .filter(|(_, result)| result.std_dev_ns / result.mean_time_ns > 0.3)
            .collect();

        if !variable_functions.is_empty() {
            println!("\n📊 Functions with high variability (>30% CV):");
            for (name, result) in variable_functions {
                let cv = result.std_dev_ns / result.mean_time_ns;
                println!("  - {}: CV = {:.1}%", name, cv * 100.0);
            }
        }
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Performance Monitor for scirs2-special");
    println!("This will benchmark core special functions for regression detection.\n");

    let mut monitor = PerformanceMonitor::new();

    // Benchmark individual functions with single values
    println!("📊 Benchmarking scalar functions...");

    monitor.benchmark_function("gamma_function", 1, || gamma(2.5));
    monitor.benchmark_function("gamma_ln", 1, || gammaln(10.5));
    monitor.benchmark_function("digamma", 1, || digamma(3.7));
    monitor.benchmark_function("beta_function", 1, || beta(2.5, 3.5));

    monitor.benchmark_function("bessel_j0", 1, || j0(5.2));
    monitor.benchmark_function("bessel_j1", 1, || j1(3.8));
    monitor.benchmark_function("bessel_y0", 1, || y0(2.1));
    monitor.benchmark_function("bessel_i0", 1, || i0(1.5));
    monitor.benchmark_function("bessel_k0", 1, || k0(2.3));

    monitor.benchmark_function("error_function", 1, || erf(1.5));
    monitor.benchmark_function("error_function_comp", 1, || erfc(2.1));
    monitor.benchmark_function("error_function_inv", 1, || erfinv(0.7));

    monitor.benchmark_function("airy_ai", 1, || ai(1.5));
    monitor.benchmark_function("airy_bi", 1, || bi(1.5));

    // Benchmark array operations
    println!("\n📈 Benchmarking array operations...");

    let arraysizes = vec![100, 1000, 10000];

    for &size in &arraysizes {
        monitor.benchmark_array_function("gamma", size, |arr| arr.mapv(|x| gamma(x)));

        monitor.benchmark_array_function("bessel_j0", size, |arr| arr.mapv(|x| j0(x)));

        monitor.benchmark_array_function("erf", size, |arr| arr.mapv(|x| erf(x)));

        // Test SIMD operations if available
        #[cfg(feature = "simd")]
        {
            monitor.benchmark_array_function("gamma_simd", size, |arr| {
                use scirs2_special::simd_ops::gamma_f64_simd;
                gamma_f64_simd(&arr.view()).unwrap()
            });
        }
    }

    // Benchmark complex operations
    println!("\n🌀 Benchmarking complex functions...");

    use num_complex::Complex64;
    let z = Complex64::new(1.5, 0.5);

    // Complex spherical harmonic benchmark (commented out - needs proper complex support)
    {
        monitor.benchmark_function("gamma_complex", 1, || {
            use scirs2_special::gamma_complex;
            gamma_complex(z).norm()
        });

        monitor.benchmark_function("erf_complex", 1, || {
            use scirs2_special::erf_complex;
            erf_complex(z).norm()
        });
    }

    // Benchmark advanced functions
    println!("\n🔬 Benchmarking advanced functions...");

    monitor.benchmark_function("elliptic_k", 1, || elliptic_k(0.7));
    monitor.benchmark_function("elliptic_e", 1, || elliptic_e(0.7));

    // monitor.benchmark_function("spherical_harmonic", 1, || sph_harm(2, 1, 1.0, 0.5).norm());

    monitor.benchmark_function("wright_bessel", 1, || {
        wright_bessel(1.0, 1.0, 1.5).unwrap_or(0.0)
    });

    // Performance stress tests
    println!("\n💪 Running stress tests...");

    // Test with challenging parameter ranges
    monitor.benchmark_function("gamma_large_arg", 1, || gamma(50.0));
    monitor.benchmark_function("gamma_small_arg", 1, || gamma(0.01));
    monitor.benchmark_function("bessel_large_arg", 1, || j0(100.0));
    monitor.benchmark_function("erf_large_arg", 1, || erf(5.0));

    // Print summary
    monitor.print_summary();

    // Export results for CI/CD
    monitor.export_results("performance_results.json")?;

    // Create additional analysis for CI
    create_ci_analysis(&monitor)?;

    println!("\n✅ Performance monitoring completed successfully!");
    println!("Results have been saved to performance_results.json");

    Ok(())
}

#[allow(dead_code)]
fn create_ci_analysis(monitor: &PerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 Creating CI/CD analysis...");

    // Create a structured output for CI consumption
    let mut ci_metrics = serde_json::Map::new();

    // Aggregate metrics by function family
    let mut gamma_times = Vec::new();
    let mut bessel_times = Vec::new();
    let mut erf_times = Vec::new();
    let mut array_times = HashMap::new();

    for (name, result) in &monitor.results {
        if name.contains("gamma") {
            gamma_times.push(result.mean_time_ns);
        } else if name.contains("bessel") || name.contains("j0") || name.contains("i0") {
            bessel_times.push(result.mean_time_ns);
        } else if name.contains("erf") {
            erf_times.push(result.mean_time_ns);
        }

        // Collect array operation times
        if name.contains("_elements") {
            let base_name = name.split('_').next().unwrap_or("unknown");
            array_times
                .entry(base_name.to_string())
                .or_insert(Vec::new())
                .push(result.mean_time_ns);
        }
    }

    // Calculate family averages
    if !gamma_times.is_empty() {
        let avg = gamma_times.iter().sum::<f64>() / gamma_times.len() as f64;
        ci_metrics.insert("gamma_family_avg_ns".to_string(), json!(avg));
    }

    if !bessel_times.is_empty() {
        let avg = bessel_times.iter().sum::<f64>() / bessel_times.len() as f64;
        ci_metrics.insert("bessel_family_avg_ns".to_string(), json!(avg));
    }

    if !erf_times.is_empty() {
        let avg = erf_times.iter().sum::<f64>() / erf_times.len() as f64;
        ci_metrics.insert("erf_family_avg_ns".to_string(), json!(avg));
    }

    // Array performance scaling analysis
    for (func, times) in array_times {
        if times.len() >= 2 {
            let scaling_factor = times.last().unwrap() / times.first().unwrap();
            ci_metrics.insert(format!("{}_scaling_factor", func), json!(scaling_factor));
        }
    }

    // Performance quality metrics
    let total_functions = monitor.results.len();
    let slow_functions = monitor
        .results
        .iter()
        .filter(|(_, r)| r.mean_time_ns > 1000.0)
        .count();
    let variable_functions = monitor
        .results
        .iter()
        .filter(|(_, r)| r.std_dev_ns / r.mean_time_ns > 0.3)
        .count();

    ci_metrics.insert("total_functions_tested".to_string(), json!(total_functions));
    ci_metrics.insert("slow_functions_count".to_string(), json!(slow_functions));
    ci_metrics.insert(
        "variable_functions_count".to_string(),
        json!(variable_functions),
    );
    ci_metrics.insert(
        "performance_score".to_string(),
        json!(
            100.0
                - (slow_functions as f64 + variable_functions as f64) / total_functions as f64
                    * 100.0
        ),
    );

    // Export CI metrics
    let ci_output = serde_json::to_string_pretty(&ci_metrics)?;
    std::fs::write("ci_performance_metrics.json", ci_output)?;

    println!("CI analysis saved to ci_performance_metrics.json");

    Ok(())
}
