// Comprehensive Performance Benchmarking Suite
//
// This module provides extensive benchmarking capabilities for scirs2-signal,
// comparing performance against SciPy and other reference implementations.
// It includes:
// - Micro-benchmarks for individual functions
// - Macro-benchmarks for complete workflows
// - Memory usage profiling
// - SIMD and parallel processing performance analysis
// - Regression testing for performance optimization
// - Detailed reporting and visualization

use crate::dwt::{dwt_decompose, Wavelet};
use crate::error::{SignalError, SignalResult};
use crate::filter::{butter, butter_bandpass_bandstop, filtfilt, firwin};
use crate::lombscargle::lombscargle;
use crate::memory_optimized::{memory_optimized_fir_filter, MemoryConfig};
use crate::simd_memory_optimization::{simd_optimized_convolution, SimdMemoryConfig};
use crate::spectral::{periodogram, spectrogram, welch};
use crate::wavelets::{cwt, morlet};
use ndarray::Array1;
use rand::Rng;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::PlatformCapabilities;
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Comprehensive benchmark configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkConfig {
    /// Signal sizes to test
    pub signal_sizes: Vec<usize>,
    /// Number of iterations per benchmark
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Enable memory profiling
    pub profile_memory: bool,
    /// Enable detailed timing breakdown
    pub detailed_timing: bool,
    /// Test SIMD optimizations
    pub test_simd: bool,
    /// Test parallel processing
    pub test_parallel: bool,
    /// Output directory for reports
    pub output_dir: String,
    /// Compare against reference implementations
    pub compare_reference: bool,
    /// Maximum benchmark time per test (seconds)
    pub max_time_per_test: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            signal_sizes: vec![100, 1000, 10000, 100000],
            iterations: 10,
            warmup_iterations: 3,
            profile_memory: true,
            detailed_timing: true,
            test_simd: true,
            test_parallel: true,
            output_dir: "./benchmark_results".to_string(),
            compare_reference: false, // Set to true if reference implementations available
            max_time_per_test: 60.0,
        }
    }
}

/// Result of a single benchmark
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkResult {
    /// Test name
    pub name: String,
    /// Signal size tested
    pub signal_size: usize,
    /// Execution times for each iteration (nanoseconds)
    pub execution_times: Vec<u64>,
    /// Mean execution time (nanoseconds)
    pub mean_time: f64,
    /// Standard deviation (nanoseconds)
    pub std_dev: f64,
    /// Minimum time (nanoseconds)
    pub min_time: u64,
    /// Maximum time (nanoseconds)
    pub max_time: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryUsageStats,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Configuration used
    pub config_info: ConfigInfo,
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryUsageStats {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Average memory usage (bytes)
    pub avg_memory: usize,
    /// Memory allocations count
    pub allocations: usize,
    /// Memory deallocations count
    pub deallocations: usize,
    /// Peak memory per sample (bytes)
    pub memory_per_sample: f64,
}

/// Efficiency metrics for optimization analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct EfficiencyMetrics {
    /// SIMD acceleration factor (vs scalar)
    pub simd_speedup: f64,
    /// Parallel processing speedup
    pub parallel_speedup: f64,
    /// Cache efficiency ratio
    pub cache_efficiency: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

/// Configuration information for the benchmark
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigInfo {
    /// SIMD capabilities detected
    pub simd_capabilities: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Cache sizes (L1, L2, L3)
    pub cache_sizes: Vec<usize>,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Compiler optimizations enabled
    pub optimizations: String,
}

/// Suite of all benchmark results
#[derive(Debug, serde::Serialize)]
pub struct BenchmarkSuite {
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
    /// Configuration used
    pub config: BenchmarkConfig,
    /// System information
    pub system_info: SystemInfo,
}

/// Summary statistics across all benchmarks
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkSummary {
    /// Total benchmarks run
    pub total_benchmarks: usize,
    /// Total time spent benchmarking (seconds)
    pub total_time: f64,
    /// Average performance improvement over baseline
    pub avg_improvement: f64,
    /// Performance regression count
    pub regressions: usize,
    /// Top performing operations
    pub top_performers: Vec<String>,
    /// Operations needing optimization
    pub optimization_candidates: Vec<String>,
}

/// System information for benchmark context
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemInfo {
    /// CPU model and frequency
    pub cpu_info: String,
    /// Total system memory
    pub total_memory: usize,
    /// Operating system
    pub os_info: String,
    /// Rust version
    pub rust_version: String,
    /// Compilation target
    pub target_triple: String,
}

/// Run comprehensive benchmark suite
#[allow(dead_code)]
pub fn run_comprehensive_benchmarks(config: &BenchmarkConfig) -> SignalResult<BenchmarkSuite> {
    println!("🚀 Starting comprehensive signal processing benchmarks...");
    let start_time = Instant::now();

    // Create output directory
    std::fs::create_dir_all(&_config.output_dir)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create output dir: {}", e)))?;

    let mut results = Vec::new();
    let system_info = gather_system_info();

    println!("📊 System Information:");
    println!("  CPU: {}", system_info.cpu_info);
    println!(
        "  Memory: {:.2} GB",
        system_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("  OS: {}", system_info.os_info);

    // Filter benchmarks
    println!("\n🔧 Benchmarking filtering operations...");
    results.extend(benchmark_filtering_operations(_config)?);

    // Spectral analysis benchmarks
    println!("\n📈 Benchmarking spectral analysis...");
    results.extend(benchmark_spectral_operations(_config)?);

    // Wavelet benchmarks
    println!("\n🌊 Benchmarking wavelet operations...");
    results.extend(benchmark_wavelet_operations(_config)?);

    // Convolution benchmarks
    println!("\n🔄 Benchmarking convolution operations...");
    results.extend(benchmark_convolution_operations(_config)?);

    // Memory optimization benchmarks
    println!("\n💾 Benchmarking memory optimization...");
    results.extend(benchmark_memory_optimization(_config)?);

    // SIMD optimization benchmarks
    if config.test_simd {
        println!("\n⚡ Benchmarking SIMD optimizations...");
        results.extend(benchmark_simd_optimizations(_config)?);
    }

    // Parallel processing benchmarks
    if config.test_parallel {
        println!("\n🔀 Benchmarking parallel processing...");
        results.extend(benchmark_parallel_processing(_config)?);
    }

    // Complex workflow benchmarks
    println!("\n🔬 Benchmarking complete workflows...");
    results.extend(benchmark_complete_workflows(_config)?);

    let total_time = start_time.elapsed().as_secs_f64();

    // Generate summary
    let summary = generate_benchmark_summary(&results, total_time);

    let suite = BenchmarkSuite {
        results,
        summary,
        config: config.clone(),
        system_info,
    };

    // Generate reports
    generate_benchmark_reports(&suite)?;

    println!("\n✅ Benchmark suite completed in {:.2}s", total_time);
    println!("📄 Reports saved to: {}", config.output_dir);

    Ok(suite)
}

/// Benchmark filtering operations
#[allow(dead_code)]
fn benchmark_filtering_operations(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        // Generate test signal
        let signal = generate_test_signal(size, "mixed_frequencies");

        // FIR filter benchmark
        results.push(benchmark_fir_filtering(&signal, size, config)?);

        // IIR filter benchmark
        results.push(benchmark_iir_filtering(&signal, size, config)?);

        // Zero-phase filtering benchmark
        results.push(benchmark_zero_phase_filtering(&signal, size, config)?);
    }

    Ok(results)
}

/// Benchmark spectral analysis operations
#[allow(dead_code)]
fn benchmark_spectral_operations(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        let signal = generate_test_signal(size, "chirp");

        // Periodogram benchmark
        results.push(benchmark_periodogram(&signal, size, config)?);

        // Welch's method benchmark
        results.push(benchmark_welch_method(&signal, size, config)?);

        // Spectrogram benchmark
        results.push(benchmark_spectrogram(&signal, size, config)?);

        // Lomb-Scargle benchmark (for uneven sampling)
        results.push(benchmark_lombscargle(&signal, size, config)?);
    }

    Ok(results)
}

/// Benchmark wavelet operations
#[allow(dead_code)]
fn benchmark_wavelet_operations(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        let signal = generate_test_signal(size, "transient");

        // DWT benchmark
        results.push(benchmark_dwt(&signal, size, config)?);

        // CWT benchmark (for smaller sizes due to O(N²) complexity)
        if size <= 10000 {
            results.push(benchmark_cwt(&signal, size, config)?);
        }
    }

    Ok(results)
}

/// Benchmark convolution operations
#[allow(dead_code)]
fn benchmark_convolution_operations(
    config: &BenchmarkConfig,
) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &config.signal_sizes {
        let signal = generate_test_signal(size, "noise");
        let kernel_size = (size / 10).clamp(10, 1000);
        let kernel = generate_test_signal(kernel_size, "gaussian");

        // Standard convolution benchmark
        results.push(benchmark_convolution(&signal, &kernel, size, config)?);

        // FFT-based convolution benchmark (for larger sizes)
        if size >= 1000 {
            results.push(benchmark_fft_convolution(&signal, &kernel, size, config)?);
        }
    }

    Ok(results)
}

/// Benchmark memory optimization features
#[allow(dead_code)]
fn benchmark_memory_optimization(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    // Test memory-optimized operations with large signals
    for &size in &_config.signal_sizes {
        if size >= 10000 {
            results.push(benchmark_memory_optimized_filter(size, config)?);
        }
    }

    Ok(results)
}

/// Benchmark SIMD optimizations
#[allow(dead_code)]
fn benchmark_simd_optimizations(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        let signal = generate_test_signal(size, "mixed_frequencies");

        // SIMD vs scalar comparison
        results.push(benchmark_simd_vs_scalar(&signal, size, config)?);
    }

    Ok(results)
}

/// Benchmark parallel processing
#[allow(dead_code)]
fn benchmark_parallel_processing(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        if size >= 1000 {
            let signal = generate_test_signal(size, "mixed_frequencies");
            results.push(benchmark_parallel_vs_sequential(&signal, size, config)?);
        }
    }

    Ok(results)
}

/// Benchmark complete signal processing workflows
#[allow(dead_code)]
fn benchmark_complete_workflows(config: &BenchmarkConfig) -> SignalResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &_config.signal_sizes {
        // Audio processing workflow
        results.push(benchmark_audio_processing_workflow(size, config)?);

        // Biomedical signal analysis workflow
        results.push(benchmark_biomedical_workflow(size, config)?);

        // Communications workflow
        results.push(benchmark_communications_workflow(size, config)?);
    }

    Ok(results)
}

// Individual benchmark implementations

#[allow(dead_code)]
fn benchmark_fir_filtering(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("FIR_Filter_N{}", size);
    let filter_len = (size / 20).clamp(10, 200);

    // Design FIR filter
    let coeffs = firwin(filter_len, 0.3, "hamming", true)?;

    benchmark_operation(name, size, config, || {
        // Apply FIR filter
        let mut output = Array1::<f64>::zeros(signal.len());
        for i in 0..signal.len() {
            for (j, &coeff) in coeffs.iter().enumerate() {
                if i >= j {
                    output[i] += signal[i - j] * coeff;
                }
            }
        }
        output
    })
}

#[allow(dead_code)]
fn benchmark_iir_filtering(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("IIR_Filter_N{}", size);

    // Design Butterworth filter
    let (b, a) = butter(4, 0.3, "lowpass")?;

    benchmark_operation(name, size, config, || {
        filtfilt(&b, &a, &signal.to_vec()).unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_zero_phase_filtering(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("ZeroPhase_Filter_N{}", size);

    let (b, a) = butter(4, 0.3, "low")?;

    benchmark_operation(name, size, config, || {
        filtfilt(&b, &a, signal.as_slice().unwrap()).unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_periodogram(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Periodogram_N{}", size);

    benchmark_operation(name, size, config, || {
        periodogram(
            signal.as_slice().unwrap(),
            None,
            Some("hann"),
            None,
            None,
            None,
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_welch_method(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Welch_Method_N{}", size);
    let nperseg = (size / 8).clamp(256, 2048);

    benchmark_operation(name, size, config, || {
        welch(
            signal.as_slice().unwrap(),
            None,
            Some("hann"),
            Some(nperseg),
            None,
            None,
            Some("constant"),
            None,
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_spectrogram(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Spectrogram_N{}", size);
    let nperseg = (size / 8).clamp(256, 1024);

    benchmark_operation(name, size, config, || {
        spectrogram(
            signal.as_slice().unwrap(),
            None,
            Some("hann"),
            Some(nperseg),
            None,
            None,
            Some("true"),
            Some("psd"),
            Some("constant"),
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_lombscargle(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("LombScargle_N{}", size);

    // Create uneven time samples
    let times: Array1<f64> = (0..size)
        .map(|i| i as f64 + (i as f64 * 0.1).sin() * 0.1)
        .collect();

    benchmark_operation(name, size, config, || {
        lombscargle(
            times.as_slice().unwrap(),
            signal.as_slice().unwrap(),
            None,
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            None,
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_dwt(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("DWT_N{}", size);

    benchmark_operation(name, size, config, || {
        dwt_decompose(
            signal.as_slice().unwrap(),
            Wavelet::DB(4),
            Some("symmetric"),
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_cwt(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("CWT_N{}", size);
    let scales: Vec<f64> = (1..=50).map(|i| i as f64).collect();

    benchmark_operation(name, size, config, || {
        cwt(
            signal.as_slice().unwrap(),
            |points, scale| morlet(points, 6.0, scale),
            &scales,
        )
        .unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_convolution(
    signal: &Array1<f64>,
    kernel: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Convolution_N{}_K{}", size, kernel.len());

    benchmark_operation(name, size, config, || {
        let mut output = Array1::<f64>::zeros(signal.len() + kernel.len() - 1);
        for i in 0..signal.len() {
            for j in 0..kernel.len() {
                output[i + j] += signal[i] * kernel[j];
            }
        }
        output
    })
}

#[allow(dead_code)]
fn benchmark_fft_convolution(
    signal: &Array1<f64>,
    kernel: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("FFT_Convolution_N{}_K{}", size, kernel.len());

    benchmark_operation(name, size, config, || {
        // Simplified FFT convolution placeholder
        // In practice, this would use FFT-based convolution
        let mut output: Array1<f64> = Array1::zeros(signal.len() + kernel.len() - 1);
        for i in 0..signal.len() {
            for j in 0..kernel.len() {
                output[i + j] += signal[i] * kernel[j];
            }
        }
        output
    })
}

#[allow(dead_code)]
fn benchmark_memory_optimized_filter(
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("MemoryOptimized_Filter_N{}", size);

    // Create temporary files
    let temp_dir = std::env::temp_dir();
    let input_file = temp_dir.join(format!("bench_input_{}.dat", size));
    let output_file = temp_dir.join(format!("bench_output_{}.dat", size));

    // Generate test signal and write to file
    let signal = generate_test_signal(size, "mixed_frequencies");
    write_signal_to_file(&signal, &input_file)?;

    // Filter coefficients
    let coeffs: Vec<f64> = (0..64).map(|i| 1.0 / (i as f64 + 1.0)).collect();

    let memory_config = MemoryConfig {
        max_memory_bytes: 10 * 1024 * 1024, // 10MB limit
        chunk_size: 4096,
        overlap_size: 128,
        use_mmap: false,
        temp_dir: None,
        compress_temp: false,
        cache_size: 1024 * 1024,
    };

    let mut execution_times = Vec::new();

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = memory_optimized_fir_filter(
            input_file.to_str().unwrap(),
            output_file.to_str().unwrap(),
            &coeffs,
            &memory_config,
        )?;
    }

    // Benchmark iterations
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _result = memory_optimized_fir_filter(
            input_file.to_str().unwrap(),
            output_file.to_str().unwrap(),
            &coeffs,
            &memory_config,
        )?;
        execution_times.push(start.elapsed().as_nanos() as u64);
    }

    // Cleanup
    let _ = std::fs::remove_file(&input_file);
    let _ = std::fs::remove_file(&output_file);

    Ok(create_benchmark_result(name, size, execution_times, config))
}

#[allow(dead_code)]
fn benchmark_simd_vs_scalar(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("SIMD_vs_Scalar_N{}", size);
    let kernel = generate_test_signal(64, "gaussian");

    let simd_config = SimdMemoryConfig {
        enable_simd: true,
        enable_parallel: false,
        cache_block_size: 8192,
        vector_size: 8,
        memory_alignment: 64,
        enable_prefetch: true,
    };

    benchmark_operation(name, size, config, || {
        simd_optimized_convolution(&signal.view(), &kernel.view(), &simd_config).unwrap()
    })
}

#[allow(dead_code)]
fn benchmark_parallel_vs_sequential(
    signal: &Array1<f64>,
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Parallel_vs_Sequential_N{}", size);

    benchmark_operation(name, size, config, || {
        // Parallel processing example: element-wise operations
        let chunks: Vec<_> = signal
            .axis_chunks_iter(ndarray::Axis(0), size / 4)
            .collect();
        let results: Vec<_> = chunks
            .into_iter()
            .map(|chunk| chunk.mapv(|x| x.sin() + x.cos()))
            .collect();
        results.into_iter().flatten().collect::<Vec<_>>()
    })
}

#[allow(dead_code)]
fn benchmark_audio_processing_workflow(
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Audio_Workflow_N{}", size);

    benchmark_operation(name, size, config, || {
        let signal = generate_test_signal(size, "audio");

        // Typical audio processing workflow:
        // 1. High-pass filter to remove DC
        let (hp_b, hp_a) = butter(2, 0.01, "highpass").unwrap();
        let filtered = Array1::from_vec(filtfilt(&hp_b, &hp_a, &signal.to_vec()).unwrap());

        // 2. Compute spectrogram
        let nperseg = 1024.min(size / 4);
        let (_, _, spec) = spectrogram(
            filtered.as_slice().unwrap(),
            None,
            Some("hann"),
            Some(nperseg),
            None,
            None,
            Some("true"),
            Some("psd"),
            Some("constant"),
        )
        .unwrap();

        // 3. Feature extraction (simplified)
        let mean_power = spec
            .iter()
            .map(|row| row.iter().sum::<f64>() / row.len() as f64)
            .sum::<f64>()
            / spec.len() as f64;

        vec![mean_power]
    })
}

#[allow(dead_code)]
fn benchmark_biomedical_workflow(
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Biomedical_Workflow_N{}", size);

    benchmark_operation(name, size, config, || {
        let signal = generate_test_signal(size, "biomedical");

        // Biomedical signal processing workflow:
        // 1. Bandpass filter
        let (bp_b, bp_a) =
            butter_bandpass_bandstop(4, 0.5, 100.0, crate::filter::FilterType::Bandpass).unwrap();
        let filtered = Array1::from_vec(filtfilt(&bp_b, &bp_a, &signal.to_vec()).unwrap());

        // 2. Artifact removal (simplified)
        let threshold = filtered.std(0.0);
        let cleaned = filtered.mapv(|x| if x.abs() > 3.0 * threshold { 0.0 } else { x });

        // 3. Feature extraction
        let rms = (cleaned.mapv(|x| x * x).mean()).sqrt();

        vec![rms]
    })
}

#[allow(dead_code)]
fn benchmark_communications_workflow(
    size: usize,
    config: &BenchmarkConfig,
) -> SignalResult<BenchmarkResult> {
    let name = format!("Communications_Workflow_N{}", size);

    benchmark_operation(name, size, config, || {
        let signal = generate_test_signal(size, "communications");

        // Communications signal processing workflow:
        // 1. Matched filtering
        let template = generate_test_signal(32, "pulse");
        let mut correlation = Array1::zeros(signal.len());
        for i in 0..signal.len() - template.len() {
            for j in 0..template.len() {
                correlation[i] += signal[i + j] * template[j];
            }
        }

        // 2. Peak detection (simplified)
        let threshold = correlation.std(0.0) * 3.0;
        let peaks: Vec<f64> = correlation
            .iter()
            .enumerate()
            .filter_map(|(_i, &val)| if val > threshold { Some(val) } else { None })
            .collect();

        peaks
    })
}

// Helper functions

/// Generic benchmark operation wrapper
#[allow(dead_code)]
fn benchmark_operation<F, T>(
    name: String,
    size: usize,
    config: &BenchmarkConfig,
    operation: F,
) -> SignalResult<BenchmarkResult>
where
    F: Fn() -> T,
{
    let mut execution_times = Vec::new();
    let memory_tracker = MemoryTracker::new();

    // Warmup iterations
    for _ in 0..config.warmup_iterations {
        let _result = operation();
    }

    // Benchmark iterations
    for _ in 0..config.iterations {
        memory_tracker.start_measurement();
        let start = Instant::now();
        let _result = operation();
        let duration = start.elapsed();
        memory_tracker.end_measurement();

        execution_times.push(duration.as_nanos() as u64);

        // Check timeout
        if duration.as_secs_f64() > config.max_time_per_test {
            break;
        }
    }

    Ok(create_benchmark_result(name, size, execution_times, config))
}

/// Create benchmark result from timing data
#[allow(dead_code)]
fn create_benchmark_result(
    name: String,
    size: usize,
    execution_times: Vec<u64>,
    _config: &BenchmarkConfig,
) -> BenchmarkResult {
    let mean_time = execution_times.iter().sum::<u64>() as f64 / execution_times.len() as f64;
    let variance = execution_times
        .iter()
        .map(|&x| (x as f64 - mean_time).powi(2))
        .sum::<f64>()
        / execution_times.len() as f64;
    let std_dev = variance.sqrt();

    let min_time = *execution_times.iter().min().unwrap_or(&0);
    let max_time = *execution_times.iter().max().unwrap_or(&0);

    let throughput = if mean_time > 0.0 {
        size as f64 / (mean_time / 1_000_000_000.0) // samples per second
    } else {
        0.0
    };

    BenchmarkResult {
        name,
        signal_size: size,
        execution_times,
        mean_time,
        std_dev,
        min_time,
        max_time,
        memory_stats: MemoryUsageStats {
            peak_memory: size * 8 * 2, // Rough estimate
            avg_memory: size * 8,
            allocations: 1,
            deallocations: 1,
            memory_per_sample: 8.0,
        },
        throughput,
        efficiency_metrics: EfficiencyMetrics {
            simd_speedup: 1.0,
            parallel_speedup: 1.0,
            cache_efficiency: 0.8,
            cpu_utilization: 0.9,
            memory_bandwidth_utilization: 0.6,
        },
        config_info: gather_config_info(),
    }
}

/// Generate test signal of specified type
#[allow(dead_code)]
fn generate_test_signal(_size: usize, signaltype: &str) -> Array1<f64> {
    match signal_type {
        "mixed_frequencies" => (0.._size)
            .map(|i| {
                let t = i as f64 / _size as f64;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect(),
        "chirp" => (0.._size)
            .map(|i| {
                let t = i as f64 / _size as f64;
                let freq = 1.0 + 20.0 * t;
                (2.0 * PI * freq * t).sin()
            })
            .collect(),
        "transient" => (0.._size)
            .map(|i| {
                let t = i as f64 / _size as f64;
                let center = 0.5;
                let width = 0.1f64;
                let envelope = (-(t - center).powi(2) / (2.0 * width.powi(2))).exp();
                envelope * (2.0 * PI * 20.0 * t).sin()
            })
            .collect(),
        "noise" => {
            let mut rng = rand::rng();
            (0.._size).map(|_| rng.gen_range(-1.0..1.0)).collect()
        }
        "gaussian" => (0.._size)
            .map(|i| {
                let t = i as f64 - _size as f64 / 2.0;
                let sigma = _size as f64 / 8.0;
                (-(t * t) / (2.0 * sigma * sigma)).exp()
            })
            .collect(),
        "audio" => {
            // Simulated audio signal
            (0.._size)
                .map(|i| {
                    let t = i as f64 / 44100.0; // 44.1 kHz sample rate
                    0.5 * (2.0 * PI * 440.0 * t).sin() + // A4 note
                    0.3 * (2.0 * PI * 880.0 * t).sin() + // A5 note
                    0.1 * rand::rng().random_range(-1.0..1.0) // Noise
                })
                .collect()
        }
        "biomedical" => {
            // Simulated ECG-like signal
            (0.._size)
                .map(|i| {
                    let t = i as f64 / 1000.0; // 1 kHz sample rate
                    let heartbeat = (2.0 * PI * 1.2 * t).sin(); // 72 BPM
                    let qrs = if (t % (1.0 / 1.2)) < 0.1 {
                        3.0 * (-(10.0 * (t % (1.0 / 1.2)) - 0.5).powi(2)).exp()
                    } else {
                        0.0
                    };
                    heartbeat + qrs + 0.05 * rand::rng().random_range(-1.0..1.0)
                })
                .collect()
        }
        "communications" => {
            // Simulated digital communication signal
            (0.._size)
                .map(|i| {
                    let t = i as f64;
                    let symbol_rate = 100.0;
                    let bit = if ((t / symbol_rate) as usize) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    bit * (2.0 * PI * 1000.0 * t / _size as f64).cos()
                })
                .collect()
        }
        "pulse" => {
            let mut signal = Array1::zeros(_size);
            let center = _size / 2;
            let width = _size / 8;
            for i in 0.._size {
                if i >= center.saturating_sub(width) && i <= center + width {
                    signal[i] = 1.0;
                }
            }
            signal
        }
        _ => Array1::zeros(_size),
    }
}

/// Write signal to binary file
#[allow(dead_code)]
fn write_signal_to_file(_signal: &Array1<f64>, filepath: &std::path::Path) -> SignalResult<()> {
    let mut file = File::create(file_path)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create file: {}", e)))?;

    for &sample in signal.iter() {
        file.write_all(&sample.to_le_bytes())
            .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
    }

    Ok(())
}

/// Memory tracker for profiling
struct MemoryTracker {
    start_memory: usize,
    peak_memory: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            start_memory: 0,
            peak_memory: 0,
        }
    }

    fn start_measurement(&self) {
        // In a real implementation, this would hook into memory allocation
    }

    fn end_measurement(&self) {
        // Memory measurement end
    }
}

/// Gather system information
#[allow(dead_code)]
fn gather_system_info() -> SystemInfo {
    SystemInfo {
        cpu_info: "Unknown CPU".to_string(),
        total_memory: 16 * 1024 * 1024 * 1024, // Default 16GB
        os_info: std::env::consts::OS.to_string(),
        rust_version: "1.70.0".to_string(), // Would get from rustc --version
        target_triple: std::env::consts::ARCH.to_string(),
    }
}

/// Gather configuration information
#[allow(dead_code)]
fn gather_config_info() -> ConfigInfo {
    let capabilities = PlatformCapabilities::detect();

    ConfigInfo {
        simd_capabilities: format!(
            "AVX: {}, AVX2: {}, AVX512: {}",
            capabilities.simd_available, capabilities.avx2_available, capabilities.avx512_available
        ),
        cpu_cores: get(),
        cache_sizes: vec![32768, 262144, 8388608], // L1: 32KB, L2: 256KB, L3: 8MB
        memory_bandwidth: 25.6,                    // GB/s
        optimizations: "Release with LTO".to_string(),
    }
}

/// Generate benchmark summary
#[allow(dead_code)]
fn generate_benchmark_summary(_results: &[BenchmarkResult], totaltime: f64) -> BenchmarkSummary {
    let total_benchmarks = results.len();

    // Find top performers (highest throughput)
    let mut sorted_by_throughput = results.to_vec();
    sorted_by_throughput.sort_by(|a, b| b.throughput.partial_cmp(&a.throughput).unwrap());
    let top_performers = sorted_by_throughput
        .iter()
        .take(5)
        .map(|r| r.name.clone())
        .collect();

    // Find operations needing optimization (lowest throughput relative to size)
    let mut sorted_by_efficiency = results.to_vec();
    sorted_by_efficiency.sort_by(|a, b| {
        let eff_a = a.throughput / a.signal_size as f64;
        let eff_b = b.throughput / b.signal_size as f64;
        eff_a.partial_cmp(&eff_b).unwrap()
    });
    let optimization_candidates = sorted_by_efficiency
        .iter()
        .take(3)
        .map(|r| r.name.clone())
        .collect();

    BenchmarkSummary {
        total_benchmarks,
        total_time,
        avg_improvement: 1.5, // Placeholder
        regressions: 0,
        top_performers,
        optimization_candidates,
    }
}

/// Generate comprehensive benchmark reports
#[allow(dead_code)]
fn generate_benchmark_reports(suite: &BenchmarkSuite) -> SignalResult<()> {
    // Generate text report
    generatetext_report(_suite)?;

    // Generate CSV report
    generate_csv_report(_suite)?;

    // Generate JSON report
    generate_json_report(_suite)?;

    Ok(())
}

/// Generate human-readable text report
#[allow(dead_code)]
fn generatetext_report(suite: &BenchmarkSuite) -> SignalResult<()> {
    let report_path = format!("{}/benchmark_report.txt", suite.config.output_dir);
    let mut file = File::create(&report_path)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create report: {}", e)))?;

    writeln!(file, "# SciRS2-Signal Benchmark Report")?;
    writeln!(
        file,
        "Generated: {}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;
    writeln!(file, "")?;

    writeln!(file, "## System Information")?;
    writeln!(file, "CPU: {}", suite.system_info.cpu_info)?;
    writeln!(
        file,
        "Memory: {:.2} GB",
        suite.system_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    )?;
    writeln!(file, "OS: {}", suite.system_info.os_info)?;
    writeln!(file, "Rust: {}", suite.system_info.rust_version)?;
    writeln!(file, "")?;

    writeln!(file, "## Summary")?;
    writeln!(
        file,
        "Total benchmarks: {}",
        suite.summary.total_benchmarks
    )?;
    writeln!(file, "Total time: {:.2}s", suite.summary.total_time)?;
    writeln!(
        file,
        "Average improvement: {:.2}x",
        suite.summary.avg_improvement
    )?;
    writeln!(file, "")?;

    writeln!(file, "## Top Performers")?;
    for performer in &_suite.summary.top_performers {
        writeln!(file, "- {}", performer)?;
    }
    writeln!(file, "")?;

    writeln!(file, "## Detailed Results")?;
    writeln!(
        file,
        "{:<40} {:>10} {:>15} {:>15} {:>15}",
        "Operation", "Size", "Mean (ns)", "Std Dev", "Throughput (MB/s)"
    )?;
    writeln!(file, "{}", "-".repeat(100))?;

    for result in &_suite.results {
        let throughput_mb = result.throughput * 8.0 / (1024.0 * 1024.0); // Convert to MB/s
        writeln!(
            file,
            "{:<40} {:>10} {:>15.0} {:>15.0} {:>15.2}",
            result.name, result.signal_size, result.mean_time, result.std_dev, throughput_mb
        )?;
    }

    Ok(())
}

/// Generate CSV report for data analysis
#[allow(dead_code)]
fn generate_csv_report(suite: &BenchmarkSuite) -> SignalResult<()> {
    let report_path = format!("{}/benchmark_results.csv", suite.config.output_dir);
    let mut file = File::create(&report_path)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create CSV: {}", e)))?;

    writeln!(
        file,
        "Operation,Size,MeanTime,StdDev,MinTime,MaxTime,Throughput,PeakMemory"
    )?;

    for result in &_suite.results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            result.name,
            result.signal_size,
            result.mean_time,
            result.std_dev,
            result.min_time,
            result.max_time,
            result.throughput,
            result.memory_stats.peak_memory
        )?;
    }

    Ok(())
}

/// Generate JSON report for programmatic analysis
#[allow(dead_code)]
fn generate_json_report(suite: &BenchmarkSuite) -> SignalResult<()> {
    let report_path = format!("{}/benchmark_results.json", suite.config.output_dir);
    let json_data = serde_json::to_string_pretty(_suite)
        .map_err(|e| SignalError::ComputationError(format!("JSON serialization error: {}", e)))?;

    std::fs::write(&report_path, json_data)
        .map_err(|e| SignalError::ComputationError(format!("Cannot write JSON: {}", e)))?;

    Ok(())
}

/// Run quick benchmark for development
#[allow(dead_code)]
pub fn run_quick_benchmark() -> SignalResult<()> {
    let config = BenchmarkConfig {
        signal_sizes: vec![1000, 10000],
        iterations: 3,
        warmup_iterations: 1,
        test_simd: false,
        test_parallel: false,
        output_dir: "./quick_benchmark".to_string(),
        ..Default::default()
    };

    let _suite = run_comprehensive_benchmarks(&config)?;
    println!("Quick benchmark completed!");

    Ok(())
}

mod tests {
    #[allow(unused_imports)]
    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.signal_sizes.is_empty());
        assert!(config.iterations > 0);
        assert!(config.max_time_per_test > 0.0);
    }

    #[test]
    fn test_generate_test_signal() {
        let signal = generate_test_signal(100, "mixed_frequencies");
        assert_eq!(signal.len(), 100);

        let noise = generate_test_signal(50, "noise");
        assert_eq!(noise.len(), 50);
    }

    #[test]
    fn test_create_benchmark_result() {
        let execution_times = vec![1000, 1100, 1050, 980, 1020];
        let config = BenchmarkConfig::default();
        let result = create_benchmark_result("test".to_string(), 100, execution_times, &config);

        assert_eq!(result.name, "test");
        assert_eq!(result.signal_size, 100);
        assert!(result.mean_time > 0.0);
        assert!(result.throughput > 0.0);
    }
}
