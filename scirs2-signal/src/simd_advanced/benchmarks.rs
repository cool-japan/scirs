//! SIMD benchmarking operations
//!
//! This module provides benchmarking functionality to measure and compare
//! the performance of SIMD-optimized operations against their scalar counterparts.
//!
//! # Features
//!
//! - **Performance comparison**: Direct SIMD vs scalar benchmarking
//! - **Throughput measurement**: Operations per second metrics
//! - **Latency analysis**: Single operation timing
//! - **Speedup calculation**: SIMD acceleration factors
//!
//! # Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::benchmark_simd_operations;
//!
//! // Benchmark SIMD operations with 8192-sample signals
//! benchmark_simd_operations(8192)?;
//! # Ok(())
//! # }
//! ```

use super::{simd_autocorrelation, simd_fir_filter, SimdConfig};
use crate::error::SignalResult;
use std::time::Instant;

/// Performance benchmark for SIMD operations
///
/// Benchmarks various SIMD-optimized signal processing operations against their
/// scalar counterparts to measure performance improvements.
///
/// This function creates test signals of the specified length and runs multiple
/// iterations of each operation to get reliable timing measurements.
///
/// # Arguments
///
/// * `signal_length` - Length of test signal to generate
///
/// # Returns
///
/// * `SignalResult<()>` - Ok(()) if benchmarking completes successfully
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::benchmark_simd_operations;
///
/// // Benchmark with different signal sizes
/// benchmark_simd_operations(1024)?;   // Small signal
/// benchmark_simd_operations(8192)?;   // Medium signal
/// benchmark_simd_operations(65536)?;  // Large signal
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn benchmark_simd_operations(signal_length: usize) -> SignalResult<()> {
    let signal: Vec<f64> = (0..signal_length).map(|i| (i as f64 * 0.1).sin()).collect();

    let coeffs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.2, 0.1]; // Simple 5-tap filter
    let mut output = vec![0.0; signal_length];

    let config = SimdConfig::default();

    // Benchmark SIMD FIR filter
    let start = Instant::now();
    for _ in 0..100 {
        simd_fir_filter(&signal, &coeffs, &mut output, &config)?;
    }
    let simd_time = start.elapsed();

    // Benchmark scalar FIR filter
    let config_scalar = SimdConfig {
        force_scalar: true,
        ..Default::default()
    };

    let start = Instant::now();
    for _ in 0..100 {
        simd_fir_filter(&signal, &coeffs, &mut output, &config_scalar)?;
    }
    let scalar_time = start.elapsed();

    println!("FIR Filter Benchmark (length: {}):", signal_length);
    println!("  SIMD time: {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    // Benchmark autocorrelation
    let start = Instant::now();
    for _ in 0..10 {
        simd_autocorrelation(&signal, 100, &config)?;
    }
    let simd_autocorr_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        simd_autocorrelation(&signal, 100, &config_scalar)?;
    }
    let scalar_autocorr_time = start.elapsed();

    println!("Autocorrelation Benchmark:");
    println!("  SIMD time: {:?}", simd_autocorr_time);
    println!("  Scalar time: {:?}", scalar_autocorr_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_autocorr_time.as_secs_f64() / simd_autocorr_time.as_secs_f64()
    );

    Ok(())
}
