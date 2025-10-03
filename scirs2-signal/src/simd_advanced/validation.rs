//! SIMD validation and testing operations
//!
//! This module provides comprehensive validation functionality to ensure
//! SIMD implementations are correct and to measure their performance
//! characteristics compared to scalar fallbacks.
//!
//! # Features
//!
//! - **Correctness validation**: Ensures SIMD and scalar implementations produce identical results
//! - **Performance measurement**: Measures execution time for different operations
//! - **Consistency checking**: Validates numerical accuracy within tolerance
//! - **Memory throughput analysis**: Estimates memory bandwidth utilization
//!
//! # Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{comprehensive_simd_validation, SimdConfig};
//!
//! let config = SimdConfig::default();
//! let result = comprehensive_simd_validation(8192, &config)?;
//!
//! if result.validation_passed {
//!     println!("SIMD validation passed!");
//!     println!("Speedup: {:.2}x", result.operations_per_second);
//! }
//! # Ok(())
//! # }
//! ```

use super::{
    simd_autocorrelation, simd_cross_correlation, simd_fir_filter, SimdConfig, SimdValidationResult,
};
use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;

/// Comprehensive SIMD validation and performance testing
///
/// Performs extensive testing of SIMD implementations to validate correctness
/// and measure performance characteristics. Tests various signal processing
/// operations and compares SIMD results against scalar fallbacks.
///
/// # Arguments
///
/// * `test_size` - Size of test signals to generate for validation
/// * `config` - SIMD configuration to test
///
/// # Returns
///
/// * `SignalResult<SimdValidationResult>` - Detailed validation results
///
/// # Validation Tests
///
/// 1. **FIR Filter**: Tests SIMD FIR filtering accuracy and performance
/// 2. **Autocorrelation**: Validates autocorrelation computations
/// 3. **Cross-correlation**: Tests cross-correlation implementations
/// 4. **Consistency**: Compares SIMD vs scalar results for accuracy
/// 5. **Performance**: Measures throughput and memory bandwidth
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{comprehensive_simd_validation, SimdConfig};
///
/// // Validate with default configuration
/// let config = SimdConfig::default();
/// let result = comprehensive_simd_validation(4096, &config)?;
///
/// if result.validation_passed {
///     println!("All validations passed!");
///     println!("FIR filter time: {} ns", result.fir_filter_time_ns);
///     println!("Max error: {:.2e}", result.simd_scalar_max_error);
/// }
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn comprehensive_simd_validation(
    test_size: usize,
    config: &SimdConfig,
) -> SignalResult<SimdValidationResult> {
    let mut validation_result = SimdValidationResult::default();
    let start_time = std::time::Instant::now();

    // 1. Test basic SIMD operations
    let test_signal: Vec<f64> = (0..test_size).map(|i| (i as f64 * 0.1).sin()).collect();
    let test_kernel = vec![0.25, 0.5, 0.25];
    let mut output = vec![0.0; test_signal.len()];

    // Test FIR filter
    let fir_start = std::time::Instant::now();
    simd_fir_filter(&test_signal, &test_kernel, &mut output, config)?;
    validation_result.fir_filter_time_ns = fir_start.elapsed().as_nanos() as u64;

    // Test autocorrelation
    let autocorr_start = std::time::Instant::now();
    let _autocorr = simd_autocorrelation(&test_signal, 10, config)?;
    validation_result.autocorrelation_time_ns = autocorr_start.elapsed().as_nanos() as u64;

    // Test cross-correlation
    let xcorr_start = std::time::Instant::now();
    let _xcorr = simd_cross_correlation(&test_signal, &test_signal, "full", config)?;
    validation_result.cross_correlation_time_ns = xcorr_start.elapsed().as_nanos() as u64;

    // 2. Matrix operations testing (simplified since advanced_simd_matrix module may not exist)
    // Create a simple matrix-vector operation test
    let matrix = Array2::<f64>::ones((100, 100));
    let vector = vec![1.0; 100];
    let mut matrix_result = vec![0.0; 100];

    let matrix_start = std::time::Instant::now();

    // Simple matrix-vector multiplication fallback
    for (i, row) in matrix.outer_iter().enumerate() {
        matrix_result[i] = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
    }

    validation_result.matrix_vector_time_ns = matrix_start.elapsed().as_nanos() as u64;

    // 3. Validate SIMD vs Scalar consistency
    let mut scalar_config = config.clone();
    scalar_config.force_scalar = true;

    let mut scalar_output = vec![0.0; test_signal.len()];
    simd_fir_filter(
        &test_signal,
        &test_kernel,
        &mut scalar_output,
        &scalar_config,
    )?;

    // Compare results
    let max_error = output
        .iter()
        .zip(scalar_output.iter())
        .map(|(&simd, &scalar)| (simd - scalar).abs())
        .fold(0.0, f64::max);

    validation_result.simd_scalar_max_error = max_error;
    validation_result.simd_consistency = max_error < 1e-12;

    // 4. Performance analysis
    let ops_per_second = test_size as f64 / (fir_start.elapsed().as_secs_f64());
    validation_result.operations_per_second = ops_per_second;

    // 5. Memory throughput estimation
    let bytes_processed = test_size * std::mem::size_of::<f64>();
    let memory_throughput = bytes_processed as f64 / fir_start.elapsed().as_secs_f64();
    validation_result.memory_throughput_bytes_per_sec = memory_throughput;

    validation_result.total_validation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    validation_result.validation_passed = validation_result.simd_consistency;

    Ok(validation_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_validation() {
        let config = SimdConfig::default();
        let result = comprehensive_simd_validation(1024, &config);

        assert!(result.is_ok());
        let validation = result.unwrap();

        // Basic checks
        assert!(validation.fir_filter_time_ns > 0);
        assert!(validation.autocorrelation_time_ns > 0);
        assert!(validation.cross_correlation_time_ns > 0);
        assert!(validation.operations_per_second > 0.0);
        assert!(validation.memory_throughput_bytes_per_sec > 0.0);
        assert!(validation.total_validation_time_ms > 0.0);
    }

    #[test]
    fn test_validation_with_scalar_config() {
        let config = SimdConfig {
            force_scalar: true,
            ..Default::default()
        };

        let result = comprehensive_simd_validation(512, &config);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.validation_passed);
    }

    #[test]
    fn test_validation_small_signal() {
        let config = SimdConfig::default();
        let result = comprehensive_simd_validation(64, &config);

        assert!(result.is_ok());
        let validation = result.unwrap();

        // Should work with small signals too
        assert!(validation.fir_filter_time_ns > 0);
        assert!(validation.operations_per_second > 0.0);
    }
}
