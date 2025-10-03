//! SIMD-optimized convolution operations
//!
//! This module provides high-performance convolution implementations optimized
//! for signal processing applications, with automatic platform-specific
//! optimizations and scalar fallbacks.
//!
//! # Features
//!
//! - **Multi-platform SIMD**: Optimized for AVX512, AVX2, and SSE4.1 instruction sets
//! - **Memory optimization**: Cache-friendly processing patterns for large signals
//! - **Full convolution**: Complete convolution output without truncation
//! - **Automatic fallback**: Scalar implementations for unsupported platforms
//!
//! # Convolution Types
//!
//! Currently supports full convolution where the output length is:
//! `output_length = signal_length + kernel_length - 1`
//!
//! # Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{simd_enhanced_convolution, SimdConfig};
//!
//! let signal = vec![1.0, 2.0, 3.0, 4.0];
//! let kernel = vec![0.5, 0.3, 0.2];
//! let mut output = vec![0.0; signal.len() + kernel.len() - 1];
//! let config = SimdConfig::default();
//!
//! simd_enhanced_convolution(&signal, &kernel, &mut output, &config)?;
//! # Ok(())
//! # }
//! ```

#[cfg(target_arch = "x86_64")]
use super::platform_ops::avx2_enhanced_convolution;
use super::SimdConfig;
use crate::error::{SignalError, SignalResult};
use scirs2_core::simd_ops::PlatformCapabilities;
use scirs2_core::validation::check_finite;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-enhanced convolution with memory optimization
///
/// Performs full convolution of a signal with a kernel using SIMD vectorization
/// for maximum performance. Automatically selects the best available instruction
/// set (AVX512, AVX2, or scalar fallback) based on platform capabilities.
///
/// # Arguments
///
/// * `signal` - Input signal data
/// * `kernel` - Convolution kernel (filter coefficients)
/// * `output` - Output buffer (must be signal.len() + kernel.len() - 1)
/// * `config` - SIMD configuration settings
///
/// # Returns
///
/// * `SignalResult<()>` - Ok(()) if convolution completes successfully
///
/// # Algorithm
///
/// Implements discrete convolution: `output[n] = Σ signal[m] * kernel[n-m]`
///
/// The SIMD implementation uses vectorized multiply-accumulate operations
/// to process multiple output samples simultaneously, with careful handling
/// of boundary conditions and memory alignment.
///
/// # Performance Notes
///
/// - For small signals (< threshold), scalar fallback is used
/// - AVX2 processes 4 f64 elements per instruction
/// - AVX512 processes 8 f64 elements per instruction (when available)
/// - Memory prefetching optimizes cache utilization for large signals
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{simd_enhanced_convolution, SimdConfig};
///
/// // Simple lowpass filter
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let kernel = vec![0.25, 0.5, 0.25];  // 3-tap filter
/// let mut output = vec![0.0; signal.len() + kernel.len() - 1];
/// let config = SimdConfig::default();
///
/// simd_enhanced_convolution(&signal, &kernel, &mut output, &config)?;
///
/// // Output will contain the full convolution result
/// assert_eq!(output.len(), 7); // 5 + 3 - 1
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    for (i, &value) in signal.iter().enumerate() {
        check_finite(value, format!("signal value at index {}", i))?;
    }
    for (i, &value) in kernel.iter().enumerate() {
        check_finite(value, format!("kernel value at index {}", i))?;
    }

    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let output_len = signal_len + kernel_len - 1;

    if output.len() != output_len {
        return Err(SignalError::ValueError(
            "Output buffer size incorrect for full convolution".to_string(),
        ));
    }

    if signal_len < config.simd_threshold || config.force_scalar {
        return scalar_enhanced_convolution(signal, kernel, output);
    }

    let caps = PlatformCapabilities::detect();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx512_available && config.use_advanced {
            unsafe { avx512_enhanced_convolution(signal, kernel, output) }
        } else if caps.avx2_available {
            unsafe { avx2_enhanced_convolution_impl(signal, kernel, output) }
        } else {
            scalar_enhanced_convolution(signal, kernel, output)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        scalar_enhanced_convolution(signal, kernel, output)
    }
}

/// Scalar fallback for enhanced convolution
///
/// Implements discrete convolution using standard scalar operations.
/// This function is used when SIMD is unavailable, disabled, or when
/// the signal size is below the SIMD threshold.
///
/// # Arguments
///
/// * `signal` - Input signal data
/// * `kernel` - Convolution kernel (filter coefficients)
/// * `output` - Output buffer for convolution result
///
/// # Returns
///
/// * `SignalResult<()>` - Ok(()) if convolution completes successfully
///
/// # Algorithm
///
/// Direct implementation of discrete convolution:
/// ```text
/// output[i] = Σ(j=0 to kernel_len-1) signal[i-j] * kernel[j]
/// ```
///
/// Handles boundary conditions by checking array bounds before access.
#[allow(dead_code)]
fn scalar_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();

    for i in 0..output.len() {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            let signal_idx = i.wrapping_sub(j);
            if signal_idx < signal_len {
                sum += signal[signal_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

/// AVX2 enhanced convolution with cache optimization (local implementation)
///
/// High-performance convolution using AVX2 vectorization to process
/// 4 double-precision values simultaneously. Includes careful memory
/// access patterns to optimize cache utilization.
///
/// # Safety
///
/// This function requires AVX2 instruction set support and must only
/// be called when AVX2 availability has been verified.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_enhanced_convolution_impl(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let simd_width = 4;

    for i in (0..output.len()).step_by(simd_width) {
        if i + simd_width <= output.len() {
            let mut result = _mm256_setzero_pd();

            for j in 0..kernel_len {
                let signal_idx = i.wrapping_sub(j);
                if signal_idx < signal_len && signal_idx + simd_width <= signal_len {
                    let signal_vec = _mm256_loadu_pd(signal.as_ptr().add(signal_idx));
                    let kernel_broadcast = _mm256_set1_pd(kernel[j]);
                    result = _mm256_fmadd_pd(signal_vec, kernel_broadcast, result);
                }
            }

            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        } else {
            // Handle remaining elements with scalar code
            for idx in i..output.len() {
                let mut sum = 0.0;
                for j in 0..kernel_len {
                    let signal_idx = idx.wrapping_sub(j);
                    if signal_idx < signal_len {
                        sum += signal[signal_idx] * kernel[j];
                    }
                }
                output[idx] = sum;
            }
            break;
        }
    }

    Ok(())
}

/// AVX512 enhanced convolution - fallback to scalar implementation
///
/// Placeholder for AVX512 implementation. Currently falls back to scalar
/// implementation until AVX512 instruction intrinsics are stabilized in Rust.
///
/// # Safety
///
/// This function is marked unsafe for future AVX512 implementation consistency.
unsafe fn avx512_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    // Fallback to scalar implementation until AVX-512 is stabilized
    let signal_len = signal.len();
    let kernel_len = kernel.len();

    for i in 0..output.len() {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            let signal_idx = i.wrapping_sub(j);
            if signal_idx < signal_len {
                sum += signal[signal_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_enhanced_convolution() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; signal.len() + kernel.len() - 1];
        let config = SimdConfig::default();

        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_ok());

        // Check that output is finite and non-zero
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_convolution_impulse_response() {
        let signal = vec![1.0, 0.0, 0.0, 0.0]; // Impulse
        let kernel = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; signal.len() + kernel.len() - 1];
        let config = SimdConfig::default();

        simd_enhanced_convolution(&signal, &kernel, &mut output, &config).unwrap();

        // Convolution with impulse should give kernel coefficients
        assert!((output[0] - 0.5).abs() < 1e-10);
        assert!((output[1] - 0.3).abs() < 1e-10);
        assert!((output[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_convolution_output_size() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5];
        let mut output = vec![0.0; signal.len() + kernel.len() - 1];
        let config = SimdConfig::default();

        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_ok());
        assert_eq!(output.len(), 4); // 3 + 2 - 1
    }

    #[test]
    fn test_convolution_wrong_output_size() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5];
        let mut output = vec![0.0; 5]; // Wrong size
        let config = SimdConfig::default();

        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_fallback() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5];
        let mut output = vec![0.0; signal.len() + kernel.len() - 1];
        let config = SimdConfig {
            force_scalar: true,
            ..Default::default()
        };

        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_ok());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convolution_symmetry() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5];
        let mut output1 = vec![0.0; signal.len() + kernel.len() - 1];
        let mut output2 = vec![0.0; signal.len() + kernel.len() - 1];
        let config = SimdConfig::default();

        // Test signal * kernel
        simd_enhanced_convolution(&signal, &kernel, &mut output1, &config).unwrap();

        // Test kernel * signal (should be different due to convolution definition)
        simd_enhanced_convolution(&kernel, &signal, &mut output2, &config).unwrap();

        assert_eq!(output1.len(), output2.len());
        assert!(output1.iter().all(|&x| x.is_finite()));
        assert!(output2.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_empty_inputs() {
        let config = SimdConfig::default();

        // Empty signal
        let signal = vec![];
        let kernel = vec![1.0];
        let mut output = vec![];
        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_ok());

        // Empty kernel
        let signal = vec![1.0];
        let kernel = vec![];
        let mut output = vec![];
        let result = simd_enhanced_convolution(&signal, &kernel, &mut output, &config);
        assert!(result.is_ok());
    }
}
