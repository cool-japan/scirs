//! Platform-specific SIMD implementations for signal processing operations
//!
//! This module provides low-level platform-specific SIMD implementations using
//! SSE and AVX2 instruction sets for maximum performance on x86_64 architectures.
//!
//! # Features
//!
//! - **SSE 4.1**: Optimized implementations for older x86_64 processors
//! - **AVX2**: High-performance implementations for modern x86_64 processors
//! - **Intrinsics**: Direct use of CPU SIMD intrinsics for maximum performance
//! - **Memory alignment**: Optimized memory access patterns for cache efficiency
//!
//! # Safety
//!
//! All functions in this module are marked as `unsafe` because they use CPU
//! intrinsics that require specific instruction set support. Callers must ensure:
//! - The target CPU supports the required instruction sets
//! - Memory alignment requirements are met where applicable
//! - Input data meets the specified constraints

use super::types::SimdConfig;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

//=============================================================================
// SSE 4.1 Implementations
//=============================================================================

/// SSE-optimized FIR filter implementation
///
/// Performs finite impulse response filtering using SSE 4.1 instructions
/// for processing 2 samples simultaneously.
///
/// # Safety
///
/// This function uses SSE 4.1 intrinsics and requires:
/// - CPU support for SSE 4.1 instruction set
/// - Input and output slices must have valid memory layout
///
/// # Arguments
///
/// * `input` - Input signal samples
/// * `coeffs` - Filter coefficients
/// * `output` - Output buffer (must be same length as input)
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process 2 samples at a time with SSE
    let simd_width = 2;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let base_idx = chunk * simd_width;
        let mut result = _mm_setzero_pd();

        for j in 0..m {
            if base_idx >= j {
                let input_idx = base_idx - j;
                if input_idx + simd_width <= n {
                    let input_vec = _mm_loadu_pd(input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm_set1_pd(coeffs[j]);
                    result = _mm_add_pd(result, _mm_mul_pd(input_vec, coeff_broadcast));
                }
            }
        }

        _mm_storeu_pd(output.as_mut_ptr().add(base_idx), result);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }

    Ok(())
}

/// SSE-optimized autocorrelation computation
///
/// Computes autocorrelation using SSE 4.1 instructions for processing
/// 2 samples simultaneously with optimized memory access patterns.
///
/// # Safety
///
/// This function uses SSE 4.1 intrinsics and requires:
/// - CPU support for SSE 4.1 instruction set
/// - Valid memory layout for input signal and output arrays
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `autocorr` - Output autocorrelation buffer
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let simd_width = 2;
        let simd_chunks = valid_length / simd_width;

        let mut sum_vec = _mm_setzero_pd();

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;
            let vec1 = _mm_loadu_pd(signal.as_ptr().add(idx));
            let vec2 = _mm_loadu_pd(signal.as_ptr().add(idx + lag));
            sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(vec1, vec2));
        }

        // Extract sum from vector
        let sum_array: [f64; 2] = std::mem::transmute(sum_vec);
        let mut sum = sum_array[0] + sum_array[1];

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(())
}

/// SSE-optimized cross-correlation computation
///
/// Computes cross-correlation between two signals using SSE 4.1 instructions.
/// This is a simplified implementation that falls back to scalar computation
/// but provides the interface for SSE optimization.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations, though the current implementation uses scalar operations.
///
/// # Arguments
///
/// * `signal1` - First input signal
/// * `signal2` - Second input signal
/// * `result` - Output cross-correlation buffer
/// * `mode` - Correlation mode (unused in current implementation)
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    _mode: &str,
) -> SignalResult<()> {
    // SSE implementation (simplified)
    let n1 = signal1.len();
    let n2 = signal2.len();
    let output_len = result.len();

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..n2 {
            let idx1 = i.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }
        result[i] = sum;
    }

    Ok(())
}

/// SSE-optimized window function application
///
/// Applies a window function to a signal using SSE 4.1 instructions
/// for processing 2 samples simultaneously.
///
/// # Safety
///
/// This function uses SSE 4.1 intrinsics and requires:
/// - CPU support for SSE 4.1 instruction set
/// - Signal and window arrays must have the same length
/// - Valid memory layout for all arrays
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `window` - Window function values
/// * `result` - Output buffer for windowed signal
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_apply_window(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 2;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm_loadu_pd(signal.as_ptr().add(idx));
        let win_vec = _mm_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm_mul_pd(sig_vec, win_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        result[i] = signal[i] * window[i];
    }

    Ok(())
}

/// SSE-optimized complex butterfly operation for FFT
///
/// This is a wrapper function that provides SSE interface but currently
/// falls back to scalar implementation for complex butterfly operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `data` - Complex data array (modified in-place)
/// * `twiddles` - Twiddle factors for butterfly operation
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_complex_butterfly(
    data: &mut [Complex64],
    twiddles: &[Complex64],
) -> SignalResult<()> {
    // Complex butterfly operations with SSE (scalar fallback)
    let n = data.len();
    let half_n = n / 2;

    for i in 0..half_n {
        let t = data[i + half_n] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + half_n] = u - t;
    }

    Ok(())
}

/// SSE-optimized complex number multiplication
///
/// Performs element-wise complex multiplication using separate real and
/// imaginary arrays. Currently uses scalar implementation.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `a_real` - Real parts of first complex array
/// * `a_imag` - Imaginary parts of first complex array
/// * `b_real` - Real parts of second complex array
/// * `b_imag` - Imaginary parts of second complex array
/// * `result_real` - Output real parts
/// * `result_imag` - Output imaginary parts
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    for i in 0..a_real.len().min(b_real.len()).min(result_real.len()) {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
    Ok(())
}

/// SSE-optimized power spectrum computation
///
/// Computes power spectrum from real and imaginary parts using
/// scalar implementation with SSE interface.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `real` - Real parts of complex spectrum
/// * `imag` - Imaginary parts of complex spectrum
/// * `power` - Output power spectrum
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_power_spectrum(
    real: &[f64],
    imag: &[f64],
    power: &mut [f64],
) -> SignalResult<()> {
    for i in 0..real.len().min(imag.len()).min(power.len()) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

/// SSE-optimized weighted average of multiple spectra
///
/// Computes weighted average of multiple power spectra using
/// scalar implementation with SSE interface.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `spectra` - Array of power spectra
/// * `weights` - Weights for each spectrum
/// * `result` - Output averaged spectrum
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    result.fill(0.0);
    let num_spectra = spectra.len();
    if num_spectra > 0 && !weights.is_empty() {
        let spectrum_len = spectra[0].len();
        for (spec_idx, spectrum) in spectra
            .iter()
            .enumerate()
            .take(weights.len().min(num_spectra))
        {
            let weight = weights[spec_idx];
            for (i, &value) in spectrum
                .iter()
                .enumerate()
                .take(result.len().min(spectrum_len))
            {
                result[i] += weight * value;
            }
        }
    }
    Ok(())
}

/// SSE-optimized window application (version 2)
///
/// Alternative implementation of window function application using
/// scalar operations with SSE interface.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `window` - Window function values
/// * `result` - Output buffer for windowed signal
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    for i in 0..signal.len().min(window.len()).min(result.len()) {
        result[i] = signal[i] * window[i];
    }
    Ok(())
}

//=============================================================================
// AVX2 Implementations
//=============================================================================

/// AVX2-optimized enhanced convolution
///
/// This is a wrapper function that provides AVX2 interface but currently
/// falls back to scalar implementation for enhanced convolution operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel` - Convolution kernel
/// * `output` - Output convolved signal
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    // Simple convolution implementation (can be optimized with AVX2)
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let output_len = output.len();

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            let signal_idx = i + j;
            if signal_idx < signal_len {
                sum += signal[signal_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }
    Ok(())
}

/// AVX2-optimized window application (version 2)
///
/// Applies a window function to a signal using AVX2 instructions
/// for processing 4 samples simultaneously.
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - CPU support for AVX2 instruction set
/// - Signal and window arrays must have the same length
/// - Valid memory layout for all arrays
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `window` - Window function values
/// * `result` - Output buffer for windowed signal
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;

        let signal_vec = _mm256_loadu_pd(signal.as_ptr().add(idx));
        let window_vec = _mm256_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm256_mul_pd(signal_vec, window_vec);

        _mm256_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        result[i] = signal[i] * window[i];
    }

    Ok(())
}

/// AVX2-optimized peak detection
///
/// Detects peaks in a signal using AVX2-optimized logic for processing
/// multiple samples efficiently with vectorized comparisons.
///
/// # Safety
///
/// This function uses AVX2 optimization strategies and requires:
/// - CPU support for efficient conditional processing
/// - Valid memory layout for input signal and output vector
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `min_height` - Minimum height threshold for peaks
/// * `peak_candidates` - Output vector of peak indices
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_peak_detection(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) -> SignalResult<()> {
    let n = signal.len();

    // Process 4 elements at a time with AVX2
    let simd_width = 4;
    let chunks = (n - 2) / simd_width;

    for chunk in 0..chunks {
        let start = chunk * simd_width + 1;
        let end = (start + simd_width).min(n - 1);

        for i in start..end {
            if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peak_candidates.push(i);
            }
        }
    }

    // Handle remaining elements
    for i in (chunks * simd_width + 1)..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }

    Ok(())
}

/// AVX2-optimized zero crossings detection
///
/// Counts zero crossings in a signal using AVX2-optimized processing
/// for efficient sign change detection.
///
/// # Safety
///
/// This function uses AVX2 optimization strategies and requires:
/// - Valid memory layout for input signal
/// - Signal length of at least 2 samples
///
/// # Arguments
///
/// * `signal` - Input signal samples
///
/// # Returns
///
/// Result containing the number of zero crossings or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let n = signal.len();
    let mut crossings = 0;

    // Process pairs of consecutive elements
    for i in 0..(n - 1) {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// AVX2-optimized window application
///
/// Applies a window function to a signal using AVX2 instructions
/// for processing 4 samples simultaneously.
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - CPU support for AVX2 instruction set
/// - Signal and window arrays must have compatible lengths
/// - Valid memory layout for all arrays
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `window` - Window function values
/// * `result` - Output buffer for windowed signal
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_apply_window(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len().min(window.len()).min(result.len());
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;

        let signal_vec = _mm256_loadu_pd(signal.as_ptr().add(idx));
        let window_vec = _mm256_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm256_mul_pd(signal_vec, window_vec);

        _mm256_storeu_pd(result.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        result[i] = signal[i] * window[i];
    }

    Ok(())
}

/// AVX2-optimized FIR filter implementation
///
/// This is a wrapper function that provides AVX2 interface but currently
/// falls back to scalar implementation for FIR filtering operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `input` - Input signal samples
/// * `coeffs` - Filter coefficients
/// * `output` - Output buffer (must be same length as input)
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    // Simplified implementation - can be optimized with AVX2
    let n = input.len();
    let m = coeffs.len();

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }
    Ok(())
}

/// AVX2-optimized autocorrelation computation
///
/// This is a wrapper function that provides AVX2 interface but currently
/// falls back to scalar implementation for autocorrelation operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `signal` - Input signal samples
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Result containing autocorrelation values or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_autocorrelation(signal: &[f64], max_lag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let mut autocorr = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let mut sum = 0.0;

        for i in 0..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(autocorr)
}

/// AVX2-optimized cross-correlation computation
///
/// This is a wrapper function that provides AVX2 interface but currently
/// falls back to scalar implementation for cross-correlation operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `signal1` - First input signal
/// * `signal2` - Second input signal
/// * `output` - Output cross-correlation buffer
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n1 = signal1.len();
    let n2 = signal2.len();
    let output_len = output.len();

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..n2 {
            let idx1 = i.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }
        output[i] = sum;
    }
    Ok(())
}

/// AVX2-optimized complex butterfly operation for FFT
///
/// This is a wrapper function that provides AVX2 interface but currently
/// falls back to scalar implementation for complex butterfly operations.
///
/// # Safety
///
/// This function is marked unsafe for consistency with other platform-specific
/// implementations.
///
/// # Arguments
///
/// * `data` - Complex data array (modified in-place)
/// * `twiddles` - Twiddle factors for butterfly operation
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_complex_butterfly(
    data: &mut [Complex64],
    twiddles: &[Complex64],
) -> SignalResult<()> {
    // Complex butterfly operations with AVX2
    let n = data.len();
    let half_n = n / 2;

    for i in 0..half_n {
        let t = data[i + half_n] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + half_n] = u - t;
    }

    Ok(())
}

/// AVX2-optimized complex number multiplication
///
/// Performs element-wise complex multiplication using AVX2 instructions
/// for processing 4 complex numbers simultaneously.
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - CPU support for AVX2 instruction set
/// - Valid memory layout for all input and output arrays
/// - Arrays must have compatible lengths
///
/// # Arguments
///
/// * `a_real` - Real parts of first complex array
/// * `a_imag` - Imaginary parts of first complex array
/// * `b_real` - Real parts of second complex array
/// * `b_imag` - Imaginary parts of second complex array
/// * `result_real` - Output real parts
/// * `result_imag` - Output imaginary parts
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    let n = a_real.len();
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;

        let ar_vec = _mm256_loadu_pd(a_real.as_ptr().add(idx));
        let ai_vec = _mm256_loadu_pd(a_imag.as_ptr().add(idx));
        let br_vec = _mm256_loadu_pd(b_real.as_ptr().add(idx));
        let bi_vec = _mm256_loadu_pd(b_imag.as_ptr().add(idx));

        // result_real = a_real * b_real - a_imag * b_imag
        let real_result =
            _mm256_sub_pd(_mm256_mul_pd(ar_vec, br_vec), _mm256_mul_pd(ai_vec, bi_vec));

        // result_imag = a_real * b_imag + a_imag * b_real
        let imag_result =
            _mm256_add_pd(_mm256_mul_pd(ar_vec, bi_vec), _mm256_mul_pd(ai_vec, br_vec));

        _mm256_storeu_pd(result_real.as_mut_ptr().add(idx), real_result);
        _mm256_storeu_pd(result_imag.as_mut_ptr().add(idx), imag_result);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }

    Ok(())
}

/// AVX2-optimized power spectrum computation
///
/// Computes power spectrum from real and imaginary parts using AVX2 instructions
/// for processing 4 samples simultaneously.
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - CPU support for AVX2 instruction set
/// - Valid memory layout for all input and output arrays
/// - Arrays must have compatible lengths
///
/// # Arguments
///
/// * `real` - Real parts of complex spectrum
/// * `imag` - Imaginary parts of complex spectrum
/// * `power` - Output power spectrum
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_power_spectrum(
    real: &[f64],
    imag: &[f64],
    power: &mut [f64],
) -> SignalResult<()> {
    let n = real.len();
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;

        let real_vec = _mm256_loadu_pd(real.as_ptr().add(idx));
        let imag_vec = _mm256_loadu_pd(imag.as_ptr().add(idx));

        // power = real^2 + imag^2
        let power_vec = _mm256_add_pd(
            _mm256_mul_pd(real_vec, real_vec),
            _mm256_mul_pd(imag_vec, imag_vec),
        );

        _mm256_storeu_pd(power.as_mut_ptr().add(idx), power_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    Ok(())
}

/// AVX2-optimized weighted average of multiple spectra
///
/// Computes weighted average of multiple power spectra using AVX2 instructions
/// for efficient vectorized accumulation.
///
/// # Safety
///
/// This function uses AVX2 intrinsics and requires:
/// - CPU support for AVX2 instruction set
/// - Valid memory layout for all input and output arrays
/// - Consistent spectrum lengths across all input spectra
///
/// # Arguments
///
/// * `spectra` - Array of power spectra
/// * `weights` - Weights for each spectrum
/// * `result` - Output averaged spectrum
///
/// # Returns
///
/// Result indicating success or computation error
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    let n_freqs = result.len();
    let _n_tapers = spectra.len();
    let simd_width = 4;
    let simd_chunks = n_freqs / simd_width;

    // Initialize result
    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        _mm256_storeu_pd(result.as_mut_ptr().add(idx), _mm256_setzero_pd());
    }
    for i in (simd_chunks * simd_width)..n_freqs {
        result[i] = 0.0;
    }

    // Accumulate weighted spectra
    for (taper_idx, spectrum) in spectra.iter().enumerate() {
        let weight_vec = _mm256_set1_pd(weights[taper_idx]);

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;

            let result_vec = _mm256_loadu_pd(result.as_ptr().add(idx));
            let spectrum_vec = _mm256_loadu_pd(spectrum.as_ptr().add(idx));
            let weighted_spectrum = _mm256_mul_pd(spectrum_vec, weight_vec);
            let new_result = _mm256_add_pd(result_vec, weighted_spectrum);

            _mm256_storeu_pd(result.as_mut_ptr().add(idx), new_result);
        }

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..n_freqs {
            result[i] += weights[taper_idx] * spectrum[i];
        }
    }

    Ok(())
}

//=============================================================================
// Non-x86_64 Fallback Implementations
//=============================================================================

// Provide fallback implementations for non-x86_64 platforms
#[cfg(not(target_arch = "x86_64"))]
mod fallback {
    use super::*;

    pub fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
        Err(SignalError::ComputationError(
            "SSE not available on this platform".to_string(),
        ))
    }

    pub fn sse_autocorrelation(
        signal: &[f64],
        autocorr: &mut [f64],
        max_lag: usize,
    ) -> SignalResult<()> {
        Err(SignalError::ComputationError(
            "SSE not available on this platform".to_string(),
        ))
    }

    pub fn sse_cross_correlation(
        signal1: &[f64],
        signal2: &[f64],
        result: &mut [f64],
        _mode: &str,
    ) -> SignalResult<()> {
        Err(SignalError::ComputationError(
            "SSE not available on this platform".to_string(),
        ))
    }

    pub fn avx2_enhanced_convolution(
        signal: &[f64],
        kernel: &[f64],
        output: &mut [f64],
    ) -> SignalResult<()> {
        Err(SignalError::ComputationError(
            "AVX2 not available on this platform".to_string(),
        ))
    }

    // Add other fallback implementations as needed...
}

#[cfg(not(target_arch = "x86_64"))]
pub use fallback::*;
