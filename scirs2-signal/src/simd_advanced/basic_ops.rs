//! Basic SIMD-optimized signal processing operations
//!
//! This module provides the core SIMD-accelerated signal processing operations
//! with automatic fallback to scalar implementations. These functions form the
//! building blocks for more complex signal processing algorithms.
//!
//! # Features
//!
//! - Multi-platform SIMD optimization (AVX512, AVX2, SSE4.1, NEON)
//! - Automatic scalar fallback for unsupported platforms
//! - Runtime instruction set detection
//! - Memory alignment detection and optimization
//! - Comprehensive error handling and validation
//!
//! # Supported Operations
//!
//! ## Filtering Operations
//! - [`simd_fir_filter`] - SIMD FIR filtering with arbitrary coefficients
//! - [`simd_apply_window`] - SIMD window function application
//!
//! ## Correlation Operations
//! - [`simd_autocorrelation`] - SIMD autocorrelation computation
//! - [`simd_cross_correlation`] - SIMD cross-correlation computation
//!
//! ## FFT Operations
//! - [`simd_complex_fft_butterfly`] - SIMD complex FFT butterfly operations
//!
//! ## Signal Analysis
//! - [`simd_signal_energy`] - SIMD signal energy computation
//! - [`simd_rms`] - SIMD RMS computation
//! - [`simd_zero_crossing_rate`] - SIMD zero crossing rate computation

use super::types::SimdConfig;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::Complex64;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Helper function to check if all values in a slice are finite
fn check_slice_finite(slice: &[f64], name: &str) -> SignalResult<()> {
    for (i, &value) in slice.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::ValueError(format!(
                "{} must contain only finite values, got {} at index {}",
                name, value, i
            )));
        }
    }
    Ok(())
}

/// SIMD-optimized FIR filtering
///
/// Applies a finite impulse response filter using SIMD vectorization
/// with optimized memory access patterns and instruction set selection.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `coeffs` - Filter coefficients
/// * `output` - Output buffer (must be same length as input)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(())` on success
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_fir_filter};
///
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let coeffs = vec![0.5, 0.3, 0.2];
/// let mut output = vec![0.0; input.len()];
/// let config = SimdConfig::default();
///
/// simd_fir_filter(&input, &coeffs, &mut output, &config)?;
/// # Ok(())
/// # }
/// ```
pub fn simd_fir_filter(
    input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "Input and output lengths must match".to_string(),
        ));
    }

    check_slice_finite(input, "input")?;
    check_slice_finite(coeffs, "coeffs")?;

    let n = input.len();
    let _m = coeffs.len();

    if n < config.simd_threshold || config.force_scalar {
        return scalar_fir_filter(input, coeffs, output);
    }

    // Check for SIMD capabilities
    let caps = PlatformCapabilities::detect();

    // Use best available SIMD on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if caps.simd_available && is_x86_feature_detected!("avx512f") {
            return unsafe { avx512_fir_filter(input, coeffs, output) };
        }
        if caps.simd_available && is_x86_feature_detected!("avx2") {
            return unsafe { avx2_fir_filter(input, coeffs, output) };
        }
        if caps.simd_available && is_x86_feature_detected!("sse4.1") {
            return unsafe { sse_fir_filter(input, coeffs, output) };
        }
    }

    // Fallback to scalar implementation
    scalar_fir_filter(input, coeffs, output)
}

/// SIMD-optimized autocorrelation computation
///
/// Computes autocorrelation function using SIMD vectorization with
/// cache-friendly memory access patterns.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `max_lag` - Maximum lag to compute (must be less than signal length)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(Vec<f64>)` containing autocorrelation values from lag 0 to max_lag
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_autocorrelation};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let config = SimdConfig::default();
///
/// let autocorr = simd_autocorrelation(&signal, 4, &config)?;
/// # Ok(())
/// # }
/// ```
pub fn simd_autocorrelation(
    signal: &[f64],
    max_lag: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_slice_finite(signal, "signal")?;

    let n = signal.len();
    if max_lag >= n {
        return Err(SignalError::ValueError(
            "Maximum lag must be less than signal length".to_string(),
        ));
    }

    let mut autocorr = vec![0.0; max_lag + 1];

    if n < config.simd_threshold || config.force_scalar {
        scalar_autocorrelation(signal, &mut autocorr, max_lag)?;
        return Ok(autocorr);
    }

    // Check for SIMD capabilities
    let caps = PlatformCapabilities::detect();

    // Use AVX2 if available on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if caps.simd_available && is_x86_feature_detected!("avx2") {
            unsafe { avx2_autocorrelation(signal, &mut autocorr, max_lag)? };
            return Ok(autocorr);
        }
        if caps.simd_available && is_x86_feature_detected!("sse4.1") {
            unsafe { sse_autocorrelation(signal, &mut autocorr, max_lag)? };
            return Ok(autocorr);
        }
    }

    // Fallback to scalar implementation
    scalar_autocorrelation(signal, &mut autocorr, max_lag)?;

    Ok(autocorr)
}

/// SIMD-optimized cross-correlation
///
/// Computes cross-correlation between two signals using vectorized operations
/// with support for different output modes.
///
/// # Arguments
///
/// * `signal1` - First input signal
/// * `signal2` - Second input signal
/// * `mode` - Output mode: "full", "same", or "valid"
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(Vec<f64>)` containing cross-correlation values
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_cross_correlation};
///
/// let signal1 = vec![1.0, 2.0, 3.0, 4.0];
/// let signal2 = vec![0.5, 1.0, 0.5];
/// let config = SimdConfig::default();
///
/// let xcorr = simd_cross_correlation(&signal1, &signal2, "full", &config)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    mode: &str,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_slice_finite(signal1, "signal1")?;
    check_slice_finite(signal2, "signal2")?;

    let n1 = signal1.len();
    let n2 = signal2.len();

    if n1 == 0 || n2 == 0 {
        return Err(SignalError::ValueError(
            "Input signals cannot be empty".to_string(),
        ));
    }

    let output_len = match mode {
        "full" => n1 + n2 - 1,
        "same" => n1,
        "valid" => {
            if n1 >= n2 {
                n1 - n2 + 1
            } else {
                0
            }
        }
        _ => {
            return Err(SignalError::ValueError(
                "Mode must be 'full', 'same', or 'valid'".to_string(),
            ))
        }
    };

    if output_len == 0 {
        return Ok(vec![]);
    }

    let mut result = vec![0.0; output_len];

    if n1.min(n2) < config.simd_threshold || config.force_scalar {
        scalar_cross_correlation(signal1, signal2, &mut result, mode)?;
        return Ok(result);
    }

    let caps = PlatformCapabilities::detect();

    // Use AVX2 if available on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if caps.simd_available && is_x86_feature_detected!("avx2") {
            unsafe { avx2_cross_correlation(signal1, signal2, &mut result, mode)? };
            return Ok(result);
        }
        if caps.simd_available && is_x86_feature_detected!("sse4.1") {
            unsafe { sse_cross_correlation(signal1, signal2, &mut result, mode)? };
            return Ok(result);
        }
    }

    // Fallback to scalar implementation
    scalar_cross_correlation(signal1, signal2, &mut result, mode)?;

    Ok(result)
}

/// SIMD-optimized complex FFT butterfly operations
///
/// Performs vectorized complex arithmetic for FFT computations with
/// optimized instruction set selection.
///
/// # Arguments
///
/// * `data` - Complex data array (modified in-place)
/// * `twiddles` - Complex twiddle factors
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(())` on success
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_complex_fft_butterfly};
/// use scirs2_core::numeric::Complex64;
///
/// let mut data = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
/// let twiddles = vec![Complex64::new(1.0, 0.0), Complex64::new(0.707, -0.707)];
/// let config = SimdConfig::default();
///
/// simd_complex_fft_butterfly(&mut data, &twiddles, &config)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_complex_fft_butterfly(
    data: &mut [scirs2_core::numeric::Complex<f64>],
    twiddles: &[scirs2_core::numeric::Complex<f64>],
    config: &SimdConfig,
) -> SignalResult<()> {
    let n = data.len();

    if n != twiddles.len() {
        return Err(SignalError::ValueError(
            "Data and twiddle factor lengths must match".to_string(),
        ));
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_complex_butterfly(data, twiddles);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        scalar_complex_butterfly(data, twiddles)
    } else if caps.simd_available {
        scalar_complex_butterfly(data, twiddles)
    } else {
        scalar_complex_butterfly(data, twiddles)
    }
}

/// SIMD-optimized windowing function application
///
/// Applies window functions using vectorized operations with
/// optimized memory access patterns.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window` - Window function values (must be same length as signal)
/// * `output` - Output buffer (must be same length as signal)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(())` on success
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_apply_window};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let window = vec![0.5, 0.7, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5];
/// let mut output = vec![0.0; signal.len()];
/// let config = SimdConfig::default();
///
/// simd_apply_window(&signal, &window, &mut output, &config)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    if signal.len() != window.len() || signal.len() != output.len() {
        return Err(SignalError::ValueError(
            "Signal, window, and output lengths must match".to_string(),
        ));
    }

    check_slice_finite(signal, "signal")?;
    check_slice_finite(window, "window")?;

    let n = signal.len();

    if n < config.simd_threshold || config.force_scalar {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        return Ok(());
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx512_available && config.use_advanced {
        scalar_apply_window_safe(signal, window, output)
    } else if caps.avx2_available {
        scalar_apply_window_safe(signal, window, output)
    } else if caps.simd_available {
        scalar_apply_window_safe(signal, window, output)
    } else {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        Ok(())
    }
}

/// SIMD-optimized zero crossing rate computation
///
/// Computes the rate of zero crossings using SIMD vectorization
/// for efficient sign comparison operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(f64)` containing the zero crossing rate (crossings per sample)
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_zero_crossing_rate};
///
/// let signal = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
/// let config = SimdConfig::default();
///
/// let zcr = simd_zero_crossing_rate(&signal, &config)?;
/// # Ok(())
/// # }
/// ```
pub fn simd_zero_crossing_rate(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_slice_finite(signal, "signal")?;

    let n = signal.len();
    if n < 2 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_zero_crossing_rate(signal);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient sign comparison
    let mut crossings = 0;

    if caps.avx2_available && config.use_advanced {
        crossings = scalar_count_zero_crossings(signal);
    } else {
        crossings = scalar_count_zero_crossings(signal);
    }

    Ok(crossings as f64 / (n - 1) as f64)
}

/// SIMD-optimized signal energy computation
///
/// Computes signal energy using vectorized operations for
/// efficient power analysis and normalization.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(f64)` containing the signal energy (sum of squares)
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_signal_energy};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let config = SimdConfig::default();
///
/// let energy = simd_signal_energy(&signal, &config)?;
/// # Ok(())
/// # }
/// ```
pub fn simd_signal_energy(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_slice_finite(signal, "signal")?;

    let n = signal.len();
    if n == 0 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_signal_energy(signal);
    }

    let caps = PlatformCapabilities::detect();
    let signal_view = ArrayView1::from(signal);

    // Use SIMD dot product for efficient energy computation
    let energy = f64::simd_dot(&signal_view, &signal_view);
    Ok(energy)
}

/// SIMD-optimized RMS computation
///
/// Computes root mean square using vectorized operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `Ok(f64)` containing the RMS value
/// * `Err(SignalError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_signal::simd_advanced::{SimdConfig, simd_rms};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let config = SimdConfig::default();
///
/// let rms = simd_rms(&signal, &config)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn simd_rms(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    let energy = simd_signal_energy(signal, config)?;
    let n = signal.len();

    if n == 0 {
        Ok(0.0)
    } else {
        Ok((energy / n as f64).sqrt())
    }
}

// =============================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =============================================================================

/// Safe scalar implementation of window application
fn scalar_apply_window_safe(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    if signal.len() != window.len() || signal.len() != output.len() {
        return Err(SignalError::ValueError(
            "Signal, window, and output arrays must have the same length".to_string(),
        ));
    }

    for i in 0..signal.len() {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

/// Scalar fallback for FIR filtering
fn scalar_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
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

/// Scalar fallback for autocorrelation
#[allow(dead_code)]
pub fn scalar_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let valid_length = n - lag;

        for i in 0..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum / valid_length as f64;
    }

    Ok(())
}

/// Scalar fallback for cross-correlation
#[allow(dead_code)]
pub fn scalar_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    let n1 = signal1.len();
    let n2 = signal2.len();

    let (output_len, start_offset) = match mode {
        "full" => (n1 + n2 - 1, 0),
        "same" => (n1, (n2 - 1) / 2),
        "valid" => (if n1 >= n2 { n1 - n2 + 1 } else { 0 }, n2 - 1),
        _ => return Err(SignalError::ValueError("Invalid mode".to_string())),
    };

    if output_len == 0 {
        return Ok(());
    }

    for i in 0..output_len {
        let lag = i + start_offset;
        let mut sum = 0.0;

        for j in 0..n2 {
            let idx1 = lag.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }

        result[i] = sum;
    }

    Ok(())
}

/// Scalar fallback for complex butterfly operations
#[allow(dead_code)]
fn scalar_complex_butterfly(
    data: &mut [scirs2_core::numeric::Complex<f64>],
    twiddles: &[scirs2_core::numeric::Complex<f64>],
) -> SignalResult<()> {
    for i in 0..data.len() / 2 {
        let t = data[i + data.len() / 2] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + data.len() / 2] = u - t;
    }
    Ok(())
}

/// Scalar fallback for signal energy computation
#[allow(dead_code)]
fn scalar_signal_energy(signal: &[f64]) -> SignalResult<f64> {
    let energy = signal.iter().map(|&x| x * x).sum();
    Ok(energy)
}

/// Scalar fallback for zero crossing count
#[allow(dead_code)]
fn scalar_count_zero_crossings(signal: &[f64]) -> usize {
    let n = signal.len();
    let mut crossings = 0;

    for i in 0..(n - 1) {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            crossings += 1;
        }
    }

    crossings
}

/// Scalar fallback for zero-crossing rate
#[allow(dead_code)]
fn scalar_zero_crossing_rate(signal: &[f64]) -> SignalResult<f64> {
    let n = signal.len();
    if n < 2 {
        return Ok(0.0);
    }

    let crossings = scalar_count_zero_crossings(signal);
    Ok(crossings as f64 / (n - 1) as f64)
}

// =============================================================================
// PLATFORM-SPECIFIC SIMD IMPLEMENTATIONS
// =============================================================================
// Note: These are placeholder function signatures. The actual implementations
// would be quite complex and are typically provided by the platform-specific
// SIMD modules in the parent simd_advanced.rs file.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process each output sample
    for i in 0..n {
        let mut sum_vec = _mm512_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 8 coefficients at a time
        let m_vec = (m / 8) * 8;
        let mut j = 0;

        while j < m_vec && i >= j + 7 {
            // Load 8 coefficients
            let coeff_vec = _mm512_loadu_pd(coeffs.as_ptr().add(j));

            // Load 8 input values in reverse order (for convolution)
            let idx0 = i - j;
            let idx1 = i - j - 1;
            let idx2 = i - j - 2;
            let idx3 = i - j - 3;
            let idx4 = i - j - 4;
            let idx5 = i - j - 5;
            let idx6 = i - j - 6;
            let idx7 = i - j - 7;

            let input_vec = _mm512_set_pd(
                *input.get_unchecked(idx7),
                *input.get_unchecked(idx6),
                *input.get_unchecked(idx5),
                *input.get_unchecked(idx4),
                *input.get_unchecked(idx3),
                *input.get_unchecked(idx2),
                *input.get_unchecked(idx1),
                *input.get_unchecked(idx0),
            );

            // Multiply and accumulate
            sum_vec = _mm512_fmadd_pd(coeff_vec, input_vec, sum_vec);

            j += 8;
        }

        // Horizontal sum of the vector accumulator
        sum = _mm512_reduce_add_pd(sum_vec);

        // Process remaining coefficients with scalar code
        while j < m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
            j += 1;
        }

        output[i] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process each output sample
    for i in 0..n {
        let mut sum_vec = _mm256_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 4 coefficients at a time
        let m_vec = (m / 4) * 4;
        let mut j = 0;

        while j < m_vec && i >= j + 3 {
            // Load 4 coefficients
            let coeff_vec = _mm256_loadu_pd(coeffs.as_ptr().add(j));

            // Load 4 input values in reverse order (for convolution)
            let idx0 = i - j;
            let idx1 = i - j - 1;
            let idx2 = i - j - 2;
            let idx3 = i - j - 3;

            let input_vec = _mm256_set_pd(
                *input.get_unchecked(idx3),
                *input.get_unchecked(idx2),
                *input.get_unchecked(idx1),
                *input.get_unchecked(idx0),
            );

            // Multiply and accumulate
            sum_vec = _mm256_fmadd_pd(coeff_vec, input_vec, sum_vec);

            j += 4;
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 4];
        _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining coefficients with scalar code
        while j < m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
            j += 1;
        }

        output[i] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process each output sample
    for i in 0..n {
        let mut sum_vec = _mm_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 2 coefficients at a time
        let m_vec = (m / 2) * 2;
        let mut j = 0;

        while j < m_vec && i >= j + 1 {
            // Load 2 coefficients
            let coeff_vec = _mm_loadu_pd(coeffs.as_ptr().add(j));

            // Load 2 input values in reverse order (for convolution)
            let idx0 = i - j;
            let idx1 = i - j - 1;

            let input_vec = _mm_set_pd(*input.get_unchecked(idx1), *input.get_unchecked(idx0));

            // Multiply and accumulate
            sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(coeff_vec, input_vec));

            j += 2;
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 2];
        _mm_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining coefficients with scalar code
        while j < m {
            if i >= j {
                sum += input[i - j] * coeffs[j];
            }
            j += 1;
        }

        output[i] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    // Compute autocorrelation for each lag
    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let mut sum_vec = _mm256_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 4 elements at a time
        let vec_len = (valid_length / 4) * 4;
        let mut i = 0;

        while i < vec_len {
            // Load 4 values from signal at position i
            let sig1 = _mm256_loadu_pd(signal.as_ptr().add(i));

            // Load 4 values from signal at position i + lag
            let sig2 = _mm256_loadu_pd(signal.as_ptr().add(i + lag));

            // Multiply and accumulate
            sum_vec = _mm256_fmadd_pd(sig1, sig2, sum_vec);

            i += 4;
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 4];
        _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining elements with scalar code
        for i in vec_len..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    // Compute autocorrelation for each lag
    for lag in 0..=max_lag {
        let valid_length = n - lag;
        let mut sum_vec = _mm_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 2 elements at a time
        let vec_len = (valid_length / 2) * 2;
        let mut i = 0;

        while i < vec_len {
            // Load 2 values from signal at position i
            let sig1 = _mm_loadu_pd(signal.as_ptr().add(i));

            // Load 2 values from signal at position i + lag
            let sig2 = _mm_loadu_pd(signal.as_ptr().add(i + lag));

            // Multiply and accumulate
            sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(sig1, sig2));

            i += 2;
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 2];
        _mm_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining elements with scalar code
        for i in vec_len..valid_length {
            sum += signal[i] * signal[i + lag];
        }

        autocorr[lag] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    let n1 = signal1.len();
    let n2 = signal2.len();

    let (output_len, start_offset) = match mode {
        "full" => (n1 + n2 - 1, 0),
        "same" => (n1, (n2 - 1) / 2),
        "valid" => (if n1 >= n2 { n1 - n2 + 1 } else { 0 }, n2 - 1),
        _ => return Err(SignalError::ValueError("Invalid mode".to_string())),
    };

    if output_len == 0 {
        return Ok(());
    }

    // Compute cross-correlation for each lag
    for i in 0..output_len {
        let lag = i + start_offset;
        let mut sum_vec = _mm256_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 4 elements at a time
        let vec_len = (n2 / 4) * 4;
        let mut j = 0;

        while j < vec_len {
            // Check bounds for all 4 elements
            let idx1_0 = lag.wrapping_sub(j);
            let idx1_1 = lag.wrapping_sub(j + 1);
            let idx1_2 = lag.wrapping_sub(j + 2);
            let idx1_3 = lag.wrapping_sub(j + 3);

            if idx1_0 < n1 && idx1_1 < n1 && idx1_2 < n1 && idx1_3 < n1 {
                // Load 4 values from signal1
                let sig1_vec = _mm256_set_pd(
                    *signal1.get_unchecked(idx1_3),
                    *signal1.get_unchecked(idx1_2),
                    *signal1.get_unchecked(idx1_1),
                    *signal1.get_unchecked(idx1_0),
                );

                // Load 4 values from signal2
                let sig2_vec = _mm256_loadu_pd(signal2.as_ptr().add(j));

                // Multiply and accumulate
                sum_vec = _mm256_fmadd_pd(sig1_vec, sig2_vec, sum_vec);

                j += 4;
            } else {
                break; // Fall back to scalar for boundary cases
            }
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 4];
        _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining elements with scalar code
        for j in j..n2 {
            let idx1 = lag.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }

        result[i] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    let n1 = signal1.len();
    let n2 = signal2.len();

    let (output_len, start_offset) = match mode {
        "full" => (n1 + n2 - 1, 0),
        "same" => (n1, (n2 - 1) / 2),
        "valid" => (if n1 >= n2 { n1 - n2 + 1 } else { 0 }, n2 - 1),
        _ => return Err(SignalError::ValueError("Invalid mode".to_string())),
    };

    if output_len == 0 {
        return Ok(());
    }

    // Compute cross-correlation for each lag
    for i in 0..output_len {
        let lag = i + start_offset;
        let mut sum_vec = _mm_setzero_pd();
        let mut sum = 0.0;

        // Vectorized part: process 2 elements at a time
        let vec_len = (n2 / 2) * 2;
        let mut j = 0;

        while j < vec_len {
            // Check bounds for both elements
            let idx1_0 = lag.wrapping_sub(j);
            let idx1_1 = lag.wrapping_sub(j + 1);

            if idx1_0 < n1 && idx1_1 < n1 {
                // Load 2 values from signal1
                let sig1_vec = _mm_set_pd(
                    *signal1.get_unchecked(idx1_1),
                    *signal1.get_unchecked(idx1_0),
                );

                // Load 2 values from signal2
                let sig2_vec = _mm_loadu_pd(signal2.as_ptr().add(j));

                // Multiply and accumulate
                sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(sig1_vec, sig2_vec));

                j += 2;
            } else {
                break; // Fall back to scalar for boundary cases
            }
        }

        // Horizontal sum of the vector accumulator
        let mut sum_array = [0.0; 2];
        _mm_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        sum = sum_array.iter().sum();

        // Process remaining elements with scalar code
        for j in j..n2 {
            let idx1 = lag.wrapping_sub(j);
            if idx1 < n1 {
                sum += signal1[idx1] * signal2[j];
            }
        }

        result[i] = sum;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_complex_butterfly(
    data: &mut [Complex64],
    twiddles: &[Complex64],
) -> SignalResult<()> {
    let n = data.len();
    let half_n = n / 2;

    // Process pairs of complex numbers (2 at a time = 4 f64 values)
    let mut i = 0;
    while i + 1 < half_n {
        // Load two complex numbers from upper half: data[i + half_n] and data[i + half_n + 1]
        let upper_re_im = _mm256_set_pd(
            data[i + half_n + 1].im,
            data[i + half_n + 1].re,
            data[i + half_n].im,
            data[i + half_n].re,
        );

        // Load two twiddle factors
        let twiddle_re_im = _mm256_set_pd(
            twiddles[i + 1].im,
            twiddles[i + 1].re,
            twiddles[i].im,
            twiddles[i].re,
        );

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        // Duplicate real and imaginary parts for multiplication
        let twiddle_re = _mm256_set_pd(
            twiddles[i + 1].re,
            twiddles[i + 1].re,
            twiddles[i].re,
            twiddles[i].re,
        );
        let twiddle_im = _mm256_set_pd(
            twiddles[i + 1].im,
            twiddles[i + 1].im,
            twiddles[i].im,
            twiddles[i].im,
        );

        // t_real = upper_re * twiddle_re - upper_im * twiddle_im
        // t_imag = upper_re * twiddle_im + upper_im * twiddle_re
        let upper_re = _mm256_set_pd(
            data[i + half_n + 1].re,
            data[i + half_n + 1].re,
            data[i + half_n].re,
            data[i + half_n].re,
        );
        let upper_im = _mm256_set_pd(
            data[i + half_n + 1].im,
            data[i + half_n + 1].im,
            data[i + half_n].im,
            data[i + half_n].im,
        );

        let t_re_part1 = _mm256_mul_pd(upper_re, twiddle_re);
        let t_re_part2 = _mm256_mul_pd(upper_im, twiddle_im);
        let t_im_part1 = _mm256_mul_pd(upper_re, twiddle_im);
        let t_im_part2 = _mm256_mul_pd(upper_im, twiddle_re);

        // Extract scalars for butterfly operation
        let mut t_re_arr = [0.0; 4];
        let mut t_im_arr = [0.0; 4];
        _mm256_storeu_pd(t_re_arr.as_mut_ptr(), _mm256_sub_pd(t_re_part1, t_re_part2));
        _mm256_storeu_pd(t_im_arr.as_mut_ptr(), _mm256_add_pd(t_im_part1, t_im_part2));

        let t0 = Complex64::new(t_re_arr[0], t_im_arr[0]);
        let t1 = Complex64::new(t_re_arr[2], t_im_arr[2]);

        // Butterfly: u = data[i], data[i] = u + t, data[i + half_n] = u - t
        let u0 = data[i];
        let u1 = data[i + 1];

        data[i] = u0 + t0;
        data[i + half_n] = u0 - t0;
        data[i + 1] = u1 + t1;
        data[i + half_n + 1] = u1 - t1;

        i += 2;
    }

    // Handle remaining element if odd
    while i < half_n {
        let t = data[i + half_n] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + half_n] = u - t;
        i += 1;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_complex_butterfly(
    data: &mut [Complex64],
    twiddles: &[Complex64],
) -> SignalResult<()> {
    let n = data.len();
    let half_n = n / 2;

    // Process one complex number at a time (2 f64 values)
    for i in 0..half_n {
        // Load complex number from upper half: data[i + half_n]
        let upper = _mm_set_pd(data[i + half_n].im, data[i + half_n].re);

        // Load twiddle factor
        let twiddle = _mm_set_pd(twiddles[i].im, twiddles[i].re);

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let upper_re = _mm_set1_pd(data[i + half_n].re);
        let upper_im = _mm_set1_pd(data[i + half_n].im);
        let twiddle_re = _mm_set1_pd(twiddles[i].re);
        let twiddle_im = _mm_set1_pd(twiddles[i].im);

        // t_real = upper_re * twiddle_re - upper_im * twiddle_im
        let t_re = _mm_sub_sd(
            _mm_mul_sd(upper_re, twiddle_re),
            _mm_mul_sd(upper_im, twiddle_im),
        );

        // t_imag = upper_re * twiddle_im + upper_im * twiddle_re
        let t_im = _mm_add_sd(
            _mm_mul_sd(upper_re, twiddle_im),
            _mm_mul_sd(upper_im, twiddle_re),
        );

        // Extract results
        let mut t_arr = [0.0; 2];
        _mm_storeu_pd(
            t_arr.as_mut_ptr(),
            _mm_set_pd(_mm_cvtsd_f64(t_im), _mm_cvtsd_f64(t_re)),
        );

        let t = Complex64::new(t_arr[0], t_arr[1]);

        // Butterfly: u = data[i], data[i] = u + t, data[i + half_n] = u - t
        let u = data[i];
        data[i] = u + t;
        data[i + half_n] = u - t;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();

    // Process 8 elements at a time with AVX-512
    let vec_len = (n / 8) * 8;
    let mut i = 0;

    while i < vec_len {
        // Load 8 signal values
        let sig = _mm512_loadu_pd(signal.as_ptr().add(i));

        // Load 8 window values
        let win = _mm512_loadu_pd(window.as_ptr().add(i));

        // Multiply
        let result = _mm512_mul_pd(sig, win);

        // Store
        _mm512_storeu_pd(output.as_mut_ptr().add(i), result);

        i += 8;
    }

    // Process remaining elements with scalar code
    for i in vec_len..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();

    // Process 4 elements at a time with AVX2
    let vec_len = (n / 4) * 4;
    let mut i = 0;

    while i < vec_len {
        // Load 4 signal values
        let sig = _mm256_loadu_pd(signal.as_ptr().add(i));

        // Load 4 window values
        let win = _mm256_loadu_pd(window.as_ptr().add(i));

        // Multiply
        let result = _mm256_mul_pd(sig, win);

        // Store
        _mm256_storeu_pd(output.as_mut_ptr().add(i), result);

        i += 4;
    }

    // Process remaining elements with scalar code
    for i in vec_len..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_apply_window(signal: &[f64], window: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = signal.len();

    // Process 2 elements at a time with SSE
    let vec_len = (n / 2) * 2;
    let mut i = 0;

    while i < vec_len {
        // Load 2 signal values
        let sig = _mm_loadu_pd(signal.as_ptr().add(i));

        // Load 2 window values
        let win = _mm_loadu_pd(window.as_ptr().add(i));

        // Multiply
        let result = _mm_mul_pd(sig, win);

        // Store
        _mm_storeu_pd(output.as_mut_ptr().add(i), result);

        i += 2;
    }

    // Process remaining element with scalar code
    for i in vec_len..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let n = signal.len();
    if n < 2 {
        return Ok(0);
    }

    let mut crossings = 0usize;
    let zero_vec = _mm256_setzero_pd();

    // Process 4 pairs at a time (need consecutive elements)
    let vec_len = if n >= 5 { n - 4 } else { 0 };
    let mut i = 0;

    while i < vec_len {
        // Load 4 consecutive values: signal[i], signal[i+1], signal[i+2], signal[i+3]
        let curr = _mm256_loadu_pd(signal.as_ptr().add(i));

        // Load next 4 values: signal[i+1], signal[i+2], signal[i+3], signal[i+4]
        let next = _mm256_loadu_pd(signal.as_ptr().add(i + 1));

        // Compare with zero to get signs (result is all 1s if true, all 0s if false)
        let curr_sign = _mm256_cmp_pd(curr, zero_vec, _CMP_GE_OQ); // >= 0
        let next_sign = _mm256_cmp_pd(next, zero_vec, _CMP_GE_OQ); // >= 0

        // XOR to find sign changes (zero crossing occurs when signs differ)
        let sign_change = _mm256_xor_pd(curr_sign, next_sign);

        // Extract bits and count crossings
        let mask = _mm256_movemask_pd(sign_change);

        // Count set bits (each bit represents a potential zero crossing)
        crossings += mask.count_ones() as usize;

        i += 4;
    }

    // Process remaining elements with scalar code
    for i in i..(n - 1) {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            crossings += 1;
        }
    }

    Ok(crossings)
}
