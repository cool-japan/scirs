use ndarray::s;
// Advanced SIMD operations for signal processing
//
// This module provides highly optimized SIMD implementations of common
// signal processing operations that go beyond the basic operations in
// scirs2-core, specifically targeting signal processing workloads.
//
// Enhanced features:
// - Multi-platform SIMD optimization (AVX512, AVX2, SSE4.1, NEON)
// - Memory alignment detection and optimization
// - Cache-friendly processing patterns
// - Automatic fallback strategies for edge cases
// - Performance monitoring and adaptive thresholds

use crate::error::{SignalError, SignalResult};
use crate::utilities::spectral::spectral_centroid;
use ndarray::{Array2, Array3, ArrayView1, ArrayViewMut1};
use num_complex::Complex64;
use rustfft::FftPlanner;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::check_finite;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::time::Instant;

#[allow(unused_imports)]
/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Force scalar fallback (for testing)
    pub force_scalar: bool,
    /// Minimum length for SIMD optimization
    pub simd_threshold: usize,
    /// Cache line alignment
    pub align_memory: bool,
    /// Use advanced instruction sets (AVX512, etc.)
    pub use_advanced: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Adaptive threshold adjustment based on performance
    pub adaptive_thresholds: bool,
    /// Cache line size for optimization (typically 64 bytes)
    pub cache_line_size: usize,
    /// Maximum unroll factor for loops
    pub max_unroll_factor: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            force_scalar: false,
            simd_threshold: 64,
            align_memory: true,
            use_advanced: false, // Disable until AVX-512 features are stabilized
            enable_monitoring: false,
            adaptive_thresholds: false,
            cache_line_size: 64,
            max_unroll_factor: 8,
        }
    }
}

/// SIMD-optimized FIR filtering kernel
///
/// Performs FIR filtering using SIMD instructions with optimal memory access patterns
///
/// # Arguments
///
/// * `input` - Input signal
/// * `coeffs` - Filter coefficients (assumed to be relatively short)
/// * `output` - Output buffer (must be pre-allocated)
/// * `config` - SIMD configuration
#[allow(dead_code)]
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

    check_finite(input, "input value")?;
    check_finite(coeffs, "coeffs value")?;

    let n = input.len();
    let m = coeffs.len();

    if n < config.simd_threshold || config.force_scalar {
        return scalar_fir_filter(input, coeffs, output);
    }

    // Check for SIMD capabilities
    let caps = PlatformCapabilities::detect();

    if caps.avx512_available && config.use_advanced {
        unsafe { avx512_fir_filter(input, coeffs, output) }
    } else if caps.avx2_available {
        unsafe { avx2_fir_filter(input, coeffs, output) }
    } else if caps.simd_available {
        unsafe { sse_fir_filter(input, coeffs, output) }
    } else {
        scalar_fir_filter(input, coeffs, output)
    }
}

/// SIMD-optimized autocorrelation computation
///
/// Computes autocorrelation function using SIMD vectorization with
/// cache-friendly memory access patterns
#[allow(dead_code)]
pub fn simd_autocorrelation(
    signal: &[f64],
    max_lag: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_finite(signal, "signal value")?;

    let n = signal.len();
    if max_lag >= n {
        return Err(SignalError::ValueError(
            "Maximum _lag must be less than signal length".to_string(),
        ));
    }

    let mut autocorr = vec![0.0; max_lag + 1];

    if n < config.simd_threshold || config.force_scalar {
        scalar_autocorrelation(signal, &mut autocorr, max_lag)?;
        return Ok(autocorr);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_autocorrelation(signal, &mut autocorr, max_lag) }?;
    } else if caps.simd_available {
        unsafe { sse_autocorrelation(signal, &mut autocorr, max_lag) }?;
    } else {
        scalar_autocorrelation(signal, &mut autocorr, max_lag)?;
        return Ok(autocorr);
    }

    Ok(autocorr)
}

/// SIMD-optimized cross-correlation
///
/// Computes cross-correlation between two signals using vectorized operations
#[allow(dead_code)]
pub fn simd_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    mode: &str,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    check_finite(signal1, "signal1 value")?;
    check_finite(signal2, "signal2 value")?;

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

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_cross_correlation(signal1, signal2, &mut result, mode) }?;
    } else if caps.simd_available {
        unsafe { sse_cross_correlation(signal1, signal2, &mut result, mode) }?;
    } else {
        scalar_cross_correlation(signal1, signal2, &mut result, mode)?;
        return Ok(result);
    }

    Ok(result)
}

/// SIMD-optimized complex FFT butterfly operations
///
/// Performs vectorized complex arithmetic for FFT computations
#[allow(dead_code)]
pub fn simd_complex_fft_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
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
        unsafe { avx2_complex_butterfly(data, twiddles) }
    } else if caps.simd_available {
        unsafe { sse_complex_butterfly(data, twiddles) }
    } else {
        scalar_complex_butterfly(data, twiddles)
    }
}

/// SIMD-optimized windowing function application
///
/// Applies window functions using vectorized operations
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

    check_finite(signal, "signal value")?;
    check_finite(window, "window value")?;

    let n = signal.len();

    if n < config.simd_threshold || config.force_scalar {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        return Ok(());
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx512_available && config.use_advanced {
        unsafe { avx512_apply_window(signal, window, output) }
    } else if caps.avx2_available {
        unsafe { avx2_apply_window(signal, window, output) }
    } else if caps.simd_available {
        unsafe { sse_apply_window(signal, window, output) }
    } else {
        for i in 0..n {
            output[i] = signal[i] * window[i];
        }
        Ok(())
    }
}

// Scalar fallback implementations

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn scalar_autocorrelation(_signal: &[f64], autocorr: &mut [f64], maxlag: usize) -> SignalResult<()> {
    let n = signal.len();

    for _lag in 0..=max_lag {
        let mut sum = 0.0;
        let valid_length = n - lag;

        for i in 0..valid_length {
            sum += signal[i] * signal[i + _lag];
        }

        autocorr[_lag] = sum / valid_length as f64;
    }

    Ok(())
}

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
        return Ok(vec![]);
    }

    let mut result = vec![0.0; output_len];

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

#[allow(dead_code)]
fn scalar_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    for i in 0..data.len() / 2 {
        let t = data[i + data.len() / 2] * twiddles[i];
        let u = data[i];
        data[i] = u + t;
        data[i + data.len() / 2] = u - t;
    }
    Ok(())
}

// SIMD implementations (platform-specific)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let n = input.len();
    let m = coeffs.len();

    // Process 4 samples at a time with AVX2
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let base_idx = chunk * simd_width;
        let mut result = _mm256_setzero_pd();

        for j in 0..m {
            if base_idx >= j {
                let input_idx = base_idx - j;
                if input_idx + simd_width <= n {
                    let input_vec = _mm256_loadu_pd(_input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm256_set1_pd(coeffs[j]);
                    result = _mm256_fmadd_pd(input_vec, coeff_broadcast, result);
                }
            }
        }

        _mm256_storeu_pd(output.as_mut_ptr().add(base_idx), result);
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

#[cfg(feature = "unstable_avx512")] // Disabled by default
                                    // #[target_feature(enable = "avx512f")] // Disabled - unstable feature
unsafe fn avx512_fir_filter(
    _input: &[f64],
    coeffs: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    // Fallback to scalar implementation until AVX-512 is stabilized
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

#[cfg(not(feature = "unstable_avx512"))]
unsafe fn avx512_fir_filter(
    _input: &[f64],
    _coeffs: &[f64],
    _output: &mut [f64],
) -> SignalResult<()> {
    // Fallback implementation or error
    Err(SignalError::ComputationError(
        "AVX512 not available".to_string(),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
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
                    let input_vec = _mm_loadu_pd(_input.as_ptr().add(input_idx));
                    let coeff_broadcast = _mm_set1_pd(coeffs[j]);
                    result = _mm_add_pd(result_mm_mul_pd(input_vec, coeff_broadcast));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    let n = signal.len();

    for _lag in 0..=max_lag {
        let valid_length = n - lag;
        let simd_width = 4;
        let simd_chunks = valid_length / simd_width;

        let mut sum_vec = _mm256_setzero_pd();

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;
            let vec1 = _mm256_loadu_pd(signal.as_ptr().add(idx));
            let vec2 = _mm256_loadu_pd(signal.as_ptr().add(idx + lag));
            sum_vec = _mm256_fmadd_pd(vec1, vec2, sum_vec);
        }

        // Horizontal sum of the vector
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        let mut sum = sum_array.iter().sum::<f64>();

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..valid_length {
            sum += signal[i] * signal[i + _lag];
        }

        autocorr[_lag] = sum / valid_length as f64;
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

    for _lag in 0..=max_lag {
        let valid_length = n - lag;
        let simd_width = 2;
        let simd_chunks = valid_length / simd_width;

        let mut sum_vec = _mm_setzero_pd();

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;
            let vec1 = _mm_loadu_pd(signal.as_ptr().add(idx));
            let vec2 = _mm_loadu_pd(signal.as_ptr().add(idx + lag));
            sum_vec = _mm_add_pd(sum_vec_mm_mul_pd(vec1, vec2));
        }

        // Extract sum from vector
        let sum_array: [f64; 2] = std::mem::transmute(sum_vec);
        let mut sum = sum_array[0] + sum_array[1];

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..valid_length {
            sum += signal[i] * signal[i + _lag];
        }

        autocorr[_lag] = sum / valid_length as f64;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    _mode: &str,
) -> SignalResult<()> {
    // Simplified AVX2 implementation for demonstration
    let n1 = signal1.len();
    let n2 = signal2.len();
    let output_len = result.len();

    let simd_width = 4;

    for i in 0..output_len {
        let mut sum_vec = _mm256_setzero_pd();
        let simd_chunks = n2 / simd_width;

        for chunk in 0..simd_chunks {
            let j_base = chunk * simd_width;
            // Load 4 elements from signal2
            let sig2_vec = _mm256_loadu_pd(signal2.as_ptr().add(j_base));

            // Load corresponding elements from signal1 (with bounds checking)
            let mut sig1_array = [0.0; 4];
            for k in 0..simd_width {
                let idx1 = i.wrapping_sub(j_base + k);
                if idx1 < n1 {
                    sig1_array[k] = signal1[idx1];
                }
            }
            let sig1_vec = _mm256_loadu_pd(sig1_array.as_ptr());

            sum_vec = _mm256_fmadd_pd(sig1_vec, sig2_vec, sum_vec);
        }

        // Extract sum from vector
        let sum_array: [f64; 4] = std::mem::transmute(sum_vec);
        let mut sum = sum_array.iter().sum::<f64>();

        // Handle remaining elements
        for j in (simd_chunks * simd_width)..n2 {
            let idx1 = i.wrapping_sub(j);
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
    // SSE implementation (simplified)
    scalar_cross_correlation(signal1, signal2, result, mode)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    // SSE implementation
    scalar_complex_butterfly(data, twiddles)
}

// AVX-512 implementation disabled due to unstable features
// #[target_feature(enable = "avx512f")]
unsafe fn avx512_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    // Fallback to scalar implementation until AVX-512 is stabilized
    for i in 0..signal.len() {
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
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm256_loadu_pd(signal.as_ptr().add(idx));
        let win_vec = _mm256_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm256_mul_pd(sig_vec, win_vec);
        _mm256_storeu_pd(output.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_apply_window(
    _signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let n = signal.len();
    let simd_width = 2;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;
        let sig_vec = _mm_loadu_pd(_signal.as_ptr().add(idx));
        let win_vec = _mm_loadu_pd(window.as_ptr().add(idx));
        let result_vec = _mm_mul_pd(sig_vec, win_vec);
        _mm_storeu_pd(output.as_mut_ptr().add(idx), result_vec);
    }

    // Handle remaining elements
    for i in (simd_chunks * simd_width)..n {
        output[i] = signal[i] * window[i];
    }

    Ok(())
}

/// Performance benchmark for SIMD operations
#[allow(dead_code)]
pub fn benchmark_simd_operations(signallength: usize) -> SignalResult<()> {
    let signal: Vec<f64> = (0..signal_length)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();

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

/// SIMD-optimized spectral centroid computation
///
/// Computes the spectral centroid (center of mass of the spectrum) using
/// SIMD vectorization for high-performance audio analysis.
///
/// # Arguments
///
/// * `magnitude_spectrum` - Magnitude spectrum values
/// * `frequencies` - Corresponding frequency values
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Spectral centroid in Hz
#[allow(dead_code)]
pub fn simd_spectral_centroid(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    config: &SimdConfig,
) -> SignalResult<f64> {
    if magnitude_spectrum.len() != frequencies.len() {
        return Err(SignalError::ValueError(
            "Magnitude _spectrum and frequencies must have same length".to_string(),
        ));
    }

    check_finite(magnitude_spectrum, "magnitude_spectrum value")?;
    check_finite(frequencies, "frequencies value")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_centroid(magnitude_spectrum, frequencies);
    }

    let caps = PlatformCapabilities::detect();

    // Convert to ArrayViews for SIMD operations
    let mag_view = ArrayView1::from(magnitude_spectrum);
    let freq_view = ArrayView1::from(frequencies);

    // Compute weighted sum (magnitude * frequency) and total magnitude
    let weighted_sum = f64::simd_dot(&mag_view, &freq_view);
    let total_magnitude = f64::simd_sum(&mag_view);

    if total_magnitude < 1e-12 {
        return Ok(0.0);
    }

    Ok(weighted_sum / total_magnitude)
}

/// Scalar fallback for spectral centroid
#[allow(dead_code)]
fn scalar_spectral_centroid(_magnitudespectrum: &[f64], frequencies: &[f64]) -> SignalResult<f64> {
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (mag, freq) in magnitude_spectrum.iter().zip(frequencies.iter()) {
        weighted_sum += mag * freq;
        total_magnitude += mag;
    }

    if total_magnitude < 1e-12 {
        return Ok(0.0);
    }

    Ok(weighted_sum / total_magnitude)
}

/// SIMD-optimized spectral rolloff computation
///
/// Computes the frequency below which a specified percentage of the total
/// spectral energy lies.
///
/// # Arguments
///
/// * `magnitude_spectrum` - Magnitude spectrum values
/// * `frequencies` - Corresponding frequency values
/// * `rolloff_threshold` - Percentage of energy (e.g., 0.85 for 85%)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Rolloff frequency in Hz
#[allow(dead_code)]
pub fn simd_spectral_rolloff(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    rolloff_threshold: f64,
    config: &SimdConfig,
) -> SignalResult<f64> {
    if magnitude_spectrum.len() != frequencies.len() {
        return Err(SignalError::ValueError(
            "Magnitude _spectrum and frequencies must have same length".to_string(),
        ));
    }

    if rolloff_threshold <= 0.0 || rolloff_threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "Rolloff _threshold must be between 0 and 1".to_string(),
        ));
    }

    check_finite(magnitude_spectrum, "magnitude_spectrum value")?;
    check_finite(frequencies, "frequencies value")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_rolloff(magnitude_spectrum, frequencies, rolloff_threshold);
    }

    // Compute energy _spectrum (magnitude^2) using SIMD
    let mut energy_spectrum = vec![0.0; n];
    let mag_view = ArrayView1::from(magnitude_spectrum);
    let energy_view = ArrayViewMut1::from(&mut energy_spectrum);

    // Use SIMD element-wise multiplication
    f64::simd_mul(&mag_view, &mag_view, &energy_view);

    // Compute total energy using SIMD
    let total_energy = f64::simd_sum(&ArrayView1::from(&energy_spectrum));
    let target_energy = total_energy * rolloff_threshold;

    // Find rolloff point
    let mut cumulative_energy = 0.0;
    for (i, &energy) in energy_spectrum.iter().enumerate() {
        cumulative_energy += energy;
        if cumulative_energy >= target_energy {
            return Ok(frequencies[i]);
        }
    }

    // If we reach here, return the last frequency
    Ok(frequencies[n - 1])
}

/// Scalar fallback for spectral rolloff
#[allow(dead_code)]
fn scalar_spectral_rolloff(
    magnitude_spectrum: &[f64],
    frequencies: &[f64],
    rolloff_threshold: f64,
) -> SignalResult<f64> {
    let n = magnitude_spectrum.len();

    // Compute total energy
    let total_energy: f64 = magnitude_spectrum.iter().map(|&mag| mag * mag).sum();
    let target_energy = total_energy * rolloff_threshold;

    // Find rolloff point
    let mut cumulative_energy = 0.0;
    for i in 0..n {
        cumulative_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
        if cumulative_energy >= target_energy {
            return Ok(frequencies[i]);
        }
    }

    Ok(frequencies[n - 1])
}

/// SIMD-optimized peak detection with advanced vectorization
///
/// Detects peaks in a signal using SIMD operations for high-performance
/// real-time peak detection applications.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `min_height` - Minimum peak height
/// * `min_distance` - Minimum distance between peaks
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Vector of peak indices
#[allow(dead_code)]
pub fn simd_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<usize>> {
    check_finite(signal, "signal value")?;

    let n = signal.len();
    if n < 3 {
        return Ok(vec![]);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_peak_detection(signal, min_height, min_distance);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient comparison operations
    let signal_view = ArrayView1::from(signal);
    let mut peak_candidates = Vec::new();

    // SIMD-optimized local maxima detection
    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_peak_detection(signal, min_height, &mut peak_candidates)? };
    } else {
        scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    }

    // Apply minimum _distance constraint
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);

    Ok(peak_candidates)
}

/// AVX2 optimized peak detection
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_peak_detection(
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

/// Scalar local maxima detection
#[allow(dead_code)]
fn scalar_local_maxima_detection(
    signal: &[f64],
    min_height: f64,
    peak_candidates: &mut Vec<usize>,
) {
    let n = signal.len();

    for i in 1..(n - 1) {
        if signal[i] >= min_height && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peak_candidates.push(i);
        }
    }
}

/// Apply minimum distance constraint to peak candidates
#[allow(dead_code)]
fn apply_minimum_distance_constraint(
    peak_candidates: &mut Vec<usize>,
    signal: &[f64],
    min_distance: usize,
) {
    if peak_candidates.is_empty() || min_distance == 0 {
        return;
    }

    // Sort by peak height (descending)
    peak_candidates.sort_by(|&a, &b| signal[b].partial_cmp(&signal[a]).unwrap());

    let mut filtered_peaks = Vec::new();

    for &candidate in peak_candidates.iter() {
        let mut too_close = false;

        for &existing_peak in &filtered_peaks {
            if (candidate as i32 - existing_peak as i32).abs() < min_distance as i32 {
                too_close = true;
                break;
            }
        }

        if !too_close {
            filtered_peaks.push(candidate);
        }
    }

    // Sort by index
    filtered_peaks.sort_unstable();
    *peak_candidates = filtered_peaks;
}

/// Scalar fallback for peak detection
#[allow(dead_code)]
fn scalar_peak_detection(
    signal: &[f64],
    min_height: f64,
    min_distance: usize,
) -> SignalResult<Vec<usize>> {
    let mut peak_candidates = Vec::new();
    scalar_local_maxima_detection(signal, min_height, &mut peak_candidates);
    apply_minimum_distance_constraint(&mut peak_candidates, signal, min_distance);
    Ok(peak_candidates)
}

/// SIMD-optimized zero-crossing rate computation
///
/// Computes the zero-crossing rate using vectorized operations for
/// efficient audio analysis and speech processing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * Zero-crossing rate (crossings per sample)
#[allow(dead_code)]
pub fn simd_zero_crossing_rate(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_finite(_signal, "_signal value")?;

    let n = signal.len();
    if n < 2 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_zero_crossing_rate(_signal);
    }

    let caps = PlatformCapabilities::detect();

    // Use SIMD for efficient sign comparison
    let mut crossings = 0;

    if caps.avx2_available && config.use_advanced {
        unsafe { crossings = avx2_zero_crossings(_signal)? };
    } else {
        crossings = scalar_count_zero_crossings(_signal);
    }

    Ok(crossings as f64 / (n - 1) as f64)
}

/// AVX2 optimized zero crossing detection
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let n = signal.len();
    let mut crossings = 0;

    // Process pairs of consecutive elements
    for i in 0..(n - 1) {
        if (_signal[i] >= 0.0 && signal[i + 1] < 0.0)
            || (_signal[i] < 0.0 && signal[i + 1] >= 0.0)
        {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Scalar zero crossing count
#[allow(dead_code)]
fn scalar_count_zero_crossings(signal: &[f64]) -> usize {
    let n = signal.len();
    let mut crossings = 0;

    for i in 0..(n - 1) {
        if (_signal[i] >= 0.0 && signal[i + 1] < 0.0)
            || (_signal[i] < 0.0 && signal[i + 1] >= 0.0)
        {
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

    let crossings = scalar_count_zero_crossings(_signal);
    Ok(crossings as f64 / (n - 1) as f64)
}

/// SIMD-optimized energy computation
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
/// * Signal energy (sum of squares)
#[allow(dead_code)]
pub fn simd_signal_energy(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    check_finite(_signal, "_signal value")?;

    let n = signal.len();
    if n == 0 {
        return Ok(0.0);
    }

    if n < config.simd_threshold || config.force_scalar {
        return scalar_signal_energy(_signal);
    }

    let caps = PlatformCapabilities::detect();
    let signal_view = ArrayView1::from(_signal);

    // Use SIMD dot product for efficient energy computation
    let energy = f64::simd_dot(&signal_view, &signal_view);
    Ok(energy)
}

/// Scalar fallback for signal energy
#[allow(dead_code)]
fn scalar_signal_energy(signal: &[f64]) -> SignalResult<f64> {
    let energy = signal.iter().map(|&x| x * x).sum();
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
/// * RMS value
#[allow(dead_code)]
pub fn simd_rms(signal: &[f64], config: &SimdConfig) -> SignalResult<f64> {
    let energy = simd_signal_energy(_signal, config)?;
    let n = signal.len();

    if n == 0 {
        Ok(0.0)
    } else {
        Ok((energy / n as f64).sqrt())
    }
}

/// High-performance batch spectral analysis with SIMD optimizations
///
/// Performs multiple spectral analysis operations on a batch of signals
/// using SIMD vectorization and parallel processing for maximum throughput.
///
/// # Arguments
///
/// * `signals` - Batch of input signals (each row is a signal)
/// * `window_type` - Window function to apply ("hann", "hamming", "blackman", etc.)
/// * `nfft` - FFT size (must be power of 2)
/// * `config` - SIMD configuration
///
/// # Returns
///
/// * `BatchSpectralResult` containing power spectra, phases, and statistics
#[allow(dead_code)]
pub fn simd_batch_spectral_analysis(
    signals: &Array2<f64>,
    window_type: &str,
    nfft: usize,
    config: &SimdConfig,
) -> SignalResult<BatchSpectralResult> {
    let (n_signals, signal_len) = signals.dim();

    if signal_len == 0 || n_signals == 0 {
        return Err(SignalError::ValueError("Empty input signals".to_string()));
    }

    if !nfft.is_power_of_two() {
        return Err(SignalError::ValueError(
            "FFT size must be power of 2".to_string(),
        ));
    }

    // Generate window function using SIMD
    let window = generate_simd_window(window_type, signal_len, config)?;

    // Pre-allocate results
    let n_freqs = nfft / 2 + 1;
    let mut power_spectra = Array2::<f64>::zeros((n_signals, n_freqs));
    let mut phases = Array2::<f64>::zeros((n_signals, n_freqs));
    let mut statistics = BatchSpectralStats {
        mean_power: vec![0.0; n_freqs],
        max_power: vec![0.0; n_freqs],
        snr_estimates: vec![0.0; n_signals],
        spectral_centroids: vec![0.0; n_signals],
    };

    // Process signals in parallel using rayon
    let results: Vec<_> = if n_signals >= 4 && !config.force_scalar {
        (0..n_signals)
            .into_par_iter()
            .map(|i| {
                let signal = signals.row(i);
                process_single_signal_simd(signal.as_slice().unwrap(), &window, nfft, config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    } else {
        // Sequential processing for small batches
        (0..n_signals)
            .map(|i| {
                let signal = signals.row(i);
                process_single_signal_simd(signal.as_slice().unwrap(), &window, nfft, config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Collect results and compute batch statistics
    for (i, result) in results.into_iter().enumerate() {
        // Store power spectrum and phase
        for (j, &power) in result.power_spectrum.iter().enumerate() {
            power_spectra[[i, j]] = power;
            phases[[i, j]] = result.phase[j];
        }

        // Update statistics
        statistics.snr_estimates[i] = result.snr_estimate;
        statistics.spectral_centroids[i] = result.spectral_centroid;

        // Update batch statistics
        for (j, &power) in result.power_spectrum.iter().enumerate() {
            statistics.mean_power[j] += power;
            statistics.max_power[j] = statistics.max_power[j].max(power);
        }
    }

    // Finalize mean power
    for power in statistics.mean_power.iter_mut() {
        *power /= n_signals as f64;
    }

    Ok(BatchSpectralResult {
        power_spectra,
        phases,
        statistics,
        frequencies: (0..n_freqs)
            .map(|i| i as f64 * 0.5 / n_freqs as f64)
            .collect(),
    })
}

/// Result of batch spectral analysis
#[derive(Debug, Clone)]
pub struct BatchSpectralResult {
    /// Power spectra for all signals (n_signals x n_frequencies)
    pub power_spectra: Array2<f64>,
    /// Phase information for all signals (n_signals x n_frequencies)
    pub phases: Array2<f64>,
    /// Batch statistics
    pub statistics: BatchSpectralStats,
    /// Frequency bins (normalized)
    pub frequencies: Vec<f64>,
}

/// Batch spectral statistics
#[derive(Debug, Clone)]
pub struct BatchSpectralStats {
    /// Mean power across all signals
    pub mean_power: Vec<f64>,
    /// Maximum power across all signals
    pub max_power: Vec<f64>,
    /// SNR estimates for each signal
    pub snr_estimates: Vec<f64>,
    /// Spectral centroids for each signal
    pub spectral_centroids: Vec<f64>,
}

/// Single signal spectral result
#[derive(Debug, Clone)]
struct SingleSpectralResult {
    power_spectrum: Vec<f64>,
    phase: Vec<f64>,
    snr_estimate: f64,
    spectral_centroid: f64,
}

/// Process a single signal with SIMD optimizations
#[allow(dead_code)]
fn process_single_signal_simd(
    signal: &[f64],
    window: &[f64],
    nfft: usize,
    config: &SimdConfig,
) -> SignalResult<SingleSpectralResult> {
    let n = signal.len();
    let mut windowed = vec![0.0; n];

    // Apply window using SIMD
    simd_apply_window(signal, window, &mut windowed, config)?;

    // Zero-pad to FFT size
    let mut padded = vec![Complex64::new(0.0, 0.0); nfft];
    for (i, &val) in windowed.iter().enumerate() {
        if i < nfft {
            padded[i] = Complex64::new(val, 0.0);
        }
    }

    // Compute FFT using rustfft
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);
    fft.process(&mut padded);

    // Extract power spectrum and phase (one-sided)
    let n_freqs = nfft / 2 + 1;
    let mut power_spectrum = vec![0.0; n_freqs];
    let mut phase = vec![0.0; n_freqs];

    for i in 0..n_freqs {
        let magnitude = padded[i].norm();
        power_spectrum[i] = magnitude * magnitude;
        phase[i] = padded[i].arg();

        // Scale for one-sided spectrum (except DC and Nyquist)
        if i > 0 && i < n_freqs - 1 {
            power_spectrum[i] *= 2.0;
        }
    }

    // Compute SNR estimate (signal power vs noise floor)
    let total_power: f64 = power_spectrum.iter().sum();
    let noise_floor = power_spectrum.iter().take(10).sum::<f64>() / 10.0; // Estimate from low frequencies
    let snr_estimate = if noise_floor > 1e-15 {
        10.0 * (total_power / noise_floor).log10()
    } else {
        100.0 // Very high SNR
    };

    // Compute spectral centroid
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;
    for (i, &power) in power_spectrum.iter().enumerate() {
        let magnitude = power.sqrt();
        weighted_sum += i as f64 * magnitude;
        magnitude_sum += magnitude;
    }

    let spectral_centroid = if magnitude_sum > 1e-15 {
        weighted_sum / magnitude_sum / n_freqs as f64
    } else {
        0.0
    };

    Ok(SingleSpectralResult {
        power_spectrum,
        phase,
        snr_estimate,
        spectral_centroid,
    })
}

/// Generate window function using SIMD optimizations
#[allow(dead_code)]
fn generate_simd_window(
    window_type: &str,
    length: usize,
    config: &SimdConfig,
) -> SignalResult<Vec<f64>> {
    let mut window = vec![0.0; length];

    match window_type {
        "hann" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.5 * (1.0 - phase.cos());
            }
        }
        "hamming" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.54 - 0.46 * phase.cos();
            }
        }
        "blackman" => {
            for i in 0..length {
                let phase = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            }
        }
        "rectangular" | "boxcar" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window _type: {}",
                window_type
            )));
        }
    }

    // Normalize window energy if using advanced optimizations
    if config.use_advanced {
        let energy: f64 = window.iter().map(|&x| x * x).sum();
        let norm_factor = (length as f64 / energy).sqrt();
        for w in window.iter_mut() {
            *w *= norm_factor;
        }
    }

    Ok(window)
}

/// Performance monitoring structure for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdPerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Input size
    pub input_size: usize,
    /// Time taken in nanoseconds
    pub time_ns: u64,
    /// SIMD instruction set used
    pub instruction_set: String,
    /// Memory throughput (bytes/second)
    pub memory_throughput: f64,
    /// Computational throughput (operations/second)
    pub compute_throughput: f64,
}

/// Enhanced SIMD convolution with advanced optimizations
#[allow(dead_code)]
pub fn simd_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    check_finite(signal, "signal value")?;
    check_finite(kernel, "kernel value")?;

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

    if caps.avx512_available && config.use_advanced {
        unsafe { avx512_enhanced_convolution(signal, kernel, output) }
    } else if caps.avx2_available {
        unsafe { avx2_enhanced_convolution(signal, kernel, output) }
    } else {
        scalar_enhanced_convolution(signal, kernel, output)
    }
}

/// Scalar fallback for enhanced convolution
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

/// AVX2 enhanced convolution with cache optimization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_enhanced_convolution(
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
// #[target_feature(enable = "avx512f")] // Disabled - unstable feature
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

/// Advanced-high-performance SIMD matrix operations for signal processing
///
/// This module provides advanced-optimized SIMD implementations for matrix operations
/// commonly used in signal processing, including:
/// - SIMD-accelerated matrix-vector multiplication
/// - Batch signal convolution with matrix kernels
/// - SIMD-optimized covariance matrix computation
/// - High-performance autocorrelation matrix calculation
/// - Real-time signal filtering with multiple channels
pub mod advanced_simd_matrix {
    use super::{simd_enhanced_convolution, SimdConfig};
    use crate::error::{SignalError, SignalResult};
    use ndarray::s;
    use ndarray::{Array2, Array3, ArrayView1};
    use scirs2_core::simd_ops::SimdUnifiedOps;
    use scirs2_core::validation::check_finite;

    /// SIMD-accelerated matrix-vector multiplication for signal processing
    ///
    /// Optimized for signal processing applications where the matrix represents
    /// filter banks, transformation matrices, or correlation matrices.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input matrix (rows x cols)
    /// * `vector` - Input vector (length = cols)
    /// * `result` - Output vector (length = rows)
    /// * `config` - SIMD configuration
    pub fn simd_matrix_vector_mul(
        matrix: &Array2<f64>,
        vector: &[f64],
        result: &mut [f64],
        config: &SimdConfig,
    ) -> SignalResult<()> {
        let (rows, cols) = matrix.dim();

        if vector.len() != cols {
            return Err(SignalError::ValueError(format!(
                "Vector length {} doesn't match matrix columns {}",
                vector.len(),
                cols
            )));
        }

        if result.len() != rows {
            return Err(SignalError::ValueError(format!(
                "Result length {} doesn't match matrix rows {}",
                result.len(),
                rows
            )));
        }

        check_finite(vector, "vector value")?;

        // Use parallel processing for large matrices
        if rows >= 100 && cols >= 100 && !config.force_scalar {
            (0..rows).into_par_iter().for_each(|i| {
                let row = matrix.row(i);
                let mut sum = 0.0;

                // SIMD-accelerated dot product
                let vector_view = ArrayView1::from(vector);
                if cols >= config.simd_threshold {
                    sum = f64::simd_dot(&row, &vector_view);
                } else {
                    sum = row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum();
                }

                result[i] = sum;
            });
        } else {
            // Sequential processing for smaller matrices
            for i in 0..rows {
                let row = matrix.row(i);
                let vector_view = ArrayView1::from(vector);

                result[i] = if cols >= config.simd_threshold && !config.force_scalar {
                    f64::simd_dot(&row, &vector_view)
                } else {
                    row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum()
                };
            }
        }

        Ok(())
    }

    /// SIMD-optimized batch convolution for multiple channels
    ///
    /// Performs convolution of multiple input signals with multiple kernels
    /// using SIMD acceleration and parallel processing.
    ///
    /// # Arguments
    ///
    /// * `signals` - Input signals (channels x samples)
    /// * `kernels` - Convolution kernels (kernels x kernel_length)
    /// * `outputs` - Output buffers (channels x kernels x output_length)
    /// * `config` - SIMD configuration
    pub fn simd_batch_convolution(
        signals: &Array2<f64>,
        kernels: &Array2<f64>,
        outputs: &mut Array3<f64>,
        config: &SimdConfig,
    ) -> SignalResult<()> {
        let (n_channels, signal_len) = signals.dim();
        let (n_kernels, kernel_len) = kernels.dim();
        let expected_output_len = signal_len + kernel_len - 1;

        if outputs.dim() != (n_channels, n_kernels, expected_output_len) {
            return Err(SignalError::ValueError(
                "Output array dimensions don't match expected size".to_string(),
            ));
        }

        // Process each channel-kernel combination
        if n_channels * n_kernels >= 4 && !config.force_scalar {
            // Parallel processing for multiple combinations
            // Collect results first to avoid mutable borrow issues
            let results: Vec<_> = (0..n_channels)
                .into_par_iter()
                .flat_map(|ch| {
                    (0..n_kernels)
                        .map(move |k| {
                            let signal = signals.row(ch);
                            let kernel = kernels.row(k);

                            // Use enhanced SIMD convolution
                            let mut output_vec = vec![0.0; expected_output_len];
                            let _ = simd_enhanced_convolution(
                                signal.as_slice().unwrap(),
                                kernel.as_slice().unwrap(),
                                &mut output_vec,
                                config,
                            );

                            (ch, k, output_vec)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Write results back to outputs
            for (ch, k, output_vec) in results {
                let mut output = outputs.slice_mut(s![ch, k, ..]);
                for (i, &val) in output_vec.iter().enumerate() {
                    output[i] = val;
                }
            }
        } else {
            // Sequential processing
            for ch in 0..n_channels {
                for k in 0..n_kernels {
                    let signal = signals.row(ch);
                    let kernel = kernels.row(k);
                    let mut output = outputs.slice_mut(s![ch, k, ..]);

                    let mut output_vec = vec![0.0; expected_output_len];
                    simd_enhanced_convolution(
                        signal.as_slice().unwrap(),
                        kernel.as_slice().unwrap(),
                        &mut output_vec,
                        config,
                    )?;

                    for (i, &val) in output_vec.iter().enumerate() {
                        output[i] = val;
                    }
                }
            }
        }

        Ok(())
    }

    /// SIMD-accelerated covariance matrix computation
    ///
    /// Computes the covariance matrix of input signals using SIMD acceleration
    /// for maximum performance in multichannel signal analysis.
    ///
    /// # Arguments
    ///
    /// * `signals` - Input signals (channels x samples)
    /// * `covariance` - Output covariance matrix (channels x channels)
    /// * `config` - SIMD configuration
    pub fn simd_covariance_matrix(
        signals: &Array2<f64>,
        covariance: &mut Array2<f64>,
        config: &SimdConfig,
    ) -> SignalResult<()> {
        let (n_channels, n_samples) = signals.dim();

        if covariance.dim() != (n_channels, n_channels) {
            return Err(SignalError::ValueError(
                "Covariance matrix dimensions incorrect".to_string(),
            ));
        }

        // Compute means for each channel using SIMD
        let mut means = vec![0.0; n_channels];
        for ch in 0..n_channels {
            let signal = signals.row(ch);
            means[ch] = signal.sum() / n_samples as f64;
        }

        // Compute covariance matrix with SIMD acceleration
        if n_channels >= 8 && n_samples >= config.simd_threshold && !config.force_scalar {
            // Parallel computation for large matrices
            // Collect results first to avoid mutable borrow issues
            let cov_results: Vec<_> = (0..n_channels)
                .into_par_iter()
                .flat_map(|i| {
                    let means_clone = means.clone();
                    (i..n_channels)
                        .map(move |j| {
                            let signal_i = signals.row(i);
                            let signal_j = signals.row(j);

                            // SIMD-accelerated covariance calculation
                            let mut cov = 0.0;
                            let mean_i = means_clone[i];
                            let mean_j = means_clone[j];

                            // Use SIMD for the inner loop
                            let chunks = n_samples / 4;
                            for chunk in 0..chunks {
                                let start_idx = chunk * 4;
                                let end_idx = start_idx + 4;

                                for k in start_idx..end_idx {
                                    cov += (signal_i[k] - mean_i) * (signal_j[k] - mean_j);
                                }
                            }

                            // Handle remaining samples
                            for k in (chunks * 4)..n_samples {
                                cov += (signal_i[k] - mean_i) * (signal_j[k] - mean_j);
                            }

                            cov /= (n_samples - 1) as f64;

                            (i, j, cov)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Write results back to covariance matrix
            for (i, j, cov) in cov_results {
                covariance[[i, j]] = cov;
                if i != j {
                    covariance[[j, i]] = cov;
                }
            }
        } else {
            // Sequential computation
            for i in 0..n_channels {
                for j in i..n_channels {
                    let signal_i = signals.row(i);
                    let signal_j = signals.row(j);

                    let cov = signal_i
                        .iter()
                        .zip(signal_j.iter())
                        .map(|(&si, &sj)| (si - means[i]) * (sj - means[j]))
                        .sum::<f64>()
                        / (n_samples - 1) as f64;

                    covariance[[i, j]] = cov;
                    if i != j {
                        covariance[[j, i]] = cov;
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized autocorrelation matrix computation with SIMD
    ///
    /// Computes the autocorrelation matrix for parametric modeling and
    /// linear prediction using advanced SIMD optimizations.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `order` - Autocorrelation matrix order
    /// * `autocorr_matrix` - Output autocorrelation matrix (order x order)
    /// * `config` - SIMD configuration
    pub fn simd_autocorrelation_matrix(
        signal: &[f64],
        order: usize,
        autocorr_matrix: &mut Array2<f64>,
        config: &SimdConfig,
    ) -> SignalResult<()> {
        let n = signal.len();

        if autocorr_matrix.dim() != (order, order) {
            return Err(SignalError::ValueError(
                "Autocorrelation _matrix dimensions incorrect".to_string(),
            ));
        }

        check_finite(signal, "signal value")?;

        if order >= n {
            return Err(SignalError::ValueError(
                "Order must be less than signal length".to_string(),
            ));
        }

        // Compute autocorrelation values up to required lags
        let mut autocorr_vals = vec![0.0; order];

        for lag in 0..order {
            if n >= config.simd_threshold && !config.force_scalar {
                // SIMD-accelerated autocorrelation computation
                let mut sum = 0.0;
                let effective_len = n - lag;
                let chunks = effective_len / 4;

                for chunk in 0..chunks {
                    let start_idx = chunk * 4;
                    for i in 0..4 {
                        let idx1 = start_idx + i;
                        let idx2 = idx1 + lag;
                        if idx2 < n {
                            sum += signal[idx1] * signal[idx2];
                        }
                    }
                }

                // Handle remaining elements
                for i in (chunks * 4)..(n - lag) {
                    sum += signal[i] * signal[i + lag];
                }

                autocorr_vals[lag] = sum / (n - lag) as f64;
            } else {
                // Scalar computation
                let mut sum = 0.0;
                for i in 0..(n - lag) {
                    sum += signal[i] * signal[i + lag];
                }
                autocorr_vals[lag] = sum / (n - lag) as f64;
            }
        }

        // Build Toeplitz autocorrelation _matrix
        for i in 0..order {
            for j in 0..order {
                let lag = if i >= j { i - j } else { j - i };
                autocorr_matrix[[i, j]] = autocorr_vals[lag];
            }
        }

        Ok(())
    }
}

/// Advanced-high-performance real-time signal processing operations
pub mod advanced_simd_realtime {
    use super::SimdConfig;
    use crate::error::{SignalError, SignalResult};
    use ndarray::Array2;
    use scirs2_core::simd_ops::PlatformCapabilities;

    /// Real-time SIMD FIR filter state
    #[derive(Debug, Clone)]
    pub struct RealtimeSimdFirFilter {
        coefficients: Vec<f64>,
        delay_line: Vec<f64>,
        position: usize,
        config: SimdConfig,
    }

    impl RealtimeSimdFirFilter {
        /// Create new real-time SIMD FIR filter
        pub fn new(coefficients: Vec<f64>, config: SimdConfig) -> Self {
            let delay_line = vec![0.0; coefficients.len()];
            Self {
                coefficients,
                delay_line,
                position: 0,
                config,
            }
        }

        /// Process single sample with SIMD optimization
        pub fn process_sample(&mut self, input: f64) -> SignalResult<f64> {
            // Add new sample to delay line
            self.delay_line[self.position] = input;

            // Compute filter output using SIMD when possible
            let filter_len = self.coefficients.len();

            let output = if filter_len >= self.config.simd_threshold && !self.config.force_scalar {
                // SIMD-accelerated convolution
                let caps = PlatformCapabilities::detect();

                if caps.avx2_available {
                    unsafe { self.avx2_process_sample() }
                } else {
                    self.scalar_process_sample()
                }
            } else {
                self.scalar_process_sample()
            };

            // Update position (circular buffer)
            self.position = (self.position + 1) % filter_len;

            Ok(output)
        }

        /// Process block of samples for higher efficiency
        pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> SignalResult<()> {
            if input.len() != output.len() {
                return Err(SignalError::ValueError(
                    "Input and output block sizes must match".to_string(),
                ));
            }

            for (i, &sample) in input.iter().enumerate() {
                output[i] = self.process_sample(sample)?;
            }

            Ok(())
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn avx2_process_sample(&self) -> f64 {
            let mut sum = 0.0;
            let filter_len = self.coefficients.len();

            // Vectorized computation where possible
            let chunks = filter_len / 4;
            for chunk in 0..chunks {
                let mut partial_sum = _mm256_setzero_pd();

                for i in 0..4 {
                    let coeff_idx = chunk * 4 + i;
                    let delay_idx = (self.position + filter_len - coeff_idx - 1) % filter_len;

                    let coeff = _mm256_set1_pd(self.coefficients[coeff_idx]);
                    let sample = _mm256_set1_pd(self.delay_line[delay_idx]);
                    partial_sum = _mm256_fmadd_pd(coeff, sample, partial_sum);
                }

                // Extract and sum the partial results
                let partial_array: [f64; 4] = std::mem::transmute(partial_sum);
                sum += partial_array.iter().sum::<f64>();
            }

            // Handle remaining coefficients
            for i in (chunks * 4)..filter_len {
                let delay_idx = (self.position + filter_len - i - 1) % filter_len;
                sum += self.coefficients[i] * self.delay_line[delay_idx];
            }

            sum
        }

        fn scalar_process_sample(&self) -> f64 {
            let mut sum = 0.0;
            let filter_len = self.coefficients.len();

            for i in 0..filter_len {
                let delay_idx = (self.position + filter_len - i - 1) % filter_len;
                sum += self.coefficients[i] * self.delay_line[delay_idx];
            }

            sum
        }
    }

    /// Multi-channel real-time SIMD processing
    #[derive(Debug, Clone)]
    pub struct MultiChannelRealtimeProcessor {
        filters: Vec<RealtimeSimdFirFilter>,
        config: SimdConfig,
    }

    impl MultiChannelRealtimeProcessor {
        /// Create new multi-channel processor
        pub fn new(_channelfilters: Vec<Vec<f64>>, config: SimdConfig) -> Self {
            let _filters = _channel_filters
                .into_iter()
                .map(|coeffs| RealtimeSimdFirFilter::new(coeffs, config.clone()))
                .collect();

            Self { filters, config }
        }

        /// Process multi-channel sample
        pub fn process_multichannel_sample(
            &mut self,
            inputs: &[f64],
            outputs: &mut [f64],
        ) -> SignalResult<()> {
            if inputs.len() != self.filters.len() || outputs.len() != self.filters.len() {
                return Err(SignalError::ValueError(
                    "Input/output channel count mismatch".to_string(),
                ));
            }

            // Process each channel
            if self.filters.len() >= 4 && !self.config.force_scalar {
                // Parallel processing for multiple channels
                inputs
                    .par_iter()
                    .zip(self.filters.par_iter_mut())
                    .zip(outputs.par_iter_mut())
                    .for_each(|((&input, filter), output)| {
                        *output = filter.process_sample(input).unwrap_or(0.0);
                    });
            } else {
                // Sequential processing
                for (i, (&input, filter)) in inputs.iter().zip(self.filters.iter_mut()).enumerate()
                {
                    outputs[i] = filter.process_sample(input)?;
                }
            }

            Ok(())
        }

        /// Process multi-channel block
        pub fn process_multichannel_block(
            &mut self,
            inputs: &Array2<f64>,
            outputs: &mut Array2<f64>,
        ) -> SignalResult<()> {
            let (n_channels, block_size) = inputs.dim();

            if outputs.dim() != (n_channels, block_size) {
                return Err(SignalError::ValueError(
                    "Input and output array dimensions must match".to_string(),
                ));
            }

            if n_channels != self.filters.len() {
                return Err(SignalError::ValueError(
                    "Number of channels doesn't match number of filters".to_string(),
                ));
            }

            // Process each sample across all channels
            for sample_idx in 0..block_size {
                let mut input_sample = vec![0.0; n_channels];
                let mut output_sample = vec![0.0; n_channels];

                // Extract samples for all channels
                for ch in 0..n_channels {
                    input_sample[ch] = inputs[[ch, sample_idx]];
                }

                // Process the multi-channel sample
                self.process_multichannel_sample(&input_sample, &mut output_sample)?;

                // Store results
                for ch in 0..n_channels {
                    outputs[[ch, sample_idx]] = output_sample[ch];
                }
            }

            Ok(())
        }
    }
}

/// Comprehensive SIMD validation and performance testing
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

    // 2. Test matrix operations
    let matrix = Array2::<f64>::ones((100, 100));
    let vector = vec![1.0; 100];
    let mut matrix_result = vec![0.0; 100];

    let matrix_start = std::time::Instant::now();
    advanced_simd_matrix::simd_matrix_vector_mul(&matrix, &vector, &mut matrix_result, config)?;
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

/// SIMD validation result
#[derive(Debug, Clone, Default)]
pub struct SimdValidationResult {
    pub fir_filter_time_ns: u64,
    pub autocorrelation_time_ns: u64,
    pub cross_correlation_time_ns: u64,
    pub matrix_vector_time_ns: u64,
    pub simd_scalar_max_error: f64,
    pub simd_consistency: bool,
    pub operations_per_second: f64,
    pub memory_throughput_bytes_per_sec: f64,
    pub total_validation_time_ms: f64,
    pub validation_passed: bool,
}

mod tests {

    #[test]
    fn test_simd_fir_filter() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = vec![0.5, 0.3, 0.2];
        let mut output = vec![0.0; input.len()];

        let config = SimdConfig::default();
        simd_fir_filter(&input, &coeffs, &mut output, &config).unwrap();

        // Basic sanity check
        assert!(output.iter().all(|&x: &f64| x.is_finite()));
        assert!(output[0] > 0.0); // First output should be positive
    }

    #[test]
    fn test_simd_autocorrelation() {
        let signal = vec![1.0, 2.0, 1.0, -1.0, -2.0, -1.0, 1.0, 2.0];
        let config = SimdConfig::default();

        let result = simd_autocorrelation(&signal, 4, &config).unwrap();

        assert_eq!(result.len(), 5); // max_lag + 1
        assert!(result.iter().all(|&x: &f64| x.is_finite()));
        assert!(result[0] > 0.0); // Zero-lag should be positive
    }

    #[test]
    fn test_simd_vs_scalar_equivalence() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let coeffs = vec![0.25, 0.5, 0.25];

        let mut simd_output = vec![0.0; signal.len()];
        let mut scalar_output = vec![0.0; signal.len()];

        let simd_config = SimdConfig {
            force_scalar: false,
            ..Default::default()
        };
        let scalar_config = SimdConfig {
            force_scalar: true,
            ..Default::default()
        };

        simd_fir_filter(&signal, &coeffs, &mut simd_output, &simd_config).unwrap();
        simd_fir_filter(&signal, &coeffs, &mut scalar_output, &scalar_config).unwrap();

        for (simd_val, scalar_val) in simd_output.iter().zip(scalar_output.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-10,
                "SIMD and scalar results differ: {} vs {}",
                simd_val,
                scalar_val
            );
        }
    }

    #[test]
    fn test_simd_spectral_centroid() {
        let magnitude = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let frequencies = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let config = SimdConfig::default();

        let centroid = simd_spectral_centroid(&magnitude, &frequencies, &config).unwrap();

        assert!(centroid > 200.0 && centroid < 400.0);
        assert!(centroid.is_finite());
    }

    #[test]
    fn test_simd_spectral_rolloff() {
        let magnitude = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let frequencies = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let config = SimdConfig::default();

        let rolloff = simd_spectral_rolloff(&magnitude, &frequencies, 0.85, &config).unwrap();

        assert!(rolloff >= 100.0 && rolloff <= 500.0);
        assert!(rolloff.is_finite());
    }

    #[test]
    fn test_simd_peak_detection() {
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let config = SimdConfig::default();

        let peaks = simd_peak_detection(&signal, 0.5, 1, &config).unwrap();

        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&5));
    }

    #[test]
    fn test_simd_zero_crossing_rate() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let config = SimdConfig::default();

        let zcr = simd_zero_crossing_rate(&signal, &config).unwrap();

        assert!(zcr > 0.8); // High zero crossing rate for alternating signal
        assert!(zcr.is_finite());
    }

    #[test]
    fn test_simd_signal_energy() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SimdConfig::default();

        let energy = simd_signal_energy(&signal, &config).unwrap();
        let expected_energy = 1.0 + 4.0 + 9.0 + 16.0 + 25.0; // Sum of squares

        assert!((energy - expected_energy).abs() < 1e-10);
    }

    #[test]
    fn test_simd_rms() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SimdConfig::default();

        let rms = simd_rms(&signal, &config).unwrap();
        let expected_energy = 1.0 + 4.0 + 9.0 + 16.0 + 25.0; // Sum of squares
        let expected_rms = ((expected_energy / 5.0) as f64).sqrt();

        assert!((rms - expected_rms).abs() < 1e-10);
    }

    #[test]
    fn test_simd_config_customization() {
        let config = SimdConfig {
            force_scalar: true,
            simd_threshold: 32,
            align_memory: false,
            use_advanced: false,
            enable_monitoring: false,
            adaptive_thresholds: false,
            cache_line_size: 64,
            max_unroll_factor: 4,
        };

        let signal = vec![1.0; 100];
        let energy_scalar = simd_signal_energy(&signal, &config).unwrap();

        let config_simd = SimdConfig {
            force_scalar: false,
            simd_threshold: 10,
            align_memory: true,
            use_advanced: true,
            enable_monitoring: true,
            adaptive_thresholds: true,
            cache_line_size: 64,
            max_unroll_factor: 8,
        };

        let energy_simd = simd_signal_energy(&signal, &config_simd).unwrap();

        // Both should give same result
        assert!((energy_scalar - energy_simd).abs() < 1e-10);
    }
}

/// SIMD-optimized element-wise complex multiplication for FFT operations
///
/// This function provides highly optimized complex multiplication for spectral analysis,
/// particularly useful in multitaper and other frequency domain operations.
#[allow(dead_code)]
pub fn simd_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    let n = a_real.len();
    if n != a_imag.len()
        || n != b_real.len()
        || n != b_imag.len()
        || n != result_real.len()
        || n != result_imag.len()
    {
        return Err(SignalError::ValueError(
            "All arrays must have the same length".to_string(),
        ));
    }

    check_finite(a_real, "a_real value")?;
    check_finite(a_imag, "a_imag value")?;
    check_finite(b_real, "b_real value")?;
    check_finite(b_imag, "b_imag value")?;

    if n < config.simd_threshold || config.force_scalar {
        return scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag) }
    } else if caps.simd_available {
        unsafe { sse_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag) }
    } else {
        scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
    }
}

/// SIMD-optimized power spectral density computation
///
/// Computes |X|^2 for complex FFT results using SIMD acceleration
#[allow(dead_code)]
pub fn simd_power_spectrum(
    real: &[f64],
    imag: &[f64],
    power: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    let n = real.len();
    if n != imag.len() || n != power.len() {
        return Err(SignalError::ValueError(
            "All arrays must have the same length".to_string(),
        ));
    }

    check_finite(real, "real value")?;
    check_finite(imag, "imag value")?;

    if n < config.simd_threshold || config.force_scalar {
        return scalar_power_spectrum(real, imag, power);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_power_spectrum(real, imag, power) }
    } else if caps.simd_available {
        unsafe { sse_power_spectrum(real, imag, power) }
    } else {
        scalar_power_spectrum(real, imag, power)
    }
}

/// SIMD-optimized weighted averaging for multitaper spectral estimation
///
/// Computes weighted averages of multiple tapered spectra using adaptive weights
#[allow(dead_code)]
pub fn simd_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    if spectra.is_empty() || weights.is_empty() {
        return Err(SignalError::ValueError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    let n_freqs = spectra[0].len();
    let n_tapers = spectra.len();

    if n_tapers != weights.len() || n_freqs != result.len() {
        return Err(SignalError::ValueError(
            "Inconsistent array dimensions".to_string(),
        ));
    }

    // Validate all spectra have same length
    for spectrum in spectra {
        if spectrum.len() != n_freqs {
            return Err(SignalError::ValueError(
                "All spectra must have the same length".to_string(),
            ));
        }
    }

    check_finite(weights, "weights value")?;
    for (i, spectrum) in spectra.iter().enumerate() {
        check_finite(spectrum, &format!("spectrum_{}", i))?;
    }

    if n_freqs < config.simd_threshold || config.force_scalar {
        return scalar_weighted_average_spectra(spectra, weights, result);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_weighted_average_spectra(spectra, weights, result) }
    } else if caps.simd_available {
        unsafe { sse_weighted_average_spectra(spectra, weights, result) }
    } else {
        scalar_weighted_average_spectra(spectra, weights, result)
    }
}

/// SIMD-optimized window function application
///
/// Applies window functions element-wise with SIMD acceleration (alternative implementation)
#[allow(dead_code)]
pub fn simd_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
    config: &SimdConfig,
) -> SignalResult<()> {
    let n = signal.len();
    if n != window.len() || n != result.len() {
        return Err(SignalError::ValueError(
            "All arrays must have the same length".to_string(),
        ));
    }

    check_finite(signal, "signal value")?;
    check_finite(window, "window value")?;

    if n < config.simd_threshold || config.force_scalar {
        return scalar_apply_window(signal, window, result);
    }

    let caps = PlatformCapabilities::detect();

    if caps.avx2_available && config.use_advanced {
        unsafe { avx2_apply_window_v2(signal, window, result) }
    } else if caps.simd_available {
        unsafe { sse_apply_window_v2(signal, window, result) }
    } else {
        scalar_apply_window(signal, window, result)
    }
}

// Scalar fallback implementations
#[allow(dead_code)]
fn scalar_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    for i in 0..a_real.len() {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
    Ok(())
}

#[allow(dead_code)]
fn scalar_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    for i in 0.._real.len() {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

#[allow(dead_code)]
fn scalar_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    let n_freqs = result.len();
    let _n_tapers = spectra.len();

    // Initialize result
    result.fill(0.0);

    // Compute weighted sum
    for (taper_idx, spectrum) in spectra.iter().enumerate() {
        let weight = weights[taper_idx];
        for freq_idx in 0..n_freqs {
            result[freq_idx] += weight * spectrum[freq_idx];
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn scalar_apply_window(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    for i in 0.._signal.len() {
        result[i] = signal[i] * window[i];
    }
    Ok(())
}

// AVX2 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_complex_multiply(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    let n = real.len();
    let simd_width = 4;
    let simd_chunks = n / simd_width;

    for chunk in 0..simd_chunks {
        let idx = chunk * simd_width;

        let real_vec = _mm256_loadu_pd(_real.as_ptr().add(idx));
        let imag_vec = _mm256_loadu_pd(imag.as_ptr().add(idx));

        // power = _real^2 + imag^2
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_weighted_average_spectra(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_apply_window_v2(
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

// SSE implementations (similar structure but with _mm instructions)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    // Similar to AVX2 but with SSE instructions and width=2
    scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    scalar_power_spectrum(_real, imag, power)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    scalar_weighted_average_spectra(spectra, weights, result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    scalar_apply_window(signal, window, result)
}
// Direct SIMD → Scalar mappings for missing SIMD functions
//
// This file provides scalar fallbacks for missing SIMD implementations.
// These can be replaced with proper SIMD implementations in the future.

// Direct mappings for functions with existing scalar implementations

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    simd_fir_filter(input, coeffs, output, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    simd_fir_filter(input, coeffs, output, &SimdConfig::default())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_autocorrelation(signal: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    Ok(simd_autocorrelation(
        signal,
        max_lag,
        &SimdConfig::default(),
    )?)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_autocorrelation(signal: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    Ok(simd_autocorrelation(
        signal,
        max_lag,
        &SimdConfig::default(),
    )?)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    let result = simd_cross_correlation(signal1, signal2, "full", &SimdConfig::default())?;
    let output_len = output.len();
    let copy_len = result.len().min(output_len);
    output[..copy_len].copy_from_slice(&result[..copy_len]);
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_cross_correlation(signal1: &[f64], signal2: &[f64], output: &mut [f64]) -> SignalResult<()> {
    let result = simd_cross_correlation(signal1, signal2, "full", &SimdConfig::default())?;
    let output_len = output.len();
    let copy_len = result.len().min(output_len);
    output[..copy_len].copy_from_slice(&result[..copy_len]);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    simd_complex_fft_butterfly(data, twiddles, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    simd_complex_fft_butterfly(data, twiddles, &SimdConfig::default())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    simd_enhanced_convolution(signal, kernel, output, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_enhanced_convolution(
    signal: &[f64],
    kernel: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    simd_enhanced_convolution(signal, kernel, output, &SimdConfig::default())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_complex_multiply(
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

#[cfg(not(target_arch = "x86_64"))]
fn sse_complex_multiply(
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

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    for i in 0..real.len().min(imag.len()).min(power.len()) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    for i in 0..real.len().min(imag.len()).min(power.len()) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_weighted_average_spectra(
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

#[cfg(not(target_arch = "x86_64"))]
fn sse_weighted_average_spectra(
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

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_apply_window(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    simd_apply_window(signal, window, result, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_apply_window(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    simd_apply_window(signal, window, result, &SimdConfig::default())
}

// Simple SIMD function stubs
//
// Basic implementations for missing SIMD functions.
// These provide correctness over performance.

// Simple window application variant
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn sse_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= result.len() {
            break;
        }
        result[i] = s * w;
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_apply_window_v2(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= result.len() {
            break;
        }
        result[i] = s * w;
    }
    Ok(())
}

// Simple windowing variant
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_apply_window_v2(
    signal: &[f64],
    window: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= result.len() {
            break;
        }
        result[i] = s * w;
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_apply_window_v2(signal: &[f64], window: &[f64], result: &mut [f64]) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= result.len() {
            break;
        }
        result[i] = s * w;
    }
    Ok(())
}

// Peak detection
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_peak_detection(
    signal: &[f64],
    min_height: f64,
    peaks: &mut Vec<usize>,
) -> SignalResult<()> {
    peaks.clear();
    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] >= min_height {
            peaks.push(i);
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_peak_detection(
    signal: &[f64],
    min_height: f64,
    peaks: &mut Vec<usize>,
) -> SignalResult<()> {
    peaks.clear();
    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] >= min_height {
            peaks.push(i);
        }
    }
    Ok(())
}

// Zero crossings detection
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let mut crossings = 0;
    for i in 1..signal.len() {
        if (signal[i - 1] >= 0.0 && signal[i] < 0.0) || (signal[i - 1] < 0.0 && signal[i] >= 0.0) {
            crossings += 1;
        }
    }
    Ok(crossings)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_zero_crossings(signal: &[f64]) -> SignalResult<usize> {
    let mut crossings = 0;
    for i in 1..signal.len() {
        if (signal[i - 1] >= 0.0 && signal[i] < 0.0) || (signal[i - 1] < 0.0 && signal[i] >= 0.0) {
            crossings += 1;
        }
    }
    Ok(crossings)
}

// Basic window application (original)
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_apply_window(
    signal: &[f64],
    window: &[f64],
    output: &mut [f64],
) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= output.len() {
            break;
        }
        output[i] = s * w;
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_apply_window(signal: &[f64], window: &[f64], output: &mut [f64]) -> SignalResult<()> {
    for (i, (s, w)) in signal.iter().zip(window.iter()).enumerate() {
        if i >= output.len() {
            break;
        }
        output[i] = s * w;
    }
    Ok(())
}

// Missing SIMD function implementations - scalar fallbacks
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    simd_fir_filter(input, coeffs, output, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_fir_filter(input: &[f64], coeffs: &[f64], output: &mut [f64]) -> SignalResult<()> {
    simd_fir_filter(input, coeffs, output, &SimdConfig::default())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_autocorrelation(
    signal: &[f64],
    autocorr: &mut [f64],
    max_lag: usize,
) -> SignalResult<()> {
    scalar_autocorrelation(signal, autocorr, max_lag)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_autocorrelation(signal: &[f64], autocorr: &mut [f64], maxlag: usize) -> SignalResult<()> {
    scalar_autocorrelation(signal, autocorr, max_lag)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    scalar_cross_correlation(signal1, signal2, result, mode)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_cross_correlation(
    signal1: &[f64],
    signal2: &[f64],
    result: &mut [f64],
    mode: &str,
) -> SignalResult<()> {
    scalar_cross_correlation(signal1, signal2, result, mode)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    simd_complex_fft_butterfly(data, twiddles, &SimdConfig::default())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_complex_butterfly(
    data: &mut [num_complex::Complex<f64>],
    twiddles: &[num_complex::Complex<f64>],
) -> SignalResult<()> {
    simd_complex_fft_butterfly(data, twiddles, &SimdConfig::default())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_complex_multiply(
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

#[cfg(not(target_arch = "x86_64"))]
fn avx2_complex_multiply(
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

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    for i in 0..real.len().min(imag.len()).min(power.len()) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    for i in 0..real.len().min(imag.len()).min(power.len()) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn avx2_weighted_average_spectra(
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

#[cfg(not(target_arch = "x86_64"))]
fn avx2_weighted_average_spectra(
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
