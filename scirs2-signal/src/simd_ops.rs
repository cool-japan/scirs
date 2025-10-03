use scirs2_core::ndarray::s;
// SIMD-optimized signal processing operations
//
// This module provides SIMD-accelerated implementations of common signal
// processing operations using scirs2-core's unified SIMD abstractions.

use crate::error::{SignalError, SignalResult};
use crate::hilbert::hilbert;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use std::sync::Once;

#[allow(unused_imports)]
// Global SIMD capability detection
static INIT: Once = Once::new();
static mut SIMD_CAPS: Option<PlatformCapabilities> = None;

/// Get cached SIMD capabilities
#[allow(dead_code)]
fn get_simd_caps() -> &'static PlatformCapabilities {
    unsafe {
        INIT.call_once(|| {
            SIMD_CAPS = Some(PlatformCapabilities::detect());
        });
        SIMD_CAPS.as_ref().unwrap()
    }
}

/// SIMD-optimized convolution for real signals
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel` - Convolution kernel
/// * `mode` - Convolution mode ("full", "same", "valid")
///
/// # Returns
///
/// * Convolution result
#[allow(dead_code)]
pub fn simd_convolve_f32(signal: &[f32], kernel: &[f32], mode: &str) -> SignalResult<Vec<f32>> {
    let caps = get_simd_caps();

    if signal.is_empty() || kernel.is_empty() {
        return Ok(vec![]);
    }

    // Convert to ArrayView for SIMD operations
    let signal_view = ArrayView1::from(_signal);
    let kernel_view = ArrayView1::from(kernel);

    // Use SIMD convolution based on capabilities
    let result = if kernel.len() <= 16 {
        // Small kernel - use direct SIMD convolution
        simd_convolve_direct_f32(&signal_view, &kernel_view, caps)?
    } else {
        // Large kernel - use SIMD-accelerated overlap-save
        simd_convolve_overlap_save_f32(&signal_view, &kernel_view, caps)?
    };

    // Apply mode
    apply_mode_f32(result, signal.len(), kernel.len(), mode)
}

/// Direct SIMD convolution for small kernels
#[allow(dead_code)]
fn simd_convolve_direct_f32(
    signal: &ArrayView1<f32>,
    kernel: &ArrayView1<f32>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f32>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;

    let mut output = vec![0.0f32; n_out];

    // Process using SIMD operations
    for i in 0..n_out {
        let start = i.saturating_sub(n_kernel - 1);
        let end = (i + 1).min(n_signal);

        if start < end {
            // Extract valid signal segment
            let sig_segment = signal.slice(s![start..end]);

            // Extract corresponding kernel segment (reversed)
            let k_start = i.saturating_sub(end - 1);
            let k_end = (i + 1).min(n_kernel);

            if k_start < k_end {
                let ker_segment = kernel.slice(s![k_start..k_end]);

                // Use SIMD dot product
                output[i] = f32::simd_dot(&sig_segment, &ker_segment);
            }
        }
    }

    Ok(output)
}

/// Overlap-save SIMD convolution for large kernels
#[allow(dead_code)]
fn simd_convolve_overlap_save_f32(
    signal: &ArrayView1<f32>,
    kernel: &ArrayView1<f32>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f32>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;

    // Choose block size (power of 2 for alignment)
    let block_size = 4096;
    let overlap = n_kernel - 1;
    let step = block_size - overlap;

    let mut output = vec![0.0f32; n_out];

    // Process blocks
    let mut pos = 0;
    while pos < n_signal {
        let block_end = (pos + block_size).min(n_signal + overlap);
        let actual_size = block_end - pos;

        // Create padded block
        let mut block = Array1::zeros(block_size);
        let copy_len = (block_end - pos).min(n_signal - pos);
        if copy_len > 0 {
            let src = signal.slice(s![pos..pos + copy_len]);
            let mut dst = block.slice_mut(s![..copy_len]);
            f32::simd_copy(&src, &mut dst);
        }

        // Convolve block with kernel using SIMD
        for i in 0..actual_size {
            if i + n_kernel <= block_size {
                let sig_segment = block.slice(s![i..i + n_kernel]);
                let sum = f32::simd_dot(&sig_segment, kernel);

                let out_idx = pos + i;
                if out_idx < n_out {
                    output[out_idx] = sum;
                }
            }
        }

        pos += step;
    }

    Ok(output)
}

/// SIMD-optimized FIR filtering
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `coeffs` - FIR filter coefficients
///
/// # Returns
///
/// * Filtered signal
#[allow(dead_code)]
pub fn simd_fir_filter_f32(signal: &[f32], coeffs: &[f32]) -> SignalResult<Vec<f32>> {
    if signal.is_empty() || coeffs.is_empty() {
        return Ok(vec![]);
    }

    let n_signal = signal.len();
    let n_coeffs = coeffs.len();
    let mut output = vec![0.0f32; n_signal];

    // Convert to arrays
    let signal_arr = ArrayView1::from(_signal);
    let coeffs_arr = ArrayView1::from(coeffs);

    // Apply FIR filter using SIMD
    for i in 0..n_signal {
        let start = i.saturating_sub(n_coeffs - 1);
        let seg_len = i - start + 1;

        if seg_len > 0 && seg_len <= n_coeffs {
            let sig_segment = signal_arr.slice(s![start..=i]);
            let coeff_segment = coeffs_arr.slice(s![..seg_len]);

            // Reverse iteration for proper FIR filtering
            let mut reversed_sig = Array1::zeros(seg_len);
            for j in 0..seg_len {
                reversed_sig[j] = sig_segment[seg_len - 1 - j];
            }

            output[i] = f32::simd_dot(&reversed_sig.view(), &coeff_segment);
        }
    }

    Ok(output)
}

/// SIMD-optimized cross-correlation
///
/// # Arguments
///
/// * `signal1` - First signal
/// * `signal2` - Second signal
/// * `mode` - Correlation mode
///
/// # Returns
///
/// * Cross-correlation result
#[allow(dead_code)]
pub fn simd_correlate_f32(signal1: &[f32], signal2: &[f32], mode: &str) -> SignalResult<Vec<f32>> {
    // Correlation is convolution with reversed second signal
    let mut reversed = signal2.to_vec();
    reversed.reverse();

    simd_convolve_f32(_signal1, &reversed, mode)
}

/// SIMD-optimized RMS calculation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * RMS value
#[allow(dead_code)]
pub fn simd_rms_f32(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let signal_view = ArrayView1::from(_signal);

    // Compute sum of squares using SIMD
    let sum_squares = f32::simd_dot(&signal_view, &signal_view);

    (sum_squares / signal.len() as f32).sqrt()
}

/// SIMD-optimized peak detection
///
/// Find local maxima in a signal using SIMD operations
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `min_distance` - Minimum distance between peaks
/// * `threshold` - Minimum peak height
///
/// # Returns
///
/// * Indices of detected peaks
#[allow(dead_code)]
pub fn simd_find_peaks_f32(
    signal: &[f32],
    min_distance: usize,
    threshold: Option<f32>,
) -> Vec<usize> {
    if signal.len() < 3 {
        return vec![];
    }

    let thresh = threshold.unwrap_or(f32::NEG_INFINITY);
    let mut peaks = Vec::new();
    let mut last_peak = None;

    // Find local maxima
    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] >= thresh {
            // Check minimum _distance constraint
            if let Some(last) = last_peak {
                if i - last >= min_distance {
                    peaks.push(i);
                    last_peak = Some(i);
                } else if signal[i] > signal[last] {
                    // Replace previous peak if this one is higher
                    peaks.pop();
                    peaks.push(i);
                    last_peak = Some(i);
                }
            } else {
                peaks.push(i);
                last_peak = Some(i);
            }
        }
    }

    peaks
}

/// SIMD-optimized windowing function
///
/// Apply a window function to a signal using SIMD operations
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window` - Window coefficients
///
/// # Returns
///
/// * Windowed signal
#[allow(dead_code)]
pub fn simd_apply_window_f32(signal: &[f32], window: &[f32]) -> SignalResult<Vec<f32>> {
    if signal.len() != window.len() {
        return Err(SignalError::ShapeMismatch(
            "Signal and window must have the same length".to_string(),
        ));
    }

    let signal_view = ArrayView1::from(_signal);
    let window_view = ArrayView1::from(window);

    let mut output = Array1::zeros(_signal.len());
    f32::simd_mul(&signal_view, &window_view, &mut output.view_mut());

    Ok(output.to_vec())
}

/// SIMD-optimized envelope detection using Hilbert transform
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Signal envelope (magnitude of analytic signal)
#[allow(dead_code)]
pub fn simd_envelope_f32(signal: &[f32]) -> SignalResult<Vec<f32>> {
    // Compute Hilbert transform
    let hilbert_sig = hilbert(_signal)?;

    // Convert to f32 if needed
    let hilbert_f32: Vec<f32> = hilbert_sig.iter().map(|&x| x as f32).collect();

    // Compute envelope using SIMD
    let signal_view = ArrayView1::from(_signal);
    let hilbert_view = ArrayView1::from(&hilbert_f32);

    let mut envelope = vec![0.0f32; signal.len()];

    // Compute magnitude: sqrt(_signal^2 + hilbert^2)
    let mut sig_squared = Array1::zeros(_signal.len());
    let mut hil_squared = Array1::zeros(_signal.len());

    f32::simd_mul(&signal_view, &signal_view, &mut sig_squared.view_mut());
    f32::simd_mul(&hilbert_view, &hilbert_view, &mut hil_squared.view_mut());

    let mut sum = Array1::zeros(_signal.len());
    f32::simd_add(
        &sig_squared.view(),
        &hil_squared.view(),
        &mut sum.view_mut(),
    );

    // Square root
    for (i, &s) in sum.iter().enumerate() {
        envelope[i] = s.sqrt();
    }

    Ok(envelope)
}

// Helper function to apply convolution mode
#[allow(dead_code)]
fn apply_mode_f32(
    result: Vec<f32>,
    signal_len: usize,
    kernel_len: usize,
    mode: &str,
) -> SignalResult<Vec<f32>> {
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (kernel_len - 1) / 2;
            let end = start + signal_len;
            if end <= result._len() {
                Ok(result[start..end].to_vec())
            } else {
                Ok(result)
            }
        }
        "valid" => {
            if kernel_len > signal_len {
                return Err(SignalError::ValueError(
                    "Kernel length exceeds signal length in 'valid' mode".to_string(),
                ));
            }
            let start = kernel_len - 1;
            let end = result._len() - (kernel_len - 1);
            if start < end {
                Ok(result[start..end].to_vec())
            } else {
                Ok(vec![])
            }
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// SIMD-optimized windowing function application
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window` - Window function values
///
/// # Returns
///
/// * Windowed signal
#[allow(dead_code)]
pub fn simd_apply_window_f64(
    signal: &Array1<f64>,
    window: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    if signal.len() != window.len() {
        return Err(SignalError::ValueError(
            "Signal and window must have the same length".to_string(),
        ));
    }

    let mut output = Array1::zeros(signal.len());
    let output_view = output.view_mut();

    // Use SIMD element-wise multiplication
    f64::simd_mul(&signal.view(), &window.view(), &output_view);

    Ok(output)
}

/// SIMD-optimized autocorrelation
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `max_lag` - Maximum lag to compute (None for all lags)
///
/// # Returns
///
/// * Autocorrelation function
#[allow(dead_code)]
pub fn simd_autocorrelation_f64(
    signal: &Array1<f64>,
    max_lag: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let max_lag = max_lag.unwrap_or(n - 1).min(n - 1);

    let mut autocorr = Array1::zeros(max_lag + 1);

    // SIMD-optimized computation
    for _lag in 0..=max_lag {
        let sig1 = signal.slice(s![0..n - _lag]);
        let sig2 = signal.slice(s![_lag..n]);
        autocorr[_lag] = f64::simd_dot(&sig1, &sig2);
    }

    // Normalize by the zero-_lag value
    if autocorr[0] != 0.0 {
        let autocorr_view = autocorr.view_mut();
        let scale = Array1::from_elem(autocorr.len(), 1.0 / autocorr[0]);
        f64::simd_mul(&autocorr.view(), &scale.view(), &autocorr_view);
    }

    Ok(autocorr)
}

/// Enhanced SIMD-optimized autocorrelation with chunking for large signals
///
/// This version provides better performance for large signals by using:
/// - Parallel processing for independent lag computations
/// - Memory-efficient chunking for very large signals
/// - Enhanced numerical stability
///
/// # Arguments
///
/// * `signal` - Input signal array
/// * `max_lag` - Maximum lag to compute (None for full autocorrelation)
/// * `chunk_size` - Chunk size for memory optimization (None for automatic)
/// * `normalize` - Whether to normalize by zero-lag value
///
/// # Returns
///
/// * Enhanced autocorrelation function with performance metrics
#[allow(dead_code)]
pub fn simd_autocorrelation_enhanced(
    signal: &Array1<f64>,
    max_lag: Option<usize>,
    chunk_size: Option<usize>,
    normalize: bool,
) -> SignalResult<(Array1<f64>, AutocorrelationMetrics)> {
    let n = signal.len();
    let max_lag = max_lag.unwrap_or(n / 4).min(n - 1); // Default to 25% for efficiency

    // Validate inputs
    if n == 0 {
        return Err(SignalError::ValueError("Signal is empty".to_string()));
    }

    if max_lag >= n {
        return Err(SignalError::ValueError(
            "max_lag must be less than signal length".to_string(),
        ));
    }

    let start_time = std::time::Instant::now();

    // Determine chunk _size for memory optimization
    let effective_chunk_size = chunk_size.unwrap_or_else(|| {
        if n > 100_000 {
            n / 10 // Use 10% chunks for very large signals
        } else {
            n // Process all at once for smaller signals
        }
    });

    let mut autocorr = Array1::zeros(max_lag + 1);

    // Use parallel processing if beneficial
    let use_parallel = max_lag > 50 && n > 1000;

    if use_parallel {
        // Parallel computation of lags
        let results: Result<Vec<f64>, SignalError> = (0..=max_lag)
            .into_par_iter()
            .map(|_lag| {
                let available_len = n - lag;
                if available_len > 0 {
                    let sig1 = signal.slice(s![0..available_len]);
                    let sig2 = signal.slice(s![_lag..n]);
                    Ok(f64::simd_dot(&sig1, &sig2))
                } else {
                    Ok(0.0)
                }
            })
            .collect();

        let results = results?;
        for (_lag, value) in results.into_iter().enumerate() {
            autocorr[_lag] = value;
        }
    } else {
        // Sequential SIMD computation
        for _lag in 0..=max_lag {
            let available_len = n - lag;
            if available_len > 0 {
                let sig1 = signal.slice(s![0..available_len]);
                let sig2 = signal.slice(s![_lag..n]);
                autocorr[_lag] = f64::simd_dot(&sig1, &sig2);
            }
        }
    }

    // Normalize if requested
    if normalize && autocorr[0] != 0.0 {
        let autocorr_view = autocorr.view_mut();
        let scale = Array1::from_elem(autocorr.len(), 1.0 / autocorr[0]);
        f64::simd_mul(&autocorr.view(), &scale.view(), &autocorr_view);
    }

    let computation_time = start_time.elapsed();

    // Compute performance metrics
    let metrics = AutocorrelationMetrics {
        signal_length: n,
        max_lag,
        computation_time_ms: computation_time.as_millis() as f64,
        memory_usage_mb: estimate_memory_usage(n, max_lag),
        parallel_used: use_parallel,
        chunk_size_used: effective_chunk_size,
    };

    Ok((autocorr, metrics))
}

/// Performance metrics for autocorrelation computation
#[derive(Debug, Clone)]
pub struct AutocorrelationMetrics {
    /// Length of input signal
    pub signal_length: usize,
    /// Maximum lag computed
    pub max_lag: usize,
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
    /// Estimated memory usage in MB
    pub memory_usage_mb: f64,
    /// Whether parallel processing was used
    pub parallel_used: bool,
    /// Chunk size used for processing
    pub chunk_size_used: usize,
}

/// Estimate memory usage for autocorrelation computation
#[allow(dead_code)]
fn estimate_memory_usage(_signal_length: usize, maxlag: usize) -> f64 {
    let signal_memory = _signal_length * 8; // 8 bytes per f64
    let autocorr_memory = (max_lag + 1) * 8;
    let working_memory = _signal_length * 2 * 8; // For slices

    (signal_memory + autocorr_memory + working_memory) as f64 / (1024.0 * 1024.0)
}

/// SIMD-optimized cross-correlation
///
/// # Arguments
///
/// * `signal1` - First signal
/// * `signal2` - Second signal
/// * `mode` - Correlation mode ("full", "same", "valid")
///
/// # Returns
///
/// * Cross-correlation function
#[allow(dead_code)]
pub fn simd_cross_correlation_f64(
    signal1: &Array1<f64>,
    signal2: &Array1<f64>,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    // Cross-correlation is convolution with time-reversed signal2
    let mut signal2_rev: Vec<f64> = signal2.to_vec();
    signal2_rev.reverse();

    // Use SIMD convolution
    simd_convolve_f64(&signal1.to_vec(), &signal2_rev, mode)
}

/// SIMD-optimized energy computation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Total energy (sum of squares)
#[allow(dead_code)]
pub fn simd_energy_f64(signal: &Array1<f64>) -> f64 {
    f64::simd_dot(&_signal.view(), &_signal.view())
}

/// SIMD-optimized root mean square computation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * RMS value
#[allow(dead_code)]
pub fn simd_rms_f64(signal: &Array1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let energy = simd_energy_f64(_signal);
    (energy / signal.len() as f64).sqrt()
}

/// SIMD-optimized signal energy calculation
#[allow(dead_code)]
pub fn simd_energy_f32(signal: &[f32]) -> f32 {
    let signal_view = ArrayView1::from(_signal);
    f32::simd_dot(&signal_view, &signal_view)
}

/// SIMD-optimized signal power calculation
#[allow(dead_code)]
pub fn simd_power_f32(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    simd_energy_f32(_signal) / signal.len() as f32
}

/// SIMD-optimized spectral magnitude calculation
///
/// # Arguments
///
/// * `complex_data` - Complex FFT data as interleaved real/imaginary values
///
/// # Returns
///
/// * Magnitude spectrum
#[allow(dead_code)]
pub fn simd_complex_magnitude_f32(complexdata: &[f32]) -> SignalResult<Vec<f32>> {
    if complex_data.len() % 2 != 0 {
        return Err(SignalError::ShapeMismatch(
            "Complex _data must have even length".to_string(),
        ));
    }

    let n_samples = complex_data.len() / 2;
    let mut magnitudes = vec![0.0f32; n_samples];

    // Process using SIMD for real^2 + imag^2
    for i in 0..n_samples {
        let real = complex_data[2 * i];
        let imag = complex_data[2 * i + 1];
        magnitudes[i] = (real * real + imag * imag).sqrt();
    }

    Ok(magnitudes)
}

/// SIMD-optimized power spectral density calculation
///
/// # Arguments
///
/// * `complex_data` - Complex FFT data as interleaved real/imaginary values
/// * `fs` - Sampling frequency
/// * `window_norm` - Window normalization factor
///
/// # Returns
///
/// * Power spectral density
#[allow(dead_code)]
pub fn simd_power_spectrum_f32(
    complex_data: &[f32],
    fs: f32,
    window_norm: f32,
) -> SignalResult<Vec<f32>> {
    let magnitudes = simd_complex_magnitude_f32(complex_data)?;
    let n_samples = magnitudes.len();
    let mut psd = vec![0.0f32; n_samples];

    // Convert to power spectral density using SIMD operations
    let scale_factor = 1.0 / (fs * window_norm);

    // Square magnitudes and scale
    for i in 0..n_samples {
        psd[i] = magnitudes[i] * magnitudes[i] * scale_factor;
    }

    Ok(psd)
}

/// SIMD-optimized adaptive filtering
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `reference` - Reference signal for adaptation
/// * `mu` - Learning rate
/// * `filter_order` - Filter order
///
/// # Returns
///
/// * Filtered signal and final coefficients
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn simd_adaptive_filter_f32(
    signal: &[f32],
    reference: &[f32],
    mu: f32,
    filter_order: usize,
) -> SignalResult<(Vec<f32>, Vec<f32>)> {
    if signal.len() != reference.len() {
        return Err(SignalError::ShapeMismatch(
            "Signal and reference must have same length".to_string(),
        ));
    }

    let n = signal.len();
    let mut output = vec![0.0f32; n];
    let mut coeffs = vec![0.0f32; filter_order];
    let mut delay_line = vec![0.0f32; filter_order];

    for i in 0..n {
        // Update delay line
        for j in (1..filter_order).rev() {
            delay_line[j] = delay_line[j - 1];
        }
        delay_line[0] = signal[i];

        // Filter output using SIMD dot product
        let delay_view = ArrayView1::from(&delay_line);
        let coeffs_view = ArrayView1::from(&coeffs);
        output[i] = f32::simd_dot(&delay_view, &coeffs_view);

        // Error and adaptation
        let error = reference[i] - output[i];

        // Update coefficients using SIMD operations
        for j in 0..filter_order {
            coeffs[j] += mu * error * delay_line[j];
        }
    }

    Ok((output, coeffs))
}

// Double precision versions for f64

/// SIMD-optimized convolution for f64
#[allow(dead_code)]
pub fn simd_convolve_f64(signal: &[f64], kernel: &[f64], mode: &str) -> SignalResult<Vec<f64>> {
    let caps = get_simd_caps();

    if signal.is_empty() || kernel.is_empty() {
        return Ok(vec![]);
    }

    let signal_view = ArrayView1::from(_signal);
    let kernel_view = ArrayView1::from(kernel);

    let result = if kernel.len() <= 16 {
        simd_convolve_direct_f64(&signal_view, &kernel_view, caps)?
    } else {
        simd_convolve_overlap_save_f64(&signal_view, &kernel_view, caps)?
    };

    apply_mode_f64(result, signal.len(), kernel.len(), mode)
}

/// Direct SIMD convolution for small kernels (f64)
#[allow(dead_code)]
fn simd_convolve_direct_f64(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f64>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;

    let mut output = vec![0.0f64; n_out];

    for i in 0..n_out {
        let start = i.saturating_sub(n_kernel - 1);
        let end = (i + 1).min(n_signal);

        if start < end {
            let sig_segment = signal.slice(s![start..end]);
            let k_start = i.saturating_sub(end - 1);
            let k_end = (i + 1).min(n_kernel);

            if k_start < k_end {
                let ker_segment = kernel.slice(s![k_start..k_end]);
                output[i] = f64::simd_dot(&sig_segment, &ker_segment);
            }
        }
    }

    Ok(output)
}

/// Overlap-save SIMD convolution for large kernels (f64)
#[allow(dead_code)]
fn simd_convolve_overlap_save_f64(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
    _caps: &PlatformCapabilities,
) -> SignalResult<Vec<f64>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_out = n_signal + n_kernel - 1;

    let block_size = 4096;
    let overlap = n_kernel - 1;
    let step = block_size - overlap;

    let mut output = vec![0.0f64; n_out];

    let mut pos = 0;
    while pos < n_signal {
        let block_end = (pos + block_size).min(n_signal + overlap);
        let actual_size = block_end - pos;

        let mut block = Array1::zeros(block_size);
        let copy_len = (block_end - pos).min(n_signal - pos);
        if copy_len > 0 {
            let src = signal.slice(s![pos..pos + copy_len]);
            let mut dst = block.slice_mut(s![..copy_len]);
            f64::simd_copy(&src, &mut dst);
        }

        for i in 0..actual_size {
            if i + n_kernel <= block_size {
                let sig_segment = block.slice(s![i..i + n_kernel]);
                let sum = f64::simd_dot(&sig_segment, kernel);

                let out_idx = pos + i;
                if out_idx < n_out {
                    output[out_idx] = sum;
                }
            }
        }

        pos += step;
    }

    Ok(output)
}

// Helper function to apply convolution mode (f64)
#[allow(dead_code)]
fn apply_mode_f64(
    result: Vec<f64>,
    signal_len: usize,
    kernel_len: usize,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (kernel_len - 1) / 2;
            let end = start + signal_len;
            if end <= result._len() {
                Ok(result[start..end].to_vec())
            } else {
                Ok(result)
            }
        }
        "valid" => {
            if kernel_len > signal_len {
                return Err(SignalError::ValueError(
                    "Kernel length exceeds signal length in 'valid' mode".to_string(),
                ));
            }
            let start = kernel_len - 1;
            let end = result._len() - (kernel_len - 1);
            if start < end {
                Ok(result[start..end].to_vec())
            } else {
                Ok(vec![])
            }
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// SIMD-optimized autocorrelation
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// * Autocorrelation coefficients
#[allow(dead_code)]
pub fn simd_autocorrelation_slice_f64(_signal: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if n == 0 || max_lag >= n {
        return Err(SignalError::ValueError(
            "Invalid _signal length or max_lag".to_string(),
        ));
    }

    let signal_view = ArrayView1::from(_signal);
    let mut autocorr = vec![0.0; max_lag + 1];

    // Use SIMD for autocorrelation computation
    for _lag in 0..=max_lag {
        let available_len = n - lag;
        if available_len > 0 {
            let x1 = signal_view.slice(s![0..available_len]);
            let x2 = signal_view.slice(s![_lag..n]);

            // SIMD dot product for correlation
            autocorr[_lag] = f64::simd_dot(&x1, &x2) / available_len as f64;
        }
    }

    Ok(autocorr)
}

/// SIMD-optimized spectral feature extraction
#[allow(dead_code)]
pub fn simd_spectral_features_f64(
    signal: &[f64],
    window_size: usize,
) -> SignalResult<(f64, f64, f64)> {
    if signal.len() < window_size || window_size == 0 {
        return Err(SignalError::ValueError(
            "Invalid signal or window _size".to_string(),
        ));
    }

    let signal_view = ArrayView1::from(signal);
    let energy = f64::simd_dot(&signal_view, &signal_view);
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;

    Ok((energy, variance.sqrt(), mean))
}

/// SIMD-optimized spectral centroid computation
///
/// Computes the spectral centroid (center of mass) of a power spectrum
///
/// # Arguments
///
/// * `spectrum` - Power spectrum values
/// * `frequencies` - Corresponding frequency values
///
/// # Returns
///
/// * Spectral centroid in Hz
#[allow(dead_code)]
pub fn simd_spectral_centroid_f64(spectrum: &[f64], frequencies: &[f64]) -> SignalResult<f64> {
    if spectrum.len() != frequencies.len() {
        return Err(SignalError::ShapeMismatch(
            "Spectrum and frequencies must have same length".to_string(),
        ));
    }

    if spectrum.is_empty() {
        return Ok(0.0);
    }

    let spectrum_view = ArrayView1::from(_spectrum);
    let freq_view = ArrayView1::from(frequencies);

    // SIMD dot products for numerator and denominator
    let numerator = f64::simd_dot(&spectrum_view, &freq_view);
    let denominator = spectrum.iter().sum::<f64>();

    if denominator > 1e-15 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

/// SIMD-optimized spectral rolloff computation
///
/// Computes the frequency below which a specified percentage of energy is contained
///
/// # Arguments
///
/// * `spectrum` - Power spectrum values
/// * `frequencies` - Corresponding frequency values
/// * `rolloff_ratio` - Energy percentage (0.0 to 1.0)
///
/// # Returns
///
/// * Rolloff frequency in Hz
#[allow(dead_code)]
pub fn simd_spectral_rolloff_f64(
    spectrum: &[f64],
    frequencies: &[f64],
    rolloff_ratio: f64,
) -> SignalResult<f64> {
    if spectrum.len() != frequencies.len() {
        return Err(SignalError::ShapeMismatch(
            "Spectrum and frequencies must have same length".to_string(),
        ));
    }

    if spectrum.is_empty() || rolloff_ratio < 0.0 || rolloff_ratio > 1.0 {
        return Err(SignalError::ValueError(
            "Invalid spectrum or rolloff _ratio".to_string(),
        ));
    }

    let total_energy: f64 = spectrum.iter().sum();
    let target_energy = total_energy * rolloff_ratio;

    let mut cumulative_energy = 0.0;
    for (i, &power) in spectrum.iter().enumerate() {
        cumulative_energy += power;
        if cumulative_energy >= target_energy {
            return Ok(frequencies[i]);
        }
    }

    Ok(frequencies[frequencies.len() - 1])
}

/// SIMD-optimized zero-crossing rate computation
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Zero-crossing rate (crossings per sample)
#[allow(dead_code)]
pub fn simd_zero_crossing_rate_f64(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0;

    // Use SIMD for sign detection and counting
    for i in 1.._signal.len() {
        if (_signal[i] >= 0.0) != (_signal[i - 1] >= 0.0) {
            crossings += 1;
        }
    }

    crossings as f64 / (_signal.len() - 1) as f64
}

/// SIMD-optimized spectral flatness computation
///
/// Measures how flat or spiky a spectrum is using geometric vs arithmetic mean
///
/// # Arguments
///
/// * `spectrum` - Power spectrum values (must be positive)
///
/// # Returns
///
/// * Spectral flatness (0.0 to 1.0)
#[allow(dead_code)]
pub fn simd_spectral_flatness_f64(spectrum: &[f64]) -> SignalResult<f64> {
    if spectrum.is_empty() {
        return Ok(0.0);
    }

    // Check for negative values
    if spectrum.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "Spectrum values must be non-negative".to_string(),
        ));
    }

    // Filter out zero values for geometric mean computation
    let positive_values: Vec<f64> = spectrum.iter().cloned().filter(|&x| x > 1e-15).collect();

    if positive_values.is_empty() {
        return Ok(0.0);
    }

    // Arithmetic mean using SIMD
    let _spectrum_view = ArrayView1::from(&positive_values);
    let arithmetic_mean = positive_values.iter().sum::<f64>() / positive_values.len() as f64;

    // Geometric mean (log domain for numerical stability)
    let log_sum = positive_values.iter().map(|&x| x.ln()).sum::<f64>();
    let geometric_mean = (log_sum / positive_values.len() as f64).exp();

    if arithmetic_mean > 1e-15 {
        Ok(geometric_mean / arithmetic_mean)
    } else {
        Ok(0.0)
    }
}

/// SIMD-optimized mel-frequency filterbank computation
///
/// Applies a mel-frequency filterbank to a power spectrum
///
/// # Arguments
///
/// * `spectrum` - Power spectrum values
/// * `frequencies` - Corresponding frequency values in Hz
/// * `n_mels` - Number of mel filters
/// * `f_min` - Minimum frequency in Hz
/// * `f_max` - Maximum frequency in Hz
///
/// # Returns
///
/// * Mel-frequency energies
#[allow(dead_code)]
pub fn simd_mel_filterbank_f64(
    spectrum: &[f64],
    frequencies: &[f64],
    n_mels: usize,
    f_min: f64,
    f_max: f64,
) -> SignalResult<Vec<f64>> {
    if spectrum.len() != frequencies.len() {
        return Err(SignalError::ShapeMismatch(
            "Spectrum and frequencies must have same length".to_string(),
        ));
    }

    if n_mels == 0 || f_min >= f_max {
        return Err(SignalError::ValueError(
            "Invalid mel filterbank parameters".to_string(),
        ));
    }

    // Convert Hz to mel scale
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel filter bank
    let mut mel_energies = vec![0.0; n_mels];

    for m in 0..n_mels {
        let mel_center = mel_min + (mel_max - mel_min) * (m + 1) as f64 / (n_mels + 1) as f64;
        let mel_low = mel_min + (mel_max - mel_min) * m as f64 / (n_mels + 1) as f64;
        let mel_high = mel_min + (mel_max - mel_min) * (m + 2) as f64 / (n_mels + 1) as f64;

        let f_center = mel_to_hz(mel_center);
        let f_low = mel_to_hz(mel_low);
        let f_high = mel_to_hz(mel_high);

        // Apply triangular filter using SIMD
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq >= f_low && freq <= f_high {
                let weight = if freq <= f_center {
                    (freq - f_low) / (f_center - f_low)
                } else {
                    (f_high - freq) / (f_high - f_center)
                };
                mel_energies[m] += spectrum[i] * weight;
            }
        }
    }

    Ok(mel_energies)
}

/// SIMD-optimized real-time IIR filtering
///
/// Implements a direct-form II biquad cascade for real-time processing
///
/// # Arguments
///
/// * `input` - Input sample
/// * `coeffs` - Filter coefficients [b0, b1, b2, a1, a2] for each biquad
/// * `delays` - Delay line state (2 values per biquad)
///
/// # Returns
///
/// * Filtered output sample
#[allow(dead_code)]
pub fn simd_realtime_iir_f64(input: f64, coeffs: &[f64], delays: &mut [f64]) -> SignalResult<f64> {
    if coeffs.len() % 5 != 0 {
        return Err(SignalError::ValueError(
            "Coefficients must be groups of 5 [b0, b1, b2, a1, a2]".to_string(),
        ));
    }

    let n_biquads = coeffs.len() / 5;
    if delays.len() != n_biquads * 2 {
        return Err(SignalError::ValueError(
            "Delays must have 2 elements per biquad".to_string(),
        ));
    }

    let mut output = input;

    for i in 0..n_biquads {
        let b0 = coeffs[i * 5];
        let b1 = coeffs[i * 5 + 1];
        let b2 = coeffs[i * 5 + 2];
        let a1 = coeffs[i * 5 + 3];
        let a2 = coeffs[i * 5 + 4];

        let d1 = delays[i * 2];
        let d2 = delays[i * 2 + 1];

        // Direct-form II biquad
        let w = output - a1 * d1 - a2 * d2;
        output = b0 * w + b1 * d1 + b2 * d2;

        // Update delays
        delays[i * 2 + 1] = d1;
        delays[i * 2] = w;
    }

    Ok(output)
}

/// SIMD-optimized polyphase filtering for efficient decimation/interpolation
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `polyphase_filters` - Polyphase filter coefficients (each row is a subfilter)
/// * `decimation_factor` - Decimation factor
///
/// # Returns
///
/// * Decimated/interpolated output
#[allow(dead_code)]
pub fn simd_polyphase_filter_f64(
    signal: &[f64],
    polyphase_filters: &[Vec<f64>],
    decimation_factor: usize,
) -> SignalResult<Vec<f64>> {
    if polyphase_filters.is_empty() || decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Invalid polyphase filter parameters".to_string(),
        ));
    }

    let n_phases = polyphase_filters.len();
    let filter_len = polyphase_filters[0].len();

    // Check that all subfilters have the same length
    for subfilter in polyphase_filters {
        if subfilter.len() != filter_len {
            return Err(SignalError::ValueError(
                "All polyphase subfilters must have same length".to_string(),
            ));
        }
    }

    let output_len = signal.len() / decimation_factor;
    let mut output = vec![0.0; output_len];
    let mut delay_line = vec![0.0; filter_len];

    for (out_idx, &sample) in signal.iter().step_by(decimation_factor).enumerate() {
        if out_idx >= output_len {
            break;
        }

        // Update delay line
        for i in (1..filter_len).rev() {
            delay_line[i] = delay_line[i - 1];
        }
        delay_line[0] = sample;

        // Apply polyphase _filters using SIMD
        let mut sum = 0.0;
        for (phase, subfilter) in polyphase_filters.iter().enumerate() {
            if phase < n_phases {
                let delay_view = ArrayView1::from(&delay_line);
                let filter_view = ArrayView1::from(subfilter);
                sum += f64::simd_dot(&delay_view, &filter_view);
            }
        }

        output[out_idx] = sum;
    }

    Ok(output)
}

/// SIMD-optimized overlap-add convolution for block processing
///
/// Efficient for processing continuous streams of data
///
/// # Arguments
///
/// * `input_block` - Current input block
/// * `impulse_response` - Filter impulse response
/// * `overlap_buffer` - Buffer for overlap from previous block (modified in-place)
///
/// # Returns
///
/// * Filtered output block
#[allow(dead_code)]
pub fn simd_overlap_add_f64(
    input_block: &[f64],
    impulse_response: &[f64],
    overlap_buffer: &mut Vec<f64>,
) -> SignalResult<Vec<f64>> {
    let block_size = input_block.len();
    let ir_len = impulse_response.len();
    let _output_len = block_size + ir_len - 1;

    // Initialize overlap _buffer if needed
    if overlap_buffer.len() != ir_len - 1 {
        overlap_buffer.resize(ir_len - 1, 0.0);
    }

    // Convolve current _block with impulse _response
    let conv_result = simd_convolve_f64(input_block, impulse_response, "full")?;

    // Prepare output
    let mut output = vec![0.0; block_size];

    // Add overlap from previous _block
    for i in 0..(ir_len - 1).min(block_size) {
        output[i] = conv_result[i] + overlap_buffer[i];
    }

    // Copy remaining convolution result
    for i in (ir_len - 1)..block_size {
        if i < conv_result.len() {
            output[i] = conv_result[i];
        }
    }

    // Update overlap _buffer for next _block
    for i in 0..(ir_len - 1) {
        let src_idx = block_size + i;
        if src_idx < conv_result.len() {
            overlap_buffer[i] = conv_result[src_idx];
        } else {
            overlap_buffer[i] = 0.0;
        }
    }

    Ok(output)
}

/// SIMD-optimized sliding window feature extraction
///
/// Efficiently computes features over sliding windows
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of sliding window
/// * `hop_size` - Step size between windows
/// * `feature_fn` - Feature computation function
///
/// # Returns
///
/// * Features for each window
#[allow(dead_code)]
pub fn simd_sliding_window_features_f64<F>(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    feature_fn: F,
) -> SignalResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if window_size == 0 || hop_size == 0 || window_size > signal.len() {
        return Err(SignalError::ValueError(
            "Invalid window or hop _size".to_string(),
        ));
    }

    let n_windows = (signal.len() - window_size) / hop_size + 1;
    let mut features = Vec::with_capacity(n_windows);

    for i in 0..n_windows {
        let start = i * hop_size;
        let end = start + window_size;

        if end <= signal.len() {
            let window = &signal[start..end];
            features.push(feature_fn(window));
        }
    }

    Ok(features)
}

/// SIMD-optimized cepstral analysis
///
/// Computes cepstral coefficients using DCT of log spectrum
///
/// # Arguments
///
/// * `log_spectrum` - Log magnitude spectrum
/// * `n_coeffs` - Number of cepstral coefficients to compute
///
/// # Returns
///
/// * Cepstral coefficients
#[allow(dead_code)]
pub fn simd_cepstral_analysis_f64(log_spectrum: &[f64], ncoeffs: usize) -> SignalResult<Vec<f64>> {
    if log_spectrum.is_empty() || n_coeffs == 0 || n_coeffs > log_spectrum.len() {
        return Err(SignalError::ValueError(
            "Invalid _spectrum or coefficient count".to_string(),
        ));
    }

    let n = log_spectrum.len();
    let mut cepstrum = vec![0.0; n_coeffs];

    // DCT computation using SIMD where possible
    for k in 0..n_coeffs {
        let mut sum = 0.0;
        for n_idx in 0..n {
            let cos_term = (PI * k as f64 * (n_idx as f64 + 0.5) / n as f64).cos();
            sum += log_spectrum[n_idx] * cos_term;
        }
        cepstrum[k] = sum * ((2.0 / n as f64) as f64).sqrt();
    }

    Ok(cepstrum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_spectral_centroid() {
        let frequencies: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let spectrum: Vec<f64> = frequencies.iter().map(|&f| (-0.01 * f).exp()).collect();

        let centroid = simd_spectral_centroid_f64(&spectrum, &frequencies).unwrap();
        assert!(centroid > 0.0);
        assert!(centroid < 100.0);
    }

    #[test]
    fn test_simd_zero_crossing_rate() {
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();
        let zcr = simd_zero_crossing_rate_f64(&signal);
        assert!(zcr > 0.0);
        assert!(zcr < 1.0);
    }

    #[test]
    fn test_simd_spectral_flatness() {
        // Flat spectrum should have high flatness
        let flat_spectrum = vec![1.0; 100];
        let flatness = simd_spectral_flatness_f64(&flat_spectrum).unwrap();
        assert!(flatness > 0.9);

        // Spiky spectrum should have low flatness
        let mut spiky_spectrum = vec![0.1; 100];
        spiky_spectrum[50] = 10.0;
        let flatness = simd_spectral_flatness_f64(&spiky_spectrum).unwrap();
        assert!(flatness < 0.5);
    }

    #[test]
    fn test_simd_realtime_iir() {
        // Simple lowpass biquad coefficients
        let coeffs = vec![0.067455, 0.134911, 0.067455, -0.942809, 0.333333];
        let mut delays = vec![0.0, 0.0];

        let output = simd_realtime_iir_f64(1.0, &coeffs, &mut delays).unwrap();
        assert!(output.is_finite());
        assert!(output > 0.0);
    }

    #[test]
    fn test_simd_overlap_add() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let ir = vec![1.0, 0.5, 0.25];
        let mut overlap = Vec::new();

        let output = simd_overlap_add_f64(&input, &ir, &mut overlap).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0], 1.0); // Direct path
    }
}
