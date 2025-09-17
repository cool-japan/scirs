//! SIMD-optimized spectral analysis operations
//!
//! This module provides highly optimized SIMD implementations for spectral analysis
//! operations including spectral centroid computation, rolloff analysis, batch processing,
//! power spectrum computation, and complex number operations.
//!
//! # Features
//!
//! - **SIMD Spectral Centroid**: Compute spectral centroid using SIMD operations
//! - **SIMD Spectral Rolloff**: Calculate spectral rolloff with SIMD acceleration
//! - **Batch Processing**: Process multiple signals simultaneously with optimized algorithms
//! - **Power Spectrum**: Efficient computation of power spectral density
//! - **Complex Operations**: SIMD-accelerated complex number multiplication
//! - **Window Functions**: Optimized window generation for spectral analysis
//!
//! # SIMD Support
//!
//! - **x86_64**: AVX2, SSE4.1 instruction sets
//! - **AArch64**: NEON instruction sets (where applicable)
//! - **Scalar fallback**: Available for all platforms

use super::basic_ops::simd_apply_window;
use super::types::{BatchSpectralResult, BatchSpectralStats, SimdConfig, SingleSpectralResult};
use crate::error::{SignalError, SignalResult};
use ndarray::{Array2, ArrayView1, ArrayViewMut1};
use num_complex::Complex64;
use rustfft::FftPlanner;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f64::consts::PI;

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

/// SIMD-optimized spectral centroid computation
///
/// Computes the spectral centroid (center of mass of the spectrum) using
/// SIMD operations for enhanced performance on large spectral data.
///
/// The spectral centroid is computed as:
/// centroid = sum(f * |X(f)|) / sum(|X(f)|)
///
/// # Arguments
///
/// * `magnitude_spectrum` - Magnitude spectrum values
/// * `frequencies` - Corresponding frequency values in Hz
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
            "Magnitude spectrum and frequencies must have same length".to_string(),
        ));
    }

    check_slice_finite(magnitude_spectrum, "magnitude_spectrum")?;
    check_slice_finite(frequencies, "frequencies")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_centroid(magnitude_spectrum, frequencies);
    }

    let _caps = PlatformCapabilities::detect();

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
fn scalar_spectral_centroid(magnitude_spectrum: &[f64], frequencies: &[f64]) -> SignalResult<f64> {
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
            "Magnitude spectrum and frequencies must have same length".to_string(),
        ));
    }

    if rolloff_threshold <= 0.0 || rolloff_threshold >= 1.0 {
        return Err(SignalError::ValueError(
            "Rolloff threshold must be between 0 and 1".to_string(),
        ));
    }

    check_slice_finite(magnitude_spectrum, "magnitude_spectrum")?;
    check_slice_finite(frequencies, "frequencies")?;

    let n = magnitude_spectrum.len();
    if n < config.simd_threshold || config.force_scalar {
        return scalar_spectral_rolloff(magnitude_spectrum, frequencies, rolloff_threshold);
    }

    // Compute energy spectrum (magnitude^2) using SIMD
    let mut energy_spectrum = vec![0.0; n];
    let mag_view = ArrayView1::from(magnitude_spectrum);
    let energy_view = ArrayViewMut1::from(&mut energy_spectrum);

    // Use SIMD element-wise multiplication
    let energy_result = f64::simd_mul(&mag_view, &mag_view);
    energy_spectrum.copy_from_slice(energy_result.as_slice().unwrap());

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

/// SIMD-optimized batch spectral analysis
///
/// Processes multiple signals simultaneously to compute power spectra,
/// phase information, and statistical metrics. Uses parallel processing
/// for optimal performance on multi-core systems.
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
                "Unknown window type: {}",
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

/// SIMD-optimized complex number multiplication
///
/// Performs element-wise complex multiplication (a + bi) * (c + di) using SIMD instructions.
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

    check_slice_finite(a_real, "a_real")?;
    check_slice_finite(a_imag, "a_imag")?;
    check_slice_finite(b_real, "b_real")?;
    check_slice_finite(b_imag, "b_imag")?;

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

    check_slice_finite(real, "real")?;
    check_slice_finite(imag, "imag")?;

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

    check_slice_finite(weights, "weights")?;
    for (i, spectrum) in spectra.iter().enumerate() {
        check_slice_finite(spectrum, &format!("spectrum_{}", i))?;
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
    for i in 0..real.len() {
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
        let result_real_vec =
            _mm256_sub_pd(_mm256_mul_pd(ar_vec, br_vec), _mm256_mul_pd(ai_vec, bi_vec));

        // result_imag = a_real * b_imag + a_imag * b_real
        let result_imag_vec =
            _mm256_add_pd(_mm256_mul_pd(ar_vec, bi_vec), _mm256_mul_pd(ai_vec, br_vec));

        _mm256_storeu_pd(result_real.as_mut_ptr().add(idx), result_real_vec);
        _mm256_storeu_pd(result_imag.as_mut_ptr().add(idx), result_imag_vec);
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
        let weight = weights[taper_idx];
        let weight_vec = _mm256_set1_pd(weight);

        for chunk in 0..simd_chunks {
            let idx = chunk * simd_width;

            let spectrum_vec = _mm256_loadu_pd(spectrum.as_ptr().add(idx));
            let result_vec = _mm256_loadu_pd(result.as_ptr().add(idx));

            let weighted_spectrum = _mm256_mul_pd(spectrum_vec, weight_vec);
            let new_result = _mm256_add_pd(result_vec, weighted_spectrum);

            _mm256_storeu_pd(result.as_mut_ptr().add(idx), new_result);
        }

        // Handle remaining elements
        for i in (simd_chunks * simd_width)..n_freqs {
            result[i] += weight * spectrum[i];
        }
    }

    Ok(())
}

// SSE implementations (fallback to scalar for simplicity in this extraction)
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
    scalar_power_spectrum(real, imag, power)
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

// Non-x86_64 implementations (fallback to scalar)
#[cfg(not(target_arch = "x86_64"))]
fn avx2_complex_multiply(
    a_real: &[f64],
    a_imag: &[f64],
    b_real: &[f64],
    b_imag: &[f64],
    result_real: &mut [f64],
    result_imag: &mut [f64],
) -> SignalResult<()> {
    scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    scalar_power_spectrum(real, imag, power)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    scalar_weighted_average_spectra(spectra, weights, result)
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
    scalar_complex_multiply(a_real, a_imag, b_real, b_imag, result_real, result_imag)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_power_spectrum(real: &[f64], imag: &[f64], power: &mut [f64]) -> SignalResult<()> {
    scalar_power_spectrum(real, imag, power)
}

#[cfg(not(target_arch = "x86_64"))]
fn sse_weighted_average_spectra(
    spectra: &[&[f64]],
    weights: &[f64],
    result: &mut [f64],
) -> SignalResult<()> {
    scalar_weighted_average_spectra(spectra, weights, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_spectral_centroid() {
        let magnitude = vec![1.0, 2.0, 1.5, 0.5, 0.1];
        let frequencies = vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0];
        let config = SimdConfig::default();

        let centroid = simd_spectral_centroid(&magnitude, &frequencies, &config).unwrap();
        assert!(centroid > 0.0 && centroid < 4000.0);
    }

    #[test]
    fn test_simd_spectral_rolloff() {
        let magnitude = vec![2.0, 1.5, 1.0, 0.5, 0.1];
        let frequencies = vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0];
        let config = SimdConfig::default();

        let rolloff = simd_spectral_rolloff(&magnitude, &frequencies, 0.85, &config).unwrap();
        assert!(rolloff >= 0.0 && rolloff <= 4000.0);
    }
}
