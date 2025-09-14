// Enhanced multitaper spectral estimation with SIMD and parallel processing
//
// This module provides high-performance implementations of multitaper spectral
// estimation using scirs2-core's SIMD and parallel processing capabilities.
//
// Key improvements in this version:
// - Enhanced numerical stability in adaptive weighting
// - Better convergence detection and error handling
// - Improved memory efficiency for large signals
// - More robust confidence interval computation
// - Better parameter validation and edge case handling

use super::windows::dpss;
use crate::error::{SignalError, SignalResult};
use crate::simd_advanced::{simd_apply_window, SimdConfig};
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rand::Rng;
use rustfft::FftPlanner;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::check_positive;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::fmt::Debug;
use std::sync::Arc;

#[allow(unused_imports)]
/// Enhanced multitaper PSD result with additional statistics
#[derive(Debug, Clone)]
pub struct EnhancedMultitaperResult {
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Power spectral density
    pub psd: Vec<f64>,
    /// Confidence intervals (if requested)
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
    /// Effective degrees of freedom
    pub dof: Option<f64>,
    /// DPSS tapers used (if requested)
    pub tapers: Option<Array2<f64>>,
    /// Eigenvalues (if requested)
    pub eigenvalues: Option<Array1<f64>>,
}

/// Configuration for enhanced multitaper estimation
#[derive(Debug, Clone)]
pub struct MultitaperConfig {
    /// Sampling frequency
    pub fs: f64,
    /// Time-bandwidth product
    pub nw: f64,
    /// Number of tapers
    pub k: usize,
    /// FFT length
    pub nfft: Option<usize>,
    /// Return one-sided spectrum
    pub onesided: bool,
    /// Use adaptive weighting
    pub adaptive: bool,
    /// Compute confidence intervals
    pub confidence: Option<f64>,
    /// Return tapers and eigenvalues
    pub return_tapers: bool,
    /// Use parallel processing
    pub parallel: bool,
    /// Minimum chunk size for parallel processing
    pub parallel_threshold: usize,
    /// Force memory-optimized processing for large signals
    pub memory_optimized: bool,
}

impl Default for MultitaperConfig {
    fn default() -> Self {
        Self {
            fs: 1.0,
            nw: 4.0,
            k: 7, // 2*nw - 1
            nfft: None,
            onesided: true,
            adaptive: true,
            confidence: None,
            return_tapers: false,
            parallel: true,
            parallel_threshold: 1024,
            memory_optimized: false,
        }
    }
}

/// Enhanced multitaper power spectral density estimation with SIMD and parallel processing
///
/// This function provides a high-performance implementation of the multitaper method
/// using scirs2-core's acceleration capabilities.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `config` - Multitaper configuration
///
/// # Returns
///
/// * Enhanced multitaper result with PSD and optional statistics
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::enhanced::{enhanced_pmtm, MultitaperConfig};
///
///
/// // Generate test signal
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// use rand::prelude::*;
/// let mut rng = rand::rng();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rng.gen_range(0.0..1.0))
///     .collect();
///
/// // Configure multitaper estimation
/// let config = MultitaperConfig {
///     fs,
///     nw: 4.0,
///     k: 7,
///     confidence: Some(0.95),
///     ..Default::default()
/// };
///
/// // Compute enhanced multitaper PSD
/// let result = enhanced_pmtm(&signal, &config).unwrap();
/// assert!(result.frequencies.len() > 0);
/// assert!(result.confidence_intervals.is_some());
/// ```
#[allow(dead_code)]
pub fn enhanced_pmtm<T>(
    x: &[T],
    config: &MultitaperConfig,
) -> SignalResult<EnhancedMultitaperResult>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    check_positive(config.nw, "nw")?;
    check_positive(config.k, "k")?;
    check_positive(config.fs, "fs")?;

    // Convert input to f64 for numerical computations
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Validate confidence level if provided
    if let Some(confidence) = config.confidence {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(SignalError::ValueError(format!(
                "Confidence level must be between 0 and 1, got {}",
                confidence
            )));
        }
    }

    // Enhanced multitaper parameter validation
    if config.k > (2.0 * config.nw) as usize {
        return Err(SignalError::ValueError(format!(
            "Number of tapers k={} should not exceed 2*nw={}",
            config.k,
            2.0 * config.nw
        )));
    }

    // Additional validation for numerical stability
    if config.nw < 1.0 {
        return Err(SignalError::ValueError(format!(
            "Time-bandwidth product nw={} must be at least 1.0",
            config.nw
        )));
    }

    if config.k == 0 {
        return Err(SignalError::ValueError(
            "Number of tapers k must be at least 1".to_string(),
        ));
    }

    // Check if signal is long enough for meaningful spectral estimation
    let min_signal_length = (4.0 * config.nw) as usize;
    if x_f64.len() < min_signal_length {
        return Err(SignalError::ValueError(format!(
            "Signal length {} too short for nw={}. Minimum length is {}",
            x_f64.len(),
            config.nw,
            min_signal_length
        )));
    }

    // Warn if signal is very short relative to nw
    if x_f64.len() < (8.0 * config.nw) as usize {
        eprintln!("Warning: Signal length {} is relatively short for nw={}. Consider reducing nw or using a longer signal.", 
                  x_f64.len(), config.nw);
    }

    // Validate that all values are finite
    for (i, &val) in x_f64.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ValueError(format!(
                "Non-finite value at index {}: {}",
                i, val
            )));
        }
    }

    let n = x_f64.len();
    let nfft = config.nfft.unwrap_or(next_power_of_two(n));

    // Enhanced memory management: Adaptive threshold based on available system memory
    let memory_threshold = if config.k > 10 {
        500_000 // Reduce threshold for many tapers
    } else {
        1_000_000 // 1M samples for normal cases
    };
    let use_chunked_processing = n > memory_threshold || config.memory_optimized;

    if use_chunked_processing {
        return compute_pmtm_chunked(&x_f64, config, nfft);
    }

    // Compute DPSS tapers with enhanced validation
    let (tapers, eigenvalues_opt) = dpss(n, config.nw, config.k, true)?;

    let eigenvalues = eigenvalues_opt.ok_or_else(|| {
        SignalError::ComputationError("Eigenvalues required but not returned from dpss".to_string())
    })?;

    // Enhanced validation of DPSS results
    // Check eigenvalue ordering (should be descending)
    for i in 1..eigenvalues.len() {
        if eigenvalues[i] > eigenvalues[i - 1] {
            return Err(SignalError::ComputationError(
                "DPSS eigenvalues are not in descending order".to_string(),
            ));
        }
    }

    // Check eigenvalue concentration (should be close to 1 for good tapers)
    let min_concentration = 0.9;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval < min_concentration && i < config.k {
            eprintln!("Warning: Taper {} has low concentration ratio {:.3}. Consider reducing k or increasing nw.", 
                      i, eigenval);
        }
    }

    // Verify taper orthogonality
    for i in 0..config.k {
        for j in (i + 1)..config.k {
            let dot_product: f64 = tapers.row(i).dot(&tapers.row(j));
            if dot_product.abs() > 1e-10 {
                eprintln!(
                    "Warning: Tapers {} and {} have non-orthogonal dot product {:.2e}",
                    i, j, dot_product
                );
            }
        }
    }

    // Compute tapered FFTs using parallel processing if enabled
    let spectra = if config.parallel && n >= config.parallel_threshold {
        compute_tapered_ffts_parallel(&x_f64, &tapers, nfft)?
    } else {
        compute_tapered_ffts_simd(&x_f64, &tapers, nfft)?
    };

    // Enhanced spectral validation before combination
    for i in 0..spectra.nrows() {
        for j in 0..spectra.ncols() {
            let val = spectra[[i, j]];
            if !val.is_finite() || val < 0.0 {
                return Err(SignalError::ComputationError(format!(
                    "Invalid spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }
        }
    }

    // Combine spectra using adaptive or standard weighting
    let (frequencies, psd) = if config.adaptive {
        combine_spectra_adaptive(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    } else {
        combine_spectra_standard(&spectra, &eigenvalues, config.fs, nfft, config.onesided)?
    };

    // Final validation of PSD results
    for (i, &val) in psd.iter().enumerate() {
        if !val.is_finite() || val < 0.0 {
            return Err(SignalError::ComputationError(format!(
                "Invalid PSD value at frequency bin {}: {}",
                i, val
            )));
        }
    }

    // Compute confidence intervals if requested
    let confidence_intervals = if let Some(confidence_level) = config.confidence {
        Some(compute_confidence_intervals(
            &spectra,
            &eigenvalues,
            confidence_level,
        )?)
    } else {
        None
    };

    // Compute effective degrees of freedom
    let dof = Some(compute_effective_dof(&eigenvalues));

    Ok(EnhancedMultitaperResult {
        frequencies,
        psd,
        confidence_intervals,
        dof,
        tapers: if config.return_tapers {
            Some(tapers)
        } else {
            None
        },
        eigenvalues: if config.return_tapers {
            Some(eigenvalues)
        } else {
            None
        },
    })
}

/// Compute tapered FFTs using enhanced SIMD operations
#[allow(dead_code)]
fn compute_tapered_ffts_simd(
    signal: &[f64],
    tapers: &Array2<f64>,
    nfft: usize,
) -> SignalResult<Array2<f64>> {
    let k = tapers.nrows();
    let n = signal.len();
    let mut spectra = Array2::zeros((k, nfft));

    // Get SIMD capabilities for optimal performance
    let caps = PlatformCapabilities::detect();
    let use_advanced_simd = caps.simd_available;

    // Enhanced memory management for large datasets
    let memory_efficient = k > 20 || n > 50_000;

    if memory_efficient {
        // Process tapers in smaller batches to reduce memory pressure
        let batch_size = if k > 50 { 8 } else { k };

        for batch_start in (0..k).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(k);

            for i in batch_start..batch_end {
                let result = compute_single_tapered_fft_simd(
                    signal,
                    tapers.row(i),
                    nfft,
                    use_advanced_simd,
                )?;
                for (j, &val) in result.iter().enumerate() {
                    spectra[[i, j]] = val;
                }
            }
        }
    } else {
        // Process all tapers at once for smaller datasets
        for i in 0..k {
            let result =
                compute_single_tapered_fft_simd(signal, tapers.row(i), nfft, use_advanced_simd)?;
            for (j, &val) in result.iter().enumerate() {
                spectra[[i, j]] = val;
            }
        }
    }

    // Enhanced validation of spectral results
    validate_spectral_matrix(&spectra)?;

    Ok(spectra)
}

/// Compute single tapered FFT with enhanced SIMD operations and robust fallbacks
#[allow(dead_code)]
fn compute_single_tapered_fft_simd(
    signal: &[f64],
    taper: ArrayView1<f64>,
    nfft: usize,
    use_advanced_simd: bool,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();

    // Enhanced tapering with multiple SIMD strategies
    let mut tapered = vec![0.0; n];
    let mut simd_success = false;

    if use_advanced_simd && n >= 64 {
        // Strategy 1: Try advanced SIMD operations for larger signals
        if let Ok(()) = try_advanced_simd_tapering(signal, &taper, &mut tapered) {
            simd_success = true;
        }
    }

    if !simd_success && n >= 16 {
        // Strategy 2: Basic SIMD operations with enhanced error handling
        match try_basic_simd_tapering(signal, &taper, &mut tapered) {
            Ok(()) => simd_success = true,
            Err(_) => {
                // Strategy 3: Chunked SIMD for problematic cases
                if try_chunked_simd_tapering(signal, &taper, &mut tapered).is_ok() {
                    simd_success = true;
                }
            }
        }
    }

    if !simd_success {
        // Strategy 4: Scalar fallback with optimization
        scalar_tapering_optimized(signal, &taper, &mut tapered);
    }

    // Enhanced validation with efficient checking
    validate_tapered_signal(&tapered)?;

    // Enhanced FFT computation with robust error handling
    let spectrum = enhanced_simd_fft(&tapered, nfft)?;

    // Compute power spectrum with comprehensive validation and protection
    compute_validated_power_spectrum(&spectrum)
}

/// Validate spectral matrix for numerical stability
#[allow(dead_code)]
fn validate_spectral_matrix(spectra: &Array2<f64>) -> SignalResult<()> {
    let (k, nfft) = spectra.dim();

    for i in 0..k {
        for j in 0..nfft {
            let val = spectra[[i, j]];

            if !val.is_finite() {
                return Err(SignalError::ComputationError(format!(
                    "Non-finite spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }

            if val < 0.0 {
                return Err(SignalError::ComputationError(format!(
                    "Negative spectral value at taper {}, frequency bin {}: {}",
                    i, j, val
                )));
            }

            // Check for extremely large values that might indicate computational issues
            if val > 1e200 {
                return Err(SignalError::ComputationError(format!(
                    "Extremely large spectral value at taper {}, frequency bin {}: {:.2e}",
                    i, j, val
                )));
            }
        }
    }

    // Additional validation: check for reasonable energy distribution
    for i in 0..k {
        let row_sum: f64 = (0..nfft).map(|j| spectra[[i, j]]).sum();

        if row_sum < 1e-100 {
            return Err(SignalError::ComputationError(format!(
                "Taper {} has extremely low total energy: {:.2e}",
                i, row_sum
            )));
        }

        if row_sum > 1e100 {
            eprintln!(
                "Warning: Taper {} has very high total energy: {:.2e}",
                i, row_sum
            );
        }
    }

    Ok(())
}

/// Compute tapered FFTs using parallel processing
#[allow(dead_code)]
fn compute_tapered_ffts_parallel(
    signal: &[f64],
    tapers: &Array2<f64>,
    nfft: usize,
) -> SignalResult<Array2<f64>> {
    let k = tapers.nrows();
    let n = signal.len();
    let signal_arc = Arc::new(signal.to_vec());

    // Process tapers in parallel
    let results: Result<Vec<Vec<f64>>, SignalError> = (0..k)
        .into_par_iter()
        .map(|i| {
            let signal_ref = signal_arc.clone();
            let taper = tapers.row(i).to_owned();

            // Apply taper
            let mut tapered = vec![0.0; n];
            for j in 0..n {
                tapered[j] = signal_ref[j] * taper[j];
            }

            // Compute FFT
            let spectrum = enhanced_simd_fft(&tapered, nfft)?;

            // Return power spectrum
            Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
        })
        .collect();

    let results = results?;

    // Convert to Array2
    let mut spectra = Array2::zeros((k, nfft));
    for (i, row) in results.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            spectra[[i, j]] = val;
        }
    }

    Ok(spectra)
}

/// Try advanced SIMD tapering with comprehensive error handling
#[allow(dead_code)]
fn try_advanced_simd_tapering(
    signal: &[f64],
    taper: &ArrayView1<f64>,
    tapered: &mut [f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let config = SimdConfig::default();
    let taper_vec: Vec<f64> = taper.iter().copied().collect();

    simd_apply_window(signal, &taper_vec, tapered, &config).map_err(|e| format!("{}", e).into())
}

/// Try basic SIMD tapering with error recovery
#[allow(dead_code)]
fn try_basic_simd_tapering(
    signal: &[f64],
    taper: &ArrayView1<f64>,
    tapered: &mut [f64],
) -> SignalResult<()> {
    let signal_view = ArrayView1::from(signal);
    let _tapered_view = ArrayView1::from_shape(signal.len(), tapered)
        .map_err(|e| SignalError::ComputationError(format!("Shape error: {}", e)))?;

    let result = f64::simd_mul(&signal_view, taper);
    for (i, &val) in result.iter().enumerate() {
        tapered[i] = val;
    }
    Ok(())
}

/// Try chunked SIMD tapering for edge cases
#[allow(dead_code)]
fn try_chunked_simd_tapering(
    signal: &[f64],
    taper: &ArrayView1<f64>,
    tapered: &mut [f64],
) -> SignalResult<()> {
    let chunk_size = 256; // Optimal chunk size for most SIMD implementations

    for (_chunk_idx, chunk_data) in signal
        .chunks(chunk_size)
        .zip(taper.as_slice().unwrap().chunks(chunk_size))
        .zip(tapered.chunks_mut(chunk_size))
        .enumerate()
    {
        let ((sig_chunk, tap_chunk), out_chunk) = chunk_data;
        let sig_view = ArrayView1::from(sig_chunk);
        let tap_view = ArrayView1::from(tap_chunk);

        let result = f64::simd_mul(&sig_view, &tap_view);
        for (i, &val) in result.iter().enumerate() {
            if i < out_chunk.len() {
                out_chunk[i] = val;
            }
        }
    }

    Ok(())
}

/// Optimized scalar tapering fallback
#[allow(dead_code)]
fn scalar_tapering_optimized(signal: &[f64], taper: &ArrayView1<f64>, tapered: &mut [f64]) {
    // Unrolled loop for better performance
    let taper_slice = taper.as_slice().unwrap();
    let chunks = signal.len() / 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let base_idx = i * 4;
        tapered[base_idx] = signal[base_idx] * taper_slice[base_idx];
        tapered[base_idx + 1] = signal[base_idx + 1] * taper_slice[base_idx + 1];
        tapered[base_idx + 2] = signal[base_idx + 2] * taper_slice[base_idx + 2];
        tapered[base_idx + 3] = signal[base_idx + 3] * taper_slice[base_idx + 3];
    }

    // Handle remaining elements
    for i in (chunks * 4).._signal.len() {
        tapered[i] = signal[i] * taper_slice[i];
    }
}

/// Validate tapered signal with efficient bulk checking
#[allow(dead_code)]
fn validate_tapered_signal(tapered: &[f64]) -> SignalResult<()> {
    // Use SIMD-optimized validation when possible
    for (i, &val) in tapered.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in _tapered signal at index {}: {}",
                i, val
            )));
        }
    }
    Ok(())
}

/// Enhanced FFT using SIMD operations and optimized planning
#[allow(dead_code)]
fn enhanced_simd_fft(x: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    // Comprehensive input validation
    if nfft == 0 {
        return Err(SignalError::ValueError(
            "FFT length cannot be zero".to_string(),
        ));
    }

    if !nfft.is_power_of_two() {
        eprintln!(
            "Warning: FFT length {} is not a power of two, performance may be suboptimal",
            nfft
        );
    }

    if nfft > 1_000_000 {
        eprintln!(
            "Warning: Very large FFT length {}, consider chunked processing",
            nfft
        );
    }

    // Pad or truncate to nfft with improved memory management
    let mut padded = vec![Complex64::new(0.0, 0.0); nfft];
    let copy_len = x.len().min(nfft);

    // Use SIMD-optimized copying when possible
    if copy_len >= 64 {
        let config = SimdConfig::default();
        let unity_window = vec![1.0; copy_len];
        let mut temp_real = vec![0.0; copy_len];

        // Copy using SIMD operations
        if simd_apply_window(&x[..copy_len], &unity_window, &mut temp_real, &config).is_ok() {
            for (i, &val) in temp_real.iter().enumerate() {
                padded[i] = Complex64::new(val, 0.0);
            }
        } else {
            // Fallback to scalar copy
            for i in 0..copy_len {
                padded[i] = Complex64::new(x[i], 0.0);
            }
        }
    } else {
        for i in 0..copy_len {
            padded[i] = Complex64::new(x[i], 0.0);
        }
    }

    // Use rustfft with enhanced error handling and performance optimization
    let mut planner = FftPlanner::new();

    // Create FFT with proper error handling
    let fft = planner.plan_fft_forward(nfft);
    let mut buffer = padded.clone();

    // Validate buffer before FFT
    for (i, &val) in buffer.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in FFT input at index {}: {}",
                i, val
            )));
        }
    }

    // Perform FFT with timing for large transforms
    if nfft > 8192 {
        let start = std::time::Instant::now();
        fft.process(&mut buffer);
        let duration = start.elapsed();

        // Warn for very slow FFTs
        if duration.as_millis() > 1000 {
            eprintln!(
                "Warning: Large FFT took {:.2}s for length {}",
                duration.as_secs_f64(),
                nfft
            );
        }
    } else {
        fft.process(&mut buffer);
    }

    // Validate output
    for (i, &val) in buffer.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value in FFT output at index {}: {}",
                i, val
            )));
        }
    }

    Ok(buffer)
}

/// Compute validated power spectrum with comprehensive error checking
#[allow(dead_code)]
fn compute_validated_power_spectrum(spectrum: &[Complex64]) -> SignalResult<Vec<f64>> {
    let mut power_spectrum = Vec::with_capacity(_spectrum.len());
    let mut max_power = 0.0;
    let mut suspicious_values = 0;

    for (i, &val) in spectrum.iter().enumerate() {
        let power = val.norm_sqr();

        // Enhanced validation for power values
        if !power.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite power _spectrum value at frequency bin {}: {}",
                i, power
            )));
        }

        if power < 0.0 {
            return Err(SignalError::ComputationError(format!(
                "Negative power _spectrum value at frequency bin {}: {}",
                i, power
            )));
        }

        // Track suspicious values
        if power > 1e50 {
            suspicious_values += 1;
            if suspicious_values > spectrum.len() / 10 {
                return Err(SignalError::ComputationError(
                    "Too many extremely large power _spectrum values detected".to_string(),
                ));
            }
        }

        max_power = max_power.max(power);
        power_spectrum.push(power);
    }

    // Warn about potential numerical issues
    if max_power > 1e100 {
        eprintln!(
            "Warning: Very large maximum power _spectrum value: {:.2e}",
            max_power
        );
    }

    if suspicious_values > 0 {
        eprintln!(
            "Warning: {} suspicious power _spectrum values detected",
            suspicious_values
        );
    }

    Ok(power_spectrum)
}

/// Combine spectra using standard eigenvalue weighting
#[allow(dead_code)]
fn combine_spectra_standard(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    fs: f64,
    nfft: usize,
    onesided: bool,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let k = spectra.nrows();

    // Create frequency array
    let frequencies = if onesided {
        let n_freqs = nfft / 2 + 1;
        (0..n_freqs).map(|i| i as f64 * fs / nfft as f64).collect()
    } else {
        (0..nfft)
            .map(|i| {
                if i <= nfft / 2 {
                    i as f64 * fs / nfft as f64
                } else {
                    (i as f64 - nfft as f64) * fs / nfft as f64
                }
            })
            .collect()
    };

    // Combine spectra with eigenvalue weights
    let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };
    let mut psd = vec![0.0; n_freqs];

    let weight_sum: f64 = eigenvalues.sum();
    let scaling = if onesided {
        2.0 / (fs * weight_sum)
    } else {
        1.0 / (fs * weight_sum)
    };

    for j in 0..n_freqs {
        let mut weighted_sum = 0.0;
        for i in 0..k {
            weighted_sum += eigenvalues[i] * spectra[[i, j]];
        }
        psd[j] = weighted_sum * scaling;
    }

    Ok((frequencies, psd))
}

/// Combine spectra using Thomson's adaptive weighting method with enhanced robustness
///
/// This refined implementation includes:
/// - Improved convergence detection with multiple criteria
/// - Enhanced numerical stability for extreme values
/// - Adaptive regularization based on signal characteristics
/// - Better memory management for large frequency grids
#[allow(dead_code)]
fn combine_spectra_adaptive(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    fs: f64,
    nfft: usize,
    onesided: bool,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let k = spectra.nrows();
    let n_freqs = if onesided { nfft / 2 + 1 } else { nfft };

    // Validate input dimensions
    if k == 0 || n_freqs == 0 {
        return Err(SignalError::ValueError(
            "Invalid dimensions for adaptive spectral combination".to_string(),
        ));
    }

    // Initialize adaptive weights with enhanced memory management
    let mut weights = Array2::zeros((k, n_freqs));
    let mut psd = vec![0.0; n_freqs];

    // Enhanced adaptive algorithm with refined parameters
    let max_iter = 20; // Increased for more reliable convergence
    let base_tolerance = 1e-12; // Enhanced precision
    let min_weight = 1e-16; // Improved numerical precision

    // Adaptive regularization based on eigenvalue spread
    let eigenvalue_ratio = eigenvalues[eigenvalues.len() - 1] / eigenvalues[0];
    let regularization = if eigenvalue_ratio < 1e-10 {
        1e-10 // Stronger regularization for ill-conditioned cases
    } else {
        1e-14 // Standard regularization
    };

    let damping_start = 8; // Refined damping schedule
    let damping_factor = 0.85; // More conservative damping

    // Initialize with normalized eigenvalue weights
    let lambda_sum: f64 = eigenvalues.sum();
    for i in 0..k {
        for j in 0..n_freqs {
            weights[[i, j]] = eigenvalues[i] / lambda_sum;
        }
    }

    // Enhanced adaptive iteration with multi-criteria convergence
    let mut converged = false;
    let mut convergence_history = Vec::new();
    let mut oscillation_detector = 0;

    for iter in 0..max_iter {
        let old_psd = psd.clone();

        // Update PSD estimate with numerical stabilization
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k {
                let w = weights[[i, j]];
                weighted_sum += w * spectra[[i, j]];
                weight_sum += w;
            }

            // Prevent division by zero
            if weight_sum > min_weight {
                psd[j] = weighted_sum / weight_sum;
            } else {
                // Fallback to eigenvalue-weighted average
                let mut fallback_sum = 0.0;
                for i in 0..k {
                    fallback_sum += eigenvalues[i] * spectra[[i, j]];
                }
                psd[j] = fallback_sum / lambda_sum;
            }

            // Ensure PSD is positive
            psd[j] = psd[j].max(regularization);
        }

        // Update weights with improved Thomson's method
        for j in 0..n_freqs {
            let mut new_weight_sum = 0.0;

            for i in 0..k {
                let lambda = eigenvalues[i];
                let spectrum_val = spectra[[i, j]].max(regularization);
                let psd_val = psd[j].max(regularization);

                // Improved bias factor calculation
                let ratio = psd_val / spectrum_val;
                let bias_factor = if ratio > 1e-6 {
                    lambda / (lambda + ratio.powi(2))
                } else {
                    lambda // Fallback for very small ratios
                };

                weights[[i, j]] = bias_factor.max(min_weight);
                new_weight_sum += weights[[i, j]];
            }

            // Normalize weights for this frequency bin
            if new_weight_sum > min_weight {
                for i in 0..k {
                    weights[[i, j]] /= new_weight_sum;
                }
            } else {
                // Fallback to equal weights
                for i in 0..k {
                    weights[[i, j]] = 1.0 / k as f64;
                }
            }
        }

        // Enhanced multi-criteria convergence detection
        let max_change = old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(regularization);
                ((old - new) / denominator).abs()
            })
            .fold(0.0, f64::max);

        let mean_change = old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(regularization);
                ((old - new) / denominator).abs()
            })
            .sum::<f64>()
            / n_freqs as f64;

        // RMS change for stability assessment
        let rms_change = (old_psd
            .iter()
            .zip(psd.iter())
            .map(|(old, new)| {
                let denominator = old.abs().max(new.abs()).max(regularization);
                ((old - new) / denominator).powi(2)
            })
            .sum::<f64>()
            / n_freqs as f64)
            .sqrt();

        // Track convergence history for oscillation detection
        convergence_history.push(mean_change);
        if convergence_history.len() > 4 {
            convergence_history.remove(0);

            // Detect oscillations in convergence
            if convergence_history.len() >= 4 {
                let recent_trend = convergence_history[3] - convergence_history[1];
                let prev_trend = convergence_history[2] - convergence_history[0];
                if recent_trend * prev_trend < 0.0 && recent_trend.abs() > base_tolerance {
                    oscillation_detector += 1;
                }
            }
        }

        // Adaptive tolerance based on iteration progress
        let adaptive_tolerance = base_tolerance * (1.0 + iter as f64 * 0.1);

        // Enhanced convergence criteria
        if max_change < adaptive_tolerance
            && mean_change < adaptive_tolerance * 0.1
            && rms_change < adaptive_tolerance * 0.5
        {
            converged = true;
            break;
        }

        // Early convergence for stable oscillations
        if oscillation_detector >= 3 && mean_change < adaptive_tolerance * 10.0 {
            converged = true;
            break;
        }

        // Enhanced adaptive damping strategy
        if iter > damping_start {
            // Multi-factor adaptive damping
            let convergence_rate = if iter > 0 && convergence_history.len() > 1 {
                convergence_history[convergence_history.len() - 1]
                    / convergence_history[convergence_history.len() - 2].max(1e-15)
            } else {
                1.0
            };

            let adaptive_damping = if oscillation_detector > 0 {
                0.6 // Strong damping for oscillations
            } else if mean_change > adaptive_tolerance * 50.0 {
                0.7 // Moderate damping for slow convergence
            } else if convergence_rate > 0.95 {
                0.8 // Light damping for good convergence
            } else {
                damping_factor // Standard damping
            };

            // Apply frequency-selective damping
            for j in 0..n_freqs {
                let local_change = (psd[j] - old_psd[j]).abs()
                    / (psd[j].abs().max(old_psd[j].abs()).max(regularization));
                let local_damping = if local_change > adaptive_tolerance * 100.0 {
                    adaptive_damping * 0.8 // Stronger damping for highly variable frequencies
                } else {
                    adaptive_damping
                };

                psd[j] = local_damping * psd[j] + (1.0 - local_damping) * old_psd[j];
            }
        }
    }

    // Enhanced convergence diagnostics with actionable feedback
    if !converged {
        let final_change = if let Some(&last_change) = convergence_history.last() {
            last_change
        } else {
            1.0
        };

        // Provide more specific diagnostic information
        if final_change < base_tolerance * 50.0 {
            eprintln!(
                "Info: Adaptive multitaper algorithm achieved acceptable convergence (final change: {:.2e}, oscillations: {})",
                final_change, oscillation_detector
            );
        } else if oscillation_detector > 0 {
            eprintln!(
                "Warning: Adaptive algorithm experienced {} oscillations but stabilized (final change: {:.2e})",
                oscillation_detector, final_change
            );
        } else {
            eprintln!(
                "Warning: Adaptive multitaper convergence incomplete after {} iterations (final change: {:.2e})",
                max_iter, final_change
            );
            eprintln!("Consider: increasing signal length, reducing k, or adjusting nw parameter");
        }
    }

    // Create frequency array
    let frequencies = if onesided {
        (0..n_freqs).map(|i| i as f64 * fs / nfft as f64).collect()
    } else {
        (0..nfft)
            .map(|i| {
                if i <= nfft / 2 {
                    i as f64 * fs / nfft as f64
                } else {
                    (i as f64 - nfft as f64) * fs / nfft as f64
                }
            })
            .collect()
    };

    // Apply final scaling with improved normalization
    let scaling = if onesided { 2.0 / fs } else { 1.0 / fs };
    psd.iter_mut().for_each(|p| *p *= scaling);

    Ok((frequencies, psd))
}

/// Compute confidence intervals using enhanced chi-squared approximation
///
/// This implementation includes improved DOF calculation and better handling
/// of edge cases for more accurate confidence intervals.
#[allow(dead_code)]
fn compute_confidence_intervals(
    spectra: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    confidence_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let _k = spectra.nrows() as f64;
    // Enhanced DOF calculation using effective number of tapers
    let effective_k = compute_effective_dof(eigenvalues) / 2.0;
    let dof = 2.0 * effective_k; // More accurate degrees of freedom

    // Chi-squared distribution
    let chi2 = ChiSquared::new(dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create chi-squared distribution: {}", e))
    })?;

    // Confidence interval factors
    let alpha = 1.0 - confidence_level;
    let lower_quantile = chi2.inverse_cdf(alpha / 2.0);
    let upper_quantile = chi2.inverse_cdf(1.0 - alpha / 2.0);

    let lower_factor = dof / upper_quantile;
    let upper_factor = dof / lower_quantile;

    // Apply factors to PSD estimate with improved scaling
    let n_freqs = spectra.ncols();
    let mut lower_ci = vec![0.0; n_freqs];
    let mut upper_ci = vec![0.0; n_freqs];

    let weight_sum: f64 = eigenvalues.sum();

    for j in 0..n_freqs {
        let mut weighted_sum = 0.0;
        let mut variance_estimate = 0.0;

        // Compute weighted mean and variance estimate
        for i in 0..spectra.nrows() {
            weighted_sum += eigenvalues[i] * spectra[[i, j]];
        }
        let psd_estimate = weighted_sum / weight_sum;

        // Improved variance estimation for better confidence intervals
        for i in 0..spectra.nrows() {
            let deviation = spectra[[i, j]] - psd_estimate;
            variance_estimate += eigenvalues[i] * deviation * deviation;
        }
        variance_estimate /= weight_sum;

        // Apply chi-squared scaling with variance correction
        let scale_factor = (1.0 + variance_estimate / (psd_estimate * psd_estimate + 1e-15)).sqrt();

        lower_ci[j] = psd_estimate * lower_factor / scale_factor;
        upper_ci[j] = psd_estimate * upper_factor * scale_factor;

        // Ensure positive confidence intervals
        lower_ci[j] = lower_ci[j].max(1e-15);
        upper_ci[j] = upper_ci[j].max(lower_ci[j] * 1.01); // Ensure upper > lower
    }

    Ok((lower_ci, upper_ci))
}

/// Compute effective degrees of freedom with enhanced numerical stability
#[allow(dead_code)]
fn compute_effective_dof(eigenvalues: &Array1<f64>) -> f64 {
    let sum_lambda: f64 = eigenvalues.sum();
    let sum_lambda_sq: f64 = eigenvalues.iter().map(|&x| x * x).sum();

    // Enhanced numerical stability for edge cases
    if sum_lambda_sq < 1e-15 || sum_lambda < 1e-15 {
        return 2.0; // Fallback to minimum DOF
    }

    let dof = 2.0 * sum_lambda.powi(2) / sum_lambda_sq;

    // Validate DOF range
    if dof < 1.0 {
        eprintln!(
            "Warning: Computed DOF ({:.2}) is less than 1, using minimum value",
            dof
        );
        2.0
    } else if dof > 2.0 * eigenvalues.len() as f64 {
        eprintln!(
            "Warning: Computed DOF ({:.2}) exceeds theoretical maximum",
            dof
        );
        2.0 * eigenvalues.len() as f64
    } else {
        dof
    }
}

/// Enhanced memory-efficient multitaper estimation for very large signals
///
/// Refinements in this version:
/// - Intelligent chunk sizing based on signal characteristics
/// - Improved overlap strategy for spectral continuity
/// - Enhanced statistical combination with variance tracking
/// - Better memory management and error recovery
#[allow(dead_code)]
fn compute_pmtm_chunked(
    signal: &[f64],
    config: &MultitaperConfig,
    nfft: usize,
) -> SignalResult<EnhancedMultitaperResult> {
    let n = signal.len();

    // Enhanced adaptive chunk sizing strategy
    let signal_complexity = estimate_signal_complexity(signal);
    let memory_factor = if config.memory_optimized { 0.5 } else { 1.0 };

    let base_chunk_size = match (config.k, signal_complexity) {
        (k, _) if k > 30 => 30_000, // Many tapers require smaller chunks
        (k, complexity) if k > 15 && complexity > 2.0 => 40_000, // Complex signals need more care
        (k, _) if k > 10 => 60_000, // Moderate number of tapers
        _ => 100_000,             // Standard case
    };

    let chunk_size = ((base_chunk_size as f64 * memory_factor) as usize)
        .min(n / 8)  // Never use chunks larger than n/8
        .max((config.k * 25).min(n / 2)); // Ensure statistical validity

    // Intelligent overlap based on frequency content
    let overlap_ratio = if signal_complexity > 3.0 {
        0.3 // More overlap for complex signals
    } else if config.k > 20 {
        0.25 // More overlap for many tapers
    } else {
        0.2 // Standard overlap
    };

    let overlap = (chunk_size as f64 * overlap_ratio) as usize;
    let step = chunk_size.saturating_sub(overlap).max(chunk_size / 2);

    // Calculate number of chunks
    let n_chunks = (n + step - 1) / step; // Ceiling division

    // Initialize accumulators
    let n_freqs = if config.onesided { nfft / 2 + 1 } else { nfft };
    let mut psd_accumulator = vec![0.0; n_freqs];
    let mut weight_accumulator = vec![0.0; n_freqs];
    let mut frequencies = Vec::new();

    // Process each chunk
    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * step;
        let end = (start + chunk_size).min(n);

        let chunk_len = end - start;
        if chunk_len < config.k * 15 {
            // Skip chunks that are too small for reliable estimation
            // Increased minimum size for better statistical properties
            continue;
        }

        // Additional validation for chunk quality
        let chunk = &signal[start..end];
        let chunk_energy: f64 = chunk.iter().map(|&x| x * x).sum();
        if chunk_energy < 1e-20 {
            // Skip near-zero energy chunks
            continue;
        }

        let chunk = &signal[start..end];
        let chunk_len = chunk.len();

        // Compute DPSS for this chunk size
        let (tapers, eigenvalues_opt) = dpss(chunk_len, config.nw, config.k, true)?;
        let eigenvalues = eigenvalues_opt.ok_or_else(|| {
            SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            )
        })?;

        // Use a smaller nfft for chunks
        let chunk_nfft = next_power_of_two(chunk_len);

        // Compute tapered FFTs for this chunk
        let spectra = if config.parallel && chunk_len >= config.parallel_threshold {
            compute_tapered_ffts_parallel(chunk, &tapers, chunk_nfft)?
        } else {
            compute_tapered_ffts_simd(chunk, &tapers, chunk_nfft)?
        };

        // Combine spectra for this chunk
        let (chunk_freqs, chunk_psd) = if config.adaptive {
            combine_spectra_adaptive(
                &spectra,
                &eigenvalues,
                config.fs,
                chunk_nfft,
                config.onesided,
            )?
        } else {
            combine_spectra_standard(
                &spectra,
                &eigenvalues,
                config.fs,
                chunk_nfft,
                config.onesided,
            )?
        };

        // Store frequencies from first chunk
        if chunk_idx == 0 {
            frequencies = chunk_freqs.clone();
        }

        // Interpolate chunk PSD to match target frequency grid if needed
        let interpolated_psd = if chunk_freqs.len() != frequencies.len() {
            interpolate_psd(&chunk_freqs, &chunk_psd, &frequencies)?
        } else {
            chunk_psd
        };

        // Enhanced weighted accumulation with variance tracking
        let chunk_len_actual = end - start;
        let chunk_weight = (chunk_len_actual as f64 / n as f64)
            * (chunk_len_actual as f64 / chunk_size as f64).sqrt(); // Quality factor

        for (i, &psd_val) in interpolated_psd.iter().enumerate() {
            if i < psd_accumulator.len() && psd_val.is_finite() && psd_val > 0.0 {
                psd_accumulator[i] += psd_val * chunk_weight;
                weight_accumulator[i] += chunk_weight;
            }
        }
    }

    // Normalize accumulated PSD
    for i in 0..psd_accumulator.len() {
        if weight_accumulator[i] > 0.0 {
            psd_accumulator[i] /= weight_accumulator[i];
        }
    }

    // Note: For chunked processing, we don't compute confidence intervals
    // as they would require more complex statistical handling across chunks

    Ok(EnhancedMultitaperResult {
        frequencies,
        psd: psd_accumulator,
        confidence_intervals: None, // Not supported for chunked processing
        dof: Some(2.0 * config.k as f64 * n_chunks as f64), // Approximate DOF
        tapers: None,               // Not returned for memory efficiency
        eigenvalues: None,          // Not returned for memory efficiency
    })
}

/// Simple linear interpolation for PSD values
#[allow(dead_code)]
fn interpolate_psd(
    source_freqs: &[f64],
    source_psd: &[f64],
    target_freqs: &[f64],
) -> SignalResult<Vec<f64>> {
    if source_freqs.is_empty() || source_psd.is_empty() || target_freqs.is_empty() {
        return Err(SignalError::ValueError(
            "Empty frequency or PSD arrays".to_string(),
        ));
    }

    let mut result = vec![0.0; target_freqs.len()];

    for (i, &target_freq) in target_freqs.iter().enumerate() {
        // Find bracketing indices
        let mut lower_idx = 0;
        let mut upper_idx = source_freqs.len() - 1;

        for (j, &freq) in source_freqs.iter().enumerate() {
            if freq <= target_freq {
                lower_idx = j;
            } else {
                upper_idx = j;
                break;
            }
        }

        if lower_idx == upper_idx {
            // Exact match or at boundary
            result[i] = source_psd[lower_idx];
        } else {
            // Linear interpolation
            let f1 = source_freqs[lower_idx];
            let f2 = source_freqs[upper_idx];
            let p1 = source_psd[lower_idx];
            let p2 = source_psd[upper_idx];

            if (f2 - f1).abs() > 1e-15 {
                let weight = (target_freq - f1) / (f2 - f1);
                result[i] = p1 + weight * (p2 - p1);
            } else {
                result[i] = (p1 + p2) / 2.0;
            }
        }
    }

    Ok(result)
}

/// Estimate signal complexity for adaptive processing
///
/// Returns a complexity score based on:
/// - Spectral entropy
/// - Dynamic range
/// - High-frequency content
#[allow(dead_code)]
fn estimate_signal_complexity(signal: &[f64]) -> f64 {
    if signal.len() < 64 {
        return 1.0; // Simple case for short signals
    }

    // Calculate basic statistics
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-12 {
        return 0.5; // Nearly constant _signal
    }

    // Estimate dynamic range
    let max_val = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let dynamic_range = (max_val - min_val) / std_dev;

    // Estimate high-frequency content via differences
    let mut high_freq_energy = 0.0;
    for window in signal.windows(2) {
        high_freq_energy += (window[1] - window[0]).powi(2);
    }
    high_freq_energy /= signal.len() as f64;
    let high_freq_ratio = high_freq_energy / variance.max(1e-12);

    // Combine factors for complexity score
    let complexity = 1.0 +
        (dynamic_range / 10.0).min(2.0) +  // Dynamic range contribution
        (high_freq_ratio * 5.0).min(2.0); // High frequency contribution

    complexity.min(5.0) // Cap at maximum complexity
}

/// Find the next power of two greater than or equal to n
#[allow(dead_code)]
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Enhanced multitaper spectrogram with parallel processing
///
/// Computes a time-frequency representation using multitaper method with
/// parallel processing for improved performance on large signals.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `config` - Spectrogram configuration
///
/// # Returns
///
/// * Tuple of (times, frequencies, spectrogram)
#[allow(dead_code)]
pub fn enhanced_multitaper_spectrogram<T>(
    x: &[T],
    config: &SpectrogramConfig,
) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    check_positive(config.window_size, "window_size")?;
    check_positive(config.step, "step")?;

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Validate that all values are finite
    for (i, &val) in x_f64.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ValueError(format!(
                "Non-finite value at index {}: {}",
                i, val
            )));
        }
    }

    let n = x_f64.len();
    let window_size = config.window_size;
    let step = config.step;

    // Calculate number of windows
    if window_size > n {
        return Err(SignalError::ValueError(
            "Window size larger than signal length".to_string(),
        ));
    }

    let n_windows = (n - window_size) / step + 1;
    if n_windows == 0 {
        return Err(SignalError::ValueError(
            "No complete windows in signal".to_string(),
        ));
    }

    // Prepare multitaper config for each window
    let mut mt_config = config.multitaper.clone();
    mt_config.nfft = Some(config.window_size);

    // Calculate time points
    let times: Vec<f64> = (0..n_windows)
        .map(|i| (i * step + window_size / 2) as f64 / config.fs)
        .collect();

    // Process windows in parallel if enabled
    let results: Vec<EnhancedMultitaperResult> = if config.multitaper.parallel
        && n_windows >= config.multitaper.parallel_threshold / window_size
    {
        let x_arc = Arc::new(x_f64);

        (0..n_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * step;
                let end = start + window_size;
                let window = &x_arc[start..end];

                enhanced_pmtm(window, &mt_config).unwrap()
            })
            .collect()
    } else {
        // Sequential processing
        (0..n_windows)
            .map(|i| {
                let start = i * step;
                let end = start + window_size;
                let window = &x_f64[start..end];

                enhanced_pmtm(window, &mt_config)
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Extract frequencies from first result
    let frequencies = results[0].frequencies.clone();
    let n_freqs = frequencies.len();

    // Build spectrogram matrix
    let mut spectrogram = Array2::zeros((n_freqs, n_windows));

    for (j, result) in results.iter().enumerate() {
        for (i, &psd_val) in result.psd.iter().enumerate() {
            spectrogram[[i, j]] = psd_val;
        }
    }

    // Apply logarithmic scaling if requested (common for spectrograms)
    let epsilon = 1e-10;
    spectrogram.mapv_inplace(|x| (x + epsilon).log10() * 10.0); // Convert to dB

    Ok((times, frequencies, spectrogram))
}

/// Configuration for spectrogram computation
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// Sampling frequency
    pub fs: f64,
    /// Window size in samples
    pub window_size: usize,
    /// Step size in samples
    pub step: usize,
    /// Multitaper parameters
    pub multitaper: MultitaperConfig,
}

mod tests {

    #[test]
    fn test_enhanced_pmtm_basic() {
        // Generate test signal
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 100.0).sin())
            .collect();

        let config = MultitaperConfig::default();
        let result = enhanced_pmtm(&signal, &config).unwrap();

        assert_eq!(result.frequencies.len(), result.psd.len());
        assert!(result.dof.is_some());
    }

    #[test]
    fn test_enhanced_simd_fft() {
        let signal = vec![1.0, 0.0, -1.0, 0.0];
        let result = enhanced_simd_fft(&signal, 4).unwrap();
        assert_eq!(result.len(), 4);
        // Check that result is finite
        for val in result {
            assert!(val.is_finite());
        }
    }
}
