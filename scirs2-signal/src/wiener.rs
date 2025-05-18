//! Wiener filtering module
//!
//! This module implements Wiener filtering techniques for signal denoising and restoration.
//! Wiener filters are optimal linear filters for signal restoration in the presence of
//! additive noise, based on a statistical approach.
//!
//! The implementation includes:
//! - Time-domain Wiener filtering for 1D signals
//! - Frequency-domain Wiener filtering for noise reduction
//! - Adaptive Wiener filtering with local variance estimation
//! - Iterative Wiener filtering for improved restoration
//!
//! # Example
//! ```ignore
//! use ndarray::Array1;
//! use scirs2_signal::wiener::wiener_filter;
//! use scirs2_signal::waveforms;
//! use rand::Rng;
//!
//! // Create a test signal
//! let fs = 1000.0;
//! let t = Array1::linspace(0.0, 1.0, 1000);
//! let clean_signal = waveforms::chirp(
//!     &t, 10.0, 1.0, 100.0, Some("linear")
//! ).unwrap();
//!
//! // Add noise
//! let mut rng = rand::thread_rng();
//! let mut noisy_signal = clean_signal.clone();
//! for i in 0..noisy_signal.len() {
//!     noisy_signal[i] += 0.5 * rng.random_range(-1.0..1.0);
//! }
//!
//! // Apply Wiener filter
//! let denoised_signal = wiener_filter(&noisy_signal, None, None).unwrap();
//! ```

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use std::cmp;

use crate::error::{SignalError, SignalResult};
use scirs2_fft;

/// Configuration for Wiener filtering
#[derive(Debug, Clone)]
pub struct WienerConfig {
    /// Filter window size for local variance estimation
    pub window_size: usize,

    /// Noise power estimate (None = auto-estimate from signal)
    pub noise_power: Option<f64>,

    /// Whether to use frequency-domain filtering
    pub frequency_domain: bool,

    /// Maximum number of iterations for iterative filtering
    pub max_iterations: usize,

    /// Prior ratio of signal power to noise power (SNR)
    pub prior_snr: Option<f64>,

    /// Regularization parameter to prevent division by zero
    pub regularization: f64,

    /// Whether to apply boundary reflection
    pub boundary: bool,
}

impl Default for WienerConfig {
    fn default() -> Self {
        Self {
            window_size: 15,
            noise_power: None,
            frequency_domain: true,
            max_iterations: 1,
            prior_snr: None,
            regularization: 1e-10,
            boundary: true,
        }
    }
}

/// Applies a Wiener filter to a noisy signal for noise reduction.
///
/// The Wiener filter is an optimal linear filter for reducing additive noise,
/// assuming stationary signal and noise processes. It works by estimating
/// the signal and noise power spectral densities.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `noise_power` - Estimated noise power (variance); if None, it's estimated from the signal
/// * `window_size` - Window size for local variance estimation; if None, default is used
///
/// # Returns
/// * The denoised signal
///
/// # Example
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::wiener::wiener_filter;
///
/// let noisy_signal = Array1::from_vec(vec![1.2, 2.3, 3.1, 2.2, 1.3, 0.2, -0.3, -1.1]);
/// let denoised = wiener_filter(&noisy_signal, None, None).unwrap();
/// ```
pub fn wiener_filter(
    signal: &Array1<f64>,
    noise_power: Option<f64>,
    window_size: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let mut config = WienerConfig::default();

    if let Some(np) = noise_power {
        config.noise_power = Some(np);
    }

    if let Some(ws) = window_size {
        config.window_size = ws;
    }

    // Use frequency-domain Wiener filter for efficiency
    wiener_filter_freq(signal, &config)
}

/// Applies a frequency-domain Wiener filter to a noisy signal.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `config` - Wiener filter configuration
///
/// # Returns
/// * The denoised signal
pub fn wiener_filter_freq(
    signal: &Array1<f64>,
    config: &WienerConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Estimate noise power if not provided
    let noise_power = match config.noise_power {
        Some(np) => np,
        None => estimate_noise_power(signal)?,
    };

    // Apply FFT to signal
    let signal_vec = signal.to_vec();
    let fft_result = match scirs2_fft::fft(&signal_vec, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute FFT for Wiener filter".to_string(),
            ))
        }
    };

    // Apply Wiener filter in frequency domain
    // H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
    let mut filtered_fft = Vec::with_capacity(fft_result.len());

    for &freq_bin in &fft_result {
        // Compute power
        let power = freq_bin.norm_sqr();

        // Apply Wiener filter
        // If we have prior SNR, use it to guide the filtering
        let snr_factor = config.prior_snr.unwrap_or(1.0);

        // Wiener filter transfer function
        let wiener_gain = power / (power + snr_factor * noise_power + config.regularization);

        // Apply gain to frequency bin
        filtered_fft.push(freq_bin * wiener_gain);
    }

    // Inverse FFT to get filtered signal
    let ifft_result = match scirs2_fft::ifft(&filtered_fft, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute inverse FFT for Wiener filter".to_string(),
            ))
        }
    };

    // Extract real part of result
    let denoised = Array1::from_iter(ifft_result.iter().take(n).map(|c| c.re));

    Ok(denoised)
}

/// Applies a time-domain Wiener filter to a noisy signal.
///
/// This implementation uses a sliding window approach to estimate local signal
/// and noise statistics, adapting the filter to non-stationary signals.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `config` - Wiener filter configuration
///
/// # Returns
/// * The denoised signal
pub fn wiener_filter_time(
    signal: &Array1<f64>,
    config: &WienerConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Ensure window size is valid
    if config.window_size < 3 {
        return Err(SignalError::ValueError(
            "Window size must be at least 3".to_string(),
        ));
    }

    // Pad signal if boundary reflection is enabled
    let half_window = config.window_size / 2;
    let padded_signal = if config.boundary {
        pad_signal(signal, half_window)
    } else {
        signal.clone()
    };

    // Estimate noise power if not provided
    let global_noise_power = match config.noise_power {
        Some(np) => np,
        None => estimate_noise_power(signal)?,
    };

    // Initialize denoised signal
    let mut denoised = Array1::zeros(n);

    // Apply adaptive Wiener filter with sliding window
    for i in 0..n {
        // Extract window centered at current sample
        let start = if config.boundary {
            i
        } else {
            cmp::max(i, half_window) - half_window
        };

        let end = cmp::min(
            if config.boundary {
                i + config.window_size
            } else {
                padded_signal.len()
            },
            start + config.window_size,
        );

        if end <= start {
            return Err(SignalError::DimensionError(
                "Invalid window bounds in Wiener filter".to_string(),
            ));
        }

        let window = padded_signal.slice(s![start..end]);

        // Compute local mean and variance
        let local_mean = window.mean().unwrap_or(0.0);
        let local_var = window
            .iter()
            .map(|&x| (x - local_mean).powi(2))
            .sum::<f64>()
            / window.len() as f64;

        // Apply Wiener filter formula
        // Wiener filter: w = max(0, (local_var - noise_var) / local_var)
        // Denoised sample = local_mean + w * (sample - local_mean)
        let sample_idx = if config.boundary { i + half_window } else { i };
        let sample = padded_signal[sample_idx];

        // Ensure variance is non-negative
        let max_var = 0.0_f64.max(local_var - global_noise_power);
        let wiener_weight = if local_var > config.regularization {
            max_var / (local_var + config.regularization)
        } else {
            0.0
        };

        // Apply filter
        denoised[i] = local_mean + wiener_weight * (sample - local_mean);
    }

    Ok(denoised)
}

/// Applies an iterative Wiener filter for enhanced signal restoration.
///
/// The iterative approach refines the signal estimate in each iteration,
/// potentially achieving better restoration than a single-pass filter.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `config` - Wiener filter configuration
///
/// # Returns
/// * The denoised signal
pub fn iterative_wiener_filter(
    signal: &Array1<f64>,
    config: &WienerConfig,
) -> SignalResult<Array1<f64>> {
    // Use provided noise power or estimate it
    let noise_power = match config.noise_power {
        Some(np) => np,
        None => estimate_noise_power(signal)?,
    };

    // Initialize with noisy signal
    let mut current_estimate = signal.clone();

    // Iterative refinement
    for _ in 0..config.max_iterations {
        // Create config with current estimate of SNR
        let mut iter_config = config.clone();

        // Estimate signal power from current estimate
        let signal_power = estimate_signal_power(&current_estimate)?;

        // Skip iteration if signal power is too small
        if signal_power < config.regularization {
            break;
        }

        // Update SNR estimate for this iteration
        let snr = signal_power / noise_power;
        iter_config.prior_snr = Some(snr);

        // Apply Wiener filter with updated parameters
        if config.frequency_domain {
            current_estimate = wiener_filter_freq(&current_estimate, &iter_config)?;
        } else {
            current_estimate = wiener_filter_time(&current_estimate, &iter_config)?;
        }
    }

    Ok(current_estimate)
}

/// Applies a 2D Wiener filter to a noisy image.
///
/// This is useful for image denoising and restoration.
///
/// # Arguments
/// * `image` - 2D array representing a noisy image
/// * `noise_power` - Estimated noise variance (None = auto-estimate)
/// * `window_size` - Window size for local variance estimation (None = default [5,5])
///
/// # Returns
/// * The denoised image
pub fn wiener_filter_2d(
    image: &Array2<f64>,
    noise_power: Option<f64>,
    window_size: Option<[usize; 2]>,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Set default window size or use provided
    let win_size = window_size.unwrap_or([5, 5]);

    // Validate window size
    if win_size[0] < 3 || win_size[1] < 3 {
        return Err(SignalError::ValueError(
            "Window size must be at least 3x3".to_string(),
        ));
    }

    // Estimate noise power if not provided
    let noise_var = match noise_power {
        Some(np) => np,
        None => {
            // Flatten image and estimate noise
            let flat_image = Array1::from_iter(image.iter().cloned());
            estimate_noise_power(&flat_image)?
        }
    };

    // Initialize denoised image
    let mut denoised = Array2::zeros((height, width));

    // Half window sizes
    let half_h = win_size[0] / 2;
    let half_w = win_size[1] / 2;

    // Apply Wiener filter with sliding window
    for i in 0..height {
        for j in 0..width {
            // Determine window boundaries (with boundary handling)
            let row_start = i.saturating_sub(half_h);
            let row_end = (i + half_h + 1).min(height);
            let col_start = j.saturating_sub(half_w);
            let col_end = (j + half_w + 1).min(width);

            // Extract window
            let window = image.slice(s![row_start..row_end, col_start..col_end]);

            // Compute local mean and variance
            let local_mean = window.mean().unwrap_or(0.0);
            let local_var = window
                .iter()
                .map(|&x| (x - local_mean).powi(2))
                .sum::<f64>()
                / (window.len() as f64);

            // Apply Wiener filter formula
            let max_var = 0.0_f64.max(local_var - noise_var);
            let wiener_weight = if local_var > 1e-10 {
                max_var / (local_var + 1e-10)
            } else {
                0.0
            };

            // Filter pixel
            denoised[[i, j]] = local_mean + wiener_weight * (image[[i, j]] - local_mean);
        }
    }

    Ok(denoised)
}

/// Applies spectral subtraction for noise reduction.
///
/// Spectral subtraction is a frequency-domain technique that estimates and
/// subtracts the noise spectrum from the signal spectrum.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `noise_power` - Estimated noise power spectrum (None = auto-estimate)
/// * `alpha` - Oversubtraction factor (typically 1.0 to 2.0)
/// * `beta` - Spectral floor parameter (0.0 to 1.0)
///
/// # Returns
/// * The denoised signal
pub fn spectral_subtraction(
    signal: &Array1<f64>,
    noise_power: Option<&Array1<f64>>,
    alpha: Option<f64>,
    beta: Option<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Set default parameters
    let alpha_factor = alpha.unwrap_or(1.0);
    let beta_factor = beta.unwrap_or(0.01);

    // Apply FFT to signal
    let signal_vec = signal.to_vec();
    let fft_result = match scirs2_fft::fft(&signal_vec, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute FFT for spectral subtraction".to_string(),
            ))
        }
    };

    // Get noise power spectrum
    let noise_spectrum = match noise_power {
        Some(np) => np.clone(),
        None => {
            // Estimate noise power from the signal
            // (Use first 100ms or 5% of signal, whichever is smaller)
            let noise_samples = (n as f64 * 0.05).min(100.0) as usize;
            if noise_samples < 4 {
                return Err(SignalError::ValueError(
                    "Signal too short to estimate noise spectrum".to_string(),
                ));
            }

            // Use first few samples as noise estimate
            let noise_segment = signal.slice(s![0..noise_samples]);

            // Compute noise spectrum
            match scirs2_fft::fft(&noise_segment.to_vec(), Some(n)) {
                Ok(noise_fft) => {
                    // Convert to power spectrum
                    Array1::from_iter(
                        noise_fft
                            .iter()
                            .take(n / 2 + 1)
                            .map(|c| c.norm_sqr() / n as f64),
                    )
                }
                Err(_) => {
                    return Err(SignalError::ComputationError(
                        "Failed to compute noise spectrum".to_string(),
                    ))
                }
            }
        }
    };

    // Apply spectral subtraction
    let mut filtered_fft = fft_result.clone();

    // Process only positive frequencies (up to Nyquist)
    for i in 0..=n / 2 {
        // Get magnitude and phase
        let mag = filtered_fft[i].norm();
        let phase = filtered_fft[i].arg();

        // Compute noise power for this bin
        let noise_power = if i < noise_spectrum.len() {
            noise_spectrum[i]
        } else {
            noise_spectrum[noise_spectrum.len() - 1]
        };

        // Apply spectral subtraction
        // S(f) = max(|X(f)| - α|N(f)|, β|X(f)|)
        let new_mag = (mag.powi(2) - alpha_factor * noise_power)
            .max(beta_factor * mag.powi(2))
            .sqrt();

        // Reconstruct complex value with new magnitude and original phase
        filtered_fft[i] = Complex64::from_polar(new_mag, phase);

        // Apply to symmetric frequency (except DC and Nyquist)
        if i > 0 && i < n / 2 {
            filtered_fft[n - i] = Complex64::from_polar(new_mag, -phase);
        }
    }

    // Inverse FFT to get filtered signal
    let ifft_result = match scirs2_fft::ifft(&filtered_fft, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute inverse FFT for spectral subtraction".to_string(),
            ))
        }
    };

    // Extract real part of result
    let denoised = Array1::from_iter(ifft_result.iter().take(n).map(|c| c.re));

    Ok(denoised)
}

/// Applies a power spectral density (PSD) based Wiener filter.
///
/// This method is particularly useful when the power spectral densities
/// of the signal and noise are known or can be estimated.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `signal_psd` - Power spectral density of the clean signal (None = estimate)
/// * `noise_psd` - Power spectral density of the noise (None = estimate)
///
/// # Returns
/// * The denoised signal
pub fn psd_wiener_filter(
    signal: &Array1<f64>,
    signal_psd: Option<&Array1<f64>>,
    noise_psd: Option<&Array1<f64>>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Apply FFT to signal
    let signal_vec = signal.to_vec();
    let fft_result = match scirs2_fft::fft(&signal_vec, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute FFT for PSD Wiener filter".to_string(),
            ))
        }
    };

    // Estimate signal PSD if not provided
    let s_psd = match signal_psd {
        Some(psd) => psd.clone(),
        None => {
            // Simple periodogram estimate
            let half_n = n / 2 + 1;
            let mut psd_est = Array1::zeros(half_n);
            for i in 0..half_n {
                psd_est[i] = fft_result[i].norm_sqr() / n as f64;
            }

            // Apply smoothing
            smooth_psd(&psd_est)
        }
    };

    // Estimate noise PSD if not provided
    let n_psd = match noise_psd {
        Some(psd) => psd.clone(),
        None => {
            // Use noise estimation method
            let noise_var = estimate_noise_power(signal)?;

            // Create flat noise PSD
            Array1::from_elem(n / 2 + 1, noise_var)
        }
    };

    // Apply Wiener filter in frequency domain
    let mut filtered_fft = fft_result.clone();

    // Process only positive frequencies (up to Nyquist)
    for i in 0..=n / 2 {
        // Get magnitude and phase
        let mag = filtered_fft[i].norm();
        let phase = filtered_fft[i].arg();

        // Get signal and noise PSDs for this bin
        let s_power = if i < s_psd.len() { s_psd[i] } else { 0.0 };
        let n_power = if i < n_psd.len() { n_psd[i] } else { 0.0 };

        // Compute Wiener filter gain
        // H(f) = S(f) / (S(f) + N(f))
        let wiener_gain = if s_power + n_power > 1e-10 {
            s_power / (s_power + n_power)
        } else {
            0.0
        };

        // Apply gain
        let new_mag = mag * wiener_gain;

        // Reconstruct complex value
        filtered_fft[i] = Complex64::from_polar(new_mag, phase);

        // Apply to symmetric frequency (except DC and Nyquist)
        if i > 0 && i < n / 2 {
            filtered_fft[n - i] = Complex64::from_polar(new_mag, -phase);
        }
    }

    // Inverse FFT to get filtered signal
    let ifft_result = match scirs2_fft::ifft(&filtered_fft, None) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute inverse FFT for PSD Wiener filter".to_string(),
            ))
        }
    };

    // Extract real part
    let denoised = Array1::from_iter(ifft_result.iter().take(n).map(|c| c.re));

    Ok(denoised)
}

/// Applies a Kalman-Wiener filter for time-varying noise conditions.
///
/// This implementation combines aspects of Kalman filtering and Wiener filtering
/// to handle non-stationary noise conditions.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `process_var` - Process noise variance (state model)
/// * `measurement_var` - Measurement noise variance
///
/// # Returns
/// * The denoised signal
pub fn kalman_wiener_filter(
    signal: &Array1<f64>,
    process_var: f64,
    measurement_var: f64,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Validate input parameters
    if process_var <= 0.0 || measurement_var <= 0.0 {
        return Err(SignalError::ValueError(
            "Variance parameters must be positive".to_string(),
        ));
    }

    // Initialize filter state
    let mut x_est = 0.0; // State estimate
    let mut p_est = 1.0; // Error covariance

    // Initialize output array
    let mut filtered = Array1::zeros(n);

    // Apply Kalman filter
    for i in 0..n {
        // Prediction step
        let x_pred = x_est;
        let p_pred = p_est + process_var;

        // Update step
        let k_gain = p_pred / (p_pred + measurement_var);
        x_est = x_pred + k_gain * (signal[i] - x_pred);
        p_est = (1.0 - k_gain) * p_pred;

        // Store filtered value
        filtered[i] = x_est;
    }

    Ok(filtered)
}

/// Helper function to estimate the noise power from a signal
fn estimate_noise_power(signal: &Array1<f64>) -> SignalResult<f64> {
    // Compute signal median
    let mut values = signal.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };

    // Compute median absolute deviation (MAD)
    let mut deviations = values
        .iter()
        .map(|&x| (x - median).abs())
        .collect::<Vec<f64>>();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    // Estimate noise variance (for Gaussian noise)
    // Scale factor 1.4826 makes MAD consistent with standard deviation for normal distribution
    let noise_std = 1.4826 * mad;
    let noise_var = noise_std.powi(2);

    Ok(noise_var)
}

/// Helper function to estimate the signal power
fn estimate_signal_power(signal: &Array1<f64>) -> SignalResult<f64> {
    // Compute mean
    let mean = signal.mean().unwrap_or(0.0);

    // Compute variance (signal power)
    let power = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;

    Ok(power)
}

/// Helper function to pad a signal for boundary handling
fn pad_signal(signal: &Array1<f64>, pad_size: usize) -> Array1<f64> {
    let n = signal.len();
    let mut padded = Array1::zeros(n + 2 * pad_size);

    // Copy original signal
    for i in 0..n {
        padded[i + pad_size] = signal[i];
    }

    // Reflect boundaries
    for i in 0..pad_size {
        // Left boundary
        padded[pad_size - 1 - i] = signal[i.min(n - 1)];

        // Right boundary
        padded[n + pad_size + i] = signal[n - 1 - i.min(n - 1)];
    }

    padded
}

/// Helper function to smooth a power spectral density estimate
fn smooth_psd(psd: &Array1<f64>) -> Array1<f64> {
    let n = psd.len();
    let mut smoothed = Array1::zeros(n);

    // Apply simple moving average smoothing
    let window_size = (n as f64 * 0.02).clamp(3.0, 15.0) as usize;
    let half_window = window_size / 2;

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        let sum = psd.slice(s![start..end]).sum();
        let count = end - start;

        smoothed[i] = sum / count as f64;
    }

    smoothed
}
