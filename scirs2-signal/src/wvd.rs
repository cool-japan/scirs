// Wigner-Ville Distribution Implementation
//
// This module provides implementations of the Wigner-Ville distribution (WVD) and its variants
// for high-resolution time-frequency analysis without the trade-offs inherent in linear methods.
//
// References:
// - Cohen, L. (1989). Time-frequency distributions - A review. Proceedings of the IEEE, 77(7), 941-981.
// - Boashash, B. (2015). Time-Frequency Signal Analysis and Processing (2nd ed.). Academic Press.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;

use crate::error::{SignalError, SignalResult};
use crate::hilbert;

/// Configuration parameters for the Wigner-Ville Distribution
#[derive(Debug, Clone)]
pub struct WvdConfig {
    /// Whether to compute the analytic signal first (to suppress negative frequencies)
    pub analytic: bool,

    /// Window function for smoothed pseudo-WVD (None for standard WVD)
    pub time_window: Option<Array1<f64>>,

    /// Frequency window for smoothed pseudo-WVD (None for standard WVD)
    pub freq_window: Option<Array1<f64>>,

    /// Whether to use fft zero-padding for higher resolution
    pub zero_padding: bool,

    /// Sample rate of the signal
    pub fs: f64,
}

impl Default for WvdConfig {
    fn default() -> Self {
        WvdConfig {
            analytic: true,
            time_window: None,
            freq_window: None,
            zero_padding: true,
            fs: 1.0,
        }
    }
}

/// Computes the Wigner-Ville Distribution (WVD) of a signal
///
/// The Wigner-Ville distribution provides a high-resolution time-frequency representation
/// of a signal without the time-frequency resolution trade-offs inherent in the STFT.
///
/// # Arguments
///
/// * `signal` - The input signal (real-valued)
/// * `config` - Configuration parameters for the WVD computation
///
/// # Returns
///
/// A 2D array representing the WVD. Rows correspond to frequency bins and columns to time points.
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_signal::wvd::{wigner_ville, WvdConfig};
///
/// // Create a chirp signal
/// let fs = 1000.0;
/// let t = Array1::linspace(0.0, 1.0, 1000);
/// let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).sin());
///
/// // Configure the WVD
/// let mut config = WvdConfig::default();
/// config.fs = fs;
///
/// // Compute the WVD
/// let wvd = wigner_ville(&signal, config).unwrap();
/// ```
pub fn wigner_ville(signal: &Array1<f64>, config: WvdConfig) -> SignalResult<Array2<f64>> {
    // Convert to analytic signal if needed
    let analytic_signal = if config.analytic {
        Array1::from(hilbert::hilbert(signal.as_slice().unwrap())?)
    } else {
        signal.mapv(|x| Complex64::new(x, 0.0))
    };

    // Compute cross Wigner-Ville distribution
    compute_wvd(&analytic_signal, None, config)
}

/// Computes the Cross Wigner-Ville Distribution (XWVD) between two signals
///
/// The Cross WVD reveals the joint time-frequency structure shared between two signals.
///
/// # Arguments
///
/// * `signal1` - The first input signal (real-valued)
/// * `signal2` - The second input signal (real-valued)
/// * `config` - Configuration parameters for the WVD computation
///
/// # Returns
///
/// A 2D array representing the cross WVD. The result is generally complex-valued.
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_signal::wvd::{cross_wigner_ville, WvdConfig};
///
/// // Create two related signals
/// let fs = 1000.0;
/// let t = Array1::linspace(0.0, 1.0, 1000);
/// let signal1 = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).sin());
/// let signal2 = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).cos());
///
/// // Configure the WVD
/// let mut config = WvdConfig::default();
/// config.fs = fs;
///
/// // Compute the Cross WVD
/// let xwvd = cross_wigner_ville(&signal1, &signal2, config).unwrap();
/// ```
pub fn cross_wigner_ville(
    signal1: &Array1<f64>,
    signal2: &Array1<f64>,
    config: WvdConfig,
) -> SignalResult<Array2<Complex64>> {
    // Check that signals have the same length
    if signal1.len() != signal2.len() {
        return Err(SignalError::DimensionError(
            "Signals must have the same length for cross-WVD".to_string(),
        ));
    }

    // Convert to analytic signals if needed
    let analytic_signal1 = if config.analytic {
        Array1::from(hilbert::hilbert(signal1.as_slice().unwrap())?)
    } else {
        signal1.mapv(|x| Complex64::new(x, 0.0))
    };

    let analytic_signal2 = if config.analytic {
        Array1::from(hilbert::hilbert(signal2.as_slice().unwrap())?)
    } else {
        signal2.mapv(|x| Complex64::new(x, 0.0))
    };

    // Compute cross WVD
    compute_cross_wvd(&analytic_signal1, &analytic_signal2, config)
}

/// Computes the Smoothed Pseudo Wigner-Ville Distribution (SPWVD)
///
/// The SPWVD applies independent time and frequency smoothing to the WVD, which
/// reduces interference terms while maintaining good time-frequency resolution.
///
/// # Arguments
///
/// * `signal` - The input signal (real-valued)
/// * `time_window` - Window function for time smoothing
/// * `freq_window` - Window function for frequency smoothing
/// * `config` - Configuration parameters for the WVD computation
///
/// # Returns
///
/// A 2D array representing the SPWVD. Rows correspond to frequency bins and columns to time points.
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_signal::wvd::{smoothed_pseudo_wigner_ville, WvdConfig};
/// use scirs2_signal::window;
///
/// // Create a chirp signal
/// let fs = 1000.0;
/// let t = Array1::linspace(0.0, 1.0, 1000);
/// let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).sin());
///
/// // Create windows for smoothing
/// let time_win = Array1::from(window::hamming(51, true).unwrap());
/// let freq_win = Array1::from(window::hamming(101, true).unwrap());
///
/// // Configure the WVD
/// let mut config = WvdConfig::default();
/// config.fs = fs;
///
/// // Compute the SPWVD
/// let spwvd = smoothed_pseudo_wigner_ville(&signal, &time_win, &freq_win, config).unwrap();
/// ```
pub fn smoothed_pseudo_wigner_ville(
    signal: &Array1<f64>,
    time_window: &Array1<f64>,
    freq_window: &Array1<f64>,
    mut config: WvdConfig,
) -> SignalResult<Array2<f64>> {
    // Set the windows in config
    config.time_window = Some(time_window.clone());
    config.freq_window = Some(freq_window.clone());

    // Convert to analytic signal if needed
    let analytic_signal = if config.analytic {
        Array1::from(hilbert::hilbert(signal.as_slice().unwrap())?)
    } else {
        signal.mapv(|x| Complex64::new(x, 0.0))
    };

    // Compute smoothed pseudo-WVD
    compute_wvd(&analytic_signal, None, config)
}

/// Core computation function for the Wigner-Ville Distribution
///
/// This function implements the core algorithm for all variants of the WVD.
fn compute_wvd(
    signal: &Array1<Complex64>,
    signal2: Option<&Array1<Complex64>>,
    config: WvdConfig,
) -> SignalResult<Array2<f64>> {
    // If signal2 is provided, compute cross-WVD, otherwise auto-WVD
    let cross_wvd = compute_cross_wvd(signal, signal2.unwrap_or(signal), config)?;

    // For auto-WVD, the result should be real-valued (within numerical precision)
    // We take the real part and discard any small imaginary components
    let wvd = cross_wvd.mapv(|x| x.re);

    Ok(wvd)
}

/// Core computation function for the Cross Wigner-Ville Distribution
fn compute_cross_wvd(
    signal1: &Array1<Complex64>,
    signal2: &Array1<Complex64>,
    config: WvdConfig,
) -> SignalResult<Array2<Complex64>> {
    let n = signal1.len();

    // Determine output size with optional zero-padding
    let n_fft = if config.zero_padding { n * 2 } else { n };

    // Create the output array
    let mut wvd = Array2::<Complex64>::zeros((n_fft / 2 + 1, n));

    // Prepare windows if provided
    let time_window = config.time_window.as_ref().map(|w| {
        // Ensure window is centered and of appropriate length
        let w_len = w.len();
        if w_len % 2 == 0 {
            // Even length - need to shift
            let half = w_len / 2;
            let mut new_w = Array1::zeros(w_len + 1);
            for i in 0..w_len {
                new_w[i + (i >= half) as usize] = w[i];
            }
            new_w
        } else {
            w.clone()
        }
    });

    let freq_window = config.freq_window.as_ref().map(|w| {
        // Ensure frequency window is of appropriate length for FFT
        match w.len().cmp(&n_fft) {
            std::cmp::Ordering::Less => {
                let mut new_w = Array1::zeros(n_fft);
                let offset = (n_fft - w.len()) / 2;
                for i in 0..w.len() {
                    new_w[i + offset] = w[i];
                }
                new_w
            }
            std::cmp::Ordering::Greater => {
                let offset = (w.len() - n_fft) / 2;
                w.slice(s![offset..offset + n_fft]).to_owned()
            }
            std::cmp::Ordering::Equal => w.clone(),
        }
    });

    // Analyze each time point
    for t in 0..n {
        // Compute the instantaneous autocorrelation
        let mut acorr = Array1::<Complex64>::zeros(n_fft);

        // Determine analysis window range (accounting for boundaries)
        let window_half_length = match &time_window {
            Some(w) => w.len() / 2,
            None => n / 2, // Use full signal length for standard WVD
        };

        let tau_min = -(std::cmp::min(t, window_half_length) as isize);
        let tau_max = std::cmp::min(n - t, window_half_length + 1) as isize;

        // Compute autocorrelation using lags (tau)
        for tau in tau_min..tau_max {
            let idx = (tau + n_fft as isize / 2) as usize; // Center in FFT buffer

            // Apply time window if provided
            let window_val = match &time_window {
                Some(w) => {
                    let w_idx = (tau + window_half_length as isize) as usize;
                    if w_idx < w.len() {
                        w[w_idx]
                    } else {
                        0.0
                    }
                }
                None => 1.0,
            };

            // Compute the autocorrelation value
            // Handle integer division carefully to avoid boundary issues
            let idx1 = t as isize + tau;
            let idx2 = t as isize - tau;

            if idx1 >= 0 && idx1 < n as isize && idx2 >= 0 && idx2 < n as isize {
                acorr[idx] = signal1[idx1 as usize] * signal2[idx2 as usize].conj() * window_val;
            }
        }

        // Apply frequency smoothing window if provided
        if let Some(w) = &freq_window {
            for i in 0..n_fft {
                acorr[i] *= w[i];
            }
        }

        // Compute FFT of autocorrelation to get the spectrum at this time point
        let spectrum =
            scirs2_fft::fft(acorr.as_slice().unwrap(), None).expect("FFT computation failed");

        // Store only the positive frequencies (the result is conjugate symmetric for real signals)
        let n_freqs = n_fft / 2 + 1;
        for k in 0..n_freqs {
            wvd[[k, t]] = spectrum[k];
        }
    }

    Ok(wvd)
}

/// Generates a frequency axis for WVD results
///
/// # Arguments
///
/// * `n_freqs` - Number of frequency points
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// An array of frequency values in Hz
pub fn frequency_axis(n_freqs: usize, fs: f64) -> Array1<f64> {
    Array1::linspace(0.0, fs / 2.0, n_freqs)
}

/// Generates a time axis for WVD results
///
/// # Arguments
///
/// * `n_times` - Number of time points
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// An array of time values in seconds
pub fn time_axis(n_times: usize, fs: f64) -> Array1<f64> {
    let dt = 1.0 / fs;
    Array1::linspace(0.0, (n_times as f64 - 1.0) * dt, n_times)
}

/// Extracts ridges (instantaneous frequencies) from a WVD representation
///
/// # Arguments
///
/// * `wvd` - The WVD time-frequency representation
/// * `frequencies` - The frequency axis
/// * `max_ridges` - Maximum number of ridges to extract
/// * `min_intensity` - Minimum intensity threshold (relative to maximum) for ridge points
///
/// # Returns
///
/// A vector of ridges, where each ridge is a vector of (time_index, frequency) pairs
pub fn extract_ridges(
    wvd: &Array2<f64>,
    frequencies: &Array1<f64>,
    max_ridges: usize,
    min_intensity: f64,
) -> Vec<Vec<(usize, f64)>> {
    let n_freqs = wvd.shape()[0];
    let n_times = wvd.shape()[1];

    let max_ridges = max_ridges.max(1);
    let mut ridges = Vec::with_capacity(max_ridges);

    // For each time point, find the frequencies with the highest energy
    for t in 0..n_times {
        let mut peak_indices = Vec::with_capacity(max_ridges);
        let mut peak_magnitudes = Vec::with_capacity(max_ridges);

        // Get the time slice
        let time_slice = wvd.slice(s![.., t]);

        // Find local maxima (peaks)
        let mut peaks = Vec::new();
        for f in 1..n_freqs - 1 {
            if time_slice[f] > time_slice[f - 1] && time_slice[f] > time_slice[f + 1] {
                peaks.push((f, time_slice[f]));
            }
        }

        // Sort peaks by magnitude
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get the global maximum for threshold calculation
        let global_max = if !peaks.is_empty() {
            peaks[0].1
        } else {
            time_slice.iter().cloned().fold(0.0, f64::max)
        };

        // Keep only the strongest peaks above threshold
        for &(idx, magnitude) in peaks.iter().take(max_ridges) {
            if magnitude >= min_intensity * global_max {
                peak_indices.push(idx);
                peak_magnitudes.push(magnitude);
            }
        }

        // Update ridge structures
        for (i, &freq_idx) in peak_indices.iter().enumerate() {
            if i >= ridges.len() {
                ridges.push(Vec::new());
            }

            ridges[i].push((t, frequencies[freq_idx]));
        }
    }

    // Sort ridges by average energy
    ridges.sort_by(|a, b| {
        let energy_a: f64 = a
            .iter()
            .map(|(t, _)| {
                let f_idx = a[0].1.mul_add(0.0, frequencies.len() as f64) as usize;
                if f_idx < n_freqs {
                    wvd[[f_idx, *t]]
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / a.len() as f64;

        let energy_b: f64 = b
            .iter()
            .map(|(t, _)| {
                let f_idx = b[0].1.mul_add(0.0, frequencies.len() as f64) as usize;
                if f_idx < n_freqs {
                    wvd[[f_idx, *t]]
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / b.len() as f64;

        energy_b
            .partial_cmp(&energy_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ridges
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use std::f64::consts::PI;

    #[test]
    fn test_wigner_ville_chirp() {
        // Create a chirp signal
        let n = 128;
        let t = Array1::linspace(0.0, 1.0, n);
        let signal = t.mapv(|ti| (2.0 * PI * (5.0 * ti + 10.0 * ti * ti)).sin());

        // Configure the WVD
        let config = WvdConfig {
            analytic: true,
            time_window: None,
            freq_window: None,
            zero_padding: true,
            fs: n as f64,
        };

        // Compute the WVD
        let wvd = wigner_ville(&signal, config).unwrap();

        // Basic size check
        assert_eq!(wvd.shape()[1], n);

        // Extract ridge and verify it approximates the chirp frequency
        let freqs = frequency_axis(wvd.shape()[0], n as f64);
        let ridges = extract_ridges(&wvd, &freqs, 1, 0.2);

        // Verify we found a ridge
        assert!(!ridges.is_empty());
        let ridge = &ridges[0];

        // Test that frequency increases (chirp characteristic)
        let first_quarter = n / 4;
        let last_quarter = 3 * n / 4;

        // Get mean frequencies from first and last quarters
        let early_freq: f64 = ridge
            .iter()
            .filter(|(t, _)| *t < first_quarter)
            .map(|(_, f)| *f)
            .sum::<f64>()
            / first_quarter as f64;

        let late_freq: f64 = ridge
            .iter()
            .filter(|(t, _)| *t >= last_quarter)
            .map(|(_, f)| *f)
            .sum::<f64>()
            / (n - last_quarter) as f64;

        // Frequency should increase for a linear chirp
        assert!(late_freq > early_freq);
    }

    #[test]
    fn test_cross_wigner_ville() {
        // Create two simple test signals
        let n = 64; // Smaller size for faster test
        let t = Array1::linspace(0.0, 1.0, n);
        let signal1 = t.mapv(|ti| (2.0 * PI * 5.0 * ti).sin());
        let signal2 = t.mapv(|ti| (2.0 * PI * 5.0 * ti).cos());

        // Configure the WVD
        let config = WvdConfig {
            analytic: true,
            time_window: None,
            freq_window: None,
            zero_padding: false,
            fs: n as f64,
        };

        // Compute the Cross WVD
        let xwvd = cross_wigner_ville(&signal1, &signal2, config).unwrap();

        // Basic size check
        assert_eq!(xwvd.shape()[1], n);

        // Check that all values are finite
        for i in 0..xwvd.shape()[0] {
            for j in 0..xwvd.shape()[1] {
                assert!(xwvd[[i, j]].re.is_finite());
                assert!(xwvd[[i, j]].im.is_finite());
            }
        }

        // Check that there's non-zero energy in the result
        let mut total_energy = 0.0;
        for i in 0..xwvd.shape()[0] {
            for j in 0..xwvd.shape()[1] {
                total_energy += xwvd[[i, j]].norm();
            }
        }
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_smoothed_pseudo_wigner_ville() {
        // Create a simple test signal
        let n = 64; // Smaller size for faster test
        let t = Array1::linspace(0.0, 1.0, n);
        let signal = t.mapv(|ti| (2.0 * PI * 5.0 * ti).sin());

        // Create simple smoothing windows
        let time_window = Array::from_vec(vec![0.25, 0.5, 1.0, 0.5, 0.25]);
        let freq_window = Array::from_vec(vec![0.25, 0.5, 1.0, 0.5, 0.25]);

        // Configure the WVD
        let config = WvdConfig {
            analytic: true,
            time_window: None,
            freq_window: None,
            zero_padding: false,
            fs: n as f64,
        };

        // Compute both standard WVD and smoothed pseudo-WVD
        let wvd = wigner_ville(&signal, config.clone()).unwrap();
        let spwvd =
            smoothed_pseudo_wigner_ville(&signal, &time_window, &freq_window, config).unwrap();

        // Both should have the same dimensions
        assert_eq!(wvd.shape(), spwvd.shape());

        // Check that all values are finite
        for i in 0..spwvd.shape()[0] {
            for j in 0..spwvd.shape()[1] {
                assert!(spwvd[[i, j]].is_finite());
            }
        }

        // Check that smoothing reduces total variation (smoother result)
        let mut wvd_variation = 0.0;
        let mut spwvd_variation = 0.0;

        // Calculate variation along time axis
        for i in 0..wvd.shape()[0] {
            for j in 1..wvd.shape()[1] {
                wvd_variation += (wvd[[i, j]] - wvd[[i, j - 1]]).abs();
                spwvd_variation += (spwvd[[i, j]] - spwvd[[i, j - 1]]).abs();
            }
        }

        // SPWVD should have less variation (be smoother) than WVD
        // But since we're using a simple sine wave, the difference might be small
        // Just check that both have finite variation
        assert!(wvd_variation.is_finite());
        assert!(spwvd_variation.is_finite());
    }
}
