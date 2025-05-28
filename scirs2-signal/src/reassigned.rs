// Reassigned Spectrogram Implementation
//
// This module provides implementations of reassigned spectrograms, which improve the
// time-frequency localization of the Short-Time Fourier Transform (STFT) by mapping the
// standard spectrogram values to time-frequency coordinates that more accurately represent
// the local signal structure.
//
// References:
// - Auger, F., & Flandrin, P. (1995). Improving the readability of time-frequency and time-scale
//   representations by the reassignment method. IEEE Transactions on Signal Processing, 43(5), 1068-1089.
// - Fulop, S. A., & Fitz, K. (2006). Algorithms for computing the time-corrected instantaneous
//   frequency (reassigned) spectrogram, with applications. The Journal of the Acoustical Society
//   of America, 119(1), 360-371.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::SignalResult;
use crate::spectral;
use crate::window;

/// Configuration parameters for reassigned spectrogram computation
#[derive(Debug, Clone)]
pub struct ReassignedConfig {
    /// Window function for STFT
    pub window: Array1<f64>,

    /// Window function for time derivative (d/dt)
    pub time_window: Option<Array1<f64>>,

    /// Window function for frequency derivative (d/dω)
    pub freq_window: Option<Array1<f64>>,

    /// Hop size (frame shift) in samples
    pub hop_size: usize,

    /// FFT size
    pub n_fft: Option<usize>,

    /// Sample rate of the signal
    pub fs: f64,

    /// Whether to return the ordinary spectrogram too
    pub return_spectrogram: bool,
}

impl Default for ReassignedConfig {
    fn default() -> Self {
        let window_size = 256;
        let win = window::hann(window_size, true).expect("Failed to create window");

        ReassignedConfig {
            window: Array1::from(win),
            time_window: None,
            freq_window: None,
            hop_size: window_size / 4,
            n_fft: None,
            fs: 1.0,
            return_spectrogram: false,
        }
    }
}

/// Result structure for reassigned spectrogram
#[derive(Debug)]
pub struct ReassignedResult {
    /// Reassigned spectrogram
    pub reassigned: Array2<f64>,

    /// Original spectrogram (only if requested)
    pub spectrogram: Option<Array2<f64>>,

    /// Time instants (in seconds)
    pub times: Array1<f64>,

    /// Frequency bins (in Hz)
    pub frequencies: Array1<f64>,

    /// Time shifts for each point (for visualization)
    pub time_shifts: Option<Array2<f64>>,

    /// Frequency shifts for each point (for visualization)
    pub freq_shifts: Option<Array2<f64>>,
}

/// Computes the reassigned spectrogram of a signal
///
/// The reassigned spectrogram provides improved time-frequency localization
/// compared to the traditional spectrogram by computing the center of gravity
/// of the energy distribution around each time-frequency point.
///
/// # Arguments
///
/// * `signal` - The input signal (real-valued)
/// * `config` - Configuration parameters for the computation
///
/// # Returns
///
/// A `ReassignedResult` structure containing the reassigned spectrogram and metadata
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_signal::reassigned::{reassigned_spectrogram, ReassignedConfig};
/// use scirs2_signal::window;
///
/// // Create a chirp signal
/// let fs = 1000.0;
/// let t = Array1::linspace(0.0, 1.0, 1000);
/// let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * (10.0 * ti + 50.0 * ti * ti)).sin());
///
/// // Configure the reassigned spectrogram
/// let mut config = ReassignedConfig::default();
/// config.window = Array1::from(window::hann(256, true).unwrap());
/// config.hop_size = 64;
/// config.fs = fs;
/// config.return_spectrogram = true;
///
/// // Compute the reassigned spectrogram
/// let result = reassigned_spectrogram(&signal, config).unwrap();
///
/// // result.reassigned contains the reassigned spectrogram
/// // result.spectrogram contains the original spectrogram (if requested)
/// ```
pub fn reassigned_spectrogram(
    signal: &Array1<f64>,
    mut config: ReassignedConfig,
) -> SignalResult<ReassignedResult> {
    // Ensure FFT size is at least window length
    let n_fft = config
        .n_fft
        .unwrap_or(next_power_of_two(config.window.len()));
    config.n_fft = Some(n_fft);

    // Derive windows for partial derivatives if not provided
    let win = &config.window;

    // Time derivative window: Recommended way is to use the time-ramped window
    let time_win = match &config.time_window {
        Some(tw) => tw.clone(),
        None => {
            let mut tw = Array1::zeros(win.len());
            let center = (win.len() - 1) as f64 / 2.0;
            for i in 0..win.len() {
                tw[i] = win[i] * (i as f64 - center);
            }
            tw
        }
    };

    // Frequency derivative window: Recommended way is to use the derivative of the window
    let freq_win = match &config.freq_window {
        Some(fw) => fw.clone(),
        None => {
            let mut fw = Array1::zeros(win.len());
            for i in 1..win.len() - 1 {
                fw[i] = (win[i + 1] - win[i - 1]) / 2.0;
            }
            fw[0] = win[1] - win[0];
            fw[win.len() - 1] = win[win.len() - 1] - win[win.len() - 2];
            fw
        }
    };

    // Compute three different STFTs
    // 1. Standard STFT with the provided window
    let stft = compute_stft(signal, win, config.hop_size, n_fft)?;

    // 2. STFT with the time derivative window
    let stft_time = compute_stft(signal, &time_win, config.hop_size, n_fft)?;

    // 3. STFT with the frequency derivative window
    let stft_freq = compute_stft(signal, &freq_win, config.hop_size, n_fft)?;

    // Compute reassignment operators
    let (time_shifts, freq_shifts) =
        compute_reassignment_operators(&stft, &stft_time, &stft_freq, config.fs)?;

    // Compute reassigned spectrogram
    let reassigned = apply_reassignment(&stft, &time_shifts, &freq_shifts)?;

    // Create time and frequency axes
    let n_frames = stft.shape()[1];
    let hop_seconds = config.hop_size as f64 / config.fs;
    let times = Array1::linspace(0.0, (n_frames as f64 - 1.0) * hop_seconds, n_frames);

    let n_freqs = reassigned.shape()[0];
    let frequencies = Array1::linspace(0.0, config.fs / 2.0, n_freqs);

    // Convert spectrogram to magnitude or power if requested
    let spectrogram = if config.return_spectrogram {
        Some(stft.mapv(|z| z.norm_sqr())) // Power spectrogram
    } else {
        None
    };

    Ok(ReassignedResult {
        reassigned,
        spectrogram,
        times,
        frequencies,
        time_shifts: None, // Optional, can be included if desired
        freq_shifts: None, // Optional, can be included if desired
    })
}

/// Compute a basic STFT without applying phase corrections
fn compute_stft(
    signal: &Array1<f64>,
    window: &Array1<f64>,
    hop_size: usize,
    n_fft: usize,
) -> SignalResult<Array2<Complex64>> {
    let stft_result = spectral::stft(
        signal.as_slice().unwrap(),
        None,               // fs
        None,               // window
        Some(window.len()), // nperseg
        Some(hop_size),     // noverlap
        Some(n_fft),        // nfft
        None,               // detrend
        None,               // boundary
        None,               // padded
    )?;

    // Extract the result
    let (_f, _t, zxx) = stft_result;
    let n_frames = zxx.len();
    let n_bins = n_fft / 2 + 1; // Only positive frequencies

    let mut stft = Array2::zeros((n_bins, n_frames));
    for j in 0..n_frames {
        for i in 0..zxx[j].len().min(n_bins) {
            stft[[i, j]] = zxx[j][i];
        }
    }

    Ok(stft)
}

/// Compute time and frequency reassignment operators
fn compute_reassignment_operators(
    stft: &Array2<Complex64>,
    stft_time: &Array2<Complex64>,
    stft_freq: &Array2<Complex64>,
    fs: f64,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_bins, n_frames) = (stft.shape()[0], stft.shape()[1]);

    let mut time_shifts = Array2::zeros((n_bins, n_frames));
    let mut freq_shifts = Array2::zeros((n_bins, n_frames));

    // Threshold for avoiding division by very small numbers
    let threshold = 1e-6;

    for i in 0..n_bins {
        let omega = 2.0 * PI * i as f64 / (2.0 * (n_bins - 1) as f64);

        for j in 0..n_frames {
            let stft_val = stft[[i, j]];
            let magnitude = stft_val.norm();

            if magnitude > threshold {
                // Time reassignment: instantaneous frequency correction
                let inst_phase_derivative = (stft_time[[i, j]] / stft_val).im;
                freq_shifts[[i, j]] = omega - inst_phase_derivative;

                // Frequency reassignment: group delay correction
                let group_delay = (stft_freq[[i, j]] / stft_val).im;
                time_shifts[[i, j]] = j as f64 - group_delay;
            } else {
                // No reassignment for very small magnitudes
                freq_shifts[[i, j]] = omega;
                time_shifts[[i, j]] = j as f64;
            }
        }
    }

    // Convert frequency shifts from normalized to Hz
    for i in 0..n_bins {
        for j in 0..n_frames {
            freq_shifts[[i, j]] = freq_shifts[[i, j]] * fs / (2.0 * PI);
        }
    }

    Ok((time_shifts, freq_shifts))
}

/// Apply reassignment to the STFT
fn apply_reassignment(
    stft: &Array2<Complex64>,
    time_shifts: &Array2<f64>,
    freq_shifts: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    let (n_bins, n_frames) = (stft.shape()[0], stft.shape()[1]);

    // Initialize reassigned spectrogram
    let mut reassigned = Array2::zeros((n_bins, n_frames));

    // Apply reassignment
    for i in 0..n_bins {
        for j in 0..n_frames {
            let magnitude = stft[[i, j]].norm_sqr();

            // Skip very small magnitudes
            if magnitude < 1e-10 {
                continue;
            }

            // Get reassigned coordinates
            let t_reassigned = time_shifts[[i, j]].round() as isize;
            let f_reassigned = freq_shifts[[i, j]] / ((n_bins - 1) as f64) * (n_bins as f64);
            let f_bin = f_reassigned.round() as isize;

            // Check if reassigned coordinates are within bounds
            if t_reassigned >= 0
                && t_reassigned < n_frames as isize
                && f_bin >= 0
                && f_bin < n_bins as isize
            {
                reassigned[[f_bin as usize, t_reassigned as usize]] += magnitude;
            }
        }
    }

    Ok(reassigned)
}

/// Find the next power of two greater than or equal to n
fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Computes a smoothed reassigned spectrogram with less data scattering
///
/// This version applies additional smoothing to reduce the "salt and pepper"
/// scattering of energy in the reassigned spectrogram.
///
/// # Arguments
///
/// * `signal` - The input signal (real-valued)
/// * `config` - Configuration parameters for the computation
/// * `smoothing_width` - Width of the smoothing (in bins) for both time and frequency
///
/// # Returns
///
/// A `ReassignedResult` structure containing the smoothed reassigned spectrogram and metadata
pub fn smoothed_reassigned_spectrogram(
    signal: &Array1<f64>,
    config: ReassignedConfig,
    smoothing_width: usize,
) -> SignalResult<ReassignedResult> {
    // Compute standard reassigned spectrogram
    let mut result = reassigned_spectrogram(signal, config)?;

    // Apply smoothing
    let smoothed = smooth_spectrogram(&result.reassigned, smoothing_width);

    // Replace the spectrogram with the smoothed version
    result.reassigned = smoothed;

    Ok(result)
}

/// Apply a simple 2D box smoothing filter to the spectrogram
fn smooth_spectrogram(spectrogram: &Array2<f64>, width: usize) -> Array2<f64> {
    let (n_bins, n_frames) = (spectrogram.shape()[0], spectrogram.shape()[1]);
    let mut smoothed = Array2::zeros((n_bins, n_frames));

    // Half width of the filter
    let half_width = width / 2;

    // Apply 2D boxcar smoothing
    for i in 0..n_bins {
        for j in 0..n_frames {
            let mut sum = 0.0;
            let mut count = 0;

            // Determine boundaries for the filter
            let f_start = i.saturating_sub(half_width);
            let f_end = if i + half_width < n_bins {
                i + half_width
            } else {
                n_bins - 1
            };

            let t_start = j.saturating_sub(half_width);
            let t_end = if j + half_width < n_frames {
                j + half_width
            } else {
                n_frames - 1
            };

            // Apply the filter
            for fi in f_start..=f_end {
                for tj in t_start..=t_end {
                    sum += spectrogram[[fi, tj]];
                    count += 1;
                }
            }

            // Store the average
            if count > 0 {
                smoothed[[i, j]] = sum / count as f64;
            }
        }
    }

    smoothed
}

/// Extracts ridges (instantaneous frequencies) from a reassigned spectrogram
///
/// # Arguments
///
/// * `spectrogram` - The reassigned spectrogram
/// * `frequencies` - The frequency axis
/// * `max_ridges` - Maximum number of ridges to extract
/// * `min_intensity` - Minimum intensity threshold (relative to maximum) for ridge points
///
/// # Returns
///
/// A vector of ridges, where each ridge is a vector of (time_index, frequency) pairs
pub fn extract_ridges(
    spectrogram: &Array2<f64>,
    frequencies: &Array1<f64>,
    max_ridges: usize,
    min_intensity: f64,
) -> Vec<Vec<(usize, f64)>> {
    let n_freqs = spectrogram.shape()[0];
    let n_times = spectrogram.shape()[1];

    let max_ridges = max_ridges.max(1);
    let mut ridges = Vec::with_capacity(max_ridges);

    // For each time point, find the frequencies with the highest energy
    for t in 0..n_times {
        let mut peak_indices = Vec::with_capacity(max_ridges);
        let mut peak_magnitudes = Vec::with_capacity(max_ridges);

        // Get the time slice
        let time_slice = spectrogram.slice(s![.., t]);

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
        let avg_energy_a = a.iter().map(|(t, _)| *t).sum::<usize>() as f64 / a.len() as f64;
        let avg_energy_b = b.iter().map(|(t, _)| *t).sum::<usize>() as f64 / b.len() as f64;
        avg_energy_b
            .partial_cmp(&avg_energy_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ridges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(256), 256);
        assert_eq!(next_power_of_two(257), 512);
    }

    #[test]
    fn test_reassigned_spectrogram_chirp() {
        // Create a chirp signal
        let n = 1024;
        let fs = 1024.0;
        let duration = n as f64 / fs;

        let t = Array1::linspace(0.0, duration, n);
        let signal = t.mapv(|ti| (2.0 * PI * (50.0 * ti + 200.0 * ti * ti)).sin());

        // Configure the reassigned spectrogram
        let config = ReassignedConfig {
            window: Array1::from(window::hann(128, true).unwrap()),
            hop_size: 32,
            fs,
            return_spectrogram: true,
            ..Default::default()
        };

        // Compute the reassigned spectrogram
        let result = reassigned_spectrogram(&signal, config).unwrap();

        // Basic size checks
        assert_eq!(result.reassigned.shape()[1], result.times.len());
        assert_eq!(result.reassigned.shape()[0], result.frequencies.len());

        // Original spectrogram should be returned
        assert!(result.spectrogram.is_some());

        // Check that the reassigned spectrogram has energy
        let mut has_energy = false;
        for f in 0..result.reassigned.shape()[0] {
            for t in 0..result.reassigned.shape()[1] {
                if result.reassigned[[f, t]] > 0.1 {
                    has_energy = true;
                    break;
                }
            }
            if has_energy {
                break;
            }
        }
        assert!(has_energy);
    }

    #[test]
    fn test_smoothed_reassigned_spectrogram() {
        // Create a test signal
        let n = 512;
        let fs = 512.0;
        let t = Array1::linspace(0.0, 1.0, n);
        let signal = t.mapv(|ti| (2.0 * PI * 50.0 * ti).sin());

        // Configure the reassigned spectrogram
        let config = ReassignedConfig {
            window: Array1::from(window::hann(64, true).unwrap()),
            hop_size: 16,
            fs,
            ..Default::default()
        };

        // Compute both standard and smoothed reassigned spectrograms
        let standard = reassigned_spectrogram(&signal, config.clone()).unwrap();
        let smoothed = smoothed_reassigned_spectrogram(&signal, config, 3).unwrap();

        // Both should have the same dimensions
        assert_eq!(standard.reassigned.shape(), smoothed.reassigned.shape());

        // Check that both spectrograms have energy
        let mut standard_has_energy = false;
        let mut smoothed_has_energy = false;

        for f in 0..standard.reassigned.shape()[0] {
            for t in 0..standard.reassigned.shape()[1] {
                if standard.reassigned[[f, t]] > 0.1 {
                    standard_has_energy = true;
                }
                if smoothed.reassigned[[f, t]] > 0.1 {
                    smoothed_has_energy = true;
                }
            }
        }

        assert!(standard_has_energy);
        assert!(smoothed_has_energy);

        // The smoothed version should generally have less noise
        // Calculate total energy outside the expected frequency band (45-55 Hz)
        let freq_bin_45hz = (45.0 / (fs / 2.0) * standard.frequencies.len() as f64) as usize;
        let freq_bin_55hz = (55.0 / (fs / 2.0) * standard.frequencies.len() as f64) as usize;

        let mut standard_noise = 0.0;
        let mut smoothed_noise = 0.0;

        for f in 0..standard.reassigned.shape()[0] {
            if f < freq_bin_45hz || f > freq_bin_55hz {
                for t in 0..standard.reassigned.shape()[1] {
                    standard_noise += standard.reassigned[[f, t]];
                    smoothed_noise += smoothed.reassigned[[f, t]];
                }
            }
        }

        // Smoothed should have less out-of-band energy
        assert!(smoothed_noise <= standard_noise * 1.1); // Allow some tolerance
    }
}
