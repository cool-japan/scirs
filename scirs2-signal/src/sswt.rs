// Synchrosqueezed Wavelet Transform Implementation
//
// This module provides an implementation of the synchrosqueezed wavelet transform (SSWT),
// which is an enhanced time-frequency analysis technique that provides sharper frequency
// localization than the standard continuous wavelet transform.
//
// Reference: Daubechies, I., Lu, J., & Wu, H. T. (2011). Synchrosqueezed wavelet transforms:
// An empirical mode decomposition-like tool. Applied and computational harmonic analysis, 30(2), 243-261.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use num_traits::Float;
use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};
use crate::wavelets;

/// Configuration parameters for the synchrosqueezed wavelet transform
#[derive(Debug, Clone)]
pub struct SynchroCwtConfig {
    /// Gamma threshold for excluding CWT coefficients below this magnitude
    pub gamma: f64,

    /// Frequency bins for the synchrosqueezed transform
    pub frequencies: Array1<f64>,

    /// Whether to return the CWT coefficients in addition to the SSWT
    pub return_cwt: bool,
}

impl Default for SynchroCwtConfig {
    fn default() -> Self {
        SynchroCwtConfig {
            gamma: 1e-8,
            frequencies: Array1::linspace(1.0, 128.0, 128),
            return_cwt: false,
        }
    }
}

/// Result type for synchrosqueezed wavelet transform operations
#[derive(Debug)]
pub struct SynchroCwtResult {
    /// Synchrosqueezed transform coefficients [frequency, time]
    pub sst: Array2<Complex64>,

    /// Continuous wavelet transform coefficients [scale, time]
    pub cwt: Option<Array2<Complex64>>,

    /// Instantaneous frequencies derived from the CWT phase [scale, time]
    pub omega: Option<Array2<f64>>,

    /// Scales used for the CWT
    pub scales: Array1<f64>,

    /// Frequencies used for the synchrosqueezed transform
    pub frequencies: Array1<f64>,
}

/// Compute the synchrosqueezed wavelet transform of a real-valued signal
///
/// # Arguments
///
/// * `signal` - The input signal as a real-valued 1D array
/// * `scales` - The scales at which to compute the CWT
/// * `wavelet_fn` - A function that generates the wavelet at a given scale and returns a `SignalResult<Vec<W>>`, where `W` can be converted to `Complex64`
/// * `center_frequency` - The center frequency of the wavelet (for frequency conversion)
/// * `config` - Configuration parameters for the transform
///
/// # Returns
///
/// A `SynchroCwtResult` containing the synchrosqueezed transform and optionally the CWT and instantaneous frequencies
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_signal::sswt::{synchrosqueezed_cwt, SynchroCwtConfig};
/// use scirs2_signal::wavelets;
///
/// // Create a chirp signal
/// let t = Array1::linspace(0.0, 10.0, 1000);
/// let f0 = 1.0;
/// let f1 = 10.0;
/// let rate = (f1 - f0) / 10.0;
/// let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * (f0 * ti + 0.5 * rate * ti * ti)).sin());
///
/// // Create logarithmically spaced scales (from 10 to 1)
/// let scales = Array1::logspace(10.0, 10.0_f64.log10(), 1.0_f64.log10(), 64);
///
/// // Configure the transform
/// let mut config = SynchroCwtConfig::default();
/// config.frequencies = Array1::linspace(1.0, 15.0, 140);
///
/// // Compute the synchrosqueezed transform
/// let result = synchrosqueezed_cwt(
///     &signal,
///     &scales,
///     |points, scale| wavelets::morlet(points, 5.0, scale),
///     5.0,
///     config
/// ).unwrap();
///
/// // The result.sst contains the synchrosqueezed transform
/// ```
pub fn synchrosqueezed_cwt<F, W>(
    signal: &Array1<f64>,
    scales: &Array1<f64>,
    wavelet_fn: F,
    center_frequency: f64,
    config: SynchroCwtConfig,
) -> SignalResult<SynchroCwtResult>
where
    F: Fn(usize, f64) -> SignalResult<Vec<W>>,
    W: Into<Complex64> + Copy,
{
    // Convert signal to Vec for wavelets::cwt
    let signal_vec: Vec<f64> = signal.iter().copied().collect();

    // Convert scales to Vec for wavelets::cwt
    let scales_vec: Vec<f64> = scales.iter().copied().collect();

    // Compute the continuous wavelet transform
    let cwt_vec = wavelets::cwt(&signal_vec, wavelet_fn, &scales_vec)?;

    // Convert the CWT result to Array2
    let n_scales = scales.len();
    let n_samples = signal.len();
    let mut cwt_coeffs = Array2::zeros((n_scales, n_samples));

    for (i, row) in cwt_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cwt_coeffs[[i, j]] = val;
        }
    }

    // Compute the instantaneous frequencies
    let omega =
        compute_instantaneous_frequencies(&cwt_coeffs, scales, signal.len(), center_frequency)?;

    // Perform the synchrosqueezing operation
    let sst = perform_synchrosqueezing(
        &cwt_coeffs,
        &omega,
        scales,
        &config.frequencies,
        config.gamma,
        center_frequency,
    )?;

    Ok(SynchroCwtResult {
        sst,
        cwt: if config.return_cwt {
            Some(cwt_coeffs)
        } else {
            None
        },
        omega: if config.return_cwt { Some(omega) } else { None },
        scales: scales.clone(),
        frequencies: config.frequencies.clone(),
    })
}

/// Compute the instantaneous frequencies from CWT coefficients
///
/// This function calculates the instantaneous frequency at each time-scale point
/// by computing the derivative of the phase of the CWT coefficients.
fn compute_instantaneous_frequencies(
    cwt: &Array2<Complex64>,
    scales: &Array1<f64>,
    n_samples: usize,
    center_frequency: f64,
) -> SignalResult<Array2<f64>> {
    let n_scales = scales.len();

    // Allocate output array
    let mut omega = Array2::zeros((n_scales, n_samples));

    // For each scale
    for (i, &scale) in scales.iter().enumerate() {
        let scale_row = cwt.slice(s![i, ..]);

        // For each time point (excluding endpoints for derivative calculation)
        for t in 1..n_samples - 1 {
            let prev = scale_row[t - 1];
            let curr = scale_row[t];
            let next = scale_row[t + 1];

            // Skip if the magnitude is too small (to avoid numerical issues)
            if curr.norm() < 1e-10 {
                omega[[i, t]] = 0.0;
                continue;
            }

            // Compute phase using central difference approximation
            let phase_diff = (next.arg() - prev.arg()) / 2.0;

            // Phase unwrapping (handle phase wrapping around ±π)
            let mut phase_diff_unwrapped = phase_diff;
            if phase_diff > PI {
                phase_diff_unwrapped -= 2.0 * PI;
            } else if phase_diff < -PI {
                phase_diff_unwrapped += 2.0 * PI;
            }

            // Convert to instantaneous frequency using the center frequency of the wavelet
            omega[[i, t]] = if phase_diff_unwrapped.abs() > 1e-10 {
                center_frequency / scale / 2.0 / PI + phase_diff_unwrapped / 2.0 / PI
            } else {
                center_frequency / scale
            };
        }

        // Handle endpoints using forward/backward differences
        if n_samples > 1 {
            // First point
            let curr = scale_row[0];
            let next = scale_row[1];
            if curr.norm() > 1e-10 {
                let mut phase_diff = next.arg() - curr.arg();
                if phase_diff > PI {
                    phase_diff -= 2.0 * PI;
                } else if phase_diff < -PI {
                    phase_diff += 2.0 * PI;
                }
                omega[[i, 0]] = center_frequency / scale / 2.0 / PI + phase_diff / 2.0 / PI;
            } else {
                omega[[i, 0]] = center_frequency / scale;
            }

            // Last point
            let prev = scale_row[n_samples - 2];
            let curr = scale_row[n_samples - 1];
            if curr.norm() > 1e-10 {
                let mut phase_diff = curr.arg() - prev.arg();
                if phase_diff > PI {
                    phase_diff -= 2.0 * PI;
                } else if phase_diff < -PI {
                    phase_diff += 2.0 * PI;
                }
                omega[[i, n_samples - 1]] =
                    center_frequency / scale / 2.0 / PI + phase_diff / 2.0 / PI;
            } else {
                omega[[i, n_samples - 1]] = center_frequency / scale;
            }
        }
    }

    Ok(omega)
}

/// Perform the synchrosqueezing operation to reassign energy in the time-frequency plane
fn perform_synchrosqueezing(
    cwt: &Array2<Complex64>,
    omega: &Array2<f64>,
    scales: &Array1<f64>,
    frequencies: &Array1<f64>,
    gamma: f64,
    _center_frequency: f64, // Unused but kept for API compatibility
) -> SignalResult<Array2<Complex64>> {
    let n_freqs = frequencies.len();
    let n_samples = cwt.shape()[1];

    // Create the output array
    let mut sst = Array2::zeros((n_freqs, n_samples));

    // Check if frequency vector is empty
    if frequencies.is_empty() {
        return Err(SignalError::ValueError(
            "Empty frequency vector for synchrosqueezing".to_string(),
        ));
    }

    // For each time point
    for t in 0..n_samples {
        // For each scale
        for (i, &scale) in scales.iter().enumerate() {
            let cwt_val = cwt[[i, t]];
            let inst_freq = omega[[i, t]];

            // Skip if coefficient magnitude is below threshold or frequency is invalid
            if cwt_val.norm() <= gamma || inst_freq <= 0.0 {
                continue;
            }

            // Find the closest frequency bin
            let freq_idx = find_closest_freq_bin(inst_freq, frequencies);

            if freq_idx < n_freqs {
                // Apply the reassignment with proper normalization
                // The factor sqrt(scale) accounts for the energy normalization in the CWT
                sst[[freq_idx, t]] += cwt_val / scale.sqrt();
            }
        }
    }

    Ok(sst)
}

/// Find the index of the closest frequency bin
fn find_closest_freq_bin(freq: f64, frequencies: &Array1<f64>) -> usize {
    let mut closest_idx = 0;
    let mut min_diff = f64::INFINITY;

    for (i, &bin_freq) in frequencies.iter().enumerate() {
        let diff = (freq - bin_freq).abs();
        if diff < min_diff {
            min_diff = diff;
            closest_idx = i;
        }
    }

    closest_idx
}

/// Generate a logarithmically spaced vector of scales for wavelet analysis
///
/// # Arguments
///
/// * `min_scale` - The minimum scale value
/// * `max_scale` - The maximum scale value
/// * `n_scales` - The number of scales to generate
///
/// # Returns
///
/// An Array1 containing logarithmically spaced scale values
///
/// # Example
///
/// ```
/// use scirs2_signal::sswt::log_scales;
///
/// let scales = log_scales(1.0, 64.0, 32);
/// assert_eq!(scales.len(), 32);
/// assert!(scales[0] >= 1.0 && scales[0] <= scales[1]);
/// assert!(scales[31] <= 64.0 && scales[31] >= scales[30]);
/// ```
pub fn log_scales(min_scale: f64, max_scale: f64, n_scales: usize) -> Array1<f64> {
    if min_scale <= 0.0 || max_scale <= 0.0 || min_scale >= max_scale {
        panic!("Scales must be positive with min_scale < max_scale");
    }

    let min_log = min_scale.ln();
    let max_log = max_scale.ln();

    let log_space = Array1::linspace(min_log, max_log, n_scales);
    log_space.mapv(|x| x.exp())
}

/// Generate a set of frequency bins for synchrosqueezed wavelet transform
///
/// # Arguments
///
/// * `min_freq` - The minimum frequency
/// * `max_freq` - The maximum frequency
/// * `n_freqs` - The number of frequency bins to generate
///
/// # Returns
///
/// An Array1 containing linearly spaced frequency values
///
/// # Example
///
/// ```
/// use scirs2_signal::sswt::frequency_bins;
///
/// let freqs = frequency_bins(1.0, 64.0, 64);
/// assert_eq!(freqs.len(), 64);
/// assert!(freqs[0] >= 1.0 && freqs[0] <= freqs[1]);
/// assert!(freqs[63] <= 64.0 && freqs[63] >= freqs[62]);
/// ```
pub fn frequency_bins(min_freq: f64, max_freq: f64, n_freqs: usize) -> Array1<f64> {
    if min_freq < 0.0 || max_freq <= 0.0 || min_freq >= max_freq {
        panic!("Frequencies must be non-negative with min_freq < max_freq");
    }

    Array1::linspace(min_freq, max_freq, n_freqs)
}

/// Extract ridges from a synchrosqueezed wavelet transform
///
/// Ridges are paths of maximum energy concentration in the time-frequency plane.
/// This function identifies the dominant frequencies at each time point.
///
/// # Arguments
///
/// * `sst` - The synchrosqueezed wavelet transform as a 2D array [frequency, time]
/// * `frequencies` - The frequency values corresponding to the first dimension of `sst`
/// * `max_ridges` - The maximum number of ridges to extract (default: 1)
///
/// # Returns
///
/// A vector of ridges, where each ridge is a vector of (time_index, frequency) pairs
pub fn extract_ridges(
    sst: &Array2<Complex64>,
    frequencies: &Array1<f64>,
    max_ridges: usize,
) -> Vec<Vec<(usize, f64)>> {
    let n_freqs = sst.shape()[0];
    let n_times = sst.shape()[1];

    let max_ridges = max_ridges.max(1);
    let mut ridges = Vec::with_capacity(max_ridges);

    // For each time point, find the frequencies with the highest energy
    for t in 0..n_times {
        let mut peak_indices = Vec::with_capacity(max_ridges);
        let mut peak_magnitudes = Vec::with_capacity(max_ridges);

        // Calculate magnitudes for this time slice
        let mut magnitudes = Vec::with_capacity(n_freqs);
        for f in 0..n_freqs {
            magnitudes.push(sst[[f, t]].norm());
        }

        // Find the max_ridges highest peaks
        for _ in 0..max_ridges {
            if let Some((idx, &mag)) = magnitudes
                .iter()
                .enumerate()
                .filter(|(idx, _)| !peak_indices.contains(idx))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                // Only add if magnitude is significant
                if mag > 0.1 * magnitudes.iter().cloned().fold(0.0, |a, b| a.max(b)) {
                    peak_indices.push(idx);
                    peak_magnitudes.push(mag);
                }
            } else {
                break; // No more peaks to find
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

    // Sort ridges by total energy
    ridges.sort_by(|a, b| {
        // Calculate total energy for each ridge
        let energy_a: f64 = a
            .iter()
            .map(|(t, freq)| {
                let freq_idx = find_closest_freq_bin(*freq, frequencies);
                sst[[freq_idx, *t]].norm()
            })
            .sum();

        let energy_b: f64 = b
            .iter()
            .map(|(t, freq)| {
                let freq_idx = find_closest_freq_bin(*freq, frequencies);
                sst[[freq_idx, *t]].norm()
            })
            .sum();

        // Sort in descending order of energy
        energy_b
            .partial_cmp(&energy_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ridges
}

/// Calculate the ridge-based reconstruction of a signal from its synchrosqueezed transform
///
/// # Arguments
///
/// * `sst` - The synchrosqueezed wavelet transform
/// * `ridge` - A vector of (time_index, frequency) pairs defining the ridge
/// * `frequencies` - The frequency bins used in the synchrosqueezed transform
///
/// # Returns
///
/// A reconstructed signal following the time-frequency ridge
pub fn reconstruct_from_ridge(
    sst: &Array2<Complex64>,
    ridge: &[(usize, f64)],
    frequencies: &Array1<f64>,
) -> SignalResult<Array1<Complex64>> {
    if ridge.is_empty() {
        return Err(SignalError::ValueError(
            "Empty ridge for reconstruction".to_string(),
        ));
    }

    let n_times = sst.shape()[1];

    // Check that the ridge covers all time points
    if ridge.len() != n_times {
        return Err(SignalError::ValueError(format!(
            "Ridge must cover all time points: expected {}, got {}",
            n_times,
            ridge.len()
        )));
    }

    // Create output signal
    let mut reconstructed = Array1::zeros(n_times);

    // For each time point
    for (t, (_, freq)) in ridge.iter().enumerate() {
        let freq_idx = find_closest_freq_bin(*freq, frequencies);
        reconstructed[t] = sst[[freq_idx, t]];
    }

    Ok(reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_log_scales() {
        let scales = log_scales(1.0, 64.0, 4);
        assert_eq!(scales.len(), 4);
        assert_relative_eq!(scales[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(scales[3], 64.0, epsilon = 1e-10);
    }

    #[test]
    fn test_frequency_bins() {
        let freqs = frequency_bins(1.0, 10.0, 10);
        assert_eq!(freqs.len(), 10);
        assert_relative_eq!(freqs[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(freqs[9], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_synchrosqueezed_cwt_chirp() {
        // Create a chirp signal
        let n_samples = 500;
        let t = Array1::linspace(0.0, 10.0, n_samples);
        let signal = t.mapv(|ti| (2.0 * PI * (1.0 * ti + 0.5 * 0.5 * ti * ti)).sin());

        // Create scales
        let scales = log_scales(1.0, 16.0, 32);

        // Configure the transform
        let config = SynchroCwtConfig {
            frequencies: frequency_bins(1.0, 10.0, 64),
            return_cwt: true,
            ..Default::default()
        };

        // Compute the synchrosqueezed transform with properly wrapped wavelet function
        let result = synchrosqueezed_cwt(
            &signal,
            &scales,
            |points, scale| wavelets::morlet(points, 5.0, scale),
            5.0,
            config,
        )
        .unwrap();

        // Verify dimensions
        assert_eq!(result.sst.shape()[0], 64); // Frequencies
        assert_eq!(result.sst.shape()[1], n_samples); // Time points

        // The CWT should be returned
        assert!(result.cwt.is_some());
        assert_eq!(result.cwt.unwrap().shape()[0], 32); // Scales

        // The instantaneous frequencies should be returned
        assert!(result.omega.is_some());

        // The synchrosqueezed transform should have energy concentrated around the chirp frequency
        // Check that the transform has reasonable values
        let mut has_significant_energy = false;
        for f in 0..result.sst.shape()[0] {
            for t in 0..result.sst.shape()[1] {
                if result.sst[[f, t]].norm() > 0.1 {
                    has_significant_energy = true;
                    break;
                }
            }
            if has_significant_energy {
                break;
            }
        }
        assert!(has_significant_energy);
    }

    #[test]
    fn test_find_closest_freq_bin() {
        let freqs = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(find_closest_freq_bin(0.9, &freqs), 0);
        assert_eq!(find_closest_freq_bin(1.4, &freqs), 0);
        assert_eq!(find_closest_freq_bin(1.6, &freqs), 1);
        assert_eq!(find_closest_freq_bin(4.7, &freqs), 4);
        assert_eq!(find_closest_freq_bin(5.2, &freqs), 4);
    }
}
