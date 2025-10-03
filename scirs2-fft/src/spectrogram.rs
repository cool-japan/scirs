//! Spectrogram module for time-frequency analysis
//!
//! This module provides functions for computing spectrograms of signals,
//! which are visual representations of the spectrum of frequencies as they
//! vary with time. It builds on the Short-Time Fourier Transform (STFT).
//!
//! A spectrogram is useful for analyzing audio signals, vibration data,
//! and other time-varying signals to understand how frequency content
//! changes over time.

use crate::error::{FFTError, FFTResult};
use crate::window::{get_window, Window};
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::f64::consts::PI;

/// Compute the Short-Time Fourier Transform (STFT) of a signal.
///
/// The STFT is used to determine the sinusoidal frequency and phase content
/// of local sections of a signal as it changes over time.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `window` - Window specification (function or array of length `nperseg`)
/// * `nperseg` - Length of each segment
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `nfft` - Length of the FFT used (default: `nperseg`)
/// * `fs` - Sampling frequency of the `x` time series (default: 1.0)
/// * `detrend` - Whether to remove the mean from each segment (default: true)
/// * `return_onesided` - If true, return half of the spectrum (real signals) (default: true)
/// * `boundary` - Behavior at boundaries (default: None)
///
/// # Returns
///
/// * A tuple containing:
///   - Frequencies vector (f)
///   - Time vector (t)
///   - STFT result matrix (Zxx) where rows are frequencies and columns are time segments
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::spectrogram::stft;
/// use scirs2_fft::window::Window;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let time = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = time.iter().map(|&ti| (2.0 * PI * (10.0 + 10.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute STFT
/// let (f, t, zxx) = stft(
///     &chirp,
///     Window::Hann,
///     256,
///     Some(128),
///     None,
///     Some(fs),
///     Some(true),
///     Some(true),
///     None,
/// ).unwrap();
///
/// // Check dimensions
/// assert_eq!(f.len(), zxx.shape()[0]);
/// assert_eq!(t.len(), zxx.shape()[1]);
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn stft<T>(
    x: &[T],
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    fs: Option<f64>,
    detrend: Option<bool>,
    return_onesided: Option<bool>,
    boundary: Option<&str>,
) -> FFTResult<(Vec<f64>, Vec<f64>, Array2<Complex64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Input validation
    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }

    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }

    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let nfft = nfft.unwrap_or(nperseg);
    if nfft < nperseg {
        return Err(FFTError::ValueError(
            "FFT length must be greater than or equal to segment length".to_string(),
        ));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(FFTError::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let detrend = detrend.unwrap_or(true);
    let return_onesided = return_onesided.unwrap_or(true);

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                FFTError::ValueError(format!("Could not convert value to f64: {val:?}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Generate window function
    let win = get_window(window, nperseg, true)?;

    // Calculate step size
    let step = nperseg - noverlap;

    // Compute number of segments
    let mut num_frames = 1 + (x_f64.len() - nperseg) / step;

    // Handle boundary conditions
    let mut padded = x_f64.clone();
    match boundary {
        Some("reflect") => {
            // Reflect signal at boundaries
            let pad_size = nperseg;
            let mut reflected = Vec::with_capacity(x_f64.len() + 2 * pad_size);
            // Left padding
            for i in (0..pad_size).rev() {
                reflected.push(x_f64[i]);
            }
            // Original signal
            reflected.extend_from_slice(&x_f64);
            // Right padding
            let len = x_f64.len();
            for i in (len - pad_size..len).rev() {
                reflected.push(x_f64[i]);
            }
            padded = reflected;
            num_frames = 1 + (padded.len() - nperseg) / step;
        }
        Some("zeros") | Some("constant") => {
            // Pad with zeros or last value
            let pad_size = nperseg;
            let mut padded_signal = Vec::with_capacity(x_f64.len() + 2 * pad_size);

            // Left padding
            if boundary == Some("zeros") {
                padded_signal.extend(vec![0.0; pad_size]);
            } else {
                padded_signal.extend(vec![x_f64[0]; pad_size]);
            }

            // Original signal
            padded_signal.extend_from_slice(&x_f64);

            // Right padding
            if boundary == Some("zeros") {
                padded_signal.extend(vec![0.0; pad_size]);
            } else {
                padded_signal.extend(vec![*x_f64.last().unwrap_or(&0.0); pad_size]);
            }

            padded = padded_signal;
            num_frames = 1 + (padded.len() - nperseg) / step;
        }
        _ => {}
    }

    // Calculate frequency values
    let freq_len = if return_onesided { nfft / 2 + 1 } else { nfft };
    let frequencies: Vec<f64> = (0..freq_len).map(|i| i as f64 * fs / nfft as f64).collect();

    // Calculate time values (center of each segment)
    let times: Vec<f64> = (0..num_frames)
        .map(|i| (i * step + nperseg / 2) as f64 / fs)
        .collect();

    // Compute STFT
    let mut stft_matrix = Array2::zeros((freq_len, num_frames));

    for (i, time_idx) in (0..padded.len() - nperseg + 1).step_by(step).enumerate() {
        if i >= num_frames {
            break;
        }

        // Extract segment
        let segment: Vec<f64> = padded[time_idx..time_idx + nperseg].to_vec();

        // Detrend if required
        let mut detrended = segment;
        if detrend {
            let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
            detrended.iter_mut().for_each(|x| *x -= mean);
        }

        // Apply window
        let windowed: Vec<f64> = detrended
            .iter()
            .zip(win.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // Pad with zeros if nfft > nperseg
        let mut padded_segment = windowed;
        if nfft > nperseg {
            padded_segment.extend(vec![0.0; nfft - nperseg]);
        }

        // Compute FFT
        let fft_result = crate::fft::fft(&padded_segment, None)?;

        // Store result (keep only first half for real signals if return_onesided is true)
        let relevant_fft = if return_onesided {
            fft_result[0..freq_len].to_vec()
        } else {
            fft_result
        };

        for (j, &value) in relevant_fft.iter().enumerate() {
            stft_matrix[[j, i]] = value;
        }
    }

    Ok((frequencies, times, stft_matrix))
}

/// Compute a spectrogram of a time-domain signal.
///
/// A spectrogram is a visual representation of the frequency spectrum of
/// a signal as it varies with time. It is often displayed as a heatmap
/// where the x-axis represents time, the y-axis represents frequency,
/// and the color intensity represents signal power.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `window` - Window specification (function or array of length `nperseg`) (default: Hann)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `nfft` - Length of the FFT used (default: `nperseg`)
/// * `detrend` - Whether to remove the mean from each segment (default: true)
/// * `scaling` - Power spectrum scaling mode: "density" or "spectrum" (default: "density")
/// * `mode` - Power spectrum mode: "psd", "magnitude", "angle", "phase" (default: "psd")
///
/// # Returns
///
/// * A tuple containing:
///   - Frequencies vector (f)
///   - Time vector (t)
///   - Spectrogram result matrix (Sxx) where rows are frequencies and columns are time segments
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::spectrogram;
/// use scirs2_fft::window::Window;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let time = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = time.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute spectrogram
/// let (f, t, sxx) = spectrogram(
///     &chirp,
///     Some(fs),
///     Some(Window::Hann),
///     Some(128),
///     Some(64),
///     None,
///     Some(true),
///     Some("density"),
///     Some("psd"),
/// ).unwrap();
///
/// // Check dimensions
/// assert_eq!(f.len(), sxx.shape()[0]);
/// assert_eq!(t.len(), sxx.shape()[1]);
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn spectrogram<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<Window>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    detrend: Option<bool>,
    scaling: Option<&str>,
    mode: Option<&str>,
) -> FFTResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let window = window.unwrap_or(Window::Hann);
    let nperseg = nperseg.unwrap_or(256);

    // Compute STFT
    let (frequencies, times, stft_result) = stft(
        x,
        window.clone(),
        nperseg,
        noverlap,
        nfft,
        Some(fs),
        detrend,
        Some(true), // Always use onesided for real signals
        None,
    )?;

    // Determine scaling factor
    let win_vals = get_window(window, nperseg, true)?;
    let win_sum_sq = win_vals.iter().map(|&x| x * x).sum::<f64>();

    let scaling = scaling.unwrap_or("density");
    let scale_factor = match scaling {
        "density" => 1.0 / (fs * win_sum_sq),
        "spectrum" => 1.0 / win_sum_sq,
        _ => {
            return Err(FFTError::ValueError(format!(
                "Unknown scaling mode: {scaling}. Use 'density' or 'spectrum'."
            )));
        }
    };

    // Compute spectrogram based on the requested mode
    let mode = mode.unwrap_or("psd");
    let spectrogram_result = match mode {
        "psd" => {
            // Power spectral density
            let mut psd = Array2::zeros(stft_result.dim());
            for (i, row) in stft_result.axis_iter(Axis(0)).enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    psd[[i, j]] = val.norm_sqr() * scale_factor;
                }
            }
            psd
        }
        "magnitude" => {
            // Magnitude spectrum (linear scale)
            let mut magnitude = Array2::zeros(stft_result.dim());
            for (i, row) in stft_result.axis_iter(Axis(0)).enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    magnitude[[i, j]] = val.norm() * scale_factor.sqrt();
                }
            }
            magnitude
        }
        "angle" | "phase" => {
            // Phase spectrum in radians or degrees
            let mut phase = Array2::zeros(stft_result.dim());
            for (i, row) in stft_result.axis_iter(Axis(0)).enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    phase[[i, j]] = val.arg();
                    if mode == "angle" {
                        // Convert to degrees
                        phase[[i, j]] = phase[[i, j]] * 180.0 / PI;
                    }
                }
            }
            phase
        }
        _ => {
            return Err(FFTError::ValueError(format!(
                "Unknown mode: {mode}. Use 'psd', 'magnitude', 'angle', or 'phase'."
            )));
        }
    };

    Ok((frequencies, times, spectrogram_result))
}

/// Compute a normalized spectrogram suitable for display as a heatmap.
///
/// This is a convenience function that computes a spectrogram and normalizes
/// its values to a range suitable for visualization. It also applies a
/// logarithmic scaling to better visualize the dynamic range of the signal.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `db_range` - Dynamic range in dB for normalization (default: 80.0)
///
/// # Returns
///
/// * A tuple containing:
///   - Frequencies vector (f)
///   - Time vector (t)
///   - Normalized spectrogram result matrix (Sxx_norm) with values in [0, 1]
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::spectrogram_normalized;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let time = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = time.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute normalized spectrogram
/// let (f, t, sxx_norm) = spectrogram_normalized(
///     &chirp,
///     Some(fs),
///     Some(128),
///     Some(64),
///     Some(80.0),
/// ).unwrap();
///
/// // Values should be normalized to [0, 1]
/// for row in sxx_norm.axis_iter(scirs2_core::ndarray::Axis(0)) {
///     for &val in row {
///         assert!((0.0..=1.0).contains(&val));
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn spectrogram_normalized<T>(
    x: &[T],
    fs: Option<f64>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    db_range: Option<f64>,
) -> FFTResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let nperseg = nperseg.unwrap_or(256);
    let db_range = db_range.unwrap_or(80.0);

    // Compute spectrogram
    let (frequencies, times, spectrogram_result) = spectrogram(
        x,
        Some(fs),
        Some(Window::Hann),
        Some(nperseg),
        noverlap,
        None,
        Some(true),
        Some("density"),
        Some("psd"),
    )?;

    // Convert to dB scale with reference to maximum value
    let max_val = spectrogram_result.iter().fold(f64::MIN, |a, &b| a.max(b));

    if max_val <= 0.0 {
        return Err(FFTError::ValueError(
            "Spectrogram has no positive values".to_string(),
        ));
    }

    // Convert to dB
    let mut spec_db = Array2::zeros(spectrogram_result.dim());
    for (i, row) in spectrogram_result.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            // Avoid taking log of zero
            let val_db = if val > 0.0 {
                10.0 * (val / max_val).log10()
            } else {
                -db_range
            };
            spec_db[[i, j]] = val_db;
        }
    }

    // Normalize to [0, 1] _range
    let mut spec_norm = Array2::zeros(spec_db.dim());
    for (i, row) in spec_db.axis_iter(Axis(0)).enumerate() {
        for (j, &val) in row.iter().enumerate() {
            spec_norm[[i, j]] = (val + db_range).max(0.0).min(db_range) / db_range;
        }
    }

    Ok((frequencies, times, spec_norm))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generate a test signal (sine wave)
    fn generate_sine_wave(_freq: f64, fs: f64, n_samples: usize) -> Vec<f64> {
        (0..n_samples)
            .map(|i| (2.0 * PI * _freq * (i as f64 / fs)).sin())
            .collect()
    }

    #[test]
    fn test_stft_dimensions() {
        // Generate a sine wave
        let fs = 1000.0;
        let signal = generate_sine_wave(100.0, fs, 1000);

        // Compute STFT
        let nperseg = 256;
        let noverlap = 128;
        let (f, t, zxx) = stft(
            &signal,
            Window::Hann,
            nperseg,
            Some(noverlap),
            None,
            Some(fs),
            Some(true),
            Some(true),
            None,
        )
        .expect("STFT computation should succeed for test data");

        // Check dimensions
        let expected_num_freqs = nperseg / 2 + 1;
        let expected_num_frames = 1 + (signal.len() - nperseg) / (nperseg - noverlap);

        assert_eq!(f.len(), expected_num_freqs);
        assert_eq!(t.len(), expected_num_frames);
        assert_eq!(zxx.shape(), &[expected_num_freqs, expected_num_frames]);
    }

    #[test]
    fn test_stft_frequency_content() {
        // Generate a sine wave with known frequency
        let fs = 1000.0;
        let freq = 100.0;
        let signal = generate_sine_wave(freq, fs, 1000);

        // Compute STFT
        let nperseg = 256;
        let (f_freq, f_t, zxx) = stft(
            &signal,
            Window::Hann,
            nperseg,
            Some(128),
            None,
            Some(fs),
            Some(true),
            Some(true),
            None,
        )
        .expect("STFT computation should succeed for frequency test");

        // Find the frequency bin closest to our signal frequency
        let freq_idx = f_freq
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - freq)
                    .abs()
                    .partial_cmp(&(b - freq).abs())
                    .expect("Frequency comparison should succeed")
            })
            .expect("Should find minimum frequency difference")
            .0;

        // Check that the power at this frequency is higher than at other frequencies
        let mean_frame_idx = zxx.shape()[1] / 2; // Use middle frame
        let power_at_freq = zxx[[freq_idx, mean_frame_idx]].norm_sqr();

        // Calculate average power across all frequencies
        let total_power: f64 = (0..zxx.shape()[0])
            .map(|i| zxx[[i, mean_frame_idx]].norm_sqr())
            .sum();
        let avg_power = total_power / zxx.shape()[0] as f64;

        // The power at our signal frequency should be much higher than average
        assert!(power_at_freq > 5.0 * avg_power);
    }

    #[test]
    fn test_spectrogram() {
        // Generate a chirp signal (frequency increasing with time)
        let fs = 1000.0;
        let n_samples = 1000;
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let chirp: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin())
            .collect();

        // Compute spectrogram
        let (f, t, sxx) = spectrogram(
            &chirp,
            Some(fs),
            Some(Window::Hann),
            Some(128),
            Some(64),
            None,
            Some(true),
            Some("density"),
            Some("psd"),
        )
        .expect("Spectrogram computation should succeed for test data");

        // Verify basic properties
        assert!(!f.is_empty());
        assert!(!t.is_empty());
        assert_eq!(sxx.shape(), &[f.len(), t.len()]);

        // Values should be non-negative for PSD
        for &val in sxx.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_spectrogram_modes() {
        // Generate a sine wave
        let fs = 1000.0;
        let signal = generate_sine_wave(100.0, fs, 1000);

        // Test different modes
        let modes = ["psd", "magnitude", "angle", "phase"];

        for &mode in &modes {
            let (f, t, sxx) = spectrogram(
                &signal,
                Some(fs),
                Some(Window::Hann),
                Some(128),
                Some(64),
                None,
                Some(true),
                Some("density"),
                Some(mode),
            )
            .expect("Spectrogram mode computation should succeed");

            // Check dimensions
            assert!(!f.is_empty());
            assert!(!t.is_empty());
            assert_eq!(sxx.shape(), &[f.len(), t.len()]);

            // For phase/angle modes, values should be in expected range
            if mode == "phase" {
                for &val in sxx.iter() {
                    assert!((-PI..=PI).contains(&val));
                }
            } else if mode == "angle" {
                for &val in sxx.iter() {
                    assert!((-180.0..=180.0).contains(&val));
                }
            }
        }
    }

    #[test]
    fn test_spectrogram_normalized() {
        // Generate a sine wave
        let fs = 1000.0;
        let signal = generate_sine_wave(100.0, fs, 1000);

        // Compute normalized spectrogram
        let (f, t, sxx) =
            spectrogram_normalized(&signal, Some(fs), Some(128), Some(64), Some(80.0))
                .expect("Normalized spectrogram should succeed");

        // Check dimensions
        assert!(!f.is_empty());
        assert!(!t.is_empty());
        assert_eq!(sxx.shape(), &[f.len(), t.len()]);

        // Values should be in range [0, 1]
        for &val in sxx.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
