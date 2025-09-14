// Spectral analysis functions
//
// This module provides functions for estimating power spectral densities and spectrograms.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rand::Rng;
use std::f64::consts::PI;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Type alias for periodogram result containing frequencies and power spectral density values
pub type PeriodogramResult = SignalResult<(Vec<f64>, Vec<f64>)>;

/// Type alias for STFT result containing frequencies, time segments, and complex values
pub type StftResult = SignalResult<(Vec<f64>, Vec<f64>, Vec<Vec<Complex64>>)>;

/// Type alias for spectrogram result containing frequencies, time segments, and magnitude values
pub type SpectrogramResult = SignalResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)>;

/// Create a window function
///
/// # Arguments
///
/// * `window_type` - Window type (e.g., "hann", "hamming", "blackman", "boxcar")
/// * `nperseg` - Window length
///
/// # Returns
///
/// * Window function as a vector of length `nperseg`
#[allow(dead_code)]
fn get_window(_windowtype: &str, nperseg: usize) -> SignalResult<Vec<f64>> {
    match _windowtype.to_lowercase().as_str() {
        "hann" => {
            let mut window = Vec::with_capacity(nperseg);
            for i in 0..nperseg {
                let value = 0.5 * (1.0 - (2.0 * PI * i as f64 / (nperseg - 1) as f64).cos());
                window.push(value);
            }
            Ok(window)
        }
        "hamming" => {
            let mut window = Vec::with_capacity(nperseg);
            for i in 0..nperseg {
                let value = 0.54 - 0.46 * (2.0 * PI * i as f64 / (nperseg - 1) as f64).cos();
                window.push(value);
            }
            Ok(window)
        }
        "blackman" => {
            let mut window = Vec::with_capacity(nperseg);
            for i in 0..nperseg {
                let value = 0.42 - 0.5 * (2.0 * PI * i as f64 / (nperseg - 1) as f64).cos()
                    + 0.08 * (4.0 * PI * i as f64 / (nperseg - 1) as f64).cos();
                window.push(value);
            }
            Ok(window)
        }
        "boxcar" | "rectangular" => Ok(vec![1.0; nperseg]),
        _ => Err(SignalError::ValueError(format!(
            "Unknown window type: {}",
            _windowtype
        ))),
    }
}

/// Apply detrending to a signal
///
/// # Arguments
///
/// * `x` - Input signal
/// * `detrend_type` - Type of detrending to apply ("constant", "linear", or "none")
///
/// # Returns
///
/// * Detrended signal
#[allow(dead_code)]
fn apply_detrend(x: &[f64], detrendtype: &str) -> SignalResult<Vec<f64>> {
    match detrendtype {
        "constant" => {
            // Remove mean
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            Ok(x.iter().map(|&v| v - mean).collect())
        }
        "linear" => {
            // Remove linear trend
            let n = x.len();
            let x_indices: Vec<f64> = (0..n).map(|i| i as f64).collect();

            // Linear regression
            let sum_x = x_indices.iter().sum::<f64>();
            let sum_y = x.iter().sum::<f64>();
            let sum_xx = x_indices.iter().map(|&x| x * x).sum::<f64>();
            let sum_xy = x_indices
                .iter()
                .zip(x.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();

            let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n as f64;

            // Remove trend
            Ok(x.iter()
                .enumerate()
                .map(|(i, &y)| y - (slope * i as f64 + intercept))
                .collect())
        }
        "none" => Ok(x.to_vec()),
        _ => Err(SignalError::ValueError(format!(
            "Unknown detrend option: {}",
            detrendtype
        ))),
    }
}

/// Estimate the power spectral density using periodogram method
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "boxcar")
/// * `nfft` - Length of the FFT (default = length of x)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
///
/// # Returns
///
/// * A tuple containing (frequencies, power spectral density)
#[allow(dead_code)]
pub fn periodogram<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<&str>,
    nfft: Option<usize>,
    detrend: Option<&str>,
    scaling: Option<&str>,
) -> PeriodogramResult
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Default parameters
    let fs_val = fs.unwrap_or(1.0);
    let nfft_val = nfft.unwrap_or(x_f64.len());
    let window_val = window.unwrap_or("boxcar");
    let detrend_val = detrend.unwrap_or("constant");
    let scaling_val = scaling.unwrap_or("density");

    // Validate parameters
    if fs_val <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Sampling frequency must be positive, got {}",
            fs_val
        )));
    }

    if nfft_val < x_f64.len() {
        return Err(SignalError::ValueError(format!(
            "NFFT must be at least as large as signal length, got {} < {}",
            nfft_val,
            x_f64.len()
        )));
    }

    // 1. Apply detrending
    let detrended = apply_detrend(&x_f64, detrend_val)?;

    // 2. Apply window
    let win = get_window(window_val, x_f64.len())?;
    let windowed: Vec<f64> = detrended
        .iter()
        .zip(win.iter())
        .map(|(&x, &w)| x * w)
        .collect();

    // Scale factor for the window
    let win_scale = win.iter().map(|&w| w.powi(2)).sum::<f64>();
    let scale = 1.0 / win_scale;

    // 3. Zero-pad the signal if needed
    let mut padded = windowed.clone();
    if nfft_val > padded.len() {
        padded.resize(nfft_val, 0.0);
    }

    // 4. Compute FFT
    let spectrum = scirs2_fft::fft(&padded, None)
        .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {}", e)))?;

    // 5. Compute periodogram (using magnitude squared)
    let periodogram: Vec<f64> = spectrum
        .iter()
        .map(|&c| c.norm_sqr() * scale / (fs_val * x_f64.len() as f64))
        .collect();

    // 6. Generate frequency axis - be careful with bounds
    let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {
        SignalError::ComputationError(format!("Frequency computation error: {}", e))
    })?;

    // 7. Keep only positive frequencies
    let mut result_freqs = Vec::new();
    let mut result_psd = Vec::new();

    let n_half = (nfft_val / 2) + (nfft_val % 2); // Handle both even and odd nfft

    for i in 0..n_half {
        if i < freqs.len() && i < periodogram.len() {
            result_freqs.push(freqs[i]);

            // Adjust scaling based on user preference
            let psd_val = if scaling_val == "density" {
                periodogram[i]
            } else {
                // "spectrum" - multiply by sampling frequency
                periodogram[i] * fs_val
            };

            result_psd.push(psd_val);
        }
    }

    Ok((result_freqs, result_psd))
}

/// Estimate the power spectral density using Welch's method
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
///
/// # Returns
///
/// * A tuple containing (frequencies, power spectral density)
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn welch<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    detrend: Option<&str>,
    scaling: Option<&str>,
) -> PeriodogramResult
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Default parameters
    let fs_val = fs.unwrap_or(1.0);
    let nperseg_val = nperseg.unwrap_or(256.min(x_f64.len()));
    let noverlap_val = noverlap.unwrap_or(nperseg_val / 2);
    let nfft_val = nfft.unwrap_or(nperseg_val);
    let window_val = window.unwrap_or("hann");
    let detrend_val = detrend.unwrap_or("constant");
    let scaling_val = scaling.unwrap_or("density");

    // Validate parameters
    if fs_val <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Sampling frequency must be positive, got {}",
            fs_val
        )));
    }

    if nfft_val < nperseg_val {
        return Err(SignalError::ValueError(format!(
            "nfft must be at least as large as nperseg, got {} < {}",
            nfft_val, nperseg_val
        )));
    }

    if noverlap_val >= nperseg_val {
        return Err(SignalError::ValueError(format!(
            "noverlap must be less than nperseg, got {} >= {}",
            noverlap_val, nperseg_val
        )));
    }

    // Create window function
    let win = get_window(window_val, nperseg_val)?;
    let win_scale = win.iter().map(|&w| w.powi(2)).sum::<f64>();
    let scale = 1.0 / win_scale;

    // Determine number of segments
    let step = nperseg_val - noverlap_val;
    let num_segments = if step > 0 {
        (x_f64.len() - noverlap_val) / step
    } else {
        0
    };

    if num_segments < 1 {
        return Err(SignalError::ValueError(
            "Not enough data points for given nperseg and noverlap".to_string(),
        ));
    }

    // Calculate frequency bins
    let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {
        SignalError::ComputationError(format!("Frequency computation error: {}", e))
    })?;

    // Keep only positive frequencies
    let n_half = (nfft_val / 2) + (nfft_val % 2); // Handle both even and odd nfft
    let result_freqs: Vec<f64> = freqs.into_iter().take(n_half).collect();

    // Initialize averaged periodogram
    let mut psd_avg = vec![0.0; n_half];

    // Process each segment
    for i in 0..num_segments {
        let start = i * step;
        let end = start + nperseg_val;

        if end > x_f64.len() {
            break;
        }

        // Extract segment
        let segment = x_f64[start..end].to_vec();

        // Detrend the segment
        let detrended = apply_detrend(&segment, detrend_val)?;

        // Apply window
        let windowed: Vec<f64> = detrended
            .iter()
            .zip(win.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // Zero-pad if needed
        let mut padded = windowed.clone();
        if nfft_val > padded.len() {
            padded.resize(nfft_val, 0.0);
        }

        // Compute FFT
        let spectrum = scirs2_fft::fft(&padded, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {}", e)))?;

        // Compute periodogram for this segment
        let segment_psd: Vec<f64> = spectrum
            .iter()
            .take(n_half)
            .map(|&c| c.norm_sqr() * scale / (fs_val * nperseg_val as f64))
            .collect();

        // Accumulate into average
        for (j, &psd) in segment_psd.iter().enumerate() {
            if j < psd_avg.len() {
                psd_avg[j] += psd;
            }
        }
    }

    // Normalize by number of segments
    for psd in &mut psd_avg {
        *psd /= num_segments as f64;
    }

    // Apply scaling
    let result_psd = if scaling_val == "density" {
        psd_avg
    } else {
        // "spectrum" - multiply by sampling frequency
        psd_avg.iter().map(|&p| p * fs_val).collect()
    };

    Ok((result_freqs, result_psd))
}

/// Apply boundary handling to a signal for STFT
///
/// # Arguments
///
/// * `x` - Input signal
/// * `nperseg` - Segment length
/// * `boundary` - Boundary handling method ("zeros", "extend", "none")
///
/// # Returns
///
/// * Padded signal
#[allow(dead_code)]
fn apply_boundary(x: &[f64], nperseg: usize, boundary: &str) -> SignalResult<Vec<f64>> {
    match boundary {
        "zeros" => {
            // Pad with zeros on both sides
            let pad_len = nperseg / 2;
            let mut padded = vec![0.0; pad_len];
            padded.extend_from_slice(x);
            padded.extend(vec![0.0; pad_len]);
            Ok(padded)
        }
        "extend" => {
            // Pad with edge values
            let pad_len = nperseg / 2;
            let mut padded = vec![x[0]; pad_len];
            padded.extend_from_slice(x);
            padded.extend(vec![*x.last().unwrap_or(&0.0); pad_len]);
            Ok(padded)
        }
        "none" => {
            // No padding
            Ok(x.to_vec())
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown boundary option: {}",
            boundary
        ))),
    }
}

/// Short-time Fourier transform
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `boundary` - How to handle boundaries ("zeros", "extend", or "none")
/// * `padded` - Whether to pad the signal (default = true)
///
/// # Returns
///
/// * A tuple containing (frequencies, time segments, STFT values)
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn stft<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    detrend: Option<&str>,
    boundary: Option<&str>,
    padded: Option<bool>,
) -> StftResult
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Default parameters
    let fs_val = fs.unwrap_or(1.0);
    let nperseg_val = nperseg.unwrap_or(256.min(x_f64.len()));
    let noverlap_val = noverlap.unwrap_or(nperseg_val / 2);
    let nfft_val = nfft.unwrap_or(nperseg_val);
    let window_val = window.unwrap_or("hann");
    let detrend_val = detrend.unwrap_or("constant");
    let boundary_val = boundary.unwrap_or("zeros");
    let padded_val = padded.unwrap_or(true);

    // Validate parameters
    if fs_val <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Sampling frequency must be positive, got {}",
            fs_val
        )));
    }

    if nfft_val < nperseg_val {
        return Err(SignalError::ValueError(format!(
            "nfft must be at least as large as nperseg, got {} < {}",
            nfft_val, nperseg_val
        )));
    }

    if noverlap_val >= nperseg_val {
        return Err(SignalError::ValueError(format!(
            "noverlap must be less than nperseg, got {} >= {}",
            noverlap_val, nperseg_val
        )));
    }

    // Create window function
    let win = get_window(window_val, nperseg_val)?;

    // Apply boundary handling if needed
    let input_signal = if padded_val {
        apply_boundary(&x_f64, nperseg_val, boundary_val)?
    } else {
        x_f64.clone()
    };

    // Determine number of segments and step size
    let step = nperseg_val - noverlap_val;
    let num_segments = if step > 0 {
        (input_signal.len() - noverlap_val) / step
    } else {
        0
    };

    if num_segments < 1 {
        return Err(SignalError::ValueError(
            "Not enough data points for given nperseg and noverlap".to_string(),
        ));
    }

    // Calculate frequency bins
    let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {
        SignalError::ComputationError(format!("Frequency computation error: {}", e))
    })?;

    // Keep only positive frequencies
    let n_half = (nfft_val / 2) + (nfft_val % 2); // Handle both even and odd nfft
    let result_freqs: Vec<f64> = freqs.into_iter().take(n_half).collect();

    // Calculate time segments
    let mut times = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let center = i * step + nperseg_val / 2;
        times.push(center as f64 / fs_val);
    }

    // Initialize STFT output
    let mut stft_output = Vec::with_capacity(n_half);
    for _ in 0..n_half {
        stft_output.push(vec![Complex64::new(0.0, 0.0); num_segments]);
    }

    // Process each segment
    for i in 0..num_segments {
        let start = i * step;
        let end = start + nperseg_val;

        if end > input_signal.len() {
            break;
        }

        // Extract segment
        let segment = input_signal[start..end].to_vec();

        // Detrend the segment
        let detrended = apply_detrend(&segment, detrend_val)?;

        // Apply window
        let windowed: Vec<f64> = detrended
            .iter()
            .zip(win.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // Zero-pad if needed
        let mut padded_segment = windowed.clone();
        if nfft_val > padded_segment.len() {
            padded_segment.resize(nfft_val, 0.0);
        }

        // Compute FFT
        let spectrum = scirs2_fft::fft(&padded_segment, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {}", e)))?;

        // Store positive frequencies in output matrix
        for (j, &val) in spectrum.iter().take(n_half).enumerate() {
            stft_output[j][i] = val;
        }
    }

    // Transpose the output to get the correct orientation (time, frequency)
    let mut transposed_output = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let mut col = Vec::with_capacity(n_half);
        for row in stft_output.iter().take(n_half) {
            col.push(row[i]);
        }
        transposed_output.push(col);
    }

    Ok((result_freqs, times, transposed_output))
}

/// Compute a spectrogram
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
/// * `mode` - Mode ("psd", "complex", "magnitude", "angle", "phase")
///
/// # Returns
///
/// * A tuple containing (frequencies, time segments, spectrogram values)
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn spectrogram<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    detrend: Option<&str>,
    scaling: Option<&str>,
    mode: Option<&str>,
) -> SpectrogramResult
where
    T: Float + NumCast + Debug,
{
    // Default parameters
    let mode_val = mode.unwrap_or("psd");
    let scaling_val = scaling.unwrap_or("density");

    // Compute STFT
    let (freqs, times, stft_values) = stft(
        x,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        Some("zeros"),
        Some(true),
    )?;

    // Extract magnitude or phase based on mode
    let spectrogram_values: Vec<Vec<f64>> = match mode_val {
        "psd" => {
            // Power spectral density
            let fs_val = fs.unwrap_or(1.0);
            let nperseg_val = nperseg.unwrap_or(256.min(x.len()));

            // Create window function for scaling
            let window_val = window.unwrap_or("hann");
            let win = get_window(window_val, nperseg_val)?;
            let win_scale = win.iter().map(|&w| w.powi(2)).sum::<f64>();
            let scale = 1.0 / win_scale;

            stft_values
                .iter()
                .map(|col| {
                    col.iter()
                        .map(|&c| {
                            let psd = c.norm_sqr() * scale / (fs_val * nperseg_val as f64);
                            if scaling_val == "density" {
                                psd
                            } else {
                                // "spectrum" - multiply by sampling frequency
                                psd * fs_val
                            }
                        })
                        .collect()
                })
                .collect()
        }
        "magnitude" => {
            // Magnitude (absolute value)
            stft_values
                .iter()
                .map(|col| col.iter().map(|&c| c.norm()).collect())
                .collect()
        }
        "angle" | "phase" => {
            // Phase angle
            stft_values
                .iter()
                .map(|col| col.iter().map(|&c| c.arg()).collect())
                .collect()
        }
        "complex" => {
            return Err(SignalError::ValueError(
                "Mode 'complex' returns complex values and is not supported for spectrogram"
                    .to_string(),
            ));
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown mode option: {}, expected 'psd', 'magnitude', 'angle', or 'phase'",
                mode_val
            )));
        }
    };

    Ok((freqs, times, spectrogram_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;
    #[test]
    fn test_periodogram_sine_wave() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Generate a sine wave at 10 Hz
        let fs = 100.0;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();
        let f = 10.0; // Hz
        let x: Vec<f64> = t.iter().map(|&t| (2.0 * PI * f * t).sin()).collect();

        // Compute periodogram
        let (freqs, psd) = periodogram(&x, Some(fs), None, None, None, None).unwrap();

        // Find the peak frequency
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, &val) in psd.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Check that the peak is at 10 Hz
        assert_relative_eq!(freqs[max_idx], f, epsilon = 1.0);

        // Check that the PSD has a reasonable length
        assert!(freqs.len() >= (x.len() / 2));
        assert!(freqs.len() <= (x.len() / 2 + 2)); // Allow a bit of flexibility
    }

    #[test]
    fn test_welch_sine_wave() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Generate a sine wave at 10 Hz with noise
        let fs = 100.0;
        let t: Vec<f64> = (0..2000).map(|i| i as f64 / fs).collect();
        let f = 10.0; // Hz
        let mut x: Vec<f64> = t.iter().map(|&t| (2.0 * PI * f * t).sin()).collect();

        // Add noise
        let mut rng = rand::rng();
        x.iter_mut().for_each(|val| {
            *val += rng.gen_range(-0.1..0.1);
        });

        // Compute Welch's periodogram
        let (freqs, psd) =
            welch(&x, Some(fs), None, Some(256), Some(128), None, None, None).unwrap();

        // Find the peak frequency
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, &val) in psd.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Check that the peak is at 10 Hz
        assert_relative_eq!(freqs[max_idx], f, epsilon = 1.0);
    }

    #[test]
    fn test_stft_sine_wave() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Generate a chirp signal (increasing frequency)
        let fs = 1000.0;
        let t: Vec<f64> = (0..2000).map(|i| i as f64 / fs).collect();
        let x: Vec<f64> = t
            .iter()
            .map(|&t| (2.0 * PI * (10.0 + 50.0 * t) * t).sin())
            .collect();

        // Compute STFT
        let (freqs, times, stft_values) = stft(
            &x,
            Some(fs),
            None,
            Some(128),
            Some(64),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Check dimensions
        assert!(!freqs.is_empty());
        assert!(!times.is_empty());
        assert_eq!(stft_values.len(), times.len());
        if !stft_values.is_empty() {
            assert_eq!(stft_values[0].len(), freqs.len());
        }

        // For a chirp, the peak frequency should increase over time
        if times.len() >= 2 {
            // Find peak frequency for first and last time segment
            let first_segment = &stft_values[0];
            let last_segment = &stft_values[times.len() - 1];

            let mut max_idx_first = 0;
            let mut max_val_first = 0.0;
            for (i, &val) in first_segment.iter().enumerate() {
                if val.norm() > max_val_first {
                    max_val_first = val.norm();
                    max_idx_first = i;
                }
            }

            let mut max_idx_last = 0;
            let mut max_val_last = 0.0;
            for (i, &val) in last_segment.iter().enumerate() {
                if val.norm() > max_val_last {
                    max_val_last = val.norm();
                    max_idx_last = i;
                }
            }

            // For a chirp, the peak frequency at the end should be higher
            assert!(freqs[max_idx_last] > freqs[max_idx_first]);
        }
    }

    #[test]
    fn test_spectrogram_modes() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Generate a test signal
        let fs = 100.0;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();
        let x: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 10.0 * t).sin()).collect();

        // Compute spectrograms with different modes
        let (_, _, psd_values) = spectrogram(
            &x,
            Some(fs),
            None,
            Some(128),
            None,
            None,
            None,
            None,
            Some("psd"),
        )
        .unwrap();
        let (_, _, mag_values) = spectrogram(
            &x,
            Some(fs),
            None,
            Some(128),
            None,
            None,
            None,
            None,
            Some("magnitude"),
        )
        .unwrap();
        let (_, _, phase_values) = spectrogram(
            &x,
            Some(fs),
            None,
            Some(128),
            None,
            None,
            None,
            None,
            Some("phase"),
        )
        .unwrap();

        // Check dimensions
        assert!(!psd_values.is_empty());
        assert!(!mag_values.is_empty());
        assert!(!phase_values.is_empty());

        // PSD values should be non-negative
        for row in &psd_values {
            for &val in row {
                assert!(val >= 0.0);
            }
        }

        // Magnitude values should be non-negative
        for row in &mag_values {
            for &val in row {
                assert!(val >= 0.0);
            }
        }

        // Phase values should be between -π and π
        for row in &phase_values {
            for &val in row {
                assert!((-PI..=PI).contains(&val));
            }
        }
    }
}
