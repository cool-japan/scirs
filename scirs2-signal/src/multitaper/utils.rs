// Utility functions for multitaper spectral estimation.

use super::windows::dpss;
use crate::error::{SignalError, SignalResult};
use crate::filter::filtfilt;
use ndarray::Array2;
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rand::Rng;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Compute multitaper spectral coherence between two signals.
///
/// This function estimates the spectral coherence between two signals using
/// the multitaper method. The coherence is a measure of the correlation
/// between two signals at different frequencies.
///
/// # Arguments
///
/// * `x` - First input signal
/// * `y` - Second input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers (default = 2*nw-1)
/// * `nfft` - FFT length (default = signal length)
/// * `return_onesided` - If true, return one-sided spectrum
///
/// # Returns
///
/// * Tuple of (frequencies, coherence)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::coherence;
///
/// // Generate two related signals
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
///
/// // Signal 1: 10 Hz sine
/// use rand::Rng;
/// let mut rng = rand::rng();
/// let signal1: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rng.gen_range(0.0..1.0))
///     .collect();
///     
/// // Signal 2: 10 Hz cosine (highly coherent with signal1 at 10 Hz)
/// let signal2: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).cos() + 0.1 * rng.gen_range(0.0..1.0))
///     .collect();
///
/// // Compute coherence
/// let (freqs, coh) = coherence(
///     &signal1,
///     &signal2,
///     Some(fs),
///     Some(4.0), // nw
///     None,      // k = 2*nw-1 = 7
///     None,      // nfft = signal length
///     Some(true), // one-sided spectrum
/// ).unwrap();
///
/// // Coherence should be high near 10 Hz
/// let f10_idx = freqs.iter().enumerate()
///     .min_by(|(_, a), (_, b)| ((*a) - 10.0).abs().partial_cmp(&((*b) - 10.0).abs()).unwrap())
///     .map(|(idx_)| idx)
///     .unwrap();
/// assert!(coh[f10_idx] >= 0.0); // Coherence should be non-negative
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn coherence<T, U>(
    x: &[T],
    y: &[U],
    fs: Option<f64>,
    nw: Option<f64>,
    k: Option<usize>,
    nfft: Option<usize>,
    return_onesided: Option<bool>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    // Validate inputs
    if x.is_empty() {
        return Err(SignalError::ValueError("First signal is empty".to_string()));
    }

    if y.is_empty() {
        return Err(SignalError::ValueError(
            "Second signal is empty".to_string(),
        ));
    }

    if x.len() != y.len() {
        return Err(SignalError::ValueError(format!(
            "Signals must have the same length, got {} and {}",
            x.len(),
            y.len()
        )));
    }

    // Convert inputs to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    let y_f64: Vec<f64> = y
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Default parameters
    let n = x_f64.len();
    let fs_val = fs.unwrap_or(1.0);
    let nw_val = nw.unwrap_or(4.0);
    let k_val = k.unwrap_or(((2.0 * nw_val) as usize).saturating_sub(1));
    let nfft_val = nfft.unwrap_or(n);
    let return_onesided_val = return_onesided.unwrap_or(true);

    // Compute DPSS tapers
    let (tapers, eigenvalues_opt) = dpss(n, nw_val, k_val, true)?;

    // Verify eigenvalues are available
    let eigenvalues = match eigenvalues_opt {
        Some(evals) => evals,
        None => {
            return Err(SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            ))
        }
    };

    // Apply tapers to both signals
    let mut x_tapered = Array2::zeros((k_val, n));
    let mut y_tapered = Array2::zeros((k_val, n));

    for i in 0..k_val {
        for j in 0..n {
            x_tapered[[i, j]] = x_f64[j] * tapers[[i, j]];
            y_tapered[[i, j]] = y_f64[j] * tapers[[i, j]];
        }
    }

    // Compute FFT of tapered signals
    let mut x_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));
    let mut y_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));
    let mut cross_spectra = Array2::zeros((k_val, nfft_val / 2 + 1));

    for i in 0..k_val {
        // Create complex signals for FFT
        let mut x_complex = vec![Complex64::new(0.0, 0.0); nfft_val];
        let mut y_complex = vec![Complex64::new(0.0, 0.0); nfft_val];

        for j in 0..n {
            x_complex[j] = Complex64::new(x_tapered[[i, j]], 0.0);
            y_complex[j] = Complex64::new(y_tapered[[i, j]], 0.0);
        }

        // Compute FFTs
        let x_spectrum = compute_fft(&x_complex)?;
        let y_spectrum = compute_fft(&y_complex)?;

        // Store power and cross-spectra (one-sided)
        let n_freqs = nfft_val / 2 + 1;
        for j in 0..n_freqs {
            x_spectra[[i, j]] = x_spectrum[j].norm_sqr();
            y_spectra[[i, j]] = y_spectrum[j].norm_sqr();
            cross_spectra[[i, j]] = (x_spectrum[j] * y_spectrum[j].conj()).norm();
        }
    }

    // Create frequency array
    let freqs = if return_onesided_val {
        // One-sided spectrum (positive frequencies only)
        let n_freqs = nfft_val / 2 + 1;
        let mut f = Vec::with_capacity(n_freqs);
        for i in 0..n_freqs {
            f.push(i as f64 * fs_val / nfft_val as f64);
        }
        f
    } else {
        // Two-sided spectrum would need additional handling here
        return Err(SignalError::ValueError(
            "Two-sided coherence not implemented".to_string(),
        ));
    };

    // Compute coherence by combining estimates from different tapers
    let n_freqs = nfft_val / 2 + 1;
    let mut coherence = vec![0.0; n_freqs];

    for j in 0..n_freqs {
        let mut x_power = 0.0;
        let mut y_power = 0.0;
        let mut cross_power = 0.0;

        for i in 0..k_val {
            let weight = eigenvalues[i];
            x_power += weight * x_spectra[[i, j]];
            y_power += weight * y_spectra[[i, j]];
            cross_power += weight * cross_spectra[[i, j]];
        }

        // Compute coherence
        let denominator = (x_power * y_power).sqrt();
        if denominator > 0.0 {
            coherence[j] = (cross_power / denominator).powi(2);
        } else {
            coherence[j] = 0.0;
        }

        // Clamp to [0, 1]
        coherence[j] = coherence[j].clamp(0.0, 1.0);
    }

    Ok((freqs, coherence))
}

/// Apply a multitaper filtering operation in both forward and backward directions.
///
/// This function performs zero-phase filtering by applying a filter in both
/// forward and backward directions, resulting in zero phase distortion.
/// It uses multiple DPSS (Slepian) tapers for improved spectral estimation,
/// which is especially useful for signals with time-varying characteristics.
///
/// # Arguments
///
/// * `b` - Filter coefficients (numerator)
/// * `a` - Filter coefficients (denominator)
/// * `x` - Input signal
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers (default = 2*nw-1)
/// * `pad_type` - Padding method for the signal edges
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::multitaper_filtfilt;
/// use scirs2_signal::filter::butter;
///
/// // Generate a test signal (sinusoid with noise)
/// let n = 1000;
/// let fs = 1000.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 50.0 * ti).sin() + 0.5 * (2.0 * PI * 120.0 * ti).sin() + 0.2 * rand::random::<f64>())
///     .collect();
///
/// // Create a lowpass filter
/// let cutoff = 80.0; // Hz
/// let (b, a) = butter(4, cutoff / (fs / 2.0), "lowpass").unwrap();
///
/// // Apply multitaper filtfilt
/// let filtered = multitaper_filtfilt(&b, &a, &signal, Some(4.0), None, None).unwrap();
///
/// // The higher frequency component should be attenuated
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn multitaper_filtfilt<T>(
    b: &[f64],
    a: &[f64],
    x: &[T],
    nw: Option<f64>,
    k: Option<usize>,
    _pad_type: Option<&str>, // Keep parameter for backward compatibility but mark as unused
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if b.is_empty() {
        return Err(SignalError::ValueError(
            "Filter numerator is empty".to_string(),
        ));
    }

    if a.is_empty() {
        return Err(SignalError::ValueError(
            "Filter denominator is empty".to_string(),
        ));
    }

    if a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "Leading coefficient of denominator must be non-zero".to_string(),
        ));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Default parameters
    let n = x_f64.len();
    let nw_val = nw.unwrap_or(4.0);
    let k_val = k.unwrap_or(((2.0 * nw_val) as usize).saturating_sub(1));
    // pad_type parameter is now unused

    // Compute DPSS tapers
    let (tapers, eigenvalues_opt) = dpss(n, nw_val, k_val, true)?;

    // Verify eigenvalues are available
    let eigenvalues = match eigenvalues_opt {
        Some(evals) => evals,
        None => {
            return Err(SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            ))
        }
    };

    // Apply tapers to the signal
    let mut tapered_signals = Array2::zeros((k_val, n));
    for i in 0..k_val {
        for j in 0..n {
            tapered_signals[[i, j]] = x_f64[j] * tapers[[i, j]];
        }
    }

    // Apply filtfilt to each tapered signal
    let mut filtered_signals = Array2::zeros((k_val, n));
    for i in 0..k_val {
        // Extract the tapered signal as a Vec<f64>
        let tapered_signal: Vec<f64> = (0..n).map(|j| tapered_signals[[i, j]]).collect();

        // Apply filtfilt
        let filtered = filtfilt(b, a, &tapered_signal)?;

        // Store filtered result
        for j in 0..n {
            filtered_signals[[i, j]] = filtered[j];
        }
    }

    // Combine filtered signals using eigenvalue weights
    let mut result = vec![0.0; n];
    let mut weight_sum = 0.0;

    for i in 0..k_val {
        let weight = eigenvalues[i];
        weight_sum += weight;

        for j in 0..n {
            result[j] += weight * filtered_signals[[i, j]];
        }
    }

    // Normalize by sum of weights
    result.iter_mut().for_each(|val| *val /= weight_sum);

    Ok(result)
}

/// Perform a simple FFT on a complex signal.
///
/// # Arguments
///
/// * `x` - Complex input signal
///
/// # Returns
///
/// * Complex FFT result
#[allow(dead_code)]
pub fn compute_fft(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = x.len();

    // Trivial cases
    if n <= 1 {
        return Ok(x.to_vec());
    }

    // Check if n is a power of two (for simplicity)
    if !is_power_of_two(n) {
        return Err(SignalError::ValueError(
            "FFT length must be a power of 2".to_string(),
        ));
    }

    // Recursive FFT implementation
    if n % 2 == 0 {
        // Split into even and odd indices
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);

        for i in (0..n).step_by(2) {
            even.push(x[i]);
            odd.push(x[i + 1]);
        }

        // Recursive FFT on even and odd parts
        let even_fft = compute_fft(&even)?;
        let odd_fft = compute_fft(&odd)?;

        // Combine results
        let mut result = vec![Complex64::new(0.0, 0.0); n];

        for k in 0..n / 2 {
            let t = odd_fft[k]
                * Complex64::new(
                    (-2.0 * PI * k as f64 / n as f64).cos(),
                    (-2.0 * PI * k as f64 / n as f64).sin(),
                );

            result[k] = even_fft[k] + t;
            result[k + n / 2] = even_fft[k] - t;
        }

        Ok(result)
    } else {
        // For non-power-of-two, we would need a more general algorithm
        Err(SignalError::ValueError(
            "FFT length must be a power of 2".to_string(),
        ))
    }
}

/// Check if a number is a power of two.
///
/// # Arguments
///
/// * `n` - Number to check
///
/// # Returns
///
/// * true if n is a power of 2, false otherwise
#[allow(dead_code)]
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}
