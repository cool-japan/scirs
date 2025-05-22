//! Adaptive multitaper spectral estimation.

use super::utils::compute_fft;
use super::windows::dpss;
use crate::error::{SignalError, SignalResult};
use ndarray::Array2;
use num_complex::Complex64;
use num_traits::{Float, NumCast};
// PI is only used in doc examples
use std::fmt::Debug;

/// Compute multitaper spectral density estimate with adaptively weighted spectra.
///
/// This function uses the method of Thomson (1982) to adaptively weight the
/// eigenspectra based on their local bias properties.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers (default = 2*nw-1)
/// * `nfft` - FFT length (default = signal length)
/// * `adaptive` - If true, use adaptive weighting (default = true)
/// * `return_onesided` - If true, return one-sided spectrum
///
/// # Returns
///
/// * Tuple of (frequencies, power spectral density)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::adaptive_psd;
/// use std::f64::consts::PI;
///
/// // Generate a test signal (sinusoid with noise)
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// use rand::Rng;
/// let mut rng = rand::rng();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rng.random_range(0.0..1.0))
///     .collect();
///
/// // Compute adaptive multitaper power spectral density
/// let (freqs, psd) = adaptive_psd(
///     &signal,
///     Some(fs),
///     Some(4.0), // nw
///     None,      // k = 2*nw-1 = 7
///     None,      // nfft = signal length
///     Some(true), // adaptive weighting
///     Some(true), // one-sided spectrum
/// ).unwrap();
///
/// // Basic verification - function should succeed
/// assert!(freqs.len() > 0);
/// assert!(psd.len() > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn adaptive_psd<T>(
    x: &[T],
    fs: Option<f64>,
    nw: Option<f64>,
    k: Option<usize>,
    nfft: Option<usize>,
    adaptive: Option<bool>,
    return_onesided: Option<bool>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
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
    let fs_val = fs.unwrap_or(1.0);
    let nw_val = nw.unwrap_or(4.0);
    let k_val = k.unwrap_or(((2.0 * nw_val) as usize).saturating_sub(1));
    let nfft_val = nfft.unwrap_or(n);
    let adaptive_val = adaptive.unwrap_or(true);
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

    // Apply tapers to the signal
    let mut tapered_signals = Array2::zeros((k_val, n));
    for i in 0..k_val {
        for j in 0..n {
            tapered_signals[[i, j]] = x_f64[j] * tapers[[i, j]];
        }
    }

    // Compute FFT of tapered signals
    let n_freqs = if return_onesided_val {
        nfft_val / 2 + 1
    } else {
        nfft_val
    };
    let mut spectra = Array2::zeros((k_val, n_freqs));

    for i in 0..k_val {
        // Create complex signal for FFT
        let mut signal_complex = vec![Complex64::new(0.0, 0.0); nfft_val];
        for j in 0..n {
            signal_complex[j] = Complex64::new(tapered_signals[[i, j]], 0.0);
        }

        // Compute FFT
        let spectrum = compute_fft(&signal_complex)?;

        // Store eigenspectra (magnitude squared)
        for j in 0..n_freqs {
            spectra[[i, j]] = spectrum[j].norm_sqr();
        }
    }

    // Create frequency array
    let freqs = if return_onesided_val {
        // One-sided spectrum (positive frequencies only)
        let mut f = Vec::with_capacity(n_freqs);
        for i in 0..n_freqs {
            f.push(i as f64 * fs_val / nfft_val as f64);
        }
        f
    } else {
        // Two-sided spectrum
        let mut f = Vec::with_capacity(nfft_val);
        for i in 0..nfft_val {
            if i <= nfft_val / 2 {
                f.push(i as f64 * fs_val / nfft_val as f64);
            } else {
                f.push((i as f64 - nfft_val as f64) * fs_val / nfft_val as f64);
            }
        }
        f
    };

    // Compute adaptively weighted PSD
    let mut psd = vec![0.0; n_freqs];

    if adaptive_val {
        // Adaptive weights (Thomson's method)
        // Initial estimate using eigenvalue weights
        let mut s_initial = vec![0.0; n_freqs];

        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k_val {
                let weight = eigenvalues[i];
                weighted_sum += weight * spectra[[i, j]];
                weight_sum += weight;
            }

            s_initial[j] = weighted_sum / weight_sum;
        }

        // Compute adaptive weights using iterative procedure
        let mut weights = Array2::zeros((k_val, n_freqs));
        let iterations = 3; // Number of iterations for convergence

        for _ in 0..iterations {
            for j in 0..n_freqs {
                // Compute weights
                let mut denominator = 0.0;

                for i in 0..k_val {
                    let numerator = eigenvalues[i];
                    denominator += numerator.powi(2) / (s_initial[j] + 1e-10);
                    weights[[i, j]] = numerator / (s_initial[j] + 1e-10);
                }

                // Normalize weights
                for i in 0..k_val {
                    weights[[i, j]] /= denominator + 1e-10;
                }

                // Update PSD estimate
                let mut weighted_sum = 0.0;
                for i in 0..k_val {
                    weighted_sum += weights[[i, j]] * spectra[[i, j]];
                }

                s_initial[j] = weighted_sum;
            }
        }

        // Final PSD with adaptive weights
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;

            for i in 0..k_val {
                weighted_sum += weights[[i, j]] * spectra[[i, j]];
            }

            // Scale by sampling frequency for correct units
            let scaling = if return_onesided_val && j > 0 && j < n_freqs - 1 {
                2.0 / fs_val // Factor of 2 for one-sided spectrum (except DC and Nyquist)
            } else {
                1.0 / fs_val
            };

            psd[j] = weighted_sum * scaling;
        }
    } else {
        // Non-adaptive (eigenvalue-weighted)
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k_val {
                let weight = eigenvalues[i];
                weighted_sum += weight * spectra[[i, j]];
                weight_sum += weight;
            }

            // Scale by sampling frequency for correct units
            let scaling = if return_onesided_val && j > 0 && j < n_freqs - 1 {
                2.0 / (fs_val * weight_sum) // Factor of 2 for one-sided spectrum
            } else {
                1.0 / (fs_val * weight_sum)
            };

            psd[j] = weighted_sum * scaling;
        }
    }

    Ok((freqs, psd))
}
