// Power spectral density estimation using multitaper methods.

use super::utils::compute_fft;
use super::windows::dpss;
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rand::Rng;
use std::fmt::Debug;

#[allow(unused_imports)]
// PI is only used in doc examples
/// Type alias for multitaper PSD calculation result
pub type MultitaperResult = (Vec<f64>, Vec<f64>, Option<Array2<f64>>, Option<Array1<f64>>);

/// Compute the multitaper power spectral density estimate of a signal.
///
/// This function uses the multitaper method with DPSS (Slepian) tapers to
/// estimate the power spectral density of a signal. The multitaper method
/// provides reduced variance and better frequency resolution compared to
/// conventional methods like Welch's method.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers (default = 2*nw-1)
/// * `nfft` - FFT length (default = signal length)
/// * `return_onesided` - If true, return one-sided spectrum
/// * `return_tapers` - If true, also return the DPSS tapers used
///
/// # Returns
///
/// * If return_tapers is false: Tuple of (frequencies, power spectral density)
/// * If return_tapers is true: Tuple of (frequencies, psd, tapers, eigenvalues)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::pmtm;
///
/// // Generate a test signal (sinusoid with noise)
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// use rand::Rng;
/// let mut rng = rand::rng();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rng.gen_range(0.0..1.0))
///     .collect();
///
/// // Compute multitaper power spectral density
/// let (freqs, psd, tapers, eigenvalues) = pmtm(
///     &signal,
///     Some(fs),
///     Some(4.0), // nw
///     None,      // k = 2*nw-1 = 7
///     None,      // nfft = signal length
///     Some(true),  // one-sided spectrum
///     Some(true)   // return tapers for this example
/// ).unwrap();
///
/// // Basic verification - function should succeed
/// assert!(freqs.len() > 0);
/// assert!(psd.len() > 0);
/// assert!(tapers.is_some());
/// assert!(eigenvalues.is_some());
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn pmtm<T>(
    x: &[T],
    fs: Option<f64>,
    nw: Option<f64>,
    k: Option<usize>,
    nfft: Option<usize>,
    return_onesided: Option<bool>,
    return_tapers: Option<bool>,
) -> SignalResult<MultitaperResult>
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
    let return_onesided_val = return_onesided.unwrap_or(true);
    let return_tapers_val = return_tapers.unwrap_or(false);

    // Compute DPSS _tapers
    let (_tapers, eigenvalues_opt) = dpss(n, nw_val, k_val, true)?;

    // Verify eigenvalues are available
    let eigenvalues = match eigenvalues_opt {
        Some(evals) => evals,
        None => {
            return Err(SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            ))
        }
    };

    // Apply _tapers to the signal
    let mut tapered_signals = Array2::zeros((k_val, n));
    for i in 0..k_val {
        for j in 0..n {
            tapered_signals[[i, j]] = x_f64[j] * tapers[[i, j]];
        }
    }

    // Compute FFT of tapered signals
    let mut spectra = Array2::zeros((k_val, nfft_val));
    for i in 0..k_val {
        // Create complex signal for FFT
        let mut signal_complex = vec![Complex64::new(0.0, 0.0); nfft_val];
        for j in 0..n {
            signal_complex[j] = Complex64::new(tapered_signals[[i, j]], 0.0);
        }

        // Compute FFT
        let spectrum = compute_fft(&signal_complex)?;

        // Store power (magnitude squared)
        for j in 0..nfft_val {
            spectra[[i, j]] = spectrum[j].norm_sqr();
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

    // Combine spectra from different _tapers using eigenvalue weights
    let mut psd = Vec::with_capacity(if return_onesided_val {
        nfft_val / 2 + 1
    } else {
        nfft_val
    });

    if return_onesided_val {
        // One-sided spectrum
        let n_freqs = nfft_val / 2 + 1;

        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k_val {
                let weight = eigenvalues[i];
                weighted_sum += weight * spectra[[i, j]];
                weight_sum += weight;
            }

            // Scale by sampling frequency for correct units
            let scaling = 2.0 / (fs_val * weight_sum); // Factor of 2 for one-sided spectrum
            psd.push(weighted_sum * scaling);
        }
    } else {
        // Two-sided spectrum
        for j in 0..nfft_val {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..k_val {
                let weight = eigenvalues[i];
                weighted_sum += weight * spectra[[i, j]];
                weight_sum += weight;
            }

            // Scale by sampling frequency for correct units
            let scaling = 1.0 / (fs_val * weight_sum);
            psd.push(weighted_sum * scaling);
        }
    }

    // Return results
    if return_tapers_val {
        Ok((freqs, psd, Some(_tapers), Some(eigenvalues)))
    } else {
        Ok((freqs, psd, None, None))
    }
}

/// Compute a multitaper spectrogram (time-frequency representation) of a signal.
///
/// This function computes a spectrogram using the multitaper method with DPSS tapers,
/// which provides reduced variance and better frequency resolution compared to
/// conventional spectrograms using a fixed window.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window_size` - Window size in samples (default = min(256, signal length))
/// * `step` - Step size in samples (default = window_size / 2)
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers (default = 2*nw-1)
/// * `nfft` - FFT length (default = window_size)
/// * `return_onesided` - If true, return one-sided spectrum
/// * `adaptive` - If true, use adaptive weighting (default = true)
///
/// # Returns
///
/// * Tuple of (times, frequencies, spectrogram)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::multitaper_spectrogram;
///
/// // Generate a test signal with changing frequency
/// let n = 2000;
/// let fs = 1000.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// use rand::Rng;
/// let mut rng = rand::rng();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| {
///         let freq = 50.0 + 200.0 * ti; // Linear chirp from 50Hz to 250Hz
///         (2.0 * PI * freq * ti).sin() + 0.1 * rng.gen_range(0.0..1.0)
///     })
///     .collect();
///
/// // Compute multitaper spectrogram
/// let (times, freqs, spec) = multitaper_spectrogram(
///     &signal,
///     Some(fs),
///     Some(256),    // window_size
///     Some(32),     // step
///     Some(4.0),    // nw
///     None,         // k = 2*nw-1 = 7
///     None,         // nfft = window_size
///     Some(true),   // one-sided spectrum
///     Some(true),   // adaptive weighting
/// ).unwrap();
///
/// // The spectrogram should show the frequency increasing over time
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn multitaper_spectrogram<T>(
    x: &[T],
    fs: Option<f64>,
    window_size: Option<usize>,
    step: Option<usize>,
    nw: Option<f64>,
    k: Option<usize>,
    nfft: Option<usize>,
    return_onesided: Option<bool>,
    adaptive: Option<bool>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
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
    let window_size_val = window_size.unwrap_or(usize::min(256, n));
    let step_val = step.unwrap_or(window_size_val / 2);
    let nw_val = nw.unwrap_or(4.0);
    let k_val = k.unwrap_or(((2.0 * nw_val) as usize).saturating_sub(1));
    let nfft_val = nfft.unwrap_or(window_size_val);
    let return_onesided_val = return_onesided.unwrap_or(true);
    let adaptive_val = adaptive.unwrap_or(true);

    // Compute number of segments and initialize spectrogram
    let n_segments = (n - window_size_val) / step_val + 1;
    let n_freqs = if return_onesided_val {
        nfft_val / 2 + 1
    } else {
        nfft_val
    };
    let mut spectrogram = Array2::zeros((n_segments, n_freqs));

    // Create time vector
    let mut times = Vec::with_capacity(n_segments);
    for i in 0..n_segments {
        let center = (i * step_val + window_size_val / 2) as f64 / fs_val;
        times.push(center);
    }

    // Create frequency vector
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

    // Compute DPSS tapers once for the window _size
    let (tapers, eigenvalues) = dpss(window_size_val, nw_val, k_val, true)?;
    // Verify eigenvalues are available
    let eigenvalues = match eigenvalues {
        Some(evals) => evals,
        None => {
            return Err(SignalError::ComputationError(
                "Eigenvalues required but not returned from dpss".to_string(),
            ))
        }
    };

    // Process each segment
    for i in 0..n_segments {
        let start = i * step_val;
        let end = start + window_size_val;

        if end > n {
            // Skip segments that exceed the signal length
            continue;
        }

        // Extract segment
        let segment: Vec<f64> = x_f64[start..end].to_vec();

        // Apply tapers to segment
        let mut tapered_segments = Array2::zeros((k_val, window_size_val));
        for j in 0..k_val {
            for m in 0..window_size_val {
                tapered_segments[[j, m]] = segment[m] * tapers[[j, m]];
            }
        }

        // Compute FFT of tapered segments
        let mut segment_spectra = Array2::zeros((k_val, n_freqs));

        for j in 0..k_val {
            // Create complex signal for FFT
            let mut signal_complex = vec![Complex64::new(0.0, 0.0); nfft_val];
            for m in 0..window_size_val {
                signal_complex[m] = Complex64::new(tapered_segments[[j, m]], 0.0);
            }

            // Compute FFT
            let spectrum = compute_fft(&signal_complex)?;

            // Store eigenspectra (magnitude squared)
            for m in 0..n_freqs {
                segment_spectra[[j, m]] = spectrum[m].norm_sqr();
            }
        }

        // Combine spectra based on adaptive or eigenvalue weighting
        if adaptive_val {
            // Adaptive weights (similar to adaptive_psd)
            // Initial estimate using eigenvalue weights
            let mut s_initial = vec![0.0; n_freqs];

            for m in 0..n_freqs {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for j in 0..k_val {
                    let weight = eigenvalues[j];
                    weighted_sum += weight * segment_spectra[[j, m]];
                    weight_sum += weight;
                }

                s_initial[m] = weighted_sum / weight_sum;
            }

            // Compute adaptive weights using iterative procedure
            let mut weights = Array2::zeros((k_val, n_freqs));
            let iterations = 3; // Number of iterations for convergence

            for _ in 0..iterations {
                for m in 0..n_freqs {
                    // Compute weights
                    let mut denominator = 0.0;

                    for j in 0..k_val {
                        let numerator = eigenvalues[j];
                        denominator += numerator.powi(2) / (s_initial[m] + 1e-10);
                        weights[[j, m]] = numerator / (s_initial[m] + 1e-10);
                    }

                    // Normalize weights
                    for j in 0..k_val {
                        weights[[j, m]] /= denominator + 1e-10;
                    }

                    // Update estimate
                    let mut weighted_sum = 0.0;
                    for j in 0..k_val {
                        weighted_sum += weights[[j, m]] * segment_spectra[[j, m]];
                    }

                    s_initial[m] = weighted_sum;
                }
            }

            // Final spectrogram with adaptive weights
            for m in 0..n_freqs {
                let mut weighted_sum = 0.0;

                for j in 0..k_val {
                    weighted_sum += weights[[j, m]] * segment_spectra[[j, m]];
                }

                // Scale by sampling frequency for correct units
                let scaling = if return_onesided_val && m > 0 && m < n_freqs - 1 {
                    2.0 / fs_val // Factor of 2 for one-sided spectrum (except DC and Nyquist)
                } else {
                    1.0 / fs_val
                };

                spectrogram[[i, m]] = weighted_sum * scaling;
            }
        } else {
            // Non-adaptive (eigenvalue-weighted)
            for m in 0..n_freqs {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for j in 0..k_val {
                    let weight = eigenvalues[j];
                    weighted_sum += weight * segment_spectra[[j, m]];
                    weight_sum += weight;
                }

                // Scale by sampling frequency for correct units
                let scaling = if return_onesided_val && m > 0 && m < n_freqs - 1 {
                    2.0 / (fs_val * weight_sum) // Factor of 2 for one-sided spectrum
                } else {
                    1.0 / (fs_val * weight_sum)
                };

                spectrogram[[i, m]] = weighted_sum * scaling;
            }
        }
    }

    Ok((times, freqs, spectrogram))
}
