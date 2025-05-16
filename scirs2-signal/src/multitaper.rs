//! Multitaper spectral estimation.
//!
//! This module provides functions for spectral analysis using multitaper methods,
//! which provide robust spectral estimates with reduced variance and bias compared
//! to conventional approaches. The implementation includes Discrete Prolate
//! Spheroidal Sequences (DPSS) tapers, also known as Slepian sequences.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array, Array1, Array2, ArrayView1, Axis};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Compute Discrete Prolate Spheroidal Sequences (DPSS), also known as Slepian sequences.
///
/// DPSS tapers are often used in multitaper spectral estimation and are designed
/// to maximize energy concentration in a specified frequency band.
///
/// # Arguments
///
/// * `n` - Length of the tapers
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers to compute (should be less than or equal to 2*nw)
/// * `return_ratios` - If true, also return the eigenvalues
///
/// # Returns
///
/// * If return_ratios is false: Array2 of DPSS tapers (shape: [k, n])
/// * If return_ratios is true: Tuple of (tapers, eigenvalues)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::dpss;
///
/// // Compute 4 DPSS tapers of length 64 with time-bandwidth product of 4
/// let (tapers, eigenvalues) = dpss(64, 4.0, 4, true).unwrap();
///
/// // Check number of tapers
/// assert_eq!(tapers.shape()[0], 4);
/// assert_eq!(tapers.shape()[1], 64);
///
/// // Eigenvalues should be decreasing and close to 1.0 for the first tapers
/// assert!(eigenvalues[0] > 0.9);
/// assert!(eigenvalues[0] > eigenvalues[1]);
/// ```
pub fn dpss(
    n: usize,
    nw: f64,
    k: usize,
    return_ratios: bool,
) -> SignalResult<(Array2<f64>, Option<Array1<f64>>)> {
    if n < 2 {
        return Err(SignalError::ValueError(
            "Length of tapers must be at least 2".to_string(),
        ));
    }

    if nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product must be positive".to_string(),
        ));
    }

    if k < 1 {
        return Err(SignalError::ValueError(
            "Number of tapers must be at least 1".to_string(),
        ));
    }

    // Maximum number of tapers that can be well-concentrated with the given nw
    let max_tapers = (2.0 * nw).floor() as usize;
    
    if k > max_tapers {
        return Err(SignalError::ValueError(format!(
            "Number of tapers k ({}) must not exceed 2*nw ({})",
            k, max_tapers
        )));
    }

    // Construct the tridiagonal matrix
    let w = nw / n as f64; // Half-bandwidth
    let n_points = n;

    // Diagonal elements
    let mut diag = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let i_float = i as f64;
        let n_float = n_points as f64;
        let val = ((n_float - 1.0) / 2.0 - i_float).powi(2) * (2.0 * PI * w).powi(2);
        diag.push(val);
    }

    // Off-diagonal elements
    let mut offdiag = Vec::with_capacity(n_points - 1);
    for i in 0..(n_points - 1) {
        let i_float = i as f64;
        offdiag.push((i_float + 1.0) * (n_points as f64 - i_float - 1.0));
    }

    // Diagonalize the tridiagonal matrix
    let (mut eigvals, mut eigvecs) = 
        tridiagonal_eig(diag.as_slice(), offdiag.as_slice())?;

    // Sort eigenvalues and eigenvectors
    let mut idx: Vec<usize> = (0..n_points).collect();
    idx.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());

    // Reorder eigenvalues and eigenvectors
    let mut sorted_eigvals = Vec::with_capacity(n_points);
    let mut sorted_eigvecs = Array2::zeros((n_points, n_points));

    for i in 0..n_points {
        sorted_eigvals.push(eigvals[idx[i]]);
        for j in 0..n_points {
            sorted_eigvecs[[i, j]] = eigvecs[[idx[i], j]];
        }
    }

    // Take the k most concentrated eigenvectors
    let mut dpss = Array2::zeros((k, n_points));
    let mut lambda = Array1::zeros(k);

    for i in 0..k {
        // Convert concentration from sin squared to sin
        lambda[i] = (1.0 - sorted_eigvals[i]).sqrt();

        // Get eigenvector and normalize
        let mut v = sorted_eigvecs.slice(s![i, ..]).to_owned();
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);

        // Ensure the first element is positive (for consistency)
        if v[0] < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }

        // Copy to output
        for j in 0..n_points {
            dpss[[i, j]] = v[j];
        }
    }

    if return_ratios {
        Ok((dpss, Some(lambda)))
    } else {
        Ok((dpss, None))
    }
}

/// Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// This is a simplified implementation for computing eigenvalues and eigenvectors 
/// of a symmetric tridiagonal matrix, which is needed for the DPSS calculation.
///
/// # Arguments
///
/// * `diag` - Diagonal elements of the matrix
/// * `offdiag` - Off-diagonal elements of the matrix
///
/// # Returns
///
/// * Tuple of (eigenvalues, eigenvectors)
fn tridiagonal_eig(diag: &[f64], offdiag: &[f64]) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    if diag.len() < 1 {
        return Err(SignalError::ValueError(
            "Diagonal must have at least one element".to_string(),
        ));
    }

    if offdiag.len() != diag.len() - 1 {
        return Err(SignalError::ValueError(
            "Off-diagonal must have one fewer element than diagonal".to_string(),
        ));
    }

    let n = diag.len();
    let mut eigvals = vec![0.0; n];
    let mut eigvecs = Array2::zeros((n, n));

    // Initialize with identity matrix
    for i in 0..n {
        eigvecs[[i, i]] = 1.0;
    }

    // Copy diagonal and off-diagonal elements
    let mut a = diag.to_vec();
    let mut b = offdiag.to_vec();

    // Number of iterations for QR algorithm
    let max_iter = 30 * n;
    let mut iter_count = 0;

    // Tolerance for convergence
    let tol = 1e-10;

    // Main QR iteration loop
    while iter_count < max_iter {
        // Find the largest index of a small off-diagonal element
        let mut m = n - 1;
        while m > 0 {
            if b[m - 1].abs() <= tol * (a[m - 1].abs() + a[m].abs()) {
                break;
            }
            m -= 1;
        }

        if m == n - 1 {
            // Last eigenvalue is isolated
            eigvals[n - 1] = a[n - 1];
            n -= 1;
            if n == 0 {
                break;
            }
            a.pop();
            b.pop();
        } else if m == 0 {
            // First eigenvalue is isolated
            eigvals[0] = a[0];
            a.copy_within(1.., 0);
            b.copy_within(1.., 0);
            a.pop();
            b.pop();
            n -= 1;
            
            // Update eigenvectors
            for i in 0..eigvecs.shape()[0] {
                for j in 0..eigvecs.shape()[1] - 1 {
                    eigvecs[[i, j]] = eigvecs[[i, j + 1]];
                }
            }
        } else {
            // Perform QR step on a[m..n, m..n]
            let mut c = 0.0;
            let mut s = 1.0;

            for i in m..n {
                let mut f = s * b[i - 1];
                b[i - 1] *= c;

                if f.abs() <= tol {
                    break;
                }

                let g = a[i];
                let r = (a[i - 1] - g).hypot(2.0 * f) * 0.5;
                let t = if a[i - 1] - g >= 0.0 { 1.0 } else { -1.0 };
                
                a[i - 1] = a[i - 1] + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;
                a[i] = g + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;
                
                c = r / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();
                s = f / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();

                // Update eigenvectors
                for j in 0..eigvecs.shape()[0] {
                    let temp = c * eigvecs[[j, i - 1]] - s * eigvecs[[j, i]];
                    eigvecs[[j, i]] = s * eigvecs[[j, i - 1]] + c * eigvecs[[j, i]];
                    eigvecs[[j, i - 1]] = temp;
                }
            }
        }

        iter_count += 1;
    }

    // If iterations didn't converge, return the best approximation
    if iter_count >= max_iter {
        for i in 0..eigvals.len() {
            if i < a.len() {
                eigvals[i] = a[i];
            }
        }
    } else {
        // Fill in remaining eigenvalues
        for i in 0..a.len() {
            eigvals[i] = a[i];
        }
    }

    Ok((eigvals, eigvecs))
}

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
/// use std::f64::consts::PI;
///
/// // Generate a test signal (sinusoid with noise)
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rand::random::<f64>())
///     .collect();
///
/// // Compute multitaper power spectral density
/// let (freqs, psd) = pmtm(
///     &signal,
///     Some(fs),
///     Some(4.0), // nw
///     None,      // k = 2*nw-1 = 7
///     None,      // nfft = signal length
///     Some(true),  // one-sided spectrum
///     Some(false), // don't return tapers
/// ).unwrap();
///
/// // Peak should be near 10 Hz
/// let max_idx = psd.iter().enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
///     .map(|(idx, _)| idx)
///     .unwrap();
/// assert!((freqs[max_idx] - 10.0).abs() < 0.5); // Within 0.5 Hz
/// ```
#[allow(clippy::too_many_arguments)]
pub fn pmtm<T>(
    x: &[T],
    fs: Option<f64>,
    nw: Option<f64>,
    k: Option<usize>,
    nfft: Option<usize>,
    return_onesided: Option<bool>,
    return_tapers: Option<bool>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Option<Array2<f64>>, Option<Array1<f64>>)>
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

    // Compute DPSS tapers
    let (tapers, eigenvalues) = dpss(n, nw_val, k_val, true)?;

    // Apply tapers to the signal
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

    // Combine spectra from different tapers using eigenvalue weights
    let mut psd = Vec::with_capacity(if return_onesided_val { nfft_val / 2 + 1 } else { nfft_val });
    
    if return_onesided_val {
        // One-sided spectrum
        let n_freqs = nfft_val / 2 + 1;
        
        for j in 0..n_freqs {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for i in 0..k_val {
                let weight = eigenvalues[i] as f64;
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
                let weight = eigenvalues[i] as f64;
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
        Ok((freqs, psd, Some(tapers), Some(eigenvalues)))
    } else {
        Ok((freqs, psd, None, None))
    }
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
fn compute_fft(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
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
        
        for k in 0..n/2 {
            let t = odd_fft[k] * Complex64::new(
                (-2.0 * PI * k as f64 / n as f64).cos(),
                (-2.0 * PI * k as f64 / n as f64).sin()
            );
            
            result[k] = even_fft[k] + t;
            result[k + n/2] = even_fft[k] - t;
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
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

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
/// use std::f64::consts::PI;
///
/// // Generate two related signals
/// let n = 1024;
/// let fs = 100.0;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
/// 
/// // Signal 1: 10 Hz sine
/// let signal1: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rand::random::<f64>())
///     .collect();
///     
/// // Signal 2: 10 Hz cosine (highly coherent with signal1 at 10 Hz)
/// let signal2: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).cos() + 0.1 * rand::random::<f64>())
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
///     .min_by(|(_, a), (_, b)| (a - 10.0).abs().partial_cmp(&(b - 10.0).abs()).unwrap())
///     .map(|(idx, _)| idx)
///     .unwrap();
/// assert!(coh[f10_idx] > 0.8); // High coherence
/// ```
#[allow(clippy::too_many_arguments)]
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
        return Err(SignalError::ValueError("Second signal is empty".to_string()));
    }
    
    if x.len() != y.len() {
        return Err(SignalError::ValueError(format!(
            "Signals must have the same length, got {} and {}",
            x.len(), y.len()
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
    let (tapers, eigenvalues) = dpss(n, nw_val, k_val, true)?;

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
            let weight = eigenvalues[i] as f64;
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
        coherence[j] = coherence[j].max(0.0).min(1.0);
    }

    Ok((freqs, coherence))
}

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
/// let signal: Vec<f64> = t.iter()
///     .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.1 * rand::random::<f64>())
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
/// // Peak should be near 10 Hz
/// let max_idx = psd.iter().enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
///     .map(|(idx, _)| idx)
///     .unwrap();
/// assert!((freqs[max_idx] - 10.0).abs() < 0.5); // Within 0.5 Hz
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
    let (tapers, eigenvalues) = dpss(n, nw_val, k_val, true)?;

    // Apply tapers to the signal
    let mut tapered_signals = Array2::zeros((k_val, n));
    for i in 0..k_val {
        for j in 0..n {
            tapered_signals[[i, j]] = x_f64[j] * tapers[[i, j]];
        }
    }

    // Compute FFT of tapered signals
    let n_freqs = if return_onesided_val { nfft_val / 2 + 1 } else { nfft_val };
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
                let weight = eigenvalues[i] as f64;
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
                    let numerator = eigenvalues[i] as f64;
                    denominator += numerator.powi(2) / (s_initial[j] + 1e-10);
                    weights[[i, j]] = numerator / (s_initial[j] + 1e-10);
                }
                
                // Normalize weights
                for i in 0..k_val {
                    weights[[i, j]] /= (denominator + 1e-10);
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
                let weight = eigenvalues[i] as f64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_dpss() {
        // Test with reasonable parameters
        let n = 64;
        let nw = 4.0;
        let k = 7;
        let (tapers, eigenvalues) = dpss(n, nw, k, true).unwrap();

        // Check dimensions
        assert_eq!(tapers.shape(), &[k, n]);
        assert_eq!(eigenvalues.unwrap().len(), k);

        // Test with invalid parameters
        assert!(dpss(0, 4.0, 7, false).is_err()); // n too small
        assert!(dpss(64, 0.0, 7, false).is_err()); // nw too small
        assert!(dpss(64, 2.0, 5, false).is_err()); // k > 2*nw
    }

    #[test]
    fn test_pmtm() {
        // Generate a test signal (sinusoid)
        let n = 128;
        let fs = 10.0;
        let freq = 2.0; // 2 Hz
        
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / fs;
            signal.push((2.0 * PI * freq * t).sin());
        }

        // Compute multitaper PSD
        let (freqs, psd, _, _) = pmtm(
            &signal,
            Some(fs),
            Some(2.5),
            Some(4),
            None,
            Some(true),
            Some(false),
        ).unwrap();

        // Find peak frequency
        let peak_idx = psd.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Check if peak is at the expected frequency
        assert!((freqs[peak_idx] - freq).abs() < 0.2);
    }

    #[test]
    fn test_coherence() {
        // Generate two test signals with shared component
        let n = 128;
        let fs = 10.0;
        let freq = 2.0; // 2 Hz
        
        let mut signal1 = Vec::with_capacity(n);
        let mut signal2 = Vec::with_capacity(n);
        
        for i in 0..n {
            let t = i as f64 / fs;
            let shared = (2.0 * PI * freq * t).sin();
            let noise1 = 0.1 * ((i % 5) as f64 - 2.0);
            let noise2 = 0.1 * ((i % 7) as f64 - 3.0);
            
            signal1.push(shared + noise1);
            signal2.push(shared + noise2);
        }

        // Compute coherence
        let (freqs, coh) = coherence(
            &signal1,
            &signal2,
            Some(fs),
            Some(2.5),
            Some(4),
            None,
            Some(true),
        ).unwrap();

        // Find coherence at signal frequency
        let f2_idx = freqs.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (a - freq).abs().partial_cmp(&(b - freq).abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Coherence should be high at the shared frequency
        assert!(coh[f2_idx] > 0.7);
    }

    #[test]
    fn test_adaptive_psd() {
        // Generate a test signal (sum of sinusoids)
        let n = 256;
        let fs = 100.0;
        let freq1 = 10.0; // 10 Hz
        let freq2 = 25.0; // 25 Hz
        
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / fs;
            signal.push(
                (2.0 * PI * freq1 * t).sin() + 
                0.5 * (2.0 * PI * freq2 * t).sin() +
                0.1 * ((i % 5) as f64 - 2.0) // Noise
            );
        }

        // Compute adaptive multitaper PSD
        let (freqs, psd) = adaptive_psd(
            &signal,
            Some(fs),
            Some(4.0),
            Some(7),
            None,
            Some(true),
            Some(true),
        ).unwrap();

        // Find peaks in PSD
        let mut peaks = Vec::new();
        for i in 1..(freqs.len() - 1) {
            if psd[i] > psd[i-1] && psd[i] > psd[i+1] && psd[i] > 0.01 * psd.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                peaks.push((freqs[i], psd[i]));
            }
        }

        // Sort peaks by power
        peaks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Check if the top two peaks are at the expected frequencies
        if peaks.len() >= 2 {
            assert!((peaks[0].0 - freq1).abs() < 0.5 || (peaks[0].0 - freq2).abs() < 0.5);
            assert!((peaks[1].0 - freq1).abs() < 0.5 || (peaks[1].0 - freq2).abs() < 0.5);
        }
    }
}