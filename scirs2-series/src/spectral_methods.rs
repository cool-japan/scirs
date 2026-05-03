//! Spectral methods for time series analysis
//!
//! This module provides comprehensive spectral analysis tools including:
//! - Power Spectral Density (PSD) estimation
//! - Multi-Taper Method (MTM) spectral estimation with DPSS tapers
//! - Spectral coherence between two series
//! - Cross Spectral Density (CSD)
//! - Wigner-Ville Distribution (WVD) for non-stationary signals
//! - Stationarity testing via local spectral variance

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_fft::{rfft, rfftfreq};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hann window coefficients of length `n`
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Apply a real window and compute the RFFT, returning `Vec<Complex64>`.
fn windowed_rfft(segment: &[f64], window: &[f64]) -> Result<Vec<Complex64>> {
    let n = segment.len();
    if window.len() != n {
        return Err(TimeSeriesError::InvalidInput(
            "window length must match segment length".to_string(),
        ));
    }
    let windowed: Vec<f64> = segment.iter().zip(window).map(|(s, w)| s * w).collect();
    rfft(&windowed, None).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))
}

/// Window power (sum of squares) for normalisation
fn window_power(window: &[f64]) -> f64 {
    window.iter().map(|w| w * w).sum::<f64>()
}

// ---------------------------------------------------------------------------
// Power Spectral Density (periodogram)
// ---------------------------------------------------------------------------

/// Compute the one-sided power spectral density using a Hann-windowed periodogram.
///
/// # Arguments
/// * `ts` - Time series samples
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
/// `(freqs, power)` where both vectors have length `n/2 + 1`.
///
/// # Errors
/// Returns an error when the time series is empty or has fewer than 4 samples.
pub fn power_spectrum(ts: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "power_spectrum requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }

    let window = hann_window(n);
    let wp = window_power(&window);
    let spectrum = windowed_rfft(ts, &window)?;

    let freqs =
        rfftfreq(n, 1.0 / fs).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    // One-sided PSD: scale by 2 for all bins except DC and Nyquist
    let n_rfft = spectrum.len();
    let scale = 1.0 / (fs * wp);
    let power: Vec<f64> = spectrum
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let p = (c.re * c.re + c.im * c.im) * scale;
            // Double-count bins that are mirrored (all except DC and Nyquist)
            if k == 0 || (n % 2 == 0 && k == n_rfft - 1) {
                p
            } else {
                2.0 * p
            }
        })
        .collect();

    Ok((freqs, power))
}

// ---------------------------------------------------------------------------
// DPSS Tapers (Discrete Prolate Spheroidal Sequences)
// ---------------------------------------------------------------------------

/// Compute Discrete Prolate Spheroidal Sequences (DPSS / Slepian tapers).
///
/// Uses the tridiagonal eigen-problem formulation (Percival & Walden 1993, §8.2).
///
/// # Arguments
/// * `n`             - Sequence length
/// * `half_bandwidth` - Time-half-bandwidth product *W* (e.g. 4.0 means NW=4)
/// * `n_tapers`      - Number of tapers to return (≤ 2*half_bandwidth - 1 is concentrating)
///
/// # Returns
/// `Array2<f64>` of shape `(n_tapers, n)` where each row is a normalised taper.
pub fn dpss_tapers(n: usize, half_bandwidth: f64, n_tapers: usize) -> Result<Array2<f64>> {
    if n < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "n must be >= 2 for DPSS computation".to_string(),
        ));
    }
    if half_bandwidth <= 0.0 || half_bandwidth >= n as f64 / 2.0 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "half_bandwidth must be in (0, {}) for n={}",
            n as f64 / 2.0,
            n
        )));
    }
    if n_tapers == 0 || n_tapers > n {
        return Err(TimeSeriesError::InvalidInput(
            "n_tapers must be in [1, n]".to_string(),
        ));
    }

    let w = half_bandwidth / n as f64; // normalised half-bandwidth

    // Build tridiagonal matrix entries
    // Main diagonal: ((n-1)/2 - i)^2 * cos(2πw),  i = 0..n-1
    // Off-diagonal: i*(n-i)/2,                     i = 1..n-1
    let diag: Vec<f64> = (0..n)
        .map(|i| {
            let k = (n as f64 - 1.0) / 2.0 - i as f64;
            k * k * (2.0 * PI * w).cos()
        })
        .collect();

    let off: Vec<f64> = (1..n).map(|i| (i as f64 * (n - i) as f64) / 2.0).collect();

    // Symmetric QR iteration to find the top `n_tapers` eigenvectors.
    // We use the power-iteration / Lanczos approach on the tridiagonal matrix
    // (good enough for moderate n; full QL algorithm for production would be better).
    // Here we implement a straightforward symmetric tridiagonal eigen-solver.
    let eigenvecs = tridiag_eigenvecs(&diag, &off, n_tapers)?;

    // eigenvecs: Vec<Vec<f64>>, each inner vec is one eigenvector of length n
    // Arrange into Array2 (n_tapers × n) and normalise rows
    let mut result = Array2::<f64>::zeros((n_tapers, n));
    for (k, evec) in eigenvecs.iter().enumerate() {
        // Ensure positive first lobe convention
        let sign = if evec[n / 2] >= 0.0 { 1.0 } else { -1.0 };
        let norm = evec
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::EPSILON);
        for j in 0..n {
            result[[k, j]] = sign * evec[j] / norm;
        }
    }

    Ok(result)
}

/// Power iteration with deflation to find the `k_want` eigenvectors of a
/// symmetric tridiagonal matrix corresponding to the largest eigenvalues.
///
/// Returns `k_want` eigenvectors in descending eigenvalue order.
fn tridiag_eigenvecs(d: &[f64], e: &[f64], k_want: usize) -> Result<Vec<Vec<f64>>> {
    let n = d.len();

    // Gershgorin upper bound on the spectral radius: used as a positive shift
    // so that (A + shift·I) has all positive eigenvalues, making the plain
    // power method converge to the eigenvector with the *largest* eigenvalue
    // of A (which is what we want for the highest-concentration DPSS tapers).
    let shift = d
        .iter()
        .enumerate()
        .map(|(i, &di)| {
            let ri = if i == 0 {
                e[0].abs()
            } else if i == n - 1 {
                e[n - 2].abs()
            } else {
                e[i - 1].abs() + e[i].abs()
            };
            di.abs() + ri
        })
        .fold(0.0_f64, f64::max)
        + 1.0;

    let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(k_want);

    for k in 0..k_want {
        // Seed: alternating-sign cosine spread, then orthogonalise against
        // previously found tapers so deflation forces convergence to the
        // next-largest eigenvector.
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                sign / (n as f64).sqrt()
            })
            .collect();
        orthogonalise_and_normalise(&mut v, &eigenvecs);

        const MAX_ITER: usize = 500;
        const TOL: f64 = 1e-13;

        for _ in 0..MAX_ITER {
            // tv = (A + shift·I)·v
            let mut tv = tridiag_matvec(d, e, &v);
            for (tvi, &vi) in tv.iter_mut().zip(v.iter()) {
                *tvi += shift * vi;
            }

            // Gram-Schmidt deflation
            orthogonalise_and_normalise(&mut tv, &eigenvecs);

            let norm = l2_norm(&tv);
            if norm < f64::EPSILON {
                break;
            }
            for val in &mut tv {
                *val /= norm;
            }

            // Convergence check (allow sign flip)
            let diff_pos: f64 = v.iter().zip(tv.iter()).map(|(&a, &b)| (a - b).abs()).sum();
            let diff_neg: f64 = v.iter().zip(tv.iter()).map(|(&a, &b)| (a + b).abs()).sum();
            let diff = diff_pos.min(diff_neg);

            v = tv;

            if diff < TOL {
                break;
            }
        }

        // Ensure the conventional positive-lobe sign (first large coefficient > 0)
        let first_large = v.iter().find(|&&x| x.abs() > 1e-10).copied().unwrap_or(0.0);
        if first_large < 0.0 {
            for val in &mut v {
                *val = -*val;
            }
        }

        eigenvecs.push(v);
    }

    Ok(eigenvecs)
}

/// Tridiagonal matrix–vector product  A*x  where A has diagonal `d` and sub-diagonal `e`.
fn tridiag_matvec(d: &[f64], e: &[f64], x: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        y[i] = d[i] * x[i];
        if i > 0 {
            y[i] += e[i - 1] * x[i - 1];
        }
        if i < n - 1 {
            y[i] += e[i] * x[i + 1];
        }
    }
    y
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Orthogonalise `v` against all vectors in `basis` using modified Gram-Schmidt,
/// then normalise.
fn orthogonalise_and_normalise(v: &mut Vec<f64>, basis: &[Vec<f64>]) {
    for b in basis {
        let dot = v.iter().zip(b.iter()).map(|(vi, bi)| vi * bi).sum::<f64>();
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi -= dot * bi;
        }
    }
    let norm = l2_norm(v);
    if norm > f64::EPSILON {
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Taper Spectrum (MTM)
// ---------------------------------------------------------------------------

/// Estimate the power spectral density using the Multi-Taper Method (MTM).
///
/// Uses DPSS (Slepian) tapers to reduce spectral leakage and variance.
///
/// # Arguments
/// * `ts`             - Time series samples
/// * `fs`             - Sampling frequency in Hz
/// * `n_tapers`       - Number of Slepian tapers (typically `2*half_bandwidth - 1`)
/// * `half_bandwidth` - Time-half-bandwidth product (e.g. 4.0)
///
/// # Returns
/// `(freqs, power)` where both vectors have length `n/2 + 1`.
pub fn multitaper_spectrum(
    ts: &[f64],
    fs: f64,
    n_tapers: usize,
    half_bandwidth: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "multitaper_spectrum requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }
    let n_tapers_actual = n_tapers.max(1);
    let tapers = dpss_tapers(n, half_bandwidth, n_tapers_actual)?;

    let freqs =
        rfftfreq(n, 1.0 / fs).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;
    let n_rfft = freqs.len();

    // Accumulate eigen-spectra
    let mut psd_sum = vec![0.0_f64; n_rfft];
    for k in 0..n_tapers_actual {
        let taper: Vec<f64> = (0..n).map(|j| tapers[[k, j]]).collect();
        let spectrum = windowed_rfft(ts, &taper)?;
        let scale = 1.0 / fs; // tapers are already normalised to unit power
        for (q, c) in spectrum.iter().enumerate() {
            let p = (c.re * c.re + c.im * c.im) * scale;
            // Two-sided → one-sided correction
            if q == 0 || (n % 2 == 0 && q == n_rfft - 1) {
                psd_sum[q] += p;
            } else {
                psd_sum[q] += 2.0 * p;
            }
        }
    }

    let n_t = n_tapers_actual as f64;
    let power: Vec<f64> = psd_sum.iter().map(|&p| p / n_t).collect();

    Ok((freqs, power))
}

// ---------------------------------------------------------------------------
// Spectral Coherence
// ---------------------------------------------------------------------------

/// Compute the magnitude-squared coherence between two time series.
///
/// The coherence is estimated using Welch's overlapping-segment averaging method.
///
/// # Arguments
/// * `x`     - First time series
/// * `y`     - Second time series (must have the same length as `x`)
/// * `fs`    - Sampling frequency in Hz
/// * `n_fft` - FFT window length (set to 0 to use the full length)
///
/// # Returns
/// `(freqs, coherence)` where coherence values are in `[0, 1]`.
pub fn spectral_coherence(
    x: &[f64],
    y: &[f64],
    fs: f64,
    n_fft: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.len() != y.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }
    let n = x.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "spectral_coherence requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }

    let seg_len = if n_fft == 0 { n } else { n_fft.min(n) };
    let step = seg_len / 2;
    let window = hann_window(seg_len);
    let n_rfft = seg_len / 2 + 1;

    let mut pxx = vec![0.0_f64; n_rfft];
    let mut pyy = vec![0.0_f64; n_rfft];
    // Cross-spectrum: real and imaginary parts accumulated separately
    let mut pxy_re = vec![0.0_f64; n_rfft];
    let mut pxy_im = vec![0.0_f64; n_rfft];
    let mut n_segs = 0usize;

    let mut start = 0;
    while start + seg_len <= n {
        let xs = &x[start..start + seg_len];
        let ys = &y[start..start + seg_len];

        let sx = windowed_rfft(xs, &window)?;
        let sy = windowed_rfft(ys, &window)?;

        for k in 0..n_rfft {
            pxx[k] += sx[k].re * sx[k].re + sx[k].im * sx[k].im;
            pyy[k] += sy[k].re * sy[k].re + sy[k].im * sy[k].im;
            // Sxy = sx* · sy  (conjugate of sx times sy)
            pxy_re[k] += sx[k].re * sy[k].re + sx[k].im * sy[k].im;
            pxy_im[k] += sx[k].re * sy[k].im - sx[k].im * sy[k].re;
        }
        n_segs += 1;
        if step == 0 {
            break;
        }
        start += step;
    }
    if n_segs == 0 {
        return Err(TimeSeriesError::ComputationError(
            "no complete segments found for coherence computation".to_string(),
        ));
    }

    let freqs = rfftfreq(seg_len, 1.0 / fs)
        .map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    let coherence: Vec<f64> = (0..n_rfft)
        .map(|k| {
            let cross_mag2 = pxy_re[k] * pxy_re[k] + pxy_im[k] * pxy_im[k];
            let denom = pxx[k] * pyy[k];
            if denom > f64::EPSILON {
                (cross_mag2 / denom).min(1.0)
            } else {
                0.0
            }
        })
        .collect();

    Ok((freqs, coherence))
}

// ---------------------------------------------------------------------------
// Cross Spectral Density
// ---------------------------------------------------------------------------

/// Compute the cross-spectral density between two time series.
///
/// Uses a single Hann-windowed FFT over the entire signal.
///
/// # Arguments
/// * `x`  - First time series
/// * `y`  - Second time series (must have the same length as `x`)
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
/// `(freqs, Sxy)` where `Sxy` are complex cross-spectral density values.
pub fn cross_spectral_density(x: &[f64], y: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<Complex64>)> {
    if x.len() != y.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }
    let n = x.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "cross_spectral_density requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }

    let window = hann_window(n);
    let wp = window_power(&window);
    let sx = windowed_rfft(x, &window)?;
    let sy = windowed_rfft(y, &window)?;

    let freqs =
        rfftfreq(n, 1.0 / fs).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    let scale = 1.0 / (fs * wp);
    let n_rfft = sx.len();

    let sxy: Vec<Complex64> = (0..n_rfft)
        .map(|k| {
            // Sxy = (sx^* · sy) * scale
            let re = (sx[k].re * sy[k].re + sx[k].im * sy[k].im) * scale;
            let im = (sx[k].re * sy[k].im - sx[k].im * sy[k].re) * scale;
            // Apply one-sided factor
            if k == 0 || (n % 2 == 0 && k == n_rfft - 1) {
                Complex64::new(re, im)
            } else {
                Complex64::new(2.0 * re, 2.0 * im)
            }
        })
        .collect();

    Ok((freqs, sxy))
}

// ---------------------------------------------------------------------------
// Wigner-Ville Distribution
// ---------------------------------------------------------------------------

/// Compute the Wigner-Ville Distribution (WVD) of a real signal.
///
/// The WVD is defined as:
/// `W(t, f) = ∫ x(t + τ/2) x*(t − τ/2) e^{-j2πfτ} dτ`
///
/// The discrete implementation uses the analytic signal (via Hilbert transform
/// approximation) and the cross-WVD of the signal with its conjugate.
///
/// # Arguments
/// * `x`  - Input signal
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
/// `Array2<f64>` of shape `(n, n)` where rows are time bins and columns
/// are frequency bins from `−fs/2` to `fs/2`.
pub fn wigner_ville_distribution(x: &[f64], fs: f64) -> Result<Array2<f64>> {
    let n = x.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "wigner_ville_distribution requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }

    // Step 1: compute the analytic signal z = x + j·H{x}
    // We approximate the Hilbert transform via FFT: zero negative frequencies.
    let analytic = analytic_signal(x)?;

    // Step 2: for each time t, compute the instantaneous correlation kernel
    //         r(t, τ) = z(t + τ) · conj(z(t − τ))   for τ in −(n-1)/2..+(n-1)/2
    // and take its DFT over τ to get W(t, f).
    let mut wvd = Array2::<f64>::zeros((n, n));
    for t in 0..n {
        // Build the kernel vector (length n) at time t
        let mut kernel = vec![Complex64::new(0.0, 0.0); n];
        for tau in 0..n {
            let t_plus = (t + tau) % n;
            let t_minus = (t + n - tau) % n;
            let z_plus = analytic[t_plus];
            let z_minus = analytic[t_minus];
            // conj(z_minus)
            let z_minus_conj = Complex64::new(z_minus.re, -z_minus.im);
            kernel[tau] = Complex64::new(
                z_plus.re * z_minus_conj.re - z_plus.im * z_minus_conj.im,
                z_plus.re * z_minus_conj.im + z_plus.im * z_minus_conj.re,
            );
        }

        // Step 3: FFT of kernel → W(t, f)
        let spectrum = rfft(&kernel.iter().map(|c| c.re).collect::<Vec<_>>(), None)
            .map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

        // Fill row t with real parts of the spectrum
        let n_rfft = spectrum.len();
        for f in 0..n_rfft.min(n) {
            wvd[[t, f]] = spectrum[f].re;
        }
    }

    Ok(wvd)
}

/// Compute the analytic signal of a real sequence using the FFT-based Hilbert transform.
fn analytic_signal(x: &[f64]) -> Result<Vec<Complex64>> {
    let n = x.len();
    // Forward FFT
    let spectrum = rfft(x, None).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    // Build full spectrum (length n) with negative-frequency zeroed out
    // rfft returns n/2+1 positive-frequency components.
    let n_rfft = n / 2 + 1;
    let mut full = vec![Complex64::new(0.0, 0.0); n];

    // DC component (k=0)
    if !spectrum.is_empty() {
        full[0] = Complex64::new(spectrum[0].re, spectrum[0].im);
    }
    // Positive frequencies: multiply by 2 (except Nyquist for even n)
    for k in 1..n_rfft {
        if n % 2 == 0 && k == n_rfft - 1 {
            // Nyquist: keep as is
            if k < n {
                full[k] = Complex64::new(spectrum[k].re, spectrum[k].im);
            }
        } else {
            if k < n {
                full[k] = Complex64::new(2.0 * spectrum[k].re, 2.0 * spectrum[k].im);
            }
        }
        // Negative frequencies (k > n/2): remain zero
    }

    // IFFT of full (we do a simple DFT to avoid depending on complex ifft interface)
    let analytic = idft_complex(&full);
    Ok(analytic)
}

/// Naive IDFT for a complex sequence (used only inside WVD; small n is acceptable).
fn idft_complex(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    let scale = 1.0 / n as f64;
    for k in 0..n {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for j in 0..n {
            let angle = 2.0 * PI * j as f64 * k as f64 / n as f64;
            re += x[j].re * angle.cos() + x[j].im * angle.sin();
            im += -x[j].re * angle.sin() + x[j].im * angle.cos();
        }
        y[k] = Complex64::new(re * scale, im * scale);
    }
    y
}

// ---------------------------------------------------------------------------
// Non-stationarity Test
// ---------------------------------------------------------------------------

/// Test for non-stationarity by comparing local power spectra across windows.
///
/// The test statistic is the coefficient of variation (std/mean) of the total
/// spectral power in successive non-overlapping windows. A higher value indicates
/// greater non-stationarity.
///
/// # Arguments
/// * `ts`     - Time series
/// * `window` - Window length in samples (must be ≥ 4)
/// * `step`   - Step size between windows (must be ≥ 1)
///
/// # Returns
/// Non-stationarity coefficient of variation (dimensionless, ≥ 0).
/// Returns 0.0 if the series is perfectly stationary.
pub fn nonstationarity_test(ts: &[f64], window: usize, step: usize) -> Result<f64> {
    let n = ts.len();
    if window < 4 {
        return Err(TimeSeriesError::InvalidInput(
            "window must be >= 4 for nonstationarity_test".to_string(),
        ));
    }
    if step == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "step must be >= 1 for nonstationarity_test".to_string(),
        ));
    }
    if n < window {
        return Err(TimeSeriesError::InsufficientData {
            message: "time series shorter than window".to_string(),
            required: window,
            actual: n,
        });
    }

    let win = hann_window(window);
    let mut powers: Vec<f64> = Vec::new();
    let mut start = 0;
    while start + window <= n {
        let seg = &ts[start..start + window];
        let spectrum = windowed_rfft(seg, &win)?;
        let total_power: f64 = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        powers.push(total_power);
        start += step;
    }

    if powers.len() < 2 {
        return Ok(0.0);
    }

    let mean = powers.iter().sum::<f64>() / powers.len() as f64;
    if mean < f64::EPSILON {
        return Ok(0.0);
    }
    let variance =
        powers.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / (powers.len() - 1) as f64;
    let cv = variance.sqrt() / mean;
    Ok(cv)
}

// ---------------------------------------------------------------------------
// Re-exports for convenience
// ---------------------------------------------------------------------------

/// Frequency array for an RFFT of length `n` with sampling frequency `fs`.
pub fn rfft_frequencies(n: usize, fs: f64) -> Result<Vec<f64>> {
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }
    rfftfreq(n, 1.0 / fs).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a pure sinusoid at frequency `freq` Hz sampled at `fs` Hz.
    fn make_sine(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_power_spectrum_peak_at_correct_freq() {
        let fs = 100.0;
        let freq = 10.0;
        let n = 256;
        let ts = make_sine(n, freq, fs);
        let (freqs, power) = power_spectrum(&ts, fs).expect("power_spectrum failed");
        assert_eq!(freqs.len(), power.len());

        // Find dominant frequency
        let max_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("empty spectrum");
        let dominant = freqs[max_idx];
        assert!(
            (dominant - freq).abs() < 1.5,
            "dominant freq {dominant} should be near {freq}"
        );
    }

    #[test]
    fn test_dpss_tapers_shape_and_orthonormality() {
        let n = 64;
        let nw = 4.0;
        let k = 7;
        let tapers = dpss_tapers(n, nw, k).expect("dpss_tapers failed");
        assert_eq!(tapers.dim(), (k, n));

        // Each taper should have unit norm
        for ki in 0..k {
            let norm: f64 = (0..n).map(|j| tapers[[ki, j]].powi(2)).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "taper {ki} not normalised: {norm}"
            );
        }

        // Consecutive tapers should be approximately orthogonal
        for ki in 0..k - 1 {
            let dot: f64 = (0..n).map(|j| tapers[[ki, j]] * tapers[[ki + 1, j]]).sum();
            assert!(
                dot.abs() < 1e-8,
                "tapers {ki} and {} not orthogonal: dot={dot}",
                ki + 1
            );
        }
    }

    #[test]
    fn test_multitaper_spectrum_peak() {
        let fs = 100.0;
        let freq = 15.0;
        let n = 256;
        let ts = make_sine(n, freq, fs);
        let (freqs, power) = multitaper_spectrum(&ts, fs, 7, 4.0).expect("MTM failed");

        let max_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("empty MTM spectrum");
        assert!(
            (freqs[max_idx] - freq).abs() < 2.0,
            "MTM peak {} should be near {}",
            freqs[max_idx],
            freq
        );
    }

    #[test]
    fn test_spectral_coherence_identical_signals() {
        let n = 128;
        let fs = 100.0;
        let ts: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let (_, coh) = spectral_coherence(&ts, &ts, fs, 64).expect("coherence failed");
        // Coherence with self should be ~1 at all frequencies with signal power
        let max_coh = coh.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_coh > 0.99, "self-coherence should be ~1, got {max_coh}");
    }

    #[test]
    fn test_spectral_coherence_orthogonal_signals() {
        let n = 128;
        let fs = 100.0;
        // Orthogonal: sin and cos at the same frequency
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 20.0 * i as f64 / fs).sin())
            .collect();
        let (_, coh) = spectral_coherence(&x, &y, fs, 64).expect("coherence failed");
        // Average coherence should be low
        let mean_coh = coh.iter().sum::<f64>() / coh.len() as f64;
        assert!(
            mean_coh < 0.6,
            "coherence between different-freq signals should be low, got {mean_coh}"
        );
    }

    #[test]
    fn test_cross_spectral_density_length() {
        let n = 64;
        let fs = 100.0;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let (freqs, sxy) = cross_spectral_density(&x, &y, fs).expect("CSD failed");
        assert_eq!(freqs.len(), n / 2 + 1);
        assert_eq!(sxy.len(), n / 2 + 1);
    }

    #[test]
    fn test_nonstationarity_test_stationary() {
        // A pure sine should be very stationary
        let n = 256;
        let fs = 100.0;
        let ts = make_sine(n, 10.0, fs);
        let cv = nonstationarity_test(&ts, 64, 32).expect("nonstationarity_test failed");
        assert!(
            cv < 1.0,
            "pure sine should have low non-stationarity, got {cv}"
        );
    }

    #[test]
    fn test_nonstationarity_test_nonstationary() {
        // A signal that changes character halfway through
        let n = 256;
        let fs = 100.0;
        let mut ts: Vec<f64> = make_sine(n / 2, 5.0, fs);
        ts.extend(make_sine(n / 2, 40.0, fs).into_iter().map(|v| v * 10.0));
        let cv = nonstationarity_test(&ts, 64, 32).expect("nonstationarity_test failed");
        // The changing-amplitude signal should have higher CV than a pure sine
        assert!(
            cv > 0.0,
            "non-stationary signal should have cv > 0, got {cv}"
        );
    }

    #[test]
    fn test_wigner_ville_distribution_shape() {
        let n = 32;
        let fs = 100.0;
        let ts = make_sine(n, 5.0, fs);
        let wvd = wigner_ville_distribution(&ts, fs).expect("WVD failed");
        assert_eq!(wvd.dim().0, n);
    }

    #[test]
    fn test_power_spectrum_error_short_input() {
        let ts = vec![1.0, 2.0];
        assert!(power_spectrum(&ts, 100.0).is_err());
    }

    #[test]
    fn test_dpss_tapers_error_invalid_params() {
        assert!(dpss_tapers(64, 0.0, 3).is_err());
        assert!(dpss_tapers(1, 4.0, 3).is_err());
        assert!(dpss_tapers(64, 4.0, 0).is_err());
    }
}
