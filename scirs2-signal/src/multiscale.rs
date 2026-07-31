//! Multiscale signal analysis: VMD, Hilbert-Huang Transform.
//!
//! This module provides advanced multiscale decomposition methods that go
//! beyond the classical wavelet or Fourier paradigms:
//!
//! - **VMD** (Variational Mode Decomposition): Data-driven decomposition into
//!   AM-FM modes by solving a constrained variational problem in the frequency
//!   domain. Avoids the mode-mixing problem of EMD.
//! - **Hilbert-Huang Transform (HHT)**: Combines EMD with the Hilbert
//!   transform to produce instantaneous amplitude and frequency for each IMF.
//!
//! The EMD and EEMD implementations live in [`crate::emd`]; this module
//! wraps them for the HHT and adds VMD.
//!
//! # References
//!
//! - Dragomiretskiy & Zosso (2014) – "Variational Mode Decomposition",
//!   *IEEE Trans. Signal Process.* 62(3), 531–544.
//! - Huang et al. (1998) – "The empirical mode decomposition and the Hilbert
//!   spectrum for nonlinear and non-stationary time series analysis",
//!   *Proc. R. Soc. Lond. A* 454, 903–995.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal FFT helpers (complex-valued)
// ─────────────────────────────────────────────────────────────────────────────

type Complex = (f64, f64); // (re, im)

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn c_sub(a: Complex, b: Complex) -> Complex {
    (a.0 - b.0, a.1 - b.1)
}
#[inline]
fn c_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
#[inline]
fn c_div(a: Complex, b: Complex) -> Complex {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
#[inline]
fn c_conj(a: Complex) -> Complex {
    (a.0, -a.1)
}
#[inline]
fn c_abs(a: Complex) -> f64 {
    (a.0 * a.0 + a.1 * a.1).sqrt()
}
#[inline]
fn c_real(a: Complex) -> f64 {
    a.0
}
#[inline]
fn c_imag(a: Complex) -> f64 {
    a.1
}
#[inline]
fn c_scale(a: Complex, s: f64) -> Complex {
    (a.0 * s, a.1 * s)
}

fn fft_complex(buf: &mut Vec<Complex>, inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reverse permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Cooley-Tukey
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wlen: Complex = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w: Complex = (1.0, 0.0);
            for jj in 0..len / 2 {
                let u = buf[i + jj];
                let v = c_mul(buf[i + jj + len / 2], w);
                buf[i + jj] = c_add(u, v);
                buf[i + jj + len / 2] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for x in buf.iter_mut() {
            *x = c_scale(*x, scale);
        }
    }
}

/// FFT of a real-valued signal, zero-padded to `n` (must be power of two).
fn rfft(x: &[f64], n: usize) -> Vec<Complex> {
    let mut buf: Vec<Complex> = x.iter().map(|&v| (v, 0.0)).collect();
    buf.resize(n, (0.0, 0.0));
    fft_complex(&mut buf, false);
    // Return full spectrum (n complex values)
    buf
}

/// IFFT, returns real part of length-n signal.
fn irfft(x: &[Complex], n: usize) -> Vec<f64> {
    let mut buf = x.to_vec();
    buf.resize(n, (0.0, 0.0));
    fft_complex(&mut buf, true);
    buf.into_iter().map(|c| c.0).collect()
}

/// Next power-of-two >= n.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1usize << (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hilbert transform
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the analytic signal via the Hilbert transform.
///
/// Returns complex analytic signal `z[n] = x[n] + j·H{x}[n]` where
/// `H{·}` is the Hilbert transform.
///
/// # Example
///
/// ```
/// use scirs2_signal::multiscale::hilbert_transform;
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
///     .collect();
/// let analytic = hilbert_transform(&signal);
/// assert_eq!(analytic.len(), n);
/// // Instantaneous amplitude should be near 1.0 for a pure sine
/// for &[re, im] in analytic.iter().skip(10).take(n - 20) {
///     let amp = (re * re + im * im).sqrt();
///     assert!((amp - 1.0).abs() < 0.05, "Amplitude {} should be near 1", amp);
/// }
/// ```
pub fn hilbert_transform(signal: &[f64]) -> Vec<[f64; 2]> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let n_fft = next_pow2(n);
    let half = n_fft / 2;
    // Compute analytic signal: take full complex IFFT.
    let mut spec2 = rfft(signal, n_fft);
    for k in 0..n_fft {
        let scale = if k == 0 || k == half {
            1.0
        } else if k < half {
            2.0
        } else {
            0.0
        };
        spec2[k] = c_scale(spec2[k], scale);
    }
    fft_complex(&mut spec2, true);

    (0..n).map(|i| [spec2[i].0, spec2[i].1]).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Hilbert-Huang Transform
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hilbert-Huang Transform (HHT) of a signal.
///
/// Decomposes the signal via EMD into IMFs, then applies the Hilbert
/// transform to each IMF to obtain instantaneous amplitude and frequency.
///
/// # Returns
///
/// A `Vec` of `(instantaneous_amplitude, instantaneous_frequency)` pairs,
/// one per IMF. Each vector has `signal.len()` elements (with boundary
/// artifacts at the edges).
///
/// # Example
///
/// ```
/// use scirs2_signal::multiscale::hht;
/// use std::f64::consts::PI;
///
/// let n = 512;
/// let sr = 512.0;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / sr).sin()
///            + 0.3 * (2.0 * PI * 50.0 * i as f64 / sr).sin())
///     .collect();
/// let hht_result = hht(&signal, sr);
/// assert!(!hht_result.is_empty(), "HHT should produce at least one IMF");
/// ```
pub fn hht(signal: &[f64], sample_rate: f64) -> Vec<(Vec<f64>, Vec<f64>)> {
    if signal.is_empty() {
        return Vec::new();
    }
    // Use the existing EMD implementation
    let config = crate::emd::EmdConfig {
        max_imfs: 8,
        sift_threshold: 0.05,
        max_sift_iterations: 100,
        boundary_condition: "mirror".to_string(),
        interpolation: "cubic".to_string(),
        min_extrema: 3,
    };
    let emd_result = match crate::emd::emd(signal, &config) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let n_imfs = emd_result.imfs.shape()[0];
    let n = signal.len();
    let dt = 1.0 / sample_rate;

    (0..n_imfs)
        .map(|i| {
            let imf: Vec<f64> = (0..n).map(|t| emd_result.imfs[[i, t]]).collect();
            let analytic = hilbert_transform(&imf);

            // Instantaneous amplitude
            let amp: Vec<f64> = analytic
                .iter()
                .map(|&[re, im]| (re * re + im * im).sqrt())
                .collect();

            // Instantaneous phase
            let phase: Vec<f64> = analytic.iter().map(|&[re, im]| im.atan2(re)).collect();

            // Unwrap phase
            let mut unwrapped = phase.clone();
            for t in 1..unwrapped.len() {
                let mut diff = unwrapped[t] - unwrapped[t - 1];
                while diff > PI {
                    diff -= 2.0 * PI;
                }
                while diff < -PI {
                    diff += 2.0 * PI;
                }
                unwrapped[t] = unwrapped[t - 1] + diff;
            }

            // Instantaneous frequency = d(phase)/dt / (2π)
            let freq: Vec<f64> = if unwrapped.len() < 2 {
                vec![0.0; unwrapped.len()]
            } else {
                let mut f = vec![0.0_f64; unwrapped.len()];
                for t in 1..unwrapped.len() - 1 {
                    f[t] = (unwrapped[t + 1] - unwrapped[t - 1]) / (4.0 * PI * dt);
                }
                // Boundary: forward/backward difference
                if unwrapped.len() >= 2 {
                    f[0] = (unwrapped[1] - unwrapped[0]) / (2.0 * PI * dt);
                    let last = unwrapped.len() - 1;
                    f[last] = (unwrapped[last] - unwrapped[last - 1]) / (2.0 * PI * dt);
                }
                f
            };

            (amp, freq)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Variational Mode Decomposition (VMD)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of Variational Mode Decomposition.
#[derive(Debug, Clone)]
pub struct VmdResult {
    /// Decomposed modes, shape `(n_modes, signal_length)`.
    pub modes: Vec<Vec<f64>>,
    /// Estimated center frequency (in normalized units [0, 0.5]) for each mode.
    pub center_frequencies: Vec<f64>,
    /// Number of outer iterations performed.
    pub n_iter: usize,
    /// Final convergence criterion value.
    pub convergence: f64,
}

/// Variational Mode Decomposition (VMD).
///
/// Decomposes `signal` into `n_modes` AM-FM modes by minimizing the
/// constrained variational problem:
///
/// ```text
/// min_{u_k, ω_k}  Σ_k || ∂_t [ (δ(t) + j/πt) * u_k(t) ] e^{-jω_k t} ||²
///   s.t.  Σ_k u_k = f
/// ```
///
/// solved in the spectral domain via alternating direction updates
/// (ADMM-like), following Dragomiretskiy & Zosso (2014).
///
/// # Arguments
///
/// * `signal` - Input signal.
/// * `n_modes` - Number of modes to extract (K).
/// * `alpha` - Bandwidth constraint: larger α → narrower spectral bandwidth
///   per mode. A typical starting point is `alpha = 2000.0`.
/// * `tau` - Noise tolerance (Lagrange multiplier step size). Use `0.0` for
///   noise-free signals, or a small positive value (e.g. `0.0`) for noisy data.
/// * `max_iter` - Maximum number of outer ADMM iterations.
/// * `tol` - Convergence tolerance (e.g. `1e-7`).
///
/// # Returns
///
/// [`VmdResult`] containing the modes and their center frequencies.
///
/// # Example
///
/// ```
/// use scirs2_signal::multiscale::{vmd, VmdResult};
/// use std::f64::consts::PI;
///
/// let n = 512;
/// // Composite signal: two AM-FM modes
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / n as f64;
///         (2.0 * PI * 5.0 * t).cos() + 0.5 * (2.0 * PI * 20.0 * t).cos()
///     })
///     .collect();
///
/// let result = vmd(&signal, 2, 2000.0, 0.0, 500, 1e-7)
///     .expect("VMD should succeed");
/// assert_eq!(result.modes.len(), 2);
/// assert_eq!(result.center_frequencies.len(), 2);
/// // Modes should have the same length as the input
/// for mode in &result.modes {
///     assert_eq!(mode.len(), n);
/// }
/// ```
pub fn vmd(
    signal: &[f64],
    n_modes: usize,
    alpha: f64,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> SignalResult<VmdResult> {
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if n_modes == 0 {
        return Err(SignalError::ValueError(
            "n_modes must be at least 1".to_string(),
        ));
    }

    // Mirror-pad to length 2N to reduce boundary effects
    let n_pad = 2 * n;
    let mut f_mirror = vec![0.0_f64; n_pad];
    for i in 0..n {
        f_mirror[i] = signal[n - 1 - i]; // left mirror
    }
    for i in 0..n {
        f_mirror[n + i] = signal[i]; // original
    }

    // FFT of mirrored signal
    let n_fft = next_pow2(n_pad);
    let f_hat = rfft(&f_mirror, n_fft);

    // Frequency axis [0, 1) normalized, only positive half
    let n_half = n_fft / 2;
    let freqs: Vec<f64> = (0..n_fft).map(|k| k as f64 / n_fft as f64).collect();

    // Initialize mode spectra u_hat_k and center frequencies omega_k
    let mut u_hat: Vec<Vec<Complex>> = (0..n_modes)
        .map(|k| {
            // Initialize with uniform distribution
            f_hat
                .iter()
                .map(|&v| c_scale(v, 1.0 / n_modes as f64))
                .collect()
        })
        .collect();

    // Initialize center frequencies uniformly in [0, 0.5]
    let mut omega: Vec<f64> = (0..n_modes)
        .map(|k| (k + 1) as f64 / (2.0 * (n_modes + 1) as f64))
        .collect();

    // Lagrange multiplier in frequency domain
    let mut lambda_hat: Vec<Complex> = vec![(0.0, 0.0); n_fft];

    let mut convergence = f64::INFINITY;
    let mut n_iter = 0usize;

    // --- ADMM outer loop ---
    'outer: for iter in 0..max_iter {
        n_iter = iter + 1;

        let u_hat_prev: Vec<Vec<Complex>> = u_hat.clone();
        let omega_prev = omega.clone();

        // Update each mode k
        for k in 0..n_modes {
            // Accumulate: sum of all other modes
            let mut acc: Vec<Complex> = f_hat.clone();
            for kk in 0..n_modes {
                if kk == k {
                    continue;
                }
                for i in 0..n_fft {
                    acc[i] = c_sub(acc[i], u_hat[kk][i]);
                }
            }
            // Subtract half Lagrange multiplier
            for i in 0..n_fft {
                acc[i] = c_sub(acc[i], c_scale(lambda_hat[i], 0.5));
            }

            // Wiener filter in frequency domain
            // u_hat_k[ω] = (f_hat - Σ_{j≠k} u_hat_j - λ/2)
            //               / (1 + 2α(ω - ω_k)²)
            for i in 0..n_fft {
                let denom = 1.0 + 2.0 * alpha * (freqs[i] - omega[k]).powi(2);
                u_hat[k][i] = c_scale(acc[i], 1.0 / denom);
            }

            // Update center frequency: weighted centroid of |u_hat_k|²
            // Only positive frequencies (0..n_half)
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..n_half {
                let power = c_abs(u_hat[k][i]).powi(2);
                num += freqs[i] * power;
                den += power;
            }
            omega[k] = if den > 1e-15 { num / den } else { omega[k] };
            // Clamp to [0, 0.5]
            omega[k] = omega[k].max(0.0).min(0.5);
        }

        // Update Lagrange multiplier
        // λ_hat += τ · (f_hat - Σ_k u_hat_k)
        if tau != 0.0 {
            let mut residual: Vec<Complex> = f_hat.clone();
            for k in 0..n_modes {
                for i in 0..n_fft {
                    residual[i] = c_sub(residual[i], u_hat[k][i]);
                }
            }
            for i in 0..n_fft {
                lambda_hat[i] = c_add(lambda_hat[i], c_scale(residual[i], tau));
            }
        }

        // Convergence check: sum over k of ||u_hat_k - u_hat_k_prev||² / ||u_hat_k_prev||²
        let mut conv_sum = 0.0_f64;
        for k in 0..n_modes {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..n_fft {
                let diff = c_sub(u_hat[k][i], u_hat_prev[k][i]);
                num += c_abs(diff).powi(2);
                den += c_abs(u_hat_prev[k][i]).powi(2);
            }
            conv_sum += if den > 1e-15 { num / den } else { num };
        }
        convergence = conv_sum;

        if convergence < tol && iter > 0 {
            break 'outer;
        }
    }

    // Reconstruct modes in time domain
    // Only take the central portion (remove mirror padding)
    let modes: Vec<Vec<f64>> = u_hat
        .iter()
        .map(|u_k| {
            // `u_k` already holds the full two-sided spectral content matched
            // against `f_hat` (both were carried through the ADMM loop over
            // *all* `n_fft` bins, not a half-strength one-sided spectrum), so
            // rebuilding a real, Hermitian-symmetric spectrum from its
            // positive half needs no extra amplitude compensation: mirroring
            // `full[0..n_half]` onto the negative-frequency bins already
            // reconstitutes 100% of the signal's energy.
            let mut full: Vec<Complex> = u_k.clone();
            // For real signal, conjugate symmetry
            for i in 1..n_half {
                full[n_fft - i] = c_conj(full[i]);
            }
            let time_domain = irfft(&full, n_fft);
            // Extract the original (non-mirrored) portion: indices [n..n+n] of n_pad
            // But after zero-padding to n_fft, scaling differs; take [n..2n] from length-n_pad
            // We padded from n_pad to n_fft, so the actual signal is at [n..2n] of n_pad
            let start = n; // skip the left-mirror half
            time_domain[start..start + n].to_vec()
        })
        .collect();

    Ok(VmdResult {
        modes,
        center_frequencies: omega,
        n_iter,
        convergence,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_hilbert_transform_length() {
        let signal = sine_wave(10.0, 256.0, 256);
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), signal.len());
    }

    #[test]
    fn test_hilbert_transform_amplitude() {
        // For a pure sine, the envelope should be approximately 1.0
        let n = 512;
        let sr = 512.0;
        let signal = sine_wave(10.0, sr, n);
        let analytic = hilbert_transform(&signal);
        // Skip edge effects
        let skip = 20;
        for &[re, im] in analytic[skip..n - skip].iter() {
            let amp = (re * re + im * im).sqrt();
            assert!(
                (amp - 1.0).abs() < 0.05,
                "Amplitude {} should be near 1.0",
                amp
            );
        }
    }

    #[test]
    fn test_hilbert_transform_quadrature() {
        // H{cos(ωt)} = sin(ωt), so the imaginary part of analytic(cos) should be sin
        let n = 256;
        let sr = 256.0;
        let freq = 5.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).cos())
            .collect();
        let analytic = hilbert_transform(&signal);
        let skip = 20;
        for i in skip..n - skip {
            let expected_im = (2.0 * PI * freq * i as f64 / sr).sin();
            let actual_im = analytic[i][1];
            assert!(
                (actual_im - expected_im).abs() < 0.05,
                "Imaginary part {} should be near sin = {} at i={}",
                actual_im,
                expected_im,
                i
            );
        }
    }

    #[test]
    fn test_hht_output_shape() {
        let n = 256;
        let sr = 256.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 5.0 * i as f64 / sr).sin()
                    + 0.5 * (2.0 * PI * 20.0 * i as f64 / sr).sin()
            })
            .collect();
        let result = hht(&signal, sr);
        assert!(!result.is_empty(), "HHT should produce IMFs");
        for (amp, freq) in &result {
            assert_eq!(amp.len(), n, "Amplitude length mismatch");
            assert_eq!(freq.len(), n, "Frequency length mismatch");
        }
    }

    #[test]
    fn test_hht_amplitude_non_negative() {
        let n = 256;
        let sr = 256.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / sr).sin())
            .collect();
        let result = hht(&signal, sr);
        for (amp, _) in &result {
            for &a in amp {
                assert!(
                    a >= 0.0,
                    "Instantaneous amplitude must be non-negative: {}",
                    a
                );
            }
        }
    }

    #[test]
    fn test_vmd_basic() {
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).cos() + 0.5 * (2.0 * PI * 20.0 * t).cos()
            })
            .collect();

        let result = vmd(&signal, 2, 2000.0, 0.0, 200, 1e-6).expect("VMD should succeed");

        assert_eq!(result.modes.len(), 2, "Should produce 2 modes");
        assert_eq!(result.center_frequencies.len(), 2);
        for mode in &result.modes {
            assert_eq!(mode.len(), n, "Mode length should match signal");
        }
    }

    #[test]
    fn test_vmd_reconstruction() {
        // Sum of modes should approximately equal the original signal
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).cos() + 0.5 * (2.0 * PI * 20.0 * t).cos()
            })
            .collect();

        let result = vmd(&signal, 2, 2000.0, 0.0, 300, 1e-7).expect("VMD should succeed");

        let reconstructed: Vec<f64> = (0..n)
            .map(|i| result.modes.iter().map(|m| m[i]).sum::<f64>())
            .collect();

        let rmse: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        let rmse = rmse.sqrt();
        // Reconstruction error should be reasonably small
        assert!(
            rmse < 0.5,
            "RMSE {} should be small for VMD reconstruction",
            rmse
        );
    }

    #[test]
    fn test_vmd_center_frequencies_ordered() {
        // With two known frequencies (5 Hz and 20 Hz on unit interval → 5/512, 20/512),
        // the recovered center frequencies should be distinct.
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).cos() + (2.0 * PI * 20.0 * t).cos()
            })
            .collect();

        let result = vmd(&signal, 2, 2000.0, 0.0, 300, 1e-7).expect("VMD should succeed");

        let freqs = &result.center_frequencies;
        assert_eq!(freqs.len(), 2);
        // Frequencies should be in [0, 0.5]
        for &f in freqs {
            assert!(
                f >= 0.0 && f <= 0.5,
                "Frequency {} out of range [0, 0.5]",
                f
            );
        }
        // The two frequencies should differ
        assert!(
            (freqs[0] - freqs[1]).abs() > 0.001,
            "Center frequencies {} and {} should differ",
            freqs[0],
            freqs[1]
        );
    }

    #[test]
    fn test_vmd_single_mode() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 10.0 * t).cos()
            })
            .collect();

        let result =
            vmd(&signal, 1, 2000.0, 0.0, 100, 1e-6).expect("VMD with 1 mode should succeed");

        assert_eq!(result.modes.len(), 1);
        assert_eq!(result.modes[0].len(), n);
    }

    #[test]
    fn test_vmd_error_on_empty() {
        let result = vmd(&[], 2, 2000.0, 0.0, 100, 1e-6);
        assert!(result.is_err(), "Empty signal should return error");
    }

    #[test]
    fn test_vmd_error_on_zero_modes() {
        let signal = vec![1.0_f64; 256];
        let result = vmd(&signal, 0, 2000.0, 0.0, 100, 1e-6);
        assert!(result.is_err(), "Zero modes should return error");
    }
}
