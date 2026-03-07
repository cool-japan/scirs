// FIR (Finite Impulse Response) filter design functions
//
// This module provides comprehensive FIR filter design capabilities including
// window-based design (firwin) and optimal equiripple design (Parks-McClellan/Remez).
// FIR filters offer linear phase response and guaranteed stability.

use super::common::validation::validate_cutoff_frequency;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
/// FIR filter design using window method
///
/// Designs a linear phase FIR filter using the window method. The filter
/// is obtained by truncating and windowing the ideal impulse response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is Nyquist frequency)
/// * `window` - Window function name ("hamming", "hann", "blackman", "kaiser", etc.)
/// * `pass_zero` - If true, the filter is lowpass; if false, highpass
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::firwin;
///
/// // Design a 65-tap lowpass filter with Hamming window
/// let h = firwin(65, 0.3, "hamming", true).expect("Operation failed");
///
/// // Design a highpass filter
/// let h = firwin(65, 0.3, "hamming", false).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn firwin<T>(
    _numtaps: usize,
    cutoff: T,
    window: &str,
    pass_zero: bool,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if _numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    let wc = validate_cutoff_frequency(cutoff)?;

    // Calculate the ideal impulse response
    let mid = (_numtaps - 1) as f64 / 2.0;
    let mut h = vec![0.0; _numtaps];

    for (i, item) in h.iter_mut().enumerate() {
        let n = i as f64 - mid;

        if n == 0.0 {
            // At n=0, use L'Hôpital's rule result
            *item = if pass_zero {
                wc / std::f64::consts::PI
            } else {
                1.0 - wc / std::f64::consts::PI
            };
        } else {
            // General case: sinc function
            let sinc_val = (wc * std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n);
            *item = if pass_zero {
                sinc_val
            } else {
                // Highpass: subtract lowpass from delta function
                if i == _numtaps / 2 {
                    1.0 - sinc_val
                } else {
                    -sinc_val
                }
            };
        }
    }

    // Apply window function
    let window_coeffs = generate_window(_numtaps, window)?;
    for (i, coeff) in h.iter_mut().enumerate() {
        *coeff *= window_coeffs[i];
    }

    // Normalize to ensure unity gain at DC (for lowpass) or Nyquist (for highpass)
    let sum: f64 = h.iter().sum();
    if pass_zero && sum.abs() > 1e-10 {
        for coeff in &mut h {
            *coeff /= sum;
        }
    } else if !pass_zero {
        // For highpass, normalize for unity gain at Nyquist
        let nyquist_response: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &coeff)| coeff * (-1.0_f64).powi(i as i32))
            .sum();
        if nyquist_response.abs() > 1e-10 {
            for coeff in &mut h {
                *coeff /= nyquist_response;
            }
        }
    }

    Ok(h)
}

/// Parks-McClellan optimal FIR filter design (Remez exchange algorithm)
///
/// Design a linear phase FIR filter using the Parks-McClellan algorithm.
/// The algorithm finds the filter coefficients that minimize the maximum
/// error between the desired and actual frequency response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `bands` - Frequency bands specified as pairs of band edges (0 to 1, where 1 is Nyquist)
/// * `desired` - Desired gain for each band
/// * `weights` - Relative weights for each band (optional)
/// * `max_iter` - Maximum number of iterations (default: 25)
/// * `grid_density` - Grid density for frequency sampling (default: 16)
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::remez;
///
/// // Design a 65-tap lowpass filter
/// // Passband: 0-0.4, Stopband: 0.45-1.0
/// let bands = vec![0.0, 0.4, 0.45, 1.0];
/// let desired = vec![1.0, 1.0, 0.0, 0.0];
/// let h = remez(65, &bands, &desired, None, None, None).expect("Operation failed");
/// ```
/// Minimum absolute difference for barycentric weight computation to avoid division by zero.
const BARY_EPSILON: f64 = 1e-15;
/// Minimum denominator magnitude for barycentric interpolation to avoid division by zero.
const BARY_MIN_DENOM: f64 = 1e-30;

#[allow(dead_code)]
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weights: Option<&[f64]>,
    max_iter: Option<usize>,
    grid_density: Option<usize>,
) -> SignalResult<Vec<f64>> {
    // Validate inputs
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if !bands.len().is_multiple_of(2) || bands.len() < 2 {
        return Err(SignalError::ValueError(
            "Bands must be specified as pairs of edges".to_string(),
        ));
    }

    if desired.len() != bands.len() {
        return Err(SignalError::ValueError(
            "Desired array must have same length as bands".to_string(),
        ));
    }

    // Check that bands are monotonically increasing
    for i in 1..bands.len() {
        if bands[i] <= bands[i - 1] {
            return Err(SignalError::ValueError(
                "Band edges must be monotonically increasing".to_string(),
            ));
        }
    }

    // Check that bands are within [0, 1]
    if bands[0] < 0.0 || bands[bands.len() - 1] > 1.0 {
        return Err(SignalError::ValueError(
            "Band edges must be between 0 and 1".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(25);
    let grid_density = grid_density.unwrap_or(16);

    let num_bands = bands.len() / 2;

    // Determine one weight per band.
    // Accepts either num_bands weights (one per band) or bands.len() weights (per edge, averaged).
    // Safety: bands.len() is already validated to be even (see check above), so
    // w.len() == bands.len() guarantees w[2*i+1] is in-bounds for i in 0..num_bands.
    let band_weights: Vec<f64> = if let Some(w) = weights {
        if w.len() == num_bands {
            w.to_vec()
        } else if w.len() == bands.len() {
            // Per-edge weights: average the two edges for each band
            (0..num_bands)
                .map(|i| (w[2 * i] + w[2 * i + 1]) / 2.0)
                .collect()
        } else {
            vec![1.0; num_bands]
        }
    } else {
        vec![1.0; num_bands]
    };

    // Filter half-order M = (N-1)/2 for Type I (odd N).
    // For even N (Type II) the same formula is used as an approximation.
    let filter_order = numtaps - 1;
    let m = filter_order / 2;

    // Number of extremal frequencies required by the alternation theorem: r = M + 2.
    // A common bug is to use M+1 here, which makes the algorithm degenerate.
    let r = m + 2;

    // Build the dense frequency grid (ω in [0, π])
    let grid_size = (grid_density * filter_order).max(r + 1);
    let mut omega_grid: Vec<f64> = Vec::with_capacity(grid_size);
    let mut desired_grid: Vec<f64> = Vec::with_capacity(grid_size);
    let mut weight_grid: Vec<f64> = Vec::with_capacity(grid_size);

    for band_idx in 0..num_bands {
        let f0 = bands[2 * band_idx];
        let f1 = bands[2 * band_idx + 1];
        let pts = ((f1 - f0) * grid_size as f64).round().max(2.0) as usize;
        for i in 0..pts {
            let t = i as f64 / (pts - 1) as f64;
            let f = f0 + (f1 - f0) * t;
            omega_grid.push(f * std::f64::consts::PI);
            // Linear interpolation for desired response within the band
            let d = desired[2 * band_idx] * (1.0 - t) + desired[2 * band_idx + 1] * t;
            desired_grid.push(d);
            weight_grid.push(band_weights[band_idx]);
        }
    }

    if omega_grid.len() < r + 1 {
        return Err(SignalError::ValueError(
            "Grid too small for the requested filter order".to_string(),
        ));
    }

    // Initialize extremal set uniformly across the grid
    let mut ext: Vec<usize> = (0..r)
        .map(|i| i * (omega_grid.len() - 1) / (r - 1))
        .collect();

    // Remez exchange iterations
    for _iter in 0..max_iter {
        // Cosine of the extremal frequencies (x-coordinates for Lagrange interpolation)
        let x: Vec<f64> = ext.iter().map(|&i| omega_grid[i].cos()).collect();
        let d: Vec<f64> = ext.iter().map(|&i| desired_grid[i]).collect();
        let w: Vec<f64> = ext.iter().map(|&i| weight_grid[i]).collect();

        // Barycentric weights: λ_i = 1 / ∏_{j≠i} (x_i − x_j)
        let bary = compute_barycentric_weights(&x);

        // Equiripple deviation δ via the alternation-theorem formula:
        //   δ = (Σ λ_i · D_i) / (Σ (−1)^i · λ_i / W_i)
        let (num_d, den_d) = delta_numerator_denominator(&bary, &d, &w);
        let delta = if den_d.abs() > BARY_MIN_DENOM {
            num_d / den_d
        } else {
            0.0
        };

        // Adjusted desired values at extremal points:
        //   E_i = D_i − (−1)^i · δ / W_i
        // These are the values that the optimal polynomial P takes at each extremal point.
        let e: Vec<f64> = (0..r)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                d[i] - sign * delta / w[i]
            })
            .collect();

        // Evaluate P on the dense grid using barycentric interpolation with E values.
        // Using D values here (as done in the old code) is the root cause of the bug.
        // We reuse a single Vec, first storing P(ω), then overwriting with |D−P|·W.
        let mut errors: Vec<f64> = Vec::with_capacity(omega_grid.len());
        for &om in &omega_grid {
            let xg = om.cos();
            let p = barycentric_eval(&bary, &x, &e, xg);
            errors.push(p); // initially stores P(ω); converted to weighted error below
        }
        // Convert P(ω) values to weighted error magnitude |D(ω) − P(ω)| · W(ω)
        for (gi, err) in errors.iter_mut().enumerate() {
            *err = ((desired_grid[gi] - *err) * weight_grid[gi]).abs();
        }

        // Find new extremal set: local maxima of |error|, keep r largest
        let mut new_ext: Vec<usize> = Vec::new();
        if errors.first().copied().unwrap_or(0.0)
            >= errors.get(1).copied().unwrap_or(0.0)
        {
            new_ext.push(0);
        }
        for i in 1..(errors.len() - 1) {
            if errors[i] >= errors[i - 1] && errors[i] >= errors[i + 1] {
                new_ext.push(i);
            }
        }
        if errors.last().copied().unwrap_or(0.0)
            >= errors.get(errors.len() - 2).copied().unwrap_or(0.0)
        {
            new_ext.push(errors.len() - 1);
        }

        new_ext.sort_by(|&a, &b| {
            errors[b]
                .partial_cmp(&errors[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        new_ext.truncate(r);
        new_ext.sort();

        if new_ext.len() < r {
            break; // cannot improve further
        }
        ext = new_ext;
    }

    // ── Coefficient extraction ────────────────────────────────────────────────
    // Rebuild barycentric interpolation for the final extremal set.
    let x_f: Vec<f64> = ext.iter().map(|&i| omega_grid[i].cos()).collect();
    let d_f: Vec<f64> = ext.iter().map(|&i| desired_grid[i]).collect();
    let w_f: Vec<f64> = ext.iter().map(|&i| weight_grid[i]).collect();
    let bary_f = compute_barycentric_weights(&x_f);

    let (num_d, den_d) = delta_numerator_denominator(&bary_f, &d_f, &w_f);
    let delta_f = if den_d.abs() > BARY_MIN_DENOM {
        num_d / den_d
    } else {
        0.0
    };

    let e_f: Vec<f64> = (0..r)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            d_f[i] - sign * delta_f / w_f[i]
        })
        .collect();

    // For a Type I filter H(ω) = Σ_{k=0}^{M} a[k] cos(kω).
    // Sample P at M+1 evenly-spaced frequencies ω_j = j·π/M (j = 0…M) using
    // barycentric interpolation, then apply the inverse DCT-I to recover a[k].
    let mut h = vec![0.0_f64; numtaps];

    if m == 0 {
        // Trivial single-tap case
        h[0] = barycentric_eval(&bary_f, &x_f, &e_f, 1.0); // ω=0
        return Ok(h);
    }

    // Evaluate P at the M+1 DCT-I nodes
    let mut p_vals = vec![0.0_f64; m + 1];
    let pi_over_m = std::f64::consts::PI / (m as f64);
    for j in 0..=m {
        let omega_j = (j as f64) * pi_over_m;
        p_vals[j] = barycentric_eval(&bary_f, &x_f, &e_f, omega_j.cos());
    }

    // Inverse DCT-I: recover cosine-series coefficients a[k].
    // From the orthogonality of the DCT-I basis on {j·π/M, j=0…M}:
    //   a[0]      = (1/M) · Σ_j  wj · p_vals[j]
    //   a[k]      = (2/M) · Σ_j  wj · p_vals[j] · cos(k·j·π/M)   (1 ≤ k ≤ M-1)
    //   a[M]      = (1/M) · Σ_j  wj · p_vals[j] · (−1)^j
    // where wj = 0.5 for j=0,M and wj = 1 otherwise.
    let inv_m = 1.0 / (m as f64);
    let mut a = vec![0.0_f64; m + 1];
    for k in 0..=m {
        let mut sum = p_vals[0] * 0.5; // j = 0, weight 0.5
        for j in 1..m {
            sum += p_vals[j] * ((k as f64 * j as f64 * pi_over_m).cos());
        }
        // j = m, weight 0.5
        sum += p_vals[m] * 0.5 * ((k as f64 * std::f64::consts::PI).cos());
        a[k] = sum * inv_m;
        // Interior coefficients have an additional factor of 2 (from DCT-I orthogonality)
        if k > 0 && k < m {
            a[k] *= 2.0;
        }
    }

    // Convert cosine-series to symmetric FIR taps:
    //   h[M]     = a[0]
    //   h[M±k]   = a[k]/2   for k = 1…M
    h[m] = a[0];
    for k in 1..=m {
        h[m - k] = a[k] / 2.0;
        h[m + k] = a[k] / 2.0;
    }

    Ok(h)
}

/// Compute barycentric weights for Lagrange interpolation at nodes `x`.
/// λ_i = 1 / ∏_{j≠i} (x_i − x_j)
fn compute_barycentric_weights(x: &[f64]) -> Vec<f64> {
    let r = x.len();
    let mut bary = vec![1.0_f64; r];
    for i in 0..r {
        for j in 0..r {
            if i != j {
                let diff = x[i] - x[j];
                if diff.abs() > BARY_EPSILON {
                    bary[i] /= diff;
                }
            }
        }
    }
    bary
}

/// Evaluate the barycentric Lagrange interpolant at `xg`.
/// Returns y[i] directly if `xg` coincides with a node.
fn barycentric_eval(bary: &[f64], x: &[f64], y: &[f64], xg: f64) -> f64 {
    let r = bary.len();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..r {
        let dx = xg - x[i];
        if dx.abs() < BARY_EPSILON {
            return y[i]; // exactly at a node
        }
        let b = bary[i] / dx;
        num += b * y[i];
        den += b;
    }
    if den.abs() > BARY_MIN_DENOM {
        num / den
    } else {
        y[0]
    }
}

/// Compute the numerator and denominator used for the equiripple deviation δ.
///   num = Σ λ_i · D_i
///   den = Σ (−1)^i · λ_i / W_i
fn delta_numerator_denominator(bary: &[f64], d: &[f64], w: &[f64]) -> (f64, f64) {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..bary.len() {
        let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        num += bary[i] * d[i];
        den += sign * bary[i] / w[i];
    }
    (num, den)
}

/// Generate a window function
///
/// Creates a window function of the specified type and length.
///
/// # Arguments
///
/// * `length` - Window length
/// * `window_type` - Window type ("hamming", "hann", "blackman", "kaiser", etc.)
///
/// # Returns
///
/// * Window coefficients as a vector
#[allow(dead_code)]
fn generate_window(_length: usize, windowtype: &str) -> SignalResult<Vec<f64>> {
    let mut window = vec![0.0; _length];

    match windowtype.to_lowercase().as_str() {
        "hamming" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                *w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos();
            }
        }
        "hann" | "hanning" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                *w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos());
            }
        }
        "blackman" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                let arg = 2.0 * std::f64::consts::PI * n / (total - 1.0);
                *w = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
            }
        }
        "rectangular" | "boxcar" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}. Supported types: hamming, hann, blackman, rectangular",
                windowtype
            )));
        }
    }

    Ok(window)
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Evaluate the frequency response magnitude of an FIR filter at frequency f ∈ [0, 1].
    fn freq_mag(h: &[f64], f: f64) -> f64 {
        let omega = f * PI;
        let (re, im) = h.iter().enumerate().fold((0.0_f64, 0.0_f64), |(re, im), (k, &hk)| {
            (re + hk * (k as f64 * omega).cos(), im - hk * (k as f64 * omega).sin())
        });
        (re * re + im * im).sqrt()
    }

    #[test]
    fn test_remez_length() {
        let bands = vec![0.0, 0.3, 0.35, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let h = remez(65, &bands, &desired, None, None, None).expect("remez failed");
        assert_eq!(h.len(), 65);
    }

    #[test]
    fn test_remez_symmetric() {
        let bands = vec![0.0, 0.3, 0.35, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let h = remez(65, &bands, &desired, None, None, None).expect("remez failed");
        for i in 0..h.len() / 2 {
            assert!(
                (h[i] - h[h.len() - 1 - i]).abs() < 1e-12,
                "filter is not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_remez_lowpass_frequency_response() {
        // Design a 65-tap lowpass filter: passband [0, 0.3], stopband [0.35, 1.0]
        let bands = vec![0.0, 0.3, 0.35, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let weights = vec![1.0, 10.0]; // 1 weight per band
        let h = remez(65, &bands, &desired, Some(&weights), Some(25), None)
            .expect("remez failed");

        // Passband: gain should be close to 1 (≥ -1 dB ≈ 0.89)
        let gain_dc = freq_mag(&h, 0.0);
        let gain_mid = freq_mag(&h, 0.15);
        let gain_edge = freq_mag(&h, 0.28);
        assert!(
            gain_dc > 0.85,
            "DC gain too low: {gain_dc:.4}"
        );
        assert!(
            gain_mid > 0.85,
            "Passband gain too low at f=0.15: {gain_mid:.4}"
        );
        assert!(
            gain_edge > 0.80,
            "Passband gain too low at f=0.28: {gain_edge:.4}"
        );

        // Stopband: gain should be small (≤ -20 dB ≈ 0.1)
        let gain_stop1 = freq_mag(&h, 0.5);
        let gain_stop2 = freq_mag(&h, 0.75);
        let gain_stop3 = freq_mag(&h, 1.0);
        assert!(
            gain_stop1 < 0.15,
            "Stopband gain too high at f=0.5: {gain_stop1:.4}"
        );
        assert!(
            gain_stop2 < 0.15,
            "Stopband gain too high at f=0.75: {gain_stop2:.4}"
        );
        assert!(
            gain_stop3 < 0.15,
            "Stopband gain too high at f=1.0: {gain_stop3:.4}"
        );
    }

    #[test]
    fn test_remez_bandpass_frequency_response() {
        // Design a 101-tap bandpass filter
        // Stopband 1: [0, 0.2], Passband: [0.25, 0.45], Stopband 2: [0.5, 1.0]
        let bands = vec![0.0, 0.2, 0.25, 0.45, 0.5, 1.0];
        let desired = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let weights = vec![10.0, 1.0, 10.0]; // 1 weight per band
        let h = remez(101, &bands, &desired, Some(&weights), Some(25), None)
            .expect("remez failed");

        // Passband gain should be close to 1
        let gain_pass = freq_mag(&h, 0.35);
        assert!(
            gain_pass > 0.8,
            "Bandpass passband gain too low: {gain_pass:.4}"
        );

        // Stopband gains should be small
        let gain_low = freq_mag(&h, 0.05);
        let gain_high = freq_mag(&h, 0.75);
        assert!(
            gain_low < 0.2,
            "Lower stopband gain too high: {gain_low:.4}"
        );
        assert!(
            gain_high < 0.2,
            "Upper stopband gain too high: {gain_high:.4}"
        );
    }
}
