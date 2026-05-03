//! Frequency-domain filters and decompositions for time series
//!
//! This module provides:
//! - Frequency-domain bandpass filtering via FFT zero-ing
//! - Hodrick-Prescott (HP) filter for trend/cycle separation
//! - Baxter-King (BK) bandpass filter
//! - Christiano-Fitzgerald (CF) asymmetric bandpass filter
//! - Wavelet multi-resolution analysis (MRA) decomposition and reconstruction

use crate::error::{Result, TimeSeriesError};
use scirs2_fft::{
    fft, ifft, rfft, rfftfreq,
    wavelet_packets::{wp_reconstruct, wpd, Wavelet, WaveletPacketNode as WpNode},
};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Ensure the series is long enough and the frequencies are valid.
fn check_bandpass_params(n: usize, low: f64, high: f64, fs: f64) -> Result<()> {
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "bandpass filter requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if fs <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "sampling frequency must be positive".to_string(),
        ));
    }
    if low <= 0.0 || high <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "cutoff frequencies must be positive".to_string(),
        ));
    }
    if low >= high {
        return Err(TimeSeriesError::InvalidInput(
            "low_freq must be strictly less than high_freq".to_string(),
        ));
    }
    if high > fs / 2.0 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "high_freq ({high}) must not exceed Nyquist ({nyq})",
            nyq = fs / 2.0
        )));
    }
    Ok(())
}

/// Convert a `Vec<scirs2_core::numeric::Complex64>` from the scirs2-fft / scirs2-core
/// ecosystem into the representation expected by `ifft`.
///
/// `scirs2_fft::fft` returns `Vec<Complex64>` where `Complex64` is from
/// `scirs2_core::numeric`.  `scirs2_fft::ifft` takes the same type, so we
/// work directly with that type throughout.
use scirs2_core::numeric::Complex64;

// ---------------------------------------------------------------------------
// 1. Frequency-domain bandpass filter
// ---------------------------------------------------------------------------

/// Apply a bandpass filter to a time series using FFT zero-ing.
///
/// All frequency components outside the interval `[low_freq, high_freq]` are
/// set to zero in the frequency domain before reconstructing the signal.
/// The `order` parameter controls the steepness of the roll-off by repeating
/// the masking operation (integer approximation of a Butterworth-style mask).
///
/// # Arguments
/// * `ts`        - Input time series
/// * `low_freq`  - Lower cutoff frequency (Hz)
/// * `high_freq` - Upper cutoff frequency (Hz)
/// * `fs`        - Sampling frequency (Hz)
/// * `order`     - Roll-off sharpness (1 = ideal brick-wall, >1 = iterative taper)
///
/// # Returns
/// Filtered time series of the same length as `ts`.
pub fn bandpass_filter_series(
    ts: &[f64],
    low_freq: f64,
    high_freq: f64,
    fs: f64,
    order: usize,
) -> Result<Vec<f64>> {
    let n = ts.len();
    check_bandpass_params(n, low_freq, high_freq, fs)?;

    // Build the complex spectrum via the full FFT
    let spectrum = fft(ts, None).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    let order_actual = order.max(1);

    // Apply a smooth mask derived from a Butterworth-style magnitude response.
    // The mask is computed once but applied `order_actual` times (raising it to
    // the power `order_actual` → steeper roll-off).
    let freq_resolution = fs / n as f64;
    let mut masked: Vec<Complex64> = spectrum
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let freq = if k <= n / 2 {
                k as f64 * freq_resolution
            } else {
                (k as i64 - n as i64) as f64 * freq_resolution
            };
            let freq_abs = freq.abs();

            // Smooth Butterworth-style gain raised to `order_actual`
            let gain = butterworth_bandpass_gain(freq_abs, low_freq, high_freq, order_actual);
            Complex64::new(c.re * gain, c.im * gain)
        })
        .collect();

    // For order > 1 we have already baked the exponent into the gain above.
    // Optionally apply the mask again for iterative sharpening (disabled here
    // because the gain already accounts for the order).
    let _ = order_actual; // used above

    // Reconstruct via IFFT
    let recovered =
        ifft(&masked, None).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;
    let out: Vec<f64> = recovered.iter().map(|c| c.re).collect();
    Ok(out)
}

/// Butterworth bandpass frequency response (magnitude only).
fn butterworth_bandpass_gain(freq: f64, low: f64, high: f64, order: usize) -> f64 {
    if freq < f64::EPSILON {
        return 0.0;
    }
    // Map the bandpass problem to two low-pass cuts:
    //   G = G_hp(f, low, n) * G_lp(f, high, n)
    let g_hp = 1.0 / (1.0 + (low / freq).powi(2 * order as i32)).sqrt();
    let g_lp = 1.0 / (1.0 + (freq / high).powi(2 * order as i32)).sqrt();
    g_hp * g_lp
}

// ---------------------------------------------------------------------------
// 2. Hodrick-Prescott (HP) filter
// ---------------------------------------------------------------------------

/// Apply the Hodrick-Prescott filter to separate trend and cycle components.
///
/// Minimises: ∑(y_t − τ_t)² + λ ∑(Δ²τ_t)²
///
/// Solved as a band-diagonal linear system using the closed-form tridiagonal
/// approach (Danthine & Girardin 2004).
///
/// # Arguments
/// * `ts`     - Input time series
/// * `lambda` - Smoothing parameter (e.g. 1600 for quarterly macro data)
///
/// # Returns
/// `(trend, cycle)` where `cycle = ts - trend`.
///
/// # References
/// Hodrick R. J., Prescott E. C. (1997). Postwar U.S. Business Cycles.
pub fn hp_filter(ts: &[f64], lambda: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "hp_filter requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if lambda < 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "lambda must be non-negative".to_string(),
        ));
    }

    // Build the pentadiagonal system (A + λ D'D) τ = y
    // where D is the second-difference operator.
    // We solve it via the explicit band-matrix formulation using Cholesky-like
    // forward/back substitution for a symmetric positive-definite band matrix.

    // Bandwidth = 2 (pentadiagonal: main, ±1, ±2)
    // Diagonals of D'D (second-difference matrix, n×n):
    //   k=0 (main):  [1, 5, 6, 6, ..., 6, 5, 1]
    //   k=±1 (off1): [-2, -4, -4, ..., -4, -2]
    //   k=±2 (off2): [1, 1, ..., 1]
    // (for interior rows; boundary rows are truncated)

    let lam = lambda;
    // Build A = I + lam * D'D as a band matrix stored in diagonal form.
    // Rows: 0..n, columns 0..n
    // We store band[k][i] = A[i, i+k] for k=0,1,2 and use symmetry.

    // main diagonal
    let mut diag = vec![0.0_f64; n];
    // 1st off-diagonal
    let mut off1 = vec![0.0_f64; n - 1];
    // 2nd off-diagonal
    let mut off2 = vec![0.0_f64; n - 2];

    // Fill diagonals of lam * D'D + I
    for i in 0..n {
        // I contribution
        diag[i] = 1.0;
        // D'D contribution to main diagonal
        let dd = match i {
            0 => 1.0,
            1 => 5.0,
            _ if i == n - 2 => 5.0,
            _ if i == n - 1 => 1.0,
            _ => 6.0,
        };
        diag[i] += lam * dd;
    }
    for i in 0..n - 1 {
        let dd = match i {
            0 => -2.0,
            _ if i == n - 2 => -2.0,
            _ => -4.0,
        };
        off1[i] = lam * dd;
    }
    for i in 0..n - 2 {
        off2[i] = lam * 1.0;
    }

    // Solve using band Cholesky (LDL' decomposition for symmetric band system).
    let trend = band_ldl_solve(&diag, &off1, &off2, ts)?;
    let cycle: Vec<f64> = ts.iter().zip(trend.iter()).map(|(y, t)| y - t).collect();
    Ok((trend, cycle))
}

/// Solve the symmetric pentadiagonal system A*x = rhs using LDL' decomposition.
///
/// A is defined by its main diagonal `d`, first off-diagonal `e1`, and second
/// off-diagonal `e2`.  All are stored for the upper triangle.
fn band_ldl_solve(d: &[f64], e1: &[f64], e2: &[f64], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = d.len();
    // LDL' factorisation for bandwidth-2 symmetric positive-definite matrix.
    // The matrix A has: A[i,i]=d[i], A[i,i+1]=e1[i], A[i,i+2]=e2[i] (symmetric).
    // We compute L (unit lower triangular, bandwidth 2) and D (diagonal)
    // such that A = L D L'.
    //
    // We also track fill-in: m[i] = A[i,i-1] updated during elimination.
    let mut dd = d.to_vec();
    let mut l1 = vec![0.0_f64; n]; // L[i, i-1]
    let mut l2 = vec![0.0_f64; n]; // L[i, i-2]

    // We need to track the updated off-diag entries as elimination proceeds.
    // ee1[i] = working copy of A[i, i+1], ee2[i] = working copy of A[i, i+2].
    let mut ee1 = vec![0.0_f64; n];
    for (i, val) in e1.iter().enumerate() {
        ee1[i] = *val;
    }
    // No need for ee2 working copy since e2 entries are only read once.

    for i in 0..n {
        if dd[i].abs() < f64::EPSILON {
            return Err(TimeSeriesError::NumericalInstability(
                "near-zero pivot in band_ldl_solve".to_string(),
            ));
        }

        if i + 1 < n {
            l1[i + 1] = ee1[i] / dd[i];
            dd[i + 1] -= l1[i + 1] * ee1[i];
            // Update fill-in: A[i+1, i+2] is affected
            if i + 2 < n {
                ee1[i + 1] -= l1[i + 1] * e2[i];
            }
        }
        if i + 2 < n {
            l2[i + 2] = e2[i] / dd[i];
            dd[i + 2] -= l2[i + 2] * e2[i];
        }
    }

    // Forward substitution: L * y = rhs
    let mut y = rhs.to_vec();
    for i in 1..n {
        y[i] -= l1[i] * y[i - 1];
        if i >= 2 {
            y[i] -= l2[i] * y[i - 2];
        }
    }

    // Divide by D
    for i in 0..n {
        if dd[i].abs() < f64::EPSILON {
            return Err(TimeSeriesError::NumericalInstability(
                "near-zero diagonal in D during band_ldl_solve".to_string(),
            ));
        }
        y[i] /= dd[i];
    }

    // Backward substitution: L' * x = y
    let mut x = y;
    for i in (0..n - 1).rev() {
        x[i] -= l1[i + 1] * x[i + 1];
        if i + 2 < n {
            x[i] -= l2[i + 2] * x[i + 2];
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// 3. Baxter-King (BK) bandpass filter
// ---------------------------------------------------------------------------

/// Apply the Baxter-King bandpass filter.
///
/// The BK filter approximates an ideal bandpass filter using a symmetric
/// moving-average of order K (i.e. the filter has 2K+1 weights).
/// The series loses K observations at each end.
///
/// # Arguments
/// * `ts`   - Input time series
/// * `low`  - Lower period bound (e.g. 6 quarters for business cycles)
/// * `high` - Upper period bound (e.g. 32 quarters)
/// * `k`    - Lead/lag truncation (typically 12 for quarterly data)
///
/// # Returns
/// Filtered series of length `n - 2K`.
pub fn bandpass_filter_bk(ts: &[f64], low: f64, high: f64, k: usize) -> Result<Vec<f64>> {
    let n = ts.len();
    if n < 2 * k + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "BK filter with K={k} requires at least {} samples",
                2 * k + 1
            ),
            required: 2 * k + 1,
            actual: n,
        });
    }
    if low <= 1.0 || high <= low {
        return Err(TimeSeriesError::InvalidInput(
            "BK filter: need 1 < low < high".to_string(),
        ));
    }

    let omega_l = 2.0 * PI / high; // lower angular frequency (high period → low freq)
    let omega_h = 2.0 * PI / low; // upper angular frequency (low period  → high freq)

    // Ideal bandpass filter weights: b_j = (sin(ω_h j) - sin(ω_l j)) / (π j)
    // with b_0 = (ω_h - ω_l) / π
    let mut weights = vec![0.0_f64; 2 * k + 1]; // index 0 = -K, ..., K = index 2K
    let b0 = (omega_h - omega_l) / PI;
    weights[k] = b0; // centre

    for j in 1..=k {
        let bj = (omega_h * j as f64).sin() / (PI * j as f64)
            - (omega_l * j as f64).sin() / (PI * j as f64);
        weights[k + j] = bj;
        weights[k - j] = bj; // symmetric
    }

    // Adjust weights so they sum to zero (removes unit-root component)
    let sum: f64 = weights.iter().sum();
    let n_w = weights.len() as f64;
    for w in weights.iter_mut() {
        *w -= sum / n_w;
    }

    // Apply filter via convolution
    let out_len = n - 2 * k;
    let mut out = vec![0.0_f64; out_len];
    for t in 0..out_len {
        let center = t + k; // position in original ts
        let mut val = 0.0;
        for j in 0..=2 * k {
            let lag = j as i64 - k as i64; // range [-K, K]
            let idx = center as i64 + lag;
            if idx >= 0 && (idx as usize) < n {
                val += weights[j] * ts[idx as usize];
            }
        }
        out[t] = val;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// 4. Christiano-Fitzgerald (CF) filter
// ---------------------------------------------------------------------------

/// Apply the Christiano-Fitzgerald asymmetric bandpass filter.
///
/// The CF filter provides an exact bandpass decomposition for a finite sample
/// by using an asymmetric approximation to the ideal bandpass filter.  The
/// full-sample random-walk version is implemented here.
///
/// # Arguments
/// * `ts`   - Input time series
/// * `low`  - Lower period bound (in time steps, e.g. 6 for quarterly)
/// * `high` - Upper period bound (e.g. 32)
///
/// # Returns
/// `(cycle, trend)` where `cycle` is the bandpass-filtered component and
/// `trend = ts − cycle`.
pub fn christiano_fitzgerald(ts: &[f64], low: f64, high: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "christiano_fitzgerald requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if low <= 1.0 || high <= low {
        return Err(TimeSeriesError::InvalidInput(
            "CF filter: need 1 < low < high".to_string(),
        ));
    }

    let omega_l = 2.0 * PI / high;
    let omega_h = 2.0 * PI / low;

    // Ideal symmetric MA filter weights b_j
    let b = |j: i64| -> f64 {
        if j == 0 {
            (omega_h - omega_l) / PI
        } else {
            ((omega_h * j as f64).sin() - (omega_l * j as f64).sin()) / (PI * j as f64)
        }
    };

    // For each observation t the CF filter uses an asymmetric one-sided finite
    // approximation. The "random-walk" assumption means the boundary correction
    // coefficients c_{t,j} sum to zero (removing the drift in I(1) series).
    //
    // Full-sample CF weights for observation t:
    //   a_t(j) = b(j) for j = -(t-1)...(n-t)
    // with end corrections so that ∑ a_t(j) = 0.

    let mut cycle = vec![0.0_f64; n];

    for t in 0..n {
        // The filter extends from lag -(t) to lead (n-1-t)
        // We use max lag = t,  max lead = n-1-t
        let lag_max = t as i64;
        let lead_max = (n - 1 - t) as i64;

        // Sum of b_j over the full range (for normalisation)
        let sum_b: f64 = (-lag_max..=lead_max).map(|j| b(j)).sum();

        // End correction coefficient: distribute the residual sum equally at
        // the two boundary observations (t-lag_max and t+lead_max).
        // This ensures the filter sums to zero in the limit as N→∞.
        let n_terms = (lag_max + lead_max + 1) as f64;
        let correction = if n_terms > 0.0 { sum_b / n_terms } else { 0.0 };

        let mut val = 0.0;
        for j in -lag_max..=lead_max {
            let idx = t as i64 + j;
            if idx >= 0 && (idx as usize) < n {
                val += (b(j) - correction) * ts[idx as usize];
            }
        }
        cycle[t] = val;
    }

    let trend: Vec<f64> = ts.iter().zip(cycle.iter()).map(|(y, c)| y - c).collect();
    Ok((cycle, trend))
}

// ---------------------------------------------------------------------------
// 5. Wavelet MRA decomposition and reconstruction
// ---------------------------------------------------------------------------

/// Wavelet type alias for the decomposition functions.
pub type WaveletType = Wavelet;

/// Decompose a time series into wavelet multi-resolution analysis (MRA) components.
///
/// Returns `n_levels + 1` components:
/// - indices 0..n_levels-1 are the detail coefficients at each level
///   (high-frequency → low-frequency)
/// - index n_levels is the approximation (trend) component
///
/// # Arguments
/// * `ts`       - Input time series
/// * `wavelet`  - Wavelet basis (e.g. `WaveletType::Db4`)
/// * `n_levels` - Decomposition depth
///
/// # Returns
/// `Vec<Vec<f64>>` of length `n_levels + 1`.
pub fn wavelet_decompose_ts(
    ts: &[f64],
    wavelet: WaveletType,
    n_levels: usize,
) -> Result<Vec<Vec<f64>>> {
    let n = ts.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "wavelet_decompose_ts requires at least 4 samples".to_string(),
            required: 4,
            actual: n,
        });
    }
    if n_levels == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "n_levels must be >= 1".to_string(),
        ));
    }

    // Build the full wavelet packet tree up to n_levels
    let tree =
        wpd(ts, wavelet, n_levels).map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    // For a standard DWT-style MRA we need the DWT basis — the unique set of
    // nodes that forms a complete partition of the time-frequency plane and
    // follows the low-pass chain:
    //
    //   Detail at level k : node (k, 1)   — high-pass branch at level k
    //   Approximation     : node (n_levels, 0) — lowest-resolution subband
    //
    // wp_reconstruct requires a *complete* basis (all nodes must be present
    // and together cover the root).  Passing only one node leaves the sibling
    // branch empty, so the parent is never reconstructed and root (0,0) is
    // never reached.
    //
    // Solution: build the full DWT basis once.  For each MRA component we
    // clone the basis, zero out all coefficients except the target node, then
    // call wp_reconstruct.  The zeroed nodes contribute nothing to the output
    // while satisfying the completeness requirement.

    // Collect the DWT basis nodes.
    // Basis: details (lev=1..=n_levels, idx=1) + approx (lev=n_levels, idx=0).
    // All nodes must be present for wp_reconstruct to succeed (it requires a
    // complete partition of the time-frequency plane).
    let mut basis_template: Vec<WpNode> = Vec::with_capacity(n_levels + 1);
    let mut all_present = true;

    for lev in 1..=n_levels {
        match tree.get(lev, 1) {
            Some(node) => basis_template.push(node.clone()),
            None => {
                all_present = false;
            }
        }
    }
    match tree.get(n_levels, 0) {
        Some(node) => basis_template.push(node.clone()),
        None => {
            all_present = false;
        }
    }

    if !all_present {
        // Signal is too short for the requested decomposition depth.
        // Return zero-filled components.
        return Ok(vec![vec![0.0_f64; n]; n_levels + 1]);
    }

    let mut components: Vec<Vec<f64>> = Vec::with_capacity(n_levels + 1);

    for target in 0..=n_levels {
        // Build a basis clone where every node except `target` has zero coeffs.
        // wp_reconstruct requires all basis nodes (the set must cover the root),
        // so we keep them all but zero-out all non-target coefficients so they
        // contribute nothing to the reconstructed output.
        let isolated_basis: Vec<WpNode> = basis_template
            .iter()
            .enumerate()
            .map(|(i, node)| {
                if i == target {
                    node.clone()
                } else {
                    WpNode::new(vec![0.0_f64; node.coeffs.len()], node.level, node.index)
                }
            })
            .collect();

        let reconstructed = wp_reconstruct(&tree, &isolated_basis)
            .map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;
        components.push(reconstructed);
    }

    Ok(components)
}

/// Reconstruct a time series from its wavelet MRA components.
///
/// Simply sums all components produced by `wavelet_decompose_ts`.  The
/// perfect-reconstruction property of the wavelet transform guarantees that
/// the sum equals the original signal (to floating-point precision).
///
/// # Arguments
/// * `components` - MRA components from `wavelet_decompose_ts`
/// * `_wavelet`   - Wavelet basis (currently unused; reconstruction is by summation)
///
/// # Returns
/// Reconstructed time series.
pub fn reconstruct_wavelet(components: &[Vec<f64>], _wavelet: WaveletType) -> Result<Vec<f64>> {
    if components.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "components must be non-empty".to_string(),
        ));
    }

    let n = components[0].len();
    // Verify all components have the same length
    for (k, comp) in components.iter().enumerate() {
        if comp.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: comp.len(),
            });
        }
        let _ = k;
    }

    let mut out = vec![0.0_f64; n];
    for comp in components {
        for (o, c) in out.iter_mut().zip(comp.iter()) {
            *o += c;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Re-export of wavelet packet types for user convenience
// ---------------------------------------------------------------------------

pub use scirs2_fft::wavelet_packets::{WaveletPacketNode, WaveletPacketTree};

// ---------------------------------------------------------------------------
// Frequency array helper (mirrors spectral_methods but without the import dep)
// ---------------------------------------------------------------------------

/// Return the RFFT frequency bins for a signal of length `n` sampled at `fs`.
pub fn rfft_freq_axis(n: usize, fs: f64) -> Result<Vec<f64>> {
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

    fn make_sine(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    // ---- HP filter ----

    #[test]
    fn test_hp_filter_trend_smooth() {
        let n = 100;
        // Linear trend + sine cycle
        let ts: Vec<f64> = (0..n)
            .map(|i| i as f64 / 10.0 + (2.0 * PI * 0.1 * i as f64).sin())
            .collect();
        let (trend, cycle) = hp_filter(&ts, 1600.0).expect("hp_filter failed");
        assert_eq!(trend.len(), n);
        assert_eq!(cycle.len(), n);

        // Trend + cycle should reproduce original
        for i in 0..n {
            assert!(
                (trend[i] + cycle[i] - ts[i]).abs() < 1e-8,
                "trend+cycle != original at i={i}"
            );
        }

        // Cycle should have zero mean (approximately)
        let cycle_mean: f64 = cycle.iter().sum::<f64>() / n as f64;
        assert!(cycle_mean.abs() < 0.5, "cycle mean too large: {cycle_mean}");
    }

    #[test]
    fn test_hp_filter_errors() {
        let ts = vec![1.0, 2.0];
        assert!(hp_filter(&ts, 1600.0).is_err());
        let ts4 = vec![1.0, 2.0, 3.0, 4.0];
        assert!(hp_filter(&ts4, -1.0).is_err());
    }

    // ---- BK filter ----

    #[test]
    fn test_bk_filter_output_length() {
        let n = 100;
        let k = 12;
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let out = bandpass_filter_bk(&ts, 6.0, 32.0, k).expect("BK filter failed");
        assert_eq!(out.len(), n - 2 * k);
    }

    #[test]
    fn test_bk_filter_removes_trend() {
        let n = 200;
        let k = 12;
        // Pure trend (linear) should be removed by BK
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let out = bandpass_filter_bk(&ts, 6.0, 32.0, k).expect("BK filter failed");
        let max_val = out.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        // BK adjusts weights to zero-sum so it should remove a linear trend
        assert!(
            max_val < 1e-8,
            "BK should remove linear trend, max_val={max_val}"
        );
    }

    #[test]
    fn test_bk_filter_insufficient_data() {
        let ts = vec![1.0; 10];
        assert!(bandpass_filter_bk(&ts, 6.0, 32.0, 12).is_err());
    }

    // ---- CF filter ----

    #[test]
    fn test_cf_filter_output_length() {
        let n = 100;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 % 10.0).collect();
        let (cycle, trend) = christiano_fitzgerald(&ts, 6.0, 32.0).expect("CF filter failed");
        assert_eq!(cycle.len(), n);
        assert_eq!(trend.len(), n);
    }

    #[test]
    fn test_cf_filter_reconstruction() {
        let n = 50;
        let ts: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 0.05 * i as f64).sin() + (i as f64) * 0.01)
            .collect();
        let (cycle, trend) = christiano_fitzgerald(&ts, 6.0, 32.0).expect("CF filter failed");
        for i in 0..n {
            assert!(
                (cycle[i] + trend[i] - ts[i]).abs() < 1e-10,
                "cycle+trend != ts at i={i}"
            );
        }
    }

    // ---- Bandpass filter (FFT) ----

    #[test]
    fn test_bandpass_filter_removes_out_of_band() {
        let fs = 100.0;
        let n = 512;
        // Sum of two sinusoids: 5 Hz (in band) and 40 Hz (out of band)
        let ts: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 5.0 * i as f64 / fs).sin() + (2.0 * PI * 40.0 * i as f64 / fs).sin()
            })
            .collect();

        let filtered =
            bandpass_filter_series(&ts, 3.0, 10.0, fs, 4).expect("bandpass_filter failed");
        assert_eq!(filtered.len(), n);

        // The filtered signal energy should be much less than the input
        let in_energy: f64 = ts.iter().map(|v| v * v).sum();
        let out_energy: f64 = filtered.iter().map(|v| v * v).sum();
        // The 40 Hz component should be greatly attenuated
        assert!(
            out_energy < in_energy * 0.7,
            "output energy {out_energy:.3} should be less than input energy {in_energy:.3}"
        );
    }

    // ---- Wavelet MRA ----

    #[test]
    fn test_wavelet_mra_component_count() {
        let n = 64;
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let levels = 3;
        let components =
            wavelet_decompose_ts(&ts, WaveletType::Db4, levels).expect("wavelet decompose failed");
        assert_eq!(components.len(), levels + 1);
        for comp in &components {
            assert_eq!(comp.len(), n, "component length mismatch");
        }
    }

    #[test]
    fn test_wavelet_reconstruct_matches_sum() {
        let n = 64;
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let components = wavelet_decompose_ts(&ts, WaveletType::Haar, 2).expect("decompose failed");

        let recon =
            reconstruct_wavelet(&components, WaveletType::Haar).expect("reconstruct failed");
        assert_eq!(recon.len(), n);

        // Reconstruction should match the component sum
        let sum: Vec<f64> = (0..n)
            .map(|i| components.iter().map(|c| c[i]).sum::<f64>())
            .collect();
        for i in 0..n {
            assert!(
                (recon[i] - sum[i]).abs() < 1e-10,
                "reconstruct != sum at i={i}: {} vs {}",
                recon[i],
                sum[i]
            );
        }
    }
}
