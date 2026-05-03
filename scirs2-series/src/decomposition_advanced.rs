//! Advanced Time Series Decomposition Methods
//!
//! This module provides three advanced decomposition algorithms:
//!
//! 1. **MSTL** — Multiple Seasonal-Trend decomposition using LOESS
//!    (Bandara, Hyndman & Bergmeir, 2021).  Handles time series with several
//!    nested seasonal periods (e.g. hourly data with daily + weekly seasonality).
//!
//! 2. **EMD** — Empirical Mode Decomposition (Huang et al., 1998).
//!    An adaptive, data-driven method that extracts Intrinsic Mode Functions
//!    (IMFs) via a sifting process using cubic spline envelope interpolation.
//!    The last residual captures the overall trend.
//!
//! 3. **SSA** — Singular Spectrum Analysis (Broomhead & King, 1986; Golyandina
//!    et al., 2001).  Embeds the series in a trajectory (Hankel) matrix, decomposes
//!    via SVD, groups components, and reconstructs via diagonal averaging.
//!
//! # References
//!
//! - Huang et al. (1998). "The empirical mode decomposition and the Hilbert spectrum
//!   for nonlinear and non-stationary time series analysis." *Proc. R. Soc. Lond. A*.
//! - Broomhead, D.S. & King, G.P. (1986). "Extracting qualitative dynamics from
//!   experimental data." *Physica D*, 20(2-3), 217-236.
//! - Bandara, K., Hyndman, R.J. & Bergmeir, C. (2021). "MSTL: A Seasonal-Trend
//!   Decomposition Algorithm for Time Series with Multiple Seasonal Patterns."
//!   *arXiv:2107.13462*.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// MSTL — Multiple Seasonal-Trend decomposition using LOESS
// ---------------------------------------------------------------------------

/// Result of Multiple Seasonal-Trend decomposition using LOESS (MSTL).
#[derive(Debug, Clone)]
pub struct MstlResult {
    /// Trend component (length = input length).
    pub trend: Vec<f64>,
    /// Seasonal components: one per period supplied (each length = input length).
    pub seasonal: Vec<Vec<f64>>,
    /// Remainder / residual component (length = input length).
    pub remainder: Vec<f64>,
}

/// Perform MSTL decomposition on a time series with multiple seasonal periods.
///
/// The algorithm iterates over the list of periods in order of increasing
/// period length. In each pass it:
/// 1. Removes the contribution of all _other_ seasonal components and the current
///    trend estimate from the data.
/// 2. Extracts the target seasonal component with a seasonal LOESS smoother.
/// 3. After all seasonal components have been refined, estimates the trend with a
///    non-seasonal LOESS smoother on the seasonally-adjusted series.
///
/// # Arguments
/// * `data`    — time series observations
/// * `periods` — list of seasonal periods, e.g. `&[7, 365]` (must each be ≥ 2)
///
/// # Returns
/// [`MstlResult`] with trend, per-period seasonal components, and remainder.
pub fn mstl(data: &[f64], periods: &[usize]) -> Result<MstlResult> {
    let n = data.len();
    if periods.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "MSTL requires at least one seasonal period".to_string(),
        ));
    }
    for &p in periods {
        if p < 2 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "period".to_string(),
                message: format!("Each period must be >= 2, got {}", p),
            });
        }
        if n < 2 * p {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "MSTL requires at least 2*period={} observations for period {}",
                    2 * p,
                    p
                ),
                required: 2 * p,
                actual: n,
            });
        }
    }

    let n_periods = periods.len();
    let mut seasonals: Vec<Vec<f64>> = vec![vec![0.0; n]; n_periods];
    let mut trend = vec![0.0_f64; n];

    let max_iter = 3_usize;

    for _outer in 0..max_iter {
        for k in 0..n_periods {
            let p = periods[k];

            // Seasonally-adjusted series: remove other seasonals + trend
            let adjusted: Vec<f64> = (0..n)
                .map(|t| {
                    let other_seasonal: f64 = seasonals
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| *idx != k)
                        .map(|(_, s)| s[t])
                        .sum();
                    data[t] - trend[t] - other_seasonal
                })
                .collect();

            // Compute seasonal component k via averaging within each seasonal position,
            // then smooth each season's profile with LOESS
            seasonals[k] = extract_seasonal_loess(&adjusted, p, 7);
        }

        // Trend: LOESS of seasonally-adjusted series
        let all_seasonal_sum: Vec<f64> = (0..n)
            .map(|t| seasonals.iter().map(|s| s[t]).sum::<f64>())
            .collect();
        let deseasonalised: Vec<f64> = (0..n).map(|t| data[t] - all_seasonal_sum[t]).collect();

        // Adaptive trend window: ~ 1.5 × n / (longest period), forced odd
        let longest_period = *periods.iter().max().unwrap_or(&1);
        let trend_span = {
            let raw = (1.5 * n as f64 / longest_period as f64).ceil() as usize;
            let raw = raw.max(5).min(n);
            if raw % 2 == 0 {
                raw + 1
            } else {
                raw
            }
        };
        trend = loess_1d(&deseasonalised, trend_span);
    }

    // Remainder
    let all_seasonal_sum: Vec<f64> = (0..n)
        .map(|t| seasonals.iter().map(|s| s[t]).sum::<f64>())
        .collect();
    let remainder: Vec<f64> = (0..n)
        .map(|t| data[t] - trend[t] - all_seasonal_sum[t])
        .collect();

    Ok(MstlResult {
        trend,
        seasonal: seasonals,
        remainder,
    })
}

/// Extract the seasonal component at period `p` from `data` using LOESS.
///
/// 1. For each seasonal position i ∈ 0..p, average all observations at that position.
/// 2. Replicate the resulting pattern across the full series length.
/// 3. Apply LOESS smoothing across the full replicated pattern with window `loess_win`.
fn extract_seasonal_loess(data: &[f64], p: usize, loess_win: usize) -> Vec<f64> {
    let n = data.len();
    let mut season_means = vec![0.0_f64; p];
    let mut season_counts = vec![0_usize; p];

    for (t, &v) in data.iter().enumerate() {
        let pos = t % p;
        season_means[pos] += v;
        season_counts[pos] += 1;
    }
    for i in 0..p {
        if season_counts[i] > 0 {
            season_means[i] /= season_counts[i] as f64;
        }
    }

    // Normalise to zero-mean
    let mean_s = season_means.iter().sum::<f64>() / p as f64;
    for v in &mut season_means {
        *v -= mean_s;
    }

    // Replicate pattern
    let replicated: Vec<f64> = (0..n).map(|t| season_means[t % p]).collect();

    // Smooth with LOESS
    loess_1d(&replicated, loess_win)
}

// ---------------------------------------------------------------------------
// EMD — Empirical Mode Decomposition
// ---------------------------------------------------------------------------

/// Result of Empirical Mode Decomposition (EMD).
#[derive(Debug, Clone)]
pub struct EmdResult {
    /// Intrinsic Mode Functions (IMFs) in order of decreasing frequency.
    pub imfs: Vec<Vec<f64>>,
    /// Residual (trend) after removing all IMFs.
    pub residue: Vec<f64>,
}

/// Perform Empirical Mode Decomposition (EMD) via the sifting algorithm.
///
/// The algorithm extracts IMFs until either:
/// - `max_imfs` IMFs have been extracted, or
/// - the residual is monotone (no local extrema), or
/// - the residual's energy is negligible.
///
/// Each IMF satisfies:
/// 1. The number of extrema and zero-crossings differ by at most 1.
/// 2. The running mean of the upper and lower envelopes is zero.
///
/// # Arguments
/// * `data`     — time series (at least 6 points required)
/// * `max_imfs` — maximum number of IMFs to extract
///
/// # Returns
/// [`EmdResult`] with `imfs` and `residue`.
pub fn emd(data: &[f64], max_imfs: usize) -> Result<EmdResult> {
    let n = data.len();
    if n < 6 {
        return Err(TimeSeriesError::InsufficientData {
            message: "EMD requires at least 6 data points".to_string(),
            required: 6,
            actual: n,
        });
    }
    if max_imfs == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "max_imfs".to_string(),
            message: "max_imfs must be >= 1".to_string(),
        });
    }

    let mut residue = data.to_vec();
    let mut imfs: Vec<Vec<f64>> = Vec::new();

    let max_sift_iter = 50_usize;
    let sd_threshold = 0.2_f64;

    for _ in 0..max_imfs {
        // Stop if residual is essentially monotone
        let (maxima_idx, minima_idx) = find_extrema(&residue);
        if maxima_idx.len() < 2 || minima_idx.len() < 2 {
            break;
        }

        // Sifting process
        let mut h = residue.clone();
        let mut prev_h = h.clone();

        for _sift in 0..max_sift_iter {
            let (max_idx, min_idx) = find_extrema(&h);
            if max_idx.len() < 2 || min_idx.len() < 2 {
                break;
            }

            // Cubic spline envelopes for upper (maxima) and lower (minima)
            let upper = cubic_spline_envelope(&h, &max_idx, true);
            let lower = cubic_spline_envelope(&h, &min_idx, false);

            // Mean envelope
            let mean_env: Vec<f64> = upper
                .iter()
                .zip(lower.iter())
                .map(|(u, l)| (u + l) / 2.0)
                .collect();

            let mut new_h: Vec<f64> = h.iter().zip(mean_env.iter()).map(|(v, m)| v - m).collect();

            // Stopping criterion: standard deviation of successive siftings
            let sd: f64 = prev_h
                .iter()
                .zip(new_h.iter())
                .map(|(p, c)| (p - c).powi(2) / (p.powi(2) + 1e-16))
                .sum();

            prev_h = h.clone();
            h = new_h.clone();

            if sd < sd_threshold {
                break;
            }
        }

        // Subtract IMF from residue
        let imf = h;
        let new_residue: Vec<f64> = residue.iter().zip(imf.iter()).map(|(r, i)| r - i).collect();
        imfs.push(imf);
        residue = new_residue;

        // Stop if energy of residue is negligible
        let res_energy: f64 = residue.iter().map(|&v| v * v).sum();
        let data_energy: f64 = data.iter().map(|&v| v * v).sum();
        if data_energy > 0.0 && res_energy / data_energy < 1e-10 {
            break;
        }
    }

    Ok(EmdResult { imfs, residue })
}

/// Find local maxima and minima indices in a slice.
///
/// An interior point is a local maximum (minimum) if it is strictly greater
/// (less) than both its neighbours.  End points are excluded.
fn find_extrema(data: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = data.len();
    let mut maxima = Vec::new();
    let mut minima = Vec::new();
    for i in 1..n - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            maxima.push(i);
        }
        if data[i] < data[i - 1] && data[i] < data[i + 1] {
            minima.push(i);
        }
    }
    (maxima, minima)
}

/// Compute a cubic spline envelope at all points of `data`, using the given
/// extreme indices as knots.
///
/// The boundary is extended by mirroring the first/last extreme to ensure the
/// spline covers the full domain `[0, n-1]`.
///
/// When `is_upper` is `true` the envelope connects local maxima; when `false`
/// it connects local minima.
fn cubic_spline_envelope(data: &[f64], extreme_idx: &[usize], _is_upper: bool) -> Vec<f64> {
    let n = data.len();
    if extreme_idx.len() < 2 {
        return data.to_vec();
    }

    // Build knot arrays with boundary extension
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    // Mirror first extreme to t = 0 if not already at 0
    if extreme_idx[0] > 0 {
        let mirrored_x = 0_f64;
        let mirrored_y = data[extreme_idx[0]];
        xs.push(mirrored_x);
        ys.push(mirrored_y);
    }

    for &idx in extreme_idx {
        xs.push(idx as f64);
        ys.push(data[idx]);
    }

    // Mirror last extreme to t = n-1 if not already there
    let last_ext = *extreme_idx.last().unwrap_or(&0);
    if last_ext < n - 1 {
        xs.push((n - 1) as f64);
        ys.push(data[last_ext]);
    }

    // Evaluate natural cubic spline at all integer positions
    natural_cubic_spline_eval(&xs, &ys, n)
}

// ---------------------------------------------------------------------------
// Natural cubic spline implementation
// ---------------------------------------------------------------------------

/// Evaluate a natural cubic spline defined by knots (xs, ys) at integer
/// positions 0..n.
///
/// Uses the standard tridiagonal system for computing second derivatives
/// (Thomas algorithm).
fn natural_cubic_spline_eval(xs: &[f64], ys: &[f64], n: usize) -> Vec<f64> {
    let m = xs.len();
    if m == 0 {
        return vec![0.0; n];
    }
    if m == 1 {
        return vec![ys[0]; n];
    }
    if m == 2 {
        // Linear interpolation / extrapolation
        let dx = xs[1] - xs[0];
        let slope = if dx.abs() > 1e-14 {
            (ys[1] - ys[0]) / dx
        } else {
            0.0
        };
        return (0..n).map(|t| ys[0] + slope * (t as f64 - xs[0])).collect();
    }

    // Compute natural cubic spline second derivatives via Thomas algorithm
    let k = m - 1; // number of intervals
    let mut h = vec![0.0_f64; k];
    for i in 0..k {
        h[i] = xs[i + 1] - xs[i];
        if h[i].abs() < 1e-14 {
            h[i] = 1e-14; // avoid division by zero
        }
    }

    // Set up tridiagonal system for interior nodes
    let size = m - 2; // interior knots
    if size == 0 {
        // Only two unique knots → linear
        let dx = xs[k] - xs[0];
        let slope = if dx.abs() > 1e-14 {
            (ys[k] - ys[0]) / dx
        } else {
            0.0
        };
        return (0..n).map(|t| ys[0] + slope * (t as f64 - xs[0])).collect();
    }

    // Build second-derivative vector sigma (natural spline: sigma[0] = sigma[m-1] = 0)
    let mut sigma = vec![0.0_f64; m];

    if size == 1 {
        // Only one interior knot — tridiagonal is 1×1, no off-diagonal entries.
        let diag0 = 2.0 * (h[0] + h[1]);
        let rhs0 = 3.0 * ((ys[2] - ys[1]) / h[1] - (ys[1] - ys[0]) / h[0]);
        sigma[1] = if diag0.abs() > 1e-14 {
            rhs0 / diag0
        } else {
            0.0
        };
    } else {
        let mut diag = vec![0.0_f64; size];
        let mut rhs = vec![0.0_f64; size];
        let mut lower = vec![0.0_f64; size - 1];
        let mut upper = vec![0.0_f64; size - 1];

        for i in 0..size {
            // i corresponds to interior node i+1 (1-indexed)
            let idx = i + 1;
            diag[i] = 2.0 * (h[idx - 1] + h[idx]);
            rhs[i] =
                3.0 * ((ys[idx + 1] - ys[idx]) / h[idx] - (ys[idx] - ys[idx - 1]) / h[idx - 1]);
            if i > 0 {
                lower[i - 1] = h[idx - 1];
            }
            if i < size - 1 {
                upper[i] = h[idx];
            }
        }

        // Thomas algorithm (forward sweep)
        let mut c_prime = vec![0.0_f64; size];
        let mut d_prime = vec![0.0_f64; size];

        // For size >= 2, upper has size-1 elements, so upper[0] is valid.
        c_prime[0] = if diag[0].abs() > 1e-14 {
            upper[0] / diag[0]
        } else {
            0.0
        };
        d_prime[0] = if diag[0].abs() > 1e-14 {
            rhs[0] / diag[0]
        } else {
            0.0
        };

        for i in 1..size {
            let denom = diag[i] - lower[i - 1] * c_prime[i - 1];
            if i < size - 1 {
                c_prime[i] = if denom.abs() > 1e-14 {
                    upper[i] / denom
                } else {
                    0.0
                };
            }
            d_prime[i] = if denom.abs() > 1e-14 {
                (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
            } else {
                0.0
            };
        }

        // Back substitution
        sigma[size] = d_prime[size - 1];
        for i in (0..size - 1).rev() {
            sigma[i + 1] = d_prime[i] - c_prime[i] * sigma[i + 2];
        }
    }
    // Natural boundary: sigma[0] = sigma[m-1] = 0

    // Evaluate spline at integer positions
    (0..n)
        .map(|t| {
            let tx = t as f64;
            // Find interval
            let seg = find_segment(xs, tx);
            let i = seg.min(k - 1);
            let dx = xs[i + 1] - xs[i];
            if dx.abs() < 1e-14 {
                return ys[i];
            }
            // Cubic Hermite interpolation using natural spline second derivatives
            // S(x) = a + b*(x-xi) + c*(x-xi)^2 + d*(x-xi)^3
            let dy = ys[i + 1] - ys[i];
            let a_coef = ys[i];
            let b_coef = dy / dx - dx * (2.0 * sigma[i] + sigma[i + 1]) / 6.0;
            let c_coef = sigma[i] / 2.0;
            let d_coef = (sigma[i + 1] - sigma[i]) / (6.0 * dx);
            let xr = tx - xs[i];
            a_coef + b_coef * xr + c_coef * xr * xr + d_coef * xr * xr * xr
        })
        .collect()
}

/// Find the segment index i such that xs[i] <= tx < xs[i+1].
/// Falls back to first/last segment for out-of-range values.
fn find_segment(xs: &[f64], tx: f64) -> usize {
    let k = xs.len();
    if k <= 1 {
        return 0;
    }
    // Binary search
    let mut lo = 0_usize;
    let mut hi = k - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= tx {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// SSA — Singular Spectrum Analysis
// ---------------------------------------------------------------------------

/// Result of Singular Spectrum Analysis (SSA).
#[derive(Debug, Clone)]
pub struct SsaResult {
    /// Reconstructed components in order of decreasing singular value.
    pub components: Vec<Vec<f64>>,
    /// Singular values (eigenvalues of trajectory matrix X·Xᵀ).
    pub eigenvalues: Vec<f64>,
    /// W-correlation matrix (shape `n_components × n_components`).
    pub w_correlations: Vec<Vec<f64>>,
}

/// Perform Singular Spectrum Analysis (SSA) on a time series.
///
/// # Algorithm
///
/// 1. **Embedding**: build the trajectory (Hankel) matrix `X` of shape `(L × K)`
///    where `L = window_length` and `K = n - L + 1`.
/// 2. **SVD**: decompose `X = U · Σ · Vᵀ`. The power-iteration SVD used here
///    approximates the top `n_components` singular triplets.
/// 3. **Grouping**: each component is kept as a rank-1 reconstruction.
/// 4. **Diagonal averaging (Hankelization)**: each rank-1 matrix is anti-diagonally
///    averaged to reconstruct a time series component.
///
/// # Arguments
/// * `data`         — time series (length `n`)
/// * `window_length`— embedding dimension `L` (1 ≤ L ≤ n/2 is typical)
/// * `n_components` — number of leading components to extract
///
/// # Returns
/// [`SsaResult`] with `components`, `eigenvalues`, and `w_correlations`.
pub fn ssa(data: &[f64], window_length: usize, n_components: usize) -> Result<SsaResult> {
    let n = data.len();

    if window_length < 1 || window_length >= n {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window_length".to_string(),
            message: format!(
                "window_length must be in [1, {}], got {}",
                n - 1,
                window_length
            ),
        });
    }
    if n_components == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_components".to_string(),
            message: "n_components must be >= 1".to_string(),
        });
    }

    let l = window_length;
    let k = n - l + 1; // number of columns in trajectory matrix

    // Cap n_components at min(l, k)
    let n_comp = n_components.min(l).min(k);

    // Build trajectory (Hankel) matrix as a flat row-major array
    // X[i][j] = data[i + j]  for i in 0..l, j in 0..k
    let x_mat: Vec<f64> = (0..l)
        .flat_map(|i| (0..k).map(move |j| data[i + j]))
        .collect();

    // Power iteration SVD to get top n_comp singular triplets
    let (u_vecs, sv_vals, v_vecs) = power_iteration_svd(&x_mat, l, k, n_comp);

    // SVD may return fewer components than requested (when residual becomes negligible).
    // We always produce exactly n_comp components, padding with zeros when needed.
    let actual_n_comp = sv_vals.len();

    let mut eigenvalues: Vec<f64> = sv_vals.iter().map(|&s| s * s).collect();
    // Pad eigenvalues to n_comp
    eigenvalues.resize(n_comp, 0.0);

    // Reconstruct each component via rank-1 outer product + diagonal averaging
    let mut components: Vec<Vec<f64>> = (0..actual_n_comp)
        .map(|c| {
            // Rank-1 matrix: r[i][j] = sigma_c * u_c[i] * v_c[j]
            let sigma = sv_vals[c];
            let u_c = &u_vecs[c];
            let v_c = &v_vecs[c];

            // Build rank-1 matrix (l × k) stored row-major
            let rank1: Vec<f64> = (0..l)
                .flat_map(|i| (0..k).map(move |j| sigma * u_c[i] * v_c[j]))
                .collect();

            // Diagonal averaging (Hankelization): average along anti-diagonals
            diagonal_average(&rank1, l, k, n)
        })
        .collect();
    // Pad with zero-components if SVD returned fewer than requested
    while components.len() < n_comp {
        components.push(vec![0.0; n]);
    }

    // Compute w-correlation matrix
    // w-correlation: corr(c_i, c_j) = <c_i, c_j>_w / sqrt(<c_i,c_i>_w * <c_j,c_j>_w)
    // with w-norm where w[t] = min(t+1, L, K, n-t) (triangular weights)
    let w_weights = ssa_w_weights(n, l, k);
    let w_correlations = compute_w_correlations(&components, &w_weights);

    Ok(SsaResult {
        components,
        eigenvalues,
        w_correlations,
    })
}

/// Triangular window weights for SSA w-correlation.
fn ssa_w_weights(n: usize, l: usize, k: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let t1 = t + 1;
            [t1, l, k, n - t].iter().copied().min().unwrap_or(1) as f64
        })
        .collect()
}

/// Compute the SSA w-correlation matrix for the given components.
fn compute_w_correlations(components: &[Vec<f64>], w: &[f64]) -> Vec<Vec<f64>> {
    let nc = components.len();
    let mut mat = vec![vec![0.0_f64; nc]; nc];

    // Precompute weighted norms
    let norms: Vec<f64> = (0..nc)
        .map(|i| {
            components[i]
                .iter()
                .zip(w.iter())
                .map(|(v, &wi)| wi * v * v)
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    for i in 0..nc {
        for j in i..nc {
            let dot: f64 = components[i]
                .iter()
                .zip(components[j].iter())
                .zip(w.iter())
                .map(|((ci, cj), &wi)| wi * ci * cj)
                .sum();
            let denom = norms[i] * norms[j];
            let corr = if denom > 1e-14 { dot / denom } else { 0.0 };
            mat[i][j] = corr;
            mat[j][i] = corr;
        }
    }
    mat
}

/// Diagonal averaging (Hankelization) of an (l × k) matrix to a length-n series.
///
/// Anti-diagonal d = i + j (d in 0..l+k-1) corresponds to series index d.
/// We average all matrix elements on the same anti-diagonal.
fn diagonal_average(mat: &[f64], l: usize, k: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n];
    let mut counts = vec![0_usize; n];

    for i in 0..l {
        for j in 0..k {
            let t = i + j;
            if t < n {
                result[t] += mat[i * k + j];
                counts[t] += 1;
            }
        }
    }

    for t in 0..n {
        if counts[t] > 0 {
            result[t] /= counts[t] as f64;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Power-iteration SVD
// ---------------------------------------------------------------------------

/// Compute the top `n_comp` singular triplets of an `(nrows × ncols)` matrix
/// stored row-major, using the power iteration method (deflation).
///
/// Returns `(U_vecs, singular_values, V_vecs)` where `U_vecs[i]` has length
/// `nrows` and `V_vecs[i]` has length `ncols`.
fn power_iteration_svd(
    mat: &[f64],
    nrows: usize,
    ncols: usize,
    n_comp: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut u_vecs: Vec<Vec<f64>> = Vec::new();
    let mut sv_vals: Vec<f64> = Vec::new();
    let mut v_vecs: Vec<Vec<f64>> = Vec::new();

    // Working copy of matrix for deflation
    let mut residual: Vec<f64> = mat.to_vec();

    let max_iter = 200_usize;
    let tol = 1e-9_f64;

    for _c in 0..n_comp {
        // Initialise v with a deterministic vector
        let mut v: Vec<f64> = (0..ncols).map(|j| ((j + 1) as f64).recip()).collect();
        normalize_vec(&mut v);

        let mut sigma = 0.0_f64;

        for _iter in 0..max_iter {
            // u = A * v  (matrix-vector product)
            let mut u = mat_vec_mul(&residual, nrows, ncols, &v);
            let new_sigma = norm_2(&u);
            if new_sigma < tol {
                break;
            }
            // Normalise u
            for ui in &mut u {
                *ui /= new_sigma;
            }
            // v = Aᵀ * u
            let mut new_v = mat_t_vec_mul(&residual, nrows, ncols, &u);
            let sv = norm_2(&new_v);
            if sv < tol {
                sigma = 0.0;
                break;
            }
            for vi in &mut new_v {
                *vi /= sv;
            }

            let delta: f64 = v
                .iter()
                .zip(new_v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            v = new_v;
            sigma = sv;

            if delta < tol {
                break;
            }
        }

        // Final u
        let mut u = mat_vec_mul(&residual, nrows, ncols, &v);
        let sv_final = norm_2(&u);
        if sv_final < tol {
            break;
        }
        for ui in &mut u {
            *ui /= sv_final;
        }

        // Deflate: residual -= sigma * u * vᵀ
        for i in 0..nrows {
            for j in 0..ncols {
                residual[i * ncols + j] -= sv_final * u[i] * v[j];
            }
        }

        u_vecs.push(u);
        sv_vals.push(sv_final);
        v_vecs.push(v);
    }

    (u_vecs, sv_vals, v_vecs)
}

/// Matrix-vector product: y = A * x, A is nrows × ncols (row-major).
fn mat_vec_mul(a: &[f64], nrows: usize, ncols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            y[i] += a[i * ncols + j] * x[j];
        }
    }
    y
}

/// Transposed matrix-vector product: y = Aᵀ * x.
fn mat_t_vec_mul(a: &[f64], nrows: usize, ncols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            y[j] += a[i * ncols + j] * x[i];
        }
    }
    y
}

/// L2 norm of a vector.
fn norm_2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// In-place L2 normalisation.
fn normalize_vec(v: &mut Vec<f64>) {
    let n = norm_2(v);
    if n > 1e-14 {
        for vi in v.iter_mut() {
            *vi /= n;
        }
    }
}

// ---------------------------------------------------------------------------
// LOESS helper (1-D, equal-spaced)
// ---------------------------------------------------------------------------

/// Lightweight LOESS smoother for equally-spaced data.
///
/// For each point t, fits a local linear regression using the `win` nearest
/// neighbours weighted by the tricube kernel.
fn loess_1d(data: &[f64], win: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let half = (win / 2).max(1).min(n - 1);

    (0..n)
        .map(|t| {
            // Window [lo, hi)
            let lo = t.saturating_sub(half);
            let hi = (t + half + 1).min(n);

            // Maximum distance in this window
            let max_dist = (t as f64 - lo as f64)
                .max(hi as f64 - 1.0 - t as f64)
                .max(1.0);

            // Tricube weights and weighted linear regression
            let mut w_sum = 0.0_f64;
            let mut wt_sum = 0.0_f64;
            let mut wtt_sum = 0.0_f64;
            let mut wy_sum = 0.0_f64;
            let mut wty_sum = 0.0_f64;

            for i in lo..hi {
                let u = (i as f64 - t as f64).abs() / max_dist;
                let u = u.min(1.0);
                let w = tricube(u);
                let ti = i as f64;
                w_sum += w;
                wt_sum += w * ti;
                wtt_sum += w * ti * ti;
                wy_sum += w * data[i];
                wty_sum += w * ti * data[i];
            }

            // Solve 2×2 weighted least squares: [w_sum, wt_sum; wt_sum, wtt_sum] * [a; b] = [wy_sum; wty_sum]
            let det = w_sum * wtt_sum - wt_sum * wt_sum;
            if det.abs() < 1e-14 {
                return if w_sum > 0.0 { wy_sum / w_sum } else { data[t] };
            }
            let a = (wtt_sum * wy_sum - wt_sum * wty_sum) / det;
            let b = (w_sum * wty_sum - wt_sum * wy_sum) / det;
            a + b * t as f64
        })
        .collect()
}

/// Tricube kernel function: (1 - |u|^3)^3 for |u| <= 1, 0 otherwise.
#[inline]
fn tricube(u: f64) -> f64 {
    if u >= 1.0 {
        0.0
    } else {
        let v = 1.0 - u * u * u;
        v * v * v
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── MSTL tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_mstl_output_lengths() {
        let n = 48_usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = mstl(&data, &[4, 12]).expect("mstl failed");
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.seasonal.len(), 2);
        assert_eq!(result.seasonal[0].len(), n);
        assert_eq!(result.seasonal[1].len(), n);
        assert_eq!(result.remainder.len(), n);
    }

    #[test]
    fn test_mstl_components_sum_to_data() {
        let n = 48_usize;
        // Simple trend + two seasonal components
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                t * 0.5 // just trend; seasonals from the decomp
                + (2.0 * std::f64::consts::PI * t / 4.0).sin() * 3.0
                + (2.0 * std::f64::consts::PI * t / 12.0).cos() * 2.0
            })
            .collect();
        let result = mstl(&data, &[4, 12]).expect("mstl failed");
        for t in 0..n {
            let reconstructed = result.trend[t]
                + result.seasonal[0][t]
                + result.seasonal[1][t]
                + result.remainder[t];
            assert!(
                (reconstructed - data[t]).abs() < 1e-6,
                "reconstruction mismatch at t={}: {} vs {}",
                t,
                reconstructed,
                data[t]
            );
        }
    }

    #[test]
    fn test_mstl_single_period() {
        let data: Vec<f64> = (0..24).map(|i| (i as f64).sin()).collect();
        let result = mstl(&data, &[6]).expect("single period mstl failed");
        assert_eq!(result.seasonal.len(), 1);
    }

    #[test]
    fn test_mstl_empty_periods_error() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(mstl(&data, &[]).is_err());
    }

    #[test]
    fn test_mstl_insufficient_data_error() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(mstl(&data, &[4]).is_err());
    }

    // ── EMD tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_emd_imfs_sum_to_original() {
        let n = 64_usize;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (t * 0.5).sin() + (t * 0.1).cos() * 2.0 + t * 0.02
            })
            .collect();
        let result = emd(&data, 5).expect("emd failed");

        // Sum of all IMFs + residue must equal original
        for t in 0..n {
            let sum: f64 = result.imfs.iter().map(|imf| imf[t]).sum::<f64>() + result.residue[t];
            assert!(
                (sum - data[t]).abs() < 1e-8,
                "IMFs + residue != data at t={}: sum={} data={}",
                t,
                sum,
                data[t]
            );
        }
    }

    #[test]
    fn test_emd_produces_at_least_one_imf() {
        let data: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.4).sin() + (i as f64 * 1.1).cos())
            .collect();
        let result = emd(&data, 5).expect("emd failed");
        assert!(result.imfs.len() >= 1, "should produce at least one IMF");
    }

    #[test]
    fn test_emd_output_lengths() {
        let n = 40_usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.3).sin() + i as f64 * 0.01)
            .collect();
        let result = emd(&data, 4).expect("emd failed");
        for (c, imf) in result.imfs.iter().enumerate() {
            assert_eq!(imf.len(), n, "IMF {} length mismatch", c);
        }
        assert_eq!(result.residue.len(), n);
    }

    #[test]
    fn test_emd_insufficient_data() {
        assert!(emd(&[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn test_emd_zero_max_imfs_error() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(emd(&data, 0).is_err());
    }

    // ── SSA tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ssa_output_shapes() {
        let n = 40_usize;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let result = ssa(&data, 10, 3).expect("ssa failed");
        assert_eq!(result.components.len(), 3);
        for comp in &result.components {
            assert_eq!(comp.len(), n);
        }
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.w_correlations.len(), 3);
        assert_eq!(result.w_correlations[0].len(), 3);
    }

    #[test]
    fn test_ssa_reconstruction_accuracy() {
        // A pure sinusoid: SSA should capture it in the first two components
        let n = 60_usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 5.0)
            .collect();
        let result = ssa(&data, 12, 4).expect("ssa failed");

        // Sum of first 2 components should approximate original well
        let reconstruction: Vec<f64> = (0..n)
            .map(|t| result.components.iter().take(2).map(|c| c[t]).sum::<f64>())
            .collect();
        let mse: f64 = reconstruction
            .iter()
            .zip(data.iter())
            .map(|(r, d)| (r - d).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 5.0, "SSA reconstruction MSE too high: {}", mse);
    }

    #[test]
    fn test_ssa_eigenvalues_nonnegative() {
        let data: Vec<f64> = (0..30).map(|i| i as f64 + (i as f64 * 0.5).sin()).collect();
        let result = ssa(&data, 5, 3).expect("ssa failed");
        for &ev in &result.eigenvalues {
            assert!(ev >= 0.0, "eigenvalues must be non-negative");
        }
    }

    #[test]
    fn test_ssa_invalid_window_error() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(ssa(&data, 0, 2).is_err());
        assert!(ssa(&data, 20, 2).is_err()); // window >= n
    }

    #[test]
    fn test_ssa_w_correlations_symmetric() {
        let data: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.4).sin() + i as f64 * 0.05)
            .collect();
        let result = ssa(&data, 8, 4).expect("ssa failed");
        let wc = &result.w_correlations;
        let nc = wc.len();
        for i in 0..nc {
            for j in 0..nc {
                assert!(
                    (wc[i][j] - wc[j][i]).abs() < 1e-10,
                    "w-correlation matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_ssa_diagonal_w_correlations_near_one() {
        let data: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.3).sin() * 5.0 + i as f64 * 0.1)
            .collect();
        let result = ssa(&data, 10, 3).expect("ssa failed");
        for i in 0..result.w_correlations.len() {
            // Diagonal elements should be 1.0 (self-correlation)
            assert!(
                (result.w_correlations[i][i] - 1.0).abs() < 1e-6,
                "diagonal w-correlation at [{0},{0}] should be 1.0, got {}",
                result.w_correlations[i][i]
            );
        }
    }

    // ── Helper tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_cubic_spline_linear() {
        // Spline through linear data should reproduce the line exactly
        let xs = vec![0.0, 5.0, 10.0];
        let ys = vec![0.0, 5.0, 10.0];
        let result = natural_cubic_spline_eval(&xs, &ys, 11);
        for t in 0..11 {
            assert!(
                (result[t] - t as f64).abs() < 1e-6,
                "linear spline mismatch at t={}: got {}",
                t,
                result[t]
            );
        }
    }

    #[test]
    fn test_loess_1d_smooths_noise() {
        let n = 20_usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let smoothed = loess_1d(&data, 5);
        assert_eq!(smoothed.len(), n);
        for &v in &smoothed {
            assert!(v.is_finite());
        }
    }
}
