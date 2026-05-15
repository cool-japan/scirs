//! Nonlinear and complexity features for time series analysis.
//!
//! This module provides a focused, high-level interface for nonlinear
//! dynamical system characterization, including:
//! - Sample entropy and permutation entropy
//! - Hurst exponent via R/S analysis
//! - Higuchi fractal dimension
//! - Largest Lyapunov exponent estimation
//! - Recurrence plot statistics (RR, DET, LAM)
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray::Array1;
//! use scirs2_series::features::nonlinear::*;
//!
//! let ts = Array1::from_vec(vec![1.0f64, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 2.0, 1.0, 3.0, 4.5]);
//! let se = sample_entropy(&ts, 2, 0.2).expect("sample entropy");
//! let pe = permutation_entropy(&ts, 3, 1).expect("permutation entropy");
//! let h  = hurst_exponent(&ts).expect("hurst exponent");
//! ```

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[inline(always)]
fn f<F: Float + FromPrimitive>(v: f64) -> F {
    F::from(v).expect("float conversion")
}

// ---------------------------------------------------------------------------
// Sample Entropy
// ---------------------------------------------------------------------------

/// Compute Sample Entropy (SampEn) of a time series.
///
/// SampEn measures the probability that patterns of length `m` that are
/// similar remain similar at length `m+1`, discounting self-matches.
/// Lower values indicate more regularity.
///
/// # Arguments
/// * `ts`  - Input time series (length ≥ `m + 2`)
/// * `m`   - Template length (embedding dimension, typically 2)
/// * `r`   - Tolerance in *absolute* units (typically `0.1–0.25 × std(ts)`)
///
/// # Returns
/// SampEn value ∈ [0, ∞). Returns an error when the series is too short or
/// when no matches are found at length `m`.
pub fn sample_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < m + 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for sample entropy (need n ≥ m+2)".to_string(),
        ));
    }

    let count_m = count_template_matches(ts, n, m, r);
    let count_m1 = count_template_matches(ts, n, m + 1, r);

    if count_m == 0 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "No template matches found at length m; cannot compute SampEn".to_string(),
        ));
    }

    let ratio =
        F::from(count_m1).expect("count_m1 to float") / F::from(count_m).expect("count_m to float");

    if ratio <= F::zero() {
        // No matches at m+1 → maximum entropy; return a large but finite value
        Ok(f(10.0))
    } else {
        Ok(-ratio.ln())
    }
}

/// Count the number of template matches of length `m` (excluding self-matches).
fn count_template_matches<F>(ts: &Array1<F>, n: usize, m: usize, r: F) -> u64
where
    F: Float + FromPrimitive + Debug,
{
    if n < m {
        return 0;
    }
    let mut count: u64 = 0;
    for i in 0..(n - m) {
        for j in (i + 1)..(n - m + 1) {
            let mut match_found = true;
            for k in 0..m {
                let xi = ts[i + k];
                let xj = ts[j + k];
                if (xi - xj).abs() > r {
                    match_found = false;
                    break;
                }
            }
            if match_found {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Permutation Entropy
// ---------------------------------------------------------------------------

/// Compute Permutation Entropy (PE) of a time series.
///
/// PE encodes each embedding window as the rank-order permutation of its
/// elements and computes the Shannon entropy over the distribution of
/// observed permutations. It is invariant to monotone transformations.
///
/// # Arguments
/// * `ts`    - Input time series (length ≥ `order`)
/// * `order` - Embedding dimension `m` (permutation length, typically 3–7)
/// * `delay` - Time delay τ between elements (typically 1)
///
/// # Returns
/// Normalised PE ∈ [0, 1], where 1 means maximum disorder.
pub fn permutation_entropy<F>(ts: &Array1<F>, order: usize, delay: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let delay = delay.max(1);
    let window_size = (order - 1) * delay + 1;

    if n < window_size {
        return Err(TimeSeriesError::FeatureExtractionError(format!(
            "Time series too short for permutation entropy (need n ≥ {})",
            window_size
        )));
    }
    if order < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Permutation entropy requires order ≥ 2".to_string(),
        ));
    }

    let num_windows = n - window_size + 1;
    let mut pattern_counts: HashMap<Vec<usize>, u64> = HashMap::new();

    for i in 0..num_windows {
        // Extract the embedded window with delay
        let window: Vec<F> = (0..order).map(|k| ts[i + k * delay]).collect();

        // Compute the rank-order permutation
        let mut ranks: Vec<usize> = (0..order).collect();
        ranks.sort_by(|&a, &b| {
            window[a]
                .partial_cmp(&window[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Encode ranks as a permutation index vector
        let mut perm = vec![0usize; order];
        for (rank, &original_idx) in ranks.iter().enumerate() {
            perm[original_idx] = rank;
        }

        *pattern_counts.entry(perm).or_insert(0) += 1;
    }

    let total = F::from(num_windows).expect("num_windows to float");
    let mut entropy = F::zero();

    for &count in pattern_counts.values() {
        let p = F::from(count).expect("count to float") / total;
        entropy = entropy - p * p.ln();
    }

    // Normalise by log(order!)
    let log_n_factorial: F = (2..=order)
        .map(|k| F::from(k).expect("k to float").ln())
        .fold(F::zero(), |acc, v| acc + v);

    if log_n_factorial <= F::zero() {
        Ok(F::zero())
    } else {
        Ok(entropy / log_n_factorial)
    }
}

// ---------------------------------------------------------------------------
// Hurst Exponent (R/S Analysis)
// ---------------------------------------------------------------------------

/// Estimate the Hurst exponent via rescaled-range (R/S) analysis.
///
/// H ≈ 0.5 indicates a random walk; H > 0.5 indicates long-range persistence
/// (trending); H < 0.5 indicates mean-reversion (anti-persistence).
///
/// # Returns
/// Hurst exponent H ∈ (0, 1).
pub fn hurst_exponent<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 20 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Hurst exponent requires at least 20 observations".to_string(),
        ));
    }

    // Generate a set of sub-series lengths (at least 10 values, at most n/2)
    let max_len = n / 2;
    let min_len = 10usize;
    if min_len >= max_len {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Series too short for multi-scale R/S analysis".to_string(),
        ));
    }

    // Logarithmically spaced lags
    let num_lags = 12usize.min(max_len - min_len + 1);
    let log_min = (min_len as f64).ln();
    let log_max = (max_len as f64).ln();

    let mut log_lens: Vec<F> = Vec::with_capacity(num_lags);
    let mut log_rs: Vec<F> = Vec::with_capacity(num_lags);

    for i in 0..num_lags {
        let t = i as f64 / (num_lags - 1).max(1) as f64;
        let sub_len = (log_min + t * (log_max - log_min)).exp().round() as usize;
        let sub_len = sub_len.max(min_len).min(max_len);

        let rs = compute_rs(ts, sub_len)?;
        if rs > F::zero() {
            log_lens.push(F::from(sub_len).expect("sub_len to float").ln());
            log_rs.push(rs.ln());
        }
    }

    if log_lens.len() < 4 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Insufficient valid R/S values for regression".to_string(),
        ));
    }

    // OLS slope = H
    Ok(ols_slope(&log_lens, &log_rs))
}

/// Compute the average R/S statistic for windows of length `sub_len`.
fn compute_rs<F>(ts: &Array1<F>, sub_len: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let num_windows = n / sub_len;
    if num_windows == 0 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Window length exceeds series length".to_string(),
        ));
    }

    let mut rs_sum = F::zero();
    let mut valid = 0usize;

    for w in 0..num_windows {
        let start = w * sub_len;
        let end = start + sub_len;
        let segment: Vec<F> = (start..end).map(|i| ts[i]).collect();

        let mean = segment.iter().copied().fold(F::zero(), |s, x| s + x)
            / F::from(sub_len).expect("sub_len float");

        // Cumulative deviation from mean
        let mut cumdev = Vec::with_capacity(sub_len);
        let mut running = F::zero();
        for &x in &segment {
            running = running + (x - mean);
            cumdev.push(running);
        }

        let range = {
            let max = cumdev
                .iter()
                .copied()
                .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
            let min = cumdev
                .iter()
                .copied()
                .fold(F::infinity(), |a, b| if b < a { b } else { a });
            max - min
        };

        // Standard deviation
        let variance = segment
            .iter()
            .copied()
            .map(|x| (x - mean) * (x - mean))
            .fold(F::zero(), |s, v| s + v)
            / F::from(sub_len).expect("sub_len float");
        let std_dev = variance.sqrt();

        if std_dev > F::zero() && range > F::zero() {
            rs_sum = rs_sum + range / std_dev;
            valid += 1;
        }
    }

    if valid == 0 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "All segments have zero standard deviation".to_string(),
        ));
    }

    Ok(rs_sum / F::from(valid).expect("valid float"))
}

// ---------------------------------------------------------------------------
// Higuchi Fractal Dimension
// ---------------------------------------------------------------------------

/// Compute the Higuchi fractal dimension of a time series.
///
/// Uses the Higuchi algorithm: for each integer interval `k`, the average
/// length of the series is computed; the slope of log(length) vs log(k) is
/// the negated fractal dimension.
///
/// # Arguments
/// * `ts`   - Input time series
/// * `kmax` - Maximum interval (typically 5–10)
///
/// # Returns
/// Fractal dimension D ∈ [1, 2] for typical signals.
pub fn fractal_dimension<F>(ts: &Array1<F>, kmax: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 2 * kmax + 1 {
        return Err(TimeSeriesError::FeatureExtractionError(format!(
            "Series too short for Higuchi FD with kmax={} (need n ≥ {})",
            kmax,
            2 * kmax + 1
        )));
    }
    if kmax < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "kmax must be ≥ 2 for Higuchi fractal dimension".to_string(),
        ));
    }

    let n_f = F::from(n).expect("n float");
    let mut log_k: Vec<F> = Vec::with_capacity(kmax - 1);
    let mut log_lk: Vec<F> = Vec::with_capacity(kmax - 1);

    for k in 1..=kmax {
        let k_f = F::from(k).expect("k float");
        let mut lk = F::zero();

        for m in 1..=k {
            // Number of intervals for starting point m and interval k
            let num_intervals = (n - m) / k;
            if num_intervals < 1 {
                continue;
            }

            let mut length = F::zero();
            for i in 1..=num_intervals {
                let idx1 = m - 1 + i * k;
                let idx0 = m - 1 + (i - 1) * k;
                if idx1 < n {
                    length = length + (ts[idx1] - ts[idx0]).abs();
                }
            }

            let normalization = n_f / (F::from(num_intervals).expect("num float") * k_f);
            lk = lk + length * normalization / k_f;
        }

        let lk = lk / F::from(k).expect("k float");
        if lk > F::zero() {
            log_k.push(k_f.ln());
            log_lk.push(lk.ln());
        }
    }

    if log_k.len() < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Insufficient valid points for Higuchi regression".to_string(),
        ));
    }

    // The fractal dimension is the magnitude of the slope
    Ok(-ols_slope(&log_k, &log_lk))
}

// ---------------------------------------------------------------------------
// Largest Lyapunov Exponent
// ---------------------------------------------------------------------------

/// Estimate the largest Lyapunov exponent (LLE) of a time series.
///
/// Uses the Rosenstein algorithm: embed the series in `m`-dimensional space
/// with delay `tau`, find nearest neighbours, and average the divergence
/// of nearby trajectories over `max_iter` steps.
///
/// # Arguments
/// * `ts`       - Input time series (length ≥ ~50 for reliable estimate)
/// * `m`        - Embedding dimension (typically 3–5)
/// * `tau`      - Time delay (typically estimated from first ACF zero)
/// * `max_iter` - Number of steps to track divergence (typically 10–20)
///
/// # Returns
/// LLE estimate. Positive values indicate chaos; near-zero indicates
/// periodic/quasi-periodic behaviour; negative indicates stable fixed point.
pub fn lyapunov_exponent<F>(ts: &Array1<F>, m: usize, tau: usize, max_iter: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let tau = tau.max(1);
    let min_n = (m - 1) * tau + max_iter + 2;

    if n < min_n {
        return Err(TimeSeriesError::FeatureExtractionError(format!(
            "Series too short for LLE (need n ≥ {})",
            min_n
        )));
    }

    // Embed the series
    let num_vectors = n - (m - 1) * tau;
    let embedded: Vec<Vec<F>> = (0..num_vectors)
        .map(|i| (0..m).map(|k| ts[i + k * tau]).collect())
        .collect();

    let nv = embedded.len();

    // For each vector, find nearest neighbour with index gap ≥ mean period
    // (using a simple minimum gap of `m` here as Theiler window)
    let theiler = m;

    // Average log divergence
    let mut divergence_sum: Vec<F> = vec![F::zero(); max_iter];
    let mut divergence_count: Vec<usize> = vec![0; max_iter];

    for i in 0..nv {
        // Find nearest neighbour (brute-force, O(N²))
        let mut min_dist = F::infinity();
        let mut nn_idx = None;

        for j in 0..nv {
            let gap = if i > j { i - j } else { j - i };
            if gap <= theiler {
                continue;
            }
            let dist = euclidean_distance_embedded(&embedded[i], &embedded[j]);
            if dist < min_dist {
                min_dist = dist;
                nn_idx = Some(j);
            }
        }

        let j = match nn_idx {
            Some(idx) => idx,
            None => continue,
        };

        if min_dist <= F::zero() {
            continue;
        }

        // Track divergence for max_iter steps
        for step in 0..max_iter {
            let ni = i + step;
            let nj = j + step;
            if ni >= nv || nj >= nv {
                break;
            }
            let d = euclidean_distance_embedded(&embedded[ni], &embedded[nj]);
            if d > F::zero() {
                divergence_sum[step] = divergence_sum[step] + (d / min_dist).ln();
                divergence_count[step] += 1;
            }
        }
    }

    // Compute average divergence per step
    let avg_div: Vec<F> = divergence_sum
        .iter()
        .zip(divergence_count.iter())
        .filter_map(|(&s, &c)| {
            if c > 0 {
                Some(s / F::from(c).expect("c float"))
            } else {
                None
            }
        })
        .collect();

    if avg_div.len() < 4 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Insufficient divergence data for LLE estimate".to_string(),
        ));
    }

    // LLE = slope of average log-divergence vs step index
    let x: Vec<F> = (0..avg_div.len())
        .map(|i| F::from(i).expect("i float"))
        .collect();
    Ok(ols_slope(&x, &avg_div))
}

/// Euclidean distance between two embedded vectors.
fn euclidean_distance_embedded<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .fold(F::zero(), |s, v| s + v)
        .sqrt()
}

// ---------------------------------------------------------------------------
// Recurrence Plot Statistics
// ---------------------------------------------------------------------------

/// Recurrence plot statistics: Recurrence Rate, Determinism, Laminarity.
///
/// Constructs a binary recurrence matrix R\[i,j\] = 1 if ‖x_i − x_j‖ ≤ ε and
/// computes summary statistics on diagonal and vertical lines.
///
/// # Arguments
/// * `ts`    - Input time series (embedded as scalar; for multi-dim extend via `m`)
/// * `eps`   - Recurrence threshold (in absolute units)
/// * `m`     - Embedding dimension (≥ 1)
/// * `tau`   - Embedding delay (≥ 1)
/// * `l_min` - Minimum diagonal / vertical line length for DET/LAM computation
///
/// # Returns
/// `RecurrenceStats` struct with RR, DET, LAM, and associated line lengths.
pub fn recurrence_rate<F>(
    ts: &Array1<F>,
    eps: F,
    m: usize,
    tau: usize,
    l_min: usize,
) -> Result<RecurrenceStats<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let tau = tau.max(1);
    let m = m.max(1);
    let l_min = l_min.max(2);
    let min_n = (m - 1) * tau + 1;

    if n < min_n + 2 {
        return Err(TimeSeriesError::FeatureExtractionError(format!(
            "Series too short for recurrence analysis (need n ≥ {})",
            min_n + 2
        )));
    }

    // Embed
    let nv = n - (m - 1) * tau;
    let embedded: Vec<Vec<F>> = (0..nv)
        .map(|i| (0..m).map(|k| ts[i + k * tau]).collect())
        .collect();

    // Build binary recurrence matrix (stored as flat boolean vector)
    let mut recurrence = vec![false; nv * nv];
    let mut rec_count: u64 = 0;

    for i in 0..nv {
        for j in 0..nv {
            let dist = euclidean_distance_embedded(&embedded[i], &embedded[j]);
            if dist <= eps {
                recurrence[i * nv + j] = true;
                rec_count += 1;
            }
        }
    }

    // Recurrence Rate (excluding main diagonal)
    let total_off_diag = (nv * nv - nv) as u64;
    let rr = if total_off_diag > 0 {
        // subtract diagonal
        let diag_count = nv as u64;
        let off_count = rec_count.saturating_sub(diag_count);
        F::from(off_count).expect("off_count float") / F::from(total_off_diag).expect("total float")
    } else {
        F::zero()
    };

    // Determinism: ratio of recurrence points in diagonal lines ≥ l_min
    let (det, avg_diag_len) = compute_diagonal_lines(&recurrence, nv, l_min);

    // Laminarity: ratio of recurrence points in vertical lines ≥ l_min
    let (lam, avg_vert_len) = compute_vertical_lines(&recurrence, nv, l_min);

    Ok(RecurrenceStats {
        recurrence_rate: rr,
        determinism: det,
        laminarity: lam,
        avg_diagonal_line_length: avg_diag_len,
        avg_vertical_line_length: avg_vert_len,
        num_vectors: nv,
    })
}

/// Compute DET and average diagonal line length from recurrence matrix.
fn compute_diagonal_lines<F: Float + FromPrimitive>(
    rec: &[bool],
    n: usize,
    l_min: usize,
) -> (F, F) {
    // Walk upper diagonals (offset > 0)
    let mut points_in_long_lines: u64 = 0;
    let mut total_rec_points: u64 = 0;
    let mut total_line_length: u64 = 0;
    let mut num_long_lines: u64 = 0;

    for start in 0..(n - 1) {
        // Diagonal starting at (0, start) and (start, 0)
        for &s in &[start, 0usize] {
            let row_start = if s == start { 0 } else { start };
            let col_start = if s == start { start } else { 0 };
            let diag_len = n - start;

            let mut run = 0usize;
            for d in 0..diag_len {
                let r = row_start + d;
                let c = col_start + d;
                if r < n && c < n && r != c && rec[r * n + c] {
                    run += 1;
                    total_rec_points += 1;
                } else {
                    if run >= l_min {
                        points_in_long_lines += run as u64;
                        total_line_length += run as u64;
                        num_long_lines += 1;
                    }
                    run = 0;
                }
            }
            if run >= l_min {
                points_in_long_lines += run as u64;
                total_line_length += run as u64;
                num_long_lines += 1;
            }
        }
    }

    let det = if total_rec_points > 0 {
        F::from(points_in_long_lines).expect("pts float")
            / F::from(total_rec_points).expect("total float")
    } else {
        F::zero()
    };

    let avg_len = if num_long_lines > 0 {
        F::from(total_line_length).expect("tll float") / F::from(num_long_lines).expect("nll float")
    } else {
        F::zero()
    };

    (det, avg_len)
}

/// Compute LAM and average vertical line length from recurrence matrix.
fn compute_vertical_lines<F: Float + FromPrimitive>(
    rec: &[bool],
    n: usize,
    l_min: usize,
) -> (F, F) {
    let mut points_in_long_lines: u64 = 0;
    let mut total_rec_points: u64 = 0;
    let mut total_line_length: u64 = 0;
    let mut num_long_lines: u64 = 0;

    for col in 0..n {
        let mut run = 0usize;
        for row in 0..n {
            if row != col && rec[row * n + col] {
                run += 1;
                total_rec_points += 1;
            } else {
                if run >= l_min {
                    points_in_long_lines += run as u64;
                    total_line_length += run as u64;
                    num_long_lines += 1;
                }
                run = 0;
            }
        }
        if run >= l_min {
            points_in_long_lines += run as u64;
            total_line_length += run as u64;
            num_long_lines += 1;
        }
    }

    let lam = if total_rec_points > 0 {
        F::from(points_in_long_lines).expect("pts float")
            / F::from(total_rec_points).expect("total float")
    } else {
        F::zero()
    };

    let avg_len = if num_long_lines > 0 {
        F::from(total_line_length).expect("tll float") / F::from(num_long_lines).expect("nll float")
    } else {
        F::zero()
    };

    (lam, avg_len)
}

/// Summary statistics from a recurrence plot analysis.
#[derive(Debug, Clone)]
pub struct RecurrenceStats<F> {
    /// Recurrence Rate (RR): fraction of off-diagonal recurrence points
    pub recurrence_rate: F,
    /// Determinism (DET): fraction of recurrence points in diagonal lines ≥ l_min
    pub determinism: F,
    /// Laminarity (LAM): fraction of recurrence points in vertical lines ≥ l_min
    pub laminarity: F,
    /// Average diagonal line length (for lines ≥ l_min)
    pub avg_diagonal_line_length: F,
    /// Average vertical line length (for lines ≥ l_min)
    pub avg_vertical_line_length: F,
    /// Number of embedded vectors
    pub num_vectors: usize,
}

// ---------------------------------------------------------------------------
// OLS slope helper
// ---------------------------------------------------------------------------

/// Simple OLS slope estimator: slope of y on x.
fn ols_slope<F: Float + FromPrimitive>(x: &[F], y: &[F]) -> F {
    let n = x.len().min(y.len());
    if n < 2 {
        return F::zero();
    }
    let nf = F::from(n).expect("n float");
    let mean_x = x[..n].iter().copied().fold(F::zero(), |s, v| s + v) / nf;
    let mean_y = y[..n].iter().copied().fold(F::zero(), |s, v| s + v) / nf;

    let (mut cov, mut var_x) = (F::zero(), F::zero());
    for i in 0..n {
        let dx = x[i] - mean_x;
        cov = cov + dx * (y[i] - mean_y);
        var_x = var_x + dx * dx;
    }

    if var_x == F::zero() {
        F::zero()
    } else {
        cov / var_x
    }
}

// ---------------------------------------------------------------------------
// Convenience bundle
// ---------------------------------------------------------------------------

/// Comprehensive nonlinear feature bundle for a time series.
#[derive(Debug, Clone)]
pub struct NonlinearFeatures<F> {
    /// Sample entropy (m=2, r=0.2*std)
    pub sample_entropy: F,
    /// Permutation entropy (order=3, delay=1, normalised)
    pub permutation_entropy: F,
    /// Hurst exponent via R/S analysis
    pub hurst_exponent: F,
    /// Higuchi fractal dimension (kmax=5)
    pub fractal_dimension: F,
    /// Largest Lyapunov exponent estimate (m=3, tau=1, max_iter=10)
    pub lyapunov_exponent: F,
    /// Recurrence statistics
    pub recurrence: RecurrenceStats<F>,
}

/// Compute all nonlinear features for a time series.
///
/// # Arguments
/// * `ts` - Input time series (recommend ≥ 100 points for reliable estimates)
///
/// # Returns
/// `NonlinearFeatures<F>` containing all computed measures.
pub fn nonlinear_features<F>(ts: &Array1<F>) -> Result<NonlinearFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Compute standard deviation for SampEn tolerance
    let n = ts.len();
    if n < 20 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Need at least 20 observations for nonlinear features".to_string(),
        ));
    }

    let mean = ts.iter().copied().fold(F::zero(), |s, x| s + x) / F::from(n).expect("n float");
    let variance = ts
        .iter()
        .copied()
        .map(|x| (x - mean) * (x - mean))
        .fold(F::zero(), |s, v| s + v)
        / F::from(n).expect("n float");
    let std_dev = variance.sqrt();
    let r = f::<F>(0.2) * std_dev;
    let r_safe = if r <= F::zero() { f(0.01) } else { r };

    let se = sample_entropy(ts, 2, r_safe).unwrap_or(F::zero());
    let pe = permutation_entropy(ts, 3, 1).unwrap_or(F::zero());
    let hurst = hurst_exponent(ts).unwrap_or(f(0.5));
    let fd = fractal_dimension(ts, 5).unwrap_or(F::zero());
    let lle = lyapunov_exponent(ts, 3, 1, 10).unwrap_or(F::zero());
    let rec = recurrence_rate(ts, r_safe, 2, 1, 2).unwrap_or(RecurrenceStats {
        recurrence_rate: F::zero(),
        determinism: F::zero(),
        laminarity: F::zero(),
        avg_diagonal_line_length: F::zero(),
        avg_vertical_line_length: F::zero(),
        num_vectors: 0,
    });

    Ok(NonlinearFeatures {
        sample_entropy: se,
        permutation_entropy: pe,
        hurst_exponent: hurst,
        fractal_dimension: fd,
        lyapunov_exponent: lle,
        recurrence: rec,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_ts(v: Vec<f64>) -> Array1<f64> {
        Array1::from_vec(v)
    }

    fn sine_wave(n: usize, freq: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin()),
        )
    }

    fn ramp(n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| i as f64 / n as f64))
    }

    #[test]
    fn test_sample_entropy_basic() {
        let ts = make_ts(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        let result = sample_entropy(&ts, 2, 0.2);
        assert!(result.is_ok(), "sample_entropy failed: {:?}", result);
        assert!(result.expect("SampEn") >= 0.0);
    }

    #[test]
    fn test_sample_entropy_too_short() {
        let ts = make_ts(vec![1.0, 2.0]);
        assert!(sample_entropy(&ts, 2, 0.2).is_err());
    }

    #[test]
    fn test_permutation_entropy_regular() {
        // Perfect alternating pattern → low entropy
        let ts = make_ts(vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]);
        let pe = permutation_entropy(&ts, 3, 1).expect("PE");
        assert!(pe >= 0.0);
        assert!(pe <= 1.0);
    }

    #[test]
    fn test_permutation_entropy_range() {
        let ts = sine_wave(64, 4.0);
        let pe = permutation_entropy(&ts, 4, 1).expect("PE sine");
        assert!((0.0..=1.0).contains(&pe), "PE out of range: {}", pe);
    }

    #[test]
    fn test_hurst_exponent_ramp() {
        // A ramp (monotone increasing) should have H close to 1
        let ts = ramp(100);
        let h = hurst_exponent(&ts).expect("Hurst ramp");
        assert!(h >= 0.0, "Hurst negative: {}", h);
        assert!(h <= 2.0, "Hurst too large: {}", h);
    }

    #[test]
    fn test_hurst_exponent_too_short() {
        let ts = make_ts(vec![1.0, 2.0, 3.0]);
        assert!(hurst_exponent(&ts).is_err());
    }

    #[test]
    fn test_fractal_dimension_basic() {
        let ts = sine_wave(64, 3.0);
        let fd = fractal_dimension(&ts, 5).expect("FD");
        // Should be between 1 and 2 for a smooth signal
        assert!(fd >= 0.0, "FD negative: {}", fd);
        assert!(fd <= 3.0, "FD too large: {}", fd);
    }

    #[test]
    fn test_lyapunov_exponent_basic() {
        let ts = ramp(80);
        let lle = lyapunov_exponent(&ts, 3, 1, 8).expect("LLE");
        // Ramp should have very small (≤ 0) or small positive LLE
        assert!(lle.is_finite(), "LLE is NaN/inf");
    }

    #[test]
    fn test_recurrence_rate_basic() {
        let ts = sine_wave(40, 3.0);
        let stats = recurrence_rate(&ts, 0.3, 2, 1, 2).expect("RR");
        assert!((0.0..=1.0).contains(&stats.recurrence_rate));
        assert!((0.0..=1.0).contains(&stats.determinism));
        assert!((0.0..=1.0).contains(&stats.laminarity));
    }

    #[test]
    fn test_nonlinear_features_bundle() {
        let ts = sine_wave(128, 5.0);
        let feats = nonlinear_features(&ts).expect("NonlinearFeatures");
        assert!(feats.sample_entropy >= 0.0);
        assert!((0.0..=1.0).contains(&feats.permutation_entropy));
        assert!(feats.hurst_exponent >= 0.0);
    }
}
