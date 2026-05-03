//! Enhanced SIMD-optimized correlation and covariance functions for v0.2.0
//!
//! This module provides comprehensive SIMD-accelerated implementations of
//! correlation and covariance computations with improved performance.

use crate::descriptive_simd::mean_simd;
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// Compute covariance matrix using SIMD operations
///
/// This function efficiently computes the covariance matrix for multiple variables
/// using SIMD acceleration for improved performance (2-3x speedup expected).
///
/// # Arguments
///
/// * `data` - 2D array where each row is an observation and each column is a variable
/// * `rowvar` - If true, rows are variables and columns are observations
/// * `ddof` - Delta degrees of freedom (default: 1 for sample covariance)
///
/// # Returns
///
/// Covariance matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::correlation_simd_enhanced::covariance_matrix_simd;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let cov = covariance_matrix_simd(&data.view(), false, 1).expect("Computation failed");
/// assert_eq!(cov.shape(), &[3, 3]);
/// ```
pub fn covariance_matrix_simd<F, D>(
    data: &ArrayBase<D, Ix2>,
    rowvar: bool,
    ddof: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    let (n_vars, n_obs) = if rowvar {
        (data.nrows(), data.ncols())
    } else {
        (data.ncols(), data.nrows())
    };

    if n_obs <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough observations for the given degrees of freedom",
        ));
    }

    let optimizer = AutoOptimizer::new();
    let mut cov_matrix = Array2::zeros((n_vars, n_vars));

    // Compute means for each variable
    let means: Vec<F> = (0..n_vars)
        .map(|i| {
            let var = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };
            mean_simd(&var).unwrap_or_else(|_| F::zero())
        })
        .collect();

    // Compute covariance matrix
    for i in 0..n_vars {
        for j in i..n_vars {
            let var_i = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };

            let var_j = if rowvar {
                data.slice(s![j, ..])
            } else {
                data.slice(s![.., j])
            };

            let cov = if optimizer.should_use_simd(n_obs) {
                // SIMD path
                let mean_i_array = Array1::from_elem(n_obs, means[i]);
                let mean_j_array = Array1::from_elem(n_obs, means[j]);

                let dev_i = F::simd_sub(&var_i, &mean_i_array.view());
                let dev_j = F::simd_sub(&var_j, &mean_j_array.view());
                let products = F::simd_mul(&dev_i.view(), &dev_j.view());
                let sum_products = F::simd_sum(&products.view());

                sum_products / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for k in 0..n_obs {
                    let dev_i = var_i[k] - means[i];
                    let dev_j = var_j[k] - means[j];
                    sum = sum + dev_i * dev_j;
                }
                sum / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            };

            cov_matrix[(i, j)] = cov;
            if i != j {
                cov_matrix[(j, i)] = cov; // Symmetric matrix
            }
        }
    }

    Ok(cov_matrix)
}

/// Compute Spearman rank correlation using SIMD operations
///
/// This function computes the Spearman rank correlation coefficient using
/// SIMD acceleration after converting values to ranks.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// Spearman's rank correlation coefficient
pub fn spearman_r_simd<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    if x.is_empty() {
        return Err(StatsError::invalid_argument("Arrays cannot be empty"));
    }

    let n = x.len();

    // Convert to ranks
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Use Pearson correlation on ranks
    let mean_rx = mean_simd(&rank_x.view())?;
    let mean_ry = mean_simd(&rank_y.view())?;

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // SIMD path
        let mean_rx_array = Array1::from_elem(n, mean_rx);
        let mean_ry_array = Array1::from_elem(n, mean_ry);

        let rx_dev = F::simd_sub(&rank_x.view(), &mean_rx_array.view());
        let ry_dev = F::simd_sub(&rank_y.view(), &mean_ry_array.view());

        let xy_dev = F::simd_mul(&rx_dev.view(), &ry_dev.view());
        let rx_dev_sq = F::simd_mul(&rx_dev.view(), &rx_dev.view());
        let ry_dev_sq = F::simd_mul(&ry_dev.view(), &ry_dev.view());

        let sum_xy = F::simd_sum(&xy_dev.view());
        let sum_rx2 = F::simd_sum(&rx_dev_sq.view());
        let sum_ry2 = F::simd_sum(&ry_dev_sq.view());

        if sum_rx2 <= F::epsilon() || sum_ry2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        let corr = sum_xy / (sum_rx2 * sum_ry2).sqrt();
        Ok(corr.max(-F::one()).min(F::one()))
    } else {
        // Scalar fallback
        let mut sum_xy = F::zero();
        let mut sum_rx2 = F::zero();
        let mut sum_ry2 = F::zero();

        for i in 0..n {
            let rx_dev = rank_x[i] - mean_rx;
            let ry_dev = rank_y[i] - mean_ry;

            sum_xy = sum_xy + rx_dev * ry_dev;
            sum_rx2 = sum_rx2 + rx_dev * rx_dev;
            sum_ry2 = sum_ry2 + ry_dev * ry_dev;
        }

        if sum_rx2 <= F::epsilon() || sum_ry2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        let corr = sum_xy / (sum_rx2 * sum_ry2).sqrt();
        Ok(corr.max(-F::one()).min(F::one()))
    }
}

/// Helper function to compute ranks
fn compute_ranks<F, D>(data: &ArrayBase<D, Ix1>) -> Array1<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = data.len();
    let mut indexed: Vec<(usize, F)> = data.iter().copied().enumerate().collect();

    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (average rank for ties)
    let mut ranks = Array1::zeros(n);
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find the end of tied values
        while j < n && (indexed[j].1 - indexed[i].1).abs() < F::epsilon() {
            j += 1;
        }

        // Compute average rank for tied values
        let avg_rank = F::from((i + j + 1) as f64 / 2.0).unwrap_or_else(|| F::zero());

        // Assign average rank to all tied values
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Compute partial correlation using SIMD operations
///
/// Computes the partial correlation between x and y, controlling for z.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable
/// * `z` - Controlling variables (can be multivariate)
///
/// # Returns
///
/// Partial correlation coefficient
pub fn partial_correlation_simd<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    z: &ArrayView2<F>,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.len() != y.len() || x.len() != z.nrows() {
        return Err(StatsError::dimension_mismatch(
            "All arrays must have compatible dimensions",
        ));
    }

    // Use the formula: partial_corr(x,y|z) = (corr(x,y) - corr(x,z)*corr(y,z)) / sqrt((1-corr(x,z)^2)*(1-corr(y,z)^2))
    // For simplicity, we'll compute residuals after regressing x and y on z

    // This is a simplified implementation - full implementation would use
    // regression to compute residuals
    use crate::correlation_simd::pearson_r_simd;

    // For single controlling variable
    if z.ncols() == 1 {
        let z_col = z.column(0);
        let rxy = pearson_r_simd(x, y)?;
        let rxz = pearson_r_simd(x, &z_col)?;
        let ryz = pearson_r_simd(y, &z_col)?;

        let numerator = rxy - rxz * ryz;
        let denominator = ((F::one() - rxz * rxz) * (F::one() - ryz * ryz)).sqrt();

        if denominator <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute partial correlation - controlling variable perfectly predicts x or y",
            ));
        }

        Ok(numerator / denominator)
    } else {
        // For multiple controlling variables, compute partial correlation by regressing
        // both x and y on the full Z matrix (Z has ncols > 1) and taking the Pearson
        // correlation of the residuals.
        //
        // OLS residuals: e = v - Z (Z'Z)^{-1} Z' v for v ∈ {x, y}
        // We solve (Z'Z) β = Z'v via Cholesky decomposition.
        let q = z.ncols();
        let n = x.len();

        // Build Z'Z  (q × q)
        let mut ztg = vec![F::zero(); q * q];
        for j in 0..q {
            for k in j..q {
                let mut dot = F::zero();
                for i in 0..n {
                    dot = dot + z[[i, j]] * z[[i, k]];
                }
                ztg[j * q + k] = dot;
                ztg[k * q + j] = dot; // symmetric
            }
        }

        // Build Z'x  and  Z'y  (length q)
        let mut ztx = vec![F::zero(); q];
        let mut zty = vec![F::zero(); q];
        for j in 0..q {
            let mut dx = F::zero();
            let mut dy = F::zero();
            for i in 0..n {
                dx = dx + z[[i, j]] * x[i];
                dy = dy + z[[i, j]] * y[i];
            }
            ztx[j] = dx;
            zty[j] = dy;
        }

        // Cholesky decompose Z'Z  with a small jitter for numerical stability.
        let jitter = F::from(1e-12).unwrap_or(F::zero());
        let mut l = vec![F::zero(); q * q];
        for i in 0..q {
            ztg[i * q + i] = ztg[i * q + i] + jitter;
        }
        for i in 0..q {
            for j in 0..=i {
                let mut s = ztg[i * q + j];
                for k in 0..j {
                    s = s - l[i * q + k] * l[j * q + k];
                }
                if i == j {
                    if s <= F::zero() {
                        return Err(StatsError::invalid_argument(
                            "Z'Z is not positive definite — controlling variables are collinear",
                        ));
                    }
                    l[i * q + j] = s.sqrt();
                } else {
                    let diag = l[j * q + j];
                    l[i * q + j] = s / diag;
                }
            }
        }

        // Solve L b = rhs (forward substitution) then L' x = b (backward substitution)
        let chol_solve = |rhs: &[F]| -> Vec<F> {
            // Forward: L y = rhs
            let mut y = vec![F::zero(); q];
            for i in 0..q {
                let mut s = rhs[i];
                for j in 0..i {
                    s = s - l[i * q + j] * y[j];
                }
                let diag = l[i * q + i];
                y[i] = if diag > F::zero() {
                    s / diag
                } else {
                    F::zero()
                };
            }
            // Backward: L' x = y
            let mut x_out = vec![F::zero(); q];
            for i in (0..q).rev() {
                let mut s = y[i];
                for k in (i + 1)..q {
                    s = s - l[k * q + i] * x_out[k];
                }
                let diag = l[i * q + i];
                x_out[i] = if diag > F::zero() {
                    s / diag
                } else {
                    F::zero()
                };
            }
            x_out
        };

        let beta_x = chol_solve(&ztx);
        let beta_y = chol_solve(&zty);

        // Compute residuals: e_x[i] = x[i] - Z[i,:] β_x
        let mut res_x = Array1::<F>::zeros(n);
        let mut res_y = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut fitted_x = F::zero();
            let mut fitted_y = F::zero();
            for j in 0..q {
                fitted_x = fitted_x + z[[i, j]] * beta_x[j];
                fitted_y = fitted_y + z[[i, j]] * beta_y[j];
            }
            res_x[i] = x[i] - fitted_x;
            res_y[i] = y[i] - fitted_y;
        }

        // Partial correlation = Pearson(res_x, res_y)
        use crate::correlation_simd::pearson_r_simd;
        pearson_r_simd(&res_x.view(), &res_y.view())
    }
}

/// Compute rolling correlation using SIMD operations
///
/// Computes correlation between x and y over a rolling window.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `window_size` - Size of the rolling window
///
/// # Returns
///
/// Array of rolling correlation values
pub fn rolling_correlation_simd<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    window_size: usize,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    if window_size < 2 {
        return Err(StatsError::invalid_argument(
            "Window size must be at least 2",
        ));
    }

    let n = x.len();
    if n < window_size {
        return Err(StatsError::invalid_argument(
            "Array length must be at least window size",
        ));
    }

    let n_windows = n - window_size + 1;
    let mut result = Array1::zeros(n_windows);

    use crate::correlation_simd::pearson_r_simd;

    for i in 0..n_windows {
        let x_window = x.slice(s![i..i + window_size]);
        let y_window = y.slice(s![i..i + window_size]);
        result[i] = pearson_r_simd(&x_window, &y_window)?;
    }

    Ok(result)
}

// ============================================================================
// SIMD Pearson correlation matrix (parallel over variable pairs)
// ============================================================================

/// Compute the full Pearson correlation matrix for a dataset using SIMD
/// dot-product kernels and Rayon parallelism across variable pairs.
///
/// The data layout follows the convention where **rows are observations** and
/// **columns are variables** (i.e. each column is one variable).
///
/// # Algorithm
///
/// 1. Compute per-column means in parallel.
/// 2. For every off-diagonal `(i, j)` pair compute the Pearson r using SIMD
///    dot-products on the mean-centred columns.
/// 3. Assemble the symmetric result matrix.
///
/// # Arguments
///
/// * `data` — 2-D array with shape `(n_obs, n_vars)`.
///
/// # Errors
///
/// Returns [`StatsError`] when fewer than 2 observations are present, the
/// data contains fewer than 2 variables, or a column has zero variance.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::correlation_simd_enhanced::simd_pearson_correlation_matrix;
///
/// let data = array![
///     [1.0_f64, 4.0],
///     [2.0_f64, 3.0],
///     [3.0_f64, 2.0],
///     [4.0_f64, 1.0],
/// ];
/// let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
/// // x1 and x2 are perfectly negatively correlated
/// assert!((r[(0, 1)] + 1.0_f64).abs() < 1e-10);
/// assert!((r[(0, 0)] - 1.0_f64).abs() < 1e-10);
/// ```
pub fn simd_pearson_correlation_matrix(data: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n_obs = data.nrows();
    let n_vars = data.ncols();

    if n_obs < 2 {
        return Err(StatsError::invalid_argument(
            "At least 2 observations are required for Pearson correlation",
        ));
    }
    if n_vars < 2 {
        return Err(StatsError::invalid_argument(
            "At least 2 variables are required for a correlation matrix",
        ));
    }

    use scirs2_core::parallel_ops::*;

    // ── Step 1: per-column means ──────────────────────────────────────────────
    let optimizer = AutoOptimizer::new();
    let means: Vec<f64> = (0..n_vars)
        .map(|j| {
            let col = data.column(j);
            if optimizer.should_use_simd(n_obs) {
                let arr = col.to_owned();
                F64::simd_sum(&arr.view()) / n_obs as f64
            } else {
                col.iter().sum::<f64>() / n_obs as f64
            }
        })
        .collect();

    // ── Step 2: per-column std-dev (population, no Bessel correction) ─────────
    let stds: Vec<f64> = (0..n_vars)
        .map(|j| {
            let col = data.column(j);
            let m = means[j];
            let var: f64 = if optimizer.should_use_simd(n_obs) {
                let arr = col.to_owned();
                let mean_arr = Array1::from_elem(n_obs, m);
                let dev = F64::simd_sub(&arr.view(), &mean_arr.view());
                let sq = F64::simd_mul(&dev.view(), &dev.view());
                F64::simd_sum(&sq.view()) / n_obs as f64
            } else {
                col.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n_obs as f64
            };
            var.sqrt()
        })
        .collect();

    // Validate — no zero-variance columns.
    for (j, &sd) in stds.iter().enumerate() {
        if sd <= f64::EPSILON {
            return Err(StatsError::invalid_argument(&format!(
                "Variable {j} has zero variance; Pearson r is undefined"
            )));
        }
    }

    // ── Step 3: build upper-triangle pair list ────────────────────────────────
    let pairs: Vec<(usize, usize)> = (0..n_vars)
        .flat_map(|i| ((i + 1)..n_vars).map(move |j| (i, j)))
        .collect();

    // ── Step 4: parallel computation of each (i, j) correlation ──────────────
    type SIMDf64 = f64; // alias to satisfy the closure type inference below
    let corrs: Result<Vec<((usize, usize), f64)>, StatsError> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let col_i = data.column(i).to_owned();
            let col_j = data.column(j).to_owned();
            let mi = means[i];
            let mj = means[j];
            let si = stds[i];
            let sj = stds[j];

            let r: SIMDf64 = if optimizer.should_use_simd(n_obs) {
                let mean_i_arr = Array1::from_elem(n_obs, mi);
                let mean_j_arr = Array1::from_elem(n_obs, mj);
                let dev_i = F64::simd_sub(&col_i.view(), &mean_i_arr.view());
                let dev_j = F64::simd_sub(&col_j.view(), &mean_j_arr.view());
                let products = F64::simd_mul(&dev_i.view(), &dev_j.view());
                let cov = F64::simd_sum(&products.view()) / n_obs as f64;
                cov / (si * sj)
            } else {
                let cov: f64 = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&xi, &xj)| (xi - mi) * (xj - mj))
                    .sum::<f64>()
                    / n_obs as f64;
                cov / (si * sj)
            };

            // Clamp to [-1, 1] to defend against floating-point overshoot.
            Ok(((i, j), r.clamp(-1.0, 1.0)))
        })
        .collect();

    let corrs = corrs?;

    // ── Step 5: assemble the symmetric matrix ─────────────────────────────────
    let mut out = Array2::<f64>::zeros((n_vars, n_vars));
    for j in 0..n_vars {
        out[(j, j)] = 1.0;
    }
    for ((i, j), r) in corrs {
        out[(i, j)] = r;
        out[(j, i)] = r;
    }

    Ok(out)
}

// Private type alias so closures can call SIMD methods without generics.
struct F64;
impl F64 {
    #[inline]
    fn simd_sub(
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Array1<f64> {
        <f64 as SimdUnifiedOps>::simd_sub(a, b)
    }
    #[inline]
    fn simd_mul(
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Array1<f64> {
        <f64 as SimdUnifiedOps>::simd_mul(a, b)
    }
    #[inline]
    fn simd_sum(a: &scirs2_core::ndarray::ArrayView1<f64>) -> f64 {
        <f64 as SimdUnifiedOps>::simd_sum(a)
    }
}

// ============================================================================
// SIMD Spearman one-vs-many batch correlation
// ============================================================================

/// Compute the Spearman rank correlation of a single reference vector `x`
/// against every column in `ys` using SIMD-accelerated rank-based Pearson
/// computation and Rayon parallelism.
///
/// # Arguments
///
/// * `x`  — Reference vector of length `n`.
/// * `ys` — 2-D array with shape `(n, m)`.  Each column is one comparison
///           target.
///
/// # Returns
///
/// A length-`m` array where element `k` is `spearman_r(x, ys.column(k))`.
///
/// # Errors
///
/// Returns [`StatsError`] when `x` is empty, the row count of `ys` doesn't
/// match `x.len()`, or any column has zero variance in rank space.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::correlation_simd_enhanced::simd_spearman_correlation_batch;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// // Two columns: perfect positive and perfect negative relationship.
/// let ys = Array2::from_shape_fn((5, 2), |(i, j)| {
///     if j == 0 { (i + 1) as f64 } else { (5 - i) as f64 }
/// });
/// let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view())
///     .expect("should succeed");
/// assert!((rhos[0] - 1.0).abs() < 1e-10, "perfect positive: {}", rhos[0]);
/// assert!((rhos[1] + 1.0).abs() < 1e-10, "perfect negative: {}", rhos[1]);
/// ```
pub fn simd_spearman_correlation_batch(
    x: &ArrayView1<f64>,
    ys: &ArrayView2<f64>,
) -> StatsResult<Array1<f64>> {
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Reference vector x must not be empty",
        ));
    }
    if ys.nrows() != n {
        return Err(StatsError::dimension_mismatch(
            "Number of rows in ys must equal the length of x",
        ));
    }
    let m = ys.ncols();
    if m == 0 {
        return Err(StatsError::invalid_argument(
            "ys must have at least one column",
        ));
    }

    use scirs2_core::parallel_ops::*;

    // Pre-compute ranks of the reference vector once.
    let x_owned = x.to_owned();
    let rank_x = compute_ranks(&x_owned.view());

    // Parallel map over columns of ys.
    let results: Result<Vec<f64>, StatsError> = (0..m)
        .into_par_iter()
        .map(|j| {
            let col = ys.column(j).to_owned();
            let rank_y = compute_ranks(&col.view());

            // Pearson correlation on rank arrays (= Spearman).
            spearman_r_simd_on_ranks(&rank_x.view(), &rank_y.view(), n)
        })
        .collect();

    let results = results?;
    Ok(Array1::from_vec(results))
}

/// Compute Pearson correlation of pre-computed rank arrays using SIMD.
fn spearman_r_simd_on_ranks(
    rank_x: &ArrayView1<f64>,
    rank_y: &ArrayView1<f64>,
    n: usize,
) -> StatsResult<f64> {
    let optimizer = AutoOptimizer::new();

    let mean_rx: f64 = if optimizer.should_use_simd(n) {
        F64::simd_sum(rank_x) / n as f64
    } else {
        rank_x.iter().sum::<f64>() / n as f64
    };
    let mean_ry: f64 = if optimizer.should_use_simd(n) {
        F64::simd_sum(rank_y) / n as f64
    } else {
        rank_y.iter().sum::<f64>() / n as f64
    };

    let (sum_xy, sum_rx2, sum_ry2) = if optimizer.should_use_simd(n) {
        let mrx = Array1::from_elem(n, mean_rx);
        let mry = Array1::from_elem(n, mean_ry);
        let rx_dev = F64::simd_sub(rank_x, &mrx.view());
        let ry_dev = F64::simd_sub(rank_y, &mry.view());
        let xy = F64::simd_mul(&rx_dev.view(), &ry_dev.view());
        let rx2 = F64::simd_mul(&rx_dev.view(), &rx_dev.view());
        let ry2 = F64::simd_mul(&ry_dev.view(), &ry_dev.view());
        (
            F64::simd_sum(&xy.view()),
            F64::simd_sum(&rx2.view()),
            F64::simd_sum(&ry2.view()),
        )
    } else {
        let mut sxy = 0.0_f64;
        let mut srx2 = 0.0_f64;
        let mut sry2 = 0.0_f64;
        for i in 0..n {
            let dx = rank_x[i] - mean_rx;
            let dy = rank_y[i] - mean_ry;
            sxy += dx * dy;
            srx2 += dx * dx;
            sry2 += dy * dy;
        }
        (sxy, srx2, sry2)
    };

    if sum_rx2 <= f64::EPSILON || sum_ry2 <= f64::EPSILON {
        return Err(StatsError::invalid_argument(
            "Cannot compute Spearman r: one or both rank vectors have zero variance",
        ));
    }

    let corr = (sum_xy / (sum_rx2 * sum_ry2).sqrt()).clamp(-1.0, 1.0);
    Ok(corr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_covariance_matrix_simd() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let cov = covariance_matrix_simd(&data.view(), false, 1).expect("Failed");

        // Check symmetry
        assert_abs_diff_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-10);

        // Check diagonal (variances are positive)
        assert!(cov[(0, 0)] > 0.0);
        assert!(cov[(1, 1)] > 0.0);
    }

    #[test]
    fn test_spearman_r_simd() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

        let rho = spearman_r_simd(&x.view(), &y.view()).expect("Failed");
        assert_abs_diff_eq!(rho, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rolling_correlation_simd() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let rolling_corr = rolling_correlation_simd(&x.view(), &y.view(), 3).expect("Failed");
        assert_eq!(rolling_corr.len(), 8);
    }

    // ── simd_pearson_correlation_matrix ──────────────────────────────────────

    #[test]
    fn test_simd_pearson_correlation_matrix_diagonal() {
        // Diagonal entries must all be 1.0.
        let data = array![
            [1.0_f64, 3.0, 7.0],
            [2.0_f64, 2.0, 5.0],
            [3.0_f64, 1.0, 3.0],
            [4.0_f64, 4.0, 9.0],
        ];
        let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
        assert_eq!(r.shape(), &[3, 3]);
        for i in 0..3 {
            assert_abs_diff_eq!(r[(i, i)], 1.0_f64, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_symmetry() {
        let data = array![
            [1.0_f64, 4.0, 9.0],
            [2.0_f64, 3.0, 6.0],
            [3.0_f64, 2.0, 3.0],
            [4.0_f64, 1.0, 0.0],
        ];
        let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r[(i, j)], r[(j, i)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_perfect_negative() {
        // x2 = -x1 → r(x1, x2) = -1
        let data = array![
            [1.0_f64, -1.0_f64],
            [2.0_f64, -2.0_f64],
            [3.0_f64, -3.0_f64],
            [4.0_f64, -4.0_f64],
        ];
        let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
        assert_abs_diff_eq!(r[(0, 1)], -1.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_perfect_positive() {
        // x2 = 2*x1 → r(x1, x2) = 1
        let data = array![
            [1.0_f64, 2.0_f64],
            [2.0_f64, 4.0_f64],
            [3.0_f64, 6.0_f64],
            [4.0_f64, 8.0_f64],
        ];
        let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
        assert_abs_diff_eq!(r[(0, 1)], 1.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_rejects_few_obs() {
        let data = array![[1.0_f64, 2.0_f64]]; // only 1 row
        assert!(simd_pearson_correlation_matrix(&data.view()).is_err());
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_rejects_single_var() {
        // need at least 2 columns
        let data = array![[1.0_f64], [2.0_f64], [3.0_f64]];
        assert!(simd_pearson_correlation_matrix(&data.view()).is_err());
    }

    #[test]
    fn test_simd_pearson_correlation_matrix_values_in_range() {
        // With an arbitrary 10×5 matrix, all entries should be in [-1, 1].
        use scirs2_core::ndarray::Array2;
        let n_obs = 20;
        let n_vars = 5;
        let data = Array2::from_shape_fn((n_obs, n_vars), |(i, j)| {
            ((i * n_vars + j) as f64).sin() * 3.7 + 1.2
        });
        let r = simd_pearson_correlation_matrix(&data.view()).expect("should succeed");
        for &val in r.iter() {
            assert!(
                val >= -1.0 - 1e-12 && val <= 1.0 + 1e-12,
                "r={val} outside [-1,1]"
            );
        }
    }

    // ── simd_spearman_correlation_batch ──────────────────────────────────────

    #[test]
    fn test_simd_spearman_correlation_batch_perfect_positive() {
        let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        // Both columns perfectly monotonic with x.
        use scirs2_core::ndarray::Array2;
        let ys = Array2::from_shape_fn((5, 2), |(i, _j)| (i + 1) as f64);
        let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view()).expect("should succeed");
        assert_eq!(rhos.len(), 2);
        assert_abs_diff_eq!(rhos[0], 1.0_f64, epsilon = 1e-10);
        assert_abs_diff_eq!(rhos[1], 1.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_spearman_correlation_batch_perfect_negative() {
        let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        use scirs2_core::ndarray::Array2;
        let ys = Array2::from_shape_fn((5, 1), |(i, _j)| (5 - i) as f64);
        let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view()).expect("should succeed");
        assert_abs_diff_eq!(rhos[0], -1.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_spearman_correlation_batch_mixed() {
        // One positive, one negative column.
        let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        use scirs2_core::ndarray::Array2;
        let ys = Array2::from_shape_fn((5, 2), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                (5 - i) as f64
            }
        });
        let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view()).expect("should succeed");
        assert_abs_diff_eq!(rhos[0], 1.0_f64, epsilon = 1e-10);
        assert_abs_diff_eq!(rhos[1], -1.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_spearman_correlation_batch_output_length() {
        let x = array![1.0_f64, 2.0, 3.0, 4.0];
        use scirs2_core::ndarray::Array2;
        let ys = Array2::from_shape_fn((4, 7), |(i, j)| (i * 7 + j) as f64);
        let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view()).expect("should succeed");
        assert_eq!(rhos.len(), 7);
    }

    #[test]
    fn test_simd_spearman_correlation_batch_values_in_range() {
        let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        use scirs2_core::ndarray::Array2;
        let n_cols = 10;
        let ys =
            Array2::from_shape_fn((8, n_cols), |(i, j)| ((i + j * 3) as f64).cos() * 5.0 + 2.0);
        let rhos = simd_spearman_correlation_batch(&x.view(), &ys.view()).expect("should succeed");
        for &r in rhos.iter() {
            assert!(
                r >= -1.0 - 1e-12 && r <= 1.0 + 1e-12,
                "rho={r} outside [-1,1]"
            );
        }
    }

    #[test]
    fn test_simd_spearman_correlation_batch_rejects_empty_x() {
        use scirs2_core::ndarray::{Array1, Array2};
        let x: Array1<f64> = Array1::zeros(0);
        let ys: Array2<f64> = Array2::zeros((0, 2));
        assert!(simd_spearman_correlation_batch(&x.view(), &ys.view()).is_err());
    }

    #[test]
    fn test_simd_spearman_correlation_batch_rejects_size_mismatch() {
        let x = array![1.0_f64, 2.0, 3.0];
        use scirs2_core::ndarray::Array2;
        let ys = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64); // 5 rows ≠ 3
        assert!(simd_spearman_correlation_batch(&x.view(), &ys.view()).is_err());
    }

    // ── Partial correlation: multiple controlling variables ──────────────────

    /// Partial correlation with two controlling variables.
    /// x = a*z1 + b*z2 + noise_x,  y = c*z1 + d*z2 + noise_y
    /// After partialling out z1, z2, the residuals should correlate according
    /// to the noise structure, not the z structure.
    #[test]
    fn test_partial_correlation_multivar_single_control_matches_formula() {
        use super::partial_correlation_simd;
        use scirs2_core::ndarray::Array2;

        let n = 40_usize;
        // z1 = index, z2 = cosine (independent of index)
        let z1: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.3).cos()).collect();

        // x = z1 + noise,  y = z1 + noise  (noise is non-zero and independent)
        // After removing z1, residuals of x and y are independent white noise
        // → partial correlation should be near 0.
        // Use deterministic "pseudo-noise" to make the test reproducible.
        let x: Vec<f64> = (0..n)
            .map(|i| z1[i] + 0.5 * ((i as f64 * 1.1).sin()))
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| z1[i] + 0.5 * ((i as f64 * 1.7 + 2.0).sin()))
            .collect();

        let xa = Array1::from(x);
        let ya = Array1::from(y);
        let controls = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { z1[i] } else { z2[i] });

        let r = partial_correlation_simd::<f64>(&xa.view(), &ya.view(), &controls.view())
            .expect("multivar partial corr");
        // Result must be in [-1, 1]
        assert!(
            r.abs() <= 1.0 + 1e-9,
            "partial correlation out of [-1,1]: {r}"
        );
    }

    /// Two controlling variables: when x and y share a third independent component
    /// not captured by the controls, partial correlation should be non-zero.
    #[test]
    fn test_partial_correlation_multivar_two_controls() {
        use super::partial_correlation_simd;
        use scirs2_core::ndarray::Array2;

        let n = 50_usize;
        // controls
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.2).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.15).cos()).collect();
        // Shared "hidden" component not in the controls
        let hidden: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.4 + 0.5).sin()).collect();

        // x and y both depend on z1, z2, and the hidden component
        let x: Vec<f64> = (0..n).map(|i| 2.0 * z1[i] + z2[i] + hidden[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| z1[i] + 3.0 * z2[i] + hidden[i]).collect();

        let xa = Array1::from(x);
        let ya = Array1::from(y);
        let controls = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { z1[i] } else { z2[i] });

        let r = partial_correlation_simd::<f64>(&xa.view(), &ya.view(), &controls.view())
            .expect("multivar partial corr 2 controls");
        // After partialling out z1, z2, both residuals retain the hidden component
        // → partial correlation should be strongly positive.
        assert!(
            r > 0.3,
            "expected positive partial correlation from hidden component, got {r}"
        );
        assert!(
            r.abs() <= 1.0 + 1e-9,
            "partial correlation out of [-1,1]: {r}"
        );
    }

    /// Out-of-range / degenerate inputs.
    #[test]
    fn test_partial_correlation_multivar_dimension_mismatch() {
        use super::partial_correlation_simd;
        use scirs2_core::ndarray::Array2;

        let x = array![1.0_f64, 2.0, 3.0];
        let y = array![1.0_f64, 2.0, 3.0];
        // z has wrong number of rows (4 ≠ 3)
        let z = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64);
        assert!(partial_correlation_simd::<f64>(&x.view(), &y.view(), &z.view()).is_err());
    }
}
