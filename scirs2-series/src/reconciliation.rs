//! Forecast Reconciliation Methods
//!
//! Forecast reconciliation transforms a set of *base* (incoherent) forecasts
//! for a hierarchical time series into *coherent* forecasts that satisfy the
//! aggregation constraints defined by the hierarchy's summing matrix S.
//!
//! # Methods implemented
//!
//! | Struct / Function | Method | Reference |
//! |---|---|---|
//! | [`MinTraceReconciliation::reconcile_ols`] | OLS reconciliation | Hyndman et al. (2011) |
//! | [`MinTraceReconciliation::reconcile_wls`] | WLS (variance-scaled) | Athanasopoulos et al. (2017) |
//! | [`MinTraceReconciliation::reconcile_mint_shrink`] | MinT Shrinkage | Wickramasuriya et al. (2019) |
//! | [`ErmReconciliation::reconcile`] | Empirical Risk Minimisation | Ben Taieb & Koo (2019) |
//! | [`nonnegative_reconcile`] | NNLS projection | Wickramasuriya et al. (2020) |
//! | [`mase_hierarchical`] | Hierarchical MASE evaluation | — |
//!
//! # References
//!
//! - Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. & Shang, H.L. (2011).
//!   "Optimal combination forecasts for hierarchical time series."
//!   *Computational Statistics & Data Analysis*, 55(9), 2579–2589.
//! - Athanasopoulos, G., Ahmed, R.A. & Hyndman, R.J. (2017).
//!   "The tourism forecasting competition." *International Journal of Forecasting*, 27, 822–844.
//! - Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. (2019).
//!   "Optimal forecast reconciliation using unbiased estimating equations."
//!   *Journal of the American Statistical Association*, 114(526), 804–819.
//! - Ben Taieb, S. & Koo, B. (2019).
//!   "Regularized regression for hierarchical forecasting without
//!   unbiasedness conditions." *KDD 2019*.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal linear-algebra helpers (no external linalg crate dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `A^T B` efficiently.
fn mat_transpose_mul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let (bm, p) = (b.shape()[0], b.shape()[1]);
    if m != bm {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m,
            actual: bm,
        });
    }
    let mut out = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += a[[k, i]] * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Compute `A B`.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (bk, n) = (b.shape()[0], b.shape()[1]);
    if k != bk {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: bk,
        });
    }
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Compute `A B` where B is a column vector (shape `(k,)`), returning a column.
fn mat_mul_vec(a: &Array2<f64>, v: &Array1<f64>) -> Result<Array1<f64>> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    if k != v.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: v.len(),
        });
    }
    let mut out = Array1::<f64>::zeros(m);
    for i in 0..m {
        for j in 0..k {
            out[i] += a[[i, j]] * v[j];
        }
    }
    Ok(out)
}

/// Invert a small symmetric positive-definite matrix via Cholesky decomposition.
///
/// Returns `Err` if the matrix is not positive definite (e.g., singular).
pub fn cholesky_inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // Cholesky: L L^T = A
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(TimeSeriesError::NumericalInstability(format!(
                        "Matrix is not positive-definite at diagonal ({i},{i}): s={s}"
                    )));
                }
                l[[i, i]] = s.sqrt();
            } else {
                let diag = l[[j, j]];
                if diag.abs() < f64::EPSILON * 1e6 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Near-zero diagonal in Cholesky".to_string(),
                    ));
                }
                l[[i, j]] = s / diag;
            }
        }
    }

    // Invert L by forward substitution: L Y = I
    let mut l_inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        for row in 0..n {
            let mut s = if row == col { 1.0 } else { 0.0 };
            for k in 0..row {
                s -= l[[row, k]] * l_inv[[k, col]];
            }
            let diag = l[[row, row]];
            if diag.abs() < f64::EPSILON * 1e6 {
                return Err(TimeSeriesError::NumericalInstability(
                    "Near-zero Cholesky diagonal during inversion".to_string(),
                ));
            }
            l_inv[[row, col]] = s / diag;
        }
    }

    // A^{-1} = (L^{-1})^T L^{-1}
    let mut a_inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            // sum_k  l_inv^T[i,k] * l_inv[k,j]  =  sum_k  l_inv[k,i] * l_inv[k,j]
            for k in 0..n {
                s += l_inv[[k, i]] * l_inv[[k, j]];
            }
            a_inv[[i, j]] = s;
        }
    }

    Ok(a_inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection matrix P for reconciliation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **projection matrix** `P = (S^T W^{-1} S)^{-1} S^T W^{-1}`
/// used by all MinT-family methods.
///
/// The reconciled forecasts are then `y_tilde = S P y_hat` where `y_hat` are
/// the base forecasts.
///
/// `W` must be a symmetric positive-definite matrix of shape `(n, n)`.
fn projection_matrix(s: &Array2<f64>, w_inv: &Array2<f64>) -> Result<Array2<f64>> {
    let n = s.shape()[0];
    let m = s.shape()[1];

    if w_inv.shape() != [n, n] {
        return Err(TimeSeriesError::InvalidInput(format!(
            "W_inv must be ({n},{n}), got ({},{})",
            w_inv.shape()[0],
            w_inv.shape()[1]
        )));
    }

    // C = S^T W^{-1}  shape (m, n)
    let c = mat_transpose_mul(s, w_inv)?; // (m, n) ← but wait: S^T is (m,n), W^{-1} is (n,n)
                                          // Actually mat_transpose_mul(s, w_inv) computes s^T * w_inv which is (m, n) * wait:
                                          // s shape (n,m), s^T shape (m,n). We want s^T w_inv = (m,n)(n,n) = (m,n).
                                          // mat_transpose_mul(s, w_inv): a=s (n,m), b=w_inv (n,n). The function computes a^T b = s^T w_inv.
                                          // Rows of a = n, cols of a = m.  a^T has shape (m,n). b has shape (n,n). -> out (m,n). Correct.
    let _ = m; // already used above

    // M = C S = S^T W^{-1} S  shape (m, m)
    let m_mat = mat_mul(&c, s)?; // (m,n) * (n,m) = (m,m)

    // M^{-1}
    let m_inv = cholesky_inverse(&m_mat)?;

    // P = M^{-1} C = M^{-1} S^T W^{-1}   shape (m, n)
    let p = mat_mul(&m_inv, &c)?; // (m,m) * (m,n) = (m,n)
    Ok(p)
}

// ─────────────────────────────────────────────────────────────────────────────
// MinTrace reconciliation
// ─────────────────────────────────────────────────────────────────────────────

/// MinTrace (MinT) forecast reconciliation.
///
/// Produces reconciled forecasts `y_tilde = S P y_hat` where P depends on the
/// covariance / weighting scheme chosen.
pub struct MinTraceReconciliation;

impl MinTraceReconciliation {
    /// **OLS reconciliation** — treats all base forecast errors as i.i.d.
    /// (`W = I`).
    ///
    /// The projection matrix simplifies to
    /// `P = (S^T S)^{-1} S^T`.
    ///
    /// # Arguments
    /// * `base_forecasts` — Array of shape `(n, h)`: one row per node, one
    ///   column per forecast horizon step.
    /// * `s` — Summing matrix of shape `(n, m)`.
    ///
    /// # Returns
    /// Reconciled forecasts of shape `(n, h)`.
    pub fn reconcile_ols(base_forecasts: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>> {
        let n = s.shape()[0];
        if base_forecasts.shape()[0] != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: base_forecasts.shape()[0],
            });
        }

        // W = I  →  W^{-1} = I
        let w_inv = Array2::<f64>::eye(n);
        let p = projection_matrix(s, &w_inv)?;

        // y_tilde = S P y_hat
        let sp = mat_mul(s, &p)?; // (n,m) * (m,n) = (n,n)
        mat_mul(&sp, base_forecasts) // (n,n) * (n,h) = (n,h)
    }

    /// **WLS reconciliation with variance scaling** — diagonal `W = diag(w)`.
    ///
    /// Each diagonal entry `w[i]` is the variance (or a proxy) of the base
    /// forecast error for node i.  Supplying `w = Array1::ones(n)` recovers
    /// OLS.
    ///
    /// # Arguments
    /// * `base_forecasts` — Array of shape `(n, h)`.
    /// * `s` — Summing matrix of shape `(n, m)`.
    /// * `w` — Length-n array of positive weights (variances).
    ///
    /// # Returns
    /// Reconciled forecasts of shape `(n, h)`.
    pub fn reconcile_wls(
        base_forecasts: &Array2<f64>,
        s: &Array2<f64>,
        w: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n = s.shape()[0];
        if base_forecasts.shape()[0] != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: base_forecasts.shape()[0],
            });
        }
        if w.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: w.len(),
            });
        }
        for (i, &wi) in w.iter().enumerate() {
            if wi <= 0.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: format!("w[{i}]"),
                    message: "Weight must be strictly positive".to_string(),
                });
            }
        }

        // W^{-1} = diag(1/w)
        let mut w_inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            w_inv[[i, i]] = 1.0 / w[i];
        }

        let p = projection_matrix(s, &w_inv)?;
        let sp = mat_mul(s, &p)?;
        mat_mul(&sp, base_forecasts)
    }

    /// **MinT Shrinkage reconciliation** — estimates the full covariance matrix
    /// of the base forecast errors from an in-sample residual matrix and applies
    /// Ledoit-Wolf shrinkage before inverting.
    ///
    /// The shrinkage target is the diagonal of the sample covariance (variance
    /// matrix).  The optimal shrinkage intensity `λ*` is computed using the
    /// Oracle Approximating Shrinkage (OAS) closed-form estimate.
    ///
    /// # Arguments
    /// * `base_forecasts` — Array of shape `(n, h)`.
    /// * `s` — Summing matrix of shape `(n, m)`.
    /// * `residuals` — Array of shape `(n, T)` of in-sample one-step-ahead
    ///   residuals, where T ≥ n + 1.
    ///
    /// # Returns
    /// Reconciled forecasts of shape `(n, h)`.
    pub fn reconcile_mint_shrink(
        base_forecasts: &Array2<f64>,
        s: &Array2<f64>,
        residuals: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n = s.shape()[0];
        let t_obs = residuals.shape()[1];

        if base_forecasts.shape()[0] != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: base_forecasts.shape()[0],
            });
        }
        if residuals.shape()[0] != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: residuals.shape()[0],
            });
        }
        if t_obs < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 residual observations for covariance estimation"
                    .to_string(),
                required: 2,
                actual: t_obs,
            });
        }

        // Sample covariance Sigma_hat = (1/(T-1)) * E E^T  (E is centred)
        let t_f64 = t_obs as f64;
        let means: Vec<f64> = (0..n)
            .map(|i| residuals.row(i).iter().copied().sum::<f64>() / t_f64)
            .collect();

        let mut sigma = Array2::<f64>::zeros((n, n));
        for t in 0..t_obs {
            for i in 0..n {
                let ei = residuals[[i, t]] - means[i];
                for j in 0..=i {
                    let ej = residuals[[j, t]] - means[j];
                    sigma[[i, j]] += ei * ej;
                    if i != j {
                        sigma[[j, i]] += ei * ej;
                    }
                }
            }
        }
        let scale = 1.0 / (t_f64 - 1.0);
        for v in sigma.iter_mut() {
            *v *= scale;
        }

        // Ledoit-Wolf shrinkage towards diagonal target D = diag(sigma)
        // lambda* (OAS formula approximation):
        // phi = ||Sigma_hat - D||_F^2 / ||Sigma_hat||_F^2  (off-diagonal contribution)
        // rho* = min(1, max(0, phi / (1 + phi * (1 - 1/n))))
        let mut sum_off_sq = 0.0_f64;
        let mut sum_all_sq = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let v = sigma[[i, j]];
                sum_all_sq += v * v;
                if i != j {
                    sum_off_sq += v * v;
                }
            }
        }

        let rho = if sum_all_sq < f64::EPSILON {
            0.0_f64
        } else {
            let phi = sum_off_sq / sum_all_sq;
            let denom = 1.0 + phi * (1.0 - 1.0 / n as f64);
            (phi / denom).clamp(0.0, 1.0)
        };

        // Shrunk covariance: W = (1-rho)*Sigma + rho*D
        let mut w_shrunk = sigma.clone();
        for i in 0..n {
            for j in 0..n {
                w_shrunk[[i, j]] *= 1.0 - rho;
            }
            // add rho * diagonal entry
            w_shrunk[[i, i]] += rho * sigma[[i, i]];
        }

        // Regularise diagonal to ensure positive definiteness.
        let eps_reg = 1e-8
            * (0..n)
                .map(|i| w_shrunk[[i, i]])
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1e-8);
        for i in 0..n {
            w_shrunk[[i, i]] += eps_reg;
        }

        let w_inv = cholesky_inverse(&w_shrunk)?;
        let p = projection_matrix(s, &w_inv)?;
        let sp = mat_mul(s, &p)?;
        mat_mul(&sp, base_forecasts)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ERM reconciliation
// ─────────────────────────────────────────────────────────────────────────────

/// **Empirical Risk Minimisation (ERM) reconciliation** (Ben Taieb & Koo, 2019).
///
/// Learns a linear reconciliation mapping `G` from a training set of
/// (base_forecast, actual) pairs by minimising the mean squared error
/// subject to the constraint that the mapping is *coherent* (i.e.
/// `S G S = S G`, which is satisfied when G maps into the column space of S).
///
/// In practice this implementation solves the unconstrained problem
/// `min_G || Y - S G Y_hat ||_F^2` using the Moore-Penrose pseudo-inverse of
/// `Y_hat` (ridge-regularised for numerical stability), then projects G onto
/// the coherence subspace via `G ← (S^T S)^{-1} S^T G`.
pub struct ErmReconciliation {
    /// Ridge regularisation parameter for the pseudo-inverse.
    pub lambda: f64,
}

impl ErmReconciliation {
    /// Create an ERM reconciler with the specified ridge penalty.
    ///
    /// `lambda = 0` corresponds to the unregularised OLS solution.
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda < 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "lambda".to_string(),
                message: "Ridge parameter must be non-negative".to_string(),
            });
        }
        Ok(Self { lambda })
    }

    /// Fit and apply ERM reconciliation.
    ///
    /// # Arguments
    /// * `base_forecasts_train` — Training base forecasts, shape `(n, T_train)`.
    /// * `actuals_train` — Training actuals, shape `(n, T_train)`.
    /// * `base_forecasts_test` — Test base forecasts, shape `(n, h)`.
    /// * `s` — Summing matrix of shape `(n, m)`.
    ///
    /// # Returns
    /// Reconciled forecasts of shape `(n, h)`.
    pub fn reconcile(
        &self,
        base_forecasts_train: &Array2<f64>,
        actuals_train: &Array2<f64>,
        base_forecasts_test: &Array2<f64>,
        s: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n = s.shape()[0];
        let t_train = base_forecasts_train.shape()[1];

        if base_forecasts_train.shape()[0] != n
            || actuals_train.shape() != base_forecasts_train.shape()
        {
            return Err(TimeSeriesError::InvalidInput(
                "Training arrays must have shape (n, T_train)".to_string(),
            ));
        }
        if base_forecasts_test.shape()[0] != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: base_forecasts_test.shape()[0],
            });
        }

        // We want G: n×n  such that S G base_train ≈ actuals_train.
        // Unconstrained solution:  G* = actuals_train * pinv(base_train) in col-space.
        // With ridge:  G* = Y * Y_hat^T (Y_hat Y_hat^T + lambda I)^{-1}
        // where Y = actuals_train, Y_hat = base_forecasts_train.

        // B = Y_hat Y_hat^T + lambda I  (n×n)
        let mut b = Array2::<f64>::zeros((n, n));
        for t in 0..t_train {
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] += base_forecasts_train[[i, t]] * base_forecasts_train[[j, t]];
                }
            }
        }
        for i in 0..n {
            b[[i, i]] += self.lambda;
        }

        // C = Y Y_hat^T  (n×n)
        let mut c = Array2::<f64>::zeros((n, n));
        for t in 0..t_train {
            for i in 0..n {
                for j in 0..n {
                    c[[i, j]] += actuals_train[[i, t]] * base_forecasts_train[[j, t]];
                }
            }
        }

        // G_raw = C B^{-1}
        let b_inv = cholesky_inverse(&b)?;
        let g_raw = mat_mul(&c, &b_inv)?;

        // Project G onto coherence subspace: G_coherent = S (S^T S)^{-1} S^T G_raw
        // First compute (S^T S)^{-1}  (m×m)
        let sts = mat_transpose_mul(s, s)?; // s^T s shape (m,m)
                                            // wait: mat_transpose_mul(a,b) computes a^T b. Here a=s (n,m), b=s (n,m).
                                            // a^T b = s^T s (m,n)(n,m) = (m,m). But let me double-check signature:
                                            // mat_transpose_mul(a,b): a (m_in,n_in), b (m_in, p) → output (n_in, p)
                                            // so mat_transpose_mul(s, s): a=s(n,m), b=s(n,m) → (m, m). Correct.
        let sts_inv = cholesky_inverse(&sts)?;

        // P_ols = (S^T S)^{-1} S^T   shape (m, n)
        let st = {
            // compute S^T explicitly
            let (rows, cols) = (s.shape()[0], s.shape()[1]);
            let mut st = Array2::<f64>::zeros((cols, rows));
            for i in 0..rows {
                for j in 0..cols {
                    st[[j, i]] = s[[i, j]];
                }
            }
            st
        };
        let p_ols = mat_mul(&sts_inv, &st)?; // (m,m)(m,n) = (m,n)
        let sp_ols = mat_mul(s, &p_ols)?; // (n,m)(m,n) = (n,n)  (= projection onto col-space of S)

        // G_coherent = sp_ols * G_raw  (n,n)(n,n) = (n,n)
        let g_coherent = mat_mul(&sp_ols, &g_raw)?;

        // Apply to test set
        mat_mul(&g_coherent, base_forecasts_test)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-negative reconciliation (NNLS projection)
// ─────────────────────────────────────────────────────────────────────────────

/// Non-negative forecast reconciliation.
///
/// Applies OLS reconciliation, then projects negative values to zero and
/// renormalises so that the aggregation constraints are approximately
/// maintained.  This is a fast heuristic; for an exact NNLS projection see
/// Wickramasuriya et al. (2020).
///
/// # Arguments
/// * `forecasts` — Base forecasts of shape `(n, h)`.
/// * `s` — Summing matrix of shape `(n, m)`.
///
/// # Returns
/// Non-negative reconciled forecasts of shape `(n, h)`.
pub fn nonnegative_reconcile(forecasts: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>> {
    // Step 1: OLS reconciliation.
    let mut reconciled = MinTraceReconciliation::reconcile_ols(forecasts, s)?;

    let m = s.shape()[1];
    let n = s.shape()[0];
    let h = reconciled.shape()[1];

    // Step 2: For each horizon h, project the bottom-level part to be ≥ 0 and
    //         then re-aggregate.
    // Find bottom-level node ids (rows of S with exactly one 1 and all 0s else).
    let bottom_ids: Vec<usize> = (0..n)
        .filter(|&i| {
            let row = s.row(i);
            let ones: usize = row.iter().filter(|&&v| v > 0.5).count();
            ones == 1
        })
        .collect();

    for t in 0..h {
        // Clamp bottom-level values to zero.
        for &i in &bottom_ids {
            if reconciled[[i, t]] < 0.0 {
                reconciled[[i, t]] = 0.0;
            }
        }

        // Re-aggregate upper levels from bottom.
        for i in 0..n {
            // Skip bottom-level rows.
            if bottom_ids.contains(&i) {
                continue;
            }
            let mut agg = 0.0_f64;
            for j in 0..m {
                // Find the bottom node corresponding to column j.
                // Column j corresponds to the j-th bottom-level node.
                // We need reconciled value for that node.
                let bottom_node = bottom_ids.get(j);
                if let Some(&bn) = bottom_node {
                    agg += s[[i, j]] * reconciled[[bn, t]];
                }
            }
            reconciled[[i, t]] = agg;
        }
    }

    Ok(reconciled)
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation: Hierarchical MASE
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **Mean Absolute Scaled Error (MASE)** for each node in the
/// hierarchy.
///
/// MASE for a single series is defined as:
/// `MASE = MAE_forecast / MAE_naive`
/// where `MAE_naive` is the mean absolute error of the in-sample naïve
/// seasonal (or non-seasonal) benchmark.
///
/// Here we use the one-step-ahead naïve benchmark (lag-1 difference on the
/// training series) for all nodes.
///
/// # Arguments
/// * `actuals` — Out-of-sample actuals, shape `(n, h)`.
/// * `forecasts` — Point forecasts, shape `(n, h)`.
/// * `training` — In-sample training data, shape `(n, T_train)`.
///
/// # Returns
/// An `Array1<f64>` of length n with the per-node MASE values.
pub fn mase_hierarchical(
    actuals: &Array2<f64>,
    forecasts: &Array2<f64>,
    training: &Array2<f64>,
) -> Result<Array1<f64>> {
    let n = actuals.shape()[0];
    let h = actuals.shape()[1];

    if forecasts.shape() != actuals.shape() {
        return Err(TimeSeriesError::InvalidInput(format!(
            "forecasts shape {:?} != actuals shape {:?}",
            forecasts.shape(),
            actuals.shape()
        )));
    }
    if training.shape()[0] != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: training.shape()[0],
        });
    }
    let t_train = training.shape()[1];
    if t_train < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "MASE requires at least 2 training observations".to_string(),
            required: 2,
            actual: t_train,
        });
    }

    let h_f64 = h as f64;
    let mut result = Array1::<f64>::zeros(n);

    for i in 0..n {
        // MAE of forecast
        let mae_f: f64 = (0..h)
            .map(|t| (actuals[[i, t]] - forecasts[[i, t]]).abs())
            .sum::<f64>()
            / h_f64;

        // Naive benchmark MAE (lag-1) on training data
        let mae_naive: f64 = (1..t_train)
            .map(|t| (training[[i, t]] - training[[i, t - 1]]).abs())
            .sum::<f64>()
            / (t_train - 1) as f64;

        result[i] = if mae_naive < f64::EPSILON {
            // Degenerate: training series is constant; return ratio w.r.t. 1.
            mae_f
        } else {
            mae_f / mae_naive
        };
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural covariance methods (WLS variants)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate diagonal WLS weights from **sample variances** of in-sample residuals.
///
/// Each weight is the sample variance of the residuals for the corresponding
/// node.  This produces the "structural scaling" variant of WLS described in
/// Hyndman et al. (2011).
///
/// # Arguments
/// * `residuals` — In-sample residuals, shape `(n, T)`.
///
/// # Returns
/// Length-n weight array (variances; all strictly positive after a small
/// regularisation floor is added).
pub fn wls_variance_weights(residuals: &Array2<f64>) -> Result<Array1<f64>> {
    let n = residuals.shape()[0];
    let t_obs = residuals.shape()[1];

    if t_obs < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 observations for variance estimation".to_string(),
            required: 2,
            actual: t_obs,
        });
    }

    let t_f64 = t_obs as f64;
    let mut weights = Array1::<f64>::zeros(n);

    for i in 0..n {
        let mean = residuals.row(i).iter().copied().sum::<f64>() / t_f64;
        let var = residuals
            .row(i)
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>()
            / (t_f64 - 1.0);
        // Floor at a small positive value to avoid division by zero.
        weights[i] = var.max(1e-10);
    }

    Ok(weights)
}

/// Estimate WLS weights as the **number of series summed** at each node (the
/// "structural" or "series-count" scaling).
///
/// For a leaf node the weight is 1; for an aggregate node it equals the number
/// of bottom-level series it aggregates.
///
/// # Arguments
/// * `s` — Summing matrix of shape `(n, m)`.
///
/// # Returns
/// Length-n weight array.
pub fn wls_structural_weights(s: &Array2<f64>) -> Array1<f64> {
    let n = s.shape()[0];
    let mut weights = Array1::<f64>::zeros(n);
    for i in 0..n {
        let count: f64 = s.row(i).iter().copied().sum();
        weights[i] = count.max(1.0);
    }
    weights
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// =============================================================================
// Alias / convenience wrappers matching the task specification
// =============================================================================

/// Convenience wrapper for OLS reconciliation (alias for [`MinTraceReconciliation::reconcile_ols`]).
///
/// P = (S^T S)^{-1} S^T
/// Reconciled: y_tilde = S P y_hat
pub struct OLSReconciliation;

impl OLSReconciliation {
    /// Reconcile `base_forecasts` (shape `[n_series, T]`) using summation matrix `s`.
    pub fn reconcile(base_forecasts: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>> {
        MinTraceReconciliation::reconcile_ols(base_forecasts, s)
    }
}

/// Convenience wrapper for WLS reconciliation (alias for [`MinTraceReconciliation::reconcile_wls`]).
///
/// Weighted by the inverse of per-series forecast error variances.
pub struct WLSReconciliation;

impl WLSReconciliation {
    /// Reconcile with explicit variance weights `w` (length `n_series`).
    pub fn reconcile(
        base_forecasts: &Array2<f64>,
        s: &Array2<f64>,
        w: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        MinTraceReconciliation::reconcile_wls(base_forecasts, s, w)
    }
}

/// Convenience wrapper for MinT reconciliation (alias for [`MinTraceReconciliation::reconcile_mint_shrink`]).
pub struct MinTReconciliation;

impl MinTReconciliation {
    /// Reconcile using the shrinkage estimator for the residual covariance matrix.
    ///
    /// P = (S^T W^{-1} S)^{-1} S^T W^{-1}
    /// W is estimated with the Schäfer-Strimmer shrinkage estimator.
    pub fn reconcile(
        base_forecasts: &Array2<f64>,
        s: &Array2<f64>,
        residuals: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        MinTraceReconciliation::reconcile_mint_shrink(base_forecasts, s, residuals)
    }
}

/// Bottom-up reconciliation.
///
/// Bottom-level forecasts are summed according to the hierarchy.
/// Upper-level forecasts are replaced by aggregating the bottom level.
pub struct BottomUpReconciliation;

impl BottomUpReconciliation {
    /// Reconcile `base_forecasts` (shape `[n_series, T]`) using `s`.
    ///
    /// Returns coherent forecasts where all upper levels are derived
    /// from the summed bottom-level.
    pub fn reconcile(base_forecasts: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>> {
        let n_all = s.shape()[0];
        let n_bottom = s.shape()[1];
        let t = base_forecasts.shape()[1];

        if base_forecasts.shape()[0] != n_all {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_all,
                actual: base_forecasts.shape()[0],
            });
        }
        if n_all < n_bottom {
            return Err(TimeSeriesError::InvalidInput(
                "S has more columns (bottom series) than rows (total series)".to_string(),
            ));
        }

        // Extract bottom-level forecasts (last n_bottom rows by convention)
        let bottom_start = n_all - n_bottom;
        let bottom = base_forecasts
            .slice(scirs2_core::ndarray::s![bottom_start.., ..])
            .to_owned();

        // Compute S * y_bottom
        let mut out = Array2::<f64>::zeros((n_all, t));
        for i in 0..n_all {
            for tt in 0..t {
                let mut sum = 0.0_f64;
                for j in 0..n_bottom {
                    sum += s[[i, j]] * bottom[[j, tt]];
                }
                out[[i, tt]] = sum;
            }
        }
        Ok(out)
    }
}

/// Method for top-down proportional disaggregation.
#[derive(Debug, Clone, PartialEq)]
pub enum TopDownMethod {
    /// Use historical averages of bottom proportions.
    AverageHistoricalProportion,
    /// Proportional to historical medians of absolute percentages.
    ProportionalMedianAbsolutePct,
    /// Simple equal-weight proportional split.
    Proportional,
}

/// Top-down reconciliation.
///
/// The top-level aggregate forecast is disaggregated to bottom series
/// using historical proportion estimates.
pub struct TopDownReconciliation {
    /// Disaggregation method.
    pub method: TopDownMethod,
}

impl TopDownReconciliation {
    /// Create a new top-down reconciliation with the specified `method`.
    pub fn new(method: TopDownMethod) -> Self {
        Self { method }
    }

    /// Reconcile using historical `actuals` (shape `[n_series, T_hist]`) to
    /// estimate proportions, then disaggregate `base_forecasts`
    /// (shape `[1, T_future]` for the aggregate) into all `n_bottom` series.
    ///
    /// `s` must have shape `[n_all, n_bottom]`.
    pub fn reconcile(
        &self,
        base_forecasts: &Array2<f64>,
        s: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_all = s.shape()[0];
        let n_bottom = s.shape()[1];
        let t = base_forecasts.shape()[1];

        if base_forecasts.shape()[0] != n_all {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_all,
                actual: base_forecasts.shape()[0],
            });
        }
        if actuals.shape()[0] != n_all {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_all,
                actual: actuals.shape()[0],
            });
        }

        let bottom_start = n_all - n_bottom;

        // Estimate proportions from actuals
        let t_hist = actuals.shape()[1];
        let mut proportions = vec![0.0_f64; n_bottom];

        for j in 0..n_bottom {
            let bottom_idx = bottom_start + j;
            let prop = match self.method {
                TopDownMethod::AverageHistoricalProportion => {
                    // Mean of bottom[j,t] / top[t] over history
                    let mut sum = 0.0_f64;
                    let mut count = 0_usize;
                    for tt in 0..t_hist {
                        let top = actuals[[0, tt]];
                        if top.abs() > 1e-12 {
                            sum += actuals[[bottom_idx, tt]] / top;
                            count += 1;
                        }
                    }
                    if count == 0 {
                        1.0 / n_bottom as f64
                    } else {
                        sum / count as f64
                    }
                }
                TopDownMethod::ProportionalMedianAbsolutePct => {
                    // Median of |bottom[j,t]| / (sum of |bottom[*,t]|)
                    let mut ratios: Vec<f64> = Vec::with_capacity(t_hist);
                    for tt in 0..t_hist {
                        let total: f64 = (0..n_bottom)
                            .map(|jj| actuals[[bottom_start + jj, tt]].abs())
                            .sum();
                        if total > 1e-12 {
                            ratios.push(actuals[[bottom_idx, tt]].abs() / total);
                        }
                    }
                    if ratios.is_empty() {
                        1.0 / n_bottom as f64
                    } else {
                        ratios
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = ratios.len() / 2;
                        if ratios.len() % 2 == 0 {
                            (ratios[mid - 1] + ratios[mid]) / 2.0
                        } else {
                            ratios[mid]
                        }
                    }
                }
                TopDownMethod::Proportional => {
                    // Equal proportions
                    1.0 / n_bottom as f64
                }
            };
            proportions[j] = prop;
        }

        // Normalise proportions to sum to 1
        let prop_sum: f64 = proportions.iter().sum();
        if prop_sum > 1e-12 {
            for p in &mut proportions {
                *p /= prop_sum;
            }
        } else {
            for p in &mut proportions {
                *p = 1.0 / n_bottom as f64;
            }
        }

        // Build output: bottom series = proportion * top_forecast
        let mut out = Array2::<f64>::zeros((n_all, t));
        for j in 0..n_bottom {
            for tt in 0..t {
                out[[bottom_start + j, tt]] = proportions[j] * base_forecasts[[0, tt]];
            }
        }

        // Upper levels = S * bottom
        let bottom_slice = out
            .slice(scirs2_core::ndarray::s![bottom_start.., ..])
            .to_owned();
        for i in 0..bottom_start {
            for tt in 0..t {
                let mut sum = 0.0_f64;
                for j in 0..n_bottom {
                    sum += s[[i, j]] * bottom_slice[[j, tt]];
                }
                out[[i, tt]] = sum;
            }
        }

        Ok(out)
    }
}

/// Utility for building a hierarchy summation matrix from a parent list.
///
/// # Example
///
/// ```rust
/// use scirs2_series::reconciliation::SummationMatrix;
///
/// // Three nodes: 0=Total, 1=GroupA, 2=GroupB
/// // Parents: GroupA (1) and GroupB (2) are bottom; Total (0) sums both.
/// let parents: Vec<Option<usize>> = vec![None, Some(0), Some(0)];
/// let s = SummationMatrix::from_parents(&parents).expect("should succeed");
/// // S should be 3×2 (3 nodes, 2 leaf/bottom nodes)
/// assert_eq!(s.shape()[0], 3);
/// assert_eq!(s.shape()[1], 2);
/// ```
pub struct SummationMatrix;

impl SummationMatrix {
    /// Build S from a parent array.
    ///
    /// Nodes with `None` as parent are assumed to be roots/aggregates.
    /// Nodes that have no children are treated as bottom (leaf) nodes.
    ///
    /// Returns the summation matrix `S` of shape `[n_nodes, n_leaves]`.
    pub fn from_parents(parents: &[Option<usize>]) -> Result<Array2<f64>> {
        let n = parents.len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Empty parent list".to_string(),
            ));
        }

        // Identify leaf nodes (nodes that are not parents of anyone)
        let mut is_parent = vec![false; n];
        for &p in parents.iter().flatten() {
            if p < n {
                is_parent[p] = true;
            }
        }
        let leaves: Vec<usize> = (0..n).filter(|&i| !is_parent[i]).collect();
        let n_leaves = leaves.len();

        if n_leaves == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "No leaf nodes found in hierarchy".to_string(),
            ));
        }

        // Build ancestor lookup: for each node, which leaves are in its subtree?
        // We do this by BFS/DFS: for each leaf, walk up the parent chain.
        let mut s = Array2::<f64>::zeros((n, n_leaves));

        for (col, &leaf) in leaves.iter().enumerate() {
            let mut node = leaf;
            loop {
                s[[node, col]] = 1.0;
                match parents[node] {
                    Some(p) if p < n => node = p,
                    _ => break,
                }
            }
        }

        Ok(s)
    }

    /// Build a balanced two-level hierarchy: one root, `n` bottom series.
    pub fn two_level(n_bottom: usize) -> Result<Array2<f64>> {
        if n_bottom == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_bottom must be at least 1".to_string(),
            ));
        }
        let mut s = Array2::<f64>::zeros((n_bottom + 1, n_bottom));
        // Row 0 = aggregate (sum all)
        for j in 0..n_bottom {
            s[[0, j]] = 1.0;
        }
        // Rows 1..=n_bottom = identity
        for j in 0..n_bottom {
            s[[j + 1, j]] = 1.0;
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchical::{summing_matrix, Hierarchy};
    use scirs2_core::ndarray::array;

    fn build_simple_hierarchy() -> (Hierarchy, Array2<f64>) {
        // Total -> [A, B] -> [A1, A2, B1]
        let mut h = Hierarchy::new();
        let total = h.add_node("Total", None).expect("failed to create total");
        let a = h.add_node("A", Some(total)).expect("failed to create a");
        let b = h.add_node("B", Some(total)).expect("failed to create b");
        h.add_node("A1", Some(a)).expect("unexpected None or Err");
        h.add_node("A2", Some(a)).expect("unexpected None or Err");
        h.add_node("B1", Some(b)).expect("unexpected None or Err");
        let s = summing_matrix(&h).expect("failed to create s");
        (h, s)
    }

    #[test]
    fn test_ols_reconciliation_coherence() {
        let (_, s) = build_simple_hierarchy();
        let n = s.shape()[0]; // 6
        let h = 2;

        // Create incoherent base forecasts
        let mut base = Array2::<f64>::zeros((n, h));
        base[[0, 0]] = 100.0;
        base[[1, 0]] = 42.0;
        base[[2, 0]] = 61.0; // incoherent: 42+61 != 100
        base[[3, 0]] = 20.0;
        base[[4, 0]] = 25.0;
        base[[5, 0]] = 58.0;
        base[[0, 1]] = 110.0;
        base[[1, 1]] = 45.0;
        base[[2, 1]] = 68.0;
        base[[3, 1]] = 22.0;
        base[[4, 1]] = 26.0;
        base[[5, 1]] = 65.0;

        let reconciled =
            MinTraceReconciliation::reconcile_ols(&base, &s).expect("failed to create reconciled");
        assert_eq!(reconciled.shape(), &[6, 2]);

        // Check coherence: total == sum of leaves
        let m = s.shape()[1]; // 3 bottom nodes
        for t in 0..h {
            let mut bottom_sum = 0.0_f64;
            for j in 0..m {
                // Find the leaf node for column j (row of S with exactly one 1)
                // rows 3,4,5 are leaves (indices 3,4,5 in hierarchy)
                bottom_sum += reconciled[[3 + j, t]];
            }
            let total = reconciled[[0, t]];
            assert!(
                (total - bottom_sum).abs() < 1e-6,
                "Coherence violated at t={t}: total={total}, bottom_sum={bottom_sum}"
            );
        }
    }

    #[test]
    fn test_wls_reconciliation() {
        let (_, s) = build_simple_hierarchy();
        let n = s.shape()[0];
        let base = Array2::<f64>::ones((n, 1));
        let weights = Array1::from_vec(vec![2.0; n]);
        let result = MinTraceReconciliation::reconcile_wls(&base, &s, &weights)
            .expect("failed to create result");
        assert_eq!(result.shape(), &[6, 1]);
    }

    #[test]
    fn test_mint_shrink_reconciliation() {
        let (_, s) = build_simple_hierarchy();
        let n = s.shape()[0];
        let h = 1;
        let base = Array2::<f64>::ones((n, h));

        // Generate simple residuals
        let t_r = 20;
        let mut residuals = Array2::<f64>::zeros((n, t_r));
        for i in 0..n {
            for t in 0..t_r {
                residuals[[i, t]] = ((i + t) as f64) * 0.1 - 0.5;
            }
        }

        let result = MinTraceReconciliation::reconcile_mint_shrink(&base, &s, &residuals)
            .expect("failed to create result");
        assert_eq!(result.shape(), &[6, 1]);
    }

    #[test]
    fn test_mase_hierarchical() {
        let n = 3;
        let h = 4;
        let t_train = 10;

        let actuals = Array2::<f64>::ones((n, h)) * 5.0;
        let forecasts = Array2::<f64>::ones((n, h)) * 4.0; // MAE = 1.0 per step
        let mut training = Array2::<f64>::zeros((n, t_train));
        // constant training: naive MAE = 0 → should fall back to raw MAE
        for i in 0..n {
            for t in 0..t_train {
                training[[i, t]] = 5.0 + (t as f64) * 0.1; // slow ramp → naive MAE = 0.1
            }
        }

        let mase =
            mase_hierarchical(&actuals, &forecasts, &training).expect("failed to create mase");
        assert_eq!(mase.len(), n);
        for &v in mase.iter() {
            assert!(v > 0.0, "MASE must be positive");
        }
    }

    #[test]
    fn test_nonnegative_reconcile() {
        let (_, s) = build_simple_hierarchy();
        let n = s.shape()[0];
        let mut base = Array2::<f64>::zeros((n, 2));
        // Set one negative base forecast.
        base[[3, 0]] = -5.0;
        base[[4, 0]] = 20.0;
        base[[5, 0]] = 30.0;
        base[[1, 0]] = 15.0;
        base[[2, 0]] = 30.0;
        base[[0, 0]] = 45.0;
        base[[3, 1]] = 10.0;
        base[[4, 1]] = 15.0;
        base[[5, 1]] = 20.0;
        base[[1, 1]] = 25.0;
        base[[2, 1]] = 20.0;
        base[[0, 1]] = 45.0;

        let result = nonnegative_reconcile(&base, &s).expect("failed to create result");
        assert_eq!(result.shape(), &[6, 2]);
        // Bottom-level entries must be non-negative.
        for t in 0..2 {
            for &i in &[3usize, 4, 5] {
                assert!(
                    result[[i, t]] >= -1e-10,
                    "Negative bottom-level value at node {i}, t={t}: {}",
                    result[[i, t]]
                );
            }
        }
    }

    #[test]
    fn test_wls_structural_weights() {
        let (_, s) = build_simple_hierarchy();
        let weights = wls_structural_weights(&s);
        // Total row sums all 3 bottom series → weight 3
        assert!((weights[0] - 3.0).abs() < 1e-9);
        // Leaf nodes → weight 1
        assert!((weights[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cholesky_inverse() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let a_inv = cholesky_inverse(&a).expect("failed to create a_inv");
        // A A^{-1} should be identity
        let prod = mat_mul(&a, &a_inv).expect("failed to create prod");
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((prod[[0, 1]]).abs() < 1e-9);
        assert!((prod[[1, 0]]).abs() < 1e-9);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-9);
    }
}
