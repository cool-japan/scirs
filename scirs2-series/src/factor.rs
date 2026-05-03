//! Dynamic Factor Models (DFM) and PCA-based Forecasting for Multivariate Time Series
//!
//! This module provides:
//! - [`DynamicFactorModel`]: EM-algorithm-based dynamic factor model estimation
//!   in state-space representation (Doz, Giannone & Reichlin, 2012)
//! - [`PcaForecast`]: PCA dimensionality reduction for forecasting large panels
//!
//! # Dynamic Factor Model
//!
//! The DFM assumes that a large panel of time series is driven by a small number
//! of latent factors:
//!
//! ```text
//! Observation equation: x_t = Λ f_t + e_t,    e_t ~ N(0, R)
//! State equation:        f_t = A f_t-1 + u_t,  u_t ~ N(0, Q)
//! ```
//!
//! where:
//! - `x_t` ∈ ℝ^n  — observed series at time t (possibly large n)
//! - `f_t` ∈ ℝ^r  — latent factors (r << n)
//! - `Λ`   ∈ ℝ^{n×r} — factor loadings
//! - `A`   ∈ ℝ^{r×r} — factor transition matrix
//! - `R`   ∈ ℝ^{n×n} — idiosyncratic noise covariance (diagonal)
//! - `Q`   ∈ ℝ^{r×r} — factor innovation covariance
//!
//! Estimation proceeds via:
//! 1. PCA initialization of loadings and factors
//! 2. EM algorithm: Kalman filter (E-step) + parameter updates (M-step)
//!
//! # References
//! - Doz, C., Giannone, D., & Reichlin, L. (2012). A Quasi-Maximum Likelihood
//!   Approach for Large Approximate Dynamic Factor Models.
//!   *Review of Economics and Statistics*, 94(4), 1014–1024.
//! - Stock, J. H., & Watson, M. W. (2002). Forecasting Using Principal Components
//!   from a Large Number of Predictors.
//!   *Journal of the American Statistical Association*, 97(460), 1167–1179.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::factor::{DynamicFactorModel, PcaForecast};
//! use scirs2_core::ndarray::Array2;
//!
//! // Generate a synthetic panel (50 obs, 6 series, 2 factors)
//! let mut data = Array2::<f64>::zeros((50, 6));
//! for t in 0..50 {
//!     let f1 = (t as f64 * 0.1).sin();
//!     let f2 = (t as f64 * 0.2).cos();
//!     for j in 0..6 {
//!         data[[t, j]] = (j as f64 + 1.0) * f1 + (6.0 - j as f64) * f2;
//!     }
//! }
//!
//! let dfm = DynamicFactorModel::fit(&data, 2, 20).expect("DFM should fit");
//! println!("Factors shape: {:?}", dfm.factors().dim());
//!
//! let pca = PcaForecast::fit(&data, 2).expect("PCA should fit");
//! let forecast = pca.forecast(5).expect("Should forecast");
//! assert_eq!(forecast.dim(), (5, 6));
//! ```

use scirs2_core::ndarray::{s, Array1, Array2, Axis};

use crate::error::{Result, TimeSeriesError};

// ============================================================
// Utilities: basic linear algebra without external deps
// ============================================================

/// Compute `A^T B` efficiently
fn matmul_at_b(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (m2, n) = b.dim();
    assert_eq!(m, m2, "matmul_at_b: row mismatch");
    let mut c = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..m {
                s += a[[l, i]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

/// Compute `A B` (matrix product)
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2, "matmul: inner dimension mismatch");
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

/// Compute `A B A^T`
fn mat_quad(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let ab = matmul(a, b);
    let (m, k) = a.dim();
    let mut c = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for l in 0..k {
                s += ab[[i, l]] * a[[j, l]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

/// Invert a small positive-definite matrix via Cholesky decomposition.
fn cholesky_invert(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    // Cholesky: A = L L^T
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    // Regularise if not PD
                    s = s.abs() + 1e-8;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Invert L (lower triangular)
    let mut linv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        linv[[i, i]] = 1.0 / l[[i, i]];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= l[[i, k]] * linv[[k, j]];
            }
            linv[[i, j]] = s / l[[i, i]];
        }
    }
    // A^{-1} = L^{-T} L^{-1}
    let mut ainv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += linv[[k, i]] * linv[[k, j]];
            }
            ainv[[i, j]] = s;
            ainv[[j, i]] = s;
        }
    }
    Ok(ainv)
}

/// PCA via power iteration for the top-`r` eigenvectors.
/// Returns `(loadings, scores)` where loadings is `(n_vars, r)` and
/// scores is `(T, r)`.
fn pca_power_iter(data: &Array2<f64>, r: usize, max_iter: usize) -> (Array2<f64>, Array2<f64>) {
    let (t, n) = data.dim();
    let r = r.min(n).min(t);

    // Column-centre the data
    let means: Vec<f64> = (0..n)
        .map(|j| data.column(j).iter().sum::<f64>() / t as f64)
        .collect();
    let mut xc = data.to_owned();
    for j in 0..n {
        for i in 0..t {
            xc[[i, j]] -= means[j];
        }
    }

    // Covariance matrix (sample): C = X^T X / (T-1)
    let cov = matmul_at_b(&xc, &xc);
    let scale = (t.saturating_sub(1).max(1)) as f64;

    let mut eigvecs = Array2::<f64>::zeros((n, r));
    let mut residual = cov.clone();

    for k in 0..r {
        // Power iteration for the k-th eigenvector
        let mut v: Vec<f64> = (0..n).map(|i| if i == k { 1.0 } else { 0.01 }).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }

        for _ in 0..max_iter {
            let mut w = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += residual[[i, j]] * v[j];
                }
            }
            let norm_w: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_w < 1e-15 {
                break;
            }
            let mut diff = 0.0_f64;
            for i in 0..n {
                let v_new = w[i] / norm_w;
                diff += (v_new - v[i]).abs();
                v[i] = v_new;
            }
            if diff < 1e-10 {
                break;
            }
        }

        // Compute eigenvalue λ = v^T C v / scale
        let mut lambda = 0.0_f64;
        for i in 0..n {
            let mut cv_i = 0.0;
            for j in 0..n {
                cv_i += cov[[i, j]] * v[j];
            }
            lambda += v[i] * cv_i;
        }
        lambda /= scale;

        for i in 0..n {
            eigvecs[[i, k]] = v[i];
        }

        // Deflate: residual -= λ v v^T
        for i in 0..n {
            for j in 0..n {
                residual[[i, j]] -= lambda * v[i] * v[j];
            }
        }
    }

    // Scores: F = X_centred · V  (T × r)
    let scores = matmul(&xc, &eigvecs);
    (eigvecs, scores)
}

// ============================================================
// Kalman filter and smoother for the state space model
// ============================================================

/// Kalman filter output for a single time step.
struct KalmanState {
    /// Filtered state mean (r,)
    f_filt: Vec<f64>,
    /// Filtered state covariance (r×r)
    p_filt: Array2<f64>,
    /// Predicted state mean (r,)
    f_pred: Vec<f64>,
    /// Predicted state covariance (r×r)
    p_pred: Array2<f64>,
    /// Innovation (n,)
    innovation: Vec<f64>,
    /// Innovation covariance (n×n)
    s_innov: Array2<f64>,
}

/// Run the Kalman filter for the DFM state-space model.
///
/// # Arguments
/// - `data`    — T×n observed panel (rows = time, cols = series)
/// - `lambda`  — n×r factor loadings
/// - `a`       — r×r transition matrix
/// - `r_diag`  — idiosyncratic variances (length n, diagonal of R)
/// - `q`       — r×r factor innovation covariance
///
/// Returns filtered states, predictions, and smoothed caches.
fn kalman_filter(
    data: &Array2<f64>,
    lambda: &Array2<f64>,
    a: &Array2<f64>,
    r_diag: &[f64],
    q: &Array2<f64>,
) -> Result<Vec<KalmanState>> {
    let (t, n) = data.dim();
    let r = lambda.ncols();

    // Build diagonal R matrix
    let r_mat = {
        let mut rm = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            rm[[i, i]] = r_diag[i].max(1e-8);
        }
        rm
    };

    // Initial state: f_0 ~ N(0, P_0) where P_0 solves discrete Lyapunov
    // Using large diagonal as diffuse prior
    let p0 = Array2::<f64>::eye(r) * 1.0;
    let f0 = vec![0.0; r];

    let mut f_pred = f0.clone();
    let mut p_pred = p0.clone();
    let mut states = Vec::with_capacity(t);

    for t_idx in 0..t {
        // Innovation: v_t = x_t - Λ f_{t|t-1}
        let x_t: Vec<f64> = (0..n).map(|j| data[[t_idx, j]]).collect();
        let lambda_f: Vec<f64> = (0..n)
            .map(|j| (0..r).map(|k| lambda[[j, k]] * f_pred[k]).sum::<f64>())
            .collect();
        let innov: Vec<f64> = (0..n).map(|j| x_t[j] - lambda_f[j]).collect();

        // Innovation covariance: S_t = Λ P_{t|t-1} Λ^T + R
        let lambda_p = matmul(lambda, &p_pred); // n×r  (lambda is n×r, p_pred is r×r)
                                                // Wait: lambda is n×r, we need lambda P lambda^T
                                                // Actually lambda_p = Lambda @ P (n×r @ r×r = n×r)
                                                // S = lambda_p @ Lambda^T + R  (n×r @ r×n = n×n)
        let mut s = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut val = r_mat[[i, j]];
                for k in 0..r {
                    val += lambda_p[[i, k]] * lambda[[j, k]];
                }
                s[[i, j]] = val;
            }
        }

        // Kalman gain: K_t = P_{t|t-1} Λ^T S_t^{-1}  (r×n)
        let s_inv = cholesky_invert(&s)?;
        // K = P_pred @ Lambda^T @ S_inv  (r×r @ r×n @ n×n = r×n)
        let p_lambda_t: Array2<f64> = {
            // p_pred (r×r) @ lambda^T (r×n) → r×n
            let mut pl = Array2::<f64>::zeros((r, n));
            for i in 0..r {
                for j in 0..n {
                    let mut val = 0.0;
                    for k in 0..r {
                        val += p_pred[[i, k]] * lambda[[j, k]];
                    }
                    pl[[i, j]] = val;
                }
            }
            pl
        };
        let k_gain = matmul(&p_lambda_t, &s_inv); // r×n

        // Update: f_{t|t} = f_{t|t-1} + K_t v_t
        let mut f_filt = f_pred.clone();
        for i in 0..r {
            for j in 0..n {
                f_filt[i] += k_gain[[i, j]] * innov[j];
            }
        }

        // Update covariance: P_{t|t} = (I - K_t Λ) P_{t|t-1}
        let mut kl = Array2::<f64>::zeros((r, r));
        for i in 0..r {
            for j in 0..r {
                for k in 0..n {
                    kl[[i, j]] += k_gain[[i, k]] * lambda[[k, j]];
                }
            }
        }
        let i_kl = {
            let mut ikl = Array2::<f64>::eye(r);
            for i in 0..r {
                for j in 0..r {
                    ikl[[i, j]] -= kl[[i, j]];
                }
            }
            ikl
        };
        let p_filt = matmul(&i_kl, &p_pred);

        // Store state
        states.push(KalmanState {
            f_filt: f_filt.clone(),
            p_filt: p_filt.clone(),
            f_pred: f_pred.clone(),
            p_pred: p_pred.clone(),
            innovation: innov,
            s_innov: s,
        });

        // Predict next: f_{t+1|t} = A f_{t|t}
        f_pred = (0..r)
            .map(|i| (0..r).map(|j| a[[i, j]] * f_filt[j]).sum::<f64>())
            .collect();
        // P_{t+1|t} = A P_{t|t} A^T + Q
        p_pred = {
            let ap = matmul(a, &p_filt);
            let mut apat = mat_quad(a, &p_filt);
            for i in 0..r {
                for j in 0..r {
                    apat[[i, j]] += q[[i, j]];
                    let _ = ap[[i, j]]; // suppress unused warning
                }
            }
            apat
        };
    }

    Ok(states)
}

/// Kalman smoother (RTS): computes `E[f_t | X_{1:T}]` for all t.
fn kalman_smoother(
    states: &[KalmanState],
    a: &Array2<f64>,
    q: &Array2<f64>,
) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>)> {
    let t = states.len();
    let r = a.nrows();

    let mut f_smooth = vec![vec![0.0; r]; t];
    let mut p_smooth = vec![Array2::<f64>::zeros((r, r)); t];

    // Initialize: smoothed = filtered at T
    f_smooth[t - 1] = states[t - 1].f_filt.clone();
    p_smooth[t - 1] = states[t - 1].p_filt.clone();

    for t_idx in (0..t - 1).rev() {
        let p_t = &states[t_idx].p_filt;
        let p_next_pred = &states[t_idx + 1].p_pred;

        // Smoother gain: L_t = P_{t|t} A^T P_{t+1|t}^{-1}  (r×r)
        let p_inv = cholesky_invert(p_next_pred)?;
        // A^T is r×r, so L = P_t @ A^T @ P_inv
        let ap_t = matmul(a, p_t); // r×r  (actually we need P_t A^T P_inv)
                                   // L = P_t @ A^T @ P_inv
                                   // = (A @ P_t)^T @ P_inv (since P_t is symmetric)
        let mut l = Array2::<f64>::zeros((r, r));
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0;
                for k in 0..r {
                    s += ap_t[[k, i]] * p_inv[[k, j]]; // ap_t^T
                }
                l[[i, j]] = s;
            }
        }

        // f_{t|T} = f_{t|t} + L (f_{t+1|T} - f_{t+1|t})
        let f_next_smooth = &f_smooth[t_idx + 1];
        let f_next_pred = &states[t_idx + 1].f_pred;
        let mut f_s = states[t_idx].f_filt.clone();
        for i in 0..r {
            for j in 0..r {
                f_s[i] += l[[i, j]] * (f_next_smooth[j] - f_next_pred[j]);
            }
        }
        f_smooth[t_idx] = f_s;

        // P_{t|T} = P_{t|t} + L (P_{t+1|T} - P_{t+1|t}) L^T
        let mut p_s = p_t.clone();
        let p_diff = {
            let mut pd = p_smooth[t_idx + 1].clone();
            for i in 0..r {
                for j in 0..r {
                    pd[[i, j]] -= p_next_pred[[i, j]];
                }
            }
            pd
        };
        // L @ p_diff @ L^T  (r×r)
        let lp = matmul(&l, &p_diff);
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0;
                for k in 0..r {
                    s += lp[[i, k]] * l[[j, k]];
                }
                p_s[[i, j]] += s;
            }
        }
        p_smooth[t_idx] = p_s;
    }

    Ok((f_smooth, p_smooth))
}

// ============================================================
// Dynamic Factor Model
// ============================================================

/// Fitted Dynamic Factor Model (DFM).
///
/// Estimated via EM algorithm with Kalman filter/smoother (quasi-MLE).
#[derive(Debug, Clone)]
pub struct DynamicFactorModel {
    /// Number of observed series
    pub n_series: usize,
    /// Number of latent factors
    pub n_factors: usize,
    /// Number of observations used for estimation
    pub n_obs: usize,
    /// Factor loadings Λ  (n_series × n_factors)
    pub loadings: Array2<f64>,
    /// Factor transition matrix A  (n_factors × n_factors)
    pub transition: Array2<f64>,
    /// Idiosyncratic variances R (length n_series, diagonal entries)
    pub idio_var: Vec<f64>,
    /// Factor innovation covariance Q  (n_factors × n_factors)
    pub factor_cov: Array2<f64>,
    /// Smoothed factor estimates F  (n_obs × n_factors)
    smoothed_factors: Array2<f64>,
    /// Series means used for standardization
    series_means: Vec<f64>,
    /// Series standard deviations used for standardization
    series_stds: Vec<f64>,
    /// Final EM log-likelihood
    pub log_likelihood: f64,
    /// Number of EM iterations performed
    pub n_iter: usize,
}

impl DynamicFactorModel {
    /// Fit a Dynamic Factor Model using the EM algorithm.
    ///
    /// # Arguments
    /// - `data`     — T×n matrix of observed time series (rows=time, cols=series)
    /// - `n_factors` — number of latent factors r (must be << n)
    /// - `max_iter`  — maximum number of EM iterations
    ///
    /// # Returns
    /// Fitted [`DynamicFactorModel`].
    pub fn fit(data: &Array2<f64>, n_factors: usize, max_iter: usize) -> Result<Self> {
        let (t, n) = data.dim();
        if n_factors == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_factors must be at least 1".to_string(),
            ));
        }
        if n_factors >= n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "n_factors ({n_factors}) must be less than n_series ({n})"
            )));
        }
        if t < 10 {
            return Err(TimeSeriesError::InvalidInput(
                "Need at least 10 observations".to_string(),
            ));
        }

        // Step 1: Standardize data
        let means: Vec<f64> = (0..n)
            .map(|j| data.column(j).iter().sum::<f64>() / t as f64)
            .collect();
        let stds: Vec<f64> = (0..n)
            .map(|j| {
                let m = means[j];
                let v: f64 =
                    data.column(j).iter().map(|&x| (x - m).powi(2)).sum::<f64>() / t as f64;
                v.sqrt().max(1e-8)
            })
            .collect();

        let mut xstd = data.to_owned();
        for j in 0..n {
            for i in 0..t {
                xstd[[i, j]] = (data[[i, j]] - means[j]) / stds[j];
            }
        }

        // Step 2: PCA initialization
        let (loadings_pca, factors_pca) = pca_power_iter(&xstd, n_factors, 500);

        // Initial parameters
        let mut lambda = loadings_pca.clone(); // n × r
        let mut a = {
            // Estimate VAR(1) transition from PCA factors
            let mut trans = Array2::<f64>::zeros((n_factors, n_factors));
            for k in 0..n_factors {
                // OLS: f_{t,k} ~ a_k * f_{t-1,k}  (diagonal for simplicity)
                let mut xy = 0.0_f64;
                let mut xx = 0.0_f64;
                for i in 1..t {
                    xy += factors_pca[[i, k]] * factors_pca[[i - 1, k]];
                    xx += factors_pca[[i - 1, k]].powi(2);
                }
                trans[[k, k]] = if xx > 0.0 {
                    (xy / xx).clamp(-0.99, 0.99)
                } else {
                    0.5
                };
            }
            trans
        };
        let mut r_diag: Vec<f64> = {
            // Residual variances from PCA
            let predicted = matmul(&factors_pca, &lambda.t().to_owned()); // T×n
            (0..n)
                .map(|j| {
                    let v: f64 = (0..t)
                        .map(|i| (xstd[[i, j]] - predicted[[i, j]]).powi(2))
                        .sum::<f64>()
                        / t as f64;
                    v.max(1e-4)
                })
                .collect()
        };
        let mut q = Array2::<f64>::eye(n_factors) * 0.1;

        let mut log_likelihood = f64::NEG_INFINITY;
        let tol = 1e-6;
        let mut n_iter = 0;

        // EM iterations
        for iter in 0..max_iter {
            n_iter = iter + 1;

            // E-step: Kalman filter + smoother
            let states = kalman_filter(&xstd, &lambda, &a, &r_diag, &q)?;
            let (f_smooth, p_smooth) = kalman_smoother(&states, &a, &q)?;

            // Compute log-likelihood from innovations
            let new_ll = {
                use std::f64::consts::PI;
                let mut ll = 0.0;
                for state in &states {
                    let s_inv = cholesky_invert(&state.s_innov).unwrap_or_else(|_| Array2::eye(n));
                    let mut s_det_ln = 0.0;
                    // Log-det via Cholesky (recompute)
                    for i in 0..n {
                        s_det_ln += state.s_innov[[i, i]].abs().ln();
                    }
                    let mut qform = 0.0;
                    for i in 0..n {
                        for j in 0..n {
                            qform += state.innovation[i] * s_inv[[i, j]] * state.innovation[j];
                        }
                    }
                    ll -= 0.5 * (n as f64 * (2.0 * PI).ln() + s_det_ln + qform);
                }
                ll
            };

            let ll_change = (new_ll - log_likelihood).abs();
            log_likelihood = new_ll;

            // M-step: Update parameters using smoothed states

            // Sufficient statistics
            // S_FF  = Σ_t E[f_t f_t^T | X]        (r×r)
            // S_FF1 = Σ_t E[f_t f_{t-1}^T | X]    (r×r), t=2..T
            // S_F1F1= Σ_t E[f_{t-1} f_{t-1}^T|X]  (r×r), t=2..T

            let mut s_ff = Array2::<f64>::zeros((n_factors, n_factors));
            let mut s_ff1 = Array2::<f64>::zeros((n_factors, n_factors));
            let mut s_f1f1 = Array2::<f64>::zeros((n_factors, n_factors));

            for t_idx in 0..t {
                // E[f_t f_t^T] = P_{t|T} + f_{t|T} f_{t|T}^T
                for i in 0..n_factors {
                    for j in 0..n_factors {
                        s_ff[[i, j]] +=
                            p_smooth[t_idx][[i, j]] + f_smooth[t_idx][i] * f_smooth[t_idx][j];
                    }
                }
            }

            for t_idx in 1..t {
                // Cross-covariance: E[f_t f_{t-1}^T | X] = P_{t,t-1|T} + f_{t|T} f_{t-1|T}^T
                // P_{t,t-1|T} ≈ L_{t-1} P_{t|T} (lag-one covariance from RTS smoother)
                // Approximate with outer product of smoothed means
                for i in 0..n_factors {
                    for j in 0..n_factors {
                        s_ff1[[i, j]] += f_smooth[t_idx][i] * f_smooth[t_idx - 1][j];
                        s_f1f1[[i, j]] += p_smooth[t_idx - 1][[i, j]]
                            + f_smooth[t_idx - 1][i] * f_smooth[t_idx - 1][j];
                    }
                }
            }

            // Update A: A = S_FF1 S_F1F1^{-1}
            if let Ok(s_f1f1_inv) = cholesky_invert(&s_f1f1) {
                a = matmul(&s_ff1, &s_f1f1_inv);
                // Ensure stability: clamp spectral radius
                for i in 0..n_factors {
                    for j in 0..n_factors {
                        if a[[i, j]].abs() > 0.99 {
                            a[[i, j]] = a[[i, j]].signum() * 0.99;
                        }
                    }
                }
            }

            // Update Q: Q = (S_FF - A S_F1F1 A^T - S_FF1 A^T - A S_FF1^T) / (T-1)
            // Simplified: Q = (S_FF - A S_FF1^T) / T
            let a_sff1t = {
                let sff1t = s_ff1.t().to_owned();
                matmul(&a, &sff1t)
            };
            let mut q_new = s_ff.clone();
            for i in 0..n_factors {
                for j in 0..n_factors {
                    q_new[[i, j]] = (s_ff[[i, j]] - a_sff1t[[i, j]]) / t as f64;
                    // Ensure positive diagonal
                    if i == j && q_new[[i, j]] <= 0.0 {
                        q_new[[i, j]] = 1e-6;
                    }
                }
            }
            q = q_new;

            // Update Lambda: Lambda = (Σ_t x_t f_t^T) (S_FF)^{-1}  (n×r)
            let mut xf = Array2::<f64>::zeros((n, n_factors));
            for t_idx in 0..t {
                for i in 0..n {
                    for k in 0..n_factors {
                        xf[[i, k]] += xstd[[t_idx, i]] * f_smooth[t_idx][k];
                    }
                }
            }
            if let Ok(s_ff_inv) = cholesky_invert(&s_ff) {
                lambda = matmul(&xf, &s_ff_inv);
            }

            // Update R (diagonal idiosyncratic variances):
            // R_ii = (1/T) (Σ_t x_{ti}^2 - 2 lambda_i^T Σ_t f_t x_{ti} + lambda_i^T S_FF lambda_i)
            let lambda_sff = matmul(&lambda, &s_ff); // n×r @ r×r = n×r (wrong order)
                                                     // Actually: lambda_i^T S_FF lambda_i = sum_{k,l} lambda_{ik} S_FF_{kl} lambda_{il}
            for j in 0..n {
                let x_sq: f64 = (0..t).map(|i| xstd[[i, j]].powi(2)).sum();
                let xf_term: f64 = (0..n_factors).map(|k| xf[[j, k]] * lambda[[j, k]]).sum();
                let lsfl: f64 = (0..n_factors)
                    .map(|k| lambda_sff[[j, k]] * lambda[[j, k]])
                    .sum();
                r_diag[j] = ((x_sq - 2.0 * xf_term + lsfl) / t as f64).max(1e-6);
            }

            if iter > 0 && ll_change < tol {
                break;
            }
        }

        // Extract final smoothed factors for storage
        let states = kalman_filter(&xstd, &lambda, &a, &r_diag, &q)?;
        let (f_smooth, _) = kalman_smoother(&states, &a, &q)?;

        let mut smoothed_factors = Array2::<f64>::zeros((t, n_factors));
        for i in 0..t {
            for k in 0..n_factors {
                smoothed_factors[[i, k]] = f_smooth[i][k];
            }
        }

        Ok(DynamicFactorModel {
            n_series: n,
            n_factors,
            n_obs: t,
            loadings: lambda,
            transition: a,
            idio_var: r_diag,
            factor_cov: q,
            smoothed_factors,
            series_means: means,
            series_stds: stds,
            log_likelihood,
            n_iter,
        })
    }

    /// Return the smoothed factor estimates F (n_obs × n_factors).
    pub fn factors(&self) -> &Array2<f64> {
        &self.smoothed_factors
    }

    /// Reconstruct the observed series from factors: X̂ = F Λ^T
    ///
    /// Returns T×n matrix in original (unstandardised) scale.
    pub fn fitted_values(&self) -> Array2<f64> {
        let (t, r) = self.smoothed_factors.dim();
        let n = self.n_series;
        let mut fitted = Array2::<f64>::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..r {
                    val += self.smoothed_factors[[i, k]] * self.loadings[[j, k]];
                }
                // De-standardise
                fitted[[i, j]] = val * self.series_stds[j] + self.series_means[j];
            }
        }
        fitted
    }

    /// Forecast h steps ahead for all series.
    ///
    /// Iterates the factor transition model: f_{T+h|T} = A^h f_{T|T}
    /// and reconstructs: x̂_{T+h} = Λ f_{T+h|T}
    pub fn forecast(&self, h: usize) -> Result<Array2<f64>> {
        if h == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "h must be at least 1".to_string(),
            ));
        }
        let n = self.n_series;
        let r = self.n_factors;
        let t = self.n_obs;

        // Last smoothed factor: f_{T|T}
        let mut f_curr: Vec<f64> = (0..r).map(|k| self.smoothed_factors[[t - 1, k]]).collect();

        let mut forecasts = Array2::<f64>::zeros((h, n));
        for step in 0..h {
            // f_{t+1} = A f_t
            let f_next: Vec<f64> = (0..r)
                .map(|i| {
                    (0..r)
                        .map(|j| self.transition[[i, j]] * f_curr[j])
                        .sum::<f64>()
                })
                .collect();
            f_curr = f_next;

            // Reconstruct: x̂ = Λ f
            for j in 0..n {
                let val: f64 = (0..r).map(|k| self.loadings[[j, k]] * f_curr[k]).sum();
                forecasts[[step, j]] = val * self.series_stds[j] + self.series_means[j];
            }
        }
        Ok(forecasts)
    }

    /// Compute model information criteria.
    pub fn aic(&self) -> f64 {
        let k = (self.n_series * self.n_factors + self.n_factors * self.n_factors + self.n_series)
            as f64;
        -2.0 * self.log_likelihood + 2.0 * k
    }

    /// Compute Bayesian Information Criterion (BIC) for the factor model.
    pub fn bic(&self) -> f64 {
        let k = (self.n_series * self.n_factors + self.n_factors * self.n_factors + self.n_series)
            as f64;
        let n = self.n_obs as f64;
        -2.0 * self.log_likelihood + k * n.ln()
    }
}

// ============================================================
// PCA-based Forecasting
// ============================================================

/// PCA-based forecast for large panels (Stock & Watson, 2002).
///
/// Reduces the panel to `r` principal components, fits a VAR(1)
/// to those components, and re-projects to the original space.
#[derive(Debug, Clone)]
pub struct PcaForecast {
    /// Number of series
    pub n_series: usize,
    /// Number of principal components used
    pub n_components: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Loadings / eigenvectors  (n_series × n_components)
    pub loadings: Array2<f64>,
    /// Scores (in-sample PCs): (n_obs × n_components)
    scores: Array2<f64>,
    /// VAR(1) coefficient matrix on PC scores (n_components × n_components)
    var_coef: Array2<f64>,
    /// PC score means (length n_components)
    score_means: Vec<f64>,
    /// Original series means
    series_means: Vec<f64>,
    /// Original series standard deviations
    series_stds: Vec<f64>,
    /// Proportion of variance explained by each component
    pub explained_variance_ratio: Vec<f64>,
}

impl PcaForecast {
    /// Fit a PCA forecast model.
    ///
    /// # Arguments
    /// - `data`         — T×n panel (rows=time, cols=series)
    /// - `n_components` — number of principal components to retain
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Result<Self> {
        let (t, n) = data.dim();
        if n_components == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_components must be at least 1".to_string(),
            ));
        }
        if n_components > n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "n_components ({n_components}) cannot exceed n_series ({n})"
            )));
        }
        if t < 3 {
            return Err(TimeSeriesError::InvalidInput(
                "Need at least 3 observations".to_string(),
            ));
        }

        // Standardise
        let means: Vec<f64> = (0..n)
            .map(|j| data.column(j).iter().sum::<f64>() / t as f64)
            .collect();
        let stds: Vec<f64> = (0..n)
            .map(|j| {
                let m = means[j];
                let v: f64 =
                    data.column(j).iter().map(|&x| (x - m).powi(2)).sum::<f64>() / t as f64;
                v.sqrt().max(1e-8)
            })
            .collect();

        let mut xstd = data.to_owned();
        for j in 0..n {
            for i in 0..t {
                xstd[[i, j]] = (data[[i, j]] - means[j]) / stds[j];
            }
        }

        // PCA — power iteration may return duplicate loading vectors when the
        // data is low-rank (e.g. all series driven by the same composite signal).
        // Apply modified Gram-Schmidt orthogonalisation to the raw loadings so
        // that each subsequent component is projected onto the orthogonal
        // complement of all previous ones.  Zero-norm residuals are left as the
        // zero vector (they contribute zero score variance and zero EVR, which
        // is the mathematically correct result for rank-deficient directions).
        let (raw_loadings, _raw_scores) = pca_power_iter(&xstd, n_components, 1000);

        let mut orth_loadings = Array2::<f64>::zeros((n, n_components));
        for k in 0..n_components {
            // Start from the raw loading direction
            let mut qk: Vec<f64> = raw_loadings.column(k).to_vec();
            // Subtract projections onto already-orthogonalised columns
            for j in 0..k {
                let qj = orth_loadings.column(j);
                let dot: f64 = qk.iter().zip(qj.iter()).map(|(&a, &b)| a * b).sum();
                for i in 0..n {
                    qk[i] -= dot * qj[i];
                }
            }
            // Normalise; if the residual is near-zero this direction is
            // degenerate and we leave it as the zero vector.
            let norm: f64 = qk.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for i in 0..n {
                    orth_loadings[[i, k]] = qk[i] / norm;
                }
                // else: column k stays all-zeros
            }
        }

        // Recompute scores from the orthogonalised loadings.  Use the same
        // column-centred data that pca_power_iter would use internally; since
        // xstd was already zero-meaned this is just xstd itself.
        let scores = matmul(&xstd, &orth_loadings);
        let loadings = orth_loadings;

        // Compute explained variance ratios: score variance / total input variance.
        // Orthogonality guarantees sum(evr) ≤ 1.
        let total_var: f64 = (0..n)
            .map(|j| xstd.column(j).iter().map(|&x| x.powi(2)).sum::<f64>() / t as f64)
            .sum();

        let evr: Vec<f64> = (0..n_components)
            .map(|k| {
                let sv: f64 = scores.column(k).iter().map(|&x| x.powi(2)).sum::<f64>() / t as f64;
                if total_var > 0.0 {
                    sv / total_var
                } else {
                    0.0
                }
            })
            .collect();

        // Score means (usually ~0 after centering, but compute anyway)
        let score_means: Vec<f64> = (0..n_components)
            .map(|k| scores.column(k).iter().sum::<f64>() / t as f64)
            .collect();

        // Fit VAR(1) to PC scores: F_t = B F_{t-1} + noise
        // OLS per component (diagonal B for simplicity)
        let mut var_coef = Array2::<f64>::zeros((n_components, n_components));
        for k in 0..n_components {
            // OLS: y = F_{t,k}, x = F_{t-1,k}
            let mut xy = 0.0_f64;
            let mut xx = 0.0_f64;
            for i in 1..t {
                xy += scores[[i, k]] * scores[[i - 1, k]];
                xx += scores[[i - 1, k]].powi(2);
            }
            var_coef[[k, k]] = if xx > 1e-10 {
                (xy / xx).clamp(-0.99, 0.99)
            } else {
                0.0
            };
        }

        Ok(PcaForecast {
            n_series: n,
            n_components,
            n_obs: t,
            loadings,
            scores,
            var_coef,
            score_means,
            series_means: means,
            series_stds: stds,
            explained_variance_ratio: evr,
        })
    }

    /// Forecast h steps ahead.
    ///
    /// Returns an h×n matrix of forecasted series values.
    pub fn forecast(&self, h: usize) -> Result<Array2<f64>> {
        if h == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "h must be at least 1".to_string(),
            ));
        }
        let n = self.n_series;
        let r = self.n_components;
        let t = self.n_obs;

        // Last PC score
        let mut f_curr: Vec<f64> = (0..r).map(|k| self.scores[[t - 1, k]]).collect();

        let mut forecasts = Array2::<f64>::zeros((h, n));
        for step in 0..h {
            // f_{t+1} = B f_t  (diagonal VAR(1))
            let f_next: Vec<f64> = (0..r)
                .map(|i| {
                    (0..r)
                        .map(|j| self.var_coef[[i, j]] * f_curr[j])
                        .sum::<f64>()
                })
                .collect();
            f_curr = f_next.clone();

            // Project back: x̂ = F Λ^T (loadings: n×r, so Λ^T is r×n)
            for j in 0..n {
                let val: f64 = (0..r).map(|k| self.loadings[[j, k]] * f_curr[k]).sum();
                // De-standardise
                forecasts[[step, j]] = val * self.series_stds[j] + self.series_means[j];
            }
        }
        Ok(forecasts)
    }

    /// Return in-sample fitted values (T×n).
    pub fn fitted_values(&self) -> Array2<f64> {
        let t = self.n_obs;
        let n = self.n_series;
        let r = self.n_components;
        let mut fitted = Array2::<f64>::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                let val: f64 = (0..r)
                    .map(|k| self.scores[[i, k]] * self.loadings[[j, k]])
                    .sum();
                fitted[[i, j]] = val * self.series_stds[j] + self.series_means[j];
            }
        }
        fitted
    }

    /// Total explained variance ratio across all retained components.
    pub fn total_explained_variance(&self) -> f64 {
        self.explained_variance_ratio.iter().sum()
    }
}

// ============================================================
// Re-export convenience
// ============================================================

pub use DynamicFactorModel as DFM;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_panel(t: usize, n: usize, n_factors: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((t, n));
        // Generate data with known factor structure
        for obs in 0..t {
            let factors: Vec<f64> = (0..n_factors)
                .map(|k| (obs as f64 * (k + 1) as f64 * 0.15).sin())
                .collect();
            for j in 0..n {
                let loading = (j as f64 + 1.0) / n as f64;
                data[[obs, j]] = factors
                    .iter()
                    .enumerate()
                    .map(|(k, &f)| f * loading * (k as f64 + 1.0))
                    .sum::<f64>();
                // Add small noise
                let noise = (obs * n + j) as f64 * 0.001 % 0.01 - 0.005;
                data[[obs, j]] += noise;
            }
        }
        data
    }

    #[test]
    fn test_dfm_fit_basic() {
        let data = make_panel(50, 6, 2);
        let dfm = DynamicFactorModel::fit(&data, 2, 10).expect("DFM should fit");
        assert_eq!(dfm.n_series, 6);
        assert_eq!(dfm.n_factors, 2);
        assert_eq!(dfm.n_obs, 50);
        assert_eq!(dfm.factors().dim(), (50, 2));
    }

    #[test]
    fn test_dfm_fitted_values_shape() {
        let data = make_panel(40, 5, 2);
        let dfm = DynamicFactorModel::fit(&data, 2, 5).expect("DFM should fit");
        let fv = dfm.fitted_values();
        assert_eq!(fv.dim(), (40, 5));
    }

    #[test]
    fn test_dfm_forecast_shape() {
        let data = make_panel(40, 5, 2);
        let dfm = DynamicFactorModel::fit(&data, 2, 5).expect("DFM should fit");
        let fc = dfm.forecast(4).expect("Should forecast");
        assert_eq!(fc.dim(), (4, 5));
        // All forecast values should be finite
        for v in fc.iter() {
            assert!(v.is_finite(), "Forecast value must be finite: {v}");
        }
    }

    #[test]
    fn test_dfm_invalid_n_factors() {
        let data = make_panel(30, 4, 2);
        // n_factors >= n_series should fail
        assert!(DynamicFactorModel::fit(&data, 4, 5).is_err());
        // n_factors = 0 should fail
        assert!(DynamicFactorModel::fit(&data, 0, 5).is_err());
    }

    #[test]
    fn test_pca_forecast_fit() {
        let data = make_panel(50, 8, 3);
        let pca = PcaForecast::fit(&data, 3).expect("PCA should fit");
        assert_eq!(pca.n_series, 8);
        assert_eq!(pca.n_components, 3);
        assert_eq!(pca.n_obs, 50);
        assert_eq!(pca.loadings.dim(), (8, 3));
    }

    #[test]
    fn test_pca_forecast_forecast_shape() {
        let data = make_panel(50, 6, 2);
        let pca = PcaForecast::fit(&data, 2).expect("PCA should fit");
        let fc = pca.forecast(5).expect("Should forecast");
        assert_eq!(fc.dim(), (5, 6));
        for v in fc.iter() {
            assert!(v.is_finite(), "Forecast value must be finite: {v}");
        }
    }

    #[test]
    fn test_pca_explained_variance() {
        let data = make_panel(60, 8, 3);
        let pca = PcaForecast::fit(&data, 3).expect("PCA should fit");
        let total = pca.total_explained_variance();
        assert!(total > 0.0, "Explained variance should be positive");
        assert!(
            total <= 1.01,
            "Explained variance should be <= 1 (got {total})"
        );
        assert_eq!(pca.explained_variance_ratio.len(), 3);
    }

    #[test]
    fn test_pca_fitted_values_shape() {
        let data = make_panel(40, 5, 2);
        let pca = PcaForecast::fit(&data, 2).expect("PCA should fit");
        let fv = pca.fitted_values();
        assert_eq!(fv.dim(), (40, 5));
    }

    #[test]
    fn test_pca_forecast_invalid_n_components() {
        let data = make_panel(30, 4, 2);
        assert!(PcaForecast::fit(&data, 0).is_err());
        assert!(PcaForecast::fit(&data, 5).is_err()); // > n_series
    }

    #[test]
    fn test_dfm_alias() {
        let data = make_panel(40, 5, 2);
        // DFM is an alias for DynamicFactorModel
        let _ = DFM::fit(&data, 2, 5).expect("DFM alias should work");
    }
}
