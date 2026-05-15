//! Markov Regime Switching Models
//!
//! Implements Markov-switching autoregressive (MS-AR) models for time series
//! that exhibit structural breaks or regime changes. Key components:
//!
//! - **MS-AR model**: Autoregressive model with regime-dependent parameters
//! - **Hamilton filter**: Forward recursion for filtered regime probabilities
//! - **EM algorithm**: Maximum-likelihood parameter estimation
//! - **Smoothed probabilities**: Kim smoother for full-sample regime inference
//! - **Multi-regime**: Supports 2 or more regimes
//! - **Transition matrix**: Full state transition probability estimation
//!
//! # References
//!
//! - Hamilton, J.D. (1989) "A New Approach to the Economic Analysis of
//!   Nonstationary Time Series and the Business Cycle"
//! - Kim, C.-J. (1994) "Dynamic Linear Models with Markov-Switching"
//! - Hamilton, J.D. (1994) "Time Series Analysis", Ch. 22

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a Markov-switching autoregressive model
#[derive(Debug, Clone)]
pub struct MSARConfig {
    /// Number of regimes (>= 2)
    pub n_regimes: usize,
    /// AR order (number of lags)
    pub ar_order: usize,
    /// Maximum EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood
    pub tolerance: f64,
    /// Whether to allow regime-dependent AR coefficients
    pub switching_ar: bool,
    /// Whether to allow regime-dependent variance
    pub switching_variance: bool,
}

impl Default for MSARConfig {
    fn default() -> Self {
        Self {
            n_regimes: 2,
            ar_order: 1,
            max_iter: 200,
            tolerance: 1e-8,
            switching_ar: true,
            switching_variance: true,
        }
    }
}

/// Parameters of a fitted MS-AR model
#[derive(Debug, Clone)]
pub struct MSARParams<F: Float> {
    /// Regime-specific intercepts, length n_regimes
    pub intercepts: Array1<F>,
    /// Regime-specific AR coefficients, shape (n_regimes, ar_order)
    pub ar_coefficients: Array2<F>,
    /// Regime-specific variances (error variance), length n_regimes
    pub variances: Array1<F>,
    /// Transition probability matrix, shape (n_regimes, n_regimes)
    /// transition_probs\[i\]\[j\] = P(S_t = j | S_{t-1} = i)
    pub transition_probs: Array2<F>,
    /// Ergodic (unconditional) regime probabilities
    pub ergodic_probs: Array1<F>,
}

/// Results from fitting a MS-AR model
#[derive(Debug, Clone)]
pub struct MSARResult<F: Float> {
    /// Estimated parameters
    pub params: MSARParams<F>,
    /// Filtered regime probabilities, shape (T, n_regimes)
    pub filtered_probs: Array2<F>,
    /// Smoothed regime probabilities, shape (T, n_regimes)
    pub smoothed_probs: Array2<F>,
    /// Log-likelihood at convergence
    pub log_likelihood: F,
    /// Number of EM iterations performed
    pub n_iterations: usize,
    /// Whether the EM algorithm converged
    pub converged: bool,
    /// Most likely regime sequence (Viterbi-like from smoothed probs)
    pub regime_sequence: Array1<usize>,
}

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/// Fit a Markov-switching autoregressive model via the EM algorithm
///
/// # Arguments
/// * `data` - Time series observations
/// * `config` - Model configuration
///
/// # Returns
/// MSARResult with estimated parameters, filtered/smoothed probabilities,
/// and regime classification
pub fn fit_msar<F>(data: &Array1<F>, config: &MSARConfig) -> Result<MSARResult<F>>
where
    F: Float + FromPrimitive + Debug + Display + std::iter::Sum,
{
    validate_config(config)?;

    let n = data.len();
    let p = config.ar_order;
    let k = config.n_regimes;

    if n <= p + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "for MS-AR model".to_string(),
            required: p + 2,
            actual: n,
        });
    }

    let effective_n = n - p; // Number of usable observations

    // Initialize parameters
    let mut params = initialize_params(data, config)?;
    let mut prev_ll = F::neg_infinity();
    let mut converged = false;
    let mut n_iter = 0;

    // EM iterations
    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // E-step: Hamilton filter + Kim smoother
        let (filtered, predicted, log_ll) = hamilton_filter(data, &params, config)?;
        let smoothed = kim_smoother(&filtered, &predicted, &params, config)?;

        // Check convergence
        let ll_change = (log_ll - prev_ll).abs();
        let ll_threshold = F::from_f64(config.tolerance).ok_or_else(|| {
            TimeSeriesError::ComputationError("Failed to convert tolerance".to_string())
        })?;

        if iter > 0 && ll_change < ll_threshold {
            converged = true;
            // Store final probabilities
            let regime_seq = classify_regimes(&smoothed);
            return Ok(MSARResult {
                params,
                filtered_probs: filtered,
                smoothed_probs: smoothed,
                log_likelihood: log_ll,
                n_iterations: n_iter,
                converged,
                regime_sequence: regime_seq,
            });
        }

        prev_ll = log_ll;

        // M-step: update parameters
        params = m_step(data, &smoothed, config)?;
    }

    // Final E-step for probabilities
    let (filtered, _predicted, log_ll) = hamilton_filter(data, &params, config)?;
    let smoothed = kim_smoother(
        &filtered,
        &{
            let (_, p, _) = hamilton_filter(data, &params, config)?;
            p
        },
        &params,
        config,
    )?;
    let regime_seq = classify_regimes(&smoothed);

    Ok(MSARResult {
        params,
        filtered_probs: filtered,
        smoothed_probs: smoothed,
        log_likelihood: log_ll,
        n_iterations: n_iter,
        converged,
        regime_sequence: regime_seq,
    })
}

/// Run the Hamilton filter on data with given parameters
///
/// Computes P(S_t = j | Y_1, ..., Y_t) for each t and regime j.
///
/// # Arguments
/// * `data` - Time series observations
/// * `params` - Model parameters
/// * `config` - Model configuration
///
/// # Returns
/// (filtered_probs, predicted_probs, log_likelihood)
pub fn hamilton_filter<F>(
    data: &Array1<F>,
    params: &MSARParams<F>,
    config: &MSARConfig,
) -> Result<(Array2<F>, Array2<F>, F)>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let n = data.len();
    let p = config.ar_order;
    let k = config.n_regimes;
    let t_eff = n - p;

    let mut filtered = Array2::zeros((t_eff, k));
    let mut predicted = Array2::zeros((t_eff, k));
    let mut log_likelihood = F::zero();

    // Initialize with ergodic probabilities
    let mut prev_filtered = params.ergodic_probs.clone();

    let two_pi = F::from_f64(2.0 * std::f64::consts::PI)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert 2*pi".to_string()))?;

    for t in 0..t_eff {
        let obs_idx = t + p;

        // Prediction step: P(S_t = j | Y_{t-1})
        let mut pred_probs = Array1::zeros(k);
        for j in 0..k {
            let mut sum = F::zero();
            for i in 0..k {
                sum = sum + params.transition_probs[[i, j]] * prev_filtered[i];
            }
            pred_probs[j] = sum;
        }

        // Compute conditional densities f(y_t | S_t = j, Y_{t-1})
        let mut densities = Array1::zeros(k);
        for j in 0..k {
            let mut mean = params.intercepts[j];
            for lag in 0..p {
                let coeff = if config.switching_ar {
                    params.ar_coefficients[[j, lag]]
                } else {
                    params.ar_coefficients[[0, lag]]
                };
                mean = mean + coeff * data[obs_idx - 1 - lag];
            }

            let var = if config.switching_variance {
                params.variances[j]
            } else {
                params.variances[0]
            };

            // Guard against degenerate variance
            let safe_var = if var < F::epsilon() {
                F::from_f64(1e-10).unwrap_or(F::epsilon())
            } else {
                var
            };

            let residual = data[obs_idx] - mean;
            let exponent = -(residual * residual) / (safe_var + safe_var);
            densities[j] = (two_pi * safe_var).sqrt().recip() * exponent.exp();

            // Clamp density to avoid numerical issues
            if densities[j] < F::epsilon() {
                densities[j] = F::from_f64(1e-300).unwrap_or(F::epsilon());
            }
        }

        // Joint probability: P(S_t = j | Y_{t-1}) * f(y_t | S_t = j, Y_{t-1})
        let mut joint = Array1::zeros(k);
        let mut marginal = F::zero();
        for j in 0..k {
            joint[j] = pred_probs[j] * densities[j];
            marginal = marginal + joint[j];
        }

        // Avoid log(0)
        if marginal < F::epsilon() {
            marginal = F::from_f64(1e-300).unwrap_or(F::epsilon());
        }

        log_likelihood = log_likelihood + marginal.ln();

        // Filter step: P(S_t = j | Y_t) = joint / marginal
        let mut filt = Array1::zeros(k);
        for j in 0..k {
            filt[j] = joint[j] / marginal;
        }

        // Store
        for j in 0..k {
            filtered[[t, j]] = filt[j];
            predicted[[t, j]] = pred_probs[j];
        }

        prev_filtered = filt;
    }

    Ok((filtered, predicted, log_likelihood))
}

/// Compute smoothed regime probabilities using Kim's algorithm
///
/// P(S_t = j | Y_1, ..., Y_T) for all t.
///
/// # Arguments
/// * `filtered` - Filtered probabilities from Hamilton filter
/// * `predicted` - Predicted probabilities from Hamilton filter
/// * `params` - Model parameters
/// * `config` - Model configuration
///
/// # Returns
/// Smoothed probabilities, shape (T_eff, n_regimes)
pub fn kim_smoother<F>(
    filtered: &Array2<F>,
    predicted: &Array2<F>,
    params: &MSARParams<F>,
    config: &MSARConfig,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let (t_eff, k) = filtered.dim();
    let mut smoothed = Array2::zeros((t_eff, k));

    // Last period: smoothed = filtered
    for j in 0..k {
        smoothed[[t_eff - 1, j]] = filtered[[t_eff - 1, j]];
    }

    // Backward recursion
    for t in (0..t_eff - 1).rev() {
        for i in 0..k {
            let mut sum = F::zero();
            for j in 0..k {
                let pred_j = predicted[[t + 1, j]];
                let safe_pred = if pred_j < F::epsilon() {
                    F::from_f64(1e-300).unwrap_or(F::epsilon())
                } else {
                    pred_j
                };

                sum = sum + params.transition_probs[[i, j]] * smoothed[[t + 1, j]] / safe_pred;
            }
            smoothed[[t, i]] = filtered[[t, i]] * sum;
        }

        // Normalize to sum to 1
        let row_sum: F = (0..k)
            .map(|j| smoothed[[t, j]])
            .fold(F::zero(), |a, x| a + x);
        if row_sum > F::epsilon() {
            for j in 0..k {
                smoothed[[t, j]] = smoothed[[t, j]] / row_sum;
            }
        }
    }

    Ok(smoothed)
}

/// Compute the transition probability matrix
///
/// Returns the probability P(S_t = j | S_{t-1} = i) for all regime pairs.
pub fn transition_matrix<F>(params: &MSARParams<F>) -> &Array2<F>
where
    F: Float,
{
    &params.transition_probs
}

/// Compute ergodic (stationary) distribution of regime probabilities
///
/// Solves pi * P = pi with sum(pi) = 1, where P is the transition matrix.
pub fn compute_ergodic_probs<F>(transition_probs: &Array2<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let k = transition_probs.nrows();
    if transition_probs.ncols() != k {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: transition_probs.ncols(),
        });
    }

    // Solve (P' - I)pi = 0, with constraint sum(pi) = 1
    // Replace last equation with sum constraint
    let mut a = Array2::<F>::zeros((k, k));
    let mut b = Array1::<F>::zeros(k);

    for i in 0..k {
        for j in 0..k {
            if i < k - 1 {
                // (P' - I)_{ij} = P_{ji} - delta_{ij}
                a[[i, j]] = transition_probs[[j, i]] - if i == j { F::one() } else { F::zero() };
            }
        }
    }

    // Last row: sum constraint
    for j in 0..k {
        a[[k - 1, j]] = F::one();
    }
    b[k - 1] = F::one();

    // Solve via Gaussian elimination
    solve_linear_system_internal(&a, &b)
}

/// Forecast from a fitted MS-AR model
///
/// Generates h-step-ahead forecasts using regime-probability-weighted predictions.
///
/// # Arguments
/// * `data` - Original time series
/// * `result` - Fitted model result
/// * `config` - Model configuration
/// * `horizon` - Number of steps to forecast
///
/// # Returns
/// Point forecasts of length horizon
pub fn forecast_msar<F>(
    data: &Array1<F>,
    result: &MSARResult<F>,
    config: &MSARConfig,
    horizon: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    if horizon == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "horizon".to_string(),
            message: "Forecast horizon must be positive".to_string(),
        });
    }

    let p = config.ar_order;
    let k = config.n_regimes;
    let n = data.len();

    if n < p {
        return Err(TimeSeriesError::InsufficientData {
            message: "for MS-AR forecast".to_string(),
            required: p,
            actual: n,
        });
    }

    let params = &result.params;

    // Last filtered probabilities as initial state
    let t_eff = result.filtered_probs.nrows();
    let mut regime_probs = Array1::zeros(k);
    for j in 0..k {
        regime_probs[j] = result.filtered_probs[[t_eff - 1, j]];
    }

    // Extended data: original + forecast values
    let mut extended: Vec<F> = data.iter().copied().collect();
    let mut forecasts = Array1::zeros(horizon);

    for h in 0..horizon {
        // Predict regime probs for next step
        let mut next_probs = Array1::zeros(k);
        for j in 0..k {
            let mut sum = F::zero();
            for i in 0..k {
                sum = sum + params.transition_probs[[i, j]] * regime_probs[i];
            }
            next_probs[j] = sum;
        }

        // Regime-weighted forecast
        let mut forecast_val = F::zero();
        let idx = n + h;

        for j in 0..k {
            let mut mean = params.intercepts[j];
            for lag in 0..p {
                let coeff = if config.switching_ar {
                    params.ar_coefficients[[j, lag]]
                } else {
                    params.ar_coefficients[[0, lag]]
                };
                if idx >= 1 + lag && (idx - 1 - lag) < extended.len() {
                    mean = mean + coeff * extended[idx - 1 - lag];
                }
            }
            forecast_val = forecast_val + next_probs[j] * mean;
        }

        forecasts[h] = forecast_val;
        extended.push(forecast_val);
        regime_probs = next_probs;
    }

    Ok(forecasts)
}

// ---------------------------------------------------------------------------
// Internal: Parameter initialization
// ---------------------------------------------------------------------------

fn initialize_params<F>(data: &Array1<F>, config: &MSARConfig) -> Result<MSARParams<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n = data.len();
    let k = config.n_regimes;
    let p = config.ar_order;

    // Compute data mean and variance
    let n_f = F::from_usize(n)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;
    let data_mean: F = data.iter().copied().sum::<F>() / n_f;
    let data_var: F = data
        .iter()
        .map(|&x| {
            let d = x - data_mean;
            d * d
        })
        .fold(F::zero(), |a, x| a + x)
        / n_f;

    // Initialize intercepts: spread across data range
    let mut intercepts = Array1::zeros(k);
    let k_f = F::from_usize(k)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert k".to_string()))?;

    // Find data min/max
    let data_min = data.iter().copied().fold(F::infinity(), |a, b| a.min(b));
    let data_max = data
        .iter()
        .copied()
        .fold(F::neg_infinity(), |a, b| a.max(b));
    let range = data_max - data_min;

    for j in 0..k {
        let j_f = F::from_usize(j)
            .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert j".to_string()))?;
        // Spread intercepts across data range
        intercepts[j] = data_min + range * (j_f + F::from_f64(0.5).unwrap_or(F::zero())) / k_f;
    }

    // Initialize AR coefficients: small values
    let mut ar_coefficients = Array2::zeros((k, p));
    if p > 0 {
        // Simple AR(1)-like initialization
        let ar1_init = F::from_f64(0.5).unwrap_or(F::zero());
        for j in 0..k {
            ar_coefficients[[j, 0]] = ar1_init;
            for lag in 1..p {
                ar_coefficients[[j, lag]] = F::zero();
            }
        }
    }

    // Initialize variances
    let mut variances = Array1::zeros(k);
    for j in 0..k {
        let j_f = F::from_usize(j)
            .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert j".to_string()))?;
        // Different variance levels per regime
        let scale = F::one() + j_f * F::from_f64(0.5).unwrap_or(F::zero());
        variances[j] = data_var * scale / k_f;
        // Ensure positive
        if variances[j] < F::epsilon() {
            variances[j] = F::from_f64(0.01).unwrap_or(F::epsilon());
        }
    }

    // Initialize transition matrix: slightly persistent
    let mut transition_probs = Array2::zeros((k, k));
    let persistence = F::from_f64(0.9).unwrap_or(F::one());
    let off_diag = (F::one() - persistence)
        / F::from_usize(k - 1).ok_or_else(|| {
            TimeSeriesError::ComputationError("Failed to convert k-1".to_string())
        })?;

    for i in 0..k {
        for j in 0..k {
            transition_probs[[i, j]] = if i == j { persistence } else { off_diag };
        }
    }

    // Ergodic probs
    let ergodic_probs = compute_ergodic_probs(&transition_probs)?;

    Ok(MSARParams {
        intercepts,
        ar_coefficients,
        variances,
        transition_probs,
        ergodic_probs,
    })
}

// ---------------------------------------------------------------------------
// Internal: M-step
// ---------------------------------------------------------------------------

fn m_step<F>(data: &Array1<F>, smoothed: &Array2<F>, config: &MSARConfig) -> Result<MSARParams<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let n = data.len();
    let p = config.ar_order;
    let k = config.n_regimes;
    let t_eff = n - p;

    // Update intercepts and AR coefficients via weighted least squares
    let mut intercepts = Array1::zeros(k);
    let mut ar_coefficients = Array2::zeros((k, p));
    let mut variances = Array1::zeros(k);

    for j in 0..k {
        // Weighted regression for regime j
        let mut sum_w = F::zero();
        let mut sum_wy = F::zero();
        let mut sum_wxx = vec![vec![F::zero(); p + 1]; p + 1]; // (p+1) x (p+1)
        let mut sum_wxy = vec![F::zero(); p + 1];

        for t in 0..t_eff {
            let obs_idx = t + p;
            let w = smoothed[[t, j]];
            sum_w = sum_w + w;

            let y = data[obs_idx];
            sum_wy = sum_wy + w * y;

            // Build regressor: [1, y_{t-1}, ..., y_{t-p}]
            let mut x = vec![F::one()];
            for lag in 0..p {
                x.push(data[obs_idx - 1 - lag]);
            }

            for a in 0..=p {
                sum_wxy[a] = sum_wxy[a] + w * x[a] * y;
                for b in 0..=p {
                    sum_wxx[a][b] = sum_wxx[a][b] + w * x[a] * x[b];
                }
            }
        }

        // Solve weighted normal equations
        let dim = p + 1;
        let mut a_mat = Array2::<F>::zeros((dim, dim));
        let mut b_vec = Array1::<F>::zeros(dim);

        for a in 0..dim {
            b_vec[a] = sum_wxy[a];
            for b in 0..dim {
                a_mat[[a, b]] = sum_wxx[a][b];
            }
        }

        // Add small regularization for numerical stability
        let reg = F::from_f64(1e-8).unwrap_or(F::epsilon());
        for a in 0..dim {
            a_mat[[a, a]] = a_mat[[a, a]] + reg;
        }

        let beta = solve_linear_system_internal(&a_mat, &b_vec).unwrap_or_else(|_| {
            let mut b = Array1::zeros(dim);
            if sum_w > F::epsilon() {
                b[0] = sum_wy / sum_w;
            }
            b
        });

        intercepts[j] = beta[0];
        for lag in 0..p {
            ar_coefficients[[j, lag]] = beta[1 + lag];
        }

        // Update variance
        let mut sum_w_resid_sq = F::zero();
        for t in 0..t_eff {
            let obs_idx = t + p;
            let w = smoothed[[t, j]];
            let mut fitted = intercepts[j];
            for lag in 0..p {
                fitted = fitted + ar_coefficients[[j, lag]] * data[obs_idx - 1 - lag];
            }
            let resid = data[obs_idx] - fitted;
            sum_w_resid_sq = sum_w_resid_sq + w * resid * resid;
        }

        variances[j] = if sum_w > F::epsilon() {
            sum_w_resid_sq / sum_w
        } else {
            F::from_f64(0.01).unwrap_or(F::epsilon())
        };

        // Ensure positive variance
        if variances[j] < F::epsilon() {
            variances[j] = F::from_f64(1e-10).unwrap_or(F::epsilon());
        }
    }

    // Update transition probabilities
    let mut transition_probs = Array2::zeros((k, k));

    for i in 0..k {
        let mut row_sum = F::zero();
        for j in 0..k {
            let mut sum = F::zero();
            // Use consecutive smoothed probabilities to estimate transitions
            for t in 0..t_eff - 1 {
                // Approximate joint P(S_t=i, S_{t+1}=j | Y)
                // Using P(S_{t+1}=j|Y) * P(S_t=i|Y) * p_{ij} / P(S_{t+1}=j|Y_{t})
                // Simplified: use product of smoothed probs (approximate)
                sum = sum + smoothed[[t, i]] * smoothed[[t + 1, j]];
            }
            transition_probs[[i, j]] = sum;
            row_sum = row_sum + sum;
        }

        // Normalize rows
        if row_sum > F::epsilon() {
            for j in 0..k {
                transition_probs[[i, j]] = transition_probs[[i, j]] / row_sum;
            }
        } else {
            // Uniform
            let uniform = F::one()
                / F::from_usize(k).ok_or_else(|| {
                    TimeSeriesError::ComputationError("Failed to convert k".to_string())
                })?;
            for j in 0..k {
                transition_probs[[i, j]] = uniform;
            }
        }
    }

    let ergodic_probs = compute_ergodic_probs(&transition_probs)?;

    Ok(MSARParams {
        intercepts,
        ar_coefficients,
        variances,
        transition_probs,
        ergodic_probs,
    })
}

// ---------------------------------------------------------------------------
// Internal: Regime classification
// ---------------------------------------------------------------------------

fn classify_regimes<F: Float>(smoothed: &Array2<F>) -> Array1<usize> {
    let (t_eff, k) = smoothed.dim();
    let mut regimes = Array1::zeros(t_eff);

    for t in 0..t_eff {
        let mut best_j = 0;
        let mut best_p = smoothed[[t, 0]];
        for j in 1..k {
            if smoothed[[t, j]] > best_p {
                best_p = smoothed[[t, j]];
                best_j = j;
            }
        }
        regimes[t] = best_j;
    }

    regimes
}

// ---------------------------------------------------------------------------
// Internal: Linear solver
// ---------------------------------------------------------------------------

fn solve_linear_system_internal<F: Float + FromPrimitive + Debug>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::epsilon() {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix encountered in regime parameter estimation".to_string(),
            ));
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < F::epsilon() {
            return Err(TimeSeriesError::NumericalInstability(
                "Zero diagonal in back substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_config(config: &MSARConfig) -> Result<()> {
    if config.n_regimes < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_regimes".to_string(),
            message: format!("Must be >= 2, got {}", config.n_regimes),
        });
    }
    if config.max_iter == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "max_iter".to_string(),
            message: "Must be positive".to_string(),
        });
    }
    if config.tolerance <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "tolerance".to_string(),
            message: format!("Must be positive, got {}", config.tolerance),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_config_validation() {
        let mut config = MSARConfig::default();
        assert!(validate_config(&config).is_ok());

        config.n_regimes = 1;
        assert!(validate_config(&config).is_err());

        config.n_regimes = 2;
        config.max_iter = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_initialize_params() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = MSARConfig::default();
        let params = initialize_params(&data, &config).expect("Init should succeed");

        assert_eq!(params.intercepts.len(), 2);
        assert_eq!(params.ar_coefficients.dim(), (2, 1));
        assert_eq!(params.variances.len(), 2);
        assert_eq!(params.transition_probs.dim(), (2, 2));
        assert_eq!(params.ergodic_probs.len(), 2);

        // Transition probs should sum to 1 per row
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| params.transition_probs[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Ergodic probs should sum to 1
        let erg_sum: f64 = params.ergodic_probs.iter().copied().sum();
        assert!((erg_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ergodic_probs_uniform() {
        // Equal transition probs => uniform ergodic
        let tp = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5])
            .expect("Failed to create matrix");
        let ergodic = compute_ergodic_probs(&tp).expect("Should succeed");
        assert!((ergodic[0] - 0.5).abs() < 1e-10);
        assert!((ergodic[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ergodic_probs_asymmetric() {
        // P = [[0.9, 0.1], [0.3, 0.7]]
        let tp = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.3, 0.7])
            .expect("Failed to create matrix");
        let ergodic = compute_ergodic_probs(&tp).expect("Should succeed");
        // pi_0 = 0.3 / (0.1 + 0.3) = 0.75, pi_1 = 0.25
        assert!((ergodic[0] - 0.75).abs() < 1e-10);
        assert!((ergodic[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_hamilton_filter_basic() {
        // Simple data with two regimes
        let data = array![
            1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 1.0, 1.1, 0.9,
            1.0, 1.1, 0.9
        ];

        let config = MSARConfig {
            n_regimes: 2,
            ar_order: 1,
            ..MSARConfig::default()
        };

        let params = initialize_params(&data, &config).expect("Init should succeed");
        let (filtered, predicted, log_ll) =
            hamilton_filter(&data, &params, &config).expect("Filter should succeed");

        assert_eq!(filtered.nrows(), data.len() - 1); // T - ar_order
        assert_eq!(filtered.ncols(), 2);
        assert!(log_ll.is_finite());

        // Filtered probs should sum to 1 at each time
        for t in 0..filtered.nrows() {
            let row_sum: f64 = (0..2).map(|j| filtered[[t, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row sum at t={t}: {row_sum}");
        }
    }

    #[test]
    fn test_kim_smoother_basic() {
        let data = array![1.0, 1.1, 0.9, 1.0, 5.0, 5.1, 4.9, 5.0, 1.0, 0.9];
        let config = MSARConfig {
            n_regimes: 2,
            ar_order: 1,
            ..MSARConfig::default()
        };

        let params = initialize_params(&data, &config).expect("Init should succeed");
        let (filtered, predicted, _) =
            hamilton_filter(&data, &params, &config).expect("Filter should succeed");
        let smoothed =
            kim_smoother(&filtered, &predicted, &params, &config).expect("Smoother should succeed");

        assert_eq!(smoothed.dim(), filtered.dim());

        // Smoothed probs should sum to 1
        for t in 0..smoothed.nrows() {
            let row_sum: f64 = (0..2).map(|j| smoothed[[t, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_classify_regimes() {
        let smoothed = Array2::from_shape_vec((4, 2), vec![0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.2, 0.8])
            .expect("Failed");
        let regimes = classify_regimes(&smoothed);
        assert_eq!(regimes[0], 0);
        assert_eq!(regimes[1], 0);
        assert_eq!(regimes[2], 1);
        assert_eq!(regimes[3], 1);
    }

    #[test]
    fn test_fit_msar_basic() {
        // Generate simple two-regime data
        let mut data_vec = Vec::new();
        // Regime 0: low mean
        for _ in 0..30 {
            data_vec.push(1.0);
        }
        // Regime 1: high mean
        for _ in 0..30 {
            data_vec.push(5.0);
        }
        // Back to regime 0
        for _ in 0..30 {
            data_vec.push(1.0);
        }
        let data = Array1::from(data_vec);

        let config = MSARConfig {
            n_regimes: 2,
            ar_order: 1,
            max_iter: 50,
            tolerance: 1e-6,
            switching_ar: true,
            switching_variance: true,
        };

        let result = fit_msar(&data, &config).expect("Fitting should succeed");

        assert_eq!(result.filtered_probs.ncols(), 2);
        assert_eq!(result.smoothed_probs.ncols(), 2);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.regime_sequence.len(), data.len() - config.ar_order);
    }

    #[test]
    fn test_fit_msar_insufficient_data() {
        let data = array![1.0, 2.0];
        let config = MSARConfig {
            ar_order: 2,
            ..MSARConfig::default()
        };
        assert!(fit_msar(&data, &config).is_err());
    }

    #[test]
    fn test_fit_msar_three_regimes() {
        let mut data_vec = Vec::new();
        for _ in 0..20 {
            data_vec.push(1.0);
        }
        for _ in 0..20 {
            data_vec.push(5.0);
        }
        for _ in 0..20 {
            data_vec.push(10.0);
        }
        let data = Array1::from(data_vec);

        let config = MSARConfig {
            n_regimes: 3,
            ar_order: 1,
            max_iter: 30,
            ..MSARConfig::default()
        };

        let result = fit_msar(&data, &config).expect("3-regime fit should succeed");
        assert_eq!(result.params.intercepts.len(), 3);
        assert_eq!(result.params.transition_probs.dim(), (3, 3));
        assert_eq!(result.filtered_probs.ncols(), 3);
    }

    #[test]
    fn test_forecast_msar() {
        let mut data_vec = Vec::new();
        for _ in 0..40 {
            data_vec.push(1.0);
        }
        for _ in 0..40 {
            data_vec.push(5.0);
        }
        let data = Array1::from(data_vec);

        let config = MSARConfig {
            n_regimes: 2,
            ar_order: 1,
            max_iter: 30,
            ..MSARConfig::default()
        };

        let result = fit_msar(&data, &config).expect("Fit should succeed");
        let forecasts = forecast_msar(&data, &result, &config, 5).expect("Forecast should succeed");

        assert_eq!(forecasts.len(), 5);
        // All forecasts should be finite
        for &f in forecasts.iter() {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_transition_matrix_access() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = MSARConfig::default();
        let params = initialize_params(&data, &config).expect("Init should succeed");
        let tm = transition_matrix(&params);
        assert_eq!(tm.dim(), (2, 2));
    }

    #[test]
    fn test_ergodic_probs_sum_to_one() {
        let tp = Array2::from_shape_vec((3, 3), vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.1, 0.7])
            .expect("Failed");
        let ergodic = compute_ergodic_probs(&tp).expect("Should succeed");
        let sum: f64 = ergodic.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // All probabilities should be positive
        for &p in ergodic.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_linear_solver() {
        // 3x3 system: x=1, y=2, z=3
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .expect("Failed");
        let b = array![1.0, 2.0, 3.0];
        let x = solve_linear_system_internal(&a, &b).expect("Should succeed");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_msar_constant_data() {
        // Constant data should still work (single effective regime)
        let data = Array1::from_elem(50, 3.0_f64);
        let config = MSARConfig {
            n_regimes: 2,
            ar_order: 1,
            max_iter: 20,
            ..MSARConfig::default()
        };
        let result = fit_msar(&data, &config).expect("Should handle constant data");
        assert!(result.log_likelihood.is_finite() || result.log_likelihood.is_nan().not());
        // At least it should not panic
    }

    trait Not {
        fn not(self) -> bool;
    }

    impl Not for bool {
        fn not(self) -> bool {
            !self
        }
    }
}
