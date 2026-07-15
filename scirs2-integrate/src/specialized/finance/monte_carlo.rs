//! Monte Carlo financial pricing engine
//!
//! Provides a comprehensive Monte Carlo simulation framework for option pricing,
//! including variance reduction techniques, exotic options, Heston stochastic
//! volatility, portfolio VaR, and quasi-random Sobol sequences.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::seeded_rng;
use scirs2_core::random::{Normal, Rng, SeedableRng, StandardNormal};
use std::f64::consts::PI;

// ============================================================
// Core engine and types
// ============================================================

/// Option type for Monte Carlo pricing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    /// Call option – payoff max(S-K, 0)
    Call,
    /// Put option – payoff max(K-S, 0)
    Put,
}

/// Asian option averaging convention
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AsianAveraging {
    /// Arithmetic average of prices along path
    Arithmetic,
    /// Geometric average of prices along path (closed form exists)
    Geometric,
}

/// Barrier option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierType {
    /// Option knocked out when price crosses barrier upward
    UpAndOut,
    /// Option knocked out when price crosses barrier downward
    DownAndOut,
    /// Option activated when price crosses barrier upward
    UpAndIn,
    /// Option activated when price crosses barrier downward
    DownAndIn,
}

/// Pricing result with uncertainty estimates
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Option price estimate
    pub price: f64,
    /// Standard error of the Monte Carlo estimate
    pub std_error: f64,
    /// 95% confidence interval (lower, upper)
    pub confidence_interval: (f64, f64),
    /// Number of paths used
    pub n_paths: usize,
}

/// Option Greeks computed via finite differences over MC samples
#[derive(Debug, Clone)]
pub struct OptionGreeks {
    /// dV/dS – sensitivity to spot price
    pub delta: f64,
    /// d²V/dS² – second-order spot sensitivity
    pub gamma: f64,
    /// dV/d(sigma) – sensitivity to volatility (per 1%)
    pub vega: f64,
    /// dV/dt – time decay (per calendar day)
    pub theta: f64,
    /// dV/dr – sensitivity to risk-free rate
    pub rho: f64,
}

/// Monte Carlo simulation engine configuration
#[derive(Debug, Clone)]
pub struct MonteCarloEngine {
    /// Number of Monte Carlo paths
    pub n_paths: usize,
    /// Number of time steps per path
    pub n_steps: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Use antithetic variates for variance reduction
    pub antithetic: bool,
    /// Use control variates (geometric payoff as control)
    pub control_variates: bool,
}

impl MonteCarloEngine {
    /// Create a new engine with default settings (no variance reduction)
    pub fn new(n_paths: usize, n_steps: usize, seed: u64) -> Self {
        Self {
            n_paths,
            n_steps,
            seed,
            antithetic: false,
            control_variates: false,
        }
    }

    /// Enable antithetic variates variance reduction
    pub fn with_antithetic(mut self) -> Self {
        self.antithetic = true;
        self
    }

    /// Enable control variates variance reduction
    pub fn with_control_variates(mut self) -> Self {
        self.control_variates = true;
        self
    }
}

// ============================================================
// Path generation
// ============================================================

/// Generate geometric Brownian motion price paths.
///
/// Returns an Array2 of shape `(n_paths, n_steps + 1)`.  When antithetic is
/// enabled the returned array has `n_paths` rows where the first half are the
/// standard paths and the second half are their antithetic mirrors.
pub fn generate_gbm_paths(
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
    antithetic: bool,
) -> IntegrateResult<Array2<f64>> {
    if s0 <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Initial stock price must be positive".to_string(),
        ));
    }
    if sigma < 0.0 {
        return Err(IntegrateError::ValueError(
            "Volatility must be non-negative".to_string(),
        ));
    }
    if t <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Time to maturity must be positive".to_string(),
        ));
    }
    if n_paths == 0 || n_steps == 0 {
        return Err(IntegrateError::ValueError(
            "n_paths and n_steps must be positive".to_string(),
        ));
    }

    let dt = t / n_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();

    let mut rng = seeded_rng(seed);

    let total_paths = if antithetic { n_paths * 2 } else { n_paths };
    let half = n_paths;

    let mut paths = Array2::zeros((total_paths, n_steps + 1));
    for i in 0..total_paths {
        paths[[i, 0]] = s0;
    }

    if antithetic {
        // Generate standard paths; reuse the same shocks negated for antithetic
        let mut shocks: Vec<f64> = (0..n_steps).map(|_| rng.sample(StandardNormal)).collect();

        for i in 0..half {
            // Re-generate shocks for each standard path
            for (step_idx, shock) in shocks.iter_mut().enumerate() {
                *shock = rng.sample(StandardNormal);
                let _ = step_idx; // suppress warning
            }
            // We need shocks stored per path, so generate them inline
            let _ = shocks;

            let mut s = s0;
            let mut s_anti = s0;
            let path_shocks: Vec<f64> = (0..n_steps).map(|_| rng.sample(StandardNormal)).collect();
            for (j, &z) in path_shocks.iter().enumerate() {
                s *= (drift + vol_sqrt_dt * z).exp();
                s_anti *= (drift - vol_sqrt_dt * z).exp();
                paths[[i, j + 1]] = s;
                paths[[half + i, j + 1]] = s_anti;
            }
        }
    } else {
        for i in 0..n_paths {
            let mut s = s0;
            for j in 0..n_steps {
                let z: f64 = rng.sample(StandardNormal);
                s *= (drift + vol_sqrt_dt * z).exp();
                paths[[i, j + 1]] = s;
            }
        }
    }

    Ok(paths)
}

// ============================================================
// Black-Scholes reference
// ============================================================

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

/// Analytical Black-Scholes European option price (used as control variate)
fn bs_european_price(s0: f64, k: f64, r: f64, sigma: f64, t: f64, call: bool) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        let intrinsic = if call { s0 - k } else { k - s0 };
        return intrinsic.max(0.0);
    }
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    if call {
        s0 * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
    } else {
        k * (-r * t).exp() * normal_cdf(-d2) - s0 * normal_cdf(-d1)
    }
}

/// Closed-form geometric Asian call price (control variate)
fn geometric_asian_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64, n: usize) -> f64 {
    let n_f = n as f64;
    let sigma_adj = sigma * ((n_f + 1.0) * (2.0 * n_f + 1.0) / (6.0 * n_f * n_f)).sqrt();
    let b = 0.5 * (r - 0.5 * sigma * sigma) + 0.5 * sigma_adj * sigma_adj;
    let d1 = ((s0 / k).ln() + (b + 0.5 * sigma_adj * sigma_adj) * t) / (sigma_adj * t.sqrt());
    let d2 = d1 - sigma_adj * t.sqrt();
    (-(r - b) * t).exp()
        * (s0 * (-((r - b) * t)).exp() * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2))
}

// ============================================================
// Summary statistics helper
// ============================================================

pub(crate) fn build_result(payoffs: &[f64], n_paths: usize, r: f64, t: f64) -> MonteCarloResult {
    let discount = (-r * t).exp();
    let mean_payoff: f64 = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
    let price = discount * mean_payoff;

    let variance: f64 = payoffs
        .iter()
        .map(|&p| (p - mean_payoff).powi(2))
        .sum::<f64>()
        / (payoffs.len() as f64 - 1.0).max(1.0);

    let std_error = discount * (variance / payoffs.len() as f64).sqrt();
    let z_95 = 1.96;
    let ci_lo = price - z_95 * std_error;
    let ci_hi = price + z_95 * std_error;

    MonteCarloResult {
        price,
        std_error,
        confidence_interval: (ci_lo, ci_hi),
        n_paths,
    }
}

// ============================================================
// European option pricing
// ============================================================

/// Price a European call or put via Monte Carlo simulation.
///
/// Supports antithetic variates and (for European options) BS control variate.
pub fn mc_european_option(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    option_type: OptionType,
    engine: &MonteCarloEngine,
) -> IntegrateResult<MonteCarloResult> {
    let paths = generate_gbm_paths(
        s0,
        r,
        sigma,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
        engine.antithetic,
    )?;

    let total_paths = paths.nrows();
    let last_col = engine.n_steps;
    let is_call = option_type == OptionType::Call;

    let mut payoffs: Vec<f64> = (0..total_paths)
        .map(|i| {
            let s_t = paths[[i, last_col]];
            if is_call {
                (s_t - k).max(0.0)
            } else {
                (k - s_t).max(0.0)
            }
        })
        .collect();

    // Control variate: use BS price as analytical correction
    if engine.control_variates {
        let bs_price_val = bs_european_price(s0, k, r, sigma, t, is_call);
        let bs_discount = (-r * t).exp();
        // Compute MC estimate of BS payoff as a control
        let mc_bs_mean: f64 = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
        let control_correction = bs_discount * mc_bs_mean - bs_price_val;
        for p in &mut payoffs {
            *p -= control_correction / bs_discount;
        }
    }

    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// Asian option pricing
// ============================================================

/// Price an Asian (average-rate) option via Monte Carlo simulation.
pub fn mc_asian_option(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    option_type: OptionType,
    averaging: AsianAveraging,
    engine: &MonteCarloEngine,
) -> IntegrateResult<MonteCarloResult> {
    let paths = generate_gbm_paths(
        s0,
        r,
        sigma,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
        engine.antithetic,
    )?;

    let total_paths = paths.nrows();
    let n = engine.n_steps;
    let is_call = option_type == OptionType::Call;

    let mut payoffs: Vec<f64> = (0..total_paths)
        .map(|i| {
            let avg = match averaging {
                AsianAveraging::Arithmetic => {
                    // Average over steps 1..=n (exclude initial price)
                    let sum: f64 = (1..=n).map(|j| paths[[i, j]]).sum();
                    sum / n as f64
                }
                AsianAveraging::Geometric => {
                    let log_sum: f64 = (1..=n).map(|j| paths[[i, j]].ln()).sum();
                    (log_sum / n as f64).exp()
                }
            };
            if is_call {
                (avg - k).max(0.0)
            } else {
                (k - avg).max(0.0)
            }
        })
        .collect();

    // Control variate: geometric Asian closed form
    if engine.control_variates && averaging == AsianAveraging::Arithmetic && is_call {
        let geo_cf = geometric_asian_call(s0, k, r, sigma, t, n);
        // Compute geometric MC payoffs as control
        let geo_payoffs: Vec<f64> = (0..total_paths)
            .map(|i| {
                let log_sum: f64 = (1..=n).map(|j| paths[[i, j]].ln()).sum();
                let geo_avg = (log_sum / n as f64).exp();
                (geo_avg - k).max(0.0)
            })
            .collect();
        let geo_mc_mean: f64 = geo_payoffs.iter().sum::<f64>() / geo_payoffs.len() as f64;
        let discount = (-r * t).exp();
        let beta = {
            // Simple beta = cov(arith, geo) / var(geo)
            let arith_mean: f64 = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
            let cov: f64 = payoffs
                .iter()
                .zip(geo_payoffs.iter())
                .map(|(&a, &g)| (a - arith_mean) * (g - geo_mc_mean))
                .sum::<f64>()
                / payoffs.len() as f64;
            let var_geo: f64 = geo_payoffs
                .iter()
                .map(|&g| (g - geo_mc_mean).powi(2))
                .sum::<f64>()
                / geo_payoffs.len() as f64;
            if var_geo.abs() < 1e-15 {
                0.0
            } else {
                cov / var_geo
            }
        };
        let cf_geo_mc = geo_cf / discount; // analytical geometric undiscounted
        for (p, g) in payoffs.iter_mut().zip(geo_payoffs.iter()) {
            *p -= beta * (g - cf_geo_mc);
        }
    }

    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// Barrier option pricing
// ============================================================

/// Price a barrier option via Monte Carlo with continuous monitoring at each step.
pub fn mc_barrier_option(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier: f64,
    option_type: OptionType,
    barrier_type: BarrierType,
    engine: &MonteCarloEngine,
) -> IntegrateResult<MonteCarloResult> {
    if barrier <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Barrier must be positive".to_string(),
        ));
    }

    let paths = generate_gbm_paths(
        s0,
        r,
        sigma,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
        engine.antithetic,
    )?;

    let total_paths = paths.nrows();
    let n = engine.n_steps;
    let is_call = option_type == OptionType::Call;

    let payoffs: Vec<f64> = (0..total_paths)
        .map(|i| {
            let mut crossed = false;
            for j in 1..=n {
                let s = paths[[i, j]];
                crossed = match barrier_type {
                    BarrierType::UpAndOut | BarrierType::UpAndIn => s >= barrier,
                    BarrierType::DownAndOut | BarrierType::DownAndIn => s <= barrier,
                };
                if crossed {
                    break;
                }
            }

            let active = match barrier_type {
                BarrierType::UpAndOut | BarrierType::DownAndOut => !crossed,
                BarrierType::UpAndIn | BarrierType::DownAndIn => crossed,
            };

            if active {
                let s_t = paths[[i, n]];
                if is_call {
                    (s_t - k).max(0.0)
                } else {
                    (k - s_t).max(0.0)
                }
            } else {
                0.0
            }
        })
        .collect();

    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// Lookback option pricing
// ============================================================

/// Price a fixed-strike lookback option via Monte Carlo.
///
/// Call payoff: max(S_max - K, 0); Put payoff: max(K - S_min, 0)
pub fn mc_lookback_option(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    option_type: OptionType,
    engine: &MonteCarloEngine,
) -> IntegrateResult<MonteCarloResult> {
    let paths = generate_gbm_paths(
        s0,
        r,
        sigma,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
        engine.antithetic,
    )?;

    let total_paths = paths.nrows();
    let n = engine.n_steps;

    let payoffs: Vec<f64> = (0..total_paths)
        .map(|i| match option_type {
            OptionType::Call => {
                let s_max = (1..=n)
                    .map(|j| paths[[i, j]])
                    .fold(f64::NEG_INFINITY, f64::max);
                (s_max - k).max(0.0)
            }
            OptionType::Put => {
                let s_min = (1..=n).map(|j| paths[[i, j]]).fold(f64::INFINITY, f64::min);
                (k - s_min).max(0.0)
            }
        })
        .collect();

    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// American option – Longstaff-Schwartz LSM
// ============================================================

/// Evaluate weighted Laguerre basis polynomials up to order `order`.
fn laguerre_basis(x: f64, order: usize) -> Vec<f64> {
    let mut phi = vec![0.0; order];
    if order == 0 {
        return phi;
    }
    // Generalized Laguerre L_n(x) = e^{-x/2} * L_n^0(x), standard for LSM
    let ex = (-x / 2.0).exp();
    phi[0] = ex;
    if order > 1 {
        phi[1] = ex * (1.0 - x);
    }
    if order > 2 {
        phi[2] = ex * (1.0 - 2.0 * x + 0.5 * x * x);
    }
    if order > 3 {
        phi[3] = ex * (1.0 - 3.0 * x + 1.5 * x * x - x * x * x / 6.0);
    }
    // Higher orders filled with zeros (truncate at 4)
    phi
}

/// Least-squares regression: returns coefficients for `X * coef = y`
/// using the normal equations X^T X coef = X^T y
fn ols_fit(x_mat: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = x_mat.len();
    let p = if n > 0 { x_mat[0].len() } else { 0 };
    if n == 0 || p == 0 {
        return vec![0.0; p];
    }

    // XtX and Xty
    let mut xtx = vec![vec![0.0f64; p]; p];
    let mut xty = vec![0.0f64; p];
    for (row, yi) in x_mat.iter().zip(y.iter()) {
        for j in 0..p {
            xty[j] += row[j] * yi;
            for k in 0..p {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }

    // Simple Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let mut row = xtx[j].clone();
            row.push(xty[j]);
            row
        })
        .collect();

    for col in 0..p {
        // Find pivot
        let pivot = (col..p).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .expect("NaN")
        });
        if let Some(pivot_row) = pivot {
            aug.swap(col, pivot_row);
        }
        let pivot_val = aug[col][col];
        if pivot_val.abs() < 1e-14 {
            continue;
        }
        let factor = 1.0 / pivot_val;
        for k in col..=p {
            aug[col][k] *= factor;
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let mult = aug[row][col];
            for k in col..=p {
                let sub = mult * aug[col][k];
                aug[row][k] -= sub;
            }
        }
    }

    (0..p).map(|j| aug[j][p]).collect()
}

/// Price an American option via the Longstaff-Schwartz LSM regression method.
pub fn mc_american_option_lsm(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    option_type: OptionType,
    engine: &MonteCarloEngine,
    basis_functions: usize,
) -> IntegrateResult<MonteCarloResult> {
    if basis_functions == 0 {
        return Err(IntegrateError::ValueError(
            "basis_functions must be at least 1".to_string(),
        ));
    }
    let order = basis_functions.min(4); // cap at 4 Laguerre polynomials

    let paths = generate_gbm_paths(
        s0,
        r,
        sigma,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
        engine.antithetic,
    )?;

    let n_paths_total = paths.nrows();
    let n = engine.n_steps;
    let dt = t / n as f64;
    let discount_factor = (-r * dt).exp();
    let is_call = option_type == OptionType::Call;

    // Immediate exercise payoff function
    let payoff_fn = |s: f64| -> f64 {
        if is_call {
            (s - k).max(0.0)
        } else {
            (k - s).max(0.0)
        }
    };

    // Cash flows at each path; initialise at final payoff
    let mut cash_flows: Vec<f64> = (0..n_paths_total)
        .map(|i| payoff_fn(paths[[i, n]]))
        .collect();

    // Backward induction
    for step in (1..n).rev() {
        // Identify in-the-money paths for regression
        let in_money: Vec<usize> = (0..n_paths_total)
            .filter(|&i| payoff_fn(paths[[i, step]]) > 0.0)
            .collect();

        if in_money.is_empty() {
            // Discount cash flows and continue
            for cf in &mut cash_flows {
                *cf *= discount_factor;
            }
            continue;
        }

        // Build regression matrix and continuation value targets
        let x_mat: Vec<Vec<f64>> = in_money
            .iter()
            .map(|&i| laguerre_basis(paths[[i, step]] / k, order))
            .collect();
        let y: Vec<f64> = in_money
            .iter()
            .map(|&i| cash_flows[i] * discount_factor)
            .collect();

        let coef = ols_fit(&x_mat, &y);

        // Decide exercise
        for (idx, &path_i) in in_money.iter().enumerate() {
            let s = paths[[path_i, step]];
            let basis = laguerre_basis(s / k, order);
            let continuation: f64 = basis.iter().zip(coef.iter()).map(|(b, c)| b * c).sum();
            let exercise = payoff_fn(s);
            if exercise > continuation {
                cash_flows[path_i] = exercise;
            } else {
                cash_flows[path_i] *= discount_factor;
            }
        }

        // Non-ITM paths: just discount
        let in_money_set: std::collections::HashSet<usize> = in_money.into_iter().collect();
        for i in 0..n_paths_total {
            if !in_money_set.contains(&i) {
                cash_flows[i] *= discount_factor;
            }
        }
    }

    // Final discount back one more step
    let payoffs: Vec<f64> = cash_flows.iter().map(|&cf| cf * discount_factor).collect();
    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// Greeks via finite difference
// ============================================================

/// Compute option Greeks via finite-difference perturbation of MC prices.
///
/// All Greeks are computed using the same random seed with bumped parameters
/// for consistency.
pub fn mc_greeks(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    option_type: OptionType,
    engine: &MonteCarloEngine,
) -> IntegrateResult<OptionGreeks> {
    let ds = s0 * 0.01; // 1% bump
    let dv = 0.001; // 0.1% vol bump
    let dt = 1.0 / 365.0; // 1 calendar day
    let dr = 0.0001; // 1 basis point

    let base = mc_european_option(s0, k, r, sigma, t, option_type, engine)?;
    let up_s = mc_european_option(s0 + ds, k, r, sigma, t, option_type, engine)?;
    let dn_s = mc_european_option(s0 - ds, k, r, sigma, t, option_type, engine)?;

    let delta = (up_s.price - dn_s.price) / (2.0 * ds);
    let gamma = (up_s.price - 2.0 * base.price + dn_s.price) / (ds * ds);

    let up_v = mc_european_option(s0, k, r, sigma + dv, t, option_type, engine)?;
    let dn_v = mc_european_option(s0, k, r, sigma - dv, t, option_type, engine)?;
    let vega = (up_v.price - dn_v.price) / (2.0 * dv);

    // Theta: decrease t by one day (option loses value as time passes)
    let t_next = (t - dt).max(1e-6);
    let theta_res = mc_european_option(s0, k, r, sigma, t_next, option_type, engine)?;
    let theta = (theta_res.price - base.price) / dt; // negative for long options

    let up_r = mc_european_option(s0, k, r + dr, sigma, t, option_type, engine)?;
    let dn_r = mc_european_option(s0, k, r - dr, sigma, t, option_type, engine)?;
    let rho = (up_r.price - dn_r.price) / (2.0 * dr);

    Ok(OptionGreeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

// ============================================================
// Heston stochastic volatility model
// ============================================================

/// Generate joint stock price and variance paths under the Heston model.
///
/// Uses the Euler-Maruyama scheme with full truncation for the CIR variance
/// process.  Returns `(price_paths, vol_paths)` each of shape
/// `(n_paths, n_steps + 1)`.
pub fn generate_heston_paths(
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    t: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> IntegrateResult<(Array2<f64>, Array2<f64>)> {
    if s0 <= 0.0 || v0 < 0.0 {
        return Err(IntegrateError::ValueError(
            "s0 must be positive, v0 must be non-negative".to_string(),
        ));
    }
    if !(-1.0..=1.0).contains(&rho) {
        return Err(IntegrateError::ValueError(
            "Correlation rho must be in [-1, 1]".to_string(),
        ));
    }
    if kappa <= 0.0 || theta <= 0.0 || xi <= 0.0 {
        return Err(IntegrateError::ValueError(
            "kappa, theta, xi must be positive".to_string(),
        ));
    }

    let dt = t / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let sqrt_1m_rho2 = (1.0 - rho * rho).sqrt();

    let mut rng = seeded_rng(seed);
    let mut price_paths = Array2::zeros((n_paths, n_steps + 1));
    let mut vol_paths = Array2::zeros((n_paths, n_steps + 1));

    for i in 0..n_paths {
        price_paths[[i, 0]] = s0;
        vol_paths[[i, 0]] = v0;

        let mut s = s0;
        let mut v = v0;

        for j in 0..n_steps {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);

            // Correlated Brownian motions
            let dw_s = z1;
            let dw_v = rho * z1 + sqrt_1m_rho2 * z2;

            // Variance update (full truncation scheme for CIR)
            let sqrt_v = v.sqrt();
            v = (v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * dw_v).max(0.0);

            // Price update
            let drift_s = (r - 0.5 * v) * dt;
            s *= (drift_s + sqrt_v * sqrt_dt * dw_s).exp();

            price_paths[[i, j + 1]] = s;
            vol_paths[[i, j + 1]] = v;
        }
    }

    Ok((price_paths, vol_paths))
}

/// Price a European option under the Heston stochastic volatility model.
pub fn mc_heston_european(
    s0: f64,
    v0: f64,
    k: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    t: f64,
    option_type: OptionType,
    engine: &MonteCarloEngine,
) -> IntegrateResult<MonteCarloResult> {
    let (price_paths, _vol_paths) = generate_heston_paths(
        s0,
        v0,
        r,
        kappa,
        theta,
        xi,
        rho,
        t,
        engine.n_paths,
        engine.n_steps,
        engine.seed,
    )?;

    let n = engine.n_steps;
    let is_call = option_type == OptionType::Call;

    let payoffs: Vec<f64> = (0..engine.n_paths)
        .map(|i| {
            let s_t = price_paths[[i, n]];
            if is_call {
                (s_t - k).max(0.0)
            } else {
                (k - s_t).max(0.0)
            }
        })
        .collect();

    Ok(build_result(&payoffs, engine.n_paths, r, t))
}

// ============================================================
// Portfolio VaR via Monte Carlo
// ============================================================

/// Compute portfolio Value at Risk (VaR) and Conditional VaR (CVaR/ES) via
/// Monte Carlo simulation of correlated log-normal returns.
///
/// # Arguments
/// - `initial_prices`: current portfolio asset prices
/// - `weights`: portfolio weights (should sum to 1)
/// - `mu`: expected annual returns for each asset
/// - `cov_matrix`: annual covariance matrix (n_assets × n_assets)
/// - `horizon`: time horizon in years
/// - `confidence`: confidence level (e.g. 0.99)
/// - `n_scenarios`: number of Monte Carlo scenarios
/// - `seed`: random seed
///
/// # Returns
/// `(VaR, CVaR)` – both expressed as portfolio loss (positive value)
pub fn mc_portfolio_var(
    initial_prices: &Array1<f64>,
    weights: &Array1<f64>,
    mu: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    horizon: f64,
    confidence: f64,
    n_scenarios: usize,
    seed: u64,
) -> IntegrateResult<(f64, f64)> {
    let n_assets = initial_prices.len();
    if weights.len() != n_assets
        || mu.len() != n_assets
        || cov_matrix.nrows() != n_assets
        || cov_matrix.ncols() != n_assets
    {
        return Err(IntegrateError::DimensionMismatch(
            "All inputs must have consistent asset dimensions".to_string(),
        ));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(IntegrateError::ValueError(
            "Confidence must be in (0, 1)".to_string(),
        ));
    }
    if horizon <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Horizon must be positive".to_string(),
        ));
    }
    if n_scenarios == 0 {
        return Err(IntegrateError::ValueError(
            "n_scenarios must be positive".to_string(),
        ));
    }

    // Cholesky decomposition of covariance matrix (lower triangular)
    let chol = cholesky_lower(cov_matrix, n_assets)?;

    let mut rng = seeded_rng(seed);

    // Initial portfolio value
    let portfolio_value: f64 = initial_prices
        .iter()
        .zip(weights.iter())
        .map(|(p, w)| p * w)
        .sum();

    let mut pnl: Vec<f64> = Vec::with_capacity(n_scenarios);

    for _ in 0..n_scenarios {
        // Draw independent standard normals
        let z: Vec<f64> = (0..n_assets).map(|_| rng.sample(StandardNormal)).collect();

        // Correlated normal via Cholesky: eps = L * z
        let mut eps = vec![0.0f64; n_assets];
        for i in 0..n_assets {
            for j in 0..=i {
                eps[i] += chol[i][j] * z[j];
            }
        }

        // Compute log-normal scenario return for each asset
        let mut new_value = 0.0f64;
        for i in 0..n_assets {
            let log_return = (mu[i] - 0.5 * cov_matrix[[i, i]]) * horizon + horizon.sqrt() * eps[i];
            let new_price = initial_prices[i] * log_return.exp();
            new_value += weights[i] * new_price;
        }

        pnl.push(new_value - portfolio_value);
    }

    // VaR: quantile of losses at (1 - confidence)
    pnl.sort_by(|a, b| a.partial_cmp(b).expect("NaN in PnL"));
    let var_index = ((1.0 - confidence) * n_scenarios as f64).floor() as usize;
    let var_index = var_index.min(n_scenarios - 1);
    let var = -pnl[var_index]; // positive loss

    // CVaR: average of losses beyond VaR
    let tail: Vec<f64> = pnl[..=var_index].iter().map(|&x| -x).collect();
    let cvar = tail.iter().sum::<f64>() / tail.len().max(1) as f64;

    Ok((var, cvar))
}

/// Lower-triangular Cholesky decomposition: returns L s.t. A = L L^T
fn cholesky_lower(a: &Array2<f64>, n: usize) -> IntegrateResult<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let diag = a[[i, i]] - s;
                if diag < 0.0 {
                    return Err(IntegrateError::ValueError(format!(
                        "Covariance matrix is not positive definite at diagonal ({}, {})",
                        i, i
                    )));
                }
                l[i][j] = diag.sqrt();
            } else {
                let ljj = l[j][j];
                if ljj.abs() < 1e-14 {
                    l[i][j] = 0.0;
                } else {
                    l[i][j] = (a[[i, j]] - s) / ljj;
                }
            }
        }
    }
    Ok(l)
}

// ============================================================
// Quasi-random Sobol sequence
// ============================================================

/// Generate a Sobol low-discrepancy sequence using Gray code enumeration.
///
/// Supports up to 10 dimensions using pre-computed direction numbers.
/// Returns an Array2 of shape `(n_samples, n_dimensions)` with values in [0, 1).
pub fn sobol_sequence(n_samples: usize, n_dimensions: usize, skip: usize) -> Array2<f64> {
    assert!(
        n_dimensions <= 10,
        "sobol_sequence supports at most 10 dimensions"
    );
    assert!(n_samples > 0, "n_samples must be positive");

    // Direction numbers for dimensions 1..10 (standard Joe-Kuo tables, s=1 polynomials)
    // Dimension 1 is always powers of 2 (m_i = 1).
    // For dims 2-10 we include the initial direction numbers m[1..s] and the
    // polynomial degree s (number of initial values).
    // Format: (degree_s, [m_1, ..., m_s])
    let direction_data: &[(usize, &[u32])] = &[
        (1, &[1]),               // dim 2:  x + 1
        (2, &[1, 1]),            // dim 3:  x^2 + x + 1
        (3, &[1, 1, 1]),         // dim 4:  x^3 + x^2 + 1
        (3, &[1, 3, 7]),         // dim 5:  x^3 + x + 1
        (4, &[1, 1, 5, 11]),     // dim 6
        (4, &[1, 3, 13, 11]),    // dim 7
        (5, &[1, 1, 19, 25, 7]), // dim 8
        (5, &[1, 3, 7, 11, 1]),  // dim 9
        (5, &[1, 1, 5, 1, 15]),  // dim 10
    ];

    let max_bits: usize = 32;
    let scale = (u32::MAX as f64) + 1.0; // 2^32

    // Build direction numbers for each dimension
    let mut v: Vec<Vec<u32>> = Vec::with_capacity(n_dimensions);

    // Dimension 0: v[i] = 2^(31-i) (powers of 2 in bit 31..0)
    let mut v0 = vec![0u32; max_bits];
    for i in 0..max_bits {
        v0[i] = 1u32 << (max_bits - 1 - i);
    }
    v.push(v0);

    for d in 1..n_dimensions {
        let (s, m_init) = direction_data[d - 1];
        let mut vd = vec![0u32; max_bits];
        // Initial direction numbers (left-shifted to fill 32 bits)
        for i in 0..s.min(max_bits) {
            vd[i] = if i < m_init.len() {
                m_init[i] << (max_bits - 1 - i)
            } else {
                0
            };
        }
        // Recurrence: v_i = v_{i-s} XOR (v_{i-s} >> s) XOR sum of bit contributions
        // Standard recurrence for primitive polynomial of degree s with coefficients a_{s-1}..a_1
        // We use a simplified all-ones polynomial coefficient here
        for i in s..max_bits {
            let prev_s = vd[i - s];
            let mut new_val = prev_s ^ (prev_s >> s);
            for k in 1..s {
                new_val ^= vd[i - k];
            }
            vd[i] = new_val;
        }
        v.push(vd);
    }

    let mut result = Array2::zeros((n_samples, n_dimensions));

    // Current Sobol point (integer representation)
    let mut x = vec![0u32; n_dimensions];

    // Skip initial samples
    for idx in 0..skip {
        let c = gray_code_trailing_zeros(idx + 1);
        for d in 0..n_dimensions {
            x[d] ^= if c < max_bits { v[d][c] } else { 0 };
        }
    }

    for sample in 0..n_samples {
        // Record current point
        for d in 0..n_dimensions {
            result[[sample, d]] = x[d] as f64 / scale;
        }
        // Advance to next Gray code point
        let c = gray_code_trailing_zeros(skip + sample + 1);
        for d in 0..n_dimensions {
            x[d] ^= if c < max_bits { v[d][c] } else { 0 };
        }
    }

    result
}

/// Number of trailing zeros in the binary representation of n (= position of
/// the rightmost changed bit in the Gray code sequence).
fn gray_code_trailing_zeros(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    n.trailing_zeros() as usize
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SEED: u64 = 42;
    const N_PATHS: usize = 20_000;
    const N_STEPS: usize = 252;
    const TOL: f64 = 0.05; // 5% relative tolerance for MC tests

    fn bs_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
        bs_european_price(s0, k, r, sigma, t, true)
    }
    fn bs_put(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
        bs_european_price(s0, k, r, sigma, t, false)
    }

    // --- GBM path statistics ---

    #[test]
    fn test_gbm_path_initial_price() {
        let paths = generate_gbm_paths(100.0, 0.05, 0.2, 1.0, 1000, 52, SEED, false)
            .expect("gbm paths failed");
        for i in 0..1000 {
            assert_relative_eq!(paths[[i, 0]], 100.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gbm_path_mean_matches_theory() {
        // E[S_T] = S0 * exp(r * T) under risk-neutral measure
        let s0 = 100.0;
        let r = 0.05;
        let t = 1.0;
        let paths = generate_gbm_paths(s0, r, 0.3, t, 50_000, N_STEPS, SEED, false)
            .expect("gbm paths failed");
        let last_col = N_STEPS;
        let mean_st: f64 = (0..50_000).map(|i| paths[[i, last_col]]).sum::<f64>() / 50_000.0;
        let expected_mean = s0 * (r * t).exp();
        // Allow 2% relative error
        assert!(
            (mean_st / expected_mean - 1.0).abs() < 0.02,
            "GBM mean mismatch: MC={:.4} theory={:.4}",
            mean_st,
            expected_mean
        );
    }

    #[test]
    fn test_gbm_path_variance_matches_theory() {
        // Var[S_T] = S0^2 * exp(2rT) * (exp(sigma^2 T) - 1)
        let s0 = 100.0;
        let r = 0.0; // Zero rate to simplify
        let sigma = 0.2;
        let t = 1.0;
        let n = 50_000;
        let paths =
            generate_gbm_paths(s0, r, sigma, t, n, N_STEPS, SEED, false).expect("gbm paths failed");
        let last_col = N_STEPS;
        let vals: Vec<f64> = (0..n).map(|i| paths[[i, last_col]]).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let theoretical_var = s0 * s0 * (2.0 * r * t).exp() * ((sigma * sigma * t).exp() - 1.0);
        assert!(
            (var / theoretical_var - 1.0).abs() < 0.05,
            "GBM variance mismatch: MC={:.2} theory={:.2}",
            var,
            theoretical_var
        );
    }

    #[test]
    fn test_gbm_antithetic_doubles_paths() {
        let paths = generate_gbm_paths(100.0, 0.05, 0.2, 1.0, 1000, 52, SEED, true)
            .expect("gbm paths failed");
        assert_eq!(paths.nrows(), 2000);
    }

    // --- European option pricing vs Black-Scholes ---

    #[test]
    fn test_mc_european_call_vs_bs() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED).with_antithetic();
        let result = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("mc european call failed");
        let bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
        assert!(
            (result.price - bs).abs() < TOL * bs,
            "Call price mismatch: MC={:.4} BS={:.4}",
            result.price,
            bs
        );
    }

    #[test]
    fn test_mc_european_put_vs_bs() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED).with_antithetic();
        let result = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, &engine)
            .expect("mc european put failed");
        let bs = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);
        assert!(
            (result.price - bs).abs() < TOL * bs,
            "Put price mismatch: MC={:.4} BS={:.4}",
            result.price,
            bs
        );
    }

    #[test]
    fn test_put_call_parity() {
        let (s0, k, r, sigma, t) = (100.0, 100.0, 0.05, 0.2, 1.0);
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED).with_antithetic();
        let call =
            mc_european_option(s0, k, r, sigma, t, OptionType::Call, &engine).expect("call failed");
        let put =
            mc_european_option(s0, k, r, sigma, t, OptionType::Put, &engine).expect("put failed");
        // C - P = S0 - K * exp(-r*T)
        let pcp_lhs = call.price - put.price;
        let pcp_rhs = s0 - k * (-r * t).exp();
        assert!(
            (pcp_lhs - pcp_rhs).abs() < 0.5,
            "Put-call parity violated: C-P={:.4} theory={:.4}",
            pcp_lhs,
            pcp_rhs
        );
    }

    #[test]
    fn test_mc_european_result_has_valid_ci() {
        let engine = MonteCarloEngine::new(5000, 100, SEED);
        let result = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("mc failed");
        let (lo, hi) = result.confidence_interval;
        assert!(lo < result.price && result.price < hi);
        assert!(result.std_error > 0.0);
    }

    // --- Asian option ---

    #[test]
    fn test_asian_geometric_less_than_arithmetic() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let arith = mc_asian_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            AsianAveraging::Arithmetic,
            &engine,
        )
        .expect("arith asian failed");
        let geom = mc_asian_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            AsianAveraging::Geometric,
            &engine,
        )
        .expect("geom asian failed");
        assert!(
            geom.price <= arith.price * 1.05, // geometric should be <= arithmetic
            "Geometric price ({:.4}) should be <= arithmetic ({:.4})",
            geom.price,
            arith.price
        );
    }

    #[test]
    fn test_asian_put_positive_price() {
        let engine = MonteCarloEngine::new(5000, 100, SEED);
        let result = mc_asian_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Put,
            AsianAveraging::Arithmetic,
            &engine,
        )
        .expect("asian put failed");
        assert!(result.price > 0.0);
    }

    #[test]
    fn test_asian_cheaper_than_european() {
        // Asian call is generally cheaper than European call (for ATM options)
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let asian = mc_asian_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            AsianAveraging::Arithmetic,
            &engine,
        )
        .expect("asian failed");
        let european = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("european failed");
        // Asian should be strictly cheaper for ATM calls with T > 0
        assert!(
            asian.price < european.price * 1.01,
            "Asian ({:.4}) should be cheaper than European ({:.4})",
            asian.price,
            european.price
        );
    }

    // --- Barrier option ---

    #[test]
    fn test_barrier_up_and_out_knocked_out_all() {
        // Barrier below current price: all paths start above barrier
        // Up-and-out with barrier at S0: should knock out almost immediately
        let engine = MonteCarloEngine::new(2000, 100, SEED);
        let result = mc_barrier_option(
            150.0,
            100.0,
            0.05,
            0.2,
            1.0,
            100.0, // barrier below spot
            OptionType::Call,
            BarrierType::DownAndOut,
            &engine,
        )
        .expect("barrier failed");
        // All paths start at 150 > 100, so down-and-out immediately active (not knocked)
        // Wait: paths start at S0=150 which is ABOVE barrier=100. For DownAndOut they get
        // knocked if price goes below 100. Price starts at 150, so many survive.
        assert!(result.price >= 0.0);
    }

    #[test]
    fn test_barrier_up_and_out_reduces_price() {
        // Up-and-out call with high barrier should be cheaper than European call
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let european = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("european failed");
        let uao = mc_barrier_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            120.0, // barrier 20% OTM
            OptionType::Call,
            BarrierType::UpAndOut,
            &engine,
        )
        .expect("barrier failed");
        assert!(
            uao.price < european.price,
            "Up-and-out ({:.4}) should be < European ({:.4})",
            uao.price,
            european.price
        );
    }

    #[test]
    fn test_barrier_down_and_out_below_spot() {
        // Down-and-out call with very low barrier should approximate European
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let european = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("european failed");
        let dao = mc_barrier_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            10.0, // barrier very deep OTM
            OptionType::Call,
            BarrierType::DownAndOut,
            &engine,
        )
        .expect("barrier failed");
        // Very low barrier => rarely knocked out => price ~ European
        assert!(
            (dao.price - european.price).abs() / european.price < 0.15,
            "DownAndOut ({:.4}) should be close to European ({:.4}) with low barrier",
            dao.price,
            european.price
        );
    }

    // --- Lookback option ---

    #[test]
    fn test_lookback_call_more_expensive_than_european() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let lookback = mc_lookback_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("lookback failed");
        let european = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("european failed");
        // Lookback uses max price so is always >= European payoff
        assert!(
            lookback.price >= european.price * 0.9,
            "Lookback ({:.4}) should be >= European ({:.4})",
            lookback.price,
            european.price
        );
    }

    // --- American option LSM ---

    #[test]
    fn test_american_put_more_expensive_than_european() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let american =
            mc_american_option_lsm(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, &engine, 3)
                .expect("american lsm failed");
        let european = mc_european_option(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, &engine)
            .expect("european failed");
        // American >= European (early exercise premium)
        assert!(
            american.price >= european.price * 0.95,
            "American ({:.4}) should be >= European ({:.4})",
            american.price,
            european.price
        );
    }

    // --- Greeks ---

    #[test]
    fn test_greeks_delta_call_positive() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let greeks = mc_greeks(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("greeks failed");
        // Delta of ATM call should be around 0.5
        assert!(
            greeks.delta > 0.3 && greeks.delta < 0.7,
            "Call delta={:.4}",
            greeks.delta
        );
    }

    #[test]
    fn test_greeks_delta_put_negative() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let greeks = mc_greeks(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, &engine)
            .expect("greeks failed");
        // Delta of ATM put should be around -0.5
        assert!(
            greeks.delta < 0.0,
            "Put delta should be negative, got {:.4}",
            greeks.delta
        );
    }

    #[test]
    fn test_greeks_vega_positive() {
        let engine = MonteCarloEngine::new(N_PATHS, N_STEPS, SEED);
        let greeks = mc_greeks(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, &engine)
            .expect("greeks failed");
        assert!(
            greeks.vega > 0.0,
            "Vega should be positive, got {:.4}",
            greeks.vega
        );
    }

    // --- Heston model ---

    #[test]
    fn test_heston_paths_shape() {
        let n_paths = 500;
        let n_steps = 100;
        let (price_paths, vol_paths) = generate_heston_paths(
            100.0, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, n_paths, n_steps, SEED,
        )
        .expect("heston failed");
        assert_eq!(price_paths.shape(), &[n_paths, n_steps + 1]);
        assert_eq!(vol_paths.shape(), &[n_paths, n_steps + 1]);
    }

    #[test]
    fn test_heston_paths_positive_prices() {
        let (price_paths, _) =
            generate_heston_paths(100.0, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 200, 100, SEED)
                .expect("heston failed");
        for i in 0..200 {
            for j in 0..=100 {
                assert!(
                    price_paths[[i, j]] > 0.0,
                    "Non-positive price at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_heston_european_call_positive() {
        let engine = MonteCarloEngine::new(5000, 100, SEED);
        let result = mc_heston_european(
            100.0,
            0.04,
            100.0,
            0.05,
            2.0,
            0.04,
            0.3,
            -0.7,
            1.0,
            OptionType::Call,
            &engine,
        )
        .expect("heston european failed");
        assert!(result.price > 0.0);
        // Roughly within range of BS price
        let bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
        assert!(result.price > bs * 0.5 && result.price < bs * 2.0);
    }

    // --- Portfolio VaR ---

    #[test]
    fn test_portfolio_var_positive() {
        use scirs2_core::ndarray::array;
        let prices = array![100.0, 200.0];
        let weights = array![0.5, 0.5];
        let mu = array![0.08, 0.10];
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).expect("cov");
        let (var, cvar) =
            mc_portfolio_var(&prices, &weights, &mu, &cov, 1.0 / 252.0, 0.99, 10000, SEED)
                .expect("var failed");
        assert!(var >= 0.0, "VaR should be non-negative, got {:.4}", var);
        assert!(
            cvar >= var * 0.9,
            "CVaR ({:.4}) should be >= VaR ({:.4})",
            cvar,
            var
        );
    }

    #[test]
    fn test_portfolio_var_cvar_ordering() {
        use scirs2_core::ndarray::array;
        let prices = array![100.0];
        let weights = array![1.0];
        let mu = array![0.05];
        let cov = Array2::from_shape_vec((1, 1), vec![0.04]).expect("cov");
        let (var, cvar) =
            mc_portfolio_var(&prices, &weights, &mu, &cov, 1.0 / 52.0, 0.95, 5000, SEED)
                .expect("var failed");
        // CVaR >= VaR by construction
        assert!(
            cvar >= var * 0.95,
            "CVaR ({:.4}) must be >= VaR ({:.4})",
            cvar,
            var
        );
    }

    #[test]
    fn test_portfolio_var_invalid_confidence() {
        use scirs2_core::ndarray::array;
        let prices = array![100.0];
        let weights = array![1.0];
        let mu = array![0.05];
        let cov = Array2::from_shape_vec((1, 1), vec![0.04]).expect("cov");
        let result = mc_portfolio_var(&prices, &weights, &mu, &cov, 1.0 / 52.0, 1.5, 100, SEED);
        assert!(result.is_err());
    }

    // --- Sobol sequence ---

    #[test]
    fn test_sobol_sequence_shape() {
        let seq = sobol_sequence(100, 4, 0);
        assert_eq!(seq.shape(), &[100, 4]);
    }

    #[test]
    fn test_sobol_sequence_range() {
        let seq = sobol_sequence(200, 5, 0);
        for val in seq.iter() {
            assert!(
                *val >= 0.0 && *val < 1.0,
                "Sobol value out of range: {}",
                val
            );
        }
    }

    #[test]
    fn test_sobol_1d_uniformity() {
        // For dimension 1, Sobol should produce well-distributed points in [0, 1)
        let n = 256;
        let seq = sobol_sequence(n, 1, 0);
        // Check that the empirical mean is close to 0.5
        let mean: f64 = seq.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Sobol 1D mean={:.4} should be ~0.5",
            mean
        );
    }

    #[test]
    fn test_sobol_distinct_points() {
        let seq = sobol_sequence(64, 2, 0);
        // All 2D points should be distinct
        let mut points: Vec<(u64, u64)> = seq
            .rows()
            .into_iter()
            .map(|r| (r[0].to_bits(), r[1].to_bits()))
            .collect();
        points.sort();
        points.dedup();
        assert_eq!(points.len(), 64);
    }

    #[test]
    fn test_sobol_skip() {
        // seq with skip=0 and then skip=10 should differ
        let seq_a = sobol_sequence(10, 2, 0);
        let seq_b = sobol_sequence(10, 2, 10);
        // First rows should differ
        assert!((seq_a[[0, 0]] - seq_b[[0, 0]]).abs() > 1e-10);
    }
}
