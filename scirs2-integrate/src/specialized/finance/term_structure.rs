//! Term structure models for interest rate curves
//!
//! Provides yield curve bootstrapping, Nelson-Siegel parametric fitting,
//! Vasicek short-rate model, and standard compounding conventions.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{seeded_rng, StandardNormal};
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::E;

// ============================================================
// Compounding conventions
// ============================================================

/// Interest rate compounding convention
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Compounding {
    /// Continuously compounded: P = exp(-r * T)
    Continuous,
    /// Annually compounded: P = (1 + r)^{-T}
    Annual,
    /// Semi-annually compounded: P = (1 + r/2)^{-2T}
    SemiAnnual,
}

/// Compute discount factor from a spot rate and maturity.
///
/// ```
/// use scirs2_integrate::specialized::finance::term_structure::{discount_factor, Compounding};
///
/// let df = discount_factor(0.05, 1.0, Compounding::Continuous);
/// assert!((df - (-0.05f64).exp()).abs() < 1e-10);
/// ```
pub fn discount_factor(spot_rate: f64, maturity: f64, compounding: Compounding) -> f64 {
    if maturity <= 0.0 {
        return 1.0;
    }
    match compounding {
        Compounding::Continuous => (-spot_rate * maturity).exp(),
        Compounding::Annual => (1.0 + spot_rate).powf(-maturity),
        Compounding::SemiAnnual => (1.0 + spot_rate / 2.0).powf(-2.0 * maturity),
    }
}

/// Convert between compounding conventions.
pub fn convert_rate(
    rate: f64,
    from: Compounding,
    to: Compounding,
    maturity: f64,
) -> IntegrateResult<f64> {
    if maturity <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Maturity must be positive".to_string(),
        ));
    }
    // First convert to continuous
    let continuous_rate = match from {
        Compounding::Continuous => rate,
        Compounding::Annual => (1.0 + rate).ln(),
        Compounding::SemiAnnual => 2.0 * (1.0 + rate / 2.0).ln(),
    };
    // Then convert to target
    let result = match to {
        Compounding::Continuous => continuous_rate,
        Compounding::Annual => continuous_rate.exp() - 1.0,
        Compounding::SemiAnnual => 2.0 * (continuous_rate / 2.0).exp() - 2.0,
    };
    Ok(result)
}

// ============================================================
// Yield curve bootstrapping
// ============================================================

/// Bootstrap a zero-coupon yield curve from par rates.
///
/// Given par coupon rates for bonds with regular coupon payments, extracts
/// the implied spot (zero-coupon) rates via sequential bootstrapping.
///
/// # Arguments
/// - `maturities`: ordered maturity times in years (strictly increasing)
/// - `par_rates`: par coupon rates (decimal, e.g. 0.05 for 5%)
/// - `frequency`: coupon payments per year (1 = annual, 2 = semi-annual)
///
/// # Returns
/// `(maturities, spot_rates)` – the bootstrapped spot rate curve using
///   continuous compounding.
///
/// # Example
/// ```
/// use scirs2_integrate::specialized::finance::term_structure::bootstrap_yield_curve;
///
/// let maturities = vec![1.0, 2.0, 3.0, 5.0];
/// let par_rates   = vec![0.02, 0.025, 0.03, 0.035];
/// let (mats, spots) = bootstrap_yield_curve(&maturities, &par_rates, 2).unwrap();
/// assert_eq!(mats.len(), spots.len());
/// ```
pub fn bootstrap_yield_curve(
    maturities: &[f64],
    par_rates: &[f64],
    frequency: usize,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    if maturities.is_empty() {
        return Err(IntegrateError::ValueError(
            "Maturities cannot be empty".to_string(),
        ));
    }
    if maturities.len() != par_rates.len() {
        return Err(IntegrateError::DimensionMismatch(
            "maturities and par_rates must have the same length".to_string(),
        ));
    }
    if frequency == 0 {
        return Err(IntegrateError::ValueError(
            "Frequency must be at least 1".to_string(),
        ));
    }
    for i in 1..maturities.len() {
        if maturities[i] <= maturities[i - 1] {
            return Err(IntegrateError::ValueError(
                "Maturities must be strictly increasing".to_string(),
            ));
        }
        if maturities[i] <= 0.0 {
            return Err(IntegrateError::ValueError(
                "All maturities must be positive".to_string(),
            ));
        }
    }
    if maturities[0] <= 0.0 {
        return Err(IntegrateError::ValueError(
            "All maturities must be positive".to_string(),
        ));
    }

    let freq_f = frequency as f64;
    let dt = 1.0 / freq_f; // time between coupon payments
    let n = maturities.len();

    let mut spot_rates: Vec<f64> = Vec::with_capacity(n);

    for idx in 0..n {
        let t_n = maturities[idx];
        let par = par_rates[idx];
        let coupon = par / freq_f; // coupon per period

        // Build coupon schedule: n_periods coupon payments
        let n_periods = (t_n * freq_f).round() as usize;
        if n_periods == 0 {
            return Err(IntegrateError::ValueError(format!(
                "Maturity {} is too short for frequency {}",
                t_n, frequency
            )));
        }

        // Sum of discount factors for intermediate coupon dates using already
        // bootstrapped spot rates (linearly interpolated)
        let mut pv_coupons = 0.0;
        for period in 1..n_periods {
            let t_c = period as f64 * dt;
            let r_c = interpolate_spot_rate(&maturities[..idx], &spot_rates, t_c);
            pv_coupons += coupon * (-r_c * t_c).exp();
        }

        // Solve for the spot rate at t_n from the par condition:
        //   (1 + coupon) * exp(-r_n * t_n) + sum_coupons = 1
        //   exp(-r_n * t_n) = (1 - pv_coupons) / (1 + coupon)
        let final_cash_flow = 1.0 + coupon; // principal + last coupon
        let disc_n = (1.0 - pv_coupons) / final_cash_flow;

        if disc_n <= 0.0 {
            return Err(IntegrateError::ComputationError(format!(
                "Bootstrapping failed at maturity {}: non-positive discount factor",
                t_n
            )));
        }

        let spot_n = -disc_n.ln() / t_n;
        spot_rates.push(spot_n);
    }

    Ok((maturities.to_vec(), spot_rates))
}

/// Linear interpolation (flat extrapolation) of spot rates for a given maturity.
fn interpolate_spot_rate(maturities: &[f64], spot_rates: &[f64], t: f64) -> f64 {
    if maturities.is_empty() || spot_rates.is_empty() {
        return 0.0;
    }
    if t <= maturities[0] {
        return spot_rates[0];
    }
    if t >= *maturities.last().expect("non-empty") {
        return *spot_rates.last().expect("non-empty");
    }
    for i in 1..maturities.len() {
        if t <= maturities[i] {
            let t0 = maturities[i - 1];
            let t1 = maturities[i];
            let r0 = spot_rates[i - 1];
            let r1 = spot_rates[i];
            return r0 + (r1 - r0) * (t - t0) / (t1 - t0);
        }
    }
    *spot_rates.last().expect("non-empty")
}

// ============================================================
// Nelson-Siegel model
// ============================================================

/// Nelson-Siegel parametric yield curve model.
///
/// Yield at maturity τ:
/// y(τ) = β₀ + β₁ · (1 - e^{-τ/λ}) / (τ/λ) + β₂ · [(1 - e^{-τ/λ}) / (τ/λ) - e^{-τ/λ}]
///
/// - β₀: long-run level (level factor)
/// - β₁: short-term slope (slope factor)  
/// - β₂: medium-term hump (curvature factor)
/// - τ (tau): decay factor controlling factor loading speed
#[derive(Debug, Clone)]
pub struct NelsonSiegel {
    /// Long-run yield level
    pub beta0: f64,
    /// Slope factor (short-term component)
    pub beta1: f64,
    /// Curvature factor (medium-term hump)
    pub beta2: f64,
    /// Decay parameter (controls loading speed)
    pub tau: f64,
}

impl NelsonSiegel {
    /// Create a new Nelson-Siegel model with given parameters.
    pub fn new(beta0: f64, beta1: f64, beta2: f64, tau: f64) -> Self {
        Self {
            beta0,
            beta1,
            beta2,
            tau,
        }
    }

    /// Compute the yield at a given maturity.
    ///
    /// ```
    /// use scirs2_integrate::specialized::finance::term_structure::NelsonSiegel;
    ///
    /// let ns = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);
    /// let y = ns.yield_at(5.0);
    /// assert!(y > 0.0 && y < 0.1);
    /// ```
    pub fn yield_at(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            // Short-end limit: y(0) = beta0 + beta1
            return self.beta0 + self.beta1;
        }
        let x = maturity / self.tau;
        let ex = (-x).exp();
        let loading1 = if x.abs() < 1e-8 {
            1.0 - x / 2.0 // Taylor expansion for stability
        } else {
            (1.0 - ex) / x
        };
        let loading2 = loading1 - ex;
        self.beta0 + self.beta1 * loading1 + self.beta2 * loading2
    }

    /// Fit the Nelson-Siegel model to observed yields via nonlinear least squares.
    ///
    /// Uses a grid search over tau followed by linear regression for the beta
    /// parameters given each tau.
    ///
    /// # Arguments
    /// - `maturities`: observed maturity times in years
    /// - `yields`: observed yields (continuously compounded)
    pub fn fit(maturities: &[f64], yields: &[f64]) -> IntegrateResult<Self> {
        if maturities.len() < 3 {
            return Err(IntegrateError::ValueError(
                "At least 3 data points required to fit Nelson-Siegel".to_string(),
            ));
        }
        if maturities.len() != yields.len() {
            return Err(IntegrateError::DimensionMismatch(
                "maturities and yields must have equal length".to_string(),
            ));
        }

        let n = maturities.len();
        let tau_grid: Vec<f64> = (1..=30).map(|i| i as f64 * 0.5).collect();

        let mut best_sse = f64::INFINITY;
        let mut best = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);

        for &tau in &tau_grid {
            // For fixed tau, compute factor loadings for each maturity
            let mut x_mat: Vec<[f64; 3]> = Vec::with_capacity(n);
            for &t in maturities.iter() {
                let loading = ns_loading(t, tau);
                x_mat.push([1.0, loading.0, loading.1]);
            }

            // Linear OLS: betas = (X^T X)^{-1} X^T y
            let betas = ns_ols(&x_mat, yields);
            let candidate = NelsonSiegel::new(betas[0], betas[1], betas[2], tau);

            let sse: f64 = maturities
                .iter()
                .zip(yields.iter())
                .map(|(&t, &y)| {
                    let y_hat = candidate.yield_at(t);
                    (y - y_hat).powi(2)
                })
                .sum();

            if sse < best_sse {
                best_sse = sse;
                best = candidate;
            }
        }

        Ok(best)
    }

    /// Compute forward rate at maturity τ.
    pub fn forward_rate(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return self.beta0 + self.beta1;
        }
        let x = maturity / self.tau;
        let ex = (-x).exp();
        self.beta0 + self.beta1 * ex + self.beta2 * x * ex
    }

    /// Compute discount factor implied by the Nelson-Siegel curve.
    pub fn discount_factor_at(&self, maturity: f64) -> f64 {
        let y = self.yield_at(maturity);
        (-y * maturity).exp()
    }
}

/// Factor loadings for the second and third NS factors at maturity t and decay tau.
fn ns_loading(t: f64, tau: f64) -> (f64, f64) {
    if t <= 0.0 {
        return (1.0, 0.0);
    }
    let x = t / tau;
    let ex = (-x).exp();
    let l1 = if x.abs() < 1e-8 {
        1.0 - x / 2.0
    } else {
        (1.0 - ex) / x
    };
    let l2 = l1 - ex;
    (l1, l2)
}

/// 3-parameter OLS fit given design matrix X (n×3) and response y.
fn ns_ols(x_mat: &[[f64; 3]], y: &[f64]) -> [f64; 3] {
    let n = x_mat.len();
    let p = 3usize;
    if n < p {
        return [0.0; 3];
    }

    let mut xtx = [[0.0f64; 3]; 3];
    let mut xty = [0.0f64; 3];

    for (row, &yi) in x_mat.iter().zip(y.iter()) {
        for j in 0..p {
            xty[j] += row[j] * yi;
            for k in 0..p {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }

    // Gaussian elimination for 3×3 system
    let mut aug = [[0.0f64; 4]; 3];
    for i in 0..p {
        for j in 0..p {
            aug[i][j] = xtx[i][j];
        }
        aug[i][p] = xty[i];
    }

    for col in 0..p {
        // Partial pivoting
        let pivot = (col..p).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .expect("NaN")
        });
        if let Some(pr) = pivot {
            aug.swap(col, pr);
        }
        let pv = aug[col][col];
        if pv.abs() < 1e-14 {
            continue;
        }
        let inv_pv = 1.0 / pv;
        for k in col..=p {
            aug[col][k] *= inv_pv;
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let m = aug[row][col];
            for k in col..=p {
                let sub = m * aug[col][k];
                aug[row][k] -= sub;
            }
        }
    }

    [aug[0][p], aug[1][p], aug[2][p]]
}

// ============================================================
// Vasicek short-rate model
// ============================================================

/// Compute the zero-coupon bond price under the Vasicek interest rate model.
///
/// Vasicek dynamics: dr = κ(θ - r) dt + σ dW
///
/// Analytical bond price: P(0, T) = exp(A(T) - B(T) * r0) where:
/// - B(T) = (1 - exp(-κT)) / κ
/// - A(T) = (B(T) - T)(κ²θ - σ²/2) / κ² - σ²B(T)² / (4κ)
///
/// # Arguments
/// - `r0`: current short rate
/// - `kappa`: mean-reversion speed
/// - `theta`: long-term mean of the short rate
/// - `sigma`: volatility of the short rate
/// - `maturity`: bond maturity in years
///
/// # Example
/// ```
/// use scirs2_integrate::specialized::finance::term_structure::vasicek_bond_price;
///
/// let price = vasicek_bond_price(0.05, 0.5, 0.04, 0.01, 5.0);
/// assert!(price > 0.0 && price < 1.0);
/// ```
pub fn vasicek_bond_price(r0: f64, kappa: f64, theta: f64, sigma: f64, maturity: f64) -> f64 {
    if maturity <= 0.0 {
        return 1.0;
    }
    if kappa.abs() < 1e-12 {
        // Degenerate case (no mean reversion)
        let a = sigma * sigma * maturity * maturity * maturity / 6.0;
        return (a - r0 * maturity).exp();
    }
    let b = (1.0 - (-kappa * maturity).exp()) / kappa;
    let term1 = (b - maturity) * (kappa * kappa * theta - 0.5 * sigma * sigma);
    let term2 = 0.25 * sigma * sigma * b * b * kappa;
    let a = (term1 - term2) / (kappa * kappa);
    (a - b * r0).exp()
}

/// Vasicek spot rate at maturity T implied by the bond price.
pub fn vasicek_spot_rate(r0: f64, kappa: f64, theta: f64, sigma: f64, maturity: f64) -> f64 {
    let price = vasicek_bond_price(r0, kappa, theta, sigma, maturity);
    if maturity <= 0.0 || price <= 0.0 {
        return r0;
    }
    -price.ln() / maturity
}

/// Simulate a Vasicek short-rate path using Euler-Maruyama discretisation.
///
/// # Arguments
/// - `r0`: initial short rate
/// - `kappa`, `theta`, `sigma`: Vasicek parameters
/// - `t`: total time horizon
/// - `n_steps`: number of discretisation steps
/// - `seed`: random seed
///
/// Returns an `Array1<f64>` of length `n_steps + 1` with the simulated path.
///
/// # Example
/// ```
/// use scirs2_integrate::specialized::finance::term_structure::vasicek_simulate;
///
/// let path = vasicek_simulate(0.05, 0.5, 0.04, 0.01, 5.0, 100, 42);
/// assert_eq!(path.len(), 101);
/// assert!((path[0] - 0.05).abs() < 1e-10);
/// ```
pub fn vasicek_simulate(
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    seed: u64,
) -> Array1<f64> {
    let dt = t / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = seeded_rng(seed);

    let mut path = Array1::zeros(n_steps + 1);
    path[0] = r0;
    let mut r = r0;

    for i in 0..n_steps {
        let z: f64 = rng.sample(StandardNormal);
        r += kappa * (theta - r) * dt + sigma * sqrt_dt * z;
        path[i + 1] = r;
    }

    path
}

/// Simulate multiple Vasicek rate paths.
///
/// Returns a matrix of shape `(n_paths, n_steps + 1)`.
pub fn vasicek_simulate_paths(
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Array2<f64> {
    let dt = t / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = seeded_rng(seed);

    let mut paths = Array2::zeros((n_paths, n_steps + 1));
    for i in 0..n_paths {
        paths[[i, 0]] = r0;
        let mut r = r0;
        for j in 0..n_steps {
            let z: f64 = rng.sample(StandardNormal);
            r += kappa * (theta - r) * dt + sigma * sqrt_dt * z;
            paths[[i, j + 1]] = r;
        }
    }
    paths
}

// ============================================================
// Yield curve analysis utilities
// ============================================================

/// Compute the continuously compounded forward rate between two maturities.
///
/// F(T1, T2) = (r(T2) * T2 - r(T1) * T1) / (T2 - T1)
pub fn forward_rate(t1: f64, t2: f64, r1: f64, r2: f64) -> IntegrateResult<f64> {
    if t2 <= t1 {
        return Err(IntegrateError::ValueError(
            "t2 must be strictly greater than t1".to_string(),
        ));
    }
    Ok((r2 * t2 - r1 * t1) / (t2 - t1))
}

/// Compute forward rates from a set of spot rates.
pub fn spot_to_forward(maturities: &[f64], spot_rates: &[f64]) -> IntegrateResult<Vec<f64>> {
    if maturities.len() != spot_rates.len() {
        return Err(IntegrateError::DimensionMismatch(
            "maturities and spot_rates must have equal length".to_string(),
        ));
    }
    if maturities.len() < 2 {
        return Err(IntegrateError::ValueError(
            "At least 2 points required to compute forward rates".to_string(),
        ));
    }
    let mut fwd = Vec::with_capacity(maturities.len() - 1);
    for i in 1..maturities.len() {
        let f = forward_rate(
            maturities[i - 1],
            maturities[i],
            spot_rates[i - 1],
            spot_rates[i],
        )?;
        fwd.push(f);
    }
    Ok(fwd)
}

/// Duration of a coupon bond (Macaulay duration in years).
pub fn macaulay_duration(
    maturities: &[f64],
    cash_flows: &[f64],
    spot_rates: &[f64],
) -> IntegrateResult<f64> {
    if maturities.len() != cash_flows.len() || maturities.len() != spot_rates.len() {
        return Err(IntegrateError::DimensionMismatch(
            "All arrays must have equal length".to_string(),
        ));
    }

    let mut pv_total = 0.0;
    let mut pv_weighted = 0.0;

    for ((&t, &cf), &r) in maturities
        .iter()
        .zip(cash_flows.iter())
        .zip(spot_rates.iter())
    {
        let pv = cf * (-r * t).exp();
        pv_total += pv;
        pv_weighted += t * pv;
    }

    if pv_total.abs() < 1e-14 {
        return Err(IntegrateError::ComputationError(
            "Total present value is zero; cannot compute duration".to_string(),
        ));
    }

    Ok(pv_weighted / pv_total)
}

/// Convexity of a bond (second derivative of price w.r.t. yield, normalised).
pub fn convexity(
    maturities: &[f64],
    cash_flows: &[f64],
    spot_rates: &[f64],
) -> IntegrateResult<f64> {
    if maturities.len() != cash_flows.len() || maturities.len() != spot_rates.len() {
        return Err(IntegrateError::DimensionMismatch(
            "All arrays must have equal length".to_string(),
        ));
    }

    let mut pv_total = 0.0;
    let mut conv_num = 0.0;

    for ((&t, &cf), &r) in maturities
        .iter()
        .zip(cash_flows.iter())
        .zip(spot_rates.iter())
    {
        let pv = cf * (-r * t).exp();
        pv_total += pv;
        conv_num += t * t * pv;
    }

    if pv_total.abs() < 1e-14 {
        return Err(IntegrateError::ComputationError(
            "Total present value is zero; cannot compute convexity".to_string(),
        ));
    }

    Ok(conv_num / pv_total)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- Discount factor ---

    #[test]
    fn test_discount_continuous() {
        let df = discount_factor(0.05, 1.0, Compounding::Continuous);
        assert_relative_eq!(df, (-0.05f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_discount_annual() {
        let df = discount_factor(0.05, 1.0, Compounding::Annual);
        assert_relative_eq!(df, 1.0 / 1.05, epsilon = 1e-10);
    }

    #[test]
    fn test_discount_semiannual() {
        let df = discount_factor(0.05, 1.0, Compounding::SemiAnnual);
        assert_relative_eq!(df, 1.0 / (1.025f64 * 1.025), epsilon = 1e-10);
    }

    #[test]
    fn test_discount_zero_maturity() {
        let df = discount_factor(0.05, 0.0, Compounding::Continuous);
        assert_relative_eq!(df, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_discount_decreases_with_maturity() {
        let df1 = discount_factor(0.05, 1.0, Compounding::Continuous);
        let df2 = discount_factor(0.05, 5.0, Compounding::Continuous);
        assert!(df2 < df1);
    }

    // --- Rate conversion ---

    #[test]
    fn test_convert_continuous_to_annual_roundtrip() {
        let rate_cont = 0.05;
        let annual = convert_rate(rate_cont, Compounding::Continuous, Compounding::Annual, 1.0)
            .expect("conversion failed");
        let back = convert_rate(annual, Compounding::Annual, Compounding::Continuous, 1.0)
            .expect("conversion failed");
        assert_relative_eq!(back, rate_cont, epsilon = 1e-10);
    }

    // --- Bootstrapping ---

    #[test]
    fn test_bootstrap_single_maturity() {
        // One-year bond with 5% annual coupon at par => spot rate = log(1.05) ≈ 4.88%
        let mats = vec![1.0];
        let par = vec![0.05];
        let (out_mats, spots) = bootstrap_yield_curve(&mats, &par, 1).expect("bootstrap failed");
        assert_eq!(out_mats.len(), 1);
        // For annual coupon at par: spot rate such that (1+c)*e^{-r} = 1
        // => r = ln(1.05) ≈ 0.04879
        let expected = (1.05f64).ln();
        assert_relative_eq!(spots[0], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_bootstrap_multiple_maturities_increasing() {
        let mats = vec![1.0, 2.0, 3.0, 5.0];
        let par = vec![0.02, 0.025, 0.03, 0.035];
        let (out_mats, spots) = bootstrap_yield_curve(&mats, &par, 2).expect("bootstrap failed");
        assert_eq!(out_mats.len(), spots.len());
        assert_eq!(spots.len(), 4);
        // All spot rates should be positive for positive par rates
        for &r in &spots {
            assert!(r > 0.0, "Spot rate should be positive: {}", r);
        }
    }

    #[test]
    fn test_bootstrap_empty_maturities_error() {
        let result = bootstrap_yield_curve(&[], &[], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_length_mismatch_error() {
        let result = bootstrap_yield_curve(&[1.0, 2.0], &[0.05], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_zero_frequency_error() {
        let result = bootstrap_yield_curve(&[1.0], &[0.05], 0);
        assert!(result.is_err());
    }

    // --- Nelson-Siegel ---

    #[test]
    fn test_ns_yield_at_positive() {
        let ns = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);
        for &t in &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let y = ns.yield_at(t);
            assert!(y.is_finite(), "NS yield not finite at t={}", t);
        }
    }

    #[test]
    fn test_ns_zero_maturity_limit() {
        // As t -> 0, yield -> beta0 + beta1
        let ns = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);
        let y_zero = ns.yield_at(0.0);
        assert_relative_eq!(y_zero, ns.beta0 + ns.beta1, epsilon = 1e-10);
    }

    #[test]
    fn test_ns_long_end_approaches_beta0() {
        // For very large t, yield -> beta0
        let ns = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);
        let y_long = ns.yield_at(200.0);
        assert!(
            (y_long - ns.beta0).abs() < 1e-4,
            "Long-end yield {:.6} should approach beta0={:.6}",
            y_long,
            ns.beta0
        );
    }

    #[test]
    fn test_ns_fit_recovers_parameters() {
        // Generate synthetic yields from known NS params and verify fit recovers them
        let true_ns = NelsonSiegel::new(0.05, -0.03, 0.015, 1.5);
        let mats: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0];
        let yields: Vec<f64> = mats.iter().map(|&t| true_ns.yield_at(t)).collect();

        let fitted = NelsonSiegel::fit(&mats, &yields).expect("NS fit failed");

        // Check that fitted yields are close to original
        for (&t, &y_true) in mats.iter().zip(yields.iter()) {
            let y_fit = fitted.yield_at(t);
            assert!(
                (y_fit - y_true).abs() < 5e-4,
                "NS fit error at t={}: fit={:.6} true={:.6}",
                t,
                y_fit,
                y_true
            );
        }
    }

    #[test]
    fn test_ns_fit_too_few_points_error() {
        let result = NelsonSiegel::fit(&[1.0, 2.0], &[0.05, 0.06]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ns_discount_factor_less_than_one() {
        let ns = NelsonSiegel::new(0.04, -0.01, 0.005, 2.0);
        for &t in &[1.0, 5.0, 10.0] {
            let df = ns.discount_factor_at(t);
            assert!(
                df > 0.0 && df < 1.0,
                "Discount factor out of range at t={}: {}",
                t,
                df
            );
        }
    }

    #[test]
    fn test_ns_forward_rate_finite() {
        let ns = NelsonSiegel::new(0.05, -0.02, 0.01, 1.5);
        for &t in &[0.5, 1.0, 5.0, 10.0] {
            let f = ns.forward_rate(t);
            assert!(f.is_finite(), "Forward rate not finite at t={}", t);
        }
    }

    // --- Vasicek ---

    #[test]
    fn test_vasicek_bond_price_at_zero() {
        let p = vasicek_bond_price(0.05, 0.5, 0.04, 0.01, 0.0);
        assert_relative_eq!(p, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vasicek_bond_price_positive() {
        let p = vasicek_bond_price(0.05, 0.5, 0.04, 0.01, 5.0);
        assert!(p > 0.0 && p < 1.0, "Bond price should be in (0, 1): {}", p);
    }

    #[test]
    fn test_vasicek_bond_price_decreases_with_rate() {
        let p_low = vasicek_bond_price(0.02, 0.5, 0.04, 0.01, 5.0);
        let p_high = vasicek_bond_price(0.08, 0.5, 0.04, 0.01, 5.0);
        assert!(p_low > p_high, "Higher rate should give lower bond price");
    }

    #[test]
    fn test_vasicek_simulate_length() {
        let path = vasicek_simulate(0.05, 0.5, 0.04, 0.01, 5.0, 100, 42);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_vasicek_simulate_initial_value() {
        let r0 = 0.05;
        let path = vasicek_simulate(r0, 0.5, 0.04, 0.01, 5.0, 100, 42);
        assert_relative_eq!(path[0], r0, epsilon = 1e-10);
    }

    #[test]
    fn test_vasicek_simulate_mean_reversion() {
        // Over long horizon, the mean should converge toward theta
        let theta = 0.04;
        let path = vasicek_simulate(0.10, 2.0, theta, 0.01, 20.0, 1000, 42);
        let tail: Vec<f64> = path.iter().skip(800).copied().collect();
        let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!(
            (tail_mean - theta).abs() < 0.02,
            "Vasicek tail mean {:.4} should be near theta {:.4}",
            tail_mean,
            theta
        );
    }

    // --- Utility functions ---

    #[test]
    fn test_forward_rate_formula() {
        // If spot rates are flat at 5%, forward rate should also be 5%
        let f = forward_rate(1.0, 2.0, 0.05, 0.05).expect("forward rate failed");
        assert_relative_eq!(f, 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_rate_invalid_order() {
        let result = forward_rate(2.0, 1.0, 0.04, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_spot_to_forward() {
        let mats = vec![1.0, 2.0, 3.0];
        let spots = vec![0.03, 0.035, 0.04];
        let fwd = spot_to_forward(&mats, &spots).expect("spot_to_forward failed");
        assert_eq!(fwd.len(), 2);
        // All forwards should be positive for increasing spot rates
        for &f in &fwd {
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_macaulay_duration_zero_coupon() {
        // Zero-coupon bond: duration = maturity
        let mats = vec![5.0];
        let cfs = vec![1000.0];
        let rates = vec![0.05];
        let dur = macaulay_duration(&mats, &cfs, &rates).expect("duration failed");
        assert_relative_eq!(dur, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convexity_positive() {
        let mats = vec![1.0, 2.0, 5.0];
        let cfs = vec![50.0, 50.0, 1050.0];
        let rates = vec![0.04, 0.045, 0.05];
        let conv = convexity(&mats, &cfs, &rates).expect("convexity failed");
        assert!(conv > 0.0);
    }
}
