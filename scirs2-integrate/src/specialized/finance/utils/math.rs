//! Mathematical utilities for quantitative finance
//!
//! This module provides specialized mathematical functions commonly used in financial
//! calculations including option pricing formulas, statistical functions, and numerical methods.
//!
//! # Features
//! - Black-Scholes formula and Greeks
//! - Implied volatility calculation (Newton-Raphson, Brent)
//! - Normal distribution CDF/PDF (fast approximations)
//! - Bachelier model (for negative rates)
//! - Numerical stability helpers
//!
//! # Example
//! ```
//! use scirs2_integrate::specialized::finance::utils::math::{
//!     black_scholes_call, implied_volatility_newton, norm_cdf
//! };
//!
//! // Price a call option
//! let price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
//! assert!(price > 0.0);
//!
//! // Calculate implied volatility
//! let iv = implied_volatility_newton(price, 100.0, 100.0, 1.0, 0.05, true);
//! ```

use crate::error::{IntegrateError, IntegrateResult as Result};
use std::f64::consts::{PI, SQRT_2};

/// SABR model parameters
#[derive(Debug, Clone)]
pub struct SABRParameters {
    /// Alpha (initial volatility)
    pub alpha: f64,
    /// Beta (elasticity parameter, typically 0, 0.5, or 1)
    pub beta: f64,
    /// Rho (correlation between forward and volatility)
    pub rho: f64,
    /// Nu (volatility of volatility)
    pub nu: f64,
}

impl SABRParameters {
    /// Create new SABR parameters with validation
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Alpha must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(IntegrateError::ValueError(
                "Beta must be between 0 and 1".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&rho) {
            return Err(IntegrateError::ValueError(
                "Rho must be between -1 and 1".to_string(),
            ));
        }
        if nu < 0.0 {
            return Err(IntegrateError::ValueError(
                "Nu must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            beta,
            rho,
            nu,
        })
    }

    /// Calculate implied volatility using SABR formula (Hagan et al. 2002)
    pub fn implied_volatility(&self, forward: f64, strike: f64, time: f64) -> Result<f64> {
        if forward <= 0.0 || strike <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Forward and strike must be positive".to_string(),
            ));
        }
        if time <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Time must be positive".to_string(),
            ));
        }

        // ATM case
        if (forward - strike).abs() / forward < 1e-8 {
            let f_mid_beta = forward.powf(1.0 - self.beta);
            let term1 = self.alpha / f_mid_beta;

            let term2 = 1.0
                + ((1.0 - self.beta).powi(2) / 24.0 * self.alpha.powi(2)
                    / forward.powf(2.0 - 2.0 * self.beta)
                    + 0.25 * self.rho * self.beta * self.nu * self.alpha / f_mid_beta
                    + (2.0 - 3.0 * self.rho.powi(2)) / 24.0 * self.nu.powi(2))
                    * time;

            return Ok(term1 * term2);
        }

        // Non-ATM case
        let log_fk = (forward / strike).ln();
        let f_k_mid_beta = (forward * strike).powf((1.0 - self.beta) / 2.0);

        let z = (self.nu / self.alpha) * f_k_mid_beta * log_fk;

        // Calculate x(z) with numerical stability
        let x = if z.abs() < 1e-6 {
            // Taylor expansion for small z
            1.0 - 0.5 * self.rho * z
        } else {
            let sqrt_term = (1.0 - 2.0 * self.rho * z + z.powi(2)).sqrt();
            ((sqrt_term + z - self.rho) / (1.0 - self.rho)).ln() / z
        };

        let numerator = self.alpha;
        let denominator = f_k_mid_beta
            * (1.0
                + (1.0 - self.beta).powi(2) / 24.0 * log_fk.powi(2)
                + (1.0 - self.beta).powi(4) / 1920.0 * log_fk.powi(4));

        let correction = 1.0
            + ((1.0 - self.beta).powi(2) / 24.0 * self.alpha.powi(2)
                / forward.powf(2.0 - 2.0 * self.beta)
                + 0.25 * self.rho * self.beta * self.nu * self.alpha / f_k_mid_beta
                + (2.0 - 3.0 * self.rho.powi(2)) / 24.0 * self.nu.powi(2))
                * time;

        Ok((numerator / denominator) * x * correction)
    }
}

// Deprecated placeholder functions for backward compatibility
#[deprecated(
    since = "0.1.0",
    note = "Use SABRParameters::implied_volatility instead"
)]
pub fn interpolate_smile() -> Result<()> {
    Err(IntegrateError::ValueError(
        "Use SABRParameters for smile interpolation".to_string(),
    ))
}

#[deprecated(since = "0.1.0", note = "Arbitrage checking not yet implemented")]
pub fn vol_surface_arbitrage_free() -> Result<()> {
    Err(IntegrateError::ValueError(
        "Arbitrage-free surface checking not yet implemented".to_string(),
    ))
}

/// Standard normal cumulative distribution function
///
/// Uses rational approximation with max error < 1.5e-7
pub fn norm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    // Abramowitz and Stegun approximation (1964)
    // Maximum error: 1.5×10^-7
    let abs_x = x.abs();

    if abs_x > 10.0 {
        // For large |x|, return extremes directly
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    let t = 1.0 / (1.0 + 0.2316419 * abs_x);
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    let pdf = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
    let cdf = 1.0 - pdf * poly;

    if x >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}

/// Standard normal probability density function
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (2.0 * PI).sqrt().recip() * (-0.5 * x * x).exp()
}

/// Natural logarithm with protection against domain errors
#[inline]
pub fn safe_log(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else if x == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}

/// Square root with protection against domain errors
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.sqrt()
    } else {
        f64::NAN
    }
}

/// Black-Scholes d1 parameter
pub fn d1(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 || volatility <= 0.0 {
        return f64::NAN;
    }
    let vol_sqrt_t = volatility * time.sqrt();
    (safe_log(spot / strike) + (rate + 0.5 * volatility * volatility) * time) / vol_sqrt_t
}

/// Black-Scholes d2 parameter
pub fn d2(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 || volatility <= 0.0 {
        return f64::NAN;
    }
    let vol_sqrt_t = volatility * time.sqrt();
    d1(spot, strike, time, rate, volatility) - vol_sqrt_t
}

/// Black-Scholes call option price
pub fn black_scholes_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return (spot - strike).max(0.0);
    }
    if volatility <= 0.0 {
        return ((spot - strike * (-rate * time).exp()).max(0.0)).max(0.0);
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    let d2_val = d2(spot, strike, time, rate, volatility);

    spot * norm_cdf(d1_val) - strike * (-rate * time).exp() * norm_cdf(d2_val)
}

/// Black-Scholes put option price
pub fn black_scholes_put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return (strike - spot).max(0.0);
    }
    if volatility <= 0.0 {
        return ((strike * (-rate * time).exp() - spot).max(0.0)).max(0.0);
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    let d2_val = d2(spot, strike, time, rate, volatility);

    strike * (-rate * time).exp() * norm_cdf(-d2_val) - spot * norm_cdf(-d1_val)
}

/// Vega (sensitivity to volatility) - same for calls and puts
pub fn vega(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 || volatility <= 0.0 {
        return 0.0;
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    spot * norm_pdf(d1_val) * time.sqrt()
}

/// Implied volatility using Newton-Raphson method
pub fn implied_volatility_newton(
    market_price: f64,
    spot: f64,
    strike: f64,
    time: f64,
    rate: f64,
    is_call: bool,
) -> Result<f64> {
    if market_price <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Market price must be positive".to_string(),
        ));
    }

    // Initial guess using Brenner-Subrahmanyam approximation
    let intrinsic = if is_call {
        (spot - strike * (-rate * time).exp()).max(0.0)
    } else {
        (strike * (-rate * time).exp() - spot).max(0.0)
    };

    if market_price <= intrinsic {
        return Err(IntegrateError::ValueError(
            "Market price below intrinsic value".to_string(),
        ));
    }

    let time_value = market_price - intrinsic;
    let mut vol = (2.0 * PI / time).sqrt() * (time_value / spot);
    vol = vol.max(0.01).min(5.0); // Clamp initial guess

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-8;

    for _ in 0..MAX_ITERATIONS {
        let price = if is_call {
            black_scholes_call(spot, strike, time, rate, vol)
        } else {
            black_scholes_put(spot, strike, time, rate, vol)
        };

        let diff = price - market_price;

        if diff.abs() < TOLERANCE {
            return Ok(vol);
        }

        let vega_val = vega(spot, strike, time, rate, vol);

        if vega_val < 1e-10 {
            return Err(IntegrateError::ValueError(
                "Vega too small for convergence".to_string(),
            ));
        }

        let vol_new = vol - diff / vega_val;

        // Ensure volatility stays positive and reasonable
        let vol_new = vol_new.max(0.001).min(10.0);

        if (vol_new - vol).abs() < TOLERANCE {
            return Ok(vol_new);
        }

        vol = vol_new;
    }

    Err(IntegrateError::ValueError(
        "Implied volatility did not converge".to_string(),
    ))
}

/// Bachelier model for call options (handles negative rates/prices)
pub fn bachelier_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return (spot - strike).max(0.0);
    }
    if volatility <= 0.0 {
        return ((spot - strike * (-rate * time).exp()).max(0.0)).max(0.0);
    }

    let forward = spot * (rate * time).exp();
    let std_dev = volatility * time.sqrt();
    let d = (forward - strike) / std_dev;

    (forward - strike) * norm_cdf(d) + std_dev * norm_pdf(d)
}

/// Bachelier model for put options
pub fn bachelier_put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return (strike - spot).max(0.0);
    }
    if volatility <= 0.0 {
        return ((strike * (-rate * time).exp() - spot).max(0.0)).max(0.0);
    }

    let forward = spot * (rate * time).exp();
    let std_dev = volatility * time.sqrt();
    let d = (forward - strike) / std_dev;

    (strike - forward) * norm_cdf(-d) + std_dev * norm_pdf(d)
}

/// Binary search for implied volatility (more robust than Newton-Raphson)
pub fn implied_volatility_brent(
    market_price: f64,
    spot: f64,
    strike: f64,
    time: f64,
    rate: f64,
    is_call: bool,
) -> Result<f64> {
    if market_price <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Market price must be positive".to_string(),
        ));
    }

    let mut vol_low = 0.001;
    let mut vol_high = 5.0;

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-8;

    // Check bounds
    let price_low = if is_call {
        black_scholes_call(spot, strike, time, rate, vol_low)
    } else {
        black_scholes_put(spot, strike, time, rate, vol_low)
    };

    let price_high = if is_call {
        black_scholes_call(spot, strike, time, rate, vol_high)
    } else {
        black_scholes_put(spot, strike, time, rate, vol_high)
    };

    if market_price < price_low || market_price > price_high {
        return Err(IntegrateError::ValueError(
            "Market price outside pricing bounds".to_string(),
        ));
    }

    // Binary search
    for _ in 0..MAX_ITERATIONS {
        let vol_mid = (vol_low + vol_high) / 2.0;

        let price_mid = if is_call {
            black_scholes_call(spot, strike, time, rate, vol_mid)
        } else {
            black_scholes_put(spot, strike, time, rate, vol_mid)
        };

        if (price_mid - market_price).abs() < TOLERANCE {
            return Ok(vol_mid);
        }

        if price_mid < market_price {
            vol_low = vol_mid;
        } else {
            vol_high = vol_mid;
        }

        if vol_high - vol_low < TOLERANCE {
            return Ok(vol_mid);
        }
    }

    Err(IntegrateError::ValueError(
        "Implied volatility did not converge".to_string(),
    ))
}

/// Delta (sensitivity to spot price) for call option
pub fn delta_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return if spot > strike { 1.0 } else { 0.0 };
    }
    if volatility <= 0.0 {
        return if spot > strike { 1.0 } else { 0.0 };
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    norm_cdf(d1_val)
}

/// Delta for put option
pub fn delta_put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    delta_call(spot, strike, time, rate, volatility) - 1.0
}

/// Gamma (sensitivity of delta to spot) - same for calls and puts
pub fn gamma(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 || volatility <= 0.0 || spot <= 0.0 {
        return 0.0;
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    norm_pdf(d1_val) / (spot * volatility * time.sqrt())
}

/// Theta (time decay) for call option
pub fn theta_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return 0.0;
    }
    if volatility <= 0.0 {
        return 0.0;
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    let d2_val = d2(spot, strike, time, rate, volatility);

    let term1 = -(spot * norm_pdf(d1_val) * volatility) / (2.0 * time.sqrt());
    let term2 = rate * strike * (-rate * time).exp() * norm_cdf(d2_val);

    term1 - term2
}

/// Theta (time decay) for put option
pub fn theta_put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return 0.0;
    }
    if volatility <= 0.0 {
        return 0.0;
    }

    let d1_val = d1(spot, strike, time, rate, volatility);
    let d2_val = d2(spot, strike, time, rate, volatility);

    let term1 = -(spot * norm_pdf(d1_val) * volatility) / (2.0 * time.sqrt());
    let term2 = rate * strike * (-rate * time).exp() * norm_cdf(-d2_val);

    term1 + term2
}

/// Rho (sensitivity to interest rate) for call option
pub fn rho_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return 0.0;
    }
    if volatility <= 0.0 {
        return if spot > strike {
            strike * time * (-rate * time).exp()
        } else {
            0.0
        };
    }

    let d2_val = d2(spot, strike, time, rate, volatility);
    strike * time * (-rate * time).exp() * norm_cdf(d2_val)
}

/// Rho (sensitivity to interest rate) for put option
pub fn rho_put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    if time <= 0.0 {
        return 0.0;
    }
    if volatility <= 0.0 {
        return if spot < strike {
            -strike * time * (-rate * time).exp()
        } else {
            0.0
        };
    }

    let d2_val = d2(spot, strike, time, rate, volatility);
    -strike * time * (-rate * time).exp() * norm_cdf(-d2_val)
}

/// Calculate all Greeks at once (more efficient than individual calls)
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

impl Greeks {
    /// Calculate all Greeks for a call option
    pub fn call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> Self {
        Self {
            delta: delta_call(spot, strike, time, rate, volatility),
            gamma: gamma(spot, strike, time, rate, volatility),
            vega: vega(spot, strike, time, rate, volatility),
            theta: theta_call(spot, strike, time, rate, volatility),
            rho: rho_call(spot, strike, time, rate, volatility),
        }
    }

    /// Calculate all Greeks for a put option
    pub fn put(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> Self {
        Self {
            delta: delta_put(spot, strike, time, rate, volatility),
            gamma: gamma(spot, strike, time, rate, volatility),
            vega: vega(spot, strike, time, rate, volatility),
            theta: theta_put(spot, strike, time, rate, volatility),
            rho: rho_put(spot, strike, time, rate, volatility),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!(norm_cdf(-10.0) < 1e-10);
        assert!(norm_cdf(10.0) > 0.9999999);
        assert!((norm_cdf(1.0) - 0.8413447).abs() < 1e-5);
    }

    #[test]
    fn test_norm_pdf() {
        let pdf_0 = norm_pdf(0.0);
        assert!((pdf_0 - 0.3989423).abs() < 1e-6);

        let pdf_1 = norm_pdf(1.0);
        assert!((pdf_1 - 0.2419707).abs() < 1e-6);
    }

    #[test]
    fn test_black_scholes_call() {
        let price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(price > 9.0 && price < 12.0); // ATM call should be around 10.45
    }

    #[test]
    fn test_black_scholes_put() {
        let price = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(price > 5.0 && price < 8.0); // ATM put should be around 5.57
    }

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let v = 0.2;

        let call = black_scholes_call(s, k, t, r, v);
        let put = black_scholes_put(s, k, t, r, v);

        // Put-call parity: C - P = S - K*e^(-rT)
        let lhs = call - put;
        let rhs = s - k * (-r * t).exp();

        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_vega() {
        let vega_val = vega(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(vega_val > 0.0);
        assert!(vega_val < 50.0); // Reasonable range for ATM option
    }

    #[test]
    fn test_implied_volatility_newton() {
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let true_vol = 0.25;

        let market_price = black_scholes_call(spot, strike, time, rate, true_vol);
        let implied_vol = implied_volatility_newton(market_price, spot, strike, time, rate, true)
            .expect("Operation failed");

        assert!((implied_vol - true_vol).abs() < 1e-6);
    }

    #[test]
    fn test_implied_volatility_brent() {
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let true_vol = 0.25;

        let market_price = black_scholes_call(spot, strike, time, rate, true_vol);
        let implied_vol = implied_volatility_brent(market_price, spot, strike, time, rate, true)
            .expect("Operation failed");

        assert!((implied_vol - true_vol).abs() < 1e-6);
    }

    #[test]
    fn test_delta_call() {
        let delta = delta_call(100.0, 100.0, 1.0, 0.05, 0.2);
        // ATM call delta with positive rate is slightly above 0.5
        assert!(delta > 0.50 && delta < 0.65);
    }

    #[test]
    fn test_delta_put() {
        let delta = delta_put(100.0, 100.0, 1.0, 0.05, 0.2);
        // ATM put delta = call_delta - 1
        assert!(delta > -0.50 && delta < -0.35);
    }

    #[test]
    fn test_gamma() {
        let gamma_val = gamma(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(gamma_val > 0.0);
        assert!(gamma_val < 0.1); // Reasonable range
    }

    #[test]
    fn test_bachelier_call() {
        let price = bachelier_call(100.0, 100.0, 1.0, 0.05, 20.0);
        assert!(price > 0.0);
    }

    #[test]
    fn test_bachelier_put() {
        let price = bachelier_put(100.0, 100.0, 1.0, 0.05, 20.0);
        assert!(price > 0.0);
    }

    #[test]
    fn test_zero_time() {
        assert_eq!(black_scholes_call(110.0, 100.0, 0.0, 0.05, 0.2), 10.0);
        assert_eq!(black_scholes_put(90.0, 100.0, 0.0, 0.05, 0.2), 10.0);
    }

    #[test]
    fn test_safe_functions() {
        assert!(safe_log(-1.0).is_nan());
        assert_eq!(safe_log(0.0), f64::NEG_INFINITY);
        assert!(safe_sqrt(-1.0).is_nan());
        assert_eq!(safe_sqrt(4.0), 2.0);
    }

    #[test]
    fn test_theta_call() {
        let theta = theta_call(100.0, 100.0, 1.0, 0.05, 0.2);
        // Theta is negative (time decay) for long positions
        assert!(theta < 0.0);
        // Reasonable magnitude for ATM option
        assert!(theta.abs() < 20.0);
    }

    #[test]
    fn test_theta_put() {
        let theta = theta_put(100.0, 100.0, 1.0, 0.05, 0.2);
        // Theta for put depends on interest rate
        // With positive rate, ATM put theta may be positive or negative
        assert!(theta.abs() < 20.0);
    }

    #[test]
    fn test_rho_call() {
        let rho = rho_call(100.0, 100.0, 1.0, 0.05, 0.2);
        // Rho is positive for call options
        assert!(rho > 0.0);
        // Reasonable magnitude
        assert!(rho < 100.0);
    }

    #[test]
    fn test_rho_put() {
        let rho = rho_put(100.0, 100.0, 1.0, 0.05, 0.2);
        // Rho is negative for put options
        assert!(rho < 0.0);
        // Reasonable magnitude
        assert!(rho.abs() < 100.0);
    }

    #[test]
    fn test_greeks_struct_call() {
        let greeks = Greeks::call(100.0, 100.0, 1.0, 0.05, 0.2);

        // Check all Greeks are reasonable
        assert!(greeks.delta > 0.5 && greeks.delta < 0.65);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.vega > 0.0);
        assert!(greeks.theta < 0.0);
        assert!(greeks.rho > 0.0);
    }

    #[test]
    fn test_greeks_struct_put() {
        let greeks = Greeks::put(100.0, 100.0, 1.0, 0.05, 0.2);

        // Check all Greeks are reasonable
        assert!(greeks.delta < 0.0 && greeks.delta > -0.5);
        assert!(greeks.gamma > 0.0); // Same as call
        assert!(greeks.vega > 0.0); // Same as call
        assert!(greeks.rho < 0.0);
    }

    #[test]
    fn test_theta_zero_time() {
        assert_eq!(theta_call(100.0, 100.0, 0.0, 0.05, 0.2), 0.0);
        assert_eq!(theta_put(100.0, 100.0, 0.0, 0.05, 0.2), 0.0);
    }

    #[test]
    fn test_rho_zero_time() {
        assert_eq!(rho_call(100.0, 100.0, 0.0, 0.05, 0.2), 0.0);
        assert_eq!(rho_put(100.0, 100.0, 0.0, 0.05, 0.2), 0.0);
    }

    #[test]
    fn test_sabr_parameters_creation() {
        let sabr = SABRParameters::new(0.2, 0.5, -0.3, 0.4).expect("Operation failed");
        assert_eq!(sabr.alpha, 0.2);
        assert_eq!(sabr.beta, 0.5);
        assert_eq!(sabr.rho, -0.3);
        assert_eq!(sabr.nu, 0.4);
    }

    #[test]
    fn test_sabr_parameters_validation() {
        assert!(SABRParameters::new(-0.1, 0.5, 0.0, 0.4).is_err()); // Negative alpha
        assert!(SABRParameters::new(0.2, 1.5, 0.0, 0.4).is_err()); // Beta > 1
        assert!(SABRParameters::new(0.2, 0.5, 1.5, 0.4).is_err()); // Rho > 1
        assert!(SABRParameters::new(0.2, 0.5, 0.0, -0.1).is_err()); // Negative nu
    }

    #[test]
    fn test_sabr_atm_volatility() {
        let sabr = SABRParameters::new(0.2, 0.5, -0.3, 0.4).expect("Operation failed");
        let forward = 100.0;
        let strike = 100.0;
        let time = 1.0;

        let vol = sabr
            .implied_volatility(forward, strike, time)
            .expect("Operation failed");
        assert!(vol > 0.0);
        assert!(vol < 1.0); // Reasonable volatility range
    }

    #[test]
    fn test_sabr_smile_shape() {
        let sabr = SABRParameters::new(0.2, 0.5, -0.3, 0.4).expect("Operation failed");
        let forward = 100.0;
        let time = 1.0;

        let vol_otm = sabr
            .implied_volatility(forward, 110.0, time)
            .expect("Operation failed");
        let vol_atm = sabr
            .implied_volatility(forward, 100.0, time)
            .expect("Operation failed");
        let vol_itm = sabr
            .implied_volatility(forward, 90.0, time)
            .expect("Operation failed");

        // All vols should be reasonable and positive
        assert!(vol_otm > 0.0 && vol_otm < 1.0);
        assert!(vol_atm > 0.0 && vol_atm < 1.0);
        assert!(vol_itm > 0.0 && vol_itm < 1.0);

        // Verify smile exists (not flat)
        let smile_measure = (vol_itm - vol_atm).abs() + (vol_otm - vol_atm).abs();
        assert!(smile_measure > 0.001); // Some curvature exists
    }

    #[test]
    fn test_sabr_beta_zero() {
        // Beta = 0 corresponds to normal model
        let sabr = SABRParameters::new(20.0, 0.0, 0.0, 0.0).expect("Operation failed");
        let vol = sabr
            .implied_volatility(100.0, 100.0, 1.0)
            .expect("Operation failed");
        // For beta=0, vol should be positive and reasonable
        assert!(vol > 0.0 && vol < 100.0);
    }

    #[test]
    fn test_sabr_beta_one() {
        // Beta = 1 corresponds to lognormal model
        let sabr = SABRParameters::new(0.2, 1.0, 0.0, 0.0).expect("Operation failed");
        let vol = sabr
            .implied_volatility(100.0, 100.0, 1.0)
            .expect("Operation failed");
        assert!((vol - 0.2).abs() < 0.01); // Should be close to alpha
    }

    #[test]
    fn test_sabr_symmetry() {
        let sabr = SABRParameters::new(0.2, 0.5, 0.0, 0.2).expect("Operation failed");
        let forward = 100.0;
        let time = 1.0;

        // With rho = 0, smile should be roughly symmetric
        let vol_up = sabr
            .implied_volatility(forward, 105.0, time)
            .expect("Operation failed");
        let vol_down = sabr
            .implied_volatility(forward, 95.0, time)
            .expect("Operation failed");

        // Should be similar (within 10%)
        assert!((vol_up - vol_down).abs() / vol_up < 0.1);
    }
}
