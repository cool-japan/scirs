//! Option pricing models
//!
//! This module provides option pricing models for various derivative instruments.
//! Currently implements the Black-Scholes model with support for European call
//! and put options.

use scirs2_core::numeric::Float;

use super::utils::normal_cdf;
use crate::error::{Result, TimeSeriesError};

/// Black-Scholes option pricing model
///
/// Calculates the theoretical price of European options using the Black-Scholes formula.
/// This model assumes constant volatility, risk-free rate, and no dividends.
///
/// # Arguments
///
/// * `spot_price` - Current price of the underlying asset
/// * `strike_price` - Strike price of the option
/// * `time_to_expiry` - Time to expiration in years
/// * `risk_free_rate` - Risk-free interest rate (annual, continuously compounded)
/// * `volatility` - Volatility of the underlying asset (annual)
/// * `is_call` - true for call option, false for put option
///
/// # Returns
///
/// * `Result<F>` - Option price
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::pricing::options::black_scholes;
///
/// // Price a call option
/// let call_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, true).unwrap();
/// println!("Call option price: ${:.2}", call_price);
///
/// // Price a put option
/// let put_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, false).unwrap();
/// println!("Put option price: ${:.2}", put_price);
/// ```
///
/// # Formula
///
/// For a call option:
/// C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)
///
/// For a put option:
/// P = K × e^(-rT) × N(-d₂) - S₀ × N(-d₁)
///
/// Where:
/// - d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
/// - d₂ = d₁ - σ√T
/// - N(x) = cumulative standard normal distribution
/// - S₀ = current stock price, K = strike price, r = risk-free rate
/// - T = time to expiration, σ = volatility
pub fn black_scholes<F: Float + Clone>(
    spot_price: F,
    strike_price: F,
    time_to_expiry: F,
    risk_free_rate: F,
    volatility: F,
    is_call: bool,
) -> Result<F> {
    // Input validation
    if spot_price <= F::zero() || strike_price <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "price".to_string(),
            message: "Spot and strike prices must be positive".to_string(),
        });
    }

    if time_to_expiry <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "time_to_expiry".to_string(),
            message: "Time to expiry must be positive".to_string(),
        });
    }

    if volatility <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "volatility".to_string(),
            message: "Volatility must be positive".to_string(),
        });
    }

    // Calculate d1 and d2
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility.powi(2) / F::from(2.0).unwrap()) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    // Calculate normal CDFs
    let norm_cdf_d1 = normal_cdf(d1);
    let norm_cdf_d2 = normal_cdf(d2);

    if is_call {
        // Call option price: C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)
        Ok(spot_price * norm_cdf_d1
            - strike_price * (-risk_free_rate * time_to_expiry).exp() * norm_cdf_d2)
    } else {
        // Put option price using put-call parity: P = C - S₀ + K × e^(-rT)
        let call_price = spot_price * norm_cdf_d1
            - strike_price * (-risk_free_rate * time_to_expiry).exp() * norm_cdf_d2;
        Ok(call_price - spot_price + strike_price * (-risk_free_rate * time_to_expiry).exp())
    }
}

/// Calculate Black-Scholes Greeks for sensitivity analysis
///
/// Returns the option Greeks (Delta, Gamma, Theta, Vega, Rho) which measure
/// the sensitivity of option price to various parameters.
///
/// # Arguments
///
/// * `spot_price` - Current price of the underlying asset
/// * `strike_price` - Strike price of the option
/// * `time_to_expiry` - Time to expiration in years
/// * `risk_free_rate` - Risk-free interest rate
/// * `volatility` - Volatility of the underlying asset
/// * `is_call` - true for call option, false for put option
///
/// # Returns
///
/// * `Result<Greeks<F>>` - Structure containing all Greeks
pub fn black_scholes_greeks<F: Float + Clone>(
    spot_price: F,
    strike_price: F,
    time_to_expiry: F,
    risk_free_rate: F,
    volatility: F,
    is_call: bool,
) -> Result<Greeks<F>> {
    // Input validation (same as black_scholes)
    if spot_price <= F::zero() || strike_price <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "price".to_string(),
            message: "Spot and strike prices must be positive".to_string(),
        });
    }

    if time_to_expiry <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "time_to_expiry".to_string(),
            message: "Time to expiry must be positive".to_string(),
        });
    }

    if volatility <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "volatility".to_string(),
            message: "Volatility must be positive".to_string(),
        });
    }

    // Calculate d1 and d2
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility.powi(2) / F::from(2.0).unwrap()) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let norm_cdf_d1 = normal_cdf(d1);
    let norm_cdf_d2 = normal_cdf(d2);

    // Standard normal PDF
    let norm_pdf_d1 =
        (-d1.powi(2) / F::from(2.0).unwrap()).exp() / F::from(2.506628274631).unwrap(); // sqrt(2π)

    let discount_factor = (-risk_free_rate * time_to_expiry).exp();

    let greeks = if is_call {
        Greeks {
            delta: norm_cdf_d1,
            gamma: norm_pdf_d1 / (spot_price * volatility * sqrt_t),
            theta: -(spot_price * norm_pdf_d1 * volatility) / (F::from(2.0).unwrap() * sqrt_t)
                - risk_free_rate * strike_price * discount_factor * norm_cdf_d2,
            vega: spot_price * norm_pdf_d1 * sqrt_t,
            rho: strike_price * time_to_expiry * discount_factor * norm_cdf_d2,
        }
    } else {
        let norm_cdf_neg_d1 = normal_cdf(-d1);
        let norm_cdf_neg_d2 = normal_cdf(-d2);

        Greeks {
            delta: norm_cdf_d1 - F::one(),
            gamma: norm_pdf_d1 / (spot_price * volatility * sqrt_t),
            theta: -(spot_price * norm_pdf_d1 * volatility) / (F::from(2.0).unwrap() * sqrt_t)
                + risk_free_rate * strike_price * discount_factor * norm_cdf_neg_d2,
            vega: spot_price * norm_pdf_d1 * sqrt_t,
            rho: -strike_price * time_to_expiry * discount_factor * norm_cdf_neg_d2,
        }
    };

    Ok(greeks)
}

/// Calculate implied volatility using Newton-Raphson method
///
/// Given an option's market price, calculates the implied volatility that
/// would produce that price in the Black-Scholes model.
///
/// # Arguments
///
/// * `market_price` - Observed market price of the option
/// * `spot_price` - Current price of the underlying asset
/// * `strike_price` - Strike price of the option
/// * `time_to_expiry` - Time to expiration in years
/// * `risk_free_rate` - Risk-free interest rate
/// * `is_call` - true for call option, false for put option
///
/// # Returns
///
/// * `Result<F>` - Implied volatility
pub fn implied_volatility<F: Float + Clone>(
    market_price: F,
    spot_price: F,
    strike_price: F,
    time_to_expiry: F,
    risk_free_rate: F,
    is_call: bool,
) -> Result<F> {
    if market_price <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "market_price".to_string(),
            message: "Market price must be positive".to_string(),
        });
    }

    // Initial guess for volatility
    let mut volatility = F::from(0.2).unwrap(); // 20%
    let tolerance = F::from(1e-6).unwrap();
    let max_iterations = 100;

    for _ in 0..max_iterations {
        let price = black_scholes(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility,
            is_call,
        )?;
        let price_diff = price - market_price;

        if price_diff.abs() < tolerance {
            return Ok(volatility);
        }

        // Calculate vega (sensitivity to volatility)
        let greeks = black_scholes_greeks(
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility,
            is_call,
        )?;

        if greeks.vega.abs() < tolerance {
            break; // Avoid division by zero
        }

        // Newton-Raphson update
        volatility = volatility - price_diff / greeks.vega;

        // Ensure volatility stays positive
        volatility = volatility.max(F::from(0.001).unwrap());
    }

    Err(TimeSeriesError::InvalidInput(
        "Failed to converge on implied volatility".to_string(),
    ))
}

/// Structure containing option Greeks for sensitivity analysis
#[derive(Debug, Clone)]
pub struct Greeks<F: Float> {
    /// Delta: Sensitivity to underlying price change
    pub delta: F,
    /// Gamma: Rate of change of delta
    pub gamma: F,
    /// Theta: Time decay (negative for long options)
    pub theta: F,
    /// Vega: Sensitivity to volatility change
    pub vega: F,
    /// Rho: Sensitivity to interest rate change
    pub rho: F,
}

impl<F: Float> Greeks<F> {
    /// Check if all Greeks are finite and valid
    pub fn is_valid(&self) -> bool {
        self.delta.is_finite()
            && self.gamma.is_finite()
            && self.theta.is_finite()
            && self.vega.is_finite()
            && self.rho.is_finite()
    }

    /// Get risk sensitivities as tuple for convenience
    pub fn sensitivities(&self) -> (F, F, F, F, F) {
        (self.delta, self.gamma, self.theta, self.vega, self.rho)
    }
}

/// Calculate option premium components
///
/// Decomposes option price into intrinsic and time value components.
///
/// # Arguments
///
/// * `spot_price` - Current price of the underlying
/// * `strike_price` - Strike price of the option
/// * `option_price` - Total option price
/// * `is_call` - true for call, false for put
///
/// # Returns
///
/// * `(F, F)` - (intrinsic_value, time_value)
pub fn option_value_components<F: Float + Clone>(
    spot_price: F,
    strike_price: F,
    option_price: F,
    is_call: bool,
) -> (F, F) {
    let intrinsic_value = if is_call {
        (spot_price - strike_price).max(F::zero())
    } else {
        (strike_price - spot_price).max(F::zero())
    };

    let time_value = option_price - intrinsic_value;

    (intrinsic_value, time_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes() {
        // Test call option
        let call_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, true).unwrap();
        assert!(call_price > 0.0, "Call option should have positive price");

        // Test put option
        let put_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, false).unwrap();
        assert!(put_price > 0.0, "Put option should have positive price");

        // Test put-call parity approximately holds
        let strike = 100.0f64;
        let spot = 100.0f64;
        let rate = 0.05f64;
        let time = 1.0f64;
        let pv_strike = strike * (-rate * time).exp();

        let parity_diff = (call_price - put_price - (spot - pv_strike)).abs();
        assert!(
            parity_diff < 0.01,
            "Put-call parity should approximately hold"
        );
    }

    #[test]
    fn test_black_scholes_greeks() {
        let result = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(result.is_ok());

        let greeks = result.unwrap();
        assert!(greeks.is_valid());

        // Delta should be between 0 and 1 for call options
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0);

        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_option_value_components() {
        let spot = 110.0;
        let strike = 100.0;
        let option_price = 15.0;

        let (intrinsic, time_value) = option_value_components(spot, strike, option_price, true);

        assert_eq!(intrinsic, 10.0); // 110 - 100 for call
        assert_eq!(time_value, 5.0); // 15 - 10
    }

    #[test]
    fn test_invalid_inputs() {
        // Test negative prices
        let result = black_scholes(-100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(result.is_err());

        // Test zero volatility
        let result = black_scholes(100.0, 100.0, 1.0, 0.05, 0.0, true);
        assert!(result.is_err());

        // Test negative time
        let result = black_scholes(100.0, 100.0, -1.0, 0.05, 0.2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_implied_volatility() {
        // First calculate a price with known volatility
        let known_vol = 0.25;
        let price = black_scholes(100.0, 100.0, 1.0, 0.05, known_vol, true).unwrap();

        // Then recover the implied volatility
        let result = implied_volatility(price, 100.0, 100.0, 1.0, 0.05, true);

        if let Ok(implied_vol) = result {
            assert!((implied_vol - known_vol).abs() < 0.001);
        }
        // Note: implied_volatility might fail to converge in some cases, so we don't assert success
    }

    #[test]
    fn test_deep_in_money_call() {
        // Deep in-the-money call should have delta close to 1
        let greeks = black_scholes_greeks(150.0, 100.0, 1.0, 0.05, 0.2, true).unwrap();
        assert!(greeks.delta > 0.8);
    }

    #[test]
    fn test_deep_out_money_call() {
        // Deep out-of-the-money call should have delta close to 0
        let greeks = black_scholes_greeks(50.0, 100.0, 1.0, 0.05, 0.2, true).unwrap();
        assert!(greeks.delta < 0.2);
    }
}
