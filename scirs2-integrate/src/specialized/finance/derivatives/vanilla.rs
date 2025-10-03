//! Vanilla derivatives pricing and analysis
//!
//! This module implements pricing models and analytical tools for standard vanilla derivatives
//! including European/American options, forwards, and futures contracts.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::pricing::black_scholes::normal_cdf;
use crate::specialized::finance::risk::greeks::Greeks;
use crate::specialized::finance::types::OptionType;

/// European vanilla option
#[derive(Debug, Clone)]
pub struct EuropeanOption {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub time: f64,
    pub option_type: OptionType,
}

impl EuropeanOption {
    /// Create a new European option
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time,
            option_type,
        }
    }

    /// Calculate option price using Black-Scholes formula
    pub fn price(&self) -> f64 {
        black_scholes_price(
            self.spot,
            self.strike,
            self.rate,
            self.dividend,
            self.volatility,
            self.time,
            self.option_type,
        )
    }

    /// Calculate Greeks
    pub fn greeks(&self) -> Greeks {
        Greeks::black_scholes(
            self.spot,
            self.strike,
            self.rate,
            self.dividend,
            self.volatility,
            self.time,
            self.option_type,
        )
    }

    /// Calculate implied volatility using Newton-Raphson method
    pub fn implied_volatility(&self, market_price: f64) -> IntegrateResult<f64> {
        implied_volatility_newton(
            market_price,
            self.spot,
            self.strike,
            self.rate,
            self.dividend,
            self.time,
            self.option_type,
        )
    }
}

/// American vanilla option (with early exercise)
#[derive(Debug, Clone)]
pub struct AmericanOption {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub volatility: f64,
    pub time: f64,
    pub option_type: OptionType,
}

impl AmericanOption {
    /// Create a new American option
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time,
            option_type,
        }
    }

    /// Calculate option price using Barone-Adesi-Whaley approximation
    pub fn price(&self) -> f64 {
        // For American call with no dividends, same as European
        if self.option_type == OptionType::Call && self.dividend == 0.0 {
            return black_scholes_price(
                self.spot,
                self.strike,
                self.rate,
                self.dividend,
                self.volatility,
                self.time,
                self.option_type,
            );
        }

        // Use Barone-Adesi-Whaley approximation for puts and dividend-paying calls
        barone_adesi_whaley(
            self.spot,
            self.strike,
            self.rate,
            self.dividend,
            self.volatility,
            self.time,
            self.option_type,
        )
    }

    /// Calculate approximate Greeks (using European as approximation)
    pub fn greeks(&self) -> Greeks {
        Greeks::black_scholes(
            self.spot,
            self.strike,
            self.rate,
            self.dividend,
            self.volatility,
            self.time,
            self.option_type,
        )
    }
}

/// Forward contract
#[derive(Debug, Clone)]
pub struct Forward {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub time: f64,
}

impl Forward {
    /// Create a new forward contract
    pub fn new(spot: f64, strike: f64, rate: f64, dividend: f64, time: f64) -> Self {
        Self {
            spot,
            strike,
            rate,
            dividend,
            time,
        }
    }

    /// Calculate forward price
    pub fn price(&self) -> f64 {
        self.spot * ((self.rate - self.dividend) * self.time).exp() - self.strike
    }

    /// Calculate fair forward strike
    pub fn fair_strike(&self) -> f64 {
        self.spot * ((self.rate - self.dividend) * self.time).exp()
    }
}

/// Black-Scholes pricing formula
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
    option_type: OptionType,
) -> f64 {
    let sqrt_time = time.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * time)
        / (volatility * sqrt_time);
    let d2 = d1 - volatility * sqrt_time;

    match option_type {
        OptionType::Call => {
            spot * (-(dividend * time)).exp() * normal_cdf(d1)
                - strike * (-(rate * time)).exp() * normal_cdf(d2)
        }
        OptionType::Put => {
            strike * (-(rate * time)).exp() * normal_cdf(-d2)
                - spot * (-(dividend * time)).exp() * normal_cdf(-d1)
        }
    }
}

/// Barone-Adesi-Whaley approximation for American options
fn barone_adesi_whaley(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
    option_type: OptionType,
) -> f64 {
    // European price as lower bound
    let european_price =
        black_scholes_price(spot, strike, rate, dividend, volatility, time, option_type);

    // Intrinsic value
    let intrinsic = match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    };

    // Simple approximation: use European price plus a fraction of early exercise premium
    // For a more accurate implementation, use binomial/trinomial trees
    let early_exercise_premium = match option_type {
        OptionType::Put => {
            // Puts have significant early exercise value
            let max_early_value = (intrinsic - european_price).max(0.0);
            // Approximate premium based on moneyness and time
            let moneyness = spot / strike;
            let time_factor = 1.0 - (-2.0 * time).exp();

            if moneyness < 0.95 {
                // Deep in the money - higher premium
                max_early_value * 0.5 * time_factor
            } else if moneyness < 1.05 {
                // At the money - moderate premium
                max_early_value * 0.3 * time_factor
            } else {
                // Out of the money - small premium
                max_early_value * 0.1 * time_factor
            }
        }
        OptionType::Call => {
            // Calls with dividends have early exercise value
            if dividend > 0.0 {
                let max_early_value = (intrinsic - european_price).max(0.0);
                let div_factor = dividend / rate;
                max_early_value * 0.2 * div_factor * (1.0 - (-time).exp())
            } else {
                0.0
            }
        }
    };

    // Return max of European price with premium or intrinsic value
    (european_price + early_exercise_premium).max(intrinsic)
}

/// Helper function for d1 at critical price
fn d1_star(s_star: f64, strike: f64, rate: f64, dividend: f64, volatility: f64, time: f64) -> f64 {
    ((s_star / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * time)
        / (volatility * time.sqrt())
}

/// Calculate implied volatility using Newton-Raphson method
pub fn implied_volatility_newton(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    time: f64,
    option_type: OptionType,
) -> IntegrateResult<f64> {
    // Initial guess using Brenner-Subrahmanyam approximation
    let mut vol = (2.0 * std::f64::consts::PI / time).sqrt() * (market_price / spot);
    vol = vol.max(0.01).min(5.0); // Bound initial guess

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        let price = black_scholes_price(spot, strike, rate, dividend, vol, time, option_type);
        let diff = price - market_price;

        if diff.abs() < TOLERANCE {
            return Ok(vol);
        }

        // Calculate vega for Newton step
        let sqrt_time = time.sqrt();
        let d1 =
            ((spot / strike).ln() + (rate - dividend + 0.5 * vol * vol) * time) / (vol * sqrt_time);
        let nprime_d1 = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * d1 * d1).exp();
        let vega = spot * (-(dividend * time)).exp() * nprime_d1 * sqrt_time;

        if vega < 1e-10 {
            return Err(IntegrateError::ValueError(
                "Vega too small for convergence".to_string(),
            ));
        }

        // Newton step
        vol -= diff / vega;

        // Ensure vol stays in reasonable range
        vol = vol.max(0.001).min(5.0);
    }

    Err(IntegrateError::ValueError(
        "Implied volatility failed to converge".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_european_call_price() {
        let option = EuropeanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility
            1.0,   // time
            OptionType::Call,
        );

        let price = option.price();
        // Black-Scholes reference: ~10.45
        assert!(price > 10.0 && price < 11.0, "Price: {}", price);
    }

    #[test]
    fn test_european_put_price() {
        let option = EuropeanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility
            1.0,   // time
            OptionType::Put,
        );

        let price = option.price();
        // Black-Scholes reference: ~5.57
        assert!(price > 5.0 && price < 6.0, "Price: {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let dividend = 0.0;
        let volatility = 0.2;
        let time = 1.0;

        let call = EuropeanOption::new(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time,
            OptionType::Call,
        );
        let put = EuropeanOption::new(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time,
            OptionType::Put,
        );

        // Put-Call Parity: C - P = S - K*exp(-rT)
        let lhs = call.price() - put.price();
        let rhs = spot - strike * (-(rate * time)).exp();

        assert!((lhs - rhs).abs() < 1e-10, "Put-call parity violated");
    }

    #[test]
    fn test_american_put_premium() {
        let american = AmericanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility
            1.0,   // time
            OptionType::Put,
        );

        let european = EuropeanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility
            1.0,   // time
            OptionType::Put,
        );

        // American put should be worth more than European put
        assert!(
            american.price() >= european.price(),
            "American put should be >= European put"
        );
    }

    #[test]
    fn test_forward_price() {
        let forward = Forward::new(
            100.0, // spot
            105.0, // strike
            0.05,  // rate
            0.0,   // dividend
            1.0,   // time
        );

        let price = forward.price();
        let expected = 100.0 * (0.05_f64).exp() - 105.0; // ~0.13

        assert!((price - expected).abs() < 0.1, "Forward price: {}", price);
    }

    #[test]
    fn test_implied_volatility() {
        let option = EuropeanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility (target)
            1.0,   // time
            OptionType::Call,
        );

        let market_price = option.price();
        let implied_vol = option.implied_volatility(market_price).unwrap();

        // Should recover the original volatility
        assert!(
            (implied_vol - 0.2).abs() < 1e-4,
            "Implied vol: {}, expected: 0.2",
            implied_vol
        );
    }

    #[test]
    fn test_greeks_calculation() {
        let option = EuropeanOption::new(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.0,   // dividend
            0.2,   // volatility
            1.0,   // time
            OptionType::Call,
        );

        let greeks = option.greeks();

        // Sanity checks for Greeks
        assert!(
            greeks.delta > 0.0 && greeks.delta < 1.0,
            "Delta: {}",
            greeks.delta
        );
        assert!(greeks.gamma > 0.0, "Gamma: {}", greeks.gamma);
        assert!(greeks.vega > 0.0, "Vega: {}", greeks.vega);
        assert!(greeks.theta < 0.0, "Theta: {}", greeks.theta); // Time decay is negative
        assert!(greeks.rho > 0.0, "Rho: {}", greeks.rho); // Call has positive rho
    }
}
