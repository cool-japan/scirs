//! Financial Pricing Models
//!
//! This module provides comprehensive financial pricing models for various
//! derivative instruments and financial products. It implements both classical
//! and modern pricing approaches used in quantitative finance.
//!
//! # Module Organization
//!
//! ## Option Pricing
//! [`options`] - Option pricing models and Greeks calculation:
//! - Black-Scholes model for European options
//! - Greeks (Delta, Gamma, Theta, Vega, Rho) calculation
//! - Implied volatility calculation
//! - Option value decomposition (intrinsic vs time value)
//!
//! ## Utility Functions
//! [`utils`] - Mathematical and financial utility functions:
//! - Normal distribution functions (CDF, PDF, quantile)
//! - Present/future value calculations
//! - Black-Scholes parameter calculations (d1, d2)
//! - Bivariate normal distribution
//! - Return conversion utilities
//!
//! # Supported Pricing Models
//!
//! ## Options
//! - **Black-Scholes**: European call and put options
//! - **Greeks Analysis**: Complete sensitivity analysis
//! - **Implied Volatility**: Market-based volatility extraction
//!
//! ## Future Extensions
//! The module is designed to accommodate additional pricing models:
//! - Binomial/trinomial trees for American options
//! - Monte Carlo simulation methods
//! - Finite difference methods for exotic options
//! - Bond pricing models (yield curve, credit models)
//! - Commodity and energy derivatives
//!
//! # Usage Examples
//!
//! ## Basic Option Pricing
//! ```rust
//! use scirs2_series::financial::pricing::options::{black_scholes, black_scholes_greeks};
//!
//! // Price a European call option
//! let spot_price = 100.0;    // Current stock price
//! let strike_price = 105.0;  // Strike price
//! let time_to_expiry = 0.25; // 3 months
//! let risk_free_rate = 0.05; // 5% annual
//! let volatility = 0.20;     // 20% annual volatility
//!
//! // Calculate option price
//! let call_price = black_scholes(spot_price, strike_price, time_to_expiry,
//!                               risk_free_rate, volatility, true).unwrap();
//! println!("Call option price: ${:.2}", call_price);
//!
//! // Calculate Greeks for risk management
//! let greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry,
//!                                  risk_free_rate, volatility, true).unwrap();
//! println!("Delta: {:.4}, Gamma: {:.4}, Vega: {:.4}",
//!          greeks.delta, greeks.gamma, greeks.vega);
//! ```
//!
//! ## Put-Call Parity Verification
//! ```rust
//! use scirs2_series::financial::pricing::{options::black_scholes, utils::present_value};
//!
//! let spot = 100.0;
//! let strike = 100.0;
//! let time = 1.0;
//! let rate = 0.05;
//! let vol = 0.2;
//!
//! let call = black_scholes(spot, strike, time, rate, vol, true).unwrap();
//! let put = black_scholes(spot, strike, time, rate, vol, false).unwrap();
//! let pv_strike = present_value(strike, rate, time);
//!
//! // Put-Call Parity: C - P = S - PV(K)
//! let parity_lhs = call - put;
//! let parity_rhs = spot - pv_strike;
//! println!("Put-Call Parity check: {:.6} ≈ {:.6}", parity_lhs, parity_rhs);
//! ```
//!
//! ## Implied Volatility Analysis
//! ```rust
//! use scirs2_series::financial::pricing::options::{black_scholes, implied_volatility};
//!
//! // Given market prices, calculate implied volatilities
//! let market_prices = [8.5, 10.2, 12.1]; // Option prices at different strikes
//! let strikes = [95.0, 100.0, 105.0];
//! let spot = 100.0;
//! let time = 0.25;
//! let rate = 0.05;
//!
//! for (price, strike) in market_prices.iter().zip(strikes.iter()) {
//!     if let Ok(impl_vol) = implied_volatility(*price, spot, *strike, time, rate, true) {
//!         println!("Strike {}: Implied Vol {:.2}%", strike, impl_vol * 100.0);
//!     }
//! }
//! ```
//!
//! ## Portfolio Option Analysis
//! ```rust
//! use scirs2_series::financial::pricing::options::{black_scholes, black_scholes_greeks, option_value_components};
//!
//! // Analyze a portfolio of options
//! struct OptionPosition {
//!     spot: f64, strike: f64, time: f64, vol: f64, is_call: bool, quantity: i32,
//! }
//!
//! let positions = vec![
//!     OptionPosition { spot: 100.0, strike: 95.0, time: 0.25, vol: 0.22, is_call: true, quantity: 100 },
//!     OptionPosition { spot: 100.0, strike: 105.0, time: 0.25, vol: 0.18, is_call: false, quantity: -50 },
//! ];
//!
//! let rate = 0.05;
//! let mut total_value = 0.0;
//! let mut total_delta = 0.0;
//!
//! for pos in &positions {
//!     let price = black_scholes(pos.spot, pos.strike, pos.time, rate, pos.vol, pos.is_call).unwrap();
//!     let greeks = black_scholes_greeks(pos.spot, pos.strike, pos.time, rate, pos.vol, pos.is_call).unwrap();
//!     
//!     total_value += price * pos.quantity as f64;
//!     total_delta += greeks.delta * pos.quantity as f64;
//! }
//!
//! println!("Portfolio Value: ${:.2}", total_value);
//! println!("Portfolio Delta: {:.2}", total_delta);
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Black-Scholes Model
//!
//! The Black-Scholes formula for European options:
//!
//! **Call Option:**
//! ```text
//! C = S0 * N(d1) - K * exp(-r * T) * N(d2)
//! ```
//!
//! **Put Option:**
//! ```text
//! P = K * exp(-r * T) * N(-d2) - S0 * N(-d1)
//! ```
//!
//! Where:
//! - `d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)`
//! - `d₂ = d₁ - σ√T`
//! - `S₀` = current stock price
//! - `K` = strike price
//! - `r` = risk-free rate
//! - `T` = time to expiration
//! - `σ` = volatility
//! - `N(x)` = cumulative standard normal distribution
//!
//! ## Greeks Formulas
//!
//! **Delta (Price Sensitivity):**
//! - Call: `Δ = N(d₁)`
//! - Put: `Δ = N(d₁) - 1`
//!
//! **Gamma (Delta Sensitivity):**
//! - `Γ = φ(d₁) / (S₀ × σ × √T)`
//!
//! **Theta (Time Decay):**
//! - Call: `Θ = -S₀ × φ(d₁) × σ / (2√T) - rK × e^(-rT) × N(d₂)`
//! - Put: `Θ = -S₀ × φ(d₁) × σ / (2√T) + rK × e^(-rT) × N(-d₂)`
//!
//! **Vega (Volatility Sensitivity):**
//! - `ν = S₀ × φ(d₁) × √T`
//!
//! **Rho (Interest Rate Sensitivity):**
//! - Call: `ρ = K × T × e^(-rT) × N(d₂)`
//! - Put: `ρ = -K × T × e^(-rT) × N(-d₂)`
//!
//! # Risk Management Applications
//!
//! ## Delta Hedging
//! ```rust
//! use scirs2_series::financial::pricing::options::black_scholes_greeks;
//!
//! // Calculate hedge ratio for delta-neutral portfolio
//! let option_greeks = black_scholes_greeks(100.0, 100.0, 0.25, 0.05, 0.2, true).unwrap();
//! let option_quantity = 1000; // Long 1000 call options
//!
//! // Hedge ratio: short delta × option quantity shares of stock
//! let hedge_shares = -option_greeks.delta * option_quantity as f64;
//! println!("Hedge with {:.0} shares of underlying stock", hedge_shares);
//! ```
//!
//! ## Gamma Risk Management
//! ```rust
//! use scirs2_series::financial::pricing::options::black_scholes_greeks;
//!
//! // Monitor gamma exposure for large price moves
//! let greeks = black_scholes_greeks(100.0, 100.0, 0.25, 0.05, 0.2, true).unwrap();
//! let position_size = 10000;
//!
//! // Estimate delta change for 1% stock move
//! let stock_move = 1.0; // $1 move
//! let delta_change = greeks.gamma * stock_move * position_size as f64;
//! println!("Delta will change by {:.0} for ${:.0} stock move", delta_change, stock_move);
//! ```
//!
//! ## Volatility Trading
//! ```rust
//! use scirs2_series::financial::pricing::options::implied_volatility;
//!
//! // Compare implied vs historical volatility for trading opportunities
//! let market_price = 8.5;
//! let historical_vol = 0.18; // 18% historical volatility
//!
//! if let Ok(implied_vol) = implied_volatility(market_price, 100.0, 100.0, 0.25, 0.05, true) {
//!     if implied_vol > historical_vol * 1.1 {
//!         println!("Option may be overpriced - consider selling volatility");
//!     } else if implied_vol < historical_vol * 0.9 {
//!         println!("Option may be underpriced - consider buying volatility");
//!     }
//! }
//! ```
//!
//! # Model Assumptions and Limitations
//!
//! ## Black-Scholes Assumptions
//! 1. **Constant volatility** - Reality shows volatility clustering and jumps
//! 2. **Constant risk-free rate** - Rates change over time
//! 3. **No dividends** - Most stocks pay dividends
//! 4. **European exercise** - Many options allow early exercise
//! 5. **Log-normal stock prices** - Markets show fat tails and skewness
//! 6. **No transaction costs** - Real trading involves costs
//!
//! ## Practical Considerations
//! - Use the model as a baseline for relative value analysis
//! - Adjust for dividends when pricing equity options
//! - Consider early exercise features for American options
//! - Account for volatility smile/skew in practice
//! - Monitor model performance and calibrate regularly
//!
//! # Performance and Numerical Considerations
//!
//! ## Accuracy
//! - Normal CDF approximation: maximum error ≈ 7.5 × 10⁻⁸
//! - Greeks calculations use analytical formulas for precision
//! - Implied volatility: Newton-Raphson with tolerance 10⁻⁶
//!
//! ## Computational Efficiency
//! - Direct analytical formulas (no numerical integration)
//! - Optimized for bulk calculations
//! - Minimal memory allocation
//! - Suitable for real-time pricing applications
//!
//! # Extension Points
//!
//! The module provides a foundation for implementing additional pricing models:
//!
//! ```rust
//! // Future: American option pricing with binomial trees
//! // pub fn american_option_binomial(params...) -> Result<f64>;
//!
//! // Future: Exotic option pricing
//! // pub fn barrier_option_pricing(params...) -> Result<f64>;
//!
//! // Future: Interest rate derivatives
//! // pub fn bond_option_pricing(params...) -> Result<f64>;
//! ```

pub mod options;
pub mod utils;

// Re-export commonly used functions for convenience
pub use options::{
    black_scholes, black_scholes_greeks, implied_volatility, option_value_components, Greeks,
};

pub use utils::{
    bivariate_normal_cdf, calculate_d1, calculate_d2, continuous_to_simple_return, future_value,
    normal_cdf, normal_pdf, normal_quantile, present_value, simple_to_continuous_return,
};
