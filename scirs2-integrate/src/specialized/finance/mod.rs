//! Financial derivatives pricing and risk management
//!
//! This module provides a comprehensive toolkit for quantitative finance, including:
//! - **Option pricing**: Black-Scholes, finite difference, Monte Carlo, binomial trees
//! - **Exotic derivatives**: Barriers, Asians, lookbacks, digitals, variance swaps
//! - **Greeks calculation**: Delta, Gamma, Vega, Theta, Rho with efficient batch computation
//! - **Implied volatility**: Newton-Raphson and binary search methods
//! - **Model calibration**: Volatility surfaces, Heston parameters, SABR interpolation
//! - **Path simulation**: Geometric Brownian motion with variance reduction
//! - **Risk metrics**: VaR (Historical, Parametric, Monte Carlo), Expected Shortfall
//!
//! # Quick Start
//!
//! ## Pricing a European Option
//!
//! ```
//! use scirs2_integrate::specialized::finance::{
//!     derivatives::vanilla::EuropeanOption,
//!     types::OptionType,
//! };
//!
//! // Create a European call option
//! let option = EuropeanOption::new(
//!     100.0,  // spot price
//!     100.0,  // strike
//!     0.05,   // risk-free rate
//!     0.0,    // dividend yield
//!     0.2,    // volatility (20%)
//!     1.0,    // maturity (1 year)
//!     OptionType::Call,
//! );
//!
//! // Price the option
//! let price = option.price();
//! println!("Option price: ${:.2}", price);
//!
//! // Calculate Greeks
//! let greeks = option.greeks();
//! println!("Delta: {:.4}", greeks.delta);
//! println!("Gamma: {:.4}", greeks.gamma);
//! println!("Vega: {:.4}", greeks.vega);
//! ```
//!
//! ## Monte Carlo Simulation
//!
//! ```
//! use scirs2_integrate::specialized::finance::{
//!     utils::{PathGenerator, PathConfig, VarianceReduction},
//! };
//!
//! // Configure path generation
//! let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0)
//!     .with_steps(252)  // Daily steps
//!     .with_variance_reduction(VarianceReduction::Antithetic);
//!
//! let generator = PathGenerator::new(config);
//!
//! // Generate 10,000 price paths
//! let paths = generator.generate_paths(10000).unwrap();
//! println!("Generated {} paths", paths.len());
//! ```
//!
//! ## Implied Volatility
//!
//! ```
//! use scirs2_integrate::specialized::finance::utils::{
//!     black_scholes_call, implied_volatility_newton,
//! };
//!
//! // Market price of a call option
//! let market_price = 10.5;
//!
//! // Calculate implied volatility
//! let iv = implied_volatility_newton(
//!     market_price,
//!     100.0,  // spot
//!     100.0,  // strike
//!     1.0,    // time
//!     0.05,   // rate
//!     true,   // is_call
//! ).unwrap();
//!
//! println!("Implied volatility: {:.2}%", iv * 100.0);
//! ```
//!
//! ## SABR Volatility Smile
//!
//! ```
//! use scirs2_integrate::specialized::finance::utils::SABRParameters;
//!
//! // Create SABR parameters
//! let sabr = SABRParameters::new(
//!     0.2,    // alpha (initial vol)
//!     0.5,    // beta (elasticity)
//!     -0.3,   // rho (correlation)
//!     0.4,    // nu (vol of vol)
//! ).unwrap();
//!
//! // Calculate implied vol across strikes
//! let forward = 100.0;
//! let time = 1.0;
//!
//! for strike in [90.0, 95.0, 100.0, 105.0, 110.0] {
//!     let vol = sabr.implied_volatility(forward, strike, time).unwrap();
//!     println!("Strike {}: vol = {:.2}%", strike, vol * 100.0);
//! }
//! ```
//!
//! ## Calibrating to Market Data
//!
//! ```
//! use scirs2_integrate::specialized::finance::utils::{
//!     ImpliedVolatilitySurface, HestonCalibrator, LossFunction,
//! };
//!
//! // Build volatility surface from market quotes
//! let mut surface = ImpliedVolatilitySurface::new(100.0, 0.05, 0.0);
//!
//! // Add market data (strike, maturity, implied vol, bid-ask spread)
//! surface.add_quote(95.0, 1.0, 0.22, Some(0.01));
//! surface.add_quote(100.0, 1.0, 0.20, Some(0.005));
//! surface.add_quote(105.0, 1.0, 0.21, Some(0.01));
//!
//! // Calibrate Heston model
//! let calibrator = HestonCalibrator::new(surface, LossFunction::WeightedMSE);
//! let result = calibrator.calibrate().unwrap();
//!
//! println!("Calibrated parameters: {:?}", result.parameters);
//! ```
//!
//! ## Exotic Options
//!
//! ```
//! use scirs2_integrate::specialized::finance::{
//!     derivatives::exotic::{AsianOption, AveragingMethod},
//!     types::OptionType,
//! };
//!
//! // Asian option with geometric averaging
//! let asian = AsianOption::new(
//!     100.0,  // spot
//!     100.0,  // strike
//!     0.05,   // rate
//!     0.0,    // dividend
//!     0.2,    // volatility
//!     1.0,    // maturity
//!     OptionType::Call,
//!     AveragingMethod::Geometric,
//!     252,    // observation points
//! ).unwrap();
//!
//! // Price using closed-form formula
//! let price = asian.price_geometric_closed_form().unwrap();
//! println!("Asian option price: ${:.2}", price);
//! ```
//!
//! # Architecture
//!
//! The module is organized into several submodules:
//!
//! - [`derivatives`]: Option contracts (vanilla, exotic, variance swaps)
//! - [`pricing`]: Pricing methods (Black-Scholes, PDE, MC, trees, Fourier)
//! - [`models`]: Stochastic models (volatility, interest rates)
//! - [`risk`]: Risk measures (Greeks, VaR, stress testing)
//! - `utils`: Utilities (math functions, calibration, simulation)
//! - [`solvers`]: PDE solvers for stochastic processes
//!
//! # Performance Features
//!
//! - **SIMD acceleration**: Via scirs2-core for vector operations
//! - **Parallel execution**: Monte Carlo paths can be generated in parallel
//! - **Variance reduction**: Antithetic variates, control variates
//! - **Efficient Greeks**: Batch calculation using automatic differentiation
//! - **Numerical stability**: Safe math functions, domain validation

pub mod advanced_monte_carlo_engine;
pub mod derivatives;
pub mod exotic_options;
pub mod ml;
pub mod models;
pub mod monte_carlo;
pub mod pricing;
pub mod realtime_risk_engine;
pub mod risk;
pub mod risk_management;
pub mod solvers;
pub mod term_structure;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use models::*;
pub use risk::greeks::Greeks;
pub use solvers::stochastic_pde::StochasticPDESolver;
pub use types::*;

// Re-export pricing methods
pub use pricing::{black_scholes::*, finite_difference::*, monte_carlo::*};

// Re-export main solver for backwards compatibility
pub use solvers::StochasticPDESolver as FinancialPDESolver;
