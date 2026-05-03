//! Machine learning applications in quantitative finance
//!
//! This module provides machine learning models and tools specifically designed for
//! financial applications including pricing, risk management, and portfolio optimization.
//!
//! # Modules
//! - `volatility_forecast`: Volatility prediction models (GARCH, EWMA, Historical)
//! - `portfolio_optim`: Mean-variance portfolio optimization
//! - `neural_pricing`: Neural network-based derivative pricing

pub mod neural_pricing;
pub mod portfolio_optim;
pub mod volatility_forecast;

// Volatility forecasting
pub use volatility_forecast::{EWMAModel, GarchModel, HistoricalVolatility, VolatilityForecaster};

// Neural pricing
pub use neural_pricing::{generate_black_scholes_training_data, DeepPricingNetwork, NeuralPricer};

// Portfolio optimization
pub use portfolio_optim::{MLPortfolioOptimizer, MeanVariancePortfolio};
