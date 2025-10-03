//! Financial volatility models
//!
//! This module provides various GARCH-type models for volatility modeling
//! in financial time series. These models capture the time-varying volatility
//! and asymmetric responses commonly observed in financial markets.
//!
//! # Available Models
//!
//! ## GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
//! The standard GARCH model for symmetric volatility modeling.
//! - [`garch::GarchModel`] - Basic GARCH implementation
//! - Captures volatility clustering and persistence
//! - Suitable for general volatility modeling
//!
//! ## EGARCH (Exponential GARCH)
//! An asymmetric volatility model using logarithmic specification.
//! - [`egarch::EgarchModel`] - EGARCH implementation
//! - Captures leverage effects without parameter restrictions
//! - Uses logarithm of variance to ensure positivity
//!
//! ## GJR-GARCH (Glosten-Jagannathan-Runkle GARCH)
//! A threshold GARCH model for asymmetric volatility.
//! - [`gjr_garch::GjrGarchModel`] - GJR-GARCH implementation
//! - Simple threshold specification for leverage effects
//! - Different response to positive vs negative shocks
//!
//! ## APARCH (Asymmetric Power ARCH)
//! A flexible model that nests many GARCH-type specifications.
//! - [`aparch::AparchModel`] - APARCH implementation
//! - Power transformation and asymmetric effects
//! - Generalizes GARCH, GJR-GARCH, and other models
//!
//! # Model Selection Guide
//!
//! Choose the appropriate model based on your needs:
//!
//! - **GARCH**: Use for basic volatility modeling when you don't expect strong asymmetric effects
//! - **EGARCH**: Use when you need to capture leverage effects and want parameter-free positivity
//! - **GJR-GARCH**: Use when you want simple threshold effects for asymmetric volatility
//! - **APARCH**: Use when you need maximum flexibility and want to nest different specifications
//!
//! # Examples
//!
//! ## Comparing Models on the Same Data
//! ```rust
//! use scirs2_series::financial::models::*;
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
//!                      0.008, -0.005, 0.011, -0.002, 0.009, 0.003, -0.007, 0.013, -0.004, 0.006,
//!                      0.002, -0.008, 0.014, -0.003, 0.010, 0.001, -0.006, 0.012, -0.001, 0.005];
//!
//! // Fit different models
//! let mut garch = garch::GarchModel::garch_11();
//! let garch_result = garch.fit(&returns).unwrap();
//!
//! let mut egarch = egarch::EgarchModel::egarch_11();  
//! let egarch_result = egarch.fit(&returns).unwrap();
//!
//! let mut gjr_garch = gjr_garch::GjrGarchModel::new();
//! let gjr_result = gjr_garch.fit(&returns).unwrap();
//!
//! let mut aparch = aparch::AparchModel::new();
//! let aparch_result = aparch.fit(&returns).unwrap();
//!
//! // Compare information criteria for model selection
//! println!("GARCH AIC: {}", garch_result.aic);
//! println!("EGARCH AIC: {}", egarch_result.aic);
//! println!("GJR-GARCH AIC: {}", gjr_result.aic);
//! println!("APARCH AIC: {}", aparch_result.aic);
//! ```

pub mod aparch;
pub mod egarch;
pub mod garch;
pub mod gjr_garch;

// Re-export main types for convenience
pub use aparch::{AparchModel, AparchParameters, AparchResult};
pub use egarch::{EgarchConfig, EgarchModel, EgarchParameters, EgarchResult};
pub use garch::{Distribution, GarchConfig, GarchModel, GarchParameters, GarchResult, MeanModel};
pub use gjr_garch::{GjrGarchModel, GjrGarchParameters, GjrGarchResult};
