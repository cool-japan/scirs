//! Volatility Estimation for Financial Time Series
//!
//! This module provides comprehensive volatility estimation techniques used
//! in quantitative finance. Volatility estimation is fundamental for risk
//! management, option pricing, and portfolio optimization.
//!
//! # Module Organization
//!
//! ## Estimators
//! [`estimators`] - Core volatility estimation algorithms:
//! - High-frequency estimators using HLOC data
//! - Time series model-based estimators
//! - Range-based and sampling estimators
//!
//! # Estimation Techniques
//!
//! ## High-Frequency Estimators
//!
//! These estimators use intraday high, low, open, close (HLOC) data and
//! are generally more efficient than return-based estimators:
//!
//! - **Garman-Klass**: Most efficient unbiased estimator
//! - **Rogers-Satchell**: Drift-independent, robust to price gaps
//! - **Yang-Zhang**: Handles overnight returns and opening gaps
//! - **Parkinson**: Simple high-low range estimator
//!
//! ## Model-Based Estimators
//!
//! These use time series models to capture volatility dynamics:
//!
//! - **GARCH**: Captures volatility clustering and persistence
//! - **EWMA**: Exponentially weighted moving average (RiskMetrics)
//!
//! ## Basic Estimators
//!
//! - **Realized Volatility**: Simple sum of squared returns
//! - **Range Volatility**: Uses high-low ranges over periods
//! - **Intraday Volatility**: High-frequency sampling-based
//!
//! # Usage Guidelines
//!
//! ## Choosing the Right Estimator
//!
//! - **For highest efficiency**: Use Garman-Klass with HLOC data
//! - **For robustness to gaps**: Use Rogers-Satchell or Yang-Zhang
//! - **For simplicity**: Use Parkinson (high-low only) or realized volatility
//! - **For time-varying volatility**: Use GARCH or EWMA
//! - **For limited data**: Use realized volatility or range-based estimators
//!
//! ## Data Requirements
//!
//! | Estimator | Data Required | Efficiency |
//! |-----------|---------------|------------|
//! | Realized Volatility | Close prices | Low |
//! | Parkinson | High, Low | Medium |
//! | Garman-Klass | HLOC | High |
//! | Rogers-Satchell | HLOC | High |
//! | Yang-Zhang | HLOC | High |
//! | GARCH | Returns series | Medium |
//! | EWMA | Returns series | Medium |
//!
//! # Examples
//!
//! ## Basic Volatility Estimation
//! ```rust
//! use scirs2_series::financial::volatility::estimators::{
//!     realized_volatility, parkinson_volatility, garman_klass_volatility
//! };
//! use scirs2_core::ndarray::array;
//!
//! // Simple realized volatility
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
//! let realized_vol = realized_volatility(&returns);
//!
//! // High-low range estimator
//! let high = array![102.0, 105.0, 103.5];
//! let low = array![98.0, 101.0, 99.5];
//! let park_vol = parkinson_volatility(&high, &low).unwrap();
//!
//! // Most efficient estimator with full HLOC data
//! let close = array![100.0, 103.0, 101.0];
//! let open = array![99.0, 102.0, 102.5];
//! let gk_vol = garman_klass_volatility(&high, &low, &close, &open).unwrap();
//! ```
//!
//! ## Time-Varying Volatility Models
//! ```rust
//! use scirs2_series::financial::volatility::estimators::{ewma_volatility, garch_volatility_estimate};
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007];
//!
//! // RiskMetrics EWMA model
//! let lambda = 0.94; // Standard daily decay factor
//! let ewma_vol = ewma_volatility(&returns, lambda).unwrap();
//!
//! // Simple GARCH(1,1) estimation
//! let window = 5;
//! let garch_vol = garch_volatility_estimate(&returns, window).unwrap();
//! ```
//!
//! ## Comprehensive Volatility Analysis
//! ```rust
//! use scirs2_series::financial::volatility::estimators::*;
//! use scirs2_core::ndarray::array;
//!
//! // HLOC data
//! let high = array![102.0, 105.0, 103.5, 106.0];
//! let low = array![98.0, 101.0, 99.5, 102.0];
//! let close = array![100.0, 103.0, 101.0, 104.0];
//! let open = array![99.0, 102.0, 102.5, 100.5];
//!
//! // Compare different estimators
//! let gk_vol = garman_klass_volatility(&high, &low, &close, &open).unwrap();
//! let rs_vol = rogers_satchell_volatility(&high, &low, &close, &open).unwrap();
//! let yz_vol = yang_zhang_volatility(&high, &low, &close, &open, 0.34).unwrap();
//! let park_vol = parkinson_volatility(&high, &low).unwrap();
//!
//! println!("Garman-Klass: {:?}", gk_vol);
//! println!("Rogers-Satchell: {:?}", rs_vol);
//! println!("Yang-Zhang: {:?}", yz_vol);
//! println!("Parkinson: {:?}", park_vol);
//! ```
//!
//! # Implementation Notes
//!
//! ## Performance
//!
//! All estimators are optimized for performance with:
//! - Efficient array operations using ndarray
//! - Minimal memory allocations
//! - Vectorized computations where possible
//!
//! ## Numerical Stability
//!
//! The implementations handle edge cases such as:
//! - Zero price ranges (high = low)
//! - Invalid price data (negative prices)
//! - Insufficient data for window-based estimators
//!
//! ## Reference Implementation
//!
//! These estimators follow standard academic and industry formulations:
//! - Garman-Klass (1980) for the GK estimator
//! - Rogers-Satchell (1991) for the RS estimator  
//! - Yang-Zhang (2000) for the YZ estimator
//! - RiskMetrics (1996) for EWMA methodology

pub mod estimators;

// Re-export commonly used estimators for convenience
pub use estimators::{
    ewma_volatility, garch_volatility_estimate, garman_klass_volatility, intraday_volatility,
    parkinson_volatility, range_volatility, realized_volatility, rogers_satchell_volatility,
    yang_zhang_volatility,
};
