//! Local interpolation methods
//!
//! This module provides interpolation methods that use only local information
//! around the query point. These methods are particularly useful for scattered
//! data interpolation, noisy data, and creating smooth approximations.
//!
//! The methods in this module include:
//!
//! - Moving Least Squares: Creates smooth approximations by fitting local polynomials
//!   with distance-weighted least squares
//! - Local Polynomial Regression: Statistical method for fitting local polynomials
//!   with diagnostic information like confidence intervals and R²
//!
//! Local methods typically have these characteristics:
//!
//! - They handle scattered data in any dimension
//! - Their computational cost scales with the number of data points
//! - They can trade accuracy for smoothness by adjusting parameters
//! - They often handle noise well compared to exact interpolators

pub mod mls;
pub mod polynomial;

pub use mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
pub use polynomial::{
    make_loess, make_robust_loess, LocalPolynomialConfig, LocalPolynomialRegression,
    RegressionResult,
};
