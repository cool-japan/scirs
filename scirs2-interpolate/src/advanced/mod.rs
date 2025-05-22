//! Advanced interpolation methods
//!
//! This module provides advanced interpolation algorithms beyond the basic methods.
//! These include specialized techniques for different types of interpolation problems:
//!
//! - **Akima splines**: Robust to outliers with reduced oscillations
//! - **Barycentric interpolation**: Stable polynomial interpolation
//! - **Radial Basis Functions (RBF)**: Scattered data interpolation
//! - **Enhanced RBF**: Advanced RBF with automatic parameter selection
//! - **Kriging**: Gaussian process regression with uncertainty quantification
//! - **Enhanced Kriging**: Direction-dependent correlations and Bayesian inference
//! - **Fast Kriging**: Approximation methods for large datasets
//! - **Thin-plate splines**: Smooth interpolation minimizing bending energy

pub mod akima;
pub mod barycentric;
pub mod enhanced_kriging;
pub mod enhanced_rbf;
// Fast Kriging algorithms for large datasets
// Currently using a reexport with defined API but placeholder implementations
// This module is under active development with algorithms being optimized
// Full implementations will be available in future updates
pub mod fast_kriging_reexports;
// Aliasing to maintain API compatibility
pub use fast_kriging_reexports as fast_kriging;
pub mod kriging;
pub mod rbf;
pub mod thinplate;
