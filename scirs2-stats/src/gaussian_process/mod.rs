//! Gaussian Process Regression Module
//!
//! This module provides a complete implementation of Gaussian Process regression,
//! compatible with scikit-learn's API. It can be used for:
//!
//! - Non-parametric function approximation
//! - Bayesian optimization
//! - Uncertainty quantification
//! - Time series forecasting
//! - Spatial interpolation
//!
//! # Features
//!
//! - Multiple kernel functions (RBF, Matérn, etc.)
//! - Prior mean functions
//! - Prediction with uncertainty estimates
//! - Hyperparameter optimization via marginal likelihood
//! - SciRS2 POLICY compliant (uses scirs2-core abstractions)
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use scirs2_stats::gaussian_process::{GaussianProcessRegressor, SquaredExponential};
//! use scirs2_core::ndarray::{array, Array2};
//!
//! let kernel = SquaredExponential::default();
//! let mut gpr = GaussianProcessRegressor::new(kernel);
//!
//! let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Operation failed");
//! let y_train = array![0.0, 1.0, 0.0];
//!
//! gpr.fit(&x_train, &y_train).expect("Operation failed");
//!
//! let x_test = Array2::from_shape_vec((1, 1), vec![1.5]).expect("Operation failed");
//! let predictions = gpr.predict(&x_test).expect("Operation failed");
//! println!("Prediction: {}", predictions[0]);
//! ```
//!
//! ## With Uncertainty
//!
//! ```
//! use scirs2_stats::gaussian_process::{GaussianProcessRegressor, SquaredExponential};
//! use scirs2_core::ndarray::{array, Array2};
//!
//! let kernel = SquaredExponential::default();
//! let mut gpr = GaussianProcessRegressor::new(kernel);
//!
//! let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Operation failed");
//! let y_train = array![0.0, 1.0, 0.0];
//!
//! gpr.fit(&x_train, &y_train).expect("Operation failed");
//!
//! let x_test = Array2::from_shape_vec((1, 1), vec![1.5]).expect("Operation failed");
//! let (mean, std) = gpr.predict_with_std(&x_test).expect("Operation failed");
//! println!("Prediction: {} ± {}", mean[0], std[0]);
//! ```
//!
//! ## Custom Kernel
//!
//! ```
//! use scirs2_stats::gaussian_process::{
//!     GaussianProcessRegressor, Matern52, SumKernel, WhiteKernel
//! };
//! use scirs2_core::ndarray::{array, Array2};
//!
//! // Matérn kernel + noise
//! let matern = Matern52::new(1.0, 1.0);
//! let noise = WhiteKernel::new(0.1);
//! let kernel = SumKernel::new(matern, noise);
//!
//! let mut gpr = GaussianProcessRegressor::new(kernel);
//!
//! let x_train = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Operation failed");
//! let y_train = array![0.0, 1.0];
//!
//! gpr.fit(&x_train, &y_train).expect("Operation failed");
//! ```

pub mod extended;
pub mod gp;
pub mod kernel;
pub mod prior;
pub mod regression;

// Re-export main types
pub use gp::GaussianProcess;
pub use kernel::{
    Kernel, Matern12, Matern32, Matern52, SquaredExponential, SumKernel, WhiteKernel,
};
pub use prior::{ConstantPrior, LinearPrior, Prior, ZeroPrior};
pub use regression::{default_gp_regressor, GaussianProcessRegressor};
