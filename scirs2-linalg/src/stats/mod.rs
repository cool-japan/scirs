//! Statistical functions for matrices
//!
//! This module provides comprehensive statistical functions for operating on matrices,
//! including:
//! * Covariance and correlation computation
//! * Matrix-variate probability distributions  
//! * Statistical hypothesis tests for matrices
//! * Random sampling from matrix distributions
//! * Bootstrap and permutation testing utilities
//!
//! ## Main Components
//!
//! * [`covariance`] - Covariance and correlation matrices
//! * [`distributions`] - Matrix-variate probability distributions
//! * [`tests`] - Statistical hypothesis tests
//! * [`sampling`] - Random sampling utilities

pub mod covariance;
pub mod distributions;
pub mod sampling;
pub mod tests;

// Re-export key functions for convenience
pub use covariance::{correlation_matrix, covariance_matrix, mahalanobis_distance};
pub use distributions::{
    matrix_normal_logpdf, sample_matrix_normal, sample_wishart, wishart_logpdf, MatrixNormalParams,
    WishartParams,
};
pub use sampling::{
    bootstrap_sample, metropolis_hastings_sample, permutation_sample, sample_multivariate_normal,
};
pub use tests::{
    box_m_test, hotelling_t2_test, mardia_normality_test, mauchly_sphericity_test, TestResult,
};
