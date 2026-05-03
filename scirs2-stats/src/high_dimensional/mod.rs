//! High-dimensional statistical methods
//!
//! This module provides methods for estimating sparse precision matrices
//! and related model selection techniques, particularly useful when the
//! number of variables p is large relative to the sample size n.
//!
//! ## Graphical Lasso (GLASSO)
//!
//! The Graphical Lasso estimates a sparse precision matrix (inverse covariance)
//! by solving the L1-penalized log-likelihood problem:
//!
//! minimize  -log det(Theta) + tr(S * Theta) + lambda * ||Theta||_1  (off-diagonal)
//!
//! where S is the sample covariance, Theta is the precision matrix, and lambda
//! controls sparsity.
//!
//! ## Model Selection
//!
//! Various criteria for choosing the regularization parameter lambda:
//! - BIC (Bayesian Information Criterion)
//! - EBIC (Extended BIC) with tunable gamma
//! - K-fold cross-validation
//! - StARS (Stability Approach to Regularization Selection)
//! - Lambda path generation (log-spaced)
//! - Partial correlation extraction

pub mod clime;
pub mod factor_adjusted_testing;
mod glasso;
pub mod hd_tests;
mod model_selection;
pub mod neighborhood_selection;

pub use glasso::{GraphicalLasso, GraphicalLassoConfig, GraphicalLassoResult, PrecisionMatrix};
pub use model_selection::{
    compute_bic, compute_ebic, cross_validate_glasso, extract_partial_correlations,
    generate_lambda_path, select_lambda_stars, CovarianceEstimate, CrossValidationResult,
    EbicResult, LambdaSelectionResult, ModelSelectionCriterion, PartialCorrelationMatrix,
    StarsResult,
};
