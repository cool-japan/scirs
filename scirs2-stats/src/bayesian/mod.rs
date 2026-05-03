//! Bayesian statistical methods
//!
//! This module provides implementations of Bayesian statistical techniques including:
//! - Conjugate priors
//! - Bayesian linear regression
//! - Hierarchical models
//! - Variational inference (CAVI, SVI, ADVI)
//!   - Stochastic Variational Inference with natural gradients and mini-batches
//!   - Automatic Differentiation Variational Inference (mean-field and full-rank)
//!   - Variational families: MeanFieldGaussian, FullRankGaussian, NormalizingFlowVI
//!   - Diagnostics: ELBO trace, gradient norms, convergence monitoring

mod advanced_mcmc;
mod conjugate;
mod enhanced_regression;
mod hierarchical;
mod regression;
pub mod smc_model_comparison;
mod variational;

pub use advanced_mcmc::*;
pub use conjugate::*;
// Import enhanced regression with aliases to avoid conflicts
pub use enhanced_regression::{
    bayesian_linear_regression_exact as enhanced_bayesian_linear_regression_exact,
    bayesian_linear_regression_vb as enhanced_bayesian_linear_regression_vb,
    BayesianRegressionConfig as EnhancedBayesianRegressionConfig,
    BayesianRegressionPrior as EnhancedBayesianRegressionPrior,
    BayesianRegressionResult as EnhancedBayesianRegressionResult,
    ConvergenceInfo as EnhancedConvergenceInfo, EnhancedBayesianRegression,
    InferenceMethod as EnhancedInferenceMethod,
};
pub use hierarchical::*;
// Import standard regression - this is the main API
pub use regression::*;
pub use variational::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
