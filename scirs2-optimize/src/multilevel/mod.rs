//! Multi-level and multi-fidelity optimization
//!
//! Provides coarse-to-fine optimization strategies and variable fidelity
//! model management for expensive function evaluations.
//!
//! # Overview
//!
//! Multi-fidelity optimization leverages cheap low-fidelity surrogates to
//! guide expensive high-fidelity evaluations:
//!
//! ```text
//! High-fidelity:  f_H(x)  (expensive, accurate)
//! Low-fidelity:   f_L(x)  (cheap, approximate)
//! Correction:     δ(x) = f_H(x_sampled) - f_L(x_sampled)
//! ```
//!
//! # Provided Algorithms
//!
//! - [`MultilevelOptimizer`]: Coarse-to-fine optimization strategy
//! - [`VariableFidelity`]: Variable fidelity model manager
//! - [`MfRbf`]: Multi-fidelity RBF surrogate with additive correction
//! - [`TrustHierarchy`]: Hierarchical trust-region management
//! - [`MultigridOptimizer`]: Multigrid-inspired optimization (V/W cycle)
//!
//! # Example
//!
//! ```no_run
//! use scirs2_optimize::multilevel::{
//!     MultilevelOptimizer, MultilevelOptions, FidelityLevel,
//! };
//!
//! // High-fidelity function (expensive)
//! let high_fi = |x: &[f64]| -> f64 {
//!     (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
//! };
//! // Low-fidelity function (cheap approximation)
//! let low_fi = |x: &[f64]| -> f64 {
//!     (x[0] - 1.8).powi(2) + (x[1] - 2.8).powi(2)
//! };
//!
//! let mut optimizer = MultilevelOptimizer::new(
//!     vec![
//!         FidelityLevel::new(low_fi, 1.0),
//!         FidelityLevel::new(high_fi, 10.0),
//!     ],
//!     vec![0.0, 0.0],
//!     MultilevelOptions::default(),
//! );
//!
//! let result = optimizer.minimize().expect("valid input");
//! println!("Minimum at: {:?}", result.x);
//! ```

pub mod methods;

pub use methods::{
    FidelityLevel, MfRbf, MfRbfOptions, MultigridOptimizer, MultigridOptions, MultilevelOptimizer,
    MultilevelOptions, MultilevelResult, TrustHierarchy, TrustHierarchyOptions, VariableFidelity,
    VariableFidelityOptions,
};
