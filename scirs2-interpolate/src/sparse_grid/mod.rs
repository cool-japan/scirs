//! Sparse grid interpolation and quadrature methods.
//!
//! - `core`: hierarchical hat-function sparse grid (existing)
//! - `smolyak`: Smolyak construction with Clenshaw-Curtis, Gauss-Legendre, and Gauss-Patterson rules
//! - `anova`: classical ANOVA decomposition with Sobol' sensitivity indices
//! - `anchored_anova`: anchor-point ANOVA decomposition (Kuo-Sloan-Wasilkowski 2010)

pub mod anchored_anova;
pub mod anova;
pub mod core;
pub mod smolyak;

pub use anchored_anova::{
    adaptive_anchored_anova_refinement, anchored_anova_decompose, AnchoredAnovaDecomposition,
    AnchoredAnovaError,
};
pub use anova::{anova_decompose, AnovaConfig, AnovaDecomposition, AnovaError};
pub use core::*;
pub use smolyak::{
    smolyak_grid, smolyak_interpolant, smolyak_quadrature, SmolyakConfig, SmolyakGrid, SmolyakRule,
};
