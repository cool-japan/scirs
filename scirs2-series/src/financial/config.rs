//! Configuration types for financial time series analysis
//!
//! This module defines configuration structures and enums that control
//! various financial analysis models including GARCH variants and related
//! time series models.

/// GARCH model configuration
#[derive(Debug, Clone)]
pub struct GarchConfig {
    /// GARCH order (p)
    pub p: usize,
    /// ARCH order (q)
    pub q: usize,
    /// Mean model type
    pub mean_model: MeanModel,
    /// Distribution for residuals
    pub distribution: Distribution,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use numerical derivatives
    pub use_numerical_derivatives: bool,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            mean_model: MeanModel::Constant,
            distribution: Distribution::Normal,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_numerical_derivatives: false,
        }
    }
}

/// EGARCH model configuration
#[derive(Debug, Clone)]
pub struct EgarchConfig {
    /// GARCH order (p)
    pub p: usize,
    /// ARCH order (q)
    pub q: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for EgarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Mean model specification for GARCH
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeanModel {
    /// Constant mean
    Constant,
    /// Zero mean
    Zero,
    /// AR(p) mean model
    AR {
        /// Autoregressive order
        order: usize,
    },
    /// ARMA(p,q) mean model
    ARMA {
        /// Autoregressive order
        ar_order: usize,
        /// Moving average order
        ma_order: usize,
    },
}

/// Distribution for GARCH residuals
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Distribution {
    /// Normal distribution
    Normal,
    /// Student's t-distribution
    StudentT,
    /// Skewed Student's t-distribution
    SkewedStudentT,
    /// Generalized Error Distribution
    GED,
}

impl Default for MeanModel {
    fn default() -> Self {
        Self::Constant
    }
}

impl Default for Distribution {
    fn default() -> Self {
        Self::Normal
    }
}
