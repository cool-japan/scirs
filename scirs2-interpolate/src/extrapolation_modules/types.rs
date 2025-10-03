//! Core type definitions for extrapolation methods
//!
//! This module contains all the foundational data types used throughout
//! the extrapolation system.

use scirs2_core::numeric::Float;

/// Enhanced extrapolation methods for interpolation.
///
/// This enum provides advanced extrapolation capabilities that go beyond
/// the basic ExtrapolateMode enum. It allows for more sophisticated boundary
/// handling and domain extension methods, including:
///
/// - Physics-informed extrapolation based on boundary derivatives
/// - Polynomial extrapolation of various orders
/// - Decay/growth models for asymptotic behavior
/// - Periodic extension of the domain
/// - Reflection-based extrapolation
/// - Domain-specific extrapolation models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationMethod {
    /// No extrapolation - return an error for points outside the domain
    Error,

    /// Use the nearest endpoint value (constant extrapolation)
    Constant,

    /// Linear extrapolation based on endpoint derivatives
    Linear,

    /// Quadratic extrapolation based on endpoint values and derivatives
    Quadratic,

    /// Cubic extrapolation preserving both values and derivatives at boundaries
    Cubic,

    /// Extend domain as if the function is periodic
    Periodic,

    /// Reflect the function at the boundaries
    Reflection,

    /// Exponential decay/growth model for asymptotic behavior
    Exponential,

    /// Power law decay/growth model for asymptotic behavior
    PowerLaw,

    /// Spline-based extrapolation using the full spline continuation
    Spline,

    /// Akima extrapolation for stable polynomial continuation
    Akima,

    /// Sinusoidal extrapolation for periodic data
    Sinusoidal,

    /// Rational function extrapolation for poles/zeros behavior
    Rational,

    /// Confidence-based extrapolation with uncertainty bands
    Confidence,

    /// Ensemble extrapolation combining multiple methods
    Ensemble,

    /// Adaptive extrapolation that selects the best method locally
    Adaptive,

    /// Autoregressive extrapolation using AR models
    Autoregressive,

    /// Return zeros for all out-of-bounds points (SciPy 'zeros' mode)
    Zeros,

    /// Use nearest boundary value (SciPy 'nearest'/'edge' mode)
    Nearest,

    /// Mirror reflection without repeating edge values (SciPy 'mirror' mode)
    Mirror,

    /// Periodic wrapping (SciPy 'wrap' mode)
    Wrap,

    /// Clamped boundary conditions with zero derivatives
    Clamped,

    /// Grid-specific mirror mode for structured grids
    GridMirror,

    /// Grid-specific constant mode for structured grids
    GridConstant,

    /// Grid-specific wrap mode for structured grids
    GridWrap,
}

/// Direction for extrapolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationDirection {
    /// Extrapolation below the lower boundary
    Lower,

    /// Extrapolation above the upper boundary
    Upper,
}

/// Ensemble combination strategies for multiple extrapolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleCombinationStrategy {
    /// Simple arithmetic mean of all method predictions
    Mean,

    /// Weighted average based on confidence scores
    WeightedMean,

    /// Use median of all predictions (robust to outliers)
    Median,

    /// Use method with highest confidence for local region
    BestMethod,

    /// Weighted combination optimized for minimum variance
    MinimumVariance,

    /// Bayesian model averaging
    BayesianAveraging,

    /// Use methods voted by majority (discrete classification)
    Voting,

    /// Stack multiple methods using a meta-learner
    Stacking,
}

/// Adaptive selection criteria for choosing extrapolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveSelectionCriterion {
    /// Minimize cross-validation error on nearby data
    CrossValidationError,

    /// Maximize smoothness at boundary
    BoundarySmoothness,

    /// Minimize curvature discontinuity
    CurvatureContinuity,

    /// Use physics-informed metrics
    PhysicsConsistency,

    /// Minimize extrapolation uncertainty
    UncertaintyMinimization,

    /// Optimize for specific application domain
    DomainSpecific,

    /// Use information-theoretic criteria (AIC/BIC)
    InformationCriterion,

    /// Combine multiple criteria with weighted scoring
    MultiCriteria,
}

/// Autoregressive model fitting methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ARFittingMethod {
    /// Yule-Walker equations (method of moments)
    YuleWalker,

    /// Burg's method (maximum entropy)
    Burg,

    /// Least squares estimation
    LeastSquares,

    /// Maximum likelihood estimation
    MaximumLikelihood,

    /// Modified covariance method
    ModifiedCovariance,

    /// Forward-backward linear prediction
    ForwardBackward,
}

/// Physics laws for informed extrapolation
#[derive(Debug, Clone, Copy)]
pub enum PhysicsLaw {
    /// Mass conservation (non-negative, decay to zero)
    MassConservation,
    /// Energy conservation (quadratic behavior)
    EnergyConservation,
    /// Momentum conservation (linear behavior)
    MomentumConservation,
}

/// Boundary condition types for physics-informed extrapolation
#[derive(Debug, Clone, Copy)]
pub enum BoundaryType {
    /// Fixed value at boundary (Dirichlet)
    Dirichlet,
    /// Fixed derivative at boundary (Neumann)
    Neumann,
    /// Linear combination of value and derivative (Robin)
    Robin,
    /// Absorbing boundary with exponential decay
    Absorbing,
}

/// Data characteristics for adaptive extrapolation
#[derive(Debug, Clone)]
pub struct DataCharacteristics<T: Float> {
    /// Whether the data appears periodic
    pub is_periodic: bool,
    /// Estimated period if periodic
    pub estimated_period: Option<T>,
    /// Whether the data is monotonic
    pub is_monotonic: bool,
    /// Whether the data follows exponential-like growth/decay
    pub is_exponential_like: bool,
    /// Whether the data is oscillatory
    pub is_oscillatory: bool,
    /// Characteristic scale of the data
    pub characteristic_scale: T,
}

impl<T: Float> Default for DataCharacteristics<T> {
    fn default() -> Self {
        Self {
            is_periodic: false,
            estimated_period: None,
            is_monotonic: false,
            is_exponential_like: false,
            is_oscillatory: false,
            characteristic_scale: T::one(),
        }
    }
}

impl<T: Float> DataCharacteristics<T> {
    /// Create new data characteristics with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether the data is periodic
    pub fn with_periodic(mut self, periodic: bool, period: Option<T>) -> Self {
        self.is_periodic = periodic;
        self.estimated_period = period;
        self
    }

    /// Set whether the data is monotonic
    pub fn with_monotonic(mut self, monotonic: bool) -> Self {
        self.is_monotonic = monotonic;
        self
    }

    /// Set whether the data follows exponential behavior
    pub fn with_exponential_like(mut self, exponential: bool) -> Self {
        self.is_exponential_like = exponential;
        self
    }

    /// Set whether the data is oscillatory
    pub fn with_oscillatory(mut self, oscillatory: bool) -> Self {
        self.is_oscillatory = oscillatory;
        self
    }

    /// Set the characteristic scale
    pub fn with_scale(mut self, scale: T) -> Self {
        self.characteristic_scale = scale;
        self
    }
}
