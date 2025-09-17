//! Core type definitions for numerical stability assessment
//!
//! This module contains all the foundational data types and enums used throughout
//! the numerical stability system.

use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{AddAssign, SubAssign};

/// Condition number and stability assessment report
#[derive(Debug, Clone)]
pub struct ConditionReport<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Estimated condition number of the matrix
    pub condition_number: F,

    /// Whether the matrix is considered well-conditioned
    pub is_well_conditioned: bool,

    /// Suggested regularization parameter if needed
    pub recommended_regularization: Option<F>,

    /// Overall stability classification
    pub stability_level: StabilityLevel,

    /// Additional diagnostic information
    pub diagnostics: StabilityDiagnostics<F>,
}

/// Classification of numerical stability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StabilityLevel {
    /// Excellent stability (condition number < 1e12)
    Excellent,
    /// Good stability (condition number < 1e14)
    Good,
    /// Marginal stability (condition number < 1e16)
    Marginal,
    /// Poor stability (condition number >= 1e16)
    Poor,
}

/// Additional diagnostic information about matrix stability
#[derive(Debug, Clone)]
pub struct StabilityDiagnostics<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Smallest singular value (if computed)
    pub min_singular_value: Option<F>,

    /// Largest singular value (if computed)
    pub max_singular_value: Option<F>,

    /// Matrix rank estimate
    pub estimated_rank: Option<usize>,

    /// Whether the matrix appears to be symmetric
    pub is_symmetric: bool,

    /// Whether the matrix appears to be positive definite
    pub is_positive_definite: Option<bool>,

    /// Machine epsilon for the floating point type
    pub machine_epsilon: F,
}

impl<F> Default for StabilityDiagnostics<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            min_singular_value: None,
            max_singular_value: None,
            estimated_rank: None,
            is_symmetric: false,
            is_positive_definite: None,
            machine_epsilon: machine_epsilon::<F>(),
        }
    }
}

/// Edge case detection report
#[derive(Debug, Clone)]
pub struct EdgeCaseReport<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Whether the matrix has nearly singular condition
    pub is_nearly_singular: bool,

    /// Whether diagonal dominance is satisfied
    pub has_diagonal_dominance: bool,

    /// Number of zero or near-zero diagonal elements
    pub zero_diagonal_count: usize,

    /// Estimated numerical rank
    pub numerical_rank: Option<usize>,

    /// Recommended treatment for the edge case
    pub recommended_treatment: EdgeCaseTreatment,

    /// Additional context about the detected issues
    pub issue_description: String,

    /// Phantom data for type parameter
    _phantom: PhantomData<F>,
}

impl<F> Default for EdgeCaseReport<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            is_nearly_singular: false,
            has_diagonal_dominance: true,
            zero_diagonal_count: 0,
            numerical_rank: None,
            recommended_treatment: EdgeCaseTreatment::None,
            issue_description: "No issues detected".to_string(),
            _phantom: PhantomData,
        }
    }
}

/// Recommended treatment for edge cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeCaseTreatment {
    /// No special treatment needed
    None,
    /// Apply Tikhonov regularization
    TikhonovRegularization,
    /// Use pivoting strategy
    Pivoting,
    /// Apply preconditioning
    Preconditioning,
    /// Switch to iterative solver
    IterativeSolver,
    /// Recommend data preprocessing
    DataPreprocessing,
}

/// Enhanced stability report with solve strategy recommendations
#[derive(Debug, Clone)]
pub struct EnhancedStabilityReport<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Basic condition assessment
    pub condition_report: ConditionReport<F>,

    /// Edge case analysis
    pub edge_case_report: EdgeCaseReport<F>,

    /// Recommended solving strategy
    pub recommended_strategy: SolveStrategy,

    /// Expected convergence information
    pub convergence_info: ConvergenceInfo<F>,

    /// Whether iterative refinement is recommended
    pub needs_iterative_refinement: bool,
}

/// Solver strategy recommendations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolveStrategy {
    /// Direct solve with LU decomposition
    DirectLU,
    /// Direct solve with QR decomposition
    DirectQR,
    /// Iterative solve with CG
    IterativeCG,
    /// Iterative solve with GMRES
    IterativeGMRES,
    /// Regularized solve
    Regularized,
}

/// Convergence information for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<F>
where
    F: Float + FromPrimitive,
{
    /// Expected number of iterations
    pub expected_iterations: usize,

    /// Recommended convergence tolerance
    pub recommended_tolerance: F,

    /// Whether preconditioning is recommended
    pub needs_preconditioning: bool,
}

/// Data points analysis for interpolation edge cases
#[derive(Debug, Clone)]
pub struct DataPointsAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Number of data points
    pub num_points: usize,

    /// Minimum distance between points
    pub min_distance: F,

    /// Maximum distance between points
    pub max_distance: F,

    /// Ratio of max to min distance
    pub distance_ratio: F,

    /// Whether points are collinear (for 2D)
    pub is_collinear: bool,

    /// Clustering score (0.0 = well distributed, 1.0 = highly clustered)
    pub clustering_score: F,

    /// Whether near-linear dependencies exist
    pub has_linear_dependencies: bool,
}

impl<F> Default for DataPointsAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            num_points: 0,
            min_distance: F::zero(),
            max_distance: F::zero(),
            distance_ratio: F::one(),
            is_collinear: false,
            clustering_score: F::zero(),
            has_linear_dependencies: false,
        }
    }
}

/// Function values analysis for interpolation edge cases
#[derive(Debug, Clone)]
pub struct FunctionValuesAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Range of function values
    pub value_range: F,

    /// Whether the function appears smooth
    pub is_smooth: bool,

    /// Smoothness score (higher = smoother)
    pub smoothness_score: F,

    /// Whether the function is monotonic
    pub is_monotonic: bool,

    /// Whether there are extreme outliers
    pub has_outliers: bool,

    /// Recommended smoothing parameter
    pub recommended_smoothing: Option<F>,
}

impl<F> Default for FunctionValuesAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            value_range: F::zero(),
            is_smooth: true,
            smoothness_score: F::one(),
            is_monotonic: false,
            has_outliers: false,
            recommended_smoothing: None,
        }
    }
}

/// Boundary effects analysis
#[derive(Debug, Clone)]
pub struct BoundaryAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Whether boundary effects are significant
    pub has_boundary_effects: bool,

    /// Recommended boundary treatment
    pub recommended_boundary_treatment: BoundaryTreatment,

    /// Extrapolation risk assessment
    pub extrapolation_risk: ExtrapolationRisk,

    /// Phantom data for type parameter
    _phantom: PhantomData<F>,
}

impl<F> Default for BoundaryAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            has_boundary_effects: false,
            recommended_boundary_treatment: BoundaryTreatment::Natural,
            extrapolation_risk: ExtrapolationRisk::Low,
            _phantom: PhantomData,
        }
    }
}

/// Boundary treatment recommendations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryTreatment {
    /// Natural boundary conditions
    Natural,
    /// Clamped boundary conditions
    Clamped,
    /// Periodic boundary conditions
    Periodic,
    /// Custom boundary conditions
    Custom,
}

/// Extrapolation risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtrapolationRisk {
    /// Low risk for extrapolation
    Low,
    /// Medium risk for extrapolation
    Medium,
    /// High risk for extrapolation
    High,
    /// Extrapolation not recommended
    Critical,
}

/// Complete edge case analysis report
#[derive(Debug, Clone)]
pub struct EdgeCaseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Analysis of data point distribution
    pub data_points: DataPointsAnalysis<F>,

    /// Analysis of function values
    pub function_values: FunctionValuesAnalysis<F>,

    /// Boundary effects analysis
    pub boundary: BoundaryAnalysis<F>,

    /// Overall recommendation
    pub overall_recommendation: String,

    /// Whether the problem is solvable with current setup
    pub is_solvable: bool,
}

impl<F> Default for EdgeCaseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            data_points: DataPointsAnalysis::default(),
            function_values: FunctionValuesAnalysis::default(),
            boundary: BoundaryAnalysis::default(),
            overall_recommendation: "No issues detected".to_string(),
            is_solvable: true,
        }
    }
}

/// Get machine epsilon for floating point type
pub fn machine_epsilon<F: Float + FromPrimitive>() -> F {
    match std::mem::size_of::<F>() {
        4 => F::from_f64(f32::EPSILON as f64).unwrap_or_else(|| {
            F::from(f32::EPSILON as f32).unwrap_or_else(|| {
                F::from_f64(1.19e-7).unwrap_or_else(|| F::from(1.19e-7).unwrap())
            })
        }), // f32
        8 => F::from_f64(f64::EPSILON).unwrap_or_else(|| {
            F::from(f64::EPSILON).unwrap_or_else(|| {
                F::from_f64(2.22e-16).unwrap_or_else(|| F::from(2.22e-16).unwrap())
            })
        }), // f64
        _ => F::from_f64(2.22e-16).unwrap_or_else(|| F::from(2.22e-16).unwrap()), // Default to f64 epsilon
    }
}

/// Classify stability level based on condition number
pub fn classify_stability<F>(condition_number: F) -> StabilityLevel
where
    F: Float + FromPrimitive,
{
    let excellent_threshold = F::from_f64(1e12).unwrap_or_else(|| F::from(1e12).unwrap());
    let good_threshold = F::from_f64(1e14).unwrap_or_else(|| F::from(1e14).unwrap());
    let marginal_threshold = F::from_f64(1e16).unwrap_or_else(|| F::from(1e16).unwrap());

    if condition_number < excellent_threshold {
        StabilityLevel::Excellent
    } else if condition_number < good_threshold {
        StabilityLevel::Good
    } else if condition_number < marginal_threshold {
        StabilityLevel::Marginal
    } else {
        StabilityLevel::Poor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_machine_epsilon() {
        let eps_f32 = machine_epsilon::<f32>();
        let eps_f64 = machine_epsilon::<f64>();

        assert!(eps_f32 > 0.0);
        assert!(eps_f64 > 0.0);
        assert!(eps_f32 > eps_f64 as f32); // f32 has larger epsilon
    }

    #[test]
    fn test_stability_classification() {
        assert_eq!(classify_stability(1e10_f64), StabilityLevel::Excellent);
        assert_eq!(classify_stability(1e13_f64), StabilityLevel::Good);
        assert_eq!(classify_stability(1e15_f64), StabilityLevel::Marginal);
        assert_eq!(classify_stability(1e17_f64), StabilityLevel::Poor);
    }

    #[test]
    fn test_default_diagnostics() {
        let diag: StabilityDiagnostics<f64> = StabilityDiagnostics::default();
        assert_eq!(diag.min_singular_value, None);
        assert_eq!(diag.max_singular_value, None);
        assert!(!diag.is_symmetric);
        assert!(diag.machine_epsilon > 0.0);
    }

    #[test]
    fn test_edge_case_report_default() {
        let report: EdgeCaseReport<f64> = EdgeCaseReport::default();
        assert!(!report.is_nearly_singular);
        assert!(report.has_diagonal_dominance);
        assert_eq!(report.zero_diagonal_count, 0);
        assert_eq!(report.recommended_treatment, EdgeCaseTreatment::None);
    }
}
