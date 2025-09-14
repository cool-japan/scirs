//! Numerical Stability Monitoring and Condition Assessment
//!
//! This module provides utilities for monitoring numerical stability in interpolation
//! algorithms, particularly for matrix operations, linear solvers, and condition
//! number estimation.
//!
//! # Overview
//!
//! Numerical stability is critical for reliable interpolation, especially when:
//! - Solving linear systems with potentially ill-conditioned matrices
//! - Computing matrix factorizations and eigenvalue decompositions
//! - Performing division operations that might involve small numbers
//! - Working with matrices that approach singularity
//!
//! This module provides tools to:
//! - Estimate condition numbers efficiently
//! - Classify stability levels based on condition numbers
//! - Suggest appropriate regularization parameters
//! - Monitor for numerical issues during computation
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array2;
//! use scirs2__interpolate::numerical_stability::{assess_matrix_condition, StabilityLevel};
//!
//! // Assess the condition of a matrix
//! let matrix = Array2::<f64>::eye(3);
//! let report = assess_matrix_condition(&matrix.view()).unwrap();
//!
//! match report.stability_level {
//!     StabilityLevel::Excellent => println!("Matrix is well-conditioned"),
//!     StabilityLevel::Poor => println!("Consider regularization: {:?}",
//!                                     report.recommended_regularization),
//!     _ => println!("Condition number: {:.2e}", report._conditionnumber),
//! }
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use statrs::statistics::Statistics;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

/// Condition number and stability assessment report
#[derive(Debug, Clone)]
pub struct ConditionReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    /// Estimated condition number of the matrix
    pub _conditionnumber: F,

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
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
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
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
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

/// Get machine epsilon for floating point type
#[allow(dead_code)]
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

/// Assess the numerical condition of a matrix
#[allow(dead_code)]
pub fn assess_matrix_condition<F>(matrix: &ArrayView2<F>) -> InterpolateResult<ConditionReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square _matrix".to_string(),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            object: "condition assessment".to_string(),
        });
    }

    let n = matrix.nrows();
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Cannot assess condition of empty _matrix".to_string(),
        });
    }

    let mut diagnostics = StabilityDiagnostics {
        is_symmetric: check_symmetry(matrix),
        ..Default::default()
    };

    // Estimate condition number
    let _conditionnumber = estimate_conditionnumber(matrix, &mut diagnostics)?;

    // Classify stability level
    let stability_level = classify_stability(_conditionnumber);

    // Determine if well-conditioned
    let is_well_conditioned = matches!(
        stability_level,
        StabilityLevel::Excellent | StabilityLevel::Good
    );

    // Suggest regularization if needed
    let recommended_regularization = if !is_well_conditioned {
        Some(suggest_regularization(_conditionnumber, &diagnostics))
    } else {
        None
    };

    Ok(ConditionReport {
        _conditionnumber,
        is_well_conditioned,
        recommended_regularization,
        stability_level,
        diagnostics,
    })
}

/// Check if a matrix is symmetric within numerical tolerance
#[allow(dead_code)]
fn check_symmetry<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    let tol = F::from_f64(1e-12).unwrap_or_else(|| {
        machine_epsilon::<F>() * F::from(1e6).unwrap_or_else(|| F::from(1000000).unwrap())
    });

    for i in 0..n {
        for j in 0..i {
            let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
            if diff > tol {
                return false;
            }
        }
    }
    true
}

/// Estimate condition number using different methods based on availability
#[allow(dead_code)]
fn estimate_conditionnumber<F>(
    matrix: &ArrayView2<F>,
    #[cfg_attr(not(feature = "linalg"), allow(unused_variables))]
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    // Try SVD-based condition number first (most accurate)
    #[cfg(feature = "linalg")]
    {
        match estimate_condition_svd(matrix, diagnostics) {
            Ok(cond) => return Ok(cond),
            Err(_) => {
                // Fall back to other methods if SVD fails
            }
        }
    }

    // Fall back to norm-based estimation
    estimate_condition_norm_based(matrix)
}

/// Estimate condition number using SVD (requires linalg feature)
#[cfg(feature = "linalg")]
#[allow(dead_code)]
fn estimate_condition_svd<F>(
    matrix: &ArrayView2<F>,
    diagnostics: &mut StabilityDiagnostics<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    #[cfg(feature = "linalg")]
    {
        use scirs2_linalg::svd;

        // Convert to f64 for SVD computation
        let matrix_f64 = matrix.mapv(|x| x.to_f64().unwrap_or(0.0));

        match svd(&matrix_f64.view(), false, None) {
            Ok((_, singular_values_)) => {
                if singular_values.is_empty() {
                    return Err(InterpolateError::ComputationError(
                        "SVD returned empty singular values".to_string(),
                    ));
                }

                let max_sv = singular_values[0];
                let min_sv = singular_values[singular_values.len() - 1];

                // Update diagnostics
                diagnostics.max_singular_value = Some(
                    F::from_f64(max_sv)
                        .unwrap_or_else(|| F::from(max_sv as f32).unwrap_or(F::zero())),
                );
                diagnostics.min_singular_value = Some(
                    F::from_f64(min_sv)
                        .unwrap_or_else(|| F::from(min_sv as f32).unwrap_or(F::zero())),
                );

                // Estimate rank
                let eps = f64::EPSILON * max_sv * (matrix.nrows() as f64).sqrt();
                let rank = singular_values.iter().filter(|&&sv| sv > eps).count();
                diagnostics.estimated_rank = Some(rank);

                // Compute condition number
                if min_sv > f64::EPSILON {
                    Ok(F::from_f64(max_sv / min_sv).unwrap_or_else(|| F::infinity()))
                } else {
                    Ok(F::infinity()) // Singular matrix
                }
            }
            Err(_) => Err(InterpolateError::ComputationError(
                "SVD computation failed".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "linalg"))]
    {
        // Fallback to norm-based estimation when linalg feature is not available
        estimate_condition_norm_based(matrix)
    }
}

/// Fallback condition number estimation using matrix norms
#[allow(dead_code)]
fn estimate_condition_norm_based<F>(matrix: &ArrayView2<F>) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    // Estimate condition number using maximum and minimum eigenvalue estimates
    let n = matrix.nrows();

    // Estimate largest and smallest eigenvalues using Gershgorin circles
    let mut min_gershgorin = F::infinity();
    let mut max_gershgorin = F::neg_infinity();

    for i in 0..n {
        let diagonal = matrix[[i, i]];
        let off_diagonal_sum = (0..n)
            .filter(|&j| j != i)
            .map(|j| matrix[[i, j]].abs())
            .fold(F::zero(), |a, b| a + b);

        let center = diagonal.abs();
        let radius = off_diagonal_sum;

        // Lower and upper bounds of Gershgorin disk
        let lower_bound = center - radius;
        let upper_bound = center + radius;

        if lower_bound > F::zero() && lower_bound < min_gershgorin {
            min_gershgorin = lower_bound;
        }

        if upper_bound > max_gershgorin {
            max_gershgorin = upper_bound;
        }
    }

    // Estimate condition number as ratio of max to min eigenvalue estimates
    if min_gershgorin.is_finite() && min_gershgorin > F::zero() && max_gershgorin.is_finite() {
        Ok(max_gershgorin / min_gershgorin)
    } else {
        // Conservative estimate for potentially ill-conditioned matrices
        Ok(F::from_f64(1e16)
            .unwrap_or_else(|| F::from(1e16 as f32).unwrap_or_else(|| F::infinity())))
    }
}

/// Classify stability level based on condition number
#[allow(dead_code)]
fn classify_stability<F>(_conditionnumber: F) -> StabilityLevel
where
    F: Float + FromPrimitive,
{
    let threshold_1e12 = F::from_f64(1e12).unwrap_or_else(|| {
        F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000i64).unwrap())
    });
    let threshold_1e14 = F::from_f64(1e14).unwrap_or_else(|| {
        F::from(1e14 as f32).unwrap_or_else(|| F::from(100000000000000i64).unwrap())
    });
    let threshold_1e16 = F::from_f64(1e16).unwrap_or_else(|| {
        F::from(1e16 as f32).unwrap_or_else(|| F::from(10000000000000000i64).unwrap())
    });

    if _conditionnumber < threshold_1e12 {
        StabilityLevel::Excellent
    } else if _conditionnumber < threshold_1e14 {
        StabilityLevel::Good
    } else if _conditionnumber < threshold_1e16 {
        StabilityLevel::Marginal
    } else {
        StabilityLevel::Poor
    }
}

/// Suggest regularization parameter based on condition number and diagnostics
#[allow(dead_code)]
fn suggest_regularization<F>(_conditionnumber: F, diagnostics: &StabilityDiagnostics<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let machine_eps = diagnostics.machine_epsilon;

    // Base regularization on condition _number and machine epsilon
    let base_reg = machine_eps * _conditionnumber.sqrt();

    // Adjust based on minimum singular value if available
    if let Some(min_sv) = diagnostics.min_singular_value {
        if min_sv < machine_eps {
            // Very small singular value, need stronger regularization
            base_reg
                * F::from_f64(100.0).unwrap_or_else(|| {
                    F::from(100.0 as f32).unwrap_or_else(|| F::from(100).unwrap())
                })
        } else {
            // Moderate regularization
            base_reg
                * F::from_f64(10.0)
                    .unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10).unwrap()))
        }
    } else {
        // Conservative regularization when no singular value information
        base_reg
            * F::from_f64(1000.0)
                .unwrap_or_else(|| F::from(1000.0 as f32).unwrap_or_else(|| F::from(1000).unwrap()))
    }
}

/// Check if a division operation is numerically safe
#[allow(dead_code)]
pub fn check_safe_division<F>(numerator: F, denominator: F) -> InterpolateResult<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::fmt::LowerExp,
{
    let eps = machine_epsilon::<F>();
    let safe_threshold = eps
        * F::from_f64(1e6)
            .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000).unwrap()));

    if denominator.abs() < safe_threshold {
        Err(InterpolateError::NumericalError(format!(
            "Division by near-zero value: {} / {} (threshold: {:.2e})",
            numerator, denominator, safe_threshold
        )))
    } else {
        Ok(numerator / denominator)
    }
}

/// Check if reciprocal operation is numerically safe
#[allow(dead_code)]
pub fn safe_reciprocal<F>(value: F) -> InterpolateResult<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::fmt::LowerExp,
{
    check_safe_division(F::one(), value)
}

/// Apply Tikhonov regularization to a matrix
#[allow(dead_code)]
pub fn apply_tikhonov_regularization<F>(
    matrix: &mut Array2<F>,
    regularization: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();

    if matrix.ncols() != n {
        return Err(InterpolateError::ShapeMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}x{}", n, matrix.ncols()),
            object: "Tikhonov regularization".to_string(),
        });
    }

    // Validate regularization parameter
    if regularization < F::zero() {
        return Err(InterpolateError::InvalidInput {
            message: "Regularization parameter must be non-negative".to_string(),
        });
    }

    // Add regularization to diagonal
    for i in 0..n {
        matrix[[i, i]] += regularization;
    }

    Ok(())
}

/// Apply adaptive regularization based on matrix characteristics
#[allow(dead_code)]
pub fn apply_adaptive_regularization<F>(
    matrix: &mut Array2<F>,
    condition_report: &ConditionReport<F>,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let regularization = match condition_report.stability_level {
        StabilityLevel::Excellent | StabilityLevel::Good => F::zero(),
        StabilityLevel::Marginal => {
            // Use moderate regularization
            let base_reg =
                machine_epsilon::<F>() * F::from_f64(1e8).unwrap_or(F::from(100000000).unwrap());
            base_reg * condition_report._conditionnumber.sqrt()
        }
        StabilityLevel::Poor => {
            // Use stronger regularization
            condition_report
                .recommended_regularization
                .unwrap_or_else(|| {
                    machine_epsilon::<F>()
                        * F::from_f64(1e12).unwrap_or(F::from(1000000000000i64).unwrap())
                })
        }
    };

    if regularization > F::zero() {
        apply_tikhonov_regularization(matrix, regularization)?;
    }

    Ok(regularization)
}

/// Enhanced edge case detection for numerical stability
#[allow(dead_code)]
pub fn detect_edge_cases<F>(
    matrix: &ArrayView2<F>,
    rhs: Option<&ArrayView1<F>>,
) -> EdgeCaseReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let mut report = EdgeCaseReport::default();
    let n = matrix.nrows();

    // Check for extreme values
    let mut min_val = F::infinity();
    let mut max_val = F::neg_infinity();
    let mut zero_count = 0;
    let mut inf_count = 0;
    let mut nan_count = 0;

    for &val in matrix.iter() {
        if val.is_nan() {
            nan_count += 1;
        } else if val.is_infinite() {
            inf_count += 1;
        } else if val == F::zero() {
            zero_count += 1;
        } else {
            min_val = min_val.min(val.abs());
            max_val = max_val.max(val.abs());
        }
    }

    report.has_nan_values = nan_count > 0;
    report.has_infinite_values = inf_count > 0;
    report.has_extreme_values = if !min_val.is_infinite() && !max_val.is_infinite() {
        let dynamic_range = max_val / min_val;
        dynamic_range
            > F::from_f64(1e15)
                .unwrap_or(F::from(1e15 as f32).unwrap_or(F::from(1000000000000000i64).unwrap()))
    } else {
        false
    };

    // Check matrix structure
    report.is_diagonal_dominant = check_diagonal_dominance(matrix);
    report.zero_diagonal_count = count_zero_diagonal_elements(matrix);
    report.sparsity_ratio = (zero_count as f64) / (n * n) as f64;

    // Check RHS if provided
    if let Some(rhs_vec) = rhs {
        report.rhs_has_extreme_values = rhs_vec.iter().any(|&x| {
            x.is_nan()
                || x.is_infinite()
                || x.abs()
                    > F::from_f64(1e100).unwrap_or(F::from_f64(1e100).unwrap_or(F::max_value()))
        });
    }

    report
}

/// Check if matrix is diagonally dominant
#[allow(dead_code)]
fn check_diagonal_dominance<F>(matrix: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    for i in 0..n {
        let diagonal = matrix[[i, i]].abs();
        let off_diagonal_sum: F = (0..n)
            .filter(|&j| j != i)
            .map(|j| matrix[[i, j]].abs())
            .fold(F::zero(), |acc, x| acc + x);

        if diagonal <= off_diagonal_sum {
            return false;
        }
    }
    true
}

/// Count zero elements on the diagonal
#[allow(dead_code)]
fn count_zero_diagonal_elements<F>(matrix: &ArrayView2<F>) -> usize
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    let zero_threshold =
        machine_epsilon::<F>() * F::from_f64(1e6).unwrap_or(F::from(1000000).unwrap());

    (0..n)
        .filter(|&i| matrix[[i, i]].abs() < zero_threshold)
        .count()
}

/// Report on edge cases and numerical issues
#[derive(Debug, Clone)]
pub struct EdgeCaseReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    /// Whether matrix contains NaN values
    pub has_nan_values: bool,
    /// Whether matrix contains infinite values
    pub has_infinite_values: bool,
    /// Whether matrix has extreme dynamic range
    pub has_extreme_values: bool,
    /// Whether matrix is diagonally dominant
    pub is_diagonal_dominant: bool,
    /// Number of zero diagonal elements
    pub zero_diagonal_count: usize,
    /// Ratio of zero elements to total elements
    pub sparsity_ratio: f64,
    /// Whether RHS vector has extreme values
    pub rhs_has_extreme_values: bool,
    /// Phantom data to use the type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F> Default for EdgeCaseReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    fn default() -> Self {
        Self {
            has_nan_values: false,
            has_infinite_values: false,
            has_extreme_values: false,
            is_diagonal_dominant: false,
            zero_diagonal_count: 0,
            sparsity_ratio: 0.0,
            rhs_has_extreme_values: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Enhanced solve with comprehensive stability monitoring and edge case detection
#[allow(dead_code)]
pub fn solve_with_enhanced_monitoring<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
) -> InterpolateResult<(Array1<F>, EnhancedStabilityReport<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + std::fmt::LowerExp
        + 'static,
{
    // Perform comprehensive edge case detection
    let edge_case_report = detect_edge_cases(&matrix.view(), Some(&rhs.view()));

    // Early exit for severe issues
    if edge_case_report.has_nan_values || edge_case_report.has_infinite_values {
        return Err(InterpolateError::NumericalError(
            "Matrix or RHS contains NaN or infinite values".to_string(),
        ));
    }

    // Assess matrix condition
    let condition_report = assess_matrix_condition(&matrix.view())?;

    // Create enhanced report
    let mut enhanced_report = EnhancedStabilityReport {
        condition_report: condition_report.clone(),
        edge_case_report,
        applied_regularization: F::zero(),
        solve_strategy: SolveStrategy::Direct,
        convergence_info: None,
    };

    // Determine solving strategy based on stability assessment
    let solution = if enhanced_report.edge_case_report.has_extreme_values
        || matches!(condition_report.stability_level, StabilityLevel::Poor)
    {
        enhanced_report.solve_strategy = SolveStrategy::RegularizedIterative;

        let mut working_matrix = matrix.clone();
        let regularization = apply_adaptive_regularization(&mut working_matrix, &condition_report)?;
        enhanced_report.applied_regularization = regularization;

        // Try iterative refinement for better accuracy
        solve_with_iterative_refinement(&working_matrix, rhs, &mut enhanced_report)?
    } else if enhanced_report.edge_case_report.zero_diagonal_count > 0 {
        enhanced_report.solve_strategy = SolveStrategy::PivotedLU;
        solve_system(matrix, rhs)?
    } else {
        enhanced_report.solve_strategy = SolveStrategy::Direct;
        solve_system(matrix, rhs)?
    };

    Ok((solution, enhanced_report))
}

/// Monitor and report numerical issues during matrix solve
#[allow(dead_code)]
pub fn solve_with_stability_monitoring<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
) -> InterpolateResult<(Array1<F>, ConditionReport<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + std::fmt::LowerExp
        + 'static,
{
    let (solution, enhanced_report) = solve_with_enhanced_monitoring(matrix, rhs)?;
    Ok((solution, enhanced_report.condition_report))
}

/// Solve linear system with iterative refinement for enhanced accuracy
#[allow(dead_code)]
fn solve_with_iterative_refinement<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
    report: &mut EnhancedStabilityReport<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + std::fmt::LowerExp
        + 'static,
{
    let mut solution = solve_system(matrix, rhs)?;
    let max_iterations = 5;
    let tolerance = machine_epsilon::<F>() * F::from_f64(1e6).unwrap_or(F::from(1000000).unwrap());

    let mut convergence_info = ConvergenceInfo {
        iterations: 0,
        final_residual: F::infinity(),
        converged: false,
    };

    for iteration in 0..max_iterations {
        // Compute residual: r = b - A*x
        let mut residual = rhs.clone();
        for i in 0..matrix.nrows() {
            let mut ax_i = F::zero();
            for j in 0..matrix.ncols() {
                ax_i += matrix[[i, j]] * solution[j];
            }
            residual[i] -= ax_i;
        }

        // Compute residual norm
        let residual_norm = residual
            .iter()
            .fold(F::zero(), |acc, &x| acc + x * x)
            .sqrt();
        convergence_info.final_residual = residual_norm;
        convergence_info.iterations = iteration + 1;

        // Check convergence
        if residual_norm < tolerance {
            convergence_info.converged = true;
            break;
        }

        // Solve for correction: A * delta_x = residual
        if let Ok(correction) = solve_system(matrix, &residual) {
            // Update solution: x = x + delta_x
            for i in 0..solution.len() {
                solution[i] += correction[i];
            }
        } else {
            break; // Stop if correction solve fails
        }
    }

    report.convergence_info = Some(convergence_info);
    Ok(solution)
}

/// Enhanced stability report with comprehensive diagnostics
#[derive(Debug, Clone)]
pub struct EnhancedStabilityReport<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    /// Basic condition assessment
    pub condition_report: ConditionReport<F>,
    /// Edge case detection results
    pub edge_case_report: EdgeCaseReport<F>,
    /// Applied regularization amount
    pub applied_regularization: F,
    /// Solving strategy used
    pub solve_strategy: SolveStrategy,
    /// Convergence information for iterative methods
    pub convergence_info: Option<ConvergenceInfo<F>>,
}

/// Strategy used for solving the linear system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolveStrategy {
    /// Direct solving without modifications
    Direct,
    /// LU factorization with pivoting
    PivotedLU,
    /// Regularized system with iterative refinement
    RegularizedIterative,
    /// Specialized method for structured matrices
    Structured,
}

/// Information about iterative convergence
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<F>
where
    F: Float + FromPrimitive + Debug + Display,
{
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: F,
    /// Whether the method converged
    pub converged: bool,
}

/// Internal function to solve linear system
#[allow(dead_code)]
fn solve_system<F>(matrix: &Array2<F>, rhs: &Array1<F>) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    #[cfg(feature = "linalg")]
    {
        use scirs2_linalg::solve;

        // Convert to f64 for solve
        let matrix_f64 = matrix.mapv(|x| x.to_f64().unwrap_or(0.0));
        let rhs_f64 = rhs.mapv(|x| x.to_f64().unwrap_or(0.0));

        match solve(&matrix_f64.view(), &rhs_f64.view(), None) {
            Ok(solution_f64) => {
                let solution = solution_f64.mapv(|x| {
                    F::from_f64(x).unwrap_or_else(|| F::from(x as f32).unwrap_or(F::zero()))
                });
                Ok(solution)
            }
            Err(_) => Err(InterpolateError::ComputationError(
                "Linear system solve failed".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "linalg"))]
    {
        // Fallback: simple Gaussian elimination for small systems
        if matrix.nrows() <= 3 {
            gaussian_elimination_small(matrix, rhs)
        } else {
            Err(InterpolateError::ComputationError(
                "Linear algebra solve requires 'linalg' feature for large systems".to_string(),
            ))
        }
    }
}

/// Simple Gaussian elimination for small systems (fallback)
#[cfg(not(feature = "linalg"))]
#[allow(dead_code)]
fn gaussian_elimination_small<F>(
    matrix: &Array2<F>,
    rhs: &Array1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut b = rhs.clone();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }

        // Check for near-zero pivot
        if a[[i, i]].abs()
            < machine_epsilon::<F>()
                * F::from_f64(1e6).unwrap_or_else(|| {
                    F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000).unwrap())
                })
        {
            return Err(InterpolateError::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                let a_ij = a[[i, j]];
                a[[k, j]] -= factor * a_ij;
            }
            let b_i = b[i];
            b[k] -= factor * b_i;
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum += a[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Analyze distances between data points to detect clustering issues
#[allow(dead_code)]
fn analyze_point_distances<F>(points: &ArrayView2<F>) -> InterpolateResult<(F, F, usize)>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let npoints = points.nrows();
    let mut min_dist = F::infinity();
    let mut max_dist = F::zero();
    let mut clustered_count = 0;

    let cluster_threshold = machine_epsilon::<F>()
        * F::from_f64(1e6)
            .unwrap_or_else(|| F::from(1e6 as f32).unwrap_or_else(|| F::from(1000000).unwrap()));

    for i in 0..npoints {
        for j in (i + 1)..npoints {
            // Compute Euclidean distance
            let mut dist_sq = F::zero();
            for k in 0..points.ncols() {
                let diff = points[[i, k]] - points[[j, k]];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();

            if dist < min_dist {
                min_dist = dist;
            }
            if dist > max_dist {
                max_dist = dist;
            }

            if dist < cluster_threshold {
                clustered_count += 1;
            }
        }
    }

    Ok((min_dist, max_dist, clustered_count))
}

/// Check for near-linear dependencies in data points
#[allow(dead_code)]
fn check_near_linear_dependencies<F>(points: &ArrayView2<F>) -> InterpolateResult<bool>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let npoints = points.nrows();
    let dim = points.ncols();

    // For fewer points than dimensions + 1, no dependencies are possible
    if npoints <= dim {
        return Ok(false);
    }

    // Check rank of the point matrix (after centering)
    let mut centeredpoints = points.to_owned();

    // Center the points
    for j in 0..dim {
        let mean = points.column(j).mean().unwrap_or(F::zero());
        for i in 0..npoints {
            centeredpoints[[i, j]] -= mean;
        }
    }

    // Try to assess rank using condition number
    if dim <= 10 && npoints <= 100 {
        // For small matrices, we can check more carefully
        let gram_matrix = compute_gram_matrix(&centeredpoints.view());
        let condition_report = assess_matrix_condition(&gram_matrix.view())?;

        // If condition number is very high, likely have linear dependencies
        Ok(condition_report._conditionnumber
            > F::from_f64(1e14).unwrap_or_else(|| {
                F::from(1e14 as f32).unwrap_or_else(|| F::from(100000000000000i64).unwrap())
            }))
    } else {
        // For larger matrices, use a simpler heuristic
        Ok(false) // Conservative: assume no dependencies for large datasets
    }
}

/// Compute Gram matrix A^T A for rank analysis
#[allow(dead_code)]
fn compute_gram_matrix<F>(points: &ArrayView2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let npoints = points.nrows();
    let dim = points.ncols();
    let mut gram = Array2::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..dim {
            let mut sum = F::zero();
            for k in 0..npoints {
                sum += points[[k, i]] * points[[k, j]];
            }
            gram[[i, j]] = sum;
        }
    }

    gram
}

/// Suggest regularization parameter based on data characteristics
#[allow(dead_code)]
fn suggest_data_based_regularization<F>(_min_distance: F, distanceratio: F) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let machine_eps = machine_epsilon::<F>();

    // Base regularization on minimum _distance
    let distance_based = _min_distance * machine_eps.sqrt();

    // Scale by _distance _ratio (more ill-conditioned data needs more regularization)
    let ratio_factor = if distanceratio
        > F::from_f64(1e12).unwrap_or_else(|| {
            F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000i64).unwrap())
        }) {
        F::from_f64(1000.0)
            .unwrap_or_else(|| F::from(1000.0 as f32).unwrap_or_else(|| F::from(1000).unwrap()))
    } else if distanceratio
        > F::from_f64(1e8)
            .unwrap_or_else(|| F::from(1e8 as f32).unwrap_or_else(|| F::from(100000000).unwrap()))
    {
        F::from_f64(100.0)
            .unwrap_or_else(|| F::from(100.0 as f32).unwrap_or_else(|| F::from(100).unwrap()))
    } else {
        F::from_f64(10.0)
            .unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10).unwrap()))
    };

    distance_based * ratio_factor
}

/// Analyze boundary effects for interpolation stability  
#[allow(dead_code)]
fn analyze_boundary_effects<F>(points: &ArrayView2<F>) -> InterpolateResult<BoundaryAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let dim = points.ncols();
    let npoints = points.nrows();

    if npoints == 0 {
        return Ok(BoundaryAnalysis {
            boundary_has_issues: false,
            extrapolation_stability: StabilityLevel::Good,
            has_natural_boundaries: true,
            boundary_gradient_norm: F::zero(),
        });
    }

    // Find bounding box
    let mut min_coords = vec![F::infinity(); dim];
    let mut max_coords = vec![F::neg_infinity(); dim];

    for i in 0..npoints {
        for j in 0..dim {
            let coord = points[[i, j]];
            if coord < min_coords[j] {
                min_coords[j] = coord;
            }
            if coord > max_coords[j] {
                max_coords[j] = coord;
            }
        }
    }

    // Compute minimum distance to boundary
    let mut min_boundary_dist = F::infinity();
    for i in 0..npoints {
        for j in 0..dim {
            let coord = points[[i, j]];
            let dist_to_min = coord - min_coords[j];
            let dist_to_max = max_coords[j] - coord;
            let boundary_dist = dist_to_min.min(dist_to_max);
            if boundary_dist < min_boundary_dist {
                min_boundary_dist = boundary_dist;
            }
        }
    }

    // Simple heuristic for boundary distribution
    let domain_sizes: Vec<F> = (0..dim).map(|j| max_coords[j] - min_coords[j]).collect();
    let avg_domain_size = domain_sizes.iter().fold(F::zero(), |a, &b| a + b)
        / F::from_usize(dim)
            .unwrap_or_else(|| F::from(dim as f32).unwrap_or_else(|| F::from(dim).unwrap()));
    let well_distributed = min_boundary_dist
        > avg_domain_size
            / F::from_f64(10.0)
                .unwrap_or_else(|| F::from(10.0 as f32).unwrap_or_else(|| F::from(10).unwrap()));

    // Extrapolation stability heuristic
    let extrapolation_stable = well_distributed && npoints > 2 * dim;

    Ok(BoundaryAnalysis {
        boundary_has_issues: !well_distributed,
        extrapolation_stability: if extrapolation_stable {
            StabilityLevel::Good
        } else {
            StabilityLevel::Marginal
        },
        has_natural_boundaries: well_distributed,
        boundary_gradient_norm: min_boundary_dist,
    })
}

/// Comprehensive edge case analysis for interpolation stability
#[allow(dead_code)]
pub fn analyze_interpolation_edge_cases<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    method_name: &str,
) -> InterpolateResult<EdgeCaseAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let mut analysis = EdgeCaseAnalysis::default();
    analysis.method_name = method_name.to_string();

    // Analyze points for numerical issues
    analysis.pointsanalysis = analyze_datapoints(points)?;
    analysis.valuesanalysis = analyze_function_values(values)?;

    // Check for interpolation-specific edge cases
    analysis.boundaryanalysis = analyze_boundary_conditions(points, values)?;

    // Assess overall stability
    analysis.overall_stability = assess_overall_interpolation_stability(&analysis);

    // Generate recommendations
    analysis.recommendations = generate_stability_recommendations(&analysis);

    Ok(analysis)
}

/// Data structure for comprehensive edge case analysis
#[derive(Debug, Clone)]
pub struct EdgeCaseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Name of the interpolation method being analyzed
    pub method_name: String,

    /// Analysis of input points
    pub pointsanalysis: DataPointsAnalysis<F>,

    /// Analysis of function values
    pub valuesanalysis: FunctionValuesAnalysis<F>,

    /// Analysis of boundary conditions and edge behavior
    pub boundaryanalysis: BoundaryAnalysis<F>,

    /// Overall stability assessment
    pub overall_stability: StabilityLevel,

    /// Specific recommendations for improving stability
    pub recommendations: Vec<String>,

    /// Timestamp of analysis
    pub analysis_timestamp: std::time::Instant,
}

impl<F> Default for EdgeCaseAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            method_name: String::new(),
            pointsanalysis: DataPointsAnalysis::default(),
            valuesanalysis: FunctionValuesAnalysis::default(),
            boundaryanalysis: BoundaryAnalysis::default(),
            overall_stability: StabilityLevel::Good,
            recommendations: Vec::new(),
            analysis_timestamp: std::time::Instant::now(),
        }
    }
}

/// Analysis of input data points
#[derive(Debug, Clone)]
pub struct DataPointsAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Number of data points
    pub point_count: usize,

    /// Dimensionality of the points
    pub dimension: usize,

    /// Minimum distance between any two points
    pub min_point_distance: F,

    /// Maximum distance between any two points
    pub max_point_distance: F,

    /// Whether points are collinear (for 2D+)
    pub are_collinear: bool,

    /// Whether points contain duplicates
    pub has_duplicates: bool,

    /// Distribution uniformity score (0.0 = uniform, 1.0 = clustered)
    pub clustering_score: F,

    /// Convex hull area/volume ratio to bounding box
    pub convex_hull_ratio: F,
}

impl<F> Default for DataPointsAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            point_count: 0,
            dimension: 0,
            min_point_distance: F::zero(),
            max_point_distance: F::zero(),
            are_collinear: false,
            has_duplicates: false,
            clustering_score: F::zero(),
            convex_hull_ratio: F::one(),
        }
    }
}

/// Analysis of function values
#[derive(Debug, Clone)]
pub struct FunctionValuesAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Range of function values (max - min)
    pub value_range: F,

    /// Standard deviation of values
    pub value_std_dev: F,

    /// Whether values contain extreme outliers
    pub has_outliers: bool,

    /// Smoothness indicator (based on second differences)
    pub smoothness_score: F,

    /// Whether function appears monotonic
    pub is_monotonic: bool,

    /// Estimated noise level in the data
    pub noise_level: F,
}

impl<F> Default for FunctionValuesAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            value_range: F::zero(),
            value_std_dev: F::zero(),
            has_outliers: false,
            smoothness_score: F::one(),
            is_monotonic: false,
            noise_level: F::zero(),
        }
    }
}

/// Analysis of boundary conditions and edge behavior
#[derive(Debug, Clone)]
pub struct BoundaryAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    /// Whether boundary points have special characteristics
    pub boundary_has_issues: bool,

    /// Estimated extrapolation stability
    pub extrapolation_stability: StabilityLevel,

    /// Whether data has natural boundary conditions
    pub has_natural_boundaries: bool,

    /// Boundary gradient estimate (for edge behavior prediction)
    pub boundary_gradient_norm: F,
}

impl<F> Default for BoundaryAnalysis<F>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            boundary_has_issues: false,
            extrapolation_stability: StabilityLevel::Good,
            has_natural_boundaries: true,
            boundary_gradient_norm: F::zero(),
        }
    }
}

/// Analyze data points for numerical issues
#[allow(dead_code)]
fn analyze_datapoints<F>(points: &ArrayView2<F>) -> InterpolateResult<DataPointsAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let npoints = points.nrows();
    let dimension = points.ncols();

    if npoints == 0 {
        return Err(InterpolateError::empty_data("point analysis"));
    }

    let mut analysis = DataPointsAnalysis {
        point_count: npoints,
        dimension,
        ..Default::default()
    };

    // Calculate minimum and maximum distances
    let mut min_dist = F::infinity();
    let mut max_dist = F::zero();
    let mut has_duplicates = false;

    for i in 0..npoints {
        for j in (i + 1)..npoints {
            let dist_sq = points
                .row(i)
                .iter()
                .zip(points.row(j).iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, x| acc + x);

            let dist = dist_sq.sqrt();

            if dist < min_dist {
                min_dist = dist;
            }
            if dist > max_dist {
                max_dist = dist;
            }

            // Check for near-duplicates
            if dist < machine_epsilon::<F>() * F::from(1000.0).unwrap_or(F::from(1000).unwrap()) {
                has_duplicates = true;
            }
        }
    }

    analysis.min_point_distance = min_dist;
    analysis.max_point_distance = max_dist;
    analysis.has_duplicates = has_duplicates;

    // Check for collinearity (simplified for 2D case)
    if dimension == 2 && npoints >= 3 {
        analysis.are_collinear = check_collinearity_2d(points);
    }

    // Calculate clustering score (simplified variance-based measure)
    analysis.clustering_score = calculate_clustering_score(points);

    Ok(analysis)
}

/// Analyze function values for stability issues
#[allow(dead_code)]
fn analyze_function_values<F>(
    values: &ArrayView1<F>,
) -> InterpolateResult<FunctionValuesAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n == 0 {
        return Err(InterpolateError::empty_data("values analysis"));
    }

    let mut analysis = FunctionValuesAnalysis::default();

    // Calculate basic statistics
    let min_val = values.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max_val = values.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    analysis.value_range = max_val - min_val;

    // Calculate mean and standard deviation
    let mean = values.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();
    let variance = values
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(n).unwrap();
    analysis.value_std_dev = variance.sqrt();

    // Check for outliers (values more than 3 standard deviations from mean)
    let threshold = analysis.value_std_dev * F::from(3.0).unwrap();
    analysis.has_outliers = values.iter().any(|&x| (x - mean).abs() > threshold);

    // Estimate smoothness (based on second differences for 1D case)
    if n >= 3 {
        analysis.smoothness_score = estimate_smoothness(values);
    }

    // Check monotonicity
    analysis.is_monotonic = check_monotonicity(values);

    // Estimate noise level
    analysis.noise_level = estimate_noise_level(values);

    Ok(analysis)
}

/// Analyze boundary conditions
#[allow(dead_code)]
fn analyze_boundary_conditions<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
) -> InterpolateResult<BoundaryAnalysis<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut analysis = BoundaryAnalysis::default();

    // Simple boundary analysis (identify extreme points)
    let npoints = points.nrows();
    let dimension = points.ncols();

    if npoints < 3 {
        analysis.boundary_has_issues = true;
        analysis.extrapolation_stability = StabilityLevel::Poor;
        return Ok(analysis);
    }

    // Find boundary points (simplified - just find min/max in each dimension)
    let mut boundary_indices = Vec::new();

    for dim in 0..dimension {
        let column = points.column(dim);
        let min_idx = column
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);
        let max_idx = column
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);

        if let Some(idx) = min_idx {
            boundary_indices.push(idx);
        }
        if let Some(idx) = max_idx {
            boundary_indices.push(idx);
        }
    }

    boundary_indices.sort_unstable();
    boundary_indices.dedup();

    // Analyze boundary values for extreme behavior
    let boundary_values: Vec<F> = boundary_indices.iter().map(|&idx| values[idx]).collect();

    if boundary_values.len() >= 2 {
        let boundary_range = boundary_values.iter().fold(F::zero(), |acc, &val| {
            acc.max((val - boundary_values[0]).abs())
        });

        let total_range = analysis.boundary_gradient_norm;
        if total_range > F::zero() && boundary_range / total_range > F::from(0.8).unwrap() {
            analysis.boundary_has_issues = true;
        }
    }

    // Estimate extrapolation stability based on boundary behavior
    analysis.extrapolation_stability = if analysis.boundary_has_issues {
        StabilityLevel::Marginal
    } else {
        StabilityLevel::Good
    };

    Ok(analysis)
}

/// Check if 2D points are collinear
#[allow(dead_code)]
fn check_collinearity_2d<F>(points: &ArrayView2<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = points.nrows();
    if n < 3 {
        return true;
    }

    let tolerance = machine_epsilon::<F>() * F::from(1000.0).unwrap_or(F::from(1000).unwrap());

    // Use cross product to check collinearity
    for i in 2..n {
        let v1x = points[[1, 0]] - points[[0, 0]];
        let v1y = points[[1, 1]] - points[[0, 1]];
        let v2x = points[[i, 0]] - points[[0, 0]];
        let v2y = points[[i, 1]] - points[[0, 1]];

        let cross_product = v1x * v2y - v1y * v2x;
        if cross_product.abs() > tolerance {
            return false;
        }
    }

    true
}

/// Calculate clustering score for point distribution
#[allow(dead_code)]
fn calculate_clustering_score<F>(points: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = points.nrows();
    if n < 2 {
        return F::zero();
    }

    // Calculate variance of nearest neighbor distances
    let mut nn_distances = Vec::new();

    for i in 0..n {
        let mut min_dist = F::infinity();
        for j in 0..n {
            if i != j {
                let dist_sq = points
                    .row(i)
                    .iter()
                    .zip(points.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, x| acc + x);
                let dist = dist_sq.sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        if min_dist.is_finite() {
            nn_distances.push(min_dist);
        }
    }

    if nn_distances.is_empty() {
        return F::zero();
    }

    // Calculate coefficient of variation
    let mean = nn_distances.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from(nn_distances.len()).unwrap();

    if mean <= F::zero() {
        return F::one(); // Maximum clustering
    }

    let variance = nn_distances
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(nn_distances.len()).unwrap();

    let cv = variance.sqrt() / mean;
    cv.min(F::one()) // Normalize to [0, 1]
}

/// Estimate smoothness of function values
#[allow(dead_code)]
fn estimate_smoothness<F>(values: &ArrayView1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n < 3 {
        return F::one();
    }

    // Calculate second differences
    let mut second_diffs = Vec::new();
    for i in 1..(n - 1) {
        let second_diff = values[i + 1] - F::from(2.0).unwrap() * values[i] + values[i - 1];
        second_diffs.push(second_diff.abs());
    }

    // Smoothness is inverse of average second difference (normalized)
    let avg_second_diff = second_diffs.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from(second_diffs.len()).unwrap();

    let value_range = values.iter().fold(F::zero(), |acc, &x| {
        acc.max(x - values.iter().fold(F::infinity(), |a, &b| a.min(b)))
    });

    if value_range <= F::zero() {
        return F::one();
    }

    let normalized_smoothness = F::one() / (F::one() + avg_second_diff / value_range);
    normalized_smoothness
}

/// Check if values are monotonic
#[allow(dead_code)]
fn check_monotonicity<F>(values: &ArrayView1<F>) -> bool
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n < 2 {
        return true;
    }

    let tolerance = machine_epsilon::<F>() * F::from(100.0).unwrap_or(F::from(100).unwrap());

    // Check for monotonic increasing
    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..n {
        let diff = values[i] - values[i - 1];
        if diff < -tolerance {
            increasing = false;
        }
        if diff > tolerance {
            decreasing = false;
        }
    }

    increasing || decreasing
}

/// Estimate noise level in function values
#[allow(dead_code)]
fn estimate_noise_level<F>(values: &ArrayView1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = values.len();
    if n < 3 {
        return F::zero();
    }

    // Estimate noise using high-frequency components (first differences)
    let mut first_diffs = Vec::new();
    for i in 1..n {
        first_diffs.push((values[i] - values[i - 1]).abs());
    }

    // Use median absolute deviation as robust noise estimate
    first_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_idx = first_diffs.len() / 2;

    if median_idx < first_diffs.len() {
        first_diffs[median_idx] * F::from(1.4826).unwrap_or(F::from(1.4826).unwrap())
    // MAD scale factor
    } else {
        F::zero()
    }
}

/// Assess overall interpolation stability based on analysis
#[allow(dead_code)]
fn assess_overall_interpolation_stability<F>(analysis: &EdgeCaseAnalysis<F>) -> StabilityLevel
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut stability_factors = Vec::new();

    // Data points stability factors
    if analysis.pointsanalysis.has_duplicates {
        stability_factors.push(StabilityLevel::Poor);
    }
    if analysis.pointsanalysis.are_collinear {
        stability_factors.push(StabilityLevel::Marginal);
    }
    if analysis.pointsanalysis.clustering_score > F::from(0.8).unwrap() {
        stability_factors.push(StabilityLevel::Marginal);
    }

    // Function values stability factors
    if analysis.valuesanalysis.has_outliers {
        stability_factors.push(StabilityLevel::Marginal);
    }
    if analysis.valuesanalysis.smoothness_score < F::from(0.3).unwrap() {
        stability_factors.push(StabilityLevel::Poor);
    }

    // Boundary analysis factors
    if analysis.boundaryanalysis.boundary_has_issues {
        stability_factors.push(analysis.boundaryanalysis.extrapolation_stability);
    }

    // Return worst stability level found
    stability_factors
        .into_iter()
        .min()
        .unwrap_or(StabilityLevel::Excellent)
}

/// Generate stability improvement recommendations
#[allow(dead_code)]
fn generate_stability_recommendations<F>(analysis: &EdgeCaseAnalysis<F>) -> Vec<String>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let mut recommendations = Vec::new();

    // Data points recommendations
    if analysis.pointsanalysis.has_duplicates {
        recommendations.push("Remove duplicate or nearly duplicate data points".to_string());
    }

    if analysis.pointsanalysis.are_collinear {
        recommendations.push("Add data points outside the current linear arrangement".to_string());
    }

    if analysis.pointsanalysis.clustering_score > F::from(0.7).unwrap() {
        recommendations.push("Add data points in sparse regions for better coverage".to_string());
    }

    if analysis.pointsanalysis.point_count < 5 {
        recommendations
            .push("Consider adding more data points for robust interpolation".to_string());
    }

    // Function values recommendations
    if analysis.valuesanalysis.has_outliers {
        recommendations
            .push("Consider robust interpolation methods or outlier removal".to_string());
    }

    if analysis.valuesanalysis.smoothness_score < F::from(0.5).unwrap() {
        recommendations
            .push("Data appears noisy - consider smoothing or regularization".to_string());
    }

    if analysis.valuesanalysis.noise_level
        > analysis.valuesanalysis.value_range * F::from(0.1).unwrap()
    {
        recommendations
            .push("High noise level detected - consider denoising preprocessing".to_string());
    }

    // Boundary recommendations
    if analysis.boundaryanalysis.boundary_has_issues {
        recommendations.push("Boundary conditions may cause extrapolation issues".to_string());
    }

    // Method-specific recommendations
    match analysis.method_name.as_str() {
        "rbf" | "enhanced_rbf" => {
            if analysis.valuesanalysis.has_outliers {
                recommendations.push(
                    "For RBF interpolation, consider robust kernels or outlier preprocessing"
                        .to_string(),
                );
            }
        }
        "kriging" | "enhanced_kriging" => {
            if analysis.pointsanalysis.clustering_score > F::from(0.8).unwrap() {
                recommendations.push(
                    "Kriging works best with well-distributed points - consider spatial design"
                        .to_string(),
                );
            }
        }
        "spline" | "bspline" => {
            if !analysis.valuesanalysis.is_monotonic && analysis.valuesanalysis.has_outliers {
                recommendations.push(
                    "Consider constrained splines for monotonic or bounded interpolation"
                        .to_string(),
                );
            }
        }
        _ => {}
    }

    recommendations
}

/// Early numerical warning system for proactive issue detection
#[allow(dead_code)]
pub fn early_numerical_warning_system<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    method_name: &str,
) -> InterpolateResult<Vec<String>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let mut warnings = Vec::new();

    // Quick checks for immediate issues
    if points.nrows() != values.len() {
        warnings.push("CRITICAL: Mismatch between number of points and values".to_string());
        return Ok(warnings);
    }

    if points.is_empty() || values.is_empty() {
        warnings.push("CRITICAL: Empty input data".to_string());
        return Ok(warnings);
    }

    // Check for NaN or infinite values
    for (i, point) in points.outer_iter().enumerate() {
        for (j, &coord) in point.iter().enumerate() {
            if !coord.is_finite() {
                warnings.push(format!(
                    "WARNING: Non-finite coordinate at point {}[{}]: {}",
                    i, j, coord
                ));
            }
        }
    }

    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            warnings.push(format!("WARNING: Non-finite value at index {}: {}", i, val));
        }
    }

    // Check for extremely small point separations
    let min_separation = calculate_minimum_point_separation(points);
    let machine_eps = machine_epsilon::<F>();

    if min_separation < machine_eps * F::from(1000.0).unwrap_or(F::from(1000).unwrap()) {
        warnings
            .push("WARNING: Very small point separations may cause numerical issues".to_string());
    }

    // Check for extreme value ranges
    let value_range = calculate_value_range(values);
    let max_abs_value = values.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));

    if max_abs_value > F::zero() {
        let relative_range = value_range / max_abs_value;
        if relative_range > F::from(1e12).unwrap_or(F::from(1e12 as f32).unwrap_or(F::infinity())) {
            warnings
                .push("WARNING: Extreme value range may cause numerical instability".to_string());
        }
    }

    // Method-specific warnings
    match method_name {
        "polynomial" | "barycentric" => {
            if points.nrows() > 15 {
                warnings.push(
                    "WARNING: High-degree polynomial interpolation may be unstable".to_string(),
                );
            }
        }
        "rbf" => {
            if points.nrows() > 10000 {
                warnings.push(
                    "WARNING: Large RBF systems may be computationally expensive".to_string(),
                );
            }
        }
        _ => {}
    }

    Ok(warnings)
}

/// Calculate minimum separation between any two points
#[allow(dead_code)]
fn calculate_minimum_point_separation<F>(points: &ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = points.nrows();
    let mut min_sep = F::infinity();

    for i in 0..n {
        for j in (i + 1)..n {
            let sep_sq = points
                .row(i)
                .iter()
                .zip(points.row(j).iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, x| acc + x);

            let sep = sep_sq.sqrt();
            if sep < min_sep {
                min_sep = sep;
            }
        }
    }

    min_sep
}

/// Calculate range of function values
#[allow(dead_code)]
fn calculate_value_range<F>(values: &ArrayView1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    if values.is_empty() {
        return F::zero();
    }

    let min_val = values.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max_val = values.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

    max_val - min_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_machine_epsilon() {
        let eps_f32: f32 = machine_epsilon();
        let eps_f64: f64 = machine_epsilon();

        assert!(eps_f32 > 0.0);
        assert!(eps_f64 > 0.0);
        assert!(eps_f32 > eps_f64 as f32); // f32 has larger epsilon
    }

    #[test]
    fn test_condition_assessment_identity() {
        let matrix = Array2::<f64>::eye(3);
        let report =
            assess_matrix_condition(&matrix.view()).expect("Failed to assess matrix condition");

        assert_relative_eq!(report._conditionnumber, 1.0, epsilon = 1e-10);
        assert!(report.is_well_conditioned);
        assert_eq!(report.stability_level, StabilityLevel::Excellent);
        assert!(report.recommended_regularization.is_none());
    }

    #[test]
    fn test_condition_assessment_ill_conditioned() {
        let mut matrix = Array2::eye(3);
        matrix[[2, 2]] = 1e-16; // Make it ill-conditioned

        let report = assess_matrix_condition(&matrix.view()).unwrap();

        assert!(report._conditionnumber > 1e12);
        assert!(!report.is_well_conditioned);
        assert!(matches!(report.stability_level, StabilityLevel::Poor));
        assert!(report.recommended_regularization.is_some());
    }

    #[test]
    fn test_symmetry_check() {
        let symmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 3.0])
            .expect("Failed to create symmetric matrix");
        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Failed to create asymmetric matrix");

        assert!(check_symmetry(&symmetric.view()));
        assert!(!check_symmetry(&asymmetric.view()));
    }

    #[test]
    fn test_safe_division() {
        assert!(check_safe_division(1.0, 2.0).is_ok());
        assert!(check_safe_division(1.0, 1e-20).is_err());

        assert_eq!(check_safe_division(6.0, 2.0).unwrap(), 3.0);
    }

    #[test]
    fn test_safe_reciprocal() {
        assert!(safe_reciprocal(2.0).is_ok());
        assert!(safe_reciprocal(1e-20).is_err());

        assert_relative_eq!(safe_reciprocal(4.0).unwrap(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_tikhonov_regularization() {
        let mut matrix = Array2::zeros((3, 3));
        matrix[[0, 0]] = 1.0;
        matrix[[1, 1]] = 2.0;
        matrix[[2, 2]] = 3.0;

        apply_tikhonov_regularization(&mut matrix, 0.1)
            .expect("Failed to apply Tikhonov regularization");

        assert_relative_eq!(matrix[[0, 0]], 1.1, epsilon = 1e-10);
        assert_relative_eq!(matrix[[1, 1]], 2.1, epsilon = 1e-10);
        assert_relative_eq!(matrix[[2, 2]], 3.1, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_solve_with_monitoring() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0])
            .expect("Failed to create matrix");
        let rhs = Array1::from_vec(vec![3.0, 2.0]);

        let (solution, report) = solve_with_stability_monitoring(&matrix, &rhs)
            .expect("Failed to solve with stability monitoring");

        assert_relative_eq!(solution[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 1.0, epsilon = 1e-10);
        assert!(report.is_well_conditioned);
    }

    #[test]
    fn test_stability_classification() {
        assert_eq!(classify_stability(1e10_f64), StabilityLevel::Excellent);
        assert_eq!(classify_stability(1e13_f64), StabilityLevel::Good);
        assert_eq!(classify_stability(1e15_f64), StabilityLevel::Marginal);
        assert_eq!(classify_stability(1e17_f64), StabilityLevel::Poor);
    }

    #[test]
    fn test_enhanced_stability() {
        // Test enhanced matrix conditioning
        let matrix = Array2::from_shape_vec((2, 2), vec![1e-15, 0.0, 0.0, 1.0]).unwrap();
        let assessment = assess_enhanced_matrix_condition(&matrix.view());
        assert!(assessment.is_ok());

        // Test adaptive regularization
        let regularization = compute_adaptive_regularization(1e-10_f64, 1e15_f64);
        assert!(regularization > 0.0);
    }
}

/// Enhanced matrix condition assessment with improved diagnostics
#[allow(dead_code)]
pub fn assess_enhanced_matrix_condition<F>(
    matrix: &ArrayView2<F>,
) -> InterpolateResult<ConditionReport<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + 'static,
{
    let mut report = assess_matrix_condition(matrix)?;

    // Enhanced diagnostics for better stability assessment
    let enhanced_diagnostics = compute_enhanced_diagnostics(matrix)?;

    // Update recommendation based on enhanced analysis
    if enhanced_diagnostics.has_tiny_eigenvalues {
        report.recommended_regularization = Some(
            enhanced_diagnostics
                .suggested_regularization
                .unwrap_or_else(|| {
                    machine_epsilon::<F>()
                        * F::from_f64(1e6).unwrap_or_else(|| F::from(1000000).unwrap())
                }),
        );
    }

    // More conservative stability classification for interpolation
    if enhanced_diagnostics.interpolation_risky {
        report.stability_level = match report.stability_level {
            StabilityLevel::Excellent => StabilityLevel::Good,
            StabilityLevel::Good => StabilityLevel::Marginal,
            StabilityLevel::Marginal => StabilityLevel::Poor,
            StabilityLevel::Poor => StabilityLevel::Poor,
        };
    }

    Ok(report)
}

/// Enhanced diagnostic information for interpolation stability
#[derive(Debug, Clone)]
struct EnhancedDiagnostics<F: Float> {
    has_tiny_eigenvalues: bool,
    interpolation_risky: bool,
    suggested_regularization: Option<F>,
}

/// Compute enhanced diagnostics for matrix stability in interpolation context
#[allow(dead_code)]
fn compute_enhanced_diagnostics<F>(
    matrix: &ArrayView2<F>,
) -> InterpolateResult<EnhancedDiagnostics<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign,
{
    let n = matrix.nrows();
    let eps = machine_epsilon::<F>();

    // Check for very small diagonal elements (indicator of instability)
    let mut has_tiny_eigenvalues = false;
    let mut min_diagonal = F::infinity();
    let mut max_diagonal = F::zero();

    for i in 0..n {
        let diag_val = matrix[[i, i]].abs();
        min_diagonal = min_diagonal.min(diag_val);
        max_diagonal = max_diagonal.max(diag_val);

        if diag_val < eps * F::from_f64(1e6).unwrap_or_else(|| F::from(1000000).unwrap()) {
            has_tiny_eigenvalues = true;
        }
    }

    // Check if interpolation is risky based on conditioning
    let diagonal_ratio = if min_diagonal > F::zero() {
        max_diagonal / min_diagonal
    } else {
        F::infinity()
    };

    let interpolation_risky = diagonal_ratio
        > F::from_f64(1e12).unwrap_or_else(|| {
            F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000i64).unwrap())
        });

    // Suggest adaptive regularization
    let suggested_regularization = if has_tiny_eigenvalues || interpolation_risky {
        Some(compute_adaptive_regularization(
            min_diagonal,
            diagonal_ratio,
        ))
    } else {
        None
    };

    Ok(EnhancedDiagnostics {
        has_tiny_eigenvalues,
        interpolation_risky,
        suggested_regularization,
    })
}

/// Compute adaptive regularization parameter based on matrix characteristics
#[allow(dead_code)]
pub fn compute_adaptive_regularization<F>(min_value: F, conditionestimate: F) -> F
where
    F: Float + FromPrimitive + Debug + Display,
{
    let eps = machine_epsilon::<F>();

    // Base regularization on machine epsilon and minimum _value
    let base_reg = eps.sqrt() * min_value.max(eps);

    // Scale by condition number _estimate
    let condition_factor = if conditionestimate
        > F::from_f64(1e15).unwrap_or_else(|| {
            F::from(1e15 as f32).unwrap_or_else(|| F::from(1000000000000000i64).unwrap())
        }) {
        F::from_f64(1000.0).unwrap_or_else(|| F::from(1000).unwrap())
    } else if conditionestimate
        > F::from_f64(1e12).unwrap_or_else(|| {
            F::from(1e12 as f32).unwrap_or_else(|| F::from(1000000000000i64).unwrap())
        })
    {
        F::from_f64(100.0).unwrap_or_else(|| F::from(100).unwrap())
    } else if conditionestimate
        > F::from_f64(1e8)
            .unwrap_or_else(|| F::from(1e8 as f32).unwrap_or_else(|| F::from(100000000).unwrap()))
    {
        F::from_f64(10.0).unwrap_or_else(|| F::from(10).unwrap())
    } else {
        F::one()
    };

    base_reg * condition_factor
}

/// Perform matrix operations with enhanced stability monitoring
#[allow(dead_code)]
pub fn enhanced_matrix_multiply<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + AddAssign + SubAssign + Copy,
{
    if a.ncols() != b.nrows() {
        return Err(InterpolateError::shape_mismatch(
            format!("({}, {})", a.nrows(), b.ncols()),
            format!(
                "({}, {}) x ({}, {})",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            ),
            "matrix multiplication",
        ));
    }

    let (m, n, k) = (a.nrows(), b.ncols(), a.ncols());
    let mut result = Array2::zeros((m, n));

    // Check for potential overflow before computation
    let max_a = a
        .iter()
        .map(|&x| x.abs())
        .fold(F::zero(), |acc, x| acc.max(x));
    let max_b = b
        .iter()
        .map(|&x| x.abs())
        .fold(F::zero(), |acc, x| acc.max(x));
    let max_product =
        max_a * max_b * F::from_usize(k).unwrap_or_else(|| F::from(k as f64).unwrap_or(F::one()));

    if max_product > F::max_value() / F::from_f64(2.0).unwrap_or_else(|| F::from(2).unwrap()) {
        return Err(InterpolateError::NumericalError(
            "Potential overflow in matrix multiplication".to_string(),
        ));
    }

    // Perform multiplication with overflow checking
    for i in 0..m {
        for j in 0..n {
            let mut sum = F::zero();
            for k_idx in 0..k {
                let product = a[[i, k_idx]] * b[[k_idx, j]];

                // Check for overflow in accumulation
                if sum.is_finite() && product.is_finite() && (sum + product).is_infinite() {
                    return Err(InterpolateError::NumericalError(format!(
                        "Overflow in matrix multiplication at ({}, {})",
                        i, j
                    )));
                }

                sum += product;
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}
