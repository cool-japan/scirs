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
//! # Modular Organization
//!
//! This module has been refactored into focused submodules:
//! - `types`: Core type definitions and basic utilities
//! - `condition`: Condition number estimation algorithms
//! - `regularization`: Regularization and stabilization methods
//! - `edge_cases`: Edge case detection and analysis
//! - `solvers`: Enhanced linear solvers with stability monitoring
//! - `data_analysis`: Data analysis utilities for interpolation problems
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array2;
//! use scirs2_interpolate::numerical_stability::{assess_matrix_condition, StabilityLevel};
//!
//! // Assess the condition of a matrix
//! let matrix = Array2::<f64>::eye(3);
//! let report = assess_matrix_condition(&matrix.view()).unwrap();
//!
//! match report.stability_level {
//!     StabilityLevel::Excellent => println!("Matrix is well-conditioned"),
//!     StabilityLevel::Poor => println!("Consider regularization: {:?}",
//!                                     report.recommended_regularization),
//!     _ => println!("Condition number: {:.2e}", report.condition_number),
//! }
//! ```

// Import the modular implementation
use crate::numerical_stability_modules;

// Re-export the public API
pub use crate::numerical_stability_modules::{
    analyze_boundary_conditions, analyze_data_points, analyze_function_values,
    analyze_interpolation_data, analyze_interpolation_edge_cases, analyze_sampling_density,
    apply_adaptive_regularization, apply_preconditioning, apply_tikhonov_regularization,
    assess_matrix_condition, check_diagonal_dominance, check_safe_division, check_symmetry,
    classify_stability, count_zero_diagonal_elements, detect_edge_cases, estimate_condition_number,
    iterative_refinement, machine_epsilon, safe_reciprocal, solve_with_enhanced_monitoring,
    solve_with_stability_monitoring, suggest_data_based_regularization, BoundaryAnalysis,
    BoundaryTreatment, ConditionReport, ConvergenceInfo, DataPointsAnalysis, DataScalingAnalysis,
    DenoisingStrategy, EdgeCaseAnalysis, EdgeCaseReport, EdgeCaseTreatment,
    EnhancedStabilityReport, ExtrapolationRisk, FunctionValuesAnalysis, InterpolationDataReport,
    InterpolationMethod, InterpolationMethodRecommendation, NoiseAnalysis, PreconditionerType,
    SamplingDensityAnalysis, SamplingStrategy, SolveStrategy, StabilityDiagnostics, StabilityLevel,
};

// Convenience re-exports for common patterns
pub use crate::numerical_stability_modules::prelude::*;

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign};

// Type aliases for convenience
pub type ConditionReportF32 = ConditionReport<f32>;
pub type ConditionReportF64 = ConditionReport<f64>;

/// Quick condition assessment for f64 matrices
pub fn quick_condition_check(matrix: &ArrayView2<f64>) -> InterpolateResult<bool> {
    let report = assess_matrix_condition(matrix)?;
    Ok(report.is_well_conditioned)
}

/// Quick stability solve for f64 systems
pub fn quick_stable_solve(
    matrix: &ArrayView2<f64>,
    rhs: &ArrayView1<f64>,
) -> InterpolateResult<Array1<f64>> {
    solve_with_stability_monitoring(matrix, rhs)
}

/// Module information and version
pub mod info {
    pub use crate::numerical_stability_modules::info::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_quick_condition_check() {
        let well_conditioned = Array2::<f64>::eye(3);
        assert!(quick_condition_check(&well_conditioned.view()).unwrap());

        let ill_conditioned =
            Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0 + 1e-15]).unwrap();
        assert!(!quick_condition_check(&ill_conditioned.view()).unwrap());
    }

    #[test]
    fn test_quick_stable_solve() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);

        let solution = quick_stable_solve(&matrix.view(), &rhs.view()).unwrap();
        assert_eq!(solution.len(), 2);

        // Verify solution: Ax should equal b
        let verification = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((verification[i] - rhs[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_modular_api_integration() {
        // Test the complete workflow using the modular API
        let matrix =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0])
                .unwrap();
        let rhs = Array1::from_vec(vec![5.0, 6.0, 5.0]);

        // Test condition assessment
        let condition_report = assess_matrix_condition(&matrix.view()).unwrap();
        assert!(condition_report.is_well_conditioned);
        assert_eq!(condition_report.stability_level, StabilityLevel::Excellent);

        // Test enhanced solving
        let (solution, enhanced_report) =
            solve_with_enhanced_monitoring(&matrix.view(), &rhs.view()).unwrap();
        assert_eq!(solution.len(), 3);
        assert!(enhanced_report.condition_report.is_well_conditioned);

        // Test edge case detection
        let edge_report = detect_edge_cases(&matrix.view()).unwrap();
        assert!(!edge_report.is_nearly_singular);
        assert!(edge_report.has_diagonal_dominance);
    }

    #[test]
    fn test_interpolation_data_analysis() {
        // Test interpolation-specific analysis
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0, 4.0, 16.0],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);

        let analysis = analyze_interpolation_edge_cases(&points.view(), &values.view()).unwrap();
        assert!(analysis.is_solvable);
        assert!(analysis.function_values.is_smooth);
        assert!(analysis.function_values.is_monotonic);

        let data_report = analyze_interpolation_data(&points.view(), &values.view()).unwrap();
        assert!(matches!(
            data_report
                .interpolation_method_recommendation
                .primary_method,
            InterpolationMethod::CubicSpline
                | InterpolationMethod::BSpline
                | InterpolationMethod::Linear
        ));
    }

    #[test]
    fn test_regularization_workflow() {
        // Test regularization on a poorly conditioned matrix
        let ill_conditioned =
            Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0 + 1e-15]).unwrap();

        let condition_report = assess_matrix_condition(&ill_conditioned.view()).unwrap();
        assert!(!condition_report.is_well_conditioned);
        assert!(condition_report.recommended_regularization.is_some());

        // Apply Tikhonov regularization
        let reg_param = condition_report.recommended_regularization.unwrap();
        let regularized =
            apply_tikhonov_regularization(&ill_conditioned.view(), reg_param).unwrap();

        let regularized_report = assess_matrix_condition(&regularized.view()).unwrap();
        assert!(regularized_report.condition_number < condition_report.condition_number);
    }

    #[test]
    fn test_safe_arithmetic() {
        // Test safe division
        assert!(check_safe_division(1.0, 2.0).is_ok());
        assert!(check_safe_division(1.0, 1e-20).is_err());

        // Test safe reciprocal
        assert!(safe_reciprocal(2.0).is_ok());
        assert_eq!(safe_reciprocal(2.0).unwrap(), 0.5);
        assert!(safe_reciprocal(1e-20).is_err());
    }

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
    fn test_data_analysis_features() {
        // Test comprehensive data analysis features
        let uniform_points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let smooth_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let data_analysis = analyze_data_points(&uniform_points.view()).unwrap();
        assert!(!data_analysis.is_collinear);
        assert!(data_analysis.clustering_score < 0.5);

        let function_analysis = analyze_function_values(&smooth_values.view()).unwrap();
        assert!(function_analysis.is_smooth);
        assert!(function_analysis.is_monotonic);
        assert!(!function_analysis.has_outliers);

        let boundary_analysis =
            analyze_boundary_conditions(&uniform_points.view(), &smooth_values.view()).unwrap();
        assert!(matches!(
            boundary_analysis.extrapolation_risk,
            ExtrapolationRisk::Low | ExtrapolationRisk::Medium
        ));
    }

    #[test]
    fn test_sampling_density_analysis() {
        let sparse_points = Array2::from_shape_vec((3, 1), vec![0.0, 5.0, 10.0]).unwrap();
        let analysis = analyze_sampling_density(&sparse_points.view(), 0.1).unwrap();

        assert!(analysis.current_density > 0.0);
        assert!(analysis.recommended_density >= analysis.current_density);
        assert!(matches!(
            analysis.sampling_strategy,
            SamplingStrategy::Uniform | SamplingStrategy::Adaptive
        ));
    }

    #[test]
    fn test_preconditioning() {
        use super::numerical_stability_modules::regularization::PreconditionerType;

        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 9.0]).unwrap();
        let (precond, inv_precond) =
            apply_preconditioning(&matrix.view(), PreconditionerType::Diagonal).unwrap();

        assert_eq!(precond.nrows(), 2);
        assert_eq!(precond.ncols(), 2);
        assert_eq!(inv_precond.nrows(), 2);
        assert_eq!(inv_precond.ncols(), 2);

        // Diagonal preconditioning should have specific values
        assert!((precond[(0, 0)] - 2.0).abs() < 1e-10); // sqrt(4) = 2
        assert!((precond[(1, 1)] - 3.0).abs() < 1e-10); // sqrt(9) = 3
    }
}
