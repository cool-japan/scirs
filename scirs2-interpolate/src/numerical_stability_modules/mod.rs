//! Numerical stability monitoring and assessment modules
//!
//! This module provides comprehensive numerical stability analysis for interpolation
//! algorithms, organized into focused submodules for maintainability and performance.

/// Core type definitions and basic utilities
pub mod types;

/// Condition number estimation algorithms
pub mod condition;

/// Regularization and stabilization methods
pub mod regularization;

/// Edge case detection and analysis
pub mod edge_cases;

/// Enhanced linear solvers with stability monitoring
pub mod solvers;

/// Data analysis utilities for interpolation problems
pub mod data_analysis;

// Re-export public API
pub use types::{
    classify_stability, machine_epsilon, BoundaryAnalysis, BoundaryTreatment, ConditionReport,
    ConvergenceInfo, DataPointsAnalysis, EdgeCaseAnalysis, EdgeCaseReport, EdgeCaseTreatment,
    EnhancedStabilityReport, ExtrapolationRisk, FunctionValuesAnalysis, SolveStrategy,
    StabilityDiagnostics, StabilityLevel,
};

pub use condition::{
    assess_matrix_condition, check_diagonal_dominance, check_symmetry,
    count_zero_diagonal_elements, estimate_condition_number,
};

pub use regularization::{
    apply_adaptive_regularization, apply_preconditioning, apply_tikhonov_regularization,
    check_safe_division, detect_edge_cases, iterative_refinement, safe_reciprocal,
    PreconditionerType,
};

pub use edge_cases::{
    analyze_boundary_conditions, analyze_data_points, analyze_function_values,
    analyze_interpolation_edge_cases,
};

pub use solvers::{solve_with_enhanced_monitoring, solve_with_stability_monitoring};

pub use data_analysis::{
    analyze_interpolation_data, analyze_sampling_density, suggest_data_based_regularization,
    DataScalingAnalysis, DenoisingStrategy, InterpolationDataReport, InterpolationMethod,
    InterpolationMethodRecommendation, NoiseAnalysis, SamplingDensityAnalysis, SamplingStrategy,
};

/// Convenience re-exports for common usage patterns
pub mod prelude {
    pub use super::condition::assess_matrix_condition;
    pub use super::data_analysis::analyze_interpolation_data;
    pub use super::edge_cases::analyze_interpolation_edge_cases;
    pub use super::regularization::{apply_tikhonov_regularization, check_safe_division};
    pub use super::solvers::solve_with_stability_monitoring;
    pub use super::types::{machine_epsilon, ConditionReport, StabilityLevel};
}

/// Module version and feature information
pub mod info {
    /// Module version
    pub const VERSION: &str = "0.1.0";

    /// List of available features
    pub const FEATURES: &[&str] = &[
        "condition_estimation",
        "svd_based_analysis",
        "norm_based_analysis",
        "tikhonov_regularization",
        "adaptive_regularization",
        "edge_case_detection",
        "iterative_refinement",
        "enhanced_solvers",
        "stability_monitoring",
        "data_analysis",
        "interpolation_recommendations",
        "noise_analysis",
        "scaling_analysis",
        "sampling_density_analysis",
    ];

    /// Get feature availability
    pub fn has_feature(feature: &str) -> bool {
        FEATURES.contains(&feature)
    }

    /// Get module information
    pub fn module_info() -> String {
        format!(
            "Numerical Stability modules v{}\nFeatures: {}\nModules: types, condition, regularization, edge_cases, solvers, data_analysis",
            VERSION,
            FEATURES.len()
        )
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    #[test]
    fn test_complete_stability_workflow() {
        // Test a complete numerical stability workflow using the modular API
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0, 1.0]);

        // Assess matrix condition
        let condition_report = assess_matrix_condition(&matrix.view()).unwrap();
        assert!(condition_report.is_well_conditioned);

        // Solve with stability monitoring
        let solution = solve_with_stability_monitoring(&matrix.view(), &rhs.view()).unwrap();

        // Verify solution quality
        assert_eq!(solution.len(), 3);
        for val in solution.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_edge_case_detection_workflow() {
        // Test edge case detection and analysis
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, // Collinear points
            ],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]); // y = x^2

        let analysis = analyze_interpolation_edge_cases(&points.view(), &values.view()).unwrap();

        assert!(analysis.data_points.is_collinear);
        assert!(analysis.function_values.is_smooth);
        assert!(analysis.is_solvable);
        assert!(
            analysis.overall_recommendation.contains("collinear")
                || analysis
                    .overall_recommendation
                    .contains("No significant issues")
        );
    }

    #[test]
    fn test_regularization_workflow() {
        // Test regularization on a poorly conditioned matrix
        let ill_conditioned = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0,
                1.0,
                1.0,
                1.0 + 1e-15, // Nearly singular
            ],
        )
        .unwrap();

        let condition_report = assess_matrix_condition(&ill_conditioned.view()).unwrap();
        assert!(!condition_report.is_well_conditioned);
        assert!(condition_report.recommended_regularization.is_some());

        // Apply regularization
        let reg_param = condition_report.recommended_regularization.unwrap();
        let regularized =
            apply_tikhonov_regularization(&ill_conditioned.view(), reg_param).unwrap();

        // Check that regularization improved the matrix
        let regularized_condition = assess_matrix_condition(&regularized.view()).unwrap();
        assert!(regularized_condition.condition_number < condition_report.condition_number);
    }

    #[test]
    fn test_data_analysis_workflow() {
        // Test comprehensive data analysis
        let points = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0, 4.0, 16.0, 5.0, 25.0, // y = x^2
            ],
        )
        .unwrap();
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);

        let report = analyze_interpolation_data(&points.view(), &values.view()).unwrap();

        assert!(!report.data_points.is_collinear);
        assert!(report.function_values.is_smooth);
        assert!(report.function_values.is_monotonic);
        assert!(!report.noise_analysis.is_noisy);

        // Check that a reasonable interpolation method is recommended
        assert!(matches!(
            report.interpolation_method_recommendation.primary_method,
            InterpolationMethod::CubicSpline
                | InterpolationMethod::BSpline
                | InterpolationMethod::Linear
        ));
    }

    #[test]
    fn test_enhanced_solver_workflow() {
        // Test enhanced solver with comprehensive monitoring
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, // Well-conditioned tridiagonal
            ],
        )
        .unwrap();
        let rhs = Array1::from_vec(vec![5.0, 6.0, 5.0]);

        let (solution, report) =
            solve_with_enhanced_monitoring(&matrix.view(), &rhs.view()).unwrap();

        // Check solution quality
        assert_eq!(solution.len(), 3);
        for val in solution.iter() {
            assert!(val.is_finite());
        }

        // Check that analysis was comprehensive
        assert!(report.condition_report.is_well_conditioned);
        assert!(!report.edge_case_report.is_nearly_singular);
        assert!(matches!(
            report.recommended_strategy,
            SolveStrategy::DirectLU | SolveStrategy::IterativeCG
        ));
        assert!(report.convergence_info.expected_iterations > 0);
    }

    #[test]
    fn test_sampling_density_analysis() {
        // Test sampling density analysis
        let sparse_points =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 0.0, 10.0, 0.0]).unwrap();

        let analysis = analyze_sampling_density(&sparse_points.view(), 0.1).unwrap();

        assert!(analysis.current_density > 0.0);
        assert!(analysis.recommended_density > 0.0);
        // For sparse sampling with high accuracy requirement, should suggest more points
        assert!(analysis.suggested_additional_points > 0 || analysis.density_adequate);
    }

    #[test]
    fn test_noise_analysis() {
        // Test with clean data
        let clean_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let clean_analysis =
            data_analysis::analyze_noise_characteristics(&clean_values.view()).unwrap();
        assert!(!clean_analysis.is_noisy);
        assert_eq!(clean_analysis.denoising_strategy, DenoisingStrategy::None);

        // Test with noisy data
        let noisy_values = Array1::from_vec(vec![0.1, 0.9, 2.1, 2.95, 4.05]);
        let noisy_analysis =
            data_analysis::analyze_noise_characteristics(&noisy_values.view()).unwrap();
        assert!(noisy_analysis.estimated_noise_level > 0.0);
        assert!(noisy_analysis.signal_noise_ratio < f64::INFINITY);
    }

    #[test]
    fn test_prelude_imports() {
        use super::prelude::*;

        // Test that prelude imports work correctly
        let matrix = Array2::<f64>::eye(2);
        let report = assess_matrix_condition(&matrix.view()).unwrap();
        assert_eq!(report.stability_level, StabilityLevel::Excellent);

        let eps = machine_epsilon::<f64>();
        assert!(eps > 0.0);
        assert!(eps < 1e-10);
    }
}
