//! B-spline modules - organized implementation of B-spline functionality
//!
//! This module provides a comprehensive B-spline implementation organized into
//! focused submodules for maintainability and performance.

/// Core type definitions and workspace management
pub mod types;

/// Core B-spline implementation with basic operations
pub mod core;

/// Advanced evaluation methods and optimization algorithms
pub mod evaluation;

/// Factory functions for creating B-splines with different configurations
pub mod factory;

/// Linear algebra solvers optimized for B-spline computations
pub mod solvers;

// Re-export public API
pub use types::{
    BSplineWorkspace, BSplineWorkspaceBuilder, ExtrapolateMode, WorkspaceConfig,
    WorkspaceMemoryStats, WorkspaceProvider,
};

pub use core::BSpline;

pub use evaluation::EvaluationStats;

pub use factory::{
    generate_knots, make_auto_bspline, make_interp_bspline, make_lsq_bspline,
    make_periodic_bspline, make_smoothing_bspline,
};

pub use solvers::{
    condition_number, lu_decomposition, matrix_multiply, matrix_vector_multiply,
    solve_least_squares, solve_linear_system, solve_multiple_rhs, solve_with_lu, transpose_matrix,
};

/// Convenience re-exports for common usage patterns
pub mod prelude {
    pub use super::core::BSpline;
    pub use super::evaluation::EvaluationStats;
    pub use super::factory::{generate_knots, make_interp_bspline, make_lsq_bspline};
    pub use super::types::{BSplineWorkspace, ExtrapolateMode};
}

/// Module version and feature information
pub mod info {
    /// Module version
    pub const VERSION: &str = "0.1.0";

    /// List of available features
    pub const FEATURES: &[&str] = &[
        "basic_evaluation",
        "fast_recursive",
        "workspace_optimization",
        "batch_processing",
        "adaptive_precision",
        "uncertainty_quantification",
        "parallel_evaluation",
        "memory_statistics",
        "structured_solvers",
        "auto_regularization",
    ];

    /// Get feature availability
    pub fn has_feature(feature: &str) -> bool {
        FEATURES.contains(&feature)
    }

    /// Get module information
    pub fn module_info() -> String {
        format!(
            "B-spline modules v{}\nFeatures: {}\nModules: types, core, evaluation, factory, solvers",
            VERSION,
            FEATURES.len()
        )
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_complete_workflow() {
        // Test a complete B-spline workflow using the modular API
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

        // Create interpolating B-spline
        let spline = make_interp_bspline(
            &x.view(),
            &y.view(),
            3, // cubic
            ExtrapolateMode::Extrapolate,
        );
        assert!(spline.is_ok());

        let spline = spline.unwrap();

        // Test evaluation
        let test_point = 2.5;
        let value = spline.evaluate(test_point);
        assert!(value.is_ok());

        // Test derivative
        let deriv = spline.derivative(test_point, 1);
        assert!(deriv.is_ok());

        // Test integration
        let integral = spline.integrate(0.0, 4.0);
        assert!(integral.is_ok());
    }

    #[test]
    fn test_workspace_optimization() {
        // Test workspace-based evaluation
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 2.0, 1.0];

        let spline =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Extrapolate).unwrap();

        let workspace = BSplineWorkspace::new();

        // Test workspace evaluation
        let value1 = spline.evaluate_with_workspace(1.5, &workspace);
        assert!(value1.is_ok());

        let value2 = spline.evaluate_with_workspace(2.5, &workspace);
        assert!(value2.is_ok());

        // Check that workspace recorded evaluations
        let stats = workspace.get_statistics();
        assert!(stats.evaluation_count >= 2);
    }

    #[test]
    fn test_advanced_evaluation() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 8.0, 27.0, 64.0, 125.0]; // y = x^3

        let spline =
            make_interp_bspline(&x.view(), &y.view(), 3, ExtrapolateMode::Extrapolate).unwrap();

        // Test fast recursive evaluation
        let fast_value = spline.evaluate_fast_recursive(2.5);
        assert!(fast_value.is_ok());

        // Test batch evaluation
        let test_points = array![1.5, 2.5, 3.5];
        let batch_values = spline.evaluate_array(&test_points.view());
        assert!(batch_values.is_ok());
        assert_eq!(batch_values.unwrap().len(), 3);
    }

    #[test]
    fn test_factory_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        // Test knot generation
        let knots = generate_knots(&x.view(), 3, "clamped");
        assert!(knots.is_ok());

        // Test least-squares fitting
        let knots = knots.unwrap();
        let lsq_spline = make_lsq_bspline(
            &x.view(),
            &y.view(),
            &knots.view(),
            3,
            None,
            ExtrapolateMode::Extrapolate,
        );
        assert!(lsq_spline.is_ok());

        // Test automatic spline creation
        let auto_spline =
            make_auto_bspline(&x.view(), &y.view(), 3, 0.1, ExtrapolateMode::Extrapolate);
        assert!(auto_spline.is_ok());
    }

    #[test]
    fn test_solver_integration() {
        // Test that the linear solvers work correctly
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];

        let solution = solve_linear_system(&a.view(), &b.view());
        assert!(solution.is_ok());

        // Test least squares solver
        let a_rect = array![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]; // 3x2 matrix
        let b_rect = array![2.0, 3.0, 4.0];

        let lsq_solution = solve_least_squares(&a_rect.view(), &b_rect.view());
        assert!(lsq_solution.is_ok());
    }
}
