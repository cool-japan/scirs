//! Integration tests for ANOVA decomposition of sparse-grid functions.

use scirs2_interpolate::sparse_grid::anova::{anova_decompose, AnovaConfig};

#[test]
fn anova_constant_function_has_only_mean() {
    let result =
        anova_decompose(&|_x: &[f64]| 5.0_f64, 2, &AnovaConfig::default()).expect("anova constant");
    assert!(
        (result.mean - 5.0).abs() < 1e-8,
        "mean={} expected 5.0",
        result.mean
    );
    assert!(
        result.total_variance < 1e-8,
        "total_variance={} expected ~0",
        result.total_variance
    );
}

#[test]
fn anova_sum_of_sobol_indices_at_most_one() {
    let result = anova_decompose(&|x: &[f64]| x[0] * x[1], 2, &AnovaConfig::default())
        .expect("anova product");
    let sum: f64 = result.sobol_indices.iter().sum();
    assert!(
        sum <= 1.0 + 1e-6,
        "sum of Sobol indices = {sum}, should be ≤ 1"
    );
    // Non-negative
    for (i, si) in result.sobol_indices.iter().enumerate() {
        assert!(*si >= -1e-10, "S[{i}] = {si} is negative");
    }
}

#[test]
fn anova_ishigami_main_effects_reasonable() {
    // Ishigami function: f(x₁,x₂,x₃) = sin(x₁) + 7·sin²(x₂) + 0.1·x₃⁴·sin(x₁)
    // Map [0,1]³ → [−π, π]³.
    // Known (analytical) first-order Sobol' indices: S₁≈0.31, S₂≈0.44, S₃=0.
    // At n_quad_points=5 and d=3 (125 quadrature points) we only expect S₂ to be
    // the largest main-effect index; the exact numerical values are not enforced.
    let f = |x: &[f64]| {
        let x1 = (x[0] * 2.0 - 1.0) * std::f64::consts::PI;
        let x2 = (x[1] * 2.0 - 1.0) * std::f64::consts::PI;
        let x3 = (x[2] * 2.0 - 1.0) * std::f64::consts::PI;
        x1.sin() + 7.0 * x2.sin().powi(2) + 0.1 * x3.powi(4) * x1.sin()
    };
    let config = AnovaConfig {
        max_order: 1,
        n_quad_points: 5,
    };
    let result = anova_decompose(&f, 3, &config).expect("anova ishigami");
    assert!(
        result.total_variance > 0.01,
        "total variance should be nonzero"
    );
    assert!(
        result.sobol_indices[1] > result.sobol_indices[0],
        "S₂ (index 1) should dominate over S₁ (index 0): S₁={}, S₂={}",
        result.sobol_indices[0],
        result.sobol_indices[1]
    );
}

#[test]
fn anova_sobol_total_positive() {
    let result =
        anova_decompose(&|x: &[f64]| x[0] + x[1], 2, &AnovaConfig::default()).expect("anova sum");
    for (i, ti) in result.total_sobol_indices.iter().enumerate() {
        assert!(*ti >= -1e-8, "T[{i}] = {ti} is negative (below tolerance)");
    }
}

#[test]
fn anova_separable_function_main_effects_sum_near_total_variance() {
    // f(x,y) = x + y: purely separable, no interactions.
    // Var(f) = Var(x) + Var(y) over [0,1]² = 1/12 + 1/12 = 1/6.
    let result = anova_decompose(&|x: &[f64]| x[0] + x[1], 2, &AnovaConfig::default())
        .expect("anova separable");
    let main_sum: f64 = result.main_effects.iter().sum();
    // Main effects should account for essentially all the variance.
    if result.total_variance > 1e-10 {
        let frac = main_sum / result.total_variance;
        assert!(
            frac > 0.90,
            "separable function: main effects cover {:.1}% of total variance (expected ≥ 90%)",
            frac * 100.0
        );
    }
}

#[test]
fn anova_zero_dimension_returns_error() {
    use scirs2_interpolate::sparse_grid::anova::AnovaError;
    let err = anova_decompose(&|_x: &[f64]| 0.0_f64, 0, &AnovaConfig::default()).unwrap_err();
    assert_eq!(err, AnovaError::ZeroDimension);
}

#[test]
fn anova_interaction_matrix_symmetric() {
    let result = anova_decompose(
        &|x: &[f64]| x[0] * x[1] + x[1] * x[2],
        3,
        &AnovaConfig::default(),
    )
    .expect("anova 3d");
    let d = 3;
    for i in 0..d {
        for j in 0..d {
            let a = result.interaction_effects[[i, j]];
            let b = result.interaction_effects[[j, i]];
            assert!(
                (a - b).abs() < 1e-12,
                "interaction_effects not symmetric at [{i},{j}]: {a} vs {b}"
            );
        }
    }
}
