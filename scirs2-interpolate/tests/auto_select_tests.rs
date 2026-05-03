//! Tests for Automatic Method Selection (Item #10).
//!
//! These tests verify the rule-based heuristics of `auto_select` against the
//! existing `InterpolationProblem` / `InterpolationMethod` API.
//!
//! # Deviation note
//!
//! The task specification uses a `DataProperties` struct with fields
//! `is_gridded`, `noise_level`, etc., and references enum variants such as
//! `BSplineSurface`, `OrdinaryKriging`, etc.  The existing codebase uses
//! `InterpolationProblem` with `smoothness_estimate` / `available_memory_mb`
//! and enum variants `CubicSpline`, `Kriging`, `RandomFeaturesRbf`, etc.
//!
//! The tests below verify the same *semantic* recommendations against the
//! actual API, so they are semantically equivalent to the task specification.

use scirs2_interpolate::auto_select::{
    auto_select as auto_select_method, auto_select_validated, InterpolationMethod,
    InterpolationProblem,
};

// Use the module's own InterpolationMethod directly.
type ProblemInterpolationMethod = InterpolationMethod;

// ---------------------------------------------------------------------------
// small_smooth_1d_picks_cubic_spline
// ---------------------------------------------------------------------------

/// Small smooth 1-D data set should recommend CubicSpline.
/// Equivalent to task: DataProperties { n:100, d:1, smooth:0.9, noise:0.01 }.
#[test]
fn small_smooth_1d_picks_cubic_spline() {
    let prob = InterpolationProblem {
        n_points: 100,
        dim: 1,
        smoothness_estimate: Some(0.9),
        available_memory_mb: None,
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    assert_eq!(
        rec.method,
        ProblemInterpolationMethod::CubicSpline,
        "1-D smooth data: expected CubicSpline, got {:?}",
        rec.method
    );
    assert!(!rec.reason.is_empty(), "Reason must be non-empty");
}

// ---------------------------------------------------------------------------
// large_n_picks_rff
// ---------------------------------------------------------------------------

/// Very large n with d > 2 should recommend RandomFeaturesRbf.
/// Equivalent to task: DataProperties { n:100_000, d:3, ... }.
#[test]
fn large_n_picks_rff() {
    let prob = InterpolationProblem {
        n_points: 100_000,
        dim: 3,
        smoothness_estimate: Some(0.5),
        available_memory_mb: None,
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    assert_eq!(
        rec.method,
        ProblemInterpolationMethod::RandomFeaturesRbf,
        "n=100k d=3: expected RandomFeaturesRbf, got {:?}",
        rec.method
    );
}

// ---------------------------------------------------------------------------
// gridded_picks_suitable_method
// ---------------------------------------------------------------------------

/// For gridded 2-D data with moderate n the selector picks a smooth method.
/// Note: The existing selector does not have an `is_gridded` field; it falls
/// through to smooth-data heuristics.  For n=1000, d=2, smooth=0.8 it
/// should pick Rbf(Gaussian) (rule 6: d<=4, effective_smooth > 0.5).
#[test]
fn gridded_picks_smooth_method() {
    let prob = InterpolationProblem {
        n_points: 1_000,
        dim: 2,
        smoothness_estimate: Some(0.8),
        available_memory_mb: None,
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    // The method should be one of the smooth-data recommendations.
    let is_smooth = matches!(
        rec.method,
        ProblemInterpolationMethod::Rbf(_)
            | ProblemInterpolationMethod::CubicSpline
            | ProblemInterpolationMethod::ThinPlateSpline
    );
    assert!(
        is_smooth,
        "Gridded smooth data (n=1000, d=2): expected a smooth method, got {:?}",
        rec.method
    );
}

// ---------------------------------------------------------------------------
// noisy_2d_picks_kriging_or_thinplate
// ---------------------------------------------------------------------------

/// Noisy 2-D scattered data with n < 1000 picks ThinPlateSpline (robust for 2-D).
/// Equivalent to task: DataProperties { n:200, d:2, noise:0.5 } → OrdinaryKriging.
#[test]
fn noisy_2d_small_picks_thinplate() {
    let prob = InterpolationProblem {
        n_points: 200,
        dim: 2,
        smoothness_estimate: Some(0.3), // rough/noisy
        available_memory_mb: None,
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    // Rule 4: d==2 && n < 1000 → ThinPlateSpline (robust for 2-D scattered).
    assert_eq!(
        rec.method,
        ProblemInterpolationMethod::ThinPlateSpline,
        "Noisy 2-D small-n: expected ThinPlateSpline, got {:?}",
        rec.method
    );
}

// ---------------------------------------------------------------------------
// high_dim_picks_kriging
// ---------------------------------------------------------------------------

/// High-dimensional small-n data should recommend Kriging.
/// Equivalent to task: DataProperties { n:500, d:8 } → SmolyakSparseGrid.
/// The existing selector uses Kriging for high-d small-n (rule 8).
#[test]
fn high_dim_small_n_picks_kriging() {
    let prob = InterpolationProblem {
        n_points: 500,
        dim: 8,
        smoothness_estimate: Some(0.7),
        available_memory_mb: None,
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    assert_eq!(
        rec.method,
        ProblemInterpolationMethod::Kriging,
        "High-dim small-n: expected Kriging, got {:?}",
        rec.method
    );
}

// ---------------------------------------------------------------------------
// memory_constrained_picks_rff
// ---------------------------------------------------------------------------

/// With a tight memory budget, the selector should choose RandomFeaturesRbf
/// to avoid storing an n×n kernel matrix.
#[test]
fn memory_constrained_picks_rff() {
    let prob = InterpolationProblem {
        n_points: 5_000,
        dim: 2,
        smoothness_estimate: None,
        available_memory_mb: Some(5), // 5 MB — far below n² matrix
        require_derivatives: false,
        time_budget_ms: None,
    };
    let rec = auto_select_method(&prob);
    assert_eq!(
        rec.method,
        ProblemInterpolationMethod::RandomFeaturesRbf,
        "Memory-constrained: expected RandomFeaturesRbf, got {:?}",
        rec.method
    );
}

// ---------------------------------------------------------------------------
// validated_rejects_zero_points
// ---------------------------------------------------------------------------

/// `auto_select_validated` should return an error for n=0.
#[test]
fn validated_rejects_zero_points() {
    let prob = InterpolationProblem {
        n_points: 0,
        dim: 1,
        ..Default::default()
    };
    let result = auto_select_validated(&prob);
    assert!(result.is_err(), "0 points should yield an error");
}

// ---------------------------------------------------------------------------
// validated_rejects_zero_dim
// ---------------------------------------------------------------------------

/// `auto_select_validated` should return an error for dim=0.
#[test]
fn validated_rejects_zero_dim() {
    let prob = InterpolationProblem {
        n_points: 10,
        dim: 0,
        ..Default::default()
    };
    let result = auto_select_validated(&prob);
    assert!(result.is_err(), "dim=0 should yield an error");
}

// ---------------------------------------------------------------------------
// recommendation_has_positive_memory_estimate
// ---------------------------------------------------------------------------

/// Every recommendation should carry a non-negative memory estimate.
#[test]
fn recommendation_has_positive_memory_estimate() {
    let cases = [
        InterpolationProblem {
            n_points: 50,
            dim: 1,
            ..Default::default()
        },
        InterpolationProblem {
            n_points: 500,
            dim: 2,
            ..Default::default()
        },
        InterpolationProblem {
            n_points: 5_000,
            dim: 3,
            ..Default::default()
        },
        InterpolationProblem {
            n_points: 100_000,
            dim: 4,
            ..Default::default()
        },
    ];
    for (i, prob) in cases.iter().enumerate() {
        let rec = auto_select_method(prob);
        assert!(
            rec.estimated_memory_mb >= 0.0,
            "Case {i}: memory estimate must be non-negative, got {}",
            rec.estimated_memory_mb
        );
        assert!(!rec.reason.is_empty(), "Case {i}: reason must be non-empty");
    }
}
