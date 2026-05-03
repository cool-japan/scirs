//! Integration tests for anchored-ANOVA decomposition.

use scirs2_interpolate::sparse_grid::anchored_anova::{
    adaptive_anchored_anova_refinement, anchored_anova_decompose, AnchoredAnovaError,
};

#[test]
fn anchored_anova_separable_function_main_effects_capture_all_variance() {
    // f(x,y) = x + y (separable), anchor = default (0.5, 0.5).
    // Both main effects should be nonzero; interaction terms should be absent.
    let result =
        anchored_anova_decompose(&|x: &[f64]| x[0] + x[1], 2, None, 1, 20).expect("decompose");
    let total: f64 = result.main_effects.iter().map(|(_, v)| v).sum();
    assert!(
        total > 0.01,
        "separable function has nonzero main-effect variance; got {total}"
    );
    assert!(
        result.interaction_effects.is_empty(),
        "max_order=1 → no interaction effects"
    );
}

#[test]
fn anchored_anova_anchor_value_matches_function() {
    let anchor = vec![0.3, 0.7];
    let result = anchored_anova_decompose(&|x: &[f64]| x[0] * x[1], 2, Some(&anchor), 1, 10)
        .expect("decompose");
    assert!(
        (result.anchor_value - 0.3 * 0.7).abs() < 1e-12,
        "anchor_value={} expected {}",
        result.anchor_value,
        0.3 * 0.7
    );
}

#[test]
fn anchored_anova_constant_function_zero_variance() {
    // Use an arbitrary non-zero constant (not π to avoid clippy::approx_constant).
    let result =
        anchored_anova_decompose(&|_x: &[f64]| 42.0_f64, 3, None, 2, 10).expect("decompose");
    for &(dim, v) in &result.main_effects {
        assert!(
            v < 1e-10,
            "dim {dim}: main-effect variance = {v} for constant function"
        );
    }
    for &(i, j, v) in &result.interaction_effects {
        assert!(
            v < 1e-10,
            "interaction ({i},{j}): variance = {v} for constant function"
        );
    }
}

#[test]
fn anchored_anova_interaction_nonnegative_for_product() {
    // f(x,y) = x*y has an interaction term; variance must be ≥ 0.
    let result =
        anchored_anova_decompose(&|x: &[f64]| x[0] * x[1], 2, None, 2, 20).expect("decompose");
    for &(i, j, v) in &result.interaction_effects {
        assert!(v >= 0.0, "interaction ({i},{j}): variance {v} is negative");
    }
}

#[test]
fn anchored_anova_anchor_dim_mismatch_returns_error() {
    let bad_anchor = vec![0.5; 5]; // d=3 but anchor length 5
    let err = anchored_anova_decompose(&|_x: &[f64]| 0.0, 3, Some(&bad_anchor), 1, 5).unwrap_err();
    assert_eq!(err, AnchoredAnovaError::AnchorDimMismatch);
}

#[test]
fn anchored_anova_custom_anchor_changes_anchor_value() {
    let anchor1 = vec![0.2, 0.3];
    let anchor2 = vec![0.8, 0.9];
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let r1 = anchored_anova_decompose(&f, 2, Some(&anchor1), 1, 10).expect("decompose1");
    let r2 = anchored_anova_decompose(&f, 2, Some(&anchor2), 1, 10).expect("decompose2");
    assert!(
        (r1.anchor_value - r2.anchor_value).abs() > 1e-6,
        "different anchors should yield different anchor values"
    );
}

#[test]
fn adaptive_anchored_anova_returns_order1_for_separable() {
    // x + y is separable → adaptive should select order-1 (no interactions)
    let result = adaptive_anchored_anova_refinement(
        &|x: &[f64]| x[0] + x[1],
        2,
        None,
        0.05, // 5 % tolerance
        30,
    )
    .expect("adaptive");
    assert!(
        result.max_order_reached <= 2,
        "max_order_reached={}",
        result.max_order_reached
    );
}

#[test]
fn anchored_anova_all_dims_have_main_effect_entry() {
    let d = 4;
    let result = anchored_anova_decompose(&|x: &[f64]| x.iter().sum::<f64>(), d, None, 1, 10)
        .expect("decompose");
    assert_eq!(
        result.main_effects.len(),
        d,
        "expected {d} main effects, got {}",
        result.main_effects.len()
    );
}
