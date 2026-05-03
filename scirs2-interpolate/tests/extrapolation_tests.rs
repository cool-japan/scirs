//! Tests for Extrapolation Modes (Item #9).
//!
//! Exercises the `Extrapolate` trait with its five `ExtrapolationBehavior`
//! variants: Nearest, Linear, Polynomial, Reflection, Periodic.

use std::f64::consts::PI;

use scirs2_interpolate::extrapolation_trait::{
    ClosureInterpolator, Extrapolate, ExtrapolationBehavior,
};

// ---------------------------------------------------------------------------
// Helper: wrap a closure into the Extrapolate trait
// ---------------------------------------------------------------------------

fn linear_fn(x_min: f64, x_max: f64, slope: f64) -> ClosureInterpolator<impl Fn(f64) -> f64> {
    ClosureInterpolator {
        x_min,
        x_max,
        f: move |x| slope * x,
    }
}

// ---------------------------------------------------------------------------
// extrap_nearest_returns_boundary_value
// ---------------------------------------------------------------------------

/// `ExtrapolationBehavior::Nearest` should clamp to the boundary value.
#[test]
fn extrap_nearest_returns_boundary_value() {
    // f(x) = 3x on [1, 5]
    let interp = ClosureInterpolator {
        x_min: 1.0,
        x_max: 5.0,
        f: |x| 3.0 * x,
    };

    // Below domain: should clamp to x=1 → f(1) = 3.0
    let val_below = interp.extrapolate(0.0, &ExtrapolationBehavior::Nearest);
    assert!(
        (val_below - 3.0).abs() < 1e-10,
        "Nearest below: expected 3.0, got {val_below}"
    );

    // Above domain: should clamp to x=5 → f(5) = 15.0
    let val_above = interp.extrapolate(7.0, &ExtrapolationBehavior::Nearest);
    assert!(
        (val_above - 15.0).abs() < 1e-10,
        "Nearest above: expected 15.0, got {val_above}"
    );

    // Inside domain: should evaluate normally → f(3) = 9.0
    let val_inside = interp.extrapolate(3.0, &ExtrapolationBehavior::Nearest);
    assert!(
        (val_inside - 9.0).abs() < 1e-10,
        "Inside domain: expected 9.0, got {val_inside}"
    );
}

// ---------------------------------------------------------------------------
// extrap_linear_matches_analytic_line
// ---------------------------------------------------------------------------

/// For f(x) = 2x on [0, 1], linear extrapolation should give ≈ 2x.
#[test]
fn extrap_linear_matches_analytic_line() {
    let interp = linear_fn(0.0, 1.0, 2.0);

    // Below domain: x = -0.5 → expected 2*(-0.5) = -1.0
    let val_below = interp.extrapolate(-0.5, &ExtrapolationBehavior::Linear);
    assert!(
        (val_below - (-1.0)).abs() < 1e-4,
        "Linear extrap below: expected -1.0, got {val_below}"
    );

    // Above domain: x = 2.0 → expected 2*2.0 = 4.0
    let val_above = interp.extrapolate(2.0, &ExtrapolationBehavior::Linear);
    assert!(
        (val_above - 4.0).abs() < 1e-4,
        "Linear extrap above: expected 4.0, got {val_above}"
    );
}

// ---------------------------------------------------------------------------
// extrap_periodic_wraps_sin
// ---------------------------------------------------------------------------

/// sin(x) on [0, 2π] with Periodic extrapolation should match sin(x mod 2π).
#[test]
fn extrap_periodic_wraps_sin() {
    let interp = ClosureInterpolator {
        x_min: 0.0,
        x_max: 2.0 * PI,
        f: |x| x.sin(),
    };

    // x = 3π should map to x = π (since 3π mod 2π = π)
    let v1 = interp.extrapolate(PI, &ExtrapolationBehavior::Periodic);
    let v2 = interp.extrapolate(3.0 * PI, &ExtrapolationBehavior::Periodic);
    assert!(
        (v1 - v2).abs() < 1e-10,
        "sin(π) ≈ sin(3π) via Periodic: {v1} vs {v2}"
    );

    // x = 4π should behave like x = 0
    let v3 = interp.extrapolate(4.0 * PI, &ExtrapolationBehavior::Periodic);
    let v0 = interp.extrapolate(0.0, &ExtrapolationBehavior::Periodic);
    assert!(
        (v3 - v0).abs() < 1e-10,
        "sin(4π) ≈ sin(0) via Periodic: {v3} vs {v0}"
    );
}

// ---------------------------------------------------------------------------
// extrap_reflection_symmetry
// ---------------------------------------------------------------------------

/// Reflected value should match the corresponding interior value.
/// For f(x) = x on [0, 1]: reflecting x = -0.3 → 0.3, so f = 0.3.
#[test]
fn extrap_reflection_symmetry() {
    let interp = ClosureInterpolator {
        x_min: 0.0,
        x_max: 1.0,
        f: |x| x,
    };

    // Reflect -0.3 → should map to 0.3 (within [0,1])
    let val = interp.extrapolate(-0.3, &ExtrapolationBehavior::Reflection);
    assert!(
        (val - 0.3).abs() < 1e-10,
        "Reflection of -0.3 into [0,1]: expected 0.3, got {val}"
    );

    // Reflect 1.4 → should map to 0.6 (2.0 - 1.4 = 0.6)
    let val2 = interp.extrapolate(1.4, &ExtrapolationBehavior::Reflection);
    assert!(
        (val2 - 0.6).abs() < 1e-10,
        "Reflection of 1.4 into [0,1]: expected 0.6, got {val2}"
    );
}

// ---------------------------------------------------------------------------
// extrap_polynomial_extrapolates_quadratic
// ---------------------------------------------------------------------------

/// Polynomial(2) should extrapolate a quadratic f(x) = x² accurately.
#[test]
fn extrap_polynomial_extrapolates_quadratic() {
    // Domain [0, 4], f(x) = x²
    let interp = ClosureInterpolator {
        x_min: 0.0,
        x_max: 4.0,
        f: |x| x * x,
    };

    // Extrapolate to x = 5: exact = 25
    let val = interp.extrapolate(5.0, &ExtrapolationBehavior::Polynomial(2));
    assert!(
        val.is_finite(),
        "Polynomial extrap at x=5 must be finite, got {val}"
    );
    // For a degree-2 Neville stencil on a quadratic, the result should be exact.
    assert!(
        (val - 25.0).abs() < 1.0,
        "Polynomial extrap at x=5: expected ~25, got {val}"
    );
}

// ---------------------------------------------------------------------------
// extrap_nearest_inside_domain_passthrough
// ---------------------------------------------------------------------------

/// Inside-domain queries with any mode should delegate to value_at.
#[test]
fn extrap_inside_domain_passthrough() {
    let interp = ClosureInterpolator {
        x_min: 0.0,
        x_max: 10.0,
        f: |x| x * x,
    };

    let modes = [
        ExtrapolationBehavior::Nearest,
        ExtrapolationBehavior::Linear,
        ExtrapolationBehavior::Polynomial(3),
        ExtrapolationBehavior::Reflection,
        ExtrapolationBehavior::Periodic,
    ];

    for mode in &modes {
        let val = interp.extrapolate(3.0, mode);
        assert!(
            (val - 9.0).abs() < 1e-10,
            "Mode {:?}: expected 9.0 for x=3 inside [0,10], got {val}",
            mode
        );
    }
}
