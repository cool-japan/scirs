//! Integration: interval arithmetic — containment property + tight bounds.
//!
//! Phase 1 fast-follow item C — verifies the soundness invariant of
//! `eval_interval`: for every formula `f` and every variable assignment
//! `xs` consistent with the per-variable intervals `vs`, the scalar value
//! `eval_real(f, xs)` (when defined) must be contained in `eval_interval(f, vs)`.

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{eval_interval, Interval, LoweredOp};

#[test]
fn containment_basic_polynomial() {
    // f(x) = x². For x ∈ [-1, 2], the range is [0, 4]. We verify the
    // containment property — the interval result must be a superset of
    // every scalar value attained inside the input interval.
    let f = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(-1.0, 2.0)]);
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
        let scalar = eval_real(&f, &EvalCtx::new(&[x])).expect("eval");
        assert!(
            r.contains(scalar),
            "x²: scalar {} not in interval {:?} for x={}",
            scalar,
            r,
            x
        );
    }
}

#[test]
fn containment_sin_full_period() {
    // sin(x) on x ∈ [0, 2π] reaches both ±1 at the critical points
    // π/2 and 3π/2. The interval rule must enumerate critical points to
    // recover the tight bounds (with outward 1-ULP widening).
    use std::f64::consts::PI;
    let f = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(0.0, 2.0 * PI)]);
    assert!(r.lo <= -1.0, "lo = {}, expected ≤ -1", r.lo);
    assert!(r.hi >= 1.0, "hi = {}, expected ≥ 1", r.hi);
}

#[test]
fn ln_negative_returns_nan() {
    // ln on a strictly-negative interval is a domain violation; the rule
    // must surface it as a NaN interval (not a panic, not a finite range).
    let f = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(-2.0, -1.0)]);
    assert!(r.is_nan(), "expected NaN interval, got {:?}", r);
}

#[test]
fn sqrt_negative_returns_nan() {
    // sqrt on a strictly-negative interval (hi < 0) is a domain violation.
    let f = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(-1.0, -0.5)]);
    assert!(r.is_nan(), "expected NaN interval, got {:?}", r);
}

#[test]
fn arcsin_out_of_domain_returns_nan() {
    // arcsin([1.5, 2.0]) is outside the domain [-1, 1].
    let f = LoweredOp::Arcsin(Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(1.5, 2.0)]);
    assert!(r.is_nan(), "expected NaN interval, got {:?}", r);
}

#[test]
fn random_containment_sweep() {
    // Multi-formula × multi-point property check. For each formula and
    // each scalar input `xv`, take a small interval around `xv`, evaluate
    // both ways, and verify containment of the scalar in the interval.
    let formulas: Vec<LoweredOp> = vec![
        LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0))),
        LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0))),
        LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0))),
        LoweredOp::Sin(Box::new(LoweredOp::Var(0))),
        LoweredOp::Cos(Box::new(LoweredOp::Var(0))),
        LoweredOp::Exp(Box::new(LoweredOp::Var(0))),
    ];

    for f in &formulas {
        for seed in 0..50 {
            let xv = (seed as f64) * 0.1 - 2.5;
            let xi = Interval::new(xv - 0.01, xv + 0.01);
            let interval_r = eval_interval(f, &[xi]);
            if interval_r.is_nan() {
                continue;
            }

            let scalar_r = eval_real(f, &EvalCtx::new(&[xv]));
            if let Ok(s) = scalar_r {
                assert!(
                    interval_r.contains(s),
                    "formula {:?}: scalar {} not in interval {:?} at x={}",
                    f,
                    s,
                    interval_r,
                    xv
                );
            }
        }
    }
}

#[test]
fn tight_monotone_exp() {
    // exp is monotone increasing on [0, 1], so the interval rule yields
    // bounds close to [exp(0), exp(1)] = [1, e]. Outward widening is
    // applied per `LoweredOp` node (here: Var, then Exp), so a few ULPs
    // of slack accumulate on each end. We test:
    //   - soundness: lo <= 1, hi >= e (containment of the true range)
    //   - tightness: the bounds are "near" 1 and e, within a generous
    //     slack that allows for ~16 ULPs of outward widening
    let f = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
    let r = eval_interval(&f, &[Interval::new(0.0, 1.0)]);
    let slack = 1e-14;
    assert!(r.lo <= 1.0, "soundness: lo = {} > 1.0", r.lo);
    assert!(
        r.lo >= 1.0 - slack,
        "tightness: lo = {} below 1 - slack",
        r.lo
    );
    assert!(r.hi >= std::f64::consts::E, "soundness: hi = {} < e", r.hi);
    assert!(
        r.hi <= std::f64::consts::E + slack,
        "tightness: hi = {} exceeds e + slack",
        r.hi
    );
}

#[test]
fn outward_widen_makes_lo_smaller_hi_bigger() {
    // Identity x → x must outward-widen: lo ≤ input lo, hi ≥ input hi.
    let f = LoweredOp::Var(0);
    let r = eval_interval(&f, &[Interval::new(1.0, 2.0)]);
    assert!(r.lo <= 1.0, "lo = {} > 1.0 (no widening)", r.lo);
    assert!(r.hi >= 2.0, "hi = {} < 2.0 (no widening)", r.hi);
}

#[test]
fn containment_div_with_safe_denominator() {
    // f(x, y) = x / y, x ∈ [1, 2], y ∈ [2, 3]. Range ⊆ [1/3, 1].
    let f = LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
    let r = eval_interval(&f, &[Interval::new(1.0, 2.0), Interval::new(2.0, 3.0)]);
    // Containment sweep over a 5x5 grid.
    for xi in 0..5 {
        for yi in 0..5 {
            let xv = 1.0 + (xi as f64) * 0.25;
            let yv = 2.0 + (yi as f64) * 0.25;
            let s = eval_real(&f, &EvalCtx::new(&[xv, yv])).expect("eval");
            assert!(
                r.contains(s),
                "x/y: {} not in {:?} at ({}, {})",
                s,
                r,
                xv,
                yv
            );
        }
    }
}
