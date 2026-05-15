//! Integration tests for `cas::solve_system` — multivariate algebraic system solver.

use scirs2_symbolic::cas::{solve_system, SystemKind, SystemSolveError};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
use std::collections::HashMap;

/// Evaluate a LoweredOp at a given set of variable bindings.
fn eval_op(op: &LoweredOp, bindings: &[f64]) -> f64 {
    let ctx = EvalCtx::new(bindings);
    eval_real(op, &ctx).unwrap_or(f64::NAN)
}

/// Evaluate an op with specific var assignments (var_id → value).
fn eval_with_map(op: &LoweredOp, map: &HashMap<usize, f64>) -> f64 {
    // Find max var id
    let max_id = map.keys().max().copied().unwrap_or(0);
    let mut bindings = vec![0.0f64; max_id + 1];
    for (&var_id, &val) in map {
        if var_id < bindings.len() {
            bindings[var_id] = val;
        }
    }
    let ctx = EvalCtx::new(&bindings);
    eval_real(op, &ctx).unwrap_or(f64::NAN)
}

// ---- Test helpers ----

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}
fn cnst(c: f64) -> LoweredOp {
    LoweredOp::Const(c)
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}
fn pow(a: LoweredOp, n: f64) -> LoweredOp {
    LoweredOp::Pow(Box::new(a), Box::new(cnst(n)))
}
fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Box::new(a))
}

// ---- Test 1: Linear 2×2 basic: {x + y = 3, x - y = 1} → {x=2, y=1} ----

#[test]
fn linear_2x2_basic() {
    // x=Var(0), y=Var(1)
    // Eq1: x + y = 3
    let eq1 = (add(var(0), var(1)), cnst(3.0));
    // Eq2: x - y = 1
    let eq2 = (sub(var(0), var(1)), cnst(1.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should succeed");

    assert_eq!(result.solutions.len(), 1, "Expected exactly 1 solution");
    assert_eq!(result.kind, SystemKind::Linear);
    assert!(result.complete);

    let sol = &result.solutions[0];
    let x_val = eval_with_map(
        sol.get(&0).expect("x should be in solution"),
        &HashMap::new(),
    );
    let y_val = eval_with_map(
        sol.get(&1).expect("y should be in solution"),
        &HashMap::new(),
    );

    assert!((x_val - 2.0).abs() < 1e-9, "Expected x=2.0, got {x_val}");
    assert!((y_val - 1.0).abs() < 1e-9, "Expected y=1.0, got {y_val}");
}

// ---- Test 2: Linear 3×3 system ----

#[test]
fn linear_3x3_rational() {
    // x=Var(0), y=Var(1), z=Var(2)
    // x + y + z = 6
    // x + 2y + 3z = 14
    // x + y + 2z = 9
    // Solution: x=1, y=2, z=3
    // Verify: 1+2+3=6 ✓, 1+4+9=14 ✓, 1+2+6=9 ✓
    let eq1 = (add(add(var(0), var(1)), var(2)), cnst(6.0));
    let eq2 = (
        add(add(var(0), mul(cnst(2.0), var(1))), mul(cnst(3.0), var(2))),
        cnst(14.0),
    );
    let eq3 = (add(add(var(0), var(1)), mul(cnst(2.0), var(2))), cnst(9.0));

    let result = solve_system(&[eq1, eq2, eq3], &[0, 1, 2]).expect("solve_system should succeed");

    assert_eq!(result.solutions.len(), 1);
    assert_eq!(result.kind, SystemKind::Linear);

    let sol = &result.solutions[0];
    let x_val = eval_with_map(sol.get(&0).expect("x"), &HashMap::new());
    let y_val = eval_with_map(sol.get(&1).expect("y"), &HashMap::new());
    let z_val = eval_with_map(sol.get(&2).expect("z"), &HashMap::new());

    assert!((x_val - 1.0).abs() < 1e-8, "Expected x=1, got {x_val}");
    assert!((y_val - 2.0).abs() < 1e-8, "Expected y=2, got {y_val}");
    assert!((z_val - 3.0).abs() < 1e-8, "Expected z=3, got {z_val}");
}

// ---- Test 3: Inconsistent linear system ----

#[test]
fn linear_inconsistent() {
    // x + y = 1 AND x + y = 2 → no solution
    let eq1 = (add(var(0), var(1)), cnst(1.0));
    let eq2 = (add(var(0), var(1)), cnst(2.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should return result");

    assert_eq!(result.kind, SystemKind::Inconsistent);
    assert!(result.solutions.is_empty());
}

// ---- Test 4: Underdetermined system ----

#[test]
fn linear_underdetermined() {
    // x + y = 1 AND 2x + 2y = 2 → same equation, underdetermined
    let eq1 = (add(var(0), var(1)), cnst(1.0));
    let eq2 = (
        add(mul(cnst(2.0), var(0)), mul(cnst(2.0), var(1))),
        cnst(2.0),
    );

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should return result");

    assert_eq!(result.kind, SystemKind::Underdetermined);
}

// ---- Test 5: Polynomial circle-line intersection ----

#[test]
fn polynomial_circle_line() {
    // x=Var(0), y=Var(1)
    // x^2 + y^2 = 1   (unit circle)
    // x + y = 0        (line through origin, slope -1)
    // Solutions: (1/√2, -1/√2) and (-1/√2, 1/√2)

    let circle = (add(pow(var(0), 2.0), pow(var(1), 2.0)), cnst(1.0));
    let line = (add(var(0), var(1)), cnst(0.0));

    let result = solve_system(&[circle, line], &[0, 1]).expect("solve_system should succeed");

    // Should find 2 solutions or at least 1
    assert!(
        !result.solutions.is_empty(),
        "Expected solutions for circle-line intersection"
    );

    // Verify each solution satisfies both equations
    for sol in &result.solutions {
        if let (Some(x_op), Some(y_op)) = (sol.get(&0), sol.get(&1)) {
            let x_val = eval_with_map(x_op, &HashMap::new());
            let y_val = eval_with_map(y_op, &HashMap::new());

            if x_val.is_finite() && y_val.is_finite() {
                // Check circle: x^2 + y^2 ≈ 1
                let circle_check = x_val * x_val + y_val * y_val;
                assert!(
                    (circle_check - 1.0).abs() < 1e-6,
                    "Circle equation not satisfied: x²+y²={circle_check} for x={x_val}, y={y_val}"
                );
                // Check line: x + y ≈ 0
                let line_check = x_val + y_val;
                assert!(
                    line_check.abs() < 1e-6,
                    "Line equation not satisfied: x+y={line_check}"
                );
            }
        }
    }
}

// ---- Test 6: Buchberger overrun returns PartialGroebner without panic ----

#[test]
fn buchberger_overrun_no_panic() {
    // Create a system with many variables and high-degree terms that forces many
    // S-polynomial computations, likely hitting the step budget.
    // x=0, y=1, z=2
    // x^2 + y^2 + z^2 = 1
    // x^2 - y^2 + z = 0
    // x*y + y*z + x*z = 0

    let eq1 = (
        add(add(pow(var(0), 2.0), pow(var(1), 2.0)), pow(var(2), 2.0)),
        cnst(1.0),
    );
    let eq2 = (
        add(sub(pow(var(0), 2.0), pow(var(1), 2.0)), var(2)),
        cnst(0.0),
    );
    let eq3 = (
        add(
            add(mul(var(0), var(1)), mul(var(1), var(2))),
            mul(var(0), var(2)),
        ),
        cnst(0.0),
    );

    // This may or may not hit budget; either way, must not panic
    let result = solve_system(&[eq1, eq2, eq3], &[0, 1, 2]);
    assert!(result.is_ok(), "Should not return an error (no panic)");
}

// ---- Test 7: Transcendental fallback with exp(x)=1 and x=y ----

#[test]
fn transcendental_fallback_exp() {
    // exp(x) = 1 → x = 0; x = y → y = 0
    // x=Var(0), y=Var(1)
    let eq1 = (LoweredOp::Exp(Box::new(var(0))), cnst(1.0));
    let eq2 = (var(0), var(1));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should succeed");

    assert!(
        !result.solutions.is_empty(),
        "Expected at least one solution"
    );

    let sol = &result.solutions[0];
    if let Some(x_op) = sol.get(&0) {
        let x_val = eval_with_map(x_op, &HashMap::new());
        assert!(
            x_val.is_finite() && x_val.abs() < 1e-8,
            "Expected x=0, got {x_val}"
        );
    }
    if let Some(y_op) = sol.get(&1) {
        let y_val = eval_with_map(y_op, &HashMap::new());
        assert!(
            y_val.is_finite() && y_val.abs() < 1e-8,
            "Expected y=0, got {y_val}"
        );
    }
}

// ---- Test 8: Transcendental bail on coupled sin/cos/ln ----

#[test]
fn transcendental_bail() {
    // sin(x) + exp(y) = 0 AND cos(x) + ln(y) = 0
    // Two unknowns, both equations have both unknowns, neither is linearly expressible
    let eq1 = (
        add(
            LoweredOp::Sin(Box::new(var(0))),
            LoweredOp::Exp(Box::new(var(1))),
        ),
        cnst(0.0),
    );
    let eq2 = (
        add(
            LoweredOp::Cos(Box::new(var(0))),
            LoweredOp::Ln(Box::new(var(1))),
        ),
        cnst(0.0),
    );

    let result = solve_system(&[eq1, eq2], &[0, 1]);
    // Should either fail with CannotEliminateTranscendental or return partial result
    // The important thing is it doesn't panic
    match result {
        Err(SystemSolveError::CannotEliminateTranscendental) => {
            // expected
        }
        Ok(_) => {
            // Also acceptable if some solution found
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 9: Empty vars error ----

#[test]
fn empty_vars_error() {
    let eq1 = (add(var(0), var(1)), cnst(3.0));
    let result = solve_system(&[eq1], &[]);
    assert!(
        matches!(result, Err(SystemSolveError::EmptyVars)),
        "Expected EmptyVars error"
    );
}

// ---- Test 10: Single equation, single var, linear ----

#[test]
fn single_linear() {
    // 3*x - 9 = 0 → x = 3
    let eq = (sub(mul(cnst(3.0), var(0)), cnst(9.0)), cnst(0.0));
    let result = solve_system(&[eq], &[0]).expect("solve_system should succeed");

    assert!(!result.solutions.is_empty());
    let sol = &result.solutions[0];
    let x_op = sol.get(&0).expect("x should be solved");
    let x_val = eval_with_map(x_op, &HashMap::new());
    assert!((x_val - 3.0).abs() < 1e-9, "Expected x=3, got {x_val}");
}

// ---- Test 11: Empty equations error ----

#[test]
fn empty_equations_error() {
    let result = solve_system(&[], &[0]);
    assert!(
        matches!(result, Err(SystemSolveError::EmptyEquations)),
        "Expected EmptyEquations error"
    );
}

// ---- Test 12: Single quadratic equation ----

#[test]
fn single_quadratic() {
    // x^2 - 4 = 0 → x = ±2
    let eq = (sub(pow(var(0), 2.0), cnst(4.0)), cnst(0.0));
    let result = solve_system(&[eq], &[0]).expect("solve_system should succeed");

    // Should find 2 solutions
    assert!(!result.solutions.is_empty(), "Expected solutions for x²=4");

    // Verify solutions satisfy x^2 = 4
    for sol in &result.solutions {
        if let Some(x_op) = sol.get(&0) {
            let x_val = eval_with_map(x_op, &HashMap::new());
            if x_val.is_finite() {
                assert!(
                    (x_val * x_val - 4.0).abs() < 1e-6,
                    "Solution x={x_val} doesn't satisfy x²=4"
                );
            }
        }
    }
}

// ---- Test 13: Linear 2×2 with negative coefficients ----

#[test]
fn linear_2x2_negative_coefficients() {
    // -2x + 3y = 5 AND 4x - y = 1
    // Solution: from eq2: y = 4x - 1. Substitute: -2x + 3(4x-1) = 5 → 10x - 3 = 5 → x=0.8, y=2.2
    let eq1 = (
        add(mul(cnst(-2.0), var(0)), mul(cnst(3.0), var(1))),
        cnst(5.0),
    );
    let eq2 = (sub(mul(cnst(4.0), var(0)), var(1)), cnst(1.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should succeed");
    assert!(!result.solutions.is_empty());

    let sol = &result.solutions[0];
    if let (Some(x_op), Some(y_op)) = (sol.get(&0), sol.get(&1)) {
        let x_val = eval_with_map(x_op, &HashMap::new());
        let y_val = eval_with_map(y_op, &HashMap::new());
        assert!((x_val - 0.8).abs() < 1e-8, "Expected x=0.8, got {x_val}");
        assert!((y_val - 2.2).abs() < 1e-8, "Expected y=2.2, got {y_val}");
    }
}

// ---- Test 14: Solution satisfies original equations ----

#[test]
fn solution_satisfies_equations() {
    // 2x + 3y = 12 AND x - y = -1
    // Solution: x=9/5=1.8, y=14/5=2.8
    let eq1_lhs = add(mul(cnst(2.0), var(0)), mul(cnst(3.0), var(1)));
    let eq1 = (eq1_lhs.clone(), cnst(12.0));
    let eq2_lhs = sub(var(0), var(1));
    let eq2 = (eq2_lhs.clone(), cnst(-1.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("solve_system should succeed");
    assert!(!result.solutions.is_empty());

    let sol = &result.solutions[0];
    let x_op = sol.get(&0).expect("x");
    let y_op = sol.get(&1).expect("y");
    let x_val = eval_with_map(x_op, &HashMap::new());
    let y_val = eval_with_map(y_op, &HashMap::new());

    // Verify eq1: 2x + 3y ≈ 12
    let check1 = 2.0 * x_val + 3.0 * y_val;
    assert!(
        (check1 - 12.0).abs() < 1e-8,
        "Eq1 not satisfied: {check1} ≠ 12"
    );

    // Verify eq2: x - y ≈ -1
    let check2 = x_val - y_val;
    assert!(
        (check2 - (-1.0)).abs() < 1e-8,
        "Eq2 not satisfied: {check2} ≠ -1"
    );
}

// ---- Test 15: Degree-3 polynomial (HighDegreePoly bail) ----

#[test]
fn polynomial_high_degree_partial_groebner_or_error() {
    // x^3 + y = 1 AND x + y^3 = 1 (symmetric cubic system)
    // This should either return PartialGroebner or some solutions
    let eq1 = (add(pow(var(0), 3.0), var(1)), cnst(1.0));
    let eq2 = (add(var(0), pow(var(1), 3.0)), cnst(1.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]);
    // Must not panic; result is ok or error
    match result {
        Ok(r) => {
            assert!(
                r.kind == SystemKind::Polynomial
                    || r.kind == SystemKind::PartialGroebner
                    || r.kind == SystemKind::Underdetermined,
                "Unexpected kind: {:?}",
                r.kind
            );
        }
        Err(e) => {
            // InternalError is not expected; CannotEliminateTranscendental is ok
            assert!(
                !matches!(e, SystemSolveError::InternalError(_)),
                "Internal error: {e:?}"
            );
        }
    }
}

// ---- Test 16: Verify linear 2×2 kind is Linear ----

#[test]
fn linear_kind_reported() {
    let eq1 = (add(var(0), var(1)), cnst(4.0));
    let eq2 = (sub(var(0), var(1)), cnst(2.0));

    let result = solve_system(&[eq1, eq2], &[0, 1]).expect("should succeed");
    assert_eq!(result.kind, SystemKind::Linear);
    assert!(result.complete);
}

// ---- Test 17: MAX_BUCHBERGER_STEPS constant is exposed ----

#[test]
fn max_buchberger_steps_exposed() {
    use scirs2_symbolic::cas::MAX_BUCHBERGER_STEPS;
    const { assert!(MAX_BUCHBERGER_STEPS > 0) }
    const { assert!(MAX_BUCHBERGER_STEPS <= 1024) }
}

// ---- Test 18: Negative eval of solution via eval_op helper ----

#[test]
fn eval_op_helper_smoke() {
    let op = LoweredOp::Const(42.0);
    let val = eval_op(&op, &[]);
    assert!((val - 42.0).abs() < 1e-12);
}
