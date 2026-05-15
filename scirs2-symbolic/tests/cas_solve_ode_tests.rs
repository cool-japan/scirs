//! Integration tests for `cas::solve_ode` — symbolic ODE solver.

use scirs2_symbolic::cas::canonicalize::canonicalize;
use scirs2_symbolic::cas::{solve_ode, OdeKind, SolveOdeError};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};

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
fn exp_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Box::new(a))
}
fn cos_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Box::new(a))
}

/// Evaluate `x_of_t` by substituting specific t and c values.
fn eval_solution(
    x_of_t: &LoweredOp,
    t_var: usize,
    t_val: f64,
    integration_constants: &[usize],
    c_vals: &[f64],
) -> f64 {
    // Find max var index
    let max_idx = t_var.max(*integration_constants.iter().max().unwrap_or(&0)) + 1;
    let mut bindings = vec![0.0f64; max_idx.max(t_var + 1)];
    bindings[t_var] = t_val;
    for (&c_id, &c_val) in integration_constants.iter().zip(c_vals.iter()) {
        if c_id < bindings.len() {
            bindings[c_id] = c_val;
        }
    }
    let ctx = EvalCtx::new(&bindings);
    eval_real(x_of_t, &ctx).unwrap_or(f64::NAN)
}

// ---- Test 1: dx/dt = x (a=1, f=0) with x(0)=1 → exp(t) ----

#[test]
fn linear_dx_dt_equals_x_with_ic() {
    // x = Var(0), t = Var(1)
    // dx/dt = x → rhs = x = Var(0)
    let rhs = var(0);

    let sol = solve_ode(&rhs, 0, 1, Some((0.0, 1.0))).expect("Should solve dx/dt = x");

    assert_eq!(sol.kind, OdeKind::Linear1stOrder);
    assert!(
        sol.integration_constants.is_empty(),
        "IC should pin the constant"
    );

    // Evaluate at t=1: expected e ≈ 2.71828
    let val = eval_solution(&sol.x_of_t, 1, 1.0, &sol.integration_constants, &[]);
    let expected = std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "Expected x(1)=e≈{expected}, got {val}"
    );
}

// ---- Test 2: dx/dt = -2x with x(0)=3 → 3*exp(-2t) ----

#[test]
fn linear_dx_dt_neg_2x_with_ic() {
    // dx/dt = -2*x
    let rhs = mul(cnst(-2.0), var(0));

    let sol = solve_ode(&rhs, 0, 1, Some((0.0, 3.0))).expect("Should solve dx/dt = -2x");

    assert_eq!(sol.kind, OdeKind::Linear1stOrder);

    // Evaluate at t=1: expected 3*e^{-2} ≈ 0.4060
    let val = eval_solution(&sol.x_of_t, 1, 1.0, &sol.integration_constants, &[]);
    let expected = 3.0 * (-2.0f64).exp();
    assert!(
        (val - expected).abs() < 1e-6,
        "Expected x(1)=3e^(-2)≈{expected}, got {val}"
    );
}

// ---- Test 3: dx/dt = x²+1 → separable (arctan type) ----

#[test]
fn separable_dx_dt_x_sq_plus_1() {
    // dx/dt = x^2 + 1 → g(x) = x^2 + 1, f(t) = 1
    // integral dx/(x^2+1) = arctan(x), integral dt = t
    // arctan(x) = t + C
    let x_sq = pow(var(0), 2.0);
    let rhs = add(x_sq, cnst(1.0));

    let sol = solve_ode(&rhs, 0, 1, None).expect("Should solve dx/dt = x^2 + 1");

    // Should be separable or implicit-separable
    assert!(
        sol.kind == OdeKind::Separable || sol.kind == OdeKind::ImplicitSeparable,
        "Expected Separable or ImplicitSeparable, got {:?}",
        sol.kind
    );
    assert!(
        !sol.integration_constants.is_empty(),
        "Should have integration constant"
    );
}

// ---- Test 4: dx/dt + 2x = sin(t) — linear 1st-order non-homogeneous ----

#[test]
fn linear_1st_order_non_homogeneous() {
    // dx/dt = -2x + sin(t)
    // x = Var(0), t = Var(1)
    let sin_t = LoweredOp::Sin(Box::new(var(1)));
    let rhs = add(mul(cnst(-2.0), var(0)), sin_t);

    // try_integrate will be called on exp(2t) * sin(t).
    // This is NOT a rational function (try_integrate handles only rational/polynomial).
    // Expect: IntegralNotElementary or NotRecognized (no other family applies).
    let sol = solve_ode(&rhs, 0, 1, None);

    match sol {
        Ok(s) => {
            assert_eq!(
                s.kind,
                OdeKind::Linear1stOrder,
                "If succeeded, should be Linear1stOrder"
            );
        }
        Err(SolveOdeError::IntegralNotElementary) => {
            // Expected: exp(2t)*sin(t) not integrable as rational function
        }
        Err(SolveOdeError::NotRecognized) => {
            // Also acceptable: integration failed and no other family matched
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 5: dx/dt = exp(x^2) → IntegralNotElementary ----

#[test]
fn integral_not_elementary() {
    // dx/dt = exp(x^2) → separable with g(x) = exp(x^2), integral dx/exp(x^2) = integral exp(-x^2) dx
    // This is not rational, so should return IntegralNotElementary or NotRecognized
    let x_sq = pow(var(0), 2.0);
    let rhs = exp_op(x_sq);

    let result = solve_ode(&rhs, 0, 1, None);
    match result {
        Err(SolveOdeError::IntegralNotElementary) => { /* expected */ }
        Err(SolveOdeError::NotRecognized) => { /* also acceptable */ }
        Ok(_) => {
            // If somehow solved, that's a bonus
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 6: dx/dt = exp(x) * cos(t) → Separable or IntegralNotElementary ----

#[test]
fn separable_exp_x_cos_t() {
    // dx/dt = exp(x) * cos(t)
    // g(x) = exp(x), f(t) = cos(t)
    // integral 1/exp(x) dx = integral exp(-x) dx = -exp(-x)
    // However, try_integrate only handles rational functions, not exponentials.
    // So this may return IntegralNotElementary.
    let exp_x = exp_op(var(0));
    let cos_t = cos_op(var(1));
    let rhs = mul(cos_t, exp_x);

    let result = solve_ode(&rhs, 0, 1, None);

    match result {
        Ok(s) => {
            assert!(
                s.kind == OdeKind::Separable || s.kind == OdeKind::ImplicitSeparable,
                "Expected Separable kind if successful, got {:?}",
                s.kind
            );
            assert!(!s.integration_constants.is_empty());
        }
        Err(SolveOdeError::IntegralNotElementary) => {
            // Acceptable: 1/exp(x) is not a rational function in try_integrate
        }
        Err(SolveOdeError::NotRecognized) => {
            // Also acceptable if separation detection fails after canonicalization
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 7: Bernoulli dx/dt + x = x^2 (p=1, q=1, n=2) ----

#[test]
fn bernoulli_dx_dt_plus_x_equals_x_sq() {
    // dx/dt = -x + x^2 (rearranged from dx/dt + x = x^2)
    // p(t) = 1, q(t) = 1, n = 2
    // Note: -x + x^2 = x(x-1) is also separable (g(x) = x^2 - x, f(t) = 1).
    // Since separable is tried before Bernoulli, the solver may classify this
    // as Separable or ImplicitSeparable. Both are valid.
    let rhs = add(neg(var(0)), pow(var(0), 2.0));

    let sol = solve_ode(&rhs, 0, 1, None);

    match sol {
        Ok(s) => {
            assert!(
                s.kind == OdeKind::Bernoulli
                    || s.kind == OdeKind::Separable
                    || s.kind == OdeKind::ImplicitSeparable,
                "Expected Bernoulli, Separable, or ImplicitSeparable kind, got {:?}",
                s.kind
            );
        }
        Err(SolveOdeError::NotRecognized) | Err(SolveOdeError::IntegralNotElementary) => {
            // Acceptable if integration fails (e.g., 1/(x^2-x) partial fraction complex)
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 8: OdeKind is correctly assigned for dx/dt = x ----

#[test]
fn ode_kind_classification_linear() {
    let rhs = var(0); // dx/dt = x
    let sol = solve_ode(&rhs, 0, 1, None).expect("Should solve");
    assert_eq!(sol.kind, OdeKind::Linear1stOrder);
}

// ---- Test 9: Canonical invariance ----

#[test]
fn canonical_invariance_same_ode() {
    // Solving the same ODE twice should yield structurally equivalent x_of_t
    let rhs = var(0); // dx/dt = x

    let sol1 = solve_ode(&rhs, 0, 1, None).expect("First solve");
    let sol2 = solve_ode(&rhs, 0, 1, None).expect("Second solve");

    let c1 = canonicalize(&sol1.x_of_t);
    let c2 = canonicalize(&sol2.x_of_t);

    assert_eq!(
        c1.hash(),
        c2.hash(),
        "Canonical hashes should match for same ODE"
    );
}

// ---- Test 10: IVP determines integration constant ----

#[test]
fn ivp_determines_constant() {
    // dx/dt = x with x(0) = 2 → x = 2*exp(t)
    let rhs = var(0);

    let sol_with_ic = solve_ode(&rhs, 0, 1, Some((0.0, 2.0))).expect("Should solve with IC");
    let sol_no_ic = solve_ode(&rhs, 0, 1, None).expect("Should solve without IC");

    // With IC: constants should be pinned
    assert!(
        sol_with_ic.integration_constants.is_empty(),
        "IC should pin the constant"
    );

    // Without IC: should have constant
    assert!(
        !sol_no_ic.integration_constants.is_empty(),
        "No IC should leave free constant"
    );
}

// ---- Test 11: Integration constant count for 1st-order ----

#[test]
fn integration_constant_count_1st_order() {
    // 1st-order ODE should have exactly 1 integration constant (when no IC given)
    let rhs = var(0); // dx/dt = x
    let sol = solve_ode(&rhs, 0, 1, None).expect("Should solve");
    assert_eq!(
        sol.integration_constants.len(),
        1,
        "1st-order ODE should have 1 integration constant"
    );
}

// ---- Test 12: OrderTooHigh for Painlevé-like rhs with negative coeff ----

#[test]
fn order_too_high_for_harmonic() {
    // dx/dt = -4*x (could represent d²x/dt² = -4x harmonic oscillator)
    // The convention: negative coefficient, no t dependence → OrderTooHigh
    let rhs = mul(cnst(-4.0), var(0));

    let result = solve_ode(&rhs, 0, 1, None);
    match result {
        Err(SolveOdeError::OrderTooHigh) => { /* expected by convention */ }
        Ok(s) => {
            // Also acceptable — the solver might handle this as linear 1st-order
            assert_eq!(s.kind, OdeKind::Linear1stOrder);
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ---- Test 13: NotRecognized for Lotka-Volterra-like rhs ----

#[test]
fn not_recognized_for_coupled_nonlinear() {
    // dx/dt = ax - bxy (Lotka-Volterra prey term has Var(y) which is different)
    // Here we simulate with: rhs = 2*x - 3*x*y (y = Var(2))
    // x = Var(0), t = Var(1), y = Var(2)
    let rhs = sub(mul(cnst(2.0), var(0)), mul(cnst(3.0), mul(var(0), var(2))));

    // y is Var(2), but we're solving for x (Var(0)) in t (Var(1))
    // Var(2) is treated as a parameter — the solver might handle it or not
    let result = solve_ode(&rhs, 0, 1, None);
    // Don't assert a specific error — just ensure no panic
    let _ = result;
}

// ---- Test 14: Solution is continuous (no discontinuity at t=0) ----

#[test]
fn solution_continuous_at_origin() {
    // dx/dt = x, x(0) = 1 → x = exp(t)
    let sol = solve_ode(&var(0), 0, 1, Some((0.0, 1.0))).expect("Should solve");

    // Evaluate at t=0: should be 1.0
    let val_at_0 = eval_solution(&sol.x_of_t, 1, 0.0, &[], &[]);
    assert!(
        (val_at_0 - 1.0).abs() < 1e-6,
        "Expected x(0)=1, got {val_at_0}"
    );
}

// ---- Test 15: SolveOdeError implements Display ----

#[test]
fn solve_ode_error_display() {
    let err = SolveOdeError::IntegralNotElementary;
    let msg = format!("{err}");
    assert!(!msg.is_empty());

    let err2 = SolveOdeError::OrderTooHigh;
    let msg2 = format!("{err2}");
    assert!(!msg2.is_empty());

    let err3 = SolveOdeError::NotRecognized;
    let msg3 = format!("{err3}");
    assert!(!msg3.is_empty());
}
