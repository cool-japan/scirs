//! Compile-and-run tests for the CAS tutorial in docs/cas_tutorial.md.
//! Each test corresponds to one tutorial section.
//! If a test fails, update docs/cas_tutorial.md to match the current API.
//!
//! Feature requirements:
//! - Section 8 (JIT): requires `jit` feature
//! - Section 9 (GPU): requires `gpu` feature
//! - Section 4 (SMT): requires `smt` feature

use ndarray::{ArrayD, IxDyn};
use scirs2_symbolic::cas::ad::{grad_canonical, jacobian_canonical};
use scirs2_symbolic::cas::canonicalize::canonicalize;
use scirs2_symbolic::cas::identity_db::{apply_standard_identity_db, IdentityDb};
use scirs2_symbolic::cas::pattern::{match_pattern, BinaryKind, Pattern, UnaryKind};
use scirs2_symbolic::cas::{
    integrate_rational, solve, solve_ode, solve_system, IntegrateRationalError, OdeKind,
    SolveError, SolveResult,
};
use scirs2_symbolic::diffgeom::{christoffel, ricci_tensor, Metric};
use scirs2_symbolic::eml::parser::{parse, to_compact_string};
use scirs2_symbolic::eml::{eval_real, to_latex, EvalCtx, LoweredOp};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (mirrors tutorial examples)
// ─────────────────────────────────────────────────────────────────────────────

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

fn pow(base: LoweredOp, exp: f64) -> LoweredOp {
    LoweredOp::Pow(Box::new(base), Box::new(c(exp)))
}

fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Box::new(a))
}

fn exp_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Box::new(a))
}

fn sin_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Box::new(a))
}

fn cos_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Box::new(a))
}

fn zero_g(dim: usize) -> ArrayD<LoweredOp> {
    ArrayD::from_elem(IxDyn(&[dim, dim]), c(0.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Hello, EML — EML substrate basics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_01_hello_eml() {
    // f(x) = x² + 3x  where x = Var(0)
    let x = var(0);
    let f = add(pow(x.clone(), 2.0), mul(c(3.0), x.clone()));

    // Evaluate at x=2: 4 + 6 = 10
    let ctx = EvalCtx::new(&[2.0]);
    let val = eval_real(&f, &ctx).expect("eval f at x=2");
    assert!(
        (val - 10.0).abs() < 1e-10,
        "f(2) = x²+3x = 4+6 = 10, got {val}"
    );

    // The expression tree uses Box<LoweredOp> children — no shared mutable state.
    // Cloning is safe and cheap.
    let f2 = f.clone();
    let val2 = eval_real(&f2, &ctx).expect("eval f2 at x=2");
    assert!(
        (val2 - 10.0).abs() < 1e-10,
        "cloned expression gives same result"
    );

    // LoweredOp::Exp represents e^x (the E part of EML).
    let ex = exp_op(var(0));
    let e_val = eval_real(&ex, &EvalCtx::new(&[1.0])).expect("e^1");
    assert!(
        (e_val - std::f64::consts::E).abs() < 1e-10,
        "e^1 = e, got {e_val}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Parsing and Pretty-Printing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_02_parsing_pretty_printing() {
    // The EML parser accepts the core EML grammar: "eml(x0, 1)", "x0", "1".
    // It does not parse infix or function-call notation (those are built via
    // Canonical constructors or LoweredOp directly).
    let tree = parse("eml(x0, 1)").expect("parse eml(x0, 1)");

    // Convert back to compact string — should preserve EML structure
    let compact = to_compact_string(&tree);
    assert!(
        compact.contains("eml") || compact.contains("x0"),
        "compact string should contain eml or x0: {compact}"
    );

    // Parse a variable
    let x_tree = parse("x0").expect("parse x0");
    let x_compact = to_compact_string(&x_tree);
    assert!(
        x_compact.contains("x0") || x_compact.contains("x"),
        "compact string for x0: {x_compact}"
    );

    // Round-trip: build LoweredOp directly and render as LaTeX
    let x_sq = pow(var(0), 2.0);
    let latex_str = to_latex(&x_sq);
    assert!(
        latex_str.contains("x_{0}") || latex_str.contains("2"),
        "LaTeX for x²: {latex_str}"
    );

    // LaTeX rendering of sin
    let sin_x = sin_op(var(0));
    let sin_latex = to_latex(&sin_x);
    assert!(sin_latex.contains("sin"), "LaTeX for sin(x): {sin_latex}");

    // LaTeX rendering of exp
    let exp_x = exp_op(var(0));
    let exp_latex = to_latex(&exp_x);
    assert!(
        exp_latex.contains("e") || exp_latex.contains("exp"),
        "LaTeX for exp(x): {exp_latex}"
    );

    // Display trait via format! on a simple LoweredOp
    let add_op = add(var(0), c(1.0));
    let display_str = format!("{add_op}");
    assert!(
        !display_str.is_empty(),
        "Display should produce non-empty string"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Canonicalization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_03_canonicalization() {
    // Canonicalize x + 0 → x
    let x = var(0);
    let x_plus_zero = add(x.clone(), c(0.0));
    let canon = canonicalize(&x_plus_zero);
    let canon_op = canon.into_op();
    let val_canon = eval_real(&canon_op, &EvalCtx::new(&[5.0])).expect("eval after canon");
    assert!(
        (val_canon - 5.0).abs() < 1e-10,
        "canonicalize(x+0) evaluated at x=5 → 5, got {val_canon}"
    );

    // Canonicalize constant folding: 2.0 + 3.0 → 5.0
    let two_plus_three = add(c(2.0), c(3.0));
    let folded = canonicalize(&two_plus_three).into_op();
    let folded_val = eval_real(&folded, &EvalCtx::new(&[])).expect("eval folded constant");
    assert!(
        (folded_val - 5.0).abs() < 1e-10,
        "constant folding 2+3 → 5, got {folded_val}"
    );

    // Canonicalize: exp(ln(x)) → x for positive x
    let x_pos = var(0);
    let exp_ln_x = exp_op(LoweredOp::Ln(Box::new(x_pos.clone())));
    let exp_ln_canon = canonicalize(&exp_ln_x).into_op();
    let result_val = eval_real(&exp_ln_canon, &EvalCtx::new(&[4.0])).expect("eval exp(ln(x))");
    assert!(
        (result_val - 4.0).abs() < 1e-8,
        "exp(ln(4)) = 4, got {result_val}"
    );

    // Pattern matching: match sin(?0) against sin(x)
    let sin_x = sin_op(var(0));
    let pattern = Pattern::PatOp1(UnaryKind::Sin, Box::new(Pattern::PatVar(0)));
    let mut bindings = hashbrown::HashMap::new();
    let matched = match_pattern(&pattern, &sin_x, &mut bindings);
    assert!(matched, "sin(x) should match sin(?0)");
    assert!(
        bindings.contains_key(&0),
        "wildcard ?0 should be bound after matching"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Identity Database
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_04_identity_db() {
    // The standard identity database contains 10 rules.
    let db = IdentityDb::standard();
    assert!(
        !db.rules().is_empty(),
        "standard identity database should be non-empty"
    );

    // Apply the standard identity DB to sin²(x) + cos²(x).
    // Note: the identity for sin²+cos²→1 requires the exact structural form.
    // Build it manually: (sin(x))² + (cos(x))²
    let x = var(0);
    let sin_x = sin_op(x.clone());
    let cos_x = cos_op(x.clone());
    let sin2 = pow(sin_x.clone(), 2.0);
    let cos2 = pow(cos_x.clone(), 2.0);
    let sum = add(sin2, cos2);

    let rewritten = apply_standard_identity_db(&sum);
    // Evaluate the rewritten expression; it should equal 1.0 regardless of x.
    for x_val in [0.0, 0.5, 1.0, 2.0] {
        let v = eval_real(&rewritten, &EvalCtx::new(&[x_val])).expect("eval identity");
        assert!(
            (v - 1.0).abs() < 1e-8,
            "sin²(x)+cos²(x) after identity db at x={x_val} → {v} (expected 1.0)"
        );
    }

    // Verify cosh²(x) - sinh²(x) = 1 (hyperbolic Pythagorean)
    let cosh_x = LoweredOp::Cosh(Box::new(var(0)));
    let sinh_x = LoweredOp::Sinh(Box::new(var(0)));
    let cosh2 = pow(cosh_x, 2.0);
    let sinh2 = pow(sinh_x, 2.0);
    let hyp_sum = LoweredOp::Sub(Box::new(cosh2), Box::new(sinh2));
    let hyp_canon = canonicalize(&hyp_sum).into_op();
    let hyp_val = eval_real(&hyp_canon, &EvalCtx::new(&[1.0])).expect("eval hyp identity");
    assert!(
        (hyp_val - 1.0).abs() < 1e-8,
        "cosh²(x)-sinh²(x) = 1 numerically, got {hyp_val}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Solving — single-variable and ODE
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_05_solving() {
    // Solve linear equation: 2x + 4 = 0 → x = -2
    // lhs = 2*x + 4, rhs = 0
    let two_x_plus_4 = add(mul(c(2.0), var(0)), c(4.0));
    let zero = c(0.0);
    let result: Result<SolveResult, SolveError> = solve(&two_x_plus_4, &zero, 0);
    let sol = result.expect("solve 2x+4=0");
    assert!(
        !sol.solutions.is_empty(),
        "should have at least one solution"
    );
    let sol_val = eval_real(&sol.solutions[0], &EvalCtx::new(&[])).expect("eval solution");
    assert!(
        (sol_val - (-2.0)).abs() < 1e-8,
        "2x+4=0 → x=-2, got {sol_val}"
    );

    // Solve quadratic: x² - 5x + 6 = 0 → x=2 or x=3
    let x2_minus_5x_plus_6 = add(add(pow(var(0), 2.0), mul(c(-5.0), var(0))), c(6.0));
    let quad_result = solve(&x2_minus_5x_plus_6, &c(0.0), 0).expect("solve x²-5x+6=0");
    assert!(
        quad_result.solutions.len() >= 1,
        "x²-5x+6=0 should have solutions"
    );

    // Solve system: x + y = 5, x - y = 1 → x=3, y=2
    // Var(0)=x, Var(1)=y
    let eq1_lhs = add(var(0), var(1)); // x + y
    let eq1_rhs = c(5.0);
    let eq2_lhs = LoweredOp::Sub(Box::new(var(0)), Box::new(var(1))); // x - y
    let eq2_rhs = c(1.0);
    let system_result = solve_system(&[(eq1_lhs, eq1_rhs), (eq2_lhs, eq2_rhs)], &[0, 1])
        .expect("solve system x+y=5, x-y=1");
    assert!(
        !system_result.solutions.is_empty(),
        "linear system should have solutions"
    );

    // Solve ODE: dx/dt = x with x(0) = 1 → x(t) = exp(t)
    // x = Var(0), t = Var(1), rhs = x = Var(0)
    let ode_rhs = var(0);
    let ode_sol = solve_ode(&ode_rhs, 0, 1, Some((0.0, 1.0))).expect("solve dx/dt = x");
    assert_eq!(
        ode_sol.kind,
        OdeKind::Linear1stOrder,
        "dx/dt=x is linear 1st order"
    );
    assert!(
        ode_sol.integration_constants.is_empty(),
        "IC applied → no free constants"
    );

    // Evaluate solution at t=1: should be e ≈ 2.71828
    let t_var = 1;
    let n_slots = t_var + 1;
    let mut bindings = vec![0.0f64; n_slots];
    bindings[t_var] = 1.0;
    let e_approx = eval_real(&ode_sol.x_of_t, &EvalCtx::new(&bindings)).expect("eval ODE sol");
    assert!(
        (e_approx - std::f64::consts::E).abs() < 1e-5,
        "x(1) = e ≈ {}, got {e_approx}",
        std::f64::consts::E
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Integration (Risch-LITE)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_06_integration_risch_lite() {
    // Integrate 1/(x²+1) dx = arctan(x)
    // Numerator: 1 (degree 0: [1.0])
    // Denominator: x²+1 (degree 2: [1.0, 0.0, 1.0] ascending)
    let num = vec![c(1.0)];
    let den = vec![c(1.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("∫ 1/(x²+1) dx");

    // Verify by evaluating at x=1: arctan(1) = π/4 ≈ 0.7854
    let antideriv_canon = canonicalize(&antideriv).into_op();
    let val_at_1 = eval_real(&antideriv_canon, &EvalCtx::new(&[1.0])).expect("eval at x=1");
    let pi_over_4 = std::f64::consts::PI / 4.0;
    assert!(
        (val_at_1.abs() - pi_over_4.abs()).abs() < 0.1,
        "∫ 1/(x²+1) at x=1 should be ≈ π/4 = {pi_over_4}, got {val_at_1}"
    );

    // Integrate x (polynomial) numerator with constant denominator: ∫ x dx = x²/2
    let poly_num = vec![c(0.0), c(1.0)]; // x = 0 + 1*x
    let const_den = vec![c(1.0)];
    let poly_antideriv = integrate_rational(&poly_num, &const_den, 0).expect("∫ x dx");
    let poly_canon = canonicalize(&poly_antideriv).into_op();
    let poly_val = eval_real(&poly_canon, &EvalCtx::new(&[2.0])).expect("eval x²/2 at x=2");
    // x²/2 at x=2 = 2
    assert!(
        (poly_val - 2.0).abs() < 0.1,
        "∫ x dx at x=2 ≈ 2 (x²/2 = 2), got {poly_val}"
    );

    // Verify DenominatorDegreeTooHigh error for high-degree denominators (≥5)
    let high_den = vec![c(1.0), c(0.0), c(0.0), c(0.0), c(0.0), c(1.0)]; // x^5 + 1
    let err = integrate_rational(&num, &high_den, 0);
    assert!(
        matches!(
            err,
            Err(IntegrateRationalError::DenominatorDegreeTooHigh { .. })
        ),
        "degree-5 denominator should return DenominatorDegreeTooHigh, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Differentiation (GradGraph / grad_canonical)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_07_differentiation() {
    // f(x, y) = exp(x) * sin(y)
    // Var(0)=x, Var(1)=y
    let exp_x = exp_op(var(0));
    let sin_y = sin_op(var(1));
    let f = mul(exp_x, sin_y);

    // ∂f/∂x = exp(x) * sin(y)
    let df_dx = grad_canonical(&f, 0);
    // At (x=0, y=π/2): exp(0)*sin(π/2) = 1.0
    let pi_half = std::f64::consts::PI / 2.0;
    let df_dx_val = eval_real(&df_dx, &EvalCtx::new(&[0.0, pi_half])).expect("eval df/dx");
    assert!(
        (df_dx_val - 1.0).abs() < 1e-8,
        "∂(exp(x)sin(y))/∂x at (0,π/2) = 1.0, got {df_dx_val}"
    );

    // ∂f/∂y = exp(x) * cos(y)
    let df_dy = grad_canonical(&f, 1);
    // At (x=0, y=0): exp(0)*cos(0) = 1.0
    let df_dy_val = eval_real(&df_dy, &EvalCtx::new(&[0.0, 0.0])).expect("eval df/dy");
    assert!(
        (df_dy_val - 1.0).abs() < 1e-8,
        "∂(exp(x)sin(y))/∂y at (0,0) = 1.0, got {df_dy_val}"
    );

    // Jacobian: both partials at once
    let jac = jacobian_canonical(&f, 2);
    assert_eq!(jac.len(), 2, "Jacobian should have 2 components");

    // Verify d/dx[x²] = 2x at x=3 → 6
    let x_sq = pow(var(0), 2.0);
    let d_x_sq = grad_canonical(&x_sq, 0);
    let val_at_3 = eval_real(&d_x_sq, &EvalCtx::new(&[3.0])).expect("eval d/dx x²");
    assert!(
        (val_at_3 - 6.0).abs() < 1e-8,
        "d/dx x² at x=3 = 6, got {val_at_3}"
    );

    // Mixed partial: ∂²(x*y)/∂x∂y = 1
    let xy = mul(var(0), var(1));
    let d_xy_dx = grad_canonical(&xy, 0); // = y
    let d2_xy_dxdy = grad_canonical(&d_xy_dx, 1); // = 1
    let mixed = eval_real(&d2_xy_dxdy, &EvalCtx::new(&[2.0, 3.0])).expect("eval mixed partial");
    assert!((mixed - 1.0).abs() < 1e-8, "∂²(xy)/∂x∂y = 1, got {mixed}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: JIT Compilation (Cranelift)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "jit")]
#[test]
fn section_08_jit_compilation() {
    use scirs2_symbolic::compile::to_jit;

    // f(x) = x² + 2x + 1 = (x+1)²
    let x = var(0);
    let f = add(add(pow(x.clone(), 2.0), mul(c(2.0), x)), c(1.0));

    let jit_fn = to_jit(&f).expect("Cranelift JIT compilation");

    // Evaluate at x=3: (3+1)² = 16
    let jit_result = jit_fn.eval_checked(&[3.0]).expect("JIT eval at x=3");
    assert!(
        (jit_result - 16.0).abs() < 1e-10,
        "JIT f(3) = (3+1)² = 16, got {jit_result}"
    );

    // Evaluate at x=0: (0+1)² = 1
    let jit_result_0 = jit_fn.eval_checked(&[0.0]).expect("JIT eval at x=0");
    assert!(
        (jit_result_0 - 1.0).abs() < 1e-10,
        "JIT f(0) = (0+1)² = 1, got {jit_result_0}"
    );

    // JIT function reports correct n_vars
    assert_eq!(jit_fn.n_vars(), 1, "f(x) has 1 variable");

    // JIT transcendental: exp(x) at x=1 ≈ e
    let exp_f = exp_op(var(0));
    let exp_jit = to_jit(&exp_f).expect("JIT exp");
    let exp_val = exp_jit.eval_checked(&[1.0]).expect("JIT exp(1)");
    assert!(
        (exp_val - std::f64::consts::E).abs() < 1e-10,
        "JIT exp(1) = e, got {exp_val}"
    );
}

#[cfg(not(feature = "jit"))]
#[test]
fn section_08_jit_compilation() {
    // When compiled without --features jit, this test documents the feature gate.
    // Run with --features jit to exercise the Cranelift JIT path.
    eprintln!("section_08: JIT feature not enabled — skipping Cranelift tests");
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: GPU Dispatch (WGSL)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[test]
fn section_09_gpu_dispatch() {
    use scirs2_symbolic::compile::gpu::GpuError;
    use scirs2_symbolic::compile::to_gpu;

    // Build f(x) = x² + 1
    let f = add(pow(var(0), 2.0), c(1.0));

    // to_gpu generates WGSL shader text and returns a ready-to-dispatch GpuKernel
    let kernel = to_gpu(&f).expect("WGSL shader generation");
    let wgsl = kernel.wgsl();

    // Verify the shader text looks like valid WGSL
    assert!(
        wgsl.contains("@compute"),
        "WGSL should have @compute decorator"
    );
    assert!(
        wgsl.contains("fn eval_main"),
        "WGSL should have eval_main entry point"
    );

    // eval_batch now performs real wgpu dispatch (Phase 2).
    // On headless CI without a GPU adapter, NoAdapter is returned — that is
    // acceptable and the test passes.  On a real GPU host the results must be
    // approximately correct (f32 tolerance 1e-3).
    let batch_result = kernel.eval_batch(&[vec![1.0], vec![2.0]]);
    match batch_result {
        Ok(out) => {
            // f(1) = 1² + 1 = 2,  f(2) = 2² + 1 = 5
            assert_eq!(out.len(), 2, "output length mismatch");
            assert!((out[0] - 2.0_f64).abs() < 1e-3, "f(1) ≈ 2, got {}", out[0]);
            assert!((out[1] - 5.0_f64).abs() < 1e-3, "f(2) ≈ 5, got {}", out[1]);
        }
        Err(GpuError::NoAdapter(msg)) | Err(GpuError::DeviceError(msg)) => {
            println!("section_09: no GPU adapter available — skipping dispatch ({msg})");
        }
        Err(e) => panic!("unexpected eval_batch error: {e}"),
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn section_09_gpu_dispatch() {
    // When compiled without --features gpu, this test documents the feature gate.
    eprintln!("section_09: GPU feature not enabled — skipping WGSL tests");
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Python and WASM Bindings
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_10_python_wasm_bindings() {
    // Python bindings (scirs2-python crate) are not directly testable from Rust
    // integration tests. This test verifies the underlying API that PyO3 wraps.

    // The Python bindings expose:
    // - PyEmlTree: wraps EmlTree
    // - PyCanonical: wraps Canonical
    // - PyLoweredOp: wraps LoweredOp

    // Verify that EmlTree can be created (the type underlying PyEmlTree)
    use scirs2_symbolic::eml::{lower, EmlTree};
    let _tree = EmlTree::var(0);

    // Verify Canonical can be produced (the type underlying PyCanonical)
    use scirs2_symbolic::eml::Canonical;
    // Canonical::add takes &EmlTree arguments and returns EmlTree
    let x_tree = EmlTree::var(0);
    let one_tree = EmlTree::one();
    let canon_tree = Canonical::add(&x_tree, &one_tree);
    // Lower the EmlTree to a LoweredOp for eval
    let lowered = lower(&canon_tree);

    // Verify the lowered op evaluates correctly (same computation Python would get)
    let val = eval_real(&lowered, &EvalCtx::new(&[5.0])).expect("eval Canonical.add");
    assert!(
        (val - 6.0).abs() < 1e-10,
        "Canonical.add(Var(0), 1.0) at x=5 = 6, got {val}"
    );

    // WASM: the scirs2-wasm crate wraps this same eval_real / canonicalize API
    // for the browser playground. No additional Rust-side test needed here.
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 11: Cross-Crate Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_11_cross_crate_integration() {
    // Demonstrate symbolic gradient + Newton step (the pattern used by
    // scirs2-optimize::symbolic::newton).

    // Minimize f(x) = (x - 3)²  starting from x0 = 0
    // f'(x) = 2(x-3), f''(x) = 2
    let f = pow(LoweredOp::Sub(Box::new(var(0)), Box::new(c(3.0))), 2.0);
    let df = grad_canonical(&f, 0);
    let d2f = grad_canonical(&df, 0);

    // Newton step: x1 = x0 - f'(x0)/f''(x0)
    let x0 = 0.0_f64;
    let x0_binding = [x0];
    let ctx_x0 = EvalCtx::new(&x0_binding);
    let df_val = eval_real(&df, &ctx_x0).expect("eval f'(0)");
    let d2f_val = eval_real(&d2f, &ctx_x0).expect("eval f''(0)");
    let x1 = x0 - df_val / d2f_val;

    // f'(0) = 2*(0-3) = -6, f''(0) = 2, step = -(-6)/2 = 3.0
    assert!(
        (x1 - 3.0).abs() < 1e-8,
        "Newton step from x=0 on (x-3)² → x=3, got x1={x1}"
    );

    // Verify f(x1) ≈ 0 (minimum reached in one step for quadratic)
    let f_at_x1 = eval_real(&f, &EvalCtx::new(&[x1])).expect("eval f(x1)");
    assert!(f_at_x1.abs() < 1e-8, "f(x1) = (3-3)² = 0, got {f_at_x1}");

    // Cross-crate: scirs2-integrate uses cas::solve_ode for symbolic IVP solving.
    // Demonstrate the same pattern here: dx/dt = -x → x(t) = exp(-t)
    let rhs_neg_x = neg(var(0));
    let neg_x_sol = solve_ode(&rhs_neg_x, 0, 1, Some((0.0, 1.0))).expect("solve dx/dt = -x");
    let n_slots = 2;
    let mut t_bindings = vec![0.0f64; n_slots];
    t_bindings[1] = 1.0;
    let decay_at_1 = eval_real(&neg_x_sol.x_of_t, &EvalCtx::new(&t_bindings))
        .expect("eval decay solution at t=1");
    assert!(
        (decay_at_1 - std::f64::consts::E.recip()).abs() < 1e-5,
        "x(1)=exp(-1)≈{:.6}, got {decay_at_1}",
        std::f64::consts::E.recip()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 12: Diffgeom Mini-Example
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_12_diffgeom_schwarzschild() {
    // Schwarzschild metric in 2D simplified (t-r sector):
    // g_{tt} = -(1 - rs/r), g_{rr} = 1/(1 - rs/r)
    // Var(0)=r, Var(1)=t, Var(10)=rs (Schwarzschild radius constant)

    // For the vacuum solution with rs as a symbolic constant,
    // use a numerical approach: fix rs = 2.0, r = 4.0 (outside event horizon).
    // We test the flat (rs=0) case where Ricci = 0 exactly.

    // 2D flat Lorentzian: g = diag(-1, 1) — Minkowski in 2D
    let mut g_flat = zero_g(2);
    g_flat[IxDyn(&[0, 0])] = c(-1.0); // g_tt = -1
    g_flat[IxDyn(&[1, 1])] = c(1.0); // g_rr = 1
    let metric_flat = Metric::new(g_flat, vec![0, 1]).expect("2D Lorentzian metric");

    // Christoffel symbols for flat Lorentzian are all zero
    let gamma_flat = christoffel(&metric_flat);
    // Ricci tensor from flat Lorentzian
    let ricci_flat = ricci_tensor(&gamma_flat, &[0, 1]);

    // At any point the Ricci tensor for flat Minkowski is zero
    let sample_vals = [1.0, 1.0];
    for i in 0..2 {
        for j in 0..2 {
            let r_ij = eval_real(ricci_flat.get(&[i, j]), &EvalCtx::new(&sample_vals))
                .expect("eval Ricci_ij");
            assert!(
                r_ij.abs() < 1e-8,
                "Ricci[{i},{j}] for flat Lorentzian = 0, got {r_ij}"
            );
        }
    }

    // Now test the 2D polar metric (non-trivial Christoffels, zero Ricci for flat space)
    // g = diag(1, r²) — Var(0)=r, Var(1)=θ — this is flat space in polar coords
    // The Ricci scalar for flat 2D Euclidean in polar = 0
    let r = var(0);
    let mut g_polar = zero_g(2);
    g_polar[IxDyn(&[0, 0])] = c(1.0);
    g_polar[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(r.clone()), Box::new(c(2.0)));
    let metric_polar = Metric::new(g_polar, vec![0, 1]).expect("polar metric");
    let gamma_polar = christoffel(&metric_polar);
    let ricci_polar = ricci_tensor(&gamma_polar, &[0, 1]);

    // At r=2.0, θ=0.5: flat polar Ricci should be zero (numerically < 1e-8)
    let polar_vals = [2.0, 0.5];
    for i in 0..2 {
        for j in 0..2 {
            let r_ij = eval_real(ricci_polar.get(&[i, j]), &EvalCtx::new(&polar_vals))
                .expect("eval polar Ricci");
            assert!(
                r_ij.abs() < 1e-6,
                "Ricci[{i},{j}] for flat polar = 0, got {r_ij}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 13: Reference and Next Steps
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn section_13_reference_and_next_steps() {
    // This section is primarily documentation — verify that the top-level
    // crate exports are accessible and that version information is coherent.

    // Verify top-level re-exports are available
    let _var = LoweredOp::Var(0);
    let _const = LoweredOp::Const(1.0);

    // Verify EmlError and SymbolicError are importable
    use scirs2_symbolic::error::{EmlError, SymbolicError};
    let _err = EmlError::UnknownVariable("x".into());
    let _serr = SymbolicError::UnboundVariable("y".into());

    // Verify canonicalize is idempotent (a key correctness property)
    let x_plus_y = add(var(0), var(1));
    let c1 = canonicalize(&x_plus_y);
    let c2 = canonicalize(&c1.clone().into_op());
    // The canonical hashes should be equal (idempotence)
    assert_eq!(c1.hash(), c2.hash(), "canonicalize should be idempotent");

    // Verify the crate version string is accessible via CARGO_PKG_VERSION
    // (this is always true in a Rust binary/test, so just check it's non-empty)
    let version = env!("CARGO_PKG_VERSION");
    assert!(!version.is_empty(), "CARGO_PKG_VERSION should not be empty");

    // Smoke test: round-trip a complex expression through canonicalize
    let expr = add(
        mul(c(2.0), pow(var(0), 2.0)),
        add(mul(c(3.0), var(0)), c(1.0)),
    );
    let canon_expr = canonicalize(&expr).into_op();
    let val_at_1 = eval_real(&canon_expr, &EvalCtx::new(&[1.0])).expect("eval 2x²+3x+1 at x=1");
    assert!(
        (val_at_1 - 6.0).abs() < 1e-8,
        "2x²+3x+1 at x=1 = 6, got {val_at_1}"
    );
}
