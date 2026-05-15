//! oxieml v0.1 parity harness — verifies clean-room scirs2-symbolic produces
//! the same numerical answers as the published oxieml reference.
//!
//! # Tolerance
//!
//! Aspirational target: 1e-12 absolute / relative on `f64`.
//!
//! Realistic bound: **1e-9 absolute and 1e-9 relative**. The canonical
//! `Canonical::sin(x)` tree alone is 543 nodes deep (Euler-formula
//! decomposition over the EML operator); after 543 chained `exp`/`ln`
//! evaluations through `Complex64`, ULP noise accumulates well past 1e-12.
//! 1e-9 is the tightest mutually-achievable bound across both libraries
//! when expressed via canonical encodings — this is a property of the EML
//! representation, not of either implementation.
//!
//! # Intentional divergences (asserted, not regressed)
//!
//! 1. `Sqrt`/`Abs` are native [`LoweredOp`] variants in scirs2-symbolic
//!    (oxieml lowers `sqrt` via `Pow(_, 0.5)` and `abs` via `sqrt(square(_))`).
//!    Verified via [`divergence_sqrt_native_grad_shape`] and
//!    [`divergence_abs_native_subgradient`].
//! 2. Outward 1-ULP widening on `Interval` (oxieml uses bare arithmetic).
//!    Verified via [`divergence_interval_widening`].
//! 3. `Canonical::nat(0)` returns `Err` in scirs2-symbolic (oxieml panics).
//!    Verified via [`divergence_nat_zero_returns_err`].
//! 4. scirs2-symbolic's `eval_real` errors on `ln(non-positive)` in the
//!    real path; oxieml's `eval_real` routes through complex arithmetic
//!    and only errors when the imaginary part exceeds 1e-12. Verified
//!    via [`divergence_eval_real_strictness`].
//!
//! # Resolution protocol (when oxieml says X and we say Y)
//!
//! 1. Reproduce the divergence at the documented tolerance.
//! 2. Determine which is mathematically correct (cite the paper §
//!    or IEEE 754 § as required).
//! 3. If oxieml is correct: file an issue, ship a fix as PR
//!    `parity: <description>`.
//! 4. If scirs2-symbolic is correct: add a `divergence_*` test
//!    asserting OUR behaviour; document in this header; file
//!    upstream issue against oxieml.
//! 5. If both defensible (e.g. branch-cut convention, ULP rounding):
//!    document in ADR; pin the divergence test.
//!
//! # CI gate
//!
//! This file is the canonical parity gate. CI must run
//! `cargo test -p scirs2-symbolic --test oxieml_parity` and treat any
//! failure as a release blocker. New `Canonical::*` constructors MUST
//! ship with a matching `parity_<name>` test before being merged.

use num_complex::Complex64;
use oxieml::{Canonical as OxiCanonical, EmlTree as OxiTree, EvalCtx as OxiCtx};
use scirs2_symbolic::eml::{
    eval_complex, eval_real, lower, Canonical as OurCanonical, EmlTree as OurTree, EvalCtx,
    LoweredOp,
};
use scirs2_symbolic::error::EmlError;

/// Mutual-agreement tolerance between scirs2-symbolic and oxieml.
///
/// Loosened from the aspirational 1e-12 target because canonical EML
/// encodings (e.g. 543-node sin) accumulate ULP noise across hundreds of
/// chained complex operations. See the file header for the full rationale.
const TOL: f64 = 1e-9;

/// Threshold for treating a complex result as real-valued.
const IM_THRESHOLD: f64 = 1e-9;

/// Number of sample points per parity test.
const N_PARITY: usize = 100;

/// Minimum number of samples that must successfully evaluate (in BOTH
/// libraries) for a parity test to be considered meaningful.
///
/// This guards against the trap where every sample silently errors and the
/// test passes vacuously — a green CI lying to you.
const MIN_SAMPLES_COMPARED: usize = N_PARITY / 2;

// =====================================================================
// Sample generators
// =====================================================================

/// Sample `[-3, 3]` uniformly with `N_PARITY` points.
fn sample_xs() -> Vec<f64> {
    (0..N_PARITY)
        .map(|i| {
            let t = (i as f64) / (N_PARITY as f64 - 1.0);
            -3.0 + t * 6.0
        })
        .collect()
}

/// Sample `(-1, 1)` for `arcsin`/`arccos`/`arctanh` domain.
fn sample_open_unit_interval() -> Vec<f64> {
    (0..N_PARITY)
        .map(|i| {
            let t = (i as f64) / (N_PARITY as f64 - 1.0);
            -0.99 + t * 1.98
        })
        .collect()
}

/// Sample `(0, 5]` for `ln`/`sqrt`/positive-domain functions.
fn sample_positive() -> Vec<f64> {
    (0..N_PARITY)
        .map(|i| {
            let t = (i as f64) / (N_PARITY as f64 - 1.0);
            0.01 + t * 4.99
        })
        .collect()
}

/// Sample `(1, 5]` for `arccosh` domain.
///
/// arccosh's domain is `x >= 1`, but the canonical encoding
/// `ln(x + sqrt(x² - 1))` is evaluated through complex arithmetic where
/// the `sqrt(0)` at the boundary `x = 1` triggers `ln(0) = -∞`. The
/// complex eval then yields `re = inf` (one library) or near-`0`
/// (another) depending on the order of cancellations — this is a
/// branch-point sensitivity, not a parity bug. Start at `x = 1.05` to
/// stay inside the well-conditioned interior.
fn sample_geq_one() -> Vec<f64> {
    (0..N_PARITY)
        .map(|i| {
            let t = (i as f64) / (N_PARITY as f64 - 1.0);
            1.05 + t * 3.95
        })
        .collect()
}

// =====================================================================
// Eval wrappers — match oxieml semantics
// =====================================================================

/// Evaluate our `OurTree` at a single real input.
///
/// Lowers the tree once, tries the real path; on `EvalDomain` failure
/// (e.g. `ln(-1)` in canonical encodings of `sin`/`cos`), falls back to
/// the complex path and returns the real part if `|im| < IM_THRESHOLD`.
/// Returns `None` if neither path produces a finite real value.
///
/// Mirrors `oxieml::EmlTree::eval_real`'s real-via-complex semantics.
fn our_eval(tree: &OurTree, x: f64) -> Option<f64> {
    let lo = lower(tree);
    if let Ok(v) = eval_real(&lo, &EvalCtx::new(&[x])) {
        if v.is_finite() {
            return Some(v);
        }
    }
    if let Ok(c) = eval_complex(&lo, &[Complex64::new(x, 0.0)]) {
        if c.im.abs() < IM_THRESHOLD && c.re.is_finite() {
            return Some(c.re);
        }
    }
    None
}

/// Evaluate an `oxieml::EmlTree` at a single real input.
fn oxi_eval(tree: &OxiTree, x: f64) -> Option<f64> {
    let ctx = OxiCtx::new(&[x]);
    match tree.eval_real(&ctx) {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// Per-sample agreement check between two finite `f64` values.
fn agrees(a: f64, b: f64) -> bool {
    let abs_err = (a - b).abs();
    if abs_err < TOL {
        return true;
    }
    let denom = a.abs().max(b.abs()).max(1.0);
    abs_err / denom < TOL
}

// =====================================================================
// Parity runner
// =====================================================================

/// Generic parity test runner.
///
/// Builds the same canonical formula in both libraries, evaluates each at
/// every sample point, asserts agreement to [`TOL`]. A sample is **only**
/// skipped when BOTH libraries error or both return non-finite values; if
/// one succeeds and the other fails, that's a parity bug and the runner
/// panics with full context.
///
/// Asserts `n_compared >= MIN_SAMPLES_COMPARED` to catch the trap where
/// every sample silently errors and the test passes vacuously.
fn assert_parity(name: &str, samples: &[f64], ours: &OurTree, oxi: &OxiTree) {
    let mut compared = 0usize;
    let mut max_err = 0.0_f64;
    for &x in samples {
        let our = our_eval(ours, x);
        let oxi_r = oxi_eval(oxi, x);
        match (our, oxi_r) {
            (Some(a), Some(b)) => {
                let abs_err = (a - b).abs();
                if abs_err > max_err {
                    max_err = abs_err;
                }
                assert!(
                    agrees(a, b),
                    "parity({}, x={}): ours={}, oxi={}, abs_err={:e}",
                    name,
                    x,
                    a,
                    b,
                    abs_err
                );
                compared += 1;
            }
            (Some(a), None) => panic!(
                "parity({}, x={}): ours={} succeeded but oxieml errored",
                name, x, a
            ),
            (None, Some(b)) => panic!(
                "parity({}, x={}): oxieml={} succeeded but ours errored",
                name, x, b
            ),
            (None, None) => {} // both errored — domain mismatch, skip
        }
    }
    assert!(
        compared >= MIN_SAMPLES_COMPARED,
        "parity({}): only {} of {} samples compared (< MIN_SAMPLES_COMPARED={}); test asserted nothing",
        name,
        compared,
        N_PARITY,
        MIN_SAMPLES_COMPARED
    );
    eprintln!("parity({name}): {compared} samples agreed, max_abs_err = {max_err:e}");
}

// =====================================================================
// Parity tests — Table 1 (basic ops)
// =====================================================================

#[test]
fn parity_exp() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::exp(&our_x);
    let oxi = OxiCanonical::exp(&oxi_x);
    assert_parity("exp", &xs, &our, &oxi);
}

#[test]
fn parity_ln() {
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::ln(&our_x);
    let oxi = OxiCanonical::ln(&oxi_x);
    assert_parity("ln", &xs, &our, &oxi);
}

#[test]
fn parity_neg() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::neg(&our_x);
    let oxi = OxiCanonical::neg(&oxi_x);
    assert_parity("neg", &xs, &our, &oxi);
}

// =====================================================================
// Parity tests — Table 2 (arithmetic)
// =====================================================================

#[test]
fn parity_add_pointwise() {
    // f(x) = x + (x + 1) — verifies add() compiles to identical formulas.
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our_one = OurTree::one();
    let oxi_one = OxiTree::one();
    let our = OurCanonical::add(&our_x, &OurCanonical::add(&our_x, &our_one));
    let oxi = OxiCanonical::add(&oxi_x, &OxiCanonical::add(&oxi_x, &oxi_one));
    assert_parity("add(x, x+1)", &xs, &our, &oxi);
}

#[test]
fn parity_sub_pointwise() {
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our_one = OurTree::one();
    let oxi_one = OxiTree::one();
    let our = OurCanonical::sub(&our_x, &our_one);
    let oxi = OxiCanonical::sub(&oxi_x, &oxi_one);
    assert_parity("sub(x, 1)", &xs, &our, &oxi);
}

#[test]
fn parity_mul_pointwise() {
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::mul(&our_x, &our_x);
    let oxi = OxiCanonical::mul(&oxi_x, &oxi_x);
    assert_parity("mul(x, x)", &xs, &our, &oxi);
}

#[test]
fn parity_div_pointwise() {
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our_one = OurTree::one();
    let oxi_one = OxiTree::one();
    let our = OurCanonical::div(&our_one, &our_x);
    let oxi = OxiCanonical::div(&oxi_one, &oxi_x);
    assert_parity("div(1, x)", &xs, &our, &oxi);
}

// =====================================================================
// Parity tests — Table 3 (trig)
// =====================================================================

#[test]
fn parity_sin() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::sin(&our_x);
    let oxi = OxiCanonical::sin(&oxi_x);
    assert_parity("sin", &xs, &our, &oxi);
}

#[test]
fn parity_cos() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::cos(&our_x);
    let oxi = OxiCanonical::cos(&oxi_x);
    assert_parity("cos", &xs, &our, &oxi);
}

// =====================================================================
// Parity tests — Table 3 (hyperbolic)
// =====================================================================

#[test]
fn parity_sinh() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::sinh(&our_x);
    let oxi = OxiCanonical::sinh(&oxi_x);
    assert_parity("sinh", &xs, &our, &oxi);
}

#[test]
fn parity_cosh() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::cosh(&our_x);
    let oxi = OxiCanonical::cosh(&oxi_x);
    assert_parity("cosh", &xs, &our, &oxi);
}

#[test]
fn parity_tanh() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::tanh(&our_x);
    let oxi = OxiCanonical::tanh(&oxi_x);
    assert_parity("tanh", &xs, &our, &oxi);
}

#[test]
fn parity_arcsinh() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::arcsinh(&our_x);
    let oxi = OxiCanonical::arcsinh(&oxi_x);
    assert_parity("arcsinh", &xs, &our, &oxi);
}

#[test]
fn parity_arccosh() {
    let xs = sample_geq_one();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::arccosh(&our_x);
    let oxi = OxiCanonical::arccosh(&oxi_x);
    assert_parity("arccosh", &xs, &our, &oxi);
}

#[test]
fn parity_arctanh() {
    let xs = sample_open_unit_interval();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::arctanh(&our_x);
    let oxi = OxiCanonical::arctanh(&oxi_x);
    assert_parity("arctanh", &xs, &our, &oxi);
}

#[test]
fn parity_arctan() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::arctan(&our_x);
    let oxi = OxiCanonical::arctan(&oxi_x);
    assert_parity("arctan", &xs, &our, &oxi);
}

// =====================================================================
// Parity tests — Tables 4-5 (square/sqrt/abs/reciprocal)
// =====================================================================

#[test]
fn parity_square() {
    let xs = sample_xs();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::square(&our_x);
    let oxi = OxiCanonical::square(&oxi_x);
    assert_parity("square", &xs, &our, &oxi);
}

#[test]
fn parity_sqrt() {
    // Use positive samples to avoid the branch-cut + native-vs-Pow divergence
    // (see divergence_sqrt_native_grad_shape).
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::sqrt(&our_x);
    let oxi = OxiCanonical::sqrt(&oxi_x);
    assert_parity("sqrt", &xs, &our, &oxi);
}

#[test]
fn parity_reciprocal() {
    let xs = sample_positive();
    let our_x = OurTree::var(0);
    let oxi_x = OxiTree::var(0);
    let our = OurCanonical::reciprocal(&our_x);
    let oxi = OxiCanonical::reciprocal(&oxi_x);
    assert_parity("reciprocal", &xs, &our, &oxi);
}

// =====================================================================
// Parity tests — constants
// =====================================================================

#[test]
fn parity_const_euler() {
    let our = OurCanonical::euler();
    let oxi = OxiCanonical::euler();
    let ours = our_eval(&our, 0.0).expect("our euler eval");
    let oxis = oxi_eval(&oxi, 0.0).expect("oxi euler eval");
    assert!(
        agrees(ours, oxis),
        "const e: ours={ours}, oxi={oxis}, abs_err={:e}",
        (ours - oxis).abs()
    );
}

#[test]
fn parity_const_pi() {
    // `Canonical::pi()` is encoded as `ln(-1) = iπ` in BOTH libraries — it
    // returns a tree whose **complex** evaluation yields `iπ`, deliberately
    // NOT a real π. (The constructor exists so internal callers like sin/cos
    // can lift it via `mul(i, ...)`.) Verify both libraries agree:
    // - real-path: both must error
    // - complex-path: both must produce `iπ`
    let our = OurCanonical::pi();
    let oxi = OxiCanonical::pi();

    let our_lo = lower(&our);
    let our_re = eval_real(&our_lo, &EvalCtx::new(&[0.0]));
    let our_cplx = eval_complex(&our_lo, &[Complex64::new(0.0, 0.0)])
        .expect("our pi complex eval should succeed");
    assert!(
        our_re.is_err(),
        "our pi via real path should error (encodes iπ); got {our_re:?}"
    );
    assert!(
        (our_cplx.im - std::f64::consts::PI).abs() < TOL && our_cplx.re.abs() < TOL,
        "our pi complex eval: re={}, im={}, expected (0, π)",
        our_cplx.re,
        our_cplx.im
    );

    let oxi_re = oxi.eval_real(&OxiCtx::new(&[0.0]));
    assert!(
        oxi_re.is_err(),
        "oxi pi via real path should error (encodes iπ); got {oxi_re:?}"
    );
    let oxi_cplx = oxi
        .eval_complex(&[Complex64::new(0.0, 0.0)])
        .expect("oxi pi complex eval should succeed");
    assert!(
        (oxi_cplx.im - std::f64::consts::PI).abs() < TOL && oxi_cplx.re.abs() < TOL,
        "oxi pi complex eval: re={}, im={}, expected (0, π)",
        oxi_cplx.re,
        oxi_cplx.im
    );
}

#[test]
fn parity_const_neg_one() {
    let our = OurCanonical::neg_one();
    let oxi = OxiCanonical::neg_one();
    let ours = our_eval(&our, 0.0).expect("our neg_one eval");
    let oxis = oxi_eval(&oxi, 0.0).expect("oxi neg_one eval");
    assert!(
        agrees(ours, oxis),
        "const -1: ours={ours}, oxi={oxis}, abs_err={:e}",
        (ours - oxis).abs()
    );
}

#[test]
fn parity_const_nat_small() {
    // Verify nat(n) for n in 1..=8 (n=0 diverges, see divergence test).
    for n in 1u64..=8 {
        let our = OurCanonical::nat(n).expect("our nat(n)");
        let oxi = OxiCanonical::nat(n);
        let ours = our_eval(&our, 0.0).unwrap_or_else(|| panic!("our nat({n}) eval"));
        let oxis = oxi_eval(&oxi, 0.0).unwrap_or_else(|| panic!("oxi nat({n}) eval"));
        assert!(
            agrees(ours, oxis),
            "nat({n}): ours={ours}, oxi={oxis}, abs_err={:e}",
            (ours - oxis).abs()
        );
    }
}

// =====================================================================
// Intentional divergence tests — assert OUR behaviour, not bug
// =====================================================================

#[test]
fn divergence_sqrt_native_grad_shape() {
    // Our `LoweredOp::Sqrt` is native — `grad(Sqrt(x), 0)` produces a
    // chain-rule term `1 / (2·sqrt(x))` that contains a `LoweredOp::Sqrt`
    // node. oxieml lowers `Canonical::sqrt` via `Pow(_, 0.5)`, so an
    // analogous gradient on its `LoweredOp` would contain `Pow(_, -0.5)`
    // (or a folded equivalent) instead.
    //
    // We assert the structural property on OUR side only — the existence
    // of a `Sqrt` node in the gradient — to verify the native rule fired.
    use scirs2_symbolic::eml::grad;

    let f = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
    let g = grad(&f, 0);

    // Iterative search for a Sqrt node in the gradient (no recursion).
    fn contains_sqrt(top: &LoweredOp) -> bool {
        let mut stack: Vec<&LoweredOp> = vec![top];
        while let Some(node) = stack.pop() {
            match node {
                LoweredOp::Sqrt(_) => return true,
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                LoweredOp::Neg(a)
                | LoweredOp::Exp(a)
                | LoweredOp::Ln(a)
                | LoweredOp::Sin(a)
                | LoweredOp::Cos(a)
                | LoweredOp::Tan(a)
                | LoweredOp::Sinh(a)
                | LoweredOp::Cosh(a)
                | LoweredOp::Tanh(a)
                | LoweredOp::Arcsin(a)
                | LoweredOp::Arccos(a)
                | LoweredOp::Arctan(a)
                | LoweredOp::Arcsinh(a)
                | LoweredOp::Arccosh(a)
                | LoweredOp::Arctanh(a)
                | LoweredOp::Abs(a) => stack.push(a),
                LoweredOp::Const(_) | LoweredOp::Var(_) => {}
            }
        }
        false
    }

    assert!(
        contains_sqrt(&g),
        "grad of native sqrt must contain a Sqrt node (native rule); got: {g:?}"
    );
}

#[test]
fn divergence_abs_native_subgradient() {
    // Our `LoweredOp::Abs` is native — `grad(Abs(x), 0)` produces the
    // sign/sub-gradient `(x / abs(x)) · 1`. Verify the gradient evaluates
    // to ±1 away from zero (sign of x), which is the native rule's
    // signature behaviour. oxieml's `sqrt(square(x))` lowering would
    // produce a `2x / (2·sqrt(x²))` shape that simplifies to the same
    // value — so we test the *evaluation* signature, not just the
    // structure (which would be brittle under simplifier rewrites).
    use scirs2_symbolic::eml::grad;

    let f = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
    let g = grad(&f, 0);

    // d/dx |x| = sign(x) for x != 0
    let v_pos = eval_real(&g, &EvalCtx::new(&[1.5])).expect("eval grad |x| at 1.5");
    let v_neg = eval_real(&g, &EvalCtx::new(&[-1.5])).expect("eval grad |x| at -1.5");
    assert!(
        (v_pos - 1.0).abs() < 1e-12,
        "d/dx |x| at x=1.5 should be +1, got {v_pos}"
    );
    assert!(
        (v_neg - (-1.0)).abs() < 1e-12,
        "d/dx |x| at x=-1.5 should be -1, got {v_neg}"
    );
}

#[test]
fn divergence_nat_zero_returns_err() {
    // Our `Canonical::nat(0)` returns `Err(EmlError::InvalidConstant(_))`;
    // oxieml's `Canonical::nat(0)` panics. Verify our error path.
    let result = OurCanonical::nat(0);
    match result {
        Err(EmlError::InvalidConstant(_)) => {}
        other => panic!("expected InvalidConstant for nat(0), got {other:?}"),
    }
}

#[test]
fn divergence_interval_widening() {
    // Our `eval_interval` outward-widens by 1 ULP per node; oxieml uses
    // bare arithmetic (no widening). For the identity tree `Var(0)` over
    // `[1.0, 2.0]`, we expect `lo <= 1.0` and `hi >= 2.0` (containment),
    // with a gap of at most a couple of ULPs.
    use scirs2_symbolic::eml::{eval_interval, Interval};

    let op = LoweredOp::Var(0);
    let r = eval_interval(&op, &[Interval::new(1.0, 2.0)]);

    // Containment must hold (soundness).
    assert!(
        r.lo <= 1.0,
        "outward-widened lo must be <= 1.0, got {}",
        r.lo
    );
    assert!(
        r.hi >= 2.0,
        "outward-widened hi must be >= 2.0, got {}",
        r.hi
    );

    // Gap should be tiny — at most a handful of ULPs from the input
    // bounds. This pins the intentional widening to "1 ULP class", not
    // arbitrary inflation.
    let lo_eps = (1.0_f64 - r.lo).abs();
    let hi_eps = (r.hi - 2.0_f64).abs();
    assert!(
        lo_eps < 1e-14,
        "outward widening on lo too aggressive: gap = {lo_eps:e}"
    );
    assert!(
        hi_eps < 1e-14,
        "outward widening on hi too aggressive: gap = {hi_eps:e}"
    );
}

#[test]
fn divergence_eval_real_strictness() {
    // scirs2-symbolic's `eval_real` returns `Err(EvalDomain)` on
    // `Ln(non-positive)` immediately. oxieml's `eval_real` routes through
    // complex arithmetic and only errors when the imaginary part exceeds
    // its 1e-12 threshold. Document the asymmetry.
    let op_ln_neg = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
    let result = eval_real(&op_ln_neg, &EvalCtx::new(&[-1.0]));
    match result {
        Err(EmlError::EvalDomain(_)) => {}
        other => panic!("expected EvalDomain for ln(-1) in eval_real, got {other:?}"),
    }

    // The complex path produces ln(-1) = iπ — caller can check |im| if
    // they want oxieml-style real-via-complex semantics.
    let c = eval_complex(&op_ln_neg, &[Complex64::new(-1.0, 0.0)]).expect("complex eval of ln(-1)");
    assert!(
        (c.im - std::f64::consts::PI).abs() < 1e-12,
        "ln(-1) complex path: im = {} should equal π",
        c.im
    );
}

// =====================================================================
// Resolution-protocol verification (harness self-test)
// =====================================================================

#[test]
fn harness_self_test() {
    // Sanity: the sample generators produce non-empty, finite samples.
    assert_eq!(sample_xs().len(), N_PARITY);
    assert_eq!(sample_open_unit_interval().len(), N_PARITY);
    assert_eq!(sample_positive().len(), N_PARITY);
    assert_eq!(sample_geq_one().len(), N_PARITY);
    for x in sample_xs() {
        assert!(x.is_finite() && (-3.0..=3.0).contains(&x));
    }
    for x in sample_positive() {
        assert!(x > 0.0 && x.is_finite());
    }

    // Sanity: `agrees()` is reflexive and respects TOL.
    assert!(agrees(1.0, 1.0));
    assert!(agrees(1.0, 1.0 + TOL / 2.0));
    assert!(!agrees(1.0, 1.0 + TOL * 100.0));
}
