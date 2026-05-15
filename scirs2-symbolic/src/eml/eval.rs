//! Iterative stack-machine evaluator for `LoweredOp`.
//!
//! Two paths:
//! - [`eval_real`] — `f64` evaluation with `EvalCtx` carrying variable bindings
//! - [`eval_complex`] — `num_complex::Complex64` evaluation
//!
//! Both share the [`crate::eml::op::LoweredOp::to_oxi_ops`] flat tape and
//! a `Vec<T>` value stack. No recursion — `Canonical::sin(x)` produces a
//! 543-node-deep tree; recursive evaluation would blow the OS stack on any
//! plausible composition.
//!
//! # Examples
//!
//! Real path on a primitive `LoweredOp`:
//!
//! ```
//! use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
//!
//! // f(x, y) = x + y
//! let op = LoweredOp::Add(
//!     Box::new(LoweredOp::Var(0)),
//!     Box::new(LoweredOp::Var(1)),
//! );
//! let ctx = EvalCtx::new(&[3.0_f64, 4.0_f64]);
//! let result = eval_real(&op, &ctx).expect("eval");
//! assert!((result - 7.0).abs() < 1e-12);
//! ```
//!
//! Complex path through a canonical encoding (canonical `sin`/`cos` use
//! Euler's formula and contain `ln(-1)` in their lowered tree, so they
//! must be evaluated via the complex path):
//!
//! ```
//! use scirs2_symbolic::eml::{Canonical, EmlTree, lower, eval_complex};
//! use num_complex::Complex64;
//!
//! let x = EmlTree::var(0);
//! let formula = Canonical::add(&Canonical::sin(&x), &Canonical::cos(&x));
//! let lowered = lower(&formula);
//! let result = eval_complex(&lowered, &[Complex64::new(0.5, 0.0)]).expect("eval");
//! // result.re ≈ sin(0.5) + cos(0.5) ≈ 1.357
//! let expected = 0.5_f64.sin() + 0.5_f64.cos();
//! assert!((result.re - expected).abs() < 1e-6);
//! ```

// Adapted from oxieml v0.1.0, src/lower.rs (lines 538-633)
//
// Iterative stack-machine evaluator. Op match arms preserved verbatim;
// API surface clean-room: returns `Result<f64, EmlError>` (oxieml's port
// returns `f64` directly with NaN on bad inputs and `unwrap_or(f64::NAN)`
// on stack underflow). Native `Sqrt` and `Abs` arms are scirs2-symbolic
// additions matching the `LoweredOp` enum's native-variant policy.

use crate::eml::op::{LoweredOp, OxiOp};
use crate::error::EmlError;
use num_complex::Complex64;

/// Evaluation context — variable bindings indexed by `usize`.
///
/// Holds a borrowed slice; the caller owns the binding storage. This is
/// deliberately minimal — for higher-level features (named bindings, scoped
/// shadowing) build adapters on top.
#[derive(Clone, Debug)]
pub struct EvalCtx<'a> {
    bindings: &'a [f64],
}

impl<'a> EvalCtx<'a> {
    /// Create an evaluation context from a slice of bindings.
    pub fn new(bindings: &'a [f64]) -> Self {
        Self { bindings }
    }

    /// Get the binding for variable `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<f64> {
        self.bindings.get(idx).copied()
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// True if no bindings.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Borrow the underlying slice (for plumbing into `eval_ops_real`).
    pub fn as_slice(&self) -> &'a [f64] {
        self.bindings
    }
}

// ---------------------------------------------------------------------
// Real path — f64
// ---------------------------------------------------------------------

/// Evaluate a `LoweredOp` at real values.
///
/// # Errors
/// - [`EmlError::UnboundVariableIndex`] — variable index exceeds context bindings
/// - [`EmlError::DivisionByZero`] — division by zero (or by a tiny denominator
///   `< 1e-300` in absolute value, to catch underflow-to-zero before it
///   produces an `Inf`)
/// - [`EmlError::EvalDomain`] — domain violation (`ln` of non-positive,
///   `sqrt` of negative, `arcsin`/`arccos` outside `[-1, 1]`, etc.)
pub fn eval_real(op: &LoweredOp, ctx: &EvalCtx<'_>) -> Result<f64, EmlError> {
    let ops = op.to_oxi_ops();
    eval_ops_real(&ops, ctx.bindings)
}

/// Evaluate a flat `OxiOp` tape at real values.
///
/// Same error contract as [`eval_real`]. Exposed so callers that have already
/// flattened a `LoweredOp` (e.g. for caching) can skip the re-flatten step.
pub fn eval_ops_real(ops: &[OxiOp], vars: &[f64]) -> Result<f64, EmlError> {
    let mut stack: Vec<f64> = Vec::with_capacity(ops.len());

    for op in ops {
        match op {
            OxiOp::Const(c) => stack.push(*c),
            OxiOp::Var(i) => {
                let v = vars
                    .get(*i)
                    .copied()
                    .ok_or(EmlError::UnboundVariableIndex {
                        idx: *i,
                        len: vars.len(),
                    })?;
                stack.push(v);
            }
            OxiOp::Add => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Add".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Add".into()))?;
                stack.push(a + b);
            }
            OxiOp::Sub => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sub".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sub".into()))?;
                stack.push(a - b);
            }
            OxiOp::Mul => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Mul".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Mul".into()))?;
                stack.push(a * b);
            }
            OxiOp::Div => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Div".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Div".into()))?;
                if b.abs() < 1e-300 {
                    return Err(EmlError::DivisionByZero);
                }
                stack.push(a / b);
            }
            OxiOp::Pow => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Pow".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Pow".into()))?;
                stack.push(a.powf(b));
            }
            OxiOp::Neg => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Neg".into()))?;
                stack.push(-c);
            }
            OxiOp::Exp => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Exp".into()))?;
                stack.push(c.exp());
            }
            OxiOp::Ln => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Ln".into()))?;
                if c <= 0.0 {
                    return Err(EmlError::EvalDomain(format!(
                        "ln({}) — argument must be positive",
                        c
                    )));
                }
                stack.push(c.ln());
            }
            OxiOp::Sin => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sin".into()))?;
                stack.push(c.sin());
            }
            OxiOp::Cos => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Cos".into()))?;
                stack.push(c.cos());
            }
            OxiOp::Tan => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Tan".into()))?;
                stack.push(c.tan());
            }
            OxiOp::Sinh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sinh".into()))?;
                stack.push(c.sinh());
            }
            OxiOp::Cosh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Cosh".into()))?;
                stack.push(c.cosh());
            }
            OxiOp::Tanh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Tanh".into()))?;
                stack.push(c.tanh());
            }
            OxiOp::Arcsin => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arcsin".into()))?;
                if !(-1.0..=1.0).contains(&c) {
                    return Err(EmlError::EvalDomain(format!(
                        "arcsin({}) — argument must be in [-1, 1]",
                        c
                    )));
                }
                stack.push(c.asin());
            }
            OxiOp::Arccos => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arccos".into()))?;
                if !(-1.0..=1.0).contains(&c) {
                    return Err(EmlError::EvalDomain(format!(
                        "arccos({}) — argument must be in [-1, 1]",
                        c
                    )));
                }
                stack.push(c.acos());
            }
            OxiOp::Arctan => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arctan".into()))?;
                stack.push(c.atan());
            }
            OxiOp::Arcsinh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arcsinh".into()))?;
                stack.push(c.asinh());
            }
            OxiOp::Arccosh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arccosh".into()))?;
                if c < 1.0 {
                    return Err(EmlError::EvalDomain(format!(
                        "arccosh({}) — argument must be ≥ 1",
                        c
                    )));
                }
                stack.push(c.acosh());
            }
            OxiOp::Arctanh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arctanh".into()))?;
                if !(-1.0..1.0).contains(&c) {
                    return Err(EmlError::EvalDomain(format!(
                        "arctanh({}) — argument must be in (-1, 1)",
                        c
                    )));
                }
                stack.push(c.atanh());
            }
            OxiOp::Sqrt => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sqrt".into()))?;
                if c < 0.0 {
                    return Err(EmlError::EvalDomain(format!(
                        "sqrt({}) — argument must be ≥ 0",
                        c
                    )));
                }
                stack.push(c.sqrt());
            }
            OxiOp::Abs => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Abs".into()))?;
                stack.push(c.abs());
            }
        }
    }

    stack
        .pop()
        .ok_or_else(|| EmlError::EvalDomain("evaluation stack empty after run".into()))
}

// ---------------------------------------------------------------------
// Complex path — Complex64
// ---------------------------------------------------------------------

/// Evaluate a `LoweredOp` at complex values.
///
/// All operations are supported without domain restrictions — the complex
/// domain is closed under `exp`, `ln`, `sqrt`, etc. (`ln` uses the principal
/// branch). `Abs` returns a real-valued result lifted to `Complex64` (i.e.
/// `Complex64::new(|z|, 0.0)`), matching `numpy.abs` semantics.
///
/// # Errors
/// - [`EmlError::UnboundVariableIndex`] — variable index exceeds bindings
/// - [`EmlError::DivisionByZero`] — division by a complex with `norm < 1e-300`
pub fn eval_complex(op: &LoweredOp, vars: &[Complex64]) -> Result<Complex64, EmlError> {
    let ops = op.to_oxi_ops();
    eval_ops_complex(&ops, vars)
}

/// Evaluate a flat `OxiOp` tape at complex values.
///
/// Same error contract as [`eval_complex`]. Exposed so callers that have
/// already flattened a `LoweredOp` can skip the re-flatten step.
pub fn eval_ops_complex(ops: &[OxiOp], vars: &[Complex64]) -> Result<Complex64, EmlError> {
    let mut stack: Vec<Complex64> = Vec::with_capacity(ops.len());

    for op in ops {
        match op {
            OxiOp::Const(c) => stack.push(Complex64::new(*c, 0.0)),
            OxiOp::Var(i) => {
                let v = vars
                    .get(*i)
                    .copied()
                    .ok_or(EmlError::UnboundVariableIndex {
                        idx: *i,
                        len: vars.len(),
                    })?;
                stack.push(v);
            }
            OxiOp::Add => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Add".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Add".into()))?;
                stack.push(a + b);
            }
            OxiOp::Sub => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sub".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sub".into()))?;
                stack.push(a - b);
            }
            OxiOp::Mul => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Mul".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Mul".into()))?;
                stack.push(a * b);
            }
            OxiOp::Div => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Div".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Div".into()))?;
                if b.norm() < 1e-300 {
                    return Err(EmlError::DivisionByZero);
                }
                stack.push(a / b);
            }
            OxiOp::Pow => {
                let b = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Pow".into()))?;
                let a = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Pow".into()))?;
                stack.push(a.powc(b));
            }
            OxiOp::Neg => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Neg".into()))?;
                stack.push(-c);
            }
            OxiOp::Exp => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Exp".into()))?;
                stack.push(c.exp());
            }
            OxiOp::Ln => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Ln".into()))?;
                stack.push(c.ln());
            }
            OxiOp::Sin => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sin".into()))?;
                stack.push(c.sin());
            }
            OxiOp::Cos => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Cos".into()))?;
                stack.push(c.cos());
            }
            OxiOp::Tan => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Tan".into()))?;
                stack.push(c.tan());
            }
            OxiOp::Sinh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sinh".into()))?;
                stack.push(c.sinh());
            }
            OxiOp::Cosh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Cosh".into()))?;
                stack.push(c.cosh());
            }
            OxiOp::Tanh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Tanh".into()))?;
                stack.push(c.tanh());
            }
            OxiOp::Arcsin => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arcsin".into()))?;
                stack.push(c.asin());
            }
            OxiOp::Arccos => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arccos".into()))?;
                stack.push(c.acos());
            }
            OxiOp::Arctan => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arctan".into()))?;
                stack.push(c.atan());
            }
            OxiOp::Arcsinh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arcsinh".into()))?;
                stack.push(c.asinh());
            }
            OxiOp::Arccosh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arccosh".into()))?;
                stack.push(c.acosh());
            }
            OxiOp::Arctanh => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Arctanh".into()))?;
                stack.push(c.atanh());
            }
            OxiOp::Sqrt => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Sqrt".into()))?;
                stack.push(c.sqrt());
            }
            OxiOp::Abs => {
                let c = stack
                    .pop()
                    .ok_or_else(|| EmlError::EvalDomain("stack underflow at Abs".into()))?;
                stack.push(Complex64::new(c.norm(), 0.0));
            }
        }
    }

    stack
        .pop()
        .ok_or_else(|| EmlError::EvalDomain("evaluation stack empty after run".into()))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::{lower, Canonical, EmlTree};

    const TOL: f64 = 1e-12;

    // ----- Real path: leaves & primitives -----------------------------

    #[test]
    fn const_eval() {
        // 3.15 (not 3.14) — `clippy::approx_constant` flags any value within
        // ULP-distance of `f64::consts::PI`.
        let op = LoweredOp::Const(3.15);
        let ctx = EvalCtx::new(&[]);
        match eval_real(&op, &ctx) {
            Ok(v) => assert!((v - 3.15).abs() < TOL),
            Err(e) => panic!("eval_real Const failed: {e:?}"),
        }
    }

    #[test]
    fn var_eval() {
        let op = LoweredOp::Var(0);
        let ctx = EvalCtx::new(&[42.0]);
        match eval_real(&op, &ctx) {
            Ok(v) => assert_eq!(v, 42.0),
            Err(e) => panic!("eval_real Var failed: {e:?}"),
        }
    }

    #[test]
    fn add_eval() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let ctx = EvalCtx::new(&[3.0, 4.0]);
        match eval_real(&op, &ctx) {
            Ok(v) => assert_eq!(v, 7.0),
            Err(e) => panic!("eval_real Add failed: {e:?}"),
        }
    }

    // ----- Real path: errors ------------------------------------------

    #[test]
    fn unbound_var_returns_err() {
        let op = LoweredOp::Var(5);
        let ctx = EvalCtx::new(&[1.0, 2.0]);
        assert!(matches!(
            eval_real(&op, &ctx),
            Err(EmlError::UnboundVariableIndex { idx: 5, len: 2 })
        ));
    }

    #[test]
    fn div_by_zero() {
        let op = LoweredOp::Div(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(LoweredOp::Const(0.0)),
        );
        let ctx = EvalCtx::new(&[]);
        assert!(matches!(
            eval_real(&op, &ctx),
            Err(EmlError::DivisionByZero)
        ));
    }

    #[test]
    fn ln_negative_returns_err() {
        let op = LoweredOp::Ln(Box::new(LoweredOp::Const(-1.0)));
        let ctx = EvalCtx::new(&[]);
        assert!(matches!(eval_real(&op, &ctx), Err(EmlError::EvalDomain(_))));
    }

    #[test]
    fn sqrt_negative_returns_err() {
        let op = LoweredOp::Sqrt(Box::new(LoweredOp::Const(-1.0)));
        let ctx = EvalCtx::new(&[]);
        assert!(matches!(eval_real(&op, &ctx), Err(EmlError::EvalDomain(_))));
    }

    // ----- Real path: canonical compositions --------------------------
    //
    // NOTE: `Canonical::sin` / `Canonical::cos` encode the Euler formula
    // `(exp(ix) ± exp(-ix))/(...)` and contain `ln(-1)` in the lowered
    // tree (see `canonical.rs:160-211` — the doc-comment explicitly says
    // "evaluates correctly through the complex evaluation path").
    // Real-path evaluation of the canonical-sin tree therefore hits a
    // domain violation by construction. The corresponding tests are
    // routed through `eval_complex`. A separate primitive-op depth test
    // exercises the real evaluator at scale.

    #[test]
    fn sin_via_canonical_eval_complex() {
        // sin(0.5) via Canonical → lower → eval_complex. The 543-deep
        // tree introduces FP roundoff; widen tol to 1e-10.
        let x = EmlTree::var(0);
        let formula = Canonical::sin(&x);
        let lowered = lower(&formula);
        let result = match eval_complex(&lowered, &[Complex64::new(0.5, 0.0)]) {
            Ok(v) => v,
            Err(e) => panic!("eval_complex(canonical sin) failed: {e:?}"),
        };
        let expected = (0.5_f64).sin();
        assert!(
            (result.re - expected).abs() < 1e-10,
            "re: got {}, expected {}",
            result.re,
            expected
        );
        assert!(
            result.im.abs() < 1e-10,
            "im should be ~0, got {}",
            result.im
        );
    }

    #[test]
    fn cos_via_canonical_eval_complex() {
        let x = EmlTree::var(0);
        let formula = Canonical::cos(&x);
        let lowered = lower(&formula);
        let result = match eval_complex(&lowered, &[Complex64::new(0.5, 0.0)]) {
            Ok(v) => v,
            Err(e) => panic!("eval_complex(canonical cos) failed: {e:?}"),
        };
        let expected = (0.5_f64).cos();
        assert!(
            (result.re - expected).abs() < 1e-10,
            "re: got {}, expected {}",
            result.re,
            expected
        );
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn deep_sin_no_overflow() {
        // sin(sin(x)) — composition of canonical sin, depth ≈ 1086.
        // Recursive eval would blow the stack. Iterative must succeed.
        // This is the load-bearing regression gate for the iterative
        // pattern — a panic here means the work stack reverted to recursion.
        // Routed through `eval_complex` because canonical sin's tree
        // contains `ln(-1)` (see module doc above).
        let x = EmlTree::var(0);
        let inner_sin = Canonical::sin(&x);
        let outer_sin = Canonical::sin(&inner_sin);
        let lowered = lower(&outer_sin);
        let result = match eval_complex(&lowered, &[Complex64::new(0.3, 0.0)]) {
            Ok(v) => v,
            Err(e) => panic!("deep_sin_no_overflow eval failed: {e:?}"),
        };
        let expected = (0.3_f64).sin().sin();
        // Tolerance widened: deep canonical compositions accumulate FP error.
        assert!(
            (result.re - expected).abs() < 1e-8,
            "re: got {}, expected {}",
            result.re,
            expected
        );
        assert!(result.im.abs() < 1e-8, "im should be ~0, got {}", result.im);
    }

    #[test]
    fn deep_primitive_chain_no_overflow_real() {
        // Real-path depth gate using primitive `LoweredOp::Sin` (NOT the
        // canonical encoding). Builds Sin(Sin(...Sin(Var(0))...)) 5000-deep;
        // a recursive evaluator would blow the OS stack here. Iterative
        // must succeed.
        let mut op = LoweredOp::Var(0);
        for _ in 0..5000 {
            op = LoweredOp::Sin(Box::new(op));
        }
        let ctx = EvalCtx::new(&[0.5_f64]);
        match eval_real(&op, &ctx) {
            Ok(v) => {
                // Iterated sin contracts toward 0; result must be finite
                // and small.
                assert!(v.is_finite(), "result not finite: {v}");
                assert!(v.abs() < 1.0, "result {} out of expected bound", v);
            }
            Err(e) => panic!("deep_primitive_chain_no_overflow_real failed: {e:?}"),
        }
    }

    // ----- Complex path -----------------------------------------------

    #[test]
    fn complex_sin_at_imag_unit() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let i = Complex64::new(0.0, 1.0);
        let result = match eval_complex(&op, &[i]) {
            Ok(v) => v,
            Err(e) => panic!("eval_complex(sin) failed: {e:?}"),
        };
        // sin(i) = i * sinh(1)
        let expected = Complex64::new(0.0, 1.0_f64.sinh());
        assert!((result - expected).norm() < TOL);
    }

    #[test]
    fn complex_exp_at_zero() {
        let op = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let result = match eval_complex(&op, &[Complex64::new(0.0, 0.0)]) {
            Ok(v) => v,
            Err(e) => panic!("eval_complex(exp) failed: {e:?}"),
        };
        assert!((result - Complex64::new(1.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn complex_ln_negative() {
        // ln(-1) = i*pi (principal branch).
        let op = LoweredOp::Ln(Box::new(LoweredOp::Const(-1.0)));
        let result = match eval_complex(&op, &[]) {
            Ok(v) => v,
            Err(e) => panic!("eval_complex(ln(-1)) failed: {e:?}"),
        };
        let expected = Complex64::new(0.0, std::f64::consts::PI);
        assert!((result - expected).norm() < TOL);
    }

    #[test]
    fn complex_unbound_var() {
        let op = LoweredOp::Var(7);
        let r = eval_complex(&op, &[Complex64::new(1.0, 0.0)]);
        assert!(matches!(
            r,
            Err(EmlError::UnboundVariableIndex { idx: 7, len: 1 })
        ));
    }

    #[test]
    fn complex_div_by_zero() {
        let op = LoweredOp::Div(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(LoweredOp::Const(0.0)),
        );
        let r = eval_complex(&op, &[]);
        assert!(matches!(r, Err(EmlError::DivisionByZero)));
    }

    // ----- Random parity ----------------------------------------------

    #[test]
    fn random_parity_real() {
        // 100 random points: scalar formulas vs their f64::* expected.
        // Local type alias avoids `clippy::type_complexity` on the Vec elem.
        type ScalarParityCase = (LoweredOp, fn(f64) -> f64);
        let test_cases: Vec<ScalarParityCase> = vec![
            (LoweredOp::Sin(Box::new(LoweredOp::Var(0))), |x| x.sin()),
            (LoweredOp::Cos(Box::new(LoweredOp::Var(0))), |x| x.cos()),
            (LoweredOp::Exp(Box::new(LoweredOp::Var(0))), |x| x.exp()),
            (LoweredOp::Sinh(Box::new(LoweredOp::Var(0))), |x| x.sinh()),
            (LoweredOp::Cosh(Box::new(LoweredOp::Var(0))), |x| x.cosh()),
            (LoweredOp::Tanh(Box::new(LoweredOp::Var(0))), |x| x.tanh()),
        ];
        for (op, expected_fn) in test_cases {
            for _ in 0..100 {
                let x = (rand_simple() - 0.5) * 4.0;
                let bindings = [x];
                let ctx = EvalCtx::new(&bindings);
                let result = match eval_real(&op, &ctx) {
                    Ok(v) => v,
                    Err(e) => panic!("random_parity_real failed at op={op:?}, x={x}: {e:?}"),
                };
                let expected = expected_fn(x);
                let abs_err = (result - expected).abs();
                let rel_ok = expected.abs() > 0.0 && (abs_err / expected.abs()) < 1e-12;
                assert!(
                    abs_err < 1e-12 || rel_ok,
                    "op={:?}, x={}: got {}, expected {} (abs_err={})",
                    op,
                    x,
                    result,
                    expected,
                    abs_err
                );
            }
        }
    }

    // Simple LCG for tests (avoid pulling rand into test build).
    fn rand_simple() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = const { Cell::new(12345) };
        }
        SEED.with(|s| {
            let v = s.get();
            let next = v
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(next);
            (next >> 32) as f64 / (1u64 << 32) as f64
        })
    }
}
