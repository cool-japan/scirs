//! Symbolic gradient on `LoweredOp`.
//!
//! Computes `d/dx_i f` symbolically via chain rule, product rule, etc.
//! Result passes through `simplify_op` before return — caller sees
//! canonical-ish form.
//!
//! # Intentional divergences from oxieml
//!
//! - **`Pow` constant-exponent fast path**: when the exponent is a `Const(n)`,
//!   we emit `n · base^(n-1) · base'` directly (mirrors `Expr::diff` behaviour
//!   in `src/diff.rs:93-108`). Avoids `ln(neg)` for integer-power formulas.
//! - **`Sqrt` native rule**: `d/dx √f = f' / (2·√f)`. Not lowered to
//!   `Pow(x, 0.5)`; would blow up at `x=0` via `(0.5)·x^(-0.5)·dx`.
//!   Documented as undefined at f=0 (returns NaN at eval).
//! - **`Abs` non-differentiable at 0**: returns subgradient `(f / abs(f)) · f'`
//!   matching legacy `src/diff.rs:135-138`. Eval at f=0 → `0/0 = NaN`.
//!   Document recommendation: use `sqrt(x² + ε)` smoothing for differentiable
//!   approximation.
//!
//! # Adapted from oxieml v0.1.0, `src/lower_grad.rs` (lines 136-303)
//!
//! Chain/product/quotient rules preserved verbatim; native `Sqrt`/`Abs`
//! branches added (oxieml uses `Pow(_, 0.5)` / `sqrt(square(_))` which blow
//! up at 0). Constant-exponent `Pow` fast path added per scirs2-symbolic
//! divergence. The recursive shape of the oxieml original is rewritten
//! here as an iterative post-order work-stack walk to avoid OS-stack
//! overflow on the 543-node-deep `Canonical::sin` tree and similar deep
//! canonical encodings.

#![warn(missing_docs)]

use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;

/// Compute the symbolic gradient `d/dx_wrt f`.
///
/// Result is passed through [`simplify_op`] for canonical form.
///
/// # Examples
/// ```
/// use scirs2_symbolic::eml::{grad, LoweredOp};
/// let f = LoweredOp::Mul(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Var(0)),
/// );
/// let g = grad(&f, 0);
/// // g represents 2*x_0 (after simplify)
/// # let _ = g;
/// ```
pub fn grad(f: &LoweredOp, wrt: usize) -> LoweredOp {
    let raw = raw_grad(f, wrt);
    simplify_op(&raw)
}

/// Compute the gradient with respect to every variable.
///
/// Returns `vec[i] = grad(f, i)` for `i in 0..f.count_vars()`.
pub fn grad_all(f: &LoweredOp) -> Vec<LoweredOp> {
    let n = f.count_vars();
    (0..n).map(|i| grad(f, i)).collect()
}

/// Build a Jacobian for a scalar-valued `f` over `n_vars` independent
/// variables.
///
/// Returns `vec[i] = grad(f, i)`. This is identical to [`grad_all`] when the
/// caller wants exactly `n_vars` partials regardless of `f.count_vars()`
/// (e.g. when `f` does not mention every variable in the input space).
///
/// For a vector-valued function, decompose into components and call
/// [`jacobian`] per component, stacking the rows.
pub fn jacobian(f: &LoweredOp, n_vars: usize) -> Vec<LoweredOp> {
    (0..n_vars).map(|i| grad(f, i)).collect()
}

/// Build a Hessian: `h[i][j] = grad(grad(f, i), j)`. `n_vars × n_vars`.
///
/// Each entry is independently simplified.
pub fn hessian(f: &LoweredOp, n_vars: usize) -> Vec<Vec<LoweredOp>> {
    (0..n_vars)
        .map(|i| {
            let gi = grad(f, i);
            (0..n_vars).map(|j| grad(&gi, j)).collect()
        })
        .collect()
}

/// Internal raw gradient (no simplify pass).
///
/// Iterative — uses a post-order work-stack so deep canonical trees
/// (e.g. `Canonical::sin`'s 543-deep encoding) do not overflow the OS
/// stack. The traversal mirrors [`crate::eml::op::LoweredOp::to_oxi_ops`]:
/// each node is pushed twice (pre-visit / post-visit). On the post-visit
/// pass we pop the gradients of the children from `stack` (right then
/// left, since right was pushed last and pops first... wait — left first
/// for binary ops because left is pushed last in the work queue and pops
/// first when scheduling, so its gradient lands on `stack` first; on the
/// post-visit we therefore pop right then left).
pub(crate) fn raw_grad(f: &LoweredOp, wrt: usize) -> LoweredOp {
    let mut work: Vec<(&LoweredOp, bool)> = vec![(f, false)];
    let mut stack: Vec<LoweredOp> = Vec::new();

    while let Some((node, visited)) = work.pop() {
        if visited {
            // Post-visit: pop child gradients (matching the order in which
            // children were scheduled) and apply the differentiation rule.
            let result = match node {
                LoweredOp::Const(_) => LoweredOp::Const(0.0),
                LoweredOp::Var(i) => {
                    if *i == wrt {
                        LoweredOp::Const(1.0)
                    } else {
                        LoweredOp::Const(0.0)
                    }
                }
                LoweredOp::Add(_, _) => {
                    // Right was pushed last in pre-visit (so it pops first
                    // off `work` and lands on `stack` first); left lands
                    // second. Pop right first, then left.
                    let db = stack.pop().expect("post-order: right gradient");
                    let da = stack.pop().expect("post-order: left gradient");
                    LoweredOp::Add(Box::new(da), Box::new(db))
                }
                LoweredOp::Sub(_, _) => {
                    let db = stack.pop().expect("post-order: right gradient");
                    let da = stack.pop().expect("post-order: left gradient");
                    LoweredOp::Sub(Box::new(da), Box::new(db))
                }
                LoweredOp::Mul(a, b) => {
                    // Product rule: d(ab)/dx = a'·b + a·b'.
                    let db = stack.pop().expect("post-order: right gradient");
                    let da = stack.pop().expect("post-order: left gradient");
                    LoweredOp::Add(
                        Box::new(LoweredOp::Mul(Box::new(da), b.clone())),
                        Box::new(LoweredOp::Mul(a.clone(), Box::new(db))),
                    )
                }
                LoweredOp::Div(a, b) => {
                    // Quotient rule: d(a/b)/dx = (a'·b - a·b') / b².
                    let db = stack.pop().expect("post-order: right gradient");
                    let da = stack.pop().expect("post-order: left gradient");
                    LoweredOp::Div(
                        Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Mul(Box::new(da), b.clone())),
                            Box::new(LoweredOp::Mul(a.clone(), Box::new(db))),
                        )),
                        Box::new(LoweredOp::Mul(b.clone(), b.clone())),
                    )
                }
                LoweredOp::Pow(base, expo) => {
                    let dexpo = stack.pop().expect("post-order: exponent gradient");
                    let dbase = stack.pop().expect("post-order: base gradient");

                    // Constant-exponent fast path (must come first to avoid
                    // emitting ln(base) for plain integer powers).
                    if let LoweredOp::Const(n) = **expo {
                        // d(base^n)/dx = n · base^(n-1) · base'
                        LoweredOp::Mul(
                            Box::new(LoweredOp::Mul(
                                Box::new(LoweredOp::Const(n)),
                                Box::new(LoweredOp::Pow(
                                    base.clone(),
                                    Box::new(LoweredOp::Const(n - 1.0)),
                                )),
                            )),
                            Box::new(dbase),
                        )
                    } else {
                        // General form: d(a^b)/dx = a^b · (b'·ln(a) + b·a'/a).
                        LoweredOp::Mul(
                            Box::new(LoweredOp::Pow(base.clone(), expo.clone())),
                            Box::new(LoweredOp::Add(
                                Box::new(LoweredOp::Mul(
                                    Box::new(dexpo),
                                    Box::new(LoweredOp::Ln(base.clone())),
                                )),
                                Box::new(LoweredOp::Mul(
                                    expo.clone(),
                                    Box::new(LoweredOp::Div(Box::new(dbase), base.clone())),
                                )),
                            )),
                        )
                    }
                }
                LoweredOp::Neg(_) => {
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Neg(Box::new(dc))
                }
                LoweredOp::Exp(c) => {
                    // d(exp(c))/dx = exp(c) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(Box::new(LoweredOp::Exp(c.clone())), Box::new(dc))
                }
                LoweredOp::Ln(c) => {
                    // d(ln(c))/dx = dc / c.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(Box::new(dc), c.clone())
                }
                LoweredOp::Sin(c) => {
                    // d(sin(c))/dx = cos(c) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(Box::new(LoweredOp::Cos(c.clone())), Box::new(dc))
                }
                LoweredOp::Cos(c) => {
                    // d(cos(c))/dx = -sin(c) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Neg(Box::new(LoweredOp::Mul(
                        Box::new(LoweredOp::Sin(c.clone())),
                        Box::new(dc),
                    )))
                }
                LoweredOp::Tan(c) => {
                    // d(tan(c))/dx = (1 + tan²(c)) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(
                        Box::new(LoweredOp::Add(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(
                                Box::new(LoweredOp::Tan(c.clone())),
                                Box::new(LoweredOp::Tan(c.clone())),
                            )),
                        )),
                        Box::new(dc),
                    )
                }
                LoweredOp::Sinh(c) => {
                    // d(sinh(c))/dx = cosh(c) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(Box::new(LoweredOp::Cosh(c.clone())), Box::new(dc))
                }
                LoweredOp::Cosh(c) => {
                    // d(cosh(c))/dx = sinh(c) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(Box::new(LoweredOp::Sinh(c.clone())), Box::new(dc))
                }
                LoweredOp::Tanh(c) => {
                    // d(tanh(c))/dx = (1 - tanh²(c)) · dc.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(
                        Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(
                                Box::new(LoweredOp::Tanh(c.clone())),
                                Box::new(LoweredOp::Tanh(c.clone())),
                            )),
                        )),
                        Box::new(dc),
                    )
                }
                LoweredOp::Arcsin(c) => {
                    // d(arcsin(c))/dx = dc / sqrt(1 - c²).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Sqrt(Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                        )))),
                    )
                }
                LoweredOp::Arccos(c) => {
                    // d(arccos(c))/dx = -dc / sqrt(1 - c²).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Neg(Box::new(LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Sqrt(Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                        )))),
                    )))
                }
                LoweredOp::Arctan(c) => {
                    // d(arctan(c))/dx = dc / (1 + c²).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Add(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                        )),
                    )
                }
                LoweredOp::Arcsinh(c) => {
                    // d(arcsinh(c))/dx = dc / sqrt(c² + 1).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Sqrt(Box::new(LoweredOp::Add(
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                            Box::new(LoweredOp::Const(1.0)),
                        )))),
                    )
                }
                LoweredOp::Arccosh(c) => {
                    // d(arccosh(c))/dx = dc / sqrt(c² - 1).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Sqrt(Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                            Box::new(LoweredOp::Const(1.0)),
                        )))),
                    )
                }
                LoweredOp::Arctanh(c) => {
                    // d(arctanh(c))/dx = dc / (1 - c²).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Sub(
                            Box::new(LoweredOp::Const(1.0)),
                            Box::new(LoweredOp::Mul(c.clone(), c.clone())),
                        )),
                    )
                }
                LoweredOp::Sqrt(c) => {
                    // d(sqrt(c))/dx = dc / (2 · sqrt(c)) — NATIVE rule.
                    // Undefined at c=0 (returns NaN at eval).
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Div(
                        Box::new(dc),
                        Box::new(LoweredOp::Mul(
                            Box::new(LoweredOp::Const(2.0)),
                            Box::new(LoweredOp::Sqrt(c.clone())),
                        )),
                    )
                }
                LoweredOp::Abs(c) => {
                    // d(abs(c))/dx = (c / abs(c)) · dc — subgradient at c=0.
                    // For a differentiable approximation use `sqrt(c² + ε)`.
                    let dc = stack.pop().expect("post-order: child gradient");
                    LoweredOp::Mul(
                        Box::new(LoweredOp::Div(
                            c.clone(),
                            Box::new(LoweredOp::Abs(c.clone())),
                        )),
                        Box::new(dc),
                    )
                }
            };
            stack.push(result);
        } else {
            // Pre-visit: schedule post-visit + push children. Push right
            // first so left is the next thing popped; this means left's
            // post-visit (and gradient push onto `stack`) happens first.
            match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    work.push((node, true));
                }
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    work.push((node, true));
                    work.push((b, false));
                    work.push((a, false));
                }
                LoweredOp::Neg(c)
                | LoweredOp::Exp(c)
                | LoweredOp::Ln(c)
                | LoweredOp::Sin(c)
                | LoweredOp::Cos(c)
                | LoweredOp::Tan(c)
                | LoweredOp::Sinh(c)
                | LoweredOp::Cosh(c)
                | LoweredOp::Tanh(c)
                | LoweredOp::Arcsin(c)
                | LoweredOp::Arccos(c)
                | LoweredOp::Arctan(c)
                | LoweredOp::Arcsinh(c)
                | LoweredOp::Arccosh(c)
                | LoweredOp::Arctanh(c)
                | LoweredOp::Sqrt(c)
                | LoweredOp::Abs(c) => {
                    work.push((node, true));
                    work.push((c, false));
                }
            }
        }
    }

    stack.pop().expect("post-order: result on stack")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn grad_const_is_zero() {
        // 3.15 (not 3.14) — `clippy::approx_constant` flags any value within
        // ULP-distance of `f64::consts::PI`.
        let f = LoweredOp::Const(3.15);
        assert_eq!(grad(&f, 0), LoweredOp::Const(0.0));
    }

    #[test]
    fn grad_var_self() {
        let f = LoweredOp::Var(0);
        assert_eq!(grad(&f, 0), LoweredOp::Const(1.0));
    }

    #[test]
    fn grad_var_other() {
        let f = LoweredOp::Var(0);
        assert_eq!(grad(&f, 1), LoweredOp::Const(0.0));
    }

    #[test]
    fn grad_x_squared() {
        // f = x^2 → f' = 2x
        let f = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        let g = grad(&f, 0);
        // Evaluate at x=3: should be 6.
        let r = eval_real(&g, &EvalCtx::new(&[3.0])).expect("eval");
        assert!(approx_eq(r, 6.0), "g(3) = {} (expected 6)", r);
    }

    #[test]
    fn grad_sin() {
        // f = sin(x) → f' = cos(x)
        let f = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let g = grad(&f, 0);
        let r = eval_real(&g, &EvalCtx::new(&[0.5])).expect("eval");
        assert!(approx_eq(r, 0.5_f64.cos()));
    }

    #[test]
    fn grad_xy() {
        // f = x*y, df/dx = y, df/dy = x
        let f = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let gx = grad(&f, 0);
        let gy = grad(&f, 1);
        let rx = eval_real(&gx, &EvalCtx::new(&[3.0, 5.0])).expect("eval gx");
        let ry = eval_real(&gy, &EvalCtx::new(&[3.0, 5.0])).expect("eval gy");
        assert!(approx_eq(rx, 5.0));
        assert!(approx_eq(ry, 3.0));
    }

    #[test]
    fn grad_chain_rule() {
        // f = sin(x²) → f' = cos(x²) · 2x
        let f = LoweredOp::Sin(Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        )));
        let g = grad(&f, 0);
        let x = 0.7;
        let r = eval_real(&g, &EvalCtx::new(&[x])).expect("eval");
        let expected = (x * x).cos() * 2.0 * x;
        assert!(
            approx_eq(r, expected),
            "g({}) = {} (expected {})",
            x,
            r,
            expected
        );
    }

    #[test]
    fn grad_quotient_rule() {
        // f = x / (x² + 1) → f' = (1 - x²) / (x² + 1)²
        let denom = LoweredOp::Add(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Const(1.0)),
        );
        let f = LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(denom));
        let g = grad(&f, 0);
        for x in [-1.5, -0.3, 0.7, 2.4] {
            let r = eval_real(&g, &EvalCtx::new(&[x])).expect("eval");
            let denom_val = x * x + 1.0;
            let expected = (1.0 - x * x) / (denom_val * denom_val);
            assert!(
                (r - expected).abs() < 1e-10,
                "g({}) = {} (expected {})",
                x,
                r,
                expected
            );
        }
    }

    #[test]
    fn grad_pow_const_fast_path_no_ln() {
        // f = x³ via Pow(x, 3) — must produce 3·x²·1 (no ln(x) anywhere),
        // so eval at x=-1 (where ln is undefined) must succeed and return 3.
        let f = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0)));
        let g = grad(&f, 0);
        let r = eval_real(&g, &EvalCtx::new(&[-1.0])).expect("eval");
        assert!(approx_eq(r, 3.0), "g(-1) = {} (expected 3)", r);
    }

    #[test]
    fn grad_sqrt_native_rule() {
        // f = sqrt(x) → f' = 1 / (2·sqrt(x))
        let f = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        let g = grad(&f, 0);
        let x = 4.0;
        let r = eval_real(&g, &EvalCtx::new(&[x])).expect("eval");
        let expected = 1.0 / (2.0 * x.sqrt());
        assert!(
            approx_eq(r, expected),
            "g({}) = {} (expected {})",
            x,
            r,
            expected
        );
    }

    #[test]
    fn grad_abs_subgradient() {
        // f = |x| → f' = x / |x| (sign function); at x=2 → 1, at x=-3 → -1.
        let f = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
        let g = grad(&f, 0);
        let r_pos = eval_real(&g, &EvalCtx::new(&[2.0])).expect("eval pos");
        let r_neg = eval_real(&g, &EvalCtx::new(&[-3.0])).expect("eval neg");
        assert!(approx_eq(r_pos, 1.0), "g(2) = {} (expected 1)", r_pos);
        assert!(approx_eq(r_neg, -1.0), "g(-3) = {} (expected -1)", r_neg);
    }

    #[test]
    fn grad_central_difference_property() {
        // For a small bank of formulas, compare symbolic grad to a central
        // difference at a grid of points (1024-trial parity, per spec).
        let formulas: Vec<LoweredOp> = vec![
            // sin(x)
            LoweredOp::Sin(Box::new(LoweredOp::Var(0))),
            // cos(x)
            LoweredOp::Cos(Box::new(LoweredOp::Var(0))),
            // exp(x)
            LoweredOp::Exp(Box::new(LoweredOp::Var(0))),
            // x · x
            LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0))),
            // sin(x) + x
            LoweredOp::Add(
                Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
                Box::new(LoweredOp::Var(0)),
            ),
            // tanh(x)
            LoweredOp::Tanh(Box::new(LoweredOp::Var(0))),
            // arctan(x)
            LoweredOp::Arctan(Box::new(LoweredOp::Var(0))),
            // x³ via Pow constant-exponent
            LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0))),
            // sqrt(1 + x²)
            LoweredOp::Sqrt(Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Const(1.0)),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Var(0)),
                    Box::new(LoweredOp::Var(0)),
                )),
            ))),
            // sin(x²) — chain rule
            LoweredOp::Sin(Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Var(0)),
            ))),
        ];
        let mut total_trials = 0;
        for f in &formulas {
            let g = grad(f, 0);
            for seed in 0..120 {
                // Avoid 0 (singularity for some formulas) and stay in a
                // domain where everything is finite.
                let x = (seed as f64) * 0.013 + 0.25;
                let h = 1e-6;
                let f_plus = match eval_real(f, &EvalCtx::new(&[x + h])) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let f_minus = match eval_real(f, &EvalCtx::new(&[x - h])) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !f_plus.is_finite() || !f_minus.is_finite() {
                    continue;
                }
                let central = (f_plus - f_minus) / (2.0 * h);
                let symbolic = match eval_real(&g, &EvalCtx::new(&[x])) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !symbolic.is_finite() {
                    continue;
                }
                let tol = 1e-5_f64.max(symbolic.abs() * 1e-6);
                assert!(
                    (central - symbolic).abs() < tol,
                    "f={:?} x={} central={} symbolic={} diff={}",
                    f,
                    x,
                    central,
                    symbolic,
                    (central - symbolic).abs()
                );
                total_trials += 1;
            }
        }
        // Sanity check: we actually exercised a sizeable number of trials.
        assert!(total_trials >= 1024, "only {} trials ran", total_trials);
    }

    #[test]
    fn jacobian_2d() {
        let f = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Var(1)),
            )),
            Box::new(LoweredOp::Var(0)),
        );
        let j = jacobian(&f, 2);
        assert_eq!(j.len(), 2);
        // df/dx = y + 1, df/dy = x
        let r0 = eval_real(&j[0], &EvalCtx::new(&[2.0, 3.0])).expect("eval r0");
        let r1 = eval_real(&j[1], &EvalCtx::new(&[2.0, 3.0])).expect("eval r1");
        assert!(approx_eq(r0, 4.0));
        assert!(approx_eq(r1, 2.0));
    }

    #[test]
    fn grad_all_matches_grad_per_var() {
        let f = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Var(1)),
            )),
            Box::new(LoweredOp::Var(2)),
        );
        let gs = grad_all(&f);
        assert_eq!(gs.len(), 3);
        let pt = [4.0_f64, 7.0, 11.0];
        let r0 = eval_real(&gs[0], &EvalCtx::new(&pt)).expect("eval g0");
        let r1 = eval_real(&gs[1], &EvalCtx::new(&pt)).expect("eval g1");
        let r2 = eval_real(&gs[2], &EvalCtx::new(&pt)).expect("eval g2");
        assert!(approx_eq(r0, 7.0)); // df/dx0 = x1
        assert!(approx_eq(r1, 4.0)); // df/dx1 = x0
        assert!(approx_eq(r2, 1.0)); // df/dx2 = 1
    }

    #[test]
    fn hessian_quadratic() {
        // f = x²·y → ∂f/∂x = 2x·y, ∂f/∂y = x²
        // H = [[2y, 2x], [2x, 0]]
        let f = LoweredOp::Mul(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Var(1)),
        );
        let h = hessian(&f, 2);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert_eq!(h[1].len(), 2);
        let pt = [3.0_f64, 5.0];
        let r00 = eval_real(&h[0][0], &EvalCtx::new(&pt)).expect("eval h00"); // 2y = 10
        let r01 = eval_real(&h[0][1], &EvalCtx::new(&pt)).expect("eval h01"); // 2x = 6
        let r10 = eval_real(&h[1][0], &EvalCtx::new(&pt)).expect("eval h10"); // 2x = 6
        let r11 = eval_real(&h[1][1], &EvalCtx::new(&pt)).expect("eval h11"); // 0
        assert!(approx_eq(r00, 10.0), "h00 = {}", r00);
        assert!(approx_eq(r01, 6.0), "h01 = {}", r01);
        assert!(approx_eq(r10, 6.0), "h10 = {}", r10);
        assert!(approx_eq(r11, 0.0), "h11 = {}", r11);
    }

    #[test]
    fn grad_deep_chain_no_overflow() {
        // Build a deep right-chain Add tree and grad through it. Must not
        // overflow the OS stack — exercises the iterative work-stack walk.
        let mut op = LoweredOp::Var(0);
        for _ in 0..5_000 {
            op = LoweredOp::Add(
                Box::new(op),
                Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
            );
        }
        let g = grad(&op, 0);
        // df/dx = 1 + Σ cos(x), evaluate at x=0 → 1 + 5000 = 5001.
        let r = eval_real(&g, &EvalCtx::new(&[0.0])).expect("eval");
        assert!(approx_eq(r, 5001.0), "deep grad eval = {}", r);
    }
}
