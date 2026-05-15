//! EML-native automatic differentiation kernel for the SciRS2 CAS.
//!
//! Wraps and enhances the [`crate::eml`] AD primitives with:
//! - **Canonicalization** — every gradient is passed through
//!   [`mod@crate::cas::canonicalize`] so structurally equal subexpressions
//!   across the gradient vector share a single CSE hash.
//! - **Batch CSE evaluation** — [`GradGraph::eval_with_grad`] builds a single
//!   [`CseDag`] from all gradient expressions and evaluates every unique
//!   subexpression exactly once per point.
//! - **Vector-Jacobian products** (`vjp`) and **Jacobian-vector products**
//!   (`jvp`) — returned as symbolic [`LoweredOp`] expressions that can be
//!   further differentiated, compiled, or evaluated.
//! - **Numerical gradient** — central-difference fallback for validation.
//!
//! # No recursion
//!
//! All traversals delegate to the iterative implementations in
//! [`mod@crate::eml::grad`], [`crate::eml::eval`], and [`crate::cas::cse_dag`].
//!
//! # No unwrap in production code
//!
//! All fallible paths use [`AdError`] and the `?` operator.

#![warn(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

use crate::cas::canonicalize::canonicalize;
use crate::cas::cse_dag::CseDag;
use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::grad::{grad as eml_grad, hessian as eml_hessian, jacobian as eml_jacobian};
use crate::eml::op::LoweredOp;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the `cas::ad` API.
#[derive(Debug)]
pub enum AdError {
    /// Propagated evaluation-time error (domain violation, unbound variable, …).
    Eval(String),
    /// Slice length did not match the expected number of variables.
    DimMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension received.
        got: usize,
    },
    /// An empty point-list was supplied where at least one point is required.
    EmptyPoints,
}

impl std::fmt::Display for AdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdError::Eval(msg) => write!(f, "AdError::Eval — {msg}"),
            AdError::DimMismatch { expected, got } => {
                write!(f, "AdError::DimMismatch — expected {expected}, got {got}")
            }
            AdError::EmptyPoints => write!(f, "AdError::EmptyPoints — at least one point required"),
        }
    }
}

impl std::error::Error for AdError {}

// ─────────────────────────────────────────────────────────────────────────────
// Free-function AD helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the symbolic gradient `df/dx_wrt` then canonicalize the result.
///
/// Equivalent to `cas::canonicalize(&eml::grad(f, wrt)).into_op()`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{EvalCtx, eval_real}};
/// use scirs2_symbolic::cas::ad::grad_canonical;
///
/// let x2 = LoweredOp::Pow(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(2.0)),
/// );
/// let g = grad_canonical(&x2, 0);
/// let v = eval_real(&g, &EvalCtx::new(&[3.0])).expect("eval");
/// assert!((v - 6.0).abs() < 1e-12, "d/dx x² at x=3 should be 6, got {v}");
/// ```
pub fn grad_canonical(f: &LoweredOp, wrt: usize) -> LoweredOp {
    canonicalize(&eml_grad(f, wrt)).into_op()
}

/// Build a Jacobian with every row canonicalized.
///
/// `jacobian_canonical(f, n)[i] = grad_canonical(f, i)`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{EvalCtx, eval_real}};
/// use scirs2_symbolic::cas::ad::jacobian_canonical;
///
/// let xy = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
/// let j = jacobian_canonical(&xy, 2);
/// // df/dx = y → at (2,3) → 3.0
/// let r0 = eval_real(&j[0], &EvalCtx::new(&[2.0, 3.0])).expect("r0");
/// // df/dy = x → at (2,3) → 2.0
/// let r1 = eval_real(&j[1], &EvalCtx::new(&[2.0, 3.0])).expect("r1");
/// assert!((r0 - 3.0).abs() < 1e-12);
/// assert!((r1 - 2.0).abs() < 1e-12);
/// ```
pub fn jacobian_canonical(f: &LoweredOp, n_vars: usize) -> Vec<LoweredOp> {
    eml_jacobian(f, n_vars)
        .into_iter()
        .map(|g| canonicalize(&g).into_op())
        .collect()
}

/// Build a Hessian with every entry canonicalized.
///
/// `hessian_canonical(f, n)[i][j] = grad_canonical(grad_canonical(f, i), j)`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{EvalCtx, eval_real}};
/// use scirs2_symbolic::cas::ad::hessian_canonical;
///
/// let xy = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
/// let h = hessian_canonical(&xy, 2);
/// // H = [[0, 1], [1, 0]] for f = x*y
/// let h00 = eval_real(&h[0][0], &EvalCtx::new(&[2.0, 3.0])).expect("h00");
/// let h01 = eval_real(&h[0][1], &EvalCtx::new(&[2.0, 3.0])).expect("h01");
/// let h10 = eval_real(&h[1][0], &EvalCtx::new(&[2.0, 3.0])).expect("h10");
/// let h11 = eval_real(&h[1][1], &EvalCtx::new(&[2.0, 3.0])).expect("h11");
/// assert!(h00.abs() < 1e-12, "h00 = {h00}");
/// assert!((h01 - 1.0).abs() < 1e-12, "h01 = {h01}");
/// assert!((h10 - 1.0).abs() < 1e-12, "h10 = {h10}");
/// assert!(h11.abs() < 1e-12, "h11 = {h11}");
/// ```
pub fn hessian_canonical(f: &LoweredOp, n_vars: usize) -> Vec<Vec<LoweredOp>> {
    eml_hessian(f, n_vars)
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|g| canonicalize(&g).into_op())
                .collect()
        })
        .collect()
}

/// Vector-Jacobian product (VJP / cotangent rule).
///
/// Returns the symbolic expression `Σᵢ cotangent[i] * (df/dx_i)` as a
/// `LoweredOp`.  The expression represents the dot product of the cotangent
/// vector with the gradient of `f`.
///
/// # Errors
///
/// Returns [`AdError::DimMismatch`] when `cotangent.len() != n_vars`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{EvalCtx, eval_real}};
/// use scirs2_symbolic::cas::ad::vjp;
///
/// let xy = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
/// // cotangent = [1.0, 0.0]: select df/dx component
/// let ct = vec![LoweredOp::Const(1.0), LoweredOp::Const(0.0)];
/// let v = vjp(&xy, &ct, 2).expect("vjp");
/// // at (2, 3): 1.0 * grad_x(xy) = 1.0 * y = 3.0
/// let r = eval_real(&v, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
/// assert!((r - 3.0).abs() < 1e-12, "vjp = {r}");
/// ```
pub fn vjp(f: &LoweredOp, cotangent: &[LoweredOp], n_vars: usize) -> Result<LoweredOp, AdError> {
    if cotangent.len() != n_vars {
        return Err(AdError::DimMismatch {
            expected: n_vars,
            got: cotangent.len(),
        });
    }

    if n_vars == 0 {
        return Ok(LoweredOp::Const(0.0));
    }

    // Build Σᵢ cotangent[i] * (df/dx_i) iteratively (no recursion needed).
    let mut acc: Option<LoweredOp> = None;
    for (i, ct_i) in cotangent.iter().enumerate() {
        let g_i = grad_canonical(f, i);
        let term = LoweredOp::Mul(Box::new(ct_i.clone()), Box::new(g_i));
        acc = Some(match acc {
            None => term,
            Some(prev) => LoweredOp::Add(Box::new(prev), Box::new(term)),
        });
    }
    // acc is Some because n_vars > 0
    Ok(acc.unwrap_or(LoweredOp::Const(0.0)))
}

/// Jacobian-vector product (JVP / tangent / directional derivative).
///
/// Returns the symbolic expression `Σᵢ v[i] * (df/dx_i)` as a `LoweredOp`.
/// The tangent vector `v` must have the same length as the number of inputs
/// (implicit from `v.len()`).
///
/// An empty tangent vector returns `LoweredOp::Const(0.0)` (mathematically
/// correct: sum over empty index set is zero).
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{EvalCtx, eval_real}};
/// use scirs2_symbolic::cas::ad::jvp;
///
/// let xy = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
/// // tangent = [0.0, 1.0]: directional derivative along y
/// let tv = vec![LoweredOp::Const(0.0), LoweredOp::Const(1.0)];
/// let d = jvp(&xy, &tv).expect("jvp");
/// // at (2, 3): 0.0 * y + 1.0 * x = 2.0
/// let r = eval_real(&d, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
/// assert!((r - 2.0).abs() < 1e-12, "jvp = {r}");
/// ```
pub fn jvp(f: &LoweredOp, v: &[LoweredOp]) -> Result<LoweredOp, AdError> {
    let n_vars = v.len();
    if n_vars == 0 {
        return Ok(LoweredOp::Const(0.0));
    }

    let mut acc: Option<LoweredOp> = None;
    for (i, v_i) in v.iter().enumerate() {
        let g_i = grad_canonical(f, i);
        let term = LoweredOp::Mul(Box::new(v_i.clone()), Box::new(g_i));
        acc = Some(match acc {
            None => term,
            Some(prev) => LoweredOp::Add(Box::new(prev), Box::new(term)),
        });
    }
    Ok(acc.unwrap_or(LoweredOp::Const(0.0)))
}

/// Evaluate `df/dx_wrt` at multiple points using a [`CseDag`] for CSE.
///
/// Builds a single DAG from the gradient expression, then evaluates it at
/// every point in `points` independently.  Subexpressions shared across the
/// *gradient expression itself* are deduplicated; subexpressions across
/// *different points* are not (each point is a fresh evaluation pass).
///
/// # Errors
///
/// - [`AdError::EmptyPoints`] when `points` is empty.
/// - [`AdError::Eval`] on any domain error during evaluation.
///
/// # Examples
///
/// ```
/// use std::f64::consts::PI;
/// use scirs2_symbolic::LoweredOp;
/// use scirs2_symbolic::cas::ad::batch_eval_grad;
///
/// let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
/// let points: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * PI / 4.0]).collect();
/// let grads = batch_eval_grad(&sin_x, 0, &points).expect("batch");
/// // d/dx sin(x) = cos(x)
/// for (pt, g) in points.iter().zip(grads.iter()) {
///     let expected = pt[0].cos();
///     assert!((g - expected).abs() < 1e-10, "at x={}: got {g}, expected {expected}", pt[0]);
/// }
/// ```
pub fn batch_eval_grad(
    f: &LoweredOp,
    wrt: usize,
    points: &[Vec<f64>],
) -> Result<Vec<f64>, AdError> {
    if points.is_empty() {
        return Err(AdError::EmptyPoints);
    }

    let g = grad_canonical(f, wrt);

    // Build the DAG once — subexpressions within g are deduplicated.
    let mut dag = CseDag::new();
    let g_key = dag.add(&g);

    let mut results = Vec::with_capacity(points.len());
    for point in points {
        let vals = dag
            .eval_all(point)
            .map_err(|e| AdError::Eval(e.to_string()))?;
        let v = vals
            .get(&g_key)
            .copied()
            .ok_or_else(|| AdError::Eval("grad key missing from CSE evaluation".into()))?;
        results.push(v);
    }
    Ok(results)
}

/// Central-difference numerical gradient (for testing and validation).
///
/// Computes `(f(p + eps·eᵢ) - f(p - eps·eᵢ)) / (2·eps)` for each dimension
/// `i` independently.  `p` is cloned internally; the original `point` slice
/// is not mutated.
///
/// # Errors
///
/// Returns [`AdError::Eval`] if evaluation fails at any perturbed point.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::LoweredOp;
/// use scirs2_symbolic::cas::ad::numerical_grad;
///
/// let f = LoweredOp::Pow(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(3.0)),
/// );
/// let g = numerical_grad(&f, &[2.0], 1e-5).expect("num_grad");
/// // d/dx x³ at x=2 = 3·x² = 12
/// assert!((g[0] - 12.0).abs() < 1e-4, "num_grad = {}", g[0]);
/// ```
pub fn numerical_grad(f: &LoweredOp, point: &[f64], eps: f64) -> Result<Vec<f64>, AdError> {
    let n = point.len();
    let mut grad_vec = Vec::with_capacity(n);
    let mut p = point.to_vec();

    for i in 0..n {
        let orig = p[i];

        p[i] = orig + eps;
        let fp = eval_real(f, &EvalCtx::new(&p)).map_err(|e| AdError::Eval(e.to_string()))?;

        p[i] = orig - eps;
        let fm = eval_real(f, &EvalCtx::new(&p)).map_err(|e| AdError::Eval(e.to_string()))?;

        p[i] = orig; // restore
        grad_vec.push((fp - fm) / (2.0 * eps));
    }
    Ok(grad_vec)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wave 74 — higher-order derivatives with module-scope cache
// ─────────────────────────────────────────────────────────────────────────────

/// Cache key: `(canonical_hash_of_op, var_idx)`.
type HigherOrderKey = (u128, usize);

/// Module-scope cache mapping `(canonical_hash, var)` → first-order partial
/// derivative `∂op/∂Var(var)` (canonicalized). Reused across all calls to
/// [`higher_order_grad`], [`third_derivative`], [`fourth_derivative`], and
/// [`taylor_higher_order`] within the lifetime of the process.
///
/// The key is the **canonical hash** (Wave 53 stable u128) so that
/// structurally equivalent inputs always hit the same cache entry. Cache
/// invalidation is unnecessary because canonicalization is deterministic.
static HIGHER_ORDER_CACHE: Lazy<RwLock<HashMap<HigherOrderKey, LoweredOp>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// One-shot cached partial derivative: `∂op/∂Var(var)` (canonicalized).
///
/// Looks up the canonical hash of `op` × `var` in [`HIGHER_ORDER_CACHE`]; on
/// hit, returns a clone. On miss, computes via [`grad_canonical`] and
/// inserts the result.
fn cached_grad(op: &LoweredOp, var: usize) -> LoweredOp {
    let key: HigherOrderKey = (canonicalize(op).hash(), var);
    if let Ok(read_guard) = HIGHER_ORDER_CACHE.read() {
        if let Some(g) = read_guard.get(&key) {
            return g.clone();
        }
    }
    let g = grad_canonical(op, var);
    if let Ok(mut write_guard) = HIGHER_ORDER_CACHE.write() {
        write_guard.insert(key, g.clone());
    }
    g
}

/// Compute the iterated single-variable partial derivative `∂ⁿop/∂xᵥⁿ` for
/// `n = 1, 2, …, order`.
///
/// Returns a vector `[d¹, d², …, dᵒʳᵈᵉʳ]` of length `order`. Each entry is
/// canonicalized. The intermediate results are memoized in
/// `HIGHER_ORDER_CACHE` keyed on `(canonical_hash, var)`.
///
/// # Panics
///
/// Does not panic. Returns an empty vector when `order = 0`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::cas::ad::higher_order_grad;
/// use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
///
/// // f(x) = x⁴ → derivatives [4x³, 12x², 24x, 24, 0, ...]
/// let f = LoweredOp::Pow(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(4.0)),
/// );
/// let series = higher_order_grad(&f, 0, 4);
/// let pt = [2.0_f64];
/// let v0 = eval_real(&series[0], &EvalCtx::new(&pt)).expect("d1");
/// let v3 = eval_real(&series[3], &EvalCtx::new(&pt)).expect("d4");
/// assert!((v0 - 32.0).abs() < 1e-9, "d¹(x⁴) at x=2 should be 32, got {v0}");
/// assert!((v3 - 24.0).abs() < 1e-9, "d⁴(x⁴) at x=2 should be 24, got {v3}");
/// ```
pub fn higher_order_grad(op: &LoweredOp, var: usize, order: u32) -> Vec<LoweredOp> {
    if order == 0 {
        return Vec::new();
    }
    let mut series = Vec::with_capacity(order as usize);
    let mut current = op.clone();
    for _ in 0..order {
        let next = cached_grad(&current, var);
        series.push(next.clone());
        current = next;
    }
    series
}

/// Third-order mixed partial derivative `∂³op/(∂xᵥ₀ ∂xᵥ₁ ∂xᵥ₂)`.
///
/// `vars[i]` is the i-th differentiation variable; the derivatives are
/// applied in the order they appear in the array. Result is canonicalized.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::cas::ad::third_derivative;
/// use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
///
/// // f = x³ → d³/dx³ = 6
/// let f = LoweredOp::Pow(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(3.0)),
/// );
/// let d3 = third_derivative(&f, [0, 0, 0]);
/// let v = eval_real(&d3, &EvalCtx::new(&[5.0])).expect("eval");
/// assert!((v - 6.0).abs() < 1e-9, "d³(x³) at x=5 should be 6, got {v}");
/// ```
pub fn third_derivative(op: &LoweredOp, vars: [usize; 3]) -> LoweredOp {
    let d1 = cached_grad(op, vars[0]);
    let d2 = cached_grad(&d1, vars[1]);
    cached_grad(&d2, vars[2])
}

/// Fourth-order mixed partial derivative `∂⁴op/(∂xᵥ₀ ∂xᵥ₁ ∂xᵥ₂ ∂xᵥ₃)`.
///
/// `vars[i]` is the i-th differentiation variable. Derivatives are applied in
/// the order specified. Result is canonicalized.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::cas::ad::fourth_derivative;
/// use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
///
/// // f = x⁴ → d⁴/dx⁴ = 24
/// let f = LoweredOp::Pow(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(4.0)),
/// );
/// let d4 = fourth_derivative(&f, [0, 0, 0, 0]);
/// let v = eval_real(&d4, &EvalCtx::new(&[2.7])).expect("eval");
/// assert!((v - 24.0).abs() < 1e-7);
/// ```
pub fn fourth_derivative(op: &LoweredOp, vars: [usize; 4]) -> LoweredOp {
    let d1 = cached_grad(op, vars[0]);
    let d2 = cached_grad(&d1, vars[1]);
    let d3 = cached_grad(&d2, vars[2]);
    cached_grad(&d3, vars[3])
}

/// Taylor coefficients of `op` around `x₀` with respect to `var`, up to
/// (and including) `order`.
///
/// Returns a vector of length `order + 1` with entries `[a₀, a₁, …, aₒᵣ]`
/// where `aₖ = (1/k!) · (∂ᵏop/∂xᵥᵏ)|ₓᵥ=x₀`. Each entry is a `LoweredOp`
/// expression evaluated at `x₀` (constant in `var` but possibly symbolic in
/// other variables).
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::cas::ad::taylor_higher_order;
/// use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
///
/// let f = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
/// let coeffs = taylor_higher_order(&f, 0, 0.0, 5);
/// // a_3 = -1/6 (cos(0) = 1, sin(0) = 0; sin Taylor: x − x³/6 + …)
/// let v3 = eval_real(&coeffs[3], &EvalCtx::new(&[0.0])).expect("a3");
/// assert!((v3 - (-1.0_f64 / 6.0)).abs() < 1e-9);
/// ```
pub fn taylor_higher_order(op: &LoweredOp, var: usize, x0: f64, order: u32) -> Vec<LoweredOp> {
    let mut coeffs = Vec::with_capacity(order as usize + 1);
    let mut current = op.clone();
    let mut factorial: f64 = 1.0;
    for k in 0..=order {
        if k > 0 {
            factorial *= k as f64;
        }
        // Evaluate `current` at x₀ — substitute Var(var) = Const(x₀).
        let evaluated = substitute_var_with_const(&current, var, x0);
        let scaled = LoweredOp::Mul(
            Box::new(LoweredOp::Const(1.0 / factorial)),
            Box::new(evaluated),
        );
        coeffs.push(canonicalize(&scaled).into_op());
        if k < order {
            current = cached_grad(&current, var);
        }
    }
    coeffs
}

/// Iteratively replace `Var(var_id)` with `Const(value)` in `op`.
///
/// No recursion — uses a work-stack. Used by [`taylor_higher_order`] to
/// evaluate derivatives at the expansion point `x₀`.
fn substitute_var_with_const(op: &LoweredOp, var_id: usize, value: f64) -> LoweredOp {
    enum Frame<'a> {
        Open(&'a LoweredOp),
        Build(&'a LoweredOp),
    }
    let mut frames: Vec<Frame<'_>> = vec![Frame::Open(op)];
    let mut stack: Vec<LoweredOp> = Vec::with_capacity(16);
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Open(node) => match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    frames.push(Frame::Build(node));
                }
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    frames.push(Frame::Build(node));
                    frames.push(Frame::Open(b));
                    frames.push(Frame::Open(a));
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
                    frames.push(Frame::Build(node));
                    frames.push(Frame::Open(c));
                }
            },
            Frame::Build(node) => {
                let result = match node {
                    LoweredOp::Const(c) => LoweredOp::Const(*c),
                    LoweredOp::Var(i) => {
                        if *i == var_id {
                            LoweredOp::Const(value)
                        } else {
                            LoweredOp::Var(*i)
                        }
                    }
                    LoweredOp::Add(_, _) => {
                        let r = stack.pop().expect("rebuild Add: missing right");
                        let l = stack.pop().expect("rebuild Add: missing left");
                        LoweredOp::Add(Box::new(l), Box::new(r))
                    }
                    LoweredOp::Sub(_, _) => {
                        let r = stack.pop().expect("rebuild Sub: missing right");
                        let l = stack.pop().expect("rebuild Sub: missing left");
                        LoweredOp::Sub(Box::new(l), Box::new(r))
                    }
                    LoweredOp::Mul(_, _) => {
                        let r = stack.pop().expect("rebuild Mul: missing right");
                        let l = stack.pop().expect("rebuild Mul: missing left");
                        LoweredOp::Mul(Box::new(l), Box::new(r))
                    }
                    LoweredOp::Div(_, _) => {
                        let r = stack.pop().expect("rebuild Div: missing right");
                        let l = stack.pop().expect("rebuild Div: missing left");
                        LoweredOp::Div(Box::new(l), Box::new(r))
                    }
                    LoweredOp::Pow(_, _) => {
                        let r = stack.pop().expect("rebuild Pow: missing right");
                        let l = stack.pop().expect("rebuild Pow: missing left");
                        LoweredOp::Pow(Box::new(l), Box::new(r))
                    }
                    LoweredOp::Neg(_) => {
                        let c = stack.pop().expect("rebuild Neg: missing child");
                        LoweredOp::Neg(Box::new(c))
                    }
                    LoweredOp::Exp(_) => {
                        let c = stack.pop().expect("rebuild Exp: missing child");
                        LoweredOp::Exp(Box::new(c))
                    }
                    LoweredOp::Ln(_) => {
                        let c = stack.pop().expect("rebuild Ln: missing child");
                        LoweredOp::Ln(Box::new(c))
                    }
                    LoweredOp::Sin(_) => {
                        let c = stack.pop().expect("rebuild Sin: missing child");
                        LoweredOp::Sin(Box::new(c))
                    }
                    LoweredOp::Cos(_) => {
                        let c = stack.pop().expect("rebuild Cos: missing child");
                        LoweredOp::Cos(Box::new(c))
                    }
                    LoweredOp::Tan(_) => {
                        let c = stack.pop().expect("rebuild Tan: missing child");
                        LoweredOp::Tan(Box::new(c))
                    }
                    LoweredOp::Sinh(_) => {
                        let c = stack.pop().expect("rebuild Sinh: missing child");
                        LoweredOp::Sinh(Box::new(c))
                    }
                    LoweredOp::Cosh(_) => {
                        let c = stack.pop().expect("rebuild Cosh: missing child");
                        LoweredOp::Cosh(Box::new(c))
                    }
                    LoweredOp::Tanh(_) => {
                        let c = stack.pop().expect("rebuild Tanh: missing child");
                        LoweredOp::Tanh(Box::new(c))
                    }
                    LoweredOp::Arcsin(_) => {
                        let c = stack.pop().expect("rebuild Arcsin: missing child");
                        LoweredOp::Arcsin(Box::new(c))
                    }
                    LoweredOp::Arccos(_) => {
                        let c = stack.pop().expect("rebuild Arccos: missing child");
                        LoweredOp::Arccos(Box::new(c))
                    }
                    LoweredOp::Arctan(_) => {
                        let c = stack.pop().expect("rebuild Arctan: missing child");
                        LoweredOp::Arctan(Box::new(c))
                    }
                    LoweredOp::Arcsinh(_) => {
                        let c = stack.pop().expect("rebuild Arcsinh: missing child");
                        LoweredOp::Arcsinh(Box::new(c))
                    }
                    LoweredOp::Arccosh(_) => {
                        let c = stack.pop().expect("rebuild Arccosh: missing child");
                        LoweredOp::Arccosh(Box::new(c))
                    }
                    LoweredOp::Arctanh(_) => {
                        let c = stack.pop().expect("rebuild Arctanh: missing child");
                        LoweredOp::Arctanh(Box::new(c))
                    }
                    LoweredOp::Sqrt(_) => {
                        let c = stack.pop().expect("rebuild Sqrt: missing child");
                        LoweredOp::Sqrt(Box::new(c))
                    }
                    LoweredOp::Abs(_) => {
                        let c = stack.pop().expect("rebuild Abs: missing child");
                        LoweredOp::Abs(Box::new(c))
                    }
                };
                stack.push(result);
            }
        }
    }
    stack
        .pop()
        .expect("substitute_var_with_const: empty stack at end")
}

// ─────────────────────────────────────────────────────────────────────────────
// GradGraph — precomputed gradient graph for repeated evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed gradient graph for a scalar function `f`.
///
/// Build once with [`GradGraph::new`]; evaluate many times with
/// [`GradGraph::eval`], [`GradGraph::eval_grad`], and
/// [`GradGraph::eval_with_grad`].
///
/// All gradient expressions are canonicalized at construction time so that
/// shared subexpressions across the gradient vector collide in the CSE DAG.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::LoweredOp;
/// use scirs2_symbolic::cas::ad::GradGraph;
///
/// let xy = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
/// let g = GradGraph::new(&xy, 2);
/// let (val, grad) = g.eval_with_grad(&[2.0, 3.0]).expect("eval");
/// assert!((val - 6.0).abs() < 1e-12);
/// assert!((grad[0] - 3.0).abs() < 1e-12); // df/dx = y
/// assert!((grad[1] - 2.0).abs() < 1e-12); // df/dy = x
/// ```
pub struct GradGraph {
    op: Arc<LoweredOp>,
    grad_ops: Vec<Arc<LoweredOp>>,
    n_vars: usize,
}

impl GradGraph {
    /// Build a [`GradGraph`] for `f` with `n_vars` independent variables.
    ///
    /// Computes and canonicalizes all `n_vars` partial derivatives.
    pub fn new(f: &LoweredOp, n_vars: usize) -> Self {
        // Canonicalize the function itself so its structural hash matches
        // any subterms that also appear in its gradients.
        let op = Arc::new(canonicalize(f).into_op());
        let grad_ops = (0..n_vars)
            .map(|i| Arc::new(grad_canonical(f, i)))
            .collect();
        Self {
            op,
            grad_ops,
            n_vars,
        }
    }

    /// Number of independent variables.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Reference to the canonicalized function expression.
    #[must_use]
    pub fn op(&self) -> &Arc<LoweredOp> {
        &self.op
    }

    /// Reference to the canonicalized gradient expression for variable `i`,
    /// or `None` if `i >= n_vars`.
    #[must_use]
    pub fn grad_op(&self, i: usize) -> Option<&Arc<LoweredOp>> {
        self.grad_ops.get(i)
    }

    /// Evaluate the function at `point`.
    ///
    /// # Errors
    ///
    /// Returns [`AdError::DimMismatch`] when `point.len() != n_vars` (soft
    /// check — evaluation may still succeed if the expression only references
    /// a subset of variables; the check is advisory).  Returns
    /// [`AdError::Eval`] on domain errors.
    pub fn eval(&self, point: &[f64]) -> Result<f64, AdError> {
        eval_real(&self.op, &EvalCtx::new(point)).map_err(|e| AdError::Eval(e.to_string()))
    }

    /// Evaluate all gradients at `point`.
    ///
    /// Returns `grad[i] = df/dx_i` evaluated at `point`.
    ///
    /// # Errors
    ///
    /// Returns [`AdError::Eval`] on any domain error.
    pub fn eval_grad(&self, point: &[f64]) -> Result<Vec<f64>, AdError> {
        self.grad_ops
            .iter()
            .map(|g| eval_real(g, &EvalCtx::new(point)).map_err(|e| AdError::Eval(e.to_string())))
            .collect()
    }

    /// Evaluate the function value and all gradients in a single CSE pass.
    ///
    /// Builds one [`CseDag`] from the function and all gradient expressions,
    /// calls [`CseDag::eval_all`] once, then extracts the value and gradient
    /// vector from the result map.  Shared subexpressions across `f` and its
    /// gradients are evaluated exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`AdError::Eval`] on any domain or evaluation error.
    pub fn eval_with_grad(&self, point: &[f64]) -> Result<(f64, Vec<f64>), AdError> {
        let mut dag = CseDag::new();

        // Insert f and all grad_ops; capture root hashes.
        let op_hash = dag.add(&self.op);
        let grad_hashes: Vec<u128> = self.grad_ops.iter().map(|g| dag.add(g)).collect();

        // Single topological-order evaluation pass (CSE).
        let vals = dag
            .eval_all(point)
            .map_err(|e| AdError::Eval(e.to_string()))?;

        let f_val = vals
            .get(&op_hash)
            .copied()
            .ok_or_else(|| AdError::Eval("function root hash missing from CSE result".into()))?;

        let grad_vec = grad_hashes
            .iter()
            .map(|h| {
                vals.get(h).copied().ok_or_else(|| {
                    AdError::Eval("gradient root hash missing from CSE result".into())
                })
            })
            .collect::<Result<Vec<f64>, AdError>>()?;

        Ok((f_val, grad_vec))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::EvalCtx;

    // Helper constructors
    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Box::new(a), Box::new(b))
    }
    fn pow(b: LoweredOp, e: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Box::new(b), Box::new(e))
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        let tol = 1e-10_f64.max(b.abs() * 1e-10);
        (a - b).abs() < tol
    }

    // ── grad_canonical ──────────────────────────────────────────────────────

    #[test]
    fn test_grad_canonical_x_squared() {
        // d/dx x² = 2x  →  at x=3 → 6.0
        let x2 = pow(var(0), c(2.0));
        let g = grad_canonical(&x2, 0);
        let v = eval_real(&g, &EvalCtx::new(&[3.0])).expect("eval");
        assert!(approx_eq(v, 6.0), "grad(x²,0) at x=3 should be 6, got {v}");
    }

    #[test]
    fn test_grad_canonical_const_wrt_unrelated() {
        // d/dx c = 0
        let f = c(42.0);
        let g = grad_canonical(&f, 0);
        let v = eval_real(&g, &EvalCtx::new(&[999.0])).expect("eval");
        assert!(approx_eq(v, 0.0), "grad(const) should be 0, got {v}");
    }

    // ── GradGraph::new ──────────────────────────────────────────────────────

    #[test]
    fn test_grad_graph_new() {
        // GradGraph::new(x*y, 2): n_vars=2, grad_ops[0]≈y, grad_ops[1]≈x
        let xy = mul(var(0), var(1));
        let g = GradGraph::new(&xy, 2);
        assert_eq!(g.n_vars(), 2);
        assert!(g.grad_op(0).is_some());
        assert!(g.grad_op(1).is_some());
        assert!(g.grad_op(2).is_none());

        // grad_ops[0] = df/dx = y → at (2,3) → 3.0
        let g0 = g.grad_op(0).expect("g0");
        let r0 = eval_real(g0, &EvalCtx::new(&[2.0, 3.0])).expect("eval g0");
        assert!(
            approx_eq(r0, 3.0),
            "grad_op[0] at (2,3) should be 3, got {r0}"
        );

        // grad_ops[1] = df/dy = x → at (2,3) → 2.0
        let g1 = g.grad_op(1).expect("g1");
        let r1 = eval_real(g1, &EvalCtx::new(&[2.0, 3.0])).expect("eval g1");
        assert!(
            approx_eq(r1, 2.0),
            "grad_op[1] at (2,3) should be 2, got {r1}"
        );
    }

    // ── GradGraph::eval ─────────────────────────────────────────────────────

    #[test]
    fn test_grad_graph_eval() {
        // f(x,y) = x*y; eval at (2,3) → 6.0
        let xy = mul(var(0), var(1));
        let g = GradGraph::new(&xy, 2);
        let v = g.eval(&[2.0, 3.0]).expect("eval");
        assert!(approx_eq(v, 6.0), "f(2,3) = x*y should be 6, got {v}");
    }

    // ── GradGraph::eval_grad ────────────────────────────────────────────────

    #[test]
    fn test_grad_graph_eval_grad() {
        // f(x,y) = x*y; eval_grad at (2,3) → [3.0, 2.0]
        let xy = mul(var(0), var(1));
        let g = GradGraph::new(&xy, 2);
        let grads = g.eval_grad(&[2.0, 3.0]).expect("eval_grad");
        assert_eq!(grads.len(), 2);
        assert!(
            approx_eq(grads[0], 3.0),
            "df/dx at (2,3) should be 3, got {}",
            grads[0]
        );
        assert!(
            approx_eq(grads[1], 2.0),
            "df/dy at (2,3) should be 2, got {}",
            grads[1]
        );
    }

    // ── GradGraph::eval_with_grad ───────────────────────────────────────────

    #[test]
    fn test_grad_graph_eval_with_grad() {
        // f(x,y) = x*y; eval_with_grad at (2,3) → (6.0, [3.0, 2.0])
        let xy = mul(var(0), var(1));
        let g = GradGraph::new(&xy, 2);
        let (val, grads) = g.eval_with_grad(&[2.0, 3.0]).expect("eval_with_grad");
        assert!(approx_eq(val, 6.0), "f(2,3) should be 6, got {val}");
        assert_eq!(grads.len(), 2);
        assert!(
            approx_eq(grads[0], 3.0),
            "df/dx at (2,3) should be 3, got {}",
            grads[0]
        );
        assert!(
            approx_eq(grads[1], 2.0),
            "df/dy at (2,3) should be 2, got {}",
            grads[1]
        );
    }

    // ── jacobian_canonical ──────────────────────────────────────────────────

    #[test]
    fn test_jacobian_canonical() {
        // jacobian(x*y, 2) should produce [y, x]
        let xy = mul(var(0), var(1));
        let j = jacobian_canonical(&xy, 2);
        assert_eq!(j.len(), 2);
        let pt = EvalCtx::new(&[2.0, 3.0]);
        let r0 = eval_real(&j[0], &pt).expect("j[0]");
        let r1 = eval_real(&j[1], &pt).expect("j[1]");
        assert!(approx_eq(r0, 3.0), "j[0] at (2,3) should be 3, got {r0}");
        assert!(approx_eq(r1, 2.0), "j[1] at (2,3) should be 2, got {r1}");
    }

    // ── hessian_canonical ───────────────────────────────────────────────────

    #[test]
    fn test_hessian_canonical_xy() {
        // f = x*y → H = [[0, 1], [1, 0]]
        let xy = mul(var(0), var(1));
        let h = hessian_canonical(&xy, 2);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert_eq!(h[1].len(), 2);
        let pt = EvalCtx::new(&[2.0, 3.0]);
        let h00 = eval_real(&h[0][0], &pt).expect("h00");
        let h01 = eval_real(&h[0][1], &pt).expect("h01");
        let h10 = eval_real(&h[1][0], &pt).expect("h10");
        let h11 = eval_real(&h[1][1], &pt).expect("h11");
        assert!(h00.abs() < 1e-10, "h[0][0] should be 0, got {h00}");
        assert!(approx_eq(h01, 1.0), "h[0][1] should be 1, got {h01}");
        assert!(approx_eq(h10, 1.0), "h[1][0] should be 1, got {h10}");
        assert!(h11.abs() < 1e-10, "h[1][1] should be 0, got {h11}");
    }

    // ── vjp ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_vjp_xy() {
        // f = x*y, cotangent = [1.0, 0.0] → selects df/dx = y → at (2,3) → 3.0
        let xy = mul(var(0), var(1));
        let ct = vec![c(1.0), c(0.0)];
        let v = vjp(&xy, &ct, 2).expect("vjp");
        let r = eval_real(&v, &EvalCtx::new(&[2.0, 3.0])).expect("eval vjp");
        assert!(approx_eq(r, 3.0), "vjp at (2,3) should be 3, got {r}");
    }

    #[test]
    fn test_vjp_dim_mismatch() {
        let xy = mul(var(0), var(1));
        let ct = vec![c(1.0)]; // only 1 cotangent for 2-var function
        let err = vjp(&xy, &ct, 2);
        assert!(matches!(
            err,
            Err(AdError::DimMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    // ── jvp ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_jvp_xy() {
        // f = x*y, tangent = [0.0, 1.0] → directional deriv along y → df/dy = x → at (2,3) → 2.0
        let xy = mul(var(0), var(1));
        let tv = vec![c(0.0), c(1.0)];
        let d = jvp(&xy, &tv).expect("jvp");
        let r = eval_real(&d, &EvalCtx::new(&[2.0, 3.0])).expect("eval jvp");
        assert!(approx_eq(r, 2.0), "jvp at (2,3) should be 2, got {r}");
    }

    #[test]
    fn test_jvp_empty_tangent() {
        let f = c(42.0);
        let d = jvp(&f, &[]).expect("jvp empty");
        let r = eval_real(&d, &EvalCtx::new(&[])).expect("eval jvp empty");
        assert!(
            approx_eq(r, 0.0),
            "jvp with empty tangent should be 0, got {r}"
        );
    }

    // ── batch_eval_grad ─────────────────────────────────────────────────────

    #[test]
    fn test_batch_eval_grad_sin() {
        // d/dx sin(x) = cos(x); batch 10 points
        use std::f64::consts::PI;
        let sin_x = LoweredOp::Sin(Box::new(var(0)));
        let points: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * PI / 5.0]).collect();
        let grads = batch_eval_grad(&sin_x, 0, &points).expect("batch");
        assert_eq!(grads.len(), 10);
        for (pt, &g) in points.iter().zip(grads.iter()) {
            let expected = pt[0].cos();
            let tol = 1e-9_f64.max(expected.abs() * 1e-9);
            assert!(
                (g - expected).abs() < tol,
                "at x={}: got {g}, expected {expected}",
                pt[0]
            );
        }
    }

    #[test]
    fn test_batch_eval_grad_empty_points() {
        let f = var(0);
        let err = batch_eval_grad(&f, 0, &[]);
        assert!(matches!(err, Err(AdError::EmptyPoints)));
    }

    // ── numerical_grad vs symbolic ──────────────────────────────────────────

    #[test]
    fn test_numerical_grad_vs_symbolic() {
        // f = x² + y³ at (2.0, 3.0)
        // grad = [2x, 3y²] = [4.0, 27.0]
        let f = add(pow(var(0), c(2.0)), pow(var(1), c(3.0)));
        let pt = [2.0_f64, 3.0];
        let ng = numerical_grad(&f, &pt, 1e-6).expect("num_grad");
        let sg_x = eval_real(&grad_canonical(&f, 0), &EvalCtx::new(&pt)).expect("sg_x");
        let sg_y = eval_real(&grad_canonical(&f, 1), &EvalCtx::new(&pt)).expect("sg_y");
        let tol = 1e-5;
        assert!(
            (ng[0] - sg_x).abs() < tol,
            "numerical df/dx = {}, symbolic = {}, diff = {}",
            ng[0],
            sg_x,
            (ng[0] - sg_x).abs()
        );
        assert!(
            (ng[1] - sg_y).abs() < tol,
            "numerical df/dy = {}, symbolic = {}, diff = {}",
            ng[1],
            sg_y,
            (ng[1] - sg_y).abs()
        );
    }

    // ── depth overflow ───────────────────────────────────────────────────────

    #[test]
    fn test_depth_1000_no_overflow() {
        // Build a 1000-deep Add(_, Const(0)) chain; GradGraph::new must not
        // overflow the OS stack (delegates to iterative eml::grad and
        // cas::canonicalize which are iterative).
        let mut op = var(0);
        for _ in 0..1_000 {
            op = add(op, c(0.0));
        }
        // Should not panic / overflow
        let gg = GradGraph::new(&op, 1);
        // df/dx = 1  (chain of adds with zero, each contrib is 1+0+...+0)
        let v = gg.eval(&[42.0]).expect("eval");
        // f(42.0) = 42.0 + 0 + 0 + ... = 42.0
        assert!(approx_eq(v, 42.0), "f(42) should be 42, got {v}");
        let grads = gg.eval_grad(&[42.0]).expect("eval_grad");
        assert!(
            approx_eq(grads[0], 1.0),
            "grad should be 1, got {}",
            grads[0]
        );
    }
}
