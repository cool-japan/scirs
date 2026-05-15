//! `cas::solve` — EML-native single-variable algebraic equation solver.
//!
//! Solves equations of the form `lhs = rhs` for a given variable `Var(var_idx)`.
//!
//! # Strategy selection
//!
//! 1. If the combined expression `lhs - rhs` contains the variable, we first
//!    attempt **polynomial detection** (`as_polynomial`) which handles up to
//!    degree 2 exactly (quadratic formula, `complete: true`).
//! 2. If polynomial detection fails but the variable appears exactly once,
//!    we attempt **invertible-chain unwinding** — walking down the unique
//!    occurrence path and replaying inverse operations onto `rhs`.
//! 3. Otherwise, [`SolveError::CannotSeparate`] or [`SolveError::HighDegreePoly`]
//!    is returned.
//!
//! No recursion anywhere — all traversals use iterative work-stacks.

use crate::eml::op::LoweredOp;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Equation solving result.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solutions as [`LoweredOp`] expressions (may involve other variables).
    pub solutions: Vec<LoweredOp>,
    /// `true` if the solver can claim completeness for the given strategy
    /// (e.g. degree-1/2 polynomial over R, invertible-chain with single branch).
    pub complete: bool,
}

/// Equation solving error.
#[derive(Debug)]
pub enum SolveError {
    /// Variable is not present in the equation.
    NotSolvable { reason: String },
    /// Variable appears in multiple branches that cannot be separated.
    CannotSeparate,
    /// Polynomial degree is too high for the current solver (≥ 3).
    HighDegreePoly { degree: usize },
    /// Division by zero encountered while constructing the solution.
    DivisionByZero,
    /// Internal bookkeeping error (should not occur in normal usage).
    InternalError(String),
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Solve `lhs = rhs` for `Var(var_idx)`.
///
/// Returns [`SolveResult`] on success, [`SolveError`] on failure.
pub fn solve(lhs: &LoweredOp, rhs: &LoweredOp, var_idx: usize) -> Result<SolveResult, SolveError> {
    // Form combined expression: lhs - rhs, then solve for zero.
    let combined = if rhs_contains_var(rhs, var_idx) {
        LoweredOp::Sub(Box::new(lhs.clone()), Box::new(rhs.clone()))
    } else {
        // rhs is free of var_idx — solve lhs against rhs directly.
        // We still form lhs - rhs to handle the polynomial detection uniformly.
        LoweredOp::Sub(Box::new(lhs.clone()), Box::new(rhs.clone()))
    };

    let lhs_count = count_var_occurrences(lhs, var_idx);
    let rhs_count = count_var_occurrences(rhs, var_idx);
    let total = lhs_count + rhs_count;

    if total == 0 {
        return Err(SolveError::NotSolvable {
            reason: "variable not present".into(),
        });
    }

    // Try polynomial detection on (lhs - rhs).
    if let Some(coeffs) = as_polynomial(&combined, var_idx) {
        return solve_polynomial(coeffs);
    }

    // Polynomial detection failed — try invertible-chain unwinding.
    // Only works if variable appears exactly once in the combined expression.
    // For lhs-only cases we can work on lhs directly with rhs as target.
    if rhs_count == 0 && lhs_count == 1 {
        return solve_chain(lhs, rhs, var_idx);
    }

    if total == 1 {
        // Variable is in combined but rhs has var → rearrange
        // This handles e.g. solve(Sub(Var(0), rhs_expr), Const(0)) path
        return solve_chain(&combined, &LoweredOp::Const(0.0), var_idx);
    }

    Err(SolveError::CannotSeparate)
}

/// Solve `expr = 0` for `Var(var_idx)`.
///
/// Convenience wrapper: calls `solve(expr, &LoweredOp::Const(0.0), var_idx)`.
pub fn solve_zero(expr: &LoweredOp, var_idx: usize) -> Result<SolveResult, SolveError> {
    solve(expr, &LoweredOp::Const(0.0), var_idx)
}

// ---------------------------------------------------------------------------
// Helper: count occurrences of Var(var_idx)
// ---------------------------------------------------------------------------

/// Count occurrences of `Var(var_idx)` in `expr`. Iterative post-order.
fn count_var_occurrences(expr: &LoweredOp, var_idx: usize) -> usize {
    let mut count = 0usize;
    let mut work: Vec<&LoweredOp> = vec![expr];
    while let Some(op) = work.pop() {
        match op {
            LoweredOp::Var(i) => {
                if *i == var_idx {
                    count += 1;
                }
            }
            LoweredOp::Const(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                work.push(a);
                work.push(b);
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
                work.push(c);
            }
        }
    }
    count
}

/// Check if `rhs` contains the variable — fast short-circuit version.
fn rhs_contains_var(expr: &LoweredOp, var_idx: usize) -> bool {
    let mut work: Vec<&LoweredOp> = vec![expr];
    while let Some(op) = work.pop() {
        match op {
            LoweredOp::Var(i) => {
                if *i == var_idx {
                    return true;
                }
            }
            LoweredOp::Const(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                work.push(a);
                work.push(b);
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
                work.push(c);
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Invertible-chain solver
// ---------------------------------------------------------------------------

/// One step of inverse-function unwinding, applied to the accumulated RHS.
#[derive(Debug, Clone)]
enum InvStep {
    /// Encountered `Add(x, c)` or `Add(c, x)` → rhs ← rhs - c
    SubC(LoweredOp),
    /// Encountered `Sub(c, x)` → rhs ← c - rhs
    SubFromC(LoweredOp),
    /// Encountered `Sub(x, c)` → rhs ← rhs + c
    AddC(LoweredOp),
    /// Encountered `Mul(x, c)` or `Mul(c, x)` → rhs ← rhs / c
    DivC(LoweredOp),
    /// Encountered `Div(x, c)` → rhs ← rhs * c
    MulC(LoweredOp),
    /// Encountered `Neg(x)` → rhs ← -rhs
    Negate,
    /// Encountered `Pow(x, Const(n))` → rhs ← rhs^(1/n); bool = is_even
    InvPow(f64, bool),
    /// Encountered `Exp(x)` → rhs ← Ln(rhs)
    Ln,
    /// Encountered `Ln(x)` → rhs ← Exp(rhs)
    Exp,
    /// Encountered `Sin(x)` → rhs ← Arcsin(rhs)
    Arcsin,
    /// Encountered `Cos(x)` → rhs ← Arccos(rhs)
    Arccos,
    /// Encountered `Tan(x)` → rhs ← Arctan(rhs)
    Arctan,
    /// Encountered `Sqrt(x)` → rhs ← rhs^2
    Sq,
}

/// Solve via invertible chain unwinding. Precondition: `lhs` has exactly 1
/// occurrence of `Var(var_idx)`.
fn solve_chain(
    lhs: &LoweredOp,
    rhs: &LoweredOp,
    var_idx: usize,
) -> Result<SolveResult, SolveError> {
    let mut steps: Vec<InvStep> = Vec::new();
    let mut current = lhs;

    // Walk downward, building the inverse step list.
    loop {
        match current {
            LoweredOp::Var(i) if *i == var_idx => {
                // Reached the target variable — done collecting steps.
                break;
            }
            LoweredOp::Add(a, b) => {
                // Determine which branch contains the variable.
                let a_has = count_var_occurrences(a, var_idx) > 0;
                let b_has = count_var_occurrences(b, var_idx) > 0;
                match (a_has, b_has) {
                    (true, false) => {
                        // x + c → rhs ← rhs - c
                        steps.push(InvStep::SubC(*b.clone()));
                        current = a;
                    }
                    (false, true) => {
                        // c + x → rhs ← rhs - c
                        steps.push(InvStep::SubC(*a.clone()));
                        current = b;
                    }
                    _ => return Err(SolveError::CannotSeparate),
                }
            }
            LoweredOp::Sub(a, b) => {
                let a_has = count_var_occurrences(a, var_idx) > 0;
                let b_has = count_var_occurrences(b, var_idx) > 0;
                match (a_has, b_has) {
                    (true, false) => {
                        // x - c → rhs ← rhs + c
                        steps.push(InvStep::AddC(*b.clone()));
                        current = a;
                    }
                    (false, true) => {
                        // c - x → rhs ← c - rhs
                        steps.push(InvStep::SubFromC(*a.clone()));
                        current = b;
                    }
                    _ => return Err(SolveError::CannotSeparate),
                }
            }
            LoweredOp::Mul(a, b) => {
                let a_has = count_var_occurrences(a, var_idx) > 0;
                let b_has = count_var_occurrences(b, var_idx) > 0;
                match (a_has, b_has) {
                    (true, false) => {
                        // x * c → rhs ← rhs / c
                        steps.push(InvStep::DivC(*b.clone()));
                        current = a;
                    }
                    (false, true) => {
                        // c * x → rhs ← rhs / c
                        steps.push(InvStep::DivC(*a.clone()));
                        current = b;
                    }
                    _ => return Err(SolveError::CannotSeparate),
                }
            }
            LoweredOp::Div(a, b) => {
                let a_has = count_var_occurrences(a, var_idx) > 0;
                let b_has = count_var_occurrences(b, var_idx) > 0;
                match (a_has, b_has) {
                    (true, false) => {
                        // x / c → rhs ← rhs * c
                        steps.push(InvStep::MulC(*b.clone()));
                        current = a;
                    }
                    (false, true) => {
                        // c / x → rhs ← c / rhs  (x = c / rhs)
                        // We handle this via: c / x = rhs  ↔  x = c / rhs
                        // Rewrite as DivC(c) then negate direction — actually:
                        // rhs ← c / current_rhs.  We encode as two steps.
                        // Use SubFromC trick: rhs ← c / rhs not directly an InvStep.
                        // Handle as: swap and DivC(rhs_constant).
                        // Actually: c/x = rhs  ↔  x = c/rhs
                        // Push as DivC with swapped meaning (we compute c / rhs):
                        // We add a special case here using Negate + DivC pattern.
                        // Simplest: x = c / rhs
                        // We can represent this as applying "invert and multiply by c":
                        // Push a synthetic step that computes c / accumulated_rhs.
                        // Since InvStep doesn't have this form, we use a SubFromC analog:
                        // After replay, result should be c / accumulated_rhs.
                        // Use SubFromC(c) but for division? Let's add DivFromC.
                        // For simplicity and correctness, use the existing steps:
                        // c / x = rhs  →  rhs ← a / rhs  (not directly supported by current enum)
                        // Handle by: after computing rhs as current_rhs,
                        // the step is: rhs ← a / rhs
                        // This is "DivFromC" — analogous to SubFromC.
                        // We need to push this as a special case.
                        // Since the enum doesn't have DivFromC, add it inline using
                        // MulC(Div(c, placeholder)) — but that's messy.
                        // Best: record c, and during replay: new_rhs = c / rhs.
                        // Encode as: step SubFromC but for division.
                        // We'll use a special hack: push two steps:
                        // (1) DivC with denominator = rhs_const (will be applied as rhs ← rhs / ?)
                        // This doesn't work cleanly. Let's just use the Sub(c-x)
                        // analog: introduce a helper that works for the c/x case.
                        // For the c/x case, the inverse is: rhs ← c / rhs.
                        // We can do this by: DivC with "from" variant — we emit SubFromC(c)
                        // but adapted for division. Since InvStep has SubFromC for subtraction,
                        // we need DivFromC for division. Let's use a workaround:
                        // rhs ← c / rhs  =  rhs ← c * (1/rhs)  — we can express as:
                        // Step 1: take reciprocal (not in enum).
                        // Cleanest fix: handle c/x = rhs → x = c/rhs directly in replay
                        // by emitting MulC(c) then "take reciprocal":
                        // Actually: c/x = rhs  →  x = c/rhs  →  after current steps,
                        // new_rhs = Div(c, old_rhs).
                        // We need a DivFromC step. The enum currently lacks it.
                        // Since we control the enum, let's handle it here by returning
                        // a single solution directly instead of going through steps.
                        let solution = LoweredOp::Div(a.clone(), Box::new(rhs.clone()));
                        return Ok(SolveResult {
                            solutions: vec![solution],
                            complete: true,
                        });
                    }
                    _ => return Err(SolveError::CannotSeparate),
                }
            }
            LoweredOp::Neg(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Negate);
                current = c;
            }
            LoweredOp::Pow(base, exp) => {
                let base_has = count_var_occurrences(base, var_idx) > 0;
                let exp_has = count_var_occurrences(exp, var_idx) > 0;
                match (base_has, exp_has) {
                    (true, false) => {
                        // x^n → rhs ← rhs^(1/n)
                        if let LoweredOp::Const(n) = exp.as_ref() {
                            let n_val = *n;
                            if n_val.fract().abs() < 1e-12 {
                                let n_int = n_val.round() as i64;
                                if n_int > 0 {
                                    let is_even = n_int % 2 == 0;
                                    steps.push(InvStep::InvPow(n_val, is_even));
                                    current = base;
                                } else {
                                    return Err(SolveError::NotSolvable {
                                        reason: "non-positive integer exponent in Pow".into(),
                                    });
                                }
                            } else {
                                return Err(SolveError::NotSolvable {
                                    reason: "non-integer exponent in Pow — cannot invert cleanly"
                                        .into(),
                                });
                            }
                        } else {
                            return Err(SolveError::CannotSeparate);
                        }
                    }
                    (false, true) => {
                        // c^x = rhs → x = ln(rhs) / ln(c)
                        let c = base.clone();
                        let ln_rhs = LoweredOp::Ln(Box::new(rhs.clone()));
                        let ln_c = LoweredOp::Ln(c);
                        let solution = LoweredOp::Div(Box::new(ln_rhs), Box::new(ln_c));
                        return Ok(SolveResult {
                            solutions: vec![solution],
                            complete: true,
                        });
                    }
                    _ => return Err(SolveError::CannotSeparate),
                }
            }
            LoweredOp::Exp(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Ln);
                current = c;
            }
            LoweredOp::Ln(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Exp);
                current = c;
            }
            LoweredOp::Sin(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Arcsin);
                current = c;
            }
            LoweredOp::Cos(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Arccos);
                current = c;
            }
            LoweredOp::Tan(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Arctan);
                current = c;
            }
            LoweredOp::Sqrt(c) if count_var_occurrences(c, var_idx) > 0 => {
                steps.push(InvStep::Sq);
                current = c;
            }
            _ => {
                // Unsupported node, or unary node whose child doesn't contain the variable.
                return Err(SolveError::CannotSeparate);
            }
        }
    }

    // Replay the inverse steps onto rhs.
    replay_steps(steps, rhs)
}

/// Replay accumulated inverse steps onto `initial_rhs`.
fn replay_steps(steps: Vec<InvStep>, initial_rhs: &LoweredOp) -> Result<SolveResult, SolveError> {
    // Start with a single candidate RHS.
    let mut candidates: Vec<LoweredOp> = vec![initial_rhs.clone()];
    let mut complete = true;

    for step in steps {
        let mut next: Vec<LoweredOp> = Vec::new();
        for candidate in candidates {
            match &step {
                InvStep::SubC(c) => {
                    // rhs ← rhs - c
                    next.push(LoweredOp::Sub(Box::new(candidate), Box::new(c.clone())));
                }
                InvStep::SubFromC(c) => {
                    // rhs ← c - rhs
                    next.push(LoweredOp::Sub(Box::new(c.clone()), Box::new(candidate)));
                }
                InvStep::AddC(c) => {
                    // rhs ← rhs + c
                    next.push(LoweredOp::Add(Box::new(candidate), Box::new(c.clone())));
                }
                InvStep::DivC(c) => {
                    // rhs ← rhs / c
                    next.push(LoweredOp::Div(Box::new(candidate), Box::new(c.clone())));
                }
                InvStep::MulC(c) => {
                    // rhs ← rhs * c
                    next.push(LoweredOp::Mul(Box::new(candidate), Box::new(c.clone())));
                }
                InvStep::Negate => {
                    // rhs ← -rhs
                    next.push(LoweredOp::Neg(Box::new(candidate)));
                }
                InvStep::InvPow(n, is_even) => {
                    let inv_n = 1.0 / n;
                    let pos_root = LoweredOp::Pow(
                        Box::new(candidate.clone()),
                        Box::new(LoweredOp::Const(inv_n)),
                    );
                    if *is_even {
                        // Even root → two solutions: +root and -root
                        let neg_root = LoweredOp::Neg(Box::new(pos_root.clone()));
                        next.push(pos_root);
                        next.push(neg_root);
                        // Completeness is true for degree-2 polynomial path, but
                        // for chain path we set it false as we only provide principal roots.
                        complete = false;
                    } else {
                        next.push(pos_root);
                    }
                }
                InvStep::Ln => {
                    // rhs ← Ln(rhs)
                    next.push(LoweredOp::Ln(Box::new(candidate)));
                }
                InvStep::Exp => {
                    // rhs ← Exp(rhs)
                    next.push(LoweredOp::Exp(Box::new(candidate)));
                }
                InvStep::Arcsin => {
                    // rhs ← Arcsin(rhs)
                    next.push(LoweredOp::Arcsin(Box::new(candidate)));
                }
                InvStep::Arccos => {
                    // rhs ← Arccos(rhs)
                    next.push(LoweredOp::Arccos(Box::new(candidate)));
                }
                InvStep::Arctan => {
                    // rhs ← Arctan(rhs)
                    next.push(LoweredOp::Arctan(Box::new(candidate)));
                }
                InvStep::Sq => {
                    // rhs ← rhs^2
                    next.push(LoweredOp::Pow(
                        Box::new(candidate),
                        Box::new(LoweredOp::Const(2.0)),
                    ));
                }
            }
        }
        candidates = next;
    }

    if candidates.is_empty() {
        return Err(SolveError::InternalError(
            "replay produced no candidates".into(),
        ));
    }

    Ok(SolveResult {
        solutions: candidates,
        complete,
    })
}

// ---------------------------------------------------------------------------
// Polynomial detection and solver
// ---------------------------------------------------------------------------

/// Compute the polynomial coefficient vector for a single node.
///
/// Binary op children results must already be on `result_stack` in the order:
/// `a` pushed first (bottom), `b` pushed second (top). This means
/// `result_stack.pop()` yields `b`'s result first, then `a`'s result.
///
/// Returns `Some(result)` where result is `Some(coeffs)` (polynomial) or `None`
/// (non-polynomial). Returns `None` at the outer level only if the result_stack
/// is unexpectedly empty (internal error — caller uses `?` to propagate).
fn compute_poly_node(
    op: &LoweredOp,
    var_idx: usize,
    result_stack: &mut Vec<Option<Vec<LoweredOp>>>,
) -> Option<Option<Vec<LoweredOp>>> {
    let inner: Option<Vec<LoweredOp>> = match op {
        LoweredOp::Const(c) => Some(vec![LoweredOp::Const(*c)]),
        LoweredOp::Var(i) => {
            if *i == var_idx {
                // 0 + 1·x
                Some(vec![LoweredOp::Const(0.0), LoweredOp::Const(1.0)])
            } else {
                // Constant w.r.t. var_idx
                Some(vec![LoweredOp::Var(*i)])
            }
        }
        LoweredOp::Neg(_) => {
            // One child result on top.
            let child = result_stack.pop()?;
            let child_coeffs = child?;
            let negated = child_coeffs
                .into_iter()
                .map(|c| LoweredOp::Neg(Box::new(c)))
                .collect();
            Some(negated)
        }
        LoweredOp::Add(_, _) => {
            // b's result is on top, a's result is below.
            let b_res = result_stack.pop()?;
            let a_res = result_stack.pop()?;
            let a_coeffs = a_res?;
            let b_coeffs = b_res?;
            Some(poly_add(a_coeffs, b_coeffs))
        }
        LoweredOp::Sub(_, _) => {
            // b's result is on top, a's result is below.
            let b_res = result_stack.pop()?;
            let a_res = result_stack.pop()?;
            let a_coeffs = a_res?;
            let b_coeffs = b_res?;
            Some(poly_sub(a_coeffs, b_coeffs))
        }
        LoweredOp::Mul(_, _) => {
            // b's result is on top, a's result is below.
            let b_res = result_stack.pop()?;
            let a_res = result_stack.pop()?;
            let a_coeffs = a_res?;
            let b_coeffs = b_res?;
            Some(poly_mul(a_coeffs, b_coeffs))
        }
        LoweredOp::Pow(base, exp) => {
            // exp's result is on top (Enter(base) ran first, Enter(exp) ran second).
            let _exp_res = result_stack.pop()?;
            let _base_res = result_stack.pop()?;
            // Special case: Pow(Var(var_idx), Const(n)) for positive integer n ≤ 20.
            if let (LoweredOp::Var(bi), LoweredOp::Const(n)) = (base.as_ref(), exp.as_ref()) {
                if *bi == var_idx && n.fract().abs() < 1e-12 {
                    let n_int = n.round() as usize;
                    if n_int <= 20 {
                        let mut coeffs = vec![LoweredOp::Const(0.0); n_int + 1];
                        coeffs[n_int] = LoweredOp::Const(1.0);
                        return Some(Some(coeffs));
                    }
                }
            }
            // General case: treat as constant if var_idx not involved.
            let base_contains = count_var_occurrences(base, var_idx) > 0;
            let exp_contains = count_var_occurrences(exp, var_idx) > 0;
            if !base_contains && !exp_contains {
                Some(vec![op.clone()])
            } else {
                // Non-monomial polynomial involvement.
                None
            }
        }
        _ => {
            // Transcendental or other op.
            let contains = count_var_occurrences(op, var_idx) > 0;
            if contains {
                None
            } else {
                Some(vec![op.clone()])
            }
        }
    };
    Some(inner)
}

/// Attempt to represent `expr` as a polynomial in `Var(var_idx)`.
///
/// Returns `Some(coeffs)` where `coeffs[k]` is the coefficient of `x^k`
/// (as a `LoweredOp` that does not involve `var_idx`), or `None` if the
/// expression is not polynomial in `var_idx`.
///
/// Implementation is iterative bottom-up using the `LoweredOp` tree.
/// Each node is assigned a `Vec<LoweredOp>` of coefficients, or `None` if
/// not polynomial. The traversal uses a post-order work-stack.
pub(crate) fn as_polynomial(expr: &LoweredOp, var_idx: usize) -> Option<Vec<LoweredOp>> {
    // We need post-order traversal with results from children.
    // Use an explicit stack that holds (node_ref, visited, child_results_slot).
    // Each node is visited twice: first to push children, then to collect.
    //
    // We simulate this with a Vec<PolyFrame>.

    /// Stack frame for polynomial coefficient computation.
    enum PolyFrame<'a> {
        /// First visit: push children.
        Enter(&'a LoweredOp),
        /// Post-visit: collect results and compute.
        Compute(&'a LoweredOp),
    }

    let mut stack: Vec<PolyFrame> = vec![PolyFrame::Enter(expr)];
    // Results storage: when Compute pops, children's results are on result_stack.
    let mut result_stack: Vec<Option<Vec<LoweredOp>>> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            PolyFrame::Enter(op) => {
                match op {
                    LoweredOp::Const(_) | LoweredOp::Var(_) => {
                        // Leaves: push Compute directly with no children.
                        stack.push(PolyFrame::Compute(op));
                    }
                    LoweredOp::Add(a, b) | LoweredOp::Sub(a, b) | LoweredOp::Mul(a, b) => {
                        // Push compute then children (children execute first).
                        stack.push(PolyFrame::Compute(op));
                        stack.push(PolyFrame::Enter(b));
                        stack.push(PolyFrame::Enter(a));
                    }
                    LoweredOp::Neg(c) => {
                        stack.push(PolyFrame::Compute(op));
                        stack.push(PolyFrame::Enter(c));
                    }
                    LoweredOp::Pow(base, exp) => {
                        // Only handle Pow(Var(var_idx), Const(n)) specially.
                        // For other Pow cases, treat as enter+compute normally.
                        stack.push(PolyFrame::Compute(op));
                        stack.push(PolyFrame::Enter(exp));
                        stack.push(PolyFrame::Enter(base));
                    }
                    _ => {
                        // Transcendental or other — treat as constant or fail.
                        stack.push(PolyFrame::Compute(op));
                    }
                }
            }
            PolyFrame::Compute(op) => {
                // Compute the polynomial coefficient vector for this node.
                // Children have already been processed and their results are on result_stack.
                // For binary ops, execution order was: Enter(a) then Enter(b), so
                // result_stack top = b's result, below = a's result.
                let result = compute_poly_node(op, var_idx, &mut result_stack)?;
                result_stack.push(result);
            }
        }
    }

    // Final result should be the single remaining value on result_stack.
    result_stack.pop().flatten()
}

/// Add two polynomial coefficient vectors.
pub(crate) fn poly_add(a: Vec<LoweredOp>, b: Vec<LoweredOp>) -> Vec<LoweredOp> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(ai), Some(bi)) => {
                result.push(LoweredOp::Add(Box::new(ai), Box::new(bi)));
            }
            (Some(ai), None) => result.push(ai),
            (None, Some(bi)) => result.push(bi),
            (None, None) => break,
        }
    }
    result
}

/// Subtract two polynomial coefficient vectors (a - b).
pub(crate) fn poly_sub(a: Vec<LoweredOp>, b: Vec<LoweredOp>) -> Vec<LoweredOp> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(ai), Some(bi)) => {
                result.push(LoweredOp::Sub(Box::new(ai), Box::new(bi)));
            }
            (Some(ai), None) => result.push(ai),
            (None, Some(bi)) => result.push(LoweredOp::Neg(Box::new(bi))),
            (None, None) => break,
        }
    }
    result
}

/// Multiply two polynomial coefficient vectors (polynomial convolution).
pub(crate) fn poly_mul(a: Vec<LoweredOp>, b: Vec<LoweredOp>) -> Vec<LoweredOp> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let result_len = a.len() + b.len() - 1;
    let mut result: Vec<Option<LoweredOp>> = vec![None; result_len];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let term = LoweredOp::Mul(Box::new(ai.clone()), Box::new(bj.clone()));
            let slot = &mut result[i + j];
            *slot = Some(match slot.take() {
                None => term,
                Some(prev) => LoweredOp::Add(Box::new(prev), Box::new(term)),
            });
        }
    }
    result
        .into_iter()
        .map(|opt| opt.unwrap_or(LoweredOp::Const(0.0)))
        .collect()
}

/// Strip trailing `Const(0.0)` entries from a coefficient vector.
///
/// Returns the effective length after stripping (i.e. last non-zero index + 1).
pub(crate) fn strip_trailing_zeros(coeffs: &[LoweredOp]) -> usize {
    let mut end = coeffs.len();
    while end > 1 {
        if let LoweredOp::Const(c) = &coeffs[end - 1] {
            if c.abs() < 1e-14 {
                end -= 1;
                continue;
            }
        }
        break;
    }
    end
}

/// Canonicalize each coefficient to fold `Add(Const, Const)` etc. down to a
/// single `Const(...)` value when possible. Used by the Cardano/Ferrari
/// dispatch to ensure literal-numeric input.
fn fold_coeffs_to_const(coeffs: &[LoweredOp]) -> Vec<LoweredOp> {
    coeffs
        .iter()
        .map(|c| {
            let canon = crate::cas::canonicalize::canonicalize(c);
            canon.into_op()
        })
        .collect()
}

/// Solve a polynomial equation given coefficient vector (already `lhs - rhs`).
///
/// `coeffs[k]` is the coefficient of `x^k`. Returns solutions or error.
fn solve_polynomial(coeffs: Vec<LoweredOp>) -> Result<SolveResult, SolveError> {
    let effective_len = strip_trailing_zeros(&coeffs);
    if effective_len == 0 {
        return Err(SolveError::NotSolvable {
            reason: "constant equation (zero coefficients after stripping)".into(),
        });
    }
    let degree = effective_len - 1;

    match degree {
        0 => {
            // Constant: 0 = constant (if non-zero, no solution; if zero, infinite).
            Err(SolveError::NotSolvable {
                reason: "constant equation".into(),
            })
        }
        1 => {
            // a*x + b = 0 → x = -b / a
            // coeffs[0] = b, coeffs[1] = a
            let b = coeffs[0].clone();
            let a = coeffs[1].clone();
            let solution = LoweredOp::Div(Box::new(LoweredOp::Neg(Box::new(b))), Box::new(a));
            Ok(SolveResult {
                solutions: vec![solution],
                complete: true,
            })
        }
        2 => {
            // a*x^2 + b*x + c = 0 → quadratic formula
            // coeffs[0] = c, coeffs[1] = b, coeffs[2] = a
            let c = coeffs[0].clone();
            let b = coeffs[1].clone();
            let a = coeffs[2].clone();

            // disc = b^2 - 4*a*c
            let disc = LoweredOp::Sub(
                Box::new(LoweredOp::Pow(
                    Box::new(b.clone()),
                    Box::new(LoweredOp::Const(2.0)),
                )),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Mul(
                        Box::new(LoweredOp::Const(4.0)),
                        Box::new(a.clone()),
                    )),
                    Box::new(c),
                )),
            );

            // two_a = 2 * a
            let two_a = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(a));

            // neg_b = -b
            let neg_b = LoweredOp::Neg(Box::new(b));

            // solution1 = (-b + sqrt(disc)) / (2*a)
            let sol1 = LoweredOp::Div(
                Box::new(LoweredOp::Add(
                    Box::new(neg_b.clone()),
                    Box::new(LoweredOp::Sqrt(Box::new(disc.clone()))),
                )),
                Box::new(two_a.clone()),
            );

            // solution2 = (-b - sqrt(disc)) / (2*a)
            let sol2 = LoweredOp::Div(
                Box::new(LoweredOp::Sub(
                    Box::new(neg_b),
                    Box::new(LoweredOp::Sqrt(Box::new(disc))),
                )),
                Box::new(two_a),
            );

            Ok(SolveResult {
                solutions: vec![sol1, sol2],
                complete: true,
            })
        }
        3 => {
            // Cardano (closed-form). Returns up to three real roots.
            // Canonicalize coefficients first to fold any unsimplified
            // Add(Const, Const) → Const before passing to the solver.
            let folded = fold_coeffs_to_const(&coeffs);
            crate::cas::cardano_ferrari::solve_cubic(&folded)
        }
        4 => {
            // Ferrari (closed-form). Returns up to four real roots.
            let folded = fold_coeffs_to_const(&coeffs);
            crate::cas::cardano_ferrari::solve_quartic(&folded)
        }
        d => Err(SolveError::HighDegreePoly { degree: d }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::{eval_real, EvalCtx};

    const TOL: f64 = 1e-9;

    fn eval_solution(sol: &LoweredOp, var_val: f64) -> f64 {
        // Solution expressions should be constant (no Var(0) in them after solving).
        // Evaluate with an empty context or a context that covers extra vars.
        let bindings = [var_val];
        let ctx = EvalCtx::new(&bindings);
        eval_real(sol, &ctx).unwrap_or(f64::NAN)
    }

    fn eval_no_var(sol: &LoweredOp) -> f64 {
        let ctx = EvalCtx::new(&[]);
        eval_real(sol, &ctx).unwrap_or(f64::NAN)
    }

    /// test 1: solve(x + 3 = 7) → x = 4
    #[test]
    fn test_solve_linear_add() {
        let lhs = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0)));
        let rhs = LoweredOp::Const(7.0);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - 4.0).abs() < TOL, "expected 4.0, got {val}");
    }

    /// test 2: solve(3 * x = 9) → x = 3
    #[test]
    fn test_solve_linear_mul() {
        let lhs = LoweredOp::Mul(Box::new(LoweredOp::Const(3.0)), Box::new(LoweredOp::Var(0)));
        let rhs = LoweredOp::Const(9.0);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - 3.0).abs() < TOL, "expected 3.0, got {val}");
    }

    /// test 3: solve(exp(x) = e) → x = ln(e) = 1
    #[test]
    fn test_solve_exp() {
        let lhs = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let rhs = LoweredOp::Const(std::f64::consts::E);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        // Solution should be Ln(Const(e)) which evaluates to 1.0
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - 1.0).abs() < TOL, "expected ~1.0, got {val}");
    }

    /// test 4: solve(ln(x) = 0) → x = exp(0) = 1
    #[test]
    fn test_solve_ln() {
        let lhs = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
        let rhs = LoweredOp::Const(0.0);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        // Solution should be Exp(Const(0.0)) = 1.0
        let val = eval_no_var(&result.solutions[0]);
        assert!(
            (val - 1.0).abs() < TOL,
            "expected 1.0 (= exp(0)), got {val}"
        );
    }

    /// test 5: solve_zero(x^2 - 4) → x = ±2
    #[test]
    fn test_solve_quadratic_roots() {
        let expr = LoweredOp::Sub(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Const(4.0)),
        );
        let result = solve_zero(&expr, 0).expect("solve_zero should succeed");
        assert_eq!(
            result.solutions.len(),
            2,
            "expected 2 solutions, got {}",
            result.solutions.len()
        );
        let mut vals: Vec<f64> = result.solutions.iter().map(eval_no_var).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (vals[0] - (-2.0)).abs() < TOL,
            "expected -2.0, got {}",
            vals[0]
        );
        assert!((vals[1] - 2.0).abs() < TOL, "expected 2.0, got {}", vals[1]);
        assert!(result.complete);
    }

    /// test 6: solve(exp(x + 1) = e) → x = 0
    #[test]
    fn test_solve_nested() {
        let inner = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let lhs = LoweredOp::Exp(Box::new(inner));
        let rhs = LoweredOp::Const(std::f64::consts::E);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - 0.0).abs() < TOL, "expected 0.0, got {val}");
    }

    /// test 7: solve(Const(5) = Const(5)) → Err(NotSolvable)
    #[test]
    fn test_solve_no_var() {
        let lhs = LoweredOp::Const(5.0);
        let rhs = LoweredOp::Const(5.0);
        let result = solve(&lhs, &rhs, 0);
        assert!(
            matches!(result, Err(SolveError::NotSolvable { .. })),
            "expected NotSolvable, got {:?}",
            result.err().map(|e| format!("{e:?}"))
        );
    }

    /// test 8: solve_zero(x^5 + x) → Err(HighDegreePoly { degree: 5 })
    ///
    /// Wave 74 lifted the cubic and quartic paths to closed-form (Cardano,
    /// Ferrari) so the solver no longer returns `HighDegreePoly` for degree
    /// 3/4 inputs. The new threshold is degree ≥ 5.
    #[test]
    fn test_solve_high_degree() {
        let expr = LoweredOp::Add(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(5.0)),
            )),
            Box::new(LoweredOp::Var(0)),
        );
        let result = solve_zero(&expr, 0);
        assert!(
            matches!(result, Err(SolveError::HighDegreePoly { degree: 5 })),
            "expected HighDegreePoly(5), got {:?}",
            result.err().map(|e| format!("{e:?}"))
        );
    }

    /// test 9: solve_zero(2*x + 6) → x = -3
    #[test]
    fn test_solve_linear_poly() {
        let expr = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Const(2.0)),
                Box::new(LoweredOp::Var(0)),
            )),
            Box::new(LoweredOp::Const(6.0)),
        );
        let result = solve_zero(&expr, 0).expect("solve_zero should succeed");
        assert_eq!(result.solutions.len(), 1);
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - (-3.0)).abs() < TOL, "expected -3.0, got {val}");
    }

    /// test 10: solve(sqrt(x) = 3) → x = 9
    #[test]
    fn test_solve_sqrt() {
        let lhs = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        let rhs = LoweredOp::Const(3.0);
        let result = solve(&lhs, &rhs, 0).expect("solve should succeed");
        assert_eq!(result.solutions.len(), 1);
        // Solution should be Pow(Const(3.0), Const(2.0)) = 9.0
        let val = eval_no_var(&result.solutions[0]);
        assert!((val - 9.0).abs() < TOL, "expected 9.0, got {val}");
    }

    // Additional helper to silence unused warning for eval_solution
    #[test]
    fn test_eval_solution_helper_smoke() {
        let op = LoweredOp::Const(42.0);
        let v = eval_solution(&op, 0.0);
        assert!((v - 42.0).abs() < 1e-12);
    }
}
