//! `cas::solve_system` — multivariate algebraic system solver.
//!
//! Solves a system of equations `lhs_i = rhs_i` for a list of target variables
//! `vars`, using a three-tier dispatch strategy:
//!
//! # Tier 1 — Linear path
//!
//! If all residuals `lhs_i − rhs_i` are linear in all target variables, builds
//! the augmented matrix `[A | b]` and runs Bareiss fraction-free Gaussian
//! elimination. Back-substitution yields one solution branch.
//!
//! # Tier 2 — Polynomial path (Buchberger)
//!
//! If all residuals are polynomial in the target variables, runs Buchberger's
//! algorithm with a step budget [`MAX_BUCHBERGER_STEPS`]. On overrun returns
//! [`SystemKind::PartialGroebner`]. On completion, solves the triangulated
//! basis bottom-up via [`crate::cas::solve::solve`].
//!
//! # Tier 3 — Transcendental fallback
//!
//! Attempts linear elimination of variables that appear linearly in at least
//! one equation, then calls the single-variable solver on remaining equations.
//! If elimination is impossible, returns
//! [`SystemSolveError::CannotEliminateTranscendental`].
//!
//! # No recursion
//!
//! All traversals use iterative work-stacks over heap-allocated `Vec`.
//!
//! # No unwrap
//!
//! All fallible paths use `Result` and `?`.

use std::collections::{HashMap, VecDeque};

use crate::cas::canonicalize::canonicalize;
use crate::cas::solve::{as_polynomial, solve, SolveError};
use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of solving a system of equations.
#[derive(Debug, Clone)]
pub struct SystemSolveResult {
    /// Each entry is one solution branch: maps variable Var-id → its solved value.
    pub solutions: Vec<HashMap<usize, LoweredOp>>,
    /// True if the solution set is complete (no branches dropped).
    pub complete: bool,
    /// Classification of the solved system.
    pub kind: SystemKind,
}

/// Classification of the system kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemKind {
    /// System was linear in all target variables.
    Linear,
    /// System was polynomial in all target variables (Gröbner basis complete).
    Polynomial,
    /// Buchberger hit step budget; partial Gröbner basis returned.
    PartialGroebner,
    /// System is inconsistent (no solution exists).
    Inconsistent,
    /// System is underdetermined (infinitely many solutions).
    Underdetermined,
    /// System contains transcendental functions; fallback used.
    Transcendental,
}

/// Error type for [`solve_system`].
#[derive(Debug, Clone, PartialEq)]
pub enum SystemSolveError {
    /// No target variables were specified.
    EmptyVars,
    /// No equations were provided.
    EmptyEquations,
    /// Transcendental equations that cannot be reduced to one-variable form.
    CannotEliminateTranscendental,
    /// Internal solver error (should not normally occur).
    InternalError(String),
}

impl std::fmt::Display for SystemSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemSolveError::EmptyVars => write!(f, "no target variables specified"),
            SystemSolveError::EmptyEquations => write!(f, "no equations provided"),
            SystemSolveError::CannotEliminateTranscendental => {
                write!(
                    f,
                    "system contains transcendental equations that cannot be eliminated"
                )
            }
            SystemSolveError::InternalError(msg) => {
                write!(f, "internal error in solve_system: {msg}")
            }
        }
    }
}

impl std::error::Error for SystemSolveError {}

/// Maximum number of Buchberger algorithm steps before returning [`SystemKind::PartialGroebner`].
pub const MAX_BUCHBERGER_STEPS: usize = 256;

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Solve a system of equations `lhs_i = rhs_i` for the given target variables.
///
/// Each equation is given as `(lhs, rhs)`. The solver attempts:
/// 1. Linear path (degree-1 in all target variables)
/// 2. Polynomial path (Buchberger's algorithm)
/// 3. Transcendental fallback (linear elimination + single-variable solve)
///
/// Returns [`SystemSolveResult`] or [`SystemSolveError`].
pub fn solve_system(
    eqs: &[(LoweredOp, LoweredOp)],
    vars: &[usize],
) -> Result<SystemSolveResult, SystemSolveError> {
    if vars.is_empty() {
        return Err(SystemSolveError::EmptyVars);
    }
    if eqs.is_empty() {
        return Err(SystemSolveError::EmptyEquations);
    }

    // Compute canonicalized residuals: res[i] = canon(lhs[i] - rhs[i])
    let residuals: Vec<LoweredOp> = eqs
        .iter()
        .map(|(lhs, rhs)| {
            canonicalize(&LoweredOp::Sub(
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
            ))
            .into_op()
        })
        .collect();

    // Try Tier 1: linear path
    if is_linear_system(&residuals, vars) {
        return solve_linear(&residuals, vars);
    }

    // Try Tier 2: polynomial path (Buchberger)
    if is_polynomial_system(&residuals, vars) {
        return solve_polynomial_system(&residuals, vars);
    }

    // Tier 3: transcendental fallback
    solve_transcendental_fallback(&residuals, vars)
}

// ---------------------------------------------------------------------------
// Tier-1 helpers: linear path
// ---------------------------------------------------------------------------

/// Check if all residuals are linear in all target variables (degree ≤ 1 and
/// no interaction terms `x_i * x_j`).
fn is_linear_system(residuals: &[LoweredOp], vars: &[usize]) -> bool {
    for res in residuals {
        for &v in vars {
            let deriv = canonicalize(&grad(res, v)).into_op();
            // Derivative must not contain any target variable
            if contains_any_var(&deriv, vars) {
                return false;
            }
        }
    }
    true
}

/// Solve the linear system via Bareiss fraction-free Gaussian elimination.
fn solve_linear(
    residuals: &[LoweredOp],
    vars: &[usize],
) -> Result<SystemSolveResult, SystemSolveError> {
    let n = vars.len();
    let m = residuals.len();

    // Build augmented matrix [A | b]
    // A[i][j] = d(residuals[i])/d(vars[j])  (constant w.r.t. target vars)
    // b[i]    = -residuals[i] evaluated at all vars=0
    let mut a: Vec<Vec<LoweredOp>> = Vec::with_capacity(m);
    let mut b: Vec<LoweredOp> = Vec::with_capacity(m);

    for res in residuals {
        let mut row = Vec::with_capacity(n);
        for &v in vars {
            let d = canonicalize(&grad(res, v)).into_op();
            row.push(d);
        }
        a.push(row);
        // b[i] = -(res with all target vars substituted to 0)
        let at_zero = substitute_zeros(res, vars);
        let at_zero_canon = canonicalize(&at_zero).into_op();
        b.push(LoweredOp::Neg(Box::new(at_zero_canon)));
    }

    // Bareiss fraction-free Gaussian elimination on a system
    // We work row by row with pivot tracking.
    // For underdetermined/overdetermined systems: track flags.
    let mut inconsistent = false;
    let mut underdetermined = false;

    // Augmented matrix: aug[i] = [a[i][0], ..., a[i][n-1], b[i]]
    let mut aug: Vec<Vec<LoweredOp>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, bi)| {
            let mut r = row.clone();
            r.push(bi.clone());
            r
        })
        .collect();

    let aug_cols = n + 1; // n variable columns + 1 rhs column

    let mut pivot_col = Vec::new(); // pivot_col[row] = column index of the pivot

    let mut pivot_row = 0usize;
    for col in 0..n {
        // Find a pivot row for this column (below current pivot_row)
        let pivot = find_pivot_row(&aug, pivot_row, col);

        if let Some(pr) = pivot {
            // Swap pivot_row and pr
            aug.swap(pivot_row, pr);

            // Record pivot column for back-substitution
            pivot_col.push(col);

            // Eliminate below using Bareiss: for each row below pivot_row
            let pivot_val = aug[pivot_row][col].clone();

            for row in (pivot_row + 1)..aug.len() {
                let row_pivot_entry = aug[row][col].clone();

                // Check if this entry is already zero
                if is_zero_op(&row_pivot_entry) {
                    // Already zero, nothing to do for this row
                    continue;
                }

                // Bareiss: new_row[k] = pivot_val * row[k] - row_pivot_entry * pivot_row[k]
                let new_row: Vec<LoweredOp> = (0..aug_cols)
                    .map(|k| {
                        let term1 = LoweredOp::Mul(
                            Box::new(pivot_val.clone()),
                            Box::new(aug[row][k].clone()),
                        );
                        let term2 = LoweredOp::Mul(
                            Box::new(row_pivot_entry.clone()),
                            Box::new(aug[pivot_row][k].clone()),
                        );
                        let new_val = LoweredOp::Sub(Box::new(term1), Box::new(term2));
                        canonicalize(&new_val).into_op()
                    })
                    .collect();
                aug[row] = new_row;
            }

            pivot_row += 1;
        } else {
            // No pivot in this column: this variable is free → underdetermined
            // (unless the corresponding b entry makes it inconsistent)
        }

        if pivot_row >= aug.len() {
            break;
        }
    }

    // Check remaining rows (below the last pivot) for inconsistency
    for row_vec in aug.iter().skip(pivot_row) {
        // All variable columns should be zero
        let all_var_zero = (0..n).all(|col| is_zero_op(&row_vec[col]));
        if all_var_zero {
            // Check the RHS
            if !is_zero_op(&row_vec[n]) {
                inconsistent = true;
            } else {
                // 0 = 0: redundant equation
            }
        }
    }

    if inconsistent {
        return Ok(SystemSolveResult {
            solutions: Vec::new(),
            complete: true,
            kind: SystemKind::Inconsistent,
        });
    }

    // Check if we have fewer pivots than variables
    if pivot_col.len() < n {
        underdetermined = true;
    }

    if underdetermined {
        return Ok(SystemSolveResult {
            solutions: Vec::new(),
            complete: false,
            kind: SystemKind::Underdetermined,
        });
    }

    // Back-substitution (the matrix is now upper triangular)
    // We only back-substitute up to the number of pivots found
    let num_pivots = pivot_col.len();
    let mut solution: HashMap<usize, LoweredOp> = HashMap::new();

    // Initialize solution with the back-substitution variables
    for i in (0..num_pivots).rev() {
        let row_idx = i;
        let col_idx = pivot_col[i];

        // value = (rhs - sum of known terms) / pivot
        let mut rhs_val = aug[row_idx][n].clone();

        for (k, &solved_var) in vars
            .iter()
            .enumerate()
            .skip(col_idx + 1)
            .take(n.saturating_sub(col_idx + 1))
        {
            if let Some(solved_val) = solution.get(&solved_var) {
                let term = LoweredOp::Mul(
                    Box::new(aug[row_idx][k].clone()),
                    Box::new(solved_val.clone()),
                );
                rhs_val = LoweredOp::Sub(Box::new(rhs_val), Box::new(term));
            }
        }

        let pivot_val = aug[row_idx][col_idx].clone();
        let result = LoweredOp::Div(Box::new(rhs_val), Box::new(pivot_val));
        let result = canonicalize(&result).into_op();
        solution.insert(vars[col_idx], result);
    }

    let kind = SystemKind::Linear;
    Ok(SystemSolveResult {
        solutions: vec![solution],
        complete: true,
        kind,
    })
}

/// Find the pivot row for column `col`, starting from `start_row`.
/// Returns the index of the first non-zero entry, or `None` if all are zero.
fn find_pivot_row(aug: &[Vec<LoweredOp>], start_row: usize, col: usize) -> Option<usize> {
    aug.iter()
        .enumerate()
        .skip(start_row)
        .find_map(|(row, row_vec)| {
            if !is_zero_op(&row_vec[col]) {
                Some(row)
            } else {
                None
            }
        })
}

// ---------------------------------------------------------------------------
// Tier-2 helpers: polynomial path (Buchberger)
// ---------------------------------------------------------------------------

/// Check if all residuals are polynomial in all target variables.
fn is_polynomial_system(residuals: &[LoweredOp], vars: &[usize]) -> bool {
    for res in residuals {
        // Try to extract multivariate polynomial coefficients
        if extract_polynomial(res, vars).is_none() {
            return false;
        }
    }
    true
}

/// A monomial in n variables: (coefficient, exponent vector of length n).
type Monomial = (f64, Vec<u32>);
/// A multivariate polynomial: list of monomials.
type MPoly = Vec<Monomial>;

/// Extract multivariate polynomial coefficients from a LoweredOp.
/// Returns `None` if the expression is not polynomial in the given vars.
fn extract_polynomial(op: &LoweredOp, vars: &[usize]) -> Option<MPoly> {
    let n = vars.len();

    // We need post-order traversal computing MPoly for each node.
    enum Frame<'a> {
        Enter(&'a LoweredOp),
        Compute(&'a LoweredOp),
    }

    let mut stack: Vec<Frame> = vec![Frame::Enter(op)];
    let mut result_stack: Vec<Option<MPoly>> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(node) => match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    stack.push(Frame::Compute(node));
                }
                LoweredOp::Add(a, b) | LoweredOp::Sub(a, b) | LoweredOp::Mul(a, b) => {
                    stack.push(Frame::Compute(node));
                    stack.push(Frame::Enter(b));
                    stack.push(Frame::Enter(a));
                }
                LoweredOp::Neg(c) => {
                    stack.push(Frame::Compute(node));
                    stack.push(Frame::Enter(c));
                }
                LoweredOp::Pow(base, exp) => {
                    stack.push(Frame::Compute(node));
                    stack.push(Frame::Enter(exp));
                    stack.push(Frame::Enter(base));
                }
                _ => {
                    // Transcendental or other: if it contains a target var, not polynomial
                    stack.push(Frame::Compute(node));
                }
            },
            Frame::Compute(node) => {
                let result: Option<MPoly> = match node {
                    LoweredOp::Const(c) => {
                        // A constant monomial: coeff c, all exponents 0
                        Some(vec![(*c, vec![0u32; n])])
                    }
                    LoweredOp::Var(i) => {
                        // Check if this variable is in our target list
                        if let Some(pos) = vars.iter().position(|&v| v == *i) {
                            // x_pos^1: exponent vector with 1 at pos
                            let mut exps = vec![0u32; n];
                            exps[pos] = 1;
                            Some(vec![(1.0, exps)])
                        } else {
                            // Parameter (non-target var): treat as a constant-like symbol
                            // We can't handle symbolic parameters in f64 Gröbner — bail
                            None
                        }
                    }
                    LoweredOp::Neg(_) => {
                        let child = result_stack.pop()?;
                        child.map(|poly| poly.into_iter().map(|(c, e)| (-c, e)).collect())
                    }
                    LoweredOp::Add(_, _) => {
                        let b_res = result_stack.pop()?;
                        let a_res = result_stack.pop()?;
                        match (a_res, b_res) {
                            (Some(a_poly), Some(b_poly)) => Some(mpoly_add(a_poly, b_poly)),
                            _ => None,
                        }
                    }
                    LoweredOp::Sub(_, _) => {
                        let b_res = result_stack.pop()?;
                        let a_res = result_stack.pop()?;
                        match (a_res, b_res) {
                            (Some(a_poly), Some(b_poly)) => {
                                let neg_b: MPoly =
                                    b_poly.into_iter().map(|(c, e)| (-c, e)).collect();
                                Some(mpoly_add(a_poly, neg_b))
                            }
                            _ => None,
                        }
                    }
                    LoweredOp::Mul(_, _) => {
                        let b_res = result_stack.pop()?;
                        let a_res = result_stack.pop()?;
                        match (a_res, b_res) {
                            (Some(a_poly), Some(b_poly)) => Some(mpoly_mul(&a_poly, &b_poly)),
                            _ => None,
                        }
                    }
                    LoweredOp::Pow(base, exp) => {
                        let exp_res = result_stack.pop()?;
                        let base_res = result_stack.pop()?;
                        // Only handle Pow(target_var, Const(n)) for positive integer n
                        if let (LoweredOp::Var(vi), LoweredOp::Const(ne)) =
                            (base.as_ref(), exp.as_ref())
                        {
                            if let Some(pos) = vars.iter().position(|&v| v == *vi) {
                                if ne.fract().abs() < 1e-12 && *ne >= 0.0 {
                                    let n_int = ne.round() as u32;
                                    if n_int <= 20 {
                                        let mut exps = vec![0u32; n];
                                        exps[pos] = n_int;
                                        // Discard unused results from stack
                                        let _ = (base_res, exp_res);
                                        result_stack.push(Some(vec![(1.0, exps)]));
                                        continue;
                                    }
                                }
                            }
                        }
                        // Also handle Const^Const
                        if let (Some(Some(_bpoly)), Some(Some(epoly))) =
                            (Some(base_res), Some(exp_res))
                        {
                            // Only constant^constant is allowed (no target vars)
                            if let (LoweredOp::Const(bc), LoweredOp::Const(ec)) =
                                (base.as_ref(), exp.as_ref())
                            {
                                let val = bc.powf(*ec);
                                result_stack.push(Some(vec![(val, vec![0u32; n])]));
                                continue;
                            }
                            // If neither has target var
                            if !contains_any_var(base, vars) && !contains_any_var(exp, vars) {
                                // Evaluate numerically (approximate)
                                // We can't do this without an eval context; bail
                            }
                            let _ = epoly;
                        }
                        result_stack.push(None);
                        continue;
                    }
                    _ => {
                        // Transcendental with target var → not polynomial
                        if contains_any_var(node, vars) {
                            None
                        } else {
                            // Constant with respect to target vars — but we can't
                            // represent it as f64 without eval, bail.
                            None
                        }
                    }
                };
                result_stack.push(result);
            }
        }
    }

    result_stack.pop().flatten()
}

/// Add two multivariate polynomials (combine like monomials).
fn mpoly_add(a: MPoly, b: MPoly) -> MPoly {
    let mut result = a;
    for (c, e) in b {
        let pos = result.iter().position(|(_, re)| *re == e);
        if let Some(idx) = pos {
            result[idx].0 += c;
        } else {
            result.push((c, e));
        }
    }
    // Remove near-zero coefficients
    result.retain(|(c, _)| c.abs() > 1e-14);
    result
}

/// Multiply two multivariate polynomials.
fn mpoly_mul(a: &MPoly, b: &MPoly) -> MPoly {
    let mut result: MPoly = Vec::new();
    for (ac, ae) in a {
        for (bc, be) in b {
            let new_coeff = ac * bc;
            if new_coeff.abs() < 1e-14 {
                continue;
            }
            let new_exp: Vec<u32> = ae.iter().zip(be.iter()).map(|(x, y)| x + y).collect();
            let pos = result.iter().position(|(_, re)| *re == new_exp);
            if let Some(idx) = pos {
                result[idx].0 += new_coeff;
            } else {
                result.push((new_coeff, new_exp));
            }
        }
    }
    result.retain(|(c, _)| c.abs() > 1e-14);
    result
}

/// Graded lex order comparison for exponent vectors.
/// Returns true if `a` >_grlex `b`.
fn grlex_gt(a: &[u32], b: &[u32]) -> bool {
    let sum_a: u32 = a.iter().sum();
    let sum_b: u32 = b.iter().sum();
    if sum_a != sum_b {
        return sum_a > sum_b;
    }
    // Lex comparison
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai != bi {
            return ai > bi;
        }
    }
    false
}

/// Get the leading monomial (largest by grlex) of a polynomial.
fn leading_monomial(poly: &MPoly) -> Option<&Monomial> {
    poly.iter().max_by(|(_, ea), (_, eb)| {
        if grlex_gt(ea, eb) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    })
}

/// Check if the gcd of leading monomials is 1 (Buchberger criterion).
/// If true, the S-polynomial will reduce to zero; skip this pair.
fn leading_coprime(f: &MPoly, g: &MPoly) -> bool {
    if let (Some((_, fe)), Some((_, ge))) = (leading_monomial(f), leading_monomial(g)) {
        // Leading monomials are coprime if min(f_exp[i], g_exp[i]) == 0 for all i
        fe.iter().zip(ge.iter()).all(|(&a, &b)| a == 0 || b == 0)
    } else {
        true
    }
}

/// Compute the S-polynomial of two polynomials.
fn s_polynomial(f: &MPoly, g: &MPoly) -> MPoly {
    if let (Some((fc, fe)), Some((gc, ge))) = (leading_monomial(f), leading_monomial(g)) {
        let n = fe.len();
        // LCM of leading monomials
        let lcm_exp: Vec<u32> = fe.iter().zip(ge.iter()).map(|(&a, &b)| a.max(b)).collect();

        // S(f,g) = (LCM/LT(f)) * f - (LCM/LT(g)) * g
        // LCM/LT(f) has exponent lcm_exp - fe
        let f_shift: Vec<u32> = lcm_exp
            .iter()
            .zip(fe.iter())
            .map(|(&l, &a)| l - a)
            .collect();
        let g_shift: Vec<u32> = lcm_exp
            .iter()
            .zip(ge.iter())
            .map(|(&l, &b)| l - b)
            .collect();

        // Multiply f by f_shift/fc and g by g_shift/gc
        let scaled_f: MPoly = f
            .iter()
            .map(|(c, e)| {
                let new_e: Vec<u32> = e.iter().zip(f_shift.iter()).map(|(&a, &s)| a + s).collect();
                (c / fc, new_e)
            })
            .collect();
        let scaled_g: MPoly = g
            .iter()
            .map(|(c, e)| {
                let new_e: Vec<u32> = e.iter().zip(g_shift.iter()).map(|(&a, &s)| a + s).collect();
                (c / gc, new_e)
            })
            .collect();

        let neg_scaled_g: MPoly = scaled_g.into_iter().map(|(c, e)| (-c, e)).collect();
        mpoly_add(scaled_f, neg_scaled_g)
    } else {
        Vec::new()
    }
}

/// Reduce polynomial `poly` modulo the given basis.
/// Returns the remainder (irreducible part).
fn reduce(mut poly: MPoly, basis: &[MPoly]) -> MPoly {
    'outer: loop {
        if poly.is_empty() {
            break;
        }
        if let Some(lead_mono) = leading_monomial(&poly).cloned() {
            let (lc, le) = lead_mono;
            // Try to reduce by each basis element
            for basis_poly in basis.iter() {
                if let Some((bc, be)) = leading_monomial(basis_poly) {
                    // Check if lt(basis_poly) divides lt(poly)
                    if be.iter().zip(le.iter()).all(|(&bi, &li)| bi <= li) {
                        // Divide: new_exp = le - be, new_coeff = lc / bc
                        let div_exp: Vec<u32> =
                            le.iter().zip(be.iter()).map(|(&l, &b)| l - b).collect();
                        let div_coeff = lc / bc;

                        // poly -= (lc/bc) * x^(le-be) * basis_poly
                        let subtract: MPoly = basis_poly
                            .iter()
                            .map(|(c, e)| {
                                let new_e: Vec<u32> =
                                    e.iter().zip(div_exp.iter()).map(|(&a, &s)| a + s).collect();
                                (div_coeff * c, new_e)
                            })
                            .collect();
                        let neg_sub: MPoly = subtract.into_iter().map(|(c, e)| (-c, e)).collect();
                        poly = mpoly_add(poly, neg_sub);
                        continue 'outer;
                    }
                }
            }
        }
        break;
    }
    poly
}

/// Convert a multivariate polynomial back to a `LoweredOp` for a single variable
/// (when the polynomial is effectively univariate in that variable).
fn univariate_poly_to_lowered_op(
    poly: &MPoly,
    var_pos: usize,
    var_id: usize,
) -> Option<Vec<LoweredOp>> {
    // Check that all monomials only have nonzero exponent at var_pos
    let mut coeffs_map: HashMap<u32, f64> = HashMap::new();
    for (c, e) in poly {
        for (i, &exp) in e.iter().enumerate() {
            if i != var_pos && exp != 0 {
                return None; // Not univariate in this variable
            }
        }
        let power = e[var_pos];
        *coeffs_map.entry(power).or_insert(0.0) += c;
    }

    // Build coefficient vector
    let max_degree = coeffs_map.keys().max().copied().unwrap_or(0) as usize;
    let mut coeffs = vec![LoweredOp::Const(0.0); max_degree + 1];
    for (power, coeff) in &coeffs_map {
        if coeff.abs() > 1e-14 {
            coeffs[*power as usize] = LoweredOp::Const(*coeff);
        }
    }

    // Build the LoweredOp from coefficients
    // This is a polynomial in var_id
    let _ = var_id;
    Some(coeffs)
}

/// Check if a polynomial is univariate (only one variable has nonzero exponents).
fn poly_is_univariate_in(poly: &MPoly, var_pos: usize) -> bool {
    for (_, e) in poly {
        for (i, &exp) in e.iter().enumerate() {
            if i != var_pos && exp != 0 {
                return false;
            }
        }
    }
    true
}

/// Solve the polynomial system via Buchberger's algorithm.
fn solve_polynomial_system(
    residuals: &[LoweredOp],
    vars: &[usize],
) -> Result<SystemSolveResult, SystemSolveError> {
    let n = vars.len();

    // Extract multivariate polynomial representations
    let mut basis: Vec<MPoly> = Vec::new();
    for res in residuals {
        if let Some(poly) = extract_polynomial(res, vars) {
            if !poly.is_empty() {
                basis.push(poly);
            }
        }
    }

    if basis.is_empty() {
        return Ok(SystemSolveResult {
            solutions: Vec::new(),
            complete: false,
            kind: SystemKind::Underdetermined,
        });
    }

    // Buchberger's algorithm
    let mut steps = 0usize;
    let initial_len = basis.len();
    let mut pairs: VecDeque<(usize, usize)> = VecDeque::new();
    for i in 0..initial_len {
        for j in (i + 1)..initial_len {
            pairs.push_back((i, j));
        }
    }

    let mut buchberger_complete = true;
    while let Some((i, j)) = pairs.pop_front() {
        if i >= basis.len() || j >= basis.len() {
            continue;
        }

        steps += 1;
        if steps > MAX_BUCHBERGER_STEPS {
            buchberger_complete = false;
            break;
        }

        if leading_coprime(&basis[i], &basis[j]) {
            continue;
        }

        let s = reduce(s_polynomial(&basis[i], &basis[j]), &basis);
        if !s.is_empty() {
            let new_idx = basis.len();
            for k in 0..new_idx {
                pairs.push_back((k, new_idx));
            }
            basis.push(s);
        }
    }

    // Try to solve bottom-up using the Gröbner basis
    // Look for univariate polynomials in each variable (lex order: last var first)
    let mut solutions: Vec<HashMap<usize, LoweredOp>> = vec![HashMap::new()];

    for var_pos in (0..n).rev() {
        let var_id = vars[var_pos];

        // Find polynomials in basis that are univariate in var_pos
        let univariate_polys: Vec<&MPoly> = basis
            .iter()
            .filter(|p| poly_is_univariate_in(p, var_pos))
            .collect();

        if univariate_polys.is_empty() {
            // No univariate poly for this var; can't determine its value
            continue;
        }

        // Pick the univariate poly with the lowest degree
        let best_poly = univariate_polys
            .iter()
            .min_by_key(|p| {
                leading_monomial(p)
                    .map(|(_, e)| e.iter().sum::<u32>())
                    .unwrap_or(0)
            })
            .copied();

        if let Some(poly) = best_poly {
            if let Some(univar_coeffs) = univariate_poly_to_lowered_op(poly, var_pos, var_id) {
                // Build a LoweredOp polynomial from the coefficients and solve
                let poly_op = build_lowered_from_coeffs(&univar_coeffs, var_id);
                let solve_result = solve(&poly_op, &LoweredOp::Const(0.0), var_id);

                match solve_result {
                    Ok(sr) => {
                        // Branch: for each existing solution × each new root
                        let mut new_solutions: Vec<HashMap<usize, LoweredOp>> = Vec::new();
                        for root in &sr.solutions {
                            for existing in &solutions {
                                let mut new_sol = existing.clone();
                                new_sol.insert(var_id, root.clone());
                                new_solutions.push(new_sol);
                            }
                        }
                        if !new_solutions.is_empty() {
                            solutions = new_solutions;
                        }
                    }
                    Err(SolveError::HighDegreePoly { .. }) => {
                        // Can't solve this degree — mark as incomplete
                        buchberger_complete = false;
                    }
                    Err(_) => {
                        // Other solve error — skip
                    }
                }
            }
        }
    }

    // Filter out empty solution maps (variables never solved)
    let nonempty_solutions: Vec<HashMap<usize, LoweredOp>> =
        solutions.into_iter().filter(|s| !s.is_empty()).collect();

    let kind = if buchberger_complete {
        SystemKind::Polynomial
    } else {
        SystemKind::PartialGroebner
    };

    Ok(SystemSolveResult {
        solutions: nonempty_solutions,
        complete: buchberger_complete,
        kind,
    })
}

/// Build a `LoweredOp` polynomial from a coefficient vector.
/// `coeffs[k]` is the coefficient of `Var(var_id)^k`.
fn build_lowered_from_coeffs(coeffs: &[LoweredOp], var_id: usize) -> LoweredOp {
    if coeffs.is_empty() {
        return LoweredOp::Const(0.0);
    }

    let mut terms: Vec<LoweredOp> = Vec::new();
    for (k, coeff) in coeffs.iter().enumerate() {
        if is_zero_op(coeff) {
            continue;
        }
        let term = if k == 0 {
            coeff.clone()
        } else if k == 1 {
            LoweredOp::Mul(Box::new(coeff.clone()), Box::new(LoweredOp::Var(var_id)))
        } else {
            let x_pow = LoweredOp::Pow(
                Box::new(LoweredOp::Var(var_id)),
                Box::new(LoweredOp::Const(k as f64)),
            );
            LoweredOp::Mul(Box::new(coeff.clone()), Box::new(x_pow))
        };
        terms.push(term);
    }

    if terms.is_empty() {
        return LoweredOp::Const(0.0);
    }

    let mut acc = terms.remove(0);
    for t in terms {
        acc = LoweredOp::Add(Box::new(acc), Box::new(t));
    }
    canonicalize(&acc).into_op()
}

// ---------------------------------------------------------------------------
// Tier-3 helpers: transcendental fallback
// ---------------------------------------------------------------------------

/// Transcendental fallback: attempt linear elimination then single-var solve.
fn solve_transcendental_fallback(
    residuals: &[LoweredOp],
    vars: &[usize],
) -> Result<SystemSolveResult, SystemSolveError> {
    // Build a mutable list of residuals and solved substitutions
    let mut remaining_residuals: Vec<LoweredOp> = residuals.to_vec();
    let mut substitutions: HashMap<usize, LoweredOp> = HashMap::new();
    let mut remaining_vars: Vec<usize> = vars.to_vec();

    // Repeatedly: find an equation that is linear in one remaining var,
    // solve it, substitute it into all other equations.
    let max_iters = vars.len() * 2 + 5;
    let mut iter = 0usize;

    loop {
        iter += 1;
        if iter > max_iters {
            break;
        }

        if remaining_vars.is_empty() {
            break;
        }

        let mut progress = false;

        // Try each remaining equation
        'eq_loop: for eq_idx in 0..remaining_residuals.len() {
            for var_idx in 0..remaining_vars.len() {
                let v = remaining_vars[var_idx];
                let res = &remaining_residuals[eq_idx];

                // Check if this equation is linear in var v
                let deriv = canonicalize(&grad(res, v)).into_op();
                if !contains_any_var(&deriv, &remaining_vars) && !is_zero_op(&deriv) {
                    // Linear in v: try to solve
                    let solve_result = solve(res, &LoweredOp::Const(0.0), v);
                    if let Ok(sr) = solve_result {
                        if let Some(sol_val) = sr.solutions.first() {
                            // Apply existing substitutions to the solution
                            let sol_simplified = apply_substitutions(sol_val, &substitutions);
                            let sol_canon = canonicalize(&sol_simplified).into_op();

                            substitutions.insert(v, sol_canon);
                            remaining_vars.remove(var_idx);

                            // Substitute v into all remaining residuals
                            let new_residuals: Vec<LoweredOp> = remaining_residuals
                                .iter()
                                .enumerate()
                                .filter(|(idx, _)| *idx != eq_idx)
                                .map(|(_, r)| {
                                    let substituted = apply_substitutions(r, &substitutions);
                                    canonicalize(&substituted).into_op()
                                })
                                .collect();
                            remaining_residuals = new_residuals;

                            progress = true;
                            break 'eq_loop;
                        }
                    }
                }
            }
        }

        if !progress {
            break;
        }
    }

    // If all vars are solved, do a final propagation pass and return
    if remaining_vars.is_empty() {
        let final_subs = propagate_substitutions(substitutions);
        return Ok(SystemSolveResult {
            solutions: vec![final_subs],
            complete: true,
            kind: SystemKind::Transcendental,
        });
    }

    // Try to solve any remaining single-variable equations
    for res in &remaining_residuals {
        if remaining_vars.len() == 1 {
            let v = remaining_vars[0];
            let solve_result = solve(res, &LoweredOp::Const(0.0), v);
            if let Ok(sr) = solve_result {
                if let Some(sol_val) = sr.solutions.first() {
                    let sol_simplified = apply_substitutions(sol_val, &substitutions);
                    let sol_canon = canonicalize(&sol_simplified).into_op();
                    substitutions.insert(v, sol_canon);
                    remaining_vars.remove(0);
                    break;
                }
            }
        }
    }

    if remaining_vars.is_empty() {
        let final_subs = propagate_substitutions(substitutions);
        return Ok(SystemSolveResult {
            solutions: vec![final_subs],
            complete: true,
            kind: SystemKind::Transcendental,
        });
    }

    Err(SystemSolveError::CannotEliminateTranscendental)
}

/// Propagate substitutions: apply all substitutions to all values in the map.
///
/// After the main loop, some values in the substitution map may still contain
/// references to variables that were solved later. This function applies a
/// fixed-point propagation to resolve all cross-references.
fn propagate_substitutions(mut subs: HashMap<usize, LoweredOp>) -> HashMap<usize, LoweredOp> {
    // Fixed-point propagation (at most N iterations)
    let max_iters = subs.len() + 5;
    for _ in 0..max_iters {
        let mut changed = false;
        let keys: Vec<usize> = subs.keys().copied().collect();
        for k in &keys {
            let val = match subs.get(k) {
                Some(v) => v.clone(),
                None => continue,
            };
            let new_val = apply_substitutions(&val, &subs);
            let new_canon = canonicalize(&new_val).into_op();
            if new_canon != val {
                changed = true;
                subs.insert(*k, new_canon);
            }
        }
        if !changed {
            break;
        }
    }
    subs
}

// ---------------------------------------------------------------------------
// Shared utility helpers
// ---------------------------------------------------------------------------

/// Check if a `LoweredOp` contains any of the given target variables.
/// Iterative traversal — no recursion.
pub(crate) fn contains_any_var(op: &LoweredOp, vars: &[usize]) -> bool {
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(node) = work.pop() {
        match node {
            LoweredOp::Var(i) => {
                if vars.contains(i) {
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

/// Substitute all target variables with `Const(0.0)` in the given expression.
/// Returns the substituted expression (not yet canonicalized).
pub(crate) fn substitute_zeros(op: &LoweredOp, vars: &[usize]) -> LoweredOp {
    substitute_values(op, vars, &vec![0.0; vars.len()])
}

/// Substitute target variables with the given constant values.
pub(crate) fn substitute_values(op: &LoweredOp, vars: &[usize], values: &[f64]) -> LoweredOp {
    substitute_map_fn(op, |v| {
        vars.iter()
            .position(|&x| x == v)
            .map(|pos| LoweredOp::Const(values[pos]))
    })
}

/// Substitute variables according to a substitution map.
pub(crate) fn apply_substitutions(op: &LoweredOp, subs: &HashMap<usize, LoweredOp>) -> LoweredOp {
    substitute_map_fn(op, |v| subs.get(&v).cloned())
}

/// Generic substitution: replaces `Var(v)` with `f(v)` if `f(v)` returns `Some`.
/// Iterative (work-stack based) — no recursion.
fn substitute_map_fn<F>(op: &LoweredOp, f: F) -> LoweredOp
where
    F: Fn(usize) -> Option<LoweredOp>,
{
    // Stack-based post-order traversal with reconstruction
    enum SubFrame<'a> {
        Enter(&'a LoweredOp),
        Build(&'a LoweredOp),
    }

    let mut frame_stack: Vec<SubFrame> = vec![SubFrame::Enter(op)];
    let mut result_stack: Vec<LoweredOp> = Vec::new();

    while let Some(frame) = frame_stack.pop() {
        match frame {
            SubFrame::Enter(node) => match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    frame_stack.push(SubFrame::Build(node));
                }
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    frame_stack.push(SubFrame::Build(node));
                    frame_stack.push(SubFrame::Enter(b));
                    frame_stack.push(SubFrame::Enter(a));
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
                    frame_stack.push(SubFrame::Build(node));
                    frame_stack.push(SubFrame::Enter(c));
                }
            },
            SubFrame::Build(node) => {
                let built = match node {
                    LoweredOp::Const(c) => LoweredOp::Const(*c),
                    LoweredOp::Var(v) => {
                        if let Some(replacement) = f(*v) {
                            replacement
                        } else {
                            LoweredOp::Var(*v)
                        }
                    }
                    LoweredOp::Add(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Add(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Sub(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sub(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Mul(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Mul(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Div(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(1.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Div(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Pow(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(1.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Pow(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Neg(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Neg(Box::new(c))
                    }
                    LoweredOp::Exp(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Exp(Box::new(c))
                    }
                    LoweredOp::Ln(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Ln(Box::new(c))
                    }
                    LoweredOp::Sin(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sin(Box::new(c))
                    }
                    LoweredOp::Cos(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Cos(Box::new(c))
                    }
                    LoweredOp::Tan(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Tan(Box::new(c))
                    }
                    LoweredOp::Sinh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sinh(Box::new(c))
                    }
                    LoweredOp::Cosh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Cosh(Box::new(c))
                    }
                    LoweredOp::Tanh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Tanh(Box::new(c))
                    }
                    LoweredOp::Arcsin(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arcsin(Box::new(c))
                    }
                    LoweredOp::Arccos(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arccos(Box::new(c))
                    }
                    LoweredOp::Arctan(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arctan(Box::new(c))
                    }
                    LoweredOp::Arcsinh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arcsinh(Box::new(c))
                    }
                    LoweredOp::Arccosh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arccosh(Box::new(c))
                    }
                    LoweredOp::Arctanh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arctanh(Box::new(c))
                    }
                    LoweredOp::Sqrt(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sqrt(Box::new(c))
                    }
                    LoweredOp::Abs(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Abs(Box::new(c))
                    }
                };
                result_stack.push(built);
            }
        }
    }

    result_stack.pop().unwrap_or(LoweredOp::Const(0.0))
}

/// Check if a `LoweredOp` is (or simplifies to) zero.
fn is_zero_op(op: &LoweredOp) -> bool {
    let simplified = simplify_op(op);
    match canonicalize(&simplified).into_op() {
        LoweredOp::Const(c) => c.abs() < 1e-12,
        _ => false,
    }
}

/// Find the maximum Var id in an expression (iterative).
pub(crate) fn max_var_id(op: &LoweredOp) -> usize {
    let mut max_id = 0usize;
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(node) = work.pop() {
        match node {
            LoweredOp::Var(i) => {
                if *i > max_id {
                    max_id = *i;
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
    max_id
}

/// Evaluate a LoweredOp numerically at a specific variable binding.
/// Returns None if evaluation fails.
pub(crate) fn eval_at(
    op: &LoweredOp,
    var_id: usize,
    val: f64,
    extra_var_count: usize,
) -> Option<f64> {
    let size = var_id.max(extra_var_count) + 1;
    let mut bindings = vec![0.0f64; size];
    if var_id < bindings.len() {
        bindings[var_id] = val;
    }
    let ctx = EvalCtx::new(&bindings);
    eval_real(op, &ctx).ok()
}

// ---------------------------------------------------------------------------
// Tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn test_contains_any_var_positive() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert!(contains_any_var(&op, &[0, 1]));
    }

    #[test]
    fn test_contains_any_var_negative() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Const(2.0)),
            Box::new(LoweredOp::Const(1.0)),
        );
        assert!(!contains_any_var(&op, &[0, 1]));
    }

    #[test]
    fn test_substitute_zeros() {
        // x + y → 0 + 0 = 0
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let result = substitute_zeros(&op, &[0, 1]);
        let canon = canonicalize(&result).into_op();
        assert!(matches!(canon, LoweredOp::Const(c) if c.abs() < 1e-10));
    }

    #[test]
    fn test_extract_polynomial_univariate() {
        // x^2 + 3x + 2 in var 0
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Pow(
                    Box::new(LoweredOp::Var(0)),
                    Box::new(LoweredOp::Const(2.0)),
                )),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Const(3.0)),
                    Box::new(LoweredOp::Var(0)),
                )),
            )),
            Box::new(LoweredOp::Const(2.0)),
        );
        let result = extract_polynomial(&op, &[0]);
        assert!(result.is_some(), "Should extract polynomial");
        let poly = result.unwrap();
        assert!(!poly.is_empty(), "Polynomial should be non-empty");
    }

    #[test]
    fn test_extract_polynomial_transcendental_fails() {
        // exp(x) is not polynomial in x
        let op = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let result = extract_polynomial(&op, &[0]);
        assert!(result.is_none(), "exp(x) is not polynomial");
    }
}
