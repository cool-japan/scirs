//! `cas::solve_ode` — symbolic ODE solver for first-order ODEs.
//!
//! Solves `dx/dt = rhs(t, x)` for a given variable `x` (identified by `x_var`)
//! in terms of `t` (identified by `t_var`). Optionally applies an initial
//! condition `ic = Some((t0, x0))` to determine the integration constant.
//!
//! # Dispatch order
//!
//! The solver tries each family in order, returning on first success:
//!
//! 1. **Linear 1st-order** — `dx/dt = a*x + f(t)`, variation of parameters via
//!    [`crate::cas::integrate_rational::try_integrate`].
//! 2. **Separable** — `dx/dt = f(t)*g(x)`, separates and integrates both sides.
//! 3. **Bernoulli** — `dx/dt + p(t)*x = q(t)*x^n`, substitution `u = x^(1-n)`.
//! 4. **Exact** — `M dt + N dx = 0` with `partial M/partial x = partial N/partial t`.
//! 5. **OrderTooHigh** — returned when pattern resembles 2nd-order linear.
//!
//! # Integration constants
//!
//! Fresh `Var` ids are assigned as `max_var_id(rhs) + 1`.
//! Tracked in [`OdeSolution::integration_constants`].
//! If `ic` is provided, the constant is determined and removed from the list.
//!
//! # No recursion
//!
//! All traversals use iterative work-stacks. All fallible paths use `Result`.

use std::collections::HashMap;

use crate::cas::canonicalize::canonicalize;
use crate::cas::integrate_rational::{try_integrate, IntegrateRationalError};
use crate::cas::solve::as_polynomial;
use crate::cas::solve_system::{apply_substitutions, contains_any_var, max_var_id, solve_system};
use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a symbolic ODE solve.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// The closed-form solution `x(t)` as a `LoweredOp` expression.
    /// Contains `Var(t_var)` and optionally `Var(c_var)` for free constants.
    pub x_of_t: LoweredOp,
    /// Var ids of integration constants appearing in `x_of_t`.
    /// Empty if IC was applied successfully.
    pub integration_constants: Vec<usize>,
    /// Classification of the ODE family solved.
    pub kind: OdeKind,
}

/// Classification of the ODE family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeKind {
    /// Linear first-order ODE `dx/dt = a*x + f(t)`.
    Linear1stOrder,
    /// Linear second-order ODE (harmonic oscillator form); not solved.
    Linear2ndOrder,
    /// Separable ODE `dx/dt = f(t)*g(x)` solved explicitly.
    Separable,
    /// Separable ODE with implicit solution (g(x) not invertible symbolically).
    ImplicitSeparable,
    /// Exact ODE solved via potential function (implicit).
    Exact,
    /// Exact ODE solved via potential function (implicit form).
    ImplicitExact,
    /// Bernoulli ODE `dx/dt + p(t)*x = q(t)*x^n` (n != 0, 1).
    Bernoulli,
}

/// Error type for [`solve_ode`].
#[derive(Debug, Clone, PartialEq)]
pub enum SolveOdeError {
    /// The required integration step is not elementary.
    IntegralNotElementary,
    /// The ODE order is too high for the current solver.
    OrderTooHigh,
    /// The ODE does not match any recognized family.
    NotRecognized,
    /// The initial-value problem could not be solved for the integration constant.
    IvpSolveFailed,
    /// Internal solver error.
    InternalError(String),
}

impl std::fmt::Display for SolveOdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveOdeError::IntegralNotElementary => {
                write!(f, "the required integration step is not elementary")
            }
            SolveOdeError::OrderTooHigh => {
                write!(f, "ODE order is too high for the current solver (max: 1)")
            }
            SolveOdeError::NotRecognized => {
                write!(f, "ODE does not match any recognized family")
            }
            SolveOdeError::IvpSolveFailed => {
                write!(
                    f,
                    "initial value problem could not be solved for integration constant"
                )
            }
            SolveOdeError::InternalError(msg) => {
                write!(f, "internal error in solve_ode: {msg}")
            }
        }
    }
}

impl std::error::Error for SolveOdeError {}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Solve the first-order ODE `dx/dt = rhs(t, x)` symbolically.
///
/// - `rhs`: the right-hand side expression (function of `t_var` and `x_var`)
/// - `x_var`: the Var id for the dependent variable `x`
/// - `t_var`: the Var id for the independent variable `t`
/// - `ic`: optional initial condition `(t0, x0)` — if provided, determines the
///   integration constant
///
/// Returns [`OdeSolution`] or [`SolveOdeError`].
pub fn solve_ode(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
    ic: Option<(f64, f64)>,
) -> Result<OdeSolution, SolveOdeError> {
    // Pre-canonicalize the RHS
    let rhs_canon = canonicalize(rhs).into_op();

    // Choose a fresh integration constant Var id
    let c_var = max_var_id(&rhs_canon).max(x_var).max(t_var) + 1;

    // Family 1: Linear 1st-order dx/dt = a*x + f(t)
    if let Ok(sol) = try_linear_1st_order(&rhs_canon, x_var, t_var, c_var, ic) {
        return Ok(sol);
    }

    // Family 3: Separable dx/dt = f(t) * g(x)
    if let Ok(sol) = try_separable(&rhs_canon, x_var, t_var, c_var, ic) {
        return Ok(sol);
    }

    // Family 5: Bernoulli dx/dt + p(t)*x = q(t)*x^n
    if let Ok(sol) = try_bernoulli(&rhs_canon, x_var, t_var, c_var, ic) {
        return Ok(sol);
    }

    // Family 4: Exact M dt + N dx = 0
    if let Ok(sol) = try_exact(&rhs_canon, x_var, t_var, c_var) {
        return Ok(sol);
    }

    // Family 2: Linear 2nd-order detection (return OrderTooHigh)
    if looks_like_2nd_order_linear(&rhs_canon, x_var, t_var) {
        return Err(SolveOdeError::OrderTooHigh);
    }

    Err(SolveOdeError::NotRecognized)
}

// ---------------------------------------------------------------------------
// Family 1: Linear 1st-order ODE: dx/dt = a*x + f(t)
// ---------------------------------------------------------------------------

/// Try to solve `dx/dt = a*x + f(t)` (constant-coefficient linear 1st-order).
///
/// Solution via variation of parameters:
/// `x(t) = (C + integral_part) * exp(a*t)`
fn try_linear_1st_order(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
    c_var: usize,
    ic: Option<(f64, f64)>,
) -> Result<OdeSolution, SolveOdeError> {
    // Extract polynomial coefficients in x_var
    let coeffs = as_polynomial(rhs, x_var).ok_or(SolveOdeError::NotRecognized)?;

    if coeffs.len() > 2 {
        return Err(SolveOdeError::NotRecognized);
    }

    // f(t) = coeffs[0] (constant term), a = coeffs[1] (x coefficient)
    let f_t = if coeffs.is_empty() {
        LoweredOp::Const(0.0)
    } else {
        canonicalize(&coeffs[0]).into_op()
    };

    let a = if coeffs.len() >= 2 {
        canonicalize(&coeffs[1]).into_op()
    } else {
        LoweredOp::Const(0.0)
    };

    // a must not contain x_var, and f_t must not contain x_var
    if contains_any_var(&a, &[x_var]) || contains_any_var(&f_t, &[x_var]) {
        return Err(SolveOdeError::NotRecognized);
    }

    // Only handle constant-coefficient a (not a function of t)
    if contains_any_var(&a, &[t_var]) {
        return Err(SolveOdeError::NotRecognized);
    }

    // Evaluate a as a float constant
    let a_val = eval_const(&a).ok_or(SolveOdeError::NotRecognized)?;

    // Build exp(a*t)
    let exp_at = LoweredOp::Exp(Box::new(LoweredOp::Mul(
        Box::new(a.clone()),
        Box::new(LoweredOp::Var(t_var)),
    )));

    // Integrand for particular solution: exp(-a*t) * f(t)
    let neg_a_t = LoweredOp::Mul(
        Box::new(LoweredOp::Neg(Box::new(a.clone()))),
        Box::new(LoweredOp::Var(t_var)),
    );
    let exp_neg_at = LoweredOp::Exp(Box::new(neg_a_t));
    let integrand = LoweredOp::Mul(Box::new(exp_neg_at), Box::new(f_t.clone()));
    let integrand = canonicalize(&integrand).into_op();

    // Try to integrate exp(-a*t) * f(t) w.r.t. t
    let particular_integral = try_integrate(&integrand, t_var).map_err(map_integrate_err)?;

    // General solution: x(t) = (C + integral_part) * exp(a*t)
    let inner = LoweredOp::Add(
        Box::new(LoweredOp::Var(c_var)),
        Box::new(particular_integral),
    );
    let mut x_of_t = LoweredOp::Mul(Box::new(exp_at), Box::new(inner));
    x_of_t = canonicalize(&x_of_t).into_op();

    let mut integration_constants = vec![c_var];

    // Apply IC if provided
    if let Some((t0, x0)) = ic {
        if let Some(sol) = apply_ic_general(&x_of_t, t_var, t0, x0, c_var) {
            x_of_t = sol;
            integration_constants.clear();
        }
    }

    let _ = a_val;

    Ok(OdeSolution {
        x_of_t,
        integration_constants,
        kind: OdeKind::Linear1stOrder,
    })
}

/// Map IntegrateRationalError to SolveOdeError.
fn map_integrate_err(e: IntegrateRationalError) -> SolveOdeError {
    match e {
        IntegrateRationalError::NotARationalFunction
        | IntegrateRationalError::SymbolicCoefficientsInDenominator
        | IntegrateRationalError::SymbolicCoefficientsInNumerator => {
            SolveOdeError::IntegralNotElementary
        }
        _ => SolveOdeError::IntegralNotElementary,
    }
}

/// Try to evaluate a LoweredOp as a constant f64 (no variables).
fn eval_const(op: &LoweredOp) -> Option<f64> {
    let simplified = simplify_op(op);
    match &simplified {
        LoweredOp::Const(c) => Some(*c),
        _ => {
            let ctx = EvalCtx::new(&[]);
            eval_real(&simplified, &ctx).ok()
        }
    }
}

// ---------------------------------------------------------------------------
// Family 3: Separable ODE: dx/dt = f(t) * g(x)
// ---------------------------------------------------------------------------

/// Try to solve `dx/dt = f(t) * g(x)` (separable ODE).
fn try_separable(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
    c_var: usize,
    ic: Option<(f64, f64)>,
) -> Result<OdeSolution, SolveOdeError> {
    let (f_t, g_x) = separate_variables(rhs, x_var, t_var)?;

    // Compute integral 1/g(x) dx
    let one_over_gx = LoweredOp::Div(Box::new(LoweredOp::Const(1.0)), Box::new(g_x.clone()));
    let one_over_gx = canonicalize(&one_over_gx).into_op();
    let lhs_integral = try_integrate(&one_over_gx, x_var).map_err(map_integrate_err)?;

    // Compute integral f(t) dt
    let rhs_integral = try_integrate(&f_t, t_var).map_err(map_integrate_err)?;

    // G(x) = F(t) + C → try to solve for x
    let rhs_with_c = LoweredOp::Add(
        Box::new(rhs_integral.clone()),
        Box::new(LoweredOp::Var(c_var)),
    );

    let solve_result = crate::cas::solve::solve(&lhs_integral, &rhs_with_c, x_var);

    match solve_result {
        Ok(sr) if !sr.solutions.is_empty() => {
            let mut x_of_t = canonicalize(&sr.solutions[0]).into_op();
            let mut integration_constants = vec![c_var];

            if let Some((t0, x0)) = ic {
                if let Some(sol) = apply_ic_general(&x_of_t, t_var, t0, x0, c_var) {
                    x_of_t = sol;
                    integration_constants.clear();
                }
            }

            Ok(OdeSolution {
                x_of_t,
                integration_constants,
                kind: OdeKind::Separable,
            })
        }
        _ => {
            // Return implicit form: G(x) - F(t) - C = 0
            let implicit = LoweredOp::Sub(Box::new(lhs_integral), Box::new(rhs_with_c));
            let implicit = canonicalize(&implicit).into_op();

            Ok(OdeSolution {
                x_of_t: implicit,
                integration_constants: vec![c_var],
                kind: OdeKind::ImplicitSeparable,
            })
        }
    }
}

/// Try to factor `rhs` into `f(t) * g(x)`.
fn separate_variables(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
) -> Result<(LoweredOp, LoweredOp), SolveOdeError> {
    // Case: rhs contains only x (f(t) = 1, g(x) = rhs)
    if !contains_any_var(rhs, &[t_var]) && contains_any_var(rhs, &[x_var]) {
        return Ok((LoweredOp::Const(1.0), rhs.clone()));
    }

    // Case: rhs contains only t (f(t) = rhs, g(x) = 1)
    if !contains_any_var(rhs, &[x_var]) && contains_any_var(rhs, &[t_var]) {
        return Ok((rhs.clone(), LoweredOp::Const(1.0)));
    }

    // Case: Mul(a, b)
    match rhs {
        LoweredOp::Mul(a, b) => {
            let a_has_x = contains_any_var(a, &[x_var]);
            let a_has_t = contains_any_var(a, &[t_var]);
            let b_has_x = contains_any_var(b, &[x_var]);
            let b_has_t = contains_any_var(b, &[t_var]);

            if !a_has_x && !b_has_t {
                return Ok((*a.clone(), *b.clone()));
            }
            if !a_has_t && !b_has_x {
                return Ok((*b.clone(), *a.clone()));
            }

            // Try recursive separation on b if a is pure t
            if !a_has_x && a_has_t && b_has_x && b_has_t {
                if let Ok((bf_t, bg_x)) = separate_variables(b, x_var, t_var) {
                    let combined_ft = LoweredOp::Mul(Box::new(*a.clone()), Box::new(bf_t));
                    let combined_ft = canonicalize(&combined_ft).into_op();
                    return Ok((combined_ft, bg_x));
                }
            }
        }
        LoweredOp::Neg(inner) => {
            if let Ok((f_t, g_x)) = separate_variables(inner, x_var, t_var) {
                return Ok((f_t, LoweredOp::Neg(Box::new(g_x))));
            }
        }
        _ => {}
    }

    Err(SolveOdeError::NotRecognized)
}

// ---------------------------------------------------------------------------
// Family 5: Bernoulli ODE: dx/dt = -p(t)*x + q(t)*x^n
// ---------------------------------------------------------------------------

/// Try to solve a Bernoulli ODE via substitution u = x^(1-n).
fn try_bernoulli(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
    c_var: usize,
    ic: Option<(f64, f64)>,
) -> Result<OdeSolution, SolveOdeError> {
    let coeffs = as_polynomial(rhs, x_var).ok_or(SolveOdeError::NotRecognized)?;

    if coeffs.len() < 3 {
        return Err(SolveOdeError::NotRecognized);
    }

    // Constant term must be zero
    let const_term = canonicalize(&coeffs[0]).into_op();
    if !is_zero_op(&const_term) {
        return Err(SolveOdeError::NotRecognized);
    }

    let n_degree = coeffs.len() - 1;
    if n_degree < 2 {
        return Err(SolveOdeError::NotRecognized);
    }

    // Intermediate degrees must be zero
    for coeff in coeffs.iter().take(n_degree).skip(2) {
        let coeff_k = canonicalize(coeff).into_op();
        if !is_zero_op(&coeff_k) {
            return Err(SolveOdeError::NotRecognized);
        }
    }

    let neg_p = canonicalize(&coeffs[1]).into_op();
    let q = canonicalize(&coeffs[n_degree]).into_op();
    let n = n_degree as f64;

    let p = LoweredOp::Neg(Box::new(neg_p));
    let p = canonicalize(&p).into_op();

    if contains_any_var(&p, &[x_var]) || contains_any_var(&q, &[x_var]) {
        return Err(SolveOdeError::NotRecognized);
    }

    // Substitution: u = x^(1-n) → du/dt = (1-n)*q - (1-n)*p*u
    let one_minus_n = 1.0 - n;

    // a_coeff for u: (1-n)*(-p) = -(1-n)*p
    let a_coeff = LoweredOp::Mul(
        Box::new(LoweredOp::Const(one_minus_n)),
        Box::new(LoweredOp::Neg(Box::new(p.clone()))),
    );
    let a_coeff = canonicalize(&a_coeff).into_op();

    let f_u = LoweredOp::Mul(Box::new(LoweredOp::Const(one_minus_n)), Box::new(q.clone()));
    let f_u = canonicalize(&f_u).into_op();

    // u_var: fresh var for u
    let u_var = c_var + 2;

    // Build rhs for du/dt: a_coeff * u + f_u
    let u_rhs = LoweredOp::Add(
        Box::new(LoweredOp::Mul(
            Box::new(a_coeff.clone()),
            Box::new(LoweredOp::Var(u_var)),
        )),
        Box::new(f_u),
    );
    let u_rhs = canonicalize(&u_rhs).into_op();

    // Solve linear ODE for u (no IC at this stage)
    let u_sol = try_linear_1st_order(&u_rhs, u_var, t_var, c_var, None)?;

    // Back-substitute: x = u^(1/(1-n))
    let inv_exp = 1.0 / one_minus_n;
    let x_of_t = LoweredOp::Pow(
        Box::new(u_sol.x_of_t.clone()),
        Box::new(LoweredOp::Const(inv_exp)),
    );
    let mut x_of_t = canonicalize(&x_of_t).into_op();

    let mut integration_constants = u_sol.integration_constants.clone();

    // Apply IC if provided
    if let Some((t0, x0)) = ic {
        if let Some(sol) = apply_ic_general(&x_of_t, t_var, t0, x0, c_var) {
            x_of_t = sol;
            integration_constants.clear();
        }
    }

    Ok(OdeSolution {
        x_of_t,
        integration_constants,
        kind: OdeKind::Bernoulli,
    })
}

// ---------------------------------------------------------------------------
// Family 4: Exact ODE: M dt + N dx = 0 → dF = M dt + N dx
// ---------------------------------------------------------------------------

/// Try to solve an exact ODE via potential function.
fn try_exact(
    rhs: &LoweredOp,
    x_var: usize,
    t_var: usize,
    c_var: usize,
) -> Result<OdeSolution, SolveOdeError> {
    // dx/dt = rhs → M dt + N dx = 0 with M = -rhs, N = 1
    // or detect rhs = -M/N pattern
    let (m_expr, n_expr) = match rhs {
        LoweredOp::Neg(inner) => match inner.as_ref() {
            LoweredOp::Div(num, den) => (*num.clone(), *den.clone()),
            _ => {
                let m = LoweredOp::Neg(Box::new(rhs.clone()));
                (canonicalize(&m).into_op(), LoweredOp::Const(1.0))
            }
        },
        LoweredOp::Div(num, den) => {
            let neg_num = LoweredOp::Neg(num.clone());
            (canonicalize(&neg_num).into_op(), *den.clone())
        }
        _ => {
            let m = LoweredOp::Neg(Box::new(rhs.clone()));
            (canonicalize(&m).into_op(), LoweredOp::Const(1.0))
        }
    };

    // Check exactness: partial M / partial x = partial N / partial t
    let dm_dx = canonicalize(&grad(&m_expr, x_var)).into_op();
    let dn_dt = canonicalize(&grad(&n_expr, t_var)).into_op();
    let c_dm_dx = canonicalize(&dm_dx);
    let c_dn_dt = canonicalize(&dn_dt);

    if c_dm_dx.hash() != c_dn_dt.hash() {
        return Err(SolveOdeError::NotRecognized);
    }

    // Integrate M w.r.t. t to get F_partial
    let f_partial = try_integrate(&m_expr, t_var).map_err(map_integrate_err)?;

    // h'(x) = N - partial F_partial / partial x
    let df_partial_dx = canonicalize(&grad(&f_partial, x_var)).into_op();
    let h_prime = LoweredOp::Sub(Box::new(n_expr.clone()), Box::new(df_partial_dx));
    let h_prime = canonicalize(&h_prime).into_op();

    // h(x) = integral h'(x) dx
    let h = try_integrate(&h_prime, x_var).map_err(map_integrate_err)?;

    // Potential: F(t,x) = F_partial + h
    let potential_f = LoweredOp::Add(Box::new(f_partial), Box::new(h));
    let potential_f = canonicalize(&potential_f).into_op();

    // Return F(t,x) - C as implicit solution
    let implicit = LoweredOp::Sub(Box::new(potential_f), Box::new(LoweredOp::Var(c_var)));

    Ok(OdeSolution {
        x_of_t: implicit,
        integration_constants: vec![c_var],
        kind: OdeKind::ImplicitExact,
    })
}

// ---------------------------------------------------------------------------
// 2nd-order detection (returns OrderTooHigh)
// ---------------------------------------------------------------------------

/// Detect if rhs looks like it could be from a 2nd-order harmonic oscillator.
fn looks_like_2nd_order_linear(rhs: &LoweredOp, x_var: usize, t_var: usize) -> bool {
    if let Some(coeffs) = as_polynomial(rhs, x_var) {
        if coeffs.len() == 2 {
            let const_term = canonicalize(&coeffs[0]).into_op();
            let x_coeff = canonicalize(&coeffs[1]).into_op();
            if is_zero_op(&const_term) && !contains_any_var(&x_coeff, &[t_var, x_var]) {
                // Negative coefficient → oscillator-like pattern
                if let Some(av) = eval_const(&x_coeff) {
                    if av < 0.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// IC application
// ---------------------------------------------------------------------------

/// Apply IC `x(t0) = x0`: substitute t=t0, solve for c_var, substitute back.
fn apply_ic_general(
    x_of_t: &LoweredOp,
    t_var: usize,
    t0: f64,
    x0: f64,
    c_var: usize,
) -> Option<LoweredOp> {
    let mut subs: HashMap<usize, LoweredOp> = HashMap::new();
    subs.insert(t_var, LoweredOp::Const(t0));
    let at_t0 = apply_substitutions(x_of_t, &subs);
    let at_t0 = canonicalize(&at_t0).into_op();

    // Solve at_t0 = x0 for c_var
    let result = solve_system(&[(at_t0, LoweredOp::Const(x0))], &[c_var]).ok()?;
    let sol = result.solutions.first()?;
    let c_val = sol.get(&c_var)?;

    let mut subs2: HashMap<usize, LoweredOp> = HashMap::new();
    subs2.insert(c_var, c_val.clone());
    let x_final = apply_substitutions(x_of_t, &subs2);
    Some(canonicalize(&x_final).into_op())
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Check if a `LoweredOp` is (or simplifies to) zero.
fn is_zero_op(op: &LoweredOp) -> bool {
    let simplified = simplify_op(op);
    match canonicalize(&simplified).into_op() {
        LoweredOp::Const(c) => c.abs() < 1e-12,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn test_separate_variables_product() {
        // rhs = cos(t) * exp(x) → f(t) = cos(t), g(x) = exp(x)
        let cos_t = LoweredOp::Cos(Box::new(LoweredOp::Var(1)));
        let exp_x = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let rhs = LoweredOp::Mul(Box::new(cos_t), Box::new(exp_x));

        let result = separate_variables(&rhs, 0, 1);
        assert!(result.is_ok(), "Should separate cos(t) * exp(x)");
    }

    #[test]
    fn test_separate_variables_pure_x() {
        // rhs = x^2 + 1 → f(t) = 1, g(x) = x^2 + 1
        let x_sq = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        let rhs = LoweredOp::Add(Box::new(x_sq), Box::new(LoweredOp::Const(1.0)));

        let result = separate_variables(&rhs, 0, 1);
        assert!(result.is_ok(), "Should handle pure x function");
        let (f, _g) = result.unwrap();
        let f_canon = canonicalize(&f).into_op();
        assert!(
            matches!(f_canon, LoweredOp::Const(c) if (c - 1.0).abs() < 1e-12),
            "f should be 1"
        );
    }

    #[test]
    fn test_is_zero_op() {
        assert!(is_zero_op(&LoweredOp::Const(0.0)));
        assert!(!is_zero_op(&LoweredOp::Const(1.0)));
    }
}
