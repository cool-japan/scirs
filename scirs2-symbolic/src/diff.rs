//! Symbolic differentiation (automatic differentiation via expression trees).
//!
//! [`diff`] computes the symbolic derivative of an [`Expr`] with respect to a named
//! variable, returning a new [`Expr`] that is often not yet in simplified form.
//! Apply [`mod@crate::simplify`] or [`crate::simplify::simplify_full`] to reduce it.
//!
//! # Rules implemented
//!
//! | Expression | Derivative |
//! |------------|-----------|
//! | `c`        | `0`       |
//! | `x`        | `1` (if `x` is the target var, else `0`) |
//! | `-f`       | `-f'`     |
//! | `f + g`    | `f' + g'` |
//! | `f - g`    | `f' - g'` |
//! | `f * g`    | `f'g + fg'` (product rule) |
//! | `f / g`    | `(f'g - fg') / g²` (quotient rule) |
//! | `f^n` (const n) | `n * f^(n-1) * f'` (power rule) |
//! | `f^g` (general) | `f^g * (g' * ln(f) + g * f' / f)` |
//! | `sin(f)`   | `cos(f) * f'` |
//! | `cos(f)`   | `-sin(f) * f'` |
//! | `tan(f)`   | `f' / cos²(f)` |
//! | `exp(f)`   | `exp(f) * f'` |
//! | `ln(f)`    | `f' / f` |
//! | `sqrt(f)`  | `f' / (2 * sqrt(f))` |
//! | `\|f\|`   | `sign(f) * f'` = `(f / \|f\|) * f'` |
//!
//! # Example
//! ```
//! use scirs2_symbolic::{Expr, diff, simplify, eval};
//! use std::collections::HashMap;
//!
//! // d/dx (x² + 3x) = 2x + 3
//! let x = Expr::var("x");
//! let f = x.clone().pow(Expr::from(2.0)) + Expr::from(3.0) * x.clone();
//! let df = simplify(&diff(&f, "x"));
//!
//! let mut vars = HashMap::new();
//! vars.insert("x", 2.0_f64);
//! // At x=2: 2*2 + 3 = 7
//! assert!((eval(&df, &vars).unwrap() - 7.0).abs() < 1e-10);
//! ```

use crate::Expr;

/// Compute the symbolic derivative of `expr` with respect to variable `var`.
///
/// The returned expression may contain redundant operations (e.g. `0 + f`, `1 * g`).
/// Apply [`mod@crate::simplify`] to reduce it.
pub fn diff(expr: &Expr, var: &str) -> Expr {
    match expr {
        // d/dx c = 0
        Expr::Const(_) => Expr::zero(),

        // d/dx x = 1,  d/dx y = 0
        Expr::Var(name) => {
            if name == var {
                Expr::one()
            } else {
                Expr::zero()
            }
        }

        // Chain rule through negation: d/dx (-f) = -(d/dx f)
        Expr::Neg(inner) => -diff(inner, var),

        // Sum rule: (f + g)' = f' + g'
        Expr::Add(a, b) => diff(a, var) + diff(b, var),

        // Difference rule: (f - g)' = f' - g'
        Expr::Sub(a, b) => diff(a, var) - diff(b, var),

        // Product rule: (fg)' = f'g + fg'
        Expr::Mul(a, b) => {
            let da = diff(a, var);
            let db = diff(b, var);
            da * (**b).clone() + (**a).clone() * db
        }

        // Quotient rule: (f/g)' = (f'g - fg') / g²
        Expr::Div(a, b) => {
            let da = diff(a, var);
            let db = diff(b, var);
            let f = (**a).clone();
            let g = (**b).clone();
            let numerator = da * g.clone() - f * db;
            let denominator = g.clone() * g;
            numerator / denominator
        }

        // Power rule (constant exponent): (f^n)' = n * f^(n-1) * f'
        // General power rule: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        Expr::Pow(base, exp) => {
            if let Expr::Const(n) = **exp {
                // Special case: d/dx f(x)^n = n * f(x)^(n-1) * f'(x)
                let new_exp = Expr::Const(n - 1.0);
                Expr::Const(n) * (**base).clone().pow(new_exp) * diff(base, var)
            } else {
                // General: f^g * (g' * ln(f) + g * f' / f)
                let f = (**base).clone();
                let g = (**exp).clone();
                let df = diff(base, var);
                let dg = diff(exp, var);
                let term1 = dg * f.clone().ln();
                let term2 = g * df / f;
                expr.clone() * (term1 + term2)
            }
        }

        // d/dx sin(f) = cos(f) * f'
        Expr::Sin(inner) => (**inner).clone().cos() * diff(inner, var),

        // d/dx cos(f) = -sin(f) * f'
        Expr::Cos(inner) => -((**inner).clone().sin() * diff(inner, var)),

        // d/dx tan(f) = f' / cos²(f)
        Expr::Tan(inner) => {
            let cos_f = (**inner).clone().cos();
            diff(inner, var) / (cos_f.clone() * cos_f)
        }

        // d/dx exp(f) = exp(f) * f'
        Expr::Exp(inner) => expr.clone() * diff(inner, var),

        // d/dx ln(f) = f' / f
        Expr::Ln(inner) => diff(inner, var) / (**inner).clone(),

        // d/dx sqrt(f) = f' / (2 * sqrt(f))
        Expr::Sqrt(inner) => {
            let two = Expr::Const(2.0);
            diff(inner, var) / (two * expr.clone())
        }

        // d/dx |f| = (f / |f|) * f'  (sign function; not defined at f=0)
        Expr::Abs(inner) => {
            let f = (**inner).clone();
            (f / expr.clone()) * diff(inner, var)
        }
    }
}

/// Compute the n-th order derivative by composing `diff` n times.
///
/// The intermediate results are simplified at each step to avoid exponential
/// expression growth.
///
/// # Example
/// ```
/// use scirs2_symbolic::{Expr, diff_n, eval};
/// use std::collections::HashMap;
///
/// // d³/dx³ (x⁴) = 24x
/// let x = Expr::var("x");
/// let f = x.clone().pow(Expr::from(4.0));
/// let d3f = diff_n(&f, "x", 3);
///
/// let mut vars = HashMap::new();
/// vars.insert("x", 2.0_f64);
/// // 24 * 2 = 48
/// assert!((eval(&d3f, &vars).unwrap() - 48.0).abs() < 1e-8);
/// ```
pub fn diff_n(expr: &Expr, var: &str, n: usize) -> Expr {
    if n == 0 {
        return expr.clone();
    }
    let mut current = diff(expr, var);
    for _ in 1..n {
        current = crate::simplify::simplify(&current);
        current = diff(&current, var);
    }
    current
}
