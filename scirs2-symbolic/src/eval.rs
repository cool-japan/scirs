//! Numeric evaluation of symbolic expressions.
//!
//! Given a variable binding map, [`eval`] traverses the expression tree and computes
//! a concrete `f64` value. Unbound variables and domain violations return
//! [`SymbolicError`] rather than panicking.
//!
//! # Example
//! ```
//! use scirs2_symbolic::{Expr, eval};
//! use std::collections::HashMap;
//!
//! let x = Expr::var("x");
//! let f = x.clone() * x.clone(); // x²
//! let mut vars = HashMap::new();
//! vars.insert("x", 3.0_f64);
//! assert_eq!(eval(&f, &vars).unwrap(), 9.0);
//! ```

use crate::{Expr, SymbolicError};
use std::collections::HashMap;

/// Evaluate a symbolic expression numerically given variable bindings.
///
/// # Errors
///
/// - [`SymbolicError::UnboundVariable`] if a `Var` node has no entry in `vars`.
/// - [`SymbolicError::DivisionByZero`] if a denominator evaluates to zero.
/// - [`SymbolicError::DomainError`] for `ln` of non-positive, `sqrt` of negative, etc.
pub fn eval(expr: &Expr, vars: &HashMap<&str, f64>) -> Result<f64, SymbolicError> {
    match expr {
        Expr::Const(v) => Ok(*v),

        Expr::Var(name) => vars
            .get(name.as_str())
            .copied()
            .ok_or_else(|| SymbolicError::UnboundVariable(name.clone())),

        Expr::Neg(e) => Ok(-eval(e, vars)?),

        Expr::Add(a, b) => Ok(eval(a, vars)? + eval(b, vars)?),
        Expr::Sub(a, b) => Ok(eval(a, vars)? - eval(b, vars)?),
        Expr::Mul(a, b) => Ok(eval(a, vars)? * eval(b, vars)?),

        Expr::Div(a, b) => {
            let bv = eval(b, vars)?;
            if bv.abs() < f64::EPSILON {
                return Err(SymbolicError::DivisionByZero);
            }
            Ok(eval(a, vars)? / bv)
        }

        Expr::Pow(base, exp) => {
            let bv = eval(base, vars)?;
            let ev = eval(exp, vars)?;
            Ok(bv.powf(ev))
        }

        Expr::Sin(e) => Ok(eval(e, vars)?.sin()),
        Expr::Cos(e) => Ok(eval(e, vars)?.cos()),
        Expr::Tan(e) => Ok(eval(e, vars)?.tan()),
        Expr::Exp(e) => Ok(eval(e, vars)?.exp()),

        Expr::Ln(e) => {
            let v = eval(e, vars)?;
            if v <= 0.0 {
                return Err(SymbolicError::DomainError(
                    "ln requires a strictly positive argument".to_string(),
                ));
            }
            Ok(v.ln())
        }

        Expr::Sqrt(e) => {
            let v = eval(e, vars)?;
            if v < 0.0 {
                return Err(SymbolicError::DomainError(
                    "sqrt requires a non-negative argument".to_string(),
                ));
            }
            Ok(v.sqrt())
        }

        Expr::Abs(e) => Ok(eval(e, vars)?.abs()),
    }
}
