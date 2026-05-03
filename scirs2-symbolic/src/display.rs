//! Display (infix notation) for symbolic expressions.
//!
//! Implements [`std::fmt::Display`] for [`Expr`], producing human-readable infix notation
//! with parentheses around all binary operations (fully explicit precedence).
//!
//! # Example
//! ```
//! use scirs2_symbolic::Expr;
//!
//! let x = Expr::var("x");
//! let f = x.clone() + Expr::from(1.0);
//! assert_eq!(format!("{f}"), "(x + 1)");
//! ```

use crate::Expr;
use std::fmt;

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(v) => {
                // Print integers without decimal point for readability.
                if v.is_finite() && *v == v.floor() && v.abs() < 1e15 {
                    write!(f, "{}", *v as i64)
                } else {
                    write!(f, "{v}")
                }
            }
            Expr::Var(name) => write!(f, "{name}"),
            Expr::Neg(e) => write!(f, "(-{e})"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Sub(a, b) => write!(f, "({a} - {b})"),
            Expr::Mul(a, b) => write!(f, "({a} * {b})"),
            Expr::Div(a, b) => write!(f, "({a} / {b})"),
            Expr::Pow(base, exp) => write!(f, "({base}^{exp})"),
            Expr::Sin(e) => write!(f, "sin({e})"),
            Expr::Cos(e) => write!(f, "cos({e})"),
            Expr::Tan(e) => write!(f, "tan({e})"),
            Expr::Exp(e) => write!(f, "exp({e})"),
            Expr::Ln(e) => write!(f, "ln({e})"),
            Expr::Sqrt(e) => write!(f, "sqrt({e})"),
            Expr::Abs(e) => write!(f, "|{e}|"),
        }
    }
}
