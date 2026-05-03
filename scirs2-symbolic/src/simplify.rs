//! Algebraic simplification of symbolic expressions.
//!
//! [`simplify`] performs a single bottom-up pass applying:
//!
//! 1. **Constant folding** — evaluate sub-expressions that are entirely numeric.
//! 2. **Identity rules** — eliminate `0+x`, `x-0`, `1*x`, `x*1`, `x^1`, `1^x`, etc.
//! 3. **Negation cancellation** — `--x` → `x`.
//! 4. **Self-subtraction** — `x - x` → `0` (for syntactically equal sub-trees).
//! 5. **Zero annihilation** — `0*x` → `0`, `x*0` → `0`, `0/x` → `0` (x≠0 statically unknown but safe).
//!
//! The pass is not exhaustive — it does not reorder terms, factor polynomials, or
//! apply trigonometric identities.  For a fully simplified result, call `simplify`
//! repeatedly until the output equals the input.
//!
//! # Example
//! ```
//! use scirs2_symbolic::{Expr, simplify};
//!
//! let x = Expr::var("x");
//! let e = Expr::from(1.0) * x.clone() + Expr::from(0.0);
//! let s = simplify(&e);
//! assert_eq!(s, x);
//! ```

use crate::Expr;

/// Simplify an expression via constant folding and algebraic identity rules.
///
/// The function recurses into every sub-expression first (post-order traversal),
/// then applies local simplification rules at each node.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        // Leaves are already fully simplified.
        Expr::Const(_) | Expr::Var(_) => expr.clone(),

        Expr::Neg(inner) => {
            let s = simplify(inner);
            match s {
                // --x → x
                Expr::Neg(e) => *e,
                // constant folding
                Expr::Const(v) => Expr::Const(-v),
                other => Expr::Neg(Box::new(other)),
            }
        }

        Expr::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                // 0 + b → b
                (Expr::Const(x), _) if *x == 0.0 => b,
                // a + 0 → a
                (_, Expr::Const(y)) if *y == 0.0 => a,
                _ => Expr::Add(Box::new(a), Box::new(b)),
            }
        }

        Expr::Sub(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x - y),
                // a - 0 → a
                (_, Expr::Const(y)) if *y == 0.0 => a,
                // x - x → 0  (syntactic equality)
                _ if a == b => Expr::Const(0.0),
                _ => Expr::Sub(Box::new(a), Box::new(b)),
            }
        }

        Expr::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                // 0 * b → 0
                (Expr::Const(x), _) if *x == 0.0 => Expr::Const(0.0),
                // a * 0 → 0
                (_, Expr::Const(y)) if *y == 0.0 => Expr::Const(0.0),
                // 1 * b → b
                (Expr::Const(x), _) if *x == 1.0 => b,
                // a * 1 → a
                (_, Expr::Const(y)) if *y == 1.0 => a,
                // -1 * b → -b
                (Expr::Const(x), _) if *x == -1.0 => Expr::Neg(Box::new(b)),
                // a * -1 → -a
                (_, Expr::Const(y)) if *y == -1.0 => Expr::Neg(Box::new(a)),
                _ => Expr::Mul(Box::new(a), Box::new(b)),
            }
        }

        Expr::Div(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                // constant folding (non-zero denominator)
                (Expr::Const(x), Expr::Const(y)) if *y != 0.0 => Expr::Const(x / y),
                // 0 / b → 0 (statically — does not check b≠0 at simplification time)
                (Expr::Const(x), _) if *x == 0.0 => Expr::Const(0.0),
                // a / 1 → a
                (_, Expr::Const(y)) if *y == 1.0 => a,
                _ => Expr::Div(Box::new(a), Box::new(b)),
            }
        }

        Expr::Pow(base, exp) => {
            let base = simplify(base);
            let exp = simplify(exp);
            match (&base, &exp) {
                // constant folding
                (Expr::Const(b), Expr::Const(e)) => Expr::Const(b.powf(*e)),
                // x^0 → 1
                (_, Expr::Const(e)) if *e == 0.0 => Expr::Const(1.0),
                // x^1 → x
                (_, Expr::Const(e)) if *e == 1.0 => base,
                // 1^x → 1
                (Expr::Const(b), _) if *b == 1.0 => Expr::Const(1.0),
                // 0^x → 0 (assumes x > 0; at simplification time we don't know)
                (Expr::Const(b), _) if *b == 0.0 => Expr::Const(0.0),
                _ => Expr::Pow(Box::new(base), Box::new(exp)),
            }
        }

        // Transcendental functions — recurse then constant-fold.
        Expr::Sin(e) => {
            let e = simplify(e);
            if let Expr::Const(v) = e {
                Expr::Const(v.sin())
            } else {
                Expr::Sin(Box::new(e))
            }
        }
        Expr::Cos(e) => {
            let e = simplify(e);
            if let Expr::Const(v) = e {
                Expr::Const(v.cos())
            } else {
                Expr::Cos(Box::new(e))
            }
        }
        Expr::Tan(e) => {
            let e = simplify(e);
            if let Expr::Const(v) = e {
                Expr::Const(v.tan())
            } else {
                Expr::Tan(Box::new(e))
            }
        }
        Expr::Exp(e) => {
            let e = simplify(e);
            match e {
                Expr::Const(v) => {
                    if v == 0.0 {
                        // exp(0) → 1
                        Expr::Const(1.0)
                    } else {
                        // constant folding: exp(c)
                        Expr::Const(v.exp())
                    }
                }
                // exp(ln(x)) → x  (inverse functions)
                Expr::Ln(inner) => *inner,
                other => Expr::Exp(Box::new(other)),
            }
        }
        Expr::Ln(e) => {
            let e = simplify(e);
            match e {
                Expr::Const(v) => {
                    if v == 1.0 {
                        // ln(1) → 0
                        Expr::Const(0.0)
                    } else if v > 0.0 {
                        // constant folding for positive arguments
                        Expr::Const(v.ln())
                    } else {
                        Expr::Ln(Box::new(Expr::Const(v)))
                    }
                }
                // ln(exp(x)) → x
                Expr::Exp(inner) => *inner,
                other => Expr::Ln(Box::new(other)),
            }
        }
        Expr::Sqrt(e) => {
            let e = simplify(e);
            match e {
                Expr::Const(v) => {
                    if v == 0.0 {
                        // sqrt(0) → 0
                        Expr::Const(0.0)
                    } else if v == 1.0 {
                        // sqrt(1) → 1
                        Expr::Const(1.0)
                    } else if v >= 0.0 {
                        // constant folding for non-negative values
                        Expr::Const(v.sqrt())
                    } else {
                        Expr::Sqrt(Box::new(Expr::Const(v)))
                    }
                }
                other => Expr::Sqrt(Box::new(other)),
            }
        }
        Expr::Abs(e) => {
            let e = simplify(e);
            if let Expr::Const(v) = e {
                Expr::Const(v.abs())
            } else {
                Expr::Abs(Box::new(e))
            }
        }
    }
}

/// Simplify repeatedly until a fixed point is reached (or at most `max_passes` times).
///
/// Useful when a single [`simplify`] pass is not enough (e.g. nested negations or
/// chains of identity rules that interact).
pub fn simplify_full(expr: &Expr, max_passes: usize) -> Expr {
    let mut current = expr.clone();
    for _ in 0..max_passes {
        let next = simplify(&current);
        if next == current {
            break;
        }
        current = next;
    }
    current
}
