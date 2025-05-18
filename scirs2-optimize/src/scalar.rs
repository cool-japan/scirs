//! Scalar optimization algorithms
//!
//! This module provides algorithms for minimizing univariate scalar functions.
//! It is similar to `scipy.optimize.minimize_scalar`.

use crate::error::OptimizeError;
use num_traits::Float;
use std::fmt;

/// Methods for scalar optimization
#[derive(Debug, Clone, Copy)]
pub enum Method {
    /// Brent method - combines parabolic interpolation with golden section search
    Brent,
    /// Bounded Brent method - Brent within specified bounds
    Bounded,
    /// Golden section search
    Golden,
}

/// Options for scalar optimization
#[derive(Debug, Clone)]
pub struct Options {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub xatol: f64,
    /// Relative tolerance
    pub xrtol: f64,
    /// Bracket for the search (optional)
    pub bracket: Option<(f64, f64, f64)>,
    /// Display convergence messages
    pub disp: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_iter: 500,
            xatol: 1e-5,
            xrtol: 1.4901161193847656e-8,
            bracket: None,
            disp: false,
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Method::Brent => write!(f, "Brent"),
            Method::Bounded => write!(f, "Bounded"),
            Method::Golden => write!(f, "Golden"),
        }
    }
}

/// Result type for scalar optimization
#[derive(Debug, Clone)]
pub struct ScalarOptimizeResult {
    /// Found minimum
    pub x: f64,
    /// Function value at the minimum
    pub fun: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of function evaluations
    pub function_evals: usize,
    /// Whether the optimization succeeded
    pub success: bool,
    /// Message describing the result
    pub message: String,
}

/// Main minimize scalar function
///
/// Minimization of scalar function of one variable.
///
/// # Arguments
///
/// * `fun` - The objective function to be minimized
/// * `bounds` - Optional bounds as (lower, upper) tuple
/// * `method` - Optimization method to use
/// * `options` - Optional algorithm options
///
/// # Returns
///
/// Returns a `ScalarOptimizeResult` containing the optimization result.
///
/// # Examples
///
/// ```no_run
/// use scirs2_optimize::scalar::{minimize_scalar, Method};
///
/// fn f(x: f64) -> f64 {
///     (x - 2.0) * x * (x + 2.0).powi(2)
/// }
///
/// // Using the Brent method
/// let result = minimize_scalar(f, None, Method::Brent, None)?;
/// println!("Minimum at x = {}", result.x);
/// println!("Function value = {}", result.fun);
///
/// // Using the bounded method
/// let bounds = Some((-3.0, -1.0));
/// let result = minimize_scalar(f, bounds, Method::Bounded, None)?;
/// println!("Bounded minimum at x = {}", result.x);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn minimize_scalar<F>(
    fun: F,
    bounds: Option<(f64, f64)>,
    method: Method,
    options: Option<Options>,
) -> Result<ScalarOptimizeResult, OptimizeError>
where
    F: Fn(f64) -> f64,
{
    let opts = options.unwrap_or_default();

    match method {
        Method::Brent => minimize_scalar_brent(fun, opts),
        Method::Bounded => {
            if let Some((a, b)) = bounds {
                minimize_scalar_bounded(fun, a, b, opts)
            } else {
                Err(OptimizeError::ValueError(
                    "Bounds are required for bounded method".to_string(),
                ))
            }
        }
        Method::Golden => minimize_scalar_golden(fun, opts),
    }
}

/// Brent's method for scalar minimization
fn minimize_scalar_brent<F>(fun: F, options: Options) -> Result<ScalarOptimizeResult, OptimizeError>
where
    F: Fn(f64) -> f64,
{
    // Implementation of Brent's method
    // This combines parabolic interpolation with golden section search

    const GOLDEN: f64 = 0.3819660112501051; // (3 - sqrt(5)) / 2
    const SQRT_EPS: f64 = 1.4901161193847656e-8;

    // Get initial bracket or use default
    let (a, _b, c) = if let Some(bracket) = options.bracket {
        bracket
    } else {
        // Use simple bracketing strategy
        let x0 = 0.0;
        let x1 = 1.0;
        bracket_minimum(&fun, x0, x1)?
    };

    let tol = 3.0 * SQRT_EPS;
    let (mut a, mut b) = if a < c { (a, c) } else { (c, a) };

    // Initialize
    let mut v = a + GOLDEN * (b - a);
    let mut w = v;
    let mut x = v;
    let mut fx = fun(x);
    let mut fv = fx;
    let mut fw = fx;

    let mut d = 0.0;
    let mut e = 0.0;
    let mut iter = 0;
    let mut feval = 1;

    while iter < options.max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + options.xatol;
        let tol2 = 2.0 * tol1;

        // Check for convergence
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(ScalarOptimizeResult {
                x,
                fun: fx,
                iterations: iter,
                function_evals: feval,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Fit parabola
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q_temp = (x - v) * (fx - fw);
            let p_temp = (x - v) * q_temp - (x - w) * r;
            let mut q_val = 2.0 * (q_temp - r);

            let p_val = if q_val > 0.0 {
                q_val = -q_val;
                -p_temp
            } else {
                p_temp
            };

            let etemp = e;
            e = d;

            // Check if parabolic interpolation is acceptable
            if p_val.abs() < (0.5 * q_val * etemp).abs()
                && p_val > q_val * (a - x)
                && p_val < q_val * (b - x)
            {
                d = p_val / q_val;
                let u = x + d;

                // f(x + d) must not be too close to a or b
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if xm > x { tol1 } else { -tol1 };
                }
            } else {
                // Golden section step
                e = if x >= xm { a - x } else { b - x };
                d = GOLDEN * e;
            }
        } else {
            // Golden section step
            e = if x >= xm { a - x } else { b - x };
            d = GOLDEN * e;
        }

        // Evaluate new point
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d > 0.0 { tol1 } else { -tol1 }
        };

        let fu = fun(u);
        feval += 1;

        // Update bracket
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }

            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }

            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        iter += 1;
    }

    Err(OptimizeError::ConvergenceError(
        "Maximum number of iterations reached".to_string(),
    ))
}

/// Bounded Brent method for scalar minimization
fn minimize_scalar_bounded<F>(
    fun: F,
    xmin: f64,
    xmax: f64,
    options: Options,
) -> Result<ScalarOptimizeResult, OptimizeError>
where
    F: Fn(f64) -> f64,
{
    if xmin >= xmax {
        return Err(OptimizeError::ValueError(
            "Lower bound must be less than upper bound".to_string(),
        ));
    }

    // Bounded version of Brent's method
    // Similar to regular Brent but ensures x stays within [xmin, xmax]

    const GOLDEN: f64 = 0.3819660112501051;
    const SQRT_EPS: f64 = 1.4901161193847656e-8;

    let tol = 3.0 * SQRT_EPS;
    let (mut a, mut b) = (xmin, xmax);

    // Initial points
    let mut v = a + GOLDEN * (b - a);
    let mut w = v;
    let mut x = v;
    let mut fx = fun(x);
    let mut fv = fx;
    let mut fw = fx;

    let mut d = 0.0;
    let mut e = 0.0;
    let mut iter = 0;
    let mut feval = 1;

    while iter < options.max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + options.xatol;
        let tol2 = 2.0 * tol1;

        // Check for convergence
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(ScalarOptimizeResult {
                x,
                fun: fx,
                iterations: iter,
                function_evals: feval,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Parabolic interpolation
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q_temp = (x - v) * (fx - fw);
            let p_temp = (x - v) * q_temp - (x - w) * r;
            let mut q_val = 2.0 * (q_temp - r);

            let p_val = if q_val > 0.0 {
                q_val = -q_val;
                -p_temp
            } else {
                p_temp
            };

            let etemp = e;
            e = d;

            if p_val.abs() < (0.5 * q_val * etemp).abs()
                && p_val > q_val * (a - x)
                && p_val < q_val * (b - x)
            {
                d = p_val / q_val;
                let u = x + d;

                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if xm > x { tol1 } else { -tol1 };
                }
            } else {
                e = if x >= xm { a - x } else { b - x };
                d = GOLDEN * e;
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = GOLDEN * e;
        }

        // Make sure we stay within bounds
        let u = (x + if d.abs() >= tol1 {
            d
        } else if d > 0.0 {
            tol1
        } else {
            -tol1
        })
        .max(xmin)
        .min(xmax);

        let fu = fun(u);
        feval += 1;

        // Update variables
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }

            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }

            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        iter += 1;
    }

    Err(OptimizeError::ConvergenceError(
        "Maximum number of iterations reached".to_string(),
    ))
}

/// Golden section search for scalar minimization
fn minimize_scalar_golden<F>(
    fun: F,
    options: Options,
) -> Result<ScalarOptimizeResult, OptimizeError>
where
    F: Fn(f64) -> f64,
{
    const GOLDEN: f64 = 0.6180339887498949; // (sqrt(5) - 1) / 2

    // Get initial bracket or use default
    let (a, _b, c) = if let Some(bracket) = options.bracket {
        bracket
    } else {
        let x0 = 0.0;
        let x1 = 1.0;
        bracket_minimum(&fun, x0, x1)?
    };

    let (mut a, mut b) = if a < c { (a, c) } else { (c, a) };

    // Initialize points
    let mut x1 = a + (1.0 - GOLDEN) * (b - a);
    let mut x2 = a + GOLDEN * (b - a);
    let mut f1 = fun(x1);
    let mut f2 = fun(x2);

    let mut iter = 0;
    let mut feval = 2;

    while iter < options.max_iter {
        if (b - a).abs() < options.xatol {
            let x = 0.5 * (a + b);
            let fx = fun(x);
            feval += 1;

            return Ok(ScalarOptimizeResult {
                x,
                fun: fx,
                iterations: iter,
                function_evals: feval,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1.0 - GOLDEN) * (b - a);
            f1 = fun(x1);
            feval += 1;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + GOLDEN * (b - a);
            f2 = fun(x2);
            feval += 1;
        }

        iter += 1;
    }

    Err(OptimizeError::ConvergenceError(
        "Maximum number of iterations reached".to_string(),
    ))
}

/// Bracket a minimum given two initial points
fn bracket_minimum<F>(fun: &F, xa: f64, xb: f64) -> Result<(f64, f64, f64), OptimizeError>
where
    F: Fn(f64) -> f64,
{
    const GOLDEN_RATIO: f64 = 1.618033988749895;
    const TINY: f64 = 1e-21;
    const MAX_ITER: usize = 50;

    let (mut a, mut b) = (xa, xb);
    let mut fa = fun(a);
    let mut fb = fun(b);

    if fa < fb {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = b + GOLDEN_RATIO * (b - a);
    let mut fc = fun(c);
    let mut iter = 0;

    while fb >= fc {
        let r = (b - a) * (fb - fc);
        let q = (b - c) * (fb - fa);
        let u = b - ((b - c) * q - (b - a) * r) / (2.0 * (q - r).max(TINY).copysign(q - r));
        let ulim = b + 100.0 * (c - b);

        let fu = if (b - u) * (u - c) > 0.0 {
            let fu = fun(u);
            if fu < fc {
                return Ok((b, u, c));
            } else if fu > fb {
                return Ok((a, b, u));
            }
            let u = c + GOLDEN_RATIO * (c - b);
            fun(u)
        } else if (c - u) * (u - ulim) > 0.0 {
            let fu = fun(u);
            if fu < fc {
                b = c;
                fb = fc;
                c = u;
                fc = fu;
                let u = c + GOLDEN_RATIO * (c - b);
                fun(u)
            } else {
                fu
            }
        } else if (u - ulim) * (ulim - c) >= 0.0 {
            let u = ulim;
            fun(u)
        } else {
            let u = c + GOLDEN_RATIO * (c - b);
            fun(u)
        };

        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = u;
        fc = fu;

        iter += 1;
        if iter >= MAX_ITER {
            return Err(OptimizeError::ValueError(
                "Failed to bracket minimum".to_string(),
            ));
        }
    }

    Ok((a, b, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_brent_method() {
        // Test function: (x - 2)^2
        let f = |x: f64| (x - 2.0).powi(2);

        let result = minimize_scalar(f, None, Method::Brent, None).unwrap();
        assert!(result.success);
        assert_abs_diff_eq!(result.x, 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounded_method() {
        // Test function: (x - 2)^2, but constrained to [-1, 1]
        let f = |x: f64| (x - 2.0).powi(2);

        let result = minimize_scalar(f, Some((-1.0, 1.0)), Method::Bounded, None).unwrap();
        assert!(result.success);
        // Allow for some numerical tolerance
        assert!(result.x > 0.99 && result.x <= 1.0);
        assert!(result.fun >= 0.99 && result.fun <= 1.01);
    }

    #[test]
    fn test_golden_method() {
        // Test function: x^4 - 2x^2 + x
        let f = |x: f64| x.powi(4) - 2.0 * x.powi(2) + x;

        let result = minimize_scalar(f, None, Method::Golden, None).unwrap();
        assert!(result.success);
        // The actual minimum depends on the implementation details
        // For the test, we just check it's in a reasonable range
        assert!(result.x > 0.5 && result.x < 1.0);
    }

    #[test]
    fn test_complex_function() {
        // Test with a more complex function
        let f = |x: f64| (x - 2.0) * x * (x + 2.0).powi(2);

        let result = minimize_scalar(f, None, Method::Brent, None).unwrap();
        assert!(result.success);
        // The minimum occurs around x ≈ 1.28
        assert!(result.x > 1.2 && result.x < 1.3);
    }
}
