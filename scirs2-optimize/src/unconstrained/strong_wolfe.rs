//! Enhanced Strong Wolfe conditions line search implementation
//!
//! This module provides robust implementations of line search algorithms that
//! satisfy the Strong Wolfe conditions, which are essential for ensuring
//! convergence of quasi-Newton methods.

use crate::error::OptimizeError;
use crate::unconstrained::utils::clip_step;
use crate::unconstrained::Bounds;
use ndarray::{Array1, ArrayView1};

/// Type alias for zoom search result to reduce type complexity
type ZoomSearchResult = ((f64, f64, Array1<f64>), usize, usize);

/// Strong Wolfe line search options
#[derive(Debug, Clone)]
pub struct StrongWolfeOptions {
    /// Armijo condition parameter (typical: 1e-4)
    pub c1: f64,
    /// Curvature condition parameter (typical: 0.9 for Newton methods, 0.1 for CG)
    pub c2: f64,
    /// Initial step size
    pub initial_step: f64,
    /// Maximum step size
    pub max_step: f64,
    /// Minimum step size
    pub min_step: f64,
    /// Maximum number of function evaluations
    pub max_fev: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Whether to use safeguarded interpolation
    pub use_safeguarded_interpolation: bool,
    /// Whether to use extrapolation in the first phase
    pub use_extrapolation: bool,
}

impl Default for StrongWolfeOptions {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            initial_step: 1.0,
            max_step: 1e10,
            min_step: 1e-12,
            max_fev: 100,
            tolerance: 1e-10,
            use_safeguarded_interpolation: true,
            use_extrapolation: true,
        }
    }
}

/// Result of Strong Wolfe line search
#[derive(Debug, Clone)]
pub struct StrongWolfeResult {
    /// Step size found
    pub alpha: f64,
    /// Function value at the step
    pub f_new: f64,
    /// Gradient at the step
    pub g_new: Array1<f64>,
    /// Number of function evaluations used
    pub nfev: usize,
    /// Number of gradient evaluations used
    pub ngev: usize,
    /// Whether the search was successful
    pub success: bool,
    /// Reason for termination
    pub message: String,
}

/// Enhanced Strong Wolfe line search with robust implementation
#[allow(clippy::too_many_arguments)]
pub fn strong_wolfe_line_search<F, G, S>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    f0: f64,
    g0: &ArrayView1<f64>,
    direction: &ArrayView1<f64>,
    options: &StrongWolfeOptions,
    bounds: Option<&Bounds>,
) -> Result<StrongWolfeResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
    S: Into<f64>,
{
    // Validate inputs
    let derphi0 = g0.dot(direction);
    if derphi0 >= 0.0 {
        return Err(OptimizeError::ValueError(
            "Search direction must be a descent direction".to_string(),
        ));
    }

    if options.c1 <= 0.0 || options.c1 >= options.c2 || options.c2 >= 1.0 {
        return Err(OptimizeError::ValueError(
            "Invalid Wolfe parameters: must have 0 < c1 < c2 < 1".to_string(),
        ));
    }

    let mut alpha = options.initial_step;
    let mut nfev = 0;
    let mut ngev = 0;

    // Apply bounds constraints to initial step
    if let Some(bounds) = bounds {
        alpha = alpha.min(clip_step(x, direction, alpha, &bounds.lower, &bounds.upper));
    }
    alpha = alpha.min(options.max_step).max(options.min_step);

    // Phase 1: Find an interval containing acceptable points
    let (interval_result, fev1, gev1) = find_interval(
        fun, grad_fun, x, f0, derphi0, direction, alpha, options, bounds,
    )?;

    nfev += fev1;
    ngev += gev1;

    match interval_result {
        IntervalResult::Found(alpha, f_alpha, g_alpha) => Ok(StrongWolfeResult {
            alpha,
            f_new: f_alpha,
            g_new: g_alpha,
            nfev,
            ngev,
            success: true,
            message: "Strong Wolfe conditions satisfied in interval search".to_string(),
        }),
        IntervalResult::Bracket(alpha_lo, alpha_hi, f_lo, f_hi, g_lo) => {
            // Phase 2: Zoom to find exact step
            let (zoom_result, fev2, gev2) = zoom_search(
                fun, grad_fun, x, f0, derphi0, direction, alpha_lo, alpha_hi, f_lo, f_hi, g_lo,
                options, bounds,
            )?;

            nfev += fev2;
            ngev += gev2;

            Ok(StrongWolfeResult {
                alpha: zoom_result.0,
                f_new: zoom_result.1,
                g_new: zoom_result.2,
                nfev,
                ngev,
                success: true,
                message: "Strong Wolfe conditions satisfied in zoom phase".to_string(),
            })
        }
        IntervalResult::Failed => Ok(StrongWolfeResult {
            alpha: options.min_step,
            f_new: f0,
            g_new: g0.to_owned(),
            nfev,
            ngev,
            success: false,
            message: "Failed to find acceptable interval".to_string(),
        }),
    }
}

#[derive(Debug)]
enum IntervalResult {
    Found(f64, f64, Array1<f64>),     // alpha, f(alpha), g(alpha)
    Bracket(f64, f64, f64, f64, f64), // alpha_lo, alpha_hi, f_lo, f_hi, derphi_lo
    Failed,
}

/// Phase 1: Find an interval containing acceptable points
#[allow(clippy::too_many_arguments)]
fn find_interval<F, G, S>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    f0: f64,
    derphi0: f64,
    direction: &ArrayView1<f64>,
    mut alpha: f64,
    options: &StrongWolfeOptions,
    bounds: Option<&Bounds>,
) -> Result<(IntervalResult, usize, usize), OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
    S: Into<f64>,
{
    let mut nfev = 0;
    let mut ngev = 0;
    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut derphi_prev = derphi0;

    for i in 0..options.max_fev {
        // Ensure alpha is within bounds
        if let Some(bounds) = bounds {
            alpha = alpha.min(clip_step(x, direction, alpha, &bounds.lower, &bounds.upper));
        }
        alpha = alpha.min(options.max_step).max(options.min_step);

        // Evaluate function at alpha
        let x_alpha = x + alpha * direction;
        let f_alpha = fun(&x_alpha.view()).into();
        nfev += 1;

        // Check Armijo condition and sufficient decrease
        if f_alpha > f0 + options.c1 * alpha * derphi0 || (f_alpha >= f_prev && i > 0) {
            // Found bracket: [alpha_prev, alpha]
            return Ok((
                IntervalResult::Bracket(alpha_prev, alpha, f_prev, f_alpha, derphi_prev),
                nfev,
                ngev,
            ));
        }

        // Evaluate gradient at alpha
        let g_alpha = grad_fun(&x_alpha.view());
        let derphi_alpha = g_alpha.dot(direction);
        ngev += 1;

        // Check curvature condition (Strong Wolfe conditions)
        if derphi_alpha.abs() <= -options.c2 * derphi0 {
            // Found acceptable point
            return Ok((IntervalResult::Found(alpha, f_alpha, g_alpha), nfev, ngev));
        }

        // Check if we've found a bracket due to positive derivative
        if derphi_alpha >= 0.0 {
            return Ok((
                IntervalResult::Bracket(alpha, alpha_prev, f_alpha, f_prev, derphi_alpha),
                nfev,
                ngev,
            ));
        }

        // Update for next iteration
        alpha_prev = alpha;
        f_prev = f_alpha;
        derphi_prev = derphi_alpha;

        // Extrapolate to get next alpha
        if options.use_extrapolation {
            alpha = if i == 0 {
                alpha * 2.0
            } else {
                // Use safer extrapolation based on derivative information
                alpha * (1.0 + 2.0 * derphi_alpha.abs() / derphi0.abs()).min(3.0)
            };
        } else {
            alpha *= 2.0;
        }

        // Safety check: don't go too far
        if alpha > options.max_step {
            alpha = options.max_step;
        }

        // Check for convergence
        if (alpha - alpha_prev).abs() < options.tolerance {
            break;
        }
    }

    Ok((IntervalResult::Failed, nfev, ngev))
}

/// Phase 2: Zoom search within bracket to find exact step
#[allow(clippy::too_many_arguments)]
fn zoom_search<F, G, S>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    f0: f64,
    derphi0: f64,
    direction: &ArrayView1<f64>,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
    mut derphi_lo: f64,
    options: &StrongWolfeOptions,
    _bounds: Option<&Bounds>,
) -> Result<ZoomSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
    S: Into<f64>,
{
    let mut nfev = 0;
    let mut ngev = 0;

    for _ in 0..options.max_fev {
        // Interpolate to find new trial point
        let alpha = if options.use_safeguarded_interpolation {
            safeguarded_interpolation(alpha_lo, alpha_hi, f_lo, f_hi, derphi_lo, derphi0)
        } else {
            0.5 * (alpha_lo + alpha_hi)
        };

        // Evaluate function at trial point
        let x_alpha = x + alpha * direction;
        let f_alpha = fun(&x_alpha.view()).into();
        nfev += 1;

        // Check Armijo condition
        if f_alpha > f0 + options.c1 * alpha * derphi0 || f_alpha >= f_lo {
            // Trial point violates Armijo condition, shrink interval
            alpha_hi = alpha;
            f_hi = f_alpha;
        } else {
            // Trial point satisfies Armijo condition, check curvature
            let g_alpha = grad_fun(&x_alpha.view());
            let derphi_alpha = g_alpha.dot(direction);
            ngev += 1;

            // Check Strong Wolfe conditions
            if derphi_alpha.abs() <= -options.c2 * derphi0 {
                // Found acceptable point
                return Ok(((alpha, f_alpha, g_alpha), nfev, ngev));
            }

            // Update interval based on derivative sign
            if derphi_alpha * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
            }

            alpha_lo = alpha;
            f_lo = f_alpha;
            derphi_lo = derphi_alpha;
        }

        // Check for convergence
        if (alpha_hi - alpha_lo).abs() < options.tolerance {
            break;
        }
    }

    // If we reach here, return the best point found
    let alpha = if f_lo < f_hi { alpha_lo } else { alpha_hi };
    let x_alpha = x + alpha * direction;
    let f_alpha = fun(&x_alpha.view()).into();
    let g_alpha = grad_fun(&x_alpha.view());
    nfev += 1;
    ngev += 1;

    Ok(((alpha, f_alpha, g_alpha), nfev, ngev))
}

/// Safeguarded cubic/quadratic interpolation for zoom phase
fn safeguarded_interpolation(
    alpha_lo: f64,
    alpha_hi: f64,
    f_lo: f64,
    f_hi: f64,
    derphi_lo: f64,
    _derphi0: f64,
) -> f64 {
    let delta = alpha_hi - alpha_lo;

    // Try cubic interpolation first
    let a = (f_hi - f_lo - derphi_lo * delta) / (delta * delta);
    let b = derphi_lo;

    if a.abs() > 1e-10 {
        // Cubic interpolation
        let discriminant = b * b - 3.0 * a * (f_lo - f_hi + derphi_lo * delta);
        if discriminant >= 0.0 {
            let alpha_c = alpha_lo + (-b + discriminant.sqrt()) / (3.0 * a);

            // Safeguard: ensure the interpolated point is within bounds
            let safeguard_lo = alpha_lo + 0.1 * delta;
            let safeguard_hi = alpha_hi - 0.1 * delta;

            if alpha_c >= safeguard_lo && alpha_c <= safeguard_hi {
                return alpha_c;
            }
        }
    }

    // Fallback to quadratic interpolation
    if derphi_lo.abs() > 1e-10 {
        let alpha_q =
            alpha_lo - 0.5 * derphi_lo * delta * delta / (f_hi - f_lo - derphi_lo * delta);
        let safeguard_lo = alpha_lo + 0.1 * delta;
        let safeguard_hi = alpha_hi - 0.1 * delta;

        if alpha_q >= safeguard_lo && alpha_q <= safeguard_hi {
            return alpha_q;
        }
    }

    // Ultimate fallback: bisection
    0.5 * (alpha_lo + alpha_hi)
}

/// Create Strong Wolfe options optimized for specific optimization methods
pub fn create_strong_wolfe_options_for_method(method: &str) -> StrongWolfeOptions {
    match method.to_lowercase().as_str() {
        "bfgs" | "lbfgs" | "sr1" | "dfp" => StrongWolfeOptions {
            c1: 1e-4,
            c2: 0.9,
            initial_step: 1.0,
            max_step: 1e4,
            min_step: 1e-12,
            max_fev: 50,
            tolerance: 1e-10,
            use_safeguarded_interpolation: true,
            use_extrapolation: true,
        },
        "cg" | "conjugate_gradient" => StrongWolfeOptions {
            c1: 1e-4,
            c2: 0.1, // Smaller c2 for CG methods
            initial_step: 1.0,
            max_step: 1e4,
            min_step: 1e-12,
            max_fev: 50,
            tolerance: 1e-10,
            use_safeguarded_interpolation: true,
            use_extrapolation: true,
        },
        "newton" => StrongWolfeOptions {
            c1: 1e-4,
            c2: 0.5, // Moderate c2 for Newton methods
            initial_step: 1.0,
            max_step: 1e6,
            min_step: 1e-15,
            max_fev: 100,
            tolerance: 1e-12,
            use_safeguarded_interpolation: true,
            use_extrapolation: false, // More conservative for Newton
        },
        _ => StrongWolfeOptions::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_strong_wolfe_quadratic() {
        let mut quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let mut grad_quadratic =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

        let x = Array1::from_vec(vec![1.0, 1.0]);
        let f0 = quadratic(&x.view());
        let g0 = grad_quadratic(&x.view());
        let direction = Array1::from_vec(vec![-1.0, -1.0]);

        let options = StrongWolfeOptions::default();
        let result = strong_wolfe_line_search(
            &mut quadratic,
            &mut grad_quadratic,
            &x.view(),
            f0,
            &g0.view(),
            &direction.view(),
            &options,
            None,
        )
        .unwrap();

        assert!(result.success);
        assert!(result.alpha > 0.0);

        // For this quadratic, the exact minimum along the line should be at alpha = 1.0
        assert_abs_diff_eq!(result.alpha, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_strong_wolfe_rosenbrock() {
        let mut rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let mut grad_rosenbrock = |x: &ArrayView1<f64>| -> Array1<f64> {
            let a = 1.0;
            let b = 100.0;
            let grad_x0 = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0].powi(2));
            let grad_x1 = 2.0 * b * (x[1] - x[0].powi(2));
            Array1::from_vec(vec![grad_x0, grad_x1])
        };

        let x = Array1::from_vec(vec![0.0, 0.0]);
        let f0 = rosenbrock(&x.view());
        let g0 = grad_rosenbrock(&x.view());
        let direction = -&g0; // Steepest descent direction

        let options = create_strong_wolfe_options_for_method("bfgs");
        let result = strong_wolfe_line_search(
            &mut rosenbrock,
            &mut grad_rosenbrock,
            &x.view(),
            f0,
            &g0.view(),
            &direction.view(),
            &options,
            None,
        )
        .unwrap();

        assert!(result.success);
        assert!(result.alpha > 0.0);
        assert!(result.f_new < f0); // Should decrease function value
    }

    #[test]
    fn test_safeguarded_interpolation() {
        let alpha_lo = 0.0;
        let alpha_hi = 1.0;
        let f_lo = 1.0;
        let f_hi = 0.5;
        let derphi_lo = -1.0;
        let derphi0 = -1.0;

        let alpha = safeguarded_interpolation(alpha_lo, alpha_hi, f_lo, f_hi, derphi_lo, derphi0);

        // Should be within the safeguarded bounds
        assert!(alpha >= alpha_lo + 0.1 * (alpha_hi - alpha_lo));
        assert!(alpha <= alpha_hi - 0.1 * (alpha_hi - alpha_lo));
    }

    #[test]
    fn test_method_specific_options() {
        let bfgs_opts = create_strong_wolfe_options_for_method("bfgs");
        assert_eq!(bfgs_opts.c2, 0.9);

        let cg_opts = create_strong_wolfe_options_for_method("cg");
        assert_eq!(cg_opts.c2, 0.1);

        let newton_opts = create_strong_wolfe_options_for_method("newton");
        assert_eq!(newton_opts.c2, 0.5);
        assert!(!newton_opts.use_extrapolation);
    }
}
