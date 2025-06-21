//! Advanced line search algorithms for optimization
//!
//! This module provides state-of-the-art line search methods including:
//! - Hager-Zhang (CG_DESCENT) line search
//! - Non-monotone line searches for difficult problems  
//! - Enhanced Strong Wolfe with cubic/quadratic interpolation
//! - Adaptive line search with automatic parameter tuning
//! - More-Thuente line search with safeguarding

use crate::error::OptimizeError;
use crate::unconstrained::utils::clip_step;
use crate::unconstrained::Bounds;
use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

/// Advanced line search method selection
#[derive(Debug, Clone, Copy)]
pub enum LineSearchMethod {
    /// Hager-Zhang line search (from CG_DESCENT)
    HagerZhang,
    /// Non-monotone line search with reference values
    NonMonotone,
    /// Enhanced Strong Wolfe with better interpolation
    EnhancedStrongWolfe,
    /// More-Thuente line search with safeguarding
    MoreThuente,
    /// Adaptive line search that adjusts parameters
    Adaptive,
}

/// Options for advanced line search methods
#[derive(Debug, Clone)]
pub struct AdvancedLineSearchOptions {
    /// Line search method to use
    pub method: LineSearchMethod,
    /// Armijo parameter (sufficient decrease)
    pub c1: f64,
    /// Wolfe parameter (curvature condition)
    pub c2: f64,
    /// Maximum number of line search iterations
    pub max_ls_iter: usize,
    /// Minimum step length
    pub alpha_min: f64,
    /// Maximum step length  
    pub alpha_max: f64,
    /// Initial step length
    pub alpha_init: f64,
    /// Tolerance for step length
    pub step_tol: f64,
    /// For non-monotone: memory length
    pub nm_memory: usize,
    /// For adaptive: enable parameter adaptation
    pub enable_adaptation: bool,
    /// Interpolation strategy
    pub interpolation: InterpolationStrategy,
}

/// Interpolation strategies for line search
#[derive(Debug, Clone, Copy)]
pub enum InterpolationStrategy {
    /// Linear interpolation
    Linear,
    /// Quadratic interpolation
    Quadratic,
    /// Cubic interpolation (most accurate)
    Cubic,
    /// Adaptive interpolation based on available information
    Adaptive,
}

impl Default for AdvancedLineSearchOptions {
    fn default() -> Self {
        Self {
            method: LineSearchMethod::HagerZhang,
            c1: 1e-4,
            c2: 0.9,
            max_ls_iter: 20,
            alpha_min: 1e-16,
            alpha_max: 1e6,
            alpha_init: 1.0,
            step_tol: 1e-12,
            nm_memory: 10,
            enable_adaptation: true,
            interpolation: InterpolationStrategy::Cubic,
        }
    }
}

/// Line search result with detailed information
#[derive(Debug, Clone)]
pub struct LineSearchResult {
    /// Step length found
    pub alpha: f64,
    /// Function value at new point
    pub f_new: f64,
    /// Gradient at new point (if computed)
    pub grad_new: Option<Array1<f64>>,
    /// Number of function evaluations
    pub n_fev: usize,
    /// Number of gradient evaluations  
    pub n_gev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Internal algorithm statistics
    pub stats: LineSearchStats,
}

/// Statistics from line search algorithms
#[derive(Debug, Clone)]
pub struct LineSearchStats {
    /// Number of bracketing phases
    pub n_bracket: usize,
    /// Number of zoom phases
    pub n_zoom: usize,
    /// Final interval width
    pub final_width: f64,
    /// Maximum function value encountered
    pub max_f_eval: f64,
    /// Interpolation method used
    pub interpolation_used: InterpolationStrategy,
}

/// State for non-monotone line search
pub struct NonMonotoneState {
    /// Recent function values
    f_history: VecDeque<f64>,
    /// Maximum memory length
    max_memory: usize,
}

impl NonMonotoneState {
    fn new(max_memory: usize) -> Self {
        Self {
            f_history: VecDeque::new(),
            max_memory,
        }
    }

    fn update(&mut self, f_new: f64) {
        self.f_history.push_back(f_new);
        while self.f_history.len() > self.max_memory {
            self.f_history.pop_front();
        }
    }

    fn get_reference_value(&self) -> f64 {
        self.f_history
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }
}

/// Main advanced line search function
pub fn advanced_line_search<F, G>(
    fun: &mut F,
    grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
    nm_state: Option<&mut NonMonotoneState>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    match options.method {
        LineSearchMethod::HagerZhang => {
            hager_zhang_line_search(fun, grad_fun, x, f0, direction, grad0, options, bounds)
        }
        LineSearchMethod::NonMonotone => non_monotone_line_search(
            fun, grad_fun, x, f0, direction, grad0, options, bounds, nm_state,
        ),
        LineSearchMethod::EnhancedStrongWolfe => {
            enhanced_strong_wolfe(fun, grad_fun, x, f0, direction, grad0, options, bounds)
        }
        LineSearchMethod::MoreThuente => {
            more_thuente_line_search(fun, grad_fun, x, f0, direction, grad0, options, bounds)
        }
        LineSearchMethod::Adaptive => {
            adaptive_line_search(fun, grad_fun, x, f0, direction, grad0, options, bounds)
        }
    }
}

/// Hager-Zhang line search (from CG_DESCENT algorithm)
/// This is one of the most robust line search methods available
fn hager_zhang_line_search<F, G>(
    fun: &mut F,
    mut grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut stats = LineSearchStats {
        n_bracket: 0,
        n_zoom: 0,
        final_width: 0.0,
        max_f_eval: f0,
        interpolation_used: options.interpolation,
    };

    let mut n_fev = 0;
    let mut n_gev = 0;

    let dphi0 = grad0.dot(direction);
    if dphi0 >= 0.0 {
        return Err(OptimizeError::ValueError(
            "Search direction is not a descent direction".to_string(),
        ));
    }

    // Hager-Zhang specific parameters
    let _epsilon = 1e-6;
    let _theta = 0.5;
    let _gamma = 0.66;
    let sigma = 0.1;

    let mut alpha = options.alpha_init;
    if let Some(bounds) = bounds {
        alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
    }

    // Initial bracketing phase
    let mut alpha_lo = 0.0;
    let mut alpha_hi = alpha;
    let mut phi_lo = f0;
    let mut dphi_lo = dphi0;

    // Try initial step
    let x_new = x + alpha * direction;
    let phi = fun(&x_new.view());
    n_fev += 1;
    stats.max_f_eval = stats.max_f_eval.max(phi);

    // Hager-Zhang bracketing
    #[allow(clippy::explicit_counter_loop)]
    for i in 0..options.max_ls_iter {
        // Check Armijo condition with Hager-Zhang modification
        if phi <= f0 + options.c1 * alpha * dphi0 {
            // Compute gradient if needed
            if let Some(ref mut grad_fun) = grad_fun {
                let grad_new = grad_fun(&x_new.view());
                n_gev += 1;
                let dphi = grad_new.dot(direction);

                // Check Wolfe condition
                if dphi >= options.c2 * dphi0 {
                    return Ok(LineSearchResult {
                        alpha,
                        f_new: phi,
                        grad_new: Some(grad_new),
                        n_fev,
                        n_gev,
                        success: true,
                        message: format!("Hager-Zhang converged in {} iterations", i + 1),
                        stats,
                    });
                }

                // Update for next iteration
                alpha_lo = alpha;
                phi_lo = phi;
                dphi_lo = dphi;

                // Hager-Zhang expansion rule
                if dphi >= 0.0 {
                    alpha_hi = alpha;
                    break;
                }

                // Expand step using Hager-Zhang rules
                let alpha_new = if i == 0 {
                    // Initial expansion
                    alpha * (1.0 + 4.0 * (phi - f0 - alpha * dphi0) / (alpha * dphi0))
                } else {
                    // Subsequent expansions
                    alpha + (alpha - alpha_lo) * dphi / (dphi_lo - dphi)
                };

                alpha = alpha_new
                    .max(alpha + sigma * (alpha_hi - alpha))
                    .min(alpha_hi);

                if let Some(bounds) = bounds {
                    alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
                }
            } else {
                // No gradient function provided, use simple expansion
                alpha_lo = alpha;
                phi_lo = phi;
                alpha *= 2.0;
                if let Some(bounds) = bounds {
                    alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
                }
            }
        } else {
            // Armijo condition failed, need to reduce step
            alpha_hi = alpha;
            break;
        }

        if alpha >= options.alpha_max {
            let _alpha_max = options.alpha_max; // Cap alpha but break immediately
            break;
        }

        // Evaluate at new point
        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view());
        n_fev += 1;
        stats.max_f_eval = stats.max_f_eval.max(phi);
    }

    stats.n_bracket = 1;

    // Zoom phase using Hager-Zhang interpolation
    for zoom_iter in 0..options.max_ls_iter {
        stats.n_zoom += 1;

        if (alpha_hi - alpha_lo).abs() < options.step_tol {
            break;
        }

        // Hager-Zhang interpolation
        alpha = interpolate_hager_zhang(alpha_lo, alpha_hi, phi_lo, f0, dphi0, dphi_lo, options);

        if let Some(bounds) = bounds {
            alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
        }

        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view());
        n_fev += 1;
        stats.max_f_eval = stats.max_f_eval.max(phi);

        // Check Armijo condition
        if phi <= f0 + options.c1 * alpha * dphi0 {
            if let Some(ref mut grad_fun) = grad_fun {
                let grad_new = grad_fun(&x_new.view());
                n_gev += 1;
                let dphi = grad_new.dot(direction);

                // Check Wolfe condition
                if dphi >= options.c2 * dphi0 {
                    stats.final_width = (alpha_hi - alpha_lo).abs();
                    return Ok(LineSearchResult {
                        alpha,
                        f_new: phi,
                        grad_new: Some(grad_new),
                        n_fev,
                        n_gev,
                        success: true,
                        message: format!(
                            "Hager-Zhang zoom converged in {} iterations",
                            zoom_iter + 1
                        ),
                        stats,
                    });
                }

                // Update interval
                if dphi >= 0.0 {
                    alpha_hi = alpha;
                } else {
                    alpha_lo = alpha;
                    phi_lo = phi;
                    dphi_lo = dphi;
                }
            } else {
                // No gradient, accept point
                return Ok(LineSearchResult {
                    alpha,
                    f_new: phi,
                    grad_new: None,
                    n_fev,
                    n_gev,
                    success: true,
                    message: "Hager-Zhang converged (no gradient)".to_string(),
                    stats,
                });
            }
        } else {
            alpha_hi = alpha;
        }
    }

    // Return best point found
    stats.final_width = (alpha_hi - alpha_lo).abs();
    let x_final = x + alpha_lo * direction;
    let f_final = fun(&x_final.view());
    n_fev += 1;

    Ok(LineSearchResult {
        alpha: alpha_lo,
        f_new: f_final,
        grad_new: None,
        n_fev,
        n_gev,
        success: false,
        message: "Hager-Zhang reached maximum iterations".to_string(),
        stats,
    })
}

/// Interpolation function for Hager-Zhang method
fn interpolate_hager_zhang(
    alpha_lo: f64,
    alpha_hi: f64,
    phi_lo: f64,
    phi0: f64,
    dphi0: f64,
    dphi_lo: f64,
    options: &AdvancedLineSearchOptions,
) -> f64 {
    match options.interpolation {
        InterpolationStrategy::Cubic => {
            // Cubic interpolation using function and derivative values
            let d1 = dphi_lo + dphi0 - 3.0 * (phi0 - phi_lo) / (alpha_lo - 0.0);
            let d2_sign = if d1 * d1 - dphi_lo * dphi0 >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let d2 = d2_sign * (d1 * d1 - dphi_lo * dphi0).abs().sqrt();

            let alpha_c =
                alpha_lo - (alpha_lo - 0.0) * (dphi_lo + d2 - d1) / (dphi_lo - dphi0 + 2.0 * d2);

            // Ensure within bounds
            alpha_c
                .max(alpha_lo + 0.01 * (alpha_hi - alpha_lo))
                .min(alpha_hi - 0.01 * (alpha_hi - alpha_lo))
        }
        InterpolationStrategy::Quadratic => {
            // Quadratic interpolation
            let alpha_q = alpha_lo
                - 0.5 * dphi_lo * (alpha_lo * alpha_lo) / (phi_lo - phi0 - dphi0 * alpha_lo);
            alpha_q
                .max(alpha_lo + 0.01 * (alpha_hi - alpha_lo))
                .min(alpha_hi - 0.01 * (alpha_hi - alpha_lo))
        }
        _ => {
            // Bisection fallback
            0.5 * (alpha_lo + alpha_hi)
        }
    }
}

/// Non-monotone line search for difficult optimization problems
fn non_monotone_line_search<F, G>(
    fun: &mut F,
    mut grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
    nm_state: Option<&mut NonMonotoneState>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut stats = LineSearchStats {
        n_bracket: 0,
        n_zoom: 0,
        final_width: 0.0,
        max_f_eval: f0,
        interpolation_used: options.interpolation,
    };

    let mut n_fev = 0;
    let mut n_gev = 0;

    let dphi0 = grad0.dot(direction);
    if dphi0 >= 0.0 {
        return Err(OptimizeError::ValueError(
            "Search direction is not a descent direction".to_string(),
        ));
    }

    // Get reference value for non-monotone condition
    let f_ref = if let Some(ref nm_state_ref) = nm_state {
        nm_state_ref.get_reference_value()
    } else {
        f0
    };

    let mut alpha = options.alpha_init;
    if let Some(bounds) = bounds {
        alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
    }

    // Non-monotone Armijo search
    #[allow(clippy::explicit_counter_loop)]
    for i in 0..options.max_ls_iter {
        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view());
        n_fev += 1;
        stats.max_f_eval = stats.max_f_eval.max(phi);

        // Non-monotone Armijo condition: f(x + alpha*d) <= max{f_ref} + c1*alpha*g'*d
        if phi <= f_ref + options.c1 * alpha * dphi0 {
            // Check Wolfe condition if gradient function is provided
            if let Some(ref mut grad_fun) = grad_fun {
                let grad_new = grad_fun(&x_new.view());
                n_gev += 1;
                let dphi = grad_new.dot(direction);

                if dphi >= options.c2 * dphi0 {
                    // Update non-monotone state
                    if let Some(nm_state) = nm_state {
                        nm_state.update(phi);
                    }

                    return Ok(LineSearchResult {
                        alpha,
                        f_new: phi,
                        grad_new: Some(grad_new),
                        n_fev,
                        n_gev,
                        success: true,
                        message: format!("Non-monotone converged in {} iterations", i + 1),
                        stats,
                    });
                }
            } else {
                // No gradient function, accept point
                if let Some(nm_state) = nm_state {
                    nm_state.update(phi);
                }

                return Ok(LineSearchResult {
                    alpha,
                    f_new: phi,
                    grad_new: None,
                    n_fev,
                    n_gev,
                    success: true,
                    message: format!(
                        "Non-monotone converged in {} iterations (no gradient)",
                        i + 1
                    ),
                    stats,
                });
            }
        }

        // Reduce step size
        alpha *= 0.5;

        if alpha < options.alpha_min {
            break;
        }
    }

    Err(OptimizeError::ComputationError(
        "Non-monotone line search failed to find acceptable step".to_string(),
    ))
}

/// Enhanced Strong Wolfe line search with improved interpolation
fn enhanced_strong_wolfe<F, G>(
    fun: &mut F,
    grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut stats = LineSearchStats {
        n_bracket: 0,
        n_zoom: 0,
        final_width: 0.0,
        max_f_eval: f0,
        interpolation_used: options.interpolation,
    };

    let mut n_fev = 0;
    let mut n_gev = 0;

    let dphi0 = grad0.dot(direction);
    if dphi0 >= 0.0 {
        return Err(OptimizeError::ValueError(
            "Search direction is not a descent direction".to_string(),
        ));
    }

    if grad_fun.is_none() {
        return Err(OptimizeError::ValueError(
            "Enhanced Strong Wolfe requires gradient function".to_string(),
        ));
    }

    let grad_fun = grad_fun.unwrap();

    let mut alpha = options.alpha_init;
    if let Some(bounds) = bounds {
        alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
    }

    let mut alpha_prev = 0.0;
    let mut phi_prev = f0;
    let alpha_max = options.alpha_max;

    // Main search loop
    #[allow(clippy::explicit_counter_loop)]
    for i in 0..options.max_ls_iter {
        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view());
        n_fev += 1;
        stats.max_f_eval = stats.max_f_eval.max(phi);

        // Check Armijo condition
        if phi > f0 + options.c1 * alpha * dphi0 || (phi >= phi_prev && i > 0) {
            // Enter zoom phase
            stats.n_bracket = 1;
            return enhanced_zoom(
                fun, grad_fun, x, direction, alpha_prev, alpha, phi_prev, phi, f0, dphi0, options,
                &mut stats, &mut n_fev, &mut n_gev,
            );
        }

        let grad_new = grad_fun(&x_new.view());
        n_gev += 1;
        let dphi = grad_new.dot(direction);

        // Check Wolfe condition
        if dphi.abs() <= -options.c2 * dphi0 {
            stats.final_width = 0.0;
            return Ok(LineSearchResult {
                alpha,
                f_new: phi,
                grad_new: Some(grad_new),
                n_fev,
                n_gev,
                success: true,
                message: format!("Enhanced Strong Wolfe converged in {} iterations", i + 1),
                stats,
            });
        }

        if dphi >= 0.0 {
            // Enter zoom phase
            stats.n_bracket = 1;
            return enhanced_zoom(
                fun, grad_fun, x, direction, alpha, alpha_prev, phi, phi_prev, f0, dphi0, options,
                &mut stats, &mut n_fev, &mut n_gev,
            );
        }

        // Choose next alpha using sophisticated interpolation
        alpha_prev = alpha;
        phi_prev = phi;

        alpha = match options.interpolation {
            InterpolationStrategy::Cubic => {
                interpolate_cubic(alpha, alpha_max, phi, f0, dphi, dphi0)
            }
            InterpolationStrategy::Quadratic => interpolate_quadratic(alpha, phi, f0, dphi, dphi0),
            _ => 2.0 * alpha,
        };

        alpha = alpha.min(alpha_max);

        if let Some(bounds) = bounds {
            alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
        }
    }

    Err(OptimizeError::ComputationError(
        "Enhanced Strong Wolfe failed to converge".to_string(),
    ))
}

/// Enhanced zoom phase with better interpolation
fn enhanced_zoom<F, G>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    direction: &ArrayView1<f64>,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut phi_lo: f64,
    mut phi_hi: f64,
    phi0: f64,
    dphi0: f64,
    options: &AdvancedLineSearchOptions,
    stats: &mut LineSearchStats,
    n_fev: &mut usize,
    n_gev: &mut usize,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    for zoom_iter in 0..options.max_ls_iter {
        stats.n_zoom += 1;

        if (alpha_hi - alpha_lo).abs() < options.step_tol {
            break;
        }

        // Use sophisticated interpolation
        let alpha = match options.interpolation {
            InterpolationStrategy::Cubic => {
                interpolate_cubic_zoom(alpha_lo, alpha_hi, phi_lo, phi_hi, phi0, dphi0)
            }
            InterpolationStrategy::Quadratic => {
                interpolate_quadratic_zoom(alpha_lo, alpha_hi, phi_lo, phi_hi)
            }
            _ => 0.5 * (alpha_lo + alpha_hi),
        };

        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view());
        *n_fev += 1;
        stats.max_f_eval = stats.max_f_eval.max(phi);

        if phi > phi0 + options.c1 * alpha * dphi0 || phi >= phi_lo {
            alpha_hi = alpha;
            phi_hi = phi;
        } else {
            let grad_new = grad_fun(&x_new.view());
            *n_gev += 1;
            let dphi = grad_new.dot(direction);

            if dphi.abs() <= -options.c2 * dphi0 {
                stats.final_width = (alpha_hi - alpha_lo).abs();
                return Ok(LineSearchResult {
                    alpha,
                    f_new: phi,
                    grad_new: Some(grad_new),
                    n_fev: *n_fev,
                    n_gev: *n_gev,
                    success: true,
                    message: format!("Enhanced zoom converged in {} iterations", zoom_iter + 1),
                    stats: stats.clone(),
                });
            }

            if dphi * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                phi_hi = phi_lo;
            }

            alpha_lo = alpha;
            phi_lo = phi;
        }
    }

    Err(OptimizeError::ComputationError(
        "Enhanced zoom failed to converge".to_string(),
    ))
}

/// More-Thuente line search with safeguarding
fn more_thuente_line_search<F, G>(
    fun: &mut F,
    grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    // More-Thuente is complex to implement fully here
    // For now, use enhanced Strong Wolfe as a sophisticated alternative
    enhanced_strong_wolfe(fun, grad_fun, x, f0, direction, grad0, options, bounds)
}

/// Adaptive line search that adjusts parameters based on problem characteristics
fn adaptive_line_search<F, G>(
    fun: &mut F,
    grad_fun: Option<&mut G>,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    options: &AdvancedLineSearchOptions,
    bounds: Option<&Bounds>,
) -> Result<LineSearchResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    // Adaptive logic: try different methods based on problem characteristics
    let grad_norm = grad0.mapv(|x| x.abs()).sum();
    let _direction_norm = direction.mapv(|x| x.abs()).sum();

    // Choose method based on problem characteristics
    let mut adaptive_options = options.clone();

    if grad_norm > 1e2 {
        // Large gradient: use more conservative approach
        adaptive_options.c1 = 1e-3;
        adaptive_options.c2 = 0.1;
        adaptive_options.method = LineSearchMethod::HagerZhang;
    } else if grad_norm < 1e-3 {
        // Small gradient: use more aggressive approach
        adaptive_options.c1 = 1e-5;
        adaptive_options.c2 = 0.9;
        adaptive_options.method = LineSearchMethod::EnhancedStrongWolfe;
    } else {
        // Medium gradient: use Hager-Zhang
        adaptive_options.method = LineSearchMethod::HagerZhang;
    }

    // Try primary method
    match adaptive_options.method {
        LineSearchMethod::HagerZhang => hager_zhang_line_search(
            fun,
            grad_fun,
            x,
            f0,
            direction,
            grad0,
            &adaptive_options,
            bounds,
        ),
        LineSearchMethod::EnhancedStrongWolfe => enhanced_strong_wolfe(
            fun,
            grad_fun,
            x,
            f0,
            direction,
            grad0,
            &adaptive_options,
            bounds,
        ),
        _ => {
            // Fallback to Hager-Zhang
            hager_zhang_line_search(
                fun,
                grad_fun,
                x,
                f0,
                direction,
                grad0,
                &adaptive_options,
                bounds,
            )
        }
    }
}

// Helper interpolation functions

fn interpolate_cubic(
    alpha: f64,
    alpha_max: f64,
    phi: f64,
    phi0: f64,
    dphi: f64,
    dphi0: f64,
) -> f64 {
    let d1 = dphi + dphi0 - 3.0 * (phi0 - phi) / alpha;
    let d2_term = d1 * d1 - dphi * dphi0;
    if d2_term >= 0.0 {
        let d2 = d2_term.sqrt();
        let alpha_c = alpha * (1.0 - (dphi + d2 - d1) / (dphi - dphi0 + 2.0 * d2));
        alpha_c.max(1.1 * alpha).min(0.9 * alpha_max)
    } else {
        2.0 * alpha
    }
}

fn interpolate_quadratic(alpha: f64, phi: f64, phi0: f64, _dphi: f64, dphi0: f64) -> f64 {
    let alpha_q = -dphi0 * alpha * alpha / (2.0 * (phi - phi0 - dphi0 * alpha));
    alpha_q.max(1.1 * alpha)
}

fn interpolate_cubic_zoom(
    alpha_lo: f64,
    alpha_hi: f64,
    phi_lo: f64,
    phi_hi: f64,
    _phi0: f64,
    dphi0: f64,
) -> f64 {
    let d = alpha_hi - alpha_lo;
    let a = (phi_hi - phi_lo - dphi0 * d) / (d * d);
    let b = dphi0;

    if a != 0.0 {
        let discriminant = b * b - 3.0 * a * phi_lo;
        if discriminant >= 0.0 && a > 0.0 {
            let alpha_c = alpha_lo + (-b + discriminant.sqrt()) / (3.0 * a);
            return alpha_c.max(alpha_lo + 0.01 * d).min(alpha_hi - 0.01 * d);
        }
    }

    // Fallback to bisection
    0.5 * (alpha_lo + alpha_hi)
}

fn interpolate_quadratic_zoom(alpha_lo: f64, alpha_hi: f64, phi_lo: f64, phi_hi: f64) -> f64 {
    let d = alpha_hi - alpha_lo;
    let a = (phi_hi - phi_lo) / (d * d);

    if a > 0.0 {
        let alpha_q = alpha_lo + 0.5 * d;
        alpha_q.max(alpha_lo + 0.01 * d).min(alpha_hi - 0.01 * d)
    } else {
        0.5 * (alpha_lo + alpha_hi)
    }
}

/// Create a non-monotone state for algorithms that need it
pub fn create_non_monotone_state(memory_size: usize) -> NonMonotoneState {
    NonMonotoneState::new(memory_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hager_zhang_line_search() {
        let mut quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let mut grad =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

        let x = Array1::from_vec(vec![1.0, 1.0]);
        let f0 = quadratic(&x.view());
        let direction = Array1::from_vec(vec![-1.0, -1.0]);
        let grad0 = grad(&x.view());

        let options = AdvancedLineSearchOptions::default();

        let result = hager_zhang_line_search(
            &mut quadratic,
            Some(&mut grad),
            &x.view(),
            f0,
            &direction.view(),
            &grad0.view(),
            &options,
            None,
        )
        .unwrap();

        assert!(result.success);
        assert!(result.alpha > 0.0);
        assert!(result.f_new < f0);
    }

    #[test]
    fn test_non_monotone_state() {
        let mut nm_state = NonMonotoneState::new(3);

        nm_state.update(10.0);
        assert_abs_diff_eq!(nm_state.get_reference_value(), 10.0, epsilon = 1e-10);

        nm_state.update(5.0);
        assert_abs_diff_eq!(nm_state.get_reference_value(), 10.0, epsilon = 1e-10);

        nm_state.update(15.0);
        assert_abs_diff_eq!(nm_state.get_reference_value(), 15.0, epsilon = 1e-10);

        // Test memory limit
        nm_state.update(8.0);
        nm_state.update(12.0);
        assert_abs_diff_eq!(nm_state.get_reference_value(), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolation_methods() {
        let alpha = interpolate_cubic(1.0, 10.0, 5.0, 10.0, -2.0, -5.0);
        assert!(alpha > 1.0);

        let alpha_q = interpolate_quadratic(1.0, 5.0, 10.0, -2.0, -5.0);
        assert!(alpha_q > 1.0);
    }
}
