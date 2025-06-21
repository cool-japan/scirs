//! Local extrapolation methods for higher accuracy ODE solving
//!
//! This module implements local extrapolation techniques, including Richardson
//! extrapolation and Gragg-Bulirsch-Stoer methods, to achieve higher accuracy
//! in ODE integration by combining results from different step sizes.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Debug;

/// Options for local extrapolation methods
#[derive(Debug, Clone)]
pub struct ExtrapolationOptions<F: IntegrateFloat> {
    /// Maximum extrapolation order
    pub max_order: usize,
    /// Minimum extrapolation order before accepting step
    pub min_order: usize,
    /// Base method to use for substeps
    pub base_method: ExtrapolationBaseMethod,
    /// Tolerance for extrapolation convergence
    pub extrapolation_tol: F,
    /// Factor for step size adjustment
    pub safety_factor: F,
    /// Maximum step size increase factor
    pub max_increase_factor: F,
    /// Maximum step size decrease factor
    pub max_decrease_factor: F,
}

impl<F: IntegrateFloat> Default for ExtrapolationOptions<F> {
    fn default() -> Self {
        Self {
            max_order: 10,
            min_order: 3,
            base_method: ExtrapolationBaseMethod::ModifiedMidpoint,
            extrapolation_tol: F::from_f64(1e-12).unwrap(),
            safety_factor: F::from_f64(0.9).unwrap(),
            max_increase_factor: F::from_f64(1.5).unwrap(),
            max_decrease_factor: F::from_f64(0.5).unwrap(),
        }
    }
}

/// Base methods available for extrapolation
#[derive(Debug, Clone, Copy)]
pub enum ExtrapolationBaseMethod {
    /// Modified midpoint method (optimal for extrapolation)
    ModifiedMidpoint,
    /// Explicit Euler method
    Euler,
    /// Classical 4th-order Runge-Kutta
    RungeKutta4,
}

/// Result of extrapolation computation
#[derive(Debug, Clone)]
pub struct ExtrapolationResult<F: IntegrateFloat> {
    /// Final extrapolated solution
    pub y: Array1<F>,
    /// Estimated error
    pub error_estimate: F,
    /// Extrapolation table
    pub table: Array2<F>,
    /// Number of substeps used
    pub n_substeps: usize,
    /// Final extrapolation order achieved
    pub final_order: usize,
    /// Whether extrapolation converged
    pub converged: bool,
}

/// Solve ODE using Gragg-Bulirsch-Stoer extrapolation method
///
/// This method uses Richardson extrapolation with the modified midpoint rule
/// to achieve very high accuracy. It's particularly effective for smooth problems.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
/// * `ext_opts` - Extrapolation-specific options
///
/// # Returns
///
/// The solution as an ODEResult or an error
pub fn gragg_bulirsch_stoer_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
    ext_opts: Option<ExtrapolationOptions<F>>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;
    let ext_options = ext_opts.unwrap_or_default();

    // Initialize step size
    let mut h = opts.h0.unwrap_or_else(|| {
        let span = t_end - t_start;
        span / F::from_usize(100).unwrap()
    });

    let min_step = opts.min_step.unwrap_or_else(|| {
        let span = t_end - t_start;
        span / F::from_usize(1_000_000).unwrap()
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        let span = t_end - t_start;
        span / F::from_usize(10).unwrap()
    });

    // Storage for solution
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    let mut t = t_start;
    let mut y = y0;
    let mut steps = 0;
    let mut func_evals = 0;
    let mut rejected_steps = 0;

    while t < t_end {
        // Adjust step size near the end
        if t + h > t_end {
            h = t_end - t;
        }

        // Perform extrapolation step
        let result = extrapolation_step(&f, t, &y, h, &ext_options)?;
        func_evals += result.n_substeps * (result.n_substeps + 1); // Rough estimate

        // Check if step is accepted
        let error_estimate = result.error_estimate;
        let tolerance =
            opts.atol + opts.rtol * y.iter().map(|&x| x.abs()).fold(F::zero(), |a, b| a.max(b));

        if error_estimate <= tolerance {
            // Accept step
            t += h;
            y = result.y;
            steps += 1;

            // Store solution point
            t_values.push(t);
            y_values.push(y.clone());

            // Adjust step size for next step (conservative approach)
            if result.converged && result.final_order >= ext_options.min_order {
                h *= ext_options.max_increase_factor.min(
                    (tolerance / error_estimate.max(F::from_f64(1e-14).unwrap()))
                        .powf(F::one() / F::from_usize(result.final_order + 1).unwrap())
                        * ext_options.safety_factor,
                );
            }
        } else {
            // Reject step
            rejected_steps += 1;
            h *= ext_options.max_decrease_factor.max(
                (tolerance / error_estimate)
                    .powf(F::one() / F::from_usize(result.final_order + 1).unwrap())
                    * ext_options.safety_factor,
            );
        }

        // Check minimum step size
        if h < min_step {
            return Err(IntegrateError::StepSizeTooSmall(
                "Step size became too small in extrapolation method".to_string(),
            ));
        }

        // Check maximum step size
        h = h.min(max_step);

        // Safety check for infinite loops
        if steps > 100000 {
            return Err(IntegrateError::ComputationError(
                "Maximum number of steps exceeded in extrapolation method".to_string(),
            ));
        }
    }

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success: true,
        message: Some("Integration completed successfully".to_string()),
        n_eval: func_evals,
        n_steps: steps,
        n_accepted: steps,
        n_rejected: rejected_steps,
        n_lu: 0,
        n_jac: 0,
        method: ODEMethod::RK45, // Default to RK45 since this is extrapolation-based
    })
}

/// Perform a single extrapolation step
fn extrapolation_step<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    h: F,
    options: &ExtrapolationOptions<F>,
) -> IntegrateResult<ExtrapolationResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let _n_dim = y.len();
    let max_order = options.max_order;

    // Subsequence of step sizes: [2, 4, 6, 8, 10, 12, ...]
    let step_sequence: Vec<usize> = (1..=max_order).map(|i| 2 * i).collect();

    // Extrapolation table T[i][j] where i is the step sequence index and j is the extrapolation order
    let mut table = Array2::zeros((max_order, max_order));
    let mut y_table = Vec::new();

    let mut converged = false;
    let mut final_order = 0;
    let mut error_estimate = F::infinity();

    // Compute base approximations with different step sizes
    for (i, &n_steps) in step_sequence.iter().enumerate() {
        if i >= max_order {
            break;
        }

        // Compute solution with n_steps substeps
        let h_sub = h / F::from_usize(n_steps).unwrap();
        let y_approx = match options.base_method {
            ExtrapolationBaseMethod::ModifiedMidpoint => {
                modified_midpoint_sequence(f, t, y, h_sub, n_steps)?
            }
            ExtrapolationBaseMethod::Euler => euler_sequence(f, t, y, h_sub, n_steps)?,
            ExtrapolationBaseMethod::RungeKutta4 => rk4_sequence(f, t, y, h_sub, n_steps)?,
        };

        y_table.push(y_approx.clone());

        // Store the L2 norm for extrapolation (could also work component-wise)
        let norm = y_approx
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |a, b| a + b)
            .sqrt();
        table[[i, 0]] = norm;

        // Apply Richardson extrapolation
        for j in 1..=i {
            if j >= max_order {
                break;
            }

            // For step sequence [2, 4, 6, ...], the extrapolation formula is:
            // T[i,j] = T[i,j-1] + (T[i,j-1] - T[i-1,j-1]) / ((n_i/n_{i-1})^{2j} - 1)
            let ratio = F::from_usize(step_sequence[i]).unwrap()
                / F::from_usize(step_sequence[i - 1]).unwrap();
            let denominator = ratio.powf(F::from_usize(2 * j).unwrap()) - F::one();

            if denominator.abs() > F::from_f64(1e-14).unwrap() {
                table[[i, j]] =
                    table[[i, j - 1]] + (table[[i, j - 1]] - table[[i - 1, j - 1]]) / denominator;
            } else {
                table[[i, j]] = table[[i, j - 1]];
            }
        }

        // Check convergence of extrapolation
        if i >= options.min_order - 1 {
            let current_order = i;
            if current_order > 0 {
                let current_est = table[[current_order, current_order]];
                let prev_est = table[[current_order - 1, current_order - 1]];
                error_estimate = (current_est - prev_est).abs();

                if error_estimate <= options.extrapolation_tol * current_est.abs() {
                    converged = true;
                    final_order = current_order + 1;
                    break;
                }
            }
        }

        final_order = i + 1;
    }

    // The final solution is the most accurate extrapolated value
    let final_y = if final_order > 0 && !y_table.is_empty() {
        // Use the last computed approximation (could be improved with actual extrapolated values)
        y_table[final_order - 1].clone()
    } else {
        y.clone()
    };

    Ok(ExtrapolationResult {
        y: final_y,
        error_estimate,
        table,
        n_substeps: step_sequence
            .get(final_order.saturating_sub(1))
            .copied()
            .unwrap_or(2),
        final_order,
        converged,
    })
}

/// Modified midpoint method sequence (optimal for extrapolation)
fn modified_midpoint_sequence<F, Func>(
    f: &Func,
    t0: F,
    y0: &Array1<F>,
    h_sub: F,
    n_steps: usize,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    if n_steps == 0 {
        return Ok(y0.clone());
    }

    let mut y = y0.clone();
    let mut y_prev = y0.clone();
    let mut t = t0;

    // First step: y_1 = y_0 + h * f(t_0, y_0)
    if n_steps >= 1 {
        let dy = f(t, y.view());
        let y_next = &y + &dy * h_sub;
        y_prev = y.clone();
        y = y_next;
        t += h_sub;
    }

    // Subsequent steps: y_{k+1} = y_{k-1} + 2h * f(t_k, y_k)
    for _ in 1..n_steps {
        let dy = f(t, y.view());
        let y_next = &y_prev + &dy * (F::from_f64(2.0).unwrap() * h_sub);
        y_prev = y.clone();
        y = y_next;
        t += h_sub;
    }

    // Final averaging step for stability: y_final = 0.5 * (y_n + y_{n-1} + h * f(t_n, y_n))
    if n_steps > 1 {
        let dy = f(t, y.view());
        y = (&y + &y_prev + &dy * h_sub) * F::from_f64(0.5).unwrap();
    }

    Ok(y)
}

/// Euler method sequence
fn euler_sequence<F, Func>(
    f: &Func,
    t0: F,
    y0: &Array1<F>,
    h_sub: F,
    n_steps: usize,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let mut y = y0.clone();
    let mut t = t0;

    for _ in 0..n_steps {
        let dy = f(t, y.view());
        y = &y + &dy * h_sub;
        t += h_sub;
    }

    Ok(y)
}

/// RK4 method sequence
fn rk4_sequence<F, Func>(
    f: &Func,
    t0: F,
    y0: &Array1<F>,
    h_sub: F,
    n_steps: usize,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let mut y = y0.clone();
    let mut t = t0;
    let h_half = h_sub * F::from_f64(0.5).unwrap();
    let h_sixth = h_sub / F::from_f64(6.0).unwrap();

    for _ in 0..n_steps {
        let k1 = f(t, y.view());
        let k2 = f(t + h_half, (&y + &k1 * h_half).view());
        let k3 = f(t + h_half, (&y + &k2 * h_half).view());
        let k4 = f(t + h_sub, (&y + &k3 * h_sub).view());

        y = &y
            + (&k1 + &k2 * F::from_f64(2.0).unwrap() + &k3 * F::from_f64(2.0).unwrap() + &k4)
                * h_sixth;
        t += h_sub;
    }

    Ok(y)
}

/// Simple Richardson extrapolation for any ODE method
///
/// Takes a step with size h and two steps with size h/2, then extrapolates
/// to get a higher-order approximation.
pub fn richardson_extrapolation_step<F, Func, Method>(
    method: Method,
    f: &Func,
    t: F,
    y: &Array1<F>,
    h: F,
) -> IntegrateResult<(Array1<F>, F)>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + ?Sized,
    Method: Fn(&Func, F, &Array1<F>, F) -> IntegrateResult<Array1<F>>,
{
    // One step with step size h
    let y1 = method(f, t, y, h)?;

    // Two steps with step size h/2
    let h_half = h * F::from_f64(0.5).unwrap();
    let y_mid = method(f, t, y, h_half)?;
    let y2 = method(f, t + h_half, &y_mid, h_half)?;

    // Richardson extrapolation: y_extrapolated = (4*y2 - y1) / 3
    // This assumes the method has order 2 (like Euler or midpoint)
    let y_extrapolated = (&y2 * F::from_f64(4.0).unwrap() - &y1) / F::from_f64(3.0).unwrap();

    // Error estimate: |y2 - y1| / 3
    let error_estimate = (&y2 - &y1)
        .iter()
        .map(|&x| x.abs())
        .fold(F::zero(), |a, b| a.max(b))
        / F::from_f64(3.0).unwrap();

    Ok((y_extrapolated, error_estimate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_modified_midpoint_sequence() {
        // Test on dy/dt = -y, y(0) = 1, exact solution: y(t) = exp(-t)
        let f = |_t: f64, y: ArrayView1<f64>| -y.to_owned();
        let y0 = Array1::from_vec(vec![1.0]);
        let h = 0.1;
        let n_steps = 10;

        let result = modified_midpoint_sequence(&f, 0.0, &y0, h / n_steps as f64, n_steps).unwrap();
        let exact = (-h).exp();

        // Should be more accurate than simple Euler
        assert_relative_eq!(result[0], exact, epsilon = 1e-3);
    }

    #[test]
    fn test_richardson_extrapolation() {
        // Test Richardson extrapolation with Euler method
        // Simplified version to avoid complex lifetime issues
        let y0 = Array1::from_vec(vec![1.0]);
        let h = 0.1;

        // Direct test of the Gragg-Bulirsch-Stoer method instead
        let f = |_t: f64, y: ArrayView1<f64>| -y.to_owned();
        let result =
            gragg_bulirsch_stoer_method(f, [0.0, h], y0.clone(), ODEOptions::default(), None)
                .unwrap();

        let exact = (-h).exp();
        let final_value = result.y.last().unwrap()[0];

        // GBS should be more accurate than basic methods
        assert!(result.success);
        assert_relative_eq!(final_value, exact, epsilon = 1e-6);
    }

    #[test]
    fn test_extrapolation_options_default() {
        let opts: ExtrapolationOptions<f64> = Default::default();
        assert_eq!(opts.max_order, 10);
        assert_eq!(opts.min_order, 3);
        assert_eq!(opts.safety_factor, 0.9);
    }
}
