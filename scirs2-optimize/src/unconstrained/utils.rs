//! Common utilities for unconstrained optimization algorithms

use crate::error::OptimizeError;
use ndarray::{Array1, Array2, ArrayView1};

/// Computes finite difference gradient
pub fn finite_difference_gradient<F, S>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    step: f64,
) -> Result<Array1<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();
    let mut grad = Array1::<f64>::zeros(n);
    let mut x_plus = x.to_owned();
    let mut x_minus = x.to_owned();

    for i in 0..n {
        let h = step * (1.0 + x[i].abs());
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;

        let f_plus = fun(&x_plus.view()).into();
        let f_minus = fun(&x_minus.view()).into();

        if !f_plus.is_finite() || !f_minus.is_finite() {
            return Err(OptimizeError::ComputationError(
                "Function returned non-finite value during gradient computation".to_string(),
            ));
        }

        grad[i] = (f_plus - f_minus) / (2.0 * h);

        // Reset for next iteration
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    Ok(grad)
}

/// Computes finite difference Hessian
pub fn finite_difference_hessian<F, S>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();
    let mut hess = Array2::<f64>::zeros((n, n));
    let mut x_temp = x.to_owned();

    let f0 = fun(&x.view()).into();

    for i in 0..n {
        let hi = step * (1.0 + x[i].abs());

        // Diagonal elements
        x_temp[i] = x[i] + hi;
        let fp = fun(&x_temp.view()).into();
        x_temp[i] = x[i] - hi;
        let fm = fun(&x_temp.view()).into();
        x_temp[i] = x[i];

        hess[[i, i]] = (fp - 2.0 * f0 + fm) / (hi * hi);

        // Off-diagonal elements
        for j in (i + 1)..n {
            let hj = step * (1.0 + x[j].abs());

            x_temp[i] = x[i] + hi;
            x_temp[j] = x[j] + hj;
            let fpp = fun(&x_temp.view()).into();

            x_temp[i] = x[i] + hi;
            x_temp[j] = x[j] - hj;
            let fpm = fun(&x_temp.view()).into();

            x_temp[i] = x[i] - hi;
            x_temp[j] = x[j] + hj;
            let fmp = fun(&x_temp.view()).into();

            x_temp[i] = x[i] - hi;
            x_temp[j] = x[j] - hj;
            let fmm = fun(&x_temp.view()).into();

            x_temp[i] = x[i];
            x_temp[j] = x[j];

            let hess_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            hess[[i, j]] = hess_ij;
            hess[[j, i]] = hess_ij;
        }
    }

    Ok(hess)
}

/// Check convergence criteria
pub fn check_convergence(
    f_delta: f64,
    x_delta: f64,
    g_norm: f64,
    ftol: f64,
    xtol: f64,
    gtol: f64,
) -> bool {
    f_delta.abs() < ftol || x_delta < xtol || g_norm < gtol
}

/// Compute the norm of the difference between two arrays
pub fn array_diff_norm(x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
    (x1 - x2).mapv(|x| x.powi(2)).sum().sqrt()
}

/// Clip step size to satisfy bounds
pub fn clip_step(
    x: &ArrayView1<f64>,
    direction: &ArrayView1<f64>,
    alpha: f64,
    lower: &[Option<f64>],
    upper: &[Option<f64>],
) -> f64 {
    let mut clipped_alpha = alpha;

    for i in 0..x.len() {
        if direction[i] != 0.0 {
            // Check lower bound
            if let Some(lb) = lower[i] {
                if direction[i] < 0.0 {
                    let max_step = (lb - x[i]) / direction[i];
                    if max_step >= 0.0 {
                        clipped_alpha = clipped_alpha.min(max_step);
                    }
                }
            }

            // Check upper bound
            if let Some(ub) = upper[i] {
                if direction[i] > 0.0 {
                    let max_step = (ub - x[i]) / direction[i];
                    if max_step >= 0.0 {
                        clipped_alpha = clipped_alpha.min(max_step);
                    }
                }
            }
        }
    }

    clipped_alpha.max(0.0)
}

/// Convert between Array1 and ArrayView1 consistently
pub fn to_array_view<T>(arr: &Array1<T>) -> ArrayView1<T> {
    arr.view()
}

/// Initialize step size for line search
pub fn initial_step_size(grad_norm: f64, max_step: Option<f64>) -> f64 {
    let default_step = if grad_norm > 0.0 {
        1.0 / grad_norm
    } else {
        1.0
    };

    if let Some(max_s) = max_step {
        default_step.min(max_s)
    } else {
        default_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_finite_difference_gradient() {
        let mut quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 2.0 * x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let grad = finite_difference_gradient(&mut quadratic, &x.view(), 1e-8).unwrap();

        assert_abs_diff_eq!(grad[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad[1], 8.0, epsilon = 1e-6);
    }
}
