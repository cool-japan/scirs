//! Line search algorithms for optimization

use crate::unconstrained::utils::clip_step;
use crate::unconstrained::Bounds;
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Backtracking line search with Armijo condition
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn backtracking_line_search<F, S>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad: &ArrayView1<f64>,
    alpha_init: f64,
    c1: f64,
    rho: f64,
    bounds: Option<&Bounds>,
) -> (f64, f64)
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let mut alpha = alpha_init;
    let slope = grad.dot(direction);

    // Handle bounds constraints
    if let Some(bounds) = bounds {
        alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
    }

    // Backtracking loop
    for _ in 0..50 {
        let x_new = x + alpha * direction;
        let f_new = fun(&x_new.view()).into();

        if f_new <= f0 + c1 * alpha * slope {
            return (alpha, f_new);
        }

        alpha *= rho;

        if alpha < 1e-16 {
            break;
        }
    }

    (alpha, f0)
}

/// Strong Wolfe line search
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn strong_wolfe_line_search<F, S, G>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    f0: f64,
    direction: &ArrayView1<f64>,
    grad0: &ArrayView1<f64>,
    alpha_init: f64,
    c1: f64,
    c2: f64,
    bounds: Option<&Bounds>,
) -> Result<(f64, f64, Array1<f64>), &'static str>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut alpha = alpha_init;
    let phi0 = f0;
    let dphi0 = grad0.dot(direction);

    if dphi0 >= 0.0 {
        return Err("Search direction must be a descent direction");
    }

    // Handle bounds constraints
    if let Some(bounds) = bounds {
        alpha = clip_step(x, direction, alpha, &bounds.lower, &bounds.upper);
    }

    let mut alpha_lo = 0.0;
    let alpha_hi = alpha;
    let mut phi_lo = phi0;
    let mut dphi_lo = dphi0;

    for _ in 0..20 {
        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view()).into();

        if phi > phi0 + c1 * alpha * dphi0 || (phi >= phi_lo && alpha_lo > 0.0) {
            return zoom(
                fun, grad_fun, x, direction, alpha_lo, alpha, phi_lo, phi, dphi_lo, phi0, dphi0,
                c1, c2,
            );
        }

        let grad_new = grad_fun(&x_new.view());
        let dphi = grad_new.dot(direction);

        if dphi.abs() <= -c2 * dphi0 {
            return Ok((alpha, phi, grad_new));
        }

        if dphi >= 0.0 {
            return zoom(
                fun, grad_fun, x, direction, alpha, alpha_lo, phi, phi_lo, dphi, phi0, dphi0, c1,
                c2,
            );
        }

        alpha_lo = alpha;
        phi_lo = phi;
        dphi_lo = dphi;

        alpha = 0.5 * (alpha + alpha_hi);
    }

    Err("Line search failed to find a step satisfying the strong Wolfe conditions")
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn zoom<F, S, G>(
    fun: &mut F,
    grad_fun: &mut G,
    x: &ArrayView1<f64>,
    direction: &ArrayView1<f64>,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut phi_lo: f64,
    mut _phi_hi: f64,
    mut _dphi_lo: f64,
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> Result<(f64, f64, Array1<f64>), &'static str>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    for _ in 0..10 {
        let alpha = 0.5 * (alpha_lo + alpha_hi);
        let x_new = x + alpha * direction;
        let phi = fun(&x_new.view()).into();

        if phi > phi0 + c1 * alpha * dphi0 || phi >= phi_lo {
            alpha_hi = alpha;
            _phi_hi = phi;
        } else {
            let grad_new = grad_fun(&x_new.view());
            let dphi = grad_new.dot(direction);

            if dphi.abs() <= -c2 * dphi0 {
                return Ok((alpha, phi, grad_new));
            }

            if dphi * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                _phi_hi = phi_lo;
            }

            alpha_lo = alpha;
            phi_lo = phi;
            _dphi_lo = dphi;
        }

        if (alpha_hi - alpha_lo).abs() < 1e-8 {
            break;
        }
    }

    Err("Zoom failed to find acceptable step")
}

/// Simple bracketing line search
#[allow(dead_code)]
pub fn bracketing_line_search<F, S>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    direction: &ArrayView1<f64>,
    bounds: Option<&Bounds>,
) -> f64
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let mut a = 0.0;
    let mut b = 1.0;

    // Handle bounds
    if let Some(bounds) = bounds {
        b = clip_step(x, direction, b, &bounds.lower, &bounds.upper);
    }

    let fa = fun(&x.view()).into();
    let mut x_new = x + b * direction;
    let fb = fun(&x_new.view()).into();

    // If b is better than a, expand the interval
    if fb < fa {
        while b < 10.0 {
            let c = 2.0 * b;
            x_new = x + c * direction;
            let fc = fun(&x_new.view()).into();

            if fc >= fb {
                break;
            }

            a = b;
            b = c;

            if let Some(bounds) = bounds {
                let max_step = clip_step(x, direction, b, &bounds.lower, &bounds.upper);
                if max_step <= b {
                    b = max_step;
                    break;
                }
            }
        }
    }

    // Binary search for the minimum
    for _ in 0..20 {
        let mid = 0.5 * (a + b);
        x_new = x + mid * direction;
        let fmid = fun(&x_new.view()).into();

        x_new = x + (mid - 0.01) * direction;
        let fleft = fun(&x_new.view()).into();

        if fleft < fmid {
            b = mid;
        } else {
            a = mid;
        }

        if (b - a) < 1e-6 {
            break;
        }
    }

    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtracking_line_search() {
        let mut quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let x = Array1::from_vec(vec![1.0, 1.0]);
        let f0 = quadratic(&x.view());
        let direction = Array1::from_vec(vec![-1.0, -1.0]);
        let grad = Array1::from_vec(vec![2.0, 2.0]);

        let (alpha, _f_new) = backtracking_line_search(
            &mut quadratic,
            &x.view(),
            f0,
            &direction.view(),
            &grad.view(),
            1.0,
            0.0001,
            0.5,
            None,
        );

        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }
}
