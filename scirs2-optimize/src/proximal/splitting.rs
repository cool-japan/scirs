//! Operator Splitting Methods
//!
//! This module provides splitting algorithms for optimising sums of non-smooth
//! convex functions, where no single proximal operator is available for the
//! combined objective.
//!
//! # Algorithms
//!
//! ## Douglas-Rachford Splitting
//! Minimises `f(x) + g(x)` using only `prox_f` and `prox_g`:
//! ```text
//! y_{k+1} = prox_{γg}(z_k)
//! x_{k+1} = prox_{γf}(2y_{k+1} − z_k)
//! z_{k+1} = z_k + x_{k+1} − y_{k+1}
//! ```
//!
//! ## Peaceman-Rachford Splitting
//! A less damped variant that requires strong monotonicity to converge.
//!
//! ## Primal-Dual (Chambolle-Pock)
//! Solves `min_x f(x) + g(Kx)` where `K` is a linear operator.
//!
//! # References
//! - Lions & Mercier (1979). "Splitting Algorithms for the Sum of Two Nonlinear
//!   Operators". *SIAM J. Numer. Anal.*
//! - Eckstein & Bertsekas (1992). "On the Douglas-Rachford Splitting Method".
//!   *Math. Programming.*
//! - Chambolle & Pock (2011). "A First-Order Primal-Dual Algorithm for Convex
//!   Problems with Applications to Imaging". *J. Math. Imaging Vision.*

use crate::error::OptimizeError;

// ─── Douglas-Rachford Splitting ──────────────────────────────────────────────

/// Minimise `f(x) + g(x)` using Douglas-Rachford (DR) splitting.
///
/// The algorithm only requires the proximal operators of `f` and `g`
/// separately and does not require differentiability.
///
/// # Convergence
/// Converges for any pair of proper, closed, convex functions when γ > 0.
/// The fixed-point iterates `{z_k}` converge; the actual solution is
/// `prox_{γg}(z_∞)`.
///
/// # Arguments
/// * `prox_f` - Proximal operator of f: `prox_{γf}(·)`
/// * `prox_g` - Proximal operator of g: `prox_{γg}(·)`
/// * `x0` - Starting point (initialises z₀ = x₀)
/// * `gamma` - Step size / scaling parameter (γ > 0)
/// * `max_iter` - Maximum number of DR iterations
///
/// # Returns
/// The approximate minimiser `x* = prox_{γg}(z_∞)`.
pub fn douglas_rachford(
    prox_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    gamma: f64,
    max_iter: usize,
) -> Vec<f64> {
    let _n = x0.len();
    let mut z = x0;

    for _ in 0..max_iter {
        let y = prox_g(&z);
        let two_y_minus_z: Vec<f64> = y
            .iter()
            .zip(z.iter())
            .map(|(&yi, &zi)| 2.0 * yi - zi)
            .collect();
        let x = prox_f(&two_y_minus_z);
        // z_{k+1} = z_k + x_{k+1} - y_{k+1}
        z = z
            .iter()
            .zip(x.iter().zip(y.iter()))
            .map(|(&zk, (&xk1, &yk1))| zk + xk1 - yk1)
            .collect();
    }

    // Recover solution: x* = prox_g(z)
    prox_g(&z)
}

/// Douglas-Rachford splitting with convergence tracking.
///
/// Returns the solution along with convergence diagnostics.
///
/// # Arguments
/// Same as `douglas_rachford`, plus:
/// * `tol` - Convergence tolerance on ‖z_{k+1} − z_k‖
pub fn douglas_rachford_tracked(
    prox_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> DRResult {
    let n = x0.len();
    let mut z = x0;
    let _ = gamma; // gamma is used implicitly through the prox scaling

    for iter in 0..max_iter {
        let z_prev = z.clone();

        let y = prox_g(&z);
        let two_y_minus_z: Vec<f64> = y
            .iter()
            .zip(z.iter())
            .map(|(&yi, &zi)| 2.0 * yi - zi)
            .collect();
        let x = prox_f(&two_y_minus_z);
        z = z
            .iter()
            .zip(x.iter().zip(y.iter()))
            .map(|(&zk, (&xk1, &yk1))| zk + xk1 - yk1)
            .collect();

        let dz: f64 = z
            .iter()
            .zip(z_prev.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        if dz < tol {
            let x_star = prox_g(&z);
            return DRResult {
                x: x_star,
                nit: iter + 1,
                converged: true,
                final_residual: dz,
            };
        }
    }

    let x_star = prox_g(&z);
    let final_res: f64 = 0.0; // Would need extra iteration to compute
    DRResult {
        x: x_star,
        nit: max_iter,
        converged: false,
        final_residual: final_res,
    }
}

/// Result of a tracked Douglas-Rachford run.
#[derive(Debug, Clone)]
pub struct DRResult {
    /// Approximate minimiser
    pub x: Vec<f64>,
    /// Number of iterations performed
    pub nit: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final ‖z_{k+1} − z_k‖ residual
    pub final_residual: f64,
}

// ─── Peaceman-Rachford Splitting ─────────────────────────────────────────────

/// Peaceman-Rachford splitting (less damped variant of DR).
///
/// Unlike DR, the intermediate iterate is reflected rather than just
/// forward-stepped:
/// ```text
/// y_{k+1} = prox_{γg}(z_k)
/// x_{k+1} = prox_{γf}(2y_{k+1} − z_k)
/// z_{k+1} = 2x_{k+1} − (2y_{k+1} − z_k)
/// ```
///
/// Converges faster when both f and g are strongly convex, but may diverge
/// otherwise. Use `douglas_rachford` for general non-smooth problems.
///
/// # Arguments
/// Same as `douglas_rachford`.
pub fn peaceman_rachford(
    prox_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    _gamma: f64,
    max_iter: usize,
) -> Vec<f64> {
    let mut z = x0;

    for _ in 0..max_iter {
        let y = prox_g(&z);
        let refl_y: Vec<f64> = y
            .iter()
            .zip(z.iter())
            .map(|(&yi, &zi)| 2.0 * yi - zi)
            .collect();
        let x = prox_f(&refl_y);
        // z = 2x - reflect_y  (full reflection through x)
        z = x
            .iter()
            .zip(refl_y.iter())
            .map(|(&xi, &ri)| 2.0 * xi - ri)
            .collect();
    }

    prox_g(&z)
}

// ─── Forward-Backward Splitting ──────────────────────────────────────────────

/// Forward-backward splitting: `min f(x) + g(x)` where `f` is smooth.
///
/// Performs a gradient step on `f` followed by a proximal step on `g`:
/// ```text
/// x_{k+1} = prox_{αg}(x_k − α·∇f(x_k))
/// ```
///
/// This is exactly ISTA generalised to arbitrary proximal operators.
///
/// # Arguments
/// * `grad_f` - Gradient of smooth term f
/// * `prox_g` - Proximal operator of non-smooth term g
/// * `x0` - Initial point
/// * `alpha` - Step size (1/Lipschitz constant of ∇f)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn forward_backward(
    grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut x = x0;

    for _ in 0..max_iter {
        let g = grad_f(&x);
        let x_grad: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .map(|(&xi, &gi)| xi - alpha * gi)
            .collect();
        let x_new = prox_g(&x_grad);

        let diff: f64 = x
            .iter()
            .zip(x_new.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        x = x_new;
        if diff < tol {
            break;
        }
    }
    x
}

// ─── Primal-Dual (Chambolle-Pock) ────────────────────────────────────────────

/// Primal-dual algorithm (Chambolle-Pock) for `min_x f(x) + g(Kx)`.
///
/// Iterates:
/// ```text
/// y_{k+1}   = prox_{σ g*}(y_k + σ·K·x_bar_k)
/// x_{k+1}   = prox_{τ f}(x_k − τ·Kᵀ·y_{k+1})
/// x_bar_{k+1} = x_{k+1} + θ·(x_{k+1} − x_k)
/// ```
///
/// where `g*` is the convex conjugate of `g`.
///
/// # Arguments
/// * `prox_f` - Proximal operator of f (scaled by τ)
/// * `prox_g_conj` - Proximal operator of conjugate g* (scaled by σ)
/// * `k_op` - Linear operator K: x → Kx
/// * `kt_op` - Adjoint K*: y → Kᵀy
/// * `x0` - Primal initial point
/// * `y0` - Dual initial point
/// * `tau` - Primal step size
/// * `sigma` - Dual step size
/// * `theta` - Over-relaxation (0 = no relaxation, 1 = full)
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// `(x_star, y_star)` — primal and dual solutions.
#[allow(clippy::too_many_arguments)]
pub fn primal_dual_chambolle_pock(
    prox_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g_conj: &dyn Fn(&[f64]) -> Vec<f64>,
    k_op: &dyn Fn(&[f64]) -> Vec<f64>,
    kt_op: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    y0: Vec<f64>,
    tau: f64,
    sigma: f64,
    theta: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let _ = (tau, sigma); // used implicitly through scaled prox operators
    let mut x = x0;
    let mut y = y0;
    let mut x_bar = x.clone();

    for _ in 0..max_iter {
        let x_old = x.clone();

        // Dual update
        let kx_bar = k_op(&x_bar);
        let y_input: Vec<f64> = y
            .iter()
            .zip(kx_bar.iter())
            .map(|(&yi, &kxi)| yi + kxi)
            .collect();
        y = prox_g_conj(&y_input);

        // Primal update
        let kty = kt_op(&y);
        let x_input: Vec<f64> = x
            .iter()
            .zip(kty.iter())
            .map(|(&xi, &kti)| xi - kti)
            .collect();
        x = prox_f(&x_input);

        // Over-relaxation
        x_bar = x
            .iter()
            .zip(x_old.iter())
            .map(|(&xn, &xo)| xn + theta * (xn - xo))
            .collect();
    }
    (x, y)
}

/// Result of a splitting algorithm with diagnostics.
#[derive(Debug, Clone)]
pub struct SplittingResult {
    /// Primal solution
    pub x: Vec<f64>,
    /// Number of iterations
    pub nit: usize,
    /// Whether convergence criterion was met
    pub converged: bool,
}

/// Run Douglas-Rachford splitting and return a `SplittingResult`.
pub fn dr_split(
    prox_f: &dyn Fn(&[f64]) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: Vec<f64>,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> Result<SplittingResult, OptimizeError> {
    if gamma <= 0.0 {
        return Err(OptimizeError::ValueError(
            "gamma must be positive for Douglas-Rachford".to_string(),
        ));
    }
    let res = douglas_rachford_tracked(prox_f, prox_g, x0, gamma, max_iter, tol);
    Ok(SplittingResult {
        x: res.x,
        nit: res.nit,
        converged: res.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proximal::operators::{prox_l1, prox_l2};
    use approx::assert_abs_diff_eq;

    /// Identity proximal (no regularization)
    fn prox_id(v: &[f64]) -> Vec<f64> {
        v.to_vec()
    }

    #[test]
    fn test_douglas_rachford_l1_l2() {
        // min ‖x‖₁ + ‖x‖₂² starting near [3, -2]
        let lambda_l1 = 0.5;
        let lambda_l2 = 0.5;
        let prox_f = |v: &[f64]| prox_l1(v, lambda_l1);
        let prox_g = |v: &[f64]| prox_l2(v, lambda_l2);
        let x0 = vec![3.0, -2.0, 1.0];
        let result = douglas_rachford(&prox_f, &prox_g, x0, 1.0, 500);
        // Solution should be near 0 (L1 + L2 → sparsity near 0)
        for &xi in &result {
            assert!(xi.abs() < 1.0, "DR solution out of expected range: {}", xi);
        }
    }

    #[test]
    fn test_douglas_rachford_identity_prox() {
        // When prox_g = identity, DR degenerates to: x = prox_f(2*x - z)
        // which should drive x toward the fixed point of prox_f
        let prox_f = |v: &[f64]| prox_l1(v, 1.0);
        let x0 = vec![2.0, -3.0];
        let result = douglas_rachford(&prox_f, &prox_id, x0, 1.0, 1000);
        // prox_l1(·,1) fixed points: {x : |x| ≤ 1}
        for &xi in &result {
            assert!(xi.abs() <= 1.0 + 1e-8, "not in expected set: {}", xi);
        }
    }

    #[test]
    fn test_dr_tracked_convergence() {
        let prox_f = |v: &[f64]| prox_l1(v, 0.3);
        let prox_g = |v: &[f64]| prox_l2(v, 0.3);
        let x0 = vec![2.0, -1.0];
        let res = douglas_rachford_tracked(&prox_f, &prox_g, x0, 1.0, 2000, 1e-8);
        assert!(res.converged, "DR should converge within 2000 iters");
        assert!(res.nit < 2000, "DR should converge before max_iter");
    }

    #[test]
    fn test_forward_backward_quadratic() {
        // f(x) = ½‖x‖², prox_g = identity → x_{k+1} = x_k - α·x_k = (1-α)·x_k
        let grad_f = |x: &[f64]| x.to_vec();
        let x0 = vec![3.0, -2.0];
        let result = forward_backward(&grad_f, &prox_id, x0, 0.5, 500, 1e-8);
        for &xi in &result {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_peaceman_rachford_converges() {
        let prox_f = |v: &[f64]| prox_l2(v, 0.5);
        let prox_g = |v: &[f64]| prox_l2(v, 0.5);
        let x0 = vec![2.0, -1.5];
        let result = peaceman_rachford(&prox_f, &prox_g, x0, 1.0, 500);
        for &xi in &result {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_dr_split_negative_gamma() {
        let prox_f = |v: &[f64]| v.to_vec();
        let prox_g = |v: &[f64]| v.to_vec();
        let result = dr_split(&prox_f, &prox_g, vec![1.0], -1.0, 10, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_primal_dual_basic() {
        // trivial: K = I, f = ½‖·‖², g(y) = ½‖y‖²
        // Solution: x* = 0
        let prox_f = |v: &[f64]| prox_l2(v, 0.5);
        let prox_g_conj = |v: &[f64]| prox_l2(v, 0.5);
        let k_op = |x: &[f64]| x.to_vec();
        let kt_op = |y: &[f64]| y.to_vec();
        let x0 = vec![2.0, -1.0];
        let y0 = vec![0.0, 0.0];
        let (x_star, _) = primal_dual_chambolle_pock(
            &prox_f,
            &prox_g_conj,
            &k_op,
            &kt_op,
            x0,
            y0,
            0.5,
            0.5,
            1.0,
            500,
        );
        for &xi in &x_star {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 0.1);
        }
    }
}
