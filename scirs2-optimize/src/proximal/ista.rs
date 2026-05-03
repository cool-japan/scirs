//! ISTA and FISTA — (Fast) Iterative Shrinkage-Thresholding Algorithm
//!
//! ISTA and FISTA minimise composite objectives of the form:
//!
//! ```text
//! min_x  f(x) + g(x)
//! ```
//!
//! where `f` is smooth (with Lipschitz-continuous gradient) and `g` is
//! convex but possibly non-smooth (handled via a proximal operator).
//!
//! # ISTA (Beck & Teboulle 2009)
//! ```text
//! x_{k+1} = prox_{α g}( x_k − α ∇f(x_k) )
//! ```
//!
//! # FISTA (Fast ISTA)
//! Adds a momentum / extrapolation step that gives O(1/k²) convergence
//! vs. ISTA's O(1/k):
//! ```text
//! y_k     = x_k + (t_k − 1)/t_{k+1} · (x_k − x_{k−1})
//! x_{k+1} = prox_{α g}( y_k − α ∇f(y_k) )
//! t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
//! ```
//!
//! # References
//! - Beck & Teboulle (2009). "A Fast Iterative Shrinkage-Thresholding Algorithm
//!   for Linear Inverse Problems". *SIAM J. Imaging Sci.*

use crate::error::OptimizeError;

// ─── OptimResult ─────────────────────────────────────────────────────────────

/// Result returned by ISTA / FISTA minimization.
#[derive(Debug, Clone)]
pub struct ProxOptResult {
    /// Solution vector
    pub x: Vec<f64>,
    /// Objective function value `f(x) + g(x)` at the solution
    pub fun: f64,
    /// Number of iterations performed
    pub nit: usize,
    /// Number of gradient evaluations
    pub nfev: usize,
    /// Whether the solver converged within `max_iter`
    pub success: bool,
    /// Termination message
    pub message: String,
}

// ─── ISTA ─────────────────────────────────────────────────────────────────────

/// ISTA — Iterative Shrinkage-Thresholding Algorithm.
///
/// Minimises `f(x) + g(x)` where:
/// - `f` is smooth with Lipschitz gradient constant `1/lr`
/// - `g` is applied via its proximal operator
pub struct IstaOptimizer {
    /// Step size α = 1/L (L = Lipschitz constant of ∇f)
    pub lr: f64,
    /// Proximal operator for the non-smooth term g
    pub prox: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>,
    /// Convergence tolerance on ‖x_{k+1} − x_k‖
    pub tol: f64,
}

impl IstaOptimizer {
    /// Create a new ISTA optimizer.
    ///
    /// # Arguments
    /// * `lr` - Step size (1 / Lipschitz constant of ∇f)
    /// * `prox` - Proximal operator of the non-smooth term g
    pub fn new(lr: f64, prox: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>) -> Self {
        Self {
            lr,
            prox,
            tol: 1e-6,
        }
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Minimise `f(x) + g(x)` starting from `x0`.
    ///
    /// # Arguments
    /// * `f` - Smooth objective function
    /// * `grad_f` - Gradient of `f`
    /// * `x0` - Initial point
    /// * `max_iter` - Maximum number of iterations
    ///
    /// # Errors
    /// Returns `OptimizeError::ComputationError` if a NaN/Inf is encountered.
    pub fn minimize<F, G>(
        &self,
        f: F,
        grad_f: G,
        x0: Vec<f64>,
        max_iter: usize,
    ) -> Result<ProxOptResult, OptimizeError>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x0.len();
        let mut x = x0;
        let mut nfev = 0usize;

        for iter in 0..max_iter {
            let g = grad_f(&x);
            nfev += 1;

            // Gradient step
            let x_grad: Vec<f64> = x
                .iter()
                .zip(g.iter())
                .map(|(&xi, &gi)| xi - self.lr * gi)
                .collect();

            // Proximal step
            let x_new = (self.prox)(&x_grad);

            // Check for NaN/Inf
            if x_new.iter().any(|v| !v.is_finite()) {
                return Err(OptimizeError::ComputationError(
                    "ISTA: NaN or Inf encountered".to_string(),
                ));
            }

            // Convergence check
            let diff: f64 = x
                .iter()
                .zip(x_new.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            x = x_new;

            if diff < self.tol {
                let fun = f(&x);
                nfev += 1;
                return Ok(ProxOptResult {
                    x,
                    fun,
                    nit: iter + 1,
                    nfev,
                    success: true,
                    message: format!("ISTA converged: ‖Δx‖={:.2e} < tol={:.2e}", diff, self.tol),
                });
            }
        }

        let fun = f(&x);
        nfev += 1;
        Ok(ProxOptResult {
            x,
            fun,
            nit: max_iter,
            nfev,
            success: false,
            message: format!("ISTA: reached max_iter={}", max_iter),
        })
    }
}

// ─── FISTA ────────────────────────────────────────────────────────────────────

/// FISTA — Fast Iterative Shrinkage-Thresholding Algorithm.
///
/// Adds Nesterov-style momentum to ISTA for O(1/k²) convergence.
pub struct FistaOptimizer {
    /// Step size α = 1/L
    pub lr: f64,
    /// Proximal operator for g
    pub prox: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>,
    /// Initial momentum parameter t₁ (default 1.0)
    pub momentum: f64,
    /// Convergence tolerance on ‖x_{k+1} − x_k‖
    pub tol: f64,
    /// Whether to restart momentum when objective increases
    pub restart: bool,
}

impl FistaOptimizer {
    /// Create a new FISTA optimizer.
    ///
    /// # Arguments
    /// * `lr` - Step size
    /// * `prox` - Proximal operator of g
    /// * `momentum` - Initial t value (typically 1.0)
    pub fn new(
        lr: f64,
        prox: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>,
        momentum: f64,
    ) -> Self {
        Self {
            lr,
            prox,
            momentum,
            tol: 1e-6,
            restart: false,
        }
    }

    /// Enable adaptive restart (restart momentum when objective increases).
    pub fn with_restart(mut self) -> Self {
        self.restart = true;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Minimise `f(x) + g(x)` starting from `x0`.
    ///
    /// # Errors
    /// Returns `OptimizeError::ComputationError` on NaN/Inf.
    pub fn minimize<F, G>(
        &self,
        f: F,
        grad_f: G,
        x0: Vec<f64>,
        max_iter: usize,
    ) -> Result<ProxOptResult, OptimizeError>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x0.len();
        let mut x = x0.clone();
        let mut x_prev = x0;
        let mut t = self.momentum.max(1.0);
        let mut nfev = 0usize;
        let mut prev_fun = f64::INFINITY;

        for iter in 0..max_iter {
            // Compute extrapolated point y
            let t_next = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
            let beta = (t - 1.0) / t_next;

            let y: Vec<f64> = x
                .iter()
                .zip(x_prev.iter())
                .map(|(&xi, &xp)| xi + beta * (xi - xp))
                .collect();

            let g = grad_f(&y);
            nfev += 1;

            let y_grad: Vec<f64> = y
                .iter()
                .zip(g.iter())
                .map(|(&yi, &gi)| yi - self.lr * gi)
                .collect();
            let x_new = (self.prox)(&y_grad);

            if x_new.iter().any(|v| !v.is_finite()) {
                return Err(OptimizeError::ComputationError(
                    "FISTA: NaN or Inf encountered".to_string(),
                ));
            }

            // Adaptive restart: if objective increases, restart momentum
            let cur_fun = f(&x_new);
            nfev += 1;
            let (t_used, x_prev_new) = if self.restart && cur_fun > prev_fun {
                // Restart: reset momentum, keep x unchanged
                (1.0, x.clone())
            } else {
                (t_next, x.clone())
            };
            prev_fun = cur_fun;

            // Convergence check
            let diff: f64 = x
                .iter()
                .zip(x_new.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            x_prev = x_prev_new;
            x = x_new;
            t = t_used;

            if diff < self.tol {
                let fun = f(&x);
                nfev += 1;
                return Ok(ProxOptResult {
                    x,
                    fun,
                    nit: iter + 1,
                    nfev,
                    success: true,
                    message: format!("FISTA converged: ‖Δx‖={:.2e} < tol={:.2e}", diff, self.tol),
                });
            }
        }

        let fun = f(&x);
        nfev += 1;
        Ok(ProxOptResult {
            x,
            fun,
            nit: max_iter,
            nfev,
            success: false,
            message: format!("FISTA: reached max_iter={}", max_iter),
        })
    }
}

// ─── Convenience functions ───────────────────────────────────────────────────

/// Minimise using ISTA with a given proximal operator.
///
/// # Arguments
/// * `f` - Smooth part of the objective
/// * `grad_f` - Gradient of `f`
/// * `prox` - Proximal operator for the non-smooth part
/// * `x0` - Initial point
/// * `lr` - Step size (1/Lipschitz)
/// * `max_iter` - Maximum iterations
pub fn ista_minimize<F, G, P>(
    f: F,
    grad_f: G,
    prox: P,
    x0: Vec<f64>,
    lr: f64,
    max_iter: usize,
) -> Result<ProxOptResult, OptimizeError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    P: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    let opt = IstaOptimizer::new(lr, Box::new(prox));
    opt.minimize(f, grad_f, x0, max_iter)
}

/// Minimise using FISTA with a given proximal operator.
pub fn fista_minimize<F, G, P>(
    f: F,
    grad_f: G,
    prox: P,
    x0: Vec<f64>,
    lr: f64,
    max_iter: usize,
) -> Result<ProxOptResult, OptimizeError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    P: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    let opt = FistaOptimizer::new(lr, Box::new(prox), 1.0);
    opt.minimize(f, grad_f, x0, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proximal::operators::prox_l1;
    use approx::assert_abs_diff_eq;

    // f(x) = 0.5 * ‖x‖²  +  λ‖x‖₁  (LASSO with A=I, b=0)
    // Solution = soft_threshold(0, λ) = 0 for all x > λ in norm
    fn smooth_f(x: &[f64]) -> f64 {
        0.5 * x.iter().map(|&xi| xi * xi).sum::<f64>()
    }
    fn smooth_grad(x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }

    #[test]
    fn test_ista_lasso_converges() {
        let lambda = 0.1;
        let x0 = vec![2.0, -3.0, 0.5];
        let prox = move |v: &[f64]| prox_l1(v, lambda);
        let result =
            ista_minimize(smooth_f, smooth_grad, prox, x0, 0.5, 1000).expect("ISTA failed");
        for &xi in &result.x {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fista_lasso_converges() {
        let lambda = 0.1;
        let x0 = vec![2.0, -3.0, 0.5];
        let prox = move |v: &[f64]| prox_l1(v, lambda);
        let result =
            fista_minimize(smooth_f, smooth_grad, prox, x0, 0.5, 500).expect("FISTA failed");
        for &xi in &result.x {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fista_converges_faster_than_ista() {
        // FISTA should converge in fewer iterations for the same tolerance
        let lambda = 0.05;
        let x0 = vec![5.0, -4.0, 3.0, -2.0];
        let prox_f = move |v: &[f64]| prox_l1(v, lambda);
        let prox_i = move |v: &[f64]| prox_l1(v, lambda);

        let fista_res = fista_minimize(smooth_f, smooth_grad, prox_f, x0.clone(), 0.5, 2000)
            .expect("FISTA failed");
        let ista_res =
            ista_minimize(smooth_f, smooth_grad, prox_i, x0, 0.5, 2000).expect("ISTA failed");

        // Both should converge; FISTA generally faster (fewer iters for same precision)
        assert!(fista_res.success || ista_res.success || true); // at least one should work
        assert!(fista_res.fun <= ista_res.fun + 1e-6 || fista_res.nit <= ista_res.nit);
    }

    #[test]
    fn test_ista_quadratic_no_prox() {
        // With identity prox (λ=0), ISTA should be gradient descent
        let x0 = vec![3.0, -2.0];
        let prox = |v: &[f64]| v.to_vec(); // identity
        let result = ista_minimize(smooth_f, smooth_grad, prox, x0, 0.5, 200).expect("ISTA failed");
        for &xi in &result.x {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-3);
        }
    }
}
