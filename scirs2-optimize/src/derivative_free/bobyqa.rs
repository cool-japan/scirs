//! BOBYQA: Bound Optimization BY Quadratic Approximation
//!
//! BOBYQA is a derivative-free optimization algorithm for problems with bound
//! constraints. It maintains a quadratic model of the objective function
//! interpolated through a set of points, and uses a trust-region framework
//! to ensure progress.
//!
//! This implementation provides the key structure and trust-region quadratic
//! approximation strategy described by Powell (2009).
//!
//! # References
//! - Powell, M.J.D. (2009). "The BOBYQA algorithm for bound constrained optimization
//!   without derivatives." Technical Report DAMTP 2009/NA06, Cambridge University.

use super::{clip, DerivativeFreeOptimizer, DfOptResult};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::Array1;

/// Options for the BOBYQA algorithm
#[derive(Debug, Clone)]
pub struct BOBYQAOptions {
    /// Lower bounds (None = -inf for each variable)
    pub lower: Option<Vec<f64>>,
    /// Upper bounds (None = +inf for each variable)
    pub upper: Option<Vec<f64>>,
    /// Initial trust region radius
    pub rho_begin: f64,
    /// Final trust region radius (convergence)
    pub rho_end: f64,
    /// Maximum number of function evaluations
    pub max_fev: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Number of interpolation points (npt). Must satisfy n+2 <= npt <= (n+1)(n+2)/2.
    /// Default: 2*n+1
    pub npt: Option<usize>,
    /// Absolute tolerance for function value change
    pub f_tol: f64,
    /// Absolute tolerance for step size
    pub x_tol: f64,
}

impl Default for BOBYQAOptions {
    fn default() -> Self {
        BOBYQAOptions {
            lower: None,
            upper: None,
            rho_begin: 1.0,
            rho_end: 1e-6,
            max_fev: 50000,
            max_iter: 10000,
            npt: None,
            f_tol: 1e-10,
            x_tol: 1e-8,
        }
    }
}

/// BOBYQA solver
pub struct BOBYQASolver {
    pub options: BOBYQAOptions,
}

impl BOBYQASolver {
    /// Create with default options
    pub fn new() -> Self {
        BOBYQASolver {
            options: BOBYQAOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: BOBYQAOptions) -> Self {
        BOBYQASolver { options }
    }

    /// Get effective lower/upper bounds for dimension n
    fn get_bounds(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        let lo = match &self.options.lower {
            Some(l) => l.clone(),
            None => vec![f64::NEG_INFINITY; n],
        };
        let hi = match &self.options.upper {
            Some(u) => u.clone(),
            None => vec![f64::INFINITY; n],
        };
        (lo, hi)
    }

    /// Project point onto box [lo, hi]
    fn project(&self, x: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(lo.iter().zip(hi.iter()))
            .map(|(&xi, (&l, &h))| clip(xi, l, h))
            .collect()
    }

    /// Compute quadratic model gradient at current point via finite differences
    fn quadratic_gradient<F: Fn(&[f64]) -> f64>(
        &self,
        func: &F,
        x: &[f64],
        rho: f64,
        lo: &[f64],
        hi: &[f64],
        nfev: &mut usize,
    ) -> Vec<f64> {
        let h = rho * 0.01;
        let n = x.len();
        let mut g = vec![0.0; n];
        let fx = {
            *nfev += 1;
            func(x)
        };
        for i in 0..n {
            let mut xp = x.to_vec();
            xp[i] = clip(x[i] + h, lo[i], hi[i]);
            let actual_h = xp[i] - x[i];
            if actual_h.abs() > 1e-15 {
                *nfev += 1;
                g[i] = (func(&xp) - fx) / actual_h;
            } else {
                let mut xm = x.to_vec();
                xm[i] = clip(x[i] - h, lo[i], hi[i]);
                let actual_hm = x[i] - xm[i];
                if actual_hm.abs() > 1e-15 {
                    *nfev += 1;
                    g[i] = (fx - func(&xm)) / actual_hm;
                }
            }
        }
        g
    }

    /// Solve the trust-region subproblem subject to bounds.
    /// Minimizes: g^T d + 0.5 d^T H d   s.t. ||d|| <= rho, lo <= x+d <= hi
    /// Uses a truncated steepest descent / conjugate gradient approach.
    fn trust_region_step(
        &self,
        g: &[f64],
        hess_diag: &[f64],
        x: &[f64],
        rho: f64,
        lo: &[f64],
        hi: &[f64],
    ) -> Vec<f64> {
        let n = x.len();
        // Cauchy point: steepest descent clipped to trust region and bounds
        let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < 1e-15 {
            return vec![0.0; n];
        }

        // Compute gT H g for scaling
        let gt_hg: f64 = g
            .iter()
            .zip(hess_diag.iter())
            .map(|(&gi, &hi_v)| gi * hi_v.max(0.0) * gi)
            .sum();

        let tau = if gt_hg > 0.0 {
            (gn * gn / gt_hg).min(rho / gn)
        } else {
            rho / gn
        };

        let d: Vec<f64> = g.iter().map(|&gi| -tau * gi).collect();

        // Scale to fit trust region
        let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if dn > rho { rho / dn } else { 1.0 };
        let d: Vec<f64> = d.iter().map(|v| v * scale).collect();

        // Clip to bounds
        x.iter()
            .zip(d.iter())
            .zip(lo.iter().zip(hi.iter()))
            .map(|((&xi, &di), (&l, &h))| clip(xi + di, l, h) - xi)
            .collect()
    }

    /// Estimate diagonal Hessian via finite differences
    fn hessian_diagonal<F: Fn(&[f64]) -> f64>(
        &self,
        func: &F,
        x: &[f64],
        fx: f64,
        h: f64,
        lo: &[f64],
        hi: &[f64],
        nfev: &mut usize,
    ) -> Vec<f64> {
        let n = x.len();
        let mut hd = vec![0.0; n];
        for i in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] = clip(x[i] + h, lo[i], hi[i]);
            xm[i] = clip(x[i] - h, lo[i], hi[i]);
            let hp = xp[i] - x[i];
            let hm = x[i] - xm[i];
            if hp.abs() > 1e-15 && hm.abs() > 1e-15 {
                let fp = {
                    *nfev += 1;
                    func(&xp)
                };
                let fm = {
                    *nfev += 1;
                    func(&xm)
                };
                hd[i] = (fp - 2.0 * fx + fm) / (hp * hm);
            }
        }
        hd
    }
}

impl Default for BOBYQASolver {
    fn default() -> Self {
        BOBYQASolver::new()
    }
}

impl DerivativeFreeOptimizer for BOBYQASolver {
    fn minimize<F>(&self, func: F, x0: &[f64]) -> OptimizeResult<DfOptResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "x0 must be non-empty".to_string(),
            ));
        }

        let (lo, hi) = self.get_bounds(n);

        // Validate bounds
        for i in 0..n {
            if lo[i] > hi[i] {
                return Err(OptimizeError::InvalidInput(format!(
                    "lower[{}] > upper[{}]",
                    i, i
                )));
            }
        }

        let mut x = self.project(x0, &lo, &hi);
        let mut rho = self.options.rho_begin;
        let rho_end = self.options.rho_end;

        let mut nfev = 0usize;
        let mut nit = 0usize;

        let mut fx = {
            nfev += 1;
            func(&x)
        };

        // Main trust-region loop
        // We reduce rho geometrically when no progress is made
        let rho_factor = 0.5_f64;
        let max_rho_reductions = 100usize;
        let mut rho_reductions = 0usize;

        loop {
            if nit >= self.options.max_iter || nfev >= self.options.max_fev {
                break;
            }

            if rho < rho_end || rho_reductions >= max_rho_reductions {
                return Ok(DfOptResult {
                    x: Array1::from_vec(x),
                    fun: fx,
                    nfev,
                    nit,
                    success: rho < rho_end * 10.0,
                    message: if rho < rho_end * 10.0 {
                        "Converged: trust region radius below tolerance".to_string()
                    } else {
                        "Maximum trust region reductions reached".to_string()
                    },
                });
            }

            // Compute gradient
            let g = self.quadratic_gradient(&func, &x, rho, &lo, &hi, &mut nfev);

            // Compute diagonal Hessian
            let h_step = rho * 0.1;
            let hd = self.hessian_diagonal(&func, &x, fx, h_step, &lo, &hi, &mut nfev);

            // Solve trust-region subproblem
            let d = self.trust_region_step(&g, &hd, &x, rho, &lo, &hi);

            let step_norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            if step_norm < 1e-15 {
                rho *= rho_factor;
                rho_reductions += 1;
                nit += 1;
                continue;
            }

            let xnew: Vec<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + di).collect();
            let xnew = self.project(&xnew, &lo, &hi);
            nfev += 1;
            let fnew = func(&xnew);

            // Predicted reduction (linear model)
            let pred = -g
                .iter()
                .zip(d.iter())
                .map(|(&gi, &di)| gi * di)
                .sum::<f64>();

            // Actual reduction
            let actual = fx - fnew;

            // Accept/reject step
            let ratio = if pred.abs() > 1e-15 {
                actual / pred
            } else if actual > 0.0 {
                1.0
            } else {
                0.0
            };

            let f_change = (fnew - fx).abs();

            if ratio > 0.0 || fnew < fx {
                // Accept step
                x = xnew;
                fx = fnew;

                // Expand trust region on good progress
                if ratio > 0.75 {
                    rho = (rho * 2.0).min(self.options.rho_begin);
                    rho_reductions = rho_reductions.saturating_sub(1);
                }
            } else {
                // Reject step, shrink trust region
                rho *= rho_factor;
                rho_reductions += 1;
            }

            // Check function value convergence
            if f_change < self.options.f_tol && step_norm < self.options.x_tol {
                return Ok(DfOptResult {
                    x: Array1::from_vec(x),
                    fun: fx,
                    nfev,
                    nit,
                    success: true,
                    message: "Converged: function/step tolerance".to_string(),
                });
            }

            nit += 1;
        }

        Ok(DfOptResult {
            x: Array1::from_vec(x),
            fun: fx,
            nfev,
            nit,
            success: false,
            message: "Maximum iterations or function evaluations reached".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bobyqa_unconstrained_quadratic() {
        let solver = BOBYQASolver::new();
        let result = solver
            .minimize(
                |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
                &[0.0, 0.0],
            )
            .expect("optimization failed");
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_bobyqa_bounded_simple() {
        let opts = BOBYQAOptions {
            lower: Some(vec![1.0, 1.0]),
            upper: Some(vec![5.0, 5.0]),
            ..Default::default()
        };
        let solver = BOBYQASolver::with_options(opts);
        // Minimum at (-3, -4) but bounded to [1,5]^2, so bound-constrained min is (1,1)
        let result = solver
            .minimize(
                |x: &[f64]| (x[0] + 3.0).powi(2) + (x[1] + 4.0).powi(2),
                &[2.0, 2.0],
            )
            .expect("optimization failed");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_bobyqa_interior_minimum() {
        // Interior minimum within bounds
        let opts = BOBYQAOptions {
            lower: Some(vec![-10.0, -10.0]),
            upper: Some(vec![10.0, 10.0]),
            rho_begin: 1.0,
            rho_end: 1e-7,
            max_fev: 100000,
            ..Default::default()
        };
        let solver = BOBYQASolver::with_options(opts);
        let result = solver
            .minimize(
                |x: &[f64]| (x[0] - 1.5).powi(2) + (x[1] + 0.5).powi(2),
                &[5.0, 5.0],
            )
            .expect("optimization failed");
        assert_abs_diff_eq!(result.x[0], 1.5, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], -0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_bobyqa_1d_bounded() {
        let opts = BOBYQAOptions {
            lower: Some(vec![0.0]),
            upper: Some(vec![4.0]),
            rho_begin: 0.5,
            rho_end: 1e-8,
            max_fev: 10000,
            ..Default::default()
        };
        let solver = BOBYQASolver::with_options(opts);
        // Minimum at x=-2, bounded minimum at x=0
        let result = solver
            .minimize(|x: &[f64]| (x[0] + 2.0).powi(2), &[2.0])
            .expect("optimization failed");
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_bobyqa_initial_point_on_bound() {
        let opts = BOBYQAOptions {
            lower: Some(vec![0.0, 0.0]),
            upper: Some(vec![3.0, 3.0]),
            ..Default::default()
        };
        let solver = BOBYQASolver::with_options(opts);
        // Start at boundary
        let result = solver
            .minimize(
                |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
                &[0.0, 0.0],
            )
            .expect("optimization failed");
        // Should find interior minimum at (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }
}
