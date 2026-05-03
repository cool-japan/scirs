//! Nelder-Mead simplex method for derivative-free optimization.
//!
//! The Nelder-Mead method maintains a simplex of n+1 points and iteratively
//! transforms it toward the optimum using reflection, expansion, contraction,
//! and shrink operations.
//!
//! # References
//! - Nelder, J.A. & Mead, R. (1965). "A simplex method for function minimization."
//!   The Computer Journal, 7(4), 308-313.
//! - Gao, F. & Han, L. (2012). "Implementing the Nelder-Mead simplex algorithm
//!   with adaptive parameters." Computational Optimization and Applications.

use super::{centroid, DerivativeFreeOptimizer, DfOptResult};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{s, Array1, Array2};

/// Options for the Nelder-Mead algorithm
#[derive(Debug, Clone)]
pub struct NelderMeadOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Maximum number of function evaluations
    pub max_fev: usize,
    /// Absolute tolerance on function value
    pub f_tol: f64,
    /// Absolute tolerance on solution point
    pub x_tol: f64,
    /// Reflection coefficient (alpha > 0)
    pub alpha: f64,
    /// Expansion coefficient (gamma > 1)
    pub gamma: f64,
    /// Contraction coefficient (0 < rho < 0.5)
    pub rho: f64,
    /// Shrink coefficient (0 < sigma < 1)
    pub sigma: f64,
    /// Initial simplex size (fraction of x0 or absolute if x0=0)
    pub initial_simplex_size: f64,
    /// Use adaptive parameters (recommended for n >= 5)
    pub adaptive: bool,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        NelderMeadOptions {
            max_iter: 10000,
            max_fev: 100000,
            f_tol: 1e-8,
            x_tol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            initial_simplex_size: 0.05,
            adaptive: false,
        }
    }
}

/// Nelder-Mead simplex optimizer
pub struct NelderMeadSolver {
    pub options: NelderMeadOptions,
}

impl NelderMeadSolver {
    /// Create with default options
    pub fn new() -> Self {
        NelderMeadSolver {
            options: NelderMeadOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: NelderMeadOptions) -> Self {
        NelderMeadSolver { options }
    }

    /// Build the initial simplex around x0
    fn build_initial_simplex(&self, x0: &[f64]) -> Array2<f64> {
        let n = x0.len();
        let mut simplex = Array2::zeros((n + 1, n));
        for j in 0..n {
            simplex[[0, j]] = x0[j];
        }
        for i in 0..n {
            for j in 0..n {
                simplex[[i + 1, j]] = x0[j];
            }
            let delta = if x0[i].abs() > 1e-8 {
                x0[i] * self.options.initial_simplex_size
            } else {
                self.options.initial_simplex_size
            };
            simplex[[i + 1, i]] = x0[i] + delta;
        }
        simplex
    }

    /// Evaluate function at all simplex vertices, return sorted (values, order)
    fn eval_and_sort<F: Fn(&[f64]) -> f64>(
        &self,
        func: &F,
        simplex: &Array2<f64>,
    ) -> (Vec<f64>, Vec<usize>) {
        let n1 = simplex.nrows();
        let mut fvals: Vec<f64> = (0..n1)
            .map(|i| func(simplex.slice(s![i, ..]).as_slice().unwrap_or(&[])))
            .collect();
        // handle non-contiguous by collecting
        let fvals_safe: Vec<f64> = (0..n1)
            .map(|i| {
                let row: Vec<f64> = simplex.row(i).iter().copied().collect();
                func(&row)
            })
            .collect();
        fvals = fvals_safe;

        let mut order: Vec<usize> = (0..n1).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();
        (sorted_fvals, order)
    }

    /// Get adaptive parameters based on dimension
    fn get_params(&self, n: usize) -> (f64, f64, f64, f64) {
        if self.options.adaptive {
            let nf = n as f64;
            let alpha = 1.0;
            let gamma = 1.0 + 2.0 / nf;
            let rho = 0.75 - 1.0 / (2.0 * nf);
            let sigma = 1.0 - 1.0 / nf;
            (alpha, gamma, rho, sigma)
        } else {
            (
                self.options.alpha,
                self.options.gamma,
                self.options.rho,
                self.options.sigma,
            )
        }
    }
}

impl Default for NelderMeadSolver {
    fn default() -> Self {
        NelderMeadSolver::new()
    }
}

impl DerivativeFreeOptimizer for NelderMeadSolver {
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
        let (alpha, gamma, rho, sigma) = self.get_params(n);

        // Build initial simplex
        let mut simplex = self.build_initial_simplex(x0);

        // Initial function evaluations
        let mut fvals: Vec<f64> = (0..n + 1)
            .map(|i| {
                let row: Vec<f64> = simplex.row(i).iter().copied().collect();
                func(&row)
            })
            .collect();

        let mut nfev = n + 1;
        let mut nit = 0;

        // Sort simplex
        let sort_simplex = |simplex: &mut Array2<f64>, fvals: &mut Vec<f64>| {
            let n1 = simplex.nrows();
            let mut order: Vec<usize> = (0..n1).collect();
            order.sort_by(|&a, &b| {
                fvals[a]
                    .partial_cmp(&fvals[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let old_simplex = simplex.clone();
            let old_fvals = fvals.clone();
            for (new_idx, &old_idx) in order.iter().enumerate() {
                simplex.row_mut(new_idx).assign(&old_simplex.row(old_idx));
                fvals[new_idx] = old_fvals[old_idx];
            }
        };

        sort_simplex(&mut simplex, &mut fvals);

        let eval = |func: &F, x: &[f64], nfev: &mut usize| -> f64 {
            *nfev += 1;
            func(x)
        };

        loop {
            // Termination checks
            if nit >= self.options.max_iter {
                let best: Vec<f64> = simplex.row(0).iter().copied().collect();
                return Ok(DfOptResult {
                    x: Array1::from_vec(best),
                    fun: fvals[0],
                    nfev,
                    nit,
                    success: false,
                    message: "Maximum iterations reached".to_string(),
                });
            }
            if nfev >= self.options.max_fev {
                let best: Vec<f64> = simplex.row(0).iter().copied().collect();
                return Ok(DfOptResult {
                    x: Array1::from_vec(best),
                    fun: fvals[0],
                    nfev,
                    nit,
                    success: false,
                    message: "Maximum function evaluations reached".to_string(),
                });
            }

            // Convergence: function value spread
            let f_spread = (fvals[n] - fvals[0]).abs();
            if f_spread < self.options.f_tol {
                let best: Vec<f64> = simplex.row(0).iter().copied().collect();
                return Ok(DfOptResult {
                    x: Array1::from_vec(best),
                    fun: fvals[0],
                    nfev,
                    nit,
                    success: true,
                    message: "Converged: function value tolerance".to_string(),
                });
            }

            // Convergence: simplex size
            let best_row: Vec<f64> = simplex.row(0).iter().copied().collect();
            let max_dist = (1..n + 1)
                .map(|i| {
                    let row: Vec<f64> = simplex.row(i).iter().copied().collect();
                    row.iter()
                        .zip(best_row.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                })
                .fold(0.0_f64, f64::max);
            if max_dist < self.options.x_tol {
                return Ok(DfOptResult {
                    x: Array1::from_vec(best_row),
                    fun: fvals[0],
                    nfev,
                    nit,
                    success: true,
                    message: "Converged: simplex size tolerance".to_string(),
                });
            }

            // Centroid of best n vertices (excluding worst)
            let xbar = centroid(&simplex, n);
            let xbar_vec: Vec<f64> = xbar.iter().copied().collect();
            let xworst: Vec<f64> = simplex.row(n).iter().copied().collect();

            // Reflection: xr = xbar + alpha * (xbar - xworst)
            let xr: Vec<f64> = xbar_vec
                .iter()
                .zip(xworst.iter())
                .map(|(&c, &w)| c + alpha * (c - w))
                .collect();
            let fr = eval(&func, &xr, &mut nfev);

            if fr < fvals[0] {
                // Expansion: xe = xbar + gamma * (xr - xbar)
                let xe: Vec<f64> = xbar_vec
                    .iter()
                    .zip(xr.iter())
                    .map(|(&c, &r)| c + gamma * (r - c))
                    .collect();
                let fe = eval(&func, &xe, &mut nfev);
                if fe < fr {
                    simplex.row_mut(n).assign(&Array1::from_vec(xe));
                    fvals[n] = fe;
                } else {
                    simplex.row_mut(n).assign(&Array1::from_vec(xr));
                    fvals[n] = fr;
                }
            } else if fr < fvals[n - 1] {
                // Accept reflection
                simplex.row_mut(n).assign(&Array1::from_vec(xr));
                fvals[n] = fr;
            } else {
                // Contraction
                let do_outside = fr < fvals[n];
                let xc_vec: Vec<f64> = if do_outside {
                    // Outside contraction
                    xbar_vec
                        .iter()
                        .zip(xr.iter())
                        .map(|(&c, &r)| c + rho * (r - c))
                        .collect()
                } else {
                    // Inside contraction
                    xbar_vec
                        .iter()
                        .zip(xworst.iter())
                        .map(|(&c, &w)| c + rho * (w - c))
                        .collect()
                };
                let fc = eval(&func, &xc_vec, &mut nfev);
                let f_compare = if do_outside { fr } else { fvals[n] };
                if fc < f_compare {
                    simplex.row_mut(n).assign(&Array1::from_vec(xc_vec));
                    fvals[n] = fc;
                } else {
                    // Shrink
                    let x0_row: Vec<f64> = simplex.row(0).iter().copied().collect();
                    for i in 1..n + 1 {
                        let xi: Vec<f64> = simplex.row(i).iter().copied().collect();
                        let new_xi: Vec<f64> = x0_row
                            .iter()
                            .zip(xi.iter())
                            .map(|(&b, &xi_j)| b + sigma * (xi_j - b))
                            .collect();
                        simplex.row_mut(i).assign(&Array1::from_vec(new_xi.clone()));
                        fvals[i] = eval(&func, &new_xi, &mut nfev);
                    }
                }
            }

            sort_simplex(&mut simplex, &mut fvals);
            nit += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_nelder_mead_quadratic() {
        let solver = NelderMeadSolver::new();
        let result = solver
            .minimize(
                |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2),
                &[0.0, 0.0],
            )
            .expect("optimization failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 3.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], -2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let opts = NelderMeadOptions {
            max_iter: 50000,
            max_fev: 500000,
            f_tol: 1e-9,
            x_tol: 1e-9,
            ..Default::default()
        };
        let solver = NelderMeadSolver::with_options(opts);
        let result = solver
            .minimize(
                |x: &[f64]| {
                    let a = 1.0 - x[0];
                    let b = x[1] - x[0].powi(2);
                    a * a + 100.0 * b * b
                },
                &[0.0, 0.0],
            )
            .expect("optimization failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nelder_mead_1d() {
        let solver = NelderMeadSolver::new();
        let result = solver
            .minimize(|x: &[f64]| (x[0] - 5.0).powi(4), &[0.0])
            .expect("optimization failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_nelder_mead_adaptive() {
        let opts = NelderMeadOptions {
            adaptive: true,
            max_iter: 10000,
            ..Default::default()
        };
        let solver = NelderMeadSolver::with_options(opts);
        // 5D sphere function
        let result = solver
            .minimize(
                |x: &[f64]| {
                    x.iter()
                        .enumerate()
                        .map(|(i, &xi)| (xi - i as f64).powi(2))
                        .sum()
                },
                &[0.0; 5],
            )
            .expect("optimization failed");
        assert!(result.fun < 1e-4, "fun={}", result.fun);
    }

    #[test]
    fn test_nelder_mead_from_nonzero_start() {
        let solver = NelderMeadSolver::new();
        let result = solver
            .minimize(|x: &[f64]| x[0].powi(2) + x[1].powi(2), &[10.0, -10.0])
            .expect("optimization failed");
        assert!(result.success);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-5);
    }
}
