//! ADMM — Alternating Direction Method of Multipliers
//!
//! ADMM solves separable convex problems of the form:
//!
//! ```text
//! min_x,z  f(x) + g(z)
//! s.t.     A·x + B·z = c
//! ```
//!
//! by iterating three simple steps:
//! ```text
//! x_{k+1} = argmin_x { f(x) + (ρ/2)‖Ax + Bz_k − c + u_k‖² }
//! z_{k+1} = argmin_z { g(z) + (ρ/2)‖Ax_{k+1} + Bz − c + u_k‖² }
//! u_{k+1} = u_k + Ax_{k+1} + Bz_{k+1} − c
//! ```
//!
//! # Provided Solvers
//! 1. `AdmmSolver` — generic interface with LASSO and consensus specialisations
//! 2. `solve_lasso` — LASSO: `min ½‖Ax−b‖² + λ‖x‖₁`
//! 3. `solve_consensus` — distributed consensus: `min Σᵢ fᵢ(x)`
//!
//! # References
//! - Boyd et al. (2011). "Distributed Optimization and Statistical Learning
//!   via the Alternating Direction Method of Multipliers". *Found. Trends ML*.

use crate::error::OptimizeError;
use crate::proximal::operators::prox_l1;

// ─── Generic ADMM Solver ─────────────────────────────────────────────────────

/// ADMM solver configuration.
#[derive(Debug, Clone)]
pub struct AdmmSolver {
    /// Augmented Lagrangian penalty parameter ρ
    pub rho: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Absolute tolerance for residuals
    pub tol_abs: f64,
    /// Relative tolerance for residuals
    pub tol_rel: f64,
}

impl Default for AdmmSolver {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iter: 1000,
            tol_abs: 1e-4,
            tol_rel: 1e-2,
        }
    }
}

impl AdmmSolver {
    /// Create a new ADMM solver.
    pub fn new(rho: f64, max_iter: usize, tol_abs: f64, tol_rel: f64) -> Self {
        Self {
            rho,
            max_iter,
            tol_abs,
            tol_rel,
        }
    }

    /// Solve LASSO: `min ½‖Ax − b‖² + λ‖x‖₁`.
    ///
    /// Uses the ADMM splitting `f(x) = ½‖Ax−b‖², g(z) = λ‖z‖₁, x=z`.
    ///
    /// # Arguments
    /// * `a` - Design matrix (m × n, row-major)
    /// * `b` - Response vector (length m)
    /// * `lambda` - Regularisation strength
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` on dimension mismatch.
    pub fn solve_lasso(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
        lambda: f64,
    ) -> Result<Vec<f64>, OptimizeError> {
        solve_lasso_admm(
            a,
            b,
            lambda,
            self.rho,
            self.max_iter,
            self.tol_abs,
            self.tol_rel,
        )
    }

    /// Solve consensus problem: `min Σᵢ fᵢ(x)` via ADMM.
    ///
    /// Each agent minimises its local function `fᵢ` while being driven towards
    /// a global consensus variable `z`.
    ///
    /// # Arguments
    /// * `local_f` - Vector of local objective functions
    /// * `x0` - Initial point (shared by all agents)
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if `local_f` is empty.
    pub fn solve_consensus(
        &self,
        local_f: Vec<Box<dyn Fn(&[f64]) -> f64>>,
        x0: Vec<f64>,
    ) -> Result<Vec<f64>, OptimizeError> {
        solve_consensus_admm(local_f, x0, self.rho, self.max_iter, self.tol_abs)
    }
}

// ─── LASSO via ADMM ──────────────────────────────────────────────────────────

/// Solve LASSO `min ½‖Ax−b‖² + λ‖x‖₁` via ADMM.
///
/// The x-update uses the closed-form ridge solution:
/// `x = (AᵀA + ρI)⁻¹ (Aᵀb + ρ(z − u))`
///
/// solved via coordinate descent on the normal equations for efficiency.
pub fn solve_lasso(a: &[Vec<f64>], b: &[f64], lambda: f64) -> Result<Vec<f64>, OptimizeError> {
    let solver = AdmmSolver::default();
    solver.solve_lasso(a, b, lambda)
}

fn solve_lasso_admm(
    a: &[Vec<f64>],
    b: &[f64],
    lambda: f64,
    rho: f64,
    max_iter: usize,
    tol_abs: f64,
    tol_rel: f64,
) -> Result<Vec<f64>, OptimizeError> {
    let m = a.len();
    if m == 0 {
        return Err(OptimizeError::ValueError(
            "Empty design matrix A".to_string(),
        ));
    }
    let n = a[0].len();
    if b.len() != m {
        return Err(OptimizeError::ValueError(format!(
            "A has {} rows but b has {} elements",
            m,
            b.len()
        )));
    }

    // Precompute AᵀA and Aᵀb
    let ata = mat_ata(a, n);
    let atb = mat_atv(a, b, n);

    // Precompute (AᵀA + ρI)
    let mut ata_rho = ata.clone();
    for i in 0..n {
        ata_rho[i * n + i] += rho;
    }

    // Factorise once (Cholesky)
    let chol = cholesky(&ata_rho, n)?;

    let mut x = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut u = vec![0.0; n]; // scaled dual variable

    for _iter in 0..max_iter {
        let x_prev = x.clone();

        // x-update: x = (AᵀA + ρI)⁻¹ (Aᵀb + ρ(z − u))
        let rhs: Vec<f64> = (0..n).map(|i| atb[i] + rho * (z[i] - u[i])).collect();
        x = chol_solve(&chol, &rhs, n)?;

        // z-update: z = prox_{(λ/ρ)‖·‖₁}(x + u)
        let xu: Vec<f64> = x.iter().zip(u.iter()).map(|(&xi, &ui)| xi + ui).collect();
        z = prox_l1(&xu, lambda / rho);

        // u-update: u = u + x − z
        for i in 0..n {
            u[i] += x[i] - z[i];
        }

        // Primal and dual residuals
        let primal_res: f64 = x
            .iter()
            .zip(z.iter())
            .map(|(&xi, &zi)| (xi - zi) * (xi - zi))
            .sum::<f64>()
            .sqrt();
        let dual_res: f64 = z
            .iter()
            .zip(x_prev.iter())
            .map(|(&zi, &xi)| rho * (zi - xi) * (zi - xi))
            .sum::<f64>()
            .sqrt();

        let norm_x: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let norm_z: f64 = z.iter().map(|&zi| zi * zi).sum::<f64>().sqrt();

        let eps_primal = (n as f64).sqrt() * tol_abs + tol_rel * norm_x.max(norm_z);
        let eps_dual = (n as f64).sqrt() * tol_abs
            + tol_rel * rho * u.iter().map(|&ui| ui * ui).sum::<f64>().sqrt();

        if primal_res < eps_primal && dual_res < eps_dual {
            return Ok(x);
        }
    }
    Ok(x)
}

// ─── Consensus ADMM ──────────────────────────────────────────────────────────

/// Solve consensus `min Σᵢ fᵢ(x)` via ADMM.
///
/// Each agent maintains its own local copy `xᵢ`; the consensus variable `z`
/// is the average. Local updates use gradient descent with fixed step size
/// derived from ρ.
pub fn solve_consensus(
    local_f: Vec<Box<dyn Fn(&[f64]) -> f64>>,
    x0: Vec<f64>,
) -> Result<Vec<f64>, OptimizeError> {
    let solver = AdmmSolver::default();
    solver.solve_consensus(local_f, x0)
}

fn solve_consensus_admm(
    local_f: Vec<Box<dyn Fn(&[f64]) -> f64>>,
    x0: Vec<f64>,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, OptimizeError> {
    let num_agents = local_f.len();
    if num_agents == 0 {
        return Err(OptimizeError::ValueError(
            "No local functions provided".to_string(),
        ));
    }
    let n = x0.len();

    // Each agent i has local copy x_i and dual variable u_i
    let mut xs: Vec<Vec<f64>> = vec![x0.clone(); num_agents];
    let mut z = x0.clone();
    let mut us: Vec<Vec<f64>> = vec![vec![0.0; n]; num_agents];

    let gd_steps = 20; // inner gradient descent iterations for x-update

    for _iter in 0..max_iter {
        let z_prev = z.clone();

        // x_i-update: argmin { f_i(x_i) + (ρ/2)‖x_i − z + u_i‖² }
        // Solved approximately with gradient descent
        for i in 0..num_agents {
            let lr_gd = 1.0 / (rho * 10.0); // conservative step
            for _step in 0..gd_steps {
                let g_approx = numerical_gradient_vec(&local_f[i], &xs[i]);
                for j in 0..n {
                    let aug_grad = g_approx[j] + rho * (xs[i][j] - z[j] + us[i][j]);
                    xs[i][j] -= lr_gd * aug_grad;
                }
            }
        }

        // z-update: z = (1/N) Σᵢ (x_i + u_i) — averaging
        for j in 0..n {
            z[j] = xs
                .iter()
                .zip(us.iter())
                .map(|(x, u)| x[j] + u[j])
                .sum::<f64>()
                / num_agents as f64;
        }

        // u_i-update
        for i in 0..num_agents {
            for j in 0..n {
                us[i][j] += xs[i][j] - z[j];
            }
        }

        // Convergence: ‖z − z_prev‖
        let dz: f64 = z
            .iter()
            .zip(z_prev.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        if dz < tol {
            return Ok(z);
        }
    }
    Ok(z)
}

/// Numerical gradient of a scalar function at `x`.
fn numerical_gradient_vec(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut xp = x.to_vec();
    let f0 = f(x);
    for i in 0..n {
        xp[i] += h;
        grad[i] = (f(&xp) - f0) / h;
        xp[i] = x[i];
    }
    grad
}

// ─── Linear Algebra Helpers ──────────────────────────────────────────────────

/// Compute AᵀA as a flat n×n row-major matrix.
fn mat_ata(a: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut ata = vec![0.0; n * n];
    let m = a.len();
    for k in 0..m {
        for i in 0..n {
            for j in 0..n {
                ata[i * n + j] += a[k][i] * a[k][j];
            }
        }
    }
    ata
}

/// Compute Aᵀv for a vector v.
fn mat_atv(a: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut atv = vec![0.0; n];
    for (row, &vi) in a.iter().zip(v.iter()) {
        for j in 0..n {
            atv[j] += row[j] * vi;
        }
    }
    atv
}

/// Cholesky factorisation of symmetric positive-definite n×n matrix (flat row-major).
/// Returns lower-triangular L as flat Vec<f64>.
fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, OptimizeError> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if s <= 0.0 {
                    return Err(OptimizeError::ComputationError(
                        "Cholesky: matrix not positive definite".to_string(),
                    ));
                }
                s.sqrt()
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < 1e-15 {
                    return Err(OptimizeError::ComputationError(
                        "Cholesky: near-zero diagonal".to_string(),
                    ));
                }
                s / ljj
            };
        }
    }
    Ok(l)
}

/// Solve L·Lᵀ·x = b using forward/backward substitution.
fn chol_solve(l: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, OptimizeError> {
    // Forward substitution: Ly = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < 1e-15 {
            return Err(OptimizeError::ComputationError(
                "chol_solve: near-zero diagonal".to_string(),
            ));
        }
        y[i] = s / lii;
    }
    // Backward substitution: Lᵀx = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < 1e-15 {
            return Err(OptimizeError::ComputationError(
                "chol_solve: near-zero diagonal".to_string(),
            ));
        }
        x[i] = s / lii;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn build_lasso_problem() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Simple 3×2 system: A = [[1,0],[0,1],[1,1]], b = [1,1,2]
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![1.0, 1.0, 2.0];
        (a, b)
    }

    #[test]
    fn test_solve_lasso_basic() {
        let (a, b) = build_lasso_problem();
        let result = solve_lasso(&a, &b, 0.01).expect("LASSO failed");
        assert_eq!(result.len(), 2);
        // Both components should be close to 1 for low lambda
        for &xi in &result {
            assert!(xi > 0.0, "LASSO solution should be positive");
            assert!(xi < 2.0, "LASSO solution should be bounded");
        }
    }

    #[test]
    fn test_solve_lasso_high_lambda_zeroes() {
        // Very high lambda → all coefficients near 0
        let (a, b) = build_lasso_problem();
        let result = solve_lasso(&a, &b, 100.0).expect("LASSO failed");
        for &xi in &result {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_admm_solver_lasso() {
        let solver = AdmmSolver::new(1.0, 500, 1e-4, 1e-2);
        let a = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let b = vec![2.0, 3.0];
        let result = solver.solve_lasso(&a, &b, 0.01).expect("ADMM LASSO failed");
        // Solution should be close to [1, 1]
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_solve_consensus_sum_of_quadratics() {
        // Each agent has f_i(x) = ½‖x − aᵢ‖²
        // Global minimum = mean of {aᵢ}
        let centers = vec![vec![1.0, 1.0], vec![3.0, 3.0], vec![5.0, 5.0]];
        let local_f: Vec<Box<dyn Fn(&[f64]) -> f64>> = centers
            .iter()
            .map(|c| {
                let c = c.clone();
                let f: Box<dyn Fn(&[f64]) -> f64> = Box::new(move |x: &[f64]| {
                    x.iter()
                        .zip(c.iter())
                        .map(|(&xi, &ci)| 0.5 * (xi - ci) * (xi - ci))
                        .sum()
                });
                f
            })
            .collect();

        let x0 = vec![0.0, 0.0];
        let result = solve_consensus(local_f, x0).expect("consensus failed");

        // Should converge towards mean = [3, 3]
        for &xi in &result {
            assert!(
                xi > 1.0 && xi < 5.0,
                "consensus solution out of range: {}",
                xi
            );
        }
    }

    #[test]
    fn test_admm_empty_local_f() {
        let result = solve_consensus(vec![], vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_dimension_mismatch() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![1.0, 2.0]; // wrong length
        let result = solve_lasso(&a, &b, 0.1);
        assert!(result.is_err());
    }
}
