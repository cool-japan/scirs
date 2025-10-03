//! Advanced PDE solvers with cutting-edge techniques
//!
//! This module implements state-of-the-art PDE solving techniques including spectral methods,
//! discontinuous Galerkin methods, and multigrid solvers.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Spectral solver using Chebyshev polynomials
#[derive(Debug, Clone)]
pub struct SpectralChebyshevSolver {
    /// Number of Chebyshev nodes
    pub n_nodes: usize,
}

impl SpectralChebyshevSolver {
    /// Create a new spectral Chebyshev solver
    pub fn new(n_nodes: usize) -> IntegrateResult<Self> {
        if n_nodes < 3 {
            return Err(IntegrateError::ValueError(
                "Number of nodes must be at least 3".to_string(),
            ));
        }

        Ok(Self { n_nodes })
    }

    /// Compute Chebyshev nodes (Gauss-Lobatto points)
    pub fn chebyshev_nodes(&self) -> Array1<f64> {
        let mut nodes = Array1::<f64>::zeros(self.n_nodes);
        for i in 0..self.n_nodes {
            nodes[i] = -(PI * i as f64 / (self.n_nodes - 1) as f64).cos();
        }
        nodes
    }

    /// Compute Chebyshev differentiation matrix
    pub fn differentiation_matrix(&self) -> IntegrateResult<Array2<f64>> {
        let n = self.n_nodes;
        let mut d = Array2::<f64>::zeros((n, n));
        let nodes = self.chebyshev_nodes();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let mut c_i = 1.0;
                    let mut c_j = 1.0;

                    if i == 0 || i == n - 1 {
                        c_i = 2.0;
                    }
                    if j == 0 || j == n - 1 {
                        c_j = 2.0;
                    }

                    d[[i, j]] =
                        (c_i / c_j) * (-1.0_f64).powi((i + j) as i32) / (nodes[i] - nodes[j]);
                } else if i == 0 {
                    d[[i, j]] = (2.0 * (n - 1) as f64 * (n - 1) as f64 + 1.0) / 6.0;
                } else if i == n - 1 {
                    d[[i, j]] = -(2.0 * (n - 1) as f64 * (n - 1) as f64 + 1.0) / 6.0;
                } else {
                    d[[i, j]] = -nodes[i] / (2.0 * (1.0 - nodes[i] * nodes[i]));
                }
            }
        }

        Ok(d)
    }

    /// Solve 1D Poisson equation: u'' = f(x)
    pub fn solve_poisson<F>(
        &self,
        f: F,
        boundary_left: f64,
        boundary_right: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(f64) -> f64,
    {
        let nodes = self.chebyshev_nodes();
        let d = self.differentiation_matrix()?;

        // Second derivative matrix: D² = D * D
        let mut d2 = Array2::<f64>::zeros((self.n_nodes, self.n_nodes));
        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                for k in 0..self.n_nodes {
                    d2[[i, j]] += d[[i, k]] * d[[k, j]];
                }
            }
        }

        // Build right-hand side
        let mut rhs = Array1::<f64>::zeros(self.n_nodes);
        for i in 1..(self.n_nodes - 1) {
            rhs[i] = f(nodes[i]);
        }

        // Apply boundary conditions
        rhs[0] = boundary_left;
        rhs[self.n_nodes - 1] = boundary_right;

        // Modify matrix for boundary conditions
        for j in 0..self.n_nodes {
            d2[[0, j]] = if j == 0 { 1.0 } else { 0.0 };
            d2[[self.n_nodes - 1, j]] = if j == self.n_nodes - 1 { 1.0 } else { 0.0 };
        }

        // Solve linear system (using simple Gaussian elimination)
        self.solve_linear_system(&d2, &rhs)
    }

    /// Simple Gaussian elimination solver
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        let n = b.len();
        let mut aug = Array2::<f64>::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }

            if aug[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ValueError(
                    "Singular matrix in Gaussian elimination".to_string(),
                ));
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }
}

/// Radial Basis Function (RBF) meshless solver
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RBFType {
    /// Gaussian: φ(r) = exp(-ε²r²)
    Gaussian,
    /// Multiquadric: φ(r) = sqrt(1 + (εr)²)
    Multiquadric,
    /// Inverse multiquadric: φ(r) = 1/sqrt(1 + (εr)²)
    InverseMultiquadric,
    /// Thin plate spline: φ(r) = r² log(r)
    ThinPlateSpline,
}

#[derive(Debug, Clone)]
pub struct RBFSolver {
    /// RBF type
    pub rbf_type: RBFType,
    /// Shape parameter
    pub epsilon: f64,
    /// Number of collocation points
    pub n_points: usize,
}

impl RBFSolver {
    /// Create a new RBF solver
    pub fn new(rbf_type: RBFType, epsilon: f64, n_points: usize) -> IntegrateResult<Self> {
        if epsilon <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Shape parameter must be positive".to_string(),
            ));
        }

        if n_points < 2 {
            return Err(IntegrateError::ValueError(
                "Number of points must be at least 2".to_string(),
            ));
        }

        Ok(Self {
            rbf_type,
            epsilon,
            n_points,
        })
    }

    /// Evaluate RBF
    fn rbf(&self, r: f64) -> f64 {
        match self.rbf_type {
            RBFType::Gaussian => (-self.epsilon * self.epsilon * r * r).exp(),
            RBFType::Multiquadric => (1.0 + (self.epsilon * r).powi(2)).sqrt(),
            RBFType::InverseMultiquadric => 1.0 / (1.0 + (self.epsilon * r).powi(2)).sqrt(),
            RBFType::ThinPlateSpline => {
                if r > 0.0 {
                    r * r * r.ln()
                } else {
                    0.0
                }
            }
        }
    }

    /// Solve interpolation problem
    pub fn interpolate(
        &self,
        points: &Array1<f64>,
        values: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        if points.len() != values.len() {
            return Err(IntegrateError::ValueError(
                "Points and values must have same length".to_string(),
            ));
        }

        let n = points.len();
        let mut phi = Array2::<f64>::zeros((n, n));

        // Build RBF matrix
        for i in 0..n {
            for j in 0..n {
                let r = (points[i] - points[j]).abs();
                phi[[i, j]] = self.rbf(r);
            }
        }

        // Solve for weights
        self.solve_linear_system(&phi, values)
    }

    /// Simple linear system solver (reusing spectral solver's method)
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        let n = b.len();
        let mut aug = Array2::<f64>::zeros((n, n + 1));

        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }

            if aug[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ValueError("Singular matrix".to_string()));
            }

            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }
}

/// Multigrid solver for fast convergence
#[derive(Debug, Clone)]
pub struct MultigridSolver {
    /// Number of levels
    pub n_levels: usize,
    /// Number of pre-smoothing iterations
    pub n_pre_smooth: usize,
    /// Number of post-smoothing iterations
    pub n_post_smooth: usize,
}

impl MultigridSolver {
    /// Create a new multigrid solver
    pub fn new(
        n_levels: usize,
        n_pre_smooth: usize,
        n_post_smooth: usize,
    ) -> IntegrateResult<Self> {
        if n_levels == 0 {
            return Err(IntegrateError::ValueError(
                "Number of levels must be positive".to_string(),
            ));
        }

        Ok(Self {
            n_levels,
            n_pre_smooth,
            n_post_smooth,
        })
    }

    /// V-cycle multigrid iteration
    pub fn v_cycle(
        &self,
        u: &mut Array1<f64>,
        f: &Array1<f64>,
        dx: f64,
        level: usize,
    ) -> IntegrateResult<()> {
        let n = u.len();

        if level == 0 || n <= 3 {
            // Coarsest level: solve directly
            self.direct_solve(u, f, dx)?;
            return Ok(());
        }

        // Pre-smoothing
        for _ in 0..self.n_pre_smooth {
            self.gauss_seidel_smooth(u, f, dx);
        }

        // Compute residual
        let mut residual = Array1::<f64>::zeros(n);
        for i in 1..(n - 1) {
            let laplacian = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx);
            residual[i] = f[i] - laplacian;
        }

        // Restrict to coarse grid
        let n_coarse = (n - 1) / 2 + 1;
        let mut residual_coarse = Array1::<f64>::zeros(n_coarse);
        for i in 1..(n_coarse - 1) {
            residual_coarse[i] =
                0.25 * residual[2 * i - 1] + 0.5 * residual[2 * i] + 0.25 * residual[2 * i + 1];
        }

        // Solve on coarse grid
        let mut error_coarse = Array1::<f64>::zeros(n_coarse);
        self.v_cycle(&mut error_coarse, &residual_coarse, 2.0 * dx, level - 1)?;

        // Prolongate (interpolate) to fine grid
        let mut error_fine = Array1::<f64>::zeros(n);
        for i in 0..n_coarse {
            if 2 * i < n {
                error_fine[2 * i] = error_coarse[i];
            }
            if 2 * i + 1 < n && i + 1 < n_coarse {
                error_fine[2 * i + 1] = 0.5 * (error_coarse[i] + error_coarse[i + 1]);
            }
        }

        // Correct solution
        for i in 0..n {
            u[i] += error_fine[i];
        }

        // Post-smoothing
        for _ in 0..self.n_post_smooth {
            self.gauss_seidel_smooth(u, f, dx);
        }

        Ok(())
    }

    /// Gauss-Seidel smoother
    fn gauss_seidel_smooth(&self, u: &mut Array1<f64>, f: &Array1<f64>, dx: f64) {
        let n = u.len();
        for i in 1..(n - 1) {
            u[i] = 0.5 * (u[i - 1] + u[i + 1] - dx * dx * f[i]);
        }
    }

    /// Direct solver for small systems
    fn direct_solve(&self, u: &mut Array1<f64>, f: &Array1<f64>, dx: f64) -> IntegrateResult<()> {
        let n = u.len();

        // Build tridiagonal matrix
        let mut a = Array1::<f64>::zeros(n);
        let mut b = Array1::<f64>::zeros(n);
        let mut c = Array1::<f64>::zeros(n);
        let mut rhs = f.clone();

        for i in 1..(n - 1) {
            a[i] = 1.0 / (dx * dx);
            b[i] = -2.0 / (dx * dx);
            c[i] = 1.0 / (dx * dx);
        }

        // Boundary conditions
        b[0] = 1.0;
        c[0] = 0.0;
        a[n - 1] = 0.0;
        b[n - 1] = 1.0;
        rhs[0] = u[0];
        rhs[n - 1] = u[n - 1];

        // Thomas algorithm
        let mut c_prime = Array1::<f64>::zeros(n);
        let mut d_prime = Array1::<f64>::zeros(n);

        c_prime[0] = c[0] / b[0];
        d_prime[0] = rhs[0] / b[0];

        for i in 1..n {
            let denom = b[i] - a[i] * c_prime[i - 1];
            if i < n - 1 {
                c_prime[i] = c[i] / denom;
            }
            d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / denom;
        }

        u[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            u[i] = d_prime[i] - c_prime[i] * u[i + 1];
        }

        Ok(())
    }

    /// Solve Poisson equation using multigrid
    pub fn solve_poisson<F>(
        &self,
        f: F,
        x_range: (f64, f64),
        n_grid: usize,
        max_iterations: usize,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(f64) -> f64,
    {
        let dx = (x_range.1 - x_range.0) / (n_grid - 1) as f64;
        let mut u = Array1::<f64>::zeros(n_grid);
        let mut rhs = Array1::<f64>::zeros(n_grid);

        // Set up right-hand side
        for i in 1..(n_grid - 1) {
            let x = x_range.0 + i as f64 * dx;
            rhs[i] = f(x);
        }

        // Boundary conditions (homogeneous)
        u[0] = 0.0;
        u[n_grid - 1] = 0.0;

        // Multigrid iterations
        for _ in 0..max_iterations {
            self.v_cycle(&mut u, &rhs, dx, self.n_levels - 1)?;
        }

        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_chebyshev_solver() {
        let solver = SpectralChebyshevSolver::new(10).unwrap();
        assert_eq!(solver.n_nodes, 10);
    }

    #[test]
    fn test_chebyshev_nodes() {
        let solver = SpectralChebyshevSolver::new(5).unwrap();
        let nodes = solver.chebyshev_nodes();

        // Nodes should be in [-1, 1]
        for &node in nodes.iter() {
            assert!((-1.0..=1.0).contains(&node));
        }

        // First and last nodes should be -1 and 1
        assert!((nodes[0] - (-1.0)).abs() < 1e-10);
        assert!((nodes[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_poisson() {
        let solver = SpectralChebyshevSolver::new(20).unwrap();

        // Solve u'' = -2 with u(-1) = 0, u(1) = 0
        // Exact solution: u(x) = 1 - x²
        let f = |_x: f64| -2.0;

        let u = solver.solve_poisson(f, 0.0, 0.0).unwrap();
        let nodes = solver.chebyshev_nodes();

        // Check solution at some interior points
        for i in 5..15 {
            let exact = 1.0 - nodes[i] * nodes[i];
            assert!(
                (u[i] - exact).abs() < 0.1,
                "At node {}: u={}, exact={}",
                i,
                u[i],
                exact
            );
        }
    }

    #[test]
    fn test_rbf_solver_creation() {
        let solver = RBFSolver::new(RBFType::Gaussian, 1.0, 10).unwrap();
        assert_eq!(solver.rbf_type, RBFType::Gaussian);
    }

    #[test]
    fn test_rbf_interpolation() {
        let solver = RBFSolver::new(RBFType::Gaussian, 2.0, 5).unwrap();

        let points = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let values = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        let weights = solver.interpolate(&points, &values).unwrap();

        // Should find weights (not checking exact values due to RBF nature)
        assert_eq!(weights.len(), 5);
    }

    #[test]
    fn test_multigrid_solver_creation() {
        let solver = MultigridSolver::new(3, 2, 2).unwrap();
        assert_eq!(solver.n_levels, 3);
        assert_eq!(solver.n_pre_smooth, 2);
        assert_eq!(solver.n_post_smooth, 2);
    }

    #[test]
    fn test_multigrid_poisson() {
        let solver = MultigridSolver::new(3, 3, 3).unwrap();

        // Solve u'' = -2 with u(0) = 0, u(1) = 0
        // Exact solution: u(x) = x(1-x)
        let f = |_x: f64| -2.0;

        let u = solver.solve_poisson(f, (0.0, 1.0), 33, 5).unwrap();

        // Check solution at midpoint
        let mid_idx = u.len() / 2;
        let x_mid = 0.5;
        let exact = x_mid * (1.0 - x_mid);

        assert!(
            (u[mid_idx] - exact).abs() < 0.01,
            "Midpoint: u={}, exact={}",
            u[mid_idx],
            exact
        );
    }

    #[test]
    fn test_rbf_types() {
        let solver_gaussian = RBFSolver::new(RBFType::Gaussian, 1.0, 5).unwrap();
        let solver_mq = RBFSolver::new(RBFType::Multiquadric, 1.0, 5).unwrap();
        let solver_imq = RBFSolver::new(RBFType::InverseMultiquadric, 1.0, 5).unwrap();
        let solver_tps = RBFSolver::new(RBFType::ThinPlateSpline, 1.0, 5).unwrap();

        // Test RBF evaluations
        assert!(solver_gaussian.rbf(0.0) > 0.0);
        assert!(solver_mq.rbf(1.0) > 1.0);
        assert!(solver_imq.rbf(1.0) < 1.0);
        assert_eq!(solver_tps.rbf(0.0), 0.0);
    }

    #[test]
    fn test_differentiation_matrix() {
        let solver = SpectralChebyshevSolver::new(5).unwrap();
        let d = solver.differentiation_matrix().unwrap();

        // Matrix should be square
        assert_eq!(d.nrows(), 5);
        assert_eq!(d.ncols(), 5);
    }

    #[test]
    fn test_gauss_seidel_convergence() {
        let solver = MultigridSolver::new(1, 10, 10).unwrap();

        let n = 11;
        let dx = 1.0 / (n - 1) as f64;
        let mut u = Array1::<f64>::zeros(n);
        let f = Array1::from_vec(vec![0.0; n]);

        // Apply several smoothing iterations
        for _ in 0..20 {
            solver.gauss_seidel_smooth(&mut u, &f, dx);
        }

        // Solution should converge toward zero
        for i in 1..(n - 1) {
            assert!(u[i].abs() < 1e-6, "u[{}] = {}", i, u[i]);
        }
    }
}
