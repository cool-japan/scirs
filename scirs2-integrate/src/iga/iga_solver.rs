//! Isogeometric Analysis (IGA) solver for elliptic BVPs on NURBS domains.
//!
//! The IGA philosophy: use the same NURBS basis functions for both the
//! geometry description and the solution approximation. This gives exact
//! geometry representation and high-order convergence for smooth problems.
//!
//! This module provides a 1-D IGA solver for the model problem:
//!
//! ```text
//! −(a(x) u')' = f(x)  on Ω = (0, 1)
//! u(0) = u_0,  u(1) = u_1   (Dirichlet BCs)
//! ```
//!
//! and a 2-D IGA solver on a rectangular parametric domain.
//!
//! ## Algorithm (1-D)
//!
//! 1. Construct B-spline basis {N_{i,p}} on [0,1].
//! 2. Assemble the stiffness matrix K[i,j] = ∫ a(x) N'_i N'_j dx
//!    and the load vector f[i] = ∫ f(x) N_i dx.
//! 3. Apply Dirichlet BCs by row/column elimination.
//! 4. Solve K u = f.
//!
//! ## References
//!
//! - Hughes, Cottrell & Bazilevs (2005), "Isogeometric Analysis: CAD, FEM, NURBS…"
//! - Cottrell, Hughes & Bazilevs (2009), "Isogeometric Analysis: Toward Integration…"

use super::bspline::BSplineBasis;
use crate::error::{IntegrateError, IntegrateResult};

// We need gaussian_elimination — pull it from panel_method via the BEM module
// but since iga is a separate module tree, we inline a small copy here.

// ---------------------------------------------------------------------------
// Inline Gaussian elimination (avoid cross-module dependency)
// ---------------------------------------------------------------------------

fn gauss_solve(a: &mut [Vec<f64>], b: &mut [f64], n: usize) -> IntegrateResult<Vec<f64>> {
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(IntegrateError::LinearSolveError(
                "Near-singular stiffness matrix in IGA solve".to_string(),
            ));
        }
        a.swap(col, max_row);
        b.swap(col, max_row);
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let sub = factor * a[col][k];
                a[row][k] -= sub;
            }
            let sub_b = factor * b[col];
            b[row] -= sub_b;
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature (element-level)
// ---------------------------------------------------------------------------

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.577_350_269_189_625_8, 0.577_350_269_189_625_8],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            vec![
                0.555_555_555_555_556,
                0.888_888_888_888_889,
                0.555_555_555_555_556,
            ],
        ),
        4 => (
            vec![
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            vec![
                0.347_854_845_137_453_8,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_8,
            ],
        ),
        _ => gauss_legendre(4),
    }
}

// ---------------------------------------------------------------------------
// IGA 1-D Solver
// ---------------------------------------------------------------------------

/// Configuration for the 1-D IGA solver.
#[derive(Debug, Clone)]
pub struct IGASolver1DConfig {
    /// Number of Gauss quadrature points per element (knot span).
    pub n_gauss: usize,
}

impl Default for IGASolver1DConfig {
    fn default() -> Self {
        Self { n_gauss: 4 }
    }
}

/// Solution of the 1-D IGA problem.
#[derive(Debug, Clone)]
pub struct IGASolution1D {
    /// B-spline coefficients (control variables) u_i.
    pub coefficients: Vec<f64>,
    /// Underlying basis.
    pub basis: BSplineBasis,
}

impl IGASolution1D {
    /// Evaluate the solution at a parameter t ∈ [0,1].
    pub fn eval(&self, t: f64) -> f64 {
        let (span, n_vals) = self.basis.eval_basis_functions(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);
        let mut val = 0.0_f64;
        for (k, &n_k) in n_vals.iter().enumerate() {
            let idx = start + k;
            if idx < self.coefficients.len() {
                val += n_k * self.coefficients[idx];
            }
        }
        val
    }

    /// Evaluate the derivative at t.
    pub fn eval_deriv(&self, t: f64) -> f64 {
        let (span, dn_vals) = self.basis.eval_basis_derivatives(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);
        let mut val = 0.0_f64;
        for (k, &dn_k) in dn_vals.iter().enumerate() {
            let idx = start + k;
            if idx < self.coefficients.len() {
                val += dn_k * self.coefficients[idx];
            }
        }
        val
    }
}

/// 1-D IGA solver for −(a u')' = f on [0,1] with Dirichlet BCs.
pub struct IGASolver1D {
    basis: BSplineBasis,
    cfg: IGASolver1DConfig,
}

impl IGASolver1D {
    /// Create a 1-D IGA solver.
    ///
    /// # Arguments
    ///
    /// * `degree` — B-spline degree p.
    /// * `n_elements` — Number of uniform knot spans.
    /// * `cfg` — Solver configuration.
    pub fn new(degree: usize, n_elements: usize, cfg: IGASolver1DConfig) -> IntegrateResult<Self> {
        let basis = BSplineBasis::uniform_open(degree, n_elements + degree)?;
        Ok(Self { basis, cfg })
    }

    /// Create from an explicit B-spline basis.
    pub fn from_basis(basis: BSplineBasis, cfg: IGASolver1DConfig) -> Self {
        Self { basis, cfg }
    }

    /// Assemble the global stiffness matrix K[i,j] = ∫ a(x) N'_i(x) N'_j(x) dx.
    fn assemble_stiffness<A>(&self, a_coeff: &A) -> Vec<Vec<f64>>
    where
        A: Fn(f64) -> f64,
    {
        let n = self.basis.n_basis;
        let mut k_mat = vec![vec![0.0_f64; n]; n];
        let knots = &self.basis.knots;
        let p = self.basis.degree;
        let (xi_ref, w_ref) = gauss_legendre(self.cfg.n_gauss);

        // Iterate over each knot span [t_i, t_{i+1}]
        for span_idx in p..=(n - 1) {
            let ta = knots[span_idx];
            let tb = knots[span_idx + 1];
            if (tb - ta).abs() < 1e-15 {
                continue; // Zero-length span (repeated knot)
            }
            let half = (tb - ta) * 0.5;
            let mid = (ta + tb) * 0.5;

            for (&xi, &w) in xi_ref.iter().zip(w_ref.iter()) {
                let t = mid + half * xi;
                let jac = half;

                let a_val = a_coeff(t);
                let (_, dn_vals) = self.basis.eval_basis_derivatives(t);
                // span from find_span corresponds to span_idx
                let start = span_idx.saturating_sub(p);

                for (ki, &dn_i) in dn_vals.iter().enumerate() {
                    let i = start + ki;
                    if i >= n {
                        continue;
                    }
                    for (kj, &dn_j) in dn_vals.iter().enumerate() {
                        let j = start + kj;
                        if j >= n {
                            continue;
                        }
                        k_mat[i][j] += a_val * dn_i * dn_j * w * jac;
                    }
                }
            }
        }
        k_mat
    }

    /// Assemble the global load vector f[i] = ∫ f(x) N_i(x) dx.
    fn assemble_load<F>(&self, f_rhs: &F) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.basis.n_basis;
        let mut f_vec = vec![0.0_f64; n];
        let knots = &self.basis.knots;
        let p = self.basis.degree;
        let (xi_ref, w_ref) = gauss_legendre(self.cfg.n_gauss);

        for span_idx in p..=(n - 1) {
            let ta = knots[span_idx];
            let tb = knots[span_idx + 1];
            if (tb - ta).abs() < 1e-15 {
                continue;
            }
            let half = (tb - ta) * 0.5;
            let mid = (ta + tb) * 0.5;

            for (&xi, &w) in xi_ref.iter().zip(w_ref.iter()) {
                let t = mid + half * xi;
                let jac = half;
                let f_val = f_rhs(t);
                let (_, n_vals) = self.basis.eval_basis_functions(t);
                let start = span_idx.saturating_sub(p);

                for (ki, &n_k) in n_vals.iter().enumerate() {
                    let i = start + ki;
                    if i < n {
                        f_vec[i] += f_val * n_k * w * jac;
                    }
                }
            }
        }
        f_vec
    }

    /// Solve −(a u')' = f with Dirichlet BCs u(0) = u0, u(1) = u1.
    ///
    /// # Arguments
    ///
    /// * `a_coeff` — Coefficient function a(x) ≥ ε > 0.
    /// * `f_rhs` — Right-hand side f(x).
    /// * `u0`, `u1` — Boundary values at x=0 and x=1.
    pub fn solve<A, F>(
        &self,
        a_coeff: &A,
        f_rhs: &F,
        u0: f64,
        u1: f64,
    ) -> IntegrateResult<IGASolution1D>
    where
        A: Fn(f64) -> f64,
        F: Fn(f64) -> f64,
    {
        let n = self.basis.n_basis;
        if n < 2 {
            return Err(IntegrateError::InvalidInput(
                "Need at least 2 basis functions".to_string(),
            ));
        }

        let mut k_mat = self.assemble_stiffness(a_coeff);
        let mut f_vec = self.assemble_load(f_rhs);

        // Apply Dirichlet BCs: strong enforcement via row/column elimination.
        // First DOF (index 0) corresponds to x=0, last (index n-1) to x=1.
        // We set u_0 = u0 and u_{n-1} = u1.

        // Modify load vector: subtract known boundary contributions
        for i in 1..(n - 1) {
            f_vec[i] -= k_mat[i][0] * u0 + k_mat[i][n - 1] * u1;
        }

        // Extract interior system (rows and cols 1..n-2)
        let n_free = n - 2;
        if n_free == 0 {
            // Only boundary DOFs: trivial
            let mut coeffs = vec![0.0_f64; n];
            coeffs[0] = u0;
            coeffs[n - 1] = u1;
            return Ok(IGASolution1D {
                coefficients: coeffs,
                basis: self.basis.clone(),
            });
        }

        let mut k_free = vec![vec![0.0_f64; n_free]; n_free];
        let mut f_free = vec![0.0_f64; n_free];
        for i in 0..n_free {
            f_free[i] = f_vec[i + 1];
            for j in 0..n_free {
                k_free[i][j] = k_mat[i + 1][j + 1];
            }
        }

        let u_free = gauss_solve(&mut k_free, &mut f_free, n_free)?;

        let mut coeffs = vec![0.0_f64; n];
        coeffs[0] = u0;
        coeffs[1..(n_free + 1)].copy_from_slice(&u_free[..n_free]);
        coeffs[n - 1] = u1;

        Ok(IGASolution1D {
            coefficients: coeffs,
            basis: self.basis.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// IGA 2-D Solver
// ---------------------------------------------------------------------------

/// 2-D IGA solution on a tensor-product B-spline domain.
#[derive(Debug, Clone)]
pub struct IGASolution2D {
    /// Control variables u_{ij} on the tensor-product mesh.
    pub coefficients: Vec<Vec<f64>>,
    /// B-spline basis in u direction.
    pub basis_u: BSplineBasis,
    /// B-spline basis in v direction.
    pub basis_v: BSplineBasis,
}

impl IGASolution2D {
    /// Evaluate the solution at (u, v).
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let (span_u, n_u) = self.basis_u.eval_basis_functions(u);
        let (span_v, n_v) = self.basis_v.eval_basis_functions(v);
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let start_u = span_u.saturating_sub(pu);
        let start_v = span_v.saturating_sub(pv);

        let mut val = 0.0_f64;
        for (ki, &n_ui) in n_u.iter().enumerate() {
            let i = start_u + ki;
            if i >= self.coefficients.len() {
                continue;
            }
            for (kj, &n_vj) in n_v.iter().enumerate() {
                let j = start_v + kj;
                if j < self.coefficients[i].len() {
                    val += n_ui * n_vj * self.coefficients[i][j];
                }
            }
        }
        val
    }
}

/// 2-D IGA solver for −∇²u = f on [0,1]² with homogeneous Dirichlet BCs.
pub struct IGASolver2D {
    basis_u: BSplineBasis,
    basis_v: BSplineBasis,
    cfg: IGASolver1DConfig,
}

impl IGASolver2D {
    /// Create a 2-D IGA solver on a uniform tensor-product mesh.
    pub fn new(
        degree: usize,
        n_elements_u: usize,
        n_elements_v: usize,
        cfg: IGASolver1DConfig,
    ) -> IntegrateResult<Self> {
        let basis_u = BSplineBasis::uniform_open(degree, n_elements_u + degree)?;
        let basis_v = BSplineBasis::uniform_open(degree, n_elements_v + degree)?;
        Ok(Self {
            basis_u,
            basis_v,
            cfg,
        })
    }

    /// Solve −∇²u = f with zero Dirichlet BCs on the entire boundary.
    ///
    /// Uses a tensor-product assembly: the global system has (n_u × n_v) DOFs.
    /// Interior DOFs are the free variables; boundary DOFs are set to zero.
    pub fn solve<F>(&self, f_rhs: &F) -> IntegrateResult<IGASolution2D>
    where
        F: Fn(f64, f64) -> f64,
    {
        let nu = self.basis_u.n_basis;
        let nv = self.basis_v.n_basis;
        let n_total = nu * nv;
        let knots_u = &self.basis_u.knots;
        let knots_v = &self.basis_v.knots;
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let (xi_ref, w_ref) = gauss_legendre(self.cfg.n_gauss);

        // Global index: idx(i, j) = i * nv + j
        let idx = |i: usize, j: usize| i * nv + j;

        // Mark boundary DOFs (zero Dirichlet)
        let is_boundary = |i: usize, j: usize| i == 0 || i == nu - 1 || j == 0 || j == nv - 1;

        let mut k_global = vec![vec![0.0_f64; n_total]; n_total];
        let mut f_global = vec![0.0_f64; n_total];

        // Assemble element by element (tensor product of 1-D spans)
        for span_i in pu..=(nu - 1) {
            let ua = knots_u[span_i];
            let ub = knots_u[span_i + 1];
            if (ub - ua).abs() < 1e-15 {
                continue;
            }
            let half_u = (ub - ua) * 0.5;
            let mid_u = (ua + ub) * 0.5;

            for span_j in pv..=(nv - 1) {
                let va = knots_v[span_j];
                let vb = knots_v[span_j + 1];
                if (vb - va).abs() < 1e-15 {
                    continue;
                }
                let half_v = (vb - va) * 0.5;
                let mid_v = (va + vb) * 0.5;

                // 2-D Gauss quadrature
                for (&xi_u, &w_u) in xi_ref.iter().zip(w_ref.iter()) {
                    let u_pt = mid_u + half_u * xi_u;
                    let (_, n_u) = self.basis_u.eval_basis_functions(u_pt);
                    let (_, dn_u) = self.basis_u.eval_basis_derivatives(u_pt);
                    let start_u = span_i.saturating_sub(pu);

                    for (&xi_v, &w_v) in xi_ref.iter().zip(w_ref.iter()) {
                        let v_pt = mid_v + half_v * xi_v;
                        let (_, n_v) = self.basis_v.eval_basis_functions(v_pt);
                        let (_, dn_v) = self.basis_v.eval_basis_derivatives(v_pt);
                        let start_v = span_j.saturating_sub(pv);

                        let jac = half_u * half_v * w_u * w_v;
                        let f_val = f_rhs(u_pt, v_pt);

                        // Assemble stiffness and load
                        for (ki, (&n_ui, &dn_ui)) in n_u.iter().zip(dn_u.iter()).enumerate() {
                            let i = start_u + ki;
                            if i >= nu {
                                continue;
                            }
                            for (kj, (&n_vj, &dn_vj)) in n_v.iter().zip(dn_v.iter()).enumerate() {
                                let j = start_v + kj;
                                if j >= nv {
                                    continue;
                                }
                                let row = idx(i, j);

                                // Load: ∫ f N_ij dΩ
                                f_global[row] += f_val * n_ui * n_vj * jac;

                                // Stiffness: ∫ ∇N_ij · ∇N_kl dΩ
                                for (ki2, (&n_ui2, &dn_ui2)) in
                                    n_u.iter().zip(dn_u.iter()).enumerate()
                                {
                                    let i2 = start_u + ki2;
                                    if i2 >= nu {
                                        continue;
                                    }
                                    for (kj2, (&n_vj2, &dn_vj2)) in
                                        n_v.iter().zip(dn_v.iter()).enumerate()
                                    {
                                        let j2 = start_v + kj2;
                                        if j2 >= nv {
                                            continue;
                                        }
                                        let col = idx(i2, j2);

                                        // ∇N_{ij} · ∇N_{i2j2} = N'_i N_j * N'_i2 N_j2 / Jac_u²
                                        //                       + N_i N'_j * N_i2 N'_j2 / Jac_v²
                                        // Note: dn_u is physical derivative only if Jacobian = half_u
                                        // Here the parametric domain IS the physical domain, so Jacobian = 1.
                                        let k_val = (dn_ui * n_vj) * (dn_ui2 * n_vj2)
                                            + (n_ui * dn_vj) * (n_ui2 * dn_vj2);
                                        k_global[row][col] += k_val * jac;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply Dirichlet BCs: zero on boundary
        for i in 0..nu {
            for j in 0..nv {
                if is_boundary(i, j) {
                    let row = idx(i, j);
                    for k in 0..n_total {
                        k_global[row][k] = 0.0;
                        k_global[k][row] = 0.0;
                    }
                    k_global[row][row] = 1.0;
                    f_global[row] = 0.0;
                }
            }
        }

        // Solve the system
        let u_flat = gauss_solve(&mut k_global, &mut f_global, n_total)?;

        // Reshape to 2-D
        let mut coeffs = vec![vec![0.0_f64; nv]; nu];
        for i in 0..nu {
            for j in 0..nv {
                coeffs[i][j] = u_flat[idx(i, j)];
            }
        }

        Ok(IGASolution2D {
            coefficients: coeffs,
            basis_u: self.basis_u.clone(),
            basis_v: self.basis_v.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Main IGASolver facade
// ---------------------------------------------------------------------------

/// Isogeometric Analysis solver (facade for 1-D and 2-D problems).
pub struct IGASolver;

impl IGASolver {
    /// Create a 1-D IGA solver.
    pub fn solver_1d(degree: usize, n_elements: usize) -> IntegrateResult<IGASolver1D> {
        IGASolver1D::new(degree, n_elements, IGASolver1DConfig::default())
    }

    /// Create a 2-D IGA solver.
    pub fn solver_2d(
        degree: usize,
        n_elements_u: usize,
        n_elements_v: usize,
    ) -> IntegrateResult<IGASolver2D> {
        IGASolver2D::new(
            degree,
            n_elements_u,
            n_elements_v,
            IGASolver1DConfig::default(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iga_1d_poisson_uniform() {
        // Solve −u'' = π² sin(πx) on [0,1], u(0)=u(1)=0.
        // Exact solution: u(x) = sin(πx).
        let solver = IGASolver1D::new(3, 8, IGASolver1DConfig { n_gauss: 4 })
            .expect("IGA 1D solver creation");

        let a = |_x: f64| 1.0_f64;
        let f =
            |x: f64| std::f64::consts::PI * std::f64::consts::PI * (std::f64::consts::PI * x).sin();

        let sol = solver.solve(&a, &f, 0.0, 0.0).expect("IGA 1D solve");

        // Check at interior points
        let test_pts = [0.25, 0.5, 0.75];
        for &x in &test_pts {
            let u_iga = sol.eval(x);
            let u_exact = (std::f64::consts::PI * x).sin();
            let err = (u_iga - u_exact).abs();
            assert!(
                err < 0.05,
                "IGA 1D u({x}) = {u_iga:.6}, exact = {u_exact:.6}, err = {err:.2e}"
            );
        }
    }

    #[test]
    fn test_iga_1d_variable_coeff() {
        // Solve −(2 u')' = −2 on [0,1], u(0)=0, u(1)=1.
        // ⟹ 2u'' = 2, u'' = 1.
        // General solution: u(x) = x²/2 + cx + d.
        // BCs: u(0)=0 ⟹ d=0; u(1)=1 ⟹ 1/2 + c = 1 ⟹ c = 1/2.
        // Exact: u(x) = x²/2 + x/2 = x(x+1)/2.
        let solver =
            IGASolver1D::new(2, 4, IGASolver1DConfig { n_gauss: 3 }).expect("IGA 1D creation");
        let a = |_x: f64| 2.0_f64;
        let f = |_x: f64| -2.0_f64;
        let sol = solver
            .solve(&a, &f, 0.0, 1.0)
            .expect("IGA 1D variable coeff solve");

        for k in 0..5 {
            let x = k as f64 * 0.2 + 0.1;
            let u_iga = sol.eval(x);
            let u_exact = x * (x + 1.0) / 2.0;
            let err = (u_iga - u_exact).abs();
            assert!(
                err < 0.05,
                "IGA 1D variable coeff u({x:.1}) = {u_iga:.6}, exact = {u_exact:.6}"
            );
        }
    }

    #[test]
    fn test_iga_1d_solution_derivative() {
        // Check that the derivative of the solution matches the exact derivative.
        let solver = IGASolver1D::new(3, 6, IGASolver1DConfig::default()).expect("IGA 1D creation");
        let a = |_x: f64| 1.0_f64;
        let pi = std::f64::consts::PI;
        let f = |x: f64| pi * pi * (pi * x).sin();
        let sol = solver.solve(&a, &f, 0.0, 0.0).expect("solve");

        let x = 0.3;
        let du_iga = sol.eval_deriv(x);
        let du_exact = pi * (pi * x).cos();
        let err = (du_iga - du_exact).abs();
        assert!(err < 0.3, "Derivative error = {err:.4} (too large)");
    }
}
