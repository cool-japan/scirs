//! Discontinuous Galerkin (DG) Method for hyperbolic conservation laws
//!
//! This module implements the Nodal Discontinuous Galerkin method for solving
//! 1D hyperbolic conservation laws of the form:
//!
//! ```text
//! ∂U/∂t + ∂F(U)/∂x = 0
//! ```
//!
//! The solution is represented as a piecewise polynomial on each element, using
//! Gauss-Legendre-Lobatto (GLL) nodes for interpolation. Numerical fluxes couple
//! adjacent elements and enforce the hyperbolic information flow direction.
//!
//! ## Algorithm Overview
//!
//! 1. Partition `[x_left, x_right]` into `n_elements` equal-width elements
//! 2. On each element use `poly_degree + 1` GLL nodes as degrees of freedom
//! 3. Evolve in time with the strong-form DG local operator + RK4 time integration
//! 4. Numerical flux at element interfaces: Lax-Friedrichs or upwind
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::dg::{Dg1dSolver, upwind_flux};
//! use std::f64::consts::PI;
//!
//! // Linear advection: ∂u/∂t + ∂u/∂x = 0
//! let mut solver = Dg1dSolver::new(0.0, 2.0 * PI, 16, 3).expect("solver creation failed");
//! solver.set_initial_condition(|x| x.sin());
//!
//! let flux     = |u: f64| u;         // F(u) = u  (linear advection)
//! let flux_d   = |_: f64| 1.0_f64;  // F'(u) = 1
//!
//! let t_end = solver.run_to(0.5, 0.5, flux, flux_d, 0.0, 0.0);
//! assert!(t_end >= 0.5);
//! ```

use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// GLL nodes and weights (up to degree 4, i.e., 5 nodes per element)
// ─────────────────────────────────────────────────────────────────────────────

/// Return Gauss-Legendre-Lobatto nodes on [-1, 1] for `n_nodes` points.
/// We support 1 through 6 nodes (polynomial degree 0–5).
fn gll_nodes(n_nodes: usize) -> IntegrateResult<Vec<f64>> {
    match n_nodes {
        1 => Ok(vec![0.0]),
        2 => Ok(vec![-1.0, 1.0]),
        3 => Ok(vec![-1.0, 0.0, 1.0]),
        4 => Ok(vec![-1.0, -1.0 / 5.0_f64.sqrt(), 1.0 / 5.0_f64.sqrt(), 1.0]),
        5 => Ok(vec![
            -1.0,
            -f64::sqrt(3.0 / 7.0),
            0.0,
            f64::sqrt(3.0 / 7.0),
            1.0,
        ]),
        6 => {
            let a = f64::sqrt(1.0 / 3.0 + 2.0 * f64::sqrt(7.0) / 21.0);
            let b = f64::sqrt(1.0 / 3.0 - 2.0 * f64::sqrt(7.0) / 21.0);
            Ok(vec![-1.0, -a, -b, b, a, 1.0])
        }
        _ => Err(IntegrateError::ValueError(format!(
            "DG: GLL nodes implemented for 1–6 nodes, got {}",
            n_nodes
        ))),
    }
}

/// Return GLL quadrature weights on [-1, 1] for `n_nodes` points.
fn gll_weights(n_nodes: usize) -> IntegrateResult<Vec<f64>> {
    match n_nodes {
        1 => Ok(vec![2.0]),
        2 => Ok(vec![1.0, 1.0]),
        3 => Ok(vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]),
        4 => Ok(vec![1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0]),
        5 => Ok(vec![
            1.0 / 10.0,
            49.0 / 90.0,
            32.0 / 45.0,
            49.0 / 90.0,
            1.0 / 10.0,
        ]),
        6 => {
            let w1 = 1.0 / 15.0;
            let w2 = (14.0 - f64::sqrt(7.0)) / 30.0;
            let w3 = (14.0 + f64::sqrt(7.0)) / 30.0;
            Ok(vec![w1, w2, w3, w3, w2, w1])
        }
        _ => Err(IntegrateError::ValueError(format!(
            "DG: GLL weights implemented for 1–6 nodes, got {}",
            n_nodes
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lagrange basis and differentiation matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate Lagrange basis polynomial l_j at point xi, given nodes `nodes`.
fn lagrange_basis(j: usize, xi: f64, nodes: &[f64]) -> f64 {
    let n = nodes.len();
    let mut val = 1.0;
    for k in 0..n {
        if k != j {
            let denom = nodes[j] - nodes[k];
            if denom.abs() < f64::EPSILON {
                return 0.0;
            }
            val *= (xi - nodes[k]) / denom;
        }
    }
    val
}

/// Compute the local differentiation matrix D_{ij} = l_j'(xi_i) on reference element.
/// D has shape (n_nodes x n_nodes).
fn differentiation_matrix(nodes: &[f64]) -> Vec<Vec<f64>> {
    let n = nodes.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Lagrange derivative formula for GLL nodes
                let mut num = 1.0;
                let mut den = nodes[j] - nodes[i];
                for k in 0..n {
                    if k != j {
                        den *= nodes[j] - nodes[k];
                    }
                    if k != i && k != j {
                        num *= nodes[i] - nodes[k];
                    }
                }
                d[i][j] = num / den;
            }
        }
        // Diagonal: use the sum condition D_{ii} = -Σ_{j≠i} D_{ij}
        let mut diag_sum = 0.0;
        for j in 0..n {
            if j != i {
                diag_sum += d[i][j];
            }
        }
        d[i][i] = -diag_sum;
    }
    d
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical flux functions (public API)
// ─────────────────────────────────────────────────────────────────────────────

/// Lax-Friedrichs numerical flux:
/// ```text
/// F*(u_L, u_R) = 0.5 (F(u_L) + F(u_R)) - 0.5 λ (u_R - u_L)
/// ```
/// where `λ` is the maximum wave speed.
pub fn lax_friedrichs_flux(
    u_left: f64,
    u_right: f64,
    flux: impl Fn(f64) -> f64,
    max_wave_speed: f64,
) -> f64 {
    0.5 * (flux(u_left) + flux(u_right)) - 0.5 * max_wave_speed * (u_right - u_left)
}

/// Upwind numerical flux for a scalar advection equation with characteristic speed `wave_speed`:
/// ```text
/// F*(u_L, u_R) = F(u_L) if wave_speed > 0, else F(u_R)
/// ```
pub fn upwind_flux(u_left: f64, u_right: f64, wave_speed: f64) -> f64 {
    if wave_speed >= 0.0 {
        u_left * wave_speed
    } else {
        u_right * wave_speed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nodal DG solver
// ─────────────────────────────────────────────────────────────────────────────

/// 1D Nodal Discontinuous Galerkin solver for scalar hyperbolic conservation laws.
///
/// Discretises `[x_left, x_right]` into `n_elements` elements, each containing
/// `n_nodes = poly_degree + 1` GLL interior nodes. The strong-form DG semi-discrete
/// operator is integrated in time with the classical RK4 scheme.
///
/// ## Memory layout
/// - `x[e][j]` — physical coordinate of node `j` in element `e`
/// - `u[e][j]` — solution value at that node
pub struct Dg1dSolver {
    /// Number of elements
    pub n_elements: usize,
    /// Nodes per element = polynomial degree + 1
    pub n_nodes: usize,
    /// Physical node coordinates: `x[element][local_node]`
    pub x: Vec<Vec<f64>>,
    /// Solution at nodes: `u[element][local_node]`
    pub u: Vec<Vec<f64>>,
    /// GLL reference nodes on [-1, 1]
    xi_ref: Vec<f64>,
    /// GLL quadrature weights on [-1, 1]
    weights: Vec<f64>,
    /// Local differentiation matrix D_{ij}
    d_mat: Vec<Vec<f64>>,
    /// Jacobian dx/dxi for each element (same for uniform mesh)
    jacobian: Vec<f64>,
    /// Inverse Jacobian
    inv_jac: Vec<f64>,
}

impl Dg1dSolver {
    /// Create a new DG solver on `[x_left, x_right]` with `n_elements` uniform elements
    /// and `poly_degree`-th order polynomials.
    ///
    /// # Errors
    /// Returns an error if `poly_degree` > 5 (GLL tables not available).
    pub fn new(
        x_left: f64,
        x_right: f64,
        n_elements: usize,
        poly_degree: usize,
    ) -> IntegrateResult<Self> {
        if n_elements == 0 {
            return Err(IntegrateError::ValueError(
                "DG: n_elements must be >= 1".into(),
            ));
        }
        let n_nodes = poly_degree + 1;
        let xi_ref = gll_nodes(n_nodes)?;
        let weights = gll_weights(n_nodes)?;
        let d_mat = differentiation_matrix(&xi_ref);

        let h = (x_right - x_left) / (n_elements as f64);
        let jac = h / 2.0; // dx/dxi for uniform elements
        let inv_jac = 1.0 / jac;

        // Build physical node coordinates
        let mut x = vec![vec![0.0_f64; n_nodes]; n_elements];
        for e in 0..n_elements {
            let x_l = x_left + (e as f64) * h;
            let x_r = x_l + h;
            for j in 0..n_nodes {
                // Map [-1, 1] -> [x_l, x_r]
                x[e][j] = 0.5 * (x_l + x_r) + 0.5 * (x_r - x_l) * xi_ref[j];
            }
        }

        let u = vec![vec![0.0_f64; n_nodes]; n_elements];
        let jacobian = vec![jac; n_elements];
        let inv_jacobian = vec![inv_jac; n_elements];

        Ok(Self {
            n_elements,
            n_nodes,
            x,
            u,
            xi_ref,
            weights,
            d_mat,
            jacobian,
            inv_jac: inv_jacobian,
        })
    }

    /// Set the initial condition by evaluating `ic(x)` at all nodes.
    pub fn set_initial_condition(&mut self, ic: impl Fn(f64) -> f64) {
        for e in 0..self.n_elements {
            for j in 0..self.n_nodes {
                self.u[e][j] = ic(self.x[e][j]);
            }
        }
    }

    /// Compute the DG spatial residual dU/dt for the current state `u_state`.
    ///
    /// Uses the strong-form DG operator:
    /// ```text
    /// dU_j/dt = -J^{-1} Σ_k D_{jk} F(U_k)  +  J^{-1} [flux_terms]
    /// ```
    fn residual(
        &self,
        u_state: &[Vec<f64>],
        flux: &dyn Fn(f64) -> f64,
        flux_deriv: &dyn Fn(f64) -> f64,
        bc_left: f64,
        bc_right: f64,
    ) -> Vec<Vec<f64>> {
        let ne = self.n_elements;
        let nn = self.n_nodes;
        let mut rhs = vec![vec![0.0_f64; nn]; ne];

        for e in 0..ne {
            // Volume term: -J^{-1} D^T F
            let inv_j = self.inv_jac[e];
            for j in 0..nn {
                let mut vol = 0.0;
                for k in 0..nn {
                    vol += self.d_mat[j][k] * flux(u_state[e][k]);
                }
                rhs[e][j] -= inv_j * vol;
            }

            // Interface fluxes
            // Left interface: between element e-1 and e
            let u_left_interface = if e == 0 {
                bc_left
            } else {
                u_state[e - 1][nn - 1]
            };
            let u_right_at_left = u_state[e][0];
            let a_l = flux_deriv(u_left_interface)
                .abs()
                .max(flux_deriv(u_right_at_left).abs());
            let f_num_left = lax_friedrichs_flux(u_left_interface, u_right_at_left, flux, a_l);

            // Right interface: between element e and e+1
            let u_left_at_right = u_state[e][nn - 1];
            let u_right_interface = if e == ne - 1 {
                bc_right
            } else {
                u_state[e + 1][0]
            };
            let a_r = flux_deriv(u_left_at_right)
                .abs()
                .max(flux_deriv(u_right_interface).abs());
            let f_num_right = lax_friedrichs_flux(u_left_at_right, u_right_interface, flux, a_r);

            // Surface terms (lift): add flux correction at endpoints
            // Using GLL mass-matrix diagonal (the weights)
            // Left endpoint: j=0
            let w0 = self.weights[0];
            rhs[e][0] -= inv_j * (f_num_left - flux(u_state[e][0])) / (w0 * self.jacobian[e] / 2.0)
                * (self.jacobian[e] / 2.0); // outward normal = -1 at left
                                            // Right endpoint: j=nn-1
            let wn = self.weights[nn - 1];
            rhs[e][nn - 1] += inv_j * (f_num_right - flux(u_state[e][nn - 1]))
                / (wn * self.jacobian[e] / 2.0)
                * (self.jacobian[e] / 2.0)
                * 1.0; // outward normal = +1 at right

            // Correct volume contribution sign (strong form: subtract numerical flux)
            rhs[e][0] -= inv_j * (flux(u_state[e][0]) - f_num_left);
            rhs[e][nn - 1] += inv_j * (f_num_right - flux(u_state[e][nn - 1]));
        }

        rhs
    }

    /// Perform a single RK4 time step with the given flux functions.
    pub fn step_rk4(
        &mut self,
        dt: f64,
        flux: impl Fn(f64) -> f64,
        flux_deriv: impl Fn(f64) -> f64,
        bc_left: f64,
        bc_right: f64,
    ) {
        let u0 = self.u.clone();

        // k1
        let k1 = self.residual(&u0, &flux, &flux_deriv, bc_left, bc_right);

        // u1 = u0 + 0.5*dt*k1
        let u1 = add_scaled(&u0, &k1, 0.5 * dt);
        let k2 = self.residual(&u1, &flux, &flux_deriv, bc_left, bc_right);

        // u2 = u0 + 0.5*dt*k2
        let u2 = add_scaled(&u0, &k2, 0.5 * dt);
        let k3 = self.residual(&u2, &flux, &flux_deriv, bc_left, bc_right);

        // u3 = u0 + dt*k3
        let u3 = add_scaled(&u0, &k3, dt);
        let k4 = self.residual(&u3, &flux, &flux_deriv, bc_left, bc_right);

        // Combine: u = u0 + dt/6 (k1 + 2k2 + 2k3 + k4)
        let ne = self.n_elements;
        let nn = self.n_nodes;
        for e in 0..ne {
            for j in 0..nn {
                self.u[e][j] =
                    u0[e][j] + dt / 6.0 * (k1[e][j] + 2.0 * k2[e][j] + 2.0 * k3[e][j] + k4[e][j]);
            }
        }
    }

    /// Run until `t_final` using the given flux, CFL condition, and boundary values.
    ///
    /// Returns the actual final time (may be slightly larger than `t_final`).
    pub fn run_to(
        &mut self,
        t_final: f64,
        cfl: f64,
        flux: impl Fn(f64) -> f64,
        flux_deriv: impl Fn(f64) -> f64,
        bc_left: f64,
        bc_right: f64,
    ) -> f64 {
        let h_min = self.x[0][1] - self.x[0][0]; // min node spacing
        let max_a = self
            .u
            .iter()
            .flat_map(|row| row.iter())
            .map(|&u| flux_deriv(u).abs())
            .fold(1.0_f64, f64::max);

        let dt_cfl = if max_a > 1e-14 {
            cfl * h_min / max_a
        } else {
            cfl * h_min
        };
        let dt = dt_cfl.min(t_final);

        let mut t = 0.0;
        while t < t_final {
            let dt_actual = dt.min(t_final - t);
            self.step_rk4(dt_actual, &flux, &flux_deriv, bc_left, bc_right);
            t += dt_actual;
        }
        t
    }

    /// Evaluate the DG solution at a point `x_eval` using its element's polynomial.
    pub fn evaluate_at(&self, x_eval: f64) -> f64 {
        // Find element
        let x_left = self.x[0][0];
        let x_right = self.x[self.n_elements - 1][self.n_nodes - 1];
        if x_eval <= x_left {
            return self.u[0][0];
        }
        if x_eval >= x_right {
            return self.u[self.n_elements - 1][self.n_nodes - 1];
        }
        let h = (x_right - x_left) / (self.n_elements as f64);
        let e = ((x_eval - x_left) / h).floor() as usize;
        let e = e.min(self.n_elements - 1);

        // Map x_eval to reference coordinate xi in [-1, 1]
        let x_l = self.x[e][0];
        let x_r = self.x[e][self.n_nodes - 1];
        let xi = 2.0 * (x_eval - x_l) / (x_r - x_l) - 1.0;

        // Evaluate interpolating polynomial at xi
        let mut val = 0.0;
        for j in 0..self.n_nodes {
            val += self.u[e][j] * lagrange_basis(j, xi, &self.xi_ref);
        }
        val
    }

    /// Sample the DG solution on a uniform grid of `n` points.
    ///
    /// Returns `(x_vals, u_vals)`.
    pub fn to_uniform_grid(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        let x_left = self.x[0][0];
        let x_right = self.x[self.n_elements - 1][self.n_nodes - 1];
        let mut xs = vec![0.0_f64; n];
        let mut us = vec![0.0_f64; n];
        for i in 0..n {
            let xp = x_left + (x_right - x_left) * (i as f64) / ((n - 1).max(1) as f64);
            xs[i] = xp;
            us[i] = self.evaluate_at(xp);
        }
        (xs, us)
    }

    /// Compute the global L2 error against an exact solution using GLL quadrature.
    pub fn l2_error(&self, exact: impl Fn(f64) -> f64) -> f64 {
        let mut err2 = 0.0;
        for e in 0..self.n_elements {
            for j in 0..self.n_nodes {
                let diff = self.u[e][j] - exact(self.x[e][j]);
                err2 += self.weights[j] * self.jacobian[e] * diff * diff;
            }
        }
        err2.sqrt()
    }

    /// Return reference to solution array.
    pub fn solution(&self) -> &Vec<Vec<f64>> {
        &self.u
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: u_new = u + alpha * k
// ─────────────────────────────────────────────────────────────────────────────

fn add_scaled(u: &[Vec<f64>], k: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    u.iter()
        .zip(k.iter())
        .map(|(row_u, row_k)| {
            row_u
                .iter()
                .zip(row_k.iter())
                .map(|(&ui, &ki)| ui + alpha * ki)
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gll_nodes_count() {
        for n in 1..=6 {
            let nodes = gll_nodes(n).expect("valid node count");
            assert_eq!(nodes.len(), n);
        }
    }

    #[test]
    fn test_gll_nodes_endpoints() {
        // GLL nodes always include ±1 for n >= 2
        for n in 2..=6 {
            let nodes = gll_nodes(n).expect("gll_nodes should succeed for valid n");
            assert!((nodes[0] + 1.0).abs() < 1e-12);
            assert!((nodes[n - 1] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gll_weights_sum() {
        // GLL weights sum to 2 (integral of 1 on [-1,1])
        for n in 1..=6 {
            let w = gll_weights(n).expect("gll_weights should succeed for valid n");
            let sum: f64 = w.iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={}: sum={}", n, sum);
        }
    }

    #[test]
    fn test_differentiation_matrix_constant() {
        // D * [1, 1, ..., 1] = 0 (derivative of constant is zero)
        let n = 4;
        let nodes = gll_nodes(n).expect("gll_nodes should succeed for n=4");
        let d = differentiation_matrix(&nodes);
        for i in 0..n {
            let row_sum: f64 = d[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "D row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_dg_linear_advection_mass_conservation() {
        // ∂u/∂t + ∂u/∂x = 0 on periodic-like domain, constant IC
        let mut solver = Dg1dSolver::new(0.0, 1.0, 4, 2)
            .expect("Dg1dSolver::new should succeed with valid params");
        solver.set_initial_condition(|_x| 1.0);
        let (_, u) = solver.to_uniform_grid(20);
        for &v in &u {
            assert!((v - 1.0).abs() < 1e-10, "constant should stay: v={}", v);
        }
    }

    #[test]
    fn test_dg_new_invalid_poly_degree() {
        let result = Dg1dSolver::new(0.0, 1.0, 4, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_dg_run_to_returns_final_time() {
        let mut solver = Dg1dSolver::new(0.0, 2.0 * PI, 8, 2)
            .expect("Dg1dSolver::new should succeed with valid params");
        solver.set_initial_condition(|x| x.sin());
        let t_end = solver.run_to(0.2, 0.5, |u| u, |_| 1.0, 0.0, 0.0);
        assert!(t_end >= 0.2 - 1e-10);
    }

    #[test]
    fn test_lax_friedrichs_symmetry() {
        let flux = |u: f64| u;
        let f = lax_friedrichs_flux(1.0, 2.0, flux, 2.0);
        assert!((f - 0.5).abs() < 1e-12, "f={}", f);
    }

    #[test]
    fn test_upwind_flux_positive_speed() {
        let f = upwind_flux(3.0, 5.0, 1.0);
        assert!((f - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_upwind_flux_negative_speed() {
        let f = upwind_flux(3.0, 5.0, -1.0);
        assert!((f + 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_dg_l2_error_zero_for_exact() {
        let mut solver = Dg1dSolver::new(0.0, 1.0, 4, 3)
            .expect("Dg1dSolver::new should succeed with valid params");
        solver.set_initial_condition(|x| x * x);
        let err = solver.l2_error(|x| x * x);
        assert!(err < 1e-12, "L2 error of exact IC should be ~0: {}", err);
    }

    #[test]
    fn test_dg_to_uniform_grid_length() {
        let solver = Dg1dSolver::new(0.0, 1.0, 4, 2)
            .expect("Dg1dSolver::new should succeed with valid params");
        let (xs, us) = solver.to_uniform_grid(33);
        assert_eq!(xs.len(), 33);
        assert_eq!(us.len(), 33);
    }
}
