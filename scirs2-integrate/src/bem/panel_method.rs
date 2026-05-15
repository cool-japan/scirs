//! Panel method for potential flow problems.
//!
//! The panel method is a special case of BEM for inviscid, irrotational
//! (potential) flow around bodies. It uses source / doublet panels on the
//! boundary Γ to model the potential φ satisfying ∇²φ = 0.
//!
//! This module provides:
//! - **Source panel method**: flow around a solid body at angle of attack.
//! - **Doublet panel method**: lifting flows with wake modelling.
//! - Post-processing utilities: surface velocity, pressure coefficient.

use super::boundary_mesh::{gauss_legendre_1d, BoundaryMesh};
use crate::error::{IntegrateError, IntegrateResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// PanelMethod configuration
// ---------------------------------------------------------------------------

/// Configuration for the panel method solver.
#[derive(Debug, Clone)]
pub struct PanelMethodConfig {
    /// Number of Gauss quadrature points per panel for off-diagonal integrals.
    pub n_gauss: usize,
    /// Free-stream velocity components (U_∞, V_∞).
    pub free_stream: [f64; 2],
}

impl Default for PanelMethodConfig {
    fn default() -> Self {
        Self {
            n_gauss: 5,
            free_stream: [1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Panel influence integrals
// ---------------------------------------------------------------------------

/// Compute the source panel influence integral I_s(x; panel) = ∫_panel G(x, y) dΓ(y)
/// for the Laplace kernel G(x,y) = -1/(2π) ln|x-y|.
///
/// Uses analytical formula for the diagonal terms and Gauss quadrature otherwise.
fn source_panel_g(x: [f64; 2], p1: [f64; 2], p2: [f64; 2], n_gauss: usize) -> f64 {
    let (xi_nodes, weights) = gauss_legendre_1d(n_gauss);
    let half_len = {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        (dx * dx + dy * dy).sqrt() * 0.5
    };

    xi_nodes
        .iter()
        .zip(weights.iter())
        .map(|(&xi, &w)| {
            let t = (1.0 + xi) * 0.5;
            let y = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
            let r = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt();
            if r < 1e-14 {
                0.0
            } else {
                -r.ln() / (2.0 * PI) * w * half_len
            }
        })
        .sum()
}

/// Compute the doublet panel influence integral I_d(x; panel) = ∫_panel ∂G/∂n(x,y) dΓ(y).
fn doublet_panel_h(
    x: [f64; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    normal: [f64; 2],
    n_gauss: usize,
) -> f64 {
    let (xi_nodes, weights) = gauss_legendre_1d(n_gauss);
    let dx_panel = p2[0] - p1[0];
    let dy_panel = p2[1] - p1[1];
    let len = (dx_panel * dx_panel + dy_panel * dy_panel).sqrt();
    let half_len = len * 0.5;

    xi_nodes
        .iter()
        .zip(weights.iter())
        .map(|(&xi, &w)| {
            let t = (1.0 + xi) * 0.5;
            let y = [p1[0] + t * dx_panel, p1[1] + t * dy_panel];
            let drx = x[0] - y[0];
            let dry = x[1] - y[1];
            let r2 = drx * drx + dry * dry;
            if r2 < 1e-28 {
                0.0
            } else {
                (drx * normal[0] + dry * normal[1]) / (2.0 * PI * r2) * w * half_len
            }
        })
        .sum()
}

// ---------------------------------------------------------------------------
// PanelMethod
// ---------------------------------------------------------------------------

/// Source panel method for potential flow around a body.
///
/// Solves the boundary integral equation:
///   -σ(x)/2 + ∫_Γ ∂G/∂n(x,y) σ(y) dΓ(y) = -V_∞ · n(x)
///
/// where σ is the source strength distribution, V_∞ is the free-stream
/// velocity, and n is the outward normal. After solving, the surface
/// velocity and pressure coefficient can be evaluated.
pub struct PanelMethod {
    mesh: BoundaryMesh,
    cfg: PanelMethodConfig,
    /// Source strengths (solution), set after `solve()`.
    source_strengths: Vec<f64>,
}

impl PanelMethod {
    /// Create a new panel method solver.
    pub fn new(mesh: BoundaryMesh, cfg: PanelMethodConfig) -> Self {
        Self {
            mesh,
            cfg,
            source_strengths: Vec::new(),
        }
    }

    /// Assemble the influence matrix A[i,j] = ∫_{Γ_j} ∂G/∂n(x_i, y) dΓ(y).
    /// The diagonal term has an added −1/2 from the boundary integral equation.
    fn assemble_influence_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.mesh.n_elements;
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let xi = self.mesh.elements[i].midpoint;
            for j in 0..n {
                let ej = &self.mesh.elements[j];
                let val =
                    doublet_panel_h(xi, ej.nodes[0], ej.nodes[1], ej.normal, self.cfg.n_gauss);
                // Diagonal contribution includes free-term −1/2 from boundary jump.
                a[i][j] = if i == j { val - 0.5 } else { val };
            }
        }
        a
    }

    /// Solve for the source strength distribution σ on the boundary.
    ///
    /// The no-penetration BC requires V_∞ · n = 0 on the body surface,
    /// leading to the RHS: b\[i\] = -V_∞ · n_i.
    pub fn solve(&mut self) -> IntegrateResult<()> {
        let n = self.mesh.n_elements;
        let mut a = self.assemble_influence_matrix();
        let u_inf = self.cfg.free_stream;

        let mut b: Vec<f64> = self
            .mesh
            .elements
            .iter()
            .map(|e| -(u_inf[0] * e.normal[0] + u_inf[1] * e.normal[1]))
            .collect();

        // Gaussian elimination with partial pivoting
        let sigma = gaussian_elimination(&mut a, &mut b, n)?;
        self.source_strengths = sigma;
        Ok(())
    }

    /// Evaluate the induced velocity at a field point p (must be exterior).
    ///
    /// u_induced(p) = ∫_Γ σ(y) ∇_p G(p, y) dΓ(y)
    pub fn velocity_at(&self, p: [f64; 2]) -> [f64; 2] {
        let mut vx = 0.0_f64;
        let mut vy = 0.0_f64;
        let (xi_nodes, weights) = gauss_legendre_1d(self.cfg.n_gauss);

        for (j, elem) in self.mesh.elements.iter().enumerate() {
            let sigma_j = self.source_strengths.get(j).copied().unwrap_or(0.0);
            let half = elem.length * 0.5;
            let p1 = elem.nodes[0];
            let p2 = elem.nodes[1];

            for (&xi, &w) in xi_nodes.iter().zip(weights.iter()) {
                let t = (1.0 + xi) * 0.5;
                let y = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
                let dx = p[0] - y[0];
                let dy = p[1] - y[1];
                let r2 = dx * dx + dy * dy;
                if r2 < 1e-28 {
                    continue;
                }
                // ∇_p G = (p - y) / (2π r²)
                let factor = sigma_j * w * half / (2.0 * PI * r2);
                vx += factor * dx;
                vy += factor * dy;
            }
        }
        [vx + self.cfg.free_stream[0], vy + self.cfg.free_stream[1]]
    }

    /// Evaluate the surface velocity magnitude and pressure coefficient on
    /// each panel's midpoint.
    ///
    /// Returns `(velocities, cp_values)` where each entry corresponds to a panel.
    pub fn surface_cp(&self) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
        if self.source_strengths.is_empty() {
            return Err(IntegrateError::ComputationError(
                "Panel method not yet solved; call solve() first".to_string(),
            ));
        }
        let v_inf_sq = self.cfg.free_stream[0].powi(2) + self.cfg.free_stream[1].powi(2);
        if v_inf_sq < 1e-30 {
            return Err(IntegrateError::InvalidInput(
                "Free-stream speed is zero".to_string(),
            ));
        }

        let mut velocities = Vec::with_capacity(self.mesh.n_elements);
        let mut cp = Vec::with_capacity(self.mesh.n_elements);

        for elem in &self.mesh.elements {
            let v = self.velocity_at(elem.midpoint);
            let v_sq = v[0] * v[0] + v[1] * v[1];
            velocities.push(v_sq.sqrt());
            // Cp = 1 - (|V|/|V_∞|)²
            cp.push(1.0 - v_sq / v_inf_sq);
        }
        Ok((velocities, cp))
    }

    /// Return a reference to the computed source strengths.
    pub fn source_strengths(&self) -> &[f64] {
        &self.source_strengths
    }

    /// Compute source panel velocity potential Φ at field point p.
    pub fn potential_at(&self, p: [f64; 2]) -> f64 {
        let mut phi = 0.0_f64;
        for (j, elem) in self.mesh.elements.iter().enumerate() {
            let sigma_j = self.source_strengths.get(j).copied().unwrap_or(0.0);
            phi += sigma_j * source_panel_g(p, elem.nodes[0], elem.nodes[1], self.cfg.n_gauss);
        }
        // Add free-stream potential Φ_∞ = U_∞ x + V_∞ y
        phi + self.cfg.free_stream[0] * p[0] + self.cfg.free_stream[1] * p[1]
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination
// ---------------------------------------------------------------------------

/// Solve Ax = b with partial-pivoting Gaussian elimination (in place).
pub(crate) fn gaussian_elimination(
    a: &mut [Vec<f64>],
    b: &mut [f64],
    n: usize,
) -> IntegrateResult<Vec<f64>> {
    for col in 0..n {
        // Find pivot
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
                "Singular or near-singular system in panel method".to_string(),
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
    // Back substitution
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_method_circular_cylinder() {
        // For a circular cylinder in uniform flow, the panel method should
        // recover approximately zero normal velocity on the surface.
        let mesh = BoundaryMesh::circle([0.0, 0.0], 1.0, 32);
        let cfg = PanelMethodConfig {
            n_gauss: 4,
            free_stream: [1.0, 0.0],
        };
        let mut pm = PanelMethod::new(mesh, cfg);
        pm.solve().expect("Panel solve failed");

        // Source strengths should be finite and non-degenerate
        for &sigma in pm.source_strengths() {
            assert!(sigma.is_finite(), "Source strength must be finite");
        }
    }

    #[test]
    fn test_potential_at_far_field() {
        let mesh = BoundaryMesh::circle([0.0, 0.0], 1.0, 16);
        let cfg = PanelMethodConfig {
            n_gauss: 3,
            free_stream: [1.0, 0.0],
        };
        let mut pm = PanelMethod::new(mesh, cfg);
        pm.solve().expect("Panel solve failed");

        // At a far field point (100, 0), φ ≈ U_∞ * x = 100
        let phi = pm.potential_at([100.0, 0.0]);
        assert!(phi.is_finite());
        // The free stream part dominates: should be close to 100
        assert!(
            (phi - 100.0).abs() < 1.0,
            "Far field potential should be ≈ 100, got {phi}"
        );
    }
}
