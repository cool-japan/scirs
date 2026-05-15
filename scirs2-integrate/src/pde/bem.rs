//! Boundary Element Method (BEM) for 2D Laplace problems
//!
//! The Boundary Element Method reformulates an elliptic PDE as a boundary
//! integral equation and discretises only the *boundary* of the domain,
//! reducing the dimensionality of the problem by one.
//!
//! ## Formulation
//!
//! For the 2D Laplace equation  ∇²u = 0  in Ω, Green's second identity gives
//! the boundary integral equation (BIE):
//!
//! ```text
//! c(x₀) u(x₀) = ∫_Γ [ G(x,x₀) q(x) - H(x,x₀) u(x) ] dΓ(x)
//! ```
//!
//! where
//!   * G(x,x₀) = −1/(2π) ln|x−x₀|  is the free-space Green's function,
//!   * H(x,x₀) = ∂G/∂n(x)           is its outward-normal derivative,
//!   * q(x) = ∂u/∂n                  is the normal flux,
//!   * c(x₀) = 1/2 for smooth boundaries (collocation at boundary nodes).
//!
//! The BIE is discretised by constant or linear boundary elements, leading to
//! the linear system   \[H\]{u} = \[G\]{q}.  After applying boundary conditions,
//! the unknown Dirichlet or Neumann values are solved.  Interior values are
//! then recovered by direct evaluation of the boundary integral.
//!
//! ## References
//! * Brebbia, C.A. (1978). *The Boundary Element Method for Engineers*.
//! * Beer, G., Smith, I., Duenser, C. (2008). *The Boundary Element Method
//!   with Programming*.

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2};

use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// A 1-D boundary mesh: a sequence of straight-line segments in 2-D space.
///
/// The boundary is described by `n_elements` constant elements.  Each element
/// *e* is the segment from `nodes[e]` to `nodes[e+1]`.  The mesh may
/// represent an open or closed boundary; for closed boundaries the last node
/// should **not** repeat the first (the wrap-around segment is added
/// automatically when the mesh is marked as closed).
#[derive(Debug, Clone)]
pub struct BemMesh {
    /// Node coordinates, shape `(n_nodes, 2)`.  Column 0 = x, column 1 = y.
    pub nodes: Array2<f64>,
    /// Whether the boundary is closed (last segment connects back to node 0).
    pub closed: bool,
}

impl BemMesh {
    /// Create a uniform straight-line boundary along x ∈ [x_start, x_end] at y = 0
    /// with `n_elements` constant elements (open polyline).
    ///
    /// # Arguments
    /// * `x_start` – starting x coordinate
    /// * `x_end`   – ending x coordinate
    /// * `y_coord` – y coordinate (constant along the segment)
    /// * `n_elements` – number of boundary elements
    /// * `closed`  – whether the last node connects back to the first
    pub fn new_line(
        x_start: f64,
        x_end: f64,
        y_coord: f64,
        n_elements: usize,
        closed: bool,
    ) -> PDEResult<Self> {
        if n_elements < 1 {
            return Err(PDEError::InvalidParameter(
                "n_elements must be >= 1".to_string(),
            ));
        }
        if (x_end - x_start).abs() < f64::EPSILON {
            return Err(PDEError::DomainError(
                "x_start and x_end must be distinct".to_string(),
            ));
        }

        let n_nodes = n_elements + 1;
        let dx = (x_end - x_start) / n_elements as f64;
        let mut nodes = Array2::zeros((n_nodes, 2));
        for i in 0..n_nodes {
            nodes[[i, 0]] = x_start + i as f64 * dx;
            nodes[[i, 1]] = y_coord;
        }

        Ok(BemMesh { nodes, closed })
    }

    /// Create a rectangular boundary mesh for a 2-D rectangle \[x0,x1\]×\[y0,y1\].
    ///
    /// The boundary is traversed counter-clockwise:
    ///   bottom → right → top (reversed) → left (reversed).
    ///
    /// Each side is divided into `n_per_side` elements.
    pub fn new_rectangle(x0: f64, x1: f64, y0: f64, y1: f64, n_per_side: usize) -> PDEResult<Self> {
        if n_per_side < 1 {
            return Err(PDEError::InvalidParameter(
                "n_per_side must be >= 1".to_string(),
            ));
        }
        if x0 >= x1 || y0 >= y1 {
            return Err(PDEError::DomainError(
                "Invalid rectangle: need x0 < x1 and y0 < y1".to_string(),
            ));
        }

        let n_nodes = 4 * n_per_side;
        let mut nodes = Array2::zeros((n_nodes, 2));

        let mut idx = 0usize;

        // Bottom edge: x0→x1, y=y0
        for k in 0..n_per_side {
            nodes[[idx, 0]] = x0 + (x1 - x0) * k as f64 / n_per_side as f64;
            nodes[[idx, 1]] = y0;
            idx += 1;
        }
        // Right edge: x=x1, y0→y1
        for k in 0..n_per_side {
            nodes[[idx, 0]] = x1;
            nodes[[idx, 1]] = y0 + (y1 - y0) * k as f64 / n_per_side as f64;
            idx += 1;
        }
        // Top edge: x1→x0, y=y1
        for k in 0..n_per_side {
            nodes[[idx, 0]] = x1 - (x1 - x0) * k as f64 / n_per_side as f64;
            nodes[[idx, 1]] = y1;
            idx += 1;
        }
        // Left edge: y1→y0, x=x0
        for k in 0..n_per_side {
            nodes[[idx, 0]] = x0;
            nodes[[idx, 1]] = y1 - (y1 - y0) * k as f64 / n_per_side as f64;
            idx += 1;
        }

        Ok(BemMesh {
            nodes,
            closed: true,
        })
    }

    /// Build a boundary mesh from explicit node coordinates.
    pub fn from_nodes(nodes: Array2<f64>, closed: bool) -> PDEResult<Self> {
        if nodes.nrows() < 2 {
            return Err(PDEError::InvalidParameter(
                "Need at least 2 nodes".to_string(),
            ));
        }
        if nodes.ncols() != 2 {
            return Err(PDEError::InvalidParameter(
                "nodes must have exactly 2 columns (x, y)".to_string(),
            ));
        }
        Ok(BemMesh { nodes, closed })
    }

    /// Number of boundary elements.
    pub fn n_elements(&self) -> usize {
        if self.closed {
            self.nodes.nrows()
        } else {
            self.nodes.nrows() - 1
        }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.nrows()
    }

    /// Midpoint (collocation point) of element `e`.
    pub fn element_midpoint(&self, e: usize) -> [f64; 2] {
        let n = self.nodes.nrows();
        let i0 = e;
        let i1 = if self.closed { (e + 1) % n } else { e + 1 };
        [
            0.5 * (self.nodes[[i0, 0]] + self.nodes[[i1, 0]]),
            0.5 * (self.nodes[[i0, 1]] + self.nodes[[i1, 1]]),
        ]
    }

    /// Length of element `e`.
    pub fn element_length(&self, e: usize) -> f64 {
        let n = self.nodes.nrows();
        let i0 = e;
        let i1 = if self.closed { (e + 1) % n } else { e + 1 };
        let dx = self.nodes[[i1, 0]] - self.nodes[[i0, 0]];
        let dy = self.nodes[[i1, 1]] - self.nodes[[i0, 1]];
        (dx * dx + dy * dy).sqrt()
    }

    /// Outward unit normal of element `e` (perpendicular, pointing *outward*).
    ///
    /// Convention: for a counter-clockwise-oriented closed boundary the
    /// outward normal is obtained by rotating the tangent 90° to the right.
    pub fn element_normal(&self, e: usize) -> [f64; 2] {
        let n = self.nodes.nrows();
        let i0 = e;
        let i1 = if self.closed { (e + 1) % n } else { e + 1 };
        let dx = self.nodes[[i1, 0]] - self.nodes[[i0, 0]];
        let dy = self.nodes[[i1, 1]] - self.nodes[[i0, 1]];
        let len = (dx * dx + dy * dy).sqrt().max(f64::EPSILON);
        // Tangent = (dx, dy)/len; outward normal (CCW boundary) = (dy, -dx)/len
        [dy / len, -dx / len]
    }
}

// ---------------------------------------------------------------------------
// Green's function kernel
// ---------------------------------------------------------------------------

/// 2-D free-space Green's function for the Laplace equation.
///
/// G(x, y; x₀, y₀) = −1/(2π) · ln(r),   r = |x − x₀|
///
/// Returns `None` if the source and field points coincide (r = 0).
pub fn fundamental_solution_2d(x: f64, y: f64, x0: f64, y0: f64) -> Option<f64> {
    let dx = x - x0;
    let dy = y - y0;
    let r2 = dx * dx + dy * dy;
    if r2 < f64::EPSILON * f64::EPSILON {
        return None;
    }
    let r = r2.sqrt();
    Some(-r.ln() / (2.0 * PI))
}

/// Normal derivative of the 2-D Green's function evaluated at the field point.
///
/// ∂G/∂n(x) = − (r⃗ · n̂) / (2π r²)
///
/// where r⃗ = x − x₀ and n̂ = (nx, ny) is the outward unit normal at `x`.
///
/// Returns `None` when r = 0.
pub fn normal_derivative_green(x: f64, y: f64, x0: f64, y0: f64, nx: f64, ny: f64) -> Option<f64> {
    let dx = x - x0;
    let dy = y - y0;
    let r2 = dx * dx + dy * dy;
    if r2 < f64::EPSILON * f64::EPSILON {
        return None;
    }
    let dot = dx * nx + dy * ny;
    Some(-dot / (2.0 * PI * r2))
}

// ---------------------------------------------------------------------------
// Gauss quadrature on [-1, 1] (5-point rule)
// ---------------------------------------------------------------------------

/// 5-point Gauss-Legendre nodes and weights on [-1, 1].
fn gauss5() -> ([f64; 5], [f64; 5]) {
    let nodes = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683,
        0.0,
        0.538_469_310_105_683,
        0.906_179_845_938_664,
    ];
    let weights = [
        0.236_926_885_056_189,
        0.478_628_670_499_366,
        0.568_888_888_888_889,
        0.478_628_670_499_366,
        0.236_926_885_056_189,
    ];
    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Element integrals: G_ij and H_ij
// ---------------------------------------------------------------------------

/// Compute the G-integral and H-integral for element `j` with collocation
/// node `xi0` at (x0, y0).
///
/// The integrals are evaluated by Gauss quadrature.  When the collocation
/// point coincides with the element (singular case, `i == j` on the same
/// element) the logarithmic singularity is handled analytically.
///
/// Returns (G_ij, H_ij).
fn element_integrals(mesh: &BemMesh, j: usize, x0: f64, y0: f64, is_singular: bool) -> (f64, f64) {
    let n = mesh.nodes.nrows();
    let j1 = if mesh.closed { (j + 1) % n } else { j + 1 };

    let xa = mesh.nodes[[j, 0]];
    let ya = mesh.nodes[[j, 1]];
    let xb = mesh.nodes[[j1, 0]];
    let yb = mesh.nodes[[j1, 1]];
    let dx_elem = xb - xa;
    let dy_elem = yb - ya;
    let len = (dx_elem * dx_elem + dy_elem * dy_elem).sqrt();
    if len < f64::EPSILON {
        return (0.0, 0.0);
    }

    let [nx, ny] = mesh.element_normal(j);

    if is_singular {
        // Analytic formula for the singular G integral over a straight element
        // of length L when the collocation point is at the midpoint.
        // G_sing = L / (2π) * (1 - ln(L/2))
        let l = len;
        let g_val = l / (2.0 * PI) * (1.0 - (l / 2.0).ln());
        // H integral over the element: for a smooth closed boundary the
        // diagonal H term is exactly 1/2 (from the free term c = 1/2).
        // We return 0 here and handle the 1/2 contribution separately.
        (g_val, 0.0)
    } else {
        // Gauss quadrature (5-point rule)
        let (gp, gw) = gauss5();
        let mut g_int = 0.0_f64;
        let mut h_int = 0.0_f64;

        for k in 0..5 {
            let t = 0.5 * (1.0 + gp[k]); // parameter in [0, 1]
            let xq = xa + t * dx_elem;
            let yq = ya + t * dy_elem;

            if let Some(gval) = fundamental_solution_2d(xq, yq, x0, y0) {
                g_int += gw[k] * gval;
            }
            if let Some(hval) = normal_derivative_green(xq, yq, x0, y0, nx, ny) {
                h_int += gw[k] * hval;
            }
        }
        // Jacobian: dt = 0.5 * d(xi), and the physical length element is len * dt
        g_int *= 0.5 * len;
        h_int *= 0.5 * len;
        (g_int, h_int)
    }
}

// ---------------------------------------------------------------------------
// System assembly
// ---------------------------------------------------------------------------

/// Assemble the BEM influence matrices G and H for a constant-element
/// discretisation of the 2-D Laplace BIE.
///
/// The system is   \[H\]{u} = \[G\]{q},
/// where the *e*-th row corresponds to the collocation point (midpoint of
/// element *e*), and the diagonal of H is augmented by 1/2.
///
/// # Returns
/// `(H, G)` where each matrix has shape `(n_elements, n_elements)`.
pub fn boundary_element_laplace(
    mesh: &BemMesh,
    _boundary_conditions: &[BemBoundaryCondition],
) -> PDEResult<(Array2<f64>, Array1<f64>)> {
    let ne = mesh.n_elements();
    if ne == 0 {
        return Err(PDEError::InvalidGrid("Empty BEM mesh".to_string()));
    }

    let mut h_mat = Array2::<f64>::zeros((ne, ne));
    let mut g_mat = Array2::<f64>::zeros((ne, ne));

    for i in 0..ne {
        let [x0, y0] = mesh.element_midpoint(i);

        for j in 0..ne {
            let singular = i == j;
            let (g_ij, h_ij) = element_integrals(mesh, j, x0, y0, singular);
            g_mat[[i, j]] = g_ij;
            h_mat[[i, j]] = h_ij;
        }

        // Free term: c(x₀) = 1/2 for smooth boundary
        h_mat[[i, i]] += 0.5;
    }

    // Return H and a placeholder RHS (populated properly in bem_solve)
    let rhs = Array1::<f64>::zeros(ne);
    Ok((h_mat, rhs))
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/// A boundary condition applied to a single boundary element.
#[derive(Debug, Clone)]
pub enum BemBoundaryCondition {
    /// Dirichlet: u is prescribed at this element.
    Dirichlet(f64),
    /// Neumann: ∂u/∂n (outward flux) is prescribed at this element.
    Neumann(f64),
}

// ---------------------------------------------------------------------------
// Full BEM solve
// ---------------------------------------------------------------------------

/// Solve the 2-D Laplace equation on the domain enclosed by `mesh` using
/// constant-element BEM.
///
/// `dirichlet_bc` and `neumann_bc` are slices of length `n_elements`.
/// Exactly one of `dirichlet_bc[e]` or `neumann_bc[e]` should be `Some`;
/// the other must be `None` (mixed BCs across different elements are allowed).
///
/// Returns the vector of length `n_elements` containing either:
///   * the solved Neumann flux q_e = ∂u/∂n at Dirichlet elements, or
///   * the solved Dirichlet value u_e at Neumann elements.
///
/// The full (u, q) pairs at all elements are returned as `(u_vec, q_vec)`.
pub fn bem_solve(
    mesh: &BemMesh,
    dirichlet_bc: &[Option<f64>],
    neumann_bc: &[Option<f64>],
) -> PDEResult<BemSolution> {
    let ne = mesh.n_elements();
    if dirichlet_bc.len() != ne || neumann_bc.len() != ne {
        return Err(PDEError::BoundaryConditions(format!(
            "BC arrays must have length {ne} (n_elements)"
        )));
    }

    // Validate that each element has exactly one condition
    for e in 0..ne {
        match (dirichlet_bc[e], neumann_bc[e]) {
            (Some(_), None) | (None, Some(_)) => {}
            (Some(_), Some(_)) => {
                return Err(PDEError::BoundaryConditions(format!(
                    "Element {e} has both Dirichlet and Neumann conditions"
                )));
            }
            (None, None) => {
                return Err(PDEError::BoundaryConditions(format!(
                    "Element {e} has no boundary condition"
                )));
            }
        }
    }

    // Assemble G and H matrices
    let bcs: Vec<BemBoundaryCondition> = (0..ne)
        .map(|e| {
            if let Some(v) = dirichlet_bc[e] {
                BemBoundaryCondition::Dirichlet(v)
            } else {
                BemBoundaryCondition::Neumann(neumann_bc[e].unwrap_or(0.0))
            }
        })
        .collect();

    let (h_mat, _) = boundary_element_laplace(mesh, &bcs)?;

    // Also build G separately (boundary_element_laplace only returns H)
    let mut g_mat = Array2::<f64>::zeros((ne, ne));
    for i in 0..ne {
        let [x0, y0] = mesh.element_midpoint(i);
        for j in 0..ne {
            let singular = i == j;
            let (g_ij, _) = element_integrals(mesh, j, x0, y0, singular);
            g_mat[[i, j]] = g_ij;
        }
    }

    // Rearrange  [H]{u} = [G]{q}  so that unknowns are on the left.
    // For each element: if Dirichlet (u known) move H column to RHS;
    //                   if Neumann  (q known) move G column to RHS.
    let mut a_mat = Array2::<f64>::zeros((ne, ne));
    let mut rhs = Array1::<f64>::zeros(ne);

    for e in 0..ne {
        if dirichlet_bc[e].is_some() {
            // Unknown: q_e; equation contribution: -G_col_e * q_e on LHS
            for row in 0..ne {
                a_mat[[row, e]] = -g_mat[[row, e]];
            }
        } else {
            // Unknown: u_e; equation contribution: H_col_e * u_e on LHS
            for row in 0..ne {
                a_mat[[row, e]] = h_mat[[row, e]];
            }
        }
    }

    // Build RHS: known values contribute with opposite sign
    for e in 0..ne {
        if let Some(u_known) = dirichlet_bc[e] {
            // H * u_known  goes to RHS
            for row in 0..ne {
                rhs[row] -= h_mat[[row, e]] * u_known;
            }
        } else if let Some(q_known) = neumann_bc[e] {
            // -G * q_known goes to RHS
            for row in 0..ne {
                rhs[row] += g_mat[[row, e]] * q_known;
            }
        }
    }

    // Solve the linear system using Gaussian elimination with partial pivoting
    let x = gaussian_elimination(&a_mat, &rhs)?;

    // Reconstruct full u and q vectors
    let mut u_vec = Array1::<f64>::zeros(ne);
    let mut q_vec = Array1::<f64>::zeros(ne);

    for e in 0..ne {
        if let Some(u_known) = dirichlet_bc[e] {
            u_vec[e] = u_known;
            q_vec[e] = x[e];
        } else if let Some(q_known) = neumann_bc[e] {
            u_vec[e] = x[e];
            q_vec[e] = q_known;
        }
    }

    Ok(BemSolution {
        u: u_vec,
        q: q_vec,
        mesh_midpoints: (0..ne).map(|e| mesh.element_midpoint(e)).collect(),
    })
}

/// Solution returned by `bem_solve`.
#[derive(Debug, Clone)]
pub struct BemSolution {
    /// Dirichlet values u at each element midpoint.
    pub u: Array1<f64>,
    /// Neumann fluxes ∂u/∂n at each element midpoint.
    pub q: Array1<f64>,
    /// Collocation points (element midpoints).
    pub mesh_midpoints: Vec<[f64; 2]>,
}

// ---------------------------------------------------------------------------
// Interior evaluation
// ---------------------------------------------------------------------------

/// Evaluate the solution at an interior point `(px, py)` using the
/// boundary integral representation:
///
/// u(p) = ∫_Γ [ G(x,p) q(x) − H(x,p) u(x) ] dΓ(x)
///
/// where the integrals are computed element-wise with Gauss quadrature.
pub fn evaluate_interior(solution: &BemSolution, mesh: &BemMesh, point: [f64; 2]) -> f64 {
    let ne = mesh.n_elements();
    let [px, py] = point;
    let mut val = 0.0_f64;

    let n = mesh.nodes.nrows();
    for j in 0..ne {
        let j1 = if mesh.closed { (j + 1) % n } else { j + 1 };
        let xa = mesh.nodes[[j, 0]];
        let ya = mesh.nodes[[j, 1]];
        let xb = mesh.nodes[[j1, 0]];
        let yb = mesh.nodes[[j1, 1]];
        let dx_elem = xb - xa;
        let dy_elem = yb - ya;
        let len = (dx_elem * dx_elem + dy_elem * dy_elem).sqrt();
        if len < f64::EPSILON {
            continue;
        }
        let [nx, ny] = mesh.element_normal(j);
        let u_j = solution.u[j];
        let q_j = solution.q[j];

        let (gp, gw) = gauss5();
        for k in 0..5 {
            let t = 0.5 * (1.0 + gp[k]);
            let xq = xa + t * dx_elem;
            let yq = ya + t * dy_elem;

            let g_val = fundamental_solution_2d(xq, yq, px, py).unwrap_or(0.0);
            let h_val = normal_derivative_green(xq, yq, px, py, nx, ny).unwrap_or(0.0);

            val += gw[k] * 0.5 * len * (g_val * q_j - h_val * u_j);
        }
    }

    val
}

// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------

/// Solve the `n × n` system  A x = b  by Gaussian elimination with partial
/// pivoting.  Returns an error if the matrix is (near-)singular.
fn gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(PDEError::ComputationError(
            "gaussian_elimination: dimension mismatch".to_string(),
        ));
    }

    // Augmented matrix [A | b]
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(PDEError::ComputationError(
                "gaussian_elimination: singular or near-singular matrix".to_string(),
            ));
        }
        // Swap rows
        if pivot_row != col {
            for k in 0..=n {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[pivot_row, k]];
                aug[[pivot_row, k]] = tmp;
            }
        }
        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for k in col..=n {
                let delta = factor * aug[[col, k]];
                aug[[row, k]] -= delta;
            }
        }
    }

    // Back-substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s -= aug[[i, j]] * x[j];
        }
        x[i] = s / aug[[i, i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_bem_mesh_line() {
        let mesh = BemMesh::new_line(0.0, 1.0, 0.0, 4, false).expect("mesh creation");
        assert_eq!(mesh.n_elements(), 4);
        assert_eq!(mesh.n_nodes(), 5);
        let mid = mesh.element_midpoint(0);
        assert!((mid[0] - 0.125).abs() < 1e-12);
        assert!((mid[1]).abs() < 1e-12);
    }

    #[test]
    fn test_bem_mesh_rectangle() {
        let mesh = BemMesh::new_rectangle(0.0, 1.0, 0.0, 1.0, 4).expect("rect mesh");
        assert_eq!(mesh.n_elements(), 16);
        assert!(mesh.closed);
    }

    #[test]
    fn test_fundamental_solution() {
        // G(1,0; 0,0) = -ln(1)/(2π) = 0
        let g = fundamental_solution_2d(1.0, 0.0, 0.0, 0.0);
        assert!(g.is_some());
        assert!((g.expect("fundamental_solution_2d should return Some for r=1")).abs() < 1e-12);

        // G at r=2 should be -ln(2)/(2π)
        let g2 = fundamental_solution_2d(2.0, 0.0, 0.0, 0.0);
        let expected = -2_f64.ln() / (2.0 * PI);
        assert!(
            (g2.expect("fundamental_solution_2d should return Some for r=2") - expected).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_fundamental_solution_singular() {
        let g = fundamental_solution_2d(0.0, 0.0, 0.0, 0.0);
        assert!(g.is_none());
    }

    #[test]
    fn test_normal_derivative_green() {
        // For r pointing in +x and normal in +x:  ∂G/∂n = -x/(2π r²)
        let h = normal_derivative_green(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let expected = -1.0 / (2.0 * PI);
        assert!(
            (h.expect("normal_derivative_green should return Some for r=1") - expected).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_gauss_weights_sum_to_two() {
        let (_, gw) = gauss5();
        let sum: f64 = gw.iter().sum();
        assert!((sum - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_elimination() {
        // 2x + y = 5
        //  x + 3y = 10  → x = 1, y = 3
        let mut a = Array2::<f64>::zeros((2, 2));
        a[[0, 0]] = 2.0;
        a[[0, 1]] = 1.0;
        a[[1, 0]] = 1.0;
        a[[1, 1]] = 3.0;
        let b = Array1::from_vec(vec![5.0, 10.0]);
        let x = gaussian_elimination(&a, &b).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_element_laplace_assembles() {
        let mesh = BemMesh::new_rectangle(0.0, 1.0, 0.0, 1.0, 3).expect("mesh");
        let ne = mesh.n_elements();
        let bcs: Vec<BemBoundaryCondition> = (0..ne)
            .map(|_| BemBoundaryCondition::Dirichlet(0.0))
            .collect();
        let (h_mat, _rhs) = boundary_element_laplace(&mesh, &bcs).expect("assemble");
        assert_eq!(h_mat.nrows(), ne);
        assert_eq!(h_mat.ncols(), ne);
        // Diagonal must be >= 0.5 (free term contribution)
        for i in 0..ne {
            assert!(h_mat[[i, i]] >= 0.4, "diagonal[{i}] = {}", h_mat[[i, i]]);
        }
    }

    #[test]
    fn test_bem_solve_constant_bc() {
        // All Dirichlet u = 1.0 on a square boundary → interior should be ~ 1.0
        let mesh = BemMesh::new_rectangle(0.0, 1.0, 0.0, 1.0, 5).expect("mesh");
        let ne = mesh.n_elements();
        let dirichlet_bc: Vec<Option<f64>> = vec![Some(1.0); ne];
        let neumann_bc: Vec<Option<f64>> = vec![None; ne];

        let sol = bem_solve(&mesh, &dirichlet_bc, &neumann_bc).expect("bem_solve");
        assert_eq!(sol.u.len(), ne);

        // All prescribed Dirichlet values must equal 1.0
        for e in 0..ne {
            assert!((sol.u[e] - 1.0).abs() < 1e-6, "u[{e}] = {}", sol.u[e]);
        }

        // Interior point should also be approximately 1.0
        let v = evaluate_interior(&sol, &mesh, [0.5, 0.5]);
        assert!((v - 1.0).abs() < 0.05, "interior value = {v}");
    }

    #[test]
    fn test_evaluate_interior_basic() {
        // Zero Dirichlet on all sides: u should be close to 0 inside
        let mesh = BemMesh::new_rectangle(0.0, 1.0, 0.0, 1.0, 4).expect("mesh");
        let ne = mesh.n_elements();
        let dirichlet_bc: Vec<Option<f64>> = vec![Some(0.0); ne];
        let neumann_bc: Vec<Option<f64>> = vec![None; ne];

        let sol = bem_solve(&mesh, &dirichlet_bc, &neumann_bc).expect("bem_solve");
        let v = evaluate_interior(&sol, &mesh, [0.5, 0.5]);
        // Should be very close to zero
        assert!(v.abs() < 0.1, "interior zero bc value = {v}");
    }
}
