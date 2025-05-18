//! Spectral Element Methods (SEM) for solving PDEs
//!
//! This module provides implementations of spectral element methods,
//! which combine the accuracy of spectral methods with the geometric
//! flexibility of finite element methods. This approach allows for
//! high-order polynomial approximations on complex geometries.
//!
//! Key features:
//! - High-order polynomial basis functions (using nodal Lagrange polynomials)
//! - Domain decomposition into elements (quadrilaterals in 2D)
//! - Gauss-Lobatto-Legendre quadrature for integration
//! - Isoparametric mapping for curved elements
//! - Exponential convergence for smooth solutions

use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use std::time::Instant;

use crate::pde::spectral::{legendre_diff_matrix, legendre_points};
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Quadrilateral element for 2D spectral element methods
#[derive(Debug, Clone)]
pub struct QuadElement {
    /// Element ID
    pub id: usize,

    /// Global coordinates of element vertices (4 corners, counterclockwise ordering)
    pub vertices: [(f64, f64); 4],

    /// Global indices of nodes in this element
    pub node_indices: Vec<usize>,

    /// Element boundary conditions (if any)
    pub boundary_edges: Vec<(usize, usize, Option<BoundaryConditionType>)>,
}

/// Spectral element mesh for 2D problems
#[derive(Debug, Clone)]
pub struct SpectralElementMesh2D {
    /// Elements in the mesh
    pub elements: Vec<QuadElement>,

    /// Global node coordinates (x, y)
    pub nodes: Vec<(f64, f64)>,

    /// Global-to-local mapping for nodes
    pub global_to_local: Vec<Vec<(usize, usize, usize)>>, // (element_id, i, j)

    /// Boundary nodes with condition info
    pub boundary_nodes: Vec<(usize, BoundaryConditionType)>,

    /// Polynomial order in each direction
    pub order: (usize, usize),

    /// Total number of nodes in the mesh
    pub num_nodes: usize,
}

impl SpectralElementMesh2D {
    /// Create a new rectangular spectral element mesh
    ///
    /// # Arguments
    ///
    /// * `x_range` - Range for the x-coordinate domain [x_min, x_max]
    /// * `y_range` - Range for the y-coordinate domain [y_min, y_max]
    /// * `nx` - Number of elements in the x direction
    /// * `ny` - Number of elements in the y direction
    /// * `order` - Polynomial order in each element (p, p)
    ///
    /// # Returns
    ///
    /// * A structured rectangular mesh of quadrilateral elements
    pub fn rectangular(
        x_range: [f64; 2],
        y_range: [f64; 2],
        nx: usize,
        ny: usize,
        order: usize,
    ) -> PDEResult<Self> {
        if nx == 0 || ny == 0 {
            return Err(PDEError::DomainError(
                "Number of elements must be at least 1 in each direction".to_string(),
            ));
        }

        if order < 1 {
            return Err(PDEError::DomainError(
                "Polynomial order must be at least 1".to_string(),
            ));
        }

        let [x_min, x_max] = x_range;
        let [y_min, y_max] = y_range;

        let dx = (x_max - x_min) / nx as f64;
        let dy = (y_max - y_min) / ny as f64;

        // Number of nodes in each direction per element (order + 1)
        let n = order + 1;

        // Generate Gauss-Lobatto-Legendre points in 1D (scaled to [0, 1])
        let (xi_pts, _) = legendre_points(n);
        let xi = (xi_pts + 1.0) * 0.5;

        // Create elements and nodes
        let mut elements = Vec::with_capacity(nx * ny);
        let mut nodes = Vec::new();
        let mut boundary_nodes = Vec::new();

        // Global node counter
        let mut node_count = 0;

        // Temporary global node indices for each element
        let mut global_indices = Array2::<usize>::zeros((nx * n - (nx - 1), ny * n - (ny - 1)));

        // First, create all nodes and assign global indices
        for j in 0..(ny * n - (ny - 1)) {
            for i in 0..(nx * n - (nx - 1)) {
                // Determine if this is an element edge or interior node
                let _is_edge = i % n == 0
                    || i == nx * n - (nx - 1) - 1
                    || j % n == 0
                    || j == ny * n - (ny - 1) - 1;

                // Compute physical coordinates
                let x_idx = i / n; // Element index in x direction
                let y_idx = j / n; // Element index in y direction

                let local_i = i % n; // Local index within element
                let local_j = j % n;

                // Handle shared nodes between elements
                if (local_i == 0 && x_idx > 0) || (local_j == 0 && y_idx > 0) {
                    // This node is shared with a previous element
                    // Skip node creation but still assign the global index
                    if local_i == 0 && local_j == 0 && x_idx > 0 && y_idx > 0 {
                        // Corner node shared by 4 elements
                        global_indices[[i, j]] = global_indices[[i - n + 1, j - n + 1]];
                    } else if local_i == 0 && x_idx > 0 {
                        // Edge node shared horizontally
                        global_indices[[i, j]] = global_indices[[i - n + 1, j]];
                    } else if local_j == 0 && y_idx > 0 {
                        // Edge node shared vertically
                        global_indices[[i, j]] = global_indices[[i, j - n + 1]];
                    }
                    continue;
                }

                // Map to physical coordinates using element index and local coordinates
                let x = x_min + (x_idx as f64 + xi[local_i]) * dx;
                let y = y_min + (y_idx as f64 + xi[local_j]) * dy;

                // Create node and store index
                nodes.push((x, y));
                global_indices[[i, j]] = node_count;

                // Add to boundary nodes if on the domain boundary
                if i == 0 || i == nx * n - (nx - 1) - 1 || j == 0 || j == ny * n - (ny - 1) - 1 {
                    let bc_type = BoundaryConditionType::Dirichlet; // Default, will be updated later
                    boundary_nodes.push((node_count, bc_type));
                }

                node_count += 1;
            }
        }

        // Initialize global to local mapping
        let mut global_to_local = vec![Vec::new(); node_count];

        // Now create elements using the global node indices
        for ey in 0..ny {
            for ex in 0..nx {
                let element_id = ey * nx + ex;

                // Element vertices (corners)
                let vertices = [
                    (x_min + ex as f64 * dx, y_min + ey as f64 * dy),
                    (x_min + (ex + 1) as f64 * dx, y_min + ey as f64 * dy),
                    (x_min + (ex + 1) as f64 * dx, y_min + (ey + 1) as f64 * dy),
                    (x_min + ex as f64 * dx, y_min + (ey + 1) as f64 * dy),
                ];

                // Collect all node indices for this element
                let mut node_indices = Vec::with_capacity(n * n);

                for j in 0..n {
                    for i in 0..n {
                        let global_i = ex * (n - 1) + i;
                        let global_j = ey * (n - 1) + j;

                        let idx = global_indices[[global_i, global_j]];
                        node_indices.push(idx);

                        // Update global to local mapping
                        global_to_local[idx].push((element_id, i, j));
                    }
                }

                // Determine element boundary edges
                let mut boundary_edges = Vec::new();

                // Check if element is on domain boundary
                if ex == 0 {
                    // Left boundary
                    boundary_edges.push((0, 3, Some(BoundaryConditionType::Dirichlet)));
                }
                if ex == nx - 1 {
                    // Right boundary
                    boundary_edges.push((1, 2, Some(BoundaryConditionType::Dirichlet)));
                }
                if ey == 0 {
                    // Bottom boundary
                    boundary_edges.push((0, 1, Some(BoundaryConditionType::Dirichlet)));
                }
                if ey == ny - 1 {
                    // Top boundary
                    boundary_edges.push((2, 3, Some(BoundaryConditionType::Dirichlet)));
                }

                // Create the element
                let element = QuadElement {
                    id: element_id,
                    vertices,
                    node_indices,
                    boundary_edges,
                };

                elements.push(element);
            }
        }

        Ok(SpectralElementMesh2D {
            elements,
            nodes,
            global_to_local,
            boundary_nodes,
            order: (order, order),
            num_nodes: node_count,
        })
    }

    /// Update the boundary conditions on the mesh
    ///
    /// # Arguments
    ///
    /// * `boundary_conditions` - Vector of boundary conditions to apply
    ///
    /// # Returns
    ///
    /// * `PDEResult<()>` - Result indicating success or error
    pub fn set_boundary_conditions(
        &mut self,
        boundary_conditions: &[BoundaryCondition<f64>],
    ) -> PDEResult<()> {
        // Map boundary conditions to mesh boundary nodes
        for bc in boundary_conditions {
            let nodes_to_update = match bc.location {
                BoundaryLocation::Lower => match bc.dimension {
                    0 => self
                        .boundary_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, (idx, _))| self.nodes[*idx].0 < 1e-10)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>(),
                    1 => self
                        .boundary_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, (idx, _))| self.nodes[*idx].1 < 1e-10)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>(),
                    _ => {
                        return Err(PDEError::DomainError(format!(
                            "Invalid dimension: {}",
                            bc.dimension
                        )))
                    }
                },
                BoundaryLocation::Upper => match bc.dimension {
                    0 => self
                        .boundary_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, (idx, _))| (self.nodes[*idx].0 - 1.0).abs() < 1e-10)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>(),
                    1 => self
                        .boundary_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, (idx, _))| (self.nodes[*idx].1 - 1.0).abs() < 1e-10)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>(),
                    _ => {
                        return Err(PDEError::DomainError(format!(
                            "Invalid dimension: {}",
                            bc.dimension
                        )))
                    }
                },
            };

            // Update boundary node types
            for node_idx in nodes_to_update {
                self.boundary_nodes[node_idx].1 = bc.bc_type;
            }

            // Update element boundary edges
            for element in &mut self.elements {
                for (_, _, bc_type) in &mut element.boundary_edges {
                    if let Some(ref mut bc_type_val) = bc_type {
                        // Check if this boundary edge is on the specified boundary
                        let edge_matches = match bc.location {
                            BoundaryLocation::Lower => match bc.dimension {
                                0 => element.vertices[0].0 < 1e-10 && element.vertices[3].0 < 1e-10,
                                1 => element.vertices[0].1 < 1e-10 && element.vertices[1].1 < 1e-10,
                                _ => false,
                            },
                            BoundaryLocation::Upper => match bc.dimension {
                                0 => {
                                    (element.vertices[1].0 - 1.0).abs() < 1e-10
                                        && (element.vertices[2].0 - 1.0).abs() < 1e-10
                                }
                                1 => {
                                    (element.vertices[2].1 - 1.0).abs() < 1e-10
                                        && (element.vertices[3].1 - 1.0).abs() < 1e-10
                                }
                                _ => false,
                            },
                        };

                        if edge_matches {
                            *bc_type_val = bc.bc_type;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Options for spectral element methods
#[derive(Debug, Clone)]
pub struct SpectralElementOptions {
    /// Polynomial order in each direction
    pub order: usize,

    /// Number of elements in x direction
    pub nx: usize,

    /// Number of elements in y direction
    pub ny: usize,

    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,

    /// Tolerance for convergence
    pub tolerance: f64,

    /// Whether to save convergence history
    pub save_convergence_history: bool,

    /// Print detailed progress information
    pub verbose: bool,
}

impl Default for SpectralElementOptions {
    fn default() -> Self {
        SpectralElementOptions {
            order: 4,
            nx: 4,
            ny: 4,
            max_iterations: 1000,
            tolerance: 1e-10,
            save_convergence_history: false,
            verbose: false,
        }
    }
}

/// Result from spectral element method solution
#[derive(Debug, Clone)]
pub struct SpectralElementResult {
    /// Solution values at each node
    pub u: Array1<f64>,

    /// Node coordinates (x, y)
    pub nodes: Vec<(f64, f64)>,

    /// Element connectivity
    pub elements: Vec<QuadElement>,

    /// Residual norm
    pub residual_norm: f64,

    /// Number of iterations performed
    pub num_iterations: usize,

    /// Computation time
    pub computation_time: f64,

    /// Convergence history
    pub convergence_history: Option<Vec<f64>>,
}

/// 2D Poisson solver using spectral element method
///
/// Solves: ∇²u = f(x,y) with appropriate boundary conditions
pub struct SpectralElementPoisson2D {
    /// Computational domain
    domain: Domain,

    /// Source term function f(x, y)
    source_term: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: SpectralElementOptions,
}

impl SpectralElementPoisson2D {
    /// Create a new spectral element Poisson solver
    ///
    /// # Arguments
    ///
    /// * `domain` - Computational domain
    /// * `source_term` - Function f(x,y) for the Poisson equation ∇²u = f(x,y)
    /// * `boundary_conditions` - Boundary conditions for the domain
    /// * `options` - Solver options (or None for defaults)
    ///
    /// # Returns
    ///
    /// * `PDEResult<Self>` - New solver instance
    pub fn new(
        domain: Domain,
        source_term: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<SpectralElementOptions>,
    ) -> PDEResult<Self> {
        // Validate domain
        if domain.dimensions() != 2 {
            return Err(PDEError::DomainError(
                "Domain must be 2-dimensional for 2D spectral element solver".to_string(),
            ));
        }

        // Validate boundary conditions
        for bc in &boundary_conditions {
            if bc.dimension >= 2 {
                return Err(PDEError::BoundaryConditions(format!(
                    "Invalid dimension {} in boundary condition",
                    bc.dimension
                )));
            }

            match bc.bc_type {
                BoundaryConditionType::Dirichlet => {}
                BoundaryConditionType::Neumann => {}
                BoundaryConditionType::Robin => {
                    if bc.coefficients.is_none() {
                        return Err(PDEError::BoundaryConditions(
                            "Robin boundary conditions require coefficients".to_string(),
                        ));
                    }
                }
                BoundaryConditionType::Periodic => {
                    return Err(PDEError::BoundaryConditions(
                        "Periodic boundary conditions not implemented for spectral element method"
                            .to_string(),
                    ));
                }
            }
        }

        let options = options.unwrap_or_default();

        Ok(SpectralElementPoisson2D {
            domain,
            source_term: Box::new(source_term),
            boundary_conditions,
            options,
        })
    }

    /// Solve the Poisson equation using spectral element method
    ///
    /// # Returns
    ///
    /// * `PDEResult<SpectralElementResult>` - Solution result
    pub fn solve(&self) -> PDEResult<SpectralElementResult> {
        let start_time = Instant::now();

        // Extract domain information
        let x_range = &self.domain.ranges[0];
        let y_range = &self.domain.ranges[1];

        // Create spectral element mesh
        let mut mesh = SpectralElementMesh2D::rectangular(
            [x_range.start, x_range.end],
            [y_range.start, y_range.end],
            self.options.nx,
            self.options.ny,
            self.options.order,
        )?;

        // Set boundary conditions
        mesh.set_boundary_conditions(&self.boundary_conditions)?;

        // --- Local operations within each element ---

        // Create differentiation matrices for reference element [-1, 1]²
        let n = self.options.order + 1;
        let d1_ref = legendre_diff_matrix(n);

        // Reference Gauss-Lobatto-Legendre points and weights
        let (xi, w) = legendre_points(n);

        // Create stiffness and mass matrix for each element
        let mut element_stiffness = vec![Array2::<f64>::zeros((n * n, n * n)); mesh.elements.len()];
        let mut element_mass = vec![Array2::<f64>::zeros((n * n, n * n)); mesh.elements.len()];
        let mut element_load = vec![Array1::<f64>::zeros(n * n); mesh.elements.len()];

        for (e_idx, element) in mesh.elements.iter().enumerate() {
            // Element vertices
            let vertices = element.vertices;

            // Element size
            let dx = vertices[1].0 - vertices[0].0;
            let dy = vertices[3].1 - vertices[0].1;

            // Jacobian determinant (assuming rectangular elements)
            let j_det = (dx * dy) / 4.0;

            // Scaled differentiation matrices for physical element
            let d1_x = d1_ref.mapv(|val| val * 2.0 / dx);
            let d1_y = d1_ref.mapv(|val| val * 2.0 / dy);

            // Precompute tensor products for efficiency
            let mut dx_tensor = Array3::<f64>::zeros((n, n, n * n));
            let mut dy_tensor = Array3::<f64>::zeros((n, n, n * n));

            for j in 0..n {
                for i in 0..n {
                    let _node = j * n + i;

                    // Fill tensor products
                    for k in 0..n {
                        // First pattern: dx_tensor[[i, j, k * n..(k + 1) * n]]
                        dx_tensor
                            .slice_mut(s![i, j, k * n..(k + 1) * n])
                            .assign(&d1_x.slice(s![k, ..]));
                        dy_tensor.slice_mut(s![i, j, k * n..(k + 1) * n]).fill(0.0);
                    }

                    // Second pattern: tensors at indices from k to n*n
                    for k in 0..n {
                        for idx in k..(n * n) {
                            if idx % n == k {
                                dx_tensor[[i, j, idx]] = 0.0;
                                dy_tensor[[i, j, idx]] = d1_y[[k, idx / n]];
                            }
                        }
                    }
                }
            }

            // Compute element stiffness matrix: K_{ij} = ∫∫ (∇φ_i ⋅ ∇φ_j) dxdy
            for i in 0..n * n {
                for j in 0..n * n {
                    // For Poisson equation: stiffness = integral of gradient dot product
                    let mut stiffness_val = 0.0;

                    for ni in 0..n {
                        for nj in 0..n {
                            // Compute ∇φ_i ⋅ ∇φ_j at quadrature point (ξ_ni, ξ_nj)
                            let dx_i = dx_tensor[[ni, nj, i]];
                            let dy_i = dy_tensor[[ni, nj, i]];
                            let dx_j = dx_tensor[[ni, nj, j]];
                            let dy_j = dy_tensor[[ni, nj, j]];

                            // Gradient dot product: ∇φ_i ⋅ ∇φ_j
                            let grad_dot = dx_i * dx_j + dy_i * dy_j;

                            // Integrate with quadrature weights
                            stiffness_val += grad_dot * w[ni] * w[nj] * j_det;
                        }
                    }

                    element_stiffness[e_idx][[i, j]] = stiffness_val;
                }
            }

            // Compute element mass matrix: M_{ij} = ∫∫ φ_i φ_j dxdy
            for i in 0..n * n {
                for j in 0..n * n {
                    // For mass matrix: just the integral of basis function products
                    let mut mass_val = 0.0;

                    // Which local nodes do i and j correspond to?
                    let i_local = (i % n, i / n);
                    let _j_local = (j % n, j / n);

                    // Mass matrix has nice tensor product structure for Lagrange polynomials
                    // If nodes are far apart, the integral is zero (orthogonality)
                    if i == j {
                        // Diagonal element - we can use the quadrature weights directly
                        mass_val = w[i_local.0] * w[i_local.1] * j_det;
                    }

                    element_mass[e_idx][[i, j]] = mass_val;
                }
            }

            // Compute element load vector: f_i = ∫∫ f(x,y) φ_i dxdy
            for i in 0..n * n {
                let mut load_val = 0.0;

                for ni in 0..n {
                    for nj in 0..n {
                        // Map reference coordinates to physical coordinates
                        let x = vertices[0].0 + (xi[ni] + 1.0) * dx / 2.0;
                        let y = vertices[0].1 + (xi[nj] + 1.0) * dy / 2.0;

                        // Evaluate source term at quadrature point
                        let source = (self.source_term)(x, y);

                        // Local basis function value at quadrature point
                        let i_local = (i % n, i / n);
                        let basis_val = if i_local.0 == ni && i_local.1 == nj {
                            1.0
                        } else {
                            0.0
                        };

                        // Integrate with quadrature weights
                        load_val += source * basis_val * w[ni] * w[nj] * j_det;
                    }
                }

                element_load[e_idx][i] = load_val;
            }
        }

        // --- Global assembly ---

        // Create global system: Au = b
        let n_dof = mesh.num_nodes;
        let mut global_matrix = Array2::<f64>::zeros((n_dof, n_dof));
        let mut global_load = Array1::<f64>::zeros(n_dof);

        // Assemble global matrix and load vector from element contributions
        for (e_idx, element) in mesh.elements.iter().enumerate() {
            for (i, &i_global) in element.node_indices.iter().enumerate() {
                // Add element load to global load
                global_load[i_global] += element_load[e_idx][i];

                for (j, &j_global) in element.node_indices.iter().enumerate() {
                    // Add element stiffness to global matrix
                    global_matrix[[i_global, j_global]] += element_stiffness[e_idx][[i, j]];
                }
            }
        }

        // Apply boundary conditions
        for &(node_idx, bc_type) in &mesh.boundary_nodes {
            match bc_type {
                BoundaryConditionType::Dirichlet => {
                    // For Dirichlet, set matrix row to identity and load to value
                    let (x, y) = mesh.nodes[node_idx];

                    // Find the appropriate boundary condition value
                    let mut bc_value = 0.0; // Default value

                    for bc in &self.boundary_conditions {
                        if bc.bc_type != BoundaryConditionType::Dirichlet {
                            continue;
                        }

                        let is_on_boundary = match (bc.dimension, bc.location) {
                            (0, BoundaryLocation::Lower) => x < 1e-10,
                            (0, BoundaryLocation::Upper) => {
                                (x - (x_range.end - x_range.start)).abs() < 1e-10
                            }
                            (1, BoundaryLocation::Lower) => y < 1e-10,
                            (1, BoundaryLocation::Upper) => {
                                (y - (y_range.end - y_range.start)).abs() < 1e-10
                            }
                            _ => false,
                        };

                        if is_on_boundary {
                            bc_value = bc.value;
                            break;
                        }
                    }

                    // Clear row and set diagonal to 1
                    for j in 0..n_dof {
                        global_matrix[[node_idx, j]] = 0.0;
                    }
                    global_matrix[[node_idx, node_idx]] = 1.0;

                    // Set load vector value
                    global_load[node_idx] = bc_value;
                }
                BoundaryConditionType::Neumann => {
                    // Neumann boundary conditions are natural in the weak form
                    // They're already accounted for in the global assembly
                    // unless there's a non-zero flux, in which case we need to
                    // add boundary integrals

                    // Find the appropriate boundary condition value
                    let (x, y) = mesh.nodes[node_idx];

                    for bc in &self.boundary_conditions {
                        if bc.bc_type != BoundaryConditionType::Neumann {
                            continue;
                        }

                        let is_on_boundary = match (bc.dimension, bc.location) {
                            (0, BoundaryLocation::Lower) => x < 1e-10,
                            (0, BoundaryLocation::Upper) => {
                                (x - (x_range.end - x_range.start)).abs() < 1e-10
                            }
                            (1, BoundaryLocation::Lower) => y < 1e-10,
                            (1, BoundaryLocation::Upper) => {
                                (y - (y_range.end - y_range.start)).abs() < 1e-10
                            }
                            _ => false,
                        };

                        if is_on_boundary {
                            let bc_value = bc.value;

                            // For non-zero Neumann BC, we need to add the boundary integral
                            if bc_value != 0.0 {
                                // This implementation is simplified - in a complete solver
                                // we would compute boundary integrals properly
                                global_load[node_idx] += bc_value;
                            }

                            break;
                        }
                    }
                }
                BoundaryConditionType::Robin => {
                    // Robin boundary conditions combine Dirichlet and Neumann
                    // For simplicity, we approximate them here by considering
                    // only the constant term of the form a*u + b*du/dn = c

                    // Find the appropriate boundary condition coefficients
                    let (x, y) = mesh.nodes[node_idx];

                    for bc in &self.boundary_conditions {
                        if bc.bc_type != BoundaryConditionType::Robin {
                            continue;
                        }

                        let is_on_boundary = match (bc.dimension, bc.location) {
                            (0, BoundaryLocation::Lower) => x < 1e-10,
                            (0, BoundaryLocation::Upper) => {
                                (x - (x_range.end - x_range.start)).abs() < 1e-10
                            }
                            (1, BoundaryLocation::Lower) => y < 1e-10,
                            (1, BoundaryLocation::Upper) => {
                                (y - (y_range.end - y_range.start)).abs() < 1e-10
                            }
                            _ => false,
                        };

                        if is_on_boundary {
                            if let Some([a, _b, c]) = bc.coefficients {
                                // For Robin BCs: a*u + b*du/dn = c
                                // We need to modify the matrix and load vector
                                global_matrix[[node_idx, node_idx]] += a;
                                global_load[node_idx] += c;
                            }

                            break;
                        }
                    }
                }
                _ => {
                    return Err(PDEError::BoundaryConditions(
                        "Unsupported boundary condition type".to_string(),
                    ));
                }
            }
        }

        // Solve the linear system
        let solution = self.solve_linear_system(&global_matrix, &global_load.view())?;

        // Compute residual
        let global_residual = {
            let mut residual = global_matrix.dot(&solution) - &global_load;

            // Exclude boundary points from residual calculation
            for &(node_idx, bc_type) in &mesh.boundary_nodes {
                if bc_type == BoundaryConditionType::Dirichlet {
                    residual[node_idx] = 0.0;
                }
            }

            residual
        };

        let residual_norm = (global_residual.iter().map(|&r| r * r).sum::<f64>()
            / (n_dof - mesh.boundary_nodes.len()) as f64)
            .sqrt();

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(SpectralElementResult {
            u: solution,
            nodes: mesh.nodes.clone(),
            elements: mesh.elements.clone(),
            residual_norm,
            num_iterations: 1, // Direct solve
            computation_time,
            convergence_history: None,
        })
    }

    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Simple Gaussian elimination with partial pivoting
        // For a real implementation, use a specialized linear algebra library

        // Create copies of A and b
        let mut a_copy = a.clone();
        let mut b_copy = b.to_owned();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_val = a_copy[[i, i]].abs();
            let mut max_row = i;

            for k in i + 1..n {
                if a_copy[[k, i]].abs() > max_val {
                    max_val = a_copy[[k, i]].abs();
                    max_row = k;
                }
            }

            // Check if matrix is singular
            if max_val < 1e-10 {
                return Err(PDEError::Other(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Swap rows if necessary
            if max_row != i {
                for j in i..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }

                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }

            // Eliminate below
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];

                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }

                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += a_copy[[i, j]] * x[j];
            }

            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }
}

impl From<SpectralElementResult> for PDESolution<f64> {
    fn from(result: SpectralElementResult) -> Self {
        // Extract node coordinates for grids
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();

        for &(x, y) in &result.nodes {
            x_coords.push(x);
            y_coords.push(y);
        }

        // Create unique sorted x and y coordinates for grid
        x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());

        x_coords.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        y_coords.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        let grids = vec![Array1::from_vec(x_coords), Array1::from_vec(y_coords)];

        // Create solution values as a 2D array for each grid point
        let mut values = Vec::new();
        let n_points = result.u.len();
        let u_reshaped = result.u.into_shape_with_order((n_points, 1)).unwrap();
        values.push(u_reshaped);

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_iterations,
            computation_time: result.computation_time,
            residual_norm: Some(result.residual_norm),
            convergence_history: result.convergence_history,
            method: "Spectral Element Method".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
