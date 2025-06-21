//! Petrov-Galerkin finite element formulations
//!
//! This module implements Petrov-Galerkin finite element methods where the
//! test functions and trial functions are chosen from different spaces.
//! This is particularly useful for:
//! - Convection-dominated problems (SUPG method)
//! - Mixed formulations (pressure-velocity in fluid flow)
//! - Stability enhancement for problematic PDEs
//!
//! # Petrov-Galerkin Method
//!
//! Unlike standard Galerkin methods where test and trial functions are the same,
//! Petrov-Galerkin methods use:
//! - Trial functions: φᵢ (solution approximation space)
//! - Test functions: ψⱼ (different space for weighting residuals)
//!
//! This flexibility allows for better stability and accuracy properties.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::pde::{BoundaryCondition, PDEResult, PDESolution, PDESolverInfo};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Petrov-Galerkin formulation types
#[derive(Debug, Clone)]
pub enum PetrovGalerkinType<F: IntegrateFloat> {
    /// Streamline Upwind Petrov-Galerkin (SUPG) for convection-diffusion
    SUPG {
        /// Convection coefficients (bₓ, bᵧ)
        convection: (F, F),
        /// Diffusion coefficient
        diffusion: F,
        /// SUPG stabilization parameter
        tau: Option<F>,
    },
    /// Galerkin Least Squares (GLS) for general stability
    GLS {
        /// Stabilization parameter
        tau: F,
    },
    /// Discontinuous Galerkin (DG) for hyperbolic problems
    DiscontinuousGalerkin {
        /// Penalty parameter for interface terms
        penalty: F,
    },
    /// Mixed formulation for Stokes/Darcy flow
    Mixed {
        /// Velocity space polynomial degree
        velocity_degree: usize,
        /// Pressure space polynomial degree
        pressure_degree: usize,
    },
    /// Custom Petrov-Galerkin with user-defined test functions
    Custom {
        /// Test function generator
        test_functions: Box<dyn Fn(F, F) -> Array1<F> + Send + Sync>,
    },
}

/// Petrov-Galerkin finite element solver
pub struct PetrovGalerkinSolver<F: IntegrateFloat> {
    /// Type of Petrov-Galerkin formulation
    formulation: PetrovGalerkinType<F>,
    /// Mesh nodes
    nodes: Array2<F>,
    /// Element connectivity
    elements: Array2<usize>,
    /// Trial function degree
    trial_degree: usize,
    /// Test function degree (may differ from trial)
    test_degree: usize,
}

impl<F: IntegrateFloat> PetrovGalerkinSolver<F> {
    /// Create new Petrov-Galerkin solver
    pub fn new(
        formulation: PetrovGalerkinType<F>,
        nodes: Array2<F>,
        elements: Array2<usize>,
        trial_degree: usize,
        test_degree: usize,
    ) -> Self {
        Self {
            formulation,
            nodes,
            elements,
            trial_degree,
            test_degree,
        }
    }
    
    /// Solve convection-diffusion equation using SUPG method
    pub fn solve_convection_diffusion(
        &self,
        source: impl Fn(F, F) -> F,
        boundary_conditions: &[BoundaryCondition<F>],
    ) -> PDEResult<PDESolution<F>> {
        match &self.formulation {
            PetrovGalerkinType::SUPG { convection, diffusion, tau } => {
                self.solve_supg(convection, *diffusion, *tau, source, boundary_conditions)
            }
            _ => Err(IntegrateError::ValueError(
                "SUPG formulation required for convection-diffusion".to_string()
            )),
        }
    }
    
    /// Solve using Streamline Upwind Petrov-Galerkin (SUPG) method
    fn solve_supg(
        &self,
        convection: &(F, F),
        diffusion: F,
        tau: Option<F>,
        source: impl Fn(F, F) -> F,
        boundary_conditions: &[BoundaryCondition<F>],
    ) -> PDEResult<PDESolution<F>> {
        let n_nodes = self.nodes.nrows();
        let mut stiffness = Array2::<F>::zeros((n_nodes, n_nodes));
        let mut rhs = Array1::<F>::zeros(n_nodes);
        
        // Assemble system matrix with SUPG stabilization
        for element_id in 0..self.elements.nrows() {
            let element = self.elements.row(element_id);
            self.assemble_supg_element(
                element_id,
                element,
                convection,
                diffusion,
                tau,
                &source,
                &mut stiffness,
                &mut rhs,
            )?;
        }
        
        // Apply boundary conditions
        self.apply_boundary_conditions(boundary_conditions, &mut stiffness, &mut rhs)?;
        
        // Solve linear system
        let solution = self.solve_linear_system(stiffness.view(), rhs.view())?;
        
        Ok(PDESolution {
            grids: vec![], // TODO: Add grid information
            values: vec![Array2::from_shape_vec((solution.len(), 1), solution.to_vec()).map_err(|_| IntegrateError::ComputationError("Shape error".to_string()))?],
            error_estimate: None,
            info: PDESolverInfo {
                num_iterations: 1,
                computation_time: 0.0,
                residual_norm: None,
                convergence_history: None,
                method: "Petrov-Galerkin SUPG".to_string(),
            },
        })
    }
    
    /// Assemble SUPG element contribution
    fn assemble_supg_element(
        &self,
        element_id: usize,
        element: ArrayView1<usize>,
        convection: &(F, F),
        diffusion: F,
        tau: Option<F>,
        source: &impl Fn(F, F) -> F,
        stiffness: &mut Array2<F>,
        rhs: &mut Array1<F>,
    ) -> IntegrateResult<()> {
        // Get element nodes
        let node_coords = self.get_element_coordinates(element)?;
        
        // Compute element geometry
        let (det_j, inv_j) = self.compute_jacobian(&node_coords)?;
        let area = det_j.abs() / F::from(2.0).unwrap(); // For triangular elements
        
        // SUPG stabilization parameter
        let tau_supg = tau.unwrap_or_else(|| self.compute_supg_tau(convection, diffusion, &node_coords));
        
        // Integration points and weights (2-point Gauss for triangles)
        let gauss_points = self.get_gauss_points();
        let gauss_weights = self.get_gauss_weights();
        
        for (gp, &weight) in gauss_points.iter().zip(gauss_weights.iter()) {
            let (xi, eta) = (gp[0], gp[1]);
            
            // Trial shape functions and derivatives
            let trial_shapes = self.trial_shape_functions(xi, eta);
            let trial_grads = self.trial_shape_gradients(xi, eta, inv_j.view())?;
            
            // Test functions (SUPG-modified)
            let test_shapes = self.supg_test_functions(xi, eta, convection, tau_supg, inv_j.view())?;
            let test_grads = self.test_shape_gradients(xi, eta, inv_j.view())?;
            
            // Physical coordinates for source evaluation
            let (x, y) = self.map_to_physical(xi, eta, &node_coords);
            let source_val = source(x, y);
            
            // Assemble element matrix
            for i in 0..element.len() {
                let global_i = element[i];
                
                // RHS contribution
                rhs[global_i] = rhs[global_i] + test_shapes[i] * source_val * weight * area;
                
                for j in 0..element.len() {
                    let global_j = element[j];
                    
                    // Diffusion term: ∫ ∇ψᵢ · ν∇φⱼ dx
                    let diffusion_term = diffusion * (
                        test_grads[[i, 0]] * trial_grads[[j, 0]] +
                        test_grads[[i, 1]] * trial_grads[[j, 1]]
                    );
                    
                    // Convection term: ∫ ψᵢ (b·∇φⱼ) dx
                    let convection_term = test_shapes[i] * (
                        convection.0 * trial_grads[[j, 0]] +
                        convection.1 * trial_grads[[j, 1]]
                    );
                    
                    stiffness[[global_i, global_j]] = stiffness[[global_i, global_j]] +
                        (diffusion_term + convection_term) * weight * area;
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute SUPG stabilization parameter
    fn compute_supg_tau(
        &self,
        convection: &(F, F),
        diffusion: F,
        node_coords: &Array2<F>,
    ) -> F {
        // Element size (characteristic length)
        let h = self.compute_element_size(node_coords);
        
        // Convection magnitude
        let b_magnitude = (convection.0 * convection.0 + convection.1 * convection.1).sqrt();
        
        if b_magnitude > F::zero() {
            // Element Peclet number
            let pe = b_magnitude * h / (F::from(2.0).unwrap() * diffusion);
            
            // SUPG parameter: τ = h/(2|b|) * coth(Pe) - 1/(2Pe)
            if pe > F::from(1.0).unwrap() {
                h / (F::from(2.0).unwrap() * b_magnitude)
            } else {
                h * h / (F::from(12.0).unwrap() * diffusion)
            }
        } else {
            F::zero()
        }
    }
    
    /// SUPG-modified test functions
    fn supg_test_functions(
        &self,
        xi: F,
        eta: F,
        convection: &(F, F),
        tau: F,
        inv_j: ArrayView2<F>,
    ) -> IntegrateResult<Array1<F>> {
        let standard_test = self.test_shape_functions(xi, eta);
        let test_grads = self.test_shape_gradients(xi, eta, inv_j)?;
        
        let mut supg_test = standard_test.clone();
        
        // Add SUPG stabilization: ψᵢ + τ(b·∇ψᵢ)
        for i in 0..supg_test.len() {
            let streamline_derivative = convection.0 * test_grads[[i, 0]] + convection.1 * test_grads[[i, 1]];
            supg_test[i] = supg_test[i] + tau * streamline_derivative;
        }
        
        Ok(supg_test)
    }
    
    /// Trial shape functions (standard linear for now)
    fn trial_shape_functions(&self, xi: F, eta: F) -> Array1<F> {
        // Linear triangular shape functions
        let zeta = F::one() - xi - eta;
        Array1::from_vec(vec![zeta, xi, eta])
    }
    
    /// Test shape functions (can be different from trial)
    fn test_shape_functions(&self, xi: F, eta: F) -> Array1<F> {
        // For standard Galerkin, same as trial functions
        // For Petrov-Galerkin, these could be different
        self.trial_shape_functions(xi, eta)
    }
    
    /// Trial shape function gradients
    fn trial_shape_gradients(&self, _xi: F, _eta: F, inv_j: ArrayView2<F>) -> IntegrateResult<Array2<F>> {
        // Linear triangular gradients in reference element
        let ref_grads = Array2::from_shape_vec((3, 2), vec![
            -F::one(), -F::one(),  // ∇N₁
             F::one(),  F::zero(), // ∇N₂
             F::zero(), F::one(),  // ∇N₃
        ]).map_err(|_| IntegrateError::ComputationError("Shape error".to_string()))?;
        
        // Transform to physical element
        let mut phys_grads = Array2::zeros((3, 2));
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..2 {
                    phys_grads[[i, j]] = phys_grads[[i, j]] + ref_grads[[i, k]] * inv_j[[k, j]];
                }
            }
        }
        
        Ok(phys_grads)
    }
    
    /// Test shape function gradients
    fn test_shape_gradients(&self, xi: F, eta: F, inv_j: ArrayView2<F>) -> IntegrateResult<Array2<F>> {
        // For standard formulation, same as trial gradients
        self.trial_shape_gradients(xi, eta, inv_j)
    }
    
    /// Get element coordinates
    fn get_element_coordinates(&self, element: ArrayView1<usize>) -> IntegrateResult<Array2<F>> {
        let mut coords = Array2::zeros((element.len(), 2));
        
        for (i, &node_id) in element.iter().enumerate() {
            if node_id >= self.nodes.nrows() {
                return Err(IntegrateError::ValueError(
                    format!("Invalid node ID: {}", node_id)
                ));
            }
            coords[[i, 0]] = self.nodes[[node_id, 0]];
            coords[[i, 1]] = self.nodes[[node_id, 1]];
        }
        
        Ok(coords)
    }
    
    /// Compute Jacobian matrix and its inverse
    fn compute_jacobian(&self, node_coords: &Array2<F>) -> IntegrateResult<(F, Array2<F>)> {
        // For linear triangular elements
        let x1 = node_coords[[0, 0]]; let y1 = node_coords[[0, 1]];
        let x2 = node_coords[[1, 0]]; let y2 = node_coords[[1, 1]];
        let x3 = node_coords[[2, 0]]; let y3 = node_coords[[2, 1]];
        
        let j11 = x2 - x1; let j12 = x3 - x1;
        let j21 = y2 - y1; let j22 = y3 - y1;
        
        let det_j = j11 * j22 - j12 * j21;
        
        if det_j.abs() < F::from(1e-12).unwrap() {
            return Err(IntegrateError::ComputationError(
                "Degenerate element (zero Jacobian)".to_string()
            ));
        }
        
        let inv_j = Array2::from_shape_vec((2, 2), vec![
            j22 / det_j, -j12 / det_j,
            -j21 / det_j, j11 / det_j,
        ]).map_err(|_| IntegrateError::ComputationError("Shape error".to_string()))?;
        
        Ok((det_j, inv_j))
    }
    
    /// Compute element characteristic size
    fn compute_element_size(&self, node_coords: &Array2<F>) -> F {
        // Diameter of element (max distance between nodes)
        let mut max_dist = F::zero();
        
        for i in 0..node_coords.nrows() {
            for j in (i + 1)..node_coords.nrows() {
                let dx = node_coords[[i, 0]] - node_coords[[j, 0]];
                let dy = node_coords[[i, 1]] - node_coords[[j, 1]];
                let dist = (dx * dx + dy * dy).sqrt();
                
                if dist > max_dist {
                    max_dist = dist;
                }
            }
        }
        
        max_dist
    }
    
    /// Map reference coordinates to physical coordinates
    fn map_to_physical(&self, xi: F, eta: F, node_coords: &Array2<F>) -> (F, F) {
        let shapes = self.trial_shape_functions(xi, eta);
        
        let mut x = F::zero();
        let mut y = F::zero();
        
        for i in 0..node_coords.nrows() {
            x = x + shapes[i] * node_coords[[i, 0]];
            y = y + shapes[i] * node_coords[[i, 1]];
        }
        
        (x, y)
    }
    
    /// Gauss integration points for triangular elements
    fn get_gauss_points(&self) -> Vec<[F; 2]> {
        // 3-point Gauss rule for triangles
        vec![
            [F::from(1.0/6.0).unwrap(), F::from(1.0/6.0).unwrap()],
            [F::from(2.0/3.0).unwrap(), F::from(1.0/6.0).unwrap()],
            [F::from(1.0/6.0).unwrap(), F::from(2.0/3.0).unwrap()],
        ]
    }
    
    /// Gauss integration weights for triangular elements
    fn get_gauss_weights(&self) -> Vec<F> {
        vec![
            F::from(1.0/6.0).unwrap(),
            F::from(1.0/6.0).unwrap(),
            F::from(1.0/6.0).unwrap(),
        ]
    }
    
    /// Apply boundary conditions to system
    fn apply_boundary_conditions(
        &self,
        boundary_conditions: &[BoundaryCondition<F>],
        stiffness: &mut Array2<F>,
        rhs: &mut Array1<F>,
    ) -> IntegrateResult<()> {
        use crate::pde::BoundaryConditionType;
        
        for bc in boundary_conditions {
            match &bc.condition_type {
                BoundaryConditionType::Dirichlet => {
                    // For Dirichlet boundary conditions: u = g on boundary
                    // Modify system: set A[i,i] = 1, A[i,j] = 0 for j≠i, b[i] = g
                    for &node_idx in &bc.nodes {
                        if node_idx < stiffness.nrows() {
                            // Clear row
                            for j in 0..stiffness.ncols() {
                                stiffness[[node_idx, j]] = F::zero();
                            }
                            // Clear column
                            for i in 0..stiffness.nrows() {
                                stiffness[[i, node_idx]] = F::zero();
                            }
                            // Set diagonal entry
                            stiffness[[node_idx, node_idx]] = F::one();
                            // Set RHS value
                            rhs[node_idx] = bc.value;
                        }
                    }
                },
                BoundaryConditionType::Neumann => {
                    // For Neumann boundary conditions: ∂u/∂n = g on boundary
                    // Add flux terms to RHS: ∫ g ψᵢ ds
                    for &node_idx in &bc.nodes {
                        if node_idx < rhs.len() {
                            // Simple approximation: add flux contribution to RHS
                            // In a complete implementation, this would integrate over boundary edges
                            let boundary_length = self.estimate_boundary_length_at_node(node_idx);
                            rhs[node_idx] = rhs[node_idx] + bc.value * boundary_length;
                        }
                    }
                },
                BoundaryConditionType::Robin => {
                    // For Robin boundary conditions: α u + β ∂u/∂n = g on boundary
                    // This modifies both stiffness matrix and RHS
                    for &node_idx in &bc.nodes {
                        if node_idx < stiffness.nrows() {
                            let boundary_length = self.estimate_boundary_length_at_node(node_idx);
                            // Add Robin term to diagonal: α * boundary_length
                            let alpha = bc.robin_alpha.unwrap_or(F::one());
                            stiffness[[node_idx, node_idx]] = stiffness[[node_idx, node_idx]] + 
                                alpha * boundary_length;
                            // Add to RHS: g * boundary_length
                            rhs[node_idx] = rhs[node_idx] + bc.value * boundary_length;
                        }
                    }
                },
                BoundaryConditionType::Periodic => {
                    // For periodic boundary conditions, couple corresponding nodes
                    // This requires identifying paired nodes on opposite boundaries
                    if bc.nodes.len() >= 2 {
                        for i in 0..(bc.nodes.len() / 2) {
                            let node1 = bc.nodes[i];
                            let node2 = bc.nodes[bc.nodes.len() / 2 + i];
                            
                            if node1 < stiffness.nrows() && node2 < stiffness.nrows() {
                                // Constraint: u[node1] - u[node2] = 0
                                // Add penalty method: λ(u₁ - u₂) = 0
                                let penalty = F::from(1e6).unwrap(); // Large penalty parameter
                                
                                stiffness[[node1, node1]] = stiffness[[node1, node1]] + penalty;
                                stiffness[[node2, node2]] = stiffness[[node2, node2]] + penalty;
                                stiffness[[node1, node2]] = stiffness[[node1, node2]] - penalty;
                                stiffness[[node2, node1]] = stiffness[[node2, node1]] - penalty;
                            }
                        }
                    }
                },
            }
        }
        
        Ok(())
    }
    
    /// Estimate boundary length contribution at a node (simplified)
    fn estimate_boundary_length_at_node(&self, _node_idx: usize) -> F {
        // Simplified estimate - in a complete implementation this would
        // compute the actual boundary segment length associated with the node
        F::from(0.1).unwrap() // Default boundary segment length
    }
    
    /// Solve linear system Ax = b
    fn solve_linear_system(&self, a: ArrayView2<F>, b: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        // Simple Gaussian elimination (for demonstration)
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));
        
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
            if max_row != k {
                for j in 0..=n {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }
            
            // Check for singular matrix
            if aug[[k, k]].abs() < F::from(1e-12).unwrap() {
                return Err(IntegrateError::ComputationError(
                    "Singular matrix in linear system".to_string()
                ));
            }
            
            // Eliminate
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
                }
            }
        }
        
        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum = sum - aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }
        
        Ok(x)
    }
}

/// Stabilized formulation factory
pub struct StabilizedFormulations;

impl StabilizedFormulations {
    /// Create SUPG formulation for convection-diffusion
    pub fn supg<F: IntegrateFloat>(
        convection: (F, F),
        diffusion: F,
    ) -> PetrovGalerkinType<F> {
        PetrovGalerkinType::SUPG {
            convection,
            diffusion,
            tau: None, // Auto-compute
        }
    }
    
    /// Create GLS formulation for general stability
    pub fn gls<F: IntegrateFloat>(tau: F) -> PetrovGalerkinType<F> {
        PetrovGalerkinType::GLS { tau }
    }
    
    /// Create discontinuous Galerkin formulation
    pub fn discontinuous_galerkin<F: IntegrateFloat>(penalty: F) -> PetrovGalerkinType<F> {
        PetrovGalerkinType::DiscontinuousGalerkin { penalty }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_supg_formulation_creation() {
        let formulation = StabilizedFormulations::supg((1.0, 0.5), 0.1);
        
        match formulation {
            PetrovGalerkinType::SUPG { convection, diffusion, tau } => {
                assert_abs_diff_eq!(convection.0, 1.0);
                assert_abs_diff_eq!(convection.1, 0.5);
                assert_abs_diff_eq!(diffusion, 0.1);
                assert!(tau.is_none()); // Auto-compute
            }
            _ => panic!("Wrong formulation type"),
        }
    }
    
    #[test]
    fn test_shape_functions() {
        // Create simple triangular mesh
        let nodes = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let elements = Array2::from_shape_vec((1, 3), vec![0, 1, 2]).unwrap();
        
        let formulation = StabilizedFormulations::supg((1.0, 0.0), 0.1);
        let solver = PetrovGalerkinSolver::new(formulation, nodes, elements, 1, 1);
        
        // Test shape functions at element center
        let shapes = solver.trial_shape_functions(1.0/3.0, 1.0/3.0);
        
        // At center of reference triangle, all shape functions should be 1/3
        assert_abs_diff_eq!(shapes[0], 1.0/3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(shapes[1], 1.0/3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(shapes[2], 1.0/3.0, epsilon = 1e-10);
        
        // Partition of unity
        let sum: f64 = shapes.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jacobian_computation() {
        let nodes = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let elements = Array2::from_shape_vec((1, 3), vec![0, 1, 2]).unwrap();
        let formulation = StabilizedFormulations::supg((1.0, 0.0), 0.1);
        let solver = PetrovGalerkinSolver::new(formulation, nodes, elements, 1, 1);
        
        let element_coords = solver.get_element_coordinates(elements.row(0)).unwrap();
        let (det_j, _inv_j) = solver.compute_jacobian(&element_coords).unwrap();
        
        // For unit right triangle, Jacobian determinant should be 1
        assert_abs_diff_eq!(det_j, 1.0, epsilon = 1e-10);
    }
}