//! Partial Differential Equation (PDE) solvers
//!
//! This module provides implementations of various numerical methods for
//! solving partial differential equations (PDEs).
//!
//! ## Supported Methods
//!
//! * Method of Lines (MOL): Converts PDEs to systems of ODEs by discretizing spatial derivatives
//! * Finite Difference Methods: Approximates derivatives using differences between grid points
//! * Finite Element Methods: Approximates solutions using basis functions on a mesh
//! * Spectral Methods: Approximates solutions using global basis functions
//!
//! ## Supported Equation Types
//!
//! * Parabolic PDEs (e.g., heat equation)
//! * Hyperbolic PDEs (e.g., wave equation)
//! * Elliptic PDEs (e.g., Poisson equation)
//! * Systems of coupled PDEs

pub mod error;
pub use error::{PDEError, PDEResult};

// Submodules for different PDE solution approaches
pub mod amr;
pub mod elliptic;
pub mod finite_difference;
pub mod finite_element;
pub mod implicit;
pub mod mesh_generation;
pub mod method_of_lines;
pub mod spectral;

use scirs2_core::ndarray::{Array1, Array2};
use std::ops::Range;

/// Enum representing different types of boundary conditions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryConditionType {
    /// Dirichlet boundary condition (fixed value)
    Dirichlet,

    /// Neumann boundary condition (fixed derivative)
    Neumann,

    /// Robin/mixed boundary condition (linear combination of value and derivative)
    Robin,

    /// Periodic boundary condition
    Periodic,
}

/// Struct representing a boundary condition for a PDE
#[derive(Debug, Clone)]
pub struct BoundaryCondition<F: 'static + std::fmt::Debug + Copy + PartialOrd> {
    /// Type of boundary condition
    pub bc_type: BoundaryConditionType,

    /// Location of the boundary (low or high end of a dimension)
    pub location: BoundaryLocation,

    /// Dimension to which this boundary condition applies
    pub dimension: usize,

    /// Value for Dirichlet conditions, or derivative value for Neumann conditions
    pub value: F,

    /// Coefficients for Robin boundary conditions (a*u + b*du/dn = c)
    /// For Robin conditions: [a, b, c]
    pub coefficients: Option<[F; 3]>,
}

/// Enum representing the location of a boundary
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryLocation {
    /// Lower boundary of the domain
    Lower,

    /// Upper boundary of the domain
    Upper,
}

/// Domain for the PDE problem
#[derive(Debug, Clone)]
pub struct Domain {
    /// Ranges defining the spatial domain for each dimension
    pub ranges: Vec<Range<f64>>,

    /// Number of grid points in each dimension
    pub grid_points: Vec<usize>,
}

impl Domain {
    /// Create a new domain with given ranges and number of grid points
    pub fn new(ranges: Vec<Range<f64>>, grid_points: Vec<usize>) -> PDEResult<Self> {
        if ranges.len() != grid_points.len() {
            return Err(PDEError::DomainError(
                "Number of ranges must match number of grid point specifications".to_string(),
            ));
        }

        for (i, range) in ranges.iter().enumerate() {
            if range.start >= range.end {
                return Err(PDEError::DomainError(format!(
                    "Invalid range for dimension {i}: start must be less than end"
                )));
            }

            if grid_points[i] < 3 {
                return Err(PDEError::DomainError(format!(
                    "At least 3 grid points required for dimension {i}"
                )));
            }
        }

        Ok(Domain {
            ranges,
            grid_points,
        })
    }

    /// Get the number of dimensions in the domain
    pub fn dimensions(&self) -> usize {
        self.ranges.len()
    }

    /// Get the grid spacing for a given dimension
    pub fn grid_spacing(&self, dimension: usize) -> PDEResult<f64> {
        if dimension >= self.dimensions() {
            return Err(PDEError::DomainError(format!(
                "Invalid dimension: {dimension}"
            )));
        }

        let range = &self.ranges[dimension];
        let n_points = self.grid_points[dimension];

        Ok((range.end - range.start) / (n_points - 1) as f64)
    }

    /// Generate a grid for the given dimension
    pub fn grid(&self, dimension: usize) -> PDEResult<Array1<f64>> {
        if dimension >= self.dimensions() {
            return Err(PDEError::DomainError(format!(
                "Invalid dimension: {dimension}"
            )));
        }

        let range = &self.ranges[dimension];
        let n_points = self.grid_points[dimension];
        let dx = (range.end - range.start) / ((n_points - 1) as f64);

        let mut grid = Array1::zeros(n_points);
        for i in 0..n_points {
            grid[i] = range.start + (i as f64) * dx;
        }

        Ok(grid)
    }

    /// Get the total number of grid points in the domain
    pub fn total_grid_points(&self) -> usize {
        self.grid_points.iter().product()
    }
}

/// Trait for PDE problems
pub trait PDEProblem<F: 'static + std::fmt::Debug + Copy + PartialOrd> {
    /// Get the domain of the PDE problem
    fn domain(&self) -> &Domain;

    /// Get the boundary conditions of the PDE problem
    fn boundary_conditions(&self) -> &[BoundaryCondition<F>];

    /// Get the number of dependent variables in the PDE
    fn num_variables() -> usize;

    /// Get the PDE terms (implementation depends on the specific PDE type)
    fn pde_terms() -> PDEResult<()>;
}

/// Trait for PDE solvers
pub trait PDESolver<F: 'static + std::fmt::Debug + Copy + PartialOrd> {
    /// Solve the PDE problem
    fn solve() -> PDEResult<PDESolution<F>>;
}

/// Solution to a PDE problem
#[derive(Debug, Clone)]
pub struct PDESolution<F: 'static + std::fmt::Debug + Copy + PartialOrd> {
    /// Grid points in each dimension
    pub grids: Vec<Array1<f64>>,

    /// Solution values
    /// For a 1D problem with one variable: u(x)
    /// For a 2D problem with one variable: u(x,y)
    /// For a 1D problem with two variables: [u(x), v(x)]
    /// Shape depends on the problem dimensions and number of variables
    pub values: Vec<Array2<F>>,

    /// Error estimate (if available)
    pub error_estimate: Option<Vec<Array2<F>>>,

    /// Additional solver information
    pub info: PDESolverInfo,
}

/// Information about the PDE solver run
#[derive(Debug, Clone)]
pub struct PDESolverInfo {
    /// Number of iterations performed
    pub num_iterations: usize,

    /// Computation time in seconds
    pub computation_time: f64,

    /// Final residual norm
    pub residual_norm: Option<f64>,

    /// Convergence history
    pub convergence_history: Option<Vec<f64>>,

    /// Method used to solve the PDE
    pub method: String,
}

/// Enum representing PDE types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PDEType {
    /// Parabolic PDE (e.g., heat equation)
    Parabolic,

    /// Hyperbolic PDE (e.g., wave equation)
    Hyperbolic,

    /// Elliptic PDE (e.g., Poisson equation)
    Elliptic,

    /// Mixed type PDE
    Mixed,
}
