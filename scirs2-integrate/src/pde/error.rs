use std::error::Error;
use std::fmt;

use crate::error::IntegrateError;

/// Errors that may occur in PDE-related operations
#[derive(Debug)]
pub enum PDEError {
    /// Error with boundary conditions specification
    BoundaryConditions(String),

    /// Error with domain specification
    DomainError(String),

    /// Error with discretization
    DiscretizationError(String),

    /// Error with method of lines integration
    MOLError(String),

    /// Error with finite difference computation
    FiniteDifferenceError(String),

    /// Error with finite element computation
    FiniteElementError(String),

    /// Error with spectral method computation
    SpectralError(String),

    /// Underlying ODE solver error during method of lines integration
    ODEError(IntegrateError),

    /// Other PDE-related errors
    Other(String),
}

impl fmt::Display for PDEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PDEError::BoundaryConditions(msg) => write!(f, "Boundary condition error: {}", msg),
            PDEError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            PDEError::DiscretizationError(msg) => write!(f, "Discretization error: {}", msg),
            PDEError::MOLError(msg) => write!(f, "Method of lines error: {}", msg),
            PDEError::FiniteDifferenceError(msg) => write!(f, "Finite difference error: {}", msg),
            PDEError::FiniteElementError(msg) => write!(f, "Finite element error: {}", msg),
            PDEError::SpectralError(msg) => write!(f, "Spectral method error: {}", msg),
            PDEError::ODEError(err) => write!(f, "ODE solver error: {}", err),
            PDEError::Other(msg) => write!(f, "PDE error: {}", msg),
        }
    }
}

impl Error for PDEError {}

/// Result type for PDE operations
pub type PDEResult<T> = Result<T, PDEError>;

// Conversion from IntegrateError to PDEError
impl From<IntegrateError> for PDEError {
    fn from(err: IntegrateError) -> Self {
        PDEError::ODEError(err)
    }
}
