//! Error types for ODE solvers.

use std::error::Error;
use std::fmt;

/// Errors that can occur during ODE solving.
#[derive(Debug)]
pub enum ODEError {
    /// The solver failed to converge to a solution.
    Convergence,
    /// The step size became too small.
    StepSizeTooSmall,
    /// The solver exceeded the maximum number of steps.
    MaxStepsExceeded,
    /// The solver encountered NaN or Inf values.
    NumericalInstability,
    /// An invalid input was provided to the solver.
    InvalidInput(String),
    /// A general error occurred.
    General(String),
}

impl fmt::Display for ODEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ODEError::Convergence => write!(f, "Failed to converge to a solution"),
            ODEError::StepSizeTooSmall => write!(f, "Step size became too small"),
            ODEError::MaxStepsExceeded => write!(f, "Exceeded maximum number of steps"),
            ODEError::NumericalInstability => write!(f, "Encountered numerical instability (NaN or Inf)"),
            ODEError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ODEError::General(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for ODEError {}
