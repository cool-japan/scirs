//! Types for DAE solver module
//!
//! This module defines the core types used by DAE solvers,
//! including method enums, options, and results.

use crate::common::IntegrateFloat;
use crate::ode::ODEMethod;
use ndarray::Array1;

/// DAE system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DAEType {
    /// Semi-explicit index-1 DAE of the form:
    /// x' = f(x, y, t)
    /// 0 = g(x, y, t)
    #[default]
    SemiExplicit,

    /// Fully implicit index-1 DAE of the form:
    /// F(x', x, t) = 0
    FullyImplicit,
}

/// DAE index classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DAEIndex {
    /// Index-1 (differentiate once to get an ODE)
    #[default]
    Index1,
    /// Index-2 (differentiate twice)
    Index2,
    /// Index-3 (differentiate three times)
    Index3,
    /// Higher-index (differentiate more than three times)
    HigherIndex,
}

/// Options for controlling the behavior of DAE solvers
#[derive(Debug, Clone)]
pub struct DAEOptions<F: IntegrateFloat> {
    /// The DAE system type
    pub dae_type: DAEType,

    /// The DAE index (classification)
    pub index: DAEIndex,

    /// The ODE solver method to use for the differential part
    pub method: ODEMethod,

    /// Relative tolerance for error control
    pub rtol: F,

    /// Absolute tolerance for error control
    pub atol: F,

    /// Initial step size (optional, if not provided, it will be estimated)
    pub h0: Option<F>,

    /// Maximum number of steps to take
    pub max_steps: usize,

    /// Maximum step size (optional)
    pub max_step: Option<F>,

    /// Minimum step size (optional)
    pub min_step: Option<F>,

    /// Maximum iterations for the nonlinear solver
    pub max_newton_iterations: usize,

    /// Tolerance for the nonlinear solver
    pub newton_tol: F,

    /// Maximum BDF order (optional, defaults to 5)
    pub max_order: Option<usize>,
}

impl<F: IntegrateFloat> Default for DAEOptions<F> {
    fn default() -> Self {
        DAEOptions {
            dae_type: DAEType::default(),
            index: DAEIndex::default(),
            method: ODEMethod::Radau, // Use Radau by default, as it's suitable for index-1 DAEs
            rtol: F::from_f64(1e-3).unwrap(),
            atol: F::from_f64(1e-6).unwrap(),
            h0: None,
            max_steps: 500,
            max_step: None,
            min_step: None,
            max_newton_iterations: 10,
            newton_tol: F::from_f64(1e-8).unwrap(),
            max_order: None,
        }
    }
}

/// Result of DAE integration
#[derive(Debug, Clone)]
pub struct DAEResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,

    /// Solution values for differential variables at time points
    pub x: Vec<Array1<F>>,

    /// Solution values for algebraic variables at time points
    pub y: Vec<Array1<F>>,

    /// Whether the integration was successful
    pub success: bool,

    /// Status message
    pub message: Option<String>,

    /// Number of function evaluations
    pub n_eval: usize,

    /// Number of constraint evaluations
    pub n_constraint_eval: usize,

    /// Number of steps taken
    pub n_steps: usize,

    /// Number of accepted steps
    pub n_accepted: usize,

    /// Number of rejected steps
    pub n_rejected: usize,

    /// Number of LU decompositions
    pub n_lu: usize,

    /// Number of Jacobian evaluations
    pub n_jac: usize,

    /// The solver method used
    pub method: ODEMethod,

    /// The DAE type
    pub dae_type: DAEType,

    /// The DAE index
    pub index: DAEIndex,
}
