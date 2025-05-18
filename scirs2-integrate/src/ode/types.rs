//! Types for ODE solver module
//!
//! This module defines the core types used by ODE solvers,
//! including method enums, options, and results.

use crate::common::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Debug;
use std::sync::Arc;

/// Type alias for time-dependent matrix function
pub type TimeFunction<F> = Arc<dyn Fn(F) -> Array2<F> + Send + Sync>;

/// Type alias for state-dependent matrix function  
pub type StateFunction<F> = Arc<dyn Fn(F, ArrayView1<F>) -> Array2<F> + Send + Sync>;

/// ODE solver method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ODEMethod {
    /// Euler method (first-order)
    Euler,
    /// Fourth-order Runge-Kutta method (fixed step size)
    RK4,
    /// Dormand-Prince method (variable step size)
    /// 5th order method with 4th order error estimate
    #[default]
    RK45,
    /// Bogacki-Shampine method (variable step size)
    /// 3rd order method with 2nd order error estimate
    RK23,
    /// Backward Differentiation Formula (BDF) method
    /// Implicit method for stiff equations
    /// Default is BDF order 2
    Bdf,
    /// Dormand-Prince method of order 8(5,3)
    /// 8th order method with 5th order error estimate
    /// High-accuracy explicit Runge-Kutta method
    DOP853,
    /// Implicit Runge-Kutta method of Radau IIA family
    /// 5th order method with 3rd order error estimate
    /// L-stable implicit method for stiff problems
    Radau,
    /// Livermore Solver for Ordinary Differential Equations with Automatic method switching
    /// Automatically switches between Adams methods (non-stiff) and BDF (stiff)
    /// Efficiently handles problems that change character during integration
    LSODA,
    /// Enhanced LSODA method with improved stiffness detection and method switching
    /// Features better Jacobian handling, adaptive order selection, and robust error control
    /// Provides detailed diagnostics about method switching decisions
    EnhancedLSODA,
    /// Enhanced BDF method with improved Jacobian handling and error estimation
    /// Features intelligent Jacobian strategy selection based on problem size
    /// Supports multiple Newton solver variants and provides better convergence
    /// Includes specialized handling for banded matrices and adaptive order selection
    EnhancedBDF,
}

/// Type of mass matrix for ODE system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassMatrixType {
    /// Identity mass matrix (standard ODE)
    #[default]
    Identity,
    /// Constant mass matrix
    Constant,
    /// Time-dependent mass matrix M(t)
    TimeDependent,
    /// State-dependent mass matrix M(t,y)
    StateDependent,
}

/// Mass matrix for ODE system of the form M(t,y)·y' = f(t,y)
pub struct MassMatrix<F: IntegrateFloat> {
    /// Type of the mass matrix
    pub matrix_type: MassMatrixType,
    /// Constant mass matrix (if applicable)
    pub constant_matrix: Option<ndarray::Array2<F>>,
    /// Function for time-dependent mass matrix
    pub time_function: Option<TimeFunction<F>>,
    /// Function for state-dependent mass matrix
    pub state_function: Option<StateFunction<F>>,
    /// Whether the mass matrix is sparse/banded
    pub is_banded: bool,
    /// Lower bandwidth for banded matrices
    pub lower_bandwidth: Option<usize>,
    /// Upper bandwidth for banded matrices
    pub upper_bandwidth: Option<usize>,
}

impl<F: IntegrateFloat> Debug for MassMatrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MassMatrix")
            .field("matrix_type", &self.matrix_type)
            .field("constant_matrix", &self.constant_matrix)
            .field("time_function", &self.time_function.is_some())
            .field("state_function", &self.state_function.is_some())
            .field("is_banded", &self.is_banded)
            .field("lower_bandwidth", &self.lower_bandwidth)
            .field("upper_bandwidth", &self.upper_bandwidth)
            .finish()
    }
}

impl<F: IntegrateFloat> Clone for MassMatrix<F> {
    fn clone(&self) -> Self {
        MassMatrix {
            matrix_type: self.matrix_type,
            constant_matrix: self.constant_matrix.clone(),
            time_function: self.time_function.clone(),
            state_function: self.state_function.clone(),
            is_banded: self.is_banded,
            lower_bandwidth: self.lower_bandwidth,
            upper_bandwidth: self.upper_bandwidth,
        }
    }
}

impl<F: IntegrateFloat> MassMatrix<F> {
    /// Create a new identity mass matrix (standard ODE)
    pub fn identity() -> Self {
        MassMatrix {
            matrix_type: MassMatrixType::Identity,
            constant_matrix: None,
            time_function: None,
            state_function: None,
            is_banded: false,
            lower_bandwidth: None,
            upper_bandwidth: None,
        }
    }

    /// Create a new constant mass matrix
    pub fn constant(matrix: ndarray::Array2<F>) -> Self {
        MassMatrix {
            matrix_type: MassMatrixType::Constant,
            constant_matrix: Some(matrix),
            time_function: None,
            state_function: None,
            is_banded: false,
            lower_bandwidth: None,
            upper_bandwidth: None,
        }
    }

    /// Create a new time-dependent mass matrix M(t)
    pub fn time_dependent<Func>(func: Func) -> Self
    where
        Func: Fn(F) -> ndarray::Array2<F> + Send + Sync + 'static,
    {
        MassMatrix {
            matrix_type: MassMatrixType::TimeDependent,
            constant_matrix: None,
            time_function: Some(Arc::new(func)),
            state_function: None,
            is_banded: false,
            lower_bandwidth: None,
            upper_bandwidth: None,
        }
    }

    /// Create a new state-dependent mass matrix M(t,y)
    pub fn state_dependent<Func>(func: Func) -> Self
    where
        Func: Fn(F, ndarray::ArrayView1<F>) -> ndarray::Array2<F> + Send + Sync + 'static,
    {
        MassMatrix {
            matrix_type: MassMatrixType::StateDependent,
            constant_matrix: None,
            time_function: None,
            state_function: Some(Arc::new(func)),
            is_banded: false,
            lower_bandwidth: None,
            upper_bandwidth: None,
        }
    }

    /// Set the matrix as banded with specified bandwidths
    pub fn with_bandwidth(mut self, lower: usize, upper: usize) -> Self {
        self.is_banded = true;
        self.lower_bandwidth = Some(lower);
        self.upper_bandwidth = Some(upper);
        self
    }

    /// Get the mass matrix at a given time and state
    pub fn evaluate(&self, t: F, y: ndarray::ArrayView1<F>) -> Option<ndarray::Array2<F>> {
        match self.matrix_type {
            MassMatrixType::Identity => None, // Identity is handled specially
            MassMatrixType::Constant => self.constant_matrix.clone(),
            MassMatrixType::TimeDependent => self.time_function.as_ref().map(|f| f(t)),
            MassMatrixType::StateDependent => self.state_function.as_ref().map(|f| f(t, y)),
        }
    }
}

/// Options for controlling the behavior of ODE solvers
#[derive(Debug, Clone)]
pub struct ODEOptions<F: IntegrateFloat> {
    /// The ODE solver method to use
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
    /// Dense output flag - whether to enable dense output
    pub dense_output: bool,
    /// Maximum order for BDF method (1-5)
    pub max_order: Option<usize>,
    /// Jacobian matrix (optional, for implicit methods)
    pub jac: Option<Array1<F>>,
    /// Whether to use a banded Jacobian matrix
    pub use_banded_jacobian: bool,
    /// Number of lower diagonals for banded Jacobian
    pub ml: Option<usize>,
    /// Number of upper diagonals for banded Jacobian
    pub mu: Option<usize>,
    /// Mass matrix for M(t,y)·y' = f(t,y) form (optional)
    pub mass_matrix: Option<MassMatrix<F>>,
    /// Strategy for Jacobian approximation/computation
    pub jacobian_strategy: Option<crate::ode::utils::jacobian::JacobianStrategy>,
}

impl<F: IntegrateFloat> Default for ODEOptions<F> {
    fn default() -> Self {
        ODEOptions {
            method: ODEMethod::default(),
            rtol: F::from_f64(1e-3).unwrap(),
            atol: F::from_f64(1e-6).unwrap(),
            h0: None,
            max_steps: 500,
            max_step: None,
            min_step: None,
            dense_output: false,
            max_order: None,
            jac: None,
            use_banded_jacobian: false,
            ml: None,
            mu: None,
            mass_matrix: None,
            jacobian_strategy: None, // Defaults to Adaptive in JacobianManager
        }
    }
}

/// Result of ODE integration
#[derive(Debug, Clone)]
pub struct ODEResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Solution values at time points
    pub y: Vec<Array1<F>>,
    /// Whether the integration was successful
    pub success: bool,
    /// Status message
    pub message: Option<String>,
    /// Number of function evaluations
    pub n_eval: usize,
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
}
