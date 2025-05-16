//! Options for ODE solvers.

/// Method to use for solving ODEs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ODEMethod {
    /// Explicit Euler method (first-order).
    Euler,
    /// 4th-order Runge-Kutta method.
    RK4,
    /// Adaptive 4th/5th-order Runge-Kutta method.
    RK45,
    /// Adaptive 2nd/3rd-order Runge-Kutta method.
    RK23,
    /// 8th-order Dormand-Prince method.
    DOP853,
    /// Backward Differentiation Formula (stiff problems).
    BDF,
    /// Implicit Runge-Kutta method (Radau IIA).
    Radau,
    /// LSODA method (automatically switches between stiff and non-stiff methods).
    LSODA,
}

/// Options for ODE solvers.
#[derive(Debug, Clone)]
pub struct ODEOptions {
    /// The method to use for solving the ODE.
    pub method: ODEMethod,
    /// The relative tolerance for adaptive step size methods.
    pub rtol: f64,
    /// The absolute tolerance for adaptive step size methods.
    pub atol: f64,
    /// The initial step size. If None, the solver will estimate a suitable value.
    pub first_step: Option<f64>,
    /// The maximum step size.
    pub max_step: Option<f64>,
    /// The maximum number of steps allowed.
    pub max_steps: usize,
    /// Whether to use dense output (interpolation between steps).
    pub dense_output: bool,
}

impl Default for ODEOptions {
    fn default() -> Self {
        Self {
            method: ODEMethod::RK45,
            rtol: 1e-3,
            atol: 1e-6,
            first_step: None,
            max_step: None,
            max_steps: 500,
            dense_output: false,
        }
    }
}

impl ODEOptions {
    /// Create a new set of options with the given method.
    pub fn new(method: ODEMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Set the relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set the absolute tolerance.
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    /// Set the initial step size.
    pub fn with_first_step(mut self, first_step: f64) -> Self {
        self.first_step = Some(first_step);
        self
    }

    /// Set the maximum step size.
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = Some(max_step);
        self
    }

    /// Set the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Enable or disable dense output.
    pub fn with_dense_output(mut self, dense_output: bool) -> Self {
        self.dense_output = dense_output;
        self
    }
}
