//! Port-Hamiltonian System definition
//!
//! A port-Hamiltonian system (pH system) is a class of dynamical systems that
//! combines Hamiltonian mechanics with network structure and energy ports. The
//! standard formulation is:
//!
//! ```text
//! dx/dt = (J(x) - R(x)) * ∇H(x) + B(x) * u
//!     y = B(x)^T * ∇H(x)
//! ```
//!
//! where:
//! - `J(x)` is the skew-symmetric interconnection matrix (structure matrix)
//! - `R(x) >= 0` is the positive semi-definite dissipation matrix
//! - `H(x)` is the Hamiltonian (energy storage function)
//! - `∇H(x)` is the gradient of the Hamiltonian
//! - `B(x)` is the input matrix (port matrix)
//! - `u` is the control input
//! - `y` is the output (power conjugate to `u`)
//!
//! The key property is the energy balance:
//! ```text
//! dH/dt = -∇H^T * R * ∇H + y^T * u ≤ y^T * u
//! ```
//! which states that the system can only dissipate energy internally; external
//! energy can only enter through the ports.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Type alias for a state-dependent matrix function.
pub type MatrixFn = Box<dyn Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync>;

/// Type alias for a scalar function of the state.
pub type ScalarFn = Box<dyn Fn(&[f64]) -> IntegrateResult<f64> + Send + Sync>;

/// Type alias for a vector function of the state.
pub type VectorFn = Box<dyn Fn(&[f64]) -> IntegrateResult<Array1<f64>> + Send + Sync>;

/// Configuration options for a Port-Hamiltonian system.
#[derive(Debug, Clone)]
pub struct PortHamiltonianConfig {
    /// Tolerance for verifying skew-symmetry of J(x)
    pub skew_sym_tol: f64,
    /// Tolerance for verifying positive semi-definiteness of R(x)
    pub psd_tol: f64,
    /// Step size for numerical gradient computation
    pub grad_epsilon: f64,
}

impl Default for PortHamiltonianConfig {
    fn default() -> Self {
        Self {
            skew_sym_tol: 1e-10,
            psd_tol: -1e-10,
            grad_epsilon: 1e-7,
        }
    }
}

/// A Port-Hamiltonian System (pH system).
///
/// This struct encapsulates all the components needed to define a port-Hamiltonian
/// system in standard form:
///
/// ```text
/// dx/dt = (J(x) - R(x)) * ∇H(x) + B * u
///     y = B^T * ∇H(x)
/// ```
///
/// # Energy Balance
///
/// The system satisfies:
/// ```text
/// dH/dt ≤ u^T * y
/// ```
/// meaning the system is passive: it can only dissipate, not generate, energy.
pub struct PortHamiltonianSystem {
    /// State dimension
    pub n_states: usize,
    /// Input/output port dimension
    pub n_ports: usize,
    /// Skew-symmetric interconnection matrix function J(x).
    /// Must satisfy J(x) + J(x)^T = 0 for all x.
    j_matrix: MatrixFn,
    /// Positive semi-definite dissipation matrix function R(x).
    /// Must satisfy v^T R(x) v >= 0 for all x, v.
    r_matrix: MatrixFn,
    /// Hamiltonian energy function H(x).
    hamiltonian: ScalarFn,
    /// Gradient of the Hamiltonian ∇H(x).
    /// If None, computed numerically via finite differences.
    grad_hamiltonian: Option<VectorFn>,
    /// Input/output port matrix B (constant or state-dependent).
    b_matrix: MatrixFn,
    /// Configuration options
    config: PortHamiltonianConfig,
}

impl PortHamiltonianSystem {
    /// Create a new Port-Hamiltonian system with constant B matrix.
    ///
    /// # Arguments
    ///
    /// * `n_states` - Dimension of the state vector
    /// * `n_ports` - Dimension of the port (input/output)
    /// * `j_fn` - Skew-symmetric interconnection matrix function
    /// * `r_fn` - Positive semi-definite dissipation matrix function
    /// * `hamiltonian` - Hamiltonian energy function
    /// * `b_matrix` - Constant input/output port matrix of shape (n_states, n_ports)
    pub fn new(
        n_states: usize,
        n_ports: usize,
        j_fn: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
        r_fn: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
        hamiltonian: impl Fn(&[f64]) -> IntegrateResult<f64> + Send + Sync + 'static,
        b_matrix: Array2<f64>,
    ) -> Self {
        let b = b_matrix.clone();
        Self {
            n_states,
            n_ports,
            j_matrix: Box::new(j_fn),
            r_matrix: Box::new(r_fn),
            hamiltonian: Box::new(hamiltonian),
            grad_hamiltonian: None,
            b_matrix: Box::new(move |_x| Ok(b.clone())),
            config: PortHamiltonianConfig::default(),
        }
    }

    /// Create a port-Hamiltonian system with an analytic gradient for the Hamiltonian.
    ///
    /// Providing an analytic gradient improves both accuracy and performance of
    /// structure-preserving integrators.
    pub fn with_grad_hamiltonian(
        mut self,
        grad_fn: impl Fn(&[f64]) -> IntegrateResult<Array1<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.grad_hamiltonian = Some(Box::new(grad_fn));
        self
    }

    /// Set a state-dependent B matrix (input matrix as a function of x).
    pub fn with_state_dependent_b(
        mut self,
        b_fn: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.b_matrix = Box::new(b_fn);
        self
    }

    /// Set configuration options.
    pub fn with_config(mut self, config: PortHamiltonianConfig) -> Self {
        self.config = config;
        self
    }

    /// Evaluate the Hamiltonian H(x).
    pub fn hamiltonian(&self, x: &[f64]) -> IntegrateResult<f64> {
        (self.hamiltonian)(x)
    }

    /// Evaluate the gradient of the Hamiltonian ∇H(x).
    ///
    /// If an analytic gradient was provided, it is used directly.
    /// Otherwise, the gradient is computed via central finite differences.
    pub fn grad_hamiltonian(&self, x: &[f64]) -> IntegrateResult<Array1<f64>> {
        if let Some(ref grad_fn) = self.grad_hamiltonian {
            return grad_fn(x);
        }
        // Numerical gradient via central differences
        let n = x.len();
        let eps = self.config.grad_epsilon;
        let mut grad = Array1::zeros(n);
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        for i in 0..n {
            x_plus[i] = x[i] + eps;
            x_minus[i] = x[i] - eps;
            let h_plus = (self.hamiltonian)(&x_plus)?;
            let h_minus = (self.hamiltonian)(&x_minus)?;
            grad[i] = (h_plus - h_minus) / (2.0 * eps);
            x_plus[i] = x[i];
            x_minus[i] = x[i];
        }
        Ok(grad)
    }

    /// Evaluate the skew-symmetric interconnection matrix J(x).
    pub fn j_matrix(&self, x: &[f64]) -> IntegrateResult<Array2<f64>> {
        (self.j_matrix)(x)
    }

    /// Evaluate the dissipation matrix R(x).
    pub fn r_matrix(&self, x: &[f64]) -> IntegrateResult<Array2<f64>> {
        (self.r_matrix)(x)
    }

    /// Evaluate the port matrix B(x).
    pub fn b_matrix(&self, x: &[f64]) -> IntegrateResult<Array2<f64>> {
        (self.b_matrix)(x)
    }

    /// Evaluate the right-hand side of the ODE: f(x, u) = (J(x) - R(x)) ∇H(x) + B(x) u
    pub fn rhs(&self, x: &[f64], u: &[f64]) -> IntegrateResult<Array1<f64>> {
        let j = self.j_matrix(x)?;
        let r = self.r_matrix(x)?;
        let grad_h = self.grad_hamiltonian(x)?;
        let b = self.b_matrix(x)?;

        // (J - R) * ∇H
        let jr = &j - &r;
        let jr_grad = jr.dot(&grad_h);

        // B * u
        let u_arr = Array1::from_vec(u.to_vec());
        let b_u = b.dot(&u_arr);

        Ok(jr_grad + b_u)
    }

    /// Evaluate the output: y = B(x)^T * ∇H(x)
    pub fn output(&self, x: &[f64]) -> IntegrateResult<Array1<f64>> {
        let grad_h = self.grad_hamiltonian(x)?;
        let b = self.b_matrix(x)?;
        Ok(b.t().dot(&grad_h))
    }

    /// Compute the power balance: dH/dt = -∇H^T R ∇H + y^T u
    ///
    /// Returns (dissipation, supply_rate) where:
    /// - dissipation = ∇H^T R ∇H >= 0
    /// - supply_rate = y^T u (power supplied through ports)
    pub fn power_balance(&self, x: &[f64], u: &[f64]) -> IntegrateResult<(f64, f64)> {
        let r = self.r_matrix(x)?;
        let grad_h = self.grad_hamiltonian(x)?;
        let y = self.output(x)?;

        let dissipation = grad_h.dot(&r.dot(&grad_h));
        let u_arr = Array1::from_vec(u.to_vec());
        let supply_rate = y.dot(&u_arr);

        Ok((dissipation, supply_rate))
    }

    /// Validate that J(x) is skew-symmetric at the given state x.
    pub fn validate_skew_symmetry(&self, x: &[f64]) -> IntegrateResult<bool> {
        let j = self.j_matrix(x)?;
        let jt = j.t().to_owned();
        let sum = &j + &jt;
        let max_err = sum.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        Ok(max_err <= self.config.skew_sym_tol)
    }

    /// Validate that R(x) is positive semi-definite at the given state x.
    ///
    /// Uses the fact that a symmetric matrix is PSD iff all eigenvalues >= 0.
    /// We check this via Gershgorin circle theorem as a necessary condition first.
    pub fn validate_psd(&self, x: &[f64]) -> IntegrateResult<bool> {
        let r = self.r_matrix(x)?;
        let n = r.nrows();
        // Quick symmetry check first
        for i in 0..n {
            for j in 0..n {
                if (r[[i, j]] - r[[j, i]]).abs() > 1e-10 {
                    return Ok(false);
                }
            }
        }
        // Gershgorin: all Gershgorin discs must be in [0, ∞)
        for i in 0..n {
            let diag = r[[i, i]];
            let off_sum: f64 = (0..n).filter(|&j| j != i).map(|j| r[[i, j]].abs()).sum();
            if diag - off_sum < self.config.psd_tol {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl std::fmt::Debug for PortHamiltonianSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PortHamiltonianSystem")
            .field("n_states", &self.n_states)
            .field("n_ports", &self.n_ports)
            .field("config", &self.config)
            .finish()
    }
}

/// Builder for constructing Port-Hamiltonian systems in a composable way.
///
/// # Example
///
/// ```rust,ignore
/// let system = PortHamiltonianBuilder::new(2, 1)
///     .with_j(|_x| Ok(array![[0.0, 1.0], [-1.0, 0.0]]))
///     .with_r(|_x| Ok(Array2::zeros((2, 2))))
///     .with_hamiltonian(|x| Ok(0.5 * x[0] * x[0] + 0.5 * x[1] * x[1]))
///     .with_b(array![[0.0], [1.0]])
///     .build()?;
/// ```
pub struct PortHamiltonianBuilder {
    n_states: usize,
    n_ports: usize,
    j_fn: Option<MatrixFn>,
    r_fn: Option<MatrixFn>,
    hamiltonian: Option<ScalarFn>,
    grad_hamiltonian: Option<VectorFn>,
    b_matrix: Option<Array2<f64>>,
    b_fn: Option<MatrixFn>,
    config: PortHamiltonianConfig,
}

impl PortHamiltonianBuilder {
    /// Create a new builder with given state and port dimensions.
    pub fn new(n_states: usize, n_ports: usize) -> Self {
        Self {
            n_states,
            n_ports,
            j_fn: None,
            r_fn: None,
            hamiltonian: None,
            grad_hamiltonian: None,
            b_matrix: None,
            b_fn: None,
            config: PortHamiltonianConfig::default(),
        }
    }

    /// Set the interconnection matrix function J(x).
    pub fn with_j(
        mut self,
        j: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.j_fn = Some(Box::new(j));
        self
    }

    /// Set the dissipation matrix function R(x).
    pub fn with_r(
        mut self,
        r: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.r_fn = Some(Box::new(r));
        self
    }

    /// Set the Hamiltonian function H(x).
    pub fn with_hamiltonian(
        mut self,
        h: impl Fn(&[f64]) -> IntegrateResult<f64> + Send + Sync + 'static,
    ) -> Self {
        self.hamiltonian = Some(Box::new(h));
        self
    }

    /// Set the analytic gradient of the Hamiltonian ∇H(x).
    pub fn with_grad_hamiltonian(
        mut self,
        gh: impl Fn(&[f64]) -> IntegrateResult<Array1<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.grad_hamiltonian = Some(Box::new(gh));
        self
    }

    /// Set a constant B matrix.
    pub fn with_b(mut self, b: Array2<f64>) -> Self {
        self.b_matrix = Some(b);
        self
    }

    /// Set a state-dependent B matrix function.
    pub fn with_b_fn(
        mut self,
        b: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        self.b_fn = Some(Box::new(b));
        self
    }

    /// Set configuration options.
    pub fn with_config(mut self, config: PortHamiltonianConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the Port-Hamiltonian system.
    pub fn build(self) -> IntegrateResult<PortHamiltonianSystem> {
        let j_fn = self
            .j_fn
            .ok_or_else(|| IntegrateError::ValueError("J matrix function is required".into()))?;
        let r_fn = self
            .r_fn
            .ok_or_else(|| IntegrateError::ValueError("R matrix function is required".into()))?;
        let hamiltonian = self
            .hamiltonian
            .ok_or_else(|| IntegrateError::ValueError("Hamiltonian function is required".into()))?;

        let b_matrix_fn: MatrixFn = if let Some(b_fn) = self.b_fn {
            b_fn
        } else {
            let b = self.b_matrix.ok_or_else(|| {
                IntegrateError::ValueError("B matrix (constant or function) is required".into())
            })?;
            Box::new(move |_x| Ok(b.clone()))
        };

        let mut system = PortHamiltonianSystem {
            n_states: self.n_states,
            n_ports: self.n_ports,
            j_matrix: j_fn,
            r_matrix: r_fn,
            hamiltonian,
            grad_hamiltonian: self.grad_hamiltonian,
            b_matrix: b_matrix_fn,
            config: self.config,
        };

        if let Some(gh) = system.grad_hamiltonian.take() {
            system.grad_hamiltonian = Some(gh);
        }

        Ok(system)
    }
}
