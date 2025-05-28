//! Symplectic integrators for Hamiltonian systems
//!
//! This module provides specialized integrators for Hamiltonian systems that
//! preserve important geometric properties such as energy conservation (to within
//! bounded error), phase space volume, and symplectic structure.
//!
//! Symplectic integrators are particularly useful for:
//! - Long-time integration of mechanical systems
//! - Molecular dynamics simulations
//! - Orbital mechanics
//! - Any conservative system where energy conservation is critical
//!
//! # Methods
//!
//! Several symplectic integration methods are provided:
//!
//! - **Symplectic Euler**: First-order, simple method (forward and backward variants)
//! - **Leapfrog/Störmer-Verlet**: Second-order, widely used for molecular dynamics
//! - **Symplectic Runge-Kutta**: Higher-order methods with improved accuracy
//! - **Composition methods**: Methods constructed by composing lower-order integrators
//!
//! # Basic Usage
//!
//! ```
//! use scirs2_integrate::symplectic::potential::HamiltonianSystem;
//! use scirs2_integrate::symplectic::leapfrog::StormerVerlet;
//! use scirs2_integrate::symplectic::SymplecticIntegrator;
//! use ndarray::array;
//!
//! // Define a simple harmonic oscillator: H = p²/2 + q²/2
//! let system = HamiltonianSystem::new(
//!     |_t, _q, p| p.clone(),  // dq/dt = p
//!     |_t, q, _p| -q.clone(), // dp/dt = -q
//! );
//!
//! // Initial conditions: (q0, p0) = (1.0, 0.0)
//! let q0 = array![1.0];
//! let p0 = array![0.0];
//! let t = 0.0;
//! let dt = 0.01;
//!
//! // Create integrator
//! let integrator = StormerVerlet::<f64>::new();
//!
//! // Take one step
//! let (q1, p1) = integrator.step(&system, t, &q0, &p0, dt).unwrap();
//!
//! // Energy should be conserved (approximately)
//! let initial_energy = 0.5_f64 * p0.dot(&p0) + 0.5_f64 * q0.dot(&q0);
//! let final_energy = 0.5_f64 * p1.dot(&p1) + 0.5_f64 * q1.dot(&q1);
//! // For a single step with dt=0.1, the energy error is acceptable
//! assert!((initial_energy - final_energy).abs() < 1e-6);
//! ```

// Public sub-modules
pub mod composition;
pub mod euler;
pub mod leapfrog;
pub mod potential;
pub mod runge_kutta;

// Re-exports for convenience
pub use composition::CompositionMethod;
pub use euler::{symplectic_euler, symplectic_euler_a, symplectic_euler_b};
pub use leapfrog::{position_verlet, velocity_verlet, StormerVerlet};
pub use potential::{HamiltonianSystem, SeparableHamiltonian};
pub use runge_kutta::{GaussLegendre4, GaussLegendre6};

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::Array1;

/// Result of symplectic integration containing state history
#[derive(Debug, Clone)]
pub struct SymplecticResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Position coordinates at each time point
    pub q: Vec<Array1<F>>,
    /// Momentum coordinates at each time point
    pub p: Vec<Array1<F>>,
    /// Total integration time
    pub total_time: F,
    /// Number of steps taken
    pub steps: usize,
    /// Function evaluations
    pub n_evaluations: usize,
    /// Relative error in energy conservation (if available)
    pub energy_relative_error: Option<F>,
}

/// Trait for symplectic integrators
pub trait SymplecticIntegrator<F: IntegrateFloat> {
    /// Perform a single integration step
    ///
    /// # Arguments
    ///
    /// * `system` - The Hamiltonian system to integrate
    /// * `t` - Current time
    /// * `q` - Current position coordinates
    /// * `p` - Current momentum coordinates
    /// * `dt` - Step size
    ///
    /// # Returns
    ///
    /// Updated position and momentum coordinates
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>;

    /// Integrate the system over a time interval
    ///
    /// # Arguments
    ///
    /// * `system` - The Hamiltonian system to integrate
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `dt` - Step size
    /// * `q0` - Initial position coordinates
    /// * `p0` - Initial momentum coordinates
    ///
    /// # Returns
    ///
    /// Integration result containing state history
    fn integrate(
        &self,
        system: &dyn HamiltonianFn<F>,
        t0: F,
        tf: F,
        dt: F,
        q0: Array1<F>,
        p0: Array1<F>,
    ) -> IntegrateResult<SymplecticResult<F>> {
        // Validate inputs
        if dt <= F::zero() {
            return Err(IntegrateError::ValueError(
                "Step size must be positive".into(),
            ));
        }

        if q0.len() != p0.len() {
            return Err(IntegrateError::ValueError(
                "Position and momentum vectors must have the same length".into(),
            ));
        }

        // Determine number of steps
        let t_span = tf - t0;
        let n_steps = (t_span / dt).ceil().to_f64().unwrap() as usize;
        let actual_dt = t_span / F::from_usize(n_steps).unwrap();

        // Initialize result containers
        let mut t = Vec::with_capacity(n_steps + 1);
        let mut q = Vec::with_capacity(n_steps + 1);
        let mut p = Vec::with_capacity(n_steps + 1);

        // Add initial state
        t.push(t0);
        q.push(q0.clone());
        p.push(p0.clone());

        // Perform integration
        let mut curr_t = t0;
        let mut curr_q = q0;
        let mut curr_p = p0;
        let mut n_evals = 0;

        for _ in 0..n_steps {
            // Calculate next state
            let (next_q, next_p) = self.step(system, curr_t, &curr_q, &curr_p, actual_dt)?;
            n_evals += 2; // Approximation: most methods use at least 2 function evaluations per step

            // Advance time
            curr_t += actual_dt;

            // Store results
            t.push(curr_t);
            q.push(next_q.clone());
            p.push(next_p.clone());

            // Update current state
            curr_q = next_q;
            curr_p = next_p;
        }

        // Calculate energy conservation error if the system provides a Hamiltonian
        let energy_error = if let Some(hamiltonian) = system.hamiltonian() {
            let initial_energy = hamiltonian(t[0], &q[0], &p[0])?;
            let final_energy = hamiltonian(t[t.len() - 1], &q[q.len() - 1], &p[p.len() - 1])?;

            // Calculate relative error
            if initial_energy.abs() > F::from_f64(1e-10).unwrap() {
                Some((final_energy - initial_energy).abs() / initial_energy.abs())
            } else {
                Some((final_energy - initial_energy).abs())
            }
        } else {
            None
        };

        Ok(SymplecticResult {
            t,
            q,
            p,
            total_time: tf - t0,
            steps: n_steps,
            n_evaluations: n_evals,
            energy_relative_error: energy_error,
        })
    }
}

/// Type alias for Hamiltonian function
pub type HamiltonianFnBox<'a, F> =
    Box<dyn Fn(F, &Array1<F>, &Array1<F>) -> IntegrateResult<F> + 'a>;

/// Trait for Hamiltonian systems
pub trait HamiltonianFn<F: IntegrateFloat> {
    /// Computes the time derivative of position coordinates: dq/dt = ∂H/∂p
    ///
    /// # Arguments
    ///
    /// * `t` - Current time
    /// * `q` - Position coordinates
    /// * `p` - Momentum coordinates
    ///
    /// # Returns
    ///
    /// Time derivative of position coordinates
    fn dq_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>>;

    /// Computes the time derivative of momentum coordinates: dp/dt = -∂H/∂q
    ///
    /// # Arguments
    ///
    /// * `t` - Current time
    /// * `q` - Position coordinates
    /// * `p` - Momentum coordinates
    ///
    /// # Returns
    ///
    /// Time derivative of momentum coordinates
    fn dp_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>>;

    /// Optional: Computes the Hamiltonian function H(q, p)
    ///
    /// This is used for energy conservation checks and diagnostics.
    /// Return None if not implemented.
    ///
    /// # Arguments
    ///
    /// * `t` - Current time
    /// * `q` - Position coordinates
    /// * `p` - Momentum coordinates
    ///
    /// # Returns
    ///
    /// Value of the Hamiltonian (energy)
    fn hamiltonian(&self) -> Option<HamiltonianFnBox<'_, F>> {
        None
    }
}
