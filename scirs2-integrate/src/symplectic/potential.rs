//! Specialized symplectic integrators for separable Hamiltonian systems
//!
//! This module provides implementations for Hamiltonian systems with
//! a separable structure H(q, p) = T(p) + V(q), which is common in
//! many physical systems. The separation into kinetic energy T(p) and
//! potential energy V(q) allows for specialized implementations.
//!
//! It also provides a generic Hamiltonian system implementation for
//! non-separable Hamiltonians.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::symplectic::HamiltonianFn;
use ndarray::Array1;
use std::fmt::{Debug, Formatter};

/// Type alias for energy function
type EnergyFunction<F> = Box<dyn Fn(F, &Array1<F>) -> F + Send + Sync>;

/// Type alias for gradient function  
type GradientFunction<F> = Box<dyn Fn(F, &Array1<F>) -> Array1<F> + Send + Sync>;

/// Type alias for Hamiltonian equations of motion
type EquationOfMotion<F> = Box<dyn Fn(F, &Array1<F>, &Array1<F>) -> Array1<F> + Send + Sync>;

/// Type alias for Hamiltonian function
type HamiltonianFunction<F> = Box<dyn Fn(F, &Array1<F>, &Array1<F>) -> F + Send + Sync>;

/// A separable Hamiltonian system with H(q, p) = T(p) + V(q)
///
/// This represents systems where the Hamiltonian can be split into
/// kinetic energy T(p) depending only on momenta and potential
/// energy V(q) depending only on positions.
///
/// Examples include:
/// - Simple harmonic oscillator: H = p²/2 + q²/2
/// - Pendulum: H = p²/2 - cos(q)
/// - Kepler problem: H = |p|²/2 - 1/|q|
pub struct SeparableHamiltonian<F: IntegrateFloat> {
    /// Kinetic energy function T(p)
    kinetic_energy: EnergyFunction<F>,

    /// Potential energy function V(q)
    potential_energy: EnergyFunction<F>,

    /// Gradient of potential energy ∇V(q)
    potential_gradient: Option<GradientFunction<F>>,

    /// Gradient of kinetic energy ∇T(p)
    kinetic_gradient: Option<GradientFunction<F>>,
}

impl<F: IntegrateFloat> Debug for SeparableHamiltonian<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeparableHamiltonian")
            .field("kinetic_energy", &"<dyn Fn>")
            .field("potential_energy", &"<dyn Fn>")
            .field("potential_gradient", &"<dyn Fn>")
            .field("kinetic_gradient", &"<dyn Fn>")
            .finish()
    }
}

impl<F: IntegrateFloat> SeparableHamiltonian<F> {
    /// Create a new separable Hamiltonian system
    ///
    /// # Arguments
    ///
    /// * `kinetic_energy` - Function computing T(p)
    /// * `potential_energy` - Function computing V(q)
    ///
    /// # Returns
    ///
    /// A new separable Hamiltonian system
    pub fn new<K, V>(kinetic_energy: K, potential_energy: V) -> Self
    where
        K: Fn(F, &Array1<F>) -> F + 'static + Send + Sync,
        V: Fn(F, &Array1<F>) -> F + 'static + Send + Sync,
    {
        SeparableHamiltonian {
            kinetic_energy: Box::new(kinetic_energy),
            potential_energy: Box::new(potential_energy),
            potential_gradient: None,
            kinetic_gradient: None,
        }
    }

    /// Add analytical gradient functions for improved performance
    ///
    /// # Arguments
    ///
    /// * `kinetic_gradient` - Function computing ∇T(p)
    /// * `potential_gradient` - Function computing ∇V(q)
    ///
    /// # Returns
    ///
    /// Self with gradients configured
    pub fn with_gradients<KG, VG>(mut self, kinetic_gradient: KG, potential_gradient: VG) -> Self
    where
        KG: Fn(F, &Array1<F>) -> Array1<F> + 'static + Send + Sync,
        VG: Fn(F, &Array1<F>) -> Array1<F> + 'static + Send + Sync,
    {
        self.kinetic_gradient = Some(Box::new(kinetic_gradient));
        self.potential_gradient = Some(Box::new(potential_gradient));
        self
    }

    /// Create a simple harmonic oscillator system
    ///
    /// H(q, p) = p²/2 + q²/2
    ///
    /// # Returns
    ///
    /// A harmonic oscillator Hamiltonian
    pub fn harmonic_oscillator() -> Self {
        let kinetic = |_t: F, p: &Array1<F>| -> F {
            p.iter().map(|&pi| pi * pi).sum::<F>() * F::from_f64(0.5).unwrap()
        };

        let potential = |_t: F, q: &Array1<F>| -> F {
            q.iter().map(|&qi| qi * qi).sum::<F>() * F::from_f64(0.5).unwrap()
        };

        let kinetic_grad = |_t: F, p: &Array1<F>| -> Array1<F> { p.to_owned() };

        let potential_grad = |_t: F, q: &Array1<F>| -> Array1<F> { q.to_owned() };

        SeparableHamiltonian::new(kinetic, potential).with_gradients(kinetic_grad, potential_grad)
    }

    /// Create a pendulum system
    ///
    /// H(q, p) = p²/2 - cos(q)
    ///
    /// # Returns
    ///
    /// A pendulum Hamiltonian
    pub fn pendulum() -> Self {
        let kinetic = |_t: F, p: &Array1<F>| -> F { F::from_f64(0.5).unwrap() * p[0] * p[0] };

        let potential = |_t: F, q: &Array1<F>| -> F { -q[0].cos() };

        let kinetic_grad = |_t: F, p: &Array1<F>| -> Array1<F> { Array1::from_vec(vec![p[0]]) };

        let potential_grad =
            |_t: F, q: &Array1<F>| -> Array1<F> { Array1::from_vec(vec![q[0].sin()]) };

        SeparableHamiltonian::new(kinetic, potential).with_gradients(kinetic_grad, potential_grad)
    }

    /// Create a 2D Kepler problem (planetary orbit)
    ///
    /// H(q, p) = |p|²/2 - 1/|q|
    ///
    /// # Returns
    ///
    /// A Kepler problem Hamiltonian
    pub fn kepler_problem() -> Self {
        let kinetic =
            |_t: F, p: &Array1<F>| -> F { F::from_f64(0.5).unwrap() * (p[0] * p[0] + p[1] * p[1]) };

        let potential = |_t: F, q: &Array1<F>| -> F {
            let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
            if r < F::from_f64(1e-10).unwrap() {
                F::zero()
            } else {
                -F::one() / r
            }
        };

        let kinetic_grad = |_t: F, p: &Array1<F>| -> Array1<F> { p.to_owned() };

        let potential_grad = |_t: F, q: &Array1<F>| -> Array1<F> {
            let r2 = q[0] * q[0] + q[1] * q[1];
            let r = r2.sqrt();

            if r < F::from_f64(1e-10).unwrap() {
                Array1::zeros(q.len())
            } else {
                let r3 = r * r2;
                let factor = F::one() / r3;
                Array1::from_vec(vec![q[0] * factor, q[1] * factor])
            }
        };

        SeparableHamiltonian::new(kinetic, potential).with_gradients(kinetic_grad, potential_grad)
    }
}

impl<F: IntegrateFloat> HamiltonianFn<F> for SeparableHamiltonian<F> {
    fn dq_dt(&self, t: F, _q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        // For separable Hamiltonian: dq/dt = ∂H/∂p = ∂T/∂p
        if let Some(grad) = &self.kinetic_gradient {
            Ok(grad(t, p))
        } else {
            // Numerical approximation using finite differences
            let h = F::from_f64(1e-6).unwrap();
            let mut dq = Array1::zeros(p.len());

            for i in 0..p.len() {
                let mut p_plus = p.to_owned();
                p_plus[i] += h;

                let mut p_minus = p.to_owned();
                p_minus[i] -= h;

                let t_plus = (self.kinetic_energy)(t, &p_plus);
                let t_minus = (self.kinetic_energy)(t, &p_minus);

                dq[i] = (t_plus - t_minus) / (F::from_f64(2.0).unwrap() * h);
            }

            Ok(dq)
        }
    }

    fn dp_dt(&self, t: F, q: &Array1<F>, _p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        // For separable Hamiltonian: dp/dt = -∂H/∂q = -∂V/∂q
        if let Some(grad) = &self.potential_gradient {
            // Negate the gradient since dp/dt = -∇V(q)
            let dp = -grad(t, q);
            Ok(dp)
        } else {
            // Numerical approximation using finite differences
            let h = F::from_f64(1e-6).unwrap();
            let mut dp = Array1::zeros(q.len());

            for i in 0..q.len() {
                let mut q_plus = q.to_owned();
                q_plus[i] += h;

                let mut q_minus = q.to_owned();
                q_minus[i] -= h;

                let v_plus = (self.potential_energy)(t, &q_plus);
                let v_minus = (self.potential_energy)(t, &q_minus);

                // Negative gradient
                dp[i] = -(v_plus - v_minus) / (F::from_f64(2.0).unwrap() * h);
            }

            Ok(dp)
        }
    }

    fn hamiltonian(
        &self,
    ) -> Option<Box<dyn Fn(F, &Array1<F>, &Array1<F>) -> IntegrateResult<F> + '_>> {
        let kinetic = &self.kinetic_energy;
        let potential = &self.potential_energy;

        Some(Box::new(move |t, q, p| {
            let t_val = kinetic(t, p);
            let v_val = potential(t, q);
            Ok(t_val + v_val)
        }))
    }
}

/// A general non-separable Hamiltonian system
///
/// This represents Hamiltonian systems where the equations of motion are
/// specified directly without assuming a separable structure.
pub struct HamiltonianSystem<F: IntegrateFloat> {
    /// Function computing dq/dt = ∂H/∂p
    dq_dt_fn: EquationOfMotion<F>,

    /// Function computing dp/dt = -∂H/∂q
    dp_dt_fn: EquationOfMotion<F>,

    /// Optional function computing the Hamiltonian H(q, p)
    hamiltonian_fn: Option<HamiltonianFunction<F>>,
}

impl<F: IntegrateFloat> HamiltonianSystem<F> {
    /// Create a new general Hamiltonian system
    ///
    /// # Arguments
    ///
    /// * `dq_dt_fn` - Function computing dq/dt = ∂H/∂p
    /// * `dp_dt_fn` - Function computing dp/dt = -∂H/∂q
    ///
    /// # Returns
    ///
    /// A new Hamiltonian system
    pub fn new<DQ, DP>(dq_dt_fn: DQ, dp_dt_fn: DP) -> Self
    where
        DQ: Fn(F, &Array1<F>, &Array1<F>) -> Array1<F> + 'static + Send + Sync,
        DP: Fn(F, &Array1<F>, &Array1<F>) -> Array1<F> + 'static + Send + Sync,
    {
        HamiltonianSystem {
            dq_dt_fn: Box::new(dq_dt_fn),
            dp_dt_fn: Box::new(dp_dt_fn),
            hamiltonian_fn: None,
        }
    }

    /// Add Hamiltonian function for energy tracking
    ///
    /// # Arguments
    ///
    /// * `hamiltonian_fn` - Function computing H(q, p)
    ///
    /// # Returns
    ///
    /// Self with Hamiltonian function configured
    pub fn with_hamiltonian<H>(mut self, hamiltonian_fn: H) -> Self
    where
        H: Fn(F, &Array1<F>, &Array1<F>) -> F + 'static + Send + Sync,
    {
        self.hamiltonian_fn = Some(Box::new(hamiltonian_fn));
        self
    }
}

impl<F: IntegrateFloat> HamiltonianFn<F> for HamiltonianSystem<F> {
    fn dq_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        Ok((self.dq_dt_fn)(t, q, p))
    }

    fn dp_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        Ok((self.dp_dt_fn)(t, q, p))
    }

    fn hamiltonian(
        &self,
    ) -> Option<Box<dyn Fn(F, &Array1<F>, &Array1<F>) -> IntegrateResult<F> + '_>> {
        if let Some(h_fn) = &self.hamiltonian_fn {
            let h = h_fn;
            Some(Box::new(move |t, q, p| Ok(h(t, q, p))))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symplectic::leapfrog::StormerVerlet;
    use crate::symplectic::SymplecticIntegrator;
    use ndarray::array;
    use std::f64::consts::PI;

    /// Test SeparableHamiltonian implementation with harmonic oscillator
    #[test]
    fn test_harmonic_oscillator() {
        // Create a harmonic oscillator
        let system = SeparableHamiltonian::harmonic_oscillator();

        // Initial state
        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];
        let t0 = 0.0_f64;

        // Verify equations of motion
        let dq = system.dq_dt(t0, &q0, &p0).unwrap();
        let dp = system.dp_dt(t0, &q0, &p0).unwrap();

        // For harmonic oscillator:
        // dq/dt = p
        // dp/dt = -q
        assert!((dq[0] - p0[0]).abs() < 1e-10_f64);
        assert!((dp[0] + q0[0]).abs() < 1e-10_f64);

        // Verify Hamiltonian function
        let h_fn = system.hamiltonian();
        if let Some(hamiltonian) = h_fn {
            let energy = hamiltonian(t0, &q0, &p0).unwrap();
            // H = p²/2 + q²/2 = 0 + 0.5 = 0.5
            assert!((energy - 0.5_f64).abs() < 1e-10);
        } else {
            panic!("Hamiltonian function not provided");
        }
    }

    /// Test integration of separable system
    #[test]
    fn test_integrate_pendulum() {
        // Create a pendulum
        let system = SeparableHamiltonian::pendulum();

        // Initial state (small angle)
        let q0 = array![0.1]; // Small angle from vertical
        let p0 = array![0.0]; // Starting from rest
        let t0 = 0.0;

        // For small angles, pendulum is approximately harmonic
        // with period T = 2π
        let period = 2.0 * PI;
        let tf = period;
        let dt = 0.01;

        // Integrate
        let integrator = StormerVerlet::new();
        let result = integrator
            .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();

        // After one period, should be close to initial state
        let q_final = &result.q[result.q.len() - 1];
        let p_final = &result.p[result.p.len() - 1];

        assert!(
            (q_final[0] - q0[0]).abs() < 0.01,
            "q: {} vs {}",
            q_final[0],
            q0[0]
        );
        assert!(
            (p_final[0] - p0[0]).abs() < 0.01,
            "p: {} vs {}",
            p_final[0],
            p0[0]
        );

        // Check energy conservation
        if let Some(err) = result.energy_relative_error {
            assert!(err < 1e-3, "Energy conservation error too large: {}", err);
        } else {
            panic!("Energy conservation error not calculated");
        }
    }

    /// Test generic Hamiltonian system
    #[test]
    fn test_generic_hamiltonian() {
        // Create a generic Hamiltonian for a harmonic oscillator
        let system = HamiltonianSystem::new(
            // dq/dt = ∂H/∂p = p
            |_t, _q, p| p.clone(),
            // dp/dt = -∂H/∂q = -q
            |_t, q, _p| -q.clone(),
        )
        .with_hamiltonian(
            // H(q, p) = p²/2 + q²/2
            |_t, q, p| 0.5 * (q[0] * q[0] + p[0] * p[0]),
        );

        // Initial state
        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];
        let t0 = 0.0_f64;

        // Verify equations of motion
        let dq = system.dq_dt(t0, &q0, &p0).unwrap();
        let dp = system.dp_dt(t0, &q0, &p0).unwrap();

        assert!((dq[0] - p0[0]).abs() < 1e-10_f64);
        assert!((dp[0] + q0[0]).abs() < 1e-10_f64);

        // Verify Hamiltonian function
        let h_fn = system.hamiltonian();
        if let Some(hamiltonian) = h_fn {
            let energy = hamiltonian(t0, &q0, &p0).unwrap();
            // H = p²/2 + q²/2 = 0 + 0.5 = 0.5
            assert!((energy - 0.5_f64).abs() < 1e-10);
        } else {
            panic!("Hamiltonian function not provided");
        }
    }
}
