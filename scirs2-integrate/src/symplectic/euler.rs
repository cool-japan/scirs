//! Symplectic Euler methods
//!
//! This module provides first-order symplectic integrators known as
//! Symplectic Euler methods. These are the simplest symplectic integrators
//! and come in two variants:
//!
//! - **Symplectic Euler A (position first)**: Updates position first, then momentum
//! - **Symplectic Euler B (momentum first)**: Updates momentum first, then position
//!
//! While first-order, these methods exactly conserve the symplectic structure
//! and provide bounded energy error over long integration times.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::symplectic::{HamiltonianFn, SymplecticIntegrator};
use ndarray::Array1;
use std::marker::PhantomData;

/// Symplectic Euler Method A - updates position first, then momentum
///
/// Algorithm:
/// 1. q_{n+1} = q_n + dt * dq/dt(t_n, q_n, p_n)
/// 2. p_{n+1} = p_n + dt * dp/dt(t_n, q_{n+1}, p_n)
#[derive(Debug, Clone)]
pub struct SymplecticEulerA<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> SymplecticEulerA<F> {
    /// Create a new Symplectic Euler A integrator
    pub fn new() -> Self {
        SymplecticEulerA {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for SymplecticEulerA<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticIntegrator<F> for SymplecticEulerA<F> {
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        // Step 1: Update position
        let dq = system.dq_dt(t, q, p)?;
        let q_new = q + &(&dq * dt);

        // Step 2: Update momentum using updated position
        let dp = system.dp_dt(t, &q_new, p)?;
        let p_new = p + &(&dp * dt);

        Ok((q_new, p_new))
    }
}

/// Symplectic Euler Method B - updates momentum first, then position
///
/// Algorithm:
/// 1. p_{n+1} = p_n + dt * dp/dt(t_n, q_n, p_n)
/// 2. q_{n+1} = q_n + dt * dq/dt(t_n, q_n, p_{n+1})
#[derive(Debug, Clone)]
pub struct SymplecticEulerB<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> SymplecticEulerB<F> {
    /// Create a new Symplectic Euler B integrator
    pub fn new() -> Self {
        SymplecticEulerB {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for SymplecticEulerB<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticIntegrator<F> for SymplecticEulerB<F> {
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        // Step 1: Update momentum
        let dp = system.dp_dt(t, q, p)?;
        let p_new = p + &(&dp * dt);

        // Step 2: Update position using updated momentum
        let dq = system.dq_dt(t, q, &p_new)?;
        let q_new = q + &(&dq * dt);

        Ok((q_new, p_new))
    }
}

/// Convenience function for the default Symplectic Euler method (variant A)
///
/// Provides a simple functional interface to the Symplectic Euler A algorithm,
/// which updates position first, then momentum.
pub fn symplectic_euler<F: IntegrateFloat>(
    system: &dyn HamiltonianFn<F>,
    t: F,
    q: &Array1<F>,
    p: &Array1<F>,
    dt: F,
) -> IntegrateResult<(Array1<F>, Array1<F>)> {
    SymplecticEulerA::new().step(system, t, q, p, dt)
}

/// Convenience function for the Symplectic Euler A method (position first)
pub fn symplectic_euler_a<F: IntegrateFloat>(
    system: &dyn HamiltonianFn<F>,
    t: F,
    q: &Array1<F>,
    p: &Array1<F>,
    dt: F,
) -> IntegrateResult<(Array1<F>, Array1<F>)> {
    SymplecticEulerA::new().step(system, t, q, p, dt)
}

/// Convenience function for the Symplectic Euler B method (momentum first)
pub fn symplectic_euler_b<F: IntegrateFloat>(
    system: &dyn HamiltonianFn<F>,
    t: F,
    q: &Array1<F>,
    p: &Array1<F>,
    dt: F,
) -> IntegrateResult<(Array1<F>, Array1<F>)> {
    SymplecticEulerB::new().step(system, t, q, p, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symplectic::potential::SeparableHamiltonian;
    use ndarray::array;

    /// Test with simple harmonic oscillator
    #[test]
    fn test_symplectic_euler() {
        // Simple harmonic oscillator: H(q, p) = p^2/2 + q^2/2
        let system = SeparableHamiltonian::new(
            // T(p) = p^2/2
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            // V(q) = q^2/2
            |_t, q| -> f64 { 0.5 * q.dot(q) },
        );

        // Initial state
        let q0 = array![1.0];
        let p0 = array![0.0];
        let t0 = 0.0;
        let dt = 0.1;

        // Single step with Symplectic Euler A
        let (q1_a, p1_a) = symplectic_euler_a(&system, t0, &q0, &p0, dt).unwrap();

        // Single step with Symplectic Euler B
        let (q1_b, p1_b) = symplectic_euler_b(&system, t0, &q0, &p0, dt).unwrap();

        // For this particular initial condition and harmonic oscillator:
        // A: q1 = 1.0, p1 = -0.1
        // B: q1 = 0.0, p1 = -1.0
        // Different values for A vs B is expected - they are different methods!
        assert!((q1_a[0] - 1.0).abs() < 1e-12);
        assert!((p1_a[0] + 0.1).abs() < 1e-12);

        // Correct expected values for Symplectic Euler B:
        // p_new = 0 + 0.1 * (-1) = -0.1
        // q_new = 1 + 0.1 * (-0.1) = 0.99
        assert!((q1_b[0] - 0.99).abs() < 1e-12);
        assert!((p1_b[0] + 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_energy_conservation() {
        // Simple harmonic oscillator
        let system = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 { 0.5 * q.dot(q) },
        );

        // Initial state with energy = 0.5
        let q0 = array![1.0];
        let p0 = array![0.0];
        let t0 = 0.0;
        let tf = 10.0;
        let dt = 0.1;

        // Integrate for multiple steps
        let integrator = SymplecticEulerA::new();
        let result = integrator.integrate(&system, t0, tf, dt, q0, p0).unwrap();

        // Energy should be conserved to within a small error
        // First-order methods will have linear drift, but it should still be small
        if let Some(error) = result.energy_relative_error {
            assert!(error < 0.1, "Energy error too large: {}", error);
        }

        // Phase space trajectory should follow an ellipse for harmonic oscillator
        // Check a few points to ensure they're close to the exact solution
        // For harmonic oscillator, points should approximately satisfy q^2 + p^2 = 1
        for i in 0..result.t.len() {
            let q = &result.q[i];
            let p = &result.p[i];
            let radius_squared = q[0] * q[0] + p[0] * p[0];
            assert!(
                (radius_squared - 1.0).abs() < 0.1,
                "Point ({}, {}) is too far from unit circle, r² = {}",
                q[0],
                p[0],
                radius_squared
            );
        }
    }
}
