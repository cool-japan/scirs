//! Leapfrog and Störmer-Verlet integrators
//!
//! This module provides second-order symplectic integrators including:
//! - **Störmer-Verlet**: A popular second-order symplectic integrator
//! - **Velocity Verlet**: A variant optimized for Hamiltonians with velocity-dependent terms
//! - **Position Verlet**: An alternative formulation that updates position first
//!
//! These methods are widely used in molecular dynamics simulations and
//! provide excellent long-term energy conservation properties.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::symplectic::{HamiltonianFn, SymplecticIntegrator};
use ndarray::Array1;
use std::marker::PhantomData;

/// Störmer-Verlet Method (also known as Leapfrog)
///
/// A second-order symplectic integrator that provides excellent
/// energy conservation for long-time integration of Hamiltonian systems.
///
/// Algorithm:
/// 1. p_{n+1/2} = p_n + (dt/2) * dp/dt(t_n, q_n, p_n)
/// 2. q_{n+1} = q_n + dt * dq/dt(t_n+dt/2, q_n, p_{n+1/2})
/// 3. p_{n+1} = p_{n+1/2} + (dt/2) * dp/dt(t_n+dt, q_{n+1}, p_{n+1/2})
#[derive(Debug, Clone)]
pub struct StormerVerlet<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> StormerVerlet<F> {
    /// Create a new Störmer-Verlet integrator
    pub fn new() -> Self {
        StormerVerlet {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for StormerVerlet<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticIntegrator<F> for StormerVerlet<F> {
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        // Half-step for momentum
        let dp1 = system.dp_dt(t, q, p)?;
        let p_half = p + &(&dp1 * (dt / (F::one() + F::one())));

        // Full step for position
        let t_half = t + dt / (F::one() + F::one());
        let dq = system.dq_dt(t_half, q, &p_half)?;
        let q_new = q + &(&dq * dt);

        // Half-step for momentum
        let t_new = t + dt;
        let dp2 = system.dp_dt(t_new, &q_new, &p_half)?;
        let p_new = p_half + &(&dp2 * (dt / (F::one() + F::one())));

        Ok((q_new, p_new))
    }
}

/// Velocity Verlet Method
///
/// A variant of the Störmer-Verlet method that is more numerically stable
/// for certain types of problems, particularly molecular dynamics.
///
/// Algorithm:
/// 1. q_{n+1} = q_n + dt * dq/dt(t_n, q_n, p_n) + (dt^2/2) * d²q/dt²(t_n, q_n, p_n)
/// 2. p_{n+1} = p_n + (dt/2) * [dp/dt(t_n, q_n, p_n) + dp/dt(t_n+dt, q_{n+1}, p_n)]
///
/// For separable Hamiltonians, this simplifies to:
/// 1. q_{n+1} = q_n + dt * p_n/m
/// 2. p_{n+1} = p_n - dt * ∇V(q_{n+1})
pub fn velocity_verlet<F: IntegrateFloat>(
    system: &dyn HamiltonianFn<F>,
    t: F,
    q: &Array1<F>,
    p: &Array1<F>,
    dt: F,
) -> IntegrateResult<(Array1<F>, Array1<F>)> {
    // Velocity Verlet is mathematically equivalent to leapfrog, just with different ordering
    // For separable Hamiltonians: H = T(p) + V(q)

    // Step 1: Half-step momentum update using force at current position
    let dp_old = system.dp_dt(t, q, p)?;
    let p_half = p + &(&dp_old * (dt / F::from(2.0).unwrap()));

    // Step 2: Full-step position update using the half-step momentum
    let dq = system.dq_dt(t + dt / F::from(2.0).unwrap(), q, &p_half)?;
    let q_new = q + &(&dq * dt);

    // Step 3: Another half-step momentum update using force at new position
    let dp_new = system.dp_dt(t + dt, &q_new, &p_half)?;
    let p_new = &p_half + &(&dp_new * (dt / F::from(2.0).unwrap()));

    Ok((q_new, p_new))
}

/// Position Verlet Method
///
/// An alternative formulation of the Verlet method that updates
/// position first, then computes momentum.
///
/// Algorithm:
/// 1. p_{n+1/2} = p_n + (dt/2) * dp/dt(t_n, q_n, p_n)
/// 2. q_{n+1} = q_n + dt * dq/dt(t_n+dt/2, q_n, p_{n+1/2})
/// 3. p_{n+1} = p_{n+1/2} + (dt/2) * dp/dt(t_n+dt, q_{n+1}, p_{n+1/2})
pub fn position_verlet<F: IntegrateFloat>(
    system: &dyn HamiltonianFn<F>,
    t: F,
    q: &Array1<F>,
    p: &Array1<F>,
    dt: F,
) -> IntegrateResult<(Array1<F>, Array1<F>)> {
    // This implementation is identical to StormerVerlet
    StormerVerlet::new().step(system, t, q, p, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symplectic::potential::SeparableHamiltonian;
    use ndarray::array;
    use std::f64::consts::PI;

    /// Test with simple harmonic oscillator
    #[test]
    fn test_verlet_harmonic() {
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

        // Integrate for a full period
        let period = 2.0 * PI;
        let steps = (period / dt).round() as usize;

        let integrator = StormerVerlet::new();
        let mut q = q0.clone();
        let mut p = p0.clone();
        let mut t = t0;

        for _ in 0..steps {
            let (q_new, p_new) = integrator.step(&system, t, &q, &p, dt).unwrap();
            q = q_new;
            p = p_new;
            t += dt;
        }

        // After one period, should be close to initial state
        assert!((q[0] - q0[0]).abs() < 0.1);
        assert!((p[0] - p0[0]).abs() < 0.1);
    }

    #[test]
    fn test_compare_velocity_verlet() {
        // Simple harmonic oscillator
        let system = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 { 0.5 * q.dot(q) },
        );

        // Initial state
        let q0 = array![1.0];
        let p0 = array![0.0];
        let t0 = 0.0;
        let dt = 0.01; // Smaller step size for better accuracy

        // Compute initial energy
        let initial_energy = 0.5 * p0.dot(&p0) + 0.5 * q0.dot(&q0);

        // Integrate for one period
        let period = 2.0 * PI;
        let steps = (period / dt).round() as usize;

        // Test StörmerVerlet
        let mut q1 = q0.clone();
        let mut p1 = p0.clone();
        for _ in 0..steps {
            let (q_new, p_new) = StormerVerlet::new()
                .step(&system, t0, &q1, &p1, dt)
                .unwrap();
            q1 = q_new;
            p1 = p_new;
        }
        let energy1 = 0.5 * p1.dot(&p1) + 0.5 * q1.dot(&q1);

        // Test velocity_verlet
        let mut q2 = q0.clone();
        let mut p2 = p0.clone();
        for _ in 0..steps {
            let (q_new, p_new) = velocity_verlet(&system, t0, &q2, &p2, dt).unwrap();
            q2 = q_new;
            p2 = p_new;
        }
        let energy2 = 0.5 * p2.dot(&p2) + 0.5 * q2.dot(&q2);

        // Both methods should conserve energy
        assert!((energy1 - initial_energy).abs() < 1e-6);
        assert!((energy2 - initial_energy).abs() < 1e-6);

        // And should return to approximately the same state after one period
        assert!((q1[0] - q0[0]).abs() < 0.01);
        assert!((q2[0] - q0[0]).abs() < 0.01);
    }

    #[test]
    fn test_energy_conservation() {
        // Kepler problem (2D planetary orbit)
        // H = |p|^2/2 - 1/|q|
        let kepler = SeparableHamiltonian::new(
            // Kinetic energy
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            // Gravitational potential
            |_t, q| -> f64 {
                let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
                if r < 1e-10 {
                    0.0
                } else {
                    -1.0 / r
                }
            },
        );

        // Initial condition for circular orbit (r=1)
        let q0 = array![1.0, 0.0]; // Starting at (1,0)
        let p0 = array![0.0, 1.0]; // Initial velocity perpendicular to radius

        // Integration parameters
        let t0 = 0.0;
        let tf = 10.0; // Several orbital periods
        let dt = 0.01;

        // Integrate using StormerVerlet
        let integrator = StormerVerlet::new();
        let result = integrator
            .integrate(&kepler, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();

        // Check energy conservation
        if let Some(error) = result.energy_relative_error {
            assert!(error < 1e-3, "Energy error too large: {}", error);
        }

        // For circular orbit, distance from origin should be approximately constant
        for i in 0..result.q.len() {
            let q = &result.q[i];
            let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
            assert!((r - 1.0).abs() < 0.01, "Orbit not circular, r = {}", r);
        }
    }
}
