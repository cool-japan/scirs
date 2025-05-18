//! Symplectic Runge-Kutta methods
//!
//! This module provides symplectic Runge-Kutta integrators for Hamiltonian systems.
//! These methods are constructed to preserve the symplectic structure and
//! offer high-order accuracy for smooth Hamiltonian systems.
//!
//! The implementations focus on Gauss-Legendre based methods which are known
//! to provide optimal order for a given number of stages.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::symplectic::{HamiltonianFn, SymplecticIntegrator};
use ndarray::{Array1, Array2};
use std::marker::PhantomData;

/// Gauss-Legendre 4th order symplectic Runge-Kutta method
///
/// This is a 2-stage method with 4th order accuracy, based on
/// Gauss-Legendre quadrature points.
#[derive(Debug, Clone)]
pub struct GaussLegendre4<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> GaussLegendre4<F> {
    /// Create a new Gauss-Legendre 4th order integrator
    pub fn new() -> Self {
        GaussLegendre4 {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for GaussLegendre4<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticIntegrator<F> for GaussLegendre4<F> {
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let two = F::one() + F::one();
        let half = F::one() / two;
        let quarter = half / two;
        let sqrt_3 = F::from_f64(3.0_f64.sqrt()).unwrap();
        let sixth = F::one() / (F::from_f64(6.0).unwrap());

        // Butcher tableau coefficients for 2-stage Gauss-Legendre method
        // c = [1/2 - sqrt(3)/6, 1/2 + sqrt(3)/6]
        let c = [half - sqrt_3 * sixth, half + sqrt_3 * sixth];

        // Coefficient matrix A
        // A = [1/4, 1/4 - sqrt(3)/6;
        //      1/4 + sqrt(3)/6, 1/4]
        let a11 = quarter;
        let a12 = quarter - sqrt_3 * sixth;
        let a21 = quarter + sqrt_3 * sixth;
        let a22 = quarter;

        // Create arrays for stage values
        let n = q.len();
        let zero = Array1::<F>::zeros(n);

        // Initial guess for stage derivatives
        let mut k_q = [zero.clone(), zero.clone()];
        let mut k_p = [zero.clone(), zero.clone()];

        // Time points for each stage
        let t1 = t + c[0] * dt;
        let t2 = t + c[1] * dt;

        // Simplified Newton iteration for implicit system
        for _ in 0..10 {
            // Fixed number of iterations - can be made adaptive
            // Calculate stage positions and momenta
            let q1 = q + &(&k_q[0] * (a11 * dt) + &k_q[1] * (a12 * dt));
            let p1 = p + &(&k_p[0] * (a11 * dt) + &k_p[1] * (a12 * dt));

            let q2 = q + &(&k_q[0] * (a21 * dt) + &k_q[1] * (a22 * dt));
            let p2 = p + &(&k_p[0] * (a21 * dt) + &k_p[1] * (a22 * dt));

            // Evaluate derivatives at stage points
            let dq1 = system.dq_dt(t1, &q1, &p1)?;
            let dp1 = system.dp_dt(t1, &q1, &p1)?;

            let dq2 = system.dq_dt(t2, &q2, &p2)?;
            let dp2 = system.dp_dt(t2, &q2, &p2)?;

            // Check for convergence
            let err1 = (&dq1 - &k_q[0])
                .iter()
                .map(|&x| x.abs())
                .fold(F::zero(), |a, b| a.max(b));
            let err2 = (&dq2 - &k_q[1])
                .iter()
                .map(|&x| x.abs())
                .fold(F::zero(), |a, b| a.max(b));
            let err3 = (&dp1 - &k_p[0])
                .iter()
                .map(|&x| x.abs())
                .fold(F::zero(), |a, b| a.max(b));
            let err4 = (&dp2 - &k_p[1])
                .iter()
                .map(|&x| x.abs())
                .fold(F::zero(), |a, b| a.max(b));

            let max_err = err1.max(err2).max(err3).max(err4);
            if max_err < F::from_f64(1e-12).unwrap() {
                break;
            }

            // Update stage derivatives
            k_q[0] = dq1;
            k_p[0] = dp1;
            k_q[1] = dq2;
            k_p[1] = dp2;
        }

        // Compute final update
        // b = [1/2, 1/2]
        let q_new = q + &(&k_q[0] * (half * dt) + &k_q[1] * (half * dt));
        let p_new = p + &(&k_p[0] * (half * dt) + &k_p[1] * (half * dt));

        Ok((q_new, p_new))
    }
}

/// Gauss-Legendre 6th order symplectic Runge-Kutta method
///
/// This is a 3-stage method with 6th order accuracy, based on
/// Gauss-Legendre quadrature points.
#[derive(Debug, Clone)]
pub struct GaussLegendre6<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> GaussLegendre6<F> {
    /// Create a new Gauss-Legendre 6th order integrator
    pub fn new() -> Self {
        GaussLegendre6 {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for GaussLegendre6<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticIntegrator<F> for GaussLegendre6<F> {
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let two = F::one() + F::one();
        let _half = F::one() / two;

        // Butcher tableau coefficients for 3-stage Gauss-Legendre method

        // Node points c = [1/2 - sqrt(15)/10, 1/2, 1/2 + sqrt(15)/10]
        let c1 = F::from_f64(0.5 - 0.1 * 15.0_f64.sqrt()).unwrap();
        let c2 = F::from_f64(0.5).unwrap();
        let c3 = F::from_f64(0.5 + 0.1 * 15.0_f64.sqrt()).unwrap();
        let c = [c1, c2, c3];

        // Coefficient matrix A (3x3)
        let a = Array2::<F>::from_shape_vec(
            (3, 3),
            vec![
                F::from_f64(5.0 / 36.0).unwrap(),
                F::from_f64(2.0 / 9.0 - 1.0 / 15.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(5.0 / 36.0 - 1.0 / 30.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(5.0 / 36.0 + 1.0 / 24.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(2.0 / 9.0).unwrap(),
                F::from_f64(5.0 / 36.0 - 1.0 / 24.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(5.0 / 36.0 + 1.0 / 30.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(2.0 / 9.0 + 1.0 / 15.0 * 15.0_f64.sqrt()).unwrap(),
                F::from_f64(5.0 / 36.0).unwrap(),
            ],
        )
        .unwrap();

        // Weights b = [5/18, 4/9, 5/18]
        let b1 = F::from_f64(5.0 / 18.0).unwrap();
        let b2 = F::from_f64(4.0 / 9.0).unwrap();
        let b3 = F::from_f64(5.0 / 18.0).unwrap();
        let b = [b1, b2, b3];

        // Create arrays for stage values
        let n = q.len();
        let zero = Array1::<F>::zeros(n);

        // Initial guess for stage derivatives
        let mut k_q = [zero.clone(), zero.clone(), zero.clone()];
        let mut k_p = [zero.clone(), zero.clone(), zero.clone()];

        // Time points for each stage
        let t1 = t + c[0] * dt;
        let t2 = t + c[1] * dt;
        let t3 = t + c[2] * dt;

        // Simplified Newton iteration for implicit system
        for _ in 0..15 {
            // More iterations for this higher-order method
            // Calculate stage positions and momenta
            let q1 = q + &(&k_q[0] * (a[[0, 0]] * dt)
                + &k_q[1] * (a[[0, 1]] * dt)
                + &k_q[2] * (a[[0, 2]] * dt));
            let p1 = p + &(&k_p[0] * (a[[0, 0]] * dt)
                + &k_p[1] * (a[[0, 1]] * dt)
                + &k_p[2] * (a[[0, 2]] * dt));

            let q2 = q + &(&k_q[0] * (a[[1, 0]] * dt)
                + &k_q[1] * (a[[1, 1]] * dt)
                + &k_q[2] * (a[[1, 2]] * dt));
            let p2 = p + &(&k_p[0] * (a[[1, 0]] * dt)
                + &k_p[1] * (a[[1, 1]] * dt)
                + &k_p[2] * (a[[1, 2]] * dt));

            let q3 = q + &(&k_q[0] * (a[[2, 0]] * dt)
                + &k_q[1] * (a[[2, 1]] * dt)
                + &k_q[2] * (a[[2, 2]] * dt));
            let p3 = p + &(&k_p[0] * (a[[2, 0]] * dt)
                + &k_p[1] * (a[[2, 1]] * dt)
                + &k_p[2] * (a[[2, 2]] * dt));

            // Evaluate derivatives at stage points
            let dq1 = system.dq_dt(t1, &q1, &p1)?;
            let dp1 = system.dp_dt(t1, &q1, &p1)?;

            let dq2 = system.dq_dt(t2, &q2, &p2)?;
            let dp2 = system.dp_dt(t2, &q2, &p2)?;

            let dq3 = system.dq_dt(t3, &q3, &p3)?;
            let dp3 = system.dp_dt(t3, &q3, &p3)?;

            // Check for convergence
            let err_max = [
                &dq1 - &k_q[0],
                &dq2 - &k_q[1],
                &dq3 - &k_q[2],
                &dp1 - &k_p[0],
                &dp2 - &k_p[1],
                &dp3 - &k_p[2],
            ]
            .iter()
            .flat_map(|arr| arr.iter().map(|&x| x.abs()))
            .fold(F::zero(), |a, b| a.max(b));

            if err_max < F::from_f64(1e-12).unwrap() {
                break;
            }

            // Update stage derivatives
            k_q[0] = dq1;
            k_p[0] = dp1;
            k_q[1] = dq2;
            k_p[1] = dp2;
            k_q[2] = dq3;
            k_p[2] = dp3;
        }

        // Compute final update using the weights
        let q_new = q + &(&k_q[0] * (b[0] * dt) + &k_q[1] * (b[1] * dt) + &k_q[2] * (b[2] * dt));
        let p_new = p + &(&k_p[0] * (b[0] * dt) + &k_p[1] * (b[1] * dt) + &k_p[2] * (b[2] * dt));

        Ok((q_new, p_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symplectic::leapfrog::StormerVerlet;
    use crate::symplectic::potential::SeparableHamiltonian;
    use ndarray::array;
    use std::f64::consts::PI;

    /// Test accuracy of high-order methods
    #[test]
    fn test_accuracy_comparison() {
        // Simple harmonic oscillator
        let system = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 { 0.5 * q.dot(q) },
        );

        // Initial state
        let q0 = array![1.0];
        let p0 = array![0.0];
        let t0 = 0.0;
        let tf = 2.0 * PI; // One full period

        // Step size
        let dt = 0.1;

        // Create methods to compare
        let verlet = StormerVerlet::new();
        let gl4 = GaussLegendre4::new();
        let gl6 = GaussLegendre6::new();

        // Integrate with all methods
        let verlet_result = verlet
            .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();
        let gl4_result = gl4
            .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();
        let gl6_result = gl6
            .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();

        // Calculate errors at the end of one period
        // Exact solution: q(2π) = 1.0, p(2π) = 0.0
        let verlet_error = ((verlet_result.q.last().unwrap()[0] - 1.0).powi(2)
            + verlet_result.p.last().unwrap()[0].powi(2))
        .sqrt();

        let gl4_error = ((gl4_result.q.last().unwrap()[0] - 1.0).powi(2)
            + gl4_result.p.last().unwrap()[0].powi(2))
        .sqrt();

        let gl6_error = ((gl6_result.q.last().unwrap()[0] - 1.0).powi(2)
            + gl6_result.p.last().unwrap()[0].powi(2))
        .sqrt();

        // Higher-order methods should be more accurate
        assert!(
            gl4_error < verlet_error,
            "GL4 error ({}) should be smaller than Verlet error ({})",
            gl4_error,
            verlet_error
        );

        assert!(
            gl6_error < gl4_error,
            "GL6 error ({}) should be smaller than GL4 error ({})",
            gl6_error,
            gl4_error
        );
    }

    /// Test energy conservation for long-time integration
    #[test]
    fn test_energy_preservation() {
        // Simple pendulum: H = p^2/2 - cos(q)
        let pendulum = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 { -q[0].cos() },
        );

        // Initial state - fairly large amplitude
        let q0 = array![0.0];
        let p0 = array![1.5]; // Enough energy for pendulum to swing significantly

        // Long integration time
        let t0 = 0.0;
        let tf = 50.0;
        let dt = 0.1;

        // Compare methods
        let verlet = StormerVerlet::new();
        let gl4 = GaussLegendre4::new();

        // Integrate
        let verlet_result = verlet
            .integrate(&pendulum, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();
        let gl4_result = gl4
            .integrate(&pendulum, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();

        // Check energy conservation
        if let (Some(verlet_error), Some(gl4_error)) = (
            verlet_result.energy_relative_error,
            gl4_result.energy_relative_error,
        ) {
            // Higher-order method should have better energy conservation
            assert!(
                gl4_error < verlet_error,
                "GL4 energy error ({}) should be smaller than Verlet error ({})",
                gl4_error,
                verlet_error
            );
        }
    }
}
