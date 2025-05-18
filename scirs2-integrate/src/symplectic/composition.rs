//! Composition methods for symplectic integration
//!
//! This module provides tools for constructing higher-order symplectic
//! integrators by composing lower-order methods with carefully chosen
//! coefficients. This technique allows creating methods of arbitrary order.
//!
//! The composition follows the principle that if Φₕ is a symplectic integrator
//! of order p, then the composition:
//!
//! Ψₕ = Φ_{c₁h} ∘ Φ_{c₂h} ∘ ... ∘ Φ_{cₘh}
//!
//! with appropriate coefficients c₁, c₂, ..., cₘ, can achieve order p+2.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::symplectic::{HamiltonianFn, SymplecticIntegrator};
use ndarray::Array1;

/// A symplectic integrator constructed by composition of a base method
#[derive(Debug, Clone)]
pub struct CompositionMethod<F: IntegrateFloat, S: SymplecticIntegrator<F>> {
    /// Base integrator
    base_method: S,
    /// Coefficients for the composition
    coefficients: Vec<F>,
}

impl<F: IntegrateFloat, S: SymplecticIntegrator<F>> CompositionMethod<F, S> {
    /// Create a new composition method from a base integrator and coefficients
    ///
    /// # Arguments
    ///
    /// * `base_method` - The base symplectic integrator
    /// * `coefficients` - Coefficients for the composition steps
    ///
    /// # Returns
    ///
    /// A new composition method
    pub fn new(base_method: S, coefficients: Vec<F>) -> Self {
        CompositionMethod {
            base_method,
            coefficients,
        }
    }

    /// Create a 4th-order composition from a 2nd-order base method
    ///
    /// This uses the standard 3-stage composition with coefficients:
    /// c₁ = c₃ = 1/(2-2^(1/3)), c₂ = -2^(1/3)/(2-2^(1/3))
    ///
    /// # Arguments
    ///
    /// * `base_method` - A second-order symplectic integrator
    ///
    /// # Returns
    ///
    /// A fourth-order symplectic integrator
    pub fn fourth_order(base_method: S) -> Self {
        // Constants for Yoshida 4th-order composition
        let two = F::one() + F::one();
        let two_to_third = two.powf(F::from_f64(1.0 / 3.0).unwrap());
        let w1 = F::one() / (two - two_to_third);
        let w0 = -two_to_third / (two - two_to_third);

        // Coefficients: [w1, w0, w1]
        let coefficients = vec![w1, w0, w1];

        CompositionMethod {
            base_method,
            coefficients,
        }
    }

    /// Create a 6th-order composition from a 2nd-order base method
    ///
    /// This uses the standard 7-stage composition with optimized coefficients
    ///
    /// # Arguments
    ///
    /// * `base_method` - A second-order symplectic integrator
    ///
    /// # Returns
    ///
    /// A sixth-order symplectic integrator
    pub fn sixth_order(base_method: S) -> Self {
        // Coefficients for 6th-order composition method (Yoshida 1990)
        let w1 = F::from_f64(0.784513610477560).unwrap();
        let w2 = F::from_f64(0.235573213359357).unwrap();
        let w3 = F::from_f64(-1.17767998417887).unwrap();
        let w4 = F::from_f64(1.31518632068391).unwrap();

        // Create symmetric coefficients
        let coefficients = vec![w1, w2, w3, w4, w3, w2, w1];

        CompositionMethod {
            base_method,
            coefficients,
        }
    }

    /// Create an 8th-order composition method from a 2nd-order base method
    ///
    /// Uses a 15-stage composition with optimized coefficients
    ///
    /// # Arguments
    ///
    /// * `base_method` - A second-order symplectic integrator
    ///
    /// # Returns
    ///
    /// An eighth-order symplectic integrator
    pub fn eighth_order(base_method: S) -> Self {
        // Coefficients for 8th-order composition (Yoshida 1990)
        let w = [
            F::from_f64(0.74167036435061).unwrap(),
            F::from_f64(-0.40910082580003).unwrap(),
            F::from_f64(0.19075471029623).unwrap(),
            F::from_f64(-0.57386247111608).unwrap(),
            F::from_f64(0.29906418130365).unwrap(),
            F::from_f64(0.33462491824529).unwrap(),
            F::from_f64(0.31529309239676).unwrap(),
            F::from_f64(-0.79688793935291).unwrap(),
        ];

        // Create symmetric coefficients [w0, w1, ..., w7, w7, ..., w1, w0]
        let mut coefficients = Vec::with_capacity(15);
        for &coef in &w {
            coefficients.push(coef);
        }
        coefficients.push(w[7]);
        for i in (0..7).rev() {
            coefficients.push(w[i]);
        }

        CompositionMethod {
            base_method,
            coefficients,
        }
    }
}

impl<F: IntegrateFloat, S: SymplecticIntegrator<F>> SymplecticIntegrator<F>
    for CompositionMethod<F, S>
{
    fn step(
        &self,
        system: &dyn HamiltonianFn<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        // Start with the initial state
        let mut t_current = t;
        let mut q_current = q.clone();
        let mut p_current = p.clone();

        // Apply each sub-step with its coefficient
        for &coef in &self.coefficients {
            let dt_sub = dt * coef;
            let (q_next, p_next) = self
                .base_method
                .step(system, t_current, &q_current, &p_current, dt_sub)?;

            q_current = q_next;
            p_current = p_next;
            t_current += dt_sub;
        }

        Ok((q_current, p_current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symplectic::leapfrog::StormerVerlet;
    use crate::symplectic::potential::SeparableHamiltonian;
    use ndarray::array;

    /// Test error convergence rates for different order methods
    #[test]
    fn test_composition_convergence() {
        // Simple harmonic oscillator
        let system = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 { 0.5 * q.dot(q) },
        );

        // Initial state
        let q0 = array![1.0];
        let p0 = array![0.0];

        // Exact solution at t=1.0: q = cos(1.0), p = -sin(1.0)
        let q_exact = array![1.0_f64.cos()];
        let p_exact = array![-1.0_f64.sin()];

        // Create base 2nd-order method
        let base_method = StormerVerlet::new();

        // Create 4th-order method
        let fourth_order = CompositionMethod::fourth_order(base_method.clone());

        // Compare error convergence for different step sizes
        let dt_values = [0.1, 0.05, 0.025];

        let mut base_errors = Vec::new();
        let mut fourth_errors = Vec::new();

        for &dt in &dt_values {
            // Integrate to t=1.0 with base method
            let base_result = base_method
                .integrate(&system, 0.0, 1.0, dt, q0.clone(), p0.clone())
                .unwrap();

            // Compute error
            let idx = base_result.q.len() - 1;
            let q_base = &base_result.q[idx];
            let p_base = &base_result.p[idx];
            let base_error =
                ((q_base[0] - q_exact[0]).powi(2) + (p_base[0] - p_exact[0]).powi(2)).sqrt();
            base_errors.push(base_error);

            // Integrate with 4th-order method
            let fourth_result = fourth_order
                .integrate(&system, 0.0, 1.0, dt, q0.clone(), p0.clone())
                .unwrap();

            // Compute error
            let idx = fourth_result.q.len() - 1;
            let q_fourth = &fourth_result.q[idx];
            let p_fourth = &fourth_result.p[idx];
            let fourth_error =
                ((q_fourth[0] - q_exact[0]).powi(2) + (p_fourth[0] - p_exact[0]).powi(2)).sqrt();
            fourth_errors.push(fourth_error);
        }

        // Check convergence rate: when dt is halved, error should decrease by 2^p for p-th order method
        // For 2nd-order method, error ratio should be ~4
        // For 4th-order method, error ratio should be ~16
        for i in 0..dt_values.len() - 1 {
            let base_ratio = base_errors[i] / base_errors[i + 1];
            let fourth_ratio = fourth_errors[i] / fourth_errors[i + 1];

            // Allow some tolerance in ratio checks
            assert!(
                base_ratio > 3.5 && base_ratio < 4.5,
                "Base method convergence rate incorrect: {}",
                base_ratio
            );
            assert!(
                fourth_ratio > 12.0 && fourth_ratio < 20.0,
                "4th-order method convergence rate incorrect: {}",
                fourth_ratio
            );
        }

        // Check that 4th-order is more accurate than base method for same step size
        for i in 0..dt_values.len() {
            assert!(
                fourth_errors[i] < base_errors[i],
                "4th-order method should be more accurate than base method"
            );
        }
    }

    /// Test energy conservation over long time integration
    #[test]
    fn test_long_time_energy_conservation() {
        // Kepler problem (2D planetary orbit)
        let kepler = SeparableHamiltonian::new(
            |_t, p| -> f64 { 0.5 * p.dot(p) },
            |_t, q| -> f64 {
                let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
                if r < 1e-10 {
                    0.0
                } else {
                    -1.0 / r
                }
            },
        );

        // Initial condition for elliptical orbit
        let q0 = array![1.0, 0.0];
        let p0 = array![0.0, 1.2]; // Slightly faster than circular orbit

        // Integration parameters
        let t0 = 0.0;
        let tf = 100.0; // Long integration time
        let dt = 0.1;

        // Base 2nd-order method
        let base_method = StormerVerlet::new();

        // 4th-order composition method
        let fourth_order = CompositionMethod::fourth_order(base_method.clone());

        // Integrate with both methods
        let base_result = base_method
            .integrate(&kepler, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();
        let fourth_result = fourth_order
            .integrate(&kepler, t0, tf, dt, q0.clone(), p0.clone())
            .unwrap();

        // Check energy conservation - 4th-order should be better
        if let (Some(base_error), Some(fourth_error)) = (
            base_result.energy_relative_error,
            fourth_result.energy_relative_error,
        ) {
            assert!(
                fourth_error < base_error,
                "4th-order method should have better energy conservation. Base: {}, 4th-order: {}",
                base_error,
                fourth_error
            );
        }
    }
}
