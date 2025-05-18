//! Dense output for ODE solvers
//!
//! This module provides data structures and functions to enable continuous
//! output from ODE solvers, allowing evaluation at any point within the
//! integration interval without recomputing the solution.

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::utils::interpolation::{
    cubic_hermite_interpolation, linear_interpolation, ContinuousOutputMethod,
};
use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};
use std::fmt::Debug;

/// Type alias for derivative function
type DerivativeFunction<F> = Box<dyn Fn(F, ArrayView1<F>) -> Array1<F>>;

/// A dense solution that supports evaluation at any time within the integration range
pub struct DenseSolution<F: IntegrateFloat> {
    /// Time points from the discrete solution
    pub t: Vec<F>,
    /// Solution values at time points
    pub y: Vec<Array1<F>>,
    /// Derivatives at time points (if available)
    pub dydt: Option<Vec<Array1<F>>>,
    /// Interpolation method to use
    pub method: ContinuousOutputMethod,
    /// Function to evaluate derivatives (if derivatives not provided)
    pub f: Option<DerivativeFunction<F>>,
}

impl<F: IntegrateFloat> Debug for DenseSolution<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseSolution")
            .field("t", &self.t)
            .field("y", &self.y)
            .field("dydt", &self.dydt)
            .field("method", &self.method)
            .field("f", &"<closure>")
            .finish()
    }
}

impl<F: IntegrateFloat> DenseSolution<F> {
    /// Create a new dense solution object from a discrete solution
    pub fn new(
        t: Vec<F>,
        y: Vec<Array1<F>>,
        dydt: Option<Vec<Array1<F>>>,
        method: Option<ContinuousOutputMethod>,
        f: Option<DerivativeFunction<F>>,
    ) -> Self {
        let interp_method = method.unwrap_or_default();

        // If cubic Hermite interpolation is requested but no derivatives are provided,
        // we need the function to compute derivatives on demand
        if interp_method == ContinuousOutputMethod::CubicHermite && dydt.is_none() && f.is_none() {
            // Fall back to linear interpolation if no derivatives available
            return DenseSolution {
                t,
                y,
                dydt: None,
                method: ContinuousOutputMethod::Linear,
                f: None,
            };
        }

        DenseSolution {
            t,
            y,
            dydt,
            method: interp_method,
            f,
        }
    }

    /// Evaluate the solution at a specific time point
    pub fn evaluate(&self, t: F) -> IntegrateResult<Array1<F>> {
        // Check if t is within bounds
        let t_min = self
            .t
            .first()
            .ok_or_else(|| IntegrateError::ComputationError("Empty solution".to_string()))?;

        let t_max = self
            .t
            .last()
            .ok_or_else(|| IntegrateError::ComputationError("Empty solution".to_string()))?;

        if t < *t_min || t > *t_max {
            return Err(IntegrateError::ValueError(format!(
                "Evaluation time {} is outside the solution range [{}, {}]",
                t, t_min, t_max
            )));
        }

        // If at an existing time point, just return that value
        for (i, &ti) in self.t.iter().enumerate() {
            if (t - ti).abs() < F::from_f64(1e-14).unwrap() {
                return Ok(self.y[i].clone());
            }
        }

        // Use the appropriate interpolation method
        match self.method {
            ContinuousOutputMethod::Linear => {
                // Use simple linear interpolation
                Ok(linear_interpolation(&self.t, &self.y, t))
            }
            ContinuousOutputMethod::CubicHermite => {
                if let Some(ref dydt) = self.dydt {
                    // Use cubic Hermite interpolation with stored derivatives
                    Ok(cubic_hermite_interpolation(&self.t, &self.y, dydt, t))
                } else if let Some(ref f) = self.f {
                    // Compute derivatives on demand
                    let mut dydt = Vec::with_capacity(self.t.len());
                    for i in 0..self.t.len() {
                        dydt.push(f(self.t[i], self.y[i].view()));
                    }

                    Ok(cubic_hermite_interpolation(&self.t, &self.y, &dydt, t))
                } else {
                    // Fall back to linear interpolation
                    Ok(linear_interpolation(&self.t, &self.y, t))
                }
            }
            ContinuousOutputMethod::MethodSpecific => {
                // Fall back to standard cubic Hermite interpolation
                // In a real implementation, method-specific interpolants would be used
                if let Some(ref dydt) = self.dydt {
                    Ok(cubic_hermite_interpolation(&self.t, &self.y, dydt, t))
                } else if let Some(ref f) = self.f {
                    let mut dydt = Vec::with_capacity(self.t.len());
                    for i in 0..self.t.len() {
                        dydt.push(f(self.t[i], self.y[i].view()));
                    }

                    Ok(cubic_hermite_interpolation(&self.t, &self.y, &dydt, t))
                } else {
                    Ok(linear_interpolation(&self.t, &self.y, t))
                }
            }
        }
    }

    /// Create a dense sequence of solution values for plotting or analysis
    pub fn dense_output(&self, n_points: usize) -> IntegrateResult<(Vec<F>, Vec<Array1<F>>)> {
        if self.t.is_empty() {
            return Err(IntegrateError::ComputationError(
                "Empty solution".to_string(),
            ));
        }

        let t_min = *self.t.first().unwrap();
        let t_max = *self.t.last().unwrap();
        let dt = (t_max - t_min) / F::from_usize(n_points - 1).unwrap();

        let mut times = Vec::with_capacity(n_points);
        let mut values = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let t = t_min + dt * F::from_usize(i).unwrap();
            times.push(t);
            values.push(self.evaluate(t)?);
        }

        Ok((times, values))
    }

    /// Extract a single component from the solution at dense points
    pub fn extract_component(
        &self,
        component: usize,
        n_points: usize,
    ) -> IntegrateResult<(Vec<F>, Vec<F>)> {
        // Make sure the component index is valid
        if self.y.is_empty() {
            return Err(IntegrateError::ComputationError(
                "Empty solution".to_string(),
            ));
        }

        let dim = self.y[0].len();
        if component >= dim {
            return Err(IntegrateError::ValueError(format!(
                "Component index {} is out of bounds (0-{})",
                component,
                dim - 1
            )));
        }

        // Get dense output
        let (times, values) = self.dense_output(n_points)?;

        // Extract the specified component
        let component_values = values.iter().map(|v| v[component]).collect();

        Ok((times, component_values))
    }
}

/// Interpolation for DOP853 (8th order Dormand-Prince) method
#[derive(Debug, Clone)]
pub struct DOP853Interpolant<F: IntegrateFloat> {
    /// Time at the beginning of the step
    pub t0: F,
    /// Step size
    pub h: F,
    /// Solution at the beginning of the step
    pub y0: Array1<F>,
    /// RK stage values
    pub k: Vec<Array1<F>>,
}

impl<F: IntegrateFloat> DOP853Interpolant<F> {
    /// Create a new DOP853 interpolant
    pub fn new(t0: F, h: F, y0: Array1<F>, k: Vec<Array1<F>>) -> Self {
        DOP853Interpolant { t0, h, y0, k }
    }

    /// Evaluate the interpolant at a specific point within the step
    pub fn evaluate(&self, t: F) -> IntegrateResult<Array1<F>> {
        // Normalized time (theta) in [0,1]
        let theta = (t - self.t0) / self.h;

        if theta < F::zero() || theta > F::one() {
            return Err(IntegrateError::ValueError(
                "Evaluation point is outside of the step".to_string(),
            ));
        }

        // These coefficients are simplified for brevity
        // In a real implementation, the full set of coefficients for the
        // DOP853 dense output would be used
        let b1 = theta;
        let b2 = theta * theta / F::from_f64(2.0).unwrap();
        let b3 = theta * theta * theta / F::from_f64(6.0).unwrap();
        let b4 = theta * theta * theta * theta / F::from_f64(24.0).unwrap();
        let b5 = theta * theta * theta * theta * theta / F::from_f64(120.0).unwrap();
        let b6 = theta * theta * theta * theta * theta * theta / F::from_f64(720.0).unwrap();
        let b7 =
            theta * theta * theta * theta * theta * theta * theta / F::from_f64(5040.0).unwrap();

        // Compute the interpolant using the stage values
        let mut result = self.y0.clone();
        result += &(self.k[0].clone() * self.h * b1);
        result += &(self.k[1].clone() * self.h * b2);
        result += &(self.k[2].clone() * self.h * b3);
        result += &(self.k[3].clone() * self.h * b4);
        result += &(self.k[4].clone() * self.h * b5);
        result += &(self.k[5].clone() * self.h * b6);

        // Only use k6 if available (depends on how many stages were stored)
        if self.k.len() > 6 {
            result += &(self.k[6].clone() * self.h * b7);
        }

        Ok(result)
    }
}

/// Interpolation for Radau method
#[derive(Debug, Clone)]
pub struct RadauInterpolant<F: IntegrateFloat> {
    /// Time at the beginning of the step
    pub t0: F,
    /// Step size
    pub h: F,
    /// Solution at the beginning of the step
    pub y0: Array1<F>,
    /// Solution at the end of the step
    pub y1: Array1<F>,
    /// Stage values
    pub k: Vec<Array1<F>>,
}

impl<F: IntegrateFloat> RadauInterpolant<F> {
    /// Create a new Radau interpolant
    pub fn new(t0: F, h: F, y0: Array1<F>, y1: Array1<F>, k: Vec<Array1<F>>) -> Self {
        RadauInterpolant { t0, h, y0, y1, k }
    }

    /// Evaluate the interpolant at a specific point within the step
    pub fn evaluate(&self, t: F) -> IntegrateResult<Array1<F>> {
        // Normalized time (theta) in [0,1]
        let theta = (t - self.t0) / self.h;

        if theta < F::zero() || theta > F::one() {
            return Err(IntegrateError::ValueError(
                "Evaluation point is outside of the step".to_string(),
            ));
        }

        // For Radau method, we use a cubic Hermite interpolation
        // matching function values at both ends and using the stage
        // values to approximate derivatives

        let h00 = F::from_f64(2.0).unwrap() * theta.powi(3)
            - F::from_f64(3.0).unwrap() * theta.powi(2)
            + F::one();
        let h10 = theta.powi(3) - F::from_f64(2.0).unwrap() * theta.powi(2) + theta;
        let h01 =
            F::from_f64(-2.0).unwrap() * theta.powi(3) + F::from_f64(3.0).unwrap() * theta.powi(2);
        let h11 = theta.powi(3) - theta.powi(2);

        // Use the first stage value as derivative at the beginning
        // and the last stage value as derivative at the end
        let dy0 = &self.k[0];
        let dy1 = if self.k.len() > 1 {
            &self.k[self.k.len() - 1]
        } else {
            &self.k[0]
        };

        // Compute interpolant
        let mut result = Array1::zeros(self.y0.dim());

        for i in 0..self.y0.len() {
            result[i] =
                h00 * self.y0[i] + h10 * self.h * dy0[i] + h01 * self.y1[i] + h11 * self.h * dy1[i];
        }

        Ok(result)
    }
}

/// Convert an ODE result to a dense solution for continuous evaluation
pub fn create_dense_solution<F, Func>(
    t: Vec<F>,
    y: Vec<Array1<F>>,
    f: Func,
    method: Option<ContinuousOutputMethod>,
) -> IntegrateResult<DenseSolution<F>>
where
    F: IntegrateFloat,
    Func: 'static + Fn(F, ArrayView1<F>) -> Array1<F>,
{
    if t.is_empty() || y.is_empty() {
        return Err(IntegrateError::ComputationError(
            "Empty solution cannot be converted to dense output".to_string(),
        ));
    }

    if t.len() != y.len() {
        return Err(IntegrateError::DimensionMismatch(
            "Time and solution vectors must have the same length".to_string(),
        ));
    }

    // Compute derivatives at all points if cubic Hermite interpolation is requested
    let interp_method = method.unwrap_or_default();
    let dydt = if interp_method == ContinuousOutputMethod::CubicHermite {
        let mut derivatives = Vec::with_capacity(t.len());
        for i in 0..t.len() {
            derivatives.push(f(t[i], y[i].view()));
        }
        Some(derivatives)
    } else {
        None
    };

    Ok(DenseSolution::new(t, y, dydt, method, Some(Box::new(f))))
}
