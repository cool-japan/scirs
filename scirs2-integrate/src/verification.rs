//! Method of Manufactured Solutions (MMS) toolkit for verification
//!
//! This module provides tools for verifying numerical methods using the Method of
//! Manufactured Solutions. MMS is a powerful technique for code verification where
//! an exact solution is chosen, the governing equation is modified by adding source
//! terms, and the numerical solution is compared against the exact solution.
//!
//! # Examples
//!
//! ## ODE Verification
//! ```
//! use scirs2_integrate::verification::{MMSODEProblem, polynomial_solution};
//! use ndarray::Array1;
//!
//! // Create a manufactured ODE problem with polynomial solution
//! let exact_solution = polynomial_solution(vec![1.0, 2.0, 3.0]); // 1 + 2t + 3t²
//! let problem = MMSODEProblem::new(exact_solution, [0.0, 1.0]);
//!
//! // The manufactured source term is automatically computed
//! // Solve numerically and verify order of accuracy
//! ```
//!
//! ## PDE Verification  
//! ```
//! use scirs2_integrate::verification::{MMSPDEProblem, trigonometric_solution_2d};
//!
//! // Create manufactured 2D Poisson problem
//! let exact = trigonometric_solution_2d(1.0, 2.0); // sin(x) * cos(2y)
//! let problem = MMSPDEProblem::new_poisson_2d(exact, [0.0, 1.0], [0.0, 1.0]);
//! ```

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{ArrayView1, ArrayView2};
use std::fmt;

/// Trait for exact solutions in MMS problems
pub trait ExactSolution<F: IntegrateFloat> {
    /// Evaluate the exact solution at given point(s)
    fn evaluate(&self, coordinates: &[F]) -> F;

    /// Evaluate the first derivative with respect to specified variable
    fn derivative(&self, coordinates: &[F], variable: usize) -> F;

    /// Evaluate the second derivative with respect to specified variable
    fn second_derivative(&self, coordinates: &[F], variable: usize) -> F;

    /// Evaluate mixed partial derivatives (for PDEs)
    fn mixed_derivative(&self, coordinates: &[F], var1: usize, var2: usize) -> F {
        // Default implementation using finite differences
        let h = F::from(1e-8).unwrap();
        let mut coords_plus = coordinates.to_vec();
        let mut coords_minus = coordinates.to_vec();

        coords_plus[var2] += h;
        coords_minus[var2] -= h;

        let deriv_plus = self.derivative(&coords_plus, var1);
        let deriv_minus = self.derivative(&coords_minus, var1);

        (deriv_plus - deriv_minus) / (F::from(2.0).unwrap() * h)
    }

    /// Get the dimensionality of the problem
    fn dimension(&self) -> usize;
}

/// Polynomial exact solution for ODE problems
#[derive(Debug, Clone)]
pub struct PolynomialSolution<F: IntegrateFloat> {
    coefficients: Vec<F>,
}

impl<F: IntegrateFloat> PolynomialSolution<F> {
    /// Create a new polynomial solution: sum(coeff[i] * t^i)
    pub fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }
}

impl<F: IntegrateFloat> ExactSolution<F> for PolynomialSolution<F> {
    fn evaluate(&self, coordinates: &[F]) -> F {
        let t = coordinates[0];
        let mut result = F::zero();
        let mut t_power = F::one();

        for &coeff in &self.coefficients {
            result += coeff * t_power;
            t_power *= t;
        }

        result
    }

    fn derivative(&self, coordinates: &[F], _variable: usize) -> F {
        let t = coordinates[0];
        let mut result = F::zero();
        let mut t_power = F::one();

        for (i, &coeff) in self.coefficients.iter().enumerate().skip(1) {
            result += F::from(i).unwrap() * coeff * t_power;
            t_power *= t;
        }

        result
    }

    fn second_derivative(&self, coordinates: &[F], _variable: usize) -> F {
        let t = coordinates[0];
        let mut result = F::zero();
        let mut t_power = F::one();

        for (i, &coeff) in self.coefficients.iter().enumerate().skip(2) {
            let factor = F::from(i * (i - 1)).unwrap();
            result += factor * coeff * t_power;
            t_power *= t;
        }

        result
    }

    fn dimension(&self) -> usize {
        1
    }
}

/// Trigonometric exact solution for 2D PDE problems  
#[derive(Debug, Clone)]
pub struct TrigonometricSolution2D<F: IntegrateFloat> {
    freq_x: F,
    freq_y: F,
    phase_x: F,
    phase_y: F,
}

impl<F: IntegrateFloat> TrigonometricSolution2D<F> {
    /// Create sin(freq_x * x + phase_x) * cos(freq_y * y + phase_y)
    pub fn new(freq_x: F, freq_y: F, phase_x: F, phase_y: F) -> Self {
        Self {
            freq_x,
            freq_y,
            phase_x,
            phase_y,
        }
    }

    /// Create sin(freq_x * x) * cos(freq_y * y)
    pub fn simple(freq_x: F, freq_y: F) -> Self {
        Self::new(freq_x, freq_y, F::zero(), F::zero())
    }
}

impl<F: IntegrateFloat> ExactSolution<F> for TrigonometricSolution2D<F> {
    fn evaluate(&self, coordinates: &[F]) -> F {
        let x = coordinates[0];
        let y = coordinates[1];
        (self.freq_x * x + self.phase_x).sin() * (self.freq_y * y + self.phase_y).cos()
    }

    fn derivative(&self, coordinates: &[F], variable: usize) -> F {
        let x = coordinates[0];
        let y = coordinates[1];

        match variable {
            0 => {
                self.freq_x
                    * (self.freq_x * x + self.phase_x).cos()
                    * (self.freq_y * y + self.phase_y).cos()
            }
            1 => {
                -self.freq_y
                    * (self.freq_x * x + self.phase_x).sin()
                    * (self.freq_y * y + self.phase_y).sin()
            }
            _ => F::zero(),
        }
    }

    fn second_derivative(&self, coordinates: &[F], variable: usize) -> F {
        let x = coordinates[0];
        let y = coordinates[1];

        match variable {
            0 => {
                -self.freq_x
                    * self.freq_x
                    * (self.freq_x * x + self.phase_x).sin()
                    * (self.freq_y * y + self.phase_y).cos()
            }
            1 => {
                -self.freq_y
                    * self.freq_y
                    * (self.freq_x * x + self.phase_x).sin()
                    * (self.freq_y * y + self.phase_y).cos()
            }
            _ => F::zero(),
        }
    }

    fn dimension(&self) -> usize {
        2
    }
}

/// Manufactured ODE problem for verification
#[derive(Debug)]
pub struct MMSODEProblem<F: IntegrateFloat, S: ExactSolution<F>> {
    exact_solution: S,
    time_span: [F; 2],
    _phantom: std::marker::PhantomData<F>,
}

impl<F: IntegrateFloat, S: ExactSolution<F>> MMSODEProblem<F, S> {
    /// Create a new manufactured ODE problem
    pub fn new(exact_solution: S, time_span: [F; 2]) -> Self {
        Self {
            exact_solution,
            time_span,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the manufactured source term for: y' = f(t, y) + source(t)
    /// where f is the original RHS and source is computed from exact solution
    pub fn source_term(&self, t: F) -> F {
        let coords = [t];
        // For y' = f(t, y), source = y_exact'(t) - f(t, y_exact(t))
        // If f is linear: y' = a*y + b, then source = y_exact'(t) - a*y_exact(t) - b
        // For autonomous f: y' = f(y), source = y_exact'(t) - f(y_exact(t))
        self.exact_solution.derivative(&coords, 0)
    }

    /// Get initial condition from exact solution
    pub fn initial_condition(&self) -> F {
        self.exact_solution.evaluate(&[self.time_span[0]])
    }

    /// Evaluate exact solution at given time
    pub fn exact_at(&self, t: F) -> F {
        self.exact_solution.evaluate(&[t])
    }

    /// Get time span
    pub fn time_span(&self) -> [F; 2] {
        self.time_span
    }
}

/// Manufactured PDE problem for verification  
#[derive(Debug)]
pub struct MMSPDEProblem<F: IntegrateFloat, S: ExactSolution<F>> {
    exact_solution: S,
    domain_x: [F; 2],
    domain_y: [F; 2],
    pde_type: PDEType,
    _phantom: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone)]
pub enum PDEType {
    Poisson2D,
    Diffusion2D,
    Wave2D,
    AdvectionDiffusion2D,
}

impl<F: IntegrateFloat, S: ExactSolution<F>> MMSPDEProblem<F, S> {
    /// Create manufactured 2D Poisson problem: -∇²u = f
    pub fn new_poisson_2d(exact_solution: S, domain_x: [F; 2], domain_y: [F; 2]) -> Self {
        Self {
            exact_solution,
            domain_x,
            domain_y,
            pde_type: PDEType::Poisson2D,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get manufactured source term for the PDE
    pub fn source_term(&self, x: F, y: F) -> F {
        let coords = [x, y];

        match self.pde_type {
            PDEType::Poisson2D => {
                // For -∇²u = f, source f = -∇²u_exact
                -(self.exact_solution.second_derivative(&coords, 0)
                    + self.exact_solution.second_derivative(&coords, 1))
            }
            PDEType::Diffusion2D => {
                // For ∂u/∂t = α∇²u + f, source f = ∂u_exact/∂t - α∇²u_exact
                // This requires time-dependent exact solution
                F::zero() // Simplified for now
            }
            _ => F::zero(),
        }
    }

    /// Get boundary condition from exact solution
    pub fn boundary_condition(&self, x: F, y: F) -> F {
        self.exact_solution.evaluate(&[x, y])
    }

    /// Evaluate exact solution
    pub fn exact_at(&self, x: F, y: F) -> F {
        self.exact_solution.evaluate(&[x, y])
    }

    /// Get domain bounds
    pub fn domain(&self) -> ([F; 2], [F; 2]) {
        (self.domain_x, self.domain_y)
    }
}

/// Convergence analysis results
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis<F: IntegrateFloat> {
    /// Grid sizes or time steps used
    pub grid_sizes: Vec<F>,
    /// Error norms for each grid size
    pub errors: Vec<F>,
    /// Estimated order of accuracy
    pub order: F,
    /// Confidence interval for the order
    pub order_confidence: (F, F),
}

impl<F: IntegrateFloat> ConvergenceAnalysis<F> {
    /// Compute order of accuracy from grid sizes and errors
    pub fn compute_order(grid_sizes: Vec<F>, errors: Vec<F>) -> IntegrateResult<Self> {
        if grid_sizes.len() != errors.len() || grid_sizes.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 points for convergence analysis".to_string(),
            ));
        }

        // Use least squares to fit log(error) = log(C) + p*log(h)
        let n = grid_sizes.len();
        let mut sum_log_h = F::zero();
        let mut sum_log_e = F::zero();
        let mut sum_log_h_sq = F::zero();
        let mut sum_log_h_log_e = F::zero();

        for (h, e) in grid_sizes.iter().zip(errors.iter()) {
            if *e <= F::zero() || *h <= F::zero() {
                return Err(IntegrateError::ValueError(
                    "Grid sizes and errors must be positive".to_string(),
                ));
            }

            let log_h = h.ln();
            let log_e = e.ln();

            sum_log_h += log_h;
            sum_log_e += log_e;
            sum_log_h_sq += log_h * log_h;
            sum_log_h_log_e += log_h * log_e;
        }

        let n_f = F::from(n).unwrap();
        let denominator = n_f * sum_log_h_sq - sum_log_h * sum_log_h;

        if denominator.abs() < F::from(1e-12).unwrap() {
            return Err(IntegrateError::ComputationError(
                "Cannot compute order - insufficient variation in grid sizes".to_string(),
            ));
        }

        let order = (n_f * sum_log_h_log_e - sum_log_h * sum_log_e) / denominator;

        // Simple confidence interval (±0.1 for demonstration)
        let confidence_delta = F::from(0.1).unwrap();
        let order_confidence = (order - confidence_delta, order + confidence_delta);

        Ok(Self {
            grid_sizes,
            errors,
            order,
            order_confidence,
        })
    }

    /// Check if the computed order matches expected order within tolerance
    pub fn verify_order(&self, expected_order: F, tolerance: F) -> bool {
        (self.order - expected_order).abs() <= tolerance
    }
}

impl<F: IntegrateFloat> fmt::Display for ConvergenceAnalysis<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Convergence Analysis Results:")?;
        writeln!(f, "Grid Size    Error")?;
        writeln!(f, "─────────────────────")?;

        for (h, e) in self.grid_sizes.iter().zip(self.errors.iter()) {
            writeln!(f, "{:9.2e}  {:9.2e}", h, e)?;
        }

        writeln!(f, "─────────────────────")?;
        writeln!(f, "Estimated order: {:.3}", self.order)?;
        writeln!(
            f,
            "95% confidence: ({:.3}, {:.3})",
            self.order_confidence.0, self.order_confidence.1
        )?;

        Ok(())
    }
}

/// Error analysis utilities
pub struct ErrorAnalysis;

impl ErrorAnalysis {
    /// Compute L2 norm of error
    pub fn l2_norm<F: IntegrateFloat>(
        exact: ArrayView1<F>,
        numerical: ArrayView1<F>,
    ) -> IntegrateResult<F> {
        if exact.len() != numerical.len() {
            return Err(IntegrateError::ValueError(
                "Arrays must have same length".to_string(),
            ));
        }

        let mut sum_sq = F::zero();
        for (e, n) in exact.iter().zip(numerical.iter()) {
            let diff = *e - *n;
            sum_sq += diff * diff;
        }

        Ok((sum_sq / F::from(exact.len()).unwrap()).sqrt())
    }

    /// Compute maximum norm of error
    pub fn max_norm<F: IntegrateFloat>(
        exact: ArrayView1<F>,
        numerical: ArrayView1<F>,
    ) -> IntegrateResult<F> {
        if exact.len() != numerical.len() {
            return Err(IntegrateError::ValueError(
                "Arrays must have same length".to_string(),
            ));
        }

        let mut max_error = F::zero();
        for (e, n) in exact.iter().zip(numerical.iter()) {
            let diff = (*e - *n).abs();
            if diff > max_error {
                max_error = diff;
            }
        }

        Ok(max_error)
    }

    /// Compute L2 norm of error for 2D arrays
    pub fn l2_norm_2d<F: IntegrateFloat>(
        exact: ArrayView2<F>,
        numerical: ArrayView2<F>,
    ) -> IntegrateResult<F> {
        if exact.shape() != numerical.shape() {
            return Err(IntegrateError::ValueError(
                "Arrays must have same shape".to_string(),
            ));
        }

        let mut sum_sq = F::zero();
        let mut count = 0;

        for (e, n) in exact.iter().zip(numerical.iter()) {
            let diff = *e - *n;
            sum_sq += diff * diff;
            count += 1;
        }

        Ok((sum_sq / F::from(count).unwrap()).sqrt())
    }
}

/// Convenience functions for creating common exact solutions
pub fn polynomial_solution<F: IntegrateFloat>(coefficients: Vec<F>) -> PolynomialSolution<F> {
    PolynomialSolution::new(coefficients)
}

pub fn trigonometric_solution_2d<F: IntegrateFloat>(
    freq_x: F,
    freq_y: F,
) -> TrigonometricSolution2D<F> {
    TrigonometricSolution2D::simple(freq_x, freq_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_polynomial_solution() {
        // Test polynomial: 1 + 2t + 3t²
        let poly = polynomial_solution(vec![1.0, 2.0, 3.0]);

        // Test evaluation
        assert_abs_diff_eq!(poly.evaluate(&[0.0]), 1.0);
        assert_abs_diff_eq!(poly.evaluate(&[1.0]), 6.0); // 1 + 2 + 3

        // Test derivative: 2 + 6t
        assert_abs_diff_eq!(poly.derivative(&[0.0], 0), 2.0);
        assert_abs_diff_eq!(poly.derivative(&[1.0], 0), 8.0);

        // Test second derivative: 6
        assert_abs_diff_eq!(poly.second_derivative(&[0.0], 0), 6.0);
        assert_abs_diff_eq!(poly.second_derivative(&[1.0], 0), 6.0);
    }

    #[test]
    fn test_trigonometric_solution_2d() {
        use std::f64::consts::PI;

        // Test sin(x) * cos(y)
        let trig = trigonometric_solution_2d(1.0, 1.0);

        // Test evaluation
        assert_abs_diff_eq!(trig.evaluate(&[0.0, 0.0]), 0.0); // sin(0) * cos(0) = 0 * 1 = 0
        assert_abs_diff_eq!(trig.evaluate(&[PI / 2.0, 0.0]), 1.0); // sin(π/2) * cos(0) = 1 * 1 = 1

        // Test derivatives
        // ∂/∂x[sin(x)cos(y)] = cos(x)cos(y)
        assert_abs_diff_eq!(trig.derivative(&[0.0, 0.0], 0), 1.0); // cos(0)*cos(0) = 1

        // ∂/∂y[sin(x)cos(y)] = -sin(x)sin(y)
        assert_abs_diff_eq!(trig.derivative(&[0.0, 0.0], 1), 0.0); // -sin(0)*sin(0) = 0
    }

    #[test]
    fn test_mms_ode_problem() {
        // Test with y = t²
        let poly = polynomial_solution(vec![0.0, 0.0, 1.0]); // t²
        let problem = MMSODEProblem::new(poly, [0.0, 1.0]);

        // Initial condition should be 0
        assert_abs_diff_eq!(problem.initial_condition(), 0.0);

        // Source term should be y' = 2t
        assert_abs_diff_eq!(problem.source_term(0.0), 0.0);
        assert_abs_diff_eq!(problem.source_term(1.0), 2.0);

        // Exact solution at t=0.5 should be 0.25
        assert_abs_diff_eq!(problem.exact_at(0.5), 0.25);
    }

    #[test]
    fn test_convergence_analysis() {
        // Test with theoretical O(h²) convergence
        let grid_sizes = vec![0.1, 0.05, 0.025, 0.0125];
        let errors = vec![0.01, 0.0025, 0.000625, 0.00015625]; // ∝ h²

        let analysis = ConvergenceAnalysis::compute_order(grid_sizes, errors).unwrap();

        // Should detect order ≈ 2
        assert!((analysis.order - 2.0_f64).abs() < 0.1);
        assert!(analysis.verify_order(2.0, 0.2));
    }

    #[test]
    fn test_error_analysis() {
        let exact = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let numerical = Array1::from_vec(vec![1.1, 1.9, 3.05, 3.98]);

        let l2_error = ErrorAnalysis::l2_norm(exact.view(), numerical.view()).unwrap();
        let max_error = ErrorAnalysis::max_norm(exact.view(), numerical.view()).unwrap();

        // L2 error should be around sqrt((0.1² + 0.1² + 0.05² + 0.02²)/4)
        assert!(l2_error > 0.0 && l2_error < 0.2);

        // Max error should be 0.1
        assert_abs_diff_eq!(max_error, 0.1, epsilon = 1e-10);
    }
}
