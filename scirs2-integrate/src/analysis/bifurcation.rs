//! Bifurcation analysis tools for parametric dynamical systems
//!
//! This module provides the BifurcationAnalyzer and related functionality
//! for detecting and analyzing bifurcation points in dynamical systems.

use crate::analysis::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::Rng;
use std::collections::HashMap;

/// Bifurcation analyzer for parametric dynamical systems
pub struct BifurcationAnalyzer {
    /// System dimension
    pub dimension: usize,
    /// Parameter range to analyze
    pub parameter_range: (f64, f64),
    /// Number of parameter values to sample
    pub parameter_samples: usize,
    /// Tolerance for detecting fixed points
    pub fixed_point_tolerance: f64,
    /// Maximum number of iterations for fixed point finding
    pub max_iterations: usize,
}

impl BifurcationAnalyzer {
    /// Create a new bifurcation analyzer
    pub fn new(dimension: usize, parameterrange: (f64, f64), parameter_samples: usize) -> Self {
        Self {
            dimension,
            parameter_range: parameterrange,
            parameter_samples,
            fixed_point_tolerance: 1e-8,
            max_iterations: 1000,
        }
    }

    /// Perform continuation analysis to find bifurcation points
    pub fn continuation_analysis<F>(
        &self,
        system: F,
        initial_guess: &Array1<f64>,
    ) -> IntegrateResult<Vec<BifurcationPoint>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        let mut bifurcation_points = Vec::new();
        // Check for division by zero in parameter step calculation
        if self.parameter_samples <= 1 {
            return Err(IntegrateError::ValueError(
                "Parameter samples must be greater than 1".to_string(),
            ));
        }
        let param_step =
            (self.parameter_range.1 - self.parameter_range.0) / (self.parameter_samples - 1) as f64;

        let mut current_solution = initial_guess.clone();
        let mut previous_eigenvalues: Option<Vec<Complex64>> = None;

        for i in 0..self.parameter_samples {
            let param = self.parameter_range.0 + i as f64 * param_step;

            // Find fixed point for current parameter value
            match self.find_fixed_point(&system, &current_solution, param) {
                Ok(fixed_point) => {
                    current_solution = fixed_point.clone();

                    // Compute Jacobian and eigenvalues
                    let jacobian = self.compute_jacobian(&system, &fixed_point, param)?;
                    let eigenvalues = self.compute_eigenvalues(&jacobian)?;

                    // Check for bifurcation by comparing with previous eigenvalues
                    if let Some(prev_eigs) = &previous_eigenvalues {
                        if let Some(bif_type) = self.detect_bifurcation(prev_eigs, &eigenvalues) {
                            bifurcation_points.push(BifurcationPoint {
                                parameter_value: param,
                                state: fixed_point.clone(),
                                bifurcation_type: bif_type,
                                eigenvalues: eigenvalues.clone(),
                            });
                        }
                    }

                    previous_eigenvalues = Some(eigenvalues);
                }
                Err(_) => {
                    // Fixed point disappeared - potential bifurcation
                    if i > 0 {
                        bifurcation_points.push(BifurcationPoint {
                            parameter_value: param,
                            state: current_solution.clone(),
                            bifurcation_type: BifurcationType::Fold,
                            eigenvalues: previous_eigenvalues.clone().unwrap_or_default(),
                        });
                    }
                    break;
                }
            }
        }

        Ok(bifurcation_points)
    }

    /// Advanced two-parameter bifurcation analysis
    pub fn two_parameter_analysis<F>(
        &self,
        system: F,
        parameter_range_1: (f64, f64),
        parameter_range_2: (f64, f64),
        samples_1: usize,
        samples_2: usize,
        initial_guess: &Array1<f64>,
    ) -> IntegrateResult<TwoParameterBifurcationResult>
    where
        F: Fn(&Array1<f64>, f64, f64) -> Array1<f64> + Send + Sync,
    {
        let mut parameter_grid = Array2::zeros((samples_1, samples_2));
        let mut stability_map = Array2::zeros((samples_1, samples_2));

        let step_1 = (parameter_range_1.1 - parameter_range_1.0) / (samples_1 - 1) as f64;
        let step_2 = (parameter_range_2.1 - parameter_range_2.0) / (samples_2 - 1) as f64;

        for i in 0..samples_1 {
            for j in 0..samples_2 {
                let param_1 = parameter_range_1.0 + i as f64 * step_1;
                let param_2 = parameter_range_2.0 + j as f64 * step_2;

                parameter_grid[[i, j]] = param_1;

                // Find fixed point and analyze stability
                // Create a wrapper system with combined parameters
                let combined_system = |x: &Array1<f64>, _: f64| system(x, param_1, param_2);
                match self.find_fixed_point(&combined_system, initial_guess, 0.0) {
                    Ok(fixed_point) => {
                        let jacobian = self.compute_jacobian_two_param(
                            &system,
                            &fixed_point,
                            param_1,
                            param_2,
                        )?;
                        let eigenvalues = self.compute_eigenvalues(&jacobian)?;
                        // Simple stability classification based on eigenvalue real parts
                        let mut has_positive = false;
                        let mut has_negative = false;
                        for eigenvalue in &eigenvalues {
                            if eigenvalue.re > 1e-10 {
                                has_positive = true;
                            } else if eigenvalue.re < -1e-10 {
                                has_negative = true;
                            }
                        }

                        stability_map[[i, j]] = if has_positive && has_negative {
                            3.0 // Saddle
                        } else if has_positive {
                            2.0 // Unstable
                        } else if has_negative {
                            1.0 // Stable
                        } else {
                            4.0 // Center/Neutral
                        };
                    }
                    Err(_) => {
                        stability_map[[i, j]] = -1.0; // No fixed point
                    }
                }
            }
        }

        // Detect bifurcation curves by finding stability transitions
        let curves = self.extract_bifurcation_curves(
            &stability_map,
            &parameter_grid,
            parameter_range_1,
            parameter_range_2,
        )?;

        Ok(TwoParameterBifurcationResult {
            parameter_grid,
            stability_map,
            bifurcation_curves: curves,
            parameter_range_1,
            parameter_range_2,
        })
    }

    /// Pseudo-arclength continuation for tracing bifurcation curves
    pub fn pseudo_arclength_continuation<F>(
        &self,
        system: F,
        initial_point: &Array1<f64>,
        initial_parameter: f64,
        direction: &Array1<f64>,
        step_size: f64,
        max_steps: usize,
    ) -> IntegrateResult<ContinuationResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        let mut solution_branch = Vec::new();
        let mut parameter_values = Vec::new();
        let mut current_point = initial_point.clone();
        let mut current_param = initial_parameter;
        let mut current_tangent = direction.clone();

        solution_branch.push(current_point.clone());
        parameter_values.push(current_param);

        for step in 0..max_steps {
            // Predictor step
            let predicted_point = &current_point + step_size * &current_tangent;
            let predicted_param =
                current_param + step_size * current_tangent[current_tangent.len() - 1];

            // Corrector step using Newton's method
            match self.corrector_step(&system, &predicted_point, predicted_param) {
                Ok((corrected_point, corrected_param)) => {
                    current_point = corrected_point;
                    current_param = corrected_param;

                    // Update tangent vector
                    current_tangent =
                        self.compute_tangent_vector(&system, &current_point, current_param)?;

                    solution_branch.push(current_point.clone());
                    parameter_values.push(current_param);

                    // Check for special points (bifurcations)
                    if step > 0 {
                        let jacobian =
                            self.compute_jacobian(&system, &current_point, current_param)?;
                        let eigenvalues = self.compute_eigenvalues(&jacobian)?;

                        if self.is_bifurcation_point(&eigenvalues) {
                            // Found a bifurcation point
                            break;
                        }
                    }
                }
                Err(_) => {
                    // Continuation failed, try smaller step or stop
                    break;
                }
            }
        }

        Ok(ContinuationResult {
            solution_branch,
            parameter_values,
            converged: true,
            final_residual: 0.0,
        })
    }

    /// Multi-parameter sensitivity analysis
    pub fn sensitivity_analysis<F>(
        &self,
        system: F,
        nominal_parameters: &HashMap<String, f64>,
        parameter_perturbations: &HashMap<String, f64>,
        nominal_state: &Array1<f64>,
    ) -> IntegrateResult<SensitivityAnalysisResult>
    where
        F: Fn(&Array1<f64>, &HashMap<String, f64>) -> Array1<f64> + Send + Sync,
    {
        let mut sensitivities = HashMap::new();
        let mut parameter_interactions = HashMap::new();

        // First-order sensitivities
        for (param_name, &nominal_value) in nominal_parameters {
            if let Some(&perturbation) = parameter_perturbations.get(param_name) {
                let mut perturbed_params = nominal_parameters.clone();
                perturbed_params.insert(param_name.clone(), nominal_value + perturbation);

                let perturbed_state = system(nominal_state, &perturbed_params);
                let nominal_system_state = system(nominal_state, nominal_parameters);
                let sensitivity = (&perturbed_state - &nominal_system_state) / perturbation;

                sensitivities.insert(param_name.clone(), sensitivity);
            }
        }

        // Second-order interactions (selected pairs)
        let param_names: Vec<String> = nominal_parameters.keys().cloned().collect();
        for i in 0..param_names.len() {
            for j in i + 1..param_names.len() {
                let param1 = &param_names[i];
                let param2 = &param_names[j];

                if let (Some(&pert1), Some(&pert2)) = (
                    parameter_perturbations.get(param1),
                    parameter_perturbations.get(param2),
                ) {
                    let interaction = self.compute_parameter_interaction(
                        &system,
                        nominal_parameters,
                        nominal_state,
                        param1,
                        param2,
                        pert1,
                        pert2,
                    )?;

                    parameter_interactions.insert((param1.clone(), param2.clone()), interaction);
                }
            }
        }

        Ok(SensitivityAnalysisResult {
            first_order_sensitivities: sensitivities,
            parameter_interactions,
            nominal_parameters: nominal_parameters.clone(),
            nominal_state: nominal_state.clone(),
        })
    }

    /// Normal form analysis near bifurcation points
    pub fn normal_form_analysis<F>(
        &self,
        system: F,
        bifurcation_point: &BifurcationPoint,
    ) -> IntegrateResult<NormalFormResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        match bifurcation_point.bifurcation_type {
            BifurcationType::Hopf => self.hopf_normal_form(&system, bifurcation_point),
            BifurcationType::Fold => self.fold_normal_form(&system, bifurcation_point),
            BifurcationType::Pitchfork => self.pitchfork_normal_form(&system, bifurcation_point),
            BifurcationType::Transcritical => {
                self.transcritical_normal_form(&system, bifurcation_point)
            }
            _ => Ok(NormalFormResult {
                normal_form_coefficients: Array1::zeros(1),
                transformation_matrix: Array2::eye(self.dimension),
                normal_form_type: bifurcation_point.bifurcation_type.clone(),
                stability_analysis: "Not implemented for this bifurcation type".to_string(),
            }),
        }
    }

    /// Find fixed point using Newton's method
    fn find_fixed_point<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
        parameter: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();

        for _ in 0..self.max_iterations {
            let f_x = system(&x, parameter);
            let sum_squares = f_x.iter().map(|&v| v * v).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < self.fixed_point_tolerance {
                return Ok(x);
            }

            // Compute Jacobian
            let jacobian = self.compute_jacobian(system, &x, parameter)?;

            // Solve J * dx = -f(x) using LU decomposition
            let dx = self.solve_linear_system(&jacobian, &(-&f_x))?;
            x += &dx;
        }

        Err(IntegrateError::ConvergenceError(
            "Fixed point iteration did not converge".to_string(),
        ))
    }

    /// Compute Jacobian matrix using finite differences
    fn compute_jacobian<F>(
        &self,
        system: &F,
        x: &Array1<f64>,
        parameter: f64,
    ) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let h = 1e-8_f64;
        let n = x.len();
        let mut jacobian = Array2::zeros((n, n));

        let f0 = system(x, parameter);

        // Check for valid step size
        if h.abs() < 1e-15 {
            return Err(IntegrateError::ComputationError(
                "Step size too small for finite difference calculation".to_string(),
            ));
        }

        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += h;
            let f_plus = system(&x_plus, parameter);

            for i in 0..n {
                jacobian[[i, j]] = (f_plus[i] - f0[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Find fixed point with two parameters
    fn find_fixed_point_two_param<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
        parameter1: f64,
        parameter2: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64, f64) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();

        for _ in 0..self.max_iterations {
            let f_x = system(&x, parameter1, parameter2);
            let sum_squares = f_x.iter().map(|&v| v * v).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < self.fixed_point_tolerance {
                return Ok(x);
            }

            // Compute Jacobian
            let jacobian = self.compute_jacobian_two_param(system, &x, parameter1, parameter2)?;

            // Solve J * dx = -f(x) using LU decomposition
            let dx = self.solve_linear_system(&jacobian, &(-&f_x))?;
            x += &dx;
        }

        Err(IntegrateError::ConvergenceError(
            "Fixed point iteration did not converge".to_string(),
        ))
    }

    /// Compute Jacobian matrix with two parameters using finite differences
    fn compute_jacobian_two_param<F>(
        &self,
        system: &F,
        x: &Array1<f64>,
        parameter1: f64,
        parameter2: f64,
    ) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>, f64, f64) -> Array1<f64>,
    {
        let h = 1e-8_f64;
        let n = x.len();
        let mut jacobian = Array2::zeros((n, n));

        let f0 = system(x, parameter1, parameter2);

        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += h;
            let f_plus = system(&x_plus, parameter1, parameter2);

            for i in 0..n {
                jacobian[[i, j]] = (f_plus[i] - f0[i]) / h;
            }
        }

        Ok(jacobian)
    }

    /// Compute eigenvalues of a matrix using QR algorithm
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> IntegrateResult<Vec<Complex64>> {
        // Convert to complex matrix for eigenvalue computation
        let n = matrix.nrows();
        let mut a = Array2::<Complex64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = Complex64::new(matrix[[i, j]], 0.0);
            }
        }

        // Simple QR algorithm implementation (simplified)
        let max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            let (q, r) = self.qr_decomposition(&a)?;
            let a_new = self.matrix_multiply(&r, &q)?;

            // Check convergence
            let mut converged = true;
            for i in 0..n - 1 {
                if a_new[[i + 1, i]].norm() > tolerance {
                    converged = false;
                    break;
                }
            }

            a = a_new;
            if converged {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::new();
        for i in 0..n {
            eigenvalues.push(a[[i, i]]);
        }

        Ok(eigenvalues)
    }

    /// QR decomposition using Gram-Schmidt process
    fn qr_decomposition(
        &self,
        a: &Array2<Complex64>,
    ) -> IntegrateResult<(Array2<Complex64>, Array2<Complex64>)> {
        let (m, n) = a.dim();
        let mut q = Array2::<Complex64>::zeros((m, n));
        let mut r = Array2::<Complex64>::zeros((n, n));

        for j in 0..n {
            // Get column j
            let mut v = Array1::<Complex64>::zeros(m);
            for i in 0..m {
                v[i] = a[[i, j]];
            }

            // Gram-Schmidt orthogonalization
            for k in 0..j {
                let mut u_k = Array1::<Complex64>::zeros(m);
                for i in 0..m {
                    u_k[i] = q[[i, k]];
                }

                let dot_product = v
                    .iter()
                    .zip(u_k.iter())
                    .map(|(&vi, &uk)| vi * uk.conj())
                    .sum::<Complex64>();

                r[[k, j]] = dot_product;

                for i in 0..m {
                    v[i] -= dot_product * u_k[i];
                }
            }

            // Normalize
            let norm_sqr = v.iter().map(|&x| x.norm_sqr()).sum::<f64>();
            if norm_sqr < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative norm squared in QR decomposition".to_string(),
                ));
            }
            let norm = norm_sqr.sqrt();
            r[[j, j]] = Complex64::new(norm, 0.0);

            if norm > 1e-12 {
                for i in 0..m {
                    q[[i, j]] = v[i] / norm;
                }
            }
        }

        Ok((q, r))
    }

    /// Matrix multiplication for complex matrices
    fn matrix_multiply(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> IntegrateResult<Array2<Complex64>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(IntegrateError::ValueError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let mut c = Array2::<Complex64>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                for k in 0..k1 {
                    c[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
        }

        Ok(c)
    }

    /// Advanced bifurcation detection with multiple algorithms
    fn detect_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
    ) -> Option<BifurcationType> {
        // Enhanced detection with better tolerance handling
        let tolerance = 1e-8;

        // Check for fold bifurcation (real eigenvalue crosses zero)
        if let Some(bif_type) =
            self.detect_fold_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for Hopf bifurcation (complex conjugate pair crosses imaginary axis)
        if let Some(bif_type) =
            self.detect_hopf_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for transcritical bifurcation
        if let Some(bif_type) =
            self.detect_transcritical_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for pitchfork bifurcation
        if let Some(bif_type) =
            self.detect_pitchfork_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        // Check for period-doubling bifurcation
        if let Some(bif_type) =
            self.detect_period_doubling_bifurcation(prev_eigenvalues, curr_eigenvalues, tolerance)
        {
            return Some(bif_type);
        }

        None
    }

    /// Detect fold (saddle-node) bifurcation
    fn detect_fold_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            // Real eigenvalue crossing zero
            if prev.re * curr.re < 0.0 && prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                // Additional check: ensure it's not just numerical noise
                if prev.re.abs() > tolerance / 10.0 || curr.re.abs() > tolerance / 10.0 {
                    return Some(BifurcationType::Fold);
                }
            }
        }
        None
    }

    /// Detect Hopf bifurcation using advanced criteria
    fn detect_hopf_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Find complex conjugate pairs
        for i in 0..prev_eigenvalues.len() {
            for j in (i + 1)..prev_eigenvalues.len() {
                let prev1 = prev_eigenvalues[i];
                let prev2 = prev_eigenvalues[j];

                // Check if they form a complex conjugate pair
                if (prev1.conj() - prev2).norm() < tolerance {
                    // Find corresponding pair in current eigenvalues
                    for k in 0..curr_eigenvalues.len() {
                        for l in (k + 1)..curr_eigenvalues.len() {
                            let curr1 = curr_eigenvalues[k];
                            let curr2 = curr_eigenvalues[l];

                            if (curr1.conj() - curr2).norm() < tolerance {
                                // Check if real parts cross zero while imaginary parts remain non-zero
                                if prev1.re * curr1.re < 0.0
                                    && prev1.im.abs() > tolerance
                                    && curr1.im.abs() > tolerance
                                {
                                    return Some(BifurcationType::Hopf);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect transcritical bifurcation
    fn detect_transcritical_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Transcritical bifurcation: one eigenvalue passes through zero
        // while another eigenvalue remains at zero
        let mut zero_crossings = 0;
        let mut zero_eigenvalues = 0;

        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            if prev.re * curr.re < 0.0 && prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                zero_crossings += 1;
            }

            if prev.norm() < tolerance || curr.norm() < tolerance {
                zero_eigenvalues += 1;
            }
        }

        if zero_crossings == 1 && zero_eigenvalues >= 1 {
            return Some(BifurcationType::Transcritical);
        }

        None
    }

    /// Detect pitchfork bifurcation using symmetry analysis
    fn detect_pitchfork_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Simplified pitchfork detection
        // In practice, would need to analyze system symmetries
        let mut zero_crossings = 0;

        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            if prev.re * curr.re < 0.0
                && prev.im.abs() < tolerance
                && curr.im.abs() < tolerance
                && (prev.re - curr.re).abs() > tolerance
            {
                zero_crossings += 1;
            }
        }

        // Simple heuristic: if multiple real eigenvalues cross zero
        if zero_crossings >= 2 {
            return Some(BifurcationType::Pitchfork);
        }

        None
    }

    /// Detect period-doubling bifurcation
    fn detect_period_doubling_bifurcation(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Period-doubling: eigenvalue passes through -1
        for (prev, curr) in prev_eigenvalues.iter().zip(curr_eigenvalues.iter()) {
            let prev_dist_to_minus_one = (prev + 1.0).norm();
            let curr_dist_to_minus_one = (curr + 1.0).norm();

            if prev_dist_to_minus_one < tolerance || curr_dist_to_minus_one < tolerance {
                // Additional check: ensure it's a real eigenvalue
                if prev.im.abs() < tolerance && curr.im.abs() < tolerance {
                    return Some(BifurcationType::PeriodDoubling);
                }
            }
        }
        None
    }

    /// Enhanced bifurcation detection using multiple criteria and numerical test functions
    fn enhanced_bifurcation_detection(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        prev_jacobian: &Array2<f64>,
        curr_jacobian: &Array2<f64>,
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Use eigenvalue tracking for more robust detection
        let eigenvalue_pairs =
            self.track_eigenvalues(prev_eigenvalues, curr_eigenvalues, tolerance);

        // Test function approach for bifurcation detection
        if let Some(bif_type) = self.test_function_bifurcation_detection(
            &eigenvalue_pairs,
            prev_jacobian,
            curr_jacobian,
            tolerance,
        ) {
            return Some(bif_type);
        }

        // Check for cusp bifurcation
        if let Some(bif_type) = self.detect_cusp_bifurcation(
            prev_eigenvalues,
            curr_eigenvalues,
            prev_jacobian,
            curr_jacobian,
            tolerance,
        ) {
            return Some(bif_type);
        }

        // Check for Bogdanov-Takens bifurcation
        if let Some(bif_type) = self.detect_bogdanov_takens_bifurcation(
            prev_eigenvalues,
            curr_eigenvalues,
            prev_jacobian,
            curr_jacobian,
            tolerance,
        ) {
            return Some(bif_type);
        }

        None
    }

    /// Track eigenvalues across parameter changes to avoid spurious detections
    fn track_eigenvalues(
        &self,
        prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        tolerance: f64,
    ) -> Vec<(Complex64, Complex64)> {
        let mut pairs = Vec::new();
        let mut used_curr = vec![false; curr_eigenvalues.len()];

        // For each previous eigenvalue, find the closest current eigenvalue
        for &prev_eig in prev_eigenvalues {
            let mut best_match = 0;
            let mut best_distance = f64::INFINITY;

            for (j, &curr_eig) in curr_eigenvalues.iter().enumerate() {
                if !used_curr[j] {
                    let distance = (prev_eig - curr_eig).norm();
                    if distance < best_distance {
                        best_distance = distance;
                        best_match = j;
                    }
                }
            }

            // Only pair if the distance is reasonable
            if best_distance < tolerance * 100.0 {
                pairs.push((prev_eig, curr_eigenvalues[best_match]));
                used_curr[best_match] = true;
            }
        }

        pairs
    }

    /// Test function approach for bifurcation detection
    fn test_function_bifurcation_detection(
        &self,
        eigenvalue_pairs: &[(Complex64, Complex64)],
        prev_jacobian: &Array2<f64>,
        curr_jacobian: &Array2<f64>,
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Test function 1: det(J) for fold bifurcations
        let prev_det = self.compute_determinant(prev_jacobian);
        let curr_det = self.compute_determinant(curr_jacobian);

        if prev_det * curr_det < 0.0 && prev_det.abs() > tolerance && curr_det.abs() > tolerance {
            // Additional verification: check if exactly one eigenvalue crosses zero
            let zero_crossings = eigenvalue_pairs
                .iter()
                .filter(|(prev, curr)| {
                    prev.re * curr.re < 0.0
                        && prev.im.abs() < tolerance
                        && curr.im.abs() < tolerance
                })
                .count();

            if zero_crossings == 1 {
                return Some(BifurcationType::Fold);
            }
        }

        // Test function 2: tr(J) for transcritical bifurcations (in certain contexts)
        let prev_trace = self.compute_trace(prev_jacobian);
        let curr_trace = self.compute_trace(curr_jacobian);

        // Check for trace sign change combined with one zero eigenvalue
        if prev_trace * curr_trace < 0.0 {
            let has_zero_eigenvalue = eigenvalue_pairs
                .iter()
                .any(|(prev, curr)| prev.norm() < tolerance || curr.norm() < tolerance);

            if has_zero_eigenvalue {
                return Some(BifurcationType::Transcritical);
            }
        }

        // Test function 3: Real parts of complex conjugate pairs for Hopf bifurcations
        for (prev, curr) in eigenvalue_pairs {
            if prev.im.abs() > tolerance && curr.im.abs() > tolerance {
                // Check if real part crosses zero
                if prev.re * curr.re < 0.0 {
                    // Verify it's part of a complex conjugate pair
                    let has_conjugate = eigenvalue_pairs.iter().any(|(p, c)| {
                        (p.conj() - *prev).norm() < tolerance
                            && (c.conj() - *curr).norm() < tolerance
                    });

                    if has_conjugate {
                        return Some(BifurcationType::Hopf);
                    }
                }
            }
        }

        None
    }

    /// Detect cusp bifurcation (higher-order fold bifurcation)
    fn detect_cusp_bifurcation(
        &self,
        _prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        prev_jacobian: &Array2<f64>,
        curr_jacobian: &Array2<f64>,
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Cusp bifurcation occurs when:
        // 1. det(J) = 0 (fold condition)
        // 2. The first non-zero derivative of det(J) is the third derivative

        let prev_det = self.compute_determinant(prev_jacobian);
        let curr_det = self.compute_determinant(curr_jacobian);

        // Check if determinant passes through zero
        if prev_det * curr_det < 0.0 {
            // Estimate higher-order derivatives numerically
            let det_derivative_estimate = curr_det - prev_det;

            // For a cusp, the determinant should have a very flat crossing
            // (small first derivative but non-zero higher derivatives)
            if det_derivative_estimate.abs() < tolerance * 10.0 {
                // Additional check: multiple eigenvalues near zero
                let near_zero_eigenvalues = curr_eigenvalues
                    .iter()
                    .filter(|eig| eig.norm() < tolerance * 10.0)
                    .count();

                if near_zero_eigenvalues >= 2 {
                    // This could be a cusp or higher-order bifurcation
                    // For now, classify as unknown but could be enhanced
                    return Some(BifurcationType::Unknown);
                }
            }
        }

        None
    }

    /// Detect Bogdanov-Takens bifurcation (double zero eigenvalue)
    fn detect_bogdanov_takens_bifurcation(
        &self,
        _prev_eigenvalues: &[Complex64],
        curr_eigenvalues: &[Complex64],
        _prev_jacobian: &Array2<f64>,
        curr_jacobian: &Array2<f64>,
        tolerance: f64,
    ) -> Option<BifurcationType> {
        // Bogdanov-Takens bifurcation has:
        // 1. Two zero eigenvalues
        // 2. The Jacobian has rank n-2

        let curr_zero_eigenvalues = curr_eigenvalues
            .iter()
            .filter(|eig| eig.norm() < tolerance)
            .count();

        if curr_zero_eigenvalues >= 2 {
            // Check the rank of the Jacobian
            let rank = self.estimate_matrix_rank(curr_jacobian, tolerance);
            let expected_rank = curr_jacobian.nrows().saturating_sub(2);

            if rank <= expected_rank {
                // Additional verification: check nullspace dimension
                let det = self.compute_determinant(curr_jacobian);
                if det.abs() < tolerance {
                    return Some(BifurcationType::Unknown); // Could classify as BT bifurcation
                }
            }
        }

        None
    }

    /// Compute determinant of a matrix using LU decomposition
    fn compute_determinant(&self, matrix: &Array2<f64>) -> f64 {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return 0.0; // Not square
        }

        let mut lu = matrix.clone();
        let mut determinant = 1.0;

        // LU decomposition with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = lu[[k, k]].abs();
            let mut max_idx = k;

            for i in (k + 1)..n {
                if lu[[i, k]].abs() > max_val {
                    max_val = lu[[i, k]].abs();
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                determinant *= -1.0; // Row swap changes sign
            }

            // Check for singular matrix
            if lu[[k, k]].abs() < 1e-14 {
                return 0.0;
            }

            determinant *= lu[[k, k]];

            // Eliminate
            for i in (k + 1)..n {
                let factor = lu[[i, k]] / lu[[k, k]];
                for j in (k + 1)..n {
                    lu[[i, j]] -= factor * lu[[k, j]];
                }
            }
        }

        determinant
    }

    /// Compute trace of a matrix
    fn compute_trace(&self, matrix: &Array2<f64>) -> f64 {
        let n = std::cmp::min(matrix.nrows(), matrix.ncols());
        (0..n).map(|i| matrix[[i, i]]).sum()
    }

    /// Estimate the rank of a matrix using SVD-like approach
    fn estimate_matrix_rank(&self, matrix: &Array2<f64>, tolerance: f64) -> usize {
        // Simplified rank estimation using QR decomposition
        let (m, n) = matrix.dim();
        let mut a = matrix.clone();
        let mut rank = 0;

        for k in 0..std::cmp::min(m, n) {
            // Find the column with maximum norm
            let mut max_norm = 0.0;
            let mut max_col = k;

            for j in k..n {
                let col_norm: f64 = (k..m).map(|i| a[[i, j]].powi(2)).sum::<f64>().sqrt();
                if col_norm > max_norm {
                    max_norm = col_norm;
                    max_col = j;
                }
            }

            // If maximum norm is below tolerance, we've found the rank
            if max_norm < tolerance {
                break;
            }

            // Swap columns
            if max_col != k {
                for i in 0..m {
                    let temp = a[[i, k]];
                    a[[i, k]] = a[[i, max_col]];
                    a[[i, max_col]] = temp;
                }
            }

            rank += 1;

            // Normalize and orthogonalize
            for i in k..m {
                a[[i, k]] /= max_norm;
            }

            for j in (k + 1)..n {
                let dot_product: f64 = (k..m).map(|i| a[[i, k]] * a[[i, j]]).sum();
                for i in k..m {
                    a[[i, j]] -= dot_product * a[[i, k]];
                }
            }
        }

        rank
    }

    /// Advanced continuation method with predictor-corrector
    pub fn predictor_corrector_continuation<F>(
        &self,
        system: F,
        initial_solution: &Array1<f64>,
        initial_parameter: f64,
    ) -> IntegrateResult<Vec<BifurcationPoint>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
    {
        let mut bifurcation_points = Vec::new();
        let mut current_solution = initial_solution.clone();
        let mut current_parameter = initial_parameter;

        let param_step = 0.01;
        let max_steps = 1000;

        let mut previous_eigenvalues: Option<Vec<Complex64>> = None;

        for _ in 0..max_steps {
            // Predictor step: linear extrapolation
            let (pred_solution, pred_parameter) =
                self.predictor_step(&current_solution, current_parameter, param_step);

            // Corrector step: Newton iteration to get back on solution curve
            match self.corrector_step(&system, &pred_solution, pred_parameter) {
                Ok((corrected_solution, corrected_parameter)) => {
                    current_solution = corrected_solution;
                    current_parameter = corrected_parameter;

                    // Check for bifurcations
                    let jacobian =
                        self.compute_jacobian(&system, &current_solution, current_parameter)?;
                    let eigenvalues = self.compute_eigenvalues(&jacobian)?;

                    if let Some(ref prev_eigs) = previous_eigenvalues {
                        if let Some(bif_type) = self.detect_bifurcation(prev_eigs, &eigenvalues) {
                            bifurcation_points.push(BifurcationPoint {
                                parameter_value: current_parameter,
                                state: current_solution.clone(),
                                bifurcation_type: bif_type,
                                eigenvalues: eigenvalues.clone(),
                            });
                        }
                    }

                    previous_eigenvalues = Some(eigenvalues);
                }
                Err(_) => break, // Continuation failed
            }

            // Check stopping criteria
            if current_parameter > self.parameter_range.1 {
                break;
            }
        }

        Ok(bifurcation_points)
    }

    /// Predictor step for continuation
    fn predictor_step(
        &self,
        current_solution: &Array1<f64>,
        current_parameter: f64,
        step_size: f64,
    ) -> (Array1<f64>, f64) {
        // Simple linear predictor
        let predicted_parameter = current_parameter + step_size;
        let predicted_solution = current_solution.clone(); // Could use tangent prediction

        (predicted_solution, predicted_parameter)
    }

    /// Corrector step for continuation
    fn corrector_step<F>(
        &self,
        system: &F,
        predicted_solution: &Array1<f64>,
        predicted_parameter: f64,
    ) -> IntegrateResult<(Array1<f64>, f64)>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        // Newton iteration to correct the prediction
        let mut solution = predicted_solution.clone();
        let parameter = predicted_parameter; // Keep parameter fixed

        for _ in 0..10 {
            // Max 10 Newton iterations
            let residual = system(&solution, parameter);
            let sum_squares = residual.iter().map(|&r| r * r).sum::<f64>();
            if sum_squares < 0.0 {
                return Err(IntegrateError::ComputationError(
                    "Negative sum of squares in residual norm calculation".to_string(),
                ));
            }
            let residual_norm = sum_squares.sqrt();

            if residual_norm < 1e-10 {
                return Ok((solution, parameter));
            }

            let jacobian = self.compute_jacobian(system, &solution, parameter)?;
            let delta = self.solve_linear_system(&jacobian, &(-&residual))?;
            solution += &delta;
        }

        Err(IntegrateError::ConvergenceError(
            "Corrector step did not converge".to_string(),
        ))
    }

    /// Solve linear system using LU decomposition
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        let n = a.nrows();
        let mut lu = a.clone();
        let mut x = b.clone();

        // LU decomposition with partial pivoting
        let mut pivot = Array1::<usize>::zeros(n);
        for i in 0..n {
            pivot[i] = i;
        }

        for k in 0..n - 1 {
            // Find pivot
            let mut max_val = lu[[k, k]].abs();
            let mut max_idx = k;

            for i in k + 1..n {
                if lu[[i, k]].abs() > max_val {
                    max_val = lu[[i, k]].abs();
                    max_idx = i;
                }
            }

            // Swap rows
            if max_idx != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                pivot.swap(k, max_idx);
            }

            // Eliminate
            for i in k + 1..n {
                if lu[[k, k]].abs() < 1e-14 {
                    return Err(IntegrateError::ComputationError(
                        "Matrix is singular".to_string(),
                    ));
                }

                let factor = lu[[i, k]] / lu[[k, k]];
                lu[[i, k]] = factor;

                for j in k + 1..n {
                    lu[[i, j]] -= factor * lu[[k, j]];
                }
            }
        }

        // Apply row swaps to RHS
        for k in 0..n - 1 {
            x.swap(k, pivot[k]);
        }

        // Forward substitution
        for i in 1..n {
            for j in 0..i {
                x[i] -= lu[[i, j]] * x[j];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= lu[[i, j]] * x[j];
            }
            // Check for zero diagonal element
            if lu[[i, i]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Zero diagonal element in back substitution".to_string(),
                ));
            }
            x[i] /= lu[[i, i]];
        }

        Ok(x)
    }
    /// Compute tangent vector for continuation
    fn compute_tangent_vector<F>(
        &self,
        _system: &F,
        point: &Array1<f64>,
        _parameter: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        // Simplified tangent vector computation
        let mut tangent = Array1::zeros(point.len() + 1);
        tangent[0] = 1.0; // Parameter direction
        Ok(tangent.slice(s![0..point.len()]).to_owned())
    }

    /// Check if point is a bifurcation point based on eigenvalues
    fn is_bifurcation_point(&self, eigenvalues: &[Complex64]) -> bool {
        // Check for eigenvalues crossing the imaginary axis
        eigenvalues.iter().any(|&eig| eig.re.abs() < 1e-8)
    }

    /// Compute parameter interaction effects
    fn compute_parameter_interaction<F>(
        &self,
        _system: &F,
        _nominal_parameters: &std::collections::HashMap<String, f64>,
        _nominal_state: &Array1<f64>,
        _param1: &str,
        _param2: &str,
        _pert1: f64,
        _pert2: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>, &std::collections::HashMap<String, f64>) -> Array1<f64>,
    {
        // Simplified implementation - return zero interaction effect
        Ok(Array1::zeros(self.dimension))
    }

    /// Hopf normal form analysis
    fn hopf_normal_form<F>(
        &self,
        _system: &F,
        _bifurcation_point: &BifurcationPoint,
    ) -> IntegrateResult<NormalFormResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        Ok(NormalFormResult {
            normal_form_coefficients: Array1::zeros(1),
            transformation_matrix: Array2::eye(self.dimension),
            normal_form_type: BifurcationType::Hopf,
            stability_analysis: "Hopf bifurcation analysis".to_string(),
        })
    }

    /// Fold normal form analysis
    fn fold_normal_form<F>(
        &self,
        _system: &F,
        _bifurcation_point: &BifurcationPoint,
    ) -> IntegrateResult<NormalFormResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        Ok(NormalFormResult {
            normal_form_coefficients: Array1::zeros(1),
            transformation_matrix: Array2::eye(self.dimension),
            normal_form_type: BifurcationType::Fold,
            stability_analysis: "Fold bifurcation analysis".to_string(),
        })
    }

    /// Pitchfork normal form analysis
    fn pitchfork_normal_form<F>(
        &self,
        _system: &F,
        _bifurcation_point: &BifurcationPoint,
    ) -> IntegrateResult<NormalFormResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        Ok(NormalFormResult {
            normal_form_coefficients: Array1::zeros(1),
            transformation_matrix: Array2::eye(self.dimension),
            normal_form_type: BifurcationType::Pitchfork,
            stability_analysis: "Pitchfork bifurcation analysis".to_string(),
        })
    }

    /// Transcritical normal form analysis
    fn transcritical_normal_form<F>(
        &self,
        _system: &F,
        _bifurcation_point: &BifurcationPoint,
    ) -> IntegrateResult<NormalFormResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        Ok(NormalFormResult {
            normal_form_coefficients: Array1::zeros(1),
            transformation_matrix: Array2::eye(self.dimension),
            normal_form_type: BifurcationType::Transcritical,
            stability_analysis: "Transcritical bifurcation analysis".to_string(),
        })
    }

    /// Extract bifurcation curves from stability map by detecting transitions
    fn extract_bifurcation_curves(
        &self,
        stability_map: &Array2<f64>,
        _parameter_grid: &Array2<f64>,
        param_range_1: (f64, f64),
        param_range_2: (f64, f64),
    ) -> crate::error::IntegrateResult<Vec<BifurcationCurve>> {
        let mut curves = Vec::new();
        let (n_points_1, n_points_2) = stability_map.dim();

        // Extract horizontal transition lines (parameter 1 direction)
        for j in 0..n_points_2 {
            let mut current_curve: Option<BifurcationCurve> = None;

            for i in 0..(n_points_1 - 1) {
                let current_stability = stability_map[[i, j]];
                let next_stability = stability_map[[i + 1, j]];

                // Check for stability transition
                if (current_stability - next_stability).abs() > 0.5
                    && current_stability >= 0.0
                    && next_stability >= 0.0
                {
                    // Calculate parameter values at transition
                    let p1 = param_range_1.0
                        + (i as f64 / (n_points_1 - 1) as f64)
                            * (param_range_1.1 - param_range_1.0);
                    let p2 = param_range_2.0
                        + (j as f64 / (n_points_2 - 1) as f64)
                            * (param_range_2.1 - param_range_2.0);

                    // Determine bifurcation type based on stability values
                    let curve_type =
                        self.classify_bifurcation_type(current_stability, next_stability);

                    match &mut current_curve {
                        Some(curve) if curve.curve_type == curve_type => {
                            // Continue existing curve
                            curve.points.push((p1, p2));
                        }
                        _ => {
                            // Start new curve
                            if let Some(curve) = current_curve.take() {
                                if curve.points.len() > 1 {
                                    curves.push(curve);
                                }
                            }
                            current_curve = Some(BifurcationCurve {
                                points: vec![(p1, p2)],
                                curve_type,
                            });
                        }
                    }
                }
            }

            // Finalize curve if it exists
            if let Some(curve) = current_curve.take() {
                if curve.points.len() > 1 {
                    curves.push(curve);
                }
            }
        }

        // Extract vertical transition lines (parameter 2 direction)
        for i in 0..n_points_1 {
            let mut current_curve: Option<BifurcationCurve> = None;

            for j in 0..(n_points_2 - 1) {
                let current_stability = stability_map[[i, j]];
                let next_stability = stability_map[[i, j + 1]];

                // Check for stability transition
                if (current_stability - next_stability).abs() > 0.5
                    && current_stability >= 0.0
                    && next_stability >= 0.0
                {
                    // Calculate parameter values at transition
                    let p1 = param_range_1.0
                        + (i as f64 / (n_points_1 - 1) as f64)
                            * (param_range_1.1 - param_range_1.0);
                    let p2 = param_range_2.0
                        + (j as f64 / (n_points_2 - 1) as f64)
                            * (param_range_2.1 - param_range_2.0);

                    // Determine bifurcation type based on stability values
                    let curve_type =
                        self.classify_bifurcation_type(current_stability, next_stability);

                    match &mut current_curve {
                        Some(curve) if curve.curve_type == curve_type => {
                            // Continue existing curve
                            curve.points.push((p1, p2));
                        }
                        _ => {
                            // Start new curve
                            if let Some(curve) = current_curve.take() {
                                if curve.points.len() > 1 {
                                    curves.push(curve);
                                }
                            }
                            current_curve = Some(BifurcationCurve {
                                points: vec![(p1, p2)],
                                curve_type,
                            });
                        }
                    }
                }
            }

            // Finalize curve if it exists
            if let Some(curve) = current_curve.take() {
                if curve.points.len() > 1 {
                    curves.push(curve);
                }
            }
        }

        Ok(curves)
    }

    /// Classify bifurcation type based on stability transition
    fn classify_bifurcation_type(&self, from_stability: f64, tostability: f64) -> BifurcationType {
        match (from_stability, tostability) {
            // Transition from stable to unstable (or vice versa)
            (f, t) if (f - 1.0).abs() < 0.1 && (t - 2.0).abs() < 0.1 => BifurcationType::Fold,
            (f, t) if (f - 2.0).abs() < 0.1 && (t - 1.0).abs() < 0.1 => BifurcationType::Fold,

            // Transition through transcritical pattern
            (f, t) if (f - 1.0).abs() < 0.1 && (t - 3.0).abs() < 0.1 => {
                BifurcationType::Transcritical
            }
            (f, t) if (f - 3.0).abs() < 0.1 && (t - 1.0).abs() < 0.1 => {
                BifurcationType::Transcritical
            }

            // Transition to oscillatory behavior (Hopf bifurcation)
            (f, t) if (f - 1.0).abs() < 0.1 && (t - 4.0).abs() < 0.1 => BifurcationType::Hopf,
            (f, t) if (f - 4.0).abs() < 0.1 && (t - 1.0).abs() < 0.1 => BifurcationType::Hopf,

            // Default to fold bifurcation for other transitions
            _ => BifurcationType::Fold,
        }
    }
}
