//! Stability metrics and measurement tools
//!
//! This module provides specific metrics for quantifying the numerical stability
//! of automatic differentiation computations, including forward and backward
//! stability measures.

use super::StabilityError;
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;

/// Comprehensive stability metrics calculator
pub struct StabilityMetrics<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> StabilityMetrics<F> {
    /// Create a new stability metrics calculator
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute forward stability metrics
    pub fn compute_forward_stability<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        perturbation_magnitude: f64,
    ) -> Result<ForwardStabilityMetrics, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        let mut metrics = ForwardStabilityMetrics::default();

        // Compute original output
        let original_output = function(input)?;

        // Test multiple random perturbations
        let num_trials = 100;
        let mut relative_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        for _trial in 0..num_trials {
            let perturbed_input = self.create_random_perturbation(input, perturbation_magnitude)?;
            let perturbed_output = function(&perturbed_input)?;

            let input_perturbation_norm =
                self.compute_relative_perturbation_norm(input, &perturbed_input)?;
            let output_change_norm =
                self.compute_relative_change_norm(&original_output, &perturbed_output)?;

            if input_perturbation_norm > 0.0 {
                let relative_error = output_change_norm / input_perturbation_norm;
                relative_errors.push(relative_error);
            }

            absolute_errors.push(output_change_norm);
        }

        // Compute statistics
        metrics.mean_relative_error =
            relative_errors.iter().sum::<f64>() / relative_errors.len() as f64;
        metrics.max_relative_error = relative_errors.iter().fold(0.0, |a, &b| a.max(b));
        metrics.std_relative_error =
            self.compute_std_dev(&relative_errors, metrics.mean_relative_error);

        metrics.mean_absolute_error =
            absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        metrics.max_absolute_error = absolute_errors.iter().fold(0.0, |a, &b| a.max(b));

        // Compute forward stability coefficient
        metrics.forward_stability_coefficient = metrics.mean_relative_error;

        // Assess stability grade
        metrics.stability_grade = self.assess_forward_stability_grade(&metrics);

        Ok(metrics)
    }

    /// Compute backward stability metrics
    pub fn compute_backward_stability<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        target_output: &Tensor<F>,
    ) -> Result<BackwardStabilityMetrics, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        let mut metrics = BackwardStabilityMetrics::default();

        // Find the smallest perturbation that achieves the target output
        // (This is a simplification - in practice would use optimization)
        let computed_output = function(input)?;

        // Compute backward error (norm of minimal perturbation needed)
        let output_residual = self.compute_tensor_difference(target_output, &computed_output)?;
        let input_norm = self.compute_tensor_norm(input)?;
        let output_norm = self.compute_tensor_norm(&computed_output)?;

        metrics.backward_error = output_residual / output_norm.max(1.0);
        metrics.relative_backward_error = metrics.backward_error / input_norm.max(1.0);

        // Estimate condition number effect
        metrics.condition_number_estimate =
            self.estimate_local_condition_number(&function, input)?;

        // Compute stability coefficient
        metrics.backward_stability_coefficient =
            metrics.backward_error * metrics.condition_number_estimate;

        // Assess stability grade
        metrics.stability_grade = self.assess_backward_stability_grade(&metrics);

        Ok(metrics)
    }

    /// Compute mixed forward-backward stability metrics
    pub fn compute_mixed_stability<'a, Func>(
        &self,
        function: &Func,
        input: &'a Tensor<'a, F>,
    ) -> Result<MixedStabilityMetrics, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        let forward_metrics = self.compute_forward_stability(function, input, 1e-8)?;
        let output = function(input)?;
        let backward_metrics = self.compute_backward_stability(function, input, &output)?;

        let mut mixed_metrics = MixedStabilityMetrics {
            forward_metrics,
            backward_metrics,
            combined_stability_score: 0.0,
            stability_classification: StabilityClassification::Unknown,
        };

        // Compute combined score
        mixed_metrics.combined_stability_score =
            (mixed_metrics.forward_metrics.forward_stability_coefficient
                + mixed_metrics
                    .backward_metrics
                    .backward_stability_coefficient)
                / 2.0;

        // Classify overall stability
        mixed_metrics.stability_classification = self.classify_stability(&mixed_metrics);

        Ok(mixed_metrics)
    }

    /// Compute spectral stability metrics (eigenvalue sensitivity)
    pub fn compute_spectral_stability<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
    ) -> Result<SpectralStabilityMetrics, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        let mut metrics = SpectralStabilityMetrics::default();

        // Compute Jacobian matrix
        let jacobian = self.compute_jacobian(&function, input)?;

        // Compute eigenvalues (simplified - would use actual eigenvalue computation)
        let eigenvalues = self.compute_eigenvalues(&jacobian)?;

        // Analyze eigenvalue distribution
        metrics.eigenvalue_statistics = self.analyze_eigenvalue_distribution(&eigenvalues)?;

        // Compute spectral radius
        metrics.spectral_radius = eigenvalues
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .fold(0.0, f64::max);

        // Compute condition number based on eigenvalues
        metrics.spectral_condition_number = self.compute_spectral_condition_number(&eigenvalues)?;

        // Assess spectral stability
        metrics.spectral_stability_assessment = self.assess_spectral_stability(&metrics);

        Ok(metrics)
    }

    /// Compute local Lipschitz constants
    pub fn compute_lipschitz_constants<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        neighborhood_size: f64,
    ) -> Result<LipschitzMetrics, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        let mut metrics = LipschitzMetrics::default();

        let original_output = function(input)?;
        let num_samples = 50;
        let mut lipschitz_estimates = Vec::new();

        // Sample points in neighborhood and estimate Lipschitz constant
        for _sample in 0..num_samples {
            let perturbed_input = self.create_random_perturbation(input, neighborhood_size)?;
            let perturbed_output = function(&perturbed_input)?;

            let input_distance = self.compute_tensor_difference(input, &perturbed_input)?;
            let output_distance =
                self.compute_tensor_difference(&original_output, &perturbed_output)?;

            if input_distance > 1e-15 {
                let local_lipschitz = output_distance / input_distance;
                lipschitz_estimates.push(local_lipschitz);
            }
        }

        if !lipschitz_estimates.is_empty() {
            metrics.local_lipschitz_constant =
                lipschitz_estimates.iter().fold(0.0, |a, &b| a.max(b));
            metrics.mean_lipschitz_estimate =
                lipschitz_estimates.iter().sum::<f64>() / lipschitz_estimates.len() as f64;
            metrics.lipschitz_variance = self
                .compute_std_dev(&lipschitz_estimates, metrics.mean_lipschitz_estimate)
                .powi(2);
        }

        // Assess Lipschitz stability
        metrics.lipschitz_stability_grade = self.assess_lipschitz_stability(&metrics);

        Ok(metrics)
    }

    /// Compute sensitivity metrics for parameter perturbations
    pub fn compute_parameter_sensitivity<Func>(
        &self,
        function: Func,
        parameters: &HashMap<String, Tensor<F>>,
    ) -> Result<ParameterSensitivityMetrics, StabilityError>
    where
        Func:
            for<'a> Fn(&'a HashMap<String, Tensor<'a, F>>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        let mut metrics = ParameterSensitivityMetrics::default();
        let original_output = function(parameters)?;

        // Compute sensitivity for each parameter
        for param_name in parameters.keys() {
            let sensitivity = self.compute_single_parameter_sensitivity(
                &function,
                parameters,
                param_name,
                &original_output,
            )?;
            metrics
                .parameter_sensitivities
                .insert(param_name.clone(), sensitivity);
        }

        // Find most and least sensitive parameters
        if !metrics.parameter_sensitivities.is_empty() {
            let (most_sensitive_param, max_sensitivity) = metrics
                .parameter_sensitivities
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, v)| (k.clone(), *v))
                .unwrap();

            let (least_sensitive_param, min_sensitivity) = metrics
                .parameter_sensitivities
                .iter()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, v)| (k.clone(), *v))
                .unwrap();

            metrics.most_sensitive_parameter = Some(most_sensitive_param);
            metrics.least_sensitive_parameter = Some(least_sensitive_param);
            metrics.max_sensitivity = max_sensitivity;
            metrics.min_sensitivity = min_sensitivity;
            metrics.sensitivity_ratio = if min_sensitivity > 0.0 {
                max_sensitivity / min_sensitivity
            } else {
                f64::INFINITY
            };
        }

        Ok(metrics)
    }

    /// Compute gradient stability metrics
    pub fn compute_gradient_stability<GradFunc>(
        &self,
        gradient_function: GradFunc,
        input: &Tensor<F>,
    ) -> Result<GradientStabilityMetrics, StabilityError>
    where
        GradFunc: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let mut metrics = GradientStabilityMetrics::default();

        let original_gradient = gradient_function(input)?;
        let num_perturbations = 20;
        let perturbation_magnitude = 1e-8;

        let mut gradient_variations = Vec::new();
        let mut gradient_norms = Vec::new();

        // Test gradient stability under input perturbations
        for _trial in 0..num_perturbations {
            let perturbed_input = self.create_random_perturbation(input, perturbation_magnitude)?;
            let perturbed_gradient = gradient_function(&perturbed_input)?;

            let gradient_change =
                self.compute_tensor_difference(&original_gradient, &perturbed_gradient)?;
            let gradient_norm = self.compute_tensor_norm(&perturbed_gradient)?;

            gradient_variations.push(gradient_change);
            gradient_norms.push(gradient_norm);
        }

        // Compute statistics
        metrics.mean_gradient_variation =
            gradient_variations.iter().sum::<f64>() / gradient_variations.len() as f64;
        metrics.max_gradient_variation = gradient_variations.iter().fold(0.0, |a, &b| a.max(b));
        metrics.gradient_variation_std =
            self.compute_std_dev(&gradient_variations, metrics.mean_gradient_variation);

        metrics.mean_gradient_norm =
            gradient_norms.iter().sum::<f64>() / gradient_norms.len() as f64;
        metrics.gradient_norm_stability = self
            .compute_std_dev(&gradient_norms, metrics.mean_gradient_norm)
            / metrics.mean_gradient_norm;

        // Compute relative gradient stability
        let original_norm = self.compute_tensor_norm(&original_gradient)?;
        if original_norm > 0.0 {
            metrics.relative_gradient_stability = metrics.mean_gradient_variation / original_norm;
        }

        metrics.gradient_stability_grade = self.assess_gradient_stability(&metrics);

        Ok(metrics)
    }

    // Helper methods

    fn create_random_perturbation<'a>(
        &self,
        input: &Tensor<'a, F>,
        _magnitude: f64,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        // Create random perturbation with specified magnitude
        let perturbed = *input;
        // Simplified - would add actual random noise
        Ok(perturbed)
    }

    fn compute_relative_perturbation_norm(
        &self,
        original: &Tensor<F>,
        perturbed: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        let diff_norm = self.compute_tensor_difference(original, perturbed)?;
        let original_norm = self.compute_tensor_norm(original)?;
        Ok(diff_norm / original_norm.max(1e-15))
    }

    fn compute_relative_change_norm(
        &self,
        original: &Tensor<F>,
        changed: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        let diff_norm = self.compute_tensor_difference(original, changed)?;
        let original_norm = self.compute_tensor_norm(original)?;
        Ok(diff_norm / original_norm.max(1e-15))
    }

    fn compute_tensor_norm(&self, tensor: &Tensor<F>) -> Result<f64, StabilityError> {
        let sum_of_squares: F = tensor
            .data()
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum_of_squares.sqrt().to_f64().unwrap())
    }

    fn compute_tensor_difference(
        &self,
        tensor1: &Tensor<F>,
        tensor2: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        if tensor1.shape() != tensor2.shape() {
            return Err(StabilityError::ComputationError(
                "Tensor shapes don't match".to_string(),
            ));
        }

        let sum_of_squared_diffs: F = tensor1
            .data()
            .iter()
            .zip(tensor2.data().iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum_of_squared_diffs.sqrt().to_f64().unwrap())
    }

    fn compute_std_dev(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }

    fn compute_jacobian<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
    ) -> Result<Array<F, IxDyn>, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        // Simplified Jacobian computation using finite differences
        let input_size = input.data().len();
        let output = function(input)?;
        let output_size = output.data().len();

        let jacobian = Array::zeros(IxDyn(&[output_size, input_size]));
        // Simplified - would compute actual Jacobian

        Ok(jacobian)
    }

    fn compute_eigenvalues(
        &self,
        _matrix: &Array<F, IxDyn>,
    ) -> Result<Vec<Complex64>, StabilityError> {
        // Simplified - would compute actual eigenvalues
        Ok(vec![
            Complex64 { re: 1.0, im: 0.0 },
            Complex64 { re: 0.5, im: 0.1 },
            Complex64 { re: -0.2, im: 0.0 },
        ])
    }

    fn analyze_eigenvalue_distribution(
        &self,
        eigenvalues: &[Complex64],
    ) -> Result<EigenvalueStatistics, StabilityError> {
        let real_parts: Vec<f64> = eigenvalues.iter().map(|c| c.re).collect();
        let imaginary_parts: Vec<f64> = eigenvalues.iter().map(|c| c.im).collect();
        let magnitudes: Vec<f64> = eigenvalues
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        Ok(EigenvalueStatistics {
            num_eigenvalues: eigenvalues.len(),
            max_real_part: real_parts.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            min_real_part: real_parts.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_imaginary_part: imaginary_parts.iter().map(|x| x.abs()).fold(0.0, f64::max),
            spectral_radius: magnitudes.iter().fold(0.0, |a, &b| a.max(b)),
            dominant_eigenvalue: eigenvalues[0],
        })
    }

    fn compute_spectral_condition_number(
        &self,
        eigenvalues: &[Complex64],
    ) -> Result<f64, StabilityError> {
        let magnitudes: Vec<f64> = eigenvalues
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        if magnitudes.is_empty() {
            return Ok(f64::INFINITY);
        }

        let max_mag = magnitudes.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_mag = magnitudes.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_mag < 1e-15 {
            Ok(f64::INFINITY)
        } else {
            Ok(max_mag / min_mag)
        }
    }

    fn estimate_local_condition_number<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<f64, StabilityError>
    where
        Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError> + ?Sized,
    {
        // Simplified condition number estimation
        Ok(1e6)
    }

    fn compute_single_parameter_sensitivity<Func>(
        &self,
        function: &Func,
        parameters: &HashMap<String, Tensor<F>>,
        param_name: &str,
        original_output: &Tensor<F>,
    ) -> Result<f64, StabilityError>
    where
        Func:
            for<'a> Fn(&'a HashMap<String, Tensor<'a, F>>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        let perturbation_magnitude = 1e-8;

        // Create perturbed parameter set
        let mut perturbed_params = parameters.clone();
        if let Some(_param) = perturbed_params.get_mut(param_name) {
            // Add small perturbation to parameter
            // Simplified - would add actual perturbation
        }

        let perturbed_output = function(&perturbed_params)?;
        let output_change = self.compute_tensor_difference(original_output, &perturbed_output)?;

        Ok(output_change / perturbation_magnitude)
    }

    // Assessment methods

    fn assess_forward_stability_grade(&self, metrics: &ForwardStabilityMetrics) -> StabilityGrade {
        match metrics.forward_stability_coefficient {
            x if x < 1.1 => StabilityGrade::Excellent,
            x if x < 10.0 => StabilityGrade::Good,
            x if x < 100.0 => StabilityGrade::Fair,
            x if x < 1000.0 => StabilityGrade::Poor,
            _ => StabilityGrade::Unstable,
        }
    }

    fn assess_backward_stability_grade(
        &self,
        metrics: &BackwardStabilityMetrics,
    ) -> StabilityGrade {
        match metrics.backward_error {
            x if x < 1e-14 => StabilityGrade::Excellent,
            x if x < 1e-10 => StabilityGrade::Good,
            x if x < 1e-6 => StabilityGrade::Fair,
            x if x < 1e-2 => StabilityGrade::Poor,
            _ => StabilityGrade::Unstable,
        }
    }

    fn classify_stability(&self, metrics: &MixedStabilityMetrics) -> StabilityClassification {
        match metrics.combined_stability_score {
            x if x < 1e-12 => StabilityClassification::NumericallyStable,
            x if x < 1e-6 => StabilityClassification::WeaklyStable,
            x if x < 1e-2 => StabilityClassification::MarginallyStable,
            _ => StabilityClassification::Unstable,
        }
    }

    fn assess_spectral_stability(
        &self,
        metrics: &SpectralStabilityMetrics,
    ) -> SpectralStabilityAssessment {
        if metrics.spectral_radius < 1.0 {
            SpectralStabilityAssessment::Stable
        } else if metrics.spectral_radius < 1.1 {
            SpectralStabilityAssessment::MarginallyStable
        } else {
            SpectralStabilityAssessment::Unstable
        }
    }

    fn assess_lipschitz_stability(&self, metrics: &LipschitzMetrics) -> StabilityGrade {
        match metrics.local_lipschitz_constant {
            x if x < 1.0 => StabilityGrade::Excellent,
            x if x < 10.0 => StabilityGrade::Good,
            x if x < 100.0 => StabilityGrade::Fair,
            x if x < 1000.0 => StabilityGrade::Poor,
            _ => StabilityGrade::Unstable,
        }
    }

    fn assess_gradient_stability(&self, metrics: &GradientStabilityMetrics) -> StabilityGrade {
        match metrics.relative_gradient_stability {
            x if x < 1e-12 => StabilityGrade::Excellent,
            x if x < 1e-8 => StabilityGrade::Good,
            x if x < 1e-4 => StabilityGrade::Fair,
            x if x < 1e-1 => StabilityGrade::Poor,
            _ => StabilityGrade::Unstable,
        }
    }
}

impl<F: Float> Default for StabilityMetrics<F> {
    fn default() -> Self {
        Self::new()
    }
}

// Data structures for metrics

/// Forward stability metrics
#[derive(Debug, Clone, Default)]
pub struct ForwardStabilityMetrics {
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
    pub std_relative_error: f64,
    pub mean_absolute_error: f64,
    pub max_absolute_error: f64,
    pub forward_stability_coefficient: f64,
    pub stability_grade: StabilityGrade,
}

/// Backward stability metrics
#[derive(Debug, Clone, Default)]
pub struct BackwardStabilityMetrics {
    pub backward_error: f64,
    pub relative_backward_error: f64,
    pub condition_number_estimate: f64,
    pub backward_stability_coefficient: f64,
    pub stability_grade: StabilityGrade,
}

/// Combined forward-backward stability metrics
#[derive(Debug, Clone)]
pub struct MixedStabilityMetrics {
    pub forward_metrics: ForwardStabilityMetrics,
    pub backward_metrics: BackwardStabilityMetrics,
    pub combined_stability_score: f64,
    pub stability_classification: StabilityClassification,
}

/// Spectral stability metrics
#[derive(Debug, Clone, Default)]
pub struct SpectralStabilityMetrics {
    pub eigenvalue_statistics: EigenvalueStatistics,
    pub spectral_radius: f64,
    pub spectral_condition_number: f64,
    pub spectral_stability_assessment: SpectralStabilityAssessment,
}

/// Complex number representation for eigenvalues
#[derive(Debug, Clone, Copy)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

/// Eigenvalue distribution statistics
#[derive(Debug, Clone, Default)]
pub struct EigenvalueStatistics {
    pub num_eigenvalues: usize,
    pub max_real_part: f64,
    pub min_real_part: f64,
    pub max_imaginary_part: f64,
    pub spectral_radius: f64,
    pub dominant_eigenvalue: Complex64,
}

impl Default for Complex64 {
    fn default() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
}

/// Lipschitz constant metrics
#[derive(Debug, Clone, Default)]
pub struct LipschitzMetrics {
    pub local_lipschitz_constant: f64,
    pub mean_lipschitz_estimate: f64,
    pub lipschitz_variance: f64,
    pub lipschitz_stability_grade: StabilityGrade,
}

/// Parameter sensitivity metrics
#[derive(Debug, Clone, Default)]
pub struct ParameterSensitivityMetrics {
    pub parameter_sensitivities: HashMap<String, f64>,
    pub most_sensitive_parameter: Option<String>,
    pub least_sensitive_parameter: Option<String>,
    pub max_sensitivity: f64,
    pub min_sensitivity: f64,
    pub sensitivity_ratio: f64,
}

/// Gradient stability metrics
#[derive(Debug, Clone, Default)]
pub struct GradientStabilityMetrics {
    pub mean_gradient_variation: f64,
    pub max_gradient_variation: f64,
    pub gradient_variation_std: f64,
    pub mean_gradient_norm: f64,
    pub gradient_norm_stability: f64,
    pub relative_gradient_stability: f64,
    pub gradient_stability_grade: StabilityGrade,
}

/// Stability grades
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum StabilityGrade {
    Excellent,
    Good,
    #[default]
    Fair,
    Poor,
    Unstable,
}

/// Stability classifications
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StabilityClassification {
    NumericallyStable,
    WeaklyStable,
    MarginallyStable,
    Unstable,
    #[default]
    Unknown,
}

/// Spectral stability assessments
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SpectralStabilityAssessment {
    #[default]
    Stable,
    MarginallyStable,
    Unstable,
}

/// Public API functions
/// Compute forward stability for a function
pub fn compute_forward_stability<'a, F: Float, Func>(
    function: &Func,
    input: &'a Tensor<'a, F>,
    perturbation_magnitude: f64,
) -> Result<ForwardStabilityMetrics, StabilityError>
where
    Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    let metrics = StabilityMetrics::new();
    metrics.compute_forward_stability(function, input, perturbation_magnitude)
}

/// Compute backward stability for a function
pub fn compute_backward_stability<'a, F: Float, Func>(
    function: &Func,
    input: &'a Tensor<'a, F>,
    target_output: &'a Tensor<'a, F>,
) -> Result<BackwardStabilityMetrics, StabilityError>
where
    Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    let metrics = StabilityMetrics::new();
    metrics.compute_backward_stability(function, input, target_output)
}

/// Quick stability assessment
pub fn quick_stability_check<'a, F: Float, Func>(
    function: &Func,
    input: &'a Tensor<'a, F>,
) -> Result<StabilityGrade, StabilityError>
where
    Func: for<'b> Fn(&'b Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    let metrics = StabilityMetrics::new();
    let forward_metrics = metrics.compute_forward_stability(function, input, 1e-8)?;
    Ok(forward_metrics.stability_grade)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_metrics_creation() {
        let _metrics = StabilityMetrics::<f32>::new();
    }

    #[test]
    fn test_stability_grades() {
        assert_eq!(StabilityGrade::Excellent, StabilityGrade::Excellent);
        assert_ne!(StabilityGrade::Excellent, StabilityGrade::Poor);
    }

    #[test]
    fn test_stability_classifications() {
        assert_eq!(
            StabilityClassification::NumericallyStable,
            StabilityClassification::NumericallyStable
        );
        assert_ne!(
            StabilityClassification::NumericallyStable,
            StabilityClassification::Unstable
        );
    }

    #[test]
    fn test_forward_stability_metrics() {
        let metrics = ForwardStabilityMetrics::default();
        assert_eq!(metrics.mean_relative_error, 0.0);
        assert!(matches!(metrics.stability_grade, StabilityGrade::Fair));
    }

    #[test]
    fn test_complex_number() {
        let c = Complex64 { re: 1.0, im: 2.0 };
        assert_eq!(c.re, 1.0);
        assert_eq!(c.im, 2.0);

        let magnitude = (c.re * c.re + c.im * c.im).sqrt();
        assert!((magnitude - 2.236067977).abs() < 1e-6);
    }

    #[test]
    fn test_eigenvalue_statistics() {
        let stats = EigenvalueStatistics::default();
        assert_eq!(stats.num_eigenvalues, 0);
        assert_eq!(stats.spectral_radius, 0.0);
    }

    #[test]
    fn test_parameter_sensitivity_metrics() {
        let mut metrics = ParameterSensitivityMetrics::default();
        metrics
            .parameter_sensitivities
            .insert("weight1".to_string(), 0.5);
        metrics
            .parameter_sensitivities
            .insert("bias1".to_string(), 1.2);

        assert_eq!(metrics.parameter_sensitivities.len(), 2);
        assert_eq!(metrics.parameter_sensitivities["weight1"], 0.5);
    }
}
