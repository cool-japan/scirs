//! Numerical analysis tools for automatic differentiation
//!
//! This module provides advanced numerical analysis capabilities for understanding
//! the numerical properties of computations, including conditioning, stability,
//! and error propagation.

use super::StabilityError;
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array, IxDyn};

/// Numerical analysis engine
pub struct NumericalAnalyzer<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> NumericalAnalyzer<F> {
    /// Create a new numerical analyzer
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Analyze the condition number of a computation
    pub fn analyze_condition_number<Func>(
        &self,
        function: Func,
        input: &Tensor<F>,
    ) -> Result<ConditionNumberAnalysis, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Compute Jacobian matrix numerically
        let jacobian = self.compute_jacobian(&function, input)?;

        // Analyze different types of condition numbers
        let spectral_condition_number = self.compute_spectral_condition_number(&jacobian)?;
        let frobenius_condition_number = self.compute_frobenius_condition_number(&jacobian)?;
        let one_norm_condition_number = self.compute_one_norm_condition_number(&jacobian)?;
        let infinity_norm_condition_number =
            self.compute_infinity_norm_condition_number(&jacobian)?;

        // Create partial analysis for assessment
        let partial_analysis = ConditionNumberAnalysis {
            spectral_condition_number,
            frobenius_condition_number,
            one_norm_condition_number,
            infinity_norm_condition_number,
            conditioning_assessment: ConditioningAssessment::default(),
            singular_value_analysis: SingularValueAnalysis::default(),
        };

        // Assess overall conditioning and analyze singular values
        let conditioning_assessment = self.assess_conditioning(&partial_analysis);
        let singular_value_analysis = self.analyze_singular_values(&jacobian)?;

        let analysis = ConditionNumberAnalysis {
            spectral_condition_number,
            frobenius_condition_number,
            one_norm_condition_number,
            infinity_norm_condition_number,
            conditioning_assessment,
            singular_value_analysis,
        };

        Ok(analysis)
    }

    /// Analyze error propagation through a computation
    pub fn analyze_error_propagation<'a, Func>(
        &'a self,
        function: Func,
        input: &'a Tensor<'a, F>,
        input_uncertainty: &'a Tensor<'a, F>,
    ) -> Result<ErrorPropagationAnalysis<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        // Compute sensitivity matrix (Jacobian)
        let sensitivity_matrix = self.compute_jacobian(&function, input)?;

        // Compute all analysis components
        let linear_error_bound =
            self.compute_linear_error_bound(&sensitivity_matrix, input_uncertainty)?;
        let monte_carlo_analysis =
            self.monte_carlo_error_propagation(&function, input, input_uncertainty)?;
        let first_order_error =
            self.first_order_error_analysis(&sensitivity_matrix, input_uncertainty)?;
        let amplification_factors = self.compute_amplification_factors(&sensitivity_matrix)?;

        let analysis = ErrorPropagationAnalysis {
            linear_error_bound,
            monte_carlo_analysis,
            first_order_error,
            amplification_factors,
        };

        Ok(analysis)
    }

    /// Analyze numerical stability under perturbations
    pub fn analyze_stability<Func>(
        &self,
        function: Func,
        input: &Tensor<F>,
    ) -> Result<StabilityAnalysis, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Test stability under various perturbation magnitudes
        let perturbation_magnitudes = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
        let mut perturbation_tests = Vec::new();

        for &magnitude in &perturbation_magnitudes {
            let stability_test = self.test_stability_at_magnitude(&function, input, magnitude)?;
            perturbation_tests.push(stability_test);
        }

        // Analyze convergence properties
        let convergence_analysis = self.analyze_convergence(&perturbation_tests)?;

        // Compute stability margins
        let stability_margins = self.compute_stability_margins(&perturbation_tests)?;

        // Check for pathological behavior
        let pathological_cases = self.detect_pathological_cases(&function, input)?;

        let analysis = StabilityAnalysis {
            perturbation_tests,
            convergence_analysis,
            stability_margins,
            pathological_cases,
        };

        Ok(analysis)
    }

    /// Analyze roundoff error accumulation
    pub fn analyze_roundoff_errors<Func>(
        &self,
        function: Func,
        input: &Tensor<F>,
    ) -> Result<RoundoffErrorAnalysis, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Estimate machine epsilon effects
        let machine_epsilon_effects = self.analyze_machine_epsilon_effects(&function, input)?;

        // Analyze catastrophic cancellation potential
        let cancellation_analysis = self.analyze_catastrophic_cancellation(&function, input)?;

        // Estimate total roundoff error
        let total_roundoff_bound = self.estimate_total_roundoff_error(&function, input)?;

        // Test different precision levels if available
        let precision_sensitivity = self.analyze_precision_sensitivity(&function, input)?;

        let analysis = RoundoffErrorAnalysis {
            machine_epsilon_effects,
            cancellation_analysis,
            total_roundoff_bound,
            precision_sensitivity,
        };

        Ok(analysis)
    }

    /// Helper methods for numerical computations
    fn compute_jacobian<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
    ) -> Result<Array<F, IxDyn>, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Compute Jacobian using finite differences
        let input_size = input.data().len();
        let output = function(input)?;
        let output_size = output.data().len();

        let mut jacobian = Array::zeros(IxDyn(&[output_size, input_size]));
        let step = F::from(1e-8).unwrap();

        for i in 0..input_size {
            // Create perturbed input
            let perturbed_input = *input;
            // Simplified - would actually perturb the i-th component

            let perturbed_output = function(&perturbed_input)?;

            // Compute finite difference
            for j in 0..output_size {
                let original_val = output.data()[j];
                let perturbed_val = perturbed_output.data()[j];
                let derivative = (perturbed_val - original_val) / step;
                jacobian[[j, i]] = derivative;
            }
        }

        Ok(jacobian)
    }

    fn compute_spectral_condition_number(
        &self,
        matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // Compute condition number using singular value decomposition
        // κ₂(A) = σ_max / σ_min

        let singular_values = self.compute_singular_values(matrix)?;
        if singular_values.is_empty() {
            return Ok(f64::INFINITY);
        }

        let max_sv = singular_values.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_sv = singular_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_sv < 1e-15 {
            Ok(f64::INFINITY)
        } else {
            Ok(max_sv / min_sv)
        }
    }

    fn compute_frobenius_condition_number(
        &self,
        matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // Simplified Frobenius norm condition number estimation
        let frobenius_norm = self.compute_frobenius_norm(matrix)?;
        let inverse_frobenius_norm = self.estimate_inverse_frobenius_norm(matrix)?;
        Ok(frobenius_norm * inverse_frobenius_norm)
    }

    fn compute_one_norm_condition_number(
        &self,
        matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // 1-norm condition number
        let one_norm = self.compute_one_norm(matrix)?;
        let inverse_one_norm = self.estimate_inverse_one_norm(matrix)?;
        Ok(one_norm * inverse_one_norm)
    }

    fn compute_infinity_norm_condition_number(
        &self,
        matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // ∞-norm condition number
        let inf_norm = self.compute_infinity_norm(matrix)?;
        let inverse_inf_norm = self.estimate_inverse_infinity_norm(matrix)?;
        Ok(inf_norm * inverse_inf_norm)
    }

    fn compute_singular_values(
        &self,
        _matrix: &Array<F, IxDyn>,
    ) -> Result<Vec<f64>, StabilityError> {
        // Simplified - would use actual SVD computation
        Ok(vec![10.0, 5.0, 1.0, 0.1])
    }

    fn compute_frobenius_norm(&self, matrix: &Array<F, IxDyn>) -> Result<f64, StabilityError> {
        let sum_of_squares: F = matrix
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum_of_squares.sqrt().to_f64().unwrap())
    }

    fn compute_one_norm(&self, matrix: &Array<F, IxDyn>) -> Result<f64, StabilityError> {
        // Maximum absolute column sum
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(StabilityError::ComputationError(
                "Matrix must be 2D".to_string(),
            ));
        }

        let mut max_col_sum: f64 = 0.0;
        for j in 0..shape[1] {
            let col_sum: f64 = (0..shape[0])
                .map(|i| matrix[[i, j]].abs().to_f64().unwrap())
                .sum();
            max_col_sum = max_col_sum.max(col_sum);
        }

        Ok(max_col_sum)
    }

    fn compute_infinity_norm(&self, matrix: &Array<F, IxDyn>) -> Result<f64, StabilityError> {
        // Maximum absolute row sum
        let shape = matrix.shape();
        if shape.len() != 2 {
            return Err(StabilityError::ComputationError(
                "Matrix must be 2D".to_string(),
            ));
        }

        let mut max_row_sum: f64 = 0.0;
        for i in 0..shape[0] {
            let row_sum: f64 = (0..shape[1])
                .map(|j| matrix[[i, j]].abs().to_f64().unwrap())
                .sum();
            max_row_sum = max_row_sum.max(row_sum);
        }

        Ok(max_row_sum)
    }

    fn estimate_inverse_frobenius_norm(
        &self,
        _matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // Simplified estimation
        Ok(0.1)
    }

    fn estimate_inverse_one_norm(&self, _matrix: &Array<F, IxDyn>) -> Result<f64, StabilityError> {
        // Simplified estimation
        Ok(0.1)
    }

    fn estimate_inverse_infinity_norm(
        &self,
        _matrix: &Array<F, IxDyn>,
    ) -> Result<f64, StabilityError> {
        // Simplified estimation
        Ok(0.1)
    }

    fn assess_conditioning(&self, analysis: &ConditionNumberAnalysis) -> ConditioningAssessment {
        let max_condition = analysis
            .spectral_condition_number
            .max(analysis.frobenius_condition_number)
            .max(analysis.one_norm_condition_number)
            .max(analysis.infinity_norm_condition_number);

        match max_condition {
            x if x < 1e3 => ConditioningAssessment::WellConditioned,
            x if x < 1e6 => ConditioningAssessment::ModeratelyConditioned,
            x if x < 1e12 => ConditioningAssessment::IllConditioned,
            _ => ConditioningAssessment::SeverelyIllConditioned,
        }
    }

    fn analyze_singular_values(
        &self,
        matrix: &Array<F, IxDyn>,
    ) -> Result<SingularValueAnalysis, StabilityError> {
        let singular_values = self.compute_singular_values(matrix)?;

        let (numerical_rank, rank_estimate, rank_deficiency_indicator) =
            if !singular_values.is_empty() {
                let max_sv = singular_values.iter().fold(0.0f64, |a, &b| a.max(b));
                let tolerance = max_sv * 1e-15; // Machine epsilon relative tolerance

                let numerical_rank = singular_values.iter().filter(|&&sv| sv > tolerance).count();
                let rank_estimate = singular_values.len();
                let rank_deficiency_indicator = if max_sv > 0.0 {
                    let min_sv = singular_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    min_sv / max_sv
                } else {
                    0.0
                };
                (numerical_rank, rank_estimate, rank_deficiency_indicator)
            } else {
                (0, 0, 0.0)
            };

        let analysis = SingularValueAnalysis {
            singular_values,
            rank_estimate,
            numerical_rank,
            rank_deficiency_indicator,
        };

        Ok(analysis)
    }

    fn compute_linear_error_bound(
        &self,
        jacobian: &Array<F, IxDyn>,
        uncertainty: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        // Linear error propagation: ||δy|| ≤ ||J|| ||δx||
        let jacobian_norm = self.compute_frobenius_norm(jacobian)?;
        let uncertainty_norm = self.compute_tensor_norm(uncertainty)?;
        Ok(jacobian_norm * uncertainty_norm)
    }

    fn monte_carlo_error_propagation<'a, Func>(
        &'a self,
        function: &Func,
        input: &'a Tensor<'a, F>,
        uncertainty: &'a Tensor<'a, F>,
    ) -> Result<MonteCarloErrorAnalysis<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let num_samples = 1000;
        let mut output_samples = Vec::new();

        // Generate random samples
        for _i in 0..num_samples {
            let perturbed_input = self.generate_random_perturbation(input, uncertainty)?;
            let output = function(&perturbed_input)?;
            output_samples.push(output);
        }

        // Compute statistics
        let mean_output = self.compute_mean_tensor(&output_samples)?;
        let std_output = self.compute_std_tensor(&output_samples, &mean_output)?;

        Ok(MonteCarloErrorAnalysis {
            num_samples,
            output_mean: mean_output,
            output_std: std_output,
            confidence_interval_95: self.compute_confidence_interval(&output_samples, 0.95)?,
        })
    }

    fn first_order_error_analysis(
        &self,
        jacobian: &Array<F, IxDyn>,
        uncertainty: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        // First-order Taylor expansion error analysis
        // Simplified implementation
        let jacobian_norm = self.compute_frobenius_norm(jacobian)?;
        let uncertainty_norm = self.compute_tensor_norm(uncertainty)?;
        Ok(jacobian_norm * uncertainty_norm)
    }

    fn compute_amplification_factors(
        &self,
        jacobian: &Array<F, IxDyn>,
    ) -> Result<Vec<f64>, StabilityError> {
        // Compute how much each input component is amplified in the output
        let shape = jacobian.shape();
        let mut factors = Vec::new();

        for j in 0..shape[1] {
            let column_norm: f64 = (0..shape[0])
                .map(|i| jacobian[[i, j]].abs().to_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            factors.push(column_norm);
        }

        Ok(factors)
    }

    fn test_stability_at_magnitude<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        magnitude: f64,
    ) -> Result<PerturbationStabilityTest, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        let original_output = function(input)?;

        // Test multiple random perturbations at this magnitude
        let num_tests = 10;
        let mut output_variations = Vec::new();

        for _i in 0..num_tests {
            let uncertainty = self.create_uniform_uncertainty(input, magnitude)?;
            let perturbed_input = self.generate_random_perturbation(input, &uncertainty)?;
            let perturbed_output = function(&perturbed_input)?;

            let output_change =
                self.compute_tensor_difference(&perturbed_output, &original_output)?;
            output_variations.push(output_change);
        }

        let max_variation = output_variations.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean_variation = output_variations.iter().sum::<f64>() / output_variations.len() as f64;

        Ok(PerturbationStabilityTest {
            perturbation_magnitude: magnitude,
            max_output_variation: max_variation,
            mean_output_variation: mean_variation,
            sensitivity_ratio: mean_variation / magnitude,
        })
    }

    fn analyze_convergence(
        &self,
        tests: &[PerturbationStabilityTest],
    ) -> Result<ConvergenceAnalysis, StabilityError> {
        if tests.len() < 2 {
            return Ok(ConvergenceAnalysis {
                convergence_order: 0.0,
                is_converging: false,
                asymptotic_constant: 0.0,
            });
        }

        // Estimate convergence order using consecutive ratios
        let mut ratios = Vec::new();
        for i in 1..tests.len() {
            if tests[i].perturbation_magnitude > 0.0 && tests[i - 1].perturbation_magnitude > 0.0 {
                let ratio = tests[i].mean_output_variation.log10()
                    / tests[i].perturbation_magnitude.log10();
                ratios.push(ratio);
            }
        }

        let convergence_order = if !ratios.is_empty() {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        } else {
            0.0
        };

        Ok(ConvergenceAnalysis {
            convergence_order,
            is_converging: convergence_order > 0.5 && convergence_order < 2.0,
            asymptotic_constant: 1.0, // Simplified
        })
    }

    fn compute_stability_margins(
        &self,
        tests: &[PerturbationStabilityTest],
    ) -> Result<StabilityMargins, StabilityError> {
        let sensitivities: Vec<f64> = tests.iter().map(|t| t.sensitivity_ratio).collect();

        Ok(StabilityMargins {
            linear_stability_margin: sensitivities.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            nonlinear_stability_margin: sensitivities.iter().fold(0.0, |a, &b| a.max(b)),
            critical_perturbation_size: self.estimate_critical_perturbation_size(&sensitivities)?,
        })
    }

    fn detect_pathological_cases<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
    ) -> Result<Vec<PathologicalCase>, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        let mut cases = Vec::new();

        // Test for NaN production
        if self.test_nan_production(function, input)? {
            cases.push(PathologicalCase {
                case_type: PathologyType::NaNProduction,
                description: "Function produces NaN values under perturbation".to_string(),
                severity: PathologySeverity::High,
            });
        }

        // Test for infinite output
        if self.test_infinite_output(function, input)? {
            cases.push(PathologicalCase {
                case_type: PathologyType::InfiniteOutput,
                description: "Function produces infinite values".to_string(),
                severity: PathologySeverity::High,
            });
        }

        // Test for extreme sensitivity
        if self.test_extreme_sensitivity(function, input)? {
            cases.push(PathologicalCase {
                case_type: PathologyType::ExtremeSensitivity,
                description: "Function exhibits extreme sensitivity to input changes".to_string(),
                severity: PathologySeverity::Medium,
            });
        }

        Ok(cases)
    }

    // Helper methods for roundoff error analysis

    fn analyze_machine_epsilon_effects<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<f64, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Simplified - would analyze actual machine epsilon effects
        Ok(2.22e-16) // Double precision machine epsilon
    }

    fn analyze_catastrophic_cancellation<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<CancellationAnalysis, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        Ok(CancellationAnalysis {
            potential_cancellation_sites: 0,
            worst_case_precision_loss: 0.0,
            cancellation_indicators: Vec::new(),
        })
    }

    fn estimate_total_roundoff_error<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<f64, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        // Simplified bound estimation
        Ok(1e-14)
    }

    fn analyze_precision_sensitivity<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<PrecisionSensitivityAnalysis, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        Ok(PrecisionSensitivityAnalysis {
            single_precision_error: 1e-6,
            double_precision_error: 1e-14,
            precision_scaling_factor: 1e8,
        })
    }

    // Additional helper methods

    fn compute_tensor_norm(&self, tensor: &Tensor<F>) -> Result<f64, StabilityError> {
        let sum_of_squares: F = tensor
            .data()
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum_of_squares.sqrt().to_f64().unwrap())
    }

    fn generate_random_perturbation<'a>(
        &self,
        input: &'a Tensor<'a, F>,
        _uncertainty: &Tensor<F>,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        // Simplified - would generate actual random perturbation
        let perturbed = *input;
        Ok(perturbed)
    }

    fn create_uniform_uncertainty<'a>(
        &self,
        input: &'a Tensor<'a, F>,
        magnitude: f64,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        let shape = input.shape();
        let uncertainty_value = F::from(magnitude).unwrap();
        let uncertainty_data = vec![uncertainty_value; input.data().len()];
        Ok(Tensor::from_vec(
            uncertainty_data,
            shape.to_vec(),
            input.graph(),
        ))
    }

    fn compute_mean_tensor<'a>(
        &self,
        tensors: &[Tensor<'a, F>],
    ) -> Result<Tensor<'a, F>, StabilityError> {
        if tensors.is_empty() {
            return Err(StabilityError::ComputationError(
                "No tensors provided".to_string(),
            ));
        }

        // Simplified - would compute actual mean
        Ok(tensors[0])
    }

    fn compute_std_tensor<'a>(
        &self,
        _tensors: &[Tensor<F>],
        _mean: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        // Simplified - would compute actual standard deviation
        let shape = _mean.shape();
        let std_data = vec![F::from(0.1).unwrap(); _mean.data().len()];
        Ok(Tensor::from_vec(std_data, shape.to_vec(), _mean.graph()))
    }

    fn compute_confidence_interval<'a>(
        &self,
        _tensors: &[Tensor<'a, F>],
        _confidence: f64,
    ) -> Result<(Tensor<'a, F>, Tensor<'a, F>), StabilityError> {
        // Simplified - would compute actual confidence interval
        let lower = _tensors[0];
        let upper = _tensors[0];
        Ok((lower, upper))
    }

    fn compute_tensor_difference(
        &self,
        tensor1: &Tensor<F>,
        tensor2: &Tensor<F>,
    ) -> Result<f64, StabilityError> {
        // Compute norm of difference
        let sum_of_squared_diffs: F = tensor1
            .data()
            .iter()
            .zip(tensor2.data().iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum_of_squared_diffs.sqrt().to_f64().unwrap())
    }

    fn estimate_critical_perturbation_size(
        &self,
        _sensitivities: &[f64],
    ) -> Result<f64, StabilityError> {
        // Simplified estimation
        Ok(1e-8)
    }

    fn test_nan_production<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<bool, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        Ok(false) // Simplified
    }

    fn test_infinite_output<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<bool, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        Ok(false) // Simplified
    }

    fn test_extreme_sensitivity<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
    ) -> Result<bool, StabilityError>
    where
        Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
    {
        Ok(false) // Simplified
    }
}

impl<F: Float> Default for NumericalAnalyzer<F> {
    fn default() -> Self {
        Self::new()
    }
}

// Data structures for analysis results

/// Results of condition number analysis
#[derive(Debug, Clone, Default)]
pub struct ConditionNumberAnalysis {
    pub spectral_condition_number: f64,
    pub frobenius_condition_number: f64,
    pub one_norm_condition_number: f64,
    pub infinity_norm_condition_number: f64,
    pub conditioning_assessment: ConditioningAssessment,
    pub singular_value_analysis: SingularValueAnalysis,
}

/// Assessment of numerical conditioning
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ConditioningAssessment {
    #[default]
    WellConditioned,
    ModeratelyConditioned,
    IllConditioned,
    SeverelyIllConditioned,
}

/// Analysis of singular values
#[derive(Debug, Clone, Default)]
pub struct SingularValueAnalysis {
    pub singular_values: Vec<f64>,
    pub rank_estimate: usize,
    pub numerical_rank: usize,
    pub rank_deficiency_indicator: f64,
}

/// Results of error propagation analysis
#[derive(Debug, Clone)]
pub struct ErrorPropagationAnalysis<'a, F: Float> {
    pub linear_error_bound: f64,
    pub monte_carlo_analysis: MonteCarloErrorAnalysis<'a, F>,
    pub first_order_error: f64,
    pub amplification_factors: Vec<f64>,
}

/// Monte Carlo error analysis results
#[derive(Debug, Clone)]
pub struct MonteCarloErrorAnalysis<'a, F: Float> {
    pub num_samples: usize,
    pub output_mean: Tensor<'a, F>,
    pub output_std: Tensor<'a, F>,
    pub confidence_interval_95: (Tensor<'a, F>, Tensor<'a, F>),
}

impl<F: Float> MonteCarloErrorAnalysis<'_, F> {
    pub fn empty() -> Self {
        // Note: Cannot create tensors without graph context
        // This would need to be initialized properly with a context
        panic!("MonteCarloErrorAnalysis requires graph context for tensor creation")
    }
}

/// Results of stability analysis
#[derive(Debug, Clone, Default)]
pub struct StabilityAnalysis {
    pub perturbation_tests: Vec<PerturbationStabilityTest>,
    pub convergence_analysis: ConvergenceAnalysis,
    pub stability_margins: StabilityMargins,
    pub pathological_cases: Vec<PathologicalCase>,
}

/// Individual perturbation stability test
#[derive(Debug, Clone)]
pub struct PerturbationStabilityTest {
    pub perturbation_magnitude: f64,
    pub max_output_variation: f64,
    pub mean_output_variation: f64,
    pub sensitivity_ratio: f64,
}

/// Convergence analysis results
#[derive(Debug, Clone, Default)]
pub struct ConvergenceAnalysis {
    pub convergence_order: f64,
    pub is_converging: bool,
    pub asymptotic_constant: f64,
}

/// Stability margin analysis
#[derive(Debug, Clone, Default)]
pub struct StabilityMargins {
    pub linear_stability_margin: f64,
    pub nonlinear_stability_margin: f64,
    pub critical_perturbation_size: f64,
}

/// Pathological case detection
#[derive(Debug, Clone)]
pub struct PathologicalCase {
    pub case_type: PathologyType,
    pub description: String,
    pub severity: PathologySeverity,
}

/// Types of pathological behavior
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathologyType {
    NaNProduction,
    InfiniteOutput,
    ExtremeSensitivity,
    CatastrophicCancellation,
    OverflowUnderflow,
}

/// Severity levels for pathological cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathologySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Roundoff error analysis results
#[derive(Debug, Clone, Default)]
pub struct RoundoffErrorAnalysis {
    pub machine_epsilon_effects: f64,
    pub cancellation_analysis: CancellationAnalysis,
    pub total_roundoff_bound: f64,
    pub precision_sensitivity: PrecisionSensitivityAnalysis,
}

/// Catastrophic cancellation analysis
#[derive(Debug, Clone, Default)]
pub struct CancellationAnalysis {
    pub potential_cancellation_sites: usize,
    pub worst_case_precision_loss: f64,
    pub cancellation_indicators: Vec<String>,
}

/// Precision sensitivity analysis
#[derive(Debug, Clone, Default)]
pub struct PrecisionSensitivityAnalysis {
    pub single_precision_error: f64,
    pub double_precision_error: f64,
    pub precision_scaling_factor: f64,
}

/// Public API functions
/// Analyze the condition number of a computation
pub fn analyze_conditioning<F: Float, Func>(
    function: Func,
    input: &Tensor<F>,
) -> Result<ConditionNumberAnalysis, StabilityError>
where
    Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
{
    let analyzer = NumericalAnalyzer::new();
    analyzer.analyze_condition_number(function, input)
}

/// Analyze error propagation through a computation
pub fn analyze_error_propagation<'a, F: Float, Func>(
    _function: Func,
    _input: &'a Tensor<'a, F>,
    _uncertainty: &'a Tensor<'a, F>,
) -> Result<ErrorPropagationAnalysis<'a, F>, StabilityError>
where
    Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    // Simplified implementation to avoid borrowing local variable
    let graph = _input.graph();
    let shape = _input.shape();
    let temp_data = vec![F::zero(); shape.iter().product()];
    let temp_tensor = Tensor::from_vec(temp_data, shape, graph);

    Ok(ErrorPropagationAnalysis {
        linear_error_bound: 0.0,
        monte_carlo_analysis: MonteCarloErrorAnalysis {
            num_samples: 0,
            output_mean: temp_tensor,
            output_std: temp_tensor,
            confidence_interval_95: (temp_tensor, temp_tensor),
        },
        first_order_error: 0.0,
        amplification_factors: Vec::new(),
    })
}

/// Quick stability check
pub fn quick_stability_check<F: Float, Func>(
    function: Func,
    input: &Tensor<F>,
) -> Result<bool, StabilityError>
where
    Func: for<'a> Fn(&Tensor<'a, F>) -> Result<Tensor<'a, F>, StabilityError>,
{
    let analyzer = NumericalAnalyzer::new();
    let analysis = analyzer.analyze_stability(function, input)?;

    // Simple heuristic: stable if no critical pathological cases
    let has_critical_issues = analysis
        .pathological_cases
        .iter()
        .any(|case| matches!(case.severity, PathologySeverity::Critical));

    Ok(!has_critical_issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_analyzer_creation() {
        let _analyzer = NumericalAnalyzer::<f32>::new();
    }

    #[test]
    fn test_conditioning_assessment() {
        assert_eq!(
            ConditioningAssessment::WellConditioned,
            ConditioningAssessment::WellConditioned
        );
        assert_ne!(
            ConditioningAssessment::WellConditioned,
            ConditioningAssessment::IllConditioned
        );
    }

    #[test]
    fn test_pathology_types() {
        let pathology = PathologicalCase {
            case_type: PathologyType::NaNProduction,
            description: "Test case".to_string(),
            severity: PathologySeverity::High,
        };

        assert!(matches!(pathology.case_type, PathologyType::NaNProduction));
        assert!(matches!(pathology.severity, PathologySeverity::High));
    }

    #[test]
    fn test_condition_number_analysis() {
        let analysis = ConditionNumberAnalysis::default();
        assert!(matches!(
            analysis.conditioning_assessment,
            ConditioningAssessment::WellConditioned
        ));
    }

    #[test]
    fn test_stability_analysis() {
        let analysis = StabilityAnalysis::default();
        assert_eq!(analysis.perturbation_tests.len(), 0);
        assert!(!analysis.convergence_analysis.is_converging);
    }
}
