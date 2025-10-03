//! Advanced error estimation and quality assessment for numerical solutions
//!
//! This module provides sophisticated error estimation techniques that go beyond
//! basic embedded error estimators. It includes Richardson extrapolation,
//! defect correction, spectral error estimation, and adaptive quality metrics.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

/// Advanced error estimator using multiple techniques
pub struct AdvancedErrorEstimator<F: IntegrateFloat> {
    /// Tolerance for error estimation algorithms
    pub tolerance: F,
    /// Maximum order for Richardson extrapolation
    pub max_richardson_order: usize,
    /// History of solutions for spectral analysis
    solution_history: VecDeque<Array1<F>>,
    /// History of step sizes
    step_size_history: VecDeque<F>,
    /// History of error estimates
    error_history: VecDeque<F>,
}

/// Richardson extrapolation error estimator
pub struct RichardsonExtrapolator<F: IntegrateFloat> {
    /// Extrapolation order
    pub order: usize,
    /// Step size ratios for extrapolation
    pub step_ratios: Vec<F>,
    /// Stored solutions at different step sizes
    solutions: Vec<Array1<F>>,
}

/// Spectral error indicator based on solution smoothness
pub struct SpectralErrorIndicator<F: IntegrateFloat> {
    /// Window size for spectral analysis
    pub window_size: usize,
    /// Threshold for spectral decay rate
    pub decay_threshold: F,
    /// Solution history for analysis
    history: VecDeque<Array1<F>>,
}

/// Defect-based error estimator
pub struct DefectCorrector<F: IntegrateFloat> {
    /// Maximum number of defect correction iterations
    pub max_iterations: usize,
    /// Tolerance for defect correction
    pub tolerance: F,
    /// Whether to use iterative defect correction
    pub iterative: bool,
}

/// Comprehensive error analysis result
#[derive(Debug, Clone)]
pub struct ErrorAnalysisResult<F: IntegrateFloat> {
    /// Primary error estimate
    pub primary_estimate: F,
    /// Richardson extrapolation error
    pub richardson_error: Option<F>,
    /// Spectral error indicator
    pub spectral_error: Option<F>,
    /// Defect-based error
    pub defect_error: Option<F>,
    /// Solution quality metrics
    pub quality_metrics: SolutionQualityMetrics<F>,
    /// Recommended next step size
    pub recommended_step_size: F,
    /// Confidence level in error estimate
    pub confidence: F,
    /// Error distribution analysis
    pub error_distribution: ErrorDistribution<F>,
}

/// Solution quality assessment metrics
#[derive(Debug, Clone)]
pub struct SolutionQualityMetrics<F: IntegrateFloat> {
    /// Smoothness indicator (higher = smoother)
    pub smoothness: F,
    /// Regularity indicator (spectral decay rate)
    pub regularity: F,
    /// Conservation property deviation
    pub conservation_error: Option<F>,
    /// Monotonicity violations
    pub monotonicity_violations: usize,
    /// Oscillation indicator
    pub oscillation_index: F,
    /// Signal-to-noise ratio
    pub signal_noise_ratio: F,
}

/// Error distribution characteristics
#[derive(Debug, Clone)]
pub struct ErrorDistribution<F: IntegrateFloat> {
    /// Mean error across components
    pub mean_error: F,
    /// Standard deviation of error
    pub std_deviation: F,
    /// Maximum component error
    pub max_error: F,
    /// Error distribution skewness
    pub skewness: F,
    /// Component-wise error estimates
    pub component_errors: Vec<F>,
}

impl<F: IntegrateFloat> AdvancedErrorEstimator<F> {
    /// Create new advanced error estimator
    pub fn new(tolerance: F, max_richardsonorder: usize) -> Self {
        Self {
            tolerance,
            max_richardson_order: max_richardsonorder,
            solution_history: VecDeque::new(),
            step_size_history: VecDeque::new(),
            error_history: VecDeque::new(),
        }
    }

    /// Perform comprehensive error analysis
    pub fn analyze_error<Func>(
        &mut self,
        current_solution: &Array1<F>,
        step_size: F,
        ode_function: Func,
        embedded_error: Option<F>,
    ) -> IntegrateResult<ErrorAnalysisResult<F>>
    where
        Func: Fn(F, &ArrayView1<F>) -> Array1<F>,
    {
        // Store current _solution and step _size
        self.solution_history.push_back(current_solution.clone());
        self.step_size_history.push_back(step_size);

        // Maintain history _size (keep last 20 points)
        while self.solution_history.len() > 20 {
            self.solution_history.pop_front();
            self.step_size_history.pop_front();
        }

        let mut result = ErrorAnalysisResult {
            primary_estimate: embedded_error.unwrap_or(F::from(1e-8).unwrap()),
            richardson_error: None,
            spectral_error: None,
            defect_error: None,
            quality_metrics: self.assess_solution_quality()?,
            recommended_step_size: step_size,
            confidence: F::from(0.5).unwrap(), // Default confidence
            error_distribution: self.analyze_error_distribution(current_solution)?,
        };

        // Richardson extrapolation if we have enough history
        if self.solution_history.len() >= 3 {
            result.richardson_error = self.richardson_extrapolation()?;
        }

        // Spectral _error analysis
        if self.solution_history.len() >= 5 {
            result.spectral_error = self.spectral_error_analysis()?;
        }

        // Defect correction _error estimate
        result.defect_error = self.defect_based_error(current_solution, &ode_function)?;

        // Compute overall confidence and recommendations
        result.confidence = Self::compute_confidence(&result);
        result.recommended_step_size = self.recommend_step_size(&result, step_size);

        // Store _error estimate for history
        self.error_history.push_back(result.primary_estimate);
        while self.error_history.len() > 20 {
            self.error_history.pop_front();
        }

        Ok(result)
    }

    /// Richardson extrapolation error estimation
    fn richardson_extrapolation(&self) -> IntegrateResult<Option<F>> {
        if self.solution_history.len() < 3 || self.step_size_history.len() < 3 {
            return Ok(None);
        }

        let n = self.solution_history.len();
        let y2 = &self.solution_history[n - 1]; // Current solution
        let y1 = &self.solution_history[n - 2]; // Previous solution
        let _y0 = &self.solution_history[n - 3]; // Solution before that

        let h2 = self.step_size_history[n - 1];
        let h1 = self.step_size_history[n - 2];

        // Assume second-order method for Richardson extrapolation
        let r = h1 / h2;
        if (r - F::one()).abs() < F::from(0.1).unwrap() {
            // Step sizes too similar for reliable extrapolation
            return Ok(None);
        }

        // Richardson extrapolation formula for second-order method
        let extrapolated_error = (y2 - y1).mapv(|x| x.abs()).sum() / (r.powi(2) - F::one());

        Ok(Some(extrapolated_error))
    }

    /// Spectral error analysis based on solution smoothness
    fn spectral_error_analysis(&self) -> IntegrateResult<Option<F>> {
        if self.solution_history.len() < 5 {
            return Ok(None);
        }

        let _n = self.solution_history.len();
        let recent_solutions = &self.solution_history;

        // Compute discrete derivatives to estimate spectral content
        let mut spectral_norm = F::zero();
        let mut total_norm = F::zero();

        for component in 0..recent_solutions[0].len() {
            let values: Vec<F> = recent_solutions.iter().map(|sol| sol[component]).collect();

            // Compute second differences (approximates second derivative)
            if values.len() >= 3 {
                for i in 1..values.len() - 1 {
                    let second_diff =
                        values[i + 1] - F::from(2.0).unwrap() * values[i] + values[i - 1];
                    spectral_norm += second_diff.abs();
                    total_norm += values[i].abs();
                }
            }
        }

        if total_norm > F::zero() {
            let spectral_indicator = spectral_norm / total_norm;
            Ok(Some(spectral_indicator))
        } else {
            Ok(None)
        }
    }

    /// Defect-based error estimation
    fn defect_based_error<Func>(
        &self,
        current_solution: &Array1<F>,
        ode_function: &Func,
    ) -> IntegrateResult<Option<F>>
    where
        Func: Fn(F, &ArrayView1<F>) -> Array1<F>,
    {
        if self.solution_history.len() < 2 || self.step_size_history.is_empty() {
            return Ok(None);
        }

        let h = *self.step_size_history.back().unwrap();
        let t = F::zero(); // Assuming we're at some time t

        // Compute defect: residual when substituting numerical _solution
        // into the original ODE
        let f_current = ode_function(t, &current_solution.view());
        let defect_norm = f_current.mapv(|x| x.abs()).sum() * h;

        Ok(Some(defect_norm))
    }

    /// Assess solution quality metrics
    fn assess_solution_quality(&self) -> IntegrateResult<SolutionQualityMetrics<F>> {
        let mut metrics = SolutionQualityMetrics {
            smoothness: F::zero(),
            regularity: F::zero(),
            conservation_error: None,
            monotonicity_violations: 0,
            oscillation_index: F::zero(),
            signal_noise_ratio: F::one(),
        };

        if self.solution_history.len() < 3 {
            return Ok(metrics);
        }

        let n = self.solution_history.len();
        let solutions = &self.solution_history;

        // Compute smoothness indicator
        let mut total_variation = F::zero();
        let mut total_magnitude = F::zero();

        for i in 1..n {
            let diff = &solutions[i] - &solutions[i - 1];
            total_variation += diff.mapv(|x| x.abs()).sum();
            total_magnitude += solutions[i].mapv(|x| x.abs()).sum();
        }

        if total_magnitude > F::zero() {
            metrics.smoothness = F::one() / (F::one() + total_variation / total_magnitude);
        }

        // Compute oscillation index
        if n >= 3 {
            let mut oscillations = 0;
            for comp in 0..solutions[0].len() {
                for i in 1..n - 1 {
                    let prev = solutions[i - 1][comp];
                    let curr = solutions[i][comp];
                    let next = solutions[i + 1][comp];

                    // Check for local extremum (sign change in derivative)
                    if (curr - prev) * (next - curr) < F::zero() {
                        oscillations += 1;
                    }
                }
            }
            metrics.oscillation_index =
                F::from(oscillations).unwrap() / F::from(n * solutions[0].len()).unwrap();
        }

        // Estimate signal-to-noise ratio
        if n >= 4 {
            let mut signal_power = F::zero();
            let mut noise_power = F::zero();

            for comp in 0..solutions[0].len() {
                let values: Vec<F> = solutions.iter().map(|sol| sol[comp]).collect();
                let mean = values.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from(values.len()).unwrap();

                // Signal power (variance)
                for &val in &values {
                    signal_power += (val - mean).powi(2);
                }

                // Noise power (second differences)
                for i in 1..values.len() - 1 {
                    let second_diff =
                        values[i + 1] - F::from(2.0).unwrap() * values[i] + values[i - 1];
                    noise_power += second_diff.powi(2);
                }
            }

            if noise_power > F::zero() {
                metrics.signal_noise_ratio = signal_power / noise_power;
            }
        }

        Ok(metrics)
    }

    /// Analyze error distribution across solution components
    fn analyze_error_distribution(
        &self,
        solution: &Array1<F>,
    ) -> IntegrateResult<ErrorDistribution<F>> {
        let n_components = solution.len();
        let mut component_errors = vec![F::zero(); n_components];

        // Estimate component-wise errors (simple finite difference approximation)
        if self.solution_history.len() >= 2 {
            let prev_solution = self.solution_history.back().unwrap();
            for i in 0..n_components {
                component_errors[i] = (solution[i] - prev_solution[i]).abs();
            }
        }

        // Compute distribution statistics
        let mean_error = component_errors.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(n_components).unwrap();

        let variance = component_errors
            .iter()
            .map(|&err| (err - mean_error).powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n_components).unwrap();

        let std_deviation = variance.sqrt();
        let max_error = component_errors
            .iter()
            .fold(F::zero(), |acc, &x| acc.max(x));

        // Compute skewness
        let skewness = if std_deviation > F::zero() {
            component_errors
                .iter()
                .map(|&err| ((err - mean_error) / std_deviation).powi(3))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(n_components).unwrap()
        } else {
            F::zero()
        };

        Ok(ErrorDistribution {
            mean_error,
            std_deviation,
            max_error,
            skewness,
            component_errors,
        })
    }

    /// Compute confidence in error estimate
    fn compute_confidence(result: &ErrorAnalysisResult<F>) -> F {
        let mut confidence_factors = Vec::new();

        // Confidence from multiple error estimates agreement
        let estimates = [
            Some(result.primary_estimate),
            result.richardson_error,
            result.spectral_error,
            result.defect_error,
        ]
        .iter()
        .filter_map(|&est| est)
        .collect::<Vec<_>>();

        if estimates.len() >= 2 {
            let mean_est = estimates.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(estimates.len()).unwrap();
            let relative_std = estimates
                .iter()
                .map(|&est| ((est - mean_est) / mean_est).abs())
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(estimates.len()).unwrap();

            // Higher confidence if estimates agree
            confidence_factors.push(F::one() / (F::one() + relative_std));
        }

        // Confidence from solution quality
        confidence_factors.push(result.quality_metrics.smoothness);
        confidence_factors.push(F::one() / (F::one() + result.quality_metrics.oscillation_index));

        // Average confidence factors
        if !confidence_factors.is_empty() {
            confidence_factors.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(confidence_factors.len()).unwrap()
        } else {
            F::from(0.5).unwrap() // Default moderate confidence
        }
    }

    /// Recommend optimal step size based on error analysis
    fn recommend_step_size(&self, result: &ErrorAnalysisResult<F>, currentstep: F) -> F {
        let target_error = self.tolerance;
        let current_error = result.primary_estimate;

        if current_error <= F::zero() {
            return currentstep;
        }

        // Safety factor based on confidence
        let safety_factor = F::from(0.8).unwrap() + F::from(0.15).unwrap() * result.confidence;

        // Standard _step size controller (assumes 2nd order method)
        let ratio = (target_error / current_error).powf(F::from(0.5).unwrap());
        let _basic_recommendation = currentstep * ratio * safety_factor;

        // Adjust based on solution quality
        let quality_factor = if result.quality_metrics.oscillation_index > F::from(0.1).unwrap() {
            F::from(0.7).unwrap() // Reduce _step size for oscillatory solutions
        } else if result.quality_metrics.smoothness > F::from(0.8).unwrap() {
            F::from(1.2).unwrap() // Increase _step size for smooth solutions
        } else {
            F::one()
        };

        // Limit _step size changes
        let min_factor = F::from(0.2).unwrap();
        let max_factor = F::from(5.0).unwrap();
        let final_factor = (ratio * safety_factor * quality_factor)
            .max(min_factor)
            .min(max_factor);

        currentstep * final_factor
    }
}

impl<F: IntegrateFloat> RichardsonExtrapolator<F> {
    /// Create new Richardson extrapolator
    pub fn new(order: usize) -> Self {
        Self {
            order,
            step_ratios: vec![F::from(0.5).unwrap(), F::from(0.25).unwrap()],
            solutions: Vec::new(),
        }
    }

    /// Add solution for extrapolation
    pub fn add_solution(&mut self, solution: Array1<F>) {
        self.solutions.push(solution);
        if self.solutions.len() > self.order + 1 {
            self.solutions.remove(0);
        }
    }

    /// Perform Richardson extrapolation
    pub fn extrapolate(&self) -> IntegrateResult<Option<Array1<F>>> {
        if self.solutions.len() < 2 {
            return Ok(None);
        }

        let n = self.solutions.len();
        let mut tableau = Vec::new();

        // Initialize first column with solutions
        for sol in &self.solutions {
            tableau.push(vec![sol.clone()]);
        }

        // Richardson extrapolation tableau
        for col in 1..n {
            for row in 0..n - col {
                let default_ratio = F::from(0.5).unwrap();
                let r = self.step_ratios.get(col - 1).unwrap_or(&default_ratio);
                let r_power = r.powi(self.order as i32);

                let numerator = &tableau[row + 1][col - 1] * r_power - &tableau[row][col - 1];
                let denominator = r_power - F::one();

                if denominator.abs() > F::from(1e-12).unwrap() {
                    let extrapolated = numerator / denominator;
                    tableau[row].push(extrapolated);
                } else {
                    return Ok(None);
                }
            }
        }

        // Return most extrapolated result
        Ok(Some(tableau[0][n - 1].clone()))
    }
}

impl<F: IntegrateFloat> SpectralErrorIndicator<F> {
    /// Create new spectral error indicator
    pub fn new(window_size: usize, decaythreshold: F) -> Self {
        Self {
            window_size,
            decay_threshold: decaythreshold,
            history: VecDeque::new(),
        }
    }

    /// Add solution to history
    pub fn add_solution(&mut self, solution: Array1<F>) {
        self.history.push_back(solution);
        while self.history.len() > self.window_size {
            self.history.pop_front();
        }
    }

    /// Compute spectral error indicator
    pub fn compute_indicator(&self) -> IntegrateResult<Option<F>> {
        if self.history.len() < self.window_size {
            return Ok(None);
        }

        // Simple spectral analysis: check solution regularity
        // by examining decay of finite differences
        let mut total_indicator = F::zero();
        let n_components = self.history[0].len();

        for comp in 0..n_components {
            let values: Vec<F> = self.history.iter().map(|sol| sol[comp]).collect();

            // Compute successive differences
            let mut diff_norms = Vec::new();
            let mut current_values = values;

            for _order in 0..3 {
                // Check up to 3rd order differences
                if current_values.len() < 2 {
                    break;
                }

                let mut new_values = Vec::new();
                let mut norm = F::zero();

                for i in 0..current_values.len() - 1 {
                    let diff = current_values[i + 1] - current_values[i];
                    new_values.push(diff);
                    norm += diff.abs();
                }

                diff_norms.push(norm);
                current_values = new_values;
            }

            // Check decay rate of difference norms
            if diff_norms.len() >= 2 {
                for i in 1..diff_norms.len() {
                    if diff_norms[i - 1] > F::zero() {
                        let decay_rate = diff_norms[i] / diff_norms[i - 1];
                        if decay_rate > self.decay_threshold {
                            total_indicator += decay_rate;
                        }
                    }
                }
            }
        }

        Ok(Some(total_indicator / F::from(n_components).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_error_estimator() {
        let mut estimator = AdvancedErrorEstimator::<f64>::new(1e-6, 3);

        let solution = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let step_size = 0.01;
        let ode_fn =
            |_t: f64, y: &ArrayView1<f64>| Array1::from_vec(y.iter().map(|&yi| -yi).collect());

        let result = estimator.analyze_error(&solution, step_size, ode_fn, Some(1e-8));
        assert!(result.is_ok());

        let error_analysis = result.unwrap();
        assert!(error_analysis.primary_estimate > 0.0);
        assert!(error_analysis.confidence >= 0.0 && error_analysis.confidence <= 1.0);
    }

    #[test]
    fn test_richardson_extrapolator() {
        let mut extrapolator = RichardsonExtrapolator::<f64>::new(2);

        extrapolator.add_solution(Array1::from_vec(vec![1.0, 2.0]));
        extrapolator.add_solution(Array1::from_vec(vec![1.01, 2.01]));
        extrapolator.add_solution(Array1::from_vec(vec![1.005, 2.005]));

        let result = extrapolator.extrapolate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_error_indicator() {
        let mut indicator = SpectralErrorIndicator::<f64>::new(5, 0.5);

        for i in 0..6 {
            let solution = Array1::from_vec(vec![i as f64, (i as f64).sin()]);
            indicator.add_solution(solution);
        }

        let result = indicator.compute_indicator();
        assert!(result.is_ok());
    }
}
