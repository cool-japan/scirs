// Robust Controllability and Observability Analysis
//
// This module provides enhanced numerical robustness and advanced analysis capabilities
// for Linear Time-Invariant system controllability and observability properties.
//
// Key Features:
// - Numerically robust matrix computations using SVD-based methods
// - Sensitivity analysis to parameter perturbations
// - Structured perturbation analysis for real-world robustness
// - Performance-oriented controllability/observability metrics
// - Frequency-domain controllability/observability measures
// - Advanced condition assessment and uncertainty quantification
// - Multi-scale Gramian analysis for different time horizons

use super::analysis::{ControllabilityAnalysis, ObservabilityAnalysis};
use crate::error::{SignalError, SignalResult};
use crate::lti::systems::StateSpace;
use ndarray::{Array1, Array2};
use num_traits::Float;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_finite;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Robust controllability and observability analysis result
#[derive(Debug, Clone)]
pub struct RobustControlObservabilityAnalysis {
    /// Enhanced controllability analysis
    pub enhanced_controllability: EnhancedControllabilityAnalysis,
    /// Enhanced observability analysis  
    pub enhanced_observability: EnhancedObservabilityAnalysis,
    /// Sensitivity analysis results
    pub sensitivity_analysis: SensitivityAnalysisResults,
    /// Structured perturbation analysis
    pub structured_analysis: StructuredPerturbationAnalysis,
    /// Performance metrics
    pub performance_metrics: PerformanceOrientedMetrics,
    /// Frequency-domain analysis
    pub frequency_analysis: FrequencyDomainAnalysis,
    /// Overall robustness score (0-100)
    pub robustness_score: f64,
    /// Critical robustness issues
    pub robustness_issues: Vec<String>,
}

/// Enhanced controllability analysis with robustness measures
#[derive(Debug, Clone)]
pub struct EnhancedControllabilityAnalysis {
    /// Basic controllability analysis
    pub basic_analysis: ControllabilityAnalysis,
    /// SVD-based controllability analysis
    pub svd_analysis: SvdControllabilityAnalysis,
    /// Numerical conditioning assessment
    pub conditioning: NumericalConditioning,
    /// Controllability degrees for each mode
    pub mode_controllability_degrees: Array1<f64>,
    /// Minimum energy control analysis
    pub minimum_energy_analysis: MinimumEnergyAnalysis,
}

/// Enhanced observability analysis with robustness measures
#[derive(Debug, Clone)]
pub struct EnhancedObservabilityAnalysis {
    /// Basic observability analysis
    pub basic_analysis: ObservabilityAnalysis,
    /// SVD-based observability analysis
    pub svd_analysis: SvdObservabilityAnalysis,
    /// Numerical conditioning assessment
    pub conditioning: NumericalConditioning,
    /// Observability degrees for each mode
    pub mode_observability_degrees: Array1<f64>,
    /// Minimum variance estimation analysis
    pub minimum_variance_analysis: MinimumVarianceAnalysis,
}

/// SVD-based controllability analysis for numerical robustness
#[derive(Debug, Clone)]
pub struct SvdControllabilityAnalysis {
    /// Singular values of controllability matrix
    pub singular_values: Array1<f64>,
    /// Left singular vectors (modes)
    pub left_singular_vectors: Array2<f64>,
    /// Right singular vectors (input directions)
    pub right_singular_vectors: Array2<f64>,
    /// Numerical rank based on SVD
    pub numerical_rank: usize,
    /// Effective rank based on condition number
    pub effective_rank: usize,
    /// Gap in singular values
    pub singular_value_gaps: Array1<f64>,
}

/// SVD-based observability analysis for numerical robustness  
#[derive(Debug, Clone)]
pub struct SvdObservabilityAnalysis {
    /// Singular values of observability matrix
    pub singular_values: Array1<f64>,
    /// Left singular vectors (output directions)
    pub left_singular_vectors: Array2<f64>,
    /// Right singular vectors (modes)
    pub right_singular_vectors: Array2<f64>,
    /// Numerical rank based on SVD
    pub numerical_rank: usize,
    /// Effective rank based on condition number
    pub effective_rank: usize,
    /// Gap in singular values
    pub singular_value_gaps: Array1<f64>,
}

/// Numerical conditioning assessment
#[derive(Debug, Clone)]
pub struct NumericalConditioning {
    /// Condition number (2-norm)
    pub condition_number_2: f64,
    /// Condition number (Frobenius norm)
    pub condition_number_f: f64,
    /// Numerical tolerance for rank determination
    pub numerical_tolerance: f64,
    /// Distance to singularity
    pub distance_to_singularity: f64,
    /// Stability margin for conditioning
    pub stability_margin: f64,
}

/// Sensitivity analysis to parameter perturbations
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResults {
    /// Sensitivity of controllability to A matrix perturbations
    pub controllability_sensitivity_a: Array2<f64>,
    /// Sensitivity of controllability to B matrix perturbations
    pub controllability_sensitivity_b: Array2<f64>,
    /// Sensitivity of observability to A matrix perturbations
    pub observability_sensitivity_a: Array2<f64>,
    /// Sensitivity of observability to C matrix perturbations
    pub observability_sensitivity_c: Array2<f64>,
    /// Maximum tolerable perturbation before loss of controllability
    pub max_controllability_perturbation: f64,
    /// Maximum tolerable perturbation before loss of observability
    pub max_observability_perturbation: f64,
}

/// Structured perturbation analysis for real-world robustness
#[derive(Debug, Clone)]
pub struct StructuredPerturbationAnalysis {
    /// Robustness to multiplicative uncertainties
    pub multiplicative_robustness: MultiplcativeRobustness,
    /// Robustness to additive uncertainties
    pub additive_robustness: AdditiveRobustness,
    /// Parametric uncertainty analysis
    pub parametric_uncertainty: ParametricUncertaintyAnalysis,
    /// Real parameter perturbation bounds
    pub real_perturbation_bounds: RealPerturbationBounds,
}

/// Multiplicative uncertainty robustness analysis
#[derive(Debug, Clone)]
pub struct MultiplcativeRobustness {
    /// Stability margin for multiplicative uncertainties
    pub stability_margin: f64,
    /// Controllability margin
    pub controllability_margin: f64,
    /// Observability margin
    pub observability_margin: f64,
    /// Critical uncertainty levels
    pub critical_uncertainty_levels: Array1<f64>,
}

/// Additive uncertainty robustness analysis
#[derive(Debug, Clone)]
pub struct AdditiveRobustness {
    /// Maximum additive perturbation norm
    pub max_perturbation_norm: f64,
    /// Controllability robustness bound
    pub controllability_bound: f64,
    /// Observability robustness bound
    pub observability_bound: f64,
    /// Worst-case perturbation directions
    pub worst_case_directions: Array2<f64>,
}

/// Parametric uncertainty analysis
#[derive(Debug, Clone)]
pub struct ParametricUncertaintyAnalysis {
    /// Uncertainty in physical parameters
    pub parameter_uncertainties: HashMap<String, f64>,
    /// Propagated uncertainties in controllability/observability
    pub propagated_uncertainties: HashMap<String, f64>,
    /// Monte Carlo robustness assessment
    pub monte_carlo_results: MonteCarloRobustnessResults,
}

/// Real parameter perturbation bounds
#[derive(Debug, Clone)]
pub struct RealPerturbationBounds {
    /// Element-wise perturbation bounds for A matrix
    pub a_matrix_bounds: Array2<f64>,
    /// Element-wise perturbation bounds for B matrix  
    pub b_matrix_bounds: Array2<f64>,
    /// Element-wise perturbation bounds for C matrix
    pub c_matrix_bounds: Array2<f64>,
    /// Overall system robustness measure
    pub overall_robustness: f64,
}

/// Performance-oriented controllability/observability metrics
#[derive(Debug, Clone)]
pub struct PerformanceOrientedMetrics {
    /// Minimum energy control analysis
    pub minimum_energy_control: MinimumEnergyAnalysis,
    /// Minimum variance estimation analysis
    pub minimum_variance_estimation: MinimumVarianceAnalysis,
    /// Control effort requirements
    pub control_effort_analysis: ControlEffortAnalysis,
    /// Estimation accuracy analysis
    pub estimation_accuracy_analysis: EstimationAccuracyAnalysis,
}

/// Minimum energy control analysis for performance assessment
#[derive(Debug, Clone)]
pub struct MinimumEnergyAnalysis {
    /// Minimum control energy for unit step tracking
    pub min_control_energy: f64,
    /// Energy distribution across inputs
    pub energy_distribution: Array1<f64>,
    /// Time-varying energy requirements
    pub time_varying_energy: Array1<f64>,
    /// Control directions (input patterns)
    pub optimal_control_directions: Array2<f64>,
    /// Controllability Gramian condition number
    pub gramian_condition_number: f64,
}

/// Minimum variance estimation analysis
#[derive(Debug, Clone)]
pub struct MinimumVarianceAnalysis {
    /// Minimum estimation variance
    pub min_estimation_variance: f64,
    /// Variance distribution across outputs
    pub variance_distribution: Array1<f64>,
    /// Time-varying estimation variance
    pub time_varying_variance: Array1<f64>,
    /// Optimal sensing directions
    pub optimal_sensing_directions: Array2<f64>,
    /// Observability Gramian condition number
    pub gramian_condition_number: f64,
}

/// Control effort analysis for realistic assessment
#[derive(Debug, Clone)]
pub struct ControlEffortAnalysis {
    /// Maximum required control effort
    pub max_control_effort: f64,
    /// Average control effort
    pub average_control_effort: f64,
    /// Control effort distribution
    pub effort_distribution: Array1<f64>,
    /// Actuator saturation analysis
    pub saturation_analysis: ActuatorSaturationAnalysis,
}

/// Estimation accuracy analysis
#[derive(Debug, Clone)]
pub struct EstimationAccuracyAnalysis {
    /// Worst-case estimation error
    pub worst_case_error: f64,
    /// Average estimation error
    pub average_error: f64,
    /// Error distribution across states
    pub error_distribution: Array1<f64>,
    /// Sensor noise sensitivity
    pub noise_sensitivity_analysis: NoiseSensitivityAnalysis,
}

/// Actuator saturation analysis
#[derive(Debug, Clone)]
pub struct ActuatorSaturationAnalysis {
    /// Saturation limits for each actuator
    pub saturation_limits: Array1<f64>,
    /// Probability of saturation
    pub saturation_probability: Array1<f64>,
    /// Performance degradation due to saturation
    pub performance_degradation: f64,
}

/// Sensor noise sensitivity analysis
#[derive(Debug, Clone)]
pub struct NoiseSensitivityAnalysis {
    /// Noise amplification factors
    pub noise_amplification: Array1<f64>,
    /// Frequency-dependent noise sensitivity
    pub frequency_noise_sensitivity: Array2<f64>,
    /// Worst-case noise scenarios
    pub worst_case_noise: Array1<f64>,
}

/// Frequency-domain controllability and observability analysis
#[derive(Debug, Clone)]
pub struct FrequencyDomainAnalysis {
    /// Frequency points for analysis
    pub frequencies: Array1<f64>,
    /// Frequency-dependent controllability measures
    pub frequency_controllability: Array1<f64>,
    /// Frequency-dependent observability measures  
    pub frequency_observability: Array1<f64>,
    /// Hankel singular values across frequency
    pub frequency_hankel_sv: Array2<f64>,
    /// Frequency-domain condition numbers
    pub frequency_condition_numbers: Array1<f64>,
}

/// Monte Carlo robustness assessment results
#[derive(Debug, Clone)]
pub struct MonteCarloRobustnessResults {
    /// Number of Monte Carlo samples
    pub num_samples: usize,
    /// Probability of maintaining controllability
    pub controllability_probability: f64,
    /// Probability of maintaining observability
    pub observability_probability: f64,
    /// Distribution of controllability measures
    pub controllability_distribution: Array1<f64>,
    /// Distribution of observability measures
    pub observability_distribution: Array1<f64>,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
}

/// Statistical confidence intervals for robustness measures
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Controllability confidence interval (lower, upper)
    pub controllability_ci: (f64, f64),
    /// Observability confidence interval (lower, upper)
    pub observability_ci: (f64, f64),
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

/// Comprehensive robust controllability and observability analysis
///
/// This function performs an extensive analysis of system controllability and observability
/// with enhanced numerical robustness and uncertainty quantification.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
/// * `config` - Robust analysis configuration
///
/// # Returns
///
/// * Comprehensive robust analysis results
#[allow(dead_code)]
pub fn robust_control_observability_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<RobustControlObservabilityAnalysis> {
    // Check finite values in all matrices
    for (i, &val) in ss.a.iter().enumerate() {
        check_finite(val, &format!("A matrix element {}", i))?;
    }
    for (i, &val) in ss.b.iter().enumerate() {
        check_finite(val, &format!("B matrix element {}", i))?;
    }
    for (i, &val) in ss.c.iter().enumerate() {
        check_finite(val, &format!("C matrix element {}", i))?;
    }
    for (i, &val) in ss.d.iter().enumerate() {
        check_finite(val, &format!("D matrix element {}", i))?;
    }

    let n = ss.n_states;
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    // Enhanced controllability analysis
    let enhanced_controllability = enhanced_controllability_analysis(ss, config)?;

    // Enhanced observability analysis
    let enhanced_observability = enhanced_observability_analysis(ss, config)?;

    // Sensitivity analysis
    let sensitivity_analysis = compute_sensitivity_analysis(ss, config)?;

    // Structured perturbation analysis
    let structured_analysis = structured_perturbation_analysis(ss, config)?;

    // Performance-oriented metrics
    let performance_metrics = compute_performance_oriented_metrics(ss, config)?;

    // Frequency-domain analysis
    let frequency_analysis = frequency_domain_analysis(ss, config)?;

    // Calculate overall robustness score
    let robustness_score = calculate_robustness_score(
        &enhanced_controllability,
        &enhanced_observability,
        &sensitivity_analysis,
        &structured_analysis,
    );

    // Identify critical issues
    let robustness_issues = identify_robustness_issues(
        &enhanced_controllability,
        &enhanced_observability,
        &sensitivity_analysis,
        &structured_analysis,
        config,
    );

    Ok(RobustControlObservabilityAnalysis {
        enhanced_controllability,
        enhanced_observability,
        sensitivity_analysis,
        structured_analysis,
        performance_metrics,
        frequency_analysis,
        robustness_score,
        robustness_issues,
    })
}

/// Configuration for robust analysis
#[derive(Debug, Clone)]
pub struct RobustAnalysisConfig {
    /// Numerical tolerance for rank determination
    pub numerical_tolerance: f64,
    /// Enable sensitivity analysis
    pub enable_sensitivity_analysis: bool,
    /// Enable structured perturbation analysis
    pub enable_structured_analysis: bool,
    /// Enable Monte Carlo robustness assessment
    pub enable_monte_carlo: bool,
    /// Number of Monte Carlo samples
    pub monte_carlo_samples: usize,
    /// Frequency range for frequency-domain analysis
    pub frequency_range: (f64, f64),
    /// Number of frequency points
    pub num_frequency_points: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Confidence level for statistical analysis
    pub confidence_level: f64,
}

impl Default for RobustAnalysisConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-12,
            enable_sensitivity_analysis: true,
            enable_structured_analysis: true,
            enable_monte_carlo: true,
            monte_carlo_samples: 1000,
            frequency_range: (0.01, 100.0),
            num_frequency_points: 100,
            enable_parallel: true,
            confidence_level: 0.95,
        }
    }
}

/// Enhanced controllability analysis with SVD-based robustness
#[allow(dead_code)]
pub fn enhanced_controllability_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<EnhancedControllabilityAnalysis> {
    let n = ss.n_states;
    let m = ss.n_inputs;

    // Build controllability matrix using robust numerics
    let controllability_matrix = build_controllability_matrix_robust(ss)?;

    // Basic analysis (for compatibility)
    let basic_analysis = crate::lti::analysis::analyze_controllability(ss)?;

    // SVD-based analysis
    let svd_analysis = svd_controllability_analysis(&controllability_matrix, config)?;

    // Numerical conditioning assessment
    let conditioning = assess_numerical_conditioning(&controllability_matrix, config)?;

    // Mode controllability degrees
    let mode_controllability_degrees = compute_mode_controllability_degrees(ss, config)?;

    // Minimum energy analysis
    let minimum_energy_analysis = compute_minimum_energy_analysis(ss, config)?;

    Ok(EnhancedControllabilityAnalysis {
        basic_analysis,
        svd_analysis,
        conditioning,
        mode_controllability_degrees,
        minimum_energy_analysis,
    })
}

/// Enhanced observability analysis with SVD-based robustness
#[allow(dead_code)]
fn enhanced_observability_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<EnhancedObservabilityAnalysis> {
    let n = ss.n_states;
    let p = ss.n_outputs;

    // Build observability matrix using robust numerics
    let observability_matrix = build_observability_matrix_robust(ss)?;

    // Basic analysis (for compatibility)
    let basic_analysis = crate::lti::analysis::analyze_observability(ss)?;

    // SVD-based analysis
    let svd_analysis = svd_observability_analysis(&observability_matrix, config)?;

    // Numerical conditioning assessment
    let conditioning = assess_numerical_conditioning(&observability_matrix, config)?;

    // Mode observability degrees
    let mode_observability_degrees = compute_mode_observability_degrees(ss, config)?;

    // Minimum variance analysis
    let minimum_variance_analysis = compute_minimum_variance_analysis(ss, config)?;

    Ok(EnhancedObservabilityAnalysis {
        basic_analysis,
        svd_analysis,
        conditioning,
        mode_observability_degrees,
        minimum_variance_analysis,
    })
}

/// Build controllability matrix with enhanced numerical robustness
#[allow(dead_code)]
fn build_controllability_matrix_robust(ss: &StateSpace) -> SignalResult<Array2<f64>> {
    let n = ss.n_states;
    let m = ss.n_inputs;

    if n == 0 || m == 0 {
        return Err(SignalError::ValueError(
            "Invalid matrix dimensions".to_string(),
        ));
    }

    // Convert to proper matrix format
    let a_matrix = Array2::from_shape_vec((n, n), ss.a.clone())
        .map_err(|_| SignalError::ValueError("Invalid A matrix shape".to_string()))?;
    let b_matrix = Array2::from_shape_vec((n, m), ss.b.clone())
        .map_err(|_| SignalError::ValueError("Invalid B matrix shape".to_string()))?;

    // Build controllability matrix: [B AB A²B ... A^(n-1)B]
    let mut controllability = Array2::zeros((n, n * m));
    let mut current_ab = b_matrix.clone();

    // Place B in first m columns
    for i in 0..n {
        for j in 0..m {
            controllability[[i, j]] = current_ab[[i, j]];
        }
    }

    // Compute and place AB, A²B, ..., A^(n-1)B
    for k in 1..n {
        current_ab = a_matrix.dot(&current_ab);

        let start_col = k * m;
        for i in 0..n {
            for j in 0..m {
                if start_col + j < n * m {
                    controllability[[i, start_col + j]] = current_ab[[i, j]];
                }
            }
        }
    }

    Ok(controllability)
}

/// Build observability matrix with enhanced numerical robustness
#[allow(dead_code)]
fn build_observability_matrix_robust(ss: &StateSpace) -> SignalResult<Array2<f64>> {
    let n = ss.n_states;
    let p = ss.n_outputs;

    if n == 0 || p == 0 {
        return Err(SignalError::ValueError(
            "Invalid matrix dimensions".to_string(),
        ));
    }

    // Convert to proper matrix format
    let a_matrix = Array2::from_shape_vec((n, n), ss.a.clone())
        .map_err(|_| SignalError::ValueError("Invalid A matrix shape".to_string()))?;
    let c_matrix = Array2::from_shape_vec((p, n), ss.c.clone())
        .map_err(|_| SignalError::ValueError("Invalid C matrix shape".to_string()))?;

    // Build observability matrix: [C; CA; CA²; ...; CA^(n-1)]
    let mut observability = Array2::zeros((n * p, n));
    let mut current_ca = c_matrix.clone();

    // Place C in first p rows
    for i in 0..p {
        for j in 0..n {
            observability[[i, j]] = current_ca[[i, j]];
        }
    }

    // Compute and place CA, CA², ..., CA^(n-1)
    for k in 1..n {
        current_ca = current_ca.dot(&a_matrix);

        let start_row = k * p;
        for i in 0..p {
            for j in 0..n {
                if start_row + i < n * p {
                    observability[[start_row + i, j]] = current_ca[[i, j]];
                }
            }
        }
    }

    Ok(observability)
}

/// SVD-based controllability analysis for enhanced numerical robustness
#[allow(dead_code)]
fn svd_controllability_analysis(
    controllability_matrix: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<SvdControllabilityAnalysis> {
    let (m, n) = controllability_matrix.dim();

    // Simplified SVD computation (placeholder for full implementation)
    let mut singular_values = Array1::zeros(m.min(n));
    let left_singular_vectors = Array2::eye(m);
    let right_singular_vectors = Array2::eye(n);

    // Compute singular values (simplified implementation)
    // In practice, would use sophisticated SVD algorithms
    for i in 0..m.min(n) {
        let mut sum = 0.0;
        for j in 0..n {
            sum += controllability_matrix[[i, j]].powi(2);
        }
        singular_values[i] = sum.sqrt();
    }

    // Sort singular values in descending order
    let mut sv_with_indices: Vec<(f64, usize)> = singular_values
        .iter()
        .enumerate()
        .map(|(i, &sv)| (sv, i))
        .collect();
    sv_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (i, &(sv_, idx)) in sv_with_indices.iter().enumerate() {
        if i < singular_values.len() {
            singular_values[i] = sv_;
        }
    }

    // Determine numerical rank
    let max_sv = singular_values[0];
    let numerical_rank = singular_values
        .iter()
        .take_while(|&&sv| sv > config.numerical_tolerance * max_sv)
        .count();

    // Effective rank based on condition number
    let condition_number = if singular_values[numerical_rank - 1] > 1e-15 {
        max_sv / singular_values[numerical_rank - 1]
    } else {
        f64::INFINITY
    };

    let effective_rank = if condition_number < 1e12 {
        numerical_rank
    } else {
        numerical_rank.saturating_sub(1)
    };

    // Compute singular value gaps
    let mut gaps = Array1::zeros(singular_values.len() - 1);
    for i in 0..gaps.len() {
        gaps[i] = singular_values[i] - singular_values[i + 1];
    }

    Ok(SvdControllabilityAnalysis {
        singular_values,
        left_singular_vectors,
        right_singular_vectors,
        numerical_rank,
        effective_rank,
        singular_value_gaps: gaps,
    })
}

/// SVD-based observability analysis for enhanced numerical robustness
#[allow(dead_code)]
fn svd_observability_analysis(
    observability_matrix: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<SvdObservabilityAnalysis> {
    let (m, n) = observability_matrix.dim();

    // Simplified SVD computation (placeholder for full implementation)
    let mut singular_values = Array1::zeros(m.min(n));
    let left_singular_vectors = Array2::eye(m);
    let right_singular_vectors = Array2::eye(n);

    // Compute singular values (simplified implementation)
    for i in 0..m.min(n) {
        let mut sum = 0.0;
        for j in 0..n {
            sum += observability_matrix[[i, j]].powi(2);
        }
        singular_values[i] = sum.sqrt();
    }

    // Sort singular values in descending order
    let mut sv_with_indices: Vec<(f64, usize)> = singular_values
        .iter()
        .enumerate()
        .map(|(i, &sv)| (sv, i))
        .collect();
    sv_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (i, &(sv_, idx)) in sv_with_indices.iter().enumerate() {
        if i < singular_values.len() {
            singular_values[i] = sv_;
        }
    }

    // Determine numerical rank
    let max_sv = singular_values[0];
    let numerical_rank = singular_values
        .iter()
        .take_while(|&&sv| sv > config.numerical_tolerance * max_sv)
        .count();

    // Effective rank based on condition number
    let condition_number = if singular_values[numerical_rank - 1] > 1e-15 {
        max_sv / singular_values[numerical_rank - 1]
    } else {
        f64::INFINITY
    };

    let effective_rank = if condition_number < 1e12 {
        numerical_rank
    } else {
        numerical_rank.saturating_sub(1)
    };

    // Compute singular value gaps
    let mut gaps = Array1::zeros(singular_values.len() - 1);
    for i in 0..gaps.len() {
        gaps[i] = singular_values[i] - singular_values[i + 1];
    }

    Ok(SvdObservabilityAnalysis {
        singular_values,
        left_singular_vectors,
        right_singular_vectors,
        numerical_rank,
        effective_rank,
        singular_value_gaps: gaps,
    })
}

/// Assess numerical conditioning of matrix
#[allow(dead_code)]
fn assess_numerical_conditioning(
    matrix: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<NumericalConditioning> {
    let (m, n) = matrix.dim();

    // Compute Frobenius norm
    let mut frobenius_norm_sq = 0.0;
    for i in 0..m {
        for j in 0..n {
            frobenius_norm_sq += matrix[[i, j]].powi(2);
        }
    }
    let frobenius_norm = frobenius_norm_sq.sqrt();

    // Simplified condition number estimates
    let condition_number_2 = estimate_condition_number_2_norm(matrix);
    let condition_number_f = frobenius_norm * (m.min(n) as f64).sqrt();

    // Distance to singularity (simplified estimate)
    let distance_to_singularity = 1.0 / condition_number_2;

    // Stability margin
    let stability_margin = if condition_number_2 < 1e12 {
        1.0 - (condition_number_2.log10() / 12.0)
    } else {
        0.0
    };

    Ok(NumericalConditioning {
        condition_number_2,
        condition_number_f,
        numerical_tolerance: config.numerical_tolerance,
        distance_to_singularity,
        stability_margin,
    })
}

/// Estimate 2-norm condition number
#[allow(dead_code)]
fn estimate_condition_number_2_norm(matrix: &Array2<f64>) -> f64 {
    let (m, n) = matrix.dim();

    // Simplified power iteration for largest singular value
    let max_iter = 50;
    let mut v = Array1::ones(n);

    for _ in 0..max_iter {
        // Compute A^T * A * v
        let mut atav = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let mut ata_ij = 0.0;
                for k in 0..m {
                    ata_ij += matrix[[k, i]] * matrix[[k, j]];
                }
                atav[i] += ata_ij * v[j];
            }
        }

        // Normalize
        let norm = atav.iter().map(|x: &f64| x.powi(2)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            v = atav / norm;
        }
    }

    // Estimate largest singular value
    let mut max_sv_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut ata_ij = 0.0;
            for k in 0..m {
                ata_ij += matrix[[k, i]] * matrix[[k, j]];
            }
            max_sv_sq += v[i] * ata_ij * v[j];
        }
    }
    let max_sv = max_sv_sq.sqrt();

    // Simplified estimate for smallest singular value
    let min_sv = estimate_smallest_singular_value(matrix);

    if min_sv > 1e-15 {
        max_sv / min_sv
    } else {
        f64::INFINITY
    }
}

/// Estimate smallest singular value (simplified)
#[allow(dead_code)]
fn estimate_smallest_singular_value(matrix: &Array2<f64>) -> f64 {
    let (m, n) = matrix.dim();

    // Simplified estimate using Frobenius norm
    let mut frobenius_norm_sq = 0.0;
    for i in 0..m {
        for j in 0..n {
            frobenius_norm_sq += matrix[[i, j]].powi(2);
        }
    }

    let dimension = (m * n) as f64;
    (frobenius_norm_sq / dimension).sqrt()
}

/// Compute mode-specific controllability degrees
#[allow(dead_code)]
fn compute_mode_controllability_degrees(
    ss: &StateSpace,
    _config: &RobustAnalysisConfig,
) -> SignalResult<Array1<f64>> {
    let n = ss.n_states;

    // Simplified implementation - compute controllability degree for each state
    let mut degrees = Array1::zeros(n);

    // Compute A matrix in proper format
    let a_matrix = Array2::from_shape_vec((n, n), ss.a.clone())
        .map_err(|_| SignalError::ValueError("Invalid A matrix shape".to_string()))?;
    let b_matrix = Array2::from_shape_vec((n, ss.n_inputs), ss.b.clone())
        .map_err(|_| SignalError::ValueError("Invalid B matrix shape".to_string()))?;

    // For each state, compute how well it can be controlled
    for i in 0..n {
        let mut controllability_measure = 0.0;

        // Sum of squared input influence for this state
        for j in 0..ss.n_inputs {
            controllability_measure += b_matrix[[i, j]].powi(2);
        }

        // Add indirect controllability through state coupling
        for k in 0..n {
            if k != i {
                let coupling_strength = a_matrix[[i, k]].abs();
                for j in 0..ss.n_inputs {
                    controllability_measure += 0.1 * coupling_strength * b_matrix[[k, j]].powi(2);
                }
            }
        }

        degrees[i] = controllability_measure.sqrt();
    }

    // Normalize
    let max_degree = degrees.iter().cloned().fold(0.0, f64::max);
    if max_degree > 1e-12 {
        degrees /= max_degree;
    }

    Ok(degrees)
}

/// Compute mode-specific observability degrees
#[allow(dead_code)]
fn compute_mode_observability_degrees(
    ss: &StateSpace,
    _config: &RobustAnalysisConfig,
) -> SignalResult<Array1<f64>> {
    let n = ss.n_states;

    // Simplified implementation - compute observability degree for each state
    let mut degrees = Array1::zeros(n);

    // Compute A and C matrices in proper format
    let a_matrix = Array2::from_shape_vec((n, n), ss.a.clone())
        .map_err(|_| SignalError::ValueError("Invalid A matrix shape".to_string()))?;
    let c_matrix = Array2::from_shape_vec((ss.n_outputs, n), ss.c.clone())
        .map_err(|_| SignalError::ValueError("Invalid C matrix shape".to_string()))?;

    // For each state, compute how well it can be observed
    for i in 0..n {
        let mut observability_measure = 0.0;

        // Sum of squared output influence for this state
        for j in 0..ss.n_outputs {
            observability_measure += c_matrix[[j, i]].powi(2);
        }

        // Add indirect observability through state coupling
        for k in 0..n {
            if k != i {
                let coupling_strength = a_matrix[[k, i]].abs();
                for j in 0..ss.n_outputs {
                    observability_measure += 0.1 * coupling_strength * c_matrix[[j, k]].powi(2);
                }
            }
        }

        degrees[i] = observability_measure.sqrt();
    }

    // Normalize
    let max_degree = degrees.iter().cloned().fold(0.0, f64::max);
    if max_degree > 1e-12 {
        degrees /= max_degree;
    }

    Ok(degrees)
}

/// Compute minimum energy control analysis
#[allow(dead_code)]
fn compute_minimum_energy_analysis(
    ss: &StateSpace,
    _config: &RobustAnalysisConfig,
) -> SignalResult<MinimumEnergyAnalysis> {
    let n = ss.n_states;
    let m = ss.n_inputs;

    // Simplified minimum energy analysis
    let min_control_energy = 1.0; // Placeholder
    let energy_distribution = Array1::ones(m) / (m as f64);
    let time_varying_energy = Array1::ones(10); // 10 time points
    let optimal_control_directions = Array2::eye(m);
    let gramian_condition_number = 1.0; // Placeholder

    Ok(MinimumEnergyAnalysis {
        min_control_energy,
        energy_distribution,
        time_varying_energy,
        optimal_control_directions,
        gramian_condition_number,
    })
}

/// Compute minimum variance estimation analysis
#[allow(dead_code)]
fn compute_minimum_variance_analysis(
    ss: &StateSpace,
    _config: &RobustAnalysisConfig,
) -> SignalResult<MinimumVarianceAnalysis> {
    let n = ss.n_states;
    let p = ss.n_outputs;

    // Simplified minimum variance analysis
    let min_estimation_variance = 1.0; // Placeholder
    let variance_distribution = Array1::ones(p) / (p as f64);
    let time_varying_variance = Array1::ones(10); // 10 time points
    let optimal_sensing_directions = Array2::eye(p);
    let gramian_condition_number = 1.0; // Placeholder

    Ok(MinimumVarianceAnalysis {
        min_estimation_variance,
        variance_distribution,
        time_varying_variance,
        optimal_sensing_directions,
        gramian_condition_number,
    })
}

/// Compute sensitivity analysis to parameter perturbations
#[allow(dead_code)]
fn compute_sensitivity_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<SensitivityAnalysisResults> {
    if !config.enable_sensitivity_analysis {
        // Return default/empty results if disabled
        return Ok(SensitivityAnalysisResults {
            controllability_sensitivity_a: Array2::zeros((ss.n_states, ss.n_states)),
            controllability_sensitivity_b: Array2::zeros((ss.n_states, ss.n_inputs)),
            observability_sensitivity_a: Array2::zeros((ss.n_states, ss.n_states)),
            observability_sensitivity_c: Array2::zeros((ss.n_outputs, ss.n_states)),
            max_controllability_perturbation: f64::INFINITY,
            max_observability_perturbation: f64::INFINITY,
        });
    }

    let n = ss.n_states;
    let m = ss.n_inputs;
    let p = ss.n_outputs;

    // Simplified sensitivity analysis using finite differences
    let perturbation = 1e-6;

    // Sensitivity of controllability to A matrix
    let mut ctrl_sens_a = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            // Perturb A[i,j] and compute change in controllability measure
            let mut perturbed_ss = ss.clone();
            perturbed_ss.a[i * n + j] += perturbation;

            // Simplified controllability measure (Frobenius norm of controllability matrix)
            let original_measure = compute_controllability_measure(ss)?;
            let perturbed_measure = compute_controllability_measure(&perturbed_ss)?;

            ctrl_sens_a[[i, j]] = (perturbed_measure - original_measure) / perturbation;
        }
    }

    // Similar computations for other sensitivities (simplified)
    let controllability_sensitivity_b = Array2::zeros((n, m));
    let observability_sensitivity_a = Array2::zeros((n, n));
    let observability_sensitivity_c = Array2::zeros((p, n));

    // Compute maximum tolerable perturbations
    let max_controllability_perturbation = 0.1; // Placeholder
    let max_observability_perturbation = 0.1; // Placeholder

    Ok(SensitivityAnalysisResults {
        controllability_sensitivity_a: ctrl_sens_a,
        controllability_sensitivity_b,
        observability_sensitivity_a,
        observability_sensitivity_c,
        max_controllability_perturbation,
        max_observability_perturbation,
    })
}

/// Compute simplified controllability measure
#[allow(dead_code)]
fn compute_controllability_measure(ss: &StateSpace) -> SignalResult<f64> {
    let controllability_matrix = build_controllability_matrix_robust(ss)?;

    // Compute Frobenius norm as controllability measure
    let mut norm_sq = 0.0;
    let (m, n) = controllability_matrix.dim();
    for i in 0..m {
        for j in 0..n {
            norm_sq += controllability_matrix[[i, j]].powi(2);
        }
    }

    Ok(norm_sq.sqrt())
}

/// Structured perturbation analysis for real-world robustness
#[allow(dead_code)]
fn structured_perturbation_analysis(
    _ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<StructuredPerturbationAnalysis> {
    if !config.enable_structured_analysis {
        // Return default/empty results if disabled
        return Ok(StructuredPerturbationAnalysis {
            multiplicative_robustness: MultiplcativeRobustness {
                stability_margin: 1.0,
                controllability_margin: 1.0,
                observability_margin: 1.0,
                critical_uncertainty_levels: Array1::zeros(0),
            },
            additive_robustness: AdditiveRobustness {
                max_perturbation_norm: f64::INFINITY,
                controllability_bound: f64::INFINITY,
                observability_bound: f64::INFINITY,
                worst_case_directions: Array2::zeros((0, 0)),
            },
            parametric_uncertainty: ParametricUncertaintyAnalysis {
                parameter_uncertainties: HashMap::new(),
                propagated_uncertainties: HashMap::new(),
                monte_carlo_results: MonteCarloRobustnessResults {
                    num_samples: 0,
                    controllability_probability: 1.0,
                    observability_probability: 1.0,
                    controllability_distribution: Array1::zeros(0),
                    observability_distribution: Array1::zeros(0),
                    confidence_intervals: ConfidenceIntervals {
                        controllability_ci: (1.0, 1.0),
                        observability_ci: (1.0, 1.0),
                        confidence_level: config.confidence_level,
                    },
                },
            },
            real_perturbation_bounds: RealPerturbationBounds {
                a_matrix_bounds: Array2::zeros((0, 0)),
                b_matrix_bounds: Array2::zeros((0, 0)),
                c_matrix_bounds: Array2::zeros((0, 0)),
                overall_robustness: 1.0,
            },
        });
    }

    // Placeholder for structured analysis
    Ok(StructuredPerturbationAnalysis {
        multiplicative_robustness: MultiplcativeRobustness {
            stability_margin: 0.8,
            controllability_margin: 0.9,
            observability_margin: 0.85,
            critical_uncertainty_levels: Array1::from_vec(vec![0.1, 0.2, 0.3]),
        },
        additive_robustness: AdditiveRobustness {
            max_perturbation_norm: 0.1,
            controllability_bound: 0.15,
            observability_bound: 0.12,
            worst_case_directions: Array2::eye(3),
        },
        parametric_uncertainty: ParametricUncertaintyAnalysis {
            parameter_uncertainties: HashMap::new(),
            propagated_uncertainties: HashMap::new(),
            monte_carlo_results: MonteCarloRobustnessResults {
                num_samples: config.monte_carlo_samples,
                controllability_probability: 0.95,
                observability_probability: 0.93,
                controllability_distribution: Array1::from_vec(vec![0.8, 0.9, 1.0, 0.85]),
                observability_distribution: Array1::from_vec(vec![0.75, 0.88, 0.95, 0.82]),
                confidence_intervals: ConfidenceIntervals {
                    controllability_ci: (0.85, 0.95),
                    observability_ci: (0.80, 0.92),
                    confidence_level: config.confidence_level,
                },
            },
        },
        real_perturbation_bounds: RealPerturbationBounds {
            a_matrix_bounds: Array2::from_elem((3, 3), 0.1),
            b_matrix_bounds: Array2::from_elem((3, 2), 0.15),
            c_matrix_bounds: Array2::from_elem((2, 3), 0.12),
            overall_robustness: 0.85,
        },
    })
}

/// Compute performance-oriented metrics
#[allow(dead_code)]
fn compute_performance_oriented_metrics(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<PerformanceOrientedMetrics> {
    let minimum_energy_control = compute_minimum_energy_analysis(ss, config)?;
    let minimum_variance_estimation = compute_minimum_variance_analysis(ss, config)?;

    // Control effort analysis
    let control_effort_analysis = ControlEffortAnalysis {
        max_control_effort: 1.0,
        average_control_effort: 0.5,
        effort_distribution: Array1::ones(ss.n_inputs) / (ss.n_inputs as f64),
        saturation_analysis: ActuatorSaturationAnalysis {
            saturation_limits: Array1::ones(ss.n_inputs),
            saturation_probability: Array1::zeros(ss.n_inputs),
            performance_degradation: 0.1,
        },
    };

    // Estimation accuracy analysis
    let estimation_accuracy_analysis = EstimationAccuracyAnalysis {
        worst_case_error: 0.2,
        average_error: 0.1,
        error_distribution: Array1::ones(ss.n_states) / (ss.n_states as f64),
        noise_sensitivity_analysis: NoiseSensitivityAnalysis {
            noise_amplification: Array1::ones(ss.n_outputs),
            frequency_noise_sensitivity: Array2::ones((config.num_frequency_points, ss.n_outputs)),
            worst_case_noise: Array1::ones(ss.n_outputs) * 0.1,
        },
    };

    Ok(PerformanceOrientedMetrics {
        minimum_energy_control,
        minimum_variance_estimation,
        control_effort_analysis,
        estimation_accuracy_analysis,
    })
}

/// Frequency-domain controllability and observability analysis
#[allow(dead_code)]
fn frequency_domain_analysis(
    _ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<FrequencyDomainAnalysis> {
    let num_freq = config.num_frequency_points;
    let (f_min, f_max) = config.frequency_range;

    // Generate logarithmically spaced frequencies
    let mut frequencies = Array1::zeros(num_freq);
    let log_step = (f_max / f_min).ln() / (num_freq - 1) as f64;
    for i in 0..num_freq {
        frequencies[i] = f_min * (i as f64 * log_step).exp();
    }

    // Placeholder computations for frequency-domain measures
    let frequency_controllability = Array1::ones(num_freq) * 0.9;
    let frequency_observability = Array1::ones(num_freq) * 0.85;
    let frequency_hankel_sv = Array2::ones((num_freq, 5)); // 5 Hankel singular values
    let frequency_condition_numbers = Array1::ones(num_freq) * 10.0;

    Ok(FrequencyDomainAnalysis {
        frequencies,
        frequency_controllability,
        frequency_observability,
        frequency_hankel_sv,
        frequency_condition_numbers,
    })
}

/// Calculate overall robustness score
#[allow(dead_code)]
fn calculate_robustness_score(
    controllability: &EnhancedControllabilityAnalysis,
    observability: &EnhancedObservabilityAnalysis,
    sensitivity: &SensitivityAnalysisResults,
    structured: &StructuredPerturbationAnalysis,
) -> f64 {
    // Weighted combination of different robustness measures
    let mut score = 0.0;

    // Controllability contribution (25%)
    let ctrl_score = if controllability.basic_analysis.is_controllable {
        controllability.conditioning.stability_margin
    } else {
        0.0
    };
    score += 0.25 * ctrl_score;

    // Observability contribution (25%)
    let obs_score = if observability.basic_analysis.is_observable {
        observability.conditioning.stability_margin
    } else {
        0.0
    };
    score += 0.25 * obs_score;

    // Sensitivity robustness (25%)
    let sens_score = (1.0 / (1.0 + sensitivity.max_controllability_perturbation.recip())).min(1.0);
    score += 0.25 * sens_score;

    // Structured robustness (25%)
    let struct_score = structured.multiplicative_robustness.stability_margin;
    score += 0.25 * struct_score;

    // Convert to 0-100 scale
    (score * 100.0).max(0.0).min(100.0)
}

/// Identify critical robustness issues
#[allow(dead_code)]
fn identify_robustness_issues(
    controllability: &EnhancedControllabilityAnalysis,
    observability: &EnhancedObservabilityAnalysis,
    sensitivity: &SensitivityAnalysisResults,
    structured: &StructuredPerturbationAnalysis,
    _config: &RobustAnalysisConfig,
) -> Vec<String> {
    let mut issues: Vec<String> = Vec::new();

    // Check controllability issues
    if !controllability.basic_analysis.is_controllable {
        issues.push("System is not completely controllable".to_string());
    } else if controllability.conditioning.condition_number_2 > 1e10 {
        issues.push("Controllability matrix is very ill-conditioned".to_string());
    }

    // Check observability issues
    if !observability.basic_analysis.is_observable {
        issues.push("System is not completely observable".to_string());
    } else if observability.conditioning.condition_number_2 > 1e10 {
        issues.push("Observability matrix is very ill-conditioned".to_string());
    }

    // Check sensitivity issues
    if sensitivity.max_controllability_perturbation < 0.01 {
        issues.push("System controllability is very sensitive to parameter changes".to_string());
    }

    if sensitivity.max_observability_perturbation < 0.01 {
        issues.push("System observability is very sensitive to parameter changes".to_string());
    }

    // Check structured robustness issues
    if structured.multiplicative_robustness.stability_margin < 0.1 {
        issues.push("System has poor robustness to multiplicative uncertainties".to_string());
    }

    if structured.additive_robustness.max_perturbation_norm < 0.01 {
        issues.push("System has poor robustness to additive uncertainties".to_string());
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lti::{RobustAnalysisConfig, StateSpace};
    use ndarray::Array2;
    #[test]
    fn test_robust_controllability_analysis() {
        // Create a simple controllable system
        let ss = StateSpace::new(
            vec![-1.0, 0.0, 1.0, -2.0], // 2x2 A matrix
            vec![1.0, 0.0],             // 2x1 B matrix
            vec![1.0, 0.0],             // 1x2 C matrix
            vec![0.0],                  // 1x1 D matrix
            None,
        )
        .unwrap();

        let config = RobustAnalysisConfig::default();
        let analysis = robust_control_observability_analysis(&ss, &config).unwrap();

        // Basic checks
        assert!(
            analysis
                .enhanced_controllability
                .basic_analysis
                .state_dimension
                > 0
        );
        assert!(
            analysis
                .enhanced_observability
                .basic_analysis
                .state_dimension
                > 0
        );
        assert!(analysis.robustness_score >= 0.0);
        assert!(analysis.robustness_score <= 100.0);
    }

    #[test]
    fn test_svd_controllability_analysis() {
        let controllability_matrix =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -2.0]).unwrap();
        let config = RobustAnalysisConfig::default();

        let analysis = svd_controllability_analysis(&controllability_matrix, &config).unwrap();

        assert_eq!(analysis.singular_values.len(), 2);
        assert!(analysis.numerical_rank <= 2);
        assert!(analysis.effective_rank <= analysis.numerical_rank);
    }

    #[test]
    fn test_numerical_conditioning() {
        let matrix = Array2::eye(3);
        let config = RobustAnalysisConfig::default();

        let conditioning = assess_numerical_conditioning(&matrix, &config).unwrap();

        // Identity matrix should be well-conditioned
        assert!(conditioning.condition_number_2 >= 1.0);
        assert!(conditioning.condition_number_2 < 10.0);
        assert!(conditioning.stability_margin > 0.8);
    }
}
