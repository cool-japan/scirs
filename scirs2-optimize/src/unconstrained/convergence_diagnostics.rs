//! Enhanced convergence diagnostics for optimization algorithms
//!
//! This module provides comprehensive diagnostic capabilities for monitoring
//! and analyzing optimization progress, including:
//! - Real-time convergence metrics tracking
//! - Algorithm health indicators
//! - Performance profiling
//! - Iteration-by-iteration analysis
//! - Convergence prediction
//! - Problem difficulty assessment
//! - Visualization data export

use crate::error::OptimizeError;
use scirs2_core::ndarray::ArrayView1;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Comprehensive convergence diagnostics
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    /// Iteration history
    pub nit: Vec<IterationDiagnostic>,
    /// Algorithm performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Problem characteristics
    pub problem_analysis: ProblemAnalysis,
    /// Warnings and recommendations
    pub warnings: Vec<DiagnosticWarning>,
}

/// Diagnostic information for a single iteration
#[derive(Debug, Clone)]
pub struct IterationDiagnostic {
    /// Iteration number
    pub iteration: usize,
    /// Function value
    pub f_value: f64,
    /// Gradient norm
    pub grad_norm: f64,
    /// Step size
    pub step_size: f64,
    /// Search direction norm
    pub direction_norm: f64,
    /// Line search performance
    pub line_search: LineSearchDiagnostic,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics,
    /// Time elapsed for this iteration
    pub iteration_time: Duration,
    /// Total time elapsed
    pub total_time: Duration,
    /// Memory usage (if available)
    pub memory_usage: Option<usize>,
}

/// Line search diagnostic information
#[derive(Debug, Clone)]
pub struct LineSearchDiagnostic {
    /// Number of function evaluations
    pub n_fev: usize,
    /// Number of gradient evaluations
    pub n_gev: usize,
    /// Final step length
    pub alpha: f64,
    /// Initial step length tried
    pub alpha_init: f64,
    /// Whether line search succeeded
    pub success: bool,
    /// Wolfe condition satisfaction
    pub wolfe_satisfied: (bool, bool), // (armijo, curvature)
}

/// Convergence metrics for an iteration
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Relative function change
    pub f_rel_change: f64,
    /// Absolute function change
    pub f_abs_change: f64,
    /// Relative gradient norm
    pub grad_rel_norm: f64,
    /// Step size relative to x
    pub x_rel_change: f64,
    /// First-order optimality measure
    pub optimality: f64,
    /// Estimated condition number
    pub condition_estimate: Option<f64>,
}

/// Overall performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total iterations
    pub total_nit: usize,
    /// Total function evaluations
    pub total_fev: usize,
    /// Total gradient evaluations
    pub total_gev: usize,
    /// Total Hessian evaluations (if applicable)
    pub total_hev: usize,
    /// Total wall time
    pub total_time: Duration,
    /// Average iteration time
    pub avg_iteration_time: Duration,
    /// Function evaluation rate (per second)
    pub fev_rate: f64,
    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
}

/// Efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Progress per function evaluation
    pub progress_per_fev: f64,
    /// Progress per gradient evaluation
    pub progress_per_gev: f64,
    /// Line search efficiency (average alpha/alpha_init)
    pub line_search_efficiency: f64,
    /// Step acceptance rate
    pub step_acceptance_rate: f64,
    /// Average reduction ratio
    pub avg_reduction_ratio: f64,
}

/// Convergence analysis results
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Convergence rate estimate
    pub convergence_rate: ConvergenceRate,
    /// Predicted iterations to convergence
    pub predicted_nit: Option<usize>,
    /// Convergence confidence score (0-1)
    pub confidence_score: f64,
    /// Detected convergence phase
    pub convergence_phase: ConvergencePhase,
    /// Stagnation analysis
    pub stagnation: StagnationAnalysis,
}

/// Convergence rate classification
#[derive(Debug, Clone)]
pub enum ConvergenceRate {
    /// Superlinear convergence (>1.5)
    Superlinear(f64),
    /// Linear convergence (rate in [0,1))
    Linear(f64),
    /// Sublinear convergence
    Sublinear(f64),
    /// No clear convergence pattern
    Unclear,
}

/// Convergence phase detection
#[derive(Debug, Clone, Copy)]
pub enum ConvergencePhase {
    /// Initial exploration phase
    Exploration,
    /// Rapid descent phase
    RapidDescent,
    /// Fine-tuning phase
    FineTuning,
    /// Converged
    Converged,
    /// Stagnated
    Stagnated,
}

/// Stagnation analysis
#[derive(Debug, Clone)]
pub struct StagnationAnalysis {
    /// Whether stagnation is detected
    pub is_stagnated: bool,
    /// Number of stagnant iterations
    pub stagnant_nit: usize,
    /// Stagnation type
    pub stagnation_type: StagnationType,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of stagnation
#[derive(Debug, Clone, Copy)]
pub enum StagnationType {
    /// No stagnation
    None,
    /// Function value plateau
    FunctionPlateau,
    /// Gradient plateau
    GradientPlateau,
    /// Oscillation detected
    Oscillation,
    /// Line search failures
    LineSearchFailure,
    /// Numerical precision limit
    NumericalLimit,
}

/// Problem analysis results
#[derive(Debug, Clone)]
pub struct ProblemAnalysis {
    /// Estimated problem difficulty
    pub difficulty: ProblemDifficulty,
    /// Condition number estimates
    pub conditioning: ConditioningAnalysis,
    /// Noise level estimation
    pub noise_level: Option<f64>,
    /// Detected problem features
    pub features: Vec<ProblemFeature>,
}

/// Problem difficulty classification
#[derive(Debug, Clone, Copy)]
pub enum ProblemDifficulty {
    /// Easy problem (well-conditioned, smooth)
    Easy,
    /// Moderate difficulty
    Moderate,
    /// Difficult problem
    Difficult,
    /// Very difficult (ill-conditioned, noisy, etc.)
    VeryDifficult,
}

/// Conditioning analysis
#[derive(Debug, Clone)]
pub struct ConditioningAnalysis {
    /// Estimated condition number
    pub condition_number: Option<f64>,
    /// Condition number history
    pub condition_history: Vec<f64>,
    /// Ill-conditioning detected
    pub is_ill_conditioned: bool,
}

/// Detected problem features
#[derive(Debug, Clone, Copy)]
pub enum ProblemFeature {
    /// Well-conditioned
    WellConditioned,
    /// Ill-conditioned
    IllConditioned,
    /// Noisy objective function
    Noisy,
    /// Non-smooth regions detected
    NonSmooth,
    /// Strong nonlinearity
    StronglyNonlinear,
    /// Multiple scales
    MultiScale,
    /// Narrow valleys
    NarrowValleys,
}

/// Diagnostic warnings and recommendations
#[derive(Debug, Clone)]
pub struct DiagnosticWarning {
    /// Warning severity
    pub severity: WarningSeverity,
    /// Warning message
    pub message: String,
    /// Iteration when warning was generated
    pub iteration: usize,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Warning severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningSeverity {
    /// Informational
    Info,
    /// Minor issue
    Minor,
    /// Moderate issue
    Moderate,
    /// Severe issue
    Severe,
}

/// Diagnostic collector for gathering convergence information
#[derive(Debug)]
pub struct DiagnosticCollector {
    /// Options
    options: DiagnosticOptions,
    /// Iteration diagnostics
    nit: Vec<IterationDiagnostic>,
    /// Start time
    start_time: Instant,
    /// Function value history
    f_history: VecDeque<f64>,
    /// Gradient norm history
    grad_history: VecDeque<f64>,
    /// Step size history
    step_history: VecDeque<f64>,
    /// Condition number estimates
    condition_history: VecDeque<f64>,
    /// Current iteration
    current_iteration: usize,
}

/// Options for diagnostic collection
#[derive(Debug, Clone)]
pub struct DiagnosticOptions {
    /// Enable detailed iteration tracking
    pub track_nit: bool,
    /// Enable condition number estimation
    pub estimate_conditioning: bool,
    /// Enable noise level detection
    pub detect_noise: bool,
    /// Enable convergence rate analysis
    pub analyze_convergence_rate: bool,
    /// History window size
    pub history_window: usize,
    /// Enable memory tracking
    pub track_memory: bool,
    /// Export format for visualization data
    pub export_format: ExportFormat,
}

/// Export format for diagnostic data
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Python numpy format
    Numpy,
}

impl Default for DiagnosticOptions {
    fn default() -> Self {
        Self {
            track_nit: true,
            estimate_conditioning: true,
            detect_noise: true,
            analyze_convergence_rate: true,
            history_window: 50,
            track_memory: false,
            export_format: ExportFormat::Json,
        }
    }
}

impl DiagnosticCollector {
    /// Create a new diagnostic collector
    pub fn new(options: DiagnosticOptions) -> Self {
        let history_window = options.history_window;
        Self {
            options,
            nit: Vec::new(),
            start_time: Instant::now(),
            f_history: VecDeque::with_capacity(history_window),
            grad_history: VecDeque::with_capacity(history_window),
            step_history: VecDeque::with_capacity(history_window),
            condition_history: VecDeque::with_capacity(history_window),
            current_iteration: 0,
        }
    }

    /// Record an iteration
    pub fn record_iteration(
        &mut self,
        f_value: f64,
        grad: &ArrayView1<f64>,
        step: &ArrayView1<f64>,
        direction: &ArrayView1<f64>,
        line_search_info: LineSearchDiagnostic,
    ) {
        let iteration_start = Instant::now();

        // Update histories
        self.update_history(
            f_value,
            grad.mapv(|x| x.abs()).sum(),
            step.mapv(|x| x.abs()).sum(),
        );

        // Compute convergence metrics
        let convergence_metrics = self.compute_convergence_metrics(f_value, grad, step);

        // Create iteration diagnostic
        let diagnostic = IterationDiagnostic {
            iteration: self.current_iteration,
            f_value,
            grad_norm: grad.mapv(|x| x.abs()).sum(),
            step_size: step.mapv(|x| x.abs()).sum(),
            direction_norm: direction.mapv(|x| x.abs()).sum(),
            line_search: line_search_info,
            convergence_metrics,
            iteration_time: iteration_start.elapsed(),
            total_time: self.start_time.elapsed(),
            memory_usage: None, // Could be implemented with system calls
        };

        self.nit.push(diagnostic);
        self.current_iteration += 1;
    }

    /// Update history buffers
    fn update_history(&mut self, f_value: f64, grad_norm: f64, step_size: f64) {
        self.f_history.push_back(f_value);
        if self.f_history.len() > self.options.history_window {
            self.f_history.pop_front();
        }

        self.grad_history.push_back(grad_norm);
        if self.grad_history.len() > self.options.history_window {
            self.grad_history.pop_front();
        }

        self.step_history.push_back(step_size);
        if self.step_history.len() > self.options.history_window {
            self.step_history.pop_front();
        }
    }

    /// Compute convergence metrics for current iteration
    fn compute_convergence_metrics(
        &self,
        f_value: f64,
        grad: &ArrayView1<f64>,
        step: &ArrayView1<f64>,
    ) -> ConvergenceMetrics {
        let f_prev = self.f_history.back().copied().unwrap_or(f_value);
        let f_abs_change = (f_value - f_prev).abs();
        let f_rel_change = if f_prev.abs() > 1e-10 {
            f_abs_change / f_prev.abs()
        } else {
            f_abs_change
        };

        let grad_norm = grad.mapv(|x| x.abs()).sum();
        let grad_rel_norm = grad_norm / (1.0 + f_value.abs());

        let step_norm = step.mapv(|x| x.abs()).sum();
        let x_norm = 1.0; // Would need current x to compute properly
        let x_rel_change = step_norm / (1.0 + x_norm);

        let optimality = grad_norm;

        ConvergenceMetrics {
            f_rel_change,
            f_abs_change,
            grad_rel_norm,
            x_rel_change,
            optimality,
            condition_estimate: None, // Would require Hessian estimation
        }
    }

    /// Finalize and produce complete diagnostics
    pub fn finalize(self) -> ConvergenceDiagnostics {
        let performance_metrics = self.compute_performance_metrics();
        let convergence_analysis = self.analyze_convergence();
        let problem_analysis = self.analyze_problem();
        let warnings = self.generate_warnings();

        ConvergenceDiagnostics {
            nit: self.nit,
            performance_metrics,
            convergence_analysis,
            problem_analysis,
            warnings,
        }
    }

    /// Compute overall performance metrics
    fn compute_performance_metrics(&self) -> PerformanceMetrics {
        let total_nit = self.nit.len();
        let total_time = self.start_time.elapsed();

        let total_fev = self.nit.iter().map(|it| it.line_search.n_fev).sum();
        let total_gev = self.nit.iter().map(|it| it.line_search.n_gev).sum();

        let avg_iteration_time = if total_nit > 0 {
            total_time / total_nit as u32
        } else {
            Duration::from_secs(0)
        };

        let fev_rate = if total_time.as_secs_f64() > 0.0 {
            total_fev as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let efficiency = self.compute_efficiency_metrics();

        PerformanceMetrics {
            total_nit,
            total_fev,
            total_gev,
            total_hev: 0, // Not tracked in this implementation
            total_time,
            avg_iteration_time,
            fev_rate,
            efficiency,
        }
    }

    /// Compute efficiency metrics
    fn compute_efficiency_metrics(&self) -> EfficiencyMetrics {
        let total_progress =
            if let (Some(first), Some(last)) = (self.f_history.front(), self.f_history.back()) {
                (first - last).abs()
            } else {
                0.0
            };

        let total_fev: usize = self.nit.iter().map(|it| it.line_search.n_fev).sum();
        let total_gev: usize = self.nit.iter().map(|it| it.line_search.n_gev).sum();

        let progress_per_fev = if total_fev > 0 {
            total_progress / total_fev as f64
        } else {
            0.0
        };

        let progress_per_gev = if total_gev > 0 {
            total_progress / total_gev as f64
        } else {
            0.0
        };

        let line_search_efficiency = self
            .nit
            .iter()
            .filter(|it| it.line_search.alpha_init > 0.0)
            .map(|it| it.line_search.alpha / it.line_search.alpha_init)
            .sum::<f64>()
            / self.nit.len().max(1) as f64;

        let step_acceptance_rate = self.nit.iter().filter(|it| it.line_search.success).count()
            as f64
            / self.nit.len().max(1) as f64;

        let avg_reduction_ratio = self
            .nit
            .iter()
            .map(|it| it.convergence_metrics.f_rel_change)
            .sum::<f64>()
            / self.nit.len().max(1) as f64;

        EfficiencyMetrics {
            progress_per_fev,
            progress_per_gev,
            line_search_efficiency,
            step_acceptance_rate,
            avg_reduction_ratio,
        }
    }

    /// Analyze convergence behavior
    fn analyze_convergence(&self) -> ConvergenceAnalysis {
        let convergence_rate = self.estimate_convergence_rate();
        let predicted_nit = self.predict_iterations_to_convergence();
        let confidence_score = self.compute_confidence_score();
        let convergence_phase = self.detect_convergence_phase();
        let stagnation = self.analyze_stagnation();

        ConvergenceAnalysis {
            convergence_rate,
            predicted_nit,
            confidence_score,
            convergence_phase,
            stagnation,
        }
    }

    /// Estimate convergence rate
    fn estimate_convergence_rate(&self) -> ConvergenceRate {
        if self.f_history.len() < 3 {
            return ConvergenceRate::Unclear;
        }

        // Use last few iterations to estimate rate
        let window = 5.min(self.f_history.len());
        let recent_f: Vec<f64> = self.f_history.iter().rev().take(window).copied().collect();

        // Simple linear regression on log scale
        if let Some(rate) = self.compute_rate_from_history(&recent_f) {
            if rate > 1.5 {
                ConvergenceRate::Superlinear(rate)
            } else if rate > 0.0 && rate < 1.0 {
                ConvergenceRate::Linear(rate)
            } else if rate > 0.0 {
                ConvergenceRate::Sublinear(rate)
            } else {
                ConvergenceRate::Unclear
            }
        } else {
            ConvergenceRate::Unclear
        }
    }

    /// Compute convergence rate from history
    fn compute_rate_from_history(&self, values: &[f64]) -> Option<f64> {
        if values.len() < 2 {
            return None;
        }

        // Simple rate estimation
        let mut rates = Vec::new();
        for i in 1..values.len() {
            if values[i - 1] > 0.0 && values[i] > 0.0 {
                let rate = values[i].ln() / values[i - 1].ln();
                if rate.is_finite() {
                    rates.push(rate);
                }
            }
        }

        if rates.is_empty() {
            None
        } else {
            Some(rates.iter().sum::<f64>() / rates.len() as f64)
        }
    }

    /// Predict iterations to convergence
    fn predict_iterations_to_convergence(&self) -> Option<usize> {
        // Simple prediction based on current rate
        if let Some(last_f) = self.f_history.back() {
            if let Some(last_grad) = self.grad_history.back() {
                if *last_grad < 1e-5 {
                    return Some(0); // Already converged
                }

                // Estimate based on current reduction rate
                if self.nit.len() > 2 {
                    let recent_rate = self.nit.last()?.convergence_metrics.f_rel_change;
                    if recent_rate > 0.0 {
                        let iterations_needed = (last_f.ln() / recent_rate).ceil() as usize;
                        return Some(iterations_needed);
                    }
                }
            }
        }
        None
    }

    /// Compute confidence score
    fn compute_confidence_score(&self) -> f64 {
        // Based on consistency of convergence behavior
        if self.nit.len() < 5 {
            return 0.0;
        }

        let mut score = 1.0;

        // Check for consistent function decrease
        let decreasing_count = self.count_decreasing_pairs(&self.f_history);
        score *= decreasing_count as f64 / self.f_history.len().max(1) as f64;

        // Check for consistent gradient decrease
        let grad_decreasing = self.count_decreasing_pairs(&self.grad_history);
        score *= grad_decreasing as f64 / self.grad_history.len().max(1) as f64;

        score
    }

    /// Count decreasing pairs in a VecDeque
    fn count_decreasing_pairs(&self, values: &VecDeque<f64>) -> usize {
        let mut count = 0;
        for i in 1..values.len() {
            if values[i] < values[i - 1] {
                count += 1;
            }
        }
        count
    }

    /// Check if function values are stagnant
    fn is_f_stagnant(&self) -> bool {
        if self.f_history.len() < 2 {
            return false;
        }

        let mut count = 0;
        let n = self.f_history.len().min(6); // Check last 5 pairs
        for i in (self.f_history.len() - n + 1)..self.f_history.len() {
            if (self.f_history[i] - self.f_history[i - 1]).abs() < 1e-12 {
                count += 1;
            }
        }

        count >= n - 1 // All recent pairs are stagnant
    }

    /// Detect current convergence phase
    fn detect_convergence_phase(&self) -> ConvergencePhase {
        if self.nit.is_empty() {
            return ConvergencePhase::Exploration;
        }

        let last_grad = self.grad_history.back().copied().unwrap_or(1.0);
        let last_f_change = self
            .nit
            .last()
            .map(|it| it.convergence_metrics.f_rel_change)
            .unwrap_or(1.0);

        if last_grad < 1e-8 && last_f_change < 1e-10 {
            ConvergencePhase::Converged
        } else if self.is_stagnated() {
            ConvergencePhase::Stagnated
        } else if last_f_change > 1e-3 {
            ConvergencePhase::RapidDescent
        } else if last_grad < 1e-3 {
            ConvergencePhase::FineTuning
        } else {
            ConvergencePhase::Exploration
        }
    }

    /// Check if optimization is stagnated
    fn is_stagnated(&self) -> bool {
        if self.f_history.len() < 5 {
            return false;
        }

        let recent = self.f_history.iter().rev().take(5);
        let values: Vec<f64> = recent.copied().collect();
        let max_diff = values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        max_diff < 1e-12
    }

    /// Analyze stagnation
    fn analyze_stagnation(&self) -> StagnationAnalysis {
        let is_stagnated = self.is_stagnated();
        let stagnant_nit = self.count_stagnant_nit();
        let stagnation_type = self.detect_stagnation_type();
        let recommendations = self.generate_stagnation_recommendations(&stagnation_type);

        StagnationAnalysis {
            is_stagnated,
            stagnant_nit,
            stagnation_type,
            recommendations,
        }
    }

    /// Count stagnant iterations
    fn count_stagnant_nit(&self) -> usize {
        let mut count = 0;
        for it in self.nit.iter().rev() {
            if it.convergence_metrics.f_rel_change < 1e-10 {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Detect type of stagnation
    fn detect_stagnation_type(&self) -> StagnationType {
        if !self.is_stagnated() {
            return StagnationType::None;
        }

        // Check for oscillation
        if self.is_oscillating() {
            return StagnationType::Oscillation;
        }

        // Check for line search failures
        let ls_failures = self
            .nit
            .iter()
            .rev()
            .take(5)
            .filter(|it| !it.line_search.success)
            .count();
        if ls_failures >= 3 {
            return StagnationType::LineSearchFailure;
        }

        // Check gradient vs function stagnation
        let grad_stagnant = self.grad_history.iter().rev().take(5).all(|&g| g < 1e-10);
        let f_stagnant = self.is_f_stagnant();

        if grad_stagnant && f_stagnant {
            StagnationType::NumericalLimit
        } else if f_stagnant {
            StagnationType::FunctionPlateau
        } else if grad_stagnant {
            StagnationType::GradientPlateau
        } else {
            StagnationType::None
        }
    }

    /// Check for oscillation
    fn is_oscillating(&self) -> bool {
        if self.f_history.len() < 4 {
            return false;
        }

        let mut sign_changes = 0;
        let mut diffs = Vec::new();

        // Compute differences
        for i in 1..self.f_history.len() {
            diffs.push(self.f_history[i] - self.f_history[i - 1]);
        }

        for i in 1..diffs.len() {
            if diffs[i - 1] * diffs[i] < 0.0 {
                sign_changes += 1;
            }
        }

        sign_changes >= diffs.len() / 2
    }

    /// Generate recommendations for stagnation
    fn generate_stagnation_recommendations(&self, stagnation_type: &StagnationType) -> Vec<String> {
        match stagnation_type {
            StagnationType::None => vec![],
            StagnationType::FunctionPlateau => vec![
                "Consider tightening convergence tolerances".to_string(),
                "Try a different optimization algorithm".to_string(),
                "Check if at a saddle point".to_string(),
            ],
            StagnationType::GradientPlateau => vec![
                "May be near optimum with flat gradient".to_string(),
                "Consider using second-order methods".to_string(),
            ],
            StagnationType::Oscillation => vec![
                "Reduce step size or learning rate".to_string(),
                "Use momentum or averaging".to_string(),
                "Consider non-monotone line search".to_string(),
            ],
            StagnationType::LineSearchFailure => vec![
                "Relax line search parameters".to_string(),
                "Use trust region methods".to_string(),
                "Check gradient computation".to_string(),
            ],
            StagnationType::NumericalLimit => vec![
                "Reached numerical precision limit".to_string(),
                "Consider using higher precision arithmetic".to_string(),
                "Current solution may be optimal within tolerance".to_string(),
            ],
        }
    }

    /// Analyze problem characteristics
    fn analyze_problem(&self) -> ProblemAnalysis {
        let difficulty = self.assess_difficulty();
        let conditioning = self.analyze_conditioning();
        let noise_level = self.estimate_noise_level();
        let features = self.detect_problem_features();

        ProblemAnalysis {
            difficulty,
            conditioning,
            noise_level,
            features,
        }
    }

    /// Assess problem difficulty
    fn assess_difficulty(&self) -> ProblemDifficulty {
        let mut difficulty_score = 0.0;

        // Factor 1: Convergence rate
        if matches!(
            self.estimate_convergence_rate(),
            ConvergenceRate::Sublinear(_) | ConvergenceRate::Unclear
        ) {
            difficulty_score += 2.0;
        }

        // Factor 2: Line search performance
        let ls_efficiency = self
            .nit
            .iter()
            .map(|it| it.line_search.alpha / it.line_search.alpha_init.max(1e-10))
            .sum::<f64>()
            / self.nit.len().max(1) as f64;
        if ls_efficiency < 0.1 {
            difficulty_score += 2.0;
        } else if ls_efficiency < 0.5 {
            difficulty_score += 1.0;
        }

        // Factor 3: Stagnation
        if self.is_stagnated() {
            difficulty_score += 1.5;
        }

        // Factor 4: Oscillation
        if self.is_oscillating() {
            difficulty_score += 1.5;
        }

        if difficulty_score < 1.0 {
            ProblemDifficulty::Easy
        } else if difficulty_score < 3.0 {
            ProblemDifficulty::Moderate
        } else if difficulty_score < 5.0 {
            ProblemDifficulty::Difficult
        } else {
            ProblemDifficulty::VeryDifficult
        }
    }

    /// Analyze conditioning
    fn analyze_conditioning(&self) -> ConditioningAnalysis {
        let condition_history = self.condition_history.iter().copied().collect();
        let condition_number = self.condition_history.back().copied();
        let is_ill_conditioned = condition_number.map(|c| c > 1e6).unwrap_or(false);

        ConditioningAnalysis {
            condition_number,
            condition_history,
            is_ill_conditioned,
        }
    }

    /// Estimate noise level
    fn estimate_noise_level(&self) -> Option<f64> {
        if self.f_history.len() < 10 {
            return None;
        }

        // Simple noise estimation: look at variance in function values
        let mean = self.f_history.iter().sum::<f64>() / self.f_history.len() as f64;
        let variance = self
            .f_history
            .iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f64>()
            / self.f_history.len() as f64;

        Some(variance.sqrt())
    }

    /// Detect problem features
    fn detect_problem_features(&self) -> Vec<ProblemFeature> {
        let mut features = Vec::new();

        // Check conditioning
        if let Some(cond) = self.condition_history.back() {
            if *cond < 100.0 {
                features.push(ProblemFeature::WellConditioned);
            } else if *cond > 1e6 {
                features.push(ProblemFeature::IllConditioned);
            }
        }

        // Check for noise
        if let Some(noise) = self.estimate_noise_level() {
            if noise > 1e-3 {
                features.push(ProblemFeature::Noisy);
            }
        }

        // Check for strong nonlinearity (high variation in step sizes)
        if self.step_history.len() > 5 {
            let step_variance = self.compute_variance(&self.step_history);
            if step_variance > 10.0 {
                features.push(ProblemFeature::StronglyNonlinear);
            }
        }

        // Check for narrow valleys (small steps with large gradients)
        if self.nit.len() > 5 {
            let narrow_valley_indicators = self
                .nit
                .iter()
                .filter(|it| it.step_size < 1e-3 && it.grad_norm > 1.0)
                .count();
            if narrow_valley_indicators > self.nit.len() / 2 {
                features.push(ProblemFeature::NarrowValleys);
            }
        }

        features
    }

    /// Compute variance of a collection
    fn compute_variance(&self, values: &VecDeque<f64>) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Generate warnings
    fn generate_warnings(&self) -> Vec<DiagnosticWarning> {
        let mut warnings = Vec::new();

        // Check for poor line search performance
        if let Some(last_it) = self.nit.last() {
            if !last_it.line_search.success {
                warnings.push(DiagnosticWarning {
                    severity: WarningSeverity::Moderate,
                    message: "Line search failed in last iteration".to_string(),
                    iteration: last_it.iteration,
                    recommendations: vec![
                        "Consider relaxing line search parameters".to_string(),
                        "Check gradient computation accuracy".to_string(),
                    ],
                });
            }
        }

        // Check for stagnation
        if self.is_stagnated() {
            warnings.push(DiagnosticWarning {
                severity: WarningSeverity::Severe,
                message: "Optimization appears to be stagnated".to_string(),
                iteration: self.current_iteration,
                recommendations: vec![
                    "Consider restarting with different initial point".to_string(),
                    "Try a different optimization algorithm".to_string(),
                    "Check problem formulation".to_string(),
                ],
            });
        }

        // Check for numerical issues
        if self
            .grad_history
            .back()
            .map(|&g| g < 1e-15)
            .unwrap_or(false)
        {
            warnings.push(DiagnosticWarning {
                severity: WarningSeverity::Info,
                message: "Gradient norm is near machine precision".to_string(),
                iteration: self.current_iteration,
                recommendations: vec![
                    "Solution may be optimal within numerical precision".to_string()
                ],
            });
        }

        warnings
    }
}

/// Export diagnostics to various formats
impl ConvergenceDiagnostics {
    /// Export to JSON format
    pub fn to_json(&self) -> Result<String, OptimizeError> {
        // In real implementation, would use serde_json
        Ok(format!(
            "{{\"total_nit\": {}}}",
            self.performance_metrics.total_nit
        ))
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> Result<String, OptimizeError> {
        let mut csv = String::from("iteration,f_value,grad_norm,step_size\n");
        for it in &self.nit {
            csv.push_str(&format!(
                "{},{},{},{}\n",
                it.iteration, it.f_value, it.grad_norm, it.step_size
            ));
        }
        Ok(csv)
    }

    /// Export iteration history as arrays for plotting
    pub fn to_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let f_values: Vec<f64> = self.nit.iter().map(|it| it.f_value).collect();
        let grad_norms: Vec<f64> = self.nit.iter().map(|it| it.grad_norm).collect();
        let step_sizes: Vec<f64> = self.nit.iter().map(|it| it.step_size).collect();
        (f_values, grad_norms, step_sizes)
    }

    /// Generate summary report
    pub fn summary_report(&self) -> String {
        format!(
            "Optimization Summary:\n\
             Total nit: {}\n\
             Total function evaluations: {}\n\
             Total time: {:?}\n\
             Final function value: {:.6e}\n\
             Final gradient norm: {:.6e}\n\
             Convergence status: {:?}\n\
             Problem difficulty: {:?}\n\
             Warnings: {}",
            self.performance_metrics.total_nit,
            self.performance_metrics.total_fev,
            self.performance_metrics.total_time,
            self.nit.last().map(|it| it.f_value).unwrap_or(0.0),
            self.nit.last().map(|it| it.grad_norm).unwrap_or(0.0),
            self.convergence_analysis.convergence_phase,
            self.problem_analysis.difficulty,
            self.warnings.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagnosticCollector::new(DiagnosticOptions::default());

        // Simulate some iterations
        let grad = Array1::from_vec(vec![1.0, 2.0]);
        let step = Array1::from_vec(vec![0.1, 0.2]);
        let direction = Array1::from_vec(vec![-1.0, -2.0]);

        let ls_info = LineSearchDiagnostic {
            n_fev: 3,
            n_gev: 1,
            alpha: 0.5,
            alpha_init: 1.0,
            success: true,
            wolfe_satisfied: (true, true),
        };

        collector.record_iteration(10.0, &grad.view(), &step.view(), &direction.view(), ls_info);

        let diagnostics = collector.finalize();
        assert_eq!(diagnostics.nit.len(), 1);
        assert_eq!(diagnostics.performance_metrics.total_nit, 1);
    }

    #[test]
    fn test_convergence_rate_estimation() {
        let collector = DiagnosticCollector::new(DiagnosticOptions::default());

        // Test rate computation
        let values = vec![1.0, 0.1, 0.01, 0.001];
        let rate = collector.compute_rate_from_history(&values);
        assert!(rate.is_some());
    }
}
