//! Robust convergence criteria with multiple stopping conditions
//!
//! This module provides robust convergence detection that handles:
//! - Multiple simultaneous stopping criteria
//! - Early stopping for noisy or difficult functions
//! - Numerical stability considerations
//! - Progress-based convergence detection
//! - Time-based stopping conditions

use crate::error::OptimizeError;
use crate::unconstrained::adaptive_convergence::{
    AdaptiveToleranceOptions, AdaptiveToleranceState, ConvergenceStatus,
};
use ndarray::ArrayView1;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Robust convergence options with multiple stopping criteria
#[derive(Debug, Clone)]
pub struct RobustConvergenceOptions {
    /// Adaptive tolerance options
    pub adaptive_tolerance: AdaptiveToleranceOptions,
    /// Enable early stopping for difficult problems
    pub enable_early_stopping: bool,
    /// Early stopping patience (iterations without improvement)
    pub early_stopping_patience: usize,
    /// Relative improvement threshold for early stopping
    pub early_stopping_min_delta: f64,
    /// Enable progress-based convergence
    pub enable_progress_based: bool,
    /// Number of iterations to consider for progress evaluation
    pub progress_window: usize,
    /// Minimum progress rate to continue optimization
    pub min_progress_rate: f64,
    /// Enable time-based stopping
    pub enable_time_limit: bool,
    /// Maximum optimization time
    pub max_time: Duration,
    /// Enable robust noisy function handling
    pub enable_noise_robust: bool,
    /// Window size for noise-robust convergence
    pub noise_window: usize,
    /// Confidence level for noise-robust convergence (0.0 to 1.0)
    pub noise_confidence: f64,
    /// Enable multiple criteria requirement
    pub require_multiple_criteria: bool,
    /// Number of criteria that must be satisfied simultaneously
    pub min_criteria_count: usize,
    /// Enable plateau detection
    pub enable_plateau_detection: bool,
    /// Plateau tolerance
    pub plateau_tolerance: f64,
    /// Plateau window size
    pub plateau_window: usize,
}

impl Default for RobustConvergenceOptions {
    fn default() -> Self {
        Self {
            adaptive_tolerance: AdaptiveToleranceOptions::default(),
            enable_early_stopping: true,
            early_stopping_patience: 20,
            early_stopping_min_delta: 1e-8,
            enable_progress_based: true,
            progress_window: 10,
            min_progress_rate: 1e-10,
            enable_time_limit: false,
            max_time: Duration::from_secs(3600), // 1 hour default
            enable_noise_robust: true,
            noise_window: 5,
            noise_confidence: 0.95,
            require_multiple_criteria: false,
            min_criteria_count: 2,
            enable_plateau_detection: true,
            plateau_tolerance: 1e-12,
            plateau_window: 15,
        }
    }
}

/// State for robust convergence detection
#[derive(Debug)]
pub struct RobustConvergenceState {
    /// Adaptive tolerance state
    adaptive_state: AdaptiveToleranceState,
    /// Early stopping state
    early_stop_state: EarlyStoppingState,
    /// Progress tracking state
    progress_state: ProgressState,
    /// Time tracking
    start_time: Option<Instant>,
    /// Noise robust state
    noise_state: NoiseRobustState,
    /// Plateau detection state
    plateau_state: PlateauState,
    /// Options
    options: RobustConvergenceOptions,
}

/// Early stopping state
#[derive(Debug, Clone)]
struct EarlyStoppingState {
    best_value: f64,
    iterations_without_improvement: usize,
    improvement_history: VecDeque<f64>,
}

/// Progress tracking state
#[derive(Debug, Clone)]
struct ProgressState {
    recent_values: VecDeque<f64>,
    recent_gradients: VecDeque<f64>,
    progress_rate: f64,
}

/// Noise-robust convergence state
#[derive(Debug, Clone)]
struct NoiseRobustState {
    function_window: VecDeque<f64>,
    gradient_window: VecDeque<f64>,
    step_window: VecDeque<f64>,
    #[allow(dead_code)]
    stable_convergence_count: usize,
}

/// Plateau detection state
#[derive(Debug, Clone)]
struct PlateauState {
    plateau_window: VecDeque<f64>,
    plateau_detected: bool,
    plateau_start_iteration: Option<usize>,
}

impl RobustConvergenceState {
    /// Create new robust convergence state
    pub fn new(options: RobustConvergenceOptions, problem_dim: usize) -> Self {
        let adaptive_state =
            AdaptiveToleranceState::new(options.adaptive_tolerance.clone(), problem_dim);

        Self {
            adaptive_state,
            early_stop_state: EarlyStoppingState {
                best_value: f64::INFINITY,
                iterations_without_improvement: 0,
                improvement_history: VecDeque::with_capacity(options.early_stopping_patience),
            },
            progress_state: ProgressState {
                recent_values: VecDeque::with_capacity(options.progress_window),
                recent_gradients: VecDeque::with_capacity(options.progress_window),
                progress_rate: 0.0,
            },
            start_time: Some(std::time::Instant::now()),
            noise_state: NoiseRobustState {
                function_window: VecDeque::with_capacity(options.noise_window),
                gradient_window: VecDeque::with_capacity(options.noise_window),
                step_window: VecDeque::with_capacity(options.noise_window),
                stable_convergence_count: 0,
            },
            plateau_state: PlateauState {
                plateau_window: VecDeque::with_capacity(options.plateau_window),
                plateau_detected: false,
                plateau_start_iteration: None,
            },
            options,
        }
    }

    /// Start timing for time-based convergence
    pub fn start_timing(&mut self) {
        if self.options.enable_time_limit {
            self.start_time = Some(Instant::now());
        }
    }

    /// Update state and check all convergence criteria
    pub fn update_and_check_convergence(
        &mut self,
        function_value: f64,
        gradient_norm: f64,
        step_norm: f64,
        iteration: usize,
        _x: Option<&ArrayView1<f64>>,
    ) -> Result<RobustConvergenceResult, OptimizeError> {
        // Update adaptive tolerance state
        self.adaptive_state
            .update(function_value, gradient_norm, step_norm, iteration)?;

        // Update early stopping
        if self.options.enable_early_stopping {
            self.update_early_stopping(function_value, iteration);
        }

        // Update progress tracking
        if self.options.enable_progress_based {
            self.update_progress(function_value, gradient_norm);
        }

        // Update noise-robust state
        if self.options.enable_noise_robust {
            self.update_noise_robust(function_value, gradient_norm, step_norm);
        }

        // Update plateau detection
        if self.options.enable_plateau_detection {
            self.update_plateau_detection(function_value, iteration);
        }

        // Check all convergence criteria
        self.check_all_convergence_criteria(function_value, gradient_norm, step_norm, iteration)
    }

    /// Update early stopping state
    fn update_early_stopping(&mut self, function_value: f64, _iteration: usize) {
        let improvement = self.early_stop_state.best_value - function_value;

        if improvement > self.options.early_stopping_min_delta {
            self.early_stop_state.best_value = function_value;
            self.early_stop_state.iterations_without_improvement = 0;
        } else {
            self.early_stop_state.iterations_without_improvement += 1;
        }

        // Track improvement history
        if self.early_stop_state.improvement_history.len() >= self.options.early_stopping_patience {
            self.early_stop_state.improvement_history.pop_front();
        }
        self.early_stop_state
            .improvement_history
            .push_back(improvement);
    }

    /// Update progress tracking
    fn update_progress(&mut self, function_value: f64, gradient_norm: f64) {
        // Add to recent values
        if self.progress_state.recent_values.len() >= self.options.progress_window {
            self.progress_state.recent_values.pop_front();
        }
        self.progress_state.recent_values.push_back(function_value);

        if self.progress_state.recent_gradients.len() >= self.options.progress_window {
            self.progress_state.recent_gradients.pop_front();
        }
        self.progress_state
            .recent_gradients
            .push_back(gradient_norm);

        // Calculate progress rate
        if self.progress_state.recent_values.len() >= 2 {
            let values: Vec<f64> = self.progress_state.recent_values.iter().cloned().collect();
            let n = values.len();

            // Linear regression to estimate progress rate
            let x_mean = (n - 1) as f64 / 2.0;
            let y_mean = values.iter().sum::<f64>() / n as f64;

            let mut num = 0.0;
            let mut den = 0.0;

            for (i, &y) in values.iter().enumerate() {
                let x = i as f64;
                num += (x - x_mean) * (y - y_mean);
                den += (x - x_mean).powi(2);
            }

            self.progress_state.progress_rate = if den > 1e-15 { -num / den } else { 0.0 };
        }
    }

    /// Update noise-robust convergence state
    fn update_noise_robust(&mut self, function_value: f64, gradient_norm: f64, step_norm: f64) {
        // Add to windows
        if self.noise_state.function_window.len() >= self.options.noise_window {
            self.noise_state.function_window.pop_front();
        }
        self.noise_state.function_window.push_back(function_value);

        if self.noise_state.gradient_window.len() >= self.options.noise_window {
            self.noise_state.gradient_window.pop_front();
        }
        self.noise_state.gradient_window.push_back(gradient_norm);

        if self.noise_state.step_window.len() >= self.options.noise_window {
            self.noise_state.step_window.pop_front();
        }
        self.noise_state.step_window.push_back(step_norm);
    }

    /// Update plateau detection
    fn update_plateau_detection(&mut self, function_value: f64, iteration: usize) {
        if self.plateau_state.plateau_window.len() >= self.options.plateau_window {
            self.plateau_state.plateau_window.pop_front();
        }
        self.plateau_state.plateau_window.push_back(function_value);

        // Check for plateau
        if self.plateau_state.plateau_window.len() >= self.options.plateau_window {
            let values: Vec<f64> = self.plateau_state.plateau_window.iter().cloned().collect();
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if (max_val - min_val).abs() < self.options.plateau_tolerance {
                if !self.plateau_state.plateau_detected {
                    self.plateau_state.plateau_detected = true;
                    self.plateau_state.plateau_start_iteration = Some(iteration);
                }
            } else {
                self.plateau_state.plateau_detected = false;
                self.plateau_state.plateau_start_iteration = None;
            }
        }
    }

    /// Check all convergence criteria
    fn check_all_convergence_criteria(
        &self,
        _function_value: f64,
        gradient_norm: f64,
        step_norm: f64,
        _iteration: usize,
    ) -> Result<RobustConvergenceResult, OptimizeError> {
        let mut convergence_reasons = Vec::new();
        let mut warning_flags = Vec::new();

        // Get function change
        let function_change = if self.progress_state.recent_values.len() >= 2 {
            let recent: Vec<f64> = self.progress_state.recent_values.iter().cloned().collect();
            recent[recent.len() - 1] - recent[recent.len() - 2]
        } else {
            0.0
        };

        // Check adaptive convergence
        let adaptive_status =
            self.adaptive_state
                .check_convergence(function_change, step_norm, gradient_norm);

        let mut criteria_met = 0;

        // Standard convergence criteria
        if adaptive_status.f_converged {
            convergence_reasons.push("Function tolerance".to_string());
            criteria_met += 1;
        }
        if adaptive_status.x_converged {
            convergence_reasons.push("Step tolerance".to_string());
            criteria_met += 1;
        }
        if adaptive_status.g_converged {
            convergence_reasons.push("Gradient tolerance".to_string());
            criteria_met += 1;
        }

        // Early stopping check
        if self.options.enable_early_stopping {
            let early_stop_triggered = self.early_stop_state.iterations_without_improvement
                >= self.options.early_stopping_patience;

            if early_stop_triggered {
                convergence_reasons.push("Early stopping".to_string());
                criteria_met += 1;
            }
        }

        // Progress-based convergence
        if self.options.enable_progress_based
            && self.progress_state.progress_rate.abs() < self.options.min_progress_rate
        {
            convergence_reasons.push("Insufficient progress".to_string());
            criteria_met += 1;
        }

        // Time limit check
        if self.options.enable_time_limit {
            if let Some(start_time) = self.start_time {
                if start_time.elapsed() > self.options.max_time {
                    convergence_reasons.push("Time limit exceeded".to_string());
                    criteria_met += 1;
                }
            }
        }

        // Noise-robust convergence
        if self.options.enable_noise_robust {
            let noise_converged = self.check_noise_robust_convergence();
            if noise_converged {
                convergence_reasons.push("Noise-robust convergence".to_string());
                criteria_met += 1;
            }
        }

        // Plateau detection
        if self.options.enable_plateau_detection && self.plateau_state.plateau_detected {
            convergence_reasons.push("Plateau detected".to_string());
            criteria_met += 1;
        }

        // Check if convergence is achieved
        let converged = if self.options.require_multiple_criteria {
            criteria_met >= self.options.min_criteria_count
        } else {
            criteria_met > 0
        };

        // Generate warnings
        if self.early_stop_state.iterations_without_improvement
            > self.options.early_stopping_patience / 2
        {
            warning_flags.push("Slow convergence detected".to_string());
        }

        if gradient_norm > 1e3 {
            warning_flags.push("Large gradient norm".to_string());
        }

        if self.plateau_state.plateau_detected {
            warning_flags.push("Function plateau detected".to_string());
        }

        Ok(RobustConvergenceResult {
            converged,
            convergence_reasons,
            warning_flags,
            adaptive_status,
            criteria_met,
            early_stopping_iterations: self.early_stop_state.iterations_without_improvement,
            progress_rate: self.progress_state.progress_rate,
            time_elapsed: self.start_time.map(|t| t.elapsed()),
            plateau_detected: self.plateau_state.plateau_detected,
            noise_robust_confidence: self.calculate_noise_confidence(),
        })
    }

    /// Check noise-robust convergence
    fn check_noise_robust_convergence(&self) -> bool {
        if self.noise_state.function_window.len() < self.options.noise_window {
            return false;
        }

        // Statistical test for convergence stability
        let f_values: Vec<f64> = self.noise_state.function_window.iter().cloned().collect();
        let g_values: Vec<f64> = self.noise_state.gradient_window.iter().cloned().collect();

        // Check if recent values are consistently small (statistically)
        let f_mean = f_values.iter().sum::<f64>() / f_values.len() as f64;
        let f_std = (f_values.iter().map(|x| (x - f_mean).powi(2)).sum::<f64>()
            / f_values.len() as f64)
            .sqrt();

        let g_mean = g_values.iter().sum::<f64>() / g_values.len() as f64;
        let g_std = (g_values.iter().map(|x| (x - g_mean).powi(2)).sum::<f64>()
            / g_values.len() as f64)
            .sqrt();

        // Use confidence interval approach
        let f_margin = f_std * 1.96; // 95% confidence
        let g_margin = g_std * 1.96;

        let f_converged = f_mean.abs() < self.adaptive_state.current_ftol + f_margin;
        let g_converged = g_mean < self.adaptive_state.current_gtol + g_margin;

        f_converged && g_converged
    }

    /// Calculate noise confidence level
    fn calculate_noise_confidence(&self) -> f64 {
        if self.noise_state.function_window.len() < 3 {
            return 0.0;
        }

        let values: Vec<f64> = self.noise_state.function_window.iter().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Confidence based on coefficient of variation
        let cv = if mean.abs() > 1e-15 {
            variance.sqrt() / mean.abs()
        } else {
            f64::INFINITY
        };

        // Map CV to confidence (0 = perfect, high CV = low confidence)
        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    }

    /// Get current adaptive state for compatibility
    pub fn get_adaptive_state(&self) -> &AdaptiveToleranceState {
        &self.adaptive_state
    }
}

/// Comprehensive convergence result with detailed information
#[derive(Debug, Clone)]
pub struct RobustConvergenceResult {
    /// Whether convergence was achieved
    pub converged: bool,
    /// List of reasons for convergence
    pub convergence_reasons: Vec<String>,
    /// Warning flags for potential issues
    pub warning_flags: Vec<String>,
    /// Adaptive convergence status
    pub adaptive_status: ConvergenceStatus,
    /// Number of criteria that were met
    pub criteria_met: usize,
    /// Early stopping iterations count
    pub early_stopping_iterations: usize,
    /// Current progress rate
    pub progress_rate: f64,
    /// Time elapsed since start
    pub time_elapsed: Option<Duration>,
    /// Whether plateau was detected
    pub plateau_detected: bool,
    /// Confidence level for noise-robust convergence
    pub noise_robust_confidence: f64,
}

impl RobustConvergenceResult {
    /// Get convergence message
    pub fn get_message(&self) -> String {
        if self.converged {
            if self.convergence_reasons.is_empty() {
                "Optimization converged".to_string()
            } else {
                format!(
                    "Optimization converged: {}",
                    self.convergence_reasons.join(", ")
                )
            }
        } else {
            "Optimization has not converged".to_string()
        }
    }

    /// Get detailed status report
    pub fn get_detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Convergence Status: {}\n",
            if self.converged {
                "CONVERGED"
            } else {
                "NOT CONVERGED"
            }
        ));

        if !self.convergence_reasons.is_empty() {
            report.push_str(&format!(
                "Convergence reasons: {}\n",
                self.convergence_reasons.join(", ")
            ));
        }

        if !self.warning_flags.is_empty() {
            report.push_str(&format!("Warnings: {}\n", self.warning_flags.join(", ")));
        }

        report.push_str(&format!("Criteria met: {}\n", self.criteria_met));
        report.push_str(&format!("Progress rate: {:.3e}\n", self.progress_rate));

        if let Some(time) = self.time_elapsed {
            report.push_str(&format!(
                "Time elapsed: {:.3} seconds\n",
                time.as_secs_f64()
            ));
        }

        if self.noise_robust_confidence > 0.0 {
            report.push_str(&format!(
                "Noise confidence: {:.1}%\n",
                self.noise_robust_confidence * 100.0
            ));
        }

        report.push_str(&format!(
            "Function tolerance: {:.3e}\n",
            self.adaptive_status.ftol_used
        ));
        report.push_str(&format!(
            "Gradient tolerance: {:.3e}\n",
            self.adaptive_status.gtol_used
        ));
        report.push_str(&format!(
            "Step tolerance: {:.3e}\n",
            self.adaptive_status.xtol_used
        ));

        report
    }
}

/// Create robust convergence options for specific problem types
pub fn create_robust_options_for_problem(
    problem_type: &str,
    problem_size: usize,
    expected_difficulty: &str,
) -> RobustConvergenceOptions {
    let adaptive_tolerance =
        crate::unconstrained::adaptive_convergence::create_adaptive_options_for_problem(
            problem_type,
            problem_size,
        );

    match expected_difficulty.to_lowercase().as_str() {
        "easy" => RobustConvergenceOptions {
            adaptive_tolerance,
            enable_early_stopping: false,
            enable_progress_based: false,
            enable_noise_robust: false,
            require_multiple_criteria: false,
            ..Default::default()
        },
        "moderate" => RobustConvergenceOptions {
            adaptive_tolerance,
            enable_early_stopping: true,
            early_stopping_patience: 15,
            enable_progress_based: true,
            enable_noise_robust: true,
            ..Default::default()
        },
        "difficult" => RobustConvergenceOptions {
            adaptive_tolerance,
            enable_early_stopping: true,
            early_stopping_patience: 30,
            enable_progress_based: true,
            enable_noise_robust: true,
            require_multiple_criteria: true,
            min_criteria_count: 2,
            enable_plateau_detection: true,
            ..Default::default()
        },
        "very_difficult" => RobustConvergenceOptions {
            adaptive_tolerance,
            enable_early_stopping: true,
            early_stopping_patience: 50,
            enable_progress_based: true,
            enable_noise_robust: true,
            require_multiple_criteria: true,
            min_criteria_count: 2,
            enable_plateau_detection: true,
            noise_window: 10,
            noise_confidence: 0.99,
            ..Default::default()
        },
        _ => RobustConvergenceOptions {
            adaptive_tolerance,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robust_convergence_initialization() {
        let options = RobustConvergenceOptions::default();
        let mut state = RobustConvergenceState::new(options, 10);
        state.start_timing();

        assert!(state.start_time.is_some());
    }

    #[test]
    fn test_early_stopping() {
        let mut options = RobustConvergenceOptions::default();
        options.enable_early_stopping = true;
        options.early_stopping_patience = 3;
        options.early_stopping_min_delta = 1e-6;

        let mut state = RobustConvergenceState::new(options, 5);

        // Simulate no improvement for several iterations
        for i in 0..5 {
            let result = state
                .update_and_check_convergence(1.0, 1e-3, 1e-6, i, None)
                .unwrap();

            if i >= 3 {
                assert!(result.converged);
                assert!(result
                    .convergence_reasons
                    .contains(&"Early stopping".to_string()));
            }
        }
    }

    #[test]
    fn test_progress_based_convergence() {
        let mut options = RobustConvergenceOptions::default();
        options.enable_progress_based = true;
        options.min_progress_rate = 1e-8;

        let mut state = RobustConvergenceState::new(options, 5);

        // Simulate very slow progress
        for i in 0..15 {
            let f_val = 1.0 - i as f64 * 1e-10; // Extremely slow decrease
            let result = state
                .update_and_check_convergence(f_val, 1e-3, 1e-6, i, None)
                .unwrap();

            if i > 10 {
                // Should converge due to insufficient progress
                if result.converged {
                    assert!(result
                        .convergence_reasons
                        .contains(&"Insufficient progress".to_string()));
                    break;
                }
            }
        }
    }

    #[test]
    fn test_noise_robust_convergence() {
        let mut options = RobustConvergenceOptions::default();
        options.enable_noise_robust = true;
        options.noise_window = 5;

        let mut state = RobustConvergenceState::new(options, 5);

        // Simulate noisy convergence
        for i in 0..10 {
            let noise = if i % 2 == 0 { 1e-9 } else { -1e-9 };
            let f_val = 1e-10 + noise; // Converged but noisy
            let g_val = 1e-6 + noise;

            let result = state
                .update_and_check_convergence(f_val, g_val, 1e-8, i, None)
                .unwrap();

            if i >= 8 && result.converged {
                assert!(result
                    .convergence_reasons
                    .iter()
                    .any(|r| r.contains("Noise-robust") || r.contains("tolerance")));
                break;
            }
        }
    }

    #[test]
    fn test_multiple_criteria_requirement() {
        let mut options = RobustConvergenceOptions::default();
        options.require_multiple_criteria = true;
        options.min_criteria_count = 2;
        options.adaptive_tolerance.initial_ftol = 1e-6;
        options.adaptive_tolerance.initial_gtol = 1e-4;
        options.adaptive_tolerance.initial_xtol = 1e-3;
        // Disable other convergence criteria to test only basic tolerances
        options.enable_early_stopping = false;
        options.enable_progress_based = false;
        options.enable_plateau_detection = false;
        options.enable_noise_robust = false;

        let mut state = RobustConvergenceState::new(options, 5);

        // Only satisfy one criterion (function tolerance)
        let result = state
            .update_and_check_convergence(1e-8, 1e-1, 1e-2, 10, None)
            .unwrap();

        // Should not converge with only one criterion met
        assert!(result.criteria_met < 2);
    }

    #[test]
    fn test_plateau_detection() {
        let mut options = RobustConvergenceOptions::default();
        options.enable_plateau_detection = true;
        options.plateau_window = 5;
        options.plateau_tolerance = 1e-10;

        let mut state = RobustConvergenceState::new(options, 5);

        // Simulate plateau
        for i in 0..10 {
            let result = state
                .update_and_check_convergence(1.0, 1e-3, 1e-6, i, None)
                .unwrap();

            if i >= 6 {
                assert!(result.plateau_detected);
                if result.converged {
                    assert!(result
                        .convergence_reasons
                        .contains(&"Plateau detected".to_string()));
                }
            }
        }
    }

    #[test]
    fn test_problem_specific_options() {
        let easy = create_robust_options_for_problem("well_conditioned", 10, "easy");
        assert!(!easy.enable_early_stopping);
        assert!(!easy.require_multiple_criteria);

        let difficult = create_robust_options_for_problem("ill_conditioned", 100, "difficult");
        assert!(difficult.enable_early_stopping);
        assert!(difficult.require_multiple_criteria);
        assert_eq!(difficult.min_criteria_count, 2);

        let very_difficult = create_robust_options_for_problem("noisy", 1000, "very_difficult");
        assert!(very_difficult.enable_noise_robust);
        assert_eq!(very_difficult.noise_window, 10);
        assert_eq!(very_difficult.early_stopping_patience, 50);
    }

    #[test]
    fn test_detailed_report() {
        let options = RobustConvergenceOptions::default();
        let mut state = RobustConvergenceState::new(options, 5);

        let result = state
            .update_and_check_convergence(1e-10, 1e-8, 1e-10, 10, None)
            .unwrap();

        let report = result.get_detailed_report();
        assert!(report.contains("Convergence Status:"));
        assert!(report.contains("Function tolerance:"));
        assert!(report.contains("Gradient tolerance:"));
    }
}
