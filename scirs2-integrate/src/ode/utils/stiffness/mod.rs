//! Stiffness detection utilities for ODE solvers
//!
//! This module provides algorithms and data structures for detecting stiffness
//! in ODE systems and making intelligent method switching decisions.

pub mod integration;

use crate::IntegrateFloat;
use ndarray::Array2;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Detection method for stiffness analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StiffnessDetectionMethod {
    /// Basic counter-based detection (original LSODA approach)
    Basic,
    /// Error pattern analysis
    ErrorPattern,
    /// Step size pattern analysis
    StepPattern,
    /// Eigenvalue estimation of the Jacobian
    EigenvalueEstimation,
    /// Combined approach using multiple indicators
    #[default]
    Combined,
}

/// Configuration for stiffness detection
#[derive(Debug, Clone)]
pub struct StiffnessDetectionConfig<F: IntegrateFloat> {
    /// Method used for stiffness detection
    pub method: StiffnessDetectionMethod,
    /// Minimum number of steps before considering method switch
    pub min_steps_before_switch: usize,
    /// Minimum number of indicators needed to detect stiffness
    pub stiffness_threshold: usize,
    /// Minimum number of indicators needed to detect non-stiffness
    pub non_stiffness_threshold: usize,
    /// Window size for analyzing error and step size patterns
    pub analysis_window: usize,
    /// Threshold for step size ratio indicating stiffness
    pub step_size_ratio_threshold: F,
    /// Threshold for error ratio indicating stiffness
    pub error_ratio_threshold: F,
    /// Whether to use eigenvalue estimation (more expensive)
    pub use_eigenvalue_estimation: bool,
    /// Minimum period between eigenvalue estimations
    pub eigenvalue_est_period: usize,
    /// Weight factors for different indicators
    pub indicator_weights: IndicatorWeights<F>,
    /// Phantom marker for the float type
    _phantom: PhantomData<F>,
}

impl<F: IntegrateFloat> Default for StiffnessDetectionConfig<F> {
    fn default() -> Self {
        StiffnessDetectionConfig {
            method: StiffnessDetectionMethod::default(),
            min_steps_before_switch: 5,
            stiffness_threshold: 3,
            non_stiffness_threshold: 5,
            analysis_window: 10,
            step_size_ratio_threshold: F::from_f64(0.1).unwrap(),
            error_ratio_threshold: F::from_f64(10.0).unwrap(),
            use_eigenvalue_estimation: false,
            eigenvalue_est_period: 25,
            indicator_weights: IndicatorWeights::default(),
            _phantom: PhantomData,
        }
    }
}

/// Weight factors for different stiffness indicators
#[derive(Debug, Clone)]
pub struct IndicatorWeights<F: IntegrateFloat> {
    /// Weight for error pattern indicators
    pub error_pattern_weight: F,
    /// Weight for step size pattern indicators
    pub step_pattern_weight: F,
    /// Weight for Newton convergence indicators
    pub newton_convergence_weight: F,
    /// Weight for eigenvalue estimation indicators
    pub eigenvalue_weight: F,
}

impl<F: IntegrateFloat> Default for IndicatorWeights<F> {
    fn default() -> Self {
        IndicatorWeights {
            error_pattern_weight: F::from_f64(1.0).unwrap(),
            step_pattern_weight: F::from_f64(1.0).unwrap(),
            newton_convergence_weight: F::from_f64(1.5).unwrap(),
            eigenvalue_weight: F::from_f64(2.0).unwrap(),
        }
    }
}

/// Enhanced stiffness detector that uses multiple indicators
#[derive(Debug, Clone)]
pub struct StiffnessDetector<F: IntegrateFloat> {
    /// Configuration for stiffness detection
    config: StiffnessDetectionConfig<F>,
    /// History of step sizes
    step_size_history: Vec<F>,
    /// History of error estimates
    error_history: Vec<F>,
    /// History of Newton iterations
    newton_iter_history: Vec<usize>,
    /// History of rejected steps
    rejected_step_history: Vec<bool>,
    /// Estimated stiffness ratio
    stiffness_ratio: F,
    /// Stiffness counter (number of indicators suggesting stiff problem)
    stiffness_indicators: usize,
    /// Non-stiffness counter (number of indicators suggesting non-stiff problem)
    non_stiffness_indicators: usize,
    /// Last time eigenvalues were estimated
    last_eigenvalue_est: usize,
    /// Current stiffness score (-1.0 to 1.0, where positive means stiff)
    stiffness_score: F,
}

impl<F: IntegrateFloat> Default for StiffnessDetector<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> StiffnessDetector<F> {
    /// Create a new stiffness detector with default configuration
    pub fn new() -> Self {
        Self::with_config(StiffnessDetectionConfig::default())
    }

    /// Create a new stiffness detector with specific configuration
    pub fn with_config(config: StiffnessDetectionConfig<F>) -> Self {
        StiffnessDetector {
            config,
            step_size_history: Vec::with_capacity(20),
            error_history: Vec::with_capacity(20),
            newton_iter_history: Vec::with_capacity(20),
            rejected_step_history: Vec::with_capacity(20),
            stiffness_ratio: F::zero(),
            stiffness_indicators: 0,
            non_stiffness_indicators: 0,
            last_eigenvalue_est: 0,
            stiffness_score: F::zero(),
        }
    }

    /// Record a step for stiffness analysis
    pub fn record_step(
        &mut self,
        step_size: F,
        error: F,
        newton_iterations: usize,
        rejected: bool,
        steps_taken: usize,
    ) {
        // Add to history
        self.step_size_history.push(step_size);
        self.error_history.push(error);
        self.newton_iter_history.push(newton_iterations);
        self.rejected_step_history.push(rejected);

        // Keep history limited to window size
        let window = self.config.analysis_window;
        if self.step_size_history.len() > window {
            self.step_size_history.remove(0);
            self.error_history.remove(0);
            self.newton_iter_history.remove(0);
            self.rejected_step_history.remove(0);
        }

        // Analyze patterns based on configuration
        match self.config.method {
            StiffnessDetectionMethod::Basic => self.analyze_basic(),
            StiffnessDetectionMethod::ErrorPattern => self.analyze_error_pattern(),
            StiffnessDetectionMethod::StepPattern => self.analyze_step_pattern(),
            StiffnessDetectionMethod::EigenvalueEstimation => {
                // Eigenvalue estimation would be done less frequently
                if steps_taken - self.last_eigenvalue_est >= self.config.eigenvalue_est_period {
                    // This would need the Jacobian, which we don't have here
                    // We'll just use a placeholder until integration with LSODA
                    self.last_eigenvalue_est = steps_taken;
                }
            }
            StiffnessDetectionMethod::Combined => {
                self.analyze_basic();
                self.analyze_error_pattern();
                self.analyze_step_pattern();

                if self.config.use_eigenvalue_estimation
                    && steps_taken - self.last_eigenvalue_est >= self.config.eigenvalue_est_period
                {
                    // Eigenvalue estimation would go here
                    self.last_eigenvalue_est = steps_taken;
                }

                // Update combined stiffness score
                self.update_stiffness_score();
            }
        }
    }

    /// Basic analysis looking at the most recent step
    fn analyze_basic(&mut self) {
        if self.step_size_history.is_empty() {
            return;
        }

        let last_idx = self.step_size_history.len() - 1;

        // Check for very small errors (indicating non-stiffness)
        if self.error_history[last_idx] < F::from_f64(0.01).unwrap() {
            self.non_stiffness_indicators += 1;
        }

        // Check for large errors (indicating stiffness)
        if self.error_history[last_idx] > self.config.error_ratio_threshold {
            self.stiffness_indicators += 1;
        }

        // Check Newton iterations (few iterations indicate non-stiffness)
        if self.newton_iter_history[last_idx] <= 2 {
            self.non_stiffness_indicators += 1;
        }

        // Check Newton iterations (many iterations indicate stiffness)
        if self.newton_iter_history[last_idx] >= 8 {
            self.stiffness_indicators += 1;
        }

        // Check rejected steps (frequent rejections may indicate stiffness)
        if self.rejected_step_history[last_idx] {
            self.stiffness_indicators += 1;
        }
    }

    /// Analyze error patterns for stiffness indicators
    fn analyze_error_pattern(&mut self) {
        if self.error_history.len() < 3 {
            return;
        }

        // Look for oscillating error patterns (can indicate stiffness)
        let mut oscillating = true;
        for i in 2..self.error_history.len() {
            if (self.error_history[i] > self.error_history[i - 1]
                && self.error_history[i - 1] > self.error_history[i - 2])
                || (self.error_history[i] < self.error_history[i - 1]
                    && self.error_history[i - 1] < self.error_history[i - 2])
            {
                oscillating = false;
                break;
            }
        }

        if oscillating {
            self.stiffness_indicators += 1;
        }

        // Check for consistent decrease in error (indicates non-stiffness)
        let mut decreasing = true;
        for i in 1..self.error_history.len() {
            if self.error_history[i] > self.error_history[i - 1] {
                decreasing = false;
                break;
            }
        }

        if decreasing && self.error_history.len() >= 3 {
            self.non_stiffness_indicators += 1;
        }
    }

    /// Analyze step size patterns for stiffness indicators
    fn analyze_step_pattern(&mut self) {
        if self.step_size_history.len() < 3 {
            return;
        }

        // Count step size decreases
        let mut decreases = 0;
        for i in 1..self.step_size_history.len() {
            if self.step_size_history[i] < self.step_size_history[i - 1] {
                decreases += 1;
            }
        }

        // High proportion of decreases might indicate stiffness
        let decrease_ratio = F::from_usize(decreases).unwrap()
            / F::from_usize(self.step_size_history.len() - 1).unwrap();

        if decrease_ratio > F::from_f64(0.7).unwrap() {
            self.stiffness_indicators += 1;
        }

        // Consistent growth in step size indicates non-stiffness
        let mut increasing = true;
        for i in 1..self.step_size_history.len() {
            if self.step_size_history[i] < self.step_size_history[i - 1] {
                increasing = false;
                break;
            }
        }

        if increasing && self.step_size_history.len() >= 3 {
            self.non_stiffness_indicators += 1;
        }
    }

    /// Update the combined stiffness score based on all indicators
    fn update_stiffness_score(&mut self) {
        let weights = &self.config.indicator_weights;

        // Calculate weighted stiffness score
        let stiff_score = F::from_usize(self.stiffness_indicators).unwrap()
            * (weights.error_pattern_weight
                + weights.step_pattern_weight
                + weights.newton_convergence_weight);

        let non_stiff_score = F::from_usize(self.non_stiffness_indicators).unwrap()
            * (weights.error_pattern_weight
                + weights.step_pattern_weight
                + weights.newton_convergence_weight);

        // Normalize to [-1, 1] range where positive is stiff
        if stiff_score > F::zero() || non_stiff_score > F::zero() {
            self.stiffness_score =
                (stiff_score - non_stiff_score) / (stiff_score + non_stiff_score).max(F::one());
        } else {
            self.stiffness_score = F::zero();
        }
    }

    /// Check if the problem is stiff based on collected indicators
    pub fn is_stiff(&self, current_method_is_stiff: bool, steps_since_switch: usize) -> bool {
        // Don't switch methods too frequently
        if steps_since_switch < self.config.min_steps_before_switch {
            return current_method_is_stiff;
        }

        // For advanced methods, use the stiffness score
        if self.config.method == StiffnessDetectionMethod::Combined {
            if current_method_is_stiff {
                // We're using BDF (stiff), consider switching to Adams (non-stiff)
                // Higher threshold to switch away from stiff method
                return self.stiffness_score > F::from_f64(-0.3).unwrap();
            } else {
                // We're using Adams (non-stiff), consider switching to BDF (stiff)
                // Lower threshold to switch to stiff method
                return self.stiffness_score > F::from_f64(0.2).unwrap();
            }
        }

        // For basic method, use simple counters
        if current_method_is_stiff {
            // Currently using BDF, check if we should switch to Adams
            self.non_stiffness_indicators >= self.config.non_stiffness_threshold
        } else {
            // Currently using Adams, check if we should switch to BDF
            self.stiffness_indicators >= self.config.stiffness_threshold
        }
    }

    /// Reset indicators after method switch
    pub fn reset_after_switch(&mut self) {
        self.stiffness_indicators = 0;
        self.non_stiffness_indicators = 0;
        self.stiffness_score = F::zero();
        // Retain history but reset counters
    }

    /// Get current stiffness score (-1.0 to 1.0, where positive means stiff)
    pub fn stiffness_score(&self) -> F {
        self.stiffness_score
    }

    /// Estimate stiffness ratio using eigenvalues of the Jacobian
    pub fn estimate_stiffness_from_jacobian(&mut self, _jacobian: &Array2<F>) -> F {
        // In a real implementation, we would:
        // 1. Estimate eigenvalues of the Jacobian (or their bounds)
        // 2. Calculate stiffness ratio as max(abs(eigenvalues))/min(abs(eigenvalues))
        // 3. Update stiffness indicators based on this ratio

        // For now, this is a placeholder that doesn't actually compute eigenvalues
        // In practice, for large systems we might use methods like power iteration
        // to estimate the largest and smallest eigenvalues

        // Dummy implementation - in reality need eigenvalue computation
        let stiffness_ratio = F::from_f64(10.0).unwrap(); // placeholder
        self.stiffness_ratio = stiffness_ratio;

        if stiffness_ratio > F::from_f64(100.0).unwrap() {
            self.stiffness_indicators += 1;
        } else if stiffness_ratio < F::from_f64(10.0).unwrap() {
            self.non_stiffness_indicators += 1;
        }

        stiffness_ratio
    }
}

/// Provides information about method switching for diagnostic purposes
#[derive(Debug, Clone)]
pub struct MethodSwitchInfo<F: IntegrateFloat> {
    /// Number of switches from non-stiff to stiff method
    pub nonstiff_to_stiff_switches: usize,
    /// Number of switches from stiff to non-stiff method
    pub stiff_to_nonstiff_switches: usize,
    /// Stiffness score at each switch point (-1.0 to 1.0)
    pub stiffness_scores: Vec<F>,
    /// Step at which each switch occurred
    pub switch_steps: Vec<usize>,
    /// Reason for each switch
    pub switch_reasons: Vec<String>,
}

impl<F: IntegrateFloat> Default for MethodSwitchInfo<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> MethodSwitchInfo<F> {
    /// Create a new method switch info tracker
    pub fn new() -> Self {
        MethodSwitchInfo {
            nonstiff_to_stiff_switches: 0,
            stiff_to_nonstiff_switches: 0,
            stiffness_scores: Vec::new(),
            switch_steps: Vec::new(),
            switch_reasons: Vec::new(),
        }
    }

    /// Record a method switch
    pub fn record_switch(
        &mut self,
        from_stiff: bool,
        step: usize,
        stiffness_score: F,
        reason: &str,
    ) {
        if from_stiff {
            self.stiff_to_nonstiff_switches += 1;
        } else {
            self.nonstiff_to_stiff_switches += 1;
        }

        self.stiffness_scores.push(stiffness_score);
        self.switch_steps.push(step);
        self.switch_reasons.push(reason.to_string());
    }

    /// Get a summary of method switching
    pub fn summary(&self) -> String {
        format!(
            "Method switching summary: {} non-stiff to stiff, {} stiff to non-stiff",
            self.nonstiff_to_stiff_switches, self.stiff_to_nonstiff_switches
        )
    }
}
