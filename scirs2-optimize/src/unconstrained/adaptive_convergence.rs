//! Adaptive tolerance selection and convergence criteria
//!
//! This module provides adaptive tolerance selection mechanisms that automatically
//! adjust convergence criteria based on problem characteristics, function scale,
//! optimization progress, and numerical stability considerations.

use crate::error::OptimizeError;
use crate::unconstrained::Options;
// Remove unused imports - they're not needed for this module
use std::collections::VecDeque;

/// Adaptive tolerance options
#[derive(Debug, Clone)]
pub struct AdaptiveToleranceOptions {
    /// Initial tolerance for function value convergence
    pub initial_ftol: f64,
    /// Initial tolerance for gradient norm convergence
    pub initial_gtol: f64,
    /// Initial tolerance for step size convergence
    pub initial_xtol: f64,
    /// Minimum allowed tolerance (floor)
    pub min_tolerance: f64,
    /// Maximum allowed tolerance (ceiling)
    pub max_tolerance: f64,
    /// How aggressively to adapt tolerances (0.0 = no adaptation, 1.0 = maximum)
    pub adaptation_rate: f64,
    /// Number of iterations to consider for adaptation
    pub history_length: usize,
    /// Whether to consider function scale in adaptation
    pub use_function_scale: bool,
    /// Whether to consider gradient scale in adaptation
    pub use_gradient_scale: bool,
    /// Whether to adapt based on stagnation detection
    pub use_stagnation_detection: bool,
    /// Whether to use problem conditioning in adaptation
    pub use_problem_conditioning: bool,
}

impl Default for AdaptiveToleranceOptions {
    fn default() -> Self {
        Self {
            initial_ftol: 1e-8,
            initial_gtol: 1e-5,
            initial_xtol: 1e-8,
            min_tolerance: 1e-15,
            max_tolerance: 1e-3,
            adaptation_rate: 0.5,
            history_length: 10,
            use_function_scale: true,
            use_gradient_scale: true,
            use_stagnation_detection: true,
            use_problem_conditioning: false, // Requires more analysis
        }
    }
}

/// State tracking for adaptive tolerance selection
#[derive(Debug, Clone)]
pub struct AdaptiveToleranceState {
    /// Current function tolerance
    pub current_ftol: f64,
    /// Current gradient tolerance
    pub current_gtol: f64,
    /// Current step tolerance
    pub current_xtol: f64,
    /// History of function values
    function_history: VecDeque<f64>,
    /// History of gradient norms
    gradient_history: VecDeque<f64>,
    /// History of step norms
    step_history: VecDeque<f64>,
    /// Estimated function scale
    function_scale: f64,
    /// Estimated gradient scale
    gradient_scale: f64,
    /// Number of stagnant iterations
    stagnant_iterations: usize,
    /// Problem dimension
    problem_dim: usize,
    /// Options
    options: AdaptiveToleranceOptions,
}

impl AdaptiveToleranceState {
    /// Create new adaptive tolerance state
    pub fn new(options: AdaptiveToleranceOptions, problem_dim: usize) -> Self {
        Self {
            current_ftol: options.initial_ftol,
            current_gtol: options.initial_gtol,
            current_xtol: options.initial_xtol,
            function_history: VecDeque::with_capacity(options.history_length),
            gradient_history: VecDeque::with_capacity(options.history_length),
            step_history: VecDeque::with_capacity(options.history_length),
            function_scale: 1.0,
            gradient_scale: 1.0,
            stagnant_iterations: 0,
            problem_dim,
            options,
        }
    }

    /// Update state with new iteration data and adapt tolerances
    pub fn update(
        &mut self,
        function_value: f64,
        gradient_norm: f64,
        step_norm: f64,
        iteration: usize,
    ) -> Result<(), OptimizeError> {
        // Add to history
        self.add_to_history(function_value, gradient_norm, step_norm);

        // Update scales
        if self.options.use_function_scale {
            self.update_function_scale();
        }
        if self.options.use_gradient_scale {
            self.update_gradient_scale();
        }

        // Detect stagnation
        if self.options.use_stagnation_detection {
            self.update_stagnation_detection();
        }

        // Adapt tolerances based on accumulated information
        self.adapt_tolerances(iteration)?;

        Ok(())
    }

    /// Add new values to history
    fn add_to_history(&mut self, function_value: f64, gradient_norm: f64, step_norm: f64) {
        // Add to function history
        if self.function_history.len() >= self.options.history_length {
            self.function_history.pop_front();
        }
        self.function_history.push_back(function_value);

        // Add to gradient history
        if self.gradient_history.len() >= self.options.history_length {
            self.gradient_history.pop_front();
        }
        self.gradient_history.push_back(gradient_norm);

        // Add to step history
        if self.step_history.len() >= self.options.history_length {
            self.step_history.pop_front();
        }
        self.step_history.push_back(step_norm);
    }

    /// Update function scale estimate
    fn update_function_scale(&mut self) {
        if self.function_history.len() < 2 {
            return;
        }

        // Estimate function scale as a robust measure of typical function values
        let mut values: Vec<f64> = self.function_history.iter().map(|&x| x.abs()).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use median as robust scale estimate
        let median_idx = values.len() / 2;
        let scale = if values.len() % 2 == 0 {
            0.5 * (values[median_idx - 1] + values[median_idx])
        } else {
            values[median_idx]
        };

        // Smooth the scale update
        let alpha = 0.1; // Smoothing factor
        self.function_scale = alpha * scale.max(1e-15) + (1.0 - alpha) * self.function_scale;
    }

    /// Update gradient scale estimate
    fn update_gradient_scale(&mut self) {
        if self.gradient_history.len() < 2 {
            return;
        }

        // Estimate gradient scale as robust measure of typical gradient norms
        let mut values: Vec<f64> = self.gradient_history.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use median as robust scale estimate
        let median_idx = values.len() / 2;
        let scale = if values.len() % 2 == 0 {
            0.5 * (values[median_idx - 1] + values[median_idx])
        } else {
            values[median_idx]
        };

        // Smooth the scale update
        let alpha = 0.1;
        self.gradient_scale = alpha * scale.max(1e-15) + (1.0 - alpha) * self.gradient_scale;
    }

    /// Update stagnation detection
    fn update_stagnation_detection(&mut self) {
        if self.function_history.len() < 3 {
            self.stagnant_iterations = 0;
            return;
        }

        // Check if function values are not decreasing significantly
        let recent_len = std::cmp::min(3, self.function_history.len());
        let recent_values: Vec<f64> = self
            .function_history
            .iter()
            .rev()
            .take(recent_len)
            .cloned()
            .collect();

        let relative_change = if recent_values.len() >= 2 {
            let f_new = recent_values[0];
            let f_old = recent_values[recent_values.len() - 1];
            if f_old.abs() > 1e-15 {
                (f_old - f_new).abs() / f_old.abs()
            } else {
                (f_old - f_new).abs()
            }
        } else {
            1.0
        };

        // Consider stagnant if relative change is very small
        if relative_change < 1e-10 {
            self.stagnant_iterations += 1;
        } else {
            self.stagnant_iterations = 0;
        }
    }

    /// Adapt tolerances based on collected information
    fn adapt_tolerances(&mut self, iteration: usize) -> Result<(), OptimizeError> {
        if iteration < 5 || self.options.adaptation_rate <= 0.0 {
            return Ok(()); // Don't adapt too early or if adaptation is disabled
        }

        // Function tolerance adaptation
        if self.options.use_function_scale {
            let scale_factor = self.function_scale;
            let adaptive_ftol = self.options.initial_ftol * scale_factor;

            self.current_ftol = self.blend_tolerance(
                self.current_ftol,
                adaptive_ftol,
                self.options.adaptation_rate * 0.5, // More conservative for ftol
            );
        }

        // Gradient tolerance adaptation
        if self.options.use_gradient_scale {
            let scale_factor = self.gradient_scale;
            let dimension_factor = (self.problem_dim as f64).sqrt(); // Account for dimension
            let adaptive_gtol = self.options.initial_gtol * scale_factor / dimension_factor;

            self.current_gtol = self.blend_tolerance(
                self.current_gtol,
                adaptive_gtol,
                self.options.adaptation_rate,
            );
        }

        // Step tolerance adaptation
        let adaptive_xtol = if self.step_history.len() >= 3 {
            // Adapt based on typical step sizes
            let typical_step =
                self.step_history.iter().sum::<f64>() / self.step_history.len() as f64;
            self.options.initial_xtol * typical_step.max(1e-12)
        } else {
            self.options.initial_xtol
        };

        self.current_xtol = self.blend_tolerance(
            self.current_xtol,
            adaptive_xtol,
            self.options.adaptation_rate * 0.3, // More conservative for xtol
        );

        // Stagnation-based adaptation
        if self.options.use_stagnation_detection && self.stagnant_iterations > 5 {
            // Relax tolerances if we're stagnating
            let relaxation_factor = 1.0 + 0.1 * (self.stagnant_iterations as f64 - 5.0).min(10.0);
            self.current_ftol *= relaxation_factor;
            self.current_gtol *= relaxation_factor;
            self.current_xtol *= relaxation_factor;
        }

        // Apply bounds
        self.current_ftol = self
            .current_ftol
            .clamp(self.options.min_tolerance, self.options.max_tolerance);
        self.current_gtol = self
            .current_gtol
            .clamp(self.options.min_tolerance, self.options.max_tolerance);
        self.current_xtol = self
            .current_xtol
            .clamp(self.options.min_tolerance, self.options.max_tolerance);

        Ok(())
    }

    /// Blend current tolerance with adaptive tolerance
    fn blend_tolerance(&self, current: f64, adaptive: f64, rate: f64) -> f64 {
        rate * adaptive + (1.0 - rate) * current
    }

    /// Get current Options struct with adapted tolerances
    pub fn get_current_options(&self, base_options: &Options) -> Options {
        let mut options = base_options.clone();
        options.ftol = self.current_ftol;
        options.gtol = self.current_gtol;
        options.xtol = self.current_xtol;
        options
    }

    /// Check if convergence is achieved with current tolerances
    pub fn check_convergence(
        &self,
        function_change: f64,
        step_norm: f64,
        gradient_norm: f64,
    ) -> ConvergenceStatus {
        let f_converged = function_change.abs() < self.current_ftol;
        let x_converged = step_norm < self.current_xtol;
        let g_converged = gradient_norm < self.current_gtol;

        ConvergenceStatus {
            f_converged,
            x_converged,
            g_converged,
            overall_converged: f_converged || x_converged || g_converged,
            ftol_used: self.current_ftol,
            xtol_used: self.current_xtol,
            gtol_used: self.current_gtol,
        }
    }

    /// Get adaptation statistics for debugging
    pub fn get_adaptation_stats(&self) -> AdaptationStats {
        AdaptationStats {
            function_scale: self.function_scale,
            gradient_scale: self.gradient_scale,
            stagnant_iterations: self.stagnant_iterations,
            current_ftol: self.current_ftol,
            current_gtol: self.current_gtol,
            current_xtol: self.current_xtol,
            history_length: self.function_history.len(),
        }
    }
}

/// Convergence status with detailed information
#[derive(Debug, Clone)]
pub struct ConvergenceStatus {
    pub f_converged: bool,
    pub x_converged: bool,
    pub g_converged: bool,
    pub overall_converged: bool,
    pub ftol_used: f64,
    pub xtol_used: f64,
    pub gtol_used: f64,
}

/// Adaptation statistics for debugging and monitoring
#[derive(Debug, Clone)]
pub struct AdaptationStats {
    pub function_scale: f64,
    pub gradient_scale: f64,
    pub stagnant_iterations: usize,
    pub current_ftol: f64,
    pub current_gtol: f64,
    pub current_xtol: f64,
    pub history_length: usize,
}

/// Create adaptive tolerance options optimized for specific problem types
pub fn create_adaptive_options_for_problem(
    problem_type: &str,
    problem_size: usize,
) -> AdaptiveToleranceOptions {
    match problem_type.to_lowercase().as_str() {
        "well_conditioned" => AdaptiveToleranceOptions {
            initial_ftol: 1e-10,
            initial_gtol: 1e-7,
            initial_xtol: 1e-10,
            adaptation_rate: 0.3,
            use_function_scale: true,
            use_gradient_scale: true,
            use_stagnation_detection: false, // Well-conditioned problems don't need stagnation handling
            ..Default::default()
        },
        "ill_conditioned" => AdaptiveToleranceOptions {
            initial_ftol: 1e-6,
            initial_gtol: 1e-4,
            initial_xtol: 1e-6,
            adaptation_rate: 0.7,
            use_function_scale: true,
            use_gradient_scale: true,
            use_stagnation_detection: true,
            use_problem_conditioning: true,
            ..Default::default()
        },
        "noisy" => AdaptiveToleranceOptions {
            initial_ftol: 1e-5,
            initial_gtol: 1e-3,
            initial_xtol: 1e-5,
            adaptation_rate: 0.8,
            history_length: 20, // Longer history for noisy problems
            use_stagnation_detection: true,
            ..Default::default()
        },
        "high_dimensional" => {
            let dimension_factor = (problem_size as f64).sqrt();
            AdaptiveToleranceOptions {
                initial_ftol: 1e-8,
                initial_gtol: 1e-5 * dimension_factor,
                initial_xtol: 1e-8,
                adaptation_rate: 0.6,
                use_gradient_scale: true,
                ..Default::default()
            }
        }
        _ => AdaptiveToleranceOptions::default(),
    }
}

/// Enhanced convergence check with adaptive tolerances
pub fn check_convergence_adaptive(
    function_change: f64,
    step_norm: f64,
    gradient_norm: f64,
    adaptive_state: &AdaptiveToleranceState,
) -> ConvergenceStatus {
    adaptive_state.check_convergence(function_change, step_norm, gradient_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_tolerance_initialization() {
        let options = AdaptiveToleranceOptions::default();
        let state = AdaptiveToleranceState::new(options.clone(), 10);

        assert_eq!(state.current_ftol, options.initial_ftol);
        assert_eq!(state.current_gtol, options.initial_gtol);
        assert_eq!(state.current_xtol, options.initial_xtol);
        assert_eq!(state.problem_dim, 10);
    }

    #[test]
    fn test_scale_adaptation() {
        let options = AdaptiveToleranceOptions {
            use_function_scale: true,
            adaptation_rate: 1.0, // Full adaptation for testing
            ..Default::default()
        };
        let mut state = AdaptiveToleranceState::new(options, 5);

        // Add some function values with known scale
        for i in 0..10 {
            let f_val = 100.0 + i as f64; // Scale around 100
            state.update(f_val, 1e-5, 1e-8, i).unwrap();
        }

        // Function scale should adapt to be around 100
        let stats = state.get_adaptation_stats();
        assert!(stats.function_scale > 50.0 && stats.function_scale < 200.0);
    }

    #[test]
    fn test_stagnation_detection() {
        let options = AdaptiveToleranceOptions {
            use_stagnation_detection: true,
            ..Default::default()
        };
        let mut state = AdaptiveToleranceState::new(options, 5);

        // Simulate stagnation by using same function value
        for i in 0..10 {
            state.update(1.0, 1e-5, 1e-8, i).unwrap();
        }

        let stats = state.get_adaptation_stats();
        assert!(stats.stagnant_iterations > 5);
    }

    #[test]
    fn test_convergence_check() {
        let options = AdaptiveToleranceOptions::default();
        let state = AdaptiveToleranceState::new(options, 5);

        let status = state.check_convergence(1e-10, 1e-10, 1e-10);
        assert!(status.overall_converged);
        assert!(status.f_converged);
        assert!(status.x_converged);
        assert!(status.g_converged);
    }

    #[test]
    fn test_problem_specific_options() {
        let well_cond = create_adaptive_options_for_problem("well_conditioned", 10);
        assert!(well_cond.initial_ftol < 1e-8);
        assert!(!well_cond.use_stagnation_detection);

        let ill_cond = create_adaptive_options_for_problem("ill_conditioned", 10);
        assert!(ill_cond.adaptation_rate > 0.5);
        assert!(ill_cond.use_stagnation_detection);

        let high_dim = create_adaptive_options_for_problem("high_dimensional", 1000);
        assert!(high_dim.initial_gtol > 1e-5); // Should be relaxed for high dimensions
    }

    #[test]
    fn test_tolerance_bounds() {
        let options = AdaptiveToleranceOptions {
            min_tolerance: 1e-12,
            max_tolerance: 1e-4,
            adaptation_rate: 1.0, // Full adaptation
            ..Default::default()
        };
        let mut state = AdaptiveToleranceState::new(options, 5);

        // Force extreme adaptation by using very large values
        for i in 0..10 {
            state.update(1e10, 1e10, 1e10, i).unwrap();
        }

        // Tolerances should be clamped within bounds
        assert!(state.current_ftol >= 1e-12);
        assert!(state.current_ftol <= 1e-4);
        assert!(state.current_gtol >= 1e-12);
        assert!(state.current_gtol <= 1e-4);
    }
}
