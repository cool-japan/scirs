//! Integration of stiffness detection with ODE solvers
//!
//! This module provides utilities for integrating the enhanced stiffness
//! detection algorithms with existing ODE solvers, particularly LSODA.

use super::{StiffnessDetector, StiffnessDetectionConfig, MethodSwitchInfo};
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Method type for ODE solvers with stiffness detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMethodType {
    /// Explicit method for non-stiff problems (e.g., Adams)
    Explicit,
    /// Implicit method for stiff problems (e.g., BDF)
    Implicit,
}

/// State information for ODE methods with stiffness detection
#[derive(Debug, Clone)]
pub struct AdaptiveMethodState<F: Float> {
    /// Current method type being used
    pub method_type: AdaptiveMethodType,
    /// Current order of the method
    pub order: usize,
    /// Stiffness detector for method switching
    pub stiffness_detector: StiffnessDetector<F>,
    /// Method switching information for diagnostics
    pub switch_info: MethodSwitchInfo<F>,
    /// Number of steps since last method switch
    pub steps_since_switch: usize,
    /// Whether the method was recently switched
    pub recently_switched: bool,
}

impl<F: Float + FromPrimitive + Debug> AdaptiveMethodState<F> {
    /// Create a new adaptive method state, starting with explicit method
    pub fn new() -> Self {
        AdaptiveMethodState {
            method_type: AdaptiveMethodType::Explicit,
            order: 1,
            stiffness_detector: StiffnessDetector::new(),
            switch_info: MethodSwitchInfo::new(),
            steps_since_switch: 0,
            recently_switched: false,
        }
    }

    /// Create a new adaptive method state with custom configuration
    pub fn with_config(config: StiffnessDetectionConfig<F>) -> Self {
        AdaptiveMethodState {
            method_type: AdaptiveMethodType::Explicit,
            order: 1,
            stiffness_detector: StiffnessDetector::with_config(config),
            switch_info: MethodSwitchInfo::new(),
            steps_since_switch: 0,
            recently_switched: false,
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
        self.stiffness_detector.record_step(
            step_size,
            error,
            newton_iterations,
            rejected,
            steps_taken,
        );
    }

    /// Check if method switching is needed and update state accordingly
    pub fn check_method_switch(&mut self, steps_taken: usize) -> bool {
        // Check if we should switch methods based on stiffness detection
        let current_is_stiff = self.method_type == AdaptiveMethodType::Implicit;
        let should_be_stiff = self.stiffness_detector.is_stiff(
            current_is_stiff,
            self.steps_since_switch,
        );
        
        // If we need to switch methods
        if should_be_stiff != current_is_stiff && !self.recently_switched {
            // Perform the switch
            self.switch_method(
                if should_be_stiff {
                    AdaptiveMethodType::Implicit
                } else {
                    AdaptiveMethodType::Explicit
                },
                steps_taken,
            );
            true
        } else {
            // Reset recently_switched flag after some steps
            if self.recently_switched && self.steps_since_switch >= 5 {
                self.recently_switched = false;
            }
            false
        }
    }

    /// Switch method type and perform necessary adjustments
    pub fn switch_method(
        &mut self,
        new_method: AdaptiveMethodType,
        step: usize,
    ) -> IntegrateResult<()> {
        let from_stiff = self.method_type == AdaptiveMethodType::Implicit;
        let stiffness_score = self.stiffness_detector.stiffness_score();
        
        // Record the method switch for diagnostics
        let reason = if new_method == AdaptiveMethodType::Implicit {
            "Problem appears stiff based on analysis"
        } else {
            "Problem appears non-stiff based on analysis"
        };
        
        self.switch_info.record_switch(from_stiff, step, stiffness_score, reason);
        
        // Switch the method type
        self.method_type = new_method;
        
        // Reset state for the new method
        self.steps_since_switch = 0;
        self.recently_switched = true;
        self.stiffness_detector.reset_after_switch();
        
        // Handle specific adjustments for each transition
        match new_method {
            AdaptiveMethodType::Implicit => {
                // Switching to stiff (BDF) method
                self.order = 1; // Start with low order for stability
            }
            AdaptiveMethodType::Explicit => {
                // Switching to non-stiff (Adams) method
                self.order = 1; // Start with low order
            }
        }
        
        Ok(())
    }

    /// Update state after a step is taken
    pub fn update_after_step(&mut self, accepted: bool) {
        self.steps_since_switch += 1;
        
        // If the step was accepted, consider order adaptation
        if accepted {
            // Order adaptation logic would go here
            // For example, increasing order if error is small and
            // we have enough history points
        }
    }

    /// Estimate stiffness from the jacobian matrix
    pub fn estimate_stiffness_from_jacobian(&mut self, jacobian: &Array2<F>) -> F {
        self.stiffness_detector.estimate_stiffness_from_jacobian(jacobian)
    }

    /// Generate a diagnostic message about method switching
    pub fn generate_diagnostic_message(&self) -> String {
        let mut message = self.switch_info.summary();
        
        if self.switch_info.nonstiff_to_stiff_switches + 
           self.switch_info.stiff_to_nonstiff_switches > 0 {
            message.push_str("\nDetailed switches:");
            
            // Add up to the last 5 switches for detailed info
            let start_idx = self.switch_info.switch_steps.len().saturating_sub(5);
            for i in start_idx..self.switch_info.switch_steps.len() {
                message.push_str(&format!(
                    "\n  Step {}: {} (stiffness score: {:.2})",
                    self.switch_info.switch_steps[i],
                    self.switch_info.switch_reasons[i],
                    self.switch_info.stiffness_scores[i]
                ));
            }
        }
        
        message
    }
}