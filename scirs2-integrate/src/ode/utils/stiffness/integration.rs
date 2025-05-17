//! Integration-related stiffness detection utilities

use crate::IntegrateFloat;

/// The current state of an adaptive method
#[derive(Debug, Clone)]
pub struct AdaptiveMethodState<F: IntegrateFloat> {
    /// Current method type
    pub method_type: AdaptiveMethodType,
    /// Steps since last method switch
    pub steps_since_switch: usize,
    /// Current order of the method
    pub order: usize,
    /// Stiffness detector configuration
    pub config: crate::ode::utils::stiffness::StiffnessDetectionConfig<F>,
    /// Stiffness detector
    pub detector: crate::ode::utils::stiffness::StiffnessDetector<F>,
}

impl<F: IntegrateFloat> AdaptiveMethodState<F> {
    /// Create with configuration
    pub fn with_config(config: crate::ode::utils::stiffness::StiffnessDetectionConfig<F>) -> Self {
        let detector = crate::ode::utils::stiffness::StiffnessDetector::with_config(config.clone());
        Self {
            method_type: AdaptiveMethodType::Adams,
            steps_since_switch: 0,
            order: 1, // Start with order 1
            config,
            detector,
        }
    }

    /// Record a step
    pub fn record_step(&mut self, _error_estimate: F) {
        self.steps_since_switch += 1;
    }

    /// Check if method should switch
    pub fn check_method_switch(&mut self) -> Option<AdaptiveMethodType> {
        if self.steps_since_switch > 10 {
            // Simple switching logic for now
            None
        } else {
            None
        }
    }

    /// Switch to a new method
    pub fn switch_method(
        &mut self,
        new_method: AdaptiveMethodType,
        _steps: usize,
    ) -> crate::error::IntegrateResult<()> {
        self.method_type = new_method;
        self.steps_since_switch = 0;
        Ok(())
    }

    /// Generate a diagnostic message about the current state
    pub fn generate_diagnostic_message(&self) -> String {
        format!(
            "AdaptiveMethodState: method={:?}, steps_since_switch={}",
            self.method_type, self.steps_since_switch
        )
    }
}

/// Type of adaptive method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveMethodType {
    /// Adams methods for non-stiff problems
    Adams,
    /// BDF methods for stiff problems
    BDF,
    /// Runge-Kutta methods
    RungeKutta,
    /// Implicit methods (Radau, etc.)
    Implicit,
    /// Explicit methods (Adams, etc.)
    Explicit,
}
