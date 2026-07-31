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

    /// Record a step for stiffness analysis.
    ///
    /// `step_size` and `error` should be the *actual* step size and
    /// tolerance-normalized local error estimate produced by the step just
    /// taken (not a placeholder constant); `newton_iterations` is the
    /// number of Newton iterations used for implicit steps (0 for explicit
    /// steps); `rejected` indicates whether the step was rejected. These
    /// are forwarded to the internal
    /// [`StiffnessDetector`](crate::ode::utils::stiffness::StiffnessDetector), whose
    /// `Basic`/`ErrorPattern`/`StepPattern`/`Combined` analyses (see
    /// `stiffness::mod`) actually consume them to update the stiffness
    /// indicators that [`Self::check_method_switch`] queries.
    pub fn record_step(
        &mut self,
        step_size: F,
        error: F,
        newton_iterations: usize,
        rejected: bool,
    ) {
        self.steps_since_switch += 1;
        self.detector.record_step(
            step_size,
            error,
            newton_iterations,
            rejected,
            self.steps_since_switch,
        );
    }

    /// Check whether the stiffness detector recommends switching methods.
    ///
    /// This is a pure query (it does not itself perform the switch): it
    /// consults the internal
    /// [`StiffnessDetector::is_stiff`](crate::ode::utils::stiffness::StiffnessDetector::is_stiff) and, if a
    /// switch is warranted, returns the recommended target method type.
    /// Callers should apply the switch (e.g. via [`Self::switch_method`])
    /// when this returns `Some`.
    ///
    /// Note on `StiffnessDetector::is_stiff`'s contract: its boolean return
    /// value means "the problem should currently be treated as stiff" (it
    /// uses different, hysteresis-biased thresholds depending on the
    /// current method purely to avoid flip-flopping, not a different
    /// *meaning* of the result). A switch is warranted exactly when that
    /// recommendation disagrees with the method we're currently running.
    pub fn check_method_switch(&self) -> Option<AdaptiveMethodType> {
        let current_is_stiff = matches!(
            self.method_type,
            AdaptiveMethodType::BDF | AdaptiveMethodType::Implicit
        );
        let recommend_stiff = self
            .detector
            .is_stiff(current_is_stiff, self.steps_since_switch);

        match (current_is_stiff, recommend_stiff) {
            (false, true) => Some(AdaptiveMethodType::BDF),
            (true, false) => Some(AdaptiveMethodType::Adams),
            _ => None,
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
        // Stale stiff/non-stiff indicators from before the switch should
        // not bias the very next switch decision.
        self.detector.reset_after_switch();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::utils::stiffness::StiffnessDetectionConfig;

    /// Before the fix, `record_step` discarded its argument entirely
    /// (`fn record_step(&mut self, _errorestimate: F)`) and
    /// `check_method_switch` was a hardcoded stub returning `None`
    /// unconditionally: no sequence of recorded steps could ever trigger a
    /// switch. Feed a sequence of genuinely worsening (growing error,
    /// shrinking step size, rejected) step records -- the signature of a
    /// developing stiff problem under an explicit method -- and confirm the
    /// detector now recommends switching from Adams to BDF.
    #[test]
    fn worsening_steps_trigger_switch_from_adams_to_bdf() {
        let mut state: AdaptiveMethodState<f64> =
            AdaptiveMethodState::with_config(StiffnessDetectionConfig::default());
        assert_eq!(state.method_type, AdaptiveMethodType::Adams);

        let mut switched = None;
        let mut h = 0.1_f64;
        let mut error = 1.0_f64;
        for _ in 0..40 {
            state.record_step(h, error, 0, true);
            if let Some(new_method) = state.check_method_switch() {
                switched = Some(new_method);
                break;
            }
            h *= 0.9; // shrinking step size
            error *= 1.3; // growing error
        }

        assert_eq!(
            switched,
            Some(AdaptiveMethodType::BDF),
            "stiffness detector never recommended switching to BDF despite \
             a sustained run of worsening (growing error, shrinking step, \
             rejected) steps"
        );
    }

    /// Symmetric check: starting from BDF, a sustained run of easy,
    /// quickly-converging, accepted steps should eventually trigger a
    /// switch back to Adams.
    #[test]
    fn well_behaved_steps_trigger_switch_from_bdf_to_adams() {
        let mut state: AdaptiveMethodState<f64> =
            AdaptiveMethodState::with_config(StiffnessDetectionConfig::default());
        state
            .switch_method(AdaptiveMethodType::Implicit, 0)
            .expect("switch_method should not fail");
        assert_eq!(state.method_type, AdaptiveMethodType::Implicit);

        let mut switched = None;
        let mut error = 0.5_f64;
        for _ in 0..40 {
            state.record_step(0.1, error, 1, false);
            if let Some(new_method) = state.check_method_switch() {
                switched = Some(new_method);
                break;
            }
            error *= 0.7; // shrinking error
        }

        assert_eq!(
            switched,
            Some(AdaptiveMethodType::Adams),
            "stiffness detector never recommended switching back to Adams \
             despite a sustained run of easy, accepted steps"
        );
    }

    /// A brand-new state (too few steps since the last switch) must never
    /// recommend switching, regardless of how extreme the very first
    /// recorded step looks.
    #[test]
    fn no_premature_switch_before_min_steps_before_switch() {
        let mut state: AdaptiveMethodState<f64> =
            AdaptiveMethodState::with_config(StiffnessDetectionConfig::default());
        state.record_step(0.1, 1e6, 0, true);
        assert_eq!(state.check_method_switch(), None);
    }
}
