//! Synaptic Models for Neuromorphic Computing
//!
//! This module implements various synaptic models including spike-timing dependent
//! plasticity (STDP), metaplasticity, and homeostatic synaptic scaling. These synapses
//! form the connections between neurons and enable learning and adaptation.

use std::collections::VecDeque;

/// Synaptic connection with STDP learning
///
/// This synapse implements spike-timing dependent plasticity (STDP), a fundamental
/// learning rule in biological neural networks. The synaptic strength is modified
/// based on the relative timing of pre- and post-synaptic spikes.
///
/// # STDP Rule
/// - If pre-synaptic spike occurs before post-synaptic spike: potentiation (strengthening)
/// - If post-synaptic spike occurs before pre-synaptic spike: depression (weakening)
/// - The magnitude of change decreases exponentially with the time difference
///
/// # Example
/// ```rust
/// use scirs2_spatial::neuromorphic::core::Synapse;
///
/// let mut synapse = Synapse::new(0, 1, 0.5);
///
/// // Simulate STDP learning
/// synapse.update_stdp(10.0, true, false);  // Pre-synaptic spike at t=10
/// synapse.update_stdp(15.0, false, true);  // Post-synaptic spike at t=15
///
/// println!("Updated weight: {}", synapse.weight()); // Should be increased
/// ```
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Pre-synaptic neuron ID
    pub pre_neuron: usize,
    /// Post-synaptic neuron ID
    pub post_neuron: usize,
    /// Synaptic weight
    weight: f64,
    /// Last pre-synaptic spike time
    last_pre_spike: f64,
    /// Last post-synaptic spike time
    last_post_spike: f64,
    /// STDP learning rate
    stdp_rate: f64,
    /// STDP time constant
    stdp_tau: f64,
    /// Minimum weight bound
    min_weight: f64,
    /// Maximum weight bound
    max_weight: f64,
}

impl Synapse {
    /// Create new synapse
    ///
    /// # Arguments
    /// * `pre_neuron` - ID of pre-synaptic neuron
    /// * `post_neuron` - ID of post-synaptic neuron
    /// * `initial_weight` - Initial synaptic weight
    ///
    /// # Returns
    /// A new `Synapse` with default STDP parameters
    pub fn new(pre_neuron: usize, post_neuron: usize, initial_weight: f64) -> Self {
        Self {
            pre_neuron,
            post_neuron,
            weight: initial_weight,
            last_pre_spike: -1000.0,
            last_post_spike: -1000.0,
            stdp_rate: 0.01,
            stdp_tau: 20.0,
            min_weight: -2.0,
            max_weight: 2.0,
        }
    }

    /// Create synapse with custom STDP parameters
    ///
    /// # Arguments
    /// * `pre_neuron` - ID of pre-synaptic neuron
    /// * `post_neuron` - ID of post-synaptic neuron
    /// * `initial_weight` - Initial synaptic weight
    /// * `stdp_rate` - STDP learning rate
    /// * `stdp_tau` - STDP time constant
    /// * `min_weight` - Minimum weight bound
    /// * `max_weight` - Maximum weight bound
    pub fn with_stdp_params(
        pre_neuron: usize,
        post_neuron: usize,
        initial_weight: f64,
        stdp_rate: f64,
        stdp_tau: f64,
        min_weight: f64,
        max_weight: f64,
    ) -> Self {
        Self {
            pre_neuron,
            post_neuron,
            weight: initial_weight,
            last_pre_spike: -1000.0,
            last_post_spike: -1000.0,
            stdp_rate,
            stdp_tau,
            min_weight,
            max_weight,
        }
    }

    /// Update synaptic weight using STDP rule
    ///
    /// Applies the spike-timing dependent plasticity learning rule based on
    /// the timing of pre- and post-synaptic spikes.
    ///
    /// # Arguments
    /// * `current_time` - Current simulation time
    /// * `pre_spiked` - Whether pre-synaptic neuron spiked
    /// * `post_spiked` - Whether post-synaptic neuron spiked
    pub fn update_stdp(&mut self, current_time: f64, pre_spiked: bool, post_spiked: bool) {
        if pre_spiked {
            self.last_pre_spike = current_time;
        }
        if post_spiked {
            self.last_post_spike = current_time;
        }

        // Apply STDP learning rule
        // Depression (LTD): post fired before pre, pre fires now
        // dt = current_time - last_post_spike > 0 means post fired before current pre spike
        if pre_spiked && self.last_post_spike > current_time - 50.0 {
            let dt = current_time - self.last_post_spike;
            if dt > 0.0 {
                let delta_w = -self.stdp_rate * (-dt / self.stdp_tau).exp();
                self.weight += delta_w;
            }
        }

        // Potentiation (LTP): pre fired before post, post fires now
        // dt = current_time - last_pre_spike > 0 means pre fired before current post spike
        if post_spiked && self.last_pre_spike > current_time - 50.0 {
            let dt = current_time - self.last_pre_spike;
            if dt > 0.0 {
                let delta_w = self.stdp_rate * (-dt / self.stdp_tau).exp();
                self.weight += delta_w;
            }
        }

        // Keep weights in reasonable bounds
        self.weight = self.weight.clamp(self.min_weight, self.max_weight);
    }

    /// Calculate synaptic current
    ///
    /// Computes the current transmitted through this synapse given the
    /// pre-synaptic spike strength.
    ///
    /// # Arguments
    /// * `pre_spike_strength` - Strength of the pre-synaptic spike
    ///
    /// # Returns
    /// Synaptic current (weight * spike strength)
    pub fn synaptic_current(&self, pre_spike_strength: f64) -> f64 {
        self.weight * pre_spike_strength
    }

    /// Get synaptic weight
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Set synaptic weight
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight.clamp(self.min_weight, self.max_weight);
    }

    /// Get pre-synaptic neuron ID
    pub fn pre_neuron(&self) -> usize {
        self.pre_neuron
    }

    /// Get post-synaptic neuron ID
    pub fn post_neuron(&self) -> usize {
        self.post_neuron
    }

    /// Get STDP learning rate
    pub fn stdp_rate(&self) -> f64 {
        self.stdp_rate
    }

    /// Set STDP learning rate
    pub fn set_stdp_rate(&mut self, rate: f64) {
        self.stdp_rate = rate;
    }

    /// Get STDP time constant
    pub fn stdp_tau(&self) -> f64 {
        self.stdp_tau
    }

    /// Set STDP time constant
    pub fn set_stdp_tau(&mut self, tau: f64) {
        self.stdp_tau = tau;
    }

    /// Get last pre-synaptic spike time
    pub fn last_pre_spike(&self) -> f64 {
        self.last_pre_spike
    }

    /// Get last post-synaptic spike time
    pub fn last_post_spike(&self) -> f64 {
        self.last_post_spike
    }

    /// Get weight bounds
    pub fn weight_bounds(&self) -> (f64, f64) {
        (self.min_weight, self.max_weight)
    }

    /// Set weight bounds
    pub fn set_weight_bounds(&mut self, min_weight: f64, max_weight: f64) {
        self.min_weight = min_weight;
        self.max_weight = max_weight;
        // Re-clamp current weight to new bounds
        self.weight = self.weight.clamp(min_weight, max_weight);
    }

    /// Check if synapse is excitatory (positive weight)
    pub fn is_excitatory(&self) -> bool {
        self.weight > 0.0
    }

    /// Check if synapse is inhibitory (negative weight)
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }

    /// Reset spike timing history
    pub fn reset_spike_history(&mut self) {
        self.last_pre_spike = -1000.0;
        self.last_post_spike = -1000.0;
    }

    /// Calculate time since last pre-synaptic spike
    pub fn time_since_pre_spike(&self, current_time: f64) -> f64 {
        current_time - self.last_pre_spike
    }

    /// Calculate time since last post-synaptic spike
    pub fn time_since_post_spike(&self, current_time: f64) -> f64 {
        current_time - self.last_post_spike
    }
}

/// Metaplastic synapse with history-dependent learning
///
/// This synapse extends basic STDP with metaplasticity - the learning rate
/// itself adapts based on the history of synaptic activity. This provides
/// more sophisticated learning dynamics.
#[derive(Debug, Clone)]
pub struct MetaplasticSynapse {
    /// Base synapse
    base_synapse: Synapse,
    /// Learning rate history
    learning_history: VecDeque<f64>,
    /// Maximum history length
    max_history_length: usize,
    /// Metaplasticity time constant
    metaplasticity_tau: f64,
    /// Base learning rate
    base_learning_rate: f64,
}

impl MetaplasticSynapse {
    /// Create new metaplastic synapse
    ///
    /// # Arguments
    /// * `pre_neuron` - ID of pre-synaptic neuron
    /// * `post_neuron` - ID of post-synaptic neuron
    /// * `initial_weight` - Initial synaptic weight
    /// * `base_learning_rate` - Base STDP learning rate
    pub fn new(
        pre_neuron: usize,
        post_neuron: usize,
        initial_weight: f64,
        base_learning_rate: f64,
    ) -> Self {
        let mut base_synapse = Synapse::new(pre_neuron, post_neuron, initial_weight);
        base_synapse.set_stdp_rate(base_learning_rate);

        Self {
            base_synapse,
            learning_history: VecDeque::new(),
            max_history_length: 100,
            metaplasticity_tau: 100.0,
            base_learning_rate,
        }
    }

    /// Update synapse with metaplasticity
    ///
    /// # Arguments
    /// * `current_time` - Current simulation time
    /// * `pre_spiked` - Whether pre-synaptic neuron spiked
    /// * `post_spiked` - Whether post-synaptic neuron spiked
    pub fn update(&mut self, current_time: f64, pre_spiked: bool, post_spiked: bool) {
        let old_weight = self.base_synapse.weight();

        // Update base synapse
        self.base_synapse
            .update_stdp(current_time, pre_spiked, post_spiked);

        // Calculate weight change
        let weight_change = (self.base_synapse.weight() - old_weight).abs();

        // Add to learning history
        self.learning_history.push_back(weight_change);
        if self.learning_history.len() > self.max_history_length {
            self.learning_history.pop_front();
        }

        // Update learning rate based on history
        self.update_learning_rate();
    }

    /// Update learning rate based on recent activity
    fn update_learning_rate(&mut self) {
        if self.learning_history.is_empty() {
            return;
        }

        // Calculate average recent activity
        let avg_activity: f64 =
            self.learning_history.iter().sum::<f64>() / self.learning_history.len() as f64;

        // Adapt learning rate: decrease if high activity, increase if low activity
        let adaptation_factor = (-avg_activity / self.metaplasticity_tau).exp();
        let new_learning_rate = self.base_learning_rate * adaptation_factor;

        self.base_synapse.set_stdp_rate(new_learning_rate);
    }

    /// Get reference to base synapse
    pub fn base_synapse(&self) -> &Synapse {
        &self.base_synapse
    }

    /// Get mutable reference to base synapse
    pub fn base_synapse_mut(&mut self) -> &mut Synapse {
        &mut self.base_synapse
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> f64 {
        self.base_synapse.stdp_rate()
    }

    /// Get average recent activity
    pub fn average_recent_activity(&self) -> f64 {
        if self.learning_history.is_empty() {
            0.0
        } else {
            self.learning_history.iter().sum::<f64>() / self.learning_history.len() as f64
        }
    }

    /// Reset metaplasticity history
    pub fn reset_history(&mut self) {
        self.learning_history.clear();
        self.base_synapse.set_stdp_rate(self.base_learning_rate);
    }
}

/// Homeostatic synapse with activity-dependent scaling
///
/// This synapse implements homeostatic scaling to maintain stable activity levels
/// by globally scaling synaptic weights based on the target activity level.
#[derive(Debug, Clone)]
pub struct HomeostaticSynapse {
    /// Base synapse
    base_synapse: Synapse,
    /// Target activity level
    target_activity: f64,
    /// Current activity estimate
    current_activity: f64,
    /// Homeostatic scaling rate
    scaling_rate: f64,
    /// Activity estimation time constant
    activity_tau: f64,
}

impl HomeostaticSynapse {
    /// Create new homeostatic synapse
    ///
    /// # Arguments
    /// * `pre_neuron` - ID of pre-synaptic neuron
    /// * `post_neuron` - ID of post-synaptic neuron
    /// * `initial_weight` - Initial synaptic weight
    /// * `target_activity` - Target activity level for homeostasis
    pub fn new(
        pre_neuron: usize,
        post_neuron: usize,
        initial_weight: f64,
        target_activity: f64,
    ) -> Self {
        Self {
            base_synapse: Synapse::new(pre_neuron, post_neuron, initial_weight),
            target_activity,
            current_activity: 0.0,
            scaling_rate: 0.001,
            activity_tau: 1000.0,
        }
    }

    /// Update synapse with homeostatic scaling
    ///
    /// # Arguments
    /// * `current_time` - Current simulation time
    /// * `pre_spiked` - Whether pre-synaptic neuron spiked
    /// * `post_spiked` - Whether post-synaptic neuron spiked
    /// * `dt` - Time step size
    pub fn update(&mut self, current_time: f64, pre_spiked: bool, post_spiked: bool, dt: f64) {
        // Update base STDP
        self.base_synapse
            .update_stdp(current_time, pre_spiked, post_spiked);

        // Update activity estimate
        let activity_input = if post_spiked { 1.0 } else { 0.0 };
        let decay = (-dt / self.activity_tau).exp();
        self.current_activity = decay * self.current_activity + (1.0 - decay) * activity_input;

        // Apply homeostatic scaling
        let activity_error = self.current_activity - self.target_activity;
        let scaling_factor = 1.0 - self.scaling_rate * activity_error;

        let current_weight = self.base_synapse.weight();
        self.base_synapse
            .set_weight(current_weight * scaling_factor);
    }

    /// Get reference to base synapse
    pub fn base_synapse(&self) -> &Synapse {
        &self.base_synapse
    }

    /// Get current activity estimate
    pub fn current_activity(&self) -> f64 {
        self.current_activity
    }

    /// Get target activity
    pub fn target_activity(&self) -> f64 {
        self.target_activity
    }

    /// Set target activity
    pub fn set_target_activity(&mut self, target: f64) {
        self.target_activity = target;
    }

    /// Get activity error (current - target)
    pub fn activity_error(&self) -> f64 {
        self.current_activity - self.target_activity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_creation() {
        let synapse = Synapse::new(0, 1, 0.5);
        assert_eq!(synapse.pre_neuron(), 0);
        assert_eq!(synapse.post_neuron(), 1);
        assert_eq!(synapse.weight(), 0.5);
        assert!(synapse.is_excitatory());
        assert!(!synapse.is_inhibitory());
    }

    #[test]
    fn test_stdp_potentiation() {
        let mut synapse = Synapse::new(0, 1, 0.5);
        let initial_weight = synapse.weight();

        // Pre-synaptic spike before post-synaptic spike (potentiation)
        synapse.update_stdp(10.0, true, false); // Pre spike at t=10
        synapse.update_stdp(15.0, false, true); // Post spike at t=15

        // Weight should increase
        assert!(synapse.weight() > initial_weight);
    }

    #[test]
    fn test_stdp_depression() {
        let mut synapse = Synapse::new(0, 1, 0.5);
        let initial_weight = synapse.weight();

        // Post-synaptic spike before pre-synaptic spike (depression)
        synapse.update_stdp(10.0, false, true); // Post spike at t=10
        synapse.update_stdp(15.0, true, false); // Pre spike at t=15

        // Weight should decrease
        assert!(synapse.weight() < initial_weight);
    }

    #[test]
    fn test_synaptic_current() {
        let synapse = Synapse::new(0, 1, 0.5);
        let current = synapse.synaptic_current(2.0);
        assert_eq!(current, 1.0); // 0.5 * 2.0
    }

    #[test]
    fn test_weight_bounds() {
        let mut synapse = Synapse::new(0, 1, 0.0);

        // Test setting weight beyond bounds
        synapse.set_weight(10.0);
        assert_eq!(synapse.weight(), 2.0); // Should be clamped to max

        synapse.set_weight(-10.0);
        assert_eq!(synapse.weight(), -2.0); // Should be clamped to min
    }

    #[test]
    fn test_inhibitory_synapse() {
        let synapse = Synapse::new(0, 1, -0.5);
        assert!(!synapse.is_excitatory());
        assert!(synapse.is_inhibitory());

        let current = synapse.synaptic_current(1.0);
        assert_eq!(current, -0.5);
    }

    #[test]
    fn test_spike_timing() {
        let mut synapse = Synapse::new(0, 1, 0.5);

        synapse.update_stdp(10.0, true, false);
        assert_eq!(synapse.last_pre_spike(), 10.0);
        assert_eq!(synapse.time_since_pre_spike(15.0), 5.0);

        synapse.update_stdp(12.0, false, true);
        assert_eq!(synapse.last_post_spike(), 12.0);
        assert_eq!(synapse.time_since_post_spike(15.0), 3.0);
    }

    #[test]
    fn test_metaplastic_synapse() {
        let mut meta_synapse = MetaplasticSynapse::new(0, 1, 0.5, 0.01);

        assert_eq!(meta_synapse.current_learning_rate(), 0.01);
        assert_eq!(meta_synapse.average_recent_activity(), 0.0);

        // Simulate some activity
        for i in 0..10 {
            meta_synapse.update(i as f64, true, false);
            meta_synapse.update(i as f64 + 0.5, false, true);
        }

        // Learning rate should adapt based on activity
        assert!(meta_synapse.average_recent_activity() > 0.0);
    }

    #[test]
    fn test_homeostatic_synapse() {
        let mut homeostatic = HomeostaticSynapse::new(0, 1, 0.5, 0.1);

        assert_eq!(homeostatic.target_activity(), 0.1);
        assert_eq!(homeostatic.current_activity(), 0.0);

        // Simulate high activity
        for _ in 0..50 {
            homeostatic.update(1.0, true, true, 0.1);
        }

        // Activity should increase
        assert!(homeostatic.current_activity() > 0.0);
        assert!(homeostatic.activity_error() != 0.0);
    }

    #[test]
    fn test_synapse_reset() {
        let mut synapse = Synapse::new(0, 1, 0.5);

        // Record some spike activity
        synapse.update_stdp(10.0, true, false);
        synapse.update_stdp(15.0, false, true);

        // Reset should clear spike history
        synapse.reset_spike_history();
        assert_eq!(synapse.last_pre_spike(), -1000.0);
        assert_eq!(synapse.last_post_spike(), -1000.0);
    }

    #[test]
    fn test_custom_stdp_parameters() {
        let synapse = Synapse::with_stdp_params(0, 1, 0.5, 0.05, 10.0, -1.0, 1.0);

        assert_eq!(synapse.stdp_rate(), 0.05);
        assert_eq!(synapse.stdp_tau(), 10.0);
        assert_eq!(synapse.weight_bounds(), (-1.0, 1.0));
    }
}
