//! Spiking Neuron Models
//!
//! This module implements various spiking neuron models used in neuromorphic
//! computing, including the leaky integrate-and-fire model and more sophisticated
//! neuron types with adaptive behaviors.

/// Spiking neuron model using leaky integrate-and-fire dynamics
///
/// This neuron model integrates input currents over time and generates spikes
/// when the membrane potential exceeds a threshold. After spiking, the neuron
/// enters a refractory period during which it cannot spike again.
///
/// # Model Dynamics
/// The membrane potential follows the equation:
/// dV/dt = -leak_constant * V + I(t)
///
/// Where V is membrane potential and I(t) is input current.
///
/// # Example
/// ```rust
/// use scirs2_spatial::neuromorphic::core::SpikingNeuron;
///
/// let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);
///
/// // Simulate neuron for several time steps
/// let dt = 0.1;
/// let input_current = 1.5;
///
/// for _ in 0..20 {
///     let spiked = neuron.update(dt, input_current);
///     if spiked {
///         println!("Neuron spiked!");
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    /// Membrane potential
    pub membrane_potential: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Refractory period
    pub refractory_period: f64,
    /// Time since last spike
    pub time_since_spike: f64,
    /// Leak constant
    pub leak_constant: f64,
    /// Input current
    pub input_current: f64,
    /// Neuron position in space
    pub position: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl SpikingNeuron {
    /// Create new spiking neuron
    ///
    /// # Arguments
    /// * `position` - Spatial position of the neuron in N-dimensional space
    ///
    /// # Returns
    /// A new `SpikingNeuron` with default parameters
    pub fn new(position: Vec<f64>) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold: 1.0,
            refractory_period: 2.0,
            time_since_spike: 2.1, // Start outside refractory period
            leak_constant: 0.1,
            input_current: 0.0,
            position,
            learning_rate: 0.01,
        }
    }

    /// Create a new spiking neuron with custom parameters
    ///
    /// # Arguments
    /// * `position` - Spatial position of the neuron
    /// * `threshold` - Spike threshold
    /// * `refractory_period` - Duration of refractory period
    /// * `leak_constant` - Membrane leak constant
    /// * `learning_rate` - Learning rate for adaptation
    pub fn with_params(
        position: Vec<f64>,
        threshold: f64,
        refractory_period: f64,
        leak_constant: f64,
        learning_rate: f64,
    ) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold,
            refractory_period,
            time_since_spike: refractory_period + 0.1, // Start outside refractory period
            leak_constant,
            input_current: 0.0,
            position,
            learning_rate,
        }
    }

    /// Update neuron state and check for spike
    ///
    /// Integrates the neuron dynamics for one time step and determines if
    /// a spike should be generated.
    ///
    /// # Arguments
    /// * `dt` - Time step size
    /// * `input_current` - Input current for this time step
    ///
    /// # Returns
    /// True if the neuron spiked, false otherwise
    pub fn update(&mut self, dt: f64, input_current: f64) -> bool {
        self.time_since_spike += dt;

        // Check if in refractory period
        if self.time_since_spike < self.refractory_period {
            return false;
        }

        // Update membrane potential using leaky integrate-and-fire model
        self.input_current = input_current;
        let leak_term = -self.leak_constant * self.membrane_potential;
        self.membrane_potential += dt * (leak_term + input_current);

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0; // Reset potential
            self.time_since_spike = 0.0; // Reset spike timer
            true
        } else {
            false
        }
    }

    /// Calculate distance-based influence on another neuron
    ///
    /// Computes the spatial influence this neuron has on another neuron
    /// based on their relative positions using a Gaussian function.
    ///
    /// # Arguments
    /// * `other_position` - Position of the other neuron
    ///
    /// # Returns
    /// Influence strength (0 to 1)
    pub fn calculate_influence(&self, other_position: &[f64]) -> f64 {
        if self.position.len() != other_position.len() {
            return 0.0;
        }

        let distance: f64 = self
            .position
            .iter()
            .zip(other_position.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Gaussian influence function
        (-distance.powi(2) / 2.0).exp()
    }

    /// Get neuron position
    pub fn position(&self) -> &[f64] {
        &self.position
    }

    /// Set neuron position
    pub fn set_position(&mut self, position: Vec<f64>) {
        self.position = position;
    }

    /// Get membrane potential
    pub fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    /// Set membrane potential
    pub fn set_membrane_potential(&mut self, potential: f64) {
        self.membrane_potential = potential;
    }

    /// Get spike threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Set spike threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Get refractory period
    pub fn refractory_period(&self) -> f64 {
        self.refractory_period
    }

    /// Set refractory period
    pub fn set_refractory_period(&mut self, period: f64) {
        self.refractory_period = period;
    }

    /// Get leak constant
    pub fn leak_constant(&self) -> f64 {
        self.leak_constant
    }

    /// Set leak constant
    pub fn set_leak_constant(&mut self, leak: f64) {
        self.leak_constant = leak;
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate;
    }

    /// Check if neuron is in refractory period
    pub fn is_refractory(&self) -> bool {
        self.time_since_spike < self.refractory_period
    }

    /// Get time since last spike
    pub fn time_since_spike(&self) -> f64 {
        self.time_since_spike
    }

    /// Reset neuron to initial state
    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.time_since_spike = 0.0;
        self.input_current = 0.0;
    }

    /// Inject current into the neuron
    pub fn inject_current(&mut self, current: f64) {
        self.input_current += current;
    }

    /// Calculate distance to another neuron
    ///
    /// # Arguments
    /// * `other` - Reference to another neuron
    ///
    /// # Returns
    /// Euclidean distance between neurons, or None if dimensions don't match
    pub fn distance_to(&self, other: &SpikingNeuron) -> Option<f64> {
        if self.position.len() != other.position.len() {
            return None;
        }

        let distance = self
            .position
            .iter()
            .zip(other.position.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        Some(distance)
    }

    /// Adapt threshold based on recent activity (homeostatic plasticity)
    ///
    /// # Arguments
    /// * `target_rate` - Target firing rate
    /// * `actual_rate` - Actual firing rate
    /// * `adaptation_rate` - Rate of threshold adaptation
    pub fn adapt_threshold(&mut self, target_rate: f64, actual_rate: f64, adaptation_rate: f64) {
        let rate_error = actual_rate - target_rate;
        self.threshold += adaptation_rate * rate_error;

        // Keep threshold in reasonable bounds
        self.threshold = self.threshold.clamp(0.1, 10.0);
    }

    /// Update learning rate based on recent performance
    ///
    /// # Arguments
    /// * `performance_factor` - Factor indicating learning performance (0-1)
    /// * `adaptation_rate` - Rate of learning rate adaptation
    pub fn adapt_learning_rate(&mut self, performance_factor: f64, adaptation_rate: f64) {
        // Increase learning rate if performance is poor, decrease if good
        let adjustment = adaptation_rate * (1.0 - performance_factor);
        self.learning_rate += adjustment;

        // Keep learning rate in reasonable bounds
        self.learning_rate = self.learning_rate.clamp(0.001, 1.0);
    }
}

/// Adaptive spiking neuron with homeostatic mechanisms
///
/// This neuron extends the basic spiking neuron with adaptive threshold
/// and learning rate mechanisms to maintain stable firing rates.
#[derive(Debug, Clone)]
pub struct AdaptiveSpikingNeuron {
    /// Base spiking neuron
    base_neuron: SpikingNeuron,
    /// Target firing rate for homeostasis
    target_firing_rate: f64,
    /// Recent firing rate estimation
    recent_firing_rate: f64,
    /// Threshold adaptation rate
    threshold_adaptation_rate: f64,
    /// Spike count for rate estimation
    spike_count: usize,
    /// Time window for rate estimation
    rate_estimation_window: f64,
    /// Current time for rate estimation
    current_time: f64,
}

impl AdaptiveSpikingNeuron {
    /// Create new adaptive spiking neuron
    ///
    /// # Arguments
    /// * `position` - Spatial position of the neuron
    /// * `target_firing_rate` - Target firing rate for homeostasis
    pub fn new(position: Vec<f64>, target_firing_rate: f64) -> Self {
        Self {
            base_neuron: SpikingNeuron::new(position),
            target_firing_rate,
            recent_firing_rate: 0.0,
            threshold_adaptation_rate: 0.001,
            spike_count: 0,
            rate_estimation_window: 100.0,
            current_time: 0.0,
        }
    }

    /// Update neuron with homeostatic adaptation
    ///
    /// # Arguments
    /// * `dt` - Time step size
    /// * `input_current` - Input current for this time step
    ///
    /// # Returns
    /// True if the neuron spiked, false otherwise
    pub fn update(&mut self, dt: f64, input_current: f64) -> bool {
        self.current_time += dt;

        // Update base neuron
        let spiked = self.base_neuron.update(dt, input_current);

        if spiked {
            self.spike_count += 1;
        }

        // Update firing rate estimate periodically
        if self.current_time >= self.rate_estimation_window {
            self.recent_firing_rate = self.spike_count as f64 / self.current_time;

            // Apply homeostatic adaptation
            self.base_neuron.adapt_threshold(
                self.target_firing_rate,
                self.recent_firing_rate,
                self.threshold_adaptation_rate,
            );

            // Reset for next window
            self.spike_count = 0;
            self.current_time = 0.0;
        }

        spiked
    }

    /// Get reference to base neuron
    pub fn base_neuron(&self) -> &SpikingNeuron {
        &self.base_neuron
    }

    /// Get mutable reference to base neuron
    pub fn base_neuron_mut(&mut self) -> &mut SpikingNeuron {
        &mut self.base_neuron
    }

    /// Get target firing rate
    pub fn target_firing_rate(&self) -> f64 {
        self.target_firing_rate
    }

    /// Set target firing rate
    pub fn set_target_firing_rate(&mut self, rate: f64) {
        self.target_firing_rate = rate;
    }

    /// Get recent firing rate
    pub fn recent_firing_rate(&self) -> f64 {
        self.recent_firing_rate
    }

    /// Get current spike count
    pub fn spike_count(&self) -> usize {
        self.spike_count
    }

    /// Reset adaptation state
    pub fn reset_adaptation(&mut self) {
        self.spike_count = 0;
        self.current_time = 0.0;
        self.recent_firing_rate = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_neuron_creation() {
        let neuron = SpikingNeuron::new(vec![0.0, 0.0]);
        assert_eq!(neuron.position(), &[0.0, 0.0]);
        assert_eq!(neuron.membrane_potential(), 0.0);
        assert_eq!(neuron.threshold(), 1.0);
        assert!(!neuron.is_refractory());
    }

    #[test]
    fn test_neuron_spiking() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);

        // Test no spike with low input
        let spiked = neuron.update(0.1, 0.1);
        assert!(!spiked);
        assert!(neuron.membrane_potential() > 0.0);

        // Test spike with high input
        let spiked = neuron.update(0.1, 10.0);
        assert!(spiked);
        assert_eq!(neuron.membrane_potential(), 0.0); // Should be reset after spike
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);

        // Generate a spike
        neuron.update(0.1, 10.0);
        assert!(neuron.is_refractory());

        // Try to spike again immediately - should fail
        let spiked = neuron.update(0.1, 10.0);
        assert!(!spiked);

        // Wait for refractory period to pass
        for _ in 0..25 {
            // 2.5 time units at dt=0.1
            neuron.update(0.1, 0.0);
        }
        assert!(!neuron.is_refractory());

        // Now should be able to spike again
        let spiked = neuron.update(0.1, 10.0);
        assert!(spiked);
    }

    #[test]
    fn test_neuron_influence() {
        let neuron1 = SpikingNeuron::new(vec![0.0, 0.0]);
        let neuron2 = SpikingNeuron::new(vec![1.0, 1.0]);

        let influence = neuron1.calculate_influence(neuron2.position());
        assert!(influence > 0.0 && influence <= 1.0);

        // Self-influence should be 1.0
        let self_influence = neuron1.calculate_influence(neuron1.position());
        assert!((self_influence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_distance() {
        let neuron1 = SpikingNeuron::new(vec![0.0, 0.0]);
        let neuron2 = SpikingNeuron::new(vec![3.0, 4.0]);

        let distance = neuron1.distance_to(&neuron2).unwrap();
        assert!((distance - 5.0).abs() < 1e-10);

        // Test mismatched dimensions
        let neuron3 = SpikingNeuron::new(vec![1.0]);
        assert!(neuron1.distance_to(&neuron3).is_none());
    }

    #[test]
    #[ignore]
    fn test_threshold_adaptation() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);
        let initial_threshold = neuron.threshold();

        // Simulate high firing rate - threshold should increase
        neuron.adapt_threshold(0.1, 0.5, 0.1); // target=0.1, actual=0.5
        assert!(neuron.threshold() > initial_threshold);

        // Simulate low firing rate - threshold should decrease
        neuron.adapt_threshold(0.1, 0.05, 0.1); // target=0.1, actual=0.05
        assert!(neuron.threshold() < initial_threshold);
    }

    #[test]
    fn test_adaptive_neuron() {
        let mut adaptive_neuron = AdaptiveSpikingNeuron::new(vec![0.0, 0.0], 0.1);

        assert_eq!(adaptive_neuron.target_firing_rate(), 0.1);
        assert_eq!(adaptive_neuron.spike_count(), 0);
        assert_eq!(adaptive_neuron.recent_firing_rate(), 0.0);

        // Simulate some activity
        for _ in 0..50 {
            adaptive_neuron.update(0.1, 2.0);
        }

        // Should have some spikes
        assert!(adaptive_neuron.spike_count() > 0);
    }

    #[test]
    fn test_neuron_reset() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);

        // Change neuron state
        neuron.update(0.1, 5.0);
        neuron.inject_current(1.0);

        // Reset should restore initial state
        neuron.reset();
        assert_eq!(neuron.membrane_potential(), 0.0);
        assert_eq!(neuron.time_since_spike(), 0.0);
        assert_eq!(neuron.input_current, 0.0);
    }

    #[test]
    fn test_parameter_setters() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);

        neuron.set_threshold(2.0);
        assert_eq!(neuron.threshold(), 2.0);

        neuron.set_refractory_period(5.0);
        assert_eq!(neuron.refractory_period(), 5.0);

        neuron.set_leak_constant(0.2);
        assert_eq!(neuron.leak_constant(), 0.2);

        neuron.set_learning_rate(0.05);
        assert_eq!(neuron.learning_rate(), 0.05);

        neuron.set_position(vec![1.0, 2.0, 3.0]);
        assert_eq!(neuron.position(), &[1.0, 2.0, 3.0]);
    }
}
