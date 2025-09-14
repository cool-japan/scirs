//! Competitive Learning Algorithms for Neuromorphic Spatial Processing
//!
//! This module implements advanced competitive learning algorithms including
//! winner-take-all dynamics, homeostatic plasticity, and multi-timescale adaptation
//! for spatial data clustering and pattern discovery.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use std::collections::VecDeque;

/// Bio-inspired competitive learning for spatial clustering
///
/// This clusterer uses winner-take-all dynamics with lateral inhibition to discover
/// clusters in spatial data. Neurons compete for activation, with the winner
/// (neuron with strongest response) being updated while others are inhibited.
///
/// # Features
/// - Winner-take-all competitive dynamics
/// - Lateral inhibition for neural competition
/// - Adaptive learning rates
/// - Neighborhood function for topological organization
/// - Distance-based neuron activation
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use scirs2_spatial::neuromorphic::algorithms::CompetitiveNeuralClusterer;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
/// ]).unwrap();
///
/// let mut clusterer = CompetitiveNeuralClusterer::new(2, 2);
/// let assignments = clusterer.fit(&points.view(), 100).unwrap();
/// println!("Cluster assignments: {:?}", assignments);
/// ```
#[derive(Debug, Clone)]
pub struct CompetitiveNeuralClusterer {
    /// Network neurons representing cluster centers
    neurons: Vec<Array1<f64>>,
    /// Learning rates for each neuron
    learning_rates: Vec<f64>,
    /// Lateral inhibition strengths
    inhibition_strengths: Array2<f64>,
    /// Winner-take-all threshold
    #[allow(dead_code)]
    wta_threshold: f64,
    /// Neighborhood function parameters
    neighborhood_sigma: f64,
}

impl CompetitiveNeuralClusterer {
    /// Create new competitive neural clusterer
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to discover
    /// * `input_dims` - Number of input dimensions
    ///
    /// # Returns
    /// A new `CompetitiveNeuralClusterer` with randomly initialized neurons
    pub fn new(num_clusters: usize, input_dims: usize) -> Self {
        let mut neurons = Vec::new();
        let mut learning_rates = Vec::new();
        let mut rng = rand::rng();

        // Initialize neurons with random weights
        for _ in 0..num_clusters {
            let weights = Array1::from_shape_fn(input_dims, |_| rng.gen_range(0.0..1.0));
            neurons.push(weights);
            learning_rates.push(0.1);
        }

        // Initialize inhibition matrix
        let inhibition_strengths =
            Array2::from_shape_fn(
                (num_clusters, num_clusters),
                |(i, j)| {
                    if i == j {
                        0.0
                    } else {
                        0.1
                    }
                },
            );

        Self {
            neurons,
            learning_rates,
            inhibition_strengths,
            wta_threshold: 0.5,
            neighborhood_sigma: 1.0,
        }
    }

    /// Configure neighborhood parameters
    ///
    /// # Arguments
    /// * `sigma` - Neighborhood function width
    /// * `wta_threshold` - Winner-take-all threshold
    pub fn with_competition_params(mut self, sigma: f64, wta_threshold: f64) -> Self {
        self.neighborhood_sigma = sigma;
        self.wta_threshold = wta_threshold;
        self
    }

    /// Train competitive network on spatial data
    ///
    /// Applies competitive learning dynamics where neurons compete for activation
    /// and the winner adapts towards the input pattern while inhibiting neighbors.
    ///
    /// # Arguments
    /// * `points` - Input spatial points (n_points × n_dims)
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    /// Cluster assignments for each input point
    pub fn fit(
        &mut self,
        points: &ArrayView2<'_, f64>,
        epochs: usize,
    ) -> SpatialResult<Array1<usize>> {
        let n_points = points.dim().0;
        let mut assignments = Array1::zeros(n_points);

        if n_points == 0 {
            return Ok(assignments);
        }

        for epoch in 0..epochs {
            // Adjust learning rate and neighborhood size
            let epoch_factor = 1.0 - (epoch as f64) / (epochs as f64);
            let current_sigma = self.neighborhood_sigma * epoch_factor;

            for (point_idx, point) in points.outer_iter().enumerate() {
                // Find winning neuron
                let winner = self.find_winner(&point.to_owned())?;
                assignments[point_idx] = winner;

                // Update winner and neighbors
                self.update_neurons(&point.to_owned(), winner, current_sigma, epoch_factor)?;

                // Apply lateral inhibition
                self.apply_lateral_inhibition(winner)?;
            }
        }

        Ok(assignments)
    }

    /// Find winning neuron using competitive dynamics
    ///
    /// Computes the neuron with the smallest distance to the input pattern.
    fn find_winner(&self, input: &Array1<f64>) -> SpatialResult<usize> {
        let mut min_distance = f64::INFINITY;
        let mut winner = 0;

        for (i, neuron) in self.neurons.iter().enumerate() {
            // Calculate Euclidean distance
            let distance: f64 = input
                .iter()
                .zip(neuron.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                winner = i;
            }
        }

        Ok(winner)
    }

    /// Update neuron weights using competitive learning
    ///
    /// Applies weight updates to the winner and its neighbors based on
    /// the neighborhood function and learning rate scheduling.
    fn update_neurons(
        &mut self,
        input: &Array1<f64>,
        winner: usize,
        sigma: f64,
        learning_factor: f64,
    ) -> SpatialResult<()> {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate neighborhood influence
            let distance_to_winner = (i as i32 - winner as i32).abs() as f64;
            let neighborhood_influence =
                (-distance_to_winner.powi(2) / (2.0 * sigma.powi(2))).exp();

            // Update neuron weights
            let effective_learning_rate =
                self.learning_rates[i] * learning_factor * neighborhood_influence;

            for (weight, &input_val) in neuron.iter_mut().zip(input.iter()) {
                *weight += effective_learning_rate * (input_val - *weight);
            }
        }

        Ok(())
    }

    /// Apply lateral inhibition between neurons
    ///
    /// Strengthens inhibitory connections and adjusts learning rates
    /// based on competitive dynamics.
    fn apply_lateral_inhibition(&mut self, winner: usize) -> SpatialResult<()> {
        // Strengthen inhibitory connections from winner to others
        for i in 0..self.neurons.len() {
            if i != winner {
                self.inhibition_strengths[[winner, i]] += 0.001;
                self.inhibition_strengths[[winner, i]] =
                    self.inhibition_strengths[[winner, i]].min(0.5);

                // Reduce learning rate of inhibited neurons
                self.learning_rates[i] *= 0.99;
                self.learning_rates[i] = self.learning_rates[i].max(0.001);
            }
        }

        // Boost winner's learning rate slightly
        self.learning_rates[winner] *= 1.001;
        self.learning_rates[winner] = self.learning_rates[winner].min(0.2);

        Ok(())
    }

    /// Get cluster centers (neuron weights)
    ///
    /// # Returns
    /// Array containing the current neuron weight vectors as cluster centers
    pub fn get_cluster_centers(&self) -> Array2<f64> {
        let num_clusters = self.neurons.len();
        let input_dims = self.neurons[0].len();

        let mut centers = Array2::zeros((num_clusters, input_dims));
        for (i, neuron) in self.neurons.iter().enumerate() {
            centers.row_mut(i).assign(neuron);
        }

        centers
    }

    /// Get current learning rates
    pub fn learning_rates(&self) -> &[f64] {
        &self.learning_rates
    }

    /// Get inhibition strength matrix
    pub fn inhibition_strengths(&self) -> &Array2<f64> {
        &self.inhibition_strengths
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.neurons.len()
    }

    /// Reset the clusterer to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::rng();
        let input_dims = self.neurons[0].len();

        // Reinitialize neuron weights
        for neuron in &mut self.neurons {
            for weight in neuron.iter_mut() {
                *weight = rng.gen_range(0.0..1.0);
            }
        }

        // Reset learning rates
        for rate in &mut self.learning_rates {
            *rate = 0.1;
        }

        // Reset inhibition strengths
        let num_clusters = self.neurons.len();
        for i in 0..num_clusters {
            for j in 0..num_clusters {
                self.inhibition_strengths[[i, j]] = if i == j { 0.0 } else { 0.1 };
            }
        }
    }
}

/// Advanced homeostatic plasticity for neuromorphic spatial learning
///
/// This clusterer implements homeostatic plasticity mechanisms that maintain
/// stable neural activity levels while adapting to input patterns. It includes
/// intrinsic plasticity, synaptic scaling, and multi-timescale adaptation.
///
/// # Features
/// - Homeostatic neurons with intrinsic plasticity
/// - Synaptic scaling for stability
/// - Multi-timescale adaptation (fast, medium, slow)
/// - Metaplasticity for adaptive learning rates
/// - Target firing rate maintenance
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use scirs2_spatial::neuromorphic::algorithms::HomeostaticNeuralClusterer;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
/// ]).unwrap();
///
/// let mut clusterer = HomeostaticNeuralClusterer::new(2, 2)
///     .with_homeostatic_params(0.1, 1000.0);
/// let assignments = clusterer.fit(&points.view(), 50).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HomeostaticNeuralClusterer {
    /// Number of clusters (output neurons)
    num_clusters: usize,
    /// Input dimension
    input_dim: usize,
    /// Output neurons with homeostatic mechanisms
    output_neurons: Vec<HomeostaticNeuron>,
    /// Synaptic weights
    weights: Array2<f64>,
    /// Global inhibition strength
    #[allow(dead_code)]
    global_inhibition: f64,
    /// Learning rate adaptation parameters
    learning_rate_adaptation: LearningRateAdaptation,
    /// Metaplasticity parameters
    metaplasticity: MetaplasticityController,
    /// Multi-timescale adaptation
    multi_timescale: MultiTimescaleAdaptation,
}

impl HomeostaticNeuralClusterer {
    /// Create new homeostatic neural clusterer
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to discover
    /// * `input_dim` - Number of input dimensions
    ///
    /// # Returns
    /// A new `HomeostaticNeuralClusterer` with default parameters
    pub fn new(num_clusters: usize, input_dim: usize) -> Self {
        let mut output_neurons = Vec::new();
        for _ in 0..num_clusters {
            output_neurons.push(HomeostaticNeuron::new());
        }

        let weights = Array2::zeros((num_clusters, input_dim));

        let learning_rate_adaptation = LearningRateAdaptation::new(0.01);
        let metaplasticity = MetaplasticityController::new(num_clusters, input_dim);
        let multi_timescale = MultiTimescaleAdaptation::new();

        Self {
            num_clusters,
            input_dim,
            output_neurons,
            weights,
            global_inhibition: 0.1,
            learning_rate_adaptation,
            metaplasticity,
            multi_timescale,
        }
    }

    /// Configure homeostatic parameters
    ///
    /// # Arguments
    /// * `target_firing_rate` - Target firing rate for neurons
    /// * `homeostatic_tau` - Time constant for homeostatic adaptation
    pub fn with_homeostatic_params(
        mut self,
        target_firing_rate: f64,
        homeostatic_tau: f64,
    ) -> Self {
        for neuron in &mut self.output_neurons {
            neuron.target_firing_rate = target_firing_rate;
            neuron.homeostatic_tau = homeostatic_tau;
        }
        self
    }

    /// Fit homeostatic clustering model
    ///
    /// Trains the network using homeostatic plasticity mechanisms to discover
    /// stable cluster representations while maintaining target activity levels.
    ///
    /// # Arguments
    /// * `points` - Input spatial points (n_points × n_dims)
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    /// Cluster assignments for each input point
    pub fn fit(&mut self, points: &ArrayView2<f64>, epochs: usize) -> SpatialResult<Array1<usize>> {
        let (n_samples, n_features) = points.dim();

        if n_features != self.input_dim {
            return Err(SpatialError::InvalidInput(
                "Input dimension mismatch".to_string(),
            ));
        }

        if n_samples == 0 {
            return Ok(Array1::zeros(0));
        }

        // Initialize weights randomly
        self.initialize_weights()?;

        let mut assignments = Array1::zeros(n_samples);
        let current_time = 0.0;
        let dt = 0.001; // 1ms time step

        for epoch in 0..epochs {
            let mut epoch_error = 0.0;

            for (sample_idx, sample) in points.outer_iter().enumerate() {
                // Forward pass with homeostatic mechanisms
                let (winner_idx, neuron_activities) = self.forward_pass_homeostatic(
                    &sample,
                    current_time + (epoch * n_samples + sample_idx) as f64 * dt,
                )?;

                assignments[sample_idx] = winner_idx;

                // Compute reconstruction error
                let reconstruction = self.weights.row(winner_idx);
                let error: f64 = sample
                    .iter()
                    .zip(reconstruction.iter())
                    .map(|(&x, &w)| (x - w).powi(2))
                    .sum();
                epoch_error += error;

                // Homeostatic learning update
                self.homeostatic_learning_update(
                    &sample,
                    winner_idx,
                    &neuron_activities,
                    error,
                    current_time + (epoch * n_samples + sample_idx) as f64 * dt,
                )?;
            }

            // Update learning rate based on performance
            self.learning_rate_adaptation
                .update_learning_rate(epoch_error / n_samples as f64);

            // Update multi-timescale adaptation
            self.multi_timescale
                .update(epoch_error / n_samples as f64, dt * n_samples as f64);

            // Homeostatic updates at end of epoch
            self.update_homeostatic_mechanisms(dt * n_samples as f64)?;
        }

        Ok(assignments)
    }

    /// Get cluster centers (weights)
    pub fn get_cluster_centers(&self) -> Array2<f64> {
        self.weights.clone()
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get neuron firing rates
    pub fn neuron_firing_rates(&self) -> Vec<f64> {
        self.output_neurons
            .iter()
            .map(|neuron| neuron.actual_firing_rate)
            .collect()
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> f64 {
        self.learning_rate_adaptation.base_rate
    }

    /// Forward pass with homeostatic mechanisms
    ///
    /// Computes neural activities with homeostatic modulation and finds
    /// the winning neuron based on highest activity.
    fn forward_pass_homeostatic(
        &mut self,
        input: &ArrayView1<f64>,
        current_time: f64,
    ) -> SpatialResult<(usize, Array1<f64>)> {
        let mut activities = Array1::zeros(self.num_clusters);

        // Compute neural activities with homeostatic modulation
        for (neuron_idx, neuron) in self.output_neurons.iter_mut().enumerate() {
            let weights_row = self.weights.row(neuron_idx);

            // Compute dot product (synaptic input)
            let synaptic_input: f64 = input
                .iter()
                .zip(weights_row.iter())
                .map(|(&x, &w)| x * w)
                .sum();

            // Apply synaptic scaling
            let scaled_input = synaptic_input * neuron.synaptic_scaling;

            // Update membrane potential
            neuron.update_membrane_potential(scaled_input, current_time);

            // Apply intrinsic excitability modulation
            let modulated_potential = neuron.membrane_potential * neuron.intrinsic_excitability;
            activities[neuron_idx] = modulated_potential;
        }

        // Find winner (highest activity)
        let winner_idx = activities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Update firing rates
        self.output_neurons[winner_idx].record_spike(current_time);

        Ok((winner_idx, activities))
    }

    /// Initialize weights randomly
    fn initialize_weights(&mut self) -> SpatialResult<()> {
        let mut rng = rand::rng();

        for mut row in self.weights.outer_iter_mut() {
            for weight in row.iter_mut() {
                *weight = rng.gen_range(0.0..1.0);
            }

            // Normalize weights
            let norm: f64 = row.iter().map(|&w| w * w).sum::<f64>().sqrt();
            if norm > 0.0 {
                for weight in row.iter_mut() {
                    *weight /= norm;
                }
            }
        }

        Ok(())
    }

    /// Homeostatic learning update
    ///
    /// Updates synaptic weights using homeostatic learning rules that
    /// incorporate metaplasticity and multi-timescale adaptation.
    fn homeostatic_learning_update(
        &mut self,
        input: &ArrayView1<f64>,
        winner_idx: usize,
        activities: &Array1<f64>,
        error: f64,
        current_time: f64,
    ) -> SpatialResult<()> {
        // Get current learning rate
        let learning_rate = self.learning_rate_adaptation.base_rate;

        // Apply metaplasticity
        let meta_modulation = self.metaplasticity.compute_modulation(winner_idx, error);

        // Apply multi-timescale adaptation
        let timescale_modulation = self.multi_timescale.get_adaptation_factor();

        // Combined learning rate
        let effective_learning_rate = learning_rate * meta_modulation * timescale_modulation;

        // Update winner weights (competitive learning with homeostatic modulation)
        let winner_neuron = &self.output_neurons[winner_idx];
        let homeostatic_factor = winner_neuron.get_homeostatic_factor();

        for (weight, &input_val) in self
            .weights
            .row_mut(winner_idx)
            .iter_mut()
            .zip(input.iter())
        {
            let weight_update =
                effective_learning_rate * homeostatic_factor * (input_val - *weight);
            *weight += weight_update;
        }

        // Update metaplasticity variables
        self.metaplasticity
            .update_metaplastic_variables(winner_idx, activities, current_time);

        Ok(())
    }

    /// Update homeostatic mechanisms
    fn update_homeostatic_mechanisms(&mut self, dt: f64) -> SpatialResult<()> {
        for neuron in &mut self.output_neurons {
            neuron.update_homeostatic_mechanisms(dt);
        }
        Ok(())
    }
}

/// Homeostatic neuron with intrinsic plasticity
///
/// This neuron implements multiple homeostatic mechanisms to maintain
/// stable activity levels including intrinsic excitability adaptation
/// and synaptic scaling.
#[derive(Debug, Clone)]
pub struct HomeostaticNeuron {
    /// Current membrane potential
    pub membrane_potential: f64,
    /// Spike threshold (adaptive)
    #[allow(dead_code)]
    pub threshold: f64,
    /// Target firing rate
    pub target_firing_rate: f64,
    /// Actual firing rate (exponential moving average)
    pub actual_firing_rate: f64,
    /// Intrinsic excitability
    pub intrinsic_excitability: f64,
    /// Homeostatic time constant
    pub homeostatic_tau: f64,
    /// Spike history for rate computation
    pub spike_history: VecDeque<f64>,
    /// Synaptic scaling factor
    pub synaptic_scaling: f64,
    /// Membrane time constant
    pub membrane_tau: f64,
    /// Last spike time
    pub last_spike_time: f64,
}

impl HomeostaticNeuron {
    /// Create new homeostatic neuron
    pub fn new() -> Self {
        Self {
            membrane_potential: 0.0,
            threshold: 1.0,
            target_firing_rate: 0.1,
            actual_firing_rate: 0.0,
            intrinsic_excitability: 1.0,
            homeostatic_tau: 1000.0, // 1 second
            spike_history: VecDeque::new(),
            synaptic_scaling: 1.0,
            membrane_tau: 10.0, // 10ms
            last_spike_time: -1000.0,
        }
    }

    /// Update membrane potential
    pub fn update_membrane_potential(&mut self, input: f64, current_time: f64) {
        let dt = current_time - self.last_spike_time;

        // Leaky integrate-and-fire dynamics
        let decay = (-dt / self.membrane_tau).exp();
        self.membrane_potential = self.membrane_potential * decay + input;
    }

    /// Record spike and update firing rate
    pub fn record_spike(&mut self, current_time: f64) {
        self.spike_history.push_back(current_time);

        // Keep only recent spikes (last 1 second)
        while let Some(&front_time) = self.spike_history.front() {
            if current_time - front_time > 1000.0 {
                self.spike_history.pop_front();
            } else {
                break;
            }
        }

        // Update actual firing rate (exponential moving average)
        let instantaneous_rate = self.spike_history.len() as f64 / 1000.0; // spikes per ms
        let alpha = 1.0 / self.homeostatic_tau;
        self.actual_firing_rate =
            (1.0 - alpha) * self.actual_firing_rate + alpha * instantaneous_rate;
    }

    /// Update homeostatic mechanisms
    pub fn update_homeostatic_mechanisms(&mut self, dt: f64) {
        // Intrinsic plasticity: adjust excitability to maintain target firing rate
        let rate_error = self.target_firing_rate - self.actual_firing_rate;
        let excitability_update = 0.001 * rate_error * dt;
        self.intrinsic_excitability += excitability_update;
        self.intrinsic_excitability = self.intrinsic_excitability.clamp(0.1, 10.0);

        // Synaptic scaling: global scaling of all synapses
        let scaling_rate = 0.0001;
        let scaling_update = scaling_rate * rate_error * dt;
        self.synaptic_scaling += scaling_update;
        self.synaptic_scaling = self.synaptic_scaling.clamp(0.1, 10.0);
    }

    /// Get homeostatic factor for learning modulation
    pub fn get_homeostatic_factor(&self) -> f64 {
        // Higher factor when firing rate is below target (need to strengthen synapses)
        let rate_ratio = self.actual_firing_rate / self.target_firing_rate.max(1e-6);
        (2.0 / (1.0 + rate_ratio)).clamp(0.1, 10.0)
    }
}

/// Learning rate adaptation mechanisms
///
/// Adapts the learning rate based on performance history to optimize
/// learning dynamics during training.
#[derive(Debug, Clone)]
pub struct LearningRateAdaptation {
    /// Base learning rate
    pub base_rate: f64,
    /// Adaptation factor
    pub adaptation_factor: f64,
    /// Performance history
    pub performance_history: VecDeque<f64>,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Maximum learning rate
    pub max_rate: f64,
    /// Minimum learning rate
    pub min_rate: f64,
}

impl LearningRateAdaptation {
    /// Create new learning rate adaptation
    pub fn new(base_rate: f64) -> Self {
        Self {
            base_rate,
            adaptation_factor: 0.1,
            performance_history: VecDeque::new(),
            adaptation_threshold: 0.1,
            max_rate: 0.1,
            min_rate: 1e-6,
        }
    }

    /// Update learning rate based on performance
    pub fn update_learning_rate(&mut self, current_error: f64) {
        self.performance_history.push_back(current_error);

        // Keep only recent errors
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        // Adapt learning rate based on error trend
        if self.performance_history.len() >= 2 {
            let recent_avg = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older_avg = self
                .performance_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .sum::<f64>()
                / 10.0;

            let performance_ratio = if older_avg > 0.0 {
                recent_avg / older_avg
            } else {
                1.0
            };

            // Adapt learning rate
            if performance_ratio < 0.95 {
                // Performance improving - increase learning rate slightly
                self.base_rate *= 1.01;
            } else if performance_ratio > 1.05 {
                // Performance degrading - decrease learning rate
                self.base_rate *= 0.99;
            }

            // Apply bounds
            self.base_rate = self.base_rate.max(self.min_rate).min(self.max_rate);
        }
    }
}

/// Metaplasticity controller for flexible learning
///
/// Controls learning plasticity based on activity history to implement
/// metaplasticity - plasticity of plasticity itself.
#[derive(Debug, Clone)]
pub struct MetaplasticityController {
    /// Metaplastic variables for each synapse
    metaplastic_variables: Array2<f64>,
    /// Metaplastic time constant
    meta_tau: f64,
    /// Plasticity threshold
    #[allow(dead_code)]
    plasticity_threshold: f64,
    /// LTP/LTD balance factor
    #[allow(dead_code)]
    ltp_ltd_balance: f64,
    /// Activity-dependent scaling
    #[allow(dead_code)]
    activity_scaling: f64,
}

impl MetaplasticityController {
    /// Create new metaplasticity controller
    pub fn new(num_clusters: usize, input_dim: usize) -> Self {
        Self {
            metaplastic_variables: Array2::ones((num_clusters, input_dim)),
            meta_tau: 10000.0, // 10 seconds
            plasticity_threshold: 0.5,
            ltp_ltd_balance: 1.0,
            activity_scaling: 1.0,
        }
    }

    /// Compute metaplastic modulation
    pub fn compute_modulation(&self, neuron_idx: usize, error: f64) -> f64 {
        let meta_var_avg = self
            .metaplastic_variables
            .row(neuron_idx)
            .mean()
            .unwrap_or(1.0);

        // Higher metaplastic variable means lower plasticity (harder to change)
        let modulation = 1.0 / (1.0 + meta_var_avg);

        // Scale by error magnitude
        modulation * (1.0 + error.abs()).ln()
    }

    /// Update metaplastic variables
    pub fn update_metaplastic_variables(
        &mut self,
        winner_idx: usize,
        activities: &Array1<f64>,
        _current_time: f64,
    ) {
        let dt = 1.0; // Assume 1ms updates
        let decay_factor = (-dt / self.meta_tau).exp();

        // Update metaplastic variables for winner
        for meta_var in self.metaplastic_variables.row_mut(winner_idx).iter_mut() {
            *meta_var = *meta_var * decay_factor + (1.0 - decay_factor) * activities[winner_idx];
        }
    }
}

/// Multi-timescale adaptation for different learning phases
///
/// Implements adaptation mechanisms operating at different timescales
/// from fast (milliseconds) to slow (minutes/hours) dynamics.
#[derive(Debug, Clone)]
pub struct MultiTimescaleAdaptation {
    /// Fast adaptation (seconds to minutes)
    fast_adaptation: AdaptationScale,
    /// Medium adaptation (minutes to hours)
    medium_adaptation: AdaptationScale,
    /// Slow adaptation (hours to days)
    slow_adaptation: AdaptationScale,
    /// Current timescale weights
    timescale_weights: Array1<f64>,
}

impl MultiTimescaleAdaptation {
    /// Create new multi-timescale adaptation
    pub fn new() -> Self {
        Self {
            fast_adaptation: AdaptationScale::new(1.0, 1.0), // 1ms timescale
            medium_adaptation: AdaptationScale::new(1000.0, 0.5), // 1s timescale
            slow_adaptation: AdaptationScale::new(60000.0, 0.1), // 1min timescale
            timescale_weights: Array1::from(vec![0.5, 0.3, 0.2]),
        }
    }

    /// Update all adaptation scales
    pub fn update(&mut self, error: f64, dt: f64) {
        self.fast_adaptation.update(error, dt);
        self.medium_adaptation.update(error, dt);
        self.slow_adaptation.update(error, dt);
    }

    /// Get combined adaptation factor
    pub fn get_adaptation_factor(&self) -> f64 {
        let fast_factor = self.fast_adaptation.memory_trace;
        let medium_factor = self.medium_adaptation.memory_trace;
        let slow_factor = self.slow_adaptation.memory_trace;

        self.timescale_weights[0] * fast_factor
            + self.timescale_weights[1] * medium_factor
            + self.timescale_weights[2] * slow_factor
    }
}

/// Individual adaptation scale
///
/// Represents adaptation operating at a specific timescale with
/// exponential memory traces and decay dynamics.
#[derive(Debug, Clone)]
pub struct AdaptationScale {
    /// Time constant for this scale
    time_constant: f64,
    /// Adaptation strength
    #[allow(dead_code)]
    adaptation_strength: f64,
    /// Memory trace
    pub memory_trace: f64,
    /// Decay factor
    #[allow(dead_code)]
    decay_factor: f64,
}

impl AdaptationScale {
    /// Create new adaptation scale
    pub fn new(time_constant: f64, adaptation_strength: f64) -> Self {
        Self {
            time_constant,
            adaptation_strength,
            memory_trace: 1.0,
            decay_factor: 0.999,
        }
    }

    /// Update this adaptation scale
    pub fn update(&mut self, error: f64, dt: f64) {
        let decay = (-dt / self.time_constant).exp();
        self.memory_trace = self.memory_trace * decay + (1.0 - decay) * (1.0 - error);
        self.memory_trace = self.memory_trace.clamp(0.0, 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_competitive_clusterer_creation() {
        let clusterer = CompetitiveNeuralClusterer::new(3, 2);
        assert_eq!(clusterer.num_clusters(), 3);
        assert_eq!(clusterer.learning_rates().len(), 3);
        assert_eq!(clusterer.inhibition_strengths().dim(), (3, 3));
    }

    #[test]
    fn test_competitive_clustering() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut clusterer = CompetitiveNeuralClusterer::new(2, 2);
        let result = clusterer.fit(&points.view(), 10);
        assert!(result.is_ok());

        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 4);

        let centers = clusterer.get_cluster_centers();
        assert_eq!(centers.dim(), (2, 2));
    }

    #[test]
    fn test_homeostatic_clusterer_creation() {
        let clusterer = HomeostaticNeuralClusterer::new(3, 2);
        assert_eq!(clusterer.num_clusters(), 3);
        assert_eq!(clusterer.neuron_firing_rates().len(), 3);
    }

    #[test]
    fn test_homeostatic_clustering() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut clusterer =
            HomeostaticNeuralClusterer::new(2, 2).with_homeostatic_params(0.1, 100.0);

        let result = clusterer.fit(&points.view(), 10);
        assert!(result.is_ok());

        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 4);

        let centers = clusterer.get_cluster_centers();
        assert_eq!(centers.dim(), (2, 2));
    }

    #[test]
    fn test_homeostatic_neuron() {
        let mut neuron = HomeostaticNeuron::new();

        // Test membrane potential update
        neuron.update_membrane_potential(1.0, 10.0);
        assert!(neuron.membrane_potential > 0.0);

        // Test spike recording
        neuron.record_spike(10.0);
        assert_eq!(neuron.spike_history.len(), 1);

        // Test homeostatic mechanisms
        neuron.update_homeostatic_mechanisms(1.0);
        assert!(neuron.intrinsic_excitability > 0.0);
        assert!(neuron.synaptic_scaling > 0.0);
    }

    #[test]
    fn test_learning_rate_adaptation() {
        let mut adaptation = LearningRateAdaptation::new(0.01);
        let initial_rate = adaptation.base_rate;

        // Simulate improving performance
        adaptation.update_learning_rate(1.0);
        adaptation.update_learning_rate(0.8);
        adaptation.update_learning_rate(0.6);

        // Rate should still be within bounds
        assert!(adaptation.base_rate >= adaptation.min_rate);
        assert!(adaptation.base_rate <= adaptation.max_rate);
    }

    #[test]
    fn test_metaplasticity_controller() {
        let controller = MetaplasticityController::new(2, 3);

        let modulation = controller.compute_modulation(0, 0.5);
        assert!(modulation > 0.0);
        assert!(modulation.is_finite());
    }

    #[test]
    fn test_multi_timescale_adaptation() {
        let mut adaptation = MultiTimescaleAdaptation::new();

        // Test update
        adaptation.update(0.5, 1.0);

        let factor = adaptation.get_adaptation_factor();
        assert!(factor > 0.0);
        assert!(factor.is_finite());
    }

    #[test]
    fn test_adaptation_scale() {
        let mut scale = AdaptationScale::new(10.0, 0.5);
        let initial_trace = scale.memory_trace;

        scale.update(0.2, 1.0);

        // Memory trace should have changed
        assert_ne!(scale.memory_trace, initial_trace);
        assert!(scale.memory_trace >= 0.0);
        assert!(scale.memory_trace <= 2.0);
    }

    #[test]
    fn test_empty_input() {
        let points = Array2::zeros((0, 2));

        let mut competitive = CompetitiveNeuralClusterer::new(2, 2);
        let result = competitive.fit(&points.view(), 10);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        let mut homeostatic = HomeostaticNeuralClusterer::new(2, 2);
        let result = homeostatic.fit(&points.view(), 10);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let points = Array2::zeros((4, 3)); // Wrong dimension
        let mut clusterer = HomeostaticNeuralClusterer::new(2, 2);

        let result = clusterer.fit(&points.view(), 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_competitive_reset() {
        let mut clusterer = CompetitiveNeuralClusterer::new(2, 2);

        // Modify some parameters
        clusterer.learning_rates[0] = 0.5;

        // Reset should restore initial state
        clusterer.reset();
        assert_eq!(clusterer.learning_rates[0], 0.1);
    }
}
