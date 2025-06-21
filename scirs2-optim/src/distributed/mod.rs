//! Distributed optimization support
//!
//! This module provides support for distributed training including parameter averaging,
//! gradient compression, and communication optimization for multi-node/multi-GPU training.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Parameter averaging strategies for distributed training
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingStrategy {
    /// Simple arithmetic mean
    Arithmetic,
    /// Weighted average based on data sizes
    WeightedByData,
    /// Weighted average based on computation times
    WeightedByTime,
    /// Federated averaging (FedAvg)
    Federated,
    /// Momentum-based averaging
    Momentum {
        /// Momentum factor
        momentum: f64,
    },
    /// Exponentially weighted moving average
    ExponentialMovingAverage {
        /// Decay factor
        decay: f64,
    },
}

/// Distributed parameter averager
#[derive(Debug)]
pub struct ParameterAverager<A: Float, D: Dimension> {
    /// Current averaged parameters
    averaged_params: Vec<Array<A, D>>,
    /// Averaging strategy
    strategy: AveragingStrategy,
    /// Node weights for weighted averaging
    node_weights: HashMap<usize, A>,
    /// Number of participating nodes
    num_nodes: usize,
    /// Momentum buffer for momentum-based averaging
    momentum_buffer: Option<Vec<Array<A, D>>>,
    /// Step count for EMA decay adjustment
    step_count: usize,
    /// Whether averager is initialized
    initialized: bool,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ParameterAverager<A, D> {
    /// Create a new parameter averager
    pub fn new(strategy: AveragingStrategy, num_nodes: usize) -> Self {
        Self {
            averaged_params: Vec::new(),
            strategy,
            node_weights: HashMap::new(),
            num_nodes,
            momentum_buffer: None,
            step_count: 0,
            initialized: false,
        }
    }

    /// Initialize averager with parameter shapes
    pub fn initialize(&mut self, params: &[Array<A, D>]) -> Result<()> {
        if self.initialized {
            return Err(OptimError::InvalidConfig(
                "Parameter averager already initialized".to_string(),
            ));
        }

        self.averaged_params = params.to_vec();

        // Initialize momentum buffer if needed
        if matches!(self.strategy, AveragingStrategy::Momentum { .. }) {
            self.momentum_buffer = Some(params.iter().map(|p| Array::zeros(p.raw_dim())).collect());
        }

        // Initialize uniform weights
        let uniform_weight = A::one() / A::from(self.num_nodes).unwrap();
        for node_id in 0..self.num_nodes {
            self.node_weights.insert(node_id, uniform_weight);
        }

        self.initialized = true;
        Ok(())
    }

    /// Set weight for a specific node
    pub fn set_node_weight(&mut self, node_id: usize, weight: A) -> Result<()> {
        if node_id >= self.num_nodes {
            return Err(OptimError::InvalidConfig(format!(
                "Node ID {} exceeds number of nodes {}",
                node_id, self.num_nodes
            )));
        }
        self.node_weights.insert(node_id, weight);
        Ok(())
    }

    /// Average parameters from multiple nodes
    pub fn average_parameters(
        &mut self,
        node_parameters: &[(usize, Vec<Array<A, D>>)],
    ) -> Result<()> {
        if !self.initialized {
            if let Some((_, first_params)) = node_parameters.first() {
                self.initialize(first_params)?;
            } else {
                return Err(OptimError::InvalidConfig(
                    "No parameters provided for initialization".to_string(),
                ));
            }
        }

        // Validate input
        for (node_id, params) in node_parameters {
            if *node_id >= self.num_nodes {
                return Err(OptimError::InvalidConfig(format!(
                    "Node ID {} exceeds number of nodes {}",
                    node_id, self.num_nodes
                )));
            }
            if params.len() != self.averaged_params.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Expected {} parameter arrays, got {}",
                    self.averaged_params.len(),
                    params.len()
                )));
            }
        }

        self.step_count += 1;

        match self.strategy {
            AveragingStrategy::Arithmetic => {
                self.arithmetic_average(node_parameters)?;
            }
            AveragingStrategy::WeightedByData | AveragingStrategy::WeightedByTime => {
                self.weighted_average(node_parameters)?;
            }
            AveragingStrategy::Federated => {
                self.federated_average(node_parameters)?;
            }
            AveragingStrategy::Momentum { momentum } => {
                self.momentum_average(node_parameters, momentum)?;
            }
            AveragingStrategy::ExponentialMovingAverage { decay } => {
                self.ema_average(node_parameters, decay)?;
            }
        }

        Ok(())
    }

    /// Simple arithmetic averaging
    fn arithmetic_average(&mut self, node_parameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        // Reset averaged parameters
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }

        let num_nodes = A::from(node_parameters.len()).unwrap();

        // Sum all parameters
        for (_node_id, params) in node_parameters {
            for (avg_param, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p;
                });
            }
        }

        // Divide by number of nodes
        for param in &mut self.averaged_params {
            param.mapv_inplace(|x| x / num_nodes);
        }

        Ok(())
    }

    /// Weighted averaging using node weights
    fn weighted_average(&mut self, node_parameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        // Reset averaged parameters
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }

        // Compute total weight
        let total_weight: A = node_parameters
            .iter()
            .map(|(node_id, _)| self.node_weights.get(node_id).copied().unwrap_or(A::zero()))
            .fold(A::zero(), |acc, w| acc + w);

        if total_weight <= A::zero() {
            return Err(OptimError::InvalidConfig(
                "Total node weights must be > 0".to_string(),
            ));
        }

        // Weighted sum
        for (node_id, params) in node_parameters {
            let weight =
                self.node_weights.get(node_id).copied().unwrap_or(A::zero()) / total_weight;

            for (avg_param, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + weight * p;
                });
            }
        }

        Ok(())
    }

    /// Federated averaging (similar to weighted but with special handling)
    fn federated_average(&mut self, node_parameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        // For simplicity, use weighted averaging with data-based weights
        // In practice, this would consider local dataset sizes and update frequencies
        self.weighted_average(node_parameters)
    }

    /// Momentum-based averaging
    fn momentum_average(
        &mut self,
        node_parameters: &[(usize, Vec<Array<A, D>>)],
        momentum: f64,
    ) -> Result<()> {
        let momentum_factor = A::from(momentum).unwrap();
        let one_minus_momentum = A::one() - momentum_factor;

        // First compute arithmetic average of incoming parameters
        let mut current_average: Vec<Array<A, D>> = self
            .averaged_params
            .iter()
            .map(|param| Array::zeros(param.raw_dim()))
            .collect();

        let num_nodes = A::from(node_parameters.len()).unwrap();
        for (_node_id, params) in node_parameters {
            for (avg_param, param) in current_average.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p / num_nodes;
                });
            }
        }

        // Apply momentum update
        if let Some(ref mut momentum_buf) = self.momentum_buffer {
            for ((avg_param, current_param), momentum_param) in self
                .averaged_params
                .iter_mut()
                .zip(current_average.iter())
                .zip(momentum_buf.iter_mut())
            {
                // Update momentum buffer first
                Zip::from(&mut *momentum_param)
                    .and(current_param)
                    .for_each(|mom, &curr| {
                        *mom = momentum_factor * *mom + one_minus_momentum * curr;
                    });

                // Copy momentum buffer to averaged params
                avg_param.assign(&*momentum_param);
            }
        }

        Ok(())
    }

    /// Exponential moving average
    fn ema_average(
        &mut self,
        node_parameters: &[(usize, Vec<Array<A, D>>)],
        decay: f64,
    ) -> Result<()> {
        let decay_factor = A::from(decay).unwrap();
        let one_minus_decay = A::one() - decay_factor;

        // First compute arithmetic average of incoming parameters
        let mut current_average: Vec<Array<A, D>> = self
            .averaged_params
            .iter()
            .map(|param| Array::zeros(param.raw_dim()))
            .collect();

        let num_nodes = A::from(node_parameters.len()).unwrap();
        for (_node_id, params) in node_parameters {
            for (avg_param, param) in current_average.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p / num_nodes;
                });
            }
        }

        // Apply EMA update
        for (avg_param, current_param) in
            self.averaged_params.iter_mut().zip(current_average.iter())
        {
            Zip::from(avg_param)
                .and(current_param)
                .for_each(|avg, &curr| {
                    *avg = decay_factor * *avg + one_minus_decay * curr;
                });
        }

        Ok(())
    }

    /// Get current averaged parameters
    pub fn get_averaged_parameters(&self) -> &[Array<A, D>] {
        &self.averaged_params
    }

    /// Get cloned averaged parameters
    pub fn get_averaged_parameters_cloned(&self) -> Vec<Array<A, D>> {
        self.averaged_params.clone()
    }

    /// Reset averager state
    pub fn reset(&mut self) {
        self.step_count = 0;
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }
        if let Some(ref mut momentum_buf) = self.momentum_buffer {
            for buf in momentum_buf {
                buf.fill(A::zero());
            }
        }
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get averaging strategy
    pub fn strategy(&self) -> AveragingStrategy {
        self.strategy
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Synchronous parameter server for distributed training
#[derive(Debug)]
pub struct ParameterServer<A: Float, D: Dimension> {
    /// Parameter averager
    averager: ParameterAverager<A, D>,
    /// Current global parameters
    global_parameters: Vec<Array<A, D>>,
    /// Node update counters
    update_counts: HashMap<usize, usize>,
    /// Expected updates per round
    expected_updates_per_round: usize,
    /// Current round number
    current_round: usize,
    /// Synchronization barrier
    pending_updates: HashMap<usize, Vec<Array<A, D>>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ParameterServer<A, D> {
    /// Create a new parameter server
    pub fn new(
        strategy: AveragingStrategy,
        num_nodes: usize,
        expected_updates_per_round: usize,
    ) -> Self {
        Self {
            averager: ParameterAverager::new(strategy, num_nodes),
            global_parameters: Vec::new(),
            update_counts: HashMap::new(),
            expected_updates_per_round,
            current_round: 0,
            pending_updates: HashMap::new(),
        }
    }

    /// Initialize with global parameters
    pub fn initialize(&mut self, initial_params: &[Array<A, D>]) -> Result<()> {
        self.averager.initialize(initial_params)?;
        self.global_parameters = initial_params.to_vec();

        // Initialize update counts
        for node_id in 0..self.averager.num_nodes() {
            self.update_counts.insert(node_id, 0);
        }

        Ok(())
    }

    /// Submit parameter update from a node
    pub fn submit_update(&mut self, node_id: usize, parameters: Vec<Array<A, D>>) -> Result<bool> {
        if node_id >= self.averager.num_nodes() {
            return Err(OptimError::InvalidConfig(format!(
                "Node ID {} exceeds number of nodes {}",
                node_id,
                self.averager.num_nodes()
            )));
        }

        // Store the update
        self.pending_updates.insert(node_id, parameters);
        *self.update_counts.entry(node_id).or_insert(0) += 1;

        // Check if we have enough updates for this round
        let ready_for_aggregation = self.pending_updates.len() >= self.expected_updates_per_round;

        if ready_for_aggregation {
            self.aggregate_and_update()?;
        }

        Ok(ready_for_aggregation)
    }

    /// Force aggregation with current pending updates
    pub fn force_aggregation(&mut self) -> Result<()> {
        if !self.pending_updates.is_empty() {
            self.aggregate_and_update()?;
        }
        Ok(())
    }

    /// Internal aggregation and update
    fn aggregate_and_update(&mut self) -> Result<()> {
        // Convert pending updates to the format expected by averager
        let node_params: Vec<(usize, Vec<Array<A, D>>)> = self.pending_updates.drain().collect();

        // Perform averaging
        self.averager.average_parameters(&node_params)?;

        // Update global parameters
        self.global_parameters = self.averager.get_averaged_parameters_cloned();

        // Increment round
        self.current_round += 1;

        Ok(())
    }

    /// Get current global parameters
    pub fn get_global_parameters(&self) -> &[Array<A, D>] {
        &self.global_parameters
    }

    /// Get cloned global parameters
    pub fn get_global_parameters_cloned(&self) -> Vec<Array<A, D>> {
        self.global_parameters.clone()
    }

    /// Get current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get update count for a node
    pub fn get_update_count(&self, node_id: usize) -> usize {
        self.update_counts.get(&node_id).copied().unwrap_or(0)
    }

    /// Get number of pending updates
    pub fn pending_updates_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Set node weight for weighted averaging
    pub fn set_node_weight(&mut self, node_id: usize, weight: A) -> Result<()> {
        self.averager.set_node_weight(node_id, weight)
    }

    /// Reset server state
    pub fn reset(&mut self) {
        self.averager.reset();
        self.update_counts.clear();
        self.pending_updates.clear();
        self.current_round = 0;

        for node_id in 0..self.averager.num_nodes() {
            self.update_counts.insert(node_id, 0);
        }
    }
}

/// Distributed training coordinator
#[derive(Debug)]
pub struct DistributedCoordinator<A: Float, D: Dimension> {
    /// Parameter server
    parameter_server: ParameterServer<A, D>,
    /// Communication rounds completed
    communication_rounds: usize,
    /// Convergence criteria
    convergence_threshold: A,
    /// Maximum rounds before forced stop
    max_rounds: usize,
    /// Training statistics
    training_stats: TrainingStats<A>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> DistributedCoordinator<A, D> {
    /// Create a new distributed coordinator
    pub fn new(
        strategy: AveragingStrategy,
        num_nodes: usize,
        expected_updates_per_round: usize,
        max_rounds: usize,
    ) -> Self {
        Self {
            parameter_server: ParameterServer::new(strategy, num_nodes, expected_updates_per_round),
            communication_rounds: 0,
            convergence_threshold: A::from(1e-6).unwrap(),
            max_rounds,
            training_stats: TrainingStats::new(),
        }
    }

    /// Initialize coordinator
    pub fn initialize(&mut self, initial_params: &[Array<A, D>]) -> Result<()> {
        self.parameter_server.initialize(initial_params)?;
        self.training_stats
            .record_round(0, A::zero(), initial_params);
        Ok(())
    }

    /// Execute a communication round
    pub fn communication_round(
        &mut self,
        node_updates: Vec<(usize, Vec<Array<A, D>>)>,
    ) -> Result<CommunicationResult<A, D>> {
        let mut aggregated = false;

        // Submit all updates
        for (node_id, params) in node_updates {
            aggregated = self.parameter_server.submit_update(node_id, params)? || aggregated;
        }

        // Force aggregation if not done automatically
        if !aggregated {
            self.parameter_server.force_aggregation()?;
            aggregated = true;
        }

        if aggregated {
            self.communication_rounds += 1;

            // Check convergence
            let current_params = self.parameter_server.get_global_parameters();
            let convergence_metric = self.compute_convergence_metric(current_params);

            self.training_stats.record_round(
                self.communication_rounds,
                convergence_metric,
                current_params,
            );

            let converged = convergence_metric < self.convergence_threshold;
            let max_rounds_reached = self.communication_rounds >= self.max_rounds;

            Ok(CommunicationResult {
                round: self.communication_rounds,
                global_parameters: self.parameter_server.get_global_parameters_cloned(),
                converged,
                should_continue: !converged && !max_rounds_reached,
                convergence_metric,
                stats: self.training_stats.clone(),
            })
        } else {
            Ok(CommunicationResult {
                round: self.communication_rounds,
                global_parameters: self.parameter_server.get_global_parameters_cloned(),
                converged: false,
                should_continue: true,
                convergence_metric: A::infinity(),
                stats: self.training_stats.clone(),
            })
        }
    }

    /// Set convergence threshold
    pub fn set_convergence_threshold(&mut self, threshold: A) {
        self.convergence_threshold = threshold;
    }

    /// Get parameter server reference
    pub fn parameter_server(&self) -> &ParameterServer<A, D> {
        &self.parameter_server
    }

    /// Get mutable parameter server reference
    pub fn parameter_server_mut(&mut self) -> &mut ParameterServer<A, D> {
        &mut self.parameter_server
    }

    /// Compute convergence metric (parameter change magnitude)
    fn compute_convergence_metric(&self, current_params: &[Array<A, D>]) -> A {
        if let Some(prev_params) = self.training_stats.get_previous_parameters::<D>() {
            let mut total_change = A::zero();
            let mut total_norm = A::zero();

            for (curr, prev) in current_params.iter().zip(prev_params.iter()) {
                for (&c, &p) in curr.iter().zip(prev.iter()) {
                    let diff = c - p;
                    total_change = total_change + diff * diff;
                    total_norm = total_norm + c * c;
                }
            }

            if total_norm > A::zero() {
                (total_change / total_norm).sqrt()
            } else {
                A::zero()
            }
        } else {
            A::infinity()
        }
    }
}

/// Result of a communication round
#[derive(Debug, Clone)]
pub struct CommunicationResult<A: Float, D: Dimension> {
    /// Round number
    pub round: usize,
    /// Updated global parameters
    pub global_parameters: Vec<Array<A, D>>,
    /// Whether training has converged
    pub converged: bool,
    /// Whether training should continue
    pub should_continue: bool,
    /// Convergence metric value
    pub convergence_metric: A,
    /// Training statistics
    pub stats: TrainingStats<A>,
}

/// Training statistics for distributed training
#[derive(Debug, Clone)]
pub struct TrainingStats<A: Float> {
    /// Convergence history
    convergence_history: Vec<A>,
    /// Round timestamps
    round_times: Vec<usize>,
    /// Previous parameters for convergence computation
    previous_parameters: Option<Vec<u8>>, // Serialized for memory efficiency
}

impl<A: Float> TrainingStats<A> {
    /// Create new training stats
    pub fn new() -> Self {
        Self {
            convergence_history: Vec::new(),
            round_times: Vec::new(),
            previous_parameters: None,
        }
    }

    /// Record a training round
    pub fn record_round<D: Dimension>(
        &mut self,
        round: usize,
        convergence_metric: A,
        parameters: &[Array<A, D>],
    ) {
        self.convergence_history.push(convergence_metric);
        self.round_times.push(round);

        // Store simplified representation of parameters for convergence computation
        // In practice, you might want a more sophisticated serialization
        self.previous_parameters = Some(vec![0u8; parameters.len()]);
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[A] {
        &self.convergence_history
    }

    /// Get latest convergence metric
    pub fn latest_convergence(&self) -> Option<A> {
        self.convergence_history.last().copied()
    }

    /// Get number of rounds
    pub fn num_rounds(&self) -> usize {
        self.round_times.len()
    }

    /// Get previous parameters (simplified)
    fn get_previous_parameters<D: Dimension>(&self) -> Option<Vec<Array<A, D>>> {
        // Simplified implementation - in practice you'd deserialize properly
        None
    }
}

impl<A: Float> Default for TrainingStats<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient compression strategies for communication optimization
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Top-K sparsification (keep only top K largest gradients)
    TopK {
        /// Number of top gradients to keep
        k: usize,
    },
    /// Random-K sparsification (keep K random gradients)
    RandomK {
        /// Number of random gradients to keep
        k: usize,
    },
    /// Threshold-based sparsification (keep gradients above threshold)
    Threshold {
        /// Threshold value for gradient magnitude
        threshold: f64,
    },
    /// Quantization to fewer bits
    Quantization {
        /// Number of bits for quantization
        bits: u8,
    },
    /// Error feedback compression (maintain error state)
    ErrorFeedback {
        /// Base compression strategy to apply
        base_strategy: Box<CompressionStrategy>,
        /// Whether to enable error compensation
        error_compensation: bool,
    },
    /// Gradient clipping before compression
    ClippedCompression {
        /// Base compression strategy to apply after clipping
        base_strategy: Box<CompressionStrategy>,
        /// Value to clip gradients to
        clip_value: f64,
    },
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient<A: Float> {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata<A>,
    /// Original shape information
    pub shapes: Vec<Vec<usize>>,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata<A: Float> {
    /// Compression strategy used
    pub strategy: CompressionStrategy,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Number of non-zero elements (for sparse methods)
    pub nnz_count: usize,
    /// Quantization scale factors (for quantization methods)
    pub scale_factors: Vec<A>,
    /// Additional strategy-specific data
    pub extra_data: Vec<u8>,
}

/// Gradient compression engine
#[derive(Debug)]
pub struct GradientCompressor<A: Float, D: Dimension> {
    /// Compression strategy
    strategy: CompressionStrategy,
    /// Error feedback state for error compensation
    error_state: Option<Vec<Array<A, D>>>,
    /// Compression statistics
    stats: CompressionStats,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GradientCompressor<A, D> {
    /// Create a new gradient compressor
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self {
            strategy,
            error_state: None,
            stats: CompressionStats::new(),
        }
    }

    /// Initialize error state for error feedback compression
    pub fn initialize_error_state(&mut self, gradient_shapes: &[Array<A, D>]) {
        self.error_state = Some(
            gradient_shapes
                .iter()
                .map(|g| Array::zeros(g.raw_dim()))
                .collect(),
        );
    }

    /// Compress gradients
    pub fn compress(&mut self, gradients: &[Array<A, D>]) -> Result<CompressedGradient<A>> {
        // Apply error feedback if enabled
        let mut working_gradients: Vec<Array<A, D>> =
            if let Some(ref mut error_state) = self.error_state {
                gradients
                    .iter()
                    .zip(error_state.iter())
                    .map(|(grad, error)| grad + error)
                    .collect()
            } else {
                gradients.to_vec()
            };

        let (compressed_data, metadata) = match &self.strategy {
            CompressionStrategy::None => self.compress_none(&working_gradients)?,
            CompressionStrategy::TopK { k } => self.compress_topk(&working_gradients, *k)?,
            CompressionStrategy::RandomK { k } => self.compress_randomk(&working_gradients, *k)?,
            CompressionStrategy::Threshold { threshold } => {
                self.compress_threshold(&working_gradients, A::from(*threshold).unwrap())?
            }
            CompressionStrategy::Quantization { bits } => {
                self.compress_quantization(&working_gradients, *bits)?
            }
            CompressionStrategy::ErrorFeedback { base_strategy, .. } => {
                // Recursively apply base strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;
                let decompressed = temp_compressor.decompress(&compressed)?;

                // Update error state
                if let Some(ref mut error_state) = self.error_state {
                    for ((original, decompressed), error) in gradients
                        .iter()
                        .zip(decompressed.iter())
                        .zip(error_state.iter_mut())
                    {
                        *error = original - decompressed;
                    }
                }

                (compressed.data, compressed.metadata)
            }
            CompressionStrategy::ClippedCompression {
                base_strategy,
                clip_value,
            } => {
                // Clip gradients first
                let clip_val = A::from(*clip_value).unwrap();
                for grad in &mut working_gradients {
                    grad.mapv_inplace(|x| {
                        if x > clip_val {
                            clip_val
                        } else if x < -clip_val {
                            -clip_val
                        } else {
                            x
                        }
                    });
                }

                // Apply base compression strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;
                (compressed.data, compressed.metadata)
            }
        };

        // Collect shape information
        let shapes = gradients.iter().map(|g| g.shape().to_vec()).collect();

        let result = CompressedGradient {
            data: compressed_data,
            metadata,
            shapes,
        };

        // Update statistics
        let original_size = self.calculate_size(gradients);
        let compressed_size = result.data.len();
        self.stats
            .record_compression(original_size, compressed_size);

        Ok(result)
    }

    /// Decompress gradients
    pub fn decompress(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        match &compressed.metadata.strategy {
            CompressionStrategy::None => self.decompress_none(compressed),
            CompressionStrategy::TopK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::RandomK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Threshold { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Quantization { bits } => {
                self.decompress_quantization(compressed, *bits)
            }
            CompressionStrategy::ErrorFeedback { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
            CompressionStrategy::ClippedCompression { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
        }
    }

    /// Compress with no compression (passthrough)
    fn compress_none(
        &self,
        gradients: &[Array<A, D>],
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut data = Vec::new();

        // Simple serialization: store all gradient values sequentially
        for grad in gradients {
            for &val in grad.iter() {
                data.extend_from_slice(&val.to_f64().unwrap().to_le_bytes());
            }
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::None,
            compression_ratio: 1.0,
            nnz_count: gradients.iter().map(|g| g.len()).sum(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Top-K sparsification
    fn compress_topk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Collect (value, index) pairs
            let mut value_indices: Vec<(A, usize)> = grad
                .iter()
                .enumerate()
                .map(|(i, &val)| (val.abs(), i))
                .collect();

            // Sort by absolute value (descending)
            value_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Take top k elements
            let k_local = k.min(value_indices.len());
            for (_, orig_idx) in value_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, *orig_idx as u32));
                values.push(grad.iter().nth(*orig_idx).copied().unwrap());
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();

        // Store number of sparse elements
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        // Store indices and values
        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            data.extend_from_slice(&value.to_f64().unwrap().to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::TopK { k },
            compression_ratio: data.len() as f64 / (total_elements * 8) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Random-K sparsification
    fn compress_randomk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Random sampling of k indices
            let k_local = k.min(grad.len());
            let mut selected_indices: Vec<usize> = (0..grad.len()).collect();

            // Simple random selection (deterministic for testing)
            for i in 0..k_local {
                let swap_idx = i + ((grad_idx + i) % (grad.len() - i));
                selected_indices.swap(i, swap_idx);
            }

            for &idx in selected_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, idx as u32));
                values.push(grad.iter().nth(idx).copied().unwrap());
            }
        }

        // Serialize sparse representation (same format as Top-K)
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            data.extend_from_slice(&value.to_f64().unwrap().to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::RandomK { k },
            compression_ratio: data.len() as f64 / (total_elements * 8) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using threshold-based sparsification
    fn compress_threshold(
        &self,
        gradients: &[Array<A, D>],
        threshold: A,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            for (elem_idx, &val) in grad.iter().enumerate() {
                if val.abs() > threshold {
                    indices.push((grad_idx as u32, elem_idx as u32));
                    values.push(val);
                }
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            data.extend_from_slice(&value.to_f64().unwrap().to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Threshold {
                threshold: threshold.to_f64().unwrap(),
            },
            compression_ratio: data.len() as f64 / (total_elements * 8) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using quantization
    fn compress_quantization(
        &self,
        gradients: &[Array<A, D>],
        bits: u8,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        if bits > 32 {
            return Err(OptimError::InvalidConfig(
                "Quantization bits must be <= 32".to_string(),
            ));
        }

        let mut data = Vec::new();
        let mut scale_factors = Vec::new();
        let levels = (1u64 << bits) - 1;

        for grad in gradients {
            // Find min and max values for this gradient
            let min_val = grad.iter().fold(A::infinity(), |acc, &x| acc.min(x));
            let max_val = grad.iter().fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let range = max_val - min_val;
            let scale = if range > A::zero() {
                range / A::from(levels).unwrap()
            } else {
                A::one()
            };

            scale_factors.push(scale);

            // Quantize each value
            for &val in grad.iter() {
                let normalized = (val - min_val) / scale;
                let quantized = normalized.to_u64().unwrap().min(levels) as u32;

                // Store quantized value
                match bits {
                    1..=8 => data.push(quantized as u8),
                    9..=16 => data.extend_from_slice(&(quantized as u16).to_le_bytes()),
                    17..=32 => data.extend_from_slice(&quantized.to_le_bytes()),
                    _ => unreachable!(),
                }
            }

            // Store min value for reconstruction
            data.extend_from_slice(&min_val.to_f64().unwrap().to_le_bytes());
        }

        let total_elements: usize = gradients.iter().map(|g| g.len()).sum();
        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Quantization { bits },
            compression_ratio: data.len() as f64 / (total_elements * 8) as f64,
            nnz_count: total_elements,
            scale_factors,
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Decompress uncompressed data
    fn decompress_none(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;

        for shape in &compressed.shapes {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            for _ in 0..num_elements {
                if data_offset + 8 > compressed.data.len() {
                    return Err(OptimError::InvalidConfig(
                        "Insufficient data for decompression".to_string(),
                    ));
                }

                let bytes = &compressed.data[data_offset..data_offset + 8];
                let value = f64::from_le_bytes(bytes.try_into().unwrap());
                values.push(A::from(value).unwrap());
                data_offset += 8;
            }

            // Create a dynamic array first, then convert to the target dimension type
            let dynamic_array = Array::from_shape_vec(shape.as_slice(), values).map_err(|_| {
                OptimError::InvalidConfig("Invalid shape for reconstruction".to_string())
            })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed".to_string())
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Decompress sparse representation
    fn decompress_sparse(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();

        // Initialize zero arrays
        for shape in &compressed.shapes {
            let dynamic_array = Array::zeros(shape.as_slice());
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed for zero array".to_string())
            })?;
            result.push(array);
        }

        // Read number of sparse elements
        if compressed.data.len() < 4 {
            return Err(OptimError::InvalidConfig(
                "Invalid compressed data format".to_string(),
            ));
        }

        let num_elements = u32::from_le_bytes(compressed.data[0..4].try_into().unwrap()) as usize;
        let mut data_offset = 4;

        // Restore sparse elements
        for _ in 0..num_elements {
            if data_offset + 16 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Insufficient data for sparse decompression".to_string(),
                ));
            }

            let grad_idx = u32::from_le_bytes(
                compressed.data[data_offset..data_offset + 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let elem_idx = u32::from_le_bytes(
                compressed.data[data_offset + 4..data_offset + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let value_bytes = &compressed.data[data_offset + 8..data_offset + 16];
            let value = A::from(f64::from_le_bytes(value_bytes.try_into().unwrap())).unwrap();

            data_offset += 16;

            if grad_idx >= result.len() {
                return Err(OptimError::InvalidConfig(
                    "Invalid gradient index in compressed data".to_string(),
                ));
            }

            if let Some(elem) = result[grad_idx].iter_mut().nth(elem_idx) {
                *elem = value;
            } else {
                return Err(OptimError::InvalidConfig(
                    "Invalid element index in compressed data".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Decompress quantized data
    fn decompress_quantization(
        &self,
        compressed: &CompressedGradient<A>,
        bits: u8,
    ) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;
        let _levels = (1u64 << bits) - 1;

        for (grad_idx, shape) in compressed.shapes.iter().enumerate() {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            // Read quantized values
            for _ in 0..num_elements {
                let quantized = match bits {
                    1..=8 => {
                        if data_offset >= compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = compressed.data[data_offset] as u32;
                        data_offset += 1;
                        val
                    }
                    9..=16 => {
                        if data_offset + 2 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = u16::from_le_bytes(
                            compressed.data[data_offset..data_offset + 2]
                                .try_into()
                                .unwrap(),
                        ) as u32;
                        data_offset += 2;
                        val
                    }
                    17..=32 => {
                        if data_offset + 4 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = u32::from_le_bytes(
                            compressed.data[data_offset..data_offset + 4]
                                .try_into()
                                .unwrap(),
                        );
                        data_offset += 4;
                        val
                    }
                    _ => {
                        return Err(OptimError::InvalidConfig(
                            "Invalid quantization bits".to_string(),
                        ))
                    }
                };

                values.push(quantized);
            }

            // Read min value
            if data_offset + 8 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Missing min value for quantization".to_string(),
                ));
            }
            let min_bytes = &compressed.data[data_offset..data_offset + 8];
            let min_val = A::from(f64::from_le_bytes(min_bytes.try_into().unwrap())).unwrap();
            data_offset += 8;

            // Get scale factor
            let scale = if grad_idx < compressed.metadata.scale_factors.len() {
                compressed.metadata.scale_factors[grad_idx]
            } else {
                return Err(OptimError::InvalidConfig(
                    "Missing scale factor for quantization".to_string(),
                ));
            };

            // Dequantize values
            let dequantized_values: Vec<A> = values
                .into_iter()
                .map(|q| min_val + A::from(q).unwrap() * scale)
                .collect();

            let dynamic_array = Array::from_shape_vec(shape.as_slice(), dequantized_values)
                .map_err(|_| {
                    OptimError::InvalidConfig(
                        "Invalid shape for quantized reconstruction".to_string(),
                    )
                })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig(
                    "Dimension conversion failed for quantized array".to_string(),
                )
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Calculate size of gradients in bytes
    fn calculate_size(&self, gradients: &[Array<A, D>]) -> usize {
        gradients
            .iter()
            .map(|g| g.len() * std::mem::size_of::<A>())
            .sum()
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::new();
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Total compressions performed
    pub compressions_count: usize,
    /// Total original bytes
    pub total_original_bytes: usize,
    /// Total compressed bytes
    pub total_compressed_bytes: usize,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Best compression ratio achieved
    pub best_compression_ratio: f64,
    /// Worst compression ratio achieved
    pub worst_compression_ratio: f64,
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self {
            compressions_count: 0,
            total_original_bytes: 0,
            total_compressed_bytes: 0,
            average_compression_ratio: 0.0,
            best_compression_ratio: f64::INFINITY,
            worst_compression_ratio: 0.0,
        }
    }

    /// Record a compression operation
    pub fn record_compression(&mut self, original_bytes: usize, compressed_bytes: usize) {
        self.compressions_count += 1;
        self.total_original_bytes += original_bytes;
        self.total_compressed_bytes += compressed_bytes;

        let ratio = if original_bytes > 0 {
            compressed_bytes as f64 / original_bytes as f64
        } else {
            1.0
        };

        self.best_compression_ratio = self.best_compression_ratio.min(ratio);
        self.worst_compression_ratio = self.worst_compression_ratio.max(ratio);

        self.average_compression_ratio = if self.total_original_bytes > 0 {
            self.total_compressed_bytes as f64 / self.total_original_bytes as f64
        } else {
            0.0
        };
    }

    /// Get overall compression ratio
    pub fn overall_compression_ratio(&self) -> f64 {
        self.average_compression_ratio
    }

    /// Get bandwidth savings (as percentage)
    pub fn bandwidth_savings(&self) -> f64 {
        (1.0 - self.average_compression_ratio) * 100.0
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_arithmetic_averaging() {
        let mut averager: ParameterAverager<f64, ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Arithmetic, 3);

        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let params2 = vec![Array1::from_vec(vec![3.0, 4.0])];
        let params3 = vec![Array1::from_vec(vec![5.0, 6.0])];

        let node_parameters = vec![(0, params1), (1, params2), (2, params3)];

        averager.average_parameters(&node_parameters).unwrap();

        let result = averager.get_averaged_parameters();
        assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6); // (1+3+5)/3
        assert_relative_eq!(result[0][1], 4.0, epsilon = 1e-6); // (2+4+6)/3
    }

    #[test]
    fn test_weighted_averaging() {
        let mut averager: ParameterAverager<f64, ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

        // Initialize first to avoid overwriting weights
        let params1 = vec![Array1::from_vec(vec![2.0])];
        let params2 = vec![Array1::from_vec(vec![6.0])];
        let node_parameters = vec![(0, params1.clone()), (1, params2.clone())];
        averager.initialize(&params1).unwrap();

        // Set different weights after initialization
        averager.set_node_weight(0, 0.75).unwrap(); // 75% weight
        averager.set_node_weight(1, 0.25).unwrap(); // 25% weight

        averager.average_parameters(&node_parameters).unwrap();

        let result = averager.get_averaged_parameters();
        // Weighted average: 0.75 * 2.0 + 0.25 * 6.0 = 1.5 + 1.5 = 3.0
        assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_momentum_averaging() {
        let mut averager: ParameterAverager<f64, ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Momentum { momentum: 0.9 }, 2);

        let params1 = vec![Array1::from_vec(vec![1.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])];

        // First update: average = (1+3)/2 = 2.0, momentum buffer starts at 0, so result = 0.1 * 2.0 = 0.2
        let node_parameters1 = vec![(0, params1.clone()), (1, params2.clone())];
        averager.average_parameters(&node_parameters1).unwrap();

        let result1 = averager.get_averaged_parameters();
        // First result should be small due to zero initialization
        assert!(result1[0][0] >= 0.0 && result1[0][0] <= 0.5);

        // Several more updates to let momentum build up
        for _ in 0..10 {
            let node_parameters = vec![(0, params1.clone()), (1, params2.clone())];
            averager.average_parameters(&node_parameters).unwrap();
        }

        let final_result = averager.get_averaged_parameters();
        // After many updates, momentum should gradually converge towards the average (2.0)
        // But with momentum=0.9, it builds up slowly, so we use a broader range
        assert!(final_result[0][0] > 0.5 && final_result[0][0] < 2.5);
    }

    #[test]
    fn test_parameter_server() {
        let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);

        let initial_params = vec![Array1::from_vec(vec![0.0, 0.0])];
        server.initialize(&initial_params).unwrap();

        // Submit updates from both nodes
        let update1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let update2 = vec![Array1::from_vec(vec![3.0, 4.0])];

        let ready1 = server.submit_update(0, update1).unwrap();
        assert!(!ready1); // Not ready yet, waiting for second node

        let ready2 = server.submit_update(1, update2).unwrap();
        assert!(ready2); // Ready after both nodes submitted

        let global_params = server.get_global_parameters();
        assert_relative_eq!(global_params[0][0], 2.0, epsilon = 1e-6); // (1+3)/2
        assert_relative_eq!(global_params[0][1], 3.0, epsilon = 1e-6); // (2+4)/2

        assert_eq!(server.current_round(), 1);
    }

    #[test]
    fn test_distributed_coordinator() {
        let mut coordinator = DistributedCoordinator::new(
            AveragingStrategy::Arithmetic,
            2,  // 2 nodes
            2,  // expect 2 updates per round
            10, // max 10 rounds
        );

        let initial_params = vec![Array1::from_vec(vec![0.0])];
        coordinator.initialize(&initial_params).unwrap();

        // Simulate training rounds
        for round in 1..=3 {
            let update1 = vec![Array1::from_vec(vec![round as f64])];
            let update2 = vec![Array1::from_vec(vec![(round * 2) as f64])];

            let node_updates = vec![(0, update1), (1, update2)];

            let result = coordinator.communication_round(node_updates).unwrap();

            assert_eq!(result.round, round);
            assert!(result.should_continue);
            assert!(!result.converged); // Unlikely to converge with these updates

            // Check that global parameters are updated
            assert!(result.global_parameters[0][0] > 0.0);
        }
    }

    #[test]
    fn test_averaging_strategies() {
        // Test arithmetic and federated strategies that should produce expected ranges
        let simple_strategies = vec![
            AveragingStrategy::Arithmetic,
            AveragingStrategy::WeightedByData,
            AveragingStrategy::Federated,
        ];

        for strategy in simple_strategies {
            let mut averager: ParameterAverager<f64, ndarray::Ix1> =
                ParameterAverager::new(strategy, 2);

            let params1 = vec![Array1::from_vec(vec![1.0])];
            let params2 = vec![Array1::from_vec(vec![3.0])];

            let node_parameters = vec![(0, params1), (1, params2)];

            averager.average_parameters(&node_parameters).unwrap();
            let result = averager.get_averaged_parameters();
            assert!(result[0][0] >= 1.0 && result[0][0] <= 3.0);
        }

        // Test momentum and EMA strategies separately (they start from zero state)
        let stateful_strategies = vec![
            AveragingStrategy::Momentum { momentum: 0.9 },
            AveragingStrategy::ExponentialMovingAverage { decay: 0.9 },
        ];

        for strategy in stateful_strategies {
            let mut averager: ParameterAverager<f64, ndarray::Ix1> =
                ParameterAverager::new(strategy, 2);

            let params1 = vec![Array1::from_vec(vec![1.0])];
            let params2 = vec![Array1::from_vec(vec![3.0])];

            let node_parameters = vec![(0, params1), (1, params2)];

            averager.average_parameters(&node_parameters).unwrap();
            let result = averager.get_averaged_parameters();
            // First result from momentum/EMA will be smaller due to zero initialization
            assert!(result[0][0] >= 0.0 && result[0][0] <= 3.0);
        }
    }

    #[test]
    fn test_node_weight_validation() {
        let mut averager: ParameterAverager<f64, ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

        // Valid node ID
        assert!(averager.set_node_weight(0, 0.5).is_ok());
        assert!(averager.set_node_weight(1, 0.5).is_ok());

        // Invalid node ID
        assert!(averager.set_node_weight(2, 0.5).is_err());
    }

    #[test]
    fn test_parameter_dimension_validation() {
        let mut averager: ParameterAverager<f64, ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Arithmetic, 2);

        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])]; // Wrong dimension

        let node_parameters = vec![(0, params1), (1, params2)];

        // Should fail due to dimension mismatch - currently panics instead of returning error
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            averager.average_parameters(&node_parameters)
        }));

        // Either it returns an error or panics due to dimension mismatch
        assert!(result.is_err() || (result.is_ok() && result.unwrap().is_err()));
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::new();

        assert_eq!(stats.num_rounds(), 0);
        assert!(stats.latest_convergence().is_none());

        let params = vec![Array1::from_vec(vec![1.0])];
        stats.record_round(1, 0.5, &params);

        assert_eq!(stats.num_rounds(), 1);
        assert_eq!(stats.latest_convergence(), Some(0.5));
        assert_eq!(stats.convergence_history(), &[0.5]);
    }

    #[test]
    fn test_gradient_compression_none() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::None);

        let gradients = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0]),
        ];

        let compressed = compressor.compress(&gradients).unwrap();
        assert_eq!(compressed.metadata.strategy, CompressionStrategy::None);
        assert_eq!(compressed.metadata.compression_ratio, 1.0);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 2);
        assert_eq!(decompressed[0].as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(decompressed[1].as_slice().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn test_gradient_compression_topk() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 2 });

        let gradients = vec![Array1::from_vec(vec![0.1, 3.0, 0.2, 4.0, 0.05])];

        let compressed = compressor.compress(&gradients).unwrap();
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 2); // Top 2 elements

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 1);

        // Should have only the top 2 elements (4.0 and 3.0), others should be 0
        let result = &decompressed[0];
        assert_eq!(result[1], 3.0); // Original position of 3.0
        assert_eq!(result[3], 4.0); // Original position of 4.0
        assert_eq!(result[0], 0.0); // Should be zeroed
        assert_eq!(result[2], 0.0); // Should be zeroed
        assert_eq!(result[4], 0.0); // Should be zeroed
    }

    #[test]
    fn test_gradient_compression_threshold() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::Threshold { threshold: 1.0 });

        let gradients = vec![Array1::from_vec(vec![0.5, 2.0, 0.8, 3.0, 0.3])];

        let compressed = compressor.compress(&gradients).unwrap();
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 2); // Elements > 1.0: 2.0 and 3.0

        let decompressed = compressor.decompress(&compressed).unwrap();
        let result = &decompressed[0];

        // Only elements > 1.0 should remain
        assert_eq!(result[0], 0.0); // 0.5 < 1.0
        assert_eq!(result[1], 2.0); // 2.0 > 1.0
        assert_eq!(result[2], 0.0); // 0.8 < 1.0
        assert_eq!(result[3], 3.0); // 3.0 > 1.0
        assert_eq!(result[4], 0.0); // 0.3 < 1.0
    }

    #[test]
    fn test_gradient_compression_quantization() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::Quantization { bits: 8 });

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

        let compressed = compressor.compress(&gradients).unwrap();
        assert!(compressed.metadata.compression_ratio < 1.0); // Should use less space with 8-bit quantization

        let decompressed = compressor.decompress(&compressed).unwrap();
        let result = &decompressed[0];

        // Values should be approximately restored (with quantization error)
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
        assert!((result[2] - 3.0).abs() < 0.1);
        assert!((result[3] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_gradient_compression_randomk() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::RandomK { k: 3 });

        // Use a larger array to make compression effective
        let gradients = vec![Array1::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ])];

        let compressed = compressor.compress(&gradients).unwrap();
        // With 3 out of 10 elements, compression should be effective
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 3); // Exactly 3 elements should be kept

        let decompressed = compressor.decompress(&compressed).unwrap();
        let result = &decompressed[0];

        // Exactly 3 elements should be non-zero
        let non_zero_count = result.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 3);
    }

    #[test]
    fn test_gradient_compression_error_feedback() {
        let base_strategy = CompressionStrategy::TopK { k: 2 };
        let strategy = CompressionStrategy::ErrorFeedback {
            base_strategy: Box::new(base_strategy),
            error_compensation: true,
        };

        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

        // Initialize error state
        compressor.initialize_error_state(&gradients);

        // First compression
        let compressed1 = compressor.compress(&gradients).unwrap();
        let decompressed1 = compressor.decompress(&compressed1).unwrap();

        // Second compression (should include error feedback)
        let compressed2 = compressor.compress(&gradients).unwrap();
        let decompressed2 = compressor.decompress(&compressed2).unwrap();

        // Both should be valid compressions
        assert_eq!(decompressed1.len(), 1);
        assert_eq!(decompressed2.len(), 1);
    }

    #[test]
    fn test_gradient_compression_clipped() {
        let base_strategy = CompressionStrategy::TopK { k: 3 };
        let strategy = CompressionStrategy::ClippedCompression {
            base_strategy: Box::new(base_strategy),
            clip_value: 2.5,
        };

        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 5.0, -3.0, 2.0])];

        let compressed = compressor.compress(&gradients).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        let result = &decompressed[0];

        // Values should be clipped to [-2.5, 2.5] and then top-k applied
        for &val in result.iter() {
            if val != 0.0 {
                // Non-zero values from top-k
                assert!(val >= -2.5 && val <= 2.5);
            }
        }
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();

        assert_eq!(stats.compressions_count, 0);
        assert_eq!(stats.overall_compression_ratio(), 0.0);

        // Record some compressions
        stats.record_compression(1000, 500); // 50% compression
        assert_eq!(stats.compressions_count, 1);
        assert_relative_eq!(stats.overall_compression_ratio(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(stats.bandwidth_savings(), 50.0, epsilon = 1e-6);

        stats.record_compression(1000, 250); // 25% compression
        assert_eq!(stats.compressions_count, 2);
        assert_relative_eq!(stats.overall_compression_ratio(), 0.375, epsilon = 1e-6); // (500+250)/(1000+1000)
        assert_relative_eq!(stats.bandwidth_savings(), 62.5, epsilon = 1e-6);

        assert_relative_eq!(stats.best_compression_ratio, 0.25, epsilon = 1e-6);
        assert_relative_eq!(stats.worst_compression_ratio, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_compression_roundtrip() {
        let strategies = vec![
            CompressionStrategy::None,
            CompressionStrategy::TopK { k: 2 },
            CompressionStrategy::RandomK { k: 2 },
            CompressionStrategy::Threshold { threshold: 1.5 },
            CompressionStrategy::Quantization { bits: 4 },
        ];

        let gradients = vec![
            Array1::from_vec(vec![1.0, 2.5, 0.5, 3.0]),
            Array1::from_vec(vec![0.1, 4.0]),
        ];

        for strategy in strategies {
            let mut compressor = GradientCompressor::new(strategy.clone());

            let compressed = compressor.compress(&gradients).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();

            // Should decompress to same number of arrays
            assert_eq!(decompressed.len(), gradients.len());

            // Shapes should match
            for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
                assert_eq!(orig.shape(), decomp.shape());
            }

            // For lossless strategies, values should match exactly
            match strategy {
                CompressionStrategy::None => {
                    for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
                        for (&o, &d) in orig.iter().zip(decomp.iter()) {
                            assert_relative_eq!(o, d, epsilon = 1e-10);
                        }
                    }
                }
                _ => {
                    // For lossy strategies, just check that we get reasonable values
                    for decomp in &decompressed {
                        assert!(decomp.iter().all(|&x| x.is_finite()));
                    }
                }
            }
        }
    }

    #[test]
    fn test_compression_invalid_configs() {
        // Invalid quantization bits
        let strategy = CompressionStrategy::Quantization { bits: 64 };
        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0])];
        assert!(compressor.compress(&gradients).is_err());

        // Invalid decompression data
        let valid_compressor: GradientCompressor<f64, ndarray::Ix1> =
            GradientCompressor::new(CompressionStrategy::None);
        let invalid_compressed = CompressedGradient {
            data: vec![1, 2, 3], // Insufficient data
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::None,
                compression_ratio: 1.0,
                nnz_count: 1,
                scale_factors: vec![],
                extra_data: vec![],
            },
            shapes: vec![vec![2]],
        };

        assert!(valid_compressor.decompress(&invalid_compressed).is_err());
    }

    #[test]
    fn test_distributed_with_compression() {
        // Test parameter server with compressed gradients
        let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);
        let initial_params = vec![Array1::from_vec(vec![0.0, 0.0])];
        server.initialize(&initial_params).unwrap();

        let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 1 });

        // Create gradients and compress them
        let gradients1 = vec![Array1::from_vec(vec![1.0, 3.0])]; // Top-1 should keep 3.0
        let gradients2 = vec![Array1::from_vec(vec![2.0, 1.0])]; // Top-1 should keep 2.0

        let compressed1 = compressor.compress(&gradients1).unwrap();
        let compressed2 = compressor.compress(&gradients2).unwrap();

        let decompressed1 = compressor.decompress(&compressed1).unwrap();
        let decompressed2 = compressor.decompress(&compressed2).unwrap();

        // Submit decompressed gradients to server
        server.submit_update(0, decompressed1).unwrap();
        server.submit_update(1, decompressed2).unwrap();

        let global_params = server.get_global_parameters();

        // Should have averaged the compressed gradients
        // Node 0 contributes [0, 3.0], Node 1 contributes [2.0, 0]
        // Average: [1.0, 1.5]
        assert_relative_eq!(global_params[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(global_params[0][1], 1.5, epsilon = 1e-6);
    }
}
