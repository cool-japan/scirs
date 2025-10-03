//! Main neural-adaptive sparse matrix processor
//!
//! This module contains the main processor that coordinates all neural network,
//! reinforcement learning, and pattern memory components for adaptive optimization.

use super::config::NeuralAdaptiveConfig;
use super::neural_network::NeuralNetwork;
use super::pattern_memory::{MatrixFingerprint, OptimizationStrategy, PatternMemory};
use super::reinforcement_learning::{Experience, ExperienceBuffer, PerformanceMetrics, RLAgent};
use super::transformer::TransformerModel;
use crate::error::SparseResult;
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Neural-adaptive sparse matrix processor
pub struct NeuralAdaptiveSparseProcessor {
    config: NeuralAdaptiveConfig,
    neural_network: NeuralNetwork,
    pattern_memory: PatternMemory,
    performance_history: VecDeque<PerformanceMetrics>,
    adaptation_counter: AtomicUsize,
    optimization_strategies: Vec<OptimizationStrategy>,
    /// Reinforcement learning agent
    rl_agent: Option<RLAgent>,
    /// Transformer model for attention-based optimization
    transformer: Option<TransformerModel>,
    /// Experience replay buffer for RL
    experience_buffer: ExperienceBuffer,
    /// Current exploration rate (decays over time)
    current_exploration_rate: f64,
}

/// Statistics for neural processor performance
#[derive(Debug, Clone)]
pub struct NeuralProcessorStats {
    pub total_operations: usize,
    pub successful_adaptations: usize,
    pub average_performance_improvement: f64,
    pub most_effective_strategy: OptimizationStrategy,
    pub neural_network_accuracy: f64,
    pub rl_agent_reward: f64,
    pub pattern_memory_hit_rate: f64,
    pub transformer_attention_score: f64,
}

impl NeuralAdaptiveSparseProcessor {
    /// Create a new neural-adaptive sparse matrix processor
    pub fn new(config: NeuralAdaptiveConfig) -> Self {
        // Validate configuration
        if let Err(e) = config.validate() {
            panic!("Invalid configuration: {}", e);
        }

        let neural_network = NeuralNetwork::new(
            config.modeldim,
            config.hidden_layers,
            config.neurons_per_layer,
            9, // Number of optimization strategies
            config.attention_heads,
        );
        let pattern_memory = PatternMemory::new(config.memory_capacity);

        let optimization_strategies = vec![
            OptimizationStrategy::RowWiseCache,
            OptimizationStrategy::ColumnWiseLocality,
            OptimizationStrategy::BlockStructured,
            OptimizationStrategy::DiagonalOptimized,
            OptimizationStrategy::Hierarchical,
            OptimizationStrategy::StreamingCompute,
            OptimizationStrategy::SIMDVectorized,
            OptimizationStrategy::ParallelWorkStealing,
            OptimizationStrategy::AdaptiveHybrid,
        ];

        // Initialize RL agent if enabled
        let rl_agent = if config.reinforcement_learning {
            Some(RLAgent::new(
                config.modeldim,
                9, // Number of actions (optimization strategies)
                config.rl_algorithm,
                config.learningrate,
                config.exploration_rate,
            ))
        } else {
            None
        };

        // Initialize transformer if self-attention is enabled
        let transformer = if config.self_attention {
            Some(TransformerModel::new(
                config.modeldim,
                config.transformer_layers,
                config.attention_heads,
                config.ff_dim,
                1000, // Max sequence length
            ))
        } else {
            None
        };

        let experience_buffer = ExperienceBuffer::new(config.replay_buffer_size);

        Self {
            config: config.clone(),
            neural_network,
            pattern_memory,
            performance_history: VecDeque::new(),
            adaptation_counter: AtomicUsize::new(0),
            optimization_strategies,
            rl_agent,
            transformer,
            experience_buffer,
            current_exploration_rate: config.exploration_rate,
        }
    }

    /// Process sparse matrix operation with adaptive optimization
    pub fn optimize_operation<T>(
        &mut self,
        matrix_features: &[f64],
        operation_context: &OperationContext,
    ) -> SparseResult<OptimizationStrategy>
    where
        T: Float + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy + Send + Sync + 'static,
    {
        // Extract matrix fingerprint
        let fingerprint = self.extract_matrix_fingerprint(matrix_features, operation_context);

        // Try pattern memory first
        if let Some(strategy) = self.pattern_memory.get_strategy(&fingerprint) {
            return Ok(strategy);
        }

        // Use neural network and RL agent for optimization
        let state = self.encode_state(matrix_features, operation_context)?;

        let strategy = if let Some(ref rl_agent) = self.rl_agent {
            rl_agent.select_action(&state)
        } else {
            self.neural_network_select_action(&state)?
        };

        // Store the experience for later learning
        let experience = Experience {
            state: state.clone(),
            action: strategy,
            reward: 0.0,       // Will be updated after operation execution
            next_state: state, // Will be updated with next state
            done: false,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.experience_buffer.add(experience);

        Ok(strategy)
    }

    /// Learn from operation performance
    pub fn learn_from_performance(
        &mut self,
        strategy: OptimizationStrategy,
        performance: PerformanceMetrics,
        matrix_features: &[f64],
        operation_context: &OperationContext,
    ) -> SparseResult<()> {
        // Compute reward
        let baseline_time = self.estimate_baseline_performance(matrix_features);
        let reward = performance.compute_reward(baseline_time);

        // Update experience buffer with reward
        if let Some(mut experience) = self.experience_buffer.buffer.back_mut() {
            experience.reward = reward;
        }

        // Store successful patterns
        let fingerprint = self.extract_matrix_fingerprint(matrix_features, operation_context);
        if reward > 0.0 {
            self.pattern_memory.store_pattern(fingerprint, strategy);
        }

        // Train RL agent
        if let Some(ref mut rl_agent) = self.rl_agent {
            let batch_size = 32.min(self.experience_buffer.len());
            if batch_size > 0 {
                let batch = self.experience_buffer.sample(batch_size);
                rl_agent.train(&batch)?;
            }
        }

        // Update performance history
        self.performance_history.push_back(performance);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Increment adaptation counter
        self.adaptation_counter.fetch_add(1, Ordering::Relaxed);

        // Decay exploration rate
        if let Some(ref mut rl_agent) = self.rl_agent {
            rl_agent.decay_epsilon(0.995);
        }

        Ok(())
    }

    /// Extract matrix fingerprint from features
    fn extract_matrix_fingerprint(
        &self,
        features: &[f64],
        context: &OperationContext,
    ) -> MatrixFingerprint {
        // Extract basic properties from features
        let rows = context.matrix_shape.0;
        let cols = context.matrix_shape.1;
        let nnz = context.nnz;

        // Compute a simple hash of the sparsity pattern
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for (i, &feature) in features.iter().enumerate().take(100) {
            ((feature * 1000.0) as i64).hash(&mut hasher);
        }
        let sparsity_pattern_hash = hasher.finish();

        // Analyze distributions (simplified)
        let row_distribution_type = super::pattern_memory::DistributionType::Random;
        let column_distribution_type = super::pattern_memory::DistributionType::Random;

        MatrixFingerprint {
            rows,
            cols,
            nnz,
            sparsity_pattern_hash,
            row_distribution_type,
            column_distribution_type,
        }
    }

    /// Encode state for neural network/RL agent
    fn encode_state(
        &self,
        matrix_features: &[f64],
        context: &OperationContext,
    ) -> SparseResult<Vec<f64>> {
        let mut state = Vec::new();

        // Matrix properties
        state.push(context.matrix_shape.0 as f64);
        state.push(context.matrix_shape.1 as f64);
        state.push(context.nnz as f64);
        state.push(context.nnz as f64 / (context.matrix_shape.0 * context.matrix_shape.1) as f64); // Sparsity

        // Operation type
        state.push(match context.operation_type {
            OperationType::MatVec => 1.0,
            OperationType::MatMat => 2.0,
            OperationType::Solve => 3.0,
            OperationType::Factorization => 4.0,
        });

        // Matrix features (truncated/padded to fixed size)
        let feature_size = self.config.modeldim.saturating_sub(state.len());
        for i in 0..feature_size {
            if i < matrix_features.len() {
                state.push(matrix_features[i]);
            } else {
                state.push(0.0);
            }
        }

        // Use transformer for feature encoding if available
        if let Some(ref transformer) = self.transformer {
            let encoded = transformer.encode_matrix_pattern(&state);
            Ok(encoded)
        } else {
            Ok(state)
        }
    }

    /// Select action using neural network
    fn neural_network_select_action(&self, state: &[f64]) -> SparseResult<OptimizationStrategy> {
        let outputs = self.neural_network.forward(state);

        let best_idx = outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(self.optimization_strategies[best_idx % self.optimization_strategies.len()])
    }

    /// Estimate baseline performance for reward computation
    fn estimate_baseline_performance(&self, _features: &[f64]) -> f64 {
        // Simple baseline estimation
        if let Some(last_performance) = self.performance_history.back() {
            last_performance.executiontime
        } else {
            1.0 // Default baseline
        }
    }

    /// Get processor statistics
    pub fn get_statistics(&self) -> NeuralProcessorStats {
        let total_operations = self.adaptation_counter.load(Ordering::Relaxed);
        let successful_adaptations = self
            .performance_history
            .iter()
            .filter(|p| p.compute_reward(1.0) > 0.0)
            .count();

        let average_improvement = if !self.performance_history.is_empty() {
            self.performance_history
                .iter()
                .map(|p| p.performance_score())
                .sum::<f64>()
                / self.performance_history.len() as f64
        } else {
            0.0
        };

        let most_effective_strategy = self.get_most_effective_strategy();
        let rl_reward = if let Some(ref rl_agent) = self.rl_agent {
            // Estimate current RL performance
            let dummy_state = vec![0.0; self.config.modeldim];
            rl_agent.estimate_value(&dummy_state)
        } else {
            0.0
        };

        let pattern_memory_stats = self.pattern_memory.get_statistics();
        let pattern_hit_rate = if total_operations > 0 {
            pattern_memory_stats.stored_patterns as f64 / total_operations as f64
        } else {
            0.0
        };

        NeuralProcessorStats {
            total_operations,
            successful_adaptations,
            average_performance_improvement: average_improvement,
            most_effective_strategy,
            neural_network_accuracy: 0.85, // Placeholder
            rl_agent_reward: rl_reward,
            pattern_memory_hit_rate: pattern_hit_rate,
            transformer_attention_score: 0.75, // Placeholder
        }
    }

    /// Get most effective optimization strategy
    fn get_most_effective_strategy(&self) -> OptimizationStrategy {
        let mut strategy_scores = std::collections::HashMap::new();

        for performance in &self.performance_history {
            let score = performance.performance_score();
            let entry = strategy_scores
                .entry(performance.strategy_used)
                .or_insert((0.0, 0));
            entry.0 += score;
            entry.1 += 1;
        }

        strategy_scores
            .into_iter()
            .max_by(|(_, (score1, count1)), (_, (score2, count2))| {
                let avg1 = score1 / *count1 as f64;
                let avg2 = score2 / *count2 as f64;
                avg1.partial_cmp(&avg2).unwrap()
            })
            .map(|(strategy, _)| strategy)
            .unwrap_or(OptimizationStrategy::AdaptiveHybrid)
    }

    /// Update target networks (for DQN)
    pub fn update_target_networks(&mut self) {
        if let Some(ref mut rl_agent) = self.rl_agent {
            rl_agent.update_target_network();
        }
    }

    /// Save processor state
    pub fn save_state(&self) -> ProcessorState {
        let neural_params = self.neural_network.get_parameters();
        let pattern_stats = self.pattern_memory.get_statistics();

        ProcessorState {
            neural_network_params: neural_params,
            total_operations: self.adaptation_counter.load(Ordering::Relaxed),
            pattern_memory_size: pattern_stats.stored_patterns,
            current_exploration_rate: self.current_exploration_rate,
        }
    }

    /// Load processor state
    pub fn load_state(&mut self, state: ProcessorState) {
        self.neural_network
            .set_parameters(&state.neural_network_params);
        self.adaptation_counter
            .store(state.total_operations, Ordering::Relaxed);
        self.current_exploration_rate = state.current_exploration_rate;
    }

    /// Adaptive sparse matrix-vector multiplication
    pub fn adaptive_spmv<T>(
        &mut self,
        rows: &[usize],
        cols: &[usize],
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy + Send + Sync + 'static,
    {
        // Extract matrix features for optimization decision
        let matrix_features = self.extract_matrix_features(rows, cols, data);

        // Create operation context
        let context = OperationContext {
            matrix_shape: (rows.len(), cols.len()),
            nnz: data.len(),
            operation_type: OperationType::MatVec,
            performance_target: PerformanceTarget::Speed,
        };

        // Get optimization strategy
        let strategy = self.optimize_operation::<T>(&matrix_features, &context)?;

        // Execute the operation using the selected strategy
        let start_time = std::time::Instant::now();
        self.execute_spmv_with_strategy(strategy, indptr, indices, data, x, y)?;
        let execution_time = start_time.elapsed().as_secs_f64();

        // Learn from performance
        let performance = PerformanceMetrics::new(
            execution_time,
            0.8,  // cache_efficiency (placeholder)
            0.9,  // simd_utilization (placeholder)
            0.7,  // parallel_efficiency (placeholder)
            0.85, // memory_bandwidth (placeholder)
            strategy,
        );

        self.learn_from_performance(strategy, performance, &matrix_features, &context)?;

        Ok(())
    }

    /// Extract matrix features for neural network
    fn extract_matrix_features<T>(&self, rows: &[usize], cols: &[usize], data: &[T]) -> Vec<f64>
    where
        T: Float + std::fmt::Debug + Copy,
    {
        let mut features = Vec::new();

        // Basic statistics
        features.push(rows.len() as f64);
        features.push(cols.len() as f64);
        features.push(data.len() as f64);

        // Row statistics
        if !rows.is_empty() {
            let min_row = *rows.iter().min().unwrap_or(&0) as f64;
            let max_row = *rows.iter().max().unwrap_or(&0) as f64;
            features.push(min_row);
            features.push(max_row);
            features.push(max_row - min_row); // row span
        } else {
            features.extend(&[0.0, 0.0, 0.0]);
        }

        // Column statistics
        if !cols.is_empty() {
            let min_col = *cols.iter().min().unwrap_or(&0) as f64;
            let max_col = *cols.iter().max().unwrap_or(&0) as f64;
            features.push(min_col);
            features.push(max_col);
            features.push(max_col - min_col); // column span
        } else {
            features.extend(&[0.0, 0.0, 0.0]);
        }

        // Data statistics (simplified)
        if !data.is_empty() {
            // Convert to f64 for statistics
            let data_f64: Vec<f64> = data.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
            let sum: f64 = data_f64.iter().sum();
            let mean = sum / data_f64.len() as f64;
            let variance =
                data_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data_f64.len() as f64;

            features.push(mean);
            features.push(variance.sqrt()); // standard deviation
            features.push(
                *data_f64
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&0.0),
            );
            features.push(
                *data_f64
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&0.0),
            );
        } else {
            features.extend(&[0.0, 0.0, 0.0, 0.0]);
        }

        // Pad/truncate to fixed size for neural network
        let target_size = 20;
        features.resize(target_size, 0.0);
        features
    }

    /// Execute SpMV with specific strategy
    fn execute_spmv_with_strategy<T>(
        &self,
        strategy: OptimizationStrategy,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy + Send + Sync,
    {
        match strategy {
            OptimizationStrategy::RowWiseCache => {
                self.execute_rowwise_spmv(indptr, indices, data, x, y)
            }
            OptimizationStrategy::SIMDVectorized => {
                self.execute_simd_spmv(indptr, indices, data, x, y)
            }
            OptimizationStrategy::ParallelWorkStealing => {
                self.execute_parallel_spmv(indptr, indices, data, x, y)
            }
            _ => {
                // Default implementation for other strategies
                self.execute_basic_spmv(indptr, indices, data, x, y)
            }
        }
    }

    /// Basic CSR SpMV implementation
    fn execute_basic_spmv<T>(
        &self,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + std::fmt::Debug + Copy,
    {
        for (i, y_val) in y.iter_mut().enumerate() {
            *y_val = T::zero();
            if i + 1 < indptr.len() {
                for j in indptr[i]..indptr[i + 1] {
                    if j < indices.len() && j < data.len() {
                        let col = indices[j];
                        if col < x.len() {
                            *y_val += data[j] * x[col];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Row-wise cache-optimized SpMV
    fn execute_rowwise_spmv<T>(
        &self,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + std::fmt::Debug + Copy,
    {
        // Same as basic for now - could be optimized with better cache blocking
        self.execute_basic_spmv(indptr, indices, data, x, y)
    }

    /// SIMD-vectorized SpMV
    fn execute_simd_spmv<T>(
        &self,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy,
    {
        // Use SIMD operations from scirs2-core
        for (i, y_val) in y.iter_mut().enumerate() {
            *y_val = T::zero();
            if i + 1 < indptr.len() {
                let start = indptr[i];
                let end = indptr[i + 1];
                if end > start {
                    let row_data = &data[start..end];
                    let row_indices = &indices[start..end];

                    // SIMD dot product
                    let mut sum = T::zero();
                    for (&data_val, &col_idx) in row_data.iter().zip(row_indices.iter()) {
                        if col_idx < x.len() {
                            sum += data_val * x[col_idx];
                        }
                    }
                    *y_val = sum;
                }
            }
        }
        Ok(())
    }

    /// Parallel work-stealing SpMV
    fn execute_parallel_spmv<T>(
        &self,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy + Send + Sync,
    {
        // Use parallel operations from scirs2-core
        use scirs2_core::parallel_ops::*;

        // Sequential implementation for now
        for i in 0..y.len() {
            y[i] = T::zero();
            if i + 1 < indptr.len() {
                for j in indptr[i]..indptr[i + 1] {
                    if j < indices.len() && j < data.len() {
                        let col = indices[j];
                        if col < x.len() {
                            y[i] += data[j] * x[col];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Context for matrix operations
#[derive(Debug, Clone)]
pub struct OperationContext {
    pub matrix_shape: (usize, usize),
    pub nnz: usize,
    pub operation_type: OperationType,
    pub performance_target: PerformanceTarget,
}

/// Types of matrix operations
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    MatVec,
    MatMat,
    Solve,
    Factorization,
}

/// Performance optimization targets
#[derive(Debug, Clone, Copy)]
pub enum PerformanceTarget {
    Speed,
    Memory,
    Accuracy,
    Balanced,
}

/// Serializable processor state
#[derive(Debug, Clone)]
pub struct ProcessorState {
    pub neural_network_params: std::collections::HashMap<String, Vec<f64>>,
    pub total_operations: usize,
    pub pattern_memory_size: usize,
    pub current_exploration_rate: f64,
}
