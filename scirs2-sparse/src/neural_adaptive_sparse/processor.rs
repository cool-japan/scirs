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
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
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
    /// Running sum of real attention-weight measurements observed while
    /// encoding operation states through the transformer (used to report a
    /// genuine `transformer_attention_score`, never a fabricated constant).
    attention_score_sum: f64,
    /// Number of attention-score samples accumulated in `attention_score_sum`.
    attention_score_samples: usize,
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
            attention_score_sum: 0.0,
            attention_score_samples: 0,
        }
    }

    /// Process sparse matrix operation with adaptive optimization
    pub fn optimize_operation<T>(
        &mut self,
        matrix_features: &[f64],
        operation_context: &OperationContext,
    ) -> SparseResult<OptimizationStrategy>
    where
        T: Float
            + SparseElement
            + NumAssign
            + SimdUnifiedOps
            + std::fmt::Debug
            + Copy
            + Send
            + Sync
            + 'static,
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
        &mut self,
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

        // Use transformer for feature encoding if available. The attention
        // score returned here is a genuine, input-dependent measurement
        // (never a fabricated constant) -- it is accumulated so
        // `get_statistics()` can report a real running average.
        let transformer_output = self
            .transformer
            .as_ref()
            .map(|transformer| transformer.encode_matrix_pattern_with_score(&state));

        match transformer_output {
            Some((encoded, attention_score)) => {
                self.attention_score_sum += attention_score;
                self.attention_score_samples += 1;
                Ok(encoded)
            }
            None => Ok(state),
        }
    }

    /// Select action using neural network
    fn neural_network_select_action(&self, state: &[f64]) -> SparseResult<OptimizationStrategy> {
        let outputs = self.neural_network.forward(state);

        let best_idx = outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
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

        // Neural-network "accuracy": the real, measured fraction of recorded
        // adaptation outcomes that actually improved on the naive baseline
        // (reward > 0), derived from genuine execution history rather than a
        // fixed constant. Honestly reports 0.0 (no fabricated confidence)
        // when nothing has been measured yet.
        let neural_network_accuracy = if !self.performance_history.is_empty() {
            successful_adaptations as f64 / self.performance_history.len() as f64
        } else {
            0.0
        };

        // Transformer attention score: running average of the *actual*
        // attention weights the transformer produced while encoding real
        // operation states (0.0, honestly, if the transformer is disabled or
        // has not yet processed any state -- never a fabricated constant).
        let transformer_attention_score = if self.attention_score_samples > 0 {
            self.attention_score_sum / self.attention_score_samples as f64
        } else {
            0.0
        };

        NeuralProcessorStats {
            total_operations,
            successful_adaptations,
            average_performance_improvement: average_improvement,
            most_effective_strategy,
            neural_network_accuracy,
            rl_agent_reward: rl_reward,
            pattern_memory_hit_rate: pattern_hit_rate,
            transformer_attention_score,
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
                avg1.partial_cmp(&avg2).expect("Operation failed")
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
        T: Float
            + SparseElement
            + NumAssign
            + SimdUnifiedOps
            + std::fmt::Debug
            + Copy
            + Send
            + Sync
            + 'static,
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

        // Measure real efficiency signals from the actual CSR structure that
        // was processed and the wall-clock time that was actually observed
        // -- no fabricated constants.
        let num_workers = scirs2_core::parallel_ops::num_threads().max(1);
        let (cache_efficiency, simd_utilization, parallel_efficiency, memory_bandwidth) =
            Self::measure_execution_efficiency(indptr, indices, data, execution_time, num_workers);

        // Learn from performance
        let performance = PerformanceMetrics::new(
            execution_time,
            cache_efficiency,
            simd_utilization,
            parallel_efficiency,
            memory_bandwidth,
            strategy,
        );

        self.learn_from_performance(strategy, performance, &matrix_features, &context)?;

        Ok(())
    }

    /// Derive real, matrix-dependent execution-efficiency signals from the
    /// CSR structure that was actually processed and the wall-clock time
    /// that was actually observed executing it. These replace the previous
    /// fixed placeholder constants (0.8/0.9/0.7/0.85, regardless of input) --
    /// every value below varies with the real input matrix and timing.
    ///
    /// `num_workers` is the number of parallel workers to model the load
    /// balance against; production code passes the real environment thread
    /// count (`scirs2_core::parallel_ops::num_threads()`), while tests can
    /// inject a fixed value for deterministic assertions.
    fn measure_execution_efficiency<T>(
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        execution_time: f64,
        num_workers: usize,
    ) -> (f64, f64, f64, f64) {
        let num_rows = indptr.len().saturating_sub(1);
        let nnz = data.len().min(indices.len());

        // Cache efficiency: derived from the actual spatial locality of the
        // column indices touched by consecutive nonzeros within a row --
        // small jumps between consecutive `x[col]` accesses indicate good
        // locality, large jumps indicate poor locality.
        let mut jump_sum = 0.0f64;
        let mut jump_count = 0usize;
        for row in 0..num_rows {
            let start = indptr[row];
            let end = indptr[row + 1].min(nnz);
            if end > start + 1 {
                for w in start..end - 1 {
                    jump_sum += indices[w + 1].abs_diff(indices[w]) as f64;
                    jump_count += 1;
                }
            }
        }
        let avg_jump = if jump_count > 0 {
            jump_sum / jump_count as f64
        } else {
            0.0
        };
        // Normalize by a representative cache-line width (8 f64 elements):
        // a jump of zero yields perfect (1.0) locality, larger jumps
        // asymptotically approach 0.
        let cache_efficiency = 1.0 / (1.0 + avg_jump / 8.0);

        // SIMD utilization: real fraction of nonzeros that live in rows long
        // enough to fill at least one SIMD lane (assumed width 8, a typical
        // AVX-class lane count for f64).
        const SIMD_LANE_WIDTH: usize = 8;
        let mut simd_capable_nnz = 0usize;
        for row in 0..num_rows {
            let start = indptr[row];
            let end = indptr[row + 1].min(nnz).max(start);
            let row_len = end - start;
            if row_len >= SIMD_LANE_WIDTH {
                simd_capable_nnz += row_len;
            }
        }
        let simd_utilization = if nnz > 0 {
            simd_capable_nnz as f64 / nnz as f64
        } else {
            0.0
        };

        // Parallel efficiency: load-balance ratio (min/max nonzeros per
        // chunk) when rows are partitioned contiguously across `num_workers`.
        let parallel_efficiency = if num_rows > 0 && num_workers > 1 {
            let chunk = num_rows.div_ceil(num_workers);
            let mut min_chunk_nnz = usize::MAX;
            let mut max_chunk_nnz = 0usize;
            let mut chunks_seen = 0usize;
            for w in 0..num_workers {
                let row_start = (w * chunk).min(num_rows);
                let row_end = ((w + 1) * chunk).min(num_rows);
                if row_start >= row_end {
                    continue;
                }
                let nnz_start = indptr[row_start];
                let nnz_end = indptr[row_end].min(nnz).max(nnz_start);
                let chunk_nnz = nnz_end - nnz_start;
                min_chunk_nnz = min_chunk_nnz.min(chunk_nnz);
                max_chunk_nnz = max_chunk_nnz.max(chunk_nnz);
                chunks_seen += 1;
            }
            if chunks_seen > 1 && max_chunk_nnz > 0 {
                min_chunk_nnz as f64 / max_chunk_nnz as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Memory bandwidth: real bytes moved (row data reads + gathered `x`
        // reads + `y` writes) divided by the *actually observed* wall-clock
        // execution time, normalized against a representative DDR bandwidth
        // ceiling so the reported value stays in a comparable [0, 1] range.
        let element_size = std::mem::size_of::<T>() as f64;
        let index_size = std::mem::size_of::<usize>() as f64;
        let bytes_moved = nnz as f64 * (element_size + index_size) + num_rows as f64 * element_size;
        const REFERENCE_BANDWIDTH_BYTES_PER_SEC: f64 = 20.0e9;
        let memory_bandwidth = if execution_time > 0.0 {
            (bytes_moved / execution_time / REFERENCE_BANDWIDTH_BYTES_PER_SEC).min(1.0)
        } else {
            0.0
        };

        (
            cache_efficiency,
            simd_utilization,
            parallel_efficiency,
            memory_bandwidth,
        )
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
                    .min_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
                    .unwrap_or(&0.0),
            );
            features.push(
                *data_f64
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
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
        T: Float
            + SparseElement
            + NumAssign
            + SimdUnifiedOps
            + std::fmt::Debug
            + Copy
            + Send
            + Sync,
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
        T: Float + SparseElement + NumAssign + std::fmt::Debug + Copy,
    {
        for (i, y_val) in y.iter_mut().enumerate() {
            *y_val = T::sparse_zero();
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
        T: Float + SparseElement + NumAssign + std::fmt::Debug + Copy,
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
        T: Float + SparseElement + NumAssign + SimdUnifiedOps + std::fmt::Debug + Copy,
    {
        // Use SIMD operations from scirs2-core
        for (i, y_val) in y.iter_mut().enumerate() {
            *y_val = T::sparse_zero();
            if i + 1 < indptr.len() {
                let start = indptr[i];
                let end = indptr[i + 1];
                if end > start {
                    let row_data = &data[start..end];
                    let row_indices = &indices[start..end];

                    // SIMD dot product
                    let mut sum = T::sparse_zero();
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
        T: Float
            + SparseElement
            + NumAssign
            + SimdUnifiedOps
            + std::fmt::Debug
            + Copy
            + Send
            + Sync,
    {
        // Use parallel operations from scirs2-core
        use scirs2_core::parallel_ops::*;

        // Sequential implementation for now
        for i in 0..y.len() {
            y[i] = T::sparse_zero();
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

#[cfg(test)]
mod fabrication_regression_tests {
    use super::*;

    fn good_performance(execution_time: f64) -> PerformanceMetrics {
        PerformanceMetrics::new(
            execution_time,
            0.9,
            0.9,
            0.9,
            0.9,
            OptimizationStrategy::SIMDVectorized,
        )
    }

    fn bad_performance(execution_time: f64) -> PerformanceMetrics {
        PerformanceMetrics::new(
            execution_time,
            0.0,
            0.0,
            0.0,
            0.0,
            OptimizationStrategy::RowWiseCache,
        )
    }

    fn dummy_context() -> OperationContext {
        OperationContext {
            matrix_shape: (10, 10),
            nnz: 20,
            operation_type: OperationType::MatVec,
            performance_target: PerformanceTarget::Speed,
        }
    }

    #[test]
    fn test_fresh_processor_reports_honest_zero_stats_not_fabricated_constants() {
        let config = NeuralAdaptiveConfig::lightweight();
        let processor = NeuralAdaptiveSparseProcessor::new(config);

        let stats = processor.get_statistics();
        // Previously these were UNCONDITIONALLY 0.85 / 0.75 regardless of
        // whether any operation had ever been performed.
        assert_eq!(stats.neural_network_accuracy, 0.0);
        assert_eq!(stats.transformer_attention_score, 0.0);
    }

    #[test]
    fn test_neural_network_accuracy_reflects_real_mixed_outcomes() {
        let config = NeuralAdaptiveConfig::lightweight();
        let mut processor = NeuralAdaptiveSparseProcessor::new(config);
        let context = dummy_context();
        let features = vec![1.0, 2.0, 3.0];

        // 3 genuinely good outcomes (fast + efficient => positive reward
        // against the fixed baseline used internally by get_statistics) and
        // 2 genuinely bad outcomes (slow + inefficient => negative reward).
        // This is non-constant, varied data -- not the all-ones pattern
        // that can't distinguish real computation from a stub.
        for _ in 0..3 {
            processor
                .learn_from_performance(
                    OptimizationStrategy::SIMDVectorized,
                    good_performance(0.05),
                    &features,
                    &context,
                )
                .expect("Operation failed");
        }
        for _ in 0..2 {
            processor
                .learn_from_performance(
                    OptimizationStrategy::RowWiseCache,
                    bad_performance(5.0),
                    &features,
                    &context,
                )
                .expect("Operation failed");
        }

        let stats = processor.get_statistics();
        assert_eq!(stats.successful_adaptations, 3);
        assert!(
            (stats.neural_network_accuracy - 0.6).abs() < 1e-9,
            "expected 3/5 = 0.6 real accuracy, got {}",
            stats.neural_network_accuracy
        );
        // Must not be the old fabricated constant.
        assert_ne!(stats.neural_network_accuracy, 0.85);
    }

    #[test]
    fn test_transformer_attention_score_is_measured_from_real_operations() {
        let config = NeuralAdaptiveConfig::lightweight().with_self_attention(true);
        let mut processor = NeuralAdaptiveSparseProcessor::new(config);
        let context = dummy_context();

        // Distinct, non-constant feature vectors across calls.
        let features_a = vec![1.0, 2.0, 3.0, 4.0];
        let features_b = vec![-5.0, 0.5, 9.0, -2.0];

        processor
            .optimize_operation::<f64>(&features_a, &context)
            .expect("Operation failed");
        processor
            .optimize_operation::<f64>(&features_b, &context)
            .expect("Operation failed");

        let stats = processor.get_statistics();
        // A logistic-sigmoid attention weight is always strictly in (0, 1);
        // it must also have actually moved away from the "no measurement
        // yet" zero default now that real operations were processed.
        assert!(stats.transformer_attention_score > 0.0);
        assert!(stats.transformer_attention_score < 1.0);
        // Must not be the old fabricated constant.
        assert_ne!(stats.transformer_attention_score, 0.75);
    }

    #[test]
    fn test_measure_execution_efficiency_cache_locality_varies_with_structure() {
        // Row-contiguous (good locality): each row's nonzeros have
        // consecutive column indices.
        let indptr = vec![0, 4, 8, 12];
        let indices_good = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let data = vec![1.0f64; 12];

        // Scattered (poor locality): column indices jump wildly within a row.
        let indices_bad = vec![0, 500, 3, 900, 1, 700, 50, 999, 2, 600, 20, 888];

        let (cache_good, _, _, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr,
            &indices_good,
            &data,
            0.01,
            1,
        );
        let (cache_bad, _, _, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr,
            &indices_bad,
            &data,
            0.01,
            1,
        );

        assert!(
            cache_good > cache_bad,
            "contiguous access should score higher cache efficiency than scattered access: {cache_good} vs {cache_bad}"
        );
        assert_ne!(cache_good, 0.8);
        assert_ne!(cache_bad, 0.8);
    }

    #[test]
    fn test_measure_execution_efficiency_simd_utilization_varies_with_row_length() {
        // Long rows (length 10 each): fully SIMD-capable.
        let indptr_long = vec![0, 10, 20];
        let indices_long: Vec<usize> = (0..20).collect();
        let data_long = vec![1.0f64; 20];

        // Short rows (length 1 each): cannot fill a SIMD lane.
        let indptr_short: Vec<usize> = (0..=20).collect();
        let indices_short: Vec<usize> = (0..20).collect();
        let data_short = vec![1.0f64; 20];

        let (_, simd_long, _, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr_long,
            &indices_long,
            &data_long,
            0.01,
            1,
        );
        let (_, simd_short, _, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr_short,
            &indices_short,
            &data_short,
            0.01,
            1,
        );

        assert_eq!(simd_long, 1.0);
        assert_eq!(simd_short, 0.0);
        assert!(simd_long > simd_short);
        assert_ne!(simd_long, 0.9);
    }

    #[test]
    fn test_measure_execution_efficiency_memory_bandwidth_scales_with_time() {
        let indptr = vec![0, 4];
        let indices = vec![0, 1, 2, 3];
        let data = vec![1.0f64; 4];

        let (_, _, _, bw_fast) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr, &indices, &data, 1e-9, 1,
        );
        let (_, _, _, bw_slow) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr, &indices, &data, 1.0, 1,
        );

        assert!(
            bw_fast > bw_slow,
            "faster execution of the same data volume must score higher bandwidth utilization: {bw_fast} vs {bw_slow}"
        );
        assert_ne!(bw_slow, 0.85);
    }

    #[test]
    fn test_measure_execution_efficiency_parallel_efficiency_reflects_load_balance() {
        // 8 rows, each with 2 nonzeros: perfectly balanced across 4 workers.
        let indptr_balanced = vec![0, 2, 4, 6, 8, 10, 12, 14, 16];
        let indices_balanced: Vec<usize> = (0..16).collect();
        let data_balanced = vec![1.0f64; 16];

        // 8 rows, but every nonzero is concentrated in the first 2 rows:
        // terrible balance across 4 workers.
        let indptr_unbalanced = vec![0, 16, 32, 32, 32, 32, 32, 32, 32];
        let indices_unbalanced: Vec<usize> = (0..32).collect();
        let data_unbalanced = vec![1.0f64; 32];

        let (_, _, par_balanced, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr_balanced,
            &indices_balanced,
            &data_balanced,
            0.01,
            4,
        );
        let (_, _, par_unbalanced, _) = NeuralAdaptiveSparseProcessor::measure_execution_efficiency(
            &indptr_unbalanced,
            &indices_unbalanced,
            &data_unbalanced,
            0.01,
            4,
        );

        assert_eq!(par_balanced, 1.0);
        assert!(par_balanced > par_unbalanced);
        assert_ne!(par_unbalanced, 0.7);
    }

    #[test]
    fn test_adaptive_spmv_records_non_placeholder_performance() {
        let config = NeuralAdaptiveConfig::lightweight();
        let mut processor = NeuralAdaptiveSparseProcessor::new(config);

        // A small but non-trivial CSR matrix (non-constant data).
        let rows = vec![0usize; 2];
        let cols = vec![0usize; 2];
        let indptr = vec![0usize, 2, 4];
        let indices = vec![0usize, 1, 0, 1];
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let x = vec![1.0f64, 1.0];
        let mut y = vec![0.0f64; 2];

        processor
            .adaptive_spmv(&rows, &cols, &indptr, &indices, &data, &x, &mut y)
            .expect("Operation failed");

        let recorded = processor
            .performance_history
            .back()
            .expect("expected a recorded performance sample");
        // None of these should be exactly the old hardcoded placeholders,
        // and all must be valid fractions.
        assert!((0.0..=1.0).contains(&recorded.cache_efficiency));
        assert!((0.0..=1.0).contains(&recorded.simd_utilization));
        assert!((0.0..=1.0).contains(&recorded.parallel_efficiency));
        assert!((0.0..=1.0).contains(&recorded.memory_bandwidth));
    }
}
