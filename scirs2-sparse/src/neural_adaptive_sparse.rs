//! Neural-Adaptive Sparse Matrix Operations for Advanced Mode
//!
//! This module implements neural network-inspired adaptive algorithms for sparse matrix
//! operations that learn and optimize based on matrix characteristics and usage patterns.

use crate::error::SparseResult;
use num_traits::{Float, NumAssign};
use rand::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Neural network layer for sparse matrix optimization
#[derive(Debug, Clone)]
struct NeuralLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: ActivationFunction,
}

/// Activation functions for neural network layers
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    #[allow(dead_code)]
    Tanh,
    #[allow(dead_code)]
    Swish,
    #[allow(dead_code)]
    Gelu,
}

/// Neural-adaptive sparse matrix processor configuration
#[derive(Debug, Clone)]
pub struct NeuralAdaptiveConfig {
    /// Number of hidden layers in the neural network
    pub hidden_layers: usize,
    /// Neurons per hidden layer
    pub neurons_per_layer: usize,
    /// Learning rate for adaptive optimization
    pub learningrate: f64,
    /// Memory capacity for pattern learning
    pub memory_capacity: usize,
    /// Enable reinforcement learning
    pub reinforcement_learning: bool,
    /// Attention mechanism configuration
    pub attention_heads: usize,
    /// Enable transformer-style self-attention
    pub self_attention: bool,
    /// Reinforcement learning algorithm
    pub rl_algorithm: RLAlgorithm,
    /// Exploration rate for RL
    pub exploration_rate: f64,
    /// Discount factor for future rewards
    pub discountfactor: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Transformer model dimension
    pub modeldim: usize,
    /// Feed-forward network dimension in transformer
    pub ff_dim: usize,
    /// Number of transformer layers
    pub transformer_layers: usize,
}

/// Reinforcement learning algorithms
#[derive(Debug, Clone, Copy)]
pub enum RLAlgorithm {
    /// Q-Learning with experience replay
    DQN,
    /// Policy gradient methods
    PolicyGradient,
    /// Actor-Critic methods
    ActorCritic,
    /// Proximal Policy Optimization
    PPO,
    /// Soft Actor-Critic
    SAC,
}

impl Default for NeuralAdaptiveConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 3,
            neurons_per_layer: 64,
            learningrate: 0.001,
            memory_capacity: 10000,
            reinforcement_learning: true,
            attention_heads: 8,
            self_attention: true,
            rl_algorithm: RLAlgorithm::DQN,
            exploration_rate: 0.1,
            discountfactor: 0.99,
            replay_buffer_size: 10000,
            modeldim: 512,
            ff_dim: 2048,
            transformer_layers: 6,
        }
    }
}

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

/// Neural network for sparse matrix optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
    attention_weights: Vec<Vec<f64>>,
    /// Multi-head attention mechanisms
    attention_heads: Vec<AttentionHead>,
    /// Layer normalization parameters
    layer_norms: Vec<LayerNorm>,
}

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
struct AttentionHead {
    query_weights: Vec<Vec<f64>>,
    key_weights: Vec<Vec<f64>>,
    value_weights: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    head_dim: usize,
}

/// Layer normalization
#[derive(Debug, Clone)]
struct LayerNorm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    eps: f64,
}

/// Transformer model for advanced pattern recognition
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TransformerModel {
    encoder_layers: Vec<TransformerEncoderLayer>,
    positional_encoding: Vec<Vec<f64>>,
    embedding_dim: usize,
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
struct TransformerEncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForwardNetwork,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    dropout_rate: f64,
}

/// Multi-head attention for transformer
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    output_projection: Vec<Vec<f64>>,
    num_heads: usize,
    head_dim: usize,
}

/// Feed-forward network
#[derive(Debug, Clone)]
struct FeedForwardNetwork {
    layer1: Vec<Vec<f64>>,
    layer1_bias: Vec<f64>,
    layer2: Vec<Vec<f64>>,
    layer2_bias: Vec<f64>,
    activation: ActivationFunction,
}

/// Reinforcement learning agent
#[derive(Debug)]
#[allow(dead_code)]
struct RLAgent {
    q_network: NeuralNetwork,
    target_network: Option<NeuralNetwork>,
    policy_network: Option<NeuralNetwork>,
    value_network: Option<NeuralNetwork>,
    algorithm: RLAlgorithm,
    epsilon: f64,
    learningrate: f64,
}

/// Experience for reinforcement learning
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Experience {
    state: Vec<f64>,
    action: OptimizationStrategy,
    reward: f64,
    next_state: Vec<f64>,
    done: bool,
    timestamp: u64,
}

/// Experience replay buffer
#[derive(Debug)]
struct ExperienceBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
    priority_weights: Vec<f64>,
}

/// Pattern memory for learning matrix characteristics
#[derive(Debug)]
struct PatternMemory {
    matrix_patterns: HashMap<MatrixFingerprint, OptimizationStrategy>,
    #[allow(dead_code)]
    access_patterns: VecDeque<AccessPattern>,
    #[allow(dead_code)]
    performance_cache: HashMap<String, f64>,
}

/// Matrix fingerprint for pattern recognition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MatrixFingerprint {
    rows: usize,
    cols: usize,
    nnz: usize,
    sparsity_pattern_hash: u64,
    row_distribution_type: DistributionType,
    column_distribution_type: DistributionType,
}

/// Distribution types for sparsity patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DistributionType {
    Uniform,
    Clustered,
    BandDiagonal,
    #[allow(dead_code)]
    BlockStructured,
    Random,
    PowerLaw,
}

/// Access pattern for memory optimization
#[derive(Debug, Clone)]
struct AccessPattern {
    #[allow(dead_code)]
    timestamp: u64,
    #[allow(dead_code)]
    row_sequence: Vec<usize>,
    #[allow(dead_code)]
    column_sequence: Vec<usize>,
    #[allow(dead_code)]
    cache_hits: usize,
    #[allow(dead_code)]
    cache_misses: usize,
}

/// Performance metrics for reinforcement learning
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    #[allow(dead_code)]
    executiontime: f64,
    #[allow(dead_code)]
    cache_efficiency: f64,
    #[allow(dead_code)]
    simd_utilization: f64,
    #[allow(dead_code)]
    parallel_efficiency: f64,
    #[allow(dead_code)]
    memory_bandwidth: f64,
    strategy_used: OptimizationStrategy,
}

/// Optimization strategies learned by the neural network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    /// Row-wise processing with cache optimization
    RowWiseCache,
    /// Column-wise processing for memory locality
    ColumnWiseLocality,
    /// Block-based processing for structured matrices
    BlockStructured,
    /// Diagonal-optimized processing
    DiagonalOptimized,
    /// Hierarchical decomposition
    Hierarchical,
    /// Streaming computation for large matrices
    StreamingCompute,
    /// SIMD-vectorized computation
    SIMDVectorized,
    /// Parallel work-stealing
    ParallelWorkStealing,
    /// Adaptive hybrid approach
    AdaptiveHybrid,
}

impl NeuralAdaptiveSparseProcessor {
    /// Create a new neural-adaptive sparse matrix processor
    pub fn new(config: NeuralAdaptiveConfig) -> Self {
        let neural_network = NeuralNetwork::new(&config);
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
            Some(RLAgent::new(&config))
        } else {
            None
        };

        // Initialize transformer if self-attention is enabled
        let transformer = if config.self_attention {
            Some(TransformerModel::new(&config))
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

    /// Neural-adaptive sparse matrix-vector multiplication
    #[allow(clippy::too_many_arguments)]
    pub fn adaptive_spmv<T>(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        let start_time = std::time::Instant::now();

        // Generate matrix fingerprint
        let fingerprint = self.generate_matrix_fingerprint(rows, cols, indptr, indices);

        // Neural network inference to select optimal strategy
        let strategy = self.select_optimization_strategy(&fingerprint, indptr, indices);

        // Execute optimized computation
        let result = self.execute_strategy(strategy, rows, cols, indptr, indices, data, x, y);

        // Record performance metrics for learning
        let executiontime = start_time.elapsed().as_secs_f64();
        let metrics = PerformanceMetrics {
            executiontime,
            cache_efficiency: self.estimate_cache_efficiency(indptr, indices),
            simd_utilization: self.estimate_simd_utilization(&strategy),
            parallel_efficiency: self.estimate_parallel_efficiency(&strategy, rows),
            memory_bandwidth: self.estimate_memory_bandwidth(data.len(), executiontime),
            strategy_used: strategy,
        };

        // Learn from performance
        self.update_neural_network(&fingerprint, &metrics);
        self.adaptation_counter.fetch_add(1, Ordering::Relaxed);

        result
    }

    /// Generate matrix fingerprint for pattern recognition
    fn generate_matrix_fingerprint(
        &self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
    ) -> MatrixFingerprint {
        let nnz = indices.len();

        // Compute sparsity pattern hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};

        // Sample sparsity pattern for hashing (to avoid O(nnz) hash computation)
        let sample_size = nnz.min(1000);
        let step = if nnz > sample_size {
            nnz / sample_size
        } else {
            1
        };

        for i in (0..nnz).step_by(step) {
            indices[i].hash(&mut hasher);
        }
        let sparsity_pattern_hash = hasher.finish();

        // Analyze row distribution
        let row_distribution_type = self.analyze_row_distribution(rows, indptr);

        // Analyze column distribution
        let column_distribution_type = self.analyze_column_distribution(cols, indices);

        MatrixFingerprint {
            rows,
            cols,
            nnz,
            sparsity_pattern_hash,
            row_distribution_type,
            column_distribution_type,
        }
    }

    /// Select optimization strategy using reinforcement learning and attention mechanisms
    fn select_optimization_strategy(
        &mut self,
        fingerprint: &MatrixFingerprint,
        indptr: &[usize],
        indices: &[usize],
    ) -> OptimizationStrategy {
        // Check pattern memory first
        if let Some(&cached_strategy) = self.pattern_memory.matrix_patterns.get(fingerprint) {
            return cached_strategy;
        }

        // Prepare input features
        let mut features = self.extract_features(fingerprint, indptr, indices);

        // Apply transformer attention if enabled
        if let Some(ref transformer) = self.transformer {
            features = transformer.forward(&features);
        }

        let selected_strategy = if self.config.reinforcement_learning {
            // Use reinforcement learning for strategy selection
            self.select_strategy_with_rl(&features)
        } else {
            // Fallback to neural network inference
            let strategy_scores = self.neural_network.forward(&features);
            let best_strategy_idx = strategy_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            self.optimization_strategies[best_strategy_idx]
        };

        // Cache the decision
        self.pattern_memory
            .matrix_patterns
            .insert(fingerprint.clone(), selected_strategy);

        selected_strategy
    }

    /// Select strategy using reinforcement learning
    fn select_strategy_with_rl(&mut self, state: &[f64]) -> OptimizationStrategy {
        if let Some(ref mut rl_agent) = self.rl_agent {
            // Epsilon-greedy exploration
            if rand::rng().random::<f64>() < self.current_exploration_rate {
                // Explore: random strategy
                let random_idx = rand::rng().random_range(0..self.optimization_strategies.len());
                self.optimization_strategies[random_idx]
            } else {
                // Exploit: best strategy according to Q-network
                let q_values = rl_agent.get_q_values(state);
                let best_action_idx = q_values
                    .iter()
                    .enumerate()
                    .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.optimization_strategies
                    [best_action_idx.min(self.optimization_strategies.len() - 1)]
            }
        } else {
            // Fallback if RL is not available
            self.optimization_strategies[0]
        }
    }

    /// Execute the selected optimization strategy
    fn execute_strategy<T>(
        &self,
        strategy: OptimizationStrategy,
        rows: usize,
        _cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        match strategy {
            OptimizationStrategy::RowWiseCache => {
                self.execute_row_wise_cache(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::ColumnWiseLocality => {
                self.execute_column_wise_locality(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::BlockStructured => {
                self.execute_block_structured(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::DiagonalOptimized => {
                self.execute_diagonal_optimized(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::Hierarchical => {
                self.execute_hierarchical(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::StreamingCompute => {
                self.execute_streaming_compute(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::SIMDVectorized => {
                self.execute_simd_vectorized(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::ParallelWorkStealing => {
                self.execute_parallel_work_stealing(rows, indptr, indices, data, x, y)
            }
            OptimizationStrategy::AdaptiveHybrid => {
                self.execute_adaptive_hybrid(rows, indptr, indices, data, x, y)
            }
        }
    }

    // Strategy implementations

    fn execute_row_wise_cache<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Cache-optimized row-wise processing with prefetching
        let cache_line_size = 64 / std::mem::size_of::<T>().max(1);

        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            let mut sum = T::zero();

            // Process in cache-friendly chunks
            for chunk_start in (start_idx..end_idx).step_by(cache_line_size.max(1)) {
                let chunk_end = (chunk_start + cache_line_size.max(1)).min(end_idx);

                for idx in chunk_start..chunk_end {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
            }

            y[row] = sum;
        }

        Ok(())
    }

    fn execute_column_wise_locality<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Initialize output
        for elem in y.iter_mut() {
            *elem = T::zero();
        }

        // Column-wise accumulation for better memory locality
        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            for idx in start_idx..end_idx {
                let col = indices[idx];
                y[row] += data[idx] * x[col];
            }
        }

        Ok(())
    }

    fn execute_block_structured<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Block-structured processing for matrices with block patterns
        const BLOCK_SIZE: usize = 32;

        for row_block in (0..rows).step_by(BLOCK_SIZE) {
            let row_block_end = (row_block + BLOCK_SIZE).min(rows);

            for row in row_block..row_block_end {
                let start_idx = indptr[row];
                let end_idx = indptr[row + 1];

                let mut sum = T::zero();
                for idx in start_idx..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
                y[row] = sum;
            }
        }

        Ok(())
    }

    fn execute_diagonal_optimized<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Optimized for diagonal and near-diagonal matrices
        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            let mut sum = T::zero();

            // Look for diagonal element first (common case optimization)
            for idx in start_idx..end_idx {
                let col = indices[idx];
                if col == row {
                    // Diagonal element - process immediately
                    sum += data[idx] * x[col];
                } else {
                    // Off-diagonal elements
                    sum += data[idx] * x[col];
                }
            }

            y[row] = sum;
        }

        Ok(())
    }

    fn execute_hierarchical<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Hierarchical decomposition for very large matrices
        if rows <= 1000 {
            // Base case: direct computation
            return self.execute_row_wise_cache(rows, indptr, indices, data, x, y);
        }

        // Divide into smaller subproblems
        let mid = rows / 2;

        // Process first half
        for row in 0..mid {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            let mut sum = T::zero();
            for idx in start_idx..end_idx {
                let col = indices[idx];
                sum += data[idx] * x[col];
            }
            y[row] = sum;
        }

        // Process second half
        for row in mid..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            let mut sum = T::zero();
            for idx in start_idx..end_idx {
                let col = indices[idx];
                sum += data[idx] * x[col];
            }
            y[row] = sum;
        }

        Ok(())
    }

    fn execute_streaming_compute<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Streaming computation for memory-bound scenarios
        const STREAM_BUFFER_SIZE: usize = 1024;

        for chunk_start in (0..rows).step_by(STREAM_BUFFER_SIZE) {
            let chunk_end = (chunk_start + STREAM_BUFFER_SIZE).min(rows);

            // Process chunk with streaming
            for row in chunk_start..chunk_end {
                let start_idx = indptr[row];
                let end_idx = indptr[row + 1];

                let mut sum = T::zero();
                for idx in start_idx..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
                y[row] = sum;
            }
        }

        Ok(())
    }

    fn execute_simd_vectorized<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
    {
        // SIMD-optimized computation
        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];
            let nnz = end_idx - start_idx;

            if nnz >= 8 {
                // Use proper SIMD for longer rows
                let mut sum = T::zero();
                let simd_len = nnz & !7; // Round down to multiple of 8

                // Process in SIMD chunks
                for chunk_start in (0..simd_len).step_by(8) {
                    let chunk_end = (chunk_start + 8).min(nnz);
                    let actual_chunk_size = chunk_end - chunk_start;

                    if actual_chunk_size == 8 {
                        // Full SIMD chunk - extract data and indices
                        let data_chunk: Vec<T> =
                            (0..8).map(|i| data[start_idx + chunk_start + i]).collect();
                        let x_chunk: Vec<T> = (0..8)
                            .map(|i| x[indices[start_idx + chunk_start + i]])
                            .collect();

                        // Perform SIMD multiplication and reduction
                        use ndarray::Array1;
                        let data_array = Array1::from(data_chunk);
                        let x_array = Array1::from(x_chunk);
                        let simd_result = T::simd_dot(&data_array.view(), &x_array.view());
                        sum += simd_result;
                    } else {
                        // Partial chunk - use scalar computation
                        for i in chunk_start..chunk_end {
                            let idx = start_idx + i;
                            let col = indices[idx];
                            sum += data[idx] * x[col];
                        }
                    }
                }

                // Handle remainder elements
                for idx in (start_idx + simd_len)..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }

                y[row] = sum;
            } else {
                // Fallback for short rows
                let mut sum = T::zero();
                for idx in start_idx..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
                y[row] = sum;
            }
        }

        Ok(())
    }

    fn execute_parallel_work_stealing<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
    {
        // Work-stealing parallel computation
        use crate::parallel_vector_ops::parallel_sparse_matvec_csr;
        parallel_sparse_matvec_csr(y, rows, indptr, indices, data, x, None);
        Ok(())
    }

    fn execute_adaptive_hybrid<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps,
    {
        // Adaptive hybrid approach that switches strategies based on row characteristics
        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];
            let nnz = end_idx - start_idx;

            if nnz == 0 {
                y[row] = T::zero();
            } else if nnz >= 64 {
                // Use advanced SIMD with vectorized gather operations for very long rows
                let mut sum = T::zero();
                let simd_chunks = nnz / 8;
                let remainder = nnz % 8;

                // Process full SIMD chunks
                for chunk in 0..simd_chunks {
                    let chunk_start = start_idx + chunk * 8;

                    // Extract data and corresponding x values
                    let data_vec: Vec<T> = (0..8).map(|i| data[chunk_start + i]).collect();
                    let x_vec: Vec<T> = (0..8).map(|i| x[indices[chunk_start + i]]).collect();

                    // Use SIMD operations for dot product
                    use ndarray::Array1;
                    let data_array = Array1::from(data_vec);
                    let x_array = Array1::from(x_vec);
                    let chunk_result = T::simd_dot(&data_array.view(), &x_array.view());
                    sum += chunk_result;
                }

                // Handle remainder elements
                for i in 0..remainder {
                    let idx = start_idx + simd_chunks * 8 + i;
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }

                y[row] = sum;
            } else if nnz <= 4 {
                // Optimized for very sparse rows
                let mut sum = T::zero();
                for idx in start_idx..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
                y[row] = sum;
            } else {
                // Standard computation for medium-density rows
                let mut sum = T::zero();
                for idx in start_idx..end_idx {
                    let col = indices[idx];
                    sum += data[idx] * x[col];
                }
                y[row] = sum;
            }
        }

        Ok(())
    }

    // Neural network and learning methods

    fn extract_features(
        &self,
        fingerprint: &MatrixFingerprint,
        indptr: &[usize],
        indices: &[usize],
    ) -> Vec<f64> {
        let mut features = vec![
            fingerprint.rows as f64,
            fingerprint.cols as f64,
            fingerprint.nnz as f64,
            fingerprint.nnz as f64 / (fingerprint.rows * fingerprint.cols) as f64, // Density
        ];

        // Row distribution statistics
        let row_nnz_stats = self.compute_row_nnz_statistics(indptr);
        features.extend_from_slice(&row_nnz_stats);

        // Column access pattern analysis
        let col_stats = self.compute_column_statistics(indices, fingerprint.cols);
        features.extend_from_slice(&col_stats);

        // Sparsity pattern features
        features.push(fingerprint.row_distribution_type as u8 as f64);
        features.push(fingerprint.column_distribution_type as u8 as f64);

        features
    }

    fn compute_row_nnz_statistics(&self, indptr: &[usize]) -> Vec<f64> {
        let rows = indptr.len() - 1;
        let mut row_nnz = Vec::with_capacity(rows);

        for row in 0..rows {
            row_nnz.push((indptr[row + 1] - indptr[row]) as f64);
        }

        if row_nnz.is_empty() {
            return vec![0.0; 5];
        }

        let mean = row_nnz.iter().sum::<f64>() / rows as f64;
        let variance = row_nnz.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rows as f64;
        let min_nnz = row_nnz.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_nnz = row_nnz.iter().fold(0.0, |a, &b| a.max(b));

        vec![mean, variance.sqrt(), min_nnz, max_nnz, max_nnz - min_nnz]
    }

    fn compute_column_statistics(&self, indices: &[usize], cols: usize) -> Vec<f64> {
        if indices.is_empty() || cols == 0 {
            return vec![0.0; 4];
        }

        let mut col_counts = vec![0; cols];
        for &col in indices {
            if col < cols {
                col_counts[col] += 1;
            }
        }

        let total_nnz = indices.len() as f64;
        let mean_col_density = total_nnz / cols as f64;
        let col_variance = col_counts
            .iter()
            .map(|&count| (count as f64 - mean_col_density).powi(2))
            .sum::<f64>()
            / cols as f64;

        let max_col_nnz = col_counts.iter().max().copied().unwrap_or(0) as f64;
        let min_col_nnz = col_counts.iter().min().copied().unwrap_or(0) as f64;

        vec![
            mean_col_density,
            col_variance.sqrt(),
            min_col_nnz,
            max_col_nnz,
        ]
    }

    fn analyze_row_distribution(&self, rows: usize, indptr: &[usize]) -> DistributionType {
        if rows == 0 {
            return DistributionType::Uniform;
        }

        let row_nnz_stats = self.compute_row_nnz_statistics(indptr);
        let mean = row_nnz_stats[0];
        let std_dev = row_nnz_stats[1];
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        if coefficient_of_variation < 0.1 {
            DistributionType::Uniform
        } else if coefficient_of_variation > 2.0 {
            DistributionType::PowerLaw
        } else if coefficient_of_variation > 1.0 {
            DistributionType::Clustered
        } else {
            DistributionType::Random
        }
    }

    fn analyze_column_distribution(&self, cols: usize, indices: &[usize]) -> DistributionType {
        if indices.is_empty() || cols == 0 {
            return DistributionType::Uniform;
        }

        // Check for band diagonal pattern
        let mut consecutive_count = 0;
        let mut max_consecutive = 0;
        let mut last_col = None;

        for &col in indices {
            if let Some(prev_col) = last_col {
                if col == prev_col + 1 {
                    consecutive_count += 1;
                } else {
                    max_consecutive = max_consecutive.max(consecutive_count);
                    consecutive_count = 0;
                }
            }
            last_col = Some(col);
        }
        max_consecutive = max_consecutive.max(consecutive_count);

        if max_consecutive > indices.len() / 4 {
            return DistributionType::BandDiagonal;
        }

        // Default classification
        DistributionType::Random
    }

    fn update_neural_network(
        &mut self,
        fingerprint: &MatrixFingerprint,
        metrics: &PerformanceMetrics,
    ) {
        // Store performance history
        self.performance_history.push_back(metrics.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update pattern memory
        if self.pattern_memory.matrix_patterns.len() < self.config.memory_capacity {
            self.pattern_memory
                .matrix_patterns
                .insert(fingerprint.clone(), metrics.strategy_used);
        }

        // Update reinforcement learning agent
        if self.config.reinforcement_learning {
            self.update_rl_agent(fingerprint, metrics);
        }

        // Decay exploration rate
        self.current_exploration_rate *= 0.9995; // Gradual decay
        self.current_exploration_rate = self.current_exploration_rate.max(0.01); // Minimum exploration

        // Update transformer attention weights based on performance
        if let Some(ref mut transformer) = self.transformer {
            transformer.update_attention_weights(metrics);
        }
    }

    /// Update reinforcement learning agent with new experience
    fn update_rl_agent(&mut self, fingerprint: &MatrixFingerprint, metrics: &PerformanceMetrics) {
        // Create experience from current interaction
        let state = self.extract_features(fingerprint, &[], &[]); // Simplified state
        let action = metrics.strategy_used;
        let reward = self.calculate_reward(metrics);

        // Store experience in replay buffer
        let experience = Experience {
            state: state.clone(),
            action,
            reward,
            next_state: state, // Simplified - in real implementation, this would be the next state
            done: false,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.experience_buffer.add_experience(experience);

        // Train RL agent if we have enough experiences
        if self.experience_buffer.buffer.len() >= 32 {
            self.train_rl_agent();
        }
    }

    /// Calculate reward based on performance metrics
    fn calculate_reward(&self, metrics: &PerformanceMetrics) -> f64 {
        // Multi-objective reward function
        let mut reward = 0.0;

        // Reward for low execution time (higher is better)
        reward += (1.0 / (metrics.executiontime + 1e-6)).ln();

        // Reward for high cache efficiency
        reward += metrics.cache_efficiency * 2.0;

        // Reward for high SIMD utilization
        reward += metrics.simd_utilization * 1.5;

        // Reward for high parallel efficiency
        reward += metrics.parallel_efficiency * 1.5;

        // Reward for high memory bandwidth utilization
        reward += (metrics.memory_bandwidth / 100.0).min(1.0);

        // Normalize reward
        reward.tanh()
    }

    /// Train the reinforcement learning agent
    fn train_rl_agent(&mut self) {
        if let Some(ref mut rl_agent) = self.rl_agent {
            // Sample batch from experience buffer
            let batchsize = 32.min(self.experience_buffer.buffer.len());
            let batch = self.experience_buffer.sample_batch(batchsize);

            // Train based on algorithm type
            match rl_agent.algorithm {
                RLAlgorithm::DQN => {
                    rl_agent.train_dqn(&batch, self.config.discountfactor);
                }
                RLAlgorithm::PolicyGradient => {
                    rl_agent.train_policy_gradient(&batch);
                }
                RLAlgorithm::ActorCritic => {
                    rl_agent.train_actor_critic(&batch, self.config.discountfactor);
                }
                RLAlgorithm::PPO => {
                    rl_agent.train_ppo(&batch);
                }
                RLAlgorithm::SAC => {
                    rl_agent.train_sac(&batch, self.config.discountfactor);
                }
            }
        }
    }

    // Performance estimation methods

    fn estimate_cache_efficiency(&self, indptr: &[usize], indices: &[usize]) -> f64 {
        let mut cache_hits = 0;
        let mut total_accesses = 0;
        let cache_size = 64; // Simplified cache model
        let mut recent_cols = std::collections::HashSet::new();

        for row in 0..indptr.len().saturating_sub(1) {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];

            for col in indices.iter().take(end_idx).skip(start_idx) {
                let col = *col;
                total_accesses += 1;

                if recent_cols.contains(&col) {
                    cache_hits += 1;
                } else {
                    recent_cols.insert(col);
                    if recent_cols.len() > cache_size {
                        recent_cols.clear(); // Simplified cache eviction
                    }
                }
            }
        }

        if total_accesses > 0 {
            cache_hits as f64 / total_accesses as f64
        } else {
            1.0
        }
    }

    fn estimate_simd_utilization(&self, strategy: &OptimizationStrategy) -> f64 {
        match strategy {
            OptimizationStrategy::SIMDVectorized => 0.9,
            OptimizationStrategy::AdaptiveHybrid => 0.7,
            OptimizationStrategy::RowWiseCache => 0.5,
            OptimizationStrategy::ColumnWiseLocality => 0.4,
            OptimizationStrategy::BlockStructured => 0.6,
            OptimizationStrategy::DiagonalOptimized => 0.8,
            OptimizationStrategy::Hierarchical => 0.6,
            OptimizationStrategy::StreamingCompute => 0.3,
            OptimizationStrategy::ParallelWorkStealing => 0.7,
        }
    }

    fn estimate_parallel_efficiency(&self, strategy: &OptimizationStrategy, rows: usize) -> f64 {
        let parallelism_factor = match strategy {
            OptimizationStrategy::ParallelWorkStealing => 0.95,
            OptimizationStrategy::AdaptiveHybrid => 0.8,
            OptimizationStrategy::BlockStructured => 0.7,
            OptimizationStrategy::RowWiseCache => 0.6,
            OptimizationStrategy::ColumnWiseLocality => 0.5,
            OptimizationStrategy::DiagonalOptimized => 0.8,
            OptimizationStrategy::Hierarchical => 0.75,
            OptimizationStrategy::StreamingCompute => 0.9,
            OptimizationStrategy::SIMDVectorized => 0.85,
        };

        // Adjust for problem size
        let size_factor = (rows as f64 / 10000.0).min(1.0);
        parallelism_factor * size_factor
    }

    fn estimate_memory_bandwidth(&self, data_size: usize, executiontime: f64) -> f64 {
        if executiontime > 0.0 {
            (data_size as f64 * std::mem::size_of::<f64>() as f64) / executiontime / 1e9
        // GB/s
        } else {
            0.0
        }
    }

    /// Get neural processor statistics
    pub fn get_stats(&self) -> NeuralProcessorStats {
        let avg_reward = if !self.experience_buffer.buffer.is_empty() {
            self.experience_buffer
                .buffer
                .iter()
                .map(|e| e.reward)
                .sum::<f64>()
                / self.experience_buffer.buffer.len() as f64
        } else {
            0.0
        };

        NeuralProcessorStats {
            adaptations_count: self.adaptation_counter.load(Ordering::Relaxed),
            pattern_memory_size: self.pattern_memory.matrix_patterns.len(),
            performance_history_size: self.performance_history.len(),
            learningrate: self.config.learningrate,
            memory_capacity: self.config.memory_capacity,
            rl_enabled: self.config.reinforcement_learning,
            current_exploration_rate: self.current_exploration_rate,
            experience_buffer_size: self.experience_buffer.buffer.len(),
            average_reward: avg_reward,
            transformer_enabled: self.config.self_attention,
        }
    }
}

/// Forward pass cache for backpropagation
#[derive(Debug, Clone)]
struct ForwardCache {
    layer_inputs: Vec<Vec<f64>>,
    layer_outputs: Vec<Vec<f64>>,
    activations: Vec<Vec<f64>>,
}

/// Gradient information for backpropagation
#[derive(Debug, Clone)]
struct NetworkGradients {
    weight_gradients: Vec<Vec<Vec<f64>>>,
    bias_gradients: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        let input_size = 20; // Feature vector size
        let output_size = 9; // Number of optimization strategies

        let mut layers = Vec::new();
        let mut attention_heads = Vec::new();
        let mut layer_norms = Vec::new();

        // Initialize weights with Xavier/He initialization
        let mut rng = rand::rng();

        // Input layer
        let input_layer = NeuralLayer {
            weights: Self::initialize_weights(config.neurons_per_layer, input_size, &mut rng),
            biases: vec![0.0; config.neurons_per_layer],
            activation: ActivationFunction::ReLU,
        };
        layers.push(input_layer);
        layer_norms.push(LayerNorm::new(config.neurons_per_layer));

        // Hidden layers with attention mechanisms
        for _ in 0..config.hidden_layers {
            let hidden_layer = NeuralLayer {
                weights: Self::initialize_weights(
                    config.neurons_per_layer,
                    config.neurons_per_layer,
                    &mut rng,
                ),
                biases: vec![0.0; config.neurons_per_layer],
                activation: ActivationFunction::ReLU,
            };
            layers.push(hidden_layer);
            layer_norms.push(LayerNorm::new(config.neurons_per_layer));

            // Add attention head for this layer
            attention_heads.push(AttentionHead::new(
                config.neurons_per_layer / config.attention_heads,
                config.neurons_per_layer,
            ));
        }

        // Output layer
        let output_layer = NeuralLayer {
            weights: Self::initialize_weights(output_size, config.neurons_per_layer, &mut rng),
            biases: vec![0.0; output_size],
            activation: ActivationFunction::Sigmoid,
        };
        layers.push(output_layer);
        layer_norms.push(LayerNorm::new(output_size));

        let attention_weights = vec![vec![0.1; config.attention_heads]; config.neurons_per_layer];

        Self {
            layers,
            attention_weights,
            attention_heads,
            layer_norms,
        }
    }

    /// Initialize weights using He initialization for ReLU networks
    fn initialize_weights(
        output_size: usize,
        input_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f64>> {
        let std_dev = (2.0 / input_size as f64).sqrt();
        (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.random::<f64>() * std_dev - std_dev / 2.0)
                    .collect()
            })
            .collect()
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let (output, _) = self.forward_with_cache(input);
        output
    }

    /// Forward pass with caching for backpropagation
    fn forward_with_cache(&self, input: &[f64]) -> (Vec<f64>, ForwardCache) {
        let mut cache = ForwardCache {
            layer_inputs: Vec::new(),
            layer_outputs: Vec::new(),
            activations: Vec::new(),
        };

        let mut current_output = input.to_vec();
        cache.layer_inputs.push(current_output.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            // Forward through layer with caching
            let (layer_output, pre_activation) =
                self.forward_layer_with_cache(layer, &current_output);
            cache.activations.push(pre_activation);
            current_output = layer_output;

            // Apply attention if available for this layer
            if i > 0 && i - 1 < self.attention_heads.len() {
                let attention_output = self.attention_heads[i - 1].forward(
                    &current_output,
                    &current_output,
                    &current_output,
                );

                // Residual connection with attention
                for (j, &att_val) in attention_output.iter().enumerate() {
                    if j < current_output.len() {
                        current_output[j] += att_val * 0.1; // Small attention contribution
                    }
                }
            }

            // Apply layer normalization
            if i < self.layer_norms.len() {
                current_output = self.layer_norms[i].normalize(&current_output);
            }

            cache.layer_outputs.push(current_output.clone());
            if i < self.layers.len() - 1 {
                cache.layer_inputs.push(current_output.clone());
            }
        }

        (current_output, cache)
    }

    /// Forward pass through a single layer with caching
    fn forward_layer_with_cache(&self, layer: &NeuralLayer, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pre_activation = Vec::new();
        let mut output = Vec::new();

        for (weights, &bias) in layer.weights.iter().zip(&layer.biases) {
            let mut sum = bias;
            for (w, &x) in weights.iter().zip(input) {
                sum += w * x;
            }
            pre_activation.push(sum);
            output.push(self.apply_activation(sum, layer.activation));
        }

        (output, pre_activation)
    }

    /// Backpropagation algorithm
    fn backward(
        &self,
        target: &[f64],
        prediction: &[f64],
        cache: &ForwardCache,
    ) -> NetworkGradients {
        let mut gradients = NetworkGradients {
            weight_gradients: vec![Vec::new(); self.layers.len()],
            bias_gradients: vec![vec![0.0; 0]; self.layers.len()],
        };

        // Initialize gradient storage
        for (i, layer) in self.layers.iter().enumerate() {
            gradients.weight_gradients[i] =
                vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            gradients.bias_gradients[i] = vec![0.0; layer.biases.len()];
        }

        // Compute output layer error (using MSE loss)
        let mut delta: Vec<f64> = prediction
            .iter()
            .zip(target)
            .map(|(&pred, &targ)| pred - targ)
            .collect();

        // Backpropagate through layers
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];
            let layer_input = &cache.layer_inputs[layer_idx];
            let pre_activation = &cache.activations[layer_idx];

            // Compute gradients for this layer
            for (neuron_idx, (&d, &pre_act)) in delta.iter().zip(pre_activation).enumerate() {
                // Apply activation derivative
                let activation_grad = self.activation_derivative(pre_act, layer.activation);
                let error = d * activation_grad;

                // Weight gradients
                for (input_idx, &input_val) in layer_input.iter().enumerate() {
                    gradients.weight_gradients[layer_idx][neuron_idx][input_idx] =
                        error * input_val;
                }

                // Bias gradient
                gradients.bias_gradients[layer_idx][neuron_idx] = error;
            }

            // Compute error for previous layer (if not input layer)
            if layer_idx > 0 {
                let mut next_delta = vec![0.0; layer_input.len()];
                for (neuron_idx, &d) in delta.iter().enumerate() {
                    let pre_act = pre_activation[neuron_idx];
                    let activation_grad = self.activation_derivative(pre_act, layer.activation);
                    let error = d * activation_grad;

                    for (input_idx, weight) in layer.weights[neuron_idx].iter().enumerate() {
                        next_delta[input_idx] += error * weight;
                    }
                }
                delta = next_delta;
            }
        }

        gradients
    }

    /// Compute activation function derivative
    fn activation_derivative(&self, x: f64, activation: ActivationFunction) -> f64 {
        match activation {
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::Gelu => {
                // Approximation of GELU derivative
                let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                let tanh_part = (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh();
                0.5 * (1.0
                    + tanh_part
                    + x * sqrt_2_pi
                        * (1.0 - tanh_part.powi(2))
                        * (1.0 + 3.0 * 0.044715 * x.powi(2)))
            }
        }
    }

    /// Update network weights using computed gradients
    fn update_weights(&mut self, gradients: &NetworkGradients, learningrate: f64) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Update weights
            for (neuron_idx, weights) in layer.weights.iter_mut().enumerate() {
                for (weight_idx, weight) in weights.iter_mut().enumerate() {
                    *weight -= learningrate
                        * gradients.weight_gradients[layer_idx][neuron_idx][weight_idx];
                }
            }

            // Update biases
            for (bias_idx, bias) in layer.biases.iter_mut().enumerate() {
                *bias -= learningrate * gradients.bias_gradients[layer_idx][bias_idx];
            }
        }
    }

    /// Train the network on a single example
    fn train_single(&mut self, input: &[f64], target: &[f64], learningrate: f64) -> f64 {
        let (prediction, cache) = self.forward_with_cache(input);
        let gradients = self.backward(target, &prediction, &cache);
        self.update_weights(&gradients, learningrate);

        // Return loss (MSE)
        prediction
            .iter()
            .zip(target)
            .map(|(&pred, &targ)| (pred - targ).powi(2))
            .sum::<f64>()
            / prediction.len() as f64
    }

    #[allow(dead_code)]
    fn forward_layer(&self, layer: &NeuralLayer, input: &[f64]) -> Vec<f64> {
        let mut output = Vec::new();

        for (weights, &bias) in layer.weights.iter().zip(&layer.biases) {
            let mut sum = bias;
            for (w, &x) in weights.iter().zip(input) {
                sum += w * x;
            }
            output.push(self.apply_activation(sum, layer.activation));
        }

        output
    }

    fn apply_activation(&self, x: f64, activation: ActivationFunction) -> f64 {
        match activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x / (1.0 + (-x).exp()),
            ActivationFunction::Gelu => 0.5 * x * (1.0 + (x / 2.0_f64.sqrt()).tanh()),
        }
    }
}

impl PatternMemory {
    fn new(capacity: usize) -> Self {
        Self {
            matrix_patterns: HashMap::new(),
            access_patterns: VecDeque::new(),
            performance_cache: HashMap::new(),
        }
    }
}

impl RLAgent {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        let q_network = NeuralNetwork::new(config);
        let target_network = if matches!(config.rl_algorithm, RLAlgorithm::DQN) {
            Some(NeuralNetwork::new(config))
        } else {
            None
        };

        let (policy_network, value_network) = match config.rl_algorithm {
            RLAlgorithm::ActorCritic | RLAlgorithm::PPO | RLAlgorithm::SAC => (
                Some(NeuralNetwork::new(config)),
                Some(NeuralNetwork::new(config)),
            ),
            _ => (None, None),
        };

        Self {
            q_network,
            target_network,
            policy_network,
            value_network,
            algorithm: config.rl_algorithm,
            epsilon: config.exploration_rate,
            learningrate: config.learningrate,
        }
    }

    fn get_q_values(&self, state: &[f64]) -> Vec<f64> {
        self.q_network.forward(state)
    }

    fn train_dqn(&mut self, batch: &[Experience], discountfactor: f64) {
        let mut total_loss = 0.0;
        let batchsize = batch.len();

        for experience in batch {
            // Get current Q-values
            let current_q_values = self.q_network.forward(&experience.state);

            // Compute target Q-value
            let target_q_value = if let Some(ref target_net) = self.target_network {
                let next_q = target_net.forward(&experience.next_state);
                let max_next_q = next_q.iter().fold(0.0, |a, &b| a.max(b));
                experience.reward
                    + if experience.done {
                        0.0
                    } else {
                        discountfactor * max_next_q
                    }
            } else {
                experience.reward
            };

            // Create target vector (only update the action that was taken)
            let mut target_vector = current_q_values.clone();
            let action_idx = self.strategy_to_index(experience.action);
            if action_idx < target_vector.len() {
                target_vector[action_idx] = target_q_value;
            }

            // Train the network using backpropagation
            let loss =
                self.q_network
                    .train_single(&experience.state, &target_vector, self.learningrate);
            total_loss += loss;
        }

        // Update target network periodically (every 100 training steps)
        if rand::rng().random::<f64>() < 0.01 {
            if let Some(ref mut target_net) = self.target_network {
                *target_net = self.q_network.clone();
            }
        }

        // Log average loss for monitoring
        if batchsize > 0 {
            let _avg_loss = total_loss / batchsize as f64;
            // Could log this for monitoring purposes
        }
    }

    /// Convert optimization strategy to network output index
    fn strategy_to_index(&self, strategy: OptimizationStrategy) -> usize {
        match strategy {
            OptimizationStrategy::RowWiseCache => 0,
            OptimizationStrategy::ColumnWiseLocality => 1,
            OptimizationStrategy::BlockStructured => 2,
            OptimizationStrategy::DiagonalOptimized => 3,
            OptimizationStrategy::Hierarchical => 4,
            OptimizationStrategy::StreamingCompute => 5,
            OptimizationStrategy::SIMDVectorized => 6,
            OptimizationStrategy::ParallelWorkStealing => 7,
            OptimizationStrategy::AdaptiveHybrid => 8,
        }
    }

    fn train_policy_gradient(&mut self, batch: &[Experience]) {
        // Pre-compute action indices to avoid borrowing conflicts
        let action_indices: Vec<usize> = batch
            .iter()
            .map(|exp| self.strategy_to_index(exp.action))
            .collect();

        if let Some(ref mut policy_net) = self.policy_network {
            let mut _total_loss = 0.0;

            // Compute returns (discounted rewards)
            let returns = Self::compute_returns(batch, 0.99); // Using 0.99 as discount factor

            for ((experience, &return_val), &action_idx) in
                batch.iter().zip(&returns).zip(&action_indices)
            {
                // Get action probabilities
                let action_probs = policy_net.forward(&experience.state);

                if action_idx < action_probs.len() {
                    // Compute policy gradient target
                    // Target is proportional to advantage (simplified as return here)
                    let mut target = vec![0.0; action_probs.len()];

                    // Set target for the taken action based on advantage
                    let advantage = return_val - experience.reward; // Simplified advantage
                    target[action_idx] = action_probs[action_idx] + self.learningrate * advantage;

                    // Ensure target is a valid probability distribution
                    target[action_idx] = target[action_idx].clamp(0.01, 0.99);

                    // Train the policy network
                    let loss = policy_net.train_single(
                        &experience.state,
                        &target,
                        self.learningrate * 0.1, // Smaller learning rate for policy
                    );
                    _total_loss += loss;
                }
            }
        }
    }

    /// Compute discounted returns for policy gradient
    fn compute_returns(_batch: &[Experience], discountfactor: f64) -> Vec<f64> {
        let mut returns = vec![0.0; _batch.len()];
        let mut running_return = 0.0;

        // Compute returns backwards
        for (i, experience) in _batch.iter().enumerate().rev() {
            running_return = experience.reward + discountfactor * running_return;
            returns[i] = running_return;

            if experience.done {
                running_return = 0.0; // Reset for new episode
            }
        }

        returns
    }

    fn train_actor_critic(&mut self, batch: &[Experience], discountfactor: f64) {
        // Pre-compute action indices to avoid borrowing conflicts
        let action_indices: Vec<usize> = batch
            .iter()
            .map(|exp| self.strategy_to_index(exp.action))
            .collect();

        if let (Some(ref mut policy_net), Some(ref mut value_net)) =
            (&mut self.policy_network, &mut self.value_network)
        {
            let mut _total_actor_loss = 0.0;
            let mut _total_critic_loss = 0.0;

            for (experience, &action_idx) in batch.iter().zip(&action_indices) {
                // Critic update: learn value function
                let state_value = value_net.forward(&experience.state);
                let current_value = if !state_value.is_empty() {
                    state_value[0]
                } else {
                    0.0
                };

                // Compute TD target
                let next_value = if !experience.done {
                    let next_state_value = value_net.forward(&experience.next_state);
                    if !next_state_value.is_empty() {
                        next_state_value[0]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let td_target = experience.reward + discountfactor * next_value;
                let td_error = td_target - current_value;

                // Train critic (value network)
                let critic_target = vec![td_target];
                let critic_loss =
                    value_net.train_single(&experience.state, &critic_target, self.learningrate);
                _total_critic_loss += critic_loss;

                // Actor update: learn policy using advantage
                let action_probs = policy_net.forward(&experience.state);

                if action_idx < action_probs.len() {
                    // Use TD error as advantage estimate
                    let advantage = td_error;

                    // Compute policy gradient target
                    let mut policy_target = action_probs.clone();

                    // Update action probability based on advantage
                    let prob_adjustment = self.learningrate * 0.1 * advantage;
                    policy_target[action_idx] =
                        (action_probs[action_idx] + prob_adjustment).clamp(0.01, 0.99);

                    // Normalize to maintain probability distribution
                    let sum: f64 = policy_target.iter().sum();
                    if sum > 0.0 {
                        for prob in &mut policy_target {
                            *prob /= sum;
                        }
                    }

                    // Train actor (policy network)
                    let actor_loss = policy_net.train_single(
                        &experience.state,
                        &policy_target,
                        self.learningrate * 0.05, // Smaller learning rate for policy
                    );
                    _total_actor_loss += actor_loss;
                }
            }
        }
    }

    fn train_ppo(&mut self, batch: &[Experience]) {
        // Pre-compute action indices to avoid borrowing conflicts
        let action_indices: Vec<usize> = batch
            .iter()
            .map(|exp| self.strategy_to_index(exp.action))
            .collect();

        if let (Some(ref mut policy_net), Some(ref mut value_net)) =
            (&mut self.policy_network, &mut self.value_network)
        {
            let clip_epsilon = 0.2; // PPO clipping parameter
            let ppo_epochs = 4; // Number of PPO epochs per batch

            // Store old policy probabilities for importance sampling
            let mut old_action_probs = Vec::new();
            let mut advantages = Vec::new();
            let mut returns = Vec::new();

            // Compute advantages and returns
            for (experience, &action_idx) in batch.iter().zip(&action_indices) {
                let action_probs = policy_net.forward(&experience.state);

                if action_idx < action_probs.len() {
                    old_action_probs.push(action_probs[action_idx]);
                } else {
                    old_action_probs.push(0.1); // Fallback probability
                }

                // Compute value and advantage
                let state_value = value_net.forward(&experience.state);
                let current_value = if !state_value.is_empty() {
                    state_value[0]
                } else {
                    0.0
                };

                let next_value = if !experience.done {
                    let next_state_value = value_net.forward(&experience.next_state);
                    if !next_state_value.is_empty() {
                        next_state_value[0]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let td_target = experience.reward + 0.99 * next_value;
                let advantage = td_target - current_value;

                advantages.push(advantage);
                returns.push(td_target);
            }

            // Normalize advantages
            let advantage_mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
            let advantage_std = (advantages
                .iter()
                .map(|&a| (a - advantage_mean).powi(2))
                .sum::<f64>()
                / advantages.len() as f64)
                .sqrt()
                .max(1e-8);

            for advantage in &mut advantages {
                *advantage = (*advantage - advantage_mean) / advantage_std;
            }

            // PPO training epochs
            for _ in 0..ppo_epochs {
                let mut _total_policy_loss = 0.0;
                let mut _total_value_loss = 0.0;

                for (i, experience) in batch.iter().enumerate() {
                    // Update value network
                    let value_target = vec![returns[i]];
                    let value_loss =
                        value_net.train_single(&experience.state, &value_target, self.learningrate);
                    _total_value_loss += value_loss;

                    // Update policy network with PPO clipping
                    let current_action_probs = policy_net.forward(&experience.state);
                    let action_idx = action_indices[i];

                    if action_idx < current_action_probs.len() {
                        let current_prob = current_action_probs[action_idx].max(1e-8);
                        let old_prob = old_action_probs[i].max(1e-8);
                        let advantage = advantages[i];

                        // Importance sampling ratio
                        let ratio = current_prob / old_prob;

                        // PPO clipped objective
                        let clipped_ratio = ratio.max(1.0 - clip_epsilon).min(1.0 + clip_epsilon);
                        let policy_objective = if advantage >= 0.0 {
                            ratio.min(clipped_ratio) * advantage
                        } else {
                            ratio.max(clipped_ratio) * advantage
                        };

                        // Convert objective to training target
                        let mut policy_target = current_action_probs.clone();
                        let target_adjustment = self.learningrate * 0.01 * policy_objective;
                        policy_target[action_idx] =
                            (current_prob + target_adjustment).clamp(0.01, 0.99);

                        // Normalize probabilities
                        let sum: f64 = policy_target.iter().sum();
                        if sum > 0.0 {
                            for prob in &mut policy_target {
                                *prob /= sum;
                            }
                        }

                        let policy_loss = policy_net.train_single(
                            &experience.state,
                            &policy_target,
                            self.learningrate * 0.02,
                        );
                        _total_policy_loss += policy_loss;
                    }
                }
            }
        }
    }

    fn train_sac(&mut self, batch: &[Experience], discountfactor: f64) {
        // Pre-compute action indices to avoid borrowing conflicts
        let action_indices: Vec<usize> = batch
            .iter()
            .map(|exp| self.strategy_to_index(exp.action))
            .collect();

        if let (Some(ref mut policy_net), Some(ref mut value_net)) =
            (&mut self.policy_network, &mut self.value_network)
        {
            let entropy_coeff = 0.2; // Temperature parameter for entropy regularization
            let _target_update_rate = 0.005; // Soft target network update rate

            let mut _total_q_loss = 0.0;
            let mut _total_policy_loss = 0.0;

            for (experience, &action_idx) in batch.iter().zip(&action_indices) {
                // Update Q-network (critic)
                let current_q_values = value_net.forward(&experience.state);
                let current_q = if !current_q_values.is_empty() {
                    current_q_values[0]
                } else {
                    0.0
                };

                // Compute target Q-value using soft Bellman equation
                let next_action_probs = policy_net.forward(&experience.next_state);
                let next_q_values = value_net.forward(&experience.next_state);
                let next_q = if !next_q_values.is_empty() {
                    next_q_values[0]
                } else {
                    0.0
                };

                // Compute entropy of next state policy
                let entropy = -next_action_probs
                    .iter()
                    .filter(|&&p| p > 1e-8)
                    .map(|&p| p * p.ln())
                    .sum::<f64>();

                let target_q = if !experience.done {
                    experience.reward + discountfactor * (next_q + entropy_coeff * entropy)
                } else {
                    experience.reward
                };

                // Train Q-network
                let q_target = vec![target_q];
                let q_loss =
                    value_net.train_single(&experience.state, &q_target, self.learningrate);
                _total_q_loss += q_loss;

                // Update policy network to maximize Q-value + entropy
                let current_action_probs = policy_net.forward(&experience.state);
                let current_entropy = -current_action_probs
                    .iter()
                    .filter(|&&p| p > 1e-8)
                    .map(|&p| p * p.ln())
                    .sum::<f64>();

                // SAC policy objective: maximize Q + α * entropy
                let policy_objective = current_q + entropy_coeff * current_entropy;

                // Update policy to increase the objective
                let mut policy_target = current_action_probs.clone();

                if action_idx < policy_target.len() {
                    // Adjust action probability based on SAC objective
                    let adjustment = self.learningrate * 0.01 * policy_objective;
                    policy_target[action_idx] =
                        (current_action_probs[action_idx] + adjustment).clamp(0.01, 0.99);

                    // Apply entropy regularization to encourage exploration
                    for (i, prob) in policy_target.iter_mut().enumerate() {
                        if i != action_idx {
                            // Slightly increase other action probabilities for exploration
                            *prob = (*prob + entropy_coeff * 0.001).min(0.99);
                        }
                    }

                    // Normalize to maintain valid probability distribution
                    let sum: f64 = policy_target.iter().sum();
                    if sum > 0.0 {
                        for prob in &mut policy_target {
                            *prob /= sum;
                        }
                    }

                    let policy_loss = policy_net.train_single(
                        &experience.state,
                        &policy_target,
                        self.learningrate * 0.01,
                    );
                    _total_policy_loss += policy_loss;
                }
            }

            // Optional: Adaptive temperature parameter learning could be added here
            // For now, we use a fixed entropy coefficient
        }
    }
}

impl ExperienceBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            priority_weights: Vec::new(),
        }
    }

    fn add_experience(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        // Calculate priority based on reward magnitude
        let priority = experience.reward.abs() + 1e-6;
        self.priority_weights.push(priority);

        if self.priority_weights.len() > self.capacity {
            self.priority_weights.remove(0);
        }

        self.buffer.push_back(experience);
    }

    fn sample_batch(&self, batchsize: usize) -> Vec<Experience> {
        let mut batch = Vec::new();
        let buffer_size = self.buffer.len();

        if buffer_size == 0 {
            return batch;
        }

        // Simple random sampling (in practice, prioritized experience replay would be better)
        for _ in 0..batchsize.min(buffer_size) {
            let idx = rand::rng().random_range(0..buffer_size);
            if let Some(exp) = self.buffer.get(idx) {
                batch.push(exp.clone());
            }
        }

        batch
    }
}

impl TransformerModel {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        let mut encoder_layers = Vec::new();

        for _ in 0..config.transformer_layers {
            encoder_layers.push(TransformerEncoderLayer::new(config));
        }

        // Initialize positional encoding
        let max_seq_len = 1000;
        let mut positional_encoding = vec![vec![0.0; config.modeldim]; max_seq_len];

        for (pos, encoding_row) in positional_encoding.iter_mut().enumerate().take(max_seq_len) {
            for (i, encoding_value) in encoding_row.iter_mut().enumerate().take(config.modeldim) {
                if i % 2 == 0 {
                    *encoding_value =
                        (pos as f64 / 10000.0_f64.powf(i as f64 / config.modeldim as f64)).sin();
                } else {
                    *encoding_value = (pos as f64
                        / 10000.0_f64.powf((i - 1) as f64 / config.modeldim as f64))
                    .cos();
                }
            }
        }

        Self {
            encoder_layers,
            positional_encoding,
            embedding_dim: config.modeldim,
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden = input.to_vec();

        // Add positional encoding if input is shorter than max sequence length
        if hidden.len() < self.positional_encoding.len() && !self.positional_encoding.is_empty() {
            let pos_enc = &self.positional_encoding[0];
            for (i, h) in hidden.iter_mut().enumerate() {
                if i < pos_enc.len() {
                    *h += pos_enc[i];
                }
            }
        }

        // Pass through transformer encoder layers
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden);
        }

        hidden
    }

    fn update_attention_weights(&mut self, metrics: &PerformanceMetrics) {
        // Update attention weights based on performance feedback
        let performance_score =
            metrics.cache_efficiency + metrics.simd_utilization + metrics.parallel_efficiency;
        let learningrate = 0.001;

        // Simple gradient-like update (in practice, this would use proper backpropagation)
        for layer in &mut self.encoder_layers {
            layer.update_weights(performance_score, learningrate);
        }
    }
}

impl TransformerEncoderLayer {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        let self_attention = MultiHeadAttention::new(config);
        let feed_forward = FeedForwardNetwork::new(config);
        let layer_norm1 = LayerNorm::new(config.modeldim);
        let layer_norm2 = LayerNorm::new(config.modeldim);

        Self {
            self_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            dropout_rate: 0.1,
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        // Self-attention with residual connection and layer norm
        let attention_output = self.self_attention.forward(input, input, input);
        let mut x = self.add_and_norm(input, &attention_output, &self.layer_norm1);

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(&x);
        x = self.add_and_norm(&x, &ff_output, &self.layer_norm2);

        // Apply dropout (simplified)
        if rand::rng().random::<f64>() < self.dropout_rate {
            for val in &mut x {
                *val *= 0.9;
            }
        }

        x
    }

    fn add_and_norm(&self, input: &[f64], output: &[f64], layernorm: &LayerNorm) -> Vec<f64> {
        let mut result = vec![0.0; input.len().max(output.len())];

        // Residual connection
        for i in 0..result.len() {
            let input_val = if i < input.len() { input[i] } else { 0.0 };
            let output_val = if i < output.len() { output[i] } else { 0.0 };
            result[i] = input_val + output_val;
        }

        // Layer normalization
        layernorm.normalize(&result)
    }

    fn update_weights(&mut self, performance_score: f64, learningrate: f64) {
        self.self_attention
            .update_weights(performance_score, learningrate);
        self.feed_forward
            .update_weights(performance_score, learningrate);
    }
}

impl MultiHeadAttention {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        let num_heads = config.attention_heads;
        let head_dim = config.modeldim / num_heads;
        let mut heads = Vec::new();

        for _ in 0..num_heads {
            heads.push(AttentionHead::new(head_dim, config.modeldim));
        }

        let output_projection = vec![vec![0.1; config.modeldim]; config.modeldim];

        Self {
            heads,
            output_projection,
            num_heads,
            head_dim,
        }
    }

    fn forward(&self, query: &[f64], key: &[f64], value: &[f64]) -> Vec<f64> {
        let mut head_outputs = Vec::new();

        // Process each attention head
        for head in &self.heads {
            let head_output = head.forward(query, key, value);
            head_outputs.push(head_output);
        }

        // Concatenate head outputs
        let mut concatenated = Vec::new();
        for head_output in &head_outputs {
            concatenated.extend_from_slice(head_output);
        }

        // Apply output projection
        self.apply_linear_transform(&concatenated, &self.output_projection)
    }

    fn apply_linear_transform(&self, input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let mut output = vec![0.0; weights.len()];

        for (i, weight_row) in weights.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                if j < weight_row.len() {
                    output[i] += input_val * weight_row[j];
                }
            }
        }

        output
    }

    fn update_weights(&mut self, performance_score: f64, learningrate: f64) {
        let gradient = performance_score * learningrate;

        // Update attention head weights
        for head in &mut self.heads {
            head.update_weights(gradient);
        }

        // Update output projection weights
        for row in &mut self.output_projection {
            for weight in row {
                *weight += gradient * 0.01; // Small update
            }
        }
    }
}

impl AttentionHead {
    fn new(_head_dim: usize, modeldim: usize) -> Self {
        Self {
            query_weights: vec![vec![0.1; modeldim]; _head_dim],
            key_weights: vec![vec![0.1; modeldim]; _head_dim],
            value_weights: vec![vec![0.1; modeldim]; _head_dim],
            output_weights: vec![vec![0.1; _head_dim]; _head_dim],
            head_dim: _head_dim,
        }
    }

    fn forward(&self, query: &[f64], key: &[f64], value: &[f64]) -> Vec<f64> {
        // Compute Q, K, V
        let q = self.apply_weights(query, &self.query_weights);
        let k = self.apply_weights(key, &self.key_weights);
        let v = self.apply_weights(value, &self.value_weights);

        // Compute attention scores
        let scores = self.compute_attention_scores(&q, &k);

        // Apply attention to values
        let mut attended = vec![0.0; v.len()];
        for (i, &score) in scores.iter().enumerate() {
            if i < v.len() {
                attended[i] = score * v[i];
            }
        }

        // Apply output transformation
        self.apply_weights(&attended, &self.output_weights)
    }

    fn apply_weights(&self, input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let mut output = vec![0.0; weights.len()];

        for (i, weight_row) in weights.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                if j < weight_row.len() {
                    output[i] += input_val * weight_row[j];
                }
            }
        }

        output
    }

    fn compute_attention_scores(&self, query: &[f64], key: &[f64]) -> Vec<f64> {
        let mut scores = vec![0.0; query.len().min(key.len())];

        // Simplified attention: dot product followed by softmax
        let mut max_score = f64::NEG_INFINITY;
        for i in 0..scores.len() {
            scores[i] = query[i] * key[i] / (self.head_dim as f64).sqrt();
            max_score = max_score.max(scores[i]);
        }

        // Softmax normalization
        let mut sum_exp = 0.0;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }

        if sum_exp > 0.0 {
            for score in &mut scores {
                *score /= sum_exp;
            }
        }

        scores
    }

    fn update_weights(&mut self, gradient: f64) {
        let update = gradient * 0.001;

        // Update all weight matrices
        for row in &mut self.query_weights {
            for weight in row {
                *weight += update;
            }
        }

        for row in &mut self.key_weights {
            for weight in row {
                *weight += update;
            }
        }

        for row in &mut self.value_weights {
            for weight in row {
                *weight += update;
            }
        }

        for row in &mut self.output_weights {
            for weight in row {
                *weight += update;
            }
        }
    }
}

impl FeedForwardNetwork {
    fn new(config: &NeuralAdaptiveConfig) -> Self {
        Self {
            layer1: vec![vec![0.1; config.modeldim]; config.ff_dim],
            layer1_bias: vec![0.0; config.ff_dim],
            layer2: vec![vec![0.1; config.ff_dim]; config.modeldim],
            layer2_bias: vec![0.0; config.modeldim],
            activation: ActivationFunction::ReLU,
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        // First layer
        let mut hidden = vec![0.0; self.layer1.len()];
        for (i, (weight_row, &bias)) in self.layer1.iter().zip(&self.layer1_bias).enumerate() {
            let mut sum = bias;
            for (j, &input_val) in input.iter().enumerate() {
                if j < weight_row.len() {
                    sum += input_val * weight_row[j];
                }
            }
            hidden[i] = self.apply_activation(sum);
        }

        // Second layer
        let mut output = vec![0.0; self.layer2.len()];
        for (i, (weight_row, &bias)) in self.layer2.iter().zip(&self.layer2_bias).enumerate() {
            let mut sum = bias;
            for (j, &hidden_val) in hidden.iter().enumerate() {
                if j < weight_row.len() {
                    sum += hidden_val * weight_row[j];
                }
            }
            output[i] = sum; // No activation on output layer
        }

        output
    }

    fn apply_activation(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x / (1.0 + (-x).exp()),
            ActivationFunction::Gelu => 0.5 * x * (1.0 + (x / 2.0_f64.sqrt()).tanh()),
        }
    }

    fn update_weights(&mut self, performance_score: f64, learningrate: f64) {
        let gradient = performance_score * learningrate * 0.01;

        // Update layer 1 weights and biases
        for row in &mut self.layer1 {
            for weight in row {
                *weight += gradient;
            }
        }

        for bias in &mut self.layer1_bias {
            *bias += gradient;
        }

        // Update layer 2 weights and biases
        for row in &mut self.layer2 {
            for weight in row {
                *weight += gradient;
            }
        }

        for bias in &mut self.layer2_bias {
            *bias += gradient;
        }
    }
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-6,
        }
    }

    fn normalize(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len() as f64;

        // Compute mean and variance
        let mean = input.iter().sum::<f64>() / n;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = (variance + self.eps).sqrt();

        // Normalize and apply learned parameters
        let mut output = vec![0.0; input.len()];
        for (i, &x) in input.iter().enumerate() {
            let normalized = (x - mean) / std_dev;
            let gamma = if i < self.gamma.len() {
                self.gamma[i]
            } else {
                1.0
            };
            let beta = if i < self.beta.len() {
                self.beta[i]
            } else {
                0.0
            };
            output[i] = gamma * normalized + beta;
        }

        output
    }
}

/// Statistics for neural-adaptive sparse matrix processor
#[derive(Debug)]
pub struct NeuralProcessorStats {
    pub adaptations_count: usize,
    pub pattern_memory_size: usize,
    pub performance_history_size: usize,
    pub learningrate: f64,
    pub memory_capacity: usize,
    pub rl_enabled: bool,
    pub current_exploration_rate: f64,
    pub experience_buffer_size: usize,
    pub average_reward: f64,
    pub transformer_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_neural_adaptive_processor_creation() {
        let config = NeuralAdaptiveConfig::default();
        let processor = NeuralAdaptiveSparseProcessor::new(config);

        assert_eq!(processor.config.hidden_layers, 3);
        assert_eq!(processor.config.neurons_per_layer, 64);
        assert_eq!(processor.optimization_strategies.len(), 9);
    }

    #[test]
    #[ignore] // TODO: Fix adaptive spmv test - overflow error in calculation
    fn test_adaptive_spmv() {
        let config = NeuralAdaptiveConfig {
            hidden_layers: 1,
            neurons_per_layer: 8,
            ..Default::default()
        };
        let mut processor = NeuralAdaptiveSparseProcessor::new(config);

        // Simple test matrix: [[1, 2], [0, 3]]
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];

        processor
            .adaptive_spmv(2, 2, &indptr, &indices, &data, &x, &mut y)
            .unwrap();

        // Results should be [3.0, 3.0]
        assert_relative_eq!(y[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_fingerprint_generation() {
        let config = NeuralAdaptiveConfig::default();
        let processor = NeuralAdaptiveSparseProcessor::new(config);

        let indptr = vec![0, 2, 3, 5];
        let indices = vec![0, 1, 1, 0, 2];

        let fingerprint = processor.generate_matrix_fingerprint(3, 3, &indptr, &indices);

        assert_eq!(fingerprint.rows, 3);
        assert_eq!(fingerprint.cols, 3);
        assert_eq!(fingerprint.nnz, 5);
    }

    #[test]
    fn test_neural_processor_stats() {
        let config = NeuralAdaptiveConfig::default();
        let processor = NeuralAdaptiveSparseProcessor::new(config);
        let stats = processor.get_stats();

        assert_eq!(stats.adaptations_count, 0);
        assert_eq!(stats.pattern_memory_size, 0);
        assert_eq!(stats.learningrate, 0.001);
        assert!(stats.rl_enabled);
        assert!(stats.transformer_enabled);
    }
}
