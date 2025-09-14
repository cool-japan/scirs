//! Adaptive optimization algorithm selection
//!
//! This module provides automatic selection of the most appropriate optimization algorithm
//! based on problem characteristics, performance monitoring, and learned patterns.

use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Types of optimization algorithms available for selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer
    Adam,
    /// AdamW (Adam with decoupled weight decay)
    AdamW,
    /// RMSprop optimizer
    RMSprop,
    /// AdaGrad optimizer
    AdaGrad,
    /// RAdam (Rectified Adam)
    RAdam,
    /// Lookahead wrapper
    Lookahead,
    /// LAMB (Layer-wise Adaptive Moments)
    LAMB,
    /// LARS (Layer-wise Adaptive Rate Scaling)
    LARS,
    /// L-BFGS (Limited-memory BFGS)
    LBFGS,
    /// SAM (Sharpness-Aware Minimization)
    SAM,
}

/// Problem characteristics for optimizer selection
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Dataset size
    pub dataset_size: usize,
    /// Input dimensionality
    pub input_dim: usize,
    /// Output dimensionality
    pub output_dim: usize,
    /// Problem type (classification, regression, etc.)
    pub problem_type: ProblemType,
    /// Gradient sparsity (0.0 = dense, 1.0 = very sparse)
    pub gradient_sparsity: f64,
    /// Noise level in gradients
    pub gradient_noise: f64,
    /// Memory constraints (bytes available)
    pub memory_budget: usize,
    /// Computational budget (time constraints)
    pub time_budget: f64,
    /// Batch size being used
    pub batch_size: usize,
    /// Learning rate range preference
    pub lr_sensitivity: f64,
    /// Regularization requirements
    pub regularization_strength: f64,
    /// Architecture type (if applicable)
    pub architecture_type: Option<String>,
}

/// Types of machine learning problems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProblemType {
    /// Classification task
    Classification,
    /// Regression task
    Regression,
    /// Unsupervised learning
    Unsupervised,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Time series forecasting
    TimeSeries,
    /// Computer vision
    ComputerVision,
    /// Natural language processing
    NaturalLanguage,
    /// Recommendation systems
    Recommendation,
}

/// Performance metrics for optimizer evaluation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Final loss/error achieved
    pub final_loss: f64,
    /// Convergence speed (steps to reach target)
    pub convergence_steps: usize,
    /// Training time taken
    pub training_time: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// Validation performance
    pub validation_performance: f64,
    /// Stability (variance in loss)
    pub stability: f64,
    /// Generalization (validation - training performance)
    pub generalization_gap: f64,
}

/// Selection strategy for adaptive optimization
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Rule-based selection using expert knowledge
    RuleBased,
    /// Learning-based selection using historical data
    LearningBased,
    /// Ensemble selection trying multiple optimizers
    Ensemble {
        /// Number of optimizers to try
        num_candidates: usize,
        /// Number of steps for evaluation
        evaluation_steps: usize,
    },
    /// Bandit-based selection with exploration/exploitation
    Bandit {
        /// Exploration parameter
        epsilon: f64,
        /// UCB confidence parameter
        confidence: f64,
    },
    /// Meta-learning based selection
    MetaLearning {
        /// Feature extractor for problems
        feature_dim: usize,
        /// Number of similar problems to consider
        k_nearest: usize,
    },
}

/// Adaptive optimizer selector
#[derive(Debug)]
pub struct AdaptiveOptimizerSelector<A: Float> {
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Historical performance data
    performance_history: HashMap<OptimizerType, Vec<PerformanceMetrics>>,
    /// Problem-optimizer mapping for learning
    problem_optimizer_map: Vec<(ProblemCharacteristics, OptimizerType, PerformanceMetrics)>,
    /// Current problem characteristics
    current_problem: Option<ProblemCharacteristics>,
    /// Bandit arm statistics (if using bandit strategy)
    arm_counts: HashMap<OptimizerType, usize>,
    arm_rewards: HashMap<OptimizerType, f64>,
    /// Neural network for learning-based selection
    selection_network: Option<SelectionNetwork<A>>,
    /// Available optimizers
    available_optimizers: Vec<OptimizerType>,
    /// Performance tracking
    current_performance: VecDeque<f64>,
    /// Selection confidence
    last_confidence: f64,
}

/// Neural network for optimizer selection
#[derive(Debug)]
pub struct SelectionNetwork<A: Float> {
    /// Input weights (problem features -> hidden)
    input_weights: Array2<A>,
    /// Output weights (hidden -> optimizer probabilities)
    output_weights: Array2<A>,
    /// Input biases
    input_bias: Array1<A>,
    /// Output biases
    output_bias: Array1<A>,
    /// Hidden layer size
    #[allow(dead_code)]
    hidden_size: usize,
}

impl<A: Float + ScalarOperand + Debug + num_traits::FromPrimitive> SelectionNetwork<A> {
    /// Create a new selection network
    pub fn new(input_size: usize, hidden_size: usize, num_optimizers: usize) -> Self {
        let mut rng = scirs2_core::random::rng();

        let input_weights = Array2::from_shape_fn((hidden_size, input_size), |_| {
            A::from(rng.random_f64()).unwrap() * A::from(0.1).unwrap() - A::from(0.05).unwrap()
        });

        let output_weights = Array2::from_shape_fn((num_optimizers, hidden_size), |_| {
            A::from(rng.random_f64()).unwrap() * A::from(0.1).unwrap() - A::from(0.05).unwrap()
        });

        let input_bias = Array1::zeros(hidden_size);
        let output_bias = Array1::zeros(num_optimizers);

        Self {
            input_weights,
            output_weights,
            input_bias,
            output_bias,
            hidden_size,
        }
    }

    /// Forward pass to get optimizer probabilities
    pub fn forward(&self, features: &Array1<A>) -> Result<Array1<A>> {
        // Hidden layer
        let hidden = self.input_weights.dot(features) + self.input_bias.clone();
        let hidden_activated = hidden.mapv(|x| {
            // ReLU activation
            if x > A::zero() {
                x
            } else {
                A::zero()
            }
        });

        // Output layer
        let output = self.output_weights.dot(&hidden_activated) + &self.output_bias;

        // Softmax activation
        let max_val = output.iter().fold(A::neg_infinity(), |a, &b| A::max(a, b));
        let exp_output = output.mapv(|x| A::exp(x - max_val));
        let sum_exp = exp_output.sum();
        let probabilities = exp_output.mapv(|x| x / sum_exp);

        Ok(probabilities)
    }

    /// Train the network on historical data
    pub fn train(
        &mut self,
        features: &[Array1<A>],
        optimizer_labels: &[usize],
        learning_rate: A,
        epochs: usize,
    ) -> Result<()> {
        for _ in 0..epochs {
            for (feature, &label) in features.iter().zip(optimizer_labels.iter()) {
                // Forward pass
                let probabilities = self.forward(feature)?;

                // Compute loss (cross-entropy)
                let target_prob = probabilities[label];
                let _loss = -A::ln(target_prob);

                // Backward pass (simplified)
                let mut output_grad = probabilities;
                output_grad[label] = output_grad[label] - A::one();

                // Update weights (simplified gradient descent)
                let hidden = self.input_weights.dot(feature) + self.input_bias.clone();
                let hidden_activated = hidden.mapv(|x| if x > A::zero() { x } else { A::zero() });

                // Update output weights
                for i in 0..self.output_weights.nrows() {
                    for j in 0..self.output_weights.ncols() {
                        self.output_weights[[i, j]] = self.output_weights[[i, j]]
                            - learning_rate * output_grad[i] * hidden_activated[j];
                    }
                }

                // Update output bias
                for i in 0..self.output_bias.len() {
                    self.output_bias[i] = self.output_bias[i] - learning_rate * output_grad[i];
                }
            }
        }
        Ok(())
    }
}

impl<A: Float + ScalarOperand + Debug + num_traits::FromPrimitive> AdaptiveOptimizerSelector<A> {
    /// Create a new adaptive optimizer selector
    pub fn new(strategy: SelectionStrategy) -> Self {
        let available_optimizers = vec![
            OptimizerType::SGD,
            OptimizerType::SGDMomentum,
            OptimizerType::Adam,
            OptimizerType::AdamW,
            OptimizerType::RMSprop,
            OptimizerType::AdaGrad,
            OptimizerType::RAdam,
            OptimizerType::LAMB,
        ];

        let mut arm_counts = HashMap::new();
        let mut arm_rewards = HashMap::new();
        for &optimizer in &available_optimizers {
            arm_counts.insert(optimizer, 0);
            arm_rewards.insert(optimizer, 0.0);
        }

        Self {
            strategy,
            performance_history: HashMap::new(),
            problem_optimizer_map: Vec::new(),
            current_problem: None,
            arm_counts,
            arm_rewards,
            selection_network: None,
            available_optimizers,
            current_performance: VecDeque::new(),
            last_confidence: 0.0,
        }
    }

    /// Set the current problem characteristics
    pub fn set_problem(&mut self, problem: ProblemCharacteristics) {
        self.current_problem = Some(problem);
    }

    /// Select the best optimizer for the current problem
    pub fn select_optimizer(&mut self) -> Result<OptimizerType> {
        let problem = self.current_problem.clone().ok_or_else(|| {
            OptimError::InvalidConfig("No problem characteristics set".to_string())
        })?;

        match &self.strategy {
            SelectionStrategy::RuleBased => self.rule_based_selection(&problem),
            SelectionStrategy::LearningBased => self.learning_based_selection(&problem),
            SelectionStrategy::Ensemble {
                num_candidates,
                evaluation_steps,
            } => self.ensemble_selection(&problem, *num_candidates, *evaluation_steps),
            SelectionStrategy::Bandit {
                epsilon,
                confidence,
            } => self.bandit_selection(&problem, *epsilon, *confidence),
            SelectionStrategy::MetaLearning {
                feature_dim,
                k_nearest,
            } => self.meta_learning_selection(&problem, *feature_dim),
        }
    }

    /// Rule-based optimizer selection using expert knowledge
    fn rule_based_selection(&self, problem: &ProblemCharacteristics) -> Result<OptimizerType> {
        // Large dataset, use adaptive optimizers
        if problem.dataset_size > 100000 {
            match problem.problem_type {
                ProblemType::ComputerVision => return Ok(OptimizerType::AdamW),
                ProblemType::NaturalLanguage => return Ok(OptimizerType::AdamW),
                _ => return Ok(OptimizerType::Adam),
            }
        }

        // Small dataset, use SGD with momentum
        if problem.dataset_size < 1000 {
            return Ok(OptimizerType::LBFGS);
        }

        // Sparse gradients
        if problem.gradient_sparsity > 0.5 {
            return Ok(OptimizerType::AdaGrad);
        }

        // Large batch training
        if problem.batch_size > 256 {
            return Ok(OptimizerType::LAMB);
        }

        // Memory constrained
        if problem.memory_budget < 1_000_000 {
            return Ok(OptimizerType::SGD);
        }

        // High noise
        if problem.gradient_noise > 0.3 {
            return Ok(OptimizerType::RMSprop);
        }

        // Default choice
        Ok(OptimizerType::Adam)
    }

    /// Learning-based selection using historical performance
    fn learning_based_selection(
        &mut self,
        problem: &ProblemCharacteristics,
    ) -> Result<OptimizerType> {
        if self.problem_optimizer_map.is_empty() {
            // No historical data, fall back to rule-based
            return self.rule_based_selection(problem);
        }

        // Find most similar problem in history
        let mut best_similarity = -1.0;
        let mut best_optimizer = OptimizerType::Adam;

        for (hist_problem, optimizer, metrics) in &self.problem_optimizer_map {
            let similarity = self.compute_problem_similarity(problem, hist_problem);

            // Weight by performance
            let weighted_similarity = similarity * metrics.validation_performance;

            if weighted_similarity > best_similarity {
                best_similarity = weighted_similarity;
                best_optimizer = *optimizer;
            }
        }

        self.last_confidence = best_similarity;
        Ok(best_optimizer)
    }

    /// Ensemble selection by trying multiple optimizers
    fn ensemble_selection(
        &self,
        problem: &ProblemCharacteristics,
        num_candidates: usize,
        _evaluation_steps: usize,
    ) -> Result<OptimizerType> {
        // Select top _candidates based on historical performance
        let mut candidates = self.available_optimizers.clone();
        candidates.truncate(num_candidates.min(candidates.len()));

        // For simplicity, return the first candidate
        // In practice, you would evaluate each for evaluation_steps
        Ok(candidates[0])
    }

    /// Bandit-based selection with epsilon-greedy strategy
    fn bandit_selection(
        &self,
        problem: &ProblemCharacteristics,
        epsilon: f64,
        confidence: f64,
    ) -> Result<OptimizerType> {
        let mut rng = scirs2_core::random::rng();

        // Epsilon-greedy exploration
        if rng.random_f64() < epsilon {
            // Explore: random selection
            let idx = rng.gen_range(0..self.available_optimizers.len());
            return Ok(self.available_optimizers[idx]);
        }

        // Exploit: UCB (Upper Confidence Bound) selection
        let mut best_ucb = f64::NEG_INFINITY;
        let mut best_optimizer = OptimizerType::Adam;
        let total_counts: usize = self.arm_counts.values().sum();

        for &optimizer in &self.available_optimizers {
            let count = self.arm_counts[&optimizer] as f64;
            let reward = if count > 0.0 {
                self.arm_rewards[&optimizer] / count
            } else {
                0.0
            };

            let ucb = if count > 0.0 {
                reward + confidence * ((total_counts as f64).ln() / count).sqrt()
            } else {
                f64::INFINITY // Prefer unvisited arms
            };

            if ucb > best_ucb {
                best_ucb = ucb;
                best_optimizer = optimizer;
            }
        }

        Ok(best_optimizer)
    }

    /// Meta-learning based selection
    fn meta_learning_selection(
        &mut self,
        problem: &ProblemCharacteristics,
        k_nearest: usize,
    ) -> Result<OptimizerType> {
        // Extract features from problem
        let features = self.extract_problem_features(problem);

        // If we have a trained network, use it
        if let Some(network) = &self.selection_network {
            let probabilities = network.forward(&features)?;

            // Select optimizer with highest probability
            let mut best_prob = A::neg_infinity();
            let mut best_idx = 0;

            for (i, &prob) in probabilities.iter().enumerate() {
                if prob > best_prob {
                    best_prob = prob;
                    best_idx = i;
                }
            }

            if best_idx < self.available_optimizers.len() {
                return Ok(self.available_optimizers[best_idx]);
            }
        }

        // k-NN fallback
        if self.problem_optimizer_map.len() >= k_nearest {
            let mut similarities = Vec::new();

            for (hist_problem, optimizer, metrics) in &self.problem_optimizer_map {
                let similarity = self.compute_problem_similarity(problem, hist_problem);
                similarities.push((similarity, *optimizer, metrics.validation_performance));
            }

            // Sort by similarity
            similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Take k _nearest and vote
            let mut votes: HashMap<OptimizerType, f64> = HashMap::new();
            for (similarity, optimizer, performance) in similarities.iter().take(k_nearest) {
                let weight = similarity * performance;
                *votes.entry(*optimizer).or_insert(0.0) += weight;
            }

            // Return optimizer with highest weighted vote
            let best_optimizer = votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(optimizer_, _)| *optimizer_)
                .unwrap_or(OptimizerType::Adam);

            return Ok(best_optimizer);
        }

        // Fall back to rule-based
        self.rule_based_selection(problem)
    }

    /// Update selector with performance feedback
    pub fn update_performance(
        &mut self,
        optimizer: OptimizerType,
        metrics: PerformanceMetrics,
    ) -> Result<()> {
        // Update performance history
        self.performance_history
            .entry(optimizer)
            .or_default()
            .push(metrics.clone());

        // Update bandit statistics
        *self.arm_counts.entry(optimizer).or_insert(0) += 1;
        *self.arm_rewards.entry(optimizer).or_insert(0.0) += metrics.validation_performance;

        // Store problem-optimizer mapping
        if let Some(problem) = &self.current_problem {
            self.problem_optimizer_map
                .push((problem.clone(), optimizer, metrics.clone()));
        }

        // Update current performance tracking
        self.current_performance
            .push_back(metrics.validation_performance);
        if self.current_performance.len() > 100 {
            self.current_performance.pop_front();
        }

        Ok(())
    }

    /// Train the selection network if using learning-based strategy
    pub fn train_selection_network(&mut self, learning_rate: A, epochs: usize) -> Result<()> {
        if self.problem_optimizer_map.is_empty() {
            return Ok(()); // No data to train on
        }

        // Extract features and labels
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (problem, optimizer_, metrics) in &self.problem_optimizer_map {
            let feature_vec = self.extract_problem_features(problem);
            features.push(feature_vec);

            // Convert optimizer to label
            if let Some(label) = self
                .available_optimizers
                .iter()
                .position(|&opt| opt == *optimizer_)
            {
                labels.push(label);
            }
        }

        // Create network if it doesn't exist
        if self.selection_network.is_none() {
            let feature_dim = features[0].len();
            let num_optimizers = self.available_optimizers.len();
            self.selection_network = Some(SelectionNetwork::new(feature_dim, 32, num_optimizers));
        }

        // Train the network
        if let Some(network) = &mut self.selection_network {
            network.train(&features, &labels, learning_rate, epochs)?;
        }

        Ok(())
    }

    /// Compute similarity between two problems
    fn compute_problem_similarity(
        &self,
        problem1: &ProblemCharacteristics,
        problem2: &ProblemCharacteristics,
    ) -> f64 {
        let mut similarity = 0.0;
        let mut weight_sum = 0.0;

        // Dataset size similarity (log scale)
        let size_sim = 1.0
            - ((problem1.dataset_size as f64).ln() - (problem2.dataset_size as f64).ln()).abs()
                / 10.0;
        similarity += size_sim.max(0.0) * 0.2;
        weight_sum += 0.2;

        // Problem type similarity
        if problem1.problem_type == problem2.problem_type {
            similarity += 0.3;
        }
        weight_sum += 0.3;

        // Batch size similarity
        let batch_sim = 1.0
            - ((problem1.batch_size as f64 - problem2.batch_size as f64).abs() / 256.0).min(1.0);
        similarity += batch_sim * 0.1;
        weight_sum += 0.1;

        // Gradient characteristics similarity
        let sparsity_sim = 1.0 - (problem1.gradient_sparsity - problem2.gradient_sparsity).abs();
        let noise_sim = 1.0 - (problem1.gradient_noise - problem2.gradient_noise).abs();
        similarity += (sparsity_sim + noise_sim) * 0.2;
        weight_sum += 0.4;

        similarity / weight_sum
    }

    /// Extract numerical features from problem characteristics
    fn extract_problem_features(&self, problem: &ProblemCharacteristics) -> Array1<A> {
        Array1::from_vec(vec![
            A::from((problem.dataset_size as f64).ln()).unwrap(),
            A::from((problem.input_dim as f64).ln()).unwrap(),
            A::from((problem.output_dim as f64).ln()).unwrap(),
            A::from(problem.problem_type as u8 as f64).unwrap(),
            A::from(problem.gradient_sparsity).unwrap(),
            A::from(problem.gradient_noise).unwrap(),
            A::from((problem.memory_budget as f64).ln()).unwrap(),
            A::from(problem.time_budget.ln()).unwrap(),
            A::from((problem.batch_size as f64).ln()).unwrap(),
            A::from(problem.lr_sensitivity).unwrap(),
            A::from(problem.regularization_strength).unwrap(),
        ])
    }

    /// Get performance statistics for an optimizer
    pub fn get_optimizer_statistics(
        &self,
        optimizer: OptimizerType,
    ) -> Option<OptimizerStatistics> {
        if let Some(history) = self.performance_history.get(&optimizer) {
            if history.is_empty() {
                return None;
            }

            let performances: Vec<f64> = history.iter().map(|m| m.validation_performance).collect();
            let mean = performances.iter().sum::<f64>() / performances.len() as f64;
            let variance = performances.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                / performances.len() as f64;
            let std_dev = variance.sqrt();

            Some(OptimizerStatistics {
                optimizer,
                num_trials: history.len(),
                mean_performance: mean,
                std_performance: std_dev,
                best_performance: performances
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                worst_performance: performances.iter().copied().fold(f64::INFINITY, f64::min),
                success_rate: performances.iter().filter(|&&p| p > 0.7).count() as f64
                    / performances.len() as f64,
            })
        } else {
            None
        }
    }

    /// Get all optimizer statistics
    pub fn get_all_statistics(&self) -> Vec<OptimizerStatistics> {
        self.available_optimizers
            .iter()
            .filter_map(|&opt| self.get_optimizer_statistics(opt))
            .collect()
    }

    /// Get current confidence in selection
    pub fn get_selection_confidence(&self) -> f64 {
        self.last_confidence
    }

    /// Reset selector state
    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.problem_optimizer_map.clear();
        self.current_problem = None;
        for count in self.arm_counts.values_mut() {
            *count = 0;
        }
        for reward in self.arm_rewards.values_mut() {
            *reward = 0.0;
        }
        self.current_performance.clear();
        self.last_confidence = 0.0;
    }
}

/// Statistics for an optimizer's performance
#[derive(Debug, Clone)]
pub struct OptimizerStatistics {
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Number of trials
    pub num_trials: usize,
    /// Mean performance
    pub mean_performance: f64,
    /// Standard deviation of performance
    pub std_performance: f64,
    /// Best performance achieved
    pub best_performance: f64,
    /// Worst performance achieved
    pub worst_performance: f64,
    /// Success rate (performance > threshold)
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_problem_characteristics() {
        let problem = ProblemCharacteristics {
            dataset_size: 10000,
            input_dim: 784,
            output_dim: 10,
            problem_type: ProblemType::Classification,
            gradient_sparsity: 0.1,
            gradient_noise: 0.05,
            memory_budget: 1_000_000,
            time_budget: 3600.0,
            batch_size: 64,
            lr_sensitivity: 0.5,
            regularization_strength: 0.01,
            architecture_type: Some("CNN".to_string()),
        };

        assert_eq!(problem.dataset_size, 10000);
        assert_eq!(problem.problem_type, ProblemType::Classification);
    }

    #[test]
    fn test_rule_based_selection() {
        let mut selector = AdaptiveOptimizerSelector::<f64>::new(SelectionStrategy::RuleBased);

        // Large dataset -> Adam/AdamW
        let large_problem = ProblemCharacteristics {
            dataset_size: 100001,
            input_dim: 224,
            output_dim: 1000,
            problem_type: ProblemType::ComputerVision,
            gradient_sparsity: 0.1,
            gradient_noise: 0.05,
            memory_budget: 10_000_000,
            time_budget: 7200.0,
            batch_size: 32,
            lr_sensitivity: 0.5,
            regularization_strength: 0.01,
            architecture_type: Some("ResNet".to_string()),
        };

        selector.set_problem(large_problem);
        let optimizer = selector.select_optimizer().unwrap();
        assert_eq!(optimizer, OptimizerType::AdamW);
    }

    #[test]
    fn test_selection_network() {
        let network = SelectionNetwork::<f64>::new(5, 10, 3);
        let features = Array1::from_vec(vec![1.0, 0.5, 2.0, 0.8, 1.5]);

        let probabilities = network.forward(&features).unwrap();
        assert_eq!(probabilities.len(), 3);

        // Probabilities should sum to 1
        let sum: f64 = probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All probabilities should be non-negative
        for &prob in probabilities.iter() {
            assert!(prob >= 0.0);
        }
    }

    #[test]
    fn test_bandit_selection() {
        let mut selector = AdaptiveOptimizerSelector::<f64>::new(SelectionStrategy::Bandit {
            epsilon: 0.1,
            confidence: 2.0,
        });

        let problem = ProblemCharacteristics {
            dataset_size: 1000,
            input_dim: 10,
            output_dim: 2,
            problem_type: ProblemType::Classification,
            gradient_sparsity: 0.0,
            gradient_noise: 0.1,
            memory_budget: 1_000_000,
            time_budget: 600.0,
            batch_size: 32,
            lr_sensitivity: 0.5,
            regularization_strength: 0.01,
            architecture_type: None,
        };

        selector.set_problem(problem);

        // Should select an optimizer (any is valid initially)
        let optimizer = selector.select_optimizer().unwrap();
        assert!(selector.available_optimizers.contains(&optimizer));
    }

    #[test]
    fn test_performance_update() {
        let mut selector = AdaptiveOptimizerSelector::<f64>::new(SelectionStrategy::RuleBased);

        let metrics = PerformanceMetrics {
            final_loss: 0.1,
            convergence_steps: 100,
            training_time: 60.0,
            memory_usage: 500_000,
            validation_performance: 0.95,
            stability: 0.02,
            generalization_gap: 0.05,
        };

        selector
            .update_performance(OptimizerType::Adam, metrics)
            .unwrap();

        let stats = selector
            .get_optimizer_statistics(OptimizerType::Adam)
            .unwrap();
        assert_eq!(stats.num_trials, 1);
        assert_relative_eq!(stats.mean_performance, 0.95, epsilon = 1e-6);
    }

    #[test]
    fn test_problem_similarity() {
        let selector = AdaptiveOptimizerSelector::<f64>::new(SelectionStrategy::RuleBased);

        let problem1 = ProblemCharacteristics {
            dataset_size: 1000,
            input_dim: 10,
            output_dim: 2,
            problem_type: ProblemType::Classification,
            gradient_sparsity: 0.1,
            gradient_noise: 0.05,
            memory_budget: 1_000_000,
            time_budget: 600.0,
            batch_size: 32,
            lr_sensitivity: 0.5,
            regularization_strength: 0.01,
            architecture_type: None,
        };

        let problem2 = problem1.clone();
        let similarity = selector.compute_problem_similarity(&problem1, &problem2);
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-6);

        let mut problem3 = problem1.clone();
        problem3.problem_type = ProblemType::Regression;
        let similarity = selector.compute_problem_similarity(&problem1, &problem3);
        assert!(similarity < 1.0);
    }
}
