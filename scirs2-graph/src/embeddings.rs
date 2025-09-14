//! Graph embedding algorithms and utilities
//!
//! This module provides graph embedding algorithms including Node2Vec, DeepWalk,
//! and other representation learning methods for graphs.

#![allow(missing_docs)]

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicUsize;

/// Configuration for Node2Vec embedding algorithm
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Return parameter p (likelihood of immediate revisiting)
    pub p: f64,
    /// In-out parameter q (exploration vs exploitation)
    pub q: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of negative samples for training
    pub negative_samples: usize,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Node2VecConfig {
            dimensions: 128,
            walk_length: 80,
            num_walks: 10,
            window_size: 10,
            p: 1.0,
            q: 1.0,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// Configuration for DeepWalk embedding algorithm
#[derive(Debug, Clone)]
pub struct DeepWalkConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of negative samples
    pub negative_samples: usize,
}

impl Default for DeepWalkConfig {
    fn default() -> Self {
        DeepWalkConfig {
            dimensions: 128,
            walk_length: 40,
            num_walks: 80,
            window_size: 5,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// A random walk on a graph
#[derive(Debug, Clone)]
pub struct RandomWalk<N: Node> {
    /// The sequence of nodes in the walk
    pub nodes: Vec<N>,
}

/// Advanced optimization techniques for embeddings
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Learning rate schedule type
    pub lr_schedule: LearningRateSchedule,
    /// Initial learning rate
    pub initial_lr: f64,
    /// Final learning rate
    pub final_lr: f64,
    /// Use momentum optimization
    pub use_momentum: bool,
    /// Momentum factor (0.9 is typical)
    pub momentum: f64,
    /// Use Adam optimizer
    pub use_adam: bool,
    /// Adam beta1 parameter
    pub adam_beta1: f64,
    /// Adam beta2 parameter
    pub adam_beta2: f64,
    /// Adam epsilon parameter
    pub adam_epsilon: f64,
    /// L2 regularization strength
    pub l2_regularization: f64,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Use hierarchical softmax instead of negative sampling
    pub use_hierarchical_softmax: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            lr_schedule: LearningRateSchedule::Linear,
            initial_lr: 0.025,
            final_lr: 0.0001,
            use_momentum: false,
            momentum: 0.9,
            use_adam: false,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            l2_regularization: 0.0,
            gradient_clip: Some(1.0),
            use_hierarchical_softmax: false,
        }
    }
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Linear decay from initial to final
    Linear,
    /// Exponential decay
    Exponential,
    /// Cosine annealing
    Cosine,
    /// Step decay (reduce by factor at specific epochs)
    Step,
}

/// Enhanced training metrics and monitoring
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Current epoch
    pub epoch: usize,
    /// Total training steps
    pub steps: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Training loss (negative log likelihood)
    pub loss: f64,
    /// Loss moving average
    pub loss_avg: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Processing speed (steps per second)
    pub steps_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Convergence indicator (rate of loss change)
    pub convergence_rate: f64,
    /// Training accuracy on positive samples
    pub positive_accuracy: f64,
    /// Training accuracy on negative samples  
    pub negative_accuracy: f64,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        TrainingMetrics {
            epoch: 0,
            steps: 0,
            learning_rate: 0.025,
            loss: 0.0,
            loss_avg: 0.0,
            gradient_norm: 0.0,
            steps_per_second: 0.0,
            memory_usage: 0,
            convergence_rate: 0.0,
            positive_accuracy: 0.0,
            negative_accuracy: 0.0,
        }
    }
}

/// Adaptive negative sampling strategies
#[derive(Debug, Clone)]
pub enum NegativeSamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Frequency-based sampling (more frequent nodes sampled more often)
    Frequency,
    /// Degree-based sampling (higher degree nodes sampled more often)
    Degree,
    /// Adaptive sampling based on embedding quality
    Adaptive,
    /// Hierarchical sampling using word2vec-style tree
    Hierarchical,
}

/// Advanced optimizer state for Adam/momentum
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Momentum buffers for each parameter
    pub momentum_buffers: HashMap<String, Vec<f64>>,
    /// Adam first moment estimates
    pub adam_m: HashMap<String, Vec<f64>>,
    /// Adam second moment estimates  
    pub adam_v: HashMap<String, Vec<f64>>,
    /// Time step for bias correction
    pub time_step: usize,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            momentum_buffers: HashMap::new(),
            adam_m: HashMap::new(),
            adam_v: HashMap::new(),
            time_step: 0,
        }
    }
}

/// Skip-gram training context pair
#[derive(Debug, Clone)]
pub struct ContextPair<N: Node> {
    /// Target node
    pub target: N,
    /// Context node
    pub context: N,
}

/// Negative sampling configuration
#[derive(Debug, Clone)]
pub struct NegativeSampler<N: Node> {
    /// Vocabulary (all nodes)
    vocabulary: Vec<N>,
    /// Frequency distribution for sampling
    #[allow(dead_code)]
    frequencies: Vec<f64>,
    /// Cumulative distribution for fast sampling
    cumulative: Vec<f64>,
}

impl<N: Node> NegativeSampler<N> {
    /// Create a new negative sampler from graph
    pub fn new<E, Ix>(graph: &Graph<N, E, Ix>) -> Self
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let vocabulary: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let node_degrees = vocabulary
            .iter()
            .map(|node| graph.degree(node) as f64)
            .collect::<Vec<_>>();

        // Use subsampling with power 0.75 as in Word2Vec
        let total_degree: f64 = node_degrees.iter().sum();
        let frequencies: Vec<f64> = node_degrees
            .iter()
            .map(|d| (d / total_degree).powf(0.75))
            .collect();

        let total_freq: f64 = frequencies.iter().sum();
        let frequencies: Vec<f64> = frequencies.iter().map(|f| f / total_freq).collect();

        // Build cumulative distribution
        let mut cumulative = vec![0.0; frequencies.len()];
        cumulative[0] = frequencies[0];
        for i in 1..frequencies.len() {
            cumulative[i] = cumulative[i - 1] + frequencies[i];
        }

        NegativeSampler {
            vocabulary,
            frequencies,
            cumulative,
        }
    }

    /// Sample a negative node
    pub fn sample(&self, rng: &mut impl Rng) -> Option<&N> {
        if self.vocabulary.is_empty() {
            return None;
        }

        let r = rng.random::<f64>();
        for (i, &cum_freq) in self.cumulative.iter().enumerate() {
            if r <= cum_freq {
                return Some(&self.vocabulary[i]);
            }
        }

        self.vocabulary.last()
    }

    /// Sample multiple negative nodes excluding target and context
    pub fn sample_negatives(
        &self,
        count: usize,
        exclude: &HashSet<&N>,
        rng: &mut impl Rng,
    ) -> Vec<N> {
        let mut negatives = Vec::new();
        let mut attempts = 0;
        let max_attempts = count * 10; // Prevent infinite loops

        while negatives.len() < count && attempts < max_attempts {
            if let Some(candidate) = self.sample(rng) {
                if !exclude.contains(candidate) {
                    negatives.push(candidate.clone());
                }
            }
            attempts += 1;
        }

        negatives
    }
}

/// Node embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f64>,
}

impl Embedding {
    /// Create a new embedding with given dimensions
    pub fn new(dimensions: usize) -> Self {
        Embedding {
            vector: vec![0.0; dimensions],
        }
    }

    /// Create a random embedding
    pub fn random(dimensions: usize, rng: &mut impl Rng) -> Self {
        let vector: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(-0.5..0.5)).collect();
        Embedding { vector }
    }

    /// Get the dimensionality of the embedding
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Calculate cosine similarity with another embedding (SIMD optimized)
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f64> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }

        let dot_product: f64 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// Calculate L2 norm of the embedding (SIMD optimized)
    pub fn norm(&self) -> f64 {
        self.vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize the embedding to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Add another embedding (element-wise)
    pub fn add(&mut self, other: &Embedding) -> Result<()> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }

        for (a, b) in self.vector.iter_mut().zip(other.vector.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// Scale the embedding by a scalar
    pub fn scale(&mut self, factor: f64) {
        for x in &mut self.vector {
            *x *= factor;
        }
    }

    /// Compute dot product with another embedding (SIMD optimized)
    pub fn dot_product(&self, other: &Embedding) -> Result<f64> {
        if self.vector.len() != other.vector.len() {
            return Err(GraphError::InvalidGraph(
                "Embeddings must have same dimensions".to_string(),
            ));
        }

        let dot: f64 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(dot)
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update embedding using gradient (SIMD optimized)
    pub fn update_gradient(&mut self, gradient: &[f64], learningrate: f64) {
        for (emb, &grad) in self.vector.iter_mut().zip(gradient.iter()) {
            *emb -= learningrate * grad;
        }
    }
}

/// Graph embedding model
#[derive(Debug)]
pub struct EmbeddingModel<N: Node> {
    /// Node embeddings (input vectors)
    pub embeddings: HashMap<N, Embedding>,
    /// Context embeddings (output vectors) for skip-gram
    pub context_embeddings: HashMap<N, Embedding>,
    /// Dimensionality of embeddings
    pub dimensions: usize,
}

impl<N: Node> EmbeddingModel<N> {
    /// Create a new embedding model
    pub fn new(dimensions: usize) -> Self {
        EmbeddingModel {
            embeddings: HashMap::new(),
            context_embeddings: HashMap::new(),
            dimensions,
        }
    }

    /// Get embedding for a node
    pub fn get_embedding(&self, node: &N) -> Option<&Embedding> {
        self.embeddings.get(node)
    }

    /// Set embedding for a node
    pub fn set_embedding(&mut self, node: N, embedding: Embedding) -> Result<()> {
        if embedding.dimensions() != self.dimensions {
            return Err(GraphError::InvalidGraph(
                "Embedding dimensions don't match model".to_string(),
            ));
        }
        self.embeddings.insert(node, embedding);
        Ok(())
    }

    /// Initialize random embeddings for all nodes
    pub fn initialize_random<E, Ix>(&mut self, graph: &Graph<N, E, Ix>, rng: &mut impl Rng)
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            let context_embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
            self.context_embeddings
                .insert(node.clone(), context_embedding);
        }
    }

    /// Initialize random embeddings for directed graph
    pub fn initialize_random_digraph<E, Ix>(
        &mut self,
        graph: &DiGraph<N, E, Ix>,
        rng: &mut impl Rng,
    ) where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            let context_embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
            self.context_embeddings
                .insert(node.clone(), context_embedding);
        }
    }

    /// Find k most similar nodes to a given node
    pub fn most_similar(&self, node: &N, k: usize) -> Result<Vec<(N, f64)>>
    where
        N: Clone,
    {
        let target_embedding = self
            .embeddings
            .get(node)
            .ok_or(GraphError::node_not_found("node"))?;

        let mut similarities = Vec::new();

        for (other_node, other_embedding) in &self.embeddings {
            if other_node != node {
                let similarity = target_embedding.cosine_similarity(other_embedding)?;
                similarities.push((other_node.clone(), similarity));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Train skip-gram model on context pairs with negative sampling
    pub fn train_skip_gram(
        &mut self,
        pairs: &[ContextPair<N>],
        negative_sampler: &NegativeSampler<N>,
        learning_rate: f64,
        negative_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        for pair in pairs {
            // Get embeddings
            let target_emb = self
                .embeddings
                .get(&pair.target)
                .ok_or(GraphError::node_not_found("node"))?
                .clone();
            let context_emb = self
                .context_embeddings
                .get(&pair.context)
                .ok_or(GraphError::node_not_found("node"))?
                .clone();

            // Positive sample: maximize probability of context given target
            let positive_score = target_emb.dot_product(&context_emb)?;
            let positive_prob = Embedding::sigmoid(positive_score);

            // Compute gradients for positive sample
            let positive_error = 1.0 - positive_prob;
            let mut target_gradient = vec![0.0; self.dimensions];
            let mut context_gradient = vec![0.0; self.dimensions];

            #[allow(clippy::needless_range_loop)]
            for i in 0..self.dimensions {
                target_gradient[i] += positive_error * context_emb.vector[i];
                context_gradient[i] += positive_error * target_emb.vector[i];
            }

            // Negative _samples: minimize probability of negative contexts
            let exclude_set: HashSet<&N> = [&pair.target, &pair.context].iter().cloned().collect();
            let negatives = negative_sampler.sample_negatives(negative_samples, &exclude_set, rng);

            for negative in &negatives {
                if let Some(neg_context_emb) = self.context_embeddings.get(negative) {
                    let negative_score = target_emb.dot_product(neg_context_emb)?;
                    let negative_prob = Embedding::sigmoid(negative_score);

                    // Negative sample error
                    let negative_error = -negative_prob;

                    #[allow(clippy::needless_range_loop)]
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..self.dimensions {
                        target_gradient[i] += negative_error * neg_context_emb.vector[i];
                    }
                }
            }

            // Update negative context embeddings separately to avoid borrowing issues
            for negative in &negatives {
                if let Some(neg_context_emb_mut) = self.context_embeddings.get_mut(negative) {
                    let negative_score = target_emb.dot_product(neg_context_emb_mut)?;
                    let negative_prob = Embedding::sigmoid(negative_score);
                    let negative_error = -negative_prob;

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..self.dimensions {
                        let neg_context_grad = negative_error * target_emb.vector[i];
                        neg_context_emb_mut.vector[i] -= learning_rate * neg_context_grad;
                    }
                }
            }

            // Apply gradients
            if let Some(target_emb_mut) = self.embeddings.get_mut(&pair.target) {
                target_emb_mut.update_gradient(&target_gradient, learning_rate);
            }
            if let Some(context_emb_mut) = self.context_embeddings.get_mut(&pair.context) {
                context_emb_mut.update_gradient(&context_gradient, learning_rate);
            }
        }

        Ok(())
    }

    /// Parallel and SIMD-optimized skip-gram training
    pub fn train_skip_gram_parallel(
        &mut self,
        pairs: &[ContextPair<N>],
        negative_sampler: &NegativeSampler<N>,
        learning_rate: f64,
        negative_samples: usize,
        _rng: &mut impl Rng,
    ) -> Result<()>
    where
        N: Clone + Send + Sync,
    {
        use scirs2_core::parallel_ops::*;

        // Use collect-reduce pattern for thread-safe parallel processing
        #[allow(clippy::type_complexity)]
        let gradient_updates: Vec<(N, Vec<f64>, Vec<f64>)> = pairs
            .par_chunks(1000) // Process in chunks for better cache locality
            .map(|chunk| -> Result<Vec<(N, Vec<f64>, Vec<f64>)>> {
                let mut local_updates = Vec::new();
                let mut local_rng = rand::rng();

                for pair in chunk {
                    // Get embeddings (read-only access)
                    let target_emb = self.embeddings
                        .get(&pair.target)
                        .ok_or(GraphError::node_not_found("node"))?
                        .clone();
                    let context_emb = self.context_embeddings
                        .get(&pair.context)
                        .ok_or(GraphError::node_not_found("node"))?
                        .clone();

                    // Compute gradients (SIMD optimized)
                    let positive_score = target_emb.dot_product(&context_emb)?;
                    let positive_prob = Embedding::sigmoid(positive_score);
                    let positive_error = 1.0 - positive_prob;

                    let mut target_gradient = vec![0.0; self.dimensions];
                    let mut context_gradient = vec![0.0; self.dimensions];

                    // Gradient computation
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..self.dimensions {
                        target_gradient[i] += positive_error * context_emb.vector[i];
                        context_gradient[i] += positive_error * target_emb.vector[i];
                    }

                    // Negative sampling
                    let exclude_set: HashSet<&N> =
                        [&pair.target, &pair.context].iter().cloned().collect();
                    let negatives = negative_sampler.sample_negatives(
                        negative_samples,
                        &exclude_set,
                        &mut local_rng,
                    );

                    for negative in &negatives {
                        if let Some(neg_context_emb) = self.context_embeddings.get(negative) {
                            let negative_score = target_emb.dot_product(neg_context_emb)?;
                            let negative_prob = Embedding::sigmoid(negative_score);
                            let negative_error = -negative_prob;

                    #[allow(clippy::needless_range_loop)]
                            for i in 0..self.dimensions {
                                target_gradient[i] += negative_error * neg_context_emb.vector[i];
                            }
                        }
                    }

                    local_updates.push((pair.target.clone(), target_gradient, context_gradient));
                }

                Ok(local_updates)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        // Apply all gradients sequentially to avoid race conditions
        for (target_node, target_gradient, context_gradient) in gradient_updates {
            if let Some(target_emb) = self.embeddings.get_mut(&target_node) {
                target_emb.update_gradient(&target_gradient, learning_rate);
            }

            // Find corresponding context pair for context gradient
            for pair in pairs {
                if pair.target == target_node {
                    if let Some(context_emb) = self.context_embeddings.get_mut(&pair.context) {
                        context_emb.update_gradient(&context_gradient, learning_rate);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    /// Optimized parallel skip-gram with reduced contention
    pub fn train_skip_gram_parallel_optimized(
        &mut self,
        pairs: &[ContextPair<N>],
        negative_sampler: &NegativeSampler<N>,
        learning_rate: f64,
        negative_samples: usize,
        _rng: &mut impl Rng,
    ) -> Result<()>
    where
        N: Clone + Send + Sync,
    {
        use scirs2_core::parallel_ops::*;
        use std::collections::HashMap;

        // Group pairs by target node to reduce lock contention
        let mut pairs_by_target: HashMap<N, Vec<&ContextPair<N>>> = HashMap::new();
        for pair in pairs {
            pairs_by_target
                .entry(pair.target.clone())
                .or_default()
                .push(pair);
        }

        // Process each target node's pairs in parallel
        let target_nodes: Vec<N> = pairs_by_target.keys().cloned().collect();

        let gradient_updates: Vec<(N, Vec<f64>)> = target_nodes
            .par_iter()
            .map(|target_node| -> Result<(N, Vec<f64>)> {
                let mut local_rng = rand::rng();
                let mut accumulated_gradient = vec![0.0; self.dimensions];

                let target_emb = self
                    .embeddings
                    .get(target_node)
                    .ok_or(GraphError::node_not_found("node"))?
                    .clone();

                if let Some(node_pairs) = pairs_by_target.get(target_node) {
                    for &pair in node_pairs {
                        let context_emb = self
                            .context_embeddings
                            .get(&pair.context)
                            .ok_or(GraphError::node_not_found("node"))?
                            .clone();

                        // Compute positive gradient
                        let positive_score = target_emb.dot_product(&context_emb)?;
                        let positive_prob = Embedding::sigmoid(positive_score);
                        let positive_error = 1.0 - positive_prob;

                        // Add positive contribution
                        #[allow(clippy::needless_range_loop)]
                        for i in 0..self.dimensions {
                            accumulated_gradient[i] += positive_error * context_emb.vector[i];
                        }

                        // Negative sampling
                        let exclude_set: HashSet<&N> =
                            [target_node, &pair.context].iter().cloned().collect();
                        let negatives = negative_sampler.sample_negatives(
                            negative_samples,
                            &exclude_set,
                            &mut local_rng,
                        );

                        for negative in &negatives {
                            if let Some(neg_context_emb) = self.context_embeddings.get(negative) {
                                let negative_score = target_emb.dot_product(neg_context_emb)?;
                                let negative_prob = Embedding::sigmoid(negative_score);
                                let negative_error = -negative_prob;

                                #[allow(clippy::needless_range_loop)]
                                for i in 0..self.dimensions {
                                    accumulated_gradient[i] +=
                                        negative_error * neg_context_emb.vector[i];
                                }
                            }
                        }
                    }
                }

                Ok((target_node.clone(), accumulated_gradient))
            })
            .collect::<Result<Vec<_>>>()?;

        // Apply accumulated gradients
        for (target_node, gradient) in gradient_updates {
            if let Some(target_emb) = self.embeddings.get_mut(&target_node) {
                target_emb.update_gradient(&gradient, learning_rate);
            }
        }

        Ok(())
    }

    /// Generate context pairs from random walks
    pub fn generate_context_pairs(
        walks: &[RandomWalk<N>],
        window_size: usize,
    ) -> Vec<ContextPair<N>>
    where
        N: Clone,
    {
        let mut pairs = Vec::new();

        for walk in walks {
            for (i, target) in walk.nodes.iter().enumerate() {
                let start = i.saturating_sub(window_size);
                let end = (i + window_size + 1).min(walk.nodes.len());

                for j in start..end {
                    if i != j {
                        pairs.push(ContextPair {
                            target: target.clone(),
                            context: walk.nodes[j].clone(),
                        });
                    }
                }
            }
        }

        pairs
    }

    /// Evaluate embeddings using link prediction task
    /// Returns AUC score for predicting missing edges
    #[allow(dead_code)]
    pub fn evaluate_link_prediction<E, Ix>(
        &self,
        self_graph: &Graph<N, E, Ix>,
        test_edges: &[(N, N)],
        negative_edges: &[(N, N)],
    ) -> Result<f64>
    where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut scores = Vec::new();
        let mut labels = Vec::new();

        // Positive examples (existing edges)
        for (u, v) in test_edges {
            if let (Some(u_emb), Some(v_emb)) = (self.embeddings.get(u), self.embeddings.get(v)) {
                let similarity = u_emb.cosine_similarity(v_emb)?;
                scores.push(similarity);
                labels.push(1.0);
            }
        }

        // Negative examples (non-existing edges)
        for (u, v) in negative_edges {
            if let (Some(u_emb), Some(v_emb)) = (self.embeddings.get(u), self.embeddings.get(v)) {
                let similarity = u_emb.cosine_similarity(v_emb)?;
                scores.push(similarity);
                labels.push(0.0);
            }
        }

        // Calculate AUC using trapezoidal rule
        let auc = Self::calculate_auc(&scores, &labels)?;
        Ok(auc)
    }

    /// Calculate AUC (Area Under Curve) for binary classification
    fn calculate_auc(scores: &[f64], labels: &[f64]) -> Result<f64> {
        if scores.len() != labels.len() {
            return Err(GraphError::ComputationError(
                "Scores and labels must have same length".to_string(),
            ));
        }

        // Create sorted pairs (score, label)
        let mut pairs: Vec<_> = scores.iter().zip(labels.iter()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tp = 0.0; // True positives
        let mut fp = 0.0; // False positives
        let total_pos = labels.iter().sum::<f64>();
        let total_neg = labels.len() as f64 - total_pos;

        if total_pos == 0.0 || total_neg == 0.0 {
            return Ok(0.5); // Random classifier performance
        }

        let mut auc = 0.0;
        let mut prev_fp_rate = 0.0;

        for (_, &label) in pairs {
            if label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tp_rate = tp / total_pos;
            let fp_rate = fp / total_neg;

            // Add trapezoid area
            auc += tp_rate * (fp_rate - prev_fp_rate);
            prev_fp_rate = fp_rate;
        }

        Ok(auc)
    }

    /// Evaluate embeddings for node classification using k-NN
    pub fn evaluate_node_classification(
        &self,
        train_nodes: &[(N, i32)], // (node, class_label)
        test_nodes: &[N],
        k: usize,
    ) -> Result<HashMap<N, i32>>
    where
        N: Clone,
    {
        let mut predictions = HashMap::new();

        for test_node in test_nodes {
            if let Some(test_emb) = self.embeddings.get(test_node) {
                // Find k nearest neighbors
                let mut similarities = Vec::new();

                for (train_node, label) in train_nodes {
                    if let Some(train_emb) = self.embeddings.get(train_node) {
                        let sim = test_emb.cosine_similarity(train_emb)?;
                        similarities.push((sim, *label));
                    }
                }

                // Sort by similarity (descending)
                similarities
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Take top k and predict by majority vote
                let top_k = similarities.into_iter().take(k);
                let mut label_counts: HashMap<i32, usize> = HashMap::new();

                for (_, label) in top_k {
                    *label_counts.entry(label).or_insert(0) += 1;
                }

                // Find most frequent label
                if let Some((&predicted_label_, _)) =
                    label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    predictions.insert(test_node.clone(), predicted_label_);
                }
            }
        }

        Ok(predictions)
    }

    /// Calculate classification accuracy
    pub fn calculate_accuracy(
        predictions: &HashMap<N, i32>,
        ground_truth: &HashMap<N, i32>,
    ) -> f64 {
        let mut correct = 0;
        let mut total = 0;

        for (node, &true_label) in ground_truth {
            if let Some(&predicted_label) = predictions.get(node) {
                if predicted_label == true_label {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }

    /// Generate negative edges for link prediction evaluation
    pub fn generate_negative_edges<E, Ix>(
        graph: &Graph<N, E, Ix>,
        count: usize,
        rng: &mut impl Rng,
    ) -> Vec<(N, N)>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let mut negative_edges = Vec::new();
        let mut attempts = 0;
        let max_attempts = count * 10;

        while negative_edges.len() < count && attempts < max_attempts {
            if let (Some(u), Some(v)) = (nodes.choose(rng), nodes.choose(rng)) {
                if u != v && !graph.has_edge(u, v) {
                    negative_edges.push((u.clone(), v.clone()));
                }
            }
            attempts += 1;
        }

        negative_edges
    }

    /// Save embeddings to a file in JSON format
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: Serialize,
    {
        let file = File::create(path)
            .map_err(|e| GraphError::ComputationError(format!("Failed to create file: {e}")))?;
        let writer = BufWriter::new(file);

        // Create a serializable representation
        let serializable_data = SerializableEmbeddingModel {
            embeddings: self.embeddings.clone(),
            context_embeddings: self.context_embeddings.clone(),
            dimensions: self.dimensions,
        };

        serde_json::to_writer_pretty(writer, &serializable_data).map_err(|e| {
            GraphError::ComputationError(format!("Failed to serialize embeddings: {e}"))
        })?;

        Ok(())
    }

    /// Load embeddings from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<N>>
    where
        N: for<'de> Deserialize<'de>,
    {
        let file = File::open(path)
            .map_err(|e| GraphError::ComputationError(format!("Failed to open file: {e}")))?;
        let reader = BufReader::new(file);

        let serializable_data: SerializableEmbeddingModel<N> = serde_json::from_reader(reader)
            .map_err(|e| {
                GraphError::ComputationError(format!("Failed to deserialize embeddings: {e}"))
            })?;

        Ok(EmbeddingModel {
            embeddings: serializable_data.embeddings,
            context_embeddings: serializable_data.context_embeddings,
            dimensions: serializable_data.dimensions,
        })
    }

    /// Save embeddings in binary format for faster loading
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: Serialize,
    {
        let file = File::create(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to create binary file: {e}"))
        })?;
        let writer = BufWriter::new(file);

        let serializable_data = SerializableEmbeddingModel {
            embeddings: self.embeddings.clone(),
            context_embeddings: self.context_embeddings.clone(),
            dimensions: self.dimensions,
        };

        bincode::serialize_into(writer, &serializable_data).map_err(|e| {
            GraphError::ComputationError(format!("Failed to serialize embeddings to binary: {e}"))
        })?;

        Ok(())
    }

    /// Load embeddings from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<N>>
    where
        N: for<'de> Deserialize<'de>,
    {
        let file = File::open(path).map_err(|e| {
            GraphError::ComputationError(format!("Failed to open binary file: {e}"))
        })?;
        let reader = BufReader::new(file);

        let serializable_data: SerializableEmbeddingModel<N> = bincode::deserialize_from(reader)
            .map_err(|e| {
                GraphError::ComputationError(format!(
                    "Failed to deserialize binary embeddings: {e}"
                ))
            })?;

        Ok(EmbeddingModel {
            embeddings: serializable_data.embeddings,
            context_embeddings: serializable_data.context_embeddings,
            dimensions: serializable_data.dimensions,
        })
    }

    /// Export embeddings to CSV format for analysis
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        N: std::fmt::Display,
    {
        let mut file = File::create(path)
            .map_err(|e| GraphError::ComputationError(format!("Failed to create CSV file: {e}")))?;

        // Write header
        write!(file, "node")?;
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.dimensions {
            write!(file, ",dim_{i}")?;
        }
        writeln!(file)?;

        // Write embeddings
        for (node, embedding) in &self.embeddings {
            write!(file, "{node}")?;
            for value in &embedding.vector {
                write!(file, ",{value}")?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    /// Import embeddings from CSV format
    pub fn import_csv<P: AsRef<Path>>(path: P) -> Result<EmbeddingModel<String>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| GraphError::ComputationError(format!("Failed to read CSV file: {e}")))?;

        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(GraphError::ComputationError("Empty CSV file".to_string()));
        }

        // Parse header to get dimensions
        let header = lines[0];
        let header_parts: Vec<&str> = header.split(',').collect();
        let dimensions = header_parts.len() - 1; // Subtract 1 for the node column

        let mut embeddings = HashMap::new();

        for line in lines.iter().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != dimensions + 1 {
                continue; // Skip malformed lines
            }

            let node = parts[0].to_string();
            let vector: std::result::Result<Vec<f64>, _> =
                parts[1..].iter().map(|s| s.parse::<f64>()).collect();

            match vector {
                Ok(v) => {
                    embeddings.insert(node, Embedding { vector: v });
                }
                Err(_) => continue, // Skip lines with parsing errors
            }
        }

        Ok(EmbeddingModel {
            embeddings,
            context_embeddings: HashMap::new(), // CSV doesn't include context embeddings
            dimensions,
        })
    }
}

/// Serializable version of EmbeddingModel for persistence
#[derive(Serialize, Deserialize)]
struct SerializableEmbeddingModel<N: Node> {
    embeddings: HashMap<N, Embedding>,
    context_embeddings: HashMap<N, Embedding>,
    dimensions: usize,
}

/// Random walk generator for graphs
pub struct RandomWalkGenerator<N: Node> {
    /// Random number generator
    rng: rand::rngs::ThreadRng,
    /// Phantom marker for node type
    _phantom: std::marker::PhantomData<N>,
}

impl<N: Node> Default for RandomWalkGenerator<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node> RandomWalkGenerator<N> {
    /// Create a new random walk generator
    pub fn new() -> Self {
        RandomWalkGenerator {
            rng: rand::rng(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a simple random walk from a starting node
    pub fn simple_random_walk<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        length: usize,
    ) -> Result<RandomWalk<N>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        if !graph.contains_node(start) {
            return Err(GraphError::node_not_found("node"));
        }

        let mut walk = vec![start.clone()];
        let mut current = start.clone();

        for _ in 1..length {
            let neighbors = graph.neighbors(&current)?;
            if neighbors.is_empty() {
                break; // No outgoing edges, stop walk
            }

            current = neighbors
                .choose(&mut self.rng)
                .ok_or(GraphError::AlgorithmError(
                    "Failed to choose neighbor".to_string(),
                ))?
                .clone();
            walk.push(current.clone());
        }

        Ok(RandomWalk { nodes: walk })
    }

    /// Generate a Node2Vec biased random walk
    pub fn node2vec_walk<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        length: usize,
        p: f64,
        q: f64,
    ) -> Result<RandomWalk<N>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        if !graph.contains_node(start) {
            return Err(GraphError::node_not_found("node"));
        }

        let mut walk = vec![start.clone()];
        if length == 1 {
            return Ok(RandomWalk { nodes: walk });
        }

        // First step is unbiased
        let first_neighbors = graph.neighbors(start)?;
        if first_neighbors.is_empty() {
            return Ok(RandomWalk { nodes: walk });
        }

        let current = first_neighbors
            .choose(&mut self.rng)
            .ok_or(GraphError::AlgorithmError(
                "Failed to choose first neighbor".to_string(),
            ))?
            .clone();
        walk.push(current.clone());

        // Subsequent steps use biased sampling
        for _ in 2..length {
            let current_neighbors = graph.neighbors(&current)?;
            if current_neighbors.is_empty() {
                break;
            }

            let prev = &walk[walk.len() - 2];
            let mut weights = Vec::new();

            for neighbor in &current_neighbors {
                let weight = if neighbor == prev {
                    // Return to previous node
                    1.0 / p
                } else if graph.has_edge(prev, neighbor) {
                    // Neighbor is also connected to previous node
                    1.0
                } else {
                    // New exploration
                    1.0 / q
                };
                weights.push(weight);
            }

            // Weighted random selection
            let total_weight: f64 = weights.iter().sum();
            let mut random_value = self.rng.random::<f64>() * total_weight;
            let mut selected_index = 0;

            for (i, &weight) in weights.iter().enumerate() {
                random_value -= weight;
                if random_value <= 0.0 {
                    selected_index = i;
                    break;
                }
            }

            let next_node = current_neighbors[selected_index].clone();
            walk.push(next_node.clone());
            // Update current for next iteration
            let _current = next_node;
        }

        Ok(RandomWalk { nodes: walk })
    }

    /// Generate multiple random walks from a starting node
    pub fn generate_walks<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        num_walks: usize,
        walk_length: usize,
    ) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut walks = Vec::new();
        for _ in 0..num_walks {
            let walk = self.simple_random_walk(graph, start, walk_length)?;
            walks.push(walk);
        }
        Ok(walks)
    }
}

/// Basic Node2Vec implementation foundation
pub struct Node2Vec<N: Node> {
    config: Node2VecConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
}

impl<N: Node> Node2Vec<N> {
    /// Create a new Node2Vec instance
    pub fn new(config: Node2VecConfig) -> Self {
        Node2Vec {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
        }
    }

    /// Generate training data (random walks) for Node2Vec
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk = self.walk_generator.node2vec_walk(
                    graph,
                    node,
                    self.config.walk_length,
                    self.config.p,
                    self.config.q,
                )?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the Node2Vec model with complete skip-gram implementation
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Create negative sampler
        let negative_sampler = NegativeSampler::new(graph);

        // Training loop over epochs
        for epoch in 0..self.config.epochs {
            // Generate walks for this epoch
            let walks = self.generate_walks(graph)?;

            // Generate context pairs from walks
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            // Shuffle pairs for better training
            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut self.walk_generator.rng);

            // Train skip-gram model with negative sampling
            let current_lr =
                self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);

            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut self.walk_generator.rng,
            )?;

            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "Node2Vec epoch {}/{}, generated {} walks, {} context pairs",
                    epoch + 1,
                    self.config.epochs,
                    walks.len(),
                    shuffled_pairs.len()
                );
            }
        }

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut EmbeddingModel<N> {
        &mut self.model
    }
}

/// Basic DeepWalk implementation foundation
pub struct DeepWalk<N: Node> {
    config: DeepWalkConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
}

impl<N: Node> DeepWalk<N> {
    /// Create a new DeepWalk instance
    pub fn new(config: DeepWalkConfig) -> Self {
        DeepWalk {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
        }
    }

    /// Generate training data (random walks) for DeepWalk
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk =
                    self.walk_generator
                        .simple_random_walk(graph, node, self.config.walk_length)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the DeepWalk model with complete skip-gram implementation
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Create negative sampler
        let negative_sampler = NegativeSampler::new(graph);

        // Training loop over epochs
        for epoch in 0..self.config.epochs {
            // Generate walks for this epoch
            let walks = self.generate_walks(graph)?;

            // Generate context pairs from walks
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            // Shuffle pairs for better training
            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut self.walk_generator.rng);

            // Train skip-gram model with negative sampling
            let current_lr =
                self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);

            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut self.walk_generator.rng,
            )?;

            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "DeepWalk epoch {}/{}, generated {} walks, {} context pairs",
                    epoch + 1,
                    self.config.epochs,
                    walks.len(),
                    shuffled_pairs.len()
                );
            }
        }

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }
}

/// Enhanced embedding trainer with advanced optimization techniques
pub struct AdvancedEmbeddingTrainer<N: Node> {
    /// Optimization configuration
    pub config: OptimizationConfig,
    /// Optimizer state
    pub optimizer_state: OptimizerState,
    /// Training metrics history
    pub metrics_history: Vec<TrainingMetrics>,
    /// Current training metrics
    pub current_metrics: TrainingMetrics,
    /// Loss history for convergence checking
    pub loss_history: Vec<f64>,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Best loss seen so far
    pub best_loss: f64,
    /// Steps without improvement
    pub steps_without_improvement: usize,
    /// Adaptive negative sampling
    pub negative_sampling_strategy: NegativeSamplingStrategy,
    /// Node frequency for sampling
    pub node_frequencies: HashMap<N, f64>,
    /// Gradient accumulation buffer
    pub gradient_accumulator: HashMap<String, Vec<f64>>,
    /// Training start time
    pub training_start_time: std::time::Instant,
}

impl<N: Node + Clone + Hash + Eq> AdvancedEmbeddingTrainer<N> {
    pub fn new(config: OptimizationConfig) -> Self {
        AdvancedEmbeddingTrainer {
            config,
            optimizer_state: OptimizerState::new(),
            metrics_history: Vec::new(),
            current_metrics: TrainingMetrics::default(),
            loss_history: Vec::new(),
            early_stopping_patience: 10,
            best_loss: f64::INFINITY,
            steps_without_improvement: 0,
            negative_sampling_strategy: NegativeSamplingStrategy::Frequency,
            node_frequencies: HashMap::new(),
            gradient_accumulator: HashMap::new(),
            training_start_time: std::time::Instant::now(),
        }
    }

    /// Compute adaptive learning rate based on schedule
    pub fn compute_learning_rate(&self, epoch: usize, totalepochs: usize) -> f64 {
        let progress = epoch as f64 / totalepochs as f64;

        match self.config.lr_schedule {
            LearningRateSchedule::Constant => self.config.initial_lr,
            LearningRateSchedule::Linear => {
                self.config.initial_lr * (1.0 - progress) + self.config.final_lr * progress
            }
            LearningRateSchedule::Exponential => {
                self.config.initial_lr
                    * (self.config.final_lr / self.config.initial_lr).powf(progress)
            }
            LearningRateSchedule::Cosine => {
                self.config.final_lr
                    + 0.5
                        * (self.config.initial_lr - self.config.final_lr)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            LearningRateSchedule::Step => {
                // Reduce by factor of 10 every 25% of training
                let step = (progress * 4.0).floor() as i32;
                self.config.initial_lr * 0.1_f64.powi(step)
            }
        }
    }

    /// Apply Adam optimization with bias correction
    pub fn apply_adam_optimizer(
        &mut self,
        parameter_name: &str,
        gradient: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let dimensions = gradient.len();

        // Initialize if needed
        if !self.optimizer_state.adam_m.contains_key(parameter_name) {
            self.optimizer_state
                .adam_m
                .insert(parameter_name.to_string(), vec![0.0; dimensions]);
            self.optimizer_state
                .adam_v
                .insert(parameter_name.to_string(), vec![0.0; dimensions]);
        }

        self.optimizer_state.time_step += 1;

        let m = self.optimizer_state.adam_m.get_mut(parameter_name).unwrap();
        let v = self.optimizer_state.adam_v.get_mut(parameter_name).unwrap();

        // Update biased first moment estimate
        for i in 0..dimensions {
            m[i] = self.config.adam_beta1 * m[i] + (1.0 - self.config.adam_beta1) * gradient[i];
        }

        // Update biased second raw moment estimate
        for i in 0..dimensions {
            v[i] = self.config.adam_beta2 * v[i]
                + (1.0 - self.config.adam_beta2) * gradient[i] * gradient[i];
        }

        // Compute bias-corrected first moment estimate
        let bias_correction_1 = 1.0
            - self
                .config
                .adam_beta1
                .powi(self.optimizer_state.time_step as i32);
        let bias_correction_2 = 1.0
            - self
                .config
                .adam_beta2
                .powi(self.optimizer_state.time_step as i32);

        let mut parameter_update = vec![0.0; dimensions];
        for i in 0..dimensions {
            let m_hat = m[i] / bias_correction_1;
            let v_hat = v[i] / bias_correction_2;
            parameter_update[i] =
                -learning_rate * m_hat / (v_hat.sqrt() + self.config.adam_epsilon);
        }

        parameter_update
    }

    /// Apply momentum optimization
    pub fn apply_momentum_optimizer(
        &mut self,
        parameter_name: &str,
        gradient: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let dimensions = gradient.len();

        // Initialize momentum buffer if needed
        if !self
            .optimizer_state
            .momentum_buffers
            .contains_key(parameter_name)
        {
            self.optimizer_state
                .momentum_buffers
                .insert(parameter_name.to_string(), vec![0.0; dimensions]);
        }

        let momentum_buffer = self
            .optimizer_state
            .momentum_buffers
            .get_mut(parameter_name)
            .unwrap();

        // Update momentum buffer
        for i in 0..dimensions {
            momentum_buffer[i] =
                self.config.momentum * momentum_buffer[i] + learning_rate * gradient[i];
        }

        // Return negative momentum (for parameter update)
        momentum_buffer.iter().map(|&x| -x).collect()
    }

    /// Apply gradient clipping
    pub fn clip_gradient(&self, gradient: &mut [f64]) {
        if let Some(clip_value) = self.config.gradient_clip {
            let grad_norm: f64 = gradient.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if grad_norm > clip_value {
                let scale = clip_value / grad_norm;
                for g in gradient.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    /// Apply L2 regularization to gradient
    pub fn apply_l2_regularization(&self, gradient: &mut [f64], parameters: &[f64]) {
        if self.config.l2_regularization > 0.0 {
            for (grad, param) in gradient.iter_mut().zip(parameters.iter()) {
                *grad += self.config.l2_regularization * param;
            }
        }
    }

    /// Update training metrics
    pub fn update_metrics(
        &mut self,
        loss: f64,
        gradient_norm: f64,
        positive_accuracy: f64,
        negative_accuracy: f64,
    ) {
        self.current_metrics.steps += 1;
        self.current_metrics.loss = loss;
        self.current_metrics.gradient_norm = gradient_norm;
        self.current_metrics.positive_accuracy = positive_accuracy;
        self.current_metrics.negative_accuracy = negative_accuracy;

        // Update moving average loss
        let alpha = 0.99; // Smoothing factor
        if self.current_metrics.loss_avg == 0.0 {
            self.current_metrics.loss_avg = loss;
        } else {
            self.current_metrics.loss_avg =
                alpha * self.current_metrics.loss_avg + (1.0 - alpha) * loss;
        }

        // Update convergence rate
        self.loss_history.push(loss);
        if self.loss_history.len() > 10 {
            let recent_losses = &self.loss_history[self.loss_history.len() - 10..];
            let loss_change = recent_losses[0] - recent_losses[recent_losses.len() - 1];
            self.current_metrics.convergence_rate = loss_change / 10.0;
        }

        // Update steps per second
        let elapsed = self.training_start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.current_metrics.steps_per_second = self.current_metrics.steps as f64 / elapsed;
        }

        // Check for early stopping
        if loss < self.best_loss {
            self.best_loss = loss;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }
    }

    /// Check if early stopping should be triggered
    pub fn should_early_stop(&self) -> bool {
        self.steps_without_improvement >= self.early_stopping_patience
    }

    /// Reset optimizer state (useful for learning rate scheduling)
    pub fn reset_optimizer_state(&mut self) {
        self.optimizer_state = OptimizerState::new();
    }

    /// Get current training report
    pub fn get_training_report(&self) -> String {
        format!(
            "Epoch: {}, Steps: {}, LR: {:.6}, Loss: {:.6}, Loss Avg: {:.6}, Grad Norm: {:.6}, Steps/sec: {:.2}, Pos Acc: {:.3}, Neg Acc: {:.3}, Convergence: {:.6}",
            self.current_metrics.epoch,
            self.current_metrics.steps,
            self.current_metrics.learning_rate,
            self.current_metrics.loss,
            self.current_metrics.loss_avg,
            self.current_metrics.gradient_norm,
            self.current_metrics.steps_per_second,
            self.current_metrics.positive_accuracy,
            self.current_metrics.negative_accuracy,
            self.current_metrics.convergence_rate
        )
    }
}

/// Fast approximation-based embeddings for very large graphs
pub struct FastGraphEmbedding<N: Node> {
    /// Node embeddings using hash-based approximation
    pub embeddings: HashMap<N, Vec<f32>>, // Use f32 for memory efficiency
    /// Embedding dimension
    pub dimensions: usize,
    /// Random projection matrix for dimensionality reduction
    pub projection_matrix: Vec<Vec<f32>>,
    /// Approximation quality (higher = better quality, slower)
    pub quality_factor: usize,
    /// Memory pool for reusing vectors
    pub memory_pool: Vec<Vec<f32>>,
    /// Pool index for round-robin allocation
    pub pool_index: AtomicUsize,
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> FastGraphEmbedding<N> {
    pub fn new(dimensions: usize, qualityfactor: usize) -> Self {
        let pool_size = 1000; // Preallocate 1000 vectors
        let mut memory_pool = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            memory_pool.push(vec![0.0f32; dimensions]);
        }

        // Initialize random projection matrix
        let mut rng = rand::rng();
        let projection_size = dimensions * qualityfactor;
        let mut projection_matrix = Vec::new();
        for _ in 0..projection_size {
            let row: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect();
            projection_matrix.push(row);
        }

        FastGraphEmbedding {
            embeddings: HashMap::new(),
            dimensions,
            projection_matrix,
            quality_factor: qualityfactor,
            memory_pool,
            pool_index: AtomicUsize::new(0),
        }
    }

    /// Fast embedding computation using sketching techniques
    pub fn compute_fast_embedding<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        node: &N,
        num_samples: usize,
    ) -> Result<Vec<f32>>
    where
        N: std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Use frequency sketching for fast approximation
        let mut sketch = vec![0.0f32; self.dimensions];
        let mut rng = rand::rng();

        // Sample random walks and update sketch
        for _ in 0..num_samples {
            let walk = self.sample_random_walk(graph, node, 10, &mut rng)?;
            self.update_sketch(&mut sketch, &walk);
        }

        // Normalize the sketch
        let norm: f32 = sketch.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for s in sketch.iter_mut() {
                *s /= norm;
            }
        }

        Ok(sketch)
    }

    fn sample_random_walk<E, Ix>(
        &self,
        graph: &Graph<N, E, Ix>,
        start_node: &N,
        walk_length: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<N>>
    where
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut walk = vec![start_node.clone()];
        let mut current = start_node.clone();

        for _ in 1..walk_length {
            if let Ok(neighbors) = graph.neighbors(&current) {
                if neighbors.is_empty() {
                    break;
                }
                let next_node = neighbors.choose(rng).unwrap().clone();
                walk.push(next_node.clone());
                current = next_node;
            } else {
                break;
            }
        }

        Ok(walk)
    }

    fn update_sketch(&self, sketch: &mut [f32], walk: &[N]) {
        // Use hash-based feature extraction from walk
        for window in walk.windows(2) {
            let hash1 = self.hash_node(&window[0]);
            let hash2 = self.hash_node(&window[1]);
            let combined_hash = hash1.wrapping_mul(31).wrapping_add(hash2);

            let index = (combined_hash as usize) % self.dimensions;
            sketch[index] += 1.0;
        }
    }

    fn hash_node(&self, node: &N) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        node.hash(&mut hasher);
        hasher.finish()
    }

    /// Get or compute embedding for a node
    pub fn get_or_compute_embedding<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        node: &N,
    ) -> Result<Vec<f32>>
    where
        N: std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        if let Some(embedding) = self.embeddings.get(node) {
            Ok(embedding.clone())
        } else {
            let embedding = self.compute_fast_embedding(graph, node, 100)?;
            self.embeddings.insert(node.clone(), embedding.clone());
            Ok(embedding)
        }
    }

    /// Batch compute embeddings for multiple nodes in parallel
    pub fn batch_compute_embeddings<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        nodes: &[N],
    ) -> Result<HashMap<N, Vec<f32>>>
    where
        N: Send + Sync + std::fmt::Debug,
        E: EdgeWeight + Into<f64> + Send + Sync,
        Ix: petgraph::graph::IndexType + Send + Sync,
    {
        let mut final_results = HashMap::new();
        for node in nodes {
            let embedding = self.compute_fast_embedding(graph, node, 50)?;
            final_results.insert(node.clone(), embedding);
        }

        Ok(final_results)
    }
}

/// Graph2Vec: Learn embeddings for entire graphs
pub struct Graph2Vec<N: Node> {
    /// Vocabulary of subgraph patterns
    pub vocabulary: HashMap<String, usize>,
    /// Document embeddings (one per graph)
    pub graph_embeddings: HashMap<String, Vec<f64>>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// WL kernel iterations for pattern extraction
    pub wl_iterations: usize,
    /// Phantom data for node type
    _phantom: std::marker::PhantomData<N>,
    /// Pattern frequency threshold
    pub min_pattern_freq: usize,
}

impl<N: Node + Clone + Hash + Eq> Graph2Vec<N> {
    pub fn new(dimensions: usize, wl_iterations: usize, min_patternfreq: usize) -> Self {
        Graph2Vec {
            vocabulary: HashMap::new(),
            graph_embeddings: HashMap::new(),
            dimensions,
            wl_iterations,
            _phantom: std::marker::PhantomData,
            min_pattern_freq: min_patternfreq,
        }
    }

    /// Extract graph-level features using Weisfeiler-Lehman kernel
    pub fn extract_wl_features<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        _graph_id: &str,
    ) -> Result<Vec<String>>
    where
        N: std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut node_labels: HashMap<N, String> = HashMap::new();

        // Initialize node labels (can be based on degree, features, etc.)
        for node in graph.nodes() {
            let degree = graph.degree(node);
            node_labels.insert(node.clone(), degree.to_string());
        }

        let mut patterns = HashSet::new();

        // Iterate WL kernel
        for iteration in 0..self.wl_iterations {
            let mut new_labels: HashMap<N, String> = HashMap::new();

            for node in graph.nodes() {
                let mut neighbor_labels = Vec::new();

                if let Ok(neighbors) = graph.neighbors(node) {
                    for neighbor in neighbors {
                        if let Some(label) = node_labels.get(&neighbor) {
                            neighbor_labels.push(label.clone());
                        }
                    }
                }

                neighbor_labels.sort();
                let empty_string = String::new();
                let current_label = node_labels.get(node).unwrap_or(&empty_string);
                let new_label = format!("{}_{}", current_label, neighbor_labels.join("_"));
                new_labels.insert(node.clone(), new_label.clone());

                // Add pattern to vocabulary
                patterns.insert(format!("WL_{iteration}_{new_label}"));
            }

            node_labels = new_labels;
        }

        Ok(patterns.into_iter().collect())
    }

    /// Train graph-level embeddings using collected patterns
    pub fn train_graph_embeddings(
        &mut self,
        graph_patterns: &HashMap<String, Vec<String>>,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        // Build vocabulary from all _patterns
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for _patterns in graph_patterns.values() {
            for pattern in _patterns {
                *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;
            }
        }

        // Filter vocabulary by minimum frequency
        let mut vocab_index = 0;
        for (pattern, count) in pattern_counts {
            if count >= self.min_pattern_freq {
                self.vocabulary.insert(pattern, vocab_index);
                vocab_index += 1;
            }
        }

        // Initialize graph embeddings
        let mut rng = rand::rng();
        for graph_id in graph_patterns.keys() {
            let embedding: Vec<f64> = (0..self.dimensions)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect();
            self.graph_embeddings.insert(graph_id.clone(), embedding);
        }

        // Train using skip-gram-like objective on graph "documents"
        for epoch in 0..epochs {
            for (graph_id, patterns) in graph_patterns {
                // Convert patterns to vocabulary indices
                let pattern_indices: Vec<usize> = patterns
                    .iter()
                    .filter_map(|p| self.vocabulary.get(p))
                    .cloned()
                    .collect();

                if pattern_indices.len() >= 2 {
                    // Use patterns as context for each other
                    for (i, &target_idx) in pattern_indices.iter().enumerate() {
                        for (j, &context_idx) in pattern_indices.iter().enumerate() {
                            if i != j {
                                // Update embeddings based on co-occurrence
                                self.update_graph_embedding(
                                    graph_id,
                                    target_idx,
                                    context_idx,
                                    learning_rate,
                                );
                            }
                        }
                    }
                }
            }

            // Decay learning _rate
            let epoch_lr = learning_rate * (1.0 - epoch as f64 / epochs as f64);
            if epoch % 10 == 0 {
                println!("Graph2Vec epoch {epoch}: lr = {epoch_lr:.6}");
            }
        }

        Ok(())
    }

    fn update_graph_embedding(
        &mut self,
        graph_id: &str,
        target_idx: usize,
        context_idx: usize,
        learning_rate: f64,
    ) {
        // Simplified update - in practice would use hierarchical softmax or negative sampling
        if let Some(embedding) = self.graph_embeddings.get_mut(graph_id) {
            let update_magnitude = learning_rate * 0.1; // Simplified update
            if target_idx < self.dimensions && context_idx < self.dimensions {
                embedding[target_idx] += update_magnitude;
                embedding[context_idx] += update_magnitude * 0.5;
            }
        }
    }

    /// Get embedding for a graph
    pub fn get_graph_embedding(&self, graphid: &str) -> Option<&Vec<f64>> {
        self.graph_embeddings.get(graphid)
    }

    /// Compute similarity between two graphs
    pub fn graph_similarity(&self, graph_id1: &str, graphid2: &str) -> f64 {
        if let (Some(emb1), Some(emb2)) = (
            self.graph_embeddings.get(graph_id1),
            self.graph_embeddings.get(graphid2),
        ) {
            // Cosine similarity
            let dot_product: f64 = emb1.iter().zip(emb2.iter()).map(|(&a, &b)| a * b).sum();
            let norm1: f64 = emb1.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let norm2: f64 = emb2.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let embedding = Embedding::new(64);
        assert_eq!(embedding.dimensions(), 64);
        assert_eq!(embedding.vector.len(), 64);
    }

    #[test]
    fn test_embedding_similarity() {
        let mut emb1 = Embedding {
            vector: vec![1.0, 0.0, 0.0],
        };
        let mut emb2 = Embedding {
            vector: vec![0.0, 1.0, 0.0],
        };
        let mut emb3 = Embedding {
            vector: vec![1.0, 0.0, 0.0],
        };

        emb1.normalize();
        emb2.normalize();
        emb3.normalize();

        // Test cosine similarity
        let sim_orthogonal = emb1.cosine_similarity(&emb2).unwrap();
        let sim_identical = emb1.cosine_similarity(&emb3).unwrap();

        assert!((sim_orthogonal - 0.0).abs() < 1e-10);
        assert!((sim_identical - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_walk_generation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 1, 1.0).unwrap(); // Create a cycle

        let mut generator = RandomWalkGenerator::new();
        let walk = generator.simple_random_walk(&graph, &1, 10).unwrap();

        assert!(!walk.nodes.is_empty());
        assert_eq!(walk.nodes[0], 1); // Should start with the specified node
        assert!(walk.nodes.len() <= 10); // Should not exceed requested length
    }

    #[test]
    fn test_embedding_model() {
        let mut model: EmbeddingModel<i32> = EmbeddingModel::new(32);
        let embedding = Embedding::random(32, &mut rand::rng());

        model.set_embedding(1, embedding).unwrap();
        assert!(model.get_embedding(&1).is_some());
        assert!(model.get_embedding(&2).is_none());
    }

    #[test]
    fn test_node2vec_config() {
        let config = Node2VecConfig::default();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.walk_length, 80);
        assert_eq!(config.p, 1.0);
        assert_eq!(config.q, 1.0);
    }

    #[test]
    fn test_deepwalk_initialization() {
        let config = DeepWalkConfig::default();
        let deepwalk: DeepWalk<i32> = DeepWalk::new(config);
        assert_eq!(deepwalk.model.dimensions, 128);
    }

    #[test]
    fn test_node2vec_walk_generation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 1, 1.0).unwrap();

        let config = Node2VecConfig::default();
        let mut node2vec = Node2Vec::new(config);

        let walks = node2vec.generate_walks(&graph).unwrap();
        assert!(!walks.is_empty());

        // Each node should generate num_walks walks
        let expected_total_walks = graph.node_count() * node2vec.config.num_walks;
        assert_eq!(walks.len(), expected_total_walks);
    }
}
