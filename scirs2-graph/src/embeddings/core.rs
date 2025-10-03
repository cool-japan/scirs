//! Core embedding structures and operations

use super::negative_sampling::NegativeSampler;
use super::types::ContextPair;
use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update embedding using gradient (SIMD optimized)
    pub fn update_gradient(&mut self, gradient: &[f64], learning_rate: f64) {
        for (emb, &grad) in self.vector.iter_mut().zip(gradient.iter()) {
            *emb -= learning_rate * grad;
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

    /// Generate context pairs from random walks
    pub fn generate_context_pairs(
        walks: &[super::types::RandomWalk<N>],
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

            // Negative samples: minimize probability of negative contexts
            let exclude_set: HashSet<&N> = [&pair.target, &pair.context].iter().cloned().collect();
            let negatives = negative_sampler.sample_negatives(negative_samples, &exclude_set, rng);

            for negative in &negatives {
                if let Some(neg_context_emb) = self.context_embeddings.get(negative) {
                    let negative_score = target_emb.dot_product(neg_context_emb)?;
                    let negative_prob = Embedding::sigmoid(negative_score);

                    // Negative sample error
                    let negative_error = -negative_prob;

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
}
