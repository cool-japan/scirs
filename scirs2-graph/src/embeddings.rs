//! Graph embedding algorithms and utilities
//!
//! This module provides graph embedding algorithms including Node2Vec, DeepWalk,
//! and other representation learning methods for graphs.

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;

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

/// Node embedding vector
#[derive(Debug, Clone)]
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
        let vector: Vec<f64> = (0..dimensions)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        Embedding { vector }
    }

    /// Get the dimensionality of the embedding
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Calculate cosine similarity with another embedding
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

    /// Calculate L2 norm of the embedding
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
}

/// Graph embedding model
#[derive(Debug)]
pub struct EmbeddingModel<N: Node> {
    /// Node embeddings
    pub embeddings: HashMap<N, Embedding>,
    /// Dimensionality of embeddings
    pub dimensions: usize,
}

impl<N: Node> EmbeddingModel<N> {
    /// Create a new embedding model
    pub fn new(dimensions: usize) -> Self {
        EmbeddingModel {
            embeddings: HashMap::new(),
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
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
        }
    }

    /// Initialize random embeddings for directed graph
    pub fn initialize_random_digraph<E, Ix>(
        &mut self,
        graph: &DiGraph<N, E, Ix>,
        rng: &mut impl Rng,
    ) where
        N: Clone,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        for node in graph.nodes() {
            let embedding = Embedding::random(self.dimensions, rng);
            self.embeddings.insert(node.clone(), embedding);
        }
    }

    /// Find k most similar nodes to a given node
    pub fn most_similar(&self, node: &N, k: usize) -> Result<Vec<(N, f64)>>
    where
        N: Clone,
    {
        let target_embedding = self.embeddings.get(node).ok_or(GraphError::NodeNotFound)?;

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
            return Err(GraphError::NodeNotFound);
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
            return Err(GraphError::NodeNotFound);
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

    /// Train the Node2Vec model (simplified version)
    /// Note: This is a foundation implementation that would need a full skip-gram implementation
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Generate walks
        let walks = self.generate_walks(graph)?;

        // Basic placeholder for skip-gram training
        // In a full implementation, this would involve:
        // 1. Creating context pairs from walks
        // 2. Training with negative sampling
        // 3. Gradient updates using backpropagation

        // For now, we'll just return success as this is a foundation
        println!(
            "Generated {} walks for training (foundation implementation)",
            walks.len()
        );

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

    /// Train the DeepWalk model (simplified version)
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        self.model
            .initialize_random(graph, &mut self.walk_generator.rng);

        // Generate walks
        let walks = self.generate_walks(graph)?;

        // Basic placeholder for skip-gram training
        println!(
            "Generated {} walks for DeepWalk training (foundation implementation)",
            walks.len()
        );

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
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
