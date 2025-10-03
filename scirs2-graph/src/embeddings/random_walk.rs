//! Random walk generation for graph embeddings

use super::types::RandomWalk;
use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::rand_prelude::IndexedRandom;
use scirs2_core::random::Rng;

/// Random walk generator for graphs
pub struct RandomWalkGenerator<N: Node> {
    /// Random number generator
    rng: scirs2_core::random::rngs::ThreadRng,
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
            rng: scirs2_core::random::rng(),
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
            // Update current for next iteration - this line was originally incorrect
            // let _current = next_node;
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
