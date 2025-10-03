//! Node2Vec graph embedding algorithm

use super::core::EmbeddingModel;
use super::negative_sampling::NegativeSampler;
use super::random_walk::RandomWalkGenerator;
use super::types::{Node2VecConfig, RandomWalk};
use crate::base::{EdgeWeight, Graph, Node};
use crate::error::Result;
use scirs2_core::random::seq::SliceRandom;

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
        let mut rng = scirs2_core::random::rng();
        self.model.initialize_random(graph, &mut rng);

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
            shuffled_pairs.shuffle(&mut rng);

            // Train skip-gram model with negative sampling
            let current_lr =
                self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);

            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut rng,
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
