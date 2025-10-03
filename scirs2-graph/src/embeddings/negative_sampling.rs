//! Negative sampling strategies for graph embeddings

use super::types::NegativeSamplingStrategy;
use crate::base::{EdgeWeight, Graph, Node};
use scirs2_core::random::Rng;
use std::collections::HashSet;

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
