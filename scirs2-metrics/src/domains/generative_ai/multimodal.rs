//! Multimodal evaluation metrics
//!
//! This module provides evaluation metrics for multimodal models
//! including cross-modal retrieval and multimodal alignment.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::iter::Sum;

use super::results::{CrossModalRetrievalResult, MultimodalAlignmentResult};

/// Multimodal metrics
pub struct MultimodalMetrics<F: Float> {
    /// Number of retrieval candidates
    pub n_retrieval_candidates: usize,
    /// Top-k values for retrieval evaluation
    pub retrieval_k_values: Vec<usize>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for MultimodalMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> MultimodalMetrics<F> {
    /// Create new multimodal metrics
    pub fn new() -> Self {
        Self {
            n_retrieval_candidates: 1000,
            retrieval_k_values: vec![1, 5, 10],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set retrieval parameters
    pub fn with_retrieval(mut self, candidates: usize, kvalues: Vec<usize>) -> Self {
        self.n_retrieval_candidates = candidates;
        self.retrieval_k_values = kvalues;
        self
    }

    /// Compute cross-modal retrieval performance
    pub fn cross_modal_retrieval(
        &self,
        query_embeddings: &Array2<F>,
        candidate_embeddings: &Array2<F>,
        ground_truth_pairs: &[(usize, usize)], // (query_idx, candidate_idx) pairs
    ) -> Result<CrossModalRetrievalResult<F>> {
        if query_embeddings.is_empty() || candidate_embeddings.is_empty() {
            return Err(MetricsError::InvalidInput("Empty embeddings".to_string()));
        }

        if query_embeddings.ncols() != candidate_embeddings.ncols() {
            return Err(MetricsError::InvalidInput(
                "Embedding dimension mismatch".to_string(),
            ));
        }

        let n_queries = query_embeddings.nrows();
        let n_candidates = candidate_embeddings.nrows();

        // Compute similarity matrix
        let mut similarities = Array2::zeros((n_queries, n_candidates));

        for i in 0..n_queries {
            for j in 0..n_candidates {
                let sim = self.cosine_similarity(
                    &query_embeddings.row(i).to_owned(),
                    &candidate_embeddings.row(j).to_owned(),
                )?;
                similarities[[i, j]] = sim;
            }
        }

        // Create ground truth lookup
        let mut gt_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(query_idx, candidate_idx) in ground_truth_pairs {
            gt_map.entry(query_idx).or_default().push(candidate_idx);
        }

        // Compute recall at k for each k value
        let mut recall_at_k = HashMap::new();
        let mut precision_at_k = HashMap::new();

        for &k in &self.retrieval_k_values {
            let mut total_recall = F::zero();
            let mut total_precision = F::zero();
            let mut valid_queries = 0;

            for query_idx in 0..n_queries {
                if let Some(gt_candidates) = gt_map.get(&query_idx) {
                    // Get top-k candidates for this query
                    let mut query_similarities: Vec<(F, usize)> = (0..n_candidates)
                        .map(|j| (similarities[[query_idx, j]], j))
                        .collect();

                    query_similarities
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                    let top_k_candidates: Vec<usize> = query_similarities
                        .iter()
                        .take(k)
                        .map(|(_, idx)| *idx)
                        .collect();

                    // Count hits
                    let hits = top_k_candidates
                        .iter()
                        .filter(|&&candidate| gt_candidates.contains(&candidate))
                        .count();

                    // Compute recall and precision for this query
                    let query_recall =
                        F::from(hits).unwrap() / F::from(gt_candidates.len()).unwrap();
                    let query_precision = F::from(hits).unwrap() / F::from(k).unwrap();

                    total_recall = total_recall + query_recall;
                    total_precision = total_precision + query_precision;
                    valid_queries += 1;
                }
            }

            if valid_queries > 0 {
                recall_at_k.insert(k, total_recall / F::from(valid_queries).unwrap());
                precision_at_k.insert(k, total_precision / F::from(valid_queries).unwrap());
            } else {
                recall_at_k.insert(k, F::zero());
                precision_at_k.insert(k, F::zero());
            }
        }

        // Compute mean reciprocal rank
        let mut mrr = F::zero();
        let mut valid_queries = 0;

        for query_idx in 0..n_queries {
            if let Some(gt_candidates) = gt_map.get(&query_idx) {
                let mut query_similarities: Vec<(F, usize)> = (0..n_candidates)
                    .map(|j| (similarities[[query_idx, j]], j))
                    .collect();

                query_similarities
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Find rank of first relevant item
                for (rank, (_, candidate_idx)) in query_similarities.iter().enumerate() {
                    if gt_candidates.contains(candidate_idx) {
                        mrr = mrr + F::one() / F::from(rank + 1).unwrap();
                        break;
                    }
                }
                valid_queries += 1;
            }
        }

        if valid_queries > 0 {
            mrr = mrr / F::from(valid_queries).unwrap();
        }

        Ok(CrossModalRetrievalResult {
            recall_at_k,
            precision_at_k,
            mean_reciprocal_rank: mrr,
            n_queries,
            n_candidates,
        })
    }

    /// Compute multimodal alignment score
    pub fn multimodal_alignment(
        &self,
        modality1_embeddings: &Array2<F>,
        modality2_embeddings: &Array2<F>,
        paired_indices: &[(usize, usize)],
    ) -> Result<MultimodalAlignmentResult<F>> {
        if modality1_embeddings.ncols() != modality2_embeddings.ncols() {
            return Err(MetricsError::InvalidInput(
                "Embedding dimension mismatch".to_string(),
            ));
        }

        if paired_indices.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No paired indices provided".to_string(),
            ));
        }

        let mut alignment_scores = Vec::with_capacity(paired_indices.len());
        let mut positive_similarities = Vec::with_capacity(paired_indices.len());

        for &(idx1, idx2) in paired_indices {
            if idx1 >= modality1_embeddings.nrows() || idx2 >= modality2_embeddings.nrows() {
                return Err(MetricsError::InvalidInput(
                    "Index out of bounds".to_string(),
                ));
            }

            let similarity = self.cosine_similarity(
                &modality1_embeddings.row(idx1).to_owned(),
                &modality2_embeddings.row(idx2).to_owned(),
            )?;

            positive_similarities.push(similarity);
            alignment_scores.push(similarity);
        }

        // Compute negative similarities (random pairs)
        let mut negative_similarities = Vec::new();
        let n_negatives = paired_indices.len() * 5; // 5x negative sampling

        for i in 0..n_negatives {
            let idx1 = i % modality1_embeddings.nrows();
            let idx2 = (i * 7) % modality2_embeddings.nrows(); // Use prime for better randomness

            // Skip if this is actually a positive pair
            if !paired_indices.contains(&(idx1, idx2)) {
                let similarity = self.cosine_similarity(
                    &modality1_embeddings.row(idx1).to_owned(),
                    &modality2_embeddings.row(idx2).to_owned(),
                )?;
                negative_similarities.push(similarity);
            }
        }

        // Compute alignment metrics
        let mean_positive_similarity = positive_similarities.iter().copied().sum::<F>()
            / F::from(positive_similarities.len()).unwrap();

        let mean_negative_similarity = if !negative_similarities.is_empty() {
            negative_similarities.iter().copied().sum::<F>()
                / F::from(negative_similarities.len()).unwrap()
        } else {
            F::zero()
        };

        let alignment_gap = mean_positive_similarity - mean_negative_similarity;

        // Compute standard deviations
        let pos_variance = positive_similarities
            .iter()
            .map(|&x| {
                let diff = x - mean_positive_similarity;
                diff * diff
            })
            .sum::<F>()
            / F::from(positive_similarities.len()).unwrap();
        let pos_std = pos_variance.sqrt();

        let neg_variance = if !negative_similarities.is_empty() {
            negative_similarities
                .iter()
                .map(|&x| {
                    let diff = x - mean_negative_similarity;
                    diff * diff
                })
                .sum::<F>()
                / F::from(negative_similarities.len()).unwrap()
        } else {
            F::zero()
        };
        let neg_std = neg_variance.sqrt();

        Ok(MultimodalAlignmentResult {
            mean_positive_similarity,
            mean_negative_similarity,
            alignment_gap,
            positive_std: pos_std,
            negative_std: neg_std,
            n_positive_pairs: positive_similarities.len(),
            n_negative_pairs: negative_similarities.len(),
        })
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let dot_product: F = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a = (a.mapv(|x| x * x).sum()).sqrt();
        let norm_b = (b.mapv(|x| x * x).sum()).sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::zero());
        }

        Ok(dot_product / (norm_a * norm_b))
    }
}
