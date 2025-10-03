//! Audio similarity and retrieval metrics
//!
//! This module provides comprehensive metrics for audio similarity measurement and
//! content-based audio retrieval tasks, including acoustic and semantic similarity measures.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Audio similarity and retrieval metrics
#[derive(Debug, Clone)]
pub struct AudioSimilarityMetrics {
    /// Content-based retrieval metrics
    pub content_based: ContentBasedRetrievalMetrics,
    /// Acoustic similarity metrics
    pub acoustic_similarity: AcousticSimilarityMetrics,
    /// Semantic similarity metrics
    pub semantic_similarity: SemanticSimilarityMetrics,
}

/// Content-based audio retrieval metrics
#[derive(Debug, Clone)]
pub struct ContentBasedRetrievalMetrics {
    /// Mean Average Precision (MAP)
    map: f64,
    /// Precision at different K values
    precision_at_k: HashMap<usize, f64>,
    /// Recall at different K values
    recall_at_k: HashMap<usize, f64>,
    /// Normalized Discounted Cumulative Gain
    ndcg: f64,
    /// Mean Reciprocal Rank
    mrr: f64,
}

/// Acoustic similarity metrics
#[derive(Debug, Clone, Default)]
pub struct AcousticSimilarityMetrics {
    /// Mel-frequency cepstral coefficient similarity
    mfcc_similarity: f64,
    /// Chroma feature similarity
    chroma_similarity: f64,
    /// Spectral centroid similarity
    spectral_centroid_similarity: f64,
    /// Zero-crossing rate similarity
    zcr_similarity: f64,
    /// Spectral rolloff similarity
    spectral_rolloff_similarity: f64,
}

/// Semantic similarity metrics
#[derive(Debug, Clone, Default)]
pub struct SemanticSimilarityMetrics {
    /// Tag-based similarity
    tag_similarity: f64,
    /// Genre classification similarity
    genre_similarity: f64,
    /// Mood classification similarity
    mood_similarity: f64,
    /// Instrument classification similarity
    instrument_similarity: f64,
    /// Semantic embedding similarity
    embedding_similarity: f64,
}

/// Audio similarity results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSimilarityResults {
    /// Content-based retrieval MAP
    pub retrieval_map: f64,
    /// Acoustic similarity score
    pub acoustic_similarity: f64,
    /// Semantic similarity score
    pub semantic_similarity: f64,
    /// Overall similarity score
    pub overall_similarity: f64,
}

impl AudioSimilarityMetrics {
    /// Create new audio similarity metrics
    pub fn new() -> Self {
        Self {
            content_based: ContentBasedRetrievalMetrics::new(),
            acoustic_similarity: AcousticSimilarityMetrics::new(),
            semantic_similarity: SemanticSimilarityMetrics::new(),
        }
    }

    /// Evaluate content-based audio retrieval
    pub fn evaluate_retrieval(
        &mut self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
        k_values: &[usize],
    ) -> Result<f64> {
        // Calculate MAP
        let map_score =
            self.content_based
                .calculate_map(query_ids, relevant_docs, retrieved_docs)?;

        // Calculate precision and recall at K
        for &k in k_values {
            let precision_k = self.content_based.calculate_precision_at_k(
                query_ids,
                relevant_docs,
                retrieved_docs,
                k,
            )?;
            let recall_k = self.content_based.calculate_recall_at_k(
                query_ids,
                relevant_docs,
                retrieved_docs,
                k,
            )?;

            self.content_based.precision_at_k.insert(k, precision_k);
            self.content_based.recall_at_k.insert(k, recall_k);
        }

        // Calculate NDCG
        let ndcg_score =
            self.content_based
                .calculate_ndcg(query_ids, relevant_docs, retrieved_docs)?;
        self.content_based.ndcg = ndcg_score;

        // Calculate MRR
        let mrr_score =
            self.content_based
                .calculate_mrr(query_ids, relevant_docs, retrieved_docs)?;
        self.content_based.mrr = mrr_score;

        Ok(map_score)
    }

    /// Evaluate acoustic similarity between audio features
    pub fn evaluate_acoustic_similarity<F>(
        &mut self,
        reference_features: &HashMap<String, ArrayView1<F>>,
        query_features: &HashMap<String, ArrayView1<F>>,
    ) -> Result<f64>
    where
        F: Float + std::fmt::Debug,
    {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (feature_type, ref_features) in reference_features {
            if let Some(query_feats) = query_features.get(feature_type) {
                let similarity = self.calculate_cosine_similarity(ref_features, query_feats)?;

                match feature_type.as_str() {
                    "mfcc" => self.acoustic_similarity.mfcc_similarity = similarity,
                    "chroma" => self.acoustic_similarity.chroma_similarity = similarity,
                    "spectral_centroid" => {
                        self.acoustic_similarity.spectral_centroid_similarity = similarity
                    }
                    "zcr" => self.acoustic_similarity.zcr_similarity = similarity,
                    "spectral_rolloff" => {
                        self.acoustic_similarity.spectral_rolloff_similarity = similarity
                    }
                    _ => {}
                }

                total_similarity += similarity;
                count += 1;
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / count as f64)
        }
    }

    /// Evaluate semantic similarity based on labels and embeddings
    pub fn evaluate_semantic_similarity(
        &mut self,
        reference_tags: &HashMap<String, Vec<String>>,
        query_tags: &HashMap<String, Vec<String>>,
        reference_embeddings: Option<&HashMap<String, ArrayView1<f64>>>,
        query_embeddings: Option<&HashMap<String, ArrayView1<f64>>>,
    ) -> Result<f64> {
        // Calculate tag-based similarity
        let tag_sim = self.calculate_tag_similarity(reference_tags, query_tags)?;
        self.semantic_similarity.tag_similarity = tag_sim;

        // Calculate embedding similarity if provided
        let embedding_sim =
            if let (Some(ref_emb), Some(query_emb)) = (reference_embeddings, query_embeddings) {
                self.calculate_embedding_similarity(ref_emb, query_emb)?
            } else {
                0.0
            };
        self.semantic_similarity.embedding_similarity = embedding_sim;

        // Calculate genre, mood, and instrument similarities from tags
        let genre_sim = self.calculate_category_similarity(reference_tags, query_tags, "genre")?;
        let mood_sim = self.calculate_category_similarity(reference_tags, query_tags, "mood")?;
        let instrument_sim =
            self.calculate_category_similarity(reference_tags, query_tags, "instrument")?;

        self.semantic_similarity.genre_similarity = genre_sim;
        self.semantic_similarity.mood_similarity = mood_sim;
        self.semantic_similarity.instrument_similarity = instrument_sim;

        // Return average semantic similarity
        Ok((tag_sim + embedding_sim + genre_sim + mood_sim + instrument_sim) / 5.0)
    }

    /// Calculate cosine similarity between two feature vectors
    fn calculate_cosine_similarity<F>(
        &self,
        features1: &ArrayView1<F>,
        features2: &ArrayView1<F>,
    ) -> Result<f64>
    where
        F: Float + std::fmt::Debug,
    {
        if features1.len() != features2.len() {
            return Err(MetricsError::InvalidInput(
                "Feature vectors must have the same length".to_string(),
            ));
        }

        let dot_product = features1
            .iter()
            .zip(features2.iter())
            .map(|(&a, &b)| a * b)
            .fold(F::zero(), |acc, x| acc + x);

        let norm1 = features1
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();

        let norm2 = features2
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();

        if norm1 > F::zero() && norm2 > F::zero() {
            Ok((dot_product / (norm1 * norm2)).to_f64().unwrap_or(0.0))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate tag-based similarity using Jaccard index
    fn calculate_tag_similarity(
        &self,
        reference_tags: &HashMap<String, Vec<String>>,
        query_tags: &HashMap<String, Vec<String>>,
    ) -> Result<f64> {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (key, ref_tags) in reference_tags {
            if let Some(query_tag_vec) = query_tags.get(key) {
                let ref_set: std::collections::HashSet<_> = ref_tags.iter().collect();
                let query_set: std::collections::HashSet<_> = query_tag_vec.iter().collect();

                let intersection = ref_set.intersection(&query_set).count();
                let union = ref_set.union(&query_set).count();

                let jaccard = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };

                total_similarity += jaccard;
                count += 1;
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / count as f64)
        }
    }

    /// Calculate embedding similarity
    fn calculate_embedding_similarity(
        &self,
        reference_embeddings: &HashMap<String, ArrayView1<f64>>,
        query_embeddings: &HashMap<String, ArrayView1<f64>>,
    ) -> Result<f64> {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (key, ref_emb) in reference_embeddings {
            if let Some(query_emb) = query_embeddings.get(key) {
                let similarity = self.calculate_cosine_similarity(ref_emb, query_emb)?;
                total_similarity += similarity;
                count += 1;
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / count as f64)
        }
    }

    /// Calculate category-specific similarity (genre, mood, instrument)
    fn calculate_category_similarity(
        &self,
        reference_tags: &HashMap<String, Vec<String>>,
        query_tags: &HashMap<String, Vec<String>>,
        category: &str,
    ) -> Result<f64> {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (key, ref_tags) in reference_tags {
            if let Some(query_tag_vec) = query_tags.get(key) {
                // Filter tags for specific category
                let ref_category_tags: Vec<_> = ref_tags
                    .iter()
                    .filter(|tag| tag.starts_with(category))
                    .collect();
                let query_category_tags: Vec<_> = query_tag_vec
                    .iter()
                    .filter(|tag| tag.starts_with(category))
                    .collect();

                if !ref_category_tags.is_empty() || !query_category_tags.is_empty() {
                    let ref_set: std::collections::HashSet<_> =
                        ref_category_tags.into_iter().collect();
                    let query_set: std::collections::HashSet<_> =
                        query_category_tags.into_iter().collect();

                    let intersection = ref_set.intersection(&query_set).count();
                    let union = ref_set.union(&query_set).count();

                    let jaccard = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    total_similarity += jaccard;
                    count += 1;
                }
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / count as f64)
        }
    }

    /// Get comprehensive similarity results
    pub fn get_results(&self) -> AudioSimilarityResults {
        let acoustic_avg = (self.acoustic_similarity.mfcc_similarity
            + self.acoustic_similarity.chroma_similarity
            + self.acoustic_similarity.spectral_centroid_similarity
            + self.acoustic_similarity.zcr_similarity
            + self.acoustic_similarity.spectral_rolloff_similarity)
            / 5.0;

        let semantic_avg = (self.semantic_similarity.tag_similarity
            + self.semantic_similarity.genre_similarity
            + self.semantic_similarity.mood_similarity
            + self.semantic_similarity.instrument_similarity
            + self.semantic_similarity.embedding_similarity)
            / 5.0;

        let overall = (self.content_based.map + acoustic_avg + semantic_avg) / 3.0;

        AudioSimilarityResults {
            retrieval_map: self.content_based.map,
            acoustic_similarity: acoustic_avg,
            semantic_similarity: semantic_avg,
            overall_similarity: overall,
        }
    }
}

impl ContentBasedRetrievalMetrics {
    /// Create new content-based retrieval metrics
    pub fn new() -> Self {
        Self {
            map: 0.0,
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            ndcg: 0.0,
            mrr: 0.0,
        }
    }

    /// Calculate Mean Average Precision (MAP)
    fn calculate_map(
        &mut self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
    ) -> Result<f64> {
        let mut total_ap = 0.0;
        let mut valid_queries = 0;

        for query_id in query_ids {
            if let (Some(relevant), Some(retrieved)) =
                (relevant_docs.get(query_id), retrieved_docs.get(query_id))
            {
                if !relevant.is_empty() {
                    let ap = self.calculate_average_precision(relevant, retrieved);
                    total_ap += ap;
                    valid_queries += 1;
                }
            }
        }

        let map_score = if valid_queries > 0 {
            total_ap / valid_queries as f64
        } else {
            0.0
        };

        self.map = map_score;
        Ok(map_score)
    }

    /// Calculate Average Precision for a single query
    fn calculate_average_precision(&self, relevant: &[String], retrieved: &[String]) -> f64 {
        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let mut precision_sum = 0.0;
        let mut relevant_found = 0;

        for (i, doc) in retrieved.iter().enumerate() {
            if relevant_set.contains(doc) {
                relevant_found += 1;
                let precision_at_i = relevant_found as f64 / (i + 1) as f64;
                precision_sum += precision_at_i;
            }
        }

        if relevant.is_empty() {
            0.0
        } else {
            precision_sum / relevant.len() as f64
        }
    }

    /// Calculate Precision at K
    fn calculate_precision_at_k(
        &self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
        k: usize,
    ) -> Result<f64> {
        let mut total_precision = 0.0;
        let mut valid_queries = 0;

        for query_id in query_ids {
            if let (Some(relevant), Some(retrieved)) =
                (relevant_docs.get(query_id), retrieved_docs.get(query_id))
            {
                let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
                let retrieved_k = &retrieved[..retrieved.len().min(k)];

                let relevant_in_k = retrieved_k
                    .iter()
                    .filter(|doc| relevant_set.contains(doc))
                    .count();

                let precision = relevant_in_k as f64 / retrieved_k.len() as f64;
                total_precision += precision;
                valid_queries += 1;
            }
        }

        if valid_queries > 0 {
            Ok(total_precision / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Recall at K
    fn calculate_recall_at_k(
        &self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
        k: usize,
    ) -> Result<f64> {
        let mut total_recall = 0.0;
        let mut valid_queries = 0;

        for query_id in query_ids {
            if let (Some(relevant), Some(retrieved)) =
                (relevant_docs.get(query_id), retrieved_docs.get(query_id))
            {
                if !relevant.is_empty() {
                    let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
                    let retrieved_k = &retrieved[..retrieved.len().min(k)];

                    let relevant_in_k = retrieved_k
                        .iter()
                        .filter(|doc| relevant_set.contains(doc))
                        .count();

                    let recall = relevant_in_k as f64 / relevant.len() as f64;
                    total_recall += recall;
                    valid_queries += 1;
                }
            }
        }

        if valid_queries > 0 {
            Ok(total_recall / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Normalized Discounted Cumulative Gain
    fn calculate_ndcg(
        &mut self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
    ) -> Result<f64> {
        // Simplified NDCG calculation assuming binary relevance
        let mut total_ndcg = 0.0;
        let mut valid_queries = 0;

        for query_id in query_ids {
            if let (Some(relevant), Some(retrieved)) =
                (relevant_docs.get(query_id), retrieved_docs.get(query_id))
            {
                if !relevant.is_empty() {
                    let ndcg = self.calculate_query_ndcg(relevant, retrieved);
                    total_ndcg += ndcg;
                    valid_queries += 1;
                }
            }
        }

        let ndcg_score = if valid_queries > 0 {
            total_ndcg / valid_queries as f64
        } else {
            0.0
        };

        Ok(ndcg_score)
    }

    /// Calculate NDCG for a single query
    fn calculate_query_ndcg(&self, relevant: &[String], retrieved: &[String]) -> f64 {
        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();

        // Calculate DCG
        let mut dcg = 0.0;
        for (i, doc) in retrieved.iter().enumerate() {
            if relevant_set.contains(doc) {
                let relevance = 1.0; // Binary relevance
                let discount = (i as f64 + 2.0).log2();
                dcg += relevance / discount;
            }
        }

        // Calculate IDCG (perfect ranking)
        let mut idcg = 0.0;
        for i in 0..relevant.len().min(retrieved.len()) {
            let relevance = 1.0;
            let discount = (i as f64 + 2.0).log2();
            idcg += relevance / discount;
        }

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }

    /// Calculate Mean Reciprocal Rank
    fn calculate_mrr(
        &mut self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
    ) -> Result<f64> {
        let mut total_rr = 0.0;
        let mut valid_queries = 0;

        for query_id in query_ids {
            if let (Some(relevant), Some(retrieved)) =
                (relevant_docs.get(query_id), retrieved_docs.get(query_id))
            {
                if !relevant.is_empty() {
                    let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();

                    // Find rank of first relevant document
                    let mut first_relevant_rank = None;
                    for (i, doc) in retrieved.iter().enumerate() {
                        if relevant_set.contains(doc) {
                            first_relevant_rank = Some(i + 1);
                            break;
                        }
                    }

                    if let Some(rank) = first_relevant_rank {
                        total_rr += 1.0 / rank as f64;
                    }
                    valid_queries += 1;
                }
            }
        }

        let mrr_score = if valid_queries > 0 {
            total_rr / valid_queries as f64
        } else {
            0.0
        };

        self.mrr = mrr_score;
        Ok(mrr_score)
    }

    /// Get MAP score
    pub fn get_map(&self) -> f64 {
        self.map
    }

    /// Get precision at K
    pub fn get_precision_at_k(&self, k: usize) -> Option<f64> {
        self.precision_at_k.get(&k).copied()
    }

    /// Get recall at K
    pub fn get_recall_at_k(&self, k: usize) -> Option<f64> {
        self.recall_at_k.get(&k).copied()
    }
}

impl AcousticSimilarityMetrics {
    /// Create new acoustic similarity metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get MFCC similarity
    pub fn get_mfcc_similarity(&self) -> f64 {
        self.mfcc_similarity
    }

    /// Get chroma similarity
    pub fn get_chroma_similarity(&self) -> f64 {
        self.chroma_similarity
    }

    /// Get spectral centroid similarity
    pub fn get_spectral_centroid_similarity(&self) -> f64 {
        self.spectral_centroid_similarity
    }
}

impl SemanticSimilarityMetrics {
    /// Create new semantic similarity metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get tag similarity
    pub fn get_tag_similarity(&self) -> f64 {
        self.tag_similarity
    }

    /// Get genre similarity
    pub fn get_genre_similarity(&self) -> f64 {
        self.genre_similarity
    }

    /// Get embedding similarity
    pub fn get_embedding_similarity(&self) -> f64 {
        self.embedding_similarity
    }
}

impl Default for AudioSimilarityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ContentBasedRetrievalMetrics {
    fn default() -> Self {
        Self::new()
    }
}
