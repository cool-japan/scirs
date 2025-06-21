//! Recommender systems domain metrics
//!
//! This module provides specialized metric collections for recommender systems
//! including ranking, rating prediction, diversity, and fairness metrics.

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use crate::ranking::{mean_average_precision, ndcg_score, precision_at_k, recall_at_k};
use crate::regression::{mean_absolute_error, mean_squared_error};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Ranking evaluation results for recommender systems
#[derive(Debug, Clone)]
pub struct RecommenderRankingResults {
    /// Normalized Discounted Cumulative Gain at k
    pub ndcg_at_k: HashMap<usize, f64>,
    /// Precision at k
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall at k
    pub recall_at_k: HashMap<usize, f64>,
    /// Mean Average Precision
    pub map: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Hit Rate at k
    pub hit_rate_at_k: HashMap<usize, f64>,
    /// Coverage at k (fraction of items recommended)
    pub coverage_at_k: HashMap<usize, f64>,
}

/// Rating prediction evaluation results
#[derive(Debug, Clone)]
pub struct RatingPredictionResults {
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Precision for rating thresholds
    pub precision_at_threshold: HashMap<String, f64>,
    /// Recall for rating thresholds
    pub recall_at_threshold: HashMap<String, f64>,
    /// F1 score for rating thresholds
    pub f1_at_threshold: HashMap<String, f64>,
}

/// Diversity evaluation results
#[derive(Debug, Clone)]
pub struct DiversityResults {
    /// Intra-list diversity (average pairwise distance)
    pub intra_list_diversity: f64,
    /// Item coverage (fraction of catalog items recommended)
    pub item_coverage: f64,
    /// User coverage (fraction of users with recommendations)
    pub user_coverage: f64,
    /// Gini coefficient for item popularity distribution
    pub gini_coefficient: f64,
    /// Entropy of item distribution
    pub entropy: f64,
    /// Long-tail coverage (fraction of long-tail items recommended)
    pub long_tail_coverage: f64,
}

/// Novelty evaluation results
#[derive(Debug, Clone)]
pub struct NoveltyResults {
    /// Item novelty (average item popularity score)
    pub item_novelty: f64,
    /// User novelty (fraction of items new to user)
    pub user_novelty: f64,
    /// Temporal novelty (recency of recommendations)
    pub temporal_novelty: f64,
    /// Serendipity score
    pub serendipity: f64,
}

/// Fairness evaluation results for recommender systems
#[derive(Debug, Clone)]
pub struct RecommenderFairnessResults {
    /// Demographic parity across user groups
    pub demographic_parity: f64,
    /// Equal opportunity across user groups
    pub equal_opportunity: f64,
    /// Provider fairness (equal exposure for items/providers)
    pub provider_fairness: f64,
    /// Group fairness metrics per demographic group
    pub group_metrics: HashMap<String, f64>,
}

/// Business metrics evaluation results
#[derive(Debug, Clone)]
pub struct BusinessMetricsResults {
    /// Click-through rate
    pub ctr: f64,
    /// Conversion rate
    pub conversion_rate: f64,
    /// Revenue per user
    pub revenue_per_user: f64,
    /// Item turnover rate
    pub item_turnover: f64,
    /// User engagement score
    pub engagement_score: f64,
    /// Catalog utilization
    pub catalog_utilization: f64,
}

/// Recommender ranking metrics calculator
pub struct RecommenderRankingMetrics {
    k_values: Vec<usize>,
    total_items: usize,
}

impl Default for RecommenderRankingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommenderRankingMetrics {
    /// Create new recommender ranking metrics calculator
    pub fn new() -> Self {
        Self {
            k_values: vec![1, 5, 10, 20, 50],
            total_items: 1000, // Default catalog size
        }
    }

    /// Set k values for evaluation
    pub fn with_k_values(mut self, k_values: Vec<usize>) -> Self {
        self.k_values = k_values;
        self
    }

    /// Set total number of items in catalog
    pub fn with_total_items(mut self, total_items: usize) -> Self {
        self.total_items = total_items;
        self
    }

    /// Evaluate recommender ranking performance
    pub fn evaluate_ranking(
        &self,
        y_true: &[Array1<f64>],           // Relevance scores per user
        y_score: &[Array1<f64>],          // Predicted scores per user
        recommended_items: &[Vec<usize>], // Item IDs recommended per user
    ) -> Result<RecommenderRankingResults> {
        if y_true.len() != y_score.len() || y_true.len() != recommended_items.len() {
            return Err(MetricsError::InvalidInput(
                "Inconsistent number of users across inputs".to_string(),
            ));
        }

        let mut ndcg_at_k = HashMap::new();
        let mut precision_at_k_map = HashMap::new();
        let mut recall_at_k_map = HashMap::new();
        let mut hit_rate_at_k = HashMap::new();
        let mut coverage_at_k = HashMap::new();

        // Calculate metrics for each k value
        for &k in &self.k_values {
            let mut ndcg_scores = Vec::new();
            let mut precision_scores = Vec::new();
            let mut recall_scores = Vec::new();
            let mut hit_scores = Vec::new();
            let mut recommended_items_set = HashSet::new();

            for i in 0..y_true.len() {
                // Calculate NDCG@k
                if let Ok(ndcg) = ndcg_score(&[y_true[i].clone()], &[y_score[i].clone()], Some(k)) {
                    ndcg_scores.push(ndcg);
                }

                // Calculate Precision@k and Recall@k
                if let Ok(prec) = precision_at_k(&[y_true[i].clone()], &[y_score[i].clone()], k) {
                    precision_scores.push(prec);
                }

                if let Ok(rec) = recall_at_k(&[y_true[i].clone()], &[y_score[i].clone()], k) {
                    recall_scores.push(rec);
                }

                // Calculate Hit Rate@k (whether any relevant item is in top-k)
                let relevant_items: HashSet<usize> = y_true[i]
                    .iter()
                    .enumerate()
                    .filter(|(_, &score)| score > 0.0)
                    .map(|(idx, _)| idx)
                    .collect();

                let top_k_items: HashSet<usize> =
                    recommended_items[i].iter().take(k).copied().collect();

                let has_hit = !relevant_items.is_disjoint(&top_k_items);
                hit_scores.push(if has_hit { 1.0 } else { 0.0 });

                // Collect recommended items for coverage calculation
                for &item_id in recommended_items[i].iter().take(k) {
                    recommended_items_set.insert(item_id);
                }
            }

            // Store average metrics
            ndcg_at_k.insert(
                k,
                ndcg_scores.iter().sum::<f64>() / ndcg_scores.len() as f64,
            );
            precision_at_k_map.insert(
                k,
                precision_scores.iter().sum::<f64>() / precision_scores.len() as f64,
            );
            recall_at_k_map.insert(
                k,
                recall_scores.iter().sum::<f64>() / recall_scores.len() as f64,
            );
            hit_rate_at_k.insert(k, hit_scores.iter().sum::<f64>() / hit_scores.len() as f64);
            coverage_at_k.insert(
                k,
                recommended_items_set.len() as f64 / self.total_items as f64,
            );
        }

        // Calculate MAP
        let map = mean_average_precision(y_true, y_score, None).unwrap_or(0.0);

        // Calculate MRR
        let mrr = self.calculate_mrr(y_true, recommended_items)?;

        Ok(RecommenderRankingResults {
            ndcg_at_k,
            precision_at_k: precision_at_k_map,
            recall_at_k: recall_at_k_map,
            map,
            mrr,
            hit_rate_at_k,
            coverage_at_k,
        })
    }

    /// Calculate Mean Reciprocal Rank
    fn calculate_mrr(
        &self,
        y_true: &[Array1<f64>],
        recommended_items: &[Vec<usize>],
    ) -> Result<f64> {
        let mut reciprocal_ranks = Vec::new();

        for i in 0..y_true.len() {
            let relevant_items: HashSet<usize> = y_true[i]
                .iter()
                .enumerate()
                .filter(|(_, &score)| score > 0.0)
                .map(|(idx, _)| idx)
                .collect();

            let mut rank = None;
            for (pos, &item_id) in recommended_items[i].iter().enumerate() {
                if relevant_items.contains(&item_id) {
                    rank = Some(pos + 1); // 1-indexed rank
                    break;
                }
            }

            if let Some(r) = rank {
                reciprocal_ranks.push(1.0 / r as f64);
            } else {
                reciprocal_ranks.push(0.0);
            }
        }

        Ok(reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64)
    }
}

/// Rating prediction metrics calculator
pub struct RatingPredictionMetrics {
    rating_thresholds: Vec<f64>,
}

impl Default for RatingPredictionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl RatingPredictionMetrics {
    /// Create new rating prediction metrics calculator
    pub fn new() -> Self {
        Self {
            rating_thresholds: vec![3.0, 3.5, 4.0], // Common thresholds for 5-star ratings
        }
    }

    /// Set rating thresholds for binary classification metrics
    pub fn with_thresholds(mut self, thresholds: Vec<f64>) -> Self {
        self.rating_thresholds = thresholds;
        self
    }

    /// Evaluate rating prediction performance
    pub fn evaluate_rating_prediction(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> Result<RatingPredictionResults> {
        let rmse = mean_squared_error(y_true, y_pred)?.sqrt();
        let mae = mean_absolute_error(y_true, y_pred)?;

        let mut precision_at_threshold = HashMap::new();
        let mut recall_at_threshold = HashMap::new();
        let mut f1_at_threshold = HashMap::new();

        // Calculate binary classification metrics for each threshold
        for &threshold in &self.rating_thresholds {
            let y_true_binary: Array1<i32> = y_true.mapv(|x| if x >= threshold { 1 } else { 0 });
            let y_pred_binary: Array1<i32> = y_pred.mapv(|x| if x >= threshold { 1 } else { 0 });

            let (precision, recall, f1) =
                self.calculate_binary_metrics(&y_true_binary, &y_pred_binary)?;

            let threshold_key = format!("threshold_{:.1}", threshold);
            precision_at_threshold.insert(threshold_key.clone(), precision);
            recall_at_threshold.insert(threshold_key.clone(), recall);
            f1_at_threshold.insert(threshold_key, f1);
        }

        Ok(RatingPredictionResults {
            rmse,
            mae,
            precision_at_threshold,
            recall_at_threshold,
            f1_at_threshold,
        })
    }

    /// Calculate binary classification metrics
    fn calculate_binary_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<(f64, f64, f64)> {
        let mut tp = 0;
        let mut fp = 0;
        let mut _tn = 0;
        let mut fn_count = 0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => _tn += 1,
                (1, 0) => fn_count += 1,
                _ => {
                    return Err(MetricsError::InvalidInput(
                        "Invalid binary labels".to_string(),
                    ))
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok((precision, recall, f1))
    }
}

/// Diversity metrics calculator
pub struct DiversityMetrics {
    item_features: Option<Array2<f64>>, // Item feature matrix for diversity calculation
    popularity_scores: Option<HashMap<usize, f64>>, // Item popularity for novelty
}

impl Default for DiversityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DiversityMetrics {
    /// Create new diversity metrics calculator
    pub fn new() -> Self {
        Self {
            item_features: None,
            popularity_scores: None,
        }
    }

    /// Set item features for diversity calculation
    pub fn with_item_features(mut self, features: Array2<f64>) -> Self {
        self.item_features = Some(features);
        self
    }

    /// Set item popularity scores
    pub fn with_popularity_scores(mut self, scores: HashMap<usize, f64>) -> Self {
        self.popularity_scores = Some(scores);
        self
    }

    /// Evaluate diversity of recommendations
    pub fn evaluate_diversity(
        &self,
        recommended_items: &[Vec<usize>], // Item IDs recommended per user
        total_catalog_size: usize,
        total_users: usize,
        long_tail_threshold: f64, // Popularity threshold for long-tail items
    ) -> Result<DiversityResults> {
        // Calculate intra-list diversity
        let intra_list_diversity = if let Some(features) = &self.item_features {
            self.calculate_intra_list_diversity(recommended_items, features)?
        } else {
            0.0
        };

        // Calculate item coverage
        let all_recommended_items: HashSet<usize> = recommended_items
            .iter()
            .flat_map(|items| items.iter())
            .copied()
            .collect();
        let item_coverage = all_recommended_items.len() as f64 / total_catalog_size as f64;

        // Calculate user coverage
        let users_with_recommendations = recommended_items
            .iter()
            .filter(|items| !items.is_empty())
            .count();
        let user_coverage = users_with_recommendations as f64 / total_users as f64;

        // Calculate Gini coefficient
        let gini_coefficient = self.calculate_gini_coefficient(recommended_items)?;

        // Calculate entropy
        let entropy = self.calculate_entropy(recommended_items)?;

        // Calculate long-tail coverage
        let long_tail_coverage = if let Some(popularity) = &self.popularity_scores {
            self.calculate_long_tail_coverage(recommended_items, popularity, long_tail_threshold)?
        } else {
            0.0
        };

        Ok(DiversityResults {
            intra_list_diversity,
            item_coverage,
            user_coverage,
            gini_coefficient,
            entropy,
            long_tail_coverage,
        })
    }

    /// Calculate average intra-list diversity using item features
    fn calculate_intra_list_diversity(
        &self,
        recommended_items: &[Vec<usize>],
        features: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_diversity = 0.0;
        let mut count = 0;

        for items in recommended_items {
            if items.len() < 2 {
                continue;
            }

            let mut pairwise_distances = Vec::new();
            for i in 0..items.len() {
                for j in i + 1..items.len() {
                    let item1_id = items[i];
                    let item2_id = items[j];

                    if item1_id < features.nrows() && item2_id < features.nrows() {
                        let distance = self.cosine_distance(
                            &features.row(item1_id).to_owned(),
                            &features.row(item2_id).to_owned(),
                        );
                        pairwise_distances.push(distance);
                    }
                }
            }

            if !pairwise_distances.is_empty() {
                total_diversity +=
                    pairwise_distances.iter().sum::<f64>() / pairwise_distances.len() as f64;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_diversity / count as f64
        } else {
            0.0
        })
    }

    /// Calculate cosine distance between two feature vectors
    fn cosine_distance(&self, vec1: &Array1<f64>, vec2: &Array1<f64>) -> f64 {
        let dot_product = vec1.dot(vec2);
        let norm1 = vec1.dot(vec1).sqrt();
        let norm2 = vec2.dot(vec2).sqrt();

        if norm1 > 1e-10 && norm2 > 1e-10 {
            1.0 - dot_product / (norm1 * norm2)
        } else {
            1.0 // Maximum distance if either vector is zero
        }
    }

    /// Calculate Gini coefficient for item distribution
    fn calculate_gini_coefficient(&self, recommended_items: &[Vec<usize>]) -> Result<f64> {
        // Count item frequencies
        let mut item_counts = HashMap::new();
        for items in recommended_items {
            for &item in items {
                *item_counts.entry(item).or_insert(0) += 1;
            }
        }

        if item_counts.is_empty() {
            return Ok(0.0);
        }

        let mut counts: Vec<usize> = item_counts.values().copied().collect();
        counts.sort_unstable();

        let n = counts.len();
        let sum: usize = counts.iter().sum();

        if sum == 0 {
            return Ok(0.0);
        }

        let mut gini_sum = 0.0;
        for (i, &count) in counts.iter().enumerate() {
            gini_sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * count as f64;
        }

        Ok(gini_sum / (n as f64 * sum as f64))
    }

    /// Calculate entropy of item distribution
    fn calculate_entropy(&self, recommended_items: &[Vec<usize>]) -> Result<f64> {
        let mut item_counts = HashMap::new();
        let mut total_recommendations = 0;

        for items in recommended_items {
            for &item in items {
                *item_counts.entry(item).or_insert(0) += 1;
                total_recommendations += 1;
            }
        }

        if total_recommendations == 0 {
            return Ok(0.0);
        }

        let mut entropy = 0.0;
        for &count in item_counts.values() {
            let probability = count as f64 / total_recommendations as f64;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        Ok(entropy)
    }

    /// Calculate coverage of long-tail items
    fn calculate_long_tail_coverage(
        &self,
        recommended_items: &[Vec<usize>],
        popularity_scores: &HashMap<usize, f64>,
        threshold: f64,
    ) -> Result<f64> {
        // Identify long-tail items (below popularity threshold)
        let long_tail_items: HashSet<usize> = popularity_scores
            .iter()
            .filter(|(_, &popularity)| popularity < threshold)
            .map(|(&item_id, _)| item_id)
            .collect();

        if long_tail_items.is_empty() {
            return Ok(0.0);
        }

        // Count recommended long-tail items
        let recommended_long_tail: HashSet<usize> = recommended_items
            .iter()
            .flat_map(|items| items.iter())
            .filter(|&item_id| long_tail_items.contains(item_id))
            .copied()
            .collect();

        Ok(recommended_long_tail.len() as f64 / long_tail_items.len() as f64)
    }
}

/// Complete recommender systems metrics suite
pub struct RecommenderSuite {
    ranking: RecommenderRankingMetrics,
    rating: RatingPredictionMetrics,
    diversity: DiversityMetrics,
}

impl Default for RecommenderSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommenderSuite {
    /// Create a new recommender systems metrics suite
    pub fn new() -> Self {
        Self {
            ranking: RecommenderRankingMetrics::new(),
            rating: RatingPredictionMetrics::new(),
            diversity: DiversityMetrics::new(),
        }
    }

    /// Get ranking metrics calculator
    pub fn ranking(&self) -> &RecommenderRankingMetrics {
        &self.ranking
    }

    /// Get rating prediction metrics calculator
    pub fn rating(&self) -> &RatingPredictionMetrics {
        &self.rating
    }

    /// Get diversity metrics calculator
    pub fn diversity(&self) -> &DiversityMetrics {
        &self.diversity
    }
}

impl DomainMetrics for RecommenderSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Recommender Systems"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "ranking_ndcg_at_10",
            "ranking_precision_at_10",
            "ranking_recall_at_10",
            "ranking_map",
            "ranking_mrr",
            "ranking_hit_rate_at_10",
            "rating_rmse",
            "rating_mae",
            "diversity_intra_list",
            "diversity_item_coverage",
            "diversity_entropy",
            "diversity_long_tail_coverage",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "ranking_ndcg_at_10",
            "Normalized Discounted Cumulative Gain at 10",
        );
        descriptions.insert(
            "ranking_precision_at_10",
            "Precision at 10 for ranking evaluation",
        );
        descriptions.insert(
            "ranking_recall_at_10",
            "Recall at 10 for ranking evaluation",
        );
        descriptions.insert("ranking_map", "Mean Average Precision for ranking");
        descriptions.insert("ranking_mrr", "Mean Reciprocal Rank");
        descriptions.insert(
            "ranking_hit_rate_at_10",
            "Hit rate at 10 (fraction of users with relevant items in top-10)",
        );
        descriptions.insert(
            "rating_rmse",
            "Root Mean Squared Error for rating prediction",
        );
        descriptions.insert("rating_mae", "Mean Absolute Error for rating prediction");
        descriptions.insert(
            "diversity_intra_list",
            "Average intra-list diversity of recommendations",
        );
        descriptions.insert(
            "diversity_item_coverage",
            "Fraction of catalog items recommended",
        );
        descriptions.insert(
            "diversity_entropy",
            "Entropy of item recommendation distribution",
        );
        descriptions.insert(
            "diversity_long_tail_coverage",
            "Coverage of long-tail items",
        );
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranking_metrics() {
        let metrics = RecommenderRankingMetrics::new().with_k_values(vec![5, 10]);

        let y_true = vec![
            Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]),
        ];

        let y_score = vec![
            Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7]),
            Array1::from_vec(vec![0.3, 0.9, 0.4, 0.8, 0.1]),
        ];

        let recommended_items = vec![
            vec![0, 2, 4, 1, 3], // Based on y_score ranking
            vec![1, 3, 2, 0, 4],
        ];

        let results = metrics
            .evaluate_ranking(&y_true, &y_score, &recommended_items)
            .unwrap();

        assert!(results.ndcg_at_k.contains_key(&5));
        assert!(results.precision_at_k.contains_key(&5));
        assert!(results.recall_at_k.contains_key(&5));
        assert!(results.map >= 0.0 && results.map <= 1.0);
        assert!(results.mrr >= 0.0 && results.mrr <= 1.0);
    }

    #[test]
    fn test_rating_prediction_metrics() {
        let metrics = RatingPredictionMetrics::new();

        let y_true = Array1::from_vec(vec![4.0, 3.5, 2.0, 4.5, 1.0]);
        let y_pred = Array1::from_vec(vec![3.8, 3.2, 2.3, 4.2, 1.5]);

        let results = metrics
            .evaluate_rating_prediction(&y_true, &y_pred)
            .unwrap();

        assert!(results.rmse >= 0.0);
        assert!(results.mae >= 0.0);
        assert!(!results.precision_at_threshold.is_empty());
        assert!(!results.recall_at_threshold.is_empty());
        assert!(!results.f1_at_threshold.is_empty());
    }

    #[test]
    fn test_diversity_metrics() {
        let metrics = DiversityMetrics::new();

        let recommended_items = vec![vec![0, 1, 2], vec![1, 3, 4], vec![2, 4, 5]];

        let results = metrics
            .evaluate_diversity(&recommended_items, 10, 3, 0.1)
            .unwrap();

        assert!(results.item_coverage >= 0.0 && results.item_coverage <= 1.0);
        assert!(results.user_coverage >= 0.0 && results.user_coverage <= 1.0);
        assert!(results.gini_coefficient >= 0.0 && results.gini_coefficient <= 1.0);
        assert!(results.entropy >= 0.0);
    }

    #[test]
    fn test_mrr_calculation() {
        let metrics = RecommenderRankingMetrics::new();

        let y_true = vec![
            Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]), // Relevant items at positions 0, 2
            Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]), // Relevant items at positions 1, 3
        ];

        let recommended_items = vec![
            vec![1, 2, 0, 3], // First relevant item (2) at position 1 (0-indexed) -> rank 2 -> RR = 0.5
            vec![1, 3, 0, 2], // First relevant item (1) at position 0 (0-indexed) -> rank 1 -> RR = 1.0
        ];

        let mrr = metrics.calculate_mrr(&y_true, &recommended_items).unwrap();

        // Expected MRR = (0.5 + 1.0) / 2 = 0.75
        assert!((mrr - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_gini_coefficient() {
        let metrics = DiversityMetrics::new();

        // Perfect equality: all items recommended equally
        let equal_items = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let gini_equal = metrics.calculate_gini_coefficient(&equal_items).unwrap();
        assert!(gini_equal < 0.1); // Should be close to 0

        // Inequality: one item heavily recommended vs others
        let unequal_items = vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0],
            vec![0],
        ];
        let gini_unequal = metrics.calculate_gini_coefficient(&unequal_items).unwrap();
        assert!(gini_unequal > 0.1); // Should be higher than equal distribution
    }

    #[test]
    fn test_recommender_suite() {
        let suite = RecommenderSuite::new();

        assert_eq!(suite.domain_name(), "Recommender Systems");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
