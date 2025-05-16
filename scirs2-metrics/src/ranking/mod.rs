//! Ranking metrics module
//!
//! This module provides metrics for evaluating ranking and recommendation models.
//! These metrics are useful for tasks like search engine ranking, recommendation
//! systems, and information retrieval.
//!
//! ## Basic Metrics
//!
//! Basic ranking metrics include Mean Reciprocal Rank (MRR), Discounted Cumulative Gain (DCG),
//! Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), and
//! Precision and Recall at k.
//!
//! ## Label Ranking Metrics
//!
//! Label ranking metrics are designed for multi-label problems where each sample can
//! have multiple relevant labels. These include Coverage Error, Label Ranking Loss,
//! and Label Ranking Average Precision Score.
//!
//! ```
//! use ndarray::Array2;
//! use scirs2_metrics::ranking::label::{
//!     coverage_error_multiple, label_ranking_loss, label_ranking_average_precision_score
//! };
//!
//! // Example: 3 samples, 5 possible labels
//! let y_true = Array2::from_shape_vec((3, 5), vec![
//!     1.0, 0.0, 1.0, 0.0, 0.0,  // Sample 1: labels 0 and 2 are relevant
//!     0.0, 0.0, 1.0, 1.0, 0.0,  // Sample 2: labels 2 and 3 are relevant
//!     0.0, 1.0, 1.0, 0.0, 1.0,  // Sample 3: labels 1, 2, and 4 are relevant
//! ]).unwrap();
//!
//! // Predicted scores for each label
//! let y_score = Array2::from_shape_vec((3, 5), vec![
//!     0.9, 0.2, 0.8, 0.3, 0.1,  // Scores for sample 1
//!     0.2, 0.3, 0.9, 0.7, 0.1,  // Scores for sample 2
//!     0.1, 0.9, 0.8, 0.2, 0.7,  // Scores for sample 3
//! ]).unwrap();
//!
//! // Calculate label ranking metrics
//! let coverage = coverage_error_multiple(&y_true, &y_score).unwrap();
//! let loss = label_ranking_loss(&y_true, &y_score).unwrap();
//! let precision = label_ranking_average_precision_score(&y_true, &y_score).unwrap();
//! ```
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::ranking::{mean_reciprocal_rank, ndcg_score,
//!     mean_average_precision, precision_at_k, recall_at_k};
//!
//! // Example: search engine results where each array is a different query
//! // Values indicate whether a result is relevant (1.0) or not (0.0)
//! let y_true = vec![
//!     array![0.0, 1.0, 0.0, 0.0, 0.0],  // First query: second result is relevant
//!     array![0.0, 0.0, 0.0, 1.0, 0.0],  // Second query: fourth result is relevant
//! ];
//! let y_score = vec![
//!     array![0.1, 0.9, 0.2, 0.3, 0.4],  // Scores for first query
//!     array![0.5, 0.6, 0.7, 0.9, 0.8],  // Scores for second query
//! ];
//!
//! // Mean Reciprocal Rank
//! let mrr = mean_reciprocal_rank(&y_true, &y_score).unwrap();
//!
//! // Normalized Discounted Cumulative Gain
//! let ndcg = ndcg_score(&y_true, &y_score, Some(5)).unwrap();
//!
//! // Mean Average Precision
//! let map = mean_average_precision(&y_true, &y_score, None).unwrap();
//!
//! // Precision at k=3
//! let precision = precision_at_k(&y_true, &y_score, 3).unwrap();
//!
//! // Recall at k=3
//! let recall = recall_at_k(&y_true, &y_score, 3).unwrap();
//! ```

use ndarray::{ArrayBase, Data, Ix1};
use num_traits::real::Real;
use std::cmp::{Ord, Ordering};

use crate::error::{MetricsError, Result};

pub mod label;

/// Calculates Mean Reciprocal Rank (MRR) for a collection of ranking tasks.
///
/// Mean Reciprocal Rank is the average of the reciprocal ranks of the first
/// relevant items. The reciprocal rank is the multiplicative inverse of the rank
/// of the first relevant item retrieved. If there is no relevant item, the
/// reciprocal rank is 0.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
///
/// # Returns
///
/// * The mean reciprocal rank (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::mean_reciprocal_rank;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 0.0, 0.0],  // First query: second result is relevant
///     array![0.0, 0.0, 0.0, 1.0, 0.0],  // Second query: fourth result is relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.3, 0.4],  // Scores for first query
///     array![0.5, 0.6, 0.7, 0.9, 0.8],  // Scores for second query (0.9 is highest)
/// ];
///
/// // The reciprocal ranks are:
/// // First query: 1/1 = 1.0 (the highest score 0.9 corresponds to the relevant item)
/// // Second query: 1/1 = 1.0 (the highest score 0.9 corresponds to the relevant item)
/// // So the MRR is (1.0 + 1.0) / 2 = 1.0
/// let mrr = mean_reciprocal_rank(&y_true, &y_score).unwrap();
/// assert!((mrr - 1.0).abs() < 1e-10);
/// ```
pub fn mean_reciprocal_rank<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut reciprocal_ranks = Vec::with_capacity(y_true.len());

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Create pairs of (score, relevance) for sorting
        let mut score_relevance: Vec<_> = scores
            .iter()
            .zip(true_relevance.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Find the first relevant item and calculate reciprocal rank
        let zero = T::zero();
        let mut found_relevant = false;
        for (rank, (_, relevance)) in score_relevance.iter().enumerate() {
            if *relevance > zero {
                reciprocal_ranks.push(1.0 / (rank as f64 + 1.0));
                found_relevant = true;
                break;
            }
        }

        // If no relevant items found, use 0 as the reciprocal rank
        if !found_relevant {
            reciprocal_ranks.push(0.0);
        }
    }

    // Mean reciprocal rank
    let mrr = reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64;
    Ok(mrr)
}

/// Helper function to calculate Discounted Cumulative Gain (DCG) for a single query
fn dcg<T>(relevance_scores: &[T], k: Option<usize>) -> f64
where
    T: Real + Clone,
{
    let limit = k
        .unwrap_or(relevance_scores.len())
        .min(relevance_scores.len());

    // DCG formula: sum(rel_i / log2(i+1)) for i=1..k
    (0..limit)
        .map(|i| {
            let rel = relevance_scores[i].to_f64().unwrap_or(0.0);
            // If relevance is binary (0 or 1), we use the standard formula
            // For graded relevance, we can use the alternative formula: (2^rel - 1) / log2(i+2)
            rel / (((i + 2) as f64).log2())
        })
        .sum()
}

/// Calculates Normalized Discounted Cumulative Gain (NDCG) at k
///
/// Normalized Discounted Cumulative Gain (NDCG) measures the ranking quality.
/// It normalizes the Discounted Cumulative Gain (DCG) by the Ideal DCG (IDCG),
/// which is the DCG of the perfect ranking.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Optional number of top elements to consider, None means consider all
///
/// # Returns
///
/// * The NDCG score (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::ndcg_score;
///
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 0.0, 0.0],  // First query: second result is relevant
///     array![0.0, 0.0, 0.0, 1.0, 0.0],  // Second query: fourth result is relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.3, 0.4],  // Scores for first query
///     array![0.5, 0.6, 0.7, 0.9, 0.8],  // Scores for second query
/// ];
///
/// // NDCG at k=5 (all results)
/// let ndcg = ndcg_score(&y_true, &y_score, Some(5)).unwrap();
/// ```
pub fn ndcg_score<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: Option<usize>,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut ndcg_scores = Vec::with_capacity(y_true.len());

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Extract relevance values and scores into vectors for easier sorting
        let relevance_vec: Vec<_> = true_relevance.iter().cloned().collect();
        let scores_vec: Vec<_> = scores.iter().cloned().collect();

        // Create pairs of (score, relevance) for sorting by score
        let mut score_relevance: Vec<_> = scores_vec
            .iter()
            .zip(relevance_vec.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Extract relevance values in score-sorted order
        let sorted_relevance: Vec<_> = score_relevance.iter().map(|(_, r)| r.clone()).collect();

        // Sort relevance values for ideal DCG calculation (descending order)
        let mut ideal_relevance = relevance_vec.clone();
        ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Calculate DCG (Discounted Cumulative Gain)
        let dcg_val = dcg(&sorted_relevance, k);

        // Calculate IDCG (Ideal Discounted Cumulative Gain)
        let idcg_val = dcg(&ideal_relevance, k);

        // Calculate NDCG (Normalized Discounted Cumulative Gain)
        let ndcg = if idcg_val > 0.0 {
            dcg_val / idcg_val
        } else {
            0.0
        };
        ndcg_scores.push(ndcg);
    }

    // Average NDCG across all queries
    let mean_ndcg = ndcg_scores.iter().sum::<f64>() / ndcg_scores.len() as f64;
    Ok(mean_ndcg)
}

/// Helper function to calculate Average Precision for a single query
fn average_precision<T>(y_true_sorted: &[T], k: Option<usize>) -> f64
where
    T: Real + Clone,
{
    let zero = T::zero();
    let k = k.unwrap_or(y_true_sorted.len());
    let limit = k.min(y_true_sorted.len());

    // Calculate the total number of relevant items
    let total_relevant = y_true_sorted.iter().filter(|&&r| r > zero).count();

    if total_relevant == 0 {
        return 0.0;
    }

    let mut cum_sum = 0.0;
    let mut running_sum = 0.0;

    for i in 0..limit {
        let rel = y_true_sorted[i].clone();
        if rel > zero {
            // Count of relevant items found so far
            running_sum += 1.0;
            // Precision at current position
            let precision = running_sum / (i as f64 + 1.0);
            cum_sum += precision;
        }
    }

    cum_sum / total_relevant as f64
}

/// Calculates Mean Average Precision (MAP) for a collection of ranking tasks.
///
/// Mean Average Precision is the mean of the average precision scores for each query.
/// It summarizes a ranking system's precision at different recall levels.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Optional number of top elements to consider, None means consider all
///
/// # Returns
///
/// * The mean average precision score (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::mean_average_precision;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 1.0, 0.0],  // First query: second and fourth results are relevant
///     array![0.0, 0.0, 1.0, 1.0, 0.0],  // Second query: third and fourth results are relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.8, 0.3],  // Scores for first query
///     array![0.3, 0.5, 0.9, 0.8, 0.1],  // Scores for second query
/// ];
///
/// // MAP at k=5 (all results)
/// let map = mean_average_precision(&y_true, &y_score, None).unwrap();
/// ```
pub fn mean_average_precision<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: Option<usize>,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut ap_scores = Vec::with_capacity(y_true.len());

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Create pairs of (score, relevance) for sorting
        let mut score_relevance: Vec<_> = scores
            .iter()
            .zip(true_relevance.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Extract relevance values in score-sorted order
        let sorted_relevance: Vec<_> = score_relevance.iter().map(|(_, r)| r.clone()).collect();

        // Calculate average precision for this query
        let ap = average_precision(&sorted_relevance, k);
        ap_scores.push(ap);
    }

    // Calculate the mean average precision
    let map = ap_scores.iter().sum::<f64>() / ap_scores.len() as f64;
    Ok(map)
}

/// Calculates precision at k for a collection of ranking tasks.
///
/// Precision at k is the proportion of relevant items among the first k
/// retrieved items.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Number of top elements to consider
///
/// # Returns
///
/// * The average precision at k score (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::precision_at_k;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 1.0, 0.0],  // First query: second and fourth results are relevant
///     array![0.0, 0.0, 1.0, 1.0, 0.0],  // Second query: third and fourth results are relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.8, 0.3],  // Scores for first query
///     array![0.3, 0.5, 0.9, 0.8, 0.1],  // Scores for second query
/// ];
///
/// // Precision at k=3
/// let prec = precision_at_k(&y_true, &y_score, 3).unwrap();
/// ```
pub fn precision_at_k<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: usize,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut precision_scores = Vec::with_capacity(y_true.len());
    let zero = T::zero();

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Create pairs of (score, relevance) for sorting
        let mut score_relevance: Vec<_> = scores
            .iter()
            .zip(true_relevance.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Get top k results (or all if k > length)
        let limit = k.min(score_relevance.len());

        // Count relevant items in top k
        let relevant_count = score_relevance[0..limit]
            .iter()
            .filter(|(_, r)| r > &zero)
            .count();

        let precision = relevant_count as f64 / limit as f64;
        precision_scores.push(precision);
    }

    // Calculate the average precision at k
    let avg_precision = precision_scores.iter().sum::<f64>() / precision_scores.len() as f64;
    Ok(avg_precision)
}

/// Calculates recall at k for a collection of ranking tasks.
///
/// Recall at k is the proportion of relevant items that are retrieved among
/// the top k items from the total number of relevant items.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Number of top elements to consider
///
/// # Returns
///
/// * The average recall at k score (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::recall_at_k;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 1.0, 0.0],  // First query: second and fourth results are relevant
///     array![0.0, 0.0, 1.0, 1.0, 0.0],  // Second query: third and fourth results are relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.8, 0.3],  // Scores for first query
///     array![0.3, 0.5, 0.9, 0.8, 0.1],  // Scores for second query
/// ];
///
/// // Recall at k=3
/// let rec = recall_at_k(&y_true, &y_score, 3).unwrap();
/// ```
pub fn recall_at_k<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: usize,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut recall_scores = Vec::with_capacity(y_true.len());
    let zero = T::zero();

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Count total relevant items
        let total_relevant = true_relevance.iter().filter(|&&r| r > zero).count();

        if total_relevant == 0 {
            // If there are no relevant items, recall is defined as 0
            recall_scores.push(0.0);
            continue;
        }

        // Create pairs of (score, relevance) for sorting
        let mut score_relevance: Vec<_> = scores
            .iter()
            .zip(true_relevance.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Get top k results (or all if k > length)
        let limit = k.min(score_relevance.len());

        // Count relevant items in top k
        let retrieved_relevant = score_relevance[0..limit]
            .iter()
            .filter(|(_, r)| r > &zero)
            .count();

        let recall = retrieved_relevant as f64 / total_relevant as f64;
        recall_scores.push(recall);
    }

    // Calculate the average recall at k
    let avg_recall = recall_scores.iter().sum::<f64>() / recall_scores.len() as f64;
    Ok(avg_recall)
}

/// Calculates Kendall's tau rank correlation coefficient.
///
/// Kendall's tau measures the ordinal association between two measured quantities.
/// It measures the similarity of the orderings when ranked by each quantity.
/// A value of 1 indicates perfect agreement, -1 indicates perfect disagreement,
/// and 0 indicates no relationship.
///
/// # Arguments
///
/// * `x` - First ranking array
/// * `y` - Second ranking array
///
/// # Returns
///
/// * Kendall's tau coefficient (float between -1 and 1)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::kendalls_tau;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// // Perfect disagreement -> tau = -1.0
/// let tau = kendalls_tau(&x, &y).unwrap();
/// assert!((tau + 1.0).abs() < 1e-10);
/// ```
pub fn kendalls_tau<T, S, R>(x: &ArrayBase<S, Ix1>, y: &ArrayBase<R, Ix1>) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if x.len() != y.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;

    // Compare each pair of observations
    for i in 0..n {
        for j in (i + 1)..n {
            let x_i = x[i].to_f64().unwrap();
            let x_j = x[j].to_f64().unwrap();
            let y_i = y[i].to_f64().unwrap();
            let y_j = y[j].to_f64().unwrap();

            // Check if concordant or discordant
            let x_diff = x_j - x_i;
            let y_diff = y_j - y_i;
            let product = x_diff * y_diff;

            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            // If product is 0, it's a tie on either x or y, which we don't count
        }
    }

    // Calculate total number of pairs
    let n_pairs = n * (n - 1) / 2;

    // Handle the case where all pairs are ties
    if n_pairs == 0 {
        return Ok(0.0);
    }

    // Calculate Kendall's tau
    let tau = (concordant as f64 - discordant as f64) / n_pairs as f64;
    Ok(tau)
}

/// Calculates Spearman's rank correlation coefficient.
///
/// Spearman's rank correlation assesses how well the relationship between two variables
/// can be described using a monotonic function. It's the Pearson correlation between rank variables.
/// A value of 1 indicates perfect positive correlation, -1 indicates perfect negative correlation,
/// and 0 indicates no correlation.
///
/// # Arguments
///
/// * `x` - First array
/// * `y` - Second array
///
/// # Returns
///
/// * Spearman's rho coefficient (float between -1 and 1)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::spearmans_rho;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// // Perfect negative correlation -> rho = -1.0
/// let rho = spearmans_rho(&x, &y).unwrap();
/// assert!((rho + 1.0).abs() < 1e-10);
/// ```
pub fn spearmans_rho<T, S, R>(x: &ArrayBase<S, Ix1>, y: &ArrayBase<R, Ix1>) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if x.len() != y.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n = x.len();

    // Convert values to ranks
    let x_ranks = rank_data(x)?;
    let y_ranks = rank_data(y)?;

    // Compute the sum of squared rank differences
    let mut d_squared_sum = 0.0;
    for i in 0..n {
        let d = x_ranks[i] - y_ranks[i];
        d_squared_sum += d * d;
    }

    // Compute Spearman's rho
    let rho = 1.0 - (6.0 * d_squared_sum) / (n as f64 * (n * n - 1) as f64);
    Ok(rho)
}

/// Helper function to convert values to ranks.
/// Handles ties by assigning average rank.
fn rank_data<T, S>(x: &ArrayBase<S, Ix1>) -> Result<Vec<f64>>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
{
    let n = x.len();

    // Create pairs of (value, original index)
    let mut value_index: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, val)| (val.to_f64().unwrap_or(0.0), i))
        .collect();

    // Sort by value
    value_index.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Assign ranks (handling ties)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let val = value_index[i].0;
        let mut j = i + 1;

        // Find all ties with the current value
        while j < n && (value_index[j].0 - val).abs() < 1e-10 {
            j += 1;
        }

        // If there are ties, assign average rank
        if j > i + 1 {
            // The average rank for tied values: (start_rank + end_rank) / 2
            let rank_avg = (2 * i + j - 1) as f64 / 2.0 + 1.0;

            for k in i..j {
                let idx = value_index[k].1;
                ranks[idx] = rank_avg;
            }
        } else {
            // No ties, assign standard rank (i+1)
            let idx = value_index[i].1;
            ranks[idx] = (i + 1) as f64;
        }

        i = j;
    }

    Ok(ranks)
}

/// Calculates Mean Average Precision at k (MAP@k) for a collection of ranking tasks.
///
/// MAP@k is a variant of MAP that only considers the top k ranked items for each query.
/// It's commonly used in search and recommendation system evaluation.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Number of top elements to consider (required)
///
/// # Returns
///
/// * The mean average precision at k score (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::map_at_k;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 1.0, 0.0],  // First query: second and fourth results are relevant
///     array![0.0, 0.0, 1.0, 1.0, 0.0],  // Second query: third and fourth results are relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.8, 0.3],  // Scores for first query
///     array![0.3, 0.5, 0.9, 0.8, 0.1],  // Scores for second query
/// ];
///
/// // MAP@3 - consider only top 3 ranked items
/// let map_k = map_at_k(&y_true, &y_score, 3).unwrap();
/// ```
pub fn map_at_k<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: usize,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that k is greater than 0
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    // Use the existing mean_average_precision function with k specified
    mean_average_precision(y_true, y_score, Some(k))
}

/// Calculates Click-Through Rate (CTR) for a collection of ranking tasks.
///
/// Click-Through Rate is the ratio of users who click on a specific link to the
/// number of total users who view a page, email, or advertisement. In the context
/// of ranking, it estimates the likelihood that a user will click on a retrieved item.
///
/// # Arguments
///
/// * `y_true` - Ground truth relevance scores or binary relevance indicators
/// * `y_score` - Predicted relevance scores for ranking
/// * `k` - Number of top elements to consider
///
/// # Returns
///
/// * The average click-through rate (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::ranking::click_through_rate;
///
/// // Example: search engine results where each array is a different query
/// // Values indicate whether a result is relevant (1.0) or not (0.0)
/// let y_true = vec![
///     array![0.0, 1.0, 0.0, 1.0, 0.0],  // First query: second and fourth results are relevant
///     array![0.0, 0.0, 1.0, 1.0, 0.0],  // Second query: third and fourth results are relevant
/// ];
/// let y_score = vec![
///     array![0.1, 0.9, 0.2, 0.8, 0.3],  // Scores for first query
///     array![0.3, 0.5, 0.9, 0.8, 0.1],  // Scores for second query
/// ];
///
/// // CTR at k=3
/// let ctr = click_through_rate(&y_true, &y_score, 3).unwrap();
/// ```
pub fn click_through_rate<T, S, R>(
    y_true: &[ArrayBase<S, Ix1>],
    y_score: &[ArrayBase<R, Ix1>],
    k: usize,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    // CTR is essentially the probability that a user will click on a retrieved item
    // We'll use a simple position-based model with exponential decay

    let mut ctr_scores = Vec::with_capacity(y_true.len());
    let zero = T::zero();

    for (true_relevance, scores) in y_true.iter().zip(y_score.iter()) {
        if true_relevance.shape() != scores.shape() {
            return Err(MetricsError::InvalidInput(format!(
                "Relevance and score arrays have different shapes: {:?} vs {:?}",
                true_relevance.shape(),
                scores.shape()
            )));
        }

        // Create pairs of (score, relevance) for sorting
        let mut score_relevance: Vec<_> = scores
            .iter()
            .zip(true_relevance.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Get top k results (or all if k > length)
        let limit = k.min(score_relevance.len());

        // Calculate position-based CTR
        let mut query_ctr = 0.0;
        for i in 0..limit {
            // Position bias: probability of examining item at position i
            // Using a simple exponential decay model: p(i) = 1/(i+1)
            let position_bias = 1.0 / (i as f64 + 1.0);

            // Relevance is the probability of clicking if examined
            let relevance = if score_relevance[i].1 > zero {
                1.0
            } else {
                0.0
            };

            // CTR = P(examine) * P(click | examine)
            query_ctr += position_bias * relevance;
        }

        // Normalize by the number of positions examined
        let normalizer = (1..=limit).map(|i| 1.0 / i as f64).sum::<f64>();
        if normalizer > 0.0 {
            query_ctr /= normalizer;
        }

        ctr_scores.push(query_ctr);
    }

    // Calculate the average CTR
    let avg_ctr = ctr_scores.iter().sum::<f64>() / ctr_scores.len() as f64;
    Ok(avg_ctr)
}
