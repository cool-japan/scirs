//! Label Ranking Metrics module
//!
//! This module provides metrics for evaluating multi-label ranking problems
//! where each sample can have multiple correct labels. These metrics are
//! particularly useful for tasks like multi-label classification and
//! recommendation systems.
//!
//! ## Main Metrics
//!
//! - `coverage_error`: Computes how far we need to go in the ranked list to cover all relevant labels
//! - `label_ranking_loss`: Computes the ranking loss which averages over samples the number of
//!   incorrectly ordered label pairs
//! - `label_ranking_average_precision_score`: Computes the average precision score for label ranking

use scirs2_core::ndarray::{ArrayBase, Data, Ix1, Ix2};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;

use crate::error::{MetricsError, Result};

/// Calculates the coverage error for multi-label ranking problems.
///
/// Coverage error measures the average number of labels that need to be included in the
/// final prediction for all true labels to be predicted. This is useful for measuring
/// how far we need to go down the ranked list of labels to cover all true labels.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary label array
/// * `y_score` - Score array
///
/// # Returns
///
/// * The coverage error (best value is y_true.cardinality(), worst is n_labels)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::ranking::label::coverage_error;
///
/// // 1 indicates a relevant label
/// let y_true = array![1.0, 1.0, 0.0, 0.0, 0.0];
///
/// // Predicted scores for each label
/// let y_score = array![0.9, 0.8, 0.7, 0.6, 0.5];
///
/// // Calculate coverage error
/// let coverage = coverage_error(&y_true, &y_score).unwrap();
/// assert_eq!(coverage, 2.0); // Since the last relevant label is at index 1
/// ```
#[allow(dead_code)]
pub fn coverage_error<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_score: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_score.shape()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check if y_true is binary (only contains 0 and 1)
    let zero = T::zero();
    let one = T::one();
    let is_binary = y_true.iter().all(|&x| x == zero || x == one);

    if !is_binary {
        return Err(MetricsError::InvalidInput(
            "y_true should be binary (0 or 1 values only)".to_string(),
        ));
    }

    // Count _true labels
    let n_true = y_true.iter().filter(|&&x| x > zero).count();

    if n_true == 0 {
        return Err(MetricsError::InvalidInput(
            "No positive labels in y_true".to_string(),
        ));
    }

    // Create pairs of (_score, relevance, index) for sorting
    let mut score_relevance_idx: Vec<_> = y_score
        .iter()
        .zip(y_true.iter())
        .enumerate()
        .map(|(idx, (s, r))| (s.clone(), r.clone(), idx))
        .collect();

    // Sort by _score in descending order
    score_relevance_idx.sort_by(|(a, _, _), (b, _, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // Find position of last relevant item
    let mut relevant_found = 0;
    for (i, (_, rel, _)) in score_relevance_idx.iter().enumerate() {
        if *rel > zero {
            relevant_found += 1;
            if relevant_found == n_true {
                return Ok((i + 1) as f64);
            }
        }
    }

    // This should not happen if we've properly counted n_true above
    Ok(y_true.len() as f64)
}

/// Calculates coverage error for multi-label ranking problems with multiple samples.
///
/// This function handles a batch of samples, computing the average coverage error
/// across all samples in the dataset.
///
/// # Arguments
///
/// * `y_true` - Array of ground truth multi-label indicators with shape (n_samples, n_labels)
/// * `y_score` - Array of predicted scores with shape (n_samples, n_labels)
///
/// # Returns
///
/// * The average coverage error across all samples
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_metrics::ranking::label::coverage_error_multiple;
///
/// // Example: 3 samples, 5 possible labels
/// // Each row is a sample, each column is a label
/// // 1 indicates a relevant label for that sample
/// let y_true = Array2::from_shape_vec((3, 5), vec![
///     1.0, 0.0, 1.0, 0.0, 0.0,  // Sample 1: labels 0 and 2 are relevant
///     0.0, 0.0, 1.0, 1.0, 0.0,  // Sample 2: labels 2 and 3 are relevant
///     0.0, 1.0, 1.0, 0.0, 1.0,  // Sample 3: labels 1, 2, and 4 are relevant
/// ]).unwrap();
///
/// // Predicted scores for each label
/// let y_score = Array2::from_shape_vec((3, 5), vec![
///     0.9, 0.2, 0.8, 0.3, 0.1,  // Scores for sample 1
///     0.2, 0.3, 0.9, 0.7, 0.1,  // Scores for sample 2
///     0.1, 0.9, 0.8, 0.2, 0.7,  // Scores for sample 3
/// ]).unwrap();
///
/// // Calculate average coverage error
/// let coverage = coverage_error_multiple(&y_true, &y_score).unwrap();
/// ```
#[allow(dead_code)]
pub fn coverage_error_multiple<T, S, R>(
    y_true: &ArrayBase<S, Ix2>,
    y_score: &ArrayBase<R, Ix2>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_score.shape()
        )));
    }

    let n_samples = y_true.shape()[0];
    let n_labels = y_true.shape()[1];

    if n_samples == 0 || n_labels == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let mut total_coverage = 0.0;
    let mut valid_samples = 0;

    // Process each sample
    for i in 0..n_samples {
        // Extract one sample's _true labels and scores
        let sample_true = y_true.slice(scirs2_core::ndarray::s![i, ..]);
        let sample_score = y_score.slice(scirs2_core::ndarray::s![i, ..]);

        // Count _true labels for this sample
        let true_count = sample_true.iter().filter(|&&x| x > T::zero()).count();

        // Skip samples with no positive labels
        if true_count == 0 {
            continue;
        }

        // Create pairs of (_score, relevance) for sorting
        let mut score_relevance: Vec<_> = sample_score
            .iter()
            .zip(sample_true.iter())
            .map(|(s, r)| (s.clone(), r.clone()))
            .collect();

        // Sort by _score in descending order
        score_relevance.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Find position of last relevant item
        let mut last_relevant_pos = 0;
        for (j, (_, rel)) in score_relevance.iter().enumerate() {
            if *rel > T::zero() {
                last_relevant_pos = j;
            }
        }

        // Coverage is the position of the last relevant item plus 1
        total_coverage += (last_relevant_pos + 1) as f64;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No samples with positive labels found".to_string(),
        ));
    }

    Ok(total_coverage / valid_samples as f64)
}

/// Calculates the label ranking loss for multi-label ranking problems.
///
/// The label ranking loss averages over the samples the number of label pairs
/// that are incorrectly ordered (where an irrelevant label is ranked higher
/// than a relevant label). The best score is 0.
///
/// # Arguments
///
/// * `y_true` - Ground truth multi-label indicator matrix or binary label array
/// * `y_score` - Score matrix with shape (n_samples, n_labels)
///
/// # Returns
///
/// * The ranking loss (0 is the best score, 1 is the worst)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::ranking::label::label_ranking_loss;
///
/// // Example: 3 samples, 5 possible labels
/// // Each row is a sample, each column is a label
/// // 1 indicates a relevant label for that sample
/// let y_true = Array2::from_shape_vec((3, 5), vec![
///     1.0, 0.0, 1.0, 0.0, 0.0,  // Sample 1: labels 0 and 2 are relevant
///     0.0, 0.0, 1.0, 1.0, 0.0,  // Sample 2: labels 2 and 3 are relevant
///     0.0, 1.0, 1.0, 0.0, 1.0,  // Sample 3: labels 1, 2, and 4 are relevant
/// ]).unwrap();
///
/// // Predicted scores for each label
/// let y_score = Array2::from_shape_vec((3, 5), vec![
///     0.9, 0.2, 0.8, 0.3, 0.1,  // Scores for sample 1
///     0.2, 0.3, 0.9, 0.7, 0.1,  // Scores for sample 2
///     0.1, 0.9, 0.8, 0.2, 0.7,  // Scores for sample 3
/// ]).unwrap();
///
/// // Calculate label ranking loss
/// let loss = label_ranking_loss(&y_true, &y_score).unwrap();
/// ```
#[allow(dead_code)]
pub fn label_ranking_loss<T, S, R>(
    y_true: &ArrayBase<S, Ix2>,
    y_score: &ArrayBase<R, Ix2>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_score.shape()
        )));
    }

    let n_samples = y_true.shape()[0];
    let n_labels = y_true.shape()[1];

    if n_samples == 0 || n_labels == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let mut ranking_loss_sum = 0.0;
    let mut valid_samples = 0;

    // Process each sample
    for i in 0..n_samples {
        // Extract one sample's _true labels and scores
        let sample_true = y_true.slice(scirs2_core::ndarray::s![i, ..]);
        let sample_score = y_score.slice(scirs2_core::ndarray::s![i, ..]);

        // Get relevant and irrelevant label indices
        let mut relevant_labels = Vec::new();
        let mut irrelevant_labels = Vec::new();

        for j in 0..n_labels {
            if sample_true[j] > zero {
                relevant_labels.push(j);
            } else {
                irrelevant_labels.push(j);
            }
        }

        // Skip samples with all labels relevant or all irrelevant
        if relevant_labels.is_empty() || irrelevant_labels.is_empty() {
            continue;
        }

        // Count incorrectly ordered pairs
        let mut n_incorrect = 0;
        for &r_idx in &relevant_labels {
            for &i_idx in &irrelevant_labels {
                // Check if irrelevant label _score >= relevant label _score
                if sample_score[i_idx] >= sample_score[r_idx] {
                    n_incorrect += 1;
                }
            }
        }

        // Normalize by the number of possible pairs
        let n_pairs = relevant_labels.len() * irrelevant_labels.len();
        ranking_loss_sum += n_incorrect as f64 / n_pairs as f64;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid samples found for ranking loss calculation".to_string(),
        ));
    }

    // Calculate average ranking loss
    Ok(ranking_loss_sum / valid_samples as f64)
}

/// Calculates the label ranking average precision score for multi-label ranking problems.
///
/// This score is the average over each ground truth label, of the ratio of true positives
/// to the sum of true positives and false positives up to that position in the ranking.
/// The best score is 1.0.
///
/// # Arguments
///
/// * `y_true` - Ground truth multi-label indicator matrix
/// * `y_score` - Score matrix with shape (n_samples, n_labels)
///
/// # Returns
///
/// * The label ranking average precision score (1.0 is the best score, 0.0 is the worst)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_metrics::ranking::label::label_ranking_average_precision_score;
///
/// // Example: 3 samples, 5 possible labels
/// // Each row is a sample, each column is a label
/// // 1 indicates a relevant label for that sample
/// let y_true = Array2::from_shape_vec((3, 5), vec![
///     1.0, 0.0, 1.0, 0.0, 0.0,  // Sample 1: labels 0 and 2 are relevant
///     0.0, 0.0, 1.0, 1.0, 0.0,  // Sample 2: labels 2 and 3 are relevant
///     0.0, 1.0, 1.0, 0.0, 1.0,  // Sample 3: labels 1, 2, and 4 are relevant
/// ]).unwrap();
///
/// // Predicted scores for each label
/// let y_score = Array2::from_shape_vec((3, 5), vec![
///     0.9, 0.2, 0.8, 0.3, 0.1,  // Scores for sample 1
///     0.2, 0.3, 0.9, 0.7, 0.1,  // Scores for sample 2
///     0.1, 0.9, 0.8, 0.2, 0.7,  // Scores for sample 3
/// ]).unwrap();
///
/// // Calculate label ranking average precision score
/// let score = label_ranking_average_precision_score(&y_true, &y_score).unwrap();
/// ```
#[allow(dead_code)]
pub fn label_ranking_average_precision_score<T, S, R>(
    y_true: &ArrayBase<S, Ix2>,
    y_score: &ArrayBase<R, Ix2>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_score.shape()
        )));
    }

    let n_samples = y_true.shape()[0];
    let n_labels = y_true.shape()[1];

    if n_samples == 0 || n_labels == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let mut lrap_sum = 0.0;
    let mut valid_samples = 0;

    // Process each sample
    for i in 0..n_samples {
        // Extract one sample's _true labels and scores
        let sample_true = y_true.slice(scirs2_core::ndarray::s![i, ..]);
        let sample_score = y_score.slice(scirs2_core::ndarray::s![i, ..]);

        // Count relevant labels for this sample
        let relevant_count = sample_true.iter().filter(|&&x| x > zero).count();

        // Skip samples with no relevant labels
        if relevant_count == 0 {
            continue;
        }

        // Create pairs of (_score, relevance, index) for sorting
        let mut score_relevance_idx: Vec<_> = sample_score
            .iter()
            .zip(sample_true.iter())
            .enumerate()
            .map(|(idx, (s, r))| (s.clone(), r.clone(), idx))
            .collect();

        // Sort by _score in descending order
        score_relevance_idx
            .sort_by(|(a, _, _), (b, _, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Calculate precision at each relevant position
        let mut precision_sum = 0.0;
        let mut n_relevant_retrieved = 0;

        for (rank, (_, rel, _)) in score_relevance_idx.iter().enumerate() {
            if *rel > zero {
                n_relevant_retrieved += 1;
                precision_sum += n_relevant_retrieved as f64 / (rank + 1) as f64;
            }
        }

        // Normalize by the number of relevant labels
        lrap_sum += precision_sum / relevant_count as f64;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "No valid samples found for calculation".to_string(),
        ));
    }

    Ok(lrap_sum / valid_samples as f64)
}
