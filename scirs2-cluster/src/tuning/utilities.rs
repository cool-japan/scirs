//! Utility functions for hyperparameter tuning
//!
//! This module contains helper functions for scoring, early stopping,
//! statistics calculation, and other common operations.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::Result;

use super::config::*;

/// Calculate standard deviation of a vector of scores
pub fn calculate_std_dev(scores: &[f64]) -> f64 {
    if scores.len() <= 1 {
        return 0.0;
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance =
        scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64;
    variance.sqrt()
}

/// Check if a score is better than the current best based on the metric
pub fn is_score_better(new_score: f64, best_score: f64, metric: &EvaluationMetric) -> bool {
    match metric {
        EvaluationMetric::SilhouetteScore
        | EvaluationMetric::CalinskiHarabaszIndex
        | EvaluationMetric::AdjustedRandIndex => new_score > best_score,
        EvaluationMetric::DaviesBouldinIndex | EvaluationMetric::Inertia => {
            new_score < best_score || best_score == f64::NEG_INFINITY
        }
        _ => new_score > best_score,
    }
}

/// Check if early stopping criteria are met
pub fn should_stop_early(
    evaluation_history: &[EvaluationResult],
    early_stop_config: &EarlyStoppingConfig,
) -> bool {
    if evaluation_history.len() < early_stop_config.patience {
        return false;
    }

    let recent_evaluations =
        &evaluation_history[evaluation_history.len() - early_stop_config.patience..];
    let best_recent = recent_evaluations
        .iter()
        .map(|r| r.score)
        .fold(f64::NEG_INFINITY, f64::max);

    let current_best = evaluation_history
        .iter()
        .map(|r| r.score)
        .fold(f64::NEG_INFINITY, f64::max);

    (current_best - best_recent) < early_stop_config.min_improvement
}

/// Create convergence information based on evaluation history
pub fn create_convergence_info(
    evaluation_history: &[EvaluationResult],
    max_evaluations: usize,
) -> ConvergenceInfo {
    ConvergenceInfo {
        converged: evaluation_history.len() >= max_evaluations,
        convergence_iteration: None,
        stopping_reason: if evaluation_history.len() >= max_evaluations {
            StoppingReason::MaxEvaluations
        } else {
            StoppingReason::EarlyStopping
        },
    }
}

/// Calculate exploration statistics from evaluation history
pub fn calculate_exploration_stats(evaluation_history: &[EvaluationResult]) -> ExplorationStats {
    let mut parameter_distributions = HashMap::new();
    let mut parameter_importance = HashMap::new();

    // Collect parameter distributions
    for result in evaluation_history {
        for (param_name, &value) in &result.parameters {
            parameter_distributions
                .entry(param_name.clone())
                .or_insert_with(Vec::new)
                .push(value);
        }
    }

    // Calculate parameter importance (simplified correlation)
    for (param_name, values) in &parameter_distributions {
        let scores: Vec<f64> = evaluation_history.iter().map(|r| r.score).collect();
        let correlation = calculate_correlation(values, &scores);
        parameter_importance.insert(param_name.clone(), correlation.abs());
    }

    ExplorationStats {
        coverage: calculate_coverage(&parameter_distributions),
        parameter_distributions,
        parameter_importance,
    }
}

/// Calculate correlation between two vectors
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();
    let sum_y_sq: f64 = y.iter().map(|a| a * a).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Calculate parameter space coverage (simplified)
fn calculate_coverage(parameter_distributions: &HashMap<String, Vec<f64>>) -> f64 {
    if parameter_distributions.is_empty() {
        return 0.0;
    }

    let mut total_coverage = 0.0;

    for values in parameter_distributions.values() {
        if values.len() <= 1 {
            continue;
        }

        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range > 0.0 {
            // Simple coverage metric based on unique values
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

            let coverage = sorted_values.len() as f64 / values.len() as f64;
            total_coverage += coverage;
        }
    }

    total_coverage / parameter_distributions.len() as f64
}

/// Calculate inertia (within-cluster sum of squares)
pub fn calculate_inertia(
    data: &Array2<f64>,
    labels: &Array1<usize>,
    centroids: &Array2<f64>,
) -> Result<f64> {
    let mut total_inertia = 0.0;

    for (i, &label) in labels.iter().enumerate() {
        if label >= centroids.nrows() {
            continue; // Skip invalid labels
        }

        let mut distance_sq = 0.0;
        for j in 0..data.ncols() {
            let diff = data[[i, j]] - centroids[[label, j]];
            distance_sq += diff * diff;
        }
        total_inertia += distance_sq;
    }

    Ok(total_inertia)
}

/// Calculate silhouette coefficient for a single sample
pub fn calculate_sample_silhouette(
    data: &Array2<f64>,
    labels: &Array1<i32>,
    sample_idx: usize,
) -> f64 {
    let sample_label = labels[sample_idx];
    let n_samples = data.nrows();

    if n_samples <= 1 {
        return 0.0;
    }

    // Calculate a(i) - mean distance to samples in same cluster
    let mut intra_cluster_distances = Vec::new();
    for j in 0..n_samples {
        if j != sample_idx && labels[j] == sample_label {
            let distance = euclidean_distance(&data.row(sample_idx), &data.row(j));
            intra_cluster_distances.push(distance);
        }
    }

    let a_i = if intra_cluster_distances.is_empty() {
        0.0
    } else {
        intra_cluster_distances.iter().sum::<f64>() / intra_cluster_distances.len() as f64
    };

    // Calculate b(i) - mean distance to nearest cluster
    let unique_labels: Vec<i32> = {
        let mut labels_vec: Vec<i32> = labels.iter().copied().collect();
        labels_vec.sort();
        labels_vec.dedup();
        labels_vec
    };

    let mut min_inter_cluster_distance = f64::INFINITY;

    for &other_label in &unique_labels {
        if other_label == sample_label {
            continue;
        }

        let mut inter_cluster_distances = Vec::new();
        for j in 0..n_samples {
            if labels[j] == other_label {
                let distance = euclidean_distance(&data.row(sample_idx), &data.row(j));
                inter_cluster_distances.push(distance);
            }
        }

        if !inter_cluster_distances.is_empty() {
            let mean_distance =
                inter_cluster_distances.iter().sum::<f64>() / inter_cluster_distances.len() as f64;
            min_inter_cluster_distance = min_inter_cluster_distance.min(mean_distance);
        }
    }

    let b_i = if min_inter_cluster_distance == f64::INFINITY {
        0.0
    } else {
        min_inter_cluster_distance
    };

    // Calculate silhouette coefficient
    if a_i == 0.0 && b_i == 0.0 {
        0.0
    } else {
        (b_i - a_i) / a_i.max(b_i)
    }
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
pub fn manhattan_distance(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
}

/// Calculate Cosine similarity between two vectors
pub fn cosine_similarity(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Normalize a vector to unit length
pub fn normalize_vector(vector: &mut [f64]) {
    let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

/// Generate linearly spaced values between start and stop
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num <= 1 {
        return vec![start];
    }

    let step = (stop - start) / (num - 1) as f64;
    (0..num).map(|i| start + i as f64 * step).collect()
}

/// Generate logarithmically spaced values between start and stop
pub fn logspace(start: f64, stop: f64, num: usize, base: f64) -> Vec<f64> {
    let linear_values = linspace(start, stop, num);
    linear_values.iter().map(|x| base.powf(*x)).collect()
}

/// Calculate percentile of a sorted vector
pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Calculate median of a vector
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    percentile(&sorted_data, 50.0)
}

/// Calculate interquartile range (IQR)
pub fn iqr(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q75 = percentile(&sorted_data, 75.0);
    let q25 = percentile(&sorted_data, 25.0);

    q75 - q25
}

/// Check if a value is an outlier using the IQR method
pub fn is_outlier(value: f64, data: &[f64], multiplier: f64) -> bool {
    if data.len() < 4 {
        return false;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q25 = percentile(&sorted_data, 25.0);
    let q75 = percentile(&sorted_data, 75.0);
    let iqr_value = q75 - q25;

    let lower_bound = q25 - multiplier * iqr_value;
    let upper_bound = q75 + multiplier * iqr_value;

    value < lower_bound || value > upper_bound
}

/// Generate hash of parameter configuration for caching
pub fn hash_parameters(params: &HashMap<String, f64>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Sort parameters for consistent hashing
    let mut sorted_params: Vec<_> = params.iter().collect();
    sorted_params.sort_by_key(|(k, _)| *k);

    for (key, value) in sorted_params {
        key.hash(&mut hasher);
        // Hash the bits of the float for deterministic results
        value.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_std_dev() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = calculate_std_dev(&scores);
        assert!((std_dev - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        use scirs2_core::ndarray::Array1;
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        let distance = euclidean_distance(&a.view(), &b.view());
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = percentile(&data, 50.0);
        assert!((p50 - 3.0).abs() < 1e-10);
    }
}
