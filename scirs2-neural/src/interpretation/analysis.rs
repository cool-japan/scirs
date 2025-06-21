//! Layer analysis and statistical evaluation for neural network interpretation
//!
//! This module provides functionality for analyzing layer activations, computing
//! statistical metrics, and evaluating attribution consistency across different
//! interpretation methods.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

use super::core::ModelInterpreter;

/// Statistical analysis of layer activations
#[derive(Debug, Clone)]
pub struct LayerAnalysisStats<F: Float + Debug> {
    /// Mean activation value
    pub mean_activation: F,
    /// Standard deviation of activations
    pub std_activation: F,
    /// Maximum activation value
    pub max_activation: F,
    /// Minimum activation value
    pub min_activation: F,
    /// Percentage of dead neurons (always zero)
    pub dead_neuron_percentage: f64,
    /// Sparsity (percentage of near-zero activations)
    pub sparsity: f64,
    /// Activation distribution histogram
    pub histogram: Vec<u32>,
    /// Histogram bin edges
    pub bin_edges: Vec<F>,
}

/// Attribution statistics for a single method
#[derive(Debug, Clone)]
pub struct AttributionStatistics<F: Float + Debug> {
    /// Mean attribution value
    pub mean: F,
    /// Mean absolute attribution value
    pub mean_absolute: F,
    /// Maximum absolute attribution value
    pub max_absolute: F,
    /// Ratio of positive attributions
    pub positive_attribution_ratio: f64,
    /// Sum of all positive attributions
    pub total_positive_attribution: F,
    /// Sum of all negative attributions
    pub total_negative_attribution: F,
}

/// Summary of interpretation analysis across multiple methods
#[derive(Debug, Clone)]
pub struct InterpretationSummary {
    /// Number of attribution methods used
    pub num_attribution_methods: usize,
    /// Average consistency across methods
    pub average_method_consistency: f64,
    /// Indices of most important features
    pub most_important_features: Vec<usize>,
    /// Overall interpretation confidence score
    pub interpretation_confidence: f64,
}

/// Analyze layer activations and compute comprehensive statistics
pub fn analyze_layer_activations<F>(
    interpreter: &mut ModelInterpreter<F>,
    layer_name: &str,
) -> Result<LayerAnalysisStats<F>>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let activations = interpreter
        .get_cached_activations(layer_name)
        .ok_or_else(|| {
            NeuralError::ComputationError(format!(
                "No cached activations found for layer: {}",
                layer_name
            ))
        })?;

    let stats = compute_layer_statistics(activations)?;
    interpreter.cache_layer_statistics(layer_name.to_string(), stats.clone());
    Ok(stats)
}

/// Compute detailed statistics for layer activations
pub fn compute_layer_statistics<F>(activations: &ArrayD<F>) -> Result<LayerAnalysisStats<F>>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let mean_activation = activations.mean().unwrap_or(F::zero());
    let variance = activations
        .mapv(|x| (x - mean_activation) * (x - mean_activation))
        .mean()
        .unwrap_or(F::zero());
    let std_activation = variance.sqrt();

    let max_activation = activations.iter().cloned().fold(F::neg_infinity(), F::max);
    let min_activation = activations.iter().cloned().fold(F::infinity(), F::min);

    // Compute dead neuron percentage (neurons that are always zero)
    let zero_threshold = F::from(1e-6).unwrap();
    let dead_neurons = activations
        .iter()
        .filter(|&&x| x.abs() < zero_threshold)
        .count();
    let dead_neuron_percentage = dead_neurons as f64 / activations.len() as f64 * 100.0;

    // Compute sparsity (percentage of near-zero activations)
    let sparsity_threshold = F::from(0.01).unwrap();
    let sparse_neurons = activations
        .iter()
        .filter(|&&x| x.abs() < sparsity_threshold)
        .count();
    let sparsity = sparse_neurons as f64 / activations.len() as f64 * 100.0;

    // Create histogram
    let num_bins = 50;
    let range = max_activation - min_activation;
    let bin_width = if range > F::zero() {
        range / F::from(num_bins).unwrap()
    } else {
        F::one()
    };

    let mut histogram = vec![0u32; num_bins];
    let mut bin_edges = Vec::with_capacity(num_bins + 1);

    for i in 0..=num_bins {
        bin_edges.push(min_activation + bin_width * F::from(i).unwrap());
    }

    for &val in activations.iter() {
        if val.is_finite() && range > F::zero() {
            let bin_idx = ((val - min_activation) / bin_width).to_usize().unwrap_or(0);
            let bin_idx = bin_idx.min(num_bins - 1);
            histogram[bin_idx] += 1;
        }
    }

    Ok(LayerAnalysisStats {
        mean_activation,
        std_activation,
        max_activation,
        min_activation,
        dead_neuron_percentage,
        sparsity,
        histogram,
        bin_edges,
    })
}

/// Compute attribution statistics for a single attribution method
pub fn compute_attribution_statistics<F>(attribution: &ArrayD<F>) -> AttributionStatistics<F>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let mean = attribution.mean().unwrap_or(F::zero());
    let abs_attribution = attribution.mapv(|x| x.abs());
    let mean_abs = abs_attribution.mean().unwrap_or(F::zero());
    let max_abs = abs_attribution.iter().cloned().fold(F::zero(), F::max);
    let positive_ratio =
        attribution.iter().filter(|&&x| x > F::zero()).count() as f64 / attribution.len() as f64;

    AttributionStatistics {
        mean,
        mean_absolute: mean_abs,
        max_absolute: max_abs,
        positive_attribution_ratio: positive_ratio,
        total_positive_attribution: attribution
            .iter()
            .filter(|&&x| x > F::zero())
            .cloned()
            .sum(),
        total_negative_attribution: attribution
            .iter()
            .filter(|&&x| x < F::zero())
            .cloned()
            .sum(),
    }
}

/// Generate interpretation summary across multiple attribution methods
pub fn generate_interpretation_summary<F>(
    attributions: &HashMap<String, ArrayD<F>>,
) -> InterpretationSummary
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let num_methods = attributions.len();

    // Find most consistent features across methods
    let mut feature_consistency_scores = Vec::new();

    if let Some((_, first_attribution)) = attributions.iter().next() {
        for i in 0..first_attribution.len() {
            let mut scores = Vec::new();
            for attribution in attributions.values() {
                if i < attribution.len() {
                    // Use iter() to access elements properly for multi-dimensional arrays
                    if let Some(value) = attribution.iter().nth(i) {
                        scores.push(value.to_f64().unwrap_or(0.0));
                    }
                }
            }

            // Compute consistency as standard deviation (lower is more consistent)
            if !scores.is_empty() {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                let variance =
                    scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                let std_dev = variance.sqrt();
                feature_consistency_scores.push(1.0 / (1.0 + std_dev)); // Higher is more consistent
            } else {
                feature_consistency_scores.push(0.0);
            }
        }
    }

    let avg_consistency = if !feature_consistency_scores.is_empty() {
        feature_consistency_scores.iter().sum::<f64>() / feature_consistency_scores.len() as f64
    } else {
        0.0
    };

    InterpretationSummary {
        num_attribution_methods: num_methods,
        average_method_consistency: avg_consistency,
        most_important_features: find_most_important_features(attributions, 10),
        interpretation_confidence: compute_interpretation_confidence(attributions),
    }
}

/// Find the most important features based on average attribution magnitude
pub fn find_most_important_features<F>(
    attributions: &HashMap<String, ArrayD<F>>,
    top_k: usize,
) -> Vec<usize>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    if attributions.is_empty() {
        return Vec::new();
    }

    // Average attributions across methods
    let first_attribution = attributions.values().next().unwrap();
    let mut averaged_attribution = Array::zeros(first_attribution.raw_dim());

    for attribution in attributions.values() {
        averaged_attribution = averaged_attribution + attribution;
    }
    averaged_attribution = averaged_attribution / F::from(attributions.len()).unwrap();

    // Find top-k features by absolute importance
    let mut feature_scores: Vec<(usize, f64)> = averaged_attribution
        .iter()
        .enumerate()
        .map(|(i, &score): (usize, &F)| (i, score.abs().to_f64().unwrap_or(0.0)))
        .collect();

    feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    feature_scores
        .into_iter()
        .take(top_k)
        .map(|(i, _)| i)
        .collect()
}

/// Compute interpretation confidence based on method consistency
pub fn compute_interpretation_confidence<F>(attributions: &HashMap<String, ArrayD<F>>) -> f64
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    if attributions.len() < 2 {
        return 1.0; // Single method, assume full confidence
    }

    // Compute pairwise correlations between attribution methods
    let methods: Vec<_> = attributions.keys().collect();
    let mut correlations = Vec::new();

    for i in 0..methods.len() {
        for j in (i + 1)..methods.len() {
            let attr1 = &attributions[methods[i]];
            let attr2 = &attributions[methods[j]];

            if attr1.len() == attr2.len() {
                let correlation = compute_correlation(attr1, attr2);
                correlations.push(correlation);
            }
        }
    }

    // Average correlation as confidence measure
    if !correlations.is_empty() {
        correlations.iter().sum::<f64>() / correlations.len() as f64
    } else {
        0.5
    }
}

/// Compute Pearson correlation coefficient between two attribution arrays
pub fn compute_correlation<F>(x: &ArrayD<F>, y: &ArrayD<F>) -> f64
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let x_mean = x.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_sum_sq = 0.0;
    let mut y_sum_sq = 0.0;

    for (x_val, y_val) in x.iter().zip(y.iter()) {
        let x_diff = x_val.to_f64().unwrap_or(0.0) - x_mean;
        let y_diff = y_val.to_f64().unwrap_or(0.0) - y_mean;

        numerator += x_diff * y_diff;
        x_sum_sq += x_diff * x_diff;
        y_sum_sq += y_diff * y_diff;
    }

    let denominator = (x_sum_sq * y_sum_sq).sqrt();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Analyze activation distribution patterns
pub fn analyze_activation_distribution<F>(stats: &LayerAnalysisStats<F>) -> HashMap<String, f64>
where
    F: Float + Debug,
{
    let mut analysis = HashMap::new();

    analysis.insert(
        "dead_neuron_percentage".to_string(),
        stats.dead_neuron_percentage,
    );
    analysis.insert("sparsity_percentage".to_string(), stats.sparsity);

    // Analyze distribution shape based on histogram
    let total_activations: u32 = stats.histogram.iter().sum();
    if total_activations > 0 {
        // Find peak bin
        let (peak_bin, _) = stats
            .histogram
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .unwrap_or((0, &0));

        let peak_position = peak_bin as f64 / stats.histogram.len() as f64;
        analysis.insert("peak_position".to_string(), peak_position);

        // Compute skewness approximation
        let mean_bin = stats
            .histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i as f64 * count as f64)
            .sum::<f64>()
            / total_activations as f64;

        analysis.insert(
            "distribution_center".to_string(),
            mean_bin / stats.histogram.len() as f64,
        );
    }

    analysis
}

/// Compare activation statistics between layers
pub fn compare_layer_statistics<F>(
    stats1: &LayerAnalysisStats<F>,
    stats2: &LayerAnalysisStats<F>,
) -> HashMap<String, f64>
where
    F: Float + Debug,
{
    let mut comparison = HashMap::new();

    let mean_ratio = if stats2.mean_activation != F::zero() {
        (stats1.mean_activation / stats2.mean_activation)
            .to_f64()
            .unwrap_or(0.0)
    } else {
        0.0
    };
    comparison.insert("mean_activation_ratio".to_string(), mean_ratio);

    let sparsity_diff = stats1.sparsity - stats2.sparsity;
    comparison.insert("sparsity_difference".to_string(), sparsity_diff);

    let dead_neuron_diff = stats1.dead_neuron_percentage - stats2.dead_neuron_percentage;
    comparison.insert("dead_neuron_difference".to_string(), dead_neuron_diff);

    comparison
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_compute_layer_statistics() {
        let activations = Array::from_vec(vec![0.0, 0.1, 0.5, 1.0, 0.0, -0.2]).into_dyn();
        let stats = compute_layer_statistics(&activations).unwrap();

        assert!(stats.mean_activation > -0.1 && stats.mean_activation < 0.5);
        assert!(stats.max_activation == 1.0);
        assert!(stats.min_activation == -0.2);
        assert!(stats.dead_neuron_percentage > 0.0);
    }

    #[test]
    fn test_compute_attribution_statistics() {
        let attribution = Array::from_vec(vec![0.5, -0.3, 0.8, 0.0, -0.1]).into_dyn();
        let stats = compute_attribution_statistics(&attribution);

        assert!(stats.positive_attribution_ratio > 0.0);
        assert!(stats.total_positive_attribution > 0.0);
        assert!(stats.total_negative_attribution < 0.0);
    }

    #[test]
    fn test_correlation_computation() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).into_dyn();
        let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0]).into_dyn();

        let correlation = compute_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_find_most_important_features() {
        let mut attributions = HashMap::new();
        attributions.insert(
            "method1".to_string(),
            Array::from_vec(vec![0.1, 0.8, 0.3, 0.9, 0.2]).into_dyn(),
        );
        attributions.insert(
            "method2".to_string(),
            Array::from_vec(vec![0.2, 0.7, 0.4, 0.8, 0.1]).into_dyn(),
        );

        let important_features = find_most_important_features(&attributions, 3);
        assert_eq!(important_features.len(), 3);
        assert_eq!(important_features[0], 3); // Index 3 has highest average attribution
    }

    #[test]
    fn test_interpretation_summary() {
        let mut attributions = HashMap::new();
        attributions.insert(
            "saliency".to_string(),
            Array::from_vec(vec![0.1, 0.5, 0.3]).into_dyn(),
        );
        attributions.insert(
            "gradcam".to_string(),
            Array::from_vec(vec![0.2, 0.4, 0.3]).into_dyn(),
        );

        let summary = generate_interpretation_summary(&attributions);
        assert_eq!(summary.num_attribution_methods, 2);
        assert!(summary.interpretation_confidence > 0.0);
    }
}
