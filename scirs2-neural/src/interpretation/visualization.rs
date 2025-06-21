//! Attention and feature visualization for neural network interpretation
//!
//! This module provides visualization capabilities for understanding neural network
//! behavior including attention visualization, feature visualization, and network dissection.

use crate::error::{NeuralError, Result};
use ndarray::ArrayD;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

/// Feature visualization method
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizationMethod {
    /// Activation maximization
    ActivationMaximization {
        /// Target layer name for activation maximization
        target_layer: String,
        /// Specific unit to maximize (None for all)
        target_unit: Option<usize>,
        /// Number of optimization iterations
        num_iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
    },
    /// Deep dream
    DeepDream {
        /// Target layer name for deep dream
        target_layer: String,
        /// Number of optimization iterations
        num_iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
        /// Factor to amplify activations
        amplify_factor: f64,
    },
    /// Feature inversion
    FeatureInversion {
        /// Target layer name for feature inversion
        target_layer: String,
        /// Weight for regularization term
        regularization_weight: f64,
    },
    /// Class Activation Mapping (CAM)
    ClassActivationMapping {
        /// Target layer for CAM
        target_layer: String,
        /// Target class index
        target_class: usize,
    },
    /// Network dissection for concept visualization
    NetworkDissection {
        /// Concept dataset for analysis
        concept_data: Vec<ArrayD<f32>>,
        /// Concept labels
        concept_labels: Vec<String>,
    },
}

/// Attention aggregation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionAggregation {
    /// Average across all heads
    Average,
    /// Maximum across all heads
    Maximum,
    /// Specific head only
    Head(usize),
    /// Weighted combination of heads
    Weighted(Vec<f64>),
}

/// Attention visualizer for transformer models
#[derive(Debug, Clone)]
pub struct AttentionVisualizer<F: Float + Debug> {
    /// Number of attention heads
    pub num_heads: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Aggregation strategy
    pub aggregation: AttentionAggregation,
    /// Cached attention weights
    pub attention_cache: HashMap<String, ArrayD<F>>,
    /// Layer names to visualize
    pub target_layers: Vec<String>,
}

/// Visualization result containing processed data
#[derive(Debug, Clone)]
pub struct VisualizationResult<F: Float + Debug> {
    /// Visualization method used
    pub method: VisualizationMethod,
    /// Generated visualization data
    pub visualization_data: ArrayD<F>,
    /// Metadata about the visualization
    pub metadata: HashMap<String, String>,
    /// Confidence or quality score
    pub quality_score: f64,
}

/// Network dissection result
#[derive(Debug, Clone)]
pub struct NetworkDissectionResult {
    /// Layer name analyzed
    pub layer_name: String,
    /// Detected concepts and their selectivity scores
    pub concept_selectivity: HashMap<String, f64>,
    /// Number of units analyzed
    pub num_units: usize,
    /// Coverage of concepts across units
    pub concept_coverage: HashMap<String, usize>,
}

impl<F> AttentionVisualizer<F>
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
    /// Create a new attention visualizer
    pub fn new(
        num_heads: usize,
        sequence_length: usize,
        aggregation: AttentionAggregation,
        target_layers: Vec<String>,
    ) -> Self {
        Self {
            num_heads,
            sequence_length,
            aggregation,
            attention_cache: HashMap::new(),
            target_layers,
        }
    }

    /// Cache attention weights for a layer
    pub fn cache_attention_weights(&mut self, layer_name: String, attention_weights: ArrayD<F>) {
        self.attention_cache.insert(layer_name, attention_weights);
    }

    /// Visualize attention patterns
    pub fn visualize_attention(&self, layer_name: &str) -> Result<ArrayD<F>> {
        let attention_weights = self.attention_cache.get(layer_name).ok_or_else(|| {
            NeuralError::ComputationError(format!(
                "No attention weights cached for layer: {}",
                layer_name
            ))
        })?;

        self.aggregate_attention_heads(attention_weights)
    }

    /// Aggregate attention across multiple heads
    pub fn aggregate_attention_heads(&self, attention_weights: &ArrayD<F>) -> Result<ArrayD<F>> {
        match &self.aggregation {
            AttentionAggregation::Average => {
                // Average across head dimension (assuming shape: [batch, heads, seq, seq])
                if attention_weights.ndim() >= 4 {
                    Ok(attention_weights.mean_axis(ndarray::Axis(1)).unwrap())
                } else {
                    Ok(attention_weights.clone())
                }
            }
            AttentionAggregation::Maximum => {
                // Maximum across head dimension
                if attention_weights.ndim() >= 4 {
                    let max_attention = attention_weights.fold_axis(
                        ndarray::Axis(1),
                        F::neg_infinity(),
                        |&acc, &x| acc.max(x),
                    );
                    Ok(max_attention)
                } else {
                    Ok(attention_weights.clone())
                }
            }
            AttentionAggregation::Head(head_idx) => {
                // Select specific head
                if attention_weights.ndim() >= 4 && *head_idx < self.num_heads {
                    Ok(attention_weights
                        .index_axis(ndarray::Axis(1), *head_idx)
                        .to_owned())
                } else {
                    Err(NeuralError::InvalidArchitecture(format!(
                        "Invalid head index {} for {} heads",
                        head_idx, self.num_heads
                    )))
                }
            }
            AttentionAggregation::Weighted(weights) => {
                // Weighted combination of heads
                if weights.len() != self.num_heads {
                    return Err(NeuralError::InvalidArchitecture(
                        "Number of weights must match number of heads".to_string(),
                    ));
                }

                if attention_weights.ndim() >= 4 {
                    let mut weighted_attention =
                        attention_weights.index_axis(ndarray::Axis(1), 0).to_owned()
                            * F::from(weights[0]).unwrap();

                    for (i, &weight) in weights.iter().enumerate().skip(1) {
                        let head_attention =
                            attention_weights.index_axis(ndarray::Axis(1), i).to_owned();
                        weighted_attention =
                            weighted_attention + head_attention * F::from(weight).unwrap();
                    }

                    Ok(weighted_attention)
                } else {
                    Ok(attention_weights.clone())
                }
            }
        }
    }

    /// Generate attention rollout visualization
    pub fn attention_rollout(&self) -> Result<ArrayD<F>> {
        // Simplified attention rollout - would normally compute across all layers
        if self.attention_cache.is_empty() {
            return Err(NeuralError::ComputationError(
                "No attention weights available for rollout".to_string(),
            ));
        }

        // For now, just return the first cached attention
        let first_attention = self.attention_cache.values().next().unwrap();
        self.aggregate_attention_heads(first_attention)
    }

    /// Visualize attention flow between tokens
    pub fn visualize_attention_flow(
        &self,
        layer_name: &str,
        token_indices: &[usize],
    ) -> Result<Vec<f64>> {
        let attention = self.visualize_attention(layer_name)?;

        let mut flow_scores = Vec::new();

        for &token_idx in token_indices {
            if token_idx < self.sequence_length {
                // Compute attention flow for this token
                let token_attention = attention.index_axis(ndarray::Axis(1), token_idx);
                let flow_score = token_attention.sum().to_f64().unwrap_or(0.0);
                flow_scores.push(flow_score);
            } else {
                flow_scores.push(0.0);
            }
        }

        Ok(flow_scores)
    }
}

/// Generate feature visualization using specified method
pub fn generate_feature_visualization<F>(
    method: &VisualizationMethod,
    input_shape: &[usize],
) -> Result<VisualizationResult<F>>
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
    match method {
        VisualizationMethod::ActivationMaximization {
            target_layer,
            target_unit,
            num_iterations,
            learning_rate,
        } => {
            // Simplified activation maximization
            let mut optimized_input = ndarray::Array::zeros(input_shape).into_dyn();

            for _iter in 0..*num_iterations {
                // Apply gradient ascent (simplified)
                optimized_input = optimized_input
                    .mapv(|x| x + F::from(*learning_rate * rand::random::<f64>()).unwrap());
            }

            let mut metadata = HashMap::new();
            metadata.insert("target_layer".to_string(), target_layer.clone());
            metadata.insert("iterations".to_string(), num_iterations.to_string());
            if let Some(unit) = target_unit {
                metadata.insert("target_unit".to_string(), unit.to_string());
            }

            Ok(VisualizationResult {
                method: method.clone(),
                visualization_data: optimized_input,
                metadata,
                quality_score: 0.8,
            })
        }
        VisualizationMethod::DeepDream {
            target_layer,
            num_iterations,
            learning_rate,
            amplify_factor,
        } => {
            // Simplified deep dream implementation
            let mut dream_input = ndarray::Array::ones(input_shape).into_dyn();

            for _iter in 0..*num_iterations {
                // Amplify activations (simplified)
                dream_input = dream_input.mapv(|x| {
                    x * F::from(*amplify_factor).unwrap()
                        + F::from(*learning_rate * rand::random::<f64>()).unwrap()
                });
            }

            let mut metadata = HashMap::new();
            metadata.insert("target_layer".to_string(), target_layer.clone());
            metadata.insert("iterations".to_string(), num_iterations.to_string());
            metadata.insert("amplify_factor".to_string(), amplify_factor.to_string());

            Ok(VisualizationResult {
                method: method.clone(),
                visualization_data: dream_input,
                metadata,
                quality_score: 0.7,
            })
        }
        VisualizationMethod::FeatureInversion {
            target_layer,
            regularization_weight,
        } => {
            // Simplified feature inversion
            let inverted_input = ndarray::Array::zeros(input_shape).into_dyn();

            let mut metadata = HashMap::new();
            metadata.insert("target_layer".to_string(), target_layer.clone());
            metadata.insert(
                "regularization".to_string(),
                regularization_weight.to_string(),
            );

            Ok(VisualizationResult {
                method: method.clone(),
                visualization_data: inverted_input,
                metadata,
                quality_score: 0.6,
            })
        }
        VisualizationMethod::ClassActivationMapping {
            target_layer,
            target_class,
        } => {
            // Simplified CAM
            let cam_result = ndarray::Array::ones(input_shape).into_dyn();

            let mut metadata = HashMap::new();
            metadata.insert("target_layer".to_string(), target_layer.clone());
            metadata.insert("target_class".to_string(), target_class.to_string());

            Ok(VisualizationResult {
                method: method.clone(),
                visualization_data: cam_result,
                metadata,
                quality_score: 0.85,
            })
        }
        VisualizationMethod::NetworkDissection {
            concept_data,
            concept_labels,
        } => {
            // Simplified network dissection
            let dissection_result = ndarray::Array::zeros(input_shape).into_dyn();

            let mut metadata = HashMap::new();
            metadata.insert("num_concepts".to_string(), concept_labels.len().to_string());
            metadata.insert("num_examples".to_string(), concept_data.len().to_string());

            Ok(VisualizationResult {
                method: method.clone(),
                visualization_data: dissection_result,
                metadata,
                quality_score: 0.75,
            })
        }
    }
}

/// Perform network dissection analysis
pub fn perform_network_dissection(
    layer_name: String,
    layer_activations: &ArrayD<f32>,
    concept_data: &[ArrayD<f32>],
    concept_labels: &[String],
) -> Result<NetworkDissectionResult> {
    if concept_data.len() != concept_labels.len() {
        return Err(NeuralError::InvalidArchitecture(
            "Number of concept examples must match number of labels".to_string(),
        ));
    }

    let mut concept_selectivity = HashMap::new();
    let mut concept_coverage = HashMap::new();

    // Simplified network dissection
    for (concept_example, concept_label) in concept_data.iter().zip(concept_labels.iter()) {
        // Compute selectivity score (simplified correlation)
        let selectivity = if layer_activations.len() == concept_example.len() {
            let correlation = layer_activations
                .iter()
                .zip(concept_example.iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum::<f64>()
                / layer_activations.len() as f64;
            correlation.abs()
        } else {
            0.0
        };

        concept_selectivity.insert(concept_label.clone(), selectivity);

        // Count units that respond to this concept
        let responsive_units = layer_activations
            .iter()
            .zip(concept_example.iter())
            .filter(|(&a, &b)| (a as f64) * (b as f64) > 0.5)
            .count();

        concept_coverage.insert(concept_label.clone(), responsive_units);
    }

    Ok(NetworkDissectionResult {
        layer_name,
        concept_selectivity,
        num_units: layer_activations.len(),
        concept_coverage,
    })
}

/// Create attention heatmap for visualization
pub fn create_attention_heatmap<F>(
    attention_weights: &ArrayD<F>,
    token_labels: &[String],
) -> Result<Vec<Vec<f64>>>
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
    if attention_weights.ndim() < 2 {
        return Err(NeuralError::InvalidArchitecture(
            "Attention weights must be at least 2D".to_string(),
        ));
    }

    let shape = attention_weights.shape();
    let seq_len = shape[shape.len() - 1];

    if token_labels.len() != seq_len {
        return Err(NeuralError::InvalidArchitecture(
            "Number of token labels must match sequence length".to_string(),
        ));
    }

    let mut heatmap = Vec::new();

    for i in 0..seq_len {
        let mut row = Vec::new();
        for j in 0..seq_len {
            // Get attention weight for position (i, j)
            let weight = if attention_weights.ndim() == 2 {
                attention_weights[[i, j]].to_f64().unwrap_or(0.0)
            } else {
                // For higher dimensions, simplified access - just use 0.5 as placeholder
                // In a real implementation, this would properly handle multi-dimensional attention
                0.5
            };
            row.push(weight);
        }
        heatmap.push(row);
    }

    Ok(heatmap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_attention_visualizer_creation() {
        let visualizer = AttentionVisualizer::<f64>::new(
            8,
            512,
            AttentionAggregation::Average,
            vec!["layer1".to_string(), "layer2".to_string()],
        );

        assert_eq!(visualizer.num_heads, 8);
        assert_eq!(visualizer.sequence_length, 512);
        assert_eq!(visualizer.target_layers.len(), 2);
    }

    #[test]
    fn test_attention_aggregation() {
        let mut visualizer = AttentionVisualizer::<f64>::new(
            2,
            4,
            AttentionAggregation::Average,
            vec!["test".to_string()],
        );

        // Create mock attention weights: [batch=1, heads=2, seq=4, seq=4]
        let attention = Array::ones((1, 2, 4, 4)).into_dyn();
        visualizer.cache_attention_weights("test".to_string(), attention);

        let aggregated = visualizer.visualize_attention("test");
        assert!(aggregated.is_ok());
    }

    #[test]
    fn test_feature_visualization() {
        let method = VisualizationMethod::ActivationMaximization {
            target_layer: "conv1".to_string(),
            target_unit: Some(5),
            num_iterations: 100,
            learning_rate: 0.01,
        };

        let result = generate_feature_visualization::<f64>(&method, &[3, 32, 32]);
        assert!(result.is_ok());

        let viz_result = result.unwrap();
        assert_eq!(viz_result.visualization_data.shape(), &[3, 32, 32]);
        assert!(viz_result.metadata.contains_key("target_layer"));
    }

    #[test]
    fn test_network_dissection() {
        let layer_activations = Array::from_vec(vec![0.5, 0.8, 0.3, 0.9]).into_dyn();
        let concept_data = vec![
            Array::from_vec(vec![0.4, 0.7, 0.2, 0.8]).into_dyn(),
            Array::from_vec(vec![0.6, 0.9, 0.4, 1.0]).into_dyn(),
        ];
        let concept_labels = vec!["dog".to_string(), "car".to_string()];

        let result = perform_network_dissection(
            "conv5".to_string(),
            &layer_activations,
            &concept_data,
            &concept_labels,
        );

        assert!(result.is_ok());
        let dissection = result.unwrap();
        assert_eq!(dissection.layer_name, "conv5");
        assert_eq!(dissection.concept_selectivity.len(), 2);
    }

    #[test]
    fn test_attention_heatmap() {
        let attention = Array::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4])
            .unwrap()
            .into_dyn();
        let tokens = vec!["hello".to_string(), "world".to_string()];

        let heatmap = create_attention_heatmap(&attention, &tokens);
        assert!(heatmap.is_ok());

        let heatmap_data = heatmap.unwrap();
        assert_eq!(heatmap_data.len(), 2);
        assert_eq!(heatmap_data[0].len(), 2);
    }
}
