//! Layer activation and feature visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing layer activations,
//! feature maps, activation histograms, and attention patterns.

use super::config::VisualizationConfig;
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;

use ndarray::{ArrayD, ScalarOperand};
use num_traits::Float;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;

/// Layer activation visualizer
#[allow(dead_code)]
pub struct ActivationVisualizer<F: Float + Debug + ScalarOperand> {
    /// Model reference
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Cached activations
    activation_cache: HashMap<String, ArrayD<F>>,
}

/// Activation visualization options
#[derive(Debug, Clone, Serialize)]
pub struct ActivationVisualizationOptions {
    /// Layers to visualize
    pub target_layers: Vec<String>,
    /// Visualization type
    pub visualization_type: ActivationVisualizationType,
    /// Normalization method
    pub normalization: ActivationNormalization,
    /// Color mapping
    pub colormap: Colormap,
    /// Aggregation method for multi-channel data
    pub aggregation: ChannelAggregation,
}

/// Types of activation visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationVisualizationType {
    /// Feature maps as heatmaps
    FeatureMaps,
    /// Activation histograms
    Histograms,
    /// Statistics summary
    Statistics,
    /// Spatial attention maps
    AttentionMaps,
    /// Activation flow
    ActivationFlow,
}

/// Activation normalization methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationNormalization {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Percentile-based normalization
    Percentile(f64, f64),
    /// Custom normalization function
    Custom(String),
}

/// Color mapping for visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Colormap {
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Inferno colormap
    Inferno,
    /// Jet colormap
    Jet,
    /// Grayscale
    Gray,
    /// Red-blue diverging
    RdBu,
    /// Custom colormap
    Custom(Vec<String>),
}

/// Channel aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ChannelAggregation {
    /// No aggregation (show all channels)
    None,
    /// Average across channels
    Mean,
    /// Maximum across channels
    Max,
    /// Minimum across channels
    Min,
    /// Standard deviation across channels
    Std,
    /// Select specific channels
    Select(Vec<usize>),
}

/// Activation statistics for a layer
#[derive(Debug, Clone, Serialize)]
pub struct ActivationStatistics<F: Float + Debug> {
    /// Layer name
    pub layer_name: String,
    /// Mean activation value
    pub mean: F,
    /// Standard deviation
    pub std: F,
    /// Minimum value
    pub min: F,
    /// Maximum value
    pub max: F,
    /// Percentiles (5%, 25%, 50%, 75%, 95%)
    pub percentiles: [F; 5],
    /// Sparsity (fraction of zero or near-zero activations)
    pub sparsity: f64,
    /// Dead neurons (always zero)
    pub dead_neurons: usize,
    /// Total neurons
    pub total_neurons: usize,
}

/// Feature map information
#[derive(Debug, Clone, Serialize)]
pub struct FeatureMapInfo {
    /// Layer name
    pub layer_name: String,
    /// Feature map index
    pub feature_index: usize,
    /// Spatial dimensions (height, width)
    pub spatial_dims: (usize, usize),
    /// Channel dimension
    pub channels: usize,
    /// Activation range (min, max)
    pub activation_range: (f64, f64),
}

/// Activation histogram data
#[derive(Debug, Clone, Serialize)]
pub struct ActivationHistogram<F: Float + Debug> {
    /// Layer name
    pub layer_name: String,
    /// Histogram bins
    pub bins: Vec<F>,
    /// Bin counts
    pub counts: Vec<usize>,
    /// Bin edges
    pub edges: Vec<F>,
    /// Total sample count
    pub total_samples: usize,
}

// Implementation for ActivationVisualizer

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + ScalarOperand + Send + Sync>
    ActivationVisualizer<F>
{
    /// Create a new activation visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            activation_cache: HashMap::new(),
        }
    }

    /// Visualize layer activations for given input
    pub fn visualize_activations(
        &mut self,
        input: &ArrayD<F>,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Compute activations
        self.compute_activations(input, &options.target_layers)?;

        // Generate visualizations based on type
        match options.visualization_type {
            ActivationVisualizationType::FeatureMaps => self.generate_feature_maps(options),
            ActivationVisualizationType::Histograms => self.generate_histograms(options),
            ActivationVisualizationType::Statistics => self.generate_statistics(options),
            ActivationVisualizationType::AttentionMaps => self.generate_attention_maps(options),
            ActivationVisualizationType::ActivationFlow => self.generate_activation_flow(options),
        }
    }

    /// Get cached activations for a layer
    pub fn get_cached_activations(&self, layer_name: &str) -> Option<&ArrayD<F>> {
        self.activation_cache.get(layer_name)
    }

    /// Clear the activation cache
    pub fn clear_cache(&mut self) {
        self.activation_cache.clear();
    }

    /// Get activation statistics for all cached layers
    pub fn get_activation_statistics(&self) -> Result<Vec<ActivationStatistics<F>>> {
        let mut stats = Vec::new();

        for (layer_name, activations) in &self.activation_cache {
            let layer_stats = self.compute_layer_statistics(layer_name, activations)?;
            stats.push(layer_stats);
        }

        Ok(stats)
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }

    fn compute_activations(&mut self, _input: &ArrayD<F>, _target_layers: &[String]) -> Result<()> {
        // TODO: Implement activation computation
        // This would run forward pass and capture intermediate activations
        Err(NeuralError::NotImplementedError(
            "Activation computation not yet implemented".to_string(),
        ))
    }

    fn generate_feature_maps(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement feature map generation
        Err(NeuralError::NotImplementedError(
            "Feature map visualization not yet implemented".to_string(),
        ))
    }

    fn generate_histograms(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement histogram generation
        Err(NeuralError::NotImplementedError(
            "Histogram visualization not yet implemented".to_string(),
        ))
    }

    fn generate_statistics(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement statistics generation
        Err(NeuralError::NotImplementedError(
            "Statistics visualization not yet implemented".to_string(),
        ))
    }

    fn generate_attention_maps(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention map generation
        Err(NeuralError::NotImplementedError(
            "Attention map visualization not yet implemented".to_string(),
        ))
    }

    fn generate_activation_flow(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement activation flow generation
        Err(NeuralError::NotImplementedError(
            "Activation flow visualization not yet implemented".to_string(),
        ))
    }

    fn compute_layer_statistics(
        &self,
        layer_name: &str,
        activations: &ArrayD<F>,
    ) -> Result<ActivationStatistics<F>> {
        let total_elements = activations.len();
        if total_elements == 0 {
            return Err(NeuralError::InvalidArgument(
                "Empty activation tensor".to_string(),
            ));
        }

        // Compute basic statistics
        let mut sum = F::zero();
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();
        let mut zero_count = 0;

        for &val in activations.iter() {
            sum = sum + val;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            if val.abs() < F::from(1e-6).unwrap_or(F::zero()) {
                zero_count += 1;
            }
        }

        let mean = sum / F::from(total_elements).unwrap_or(F::one());

        // Compute standard deviation
        let mut variance_sum = F::zero();
        for &val in activations.iter() {
            let diff = val - mean;
            variance_sum = variance_sum + diff * diff;
        }
        let variance = variance_sum / F::from(total_elements - 1).unwrap_or(F::one());
        let std = variance.sqrt();

        // Compute percentiles (simplified implementation)
        let mut sorted_values: Vec<F> = activations.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentiles = [
            sorted_values[total_elements * 5 / 100],  // 5%
            sorted_values[total_elements * 25 / 100], // 25%
            sorted_values[total_elements * 50 / 100], // 50%
            sorted_values[total_elements * 75 / 100], // 75%
            sorted_values[total_elements * 95 / 100], // 95%
        ];

        let sparsity = zero_count as f64 / total_elements as f64;

        Ok(ActivationStatistics {
            layer_name: layer_name.to_string(),
            mean,
            std,
            min: min_val,
            max: max_val,
            percentiles,
            sparsity,
            dead_neurons: zero_count,
            total_neurons: total_elements,
        })
    }
}

// Default implementations for configuration types

impl Default for ActivationVisualizationOptions {
    fn default() -> Self {
        Self {
            target_layers: Vec::new(),
            visualization_type: ActivationVisualizationType::FeatureMaps,
            normalization: ActivationNormalization::MinMax,
            colormap: Colormap::Viridis,
            aggregation: ChannelAggregation::Mean,
        }
    }
}

impl Default for FeatureMapInfo {
    fn default() -> Self {
        Self {
            layer_name: "unknown".to_string(),
            feature_index: 0,
            spatial_dims: (1, 1),
            channels: 1,
            activation_range: (0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;

    #[test]
    fn test_activation_visualizer_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let visualizer = ActivationVisualizer::new(model, config);

        assert!(visualizer.activation_cache.is_empty());
    }

    #[test]
    fn test_activation_visualization_options_default() {
        let options = ActivationVisualizationOptions::default();
        assert_eq!(
            options.visualization_type,
            ActivationVisualizationType::FeatureMaps
        );
        assert_eq!(options.normalization, ActivationNormalization::MinMax);
        assert_eq!(options.colormap, Colormap::Viridis);
        assert_eq!(options.aggregation, ChannelAggregation::Mean);
    }

    #[test]
    fn test_activation_visualization_types() {
        let types = vec![
            ActivationVisualizationType::FeatureMaps,
            ActivationVisualizationType::Histograms,
            ActivationVisualizationType::Statistics,
            ActivationVisualizationType::AttentionMaps,
            ActivationVisualizationType::ActivationFlow,
        ];

        assert_eq!(types.len(), 5);
        assert_eq!(types[0], ActivationVisualizationType::FeatureMaps);
    }

    #[test]
    fn test_normalization_methods() {
        let none = ActivationNormalization::None;
        let minmax = ActivationNormalization::MinMax;
        let zscore = ActivationNormalization::ZScore;
        let percentile = ActivationNormalization::Percentile(5.0, 95.0);

        assert_eq!(none, ActivationNormalization::None);
        assert_eq!(minmax, ActivationNormalization::MinMax);
        assert_eq!(zscore, ActivationNormalization::ZScore);

        match percentile {
            ActivationNormalization::Percentile(low, high) => {
                assert_eq!(low, 5.0);
                assert_eq!(high, 95.0);
            }
            _ => panic!("Expected percentile normalization"),
        }
    }

    #[test]
    fn test_colormaps() {
        let colormaps = vec![
            Colormap::Viridis,
            Colormap::Plasma,
            Colormap::Inferno,
            Colormap::Jet,
            Colormap::Gray,
            Colormap::RdBu,
        ];

        assert_eq!(colormaps.len(), 6);
        assert_eq!(colormaps[0], Colormap::Viridis);

        let custom = Colormap::Custom(vec!["#ff0000".to_string(), "#00ff00".to_string()]);
        match custom {
            Colormap::Custom(colors) => assert_eq!(colors.len(), 2),
            _ => panic!("Expected custom colormap"),
        }
    }

    #[test]
    fn test_channel_aggregation() {
        let aggregations = vec![
            ChannelAggregation::None,
            ChannelAggregation::Mean,
            ChannelAggregation::Max,
            ChannelAggregation::Min,
            ChannelAggregation::Std,
            ChannelAggregation::Select(vec![0, 1, 2]),
        ];

        assert_eq!(aggregations.len(), 6);
        assert_eq!(aggregations[1], ChannelAggregation::Mean);

        match &aggregations[5] {
            ChannelAggregation::Select(channels) => assert_eq!(channels.len(), 3),
            _ => panic!("Expected select aggregation"),
        }
    }

    #[test]
    fn test_feature_map_info_default() {
        let info = FeatureMapInfo::default();
        assert_eq!(info.layer_name, "unknown");
        assert_eq!(info.feature_index, 0);
        assert_eq!(info.spatial_dims, (1, 1));
        assert_eq!(info.channels, 1);
        assert_eq!(info.activation_range, (0.0, 1.0));
    }

    #[test]
    fn test_cache_operations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let mut visualizer = ActivationVisualizer::new(model, config);

        assert!(visualizer.activation_cache.is_empty());
        assert!(visualizer.get_cached_activations("test_layer").is_none());

        visualizer.clear_cache();
        assert!(visualizer.activation_cache.is_empty());
    }
}
