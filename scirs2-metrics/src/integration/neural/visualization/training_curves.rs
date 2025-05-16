//! Training curve visualizations for neural networks
//!
//! This module provides utilities for visualizing training curves from neural
//! network training, such as loss and accuracy over epochs.

use crate::visualization::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use std::collections::HashMap;
use std::error::Error;

/// Visualizer for neural network training history
pub struct TrainingHistoryVisualizer {
    /// Title of the visualization
    title: String,
    /// Metrics to visualize
    metrics: Vec<String>,
    /// Training history data (epoch -> metric -> value)
    history: Vec<HashMap<String, f64>>,
    /// Validation history data (epoch -> metric -> value)
    val_history: Option<Vec<HashMap<String, f64>>>,
    /// X label (default: "Epoch")
    x_label: String,
    /// Y label (default: metric name)
    y_label: Option<String>,
}

impl TrainingHistoryVisualizer {
    /// Create a new training history visualizer
    pub fn new(
        title: impl Into<String>,
        metrics: Vec<String>,
        history: Vec<HashMap<String, f64>>,
    ) -> Self {
        Self {
            title: title.into(),
            metrics,
            history,
            val_history: None,
            x_label: "Epoch".to_string(),
            y_label: None,
        }
    }

    /// Add validation history
    pub fn with_validation(mut self, val_history: Vec<HashMap<String, f64>>) -> Self {
        self.val_history = Some(val_history);
        self
    }

    /// Set x-axis label
    pub fn with_x_label(mut self, x_label: impl Into<String>) -> Self {
        self.x_label = x_label.into();
        self
    }

    /// Set y-axis label
    pub fn with_y_label(mut self, y_label: impl Into<String>) -> Self {
        self.y_label = Some(y_label.into());
        self
    }
}

impl MetricVisualizer for TrainingHistoryVisualizer {
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>> {
        let mut data = VisualizationData::new();

        // Create x values (epochs)
        let epochs: Vec<f64> = (0..self.history.len()).map(|i| i as f64).collect();
        data.add_series("epochs", epochs.clone());

        // Add data for each metric
        for metric_name in &self.metrics {
            // Training data
            let metric_values: Vec<f64> = self
                .history
                .iter()
                .map(|epoch_data| *epoch_data.get(metric_name).unwrap_or(&f64::NAN))
                .collect();

            data.add_series(metric_name.clone(), metric_values);

            // Validation data if available
            if let Some(val_history) = &self.val_history {
                let val_metric_values: Vec<f64> = val_history
                    .iter()
                    .map(|epoch_data| *epoch_data.get(metric_name).unwrap_or(&f64::NAN))
                    .collect();

                data.add_series(format!("val_{}", metric_name), val_metric_values);
            }
        }

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(PlotType::Line);
        metadata.set_x_label(self.x_label.clone());

        if let Some(y_label) = &self.y_label {
            metadata.set_y_label(y_label.clone());
        } else if self.metrics.len() == 1 {
            metadata.set_y_label(self.metrics[0].clone());
        }

        metadata
    }
}

/// Create a training history visualizer from a neural network's training history
pub fn training_history_visualization(
    metric_names: Vec<String>,
    history: Vec<HashMap<String, f64>>,
    val_history: Option<Vec<HashMap<String, f64>>>,
) -> Box<dyn MetricVisualizer> {
    let mut visualizer = TrainingHistoryVisualizer::new(
        format!("Training History ({})", metric_names.join(", ")),
        metric_names,
        history,
    );

    if let Some(val_history) = val_history {
        visualizer = visualizer.with_validation(val_history);
    }

    Box::new(visualizer)
}
