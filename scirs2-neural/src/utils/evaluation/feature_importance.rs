//! Feature importance visualization for machine learning models
//!
//! This module provides tools for visualizing and analyzing feature importance
//! metrics from machine learning models, helping to understand which features
//! have the greatest impact on model predictions.

use crate::error::{NeuralError, Result};
use crate::utils::colors::{
    colored_metric_cell, colorize, gradient_color, stylize, ColorOptions, Style,
};
use ndarray::Array1;
use num_traits::Float;
use std::fmt::{Debug, Display};

/// Feature importance visualization for machine learning models
///
/// This struct facilitates the visualization and analysis of feature importance
/// scores from machine learning models, helping to identify which features
/// contribute most to predictions.
pub struct FeatureImportance<F: Float + Debug + Display> {
    /// Names of the features
    pub feature_names: Vec<String>,
    /// Importance scores for each feature
    pub importance: Array1<F>,
}

impl<F: Float + Debug + Display> FeatureImportance<F> {
    /// Create a new feature importance visualization
    ///
    /// # Arguments
    ///
    /// * `feature_names` - Names of features
    /// * `importance` - Importance scores
    ///
    /// # Returns
    ///
    /// * `Result<FeatureImportance<F>>` - The feature importance visualization
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::Array1;
    /// use scirs2_neural::utils::evaluation::FeatureImportance;
    ///
    /// // Create feature names and importance scores
    /// let feature_names = vec![
    ///     "Age".to_string(),
    ///     "Income".to_string(),
    ///     "Education".to_string(),
    ///     "Location".to_string()
    /// ];
    /// let importance = Array1::from_vec(vec![0.35, 0.25, 0.20, 0.10]);
    ///
    /// // Create feature importance visualization
    /// let feature_importance = FeatureImportance::<f64>::new(feature_names, importance).unwrap();
    /// ```
    pub fn new(feature_names: Vec<String>, importance: Array1<F>) -> Result<Self> {
        if feature_names.len() != importance.len() {
            return Err(NeuralError::ValidationError(
                "Number of feature names must match number of importance scores".to_string(),
            ));
        }

        Ok(FeatureImportance {
            feature_names,
            importance,
        })
    }

    /// Get the top-k most important features
    ///
    /// # Arguments
    ///
    /// * `k` - Number of top features to return
    ///
    /// # Returns
    ///
    /// * `(Vec<String>, Array1<F>)` - Feature names and importance scores
    pub fn top_k(&self, k: usize) -> (Vec<String>, Array1<F>) {
        let k = std::cmp::min(k, self.feature_names.len());
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();

        // Sort indices by importance (descending)
        indices.sort_by(|&a, &b| {
            self.importance[b]
                .partial_cmp(&self.importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top-k features
        let top_indices = indices[..k].to_vec();
        let top_names = top_indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();
        let top_importance = Array1::from_iter(top_indices.iter().map(|&i| self.importance[i]));

        (top_names, top_importance)
    }

    /// Create an ASCII bar chart of feature importance
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the chart
    /// * `width` - Width of the bar chart
    /// * `k` - Number of top features to include (None for all)
    ///
    /// # Returns
    ///
    /// * `String` - ASCII bar chart
    pub fn to_ascii(&self, title: Option<&str>, width: usize, k: Option<usize>) -> String {
        self.to_ascii_with_options(title, width, k, &ColorOptions::default())
    }

    /// Create an ASCII bar chart of feature importance with color options
    ///
    /// This method creates a bar chart visualization with customizable colors,
    /// showing feature importance scores in descending order.
    ///
    /// # Arguments
    ///
    /// * `title` - Optional title for the chart
    /// * `width` - Width of the bar chart
    /// * `k` - Number of top features to include (None for all)
    /// * `color_options` - Color options for visualization
    ///
    /// # Returns
    ///
    /// * `String` - ASCII bar chart with colors
    pub fn to_ascii_with_options(
        &self,
        title: Option<&str>,
        width: usize,
        k: Option<usize>,
        color_options: &ColorOptions,
    ) -> String {
        let (features, importance) = if let Some(top_k) = k {
            self.top_k(top_k)
        } else {
            (self.feature_names.clone(), self.importance.clone())
        };

        // Pre-allocate result string with estimated capacity
        let mut result = String::with_capacity(features.len() * 80);

        // Add title if provided
        if let Some(title_text) = title {
            if color_options.enabled {
                result.push_str(&stylize(title_text, Style::Bold));
            } else {
                result.push_str(title_text);
            }
            result.push_str("\n\n");
        }

        // Determine maximum importance for scaling
        let max_importance =
            importance
                .iter()
                .copied()
                .fold(F::zero(), |acc, v| if v > acc { v } else { acc });

        // Get maximum feature name length for alignment
        let max_name_len = features
            .iter()
            .map(|name| name.len())
            .max()
            .unwrap_or(10)
            .max(10);

        // Determine available width for bars
        let bar_area_width = width.saturating_sub(max_name_len + 10);

        // Constants for formatting
        const ANSI_PADDING: usize = 9; // Extra space needed for ANSI color codes

        // Create a sorted index
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.sort_by(|&a, &b| {
            importance[b]
                .partial_cmp(&importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Draw the bar chart
        for &idx in &indices {
            let feature_name = &features[idx];
            let imp = importance[idx];

            // Scale the importance to fit in the available width
            let bar_length = if max_importance > F::zero() {
                let ratio = (imp / max_importance).to_f64().unwrap_or(0.0);
                (ratio * bar_area_width as f64).round() as usize
            } else {
                0
            };

            // Format feature name with styling
            let formatted_name = if color_options.enabled {
                stylize(feature_name, Style::Bold).to_string()
            } else {
                feature_name.clone()
            };

            // Format importance value with coloring
            let normalized_imp = if max_importance > F::zero() {
                (imp / max_importance).to_f64().unwrap_or(0.0)
            } else {
                0.0
            };

            let formatted_imp = if color_options.enabled {
                colored_metric_cell(format!("{:.3}", imp), normalized_imp, color_options)
            } else {
                format!("{:.3}", imp)
            };

            // Format the bar with coloring
            let bar = if color_options.enabled {
                if let Some(color) = gradient_color(normalized_imp, color_options) {
                    colorize("█".repeat(bar_length), color)
                } else {
                    "█".repeat(bar_length)
                }
            } else {
                "█".repeat(bar_length)
            };

            // Format the line with consistent width calculations
            let name_padding = if color_options.enabled {
                ANSI_PADDING
            } else {
                0
            };
            result.push_str(&format!(
                "{:<width$} | {} |{}|\n",
                formatted_name,
                formatted_imp,
                bar,
                width = max_name_len + name_padding
            ));
        }

        result
    }
}
