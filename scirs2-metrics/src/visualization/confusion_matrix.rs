//! Confusion matrix visualization
//!
//! This module provides tools for visualizing confusion matrices.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use std::error::Error;

use super::{ColorMap, MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::classification::confusion_matrix;
use crate::error::Result;

/// Confusion matrix visualizer
///
/// This struct provides methods for visualizing confusion matrices.
#[derive(Debug, Clone)]
pub struct ConfusionMatrixVisualizer<'a, T, S>
where
    T: Clone + PartialEq + std::fmt::Debug + std::hash::Hash + Ord + num_traits::NumCast,
    S: Data<Elem = T>,
{
    /// Confusion matrix data
    matrix: Array2<f64>,
    /// Class labels
    labels: Option<Vec<String>>,
    /// Title for the plot
    title: String,
    /// Whether to normalize the confusion matrix
    normalize: bool,
    /// Color map to use
    color_map: ColorMap,
    /// Whether to include text labels in the visualization
    include_text: bool,
    /// Original y_true data
    y_true: Option<&'a ArrayBase<S, Ix2>>,
    /// Original y_pred data
    y_pred: Option<&'a ArrayBase<S, Ix2>>,
}

impl<'a, T, S> ConfusionMatrixVisualizer<'a, T, S>
where
    T: Clone + PartialEq + std::fmt::Debug + std::hash::Hash + Ord + num_traits::NumCast + 'static,
    S: Data<Elem = T>,
{
    /// Create a new ConfusionMatrixVisualizer from a confusion matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - Confusion matrix as a 2D array
    /// * `labels` - Optional class labels
    ///
    /// # Returns
    ///
    /// * A new ConfusionMatrixVisualizer
    pub fn new(matrix: Array2<f64>, labels: Option<Vec<String>>) -> Self {
        ConfusionMatrixVisualizer {
            matrix,
            labels,
            title: "Confusion Matrix".to_string(),
            normalize: false,
            color_map: ColorMap::BlueRed,
            include_text: true,
            y_true: None,
            y_pred: None,
        }
    }

    /// Create a ConfusionMatrixVisualizer from true and predicted labels
    ///
    /// # Arguments
    ///
    /// * `y_true` - True labels
    /// * `y_pred` - Predicted labels
    /// * `labels` - Optional class labels
    ///
    /// # Returns
    ///
    /// * A new ConfusionMatrixVisualizer
    pub fn from_labels(
        y_true: &'a ArrayBase<S, Ix2>,
        y_pred: &'a ArrayBase<S, Ix2>,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        // We'll calculate the confusion matrix later during visualization
        // to allow for changes in normalization
        Ok(ConfusionMatrixVisualizer {
            matrix: Array2::zeros((0, 0)),
            labels,
            title: "Confusion Matrix".to_string(),
            normalize: false,
            color_map: ColorMap::BlueRed,
            include_text: true,
            y_true: Some(y_true),
            y_pred: Some(y_pred),
        })
    }

    /// Set whether to normalize the confusion matrix
    ///
    /// # Arguments
    ///
    /// * `normalize` - Whether to normalize the confusion matrix
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the title for the plot
    ///
    /// # Arguments
    ///
    /// * `title` - Title for the plot
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set the color map to use
    ///
    /// # Arguments
    ///
    /// * `color_map` - Color map to use
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_color_map(mut self, color_map: ColorMap) -> Self {
        self.color_map = color_map;
        self
    }

    /// Set whether to include text labels in the visualization
    ///
    /// # Arguments
    ///
    /// * `include_text` - Whether to include text labels
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_include_text(mut self, include_text: bool) -> Self {
        self.include_text = include_text;
        self
    }

    /// Get the confusion matrix
    ///
    /// If raw data (y_true and y_pred) was provided, this will calculate
    /// the confusion matrix on demand. Otherwise, it returns the stored matrix.
    ///
    /// # Returns
    ///
    /// * Result containing the confusion matrix
    fn get_matrix(&self) -> Result<Array2<f64>> {
        if self.y_true.is_some() && self.y_pred.is_some() {
            let y_true = self.y_true.unwrap();
            let y_pred = self.y_pred.unwrap();

            // Calculate confusion matrix
            let (cm, _) = confusion_matrix(y_true, y_pred, None)?;

            // Normalize if requested
            if self.normalize {
                // Normalize by row (true labels)
                let mut normalized = Array2::zeros(cm.dim());
                for (i, row) in cm.outer_iter().enumerate() {
                    let row_sum: f64 = row.sum() as f64;
                    if row_sum > 0.0 {
                        for (j, &val) in row.iter().enumerate() {
                            normalized[[i, j]] = val as f64 / row_sum;
                        }
                    }
                }
                Ok(normalized)
            } else {
                // Convert u64 to f64
                let float_cm = cm.mapv(|x| x as f64);
                Ok(float_cm)
            }
        } else {
            // Return the stored matrix
            if self.normalize {
                // Normalize by row (true labels)
                let mut normalized = Array2::zeros(self.matrix.dim());
                for (i, row) in self.matrix.outer_iter().enumerate() {
                    let row_sum = row.sum();
                    if row_sum > 0.0 {
                        for (j, &val) in row.iter().enumerate() {
                            normalized[[i, j]] = val / row_sum;
                        }
                    }
                }
                Ok(normalized)
            } else {
                Ok(self.matrix.clone())
            }
        }
    }
}

impl<T, S> MetricVisualizer for ConfusionMatrixVisualizer<'_, T, S>
where
    T: Clone + PartialEq + std::fmt::Debug + std::hash::Hash + Ord + num_traits::NumCast + 'static,
    S: Data<Elem = T>,
{
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        let matrix = self
            .get_matrix()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Convert matrix to vector of vectors for the heatmap
        let n_classes = matrix.shape()[0];
        let mut z = Vec::with_capacity(n_classes);

        for i in 0..n_classes {
            let mut row = Vec::with_capacity(n_classes);
            for j in 0..n_classes {
                row.push(matrix[[i, j]]);
            }
            z.push(row);
        }

        // Generate x and y coordinate ranges
        let x = (0..n_classes).map(|i| i as f64).collect::<Vec<_>>();
        let y = (0..n_classes).map(|i| i as f64).collect::<Vec<_>>();

        // Generate labels if provided
        let x_labels = if let Some(labels) = &self.labels {
            Some(labels.clone())
        } else {
            Some((0..n_classes).map(|i| i.to_string()).collect())
        };

        let y_labels = x_labels.clone();

        Ok(VisualizationData {
            x,
            y,
            z: Some(z),
            series_names: None,
            x_labels,
            y_labels,
            auxiliary_data: std::collections::HashMap::new(),
            auxiliary_metadata: std::collections::HashMap::new(),
            series: std::collections::HashMap::new(),
        })
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        VisualizationMetadata {
            title: self.title.clone(),
            x_label: "Predicted label".to_string(),
            y_label: "True label".to_string(),
            plot_type: PlotType::Heatmap,
            description: Some(
                "Confusion matrix showing the counts of true vs. predicted class labels"
                    .to_string(),
            ),
        }
    }
}

/// Create a confusion matrix visualization from a confusion matrix
///
/// # Arguments
///
/// * `matrix` - Confusion matrix as a 2D array
/// * `labels` - Optional class labels
/// * `normalize` - Whether to normalize the confusion matrix
///
/// # Returns
///
/// * Result containing a type-erased ConfusionMatrixVisualizer that implements MetricVisualizer
pub fn confusion_matrix_visualization(
    matrix: Array2<f64>,
    labels: Option<Vec<String>>,
    normalize: bool,
) -> Box<dyn MetricVisualizer> {
    // Create a special ConfusionMatrixVisualizer implementation for f64 that doesn't require Hash+Ord
    #[allow(dead_code)]
    struct F64ConfusionMatrixVisualizer {
        matrix: Array2<f64>,
        labels: Option<Vec<String>>,
        title: String,
        normalize: bool,
        color_map: ColorMap,
        include_text: bool,
    }

    impl MetricVisualizer for F64ConfusionMatrixVisualizer {
        fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
            // Convert matrix for visualization
            let matrix = if self.normalize {
                // Normalize by row (true labels)
                let mut normalized = Array2::zeros(self.matrix.dim());
                for (i, row) in self.matrix.outer_iter().enumerate() {
                    let row_sum: f64 = row.sum();
                    if row_sum > 0.0 {
                        for (j, &val) in row.iter().enumerate() {
                            normalized[[i, j]] = val / row_sum;
                        }
                    }
                }
                normalized
            } else {
                self.matrix.clone()
            };

            // Convert matrix to vector of vectors for the heatmap
            let n_classes = matrix.shape()[0];
            let mut z = Vec::with_capacity(n_classes);

            for i in 0..n_classes {
                let mut row = Vec::with_capacity(n_classes);
                for j in 0..n_classes {
                    row.push(matrix[[i, j]]);
                }
                z.push(row);
            }

            // Generate x and y coordinate ranges
            let x = (0..n_classes).map(|i| i as f64).collect::<Vec<_>>();
            let y = (0..n_classes).map(|i| i as f64).collect::<Vec<_>>();

            // Generate labels if provided
            let x_labels = if let Some(labels) = &self.labels {
                Some(labels.clone())
            } else {
                Some((0..n_classes).map(|i| i.to_string()).collect())
            };

            let y_labels = x_labels.clone();

            Ok(VisualizationData {
                x,
                y,
                z: Some(z),
                series_names: None,
                x_labels,
                y_labels,
                auxiliary_data: std::collections::HashMap::new(),
                auxiliary_metadata: std::collections::HashMap::new(),
                series: std::collections::HashMap::new(),
            })
        }

        fn get_metadata(&self) -> VisualizationMetadata {
            VisualizationMetadata {
                title: self.title.clone(),
                x_label: "Predicted label".to_string(),
                y_label: "True label".to_string(),
                plot_type: PlotType::Heatmap,
                description: Some(
                    "Confusion matrix showing the counts of true vs. predicted class labels"
                        .to_string(),
                ),
            }
        }
    }

    Box::new(F64ConfusionMatrixVisualizer {
        matrix,
        labels,
        title: "Confusion Matrix".to_string(),
        normalize,
        color_map: ColorMap::BlueRed,
        include_text: true,
    })
}

/// Create a confusion matrix visualization from true and predicted labels
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `y_pred` - Predicted labels
/// * `labels` - Optional class labels
/// * `normalize` - Whether to normalize the confusion matrix
///
/// # Returns
///
/// * Result containing a box of dyn MetricVisualizer
pub fn confusion_matrix_from_labels<'a, T, S>(
    y_true: &'a ArrayBase<S, Ix2>,
    y_pred: &'a ArrayBase<S, Ix2>,
    labels: Option<Vec<String>>,
    normalize: bool,
) -> Result<Box<dyn MetricVisualizer + 'a>>
where
    T: Clone + PartialEq + std::fmt::Debug + std::hash::Hash + Ord + num_traits::NumCast + 'static,
    S: Data<Elem = T>,
{
    let visualizer = ConfusionMatrixVisualizer::from_labels(y_true, y_pred, labels)?;
    Ok(Box::new(visualizer.with_normalize(normalize)))
}
