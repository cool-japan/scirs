//! Learning curve visualization
//!
//! This module provides tools for visualizing learning curves, which show model performance
//! as a function of training set size.

use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::error::{MetricsError, Result};
use ndarray::{ArrayBase, Data, Ix1, Ix2};

/// Learning curve data
///
/// This struct holds the data for a learning curve.
#[derive(Debug, Clone)]
pub struct LearningCurveData {
    /// Training set sizes
    pub train_sizes: Vec<usize>,
    /// Training scores for each training set size
    pub train_scores: Vec<Vec<f64>>,
    /// Validation scores for each training set size
    pub validation_scores: Vec<Vec<f64>>,
}

/// Learning curve visualizer
///
/// This struct provides methods for visualizing learning curves.
#[derive(Debug, Clone)]
pub struct LearningCurveVisualizer {
    /// Learning curve data
    data: LearningCurveData,
    /// Title for the plot
    title: String,
    /// Whether to show standard deviation
    show_std: bool,
    /// Scoring metric name
    scoring: String,
}

impl LearningCurveVisualizer {
    /// Create a new LearningCurveVisualizer
    ///
    /// # Arguments
    ///
    /// * `data` - Learning curve data
    ///
    /// # Returns
    ///
    /// * A new LearningCurveVisualizer
    pub fn new(data: LearningCurveData) -> Self {
        LearningCurveVisualizer {
            data,
            title: "Learning Curve".to_string(),
            show_std: true,
            scoring: "Score".to_string(),
        }
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

    /// Set whether to show standard deviation
    ///
    /// # Arguments
    ///
    /// * `show_std` - Whether to show standard deviation
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_std(mut self, show_std: bool) -> Self {
        self.show_std = show_std;
        self
    }

    /// Set the scoring metric name
    ///
    /// # Arguments
    ///
    /// * `scoring` - Scoring metric name
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_scoring(mut self, scoring: String) -> Self {
        self.scoring = scoring;
        self
    }

    /// Compute mean and standard deviation of scores
    ///
    /// # Arguments
    ///
    /// * `scores` - Scores for each training set size
    ///
    /// # Returns
    ///
    /// * (mean_scores, std_scores)
    fn compute_statistics(&self, scores: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n = scores.len();
        let mut mean_scores = Vec::with_capacity(n);
        let mut std_scores = Vec::with_capacity(n);

        for fold_scores in scores {
            // Compute mean
            let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            mean_scores.push(mean);

            // Compute standard deviation
            let variance = fold_scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                / fold_scores.len() as f64;
            std_scores.push(variance.sqrt());
        }

        (mean_scores, std_scores)
    }
}

impl MetricVisualizer for LearningCurveVisualizer {
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        // Compute statistics for train and validation scores
        let (train_mean, train_std) = self.compute_statistics(&self.data.train_scores);
        let (val_mean, val_std) = self.compute_statistics(&self.data.validation_scores);

        // Convert train_sizes to f64 for plotting
        let train_sizes: Vec<f64> = self.data.train_sizes.iter().map(|&s| s as f64).collect();

        // Prepare data for visualization
        let mut x = Vec::new();
        let mut y = Vec::new();

        // Add training scores
        x.extend_from_slice(&train_sizes);
        y.extend_from_slice(&train_mean);

        // Add validation scores
        x.extend_from_slice(&train_sizes);
        y.extend_from_slice(&val_mean);

        // Prepare series names
        let mut series_names = vec!["Training score".to_string(), "Validation score".to_string()];

        // Add standard deviation series if requested
        if self.show_std {
            // Add upper and lower bounds for training scores
            x.extend_from_slice(&train_sizes);
            x.extend_from_slice(&train_sizes);

            let train_upper: Vec<f64> = train_mean
                .iter()
                .zip(train_std.iter())
                .map(|(&m, &s)| m + s)
                .collect();

            let train_lower: Vec<f64> = train_mean
                .iter()
                .zip(train_std.iter())
                .map(|(&m, &s)| m - s)
                .collect();

            y.extend_from_slice(&train_upper);
            y.extend_from_slice(&train_lower);

            // Add upper and lower bounds for validation scores
            x.extend_from_slice(&train_sizes);
            x.extend_from_slice(&train_sizes);

            let val_upper: Vec<f64> = val_mean
                .iter()
                .zip(val_std.iter())
                .map(|(&m, &s)| m + s)
                .collect();

            let val_lower: Vec<f64> = val_mean
                .iter()
                .zip(val_std.iter())
                .map(|(&m, &s)| m - s)
                .collect();

            y.extend_from_slice(&val_upper);
            y.extend_from_slice(&val_lower);

            // Add series names for standard deviation bounds
            series_names.push("Training score +/- std".to_string());
            series_names.push("Training score +/- std".to_string());
            series_names.push("Validation score +/- std".to_string());
            series_names.push("Validation score +/- std".to_string());
        }

        Ok(VisualizationData {
            x,
            y,
            z: None,
            series_names: Some(series_names),
            x_labels: None,
            y_labels: None,
            auxiliary_data: std::collections::HashMap::new(),
            auxiliary_metadata: std::collections::HashMap::new(),
            series: std::collections::HashMap::new(),
        })
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        VisualizationMetadata {
            title: self.title.clone(),
            x_label: "Training examples".to_string(),
            y_label: self.scoring.clone(),
            plot_type: PlotType::Line,
            description: Some(
                "Learning curve showing model performance as a function of training set size"
                    .to_string(),
            ),
        }
    }
}

/// Create a learning curve visualization
///
/// # Arguments
///
/// * `train_sizes` - Training set sizes
/// * `train_scores` - Training scores for each training set size
/// * `validation_scores` - Validation scores for each training set size
/// * `scoring` - Scoring metric name
///
/// # Returns
///
/// * A LearningCurveVisualizer
pub fn learning_curve_visualization(
    train_sizes: Vec<usize>,
    train_scores: Vec<Vec<f64>>,
    validation_scores: Vec<Vec<f64>>,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer> {
    // Validate inputs
    if train_sizes.is_empty() || train_scores.is_empty() || validation_scores.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Learning curve data cannot be empty".to_string(),
        ));
    }

    if train_scores.len() != train_sizes.len() || validation_scores.len() != train_sizes.len() {
        return Err(MetricsError::InvalidInput(
            "Number of train/validation scores must match number of training sizes".to_string(),
        ));
    }

    let data = LearningCurveData {
        train_sizes,
        train_scores,
        validation_scores,
    };

    let scoring_string = scoring.into();
    Ok(LearningCurveVisualizer::new(data).with_scoring(scoring_string))
}

/// Generate a learning curve by evaluating a model on different training set sizes
///
/// This function is a placeholder for the actual implementation, which would require
/// integration with model training functionality.
///
/// # Arguments
///
/// * `X` - Feature matrix
/// * `y` - Target values
/// * `model` - Model to evaluate (would typically be a trait object implementing a Model trait)
/// * `train_sizes` - Training set sizes to evaluate
/// * `cv` - Number of cross-validation folds
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A LearningCurveVisualizer
pub fn learning_curve<T, S1, S2>(
    _x: &ArrayBase<S1, Ix2>,
    _y: &ArrayBase<S2, Ix1>,
    _model: &impl Clone,
    train_sizes: &[usize],
    _cv: usize,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer>
where
    T: Clone + 'static,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    // This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Split the data into cross-validation folds
    // 2. For each training set size:
    //    a. Subsample the training data
    //    b. Train the model on the subsampled data
    //    c. Evaluate on both the training data and validation data
    //    d. Record the scores

    // For now, just return a dummy learning curve
    let n_sizes = train_sizes.len();
    let train_scores = vec![vec![0.8, 0.82, 0.85]; n_sizes];
    let validation_scores = vec![vec![0.7, 0.72, 0.75]; n_sizes];

    learning_curve_visualization(
        train_sizes.to_vec(),
        train_scores,
        validation_scores,
        scoring,
    )
}
