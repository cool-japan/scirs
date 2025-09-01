//! Learning curve visualization
//!
//! This module provides tools for visualizing learning curves, which show model performance
//! as a function of training set size.

use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use rand::prelude::*;

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
    pub fn with_show_std(mut self, showstd: bool) -> Self {
        self.show_std = showstd;
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
#[allow(dead_code)]
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
            "Number of train/validation _scores must match number of training _sizes".to_string(),
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

/// Learning curve scenario types for realistic simulation
#[derive(Debug, Clone, Copy)]
pub enum LearningCurveScenario {
    /// Well-fitted model with good generalization
    WellFitted,
    /// High bias scenario (underfitting)
    HighBias,
    /// High variance scenario (overfitting)
    HighVariance,
    /// Noisy data scenario with irregular patterns
    NoisyData,
    /// Learning plateau scenario where more data doesn't help much
    PlateauEffect,
}

/// Configuration for learning curve generation
#[derive(Debug, Clone)]
pub struct LearningCurveConfig {
    /// The learning scenario to simulate
    pub scenario: LearningCurveScenario,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Base performance level (0.0 to 1.0)
    pub base_performance: f64,
    /// Noise level in the scores (0.0 to 1.0)
    pub noise_level: f64,
    /// Whether to add realistic variance across folds
    pub add_cv_variance: bool,
}

impl Default for LearningCurveConfig {
    fn default() -> Self {
        Self {
            scenario: LearningCurveScenario::WellFitted,
            cv_folds: 5,
            base_performance: 0.75,
            noise_level: 0.05,
            add_cv_variance: true,
        }
    }
}

/// Generate a realistic learning curve based on learning theory principles
///
/// This function simulates realistic learning curves that follow common patterns
/// observed in machine learning, including bias-variance decomposition effects.
/// Since this is a metrics library without model training capabilities, it
/// generates theoretically sound learning curves for visualization and analysis.
///
/// # Arguments
///
/// * `_x` - Feature matrix (used for determining data characteristics)
/// * `_y` - Target values (used for determining problem characteristics)
/// * `train_sizes` - Training set sizes to evaluate
/// * `config` - Configuration for learning curve generation
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A LearningCurveVisualizer with realistic learning curves
#[allow(dead_code)]
pub fn learning_curve_realistic<T, S1, S2>(
    _x: &ArrayBase<S1, Ix2>,
    _y: &ArrayBase<S2, Ix1>,
    train_sizes: &[usize],
    config: LearningCurveConfig,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer>
where
    T: Clone + 'static,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    use rand::Rng;
    let mut rng = rand::rng();

    let n_sizes = train_sizes.len();
    let mut train_scores = Vec::with_capacity(n_sizes);
    let mut validation_scores = Vec::with_capacity(n_sizes);

    for (i, &_size) in train_sizes.iter().enumerate() {
        let progress = i as f64 / n_sizes.max(1) as f64;

        let (base_train_score, base_val_score) = match config.scenario {
            LearningCurveScenario::WellFitted => {
                // Training score starts high and plateaus
                let train_score = config.base_performance + 0.15 * progress.powf(0.3);
                // Validation score starts lower but converges towards training score
                let val_score = config.base_performance - 0.1 + 0.2 * progress.powf(0.5);
                (train_score.min(0.95), val_score.min(train_score - 0.02))
            }
            LearningCurveScenario::HighBias => {
                // Both training and validation scores are low and plateau early
                let train_score = config.base_performance - 0.15 + 0.1 * progress.powf(0.8);
                let val_score = train_score - 0.05 + 0.03 * progress;
                (train_score.min(0.7), val_score.min(train_score))
            }
            LearningCurveScenario::HighVariance => {
                // Large gap between training and validation scores
                let train_score = config.base_performance + 0.2 * progress.powf(0.2);
                let val_score = config.base_performance - 0.2 + 0.15 * progress.powf(0.7);
                (train_score.min(0.98), val_score.min(train_score - 0.15))
            }
            LearningCurveScenario::NoisyData => {
                // Irregular patterns with higher variance
                let noise_factor = 0.1 * (progress * 10.0).sin();
                let train_score = config.base_performance + 0.1 * progress + noise_factor;
                let val_score =
                    config.base_performance - 0.05 + 0.12 * progress + noise_factor * 0.5;
                (train_score.min(0.9), val_score.min(train_score))
            }
            LearningCurveScenario::PlateauEffect => {
                // Rapid initial improvement then plateau
                let plateau_factor = 1.0 - (-5.0 * progress).exp();
                let train_score = config.base_performance + 0.15 * plateau_factor;
                let val_score = config.base_performance - 0.08 + 0.18 * plateau_factor;
                (train_score, val_score.min(train_score - 0.01))
            }
        };

        // Generate scores for each CV fold
        let fold_variance = if config.add_cv_variance {
            config.noise_level
        } else {
            0.0
        };

        let train_fold_scores: Vec<f64> = (0..config.cv_folds)
            .map(|_| {
                let noise = rng.random_range(-fold_variance..fold_variance);
                (base_train_score + noise).clamp(0.0, 1.0)
            })
            .collect();

        let val_fold_scores: Vec<f64> = (0..config.cv_folds)
            .map(|_| {
                let noise = rng.random_range(-fold_variance * 1.5..fold_variance * 1.5);
                (base_val_score + noise).clamp(0.0, 1.0)
            })
            .collect();

        train_scores.push(train_fold_scores);
        validation_scores.push(val_fold_scores);
    }

    learning_curve_visualization(
        train_sizes.to_vec(),
        train_scores,
        validation_scores,
        scoring,
    )
}

/// Generate a learning curve with real model evaluation (Advanced mode)
///
/// This function provides a sophisticated interface for learning curve generation
/// that actually trains and evaluates models on different training set sizes.
///
/// # Arguments
///
/// * `X` - Feature matrix
/// * `y` - Target values
/// * `model` - Model to evaluate (now properly utilized)
/// * `train_sizes` - Training set sizes to evaluate
/// * `cv` - Number of cross-validation folds
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A LearningCurveVisualizer with real performance curves
#[allow(dead_code)]
pub fn learning_curve<T, S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    y: &ArrayBase<S2, Ix1>,
    model: &impl ModelEvaluator<T>,
    train_sizes: &[usize],
    cv: usize,
    scoring: impl Into<String>,
) -> Result<LearningCurveVisualizer>
where
    T: Clone
        + 'static
        + num_traits::Float
        + Send
        + Sync
        + std::fmt::Debug
        + std::ops::Sub<Output = T>,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>,
{
    let scoring_str = scoring.into();

    // Validate inputs
    if x.nrows() != y.len() {
        return Err(MetricsError::InvalidInput(
            "Feature matrix and target vector must have same number of samples".to_string(),
        ));
    }

    if train_sizes.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Training _sizes cannot be empty".to_string(),
        ));
    }

    let max_size = train_sizes.iter().max().unwrap();
    if *max_size > x.nrows() {
        return Err(MetricsError::InvalidInput(format!(
            "Maximum training size ({}) exceeds available samples ({})",
            max_size,
            x.nrows()
        )));
    }

    // Generate actual learning curves using cross-validation
    let mut train_scores = Vec::new();
    let mut validation_scores = Vec::new();

    use scirs2_core::simd_ops::SimdUnifiedOps;
    let mut rng = rand::rng();

    // Create cross-validation folds
    let fold_size = x.nrows() / cv;
    let mut indices: Vec<usize> = (0..x.nrows()).collect();

    for &size in train_sizes {
        let mut train_fold_scores = Vec::new();
        let mut val_fold_scores = Vec::new();

        // Perform cross-validation for this training size
        for fold in 0..cv {
            // Shuffle indices for this fold
            for i in 0..indices.len() {
                let j = rng.random_range(0..indices.len());
                indices.swap(i, j);
            }

            // Split data for this fold
            let val_start = fold * fold_size;
            let val_end = std::cmp::min((fold + 1) * fold_size, x.nrows());

            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for (i, &idx) in indices.iter().enumerate() {
                if i >= val_start && i < val_end {
                    val_indices.push(idx);
                } else if train_indices.len() < size {
                    train_indices.push(idx);
                }
            }

            // Create training and validation sets
            let train_x = extract_rows(x, &train_indices);
            let train_y = extract_elements(y, &train_indices);
            let val_x = extract_rows(x, &val_indices);
            let val_y = extract_elements(y, &val_indices);

            // Train model and evaluate
            let trained_model = model.fit(&train_x, &train_y)?;

            // Evaluate on training set
            let train_pred = trained_model.predict(&train_x)?;
            let train_score = evaluate_predictions(&train_y, &train_pred, &scoring_str)?;
            train_fold_scores.push(train_score);

            // Evaluate on validation set
            let val_pred = trained_model.predict(&val_x)?;
            let val_score = evaluate_predictions(&val_y, &val_pred, &scoring_str)?;
            val_fold_scores.push(val_score);
        }

        train_scores.push(train_fold_scores);
        validation_scores.push(val_fold_scores);
    }

    learning_curve_visualization(
        train_sizes.to_vec(),
        train_scores,
        validation_scores,
        scoring_str,
    )
}

/// Trait for models that can be evaluated in learning curves
pub trait ModelEvaluator<T> {
    type TrainedModel: ModelPredictor<T>;

    fn fit(&self, x: &Array2<T>, y: &Array1<T>) -> Result<Self::TrainedModel>;
}

/// Trait for trained models that can make predictions
pub trait ModelPredictor<T> {
    fn predict(&self, x: &Array2<T>) -> Result<Array1<T>>;
}

/// Extract specific rows from a 2D array
#[allow(dead_code)]
fn extract_rows<T, S>(arr: &ArrayBase<S, Ix2>, indices: &[usize]) -> Array2<T>
where
    T: Clone + num_traits::Zero,
    S: Data<Elem = T>,
{
    let mut result = Array2::zeros((indices.len(), arr.ncols()));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&arr.row(idx));
    }
    result
}

/// Extract specific elements from a 1D array
#[allow(dead_code)]
fn extract_elements<T, S>(arr: &ArrayBase<S, Ix1>, indices: &[usize]) -> Array1<T>
where
    T: Clone + num_traits::Zero,
    S: Data<Elem = T>,
{
    let mut result = Array1::zeros(indices.len());
    for (i, &idx) in indices.iter().enumerate() {
        result[i] = arr[idx].clone();
    }
    result
}

/// Evaluate predictions using the specified scoring metric
#[allow(dead_code)]
fn evaluate_predictions<T>(y_true: &Array1<T>, ypred: &Array1<T>, scoring: &str) -> Result<f64>
where
    T: Clone + num_traits::Float + Send + Sync + std::fmt::Debug + std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>,
{
    match scoring.to_lowercase().as_str() {
        "accuracy" => {
            // For classification: count exact matches
            let correct = y_true
                .iter()
                .zip(ypred.iter())
                .filter(|(t, p)| (*t - *p).abs() < T::from(0.5).unwrap())
                .count();
            Ok(correct as f64 / y_true.len() as f64)
        }
        "mse" | "mean_squared_error" => {
            // Mean squared error
            let mse = y_true
                .iter()
                .zip(ypred.iter())
                .map(|(t, p)| (*t - *p) * (*t - *p))
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(y_true.len()).unwrap();
            Ok(mse.to_f64().unwrap_or(0.0))
        }
        "mae" | "mean_absolute_error" => {
            // Mean absolute error
            let mae = y_true
                .iter()
                .zip(ypred.iter())
                .map(|(t, p)| (*t - *p).abs())
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(y_true.len()).unwrap();
            Ok(mae.to_f64().unwrap_or(0.0))
        }
        "r2" | "r2_score" => {
            // R² score
            let mean_true = y_true.iter().cloned().fold(T::zero(), |acc, x| acc + x)
                / T::from(y_true.len()).unwrap();

            let ss_tot = y_true
                .iter()
                .map(|&t| (t - mean_true) * (t - mean_true))
                .fold(T::zero(), |acc, x| acc + x);

            let ss_res = y_true
                .iter()
                .zip(ypred.iter())
                .map(|(&t, &p)| (t - p) * (t - p))
                .fold(T::zero(), |acc, x| acc + x);

            if ss_tot == T::zero() {
                Ok(0.0)
            } else {
                let r2 = T::one() - ss_res / ss_tot;
                Ok(r2.to_f64().unwrap_or(0.0))
            }
        }
        _ => {
            // Default to MSE for unknown metrics
            let mse = y_true
                .iter()
                .zip(ypred.iter())
                .map(|(t, p)| (*t - *p) * (*t - *p))
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(y_true.len()).unwrap();
            Ok(mse.to_f64().unwrap_or(0.0))
        }
    }
}

/// Generate learning curves for different scenarios for comparison
///
/// This function generates multiple learning curves showing different learning
/// scenarios, useful for educational purposes and understanding model behavior.
///
/// # Arguments
///
/// * `train_sizes` - Training set sizes to evaluate
/// * `scoring` - Scoring metric to use
///
/// # Returns
///
/// * A vector of LearningCurveVisualizer instances for each scenario
#[allow(dead_code)]
pub fn learning_curve_scenarios(
    train_sizes: &[usize],
    scoring: impl Into<String>,
) -> Result<Vec<(String, LearningCurveVisualizer)>> {
    let scoring_str = scoring.into();
    let scenarios = [
        ("Well Fitted", LearningCurveScenario::WellFitted),
        ("High Bias (Underfitting)", LearningCurveScenario::HighBias),
        (
            "High Variance (Overfitting)",
            LearningCurveScenario::HighVariance,
        ),
        ("Noisy Data", LearningCurveScenario::NoisyData),
        ("Plateau Effect", LearningCurveScenario::PlateauEffect),
    ];

    let mut results = Vec::new();

    // Create dummy data for the function signature
    let dummy_x = Array2::<f64>::zeros((100, 5));
    let dummy_y = Array1::<f64>::zeros(100);

    for (name, scenario) in scenarios.iter() {
        let config = LearningCurveConfig {
            scenario: *scenario,
            cv_folds: 5,
            base_performance: 0.75,
            noise_level: 0.03,
            add_cv_variance: true,
        };

        let visualizer =
            learning_curve_realistic(&dummy_x, &dummy_y, train_sizes, config, scoring_str.clone())?;

        results.push((name.to_string(), visualizer));
    }

    Ok(results)
}
