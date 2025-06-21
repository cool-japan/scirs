//! Online/streaming evaluation capabilities
//!
//! This module provides utilities for computing metrics incrementally as new data
//! points arrive, enabling real-time evaluation and handling of large datasets
//! that don't fit in memory.
//!
//! # Features
//!
//! - **Incremental computation**: Update metrics with new data points one at a time
//! - **Memory efficient**: Constant memory usage regardless of data size
//! - **Windowed metrics**: Track metrics over sliding windows
//! - **Multiple metric types**: Classification, regression, and ranking metrics
//! - **Reset capabilities**: Start new evaluation periods
//!
//! # Examples
//!
//! ## Basic Streaming Classification Metrics
//!
//! ```
//! use scirs2_metrics::streaming::StreamingClassificationMetrics;
//!
//! let mut metrics = StreamingClassificationMetrics::new();
//!
//! // Process data points one at a time
//! metrics.update(1, 1);  // true_label=1, pred_label=1 (correct)
//! metrics.update(0, 1);  // true_label=0, pred_label=1 (incorrect)
//! metrics.update(1, 1);  // true_label=1, pred_label=1 (correct)
//!
//! println!("Current accuracy: {:.4}", metrics.accuracy());
//! println!("Total samples processed: {}", metrics.sample_count());
//! ```
//!
//! ## Windowed Metrics
//!
//! ```
//! use scirs2_metrics::streaming::WindowedClassificationMetrics;
//!
//! let mut metrics = WindowedClassificationMetrics::new(100); // Window size of 100
//!
//! for i in 0..150 {
//!     let true_label = i % 2;
//!     let pred_label = if i < 75 { i % 2 } else { (i + 1) % 2 }; // Accuracy degrades
//!     metrics.update(true_label, pred_label);
//!     
//!     if i % 25 == 0 {
//!         println!("Accuracy at step {}: {:.4}", i, metrics.accuracy());
//!     }
//! }
//! ```

use crate::error::{MetricsError, Result};
use num_traits::Float;
use std::collections::VecDeque;

/// Streaming classification metrics with incremental computation
#[derive(Debug, Clone)]
pub struct StreamingClassificationMetrics {
    total_samples: usize,
    correct_predictions: usize,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
}

impl Default for StreamingClassificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingClassificationMetrics {
    /// Creates a new streaming classification metrics calculator
    pub fn new() -> Self {
        Self {
            total_samples: 0,
            correct_predictions: 0,
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
        }
    }

    /// Updates metrics with a new prediction (binary classification)
    pub fn update(&mut self, true_label: i32, pred_label: i32) {
        self.total_samples += 1;

        if true_label == pred_label {
            self.correct_predictions += 1;
        }

        // Assuming binary classification (0/1)
        match (true_label, pred_label) {
            (1, 1) => self.true_positives += 1,
            (0, 1) => self.false_positives += 1,
            (0, 0) => self.true_negatives += 1,
            (1, 0) => self.false_negatives += 1,
            _ => {} // Handle multi-class later
        }
    }

    /// Updates metrics with multiple predictions at once
    pub fn update_batch(&mut self, true_labels: &[i32], pred_labels: &[i32]) -> Result<()> {
        if true_labels.len() != pred_labels.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted labels must have the same length".to_string(),
            ));
        }

        for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
            self.update(true_label, pred_label);
        }

        Ok(())
    }

    /// Computes current accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_samples as f64
        }
    }

    /// Computes current precision
    pub fn precision(&self) -> f64 {
        let total_positive_predictions = self.true_positives + self.false_positives;
        if total_positive_predictions == 0 {
            0.0
        } else {
            self.true_positives as f64 / total_positive_predictions as f64
        }
    }

    /// Computes current recall (sensitivity)
    pub fn recall(&self) -> f64 {
        let total_actual_positives = self.true_positives + self.false_negatives;
        if total_actual_positives == 0 {
            0.0
        } else {
            self.true_positives as f64 / total_actual_positives as f64
        }
    }

    /// Computes current F1 score
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Computes current specificity
    pub fn specificity(&self) -> f64 {
        let total_actual_negatives = self.true_negatives + self.false_positives;
        if total_actual_negatives == 0 {
            0.0
        } else {
            self.true_negatives as f64 / total_actual_negatives as f64
        }
    }

    /// Gets the total number of samples processed
    pub fn sample_count(&self) -> usize {
        self.total_samples
    }

    /// Gets the confusion matrix components
    pub fn confusion_matrix(&self) -> (usize, usize, usize, usize) {
        (
            self.true_positives,
            self.false_positives,
            self.true_negatives,
            self.false_negatives,
        )
    }

    /// Resets all metrics to start a new evaluation period
    pub fn reset(&mut self) {
        self.total_samples = 0;
        self.correct_predictions = 0;
        self.true_positives = 0;
        self.false_positives = 0;
        self.true_negatives = 0;
        self.false_negatives = 0;
    }
}

/// Streaming regression metrics with incremental computation
#[derive(Debug, Clone)]
pub struct StreamingRegressionMetrics<F: Float> {
    total_samples: usize,
    sum_squared_errors: F,
    sum_absolute_errors: F,
    sum_true_values: F,
    sum_true_squared: F,
    sum_pred_values: F,
    min_error: Option<F>,
    max_error: Option<F>,
}

impl<F: Float> Default for StreamingRegressionMetrics<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> StreamingRegressionMetrics<F> {
    /// Creates a new streaming regression metrics calculator
    pub fn new() -> Self {
        Self {
            total_samples: 0,
            sum_squared_errors: F::zero(),
            sum_absolute_errors: F::zero(),
            sum_true_values: F::zero(),
            sum_true_squared: F::zero(),
            sum_pred_values: F::zero(),
            min_error: None,
            max_error: None,
        }
    }

    /// Updates metrics with a new prediction
    pub fn update(&mut self, true_value: F, pred_value: F) {
        self.total_samples += 1;

        let error = true_value - pred_value;
        let abs_error = error.abs();
        let squared_error = error * error;

        self.sum_squared_errors = self.sum_squared_errors + squared_error;
        self.sum_absolute_errors = self.sum_absolute_errors + abs_error;
        self.sum_true_values = self.sum_true_values + true_value;
        self.sum_true_squared = self.sum_true_squared + (true_value * true_value);
        self.sum_pred_values = self.sum_pred_values + pred_value;

        // Update min/max error
        match self.min_error {
            None => self.min_error = Some(abs_error),
            Some(current_min) => {
                if abs_error < current_min {
                    self.min_error = Some(abs_error);
                }
            }
        }

        match self.max_error {
            None => self.max_error = Some(abs_error),
            Some(current_max) => {
                if abs_error > current_max {
                    self.max_error = Some(abs_error);
                }
            }
        }
    }

    /// Updates metrics with multiple predictions at once
    pub fn update_batch(&mut self, true_values: &[F], pred_values: &[F]) -> Result<()> {
        if true_values.len() != pred_values.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted values must have the same length".to_string(),
            ));
        }

        for (&true_value, &pred_value) in true_values.iter().zip(pred_values.iter()) {
            self.update(true_value, pred_value);
        }

        Ok(())
    }

    /// Computes current Mean Squared Error (MSE)
    pub fn mse(&self) -> F {
        if self.total_samples == 0 {
            F::zero()
        } else {
            self.sum_squared_errors / F::from(self.total_samples).unwrap()
        }
    }

    /// Computes current Root Mean Squared Error (RMSE)
    pub fn rmse(&self) -> F {
        self.mse().sqrt()
    }

    /// Computes current Mean Absolute Error (MAE)
    pub fn mae(&self) -> F {
        if self.total_samples == 0 {
            F::zero()
        } else {
            self.sum_absolute_errors / F::from(self.total_samples).unwrap()
        }
    }

    /// Computes current R² score
    pub fn r2_score(&self) -> F {
        if self.total_samples == 0 {
            F::zero()
        } else {
            let n = F::from(self.total_samples).unwrap();
            let mean_true = self.sum_true_values / n;

            // Total sum of squares
            let ss_tot = self.sum_true_squared - n * mean_true * mean_true;

            // Residual sum of squares
            let ss_res = self.sum_squared_errors;

            if ss_tot == F::zero() {
                F::zero()
            } else {
                F::one() - (ss_res / ss_tot)
            }
        }
    }

    /// Gets the minimum absolute error seen so far
    pub fn min_error(&self) -> Option<F> {
        self.min_error
    }

    /// Gets the maximum absolute error seen so far
    pub fn max_error(&self) -> Option<F> {
        self.max_error
    }

    /// Gets the total number of samples processed
    pub fn sample_count(&self) -> usize {
        self.total_samples
    }

    /// Resets all metrics to start a new evaluation period
    pub fn reset(&mut self) {
        self.total_samples = 0;
        self.sum_squared_errors = F::zero();
        self.sum_absolute_errors = F::zero();
        self.sum_true_values = F::zero();
        self.sum_true_squared = F::zero();
        self.sum_pred_values = F::zero();
        self.min_error = None;
        self.max_error = None;
    }
}

/// Windowed classification metrics using a sliding window
#[derive(Debug, Clone)]
pub struct WindowedClassificationMetrics {
    window_size: usize,
    predictions: VecDeque<(i32, i32)>, // (true_label, pred_label)
    metrics: StreamingClassificationMetrics,
}

impl WindowedClassificationMetrics {
    /// Creates a new windowed classification metrics calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            predictions: VecDeque::with_capacity(window_size),
            metrics: StreamingClassificationMetrics::new(),
        }
    }

    /// Updates metrics with a new prediction, maintaining the sliding window
    pub fn update(&mut self, true_label: i32, pred_label: i32) {
        // If window is full, remove the oldest prediction
        if self.predictions.len() >= self.window_size {
            if let Some((old_true, old_pred)) = self.predictions.pop_front() {
                // Subtract the old prediction from metrics
                self.subtract_prediction(old_true, old_pred);
            }
        }

        // Add the new prediction
        self.predictions.push_back((true_label, pred_label));
        self.metrics.update(true_label, pred_label);
    }

    /// Removes a prediction from the metrics (for sliding window)
    fn subtract_prediction(&mut self, true_label: i32, pred_label: i32) {
        if self.metrics.total_samples > 0 {
            self.metrics.total_samples -= 1;
        }

        if true_label == pred_label && self.metrics.correct_predictions > 0 {
            self.metrics.correct_predictions -= 1;
        }

        match (true_label, pred_label) {
            (1, 1) => {
                if self.metrics.true_positives > 0 {
                    self.metrics.true_positives -= 1;
                }
            }
            (0, 1) => {
                if self.metrics.false_positives > 0 {
                    self.metrics.false_positives -= 1;
                }
            }
            (0, 0) => {
                if self.metrics.true_negatives > 0 {
                    self.metrics.true_negatives -= 1;
                }
            }
            (1, 0) => {
                if self.metrics.false_negatives > 0 {
                    self.metrics.false_negatives -= 1;
                }
            }
            _ => {}
        }
    }

    /// Gets the current window size (number of predictions in window)
    pub fn current_window_size(&self) -> usize {
        self.predictions.len()
    }

    /// Gets the maximum window size
    pub fn max_window_size(&self) -> usize {
        self.window_size
    }

    /// Delegates metric calculations to the underlying streaming metrics
    pub fn accuracy(&self) -> f64 {
        self.metrics.accuracy()
    }

    pub fn precision(&self) -> f64 {
        self.metrics.precision()
    }

    pub fn recall(&self) -> f64 {
        self.metrics.recall()
    }

    pub fn f1_score(&self) -> f64 {
        self.metrics.f1_score()
    }

    pub fn sample_count(&self) -> usize {
        self.metrics.sample_count()
    }

    /// Clears the window and resets metrics
    pub fn reset(&mut self) {
        self.predictions.clear();
        self.metrics.reset();
    }
}

/// Windowed regression metrics using a sliding window
#[derive(Debug, Clone)]
pub struct WindowedRegressionMetrics<F: Float> {
    window_size: usize,
    predictions: VecDeque<(F, F)>, // (true_value, pred_value)
}

impl<F: Float> WindowedRegressionMetrics<F> {
    /// Creates a new windowed regression metrics calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            predictions: VecDeque::with_capacity(window_size),
        }
    }

    /// Updates metrics with a new prediction, maintaining the sliding window
    pub fn update(&mut self, true_value: F, pred_value: F) {
        // If window is full, remove the oldest prediction
        if self.predictions.len() >= self.window_size {
            self.predictions.pop_front();
        }

        // Add the new prediction
        self.predictions.push_back((true_value, pred_value));
    }

    /// Computes MSE over the current window
    pub fn mse(&self) -> F {
        if self.predictions.is_empty() {
            return F::zero();
        }

        let sum_squared_errors = self
            .predictions
            .iter()
            .map(|(true_val, pred_val)| {
                let error = *true_val - *pred_val;
                error * error
            })
            .fold(F::zero(), |acc, x| acc + x);

        sum_squared_errors / F::from(self.predictions.len()).unwrap()
    }

    /// Computes RMSE over the current window
    pub fn rmse(&self) -> F {
        self.mse().sqrt()
    }

    /// Computes MAE over the current window
    pub fn mae(&self) -> F {
        if self.predictions.is_empty() {
            return F::zero();
        }

        let sum_absolute_errors = self
            .predictions
            .iter()
            .map(|(true_val, pred_val)| (*true_val - *pred_val).abs())
            .fold(F::zero(), |acc, x| acc + x);

        sum_absolute_errors / F::from(self.predictions.len()).unwrap()
    }

    /// Gets the current window size (number of predictions in window)
    pub fn current_window_size(&self) -> usize {
        self.predictions.len()
    }

    /// Gets the maximum window size
    pub fn max_window_size(&self) -> usize {
        self.window_size
    }

    /// Clears the window
    pub fn reset(&mut self) {
        self.predictions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_classification_metrics() {
        let mut metrics = StreamingClassificationMetrics::new();

        // Test initial state
        assert_eq!(metrics.accuracy(), 0.0);
        assert_eq!(metrics.sample_count(), 0);

        // Test perfect predictions
        metrics.update(1, 1);
        metrics.update(0, 0);
        metrics.update(1, 1);

        assert_eq!(metrics.accuracy(), 1.0);
        assert_eq!(metrics.sample_count(), 3);
        assert_eq!(metrics.precision(), 1.0);
        assert_eq!(metrics.recall(), 1.0);
        assert_eq!(metrics.f1_score(), 1.0);

        // Test some incorrect predictions
        metrics.update(1, 0); // False negative
        metrics.update(0, 1); // False positive

        assert_eq!(metrics.accuracy(), 0.6); // 3/5 correct
        assert_eq!(metrics.sample_count(), 5);

        let (tp, fp, tn, fn_) = metrics.confusion_matrix();
        assert_eq!(tp, 2);
        assert_eq!(fp, 1);
        assert_eq!(tn, 1);
        assert_eq!(fn_, 1);
    }

    #[test]
    fn test_streaming_regression_metrics() {
        let mut metrics = StreamingRegressionMetrics::<f64>::new();

        // Test initial state
        assert_eq!(metrics.mse(), 0.0);
        assert_eq!(metrics.mae(), 0.0);
        assert_eq!(metrics.sample_count(), 0);

        // Test perfect predictions
        metrics.update(1.0, 1.0);
        metrics.update(2.0, 2.0);
        metrics.update(3.0, 3.0);

        assert_eq!(metrics.mse(), 0.0);
        assert_eq!(metrics.mae(), 0.0);
        assert_eq!(metrics.rmse(), 0.0);
        assert_eq!(metrics.sample_count(), 3);

        // Test with some errors
        metrics.update(4.0, 5.0); // Error = -1.0, squared = 1.0, abs = 1.0
        metrics.update(6.0, 4.0); // Error = 2.0, squared = 4.0, abs = 2.0

        // MSE = (0 + 0 + 0 + 1 + 4) / 5 = 1.0
        assert_eq!(metrics.mse(), 1.0);
        // MAE = (0 + 0 + 0 + 1 + 2) / 5 = 0.6
        assert_eq!(metrics.mae(), 0.6);
        assert_eq!(metrics.rmse(), 1.0);
        assert_eq!(metrics.min_error(), Some(0.0));
        assert_eq!(metrics.max_error(), Some(2.0));
    }

    #[test]
    fn test_windowed_classification_metrics() {
        let mut metrics = WindowedClassificationMetrics::new(3);

        assert_eq!(metrics.current_window_size(), 0);
        assert_eq!(metrics.max_window_size(), 3);

        // Fill the window
        metrics.update(1, 1); // Correct
        metrics.update(0, 0); // Correct
        metrics.update(1, 0); // Incorrect

        assert_eq!(metrics.current_window_size(), 3);
        assert_eq!(metrics.accuracy(), 2.0 / 3.0);

        // Add one more - should remove the first correct prediction
        metrics.update(0, 1); // Incorrect

        assert_eq!(metrics.current_window_size(), 3);
        assert_eq!(metrics.accuracy(), 1.0 / 3.0); // Only one correct (0,0) remaining
    }

    #[test]
    fn test_windowed_regression_metrics() {
        let mut metrics = WindowedRegressionMetrics::<f64>::new(2);

        assert_eq!(metrics.current_window_size(), 0);
        assert_eq!(metrics.max_window_size(), 2);

        // Add two predictions
        metrics.update(1.0, 1.0); // Perfect
        metrics.update(2.0, 3.0); // Error = -1.0, squared = 1.0

        assert_eq!(metrics.current_window_size(), 2);
        assert_eq!(metrics.mse(), 0.5); // (0 + 1) / 2
        assert_eq!(metrics.mae(), 0.5); // (0 + 1) / 2

        // Add third prediction - should remove the first
        metrics.update(4.0, 2.0); // Error = 2.0, squared = 4.0

        assert_eq!(metrics.current_window_size(), 2);
        assert_eq!(metrics.mse(), 2.5); // (1 + 4) / 2
        assert_eq!(metrics.mae(), 1.5); // (1 + 2) / 2
    }

    #[test]
    fn test_batch_updates() {
        let mut metrics = StreamingClassificationMetrics::new();

        let true_labels = vec![1, 0, 1, 0];
        let pred_labels = vec![1, 0, 0, 1];

        metrics.update_batch(&true_labels, &pred_labels).unwrap();

        assert_eq!(metrics.sample_count(), 4);
        assert_eq!(metrics.accuracy(), 0.5); // 2 correct out of 4
    }

    #[test]
    fn test_reset_functionality() {
        let mut metrics = StreamingClassificationMetrics::new();

        metrics.update(1, 1);
        metrics.update(0, 0);

        assert_eq!(metrics.sample_count(), 2);
        assert_eq!(metrics.accuracy(), 1.0);

        metrics.reset();

        assert_eq!(metrics.sample_count(), 0);
        assert_eq!(metrics.accuracy(), 0.0);
    }
}
