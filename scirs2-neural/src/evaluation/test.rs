//! Test set evaluation utilities
//!
//! This module provides utilities for evaluating models on test sets.

use super::{EvaluationConfig, Evaluator, MetricType};
use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{s, Array, Axis, Ix2, IxDyn, ScalarOperand, Slice};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
/// Configuration for test set evaluation
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Metrics to compute during evaluation
    pub metrics: Vec<MetricType>,
    /// Number of batches to evaluate (None for all batches)
    pub steps: Option<usize>,
    /// Verbosity level
    pub verbose: usize,
    /// Generate prediction outputs
    pub generate_predictions: bool,
    /// Whether to save model outputs during evaluation
    pub save_outputs: bool,
}
impl Default for TestConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 0,
            metrics: vec![MetricType::Loss, MetricType::Accuracy],
            steps: None,
            verbose: 1,
            generate_predictions: false,
            save_outputs: false,
        }
    }
/// Prediction output from a test set evaluation
pub struct PredictionOutput<
    F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync,
> {
    /// Model predictions
    pub predictions: Array<F, IxDyn>,
    /// Ground truth targets
    pub targets: Array<F, IxDyn>,
    /// Prediction classes (for classification tasks)
    pub classes: Option<Array<F, IxDyn>>,
    /// Class probabilities (for classification tasks)
    pub probabilities: Option<Array<F, IxDyn>>,
/// Test set evaluator for model testing
#[derive(Debug)]
pub struct TestEvaluator<
    /// Configuration for test set evaluation
    pub config: TestConfig,
    /// Evaluator for test metrics
    evaluator: Evaluator<F>,
    /// Prediction outputs
    prediction_outputs: Option<PredictionOutput<F>>,
impl<F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync>
    TestEvaluator<F>
{
    /// Create a new test set evaluator
    pub fn new(config: TestConfig) -> Result<Self> {
        // Create evaluator
        let eval_config = EvaluationConfig {
            batch_size: config.batch_size,
            shuffle: false,
            num_workers: config.num_workers,
            metrics: config.metrics.clone(),
            steps: config.steps,
            verbose: config.verbose,
        };
        let evaluator = Evaluator::new(eval_config)?;
        Ok(Self {
            config,
            evaluator,
            prediction_outputs: None,
        })
    /// Evaluate a model on a test set
    pub fn evaluate<L: Layer<F>>(
        &mut self,
        model: &mut L,
        dataset: &dyn Dataset<F>,
        loss_fn: Option<&dyn crate::losses::Loss<F>>,
    ) -> Result<HashMap<String, F>> {
        // Set model to evaluation mode
        let mut restore_training = false;
        if model.is_training() {
            restore_training = true;
            model.set_training(false);
        // Evaluate metrics
        let metrics = self.evaluator.evaluate(model, dataset, loss_fn)?;
        // Generate predictions if needed
        if self.config.generate_predictions {
            self.generate_predictions(model, dataset)?;
        // Restore model training mode
        if restore_training {
            model.set_training(true);
        Ok(metrics)
    /// Generate predictions for a test set
    pub fn generate_predictions<L: Layer<F>>(
    ) -> Result<()> {
        // We need to convert from &dyn Dataset<F> to a concrete type to use with DataLoader
        // For now, let's handle this without a proper DataLoader by implementing the functionality directly
        // Calculate number of batches
        let num_samples = dataset.len();
        let num_batches = num_samples / self.config.batch_size
            + if num_samples % self.config.batch_size > 0 {
                1
            } else {
                0
            };
        // Number of steps to evaluate
        let steps = self.config.steps.unwrap_or(num_batches);
        // Lists to store all predictions and targets
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        // Show progress based on verbosity
        if self.config.verbose > 0 {
            println!("Generating predictions for {} samples", dataset.len());
        // Loop through batches
        let mut batch_count = 0;
        // Generate indices
        let indices: Vec<usize> = (0..dataset.len()).collect();
        // Process each batch
        for batch_idx in 0..steps.min(num_batches) {
            // Determine batch range
            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(dataset.len());
            let batch_indices = &indices[start_idx..end_idx];
            // Load first sample to determine shapes
            let (first_x, first_y) = dataset.get(batch_indices[0])?;
            // Create batch arrays
            let batch_xshape = [batch_indices.len()]
                .iter()
                .chain(first_x.shape())
                .cloned()
                .collect::<Vec<_>>();
            let batch_yshape = [batch_indices.len()]
                .chain(first_y.shape())
            let mut batch_x = Array::zeros(IxDyn(&batch_xshape));
            let mut batch_y = Array::zeros(IxDyn(&batch_yshape));
            // Fill batch arrays
            for (i, &idx) in batch_indices.iter().enumerate() {
                let (x, y) = dataset.get(idx)?;
                // Copy data into batch arrays
                let mut batch_x_slice = batch_x.slice_mut(scirs2_core::ndarray::s![i, ..]);
                batch_x_slice.assign(&x);
                let mut batch_y_slice = batch_y.slice_mut(scirs2_core::ndarray::s![i, ..]);
                batch_y_slice.assign(&y);
            }
            // Forward pass
            let outputs = model.forward(&batch_x)?;
            // Store predictions and targets
            all_predictions.push(outputs.clone());
            all_targets.push(batch_y.clone());
            batch_count += 1;
            // Print progress if verbose
            if self.config.verbose == 2 {
                println!("Batch {}/{}", batch_count, steps);
        // Concatenate predictions and targets
        if all_predictions.is_empty() || all_targets.is_empty() {
            return Err(Error::ValidationError(
                "No predictions generated".to_string(),
            ));
        // Get dimensions from first batch to determine output structure
        let first_pred = &all_predictions[0];
        let first_target = &all_targets[0];
        // Calculate total samples across all batches
        let total_samples: usize = all_predictions.iter().map(|pred| pred.shape()[0]).sum();
        // Create combined arrays with proper dimensions
        let mut combinedshape_pred = first_pred.shape().to_vec();
        combinedshape_pred[0] = total_samples;
        let mut combinedshape_target = first_target.shape().to_vec();
        combinedshape_target[0] = total_samples;
        let mut combined_preds = Array::<F>::zeros(IxDyn(&combinedshape_pred));
        let mut combined_targets = Array::<F>::zeros(IxDyn(&combinedshape_target));
        // Concatenate all batch predictions and targets
        let mut sample_offset = 0;
        for (pred_batch, target_batch) in all_predictions.iter().zip(all_targets.iter()) {
            let batch_size = pred_batch.shape()[0];
            // Validate batch dimensions match expected structure
            if pred_batch.shape().len() != first_pred.shape().len()
                || target_batch.shape().len() != first_target.shape().len()
            {
                return Err(Error::ValidationError(
                    "Inconsistent batch dimensions across samples".to_string(),
                ));
            // Verify non-batch dimensions match
            for dim_idx in 1..pred_batch.shape().len() {
                if pred_batch.shape()[dim_idx] != first_pred.shape()[dim_idx] {
                    return Err(Error::ValidationError(format!(
                        "Prediction dimension {} mismatch in batch",
                        dim_idx
                    )));
                }
            for dim_idx in 1..target_batch.shape().len() {
                if target_batch.shape()[dim_idx] != first_target.shape()[dim_idx] {
                        "Target dimension {} mismatch in batch",
            // Copy batch data to combined arrays
            let pred_slice_range = sample_offset..(sample_offset + batch_size);
            let target_slice_range = sample_offset..(sample_offset + batch_size);
            // Handle different dimensionalities properly
            match (pred_batch.ndim(), target_batch.ndim()) {
                (1, 1) => {
                    // 1D case: regression or binary classification
                    combined_preds
                        .slice_mut(s![pred_slice_range, ..])
                        .assign(&pred_batch.view());
                    combined_targets
                        .slice_mut(s![target_slice_range, ..])
                        .assign(&target_batch.view());
                (2, 1) => {
                    // 2D predictions, 1D targets: multi-class classification
                        .slice_mut(s![target_slice_range])
                (2, 2) => {
                    // 2D both: multi-class with one-hot targets or sequence tasks
                (pred_ndim, target_ndim) => {
                    // Higher dimensional cases: use axis-based concatenation
                    if pred_ndim >= 3 {
                        // For 3D+ tensors (e.g., sequence data, image data)
                        let mut pred_slice = combined_preds.slice_axis_mut(
                            Axis(0),
                            Slice::from(sample_offset..(sample_offset + batch_size)),
                        );
                        pred_slice.assign(&pred_batch.view());
                    }
                    if target_ndim >= 3 {
                        // For 3D+ tensors
                        let mut target_slice = combined_targets.slice_axis_mut(
                        target_slice.assign(&target_batch.view());
            sample_offset += batch_size;
        // Verify concatenation completed successfully
        if sample_offset != total_samples {
            return Err(Error::ValidationError(format!(
                "Concatenation error: expected {} samples, got {}",
                total_samples, sample_offset
            )));
        // Extract prediction classes and probabilities for classification tasks
        let classes = if first_pred.ndim() > 1 && first_pred.shape()[1] > 1 {
            // Multi-class classification
            let mut class_indices = Array::<F>::zeros(IxDyn(&[combined_preds.shape()[0], 1]));
            for i in 0..combined_preds.shape()[0] {
                let mut max_idx = 0;
                let mut max_val = combined_preds[[i, 0]];
                for j in 1..combined_preds.shape()[1] {
                    if combined_preds[[i, j]] > max_val {
                        max_idx = j;
                        max_val = combined_preds[[i, j]];
                class_indices[[i, 0]] = F::from(max_idx).unwrap();
            Some(class_indices)
        } else {
            // Binary classification or regression
            None
        // Compute probabilities for classification
        let probabilities = if first_pred.ndim() > 1 && first_pred.shape()[1] > 1 {
            // Apply softmax to get probabilities
            let mut probs = combined_preds.clone();
            for i in 0..probs.shape()[0] {
                // Find max for numerical stability
                let mut max_val = probs[[i, 0]];
                for j in 1..probs.shape()[1] {
                    if probs[[i, j]] > max_val {
                        max_val = probs[[i, j]];
                // Compute exp and sum
                let mut sum = F::zero();
                for j in 0..probs.shape()[1] {
                    probs[[i, j]] = (probs[[i, j]] - max_val).exp();
                    sum = sum + probs[[i, j]];
                // Normalize
                if sum > F::zero() {
                    for j in 0..probs.shape()[1] {
                        probs[[i, j]] = probs[[i, j]] / sum;
            Some(probs)
        } else if first_pred.ndim() == 2 && first_pred.shape()[1] == 1 {
            // Binary classification with sigmoid outputs
            let mut probs = Array::<F>::zeros(IxDyn(&[combined_preds.shape()[0], 2]));
                let p = combined_preds[[i, 0]];
                probs[[i, 0]] = F::one() - p;
                probs[[i, 1]] = p;
            // Regression
        // Store prediction outputs
        self.prediction_outputs = Some(PredictionOutput {
            predictions: combined_preds,
            targets: combined_targets,
            classes,
            probabilities,
        });
        Ok(())
    /// Get prediction outputs
    pub fn get_prediction_outputs(&self) -> Option<&PredictionOutput<F>> {
        self.prediction_outputs.as_ref()
    /// Generate a classification report for a classification task
    pub fn classification_report(&self) -> Result<String> {
        if let Some(ref outputs) = self.prediction_outputs {
            if let Some(ref classes) = outputs.classes {
                let mut report = String::new();
                report.push_str("Classification Report\n");
                report.push_str("---------------------\n");
                // Extract class indices
                let pred_classes = classes.clone();
                let target_classes = if outputs.targets.ndim() > 1 && outputs.targets.shape()[1] > 1
                {
                    // One-hot encoded targets
                    let mut class_indices =
                        Array::<F>::zeros(IxDyn(&[outputs.targets.shape()[0], 1]));
                    for i in 0..outputs.targets.shape()[0] {
                        let mut max_idx = 0;
                        let mut max_val = outputs.targets[[i, 0]];
                        for j in 1..outputs.targets.shape()[1] {
                            if outputs.targets[[i, j]] > max_val {
                                max_idx = j;
                                max_val = outputs.targets[[i, j]];
                            }
                        }
                        class_indices[[i, 0]] = F::from(max_idx).unwrap();
                    class_indices
                } else if outputs.targets.ndim() == 2 && outputs.targets.shape()[1] == 1 {
                    // Binary targets
                        class_indices[[i, 0]] = if outputs.targets[[i, 0]] >= F::from(0.5).unwrap()
                        {
                            F::from(1).unwrap()
                        } else {
                            F::zero()
                        };
                } else {
                    // Convert to dynamic dimension array
                    let mut dyn_targets = Array::<F>::zeros(outputs.targets.raw_dim());
                    dyn_targets.assign(&outputs.targets);
                    dyn_targets
                };
                // Compute metrics per class
                let n_classes = if let Some(ref probs) = outputs.probabilities {
                    probs.shape()[1]
                    // Estimate number of classes from unique values
                    let mut unique_classes = std::collections::HashSet::new();
                    for i in 0..pred_classes.shape()[0] {
                        unique_classes.insert(pred_classes[[i, 0]].to_usize().unwrap_or(0));
                        unique_classes.insert(target_classes[[i, 0]].to_usize().unwrap_or(0));
                    unique_classes.len()
                for class_idx in 0..n_classes {
                    let class_f = F::from(class_idx).unwrap();
                    // Count TP, FP, FN, TN
                    let mut tp = 0;
                    let mut fp = 0;
                    let mut fn_ = 0;
                    let mut _tn = 0;
                        let pred = pred_classes[[i, 0]];
                        let target = target_classes[[i, 0]];
                        if pred == class_f && target == class_f {
                            tp += 1;
                        } else if pred == class_f && target != class_f {
                            fp += 1;
                        } else if pred != class_f && target == class_f {
                            fn_ += 1;
                            _tn += 1;
                    // Compute metrics
                    let precision = if tp + fp > 0 {
                        tp as f64 / (tp + fp) as f64
                    } else {
                        0.0
                    };
                    let recall = if tp + fn_ > 0 {
                        tp as f64 / (tp + fn_) as f64
                    let f1 = if precision + recall > 0.0 {
                        2.0 * precision * recall / (precision + recall)
                    let support = tp + fn_;
                    report.push_str(&format!(
                        "Class {}: precision={:.4}, recall={:.4}, f1-score={:.4}, support={}\n",
                        class_idx, precision, recall, f1, support
                    ));
                // Compute overall accuracy
                let mut correct = 0;
                for i in 0..pred_classes.shape()[0] {
                    if pred_classes[[i, 0]] == target_classes[[i, 0]] {
                        correct += 1;
                let accuracy = correct as f64 / pred_classes.shape()[0] as f64;
                report.push_str(&format!("\nAccuracy: {:.4}\n", accuracy));
                Ok(report)
                Err(Error::ValidationError(
                    "No class predictions available".to_string(),
                ))
            Err(Error::ValidationError(
                "No predictions available, call evaluate() or generate_predictions() first"
                    .to_string(),
            ))
    /// Generate a confusion matrix for a classification task
    pub fn confusion_matrix(&self) -> Result<Array<usize, Ix2>> {
                // Determine number of classes
                // Initialize confusion matrix
                let mut cm = Array::<usize>::zeros((n_classes, n_classes));
                // Fill confusion matrix
                    let pred = pred_classes[[i, 0]].to_usize().unwrap_or(0);
                    let target = target_classes[[i, 0]].to_usize().unwrap_or(0);
                    if pred < n_classes && target < n_classes {
                        cm[[target, pred]] += 1;
                Ok(cm)
