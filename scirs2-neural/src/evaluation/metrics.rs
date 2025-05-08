//! Evaluation metrics
//!
//! This module provides various metrics for evaluating model performance
//! during training and testing.

use super::Metric;
use ndarray::{Array, Axis, Ix1, Ix2, IxDyn, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Loss metric for tracking model loss
#[derive(Debug, Clone)]
pub struct LossMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> {
    /// Total loss
    total_loss: F,
    /// Number of batches
    num_batches: usize,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> LossMetric<F> {
    /// Create a new loss metric
    pub fn new() -> Self {
        Self {
            total_loss: F::zero(),
            num_batches: 0,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for LossMetric<F>
{
    fn update(
        &mut self,
        _predictions: &Array<F, IxDyn>,
        _targets: &Array<F, IxDyn>,
        loss: Option<F>,
    ) {
        if let Some(loss) = loss {
            self.total_loss = self.total_loss + loss;
            self.num_batches += 1;
        }
    }

    fn reset(&mut self) {
        self.total_loss = F::zero();
        self.num_batches = 0;
    }

    fn result(&self) -> F {
        if self.num_batches > 0 {
            self.total_loss / F::from(self.num_batches).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "loss"
    }
}

/// Accuracy metric for classification tasks
#[derive(Debug, Clone)]
pub struct AccuracyMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync>
{
    /// Number of correct predictions
    correct: usize,
    /// Total number of samples
    total: usize,
    /// Phantom data for float type
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> AccuracyMetric<F> {
    /// Create a new accuracy metric
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
            _phantom: PhantomData,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for AccuracyMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // For multi-class classification, find the class with highest probability
        let preds = predictions.clone();
        let targets = targets.clone();

        // Flatten to 2D (samples x classes) if needed
        let preds_2d = if preds.ndim() > 2 {
            let batch_size = preds.shape()[0];
            let total_classes = preds.len() / batch_size;
            preds
                .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
        } else if preds.ndim() == 1 {
            // Binary classification with single output
            preds
                .clone()
                .into_shape_with_order(IxDyn(&[preds.len(), 1]))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
        } else {
            preds.into_dimensionality::<Ix2>().unwrap()
        };

        let targets_2d = if targets.ndim() > 2 {
            let batch_size = targets.shape()[0];
            let total_classes = targets.len() / batch_size;
            targets
                .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
        } else if targets.ndim() == 1 {
            // Binary classification with single output
            targets
                .clone()
                .into_shape_with_order(IxDyn(&[targets.len(), 1]))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
        } else {
            targets.into_dimensionality::<Ix2>().unwrap()
        };

        // Get predicted classes (argmax along class dimension)
        let pred_classes = preds_2d.map_axis(Axis(1), |row| {
            let mut max_idx = 0;
            let mut max_val = row[0];

            for (i, &val) in row.iter().enumerate().skip(1) {
                if val > max_val {
                    max_idx = i;
                    max_val = val;
                }
            }

            F::from(max_idx).unwrap()
        });

        // Get target classes (argmax for one-hot, direct for class indices)
        let target_classes = if targets_2d.shape()[1] > 1 {
            // One-hot encoded
            targets_2d.map_axis(Axis(1), |row| {
                let mut max_idx = 0;
                let mut max_val = row[0];

                for (i, &val) in row.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_idx = i;
                        max_val = val;
                    }
                }

                F::from(max_idx).unwrap()
            })
        } else {
            // Direct class indices
            targets_2d.index_axis(Axis(1), 0).to_owned()
        };

        // Count correct predictions
        for (pred, target) in pred_classes.iter().zip(target_classes.iter()) {
            if (*pred - *target).abs() < F::from(1e-6).unwrap() {
                self.correct += 1;
            }
        }

        self.total += pred_classes.len();
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn result(&self) -> F {
        if self.total > 0 {
            F::from(self.correct as f64 / self.total as f64).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Precision metric for classification tasks
#[derive(Debug, Clone)]
pub struct PrecisionMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync>
{
    /// True positives
    tp: usize,
    /// False positives
    fp: usize,
    /// Current threshold
    threshold: F,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> PrecisionMetric<F> {
    /// Create a new precision metric
    pub fn new() -> Self {
        Self {
            tp: 0,
            fp: 0,
            threshold: F::from(0.5).unwrap(),
        }
    }

    /// Create a new precision metric with a custom threshold
    pub fn with_threshold(threshold: F) -> Self {
        Self {
            tp: 0,
            fp: 0,
            threshold,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for PrecisionMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Handle binary classification first
        if predictions.shape()[predictions.ndim() - 1] == 1 || predictions.ndim() == 1 {
            let preds = predictions
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap_or_else(|_| {
                    predictions
                        .clone()
                        .into_shape_with_order(IxDyn(&[predictions.len(), 1]))
                        .unwrap()
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                });

            let targets = targets
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap_or_else(|_| {
                    targets
                        .clone()
                        .into_shape_with_order(IxDyn(&[targets.len(), 1]))
                        .unwrap()
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                });

            // Apply threshold
            for (pred, target) in preds.iter().zip(targets.iter()) {
                let pred_class = if *pred >= self.threshold { 1 } else { 0 };
                let target_class = if *target >= F::from(0.5).unwrap() {
                    1
                } else {
                    0
                };

                if pred_class == 1 && target_class == 1 {
                    self.tp += 1;
                } else if pred_class == 1 && target_class == 0 {
                    self.fp += 1;
                }
            }
        } else {
            // Multi-class classification
            let preds = predictions.clone();
            let targets = targets.clone();

            // Flatten to 2D (samples x classes) if needed
            let preds_2d = if preds.ndim() > 2 {
                let batch_size = preds.shape()[0];
                let total_classes = preds.len() / batch_size;
                preds
                    .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
            } else {
                preds.into_dimensionality::<Ix2>().unwrap()
            };

            let targets_2d = if targets.ndim() > 2 {
                let batch_size = targets.shape()[0];
                let total_classes = targets.len() / batch_size;
                targets
                    .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
            } else {
                targets.into_dimensionality::<Ix2>().unwrap()
            };

            // Get predicted classes (argmax along class dimension)
            let pred_classes = preds_2d.map_axis(Axis(1), |row| {
                let mut max_idx = 0;
                let mut max_val = row[0];

                for (i, &val) in row.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_idx = i;
                        max_val = val;
                    }
                }

                max_idx
            });

            // Get target classes (argmax for one-hot, direct for class indices)
            let target_classes = if targets_2d.shape()[1] > 1 {
                // One-hot encoded
                targets_2d.map_axis(Axis(1), |row| {
                    let mut max_idx = 0;
                    let mut max_val = row[0];

                    for (i, &val) in row.iter().enumerate().skip(1) {
                        if val > max_val {
                            max_idx = i;
                            max_val = val;
                        }
                    }

                    max_idx
                })
            } else {
                // Direct class indices
                targets_2d
                    .index_axis(Axis(1), 0)
                    .mapv(|x| x.to_usize().unwrap_or(0))
            };

            // Count TP and FP for each class
            let num_classes = preds_2d.shape()[1];
            for c in 0..num_classes {
                let class_preds = pred_classes.mapv(|x| if x == c { 1 } else { 0 });
                let class_targets = target_classes.mapv(|x| if x == c { 1 } else { 0 });

                for (pred, target) in class_preds.iter().zip(class_targets.iter()) {
                    if *pred == 1 && *target == 1 {
                        self.tp += 1;
                    } else if *pred == 1 && *target == 0 {
                        self.fp += 1;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.tp = 0;
        self.fp = 0;
    }

    fn result(&self) -> F {
        if self.tp + self.fp > 0 {
            F::from(self.tp as f64 / (self.tp + self.fp) as f64).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "precision"
    }
}

/// Recall metric for classification tasks
#[derive(Debug, Clone)]
pub struct RecallMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> {
    /// True positives
    tp: usize,
    /// False negatives
    fn_: usize,
    /// Current threshold
    threshold: F,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> RecallMetric<F> {
    /// Create a new recall metric
    pub fn new() -> Self {
        Self {
            tp: 0,
            fn_: 0,
            threshold: F::from(0.5).unwrap(),
        }
    }

    /// Create a new recall metric with a custom threshold
    pub fn with_threshold(threshold: F) -> Self {
        Self {
            tp: 0,
            fn_: 0,
            threshold,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for RecallMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Handle binary classification first
        if predictions.shape()[predictions.ndim() - 1] == 1 || predictions.ndim() == 1 {
            let preds = predictions
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap_or_else(|_| {
                    predictions
                        .clone()
                        .into_shape_with_order(IxDyn(&[predictions.len(), 1]))
                        .unwrap()
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                });

            let targets = targets
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap_or_else(|_| {
                    targets
                        .clone()
                        .into_shape_with_order(IxDyn(&[targets.len(), 1]))
                        .unwrap()
                        .into_dimensionality::<Ix2>()
                        .unwrap()
                });

            // Apply threshold
            for (pred, target) in preds.iter().zip(targets.iter()) {
                let pred_class = if *pred >= self.threshold { 1 } else { 0 };
                let target_class = if *target >= F::from(0.5).unwrap() {
                    1
                } else {
                    0
                };

                if pred_class == 1 && target_class == 1 {
                    self.tp += 1;
                } else if pred_class == 0 && target_class == 1 {
                    self.fn_ += 1;
                }
            }
        } else {
            // Multi-class classification
            let preds = predictions.clone();
            let targets = targets.clone();

            // Flatten to 2D (samples x classes) if needed
            let preds_2d = if preds.ndim() > 2 {
                let batch_size = preds.shape()[0];
                let total_classes = preds.len() / batch_size;
                preds
                    .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
            } else {
                preds.into_dimensionality::<Ix2>().unwrap()
            };

            let targets_2d = if targets.ndim() > 2 {
                let batch_size = targets.shape()[0];
                let total_classes = targets.len() / batch_size;
                targets
                    .into_shape_with_order(IxDyn(&[batch_size, total_classes]))
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
            } else {
                targets.into_dimensionality::<Ix2>().unwrap()
            };

            // Get predicted classes (argmax along class dimension)
            let pred_classes = preds_2d.map_axis(Axis(1), |row| {
                let mut max_idx = 0;
                let mut max_val = row[0];

                for (i, &val) in row.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_idx = i;
                        max_val = val;
                    }
                }

                max_idx
            });

            // Get target classes (argmax for one-hot, direct for class indices)
            let target_classes = if targets_2d.shape()[1] > 1 {
                // One-hot encoded
                targets_2d.map_axis(Axis(1), |row| {
                    let mut max_idx = 0;
                    let mut max_val = row[0];

                    for (i, &val) in row.iter().enumerate().skip(1) {
                        if val > max_val {
                            max_idx = i;
                            max_val = val;
                        }
                    }

                    max_idx
                })
            } else {
                // Direct class indices
                targets_2d
                    .index_axis(Axis(1), 0)
                    .mapv(|x| x.to_usize().unwrap_or(0))
            };

            // Count TP and FN for each class
            let num_classes = preds_2d.shape()[1];
            for c in 0..num_classes {
                let class_preds = pred_classes.mapv(|x| if x == c { 1 } else { 0 });
                let class_targets = target_classes.mapv(|x| if x == c { 1 } else { 0 });

                for (pred, target) in class_preds.iter().zip(class_targets.iter()) {
                    if *pred == 1 && *target == 1 {
                        self.tp += 1;
                    } else if *pred == 0 && *target == 1 {
                        self.fn_ += 1;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.tp = 0;
        self.fn_ = 0;
    }

    fn result(&self) -> F {
        if self.tp + self.fn_ > 0 {
            F::from(self.tp as f64 / (self.tp + self.fn_) as f64).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "recall"
    }
}

/// F1 score metric for classification tasks
#[derive(Debug, Clone)]
pub struct F1ScoreMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> {
    /// Precision metric
    precision: PrecisionMetric<F>,
    /// Recall metric
    recall: RecallMetric<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> F1ScoreMetric<F> {
    /// Create a new F1 score metric
    pub fn new() -> Self {
        Self {
            precision: PrecisionMetric::new(),
            recall: RecallMetric::new(),
        }
    }

    /// Create a new F1 score metric with a custom threshold
    pub fn with_threshold(threshold: F) -> Self {
        Self {
            precision: PrecisionMetric::with_threshold(threshold),
            recall: RecallMetric::with_threshold(threshold),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for F1ScoreMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        self.precision.update(predictions, targets, None);
        self.recall.update(predictions, targets, None);
    }

    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }

    fn result(&self) -> F {
        let precision = self.precision.result();
        let recall = self.recall.result();

        if precision + recall > F::zero() {
            let two = F::from(2.0).unwrap();
            (two * precision * recall) / (precision + recall)
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "f1_score"
    }
}

/// Mean squared error metric for regression tasks
#[derive(Debug, Clone)]
pub struct MeanSquaredErrorMetric<
    F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync,
> {
    /// Sum of squared errors
    sum_squared_error: F,
    /// Number of samples
    count: usize,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync>
    MeanSquaredErrorMetric<F>
{
    /// Create a new mean squared error metric
    pub fn new() -> Self {
        Self {
            sum_squared_error: F::zero(),
            count: 0,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for MeanSquaredErrorMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Flatten arrays
        let preds_flat = predictions
            .clone()
            .into_shape_with_order(IxDyn(&[predictions.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let targets_flat = targets
            .clone()
            .into_shape_with_order(IxDyn(&[targets.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();

        // Compute squared error
        for (pred, target) in preds_flat.iter().zip(targets_flat.iter()) {
            let error = *pred - *target;
            self.sum_squared_error = self.sum_squared_error + error * error;
        }

        self.count += preds_flat.len();
    }

    fn reset(&mut self) {
        self.sum_squared_error = F::zero();
        self.count = 0;
    }

    fn result(&self) -> F {
        if self.count > 0 {
            self.sum_squared_error / F::from(self.count).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "mean_squared_error"
    }
}

/// Mean absolute error metric for regression tasks
#[derive(Debug, Clone)]
pub struct MeanAbsoluteErrorMetric<
    F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync,
> {
    /// Sum of absolute errors
    sum_absolute_error: F,
    /// Number of samples
    count: usize,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync>
    MeanAbsoluteErrorMetric<F>
{
    /// Create a new mean absolute error metric
    pub fn new() -> Self {
        Self {
            sum_absolute_error: F::zero(),
            count: 0,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for MeanAbsoluteErrorMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Flatten arrays
        let preds_flat = predictions
            .clone()
            .into_shape_with_order(IxDyn(&[predictions.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let targets_flat = targets
            .clone()
            .into_shape_with_order(IxDyn(&[targets.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();

        // Compute absolute error
        for (pred, target) in preds_flat.iter().zip(targets_flat.iter()) {
            let error = (*pred - *target).abs();
            self.sum_absolute_error = self.sum_absolute_error + error;
        }

        self.count += preds_flat.len();
    }

    fn reset(&mut self) {
        self.sum_absolute_error = F::zero();
        self.count = 0;
    }

    fn result(&self) -> F {
        if self.count > 0 {
            self.sum_absolute_error / F::from(self.count).unwrap()
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "mean_absolute_error"
    }
}

/// R-squared metric for regression tasks
#[derive(Debug, Clone)]
pub struct RSquaredMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync>
{
    /// Sum of squared errors
    sum_squared_error: F,
    /// Sum of squared differences from mean
    sum_squared_total: F,
    /// Number of samples
    count: usize,
    /// Mean of targets
    mean: F,
    /// First update flag
    first_update: bool,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> RSquaredMetric<F> {
    /// Create a new R-squared metric
    pub fn new() -> Self {
        Self {
            sum_squared_error: F::zero(),
            sum_squared_total: F::zero(),
            count: 0,
            mean: F::zero(),
            first_update: true,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for RSquaredMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Flatten arrays
        let preds_flat = predictions
            .clone()
            .into_shape_with_order(IxDyn(&[predictions.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let targets_flat = targets
            .clone()
            .into_shape_with_order(IxDyn(&[targets.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();

        // Compute mean of targets if first update
        if self.first_update {
            let mut sum = F::zero();
            for &target in targets_flat.iter() {
                sum = sum + target;
            }
            self.mean = sum / F::from(targets_flat.len()).unwrap();
            self.first_update = false;
        }

        // Compute squared error and total
        for (pred, target) in preds_flat.iter().zip(targets_flat.iter()) {
            let error = *pred - *target;
            self.sum_squared_error = self.sum_squared_error + error * error;

            let diff_from_mean = *target - self.mean;
            self.sum_squared_total = self.sum_squared_total + diff_from_mean * diff_from_mean;
        }

        self.count += preds_flat.len();
    }

    fn reset(&mut self) {
        self.sum_squared_error = F::zero();
        self.sum_squared_total = F::zero();
        self.count = 0;
        self.mean = F::zero();
        self.first_update = true;
    }

    fn result(&self) -> F {
        if self.count > 0 && self.sum_squared_total > F::zero() {
            F::one() - (self.sum_squared_error / self.sum_squared_total)
        } else {
            F::zero()
        }
    }

    fn name(&self) -> &str {
        "r_squared"
    }
}

/// Area under ROC curve metric
#[derive(Debug, Clone)]
pub struct AUCMetric<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> {
    /// Prediction scores
    scores: Vec<F>,
    /// True labels
    labels: Vec<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> AUCMetric<F> {
    /// Create a new AUC metric
    pub fn new() -> Self {
        Self {
            scores: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Compute AUC from scores and labels
    fn compute_auc(&self) -> F {
        if self.scores.is_empty() || self.labels.is_empty() {
            return F::zero();
        }

        // Combine scores and labels into pairs
        let mut pairs: Vec<(F, F)> = self
            .scores
            .iter()
            .cloned()
            .zip(self.labels.iter().cloned())
            .collect();

        // Sort by score in descending order
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Count positive and negative samples
        let num_pos = self.labels.iter().filter(|&&l| l > F::zero()).count();
        let num_neg = self.labels.len() - num_pos;

        if num_pos == 0 || num_neg == 0 {
            return F::zero();
        }

        // Compute AUC
        let mut sum_ranks = F::zero();
        let mut pos_count = 0;

        for (i, (_, label)) in pairs.iter().enumerate() {
            if *label > F::zero() {
                sum_ranks = sum_ranks + F::from(i + 1).unwrap();
                pos_count += 1;
            }
        }

        let pos_count = F::from(pos_count).unwrap();
        let num_pos = F::from(num_pos).unwrap();
        let num_neg = F::from(num_neg).unwrap();

        // Calculate AUC
        let auc = (sum_ranks - (pos_count * (pos_count + F::one())) / F::from(2.0).unwrap())
            / (num_pos * num_neg);

        auc
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + Send + Sync> Metric<F>
    for AUCMetric<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // Handle binary classification
        let preds = if predictions.ndim() == 2 && predictions.shape()[1] == 2 {
            // Multi-class with 2 classes - use probability of positive class
            let mut probs = Vec::with_capacity(predictions.shape()[0]);
            for i in 0..predictions.shape()[0] {
                probs.push(predictions[[i, 1]]);
            }
            probs
        } else if predictions.ndim() == 2 && predictions.shape()[1] == 1 {
            // Binary with single output
            predictions.iter().cloned().collect()
        } else if predictions.ndim() == 1 {
            // Binary with single output
            predictions.iter().cloned().collect()
        } else {
            // Not supported for multi-class with more than 2 classes
            return;
        };

        // Extract labels
        let labels = if targets.ndim() == 2 && targets.shape()[1] == 2 {
            // One-hot encoded with 2 classes
            let mut labs = Vec::with_capacity(targets.shape()[0]);
            for i in 0..targets.shape()[0] {
                labs.push(targets[[i, 1]]);
            }
            labs
        } else if targets.ndim() == 2 && targets.shape()[1] == 1 {
            // Binary with single output
            targets.iter().cloned().collect()
        } else if targets.ndim() == 1 {
            // Binary with single output
            targets.iter().cloned().collect()
        } else {
            // Not supported for multi-class with more than 2 classes
            return;
        };

        // Add to scores and labels
        self.scores.extend(preds);
        self.labels.extend(labels);
    }

    fn reset(&mut self) {
        self.scores.clear();
        self.labels.clear();
    }

    fn result(&self) -> F {
        self.compute_auc()
    }

    fn name(&self) -> &str {
        "auc"
    }
}
