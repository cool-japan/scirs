//! Foundation model evaluation metrics
//!
//! This module provides evaluation metrics for foundation models
//! including zero-shot and few-shot learning performance.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::iter::Sum;

use super::results::FewShotResult;

/// Foundation model metrics
pub struct FoundationModelMetrics<F: Float> {
    /// Number of shots for few-shot evaluation
    pub n_shots: Vec<usize>,
    /// Number of tasks for evaluation
    pub n_tasks: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for FoundationModelMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand>
    FoundationModelMetrics<F>
{
    /// Create new foundation model metrics
    pub fn new() -> Self {
        Self {
            n_shots: vec![1, 5, 10],
            n_tasks: 10,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set few-shot evaluation parameters
    pub fn with_few_shot(mut self, shots: Vec<usize>) -> Self {
        self.n_shots = shots;
        self
    }

    /// Set number of evaluation tasks
    pub fn with_tasks(mut self, n: usize) -> Self {
        self.n_tasks = n;
        self
    }

    /// Compute zero-shot performance
    pub fn zero_shot_accuracy(
        &self,
        predictions: &Array1<F>,
        targets: &Array1<usize>,
    ) -> Result<F> {
        if predictions.len() != targets.len() {
            return Err(MetricsError::InvalidInput(
                "Prediction and target length mismatch".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Ok(F::zero());
        }

        let mut correct = 0;
        for (i, &target) in targets.iter().enumerate() {
            // Convert prediction to class (assuming binary classification for simplicity)
            let predicted_class = if predictions[i] > F::from(0.5).unwrap() {
                1
            } else {
                0
            };
            if predicted_class == target {
                correct += 1;
            }
        }

        Ok(F::from(correct).unwrap() / F::from(predictions.len()).unwrap())
    }

    /// Compute few-shot learning performance
    pub fn few_shot_performance(
        &self,
        support_representations: &Array2<F>,
        support_labels: &Array1<usize>,
        query_representations: &Array2<F>,
        query_labels: &Array1<usize>,
        n_shot: usize,
    ) -> Result<FewShotResult<F>> {
        if support_representations.nrows() != support_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Support data length mismatch".to_string(),
            ));
        }

        if query_representations.nrows() != query_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Query data length mismatch".to_string(),
            ));
        }

        let n_classes = support_labels.iter().max().unwrap_or(&0) + 1;

        // Select n_shot examples per class
        let mut selected_support = Vec::new();
        let mut selected_labels = Vec::new();

        for class in 0..n_classes {
            let class_indices: Vec<usize> = support_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .take(n_shot)
                .collect();

            for &idx in &class_indices {
                selected_support.push(support_representations.row(idx).to_owned());
                selected_labels.push(class);
            }
        }

        if selected_support.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No support examples selected".to_string(),
            ));
        }

        // Perform nearest neighbor classification
        let mut correct = 0;
        let mut per_class_correct = vec![0; n_classes];
        let mut per_class_total = vec![0; n_classes];

        for (i, &true_label) in query_labels.iter().enumerate() {
            let query_sample = query_representations.row(i);

            let mut best_distance = F::infinity();
            let mut predicted_class = 0;

            for (j, support_sample) in selected_support.iter().enumerate() {
                let distance = self.euclidean_distance(&query_sample.to_owned(), support_sample)?;

                if distance < best_distance {
                    best_distance = distance;
                    predicted_class = selected_labels[j];
                }
            }

            per_class_total[true_label] += 1;
            if predicted_class == true_label {
                correct += 1;
                per_class_correct[true_label] += 1;
            }
        }

        let overall_accuracy = F::from(correct).unwrap() / F::from(query_labels.len()).unwrap();

        let mut per_class_accuracies = Vec::with_capacity(n_classes);
        for class in 0..n_classes {
            if per_class_total[class] > 0 {
                let acc = F::from(per_class_correct[class]).unwrap()
                    / F::from(per_class_total[class]).unwrap();
                per_class_accuracies.push(acc);
            } else {
                per_class_accuracies.push(F::zero());
            }
        }

        let balanced_accuracy =
            per_class_accuracies.iter().copied().sum::<F>() / F::from(n_classes).unwrap();

        Ok(FewShotResult {
            overall_accuracy,
            balanced_accuracy,
            per_class_accuracies,
            n_shot,
            n_classes,
            n_query_samples: query_labels.len(),
        })
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq.sqrt())
    }
}
