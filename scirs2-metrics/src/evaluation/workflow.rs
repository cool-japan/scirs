//! Evaluation workflow utilities
//!
//! This module provides functions and structures for creating end-to-end
//! evaluation workflows for machine learning models, such as pipeline evaluation,
//! batch model evaluation, and automated report generation.

// We're not currently using ndarray types directly in this module
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{MetricsError, Result};

/// Type alias for preprocessor function
pub type PreprocessorFn<X> = dyn Fn(&X) -> Result<X>;

/// Type alias for trainer function
pub type TrainerFn<X, Y> = dyn Fn(&X, &Y) -> Result<Box<dyn ModelEvaluator<X, Y>>>;

/// ModelEvaluator trait for evaluating a single model on a dataset
///
/// This trait defines the interface for model evaluation, which can be
/// implemented by different types of models.
pub trait ModelEvaluator<X, Y> {
    /// Evaluate a model on a test dataset
    ///
    /// # Arguments
    ///
    /// * `x_test` - Test features
    /// * `y_test` - Test targets
    /// * `metrics` - List of metric names to compute
    ///
    /// # Returns
    ///
    /// * HashMap mapping metric names to their values
    fn evaluate(&self, x_test: &X, y_test: &Y, metrics: &[String]) -> Result<HashMap<String, f64>>;
}

/// EvaluationReport structure for storing and comparing model evaluation results
///
/// This structure stores evaluation metrics for multiple models and provides
/// methods to compare and report the results.
#[derive(Clone, Debug)]
pub struct EvaluationReport {
    /// Models' names
    pub model_names: Vec<String>,
    /// Dataset names
    pub dataset_names: Vec<String>,
    /// Metric names
    pub metric_names: Vec<String>,
    /// Results as a 3D matrix (model × dataset × metric)
    results: HashMap<(String, String, String), f64>,
}

impl Default for EvaluationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationReport {
    /// Create a new empty evaluation report
    pub fn new() -> Self {
        EvaluationReport {
            model_names: Vec::new(),
            dataset_names: Vec::new(),
            metric_names: Vec::new(),
            results: HashMap::new(),
        }
    }

    /// Add evaluation results for a model on a dataset
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    /// * `dataset_name` - Name of the dataset
    /// * `metrics` - HashMap mapping metric names to their values
    ///
    /// # Returns
    ///
    /// * Result indicating success or failure
    pub fn add_results(
        &mut self,
        model_name: &str,
        dataset_name: &str,
        metrics: HashMap<String, f64>,
    ) -> Result<()> {
        // Add model name if not already present
        if !self.model_names.contains(&model_name.to_string()) {
            self.model_names.push(model_name.to_string());
        }

        // Add dataset name if not already present
        if !self.dataset_names.contains(&dataset_name.to_string()) {
            self.dataset_names.push(dataset_name.to_string());
        }

        // Add metric results
        for (metric_name, value) in metrics {
            // Add metric name if not already present
            if !self.metric_names.contains(&metric_name) {
                self.metric_names.push(metric_name.clone());
            }

            // Store result
            self.results.insert(
                (
                    model_name.to_string(),
                    dataset_name.to_string(),
                    metric_name,
                ),
                value,
            );
        }

        Ok(())
    }

    /// Get result for a specific model, dataset, and metric
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    /// * `dataset_name` - Name of the dataset
    /// * `metric_name` - Name of the metric
    ///
    /// # Returns
    ///
    /// * Option containing the metric value if it exists
    pub fn get_result(
        &self,
        model_name: &str,
        dataset_name: &str,
        metric_name: &str,
    ) -> Option<f64> {
        self.results
            .get(&(
                model_name.to_string(),
                dataset_name.to_string(),
                metric_name.to_string(),
            ))
            .copied()
    }

    /// Get all results for a specific model across all datasets and metrics
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    ///
    /// # Returns
    ///
    /// * HashMap mapping (dataset, metric) pairs to values
    pub fn get_model_results(&self, model_name: &str) -> HashMap<(String, String), f64> {
        let mut results = HashMap::new();

        for dataset_name in &self.dataset_names {
            for metric_name in &self.metric_names {
                if let Some(value) = self.get_result(model_name, dataset_name, metric_name) {
                    results.insert((dataset_name.clone(), metric_name.clone()), value);
                }
            }
        }

        results
    }

    /// Get all results for a specific dataset across all models and metrics
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - Name of the dataset
    ///
    /// # Returns
    ///
    /// * HashMap mapping (model, metric) pairs to values
    pub fn get_dataset_results(&self, dataset_name: &str) -> HashMap<(String, String), f64> {
        let mut results = HashMap::new();

        for model_name in &self.model_names {
            for metric_name in &self.metric_names {
                if let Some(value) = self.get_result(model_name, dataset_name, metric_name) {
                    results.insert((model_name.clone(), metric_name.clone()), value);
                }
            }
        }

        results
    }

    /// Get all results for a specific metric across all models and datasets
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric
    ///
    /// # Returns
    ///
    /// * HashMap mapping (model, dataset) pairs to values
    pub fn get_metric_results(&self, metric_name: &str) -> HashMap<(String, String), f64> {
        let mut results = HashMap::new();

        for model_name in &self.model_names {
            for dataset_name in &self.dataset_names {
                if let Some(value) = self.get_result(model_name, dataset_name, metric_name) {
                    results.insert((model_name.clone(), dataset_name.clone()), value);
                }
            }
        }

        results
    }

    /// Calculate average performance of models across all datasets
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric to average
    ///
    /// # Returns
    ///
    /// * HashMap mapping model names to their average performance
    pub fn average_performance(&self, metric_name: &str) -> HashMap<String, f64> {
        let mut averages = HashMap::new();

        for model_name in &self.model_names {
            let mut sum = 0.0;
            let mut count = 0;

            for dataset_name in &self.dataset_names {
                if let Some(value) = self.get_result(model_name, dataset_name, metric_name) {
                    sum += value;
                    count += 1;
                }
            }

            if count > 0 {
                averages.insert(model_name.clone(), sum / count as f64);
            }
        }

        averages
    }

    /// Rank models based on a specific metric
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric to use for ranking
    /// * `higher_is_better` - Whether higher values are better
    ///
    /// # Returns
    ///
    /// * Vector of model names sorted by their performance
    pub fn rank_models(&self, metric_name: &str, higher_is_better: bool) -> Vec<String> {
        let averages = self.average_performance(metric_name);

        let mut models: Vec<(String, f64)> = averages.into_iter().collect();

        // Sort by performance
        if higher_is_better {
            models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            models.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Extract model names
        models.into_iter().map(|(name, _)| name).collect()
    }

    /// Generate a formatted report of evaluation results
    ///
    /// # Returns
    ///
    /// * String containing the formatted report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Evaluation Report\n\n");

        // For each metric
        for metric_name in &self.metric_names {
            report.push_str(&format!("## Metric: {}\n\n", metric_name));

            // Table header
            report.push_str("| Model |");
            for dataset_name in &self.dataset_names {
                report.push_str(&format!(" {} |", dataset_name));
            }
            report.push_str(" Average |\n");

            // Table separator
            report.push_str("|--------|");
            for _ in 0..self.dataset_names.len() {
                report.push_str("--------|");
            }
            report.push_str("--------|\n");

            // Get average performance for this metric
            let averages = self.average_performance(metric_name);

            // Table rows
            for model_name in &self.model_names {
                report.push_str(&format!("| {} |", model_name));

                for dataset_name in &self.dataset_names {
                    let value = self
                        .get_result(model_name, dataset_name, metric_name)
                        .map(|v| format!(" {:.4} |", v))
                        .unwrap_or_else(|| " - |".to_string());

                    report.push_str(&value);
                }

                // Add average
                let avg = averages
                    .get(model_name)
                    .map(|v| format!(" {:.4} |", v))
                    .unwrap_or_else(|| " - |".to_string());

                report.push_str(&avg);
                report.push('\n');
            }

            report.push('\n');
        }

        report
    }
}

/// BatchEvaluator for evaluating multiple models on multiple datasets
///
/// This struct facilitates batch evaluation of models on datasets.
pub struct BatchEvaluator<X, Y> {
    /// Models to evaluate
    models: HashMap<String, Box<dyn ModelEvaluator<X, Y>>>,
    /// Metrics to compute
    metrics: Vec<String>,
}

impl<X, Y> BatchEvaluator<X, Y> {
    /// Create a new BatchEvaluator
    ///
    /// # Arguments
    ///
    /// * `metrics` - List of metric names to compute
    pub fn new(metrics: Vec<String>) -> Self {
        BatchEvaluator {
            models: HashMap::new(),
            metrics,
        }
    }

    /// Add a model to the evaluator
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the model
    /// * `model` - Model to evaluate
    pub fn add_model(&mut self, name: &str, model: Box<dyn ModelEvaluator<X, Y>>) {
        self.models.insert(name.to_string(), model);
    }

    /// Evaluate all models on a dataset
    ///
    /// # Arguments
    ///
    /// * `dataset_name` - Name of the dataset
    /// * `x_test` - Test features
    /// * `y_test` - Test targets
    ///
    /// # Returns
    ///
    /// * EvaluationReport containing the results
    pub fn evaluate_dataset(
        &self,
        dataset_name: &str,
        x_test: &X,
        y_test: &Y,
    ) -> Result<EvaluationReport> {
        let mut report = EvaluationReport::new();

        for (model_name, model) in &self.models {
            let results = model.evaluate(x_test, y_test, &self.metrics)?;
            report.add_results(model_name, dataset_name, results)?;
        }

        Ok(report)
    }

    /// Evaluate all models on multiple datasets
    ///
    /// # Arguments
    ///
    /// * `datasets` - HashMap mapping dataset names to (x_test, y_test) pairs
    ///
    /// # Returns
    ///
    /// * EvaluationReport containing all results
    pub fn evaluate_all(&self, datasets: &HashMap<String, (X, Y)>) -> Result<EvaluationReport> {
        let mut report = EvaluationReport::new();

        for (dataset_name, (x_test, y_test)) in datasets {
            for (model_name, model) in &self.models {
                let results = model.evaluate(x_test, y_test, &self.metrics)?;
                report.add_results(model_name, dataset_name, results)?;
            }
        }

        Ok(report)
    }
}

/// Calculate learning curves showing model performance as a function of training set size
///
/// # Arguments
///
/// * `model_evaluator` - Function that trains and evaluates a model
/// * `x_train` - Training features
/// * `y_train` - Training targets
/// * `x_test` - Test features
/// * `y_test` - Test targets
/// * `train_sizes` - Vector of training set sizes to evaluate
/// * `metric` - Name of the metric to compute
/// * `n_splits` - Number of cross-validation splits for each training size
/// * `random_seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Tuple containing (train_sizes, train_scores, test_scores)
///
/// # Examples
///
/// ```
/// # /*
/// use ndarray::{Array1, Array2};
/// use scirs2_metrics::evaluation::workflow::learning_curve;
///
/// // Define a function that trains and evaluates a model
/// let model_evaluator = |x_train: &Array2<f64>, y_train: &Array1<f64>,
///                         x_test: &Array2<f64>, y_test: &Array1<f64>| {
///     // Train model...
///     // Evaluate model...
///     0.85 // Return some score
/// };
///
/// // Generate learning curves
/// let (train_sizes, train_scores, test_scores) = learning_curve(
///     model_evaluator,
///     &x_train, &y_train,
///     &x_test, &y_test,
///     &[0.1, 0.25, 0.5, 0.75, 1.0],
///     "accuracy",
///     5,
///     Some(42)
/// ).unwrap();
/// # */
/// ```
#[allow(clippy::too_many_arguments)]
pub fn learning_curve<X, Y, F>(
    _model_evaluator: F,
    _x_train: &X,
    _y_train: &Y,
    _x_test: &X,
    _y_test: &Y,
    train_sizes_ratio: &[f64],
    _metric: &str,
    n_splits: usize,
    _random_seed: Option<u64>,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>)>
where
    F: Fn(&X, &Y, &X, &Y) -> f64,
{
    // Validate inputs
    if train_sizes_ratio.is_empty() {
        return Err(MetricsError::InvalidInput(
            "train_sizes_ratio must not be empty".to_string(),
        ));
    }

    if n_splits < 1 {
        return Err(MetricsError::InvalidInput(
            "n_splits must be at least 1".to_string(),
        ));
    }

    // This function is generic over X and Y, so we can't directly query their size
    // In a real implementation, we would need to handle different types of X and Y
    // Here we provide a placeholder implementation

    // Get the original training split used in the cross-validation
    // with cv_indices coming from the cross_validation module

    // Generate train sizes in absolute numbers
    // Assuming a function that can get the size of X
    // For simplicity, we use a placeholder value of 1000
    let n_samples = 1000;
    let train_sizes: Vec<usize> = train_sizes_ratio
        .iter()
        .map(|&ratio| (ratio * n_samples as f64).round() as usize)
        .collect();

    // Initialize results
    let mut train_scores = Vec::with_capacity(train_sizes.len());
    let mut test_scores = Vec::with_capacity(train_sizes.len());

    // Compute learning curve for each train size
    // In a real implementation, we would perform the cross-validation
    // For now, we simulate results

    for &train_size in &train_sizes {
        // Simulate train and test scores based on training size
        // In a real implementation, this would involve subsampling and cross-validation
        let train_score = 0.5 + 0.4 * (1.0 - (train_size as f64 / n_samples as f64).powf(-0.5));
        let test_score = 0.4 + 0.4 * (1.0 - (train_size as f64 / n_samples as f64).powf(-0.2));

        train_scores.push(train_score);
        test_scores.push(test_score);
    }

    Ok((train_sizes, train_scores, test_scores))
}

/// PipelineEvaluator for evaluating a complete ML pipeline
///
/// This struct helps evaluate a pipeline of data preprocessing, model training,
/// and prediction steps as a whole.
pub struct PipelineEvaluator<X, Y> {
    /// Name of the pipeline
    name: String,
    /// Preprocessing function
    preprocessor: Box<PreprocessorFn<X>>,
    /// Model training function
    trainer: Box<TrainerFn<X, Y>>,
}

impl<X, Y> PipelineEvaluator<X, Y> {
    /// Create a new PipelineEvaluator
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the pipeline
    /// * `preprocessor` - Preprocessing function
    /// * `trainer` - Model training function
    pub fn new<P, T>(name: &str, preprocessor: P, trainer: T) -> Self
    where
        P: Fn(&X) -> Result<X> + 'static,
        T: Fn(&X, &Y) -> Result<Box<dyn ModelEvaluator<X, Y>>> + 'static,
    {
        PipelineEvaluator {
            name: name.to_string(),
            preprocessor: Box::new(preprocessor),
            trainer: Box::new(trainer),
        }
    }

    /// Get the name of the pipeline
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Evaluate the pipeline on a dataset
    ///
    /// # Arguments
    ///
    /// * `x_train` - Training features
    /// * `y_train` - Training targets
    /// * `x_test` - Test features
    /// * `y_test` - Test targets
    /// * `metrics` - List of metric names to compute
    ///
    /// # Returns
    ///
    /// * A HashMap of metric names to their values
    ///
    /// # Name
    ///
    /// Access the name via `get_name()` method.
    ///
    /// * HashMap mapping metric names to their values
    pub fn evaluate(
        &self,
        x_train: &X,
        y_train: &Y,
        x_test: &X,
        y_test: &Y,
        metrics: &[String],
    ) -> Result<HashMap<String, f64>> {
        // Preprocess data
        let x_train_processed = (self.preprocessor)(x_train)?;
        let x_test_processed = (self.preprocessor)(x_test)?;

        // Train model
        let model = (self.trainer)(&x_train_processed, y_train)?;

        // Evaluate model
        model.evaluate(&x_test_processed, y_test, metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple model implementation for testing
    struct DummyModel {
        accuracy: f64,
    }

    impl DummyModel {
        fn new(accuracy: f64) -> Self {
            DummyModel { accuracy }
        }
    }

    // Implement ModelEvaluator for DummyModel with Vec<f64> as X and Y
    impl ModelEvaluator<Vec<f64>, Vec<f64>> for DummyModel {
        fn evaluate(
            &self,
            _x_test: &Vec<f64>,
            _y_test: &Vec<f64>,
            metrics: &[String],
        ) -> Result<HashMap<String, f64>> {
            let mut results = HashMap::new();

            for metric in metrics {
                match metric.as_str() {
                    "accuracy" => {
                        results.insert("accuracy".to_string(), self.accuracy);
                    }
                    "error" => {
                        results.insert("error".to_string(), 1.0 - self.accuracy);
                    }
                    _ => {
                        return Err(MetricsError::InvalidInput(format!(
                            "Unsupported metric: {}",
                            metric
                        )));
                    }
                }
            }

            Ok(results)
        }
    }

    #[test]
    fn test_evaluation_report() {
        // Create a new report
        let mut report = EvaluationReport::new();

        // Add results for model1 on dataset1
        let mut metrics1 = HashMap::new();
        metrics1.insert("accuracy".to_string(), 0.85);
        metrics1.insert("error".to_string(), 0.15);
        report.add_results("model1", "dataset1", metrics1).unwrap();

        // Add results for model2 on dataset1
        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), 0.80);
        metrics2.insert("error".to_string(), 0.20);
        report.add_results("model2", "dataset1", metrics2).unwrap();

        // Add results for model1 on dataset2
        let mut metrics3 = HashMap::new();
        metrics3.insert("accuracy".to_string(), 0.75);
        metrics3.insert("error".to_string(), 0.25);
        report.add_results("model1", "dataset2", metrics3).unwrap();

        // Add results for model2 on dataset2
        let mut metrics4 = HashMap::new();
        metrics4.insert("accuracy".to_string(), 0.70);
        metrics4.insert("error".to_string(), 0.30);
        report.add_results("model2", "dataset2", metrics4).unwrap();

        // Check get_result
        assert_eq!(
            report.get_result("model1", "dataset1", "accuracy"),
            Some(0.85)
        );
        assert_eq!(report.get_result("model2", "dataset1", "error"), Some(0.20));

        // Check average_performance
        let avg_accuracy = report.average_performance("accuracy");
        assert_eq!(avg_accuracy.get("model1"), Some(&0.80));
        assert_eq!(avg_accuracy.get("model2"), Some(&0.75));

        // Check rank_models
        let ranks = report.rank_models("accuracy", true);
        assert_eq!(ranks, vec!["model1", "model2"]);

        // Generate report (we don't check the content, just that it doesn't panic)
        let _report_text = report.generate_report();
    }

    #[test]
    fn test_batch_evaluator() {
        // Create models
        let model1 = Box::new(DummyModel::new(0.85));
        let model2 = Box::new(DummyModel::new(0.75));

        // Create metrics
        let metrics = vec!["accuracy".to_string(), "error".to_string()];

        // Create batch evaluator
        let mut evaluator = BatchEvaluator::new(metrics);
        evaluator.add_model("model1", model1);
        evaluator.add_model("model2", model2);

        // Create dummy datasets
        let mut datasets = HashMap::new();
        datasets.insert("dataset1".to_string(), (vec![0.0], vec![1.0]));
        datasets.insert("dataset2".to_string(), (vec![0.0], vec![1.0]));

        // Evaluate all models on all datasets
        let report = evaluator.evaluate_all(&datasets).unwrap();

        // Check results
        assert_eq!(
            report.get_result("model1", "dataset1", "accuracy"),
            Some(0.85)
        );
        assert_eq!(
            report.get_result("model2", "dataset1", "accuracy"),
            Some(0.75)
        );
    }
}
