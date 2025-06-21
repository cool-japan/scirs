//! Custom metric definition framework
//!
//! This module provides a framework for defining custom metrics that integrate
//! seamlessly with the rest of the scirs2-metrics ecosystem.
//!
//! # Features
//!
//! - **Trait-based design**: Define custom metrics by implementing simple traits
//! - **Type safety**: Leverage Rust's type system for metric validation
//! - **Integration**: Custom metrics work with evaluation pipelines and visualization
//! - **Performance**: Zero-cost abstractions with compile-time optimization
//! - **Composability**: Combine custom metrics with built-in metrics
//!
//! # Examples
//!
//! ## Defining a Custom Classification Metric
//!
//! ```
//! use scirs2_metrics::custom::{ClassificationMetric, MetricResult};
//! use ndarray::Array1;
//!
//! struct CustomAccuracy;
//!
//! impl ClassificationMetric<f64> for CustomAccuracy {
//!     fn name(&self) -> &'static str {
//!         "custom_accuracy"
//!     }
//!
//!     fn compute(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> MetricResult<f64> {
//!         if y_true.len() != y_pred.len() {
//!             return Err("Arrays must have the same length".into());
//!         }
//!
//!         let correct = y_true.iter()
//!             .zip(y_pred.iter())
//!             .filter(|(true_val, pred_val)| true_val == pred_val)
//!             .count();
//!
//!         Ok(correct as f64 / y_true.len() as f64)
//!     }
//!
//!     fn higher_is_better(&self) -> bool {
//!         true
//!     }
//! }
//! ```
//!
//! ## Defining a Custom Regression Metric
//!
//! ```
//! use scirs2_metrics::custom::{RegressionMetric, MetricResult};
//! use ndarray::Array1;
//!
//! struct LogCoshError;
//!
//! impl RegressionMetric<f64> for LogCoshError {
//!     fn name(&self) -> &'static str {
//!         "log_cosh_error"
//!     }
//!
//!     fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricResult<f64> {
//!         if y_true.len() != y_pred.len() {
//!             return Err("Arrays must have the same length".into());
//!         }
//!
//!         let error: f64 = y_true.iter()
//!             .zip(y_pred.iter())
//!             .map(|(true_val, pred_val)| {
//!                 let diff = pred_val - true_val;
//!                 (diff.cosh()).ln()
//!             })
//!             .sum();
//!
//!         Ok(error / y_true.len() as f64)
//!     }
//!
//!     fn higher_is_better(&self) -> bool {
//!         false
//!     }
//! }
//! ```

use crate::error::Result as MetricsResult;
use ndarray::Array1;
use num_traits::Float;
use std::fmt;

/// Result type for custom metric computations
pub type MetricResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Trait for defining custom classification metrics
pub trait ClassificationMetric<F: Float> {
    /// Returns the name of the metric
    fn name(&self) -> &'static str;

    /// Computes the metric value given true and predicted labels
    fn compute(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> MetricResult<F>;

    /// Returns whether higher values indicate better performance
    fn higher_is_better(&self) -> bool;

    /// Optional: Returns a description of the metric
    fn description(&self) -> Option<&'static str> {
        None
    }

    /// Optional: Returns the valid range of the metric (min, max)
    fn value_range(&self) -> Option<(F, F)> {
        None
    }
}

/// Trait for defining custom regression metrics
pub trait RegressionMetric<F: Float> {
    /// Returns the name of the metric
    fn name(&self) -> &'static str;

    /// Computes the metric value given true and predicted values
    fn compute(&self, y_true: &Array1<F>, y_pred: &Array1<F>) -> MetricResult<F>;

    /// Returns whether higher values indicate better performance
    fn higher_is_better(&self) -> bool;

    /// Optional: Returns a description of the metric
    fn description(&self) -> Option<&'static str> {
        None
    }

    /// Optional: Returns the valid range of the metric (min, max)
    fn value_range(&self) -> Option<(F, F)> {
        None
    }
}

/// Trait for defining custom clustering metrics
pub trait ClusteringMetric<F: Float> {
    /// Returns the name of the metric
    fn name(&self) -> &'static str;

    /// Computes the metric value given data points and cluster labels
    fn compute(&self, data: &Array1<F>, labels: &Array1<i32>) -> MetricResult<F>;

    /// Returns whether higher values indicate better performance
    fn higher_is_better(&self) -> bool;

    /// Optional: Returns a description of the metric
    fn description(&self) -> Option<&'static str> {
        None
    }

    /// Optional: Returns the valid range of the metric (min, max)
    fn value_range(&self) -> Option<(F, F)> {
        None
    }
}

/// A wrapper that combines multiple custom metrics into a single evaluator
pub struct CustomMetricSuite<F: Float> {
    classification_metrics: Vec<Box<dyn ClassificationMetric<F> + Send + Sync>>,
    regression_metrics: Vec<Box<dyn RegressionMetric<F> + Send + Sync>>,
    clustering_metrics: Vec<Box<dyn ClusteringMetric<F> + Send + Sync>>,
}

impl<F: Float> Default for CustomMetricSuite<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> CustomMetricSuite<F> {
    /// Creates a new empty metric suite
    pub fn new() -> Self {
        Self {
            classification_metrics: Vec::new(),
            regression_metrics: Vec::new(),
            clustering_metrics: Vec::new(),
        }
    }

    /// Adds a classification metric to the suite
    pub fn add_classification_metric<M>(&mut self, metric: M) -> &mut Self
    where
        M: ClassificationMetric<F> + Send + Sync + 'static,
    {
        self.classification_metrics.push(Box::new(metric));
        self
    }

    /// Adds a regression metric to the suite
    pub fn add_regression_metric<M>(&mut self, metric: M) -> &mut Self
    where
        M: RegressionMetric<F> + Send + Sync + 'static,
    {
        self.regression_metrics.push(Box::new(metric));
        self
    }

    /// Adds a clustering metric to the suite
    pub fn add_clustering_metric<M>(&mut self, metric: M) -> &mut Self
    where
        M: ClusteringMetric<F> + Send + Sync + 'static,
    {
        self.clustering_metrics.push(Box::new(metric));
        self
    }

    /// Evaluates all classification metrics
    pub fn evaluate_classification(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> MetricsResult<CustomMetricResults<F>> {
        let mut results = CustomMetricResults::new("classification");

        for metric in &self.classification_metrics {
            match metric.compute(y_true, y_pred) {
                Ok(value) => {
                    results.add_result(metric.name(), value, metric.higher_is_better());
                }
                Err(e) => {
                    eprintln!("Warning: Failed to compute {}: {}", metric.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Evaluates all regression metrics
    pub fn evaluate_regression(
        &self,
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
    ) -> MetricsResult<CustomMetricResults<F>> {
        let mut results = CustomMetricResults::new("regression");

        for metric in &self.regression_metrics {
            match metric.compute(y_true, y_pred) {
                Ok(value) => {
                    results.add_result(metric.name(), value, metric.higher_is_better());
                }
                Err(e) => {
                    eprintln!("Warning: Failed to compute {}: {}", metric.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Evaluates all clustering metrics
    pub fn evaluate_clustering(
        &self,
        data: &Array1<F>,
        labels: &Array1<i32>,
    ) -> MetricsResult<CustomMetricResults<F>> {
        let mut results = CustomMetricResults::new("clustering");

        for metric in &self.clustering_metrics {
            match metric.compute(data, labels) {
                Ok(value) => {
                    results.add_result(metric.name(), value, metric.higher_is_better());
                }
                Err(e) => {
                    eprintln!("Warning: Failed to compute {}: {}", metric.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Gets the names of all registered metrics
    pub fn metric_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for metric in &self.classification_metrics {
            names.push(format!("classification:{}", metric.name()));
        }

        for metric in &self.regression_metrics {
            names.push(format!("regression:{}", metric.name()));
        }

        for metric in &self.clustering_metrics {
            names.push(format!("clustering:{}", metric.name()));
        }

        names
    }
}

/// Container for custom metric evaluation results
#[derive(Debug, Clone)]
pub struct CustomMetricResults<F: Float> {
    metric_type: String,
    results: Vec<CustomMetricResult<F>>,
}

#[derive(Debug, Clone)]
pub struct CustomMetricResult<F: Float> {
    pub name: String,
    pub value: F,
    pub higher_is_better: bool,
}

impl<F: Float> CustomMetricResults<F> {
    /// Creates a new results container
    pub fn new(metric_type: &str) -> Self {
        Self {
            metric_type: metric_type.to_string(),
            results: Vec::new(),
        }
    }

    /// Adds a metric result
    pub fn add_result(&mut self, name: &str, value: F, higher_is_better: bool) {
        self.results.push(CustomMetricResult {
            name: name.to_string(),
            value,
            higher_is_better,
        });
    }

    /// Gets all results
    pub fn results(&self) -> &[CustomMetricResult<F>] {
        &self.results
    }

    /// Gets the metric type
    pub fn metric_type(&self) -> &str {
        &self.metric_type
    }

    /// Gets a specific result by name
    pub fn get(&self, name: &str) -> Option<&CustomMetricResult<F>> {
        self.results.iter().find(|r| r.name == name)
    }

    /// Gets the best result according to the metric's optimization direction
    pub fn best_result(&self) -> Option<&CustomMetricResult<F>> {
        self.results.iter().max_by(|a, b| {
            let a_val = if a.higher_is_better {
                a.value
            } else {
                -a.value
            };
            let b_val = if b.higher_is_better {
                b.value
            } else {
                -b.value
            };
            a_val
                .partial_cmp(&b_val)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl<F: Float + fmt::Display> fmt::Display for CustomMetricResults<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Custom {} Metrics:", self.metric_type)?;
        writeln!(f, "{:-<50}", "")?;

        for result in &self.results {
            let direction = if result.higher_is_better {
                "↑"
            } else {
                "↓"
            };
            writeln!(
                f,
                "{:<30} {:<15} {}",
                result.name,
                format!("{:.6}", result.value),
                direction
            )?;
        }

        Ok(())
    }
}

/// Macro for easy metric trait implementation
#[macro_export]
macro_rules! classification_metric {
    ($name:ident, $metric_name:expr, $higher_is_better:expr, $compute:expr) => {
        struct $name;

        impl $crate::custom::ClassificationMetric<f64> for $name {
            fn name(&self) -> &'static str {
                $metric_name
            }

            fn compute(
                &self,
                y_true: &ndarray::Array1<i32>,
                y_pred: &ndarray::Array1<i32>,
            ) -> $crate::custom::MetricResult<f64> {
                $compute(y_true, y_pred)
            }

            fn higher_is_better(&self) -> bool {
                $higher_is_better
            }
        }
    };
}

/// Macro for easy regression metric implementation
#[macro_export]
macro_rules! regression_metric {
    ($name:ident, $metric_name:expr, $higher_is_better:expr, $compute:expr) => {
        struct $name;

        impl $crate::custom::RegressionMetric<f64> for $name {
            fn name(&self) -> &'static str {
                $metric_name
            }

            fn compute(
                &self,
                y_true: &ndarray::Array1<f64>,
                y_pred: &ndarray::Array1<f64>,
            ) -> $crate::custom::MetricResult<f64> {
                $compute(y_true, y_pred)
            }

            fn higher_is_better(&self) -> bool {
                $higher_is_better
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct TestAccuracy;

    impl ClassificationMetric<f64> for TestAccuracy {
        fn name(&self) -> &'static str {
            "test_accuracy"
        }

        fn compute(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> MetricResult<f64> {
            if y_true.len() != y_pred.len() {
                return Err("Length mismatch".into());
            }

            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(a, b)| a == b)
                .count();

            Ok(correct as f64 / y_true.len() as f64)
        }

        fn higher_is_better(&self) -> bool {
            true
        }
    }

    struct TestMSE;

    impl RegressionMetric<f64> for TestMSE {
        fn name(&self) -> &'static str {
            "test_mse"
        }

        fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricResult<f64> {
            if y_true.len() != y_pred.len() {
                return Err("Length mismatch".into());
            }

            let mse = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / y_true.len() as f64;

            Ok(mse)
        }

        fn higher_is_better(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_custom_classification_metric() {
        let metric = TestAccuracy;
        let y_true = array![1, 0, 1, 1, 0];
        let y_pred = array![1, 0, 0, 1, 0];

        let result = metric.compute(&y_true, &y_pred).unwrap();
        assert_eq!(result, 0.8);
        assert!(metric.higher_is_better());
    }

    #[test]
    fn test_custom_regression_metric() {
        let metric = TestMSE;
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];

        let result = metric.compute(&y_true, &y_pred).unwrap();
        // MSE = ((1.0-1.1)² + (2.0-2.1)² + (3.0-2.9)²) / 3 = (0.01 + 0.01 + 0.01) / 3 = 0.01
        assert!((result - 0.01).abs() < 1e-10);
        assert!(!metric.higher_is_better());
    }

    #[test]
    fn test_metric_suite() {
        let mut suite = CustomMetricSuite::new();
        suite.add_classification_metric(TestAccuracy);
        suite.add_regression_metric(TestMSE);

        // Test classification
        let y_true_cls = array![1, 0, 1, 1, 0];
        let y_pred_cls = array![1, 0, 0, 1, 0];
        let cls_results = suite
            .evaluate_classification(&y_true_cls, &y_pred_cls)
            .unwrap();

        assert_eq!(cls_results.results().len(), 1);
        assert_eq!(cls_results.get("test_accuracy").unwrap().value, 0.8);

        // Test regression
        let y_true_reg = array![1.0, 2.0, 3.0];
        let y_pred_reg = array![1.1, 2.1, 2.9];
        let reg_results = suite.evaluate_regression(&y_true_reg, &y_pred_reg).unwrap();

        assert_eq!(reg_results.results().len(), 1);
        assert!((reg_results.get("test_mse").unwrap().value - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_metric_names() {
        let mut suite = CustomMetricSuite::new();
        suite.add_classification_metric(TestAccuracy);
        suite.add_regression_metric(TestMSE);

        let names = suite.metric_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"classification:test_accuracy".to_string()));
        assert!(names.contains(&"regression:test_mse".to_string()));
    }

    #[test]
    fn test_best_result() {
        let mut results = CustomMetricResults::new("test");
        results.add_result("metric1", 0.8, true); // higher is better
        results.add_result("metric2", 0.2, false); // lower is better

        let best = results.best_result().unwrap();
        // Both metrics are equally good in their respective directions
        // but metric1 should be selected as it comes first
        assert_eq!(best.name, "metric1");
    }
}
