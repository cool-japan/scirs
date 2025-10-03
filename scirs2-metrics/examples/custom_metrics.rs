//! Example of defining and using custom metrics
//!
//! This example demonstrates how to define custom metrics and integrate them
//! with the scirs2-metrics framework.

use scirs2_core::ndarray::array;
use scirs2_metrics::custom::{
    ClassificationMetric, CustomMetricSuite, MetricResult, RegressionMetric,
};

/// Custom classification metric: Balanced accuracy considering class imbalance
struct BalancedAccuracy;

impl ClassificationMetric<f64> for BalancedAccuracy {
    fn name(&self) -> &'static str {
        "balanced_accuracy"
    }

    fn compute(
        &self,
        y_true: &scirs2_core::ndarray::Array1<i32>,
        y_pred: &scirs2_core::ndarray::Array1<i32>,
    ) -> MetricResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err("Arrays must have the same length".into());
        }

        // Get unique classes
        let mut classes = y_true.to_vec();
        classes.extend(y_pred.iter());
        classes.sort_unstable();
        classes.dedup();

        let mut class_recalls = Vec::new();

        for &class in &classes {
            let true_positives = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&true_val, &pred_val)| true_val == class && pred_val == class)
                .count() as f64;

            let total_actual = y_true.iter().filter(|&&val| val == class).count() as f64;

            if total_actual > 0.0 {
                class_recalls.push(true_positives / total_actual);
            }
        }

        if class_recalls.is_empty() {
            return Err("No valid classes found".into());
        }

        let balanced_accuracy = class_recalls.iter().sum::<f64>() / class_recalls.len() as f64;
        Ok(balanced_accuracy)
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn description(&self) -> Option<&'static str> {
        Some("Balanced accuracy that accounts for class imbalance by averaging recall across all classes")
    }

    fn value_range(&self) -> Option<(f64, f64)> {
        Some((0.0, 1.0))
    }
}

/// Custom regression metric: Symmetric Mean Absolute Percentage Error (SMAPE)
struct Smape;

impl RegressionMetric<f64> for Smape {
    fn name(&self) -> &'static str {
        "smape"
    }

    fn compute(
        &self,
        y_true: &scirs2_core::ndarray::Array1<f64>,
        y_pred: &scirs2_core::ndarray::Array1<f64>,
    ) -> MetricResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err("Arrays must have the same length".into());
        }

        let mut total_error = 0.0;
        let mut valid_points = 0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            let denominator = (true_val.abs() + pred_val.abs()) / 2.0;

            if denominator > 1e-10 {
                // Avoid division by zero
                let error = (true_val - pred_val).abs() / denominator;
                total_error += error;
                valid_points += 1;
            }
        }

        if valid_points == 0 {
            return Err("No valid data points for SMAPE calculation".into());
        }

        Ok((total_error / valid_points as f64) * 100.0)
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn description(&self) -> Option<&'static str> {
        Some("Symmetric Mean Absolute Percentage Error - percentage error metric that treats over and under-forecasting equally")
    }

    fn value_range(&self) -> Option<(f64, f64)> {
        Some((0.0, 200.0))
    }
}

/// Custom regression metric: Mean Absolute Scaled Error (MASE)
struct Mase {
    baseline_mae: f64,
}

impl Mase {
    fn new(_baselinemae: f64) -> Self {
        Self {
            baseline_mae: _baselinemae,
        }
    }
}

impl RegressionMetric<f64> for Mase {
    fn name(&self) -> &'static str {
        "mase"
    }

    fn compute(
        &self,
        y_true: &scirs2_core::ndarray::Array1<f64>,
        y_pred: &scirs2_core::ndarray::Array1<f64>,
    ) -> MetricResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err("Arrays must have the same length".into());
        }

        if self.baseline_mae <= 0.0 {
            return Err("Baseline MAE must be positive".into());
        }

        let mae = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&true_val, &pred_val)| (true_val - pred_val).abs())
            .sum::<f64>()
            / y_true.len() as f64;

        Ok(mae / self.baseline_mae)
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn description(&self) -> Option<&'static str> {
        Some("Mean Absolute Scaled Error - scaled version of MAE using a baseline model's performance")
    }

    fn value_range(&self) -> Option<(f64, f64)> {
        Some((0.0, f64::INFINITY))
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Custom Metrics Example");
    println!("=====================");

    // Test custom classification metric
    println!("\n1. Custom Classification Metric (Balanced Accuracy)");
    println!("---------------------------------------------------");

    let y_true_cls = array![0, 0, 0, 1, 1, 1, 2, 2];
    let y_pred_cls = array![0, 0, 1, 1, 1, 2, 2, 2];

    let balanced_acc = BalancedAccuracy;
    let result = balanced_acc.compute(&y_true_cls, &y_pred_cls)?;

    println!("Metric: {}", balanced_acc.name());
    println!(
        "Description: {}",
        balanced_acc.description().unwrap_or("No description")
    );
    println!("Value Range: {:?}", balanced_acc.value_range());
    println!("Higher is Better: {}", balanced_acc.higher_is_better());
    println!("Result: {result:.4}");

    // Test custom regression metrics
    println!("\n2. Custom Regression Metrics (SMAPE and MASE)");
    println!("---------------------------------------------");

    let y_true_reg = array![100.0, 150.0, 200.0, 120.0, 180.0];
    let y_pred_reg = array![95.0, 155.0, 190.0, 125.0, 175.0];

    // SMAPE
    let smape = Smape;
    let smape_result = smape.compute(&y_true_reg, &y_pred_reg)?;

    println!("\nSMAPE:");
    println!("Metric: {}", smape.name());
    println!(
        "Description: {}",
        smape.description().unwrap_or("No description")
    );
    println!("Value Range: {:?}", smape.value_range());
    println!("Higher is Better: {}", smape.higher_is_better());
    println!("Result: {smape_result:.4}%");

    // MASE (using naive baseline MAE)
    let baseline_mae = 10.0; // Assume this comes from a baseline model
    let mase = Mase::new(baseline_mae);
    let mase_result = mase.compute(&y_true_reg, &y_pred_reg)?;

    println!("\nMASE:");
    println!("Metric: {}", mase.name());
    println!(
        "Description: {}",
        mase.description().unwrap_or("No description")
    );
    println!("Value Range: {:?}", mase.value_range());
    println!("Higher is Better: {}", mase.higher_is_better());
    println!("Result: {mase_result:.4}");
    println!("(Values < 1.0 indicate better than baseline, > 1.0 indicate worse)");

    // Test metric suite
    println!("\n3. Custom Metric Suite");
    println!("----------------------");

    let mut suite = CustomMetricSuite::new();
    suite.add_classification_metric(BalancedAccuracy);
    suite.add_regression_metric(Smape);
    suite.add_regression_metric(Mase::new(baseline_mae));

    println!("Registered metrics: {:?}", suite.metric_names());

    // Evaluate classification metrics
    let cls_results = suite.evaluate_classification(&y_true_cls, &y_pred_cls)?;
    println!("\nClassification Results:");
    println!("{cls_results}");

    // Evaluate regression metrics
    let reg_results = suite.evaluate_regression(&y_true_reg, &y_pred_reg)?;
    println!("Regression Results:");
    println!("{reg_results}");

    // Find best metrics
    if let Some(best_cls) = cls_results.best_result() {
        println!(
            "Best classification metric: {} = {:.4}",
            best_cls.name, best_cls.value
        );
    }

    if let Some(best_reg) = reg_results.best_result() {
        println!(
            "Best regression metric: {} = {:.4}",
            best_reg.name, best_reg.value
        );
    }

    // Demonstrate metric usage with different data types
    println!("\n4. Edge Cases and Validation");
    println!("----------------------------");

    // Test with mismatched array lengths
    let y_true_short = array![0, 1];
    let y_pred_long = array![0, 1, 2];

    match balanced_acc.compute(&y_true_short, &y_pred_long) {
        Ok(_) => println!("Unexpected success with mismatched arrays"),
        Err(e) => println!("Expected error with mismatched arrays: {e}"),
    }

    // Test with edge case data
    let y_true_edge = array![1.0, 1.0, 1.0]; // All same values
    let y_pred_edge = array![1.0, 1.0, 1.0];

    let smape_edge = smape.compute(&y_true_edge, &y_pred_edge)?;
    println!("SMAPE with identical values: {smape_edge:.4}%");

    println!("\nCustom metrics example completed successfully!");

    Ok(())
}
