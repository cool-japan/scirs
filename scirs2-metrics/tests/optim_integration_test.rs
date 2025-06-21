//! Integration tests for scirs2-optim compatibility
//!
//! These tests demonstrate how scirs2-metrics can be used with external optimizers
//! and schedulers without circular dependencies.

use approx::assert_abs_diff_eq;
use ndarray::array;
use scirs2_metrics::classification::accuracy_score;
use scirs2_metrics::integration::optim::{
    MetricOptimizer, MetricSchedulerTrait, OptimizationMode, SchedulerConfig,
};

#[test]
fn test_metric_optimizer_basic_functionality() {
    // Test the basic metric optimizer functionality
    let mut optimizer = MetricOptimizer::<f64>::new("accuracy", true);

    assert_eq!(optimizer.metric_name(), "accuracy");
    assert_eq!(optimizer.mode(), OptimizationMode::Maximize);
    assert!(optimizer.best_value().is_none());
    assert_eq!(optimizer.history().len(), 0);

    // Add some values
    optimizer.add_value(0.8);
    optimizer.add_value(0.85);
    optimizer.add_value(0.82); // Worse than previous best
    optimizer.add_value(0.9); // New best

    assert_eq!(optimizer.best_value(), Some(0.9));
    assert_eq!(optimizer.history().len(), 4);

    // Test improvement detection
    assert!(optimizer.is_improvement(0.95));
    assert!(!optimizer.is_improvement(0.85));
}

#[test]
fn test_optimization_modes() {
    // Test maximize mode
    let mut maximizer = MetricOptimizer::<f64>::new("accuracy", true);
    maximizer.add_value(0.8);
    maximizer.add_value(0.9);
    assert!(maximizer.is_better(0.9, 0.8));
    assert!(!maximizer.is_better(0.8, 0.9));

    // Test minimize mode
    let mut minimizer = MetricOptimizer::<f64>::new("loss", false);
    minimizer.add_value(0.9);
    minimizer.add_value(0.8);
    assert!(minimizer.is_better(0.8, 0.9));
    assert!(!minimizer.is_better(0.9, 0.8));
}

#[test]
fn test_additional_metrics_tracking() {
    let mut optimizer = MetricOptimizer::<f64>::new("f1_score", true);

    // Add main metric
    optimizer.add_value(0.8);
    optimizer.add_value(0.85);

    // Add additional metrics
    optimizer.add_additional_value("precision", 0.82);
    optimizer.add_additional_value("precision", 0.87);
    optimizer.add_additional_value("recall", 0.78);
    optimizer.add_additional_value("recall", 0.83);

    // Check additional metrics
    let precision_history = optimizer.additional_metric_history("precision").unwrap();
    assert_eq!(precision_history, &[0.82, 0.87]);

    let recall_history = optimizer.additional_metric_history("recall").unwrap();
    assert_eq!(recall_history, &[0.78, 0.83]);

    assert!(optimizer.additional_metric_history("nonexistent").is_none());
}

#[test]
fn test_scheduler_config_creation() {
    let optimizer = MetricOptimizer::<f64>::new("accuracy", true);

    let config = optimizer.create_scheduler_config(0.01, 0.5, 10, 1e-6);

    assert_eq!(config.initial_lr, 0.01);
    assert_eq!(config.factor, 0.5);
    assert_eq!(config.patience, 10);
    assert_eq!(config.min_lr, 1e-6);
    assert_eq!(config.mode, OptimizationMode::Maximize);
    assert_eq!(config.metric_name, "accuracy");

    // Test tuple conversion
    let (initial_lr, factor, patience, min_lr, mode) = config.as_tuple();
    assert_eq!(initial_lr, 0.01);
    assert_eq!(factor, 0.5);
    assert_eq!(patience, 10);
    assert_eq!(min_lr, 1e-6);
    assert_eq!(mode, OptimizationMode::Maximize);
}

#[test]
fn test_scheduler_config_manual_creation() {
    let config = SchedulerConfig::new(
        0.001,
        0.8,
        5,
        1e-7,
        OptimizationMode::Minimize,
        "loss".to_string(),
    );

    assert_eq!(config.initial_lr, 0.001);
    assert_eq!(config.factor, 0.8);
    assert_eq!(config.patience, 5);
    assert_eq!(config.min_lr, 1e-7);
    assert_eq!(config.mode, OptimizationMode::Minimize);
    assert_eq!(config.metric_name, "loss");
}

#[test]
fn test_real_world_workflow() {
    // Simulate a real training workflow with metrics
    let y_true = array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
    let mut optimizer = MetricOptimizer::<f64>::new("accuracy", true);

    // Simulate training epochs with improving accuracy
    let predictions = [
        array![1, 0, 0, 1, 0, 1, 1, 0, 1, 0], // epoch 1: accuracy = 0.7
        array![1, 0, 1, 1, 0, 1, 0, 0, 1, 0], // epoch 2: accuracy = 0.8
        array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1], // epoch 3: accuracy = 1.0
        array![1, 0, 1, 1, 1, 1, 0, 0, 1, 1], // epoch 4: accuracy = 0.9 (slight overfitting)
    ];

    let mut best_epoch = 0;
    let mut _scheduler_updated = false;

    for (epoch, y_pred) in predictions.iter().enumerate() {
        let accuracy = accuracy_score(&y_true, y_pred).unwrap();

        // Track if this is an improvement
        let is_improvement = optimizer.is_improvement(accuracy);
        if is_improvement {
            best_epoch = epoch;
        }

        // Add the metric value
        optimizer.add_value(accuracy);

        // Simulate scheduler logic - reduce LR if no improvement for 2 epochs
        if !is_improvement && optimizer.history().len() >= 3 {
            // Check if last 2 epochs showed no improvement
            let recent_best = optimizer.history()[optimizer.history().len() - 3..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            if recent_best == 0 {
                // No improvement in recent epochs
                _scheduler_updated = true;
            }
        }
    }

    // Verify results
    assert_eq!(optimizer.best_value(), Some(1.0));
    assert_eq!(best_epoch, 2); // Third epoch had perfect accuracy
    assert_eq!(optimizer.history().len(), 4);

    // Create scheduler configuration for external use
    let scheduler_config = optimizer.create_scheduler_config(0.01, 0.5, 2, 1e-6);
    assert_eq!(scheduler_config.patience, 2);
    assert_eq!(scheduler_config.mode, OptimizationMode::Maximize);
}

#[test]
fn test_integration_with_different_numeric_types() {
    // Test with f32
    let mut optimizer_f32 = MetricOptimizer::<f32>::new("mse", false);
    optimizer_f32.add_value(0.5_f32);
    optimizer_f32.add_value(0.3_f32);
    assert_eq!(optimizer_f32.best_value(), Some(0.3_f32));

    // Test with f64
    let mut optimizer_f64 = MetricOptimizer::<f64>::new("r2", true);
    optimizer_f64.add_value(0.85_f64);
    optimizer_f64.add_value(0.92_f64);
    assert_eq!(optimizer_f64.best_value(), Some(0.92_f64));
}

#[test]
fn test_reset_functionality() {
    let mut optimizer = MetricOptimizer::<f64>::new("f1", true);

    // Add some data
    optimizer.add_value(0.7);
    optimizer.add_value(0.8);
    optimizer.add_additional_value("precision", 0.85);

    assert_eq!(optimizer.history().len(), 2);
    assert_eq!(optimizer.best_value(), Some(0.8));
    assert!(optimizer.additional_metric_history("precision").is_some());

    // Reset
    optimizer.reset();

    assert_eq!(optimizer.history().len(), 0);
    assert_eq!(optimizer.best_value(), None);
    assert!(optimizer.additional_metric_history("precision").is_none());
}

#[test]
fn test_display_optimization_mode() {
    assert_eq!(format!("{}", OptimizationMode::Minimize), "minimize");
    assert_eq!(format!("{}", OptimizationMode::Maximize), "maximize");
}

/// Mock implementation of MetricSchedulerTrait for testing
struct MockScheduler<F> {
    learning_rate: F,
    mode: OptimizationMode,
}

impl<F: Clone> MockScheduler<F> {
    fn new(lr: F) -> Self {
        Self {
            learning_rate: lr,
            mode: OptimizationMode::Minimize,
        }
    }
}

impl MetricSchedulerTrait<f64> for MockScheduler<f64> {
    fn step_with_metric(&mut self, _metric: f64) -> f64 {
        // Simple mock: reduce LR by 10% on each step
        self.learning_rate *= 0.9;
        self.learning_rate
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn reset(&mut self) {
        self.learning_rate = 0.01; // Reset to default
    }

    fn set_mode(&mut self, mode: OptimizationMode) {
        self.mode = mode;
    }
}

#[test]
fn test_external_scheduler_trait() {
    let mut scheduler = MockScheduler::new(0.01);

    assert_eq!(scheduler.get_learning_rate(), 0.01);

    // Step with metric (should reduce LR)
    let new_lr = scheduler.step_with_metric(0.5);
    assert_abs_diff_eq!(new_lr, 0.009, epsilon = 1e-10);
    assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.009, epsilon = 1e-10);

    // Test mode setting
    scheduler.set_mode(OptimizationMode::Maximize);

    // Test reset
    scheduler.reset();
    assert_eq!(scheduler.get_learning_rate(), 0.01);
}
