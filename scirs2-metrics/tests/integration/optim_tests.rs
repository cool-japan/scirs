//! Tests for the optim integration module

use approx::assert_abs_diff_eq;
use ndarray::{Array1, ArrayD};
use num_traits::{Float, FromPrimitive};
use scirs2_metrics::integration::optim::{
    HyperParameter, HyperParameterSearchResult, HyperParameterTuner, MetricOptimizer,
    MetricScheduler, OptimizationMode,
};
use std::collections::HashMap;

/// Test the MetricOptimizer
#[test]
fn test_metric_optimizer() {
    let mut optimizer = MetricOptimizer::new("accuracy", true);

    // Test metric history recording
    optimizer.add_value(0.75);
    optimizer.add_value(0.78);
    optimizer.add_value(0.80);

    // Check history and best value
    assert_eq!(optimizer.history().len(), 3);
    assert_eq!(optimizer.best_value(), Some(0.80));

    // Test additional metrics
    optimizer.add_additional_value("loss", 0.5);
    optimizer.add_additional_value("loss", 0.4);
    optimizer.add_additional_value("loss", 0.35);

    // Check additional metrics history
    let loss_history = optimizer.additional_metric_history("loss").unwrap();
    assert_eq!(loss_history.len(), 3);
    assert_abs_diff_eq!(loss_history[0], 0.5, epsilon = 1e-10);

    // Test improvement check
    assert!(optimizer.is_improvement(0.85));
    assert!(!optimizer.is_improvement(0.75));

    // Test reset
    optimizer.reset();
    assert_eq!(optimizer.history().len(), 0);
    assert_eq!(optimizer.best_value(), None);
    assert!(optimizer
        .additional_metric_history("loss")
        .unwrap()
        .is_empty());
}

/// Test the MetricScheduler
#[test]
fn test_metric_scheduler() {
    // Create a scheduler for minimizing a loss metric
    let mut scheduler = MetricScheduler::new(
        0.1,        // Initial learning rate
        0.5,        // Factor
        2,          // Patience
        0.001,      // Minimum learning rate
        "val_loss", // Metric name
        false,      // Maximize? No, minimize
    );

    // Initial learning rate
    assert_eq!(scheduler.get_learning_rate(), 0.1);

    // First value - always considered an improvement
    let lr = scheduler.step_with_metric(1.0);
    assert_eq!(lr, 0.1);

    // Second value - worse than first, but patience is 2
    let lr = scheduler.step_with_metric(1.1);
    assert_eq!(lr, 0.1);

    // Third value - worse again, should reduce learning rate
    let lr = scheduler.step_with_metric(1.2);
    assert_eq!(lr, 0.05); // 0.1 * 0.5

    // Improvement
    let lr = scheduler.step_with_metric(0.9);
    assert_eq!(lr, 0.05);

    // Test multiple decreases and minimum limit
    let lr = scheduler.step_with_metric(1.0);
    let lr = scheduler.step_with_metric(1.1);
    assert_eq!(lr, 0.025); // 0.05 * 0.5

    let lr = scheduler.step_with_metric(1.2);
    let lr = scheduler.step_with_metric(1.3);
    assert_eq!(lr, 0.0125); // 0.025 * 0.5

    // Test minimum learning rate
    for _ in 0..10 {
        scheduler.step_with_metric(1.5);
    }
    assert_eq!(scheduler.get_learning_rate(), 0.001); // Minimum learning rate

    // Test history
    assert!(scheduler.history().len() > 1);
    assert!(scheduler.metric_history().len() > 0);

    // Test reset
    scheduler.reset();
    assert_eq!(scheduler.get_learning_rate(), 0.1);
    assert_eq!(scheduler.history().len(), 1);
    assert_eq!(scheduler.metric_history().len(), 0);
}

/// Test the HyperParameter
#[test]
fn test_hyperparameter() {
    // Test continuous hyperparameter
    let mut hp_continuous = HyperParameter::new("learning_rate", 0.01, 0.001, 0.1);
    assert_eq!(hp_continuous.name(), "learning_rate");
    assert_eq!(hp_continuous.value(), 0.01);

    // Test setting value
    hp_continuous.set_value(0.05).unwrap();
    assert_eq!(hp_continuous.value(), 0.05);

    // Test discrete hyperparameter
    let hp_discrete = HyperParameter::discrete("batch_size", 32.0, 16.0, 128.0, 16.0);
    assert_eq!(hp_discrete.name(), "batch_size");
    assert_eq!(hp_discrete.value(), 32.0);

    // Test categorical hyperparameter
    let hp_categorical = HyperParameter::categorical("activation", 0.0, vec![0.0, 1.0, 2.0]);
    assert_eq!(hp_categorical.name(), "activation");
    assert_eq!(hp_categorical.value(), 0.0);

    // Test value constraints
    let result = hp_continuous.set_value(0.2);
    assert!(result.is_err());

    let result = hp_continuous.set_value(0.0005);
    assert!(result.is_err());
}

/// Test the HyperParameterTuner
#[test]
fn test_hyperparameter_tuner() {
    // Create hyperparameters
    let params = vec![
        HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
        HyperParameter::new("weight_decay", 0.0001, 0.0, 0.001),
    ];

    // Create tuner
    let mut tuner = HyperParameterTuner::new(params, "accuracy", true, 5);

    // Test random parameter generation
    let random_params = tuner.random_params();
    assert!(random_params.contains_key("learning_rate"));
    assert!(random_params.contains_key("weight_decay"));

    let lr = random_params["learning_rate"];
    assert!(lr >= 0.001 && lr <= 0.1);

    // Test random search
    let eval_fn = |params: &HashMap<f64, f64>| -> Result<f64, Box<dyn std::error::Error>> {
        // Simple evaluation function that favors higher learning rates
        // and lower weight decay values
        let lr = params["learning_rate"];
        let wd = params["weight_decay"];

        let score = 0.5 + 0.3 * (lr - 0.001) / 0.099 - 0.2 * wd / 0.001;
        Ok(score)
    };

    let result = tuner.random_search(eval_fn).unwrap();

    // Check result properties
    assert!(result.best_metric() > 0.0);
    assert_eq!(result.best_params().len(), 2);
}

/// Test the HyperParameterSearchResult
#[test]
fn test_hyperparameter_search_result() {
    // Create a result
    let mut params = HashMap::new();
    params.insert("learning_rate".to_string(), 0.01);
    params.insert("weight_decay".to_string(), 0.0001);

    let mut result =
        HyperParameterSearchResult::new("accuracy", OptimizationMode::Maximize, 0.75, params);

    // Add evaluations
    let mut params1 = HashMap::new();
    params1.insert("learning_rate".to_string(), 0.02);
    params1.insert("weight_decay".to_string(), 0.0002);
    result.add_evaluation(params1, 0.78);

    let mut params2 = HashMap::new();
    params2.insert("learning_rate".to_string(), 0.03);
    params2.insert("weight_decay".to_string(), 0.0003);
    result.add_evaluation(params2, 0.72);

    // Test that the best parameters and metric were updated
    assert_eq!(result.best_metric(), 0.78);
    assert_eq!(result.best_params()["learning_rate"], 0.02);

    // Test history
    assert_eq!(result.history().len(), 2);
}
