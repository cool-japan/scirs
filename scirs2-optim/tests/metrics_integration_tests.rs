//! Tests for the metrics integration

#[cfg(feature = "metrics_integration")]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, ArrayView1, Ix1};
    use scirs2_optim::metrics::{MetricBasedReduceOnPlateau, MetricOptimizer, MetricScheduler};
    use scirs2_optim::optimizers::{Optimizer, SGD};
    use std::collections::HashMap;

    /// Test the MetricOptimizer with SGD
    #[test]
    fn test_metric_optimizer_sgd() {
        // Create SGD optimizer with lr=0.1
        let sgd = SGD::new(0.1);

        // Create MetricOptimizer wrapping SGD, optimizing accuracy (maximizing)
        let mut optimizer = MetricOptimizer::new(sgd, "accuracy", true);

        // Create some parameters and gradients
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Step with optimizer
        let updated_params = optimizer.step(&params, &grads).unwrap();

        // Check that parameters were updated correctly
        assert_ne!(updated_params[0], params[0]);
        assert_ne!(updated_params[1], params[1]);
        assert_ne!(updated_params[2], params[2]);

        // Update optimizer with a metric value
        optimizer.update_metric(0.75).unwrap();

        // Check metric adapter
        assert_eq!(optimizer.metric_adapter().metric_name(), "accuracy");
        assert_eq!(optimizer.metric_adapter().history().len(), 1);

        // Update with multiple metrics
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.8);
        metrics.insert("loss".to_string(), 0.3);
        optimizer.update_metrics(metrics).unwrap();

        // Check that metrics were updated
        assert_eq!(optimizer.metric_adapter().history().len(), 2);
        assert_abs_diff_eq!(optimizer.metric_adapter().history()[1], 0.8);

        // Check learning rate
        assert_abs_diff_eq!(optimizer.get_learning_rate(), 0.1);
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
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // First value - always considered an improvement
        let lr = scheduler.step_with_metric(1.0);
        assert_abs_diff_eq!(lr, 0.1);

        // Second value - worse than first, but patience is 2
        let lr = scheduler.step_with_metric(1.1);
        assert_abs_diff_eq!(lr, 0.1);

        // Third value - worse again, should reduce learning rate
        let lr = scheduler.step_with_metric(1.2);
        assert_abs_diff_eq!(lr, 0.05); // 0.1 * 0.5

        // Step without metric (should not change lr)
        let lr = scheduler.step();
        assert_abs_diff_eq!(lr, 0.05);

        // Test with optimizer
        let mut sgd = SGD::new(0.2);

        // Apply scheduler to optimizer
        scheduler.apply_to(&mut sgd);

        // Check that optimizer's learning rate was updated
        assert_abs_diff_eq!(sgd.get_learning_rate(), 0.05);
    }

    /// Test the MetricBasedReduceOnPlateau
    #[test]
    fn test_metric_based_reduce_on_plateau() {
        // Create a scheduler
        let mut scheduler = MetricBasedReduceOnPlateau::new(
            0.1,        // Initial learning rate
            0.5,        // Factor
            2,          // Patience
            0.001,      // Minimum learning rate
            "val_loss", // Metric name
            false,      // Maximize? No, minimize
        );

        // Initial learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Simulate training with plateau in validation loss
        let lr = scheduler.step_with_metric(1.0);
        assert_abs_diff_eq!(lr, 0.1);

        let lr = scheduler.step_with_metric(0.9);
        assert_abs_diff_eq!(lr, 0.1);

        let lr = scheduler.step_with_metric(0.85);
        assert_abs_diff_eq!(lr, 0.1);

        // Now validate loss plateaus
        let lr = scheduler.step_with_metric(0.84);
        assert_abs_diff_eq!(lr, 0.1);

        let lr = scheduler.step_with_metric(0.84);
        assert_abs_diff_eq!(lr, 0.1);

        let lr = scheduler.step_with_metric(0.84);
        assert_abs_diff_eq!(lr, 0.05); // Learning rate reduced

        // Check history
        assert_eq!(scheduler.metric_history().len(), 6);
        assert!(scheduler.lr_history().len() >= 2);

        // Test metrics other than the default step method
        let lr = scheduler.step();
        assert_abs_diff_eq!(lr, 0.05); // Remains the same without metric value

        // Test application to optimizer
        let mut sgd = SGD::new(0.2);
        scheduler.apply_to(&mut sgd);
        assert_abs_diff_eq!(sgd.get_learning_rate(), 0.05);
    }
}

#[cfg(not(feature = "metrics_integration"))]
#[test]
fn test_metrics_integration_feature_disabled() {
    // This test is just a placeholder when the feature is disabled
    assert!(true);
}
