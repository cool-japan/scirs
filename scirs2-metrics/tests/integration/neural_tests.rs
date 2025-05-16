//! Tests for scirs2-neural integration

use approx::assert_abs_diff_eq;
use ndarray::{array, Array, Array1, Array2, Ix1, Ix2, IxDyn};
use scirs2_metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};
use scirs2_metrics::integration::neural::{
    neural_confusion_matrix_visualization, neural_precision_recall_curve_visualization,
    neural_roc_curve_visualization, training_history_visualization, MetricsCallback,
    NeuralMetricAdapter,
};
use scirs2_metrics::regression::{mean_squared_error, r2_score};
use std::collections::HashMap;

/// Test that the neural metric adapter produces the same results as direct metric calls
#[test]
fn test_metric_adapter_consistency() {
    // Create binary classification data
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0];

    // Create metric adapters
    let accuracy_adapter = NeuralMetricAdapter::<f64>::accuracy();
    let precision_adapter = NeuralMetricAdapter::<f64>::precision();
    let recall_adapter = NeuralMetricAdapter::<f64>::recall();
    let f1_adapter = NeuralMetricAdapter::<f64>::f1_score();

    // Calculate metrics using adapters
    let adapter_accuracy = accuracy_adapter
        .compute(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();
    let adapter_precision = precision_adapter
        .compute(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();
    let adapter_recall = recall_adapter
        .compute(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();
    let adapter_f1 = f1_adapter
        .compute(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();

    // Calculate metrics directly
    let direct_accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    let direct_precision = precision_score(&y_true, &y_pred, None, None, None).unwrap();
    let direct_recall = recall_score(&y_true, &y_pred, None, None, None).unwrap();
    let direct_f1 = f1_score(&y_true, &y_pred, None, None, None).unwrap();

    // Compare results
    assert_abs_diff_eq!(adapter_accuracy, direct_accuracy, epsilon = 1e-10);
    assert_abs_diff_eq!(adapter_precision, direct_precision, epsilon = 1e-10);
    assert_abs_diff_eq!(adapter_recall, direct_recall, epsilon = 1e-10);
    assert_abs_diff_eq!(adapter_f1, direct_f1, epsilon = 1e-10);

    // Test regression metrics
    let y_true_reg = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred_reg = array![1.1, 2.1, 2.9, 4.2, 5.0];

    // Create regression metric adapters
    let mse_adapter = NeuralMetricAdapter::<f64>::mse();
    let r2_adapter = NeuralMetricAdapter::<f64>::r2();

    // Calculate metrics using adapters
    let adapter_mse = mse_adapter
        .compute(
            &y_pred_reg.clone().into_dyn(),
            &y_true_reg.clone().into_dyn(),
        )
        .unwrap();
    let adapter_r2 = r2_adapter
        .compute(
            &y_pred_reg.clone().into_dyn(),
            &y_true_reg.clone().into_dyn(),
        )
        .unwrap();

    // Calculate metrics directly
    let direct_mse = mean_squared_error(&y_true_reg, &y_pred_reg).unwrap();
    let direct_r2 = r2_score(&y_true_reg, &y_pred_reg).unwrap();

    // Compare results
    assert_abs_diff_eq!(adapter_mse, direct_mse, epsilon = 1e-10);
    assert_abs_diff_eq!(adapter_r2, direct_r2, epsilon = 1e-10);
}

/// Test the MetricsCallback functionality
#[test]
fn test_metrics_callback() {
    // Create binary classification data
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0];

    // Create metric adapters
    let metrics = vec![
        NeuralMetricAdapter::<f64>::accuracy(),
        NeuralMetricAdapter::<f64>::precision(),
        NeuralMetricAdapter::<f64>::recall(),
        NeuralMetricAdapter::<f64>::f1_score(),
    ];

    // Create callback
    let mut callback = MetricsCallback::new(metrics, false);

    // Compute metrics
    let results = callback
        .compute_metrics(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();

    // Calculate metrics directly
    let direct_accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    let direct_precision = precision_score(&y_true, &y_pred, None, None, None).unwrap();
    let direct_recall = recall_score(&y_true, &y_pred, None, None, None).unwrap();
    let direct_f1 = f1_score(&y_true, &y_pred, None, None, None).unwrap();

    // Check all results are present
    assert!(results.contains_key("accuracy"));
    assert!(results.contains_key("precision"));
    assert!(results.contains_key("recall"));
    assert!(results.contains_key("f1_score"));

    // Compare results
    assert_abs_diff_eq!(results["accuracy"], direct_accuracy, epsilon = 1e-10);
    assert_abs_diff_eq!(results["precision"], direct_precision, epsilon = 1e-10);
    assert_abs_diff_eq!(results["recall"], direct_recall, epsilon = 1e-10);
    assert_abs_diff_eq!(results["f1_score"], direct_f1, epsilon = 1e-10);

    // Test history recording
    callback.record_history();
    assert_eq!(callback.history().len(), 1);

    // Make a second record
    callback.record_history();
    assert_eq!(callback.history().len(), 2);
}

/// Test the visualization adapters
#[test]
fn test_visualization_adapters() {
    // Create binary classification data
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y_score = array![0.1, 0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.8, 0.5, 0.9];

    // Create ROC curve visualization
    let roc_viz = neural_roc_curve_visualization(
        &y_true.clone().into_dyn(),
        &y_score.clone().into_dyn(),
        Some(0.75),
    )
    .unwrap();

    // Check visualization metadata
    let viz_metadata = roc_viz.get_metadata();
    assert!(viz_metadata.title().contains("ROC"));
    assert_eq!(viz_metadata.x_label(), "False Positive Rate");
    assert_eq!(viz_metadata.y_label(), "True Positive Rate");

    // Test PR curve visualization
    let pr_viz = neural_precision_recall_curve_visualization(
        &y_true.clone().into_dyn(),
        &y_score.clone().into_dyn(),
        None,
    )
    .unwrap();

    // Check visualization metadata
    let viz_metadata = pr_viz.get_metadata();
    assert!(viz_metadata.title().contains("Precision-Recall"));
    assert_eq!(viz_metadata.x_label(), "Recall");
    assert_eq!(viz_metadata.y_label(), "Precision");

    // Test confusion matrix visualization
    let labels = vec!["Negative".to_string(), "Positive".to_string()];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let cm_viz = neural_confusion_matrix_visualization(
        &y_true.clone().into_dyn(),
        &y_pred.clone().into_dyn(),
        Some(labels),
        false,
    )
    .unwrap();

    // Check visualization metadata
    let viz_metadata = cm_viz.get_metadata();
    assert!(viz_metadata.title().contains("Confusion Matrix"));

    // Test training history visualization
    let metrics = vec!["loss".to_string(), "accuracy".to_string()];

    let mut epoch1 = HashMap::new();
    epoch1.insert("loss".to_string(), 0.5);
    epoch1.insert("accuracy".to_string(), 0.7);

    let mut epoch2 = HashMap::new();
    epoch2.insert("loss".to_string(), 0.3);
    epoch2.insert("accuracy".to_string(), 0.8);

    let history = vec![epoch1, epoch2];

    let history_viz = training_history_visualization(metrics, history, None);

    // Check visualization metadata
    let viz_metadata = history_viz.get_metadata();
    assert!(viz_metadata.title().contains("Training History"));
    assert_eq!(viz_metadata.x_label(), "Epoch");
}

/// Test creating a custom metric adapter
#[test]
fn test_custom_metric_adapter() {
    // Create custom metric adapter
    let custom_metric = NeuralMetricAdapter::new(
        "custom_metric",
        Box::new(|preds, targets| {
            // A simple custom metric that computes the ratio of matching elements
            let preds_flat = preds.clone().into_dimensionality::<Ix1>().unwrap();
            let targets_flat = targets.clone().into_dimensionality::<Ix1>().unwrap();

            let mut matching = 0;
            for (p, t) in preds_flat.iter().zip(targets_flat.iter()) {
                if (p - t).abs() < 1e-10 {
                    matching += 1;
                }
            }

            Ok(matching as f64 / preds_flat.len() as f64)
        }),
    );

    // Test data
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0, 5.0, 6.0];

    // Compute custom metric
    let result = custom_metric
        .compute(&y_pred.clone().into_dyn(), &y_true.clone().into_dyn())
        .unwrap();

    // Expected result: 3 matching elements out of 5
    assert_abs_diff_eq!(result, 0.6, epsilon = 1e-10);
}
