//! Example of integrating scirs2-metrics with scirs2-neural
//!
//! This example shows how to use scirs2-metrics metrics with scirs2-neural models.
//! To run this example, enable the 'neural_common' feature:
//!
//! ```bash
//! cargo run --example neural_integration --features neural_common
//! ```

use ndarray::{Array, Array1, Array2, Ix2, IxDyn};
use scirs2_metrics::classification::{
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
};
use scirs2_metrics::integration::neural::NeuralMetricAdapter;
use scirs2_metrics::regression::{mean_absolute_error, mean_squared_error, r2_score};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Metrics integration example");
    println!("--------------------------");

    // Create some example data (a simple regression problem)
    let targets = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let predictions = Array1::from_vec(vec![1.2, 1.8, 3.3, 3.5, 5.2]);

    // Create metric adapters
    let mse_adapter = NeuralMetricAdapter::<f64>::mse();
    let mae_adapter = NeuralMetricAdapter::<f64>::mae();
    let r2_adapter = NeuralMetricAdapter::<f64>::r2();

    // Compute metrics
    let mse = mse_adapter.compute(&predictions.clone().into_dyn(), &targets.clone().into_dyn())?;
    let mae = mae_adapter.compute(&predictions.clone().into_dyn(), &targets.clone().into_dyn())?;
    let r2 = r2_adapter.compute(&predictions.clone().into_dyn(), &targets.clone().into_dyn())?;

    println!("Regression metrics:");
    println!("  MSE:  {:.4}", mse);
    println!("  MAE:  {:.4}", mae);
    println!("  R²:   {:.4}", r2);

    println!("\nComparison with direct calls:");
    println!("  MSE:  {:.4}", mean_squared_error(&targets, &predictions)?);
    println!(
        "  MAE:  {:.4}",
        mean_absolute_error(&targets, &predictions)?
    );
    println!("  R²:   {:.4}", r2_score(&targets, &predictions)?);

    // Classification example (binary)
    let binary_targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    let binary_predictions = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);
    let binary_scores = Array1::from_vec(vec![0.2, 0.9, 0.3, 0.4, 0.8]);

    // Create metric adapters
    let accuracy_adapter = NeuralMetricAdapter::<f64>::accuracy();
    let precision_adapter = NeuralMetricAdapter::<f64>::precision();
    let recall_adapter = NeuralMetricAdapter::<f64>::recall();
    let f1_adapter = NeuralMetricAdapter::<f64>::f1_score();
    let roc_auc_adapter = NeuralMetricAdapter::<f64>::roc_auc();

    // Compute metrics
    let accuracy = accuracy_adapter.compute(
        &binary_predictions.clone().into_dyn(),
        &binary_targets.clone().into_dyn(),
    )?;
    let precision = precision_adapter.compute(
        &binary_predictions.clone().into_dyn(),
        &binary_targets.clone().into_dyn(),
    )?;
    let recall = recall_adapter.compute(
        &binary_predictions.clone().into_dyn(),
        &binary_targets.clone().into_dyn(),
    )?;
    let f1 = f1_adapter.compute(
        &binary_predictions.clone().into_dyn(),
        &binary_targets.clone().into_dyn(),
    )?;
    let roc_auc = roc_auc_adapter.compute(
        &binary_scores.clone().into_dyn(),
        &binary_targets.clone().into_dyn(),
    )?;

    println!("\nBinary classification metrics:");
    println!("  Accuracy:   {:.4}", accuracy);
    println!("  Precision:  {:.4}", precision);
    println!("  Recall:     {:.4}", recall);
    println!("  F1 Score:   {:.4}", f1);
    println!("  ROC AUC:    {:.4}", roc_auc);

    println!("\nComparison with direct calls:");
    println!(
        "  Accuracy:   {:.4}",
        accuracy_score(&binary_targets, &binary_predictions)?
    );
    println!(
        "  Precision:  {:.4}",
        precision_score(&binary_targets, &binary_predictions, None, None, None)?
    );
    println!(
        "  Recall:     {:.4}",
        recall_score(&binary_targets, &binary_predictions, None, None, None)?
    );
    println!(
        "  F1 Score:   {:.4}",
        f1_score(&binary_targets, &binary_predictions, None, None, None)?
    );
    println!(
        "  ROC AUC:    {:.4}",
        roc_auc_score(&binary_targets, &binary_scores)?
    );

    // Multiclass classification example
    let multiclass_targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0, 1.0]);

    // One-hot encoded predictions (batch_size x num_classes)
    let multiclass_predictions = Array2::<f64>::from_shape_vec(
        (5, 3),
        vec![
            0.9, 0.1, 0.0, // Sample 1 (predicted class 0)
            0.1, 0.8, 0.1, // Sample 2 (predicted class 1)
            0.0, 0.2, 0.8, // Sample 3 (predicted class 2)
            0.7, 0.2, 0.1, // Sample 4 (predicted class 0)
            0.3, 0.4, 0.3, // Sample 5 (predicted class 1) - less confident
        ],
    )?;

    // Convert to class predictions (for metrics that need class labels)
    let multiclass_pred_classes = multiclass_predictions.map_axis(ndarray::Axis(1), |row| {
        let mut max_idx = 0;
        let mut max_val = row[0];

        for (i, &val) in row.iter().enumerate().skip(1) {
            if val > max_val {
                max_idx = i;
                max_val = val;
            }
        }

        max_idx as f64
    });

    // Create custom adapter for multiclass F1 score
    let multiclass_f1_adapter = NeuralMetricAdapter::new(
        "multiclass_f1",
        Box::new(|preds, targets| {
            // Use macro averaging for multiclass
            Ok(f1_score(
                &targets.clone().into_dimensionality::<ndarray::Ix1>()?,
                &preds.clone().into_dimensionality::<ndarray::Ix1>()?,
                Some("macro"),
                None,
                None,
            )?)
        }),
    );

    // Compute metrics
    let multiclass_f1 = multiclass_f1_adapter.compute(
        &multiclass_pred_classes.clone().into_dyn(),
        &multiclass_targets.clone().into_dyn(),
    )?;

    println!("\nMulticlass classification metrics:");
    println!("  F1 Score (macro):  {:.4}", multiclass_f1);

    // When enabling the neural_common feature, you can use these adapters
    // directly with scirs2-neural's training loops
    #[cfg(feature = "neural_common")]
    {
        use scirs2_metrics::integration::neural::MetricsCallback;

        println!("\nWith neural_common feature enabled:");
        let metrics = vec![
            NeuralMetricAdapter::<f64>::accuracy(),
            NeuralMetricAdapter::<f64>::precision(),
            NeuralMetricAdapter::<f64>::recall(),
            NeuralMetricAdapter::<f64>::f1_score(),
        ];

        let mut callback = MetricsCallback::new(metrics, true);
        let metric_names = callback.metric_names();

        println!("  Created MetricsCallback with metrics: {:?}", metric_names);

        // Example of computing metrics with the callback
        let results =
            callback.compute_metrics(&binary_predictions.into_dyn(), &binary_targets.into_dyn())?;

        println!("  Results:");
        for (name, value) in results {
            println!("    {}: {:.4}", name, value);
        }
    }

    Ok(())
}
