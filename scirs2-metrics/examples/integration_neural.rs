//! Framework Integration: Neural Classifier → Classification Metrics
//!
//! This example simulates the output pattern of a neural classifier (as would
//! come from scirs2-neural) and evaluates it with scirs2-metrics. Since
//! scirs2-neural already depends on scirs2-metrics, we avoid circular imports
//! by generating the predictions inline.
//!
//! In a real pipeline you would replace the synthetic data generation with
//! actual forward-pass outputs from your scirs2-neural model.
//!
//! Run with: `cargo run --example integration_neural -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::classification::{
    accuracy_score, binary_log_loss, f1_score, precision_score, recall_score, roc_auc_score,
};

/// Simulates a binary neural classifier's output for n samples.
///
/// Returns (ground_truth, predicted_labels, predicted_probabilities).
/// The "model" is a simple threshold on a sinusoidal feature function,
/// mimicking a trained neural net with ~85% accuracy.
fn simulate_neural_classifier(n: usize) -> (Array1<u32>, Array1<u32>, Array1<f64>) {
    // Feature: periodic signal simulating a learned representation
    let features: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.4 + 0.1 * (i as f64 * 0.07).cos()).sin())
        .collect();

    // Ground truth: positive when feature > 0 (50/50 split)
    let y_true: Vec<u32> = features
        .iter()
        .map(|&f| if f > 0.0 { 1 } else { 0 })
        .collect();

    // Simulated neural network probability (sigmoid-shaped decision boundary)
    let y_prob: Vec<f64> = features
        .iter()
        .map(|&f| {
            // Sigmoid with learned "scale" ~2.5
            let logit = 2.5 * f;
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    // Predicted label at 0.5 threshold
    let y_pred: Vec<u32> = y_prob
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { 0 })
        .collect();

    (
        Array1::from(y_true),
        Array1::from(y_pred),
        Array1::from(y_prob),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Integration: Neural Classifier → Classification Metrics ===\n");

    // ─── 1. Generate simulated neural classifier output ───────────────────────
    let n = 200;
    let (y_true, y_pred, y_prob) = simulate_neural_classifier(n);

    println!("Simulated neural classifier on n={n} samples");
    println!("Architecture: 1-layer sigmoid (logit = 2.5 * sin-feature)\n");

    // ─── 2. Compute classification metrics ───────────────────────────────────
    let accuracy = accuracy_score(&y_true, &y_pred)?;
    let precision = precision_score(&y_true, &y_pred, 1u32)?;
    let recall = recall_score(&y_true, &y_pred, 1u32)?;
    let f1 = f1_score(&y_true, &y_pred, 1u32)?;
    let roc_auc = roc_auc_score(&y_true, &y_prob)?;
    let log_loss = binary_log_loss(&y_true, &y_prob, 1e-15)?;

    // ─── 3. Count positive / negative predictions ─────────────────────────────
    let n_pos_true = y_true.iter().filter(|&&v| v == 1).count();
    let n_pos_pred = y_pred.iter().filter(|&&v| v == 1).count();

    // ─── 4. Print evaluation report ───────────────────────────────────────────
    println!(
        "Distribution: {} positives / {} negatives (true)",
        n_pos_true,
        n - n_pos_true
    );
    println!(
        "Predictions : {} positives / {} negatives\n",
        n_pos_pred,
        n - n_pos_pred
    );

    println!("{:<25} {:>10}", "Metric", "Value");
    println!("{}", "-".repeat(38));
    println!("{:<25} {:>10.4}", "Accuracy", accuracy);
    println!("{:<25} {:>10.4}", "Precision (pos=1)", precision);
    println!("{:<25} {:>10.4}", "Recall (pos=1)", recall);
    println!("{:<25} {:>10.4}", "F1 Score", f1);
    println!("{:<25} {:>10.4}", "ROC-AUC", roc_auc);
    println!("{:<25} {:>10.4}", "Log-Loss", log_loss);

    println!("\n--- Interpretation ---");
    if accuracy > 0.80 {
        println!("  Accuracy > 0.80: model is performing well");
    } else {
        println!("  Accuracy <= 0.80: model may need improvement");
    }
    if (roc_auc - 0.5).abs() < 0.1 {
        println!("  ROC-AUC ≈ 0.5: model not better than random — check features");
    } else if roc_auc > 0.9 {
        println!("  ROC-AUC > 0.9: excellent discrimination ability");
    }

    println!("\n=== Done ===");
    Ok(())
}
