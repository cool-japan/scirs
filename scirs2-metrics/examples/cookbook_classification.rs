//! Cookbook: Classification Metrics
//!
//! This example demonstrates how to evaluate binary classification models using
//! scirs2-metrics. It covers accuracy, precision, recall, F1, ROC-AUC, and log-loss
//! on synthetic binary classification data.
//!
//! Run with: `cargo run --example cookbook_classification -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::classification::{
    accuracy_score, binary_log_loss, f1_score, precision_score, recall_score, roc_auc_score,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Classification Metrics ===\n");

    // ─── 1. Synthetic binary classification data (n=20) ─────────────────────
    // Simulate ground-truth labels and predicted labels / probability scores.
    // A "good" classifier: correct on 16 out of 20 samples.
    let n: usize = 20;

    // Ground-truth: alternating 0/1 labels (10 positives, 10 negatives)
    let y_true_raw: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
    // Predicted labels: matches y_true except for indices 3, 7, 11, 15
    let y_pred_raw: Vec<u32> = (0..n)
        .map(|i| {
            if [3usize, 7, 11, 15].contains(&i) {
                ((i + 1) % 2) as u32
            } else {
                (i % 2) as u32
            }
        })
        .collect();
    // Probability scores for the positive class (higher = more likely positive)
    // Scores are constructed to produce a reasonably high AUC
    let y_prob_raw: Vec<f64> = (0..n)
        .map(|i| {
            let base = if i % 2 == 1 { 0.7 } else { 0.3 };
            // Add a small sinusoidal perturbation to vary the scores
            base + 0.1 * ((i as f64 * 0.314).sin())
        })
        .collect();

    let y_true = Array1::from(y_true_raw.clone());
    let y_pred = Array1::from(y_pred_raw);
    let y_prob = Array1::from(y_prob_raw.clone());

    // ─── 2. Compute metrics ──────────────────────────────────────────────────
    let accuracy = accuracy_score(&y_true, &y_pred)?;
    let precision = precision_score(&y_true, &y_pred, 1u32)?;
    let recall = recall_score(&y_true, &y_pred, 1u32)?;
    let f1 = f1_score(&y_true, &y_pred, 1u32)?;
    let roc_auc = roc_auc_score(&y_true, &y_prob)?;

    // binary_log_loss takes u32 ground-truth and f64 probabilities
    // eps clips probabilities away from 0/1 for numerical stability
    let log_loss = binary_log_loss(&y_true, &y_prob, 1e-15)?;

    // ─── 3. Print results with commentary ───────────────────────────────────
    println!("Dataset: n={n} samples, 10 positives, 10 negatives");
    println!("Errors: 4 samples misclassified (indices 3, 7, 11, 15)\n");

    println!("{:<25} {:>10}  Commentary", "Metric", "Value");
    println!("{}", "-".repeat(60));

    println!(
        "{:<25} {:>10.4}  Overall fraction correct. Use when classes balanced.",
        "Accuracy", accuracy
    );
    println!(
        "{:<25} {:>10.4}  TP/(TP+FP). Use when false positives are costly.",
        "Precision (pos=1)", precision
    );
    println!(
        "{:<25} {:>10.4}  TP/(TP+FN). Use when false negatives are costly.",
        "Recall (pos=1)", recall
    );
    println!(
        "{:<25} {:>10.4}  Harmonic mean of Precision & Recall.",
        "F1 Score (pos=1)", f1
    );
    println!(
        "{:<25} {:>10.4}  Rank-based: 0.5=random, 1.0=perfect ranking.",
        "ROC-AUC", roc_auc
    );
    println!(
        "{:<25} {:>10.4}  Lower is better. Penalises confident wrong predictions.",
        "Log-Loss", log_loss
    );

    println!("\n--- When to use which metric ---");
    println!("  Accuracy  : balanced classes, equal cost for errors");
    println!("  Precision : minimise false alarms (spam detection, fraud alerts)");
    println!("  Recall    : minimise missed detections (cancer screening, fraud)");
    println!("  F1        : imbalanced classes, single summary of prec+recall");
    println!("  ROC-AUC   : comparing model ranks regardless of threshold choice");
    println!("  Log-Loss  : training/calibration — rewards confident correct preds");

    println!("\n=== Done ===");
    Ok(())
}
