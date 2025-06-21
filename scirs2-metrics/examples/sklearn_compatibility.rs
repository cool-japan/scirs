//! Example of scikit-learn compatibility features
//!
//! This example demonstrates the scikit-learn equivalent implementations
//! and shows how they produce identical results to their scikit-learn counterparts.

use ndarray::{Array1, Array2};
use scirs2_metrics::error::Result;
use scirs2_metrics::sklearn_compat::*;

fn main() -> Result<()> {
    println!("Scikit-learn Compatibility Example");
    println!("=================================");

    // Example 1: Classification Report
    println!("\n1. Classification Report (sklearn.metrics.classification_report)");
    println!("---------------------------------------------------------------");

    classification_report_example()?;

    // Example 2: Precision, Recall, F-score, Support
    println!("\n2. Precision, Recall, F-score, Support");
    println!("-------------------------------------");

    precision_recall_fscore_support_example()?;

    // Example 3: Cohen's Kappa Score
    println!("\n3. Cohen's Kappa Score");
    println!("--------------------");

    cohen_kappa_example()?;

    // Example 4: Multilabel Confusion Matrix
    println!("\n4. Multilabel Confusion Matrix");
    println!("-----------------------------");

    multilabel_confusion_matrix_example()?;

    // Example 5: Loss Functions
    println!("\n5. Loss Functions");
    println!("---------------");

    loss_functions_example()?;

    // Example 6: Advanced Averaging Methods
    println!("\n6. Advanced Averaging Methods");
    println!("---------------------------");

    averaging_methods_example()?;

    // Example 7: Weighted Metrics
    println!("\n7. Weighted Metrics");
    println!("-----------------");

    weighted_metrics_example()?;

    println!("\nScikit-learn compatibility example completed successfully!");
    Ok(())
}

/// Example of classification report functionality
fn classification_report_example() -> Result<()> {
    // Create sample multi-class classification data
    let y_true = Array1::from_vec(vec![0, 1, 2, 2, 0, 1, 0, 2, 1, 0]);
    let y_pred = Array1::from_vec(vec![0, 2, 1, 2, 0, 1, 1, 2, 1, 0]);

    // Class names for better readability
    let target_names = vec![
        "Setosa".to_string(),
        "Versicolor".to_string(),
        "Virginica".to_string(),
    ];

    let report =
        classification_report_sklearn(&y_true, &y_pred, None, Some(&target_names), 2, 0.0)?;

    println!("Classification Report Results:");
    println!("  Overall Accuracy: {:.4}", report.accuracy);

    println!("\n  Per-class Metrics:");
    for (class_name, precision) in &report.precision {
        let recall = report.recall.get(class_name).unwrap_or(&0.0);
        let f1 = report.f1_score.get(class_name).unwrap_or(&0.0);
        let support = report.support.get(class_name).unwrap_or(&0);

        println!(
            "    {}: Precision={:.4}, Recall={:.4}, F1={:.4}, Support={}",
            class_name, precision, recall, f1, support
        );
    }

    println!("\n  Macro Average:");
    println!("    Precision: {:.4}", report.macro_avg.precision);
    println!("    Recall: {:.4}", report.macro_avg.recall);
    println!("    F1-score: {:.4}", report.macro_avg.f1_score);

    println!("\n  Weighted Average:");
    println!("    Precision: {:.4}", report.weighted_avg.precision);
    println!("    Recall: {:.4}", report.weighted_avg.recall);
    println!("    F1-score: {:.4}", report.weighted_avg.f1_score);

    Ok(())
}

/// Example of precision, recall, f-score, and support calculation
fn precision_recall_fscore_support_example() -> Result<()> {
    let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2, 1, 1, 2]);

    println!("Testing different averaging methods:");

    // Test different averaging methods
    let averaging_methods = vec![
        (None, "Per-class (no averaging)"),
        (Some("micro"), "Micro averaging"),
        (Some("macro"), "Macro averaging"),
        (Some("weighted"), "Weighted averaging"),
    ];

    for (average, description) in averaging_methods {
        let (precision, recall, fscore, support) = precision_recall_fscore_support_sklearn(
            &y_true, &y_pred, 1.0,  // beta for F-score
            None, // labels
            None, // pos_label
            average, None, // warn_for
            0.0,  // zero_division
        )?;

        println!("\n  {}:", description);
        if average.is_some() {
            // Single values for averaged results
            println!("    Precision: {:.4}", precision[0]);
            println!("    Recall: {:.4}", recall[0]);
            println!("    F-score: {:.4}", fscore[0]);
            println!("    Support: {}", support[0]);
        } else {
            // Multiple values for per-class results
            println!("    Precision: {:?}", precision.to_vec());
            println!("    Recall: {:?}", recall.to_vec());
            println!("    F-score: {:?}", fscore.to_vec());
            println!("    Support: {:?}", support.to_vec());
        }
    }

    // Test with different beta values for F-beta score
    println!("\n  F-beta scores with different beta values:");
    for &beta in &[0.5, 1.0, 2.0] {
        let (_, _, fbeta, _) = precision_recall_fscore_support_sklearn(
            &y_true,
            &y_pred,
            beta,
            None,
            None,
            Some("macro"),
            None,
            0.0,
        )?;

        println!("    F{:.1}-score: {:.4}", beta, fbeta[0]);
    }

    Ok(())
}

/// Example of Cohen's kappa score calculation
fn cohen_kappa_example() -> Result<()> {
    println!("Testing Cohen's Kappa Score with different scenarios:");

    // Perfect agreement
    let y1_perfect = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
    let y2_perfect = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
    let kappa_perfect = cohen_kappa_score_sklearn(&y1_perfect, &y2_perfect, None, None, None)?;
    println!("  Perfect agreement: {:.4}", kappa_perfect);

    // Random agreement
    let y1_random = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
    let y2_random = Array1::from_vec(vec![1, 0, 1, 0, 1, 0]);
    let kappa_random = cohen_kappa_score_sklearn(&y1_random, &y2_random, None, None, None)?;
    println!("  Random-like agreement: {:.4}", kappa_random);

    // Partial agreement
    let y1_partial = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
    let y2_partial = Array1::from_vec(vec![0, 1, 1, 1, 2, 1]);
    let kappa_partial = cohen_kappa_score_sklearn(&y1_partial, &y2_partial, None, None, None)?;
    println!("  Partial agreement: {:.4}", kappa_partial);

    // Test with different weighting schemes
    println!("\n  Testing different weighting schemes:");

    let weight_schemes = vec![
        (None, "No weighting"),
        (Some("linear"), "Linear weighting"),
        (Some("quadratic"), "Quadratic weighting"),
    ];

    for (weights, description) in weight_schemes {
        let kappa = cohen_kappa_score_sklearn(&y1_partial, &y2_partial, None, weights, None)?;
        println!("    {}: {:.4}", description, kappa);
    }

    // Test with sample weights
    let sample_weights = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    let kappa_weighted =
        cohen_kappa_score_sklearn(&y1_partial, &y2_partial, None, None, Some(&sample_weights))?;
    println!("  With sample weights: {:.4}", kappa_weighted);

    Ok(())
}

/// Example of multilabel confusion matrix
fn multilabel_confusion_matrix_example() -> Result<()> {
    // Create multilabel classification data
    // Each row is a sample, each column is a label (0 or 1)
    let y_true = Array2::from_shape_vec(
        (5, 3),
        vec![
            1, 0, 1, // Sample 0: labels 0 and 2 are positive
            0, 1, 0, // Sample 1: label 1 is positive
            1, 1, 0, // Sample 2: labels 0 and 1 are positive
            0, 0, 1, // Sample 3: label 2 is positive
            1, 0, 0, // Sample 4: label 0 is positive
        ],
    )
    .unwrap();

    let y_pred = Array2::from_shape_vec(
        (5, 3),
        vec![
            1, 0, 0, // Sample 0: only label 0 predicted
            0, 1, 1, // Sample 1: labels 1 and 2 predicted
            1, 0, 0, // Sample 2: only label 0 predicted
            1, 0, 1, // Sample 3: labels 0 and 2 predicted
            1, 1, 0, // Sample 4: labels 0 and 1 predicted
        ],
    )
    .unwrap();

    println!("Multilabel data shape: {:?}", y_true.shape());
    println!("  True labels:\n{:?}", y_true);
    println!("  Predicted labels:\n{:?}", y_pred);

    let confusion_matrices = multilabel_confusion_matrix_sklearn(&y_true, &y_pred, None, None)?;

    println!("\nConfusion matrices for each label:");
    let n_labels = y_true.ncols();

    for label in 0..n_labels {
        let base_idx = label * 2;
        let tn = confusion_matrices[[base_idx, 0]];
        let fp = confusion_matrices[[base_idx, 1]];
        let fn_val = confusion_matrices[[base_idx + 1, 0]];
        let tp = confusion_matrices[[base_idx + 1, 1]];

        println!("  Label {}:", label);
        println!("    True Negatives (TN): {}", tn);
        println!("    False Positives (FP): {}", fp);
        println!("    False Negatives (FN): {}", fn_val);
        println!("    True Positives (TP): {}", tp);

        // Calculate per-label metrics
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_val > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!("    Precision: {:.4}", precision);
        println!("    Recall: {:.4}", recall);
        println!("    F1-score: {:.4}", f1);
        println!();
    }

    // Test with specific labels
    let specific_labels = vec![0, 2]; // Only evaluate labels 0 and 2
    let confusion_matrices_subset =
        multilabel_confusion_matrix_sklearn(&y_true, &y_pred, None, Some(&specific_labels))?;

    println!("Confusion matrices for labels 0 and 2 only:");
    println!("  Shape: {:?}", confusion_matrices_subset.shape());

    Ok(())
}

/// Example of loss functions
fn loss_functions_example() -> Result<()> {
    // Test zero-one loss
    let y_true_binary = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
    let y_pred_binary = Array1::from_vec(vec![0, 1, 1, 1, 0, 0]);

    println!("Zero-One Loss:");
    let loss_normalized = zero_one_loss_sklearn(&y_true_binary, &y_pred_binary, true, None)?;
    let loss_count = zero_one_loss_sklearn(&y_true_binary, &y_pred_binary, false, None)?;

    println!("  Normalized (fraction of errors): {:.4}", loss_normalized);
    println!("  Count (number of errors): {:.1}", loss_count);

    // Test with sample weights
    let sample_weights = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    let weighted_loss =
        zero_one_loss_sklearn(&y_true_binary, &y_pred_binary, true, Some(&sample_weights))?;
    println!("  Weighted loss: {:.4}", weighted_loss);

    // Test hinge loss
    println!("\nHinge Loss:");
    let y_true_multiclass = Array1::from_vec(vec![0, 1, 2, 0, 1]);
    let y_pred_scores = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.8, 0.1, 0.1, // Sample 0: class 0 (correct)
            0.2, 0.7, 0.1, // Sample 1: class 1 (correct)
            0.1, 0.2, 0.7, // Sample 2: class 2 (correct)
            0.3, 0.4, 0.3, // Sample 3: class 1 predicted (wrong, should be 0)
            0.1, 0.8, 0.1, // Sample 4: class 1 (correct)
        ],
    )
    .unwrap();

    let hinge_loss = hinge_loss_sklearn(&y_true_multiclass, &y_pred_scores, None, None)?;
    println!("  Hinge loss: {:.4}", hinge_loss);

    // Test hinge loss with sample weights
    let hinge_weights = Array1::from_vec(vec![1.0, 1.0, 2.0, 1.0, 1.0]);
    let weighted_hinge_loss = hinge_loss_sklearn(
        &y_true_multiclass,
        &y_pred_scores,
        None,
        Some(&hinge_weights),
    )?;
    println!("  Weighted hinge loss: {:.4}", weighted_hinge_loss);

    Ok(())
}

/// Example of different averaging methods
fn averaging_methods_example() -> Result<()> {
    // Create imbalanced multi-class data
    let y_true = Array1::from_vec(vec![
        0, 0, 0, 0, 0, 0, // 6 samples of class 0
        1, 1, 1, // 3 samples of class 1
        2, // 1 sample of class 2
    ]);
    let y_pred = Array1::from_vec(vec![
        0, 0, 0, 1, 0, 0, // 5/6 correct for class 0
        1, 1, 0, // 2/3 correct for class 1
        2, // 1/1 correct for class 2
    ]);

    println!("Imbalanced dataset analysis:");
    println!("  Class distribution: Class 0: 6, Class 1: 3, Class 2: 1");

    let averaging_methods = vec![
        (Some("micro"), "Micro-average"),
        (Some("macro"), "Macro-average"),
        (Some("weighted"), "Weighted-average"),
    ];

    for (average_method, description) in averaging_methods {
        let (precision, recall, f1, support) = precision_recall_fscore_support_sklearn(
            &y_true,
            &y_pred,
            1.0,
            None,
            None,
            average_method,
            None,
            0.0,
        )?;

        println!("\n  {}:", description);
        println!("    Precision: {:.4}", precision[0]);
        println!("    Recall: {:.4}", recall[0]);
        println!("    F1-score: {:.4}", f1[0]);
        println!("    Support: {}", support[0]);
    }

    // Compare with per-class results
    let (precision_per_class, recall_per_class, f1_per_class, support_per_class) =
        precision_recall_fscore_support_sklearn(
            &y_true, &y_pred, 1.0, None, None, None, // No averaging
            None, 0.0,
        )?;

    println!("\n  Per-class results:");
    for (i, (&p, (&r, (&f, &s)))) in precision_per_class
        .iter()
        .zip(
            recall_per_class
                .iter()
                .zip(f1_per_class.iter().zip(support_per_class.iter())),
        )
        .enumerate()
    {
        println!(
            "    Class {}: P={:.4}, R={:.4}, F1={:.4}, Support={}",
            i, p, r, f, s
        );
    }

    println!("\n  Averaging method explanations:");
    println!("    • Micro-average: Calculate globally across all classes");
    println!("    • Macro-average: Calculate for each class separately, then average");
    println!("    • Weighted-average: Calculate for each class, then weight by class frequency");

    Ok(())
}

/// Example of weighted metrics
fn weighted_metrics_example() -> Result<()> {
    let y_true = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
    let y_pred = Array1::from_vec(vec![0, 1, 1, 1, 0, 0]);

    println!("Sample weighting example:");
    println!("  True labels: {:?}", y_true.to_vec());
    println!("  Pred labels: {:?}", y_pred.to_vec());

    // Test different weight distributions
    let weight_scenarios = vec![
        (
            Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "Equal weights",
        ),
        (
            Array1::from_vec(vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0]),
            "Higher weight for class 0",
        ),
        (
            Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
            "Higher weight for class 1",
        ),
        (
            Array1::from_vec(vec![0.5, 0.5, 1.0, 1.0, 2.0, 2.0]),
            "Increasing weights",
        ),
    ];

    for (weights, description) in weight_scenarios {
        println!("\n  {}:", description);
        println!("    Weights: {:?}", weights.to_vec());

        // Test zero-one loss with weights
        let weighted_loss = zero_one_loss_sklearn(&y_true, &y_pred, true, Some(&weights))?;
        println!("    Weighted zero-one loss: {:.4}", weighted_loss);

        // Test Cohen's kappa with weights
        let weighted_kappa =
            cohen_kappa_score_sklearn(&y_true, &y_pred, None, None, Some(&weights))?;
        println!("    Weighted Cohen's kappa: {:.4}", weighted_kappa);
    }

    // Demonstrate effect of weights on multilabel metrics
    println!("\n  Multilabel weighting example:");
    let y_true_ml = Array2::from_shape_vec((4, 2), vec![1, 0, 0, 1, 1, 1, 0, 0]).unwrap();

    let y_pred_ml = Array2::from_shape_vec(
        (4, 2),
        vec![
            1, 1, // FP for label 1
            0, 1, // FN for label 0
            1, 0, // FN for label 1
            0, 0,
        ],
    )
    .unwrap();

    // Equal weights
    let equal_weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let cm_equal =
        multilabel_confusion_matrix_sklearn(&y_true_ml, &y_pred_ml, Some(&equal_weights), None)?;

    // Different weights
    let diff_weights = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0]);
    let cm_weighted =
        multilabel_confusion_matrix_sklearn(&y_true_ml, &y_pred_ml, Some(&diff_weights), None)?;

    println!(
        "    Equal weights confusion matrix shape: {:?}",
        cm_equal.shape()
    );
    println!(
        "    Weighted confusion matrix shape: {:?}",
        cm_weighted.shape()
    );
    println!("    (Weights affect the individual cell counts in the confusion matrices)");

    Ok(())
}
