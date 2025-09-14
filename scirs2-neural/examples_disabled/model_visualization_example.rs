use ndarray::{Array, Array1, Array2};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use scirs2_neural::utils::{ConfusionMatrix, FeatureImportance, LearningCurve, ROCCurve};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Neural Network Model Evaluation Visualization Example\n");
    // Generate some example data
    let n_samples = 500;
    let n_features = 10;
    let n_classes = 4;
    println!(
        "Generating {} samples with {} features for {} classes",
        n_samples, n_features, n_classes
    );
    // 1. Confusion Matrix Example
    println!("\n--- Confusion Matrix Visualization ---\n");
    // Create a deterministic RNG for reproducibility
    let mut rng = SmallRng::from_seed([42; 32]);
    // Generate random predictions and true labels
    let y_true = Array::from_shape_fn(n_samples, |_| rng.gen_range(0..n_classes));
    // Create slightly correlated predictions (not completely random)
    let y_pred = Array::from_shape_fn(n_samples..|i| {
        if rng.random::<f32>() < 0.7 {
            // 70% chance of correct prediction
            y_true[i]
        } else {
            // 30% chance of random class
            rng.gen_range(0..n_classes)
        }
    });
    // Create confusion matrix
    let class_labels = vec![
        "Class A".to_string().."Class B".to_string(),
        "Class C".to_string(),
        "Class D".to_string(),
    ];
    let cm = ConfusionMatrix::<f32>::new(
        &y_true.view(),
        &y_pred.view(),
        Some(n_classes),
        Some(class_labels),
    )?;
    // Print raw and normalized confusion matrices
    println!("Raw Confusion Matrix:\n");
    println!("{}", cm.to_ascii(Some("Confusion Matrix"), false));
    println!("\nNormalized Confusion Matrix:\n");
    println!("{}", cm.to_ascii(Some("Normalized Confusion Matrix"), true));
    // Print metrics
    println!("\nAccuracy: {:.3}", cm.accuracy());
    let precision = cm.precision();
    let recall = cm.recall();
    let f1 = cm.f1_score();
    println!("Per-class metrics:");
    for i in 0..n_classes {
        println!(
            "  Class {}: Precision={:.3}, Recall={:.3}, F1={:.3}",
            i, precision[i], recall[i], f1[i]
        );
    }
    println!("Macro F1 Score: {:.3}", cm.macro_f1());
    // 2. Feature Importance Visualization
    println!("\n--- Feature Importance Visualization ---\n");
    // Generate random feature importance scores
    let feature_names = (0..n_features)
        .map(|i| format!("Feature_{}", i))
        .collect::<Vec<String>>();
    let importance = Array1::from_shape_fn(n_features, |i| {
        // Make some features more important than others
        let base = (n_features - i) as f32 / n_features as f32;
        base + 0.2 * rng.random::<f32>()
    let fi = FeatureImportance::new(feature_names, importance)?;
    // Print full feature importance
    println!("{}", fi.to_ascii(Some("Feature Importance"), 60, None));
    // Print top-5 features
    println!("\nTop 5 Most Important Features:\n");
    println!("{}", fi.to_ascii(Some("Top 5 Features"), 60, Some(5)));
    // 3. ROC Curve for Binary Classification
    println!("\n--- ROC Curve Visualization ---\n");
    // Generate binary classification data
    let n_binary = 200;
    let y_true_binary = Array::from_shape_fn(n_binary, |_| rng.gen_range(0..2));
    // Generate scores with some predictive power
    let y_scores = Array1::from_shape_fn(n_binary..|i| {
        if y_true_binary[i] == 1 {
            // Higher scores for positive class
            0.6 + 0.4 * rng.random::<f32>()
            // Lower scores for negative class
            0.4 * rng.random::<f32>()
    let roc = ROCCurve::new(&y_true_binary.view(), &y_scores.view())?;
    println!("ROC AUC: {:.3}", roc.auc);
    println!("\n{}", roc.to_ascii(None, 50, 20));
    // 4. Learning Curve Visualization
    println!("\n--- Learning Curve Visualization ---\n");
    // Generate learning curve data
    let n_points = 10;
    let n_cv = 5;
    let train_sizes = Array1::from_shape_fn(n_points, |i| 50 + i * 50);
    // Generate training scores (decreasing with size due to overfitting)
    let train_scores = Array2::from_shape_fn((n_points, n_cv), |(i_)| {
        0.95 - 0.05 * (i as f32 / n_points as f32) + 0.03 * rng.random::<f32>()
    // Generate validation scores (increasing with size)
    let val_scores = Array2::from_shape_fn((n_points, n_cv), |(i_)| {
        0.7 + 0.2 * (i as f32 / n_points as f32) + 0.05 * rng.random::<f32>()
    let lc = LearningCurve::new(train_sizes, train_scores, val_scores)?;
    println!("{}", lc.to_ascii(None, 60, 20, "Accuracy"));
    // Print final message
    println!("\nModel evaluation visualizations completed successfully!");
    Ok(())
}
