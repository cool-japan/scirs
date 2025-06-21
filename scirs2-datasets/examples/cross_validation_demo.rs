//! Cross-validation utilities demonstration
//!
//! This example demonstrates the use of K-fold, stratified K-fold, and time series
//! cross-validation utilities provided by scirs2-datasets.

use ndarray::{Array1, Array2};
use scirs2_datasets::{k_fold_split, stratified_k_fold_split, time_series_split, Dataset};

fn main() {
    println!("=== Cross-Validation Demonstration ===\n");

    // Create sample dataset
    let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64 / 10.0).collect()).unwrap();
    let target = Array1::from(
        (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>(),
    );

    let dataset = Dataset::new(data.clone(), Some(target.clone()))
        .with_description("Sample dataset for cross-validation demo".to_string());

    println!("Dataset info:");
    println!("- Samples: {}", dataset.n_samples());
    println!("- Features: {}", dataset.n_features());
    println!("- Description: {}\n", dataset.description.as_ref().unwrap());

    // Demonstrate K-fold cross-validation
    println!("=== K-Fold Cross-Validation (k=5) ===");
    let k_folds = k_fold_split(dataset.n_samples(), 5, true, Some(42)).unwrap();

    for (i, (train_indices, val_indices)) in k_folds.iter().enumerate() {
        println!(
            "Fold {}: Train size: {}, Validation size: {}",
            i + 1,
            train_indices.len(),
            val_indices.len()
        );
        println!(
            "  Train indices: {:?}",
            &train_indices[..5.min(train_indices.len())]
        );
        println!("  Val indices: {:?}", val_indices);
    }
    println!();

    // Demonstrate Stratified K-fold cross-validation
    println!("=== Stratified K-Fold Cross-Validation (k=4) ===");
    let stratified_folds = stratified_k_fold_split(&target, 4, true, Some(42)).unwrap();

    for (i, (train_indices, val_indices)) in stratified_folds.iter().enumerate() {
        // Calculate class distribution in validation set
        let val_targets: Vec<f64> = val_indices.iter().map(|&idx| target[idx]).collect();
        let class_0_count = val_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = val_targets.iter().filter(|&&x| x == 1.0).count();

        println!(
            "Fold {}: Train size: {}, Validation size: {}",
            i + 1,
            train_indices.len(),
            val_indices.len()
        );
        println!(
            "  Class distribution in validation: Class 0: {}, Class 1: {}",
            class_0_count, class_1_count
        );
    }
    println!();

    // Demonstrate Time Series cross-validation
    println!("=== Time Series Cross-Validation ===");
    let ts_folds = time_series_split(dataset.n_samples(), 3, 3, 1).unwrap();

    for (i, (train_indices, val_indices)) in ts_folds.iter().enumerate() {
        println!(
            "Split {}: Train size: {}, Test size: {}",
            i + 1,
            train_indices.len(),
            val_indices.len()
        );
        println!(
            "  Train range: {} to {}",
            train_indices.first().unwrap_or(&0),
            train_indices.last().unwrap_or(&0)
        );
        println!(
            "  Test range: {} to {}",
            val_indices.first().unwrap_or(&0),
            val_indices.last().unwrap_or(&0)
        );
    }
    println!();

    // Demonstrate usage with Dataset methods
    println!("=== Using Cross-Validation with Dataset ===");
    let first_fold = &k_folds[0];
    let (train_indices, val_indices) = first_fold;

    // Create training subset
    let train_data = data.select(ndarray::Axis(0), train_indices);
    let train_target = target.select(ndarray::Axis(0), train_indices);
    let train_dataset = Dataset::new(train_data, Some(train_target))
        .with_description("Training fold from K-fold CV".to_string());

    // Create validation subset
    let val_data = data.select(ndarray::Axis(0), val_indices);
    let val_target = target.select(ndarray::Axis(0), val_indices);
    let val_dataset = Dataset::new(val_data, Some(val_target))
        .with_description("Validation fold from K-fold CV".to_string());

    println!(
        "Training dataset: {} samples, {} features",
        train_dataset.n_samples(),
        train_dataset.n_features()
    );
    println!(
        "Validation dataset: {} samples, {} features",
        val_dataset.n_samples(),
        val_dataset.n_features()
    );

    println!("\n=== Cross-Validation Demo Complete ===");
}
