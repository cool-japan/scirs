//! Sampling and bootstrapping utilities demonstration
//!
//! This example demonstrates the use of random sampling and stratified sampling
//! utilities provided by scirs2-datasets.

use ndarray::Array1;
use scirs2_datasets::{load_iris, random_sample, stratified_sample, Dataset};

fn main() {
    println!("=== Sampling and Bootstrapping Demonstration ===\n");

    // Load the Iris dataset for demonstration
    let iris = load_iris().unwrap();
    let n_samples = iris.n_samples();

    println!("Original Iris dataset:");
    println!("- Samples: {}", n_samples);
    println!("- Features: {}", iris.n_features());

    if let Some(target) = &iris.target {
        let class_counts = count_classes(target);
        println!("- Class distribution: {:?}\n", class_counts);
    }

    // Demonstrate random sampling without replacement
    println!("=== Random Sampling (without replacement) ===");
    let sample_size = 30;
    let random_indices = random_sample(n_samples, sample_size, false, Some(42)).unwrap();

    println!(
        "Sampled {} indices from {} total samples",
        sample_size, n_samples
    );
    println!(
        "Sample indices: {:?}",
        &random_indices[..10.min(random_indices.len())]
    );

    // Create a subset dataset
    let sample_data = iris.data.select(ndarray::Axis(0), &random_indices);
    let sample_target = iris
        .target
        .as_ref()
        .map(|t| t.select(ndarray::Axis(0), &random_indices));
    let sample_dataset = Dataset::new(sample_data, sample_target)
        .with_description("Random sample from Iris dataset".to_string());

    println!(
        "Random sample dataset: {} samples, {} features",
        sample_dataset.n_samples(),
        sample_dataset.n_features()
    );

    if let Some(target) = &sample_dataset.target {
        let sample_class_counts = count_classes(target);
        println!("Sample class distribution: {:?}\n", sample_class_counts);
    }

    // Demonstrate bootstrap sampling (with replacement)
    println!("=== Bootstrap Sampling (with replacement) ===");
    let bootstrap_size = 200; // More than original dataset size
    let bootstrap_indices = random_sample(n_samples, bootstrap_size, true, Some(42)).unwrap();

    println!(
        "Bootstrap sampled {} indices from {} total samples",
        bootstrap_size, n_samples
    );
    println!(
        "Bootstrap may have duplicates - first 10 indices: {:?}",
        &bootstrap_indices[..10]
    );

    // Count frequency of each index in bootstrap sample
    let mut index_counts = vec![0; n_samples];
    for &idx in &bootstrap_indices {
        index_counts[idx] += 1;
    }
    let max_count = *index_counts.iter().max().unwrap();
    let zero_count = index_counts.iter().filter(|&&count| count == 0).count();

    println!("Bootstrap statistics:");
    println!("- Maximum frequency of any sample: {}", max_count);
    println!(
        "- Number of original samples not selected: {}\n",
        zero_count
    );

    // Demonstrate stratified sampling
    println!("=== Stratified Sampling ===");
    if let Some(target) = &iris.target {
        let stratified_size = 30;
        let stratified_indices = stratified_sample(target, stratified_size, Some(42)).unwrap();

        println!(
            "Stratified sampled {} indices maintaining class proportions",
            stratified_size
        );

        // Create stratified subset
        let stratified_data = iris.data.select(ndarray::Axis(0), &stratified_indices);
        let stratified_target = target.select(ndarray::Axis(0), &stratified_indices);
        let stratified_dataset = Dataset::new(stratified_data, Some(stratified_target))
            .with_description("Stratified sample from Iris dataset".to_string());

        println!(
            "Stratified sample dataset: {} samples, {} features",
            stratified_dataset.n_samples(),
            stratified_dataset.n_features()
        );

        let stratified_class_counts = count_classes(&stratified_dataset.target.unwrap());
        println!(
            "Stratified sample class distribution: {:?}",
            stratified_class_counts
        );

        // Verify proportions are maintained
        let original_proportions = calculate_proportions(&count_classes(target));
        let stratified_proportions = calculate_proportions(&stratified_class_counts);

        println!("Class proportion comparison:");
        for (&class, &original_prop) in &original_proportions {
            let stratified_prop = stratified_proportions.get(&class).unwrap_or(&0.0);
            println!(
                "  Class {}: Original {:.2}%, Stratified {:.2}%",
                class,
                original_prop * 100.0,
                stratified_prop * 100.0
            );
        }
    }

    // Demonstrate practical use case: creating training/validation splits
    println!("\n=== Practical Example: Multiple Train/Validation Splits ===");
    for i in 1..=3 {
        let split_indices = random_sample(n_samples, 100, false, Some(42 + i)).unwrap();
        let (train_indices, val_indices) = split_indices.split_at(80);

        println!(
            "Split {}: {} training samples, {} validation samples",
            i,
            train_indices.len(),
            val_indices.len()
        );
    }

    println!("\n=== Sampling Demo Complete ===");
}

/// Count the number of samples in each class
fn count_classes(targets: &Array1<f64>) -> std::collections::HashMap<i64, usize> {
    let mut counts = std::collections::HashMap::new();
    for &target in targets.iter() {
        let class = target.round() as i64;
        *counts.entry(class).or_insert(0) += 1;
    }
    counts
}

/// Calculate class proportions
fn calculate_proportions(
    counts: &std::collections::HashMap<i64, usize>,
) -> std::collections::HashMap<i64, f64> {
    let total: usize = counts.values().sum();
    counts
        .iter()
        .map(|(&class, &count)| (class, count as f64 / total as f64))
        .collect()
}
