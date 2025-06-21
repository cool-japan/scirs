//! Data balancing utilities demonstration
//!
//! This example demonstrates the use of data balancing utilities for handling
//! imbalanced datasets in machine learning applications.

use ndarray::{Array1, Array2};
use scirs2_datasets::{
    create_balanced_dataset, generate_synthetic_samples, load_iris, random_oversample,
    random_undersample, BalancingStrategy,
};

fn main() {
    println!("=== Data Balancing Utilities Demonstration ===\n");

    // Create an artificially imbalanced dataset for demonstration
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            // Class 0 (minority): 2 samples
            1.0, 1.0, 1.2, 1.1, // Class 1 (majority): 6 samples
            5.0, 5.0, 5.1, 5.2, 4.9, 4.8, 5.3, 5.1, 4.8, 5.3, 5.0, 4.9,
            // Class 2 (moderate): 2 samples
            10.0, 10.0, 10.1, 9.9,
        ],
    )
    .unwrap();

    let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);

    println!("Original imbalanced dataset:");
    print_class_distribution(&targets);
    println!("Total samples: {}\n", data.nrows());

    // Demonstrate random oversampling
    println!("=== Random Oversampling =======================");
    let (oversampled_data, oversampled_targets) =
        random_oversample(&data, &targets, Some(42)).unwrap();

    println!("After random oversampling:");
    print_class_distribution(&oversampled_targets);
    println!("Total samples: {}\n", oversampled_data.nrows());

    // Demonstrate random undersampling
    println!("=== Random Undersampling ======================");
    let (undersampled_data, undersampled_targets) =
        random_undersample(&data, &targets, Some(42)).unwrap();

    println!("After random undersampling:");
    print_class_distribution(&undersampled_targets);
    println!("Total samples: {}\n", undersampled_data.nrows());

    // Demonstrate SMOTE-like synthetic sample generation
    println!("=== Synthetic Sample Generation (SMOTE-like) ==");

    // Generate 4 synthetic samples for class 0 (minority class)
    let (synthetic_data, synthetic_targets) =
        generate_synthetic_samples(&data, &targets, 0.0, 4, 1, Some(42)).unwrap();

    println!(
        "Generated {} synthetic samples for class 0",
        synthetic_data.nrows()
    );
    println!("Synthetic samples (first 3 features of each):");
    for i in 0..synthetic_data.nrows() {
        println!(
            "  Sample {}: [{:.3}, {:.3}] -> class {}",
            i,
            synthetic_data[[i, 0]],
            synthetic_data[[i, 1]],
            synthetic_targets[i]
        );
    }
    println!();

    // Demonstrate unified balancing strategies
    println!("=== Unified Balancing Strategies ==============");

    // Strategy 1: Random Oversampling
    let (balanced_over, targets_over) = create_balanced_dataset(
        &data,
        &targets,
        BalancingStrategy::RandomOversample,
        Some(42),
    )
    .unwrap();

    println!("Strategy: Random Oversampling");
    print_class_distribution(&targets_over);
    println!("Total samples: {}", balanced_over.nrows());

    // Strategy 2: Random Undersampling
    let (balanced_under, targets_under) = create_balanced_dataset(
        &data,
        &targets,
        BalancingStrategy::RandomUndersample,
        Some(42),
    )
    .unwrap();

    println!("\nStrategy: Random Undersampling");
    print_class_distribution(&targets_under);
    println!("Total samples: {}", balanced_under.nrows());

    // Strategy 3: SMOTE with k=1 neighbors
    let (balanced_smote, targets_smote) = create_balanced_dataset(
        &data,
        &targets,
        BalancingStrategy::SMOTE { k_neighbors: 1 },
        Some(42),
    )
    .unwrap();

    println!("\nStrategy: SMOTE (k_neighbors=1)");
    print_class_distribution(&targets_smote);
    println!("Total samples: {}", balanced_smote.nrows());

    // Demonstrate with real-world dataset
    println!("\n=== Real-world Example: Iris Dataset ==========");

    let iris = load_iris().unwrap();
    if let Some(iris_targets) = &iris.target {
        println!("Original Iris dataset:");
        print_class_distribution(iris_targets);

        // Apply oversampling to iris (it's already balanced, but for demonstration)
        let (iris_balanced, iris_balanced_targets) =
            random_oversample(&iris.data, iris_targets, Some(42)).unwrap();

        println!("\nIris after oversampling (should remain the same):");
        print_class_distribution(&iris_balanced_targets);
        println!("Total samples: {}", iris_balanced.nrows());

        // Create artificial imbalance by removing some samples
        let indices_to_keep: Vec<usize> = (0..150)
            .filter(|&i| {
                let class = iris_targets[i].round() as i64;
                // Keep all of class 0, 30 of class 1, 10 of class 2
                match class {
                    0 => true,    // Keep all 50
                    1 => i < 80,  // Keep first 30 (indices 50-79)
                    2 => i < 110, // Keep first 10 (indices 100-109)
                    _ => false,
                }
            })
            .collect();

        let imbalanced_data = iris.data.select(ndarray::Axis(0), &indices_to_keep);
        let imbalanced_targets = iris_targets.select(ndarray::Axis(0), &indices_to_keep);

        println!("\nArtificially imbalanced Iris:");
        print_class_distribution(&imbalanced_targets);

        // Balance it using SMOTE
        let (rebalanced_data, rebalanced_targets) = create_balanced_dataset(
            &imbalanced_data,
            &imbalanced_targets,
            BalancingStrategy::SMOTE { k_neighbors: 3 },
            Some(42),
        )
        .unwrap();

        println!("\nAfter SMOTE rebalancing:");
        print_class_distribution(&rebalanced_targets);
        println!("Total samples: {}", rebalanced_data.nrows());
    }

    println!("\n=== Performance Comparison ====================");

    // Show the tradeoffs between different strategies
    println!("Strategy Comparison Summary:");
    println!("┌─────────────────────┬──────────────┬─────────────────────────────────┐");
    println!("│ Strategy            │ Final Size   │ Characteristics                 │");
    println!("├─────────────────────┼──────────────┼─────────────────────────────────┤");
    println!(
        "│ Random Oversample   │ {} samples   │ Increases data size, duplicates │",
        balanced_over.nrows()
    );
    println!(
        "│ Random Undersample  │ {} samples    │ Reduces data size, loses info   │",
        balanced_under.nrows()
    );
    println!(
        "│ SMOTE               │ {} samples   │ Increases size, synthetic data  │",
        balanced_smote.nrows()
    );
    println!("└─────────────────────┴──────────────┴─────────────────────────────────┘");

    println!("\n=== Balancing Demo Complete ====================");
}

/// Print the class distribution of targets
fn print_class_distribution(targets: &Array1<f64>) {
    let mut class_counts = std::collections::HashMap::new();
    for &target in targets.iter() {
        let class = target.round() as i64;
        *class_counts.entry(class).or_insert(0) += 1;
    }

    let mut classes: Vec<_> = class_counts.keys().cloned().collect();
    classes.sort();

    print!("Class distribution: ");
    for (i, &class) in classes.iter().enumerate() {
        let count = class_counts[&class];
        if i > 0 {
            print!(", ");
        }
        print!("Class {} ({} samples)", class, count);
    }
    println!();
}
