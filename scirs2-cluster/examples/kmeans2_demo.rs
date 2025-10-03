//! Demonstration of enhanced K-means with multiple initialization methods
//!
//! This example shows how to use kmeans2 with different initialization strategies.

use scirs2_cluster::vq::{kmeans2, whiten, MinitMethod, MissingMethod};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

#[allow(dead_code)]
fn generate_blobs(
    n_samples: usize,
    centers: &Array2<f64>,
    cluster_std: f64,
) -> (Array2<f64>, Array1<i32>) {
    let mut rng = thread_rng();
    let n_features = centers.shape()[1];
    let n_clusters = centers.shape()[0];
    let samples_per_cluster = n_samples / n_clusters;

    let mut data = Array2::<f64>::zeros((n_samples, n_features));
    let mut labels = Array1::<i32>::zeros(n_samples);

    for i in 0..n_clusters {
        let start_idx = i * samples_per_cluster;
        let end_idx = if i == n_clusters - 1 {
            n_samples
        } else {
            (i + 1) * samples_per_cluster
        };

        for idx in start_idx..end_idx {
            for j in 0..n_features {
                let center = centers[[i, j]];
                data[[idx, j]] = center + cluster_std * (rng.random::<f64>() - 0.5) * 2.0;
            }
            labels[idx] = i as i32;
        }
    }

    (data, labels)
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Enhanced K-means Clustering Demo");
    println!("===============================\n");

    // Generate three clusters
    let centers = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.0, 0.0, // Cluster 1
            5.0, 5.0, // Cluster 2
            10.0, 0.0, // Cluster 3
        ],
    )?;

    let (data, true_labels) = generate_blobs(300, &centers, 1.5);

    // Whiten the data
    let whitened_data = whiten(&data)?;

    // Compare different initialization methods
    let init_methods = vec![
        ("Random", MinitMethod::Random),
        ("Points", MinitMethod::Points),
        ("K-means++", MinitMethod::PlusPlus),
    ];

    for (method_name, init_method) in init_methods {
        println!("Testing {} initialization:", method_name);

        let (centroids, predicted_labels) = kmeans2(
            whitened_data.view(),
            3,
            Some(10),                  // max iterations
            Some(1e-4),                // convergence threshold
            Some(init_method),         // initialization method
            Some(MissingMethod::Warn), // handle empty clusters by warning
            Some(true),                // check finite values
            Some(42),                  // random seed for reproducibility
        )?;

        // Calculate accuracy (simple measure)
        let mut correct = 0;
        for i in 0..true_labels.len() {
            // Simple accuracy assuming cluster ordering matches
            if true_labels[i] == predicted_labels[i] as i32 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / true_labels.len() as f64;

        println!("  Centroids shape: {:?}", centroids.shape());
        println!("  Accuracy (naive): {:.3}", accuracy);
        println!();
    }

    // Demonstrate error handling for empty clusters
    println!("Testing empty cluster handling:");

    // Create a dataset where one cluster might become empty
    let sparse_data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
        ],
    )?;

    // Try to create 3 clusters from 2 natural groups - might create empty cluster
    match kmeans2(
        sparse_data.view(),
        3,
        Some(10),
        Some(1e-4),
        Some(MinitMethod::Random),
        Some(MissingMethod::Raise), // This will raise an error on empty clusters
        Some(true),
        Some(12345),
    ) {
        Ok((centroids, labels)) => {
            println!("  Successfully created 3 clusters");
            println!("  Centroids: {:?}", centroids);
            println!("  Labels: {:?}", labels);
        }
        Err(e) => {
            println!("  Error as expected: {}", e);
        }
    }

    Ok(())
}
