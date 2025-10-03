use scirs2_cluster::density::{dbscan, labels, DistanceMetric};
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() {
    println!("DBSCAN Clustering Example");
    println!("========================");

    // Create a dataset with clusters of different shapes and noise points
    let data = Array2::from_shape_vec(
        (16, 2),
        vec![
            // Cluster 1 - dense cluster
            1.0, 1.0, 1.3, 1.2, 1.1, 1.5, 1.5, 1.3, 1.7, 1.7,
            // Cluster 2 - another dense cluster
            5.0, 5.0, 5.2, 5.3, 5.1, 4.7, 4.8, 5.2, 5.3, 4.9,
            // Cluster 3 - elongated cluster
            8.0, 1.0, 8.5, 1.1, 9.0, 1.2, 9.5, 1.3, 10.0, 1.4, // Noise point
            3.0, 7.0,
        ],
    )
    .unwrap();

    // Print the dataset
    println!("\nDataset:");
    for i in 0..data.shape()[0] {
        println!("  Point {}: ({}, {})", i, data[[i, 0]], data[[i, 1]]);
    }

    // Parameters to experiment with
    let eps_values = [0.5, 0.8, 1.2];
    let min_samples_values = [2, 3, 4];

    // Run DBSCAN with different parameters
    println!("\nRunning DBSCAN with different parameters:");

    for &eps in &eps_values {
        for &min_samples in &min_samples_values {
            println!("\neps={eps}, min_samples={min_samples}");

            let labels_array = dbscan(
                data.view(),
                eps,
                min_samples,
                Some(DistanceMetric::Euclidean),
            )
            .unwrap();

            // Count number of clusters (excluding noise)
            let mut unique_labels = std::collections::HashSet::new();
            for &label in labels_array.iter() {
                if label != labels::NOISE {
                    unique_labels.insert(label);
                }
            }
            let num_clusters = unique_labels.len();

            // Count noise points
            let noise_count = labels_array
                .iter()
                .filter(|&&label| label == labels::NOISE)
                .count();

            println!(
                "  Found {} clusters and {} noise points",
                num_clusters, noise_count
            );

            // Print cluster assignments
            println!("  Cluster assignments:");
            for i in 0..data.shape()[0] {
                let label_str = if labels_array[i] == labels::NOISE {
                    "NOISE".to_string()
                } else {
                    format!("{}", labels_array[i])
                };

                println!(
                    "    Point {}: ({}, {}) -> Cluster {}",
                    i,
                    data[[i, 0]],
                    data[[i, 1]],
                    label_str
                );
            }

            // Print cluster stats
            println!("  Cluster sizes:");
            if num_clusters > 0 {
                let mut cluster_counts = vec![0; num_clusters];
                for &label in labels_array.iter() {
                    if label != labels::NOISE {
                        cluster_counts[label as usize] += 1;
                    }
                }

                for (i, &count) in cluster_counts.iter().enumerate() {
                    println!("    Cluster {i}: {count} points");
                }
            }

            if noise_count > 0 {
                println!("    Noise: {noise_count} points");
            }
        }
    }

    // Recommended parameters
    println!("\nRecommended parameters for this dataset:");
    println!("  eps=0.8, min_samples=3");

    let best_labels = dbscan(data.view(), 0.8, 3, Some(DistanceMetric::Euclidean)).unwrap();

    // Print final clustering
    println!("\nFinal clustering result:");
    for (i, &label) in best_labels.iter().enumerate() {
        let label_str = if label == labels::NOISE {
            "NOISE".to_string()
        } else {
            format!("{}", label)
        };

        println!(
            "  Point {}: ({}, {}) -> {}",
            i,
            data[[i, 0]],
            data[[i, 1]],
            label_str
        );
    }
}
