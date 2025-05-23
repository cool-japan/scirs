use ndarray::Array2;
use scirs2_cluster::birch::{birch, BirchOptions};
use scirs2_cluster::metrics::silhouette_score;

fn main() {
    println!("BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) Demo");
    println!("{}", "=".repeat(70));

    // Generate synthetic data with multiple clusters
    let data = generate_clustered_data();
    println!("Generated {} data points in 2D space", data.shape()[0]);

    // Test BIRCH with different parameters
    let thresholds = vec![0.5, 1.0, 1.5, 2.0];
    let n_clusters_options = vec![Some(2), Some(3), Some(4), None];

    for threshold in thresholds {
        println!("\nTesting with threshold = {}", threshold);

        for n_clusters in &n_clusters_options {
            let options = BirchOptions {
                threshold,
                n_clusters: *n_clusters,
                branching_factor: 50,
            };

            match birch(data.view(), options) {
                Ok((centroids, labels)) => {
                    let actual_clusters = centroids.shape()[0];
                    print!(
                        "  n_clusters={:?} => {} clusters found",
                        n_clusters
                            .map(|n| n.to_string())
                            .unwrap_or("auto".to_string()),
                        actual_clusters
                    );

                    // Calculate silhouette score if we have more than one cluster
                    let unique_labels: std::collections::HashSet<_> =
                        labels.iter().cloned().collect();

                    if unique_labels.len() > 1 {
                        match silhouette_score(data.view(), labels.view()) {
                            Ok(score) => println!(", silhouette score: {:.3}", score),
                            Err(_) => println!(", silhouette score: N/A"),
                        }
                    } else {
                        println!(", silhouette score: N/A (single cluster)");
                    }
                }
                Err(e) => {
                    println!(
                        "  n_clusters={:?} => Error: {}",
                        n_clusters
                            .map(|n| n.to_string())
                            .unwrap_or("auto".to_string()),
                        e
                    );
                }
            }
        }
    }

    // Demonstrate incremental nature of BIRCH
    println!("\n\nIncremental clustering demonstration:");
    println!("{}", "-".repeat(40));

    let options = BirchOptions {
        threshold: 1.0,
        n_clusters: Some(3),
        ..Default::default()
    };

    // Process data in batches
    let batch_size = 50;
    let n_batches = data.shape()[0].div_ceil(batch_size);

    println!("Processing {} batches of size {}", n_batches, batch_size);

    // Note: The simplified BIRCH implementation doesn't support true incremental updates
    // In a full implementation, we would update the CF-tree incrementally
    let result = birch(data.view(), options);

    match result {
        Ok((centroids, labels)) => {
            println!("Final clustering complete:");
            println!("  Number of clusters: {}", centroids.shape()[0]);

            // Print cluster sizes
            let mut cluster_sizes = vec![0; centroids.shape()[0]];
            for &label in labels.iter() {
                if label >= 0 && (label as usize) < cluster_sizes.len() {
                    cluster_sizes[label as usize] += 1;
                }
            }

            println!("  Cluster sizes:");
            for (i, &size) in cluster_sizes.iter().enumerate() {
                println!("    Cluster {}: {} points", i, size);
            }

            println!("\n  Cluster centroids:");
            for (i, centroid) in centroids.outer_iter().enumerate() {
                println!(
                    "    Cluster {}: [{:.2}, {:.2}]",
                    i, centroid[0], centroid[1]
                );
            }
        }
        Err(e) => {
            println!("Error in final clustering: {}", e);
        }
    }
}

fn generate_clustered_data() -> Array2<f64> {
    let mut data = Vec::new();

    // Cluster 1: centered at (2, 2)
    for i in 0..50 {
        let x = 2.0 + ((i % 10) as f64 - 5.0) * 0.2;
        let y = 2.0 + ((i / 10) as f64 - 2.5) * 0.2;
        data.push(x);
        data.push(y);
    }

    // Cluster 2: centered at (6, 2)
    for i in 0..40 {
        let x = 6.0 + ((i % 8) as f64 - 4.0) * 0.15;
        let y = 2.0 + ((i / 8) as f64 - 2.5) * 0.15;
        data.push(x);
        data.push(y);
    }

    // Cluster 3: centered at (4, 6)
    for i in 0..30 {
        let x = 4.0 + ((i % 6) as f64 - 3.0) * 0.2;
        let y = 6.0 + ((i / 6) as f64 - 2.5) * 0.2;
        data.push(x);
        data.push(y);
    }

    // Some outliers
    data.extend_from_slice(&[0.0, 0.0, 8.0, 8.0, 0.0, 8.0, 8.0, 0.0]);

    Array2::from_shape_vec((124, 2), data).unwrap()
}
