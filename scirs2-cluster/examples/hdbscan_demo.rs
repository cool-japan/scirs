use ndarray::Array2;
use scirs2_cluster::{hdbscan, ClusterSelectionMethod, HDBSCANOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HDBSCAN Clustering Example");
    println!("=========================\n");

    // Create a dataset with clusters of varying densities
    let data = Array2::from_shape_vec(
        (20, 2),
        vec![
            // Dense cluster 1
            1.0, 1.0, 1.1, 1.1, 1.2, 0.9, 0.9, 1.2, 1.0, 0.8, // Dense cluster 2
            5.0, 5.0, 5.1, 5.1, 5.2, 4.9, 4.9, 5.2, 5.0, 4.8, // Sparse cluster 3
            10.0, 10.0, 10.5, 10.5, 11.0, 10.0, 10.0, 11.0, 11.5, 11.5, // Noise points
            3.0, 8.0, 7.0, 2.0, 8.0, 7.0, 2.0, 3.0, 15.0, 15.0,
        ],
    )?;

    println!("Running HDBSCAN with default parameters...");

    // Run HDBSCAN with default parameters
    let result = hdbscan(data.view(), None)?;

    println!("\nCluster labels:");
    for (i, &label) in result.labels.iter().enumerate() {
        println!(
            "Point {} ({:.1}, {:.1}): Cluster {}",
            i,
            data[[i, 0]],
            data[[i, 1]],
            if label == -1 {
                "Noise".to_string()
            } else {
                label.to_string()
            },
        );
    }

    // Count points in each cluster
    let mut cluster_counts = std::collections::HashMap::new();
    for &label in result.labels.iter() {
        *cluster_counts.entry(label).or_insert(0) += 1;
    }

    println!("\nCluster statistics:");
    for (label, count) in cluster_counts.iter() {
        if *label == -1 {
            println!("Noise points: {}", count);
        } else {
            println!("Cluster {}: {} points", label, count);
        }
    }

    // Run with different parameters
    println!("\nRunning HDBSCAN with custom parameters...");
    let options = HDBSCANOptions {
        min_cluster_size: 3,
        min_samples: Some(2),
        cluster_selection_method: ClusterSelectionMethod::Leaf,
        ..Default::default()
    };

    let result2 = hdbscan(data.view(), Some(options))?;

    // Count points in each cluster with new parameters
    let mut cluster_counts2 = std::collections::HashMap::new();
    for &label in result2.labels.iter() {
        *cluster_counts2.entry(label).or_insert(0) += 1;
    }

    println!("\nCluster statistics with custom parameters:");
    for (label, count) in cluster_counts2.iter() {
        if *label == -1 {
            println!("Noise points: {}", count);
        } else {
            println!("Cluster {}: {} points", label, count);
        }
    }

    Ok(())
}
