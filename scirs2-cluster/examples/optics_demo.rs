use ndarray::{Array1, Array2};
use rand::distr::Uniform;
use rand::prelude::*;
use scirs2_cluster::density::optics::{extract_dbscan_clustering, extract_xi_clusters, optics};
use scirs2_cluster::density::DistanceMetric;

fn main() {
    // Generate synthetic data with clusters of varying density
    let data = generate_data();

    println!("Running OPTICS on {} data points...", data.shape()[0]);

    // Run OPTICS algorithm
    let result = optics(data.view(), 5, None, Some(DistanceMetric::Euclidean)).unwrap();

    println!("OPTICS ordering complete.");
    println!(
        "First 10 points in ordering: {:?}",
        &result.ordering[..10.min(result.ordering.len())]
    );
    println!(
        "First 10 reachability distances: {:?}",
        &result.reachability[..10.min(result.reachability.len())]
    );

    // Extract clusters using different methods
    println!("\nExtracting clusters using epsilon = 0.5 (DBSCAN-like clustering):");
    let dbscan_labels = extract_dbscan_clustering(&result, 0.5);
    print_cluster_stats(&dbscan_labels);

    println!("\nExtracting clusters using xi = 0.05 (steepness-based clustering):");
    match extract_xi_clusters(&result, 0.05, 5) {
        Ok(xi_labels) => {
            print_cluster_stats(&xi_labels);
        }
        Err(e) => {
            println!("Error extracting xi clusters: {}", e);
        }
    }
}

fn generate_data() -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate three clusters with different densities

    // Dense cluster (30 points)
    let x_dist1 = Uniform::new(-0.3, 0.3).unwrap();
    let y_dist1 = Uniform::new(-0.3, 0.3).unwrap();
    let mut data = Vec::with_capacity(100);

    for _ in 0..30 {
        let x = rng.sample(x_dist1) + 1.0;
        let y = rng.sample(y_dist1) + 1.0;
        data.push(x);
        data.push(y);
    }

    // Medium density cluster (20 points)
    let x_dist2 = Uniform::new(-0.5, 0.5).unwrap();
    let y_dist2 = Uniform::new(-0.5, 0.5).unwrap();
    for _ in 0..20 {
        let x = rng.sample(x_dist2) + 4.0;
        let y = rng.sample(y_dist2) + 1.0;
        data.push(x);
        data.push(y);
    }

    // Sparse cluster (15 points)
    let x_dist3 = Uniform::new(-1.0, 1.0).unwrap();
    let y_dist3 = Uniform::new(-1.0, 1.0).unwrap();
    for _ in 0..15 {
        let x = rng.sample(x_dist3) + 2.5;
        let y = rng.sample(y_dist3) + 5.0;
        data.push(x);
        data.push(y);
    }

    // Add some noise (10 points)
    let x_noise = Uniform::new(-1.0, 7.0).unwrap();
    let y_noise = Uniform::new(-1.0, 7.0).unwrap();
    for _ in 0..10 {
        let x = rng.sample(x_noise);
        let y = rng.sample(y_noise);
        data.push(x);
        data.push(y);
    }

    // Convert to ndarray
    Array2::from_shape_vec((75, 2), data).unwrap()
}

fn print_cluster_stats(labels: &Array1<i32>) {
    // Find number of clusters (max label + 1)
    let max_label = labels.iter().max().unwrap_or(&-1);
    let num_clusters = if *max_label >= 0 {
        (max_label + 1) as usize
    } else {
        0
    };

    println!("Number of clusters found: {}", num_clusters);

    // Count points in each cluster
    let mut cluster_counts = vec![0; num_clusters + 1]; // +1 for noise (-1)

    for &label in labels.iter() {
        if label == -1 {
            cluster_counts[num_clusters] += 1; // Count noise points
        } else {
            cluster_counts[label as usize] += 1;
        }
    }

    // Print cluster sizes
    for i in 0..num_clusters {
        println!("Cluster {}: {} points", i, cluster_counts[i]);
    }
    println!("Noise points: {}", cluster_counts[num_clusters]);

    // Calculate percentage of noise
    let noise_percentage = (cluster_counts[num_clusters] as f64 / labels.len() as f64) * 100.0;
    println!("Noise percentage: {:.1}%", noise_percentage);
}
