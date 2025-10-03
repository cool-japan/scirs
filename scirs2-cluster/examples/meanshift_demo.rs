use scirs2_cluster::meanshift::{estimate_bandwidth, mean_shift, MeanShiftOptions};
use scirs2_core::ndarray::{array, Array2};
use scirs2_core::random::prelude::*;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mean Shift Clustering Demo");
    println!("==========================\n");

    // Example 1: Simple 2D dataset
    println!("Example 1: Simple 2D dataset");
    let data = array![
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 0.0],
        [4.0, 7.0],
        [3.0, 5.0],
        [3.0, 6.0]
    ];

    // Estimate bandwidth automatically
    let bandwidth = estimate_bandwidth(&data.view(), Some(0.5), None, None)?;
    println!("Estimated bandwidth: {:.3}", bandwidth);

    // Run Mean Shift with estimated bandwidth
    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        ..Default::default()
    };

    let (centers, labels) = mean_shift(&data.view(), options)?;

    println!("Number of clusters: {}", centers.nrows());
    println!("Cluster centers:");
    for (i, center) in centers.rows().into_iter().enumerate() {
        println!("  Cluster {}: {:?}", i, center);
    }
    println!("Labels: {:?}", labels);
    println!();

    // Example 2: Generated blobs
    println!("Example 2: Generated blobs with 3 clusters");

    // Create 3 clusters with different sizes
    let n_samples = 300;
    let n_features = 2;
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster 1
    let cluster1_size = 100;
    let mut cluster1 = Array2::zeros((cluster1_size, n_features));
    for i in 0..cluster1_size {
        for j in 0..n_features {
            // Generate random normal values with mean 0.0 and std 1.0
            cluster1[[i, j]] = rng.random::<f64>() * 1.0 + 0.0;
        }
    }

    // Generate cluster 2
    let cluster2_size = 100;
    let mut cluster2 = Array2::zeros((cluster2_size, n_features));
    for i in 0..cluster2_size {
        for j in 0..n_features {
            // Generate random normal values with mean 5.0 and std 1.0
            cluster2[[i, j]] = rng.random::<f64>() * 1.0 + 5.0;
        }
    }

    // Generate cluster 3
    let cluster3_size = 100;
    let mut cluster3 = Array2::zeros((cluster3_size, n_features));
    for i in 0..cluster3_size {
        for j in 0..n_features {
            // Generate random normal values with mean 0.0 and std 1.0
            cluster3[[i, j]] = rng.random::<f64>() * 1.0 + 0.0;
        }
    }

    // Shift cluster 3
    for mut row in cluster3.rows_mut() {
        row[0] += 10.0;
        row[1] -= 5.0;
    }

    // Combine all clusters
    let n_total = cluster1.nrows() + cluster2.nrows() + cluster3.nrows();
    let mut data = Array2::zeros((n_total, n_features));

    // Copy data from clusters to combined array
    let mut row_idx = 0;
    for i in 0..cluster1.nrows() {
        for j in 0..n_features {
            data[[row_idx, j]] = cluster1[[i, j]];
        }
        row_idx += 1;
    }

    for i in 0..cluster2.nrows() {
        for j in 0..n_features {
            data[[row_idx, j]] = cluster2[[i, j]];
        }
        row_idx += 1;
    }

    for i in 0..cluster3.nrows() {
        for j in 0..n_features {
            data[[row_idx, j]] = cluster3[[i, j]];
        }
        row_idx += 1;
    }

    // Estimate bandwidth
    let bandwidth = estimate_bandwidth(&data.view(), None, Some(n_samples / 10), Some(seed))?;
    println!("Estimated bandwidth on larger dataset: {:.3}", bandwidth);

    // Run Mean Shift with bin seeding for efficiency
    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        bin_seeding: true,
        min_bin_freq: 5,
        ..Default::default()
    };

    let (centers, labels) = mean_shift(&data.view(), options)?;

    println!("Number of clusters found: {}", centers.nrows());
    println!("Cluster centers:");
    for (i, center) in centers.rows().into_iter().enumerate() {
        println!("  Cluster {}: [{:.2}, {:.2}]", i, center[0], center[1]);
    }

    // Count samples in each cluster
    let mut cluster_counts = vec![0; centers.nrows()];
    for &label in labels.iter() {
        if label >= 0 && (label as usize) < cluster_counts.len() {
            cluster_counts[label as usize] += 1;
        }
    }

    println!("Samples per cluster:");
    for (i, &count) in cluster_counts.iter().enumerate() {
        println!("  Cluster {}: {} samples", i, count);
    }

    // Example 3: Mean Shift with no cluster_all
    println!("\nExample 3: Mean Shift with cluster_all=false");

    // Add some noise points
    // Just reuse the data
    let noise_size = 20;
    let mut noise = Array2::zeros((noise_size, n_features));
    for i in 0..noise_size {
        for j in 0..n_features {
            // Generate uniform random values between -10.0 and 20.0
            noise[[i, j]] = rng.random::<f64>() * 30.0 - 10.0;
        }
    }

    // Combine data and noise
    let mut combined_data = Array2::zeros((data.nrows() + noise.nrows(), n_features));
    for (i, row) in data.rows().into_iter().enumerate() {
        for j in 0..n_features {
            combined_data[[i, j]] = row[j];
        }
    }

    let data_rows = data.nrows();
    for (i, row) in noise.rows().into_iter().enumerate() {
        for j in 0..n_features {
            combined_data[[data_rows + i, j]] = row[j];
        }
    }

    // Run Mean Shift with cluster_all=false
    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        bin_seeding: true,
        min_bin_freq: 5,
        cluster_all: false,
        ..Default::default()
    };

    let (centers, labels) = mean_shift(&combined_data.view(), options)?;

    println!("Number of clusters found: {}", centers.nrows());

    // Count noise points
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    println!("Number of noise points (label=-1): {}", noise_count);

    // Verify that all noise points have been correctly identified
    let signal_count = combined_data.nrows() - noise_count;
    println!("Signal points: {}", signal_count);

    Ok(())
}
