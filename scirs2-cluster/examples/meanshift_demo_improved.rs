use scirs2_cluster::meanshift::{estimate_bandwidth, mean_shift, MeanShiftOptions};
use scirs2_core::ndarray::{array, Array2};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mean Shift Clustering Demo (Improved)");
    println!("=====================================\n");

    // Example 1: Well-separated clusters
    println!("Example 1: Well-separated 2D clusters");
    let data1 = array![
        // Cluster 1
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.15, 0.15],
        // Cluster 2
        [3.0, 3.0],
        [3.1, 3.1],
        [3.2, 3.0],
        [3.0, 3.2],
        [3.15, 3.15],
        // Cluster 3
        [6.0, 0.0],
        [6.1, 0.1],
        [6.2, 0.0],
        [6.0, 0.2],
        [6.15, 0.15],
    ];

    // Use a bandwidth that's larger than within-cluster distance but smaller than between-cluster distance
    let bandwidth = 1.0; // Within-cluster: ~0.2, Between-cluster: ~3.0
    println!("Using bandwidth: {}", bandwidth);

    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        bin_seeding: false,
        ..Default::default()
    };

    let (centers, labels) = mean_shift(&data1.view(), options)?;
    println!("Number of clusters found: {}", centers.nrows());

    // Count points per cluster
    let mut cluster_counts = vec![0; centers.nrows()];
    for &label in labels.iter() {
        if label >= 0 {
            cluster_counts[label as usize] += 1;
        }
    }

    println!("Cluster centers and sizes:");
    for (i, (center, count)) in centers.outer_iter().zip(cluster_counts.iter()).enumerate() {
        println!(
            "  Cluster {}: center=({:.2}, {:.2}), size={}",
            i, center[0], center[1], count
        );
    }

    // Example 2: Demonstrate bandwidth estimation
    println!("\n\nExample 2: Bandwidth estimation on larger dataset");

    // Generate more data
    let mut data_vec = Vec::new();

    // Add three Gaussian-like clusters
    use scirs2_core::random::Rng;
    let mut rng = scirs2_core::random::rng();

    // Cluster 1 around (0, 0)
    for _ in 0..50 {
        data_vec.push(rng.random::<f64>() * 0.5 - 0.25);
        data_vec.push(rng.random::<f64>() * 0.5 - 0.25);
    }

    // Cluster 2 around (5, 5)
    for _ in 0..50 {
        data_vec.push(5.0 + rng.random::<f64>() * 0.5 - 0.25);
        data_vec.push(5.0 + rng.random::<f64>() * 0.5 - 0.25);
    }

    // Cluster 3 around (5, 0)
    for _ in 0..50 {
        data_vec.push(5.0 + rng.random::<f64>() * 0.5 - 0.25);
        data_vec.push(rng.random::<f64>() * 0.5 - 0.25);
    }

    let data2 = Array2::from_shape_vec((150, 2), data_vec)?;

    // Estimate bandwidth with different quantiles
    let quantiles = vec![0.1, 0.2, 0.3];

    for quantile in quantiles {
        let estimated_bw = estimate_bandwidth(&data2.view(), Some(quantile), None, None)?;
        println!(
            "\nQuantile {}: Estimated bandwidth = {:.3}",
            quantile, estimated_bw
        );

        let options = MeanShiftOptions {
            bandwidth: Some(estimated_bw),
            ..Default::default()
        };

        match mean_shift(&data2.view(), options) {
            Ok(centers_) => {
                println!("  Clusters found: {}", centers.nrows());
            }
            Err(e) => println!("  Error: {}", e),
        }
    }

    // Example 3: Using manual bandwidth tuning
    println!("\n\nExample 3: Manual bandwidth tuning");
    println!("Testing bandwidths from 0.1 to 2.0:");

    let test_bandwidths = vec![0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];

    for bw in test_bandwidths {
        let options = MeanShiftOptions {
            bandwidth: Some(bw),
            ..Default::default()
        };

        match mean_shift(&data2.view(), options) {
            Ok(centers_) => {
                print!("  Bandwidth {:.1}: {} clusters", bw, centers.nrows());
                if centers.nrows() == 3 {
                    print!(" ✓ (optimal)");
                }
                println!();
            }
            Err(e) => println!("  Bandwidth {:.1}: Error - {}", bw, e),
        }
    }

    Ok(())
}
