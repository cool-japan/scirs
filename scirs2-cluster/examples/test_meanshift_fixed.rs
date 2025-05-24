use ndarray::Array2;
use scirs2_cluster::meanshift::{mean_shift, MeanShiftOptions};

fn main() {
    println!("Testing Mean Shift after merge threshold fix...\n");

    // Create test data with 3 clear clusters
    let data = Array2::from_shape_vec((15, 2), vec![
        // Cluster 1
        0.0, 0.0,
        0.1, 0.0,
        0.0, 0.1,
        0.1, 0.1,
        0.05, 0.05,
        // Cluster 2
        5.0, 5.0,
        5.1, 5.0,
        5.0, 5.1,
        5.1, 5.1,
        5.05, 5.05,
        // Cluster 3
        10.0, 0.0,
        10.1, 0.0,
        10.0, 0.1,
        10.1, 0.1,
        10.05, 0.05,
    ]).unwrap();

    println!("Test data with 3 well-separated clusters");
    println!("Cluster spacing: ~5.0, within-cluster spacing: ~0.1\n");
    
    // Test with different bandwidths
    let bandwidths = vec![0.5, 1.0, 2.0, 3.0];
    
    for bandwidth in bandwidths {
        println!("Bandwidth = {}", bandwidth);
        
        let options = MeanShiftOptions {
            bandwidth: Some(bandwidth),
            ..Default::default()
        };
        
        match mean_shift(&data.view(), options) {
            Ok((centers, labels)) => {
                println!("  Number of clusters: {}", centers.nrows());
                
                // Count points per cluster
                let mut cluster_counts = vec![0; centers.nrows()];
                for &label in labels.iter() {
                    if label >= 0 && (label as usize) < cluster_counts.len() {
                        cluster_counts[label as usize] += 1;
                    }
                }
                
                println!("  Points per cluster: {:?}", cluster_counts);
                println!("  Cluster centers:");
                for (i, center) in centers.outer_iter().enumerate() {
                    println!("    Cluster {}: [{:.2}, {:.2}]", i, center[0], center[1]);
                }
            }
            Err(e) => println!("  Mean Shift failed: {}", e),
        }
        println!();
    }
}