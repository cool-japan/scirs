use ndarray::{Array2, array};
use scirs2_cluster::meanshift::{mean_shift, MeanShiftOptions, get_bin_seeds};

fn main() {
    println!("Debugging Mean Shift implementation...\n");

    // Create very simple test data with 2 distinct clusters
    let data = array![
        // Cluster 1
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        // Cluster 2
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
    ];

    println!("Test data:\n{:?}\n", data);

    // Test with bandwidth that should keep clusters separate
    let bandwidth = 2.0;
    println!("Using bandwidth = {}", bandwidth);
    
    // First check what seeds are generated
    println!("\nChecking bin seeds:");
    let bin_seeds = get_bin_seeds(&data.view(), bandwidth, 1);
    println!("Number of bin seeds: {}", bin_seeds.nrows());
    for (i, seed) in bin_seeds.outer_iter().enumerate() {
        println!("  Seed {}: [{:.2}, {:.2}]", i, seed[0], seed[1]);
    }
    
    // Now run mean shift without bin seeding (use all points as seeds)
    println!("\nRunning Mean Shift without bin seeding:");
    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        bin_seeding: false,
        ..Default::default()
    };
    
    match mean_shift(&data.view(), options) {
        Ok((centers, labels)) => {
            println!("Number of clusters: {}", centers.nrows());
            println!("Labels: {:?}", labels);
            println!("Centers:");
            for (i, center) in centers.outer_iter().enumerate() {
                println!("  Cluster {}: [{:.2}, {:.2}]", i, center[0], center[1]);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Try with bin seeding
    println!("\nRunning Mean Shift with bin seeding:");
    let options = MeanShiftOptions {
        bandwidth: Some(bandwidth),
        bin_seeding: true,
        ..Default::default()
    };
    
    match mean_shift(&data.view(), options) {
        Ok((centers, labels)) => {
            println!("Number of clusters: {}", centers.nrows());
            println!("Labels: {:?}", labels);
            println!("Centers:");
            for (i, center) in centers.outer_iter().enumerate() {
                println!("  Cluster {}: [{:.2}, {:.2}]", i, center[0], center[1]);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}