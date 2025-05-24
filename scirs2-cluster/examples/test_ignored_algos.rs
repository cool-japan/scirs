use ndarray::Array2;
use scirs2_cluster::{
    dbscan_clustering, hdbscan, HDBSCANOptions,
    density::{optics, DistanceMetric},
    meanshift::{mean_shift, MeanShiftOptions, estimate_bandwidth},
    spectral::{spectral_clustering, SpectralClusteringOptions, AffinityMode},
};

fn main() {
    println!("Testing previously ignored algorithms...\n");

    // Create test data with clear clusters
    let data = Array2::from_shape_vec((20, 2), vec![
        // Cluster 1
        1.0, 1.0,
        1.2, 1.1,
        0.9, 1.2,
        1.1, 0.9,
        1.0, 1.1,
        // Cluster 2
        5.0, 5.0,
        5.1, 4.9,
        4.9, 5.1,
        5.2, 5.0,
        5.0, 5.2,
        // Cluster 3
        9.0, 1.0,
        9.1, 1.1,
        8.9, 0.9,
        9.2, 1.0,
        9.0, 1.2,
        // Noise points
        3.0, 3.0,
        7.0, 7.0,
        2.0, 8.0,
        8.0, 8.0,
        6.0, 2.0,
    ]).unwrap();

    // Test HDBSCAN
    println!("Testing HDBSCAN:");
    let hdbscan_options = HDBSCANOptions {
        min_cluster_size: 3,
        min_samples: Some(2),
        ..Default::default()
    };
    match hdbscan(data.view(), Some(hdbscan_options)) {
        Ok(result) => {
            println!("  ✓ HDBSCAN succeeded");
            println!("    Labels: {:?}", result.labels);
            
            // Test dbscan_clustering extraction
            match dbscan_clustering(&result, 1.0) {
                Ok(dbscan_labels) => {
                    println!("  ✓ DBSCAN extraction succeeded");
                    println!("    DBSCAN labels: {:?}", dbscan_labels);
                }
                Err(e) => println!("  ✗ DBSCAN extraction failed: {}", e),
            }
        }
        Err(e) => println!("  ✗ HDBSCAN failed: {}", e),
    }

    // Test OPTICS
    println!("\nTesting OPTICS:");
    match optics::optics(data.view(), 3, None, Some(DistanceMetric::Euclidean)) {
        Ok(result) => {
            println!("  ✓ OPTICS succeeded");
            println!("    Ordering: {:?}", result.ordering);
            println!("    Reachability (first 5): {:?}", &result.reachability[..5]);
        }
        Err(e) => println!("  ✗ OPTICS failed: {}", e),
    }

    // Test Mean Shift
    println!("\nTesting Mean Shift:");
    
    // First test bandwidth estimation
    match estimate_bandwidth(&data.view(), Some(0.5), None, None) {
        Ok(bandwidth) => {
            println!("  ✓ Bandwidth estimation succeeded: {}", bandwidth);
            
            let meanshift_options = MeanShiftOptions {
                bandwidth: Some(bandwidth),
                ..Default::default()
            };
            match mean_shift(&data.view(), meanshift_options) {
                Ok((centers, labels)) => {
                    println!("  ✓ Mean Shift succeeded");
                    println!("    Number of clusters: {}", centers.nrows());
                    println!("    Labels: {:?}", labels);
                }
                Err(e) => println!("  ✗ Mean Shift failed: {}", e),
            }
        }
        Err(e) => println!("  ✗ Bandwidth estimation failed: {}", e),
    }

    // Test Spectral Clustering
    println!("\nTesting Spectral Clustering:");
    let spectral_options = SpectralClusteringOptions {
        affinity: AffinityMode::RBF,
        gamma: 0.5,
        ..Default::default()
    };
    match spectral_clustering(data.view(), 3, Some(spectral_options)) {
        Ok((embeddings, labels)) => {
            println!("  ✓ Spectral Clustering succeeded");
            println!("    Labels: {:?}", labels);
            println!("    Embedding shape: {:?}", embeddings.shape());
        }
        Err(e) => println!("  ✗ Spectral Clustering failed: {}", e),
    }

    println!("\nAll tests completed!");
}