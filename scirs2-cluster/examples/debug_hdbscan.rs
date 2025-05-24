use ndarray::Array2;
use scirs2_cluster::{hdbscan, HDBSCANOptions};

fn main() {
    println!("Debugging HDBSCAN implementation...\n");

    // Create very simple test data with two very clear clusters
    let data = Array2::from_shape_vec((10, 2), vec![
        // Cluster 1 - very tight
        0.0, 0.0,
        0.1, 0.0,
        0.0, 0.1,
        0.1, 0.1,
        0.05, 0.05,
        // Cluster 2 - very tight
        10.0, 10.0,
        10.1, 10.0,
        10.0, 10.1,
        10.1, 10.1,
        10.05, 10.05,
    ]).unwrap();

    println!("Test data shape: {:?}", data.shape());
    println!("Data:\n{:?}\n", data);

    // Try different parameters
    let param_sets = vec![
        (2, 2, false),
        (2, 1, false),
        (3, 2, false),
        (2, 2, true),
    ];

    for (min_cluster_size, min_samples, allow_single) in param_sets {
        println!("Testing with min_cluster_size={}, min_samples={}, allow_single_cluster={}",
                 min_cluster_size, min_samples, allow_single);
        
        let options = HDBSCANOptions {
            min_cluster_size,
            min_samples: Some(min_samples),
            allow_single_cluster: allow_single,
            ..Default::default()
        };

        match hdbscan(data.view(), Some(options)) {
            Ok(result) => {
                println!("  Labels: {:?}", result.labels);
                println!("  Probabilities: {:?}", result.probabilities);
                
                // Count clusters
                let mut unique_labels = std::collections::HashSet::new();
                for &label in result.labels.iter() {
                    if label >= 0 {
                        unique_labels.insert(label);
                    }
                }
                println!("  Number of clusters found: {}", unique_labels.len());
                
                // Check condensed tree
                if let Some(ref tree) = result.condensed_tree {
                    println!("  Condensed tree size: {} edges", tree.parent.len());
                }
            }
            Err(e) => {
                println!("  ERROR: {}", e);
            }
        }
        println!();
    }
}