use scirs2_cluster::hierarchy::{fcluster, linkage, ClusterCriterion, LinkageMethod, Metric};
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() {
    println!("Hierarchical Clustering Example");
    println!("==============================");

    // Create a simple dataset with three clusters
    let data = Array2::from_shape_vec(
        (9, 2),
        vec![
            // Cluster 1
            1.0, 1.0, 1.5, 1.2, 1.2, 1.5, // Cluster 2
            5.0, 5.0, 5.5, 5.2, 5.2, 5.5, // Cluster 3
            9.0, 1.0, 9.5, 1.2, 9.2, 1.5,
        ],
    )
    .unwrap();

    // Print the dataset
    println!("\nDataset:");
    for i in 0..data.shape()[0] {
        println!("  Point {}: ({}, {})", i, data[[i, 0]], data[[i, 1]]);
    }

    // Calculate linkage matrix with different methods
    println!("\nPerforming hierarchical clustering with different linkage methods...");

    let linkage_methods = [
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ];

    for &method in &linkage_methods {
        println!("\n{method:?} Linkage:");

        // Calculate linkage matrix
        let linkage_matrix = linkage(data.view(), method, Metric::Euclidean).unwrap();

        // Print the linkage matrix (first few rows)
        println!("  Linkage Matrix (first few rows):");
        for i in 0..3.min(linkage_matrix.shape()[0]) {
            println!(
                "    [{}, {}] merged at distance {:.2} (size: {})",
                linkage_matrix[[i, 0]],
                linkage_matrix[[i, 1]],
                linkage_matrix[[i, 2]],
                linkage_matrix[[i, 3]]
            );
        }

        // Extract clusters with different number of clusters
        for n_clusters in 2..=4 {
            let labels = fcluster(
                &linkage_matrix,
                n_clusters,
                Some(ClusterCriterion::MaxClust),
            )
            .unwrap();

            println!("\n  {n_clusters} Clusters:");
            // Count elements in each cluster
            let mut cluster_counts = vec![0; n_clusters];
            for &label in labels.iter() {
                cluster_counts[label] += 1;
            }

            // Print cluster sizes
            for (i, &count) in cluster_counts.iter().enumerate() {
                println!("    Cluster {i}: {count} points");
            }

            // Print points in each cluster
            for cluster_idx in 0..n_clusters {
                println!("    Points in Cluster {cluster_idx}:");
                for (point_idx, &label) in labels.iter().enumerate() {
                    if label == cluster_idx {
                        println!(
                            "      Point {}: ({}, {})",
                            point_idx,
                            data[[point_idx, 0]],
                            data[[point_idx, 1]]
                        );
                    }
                }
            }
        }
    }
}
