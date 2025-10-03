//! Example demonstrating the Leader clustering algorithm

use scirs2_cluster::{
    euclidean_distance, leader_clustering, manhattan_distance, LeaderClustering, LeaderNode,
    LeaderTree,
};
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Leader Algorithm Demo ===\n");

    // Generate sample data with clear clusters
    let data = array![
        // Cluster 1 (around 1, 1)
        [1.0, 1.0],
        [1.2, 0.8],
        [0.8, 1.2],
        [1.1, 1.1],
        // Cluster 2 (around 5, 5)
        [5.0, 5.0],
        [5.2, 4.8],
        [4.8, 5.2],
        [5.1, 5.1],
        // Cluster 3 (around 10, 1)
        [10.0, 1.0],
        [10.2, 0.8],
        [9.8, 1.2],
        [10.1, 1.1],
    ];

    println!("Data points:");
    for (i, row) in data.rows().into_iter().enumerate() {
        println!("  Point {}: [{:.1}, {:.1}]", i, row[0], row[1]);
    }
    println!();

    // Example 1: Basic leader clustering with Euclidean distance
    println!("Example 1: Basic Leader Clustering (Euclidean distance, threshold=2.0)");
    let (leaders, labels) = leader_clustering(data.view(), 2.0, euclidean_distance)?;

    println!("Number of clusters: {}", leaders.nrows());
    println!("Cluster leaders:");
    for (i, leader) in leaders.rows().into_iter().enumerate() {
        println!("  Leader {}: [{:.2}, {:.2}]", i, leader[0], leader[1]);
    }
    println!("Cluster assignments: {:?}\n", labels);

    // Example 2: Different threshold
    println!("Example 2: Leader Clustering with smaller threshold (1.0)");
    let (leaders2, labels2) = leader_clustering(data.view(), 1.0, euclidean_distance)?;
    println!("Number of clusters: {}", leaders2.nrows());
    println!("Cluster assignments: {:?}\n", labels2);

    // Example 3: Manhattan distance
    println!("Example 3: Leader Clustering with Manhattan distance");
    let (leaders3, labels3) = leader_clustering(data.view(), 3.0, manhattan_distance)?;
    println!("Number of clusters: {}", leaders3.nrows());
    println!("Cluster assignments: {:?}\n", labels3);

    // Example 4: Using LeaderClustering class
    println!("Example 4: Using LeaderClustering class");
    let mut leader_model = LeaderClustering::new(2.0)?;

    // Fit the model
    leader_model.fit(data.view())?;
    println!("Number of clusters found: {}", leader_model.n_clusters());

    // Get cluster centers
    let centers = leader_model.get_leaders();
    println!("Cluster centers:");
    for (i, center) in centers.rows().into_iter().enumerate() {
        println!("  Center {}: [{:.2}, {:.2}]", i, center[0], center[1]);
    }

    // Predict on new data
    let new_data = array![
        [1.5, 1.5],  // Should be assigned to cluster 1
        [5.5, 5.5],  // Should be assigned to cluster 2
        [10.5, 1.5], // Should be assigned to cluster 3
        [7.5, 3.0],  // Between clusters, will be assigned to closest
    ];

    let predictions = leader_model.predict(new_data.view())?;
    println!("\nPredictions for new data:");
    for (i, (point, &label)) in new_data
        .rows()
        .into_iter()
        .zip(predictions.iter())
        .enumerate()
    {
        println!(
            "  Point [{:.1}, {:.1}] -> Cluster {}",
            point[0], point[1], label
        );
    }
    println!();

    // Example 5: Hierarchical Leader Tree
    println!("Example 5: Hierarchical Leader Tree");
    let thresholds = vec![5.0, 2.0, 1.0];
    let tree = LeaderTree::build_hierarchical(data.view(), &thresholds)?;

    println!("Hierarchical clustering with thresholds: {:?}", thresholds);
    println!("Number of root clusters: {}", tree.roots.len());
    println!("Total nodes in tree: {}", tree.node_count());

    // Print tree structure
    println!("\nTree structure:");
    for (i, root) in tree.roots.iter().enumerate() {
        print_tree_node(root, 0, i);
    }

    // Example 6: Order dependency demonstration
    println!("\nExample 6: Demonstrating order dependency");
    let shuffled_data = array![
        // Same data but in different order
        [5.0, 5.0],
        [1.0, 1.0],
        [10.0, 1.0],
        [1.2, 0.8],
        [5.2, 4.8],
        [10.2, 0.8],
        [0.8, 1.2],
        [4.8, 5.2],
        [9.8, 1.2],
        [1.1, 1.1],
        [5.1, 5.1],
        [10.1, 1.1],
    ];

    let (leaders_orig, _labels_orig) = leader_clustering(data.view(), 1.5, euclidean_distance)?;
    let (leaders_shuf, _labels_shuf) =
        leader_clustering(shuffled_data.view(), 1.5, euclidean_distance)?;

    println!("Original order - {} clusters", leaders_orig.nrows());
    println!("Shuffled order - {} clusters", leaders_shuf.nrows());
    println!("(Results may differ due to order-dependent nature of Leader algorithm)");

    Ok(())
}

#[allow(dead_code)]
fn print_tree_node<F: scirs2_core::numeric::Float + std::fmt::Display>(
    node: &LeaderNode<F>,
    depth: usize,
    index: usize,
) {
    let indent = "  ".repeat(depth);
    println!(
        "{}Node {}: [{:.2}, {:.2}] ({} members)",
        indent,
        index,
        node.leader[0],
        node.leader[1],
        node.members.len()
    );

    for (i, child) in node.children.iter().enumerate() {
        print_tree_node(child, depth + 1, i);
    }
}
