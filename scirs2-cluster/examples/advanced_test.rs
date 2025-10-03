//! Simple test for Advanced clustering functionality

use scirs2_cluster::advanced_clustering::AdvancedClusterer;
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Advanced Clustering...");

    // Create simple test data with two clear clusters
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // Cluster 1
            1.2, 1.1, 0.9, 1.3, 5.0, 5.0, // Cluster 2
            5.2, 5.1, 4.9, 5.3,
        ],
    )?;

    println!("Data shape: {:?}", data.shape());

    // Create Advanced clusterer with basic configuration
    let mut clusterer = AdvancedClusterer::new();

    println!("Running Advanced clustering...");
    let result = clusterer.cluster(&data.view())?;

    println!("Clustering completed successfully!");
    println!("Clusters: {:?}", result.clusters);
    println!("Centroids shape: {:?}", result.centroids.shape());
    println!("Selected algorithm: {}", result.selected_algorithm);
    println!("AI speedup: {:.2}x", result.ai_speedup);
    println!("Quantum advantage: {:.2}x", result.quantum_advantage);
    println!("Neuromorphic benefit: {:.2}x", result.neuromorphic_benefit);
    println!("Confidence: {:.2}", result.confidence);
    println!(
        "Silhouette score: {:.3}",
        result.performance.silhouette_score
    );
    println!("Execution time: {:.3}s", result.performance.execution_time);

    // Test with AI features enabled
    println!("\nTesting with AI features enabled...");
    let mut ai_clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true)
        .with_meta_learning(true);

    let ai_result = ai_clusterer.cluster(&data.view())?;

    println!("AI clustering completed!");
    println!("Selected algorithm: {}", ai_result.selected_algorithm);
    println!("AI speedup: {:.2}x", ai_result.ai_speedup);
    println!("Quantum advantage: {:.2}x", ai_result.quantum_advantage);
    println!(
        "Meta-learning improvement: {:.2}x",
        ai_result.meta_learning_improvement
    );

    Ok(())
}
