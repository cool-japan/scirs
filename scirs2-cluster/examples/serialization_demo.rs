//! Example demonstrating clustering model serialization and persistence

use scirs2_cluster::{
    hierarchy::{linkage, LinkageMethod},
    kmeans_to_model, save_hierarchy, save_kmeans,
    vq::kmeans2,
    HierarchicalModel, KMeansModel, Metric, SerializableModel,
};
use scirs2_core::ndarray::{array, Array2};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let data = array![
        [1.0, 2.0],
        [1.2, 1.8],
        [0.8, 2.1],
        [5.0, 4.5],
        [5.2, 4.8],
        [4.8, 4.6],
        [9.0, 8.5],
        [9.2, 8.8],
        [8.8, 8.6]
    ];

    println!("=== K-means Serialization Example ===\n");
    kmeans_example(&data)?;

    println!("\n=== Hierarchical Clustering Serialization Example ===\n");
    hierarchical_example(&data)?;

    println!("\n=== Model Loading and Prediction Example ===\n");
    model_loading_example()?;

    Ok(())
}

#[allow(dead_code)]
fn kmeans_example(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Perform K-means clustering
    let (centroids, labels) = kmeans2(data.view(), 3, None, None, None, None, None, None)?;
    println!(
        "K-means clustering completed with {} clusters",
        centroids.nrows()
    );
    println!("Cluster assignments: {:?}", labels);

    // Convert to serializable model
    let model = kmeans_to_model(centroids.clone(), Some(labels.clone()), 10, 100.0);

    // Save to JSON file
    model.save_to_file("kmeans_model.json")?;
    println!("K-means model saved to 'kmeans_model.json'");

    // Alternative: Use convenience function
    save_kmeans(&model, "kmeans_model_alt.json")?;
    println!("K-means model saved using convenience function");

    Ok(())
}

#[allow(dead_code)]
fn hierarchical_example(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Perform hierarchical clustering
    let linkage_matrix = linkage(data.view(), LinkageMethod::Average, Metric::Euclidean)?;
    let n_observations = data.nrows();

    // Create model with sample labels
    let labels = vec![
        "Sample_A".to_string(),
        "Sample_B".to_string(),
        "Sample_C".to_string(),
        "Sample_D".to_string(),
        "Sample_E".to_string(),
        "Sample_F".to_string(),
        "Sample_G".to_string(),
        "Sample_H".to_string(),
        "Sample_I".to_string(),
    ];

    let model = HierarchicalModel::new(
        linkage_matrix.clone(),
        n_observations,
        "average".to_string(),
        Some(labels.clone()),
    );

    // Save to JSON file
    model.save_to_file("hierarchy_model.json")?;
    println!("Hierarchical model saved to 'hierarchy_model.json'");

    // Export to Newick format
    let newick = model.to_newick()?;
    println!("\nNewick format:");
    println!("{}", newick);

    // Export to JSON tree format
    let json_tree = model.to_json_tree()?;
    println!("\nJSON tree format:");
    println!("{}", serde_json::to_string_pretty(&json_tree)?);

    // Alternative: Use convenience function
    save_hierarchy(&model, "hierarchy_model_alt.json")?;
    println!("\nHierarchical model saved using convenience function");

    Ok(())
}

#[allow(dead_code)]
fn model_loading_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load K-means model
    let loaded_kmeans = KMeansModel::load_from_file("kmeans_model.json")?;
    println!(
        "Loaded K-means model with {} clusters",
        loaded_kmeans.n_clusters
    );
    println!("Centroids shape: {:?}", loaded_kmeans.centroids.shape());

    // Make predictions on new data
    let new_data = array![[1.1, 1.9], [5.1, 4.7], [8.9, 8.7], [2.0, 2.5], [6.0, 5.0]];

    let predictions = loaded_kmeans.predict(new_data.view())?;
    println!("\nPredictions for new data: {:?}", predictions);

    // Load hierarchical model
    let loaded_hierarchy = HierarchicalModel::load_from_file("hierarchy_model.json")?;
    println!("\nLoaded hierarchical model:");
    println!("  Method: {}", loaded_hierarchy.method);
    println!(
        "  Number of observations: {}",
        loaded_hierarchy.n_observations
    );
    println!(
        "  Linkage matrix shape: {:?}",
        loaded_hierarchy.linkage.shape()
    );

    // Clean up temporary files
    std::fs::remove_file("kmeans_model.json").ok();
    std::fs::remove_file("kmeans_model_alt.json").ok();
    std::fs::remove_file("hierarchy_model.json").ok();
    std::fs::remove_file("hierarchy_model_alt.json").ok();

    Ok(())
}
