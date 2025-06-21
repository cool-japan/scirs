//! Custom Distance Metrics Demo
//!
//! This example demonstrates the new custom distance metrics functionality
//! implemented in scirs2-cluster. It shows how different distance metrics
//! can significantly affect clustering results.

use ndarray::{Array1, Array2};
use scirs2_cluster::metrics::silhouette_score;
use scirs2_cluster::preprocess::standardize;
use scirs2_cluster::vq::{
    kmeans_with_metric, ChebyshevDistance, CorrelationDistance, CosineDistance, EuclideanDistance,
    KMeansInit, KMeansOptions, MahalanobisDistance, ManhattanDistance,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Custom Distance Metrics Demo");
    println!("============================\n");

    // Generate sample data with distinct patterns for different metrics
    let data = generate_test_data();
    println!(
        "Generated dataset with {} samples and {} features",
        data.shape()[0],
        data.shape()[1]
    );

    // Standardize the data for better comparison
    let standardized_data = standardize(data.view(), true)?;
    println!("Data standardized for fair comparison\n");

    // Demo different distance metrics
    demo_euclidean_distance(&standardized_data)?;
    demo_manhattan_distance(&standardized_data)?;
    demo_chebyshev_distance(&standardized_data)?;
    demo_cosine_distance(&standardized_data)?;
    demo_correlation_distance(&standardized_data)?;
    demo_mahalanobis_distance(&standardized_data)?;

    println!("Demo completed successfully!");
    Ok(())
}

fn generate_test_data() -> Array2<f64> {
    // Create a more complex dataset with multiple patterns
    let mut data = Vec::new();

    // Cluster 1: Tight circular cluster at origin
    for i in 0..30 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 30.0;
        let radius = 0.5 + (i as f64 * 0.01) % 0.3;
        data.push(radius * angle.cos());
        data.push(radius * angle.sin());
    }

    // Cluster 2: Elliptical cluster
    for i in 0..30 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 30.0;
        let x = 5.0 + 2.0 * angle.cos() + (i as f64 * 0.02) % 0.4 - 0.2;
        let y = 3.0 + 0.8 * angle.sin() + (i as f64 * 0.03) % 0.4 - 0.2;
        data.push(x);
        data.push(y);
    }

    // Cluster 3: Linear pattern (high correlation)
    for i in 0..30 {
        let x = -3.0 + (i as f64) * 0.1;
        let y = 2.0 * x + 1.0 + (i as f64 * 0.05) % 0.6 - 0.3;
        data.push(x);
        data.push(y);
    }

    Array2::from_shape_vec((90, 2), data).unwrap()
}

fn demo_euclidean_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Euclidean Distance (L2 norm)");
    println!("================================");

    let metric = Box::new(EuclideanDistance);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Euclidean K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Spherical clusters, equal feature importance");

    // Count cluster sizes
    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}

fn demo_manhattan_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Manhattan Distance (L1 norm)");
    println!("================================");

    let metric = Box::new(ManhattanDistance);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Manhattan K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Robust to outliers, grid-like data");

    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}

fn demo_chebyshev_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Chebyshev Distance (L∞ norm)");
    println!("================================");

    let metric = Box::new(ChebyshevDistance);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Chebyshev K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Maximum feature difference, uniform feature importance");

    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}

fn demo_cosine_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Cosine Distance");
    println!("==================");

    let metric = Box::new(CosineDistance);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Cosine K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Text data, high-dimensional sparse data, direction matters more than magnitude");

    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}

fn demo_correlation_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Correlation Distance");
    println!("=======================");

    let metric = Box::new(CorrelationDistance);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Correlation K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Time series, finding similar patterns regardless of scale");

    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}

fn demo_mahalanobis_distance(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Mahalanobis Distance");
    println!("=======================");

    let metric = Box::new(MahalanobisDistance::from_data(data.view())?);
    let options = KMeansOptions {
        init_method: KMeansInit::KMeansPlusPlus,
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = kmeans_with_metric(data.view(), 3, metric, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Mahalanobis K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Best for: Correlated features, elliptical clusters, accounts for data covariance");

    let mut cluster_counts = vec![0; 3];
    for &label in labels.iter() {
        cluster_counts[label] += 1;
    }
    println!("  Cluster sizes: {:?}\n", cluster_counts);
    Ok(())
}
