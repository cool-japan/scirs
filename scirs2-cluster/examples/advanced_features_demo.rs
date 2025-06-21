//! Advanced Clustering Features Demo
//!
//! This example demonstrates the advanced features implemented in scirs2-cluster:
//! - Weighted K-means clustering
//! - SIMD-accelerated distance computations
//! - Parallel K-means clustering
//! - Various initialization methods

use ndarray::{Array1, Array2};
use scirs2_cluster::metrics::silhouette_score;
use scirs2_cluster::preprocess::standardize;
use scirs2_cluster::vq::{
    distance_to_centroids_simd, kmeans_with_options, minibatch_kmeans, pairwise_euclidean_simd,
    parallel_kmeans, weighted_kmeans, KMeansInit, KMeansOptions, MiniBatchKMeansOptions,
    ParallelKMeansOptions, WeightedKMeansOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced Clustering Features Demo");
    println!("=================================\n");

    // Generate sample data with 3 clear clusters
    let data = generate_sample_data();
    println!(
        "Generated dataset with {} samples and {} features",
        data.shape()[0],
        data.shape()[1]
    );

    // Standardize the data
    let standardized_data = standardize(data.view(), true)?;
    println!("Data standardized for better clustering performance\n");

    // Demo 1: Standard K-means with different initialization methods
    demo_initialization_methods(&standardized_data)?;

    // Demo 2: Weighted K-means
    demo_weighted_kmeans(&standardized_data)?;

    // Demo 3: Parallel K-means
    demo_parallel_kmeans(&standardized_data)?;

    // Demo 4: Mini-batch K-means
    demo_minibatch_kmeans(&standardized_data)?;

    // Demo 5: SIMD distance computations
    demo_simd_distances(&standardized_data)?;

    println!("Demo completed successfully!");
    Ok(())
}

fn generate_sample_data() -> Array2<f64> {
    // Create 3 clusters of data
    let mut data = Vec::new();

    // Cluster 1: around (0, 0)
    for i in 0..50 {
        let noise_x = (i as f64 * 0.01) % 1.0 - 0.5;
        let noise_y = (i as f64 * 0.017) % 1.0 - 0.5;
        data.push(0.0 + noise_x);
        data.push(0.0 + noise_y);
    }

    // Cluster 2: around (5, 5)
    for i in 0..50 {
        let noise_x = (i as f64 * 0.011) % 1.0 - 0.5;
        let noise_y = (i as f64 * 0.019) % 1.0 - 0.5;
        data.push(5.0 + noise_x);
        data.push(5.0 + noise_y);
    }

    // Cluster 3: around (-3, 4)
    for i in 0..50 {
        let noise_x = (i as f64 * 0.013) % 1.0 - 0.5;
        let noise_y = (i as f64 * 0.023) % 1.0 - 0.5;
        data.push(-3.0 + noise_x);
        data.push(4.0 + noise_y);
    }

    Array2::from_shape_vec((150, 2), data).unwrap()
}

fn demo_initialization_methods(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. K-means with Different Initialization Methods");
    println!("================================================");

    let init_methods = [
        ("Random", KMeansInit::Random),
        ("K-means++", KMeansInit::KMeansPlusPlus),
        ("K-means||", KMeansInit::KMeansParallel),
    ];

    for (name, method) in &init_methods {
        let options = KMeansOptions {
            init_method: *method,
            n_init: 1, // Just one run for demo
            random_seed: Some(42),
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let (_centroids, labels) = kmeans_with_options(data.view(), 3, Some(options))?;
        let duration = start.elapsed();

        // Convert labels from usize to i32 for silhouette_score
        let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
        let silhouette = silhouette_score(data.view(), labels_i32.view())?;

        println!(
            "  {:<12}: Silhouette = {:.3}, Time = {:?}",
            name, silhouette, duration
        );
    }
    println!();
    Ok(())
}

fn demo_weighted_kmeans(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Weighted K-means Clustering");
    println!("==============================");

    // Create weights that favor the first cluster
    let mut weights = Array1::ones(150);
    for i in 0..50 {
        weights[i] = 5.0; // Give 5x weight to first cluster
    }

    let options = WeightedKMeansOptions {
        n_init: 1,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = weighted_kmeans(data.view(), weights.view(), 3, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Weighted K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );

    // Count samples in each cluster
    let mut cluster_counts = vec![0; 3];
    let mut weighted_counts = vec![0.0; 3];
    for i in 0..labels.len() {
        cluster_counts[labels[i]] += 1;
        weighted_counts[labels[i]] += weights[i];
    }

    println!("  Cluster counts: {:?}", cluster_counts);
    println!("  Weighted counts: {:.1?}\n", weighted_counts);
    Ok(())
}

fn demo_parallel_kmeans(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Parallel K-means Clustering");
    println!("==============================");

    let options = ParallelKMeansOptions {
        n_init: 5,
        random_seed: Some(42),
        n_threads: Some(4), // Use 4 threads
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = parallel_kmeans(data.view(), 3, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Parallel K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Used parallel computation with 4 threads\n");
    Ok(())
}

fn demo_minibatch_kmeans(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Mini-batch K-means Clustering");
    println!("=================================");

    let options = MiniBatchKMeansOptions {
        batch_size: 50,
        max_iter: 100,
        random_seed: Some(42),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (_centroids, labels) = minibatch_kmeans(data.view(), 3, Some(options))?;
    let duration = start.elapsed();

    let labels_i32: Array1<i32> = labels.mapv(|x| x as i32);
    let silhouette = silhouette_score(data.view(), labels_i32.view())?;

    println!(
        "  Mini-batch K-means: Silhouette = {:.3}, Time = {:?}",
        silhouette, duration
    );
    println!("  Used batch size of 50 for faster convergence\n");
    Ok(())
}

fn demo_simd_distances(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. SIMD Distance Computations");
    println!("=============================");

    // Compute pairwise distances using SIMD
    let start = std::time::Instant::now();
    let pairwise_distances = pairwise_euclidean_simd(data.view());
    let simd_duration = start.elapsed();

    println!(
        "  SIMD pairwise distances: {} distances computed in {:?}",
        pairwise_distances.len(),
        simd_duration
    );

    // Create some centroids for distance-to-centroids demo
    let centroids = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.0, 0.0, // Center of cluster 1
            5.0, 5.0, // Center of cluster 2
            -3.0, 4.0, // Center of cluster 3
        ],
    )?;

    let start = std::time::Instant::now();
    let centroid_distances = distance_to_centroids_simd(data.view(), centroids.view());
    let simd_centroid_duration = start.elapsed();

    println!(
        "  SIMD centroid distances: {}x{} distances computed in {:?}",
        centroid_distances.shape()[0],
        centroid_distances.shape()[1],
        simd_centroid_duration
    );

    // Verify SIMD results by comparing with a few manual calculations
    let manual_dist = ((data[[0, 0]] - centroids[[0, 0]]).powi(2)
        + (data[[0, 1]] - centroids[[0, 1]]).powi(2))
    .sqrt();
    let simd_dist = centroid_distances[[0, 0]];
    println!(
        "  Verification: Manual distance = {:.6}, SIMD distance = {:.6}",
        manual_dist, simd_dist
    );
    println!(
        "  Difference: {:.2e} (should be very small)\n",
        (manual_dist - simd_dist).abs()
    );

    Ok(())
}
