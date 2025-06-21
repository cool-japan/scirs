use ndarray::Array2;
use scirs2_cluster::vq::{
    kmeans2, kmeans_plus_plus, kmeans_with_options, KMeansOptions, MinitMethod, MissingMethod,
};

fn main() {
    println!("K-means Clustering Example");
    println!("=========================");

    // Create a simple dataset with three clusters
    let data = Array2::from_shape_vec(
        (15, 2),
        vec![
            // Cluster 1
            1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.8, 1.0, 1.2, 1.5, // Cluster 2
            8.0, 8.0, 8.5, 8.5, 9.0, 9.0, 7.5, 8.0, 8.2, 7.8, // Cluster 3
            5.0, 1.0, 4.5, 1.5, 5.5, 2.0, 6.0, 1.0, 4.8, 0.5,
        ],
    )
    .unwrap();

    println!("\nRunning standard K-means with k=3...");
    let (centroids, labels) = kmeans2(
        data.view(),
        3,
        Some(10), // iterations
        None,     // threshold
        Some(MinitMethod::Random),
        Some(MissingMethod::Warn),
        Some(true), // check_finite
        Some(42),   // random_seed
    )
    .unwrap();

    println!("\nCentroids:");
    for (i, centroid) in centroids.outer_iter().enumerate() {
        println!("  Cluster {}: ({:.2}, {:.2})", i, centroid[0], centroid[1]);
    }

    println!("\nCluster assignments:");
    for i in 0..data.shape()[0] {
        println!(
            "  Point ({:.2}, {:.2}) -> Cluster {}",
            data[[i, 0]],
            data[[i, 1]],
            labels[i]
        );
    }

    // Count points in each cluster
    let mut counts = [0; 3];
    for &label in labels.iter() {
        counts[label] += 1;
    }

    println!("\nCluster sizes:");
    for (i, &count) in counts.iter().enumerate() {
        println!("  Cluster {}: {} points", i, count);
    }

    // Run K-means++
    println!("\nRunning K-means++ with k=3...");
    let _initial_centroids = kmeans_plus_plus(data.view(), 3, None).unwrap();
    // Using the default options but with k-means++ initialization
    let options = KMeansOptions::<f64>::default();
    let (centroids_pp, _labels_pp) = kmeans_with_options(data.view(), 3, Some(options)).unwrap();

    println!("\nK-means++ Centroids:");
    for (i, centroid) in centroids_pp.outer_iter().enumerate() {
        println!("  Cluster {}: ({:.2}, {:.2})", i, centroid[0], centroid[1]);
    }
}
