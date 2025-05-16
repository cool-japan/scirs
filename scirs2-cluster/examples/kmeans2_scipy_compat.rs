//! SciPy-compatible kmeans2 demonstration
//!
//! This example shows the enhanced kmeans2 function that provides
//! compatibility with SciPy's kmeans2 interface.

use ndarray::Array2;
use scirs2_cluster::metrics::silhouette_score;
use scirs2_cluster::vq::{kmeans2, whiten, MinitMethod, MissingMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciPy-compatible kmeans2 Demo");
    println!("============================\n");

    // Generate sample data
    let data = generate_sample_data();

    // Whiten the data (normalize features)
    println!("1. Whitening data...");
    let whitened_data = whiten(&data)?;
    println!("   Original shape: {:?}", data.shape());
    println!("   Whitened shape: {:?}", whitened_data.shape());

    // Test different initialization methods
    println!("\n2. Testing initialization methods:");

    let init_methods = vec![
        ("random", MinitMethod::Random),
        ("points", MinitMethod::Points),
        ("++", MinitMethod::PlusPlus),
    ];

    for (name, init_method) in init_methods {
        println!("\n   Method: '{}'", name);

        let (centroids, labels) = kmeans2(
            whitened_data.view(),
            3,                         // k clusters
            Some(10),                  // iterations
            Some(1e-5),                // threshold
            Some(init_method),         // initialization method
            Some(MissingMethod::Warn), // handle empty clusters
            Some(true),                // check finite values
            Some(42),                  // random seed
        )?;

        println!("   Centroids shape: {:?}", centroids.shape());
        println!("   Labels shape: {:?}", labels.shape());

        // Convert labels to i32 for metric calculation
        let labels_i32 = labels.mapv(|x| x as i32);

        let score = silhouette_score(data.view(), labels_i32.view())?;
        println!("   Silhouette score: {:.3}", score);
    }

    // Test empty cluster handling
    println!("\n3. Testing empty cluster handling:");

    // Create data that might lead to empty clusters
    let sparse_data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
        ],
    )?;

    // Test with warning on empty clusters
    println!("\n   Testing with MissingMethod::Warn:");
    match kmeans2(
        sparse_data.view(),
        4, // More clusters than natural groups
        Some(5),
        Some(1e-4),
        Some(MinitMethod::Random),
        Some(MissingMethod::Warn),
        Some(true),
        Some(123),
    ) {
        Ok((centroids, labels)) => {
            println!("   Success: {} centroids found", centroids.shape()[0]);
            println!("   Unique labels: {:?}", get_unique_labels(&labels));
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    // Test with error on empty clusters
    println!("\n   Testing with MissingMethod::Raise:");
    match kmeans2(
        sparse_data.view(),
        5, // Even more clusters
        Some(5),
        Some(1e-4),
        Some(MinitMethod::Random),
        Some(MissingMethod::Raise),
        Some(true),
        Some(456),
    ) {
        Ok((centroids, labels)) => {
            println!("   Success: {} centroids found", centroids.shape()[0]);
            println!("   Unique labels: {:?}", get_unique_labels(&labels));
        }
        Err(e) => {
            println!("   Error (expected): {}", e);
        }
    }

    Ok(())
}

fn generate_sample_data() -> Array2<f64> {
    let mut data = Vec::new();

    // Cluster 1
    for _ in 0..30 {
        let x = rand::random::<f64>() * 2.0 - 1.0;
        let y = rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
    }

    // Cluster 2
    for _ in 0..30 {
        let x = 5.0 + rand::random::<f64>() * 2.0 - 1.0;
        let y = 5.0 + rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
    }

    // Cluster 3
    for _ in 0..30 {
        let x = -3.0 + rand::random::<f64>() * 2.0 - 1.0;
        let y = 4.0 + rand::random::<f64>() * 2.0 - 1.0;
        data.push(x);
        data.push(y);
    }

    Array2::from_shape_vec((90, 2), data).unwrap()
}

fn get_unique_labels(labels: &ndarray::Array1<usize>) -> Vec<usize> {
    let mut unique = labels.to_vec();
    unique.sort();
    unique.dedup();
    unique
}
