use ndarray::Array2;
use rand::distr::Uniform;
use rand::prelude::*;
use rand::seq::SliceRandom;
use scirs2_cluster::vq::{
    kmeans2, minibatch_kmeans, MiniBatchKMeansOptions, MinitMethod, MissingMethod,
};

#[allow(dead_code)]
fn main() {
    // Generate random data with two clusters
    let data = generate_clustered_data(1000, 2);
    println!(
        "Generated dataset with {} samples in 2D space",
        data.shape()[0]
    );

    // Time standard k-means
    let start_time = std::time::Instant::now();
    let (centroids_std, labels_std) = kmeans2(
        data.view(),
        2,
        Some(10), // iterations
        None,     // threshold
        Some(MinitMethod::Random),
        Some(MissingMethod::Warn),
        Some(true), // check_finite
        Some(42),   // random_seed
    )
    .unwrap();
    let kmeans_duration = start_time.elapsed();

    // Time mini-batch k-means with various batch sizes
    let batch_sizes = [32, 64, 128, 256, 512];

    for &batch_size in &batch_sizes {
        let options = MiniBatchKMeansOptions {
            batch_size,
            max_iter: 100,
            random_seed: Some(42), // For reproducibility
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let (centroids_mb, labels_mb) = minibatch_kmeans(data.view(), 2, Some(options)).unwrap();
        let mb_duration = start_time.elapsed();

        // Calculate agreement with standard k-means
        let agreement = calculate_cluster_agreement(&labels_std, &labels_mb);

        println!(
            "Batch size: {}, Time: {:.2?}, Speedup: {:.2}x, Agreement with standard k-means: {:.1}%",
            batch_size,
            mb_duration,
            kmeans_duration.as_secs_f64() / mb_duration.as_secs_f64(),
            agreement * 100.0
        );

        println!("Standard k-means centroids:\n{:?}", centroids_std);
        println!("Mini-batch k-means centroids:\n{:?}", centroids_mb);
    }

    // Now benchmark with increasing dataset sizes
    println!("\n--- Scaling with dataset size ---");

    let dataset_sizes = [1_000, 10_000, 50_000, 100_000];
    for &size in &dataset_sizes {
        // Generate larger dataset
        let large_data = generate_clustered_data(size, 2);

        // Time standard k-means (only for small datasets)
        let kmeans_duration = if size <= 10_000 {
            let start_time = std::time::Instant::now();
            let _ = kmeans2(
                large_data.view(),
                2,
                Some(10), // iterations
                None,     // threshold
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(true), // check_finite
                Some(42),   // random_seed
            )
            .unwrap();
            let duration = start_time.elapsed();
            println!("Standard k-means with {} samples: {:.2?}", size, duration);
            duration
        } else {
            println!("Skipping standard k-means for {} samples (too slow)", size);
            std::time::Duration::from_secs(0)
        };

        // Time mini-batch k-means
        let options = MiniBatchKMeansOptions {
            batch_size: 256,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let _ = minibatch_kmeans(large_data.view(), 2, Some(options)).unwrap();
        let mb_duration = start_time.elapsed();

        println!(
            "Mini-batch k-means with {} samples: {:.2?}",
            size, mb_duration
        );

        if size <= 10_000 {
            println!(
                "Speedup for {} samples: {:.2}x",
                size,
                kmeans_duration.as_secs_f64() / mb_duration.as_secs_f64()
            );
        }
        println!();
    }
}

/// Generate random data with clusters
#[allow(dead_code)]
fn generate_clustered_data(n_samples: usize, ndim: usize) -> Array2<f64> {
    let mut rng = rand::rng();

    // Define distributions for two clusters
    let cluster1_dist = Uniform::new(-0.5, 0.5).unwrap();
    let cluster2_dist = Uniform::new(-0.5, 0.5).unwrap();

    // Create centers for the clusters
    let center1: Vec<f64> = (0..ndim).map(|_| rng.random_range(0.0..2.0)).collect();
    let center2: Vec<f64> = (0..ndim).map(|_| rng.random_range(8.0..10.0)).collect();

    // Allocate data array
    let mut data = Vec::with_capacity(n_samples * ndim);

    // Generate data points
    for i in 0..n_samples {
        let center = if i < n_samples / 2 {
            &center1
        } else {
            &center2
        };
        let dist = if i < n_samples / 2 {
            &cluster1_dist
        } else {
            &cluster2_dist
        };

        for &center_j in center.iter() {
            data.push(center_j + rng.sample(*dist));
        }
    }

    // Shuffle the data
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_data = Vec::with_capacity(n_samples * ndim);
    for &idx in &indices {
        let start = idx * ndim;
        let end = start + ndim;
        shuffled_data.extend_from_slice(&data[start..end]);
    }

    Array2::from_shape_vec((n_samples, ndim), shuffled_data).unwrap()
}

/// Calculate the agreement between two clusterings (adjusting for label permutation)
#[allow(dead_code)]
fn calculate_cluster_agreement(
    labels1: &ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>,
    labels2: &ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>,
) -> f64 {
    let n_samples = labels1.len();

    // Try both label mappings (since clusters might be labeled differently)
    let mapping1 = (0, 1); // no swap
    let mapping2 = (1, 0); // swap labels

    // Calculate agreement with both mappings
    let agreement1 = labels1
        .iter()
        .zip(labels2.iter())
        .filter(|(&l1, &l2)| {
            if l1 == 0 {
                l2 == mapping1.0
            } else {
                l2 == mapping1.1
            }
        })
        .count() as f64
        / n_samples as f64;

    let agreement2 = labels1
        .iter()
        .zip(labels2.iter())
        .filter(|(&l1, &l2)| {
            if l1 == 0 {
                l2 == mapping2.0
            } else {
                l2 == mapping2.1
            }
        })
        .count() as f64
        / n_samples as f64;

    // Return the best agreement
    agreement1.max(agreement2)
}
