use ndarray::Array2;
use scirs2_cluster::vq::{
    kmeans_with_options, parallel_kmeans, KMeansOptions, ParallelKMeansOptions,
};
use std::time::Instant;

fn main() {
    println!("Parallel K-means Clustering Demo");
    println!("{}", "=".repeat(50));

    // Test with different dataset sizes
    let dataset_sizes = vec![1_000, 5_000, 10_000, 50_000];
    let n_features = 20;
    let k = 10;

    for &n_samples in &dataset_sizes {
        println!(
            "\nDataset size: {} samples × {} features",
            n_samples, n_features
        );
        println!("{}", "-".repeat(40));

        // Generate synthetic data
        let data = generate_clustered_data(n_samples, n_features, k);

        // Standard K-means
        let std_options = KMeansOptions {
            n_init: 3,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };

        let start = Instant::now();
        let (std_centroids, std_labels) =
            kmeans_with_options(data.view(), k, Some(std_options)).unwrap();
        let std_duration = start.elapsed();

        // Parallel K-means
        let par_options = ParallelKMeansOptions {
            n_init: 3,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };

        let start = Instant::now();
        let (par_centroids, par_labels) =
            parallel_kmeans(data.view(), k, Some(par_options)).unwrap();
        let par_duration = start.elapsed();

        // Calculate speedup
        let speedup = std_duration.as_secs_f64() / par_duration.as_secs_f64();

        println!("Standard K-means:  {:?}", std_duration);
        println!("Parallel K-means:  {:?}", par_duration);
        println!("Speedup:           {:.2}x", speedup);

        // Compare results (they should be similar but may not be identical due to randomness)
        let agreement = calculate_cluster_agreement(&std_labels, &par_labels);
        println!("Label agreement:   {:.1}%", agreement * 100.0);

        // Compare inertias
        let std_inertia = calculate_inertia(&data, &std_labels, &std_centroids);
        let par_inertia = calculate_inertia(&data, &par_labels, &par_centroids);
        println!("Standard inertia:  {:.2}", std_inertia);
        println!("Parallel inertia:  {:.2}", par_inertia);
        println!("Inertia ratio:     {:.4}", par_inertia / std_inertia);
    }

    // Test with different numbers of threads
    println!("\n\nTesting different thread counts");
    println!("{}", "=".repeat(50));

    let n_samples = 20_000;
    let data = generate_clustered_data(n_samples, n_features, k);

    for n_threads in [1, 2, 4, 8] {
        let options = ParallelKMeansOptions {
            n_init: 3,
            max_iter: 100,
            random_seed: Some(42),
            n_threads: Some(n_threads),
            ..Default::default()
        };

        let start = Instant::now();
        let _ = parallel_kmeans(data.view(), k, Some(options)).unwrap();
        let duration = start.elapsed();

        println!("{} thread(s): {:?}", n_threads, duration);
    }

    // Test scaling with number of clusters
    println!("\n\nScaling with number of clusters");
    println!("{}", "=".repeat(50));

    let n_samples = 10_000;
    let data = generate_clustered_data(n_samples, n_features, 20);

    for k in [5, 10, 20, 50] {
        let std_options = KMeansOptions {
            n_init: 1,
            max_iter: 50,
            random_seed: Some(42),
            ..Default::default()
        };

        let par_options = ParallelKMeansOptions {
            n_init: 1,
            max_iter: 50,
            random_seed: Some(42),
            ..Default::default()
        };

        let start = Instant::now();
        let _ = kmeans_with_options(data.view(), k, Some(std_options)).unwrap();
        let std_duration = start.elapsed();

        let start = Instant::now();
        let _ = parallel_kmeans(data.view(), k, Some(par_options)).unwrap();
        let par_duration = start.elapsed();

        let speedup = std_duration.as_secs_f64() / par_duration.as_secs_f64();

        println!(
            "k={:2}: Standard {:?}, Parallel {:?}, Speedup {:.2}x",
            k, std_duration, par_duration, speedup
        );
    }
}

fn generate_clustered_data(n_samples: usize, n_features: usize, n_clusters: usize) -> Array2<f64> {
    let mut data = Vec::with_capacity(n_samples * n_features);

    // Generate data with clear clusters
    for i in 0..n_samples {
        let cluster = i % n_clusters;

        for _j in 0..n_features {
            // Base value for cluster separation
            let base = (cluster * 10) as f64;

            // Add some noise
            let noise = (rand::random::<f64>() - 0.5) * 2.0;

            data.push(base + noise);
        }
    }

    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

fn calculate_cluster_agreement(
    labels1: &ndarray::Array1<usize>,
    labels2: &ndarray::Array1<usize>,
) -> f64 {
    let n_samples = labels1.len();

    // Build mapping from labels1 to labels2
    let mut label_map = std::collections::HashMap::new();

    for i in 0..n_samples {
        let l1 = labels1[i];
        let l2 = labels2[i];

        *label_map.entry((l1, l2)).or_insert(0) += 1;
    }

    // Find best mapping
    let mut best_mapping = std::collections::HashMap::new();
    let unique_labels1: std::collections::HashSet<_> = labels1.iter().cloned().collect();

    for l1 in unique_labels1 {
        let mut best_l2 = 0;
        let mut best_count = 0;

        for (&(label1, label2), &count) in &label_map {
            if label1 == l1 && count > best_count {
                best_l2 = label2;
                best_count = count;
            }
        }

        best_mapping.insert(l1, best_l2);
    }

    // Calculate agreement
    let mut agreement_count = 0;
    for i in 0..n_samples {
        if let Some(&mapped_label) = best_mapping.get(&labels1[i]) {
            if mapped_label == labels2[i] {
                agreement_count += 1;
            }
        }
    }

    agreement_count as f64 / n_samples as f64
}

fn calculate_inertia(
    data: &Array2<f64>,
    labels: &ndarray::Array1<usize>,
    centroids: &Array2<f64>,
) -> f64 {
    let mut inertia = 0.0;

    for (i, sample) in data.outer_iter().enumerate() {
        let label = labels[i];
        let centroid = centroids.row(label);

        let mut dist_sq = 0.0;
        for j in 0..sample.len() {
            let diff = sample[j] - centroid[j];
            dist_sq += diff * diff;
        }

        inertia += dist_sq;
    }

    inertia
}
