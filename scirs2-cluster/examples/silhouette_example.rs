use ndarray::Array2;
use scirs2_cluster::metrics::{silhouette_samples, silhouette_score};
use scirs2_cluster::vq::{kmeans2, MinitMethod, MissingMethod};

fn main() {
    // Generate synthetic data with clusters
    let data = generate_data();

    println!("Evaluating clustering quality with silhouette coefficient...");

    // Try different numbers of clusters
    for k in 2..6 {
        // Run k-means clustering
        let (_, labels) = kmeans2(
            data.view(),
            k,
            Some(10),
            None,
            Some(MinitMethod::PlusPlus),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )
        .unwrap();

        // Convert labels to i32 array for silhouette calculation
        let labels_i32 = labels.mapv(|x| x as i32);

        // Calculate silhouette score
        let score = silhouette_score(data.view(), labels_i32.view()).unwrap();

        println!("K = {}: Silhouette score = {:.4}", k, score);

        // For k=3, also show individual sample scores
        if k == 3 {
            let sample_scores = silhouette_samples(data.view(), labels_i32.view()).unwrap();

            println!("\nSilhouette scores for individual samples (k=3):");

            // Count samples in each cluster
            let mut cluster_sizes = vec![0; k];
            for &label in labels_i32.iter() {
                if label >= 0 && (label as usize) < k {
                    cluster_sizes[label as usize] += 1;
                }
            }

            // Print statistics per cluster
            for (cluster_id, _) in cluster_sizes.iter().enumerate().take(k) {
                let mut cluster_scores = Vec::new();

                for (idx, &label) in labels_i32.iter().enumerate() {
                    if label == cluster_id as i32 {
                        cluster_scores.push(sample_scores[idx]);
                    }
                }

                if !cluster_scores.is_empty() {
                    // Calculate mean score for this cluster
                    let sum: f64 = cluster_scores.iter().sum();
                    let mean = sum / (cluster_scores.len() as f64);

                    // Sort scores for percentiles
                    cluster_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    println!(
                        "Cluster {}: {} samples, Mean score: {:.4}, Min: {:.4}, Max: {:.4}",
                        cluster_id,
                        cluster_sizes[cluster_id],
                        mean,
                        cluster_scores[0],
                        cluster_scores[cluster_scores.len() - 1]
                    );
                }
            }
        }
    }

    println!("\nNote: Higher silhouette scores indicate better-defined clusters.");
    println!("The optimal number of clusters is often the one with the highest score.");
}

fn generate_data() -> Array2<f64> {
    // Create a dataset with 3 natural clusters
    let mut data = Vec::with_capacity(150);

    // Cluster 1
    for i in 0..40 {
        let x = 1.5 + (i % 10) as f64 * 0.1;
        let y = 1.5 + (i / 10) as f64 * 0.1;
        data.push(x);
        data.push(y);
    }

    // Cluster 2
    for i in 0..30 {
        let x = 6.0 + (i % 6) as f64 * 0.1;
        let y = 1.0 + (i / 6) as f64 * 0.1;
        data.push(x);
        data.push(y);
    }

    // Cluster 3
    for i in 0..20 {
        let x = 3.0 + (i % 5) as f64 * 0.1;
        let y = 6.0 + (i / 5) as f64 * 0.1;
        data.push(x);
        data.push(y);
    }

    // Convert to ndarray
    Array2::from_shape_vec((90, 2), data).unwrap()
}
