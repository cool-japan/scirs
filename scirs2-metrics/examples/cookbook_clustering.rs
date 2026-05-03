//! Cookbook: Clustering Metrics
//!
//! This example demonstrates silhouette score, Davies-Bouldin index, and
//! Calinski-Harabasz index on synthetic cluster data.
//!
//! Run with: `cargo run --example cookbook_clustering -p scirs2-metrics`

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::clustering::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};

/// Build a synthetic 2D dataset with `n_per_cluster` points per cluster,
/// centred at the provided `centres`. Points are deterministically spread
/// around each centre.
fn make_blobs(
    centres: &[[f64; 2]],
    n_per_cluster: usize,
    spread: f64,
) -> (Array2<f64>, Array1<usize>) {
    let n_clusters = centres.len();
    let n_total = n_clusters * n_per_cluster;
    let mut data = Array2::zeros((n_total, 2));
    let mut labels = Array1::zeros(n_total);

    for (c_idx, centre) in centres.iter().enumerate() {
        for p in 0..n_per_cluster {
            let row = c_idx * n_per_cluster + p;
            // Deterministic spiral-like perturbation so points ring the centre
            let angle = (p as f64) * std::f64::consts::TAU / n_per_cluster as f64;
            let r = spread * (p as f64 / n_per_cluster as f64);
            data[[row, 0]] = centre[0] + r * angle.cos();
            data[[row, 1]] = centre[1] + r * angle.sin();
            labels[row] = c_idx;
        }
    }

    (data, labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Clustering Metrics ===\n");

    // ─── 1. Synthetic cluster data: well-separated ───────────────────────────
    let centres_good = vec![
        [0.0f64, 0.0],
        [10.0, 0.0],
        [5.0, 8.66], // equilateral triangle at distance ~10
    ];
    let (x_good, labels_good) = make_blobs(&centres_good, 20, 1.0);

    // ─── 2. Synthetic cluster data: overlapping clusters ────────────────────
    let centres_bad = vec![
        [0.0f64, 0.0],
        [2.0, 0.0], // very close centres → heavy overlap
        [1.0, 1.7],
    ];
    let (x_bad, labels_bad) = make_blobs(&centres_bad, 20, 1.5);

    // ─── 3. Compute metrics ──────────────────────────────────────────────────
    // Silhouette: range [-1, 1]; higher is better
    let sil_good: f64 = silhouette_score(&x_good, &labels_good, "euclidean")?;
    let sil_bad: f64 = silhouette_score(&x_bad, &labels_bad, "euclidean")?;

    // Davies-Bouldin: range [0, ∞); lower is better
    let db_good: f64 = davies_bouldin_score(&x_good, &labels_good)?;
    let db_bad: f64 = davies_bouldin_score(&x_bad, &labels_bad)?;

    // Calinski-Harabasz: range [0, ∞); higher is better
    let ch_good: f64 = calinski_harabasz_score(&x_good, &labels_good)?;
    let ch_bad: f64 = calinski_harabasz_score(&x_bad, &labels_bad)?;

    // ─── 4. Print results with commentary ───────────────────────────────────
    println!("Dataset A (well-separated): 3 clusters, centres ~10 apart, spread=1.0");
    println!("Dataset B (overlapping)   : 3 clusters, centres ~2 apart,  spread=1.5");
    println!("n_per_cluster=20, total n=60 each\n");

    println!(
        "{:<35} {:>12}  {:>12}",
        "Metric", "Separated", "Overlapping"
    );
    println!("{}", "-".repeat(63));

    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Silhouette (↑ better, max=1)", sil_good, sil_bad
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Davies-Bouldin (↓ better, min=0)", db_good, db_bad
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Calinski-Harabasz (↑ better)", ch_good, ch_bad
    );

    println!("\n--- When to use which metric ---");
    println!("  Silhouette        : intrinsic quality; robust, general-purpose");
    println!("  Davies-Bouldin    : easy to interpret; sensitive to outliers");
    println!("  Calinski-Harabasz : fast O(k·n); good for comparing k choices");

    println!("\n--- Choosing k (number of clusters) ---");
    println!("  Plot metric vs k and look for an 'elbow' or peak.");
    println!("  Silhouette peak and Calinski-Harabasz peak often agree.");
    println!("  Davies-Bouldin valley typically aligns with the same k.");

    println!("\n=== Done ===");
    Ok(())
}
