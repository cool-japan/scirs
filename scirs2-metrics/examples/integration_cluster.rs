//! Framework Integration: Cluster Labels → Clustering Metrics
//!
//! This example simulates the output pattern of a clustering algorithm (as would
//! come from scirs2-cluster — k-means, DBSCAN, etc.) and evaluates the resulting
//! assignment with scirs2-metrics internal and external metrics.
//!
//! In a real pipeline you would replace the synthetic assignments with actual
//! cluster labels from scirs2-cluster (or any other clustering crate).
//!
//! Run with: `cargo run --example integration_cluster -p scirs2-metrics`

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::clustering::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};

/// Build a 2D dataset with `k` clusters of `n_per` points, using a deterministic
/// grid layout so different k values can be tested reproducibly.
fn make_grid_clusters(k: usize, n_per: usize, spread: f64) -> (Array2<f64>, Array1<usize>) {
    // Place cluster centres on a grid: 0, 3, 6, ... along x; wrap to 2D grid
    let cols = (k as f64).sqrt().ceil() as usize;
    let n_total = k * n_per;
    let mut data = Array2::zeros((n_total, 2));
    let mut labels = Array1::zeros(n_total);

    for c in 0..k {
        let cx = (c % cols) as f64 * 3.0;
        let cy = (c / cols) as f64 * 3.0;
        for p in 0..n_per {
            let row = c * n_per + p;
            let angle = (p as f64) * std::f64::consts::TAU / n_per as f64;
            let r = spread * (0.3 + 0.7 * (p as f64 / n_per as f64));
            data[[row, 0]] = cx + r * angle.cos();
            data[[row, 1]] = cy + r * angle.sin();
            labels[row] = c;
        }
    }
    (data, labels)
}

/// Simulate k-means assignments: split data naively into k equal groups.
/// Returns intentionally suboptimal labels to show contrast.
fn naive_assignment(n_total: usize, k: usize) -> Array1<usize> {
    Array1::from((0..n_total).map(|i| i * k / n_total).collect::<Vec<_>>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Integration: Cluster Assignments → Clustering Metrics ===\n");

    // ─── 1. Generate simulated cluster data ──────────────────────────────────
    let k = 4;
    let n_per = 25; // 25 points per cluster → 100 total
    let spread = 0.8;

    let (x, true_labels) = make_grid_clusters(k, n_per, spread);
    let n_total = k * n_per;

    // "Good" assignment = ground-truth labels (as k-means would ideally produce)
    // "Naive" assignment = sequential block split (poor quality)
    let naive_labels = naive_assignment(n_total, k);

    println!("Simulated k-means on {n_total} samples, k={k} clusters");
    println!("Grid layout: clusters placed 3 units apart, spread={spread:.1}\n");

    // ─── 2. Compute internal clustering metrics ───────────────────────────────
    // Internal metrics: no ground truth needed
    let sil_good: f64 = silhouette_score(&x, &true_labels, "euclidean")?;
    let sil_naive: f64 = silhouette_score(&x, &naive_labels, "euclidean")?;

    let db_good: f64 = davies_bouldin_score(&x, &true_labels)?;
    let db_naive: f64 = davies_bouldin_score(&x, &naive_labels)?;

    let ch_good: f64 = calinski_harabasz_score(&x, &true_labels)?;
    let ch_naive: f64 = calinski_harabasz_score(&x, &naive_labels)?;

    // ─── 3. Print metrics comparison ──────────────────────────────────────────
    println!(
        "{:<35} {:>12}  {:>12}",
        "Internal Metric", "Good (k-means)", "Naive (bad)"
    );
    println!("{}", "-".repeat(63));

    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Silhouette (↑ better, max=1)", sil_good, sil_naive
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Davies-Bouldin (↓ better)", db_good, db_naive
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}",
        "Calinski-Harabasz (↑ better)", ch_good, ch_naive
    );

    // ─── 4. Elbow-method sweep ────────────────────────────────────────────────
    println!("\n--- Elbow-method: Silhouette vs k (true cluster structure) ---");
    println!("{:<10} {:>15}", "k", "Silhouette");
    println!("{}", "-".repeat(28));

    for trial_k in 2..=6usize {
        let (x_trial, labels_trial) = make_grid_clusters(trial_k, n_per, spread);
        let sil: f64 = silhouette_score(&x_trial, &labels_trial, "euclidean")?;
        let marker = if trial_k == k { " ← true k" } else { "" };
        println!("{:<10} {:>15.4}{}", trial_k, sil, marker);
    }

    println!("\n--- Tips for pipeline integration ---");
    println!("  1. Run clustering (k-means, DBSCAN) to obtain `labels: Array1<usize>`");
    println!("  2. Pass (`x`, `labels`) to silhouette_score / davies_bouldin_score");
    println!("  3. If ground truth is available, use adjusted_rand_index from `clustering::external_metrics`");
    println!("  4. Plot Silhouette vs k to find the optimal number of clusters");

    println!("\n=== Done ===");
    Ok(())
}
