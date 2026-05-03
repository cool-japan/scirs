//! Framework Integration: Transformation Outputs → Distribution Metrics
//!
//! This example simulates the output pattern of a data transformation pipeline
//! (as would come from scirs2-transform or a feature extractor) and evaluates
//! how well the transformed distribution matches a reference using scirs2-metrics.
//!
//! Typical use cases:
//!  - Domain adaptation: does transformed source domain match target?
//!  - Data augmentation: does augmented set preserve original statistics?
//!  - Normalisation: does the transformed data follow the expected distribution?
//!
//! In a real pipeline you would replace the synthetic data with actual
//! scirs2-transform or scirs2-neural embedding outputs.
//!
//! Run with: `cargo run --example integration_transform -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::anomaly::{
    js_divergence, kl_divergence, maximum_mean_discrepancy, wasserstein_distance,
};
use scirs2_metrics::distance::{cosine_similarity, euclidean_distance};

/// Simulates a feature extractor: maps raw indices → embedding values.
/// In production this would be a neural encoder or a learned transform.
fn simulate_feature_extractor(n: usize, shift: f64, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64);
            scale * (std::f64::consts::PI * x).sin() + shift
        })
        .collect()
}

/// Build a histogram (probability mass function) from continuous samples.
fn histogram(samples: &[f64], n_bins: usize) -> Vec<f64> {
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let width = (max - min) / n_bins as f64;

    let mut counts = vec![0usize; n_bins];
    for &v in samples {
        let bin = ((v - min) / width) as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }

    let total = samples.len() as f64;
    counts.iter().map(|&c| c as f64 / total).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Integration: Transformation Outputs → Distribution Metrics ===\n");

    let n: usize = 200;
    let n_bins: usize = 20;

    // ─── 1. Reference distribution (source domain) ────────────────────────────
    let source = simulate_feature_extractor(n, 0.0, 1.0);

    // ─── 2. Simulated transformation outputs ──────────────────────────────────
    // Good transform: small shift (domain adaptation did well)
    let transformed_good = simulate_feature_extractor(n, 0.05, 1.02);
    // Bad transform: large shift + scale change (domain gap not bridged)
    let transformed_bad = simulate_feature_extractor(n, 1.5, 2.0);

    // ─── 3. Build histograms for discrete divergence metrics ─────────────────
    // (KL/JS divergences require PMFs; add small epsilon to avoid log(0))
    let eps = 1e-9;
    let source_hist: Vec<f64> = histogram(&source, n_bins)
        .into_iter()
        .map(|v| v + eps)
        .collect();
    let good_hist: Vec<f64> = histogram(&transformed_good, n_bins)
        .into_iter()
        .map(|v| v + eps)
        .collect();
    let bad_hist: Vec<f64> = histogram(&transformed_bad, n_bins)
        .into_iter()
        .map(|v| v + eps)
        .collect();

    // Renormalize after epsilon addition
    let renorm = |h: Vec<f64>| -> Vec<f64> {
        let s: f64 = h.iter().sum();
        h.into_iter().map(|v| v / s).collect()
    };
    let source_pmf = renorm(source_hist);
    let good_pmf = renorm(good_hist);
    let bad_pmf = renorm(bad_hist);

    // ─── 4. Convert to ndarray for anomaly module ─────────────────────────────
    let src_arr = Array1::from(source.clone());
    let good_arr = Array1::from(transformed_good.clone());
    let bad_arr = Array1::from(transformed_bad.clone());

    let src_pmf_arr = Array1::from(source_pmf.clone());
    let good_pmf_arr = Array1::from(good_pmf.clone());
    let bad_pmf_arr = Array1::from(bad_pmf.clone());

    // ─── 5. Compute distribution metrics ──────────────────────────────────────
    // Wasserstein on raw samples (distribution shift in embedding space)
    let w1_good: f64 = wasserstein_distance(&src_arr, &good_arr)?;
    let w1_bad: f64 = wasserstein_distance(&src_arr, &bad_arr)?;

    // MMD on raw samples (kernel-based two-sample test)
    let mmd_good: f64 = maximum_mean_discrepancy(&src_arr, &good_arr, None)?;
    let mmd_bad: f64 = maximum_mean_discrepancy(&src_arr, &bad_arr, None)?;

    // KL / JS divergence on histograms (information-theoretic)
    let kl_good: f64 = kl_divergence(&src_pmf_arr, &good_pmf_arr)?;
    let kl_bad: f64 = kl_divergence(&src_pmf_arr, &bad_pmf_arr)?;
    let js_good: f64 = js_divergence(&src_pmf_arr, &good_pmf_arr)?;
    let js_bad: f64 = js_divergence(&src_pmf_arr, &bad_pmf_arr)?;

    // Point-wise distance metrics (first point as representative vector)
    let src_pt = Array1::from(source_pmf.clone());
    let good_pt = Array1::from(good_pmf.clone());
    let bad_pt = Array1::from(bad_pmf.clone());

    let euc_good: f64 = euclidean_distance(&src_pt, &good_pt)?;
    let euc_bad: f64 = euclidean_distance(&src_pt, &bad_pt)?;
    let cos_good: f64 = cosine_similarity(&src_pt, &good_pt)?;
    let cos_bad: f64 = cosine_similarity(&src_pt, &bad_pt)?;

    // ─── 6. Print results ────────────────────────────────────────────────────
    println!("Source: sin-feature extractor, n={n}");
    println!("Good transform: shift=0.05, scale=1.02 (small domain gap)");
    println!("Bad  transform: shift=1.5,  scale=2.0  (large domain gap)\n");

    println!(
        "{:<38} {:>12}  {:>12}",
        "Metric (source vs transform)", "Good", "Bad"
    );
    println!("{}", "-".repeat(66));

    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "Wasserstein-1 (↓ → aligned)", w1_good, w1_bad
    );
    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "MMD (↓ → aligned)", mmd_good, mmd_bad
    );
    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "KL divergence KL(P||Q) (↓)", kl_good, kl_bad
    );
    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "JS divergence (↓, max=ln2)", js_good, js_bad
    );
    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "Euclidean dist (histogram)", euc_good, euc_bad
    );
    println!(
        "{:<38} {:>12.6}  {:>12.6}",
        "Cosine similarity (↑ → similar)", cos_good, cos_bad
    );

    println!("\n--- Tips for transform pipeline integration ---");
    println!("  Use Wasserstein or MMD to detect dataset shift before/after transform");
    println!("  KL/JS divergence works on histogram representations of embeddings");
    println!("  Cosine similarity on aggregate histograms captures shape similarity");
    println!("  Threshold: Wasserstein > 0.1 typically indicates meaningful shift");

    println!("\n=== Done ===");
    Ok(())
}
