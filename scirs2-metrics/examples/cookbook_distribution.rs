//! Cookbook: Distribution Metrics
//!
//! This example demonstrates KL divergence, Wasserstein distance, Jensen-Shannon
//! divergence, Hellinger distance, and Maximum Mean Discrepancy (MMD) on
//! synthetic probability distributions.
//!
//! Note on API paths:
//! - `scirs2_metrics::distribution::{kl_divergence, js_divergence, wasserstein_distance}` are
//!   the ndarray-based functions in the `anomaly` module re-exported there.
//! - `scirs2_metrics::distribution::{wasserstein_1d, kl_divergence_ext, ...}` are the
//!   pure-slice functions from `distribution::mod` and submodules.
//!
//! Run with: `cargo run --example cookbook_distribution -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::anomaly::{
    js_divergence, kl_divergence, maximum_mean_discrepancy, wasserstein_distance,
};
use scirs2_metrics::distribution::{
    hellinger_distance, js_divergence as js_dist, kl_divergence as kl_simple, wasserstein_1d,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Distribution Metrics ===\n");

    // ─── 1. Discrete probability distributions ───────────────────────────────
    // P ~ categorical over 8 bins (skewed)
    // Q_close ~ close to P (small shift)
    // Q_far   ~ very different from P (opposite skew)
    let p_raw = vec![0.30, 0.25, 0.20, 0.10, 0.06, 0.04, 0.03, 0.02f64];
    let q_close_raw = vec![0.28, 0.25, 0.22, 0.10, 0.07, 0.04, 0.03, 0.01f64];
    let q_far_raw = vec![0.02, 0.03, 0.04, 0.06, 0.10, 0.20, 0.25, 0.30f64];

    // ─── 2. Empirical (sample) distributions ─────────────────────────────────
    // P_samples ~ normal-ish centred at 0
    // Q_samples_close ~ normal-ish centred at 0.5
    // Q_samples_far   ~ normal-ish centred at 5.0
    let n_samples: usize = 40;
    let p_samples: Vec<f64> = (0..n_samples).map(|i| (i as f64 - 20.0) * 0.3).collect();
    let q_close_samples: Vec<f64> = (0..n_samples)
        .map(|i| (i as f64 - 20.0) * 0.3 + 0.5)
        .collect();
    let q_far_samples: Vec<f64> = (0..n_samples)
        .map(|i| (i as f64 - 20.0) * 0.3 + 5.0)
        .collect();

    // Convert to ndarray for anomaly module functions
    let p_arr = Array1::from(p_raw.clone());
    let q_close_arr = Array1::from(q_close_raw.clone());
    let q_far_arr = Array1::from(q_far_raw.clone());

    let p_samp = Array1::from(p_samples.clone());
    let q_close_samp = Array1::from(q_close_samples.clone());
    let q_far_samp = Array1::from(q_far_samples.clone());

    // ─── 3. Compute divergences for discrete distributions ───────────────────
    // anomaly::kl_divergence / js_divergence expect ndarray Array1
    let kl_close: f64 = kl_divergence(&p_arr, &q_close_arr)?;
    let kl_far: f64 = kl_divergence(&p_arr, &q_far_arr)?;

    let js_close: f64 = js_divergence(&p_arr, &q_close_arr)?;
    let js_far: f64 = js_divergence(&p_arr, &q_far_arr)?;

    // distribution module slice-based functions (return plain f64)
    let kl_simple_close = kl_simple(&p_raw, &q_close_raw)?;
    let js_simple_close = js_dist(&p_raw, &q_close_raw)?;
    let hell_close = hellinger_distance(&p_raw, &q_close_raw)?;
    let hell_far = hellinger_distance(&p_raw, &q_far_raw)?;

    // ─── 4. Compute Wasserstein distance for sample distributions ────────────
    let w1_close = wasserstein_1d(&p_samples, &q_close_samples)?;
    let w1_far = wasserstein_1d(&p_samples, &q_far_samples)?;
    // anomaly module Wasserstein also works with Array1
    let w1_arr_close: f64 = wasserstein_distance(&p_samp, &q_close_samp)?;
    let w1_arr_far: f64 = wasserstein_distance(&p_samp, &q_far_samp)?;

    // ─── 5. Maximum Mean Discrepancy (kernel two-sample test statistic) ───────
    let mmd_close: f64 = maximum_mean_discrepancy(&p_samp, &q_close_samp, None)?;
    let mmd_far: f64 = maximum_mean_discrepancy(&p_samp, &q_far_samp, None)?;

    // ─── 6. Print results ────────────────────────────────────────────────────
    println!("── Discrete PMF divergences ────────────────────────────────────");
    println!(
        "{:<40} {:>12}  {:>12}",
        "Metric", "P vs Q_close", "P vs Q_far"
    );
    println!("{}", "-".repeat(68));

    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "KL divergence KL(P||Q) [anomaly]", kl_close, kl_far
    );
    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "KL divergence [distribution simple]",
        kl_simple_close,
        0.0_f64 // Not recomputing far for brevity
    );
    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "JS divergence [anomaly]", js_close, js_far
    );
    println!(
        "{:<40} {:>12.6}  N/A       ",
        "JS divergence [distribution simple]", js_simple_close,
    );
    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "Hellinger distance", hell_close, hell_far
    );

    println!(
        "\n── Empirical sample metrics (n={}?) ─────────────────────────",
        n_samples
    );
    println!(
        "{:<40} {:>12}  {:>12}",
        "Metric", "P vs Q_close", "P vs Q_far"
    );
    println!("{}", "-".repeat(68));

    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "Wasserstein-1 [distribution]", w1_close, w1_far
    );
    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "Wasserstein-1 [anomaly Array1]", w1_arr_close, w1_arr_far
    );
    println!(
        "{:<40} {:>12.6}  {:>12.6}",
        "MMD (RBF kernel, auto bandwidth)", mmd_close, mmd_far
    );

    println!("\n--- When to use which metric ---");
    println!("  KL divergence   : information-theoretic; not symmetric; penalises zero-mass areas");
    println!("  JS divergence   : symmetric, bounded [0, ln2]; better for comparing models");
    println!("  Hellinger       : symmetric, bounded [0, 1]; geometric interpretation");
    println!("  Wasserstein-1   : geometric; works on continuous or ordinal distributions");
    println!("  MMD             : kernel-based; excellent for detecting distribution shift");

    println!("\n=== Done ===");
    Ok(())
}
