//! Cookbook: Calibration Metrics
//!
//! This example demonstrates Expected Calibration Error (ECE), Maximum
//! Calibration Error (MCE), Brier Score, and Brier Skill Score on synthetic
//! probability calibration data.
//!
//! Run with: `cargo run --example cookbook_calibration -p scirs2-metrics`

use scirs2_metrics::calibration::{
    brier_score, brier_skill_score, expected_calibration_error, log_loss,
    maximum_calibration_error, reliability_diagram_data,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Calibration Metrics ===\n");

    // ─── 1. Synthetic calibration data ──────────────────────────────────────
    // A well-calibrated classifier outputs probabilities that match observed
    // frequencies. If the model says P=0.8, about 80% of those examples should
    // actually be positive.
    //
    // We generate three scenarios:
    //   - perfect: predicted prob matches empirical frequency exactly
    //   - overconfident: model pushes probs toward 0 and 1 (too sharp)
    //   - underconfident: probs stay near 0.5 regardless of true label

    let n: usize = 50;

    // Ground truth: roughly 50% positive
    let y_true: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

    // Perfect calibration: probabilities ~ fraction of positives in that region
    let y_prob_perfect: Vec<f64> = (0..n)
        .map(|i| {
            // Alternate between 0.8 (should be ~80% correct) and 0.2 (20% correct)
            // Since y_true[i]=1.0 when i is even, use 0.8 for even, 0.2 for odd
            if i % 2 == 0 {
                0.80
            } else {
                0.20
            }
        })
        .collect();

    // Overconfident model: pushes probs further toward extremes
    let y_prob_overconfident: Vec<f64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                0.98 // over-commits to positives
            } else {
                0.02 // over-commits to negatives
            }
        })
        .collect();

    // Underconfident model: probs close to 0.5 regardless
    let y_prob_underconfident: Vec<f64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                0.60 // weakly positive
            } else {
                0.40 // weakly negative
            }
        })
        .collect();

    // ─── 2. Compute calibration metrics ──────────────────────────────────────
    let n_bins = 10;

    let ece_perfect = expected_calibration_error(&y_true, &y_prob_perfect, n_bins)?;
    let ece_over = expected_calibration_error(&y_true, &y_prob_overconfident, n_bins)?;
    let ece_under = expected_calibration_error(&y_true, &y_prob_underconfident, n_bins)?;

    let mce_perfect = maximum_calibration_error(&y_true, &y_prob_perfect, n_bins)?;
    let mce_over = maximum_calibration_error(&y_true, &y_prob_overconfident, n_bins)?;
    let mce_under = maximum_calibration_error(&y_true, &y_prob_underconfident, n_bins)?;

    let brier_perfect = brier_score(&y_true, &y_prob_perfect)?;
    let brier_over = brier_score(&y_true, &y_prob_overconfident)?;
    let brier_under = brier_score(&y_true, &y_prob_underconfident)?;

    let bss_perfect = brier_skill_score(&y_true, &y_prob_perfect)?;
    let bss_over = brier_skill_score(&y_true, &y_prob_overconfident)?;
    let bss_under = brier_skill_score(&y_true, &y_prob_underconfident)?;

    let ll_perfect = log_loss(&y_true, &y_prob_perfect)?;
    let ll_over = log_loss(&y_true, &y_prob_overconfident)?;
    let ll_under = log_loss(&y_true, &y_prob_underconfident)?;

    // ─── 3. Print calibration metrics table ──────────────────────────────────
    println!("Dataset: n={n} samples, 50% positives, 3 probability models\n");
    println!(
        "{:<35} {:>12}  {:>12}  {:>12}",
        "Metric", "Well-cal.", "Overconf.", "Underconf."
    );
    println!("{}", "-".repeat(77));

    println!(
        "{:<35} {:>12.4}  {:>12.4}  {:>12.4}",
        "ECE (↓ better, min=0)", ece_perfect, ece_over, ece_under
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}  {:>12.4}",
        "MCE (↓ better, worst-bin ECE)", mce_perfect, mce_over, mce_under
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}  {:>12.4}",
        "Brier Score (↓ better, max=1)", brier_perfect, brier_over, brier_under
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}  {:>12.4}",
        "Brier Skill Score (↑ better)", bss_perfect, bss_over, bss_under
    );
    println!(
        "{:<35} {:>12.4}  {:>12.4}  {:>12.4}",
        "Log-Loss (↓ better)", ll_perfect, ll_over, ll_under
    );

    // ─── 4. Reliability diagram bins ─────────────────────────────────────────
    println!("\n--- Reliability diagram bins (well-calibrated model) ---");
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Bin centre", "Mean conf.", "Fraction pos", "Count"
    );
    println!("{}", "-".repeat(52));

    let diagram = reliability_diagram_data(&y_true, &y_prob_perfect, n_bins)?;
    for bin in &diagram.bins {
        if bin.count > 0 {
            println!(
                "{:<12.2} {:>12.4} {:>12.4} {:>12}",
                (bin.bin_lower + bin.bin_upper) * 0.5,
                bin.mean_predicted,
                bin.fraction_positive,
                bin.count
            );
        }
    }

    println!("\n--- When to use which metric ---");
    println!("  ECE        : average calibration gap (standard for NLP/vision models)");
    println!("  MCE        : worst-bin calibration; use when tails matter");
    println!("  Brier Score: proper scoring rule; combines sharpness + calibration");
    println!("  BSS        : Brier improvement over climatology baseline (>0 = useful)");
    println!("  Log-Loss   : penalises overconfident wrong predictions exponentially");

    println!("\n=== Done ===");
    Ok(())
}
