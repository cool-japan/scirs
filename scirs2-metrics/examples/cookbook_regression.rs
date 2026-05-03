//! Cookbook: Regression Metrics
//!
//! This example demonstrates how to evaluate regression models using scirs2-metrics.
//! It covers MSE, RMSE, MAE, R², MAPE, and explained variance on synthetic data.
//!
//! Run with: `cargo run --example cookbook_regression -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score, root_mean_squared_error,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cookbook: Regression Metrics ===\n");

    // ─── 1. Synthetic regression data ───────────────────────────────────────
    // True values: y = 2x + noise_free baseline
    // Predictions: y_hat = 2x + small_error  (good model)
    //             y_baseline = mean(y)        (naive baseline)
    let n: usize = 30;

    let y_true_vec: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 5.0).collect();
    // Good model: small additive error growing slightly with i
    let y_pred_good_vec: Vec<f64> = (0..n)
        .map(|i| 2.0 * i as f64 + 5.0 + ((i as f64 * 0.5).sin()) * 2.0)
        .collect();
    // Mediocre model: larger systematic bias
    let y_pred_bad_vec: Vec<f64> = (0..n)
        .map(|i| 1.5 * i as f64 + 8.0 + ((i as f64 * 0.5).cos()) * 6.0)
        .collect();

    let y_true = Array1::from(y_true_vec.clone());
    let y_good = Array1::from(y_pred_good_vec);
    let y_bad = Array1::from(y_pred_bad_vec);

    // ─── 2. Compute metrics for both models ─────────────────────────────────
    let mse_good = mean_squared_error(&y_true, &y_good)?;
    let mse_bad = mean_squared_error(&y_true, &y_bad)?;

    let rmse_good = root_mean_squared_error(&y_true, &y_good)?;
    let rmse_bad = root_mean_squared_error(&y_true, &y_bad)?;

    let mae_good = mean_absolute_error(&y_true, &y_good)?;
    let mae_bad = mean_absolute_error(&y_true, &y_bad)?;

    let r2_good = r2_score(&y_true, &y_good)?;
    let r2_bad = r2_score(&y_true, &y_bad)?;

    let mape_good = mean_absolute_percentage_error(&y_true, &y_good)?;
    let mape_bad = mean_absolute_percentage_error(&y_true, &y_bad)?;

    let ev_good = explained_variance_score(&y_true, &y_good)?;
    let ev_bad = explained_variance_score(&y_true, &y_bad)?;

    // ─── 3. Print results with commentary ───────────────────────────────────
    println!("Dataset: n={n} samples, y = 2x + 5 (linear signal)\n");
    println!("{:<30} {:>12}  {:>12}", "Metric", "Good model", "Bad model");
    println!("{}", "-".repeat(58));

    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "MSE (↓ better)", mse_good, mse_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "RMSE (↓ better, same units)", rmse_good, rmse_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "MAE (↓ better, robust)", mae_good, mae_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "R² (↑ better, max=1)", r2_good, r2_bad
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "MAPE % (↓ better)",
        mape_good * 100.0,
        mape_bad * 100.0
    );
    println!(
        "{:<30} {:>12.4}  {:>12.4}",
        "Explained Variance (↑, max=1)", ev_good, ev_bad
    );

    println!("\n--- When to use which metric ---");
    println!("  MSE        : penalises large errors heavily (outlier-sensitive)");
    println!("  RMSE       : like MSE but in the same units as the target");
    println!("  MAE        : robust to outliers, interpretable in target units");
    println!("  R²         : proportion of variance explained; context-free comparison");
    println!("  MAPE       : interpretable as % error; fails when y_true ≈ 0");
    println!("  Expl. Var. : like R² but ignores mean offset (use when bias is unimportant)");

    println!("\n=== Done ===");
    Ok(())
}
