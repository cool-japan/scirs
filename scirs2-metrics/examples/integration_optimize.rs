//! Framework Integration: Optimizer Convergence → Regression Metrics
//!
//! This example simulates the output pattern of an iterative optimizer (as would
//! come from scirs2-optimize) — recording per-iteration predictions — and
//! evaluates regression quality at each stage using scirs2-metrics.
//!
//! In a real pipeline you would replace the synthetic trajectory with actual
//! predictions from a scirs2-optimize solver (e.g. gradient descent, L-BFGS).
//!
//! Run with: `cargo run --example integration_optimize -p scirs2-metrics`

use scirs2_core::ndarray::Array1;
use scirs2_metrics::regression::{mean_absolute_error, mean_squared_error, r2_score};

/// Simulates an optimizer converging on a linear regression problem.
///
/// Returns a list of (iteration, predictions) tuples where predictions improve
/// as the optimizer takes more steps toward the minimum.
fn simulate_optimizer_trajectory(y_true: &[f64], n_iterations: usize) -> Vec<(usize, Vec<f64>)> {
    let n = y_true.len();
    let mut trajectory = Vec::with_capacity(n_iterations);

    // Initial "guess": constant prediction = mean of a naive guess (0.0)
    // The optimizer progressively improves toward the true values.
    // We model convergence as: y_pred[iter] = y_true * progress + (1-progress) * y_naive
    // where progress = 1 - (1 - learning_rate)^iter
    let y_naive: Vec<f64> = vec![0.0; n]; // initial guess
    let learning_rate: f64 = 0.15;

    for iter in 0..=n_iterations {
        let progress = 1.0 - (1.0 - learning_rate).powi(iter as i32);
        let preds: Vec<f64> = y_true
            .iter()
            .zip(y_naive.iter())
            .map(|(&yt, &yn)| yn + progress * (yt - yn))
            .collect();
        trajectory.push((iter, preds));
    }

    trajectory
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Integration: Optimizer Convergence → Regression Metrics ===\n");

    // ─── 1. Define target regression problem ──────────────────────────────────
    let n: usize = 100;
    // True signal: quadratic with noise
    let y_true_vec: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64);
            3.0 * x * x - 2.0 * x + 1.0 + 0.05 * ((i as f64 * 1.23).sin())
        })
        .collect();

    // ─── 2. Run simulated optimizer ───────────────────────────────────────────
    let n_iter = 30;
    let trajectory = simulate_optimizer_trajectory(&y_true_vec, n_iter);
    let y_true = Array1::from(y_true_vec.clone());

    // ─── 3. Compute metrics at key iterations ────────────────────────────────
    let checkpoints = [0, 1, 2, 5, 10, 20, 30];

    println!("Problem: quadratic regression, n={n} samples");
    println!("Optimizer: gradient descent simulation, {n_iter} iterations\n");

    println!(
        "{:<10} {:>12}  {:>12}  {:>12}",
        "Iteration", "MSE", "MAE", "R²"
    );
    println!("{}", "-".repeat(52));

    for &iter in &checkpoints {
        let (_, preds) = &trajectory[iter];
        let y_pred = Array1::from(preds.clone());
        let mse = mean_squared_error(&y_true, &y_pred)?;
        let mae = mean_absolute_error(&y_true, &y_pred)?;
        let r2 = r2_score(&y_true, &y_pred)?;

        println!("{:<10} {:>12.6}  {:>12.6}  {:>12.6}", iter, mse, mae, r2);
    }

    // ─── 4. Print convergence summary ────────────────────────────────────────
    let (_, final_preds) = &trajectory[n_iter];
    let y_final = Array1::from(final_preds.clone());
    let final_r2 = r2_score(&y_true, &y_final)?;

    println!("\n--- Convergence Summary ---");
    println!("  Initial R²: {:.4}", {
        let (_, init) = &trajectory[0];
        r2_score(&y_true, &Array1::from(init.clone()))?
    });
    println!("  Final  R²: {:.4} (after {} iterations)", final_r2, n_iter);

    if final_r2 > 0.99 {
        println!("  Status: Converged — R² > 0.99");
    } else if final_r2 > 0.9 {
        println!("  Status: Near-converged — may benefit from more iterations");
    } else {
        println!("  Status: Not yet converged — consider tuning step size");
    }

    println!("\n--- Tips for optimizer metric tracking ---");
    println!("  Log MSE/MAE each iteration to detect divergence early");
    println!("  R² > 0.99 typically indicates sufficient convergence for regression");
    println!("  Compare MAE trend: flat MAE after many iterations = local minimum");

    println!("\n=== Done ===");
    Ok(())
}
