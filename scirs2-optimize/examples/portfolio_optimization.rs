//! Portfolio Optimization Example — SciRS2 Optimize
//!
//! Markowitz mean-variance portfolio optimization:
//!   1. Generate a synthetic return covariance matrix and expected returns
//!   2. Minimize portfolio variance subject to:
//!        - weights sum to 1  (equality)
//!        - expected return >= target  (inequality)
//!        - each weight >= 0  (long-only via inequality)
//!   3. Sweep over risk levels to trace the efficient frontier
//!
//! Run with: cargo run -p scirs2-optimize --example portfolio_optimization

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_optimize::constrained::{minimize_constrained, Constraint, ConstraintKind, Method};
use scirs2_optimize::OptimizeError;

// ------------------------------------------------------------------ //
//  Synthetic market data                                              //
// ------------------------------------------------------------------ //

/// Generate a positive-definite covariance matrix from random correlations.
fn synthetic_covariance(n: usize) -> Array2<f64> {
    // Volatilities roughly between 10 % and 40 %
    let vols: Vec<f64> = (0..n)
        .map(|i| 0.10 + 0.30 * (i as f64 / (n - 1) as f64))
        .collect();

    // Build a random correlation matrix via a factor model:
    //   Corr = F F^T + diag(1 - F_rowsq)
    // where F is n×k with entries in [0, 0.5].
    let k = 2usize; // two common factors
                    // Deterministic factor loadings using a simple pattern
    let factor: Vec<f64> = (0..n * k)
        .map(|idx| {
            let i = idx / k;
            let j = idx % k;
            0.3 + 0.2 * ((i * 7 + j * 13) % 5) as f64 / 4.0
        })
        .collect();

    let mut cov = Array2::zeros((n, n));
    for i in 0..n {
        for jj in 0..n {
            let mut corr = 0.0_f64;
            for f in 0..k {
                corr += factor[i * k + f] * factor[jj * k + f];
            }
            // Clip to valid correlation range
            let corr = corr.min(0.95);
            cov[[i, jj]] = corr * vols[i] * vols[jj];
        }
        // Ensure diagonal = vol^2
        cov[[i, i]] = vols[i] * vols[i];
    }
    cov
}

/// Expected annual returns (synthetic): ranging from 5 % to 20 %.
fn synthetic_returns(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| 0.05 + 0.15 * (i as f64 / (n - 1) as f64))
}

// ------------------------------------------------------------------ //
//  Objective: portfolio variance = w^T Σ w                           //
// ------------------------------------------------------------------ //

fn portfolio_variance(w: &[f64], cov: &Array2<f64>) -> f64 {
    let n = w.len();
    let mut var = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            var += w[i] * cov[[i, j]] * w[j];
        }
    }
    var
}

// ------------------------------------------------------------------ //
//  Efficient frontier sweep                                           //
// ------------------------------------------------------------------ //

fn optimize_portfolio(
    cov: &Array2<f64>,
    mu: &Array1<f64>,
    target_return: f64,
) -> Result<Array1<f64>, OptimizeError> {
    let n = mu.len();

    // --- objective (clone-able closures via captured references) ---
    let cov_ref = cov.clone();
    let objective = move |w: &[f64]| portfolio_variance(w, &cov_ref);

    // --- constraints ---
    // 1. Σ wᵢ = 1  (equality)
    let eq_con: fn(&[f64]) -> f64 = |w: &[f64]| w.iter().sum::<f64>() - 1.0;

    // 2. Σ μᵢ wᵢ ≥ target_return  →  (Σ μᵢ wᵢ - target) ≥ 0  (inequality)
    let mu_owned = mu.to_owned();
    let mu_slice: Vec<f64> = mu_owned.to_vec();
    let mu_for_closure = mu_slice.clone();
    let ret_con: fn(&[f64]) -> f64 = {
        // We can't capture mu in a fn pointer, so we rebuild via a global static trick.
        // Instead we use a static cell approach: just hard-code via a closure cast.
        // Since ConstraintFn = fn(&[f64]) -> f64 (bare fn), we compute the closure inline.
        // Solution: use a thread_local to pass mu into the fn pointer.
        |_: &[f64]| 0.0 // placeholder — we implement via minimize_slsqp directly below
    };
    // Because ConstraintFn is a bare fn pointer we cannot capture environment;
    // use minimize_slsqp directly with arc closures wrapped in a trait object approach.
    // For this example, we use the SLSQP interface which accepts bare fn.
    // We work around the capture problem by embedding target_return and mu in a thread_local.
    std::thread_local! {
        static MU_TARGET: std::cell::RefCell<(Vec<f64>, f64)> =
            const { std::cell::RefCell::new((Vec::new(), 0.0)) };
    }
    MU_TARGET.with(|cell| {
        *cell.borrow_mut() = (mu_for_closure.clone(), target_return);
    });

    let return_constraint: fn(&[f64]) -> f64 = |w| {
        MU_TARGET.with(|cell| {
            let borrow = cell.borrow();
            let (mu_v, tgt) = &*borrow;
            let ret: f64 = mu_v.iter().zip(w.iter()).map(|(m, wi)| m * wi).sum();
            ret - tgt
        })
    };

    // 3. wᵢ ≥ 0  (one fn per asset, long-only)
    //    We encode all non-negativity via a single inequality function min(w) ≥ 0.
    let nonneg_con: fn(&[f64]) -> f64 = |w| w.iter().cloned().fold(f64::INFINITY, f64::min);

    let constraints: Vec<Constraint> = vec![
        Constraint::new(eq_con, ConstraintKind::Equality),
        Constraint::new(return_constraint, ConstraintKind::Inequality),
        Constraint::new(nonneg_con, ConstraintKind::Inequality),
    ];

    // Initial guess: equal weights
    let x0 = Array1::from_elem(n, 1.0 / n as f64);

    let result = minimize_constrained(objective, &x0, &constraints, Method::SLSQP, None)?;
    Ok(result.x)
}

// ------------------------------------------------------------------ //
//  main                                                               //
// ------------------------------------------------------------------ //

fn main() {
    const N_ASSETS: usize = 6;
    println!("=== SciRS2 Markowitz Portfolio Optimization ===\n");

    let cov = synthetic_covariance(N_ASSETS);
    let mu = synthetic_returns(N_ASSETS);

    // Print asset summary
    println!("Asset parameters:");
    println!("{:<8} {:>12} {:>12}", "Asset", "Exp. Return", "Volatility");
    println!("{}", "-".repeat(36));
    for i in 0..N_ASSETS {
        println!(
            "{:<8} {:>11.2}% {:>11.2}%",
            format!("A{}", i + 1),
            mu[i] * 100.0,
            cov[[i, i]].sqrt() * 100.0
        );
    }
    println!();

    // Efficient frontier: sweep target returns from min to max
    let mu_min = mu[0];
    let mu_max = mu[N_ASSETS - 1];
    let n_points = 8usize;

    println!("--- Efficient Frontier ---");
    println!("{:>14}  {:>14}  Weights", "Target Return", "Port. Vol");
    println!("{}", "-".repeat(72));

    for k in 0..n_points {
        let target = mu_min + (mu_max - mu_min) * (k as f64 / (n_points - 1) as f64);
        match optimize_portfolio(&cov, &mu, target) {
            Ok(weights) => {
                let vol = portfolio_variance(weights.as_slice().expect("contiguous"), &cov).sqrt();
                let weight_str: Vec<String> = weights
                    .iter()
                    .map(|w| format!("{:.3}", w.max(0.0)))
                    .collect();
                println!(
                    "{:>13.2}%  {:>13.2}%  [{}]",
                    target * 100.0,
                    vol * 100.0,
                    weight_str.join(", ")
                );
            }
            Err(e) => {
                println!("{:>13.2}%  optimization failed: {}", target * 100.0, e);
            }
        }
    }

    // Minimum-variance portfolio (no return constraint)
    println!("\n--- Minimum-Variance Portfolio ---");
    let cov_mv = cov.clone();
    let obj_mv = move |w: &[f64]| portfolio_variance(w, &cov_mv);
    let eq_mv: fn(&[f64]) -> f64 = |w| w.iter().sum::<f64>() - 1.0;
    let nn_mv: fn(&[f64]) -> f64 = |w| w.iter().cloned().fold(f64::INFINITY, f64::min);
    let cons_mv: Vec<Constraint> = vec![
        Constraint::new(eq_mv, ConstraintKind::Equality),
        Constraint::new(nn_mv, ConstraintKind::Inequality),
    ];
    let x0_mv = Array1::from_elem(N_ASSETS, 1.0 / N_ASSETS as f64);
    match minimize_constrained(obj_mv, &x0_mv, &cons_mv, Method::SLSQP, None) {
        Ok(w_mv) => {
            let vol_mv = portfolio_variance(w_mv.x.as_slice().expect("contiguous"), &cov).sqrt();
            let ret_mv: f64 = mu.iter().zip(w_mv.x.iter()).map(|(m, w)| m * w).sum();
            println!("Minimum variance portfolio:");
            println!("  Return   : {:.2}%", ret_mv * 100.0);
            println!("  Volatility: {:.2}%", vol_mv * 100.0);
            print!("  Weights  : [");
            for (i, w) in w_mv.x.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.3}", w.max(0.0_f64));
            }
            println!("]");
        }
        Err(e) => println!("Min-variance optimization failed: {}", e),
    }

    println!("\nDone.");
}
