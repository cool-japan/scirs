//! Comprehensive example of all least squares methods in scirs2-optimize
//!
//! This demonstrates:
//! - Standard least squares
//! - Robust least squares (for outliers)  
//! - Weighted least squares (for heteroscedastic data)
//! - Bounded least squares (with constraints)

use ndarray::{array, Array1, Array2};
use scirs2_optimize::{
    least_squares::{
        bounded_least_squares, least_squares, robust_least_squares, weighted_least_squares,
        BisquareLoss, HuberLoss, Method,
    },
    Bounds,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comprehensive Least Squares Example");
    println!("==================================");
    println!();

    // Generate synthetic data: y = 1.5 + 0.5*x + noise
    let x_data = array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let true_intercept = 1.5;
    let true_slope = 0.5;

    // Clean data with small noise
    let y_clean = &x_data * true_slope
        + true_intercept
        + array![0.1, -0.15, 0.05, -0.1, 0.12, -0.08, 0.03, -0.06, 0.09, -0.11, 0.07];

    // Data with outliers
    let mut y_outliers = y_clean.clone();
    y_outliers[3] = 10.0; // Outlier
    y_outliers[7] = -2.0; // Outlier

    // Data with varying variance (heteroscedastic)
    let y_hetero = &x_data * true_slope
        + true_intercept
        + array![0.1, -0.2, 0.3, -0.5, 0.8, -1.2, 1.5, -2.0, 2.5, -3.0, 3.5];

    // Variance increases with x (weights should be inversely proportional)
    let weights = array![10., 9., 8., 7., 6., 5., 4., 3., 2., 1.5, 1.];

    // Residual function for linear regression
    fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
        let n = data.len() / 2;
        let x_vals = &data[0..n];
        let y_vals = &data[n..];

        let mut res = Array1::zeros(n);
        for i in 0..n {
            res[i] = y_vals[i] - (params[0] + params[1] * x_vals[i]);
        }
        res
    }

    // Jacobian (optional - improves convergence)
    fn jacobian(_params: &[f64], data: &[f64]) -> Array2<f64> {
        let n = data.len() / 2;
        let x_vals = &data[0..n];

        let mut jac = Array2::zeros((n, 2));
        for i in 0..n {
            jac[[i, 0]] = -1.0; // d/d(intercept)
            jac[[i, 1]] = -x_vals[i]; // d/d(slope)
        }
        jac
    }

    // Initial guess
    let x0 = array![0.0, 0.0];

    // 1. Standard Least Squares on Clean Data
    println!("1. Standard Least Squares (clean data)");
    println!("--------------------------------------");
    let mut data = Array1::zeros(22);
    for i in 0..11 {
        data[i] = x_data[i];
        data[i + 11] = y_clean[i];
    }

    let result_clean = least_squares(
        residual,
        &x0,
        Method::LevenbergMarquardt,
        Some(jacobian),
        &data,
        None,
    )?;

    println!(
        "Estimated: intercept = {:.3}, slope = {:.3}",
        result_clean.x[0], result_clean.x[1]
    );
    println!(
        "True:      intercept = {:.3}, slope = {:.3}",
        true_intercept, true_slope
    );
    println!("Residual norm: {:.3}", (result_clean.fun * 2.0).sqrt());
    println!();

    // 2. Standard vs Robust on Data with Outliers
    println!("2. Comparing Standard vs Robust (data with outliers)");
    println!("--------------------------------------------------");

    // Prepare data with outliers
    for i in 0..11 {
        data[i] = x_data[i];
        data[i + 11] = y_outliers[i];
    }

    // Standard least squares (sensitive to outliers)
    let result_standard = least_squares(
        residual,
        &x0,
        Method::LevenbergMarquardt,
        Some(jacobian),
        &data,
        None,
    )?;

    println!(
        "Standard LS: intercept = {:.3}, slope = {:.3}",
        result_standard.x[0], result_standard.x[1]
    );

    // Robust least squares with Huber loss
    let huber_loss = HuberLoss::new(1.0);
    let result_huber =
        robust_least_squares(residual, &x0, huber_loss, Some(jacobian), &data, None)?;

    println!(
        "Huber LS:    intercept = {:.3}, slope = {:.3}",
        result_huber.x[0], result_huber.x[1]
    );

    // Robust least squares with Bisquare loss (more aggressive outlier rejection)
    let bisquare_loss = BisquareLoss::new(4.685);
    let result_bisquare =
        robust_least_squares(residual, &x0, bisquare_loss, Some(jacobian), &data, None)?;

    println!(
        "Bisquare LS: intercept = {:.3}, slope = {:.3}",
        result_bisquare.x[0], result_bisquare.x[1]
    );
    println!(
        "True:        intercept = {:.3}, slope = {:.3}",
        true_intercept, true_slope
    );
    println!();

    // 3. Weighted Least Squares on Heteroscedastic Data
    println!("3. Weighted Least Squares (heteroscedastic data)");
    println!("-----------------------------------------------");

    // Prepare heteroscedastic data
    for i in 0..11 {
        data[i] = x_data[i];
        data[i + 11] = y_hetero[i];
    }

    // Unweighted (assumes constant variance)
    let result_unweighted = least_squares(
        residual,
        &x0,
        Method::LevenbergMarquardt,
        Some(jacobian),
        &data,
        None,
    )?;

    println!(
        "Unweighted LS: intercept = {:.3}, slope = {:.3}",
        result_unweighted.x[0], result_unweighted.x[1]
    );

    // Weighted (accounts for varying variance)
    let result_weighted =
        weighted_least_squares(residual, &x0, &weights, Some(jacobian), &data, None)?;

    println!(
        "Weighted LS:   intercept = {:.3}, slope = {:.3}",
        result_weighted.x[0], result_weighted.x[1]
    );
    println!(
        "True:          intercept = {:.3}, slope = {:.3}",
        true_intercept, true_slope
    );
    println!();

    // 4. Bounded Least Squares
    println!("4. Bounded Least Squares (with constraints)");
    println!("-----------------------------------------");

    // Use clean data but add unrealistic initial guess
    for i in 0..11 {
        data[i] = x_data[i];
        data[i + 11] = y_clean[i];
    }

    // Without bounds, from a bad initial guess
    let x0_bad = array![10.0, -2.0];
    let result_unbounded = least_squares(
        residual,
        &x0_bad,
        Method::LevenbergMarquardt,
        Some(jacobian),
        &data,
        None,
    )?;

    println!(
        "Without bounds: intercept = {:.3}, slope = {:.3}",
        result_unbounded.x[0], result_unbounded.x[1]
    );

    // With bounds: constrain intercept to [0, 3] and slope to [0, 1]
    let bounds = Bounds::new(&[(Some(0.0), Some(3.0)), (Some(0.0), Some(1.0))]);

    let result_bounded =
        bounded_least_squares(residual, &x0_bad, Some(bounds), Some(jacobian), &data, None)?;

    println!(
        "With bounds:    intercept = {:.3}, slope = {:.3}",
        result_bounded.x[0], result_bounded.x[1]
    );
    println!(
        "True:           intercept = {:.3}, slope = {:.3}",
        true_intercept, true_slope
    );
    println!();

    // 5. Combined approach: Weighted + Robust
    println!("5. Combined: Weighted + Robust (heteroscedastic with outliers)");
    println!("-----------------------------------------------------------");

    // Create data with both outliers and heteroscedasticity
    let mut y_combined = y_hetero.clone();
    y_combined[3] = 12.0; // Outlier
    y_combined[9] = -8.0; // Outlier

    for i in 0..11 {
        data[i] = x_data[i];
        data[i + 11] = y_combined[i];
    }

    // First apply weighting, then robust method
    // (In practice, you might implement a combined method)

    // Standard weighted (affected by outliers)
    let result_weighted_only =
        weighted_least_squares(residual, &x0, &weights, Some(jacobian), &data, None)?;

    println!(
        "Weighted only:  intercept = {:.3}, slope = {:.3}",
        result_weighted_only.x[0], result_weighted_only.x[1]
    );

    // For comparison, robust without weights
    let result_robust_only = robust_least_squares(
        residual,
        &x0,
        HuberLoss::new(1.0),
        Some(jacobian),
        &data,
        None,
    )?;

    println!(
        "Robust only:    intercept = {:.3}, slope = {:.3}",
        result_robust_only.x[0], result_robust_only.x[1]
    );
    println!(
        "True:           intercept = {:.3}, slope = {:.3}",
        true_intercept, true_slope
    );
    println!();

    // Summary
    println!("Summary");
    println!("-------");
    println!("- Standard LS: Best for clean data with constant variance");
    println!("- Robust LS: Handles outliers effectively");
    println!("- Weighted LS: Accounts for varying variance (heteroscedasticity)");
    println!("- Bounded LS: Enforces parameter constraints");
    println!("- Combined approaches can handle multiple data issues");

    Ok(())
}
