//! Example demonstrating robust least squares optimization
//!
//! This example shows how to use robust loss functions to handle outliers in data.

use ndarray::{array, Array1, Array2};
use scirs2_optimize::least_squares::{
    robust_least_squares, BisquareLoss, CauchyLoss, HuberLoss, RobustOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create synthetic data with outliers
    // True model: y = 1.0 + 2.0 * x
    let x_data = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y_data = array![
        1.1,  // Slight noise
        3.0,  // Good data
        5.2,  // Slight noise
        6.8,  // Slight noise
        15.0, // OUTLIER (should be ~9)
        11.1, // Good data
        13.0, // Good data
        14.9, // Slight noise
        17.2, // Good data
        30.0, // OUTLIER (should be ~19)
    ];

    // Define residual function for linear regression
    fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
        let n = data.len() / 2;
        let x_vals = &data[0..n];
        let y_vals = &data[n..];

        let mut res = Array1::zeros(n);
        for i in 0..n {
            // Model: y = params[0] + params[1] * x
            res[i] = y_vals[i] - (params[0] + params[1] * x_vals[i]);
        }
        res
    }

    // Define Jacobian (optional - will use finite differences if not provided)
    fn jacobian(_params: &[f64], data: &[f64]) -> Array2<f64> {
        let n = data.len() / 2;
        let x_vals = &data[0..n];

        let mut jac = Array2::zeros((n, 2));
        for i in 0..n {
            jac[[i, 0]] = -1.0; // Derivative w.r.t. intercept
            jac[[i, 1]] = -x_vals[i]; // Derivative w.r.t. slope
        }
        jac
    }

    // Concatenate x and y data for the function interface
    let mut data = Array1::zeros(20);
    for i in 0..10 {
        data[i] = x_data[i];
        data[i + 10] = y_data[i];
    }

    // Initial guess
    let x0 = array![0.0, 0.0];

    println!("Robust Least Squares Example");
    println!("============================");
    println!("True parameters: intercept = 1.0, slope = 2.0");
    println!();

    // Test with standard least squares (for comparison)
    // Note: Using squared loss (Huber with very large delta is similar)
    let squared_loss = HuberLoss::new(1e6);
    let result_squared =
        robust_least_squares(residual, &x0, squared_loss, Some(jacobian), &data, None)?;

    println!("Standard Least Squares (sensitive to outliers):");
    println!("  Intercept: {:.3}", result_squared.x[0]);
    println!("  Slope: {:.3}", result_squared.x[1]);
    println!("  Cost: {:.3}", result_squared.fun);
    println!();

    // Test with Huber loss
    let huber_loss = HuberLoss::new(1.0);
    let result_huber =
        robust_least_squares(residual, &x0, huber_loss, Some(jacobian), &data, None)?;

    println!("Huber Loss (robust to outliers):");
    println!("  Intercept: {:.3}", result_huber.x[0]);
    println!("  Slope: {:.3}", result_huber.x[1]);
    println!("  Cost: {:.3}", result_huber.fun);
    println!();

    // Test with Bisquare loss
    let bisquare_loss = BisquareLoss::new(4.685);
    let result_bisquare =
        robust_least_squares(residual, &x0, bisquare_loss, Some(jacobian), &data, None)?;

    println!("Bisquare Loss (strongly robust to outliers):");
    println!("  Intercept: {:.3}", result_bisquare.x[0]);
    println!("  Slope: {:.3}", result_bisquare.x[1]);
    println!("  Cost: {:.3}", result_bisquare.fun);
    println!();

    // Test with Cauchy loss
    let cauchy_loss = CauchyLoss::new(1.0);
    let result_cauchy =
        robust_least_squares(residual, &x0, cauchy_loss, Some(jacobian), &data, None)?;

    println!("Cauchy Loss (very robust to outliers):");
    println!("  Intercept: {:.3}", result_cauchy.x[0]);
    println!("  Slope: {:.3}", result_cauchy.x[1]);
    println!("  Cost: {:.3}", result_cauchy.fun);
    println!();

    // Test with custom options
    let mut options = RobustOptions::default();
    options.max_iter = 50;
    options.irls_max_iter = 30;

    let huber_loss_custom = HuberLoss::new(0.5);
    let result_custom = robust_least_squares(
        residual,
        &x0,
        huber_loss_custom,
        Some(jacobian),
        &data,
        Some(options),
    )?;

    println!("Huber Loss with custom options (delta=0.5):");
    println!("  Intercept: {:.3}", result_custom.x[0]);
    println!("  Slope: {:.3}", result_custom.x[1]);
    println!("  Cost: {:.3}", result_custom.fun);
    println!("  Iterations: {}", result_custom.nit);
    println!();

    println!("Analysis:");
    println!("---------");
    println!("Notice how the robust methods produce estimates closer to the true");
    println!("parameters (1.0, 2.0) despite the presence of outliers in the data.");
    println!("The standard least squares is heavily influenced by the outliers,");
    println!("while the robust methods downweight their influence.");

    Ok(())
}
