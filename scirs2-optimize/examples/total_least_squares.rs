//! Example demonstrating total least squares (errors-in-variables)
//!
//! Shows how to fit a line when both x and y measurements have errors.

use ndarray::{array, Array1};
use scirs2_optimize::least_squares::{total_least_squares, TLSMethod, TotalLeastSquaresOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Total Least Squares Example");
    println!("==========================");
    println!();

    // Example 1: Simple case with equal errors in both variables
    println!("Example 1: Equal errors in x and y");
    println!("---------------------------------");

    // True line: y = 2x + 1
    let true_slope = 2.0;
    let true_intercept = 1.0;

    // True values
    let x_true = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_true = &x_true * true_slope + true_intercept;

    // Add errors to both x and y
    let x_errors = array![0.1, -0.15, 0.08, -0.12, 0.05, -0.09];
    let y_errors = array![-0.08, 0.12, -0.1, 0.15, -0.07, 0.11];

    let x_measured = &x_true + &x_errors;
    let y_measured = &y_true + &y_errors;

    println!("True line: y = {:.1}x + {:.1}", true_slope, true_intercept);

    // Compare ordinary least squares (OLS) with total least squares (TLS)

    // OLS (minimizes only vertical distances)
    let _n = x_measured.len() as f64;
    let x_mean = x_measured.mean().unwrap();
    let y_mean = y_measured.mean().unwrap();

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x_measured.len() {
        let dx = x_measured[i] - x_mean;
        let dy = y_measured[i] - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    let ols_slope = num / den;
    let ols_intercept = y_mean - ols_slope * x_mean;

    println!("OLS result: y = {:.3}x + {:.3}", ols_slope, ols_intercept);

    // TLS (minimizes orthogonal distances)
    let tls_result = total_least_squares::<_, _, _, _>(
        &x_measured,
        &y_measured,
        None::<&Array1<f64>>,
        None::<&Array1<f64>>,
        None,
    )?;

    println!(
        "TLS result: y = {:.3}x + {:.3}",
        tls_result.slope, tls_result.intercept
    );
    println!(
        "Orthogonal residuals: {:.6}",
        tls_result.orthogonal_residuals
    );
    println!();

    // Example 2: Different error variances
    println!("Example 2: Different error variances");
    println!("-----------------------------------");

    // Simulate measurements with different error variances
    let x_measured2 = array![0.5, 1.2, 1.8, 3.1, 3.9, 5.2];
    let y_measured2 = array![2.1, 3.8, 4.9, 7.3, 8.7, 11.5];

    // Known error variances (x has smaller errors than y)
    let x_variance = array![0.01, 0.01, 0.02, 0.01, 0.015, 0.01];
    let y_variance = array![0.1, 0.08, 0.12, 0.09, 0.11, 0.1];

    // Weighted TLS
    let weighted_result = total_least_squares(
        &x_measured2,
        &y_measured2,
        Some(&x_variance),
        Some(&y_variance),
        None,
    )?;

    println!("Measurements with different error variances");
    println!("x variance: {:?}", x_variance);
    println!("y variance: {:?}", y_variance);
    println!(
        "Weighted TLS: y = {:.3}x + {:.3}",
        weighted_result.slope, weighted_result.intercept
    );
    println!();

    // Example 3: Comparison of methods
    println!("Example 3: Comparison of TLS methods");
    println!("-----------------------------------");

    let x_measured3 = array![0.8, 1.6, 2.3, 3.2, 4.1, 4.9];
    let y_measured3 = array![2.7, 4.2, 5.8, 7.5, 9.1, 10.8];

    // SVD method
    let mut options_svd = TotalLeastSquaresOptions::default();
    options_svd.method = TLSMethod::SVD;

    let result_svd = total_least_squares::<_, _, _, _>(
        &x_measured3,
        &y_measured3,
        None::<&Array1<f64>>,
        None::<&Array1<f64>>,
        Some(options_svd),
    )?;

    // Iterative method
    let mut options_iter = TotalLeastSquaresOptions::default();
    options_iter.method = TLSMethod::Iterative;

    let result_iter = total_least_squares::<_, _, _, _>(
        &x_measured3,
        &y_measured3,
        None::<&Array1<f64>>,
        None::<&Array1<f64>>,
        Some(options_iter),
    )?;

    println!(
        "SVD method:      y = {:.3}x + {:.3}",
        result_svd.slope, result_svd.intercept
    );
    println!(
        "Iterative method: y = {:.3}x + {:.3} (iterations: {})",
        result_iter.slope, result_iter.intercept, result_iter.iterations
    );
    println!();

    // Example 4: Application to calibration
    println!("Example 4: Instrument calibration");
    println!("--------------------------------");
    println!("Calibrating one instrument against another (both have errors)");

    // Simulated instrument readings
    let instrument_a = array![10.2, 20.5, 30.1, 40.8, 50.3, 60.9, 71.2, 80.7];
    let instrument_b = array![9.8, 19.7, 29.9, 39.5, 49.8, 59.6, 70.1, 79.5];

    // Both instruments have measurement errors
    let var_a = Array1::from_elem(8, 0.5); // Constant variance for simplicity
    let var_b = Array1::from_elem(8, 0.4);

    let calibration = total_least_squares(
        &instrument_a,
        &instrument_b,
        Some(&var_a),
        Some(&var_b),
        None,
    )?;

    println!(
        "Calibration equation: B = {:.3} * A + {:.3}",
        calibration.slope, calibration.intercept
    );
    println!("Corrected values:");
    println!("A_corrected: {:?}", calibration.x_corrected);
    println!("B_corrected: {:?}", calibration.y_corrected);
    println!();

    // Summary
    println!("Summary");
    println!("-------");
    println!("Total Least Squares is useful when:");
    println!("1. Both variables have measurement errors");
    println!("2. You want to minimize perpendicular distances to the line");
    println!("3. Error variances are known (for weighted TLS)");
    println!("4. Calibrating instruments against each other");
    println!();
    println!("The method finds the line that best fits the data considering");
    println!("errors in both dimensions, unlike ordinary least squares which");
    println!("only considers vertical errors.");

    Ok(())
}
