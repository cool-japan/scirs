use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::{linear_regression, polyfit};

// Temporarily commenting out multilinear_regression due to linalg crate errors
// use scirs2_stats::{linear_regression, multilinear_regression, polyfit};

#[allow(dead_code)]
fn main() {
    println!("Enhanced Regression Examples");
    println!("===========================\n");

    // Basic linear regression example (y = 3x + 2)
    println!("1. Simple Linear Regression Example");
    println!("----------------------------------");

    let x_simple = Array2::from_shape_vec(
        (5, 2),
        vec![
            1.0, 1.0, // intercept and x columns
            1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0,
        ],
    )
    .unwrap();

    let y_simple = array![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 2 + 3x

    let results = linear_regression(&x_simple.view(), &y_simple.view(), None).unwrap();
    println!("{}", results.summary());

    // Prediction example
    let x_new = Array2::from_shape_vec(
        (2, 2),
        vec![
            1.0, 6.0, // new data points
            1.0, 7.0,
        ],
    )
    .unwrap();

    let predictions = results.predict(&x_new.view()).unwrap();
    println!("Predictions for new data points:");
    println!("  x = 6 -> y = {:.1}", predictions[0]); // should be 20
    println!("  x = 7 -> y = {:.1}", predictions[1]); // should be 23
    println!();

    // Multiple linear regression example (y = 1.0 + 2.0*x1 + 3.0*x2)
    println!("2. Multiple Linear Regression Example");
    println!("-----------------------------------");

    let x_multi = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 0.0, 1.0, // intercept, x1, x2
            1.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 3.0, 1.0, 1.0, 4.0, 2.0, 1.0, 5.0, 0.0,
        ],
    )
    .unwrap();

    let y_multi = array![4.0, 9.0, 5.0, 10.0, 15.0, 11.0]; // y = 1 + 2*x1 + 3*x2

    let results_multi = linear_regression(&x_multi.view(), &y_multi.view(), None).unwrap();
    println!("{}", results_multi.summary());

    // Traditional multilinear regression (temporarily commented out due to linalg crate errors)
    /*
    let (coeffs, residuals, rank, sv) = multilinear_regression(&x_multi.view(), &y_multi.view()).unwrap();
    println!("Traditional multilinear regression output:");
    println!("  Coefficients: {:.4?}", coeffs);
    println!("  Rank: {}", rank);
    println!("  Top singular values: {:.4?}", sv.iter().take(2).collect::<Vec<_>>());
    println!();
    */
    println!("Traditional multilinear regression (skipped due to linalg crate errors)");
    println!();

    // Polynomial regression example
    println!("3. Polynomial Regression Example");
    println!("------------------------------");

    // Data following y = 2x^2 - 3x + 1
    let x_poly = array![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let y_poly = array![
        // y = 2x^2 - 3x + 1
        2.0 * (-2.0_f64).powi(2) - 3.0 * (-2.0) + 1.0,
        2.0 * (-1.5_f64).powi(2) - 3.0 * (-1.5) + 1.0,
        2.0 * (-1.0_f64).powi(2) - -3.0 + 1.0,
        2.0 * (-0.5_f64).powi(2) - 3.0 * (-0.5) + 1.0,
        2.0 * (0.0_f64).powi(2) - 3.0 * (0.0) + 1.0,
        2.0 * (0.5_f64).powi(2) - 3.0 * (0.5) + 1.0,
        2.0 * (1.0_f64).powi(2) - 3.0 * (1.0) + 1.0,
        2.0 * (1.5_f64).powi(2) - 3.0 * (1.5) + 1.0,
        2.0 * (2.0_f64).powi(2) - 3.0 * (2.0) + 1.0,
    ];

    // Fit polynomial of degree 2
    let poly_coeffs = polyfit(&x_poly.view(), &y_poly.view(), 2).unwrap();
    println!("Polynomial coefficients (highest degree first):");
    println!(
        "  y = {:.1}x^2 + {:.1}x + {:.1}",
        poly_coeffs.coefficients[2], poly_coeffs.coefficients[1], poly_coeffs.coefficients[0]
    );

    // To fit with the enhanced regression API, we need to create a design matrix
    // with columns [1, x, x^2]
    let mut x_poly_design = Array2::<f64>::zeros((x_poly.len(), 3));
    for i in 0..x_poly.len() {
        x_poly_design[[i, 0]] = 1.0; // intercept
        x_poly_design[[i, 1]] = x_poly[i]; // x
        x_poly_design[[i, 2]] = x_poly[i].powi(2); // x^2
    }

    let poly_results = linear_regression(&x_poly_design.view(), &y_poly.view(), None).unwrap();
    println!("\nPolynomial regression using multiple linear regression:");
    println!("{}", poly_results.summary());

    // Make predictions at specific points
    let x_test = array![-1.75, -0.75, 0.25, 1.25, 1.75];

    println!("\nModel predictions:");
    println!("  x   |   Expected   |   Predicted");
    println!("-----------------------------------");

    for &x in x_test.iter() {
        let x_f64: f64 = x;
        // Calculate expected value using the true polynomial
        let y_expected = 2.0 * x_f64.powi(2) - 3.0 * x_f64 + 1.0;

        // Create a design matrix row for prediction
        let x_pred = Array2::from_shape_vec((1, 3), vec![1.0, x_f64, x_f64.powi(2)]).unwrap();
        let y_pred = poly_results.predict(&x_pred.view()).unwrap()[0];

        println!(" {:.2} |   {:.4}   |   {:.4}", x_f64, y_expected, y_pred);
    }
}
