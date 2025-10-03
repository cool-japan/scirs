use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::RegressionResults;

#[allow(dead_code)]
fn main() {
    println!("Regression Summary Example");
    println!("=========================\n");

    // Create a simple regression model results manually
    let coefficients = array![2.0, 1.5]; // intercept and slope
    let std_errors = array![0.2, 0.1];
    let t_values = array![10.0, 15.0];
    let p_values = array![0.0001, 0.00001];

    // Confidence intervals (lower and upper bounds for each coefficient)
    let mut conf_intervals = Array2::zeros((2, 2));
    conf_intervals[[0, 0]] = 1.6; // intercept lower bound
    conf_intervals[[0, 1]] = 2.4; // intercept upper bound
    conf_intervals[[1, 0]] = 1.3; // slope lower bound
    conf_intervals[[1, 1]] = 1.7; // slope upper bound

    // Model statistics
    let r_squared = 0.95;
    let adj_r_squared = 0.94;
    let f_statistic = 180.0;
    let f_p_value = 0.00001;
    let residual_std_error = 0.3;
    let df_residuals = 8;

    // Sample data
    let residuals = array![-0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.2, 0.1, 0.1];
    let fitted_values = array![2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 15.5];

    // Create the RegressionResults struct
    let results = RegressionResults {
        coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; 10], // All points are considered inliers
    };

    // Display the summary
    println!("{}", results.summary());

    // Create test data for prediction
    let x_new = Array2::from_shape_vec(
        (2, 2),
        vec![
            1.0, 10.0, // Two new data points with 2 variables each (intercept and x)
            1.0, 20.0,
        ],
    )
    .unwrap();

    // Make predictions
    let predictions = results.predict(&x_new.view()).unwrap();

    println!("\nPredictions for new data:");
    println!("  x = 10 -> y = {:.1}", predictions[0]); // should be 2.0 + 1.5*10 = 17.0
    println!("  x = 20 -> y = {:.1}", predictions[1]); // should be 2.0 + 1.5*20 = 32.0
}
