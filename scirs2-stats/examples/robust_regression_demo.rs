use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_stats::regression::{
    huber_regression, linear_regression, ransac, theilslopes, HuberT, RegressionResults,
};
use statrs::statistics::Statistics;

#[allow(dead_code)]
fn generatedata_with_outliers() -> (Array2<f64>, Array1<f64>) {
    // Generate simple linear data with slope ~2, intercept ~1
    let x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    let n = x_values.len();

    // Create design matrix with intercept column
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { x_values[i] });

    // True y values would be: y = 1 + 2*x
    let mut y_values = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

    // Add outliers at positions 2, 5, and 8
    y_values[2] = 12.0; // Should be 4.0
    y_values[5] = 1.0; // Should be 7.0
    y_values[8] = 18.0; // Should be 10.0

    let y = Array1::from(y_values);
    (x, y)
}

#[allow(dead_code)]
fn print_regression_summary(name: &str, results: &RegressionResults<f64>) {
    println!("\n{} Regression Results:", name);
    println!("Coefficients: {:?}", results.coefficients);
    println!("R-squared: {:.4}", results.r_squared);

    // Calculate MSE from residuals
    let mse = results.residuals.mapv(|r| r * r).sum() / results.residuals.len() as f64;
    println!("MSE: {:.4}", mse);

    // Count inliers
    let inlier_count = results.inlier_mask.iter().filter(|&&x| x).count();
    println!("Inliers: {}/{}", inlier_count, results.inlier_mask.len());

    // Print outlier indices
    let outlier_indices: Vec<usize> = results
        .inlier_mask
        .iter()
        .enumerate()
        .filter(|(_, &is_inlier)| !is_inlier)
        .map(|(idx_, _)| idx_)
        .collect();
    if !outlier_indices.is_empty() {
        println!("Outlier indices: {:?}", outlier_indices);
    }
}

#[allow(dead_code)]
fn main() {
    // Generate data with outliers
    let (x, y) = generatedata_with_outliers();

    println!("Data points:");
    for i in 0..y.len() {
        println!("x[{}] = {:.1}, y[{}] = {:.1}", i, x[[i, 1]], i, y[i]);
    }

    // 1. Ordinary Least Squares (OLS)
    let ols_results =
        linear_regression(&x.view(), &y.view(), Some(0.95)).expect("OLS regression failed");
    print_regression_summary("OLS", &ols_results);

    // 2. Theil-Sen Estimator
    let theilsen_results = theilslopes(
        &x.column(1).view(),
        &y.view(),
        Some(0.95),
        Some("approximate"),
    )
    .expect("Theil-Sen regression failed");
    println!("\nTheil-Sen Regression Results:");
    println!("Intercept: {:.4}", theilsen_results.intercept);
    println!("Slope: {:.4}", theilsen_results.slope);

    // Convert to RegressionResults for comparison
    let theilsen_coeffs = array![theilsen_results.intercept, theilsen_results.slope];
    let y_pred_theilsen = x.dot(&theilsen_coeffs);
    let residuals_theilsen = &y - &y_pred_theilsen;
    let ss_res = residuals_theilsen.mapv(|r| r * r).sum();
    let y_mean = y.clone().mean();
    let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared_theilsen = 1.0 - (ss_res / ss_tot);
    let _mse_theilsen = ss_res / y.len() as f64;

    let theilsen_regression_results = RegressionResults {
        coefficients: theilsen_coeffs,
        std_errors: array![0.0, 0.0], // Not available for Theil-Sen
        t_values: array![0.0, 0.0],
        p_values: array![0.0, 0.0],
        conf_intervals: Array2::zeros((2, 2)),
        r_squared: r_squared_theilsen,
        adj_r_squared: r_squared_theilsen, // Simple approximation
        f_statistic: 0.0,
        f_p_value: 0.0,
        residual_std_error: 0.0,
        df_residuals: y.len() - 2,
        residuals: residuals_theilsen,
        fitted_values: y_pred_theilsen,
        inlier_mask: vec![true; y.len()], // All points considered
    };
    print_regression_summary("Theil-Sen (converted)", &theilsen_regression_results);

    // 3. RANSAC
    // Extract column as 2D array for RANSAC
    let mut x_ransac = Array2::zeros((x.nrows(), 1));
    for i in 0..x.nrows() {
        x_ransac[[i, 0]] = x[[i, 1]];
    }

    let ransac_results = ransac(
        &x_ransac.view(),
        &y.view(),
        None,      // Use default min_samples
        Some(2.0), // Residual threshold
        None,      // Use default max_trials
        None,      // Use default stop_probability
        Some(42),  // Random seed
    )
    .expect("RANSAC regression failed");
    print_regression_summary("RANSAC", &ransac_results);

    // 4. Huber Regression
    let huber_results = huber_regression(
        &x.view(),
        &y.view(),
        Some(1.35),  // Default epsilon for Huber loss
        Some(false), // fit_intercept=false (already in design matrix)
        None,        // Use default scale
        None,        // Use default max_iter
        None,        // Use default tolerance
        Some(0.95),
    )
    .expect("Huber regression failed");
    print_regression_summary("Huber", &huber_results);

    // Compare coefficients
    println!("\nComparison of Coefficients:");
    println!("Method      | Intercept | Slope");
    println!("------------|-----------|------");
    println!("True values |    1.0    |  2.0");
    println!(
        "OLS         |   {:.3}   | {:.3}",
        ols_results.coefficients[0], ols_results.coefficients[1]
    );
    println!(
        "Theil-Sen   |   {:.3}   | {:.3}",
        theilsen_results.intercept, theilsen_results.slope
    );
    println!(
        "RANSAC      |   {:.3}   | {:.3}",
        ransac_results.coefficients[0], ransac_results.coefficients[1]
    );
    println!(
        "Huber       |   {:.3}   | {:.3}",
        huber_results.coefficients[0], huber_results.coefficients[1]
    );

    // R-squared comparison
    println!("\nR-squared Comparison:");
    println!("OLS:       {:.4}", ols_results.r_squared);
    println!("Theil-Sen: {:.4}", r_squared_theilsen);
    println!("RANSAC:    {:.4}", ransac_results.r_squared);
    println!("Huber:     {:.4}", huber_results.r_squared);

    // Test different Huber T values
    println!("\nHuber Regression with Different T Values:");
    for &t_val in &[0.5, 1.0, 1.35, 2.0, 3.0] {
        let _huber_model = HuberT::with_t(t_val);
        let huber_t_results = huber_regression(
            &x.view(),
            &y.view(),
            Some(t_val),
            Some(false),
            None,
            None,
            None,
            Some(0.95),
        )
        .expect("Huber regression failed");

        let huber_t_inliers = huber_t_results.inlier_mask.iter().filter(|&&x| x).count();
        println!(
            "T = {:.2}: Coefficients = [{:.3}, {:.3}], Inliers = {}/{}",
            t_val,
            huber_t_results.coefficients[0],
            huber_t_results.coefficients[1],
            huber_t_inliers,
            huber_t_results.inlier_mask.len()
        );
    }
}
