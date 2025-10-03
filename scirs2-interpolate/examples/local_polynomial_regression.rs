use scirs2_core::ndarray::{Array1, Axis};
use scirs2_interpolate::local::mls::PolynomialBasis;
use scirs2_interpolate::local::polynomial::{
    make_loess, make_robust_loess, LocalPolynomialConfig, LocalPolynomialRegression,
};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Local Polynomial Regression (LOESS) Example");
    println!("--------------------------------------------\n");

    // Create a 1D example dataset with noise
    let n_points = 100;
    let x_array = Array1::linspace(0.0, 10.0, n_points);
    let mut y_array = Array1::zeros(n_points);

    for (i, &x) in x_array.iter().enumerate() {
        // Generate y = sin(x) + noise
        let y_true = f64::sin(x);
        let noise = (scirs2_core::random::random::<f64>() - 0.5) * 0.2; // Noise with ±0.1 amplitude
        y_array[i] = y_true + noise;
    }

    // Convert x to 2D array
    let x_2d = x_array.clone().insert_axis(Axis(1));

    // Simple LOESS with default settings
    println!("Creating a LOESS model with different bandwidths...");
    let loess_small = make_loess(x_2d.clone(), y_array.clone(), 0.1)?; // Small bandwidth
    let loess_medium = make_loess(x_2d.clone(), y_array.clone(), 0.3)?; // Medium bandwidth
    let loess_large = make_loess(x_2d.clone(), y_array.clone(), 0.7)?; // Large bandwidth

    // Create evaluation points
    let eval_x = Array1::linspace(0.0, 10.0, 50);
    let eval_x_2d = eval_x.clone().insert_axis(Axis(1));

    // Evaluate models
    println!("Evaluating models at test points...");
    let small_fit = loess_small.fit_multiple(&eval_x_2d.view())?;
    let medium_fit = loess_medium.fit_multiple(&eval_x_2d.view())?;
    let large_fit = loess_large.fit_multiple(&eval_x_2d.view())?;

    // Compute true values for comparison
    let mut true_values = Array1::zeros(eval_x.len());
    for (i, &x) in eval_x.iter().enumerate() {
        true_values[i] = f64::sin(x);
    }

    // Compute mean squared errors
    let mse_small = compute_mse(&small_fit, &true_values);
    let mse_medium = compute_mse(&medium_fit, &true_values);
    let mse_large = compute_mse(&large_fit, &true_values);

    println!("\nMean Squared Error (MSE) at evaluation points:");
    println!("  Small bandwidth (0.1):  {:.6}", mse_small);
    println!("  Medium bandwidth (0.3): {:.6}", mse_medium);
    println!("  Large bandwidth (0.7):  {:.6}", mse_large);

    // Try cross-validation for bandwidth selection
    println!("\nSelecting optimal bandwidth by cross-validation...");
    let bandwidths = vec![0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

    // Use a model for cross-validation
    let cv_model = make_loess(x_2d.clone(), y_array.clone(), 0.3)?;
    let optimal_bandwidth = cv_model.select_bandwidth(&bandwidths)?;

    println!("  Optimal bandwidth: {:.3}", optimal_bandwidth);

    // Create model with optimal bandwidth
    let optimal_model = make_loess(x_2d.clone(), y_array.clone(), optimal_bandwidth)?;
    let optimal_fit = optimal_model.fit_multiple(&eval_x_2d.view())?;
    let mse_optimal = compute_mse(&optimal_fit, &true_values);

    println!("  MSE with optimal bandwidth: {:.6}", mse_optimal);

    // Demonstrate robust modeling with confidence intervals
    println!("\nDemonstrating robust LOESS with confidence intervals...");

    // Add some outliers to the data
    let mut y_outliers = y_array.clone();
    y_outliers[10] += 1.0; // Add outlier
    y_outliers[30] -= 1.0; // Add outlier
    y_outliers[50] += 1.5; // Add outlier
    y_outliers[70] -= 1.5; // Add outlier

    // Create robust model with 95% confidence intervals
    let robust_model = make_robust_loess(
        x_2d.clone(),
        y_outliers.clone(),
        0.3,  // Medium bandwidth
        0.95, // 95% confidence level
    )?;

    // Evaluate at a specific point
    let test_point = Array1::from_vec(vec![5.0]);
    let result = robust_model.fit_at_point(&test_point.view())?;

    println!("Results at x = 5.0:");
    println!("  True value:     {:.4}", f64::sin(5.0));
    println!("  Fitted value:   {:.4}", result.value);
    println!("  Standard error: {:.4}", result.std_error);

    if let Some((lower, upper)) = result.confidence_interval {
        println!("  95% CI:         ({:.4}, {:.4})", lower, upper);
    }

    println!("  R-squared:      {:.4}", result.r_squared);
    println!("  Effective df:   {:.4}", result.effective_df);

    // Show effect of basis function degree
    println!("\nEffect of polynomial basis degree:");

    // Create models with different basis functions
    let config_constant = LocalPolynomialConfig {
        bandwidth: 0.3,
        basis: PolynomialBasis::Constant,
        ..LocalPolynomialConfig::default()
    };

    let config_linear = LocalPolynomialConfig {
        bandwidth: 0.3,
        basis: PolynomialBasis::Linear,
        ..LocalPolynomialConfig::default()
    };

    let config_quadratic = LocalPolynomialConfig {
        bandwidth: 0.3,
        basis: PolynomialBasis::Quadratic,
        ..LocalPolynomialConfig::default()
    };

    let constant_model =
        LocalPolynomialRegression::with_config(x_2d.clone(), y_array.clone(), config_constant)?;

    let linear_model =
        LocalPolynomialRegression::with_config(x_2d.clone(), y_array.clone(), config_linear)?;

    let quadratic_model =
        LocalPolynomialRegression::with_config(x_2d.clone(), y_array.clone(), config_quadratic)?;

    // Evaluate models
    let constant_fit = constant_model.fit_multiple(&eval_x_2d.view())?;
    let linear_fit = linear_model.fit_multiple(&eval_x_2d.view())?;
    let quadratic_fit = quadratic_model.fit_multiple(&eval_x_2d.view())?;

    // Compute MSEs
    let mse_constant = compute_mse(&constant_fit, &true_values);
    let mse_linear = compute_mse(&linear_fit, &true_values);
    let mse_quadratic = compute_mse(&quadratic_fit, &true_values);

    println!("Mean Squared Error by basis degree:");
    println!("  Constant:  {:.6}", mse_constant);
    println!("  Linear:    {:.6}", mse_linear);
    println!("  Quadratic: {:.6}", mse_quadratic);

    println!("\nNote: For visualization, you would typically save the evaluation points and");
    println!("fitted values to a file, then plot using an external tool like");
    println!("matplotlib, gnuplot, or create an interactive visualization with plotly.");

    Ok(())
}

// Compute mean squared error between predicted and true values
#[allow(dead_code)]
fn compute_mse(predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let n = predicted.len();
    let mut sum_squared_error = 0.0;

    for i in 0..n {
        let error = predicted[i] - actual[i];
        sum_squared_error += error * error;
    }

    sum_squared_error / n as f64
}
