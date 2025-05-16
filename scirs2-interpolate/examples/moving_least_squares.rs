use ndarray::{Array1, Array2};
use scirs2_interpolate::local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Moving Least Squares Interpolation Example");
    println!("------------------------------------------\n");

    // Create a regular grid of points for visualization
    let n_grid = 20;
    let mut grid_points = Vec::with_capacity(n_grid * n_grid);
    let mut grid_coords_x = Vec::with_capacity(n_grid * n_grid);
    let mut grid_coords_y = Vec::with_capacity(n_grid * n_grid);

    for i in 0..n_grid {
        let x = i as f64 / (n_grid - 1) as f64;
        for j in 0..n_grid {
            let y = j as f64 / (n_grid - 1) as f64;
            grid_coords_x.push(x);
            grid_coords_y.push(y);
            grid_points.push(vec![x, y]);
        }
    }

    let grid_points_array = Array2::from_shape_vec(
        (n_grid * n_grid, 2),
        grid_points.into_iter().flatten().collect(),
    )?;

    // Create scattered training data (50 random points)
    let n_samples = 50;
    // No need to create rng instance as we're using rand::random directly
    let mut sample_points = Vec::with_capacity(n_samples);
    let mut sample_values = Vec::with_capacity(n_samples);

    // Generate random scattered data
    for _ in 0..n_samples {
        let x = rand::random::<f64>();
        let y = rand::random::<f64>();

        // Target function: z = sin(2πx) * cos(2πy) + some noise
        let true_z =
            f64::sin(2.0 * std::f64::consts::PI * x) * f64::cos(2.0 * std::f64::consts::PI * y);

        // Add some noise
        let noise = (rand::random::<f64>() - 0.5) * 0.1;
        let z = true_z + noise;

        sample_points.push(vec![x, y]);
        sample_values.push(z);
    }

    let sample_points_array = Array2::from_shape_vec(
        (n_samples, 2),
        sample_points.into_iter().flatten().collect(),
    )?;

    let sample_values_array = Array1::from_vec(sample_values);

    // Create interpolators with different settings
    let mls_constant = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Constant,
        0.1, // bandwidth
    )?;

    let mls_linear = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Linear,
        0.1, // bandwidth
    )?;

    let mls_quadratic = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Quadratic,
        0.1, // bandwidth
    )?;

    // Interpolate on grid for visualization
    println!("Interpolating on a {n_grid}x{n_grid} grid with different polynomial bases...");

    let constant_results = mls_constant.evaluate_multi(&grid_points_array.view())?;
    let linear_results = mls_linear.evaluate_multi(&grid_points_array.view())?;
    let quadratic_results = mls_quadratic.evaluate_multi(&grid_points_array.view())?;

    // Also compute true function values on grid
    let mut true_values = Vec::with_capacity(n_grid * n_grid);
    for i in 0..grid_points_array.shape()[0] {
        let x = grid_points_array[[i, 0]];
        let y = grid_points_array[[i, 1]];
        let true_z =
            f64::sin(2.0 * std::f64::consts::PI * x) * f64::cos(2.0 * std::f64::consts::PI * y);
        true_values.push(true_z);
    }
    let true_grid_values = Array1::from_vec(true_values);

    // Compute error metrics
    let constant_mse = compute_mse(&constant_results, &true_grid_values);
    let linear_mse = compute_mse(&linear_results, &true_grid_values);
    let quadratic_mse = compute_mse(&quadratic_results, &true_grid_values);

    println!("\nMean Squared Error (MSE) on grid:");
    println!("  Constant basis: {:.6}", constant_mse);
    println!("  Linear basis:   {:.6}", linear_mse);
    println!("  Quadratic basis: {:.6}", quadratic_mse);

    // Try different weight functions with linear basis
    println!("\nTrying different weight functions with linear basis...");

    let mls_gaussian = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Linear,
        0.1, // bandwidth
    )?;

    let mls_wendland = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::WendlandC2,
        PolynomialBasis::Linear,
        0.2, // bandwidth
    )?;

    let mls_inverse = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::InverseDistance,
        PolynomialBasis::Linear,
        0.1, // bandwidth
    )?;

    let mls_cubic = MovingLeastSquares::new(
        sample_points_array.clone(),
        sample_values_array.clone(),
        WeightFunction::CubicSpline,
        PolynomialBasis::Linear,
        0.2, // bandwidth
    )?;

    let gaussian_results = mls_gaussian.evaluate_multi(&grid_points_array.view())?;
    let wendland_results = mls_wendland.evaluate_multi(&grid_points_array.view())?;
    let inverse_results = mls_inverse.evaluate_multi(&grid_points_array.view())?;
    let cubic_results = mls_cubic.evaluate_multi(&grid_points_array.view())?;

    let gaussian_mse = compute_mse(&gaussian_results, &true_grid_values);
    let wendland_mse = compute_mse(&wendland_results, &true_grid_values);
    let inverse_mse = compute_mse(&inverse_results, &true_grid_values);
    let cubic_mse = compute_mse(&cubic_results, &true_grid_values);

    println!("\nMSE for different weight functions (linear basis):");
    println!("  Gaussian:      {:.6}", gaussian_mse);
    println!("  Wendland C2:   {:.6}", wendland_mse);
    println!("  Inverse dist.: {:.6}", inverse_mse);
    println!("  Cubic spline:  {:.6}", cubic_mse);

    // Experiment with different bandwidth values
    println!("\nEffect of bandwidth parameter (Gaussian weights, linear basis):");
    let bandwidths = [0.05, 0.1, 0.2, 0.5, 1.0];

    for &bandwidth in &bandwidths {
        let mls = MovingLeastSquares::new(
            sample_points_array.clone(),
            sample_values_array.clone(),
            WeightFunction::Gaussian,
            PolynomialBasis::Linear,
            bandwidth,
        )?;

        let results = mls.evaluate_multi(&grid_points_array.view())?;
        let mse = compute_mse(&results, &true_grid_values);

        println!("  Bandwidth = {:.2}: MSE = {:.6}", bandwidth, mse);
    }

    println!("\nNote: For visualization, you would typically save the grid coordinates and");
    println!("interpolated values to a file, then plot using an external tool like");
    println!("matplotlib, gnuplot, or create an interactive visualization with plotly.");

    Ok(())
}

// Compute mean squared error between predicted and true values
fn compute_mse(predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let n = predicted.len();
    let mut sum_squared_error = 0.0;

    for i in 0..n {
        let error = predicted[i] - actual[i];
        sum_squared_error += error * error;
    }

    sum_squared_error / n as f64
}
