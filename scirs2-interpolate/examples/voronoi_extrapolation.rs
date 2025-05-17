//! Example demonstrating extrapolation methods for Voronoi-based interpolation
//!
//! This example shows how to use different extrapolation methods to handle
//! queries outside the convex hull of the input data points.

use ndarray::{Array1, Array2};
use rand::Rng;
use scirs2_interpolate::voronoi::{
    constant_value_extrapolation, inverse_distance_extrapolation, linear_gradient_extrapolation,
    make_sibson_interpolator, nearest_neighbor_extrapolation,
};
use scirs2_interpolate::Extrapolation;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Generate scattered data points in a unit square
    let n_points = 50;
    let mut rng = rand::rng();

    // Create points in a specific region (unit square)
    let mut points_vec = Vec::with_capacity(n_points * 2);
    for _ in 0..n_points {
        let x = rng.random_range(0.0..=1.0);
        let y = rng.random_range(0.0..=1.0);
        points_vec.push(x);
        points_vec.push(y);
    }

    let points = Array2::from_shape_vec((n_points, 2), points_vec)?;

    // Create values with a test function
    let mut values_vec = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        // Test function: f(x,y) = x² + y²
        let value = f64::powi(x, 2) + f64::powi(y, 2);

        values_vec.push(value);
    }

    let values = Array1::from_vec(values_vec);

    // Create the interpolator
    println!("Creating Sibson interpolator...");
    let interpolator = make_sibson_interpolator(points.clone(), values.clone())?;

    // Generate a set of query points including points outside the domain
    let mut query_points = Vec::new();
    for x in (-5..15).map(|v| v as f64 * 0.2) {
        for y in (-5..15).map(|v| v as f64 * 0.2) {
            query_points.push(x);
            query_points.push(y);
        }
    }

    let queries = Array2::from_shape_vec((query_points.len() / 2, 2), query_points)?;

    // Try different extrapolation methods

    // 1. Nearest Neighbor extrapolation
    println!("\nUsing Nearest Neighbor extrapolation...");
    let nn_params = nearest_neighbor_extrapolation();
    let nn_results = interpolator.interpolate_or_extrapolate_multi(&queries.view(), &nn_params)?;

    // 2. Inverse Distance Weighting extrapolation
    println!("Using Inverse Distance Weighting extrapolation...");
    let idw_params = inverse_distance_extrapolation(5, 2.0);
    let idw_results =
        interpolator.interpolate_or_extrapolate_multi(&queries.view(), &idw_params)?;

    // 3. Linear Gradient extrapolation
    println!("Using Linear Gradient extrapolation...");
    let linear_params = linear_gradient_extrapolation();
    let linear_results =
        interpolator.interpolate_or_extrapolate_multi(&queries.view(), &linear_params)?;

    // 4. Constant Value extrapolation
    println!("Using Constant Value extrapolation...");
    let constant_params = constant_value_extrapolation(1.0);
    let constant_results =
        interpolator.interpolate_or_extrapolate_multi(&queries.view(), &constant_params)?;

    // Compare results at specific test points outside the domain
    let test_points = vec![(-0.5, -0.5), (1.5, 1.5), (2.0, 0.5), (-1.0, 0.5)];

    println!("\nComparison of extrapolation methods at test points:");
    println!("Point          | True Value | NN         | IDW        | Linear     | Constant");
    println!("-------------- | ---------- | ---------- | ---------- | ---------- | ----------");

    for &(x, y) in &test_points {
        // Compute true function value
        let true_value = f64::powi(x, 2) + f64::powi(y, 2);

        // Find the index of this point in our queries array
        let idx = queries
            .rows()
            .into_iter()
            .position(|row| f64::abs(row[0] - x) < 1e-10 && f64::abs(row[1] - y) < 1e-10);

        if let Some(idx) = idx {
            println!(
                "({:5.2}, {:5.2}) | {:10.6} | {:10.6} | {:10.6} | {:10.6} | {:10.6}",
                x,
                y,
                true_value,
                nn_results[idx],
                idw_results[idx],
                linear_results[idx],
                constant_results[idx],
            );
        }
    }

    // Compute and compare errors for each method
    println!("\nComputing Mean Squared Error (MSE) for each method...");

    // Define a function to calculate MSE
    let calculate_mse = |results: &Array1<f64>| -> f64 {
        let mut sum_squared_error = 0.0;
        let mut count = 0;

        for (i, row) in queries.rows().into_iter().enumerate() {
            let x = row[0];
            let y = row[1];

            // Skip points within the unit square (those are interpolated, not extrapolated)
            if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) {
                continue;
            }

            // Compute true function value
            let true_value = f64::powi(x, 2) + f64::powi(y, 2);

            // Compute squared error
            let error = true_value - results[i];
            sum_squared_error += f64::powi(error, 2);
            count += 1;
        }

        // Compute MSE
        sum_squared_error / count as f64
    };

    let nn_mse = calculate_mse(&nn_results);
    let idw_mse = calculate_mse(&idw_results);
    let linear_mse = calculate_mse(&linear_results);
    let constant_mse = calculate_mse(&constant_results);

    println!("Nearest Neighbor MSE: {:.6}", nn_mse);
    println!("Inverse Distance Weighting MSE: {:.6}", idw_mse);
    println!("Linear Gradient MSE: {:.6}", linear_mse);
    println!("Constant Value MSE: {:.6}", constant_mse);

    // Determine the best method based on MSE
    let mut methods = [
        ("Nearest Neighbor", nn_mse),
        ("Inverse Distance Weighting", idw_mse),
        ("Linear Gradient", linear_mse),
        ("Constant Value", constant_mse),
    ];

    methods.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nRanking of extrapolation methods by MSE (lower is better):");
    for (i, (method, mse)) in methods.iter().enumerate() {
        println!("{}. {}: {:.6}", i + 1, method, mse);
    }

    // Explain why this ranking occurs for this particular function
    println!("\nNote: For the quadratic function f(x,y) = x² + y², the Linear Gradient");
    println!("extrapolation typically performs well because the function's derivatives");
    println!("increase linearly with distance from the origin. In contrast, the Constant");
    println!("Value method ignores the function's behavior, and Nearest Neighbor doesn't");
    println!("account for the function's growth rate outside the domain.");

    Ok(())
}
