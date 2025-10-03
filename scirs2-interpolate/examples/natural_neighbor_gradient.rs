//! Example demonstrating gradient estimation for Natural Neighbor interpolation
//!
//! This example shows how to use the gradient estimation feature of
//! Voronoi-based Natural Neighbor interpolation methods.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use scirs2_interpolate::voronoi::{
    make_sibson_interpolator, GradientEstimation, InterpolateWithGradient, InterpolationMethod,
    NaturalNeighborInterpolator,
};
use std::error::Error;

// Define a test function and its analytical gradient
#[allow(dead_code)]
fn test_function(x: f64, y: f64) -> f64 {
    // Function: f(x,y) = sin(x)*cos(y)
    x.sin() * y.cos()
}

#[allow(dead_code)]
fn analytical_gradient(x: f64, y: f64) -> Vec<f64> {
    // Gradient of f(x,y) = sin(x)*cos(y):
    // df/dx = cos(x)*cos(y)
    // df/dy = -sin(x)*sin(y)
    vec![x.cos() * y.cos(), -x.sin() * y.sin()]
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    // Generate scattered data points
    let n_points = 100;
    let mut rng = scirs2_core::random::rng();

    // Create points in a 2D domain
    let mut points_vec = Vec::with_capacity(n_points * 2);
    let mut values_vec = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        let x = rng.random_range(0.0..2.0 * std::f64::consts::PI);
        let y = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        points_vec.push(x);
        points_vec.push(y);

        // Compute the function value
        values_vec.push(test_function(x, y));
    }

    let points = Array2::from_shape_vec((n_points, 2), points_vec)?;
    let values = Array1::from_vec(values_vec);

    // Create a Sibson interpolator
    println!("Creating Sibson interpolator...");
    let interpolator = make_sibson_interpolator(points.clone(), values.clone())?;

    // Create some test points
    let test_points = vec![
        vec![0.5, 0.5],
        vec![1.0, 1.0],
        vec![1.5, 1.5],
        vec![2.0, 2.0],
    ];

    println!("\nComputing gradients at test points:");
    println!("Point            | Interpolated Value | Estimated Gradient           | Analytical Gradient");
    println!("---------------- | ------------------ | ---------------------------- | ----------------------------");

    for point in &test_points {
        let x = point[0];
        let y = point[1];
        let query = Array1::from_vec(point.clone());

        // Get the interpolated value
        let interpolated_value = interpolator.interpolate(&query.view())?;

        // Compute the gradient using our gradient estimation
        let gradient = interpolator.gradient(&query.view())?;

        // Compute the analytical gradient for comparison
        let analytical = analytical_gradient(x, y);

        // Display the results
        println!(
            "({:6.3}, {:6.3}) | {:18.15} | ({:12.9}, {:12.9}) | ({:12.9}, {:12.9})",
            x, y, interpolated_value, gradient[0], gradient[1], analytical[0], analytical[1]
        );
    }

    // Demonstrate the combined interpolate_with_gradient method
    println!("\nUsing interpolate_with_gradient method:");
    println!("Point            | Value and Gradient");
    println!("---------------- | --------------------------------------------");

    for point in &test_points {
        let query = Array1::from_vec(point.clone());

        // Get both value and gradient in one call
        let result = interpolator.interpolate_with_gradient(&query.view())?;

        println!(
            "({:6.3}, {:6.3}) | value: {:10.7}, gradient: ({:10.7}, {:10.7})",
            point[0], point[1], result.value, result.gradient[0], result.gradient[1]
        );
    }

    // Compare Sibson and Laplace gradient estimation
    println!("\nComparing Sibson and Laplace gradient estimation:");
    println!("Point            | Sibson Gradient               | Laplace Gradient              | Analytical Gradient");
    println!("---------------- | ------------------------------ | ----------------------------- | -----------------------------");

    let sibson = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Sibson,
    )?;

    let laplace = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Laplace,
    )?;

    for point in &test_points {
        let x = point[0];
        let y = point[1];
        let query = Array1::from_vec(point.clone());

        // Compute gradients using both methods
        let sibson_gradient = sibson.gradient(&query.view())?;
        let laplace_gradient = laplace.gradient(&query.view())?;

        // Compute the analytical gradient for comparison
        let analytical = analytical_gradient(x, y);

        // Display the results
        println!(
            "({:6.3}, {:6.3}) | ({:12.9}, {:12.9}) | ({:12.9}, {:12.9}) | ({:12.9}, {:12.9})",
            x,
            y,
            sibson_gradient[0],
            sibson_gradient[1],
            laplace_gradient[0],
            laplace_gradient[1],
            analytical[0],
            analytical[1]
        );
    }

    // Compute errors
    println!("\nComputing average errors over a regular grid:");

    let grid_size = 10;
    let x_vals = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, grid_size);
    let y_vals = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, grid_size);

    let mut sibson_error_sum = 0.0;
    let mut laplace_error_sum = 0.0;
    let mut count = 0;

    for &x in x_vals.iter() {
        for &y in y_vals.iter() {
            let query = Array1::from_vec(vec![x, y]);

            // Compute gradients
            let sibson_gradient = sibson.gradient(&query.view())?;
            let laplace_gradient = laplace.gradient(&query.view())?;

            // Compute analytical gradient
            let analytical = analytical_gradient(x, y);

            // Compute errors (Euclidean distance between estimated and analytical gradients)
            let sibson_error = ((sibson_gradient[0] - analytical[0]).powi(2)
                + (sibson_gradient[1] - analytical[1]).powi(2))
            .sqrt();

            let laplace_error = ((laplace_gradient[0] - analytical[0]).powi(2)
                + (laplace_gradient[1] - analytical[1]).powi(2))
            .sqrt();

            sibson_error_sum += sibson_error;
            laplace_error_sum += laplace_error;
            count += 1;
        }
    }

    let sibson_avg_error = sibson_error_sum / count as f64;
    let laplace_avg_error = laplace_error_sum / count as f64;

    println!("Average Sibson gradient error: {}", sibson_avg_error);
    println!("Average Laplace gradient error: {}", laplace_avg_error);

    if sibson_avg_error < laplace_avg_error {
        println!("Sibson's method provides more accurate gradients for this function");
    } else {
        println!("Laplace's method provides more accurate gradients for this function");
    }

    Ok(())
}
