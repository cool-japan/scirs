//! Improved Griddata Linear Interpolation Demonstration
//!
//! This example demonstrates the enhanced griddata linear interpolation
//! implementation that now uses proper triangulation and barycentric
//! coordinates instead of falling back to RBF interpolation.

use ndarray::{Array1, Array2};
use scirs2_interpolate::griddata::{griddata, GriddataMethod};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Improved Griddata Linear Interpolation Demo ===\n");

    // 1. Demonstrate 1D linear interpolation
    println!("1. 1D Linear Interpolation:");
    demonstrate_1d_linear()?;

    // 2. Demonstrate 2D triangulation-based interpolation
    println!("\n2. 2D Triangulation-Based Interpolation:");
    demonstrate_2d_triangulation()?;

    // 3. Demonstrate higher-dimensional interpolation
    println!("\n3. Higher-Dimensional Interpolation:");
    demonstrate_nd_interpolation()?;

    // 4. Compare with other methods
    println!("\n4. Method Comparison:");
    compare_interpolation_methods()?;

    // 5. Show edge case handling
    println!("\n5. Edge Case Handling:");
    demonstrate_edge_cases()?;

    println!("\n=== Demo Complete ===");

    println!("\n✅ Enhanced Griddata Features:");
    println!("• Proper 1D linear interpolation with sorted points");
    println!("• 2D triangulation using barycentric coordinates");
    println!("• Higher-dimensional inverse distance weighting");
    println!("• Robust handling of degenerate cases");
    println!("• Efficient algorithms for different data sizes");
    println!("• No longer falls back to RBF for linear interpolation");

    Ok(())
}

fn demonstrate_1d_linear() -> Result<(), Box<dyn std::error::Error>> {
    // Create 1D test data
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]; // x^2

    let points = Array2::from_shape_vec((6, 1), x_data.clone())?;
    let values = Array1::from_vec(y_data.clone());

    // Create query points
    let query_x = vec![0.5, 1.5, 2.5, 3.5, 4.5];
    let query_points = Array2::from_shape_vec((5, 1), query_x.clone())?;

    // Perform linear interpolation
    let results = griddata(
        &points.view(),
        &values.view(),
        &query_points.view(),
        GriddataMethod::Linear,
        None,
        None, // workers parameter
    )?;

    println!("   Data points: {:?}", x_data);
    println!("   Values: {:?}", y_data);
    println!("   Query points: {:?}", query_x);
    println!("   Interpolated: {:?}", results.to_vec());

    // Verify linear interpolation (should be between neighboring values)
    for (i, &x) in query_x.iter().enumerate() {
        let expected = x * x; // Actual function value
        let interpolated = results[i];
        println!(
            "   f({:.1}) = {:.3} (exact: {:.3}, diff: {:.3})",
            x,
            interpolated,
            expected,
            (interpolated as f64 - expected as f64).abs()
        );
    }

    Ok(())
}

fn demonstrate_2d_triangulation() -> Result<(), Box<dyn std::error::Error>> {
    // Create 2D test data - unit square with function f(x,y) = x + y
    let data_points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0], // Corners
        [0.5, 0.0],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.5, 1.0], // Edge midpoints
        [0.5, 0.5], // Center
    ];

    let points = Array2::from_shape_vec((9, 2), data_points.iter().flatten().cloned().collect())?;

    let values = Array1::from_vec(data_points.iter().map(|&[x, y]| x + y).collect());

    // Create query grid
    let query_data = vec![
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75], // Quadrant centers
        [0.3, 0.7],
        [0.8, 0.2],
        [0.1, 0.9], // Other test points
    ];

    let query_points =
        Array2::from_shape_vec((7, 2), query_data.iter().flatten().cloned().collect())?;

    // Perform 2D linear interpolation
    let results = griddata(
        &points.view(),
        &values.view(),
        &query_points.view(),
        GriddataMethod::Linear,
        None,
        None, // workers parameter
    )?;

    println!("   Using triangulation-based linear interpolation:");
    for (i, &[x, y]) in query_data.iter().enumerate() {
        let expected = x + y;
        let interpolated = results[i];
        let error = (interpolated as f64 - expected as f64).abs();
        println!(
            "   f({:.2}, {:.2}) = {:.4} (exact: {:.4}, error: {:.4})",
            x, y, interpolated, expected, error
        );
    }

    println!(
        "   Maximum error: {:.2e}",
        query_data
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| (results[i] as f64 - (x + y) as f64).abs())
            .fold(0.0, f64::max)
    );

    Ok(())
}

fn demonstrate_nd_interpolation() -> Result<(), Box<dyn std::error::Error>> {
    // Create 3D test data
    let n_points = 20;
    let mut points_vec = Vec::with_capacity(n_points * 3);
    let mut values_vec = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let t = i as f64 / (n_points - 1) as f64;
        let x = t;
        let y = (2.0 * PI * t).sin();
        let z = (4.0 * PI * t).cos();

        points_vec.extend_from_slice(&[x, y, z]);
        values_vec.push(x + y.abs() + z.abs()); // Test function
    }

    let points = Array2::from_shape_vec((n_points, 3), points_vec)?;
    let values = Array1::from_vec(values_vec);

    // Create query points
    let query_data = vec![[0.25, 0.5, 0.0], [0.5, 0.0, 0.5], [0.75, -0.5, -0.5]];

    let query_points =
        Array2::from_shape_vec((3, 3), query_data.iter().flatten().cloned().collect())?;

    // Perform 3D interpolation
    let results = griddata(
        &points.view(),
        &values.view(),
        &query_points.view(),
        GriddataMethod::Linear,
        None,
        None, // workers parameter
    )?;

    println!("   3D interpolation using inverse distance weighting:");
    for (i, &[x, y, z]) in query_data.iter().enumerate() {
        let interpolated = results[i];
        println!("   f({:.2}, {:.2}, {:.2}) = {:.4}", x, y, z, interpolated);
    }

    Ok(())
}

fn compare_interpolation_methods() -> Result<(), Box<dyn std::error::Error>> {
    // Create 2D test data
    let data_points = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 1.5], [0.5, 1.0], [1.5, 0.5]];

    let points = Array2::from_shape_vec((5, 2), data_points.iter().flatten().cloned().collect())?;

    // Test function: f(x,y) = x² + y²
    let values = Array1::from_vec(data_points.iter().map(|&[x, y]| x * x + y * y).collect());

    let query_point = Array2::from_shape_vec((1, 2), vec![1.0, 0.5])?;

    // Compare different methods
    let methods = vec![
        (GriddataMethod::Linear, "Linear (Triangulation)"),
        (GriddataMethod::Nearest, "Nearest Neighbor"),
        (GriddataMethod::Cubic, "Cubic (RBF-based)"),
    ];

    println!("   Comparing methods at point (1.0, 0.5):");
    let expected = 1.0 * 1.0 + 0.5 * 0.5; // 1.25

    for (method, name) in methods {
        match griddata(
            &points.view(),
            &values.view(),
            &query_point.view(),
            method,
            None,
            None, // workers parameter
        ) {
            Ok(result) => {
                let error = (result[0] as f64 - expected as f64).abs();
                println!("   {}: {:.4} (error: {:.4})", name, result[0], error);
            }
            Err(e) => {
                println!("   {}: Error - {}", name, e);
            }
        }
    }

    Ok(())
}

fn demonstrate_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing edge cases:");

    // Test 1: Single data point
    let single_point = Array2::from_shape_vec((1, 2), vec![1.0, 1.0])?;
    let single_value = Array1::from_vec(vec![5.0]);
    let query = Array2::from_shape_vec((1, 2), vec![2.0, 2.0])?;

    match griddata(
        &single_point.view(),
        &single_value.view(),
        &query.view(),
        GriddataMethod::Linear,
        Some(99.0),
        None, // workers parameter
    ) {
        Ok(result) => println!("   Single point interpolation: {:.1}", result[0]),
        Err(e) => println!("   Single point error: {}", e),
    }

    // Test 2: Collinear points in 2D
    let collinear_points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])?;
    let collinear_values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let query_collinear = Array2::from_shape_vec((1, 2), vec![0.5, 0.5])?;

    match griddata(
        &collinear_points.view(),
        &collinear_values.view(),
        &query_collinear.view(),
        GriddataMethod::Linear,
        None,
        None, // workers parameter
    ) {
        Ok(result) => println!("   Collinear points interpolation: {:.2}", result[0]),
        Err(e) => println!("   Collinear points error: {}", e),
    }

    // Test 3: Outside convex hull
    let triangle_points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0])?;
    let triangle_values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let outside_query = Array2::from_shape_vec((1, 2), vec![2.0, 2.0])?;

    match griddata(
        &triangle_points.view(),
        &triangle_values.view(),
        &outside_query.view(),
        GriddataMethod::Linear,
        Some(-1.0),
        None, // workers parameter
    ) {
        Ok(result) => println!("   Outside convex hull: {:.2}", result[0]),
        Err(e) => println!("   Outside hull error: {}", e),
    }

    // Test 4: Exact data point match
    let exact_query = Array2::from_shape_vec((1, 2), vec![1.0, 0.0])?;
    match griddata(
        &triangle_points.view(),
        &triangle_values.view(),
        &exact_query.view(),
        GriddataMethod::Linear,
        None,
        None, // workers parameter
    ) {
        Ok(result) => println!("   Exact data point match: {:.1}", result[0]),
        Err(e) => println!("   Exact match error: {}", e),
    }

    Ok(())
}
