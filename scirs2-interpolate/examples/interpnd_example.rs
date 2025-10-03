use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn}; // arrayとDimは使用していない
use scirs2_interpolate::interpnd::{
    make_interp_nd, make_interp_scattered, map_coordinates, ExtrapolateMode, InterpolationMethod,
    ScatteredInterpolationMethod, ScatteredInterpolatorParams,
};

#[allow(dead_code)]
fn main() {
    println!("N-dimensional Interpolation Examples");
    println!("===================================");

    // Example 1: Regular Grid Interpolation
    println!("\nExample 1: Regular Grid Interpolation (2D)");
    regular_grid_example();

    // Example 2: Scattered Data Interpolation
    println!("\nExample 2: Scattered Data Interpolation (2D)");
    scattered_data_example();

    // Example 3: Map Coordinates
    println!("\nExample 3: Map Coordinates - Resampling a Grid");
    map_coordinates_example();
}

#[allow(dead_code)]
fn regular_grid_example() {
    // Create a 2D grid
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let points = vec![x, y];

    // Create values on the grid (z = x^2 + y^2)
    let mut values = Array::zeros(IxDyn(&[3, 4]));
    for i in 0..3 {
        for j in 0..4 {
            let idx = [i, j];
            values[idx.as_slice()] = (i * i + j * j) as f64;
        }
    }

    println!("Grid values:");
    for i in 0..3 {
        for j in 0..4 {
            let idx = [i, j];
            print!("{:.1} ", values[idx.as_slice()]);
        }
        println!();
    }

    // Create the interpolator
    let interp = make_interp_nd(
        points,
        values,
        InterpolationMethod::Linear,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Test points for interpolation
    let test_points = Array2::from_shape_vec(
        (5, 2),
        vec![
            0.5, 0.5, // Between grid points
            1.5, 1.5, // Between grid points
            0.0, 0.0, // On grid point
            2.0, 3.0, // On grid point
            2.5, 2.5, // Outside grid (extrapolation)
        ],
    )
    .unwrap();

    // Perform interpolation
    let results = interp.__call__(&test_points.view()).unwrap();

    // Display results
    println!("\nInterpolation results:");
    for i in 0..test_points.shape()[0] {
        println!(
            "Point ({}, {}) -> Value: {:.2}",
            test_points[[i, 0]],
            test_points[[i, 1]],
            results[i]
        );
    }
}

#[allow(dead_code)]
fn scattered_data_example() {
    // Create scattered points in 2D
    let points = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, // Origin
            1.0, 0.0, // Right
            0.0, 1.0, // Up
            1.0, 1.0, // Top-right
            0.5, 0.5, // Center
            0.7, 0.3, // Random point
        ],
    )
    .unwrap();

    // Create values at those points (z = x^2 + y^2)
    let values = Array1::from_vec(vec![
        0.0,  // Origin
        1.0,  // Right
        1.0,  // Up
        2.0,  // Top-right
        0.5,  // Center
        0.58, // Random point
    ]);

    println!("Scattered data points:");
    for i in 0..points.shape()[0] {
        println!(
            "Point ({}, {}) -> Value: {:.2}",
            points[[i, 0]],
            points[[i, 1]],
            values[i]
        );
    }

    // Create the interpolator with IDW (Inverse Distance Weighting)
    let interp_idw = make_interp_scattered(
        points.clone(),
        values.clone(),
        ScatteredInterpolationMethod::IDW,
        ExtrapolateMode::Extrapolate,
        Some(ScatteredInterpolatorParams::IDW { power: 2.0 }),
    )
    .unwrap();

    // Create the interpolator with Nearest Neighbor
    let interp_nearest = make_interp_scattered(
        points.clone(),
        values.clone(),
        ScatteredInterpolationMethod::Nearest,
        ExtrapolateMode::Extrapolate,
        None,
    )
    .unwrap();

    // Test points for interpolation
    let test_points = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.25, 0.25, // Between points
            0.75, 0.75, // Between points
            1.5, 1.5, // Outside points (extrapolation)
        ],
    )
    .unwrap();

    // Perform interpolation with both methods
    let results_idw = interp_idw.__call__(&test_points.view()).unwrap();
    let results_nearest = interp_nearest.__call__(&test_points.view()).unwrap();

    // Display results
    println!("\nInterpolation results:");
    for i in 0..test_points.shape()[0] {
        println!(
            "Point ({:.2}, {:.2}) -> IDW: {:.3}, Nearest: {:.3}",
            test_points[[i, 0]],
            test_points[[i, 1]],
            results_idw[i],
            results_nearest[i]
        );
    }
}

#[allow(dead_code)]
fn map_coordinates_example() {
    // Create a 2D grid (3x3)
    let x_old = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y_old = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let old_grid = vec![x_old, y_old];

    // Create values on the old grid (z = x^2 + y^2)
    let mut old_values = Array::zeros(IxDyn(&[3, 3]));
    for i in 0..3 {
        for j in 0..3 {
            let idx = [i, j];
            old_values[idx.as_slice()] = (i * i + j * j) as f64;
        }
    }

    println!("Original grid values (3x3):");
    for i in 0..3 {
        for j in 0..3 {
            let idx = [i, j];
            print!("{:.1} ", old_values[idx.as_slice()]);
        }
        println!();
    }

    // Create a new, finer grid (5x5)
    let x_new = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    let y_new = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    let new_grid = vec![x_new, y_new];

    // Map the old values to the new grid
    let new_values =
        map_coordinates(old_grid, old_values, new_grid, InterpolationMethod::Linear).unwrap();

    // Display the new values
    println!("\nInterpolated grid values (5x5):");
    for i in 0..5 {
        for j in 0..5 {
            let idx = [i, j];
            print!("{:.1} ", new_values[idx.as_slice()]);
        }
        println!();
    }
}
