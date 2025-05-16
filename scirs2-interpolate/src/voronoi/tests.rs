//! Tests for Voronoi-based interpolation methods

#[cfg(test)]
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};

#[cfg(test)]
use crate::voronoi::extrapolation::Extrapolation;
#[cfg(test)]
use crate::voronoi::gradient::{GradientEstimation, InterpolateWithGradient};

#[cfg(test)]
use crate::voronoi::extrapolation::{constant_value_extrapolation, inverse_distance_extrapolation, linear_gradient_extrapolation, nearest_neighbor_extrapolation};
#[cfg(test)]
use crate::voronoi::natural::{InterpolationMethod, NaturalNeighborInterpolator, make_laplace_interpolator, make_sibson_interpolator};
#[cfg(test)]
use crate::voronoi::parallel::{ParallelConfig, ParallelNaturalNeighborInterpolator, make_parallel_laplace_interpolator, make_parallel_sibson_interpolator};

// Test helper functions
#[cfg(test)]
fn create_3d_test_data() -> (Array2<f64>, Array1<f64>) {
    // Create a simple 3D test dataset
    let points = Array2::from_shape_vec(
        (8, 3),
        vec![
            0.0, 0.0, 0.0, // Cube corners
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            // Add a point in the center
        ],
    )
    .unwrap();

    // Function: f(x,y,z) = x + 2*y + 3*z
    let values = Array1::from_vec(vec![
        0.0, // f(0,0,0) = 0
        1.0, // f(1,0,0) = 1
        3.0, // f(1,1,0) = 3
        2.0, // f(0,1,0) = 2
        3.0, // f(0,0,1) = 3
        4.0, // f(1,0,1) = 4
        6.0, // f(1,1,1) = 6
        5.0, // f(0,1,1) = 5
    ]);

    (points, values)
}

#[test]
fn test_natural_neighbor_exact_points() {
    // Test that the interpolator returns exact values at data points
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create both types of interpolators
    let sibson = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Sibson,
    )
    .unwrap();

    let laplace = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Laplace,
    )
    .unwrap();

    // Test each data point
    for i in 0..points.nrows() {
        let point = points.row(i).to_owned();

        // Both interpolators should give exact values at data points
        let sibson_result = sibson.interpolate(&point.view()).unwrap();
        let laplace_result = laplace.interpolate(&point.view()).unwrap();

        assert_abs_diff_eq!(sibson_result, values[i], epsilon = 1e-10);
        assert_abs_diff_eq!(laplace_result, values[i], epsilon = 1e-10);
    }
}

#[test]
fn test_natural_neighbor_helpers() {
    // Test the helper functions for creating interpolators
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create interpolators using helper functions
    let sibson = make_sibson_interpolator(points.clone(), values.clone()).unwrap();
    let laplace = make_laplace_interpolator(points.clone(), values.clone()).unwrap();

    // Check the methods
    assert_eq!(sibson.method(), InterpolationMethod::Sibson);
    assert_eq!(laplace.method(), InterpolationMethod::Laplace);

    // Test interpolation at a point in the middle
    let mid_point = Array1::from_vec(vec![0.5, 0.5]);

    // This is an exact data point, so should give the exact value
    let sibson_result = sibson.interpolate(&mid_point.view()).unwrap();
    let laplace_result = laplace.interpolate(&mid_point.view()).unwrap();

    assert_abs_diff_eq!(sibson_result, 1.5, epsilon = 1e-10);
    assert_abs_diff_eq!(laplace_result, 1.5, epsilon = 1e-10);
}

#[test]
fn test_interpolate_multi() {
    // Test that the interpolator can handle multiple query points
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Create a set of query points
    let queries = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.25, 0.25, // Between (0,0) and (0.5,0.5)
            0.75, 0.75, // Between (1,1) and (0.5,0.5)
            0.5, 0.5, // Exact data point
        ],
    )
    .unwrap();

    // Interpolate multiple points
    let results = interpolator.interpolate_multi(&queries.view()).unwrap();

    // The middle point (0.5, 0.5) should be exactly 1.5
    assert_abs_diff_eq!(results[2], 1.5, epsilon = 1e-10);

    // Check that the other points are within reasonable bounds
    assert!(results[0] >= 0.0 && results[0] <= 1.5);
    assert!(results[1] >= 1.5 && results[1] <= 2.0);
}

#[test]
fn test_linear_function_reproduction() {
    // Test that natural neighbor interpolation can exactly reproduce a linear function
    // since it should have linear precision

    // Create a grid of points
    let mut points_vec = Vec::new();
    let mut values_vec = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64;
            let y = j as f64;

            points_vec.push(x);
            points_vec.push(y);

            // Linear function f(x,y) = 2x + 3y
            values_vec.push(2.0 * x + 3.0 * y);
        }
    }

    let points = Array2::from_shape_vec((25, 2), points_vec).unwrap();
    let values = Array1::from_vec(values_vec);

    // Create both types of interpolators
    let sibson = make_sibson_interpolator(points.clone(), values.clone()).unwrap();
    let laplace = make_laplace_interpolator(points.clone(), values.clone()).unwrap();

    // Test at several non-grid points
    let test_points = vec![(1.5, 2.3), (3.4, 1.7), (2.5, 2.5)];

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Expected value from the linear function
        let expected = 2.0 * x + 3.0 * y;

        // Interpolated values
        let sibson_result = sibson.interpolate(&query.view()).unwrap();
        let laplace_result = laplace.interpolate(&query.view()).unwrap();

        // Both methods should reproduce linear functions exactly (or very close)
        assert_abs_diff_eq!(sibson_result, expected, epsilon = 1e-4);
        assert_abs_diff_eq!(laplace_result, expected, epsilon = 1e-4);
    }
}

#[test]
fn test_3d_natural_neighbor_exact_points() {
    // Test that the interpolator returns exact values at data points for 3D
    let (points, values) = create_3d_test_data();

    // Create both types of interpolators
    let sibson = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Sibson,
    )
    .unwrap();

    let laplace = NaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Laplace,
    )
    .unwrap();

    // Test each data point
    for i in 0..points.nrows() {
        let point = points.row(i).to_owned();

        // Both interpolators should give exact values at data points
        let sibson_result = sibson.interpolate(&point.view()).unwrap();
        let laplace_result = laplace.interpolate(&point.view()).unwrap();

        assert_abs_diff_eq!(sibson_result, values[i], epsilon = 1e-10);
        assert_abs_diff_eq!(laplace_result, values[i], epsilon = 1e-10);
    }
}

#[test]
fn test_3d_linear_function_reproduction() {
    // Test that natural neighbor interpolation can exactly reproduce a linear function in 3D
    // Create a grid of 3D points
    let (points, values) = create_3d_test_data();

    // Create both types of interpolators
    let sibson = make_sibson_interpolator(points.clone(), values.clone()).unwrap();
    let laplace = make_laplace_interpolator(points.clone(), values.clone()).unwrap();

    // Test at several non-grid points
    let test_points = vec![
        (0.5, 0.5, 0.5), // Center of the cube
        (0.25, 0.75, 0.3),
        (0.8, 0.2, 0.6),
    ];

    for (x, y, z) in test_points {
        let query = Array1::from_vec(vec![x, y, z]);

        // Expected value from the linear function f(x,y,z) = x + 2*y + 3*z
        let expected = x + 2.0 * y + 3.0 * z;

        // Interpolated values
        let sibson_result = sibson.interpolate(&query.view()).unwrap();
        let laplace_result = laplace.interpolate(&query.view()).unwrap();

        // Both methods should reproduce linear functions exactly (or very close)
        // We use a larger epsilon for 3D due to the approximations in the implementation
        assert_abs_diff_eq!(sibson_result, expected, epsilon = 1e-2);
        assert_abs_diff_eq!(laplace_result, expected, epsilon = 1e-2);
    }
}

#[test]
fn test_voronoi_diagram_access() {
    // Test that we can access the underlying Voronoi diagram
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Access the voronoi diagram
    let diagram = interpolator.voronoi_diagram();

    // Basic checks
    assert_eq!(diagram.cells.len(), 5);
    assert_eq!(diagram.dim, 2);

    // Each cell should have a non-zero measure (area in 2D)
    for cell in &diagram.cells {
        assert!(cell.measure > 0.0);
    }
}

#[test]
fn test_method_setting() {
    // Test that we can change the interpolation method
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create interpolator with Sibson method
    let mut interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();
    assert_eq!(interpolator.method(), InterpolationMethod::Sibson);

    // Change to Laplace method
    interpolator.set_method(InterpolationMethod::Laplace);
    assert_eq!(interpolator.method(), InterpolationMethod::Laplace);

    // Interpolate at a point and ensure it still works
    let query = Array1::from_vec(vec![0.25, 0.25]);
    let result = interpolator.interpolate(&query.view()).unwrap();

    // Result should be a reasonable value
    assert!(result >= 0.0 && result <= 2.0);
}

#[test]
fn test_parallel_interpolation() {
    // Test that parallel interpolation gives the same results as sequential
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create both sequential and parallel interpolators
    let sequential = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    let config = ParallelConfig {
        min_points_per_thread: 2,
        target_threads: Some(2),
    };

    let parallel =
        make_parallel_sibson_interpolator(points.clone(), values.clone(), Some(config)).unwrap();

    // Create a set of query points
    let mut query_points = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            query_points.push(x as f64 * 0.25);
            query_points.push(y as f64 * 0.25);
        }
    }

    let queries = Array2::from_shape_vec((25, 2), query_points).unwrap();

    // Run both interpolation methods
    let sequential_results = sequential.interpolate_multi(&queries.view()).unwrap();
    let parallel_results = parallel.interpolate_multi(&queries.view()).unwrap();

    // Verify the results are the same
    for i in 0..queries.nrows() {
        assert_abs_diff_eq!(sequential_results[i], parallel_results[i], epsilon = 1e-10);
    }
}

#[test]
fn test_parallel_3d_interpolation() {
    // Test that parallel interpolation works for 3D data
    let (points, values) = create_3d_test_data();

    // Create parallel interpolator
    let config = ParallelConfig {
        min_points_per_thread: 2,
        target_threads: Some(2),
    };

    let parallel =
        make_parallel_sibson_interpolator(points.clone(), values.clone(), Some(config)).unwrap();

    // Create a set of query points inside the cube
    let mut query_points = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let x = i as f64 * 0.5;
                let y = j as f64 * 0.5;
                let z = k as f64 * 0.5;
                query_points.push(x);
                query_points.push(y);
                query_points.push(z);
            }
        }
    }

    let queries = Array2::from_shape_vec((27, 3), query_points).unwrap();

    // Run parallel interpolation
    let results = parallel.interpolate_multi(&queries.view()).unwrap();

    // Verify the results are reasonable
    for i in 0..queries.nrows() {
        let x = queries[[i, 0]];
        let y = queries[[i, 1]];
        let z = queries[[i, 2]];

        // The function we used is f(x,y,z) = x + 2*y + 3*z
        let expected = x + 2.0 * y + 3.0 * z;

        // Allow for some approximation error
        assert_abs_diff_eq!(results[i], expected, epsilon = 1e-2);
    }
}

#[test]
fn test_parallel_config() {
    // Test that we can change the parallel configuration
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);

    // Create parallel interpolator with default config
    let mut parallel = ParallelNaturalNeighborInterpolator::new(
        points.clone(),
        values.clone(),
        InterpolationMethod::Sibson,
        None,
    )
    .unwrap();

    // Update the config
    let new_config = ParallelConfig {
        min_points_per_thread: 5,
        target_threads: Some(4),
    };

    parallel.set_parallel_config(new_config);

    // Interpolate at a point and ensure it still works
    let query = Array1::from_vec(vec![0.25, 0.25]);
    let result = parallel.interpolate(&query.view()).unwrap();

    // Result should be a reasonable value
    assert!(result >= 0.0 && result <= 2.0);
}

#[test]
fn test_gradient_linear_function() {
    // Test that gradient estimation works correctly for a linear function
    let mut points_vec = Vec::new();
    let mut values_vec = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64;
            let y = j as f64;

            points_vec.push(x);
            points_vec.push(y);

            // Linear function f(x,y) = 2x + 3y
            values_vec.push(2.0 * x + 3.0 * y);
        }
    }

    let points = Array2::from_shape_vec((25, 2), points_vec).unwrap();
    let values = Array1::from_vec(values_vec);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Test at several points
    let test_points = vec![(1.5, 2.3), (3.4, 1.7), (2.5, 2.5)];

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Compute gradient
        let gradient = interpolator.gradient(&query.view()).unwrap();

        // For a linear function f(x,y) = 2x + 3y, the gradient is [2, 3]
        assert_abs_diff_eq!(gradient[0], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(gradient[1], 3.0, epsilon = 1e-4);
    }
}

#[test]
fn test_gradient_quadratic_function() {
    // Test that gradient estimation works correctly for a quadratic function
    let mut points_vec = Vec::new();
    let mut values_vec = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64;
            let y = j as f64;

            points_vec.push(x);
            points_vec.push(y);

            // Quadratic function f(x,y) = x^2 + y^2
            values_vec.push(x.powi(2) + y.powi(2));
        }
    }

    let points = Array2::from_shape_vec((25, 2), points_vec).unwrap();
    let values = Array1::from_vec(values_vec);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Test at several points
    let test_points = vec![(1.0, 2.0), (2.0, 1.0), (2.0, 2.0)];

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Compute gradient
        let gradient = interpolator.gradient(&query.view()).unwrap();

        // For f(x,y) = x^2 + y^2, the gradient is [2x, 2y]
        let expected_dx = 2.0 * x;
        let expected_dy = 2.0 * y;

        assert_abs_diff_eq!(gradient[0], expected_dx, epsilon = 0.2);
        assert_abs_diff_eq!(gradient[1], expected_dy, epsilon = 0.2);
    }
}

#[test]
fn test_interpolate_with_gradient() {
    // Test the combined interpolate_with_gradient method
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    // Linear function f(x,y) = 2x + 3y
    let values = Array1::from_vec(vec![0.0, 2.0, 3.0, 5.0, 2.5]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Test at a point
    let query = Array1::from_vec(vec![0.25, 0.25]);

    // Get both value and gradient
    let result = interpolator
        .interpolate_with_gradient(&query.view())
        .unwrap();

    // Check the value
    let expected_value = 2.0 * 0.25 + 3.0 * 0.25; // f(0.25, 0.25) = 1.25
    assert_abs_diff_eq!(result.value, expected_value, epsilon = 0.1);

    // Check the gradient
    assert_abs_diff_eq!(result.gradient[0], 2.0, epsilon = 0.1);
    assert_abs_diff_eq!(result.gradient[1], 3.0, epsilon = 0.1);
}

#[test]
fn test_3d_gradient() {
    // Test gradient estimation in 3D
    let (points, values) = create_3d_test_data();

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Test at points inside the cube
    let test_points = vec![(0.5, 0.5, 0.5), (0.25, 0.75, 0.3)];

    for (x, y, z) in test_points {
        let query = Array1::from_vec(vec![x, y, z]);

        // Compute gradient
        let gradient = interpolator.gradient(&query.view()).unwrap();

        // For f(x,y,z) = x + 2y + 3z, the gradient is [1, 2, 3]
        assert_abs_diff_eq!(gradient[0], 1.0, epsilon = 0.2);
        assert_abs_diff_eq!(gradient[1], 2.0, epsilon = 0.2);
        assert_abs_diff_eq!(gradient[2], 3.0, epsilon = 0.2);
    }
}

#[test]
fn test_extrapolation_nearest_neighbor() {
    // Test nearest neighbor extrapolation
    let points =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    // Function f(x,y) = x + y
    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Points outside the unit square
    let test_points = vec![(2.0, 2.0), (-1.0, 0.5)];

    // Create nearest neighbor extrapolation params
    let params = nearest_neighbor_extrapolation();

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Extrapolate
        let result = interpolator.extrapolate(&query.view(), &params).unwrap();

        // Verify that we get the value of the nearest data point
        // We don't know which is closest without computing distances,
        // but we can verify it matches one of the input points
        assert!(values.iter().any(|&v| ((v as f64) - (result as f64)).abs() < 1e-10f64));
    }
}

#[test]
fn test_extrapolation_inverse_distance() {
    // Test inverse distance weighting extrapolation
    let points =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    // Function f(x,y) = x + y
    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Create inverse distance weighting extrapolation params
    let params = inverse_distance_extrapolation(4, 2.0);

    // Test at a point outside the domain
    let query = Array1::from_vec(vec![2.0, 0.5]);

    // Extrapolate
    let result = interpolator.extrapolate(&query.view(), &params).unwrap();

    // All our points have values between 0 and 2, so the result should be in that range
    assert!(result >= 0.0 && result <= 2.0);

    // The result should be closer to the values of points near (1,0) and (1,1)
    // which are 1.0 and 2.0 respectively
    assert!(result > 0.5);
}

#[test]
fn test_extrapolation_linear_gradient() {
    // Test linear gradient extrapolation
    let points = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .unwrap();

    // Linear function f(x,y) = 2x + 3y
    let values = Array1::from_vec(vec![0.0, 2.0, 3.0, 5.0, 2.5]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Create linear gradient extrapolation params
    let params = linear_gradient_extrapolation();

    // Test at a point outside the domain
    let query = Array1::from_vec(vec![2.0, 2.0]);

    // Extrapolate
    let result = interpolator.extrapolate(&query.view(), &params).unwrap();

    // For a linear function, linear extrapolation should be quite accurate
    let expected = 2.0 * 2.0 + 3.0 * 2.0; // f(2,2) = 10
    assert_abs_diff_eq!(result, expected, epsilon = 1.0);
}

#[test]
fn test_extrapolation_constant_value() {
    // Test constant value extrapolation
    let points =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    // Function f(x,y) = x + y
    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Create constant value extrapolation params
    let constant_value = 42.0;
    let params = constant_value_extrapolation(constant_value);

    // Test at several points outside the domain
    let test_points = vec![(2.0, 2.0), (-1.0, 0.5), (10.0, -10.0)];

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Extrapolate
        let result = interpolator.extrapolate(&query.view(), &params).unwrap();

        // The result should be exactly the constant value
        assert_abs_diff_eq!(result, constant_value, epsilon = 1e-10);
    }
}

#[test]
fn test_interpolate_or_extrapolate() {
    // Test the combined interpolate_or_extrapolate method
    let points =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    // Function f(x,y) = x + y
    let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

    // Create interpolator
    let interpolator = make_sibson_interpolator(points.clone(), values.clone()).unwrap();

    // Create extrapolation params
    let params = nearest_neighbor_extrapolation();

    // Test at points inside and outside the domain
    let test_points = vec![
        (0.5, 0.5), // Inside (should be interpolated)
        (2.0, 2.0), // Outside (should be extrapolated)
    ];

    for (x, y) in test_points {
        let query = Array1::from_vec(vec![x, y]);

        // Interpolate or extrapolate
        let result = interpolator
            .interpolate_or_extrapolate(&query.view(), &params)
            .unwrap();

        if x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0 {
            // Inside the domain - should be close to x + y
            assert_abs_diff_eq!(result, x + y, epsilon = 0.2);
        } else {
            // Outside the domain - should be one of the input values (nearest neighbor)
            assert!(values.iter().any(|&v| ((v as f64) - (result as f64)).abs() < 1e-10f64));
        }
    }
}
