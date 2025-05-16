//! Example demonstrating parallel Natural Neighbor interpolation
//!
//! This example shows how to use parallel Voronoi-based Natural Neighbor
//! interpolation methods and compares performance with sequential methods.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use scirs2_interpolate::voronoi::{
    make_parallel_sibson_interpolator, make_sibson_interpolator, InterpolationMethod,
    NaturalNeighborInterpolator, ParallelConfig, ParallelNaturalNeighborInterpolator,
};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    // Generate scattered data points
    let n_points = 500;
    let mut rng = rand::thread_rng();

    // Create points in a 2D domain
    let mut points_vec = Vec::with_capacity(n_points * 2);
    for _ in 0..n_points {
        let x = rng.gen_range(0.0..10.0);
        let y = rng.gen_range(0.0..10.0);
        points_vec.push(x);
        points_vec.push(y);
    }

    let points = Array2::from_shape_vec((n_points, 2), points_vec)?;

    // Create values with a test function
    let mut values_vec = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        // Test function: peaks with exponential decay
        let value = 2.0 * (-(x - 5.0).powi(2) - (y - 5.0).powi(2) / 8.0).exp()
            + (-(x - 2.0).powi(2) - (y - 7.0).powi(2) / 2.0).exp()
            + 0.5 * (-(x - 8.0).powi(2) - (y - 3.0).powi(2) / 4.0).exp();

        values_vec.push(value);
    }

    let values = Array1::from_vec(values_vec);

    // Create interpolators
    println!("Creating sequential Sibson interpolator...");
    let sequential_interpolator = make_sibson_interpolator(points.clone(), values.clone())?;

    println!("Creating parallel Sibson interpolator...");
    let parallel_config = ParallelConfig {
        min_points_per_thread: 10,
        target_threads: None, // Use all available cores
    };
    let parallel_interpolator =
        make_parallel_sibson_interpolator(points.clone(), values.clone(), Some(parallel_config))?;

    // Generate a grid of query points for interpolation
    let grid_size = 100;
    let x_vals = Array1::linspace(0.0, 10.0, grid_size);
    let y_vals = Array1::linspace(0.0, 10.0, grid_size);

    // Generate the query points (flattened grid)
    let mut queries_vec = Vec::with_capacity(grid_size * grid_size * 2);
    for &x in x_vals.iter() {
        for &y in y_vals.iter() {
            queries_vec.push(x);
            queries_vec.push(y);
        }
    }
    let queries = Array2::from_shape_vec((grid_size * grid_size, 2), queries_vec)?;

    // Benchmark sequential interpolation
    println!(
        "Running sequential interpolation on {:?} points...",
        queries.nrows()
    );
    let start = Instant::now();
    let sequential_results = sequential_interpolator.interpolate_multi(&queries.view())?;
    let sequential_duration = start.elapsed();
    println!("Sequential interpolation took {:?}", sequential_duration);

    // Benchmark parallel interpolation
    println!(
        "Running parallel interpolation on {:?} points...",
        queries.nrows()
    );
    let start = Instant::now();
    let parallel_results = parallel_interpolator.interpolate_multi(&queries.view())?;
    let parallel_duration = start.elapsed();
    println!("Parallel interpolation took {:?}", parallel_duration);

    // Verify both methods produce the same results
    let max_diff = sequential_results
        .iter()
        .zip(parallel_results.iter())
        .map(|(s, p)| (*s - *p).abs())
        .fold(0.0, |a, b| a.max(b));

    println!(
        "Maximum difference between sequential and parallel results: {}",
        max_diff
    );

    // Calculate speedup
    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!("Speedup factor: {:.2}x", speedup);

    Ok(())
}
