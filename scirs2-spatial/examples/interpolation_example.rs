use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::Rng;
use scirs2_spatial::interpolate::{
    IDWInterpolator, NaturalNeighborInterpolator, RBFInterpolator, RBFKernel,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spatial Interpolation Examples");
    println!("=============================\n");

    // Create a simple test dataset
    let points = array![
        [0.0, 0.0], // 0: bottom-left
        [1.0, 0.0], // 1: bottom-right
        [0.0, 1.0], // 2: top-left
        [1.0, 1.0], // 3: top-right
        [0.5, 0.5], // 4: center
    ];

    // Generate values using a test function
    let values = generate_test_function(&points);

    println!("Test dataset:");
    println!("  Points:");
    for i in 0..points.nrows() {
        println!("    {}: [{:.2}, {:.2}]", i, points[[i, 0]], points[[i, 1]]);
    }

    println!("  Values:");
    for i in 0..values.len() {
        println!("    {}: {:.4}", i, values[i]);
    }

    // Define a grid of points for evaluation
    let grid_size = 5;
    let grid_points = create_grid(0.0, 1.0, 0.0, 1.0, grid_size);

    // ===== Natural Neighbor Interpolation =====
    println!("\nNatural Neighbor Interpolation");
    println!("------------------------------");

    let nn_interp = NaturalNeighborInterpolator::new(&points.view(), &values.view())?;
    let nn_results = nn_interp.interpolate_many(&grid_points.view())?;

    println!("Natural Neighbor interpolation results on a {grid_size}x{grid_size} grid:");
    print_grid(grid_size, &nn_results);

    // ===== Radial Basis Function Interpolation =====
    println!("\nRadial Basis Function Interpolation");
    println!("---------------------------------");

    // Try different kernels
    let kernels = [
        (RBFKernel::Gaussian, "Gaussian"),
        (RBFKernel::Multiquadric, "Multiquadric"),
        (RBFKernel::InverseMultiquadric, "Inverse Multiquadric"),
        (RBFKernel::ThinPlateSpline, "Thin Plate Spline"),
    ];

    for &(kernel, name) in &kernels {
        let rbf_interp = RBFInterpolator::new(&points.view(), &values.view(), kernel, None, None)?;

        let rbf_results = rbf_interp.interpolate_many(&grid_points.view())?;

        println!("\nRBF interpolation with {name} kernel:");
        print_grid(grid_size, &rbf_results);
    }

    // ===== Inverse Distance Weighting Interpolation =====
    println!("\nInverse Distance Weighting Interpolation");
    println!("-------------------------------------");

    // Try different power values
    let powers = [1.0, 2.0, 4.0];

    for &power in &powers {
        let idw_interp = IDWInterpolator::new(&points.view(), &values.view(), power, None)?;

        let idw_results = idw_interp.interpolate_many(&grid_points.view())?;

        println!("\nIDW interpolation with power={power}:");
        print_grid(grid_size, &idw_results);
    }

    // ===== Comparison on a larger random dataset =====
    println!("\nComparison on a Larger Random Dataset");
    println!("-----------------------------------");

    // Generate random points
    let n_points = 50;
    let random_points = generate_random_points(n_points, 0.0, 1.0, 0.0, 1.0);

    // Generate values using the test function
    let random_values = generate_test_function(&random_points);

    println!("Generated {n_points} random points");

    // Define evaluation grid
    let eval_grid = create_grid(0.0, 1.0, 0.0, 1.0, 10);

    // Create interpolators
    let nn_interp = NaturalNeighborInterpolator::new(&random_points.view(), &random_values.view())?;

    let rbf_interp = RBFInterpolator::new(
        &random_points.view(),
        &random_values.view(),
        RBFKernel::Gaussian,
        None,
        None,
    )?;

    let idw_interp = IDWInterpolator::new(&random_points.view(), &random_values.view(), 2.0, None)?;

    // Interpolate and calculate errors
    let true_values = generate_test_function(&eval_grid);

    let nn_pred = nn_interp.interpolate_many(&eval_grid.view())?;
    let rbf_pred = rbf_interp.interpolate_many(&eval_grid.view())?;
    let idw_pred = idw_interp.interpolate_many(&eval_grid.view())?;

    let nn_rmse = calculate_rmse(&true_values, &nn_pred);
    let rbf_rmse = calculate_rmse(&true_values, &rbf_pred);
    let idw_rmse = calculate_rmse(&true_values, &idw_pred);

    println!("Root Mean Square Error (RMSE) on evaluation grid:");
    println!("  Natural Neighbor: {nn_rmse:.6}");
    println!("  RBF (Gaussian):   {rbf_rmse:.6}");
    println!("  IDW (power=2):    {idw_rmse:.6}");

    println!("\nExample completed successfully!");
    Ok(())
}

/// Generate test function values for a set of points
#[allow(dead_code)]
fn generate_test_function(points: &Array2<f64>) -> Array1<f64> {
    let n = points.nrows();
    let mut values = Array1::zeros(n);

    // Test function: f(x,y) = sin(pi*x) * cos(pi*y) + (x-0.5)^2 + (y-0.5)^2
    for i in 0..n {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        values[i] = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos()
            + (x - 0.5).powi(2)
            + (y - 0.5).powi(2);
    }

    values
}

/// Create a regular grid of points
#[allow(dead_code)]
fn create_grid(x_min: f64, x_max: f64, y_min: f64, y_max: f64, size: usize) -> Array2<f64> {
    let n_points = size * size;
    let mut grid = Array2::zeros((n_points, 2));

    let x_step = (x_max - x_min) / (size - 1) as f64;
    let y_step = (y_max - y_min) / (size - 1) as f64;

    let mut idx = 0;
    for i in 0..size {
        let y = y_min + i as f64 * y_step;
        for j in 0..size {
            let x = x_min + j as f64 * x_step;
            grid[[idx, 0]] = x;
            grid[[idx, 1]] = y;
            idx += 1;
        }
    }

    grid
}

/// Generate random points in a given range
#[allow(dead_code)]
fn generate_random_points(n: usize, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let mut points = Array2::zeros((n, 2));

    for i in 0..n {
        points[[i, 0]] = rng.random_range(x_min..x_max);
        points[[i, 1]] = rng.random_range(y_min..y_max);
    }

    points
}

/// Print interpolation results as a grid
#[allow(dead_code)]
fn print_grid(size: usize, values: &Array1<f64>) {
    for i in 0..size {
        let mut row = String::new();
        for j in 0..size {
            let idx = i * size + j;
            row.push_str(&format!("{:.4}  ", values[idx]));
        }
        println!("  {row}");
    }
}

/// Calculate the root mean square error between two arrays
#[allow(dead_code)]
fn calculate_rmse(truth: &Array1<f64>, pred: &Array1<f64>) -> f64 {
    let n = truth.len();
    let mut sum_sq_err = 0.0;

    for i in 0..n {
        let err = truth[i] - pred[i];
        sum_sq_err += err * err;
    }

    (sum_sq_err / n as f64).sqrt()
}
