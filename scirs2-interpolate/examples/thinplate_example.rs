use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_interpolate::make_thinplate_interpolator;

/// Generate a grid of points in the 2D square [min, max] x [min, max]
fn generate_grid(min: f64, max: f64, resolution: usize) -> Array2<f64> {
    let step = (max - min) / (resolution as f64 - 1.0);
    let mut grid = Array2::zeros((resolution * resolution, 2));

    for i in 0..resolution {
        let x = min + (i as f64) * step;
        for j in 0..resolution {
            let y = min + (j as f64) * step;
            let idx = i * resolution + j;
            grid[[idx, 0]] = x;
            grid[[idx, 1]] = y;
        }
    }

    grid
}

/// Generate scattered data points with an underlying function f(x,y) = x^2 + sin(y)
fn generate_scattered_data(n_points: usize, add_noise: bool) -> (Array2<f64>, Array1<f64>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Create a seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate random (x,y) points in [-1, 1] x [-1, 1]
    let mut points = Array2::zeros((n_points, 2));
    for i in 0..n_points {
        points[[i, 0]] = 2.0 * rng.random::<f64>() - 1.0; // x in [-1, 1]
        points[[i, 1]] = 2.0 * rng.random::<f64>() - 1.0; // y in [-1, 1]
    }

    // Compute function values
    let mut values = Array1::zeros(n_points);
    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];

        // Function: f(x,y) = x^2 + sin(y)
        values[i] = f64::powi(x, 2) + f64::sin(y);

        // Add noise if specified
        if add_noise {
            values[i] += 0.05 * (2.0 * rng.random::<f64>() - 1.0);
        }
    }

    (points, values)
}

/// Compute function values for given points: f(x,y) = x^2 + sin(y)
fn true_function(points: &ArrayView2<f64>) -> Array1<f64> {
    let n_points = points.nrows();
    let mut values = Array1::zeros(n_points);

    for i in 0..n_points {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        values[i] = f64::powi(x, 2) + f64::sin(y);
    }

    values
}

/// Compute mean squared error between prediction and truth
fn compute_mse(pred: &ArrayView1<f64>, truth: &ArrayView1<f64>) -> f64 {
    let n = pred.len();
    let mut sum_sq_error = 0.0;

    for i in 0..n {
        let error = pred[i] - truth[i];
        sum_sq_error += error * error;
    }

    sum_sq_error / (n as f64)
}

fn main() {
    println!("Thin-Plate Spline Interpolation Example");
    println!("======================================\n");

    // 1. Basic thin-plate spline interpolation
    basic_tps_example();

    // 2. Smoothing with thin-plate splines
    smoothing_tps_example();

    // 3. Thin-plate splines for data with noise
    noisy_data_example();
}

fn basic_tps_example() {
    println!("1. Basic Thin-Plate Spline Interpolation");
    println!("---------------------------------------");

    // Generate scattered data points
    let (train_points, train_values) = generate_scattered_data(20, false);

    println!("Training data: {} points", train_points.nrows());
    println!("First few points:");
    for i in 0..5.min(train_points.nrows()) {
        println!(
            "  ({:.2}, {:.2}) → {:.4}",
            train_points[[i, 0]],
            train_points[[i, 1]],
            train_values[i]
        );
    }

    // Create a thin-plate spline interpolator
    let tps = make_thinplate_interpolator(
        &train_points.view(),
        &train_values.view(),
        0.0, // No smoothing
    )
    .unwrap();

    // Create test points on a grid
    let test_points = generate_grid(-1.0, 1.0, 5);

    // True function values
    let true_values = true_function(&test_points.view());

    // Interpolate at test points
    let predicted_values = tps.evaluate(&test_points.view()).unwrap();

    // Compute error
    let mse = compute_mse(&predicted_values.view(), &true_values.view());

    println!("\nEvaluation on {} test points:", test_points.nrows());
    println!("  Mean Squared Error: {:.6}", mse);

    println!("\nComparison at selected test points:");
    println!("    (x,y)     | Predicted | True Value");
    println!("-------------|-----------|------------");

    for i in 0..5.min(test_points.nrows()) {
        println!(
            "  ({:.2}, {:.2}) |   {:.4}   |   {:.4}",
            test_points[[i, 0]],
            test_points[[i, 1]],
            predicted_values[i],
            true_values[i]
        );
    }

    println!();
}

fn smoothing_tps_example() {
    println!("2. Effect of Smoothing Parameter");
    println!("------------------------------");

    // Generate scattered data points with noise
    let (train_points, train_values) = generate_scattered_data(30, true);

    println!("Training data with noise: {} points", train_points.nrows());

    // Create thin-plate spline interpolators with different smoothing
    let smoothing_params = [0.0, 0.001, 0.01, 0.1, 1.0];

    // Create test points on a grid
    let test_points = generate_grid(-1.0, 1.0, 10);

    // True function values
    let true_values = true_function(&test_points.view());

    println!("\nEffect of smoothing parameter on test error:");
    println!("Smoothing | Mean Squared Error");
    println!("----------|-------------------");

    for &smoothing in smoothing_params.iter() {
        // Create TPS with current smoothing
        let tps =
            make_thinplate_interpolator(&train_points.view(), &train_values.view(), smoothing)
                .unwrap();

        // Interpolate at test points
        let predicted_values = tps.evaluate(&test_points.view()).unwrap();

        // Compute error
        let mse = compute_mse(&predicted_values.view(), &true_values.view());

        println!("  {:.4}   |      {:.6}", smoothing, mse);
    }

    println!();
}

fn noisy_data_example() {
    println!("3. Handling Noisy Data");
    println!("--------------------");

    // Generate scattered data points with significant noise
    let (mut train_points, mut train_values) = generate_scattered_data(25, true);

    // Add an outlier
    if train_points.nrows() > 0 {
        // Replace the last point with an outlier
        let last_idx = train_points.nrows() - 1;
        train_points[[last_idx, 0]] = 0.0;
        train_points[[last_idx, 1]] = 0.0;
        train_values[last_idx] = 5.0; // This is far from the true value (should be ~0.0)

        println!("Added outlier at (0.0, 0.0) with value 5.0");
        println!(
            "True value at this point should be {:.4}",
            f64::powi(0.0, 2) + f64::sin(0.0)
        );
    }

    // Create thin-plate spline interpolators with and without smoothing
    let tps_exact = make_thinplate_interpolator(
        &train_points.view(),
        &train_values.view(),
        0.0, // No smoothing
    )
    .unwrap();

    let tps_smoothed = make_thinplate_interpolator(
        &train_points.view(),
        &train_values.view(),
        0.1, // With smoothing
    )
    .unwrap();

    // Evaluate at the outlier point
    let outlier_point = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let true_value = true_function(&outlier_point.view())[0];

    let exact_pred = tps_exact.evaluate(&outlier_point.view()).unwrap()[0];
    let smoothed_pred = tps_smoothed.evaluate(&outlier_point.view()).unwrap()[0];

    println!("\nInterpolation at the outlier point (0.0, 0.0):");
    println!("  True value:          {:.4}", true_value);
    println!("  Outlier value:       {:.4}", 5.0);
    println!(
        "  Exact TPS:           {:.4} (error: {:.4})",
        exact_pred,
        f64::abs(exact_pred - true_value)
    );
    println!(
        "  Smoothed TPS:        {:.4} (error: {:.4})",
        smoothed_pred,
        f64::abs(smoothed_pred - true_value)
    );

    // Evaluate at some nearby points
    let nearby_points =
        Array2::from_shape_vec((4, 2), vec![0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1]).unwrap();

    let true_nearby = true_function(&nearby_points.view());
    let exact_nearby = tps_exact.evaluate(&nearby_points.view()).unwrap();
    let smoothed_nearby = tps_smoothed.evaluate(&nearby_points.view()).unwrap();

    println!("\nEffect of the outlier on nearby points:");
    println!("    (x,y)    |  True  | Exact TPS | Smoothed TPS");
    println!("-------------|--------|-----------|-------------");

    for i in 0..nearby_points.nrows() {
        println!(
            "  ({:.2}, {:.2}) | {:.4} |   {:.4}  |     {:.4}",
            nearby_points[[i, 0]],
            nearby_points[[i, 1]],
            true_nearby[i],
            exact_nearby[i],
            smoothed_nearby[i]
        );
    }

    // Compute overall MSE on a grid
    let test_grid = generate_grid(-1.0, 1.0, 10);
    let true_grid = true_function(&test_grid.view());

    let exact_grid = tps_exact.evaluate(&test_grid.view()).unwrap();
    let smoothed_grid = tps_smoothed.evaluate(&test_grid.view()).unwrap();

    let exact_mse = compute_mse(&exact_grid.view(), &true_grid.view());
    let smoothed_mse = compute_mse(&smoothed_grid.view(), &true_grid.view());

    println!("\nOverall Mean Squared Error on test grid:");
    println!("  Exact TPS:     {:.6}", exact_mse);
    println!("  Smoothed TPS:  {:.6}", smoothed_mse);

    if smoothed_mse < exact_mse {
        println!(
            "\nSmoothing reduces the overall error by {:.2}%",
            100.0 * (exact_mse - smoothed_mse) / exact_mse
        );
    } else {
        println!("\nExact interpolation performs better on this dataset");
    }
}
