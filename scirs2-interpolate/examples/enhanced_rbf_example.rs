use ndarray::{Array1, Array2};
use scirs2_interpolate::advanced::enhanced_rbf::{
    make_accurate_rbf, make_auto_rbf, make_fast_rbf, EnhancedRBFInterpolator, EnhancedRBFKernel,
    KernelType,
};
use scirs2_interpolate::advanced::rbf::{RBFInterpolator, RBFKernel};

fn main() {
    println!("Enhanced RBF Interpolation Examples");
    println!("==================================\n");

    // Example 1: Basic Comparison with Standard RBF
    println!("Example 1: Comparing Standard and Enhanced RBF");
    println!("--------------------------------------------");

    // Create 2D points in a grid
    let n_points = 11;
    let mut points = Vec::with_capacity(n_points * n_points);
    let mut values = Vec::with_capacity(n_points * n_points);

    for i in 0..n_points {
        for j in 0..n_points {
            let x = i as f64 / (n_points as f64 - 1.0) * 4.0 - 2.0;
            let y = j as f64 / (n_points as f64 - 1.0) * 4.0 - 2.0;

            points.push(x);
            points.push(y);

            // Function with two peaks: f(x,y) = exp(-x²-y²) + 0.5*exp(-(x-1)²-(y-1)²)
            let f = f64::exp(-x * x - y * y)
                + 0.5 * f64::exp(-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0));
            values.push(f);
        }
    }

    let points_array = Array2::from_shape_vec((n_points * n_points, 2), points).unwrap();
    let values_array = Array1::from_vec(values);

    // Create a standard RBF interpolator
    let standard_rbf = RBFInterpolator::new(
        &points_array.view(),
        &values_array.view(),
        RBFKernel::Gaussian,
        1.0,
    )
    .unwrap();

    // Create an enhanced RBF interpolator with similar parameters
    let enhanced_rbf = EnhancedRBFInterpolator::builder()
        .with_standard_kernel(RBFKernel::Gaussian)
        .with_epsilon(1.0)
        .build(&points_array.view(), &values_array.view())
        .unwrap();

    // Test interpolation at some points
    let test_points = vec![[0.5, 0.5], [1.5, -0.5], [-1.0, 1.0], [0.0, 0.0]];

    let test_array = Array2::from_shape_vec(
        (test_points.len(), 2),
        test_points.iter().flat_map(|p| vec![p[0], p[1]]).collect(),
    )
    .unwrap();

    // Evaluate with both interpolators
    let standard_result = standard_rbf.interpolate(&test_array.view()).unwrap();
    let enhanced_result = enhanced_rbf.interpolate(&test_array.view()).unwrap();

    // Display the results
    println!("Interpolation results at test points:");
    println!("  Point       | Standard RBF  | Enhanced RBF");
    println!("------------------------------------------");

    for i in 0..test_points.len() {
        println!(
            "  ({:5.2}, {:5.2}) | {:12.6} | {:12.6}",
            test_points[i][0], test_points[i][1], standard_result[i], enhanced_result[i]
        );
    }

    // Show the exact function values for comparison
    println!("\nExact function values at test points:");
    for point in &test_points {
        let x = point[0];
        let y = point[1];
        let exact =
            (-x * x - y * y).exp() + 0.5 * (-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0)).exp();
        println!("  ({:5.2}, {:5.2}) = {:12.6}", x, y, exact);
    }

    // Example 2: Comparing Different Enhanced Kernels
    println!("\nExample 2: Comparing Different Enhanced Kernels");
    println!("-------------------------------------------");

    // Create several enhanced RBF interpolators with different kernels
    let kernels = vec![
        ("Gaussian", KernelType::Standard(RBFKernel::Gaussian)),
        (
            "Multiquadric",
            KernelType::Standard(RBFKernel::Multiquadric),
        ),
        (
            "Matern 3/2",
            KernelType::Enhanced(EnhancedRBFKernel::Matern32),
        ),
        (
            "Matern 5/2",
            KernelType::Enhanced(EnhancedRBFKernel::Matern52),
        ),
        (
            "Wendland",
            KernelType::Enhanced(EnhancedRBFKernel::Wendland),
        ),
        (
            "Polyharmonic k=3",
            KernelType::Enhanced(EnhancedRBFKernel::Polyharmonic(3)),
        ),
    ];

    // Sample a smaller subset of points for this example
    let mut sparse_points = Vec::new();
    let mut sparse_values = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64 / 4.0 * 4.0 - 2.0;
            let y = j as f64 / 4.0 * 4.0 - 2.0;

            sparse_points.push(x);
            sparse_points.push(y);

            // Same function as before
            let f = f64::exp(-x * x - y * y)
                + 0.5 * f64::exp(-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0));
            sparse_values.push(f);
        }
    }

    let sparse_points_array = Array2::from_shape_vec((25, 2), sparse_points).unwrap();
    let sparse_values_array = Array1::from_vec(sparse_values);

    // Create interpolators with different kernels
    let mut interpolators = Vec::new();
    for (name, kernel) in &kernels {
        let interp = EnhancedRBFInterpolator::builder()
            .with_kernel(*kernel)
            .with_epsilon(1.0)
            .build(&sparse_points_array.view(), &sparse_values_array.view())
            .unwrap();

        interpolators.push((name, interp));
    }

    // Test interpolation at a grid of points
    let n_grid = 21;
    let mut grid_points = Vec::with_capacity(n_grid * n_grid * 2);
    for i in 0..n_grid {
        for j in 0..n_grid {
            let x = i as f64 / (n_grid as f64 - 1.0) * 4.0 - 2.0;
            let y = j as f64 / (n_grid as f64 - 1.0) * 4.0 - 2.0;
            grid_points.push(x);
            grid_points.push(y);
        }
    }

    let grid_array = Array2::from_shape_vec((n_grid * n_grid, 2), grid_points).unwrap();

    // Calculate exact values for comparison
    let mut exact_values = Vec::with_capacity(n_grid * n_grid);
    for i in 0..n_grid {
        for j in 0..n_grid {
            let x = i as f64 / (n_grid as f64 - 1.0) * 4.0 - 2.0;
            let y = j as f64 / (n_grid as f64 - 1.0) * 4.0 - 2.0;
            let f = f64::exp(-x * x - y * y)
                + 0.5 * f64::exp(-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0));
            exact_values.push(f);
        }
    }

    let exact_array = Array1::from_vec(exact_values);

    // Evaluate and compute errors for each kernel
    println!("Mean Square Error for each kernel:");
    for (name, interp) in &interpolators {
        let result = interp.interpolate(&grid_array.view()).unwrap();

        // Compute mean square error
        let mut sum_sq_error = 0.0;
        for i in 0..result.len() {
            let error = result[i] - exact_array[i];
            sum_sq_error += error * error;
        }
        let mse = sum_sq_error / result.len() as f64;

        println!("  {}: {:.6e}", name, mse);
    }

    // Example 3: Automatic Parameter Selection
    println!("\nExample 3: Automatic Parameter Selection");
    println!("-------------------------------------");

    // Generate data with noise
    let n_noisy = 30;
    let mut noisy_points = Vec::with_capacity(n_noisy * 2);
    let mut noisy_values = Vec::with_capacity(n_noisy);

    let mut rng = rand::rng();
    use rand::Rng;

    for _ in 0..n_noisy {
        let x = rng.random_range(-2.0..2.0);
        let y = rng.random_range(-2.0..2.0);

        noisy_points.push(x);
        noisy_points.push(y);

        // Function with noise
        let f = f64::exp(-x * x - y * y)
            + 0.5 * f64::exp(-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0));
        let noise = rng.random_range(-0.05..0.05);
        noisy_values.push(f + noise);
    }

    let noisy_points_array = Array2::from_shape_vec((n_noisy, 2), noisy_points).unwrap();
    let noisy_values_array = Array1::from_vec(noisy_values);

    // Create interpolators with different strategies
    let auto_rbf = make_auto_rbf(&noisy_points_array.view(), &noisy_values_array.view()).unwrap();
    let accurate_rbf =
        make_accurate_rbf(&noisy_points_array.view(), &noisy_values_array.view()).unwrap();
    let fast_rbf = make_fast_rbf(&noisy_points_array.view(), &noisy_values_array.view()).unwrap();

    // Display the properties of each interpolator
    println!("Auto RBF:      {}", auto_rbf.description());
    println!("Accurate RBF:  {}", accurate_rbf.description());
    println!("Fast RBF:      {}", fast_rbf.description());

    // Test interpolation at specific points
    let test_array = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();

    let auto_result = auto_rbf.interpolate(&test_array.view()).unwrap();
    let accurate_result = accurate_rbf.interpolate(&test_array.view()).unwrap();
    let fast_result = fast_rbf.interpolate(&test_array.view()).unwrap();

    // Exact value
    let x = 0.5;
    let y = 0.5;
    let exact =
        f64::exp(-x * x - y * y) + 0.5 * f64::exp(-(x - 1.0) * (x - 1.0) - (y - 1.0) * (y - 1.0));

    println!("\nInterpolation at (0.5, 0.5):");
    println!("  Exact:      {:.6}", exact);
    println!(
        "  Auto:       {:.6} (error: {:.6})",
        auto_result[0],
        f64::abs(auto_result[0] - exact)
    );
    println!(
        "  Accurate:   {:.6} (error: {:.6})",
        accurate_result[0],
        f64::abs(accurate_result[0] - exact)
    );
    println!(
        "  Fast:       {:.6} (error: {:.6})",
        fast_result[0],
        f64::abs(fast_result[0] - exact)
    );

    // Example 4: Multi-scale RBF for Complex Surfaces
    println!("\nExample 4: Multi-scale RBF for Complex Surfaces");
    println!("------------------------------------------");

    // Create a more complex function
    let n_complex = 50;
    let mut complex_points = Vec::with_capacity(n_complex * 2);
    let mut complex_values = Vec::with_capacity(n_complex);

    for _ in 0..n_complex {
        let x = rng.random_range(-3.0..3.0);
        let y = rng.random_range(-3.0..3.0);

        complex_points.push(x);
        complex_points.push(y);

        // Function with multiple scales: global trend + local features
        let global = 0.1 * (x * x + y * y); // Quadratic bowl
        let medium = 0.5 * f64::exp(-(x * x + y * y) / 4.0); // Medium-scale Gaussian
        let local = 0.3 * f64::exp(-(x * x + y * y) * 4.0); // Small-scale Gaussian
        let spike = if x * x + y * y < 0.2 { 0.8 } else { 0.0 }; // Sharp feature

        complex_values.push(global + medium + local + spike);
    }

    let complex_points_array = Array2::from_shape_vec((n_complex, 2), complex_points).unwrap();
    let complex_values_array = Array1::from_vec(complex_values);

    // Create single-scale and multi-scale interpolators
    let single_scale = EnhancedRBFInterpolator::builder()
        .with_enhanced_kernel(EnhancedRBFKernel::Wendland)
        .with_epsilon(1.0)
        .with_multiscale(false)
        .build(&complex_points_array.view(), &complex_values_array.view())
        .unwrap();

    let multi_scale = EnhancedRBFInterpolator::builder()
        .with_enhanced_kernel(EnhancedRBFKernel::Wendland)
        .with_epsilon(1.0)
        .with_multiscale(true)
        .with_scale_parameters(Array1::from_vec(vec![0.1, 0.5, 2.0])) // Multiple scales
        .build(&complex_points_array.view(), &complex_values_array.view())
        .unwrap();

    // Calculate leave-one-out cross-validation error
    let single_loo = single_scale.leave_one_out_cv().unwrap();
    let multi_loo = multi_scale.leave_one_out_cv().unwrap();

    println!("Leave-one-out cross-validation error:");
    println!("  Single-scale RBF: {:.6e}", single_loo);
    println!("  Multi-scale RBF:  {:.6e}", multi_loo);
    println!(
        "  Improvement:      {:.2}%",
        100.0 * (1.0 - multi_loo / single_loo)
    );

    // Calculate interpolation error at sample points
    let (single_mse, _, single_max) = single_scale.calculate_error().unwrap();
    let (multi_mse, _, multi_max) = multi_scale.calculate_error().unwrap();

    println!("\nInterpolation error at sample points:");
    println!(
        "  Single-scale RBF MSE: {:.6e}, Max Error: {:.6}",
        single_mse, single_max
    );
    println!(
        "  Multi-scale RBF MSE:  {:.6e}, Max Error: {:.6}",
        multi_mse, multi_max
    );

    println!("\nAll examples completed successfully!");
}
