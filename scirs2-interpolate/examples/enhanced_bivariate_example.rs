use ndarray::{array, Array1, Array2};
use scirs2_interpolate::{
    BivariateInterpolator, RectBivariateSpline, SmoothBivariateSplineBuilder,
};

/// Create a 2D test function: f(x,y) = sin(πx) * cos(πy)
fn test_function(x: f64, y: f64) -> f64 {
    let pi = std::f64::consts::PI;
    f64::sin(pi * x) * f64::cos(pi * y)
}

/// Generate a 2D grid of data points
fn generate_grid_data(nx: usize, ny: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let x = Array1::linspace(0.0, 1.0, nx);
    let y = Array1::linspace(0.0, 1.0, ny);

    let mut z = Array2::zeros((nx, ny));

    for i in 0..nx {
        for j in 0..ny {
            z[[i, j]] = test_function(x[i], y[j]);
        }
    }

    (x, y, z)
}

/// Generate scattered data points
fn generate_scattered_data(n_points: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Create a seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    let mut x = Array1::zeros(n_points);
    let mut y = Array1::zeros(n_points);
    let mut z = Array1::zeros(n_points);

    for i in 0..n_points {
        x[i] = rng.random::<f64>();
        y[i] = rng.random::<f64>();
        z[i] = test_function(x[i], y[i]);
    }

    (x, y, z)
}

/// Compute mean squared error between predicted and actual values
fn compute_mse(predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
    let mut sum_sq_error = 0.0;
    let n = predicted.len();

    for i in 0..predicted.shape()[0] {
        for j in 0..predicted.shape()[1] {
            let error = predicted[[i, j]] - actual[[i, j]];
            sum_sq_error += error * error;
        }
    }

    sum_sq_error / (n as f64)
}

fn main() {
    println!("Enhanced Bivariate Spline Interpolation Example");
    println!("=============================================\n");

    // Example 1: Comparison of different spline degrees on a rectangular grid
    rect_bivariate_example();

    // Example 2: Smooth bivariate spline with different smoothing parameters
    smooth_bivariate_example();

    // Example 3: Bivariate spline derivatives and integration
    bivariate_calculus_example();
}

fn rect_bivariate_example() {
    println!("1. RectBivariateSpline with Different Degrees");
    println!("-------------------------------------------");

    // Generate a grid of data points
    let (x, y, z) = generate_grid_data(10, 10);

    println!("Grid data: {} x {} points", x.len(), y.len());

    // Create bivariate splines with different degrees
    let spline_linear =
        RectBivariateSpline::new(&x.view(), &y.view(), &z.view(), None, 1, 1, None).unwrap();

    let spline_quadratic =
        RectBivariateSpline::new(&x.view(), &y.view(), &z.view(), None, 2, 2, None).unwrap();

    let spline_cubic =
        RectBivariateSpline::new(&x.view(), &y.view(), &z.view(), None, 3, 3, None).unwrap();

    // Create a finer grid for testing
    let x_test = Array1::linspace(0.0, 1.0, 20);
    let y_test = Array1::linspace(0.0, 1.0, 20);

    // Evaluate all splines on the test grid
    let result_linear = spline_linear
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();
    let result_quadratic = spline_quadratic
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();
    let result_cubic = spline_cubic
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();

    // Create the ground truth for comparison
    let mut z_true = Array2::zeros((x_test.len(), y_test.len()));
    for i in 0..x_test.len() {
        for j in 0..y_test.len() {
            z_true[[i, j]] = test_function(x_test[i], y_test[j]);
        }
    }

    // Calculate mean squared errors
    let mse_linear = compute_mse(&result_linear, &z_true);
    let mse_quadratic = compute_mse(&result_quadratic, &z_true);
    let mse_cubic = compute_mse(&result_cubic, &z_true);

    println!("\nMean Squared Errors:");
    println!("  Linear (kx=ky=1):     {:.6e}", mse_linear);
    println!("  Quadratic (kx=ky=2):  {:.6e}", mse_quadratic);
    println!("  Cubic (kx=ky=3):      {:.6e}", mse_cubic);

    // Compare at a specific point
    let test_x = array![0.5];
    let test_y = array![0.5];

    let val_linear = spline_linear
        .evaluate(&test_x.view(), &test_y.view(), false)
        .unwrap()[[0, 0]];
    let val_quadratic = spline_quadratic
        .evaluate(&test_x.view(), &test_y.view(), false)
        .unwrap()[[0, 0]];
    let val_cubic = spline_cubic
        .evaluate(&test_x.view(), &test_y.view(), false)
        .unwrap()[[0, 0]];
    let val_true = test_function(0.5, 0.5);

    println!("\nValues at (0.5, 0.5):");
    println!("  True value:           {:.6}", val_true);
    println!(
        "  Linear spline:        {:.6} (error: {:.6})",
        val_linear,
        (val_linear - val_true).abs()
    );
    println!(
        "  Quadratic spline:     {:.6} (error: {:.6})",
        val_quadratic,
        (val_quadratic - val_true).abs()
    );
    println!(
        "  Cubic spline:         {:.6} (error: {:.6})",
        val_cubic,
        (val_cubic - val_true).abs()
    );

    println!();
}

fn smooth_bivariate_example() {
    println!("2. SmoothBivariateSpline with Different Smoothing Parameters");
    println!("-------------------------------------------------------");

    // Generate scattered data points with some noise
    let (x, y, mut z) = generate_scattered_data(200);

    // Add some noise to the z values
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(99);

    for i in 0..z.len() {
        z[i] += 0.1 * (2.0 * rng.random::<f64>() - 1.0);
    }

    println!("Generated {} scattered data points with noise", x.len());

    // Create splines with different smoothing parameters
    let spline_exact = SmoothBivariateSplineBuilder::new(&x.view(), &y.view(), &z.view())
        .with_degrees(3, 3)
        .with_smoothing(0.0)
        .build()
        .unwrap();

    let spline_light = SmoothBivariateSplineBuilder::new(&x.view(), &y.view(), &z.view())
        .with_degrees(3, 3)
        .with_smoothing(0.1)
        .build()
        .unwrap();

    let spline_medium = SmoothBivariateSplineBuilder::new(&x.view(), &y.view(), &z.view())
        .with_degrees(3, 3)
        .with_smoothing(1.0)
        .build()
        .unwrap();

    let spline_heavy = SmoothBivariateSplineBuilder::new(&x.view(), &y.view(), &z.view())
        .with_degrees(3, 3)
        .with_smoothing(10.0)
        .build()
        .unwrap();

    // Create a grid for evaluation
    let x_test = Array1::linspace(0.0, 1.0, 20);
    let y_test = Array1::linspace(0.0, 1.0, 20);

    // Evaluate all splines on the test grid
    let result_exact = spline_exact
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();
    let result_light = spline_light
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();
    let result_medium = spline_medium
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();
    let result_heavy = spline_heavy
        .evaluate(&x_test.view(), &y_test.view(), true)
        .unwrap();

    // Create the ground truth for comparison
    let mut z_true = Array2::zeros((x_test.len(), y_test.len()));
    for i in 0..x_test.len() {
        for j in 0..y_test.len() {
            z_true[[i, j]] = test_function(x_test[i], y_test[j]);
        }
    }

    // Calculate mean squared errors
    let mse_exact = compute_mse(&result_exact, &z_true);
    let mse_light = compute_mse(&result_light, &z_true);
    let mse_medium = compute_mse(&result_medium, &z_true);
    let mse_heavy = compute_mse(&result_heavy, &z_true);

    println!("\nMean Squared Errors Against True Function:");
    println!("  Exact (s=0.0):      {:.6e}", mse_exact);
    println!("  Light (s=0.1):      {:.6e}", mse_light);
    println!("  Medium (s=1.0):     {:.6e}", mse_medium);
    println!("  Heavy (s=10.0):     {:.6e}", mse_heavy);

    // Find the best smoothing parameter
    let best_mse = mse_exact.min(mse_light.min(mse_medium.min(mse_heavy)));

    println!("\nBest smoothing parameter:");
    if best_mse == mse_exact {
        println!("  Exact interpolation (s=0.0) gives the best result");
    } else if best_mse == mse_light {
        println!("  Light smoothing (s=0.1) gives the best result");
    } else if best_mse == mse_medium {
        println!("  Medium smoothing (s=1.0) gives the best result");
    } else {
        println!("  Heavy smoothing (s=10.0) gives the best result");
    }

    println!();
}

fn bivariate_calculus_example() {
    println!("3. Bivariate Spline Derivatives and Integration");
    println!("--------------------------------------------");

    // Generate a grid of data points
    let (x, y, z) = generate_grid_data(15, 15);

    // Create a cubic spline
    let spline =
        RectBivariateSpline::new(&x.view(), &y.view(), &z.view(), None, 3, 3, None).unwrap();

    // Test derivatives at a specific point
    let test_x = array![0.5];
    let test_y = array![0.5];

    // Compute partial derivatives at (0.5, 0.5)
    let val = spline
        .evaluate(&test_x.view(), &test_y.view(), false)
        .unwrap()[[0, 0]];
    let dx = spline
        .evaluate_derivative(&test_x.view(), &test_y.view(), 1, 0, false)
        .unwrap()[[0, 0]];
    let dy = spline
        .evaluate_derivative(&test_x.view(), &test_y.view(), 0, 1, false)
        .unwrap()[[0, 0]];
    let dxy = spline
        .evaluate_derivative(&test_x.view(), &test_y.view(), 1, 1, false)
        .unwrap()[[0, 0]];

    // Compute the exact derivatives for the test function
    let pi = std::f64::consts::PI;
    let true_val = test_function(0.5, 0.5);
    let true_dx = pi * f64::cos(pi * 0.5) * f64::cos(pi * 0.5);
    let true_dy = -pi * f64::sin(pi * 0.5) * f64::sin(pi * 0.5);
    let true_dxy = -pi * pi * f64::cos(pi * 0.5) * f64::sin(pi * 0.5);

    println!("\nValues and derivatives at (0.5, 0.5):");
    println!("  f(0.5, 0.5):        {:.6} (true: {:.6})", val, true_val);
    println!("  df/dx:              {:.6} (true: {:.6})", dx, true_dx);
    println!("  df/dy:              {:.6} (true: {:.6})", dy, true_dy);
    println!("  d²f/dxdy:           {:.6} (true: {:.6})", dxy, true_dxy);

    // Calculate integral over different regions
    // Test the integral over [0,1] x [0,1]
    let integral_full = spline.integral(0.0, 1.0, 0.0, 1.0).unwrap();

    // True integral of sin(πx) * cos(πy) over [0,1] x [0,1] is 0
    let true_integral_full = 0.0;

    // Test integrals over smaller regions
    let integral_q1 = spline.integral(0.0, 0.5, 0.0, 0.5).unwrap();
    let integral_q2 = spline.integral(0.5, 1.0, 0.0, 0.5).unwrap();
    let integral_q3 = spline.integral(0.0, 0.5, 0.5, 1.0).unwrap();
    let integral_q4 = spline.integral(0.5, 1.0, 0.5, 1.0).unwrap();

    println!("\nDefinite integrals:");
    println!(
        "  ∫∫f(x,y) dxdy over [0,1]×[0,1]: {:.6} (true: {:.6})",
        integral_full, true_integral_full
    );
    println!("  Quarter regions:");
    println!("    [0.0, 0.5]×[0.0, 0.5]: {:.6}", integral_q1);
    println!("    [0.5, 1.0]×[0.0, 0.5]: {:.6}", integral_q2);
    println!("    [0.0, 0.5]×[0.5, 1.0]: {:.6}", integral_q3);
    println!("    [0.5, 1.0]×[0.5, 1.0]: {:.6}", integral_q4);

    // Verify that the sum of quadrants equals the full integral
    let sum_quadrants = integral_q1 + integral_q2 + integral_q3 + integral_q4;
    println!(
        "  Sum of quadrants: {:.6} (should equal full integral)",
        sum_quadrants
    );
    println!(
        "  Difference: {:.6e}",
        (sum_quadrants - integral_full).abs()
    );

    println!();
}
