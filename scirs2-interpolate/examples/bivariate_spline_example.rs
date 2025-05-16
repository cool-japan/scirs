use ndarray::{array, Array2};
use scirs2_interpolate::{BivariateInterpolator, RectBivariateSpline};

fn main() {
    println!("Bivariate Spline Interpolation Example");
    println!("=====================================\n");

    // Example 1: RectBivariateSpline on a rectangular grid - simplified with a smaller grid
    println!("Example 1: RectBivariateSpline on a rectangular grid (simplified)");

    // Create x and y coordinates (1D arrays)
    let x = array![0.0f64, 1.0, 2.0];
    let y = array![0.0f64, 1.0, 2.0];

    // Create 2D grid values (z = x^2 + y^2)
    let mut z = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            z[[i, j]] = f64::powi(x[i], 2) + f64::powi(y[j], 2);
        }
    }

    // Create the bivariate spline interpolator
    let rect_spline =
        RectBivariateSpline::new(&x.view(), &y.view(), &z.view(), None, 1, 1, None).unwrap();

    // Evaluate at a single point
    let xi = array![0.5f64];
    let yi = array![0.5f64];

    println!("Input grid points:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:5.1} ", z[[i, j]]);
        }
        println!();
    }

    println!("\nAttempting to interpolate at x=0.5, y=0.5");
    println!(
        "Exact value at (0.5, 0.5): {:5.1}",
        f64::powi(xi[0], 2) + f64::powi(yi[0], 2)
    );

    match rect_spline.evaluate(&xi.view(), &yi.view(), false) {
        Ok(result) => println!("Interpolated value: {:5.1}", result[[0, 0]]),
        Err(e) => println!("Error: {}", e),
    }

    // Example 2: SmoothBivariateSpline with scattered data - simplified
    println!("\n\nExample 2: SmoothBivariateSpline with scattered data (simplified)");

    // Create scattered data - with enough points for kx=1, ky=1 (need at least 4 points)
    let x_scattered = array![0.0f64, 0.0, 1.0, 1.0];
    let y_scattered = array![0.0f64, 1.0, 0.0, 1.0];
    let z_scattered = array![0.0f64, 1.0, 1.0, 2.0];

    // Create the smooth bivariate spline using the builder pattern
    let smooth_spline = scirs2_interpolate::SmoothBivariateSplineBuilder::new(
        &x_scattered.view(),
        &y_scattered.view(),
        &z_scattered.view(),
    )
    .with_degrees(1, 1)
    .build()
    .unwrap();

    // Evaluate at a single point
    let xi = array![0.5f64];
    let yi = array![0.5f64];

    println!("Input scattered points: (x, y) -> z");
    for i in 0..x_scattered.len() {
        println!(
            "({:.1}, {:.1}) -> {:.1}",
            x_scattered[i], y_scattered[i], z_scattered[i]
        );
    }

    println!("\nAttempting to interpolate at x=0.5, y=0.5");
    println!(
        "Exact value at (0.5, 0.5): {:5.1}",
        f64::powi(xi[0], 2) + f64::powi(yi[0], 2)
    );

    match smooth_spline.evaluate(&xi.view(), &yi.view(), false) {
        Ok(result) => println!("Interpolated value: {:5.1}", result[[0, 0]]),
        Err(e) => println!("Error: {}", e),
    }

    println!("\nNote: The current implementation of SmoothBivariateSpline is a placeholder.");
    println!("In a complete implementation, the interpolation would better fit the data.");
}
