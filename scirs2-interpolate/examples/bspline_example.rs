use ndarray::{array, Array1};
use scirs2_interpolate::{
    generate_knots, make_interp_bspline, make_lsq_bspline, BSpline, BSplineExtrapolateMode,
};

fn main() {
    println!("SciRS2 B-Spline Interpolation Examples");
    println!("=====================================\n");

    // Example 1: Basic B-spline usage
    basic_bspline_example();

    // Example 2: B-spline derivatives and integration
    bspline_calculus_example();

    // Example 3: Interpolation with B-splines
    bspline_interpolation_example();

    // Example 4: Least-squares fitting with B-splines
    bspline_least_squares_example();
}

fn basic_bspline_example() {
    println!("Example 1: Basic B-spline Usage");
    println!("-------------------------------");

    // Create a quadratic B-spline (degree = 2)
    let knots = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let coeffs = array![-1.0, 2.0, 0.0, -1.0];
    let degree = 2;

    // Create the B-spline
    let spline = BSpline::new(
        &knots.view(),
        &coeffs.view(),
        degree,
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Evaluate at different points
    let points = array![1.5, 2.5, 3.5, 4.5];

    println!("Knots: {:?}", knots);
    println!("Coefficients: {:?}", coeffs);
    println!("Degree: {}\n", degree);

    println!("Evaluation results:");
    println!("  x   |   f(x)");
    println!("------|--------");

    for point in points.iter() {
        let value = spline.evaluate(*point).unwrap();
        println!(" {:.1} |  {:.4}", point, value);
    }

    // Create and evaluate a basis element
    println!("\nBasis Element Example:");

    let basis = BSpline::basis_element(
        degree,
        1, // Index of the basis function
        &knots.view(),
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    println!("B-spline basis element B_{{1,2}}");

    for point in array![1.5, 2.0, 2.5, 3.0, 3.5].iter() {
        let value = basis.evaluate(*point).unwrap();
        println!("B_{{1,2}}({:.1}) = {:.4}", point, value);
    }

    println!();
}

fn bspline_calculus_example() {
    println!("Example 2: B-spline Derivatives and Integration");
    println!("----------------------------------------------");

    // Create a cubic B-spline (degree = 3)
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let coeffs = array![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
    let degree = 3;

    // Create the B-spline
    let spline = BSpline::new(
        &knots.view(),
        &coeffs.view(),
        degree,
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Evaluate at different points
    let points = array![0.5, 1.5, 2.5, 3.5];

    println!("Cubic B-spline with knots: {:?}", knots);
    println!("and coefficients: {:?}\n", coeffs);

    println!("Evaluation and derivatives:");
    println!("  x   |   f(x)   |   f'(x)  |   f''(x)");
    println!("------|----------|----------|----------");

    for point in points.iter() {
        let value = spline.evaluate(*point).unwrap();
        let first_deriv = spline.derivative(*point, 1).unwrap();
        let second_deriv = spline.derivative(*point, 2).unwrap();

        println!(
            " {:.1} |  {:.4}  |  {:.4}  |  {:.4}",
            point, value, first_deriv, second_deriv
        );
    }

    // Compute some definite integrals
    let intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 4.0)];

    println!("\nDefinite integrals:");
    println!("Interval |  ∫f(x)dx");
    println!("---------|----------");

    for (a, b) in intervals.iter() {
        let integral = spline.integrate(*a, *b).unwrap();
        println!("[{:.1}, {:.1}] |  {:.4}", a, b, integral);
    }

    println!();
}

fn bspline_interpolation_example() {
    println!("Example 3: Interpolation with B-splines");
    println!("--------------------------------------");

    // Create some data points
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![0.0, 0.8, 0.9, 0.1, -0.8, -1.0]; // Some arbitrary function

    println!("Data points:");
    for i in 0..x.len() {
        print!("({:.1}, {:.1}) ", x[i], y[i]);
    }
    println!("\n");

    // Create an interpolating cubic B-spline
    let cubic_spline = make_interp_bspline(
        &x.view(),
        &y.view(),
        3, // Cubic spline
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Generate test points
    let test_points = array![0.5, 1.5, 2.5, 3.5, 4.5];

    println!("Cubic B-spline interpolation results:");
    println!("  x   |   f(x)");
    println!("------|--------");

    for point in test_points.iter() {
        let value = cubic_spline.evaluate(*point).unwrap();
        println!(" {:.1} |  {:.4}", point, value);
    }

    // Try different knot generation styles
    println!("\nDifferent knot placement styles:");

    let styles = ["uniform", "average", "clamped"];

    for style in styles.iter() {
        let knots = generate_knots(&x.view(), 3, style).unwrap();
        println!("{} knots: {:?}", style, knots);
    }

    println!();
}

fn bspline_least_squares_example() {
    println!("Example 4: Least-squares Fitting with B-splines");
    println!("----------------------------------------------");

    // Create more data points (with some noise)
    let x = array![
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
        3.6, 3.8, 4.0
    ];

    // Function with noise: y = sin(x) + noise
    let mut y = Array1::zeros(x.len());
    for (i, &xi) in x.iter().enumerate() {
        // Add some random noise (simulated here with a deterministic pattern)
        let noise = 0.05 * f64::sin(i as f64 / 10.0);
        y[i] = f64::sin(xi) + noise;
    }

    println!("Data points (showing first 5):");
    for i in 0..5 {
        print!("({:.1}, {:.4}) ", x[i], y[i]);
    }
    println!("... and {} more points", x.len() - 5);

    // Create a set of knots for least-squares fitting
    // Fewer knots than data points for a smoother fit
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];

    // Do the least-squares fit
    let lsq_spline = make_lsq_bspline(
        &x.view(),
        &y.view(),
        &knots.view(),
        3,    // Cubic spline
        None, // No weights
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Generate test points
    let test_points = array![0.0, 1.0, 2.0, 3.0, 4.0];

    println!("\nLeast-squares B-spline fit results:");
    println!("  x   |   f(x)   |   sin(x)  |   Error");
    println!("------|----------|-----------|--------");

    for point in test_points.iter() {
        let value = lsq_spline.evaluate(*point).unwrap();
        let true_value = f64::sin(*point);
        let error = value - true_value;

        println!(
            " {:.1} |  {:.6} |  {:.6} |  {:.6}",
            point, value, true_value, error
        );
    }

    // Compare with a weighted fit (higher weights near x=2)
    let mut weights = Array1::ones(x.len());
    for (i, &xi) in x.iter().enumerate() {
        // More weight near x=2
        let weight_factor = f64::exp(-2.0 * f64::powi(xi - 2.0, 2));
        weights[i] = 1.0 + 5.0 * weight_factor;
    }

    let weighted_spline = make_lsq_bspline(
        &x.view(),
        &y.view(),
        &knots.view(),
        3, // Cubic spline
        Some(&weights.view()),
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    println!("\nWeighted least-squares B-spline fit (higher weights near x=2):");
    println!("  x   |  Weighted f(x) |  Unweighted f(x)");
    println!("------|----------------|------------------");

    for point in test_points.iter() {
        let weighted_value = weighted_spline.evaluate(*point).unwrap();
        let unweighted_value = lsq_spline.evaluate(*point).unwrap();

        println!(
            " {:.1} |     {:.6}    |      {:.6}",
            point, weighted_value, unweighted_value
        );
    }

    println!();
}
