use ndarray::Array1;
use scirs2_interpolate::{
    make_adaptive_bspline, make_lsq_bspline, BSplineExtrapolateMode, ExtrapolateMode,
    MultiscaleBSpline, RefinementCriterion,
};

// Helper function to calculate squared difference
fn squared_diff<T>(a: &T, b: &T) -> f64
where
    T: std::ops::Sub<Output = f64> + Copy,
{
    let diff = *a - *b;
    f64::powi(diff, 2)
}

// Helper function to calculate absolute difference
fn abs_diff<T>(a: &T, b: &T) -> f64
where
    T: std::ops::Sub<Output = f64> + Copy,
{
    let diff = *a - *b;
    f64::abs(diff)
}

// Helper function to extract scalar value from array
fn extract_scalar(arr: &Array1<f64>) -> f64 {
    arr[0]
}

fn main() {
    println!("Multiscale B-Splines with Adaptive Refinement Example");
    println!("===================================================\n");

    // Example 1: Simple function with adaptive refinement
    println!("Example 1: Basic Adaptive Refinement");
    println!("---------------------------------");

    // Create a sampled sine function with some noise
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 101);
    let noise = Array1::from_vec((0..101).map(|i| f64::sin(i as f64 * 0.3) * 0.05).collect());
    let y = x.mapv(f64::sin) + &noise;

    // Create a regular B-spline with fixed number of knots
    // Create knots for the B-spline
    let t = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 8);

    let regular_spline = make_lsq_bspline(
        &x.view(),
        &y.view(),
        &t.view(),
        3,
        None,
        BSplineExtrapolateMode::Error,
    )
    .unwrap();

    // Create a multiscale B-spline starting with few knots
    let mut adaptive_spline =
        MultiscaleBSpline::new(&x.view(), &y.view(), 4, 3, 5, 0.02, ExtrapolateMode::Error)
            .unwrap();

    println!("Initial B-spline (level 0):");
    println!(
        "  Number of knots: {}",
        adaptive_spline.get_knots_per_level()[0]
    );

    // Perform adaptive refinement
    let num_added = adaptive_spline
        .auto_refine(RefinementCriterion::AbsoluteError, 4)
        .unwrap();

    println!("After auto-refinement:");
    println!("  Number of refinement levels added: {}", num_added);
    println!(
        "  Total number of levels: {}",
        adaptive_spline.get_num_levels()
    );

    // Report knots at each level
    println!("\nKnots per level:");
    let knots_per_level = adaptive_spline.get_knots_per_level();
    for (i, &knots) in knots_per_level.iter().enumerate() {
        println!("  Level {}: {} knots", i, knots);
    }

    // Evaluate both splines on a fine grid
    let x_fine = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 201);
    let y_regular = Array1::from_vec(
        x_fine
            .iter()
            .map(|&x| regular_spline.evaluate(x).unwrap())
            .collect(),
    );
    let y_adaptive = Array1::from_vec(
        x_fine
            .iter()
            .map(|&x| {
                // Create a single-element array for evaluation
                let x_single = Array1::from_vec(vec![x]);
                adaptive_spline.evaluate(&x_single.view()).unwrap()[0]
            })
            .collect(),
    );
    let y_exact = x_fine.mapv(f64::sin);

    // Calculate errors
    let mse_regular = y_regular
        .iter()
        .zip(y_exact.iter())
        .map(|(y_pred, y_true)| squared_diff(y_pred, y_true))
        .sum::<f64>()
        / y_regular.len() as f64;

    let mse_adaptive = y_adaptive
        .iter()
        .zip(y_exact.iter())
        .map(|(y_pred, y_true)| squared_diff(y_pred, y_true))
        .sum::<f64>()
        / y_adaptive.len() as f64;

    println!("\nMean Square Error (versus exact sine):");
    println!("  Regular B-spline (8 knots):     {:.8}", mse_regular);
    println!("  Adaptive B-spline (final level): {:.8}", mse_adaptive);

    // Example 2: Function with sharp features
    println!("\nExample 2: Function with Sharp Features");
    println!("------------------------------------");

    // Create a function with a sharp feature (step function with smooth transition)
    let x2 = Array1::linspace(0.0, 10.0, 201);
    let y2 = x2.mapv(|v| {
        if v < 4.0 {
            0.5
        } else if v > 6.0 {
            2.5
        } else {
            // Smooth transition using sigmoid
            0.5 + 2.0 / (1.0 + f64::exp(-3.0 * (v - 5.0)))
        }
    });

    // Create adaptive B-splines with different refinement criteria
    println!("Creating adaptive B-splines with different refinement criteria...");

    let spline_abs = make_adaptive_bspline(
        &x2.view(),
        &y2.view(),
        5,
        3,
        0.05,
        RefinementCriterion::AbsoluteError,
        3,
        ExtrapolateMode::Error,
    )
    .unwrap();

    let spline_curv = make_adaptive_bspline(
        &x2.view(),
        &y2.view(),
        5,
        3,
        0.5,
        RefinementCriterion::Curvature,
        3,
        ExtrapolateMode::Error,
    )
    .unwrap();

    let spline_comb = make_adaptive_bspline(
        &x2.view(),
        &y2.view(),
        5,
        3,
        0.05,
        RefinementCriterion::Combined,
        3,
        ExtrapolateMode::Error,
    )
    .unwrap();

    println!("\nRefinement levels and knots:");
    println!(
        "  AbsoluteError criterion: {} levels, {} total knots",
        spline_abs.get_num_levels(),
        spline_abs.get_knots_per_level().last().unwrap()
    );

    println!(
        "  Curvature criterion:     {} levels, {} total knots",
        spline_curv.get_num_levels(),
        spline_curv.get_knots_per_level().last().unwrap()
    );

    println!(
        "  Combined criterion:      {} levels, {} total knots",
        spline_comb.get_num_levels(),
        spline_comb.get_knots_per_level().last().unwrap()
    );

    // Evaluate at points around the transition zone
    let x_trans = Array1::linspace(3.0, 7.0, 41);
    let y_abs = Array1::from_vec(
        x_trans
            .iter()
            .map(|&x| {
                {
                    // Create a single-element array for evaluation
                    let x_single = Array1::from_vec(vec![x]);
                    extract_scalar(&spline_abs.evaluate(&x_single.view()).unwrap())
                }
            })
            .collect(),
    );
    let y_curv = Array1::from_vec(
        x_trans
            .iter()
            .map(|&x| {
                {
                    // Create a single-element array for evaluation
                    let x_single = Array1::from_vec(vec![x]);
                    extract_scalar(&spline_curv.evaluate(&x_single.view()).unwrap())
                }
            })
            .collect(),
    );
    let y_comb = Array1::from_vec(
        x_trans
            .iter()
            .map(|&x| {
                {
                    // Create a single-element array for evaluation
                    let x_single = Array1::from_vec(vec![x]);
                    extract_scalar(&spline_comb.evaluate(&x_single.view()).unwrap())
                }
            })
            .collect(),
    );

    // Calculate actual values for comparison
    let y_actual = x_trans.mapv(|v| {
        if v < 4.0 {
            0.5
        } else if v > 6.0 {
            2.5
        } else {
            0.5 + 2.0 / (1.0 + f64::exp(-3.0 * (v - 5.0)))
        }
    });

    // Calculate errors in the transition region
    let mse_abs = y_abs
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| squared_diff(y_pred, y_true))
        .sum::<f64>()
        / y_abs.len() as f64;

    let mse_curv = y_curv
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| squared_diff(y_pred, y_true))
        .sum::<f64>()
        / y_curv.len() as f64;

    let mse_comb = y_comb
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| squared_diff(y_pred, y_true))
        .sum::<f64>()
        / y_comb.len() as f64;

    println!("\nMean Square Error in the transition region (3.0 to 7.0):");
    println!("  AbsoluteError criterion: {:.8}", mse_abs);
    println!("  Curvature criterion:     {:.8}", mse_curv);
    println!("  Combined criterion:      {:.8}", mse_comb);

    println!("\nObservation: The curvature-based criterion performs better at capturing the");
    println!("shape of the transition, while error-based adds knots where errors are largest.");

    // Example 3: Switching between levels
    println!("\nExample 3: Switching Between Refinement Levels");
    println!("-------------------------------------------");

    // Create a more complex function
    let x3 = Array1::linspace(0.0, 10.0, 201);
    let y3 = x3.mapv(|v| f64::sin(v) + 0.5 * f64::sin(v * 2.0) + 0.1 * f64::powi(v, 2) / 10.0);

    // Create a multiscale B-spline with multiple refinement levels
    let mut multi_spline = make_adaptive_bspline(
        &x3.view(),
        &y3.view(),
        5,
        3,
        0.01,
        RefinementCriterion::Combined,
        5,
        ExtrapolateMode::Error,
    )
    .unwrap();

    let num_levels = multi_spline.get_num_levels();
    println!(
        "Created multiscale B-spline with {} refinement levels",
        num_levels
    );

    // Calculate errors at each level
    println!("\nErrors at each refinement level:");
    let x_test = Array1::linspace(0.0, 10.0, 101);
    let y_test =
        x_test.mapv(|v| f64::sin(v) + 0.5 * f64::sin(v * 2.0) + 0.1 * f64::powi(v, 2) / 10.0);

    for level in 0..num_levels {
        // Switch to this level
        multi_spline.switch_level(level);

        // Evaluate at test points
        let y_approx = Array1::from_vec(
            x_test
                .iter()
                .map(|&x| {
                    {
                        // Create a single-element array for evaluation
                        let x_single = Array1::from_vec(vec![x]);
                        multi_spline.evaluate(&x_single.view()).unwrap()[0]
                    }
                })
                .collect(),
        );

        // Calculate MSE
        let mse = y_test
            .iter()
            .zip(y_approx.iter())
            .map(|(y_true, y_pred)| squared_diff(y_true, y_pred))
            .sum::<f64>()
            / y_test.len() as f64;

        println!(
            "  Level {}: MSE = {:.8}, Knots = {}",
            level,
            mse,
            multi_spline.get_knots_per_level()[level]
        );
    }

    // Switch back to finest level
    multi_spline.switch_level(num_levels - 1);

    println!("\nObservation: Error decreases with each refinement level, at the cost of");
    println!("increased complexity (more knots and coefficients).");

    // Example 4: Adaptively fitting complicated data
    println!("\nExample 4: Adaptively Fitting Complicated Data");
    println!("-------------------------------------------");

    // Create a function with multiple localized features
    let x4 = Array1::linspace(0.0, 10.0, 501);
    let y4 = x4.mapv(|v| {
        let base = f64::sin(v) / 2.0;
        let bumps = 2.0 * f64::exp(-5.0 * f64::powi(v - 2.5, 2))
            + 1.5 * f64::exp(-10.0 * f64::powi(v - 7.0, 2));
        let oscillation = if v > 4.0 && v < 6.0 {
            0.5 * f64::sin(5.0 * v)
        } else {
            0.0
        };

        base + bumps + oscillation
    });

    // Create a multiscale B-spline with aggressive refinement
    let adaptive_spline = make_adaptive_bspline(
        &x4.view(),
        &y4.view(),
        10,
        3,
        0.005,
        RefinementCriterion::Combined,
        4,
        ExtrapolateMode::Error,
    )
    .unwrap();

    println!("Created multiscale B-spline to fit complex data:");
    println!("  Number of levels: {}", adaptive_spline.get_num_levels());
    println!(
        "  Initial knots: {}",
        adaptive_spline.get_knots_per_level()[0]
    );
    println!(
        "  Final knots: {}",
        adaptive_spline.get_knots_per_level().last().unwrap()
    );

    // Evaluate the spline at the original points
    let y_approx = Array1::from_vec(
        x4.iter()
            .map(|&x| {
                // Create a single-element array for evaluation
                let x_single = Array1::from_vec(vec![x]);
                adaptive_spline.evaluate(&x_single.view()).unwrap()[0]
            })
            .collect(),
    );

    // Calculate error statistics
    let errors = y4
        .iter()
        .zip(y_approx.iter())
        .map(|(y_true, y_pred)| abs_diff(y_true, y_pred))
        .collect::<Vec<_>>();

    let max_error = errors
        .iter()
        .fold(0.0f64, |max, &err| if err > max { err } else { max });
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

    println!("\nError statistics:");
    println!("  Maximum absolute error: {:.6}", max_error);
    println!("  Mean absolute error:    {:.6}", mean_error);

    // Create arrays to store derivatives
    let mut first_deriv = Array1::zeros(x4.len());
    let mut second_deriv = Array1::zeros(x4.len());

    // Calculate derivatives at each point
    for (i, &x) in x4.iter().enumerate() {
        // Create a single-element array for this point
        let x_single = Array1::from_vec(vec![x]);

        // Get derivatives at this point
        let d1 = adaptive_spline.derivative(1, &x_single.view()).unwrap();
        let d2 = adaptive_spline.derivative(2, &x_single.view()).unwrap();

        // Store the derivatives
        first_deriv[i] = d1[0]; // Extract scalar value
        second_deriv[i] = d2[0]; // Extract scalar value
    }

    // Find maximum curvature regions
    let max_curvature = second_deriv.iter().fold(0.0f64, |max, &d2| {
        if f64::abs(d2) > max {
            f64::abs(d2)
        } else {
            max
        }
    });

    println!("\nDerivative statistics:");
    println!(
        "  Maximum first derivative magnitude:  {:.6}",
        first_deriv
            .iter()
            .fold(0.0f64, |max, &d1| if f64::abs(d1) > max {
                f64::abs(d1)
            } else {
                max
            })
    );
    println!(
        "  Maximum second derivative magnitude: {:.6}",
        max_curvature
    );

    println!("\nAll examples completed successfully!");
}
