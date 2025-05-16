use ndarray::{Array1, Axis};
use scirs2_interpolate::{
    make_adaptive_bspline, make_lsq_bspline, ExtrapolateMode, MultiscaleBSpline,
    RefinementCriterion,
};

fn main() {
    println!("Multiscale B-Splines with Adaptive Refinement Example");
    println!("===================================================\n");

    // Example 1: Simple function with adaptive refinement
    println!("Example 1: Basic Adaptive Refinement");
    println!("---------------------------------");

    // Create a sampled sine function with some noise
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 101);
    let noise = Array1::from_vec((0..101).map(|i| (i as f64 * 0.3).sin() * 0.05).collect());
    let y = x.mapv(|v| v.sin()) + &noise;

    // Create a regular B-spline with fixed number of knots
    let regular_spline =
        make_lsq_bspline(&x.view(), &y.view(), 8, 3, ExtrapolateMode::Error).unwrap();

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
    let y_regular = regular_spline.evaluate(&x_fine.view()).unwrap();
    let y_adaptive = adaptive_spline.evaluate(&x_fine.view()).unwrap();
    let y_exact = x_fine.mapv(|v| v.sin());

    // Calculate errors
    let mse_regular = y_regular
        .iter()
        .zip(y_exact.iter())
        .map(|(y_pred, y_true)| (y_pred - y_true).powi(2))
        .sum::<f64>()
        / y_regular.len() as f64;

    let mse_adaptive = y_adaptive
        .iter()
        .zip(y_exact.iter())
        .map(|(y_pred, y_true)| (y_pred - y_true).powi(2))
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
            0.5 + 2.0 / (1.0 + (-3.0 * (v - 5.0)).exp())
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
    let y_abs = spline_abs.evaluate(&x_trans.view()).unwrap();
    let y_curv = spline_curv.evaluate(&x_trans.view()).unwrap();
    let y_comb = spline_comb.evaluate(&x_trans.view()).unwrap();

    // Calculate actual values for comparison
    let y_actual = x_trans.mapv(|v| {
        if v < 4.0 {
            0.5
        } else if v > 6.0 {
            2.5
        } else {
            0.5 + 2.0 / (1.0 + (-3.0 * (v - 5.0)).exp())
        }
    });

    // Calculate errors in the transition region
    let mse_abs = y_abs
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| (y_pred - y_true).powi(2))
        .sum::<f64>()
        / y_abs.len() as f64;

    let mse_curv = y_curv
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| (y_pred - y_true).powi(2))
        .sum::<f64>()
        / y_curv.len() as f64;

    let mse_comb = y_comb
        .iter()
        .zip(y_actual.iter())
        .map(|(y_pred, y_true)| (y_pred - y_true).powi(2))
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
    let y3 = x3.mapv(|v| v.sin() + 0.5 * (v * 2.0).sin() + 0.1 * v.powi(2) / 10.0);

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
    let y_test = x_test.mapv(|v| v.sin() + 0.5 * (v * 2.0).sin() + 0.1 * v.powi(2) / 10.0);

    for level in 0..num_levels {
        // Switch to this level
        multi_spline.switch_level(level);

        // Evaluate at test points
        let y_approx = multi_spline.evaluate(&x_test.view()).unwrap();

        // Calculate MSE
        let mse = y_test
            .iter()
            .zip(y_approx.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
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
        let base = v.sin() / 2.0;
        let bumps =
            2.0 * (-5.0 * (v - 2.5).powi(2)).exp() + 1.5 * (-10.0 * (v - 7.0).powi(2)).exp();
        let oscillation = if v > 4.0 && v < 6.0 {
            0.5 * (5.0 * v).sin()
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
    let y_approx = adaptive_spline.evaluate(&x4.view()).unwrap();

    // Calculate error statistics
    let errors = y4
        .iter()
        .zip(y_approx.iter())
        .map(|(y_true, y_pred)| (y_true - y_pred).abs())
        .collect::<Vec<_>>();

    let max_error = errors
        .iter()
        .fold(0.0f64, |max, &err| if err > max { err } else { max });
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

    println!("\nError statistics:");
    println!("  Maximum absolute error: {:.6}", max_error);
    println!("  Mean absolute error:    {:.6}", mean_error);

    // Calculate first and second derivatives
    let first_deriv = adaptive_spline.derivative(1, &x4.view()).unwrap();
    let second_deriv = adaptive_spline.derivative(2, &x4.view()).unwrap();

    // Find maximum curvature regions
    let max_curvature =
        second_deriv.iter().fold(
            0.0f64,
            |max, &d2| if d2.abs() > max { d2.abs() } else { max },
        );

    println!("\nDerivative statistics:");
    println!(
        "  Maximum first derivative magnitude:  {:.6}",
        first_deriv
            .iter()
            .fold(0.0f64, |max, &d1| if d1.abs() > max {
                d1.abs()
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
