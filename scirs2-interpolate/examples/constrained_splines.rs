use ndarray::{s, Array1};
use scirs2_interpolate::bspline::ExtrapolateMode as BSplineExtrapolateMode;
use scirs2_interpolate::constrained::{ConstrainedSpline, Constraint};

fn main() {
    println!("Constrained Splines Examples");
    println!("===========================\n");

    // Create some test data that's not monotonic or convex
    let x = Array1::linspace(0.0, 10.0, 15);
    let y = x.mapv(|v| {
        f64::powi(v - 5.0, 2) * 0.1 + // Parabola
        f64::sin(v * 0.8) * 2.0 + // Add sine wave
        (v * 0.2) // Add linear trend
    });

    println!("Example 1: Monotone Increasing Spline");
    println!("-------------------------------------");
    // Create constraints for monotone increasing
    let constraint = Constraint::monotone_increasing(None, None);

    // Use interpolate method with fitting method
    let monotone_inc = ConstrainedSpline::penalized(
        &x.view(),
        &y.view(),
        vec![constraint],
        8,   // number of knots
        3,   // degree
        0.1, // smoothing parameter
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Evaluate on a finer grid
    let x_fine = Array1::linspace(0.0, 10.0, 100);
    // Compute function values at each point in x_fine
    let mut y_fine = Array1::zeros(x_fine.len());
    for (i, &x_val) in x_fine.iter().enumerate() {
        y_fine[i] = monotone_inc.evaluate(x_val).unwrap();
    }

    // Compute first derivatives to verify monotonicity
    let mut dy_fine = Array1::zeros(x_fine.len());
    for (i, &x_val) in x_fine.iter().enumerate() {
        dy_fine[i] = monotone_inc.derivative(x_val, 1).unwrap();
    }

    println!("Original data points: {:?}", y);
    println!(
        "First few values from monotonic fit: {:?}",
        &y_fine.slice(s![0..5])
    );
    println!(
        "First few derivatives (should all be positive): {:?}",
        &dy_fine.slice(s![0..5])
    );

    let min_deriv = dy_fine.fold(f64::INFINITY, |a, &b| a.min(b));
    println!(
        "Minimum derivative value: {} (should be >= 0 for monotone increasing)",
        min_deriv
    );

    println!("\nExample 2: Convex Spline");
    println!("----------------------");
    // Create constraints for convex
    let constraint = Constraint::convex(None, None);

    // Use least squares method with constraints
    let convex = ConstrainedSpline::least_squares(
        &x.view(),
        &y.view(),
        vec![constraint],
        10, // number of knots
        3,  // degree
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Compute function values
    let mut y_convex = Array1::zeros(x_fine.len());
    for (i, &x_val) in x_fine.iter().enumerate() {
        y_convex[i] = convex.evaluate(x_val).unwrap();
    }

    // Compute second derivatives to verify convexity
    let mut d2y_convex = Array1::zeros(x_fine.len());
    for (i, &x_val) in x_fine.iter().enumerate() {
        d2y_convex[i] = convex.derivative(x_val, 2).unwrap();
    }

    println!(
        "First few values from convex fit: {:?}",
        &y_convex.slice(s![0..5])
    );
    println!(
        "First few second derivatives (should all be positive for convex): {:?}",
        &d2y_convex.slice(s![0..5])
    );

    let min_d2 = d2y_convex.fold(f64::INFINITY, |a, &b| a.min(b));
    println!(
        "Minimum second derivative value: {} (should be >= 0 for convex)",
        min_d2
    );

    println!("\nExample 3: Multiple Constraints (Monotone + Bounded)");
    println!("--------------------------------------------------");

    // Create constraints
    let constraints = vec![
        Constraint::monotone_increasing(None, None),
        // Create an upper bound constraint at 15.0
        Constraint::upper_bound(None, None, 15.0),
        // Create a lower bound constraint at 0.0
        Constraint::lower_bound(None, None, 0.0),
    ];

    let multi_constraint = ConstrainedSpline::interpolate(
        &x.view(),
        &y.view(),
        constraints,
        3,
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Compute function values and derivatives
    let mut y_multi = Array1::zeros(x_fine.len());
    let mut dy_multi = Array1::zeros(x_fine.len());
    for (i, &x_val) in x_fine.iter().enumerate() {
        y_multi[i] = multi_constraint.evaluate(x_val).unwrap();
        dy_multi[i] = multi_constraint.derivative(x_val, 1).unwrap();
    }

    println!(
        "First few values from multi-constrained fit: {:?}",
        &y_multi.slice(s![0..5])
    );
    println!(
        "Min value: {} (should be >= 0)",
        y_multi.fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "Max value: {} (should be <= 15)",
        y_multi.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "Min derivative: {} (should be >= 0)",
        dy_multi.fold(f64::INFINITY, |a, &b| a.min(b))
    );

    println!("\nExample 4: Regional Constraints");
    println!("-----------------------------");

    // Create data with different behaviors in different regions
    let x_region = Array1::linspace(0.0, 10.0, 20);
    let y_region = x_region.mapv(|v| {
        if v < 3.0 {
            // Increasing region
            v * 1.5 + f64::sin(v)
        } else if v < 7.0 {
            // Decreasing region
            10.0 - (v - 3.0) * 1.2 + f64::sin(v * 2.0) * 0.5
        } else {
            // Convex region
            f64::powi(v - 7.0, 2) * 0.3 + 2.0
        }
    });

    // Different constraints in different regions
    let region_constraints = vec![
        // Monotone increasing in range 0-3
        Constraint::monotone_increasing(Some(0.0), Some(3.0)),
        // Monotone decreasing in range 3-7
        Constraint::monotone_decreasing(Some(3.0), Some(7.0)),
        // Convex in range 7-10
        Constraint::convex(Some(7.0), Some(10.0)),
    ];

    let regional = ConstrainedSpline::penalized(
        &x_region.view(),
        &y_region.view(),
        region_constraints,
        15,
        3,
        0.01,
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let x_fine_region = Array1::linspace(0.0, 10.0, 200);

    // Compute function values and derivatives
    let mut y_fine_region = Array1::zeros(x_fine_region.len());
    let mut dy_region = Array1::zeros(x_fine_region.len());
    let mut d2y_region = Array1::zeros(x_fine_region.len());

    for (i, &x_val) in x_fine_region.iter().enumerate() {
        y_fine_region[i] = regional.evaluate(x_val).unwrap();
        dy_region[i] = regional.derivative(x_val, 1).unwrap();
        d2y_region[i] = regional.derivative(x_val, 2).unwrap();
    }

    // Verify constraints in each region
    let region1_indices = x_fine_region
        .iter()
        .enumerate()
        .filter(|(_, &x)| x < 3.0)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let region2_indices = x_fine_region
        .iter()
        .enumerate()
        .filter(|(_, &x)| (3.0..7.0).contains(&x))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let region3_indices = x_fine_region
        .iter()
        .enumerate()
        .filter(|(_, &x)| x >= 7.0)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let min_dy_region1 = region1_indices
        .iter()
        .map(|&i| dy_region[i])
        .fold(f64::INFINITY, |a, b| a.min(b));

    let max_dy_region2 = region2_indices
        .iter()
        .map(|&i| dy_region[i])
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let min_d2y_region3 = region3_indices
        .iter()
        .map(|&i| d2y_region[i])
        .fold(f64::INFINITY, |a, b| a.min(b));

    println!(
        "Region 1 (0-3): Min derivative = {} (should be >= 0)",
        min_dy_region1
    );
    println!(
        "Region 2 (3-7): Max derivative = {} (should be <= 0)",
        max_dy_region2
    );
    println!(
        "Region 3 (7-10): Min second derivative = {} (should be >= 0)",
        min_d2y_region3
    );

    println!("\nExample 5: Custom Constraint Combination");
    println!("-------------------------------------");

    // Create data with multiple behaviors
    let x_custom = Array1::linspace(0.0, 1.0, 15);
    let y_custom = x_custom.mapv(|v| f64::powf(v, 3.0) - 0.5 * v + f64::sin(v * 10.0) * 0.05);

    // Impose monotonicity and convexity together
    let constraints = vec![
        Constraint::monotone_increasing(None, None),
        Constraint::convex(None, None),
    ];

    let custom = ConstrainedSpline::penalized(
        &x_custom.view(),
        &y_custom.view(),
        constraints,
        10,   // number of knots
        3,    // degree
        0.01, // smoothing parameter
        BSplineExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let x_fine_custom = Array1::linspace(0.0, 1.0, 100);

    // Compute function values and derivatives
    let mut y_fine_custom = Array1::zeros(x_fine_custom.len());
    let mut dy_custom = Array1::zeros(x_fine_custom.len());
    let mut d2y_custom = Array1::zeros(x_fine_custom.len());

    for (i, &x_val) in x_fine_custom.iter().enumerate() {
        y_fine_custom[i] = custom.evaluate(x_val).unwrap();
        dy_custom[i] = custom.derivative(x_val, 1).unwrap();
        d2y_custom[i] = custom.derivative(x_val, 2).unwrap();
    }

    let min_dy = dy_custom.fold(f64::INFINITY, |a, &b| a.min(b));
    let min_d2y = d2y_custom.fold(f64::INFINITY, |a, &b| a.min(b));

    println!("Minimum first derivative: {} (should be >= 0)", min_dy);
    println!("Minimum second derivative: {} (should be >= 0)", min_d2y);

    println!("\nAll examples completed successfully!");
}
