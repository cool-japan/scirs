use ndarray::{Array1, Axis};
use scirs2_interpolate::constrained::{
    ConstrainedSpline, Constraint, ConstraintRegion, ConstraintType, FittingMethod,
};
use scirs2_interpolate::ExtrapolateMode;

fn main() {
    println!("Constrained Splines Examples");
    println!("===========================\n");

    // Create some test data that's not monotonic or convex
    let x = Array1::linspace(0.0, 10.0, 15);
    let y = x.mapv(|v| {
        (v - 5.0).powi(2) * 0.1 + // Parabola
        f64::sin(v * 0.8) * 2.0 + // Add sine wave
        (v * 0.2) // Add linear trend
    });

    println!("Example 1: Monotone Increasing Spline");
    println!("-------------------------------------");
    let monotone_inc = ConstrainedSpline::monotone_increasing_spline(
        &x.view(),
        &y.view(),
        FittingMethod::Penalized,
        8,
        3,
        0.1,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Evaluate on a finer grid
    let x_fine = Array1::linspace(0.0, 10.0, 100);
    let y_fine = monotone_inc.evaluate(&x_fine.view()).unwrap();

    // Compute first derivatives to verify monotonicity
    let dy_fine = monotone_inc.derivative(1, &x_fine.view()).unwrap();

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
    let convex = ConstrainedSpline::convex_spline(
        &x.view(),
        &y.view(),
        FittingMethod::LeastSquares,
        10,
        3,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let y_convex = convex.evaluate(&x_fine.view()).unwrap();

    // Compute second derivatives to verify convexity
    let d2y_convex = convex.derivative(2, &x_fine.view()).unwrap();

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
        Constraint::new(ConstraintType::MonotoneIncreasing, ConstraintRegion::Full),
        Constraint::new(ConstraintType::UpperBound(15.0), ConstraintRegion::Full),
        Constraint::new(ConstraintType::LowerBound(0.0), ConstraintRegion::Full),
    ];

    let multi_constraint = ConstrainedSpline::interpolate(
        &x.view(),
        &y.view(),
        constraints,
        3,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let y_multi = multi_constraint.evaluate(&x_fine.view()).unwrap();
    let dy_multi = multi_constraint.derivative(1, &x_fine.view()).unwrap();

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
            (v - 7.0).powi(2) * 0.3 + 2.0
        }
    });

    // Different constraints in different regions
    let region_constraints = vec![
        Constraint::new(
            ConstraintType::MonotoneIncreasing,
            ConstraintRegion::Range(0.0, 3.0),
        ),
        Constraint::new(
            ConstraintType::MonotoneDecreasing,
            ConstraintRegion::Range(3.0, 7.0),
        ),
        Constraint::new(ConstraintType::Convex, ConstraintRegion::Range(7.0, 10.0)),
    ];

    let regional = ConstrainedSpline::penalized(
        &x_region.view(),
        &y_region.view(),
        region_constraints,
        15,
        3,
        0.01,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let x_fine_region = Array1::linspace(0.0, 10.0, 200);
    let y_fine_region = regional.evaluate(&x_fine_region.view()).unwrap();
    let dy_region = regional.derivative(1, &x_fine_region.view()).unwrap();
    let d2y_region = regional.derivative(2, &x_fine_region.view()).unwrap();

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
        .filter(|(_, &x)| x >= 3.0 && x < 7.0)
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
    let y_custom = x_custom.mapv(|v| v.powf(3.0) - 0.5 * v + f64::sin(v * 10.0) * 0.05);

    // Impose monotonicity and convexity together
    let custom = ConstrainedSpline::monotone_convex_spline(
        &x_custom.view(),
        &y_custom.view(),
        FittingMethod::Penalized,
        10,
        3,
        0.01,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let x_fine_custom = Array1::linspace(0.0, 1.0, 100);
    let y_fine_custom = custom.evaluate(&x_fine_custom.view()).unwrap();
    let dy_custom = custom.derivative(1, &x_fine_custom.view()).unwrap();
    let d2y_custom = custom.derivative(2, &x_fine_custom.view()).unwrap();

    let min_dy = dy_custom.fold(f64::INFINITY, |a, &b| a.min(b));
    let min_d2y = d2y_custom.fold(f64::INFINITY, |a, &b| a.min(b));

    println!("Minimum first derivative: {} (should be >= 0)", min_dy);
    println!("Minimum second derivative: {} (should be >= 0)", min_d2y);

    println!("\nAll examples completed successfully!");
}
