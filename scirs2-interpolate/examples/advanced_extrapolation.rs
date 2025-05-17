use ndarray::Array1;
use scirs2_interpolate::{
    make_antisymmetric_boundary, make_cubic_extrapolator, make_exponential_extrapolator,
    make_linear_extrapolator, make_linear_gradient_boundary, make_periodic_boundary,
    make_periodic_extrapolator, make_reflection_extrapolator, make_symmetric_boundary,
    make_zero_gradient_boundary, make_zero_value_boundary, BoundaryResult, ExtrapolationMethod,
    Extrapolator,
};

// Helper to run extrapolation for example points and display results
fn demonstrate_extrapolation<F>(name: &str, extrap_values: &[(f64, f64)], f: F)
where
    F: Fn(f64) -> Result<f64, String>,
{
    println!("--- {} Extrapolation ---", name);
    for &(x, expected) in extrap_values {
        match f(x) {
            Ok(value) => println!(
                "  f({:.2}) = {:.6} {}",
                x,
                value,
                if f64::abs(value - expected) < 1e-6 {
                    "(correct)".to_string()
                } else {
                    format!("(expected: {:.6})", expected)
                }
            ),
            Err(e) => println!("  f({:.2}) = Error: {}", x, e),
        }
    }
    println!();
}

fn main() {
    println!("Advanced Extrapolation and Boundary Handling Examples");
    println!("====================================================\n");

    // Example 1: Comparing Different Extrapolation Methods
    println!("Example 1: Comparing Different Extrapolation Methods");
    println!("--------------------------------------------------");

    // Sample domain: [0, 1] with f(x) = x^2
    let lower_bound = 0.0;
    let upper_bound = 1.0;
    let lower_value = 0.0;
    let upper_value = 1.0;
    let lower_derivative = 0.0;
    let upper_derivative = 2.0;

    // Create various extrapolators
    let constant_extrap = Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Constant,
        ExtrapolationMethod::Constant,
    );

    let linear_extrap = make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    );

    let periodic_extrap = make_periodic_extrapolator(lower_bound, upper_bound, None);

    let reflection_extrap = make_reflection_extrapolator(lower_bound, upper_bound);

    // Create functions for demonstrating each extrapolator
    let constant_fn = |x: f64| -> Result<f64, String> {
        constant_extrap.extrapolate(x).map_err(|e| e.to_string())
    };

    let linear_fn =
        |x: f64| -> Result<f64, String> { linear_extrap.extrapolate(x).map_err(|e| e.to_string()) };

    let periodic_fn = |x: f64| -> Result<f64, String> {
        match periodic_extrap.extrapolate(x) {
            Ok(val) => Ok(val),
            Err(e) => {
                // Handle MappedPoint error for periodic extrapolation
                if let scirs2_interpolate::InterpolateError::MappedPoint(mapped_x) = e {
                    // For demonstration, calculate exact value at mapped point
                    // In real usage, you'd interpolate at the mapped point
                    Ok(f64::powi(mapped_x, 2))
                } else {
                    Err(e.to_string())
                }
            }
        }
    };

    let reflection_fn = |x: f64| -> Result<f64, String> {
        match reflection_extrap.extrapolate(x) {
            Ok(val) => Ok(val),
            Err(e) => {
                // Handle MappedPoint error for reflection extrapolation
                if let scirs2_interpolate::InterpolateError::MappedPoint(mapped_x) = e {
                    // Calculate exact value at mapped point
                    Ok(f64::powi(mapped_x, 2))
                } else {
                    Err(e.to_string())
                }
            }
        }
    };

    // Points to evaluate (inside domain, below, and above)
    let inside_domain = vec![(0.25, 0.0625), (0.5, 0.25), (0.75, 0.5625)];
    let below_domain = vec![(-0.5, 0.0), (-1.0, 0.0), (-1.5, 0.0)];
    let above_domain = vec![(1.5, 1.0), (2.0, 1.0), (2.5, 1.0)];

    // Expected values for linear extrapolation
    let linear_below = vec![(-0.5, -0.0), (-1.0, -0.0), (-1.5, -0.0)];
    let linear_above = vec![(1.5, 2.0), (2.0, 3.0), (2.5, 4.0)];

    // Expected values for periodic extrapolation (f(x) = (x mod 1)^2)
    let periodic_below = vec![(-0.5, 0.25), (-1.0, 0.0), (-1.5, 0.25)];
    let periodic_above = vec![(1.5, 0.25), (2.0, 0.0), (2.5, 0.25)];

    // Expected values for reflection extrapolation
    let reflection_below = vec![(-0.5, 0.25), (-1.0, 1.0), (-1.5, 0.25)];
    let reflection_above = vec![(1.5, 0.25), (2.0, 0.0), (2.5, 0.25)];

    // Demonstrate constant extrapolation
    println!("Points inside domain (f(x) = x^2):");
    demonstrate_extrapolation("Inside Domain", &inside_domain, |x| Ok(f64::powi(x, 2)));

    // NOTE: In real usage, for points inside domain you'd use interpolation,
    // not extrapolation. This is just for demonstration.

    println!("Extrapolation below domain (x < 0):");
    demonstrate_extrapolation("Constant", &below_domain, constant_fn);
    demonstrate_extrapolation("Linear", &linear_below, linear_fn);
    demonstrate_extrapolation("Periodic", &periodic_below, periodic_fn);
    demonstrate_extrapolation("Reflection", &reflection_below, reflection_fn);

    println!("Extrapolation above domain (x > 1):");
    demonstrate_extrapolation("Constant", &above_domain, constant_fn);
    demonstrate_extrapolation("Linear", &linear_above, linear_fn);
    demonstrate_extrapolation("Periodic", &periodic_above, periodic_fn);
    demonstrate_extrapolation("Reflection", &reflection_above, reflection_fn);

    // Example 2: Advanced Polynomial Extrapolation
    println!("\nExample 2: Advanced Polynomial Extrapolation");
    println!("------------------------------------------");

    // For a cubic function f(x) = x^3, demonstrate quadratic and cubic extrapolation
    let lower_bound = 0.0;
    let upper_bound = 1.0;
    let lower_value = 0.0;
    let upper_value = 1.0;
    let lower_derivative = 0.0;
    let upper_derivative = 3.0;
    let lower_second_derivative = 0.0;
    let upper_second_derivative = 6.0;

    // Create extrapolators
    let cubic_extrap = make_cubic_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        lower_second_derivative,
        upper_second_derivative,
    );

    let cubic_fn =
        |x: f64| -> Result<f64, String> { cubic_extrap.extrapolate(x).map_err(|e| e.to_string()) };

    // Expected values for cubic extrapolation of f(x) = x^3
    // TODO: The cubic extrapolation is an approximation, so exact matches are not expected
    let cubic_below = vec![(-0.5, -0.125), (-1.0, -1.0), (-1.5, -3.375)];
    let cubic_above = vec![(1.5, 3.375), (2.0, 8.0), (2.5, 15.625)];

    println!("Cubic extrapolation of f(x) = x^3:");
    demonstrate_extrapolation("Below domain", &cubic_below, cubic_fn);
    demonstrate_extrapolation("Above domain", &cubic_above, cubic_fn);

    println!("Exact values of f(x) = x^3 for comparison:");
    demonstrate_extrapolation("Below domain", &cubic_below, |x| Ok(f64::powi(x, 3)));
    demonstrate_extrapolation("Above domain", &cubic_above, |x| Ok(f64::powi(x, 3)));

    // Example 3: Physics-Based Extrapolation
    println!("\nExample 3: Physics-Based Extrapolation");
    println!("---------------------------------");

    // Exponential decay/growth for physical processes
    let lower_bound = 0.0;
    let upper_bound = 1.0;
    let lower_value = 1.0;
    let upper_value = std::f64::consts::E; // e^1
    let lower_derivative = 1.0;
    let upper_derivative = std::f64::consts::E; // e^1

    // Create exponential extrapolator
    let exp_extrap = make_exponential_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        1.0,
        1.0, // rates
    );

    let exp_fn =
        |x: f64| -> Result<f64, String> { exp_extrap.extrapolate(x).map_err(|e| e.to_string()) };

    // Test points for comparing with exact exponential
    let exp_points = vec![
        (-1.0, 0.36787944117144233), // e^-1
        (-2.0, 0.1353352832366127),  // e^-2
        (2.0, 7.3890560989306495),   // e^2
        (3.0, 20.085536923187668),   // e^3
    ];

    println!("Exponential extrapolation compared to f(x) = e^x:");
    demonstrate_extrapolation("Exponential extrapolation", &exp_points, exp_fn);
    demonstrate_extrapolation("Exact exponential", &exp_points, |x| Ok(f64::exp(x)));

    // Example 4: Boundary Handling for PDEs
    println!("\nExample 4: Boundary Handling for PDEs");
    println!("---------------------------------");

    // Domain [0, 10] with various boundary conditions
    let domain_points = Array1::linspace(0.0, 10.0, 11);
    let values = domain_points.mapv(f64::sin);

    // Create various boundary handlers
    let zero_gradient = make_zero_gradient_boundary(0.0, 10.0);
    let zero_value = make_zero_value_boundary(0.0, 10.0);
    let periodic = make_periodic_boundary(0.0, 10.0);
    let symmetric = make_symmetric_boundary(0.0, 10.0);
    let antisymmetric = make_antisymmetric_boundary(0.0, 10.0);
    let linear_gradient = make_linear_gradient_boundary(0.0, 10.0, 0.5, -0.3);

    // Demonstrate boundary handling at specific points
    let test_points = vec![-2.0, -1.0, -0.5, 10.5, 11.0, 12.0];

    println!("Zero Gradient boundaries (Neumann condition):");
    for &x in &test_points {
        match zero_gradient.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::MappedPoint(mapped_x)) => {
                // For zero gradient, we map to the boundary point
                let idx = if mapped_x <= 0.0 { 0 } else { 10 };
                println!(
                    "  f({:.2}) → mapped to boundary: f({:.2}) = {:.6}",
                    x, mapped_x, values[idx]
                );
            }
            Ok(BoundaryResult::InsideDomain(_)) => {
                // This shouldn't happen with our test points
                println!("  f({:.2}) is inside domain", x);
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }
    println!();

    println!("Zero Value boundaries (Dirichlet condition):");
    for &x in &test_points {
        match zero_value.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::DirectValue(value)) => {
                println!("  f({:.2}) = {:.6}", x, value);
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }
    println!();

    println!("Periodic boundaries:");
    for &x in &test_points {
        match periodic.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::MappedPoint(mapped_x)) => {
                // Need to find nearest point in our domain_points
                let nearest_idx = domain_points
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let diff_a = f64::abs(mapped_x - *a);
                        let diff_b = f64::abs(mapped_x - *b);
                        diff_a.partial_cmp(&diff_b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();
                println!(
                    "  f({:.2}) → mapped to: f({:.2}) ≈ {:.6}",
                    x, mapped_x, values[nearest_idx]
                );
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }
    println!();

    println!("Symmetric boundaries (even function reflection):");
    for &x in &test_points {
        match symmetric.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::MappedPoint(mapped_x)) => {
                // Need to find nearest point in our domain_points
                let nearest_idx = domain_points
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let diff_a = f64::abs(mapped_x - *a);
                        let diff_b = f64::abs(mapped_x - *b);
                        diff_a.partial_cmp(&diff_b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();
                println!(
                    "  f({:.2}) → mapped to: f({:.2}) ≈ {:.6}",
                    x, mapped_x, values[nearest_idx]
                );
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }
    println!();

    println!("Antisymmetric boundaries (odd function reflection):");
    for &x in &test_points {
        match antisymmetric.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::DirectValue(value)) => {
                println!("  f({:.2}) = {:.6}", x, value);
            }
            Ok(BoundaryResult::MappedPointWithSignChange(mapped_x)) => {
                println!("  f({:.2}) → mapped to: -f({:.2})", x, mapped_x);
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }
    println!();

    println!("Linear Gradient boundaries:");
    for &x in &test_points {
        match linear_gradient.map_point(x, Some(&values.view()), Some(&domain_points.view())) {
            Ok(BoundaryResult::DirectValue(value)) => {
                println!("  f({:.2}) = {:.6}", x, value);
            }
            _ => println!("  f({:.2}) has unexpected result", x),
        }
    }

    println!("\nAll examples completed successfully!");
}
