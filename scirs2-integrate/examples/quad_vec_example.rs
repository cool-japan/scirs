use ndarray::arr1;
use scirs2_integrate::quad_vec::{quad_vec, NormType, QuadRule, QuadVecOptions};
use std::f64::consts::PI;

fn main() {
    println!("Vector-valued Integration Examples");
    println!("=================================\n");

    // Example 1: Basic vector-valued integrand
    println!("Example 1: Integrate [x, x^2] from 0 to 1");
    println!("Exact results: [0.5, 0.333...]");

    let f = |x: f64| arr1(&[x, x * x]);
    let result = quad_vec(f, 0.0, 1.0, None).unwrap();

    println!("quad_vec results:");
    println!(
        "  Integral: [{:.10}, {:.10}]",
        result.integral[0], result.integral[1]
    );
    println!(
        "  Error estimates: [{:.10e}, {:.10e}]",
        result.error[0], result.error[1]
    );
    println!("  Function evaluations: {}", result.nfev);
    println!("  Number of subintervals: {}", result.nintervals);
    println!("  Success: {}\n", result.success);

    // Example 2: Trigonometric functions
    println!("Example 2: Integrate [sin(x), cos(x)] from 0 to π");
    println!("Exact results: [2.0, 0.0]");

    let f = |x: f64| arr1(&[x.sin(), x.cos()]);
    let result = quad_vec(f, 0.0, PI, None).unwrap();

    println!("quad_vec results:");
    println!(
        "  Integral: [{:.10}, {:.10}]",
        result.integral[0], result.integral[1]
    );
    println!(
        "  Error estimates: [{:.10e}, {:.10e}]",
        result.error[0], result.error[1]
    );
    println!("  Function evaluations: {}", result.nfev);
    println!("  Number of subintervals: {}", result.nintervals);
    println!("  Success: {}\n", result.success);

    // Example 3: Higher-dimensional vector output
    println!("Example 3: Integrate [sin(x), cos(x), x, x^2, e^(-x)] from 0 to 2");
    println!("Exact results: [1.4161..., -0.9093..., 2.0, 2.6667..., 0.8647...]");

    let f = |x: f64| arr1(&[x.sin(), x.cos(), x, x * x, (-x).exp()]);
    let result = quad_vec(f, 0.0, 2.0, None).unwrap();

    println!("quad_vec results:");
    println!("  Integral:");
    println!("    sin(x): {:.10}", result.integral[0]);
    println!("    cos(x): {:.10}", result.integral[1]);
    println!("    x: {:.10}", result.integral[2]);
    println!("    x^2: {:.10}", result.integral[3]);
    println!("    e^(-x): {:.10}", result.integral[4]);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Success: {}\n", result.success);

    // Example 4: Using different quadrature rules
    println!("Example 4: Integration with different quadrature rules");
    println!("Integrating [sin(x)] from 0 to π with exact result [2.0]");

    let f = |x: f64| arr1(&[x.sin()]);

    // Try different rules
    let rules = [QuadRule::GK21, QuadRule::GK15, QuadRule::Trapezoid];
    let rule_names = ["Gauss-Kronrod 21", "Gauss-Kronrod 15", "Trapezoid"];

    for (i, rule) in rules.iter().enumerate() {
        let options = QuadVecOptions {
            rule: *rule,
            ..Default::default()
        };

        let result = quad_vec(f, 0.0, PI, Some(options)).unwrap();

        println!("  Rule: {}", rule_names[i]);
        println!("    Integral: {:.10}", result.integral[0]);
        println!("    Error: {:.10e}", result.error[0]);
        println!("    Function evaluations: {}", result.nfev);
        println!("    Success: {}\n", result.success);
    }

    // Example 5: Comparison of error norms (L2 vs Max)
    println!("Example 5: Different error norms");
    println!("Integrating [x, 10*x^2, 100*x^3] from 0 to 1");

    let f = |x: f64| arr1(&[x, 10.0 * x * x, 100.0 * x * x * x]);

    // Try different norms
    let norms = [NormType::L2, NormType::Max];
    let norm_names = ["L2 norm", "Max norm"];

    for (i, norm) in norms.iter().enumerate() {
        let options = QuadVecOptions {
            norm: *norm,
            epsrel: 1e-6, // Relatively loose tolerance to show difference
            ..Default::default()
        };

        let result = quad_vec(f, 0.0, 1.0, Some(options)).unwrap();

        println!("  Norm: {}", norm_names[i]);
        println!(
            "    Integral: [{:.6}, {:.6}, {:.6}]",
            result.integral[0], result.integral[1], result.integral[2]
        );
        println!(
            "    Error: [{:.6e}, {:.6e}, {:.6e}]",
            result.error[0], result.error[1], result.error[2]
        );
        println!("    Function evaluations: {}", result.nfev);
        println!("    Subintervals: {}", result.nintervals);
        println!("    Success: {}\n", result.success);
    }

    // Example 6: Specifying breakpoints for discontinuous functions
    println!("Example 6: Function with discontinuity");
    println!("Integrating a step function from 0 to 2 with step at x=1");

    // Step function: [1, 0] for x < 1, [0, 1] for x >= 1
    let f = |x: f64| {
        if x < 1.0 {
            arr1(&[1.0, 0.0])
        } else {
            arr1(&[0.0, 1.0])
        }
    };

    // First try without specifying the breakpoint
    let result_no_breakpoint = quad_vec(f, 0.0, 2.0, None).unwrap();

    println!("  Without breakpoint:");
    println!(
        "    Integral: [{:.10}, {:.10}]",
        result_no_breakpoint.integral[0], result_no_breakpoint.integral[1]
    );
    println!(
        "    Error: [{:.10e}, {:.10e}]",
        result_no_breakpoint.error[0], result_no_breakpoint.error[1]
    );
    println!("    Function evaluations: {}", result_no_breakpoint.nfev);
    println!("    Subintervals: {}", result_no_breakpoint.nintervals);
    println!("    Success: {}\n", result_no_breakpoint.success);

    // Now with the breakpoint specified
    let options = QuadVecOptions {
        points: Some(vec![1.0]),
        ..Default::default()
    };

    let result_with_breakpoint = quad_vec(f, 0.0, 2.0, Some(options)).unwrap();

    println!("  With breakpoint at x=1:");
    println!(
        "    Integral: [{:.10}, {:.10}]",
        result_with_breakpoint.integral[0], result_with_breakpoint.integral[1]
    );
    println!(
        "    Error: [{:.10e}, {:.10e}]",
        result_with_breakpoint.error[0], result_with_breakpoint.error[1]
    );
    println!("    Function evaluations: {}", result_with_breakpoint.nfev);
    println!("    Subintervals: {}", result_with_breakpoint.nintervals);
    println!("    Success: {}", result_with_breakpoint.success);
}
