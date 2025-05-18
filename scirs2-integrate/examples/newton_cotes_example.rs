use scirs2_integrate::newton_cotes::{newton_cotes, newton_cotes_integrate, NewtonCotesType};
use std::f64::consts::PI;

fn main() {
    println!("Newton-Cotes Quadrature Examples");
    println!("================================\n");

    // Example 1: Basic rules - show points and weights
    println!("Example 1: Common Newton-Cotes Rules\n");

    let closed_rules = [
        ("Midpoint (n=1)", 1),
        ("Trapezoidal Rule (n=2)", 2),
        ("Simpson's Rule (n=3)", 3),
        ("Simpson's 3/8 Rule (n=4)", 4),
        ("Boole's Rule (n=5)", 5),
    ];

    println!("Closed Newton-Cotes Formulas:");
    for (name, n) in closed_rules {
        println!("\n{}", name);
        match newton_cotes::<f64>(n, NewtonCotesType::Closed, None, None) {
            Ok(result) => {
                println!("  Points: {:?}", result.points);
                println!("  Weights: {:?}", result.weights);
                println!("  Degree of exactness: {}", result.degree);
                println!("  Error coefficient: {:.10e}", result.error_coefficient);
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    println!("\nOpen Newton-Cotes Formulas:");
    for n in 3..6 {
        println!("\n{}-point Open Formula", n);
        match newton_cotes::<f64>(n, NewtonCotesType::Open, None, None) {
            Ok(result) => {
                println!("  Points: {:?}", result.points);
                println!("  Weights: {:?}", result.weights);
                println!("  Degree of exactness: {}", result.degree);
                println!("  Error coefficient: {:.10e}", result.error_coefficient);
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    // Example 2: Integration examples with different functions
    println!("\n\nExample 2: Integrating Functions with Newton-Cotes");

    // Test functions - use type erasure with Box to avoid type mismatch
    let test_cases: Vec<(&str, Box<dyn Fn(f64) -> f64>, f64, f64, f64)> = vec![
        ("x²", Box::new(|x: f64| x * x), 0.0, 1.0, 1.0 / 3.0),
        ("sin(x)", Box::new(|x: f64| x.sin()), 0.0, PI, 2.0),
        (
            "1/√(1-x²)",
            Box::new(|x: f64| 1.0 / (1.0 - x * x).sqrt()),
            -0.9,
            0.9,
            2.0 * 0.9_f64.asin(),
        ),
        (
            "exp(-x²)",
            Box::new(|x: f64| (-x * x).exp()),
            -1.0,
            1.0,
            2.0 * (1.0 - (-1.0_f64).exp()) / (2.0_f64.sqrt()),
        ),
    ];

    for (name, f, a, b, exact) in test_cases {
        println!("\nIntegrating f(x) = {} from {} to {}", name, a, b);
        println!("Exact value: {:.10}", exact);

        // Try different Newton-Cotes rules with increasing numbers of points
        println!("Closed Newton-Cotes:");
        for n in [2, 3, 5, 7, 9] {
            match newton_cotes_integrate(&f, a, b, n, NewtonCotesType::Closed) {
                Ok((result, error)) => {
                    let actual_error = (result - exact).abs();
                    println!(
                        "  n={}: {:.10} (estimated error: {:.10e}, actual error: {:.10e})",
                        n, result, error, actual_error
                    );
                }
                Err(e) => {
                    println!("  n={}: Error: {}", n, e);
                }
            }
        }

        println!("Open Newton-Cotes:");
        for n in [3, 4, 5, 6, 7] {
            match newton_cotes_integrate(&f, a, b, n, NewtonCotesType::Open) {
                Ok((result, error)) => {
                    let actual_error = (result - exact).abs();
                    println!(
                        "  n={}: {:.10} (estimated error: {:.10e}, actual error: {:.10e})",
                        n, result, error, actual_error
                    );
                }
                Err(e) => {
                    println!("  n={}: Error: {}", n, e);
                }
            }
        }
    }

    // Example 3: Integration with composite Newton-Cotes rules
    println!("\n\nExample 3: Composite Newton-Cotes Integration");

    // Define a function to integrate using composite rules
    let f = |x: f64| 1.0 / (1.0 + x * x); // f(x) = 1/(1+x²), with integral = arctan(x)
    let a: f64 = 0.0;
    let b: f64 = 1.0;
    let exact = b.atan() - a.atan();

    println!("Integrating f(x) = 1/(1+x²) from {} to {}", a, b);
    println!("Exact value: {:.10}", exact);

    // Implement composite rules with different numbers of panels
    let panels = [2, 4, 8, 16, 32];

    for panels_count in panels {
        let panel_width = (b - a) / panels_count as f64;

        // Composite trapezoidal rule
        let mut trap_sum = 0.0;
        for i in 0..panels_count {
            let panel_a = a + i as f64 * panel_width;
            let panel_b = panel_a + panel_width;
            let (panel_result, _) =
                newton_cotes_integrate(&f, panel_a, panel_b, 2, NewtonCotesType::Closed).unwrap();
            trap_sum += panel_result;
        }

        // Composite Simpson's rule (requires even number of panels)
        let mut simp_sum = 0.0;
        if panels_count % 2 == 0 {
            for i in 0..panels_count / 2 {
                let panel_a = a + 2.0 * i as f64 * panel_width;
                let panel_b = panel_a + 2.0 * panel_width;
                let (panel_result, _) =
                    newton_cotes_integrate(&f, panel_a, panel_b, 3, NewtonCotesType::Closed)
                        .unwrap();
                simp_sum += panel_result;
            }
        }

        let trap_error = (trap_sum - exact).abs();
        println!(
            "\nComposite Trapezoidal Rule with {} panels: {:.10} (error: {:.10e})",
            panels_count, trap_sum, trap_error
        );

        if panels_count % 2 == 0 {
            let simp_error = (simp_sum - exact).abs();
            println!(
                "Composite Simpson's Rule with {} panels: {:.10} (error: {:.10e})",
                panels_count, simp_sum, simp_error
            );
        }
    }

    // Example 4: Higher-order formulas and limitations
    println!("\n\nExample 4: Higher-Order Newton-Cotes Formulas and Limitations");

    println!("Newton-Cotes formulas can have issues for higher orders due to Runge's phenomenon.");
    println!("Here are higher-order closed formula weights for [0,1]:");

    for n in [5, 7, 9, 11, 13, 15] {
        println!("\nn = {} points:", n);
        match newton_cotes::<f64>(n, NewtonCotesType::Closed, None, None) {
            Ok(result) => {
                println!("  Weights sum: {:.10}", result.weights.iter().sum::<f64>());

                // Check if there are negative weights (which can lead to instability)
                let neg_weights = result.weights.iter().filter(|&&w| w < 0.0).count();
                if neg_weights > 0 {
                    println!("  Warning: Formula has {} negative weights", neg_weights);
                }

                // Test with polynomial that should be integrated exactly
                let exact_degree = result.degree;
                let poly = |x: f64| x.powi(exact_degree as i32);
                let exact_integral = 1.0 / (exact_degree as f64 + 1.0);

                let (poly_result, _) =
                    newton_cotes_integrate(&poly, 0.0, 1.0, n, NewtonCotesType::Closed).unwrap();
                let poly_error = (poly_result - exact_integral).abs();

                println!("  Integration of x^{} (should be exact):", exact_degree);
                println!(
                    "    Result: {:.10}, Exact: {:.10}, Error: {:.10e}",
                    poly_result, exact_integral, poly_error
                );
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    println!("\nConclusion:");
    println!("1. Lower-order methods (n ≤ 7) are generally more stable");
    println!("2. Composite rules (many panels with low-order methods) are typically preferred");
    println!(
        "3. For high accuracy, adaptive quadrature or Gaussian quadrature may be better choices"
    );
}
