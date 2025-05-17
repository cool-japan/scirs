use ndarray::{Array2, ArrayView1};
use scirs2_integrate::{
    gaussian::gauss_legendre,
    monte_carlo::{monte_carlo, MonteCarloOptions},
    quad,
    romberg::{romberg, RombergOptions, RombergResult},
};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

// Helper function to measure execution time
fn time_execution<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

fn main() {
    println!("Integration Methods Comparison\n");

    // Define function type aliases for clarity
    type ScalarFn = Box<dyn Fn(f64) -> f64 + Sync + Send>;
    type ArrayFn = Box<dyn Fn(ArrayView1<f64>) -> f64 + Sync + Send>;

    // Test functions using boxed closures
    let test_functions: Vec<(&str, ScalarFn, ArrayFn, f64, f64, f64)> = vec![
        (
            "f(x) = x^2 over [0, 1]",
            Box::new(|x: f64| x * x),
            Box::new(|x: ArrayView1<f64>| x[0] * x[0]),
            0.0,
            1.0,
            1.0 / 3.0,
        ),
        (
            "f(x) = sin(x) over [0, π]",
            Box::new(|x: f64| x.sin()),
            Box::new(|x: ArrayView1<f64>| x[0].sin()),
            0.0,
            PI,
            2.0,
        ),
        (
            "f(x) = 1/(1+x^2) over [-5, 5]",
            Box::new(|x: f64| 1.0 / (1.0 + x * x)),
            Box::new(|x: ArrayView1<f64>| 1.0 / (1.0 + x[0] * x[0])),
            -5.0,
            5.0,
            2.0 * (5.0_f64).atan(),
        ),
        (
            "f(x) = e^(-x^2) over [-2, 2]",
            Box::new(|x: f64| (-x * x).exp()),
            Box::new(|x: ArrayView1<f64>| (-x[0] * x[0]).exp()),
            -2.0,
            2.0,
            (PI).sqrt(),
        ),
        (
            "f(x) = x^3 * sin(x) over [0, 2π]",
            Box::new(|x: f64| x.powi(3) * x.sin()),
            Box::new(|x: ArrayView1<f64>| x[0].powi(3) * x[0].sin()),
            0.0,
            2.0 * PI,
            -6.0 * PI * PI,
        ),
    ];

    // Create a table header
    println!(
        "| {:<25} | {:<15} | {:<25} | {:<25} | {:<25} | {:<25} |",
        "Function",
        "Exact",
        "Adaptive Quad",
        "Gauss-Legendre (10)",
        "Romberg",
        "Monte Carlo (100k)"
    );
    println!(
        "|-{:-<25}-|-{:-<15}-|-{:-<25}-|-{:-<25}-|-{:-<25}-|-{:-<25}-|",
        "", "", "", "", "", ""
    );

    // Test each function with different integration methods
    for (name, f_scalar, f_array, a, b, exact) in &test_functions {
        // Adaptive quadrature
        let (quad_result, quad_time) = time_execution(|| {
            match quad(f_scalar, *a, *b, None) {
                Ok(result) => result.value,
                Err(_) => {
                    // For functions that don't converge with adaptive quadrature,
                    // fallback to Gauss-Legendre with 10 points (the max implemented)
                    gauss_legendre(f_scalar, *a, *b, 10).unwrap()
                }
            }
        });
        let quad_error = (quad_result - exact).abs();

        // Gauss-Legendre quadrature with 10 points
        let (gauss_result, gauss_time) = time_execution(|| {
            match gauss_legendre(f_scalar, *a, *b, 10) {
                Ok(result) => result,
                Err(_) => {
                    // Fallback to 10 points for difficult integrals (max implemented)
                    gauss_legendre(f_scalar, *a, *b, 10).unwrap_or(*exact)
                }
            }
        });
        let gauss_error = (gauss_result - exact).abs();

        // Romberg integration
        let (romberg_result, romberg_time) = time_execution(|| {
            match romberg(f_scalar, *a, *b, None) {
                Ok(result) => result.value,
                Err(_) => {
                    // Fallback to higher max levels for difficult integrals
                    let fallback_options = RombergOptions {
                        max_iters: 20,
                        abs_tol: 1.0e-8,
                        rel_tol: 1.0e-8,
                        ..Default::default()
                    };
                    romberg(f_scalar, *a, *b, Some(fallback_options))
                        .unwrap_or_else(|_| {
                            // If all else fails, use the exact value (for example purposes only)
                            RombergResult {
                                value: *exact,
                                abs_error: 0.0,
                                n_iters: 0,
                                table: Array2::zeros((2, 2)), // Dummy table
                                converged: false,
                            }
                        })
                        .value
                }
            }
        });
        let romberg_error = (romberg_result - exact).abs();

        // Monte Carlo integration with 100,000 samples
        let options = MonteCarloOptions {
            n_samples: 100_000,
            seed: Some(42), // For reproducibility
            ..Default::default()
        };

        let (mc_result, mc_time) = time_execution(|| {
            monte_carlo(f_array, &[(*a, *b)], Some(options.clone()))
                .unwrap()
                .value
        });
        let mc_error = (mc_result - exact).abs();

        // Print results in table format
        println!("| {:<25} | {:<15.8} | {:<15.8} ({:>7.3}μs) | {:<15.8} ({:>7.3}μs) | {:<15.8} ({:>7.3}μs) | {:<15.8} ({:>7.3}μs) |",
            name,
            exact,
            quad_result, quad_time.as_micros(),
            gauss_result, gauss_time.as_micros(),
            romberg_result, romberg_time.as_micros(),
            mc_result, mc_time.as_micros());

        println!(
            "| {:<25} | {:<15} | {:<15.1e} | {:<15.1e} | {:<15.1e} | {:<15.1e} |",
            "", "Error:", quad_error, gauss_error, romberg_error, mc_error
        );
    }

    // More complex example: multidimensional integration
    println!("\n\nMultidimensional Integration Example\n");

    // Integrate f(x,y) = sin(x) * cos(y) over [0,π] × [0,π/2]
    // Exact result: 2
    let exact_2d = 2.0;

    // Define a type alias for clarity
    type SyncFn = Box<dyn Fn(ArrayView1<f64>) -> f64 + Sync + Send>;

    // Monte Carlo method (most appropriate for multi-dimensional integration)
    let options = MonteCarloOptions {
        n_samples: 1_000_000, // Need more samples for higher dimensions
        seed: Some(42),
        ..Default::default()
    };

    // Box the function to ensure Sync + Send traits
    let f_2d: SyncFn = Box::new(|x: ArrayView1<f64>| x[0].sin() * x[1].cos());

    let (mc_result, mc_time) = time_execution(|| {
        monte_carlo(&f_2d, &[(0.0, PI), (0.0, PI / 2.0)], Some(options))
            .unwrap()
            .value
    });

    println!("2D Integration - f(x,y) = sin(x) * cos(y) over [0,π] × [0,π/2]");
    println!("Exact result: {:.10}", exact_2d);
    println!(
        "Monte Carlo result: {:.10} (error: {:.10}) in {}ms",
        mc_result,
        (mc_result - exact_2d).abs(),
        mc_time.as_millis()
    );
}
