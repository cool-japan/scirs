//! Performance demonstration comparing different integration methods
//!
//! This example showcases the performance characteristics of various integration
//! methods and provides a simple timing framework for basic comparisons.

use ndarray::Array1;
use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use scirs2_integrate::quad::{quad, QuadOptions};
use std::time::Instant;

/// Simple timing utility
fn time_function<F, R>(f: F, name: &str) -> (f64, R)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f64();
    println!("{}: {:.6} seconds", name, time_secs);
    (time_secs, result)
}

/// Average timing over multiple runs
#[allow(dead_code)]
fn time_function_averaged<F, R>(mut f: F, name: &str, runs: usize) -> (f64, R)
where
    F: FnMut() -> R,
{
    let mut total_time = 0.0;
    let mut result = None;

    for i in 0..runs {
        let start = Instant::now();
        let res = f();
        let duration = start.elapsed();
        total_time += duration.as_secs_f64();

        if i == 0 {
            result = Some(res);
        }
    }

    let avg_time = total_time / runs as f64;
    println!("{} (avg of {} runs): {:.6} seconds", name, runs, avg_time);
    (avg_time, result.unwrap())
}

// Test problems
fn exponential_decay(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
    Array1::from_vec(vec![-y[0]])
}

fn harmonic_oscillator(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
    Array1::from_vec(vec![y[1], -y[0]])
}

fn van_der_pol_stiff(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
    let mu = 100.0; // Stiff parameter
    Array1::from_vec(vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
}

fn polynomial_cubic(x: f64) -> f64 {
    x * x * x
}

fn oscillatory_integrand(x: f64) -> f64 {
    (10.0 * x).sin()
}

fn gaussian_2d(x: ndarray::ArrayView1<f64>) -> f64 {
    let r2 = x[0] * x[0] + x[1] * x[1];
    (-r2).exp()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("scirs2-integrate Performance Demonstration");
    println!("==========================================");
    println!();

    // ODE Solver Performance
    println!("ODE Solver Performance:");
    println!("-----------------------");

    // Simple exponential decay - different methods
    let y0_simple = Array1::from_vec(vec![1.0]);
    let t_span_simple = [0.0, 5.0];

    let methods = vec![
        (ODEMethod::RK45, "RK45"),
        (ODEMethod::DOP853, "DOP853"),
        (ODEMethod::Bdf, "BDF"),
        (ODEMethod::LSODA, "LSODA"),
    ];

    for (method, name) in methods {
        let opts = ODEOptions {
            method,
            rtol: 1e-6,
            atol: 1e-9,
            ..Default::default()
        };

        let (_time, result) = time_function(
            || {
                solve_ivp(
                    exponential_decay,
                    t_span_simple,
                    y0_simple.clone(),
                    Some(opts),
                )
            },
            &format!("Exponential decay ({})", name),
        );

        if let Ok(sol) = result {
            let final_value = sol.y.last().unwrap()[0];
            let exact = (-5.0_f64).exp();
            let error = (final_value - exact).abs();
            println!("  Final value: {:.6}, Error: {:.2e}", final_value, error);
        }
    }

    println!();

    // Harmonic oscillator - energy conservation
    println!("Harmonic Oscillator (energy conservation test):");
    let y0_harm = Array1::from_vec(vec![1.0, 0.0]);
    let t_span_harm = [0.0, 20.0]; // Long integration

    for (method, name) in &[(ODEMethod::RK45, "RK45"), (ODEMethod::DOP853, "DOP853")] {
        let opts = ODEOptions {
            method: *method,
            rtol: 1e-8,
            atol: 1e-11,
            ..Default::default()
        };

        let (_time, result) = time_function(
            || {
                solve_ivp(
                    harmonic_oscillator,
                    t_span_harm,
                    y0_harm.clone(),
                    Some(opts),
                )
            },
            &format!("Harmonic oscillator ({})", name),
        );

        if let Ok(sol) = result {
            let final_y = sol.y.last().unwrap();
            let energy = 0.5 * (final_y[0] * final_y[0] + final_y[1] * final_y[1]);
            let energy_error = (energy - 0.5).abs();
            println!("  Energy conservation error: {:.2e}", energy_error);
        }
    }

    println!();

    // Stiff problem
    println!("Stiff Van der Pol Oscillator:");
    let y0_vdp = Array1::from_vec(vec![2.0, 0.0]);
    let t_span_vdp = [0.0, 30.0];

    for (method, name) in &[(ODEMethod::Bdf, "BDF"), (ODEMethod::LSODA, "LSODA")] {
        let opts = ODEOptions {
            method: *method,
            rtol: 1e-4,
            atol: 1e-7,
            ..Default::default()
        };

        let (_time, result) = time_function(
            || solve_ivp(van_der_pol_stiff, t_span_vdp, y0_vdp.clone(), Some(opts)),
            &format!("Van der Pol stiff ({})", name),
        );

        if let Ok(sol) = result {
            println!(
                "  Final steps: {}, Function evaluations: {}",
                sol.n_steps, sol.n_eval
            );
        }
    }

    println!();

    // Quadrature Performance
    println!("Quadrature Performance:");
    println!("-----------------------");

    // Polynomial integration (exact answer known)
    let (_time, result) = time_function(
        || {
            let opts = QuadOptions {
                abs_tol: 1e-12,
                rel_tol: 1e-12,
                max_evals: 1000,
                ..Default::default()
            };
            quad(polynomial_cubic, 0.0, 1.0, Some(opts))
        },
        "Polynomial x³ integration",
    );

    if let Ok(quad_result) = result {
        let exact = 0.25;
        let actual_error = (quad_result.value - exact).abs();
        println!(
            "  Result: {:.12}, Estimated error: {:.2e}, Actual error: {:.2e}",
            quad_result.value, quad_result.abs_error, actual_error
        );
    }

    // Oscillatory function
    let (_time, result) = time_function(
        || {
            let opts = QuadOptions {
                abs_tol: 1e-10,
                rel_tol: 1e-10,
                max_evals: 2000,
                ..Default::default()
            };
            quad(oscillatory_integrand, 0.0, 1.0, Some(opts))
        },
        "Oscillatory sin(10x) integration",
    );

    if let Ok(quad_result) = result {
        println!(
            "  Result: {:.10}, Estimated error: {:.2e}",
            quad_result.value, quad_result.abs_error
        );
    }

    println!();

    // Monte Carlo Integration
    println!("Monte Carlo Integration:");
    println!("------------------------");

    let sample_counts = vec![1_000, 10_000, 100_000];

    for &n_samples in &sample_counts {
        let (_time, result) = time_function(
            || {
                let ranges = vec![(-2.0, 2.0), (-2.0, 2.0)];
                let opts = MonteCarloOptions {
                    n_samples,
                    ..Default::default()
                };
                monte_carlo(gaussian_2d, &ranges, Some(opts))
            },
            &format!("2D Gaussian MC ({} samples)", n_samples),
        );

        if let Ok(mc_result) = result {
            let exact = std::f64::consts::PI; // ∫∫ exp(-x²-y²) dx dy over [-∞,∞]² = π
            let domain_exact = exact * (1.0 - (-4.0_f64).exp()).powi(2); // Scaled for [-2,2]²
            let error = (mc_result.value - domain_exact).abs();
            println!(
                "  Result: {:.6}, Error: {:.2e}, Std: {:.2e}",
                mc_result.value, error, mc_result.std_error
            );
        }
    }

    println!();

    // Parallel Monte Carlo Performance (if available)
    #[cfg(feature = "parallel")]
    {
        println!("Parallel Monte Carlo Performance:");
        println!("---------------------------------");

        use scirs2_integrate::monte_carlo_parallel::{
            parallel_monte_carlo, ParallelMonteCarloOptions,
        };

        let n_samples = 1_000_000;
        let ranges = vec![(-2.0, 2.0), (-2.0, 2.0)];

        // Sequential
        let (seq_time, _seq_result) = time_function(
            || {
                let opts = MonteCarloOptions {
                    n_samples,
                    ..Default::default()
                };
                monte_carlo(gaussian_2d, &ranges, Some(opts))
            },
            "Sequential MC (1M samples)",
        );

        // Parallel
        let (par_time, _par_result) = time_function(
            || {
                let opts = ParallelMonteCarloOptions {
                    n_samples,
                    batch_size: n_samples / 8,
                    ..Default::default()
                };
                parallel_monte_carlo(gaussian_2d, &ranges, Some(opts))
            },
            "Parallel MC (1M samples)",
        );

        if seq_time > 0.0 && par_time > 0.0 {
            let speedup = seq_time / par_time;
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    println!();

    // Memory usage demonstration
    println!("Large System Performance:");
    println!("-------------------------");

    let system_sizes = vec![50, 100, 200];

    for &n in &system_sizes {
        // Create a large linear ODE system: dy/dt = A*y
        let (_time, result) = time_function(
            || {
                use ndarray::Array2;

                let a_matrix = Array2::from_shape_fn((n, n), |(i, j)| {
                    if i == j {
                        -1.0 // diagonal
                    } else if i.abs_diff(j) == 1 {
                        0.1 // off-diagonal
                    } else {
                        0.0
                    }
                });

                let linear_system = move |_t: f64, y: ndarray::ArrayView1<f64>| {
                    let y_owned = y.to_owned();
                    a_matrix.dot(&y_owned)
                };

                let y0 = Array1::ones(n);
                let opts = ODEOptions {
                    method: ODEMethod::Bdf, // Good for large systems
                    rtol: 1e-3,
                    atol: 1e-6,
                    ..Default::default()
                };

                solve_ivp(linear_system, [0.0, 1.0], y0, Some(opts))
            },
            &format!("Large linear system ({}×{})", n, n),
        );

        if let Ok(sol) = result {
            println!("  Steps: {}, Function evals: {}", sol.n_steps, sol.n_eval);
        }
    }

    println!();
    println!("Performance demonstration completed!");
    println!("For more detailed benchmarks, run:");
    println!("  cargo bench --bench scipy_comparison");
    println!("  python benches/scipy_reference.py");

    Ok(())
}
