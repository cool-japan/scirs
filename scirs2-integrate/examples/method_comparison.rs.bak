use ndarray::{array, ArrayView1};
use scirs2_integrate::{
    gaussian::gauss_legendre,
    monte_carlo::{monte_carlo, MonteCarloOptions},
    ode::{solve_ivp, ODEMethod, ODEOptions},
    quad::{quad, simpson, trapezoid},
    romberg::{self, romberg},
};
use std::marker::PhantomData;
use std::time::Instant;

/// A helper function to time and report the result of an integration method
fn time_integration<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("{}: {:?}", name, elapsed);
    result
}

fn main() {
    // Define a test function with a known analytical result
    // f(x) = x^2, which has the integral ∫x^2 dx = x^3/3
    // Over [0,1], the integral equals 1/3
    let f_1d = |x: f64| x * x;

    // Analytical result
    let exact_result_1d = 1.0 / 3.0;
    println!("Integrating f(x) = x^2 over [0,1]");
    println!("Exact result: {}", exact_result_1d);
    println!("\nPerformance and accuracy comparison:");

    // Test different methods
    let trap_result = time_integration("Trapezoid rule (n=1000)", || {
        trapezoid(f_1d, 0.0, 1.0, 1000)
    });
    println!(
        "  Result: {}, Error: {:.2e}",
        trap_result,
        (trap_result - exact_result_1d).abs()
    );

    let simp_result = time_integration("Simpson's rule (n=1000)", || {
        simpson(f_1d, 0.0, 1.0, 1000).unwrap()
    });
    println!(
        "  Result: {}, Error: {:.2e}",
        simp_result,
        (simp_result - exact_result_1d).abs()
    );

    let gauss_result = time_integration("Gauss-Legendre (n=5)", || {
        gauss_legendre(f_1d, 0.0, 1.0, 5).unwrap()
    });

    println!(
        "  Result: {}, Error: {:.2e}",
        gauss_result,
        (gauss_result - exact_result_1d).abs()
    );

    let romberg_result = time_integration("Romberg", || romberg(f_1d, 0.0, 1.0, None).unwrap());
    println!(
        "  Result: {}, Error: {:.2e}",
        romberg_result.value,
        (romberg_result.value - exact_result_1d).abs()
    );

    // Adaptive quadrature is not available as a public API yet
    // let adaptive_result = time_integration("Adaptive quadrature", || {
    //     adaptive_quad(f_1d, 0.0, 1.0, 1e-10, 1e-10, None).unwrap()
    // });
    // println!("  Result: {}, Error: {:.2e}",
    //          adaptive_result.0, (adaptive_result.0 - exact_result_1d).abs());

    let quad_result = time_integration("General-purpose quad", || {
        quad(f_1d, 0.0, 1.0, None).unwrap()
    });
    println!(
        "  Result: {}, Error: {:.2e}",
        quad_result.value,
        (quad_result.value - exact_result_1d).abs()
    );

    // Monte Carlo integration
    println!("\nMonte Carlo integration with increasing samples:");
    let options_base = MonteCarloOptions {
        seed: Some(42), // For reproducibility
        _phantom: PhantomData,
        ..Default::default()
    };

    for n_samples in [1000, 10_000, 100_000, 1_000_000] {
        let options = MonteCarloOptions {
            n_samples,
            ..options_base.clone()
        };

        let monte_result = time_integration(&format!("Monte Carlo (n={})", n_samples), || {
            monte_carlo(
                |x: ArrayView1<f64>| x[0] * x[0],
                &[(0.0, 1.0)],
                Some(options),
            )
            .unwrap()
        });

        println!(
            "  Result: {}, Error: {:.2e}, Std Error: {:.2e}",
            monte_result.value,
            (monte_result.value - exact_result_1d).abs(),
            monte_result.std_error
        );
    }

    // 2D integration example
    println!("\nMultidimensional integration example:");
    println!("Integrating f(x,y) = x^2 + y^2 over [0,1]×[0,1], exact result = 2/3");

    let f_2d = |x: ArrayView1<f64>| x[0] * x[0] + x[1] * x[1];
    let exact_result_2d = 2.0 / 3.0;

    let mc_2d_result = time_integration("Monte Carlo 2D (n=100,000)", || {
        let options = MonteCarloOptions {
            n_samples: 100_000,
            seed: Some(42),
            _phantom: PhantomData,
            ..Default::default()
        };

        monte_carlo(f_2d, &[(0.0, 1.0), (0.0, 1.0)], Some(options)).unwrap()
    });

    println!(
        "  Result: {}, Error: {:.2e}, Std Error: {:.2e}",
        mc_2d_result.value,
        (mc_2d_result.value - exact_result_2d).abs(),
        mc_2d_result.std_error
    );

    let romberg_2d_result = time_integration("Romberg 2D", || {
        romberg::multi_romberg(f_2d, &[(0.0, 1.0), (0.0, 1.0)], None).unwrap()
    });

    println!(
        "  Result: {}, Error: {:.2e}",
        romberg_2d_result,
        (romberg_2d_result - exact_result_2d).abs()
    );

    // ODE solver comparison
    println!("\nODE solver comparison:");
    println!("Solving y' = -y, y(0) = 1, exact solution: y(t) = exp(-t)");
    println!("Comparing at t = 10, exact = 4.5399e-5");

    let exact_ode_result = (-10.0_f64).exp();
    let decay_system = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Create a map of methods to test
    let methods = [
        ("RK23", ODEMethod::RK23),
        ("RK45", ODEMethod::RK45),
        ("DOP853", ODEMethod::DOP853),
        ("BDF", ODEMethod::Bdf),
        ("Radau (experimental)", ODEMethod::Radau),
        ("LSODA (experimental)", ODEMethod::LSODA),
    ];

    for (method_name, method) in methods {
        let result = time_integration(&format!("{method_name}"), || {
            solve_ivp(
                decay_system,
                [0.0, 10.0],
                array![1.0],
                Some(ODEOptions {
                    method,
                    rtol: 1e-6,
                    atol: 1e-8,
                    max_steps: 1000,
                    ..Default::default()
                }),
            )
        });

        match result {
            Ok(res) => {
                let y_final = res.y.last().unwrap()[0];
                println!(
                    "  Result: {:.6e}, Error: {:.2e}, Steps: {}, Function evals: {}",
                    y_final,
                    (y_final - exact_ode_result).abs(),
                    res.n_steps,
                    res.n_eval
                );
                if method == ODEMethod::LSODA || method == ODEMethod::Radau {
                    if let Some(msg) = res.message {
                        println!("  Message: {}", msg);
                    }
                }
            }
            Err(e) => {
                println!("  Failed: {}", e);
                if method == ODEMethod::LSODA || method == ODEMethod::Radau {
                    println!("  Note: This method is still experimental");
                }
            }
        }
    }
}
