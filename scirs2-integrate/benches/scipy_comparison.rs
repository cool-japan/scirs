//! Comprehensive performance benchmarks comparing scirs2-integrate with SciPy
//!
//! This benchmark suite provides direct timing comparisons between our Rust
//! implementations and SciPy's integrate module for various problem types.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use scirs2_integrate::cubature::{cubature, CubatureOptions};
use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use scirs2_integrate::quad::{quad, QuadOptions};

/// Test problems for ODE benchmarking
mod ode_problems {
    use ndarray::Array1;

    /// Simple exponential decay: dy/dt = -y, y(0) = 1
    pub fn exponential_decay(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![-y[0]])
    }

    /// Harmonic oscillator: d²x/dt² + x = 0, converted to first order system
    pub fn harmonic_oscillator(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![y[1], -y[0]])
    }

    /// Van der Pol oscillator (stiff for large mu): d²x/dt² - mu*(1-x²)*dx/dt + x = 0
    #[allow(dead_code)]
    pub fn van_der_pol(mu: f64) -> impl Fn(f64, ndarray::ArrayView1<f64>) -> Array1<f64> + 'static {
        move |_t: f64, y: ndarray::ArrayView1<f64>| {
            Array1::from_vec(vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
        }
    }

    /// Lotka-Volterra predator-prey model
    #[allow(dead_code)]
    pub fn lotka_volterra(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    ) -> impl Fn(f64, ndarray::ArrayView1<f64>) -> Array1<f64> + 'static {
        move |_t: f64, y: ndarray::ArrayView1<f64>| {
            let x = y[0]; // prey
            let y_val = y[1]; // predator
            Array1::from_vec(vec![
                a * x - b * x * y_val,      // dx/dt
                -c * y_val + d * x * y_val, // dy/dt
            ])
        }
    }

    /// N-body problem (simplified 3-body)
    pub fn three_body_problem(_t: f64, y: ndarray::ArrayView1<f64>) -> Array1<f64> {
        let mut dydt = Array1::zeros(y.len());

        // Positions: x1, y1, x2, y2, x3, y3
        // Velocities: vx1, vy1, vx2, vy2, vx3, vy3
        for i in 0..3 {
            dydt[i * 2] = y[6 + i * 2]; // dx_i/dt = vx_i
            dydt[i * 2 + 1] = y[6 + i * 2 + 1]; // dy_i/dt = vy_i
        }

        // Gravitational forces
        let m = [1.0, 1.0, 1.0]; // masses
        for i in 0..3 {
            let mut fx = 0.0;
            let mut fy = 0.0;

            for j in 0..3 {
                if i != j {
                    let dx = y[j * 2] - y[i * 2];
                    let dy = y[j * 2 + 1] - y[i * 2 + 1];
                    let r = (dx * dx + dy * dy).sqrt();
                    let r3 = r * r * r + 1e-10; // softening

                    fx += m[j] * dx / r3;
                    fy += m[j] * dy / r3;
                }
            }

            dydt[6 + i * 2] = fx; // dvx_i/dt
            dydt[6 + i * 2 + 1] = fy; // dvy_i/dt
        }

        dydt
    }
}

/// Integration test problems for quadrature benchmarking
mod quadrature_problems {

    /// Simple polynomial: f(x) = x^3
    pub fn polynomial_cubic(x: f64) -> f64 {
        x * x * x
    }

    /// Oscillatory function: f(x) = sin(10*x)
    pub fn oscillatory(x: f64) -> f64 {
        (10.0 * x).sin()
    }

    /// Exponential function: f(x) = exp(-x^2)
    pub fn gaussian(x: f64) -> f64 {
        (-x * x).exp()
    }

    /// Nearly singular function: f(x) = 1/sqrt(x)
    pub fn nearly_singular(x: f64) -> f64 {
        if x > 1e-10 {
            1.0 / x.sqrt()
        } else {
            1.0 / (1e-10_f64).sqrt()
        }
    }

    /// Multi-dimensional test function: f(x,y) = exp(-(x^2 + y^2))
    pub fn multivariate_gaussian(x: &[f64]) -> f64 {
        let r2: f64 = x.iter().map(|&xi| xi * xi).sum();
        (-r2).exp()
    }

    /// High-dimensional oscillatory function
    #[allow(dead_code)]
    pub fn high_dim_oscillatory(x: &[f64]) -> f64 {
        let sum: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi)
            .sum();
        sum.sin()
    }
}

/// Benchmark ODE solvers
fn bench_ode_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("ODE Solvers");

    // Test different problem sizes and types
    let problems = vec![
        ("exponential_decay", 1, [0.0, 1.0], vec![1.0]),
        ("harmonic_oscillator", 2, [0.0, 10.0], vec![1.0, 0.0]),
        ("van_der_pol_mild", 2, [0.0, 10.0], vec![1.0, 0.0]),
        ("van_der_pol_stiff", 2, [0.0, 10.0], vec![1.0, 0.0]),
        ("lotka_volterra", 2, [0.0, 15.0], vec![10.0, 5.0]),
        (
            "three_body",
            12,
            [0.0, 5.0],
            vec![
                1.0, 0.0, -0.5, 0.866, -0.5, -0.866, // positions
                0.0, 0.5, -0.433, -0.25, 0.433, -0.25, // velocities
            ],
        ),
    ];

    let methods = vec![
        ODEMethod::RK45,
        ODEMethod::DOP853,
        ODEMethod::Bdf,
        ODEMethod::Radau,
        ODEMethod::LSODA,
    ];

    for (prob_name, dim, t_span, y0) in problems {
        for method in &methods {
            let parameter = format!("{}_{:?}_dim{}", prob_name, method, dim);

            group.bench_with_input(
                BenchmarkId::new("solve_ivp", &parameter),
                &(prob_name, method, t_span, y0.clone()),
                |b, &(prob_name, method, t_span, ref y0)| {
                    b.iter(|| {
                        let y0_array = Array1::from_vec(y0.clone());
                        let opts = ODEOptions {
                            rtol: 1e-6,
                            atol: 1e-9,
                            method: *method,
                            ..Default::default()
                        };

                        let result = match prob_name {
                            "exponential_decay" => solve_ivp(
                                ode_problems::exponential_decay,
                                t_span,
                                y0_array,
                                Some(opts),
                            ),
                            "harmonic_oscillator" => solve_ivp(
                                ode_problems::harmonic_oscillator,
                                t_span,
                                y0_array,
                                Some(opts),
                            ),
                            "van_der_pol_mild" => {
                                let mu = 1.0;
                                solve_ivp(
                                    move |_t, y| {
                                        Array1::from_vec(vec![
                                            y[1],
                                            mu * (1.0 - y[0] * y[0]) * y[1] - y[0],
                                        ])
                                    },
                                    t_span,
                                    y0_array,
                                    Some(opts),
                                )
                            }
                            "van_der_pol_stiff" => {
                                let mu = 100.0;
                                solve_ivp(
                                    move |_t, y| {
                                        Array1::from_vec(vec![
                                            y[1],
                                            mu * (1.0 - y[0] * y[0]) * y[1] - y[0],
                                        ])
                                    },
                                    t_span,
                                    y0_array,
                                    Some(opts),
                                )
                            }
                            "lotka_volterra" => {
                                let (a, b, c, d) = (1.5, 1.0, 3.0, 1.0);
                                solve_ivp(
                                    move |_t, y| {
                                        let x = y[0]; // prey
                                        let y_val = y[1]; // predator
                                        Array1::from_vec(vec![
                                            a * x - b * x * y_val,      // dx/dt
                                            -c * y_val + d * x * y_val, // dy/dt
                                        ])
                                    },
                                    t_span,
                                    y0_array,
                                    Some(opts),
                                )
                            }
                            "three_body" => solve_ivp(
                                ode_problems::three_body_problem,
                                t_span,
                                y0_array,
                                Some(opts),
                            ),
                            _ => panic!("Unknown problem"),
                        };

                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark quadrature methods
fn bench_quadrature_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quadrature Methods");

    let problems = vec![
        ("polynomial_cubic", 0.0, 1.0),
        ("oscillatory", 0.0, 1.0),
        ("gaussian", -3.0, 3.0),
        ("nearly_singular", 1e-6, 1.0),
    ];

    for (prob_name, a, b) in problems {
        let parameter = format!("{}_{:.1e}_{:.1e}", prob_name, a, b);

        group.bench_with_input(
            BenchmarkId::new("quad", &parameter),
            &(prob_name, a, b),
            |bench, &(prob_name, a, b)| {
                bench.iter(|| {
                    let opts = QuadOptions {
                        abs_tol: 1e-10,
                        rel_tol: 1e-10,
                        max_evals: 1000,
                        ..Default::default()
                    };

                    let result = match prob_name {
                        "polynomial_cubic" => {
                            quad(quadrature_problems::polynomial_cubic, a, b, Some(opts))
                        }
                        "oscillatory" => quad(quadrature_problems::oscillatory, a, b, Some(opts)),
                        "gaussian" => quad(quadrature_problems::gaussian, a, b, Some(opts)),
                        "nearly_singular" => {
                            quad(quadrature_problems::nearly_singular, a, b, Some(opts))
                        }
                        _ => panic!("Unknown problem"),
                    };

                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark multidimensional integration
fn bench_multidimensional_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multidimensional Integration");

    let dimensions = vec![2, 3, 4, 5, 6];

    for &dim in &dimensions {
        // Monte Carlo integration
        group.bench_with_input(
            BenchmarkId::new("monte_carlo", format!("gaussian_{}d", dim)),
            &dim,
            |b, &dim| {
                b.iter(|| {
                    let ranges: Vec<(f64, f64)> = (0..dim).map(|_| (-2.0, 2.0)).collect();
                    let opts = MonteCarloOptions {
                        n_samples: 100_000,
                        ..Default::default()
                    };

                    let result = monte_carlo(
                        |x| quadrature_problems::multivariate_gaussian(x.as_slice().unwrap()),
                        &ranges,
                        Some(opts),
                    );

                    black_box(result)
                })
            },
        );

        // Cubature integration (for reasonable dimensions)
        if dim <= 4 {
            group.bench_with_input(
                BenchmarkId::new("cubature", format!("gaussian_{}d", dim)),
                &dim,
                |b, &dim| {
                    b.iter(|| {
                        let ranges: Vec<(
                            scirs2_integrate::cubature::Bound<f64>,
                            scirs2_integrate::cubature::Bound<f64>,
                        )> = (0..dim)
                            .map(|_| {
                                (
                                    scirs2_integrate::cubature::Bound::Finite(-2.0),
                                    scirs2_integrate::cubature::Bound::Finite(2.0),
                                )
                            })
                            .collect();
                        let opts = CubatureOptions {
                            rel_tol: 1e-6,
                            abs_tol: 1e-9,
                            max_evals: 100_000,
                            ..Default::default()
                        };

                        let result = cubature(
                            |x: &Array1<f64>| {
                                quadrature_problems::multivariate_gaussian(x.as_slice().unwrap())
                            },
                            &ranges,
                            Some(opts),
                        );

                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel operations
#[cfg(feature = "parallel")]
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Operations");

    // Parallel Monte Carlo vs sequential
    let dimensions = vec![3, 4, 5];
    let sample_counts = vec![10_000, 100_000, 1_000_000];

    for &dim in &dimensions {
        for &n_samples in &sample_counts {
            let ranges: Vec<(f64, f64)> = (0..dim).map(|_| (-2.0, 2.0)).collect();

            // Sequential version
            group.bench_with_input(
                BenchmarkId::new(
                    "monte_carlo_sequential",
                    format!("{}d_{}k", dim, n_samples / 1000),
                ),
                &(dim, n_samples),
                |b, &(_dim, n_samples)| {
                    b.iter(|| {
                        let opts = MonteCarloOptions {
                            n_samples,
                            ..Default::default()
                        };

                        let result = monte_carlo(
                            |x| quadrature_problems::multivariate_gaussian(x.as_slice().unwrap()),
                            &ranges,
                            Some(opts),
                        );

                        black_box(result)
                    })
                },
            );

            // Parallel version
            group.bench_with_input(
                BenchmarkId::new(
                    "monte_carlo_parallel",
                    format!("{}d_{}k", dim, n_samples / 1000),
                ),
                &(dim, n_samples),
                |b, &(_dim, n_samples)| {
                    b.iter(|| {
                        use scirs2_integrate::monte_carlo_parallel::{
                            parallel_monte_carlo, ParallelMonteCarloOptions,
                        };

                        let opts = ParallelMonteCarloOptions {
                            n_samples,
                            batch_size: n_samples / 8,
                            ..Default::default()
                        };

                        let result = parallel_monte_carlo(
                            |x: ndarray::ArrayView1<f64>| {
                                quadrature_problems::multivariate_gaussian(x.as_slice().unwrap())
                            },
                            &ranges,
                            Some(opts),
                        );

                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Memory usage benchmark
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage");

    // Large ODE systems
    let system_sizes = vec![100, 500, 1000];

    for &n in &system_sizes {
        group.bench_with_input(
            BenchmarkId::new("large_ode_system", format!("{}x{}", n, n)),
            &n,
            |b, &n| {
                b.iter(|| {
                    // Create a large linear ODE system: dy/dt = A*y
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

                    let result = solve_ivp(linear_system, [0.0, 1.0], y0, Some(opts));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Accuracy vs performance trade-off benchmark
fn bench_accuracy_performance_tradeoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("Accuracy vs Performance");

    let tolerances = vec![1e-3, 1e-6, 1e-9, 1e-12];

    for &tol in &tolerances {
        group.bench_with_input(
            BenchmarkId::new("ode_tolerance", format!("rtol_{:.0e}", tol)),
            &tol,
            |b, &tol| {
                b.iter(|| {
                    let opts = ODEOptions {
                        rtol: tol,
                        atol: tol * 1e-3,
                        method: ODEMethod::DOP853,
                        ..Default::default()
                    };

                    let y0 = Array1::from_vec(vec![1.0, 0.0]);
                    let result = solve_ivp(
                        ode_problems::harmonic_oscillator,
                        [0.0, 10.0],
                        y0,
                        Some(opts),
                    );

                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_ode_solvers,
    bench_quadrature_methods,
    bench_multidimensional_integration,
    bench_memory_usage,
    bench_accuracy_performance_tradeoff
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_operations);

// Main benchmark runner
#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);
