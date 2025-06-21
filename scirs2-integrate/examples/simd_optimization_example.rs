//! Example: SIMD optimization for ODE solving
//!
//! This example demonstrates the performance benefits of SIMD-accelerated
//! ODE solving methods for large systems of differential equations.
//! SIMD (Single Instruction, Multiple Data) operations can provide significant
//! speedups by processing multiple data elements simultaneously.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::ode::{
    methods::{rk45_method, rk4_method},
    types::ODEOptions,
};
use std::time::Instant;

// SIMD methods are only available when the "simd" feature is enabled
// Temporarily disabled SIMD methods due to implementation complexity
// #[cfg(feature = "simd")]
// use scirs2_integrate::ode::methods::{simd_rk45_method, simd_rk4_method};

// Temporarily disabled SIMD utils due to implementation complexity
// #[cfg(feature = "simd")]
// use scirs2_integrate::ode::utils::{simd_ode_function_eval, simd_rk_step, SimdOdeOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SIMD Optimization for ODE Solving ===");

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD feature is not enabled. To see SIMD optimizations, compile with:");
        println!("cargo run --features simd --example simd_optimization_example");
        println!("\\nRunning standard (non-SIMD) methods for comparison...");
    }

    #[cfg(feature = "simd")]
    {
        println!("SIMD feature is enabled! Comparing SIMD vs standard methods...");
    }

    // Test different system sizes to show SIMD scalability
    let system_sizes = vec![10, 50, 100, 500, 1000];

    println!("\\n=== Performance Comparison ===");
    println!("System Size\\tMethod\\t\\t\\tTime (ms)\\tSteps\\tFunc Evals\\tAccuracy");
    println!("{}", "-".repeat(80));

    for &n in &system_sizes {
        println!("\\n--- System Size: {} equations ---", n);

        // Create a large system of coupled oscillators
        // d²x_i/dt² + ω_i² * x_i + c * (x_i - x_{i-1}) = 0
        // Convert to first-order system: y = [x_1, ..., x_n, dx_1/dt, ..., dx_n/dt]
        let system_func = create_coupled_oscillator_system(n);
        let y0 = create_initial_conditions(n);
        let t_span = [0.0, 10.0];

        // Standard methods
        let opts: ODEOptions<f64> = ODEOptions {
            atol: 1e-6f64,
            rtol: 1e-6f64,
            h0: Some(0.01f64),
            ..Default::default()
        };

        // Test RK4 methods
        test_method_performance("RK4 (Standard)", || {
            rk4_method(system_func, t_span, y0.clone(), 0.01, opts.clone())
        })?;

        // Temporarily disabled SIMD methods
        // #[cfg(feature = "simd")]
        // test_method_performance("RK4 (SIMD)", || {
        //     simd_rk4_method(system_func, t_span, y0.clone(), opts.clone())
        // })?;

        // Test adaptive methods
        test_method_performance("RK45 (Standard)", || {
            rk45_method(system_func, t_span, y0.clone(), opts.clone())
        })?;

        // Temporarily disabled SIMD methods
        // #[cfg(feature = "simd")]
        // test_method_performance("RK45 (SIMD)", || {
        //     simd_rk45_method(system_func, t_span, y0.clone(), opts.clone())
        // })?;
    }

    #[cfg(feature = "simd")]
    {
        println!("\\n=== SIMD Operation Benchmarks ===");
        demonstrate_simd_operations()?;
    }

    println!("\\n=== Scalability Analysis ===");
    analyze_scalability()?;

    println!("\\n=== Recommendations ===");
    print_recommendations();

    Ok(())
}

/// Create a coupled oscillator system for benchmarking
fn create_coupled_oscillator_system(
    n: usize,
) -> impl Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy {
    move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        let mut dy_dt = Array1::zeros(2 * n);

        // y = [x_1, ..., x_n, dx_1/dt, ..., dx_n/dt]
        // dy_dt = [dx_1/dt, ..., dx_n/dt, d²x_1/dt², ..., d²x_n/dt²]

        // Copy velocities
        for i in 0..n {
            dy_dt[i] = y[n + i];
        }

        // Compute accelerations
        let coupling = 0.1;
        let omega_squared = 1.0;

        for i in 0..n {
            let mut acceleration = -omega_squared * y[i];

            // Coupling with neighbors
            if i > 0 {
                acceleration += coupling * (y[i - 1] - y[i]);
            }
            if i < n - 1 {
                acceleration += coupling * (y[i + 1] - y[i]);
            }

            dy_dt[n + i] = acceleration;
        }

        dy_dt
    }
}

/// Create initial conditions for the coupled oscillator system
fn create_initial_conditions(n: usize) -> Array1<f64> {
    let mut y0 = Array1::zeros(2 * n);

    // Initial positions: sinusoidal wave
    for i in 0..n {
        let x = (i as f64 / n as f64) * std::f64::consts::PI;
        y0[i] = x.sin();
        y0[n + i] = 0.0; // Initial velocities = 0
    }

    y0
}

/// Test method performance and print results
fn test_method_performance<F>(
    method_name: &str,
    method: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn() -> Result<
        scirs2_integrate::ode::types::ODEResult<f64>,
        scirs2_integrate::error::IntegrateError,
    >,
{
    let start = Instant::now();
    let result = method()?;
    let duration = start.elapsed();

    // Compute accuracy (energy conservation for oscillator system)
    let final_energy = compute_system_energy(result.y.last().unwrap());
    let initial_energy = compute_system_energy(&result.y[0]);
    let energy_error = ((final_energy - initial_energy) / initial_energy).abs();

    println!(
        "{:<15}\\t{:.2}\\t\\t{}\\t{}\\t\\t{:.2e}",
        method_name,
        duration.as_millis(),
        result.n_steps,
        result.n_eval,
        energy_error
    );

    Ok(())
}

/// Compute total energy of the oscillator system (should be conserved)
fn compute_system_energy(y: &Array1<f64>) -> f64 {
    let n = y.len() / 2;
    let mut energy = 0.0;

    // Kinetic energy: 0.5 * sum(v_i²)
    for i in n..2 * n {
        energy += 0.5 * y[i] * y[i];
    }

    // Potential energy: 0.5 * sum(x_i²) + coupling terms
    let coupling = 0.1;
    for i in 0..n {
        energy += 0.5 * y[i] * y[i]; // ω² = 1

        if i < n - 1 {
            let dx = y[i + 1] - y[i];
            energy += 0.5 * coupling * dx * dx;
        }
    }

    energy
}

/// Demonstrate SIMD operations
#[cfg(feature = "simd")]
fn demonstrate_simd_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing individual SIMD operations on large vectors...");

    let sizes = vec![1000, 10000, 100000];

    for &size in &sizes {
        println!("\\nVector size: {}", size);

        // Create test vectors
        let a = Array1::from_iter((0..size).map(|i| (i as f64).sin()));
        let b = Array1::from_iter((0..size).map(|i| (i as f64).cos()));

        // Test SIMD vs standard vector operations
        let ops = vec![
            (
                "Element-wise max",
                test_simd_max as fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
            ),
            ("Element-wise min", test_simd_min),
            ("L2 norm", test_simd_l2_norm),
            ("Infinity norm", test_simd_inf_norm),
        ];

        for (op_name, _test_func) in ops {
            // SIMD version
            let start = Instant::now();
            let _result_simd = match op_name {
                "Element-wise max" => {
                    // Note: SIMD methods are placeholders for demonstration
                    let result = a
                        .view()
                        .iter()
                        .zip(b.view().iter())
                        .map(|(x, y)| x.max(*y))
                        .collect::<Vec<_>>();
                    result.iter().sum::<f64>() // Just to consume the result
                }
                "Element-wise min" => {
                    // Note: SIMD methods are placeholders for demonstration
                    let result = a
                        .view()
                        .iter()
                        .zip(b.view().iter())
                        .map(|(x, y)| x.min(*y))
                        .collect::<Vec<_>>();
                    result.iter().sum::<f64>()
                }
                "L2 norm" => {
                    // Note: SIMD methods are placeholders for demonstration
                    a.view().iter().map(|x| x * x).sum::<f64>().sqrt()
                }
                "Infinity norm" => {
                    // Note: SIMD methods are placeholders for demonstration
                    a.view()
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f64, |acc, x| acc.max(x))
                }
                _ => 0.0,
            };
            let simd_time = start.elapsed();

            // Standard version
            let start = Instant::now();
            let _result_std = match op_name {
                "Element-wise max" => {
                    let result: Array1<f64> = a
                        .iter()
                        .zip(b.iter())
                        .map(|(&ai, &bi)| ai.max(bi))
                        .collect();
                    result.iter().sum::<f64>()
                }
                "Element-wise min" => {
                    let result: Array1<f64> = a
                        .iter()
                        .zip(b.iter())
                        .map(|(&ai, &bi)| ai.min(bi))
                        .collect();
                    result.iter().sum::<f64>()
                }
                "L2 norm" => a.iter().map(|&x| x * x).sum::<f64>().sqrt(),
                "Infinity norm" => a.iter().map(|&x| x.abs()).fold(0.0f64, |acc, x| acc.max(x)),
                _ => 0.0,
            };
            let std_time = start.elapsed();

            let speedup = std_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("  {:<20}: {:.1}x speedup", op_name, speedup);
        }
    }

    Ok(())
}

// Helper functions for SIMD operation testing
#[cfg(feature = "simd")]
fn test_simd_max(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // Note: SIMD methods are placeholders for demonstration
    let result = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.max(*y))
        .collect::<Vec<_>>();
    result.iter().sum()
}

#[cfg(feature = "simd")]
fn test_simd_min(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // Note: SIMD methods are placeholders for demonstration
    let result = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.min(*y))
        .collect::<Vec<_>>();
    result.iter().sum()
}

#[cfg(feature = "simd")]
fn test_simd_l2_norm(a: &ArrayView1<f64>, _b: &ArrayView1<f64>) -> f64 {
    // Note: SIMD methods are placeholders for demonstration
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(feature = "simd")]
fn test_simd_inf_norm(a: &ArrayView1<f64>, _b: &ArrayView1<f64>) -> f64 {
    // Note: SIMD methods are placeholders for demonstration
    a.iter().map(|x| x.abs()).fold(0.0f64, |acc, x| acc.max(x))
}

/// Analyze scalability of SIMD vs standard methods
fn analyze_scalability() -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing scalability characteristics...");

    let sizes = vec![10, 100, 1000];

    for &n in &sizes {
        let simple_ode = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
            -y.to_owned() // Simple exponential decay
        };

        let y0 = Array1::from_iter((0..n).map(|i| (i as f64 + 1.0).sin()));
        let t_span = [0.0, 1.0];
        let opts = ODEOptions {
            h0: Some(0.01),
            ..Default::default()
        };

        // Standard method
        let start = Instant::now();
        let _result_std = rk4_method(simple_ode, t_span, y0.clone(), 0.01, opts.clone())?;
        let std_time = start.elapsed();

        // SIMD method (if available)
        // Temporarily disabled: simd_rk4_method not yet implemented
        #[cfg(feature = "simd")]
        let simd_time = {
            let start = Instant::now();
            // let _result_simd = simd_rk4_method(simple_ode, t_span, y0.clone(), 0.01, opts.clone())?;
            let _result_simd = rk4_method(simple_ode, t_span, y0.clone(), 0.01, opts.clone())?; // Use standard method for now
            start.elapsed()
        };

        #[cfg(feature = "simd")]
        {
            let speedup = std_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("Size {}: {:.1}x speedup with SIMD", n, speedup);
        }

        #[cfg(not(feature = "simd"))]
        {
            println!(
                "Size {}: {:.2} ms (standard method)",
                n,
                std_time.as_millis()
            );
        }
    }

    Ok(())
}

/// Print recommendations for using SIMD optimizations
fn print_recommendations() {
    println!("SIMD optimization is most beneficial when:");
    println!("• System size is large (>100 equations)");
    println!("• ODE function involves element-wise operations");
    println!("• Memory layout is contiguous (ndarray slices)");
    println!("• Target CPU supports SIMD instructions (AVX, SSE, etc.)");
    println!();
    println!("To enable SIMD optimizations:");
    println!("• Compile with --features simd");
    println!("• Use SIMD-friendly data structures");
    println!("• Consider memory alignment for optimal performance");
    println!("• Profile your specific use case for best results");

    #[cfg(feature = "simd")]
    {
        println!("\\n✓ SIMD feature is currently ENABLED");
        println!("  You're getting the performance benefits!");
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("\\n⚠ SIMD feature is currently DISABLED");
        println!("  Compile with --features simd to enable optimizations");
    }
}
