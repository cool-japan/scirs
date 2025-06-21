//! Example: Local extrapolation methods for high-accuracy ODE solving
//!
//! This example demonstrates the use of local extrapolation techniques,
//! including Richardson extrapolation and the Gragg-Bulirsch-Stoer method,
//! to achieve very high accuracy in ODE integration.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::ode::{
    methods::{
        euler_method, gragg_bulirsch_stoer_method, richardson_extrapolation_step, rk4_method,
        ExtrapolationBaseMethod, ExtrapolationOptions,
    },
    types::ODEOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Local Extrapolation Methods for ODE Solving ===");

    // Test problem 1: Simple exponential decay dy/dt = -y, y(0) = 1
    // Exact solution: y(t) = exp(-t)
    println!("\n1. Exponential Decay Problem: dy/dt = -y, y(0) = 1");
    println!("   Exact solution: y(t) = exp(-t)");

    let exponential_decay = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { -y.to_owned() };

    let y0_exp = Array1::from_vec(vec![1.0]);
    let t_span_exp = [0.0, 1.0];
    let exact_exp = |t: f64| (-t).exp();

    test_extrapolation_methods(
        exponential_decay,
        t_span_exp,
        y0_exp,
        exact_exp,
        "Exponential Decay",
    )?;

    // Test problem 2: Harmonic oscillator d²y/dt² + y = 0
    // Convert to system: dy₁/dt = y₂, dy₂/dt = -y₁
    // With y₁(0) = 1, y₂(0) = 0, exact solution: y₁(t) = cos(t), y₂(t) = -sin(t)
    println!("\n2. Harmonic Oscillator: d²y/dt² + y = 0");
    println!("   System: dy₁/dt = y₂, dy₂/dt = -y₁");
    println!("   Exact solution: y₁(t) = cos(t), y₂(t) = -sin(t)");

    let harmonic_oscillator =
        |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![y[1], -y[0]]) };

    let y0_harm = Array1::from_vec(vec![1.0, 0.0]);
    let t_span_harm = [0.0, 2.0 * std::f64::consts::PI];
    let exact_harm = |t: f64| t.cos(); // For y₁ component

    test_extrapolation_methods(
        harmonic_oscillator,
        t_span_harm,
        y0_harm,
        exact_harm,
        "Harmonic Oscillator",
    )?;

    // Test problem 3: Van der Pol oscillator (stiff for large μ)
    // d²y/dt² - μ(1-y²)dy/dt + y = 0, with μ = 1
    println!("\n3. Van der Pol Oscillator: d²y/dt² - μ(1-y²)dy/dt + y = 0, μ = 1");
    println!("   System: dy₁/dt = y₂, dy₂/dt = μ(1-y₁²)y₂ - y₁");

    let mu = 1.0;
    let van_der_pol = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
    };

    let y0_vdp = Array1::from_vec(vec![2.0, 0.0]);
    let t_span_vdp = [0.0, 10.0];

    // For Van der Pol, we don't have a simple exact solution, so we'll compare methods
    test_van_der_pol_extrapolation(van_der_pol, t_span_vdp, y0_vdp)?;

    println!("\n=== Richardson Extrapolation Demonstration ===");
    demonstrate_richardson_extrapolation()?;

    Ok(())
}

fn test_extrapolation_methods<F, ExactFunc>(
    f: F,
    t_span: [f64; 2],
    y0: Array1<f64>,
    exact_func: ExactFunc,
    problem_name: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy,
    ExactFunc: Fn(f64) -> f64,
{
    let t_end = t_span[1];
    let exact_value = exact_func(t_end);

    println!("\nComparison for {}:", problem_name);
    println!("Method\\t\\t\\tFinal Value\\t\\tError\\t\\t\\tSteps\\tFunc Evals");
    println!("{}", "-".repeat(80));

    // Standard methods for comparison
    let methods = vec![
        ("Euler", "euler"),
        ("RK4", "rk4"),
        ("GBS (Modified Midpoint)", "gbs_mm"),
        ("GBS (Euler)", "gbs_euler"),
        ("GBS (RK4)", "gbs_rk4"),
    ];

    let step_sizes = vec![0.1, 0.05, 0.01];

    for step_size in step_sizes {
        println!("\nStep size h = {}", step_size);

        for (method_name, method_id) in &methods {
            let opts = ODEOptions {
                atol: 1e-8,
                rtol: 1e-8,
                h0: Some(step_size),
                max_step: Some(step_size),
                min_step: Some(step_size / 1000.0),
                max_steps: 10000,
                ..Default::default()
            };

            let result = match *method_id {
                "euler" => euler_method(f, t_span, y0.clone(), step_size, opts),
                "rk4" => rk4_method(f, t_span, y0.clone(), step_size, opts),
                "gbs_mm" => {
                    let ext_opts = ExtrapolationOptions {
                        base_method: ExtrapolationBaseMethod::ModifiedMidpoint,
                        max_order: 8,
                        min_order: 3,
                        ..Default::default()
                    };
                    gragg_bulirsch_stoer_method(f, t_span, y0.clone(), opts, Some(ext_opts))
                }
                "gbs_euler" => {
                    let ext_opts = ExtrapolationOptions {
                        base_method: ExtrapolationBaseMethod::Euler,
                        max_order: 6,
                        min_order: 3,
                        ..Default::default()
                    };
                    gragg_bulirsch_stoer_method(f, t_span, y0.clone(), opts, Some(ext_opts))
                }
                "gbs_rk4" => {
                    let ext_opts = ExtrapolationOptions {
                        base_method: ExtrapolationBaseMethod::RungeKutta4,
                        max_order: 4,
                        min_order: 2,
                        ..Default::default()
                    };
                    gragg_bulirsch_stoer_method(f, t_span, y0.clone(), opts, Some(ext_opts))
                }
                _ => unreachable!(),
            };

            match result {
                Ok(ode_result) => {
                    let final_value = ode_result.y.last().unwrap()[0];
                    let error = (final_value - exact_value).abs();

                    println!(
                        "{:<20}\\t{:.6e}\\t\\t{:.2e}\\t\\t{}\\t{}",
                        method_name, final_value, error, ode_result.n_steps, ode_result.n_eval
                    );
                }
                Err(e) => {
                    println!("{:<20}\\tFAILED: {}", method_name, e);
                }
            }
        }
    }

    Ok(())
}

fn test_van_der_pol_extrapolation<F>(
    f: F,
    t_span: [f64; 2],
    y0: Array1<f64>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy,
{
    println!("\nVan der Pol Oscillator - Method Comparison:");
    println!("Method\\t\\t\\tFinal y₁\\t\\tFinal y₂\\t\\tSteps\\tFunc Evals");
    println!("{}", "-".repeat(75));

    let opts = ODEOptions {
        atol: 1e-6,
        rtol: 1e-6,
        h0: Some(0.1),
        max_steps: 10000,
        ..Default::default()
    };

    // Test different extrapolation methods
    let ext_methods = vec![
        (
            "GBS (Mod. Midpoint)",
            ExtrapolationBaseMethod::ModifiedMidpoint,
        ),
        ("GBS (Euler)", ExtrapolationBaseMethod::Euler),
        ("GBS (RK4)", ExtrapolationBaseMethod::RungeKutta4),
    ];

    for (method_name, base_method) in ext_methods {
        let ext_opts = ExtrapolationOptions {
            base_method,
            max_order: 8,
            min_order: 3,
            ..Default::default()
        };

        match gragg_bulirsch_stoer_method(f, t_span, y0.clone(), opts.clone(), Some(ext_opts)) {
            Ok(result) => {
                let final_y = result.y.last().unwrap();
                println!(
                    "{:<20}\\t{:.6}\\t\\t{:.6}\\t\\t{}\\t{}",
                    method_name, final_y[0], final_y[1], result.n_steps, result.n_eval
                );
            }
            Err(e) => {
                println!("{:<20}\\tFAILED: {}", method_name, e);
            }
        }
    }

    Ok(())
}

fn demonstrate_richardson_extrapolation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nRichardson Extrapolation with Single Steps:");
    println!("Problem: dy/dt = -y, y(0) = 1, step from t=0 to t=0.1");
    println!("Exact solution at t=0.1: {:.10}", (-0.1_f64).exp());

    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { -y.to_owned() };

    let y0 = Array1::from_vec(vec![1.0]);
    let h = 0.1f64;
    let exact = (-h).exp();

    // Simple Euler step
    let euler_step = |f: &dyn Fn(f64, ArrayView1<f64>) -> Array1<f64>,
                      t: f64,
                      y: &Array1<f64>,
                      h: f64|
     -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let dy = f(t, y.view());
        Ok(y + &dy * h)
    };

    // Single Euler step
    let y_euler = euler_step(&f, 0.0, &y0, h)?;
    let error_euler = (y_euler[0] - exact).abs();

    // Richardson extrapolation
    let (y_rich, error_est) = richardson_extrapolation_step(
        |f, t, y, h| {
            euler_step(f, t, y, h)
                .map_err(|e| scirs2_integrate::error::IntegrateError::ValueError(e.to_string()))
        },
        &f,
        0.0,
        &y0,
        h,
    )?;
    let error_rich = (y_rich[0] - exact).abs();

    println!("\\nMethod\\t\\t\\tResult\\t\\t\\tActual Error\\t\\tError Estimate");
    println!("{}", "-".repeat(70));
    println!(
        "Euler\\t\\t\\t{:.10}\\t{:.2e}\\t\\t-",
        y_euler[0], error_euler
    );
    println!(
        "Richardson Extrap\\t{:.10}\\t{:.2e}\\t\\t{:.2e}",
        y_rich[0], error_rich, error_est
    );
    println!("\\nImprovement factor: {:.1}x", error_euler / error_rich);

    Ok(())
}
