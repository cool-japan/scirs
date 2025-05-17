use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::time::Instant;

// Van der Pol oscillator - stiff when mu is large
// dy/dt = [y1, mu * (1 - y0^2) * y1 - y0]
fn van_der_pol(mu: f64) -> impl Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy {
    move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]]
}

// Robertson chemical reaction system - a classic stiff ODE system
// dy/dt = [-0.04*y0 + 1e4*y1*y2, 0.04*y0 - 1e4*y1*y2 - 3e7*y1^2, 3e7*y1^2]
fn robertson(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    array![
        -0.04 * y[0] + 1.0e4 * y[1] * y[2],
        0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1].powi(2),
        3.0e7 * y[1].powi(2)
    ]
}

// HIRES problem (High Irradiance RESponse) - stiff problem from chemical kinetics
fn hires(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    let mut dy = Array1::<f64>::zeros(8);

    dy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
    dy[1] = 1.71 * y[0] - 8.75 * y[1];
    dy[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
    dy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
    dy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
    dy[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
    dy[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
    dy[7] = -280.0 * y[5] * y[7] + 1.81 * y[6];

    dy
}

// Solve a problem with both standard BDF and enhanced BDF, then compare results
fn compare_methods<Func>(
    name: &str,
    f: Func,
    t_span: [f64; 2],
    y0: Array1<f64>,
    rtol: f64,
    atol: f64,
) -> IntegrateResult<()>
where
    Func: Fn(f64, ArrayView1<f64>) -> Array1<f64> + Copy,
{
    println!("\n=== {} ===", name);

    // Standard BDF
    let start = Instant::now();
    let result_std = solve_ivp(
        f,
        t_span,
        y0.clone(),
        Some(ODEOptions {
            method: ODEMethod::Bdf,
            rtol,
            atol,
            ..Default::default()
        }),
    )?;
    let std_time = start.elapsed();

    println!(
        "Standard BDF: solved in {:.3} ms, {} steps ({} accepted, {} rejected), {} Jacobians",
        std_time.as_secs_f64() * 1000.0,
        result_std.n_steps,
        result_std.n_accepted,
        result_std.n_rejected,
        result_std.n_jac
    );

    // Enhanced BDF
    let start = Instant::now();
    let result_enh = solve_ivp(
        f,
        t_span,
        y0,
        Some(ODEOptions {
            method: ODEMethod::EnhancedBDF,
            rtol,
            atol,
            ..Default::default()
        }),
    )?;
    let enh_time = start.elapsed();

    println!(
        "Enhanced BDF: solved in {:.3} ms, {} steps ({} accepted, {} rejected), {} Jacobians",
        enh_time.as_secs_f64() * 1000.0,
        result_enh.n_steps,
        result_enh.n_accepted,
        result_enh.n_rejected,
        result_enh.n_jac
    );

    // Calculate relative performance
    let speedup = std_time.as_secs_f64() / enh_time.as_secs_f64();
    let step_reduction =
        (result_std.n_steps as f64 - result_enh.n_steps as f64) / result_std.n_steps as f64 * 100.0;
    let jac_reduction =
        (result_std.n_jac as f64 - result_enh.n_jac as f64) / result_std.n_jac as f64 * 100.0;

    println!(
        "Performance: {:.2}x speedup, {:.1}% fewer steps, {:.1}% fewer Jacobian evaluations",
        speedup, step_reduction, jac_reduction
    );

    // Compare last values to ensure accuracy
    if !result_std.success || !result_enh.success {
        println!("WARNING: One or more solvers did not report success!");
    }

    let last_y_std = result_std.y.last().unwrap();
    let last_y_enh = result_enh.y.last().unwrap();

    let max_abs_diff = last_y_std
        .iter()
        .zip(last_y_enh.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, |acc, x| acc.max(x));

    println!(
        "Solution difference: {:.2e} (maximum absolute)",
        max_abs_diff
    );

    Ok(())
}

fn main() -> IntegrateResult<()> {
    println!("Comparing standard BDF vs. Enhanced BDF methods");
    println!("===============================================\n");

    // Test 1: Van der Pol with moderate stiffness (mu=10)
    compare_methods(
        "Van der Pol (mu=10)",
        van_der_pol(10.0),
        [0.0, 30.0],
        array![2.0, 0.0],
        1e-6,
        1e-8,
    )?;

    // Test 2: Van der Pol with high stiffness (mu=1000)
    compare_methods(
        "Van der Pol (mu=1000, very stiff)",
        van_der_pol(1000.0),
        [0.0, 3000.0],
        array![2.0, 0.0],
        1e-6,
        1e-8,
    )?;

    // Test 3: Robertson chemical system
    compare_methods(
        "Robertson chemical system",
        robertson,
        [0.0, 1e11],
        array![1.0, 0.0, 0.0],
        1e-4,
        1e-8,
    )?;

    // Test 4: HIRES problem
    compare_methods(
        "HIRES (High Irradiance RESponse)",
        hires,
        [0.0, 321.8122],
        array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026],
        1e-4,
        1e-8,
    )?;

    println!("\nAll tests completed successfully.");
    println!("\nNote on performance metrics:");
    println!("1. Enhanced BDF uses intelligent Jacobian strategy selection based on problem size");
    println!("2. For large systems, Jacobian reuse (modified Newton) provides significant speedup");
    println!("3. Better error estimation allows for larger step sizes with the same accuracy");
    println!("4. Adaptive order selection further improves efficiency for smooth regions");

    Ok(())
}
