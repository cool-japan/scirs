use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::utils::dense_output::{create_dense_solution, DenseSolution};
use scirs2_integrate::ode::utils::interpolation::ContinuousOutputMethod;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;
use std::io::Write;

/// This example demonstrates the continuous (dense) output capabilities
/// of the ODE solvers, which allow evaluating the solution at any point
/// within the integration interval.
/// Pendulum system: d²θ/dt² + (g/L)·sin(θ) = 0
///
/// We convert this to a system of first-order ODEs:
/// dθ/dt = ω
/// dω/dt = -(g/L)·sin(θ)
fn pendulum(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = θ (angle from vertical, radians)
    // y[1] = ω (angular velocity)

    // Parameters
    let g_over_l = 1.0; // Gravity / Length

    // Equations of motion
    array![y[1], -g_over_l * y[0].sin()]
}

/// Two-body problem (planet orbiting a star)
///
/// We model this as a system of ODEs in Cartesian coordinates:
/// dx/dt = vx
/// dy/dt = vy
/// dvx/dt = -μ·x/r³
/// dvy/dt = -μ·y/r³
///
/// Where r = sqrt(x² + y²) is the distance from the origin
fn two_body(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = x (position x)
    // y[1] = y (position y)
    // y[2] = vx (velocity x)
    // y[3] = vy (velocity y)

    // Parameter (gravitational constant * central mass)
    let mu = 1.0;

    // Distance from origin
    let r = (y[0] * y[0] + y[1] * y[1]).sqrt();

    // Avoid division by zero
    if r < 1e-10 {
        return array![y[2], y[3], 0.0, 0.0];
    }

    // Gravitational acceleration
    let r3 = r.powi(3);
    let ax = -mu * y[0] / r3;
    let ay = -mu * y[1] / r3;

    array![
        y[2], // dx/dt = vx
        y[3], // dy/dt = vy
        ax,   // dvx/dt = -μ·x/r³
        ay    // dvy/dt = -μ·y/r³
    ]
}

/// Van der Pol oscillator
///
/// This is a classic non-linear oscillator with limit cycle behavior:
/// d²x/dt² - μ·(1 - x²)·dx/dt + x = 0
///
/// We convert to first-order system:
/// dx/dt = y
/// dy/dt = μ·(1 - x²)·y - x
fn van_der_pol(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
    // State variables:
    // y[0] = x (position)
    // y[1] = y (velocity)

    // Parameter (controls nonlinearity and stiffness)
    let mu = 2.0;

    array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]]
}

/// Writes a CSV file with the solution
fn write_solution_csv(
    filename: &str,
    solution: &DenseSolution<f64>,
    component_indices: &[usize],
    n_points: usize,
) -> IntegrateResult<()> {
    // Create a dense output
    let (times, values) = solution.dense_output(n_points)?;

    // Open file for writing
    let mut file = std::fs::File::create(filename)
        .map_err(|e| scirs2_integrate::error::IntegrateError::ComputationError(e.to_string()))?;

    // Write header
    let mut header = String::from("t");
    for &idx in component_indices {
        header.push_str(&format!(",y{}", idx));
    }
    writeln!(file, "{}", header)
        .map_err(|e| scirs2_integrate::error::IntegrateError::ComputationError(e.to_string()))?;

    // Write data
    for (i, t) in times.iter().enumerate() {
        let mut line = format!("{}", t);
        for &idx in component_indices {
            line.push_str(&format!(",{}", values[i][idx]));
        }
        writeln!(file, "{}", line).map_err(|e| {
            scirs2_integrate::error::IntegrateError::ComputationError(e.to_string())
        })?;
    }

    println!("Wrote solution to {}", filename);
    Ok(())
}

/// Pendulum simulation with continuous output
fn pendulum_simulation() -> IntegrateResult<()> {
    println!("\n=== Pendulum Simulation ===");

    // Initial state: θ=π/4, ω=0 (start from 45 degrees with no velocity)
    let y0 = array![PI / 4.0, 0.0];
    let t_span = [0.0, 20.0];

    // Use the RK45 method
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        ..Default::default()
    };

    println!("Solving pendulum ODE...");
    let result = solve_ivp(pendulum, t_span, y0, Some(options))?;

    println!("Solution has {} points", result.t.len());
    println!("First few times: {:?}", &result.t[0..5.min(result.t.len())]);

    // Create a dense solution
    println!("Creating dense solution...");
    let dense_solution = create_dense_solution(
        result.t,
        result.y,
        pendulum,
        Some(ContinuousOutputMethod::CubicHermite),
    )?;

    // Demonstrate dense evaluation by computing energy at many points
    // Energy = T + V = (1/2)mL²ω² + mgL(1-cos(θ))
    // With m=1, L=1, g=1: E = (1/2)ω² + (1-cos(θ))
    let n_points = 1000;
    let (_times, values) = dense_solution.dense_output(n_points)?;

    println!("Computing energy conservation...");
    let mut energies = Vec::with_capacity(n_points);
    for value in values.iter().take(n_points) {
        let theta = value[0];
        let omega = value[1];
        let kinetic = 0.5 * omega.powi(2);
        let potential = 1.0 - theta.cos();
        let energy = kinetic + potential;
        energies.push(energy);
    }

    // Check energy conservation
    let mean_energy: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
    let max_deviation = energies
        .iter()
        .fold(0.0_f64, |max, &e| max.max((e - mean_energy).abs()));

    println!("Energy analysis:");
    println!("  Mean energy: {:.8}", mean_energy);
    println!("  Maximum deviation: {:.8e}", max_deviation);
    println!("  Relative error: {:.8e}", max_deviation / mean_energy);

    // Write solution to CSV for visualization
    write_solution_csv("pendulum_solution.csv", &dense_solution, &[0, 1], 500)?;

    Ok(())
}

/// Two-body problem simulation (planet orbiting a star)
fn two_body_simulation() -> IntegrateResult<()> {
    println!("\n=== Two-Body Simulation ===");

    // Initial conditions for circular orbit: x=1, y=0, vx=0, vy=1
    let y0 = array![1.0, 0.0, 0.0, 1.0];
    let t_span = [0.0, 12.0]; // integrate for approx. 2 orbits

    // Use the DOP853 method for high accuracy
    let options = ODEOptions {
        method: ODEMethod::DOP853,
        rtol: 1e-10,
        atol: 1e-10,
        ..Default::default()
    };

    println!("Solving two-body ODE...");
    let result = solve_ivp(two_body, t_span, y0, Some(options))?;

    println!("Solution has {} points", result.t.len());

    // Create a dense solution
    println!("Creating dense solution...");
    let dense_solution = create_dense_solution(
        result.t,
        result.y,
        two_body,
        Some(ContinuousOutputMethod::CubicHermite),
    )?;

    // Demonstrate dense output by computing angular momentum at many points
    // L = r × p = x*vy - y*vx
    let n_points = 1000;
    let (_times, values) = dense_solution.dense_output(n_points)?;

    println!("Computing angular momentum conservation...");
    let mut angular_momenta = Vec::with_capacity(n_points);
    for value in values.iter().take(n_points) {
        let x = value[0];
        let y = value[1];
        let vx = value[2];
        let vy = value[3];

        let angular_momentum = x * vy - y * vx;
        angular_momenta.push(angular_momentum);
    }

    // Check angular momentum conservation
    let mean_l: f64 = angular_momenta.iter().sum::<f64>() / angular_momenta.len() as f64;
    let max_deviation = angular_momenta
        .iter()
        .fold(0.0_f64, |max, &l| max.max((l - mean_l).abs()));

    println!("Angular momentum analysis:");
    println!("  Mean angular momentum: {:.10}", mean_l);
    println!("  Maximum deviation: {:.10e}", max_deviation);
    println!("  Relative error: {:.10e}", max_deviation / mean_l);

    // Write solution to CSV for visualization
    write_solution_csv("two_body_solution.csv", &dense_solution, &[0, 1], 500)?;

    // Also write radius over time to verify circular orbit
    let (t_dense, values_dense) = dense_solution.dense_output(500)?;

    let mut file = std::fs::File::create("two_body_radius.csv")
        .map_err(|e| scirs2_integrate::error::IntegrateError::ComputationError(e.to_string()))?;

    writeln!(file, "t,radius").unwrap();
    for i in 0..t_dense.len() {
        let x = values_dense[i][0];
        let y = values_dense[i][1];
        let r = (x * x + y * y).sqrt();
        writeln!(file, "{},{}", t_dense[i], r).unwrap();
    }

    println!("Wrote radius data to two_body_radius.csv");

    Ok(())
}

/// Van der Pol oscillator simulation with continuous output
fn van_der_pol_simulation() -> IntegrateResult<()> {
    println!("\n=== Van der Pol Simulation ===");

    // Initial state: x=2, y=0
    let y0 = array![2.0, 0.0];
    let t_span = [0.0, 20.0];

    // Use the enhanced BDF method for stiff problems
    let options = ODEOptions {
        method: ODEMethod::EnhancedBDF,
        rtol: 1e-6,
        atol: 1e-8,
        ..Default::default()
    };

    println!("Solving Van der Pol ODE...");
    let result = solve_ivp(van_der_pol, t_span, y0, Some(options))?;

    println!(
        "Solution has {} points with {} steps",
        result.t.len(),
        result.n_steps
    );

    // Create a dense solution
    println!("Creating dense solution...");
    let _dense_solution = create_dense_solution(
        result.t.clone(),
        result.y.clone(),
        van_der_pol,
        Some(ContinuousOutputMethod::CubicHermite),
    )?;

    // Compare different interpolation methods
    println!("\nInterpolation Method Comparison at t=7.5:");
    let t_test = 7.5;

    // Find closest actual solution points
    let mut closest_i = 0;
    let mut min_dist = f64::MAX;
    for (i, &t) in result.t.iter().enumerate() {
        let dist = (t - t_test).abs();
        if dist < min_dist {
            min_dist = dist;
            closest_i = i;
        }
    }

    // Get closest times
    let t_before = if closest_i > 0 {
        result.t[closest_i - 1]
    } else {
        result.t[0]
    };
    let t_at = result.t[closest_i];
    let t_after = if closest_i < result.t.len() - 1 {
        result.t[closest_i + 1]
    } else {
        t_at
    };

    println!(
        "Closest solution points: t = {:.6}, {:.6}, {:.6}",
        t_before, t_at, t_after
    );

    // Create dense solutions with different interpolation methods
    let linear_solution = create_dense_solution(
        result.t.clone(),
        result.y.clone(),
        van_der_pol,
        Some(ContinuousOutputMethod::Linear),
    )?;

    let cubic_solution = create_dense_solution(
        result.t.clone(),
        result.y.clone(),
        van_der_pol,
        Some(ContinuousOutputMethod::CubicHermite),
    )?;

    // Evaluate at test point
    let linear_y = linear_solution.evaluate(t_test)?;
    let cubic_y = cubic_solution.evaluate(t_test)?;

    println!(
        "Linear interpolation:    x = {:.8}, y = {:.8}",
        linear_y[0], linear_y[1]
    );
    println!(
        "Cubic Hermite interp.:   x = {:.8}, y = {:.8}",
        cubic_y[0], cubic_y[1]
    );

    // Generate very high resolution "truth" solution to compare against
    let fine_options = ODEOptions {
        method: ODEMethod::EnhancedBDF,
        rtol: 1e-10,
        atol: 1e-12,
        max_steps: 10000,
        ..Default::default()
    };

    // Solve just around the test point
    let fine_result = solve_ivp(
        van_der_pol,
        [t_test - 0.1, t_test + 0.1],
        array![cubic_y[0], cubic_y[1]], // Start from our interpolated solution
        Some(fine_options),
    )?;

    // Find closest point to t_test in fine solution
    let mut reference_y = None;
    let mut min_dist = f64::MAX;
    for i in 0..fine_result.t.len() {
        let dist = (fine_result.t[i] - t_test).abs();
        if dist < min_dist {
            min_dist = dist;
            reference_y = Some(fine_result.y[i].clone());
        }
    }

    if let Some(ref_y) = reference_y {
        println!(
            "Reference solution:      x = {:.8}, y = {:.8}",
            ref_y[0], ref_y[1]
        );

        println!("\nInterpolation errors:");
        println!(
            "Linear: {:.2e} (x), {:.2e} (y)",
            (linear_y[0] - ref_y[0]).abs(),
            (linear_y[1] - ref_y[1]).abs()
        );
        println!(
            "Cubic:  {:.2e} (x), {:.2e} (y)",
            (cubic_y[0] - ref_y[0]).abs(),
            (cubic_y[1] - ref_y[1]).abs()
        );
    }

    // Write solution to CSV for visualization
    write_solution_csv("van_der_pol_solution.csv", &cubic_solution, &[0, 1], 1000)?;

    // Generate phase plane for Van der Pol
    println!("\nGenerating phase plane visualization...");
    let mut phase_file = std::fs::File::create("van_der_pol_phase.csv")
        .map_err(|e| scirs2_integrate::error::IntegrateError::ComputationError(e.to_string()))?;

    writeln!(phase_file, "x,y").unwrap();

    // Use dense output with many points
    let (_, values) = cubic_solution.dense_output(2000)?;
    for v in values {
        writeln!(phase_file, "{},{}", v[0], v[1]).unwrap();
    }

    println!("Wrote phase plane data to van_der_pol_phase.csv");

    Ok(())
}

fn main() -> IntegrateResult<()> {
    println!("Continuous Output Demonstrations");
    println!("================================");

    // Run all simulations
    pendulum_simulation()?;
    two_body_simulation()?;
    van_der_pol_simulation()?;

    println!("\nFinished all demonstrations.");
    println!("\nSummary of Continuous Output Benefits:");
    println!("1. Enables accurate solution evaluation at any point within the integration range");
    println!("2. Useful for visualization and post-processing of solutions");
    println!("3. Helps verify conservation properties of dynamical systems");
    println!("4. Allows comparison of different interpolation methods");
    println!("5. More efficient than re-solving with smaller step sizes");

    Ok(())
}
