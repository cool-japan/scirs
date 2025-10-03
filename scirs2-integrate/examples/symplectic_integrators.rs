//! Example demonstrating symplectic integrators for Hamiltonian systems
//!
//! This example shows how to use different symplectic integrators
//! for various Hamiltonian systems, highlighting energy conservation
//! properties and comparing the accuracy of different methods.

use scirs2_core::ndarray::{array, Array1};
use scirs2_integrate::symplectic::{
    position_verlet, velocity_verlet, CompositionMethod, GaussLegendre4, GaussLegendre6,
    HamiltonianFn, SeparableHamiltonian, StormerVerlet, SymplecticIntegrator,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("Symplectic Integrators Example");
    println!("==============================\n");

    // Run examples
    simple_harmonic_oscillator();
    pendulum();
    kepler_orbit();
    compare_methods();

    println!("\nAll examples completed successfully!");
}

/// Simple harmonic oscillator example
#[allow(dead_code)]
fn simple_harmonic_oscillator() {
    println!("1. Simple Harmonic Oscillator");
    println!("-----------------------------");

    // Create a harmonic oscillator: H(q, p) = p²/2 + q²/2
    let system = SeparableHamiltonian::harmonic_oscillator();

    // Initial conditions: unit displacement, zero velocity
    let q0 = array![1.0];
    let p0 = array![0.0];

    // Integration parameters
    let t0 = 0.0;
    let period = 2.0 * PI; // Period of oscillation
    let tf = 2.0 * period; // Integrate for two periods
    let dt = 0.1;

    // Create integrator
    let integrator = StormerVerlet::new();

    // Integrate
    let result = integrator
        .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
        .unwrap();

    // Print results
    println!("  Periods: 2.0");
    println!("  Steps: {}", result.steps);
    println!("  Function evaluations: {}", result.n_evaluations);

    if let Some(energy_error) = result.energy_relative_error {
        println!("  Relative energy error: {energy_error:.2e}");
    }

    // Check final state
    let final_q = result.q.last().unwrap()[0];
    let final_p = result.p.last().unwrap()[0];

    // Exact solution at t = 2*period should be the initial state
    println!("  Final state: q = {final_q:.6}, p = {final_p:.6}");
    println!("  Expected: q = 1.000000, p = 0.000000");
    println!();
}

/// Pendulum example
#[allow(dead_code)]
fn pendulum() {
    println!("2. Pendulum System");
    println!("------------------");

    // Create a pendulum: H(q, p) = p²/2 - cos(q)
    let system = SeparableHamiltonian::pendulum();

    // Initial conditions: large amplitude oscillation
    let q0 = array![1.0]; // Initial angle (in radians)
    let p0 = array![0.0]; // Starting from rest

    // Integration parameters
    let t0 = 0.0;
    let tf = 20.0; // Integrate for a long time
    let dt = 0.05;

    // Create integrator
    let integrator = GaussLegendre4::new();

    // Integrate
    let start = Instant::now();
    let result = integrator
        .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
        .unwrap();
    let duration = start.elapsed();

    // Print results
    println!("  Integration time: {duration:.2?}");
    println!("  Steps: {}", result.steps);
    println!("  Function evaluations: {}", result.n_evaluations);

    if let Some(energy_error) = result.energy_relative_error {
        println!("  Relative energy error: {energy_error:.2e}");
    }

    // Calculate the initial and final energy explicitly
    if let Some(h_fn) = system.hamiltonian() {
        let initial_energy = h_fn(t0, &q0, &p0).unwrap();
        let final_energy = h_fn(
            result.t.last().unwrap().to_owned(),
            result.q.last().unwrap(),
            result.p.last().unwrap(),
        )
        .unwrap();

        println!("  Initial energy: {initial_energy:.6}");
        println!("  Final energy: {final_energy:.6}");
        println!(
            "  Absolute energy change: {:.2e}",
            f64::abs(final_energy - initial_energy)
        );
    }
    println!();
}

/// Kepler orbit (planetary motion) example
#[allow(dead_code)]
fn kepler_orbit() {
    println!("3. Kepler Problem (Planetary Orbit)");
    println!("----------------------------------");

    // Create a Kepler problem: H(q, p) = |p|²/2 - 1/|q|
    let system = SeparableHamiltonian::kepler_problem();

    // Initial conditions for elliptical orbit
    let q0 = array![1.0, 0.0]; // Initial position on x-axis
    let p0 = array![0.0, 1.2]; // Initial velocity in y-direction

    // Integration parameters
    let t0 = 0.0;
    let tf = 20.0; // Several orbits
    let dt = 0.02;

    // Use a composition method for higher accuracy
    let base_method = StormerVerlet::new();
    let integrator = CompositionMethod::fourth_order(base_method);

    // Integrate
    let result = integrator
        .integrate(&system, t0, tf, dt, q0.clone(), p0.clone())
        .unwrap();

    // Print results
    println!("  Steps: {}", result.steps);
    println!("  Function evaluations: {}", result.n_evaluations);

    if let Some(energy_error) = result.energy_relative_error {
        println!("  Relative energy error: {energy_error:.2e}");
    }

    // Calculate orbital properties
    let mut min_radius = f64::MAX;
    let mut max_radius: f64 = 0.0;

    for i in 0..result.q.len() {
        let q = &result.q[i];
        let radius = f64::sqrt(q[0] * q[0] + q[1] * q[1]);

        min_radius = min_radius.min(radius);
        max_radius = max_radius.max(radius);
    }

    let eccentricity = (max_radius - min_radius) / (max_radius + min_radius);

    println!("  Perihelion: {min_radius:.6}");
    println!("  Aphelion: {max_radius:.6}");
    println!("  Orbital eccentricity: {eccentricity:.6}");

    // Check if orbit is closed
    let final_q = result.q.last().unwrap();
    let final_radius = f64::sqrt(final_q[0] * final_q[0] + final_q[1] * final_q[1]);

    println!("  Final radius: {final_radius:.6}");
    println!();
}

/// Compare different symplectic methods
#[allow(dead_code)]
fn compare_methods() {
    println!("4. Comparison of Symplectic Methods");
    println!("---------------------------------");

    // Create a pendulum: H(q, p) = p²/2 - cos(q)
    let system = SeparableHamiltonian::pendulum();

    // Initial conditions: large amplitude oscillation
    let q0 = array![2.0]; // Large initial angle
    let p0 = array![0.0]; // Starting from rest

    // Integration parameters
    let t0 = 0.0;
    let tf = 100.0; // Long-time integration
    let dt = 0.1;

    // Define different methods
    let methods = [
        (
            "Störmer-Verlet",
            integrate_with_method(&StormerVerlet::new(), &system, t0, tf, dt, &q0, &p0),
        ),
        (
            "Velocity Verlet",
            integrate_with_velocity_verlet(&system, t0, tf, dt, &q0, &p0),
        ),
        (
            "Position Verlet",
            integrate_with_position_verlet(&system, t0, tf, dt, &q0, &p0),
        ),
        (
            "Gauss-Legendre 4",
            integrate_with_method(&GaussLegendre4::new(), &system, t0, tf, dt, &q0, &p0),
        ),
        (
            "Gauss-Legendre 6",
            integrate_with_method(&GaussLegendre6::new(), &system, t0, tf, dt, &q0, &p0),
        ),
        (
            "Composition 4th",
            integrate_with_composition(&system, t0, tf, dt, &q0, &p0, 4),
        ),
        (
            "Composition 6th",
            integrate_with_composition(&system, t0, tf, dt, &q0, &p0, 6),
        ),
    ];

    // Print results in a table
    println!("  Method               | Energy Error | Time (ms) | Evals");
    println!("  --------------------|--------------|-----------|---------");

    for (name, result) in methods.iter() {
        let energy_error = result.energy_relative_error.unwrap_or(f64::NAN);
        let time_ms = result.time_ms;
        let evals = result.n_evaluations;

        println!("  {name:<20} | {energy_error:.2e} | {time_ms:.2} | {evals}");
    }

    println!();
    println!("  Notes:");
    println!("  - Störmer-Verlet and Position Verlet are identical for separable Hamiltonians");
    println!(
        "  - Higher-order methods (GL6, Composition 6th) are more accurate but more expensive"
    );
    println!("  - For long-time integration, higher-order methods often justify their cost");
    println!();
}

/// Helper structure for method comparison
struct MethodResult {
    energy_relative_error: Option<f64>,
    time_ms: f64,
    n_evaluations: usize,
}

/// Helper function to integrate with a specific method
#[allow(dead_code)]
fn integrate_with_method<S: SymplecticIntegrator<f64>>(
    method: &S,
    system: &SeparableHamiltonian<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    q0: &Array1<f64>,
    p0: &Array1<f64>,
) -> MethodResult {
    let start = Instant::now();
    let result = method
        .integrate(system, t0, tf, dt, q0.clone(), p0.clone())
        .unwrap();
    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;

    MethodResult {
        energy_relative_error: result.energy_relative_error,
        time_ms,
        n_evaluations: result.n_evaluations,
    }
}

/// Helper function to integrate with velocity_verlet
#[allow(dead_code)]
fn integrate_with_velocity_verlet(
    system: &SeparableHamiltonian<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    q0: &Array1<f64>,
    p0: &Array1<f64>,
) -> MethodResult {
    let start = Instant::now();

    // Perform manual integration since velocity_verlet is a function, not a struct
    let t_span = tf - t0;
    let n_steps = (t_span / dt).ceil() as usize;
    let actual_dt = t_span / (n_steps as f64);

    let mut t = Vec::with_capacity(n_steps + 1);
    let mut q = Vec::with_capacity(n_steps + 1);
    let mut p = Vec::with_capacity(n_steps + 1);

    t.push(t0);
    q.push(q0.clone());
    p.push(p0.clone());

    let mut curr_t = t0;
    let mut curr_q = q0.clone();
    let mut curr_p = p0.clone();
    let mut n_evals = 0;

    for _ in 0..n_steps {
        let (next_q, next_p) =
            velocity_verlet(system, curr_t, &curr_q, &curr_p, actual_dt).unwrap();
        n_evals += 2;

        curr_t += actual_dt;

        t.push(curr_t);
        q.push(next_q.clone());
        p.push(next_p.clone());

        curr_q = next_q;
        curr_p = next_p;
    }

    // Calculate energy error
    let energy_error = if let Some(hamiltonian) = system.hamiltonian() {
        let initial_energy = hamiltonian(t[0], &q[0], &p[0]).unwrap();
        let final_energy = hamiltonian(t[t.len() - 1], &q[q.len() - 1], &p[p.len() - 1]).unwrap();

        if initial_energy.abs() > 1e-10 {
            Some((final_energy - initial_energy).abs() / initial_energy.abs())
        } else {
            Some((final_energy - initial_energy).abs())
        }
    } else {
        None
    };

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;

    MethodResult {
        energy_relative_error: energy_error,
        time_ms,
        n_evaluations: n_evals,
    }
}

/// Helper function to integrate with position_verlet
#[allow(dead_code)]
fn integrate_with_position_verlet(
    system: &SeparableHamiltonian<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    q0: &Array1<f64>,
    p0: &Array1<f64>,
) -> MethodResult {
    let start = Instant::now();

    // Perform manual integration
    let t_span = tf - t0;
    let n_steps = (t_span / dt).ceil() as usize;
    let actual_dt = t_span / (n_steps as f64);

    let mut t = Vec::with_capacity(n_steps + 1);
    let mut q = Vec::with_capacity(n_steps + 1);
    let mut p = Vec::with_capacity(n_steps + 1);

    t.push(t0);
    q.push(q0.clone());
    p.push(p0.clone());

    let mut curr_t = t0;
    let mut curr_q = q0.clone();
    let mut curr_p = p0.clone();
    let mut n_evals = 0;

    for _ in 0..n_steps {
        let (next_q, next_p) =
            position_verlet(system, curr_t, &curr_q, &curr_p, actual_dt).unwrap();
        n_evals += 2;

        curr_t += actual_dt;

        t.push(curr_t);
        q.push(next_q.clone());
        p.push(next_p.clone());

        curr_q = next_q;
        curr_p = next_p;
    }

    // Calculate energy error
    let energy_error = if let Some(hamiltonian) = system.hamiltonian() {
        let initial_energy = hamiltonian(t[0], &q[0], &p[0]).unwrap();
        let final_energy = hamiltonian(t[t.len() - 1], &q[q.len() - 1], &p[p.len() - 1]).unwrap();

        if initial_energy.abs() > 1e-10 {
            Some((final_energy - initial_energy).abs() / initial_energy.abs())
        } else {
            Some((final_energy - initial_energy).abs())
        }
    } else {
        None
    };

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;

    MethodResult {
        energy_relative_error: energy_error,
        time_ms,
        n_evaluations: n_evals,
    }
}

/// Helper function to integrate with composition methods
#[allow(dead_code)]
fn integrate_with_composition(
    system: &SeparableHamiltonian<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    q0: &Array1<f64>,
    p0: &Array1<f64>,
    order: usize,
) -> MethodResult {
    let base_method = StormerVerlet::new();

    let method = match order {
        4 => CompositionMethod::fourth_order(base_method),
        6 => CompositionMethod::sixth_order(base_method),
        8 => CompositionMethod::eighth_order(base_method),
        _ => CompositionMethod::fourth_order(base_method),
    };

    integrate_with_method(&method, system, t0, tf, dt, q0, p0)
}
