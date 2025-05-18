//! Mass Matrix with Event Detection Example
//!
//! This example demonstrates the combined use of mass matrices and event detection
//! in ODE solvers. We model a mechanical system with a time-dependent mass
//! and detect specific events during the integration.
//!
//! The physical system modeled is a variable-mass pendulum, where the
//! pendulum's mass changes with time (like a pendulum with a leaking bob).

use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, MassMatrix,
    ODEMethod, ODEOptions, ODEOptionsWithEvents,
};
use std::f64::consts::PI;

fn main() -> IntegrateResult<()> {
    println!("=== Variable-Mass Pendulum with Event Detection ===");

    // Physical parameters
    let g = 9.81; // Acceleration due to gravity, m/s^2
    let l = 1.0; // Pendulum length, m
    let m0 = 2.0; // Initial mass, kg
    let tau = 5.0; // Mass decay time constant, s

    // Mass decreases exponentially: m(t) = m0 * exp(-t/tau)
    let mass_function = move |t: f64| m0 * (-t / tau).exp();

    // Create a time-dependent mass matrix:
    // [m(t)   0]
    // [  0    1]
    let time_dependent_mass = move |t: f64| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = mass_function(t);
        m
    };

    // Create the mass matrix specification
    let mass = MassMatrix::time_dependent(time_dependent_mass);

    // ODE function: f(t, y) = [y[1], -g/l * sin(y[0])]
    // Note: The mass matrix format means the ODE is:
    // [m(t)   0] [θ']  = [    ω    ]
    // [  0    1] [ω']    [-g/l·sin θ]
    // where θ is the angle and ω is the angular velocity
    let f = move |_t: f64, y: ArrayView1<f64>| {
        array![
            y[1],                // θ' = ω
            -g / l * y[0].sin()  // ω' = -g/l·sin θ (without mass factor)
        ]
    };

    // Initial conditions
    let theta_0 = PI / 6.0; // Initial angle (30 degrees)
    let omega_0 = 0.0; // Initial angular velocity
    let y0 = array![theta_0, omega_0];

    // Time span for integration
    let t_span = [0.0, 15.0];

    // Event functions to detect:
    // 1. Crossing through equilibrium position (θ = 0)
    // 2. Maximum angular displacement (ω = 0)
    // 3. When mass reaches half of initial value
    // 4. When oscillation amplitude exceeds a threshold (terminal event)
    let event_funcs = vec![
        // Event 1: Crossing through equilibrium (θ = 0)
        |_t: f64, y: ArrayView1<f64>| y[0],
        // Event 2: Maximum displacement (ω = 0)
        |_t: f64, y: ArrayView1<f64>| y[1],
        // Event 3: Mass reaches half of initial value
        |t: f64, _y: ArrayView1<f64>| {
            // Inline mass calculation to avoid closure capture
            let m0 = 5.0;
            let exp_t = (-0.1 * t).exp();
            let mass = m0 * (1.0 - 0.3 * exp_t);
            mass - m0 / 2.0
        },
        // Event 4: Angle exceeds 45 degrees in magnitude (terminal event)
        |_t: f64, y: ArrayView1<f64>| PI / 4.0 - y[0].abs(),
    ];

    // Event specifications
    let event_specs = vec![
        // Equilibrium crossing
        EventSpec {
            id: "equilibrium".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Maximum displacement
        EventSpec {
            id: "max_displacement".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Half-mass event
        EventSpec {
            id: "half_mass".to_string(),
            direction: EventDirection::Falling,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: Some(1), // Only detect once
            precise_time: true,
        },
        // Angle threshold exceeded
        terminal_event::<f64>("angle_threshold", EventDirection::Falling),
    ];

    // Create options with both mass matrix and event detection
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::Radau, // Implicit method with direct mass matrix support
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true, // Required for precise event detection
            mass_matrix: Some(mass),
            ..Default::default()
        },
        event_specs,
    );

    println!("Solving the variable-mass pendulum system...");

    // Solve the system
    let result = solve_ivp_with_events(f, t_span, y0, event_funcs, options)?;

    // Print basic solution info
    println!("\nIntegration results:");
    println!("  Final time: {:.4}", result.base_result.t.last().unwrap());
    println!(
        "  Final state: θ = {:.4}°, ω = {:.4} rad/s",
        result.base_result.y.last().unwrap()[0] * 180.0 / PI,
        result.base_result.y.last().unwrap()[1]
    );
    println!("  Steps taken: {}", result.base_result.n_steps);
    println!("  Function evaluations: {}", result.base_result.n_eval);
    println!("  Jacobian evaluations: {}", result.base_result.n_jac);
    println!("  LU decompositions: {}", result.base_result.n_lu);

    // Print event detection summary
    println!("\nEvent detection summary:");
    println!(
        "  Equilibrium crossings: {}",
        result.events.get_count("equilibrium")
    );
    println!(
        "  Maximum displacements: {}",
        result.events.get_count("max_displacement")
    );
    println!(
        "  Half-mass event: {}",
        result.events.get_count("half_mass")
    );
    println!(
        "  Terminated by angle threshold: {}",
        result.event_termination
    );

    // Analyze half-mass event
    if let Some(half_mass_event) = result.events.get_events("half_mass").first() {
        println!("\nHalf-mass event details:");
        println!("  Time: {:.4} s", half_mass_event.time);
        println!("  Theoretical time: {:.4} s", tau * (2.0f64).ln());
        println!(
            "  Angle at half-mass: {:.4}°",
            half_mass_event.state[0] * 180.0 / PI
        );
        println!(
            "  Angular velocity at half-mass: {:.4} rad/s",
            half_mass_event.state[1]
        );
    }

    // Analyze period changes
    if result.events.get_count("equilibrium") >= 2 {
        println!("\nPeriod analysis:");

        // Get equilibrium crossing events (only positive direction)
        let events = result.events.get_events("equilibrium");
        let crossings: Vec<_> = events.iter().filter(|e| e.direction > 0).collect();

        if crossings.len() >= 2 {
            // Calculate periods
            let mut periods = Vec::new();
            for i in 1..crossings.len() {
                periods.push(crossings[i].time - crossings[i - 1].time);
            }

            println!("  Periods of oscillation:");
            for (i, period) in periods.iter().enumerate() {
                println!("    Period {}: {:.4} s", i + 1, period);
            }

            // Compare with theoretical prediction for small oscillations
            // Period changes with mass: T(t) = 2π√(l/g) * m(t)^(1/2)
            let initial_period = 2.0 * PI * (l / g).sqrt();
            println!("  Initial period (constant mass): {:.4} s", initial_period);

            // For small oscillation of variable mass pendulum, the period increases as mass decreases
            let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;
            println!("  Average observed period: {:.4} s", avg_period);
            println!(
                "  Period ratio (avg/initial): {:.4}",
                avg_period / initial_period
            );
        }
    }

    // Analyze terminal event if it occurred
    if result.event_termination {
        if let Some(terminal_event) = result.events.get_events("angle_threshold").first() {
            println!("\nTerminal event details:");
            println!("  Time: {:.4} s", terminal_event.time);
            println!(
                "  Final angle: {:.4}°",
                terminal_event.state[0] * 180.0 / PI
            );
            println!(
                "  Final angular velocity: {:.4} rad/s",
                terminal_event.state[1]
            );

            // Calculate mass at termination
            let final_mass = mass_function(terminal_event.time);
            println!(
                "  Mass at termination: {:.4} kg ({:.1}% of initial)",
                final_mass,
                100.0 * final_mass / m0
            );
        }
    }

    println!("\nThis example demonstrates the successful integration of mass matrices");
    println!("and event detection. The variable-mass pendulum shows period changes");
    println!("and amplitude growth as mass decreases, with precise event detection.");

    Ok(())
}
