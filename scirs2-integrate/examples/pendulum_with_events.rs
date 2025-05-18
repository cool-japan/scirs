//! Pendulum Example with Event Detection
//!
//! This example demonstrates the use of event detection with ODE solvers
//! to simulate a simple pendulum and detect specific conditions during the integration.
//!
//! The model consists of two state variables:
//! - y[0] = angular position of the pendulum (theta)
//! - y[1] = angular velocity of the pendulum (dtheta/dt)
//!
//! We use event detection to:
//! 1. Detect when the pendulum crosses the downward equilibrium position (theta = 0)
//! 2. Detect when the pendulum reaches maximum amplitude (dtheta/dt = 0)
//! 3. Detect when the pendulum reaches a specific angle (terminal condition)

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, ODEMethod,
    ODEOptions, ODEOptionsWithEvents,
};
use std::f64::consts::PI;

fn main() -> IntegrateResult<()> {
    // Pendulum parameters
    let g = 9.81; // acceleration due to gravity (m/s²)
    let l = 1.0; // pendulum length (m)
    let omega = ((g / l) as f64).sqrt(); // natural frequency

    // Initial conditions
    let theta_0 = PI / 6.0; // initial angle (30 degrees, in radians)
    let omega_0 = 0.0; // initial angular velocity
    let y0 = array![theta_0, omega_0];

    // Time span (enough for several oscillations)
    let t_span = [0.0, 10.0];

    // ODE function for a simple pendulum: dy/dt = [y[1], -omega²*sin(y[0])]
    let f = move |_t: f64, y: ArrayView1<f64>| {
        array![
            y[1],
            -omega * omega * y[0].sin() // small angle approximation: sin(theta) ≈ theta
        ]
    };

    // Event functions to detect various conditions:
    let event_funcs = vec![
        // Event 1: Detect crossing through the equilibrium position (theta = 0)
        |_t: f64, y: ArrayView1<f64>| y[0],
        // Event 2: Detect when pendulum reaches maximum amplitude (dtheta/dt = 0)
        |_t: f64, y: ArrayView1<f64>| y[1],
        // Event 3: Detect when pendulum reaches a specific angle (theta = -20 degrees)
        |_t: f64, y: ArrayView1<f64>| y[0] + PI / 9.0, // +20 degrees
    ];

    // Event specifications
    let event_specs = vec![
        // Equilibrium crossing events
        EventSpec {
            id: "equilibrium_crossing".to_string(),
            direction: EventDirection::Both, // Detect both upward and downward crossings
            action: EventAction::Continue,   // Continue integration
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Maximum amplitude events
        EventSpec {
            id: "max_amplitude".to_string(),
            direction: EventDirection::Both, // Both when velocity becomes zero
            action: EventAction::Continue,   // Continue integration
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Terminal event: stop when pendulum reaches -20 degrees
        terminal_event::<f64>("angle_reached", EventDirection::Falling),
    ];

    // Solver options
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true, // Needed for precise event detection
            ..Default::default()
        },
        event_specs,
    );

    println!("Simulating pendulum with event detection...");

    // Solve the ODE with event detection
    let result = solve_ivp_with_events(f, t_span, y0, event_funcs, options)?;

    // Print results
    println!("\nSimulation results:");
    println!(
        "  Integration terminated at t = {:.6}",
        result.base_result.t.last().unwrap()
    );
    println!(
        "  Final state: theta = {:.6} rad, omega = {:.6} rad/s",
        result.base_result.y.last().unwrap()[0],
        result.base_result.y.last().unwrap()[1]
    );

    // Print number of events detected
    println!("\nEvent detection summary:");
    println!(
        "  Equilibrium crossings detected: {}",
        result.events.get_count("equilibrium_crossing")
    );
    println!(
        "  Maximum amplitude points detected: {}",
        result.events.get_count("max_amplitude")
    );
    println!("  Terminal event triggered: {}", result.event_termination);

    // Analyze the period of oscillation
    if result.events.get_count("equilibrium_crossing") >= 2 {
        let crossings = result.events.get_events("equilibrium_crossing");
        let mut periods = Vec::new();

        // Calculate periods from consecutive upward crossings (positive direction)
        let mut last_time = None;
        for event in crossings {
            if event.direction > 0 {
                // Rising direction (positive crossing)
                if let Some(prev_time) = last_time {
                    periods.push(event.time - prev_time);
                }
                last_time = Some(event.time);
            }
        }

        if !periods.is_empty() {
            let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;

            println!("\nPeriod analysis:");
            println!("  Calculated periods: {:?}", periods);
            println!("  Average observed period: {:.6} s", avg_period);

            // Compare with theoretical period for small oscillations: T = 2π * sqrt(l/g)
            let theoretical_period = 2.0 * PI * ((l / g) as f64).sqrt();
            println!(
                "  Theoretical period (small angle): {:.6} s",
                theoretical_period
            );
            println!(
                "  Difference: {:.2}%",
                100.0 * (avg_period - theoretical_period).abs() / theoretical_period
            );
        }
    }

    // Print the event details
    println!("\nEquilibrium crossing events:");
    for (i, event) in result
        .events
        .get_events("equilibrium_crossing")
        .iter()
        .enumerate()
    {
        if i < 5 {
            // Just show the first few
            println!(
                "  t = {:.6}, theta = {:.6}, direction = {}",
                event.time, event.state[0], event.direction
            );
        }
    }

    println!("\nMaximum amplitude events:");
    for (i, event) in result.events.get_events("max_amplitude").iter().enumerate() {
        if i < 5 {
            // Just show the first few
            println!(
                "  t = {:.6}, theta = {:.6} rad ({:.1}°)",
                event.time,
                event.state[0],
                event.state[0] * 180.0 / PI
            );
        }
    }

    println!("\nTerminal event:");
    if let Some(terminal) = result.events.get_events("angle_reached").first() {
        println!(
            "  Reached target angle at t = {:.6}, theta = {:.6} rad ({:.1}°)",
            terminal.time,
            terminal.state[0],
            terminal.state[0] * 180.0 / PI
        );
    } else {
        println!("  Terminal event not triggered");
    }

    Ok(())
}
