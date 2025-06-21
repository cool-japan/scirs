//! Chemical Reaction System with Event Detection
//!
//! This example demonstrates using event detection with ODE solvers to simulate
//! a chemical reaction system and detect when concentrations reach specific thresholds.
//!
//! The model is the Brusselator system, a theoretical model for autocatalytic reactions:
//!   A → X
//!   2X + Y → 3X
//!   B + X → Y + D
//!   X → E
//!
//! Where:
//! - y[0] = concentration of species X
//! - y[1] = concentration of species Y
//!
//! Parameters A and B control the behavior of the system. For B > 1 + A^2, the system
//! exhibits oscillatory behavior, which we'll detect using events.

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::{IntegrateError, IntegrateResult};
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, ODEMethod,
    ODEOptions, ODEOptionsWithEvents,
};
use std::fs::File;
use std::io::Write;

fn main() -> IntegrateResult<()> {
    // Brusselator model parameters
    let a = 1.0; // Parameter A
    let b = 3.0; // Parameter B (> 1 + A^2 for oscillatory behavior)

    // Initial conditions
    let x0 = 1.0; // Initial concentration of X
    let y0 = 1.0; // Initial concentration of Y
    let initial_state = array![x0, y0];

    // Time span
    let t_span = [0.0, 100.0]; // Integrate for a long time to observe oscillations

    // ODE function for Brusselator model:
    // dx/dt = a + x^2*y - b*x - x
    // dy/dt = b*x - x^2*y
    let f = move |_t: f64, state: ArrayView1<f64>| {
        let x = state[0];
        let y = state[1];

        array![
            a + x * x * y - b * x - x, // dx/dt
            b * x - x * x * y          // dy/dt
        ]
    };

    // Define event functions
    type EventFunc = Box<dyn Fn(f64, ArrayView1<f64>) -> f64>;
    let event_funcs: Vec<EventFunc> = vec![
        // Event 1: dx/dt = 0 (X maximum or minimum)
        Box::new(move |_t: f64, state: ArrayView1<f64>| -> f64 {
            let x = state[0];
            let y = state[1];
            a + x * x * y - b * x - x // dx/dt
        }),
        // Event 2: dy/dt = 0 (Y maximum or minimum)
        Box::new(move |_t: f64, state: ArrayView1<f64>| -> f64 {
            let x = state[0];
            let y = state[1];
            b * x - x * x * y // dy/dt
        }),
        // Event 3: X concentration crosses threshold of 2.0
        Box::new(move |_t: f64, state: ArrayView1<f64>| -> f64 { state[0] - 2.0 }),
        // Event 4: Time limit reached (t = 50)
        Box::new(move |t: f64, _state: ArrayView1<f64>| -> f64 { t - 50.0 }),
    ];

    // Event specifications
    let event_specs = vec![
        // X maximum events
        EventSpec {
            id: "x_maximum".to_string(),
            direction: EventDirection::Falling, // dx/dt changes from positive to negative
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Y maximum events
        EventSpec {
            id: "y_maximum".to_string(),
            direction: EventDirection::Falling, // dy/dt changes from positive to negative
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // X threshold crossing events
        EventSpec {
            id: "x_threshold".to_string(),
            direction: EventDirection::Both, // Detect both upward and downward crossings
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Terminal event: stop after a certain time
        terminal_event::<f64>("end_time", EventDirection::Rising),
    ];

    // Solver options
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::EnhancedLSODA, // Use LSODA for potentially stiff system
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true, // Needed for precise event detection
            ..Default::default()
        },
        event_specs,
    );

    println!("Simulating Brusselator chemical reaction system with event detection...");
    println!("Parameters: A = {}, B = {}", a, b);
    println!("Oscillatory behavior expected: {}", b > 1.0 + a * a);

    // Solve the ODE with event detection
    let result = solve_ivp_with_events(f, t_span, initial_state, event_funcs, options)?;

    // Print results
    println!("\nSimulation results:");
    println!("  Number of time steps: {}", result.base_result.t.len());
    println!(
        "  Integration terminated at t = {:.4}",
        result.base_result.t.last().unwrap()
    );
    println!(
        "  Final state: X = {:.6}, Y = {:.6}",
        result.base_result.y.last().unwrap()[0],
        result.base_result.y.last().unwrap()[1]
    );

    // Analyze the oscillation period
    let x_maxima = result.events.get_events("x_maximum");
    if !x_maxima.is_empty() {
        println!("\nOscillation Analysis:");
        println!("  Number of X maxima detected: {}", x_maxima.len());

        // Calculate periods between consecutive maxima
        if x_maxima.len() > 1 {
            let mut periods = Vec::new();
            for i in 1..x_maxima.len() {
                periods.push(x_maxima[i].time - x_maxima[i - 1].time);
            }

            let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;

            println!("  Average oscillation period: {:.4} time units", avg_period);
            println!(
                "  Oscillation frequency: {:.4} cycles/time unit",
                1.0 / avg_period
            );

            // Analyze amplitude variation
            let mut amplitudes = Vec::new();
            for event in x_maxima {
                amplitudes.push(event.state[0]);
            }

            println!(
                "  X amplitude range: {:.4} to {:.4}",
                amplitudes.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                amplitudes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );
        }
    }

    // Analyze threshold crossings
    let threshold_events = result.events.get_events("x_threshold");
    println!("\nThreshold Analysis:");
    println!(
        "  Number of times X crossed threshold (2.0): {}",
        threshold_events.len()
    );

    if !threshold_events.is_empty() {
        println!(
            "  First threshold crossing at t = {:.4}",
            threshold_events[0].time
        );

        // Check if X spends more time above or below threshold
        let mut time_above = 0.0;
        let mut time_below = 0.0;
        let mut above = threshold_events[0].direction > 0; // First crossing direction

        for i in 0..threshold_events.len() {
            let current_time = threshold_events[i].time;
            let next_time = if i + 1 < threshold_events.len() {
                threshold_events[i + 1].time
            } else {
                *result.base_result.t.last().unwrap()
            };

            let interval = next_time - current_time;

            if above {
                time_above += interval;
            } else {
                time_below += interval;
            }

            above = !above; // Flip state after each crossing
        }

        let total_time =
            *result.base_result.t.last().unwrap() - *result.base_result.t.first().unwrap();
        println!(
            "  Time spent above threshold: {:.1}% ({:.4} time units)",
            100.0 * time_above / total_time,
            time_above
        );
        println!(
            "  Time spent below threshold: {:.1}% ({:.4} time units)",
            100.0 * time_below / total_time,
            time_below
        );
    }

    // Write results to a CSV file for plotting
    save_results_to_csv(&result)?;

    println!("\nResults saved to 'brusselator_results.csv'");

    Ok(())
}

/// Save the results to a CSV file for plotting
fn save_results_to_csv(
    result: &scirs2_integrate::ode::ODEResultWithEvents<f64>,
) -> IntegrateResult<()> {
    let mut file = File::create("brusselator_results.csv")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to create file: {}", e)))?;

    // Write header
    writeln!(file, "time,x,y,x_maximum,y_maximum,x_threshold")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write header: {}", e)))?;

    // Get event times for each event type
    let x_max_times: Vec<f64> = result
        .events
        .get_events("x_maximum")
        .iter()
        .map(|e| e.time)
        .collect();

    let y_max_times: Vec<f64> = result
        .events
        .get_events("y_maximum")
        .iter()
        .map(|e| e.time)
        .collect();

    let x_thresh_times: Vec<f64> = result
        .events
        .get_events("x_threshold")
        .iter()
        .map(|e| e.time)
        .collect();

    // Write data points and mark events
    for i in 0..result.base_result.t.len() {
        let t = result.base_result.t[i];
        let x = result.base_result.y[i][0];
        let y = result.base_result.y[i][1];

        // Check if any events occurred at this time (with small tolerance)
        let x_max = x_max_times.iter().any(|&et| (t - et).abs() < 1e-6);
        let y_max = y_max_times.iter().any(|&et| (t - et).abs() < 1e-6);
        let x_thresh = x_thresh_times.iter().any(|&et| (t - et).abs() < 1e-6);

        writeln!(
            file,
            "{},{},{},{},{},{}",
            t, x, y, x_max as i32, y_max as i32, x_thresh as i32
        )
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write data: {}", e)))?;
    }

    Ok(())
}
