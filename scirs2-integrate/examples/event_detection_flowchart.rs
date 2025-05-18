//! Event Detection Flowchart Example
//!
//! This example demonstrates the event detection system in a simple flow
//! that helps users understand how the different components work together.
//! It simulates a simple system with multiple events and explains what happens at each step.

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, ODEMethod,
    ODEOptions, ODEOptionsWithEvents,
};

fn main() -> IntegrateResult<()> {
    println!("=== Event Detection Flowchart Example ===");
    println!("This example will demonstrate the event detection system flow\n");

    // Step 1: Define the ODE system
    println!("Step 1: Define an ODE system");
    println!("  Our system is a simple damped oscillator:");
    println!("  y''(t) + 0.2*y'(t) + y(t) = 0");
    println!("  Initial conditions: y(0) = 1, y'(0) = 0");
    println!("  This produces a damped oscillation that gradually decays.\n");

    // Convert to a first-order system:
    // y[0] = y
    // y[1] = y'
    // dy[0]/dt = y[1]
    // dy[1]/dt = -y[0] - 0.2*y[1]
    let f = |_t: f64, y: ArrayView1<f64>| {
        array![
            y[1],               // dy/dt = y'
            -y[0] - 0.2 * y[1]  // dy'/dt = -y - 0.2*y'
        ]
    };

    // Initial conditions
    let y0 = array![1.0, 0.0]; // y(0) = 1, y'(0) = 0

    // Step 2: Define event functions
    println!("Step 2: Define event functions");
    println!("  Event functions return a value that crosses zero when an event occurs.");
    println!("  We'll define three event functions:");
    println!("  1. Zero crossing: when y = 0");
    println!("  2. Peak detection: when y' = 0");
    println!("  3. Threshold: when y < 0.1 (terminal event)\n");

    let event_funcs = vec![
        // Event 1: Zero crossing (when y = 0)
        |_t: f64, y: ArrayView1<f64>| y[0],
        // Event 2: Peak detection (when y' = 0)
        |_t: f64, y: ArrayView1<f64>| y[1],
        // Event 3: Amplitude below threshold (when y < 0.1)
        |_t: f64, y: ArrayView1<f64>| y[0].abs() - 0.1,
    ];

    // Step 3: Configure event specifications
    println!("Step 3: Configure event specifications");
    println!("  Each event needs configuration to specify:");
    println!("  - Direction: rising, falling, or both");
    println!("  - Action: continue or stop");
    println!("  - Other parameters like precision and threshold\n");

    let event_specs = vec![
        // Event 1: Zero crossing (track both directions)
        EventSpec {
            id: "zero_crossing".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Event 2: Peak detection (track both directions)
        EventSpec {
            id: "peak".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Event 3: Terminal event (stop when amplitude falls below threshold)
        terminal_event::<f64>("threshold", EventDirection::Falling),
    ];

    // Step 4: Set up solver options
    println!("Step 4: Set up solver options");
    println!("  Solver options determine how the ODE is solved.");
    println!("  For event detection, we need to enable dense output.\n");

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true, // Required for precise event detection
            ..Default::default()
        },
        event_specs,
    );

    // Step 5: Solve the ODE with event detection
    println!("Step 5: Solve the ODE with event detection");
    println!("  The solver will integrate the ODE and detect events along the way.\n");

    let result = solve_ivp_with_events(
        f,
        [0.0, 50.0], // Time span (integration will stop early due to terminal event)
        y0,
        event_funcs,
        options,
    )?;

    // Step 6: Analyze the results
    println!("Step 6: Analyze the results");
    println!("  The solver detected events and recorded them:");
    println!(
        "  - Zero crossings: {}",
        result.events.get_count("zero_crossing")
    );
    println!("  - Peaks detected: {}", result.events.get_count("peak"));
    println!(
        "  - Threshold triggered: {}",
        result.events.get_count("threshold")
    );

    println!("\nEvent timeline:");

    // Combine all events and sort by time
    let mut all_events = Vec::new();
    for event in &result.events.events {
        all_events.push((event.time, &event.id, event.state[0], event.direction));
    }

    // Sort by time
    all_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (time, id, value, direction) in all_events {
        let dir_str = match (id.as_str(), direction) {
            ("zero_crossing", 1) => "↑ (rising)",
            ("zero_crossing", -1) => "↓ (falling)",
            ("peak", 1) => "↑ (valley)",
            ("peak", -1) => "↓ (peak)",
            ("threshold", _) => "↓ (terminal)",
            _ => "?",
        };

        println!(
            "  t = {:.4}: {:15} | y = {:.6} | direction: {}",
            time, id, value, dir_str
        );
    }

    // Report final state
    println!("\nFinal state:");
    println!("  t = {:.4}", result.base_result.t.last().unwrap());
    println!("  y = {:.6}", result.base_result.y.last().unwrap()[0]);
    println!("  y' = {:.6}", result.base_result.y.last().unwrap()[1]);
    println!("  Terminated by event: {}", result.event_termination);

    // Step 7: Visualize dense output (just explain the capability)
    println!("\nStep 7: Using dense output");
    println!("  Dense output allows evaluating the solution at any time point.");
    println!("  Example: Evaluate at intermediate points between events.");

    if let Some(ref dense) = result.dense_output {
        // Get first zero crossing time
        let zero_events = result.events.get_events("zero_crossing");
        if !zero_events.is_empty() {
            let first_zero = zero_events[0].time;

            // Evaluate at 5 equally-spaced points before the first zero crossing
            let t_start = 0.0;
            let delta = (first_zero - t_start) / 6.0;

            println!("\n  Values approaching first zero crossing:");
            for i in 1..=5 {
                let t = t_start + i as f64 * delta;
                let y = dense.evaluate(t)?;
                println!("    t = {:.4}, y = {:.6}, y' = {:.6}", t, y[0], y[1]);
            }
        }
    }

    println!("\nEvent detection flowchart complete!");
    println!("This demonstrates how event detection is integrated with the ODE solver.");

    Ok(())
}
