//! Bouncing Ball Example with Event Detection
//!
//! This example demonstrates the use of event detection with ODE solvers
//! to simulate a bouncing ball under gravity.
//!
//! The model consists of two state variables:
//! - y[0] = height of the ball
//! - y[1] = velocity of the ball (positive upward)
//!
//! Event detection is used to identify when the ball hits the ground (y[0] = 0)
//! with a downward velocity. At each impact, the velocity is reversed and reduced
//! by a coefficient of restitution to simulate energy loss.

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::{IntegrateError, IntegrateResult};
use scirs2_integrate::ode::{
    solve_ivp_with_events, EventAction, EventDirection, EventSpec, ODEMethod, ODEOptions,
    ODEOptionsWithEvents, ODEResultWithEvents,
};
use std::fs::File;
use std::io::Write;

fn main() -> IntegrateResult<()> {
    // Physical parameters
    let g = 9.81; // acceleration due to gravity, m/s^2
    let coef_restitution = 0.8; // coefficient of restitution (energy loss on bounce)

    // Initial conditions
    let height_0 = 10.0; // initial height, m
    let velocity_0 = 0.0; // initial velocity, m/s
    let y0 = array![height_0, velocity_0];

    // Time span
    let t_span = [0.0, 15.0]; // integrate from 0 to 15 seconds

    // ODE function: dy/dt = [y[1], -g]
    // Where:
    // - y[0] is the height
    // - y[1] is the velocity
    let f = move |_t: f64, y: ArrayView1<f64>| array![y[1], -g];

    // Event function: detect when ball hits the ground (y[0] = 0)
    // The event function must return a value that crosses zero when the event occurs
    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // height = 0 is the event
    ];

    // Event specification: detect when the ball hits the ground
    let event_specs = vec![EventSpec {
        id: "ground_impact".to_string(),
        direction: EventDirection::Falling, // Only detect when height goes from positive to zero
        action: EventAction::Continue,      // Don't stop the simulation on impact
        threshold: 1e-8,
        max_count: None,    // Allow unlimited bounces
        precise_time: true, // Accurately locate the impact time
    }];

    // Solver options
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

    println!("Simulating bouncing ball with event detection...");

    // First solution: without handling bounces, just detect the impacts
    let result =
        solve_ivp_with_events(f, t_span, y0.clone(), event_funcs.clone(), options.clone())?;

    println!("Without bounce handling:");
    println!(
        "  Number of ground impacts: {}",
        result.events.get_count("ground_impact")
    );
    print_trajectory_summary(&result);

    // Second solution: with bounce handling
    // We need to create a modified solver that applies the bounce condition at each event
    let mut traj_times = Vec::<f64>::new();
    let mut traj_heights = Vec::<f64>::new();
    let mut traj_velocities = Vec::<f64>::new();
    let mut bounce_times = Vec::<f64>::new();
    let mut bounce_velocities = Vec::<f64>::new();

    // Start with initial conditions
    let mut t = t_span[0];
    let mut y = y0.clone();
    traj_times.push(t);
    traj_heights.push(y[0]);
    traj_velocities.push(y[1]);

    // Integrate in segments, applying bounce condition at each impact
    while t < t_span[1] {
        // Integrate until next impact or end of time span
        let segment_result = solve_ivp_with_events(
            f,
            [t, t_span[1]],
            y.clone(),
            event_funcs.clone(),
            options.clone(),
        )?;

        // Append trajectory data (excluding the first point to avoid duplicates)
        for i in 1..segment_result.base_result.t.len() {
            traj_times.push(segment_result.base_result.t[i]);
            traj_heights.push(segment_result.base_result.y[i][0]);
            traj_velocities.push(segment_result.base_result.y[i][1]);
        }

        // Check if an impact occurred
        if !segment_result.events.events.is_empty() {
            // Get the last event (impact)
            let impact = segment_result.events.events.last().unwrap();
            t = impact.time;

            // Apply bounce condition: reverse velocity and apply coefficient of restitution
            // Note: velocity should be negative just before impact
            let velocity_before = impact.state[1];
            if velocity_before >= 0.0 {
                println!(
                    "Warning: Velocity before impact should be negative, got {}",
                    velocity_before
                );
            }

            // Calculate velocity after bounce (reverse and reduce by coefficient of restitution)
            let velocity_after = -velocity_before * coef_restitution;

            // Record the bounce
            bounce_times.push(t);
            bounce_velocities.push(velocity_after);

            // Update state for next integration segment
            y = array![0.0, velocity_after];

            println!(
                "Bounce at t = {:.4}: velocity before = {:.4}, velocity after = {:.4}",
                t, velocity_before, velocity_after
            );
        } else {
            // No more impacts, we're done
            break;
        }

        // If velocity after bounce is too small, stop simulation
        if y[1].abs() < 1e-3 {
            println!(
                "Ball essentially stopped at t = {:.4} with velocity = {:.6}",
                t, y[1]
            );
            break;
        }
    }

    println!("\nWith bounce handling:");
    println!("  Number of bounces: {}", bounce_times.len());
    println!(
        "  Total simulation time: {:.4}",
        traj_times.last().unwrap_or(&t_span[1])
    );

    // Write results to CSV file for plotting
    save_results_to_csv(&traj_times, &traj_heights, &traj_velocities, &bounce_times)?;

    println!("\nResults saved to 'bouncing_ball_results.csv'");
    println!("To visualize the results, you can use any plotting tool or library.");

    Ok(())
}

/// Print a summary of the trajectory
fn print_trajectory_summary(result: &ODEResultWithEvents<f64>) {
    println!("  Integration results:");
    println!("    Number of time steps: {}", result.base_result.t.len());
    println!("    Start time: {:?}", result.base_result.t.first());
    println!("    End time: {:?}", result.base_result.t.last());

    println!("  First few time points:");
    let n_show = std::cmp::min(5, result.base_result.t.len());
    for i in 0..n_show {
        println!(
            "    t = {:?}, height = {:?}, velocity = {:?}",
            result.base_result.t[i], result.base_result.y[i][0], result.base_result.y[i][1]
        );
    }

    if let Some(first_event) = result.events.get_events("ground_impact").first() {
        println!(
            "  First impact at time = {:?}, velocity = {:?}",
            first_event.time, first_event.state[1]
        );
    }
}

/// Save the results to a CSV file for plotting
fn save_results_to_csv(
    times: &[f64],
    heights: &[f64],
    velocities: &[f64],
    bounce_times: &[f64],
) -> IntegrateResult<()> {
    let mut file = File::create("bouncing_ball_results.csv")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to create file: {}", e)))?;

    // Write header
    writeln!(file, "time,height,velocity,is_bounce")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write header: {}", e)))?;

    // Write data
    for i in 0..times.len() {
        let is_bounce = bounce_times.contains(&times[i]);
        writeln!(
            file,
            "{},{},{},{}",
            times[i], heights[i], velocities[i], is_bounce as i32
        )
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write data: {}", e)))?;
    }

    Ok(())
}
