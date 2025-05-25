//! Hybrid System Example with Event Detection
//!
//! This example demonstrates using event detection to simulate a hybrid system
//! with discontinuous dynamics. The model represents a thermostat controlling
//! the temperature of a room.
//!
//! The system has two modes:
//! 1. Heater ON - Temperature increases exponentially toward a maximum temperature
//! 2. Heater OFF - Temperature decreases exponentially toward the ambient temperature
//!
//! The control logic switches between these modes based on temperature thresholds.

// FIXME: This example has compilation errors - closure type mismatches and IO errors
#![allow(dead_code)]

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::{IntegrateError, IntegrateResult};
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, ODEMethod,
    ODEOptions, ODEOptionsWithEvents,
};

type EventFunc = Box<dyn Fn(f64, ArrayView1<f64>) -> f64>;
use std::fs::File;
use std::io::Write;

// System parameters
const AMBIENT_TEMP: f64 = 15.0; // Ambient temperature (°C)
const MAX_TEMP: f64 = 30.0; // Maximum temperature with heater on (°C)
const HEATER_ON_THRESHOLD: f64 = 18.0; // Turn heater on below this temperature (°C)
const HEATER_OFF_THRESHOLD: f64 = 22.0; // Turn heater off above this temperature (°C)
const COOLING_RATE: f64 = 0.1; // Rate of cooling
const HEATING_RATE: f64 = 0.2; // Rate of heating

// Mode definitions
#[derive(Debug, Clone, Copy, PartialEq)]
enum HeaterMode {
    On,
    Off,
}

fn main() -> IntegrateResult<()> {
    println!("=== Hybrid System (Thermostat) with Event Detection ===");

    // Initial conditions
    let initial_temp = 17.0; // °C
    let initial_mode = HeaterMode::Off;

    // Initial state vector: [temperature, mode (0 for OFF, 1 for ON)]
    let y0 = array![initial_temp, initial_mode as u8 as f64];

    // Time span for simulation (6 hours)
    let t_span = [0.0, 6.0]; // Hours

    // Record system evolution
    let mut times = Vec::new();
    let mut temperatures = Vec::new();
    let mut modes = Vec::new();
    let mut event_times = Vec::new();
    let mut event_types = Vec::new();

    // Add initial state
    times.push(t_span[0]);
    temperatures.push(y0[0]);
    modes.push(if y0[1] < 0.5 {
        HeaterMode::Off
    } else {
        HeaterMode::On
    });

    // Start simulation loop
    let mut current_time = t_span[0];
    let mut current_state = y0.clone();

    println!(
        "Starting simulation with temperature = {:.1}°C, heater = {:?}",
        current_state[0],
        if current_state[1] < 0.5 {
            HeaterMode::Off
        } else {
            HeaterMode::On
        }
    );

    // Integrate in segments, applying mode switching at events
    while current_time < t_span[1] {
        // Current heater mode
        let current_mode = if current_state[1] < 0.5 {
            HeaterMode::Off
        } else {
            HeaterMode::On
        };

        // Define ODE function based on current mode
        let f = move |_t: f64, y: ArrayView1<f64>| {
            let temp = y[0];
            let mode = if y[1] < 0.5 {
                HeaterMode::Off
            } else {
                HeaterMode::On
            };

            let dtemp_dt = match mode {
                HeaterMode::Off => COOLING_RATE * (AMBIENT_TEMP - temp),
                HeaterMode::On => HEATING_RATE * (MAX_TEMP - temp),
            };

            // Mode doesn't change during integration
            array![dtemp_dt, 0.0]
        };

        // Define event functions based on current mode
        let t_end = t_span[1]; // Capture the value to avoid lifetime issues
        let (event_funcs, event_specs) = match current_mode {
            HeaterMode::Off => {
                // When heater is OFF, detect if temperature drops below HEATER_ON_THRESHOLD
                let event_funcs: Vec<EventFunc> = vec![
                    // Event 1: Temperature drops below heater-on threshold
                    Box::new(|_t: f64, y: ArrayView1<f64>| y[0] - HEATER_ON_THRESHOLD),
                    // Event 2: End of simulation
                    Box::new(move |t: f64, _y: ArrayView1<f64>| t - t_end),
                ];

                let event_specs = vec![
                    // Switch on heater when temperature drops below threshold
                    EventSpec {
                        id: "heater_on".to_string(),
                        direction: EventDirection::Falling, // Detect falling below threshold
                        action: EventAction::Stop,          // Stop to update mode
                        threshold: 1e-8,
                        max_count: Some(1), // Only need first crossing
                        precise_time: true,
                    },
                    // End of simulation
                    terminal_event::<f64>("end_time", EventDirection::Rising),
                ];

                (event_funcs, event_specs)
            }
            HeaterMode::On => {
                // When heater is ON, detect if temperature rises above HEATER_OFF_THRESHOLD
                let event_funcs: Vec<EventFunc> = vec![
                    // Event 1: Temperature rises above heater-off threshold
                    Box::new(|_t: f64, y: ArrayView1<f64>| y[0] - HEATER_OFF_THRESHOLD),
                    // Event 2: End of simulation
                    Box::new(move |t: f64, _y: ArrayView1<f64>| t - t_end),
                ];

                let event_specs = vec![
                    // Switch off heater when temperature rises above threshold
                    EventSpec {
                        id: "heater_off".to_string(),
                        direction: EventDirection::Rising, // Detect rising above threshold
                        action: EventAction::Stop,         // Stop to update mode
                        threshold: 1e-8,
                        max_count: Some(1), // Only need first crossing
                        precise_time: true,
                    },
                    // End of simulation
                    terminal_event::<f64>("end_time", EventDirection::Rising),
                ];

                (event_funcs, event_specs)
            }
        };

        // Create ODE options with events
        let options = ODEOptionsWithEvents::new(
            ODEOptions {
                method: ODEMethod::RK45,
                rtol: 1e-6,
                atol: 1e-8,
                dense_output: true,
                ..Default::default()
            },
            event_specs,
        );

        // Integrate until next event
        let result = solve_ivp_with_events(
            f,
            [current_time, t_span[1]],
            current_state.clone(),
            event_funcs,
            options,
        )?;

        // Extract results
        for i in 1..result.base_result.t.len() {
            times.push(result.base_result.t[i]);
            temperatures.push(result.base_result.y[i][0]);
            modes.push(if result.base_result.y[i][1] < 0.5 {
                HeaterMode::Off
            } else {
                HeaterMode::On
            });
        }

        // Update current time and state
        current_time = *result.base_result.t.last().unwrap();
        current_state = result.base_result.y.last().unwrap().clone();

        // If a mode-switching event occurred
        if result.events.get_count("heater_on") > 0 || result.events.get_count("heater_off") > 0 {
            // Get the event
            let event_id = if result.events.get_count("heater_on") > 0 {
                "heater_on"
            } else {
                "heater_off"
            };

            let event = result.events.get_events(event_id)[0];

            // Record the event
            event_times.push(event.time);
            event_types.push(event_id.to_string());

            // Apply mode switch
            match event_id {
                "heater_on" => {
                    println!(
                        "t = {:.3}h: Heater turned ON at {:.2}°C",
                        event.time, event.state[0]
                    );
                    current_state[1] = 1.0; // Switch to ON mode
                }
                "heater_off" => {
                    println!(
                        "t = {:.3}h: Heater turned OFF at {:.2}°C",
                        event.time, event.state[0]
                    );
                    current_state[1] = 0.0; // Switch to OFF mode
                }
                _ => {}
            }
        } else if result.event_termination {
            // End of simulation reached
            println!(
                "t = {:.3}h: End of simulation at {:.2}°C",
                current_time, current_state[0]
            );
            break;
        }
    }

    // Print summary statistics
    println!("\nSimulation Summary:");
    println!(
        "  Duration: {:.2} hours",
        times.last().unwrap() - times.first().unwrap()
    );
    println!(
        "  Initial temperature: {:.2}°C",
        temperatures.first().unwrap()
    );
    println!("  Final temperature: {:.2}°C", temperatures.last().unwrap());
    println!("  Mode switches: {}", event_times.len());

    // Calculate time in each mode
    let mut time_heater_on = 0.0;
    let mut time_heater_off = 0.0;
    let mut prev_time = times[0];
    let mut prev_mode = modes[0];

    for (time, mode) in times.iter().zip(modes.iter()).skip(1) {
        let dt = time - prev_time;
        match prev_mode {
            HeaterMode::On => time_heater_on += dt,
            HeaterMode::Off => time_heater_off += dt,
        }
        prev_time = *time;
        prev_mode = *mode;
    }

    let total_time = times.last().unwrap() - times.first().unwrap();
    println!(
        "  Time with heater ON: {:.2}h ({:.1}%)",
        time_heater_on,
        100.0 * time_heater_on / total_time
    );
    println!(
        "  Time with heater OFF: {:.2}h ({:.1}%)",
        time_heater_off,
        100.0 * time_heater_off / total_time
    );

    // Calculate temperature statistics
    let avg_temp = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
    let min_temp = temperatures.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_temp = temperatures
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("  Average temperature: {:.2}°C", avg_temp);
    println!(
        "  Temperature range: {:.2}°C to {:.2}°C",
        min_temp, max_temp
    );

    // Save results to CSV
    save_to_csv(&times, &temperatures, &modes, &event_times, &event_types)?;
    println!("\nResults saved to 'thermostat_simulation.csv'");

    Ok(())
}

fn save_to_csv(
    times: &[f64],
    temperatures: &[f64],
    modes: &[HeaterMode],
    event_times: &[f64],
    event_types: &[String],
) -> IntegrateResult<()> {
    let mut file = File::create("thermostat_simulation.csv")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to create file: {}", e)))?;

    // Write header
    writeln!(file, "time,temperature,heater_mode,is_event,event_type")
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write to file: {}", e)))?;

    // Write data
    for i in 0..times.len() {
        let t = times[i];

        // Check if this time point is an event
        let is_event = event_times.iter().position(|&et| (et - t).abs() < 1e-6);
        let event_info = if let Some(idx) = is_event {
            format!("1,{}", event_types[idx])
        } else {
            "0,".to_string()
        };

        // Write row
        writeln!(
            file,
            "{},{},{},{}",
            t,
            temperatures[i],
            if modes[i] == HeaterMode::On { 1 } else { 0 },
            event_info
        )
        .map_err(|e| IntegrateError::ComputationError(format!("Failed to write to file: {}", e)))?;
    }

    Ok(())
}
