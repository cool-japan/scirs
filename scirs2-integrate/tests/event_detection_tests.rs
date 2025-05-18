//! Unit tests for event detection in ODE solvers
//!
//! These tests verify the event detection and handling functionality by
//! testing with simple ODE systems that have predictable event conditions.

use approx::assert_relative_eq;
use ndarray::{array, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, EventAction, EventDirection, EventSpec, ODEMethod, ODEOptions,
    ODEOptionsWithEvents,
};

/// Test basic event detection with a linear ODE
#[test]
fn test_basic_event_detection() -> IntegrateResult<()> {
    // Simple ODE: dy/dt = 1.0 (meaning y = t + y0)
    // Event: y crosses y = 2.0

    let f = |_t: f64, _y: ArrayView1<f64>| array![1.0];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0] - 2.0, // Event when y = 2.0
    ];

    let event_specs = vec![EventSpec {
        id: "threshold".to_string(),
        direction: EventDirection::Rising,
        action: EventAction::Continue,
        threshold: 1e-10,
        max_count: None,
        precise_time: true,
    }];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    // Solve with initial condition y0 = 0.0, so event should occur at t = 2.0
    let result = solve_ivp_with_events(f, [0.0, 10.0], array![0.0], event_funcs, options)?;

    // Check if exactly one event was detected
    assert_eq!(
        result.events.get_count("threshold"),
        1,
        "Expected exactly one threshold crossing"
    );

    // Get the event
    let event = result.events.get_events("threshold")[0];

    // Check event time (should be very close to 2.0)
    assert_relative_eq!(event.time, 2.0, epsilon = 1e-8);

    // Check event state
    assert_relative_eq!(event.state[0], 2.0, epsilon = 1e-8);

    // Check event direction
    assert_eq!(event.direction, 1, "Event direction should be positive");

    Ok(())
}

/// Test terminal event stopping integration
#[test]
fn test_terminal_event() -> IntegrateResult<()> {
    // Simple harmonic oscillator: d²y/dt² = -y
    // Implemented as a system:
    // dy₁/dt = y₂
    // dy₂/dt = -y₁

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // Event when y = 0
    ];

    let event_specs = vec![EventSpec {
        id: "zero_crossing".to_string(),
        direction: EventDirection::Falling, // Falling through zero
        action: EventAction::Stop,          // Terminal event
        threshold: 1e-10,
        max_count: Some(1), // Stop at first crossing
        precise_time: true,
    }];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    // Initial condition: y(0) = 1.0, y'(0) = 0.0
    // First zero crossing should be at t = π/2
    let result = solve_ivp_with_events(
        f,
        [0.0, 10.0], // Integration should stop before t = 10.0
        array![1.0, 0.0],
        event_funcs,
        options,
    )?;

    // Check if integration was terminated by the event
    assert!(
        result.event_termination,
        "Integration should terminate due to event"
    );

    // Check if exactly one event was detected
    assert_eq!(
        result.events.get_count("zero_crossing"),
        1,
        "Expected exactly one zero crossing"
    );

    // Get the event
    let event = result.events.get_events("zero_crossing")[0];

    // Check event time (should be close to π/2 ≈ 1.5708)
    assert_relative_eq!(event.time, std::f64::consts::PI / 2.0, epsilon = 1e-4);

    // Check event state
    assert_relative_eq!(event.state[0], 0.0, epsilon = 1e-8);

    // Verify integration stopped at the event time
    let final_time = result.base_result.t.last().unwrap();
    assert_relative_eq!(*final_time, event.time, epsilon = 1e-10);

    Ok(())
}

/// Test event detection with precise time location
#[test]
fn test_precise_event_location() -> IntegrateResult<()> {
    // Cubic function that crosses zero with steep gradient
    // dy/dt = 3t² (solution: y = t³ + y0)

    let f = |t: f64, _y: ArrayView1<f64>| array![3.0 * t * t];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // Event when y = 0
    ];

    // We'll start with y(0) = -1.0, so event should occur at t = 1.0 (when t³ = 1.0)

    // Test with both precise time location enabled and disabled
    for &precise_time in &[true, false] {
        let event_specs = vec![EventSpec {
            id: "zero".to_string(),
            direction: EventDirection::Rising,
            action: EventAction::Continue,
            threshold: 1e-10,
            max_count: None,
            precise_time,
        }];

        let options = ODEOptionsWithEvents::new(
            ODEOptions {
                method: ODEMethod::RK45,
                rtol: 1e-9,
                atol: 1e-11,
                dense_output: true,
                max_step: Some(0.1), // Force smaller steps for better accuracy
                ..Default::default()
            },
            event_specs,
        );

        let result =
            solve_ivp_with_events(f, [0.0, 2.0], array![-1.0], event_funcs.clone(), options)?;

        // Check if exactly one event was detected
        assert_eq!(
            result.events.get_count("zero"),
            1,
            "Expected exactly one zero crossing"
        );

        // Get the event
        let event = result.events.get_events("zero")[0];

        // Check event time (should be close to 1.0)
        let expected_tolerance = if precise_time { 3e-2 } else { 5e-2 };
        assert_relative_eq!(event.time, 1.0, epsilon = expected_tolerance);

        // Check event state
        assert_relative_eq!(event.state[0], 0.0, epsilon = 1e-1);
    }

    Ok(())
}

/// Test multiple events in both directions
#[test]
fn test_multiple_events() -> IntegrateResult<()> {
    // Simple harmonic oscillator: d²y/dt² = -y
    // Solution: y(t) = cos(t) for y(0) = 1, y'(0) = 0

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // Event when y = 0
    ];

    let event_specs = vec![EventSpec {
        id: "zero".to_string(),
        direction: EventDirection::Both, // Detect both directions
        action: EventAction::Continue,
        threshold: 1e-10,
        max_count: None,
        precise_time: true,
    }];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    // Integrate for 3 periods (each period is 2π)
    let t_end = 3.0 * 2.0 * std::f64::consts::PI;

    let result = solve_ivp_with_events(f, [0.0, t_end], array![1.0, 0.0], event_funcs, options)?;

    // Should get 6 zero-crossings (2 per period)
    assert_eq!(
        result.events.get_count("zero"),
        6,
        "Expected 6 zero crossings (2 per period)"
    );

    // Check event times - should be at odd multiples of π/2
    let events = result.events.get_events("zero");

    for (i, event) in events.iter().enumerate() {
        let expected_time = (i as f64 + 0.5) * std::f64::consts::PI;
        assert_relative_eq!(event.time, expected_time, epsilon = 1e-4);

        // Check zero-crossing direction (alternating)
        let expected_direction = if i % 2 == 0 { -1 } else { 1 };
        assert_eq!(
            event.direction, expected_direction,
            "Event direction should alternate"
        );
    }

    Ok(())
}

/// Test max_count feature to limit number of detected events
#[test]
fn test_max_count() -> IntegrateResult<()> {
    // Simple oscillator to generate many events
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // Event when y = 0
    ];

    let event_specs = vec![EventSpec {
        id: "zero".to_string(),
        direction: EventDirection::Both,
        action: EventAction::Continue,
        threshold: 1e-10,
        max_count: Some(3), // Only record first 3 events
        precise_time: true,
    }];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    // Integrate for 5 periods (should give 10 zero-crossings, but max_count is 3)
    let t_end = 5.0 * 2.0 * std::f64::consts::PI;

    let result = solve_ivp_with_events(f, [0.0, t_end], array![1.0, 0.0], event_funcs, options)?;

    // Should only get 3 events due to max_count
    assert_eq!(
        result.events.get_count("zero"),
        3,
        "Expected only 3 events due to max_count"
    );

    // Check that integration continued to the end time
    assert_relative_eq!(*result.base_result.t.last().unwrap(), t_end, epsilon = 1e-3);

    Ok(())
}

/// Test event handling with multiple event functions
#[test]
fn test_multiple_event_functions() -> IntegrateResult<()> {
    // Simple harmonic oscillator
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Two event functions:
    // 1. Zero crossing (y = 0)
    // 2. Maximum/minimum (y' = 0)
    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // Event when y = 0
        |_t: f64, y: ArrayView1<f64>| y[1], // Event when y' = 0
    ];

    let event_specs = vec![
        EventSpec {
            id: "zero_crossing".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-10,
            max_count: None,
            precise_time: true,
        },
        EventSpec {
            id: "extremum".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-10,
            max_count: None,
            precise_time: true,
        },
    ];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    // Integrate for 2 periods
    let t_end = 2.0 * 2.0 * std::f64::consts::PI;

    let result = solve_ivp_with_events(f, [0.0, t_end], array![1.0, 0.0], event_funcs, options)?;

    // Should get 4 zero-crossings (2 per period)
    assert_eq!(
        result.events.get_count("zero_crossing"),
        4,
        "Expected 4 zero crossings"
    );

    // Should get 4 extrema (2 per period)
    assert_eq!(
        result.events.get_count("extremum"),
        4,
        "Expected 4 extrema (max/min)"
    );

    // Check that extrema and zero crossings alternate and are spaced by π/2
    let zero_events = result.events.get_events("zero_crossing");
    let extrema_events = result.events.get_events("extremum");

    // For cos(t) starting at t=0:
    // - Extrema at t = 0, π, 2π, 3π (y' = 0)
    // - Zero crossings at t = π/2, 3π/2, 5π/2, 7π/2 (y = 0)

    // Check zero crossings
    for (i, event) in zero_events.iter().enumerate() {
        let expected_time = (2.0 * i as f64 + 1.0) * std::f64::consts::PI / 2.0;
        assert_relative_eq!(event.time, expected_time, epsilon = 1e-2);
    }

    // Check extrema (note: event detection might not catch the initial extremum at t=0)
    for (i, event) in extrema_events.iter().enumerate() {
        // If we missed the initial extremum at t=0, the first detected extremum will be at t=π
        let offset = if extrema_events[0].time > 0.5 {
            1.0
        } else {
            0.0
        };
        let expected_time = (i as f64 + offset) * std::f64::consts::PI;
        assert_relative_eq!(event.time, expected_time, epsilon = 1e-2);
    }

    Ok(())
}

/// Test that dense output evaluation works correctly at event times
#[test]
fn test_dense_output_at_event() -> IntegrateResult<()> {
    // Simple harmonic oscillator: solution y(t) = cos(t), y'(t) = -sin(t)
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0] + 0.5, // Event when y = -0.5
    ];

    let event_specs = vec![EventSpec {
        id: "threshold".to_string(),
        direction: EventDirection::Falling,
        action: EventAction::Continue,
        threshold: 1e-10,
        max_count: None,
        precise_time: true,
    }];

    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        },
        event_specs,
    );

    let result = solve_ivp_with_events(
        f,
        [0.0, 2.0 * std::f64::consts::PI],
        array![1.0, 0.0],
        event_funcs,
        options,
    )?;

    // Check if event was detected
    assert_eq!(
        result.events.get_count("threshold"),
        1,
        "Expected one threshold crossing"
    );

    // Get the event
    let event = result.events.get_events("threshold")[0];

    // Event should occur when cos(t) = -0.5, which is t = 2π/3 or t = 4π/3
    // Since we're starting with y = 1 and looking for falling crossings, it's t = 2π/3
    let expected_time = 2.0 * std::f64::consts::PI / 3.0;
    assert_relative_eq!(event.time, expected_time, epsilon = 1e-4);

    // Check event state
    assert_relative_eq!(event.state[0], -0.5, epsilon = 1e-8);

    // Now check that dense output evaluation gives the same result at event time
    if let Some(ref dense) = result.dense_output {
        let y_at_event = dense.evaluate(event.time)?;
        assert_relative_eq!(y_at_event[0], -0.5, epsilon = 1e-8);

        // Test at a few other times to verify dense output
        let test_times = vec![
            std::f64::consts::PI / 4.0,       // y = cos(π/4) ≈ 0.7071
            std::f64::consts::PI,             // y = cos(π) = -1
            3.0 * std::f64::consts::PI / 2.0, // y = cos(3π/2) = 0
        ];

        for &t in &test_times {
            let y_evaluated = dense.evaluate(t)?;
            let expected_y = (t.cos(), -t.sin()); // (y, y')

            assert_relative_eq!(y_evaluated[0], expected_y.0, epsilon = 1e-4);
            assert_relative_eq!(y_evaluated[1], expected_y.1, epsilon = 1e-4);
        }
    } else {
        panic!("Dense output should be available");
    }

    Ok(())
}
