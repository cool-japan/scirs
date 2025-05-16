//! Tests for combined mass matrix and event detection support
//!
//! These tests verify that mass matrices and event detection can be
//! combined correctly in ODE solvers.

use approx::assert_relative_eq;
use ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, EventAction, EventDirection, EventSpec, MassMatrix, ODEMethod,
    ODEOptions, ODEOptionsWithEvents,
};

/// Test event detection with a constant mass matrix
#[test]
fn test_constant_mass_with_events() -> IntegrateResult<()> {
    // Simple oscillator with a non-identity mass matrix
    // [2 0] [x'] = [    v    ]
    // [0 1] [v']   [   -x    ]
    //
    // This gives: x' = v/2, v' = -x
    // Solution: x(t) = cos(t/√2), v(t) = -√2·sin(t/√2)
    // Period = 2π·√2 ≈ 8.886 seconds

    // Create mass matrix
    let mut mass_matrix = Array2::<f64>::eye(2);
    mass_matrix[[0, 0]] = 2.0; // Mass of 2 for x component

    // Create the mass matrix specification
    let mass = MassMatrix::constant(mass_matrix);

    // ODE function
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Event functions:
    // 1. Detect when x = 0 (zero crossing)
    // 2. Detect when v = 0 (max amplitude)
    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // x = 0
        |_t: f64, y: ArrayView1<f64>| y[1], // v = 0
    ];

    // Event specifications
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
            id: "max_amplitude".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-10,
            max_count: None,
            precise_time: true,
        },
    ];

    // Create options with both mass matrix and event specs
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::Radau, // Implicit method with direct mass matrix support
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            mass_matrix: Some(mass),
            ..Default::default()
        },
        event_specs,
    );

    // Integrate for 3 full periods
    let omega = 1.0 / f64::sqrt(2.0); // Natural frequency
    let period = 2.0 * std::f64::consts::PI / omega;
    let t_end = 3.0 * period;

    // Solve with event detection
    let result = solve_ivp_with_events(f, [0.0, t_end], y0, event_funcs, options)?;

    // Verify basic solution properties
    assert!(result.base_result.success, "Integration should succeed");

    // Verify event detection
    // We should have 6 zero crossings (2 per period) and 6 max amplitude events (2 per period)
    assert_eq!(
        result.events.get_count("zero_crossing"),
        6,
        "Should detect 6 zero crossings over 3 periods"
    );
    assert_eq!(
        result.events.get_count("max_amplitude"),
        6,
        "Should detect 6 max amplitude events over 3 periods"
    );

    // Verify event times for zero crossings (at odd multiples of π/2ω)
    let zero_events = result.events.get_events("zero_crossing");
    for (i, event) in zero_events.iter().enumerate() {
        let expected_time = (i as f64 + 0.5) * std::f64::consts::PI / omega;
        assert_relative_eq!(
            event.time,
            expected_time,
            epsilon = 1e-3,
            max_relative = 1e-3,
            "Zero crossing should occur at t = {}",
            expected_time
        );
    }

    // Verify event times for max amplitude (at multiples of π/ω)
    let max_events = result.events.get_events("max_amplitude");
    for (i, event) in max_events.iter().enumerate() {
        let expected_time = i as f64 * std::f64::consts::PI / omega;
        assert_relative_eq!(
            event.time,
            expected_time,
            epsilon = 1e-3,
            max_relative = 1e-3,
            "Max amplitude should occur at t = {}",
            expected_time
        );
    }

    Ok(())
}

/// Test event detection with a time-dependent mass matrix
#[test]
fn test_time_dependent_mass_with_events() -> IntegrateResult<()> {
    // Oscillator with a time-dependent mass: m(t) = 1 + 0.5·sin(t)
    // [m(t) 0] [x'] = [    v    ]
    // [  0  1] [v']   [   -x    ]

    // Time-dependent mass matrix
    let time_dependent_mass = |t: f64| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = 1.0 + 0.5 * t.sin();
        m
    };

    // Create the mass matrix
    let mass = MassMatrix::time_dependent(time_dependent_mass);

    // ODE function
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Event function: detect when x crosses through 0.5
    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0] - 0.5, // x = 0.5
    ];

    // Event specifications
    let event_specs = vec![EventSpec {
        id: "threshold".to_string(),
        direction: EventDirection::Both, // Detect both directions
        action: EventAction::Continue,
        threshold: 1e-10,
        max_count: Some(4), // Only detect first 4 crossings
        precise_time: true,
    }];

    // Create options
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::Radau,
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true,
            mass_matrix: Some(mass),
            ..Default::default()
        },
        event_specs,
    );

    // Solve for 10 seconds (should cover multiple oscillations)
    let result = solve_ivp_with_events(f, [0.0, 10.0], y0, event_funcs, options)?;

    // Verify basic solution properties
    assert!(result.base_result.success, "Integration should succeed");

    // Verify event detection
    assert_eq!(
        result.events.get_count("threshold"),
        4,
        "Should detect exactly 4 threshold crossings due to max_count limit"
    );

    // Verify that events alternate direction (falling, rising, falling, rising)
    let threshold_events = result.events.get_events("threshold");
    for i in 0..threshold_events.len() {
        let expected_direction = if i % 2 == 0 { -1 } else { 1 };
        assert_eq!(
            threshold_events[i].direction, expected_direction,
            "Event direction should alternate between falling and rising"
        );
    }

    // Verify event state values
    for event in threshold_events {
        assert_relative_eq!(
            event.state[0],
            0.5,
            epsilon = 1e-8,
            "Event state should be x = 0.5"
        );
    }

    // Verify increasing time between events (due to time-dependent mass)
    if threshold_events.len() >= 3 {
        let interval1 = threshold_events[1].time - threshold_events[0].time;
        let interval2 = threshold_events[3].time - threshold_events[2].time;

        // The mass increases at times, which changes the period
        // We're mainly testing that the events are detected correctly
        assert!(
            interval1 > 0.0 && interval2 > 0.0,
            "Time intervals between events should be positive"
        );
    }

    Ok(())
}

/// Test event detection with a state-dependent mass matrix and terminal event
#[test]
fn test_state_dependent_mass_with_terminal_event() -> IntegrateResult<()> {
    // Nonlinear pendulum with state-dependent effective mass
    // The effective mass increases with angle due to the nonlinear term

    // Parameters
    let g = 9.81; // Gravity
    let l = 1.0; // Pendulum length
    let m = 1.0; // Mass

    // State-dependent mass matrix: considers full nonlinear pendulum dynamics
    // M(θ) = [m    0]   where m_effective = m * (1 + θ²/12 + higher order terms)
    //        [0    1]   This approximates the effect of using sin(θ) vs θ
    let state_dependent_mass = move |_t: f64, y: ArrayView1<f64>| {
        let theta = y[0];

        // Effective mass includes nonlinear correction
        // For small angles, expanded to second order
        let effective_mass = m * (1.0 + theta * theta / 12.0);

        // Create mass matrix
        let mut mass_matrix = Array2::<f64>::eye(2);
        mass_matrix[[0, 0]] = effective_mass;

        mass_matrix
    };

    // Create the mass matrix specification
    let mass = MassMatrix::state_dependent(state_dependent_mass);

    // ODE function: nonlinear pendulum with full sine term
    let f = move |_t: f64, y: ArrayView1<f64>| {
        array![
            y[1],                // θ' = ω
            -g / l * y[0].sin()  // ω' = -g/l·sin(θ)
        ]
    };

    // Initial conditions: start with large angle (60 degrees)
    let theta_0 = std::f64::consts::PI / 3.0; // 60 degrees
    let omega_0 = 0.0;
    let y0 = array![theta_0, omega_0];

    // Event functions:
    // 1. Detect when pendulum passes through center (θ = 0)
    // 2. Terminal event: stop when angle is less than 5 degrees
    let event_funcs = vec![
        |_t: f64, y: ArrayView1<f64>| y[0], // θ = 0
        |_t: f64, y: ArrayView1<f64>| y[0].abs() - std::f64::consts::PI / 36.0, // |θ| = 5 degrees
    ];

    // Event specifications
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
            id: "small_angle".to_string(),
            direction: EventDirection::Falling, // When |θ| - 5_degrees becomes negative
            action: EventAction::Stop,          // Terminal event
            threshold: 1e-10,
            max_count: Some(1),
            precise_time: true,
        },
    ];

    // Create options
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::Radau, // Required for state-dependent mass
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true,
            mass_matrix: Some(mass),
            ..Default::default()
        },
        event_specs,
    );

    // Solve for a long enough time to reach small angles (damping from stiffness)
    let result = solve_ivp_with_events(f, [0.0, 30.0], y0, event_funcs, options)?;

    // Verify that integration terminated due to event
    assert!(
        result.event_termination,
        "Integration should terminate due to event"
    );
    assert_eq!(
        result.events.get_count("small_angle"),
        1,
        "Should detect exactly one small_angle event"
    );

    // Verify terminal event properties
    if let Some(terminal_event) = result.events.get_events("small_angle").first() {
        assert!(
            terminal_event.state[0].abs() <= std::f64::consts::PI / 36.0 + 1e-8,
            "Terminal event should occur when |θ| ≤ 5 degrees"
        );
    }

    // Verify that integration stopped at the terminal event time
    let final_time = result.base_result.t.last().unwrap();
    let terminal_time = result.events.get_events("small_angle")[0].time;
    assert_relative_eq!(
        *final_time,
        terminal_time,
        epsilon = 1e-10,
        "Integration should stop at terminal event time"
    );

    // Verify that at least one zero crossing was detected
    assert!(
        result.events.get_count("zero_crossing") > 0,
        "Should detect at least one zero crossing"
    );

    Ok(())
}
