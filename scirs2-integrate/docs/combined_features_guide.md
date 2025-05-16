# Combined Mass Matrix and Event Detection Guide

This guide explains how to effectively use mass matrices and event detection together in the ODE solvers of the scirs2-integrate module. These powerful features can be combined to solve complex problems with constraints and special conditions.

## Overview

When solving differential equations of the form M(t,y)·y' = f(t,y) with conditions that require precise detection (like zero-crossings or thresholds), you can leverage both the mass matrix and event detection capabilities of the solvers.

The key components to use these features together are:

1. **Mass Matrix Specification**: Define the mass matrix as constant, time-dependent, or state-dependent
2. **Event Functions**: Define functions that cross zero when events of interest occur
3. **Event Specifications**: Configure how events are detected and handled
4. **Solver Method Selection**: Choose an appropriate solver that supports both features

## Creating Mass Matrices

Mass matrices are defined using the `MassMatrix` struct:

```rust
// Constant mass matrix
let mass_matrix = Array2::<f64>::eye(2);
let constant_mass = MassMatrix::constant(mass_matrix);

// Time-dependent mass matrix
let time_dependent_mass = |t: f64| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 1.0 + 0.5 * t.sin();
    m
};
let mass = MassMatrix::time_dependent(time_dependent_mass);

// State-dependent mass matrix
let state_dependent_mass = |_t: f64, y: ArrayView1<f64>| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 1.0 + y[0] * y[0];
    m
};
let mass = MassMatrix::state_dependent(state_dependent_mass);
```

## Defining Events

Events are defined using functions that cross zero when the event occurs:

```rust
// Event functions
let event_funcs = vec![
    |_t: f64, y: ArrayView1<f64>| y[0],          // Event when y[0] = 0
    |_t: f64, y: ArrayView1<f64>| y[1],          // Event when y[1] = 0
    |t: f64, _y: ArrayView1<f64>| t - 5.0,       // Event when t = 5.0
    |_t: f64, y: ArrayView1<f64>| y[0] - y[1],   // Event when y[0] = y[1]
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
    // Additional event specifications...
];
```

## Choosing the Right Solver

For combined mass matrix and event detection:

- **Explicit Methods**: Work well with constant and time-dependent mass matrices
- **Implicit Methods**: Required for state-dependent mass matrices
- **Radau Method**: The most robust option for all types of mass matrices

```rust
let options = ODEOptionsWithEvents::new(
    ODEOptions {
        method: ODEMethod::Radau,  // Best for state-dependent mass matrices
        rtol: 1e-6,
        atol: 1e-8,
        dense_output: true,  // Required for precise event detection
        mass_matrix: Some(mass),
        ..Default::default()
    },
    event_specs
);
```

## Solving the System

Use the `solve_ivp_with_events` function to solve the system:

```rust
let result = solve_ivp_with_events(
    f,              // ODE function: f(t, y)
    [0.0, 10.0],    // Time span
    y0,             // Initial conditions
    event_funcs,    // Event functions
    options         // Options with mass matrix and events
)?;
```

## Analyzing the Results

The results include both the solution and detected events:

```rust
// Access solution
let final_time = result.base_result.t.last().unwrap();
let final_state = result.base_result.y.last().unwrap();

// Access events
let event_count = result.events.get_count("zero_crossing");
let events = result.events.get_events("zero_crossing");

// Check if terminated by event
if result.event_termination {
    println!("Integration terminated due to event");
}

// Get solution at specific time using dense output
if let Some(solution_at_time) = result.at_time(3.5)? {
    println!("Solution at t=3.5: {:?}", solution_at_time);
}
```

## Performance Considerations

When using both features together:

1. **Solver Selection**: Radau is generally the most reliable method
2. **Tolerance Settings**: Use appropriate tolerances (typically 1e-6 to 1e-10)
3. **Event Precision**: Set `precise_time: true` for important events
4. **Mass Matrix Type**: Use the simplest mass matrix type that meets your needs
5. **Limiting Events**: Use `max_count` to limit detection of frequent events

## Example: Variable-Mass Pendulum

This example demonstrates a pendulum with variable mass and event detection:

```rust
// Time-dependent mass matrix (decreasing mass)
let time_dependent_mass = |t: f64| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 2.0 * (-t / 5.0).exp();
    m
};
let mass = MassMatrix::time_dependent(time_dependent_mass);

// ODE function for a pendulum
let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -9.81 * y[0].sin()];

// Event function to detect equilibrium crossing
let event_funcs = vec![|_t: f64, y: ArrayView1<f64>| y[0]];
let event_specs = vec![
    EventSpec {
        id: "equilibrium".to_string(),
        direction: EventDirection::Both,
        action: EventAction::Continue,
        threshold: 1e-8,
        max_count: None,
        precise_time: true,
    }
];

// Solve the system
let options = ODEOptionsWithEvents::new(
    ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        dense_output: true,
        mass_matrix: Some(mass),
        ..Default::default()
    },
    event_specs
);

let result = solve_ivp_with_events(f, [0.0, 20.0], array![1.0, 0.0], event_funcs, options)?;

// Analyze changing periods as mass decreases
let crossings = result.events.get_events("equilibrium");
for i in 1..crossings.len() {
    if crossings[i].direction == crossings[i-1].direction {
        let period = crossings[i].time - crossings[i-1].time;
        println!("Period at t={:.2}: {:.4}", crossings[i].time, period);
    }
}
```

## Example: State-Dependent Mass Matrix with Terminal Event

This example demonstrates a system with a state-dependent mass matrix and a terminal event:

```rust
// State-dependent mass matrix (position-dependent)
let state_dependent_mass = |_t: f64, y: ArrayView1<f64>| {
    let mut m = Array2::<f64>::eye(2);
    // Effective mass increases quadratically with position
    m[[0, 0]] = 1.0 + 0.1 * y[0] * y[0];
    m
};
let mass = MassMatrix::state_dependent(state_dependent_mass);

// ODE function with nonlinear forcing
let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0] - 0.1 * y[0].powi(3)];

// Event functions: detect when velocity is zero and a terminal event
let event_funcs = vec![
    |_t: f64, y: ArrayView1<f64>| y[1],
    |_t: f64, y: ArrayView1<f64>| 2.0 - y[0].abs()
];

// Event specifications
let event_specs = vec![
    EventSpec {
        id: "turning_point".to_string(),
        direction: EventDirection::Both,
        action: EventAction::Continue,
        threshold: 1e-8,
        max_count: None,
        precise_time: true,
    },
    EventSpec {
        id: "threshold".to_string(),
        direction: EventDirection::Falling,
        action: EventAction::Stop,
        threshold: 1e-8,
        max_count: Some(1),
        precise_time: true,
    }
];

// Solve the system
let options = ODEOptionsWithEvents::new(
    ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        dense_output: true,
        mass_matrix: Some(mass),
        ..Default::default()
    },
    event_specs
);

let result = solve_ivp_with_events(f, [0.0, 10.0], array![0.0, 1.0], event_funcs, options)?;

// Check if terminated by threshold event
if result.event_termination {
    println!("Integration terminated when |x| reached 2.0");
    if let Some(event) = result.events.get_events("threshold").first() {
        println!("Termination time: {:.4}", event.time);
    }
}
```

## Compatibility Notes

- **Solver Methods**: All ODE methods support event detection, but only some (like Radau) support all mass matrix types
- **Dense Output**: Always enabled when using event detection
- **State-Dependent Mass**: Requires implicit methods (Radau is recommended)
- **Event Precision**: Setting `precise_time: true` provides more accurate event timing but is more computationally expensive

## Common Pitfalls

1. **Using explicit methods with state-dependent mass matrices**: This can lead to instability or incorrect results
2. **Forgetting to enable dense output**: This is required for precise event detection
3. **Too many events**: Detecting too many events can slow down the solver significantly
4. **Poor event function definition**: Ensure event functions cross zero cleanly (avoid discontinuities)
5. **Inconsistent mass matrix**: Ensure the mass matrix is consistent with the dimensions of your system

## Further Reading

- See `mass_matrix_example.rs` for examples of different mass matrix types
- See `bouncing_ball_with_events.rs` for basic event detection usage
- See `pendulum_with_events.rs` for more complex event detection
- See `mass_matrix_with_events.rs` for combined usage of both features
- See `state_dependent_mass_with_events.rs` for an advanced example with state-dependent mass