# Event Detection in ODE Solvers

This guide explains how to use event detection in the scirs2-integrate ODE solvers.

## Overview

Event detection allows you to detect specific conditions during ODE integration, such as:

- When a state variable crosses a threshold (e.g., when y = 0)
- When a derivative becomes zero (e.g., at peaks or valleys)
- When a complex condition is met (e.g., when energy exceeds a threshold)

The event detection system can precisely locate the time at which these events occur and optionally terminate the integration when an event is detected.

## Key Components

The event detection system has three main components:

1. **Event Functions**: Functions that return a value which crosses zero when an event occurs
2. **Event Specifications**: Configuration for each event, defining its behavior
3. **Solver Integration**: Using `solve_ivp_with_events` instead of the standard `solve_ivp`

## Basic Usage

Here's a simple example that detects when a state variable crosses zero:

```rust
use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{
    solve_ivp_with_events, ODEMethod, ODEOptions, EventSpec,
    EventDirection, EventAction, ODEOptionsWithEvents
};

// Define the ODE system (simple harmonic oscillator)
let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

// Define event function: detect when y[0] crosses zero
let event_funcs = vec![|_t: f64, y: ArrayView1<f64>| y[0]];

// Define event specification
let event_specs = vec![
    EventSpec {
        id: "zero_crossing".to_string(),
        direction: EventDirection::Both,    // Detect both up and down crossings
        action: EventAction::Continue,      // Continue integration after event
        threshold: 1e-8,                    // Numerical threshold
        max_count: None,                    // Unlimited events
        precise_time: true,                 // Use interpolation for precise timing
    }
];

// Create solver options with events
let options = ODEOptionsWithEvents::new(
    ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        dense_output: true,  // Required for precise event detection
        ..Default::default()
    },
    event_specs,
);

// Solve the ODE with event detection
let result = solve_ivp_with_events(
    f,
    [0.0, 10.0],
    array![1.0, 0.0],
    event_funcs,
    options,
).unwrap();

// Access detected events
println!("Number of zero crossings: {}", result.events.get_count("zero_crossing"));
for event in result.events.get_events("zero_crossing") {
    println!("Event at t = {}: y = {:?}", event.time, event.state);
}
```

## Event Functions

Event functions have the signature:

```rust
Fn(F, ArrayView1<F>) -> F
```

where `F` is the float type (typically `f64`). These functions should return a value that crosses zero when the event occurs. For example:

- To detect when `y[0] = 5`, use: `|_t, y| y[0] - 5.0`
- To detect when `y[0] = y[1]`, use: `|_t, y| y[0] - y[1]`
- To detect when the norm exceeds a threshold: `|_t, y| 1.0 - (y[0]*y[0] + y[1]*y[1]).sqrt()`

## Event Specification Options

`EventSpec` has several configuration options:

- `id`: String identifier for the event type
- `direction`: When to trigger the event
  - `EventDirection::Rising`: When the function crosses from negative to positive
  - `EventDirection::Falling`: When the function crosses from positive to negative
  - `EventDirection::Both`: Detect crossings in both directions
- `action`: What to do when the event is detected
  - `EventAction::Continue`: Continue integration
  - `EventAction::Stop`: Terminate integration
- `threshold`: Numerical threshold for considering a crossing (prevents chattering)
- `max_count`: Optional limit on the number of events to detect
- `precise_time`: Whether to use interpolation for precise event timing

## Terminal Events

For convenience, there's a helper function to create terminal events:

```rust
let terminal_event_spec = terminal_event::<f64>("event_name", EventDirection::Rising);
```

This creates an event that will stop the integration when triggered.

## Multiple Event Types

You can define multiple event functions and specifications:

```rust
let event_funcs = vec![
    |_t, y| y[0],           // Event 1: y[0] = 0
    |_t, y| y[1],           // Event 2: y[1] = 0
    |_t, y| y[0] - 0.5      // Event 3: y[0] = 0.5
];

let event_specs = vec![
    EventSpec { id: "zero_crossing".to_string(), ... },
    EventSpec { id: "extremum".to_string(), ... },
    terminal_event::<f64>("threshold", EventDirection::Rising)
];
```

## Accessing Event Information

The `ODEResultWithEvents` returned by `solve_ivp_with_events` contains:

- `base_result`: The standard ODE solution
- `events`: Record of all detected events
- `dense_output`: Dense output for solution evaluation at any time point
- `event_termination`: Flag indicating if integration was terminated by an event

You can access events using:

```rust
// Get count of specific event type
let count = result.events.get_count("zero_crossing");

// Get all events of a specific type
let zero_events = result.events.get_events("zero_crossing");

// Get all events in chronological order
for event in &result.events.events {
    println!("{}: t = {}", event.id, event.time);
}

// Check if integration was terminated by an event
if result.event_termination {
    println!("Integration terminated by event");
}
```

## Dense Output

The dense output feature allows you to evaluate the solution at any time point:

```rust
if let Some(ref dense) = result.dense_output {
    // Evaluate at a specific time
    let y_at_time = dense.evaluate(3.14)?;
    println!("y(3.14) = {:?}", y_at_time);
    
    // Evaluate at event time
    let event_time = result.events.events[0].time;
    let y_at_event = dense.evaluate(event_time)?;
}
```

## Handling Discontinuous Systems

For systems with discontinuities at events (like impacts), you'll need to handle each event manually:

1. Integrate until the next event
2. Apply the discontinuity rule (e.g., reverse velocity)
3. Continue integration from the new state

See the `bouncing_ball_with_events.rs` example for a complete implementation of this approach.

## Performance Considerations

1. Event detection increases computational cost due to:
   - Interpolation for precise event location
   - Additional function evaluations
   - Dense output generation

2. Use `precise_time: false` for less critical events to reduce overhead

3. Use `max_count` when appropriate to limit event detection

## Common Use Cases

Event detection is useful for many applications:

- **Physics simulations**: Detecting collisions, threshold crossings
- **Chemical reactions**: Detecting when a species reaches a concentration
- **Control systems**: Detecting when a state crosses a threshold
- **Periodic systems**: Detecting period completion
- **Bifurcation analysis**: Detecting qualitative changes in behavior

## Advanced Topics

### Root Finding for Precise Event Location

Event detection uses bisection search to precisely locate event times. This works by:

1. Detecting when the event function changes sign between steps
2. Using bisection with dense output to find the exact crossing time
3. Evaluating the solution at the event time

### Event Direction

The event direction is recorded in each event:
- `1` for Rising (negative to positive)
- `-1` for Falling (positive to negative)

This can be useful for distinguishing types of crossings.

### Custom Event Handling

For complex systems, you may need to implement custom event handling logic. The general approach is:

1. Integrate until the next event
2. Process the event and determine the new state
3. Continue integration with the new state

This approach allows for sophisticated behaviors like:
- State discontinuities (impacts, switches)
- Mode changes in hybrid systems
- Complex event-based control logic

See the examples directory for more detailed implementations.