# LSODA Solver Guide

## Overview

LSODA (Livermore Solver for Ordinary Differential Equations with Automatic method switching) is an adaptive ODE solver that automatically switches between non-stiff and stiff methods during integration. This makes it particularly effective for problems that change character or when the stiffness of the problem is not known in advance.

## Key Features

- **Automatic Method Switching**: Dynamically switches between Adams methods (non-stiff) and BDF methods (stiff)
- **Adaptive Step Size Control**: Adjusts step size based on error estimates and problem characteristics
- **Problem-Scale Detection**: Automatically determines appropriate scales for the problem
- **Enhanced Stability**: Includes specialized strategies for handling challenging problems
- **Comprehensive Error Control**: Maintains both absolute and relative error within specified tolerances

## When to Use LSODA

LSODA is especially valuable when:

1. **The problem changes character** during integration (e.g., from non-stiff to stiff)
2. **The stiffness of the problem is unknown** in advance
3. **Robustness is more important than raw performance** for a specific problem type
4. **You want a general-purpose solver** that works well across many different ODE systems

## Implementation Details

### Methods Used

- **Non-stiff regions**: Adams-Moulton predictor-corrector methods (orders 1-12)
- **Stiff regions**: Backward Differentiation Formula (BDF) methods (orders 1-5)

### Stiffness Detection

The implementation uses multiple indicators to detect stiffness:

- **Relative step size**: Very small steps relative to the problem scale suggest stiffness
- **Efficiency ratio**: The ratio of accepted to rejected steps indicates potential stiffness
- **Recent rejection rate**: High rejection rates can signal stiffness
- **Step size limits**: Consistently operating near the minimum step size is a stiffness indicator

### Method Switching Logic

Method switching includes "hysteresis" to prevent rapid oscillation between methods:

- **Non-stiff to stiff**: Requires multiple indicators of stiffness
- **Stiff to non-stiff**: Very conservative, requires clear evidence of non-stiff behavior

## Usage

### Basic Usage

```rust
use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    // Define ODE system: y' = -50*y (a stiff problem)
    let stiff_system = |_t: f64, y: ArrayView1<f64>| array![-50.0 * y[0]];
    
    // Solve using LSODA
    let result = solve_ivp(
        stiff_system,
        [0.0, 1.0],  // time span [t_start, t_end]
        array![1.0],  // initial condition
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 1000,
            ..Default::default()
        }),
    );
    
    // Process results
    match result {
        Ok(sol) => {
            println!("Final value: {}", sol.y.last().unwrap()[0]);
            println!("Method switches: {}", sol.message.unwrap_or_default());
        }
        Err(e) => println!("Error: {}", e),
    }
}
```

### Optimization Tips

1. **Adjust tolerances**: Increase `rtol` and `atol` for less demanding accuracy requirements and better performance
2. **Initial step size**: Provide a reasonable initial step size (`h0`) if you have domain knowledge
3. **Monitor method switching**: Check the result's `message` field to see method switching information

```rust
// Example with custom settings
let result = solve_ivp(
    stiff_system,
    [0.0, 1.0],
    array![1.0],
    Some(ODEOptions {
        method: ODEMethod::LSODA,
        rtol: 1e-4,  // Relaxed relative tolerance
        atol: 1e-6,  // Relaxed absolute tolerance
        h0: Some(0.01),  // Suggested initial step size
        max_steps: 2000,  // Allow more steps for very challenging problems
        ..Default::default()
    }),
);
```

## Example Problems

### Van der Pol Oscillator

The Van der Pol oscillator is a classic example of a problem that changes stiffness:

```rust
// Van der Pol oscillator with μ = 10 (moderately stiff)
let mu = 10.0;
let van_der_pol = |_t: f64, y: ArrayView1<f64>| {
    array![
        y[1],
        mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
    ]
};

let result = solve_ivp(
    van_der_pol,
    [0.0, 20.0],
    array![2.0, 0.0],
    Some(ODEOptions {
        method: ODEMethod::LSODA,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 5000,
        ..Default::default()
    }),
);
```

### Chemical Kinetics

Chemical reaction systems often exhibit stiffness due to widely varying reaction rates:

```rust
// Simplified chemical kinetics problem
// A → B (slow)
// B → C (fast)
let chemical_kinetics = |_t: f64, y: ArrayView1<f64>| {
    let k1 = 0.1;    // slow rate constant
    let k2 = 1000.0; // fast rate constant
    
    array![
        -k1 * y[0],           // rate of change of A
        k1 * y[0] - k2 * y[1], // rate of change of B
        k2 * y[1]             // rate of change of C
    ]
};
```

## Comparison with Other Methods

- **RK45/RK23 (non-stiff)**: Faster for non-stiff problems, but fails on stiff problems
- **BDF (stiff)**: Slightly more efficient for consistently stiff problems
- **Radau (stiff)**: Higher accuracy for very stiff problems, but more computational cost

## References

- Petzold, L. (1983). Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations. SIAM Journal on Scientific and Statistical Computing, 4(1), 136-148.
- Hindmarsh, A. C. (1983). ODEPACK, a systematized collection of ODE solvers. Scientific Computing, 55-64.