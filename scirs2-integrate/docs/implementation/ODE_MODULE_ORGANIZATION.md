# ODE Module Organization

## Overview

The ODE (Ordinary Differential Equation) module has been refactored to improve maintainability, extensibility, and organization. The original monolithic implementation (over 3,700 lines) has been modularized into smaller, focused files.

This document explains the new module organization and how the components interact.

## Directory Structure

```
src/ode/
├── mod.rs                 # Module entry point and re-exports
├── types.rs               # Core type definitions
├── solver.rs              # Main solve_ivp function
├── methods/               # ODE solver implementations
│   ├── mod.rs             # Method re-exports
│   ├── explicit.rs        # Explicit methods (Euler, RK4)
│   ├── adaptive.rs        # Adaptive methods (RK45, RK23, DOP853)
│   ├── implicit.rs        # Implicit methods (BDF, Radau)
│   └── lsoda.rs           # LSODA method (auto-switching)
└── utils/                 # Utility functions and helpers
    ├── mod.rs             # Utility re-exports
    ├── step_control.rs    # Step size control
    ├── interpolation.rs   # Solution interpolation
    └── diagnostics.rs     # Diagnostic tools
```

## Key Components

### Core Types (`types.rs`)

Contains the primary types used across the ODE module:

- `ODEMethod`: Enum of available solver methods
- `ODEOptions`: Options for configuring ODE solvers
- `ODEResult`: Result structure for solution data

### Solver Interface (`solver.rs`)

Provides the main entry point:

- `solve_ivp`: Generic solver that dispatches to appropriate method implementations
- Handles option parsing and validation

### Method Implementations (`methods/`)

Organized by method categories:

- `explicit.rs`: Simple fixed-step methods (Euler, RK4)
- `adaptive.rs`: Variable step-size methods (RK45, RK23, DOP853)
- `implicit.rs`: Methods for stiff problems (BDF, Radau)
- `lsoda.rs`: Automatic method-switching for problems that change character

### Utilities (`utils/`)

Common functionality used across solvers:

- `step_control.rs`: Step size selection and error control
- `interpolation.rs`: Dense output and solution interpolation
- `diagnostics.rs`: Problem analysis tools

## Implementation Status

### Fully Implemented

- Module structure and organization
- Types and interfaces
- Fixed step methods (Euler, RK4)
- Simple adaptive methods (RK45, RK23)

### Partially Implemented

- LSODA module (structure only with stubs that call original implementation)
- Implicit method stubs

### Transition Strategy

For backward compatibility, the module currently takes these approaches:

1. For fully implemented methods (Euler, RK4, RK45, RK23):
   - The methods use new implementations that internally call original code
   - Results are properly converted between old and new result types

2. For partially implemented methods (BDF, Radau, LSODA):
   - Stub implementations redirect to the original implementation
   - Original method provides results that are mapped to the new result type

### Extension Points

The new module structure provides clear extension points:

1. **Adding New Methods**:
   - Add implementation to the appropriate category file or create a new file
   - Add enum variant to `ODEMethod`
   - Update the dispatcher in `solve_ivp`

2. **Enhancing Existing Methods**:
   - Refine implementations in their respective files
   - Method-specific utilities can go in their own subfolders

3. **Extending Functionality**:
   - Add new utility modules in the `utils/` directory
   - Add cross-cutting concerns to `solver.rs`

## Completion Roadmap

To fully transition to the new structure, these steps remain:

1. Implement complex methods (BDF, Radau, LSODA) in the new structure
2. Add comprehensive tests for each method
3. Update examples to use the new structure
4. Once stable, remove the original implementation and update imports

## Usage Examples

Basic usage remains the same:

```rust
use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

// Define ODE system: dy/dt = -y
let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

// Solve the ODE
let result = solve_ivp(
    f,
    [0.0, 2.0],    // time span [t_start, t_end]
    array![1.0],   // initial condition
    Some(ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        ..Default::default()
    }),
).unwrap();

// Access the solution
let final_time = result.t.last().unwrap();
let final_value = result.y.last().unwrap()[0];
println!("y({}) = {}", final_time, final_value);
```