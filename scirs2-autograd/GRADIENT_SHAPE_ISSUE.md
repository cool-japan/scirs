# Gradient Shape Propagation Issue

## Problem Description

The autograd system has an architectural limitation where gradient shapes are not properly propagated for certain operations, particularly those where the input and output shapes differ significantly (e.g., matrix → scalar operations like `trace`).

## Symptoms

When computing gradients like `grad(trace(inv(A)), A)`, the expected gradient should be a matrix with the same shape as `A`, but instead a scalar gradient is returned.

## Root Cause

The gradient computation in `src/gradient.rs` uses a hardcoded approach based on operation names rather than calling the `grad` methods defined for each operation. This was done to avoid "lifetime issues with GradientContext" as mentioned in the code comments around line 317:

```rust
// FIXME: Terrible hack to avoid lifetime issues with GradientContext
// The real fix is to call `op.grad(...)` for all ops, but that requires
// refactoring the lifetime management
```

## Specific Issues

1. **Shape Information Loss**: The gradient computation cannot access runtime shape information, only static shape metadata.
2. **Operation-Specific Gradients**: Operations with complex gradient shapes (trace, determinant, etc.) cannot be properly handled.
3. **Chain Rule Application**: When gradients flow through multiple operations with shape changes, the shape information is lost.

## Affected Operations

- `trace`: Returns scalar but gradient should be identity matrix
- `determinant`: Returns scalar but gradient should be det(A) * inv(A)^T
- Matrix inverse with trace: Compound operations lose gradient shape
- Any custom operation where output shape differs from input shape

## Current Workarounds

Tests for gradient computation of these operations are marked with:
```rust
#[ignore = "Gradient shape propagation architectural limitation"]
```

## Proposed Solution

The gradient computation system needs to be refactored to:
1. Properly call the `grad` methods of operations
2. Resolve the lifetime issues with `GradientContext`
3. Maintain shape information throughout the gradient computation

This would be a significant architectural change affecting the entire autograd system.

## Impact

- Matrix calculus operations cannot compute proper gradients
- Limits the usefulness of autograd for linear algebra operations
- Requires manual gradient computation for affected operations