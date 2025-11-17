# Issue #70: Custom Gradient/Hessian Support for minimize_trust_constr

## Problem Statement

Currently, `minimize_trust_constr` only supports automatic gradient and Hessian computation via finite differences. Users cannot provide custom analytical derivatives, which limits:

1. **Performance**: Finite differences require O(n) additional function evaluations per iteration
2. **Accuracy**: Numerical derivatives are less accurate than analytical derivatives
3. **Migration**: Makes it difficult to port SciPy code that uses custom gradients/Hessians

### Current Implementation

**File**: `scirs2-optimize/src/constrained/trust_constr.rs`

**Current Signature**:
```rust
pub fn minimize_trust_constr<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<ConstraintFn>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
```

**Gradient Computation** (Lines 37-44):
```rust
// Calculate initial gradient using finite differences
let mut g = Array1::zeros(n);
for i in 0..n {
    let mut x_h = x.clone();
    x_h[i] += eps;
    let f_h = func(x_h.as_slice().unwrap());
    g[i] = (f_h - f) / eps;
    nfev += 1;
}
```

**Hessian Approximation** (Lines 80-81):
```rust
// Initialize approximation of the Hessian of the Lagrangian
let mut b = Array2::eye(n);
```

Currently uses BFGS update (identity matrix initialization).

## SciPy Reference

SciPy's `scipy.optimize.minimize` with `method='trust-constr'` supports:

```python
from scipy.optimize import minimize

result = minimize(
    fun=objective,
    x0=x0,
    method='trust-constr',
    jac=gradient_function,  # Custom gradient
    hess=hessian_function,  # Custom Hessian
    constraints=constraints
)
```

### Supported `jac` Options in SciPy

1. **`None`**: Finite differences (default)
2. **`callable`**: User-provided function `jac(x) -> ndarray`
3. **`'2-point'`**: Forward finite differences
4. **`'3-point'`**: Central finite differences
5. **`'cs'`**: Complex-step derivative

### Supported `hess` Options in SciPy

1. **`None`**: BFGS approximation (default)
2. **`callable`**: User-provided function `hess(x) -> ndarray`
3. **`HessianUpdateStrategy`**: BFGS, SR1, etc.

## Proposed Solution

### Phase 1: Design API (v0.2.0)

Introduce optional gradient and Hessian parameters:

```rust
/// Options for custom derivatives
pub enum GradientOption<G> {
    /// Automatic gradient via finite differences (default)
    FiniteDiff { method: FiniteDiffMethod },
    /// User-provided gradient function
    Custom(G),
}

pub enum FiniteDiffMethod {
    TwoPoint,   // Forward differences (current)
    ThreePoint, // Central differences (more accurate)
}

pub enum HessianOption<H> {
    /// BFGS approximation (default)
    BFGS,
    /// User-provided Hessian function
    Custom(H),
    /// SR1 approximation
    SR1,
}

/// Enhanced function signature
pub fn minimize_trust_constr_with_derivatives<F, G, H, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    gradient: GradientOption<G>,
    hessian: HessianOption<H>,
    constraints: &[Constraint<ConstraintFn>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Array1<f64>,
    H: Fn(&[f64]) -> Array2<f64>,
    S: Data<Elem = f64>,
```

### Phase 2: Backward Compatibility

Keep the existing API and add a new one:

```rust
// Existing API - unchanged
pub fn minimize_trust_constr<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<ConstraintFn>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Delegate to new implementation with finite differences
    minimize_trust_constr_with_derivatives(
        func,
        x0,
        GradientOption::FiniteDiff { method: FiniteDiffMethod::TwoPoint },
        HessianOption::BFGS,
        constraints,
        options,
    )
}
```

### Phase 3: Constraint Jacobians

Similarly, allow custom Jacobians for constraints:

```rust
pub struct Constraint<F, J = AutoJacobian> {
    pub fun: F,
    pub jac: J,  // Constraint Jacobian
    pub kind: ConstraintKind,
}

pub enum ConstraintJacobian<J> {
    Auto,      // Finite differences
    Custom(J), // User-provided
}
```

## Implementation Plan

### v0.2.0 Milestone

#### Week 1-2: Core Infrastructure
- [ ] Define `GradientOption` and `HessianOption` enums
- [ ] Implement `FiniteDiffMethod::ThreePoint` (central differences)
- [ ] Create abstraction for derivative computation

#### Week 3-4: Trust-Constr Integration
- [ ] Refactor `minimize_trust_constr` to use derivative abstractions
- [ ] Implement `minimize_trust_constr_with_derivatives`
- [ ] Add backward-compatible wrapper

#### Week 5: Constraint Jacobians
- [ ] Extend `Constraint` struct with Jacobian support
- [ ] Update constraint evaluation logic
- [ ] Implement custom constraint Jacobian handling

#### Week 6: Testing & Documentation
- [ ] Unit tests with analytical derivatives
- [ ] Comparison tests: custom vs finite-diff
- [ ] Performance benchmarks
- [ ] Update documentation and examples

#### Week 7: Other Optimizers
- [ ] Apply same pattern to `minimize_slsqp`
- [ ] Apply same pattern to `minimize_interior_point`
- [ ] Unified API across all constrained optimizers

## Usage Examples

### Example 1: Custom Gradient Only

```rust
use scirs2_optimize::constrained::{
    minimize_trust_constr_with_derivatives,
    GradientOption, HessianOption,
};
use scirs2_core::ndarray::{array, Array1};

fn objective(x: &[f64]) -> f64 {
    (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
}

fn gradient(x: &[f64]) -> Array1<f64> {
    array![
        2.0 * (x[0] - 1.0),
        2.0 * (x[1] - 2.0),
    ]
}

let x0 = array![0.0, 0.0];
let result = minimize_trust_constr_with_derivatives(
    objective,
    &x0,
    GradientOption::Custom(gradient),
    HessianOption::BFGS,  // Still use BFGS approximation
    &[],
    &Default::default(),
)?;
```

### Example 2: Custom Gradient and Hessian

```rust
fn hessian(x: &[f64]) -> Array2<f64> {
    array![
        [2.0, 0.0],
        [0.0, 2.0],
    ]
}

let result = minimize_trust_constr_with_derivatives(
    objective,
    &x0,
    GradientOption::Custom(gradient),
    HessianOption::Custom(hessian),
    &[],
    &Default::default(),
)?;
```

### Example 3: Central Differences (More Accurate)

```rust
let result = minimize_trust_constr_with_derivatives(
    objective,
    &x0,
    GradientOption::FiniteDiff {
        method: FiniteDiffMethod::ThreePoint,
    },
    HessianOption::BFGS,
    &[],
    &Default::default(),
)?;
```

## Performance Impact

### Expected Improvements with Custom Derivatives

For a problem with **n = 100 variables**:

| Method | Function Evals/Iter | Speedup |
|--------|---------------------|---------|
| Finite Diff Gradient | ~101 | 1x (baseline) |
| Custom Gradient | 1 | **~100x** |
| Custom Gradient + Hessian | 1 | **~100x + better convergence** |

### Accuracy Improvements

Analytical derivatives eliminate:
- Truncation errors from finite differences
- Cancellation errors in subtraction
- Step size selection issues

Expected: **2-3 orders of magnitude** better gradient accuracy.

## Migration from SciPy

### Before (SciPy):
```python
from scipy.optimize import minimize

result = minimize(
    fun=lambda x: (x[0]-1)**2 + (x[1]-2)**2,
    x0=[0, 0],
    method='trust-constr',
    jac=lambda x: [2*(x[0]-1), 2*(x[1]-2)],
    hess=lambda x: [[2, 0], [0, 2]],
)
```

### After (SciRS2 v0.2.0):
```rust
let result = minimize_trust_constr_with_derivatives(
    |x| (x[0]-1.0).powi(2) + (x[1]-2.0).powi(2),
    &array![0.0, 0.0],
    GradientOption::Custom(|x| array![2.0*(x[0]-1.0), 2.0*(x[1]-2.0)]),
    HessianOption::Custom(|_| array![[2.0, 0.0], [0.0, 2.0]]),
    &[],
    &Default::default(),
)?;
```

Nearly 1-to-1 translation!

## Breaking Changes

This is a **non-breaking addition** if implemented correctly:

✅ **Backward Compatible**:
- Existing `minimize_trust_constr` API remains unchanged
- New API is opt-in via `minimize_trust_constr_with_derivatives`

⚠️ **Optional Breaking Change** (for v0.2.0):
- Could unify all APIs to use the new signature
- Provide migration guide for users

## Testing Strategy

### Unit Tests
- [ ] Test with analytical vs numerical gradients
- [ ] Verify gradient accuracy
- [ ] Test Hessian approximations vs exact Hessian
- [ ] Test all finite difference methods

### Integration Tests
- [ ] Rosenbrock function with analytical derivatives
- [ ] Constrained optimization problems
- [ ] Compare convergence: custom vs auto derivatives

### Performance Tests
- [ ] Benchmark function evaluation counts
- [ ] Measure wall-clock time improvements
- [ ] Profile memory usage

## Documentation Requirements

### API Documentation
- [ ] Document all new types and functions
- [ ] Provide usage examples
- [ ] Migration guide from SciPy
- [ ] Performance comparison table

### User Guide
- [ ] When to use custom derivatives
- [ ] How to compute analytical derivatives
- [ ] Debugging tips for custom derivatives
- [ ] Common pitfalls and solutions

## Risks and Mitigation

### Risk 1: Complex API
**Mitigation**: Provide simple examples and good defaults

### Risk 2: User-Provided Gradient Errors
**Mitigation**:
- Add optional gradient checking mode
- Compare user gradient with finite differences
- Provide helpful error messages

### Risk 3: Type Complexity
**Mitigation**:
- Use type aliases
- Provide builder pattern for complex options

## Conclusion

This feature is **highly valuable** for SciRS2 users and brings SciPy compatibility closer. The implementation is straightforward and can be done in a backward-compatible manner.

**Recommendation**:
- **v0.2.0**: Full implementation with new API
- **v0.1.0**: Document as future enhancement

**Priority**: High (SciPy compatibility is a core goal)

## References

- SciPy optimize.minimize docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- Trust-region methods: Conn, Gould, Toint (2000)
- Issue #70: https://github.com/cool-japan/scirs/issues/70
