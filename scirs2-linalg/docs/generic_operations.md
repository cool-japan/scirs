# Type-Generic Linear Algebra Operations

The `scirs2-linalg` library provides type-generic interfaces for linear algebra operations, allowing you to write code that works with different numeric types (f32, f64) while maintaining type safety.

## Usage

```rust
use scirs2_linalg::prelude::*;
use scirs2_linalg::generic::PrecisionSelector;
use ndarray::array;

// Works with f64
let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
let det = gdet(&a.view()).unwrap();
let inv = ginv(&a.view()).unwrap();
let svd = gsvd(&a.view(), false).unwrap();

// Also works with f32
let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let det = gdet(&a.view()).unwrap();
let inv = ginv(&a.view()).unwrap();
```

## Generic Functions

The following generic functions are available:

- `gemm` - Matrix multiplication
- `gemv` - Matrix-vector multiplication
- `gdet` - Determinant
- `ginv` - Matrix inverse
- `gnorm` - Matrix norms
- `gsvd` - Singular Value Decomposition
- `gqr` - QR decomposition
- `geig` - Eigendecomposition
- `gsolve` - Linear system solver

## LinalgScalar Trait

The `LinalgScalar` trait unifies numeric types suitable for linear algebra operations:

```rust
pub trait LinalgScalar:
    Clone
    + Debug
    + Default
    + PartialEq
    + NumAssign
    + Sum
    + for<'a> Sum<&'a Self>
    + ndarray::ScalarOperand
    + 'static
{
    type Real: Float + NumAssign + Sum + Debug + Default + 'static;
    
    fn to_f64(&self) -> Result<f64, LinalgError>;
    fn from_f64(v: f64) -> Result<Self, LinalgError>;
    fn abs(&self) -> Self::Real;
    fn is_zero(&self) -> bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(&self) -> Self;
    fn conj(&self) -> Self;
    fn real(&self) -> Self::Real;
    fn epsilon() -> Self::Real;
}
```

## Automatic Precision Selection

The library includes a `PrecisionSelector` trait for automatic precision selection based on problem characteristics:

```rust
use scirs2_linalg::generic::PrecisionSelector;

// Check if high precision is needed based on condition number
let condition_number = 1e7;
if f32::should_use_high_precision(condition_number) {
    // Use f64 operations
} else {
    // Use f32 operations
}
```

## Example

See the full example in `examples/generic_example.rs`:

```bash
cargo run --example generic_example
```

## Future Extensions

The generic interface is designed to be extended with:

- Complex number support (Complex<f32>, Complex<f64>)
- Extended precision types
- Automatic precision escalation for ill-conditioned problems
- GPU-accelerated types