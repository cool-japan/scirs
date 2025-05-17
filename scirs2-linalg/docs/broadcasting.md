# NumPy-style Broadcasting for Higher-Dimensional Arrays

The `scirs2-linalg` library provides NumPy-style broadcasting support for linear algebra operations on arrays with more than 2 dimensions.

## Overview

Broadcasting allows operations on arrays with different shapes by automatically expanding dimensions to match. This is particularly useful for:

- Batch operations on multiple matrices
- Applying the same operation to multiple data points
- Efficient vectorized computations

## Key Features

- **3D Array Support**: Optimized operations for 3D arrays (batch of matrices)
- **Dynamic Dimensional Arrays**: Support for arrays with arbitrary dimensions
- **Broadcast Compatibility Checking**: Verify if arrays can be broadcast together
- **Batch Matrix Operations**: Matrix multiplication and matrix-vector multiplication

## Usage Examples

### 3D Array Broadcasting

```rust
use ndarray::array;
use scirs2_linalg::prelude::*;

// Batch of 2x2 matrices
let a = array![
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]]
];

// Another batch of matrices
let b = array![
    [[1.0, 0.0], [0.0, 1.0]],
    [[2.0, 0.0], [0.0, 2.0]]
];

// Batch matrix multiplication
let c = broadcast_matmul_3d(&a, &b).unwrap();
```

### Dynamic Dimensional Arrays

```rust
// Convert to dynamic arrays for arbitrary dimensions
let a_dyn = array![[[1.0, 2.0], [3.0, 4.0]]].into_dyn();
let b_dyn = array![[[5.0, 6.0], [7.0, 8.0]]].into_dyn();

// Works with any number of dimensions
let result = broadcast_matmul(&a_dyn, &b_dyn).unwrap();
```

### Broadcasting Different Batch Sizes

```rust
// Batch of 2 matrices
let a = array![
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]]
];

// Single matrix (will be broadcast)
let b = array![[[2.0, 0.0], [0.0, 2.0]]];

// Broadcasting automatically applies b to both matrices in a
let c = broadcast_matmul_3d(&a, &b).unwrap();
```

## Broadcasting Rules

1. **Dimension Compatibility**: Two dimensions are compatible when:
   - They are equal, or
   - One of them is 1

2. **Shape Broadcasting**: Arrays are broadcast together by:
   - Starting from the trailing dimensions
   - Working backwards, checking compatibility
   - Expanding dimensions of size 1 to match

3. **Matrix Operations**: For matrix operations:
   - The last two dimensions are treated as matrices
   - Leading dimensions are broadcast together
   - Matrix dimensions must be compatible for the operation

## API Reference

### Core Functions

- `broadcast_matmul_3d` - Matrix multiplication for 3D arrays
- `broadcast_matmul` - Matrix multiplication for dynamic dimensional arrays
- `broadcast_matvec` - Matrix-vector multiplication for dynamic arrays

### Trait Extensions

- `BroadcastExt` - Trait providing broadcasting utilities:
  - `broadcast_compatible` - Check if two arrays can be broadcast
  - `broadcast_shape` - Get the resulting shape after broadcasting

## Performance Considerations

- 3D operations are optimized for common batch processing scenarios
- Dynamic operations handle arbitrary dimensions but may have overhead
- Consider using specialized 3D functions when dimension is known

## Future Enhancements

- Full NumPy-style broadcasting for all combinations
- GPU acceleration for batch operations
- Automatic parallelization for large batches
- Support for more operations (solve, decompositions)

## Example Program

See the complete example in `examples/broadcast_example.rs`:

```bash
cargo run --example broadcast_example
```

This demonstrates:
- 3D array operations
- Dynamic dimensional arrays
- Broadcasting with different batch sizes
- Compatibility checking