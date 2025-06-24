# Parallel Abstractions in scirs2-core

## Overview

The `scirs2-core::parallel_ops` module provides a unified interface for parallel operations across the SciRS2 project. It wraps Rayon functionality when the `parallel` feature is enabled and provides sequential fallbacks when it's disabled.

## Purpose

This abstraction layer serves several purposes:

1. **Policy Compliance**: Enforces the project policy that modules should use core abstractions rather than direct dependencies
2. **Feature Flexibility**: Allows parallel processing to be toggled via feature flags
3. **Graceful Degradation**: Provides sequential fallbacks when parallel processing is disabled
4. **Consistent API**: Ensures consistent parallel patterns across all SciRS2 modules

## Design Approach

The module takes a pragmatic approach:

- When `parallel` feature is enabled: Directly re-exports `rayon::prelude::*`
- When `parallel` feature is disabled: Provides trait-compatible sequential fallbacks

This design ensures:
- Zero overhead when using parallel features
- Full compatibility with existing Rayon code
- Smooth transition between parallel and sequential execution

## Usage

### For Module Authors

Replace direct Rayon imports:
```rust
// Old - direct dependency
use rayon::prelude::*;

// New - use core abstractions
use scirs2_core::parallel_ops::*;
```

### In Cargo.toml

Remove direct rayon dependency:
```toml
[dependencies]
# Remove this:
# rayon = { workspace = true }

# Keep this:
scirs2-core = { workspace = true, features = ["parallel"] }
```

### Code Example

```rust
use scirs2_core::parallel_ops::*;

// Works with or without the parallel feature
let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| x * x)
    .collect();

// Check if parallel is enabled
if is_parallel_enabled() {
    println!("Using {} threads", num_threads());
}
```

## Helper Functions

The module provides additional helper functions:

- `par_range(start, end)` - Create a parallel iterator from a range
- `par_chunks(slice, size)` - Process slices in parallel chunks
- `par_scope(closure)` - Execute in a parallel scope
- `par_join(a, b)` - Execute two closures in parallel
- `is_parallel_enabled()` - Check if parallel processing is available
- `num_threads()` - Get the number of threads used for parallel operations

## Migration Status

### Completed Modules
- ✅ scirs2-spatial - Successfully migrated to use core parallel abstractions
- ✅ scirs2-core - Provides the parallel abstractions

### Pending Modules
- 🔄 scirs2-optimize - Already migrated SIMD, parallel migration pending
- 🔄 scirs2-fft - Already migrated SIMD, parallel migration pending
- 🔄 Other modules - To be migrated as needed

## Benefits

1. **Simplified Dependencies**: Modules no longer need direct rayon dependency
2. **Consistent Behavior**: All modules behave consistently with feature flags
3. **Future Flexibility**: Easy to switch parallel backends if needed
4. **Testing**: Easier to test both parallel and sequential code paths
5. **Policy Compliance**: Follows the strict project policy for using core abstractions