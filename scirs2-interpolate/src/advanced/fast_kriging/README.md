# Fast Kriging Module

This directory contains the modular implementation of fast kriging algorithms for large spatial datasets.

## Structure

The module is organized into the following components:

- `mod.rs`: Core types and re-exports
- `ordinary.rs`: Ordinary kriging implementation
- `universal.rs`: Universal kriging with trend modeling
- `variogram.rs`: Variogram estimation and modeling
- `covariance.rs`: Covariance functions and distance calculations
- `acceleration.rs`: Acceleration techniques for large datasets

## Implementation Status

This modular structure is a work in progress. The original implementation has been preserved in the parent directory as `fast_kriging_reexports.rs`, which is currently used to provide backward compatibility.

## Planned Enhancements

1. **Performance Optimization**: SIMD acceleration for matrix operations
2. **Memory Efficiency**: Streaming algorithms for very large datasets
3. **GPU Support**: Optional GPU acceleration via CUDA or OpenCL
4. **Distributed Computing**: Support for distributed kriging computations

## Usage

Current usage remains the same as the original module, as we're maintaining compatibility during refactoring:

```rust
use scirs2_interpolate::advanced::fast_kriging::{
    FastKriging, FastKrigingMethod, FastKrigingBuilder, make_local_kriging
};
```

## Roadmap

1. Complete the refactoring with full test coverage
2. Add performance benchmarks to track improvements
3. Implement SIMD acceleration for key operations
4. Add support for new approximation methods
5. Improve documentation with detailed examples

## References

- See `fast_kriging_refactoring_notes.md` for details on the refactoring process
- See `REFACTORING_PLAN.md` for the overall project refactoring guidelines