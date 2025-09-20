# API Freeze Documentation for scirs2-core 1.0

## Overview

This document outlines the frozen API surface for scirs2-core version 1.0. These APIs are guaranteed to remain stable throughout the 1.x version series, following semantic versioning principles.

## API Stability Guarantees

### What We Guarantee

1. **Function Signatures**: Function names, parameter types, and return types will not change
2. **Trait Definitions**: Trait methods and associated types remain stable
3. **Type Definitions**: Public struct and enum definitions remain compatible
4. **Module Structure**: Public module paths remain accessible

### What May Change

1. **Implementation Details**: Internal algorithms may be optimized
2. **Performance Characteristics**: Speed and memory usage may improve
3. **Private APIs**: Non-public interfaces may change without notice
4. **Documentation**: Doc comments may be enhanced or clarified

## Frozen API Categories

### Core Error Handling
- `CoreError` - Main error type for the library
- `CoreResult<T>` - Result type alias for core operations
- `ErrorContext` - Context information for errors
- `ErrorKind` - Error categorization
- `PreciseError` - Detailed error information

### Validation Functions
- `check_finite(value, name)` - Validate finite numbers
- `check_array_finite(array, name)` - Validate finite arrays
- `check_positive(value, name)` - Validate positive values
- `check_shape(array, expected, name)` - Validate array shapes
- `check_in_bounds(index, bounds, name)` - Validate indices

### Numeric Operations
- `NumericOps` trait - Generic numeric operations
- `Complex<T>` - Complex number support
- `Precision` - Precision tracking utilities
- `StableAlgorithms` - Numerically stable implementations

### Array Protocol
- `ArrayProtocol` - Core array interface
- `ArrayLike` - Types that can be treated as arrays
- `IntoArray` - Conversion to arrays
- `ArrayView` - Immutable array views
- `ArrayViewMut` - Mutable array views

### Memory Efficient Operations (Feature: memory_efficient)
- `MemoryMappedArray` - Memory-mapped array storage
- `ChunkedArray` - Chunked array processing
- `LazyArray` - Lazy evaluation arrays
- `OutOfCoreArray` - Out-of-core computation
- `create_mmap()` - Create memory-mapped arrays
- `open_mmap()` - Open existing memory maps
- `chunk_wise_op()` - Chunked operations
- `chunk_wise_reduce()` - Chunked reductions

### SIMD Operations (Feature: simd)
- `SimdOps` trait - SIMD operations interface
- `SimdUnifiedOps` trait - Unified SIMD API
- `PlatformCapabilities` - SIMD feature detection

### Parallel Operations (Feature: parallel)
- `par_chunks()` - Parallel chunk iteration
- `par_chunks_mut()` - Mutable parallel chunks
- `par_range()` - Parallel range iteration
- `par_join()` - Parallel join operation
- `par_scope()` - Scoped parallelism
- `num_threads()` - Thread count query
- `is_parallel_enabled()` - Parallelism check

### Configuration System
- `Config` - Configuration storage
- `ConfigValue` - Configuration values
- `get_config()` - Retrieve configuration
- `set_config_value()` - Update configuration

### Constants
- `math::PI` - Mathematical constant π
- `math::E` - Euler's number
- `physical::C` - Speed of light
- `physical::G` - Gravitational constant

### Resource Discovery
- `SystemResources` - System resource information
- `get_system_resources()` - Query system capabilities
- `get_available_memory()` - Available memory query
- `is_gpu_available()` - GPU availability check

## Migration Guide

### From Beta to 1.0

If you're migrating from the beta versions to 1.0:

1. **Review Breaking Changes**: Check the CHANGELOG for any breaking changes
2. **Update Feature Flags**: Ensure feature flags match the new structure
3. **API Renames**: Some APIs may have been renamed for clarity
4. **Performance Tuning**: Re-benchmark performance-critical code

### Future Compatibility

To ensure your code remains compatible with future 1.x releases:

1. **Use Public APIs Only**: Avoid depending on private implementation details
2. **Feature Detection**: Use feature flags appropriately
3. **Error Handling**: Handle all error variants properly
4. **Version Checking**: Use the version API for runtime checks

## Deprecation Policy

When APIs need to be deprecated:

1. **Warning Period**: Deprecated in minor version (e.g., 1.2)
2. **Migration Guide**: Clear migration path provided
3. **Removal**: Not before next major version (2.0)
4. **Runtime Warnings**: Optional deprecation warnings

## Examples

### Basic Usage
```rust
use scirs2_core::{CoreResult, check_finite, check_positive};

fn safe_divide(a: f64, b: f64) -> CoreResult<f64> {
    check_finite(a, "numerator")?;
    check_finite(b, "denominator")?;
    check_positive(b.abs(), "denominator magnitude")?;
    Ok(a / b)
}
```

### Memory Efficient Operations
```rust
#[cfg(feature = "memory_efficient")]
use scirs2_core::{create_mmap, chunk_wise_op};

#[cfg(feature = "memory_efficient")]
fn process_large_data() -> CoreResult<()> {
    let data = create_mmap::<f64>("large_file.bin", 1_000_000)?;
    chunk_wise_op(&data, |chunk| {
        // Process chunk
        chunk.iter().sum::<f64>()
    })?;
    Ok(())
}
```

### Parallel Processing
```rust
#[cfg(feature = "parallel")]
use scirs2_core::{par_chunks, par_range};

#[cfg(feature = "parallel")]
fn parallel_sum(data: &[f64]) -> f64 {
    par_chunks(data, 1000)
        .map(|chunk| chunk.iter().sum::<f64>())
        .sum()
}
```

## Support

For questions about API stability or migration:

1. **Documentation**: https://docs.rs/scirs2-core
2. **Issue Tracker**: https://github.com/scirs/scirs2-core/issues
3. **Migration Guide**: See MIGRATION.md
4. **Community**: Discord/Forums

## Version History

- **1.0.0** - Initial stable release with frozen API
- **0.1.0-beta.2** - Final beta before API freeze

---

*This document is part of the scirs2-core 1.0 release.*
*Last updated: 2025-06-27*