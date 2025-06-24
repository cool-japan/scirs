# SciRS2 Core Quick Reference

This is a quick reference guide to the most commonly used features in the `scirs2-core` module.

## ✅ Validation Functions

```rust
use scirs2_core::validation::{
    // Value validation
    check_positive(value, "parameter_name"),            // value > 0
    check_non_negative(value, "parameter_name"),        // value >= 0
    check_probability(value, "parameter_name"),         // 0 <= value <= 1
    check_in_bounds(value, min, max, "parameter_name"), // min <= value <= max
    check_finite(value, "parameter_name"),              // value is finite

    // Array validation
    check_probabilities(&arr, "parameter_name"),        // all elements in [0, 1]
    check_probabilities_sum_to_one(&arr, "name", tol),  // sum(arr) == 1
    check_array_finite(&arr, "parameter_name"),         // all values finite
    
    // Shape validation
    check_1d(&arr, "parameter_name"),                   // array is 1D
    check_2d(&arr, "parameter_name"),                   // array is 2D
    check_shape(&arr, &expected, "parameter_name"),     // array has expected shape
    check_same_shape(&a, &b, "a", "b"),                 // arrays have same shape
    check_square(&mat, "parameter_name"),               // matrix is square
};
```

## 🛠️ Utility Functions 

```rust
use scirs2_core::utils::{
    // Array comparison
    is_close(a, b, abs_tol, rel_tol),                   // scalar comparison
    points_equal(&point1, &point2, tol),                // slice comparison
    arrays_equal(&array1, &array2, tol),                // array comparison
    
    // Array generation
    linspace(start, end, num),                          // linear space
    logspace(start, end, num, base),                    // logarithmic space
    arange(start, stop, step),                          // range with step
    
    // Array manipulation
    fill_diagonal(array, val),                          // set matrix diagonal
    pad_array(&arr, &[(1, 2)], "constant", val),        // pad array
    get_window("hamming", length, periodic),            // window function
    normalize(&vector, "energy"),                       // normalize (energy, peak, sum, max)
    
    // Element-wise operations
    maximum(&a, &b),                                    // element-wise max
    minimum(&a, &b),                                    // element-wise min
    
    // Numerical calculus
    differentiate(x, h, |x| Ok(f(x))),                  // derivative at x
    integrate(a, b, n, |x| Ok(f(x))),                   // integral from a to b
    
    // Other utilities
    prod(vec![1, 2, 3]),                                // product
    all(vec, |x| predicate(x)),                         // check all satisfy
    any(vec, |x| predicate(x)),                         // check any satisfies
};
```

## 🚀 SIMD Operations 

```rust
use scirs2_core::simd::{
    simd_add(&a, &b, &mut c),                           // c = a + b
    simd_subtract(&a, &b, &mut c),                      // c = a - b
    simd_multiply(&a, &b, &mut c),                      // c = a * b
    simd_divide(&a, &b, &mut c),                        // c = a / b
    simd_min(&a, &b, &mut c),                           // c = min(a, b)
    simd_max(&a, &b, &mut c),                           // c = max(a, b)
};
```

## 💾 Caching

```rust
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};
use std::cell::RefCell;

// Option 1: Basic cache
let mut cache = TTLSizedCache::<K, V>::new(size, ttl);
cache.insert(key, value);
if let Some(val) = cache.get(&key) { /* use val */ }

// Option 2: Builder pattern
let cache = RefCell::new(
    CacheBuilder::new()
        .with_size(100)       // max 100 items
        .with_ttl(3600)       // 1 hour expiry
        .build_sized_cache()
);
```

## ⚠️ Error Handling

```rust
// 1. Add CoreError to your module's error enum
use scirs2_core::error::CoreError;

#[derive(Debug, Error)]
pub enum ModuleError {
    // ...other errors...
    
    #[error("{0}")]
    CoreError(#[from] CoreError),  // Auto-convert from CoreError
}

// 2. Use core error types
use scirs2_core::{CoreResult, value_err_loc};

fn calculate(x: f64) -> CoreResult<f64> {
    if x < 0.0 {
        return Err(value_err_loc!("Expected non-negative value, got {}", x));
    }
    Ok(x.sqrt())
}

// 3. Convert to your module's error type
fn module_function(x: f64) -> Result<f64, ModuleError> {
    let value = calculate(x).map_err(ModuleError::from)?;
    Ok(value)
}
```

## 🔄 Parallel Processing

```rust
use scirs2_core::parallel::{
    parallel_map(&data, |x| expensive_function(x)),     // parallel map
    parallel_for_each(&mut data, |x| modify(x)),        // parallel for_each
    parallel_reduce(&data, || 0, |acc, x| acc + x),     // parallel reduce
};
```

## ⚙️ Configuration System

```rust
use scirs2_core::config::{Config, set_global_config, get_global_config};

// Setup global config
let mut config = Config::default();
config.set_precision(1e-10);
config.set_max_iterations(1000);
set_global_config(config);

// Access in any function
fn compute() {
    let config = get_global_config();
    let precision = config.precision();
    // ...
}
```

## Feature Flags

Add only the features you need in your `Cargo.toml`:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["validation", "simd", "parallel", "cache"] }
```

For more details, see [Core Module Usage Guidelines](../docs/core_module_usage.md).