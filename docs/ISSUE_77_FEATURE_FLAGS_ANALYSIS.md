# Issue #77: Feature Flags Analysis and Migration Plan

## Problem Statement

Currently, all non-core SciRS2 crates force-enable multiple `scirs2-core` features in their `[dependencies]` section. This prevents users from disabling these features even with `default-features = false`.

### Example Problem

In `scirs2-linalg/Cargo.toml`:
```toml
[dependencies]
scirs2-core = { workspace = true, features = ["linalg", "parallel", "simd", "random", "array"] }
```

When a user tries to use `scirs2-linalg` with minimal features:
```toml
[dependencies]
scirs2-linalg = { version = "0.1.0", default-features = false }
```

The features `["linalg", "parallel", "simd", "random", "array"]` are **still enabled** because they are hardcoded in the dependency declaration.

## Impact Analysis

### Affected Crates (All 20+ crates)

| Crate | Forced Features |
|-------|-----------------|
| scirs2-autograd | parallel, array, random |
| scirs2-cluster | parallel, simd, array, random |
| scirs2-datasets | cache, random, gpu, parallel, array |
| scirs2-fft | simd, parallel, gpu, validation, random, array |
| scirs2-graph | parallel, simd, random, array |
| scirs2-integrate | random, array |
| scirs2-interpolate | simd, parallel, random, array |
| scirs2-io | parallel, simd, gpu, array, random |
| scirs2-linalg | linalg, parallel, simd, random, array |
| scirs2-metrics | parallel, simd, array, random |
| scirs2-ndimage | validation, memory_efficient, parallel, array, random |
| scirs2-neural | random, array, parallel |
| scirs2-optimize | simd, parallel, linalg, gpu, random, array |
| scirs2-series | linalg, array, random |
| scirs2-signal | array, random, parallel |
| scirs2-sparse | array, random, gpu |
| scirs2-spatial | simd, parallel, linalg, random, types |
| scirs2-special | validation, simd, array, random |
| scirs2-stats | validation, parallel, simd, linalg, random, array |
| scirs2-text | parallel, simd, array, random |
| scirs2-transform | parallel, simd, array, random |
| scirs2-vision | parallel, simd, array, random |

## Proposed Solution

### Phase 1: Categorize Features (v0.1.0-rc.3)

For each crate, categorize features into:

#### Mandatory Features
Features that are **always required** for the crate to function:
- `array` - Nearly all crates need ndarray functionality
- `random` - Most scientific computing crates need RNG

#### Optional Features
Features that should be user-controllable:
- `simd` - SIMD acceleration (optional performance enhancement)
- `parallel` - Parallel processing (optional performance enhancement)
- `gpu` - GPU acceleration (requires hardware)
- `linalg` - BLAS/LAPACK (heavy dependency)
- `validation` - Input validation (optional safety checks)

### Phase 2: Refactor Cargo.toml Structure (v0.2.0)

**Before:**
```toml
[dependencies]
scirs2-core = { workspace = true, features = ["parallel", "simd", "random", "array"] }

[features]
default = ["simd"]
simd = []
parallel = []
```

**After:**
```toml
[dependencies]
# Only mandatory features here
scirs2-core = { workspace = true, features = ["array"] }

[features]
# Default features can be disabled with default-features = false
default = ["simd", "parallel"]

# Each feature explicitly enables corresponding scirs2-core feature
simd = ["scirs2-core/simd"]
parallel = ["scirs2-core/parallel"]
random = ["scirs2-core/random"]
gpu = ["scirs2-core/gpu"]
```

### Phase 3: Update Documentation (v0.2.0)

Update all README files and documentation to explain:
1. Available feature flags
2. Default vs optional features
3. How to minimize dependencies
4. Performance implications of disabling features

### Example: scirs2-linalg Refactoring

**Current (v0.1.0-rc.2):**
```toml
[dependencies]
scirs2-core = { workspace = true, features = ["linalg", "parallel", "simd", "random", "array"] }

[features]
default = ["linalg", "simd"]
```

**Proposed (v0.2.0):**
```toml
[dependencies]
# Minimal required features only
scirs2-core = { workspace = true, features = ["array"] }

[features]
# Default features - can be disabled by users
default = ["linalg", "simd"]

# Core functionality
linalg = ["scirs2-core/linalg"]          # BLAS/LAPACK support
array = ["scirs2-core/array"]            # Always enabled via deps

# Performance features
simd = ["scirs2-core/simd"]              # SIMD acceleration
parallel = ["scirs2-core/parallel"]      # Parallel processing

# Optional functionality
random = ["scirs2-core/random"]          # Random number generation
gpu = ["scirs2-core/gpu"]                # GPU acceleration
autograd = ["dep:scirs2-autograd"]       # Automatic differentiation
```

## Migration Strategy

### For v0.1.0-rc.3 (Current Release)

1. **Document the issue**: Create this analysis document
2. **Communicate to users**: Add note in CHANGELOG.md about the limitation
3. **Plan for v0.2.0**: Schedule feature flag refactoring for next major version

### For v0.2.0 (Breaking Changes Allowed)

1. **Phase 1 (Week 1-2)**: Analyze each crate to identify mandatory vs optional features
2. **Phase 2 (Week 3-4)**: Refactor all Cargo.toml files
3. **Phase 3 (Week 5)**: Update documentation and migration guide
4. **Phase 4 (Week 6)**: Test with various feature combinations
5. **Phase 5 (Week 7)**: Release with clear migration guide

## Benefits

After migration, users will be able to:

```toml
# Minimal build - only core array functionality
[dependencies]
scirs2-linalg = { version = "0.2.0", default-features = false }

# Custom feature selection
[dependencies]
scirs2-linalg = { version = "0.2.0", default-features = false, features = ["linalg", "parallel"] }

# All features (same as before)
[dependencies]
scirs2-linalg = { version = "0.2.0" }  # or with features = ["default"]
```

This provides:
- **Reduced build times**: Users can exclude heavy dependencies (BLAS, GPU, etc.)
- **Smaller binaries**: Exclude unused SIMD or GPU code
- **Better control**: Fine-grained feature selection
- **Compatibility**: Better support for embedded/WASM targets

## Backward Compatibility

This is a **breaking change** because:
- Users relying on implicit feature enablement may need to explicitly enable features
- Build configurations may need adjustment
- Documentation and examples may need updates

Therefore, this should be done in a major version bump (v0.2.0).

## Recommendation

**For v0.1.0-rc.3:**
- Document this limitation in KNOWN_LIMITATIONS.md
- Add a note in README.md about feature flags
- Create GitHub issue comment explaining the plan

**For v0.2.0:**
- Implement the full refactoring
- Provide comprehensive migration guide
- Update all documentation and examples

## Implementation Checklist

- [ ] Document issue in KNOWN_LIMITATIONS.md
- [ ] Add README note about current feature flag behavior
- [ ] Create feature categorization matrix for all crates
- [ ] Refactor Cargo.toml files (v0.2.0)
- [ ] Update documentation (v0.2.0)
- [ ] Create migration guide (v0.2.0)
- [ ] Test all feature combinations (v0.2.0)
- [ ] Update CI to test minimal feature sets (v0.2.0)

## References

- Issue #77: https://github.com/cool-japan/scirs/issues/77
- Cargo Book - Features: https://doc.rust-lang.org/cargo/reference/features.html
- Best Practices for Feature Flags: https://rust-lang.github.io/api-guidelines/
