# Known Limitations - SciRS2 v0.1.0-rc.2

This document describes known limitations and issues in the current release of SciRS2.

## Feature Flags (Issue #77)

**Status**: Documented, planned for v0.2.0

**Problem**: Non-core packages force-enable certain `scirs2-core` features, preventing users from disabling them.

### Current Behavior

When using SciRS2 crates, certain features are force-enabled even when specifying `default-features = false`. For example:

```toml
# User's Cargo.toml
[dependencies]
scirs2-linalg = { version = "0.1.0-rc.2", default-features = false }
```

The features `["linalg", "parallel", "simd", "random", "array"]` are **still enabled** because they are hardcoded in `scirs2-linalg`'s dependency declaration.

### Impact

- Users cannot minimize build dependencies by disabling features
- Longer build times for users who don't need all features
- Larger binary sizes
- Incompatibility with some embedded/WASM targets that don't support certain features

### Workaround

Currently, there is no workaround. All listed features will be enabled when using the crate.

### Solution Timeline

This will be addressed in **v0.2.0** with a comprehensive refactoring of feature flags across all crates. The refactoring will:

1. Move optional features from `[dependencies]` to `[features]`
2. Allow users to selectively enable/disable features
3. Provide minimal feature sets for embedded/WASM targets
4. Maintain backward compatibility through sensible defaults

For detailed analysis and migration plan, see [docs/ISSUE_77_FEATURE_FLAGS_ANALYSIS.md](docs/ISSUE_77_FEATURE_FLAGS_ANALYSIS.md).

## Python Bindings (ndarray 0.17 Incompatibility)

**Status**: Temporarily unavailable, waiting for upstream fix

**Problem**: The `numpy` Rust crate (v0.27.0) only supports ndarray < 0.17, but SciRS2 has migrated to ndarray 0.17.1.

### Current Behavior

Python bindings are feature-gated and will cause build errors if enabled:

```bash
# This will fail
cargo build --all-features

# This works (default features only)
cargo build
```

### Affected Crates

- `scirs2-cluster` - `pyo3` feature
- `scirs2-series` - `python` feature

### Impact

- Python bindings (`pyo3`, `python` features) cannot be used
- Default builds work fine (Python bindings not enabled by default)
- CI/CD should avoid `--all-features` flag

### Workaround

**For now**: Do not enable `pyo3` or `python` features

```toml
# Don't do this
[dependencies]
scirs2-cluster = { version = "0.1.0-rc.2", features = ["pyo3"] }  # ❌ Will fail

# Do this instead
[dependencies]
scirs2-cluster = "0.1.0-rc.2"  # ✅ Works (default features)
```

### Solution Timeline

**v0.2.0**: Re-enable Python bindings when one of the following occurs:
1. `numpy` crate releases version with ndarray 0.17 support (Issue #76 dependency)
2. We implement direct PyO3 bindings without the `numpy` crate
3. We add conditional compilation for ndarray 0.16 vs 0.17

**Tracking**: This is blocked by upstream `numpy` crate compatibility with ndarray 0.17.

### Technical Details

The `numpy` Rust crate depends on `ndarray ^0.15, < 0.17`. When SciRS2 uses ndarray 0.17, Cargo includes both:
- ndarray 0.17.1 (for SciRS2)
- ndarray 0.16.1 (for numpy crate)

This causes type mismatches because `ArrayBase` from 0.16 and 0.17 are different types.

## Platform-Specific Issues

### Windows Platform

**Status**: Partially supported, full support planned for v0.2.0

- **OpenBLAS/BLAS Runtime Errors**: Some tests fail on Windows 11 Pro due to OpenBLAS/BLAS library issues
- **Build Status**: All subcrates build successfully with `cargo build`
- **Test Status**: Most tests pass, but BLAS-dependent tests may encounter runtime errors

### BLAS/LAPACK Dependencies

**Status**: Requires system libraries, documented in README.md

**Problem**: Users encounter linking errors when system BLAS/LAPACK libraries are not installed.

**Solution**: Install platform-specific libraries before building (see [README.md Installation section](README.md#system-dependencies))

## SciRS2 POLICY Implementation

**Status**: Framework complete, full migration ongoing

- **Policy Established**: Complete SciRS2 POLICY framework with layered abstraction architecture (v0.1.0-beta.4)
- **Core Abstractions Complete**: scirs2-core provides comprehensive abstractions for rand, ndarray, and all dependencies
- **Migration Status**: All 23 crates are POLICY-compliant (v0.1.0-rc.2)
- **Backward Compatibility**: Direct usage still works but core abstractions are recommended for new code

## Autograd Module

**Status**: Stable with known limitations

- **Gradient Shape Propagation**: Some complex operations may have limitations in gradient shape inference (Issue #1)
- **Graph Context Requirements**: Some stability tests require proper graph context initialization

## Unimplemented Features

The following features are planned for future releases:

### v0.2.0
- **Cholesky decomposition** - Advanced linear algebra
- **Thin Plate Spline solver** - Interpolation
- **Feature Flags Refactoring** - User-controllable features (Issue #77)
- **Full Windows Support** - Complete BLAS/LAPACK compatibility

### Future Versions
- Some advanced linear algebra decompositions
- Additional special functions
- Extended GPU backend support

## Performance Tests

**Status**: Working as intended

Benchmark and performance tests are excluded from regular CI runs (404 tests marked as ignored) to optimize build times.

**To run all tests including benchmarks:**
```bash
cargo test -- --ignored
# or with nextest
cargo nextest run --all-features
```

## Hardware-Dependent Features

**Status**: Working with fallbacks

- GPU acceleration features require compatible hardware and drivers
- Tests automatically fall back to CPU implementations when GPU is unavailable
- Specialized hardware support (FPGA, ASIC) uses mock implementations when hardware is not present

## Test Coverage

**Current Status (v0.1.0-rc.2)**:
- Total tests: 9,300+ across all modules
- Regular CI tests: All passing ✅
- Performance tests: Included in full test suite (run with `--all-features`)

---

## Reporting Issues

For the most up-to-date information on limitations and ongoing development, please check our [GitHub Issues](https://github.com/cool-japan/scirs/issues).

To report a new issue:
1. Check existing issues to avoid duplicates
2. Provide a minimal reproducible example
3. Include your platform, Rust version, and SciRS2 version
4. Describe expected vs actual behavior

## Contributing

We welcome contributions to address these limitations! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
