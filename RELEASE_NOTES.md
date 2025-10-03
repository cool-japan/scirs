# Release Notes - SciRS2

## 🚀 v0.1.0-rc.1 (2025-10-03) - Release Candidate 1

### 🎯 First Release Candidate - Production Readiness Verification

This is the first Release Candidate for SciRS2 v0.1.0, focusing on comprehensive platform testing and final preparation for stable release.

#### 🖥️ Platform Compatibility Testing:
- **macOS Support**: Verified on Apple M3 (ARM64), macOS 15.6.1, 24GB RAM
  - ✅ All 9,800+ tests passing with `cargo nextest run --nff --all-features`
  - ✅ Build succeeds with zero warnings
- **Linux Support**: Verified on x86_64 with required system libraries
  - ✅ All 9,800+ tests passing
  - ✅ Complete functionality with proper environment configuration
- **Linux + CUDA**: Verified on x86_64 with NVIDIA GPU
  - ✅ All 9,800+ tests passing
  - ✅ GPU acceleration features working correctly
- **Windows Support**: Partial support on Windows 11 Pro x86_64
  - ✅ `cargo build` succeeds
  - ⚠️ Some crate tests fail (full compatibility planned for v0.2.0)

#### 📝 Documentation Updates:
- **Platform Compatibility Guide**: Added comprehensive platform testing information to README
- **Version Consistency**: Updated all version references from beta.4 to rc.1
- **Release Documentation**: Updated lib.rs documentation across all 23 crates
- **Roadmap Clarification**: Next release is v0.1.0 (stable), not beta.5

#### 🔧 Release Preparation:
- **Version Bumps**: All crates updated to 0.1.0-rc.1
- **Dependency Updates**: Workspace dependencies aligned to rc.1
- **Date Updates**: Release dates updated to October 03, 2025
- **Build Verification**: All modules compile cleanly with zero warnings

#### 🏗️ Build System Improvements:
- **System Library Priority**: Removed `openblas-src` from workspace dependencies
  - macOS: Uses Accelerate framework (system BLAS, no source build required)
  - Linux: Uses system OpenBLAS (openblas-system feature, no source build required)
  - Windows: Uses system BLAS libraries where available
- **Faster Builds**: Eliminated source compilation of BLAS libraries, significantly reducing build times
- **Platform Optimization**: Leverages platform-specific optimized BLAS implementations
- **Simplified Dependencies**: Removed `intel-mkl-src` to avoid conflicts with system BLAS

#### 🎯 Code Quality & Architecture Overhaul:
- **100% POLICY Compliance**: All 23 crates now fully comply with SciRS2 POLICY
  - Only `scirs2-core` has direct external dependencies
  - All other crates use `scirs2_core::` abstractions
  - ~600+ import statements updated to unified architecture
  - ~100+ external dependencies centralized in scirs2-core
- **Zero Warnings Achievement**: Complete workspace builds with zero warnings
  - Fixed parallel execution errors (Sync trait bounds, Rayon patterns)
  - Added 100+ documentation comments for GPU and memory management
  - Resolved POLICY violations and conditional compilation issues
  - All 23 crates (lib, tests, examples) compile cleanly
- **Code Cleanup**: Significant TODO/FIXME reduction
  - Removed obsolete TODO markers and implemented pending features
  - Refactored complex code sections for better maintainability
  - Enhanced error handling and validation across modules
- **Massive Scale**: 3,016 files modified in comprehensive RC.1 transformation
  - 18,081 insertions across all changes
  - 15,487 deletions for cleanup and refactoring
  - Major commits: POLICY migration (470 files), zero warnings fixes, TODO/FIXME cleanup

#### 📋 Technical Details:
- **Release Date**: October 03, 2025
- **Test Coverage**: 9,800+ tests passing on supported platforms
- **Breaking Changes**: None (API compatible with beta.4)
- **Next Release**: v0.1.0 (stable) - Q4 2026

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-rc.1"
scirs2 = "0.1.0-rc.1"
```

#### 🎉 Release Candidate Status:
This RC.1 release marks the transition from beta to release candidate status. After community testing and feedback, the next major release will be v0.1.0 stable with long-term API stability guarantees.

---

## 🚀 v0.1.0-beta.4 (2025-10-01) - Release Stabilization

### 🎯 Focus on Stability and Documentation

This release concentrates on stabilizing the ecosystem architecture established in beta.3 and preparing for production deployment.

#### 📝 Documentation and Release Process:
- **Version Management**: Comprehensive version updates across all documentation files
- **Release Workflow**: Streamlined process for version updates and release preparation
- **Documentation Consistency**: Ensured all references point to current beta.4 release
- **API Documentation**: Maintained up-to-date examples and usage guides

#### 🔧 Stability Improvements:
- **Ecosystem Refinement**: Continued implementation of SciRS2 POLICY abstractions
- **Build Verification**: All modules compile cleanly with zero warnings
- **Test Coverage**: Maintained 9,800+ passing tests across all modules
- **Platform Support**: Verified compatibility on Linux, macOS, and Windows

#### 🏗️ Architectural Consistency:
- **Policy Compliance**: Ongoing migration to scirs2-core abstractions
- **API Stability**: No breaking changes from beta.3
- **Performance Validation**: Verified benchmark results across platforms
- **SciPy Compatibility**: Maintained compatibility while improving Rust idioms

#### 📋 Technical Details:
- **Release Date**: October 01, 2025
- **Breaking Changes**: None (API compatible with beta.3)
- **New Features**: Documentation and process improvements
- **Deprecated**: None

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-beta.4"
scirs2 = "0.1.0-beta.4"
```

---

## 🚀 v0.1.0-beta.4 (2025-09-29) - SciRS2 POLICY & Major Modernization

### 🎯 SciRS2 Ecosystem Architecture Implementation

This release establishes the foundational architecture for the SciRS2 ecosystem with comprehensive policy framework and major dependency modernization.

#### 🏗️ SciRS2 POLICY Framework:
- **Layered Abstraction Architecture**: Established core-only external dependency policy
- **Ecosystem Consistency**: All non-core crates must use scirs2-core abstractions
- **Policy Documentation**: Complete [SciRS2 Ecosystem Policy](SCIRS2_POLICY.md) with guidelines and enforcement strategies
- **Migration Strategy**: Phased approach for systematic refactoring to scirs2-core abstractions

#### 🔧 Major Dependency Updates:
- **Comprehensive Modernization**: Updated all dependencies to latest available versions
- **156 Files Changed**: Extensive updates across the entire workspace
- **Enhanced Performance**: Improved SIMD operations, numerical algorithms, and spatial computations
- **Advanced Random Generation**: Enhanced ecosystem integration with cutting-edge MCMC and neural sampling

#### 🖥️ GPU and Platform Enhancements:
- **CUDA/Linux Optimization**: Significant improvements to CUDA backend for Linux platforms
- **WebGPU Backend**: Major enhancements (333+ lines) for better cross-platform GPU support
- **Memory-Mapped Operations**: Advanced chunking, zero-copy serialization, and large dataset handling
- **Sparse Matrix GPU**: Enhanced GPU operation support for sparse matrix computations

#### 📊 Advanced Feature Stabilization:
- **Real-time Processing**: Improved streaming capabilities in scirs2-io with better error handling
- **Distributed Computing**: Enhanced distributed processing in scirs2-transform with fault tolerance
- **Performance Validation**: Comprehensive SIMD performance validation with automated benchmarking
- **High-Dimensional Interpolation**: Enhanced parallel algorithms for advanced interpolation

#### 📋 Technical Details:
- **Policy Effective Date**: v0.1.0-beta.4
- **Benefits**: Consistent APIs, centralized version control, type safety, maintainability
- **Migration Plan**: Automated tooling for transitioning to scirs2-core abstractions
- **Documentation**: Complete usage guidelines and implementation examples

#### 🎯 Impact:
- **Ecosystem Consistency**: Foundation for unified SciRS2 development patterns
- **Performance**: Better optimization opportunities through centralized abstractions
- **Maintainability**: Simplified dependency management and version control
- **Future-Proofing**: Scalable architecture for continued ecosystem growth

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-beta.4"
scirs2 = "0.1.0-beta.4"
```

---

## 🔧 v0.1.0-beta.2 (2025-09-16) - Critical Compilation Fixes

### 🚨 Emergency Release - Fixes crates.io Compilation Errors

This is a critical hotfix release that resolves all compilation errors present in v0.1.0-beta.2 when downloaded from crates.io.

#### 🔥 Fixed Compilation Errors (100% resolved):
- **Variable Name Inconsistencies**: Fixed `chunk_size`/`chunksize`, `op_name`/`opname`, `target_unit`/`targetunit` mismatches
- **Undefined Variable References**: Fixed unresolved variables in batch conversion functions (`sequential`, `simd`, `parallel`, `simd_parallel`)
- **Pattern Match Errors**: Corrected pattern matching in SIMD conversion loops (`for (0, &val)` → `for (i, &val)` with proper indexing)
- **Function Name Conflicts**: Resolved duplicate `center()` function definitions by renaming constructor to `centered()`
- **Type Field References**: Fixed `type_info` field reference consistency

#### 📋 Technical Details:
- **Files Modified**: `batch_conversions.rs`, `types.rs`, `dynamic_dispatch.rs`
- **Total Errors Fixed**: 20+ compilation errors reduced to zero
- **Build Verification**: ✅ `cargo build` successful
- **Lint Check**: ✅ `cargo clippy` with zero warnings
- **Publication Ready**: ✅ `cargo publish --dry-run` successful

---

## 🎉 v0.1.0-beta.1 - First Beta Release!

We are excited to announce the first beta release of SciRS2, a comprehensive scientific computing and AI/ML infrastructure in Rust. After months of development, we've reached a significant milestone with over 2 million lines of code and 9,800+ tests.

## ✨ Key Features

### Scientific Computing Core
- **Linear Algebra**: Complete matrix operations, decompositions, eigensolvers
- **Statistics**: Full distribution suite, descriptive statistics, hypothesis tests
- **Optimization**: Unconstrained/constrained optimization, root finding, least squares
- **Integration**: Numerical integration, ODE solvers, boundary value problems
- **FFT**: High-performance Fast Fourier Transform with GPU acceleration
- **Signal Processing**: Filtering, spectral analysis, wavelet transforms
- **Special Functions**: Bessel, gamma, elliptic, and more mathematical functions

### AI/ML Infrastructure  
- **Automatic Differentiation**: Reverse-mode autodiff engine for gradient computation
- **Neural Networks**: Layer abstractions, optimizers, and model building blocks
- **Computer Vision**: Image processing, feature detection, streaming operations
- **Time Series**: Analysis and forecasting tools
- **Metrics**: Comprehensive ML evaluation metrics

### Performance Features
- **SIMD Optimization**: Hardware-accelerated operations using unified SIMD ops
- **GPU Support**: CUDA acceleration with automatic CPU fallback
- **Parallel Processing**: Multi-core computation for intensive operations
- **Memory Efficiency**: Optimized memory management for large datasets

## 📊 Project Statistics

- **Total Lines of Code**: 2,000,000+
- **Number of Modules**: 24 specialized crates
- **Test Coverage**: 9,300+ tests
- **Passing Tests**: All regular tests passing ✅
- **Ignored Tests**: ~600 (benchmarks and hardware-specific)

## 🔧 Recent Improvements

### GPU Test Adaptations
Fixed 12 GPU-dependent tests in the FFT module:
- CUDA initialization tests now gracefully handle missing hardware
- GPU kernel tests provide mock implementations for CPU-only systems
- Multi-GPU tests adapt to available hardware configurations
- Specialized hardware tests (FPGA/ASIC) use appropriate fallbacks

### Test Infrastructure
- Added proper test helpers for graph context management
- Improved test organization and categorization
- Clear separation between functional tests and benchmarks

## ⚠️ Known Limitations

### Autograd Module
- Some gradient shape propagation limitations in complex operations (Issue #1)
- Graph context requirements for certain stability tests
- Workarounds and helper functions are provided

### Unimplemented Features
These features are planned for upcoming releases:
- Cholesky decomposition (0.2.0)
- Thin Plate Spline solver (0.2.0)
- Additional advanced linear algebra decompositions

### Performance Tests
- 404 benchmark tests are ignored by default to optimize CI build times
- Run `cargo test -- --ignored` for full benchmark suite

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.1.0-rc.1"  # Use latest version

# Or select specific modules:
scirs2-linalg = "0.1.0-rc.1"
scirs2-stats = "0.1.0-rc.1"
scirs2-autograd = "0.1.0-rc.1"
```

## 🚀 Quick Start

```rust
use scirs2::prelude::*;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and manipulate matrices
    let a = Array2::eye(3);
    let (u, s, vt) = scirs2::linalg::decomposition::svd(&a)?;
    
    // Statistical computations
    let normal = scirs2::stats::distributions::normal::Normal::new(0.0, 1.0)?;
    let samples = normal.random_sample(1000, None)?;
    
    Ok(())
}
```

## 🔄 Migration from SciPy

SciRS2 provides SciPy-compatible APIs where reasonable, making migration straightforward:

| SciPy | SciRS2 |
|-------|--------|
| `scipy.linalg.svd()` | `scirs2::linalg::decomposition::svd()` |
| `scipy.stats.norm()` | `scirs2::stats::distributions::normal::Normal` |
| `scipy.optimize.minimize()` | `scirs2::optimize::minimize()` |
| `scipy.fft.fft()` | `scirs2::fft::fft()` |

## 🐛 Bug Reports and Feedback

This is a beta release, and we welcome your feedback! Please report issues at:
https://github.com/cool-japan/scirs/issues

## 🔮 What's Next

### Version 0.2.0 (Planned)
- Complete Cholesky decomposition implementation
- Thin Plate Spline solver
- Enhanced GPU backend support
- Performance optimizations based on user feedback

### Long-term Roadmap
- Python bindings for easier migration
- WebAssembly support for browser deployment
- Extended hardware support (ARM, RISC-V)
- Domain-specific extensions (finance, bioinformatics)

## 🙏 Acknowledgments

Thank you to all contributors who made this release possible. Special thanks to:
- The Rust scientific computing community
- Early adopters and testers
- The SciPy and NumPy teams for inspiration

## 📜 License

Dual-licensed under MIT and Apache 2.0.

---

**Note**: This is a beta release. While core functionality is stable and well-tested, some features are still under development. Production use should be carefully evaluated based on your specific requirements.

For detailed documentation, visit: https://docs.rs/scirs2/0.1.0-rc.1/