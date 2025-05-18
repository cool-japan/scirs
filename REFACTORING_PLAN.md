# SCIRS Refactoring Plan

This document outlines a plan for refactoring the largest files in the SCIRS codebase to improve maintainability, readability, and organization.

## Top 20 Largest Files

Based on file size analysis, these are the largest files in the codebase that would benefit from refactoring:

1. **scirs2-neural/src/linalg/batch_operations.rs** (148KB, 4416 lines)
2. **scirs2-series/src/trends.rs** (108KB, 3340 lines)
3. **scirs2-autograd/src/tensor_ops/mod.rs** (84KB, 3093 lines)
4. **scirs2-metrics/src/regression.rs** (96KB, 2837 lines)
5. **scirs2-metrics/src/clustering.rs** (96KB, 2831 lines)
6. **scirs2-sparse/src/linalg.rs** (80KB, 2711 lines)
7. **scirs2-core/src/ndarray_ext/stats.rs** (80KB, 2495 lines)
8. **scirs2-fft/src/hfft.rs** (84KB, 2293 lines)
9. **scirs2-signal/src/dwt.rs** (68KB, 2215 lines)
10. **scirs2-fft/src/fft.rs** (76KB, 2194 lines)
11. **scirs2-spatial/src/collision.rs** (64KB, 2090 lines)
12. **scirs2-special/src/bessel.rs** (68KB, 2087 lines)
13. **scirs2-interpolate/src/advanced/fast_kriging.rs** (72KB, 2071 lines)
14. **scirs2-signal/src/bss.rs** (64KB, 2007 lines)
15. **scirs2-series/src/decomposition.rs** (64KB, 1916 lines)
16. **scirs2-neural/src/utils/evaluation.rs** (64KB, 1806 lines)
17. **scirs2-neural/src/layers/recurrent.rs** (64KB, 1850 lines)
18. **scirs2-text/src/spelling.rs** (60KB, 1825 lines)
19. **scirs2-spatial/src/rtree.rs** (60KB, 1806 lines)
20. **scirs2-optimize/src/sparse_numdiff.rs** (60KB, ~1800 lines)

## Refactoring Guidelines

The following principles should guide the refactoring process:

1. **Modular Structure**: Break down large files into smaller, focused modules.
2. **Single Responsibility**: Each module should have a clear, single responsibility.
3. **Consistent Interface**: Maintain consistent public interfaces during refactoring.
4. **Comprehensive Tests**: Ensure all refactored code has proper test coverage.
5. **Documentation**: Update and enhance documentation during refactoring.

## Detailed Refactoring Plans

### 1. scirs2-neural/src/linalg/batch_operations.rs

This file contains various batch matrix operations optimized for neural networks.

**Proposed structure**:
- `batch_operations/mod.rs` - Main module and common utilities
- `batch_operations/matmul.rs` - Batch matrix multiplication operations
- `batch_operations/normalization.rs` - Batch normalization operations
- `batch_operations/attention.rs` - Attention mechanism operations
- `batch_operations/convolution.rs` - Convolution operations
- `batch_operations/rnn.rs` - RNN/LSTM specific operations

### 2. scirs2-series/src/trends.rs

This file contains trend estimation and filtering methods for time series.

**Proposed structure**:
- `trends/mod.rs` - Main module and common types
- `trends/spline.rs` - Spline-based trend estimation
- `trends/robust.rs` - Robust trend filtering methods
- `trends/piecewise.rs` - Piecewise trend estimation
- `trends/confidence.rs` - Trend confidence intervals
- `trends/seasonal.rs` - Seasonal trend decomposition

### 3. scirs2-autograd/src/tensor_ops/mod.rs

This file contains a collection of functions for manipulating tensor objects.

**Proposed structure**:
- Keep the modular organization but refactor the implementation in the main mod.rs
- Move shared code from mod.rs to a separate `tensor_ops/common.rs` file
- Create higher-level abstractions where possible

### 4. scirs2-metrics/src/regression.rs

This file contains regression metrics implementations.

**Proposed structure**:
- `regression/mod.rs` - Main module and common types
- `regression/error.rs` - Error metrics (MSE, MAE, etc.)
- `regression/correlation.rs` - Correlation metrics (R², etc.)
- `regression/residual.rs` - Residual analysis
- `regression/robust.rs` - Robust regression metrics

### 5. scirs2-metrics/src/clustering.rs

This file contains clustering metrics implementations.

**Proposed structure**:
- `clustering/mod.rs` - Main module and common types
- `clustering/internal.rs` - Internal validation metrics
- `clustering/external.rs` - External validation metrics
- `clustering/distance.rs` - Distance-based metrics
- `clustering/density.rs` - Density-based metrics

### 6. scirs2-sparse/src/linalg.rs

This file contains linear algebra operations for sparse matrices.

**Proposed structure**:
- `linalg/mod.rs` - Main module and common types
- `linalg/factorization.rs` - Matrix factorization operations
- `linalg/solvers.rs` - Linear system solvers
- `linalg/eigenvalue.rs` - Eigenvalue/eigenvector operations
- `linalg/operations.rs` - Basic matrix operations

### 7. scirs2-core/src/ndarray_ext/stats.rs

This file contains statistical functions for ndarray.

**Proposed structure**:
- `ndarray_ext/stats/mod.rs` - Main module and common utilities
- `ndarray_ext/stats/descriptive.rs` - Descriptive statistics
- `ndarray_ext/stats/hypothesis.rs` - Hypothesis testing
- `ndarray_ext/stats/distribution.rs` - Distribution-related functions
- `ndarray_ext/stats/correlation.rs` - Correlation functions

### 8. scirs2-fft/src/hfft.rs

This file contains Hermitian FFT implementations.

**Proposed structure**:
- `hfft/mod.rs` - Main module and common types
- `hfft/real_to_complex.rs` - Real-to-complex transforms
- `hfft/complex_to_real.rs` - Complex-to-real transforms
- `hfft/symmetric.rs` - Symmetric FFT operations
- `hfft/utility.rs` - Utility functions

### 9. scirs2-signal/src/dwt.rs

This file contains discrete wavelet transform implementations.

**Proposed structure**:
- `dwt/mod.rs` - Main module and common types
- `dwt/transform.rs` - Core transform algorithms
- `dwt/filters.rs` - Wavelet filters
- `dwt/boundary.rs` - Boundary handling methods
- `dwt/multiscale.rs` - Multiscale analysis functions

### 10. scirs2-fft/src/fft.rs

This file contains FFT implementations.

**Proposed structure**:
- `fft/mod.rs` - Main module and common types
- `fft/algorithms.rs` - Core FFT algorithms
- `fft/planning.rs` - FFT planning and optimization
- `fft/utility.rs` - Utility functions
- `fft/windowing.rs` - Window functions

## Implementation Plan

The refactoring should proceed in the following phases:

1. **Analysis Phase** (2 weeks):
   - Detailed code analysis of each file
   - Create more specific refactoring plans
   - Set up metrics to measure improvement

2. **Implementation Phase** (8-10 weeks):
   - Refactor files in order of priority
   - Maintain complete test coverage throughout
   - Regular code reviews for each refactored module

3. **Verification Phase** (2 weeks):
   - Performance testing before and after
   - API compatibility verification
   - Documentation updates

4. **Integration Phase** (2 weeks):
   - Integrate all refactored modules
   - Final testing and benchmarking
   - Release notes and documentation updates

## Success Metrics

The success of this refactoring effort will be measured by:

1. Reduction in file sizes (no file should exceed 1000 lines)
2. Maintained or improved test coverage
3. No regressions in functionality or performance
4. Improved code organization and maintainability
5. Easier navigation and understanding for new contributors

## Priority Order

Refactoring should be tackled in the following order:

1. scirs2-neural/src/linalg/batch_operations.rs (highest priority)
2. scirs2-series/src/trends.rs
3. scirs2-autograd/src/tensor_ops/mod.rs
4. scirs2-metrics/src/regression.rs
5. scirs2-metrics/src/clustering.rs

These files should be addressed first as they are the largest and most complex, with the remaining files following afterward.