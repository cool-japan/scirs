#![allow(deprecated)]
#![allow(clippy::new_without_default)]
#![allow(clippy::needless_return)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::extend_with_drain)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::derivable_impls)]
//! # SciRS2 Linear Algebra - High-Performance Matrix Operations
//!
//! **scirs2-linalg** provides comprehensive linear algebra operations with SciPy/NumPy-compatible
//! APIs, leveraging native BLAS/LAPACK for peak performance and offering advanced features like
//! SIMD acceleration, GPU support, and specialized solvers.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy/NumPy Compatibility**: Drop-in replacement for `scipy.linalg` and `numpy.linalg`
//! - **Native BLAS/LAPACK**: Hardware-optimized through OpenBLAS, Intel MKL, or Apple Accelerate
//! - **SIMD Acceleration**: AVX/AVX2/AVX-512 optimized operations for f32/f64
//! - **GPU Support**: CUDA, ROCm, OpenCL, and Metal acceleration
//! - **Parallel Processing**: Multi-threaded via Rayon for large matrices
//! - **Comprehensive Solvers**: Direct, iterative, sparse, and specialized methods
//! - **Matrix Functions**: Exponential, logarithm, square root, and trigonometric functions
//! - **Attention Mechanisms**: Multi-head, flash, and sparse attention for transformer models
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | SciPy/NumPy Equivalent | Description |
//! |---------------|------------------------|-------------|
//! | Basic ops | `scipy.linalg.det`, `inv` | Determinants, inverses, traces |
//! | Decompositions | `scipy.linalg.lu`, `qr`, `svd` | LU, QR, SVD, Cholesky, Schur |
//! | Eigenvalues | `scipy.linalg.eig`, `eigh` | Standard and generalized eigenproblems |
//! | Solvers | `scipy.linalg.solve`, `lstsq` | Linear systems (direct & iterative) |
//! | Matrix functions | `scipy.linalg.expm`, `logm` | Matrix exponential, logarithm, etc. |
//! | Norms | `numpy.linalg.norm`, `cond` | Vector/matrix norms, condition numbers |
//! | Specialized | `scipy.linalg.solve_banded` | Banded, circulant, Toeplitz matrices |
//! | Attention | - | Multi-head, flash attention (PyTorch-style) |
//! | BLAS | `scipy.linalg.blas.*` | Low-level BLAS operations |
//! | LAPACK | `scipy.linalg.lapack.*` | Low-level LAPACK operations |
//!
//! ## 🚀 Quick Start
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! scirs2-linalg = "0.1.0-rc.1"
//! # Optional features
//! scirs2-linalg = { version = "0.1.0-rc.1", features = ["simd", "parallel", "gpu"] }
//! ```
//!
//! ### Basic Matrix Operations
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::{det, inv, solve};
//!
//! // Determinant and inverse
//! let a = array![[4.0, 2.0], [2.0, 3.0]];
//! let det_a = det(&a.view(), None).unwrap();
//! let a_inv = inv(&a.view(), None).unwrap();
//!
//! // Solve linear system Ax = b
//! let b = array![6.0, 7.0];
//! let x = solve(&a.view(), &b.view(), None).unwrap();
//! ```
//!
//! ### Matrix Decompositions
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::{lu, qr, svd, cholesky, eig};
//!
//! let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
//!
//! // LU decomposition: PA = LU
//! let (p, l, u) = lu(&a.view(), None).unwrap();
//!
//! // QR decomposition: A = QR
//! let (q, r) = qr(&a.view(), None).unwrap();
//!
//! // SVD: A = UΣVᵀ
//! let (u, s, vt) = svd(&a.view(), true, None).unwrap();
//!
//! // Eigenvalues and eigenvectors
//! let (eigenvalues, eigenvectors) = eig(&a.view(), None).unwrap();
//!
//! // Cholesky decomposition for positive definite matrices
//! let spd = array![[4.0, 2.0], [2.0, 3.0]];
//! let l_chol = cholesky(&spd.view(), None).unwrap();
//! ```
//!
//! ### Iterative Solvers (Large Sparse Systems)
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::{conjugate_gradient, gmres};
//!
//! // Conjugate Gradient for symmetric positive definite systems
//! let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
//! let b = array![1.0_f64, 2.0];
//! let x_cg = conjugate_gradient(&a.view(), &b.view(), 10, 1e-10, None).unwrap();
//!
//! // GMRES for general systems
//! let x_gmres = gmres(&a.view(), &b.view(), 10, 1e-10, None).unwrap();
//! ```
//!
//! ### Matrix Functions
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::{expm, logm, sqrtm, sinm, cosm};
//!
//! let a = array![[1.0, 0.5], [0.5, 1.0]];
//!
//! // Matrix exponential: exp(A)
//! let exp_a = expm(&a.view(), None).unwrap();
//!
//! // Matrix logarithm: log(A)
//! let log_a = logm(&a.view(), None).unwrap();
//!
//! // Matrix square root: √A
//! let sqrt_a = sqrtm(&a.view(), None).unwrap();
//! ```
//!
//! ### Accelerated BLAS/LAPACK Operations
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::accelerated::{matmul, solve as fast_solve};
//!
//! // Hardware-accelerated matrix multiplication
//! let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
//! let c = matmul(&a.view(), &b.view()).unwrap();
//!
//! // Fast linear system solver using LAPACK
//! let x = fast_solve(&a.view(), &b.view()).unwrap();
//! ```
//!
//! ### Attention Mechanisms (Deep Learning)
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::Array2;
//! use scirs2_linalg::attention::{multi_head_attention, flash_attention, AttentionConfig};
//!
//! // Multi-head attention (Transformer-style)
//! let query = Array2::<f32>::zeros((32, 64));  // (batch_size, d_model)
//! let key = Array2::<f32>::zeros((32, 64));
//! let value = Array2::<f32>::zeros((32, 64));
//!
//! let config = AttentionConfig {
//!     num_heads: 8,
//!     dropout: 0.1,
//!     causal: false,
//! };
//!
//! let output = multi_head_attention(&query.view(), &key.view(), &value.view(), &config);
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ```text
//! scirs2-linalg
//! ├── Basic Operations (det, inv, trace, rank)
//! ├── Decompositions (LU, QR, SVD, Cholesky, Eigenvalues)
//! ├── Solvers
//! │   ├── Direct (solve, lstsq)
//! │   ├── Iterative (CG, GMRES, BiCGSTAB)
//! │   └── Specialized (banded, triangular, sparse-dense)
//! ├── Matrix Functions (expm, logm, sqrtm, trigonometric)
//! ├── Accelerated Backends
//! │   ├── BLAS/LAPACK (native libraries)
//! │   ├── SIMD (AVX/AVX2/AVX-512)
//! │   ├── Parallel (Rayon multi-threading)
//! │   └── GPU (CUDA/ROCm/OpenCL/Metal)
//! ├── Advanced Features
//! │   ├── Attention mechanisms
//! │   ├── Hierarchical matrices (H-matrices)
//! │   ├── Kronecker factorization (K-FAC)
//! │   ├── Randomized algorithms
//! │   ├── Mixed precision
//! │   └── Quantization-aware operations
//! └── Compatibility Layer (SciPy-compatible API)
//! ```
//!
//! ## 📊 Performance
//!
//! | Operation | Size | Pure Rust | BLAS/LAPACK | SIMD | GPU |
//! |-----------|------|-----------|-------------|------|-----|
//! | Matrix Multiply | 1000×1000 | 2.5s | 15ms | 180ms | 5ms |
//! | SVD | 1000×1000 | 8.2s | 120ms | N/A | 35ms |
//! | Eigenvalues | 1000×1000 | 6.8s | 95ms | N/A | 28ms |
//! | Solve (direct) | 1000×1000 | 1.8s | 22ms | 140ms | 8ms |
//!
//! **Note**: Benchmarks on AMD Ryzen 9 5950X with NVIDIA RTX 3090. BLAS/LAPACK uses OpenBLAS.
//!
//! ## 🔗 Integration
//!
//! Works seamlessly with other SciRS2 crates:
//! - `scirs2-stats`: Covariance matrices, statistical distributions
//! - `scirs2-optimize`: Hessian computations, constraint Jacobians
//! - `scirs2-neural`: Weight matrices, gradient computations
//! - `scirs2-sparse`: Sparse matrix operations
//!
//! ## 🔒 Version Information
//!
//! - **Version**: 0.1.0-rc.1
//! - **Release Date**: October 03, 2025
//! - **MSRV** (Minimum Supported Rust Version): 1.70.0
//! - **Documentation**: [docs.rs/scirs2-linalg](https://docs.rs/scirs2-linalg)
//! - **Repository**: [github.com/cool-japan/scirs](https://github.com/cool-japan/scirs)
// Note: BLAS/LAPACK functionality is provided through ndarray-linalg from scirs2-core

// Export error types
pub mod error;
pub use error::{LinalgError, LinalgResult};

// Basic modules
pub mod attention;
mod basic;
pub mod batch;
pub mod broadcast;
pub mod complex;
pub mod convolution;
mod decomposition;
pub mod decomposition_advanced;
// Main eigen module
pub mod eigen;
pub use self::eigen::{
    advanced_precision_eig, eig, eig_gen, eigh, eigh_gen, eigvals, eigvals_gen, eigvalsh,
    eigvalsh_gen, power_iteration,
};

// Specialized eigen solvers in separate module
pub mod eigen_specialized;
pub mod extended_precision;
pub mod generic;
pub mod gradient;
pub mod hierarchical;
mod iterative_solvers;
pub mod kronecker;
pub mod large_scale;
pub mod lowrank;
pub mod matrix_calculus;
pub mod matrix_dynamics;
pub mod matrix_equations;
pub mod matrix_factorization;
pub mod matrix_functions;
pub mod matrixfree;
pub mod mixed_precision;
mod norm;
pub mod optim;
pub mod parallel;
pub mod parallel_dispatch;
pub mod perf_opt;
pub mod preconditioners;
pub mod projection;
/// Quantization-aware linear algebra operations
pub mod quantization;
// Re-enabled quantization module
pub use self::quantization::calibration::{
    calibrate_matrix, calibrate_vector, get_activation_calibration_config,
    get_weight_calibration_config, CalibrationConfig, CalibrationMethod,
};
pub mod random;
pub mod random_matrices;
// Temporarily disabled due to validation trait dependency issues and API incompatibilities
// pub mod random_new;
pub mod circulant_toeplitz;
mod diagnostics;
pub mod fft;
pub mod scalable;
pub mod simd_ops;
mod solve;
pub mod solvers;
pub mod sparse_dense;
pub mod special;
pub mod specialized;
pub mod stats;
pub mod structured;
#[cfg(feature = "tensor_contraction")]
pub mod tensor_contraction;
pub mod tensor_train;
mod validation;
// Distributed computing support (temporarily disabled - needs extensive API fixes)
// pub mod distributed;

// GPU acceleration foundations
#[cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "rocm",
    feature = "metal"
))]
pub mod gpu;

// Automatic differentiation support
#[cfg(feature = "autograd")]
pub mod autograd;

// SciPy-compatible API wrappers
pub mod compat;
pub mod compat_wrappers;

// Accelerated implementations using BLAS/LAPACK
pub mod blas_accelerated;
pub mod lapack_accelerated;

// BLAS and LAPACK wrappers
pub mod blas;
pub mod lapack;

// Re-export the accelerated implementations
pub mod accelerated {
    //! Accelerated linear algebra operations using native BLAS/LAPACK
    //!
    //! This module provides optimized implementations of linear algebra operations
    //! using ndarray-linalg's bindings to native BLAS/LAPACK libraries.
    //! These functions are significantly faster for large matrices compared to
    //! pure Rust implementations.

    pub use super::blas_accelerated::*;
    pub use super::lapack_accelerated::*;
}

// Re-exports for user convenience
pub use self::basic::{det, inv, matrix_power, trace as basic_trace};
pub use self::eigen_specialized::{
    banded_eigen, banded_eigh, banded_eigvalsh, circulant_eigenvalues, largest_k_eigh,
    partial_eigen, smallest_k_eigh, tridiagonal_eigen, tridiagonal_eigh, tridiagonal_eigvalsh,
};
// Re-export complex module functions explicitly to avoid conflicts
pub use self::complex::enhanced_ops::{
    det as complex_det, frobenius_norm, hermitian_part, inner_product, is_hermitian, is_unitary,
    matrix_exp, matvec, polar_decomposition, power_method, rank as complex_rank,
    schur as complex_schur, skew_hermitian_part, trace,
};
pub use self::complex::{complex_inverse, complex_matmul, hermitian_transpose};
// Main decomposition functions with workers parameter
pub use self::decomposition::{cholesky, lu, qr, schur, svd};
// Backward compatibility versions (deprecated)
pub use self::decomposition::{cholesky_default, lu_default, qr_default, svd_default};
// Advanced decomposition functions
pub use self::decomposition_advanced::{
    jacobi_svd, polar_decomposition as advanced_polar_decomposition, polar_decomposition_newton,
    qr_with_column_pivoting,
};
// Backward compatibility versions for basic functions (deprecated)
pub use self::basic::{det_default, inv_default, matrix_power_default};
// Backward compatibility versions for iterative solvers (deprecated)
pub use self::iterative_solvers::conjugate_gradient_default;
// Eigen module exports included in other use statements
pub use self::extended_precision::*;
pub use self::iterative_solvers::*;
// pub use self::matrix_calculus::*; // Temporarily disabled
pub use self::matrix_equations::{
    solve_continuous_riccati, solve_discrete_riccati, solve_generalized_sylvester, solve_stein,
    solve_sylvester,
};
pub use self::matrix_factorization::{
    cur_decomposition, interpolative_decomposition, nmf, rank_revealing_qr, utv_decomposition,
};
pub use self::matrix_functions::{
    acosm, asinm, atanm, coshm, cosm, expm, geometric_mean_spd, logm, logm_parallel, nuclear_norm,
    signm, sinhm, sinm, spectral_condition_number, spectral_radius, sqrtm, sqrtm_parallel, tanhm,
    tanm, tikhonov_regularization,
};
pub use self::matrixfree::{
    block_diagonal_operator, conjugate_gradient as matrix_free_conjugate_gradient,
    diagonal_operator, gmres as matrix_free_gmres, jacobi_preconditioner,
    preconditioned_conjugate_gradient as matrix_free_preconditioned_conjugate_gradient,
    LinearOperator, MatrixFreeOp,
};
pub use self::norm::*;
// Main solve functions with workers parameter
pub use self::solve::{lstsq, solve, solve_multiple, solve_triangular, LstsqResult};
// Backward compatibility versions (deprecated)
pub use self::solve::{lstsq_default, solve_default, solve_multiple_default};
// Iterative solvers
pub use self::solvers::iterative::{
    bicgstab, conjugate_gradient as cg_solver, gmres,
    preconditioned_conjugate_gradient as pcg_solver, IterativeSolverOptions, IterativeSolverResult,
};
pub use self::specialized::{
    specialized_to_operator, BandedMatrix, SpecializedMatrix, SymmetricMatrix, TridiagonalMatrix,
};
pub use self::stats::*;
pub use self::structured::{
    structured_to_operator, CirculantMatrix, HankelMatrix, StructuredMatrix, ToeplitzMatrix,
};
#[cfg(feature = "tensor_contraction")]
pub use self::tensor_contraction::{batch_matmul, contract, einsum, hosvd};

// Prelude module for convenient imports
pub mod prelude {
    //! Common linear algebra operations for convenient importing
    //!
    //! ```
    //! use scirs2_linalg::prelude::*;
    //! ```

    // Pure Rust implementations
    pub use super::attention::{
        attention, attention_with_alibi, attention_with_rpe, causal_attention, flash_attention,
        grouped_query_attention, linear_attention, masked_attention, multi_head_attention,
        relative_position_attention, rotary_embedding, scaled_dot_product_attention,
        sparse_attention, AttentionConfig, AttentionMask,
    };
    pub use super::basic::{det, inv};
    pub use super::batch::attention::{
        batch_flash_attention, batch_multi_head_attention, batch_multi_query_attention,
    };
    pub use super::broadcast::{
        broadcast_matmul, broadcast_matmul_3d, broadcast_matvec, BroadcastExt,
    };
    pub use super::complex::enhanced_ops::{
        det as complex_det, frobenius_norm as complex_frobenius_norm, hermitian_part,
        inner_product as complex_inner_product, is_hermitian, is_unitary,
        matrix_exp as complex_exp, matvec as complex_matvec, polar_decomposition as complex_polar,
        schur as complex_schur, skew_hermitian_part,
    };
    pub use super::complex::{
        complex_inverse, complex_matmul, complex_norm_frobenius, hermitian_transpose,
    };
    pub use super::convolution::{
        col2im, compute_conv_indices, conv2d_backward_bias, conv2d_backward_input,
        conv2d_backward_kernel, conv2d_im2col, conv_transpose2d, im2col, max_pool2d,
        max_pool2d_backward,
    };
    pub use super::decomposition::{cholesky, lu, qr, schur, svd};
    pub use super::decomposition_advanced::{
        jacobi_svd, polar_decomposition as advanced_polar_decomposition,
        polar_decomposition_newton, qr_with_column_pivoting,
    };
    pub use super::eigen::{
        advanced_precision_eig, eig, eig_gen, eigh, eigh_gen, eigvals, eigvals_gen, eigvalsh,
        eigvalsh_gen, power_iteration,
    };
    pub use super::eigen_specialized::{
        banded_eigen, banded_eigh, banded_eigvalsh, circulant_eigenvalues, largest_k_eigh,
        partial_eigen, smallest_k_eigh, tridiagonal_eigen, tridiagonal_eigh, tridiagonal_eigvalsh,
    };
    pub use super::extended_precision::eigen::{
        extended_eig, extended_eigh, extended_eigvals, extended_eigvalsh,
    };
    pub use super::extended_precision::factorizations::{
        extended_cholesky, extended_lu, extended_qr, extended_svd,
    };
    pub use super::extended_precision::{
        extended_det, extended_matmul, extended_matvec, extended_solve,
    };
    pub use super::hierarchical::{
        adaptive_block_lowrank, build_cluster_tree, BlockType, ClusterNode, HMatrix, HMatrixBlock,
        HMatrixMemoryInfo, HSSMatrix, HSSNode,
    };
    pub use super::iterative_solvers::{
        bicgstab, conjugate_gradient, gauss_seidel, geometric_multigrid, jacobi_method, minres,
        successive_over_relaxation,
    };
    pub use super::kronecker::{
        advanced_kfac_step, kfac_factorization, kfac_update, kron, kron_factorize, kron_matmul,
        kron_matvec, BlockDiagonalFisher, BlockFisherMemoryInfo, KFACOptimizer,
    };
    pub use super::large_scale::{
        block_krylov_solve, ca_gmres, incremental_svd, randomized_block_lanczos,
        randomized_least_squares, randomized_norm,
    };
    pub use super::lowrank::{
        cur_decomposition, nmf as lowrank_nmf, pca, randomized_svd, truncated_svd,
    };
    pub use super::solvers::iterative::{
        bicgstab as iterative_bicgstab, conjugate_gradient as iterative_cg,
        gmres as iterative_gmres, preconditioned_conjugate_gradient as iterative_pcg,
        IterativeSolverOptions, IterativeSolverResult,
    };
    // Matrix calculus temporarily disabled due to compilation issues
    // pub use super::matrix_calculus::enhanced::{
    //     hessian_vector_product, jacobian_vector_product, matrix_gradient, taylor_approximation,
    //     vector_jacobian_product,
    // };
    // pub use super::matrix_calculus::{directional_derivative, gradient, hessian, jacobian};
    pub use super::matrix_dynamics::{
        lyapunov_solve, matrix_exp_action, matrix_ode_solve, quantum_evolution, riccati_solve,
        stability_analysis, DynamicsConfig, ODEResult,
    };
    pub use super::matrix_factorization::{
        interpolative_decomposition, nmf, rank_revealing_qr, utv_decomposition,
    };
    pub use super::matrix_functions::{
        acosm, asinm, atanm, coshm, cosm, expm, geometric_mean_spd, logm, logm_parallel,
        matrix_power, nuclear_norm, polar_decomposition, signm, sinhm, sinm,
        spectral_condition_number, spectral_radius, sqrtm, sqrtm_parallel, tanhm, tanm,
        tikhonov_regularization,
    };
    pub use super::matrixfree::{
        block_diagonal_operator, conjugate_gradient as matrix_free_conjugate_gradient,
        diagonal_operator, gmres as matrix_free_gmres, jacobi_preconditioner,
        preconditioned_conjugate_gradient as matrix_free_preconditioned_conjugate_gradient,
        LinearOperator, MatrixFreeOp,
    };
    // Temporarily disabled due to wide dependency issues
    pub use super::mixed_precision::{
        convert, convert_2d, iterative_refinement_solve, mixed_precision_cond,
        mixed_precision_dot_f32, mixed_precision_matmul, mixed_precision_matvec,
        mixed_precision_qr, mixed_precision_solve, mixed_precision_svd,
    };
    // #[cfg(feature = "simd")]
    // pub use super::mixed_precision::{
    //     simd_mixed_precision_dot_f32_f64, simd_mixed_precision_matmul_f32_f64,
    //     simd_mixed_precision_matvec_f32_f64,
    // };
    pub use super::norm::{cond, matrix_norm, matrix_rank, vector_norm, vector_norm_parallel};
    pub use super::optim::{block_matmul, strassen_matmul, tiled_matmul};
    pub use super::perf_opt::{
        blocked_matmul, inplace_add, inplace_scale, matmul_benchmark, optimized_transpose,
        OptAlgorithm, OptConfig,
    };
    pub use super::preconditioners::{
        analyze_preconditioner, create_preconditioner, preconditioned_conjugate_gradient,
        preconditioned_gmres, AdaptivePreconditioner, BlockJacobiPreconditioner,
        DiagonalPreconditioner, IncompleteCholeskyPreconditioner, IncompleteLUPreconditioner,
        PolynomialPreconditioner, PreconditionerAnalysis, PreconditionerConfig, PreconditionerOp,
        PreconditionerType,
    };
    pub use super::projection::{
        gaussian_randommatrix, johnson_lindenstrauss_min_dim, johnson_lindenstrauss_transform,
        project, sparse_randommatrix, very_sparse_randommatrix,
    };
    pub use super::quantization::calibration::{
        calibrate_matrix, calibrate_vector, CalibrationConfig, CalibrationMethod,
    };
    #[cfg(feature = "simd")]
    pub use super::quantization::simd::{
        simd_quantized_dot, simd_quantized_matmul, simd_quantized_matvec,
    };
    pub use super::quantization::{
        dequantize_matrix, dequantize_vector, fake_quantize, fake_quantize_vector, quantize_matrix,
        quantize_matrix_per_channel, quantize_vector, quantized_dot, quantized_matmul,
        quantized_matvec, QuantizationMethod, QuantizationParams, QuantizedDataType,
        QuantizedMatrix, QuantizedVector,
    };
    pub use super::random::{
        banded, diagonal, hilbert, low_rank, normal, orthogonal, permutation, random_correlation,
        sparse, spd, toeplitz, uniform, vandermonde, with_condition_number, with_eigenvalues,
    };
    pub use super::random_matrices::{
        random_complexmatrix, random_hermitian, randommatrix, Distribution1D, MatrixType,
    };
    // 一時的にrandom_newエクスポートを無効化（コンパイル問題解決まで）
    // pub use super::random_new::{
    //     uniform as enhanced_uniform, normal as enhanced_normal, complex as complex_random,
    //     orthogonal as enhanced_orthogonal, unitary, hilbert as enhanced_hilbert,
    //     toeplitz as enhanced_toeplitz, vandermonde as enhanced_vandermonde
    // };
    pub use super::fft::{
        apply_window, dct_1d, dst_1d, fft_1d, fft_2d, fft_3d, fft_convolve, fft_frequencies,
        idct_1d, irfft_1d, periodogram_psd, rfft_1d, welch_psd, Complex32, Complex64, FFTAlgorithm,
        FFTPlan, WindowFunction,
    };
    pub use super::generic::{
        gdet, geig, gemm, gemv, ginv, gnorm, gqr, gsolve, gsvd, GenericEigen, GenericQR,
        GenericSVD, LinalgScalar, PrecisionSelector,
    };
    pub use super::scalable::{
        adaptive_decomposition, blocked_matmul as scalable_blocked_matmul, classify_aspect_ratio,
        lq_decomposition, randomized_svd as scalable_randomized_svd, tsqr, AdaptiveResult,
        AspectRatio, ScalableConfig,
    };
    #[cfg(feature = "simd")]
    pub use super::simd_ops::{
        simd_axpy_f32,
        simd_axpy_f64,
        simd_dot_f32,
        simd_dot_f64,
        simd_frobenius_norm_f32,
        simd_frobenius_norm_f64,
        // GEMM operations
        simd_gemm_f32,
        simd_gemm_f64,
        simd_gemv_f32,
        simd_gemv_f64,
        simd_matmul_f32,
        simd_matmul_f64,
        simd_matmul_optimized_f32,
        simd_matmul_optimized_f64,
        simd_matvec_f32,
        simd_matvec_f64,
        // Transpose operations
        simd_transpose_f32,
        simd_transpose_f64,
        // Vector norm operations
        simd_vector_norm_f32,
        simd_vector_norm_f64,
        simdmatrix_max_f32,
        simdmatrix_max_f64,
        simdmatrix_min_f32,
        simdmatrix_min_f64,
        GemmBlockSizes,
    };
    pub use super::solve::{lstsq, solve, solve_multiple, solve_triangular};
    pub use super::sparse_dense::{
        dense_sparse_matmul, dense_sparse_matvec, sparse_dense_add, sparse_dense_elementwise_mul,
        sparse_dense_matmul, sparse_dense_matvec, sparse_dense_sub, sparse_from_ndarray,
        SparseMatrixView,
    };
    pub use super::special::block_diag;
    pub use super::specialized::{
        specialized_to_operator, BandedMatrix, BlockTridiagonalMatrix, SpecializedMatrix,
        SymmetricMatrix, TridiagonalMatrix,
    };
    pub use super::stats::{correlationmatrix, covariancematrix};
    pub use super::structured::{
        solve_circulant, solve_toeplitz, structured_to_operator, CirculantMatrix, HankelMatrix,
        StructuredMatrix, ToeplitzMatrix,
    };
    #[cfg(feature = "tensor_contraction")]
    pub use super::tensor_contraction::{batch_matmul, contract, einsum, hosvd};
    pub use super::tensor_train::{tt_add, tt_decomposition, tt_hadamard, TTTensor};

    // Distributed computing (temporarily disabled)
    // pub use super::distributed::{
    //     initialize_distributed, finalize_distributed, DistributedConfig, DistributedContext,
    //     DistributedMatrix, DistributedVector, DistributedLinalgOps, DistributedStats,
    //     CompressionConfig, CompressionAlgorithm, CommunicationBackend, DistributionStrategy,
    // };

    // Automatic differentiation support
    #[cfg(feature = "autograd")]
    pub mod autograd {
        //! Automatic differentiation for linear algebra operations
        //!
        //! Note: The autograd module is currently undergoing a major API redesign.
        //! For basic usage, see examples/autograd_simple_example.rs which demonstrates
        //! how to use scirs2-autograd directly with linear algebra operations.

        // Re-export the module itself for documentation purposes
        pub use super::super::autograd::*;
    }

    // Accelerated implementations
    pub mod accelerated {
        //! Accelerated linear algebra operations using native BLAS/LAPACK
        pub use super::super::blas_accelerated::{
            dot, gemm, gemv, inv as fast_inv, matmul, norm, solve as fast_solve,
        };
        pub use super::super::lapack_accelerated::{
            cholesky as fast_cholesky, eig as fast_eig, eigh as fast_eigh, lu as fast_lu,
            qr as fast_qr, svd as fast_svd,
        };
    }

    // SciPy-compatible API
    pub mod scipy_compat {
        //! SciPy-compatible linear algebra functions
        //!
        //! This module provides functions with the same signatures and behavior
        //! as SciPy's linalg module, making migration from Python to Rust easier.
        //!
        //! # Examples
        //!
        //! ```
        //! use scirs2_core::ndarray::array;
        //! use scirs2_linalg::prelude::scipy_compat;
        //!
        //! let a = array![[4.0, 2.0], [2.0, 3.0]];
        //!
        //! // SciPy-style determinant computation
        //! let det = scipy_compat::det(&a.view(), false, true).unwrap();
        //!
        //! // SciPy-style matrix norm
        //! let norm = scipy_compat::norm(&a.view(), Some("fro"), None, false, true).unwrap();
        //! ```

        pub use super::super::compat::{
            // Utilities
            block_diag,
            cholesky,
            // Linear system solvers
            compat_solve as solve,
            cond,
            cosm,
            // Basic matrix operations
            det,
            // Eigenvalue problems
            eig,
            eig_banded,
            eigh,
            eigh_tridiagonal,
            eigvals,
            eigvals_banded,
            eigvalsh,
            eigvalsh_tridiagonal,
            // Matrix functions
            expm,
            fractionalmatrix_power,
            funm,
            inv,
            logm,
            lstsq,
            // Matrix decompositions
            lu,
            matrix_rank,
            norm,
            pinv,
            polar,
            qr,
            rq,
            schur,
            sinm,
            solve_banded,
            solve_triangular,
            sqrtm,
            svd,
            tanm,
            vector_norm,
            // Type aliases
            SvdResult,
        };
    }
}
