#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_return)]
#![allow(clippy::manual_ok_err)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_slice_fill)]
#![allow(dead_code)]
//! # SciRS2 Sparse - Sparse Matrix Operations
//!
//! **scirs2-sparse** provides comprehensive sparse matrix formats and operations modeled after SciPy's
//! `sparse` module, offering CSR, CSC, COO, DOK, LIL, DIA, BSR formats with efficient algorithms
//! for large-scale sparse linear algebra, eigenvalue problems, and graph operations.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy Compatibility**: Drop-in replacement for `scipy.sparse` classes
//! - **Multiple Formats**: CSR, CSC, COO, DOK, LIL, DIA, BSR with easy conversion
//! - **Efficient Operations**: Sparse matrix-vector/matrix multiplication
//! - **Linear Solvers**: Direct (LU, Cholesky) and iterative (CG, GMRES) solvers
//! - **Eigenvalue Solvers**: ARPACK-based sparse eigenvalue computation
//! - **Array API**: Modern NumPy-compatible array interface (recommended)
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Format | SciPy Equivalent | Description |
//! |---------------|------------------|-------------|
//! | `CsrArray` | `scipy.sparse.csr_array` | Compressed Sparse Row (efficient row slicing) |
//! | `CscArray` | `scipy.sparse.csc_array` | Compressed Sparse Column (efficient column slicing) |
//! | `CooArray` | `scipy.sparse.coo_array` | Coordinate format (efficient construction) |
//! | `DokArray` | `scipy.sparse.dok_array` | Dictionary of Keys (efficient element access) |
//! | `LilArray` | `scipy.sparse.lil_array` | List of Lists (efficient incremental construction) |
//! | `DiaArray` | `scipy.sparse.dia_array` | Diagonal format (efficient banded matrices) |
//! | `BsrArray` | `scipy.sparse.bsr_array` | Block Sparse Row (efficient block operations) |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-sparse = "0.6.3"
//! ```
//!
//! ```rust
//! use scirs2_sparse::csr_array::CsrArray;
//!
//! // Create sparse matrix from triplets (row, col, value)
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let sparse = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");
//! ```
//!
//! ## 🔒 Version: 0.6.3 (July 22, 2026)
//!
//! ## Matrix vs. Array API
//!
//! This module provides both a matrix-based API and an array-based API,
//! following SciPy's transition to a more NumPy-compatible array interface.
//!
//! When using the array interface (e.g., `CsrArray`), please note that:
//!
//! - `*` performs element-wise multiplication, not matrix multiplication
//! - Use `dot()` method for matrix multiplication
//! - Operations like `sum` produce arrays, not matrices
//! - Array-style slicing operations return scalars, 1D, or 2D arrays
//!
//! For new code, we recommend using the array interface, which is more consistent
//! with the rest of the numerical ecosystem.
//!
//! ## Examples
//!
//! ### Matrix API (Legacy)
//!
//! ```
//! use scirs2_sparse::csr::CsrMatrix;
//!
//! // Create a sparse matrix in CSR format
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
//! ```
//!
//! ### Array API (Recommended)
//!
//! ```
//! use scirs2_sparse::csr_array::CsrArray;
//!
//! // Create a sparse array in CSR format
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! // From triplets (COO-like construction)
//! let array = CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");
//!
//! // Or directly from CSR components
//! // let array = CsrArray::new(...);
//! ```

// Export error types
pub mod error;
pub use error::{SparseError, SparseResult};

// Base trait for sparse arrays
pub mod sparray;
pub use sparray::{is_sparse, SparseArray, SparseSum};

// Trait for symmetric sparse arrays
pub mod sym_sparray;
pub use sym_sparray::SymSparseArray;

// No spatial module in sparse

// Array API (recommended)
pub mod csr_array;
pub use csr_array::CsrArray;

pub mod csc_array;
pub use csc_array::CscArray;

pub mod coo_array;
pub use coo_array::CooArray;

pub mod dok_array;
pub use dok_array::DokArray;

pub mod lil_array;
pub use lil_array::LilArray;

pub mod dia_array;
pub use dia_array::DiaArray;

pub mod bsr_array;
pub use bsr_array::BsrArray;

pub mod banded_array;
pub use banded_array::BandedArray;

// Symmetric array formats
pub mod sym_csr;
pub use sym_csr::{SymCsrArray, SymCsrMatrix};

pub mod sym_coo;
pub use sym_coo::{SymCooArray, SymCooMatrix};

// Legacy matrix formats
pub mod csr;
pub use csr::CsrMatrix;

pub mod csc;
pub use csc::CscMatrix;

pub mod coo;
pub use coo::CooMatrix;

pub mod dok;
pub use dok::DokMatrix;

pub mod lil;
pub use lil::LilMatrix;

pub mod dia;
pub use dia::DiaMatrix;

pub mod bsr;
pub use bsr::BsrMatrix;

pub mod banded;
pub use banded::BandedMatrix;

// Utility functions
pub mod utils;

// Linear algebra with sparse matrices
pub mod linalg;
// Re-export the main functions from the reorganized linalg module
pub use linalg::{
    // Functions from solvers
    add,
    // Functions from iterative
    bicg,
    bicgstab,
    cg,
    cholesky_decomposition,
    // Enhanced operators
    convolution_operator,
    diag_matrix,
    eigs,
    eigsh,
    enhanced_add,
    enhanced_diagonal,
    enhanced_scale,
    enhanced_subtract,
    expm,
    // Functions from matfuncs
    expm_multiply,
    eye,
    finite_difference_operator,
    // GCROT solver
    gcrot,
    gmres,
    incomplete_cholesky,
    incomplete_lu,
    inv,
    // IRAM eigenvalue solver (v0.3.0)
    iram,
    iram_shift_invert,
    lanczos,
    // Decomposition functions
    lu_decomposition,
    matmul,
    matrix_power,
    multiply,
    norm,
    onenormest,
    // Eigenvalue functions
    power_iteration,
    qr_decomposition,
    // Specialized solvers (v0.2.0)
    solve_arrow_matrix,
    solve_banded_system,
    solve_block_2x2,
    solve_kronecker_system,
    solve_saddle_point,
    sparse_direct_solve,
    sparse_lstsq,
    spsolve,
    svd_truncated,
    // SVD functions
    svds,
    // TFQMR solver
    tfqmr,
    // IRAM (Arnoldi) eigenvalue solver (v0.3.0)
    ArnoldiConfig,
    ArpackOptions,
    // Interfaces
    AsLinearOperator,
    // Types from iterative
    BiCGOptions,
    BiCGSTABOptions,
    BiCGSTABResult,
    // Enhanced operator types
    BoundaryCondition,
    CGOptions,
    CGSOptions,
    CGSResult,
    CholeskyResult,
    ConvolutionMode,
    ConvolutionOperator,
    // Operator types
    DiagonalOperator,
    EigenResult,
    EigenvalueMethod,
    EnhancedDiagonalOperator,
    EnhancedDifferenceOperator,
    EnhancedOperatorOptions,
    EnhancedScaledOperator,
    EnhancedSumOperator,
    FiniteDifferenceOperator,
    GCROTOptions,
    GCROTResult,
    GMRESOptions,
    ICOptions,
    // Preconditioners
    ILU0Preconditioner,
    ILUOptions,
    IdentityOperator,
    IterationResult,
    JacobiPreconditioner,
    // Decomposition types
    LUResult,
    LanczosOptions,
    LinearOperator,
    // Eigenvalue types
    PowerIterationOptions,
    QRResult,
    SSORPreconditioner,
    // SVD types
    SVDOptions,
    SVDResult,
    ScaledIdentityOperator,
    TFQMROptions,
    TFQMRResult,
};

// Format conversions
pub mod convert;

// Construction utilities
pub mod construct;
pub mod construct_sym;

// Combining arrays
pub mod combine;
pub use combine::{block_diag, bmat, hstack, kron, kronsum, tril, triu, vstack};

// Index dtype handling utilities
pub mod index_dtype;
pub use index_dtype::{can_cast_safely, get_index_dtype, safely_cast_index_arrays};

// Optimized operations for symmetric sparse formats
pub mod sym_ops;

// Tensor-based sparse operations (v0.2.0)
pub mod tensor_sparse;

// Enhanced BSR format with flat storage and Block LU (v0.3.0)
pub mod bsr_enhanced;
pub use bsr_enhanced::{block_lu, block_lu_solve, BlockLUResult, EnhancedBsr};

// Enhanced DIA format with banded operations (v0.3.0)
pub mod dia_enhanced;
pub use dia_enhanced::{banded_solve, dia_tridiagonal_solve, tridiagonal_solve, EnhancedDia};

// Compressed Sparse Fiber (CSF) tensor format (v0.3.0)
pub mod csf_tensor;
pub use csf_tensor::CsfTensor;

// Advanced sparse formats (v0.4.0): SELL-C-sigma, CSR5, CSF (standalone), Polynomial Preconditioners
pub mod formats;
pub use formats::csf::CsfTensor as CsfTensorStandalone;
pub use formats::csr5::Csr5Matrix;
pub use formats::poly_precond::{
    ChebyshevPreconditioner, NeumannPreconditioner as NeumannPreconditionerPoly, PolyPrecondConfig,
};
pub use formats::sell::SellMatrix;

// Sparse matrix utility functions (v0.3.0)
pub mod sparse_functions;
pub use sparse_functions::{
    sparse_block_diag, sparse_diag_matrix, sparse_diags, sparse_eye, sparse_eye_rect,
    sparse_hstack, sparse_kron, sparse_kronsum, sparse_random, sparse_vstack,
};
pub use sym_ops::{
    sym_coo_matvec, sym_csr_matvec, sym_csr_quadratic_form, sym_csr_rank1_update, sym_csr_trace,
};

// Tensor operations (v0.2.0)
pub use tensor_sparse::{khatri_rao_product, CPDecomposition, SparseTensor, TuckerDecomposition};

// GPU-accelerated operations
pub mod gpu;
pub mod gpu_kernel_execution;
pub mod gpu_ops;
pub mod gpu_preconditioner;
pub mod gpu_spmv_implementation;
pub use gpu_kernel_execution::{
    calculate_adaptive_workgroup_size, execute_spmv_kernel, execute_symmetric_spmv_kernel,
    execute_triangular_solve_kernel, GpuKernelConfig, GpuMemoryManager as GpuKernelMemoryManager,
    GpuPerformanceProfiler, MemoryStrategy,
};
pub use gpu_ops::{
    gpu_sparse_matvec, gpu_sym_sparse_matvec, AdvancedGpuOps, GpuKernelScheduler, GpuMemoryManager,
    GpuOptions, GpuProfiler, OptimizedGpuOps,
};
pub use gpu_spmv_implementation::GpuSpMV;

// Memory-efficient algorithms and patterns
pub mod memory_efficient;
pub use memory_efficient::{
    streaming_sparse_matvec, CacheAwareOps, MemoryPool, MemoryTracker, OutOfCoreProcessor,
};

// SIMD-accelerated operations
pub mod simd_ops;
pub use simd_ops::{
    simd_csr_matvec, simd_sparse_elementwise, simd_sparse_linear_combination, simd_sparse_matmul,
    simd_sparse_norm, simd_sparse_scale, simd_sparse_transpose, ElementwiseOp, SimdOptions,
};

// Parallel vector operations for iterative solvers
pub mod parallel_vector_ops;
pub use parallel_vector_ops::{
    advanced_sparse_matvec_csr, parallel_axpy, parallel_dot, parallel_linear_combination,
    parallel_norm2, parallel_sparse_matvec_csr, parallel_vector_add, parallel_vector_copy,
    parallel_vector_scale, parallel_vector_sub, ParallelVectorOptions,
};

// Enhanced iterative solvers with preconditioners and sparse utilities (v0.3.0)
pub mod iterative_solvers;
pub use iterative_solvers::{
    // Solvers
    bicgstab as enhanced_bicgstab,
    cg as enhanced_cg,
    chebyshev,
    // Sparse utility functions
    estimate_spectral_radius,
    gmres as enhanced_gmres,
    sparse_diagonal,
    sparse_norm,
    sparse_trace,
    // Preconditioners (Array1-based interface)
    ILU0Preconditioner as EnhancedILU0Preconditioner,
    // Configuration and result types
    IterativeSolverConfig,
    JacobiPreconditioner as EnhancedJacobiPreconditioner,
    NormType,
    Preconditioner,
    SSORPreconditioner as EnhancedSSORPreconditioner,
    SolverResult,
};

// LOBPCG eigensolver (v0.3.0)
pub mod lobpcg;
pub use lobpcg::{
    lobpcg as lobpcg_eigensolver, lobpcg_generalised, EigenTarget, LobpcgConfig, LobpcgResult,
};

// Advanced Krylov subspace eigensolvers (v0.3.0)
pub mod krylov;
pub use krylov::{
    iram as krylov_iram, thick_restart_lanczos, IramConfig, KrylovEigenResult,
    ThickRestartLanczosConfig, WhichEigenvalues,
};

// Sparse matrix utilities (v0.3.0)
pub mod sparse_utils;
pub use sparse_utils::{
    condest_1norm, permute_matrix, reverse_cuthill_mckee, sparse_add, sparse_extract_diagonal,
    sparse_identity, sparse_kronecker, sparse_matrix_norm, sparse_matrix_trace, sparse_scale,
    sparse_sub, sparse_transpose, spgemm, RcmResult, SparseNorm,
};

// Incomplete factorization preconditioners (v0.3.0)
pub mod incomplete_factorizations;
pub use incomplete_factorizations::{Ic0, Ilu0 as Ilu0Enhanced, IluK, Ilut, IlutConfig, Milu};

// Sparse direct solvers (v0.3.0)
pub mod direct_solver;
pub use direct_solver::{
    amd_ordering, elimination_tree, inverse_perm, nested_dissection_ordering,
    sparse_cholesky_solve, sparse_lu_solve, symbolic_cholesky, SparseCholResult,
    SparseCholeskySolver, SparseLuResult, SparseLuSolver, SparseSolver, SymbolicAnalysis,
};

// Sparse QR factorization (v0.3.0)
pub mod sparse_qr;
pub use sparse_qr::{
    apply_q, apply_qt, extract_q_dense, numerical_rank, sparse_least_squares,
    sparse_qr as sparse_qr_factorize, SparseLeastSquaresResult, SparseQrConfig, SparseQrResult,
};

// Sparse matrix reordering algorithms (v0.4.0)
pub mod reorder;
pub use reorder::{
    amd, amd_simple, apply_permutation as reorder_apply_permutation, apply_permutation_csr_array,
    bandwidth as reorder_bandwidth, cuthill_mckee, cuthill_mckee_full, distance2_color,
    dsatur_color, greedy_color, nested_dissection, nested_dissection_full,
    nested_dissection_with_config, profile as reorder_profile,
    reverse_cuthill_mckee as reorder_rcm, reverse_cuthill_mckee_full, verify_coloring,
    verify_distance2_coloring, AdjacencyGraph, AmdResult, ColoringResult, CuthillMcKeeResult,
    GreedyOrdering, NestedDissectionConfig, NestedDissectionResult,
};

// Unified sparse eigenvalue interface (v0.3.0)
pub mod sparse_eigen;
pub use sparse_eigen::{
    cayley_transform_matvec, check_eigenpairs, compute_residuals, shift_invert_eig, sparse_eig,
    sparse_eigs, sparse_eigsh, EigenMethod, EigenvalueTarget, SparseEigenConfig, SparseEigenResult,
    SpectralTransform,
};

// Quantum-inspired sparse matrix operations (Advanced mode)
pub mod quantum_inspired_sparse;
pub use quantum_inspired_sparse::{
    QuantumProcessorStats, QuantumSparseConfig, QuantumSparseProcessor, QuantumStrategy,
};

// Neural-adaptive sparse matrix operations (Advanced mode)
pub mod neural_adaptive_sparse;
pub use neural_adaptive_sparse::{
    NeuralAdaptiveConfig, NeuralAdaptiveSparseProcessor, NeuralProcessorStats, OptimizationStrategy,
};

// Quantum-Neural hybrid optimization (Advanced mode)
pub mod quantum_neural_hybrid;
pub use quantum_neural_hybrid::{
    HybridStrategy, QuantumNeuralConfig, QuantumNeuralHybridProcessor, QuantumNeuralHybridStats,
};

// Adaptive memory compression for advanced-large sparse matrices (Advanced mode)
pub mod adaptive_memory_compression;
pub use adaptive_memory_compression::{
    AdaptiveCompressionConfig, AdaptiveMemoryCompressor, CompressedMatrix, CompressionAlgorithm,
    MemoryStats,
};

// Real-time performance monitoring and adaptation (Advanced mode)
pub mod realtime_performance_monitor;
pub use realtime_performance_monitor::{
    Alert, AlertSeverity, PerformanceMonitorConfig, PerformanceSample, ProcessorType,
    RealTimePerformanceMonitor,
};

// Cholesky update/downdate
pub mod cholesky_update;
// Distributed sparse operations (SpMV halo exchange, dist AMG)
pub mod distributed;
// Learned multigrid smoother
pub mod learned_smoother;
// Low-rank sparse update
pub mod low_rank_update;
// ML-based preconditioner
pub mod ml_preconditioner;
// Parallel AMG
pub mod parallel_amg;

// Compressed sparse graph algorithms
pub mod csgraph;
pub use csgraph::{
    all_pairs_shortest_path,
    bellman_ford_single_source,
    // Centrality measures (v0.2.0)
    betweenness_centrality,
    bfs_distances,
    // Traversal algorithms
    breadth_first_search,
    closeness_centrality,
    // Community detection (v0.2.0)
    community_detection,
    compute_laplacianmatrix,
    connected_components,
    degree_matrix,
    depth_first_search,
    dijkstra_single_source,
    // Max flow algorithms (v0.2.0)
    dinic,
    edmonds_karp,
    eigenvector_centrality,
    floyd_warshall,
    ford_fulkerson,
    has_path,
    is_connected,
    is_laplacian,
    is_spanning_tree,
    // Minimum spanning trees
    kruskal_mst,
    label_propagation,
    // Laplacian matrices
    laplacian,
    largest_component,
    louvain_communities,
    min_cut,
    minimum_spanning_tree,
    modularity,
    num_edges,
    num_vertices,
    pagerank,
    prim_mst,
    reachable_vertices,
    reconstruct_path,
    // Graph algorithms
    shortest_path,
    // Shortest path algorithms
    single_source_shortest_path,
    spanning_tree_weight,
    strongly_connected_components,
    to_adjacency_list,
    topological_sort,
    traversegraph,
    // Connected components
    undirected_connected_components,
    // Graph utilities
    validate_graph,
    weakly_connected_components,
    LaplacianType,
    MSTAlgorithm,
    // Max flow types (v0.2.0)
    MaxFlowResult,
    // Enums and types
    ShortestPathMethod,
    TraversalOrder,
};

// Graph Laplacian and spectral graph operations
pub mod graph_laplacian;
pub use graph_laplacian::{
    fiedler_vector, graph_laplacian, normalized_laplacian, random_walk_laplacian,
};

// Re-export warnings from scipy for compatibility
pub struct SparseEfficiencyWarning;
pub struct SparseWarning;

/// Check if an object is a sparse array
#[allow(dead_code)]
pub fn is_sparse_array<T>(obj: &dyn SparseArray<T>) -> bool
where
    T: scirs2_core::SparseElement + std::ops::Div<Output = T> + PartialOrd + 'static,
{
    sparray::is_sparse(obj)
}

/// Check if an object is a symmetric sparse array
#[allow(dead_code)]
pub fn is_sym_sparse_array<T>(obj: &dyn SymSparseArray<T>) -> bool
where
    T: scirs2_core::SparseElement
        + std::ops::Div<Output = T>
        + scirs2_core::Float
        + PartialOrd
        + 'static,
{
    obj.is_symmetric()
}

/// Check if an object is a sparse matrix (legacy API)
#[allow(dead_code)]
pub fn is_sparse_matrix(obj: &dyn std::any::Any) -> bool {
    obj.is::<CsrMatrix<f64>>()
        || obj.is::<CscMatrix<f64>>()
        || obj.is::<CooMatrix<f64>>()
        || obj.is::<DokMatrix<f64>>()
        || obj.is::<LilMatrix<f64>>()
        || obj.is::<DiaMatrix<f64>>()
        || obj.is::<BsrMatrix<f64>>()
        || obj.is::<SymCsrMatrix<f64>>()
        || obj.is::<SymCooMatrix<f64>>()
        || obj.is::<CsrMatrix<f32>>()
        || obj.is::<CscMatrix<f32>>()
        || obj.is::<CooMatrix<f32>>()
        || obj.is::<DokMatrix<f32>>()
        || obj.is::<LilMatrix<f32>>()
        || obj.is::<DiaMatrix<f32>>()
        || obj.is::<BsrMatrix<f32>>()
        || obj.is::<SymCsrMatrix<f32>>()
        || obj.is::<SymCooMatrix<f32>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csr_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array =
            CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));
    }

    #[test]
    fn test_coo_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array =
            CooArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));
    }

    #[test]
    fn test_dok_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = DokArray::from_triplets(&rows, &cols, &data, shape).expect("Operation failed");

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));

        // Test setting and getting values
        let mut array = DokArray::<f64>::new((2, 2));
        array.set(0, 0, 1.0).expect("Operation failed");
        array.set(1, 1, 2.0).expect("Operation failed");

        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 0.0);
        assert_eq!(array.get(1, 1), 2.0);

        // Test removing zeros
        array.set(0, 0, 0.0).expect("Operation failed");
        assert_eq!(array.nnz(), 1);
    }

    #[test]
    fn test_lil_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = LilArray::from_triplets(&rows, &cols, &data, shape).expect("Operation failed");

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));

        // Test setting and getting values
        let mut array = LilArray::<f64>::new((2, 2));
        array.set(0, 0, 1.0).expect("Operation failed");
        array.set(1, 1, 2.0).expect("Operation failed");

        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 0.0);
        assert_eq!(array.get(1, 1), 2.0);

        // Test sorted indices
        assert!(array.has_sorted_indices());

        // Test removing zeros
        array.set(0, 0, 0.0).expect("Operation failed");
        assert_eq!(array.nnz(), 1);
    }

    #[test]
    fn test_dia_array() {
        use scirs2_core::ndarray::Array1;

        // Create a 3x3 diagonal matrix with main diagonal + upper diagonal
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
        ];
        let offsets = vec![0, 1]; // Main diagonal and k=1
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).expect("Operation failed");

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5); // 3 on main diagonal, 2 on upper diagonal
        assert!(is_sparse_array(&array));

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(1, 1), 2.0);
        assert_eq!(array.get(2, 2), 3.0);
        assert_eq!(array.get(0, 1), 4.0);
        assert_eq!(array.get(1, 2), 5.0);
        assert_eq!(array.get(0, 2), 0.0);

        // Test from_triplets
        let rows = vec![0, 0, 1, 1, 2];
        let cols = vec![0, 1, 1, 2, 2];
        let data_vec = vec![1.0, 4.0, 2.0, 5.0, 3.0];

        let array2 =
            DiaArray::from_triplets(&rows, &cols, &data_vec, shape).expect("Operation failed");

        // Should have same values
        assert_eq!(array2.get(0, 0), 1.0);
        assert_eq!(array2.get(1, 1), 2.0);
        assert_eq!(array2.get(2, 2), 3.0);
        assert_eq!(array2.get(0, 1), 4.0);
        assert_eq!(array2.get(1, 2), 5.0);

        // Test conversion to other formats
        let csr = array.to_csr().expect("Operation failed");
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 1), 4.0);
    }

    #[test]
    fn test_format_conversions() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        // Create a COO array
        let coo =
            CooArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

        // Convert to CSR
        let csr = coo.to_csr().expect("Operation failed");

        // Check values are preserved
        let coo_dense = coo.to_array();
        let csr_dense = csr.to_array();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(coo_dense[[i, j]], csr_dense[[i, j]]);
            }
        }
    }

    #[test]
    fn test_dot_product() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        // Create arrays in different formats
        let coo =
            CooArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");
        let csr =
            CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

        // Compute dot product (matrix multiplication)
        let coo_result = coo.dot(&coo).expect("Operation failed");
        let csr_result = csr.dot(&csr).expect("Operation failed");

        // Check results match
        let coo_dense = coo_result.to_array();
        let csr_dense = csr_result.to_array();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(coo_dense[[i, j]], csr_dense[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sym_csr_array() {
        // Create a symmetric matrix
        let data = vec![2.0, 1.0, 2.0, 3.0, 0.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 2, 0, 1, 2];
        let indptr = vec![0, 1, 3, 7];

        let sym_matrix =
            SymCsrMatrix::new(data, indptr, indices, (3, 3)).expect("Operation failed");
        let sym_array = SymCsrArray::new(sym_matrix);

        assert_eq!(sym_array.shape(), (3, 3));
        assert!(is_sym_sparse_array(&sym_array));

        // Check values
        assert_eq!(SparseArray::get(&sym_array, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&sym_array, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&sym_array, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array, 1, 2), 3.0);
        assert_eq!(SparseArray::get(&sym_array, 2, 1), 3.0); // Symmetric element

        // Convert to standard CSR
        let csr = SymSparseArray::to_csr(&sym_array).expect("Operation failed");
        assert_eq!(csr.nnz(), 10); // Full matrix with symmetric elements
    }

    #[test]
    fn test_sym_coo_array() {
        // Create a symmetric matrix in COO format
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym_matrix = SymCooMatrix::new(data, rows, cols, (3, 3)).expect("Operation failed");
        let sym_array = SymCooArray::new(sym_matrix);

        assert_eq!(sym_array.shape(), (3, 3));
        assert!(is_sym_sparse_array(&sym_array));

        // Check values
        assert_eq!(SparseArray::get(&sym_array, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&sym_array, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&sym_array, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array, 1, 2), 3.0);
        assert_eq!(SparseArray::get(&sym_array, 2, 1), 3.0); // Symmetric element

        // Test from_triplets with enforce symmetry
        // Input is intentionally asymmetric - will be fixed by enforce_symmetric=true
        let rows2 = vec![0, 0, 1, 1, 2, 1, 0];
        let cols2 = vec![0, 1, 1, 2, 2, 0, 2];
        let data2 = vec![2.0, 1.5, 2.0, 3.5, 1.0, 0.5, 0.0];

        let sym_array2 = SymCooArray::from_triplets(&rows2, &cols2, &data2, (3, 3), true)
            .expect("Operation failed");

        // Should average the asymmetric values
        assert_eq!(SparseArray::get(&sym_array2, 0, 1), 1.0); // Average of 1.5 and 0.5
        assert_eq!(SparseArray::get(&sym_array2, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array2, 0, 2), 0.0); // Zero element
    }

    #[test]
    fn test_construct_sym_utils() {
        // Test creating an identity matrix
        let eye = construct_sym::eye_sym_array::<f64>(3, "csr").expect("Operation failed");

        assert_eq!(eye.shape(), (3, 3));
        assert_eq!(SparseArray::get(&*eye, 0, 0), 1.0);
        assert_eq!(SparseArray::get(&*eye, 1, 1), 1.0);
        assert_eq!(SparseArray::get(&*eye, 2, 2), 1.0);
        assert_eq!(SparseArray::get(&*eye, 0, 1), 0.0);

        // Test creating a tridiagonal matrix - with coo format since csr had issues
        let diag = vec![2.0, 2.0, 2.0];
        let offdiag = vec![1.0, 1.0];

        let tri =
            construct_sym::tridiagonal_sym_array(&diag, &offdiag, "coo").expect("Operation failed");

        assert_eq!(tri.shape(), (3, 3));
        assert_eq!(SparseArray::get(&*tri, 0, 0), 2.0); // Main diagonal
        assert_eq!(SparseArray::get(&*tri, 1, 1), 2.0);
        assert_eq!(SparseArray::get(&*tri, 2, 2), 2.0);
        assert_eq!(SparseArray::get(&*tri, 0, 1), 1.0); // Off-diagonal
        assert_eq!(SparseArray::get(&*tri, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&*tri, 1, 2), 1.0);
        assert_eq!(SparseArray::get(&*tri, 0, 2), 0.0); // Zero element

        // Test creating a banded matrix
        let diagonals = vec![
            vec![2.0, 2.0, 2.0, 2.0, 2.0], // Main diagonal
            vec![1.0, 1.0, 1.0, 1.0],      // First off-diagonal
            vec![0.5, 0.5, 0.5],           // Second off-diagonal
        ];

        let band = construct_sym::banded_sym_array(&diagonals, 5, "csr").expect("Operation failed");

        assert_eq!(band.shape(), (5, 5));
        assert_eq!(SparseArray::get(&*band, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&*band, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&*band, 0, 2), 0.5);
        assert_eq!(SparseArray::get(&*band, 2, 0), 0.5); // Symmetric element
    }

    #[test]
    fn test_sym_conversions() {
        // Create a symmetric matrix
        // Lower triangular part only
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym_coo = SymCooArray::from_triplets(&rows, &cols, &data, (3, 3), true)
            .expect("Operation failed");

        // Convert to symmetric CSR
        let sym_csr = sym_coo.to_sym_csr().expect("Operation failed");

        // Check values are preserved
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    SparseArray::get(&sym_coo, i, j),
                    SparseArray::get(&sym_csr, i, j)
                );
            }
        }

        // Convert to standard formats
        let csr = SymSparseArray::to_csr(&sym_coo).expect("Operation failed");
        let coo = SymSparseArray::to_coo(&sym_csr).expect("Operation failed");

        // Check full symmetric matrix in standard formats
        assert_eq!(csr.nnz(), 7); // Accounts for symmetric pairs
        assert_eq!(coo.nnz(), 7);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(SparseArray::get(&csr, i, j), SparseArray::get(&coo, i, j));
                assert_eq!(
                    SparseArray::get(&csr, i, j),
                    SparseArray::get(&sym_csr, i, j)
                );
            }
        }
    }
}
