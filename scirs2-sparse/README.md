# scirs2-sparse

[![crates.io](https://img.shields.io/crates/v/scirs2-sparse)](https://crates.io/crates/scirs2-sparse)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-sparse)](https://docs.rs/scirs2-sparse)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**Sparse matrix library for Rust, modeled after SciPy's sparse module.**

`scirs2-sparse` provides comprehensive sparse matrix formats, iterative solvers, eigenvalue algorithms, preconditioners, graph algorithms, and advanced sparse linear algebra — all in pure Rust with no C or Fortran dependencies.

Tested: 1093/1093 tests passing with default features, 1093/1093 with `--all-features` (verified 2026-07-15).

## Installation

```toml
[dependencies]
scirs2-sparse = "0.6.1"
```

With optional acceleration:

```toml
[dependencies]
scirs2-sparse = { version = "0.6.1", features = ["parallel", "simd"] }
```

## Features (v0.6.1)

### Sparse Matrix Formats

| Format | Type | Best For |
|--------|------|----------|
| `CsrArray` / `CsrMatrix` | Compressed Sparse Row | Row slicing, SpMV |
| `CscArray` / `CscMatrix` | Compressed Sparse Column | Column slicing, SpMV^T |
| `CooArray` / `CooMatrix` | Coordinate (triplet) | Incremental construction |
| `DokArray` / `DokMatrix` | Dictionary of Keys | Random element access |
| `LilArray` / `LilMatrix` | List of Lists | Row-wise incremental build |
| `DiaArray` / `DiaMatrix` | Diagonal | Banded matrices |
| `BsrArray` / `BsrMatrix` | Block Sparse Row | Block-structured problems |
| `EllpackArray` | ELLPACK | GPU-friendly, regular nnz/row |
| `BcsrArray` | Block CSR | Dense block substructure |

Additional: `SymCsrArray`, `SymCooArray` for symmetric matrices with half-storage.

### Iterative Solvers

- Conjugate Gradient (CG) for SPD systems
- BiCG, BiCGSTAB, CGS
- GMRES (restarted) and LGMRES (augmented with deflation vectors)
- MINRES for symmetric indefinite systems
- SYMMLQ
- GCROT, TFQMR
- SOR, SSOR iterative relaxation
- Saddle-point system solver (block preconditioned)
- LSQR and LSMR for least-squares problems

### Preconditioners

- Jacobi (diagonal scaling)
- SSOR (symmetric successive over-relaxation)
- Incomplete Cholesky (IC) factorization
- Incomplete LU (ILU) with fill-level control
- SPAI (Sparse Approximate Inverse)
- Block Jacobi preconditioner
- Additive Schwarz (domain decomposition)
- Algebraic Multigrid (AMG) — smoothed aggregation
- Hierarchical matrix (H-matrix) approximation

### Eigenvalue Solvers

- LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) for extreme eigenvalues
- IRAM (Implicitly Restarted Arnoldi Method) for general sparse matrices
- Shift-and-invert mode for interior eigenvalues
- Generalized eigenvalue problem `Ax = λBx`
- Lanczos iteration for symmetric matrices
- 2-norm and condition number estimation

### Graph Algorithms (csgraph module)

- Shortest paths: Dijkstra (binary heap), Bellman-Ford, Floyd-Warshall
- Connected components (undirected, strongly connected, weakly connected)
- BFS and DFS traversal with path reconstruction
- Minimum spanning tree: Kruskal and Prim
- Graph Laplacian matrices: standard, normalized, random-walk
- Algebraic connectivity (Fiedler value and vector)
- Topological sort
- Spectral clustering via sparse eigensolver

### Sparse Linear Algebra Utilities

- SpMV (sparse matrix-vector multiply) and SpMM (sparse-dense matrix multiply)
- Sparse Kronecker product and Kronecker sum
- Block diagonal construction
- Horizontal and vertical stacking
- Matrix norms: Frobenius, 1-norm, infinity-norm, spectral norm (power method)
- Sparse matrix exponential `expm` and `expm_multiply`
- Linear operator abstraction with composition, transpose, power

### Format Conversions

Bidirectional conversion among all formats: COO, CSR, CSC, BSR, LIL, DOK, DIA, ELLPACK, dense ndarray.

### Advanced Formats

- DIA (diagonal format) with efficient matvec for banded matrices
- BCSR (Block CSR) for problems with dense block substructure
- ELLPACK for GPU-friendly uniform-nnz storage

### Domain Decomposition

- Additive Schwarz method with overlap
- Restricted Additive Schwarz (RAS)
- Coarse-grid correction (two-level Schwarz)

### Neural Adaptive Sparse

- Neural network for adaptive sparsity pattern prediction
- Learned sparse preconditioner
- Data-driven reordering heuristics

## Usage Examples

### Build a CSR matrix from triplets

```rust
use scirs2_sparse::csr_array::CsrArray;

let rows = vec![0usize, 0, 1, 2, 2];
let cols = vec![0usize, 2, 2, 0, 1];
let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];

let a = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;
println!("nnz = {}", a.nnz());
```

### Sparse matrix-vector multiply

```rust
use scirs2_sparse::linalg::spmv;

let b = vec![1.0_f64, 1.0, 1.0];
let x = spmv(&a, &b)?;
```

### Iterative solve with preconditioner

```rust
use scirs2_sparse::linalg::solvers::{lgmres, LgmresParams};
use scirs2_sparse::linalg::preconditioners::IluPreconditioner;

let prec = IluPreconditioner::new(&a, 0)?;
let params = LgmresParams { tol: 1e-10, maxiter: 500, inner_m: 30, ..Default::default() };
let x = lgmres(&a, &b, Some(&prec), params)?;
```

### LOBPCG eigenvalue solver

```rust
use scirs2_sparse::linalg::eigen::lobpcg;

// Compute 5 smallest eigenvalues
let (eigenvalues, eigenvectors) = lobpcg(&a, 5, None, Some(1e-8), Some(300))?;
```

### Graph algorithms

```rust
use scirs2_sparse::csgraph::{shortest_path, connected_components};

let (n_components, labels) = connected_components(&adjacency, false)?;
let dist_matrix = shortest_path(&adjacency, None, false, true)?;
```

### Algebraic multigrid preconditioner

```rust
use scirs2_sparse::linalg::algebraic_multigrid::SmoothedAggregationAMG;

let amg = SmoothedAggregationAMG::setup(&a, Default::default())?;
let x = amg.solve(&b, Some(1e-10), Some(100))?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Multi-threaded SpMV and solver kernels via Rayon (`scirs2-core/parallel`, in `default`) |
| `simd` | SIMD-accelerated dot product and SpMV inner loops (`scirs2-core/simd`) |
| `random` | RNG support for neural-adaptive-sparse and randomized algorithms (`scirs2-core/random`, in `default`) |
| `gpu` | GPU sparse BLAS APIs (`scirs2-core/gpu`) — see the CPU-simulation caveat under "GPU Sparse BLAS" in `TODO.md`: real algorithms, CPU-executed |

Note: this crate's own `Cargo.toml` has no `serde` feature (corrected 2026-07-15 — the previous table entry named a feature that does not exist here).

## Links

- [API Documentation](https://docs.rs/scirs2-sparse)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
