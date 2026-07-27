# scirs2-sparse Development TODO

## Status: v0.6.3 (released, 2026-07-27)

Untouched by this release's fix work (no sparse-specific changes shipped in 0.6.3); the test results below remain accurate since the crate source is unchanged.

scirs2-sparse's own test suite (freshly re-run 2026-07-22): 1093 tests pass, 3 skipped, 0 failed with default features; 1093 tests pass, 3 skipped, 0 failed with `--all-features`.

## v0.3.3 — COMPLETED

### New Sparse Formats
- ELLPACK/ITPACK format with uniform nnz-per-row storage (GPU-friendly)
- BCSR (Block CSR) for problems with dense block substructure
- `SymCsrArray` / `SymCooArray` — symmetric half-storage variants
- Enhanced index dtype handling with automatic i32/i64 selection

### Eigenvalue Solvers
- LOBPCG (Locally Optimal Block Preconditioned CG) for extreme eigenvalues
- IRAM (Implicitly Restarted Arnoldi Method) for non-symmetric matrices
- Shift-and-invert eigenvalue mode for interior spectrum
- Generalized eigenvalue problem `Ax = λBx`
- Lanczos iteration for symmetric SPD matrices

### Advanced Preconditioners
- Block Jacobi preconditioner with block-diagonal factorization
- SPAI (Sparse Approximate Inverse) via minimization in Frobenius norm
- Additive Schwarz method with configurable overlap
- Restricted Additive Schwarz (RAS)
- Two-level Schwarz with coarse correction
- Smoothed aggregation AMG (Algebraic Multigrid) — full setup and solve cycle

### Iterative Solvers
- SYMMLQ for symmetric indefinite systems
- LGMRES (augmented GMRES with deflation)
- Recycled Krylov (GCRO-DR style)
- LSQR and LSMR for least-squares problems
- SOR and SSOR iterative relaxation solvers
- Saddle-point block preconditioned solver
- GCROT and TFQMR

### Graph Algorithms (csgraph)
- Graph Laplacian: standard, normalized, random-walk variants
- Algebraic connectivity (Fiedler eigenvalue/vector) via LOBPCG
- Spectral clustering using sparse eigensolver output
- Enhanced connected component labeling with weak/strong modes

### Hierarchical Matrices
- H-matrix structure (row/column cluster tree)
- Adaptive cross approximation (ACA) for off-diagonal block compression
- H-matrix-vector multiply
- H-matrix preconditioner apply

### Domain Decomposition
- Additive Schwarz with overlap
- Restricted Additive Schwarz
- Two-level method with coarse-grid solve
- Subdomain interface identification from CSR graph

### Neural Adaptive Sparse
- Neural network for sparsity pattern prediction
- Learned sparse preconditioner training loop
- Data-driven fill-reducing reordering heuristics

### Other Additions
- Sparse matrix exponential (`expm` via scaling/squaring on CSR)
- `expm_multiply`: compute `expm(A) @ v` without forming `expm(A)`
- Saddle-point system (block 2x2) specialized solver
- Smoothed aggregation setup: strength of connection, aggregation, prolongation smoothing
- Sparse format benchmark suite

---

## v0.4.0 — Planned

### GPU Sparse BLAS
- [x] CSR SpMV on GPU via compute shaders (OxiFFT GPU backend model) — Implemented in v0.4.0 (`gpu/spmv.rs`)
- [x] GPU-accelerated COO/CSR construction from triplets — Implemented in v0.4.0 (`gpu/construction.rs`)
- [x] GPU BiCGSTAB and CG solvers — Implemented in v0.4.0 (`gpu/solvers.rs`)
- [x] Mixed CPU/GPU preconditioning (ILU on CPU, SpMV on GPU) — Implemented in v0.4.2 (`gpu_preconditioner.rs`)

Status (verified 2026-07-15): all four items above are real, tested algorithms, but per their own module doc comments they currently execute as CPU-side simulations of the intended GPU pattern, not actual GPU dispatch — `gpu/spmv.rs` states it provides "CPU-side SIMD-friendly implementations that serve as compute-shader placeholders" (`GpuSpMvBackend::Cpu` is the only wired variant; `WebGpu` is "feature-gated, not yet wired"), and `gpu/solvers.rs` states "The CPU computation paths mirror what GPU compute shaders would execute." Checked off because the row-parallel/chunked algorithms themselves are complete and correct, not because real GPU hardware dispatch exists.

### Distributed Sparse Solvers
- [x] Distributed CSR with row-based partitioning — Implemented in v0.4.0 (`distributed/csr.rs`, `distributed/partition.rs`)
- [x] Distributed SpMV with halo exchange — Implemented in v0.4.0 (`distributed/halo_exchange.rs`)
- [x] Distributed AMG via `scirs2-core` ring allreduce — Implemented in v0.4.0 (`distributed/dist_amg.rs`)

Status (verified 2026-07-15): same caveat as GPU Sparse BLAS above — `distributed/csr.rs` and `distributed/halo_exchange.rs` explicitly document the halo exchange as "simulated via shared memory ... a real distributed implementation would replace the halo broadcast with MPI or similar," and `distributed/dist_amg.rs` uses a local "simulated (shared-memory) communication pattern," not an actual `scirs2-core` ring-allreduce call (no `scirs2_core` import is present in that file at all — the third bullet's "via `scirs2-core` ring allreduce" description is inaccurate as written). The partitioning, halo-identification, and AMG-hierarchy math are real and tested single-process; multi-process/MPI execution is not implemented.

### Graph Algorithm Enhancements
- [x] Approximate graph coloring for parallel Gauss-Seidel — Implemented in v0.4.0
- [x] Nested dissection reordering (via graph partitioning) — Implemented in v0.4.0
- [x] Cuthill-McKee and reverse Cuthill-McKee reordering — Implemented in v0.4.0
- [x] Approximate minimum degree (AMD) reordering — Implemented in v0.4.0

### Format Enhancements
- [x] Sliced ELLPACK (SELL) for variable nnz-per-row with GPU padding — Implemented in v0.4.0
- [x] CSR5 format for GPU-friendly SpMV — Implemented in v0.4.0
- [x] Compressed sparse fiber (CSF) for sparse tensors — Implemented in v0.4.0

### Additional Preconditioners
- [x] Multilevel ILU (MILUE) with coarse correction — Implemented in v0.4.0 (`preconditioners/milue.rs`)
- [x] Sparse approximate inverse via AINV (robust incomplete factorization) — Implemented in v0.4.0 (`preconditioners/ainv.rs`)
- [x] Polynomial preconditioner (Chebyshev acceleration) — Implemented in v0.4.0

---

## Known Issues / Technical Debt

- `krylov.rs` was deleted and replaced by the `krylov/` submodule; ensure no lingering re-exports break downstream
- `neural_adaptive_sparse/neural_network.rs` needs more training data and documented hyperparameters
- [x] H-matrix ACA convergence is validated against dense reference — see `test_aca_rank1`, `test_aca_plus_rank1`, `test_hmatrix_truncate`, `test_aca_error_cases` in `src/hierarchical/low_rank.rs` (Frobenius-error checks against explicit dense reconstructions); `src/hierarchical/cluster_tree.rs` provides `ClusterTree`/`build_cluster_tree`/`admissibility_check` beyond a bare structural sketch
- GPU sparse code lives in the top-level `gpu/` module and `gpu_preconditioner.rs` (not `linalg/`, which has no `feature = "gpu"` gates); see the CPU-simulation caveat under "GPU Sparse BLAS" above
- [x] No source file currently exceeds 2000 lines (verified 2026-07-15: largest is `src/linalg/iterative.rs` at 1800 lines); re-check with `rslines 50` if new files grow past the threshold
- SPAI preconditioner setup cost is O(n * bandwidth^2); document when to prefer it over ILU
- Saddle-point solver assumes 2x2 block structure; generalize to block-n
