# scirs2-linalg Development TODO

## v0.3.3 — COMPLETED

### Iterative Solvers
- GMRES (restarted) with configurable restart parameter
- GMRES-DR / recycled Krylov subspace (GCRO-DR style) — rewritten Feb 26, 2026
- Augmented Krylov (LGMRES-style) deflation
- Preconditioned Conjugate Gradient (PCG) with flexible preconditioning
- BiCGStab (stabilized bi-conjugate gradient)
- MINRES, SYMMLQ for symmetric indefinite systems
- Arnoldi iteration and thick-restart Lanczos
- Rewritten Lanczos QL eigensolver (fixed Feb 26, 2026)

### Randomized Linear Algebra
- Randomized SVD: Halko-Martinsson-Tropp with power iteration and oversampling
- Nystrom extension for low-rank PSD kernel approximation
- Randomized eigensolvers via subspace iteration
- Sketching: Gaussian sketch, CountSketch, SRHT (Subsampled Randomized Hadamard Transform)

### Tensor Decompositions
- CP-ALS (Canonical Polyadic via Alternating Least Squares)
- Tucker-HOOI (Higher-Order Orthogonal Iteration)
- Tensor contractions and mode-n products
- Hierarchical Tucker basics
- Tensor-train format representation

### Matrix Functions
- `expm` via Pade approximant with scaling and squaring
- `logm` via inverse scaling and squaring (Schur-based)
- `sqrtm` via Schur decomposition (Björck-Hammarling)
- `signm` via Newton iteration
- Matrix trigonometric functions (Schur-based): sin, cos, tan, sinh, cosh
- Polar decomposition (QDWH algorithm)
- Pade approximant module for arbitrary rational approximations

### Control Theory
- Continuous algebraic Riccati equation (CARE) via Newton + Hamiltonian Schur
- Discrete algebraic Riccati equation (DARE)
- Lyapunov equation (continuous and discrete, Bartels-Stewart)
- Sylvester equation solver (Bartels-Stewart with 2x2 block fix, Feb 26, 2026)
- Controllability / observability Gramians
- Balanced truncation model order reduction

### Structured Matrices
- Toeplitz matvec via FFT (O(n log n))
- Circulant diagonalization via FFT
- Hankel matvec
- Cauchy matrix: O(n^2) matvec, displacement structure
- Companion matrix and polynomial root finding
- Block tridiagonal direct solver

### Numerical Analysis
- Perturbation bounds for eigenvalues and singular values (Davis-Kahan, Weyl)
- Backward error analysis for linear systems
- Numerical range (field of values) estimation
- Condition number estimation (LAPACK-style power method)
- Matrix pencil solver for polynomial eigenvalue problems

### Other Additions
- CUR decomposition (DEIM-based column/row selection)
- Nuclear norm minimization (alternating projections, soft-impute)
- Matrix completion with soft-impute algorithm
- Indefinite LDL^T factorization (Bunch-Kaufman pivoting)
- Sparse-dense hybrid operations
- `number_theory.rs`: modular arithmetic, integer lattice algorithms

---

## v0.4.0 — Planned

### Mixed-Precision Arithmetic
- [x] f16/bf16 storage with f32 accumulation for matmul — Implemented in v0.4.0 (`mixed_precision/f16_gemm.rs`, `mixed_precision/gemm.rs`, `mixed_precision/operations.rs`)
- [x] Iterative refinement with higher-precision residual correction — Implemented in v0.4.0
- [x] Automatically select precision based on condition number estimate — implemented in v0.4.2 (`auto_precision.rs`)

### Structured Matrix Exploits
- [x] Hierarchical matrix (H-matrix) compression for dense-but-rank-structured matrices — Implemented in v0.4.2 (`hmatrix.rs`: ACA, matvec, SVD recompression, Frobenius norm)
- [x] H^2-matrix arithmetic (O(n log n) matrix-vector products) — Implemented in v0.4.0 (`hmatrix_h2.rs`)
- [x] Sequentially semi-separable (SSS) matrix operations — Implemented in v0.4.0 (`sss_matrix.rs`)

### Distributed Linear Algebra
- [x] Distributed dense matmul (ScaLAPACK-style 2D block cyclic layout) — Implemented in v0.4.0 (`distributed/algorithms/gemm.rs`, SUMMA)
- [x] Distributed QR via Householder with communication-avoiding variants — Implemented in v0.4.0 (`distributed/algorithms/qr.rs`, `scalable.rs`: TSQR)
- [x] Distributed SVD via Lanczos — Implemented in v0.4.0 (`distributed/algorithms/svd.rs`)

### GPU Acceleration
- [x] GPU-accelerated GEMM via OxiBLAS GPU backend — Implemented in v0.4.0 (`gpu_gemm/` module)
- [x] GPU eigensolvers (cuSOLVER-equivalent in pure Rust) — implemented in v0.4.2 (`gpu_eigen.rs`)
- [x] Mixed CPU/GPU solver: factor on GPU, refine on CPU — implemented in v0.4.2 (`gpu_eigen.rs`)

### Additional Algorithms
- [x] Rank-revealing QR (RRQR) with column pivoting — Implemented in v0.4.0
- [x] URV decomposition for rank-deficient systems — Implemented in v0.4.0
- [x] Contour integral eigensolver (FEAST) — Implemented in v0.4.0
- [x] Zolotarev rational approximations for matrix functions — Implemented in v0.4.0 (`matrix_functions_zolotarev.rs`)

---

## Known Issues / Technical Debt

- Some matrix function files exceed 2000 lines; use `rslines 50` to find candidates for splitting
- Lanczos eigensolver was rewritten Feb 26, 2026 after QL deflation bug; needs more stress tests on near-degenerate spectra
- Bartels-Stewart Sylvester 2x2 block handling was patched Feb 26, 2026; audit complex case
- GMRES recycled Krylov was substantially rewritten Feb 26, 2026; regression tests cover Poisson/convection-diffusion but not all corner cases
- Quantization-aware operations in `quantization/` need benchmarks comparing to GGML/bitsandbytes reference
- Control theory module (`control/`) lacks integration tests against MATLAB/Octave reference values

---

## Wave 72 — Einstein summation engine (2026-05-06)

- [x] **General tensor contraction (Einstein summation engine)** (completed 2026-05-07)
  - **Goal:** Fill the two stubs at `scirs2-linalg/src/autograd/tensor_algebra.rs:249` and `:509` with a real Einstein-summation implementation.
  - **Design:** `pub fn einsum(eq: &str, ops: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, EinsumError>` and gradient pass `pub fn einsum_grad(eq: &str, grad_out: ArrayViewD<f64>, ops: &[ArrayViewD<f64>]) -> Vec<ArrayD<f64>>`. Equation parser: tokenise on `","` and `"->"`. Build index-correspondence table: output shape from output indices, summation indices = union − output. Iterate output cells with nested loops over summation indices. Shortcuts: 2-operand binary contraction → general_dot when eq matches "ij,jk->ik"; diagonal "ii->i" and trace "ii->" get dedicated paths. Multi-operand: sequence pairwise, smallest-intermediate-first. Gradient: standard einsum rule, swap op[k] indices into output position. v0.4.4 scope: arbitrary-rank, no-ellipsis-broadcasting-mismatch; ellipsis "...ij,...jk->...ik" for same-batch-size operands.
  - **Files:** `scirs2-linalg/src/autograd/tensor_algebra.rs` (fill stubs, add private helpers); `scirs2-linalg/src/autograd/mod.rs` (re-export `einsum`, `einsum_grad`, `EinsumError`).
  - **Prerequisites:** existing `general_dot` / `kron` / `swapaxes` in autograd.
  - **Tests:** ≥8 in `tests/tensor_algebra_einsum_tests.rs`: einsum("ij,jk->ik",A,B)==A.dot(B); trace "ii->"; diagonal "ii->i"; Frobenius "ij,ij->"; 3-operand sequential matmul to 1e-12; rank-3 batched matmul shape; ellipsis batch; autograd gradient vs numerical to 1e-8.
  - **Risk:** Edge case "ii" self-contraction. Mitigation: explicit detection + dedicated path.

## Wave 74 — Autograd factorizations n×n (2026-05-08)

- [x] **Autodiff backward passes for general n×n factorizations** (planned 2026-05-08, resumed 2026-05-15) (DONE Wave 75, 2026-05-15)
  - **Goal:** Wire the FD-validated Murray (2016) / Giles (2008) backward formulas into the live `scirs2-autograd` Op-trait API. End-users get correct gradients for n > 2 from `scirs2_autograd::tensor_ops::{cholesky,lu,qr,sqrtm_pd,logm}`.
  - **Wave 75 refinements applied:**
    - `pinv` backward already correct at `scirs2-autograd/src/tensor_ops/matrix_ops.rs:121-196` — NOT overwritten.
    - `sqrtm` ships as `sqrtm_pd` (SPD-restricted; Sylvester backward only well-posed for SPD). General `sqrtm` deferred.
    - `MatrixLogOp` / `MatrixSqrtOp` grads live in `matrix_functions.rs:756-758` (NOT `decomposition_ops.rs`). Target `matrix_functions.rs` for Slice 2B.
    - LU/QR backward use combined multi-output Op (NOT SVDExtractOp pass-through anti-pattern).
    - Zero FD-check tests exist on live Op grads today — Slice 2 adds ≥1 FD-check per fixed op.
  - **Files (Slice 2A):** `scirs2-autograd/src/tensor_ops/decomposition_ops.rs` — replace `CholeskyOp::grad`, `LUOp::grad`, `QROp::grad` with Murray/Townsend formulas. Keep `LUExtractOp` / `QRExtractOp` as thin pure-forward extractors (parent Op owns gradient).
  - **Files (Slice 2B):** `scirs2-autograd/src/tensor_ops/matrix_functions.rs` — replace `MatrixSqrtOp::grad` (Sylvester, SPD-only) and `MatrixLogOp::grad` (Daleckii-Krein spectral expansion). Delete dead files `scirs2-linalg/src/autograd/{factorizations,special,transformations,batch}.rs`. Update `scirs2-linalg/src/autograd/mod.rs` doc comment. Flip this item `[~]` → `[x]`.
  - **Tests (mandatory):** 5 new end-to-end integration tests in `scirs2-linalg/tests/autograd_factorizations_nxn_tests.rs` — each includes an FD-vs-analytical-grad central-difference assertion (≤1e-6 well-conditioned, ≤1e-4 ill-conditioned):
    - `cholesky_backward_via_autograd_5x5_spd` (FD-check)
    - `lu_backward_via_autograd_5x5_general` (FD-check)
    - `qr_backward_via_autograd_5x5_orthogonal` (FD-check)
    - `pinv_backward_via_autograd_5x3_overdetermined` (verification only — no overwrite)
    - `sqrtm_pd_backward_via_autograd_4x4_spd` (FD-check, SPD input)
    - `logm_backward_via_autograd_3x3_via_pade` (FD-check)
  - **Completion criteria:** All 6 above + existing 11 reference-math tests pass; 4 dead files deleted; `cargo clippy -p scirs2-autograd -p scirs2-linalg --all-features --all-targets -- -D warnings` zero warnings.
