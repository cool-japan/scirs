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

- [~] **Autodiff backward passes for general n×n factorizations** (planned 2026-05-08)
  - **Goal:** Replace the six "not yet implemented in autodiff for n > 2" error returns in `autograd/factorizations.rs:48,246,421`, `autograd/special.rs:44,281,562`, `autograd/transformations.rs:115`, `autograd/batch.rs:547` with full implementations. Foundation primitives for normalizing flows (matrix-square-root Jacobian), GP marginal-likelihood (log-det Cholesky), least-squares heads (pseudo-inverse).
  - **Design:** Six derivations from Murray (2016) "Differentiation of the Cholesky decomposition" / Giles (2008):
    1. **LU backward** (`factorizations.rs:48`): `dA = L⁻ᵀ · phi(L^T · dL · L^(-T) - U · dU · U⁻¹) · L⁻ᵀ + L · dU · U⁻¹` where `phi(M)` zeros above-diagonal entries; two triangular solves O(n³).
    2. **QR backward** (`factorizations.rs:246`): `dA = (dQ + Q · phi(M^T - M)) · R⁻ᵀ` where `M = R · dR^T - dQ^T · Q`.
    3. **Cholesky backward** (`factorizations.rs:421`): Murray 2016: `dA = ½ · L⁻ᵀ · phi(L^T · dL + L^T · dL^T) · L⁻¹` where `phi` symmetrises and zeros below-diagonal.
    4. **Pseudo-inverse backward** (`special.rs:44`): SVD-regularized when `cond(A) > 1e10`.
    5. **Matrix square-root backward** (`special.rs:281`): Sylvester-equation `S · dA + dA · S = ds` via existing Bartels-Stewart.
    6. **Matrix log backward** (`special.rs:562`): Padé approximation chained with Schur-based forward `logm`.
  - **Batch dimension** (`batch.rs:547`): rayon `for_each` parallelism on independent sample-wise backward passes.
  - **Files:** `scirs2-linalg/src/autograd/factorizations.rs`, `scirs2-linalg/src/autograd/special.rs`, `scirs2-linalg/src/autograd/transformations.rs`, `scirs2-linalg/src/autograd/batch.rs`, `scirs2-linalg/src/autograd/mod.rs`, `scirs2-linalg/TODO.md`.
  - **Prerequisites:** existing `solve_triangular`, `solve_sylvester`, `logm`, `sqrtm` forward; existing autograd tape.
  - **Tests:** ≥ 12 in `scirs2-linalg/tests/autograd_factorizations_nxn_tests.rs` including `lu_backward_3x3_random` (fd-grad ≤ 1e-7), `qr_backward_thin_5x3`, `cholesky_backward_5x5_spd`, `pinv_backward_overdetermined_5x3`, `sqrtm_backward_4x4_general`, `logm_backward_3x3_via_pade`, `batch_lift_5_x_3x3_lu_consistent`.
  - **Risk:** Numerical stability of pinv for ill-conditioned matrices; mitigated by SVD regularization. Sylvester sqrtm needs Bartels-Stewart 2×2 block fix (Feb 26, 2026 patch — verify conjugate-pair eigenvalues).
  - **Status (2026-05-08, partial / deviated):** Math delivered as standalone reference functions in `tests/autograd_factorizations_nxn_tests.rs` (12 tests, FD-validated to ≤ 1e-4 on ill-conditioned cases, ≤ 1e-6 on well-conditioned). **Tape integration deferred:** `src/autograd/{factorizations,special,transformations,batch,tensor_algebra}.rs` are dead/orphan files — they reference an old `Tensor<F>` API with public `.data`/`.requires_grad`/`.node` fields that no longer exists in `scirs2-autograd::tensor::Tensor<'graph, F>`. `mod.rs` shadows them with empty `pub mod X {}` placeholders since at least Wave 53. Re-enabling them produces 562 build errors. The "stub error returns" the goal asks to replace are unreachable. Future wave: rewrite the autograd module against the current `Op`-trait API and call into the reference math. Item left `[~]`.
