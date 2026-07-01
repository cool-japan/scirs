# TODO_CUDA — Real-Hardware Validation of the oxicuda CUDA Paths

> **Status: DONE (first pass) — validated on real hardware 2026-06-29.** Ran on a
> Linux box with a live NVIDIA RTX A4000 (sm_86, CUDA 12.4). **All 10 oxicuda CUDA
> paths now execute correctly on-device** (§1 build, §2 smoke, §3 correctness, plus
> §4 transparent dispatch and §7 fallbacks — see the §10 sign-off matrix). The
> first run exposed 4 oxicuda root causes (3 of them *silent fabrications*: a no-op
> Cholesky kernel, a comment-only conv kernel, and an f32/f64 PTX type mismatch +
> 1-D launch-geometry bug); all fixed and re-verified. A follow-on hardening pass
> fixed ~9 more sparse kernels, made solver LU genuinely on-device, converted
> fabricated solver QR/SVD/eig to honest CPU fallbacks, and implemented the dnn
> standard-conv kernels — oxicuda workspace: **4,267 tests green, 0 fail, clippy
> clean**. Remaining items (§5 bench tune, §6 multi-GPU, and the oxicuda-blas
> leading-dimension bug behind the Cholesky n>64 blocked path) are recorded in §10
> and are not exercised by the 10 paths. See "First-run results" and "Fixes
> applied" below.

| Field | Value |
|---|---|
| Validation host / engineer | Paperspace Linux box (`nqbxenjhwk`, x86_64) / KitaSan — automated `/ucont`, Claude Opus 4.8 |
| GPU model (and class: GeForce vs data-center) | **NVIDIA RTX A4000, 16 GB** — GA104, compute capability **sm_86**, workstation Ampere (consumer-class f64 = ~1/64 of f32 throughput) |
| NVIDIA driver version | **550.144.03** |
| CUDA / libcuda version | CUDA **12.4** (driver runtime); `nvcc` 12.0; `libcuda.so.550.144.03` |
| `../oxicuda` commit | `abd66b9` (at session start; updated by in-session fixes) |
| SciRS2 branch / commit | `0.6.0` / `d63857f9` (at session start) |
| Date started / finished | started **2026-06-29** / finished _in progress_ |

## First-run results (2026-06-29, RTX A4000) — pre-fix smoke sweep

| scirs path | pattern | result | root cause |
|---|---|---|---|
| special `cuda_erf_batch` | PTX custom kernel | **PASS** | — |
| stats normal pdf/cdf | PTX custom kernel | **PASS** | — |
| symbolic `cuda_eval_batch` | PTX custom kernel | **PASS** | — |
| fft `cuda_fft_1d/ifft_1d` | oxicuda-fft lib | **PASS** | — |
| linalg `cuda_gemm` | oxicuda-blas GEMM | FAIL: invalid PTX | #1 oxicuda-blas/oxicuda-ptx GEMM f64 codegen (f32 reg bank + `0f00000000` literal) |
| interpolate `cuda_eval_gemm` | oxicuda-blas GEMM | FAIL: invalid PTX | #1 (same) |
| optimize `cuda_hessian_vector_product` | oxicuda-blas GEMM | FAIL: invalid PTX | #1 (same) |
| datasets `cuda_regression_target` | oxicuda-blas GEMM | FAIL: invalid PTX | #1 (same) |
| graph `cuda_spmv_csr` | oxicuda-sparse SpMV | FAIL: invalid PTX | #2 oxicuda-sparse f64 PTX codegen |
| linalg `cuda_solve_spd` | oxicuda-solver Cholesky | FAIL: wrong (diff 1.79) | #3 oxicuda-solver Cholesky/solve |
| interpolate `cuda_rbf_solve` | oxicuda-solver Cholesky | FAIL: wrong (diff 0.84) | #3 (same) |
| vision `cuda_convolve_2d` | oxicuda-dnn conv | FAIL: wrong (diff 95) | #4 oxicuda-dnn conv_forward |

Root-cause diagnosis (`ptxas -arch=sm_86` on the dumped GEMM PTX): the naive f64
GEMM kernel declared its value registers `.reg .f32 %f<16>;` and used an f32 zero
literal `0f00000000` while emitting `.f64` instructions — "Arguments mismatch" for
every f64 op. The `oxicuda-ptx` DSL custom kernels (erf etc.) and `oxicuda-fft` are
unaffected, which is why those 4 paths pass.

## Fixes applied (2026-06-29) — all 10 paths now PASS on the A4000

Four oxicuda root causes fixed (each verified on-device + with oxicuda's own
tests + `ptxas -arch=sm_86` + clippy `-D warnings`). A recurring theme: **three of
the four were silent fabrications** — code that compiled, launched, and returned
plausible-but-fake values, only exposed by first real-hardware execution.

1. **oxicuda-blas / oxicuda-ptx GEMM f64 codegen** (unblocked linalg `cuda_gemm`,
   interpolate `cuda_eval_gemm`, optimize `cuda_hessian_vector_product`, datasets
   `cuda_regression_target`). Two layered bugs: (a) `GemmTemplate::generate()`
   declared the value register bank `.reg .f32` and zeroed with `0f00000000` while
   emitting `.f64` ops → invalid PTX; (b) once that loaded, a **launch-geometry
   bug** (kernel derived `row` from a 2-D block but the dispatcher launches 1-D
   blocks → only row 0 written, returned `[[58,64],[0,0]]`). Fix: value bank typed
   to the accumulator + correct `zero_literal()`, full mixed-precision `cvt`
   handling, and a **geometry-independent grid-stride loop** (correct under any
   launch shape, split-K, non-square). Also audited+fixed the other f64-reachable
   generators (`simt.rs`, `bandwidth_opt.rs` incl. a stray `prefetch` operand bug,
   `splitk.rs`, `cooperative.rs` f64 register bug, `epilogue.rs`).
   Files: `oxicuda-ptx/src/{ir/types.rs, templates/gemm.rs}`,
   `oxicuda-blas/src/level3/gemm/{precision.rs(new), simt.rs, bandwidth_opt.rs, splitk.rs, cooperative.rs, epilogue.rs}`.
2. **oxicuda-sparse SpMV f64 PTX** (unblocked graph `cuda_spmv_csr`). Two bugs:
   (a) label `$`-prefix mismatch — branches emitted `bra L__…` but labels were
   defined `$L__…` → `ptxas: Unknown symbol`; (b) `shfl.sync.down.b64` (PTX `shfl`
   supports only `.b32`) → for f64, unpack to two `.b32` halves, shuffle each,
   repack. Files: `oxicuda-sparse/src/{ptx_helpers.rs, ops/spmv.rs}`.
3. **oxicuda-solver Cholesky** (unblocked linalg `cuda_solve_spd`, interpolate
   `cuda_rbf_solve`). `emit_panel_cholesky` was a **no-op kernel** (`let _=(…); ret;`)
   — it never factored, so the solve ran against the unfactored matrix. Fix: a real
   single-CTA right-looking Cholesky (Lower+Upper) in global memory.
   File: `oxicuda-solver/src/dense/cholesky.rs`.
4. **oxicuda-dnn conv_forward** (unblocked vision `cuda_convolve_2d`). The
   single-channel conv routes to the depthwise engine, whose `emit_depthwise_body`
   was **comment-only** (no loads/fma/store) → page-zeroed output, diff 95. Fix: a
   real branchless depthwise cross-correlation (no kernel flip, implicit zero pad).
   File: `oxicuda-dnn/src/conv/fprop/direct.rs`.

### Hardening pass (2026-06-29, after the core 10) — RESOLVED follow-ups
A second pass eradicated the latent bugs the core fixes uncovered. oxicuda
workspace after: **4,267 tests green, 0 fail, clippy clean, ptxas-validated**.
- **oxicuda-sparse** ✅: the label-`$` + f64-`shfl.b64` antipatterns fixed across
  `spmv_bsr/ell/csr5`, `spgemm`, `spmm`, `sptrsv`, `sddmm`, `krylov`,
  `mixed_precision_spmv` (the last was an *advertised-but-never-assembled*
  FP64/FP16/BF16 capability — now real). 36 PTX variants ptxas-validated.
- **oxicuda-solver** ✅: LU is now a **real on-device** strided BLAS-free kernel;
  QR/SVD/eig were *silent fabrications* (column norms read as singular values, the
  un-tridiagonalized diagonal read as eigenvalues, QR never factorizing) → converted
  to **real, correct, loudly-marked CPU fallbacks**; a pre-existing QR Qᵀ/Q bug fixed.
  **Blocked Cholesky (n>64)** — which had relied on the oxicuda-blas level-3 ld bug
  (reproduced: n=80 gave max|A−LLᵀ|≈103.56 / NaN solves) — given its **own strided
  TRSM/SYRK device kernels**, now correct & adversarially verified to n=256.
- **oxicuda-dnn** ✅: `emit_implicit_gemm_body` (standard C>1 conv) and
  `Conv1x1::execute` implemented as real cross-correlation kernels (NCHW+NHWC,
  f32/f64, groups/stride/dilation).

### Still open (deeper; NOT used by the validated factorizations or the 10 paths)
- **oxicuda-blas level-3 GEMM/TRSM ignore leading dimensions** on strided
  sub-blocks. This was the *underlying* cause behind LU and blocked-Cholesky
  corruption; both now sidestep it with their own ld-honoring kernels, but the BLAS
  routines themselves should be fixed (or any future blocked factorization using
  them will be wrong). The contiguous full-matrix GEMM/TRSM (ld==rows) is fine.
- Real **on-device QR/SVD/eig** (currently correct, marked CPU fallbacks) and a
  **tiled/Tensor-Core** GEMM/conv fast path (current kernels are naive
  one-thread-per-output) — large deferred performance work.

## Why this document exists

SciRS2 0.6.x added per-crate CUDA acceleration through the pure-Rust `oxicuda-*`
crates (local `../oxicuda` path dependencies). Each integration lives in a
`src/gpu_cuda.rs` module behind an **off-by-default `cuda` feature**.

**All of it is compile-verified only.** The development machine is macOS, and
Apple dropped NVIDIA support years ago: there is no NVIDIA GPU and no CUDA driver
on the dev box. The code still *compiles* on macOS with `--features cuda` because
`oxicuda-driver` `dlopen`s the NVIDIA driver (`libcuda`) at **runtime**, not at
link time. But the runtime probe —

```rust
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count().map(|c| c > 0).unwrap_or(false)
}
```

— always returns `false` on macOS (driver init fails / zero devices). Every
public CUDA entry point is guarded by this probe, and every CUDA smoke test
early-returns through it:

```rust
if !cuda_is_available() {
    eprintln!("skipping: no NVIDIA CUDA device");
    return;
}
```

**Consequence: not one line of actual device code — no oxicuda call signature, no
buffer upload/download, no kernel launch, no matrix-layout assumption — has ever
executed.** Correctness today rests entirely on hand-derivation and CPU-side
reasoning. This document is the plan to change that.

## Out of scope (read first)

- **wgpu / WebGPU portability layer** (`GpuNdarray` / `GpuContext` in
  `scirs2-core`, plus the per-crate `wgpu` / `gpu_*` modules). These **run on
  macOS via Metal** and are already validated there. They are an entirely
  separate GPU story (f32, portable) from the oxicuda CUDA paths (f64,
  NVIDIA-only). Nothing in this file touches them.
- **`cuda` features that are NOT oxicuda** — see §11. `scirs2-cluster`,
  `scirs2-ndimage`, `scirs2-spatial`, and `scirs2-integration-tests` each have a
  `cuda` feature with a different (or empty) meaning; they are not part of this
  oxicuda validation.

## The oxicuda CUDA surface (the 10 crates under validation)

Each crate has a `src/gpu_cuda.rs` with a `cuda_is_available()` probe, one or
more public entry points, and at least one `*_or_skip` smoke test. The three
implementation patterns:

- **BLAS / Solver (library call):** hands dense linear algebra to `oxicuda-blas`
  / `oxicuda-solver`.
- **PTX → JIT → launch (custom kernel):** generates `f64` PTX with `oxicuda-ptx`,
  then loads + launches it with `oxicuda-launch`.
- **Domain lib:** calls a higher-level oxicuda library (`oxicuda-fft`,
  `oxicuda-sparse`, `oxicuda-dnn`).

| Crate | `cuda` pulls (oxicuda) | Entry points (`src/gpu_cuda.rs::`) | Pattern |
|---|---|---|---|
| scirs2-linalg | driver, memory, blas, solver | `cuda_gemm`, `cuda_solve_spd` | BLAS/Solver |
| scirs2-interpolate | driver, memory, blas, solver | `cuda_rbf_solve`, `cuda_eval_gemm` | BLAS/Solver |
| scirs2-optimize | driver, memory, blas | `cuda_hessian_vector_product` | BLAS/Solver |
| scirs2-datasets | driver, memory, blas | `cuda_regression_target` | BLAS/Solver |
| scirs2-special | driver, memory, launch, ptx | `generate_erf_ptx`, `cuda_erf_batch` | PTX→JIT |
| scirs2-stats | driver, memory, launch, ptx | `generate_normal_pdf_ptx`, `generate_normal_cdf_ptx`, `cuda_normal_pdf_batch`, `cuda_normal_cdf_batch` | PTX→JIT |
| scirs2-symbolic | driver, memory, ptx, launch | `generate_ptx`, `cuda_eval_batch` | PTX→JIT |
| scirs2-fft | driver, memory, fft | `cuda_fft_1d`, `cuda_ifft_1d` | Domain (oxicuda-fft) |
| scirs2-graph | driver, memory, sparse | `cuda_spmv_csr` | Domain (oxicuda-sparse) |
| scirs2-vision | driver, memory, dnn | `cuda_convolve_2d` | Domain (oxicuda-dnn) |

---

## 0. Environment & prerequisites

- [ ] NVIDIA GPU physically present; `nvidia-smi` lists it and a driver version.
- [ ] `libcuda` is loadable at runtime (the driver, not just the CUDA toolkit).
- [ ] Record GPU model, driver version, CUDA version in the header table above.
- [ ] `../oxicuda` is checked out next to this repo (the `cuda` features use
  `path = "../oxicuda/..."` deps) and is at a commit whose **public API matches
  the call sites** in each `gpu_cuda.rs` (see §8 for the drift check).
- [ ] Sanity probe: a small throwaway `main` (or one `cargo test`) that calls
  `oxicuda_driver::init()` and `oxicuda_driver::device::Device::count()` and
  confirms `init().is_ok()` and `count > 0`. If this fails, nothing else here can
  pass — fix the driver/toolkit first.
- [ ] Understand the macOS mechanism you are escaping: oxicuda `dlopen`s the
  driver at runtime, so the crates **compiled** on macOS but `cuda_is_available()`
  returned `false`, so all device code was dead. On this box it must return
  `true` — verify that before trusting anything below.

## 1. Build & feature-gate verification (real CUDA box)

These crates have only ever been built on a **driver-less** host. Rebuild each
with the `cuda` feature where a real driver is present.

- [ ] `cargo build -p scirs2-linalg --features cuda`
- [ ] `cargo build -p scirs2-interpolate --features cuda`
- [ ] `cargo build -p scirs2-optimize --features cuda`
- [ ] `cargo build -p scirs2-datasets --features cuda`
- [ ] `cargo build -p scirs2-special --features cuda`
- [ ] `cargo build -p scirs2-stats --features cuda`
- [ ] `cargo build -p scirs2-symbolic --features cuda`
- [ ] `cargo build -p scirs2-fft --features cuda`
- [ ] `cargo build -p scirs2-graph --features cuda`
- [ ] `cargo build -p scirs2-vision --features cuda`
- [ ] Umbrella facade: `cargo build -p scirs2 --features cuda` (confirm the
  facade's `cuda` feature fans out to the per-crate `cuda` features as intended).
- [ ] `cargo build -p scirs2-integration-tests --features cuda` (its `cuda` =
  `["scirs2-linalg/cuda"]`).
- [ ] No new warnings under `--features cuda` (no-warnings policy).

## 2. First-ever smoke-test execution — HIGHEST RISK

Each crate ships a `*_or_skip` test that, until now, has **always taken the skip
branch** because `cuda_is_available()` was `false`. On this box the probe returns
`true`, so these tests **run the real device path for the very first time** —
exercising every oxicuda call signature, buffer layout, descriptor, and kernel
launch that has never run. Expect to find and fix real bugs here.

- [ ] `cargo test -p scirs2-linalg --features cuda gpu_cuda`
      (`cuda_gemm_or_skip`, `cuda_solve_spd_or_skip`)
- [ ] `cargo test -p scirs2-interpolate --features cuda gpu_cuda`
      (`cuda_rbf_solve_or_skip`, `cuda_eval_gemm_or_skip`)
- [ ] `cargo test -p scirs2-optimize --features cuda gpu_cuda`
      (`cuda_hessian_vector_product_or_skip`)
- [ ] `cargo test -p scirs2-datasets --features cuda`
      (the `gpu_cuda` `*_or_skip` test + the `make_regression` dispatch test)
- [ ] `cargo test -p scirs2-special --features cuda gpu_cuda`
      (`cuda_erf_batch_or_skip`)
- [ ] `cargo test -p scirs2-stats --features cuda gpu_cuda`
      (normal PDF / CDF `*_or_skip`)
- [ ] `cargo test -p scirs2-symbolic --features cuda gpu_cuda`
      (`cuda_eval_linear_or_skip`)
- [ ] `cargo test -p scirs2-fft --features cuda gpu_cuda`
      (`cuda_fft_roundtrip_or_skip`)
- [ ] `cargo test -p scirs2-graph --features cuda gpu_cuda`
      (`cuda_spmv_path_graph_or_skip`)
- [ ] `cargo test -p scirs2-vision --features cuda gpu_cuda`
      (`cuda_convolve_2d_or_skip`)
- [ ] Confirm each test prints its compute (NOT the
  `skipping: no NVIDIA CUDA device` line). If you still see the skip line, the
  probe is returning `false` — fix §0 before trusting anything else.

## 3. Numerical correctness per entry point

Compare every CUDA result against the CPU source-of-truth within tolerance. Note
the two PTX transcendental paths (`erf`, normal CDF) cap mathematical accuracy at
the **Abramowitz & Stegun 7.1.26** polynomial (~1.5e-7 absolute) even though the
arithmetic is full `f64`, so use a ~1e-6 tolerance there, not 1e-12.

### 3a. BLAS / Solver pattern

- [ ] `scirs2-linalg/src/gpu_cuda.rs::cuda_gemm` vs `a.dot(b)`. **Stress
  NON-square shapes** (e.g. 2×3 · 3×2, 17×5 · 5×31): the `Layout::RowMajor` /
  `Transpose::NoTrans` descriptor layout was hand-derived from oxicuda's own
  `../oxicuda/crates/oxicuda-blas/benches/gemm_f64_4096.rs` (a square N×N bench)
  and has **never been runtime-checked on a non-square problem** — a
  transposed / row-vs-col bug would pass on square inputs and only surface here.
- [ ] `scirs2-linalg/src/gpu_cuda.rs::cuda_solve_spd` vs a CPU SPD solve (build
  an SPD `A`, check `A·x ≈ b`). Verifies the Cholesky lower-fill +
  `cholesky_solve` path and the symmetric row/col-major coincidence assumption.
- [ ] `scirs2-interpolate/src/gpu_cuda.rs::cuda_rbf_solve` vs CPU dense SPD solve.
- [ ] `scirs2-interpolate/src/gpu_cuda.rs::cuda_eval_gemm` vs CPU `Phi·weights`
  (the RowMajor-`A` × ColMajor-vector → ColMajor-output layout is the template
  reused by optimize below — validate it here first).
- [ ] `scirs2-optimize/src/gpu_cuda.rs::cuda_hessian_vector_product` vs CPU
  `h.dot(v)` for a non-trivial (and ideally non-symmetric, to catch layout bugs)
  `H`.
- [ ] `scirs2-datasets/src/gpu_cuda.rs::cuda_regression_target` vs CPU `X·coef`.

### 3b. PTX → JIT → launch pattern

- [ ] `scirs2-special/src/gpu_cuda.rs::cuda_erf_batch` vs CPU `crate::erf` across
  the full `f64` range **including the tails** (large `|x|` → ±1). Tolerance
  ~1e-6 (A&S 7.1.26). Also inspect `generate_erf_ptx()` output and confirm it
  contains the expected f64 ops and **no `.approx.f64`** (the honest-f64
  invariant the unit test already asserts).
- [ ] `scirs2-stats/src/gpu_cuda.rs::cuda_normal_pdf_batch` vs CPU normal PDF.
- [ ] `scirs2-stats/src/gpu_cuda.rs::cuda_normal_cdf_batch` vs CPU normal CDF
  (~1e-6, A&S). Inspect `generate_normal_pdf_ptx()` / `generate_normal_cdf_ptx()`.
- [ ] `scirs2-symbolic/src/gpu_cuda.rs::cuda_eval_batch` vs CPU
  `eml::eval_real`. **IMPORTANT — supported-op subset:** the PTX generator
  currently supports **only `{Const, Var, Add, Sub, Mul}`**; `validate_supported`
  rejects everything else (Div, Pow, Neg, and all transcendentals) with
  `CudaEvalError::Unsupported`. So:
    - [ ] Correctness over a representative set of **nested
      Add/Sub/Mul/Const/Var** `LoweredOp` trees (deep nesting, multiple vars,
      multiple consts; this exercises the post-order constant-buffer +
      `fma(a, b, 0)`-as-`Mul` machinery).
    - [ ] Confirm a transcendental / `Div` / `Pow` op returns
      `Err(CudaEvalError::Unsupported)` — **no silent wrong answer** (the unit
      test checks `Sin` already; extend this if/when the subset is broadened).

### 3c. Domain-library pattern

- [ ] `scirs2-fft/src/gpu_cuda.rs::cuda_fft_1d` vs OxiFFT CPU forward FFT, and
  `cuda_ifft_1d` vs CPU inverse. Check the **forward→inverse round-trip ≈
  identity**, both **power-of-two and non-power-of-two** sizes, and the
  **normalization convention**: forward is unnormalized, inverse divides by `N`
  (SciPy convention, matching scirs2-fft CPU `fft` / `ifft`).
- [ ] `scirs2-graph/src/gpu_cuda.rs::cuda_spmv_csr` vs CPU
  `compressed::CsrGraph::spmv`. Verify the **CSR contract**: `i32` indices,
  `row_offsets` length `n_rows + 1` with `row_offsets[last] == nnz`,
  `SpMVAlgo::Adaptive`. Test directed / asymmetric adjacencies and varying
  nnz/row (scalar- vs vector-per-row kernel selection).
- [ ] `scirs2-vision/src/gpu_cuda.rs::cuda_convolve_2d` vs
  `simd_ops::simd_convolve_2d`. **CRITICAL semantics it must reproduce exactly**
  (read the module docs in that file):
    - Output is the **same `H×W` size** as the input, but it is a **`valid`
      convolution** (`out_h = H − k_h + 1`, `out_w = W − k_w + 1`) **embedded at
      offset `(k_half_h, k_half_w)` into a `zeros((H, W))` array** — the border
      ring is **hard zeros**, NOT a zero-padded `same` convolution.
    - It is **cross-correlation** — the kernel is uploaded **AS-IS, with no 180°
      flip** (both `simd_convolve_2d` and `oxicuda-dnn conv_forward` use the cuDNN
      cross-correlation convention).
    - [ ] Verify the **`DnnError::WorkspaceRequired(bytes)` retry path actually
      fires and succeeds**: that branch (allocate a `bytes` `DeviceBuffer<u8>`
      workspace, re-call `conv_forward` with `Some(&mut workspace)`) is only
      reached if the engine selects a scratch-needing algorithm — force / observe
      it on real hardware and confirm it yields the same result.
    - [ ] Confirm the odd-kernel / empty-image / oversized-kernel validation
      still rejects before any device call (already runs GPU-free).

## 4. Transparent runtime-dispatch validation

Two CPU entry points conditionally route to the CUDA path above a size
threshold. The thresholds are deliberately **huge** so unit / doctests stay on
the bit-stable CPU path — meaning the GPU branch is exercised **only** by
explicitly large inputs you must construct here.

### 4a. `scirs2-linalg/src/blas_accelerated.rs::matmul`

Routes to `gpu_cuda::cuda_gemm` when **all** hold: `F == f64`,
`cuda_is_available()`, and `a.nrows() * a.ncols() * b.ncols() >=
CUDA_MATMUL_MIN_FLOPS`, where:

```rust
const CUDA_MATMUL_MIN_FLOPS: usize = 1 << 21; // 2,097,152  (≈ a 128×128×128 GEMM)
```

- [ ] Construct an `f64` GEMM with `m * k * n >= 2_097_152` and confirm the CUDA
  branch actually engages (e.g. temporary logging, or diff against `a.dot(b)`).
- [ ] CPU vs CUDA agree within tolerance (GPU accumulates in a different
  associative order → last-ULP differences are acceptable).
- [ ] `f32` inputs of the same size stay on CPU (the `TypeId == f64` guard).
- [ ] Sub-threshold `f64` inputs stay on CPU (bit-identical to before).
- [ ] On a forced CUDA `Err`, `matmul` falls through to `a.dot(b)` (CPU is the
  source of truth) — verify no panic, correct result.

### 4b. `scirs2-datasets/src/generators/basic.rs::make_regression`

Routes the target product `y = X·coef` to `gpu_cuda::cuda_regression_target` when
`cuda_is_available()` and `n_samples * n_features >= CUDA_REGRESSION_MIN_ELEMS`,
where:

```rust
const CUDA_REGRESSION_MIN_ELEMS: usize = 1_000_000;
```

- [ ] Construct `n_samples * n_features >= 1_000_000` and confirm the CUDA branch
  engages.
- [ ] CPU vs CUDA targets agree within tolerance.
- [ ] **Fixed-seed reproducibility:** with a fixed `randomseed`, the output must
  be identical whether or not the GPU branch runs. The coefficient draws, the
  feature draws, and especially the **per-sample noise RNG draw (once per sample,
  ascending index order)** happen on the CPU around the GPU offload, so the GPU
  must only replace the noise-free `linear = X·coef` and must not perturb RNG
  order. Verify byte-stable targets across a CPU-only run and a GPU run.
- [ ] Sub-threshold sizes stay on CPU.
- [ ] On a forced CUDA `Err`, the CPU `linear` result is silently retained — no
  panic, correct result.

## 5. Performance benchmarking & threshold tuning

The dispatch thresholds in §4 (and any internal size cutoffs) were **guesses made
on a GPU-less box**. Re-tune them on real hardware. **Do not record any numbers
in this section as fact until measured — leave the blanks.**

- [ ] Benchmark each GPU entry point vs its CPU equivalent across a sweep of
  sizes, **including host↔device transfer cost** (upload + download dominate at
  small sizes).
- [ ] Find the true CPU/GPU crossover size per path: _________________
- [ ] Re-tune `CUDA_MATMUL_MIN_FLOPS` (linalg) to the measured crossover: ______
- [ ] Re-tune `CUDA_REGRESSION_MIN_ELEMS` (datasets) to the measured crossover: __
- [ ] **f64-throughput caveat:** consumer **GeForce** cards have heavily reduced
  `f64` throughput (often 1/32–1/64 of `f32`), so the GEMM / Cholesky paths may
  be *slower* than CPU on such cards regardless of size. **Data-center** cards
  (A100 / H100-class) do not have this penalty. Record the GPU model and class
  with every benchmark; a threshold tuned on a GeForce may be wrong on a Tesla
  and vice-versa.
- [ ] Decide and document whether any path should stay CPU-only on low-f64
  hardware.

## 6. Multi-GPU / device selection

- [ ] Every `gpu_cuda.rs` `build_context()` / `build_handle()` hard-selects
  **device 0** (`Device::get(0)`). Untested with more than one GPU.
- [ ] Validate behavior on a multi-GPU box (does device-0 selection do the right
  thing? any cross-device pointer hazard?).
- [ ] Decide whether to expose device selection (env var / argument / context)
  and, if so, thread it through all 10 crates consistently. (Currently none do.)

## 7. Error paths & robustness

- [ ] **Device OOM:** force `DeviceBuffer::alloc` / `DeviceBuffer::from_host` to
  fail (allocate beyond VRAM) and confirm each entry point returns its crate
  error (`LinalgError::ComputationError`, `VisionError::GpuError`,
  `CudaEvalError::Launch`, …) and **never panics**.
- [ ] **Vision workspace retry (again):** confirm `DnnError::WorkspaceRequired`
  is handled, not propagated as a hard error (§3c).
- [ ] **Driver present but no device / init failure:** `cuda_is_available()` must
  return `false` and every entry point must return an error (not panic). This is
  the path that has "worked" on macOS; confirm it also holds on a broken / zero
  NVIDIA setup.
- [ ] **No reachable `.unwrap()` / `.expect()` on a CUDA error path** (no-unwrap
  policy). Audit each `gpu_cuda.rs`: every device call goes through
  `.map_err(...)?`; symbolic's emit-walk uses total `.unwrap_or_else(||
  zero_reg.clone())` and graph uses `.unwrap_or(&0)` (neither panics). Confirm
  nothing panics on a real device error.

## 8. oxicuda API-drift check

The call sites were written against a specific `../oxicuda`. Before trusting any
result, confirm the signatures each `gpu_cuda.rs` calls still exist with the same
shape (and update the call sites if oxicuda has moved):

- [ ] `oxicuda-blas`: `level3::gemm_api::gemm::<f64>(handle, Transpose, Transpose,
  alpha, &MatrixDesc, &MatrixDesc, beta, &mut MatrixDescMut)`;
  `BlasHandle::new(&ctx)`; `types::{FillMode, Layout, MatrixDesc, MatrixDescMut,
  Transpose}`; `MatrixDesc::from_buffer(..)`.
- [ ] `oxicuda-solver`: `dense::cholesky::<f64>(&mut handle, FillMode, &mut buf,
  n, n)`; `dense::cholesky_solve::<f64>(&handle, FillMode, &buf, &mut rhs, n,
  nrhs)`; `SolverHandle::new(&ctx)`.
- [ ] `oxicuda-dnn`: `conv::conv_forward::<f64>(&handle, &TensorDesc, &TensorDesc,
  &mut TensorDescMut, &ConvolutionDescriptor, Option<&mut DeviceBuffer<u8>>)`;
  `DnnHandle::new(&ctx)`; `ConvolutionDescriptor::conv2d(..)`;
  `TensorDesc::nchw(..)`; the `DnnError::WorkspaceRequired(bytes)` variant still
  exists.
- [ ] `oxicuda-fft`: `FftPlan::new_1d(n, FftType::C2C, batch)`,
  `.with_precision(FftPrecision::Double)`; `FftHandle::new(&ctx)`;
  `handle.execute(&plan, in_ptr, out_ptr, FftDirection::{Forward, Inverse})`;
  `Complex<f64>`.
- [ ] `oxicuda-sparse`: `ops::spmv::<f64>(&handle, SpMVAlgo::Adaptive, alpha,
  &CsrMatrix, x_ptr, beta, y_ptr)`; `CsrMatrix::<f64>::from_host(n_rows, n_cols,
  row_offsets, col_indices, values)`; `SparseHandle::new(&ctx)`.
- [ ] `oxicuda-ptx` / `oxicuda-launch`: the `KernelBuilder` / `BodyBuilder` DSL
  (`erf_f64`, `exp_f64`, `fma_f64`, `add_f64`, `sub_f64`, `mov_imm_f64`,
  `load_global_f64`, `store_global_f64`, `global_thread_id_x`, …);
  `Module::from_ptx`, `Kernel::from_module`, `LaunchParams::new`,
  `grid_size_for`. **Note the DSL gaps the code works around:** no public
  `mul_f64` (uses `fma(a, b, 0)`); no `f64` immediates in symbolic (it hoists
  consts to a device buffer). If oxicuda has since added `mul.f64` / `f64`
  immediates / `div` / transcendentals, the symbolic and interpolate "PENDING"
  notes can be revisited.
- [ ] `oxicuda-driver` / `oxicuda-memory`: `init()`, `device::Device::{count,
  get}`, `Context::new`, `stream::Stream::{new, synchronize}`,
  `DeviceBuffer::{alloc, from_host, copy_to_host, as_device_ptr}`.

## 9. CI / recording

- [ ] **There is no CUDA CI and there will not be one.** Per COOLJAPAN policy only
  `pypi-publish.yml` / `npm-publish.yml` GitHub workflows are allowed, and
  GitHub's hosted runners have no NVIDIA GPU anyway. All validation here is
  **manual on an NVIDIA box**.
- [ ] Record every result **back into this file**: fill the header table (GPU
  model, driver, CUDA version, oxicuda commit, date) and tick the boxes. Note any
  bug found + its fix commit next to the relevant item.

## 10. Sign-off matrix

Filled on real hardware **2026-06-29, NVIDIA RTX A4000 (sm_86, CUDA 12.4)**
(`Build` = §1, `Smoke` = §2, `Correctness` = §3, `Bench` = §5). `Build`/`Smoke`/
`Correctness` are all on-device PASS (every `*_or_skip` ran the device path — no
skip line — and matched the CPU source-of-truth within the documented tolerance:
1e-9 for exact arithmetic, ~1e-6 for the A&S transcendental paths). `Bench` (§5)
is deliberately **deferred**: the A4000 is GA104 with ~1/64 f64 throughput, so a
crossover/threshold tune would be card-specific and is not a correctness gate —
see §5.

| Crate | Build | Smoke | Correctness | Bench |
|---|---|---|---|---|
| scirs2-linalg | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-interpolate | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-optimize | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-datasets | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-special | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-stats | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-symbolic | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-fft | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-graph | [x] | [x] | [x] | [ ] (deferred, §5) |
| scirs2-vision | [x] | [x] | [x] | [ ] (deferred, §5) |

### §3/§4/§7 coverage actually added to the scirs `*_or_skip` suites
- §3a non-square GEMM stress (`linalg` 17×5·5×31), HVP non-symmetric (`optimize`).
- §3b erf tails→±1 (`special`), normal PDF(1e-9)/CDF(1e-6) tails (`stats`), deep
  Add/Sub/Mul nesting + `Unsupported`-op honest-`Err` rejection (`symbolic`).
- §3c vector/warp-shuffle SpMV on a directed/asymmetric graph (`graph`, avg 4.6
  nnz/row), non-power-of-two FFT + normalization (`fft`), non-symmetric kernel on
  a non-square image + hard-zero border (`vision`).
- §4a `matmul` 130³ f64 dispatch (GPU branch engaged; f32 & sub-threshold stay
  CPU), §4b `make_regression` ≥1M-elem dispatch with fixed-seed **byte-stability**
  proven, §7 forced-`Err`→CPU fallback never panics.

### Remaining (recorded honestly; NOT exercised by the 10 paths)
- **§5 bench/threshold tune** — deferred (GeForce-class f64 throughput caveat).
- **§6 multi-GPU** — N/A (single A4000); device-0 hard-select unchanged.
- **oxicuda-blas level-3 GEMM/TRSM ignore leading dimensions** on strided
  sub-blocks. Both LU and **blocked Cholesky (n>64) now sidestep this with their own
  ld-honoring strided device kernels** (Cholesky verified to n=256), so the dense
  factorizations scirs uses are correct at any n. But the oxicuda-blas level-3
  routines themselves should still be fixed (deeper follow-up; the contiguous
  full-matrix ld==rows path is fine).
- Real on-device QR/SVD/eig (currently correct, loudly-marked CPU fallbacks) and a
  tiled/Tensor-Core GEMM/conv fast path — large deferred performance work.

## 11. Out of scope (separate efforts)

- **Non-oxicuda `cuda` features** (do NOT fold these into the matrix above):
    - `scirs2-cluster` — `cuda = ["gpu"]` (re-exports the wgpu `gpu` feature; not oxicuda).
    - `scirs2-ndimage` — `cuda = []` with its own `src/backend/cuda.rs` (separate backend).
    - `scirs2-spatial` — `cuda = []`, capability-detection only.
    - `scirs2-integration-tests` — `cuda = ["scirs2-linalg/cuda"]` (a re-export
      used to build §1's umbrella check; the actual code under test is
      scirs2-linalg's).
- **wgpu / Metal portability layer** — validated on macOS, f32, portable; an
  entirely separate GPU effort from this NVIDIA-only f64 oxicuda surface.
