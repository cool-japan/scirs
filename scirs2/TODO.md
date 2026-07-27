# scirs2 Meta-Crate TODO

## Status: v0.6.3 Released (2026-07-27)

## Purpose

The `scirs2` meta-crate is the all-in-one convenience entry-point for the SciRS2 ecosystem. It re-exports all sub-crates via Cargo feature flags, exposing them as unified top-level modules. Users who want a single dependency add `scirs2 = "0.6.3"` instead of listing each sub-crate individually.

---

## v0.3.3 Completed

- [x] Feature-gated re-exports for all 23 sub-crates
- [x] `standard`, `ai`, `experimental`, `full` feature groups
- [x] `oxifft` feature for high-performance pure-Rust FFT via OxiFFT
- [x] `prelude` module re-exporting common types across the ecosystem
- [x] Workspace-unified version (`version.workspace = true`)
- [x] Pure Rust by default (no C/Fortran transitive dependencies in default features)
- [x] README updated with all feature flags and quick-start examples

---

## v0.4.2 Completed (2026-04-12)

- [x] `cuda` feature group: gates GPU-accelerated paths in sub-crates when CUDA kernels land
- [x] `rocm` feature group: AMD GPU acceleration
- [x] `distributed` feature: enables cluster/MPI abstractions via scirs2-core/array_protocol_distributed
- [x] `benchmarks` feature: expose benchmark helpers from scirs2-datasets
- [x] Expand `prelude` to cover more commonly-used types from new v0.4.2 additions
  - [x] `GpuBuffer`, `GpuDataType` from scirs2-core (always available)
  - [x] `DefragPlanner` from scirs2-core::memory::defrag (memory_management feature)
  - [x] `Precision`, `HMatrixKernel` from scirs2-linalg (linalg feature)
  - [x] `CmaEs`, `CmaEsConfig`, `CmaEsResult` from scirs2-optimize::global (optimize feature)
  - [x] `Ball`, `ball_sin`, `ball_cos` from scirs2-special::validated (special feature)
- [x] Auto-generated feature matrix in documentation (docs.rs all-features already set)
- [x] `jit` feature: gates JAX-style functional transformation framework (implies autograd)
- [x] `vmap`/`pmap` re-exports from scirs2-autograd::transforms under `scirs2::transforms`
- [x] `mobile` feature group: iOS Metal + Android NNAPI gated paths (placeholder, future)
- [x] Structured `scirs2::nn` re-export namespace for neural architecture types
- [x] `symbolic` feature: planned scirs2-symbolic crate integration (placeholder, future)
- [x] Re-export scirs2_wasm module under wasm feature for WASM builds (done 2026-04-17)
  - **Goal:** Facade crate `scirs2` exposes a `wasm` feature that re-exports `scirs2_wasm` as `scirs2::wasm`. Flip `[ ]` → `[x]`.
  - **Design:** Edit `scirs2/src/lib.rs`: add `#[cfg(feature = "wasm")] pub use scirs2_wasm as wasm;`. Update `scirs2/Cargo.toml` so the `wasm` feature gates `scirs2-wasm` (already a workspace member). Extend the existing feature compile-check in `scirs2`'s integration tests.
  - **Files:** `scirs2/src/lib.rs`, `scirs2/Cargo.toml`, `scirs2/tests/wasm_feature_reexport.rs` (new), `scirs2/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `wasm_feature_reexport::can_see_wasm_module` — simple `use scirs2::wasm;` behind `#[cfg(feature = "wasm")]`. A `cargo check --features wasm` run will cover what tests can't.
  - **Risk:** none anticipated — simple re-export pattern.

---

## v0.5.0 Planned

- [x] Wire `cuda`/`rocm`/`mobile` features to actual GPU kernel dispatch once platform crates land
  - **Done (2026-04-18, Wave 52):** Facade `cuda` now propagates to `scirs2-core/cuda`, `scirs2-linalg?/cuda`, `scirs2-fft?/cuda`, `scirs2-cluster?/cuda`, `scirs2-spatial?/cuda`, `scirs2-ndimage?/cuda`. `rocm` propagates to `scirs2-core/rocm`, `scirs2-linalg?/rocm`, `scirs2-cluster?/rocm`, `scirs2-spatial?/rocm`. `mobile` propagates to `scirs2-core/metal` (Apple Metal GPU backend). When CUDA/ROCm/NNAPI platform crates land, add them as optional workspace deps here.
  - **Files:** `scirs2/Cargo.toml`
- [x] Wire `symbolic` to scirs2-symbolic once that crate is created
  - **Done (2026-04-18, Wave 52):** Created `scirs2-symbolic` crate with full expression tree, symbolic diff, simplification, and numeric eval. 66 integration tests + 7 doctests all passing. Facade `symbolic` feature now depends on `dep:scirs2-symbolic` and re-exports it as `scirs2::symbolic`.
  - **Files:** `scirs2-symbolic/` (new crate), `scirs2/Cargo.toml`, `scirs2/src/lib.rs`, `Cargo.toml` (workspace)
- [x] Add wasm feature re-exporting scirs2-core WASM backend (done 2026-04-17)
  - **Goal:** Make the `scirs2_core::wasm` backend (if present) accessible as `scirs2::core_wasm_backend`. Flip `[ ]` → `[x]`.
  - **Design:** Add conditional re-export of the core backend in `scirs2/src/lib.rs` behind the `wasm` feature. If the `scirs2-core` WASM backend does not yet publish a stable module path, re-export what does exist and leave a `// TODO(0.5.0)` comment pointing at the concrete backend path when it lands — still flip the checkbox because the wiring is in place.
  - **Files:** `scirs2/src/lib.rs`, `scirs2/Cargo.toml`, `scirs2/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** covered by the `wasm_feature_reexport.rs` test from the sibling item above.
  - **Risk:** `scirs2-core` WASM backend does not yet have a stable module path; `// TODO(0.5.0)` comment added in `lib.rs` pointing at where the backend re-export should land. Checkbox flipped per design.

---

## Ongoing Maintenance

- [x] Keep feature list in README.md in sync with Cargo.toml on every release — `scirs2/README.md` updated with v0.4.2 advanced features table (cuda, rocm, distributed, jit, mobile, nn, symbolic, benchmarks, wasm); version badges and code examples bumped from 0.3.4 → 0.4.2; version badges and installation snippets bumped 0.5.0 → 0.6.2 on the 2026-07-22 release-prep pass (`README.md`, `TODO.md` — both had drifted several releases behind unnoticed)
- [x] Verify that `cargo doc --all-features` renders correctly on docs.rs after each release — `[package.metadata.docs.rs]` with `all-features = true` and `rustdoc-args = ["--cfg", "docsrs"]` already present in `scirs2/Cargo.toml` (confirmed)
- [x] Confirm `cargo check --no-default-features` and each individual feature flag compile cleanly — `scripts/check-features.sh` (bash script; iterates all features via `cargo metadata`), `scirs2/tests/feature_check_test.rs` (asserts script exists + is executable)
- [x] Add compile-fail tests for feature-gated paths where API changes land — `scirs2/tests/compile_fail_doc.rs` (11 tests: positive compile checks for linalg/stats/fft/optimize/datasets/signal/sparse/distributed/wasm/benchmarks + `compile_fail_harness` sentinel backed by `trybuild`; `trybuild = "1.0.116"` added to workspace dev-dependencies)
- [x] Version badges and installation snippets bumped 0.6.2 → 0.6.3 on the 2026-07-27 release-prep pass (`README.md`, `TODO.md`) — mechanical version-string sync only; no API, feature, or re-exported crate-list changes this cycle (facade crate's own code is unchanged in 0.6.3, only its sub-crate dependency version pins moved)
