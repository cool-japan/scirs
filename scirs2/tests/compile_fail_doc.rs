//! Compile-fail documentation tests for feature-gated API paths.
//!
//! These tests verify that:
//! 1. Feature-gated modules expose the correct types when enabled, by
//!    confirming that key type-level and value-level symbols compile.
//! 2. The compile-fail harness (trybuild) infrastructure is in place for
//!    future negative compilation tests.
//!
//! ## Adding compile-fail tests
//!
//! Place failing `.rs` files in `tests/compile_fail/` alongside their expected
//! `.stderr` output.  Run with:
//!
//! ```sh
//! cargo test -p scirs2 --test compile_fail_doc -- compile_fail_harness
//! ```
//!
//! The `TRYBUILD=overwrite` env var regenerates expected `.stderr` snapshots.

// ── Positive compile checks (enabled features) ───────────────────────────────

/// When the `linalg` feature is active, `scirs2::linalg` module resolves and
/// key error types are accessible.
#[test]
#[cfg(feature = "linalg")]
fn feature_linalg_module_accessible() {
    // `LinalgError` is always exported; its in-memory size must be non-zero.
    let sz = std::mem::size_of::<scirs2::linalg::LinalgError>();
    assert!(sz > 0, "LinalgError size must be > 0");
}

/// When the `stats` feature is active, `scirs2::stats` resolves.
#[test]
#[cfg(feature = "stats")]
fn feature_stats_module_accessible() {
    let sz = std::mem::size_of::<scirs2::stats::StatsError>();
    assert!(sz > 0, "StatsError size must be > 0");
}

/// When the `fft` feature is active, `scirs2::fft` resolves.
#[test]
#[cfg(feature = "fft")]
fn feature_fft_module_accessible() {
    // FftError is exported from scirs2-fft; its presence validates the path.
    let sz = std::mem::size_of::<scirs2::fft::FFTError>();
    assert!(sz > 0, "FFTError size must be > 0");
}

/// When the `optimize` feature is active, `scirs2::optimize` resolves.
#[test]
#[cfg(feature = "optimize")]
fn feature_optimize_module_accessible() {
    let sz = std::mem::size_of::<scirs2::optimize::OptimizeError>();
    assert!(sz > 0, "OptimizeError size must be > 0");
}

/// When the `datasets` feature is active, `scirs2::datasets::load_iris` resolves.
#[test]
#[cfg(feature = "datasets")]
fn feature_datasets_module_accessible() {
    // `load_iris` must be a callable function — just verify it resolves.
    let f = scirs2::datasets::load_iris;
    let result = f();
    assert!(result.is_ok(), "load_iris() must succeed");
}

/// When the `signal` feature is active, `scirs2::signal` resolves.
#[test]
#[cfg(feature = "signal")]
fn feature_signal_module_accessible() {
    let sz = std::mem::size_of::<scirs2::signal::SignalError>();
    assert!(sz > 0, "SignalError size must be > 0");
}

/// When the `sparse` feature is active, `scirs2::sparse` resolves.
#[test]
#[cfg(feature = "sparse")]
fn feature_sparse_module_accessible() {
    let sz = std::mem::size_of::<scirs2::sparse::SparseError>();
    assert!(sz > 0, "SparseError size must be > 0");
}

/// When the `distributed` feature is active, `scirs2_core` distributed
/// primitive functions must be accessible and functional.
#[test]
#[cfg(feature = "distributed")]
fn feature_distributed_resolves() {
    use scirs2_core::distributed::par_iter::par_map;
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let doubled = par_map(&data, |x| x * 2, None);
    assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
}

/// When the `wasm` feature is active, `scirs2::wasm` resolves.
#[test]
#[cfg(feature = "wasm")]
fn feature_wasm_module_accessible() {
    // WasmError is in the `error` submodule of scirs2-wasm.
    let sz = std::mem::size_of::<scirs2::wasm::error::WasmError>();
    assert!(sz > 0, "WasmError size must be > 0");
}

/// When the `benchmarks` feature is active (implies `datasets`),
/// `scirs2::datasets` resolves.
#[test]
#[cfg(feature = "benchmarks")]
fn feature_benchmarks_implies_datasets() {
    // `benchmarks` enables `datasets`; load_iris must be accessible.
    let result = scirs2::datasets::load_iris();
    assert!(
        result.is_ok(),
        "load_iris() must succeed under benchmarks feature"
    );
}

// ── Compile-fail harness ──────────────────────────────────────────────────────

/// The compile-fail harness test: runs trybuild on files in `tests/compile_fail/`
/// when that directory exists and contains `.rs` files.
///
/// This test is a no-op when no compile-fail cases are present — it just
/// confirms the infrastructure is wired so future cases can be added by
/// creating `.rs` + `.stderr` file pairs in `tests/compile_fail/`.
#[test]
fn compile_fail_harness() {
    let compile_fail_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("compile_fail");

    if !compile_fail_dir.exists() {
        // No compile-fail cases yet; infrastructure is present and ready.
        return;
    }

    let rs_files: Vec<_> = std::fs::read_dir(&compile_fail_dir)
        .map(|rd| {
            rd.filter_map(|entry| {
                let e = entry.ok()?;
                let path = e.path();
                if path.extension().is_some_and(|ext| ext == "rs") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect()
        })
        .unwrap_or_default();

    if rs_files.is_empty() {
        // Directory exists but no .rs files yet; still a no-op.
        return;
    }

    // When compile-fail cases exist, run them through trybuild.
    let t = trybuild::TestCases::new();
    t.compile_fail(compile_fail_dir.join("*.rs"));
}
