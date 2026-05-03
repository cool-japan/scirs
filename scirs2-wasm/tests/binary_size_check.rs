//! WASM binary size regression tracking.
//!
//! This test documents the target WASM binary size budget.
//! Actual enforcement happens in CI via `scripts/check-wasm-size.sh`.
//!
//! To measure the current binary size locally:
//! ```bash
//! cargo build --target wasm32-unknown-unknown --release -p scirs2-wasm
//! # or, for the wasm-pack output:
//! wasm-pack build --release -p scirs2-wasm
//! ls -lh scirs2-wasm/pkg/scirs2_wasm_bg.wasm
//! ```

/// Target maximum WASM binary size in bytes (2 MiB for core functionality).
///
/// This constant is intentionally public so downstream consumers can reference
/// the documented budget without running the build.
pub const WASM_BINARY_SIZE_LIMIT_BYTES: usize = 2 * 1024 * 1024; // 2 MiB

/// WASM binary size budget in a human-readable label.
pub const WASM_BINARY_SIZE_LIMIT_LABEL: &str = "2 MiB";

#[test]
fn wasm_binary_size_budget_is_documented() {
    // Verify the documented constants are self-consistent.
    assert_eq!(WASM_BINARY_SIZE_LIMIT_BYTES, 2 * 1024 * 1024);
    assert!(!WASM_BINARY_SIZE_LIMIT_LABEL.is_empty());

    // The actual binary size check runs during CI (see scripts/check-wasm-size.sh).
    // This test ensures the budget constants are present and compile correctly.
    let budget_mb = WASM_BINARY_SIZE_LIMIT_BYTES / (1024 * 1024);
    assert_eq!(budget_mb, 2);
}

#[test]
fn wasm_pkg_version_is_present() {
    // Verify that the package metadata embeds a version string.
    // When compiled to WASM this same string appears in the .js glue code.
    let version = env!("CARGO_PKG_VERSION");
    assert!(!version.is_empty(), "CARGO_PKG_VERSION must not be empty");
    assert!(
        version.contains('.'),
        "CARGO_PKG_VERSION should be a semver string (contains '.')"
    );
}
