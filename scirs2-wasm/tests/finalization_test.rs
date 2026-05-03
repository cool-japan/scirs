//! Tests for the FinalizationRegistry JavaScript helper file.
//!
//! These tests verify that the JS artefact exists and contains the expected
//! identifiers.  Actual FinalizationRegistry behaviour can only be validated
//! in a JavaScript runtime (browser or Node.js); these Rust tests cover the
//! structural aspects only.

use std::fs;
use std::path::Path;

/// Absolute path to the `js/finalization.js` file.
fn finalization_js_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("js/finalization.js")
}

#[test]
fn finalization_js_file_exists() {
    let path = finalization_js_path();
    assert!(
        path.exists(),
        "js/finalization.js should exist at {}",
        path.display()
    );
}

#[test]
fn finalization_contains_registry_class() {
    let path = finalization_js_path();
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    assert!(
        content.contains("FinalizationRegistry"),
        "finalization.js should reference FinalizationRegistry"
    );
    assert!(
        content.contains("ManagedWasm"),
        "finalization.js should export the ManagedWasm class"
    );
}

#[test]
fn finalization_exports_utilities() {
    let path = finalization_js_path();
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    assert!(
        content.contains("isFinalizationRegistrySupported"),
        "finalization.js should export isFinalizationRegistrySupported"
    );
    assert!(
        content.contains("createFinalizationPolyfill"),
        "finalization.js should export createFinalizationPolyfill"
    );
}
