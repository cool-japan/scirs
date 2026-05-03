//! Tests for the scirs2-wasm React hooks package.
//!
//! Verifies that the JS helper files exist and contain the expected exports.
//! Full React integration testing requires a JavaScript runtime and is handled
//! in the JS test suite.

use std::fs;
use std::path::Path;

/// Absolute path to the `js/react-hooks/` directory.
fn react_hooks_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("js/react-hooks")
}

#[test]
fn react_hooks_js_exists() {
    let path = react_hooks_dir().join("useScirs2.js");
    assert!(
        path.exists(),
        "js/react-hooks/useScirs2.js should exist at {}",
        path.display()
    );
}

#[test]
fn react_hooks_package_json_exists() {
    let path = react_hooks_dir().join("package.json");
    assert!(
        path.exists(),
        "js/react-hooks/package.json should exist at {}",
        path.display()
    );
}

#[test]
fn react_hooks_exports_use_scirs2() {
    let path = react_hooks_dir().join("useScirs2.js");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    assert!(
        content.contains("useScirs2"),
        "useScirs2.js should export the useScirs2 hook"
    );
    assert!(
        content.contains("useScirs2Compute"),
        "useScirs2.js should export the useScirs2Compute hook"
    );
    assert!(
        content.contains("useScirs2Array"),
        "useScirs2.js should export the useScirs2Array hook"
    );
}

#[test]
fn react_hooks_package_json_valid() {
    let path = react_hooks_dir().join("package.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    // Must name the package and declare react peer dep.
    assert!(
        content.contains("@scirs2/react-hooks"),
        "package.json should set the package name"
    );
    assert!(
        content.contains("react"),
        "package.json should list react as a peer dependency"
    );
}
