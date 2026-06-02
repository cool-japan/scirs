//! Integration tests for `cargo-scirs2-policy` linter.
//!
//! Each test exercises one logical scenario using temporary directories so that
//! tests remain self-contained and do not depend on the real workspace layout.

use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a unique temporary directory for a test.
fn temp_dir(suffix: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "policy_integ_{}_{}_{}",
        std::process::id(),
        suffix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
    ));
    fs::create_dir_all(&base).expect("create temp dir");
    base
}

/// Write `content` to `<dir>/src/<filename>`, creating `src/` if needed.
fn write_src(dir: &Path, filename: &str, content: &str) {
    let src = dir.join("src");
    fs::create_dir_all(&src).expect("create src dir");
    fs::write(src.join(filename), content).expect("write file");
}

/// Write a minimal `Cargo.toml` in `dir`.
fn write_cargo_toml(dir: &Path, content: &str) {
    fs::write(dir.join("Cargo.toml"), content).expect("write Cargo.toml");
}

/// Build a [`WorkspaceInfo`] with a single crate at `dir`.
fn single_crate_workspace(
    dir: &Path,
    name: &str,
    is_core: bool,
) -> cargo_scirs2_policy::workspace::WorkspaceInfo {
    cargo_scirs2_policy::workspace::WorkspaceInfo {
        root: dir.parent().unwrap_or(dir).to_path_buf(),
        crates: vec![cargo_scirs2_policy::workspace::CrateInfo {
            name: name.to_string(),
            path: dir.to_path_buf(),
            is_core,
        }],
    }
}

// ---------------------------------------------------------------------------
// Re-export the internal modules we need in integration tests
// ---------------------------------------------------------------------------

use cargo_scirs2_policy::checks::banned_deps::BannedDepCheck;
use cargo_scirs2_policy::checks::banned_imports::BannedImportCheck;
use cargo_scirs2_policy::checks::unwrap_check::UnwrapCheck;
use cargo_scirs2_policy::report::{json_report, print_report};
use cargo_scirs2_policy::violation::{exit_code, format_violation, PolicyViolation, Severity};
use cargo_scirs2_policy::workspace::{discover_workspace, walk_rust_files};

// ===========================================================================
// 1. Detect banned import `use rand::*`
// ===========================================================================

#[test]
fn test_detect_banned_import_rand() {
    let dir = temp_dir("rand_import");
    write_src(&dir, "lib.rs", "use rand::Rng;\nfn main() {}\n");
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = BannedImportCheck.run(&ws);
    assert!(
        !violations.is_empty(),
        "Should detect 'use rand::' in non-core crate"
    );
    assert!(
        violations.iter().any(|v| v.message.contains("rand")),
        "Violation message should mention 'rand'"
    );
    assert_eq!(violations[0].line, 1, "Violation should be on line 1");

    let _ = fs::remove_dir_all(&dir);
}

// ===========================================================================
// 2. Detect banned dep `flate2` in Cargo.toml
// ===========================================================================

#[test]
fn test_detect_banned_dep_flate2() {
    let dir = temp_dir("flate2_dep");
    write_cargo_toml(
        &dir,
        "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n[dependencies]\nflate2 = \"1.0\"\n",
    );
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = BannedDepCheck.run(&ws);
    assert!(
        violations.iter().any(|v| v.message.contains("flate2")),
        "Should detect banned dep 'flate2'"
    );
    assert_eq!(
        violations
            .iter()
            .find(|v| v.message.contains("flate2"))
            .unwrap()
            .severity,
        Severity::Error,
        "Non-optional flate2 should be Error severity"
    );

    let _ = fs::remove_dir_all(&dir);
}

// ===========================================================================
// 3. Optional dep should be Warning, not Error
// ===========================================================================

#[test]
fn test_allowed_optional_dep_is_warning_not_error() {
    let dir = temp_dir("optional_dep");
    write_cargo_toml(
        &dir,
        "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n\
         [dependencies]\nflate2 = { version = \"1.0\", optional = true }\n",
    );
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = BannedDepCheck.run(&ws);
    let v = violations
        .iter()
        .find(|v| v.message.contains("flate2"))
        .expect("flate2 violation");
    assert_eq!(
        v.severity,
        Severity::Warning,
        "Optional flate2 should be Warning (not Error) to allow feature-gated usage review"
    );

    let _ = fs::remove_dir_all(&dir);
}

// ===========================================================================
// 4. Workspace discovery finds crates
// ===========================================================================

#[test]
fn test_workspace_discovery() {
    let root = temp_dir("ws_disc");

    // Write workspace root
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crate-a\", \"crate-b\"]\n",
    )
    .expect("write workspace Cargo.toml");

    // Crate A
    let crate_a = root.join("crate-a");
    fs::create_dir_all(crate_a.join("src")).expect("crate-a src");
    fs::write(
        crate_a.join("Cargo.toml"),
        "[package]\nname = \"crate-a\"\nversion = \"0.1.0\"\n",
    )
    .expect("crate-a Cargo.toml");
    fs::write(crate_a.join("src").join("lib.rs"), "").expect("lib.rs");

    // Crate B
    let crate_b = root.join("crate-b");
    fs::create_dir_all(crate_b.join("src")).expect("crate-b src");
    fs::write(
        crate_b.join("Cargo.toml"),
        "[package]\nname = \"crate-b\"\nversion = \"0.1.0\"\n",
    )
    .expect("crate-b Cargo.toml");
    fs::write(crate_b.join("src").join("lib.rs"), "").expect("lib.rs");

    let ws = discover_workspace(&root);
    assert_eq!(ws.crates.len(), 2, "Should discover 2 crates");
    let names: Vec<&str> = ws.crates.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"crate-a"), "Should find crate-a");
    assert!(names.contains(&"crate-b"), "Should find crate-b");

    let _ = fs::remove_dir_all(&root);
}

// ===========================================================================
// 5. Unwrap in production code → violation
// ===========================================================================

#[test]
fn test_unwrap_in_prod() {
    let dir = temp_dir("unwrap_prod");
    write_src(
        &dir,
        "lib.rs",
        "fn foo() {\n    let x = bar().unwrap();\n}\n",
    );
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = UnwrapCheck.run(&ws);
    assert!(
        !violations.is_empty(),
        "Should detect .unwrap() in production code"
    );
    assert_eq!(violations[0].line, 2, "Violation should be on line 2");

    let _ = fs::remove_dir_all(&dir);
}

// ===========================================================================
// 6. Unwrap in #[cfg(test)] block → no violation
// ===========================================================================

#[test]
fn test_unwrap_in_test_block_no_violation() {
    let dir = temp_dir("unwrap_cfg");
    write_src(
        &dir,
        "lib.rs",
        "fn prod() -> i32 { 42 }\n\
         \n\
         #[cfg(test)]\n\
         mod tests {\n\
             #[test]\n\
             fn test_it() {\n\
                 let r: Result<i32, &str> = Ok(1);\n\
                 assert_eq!(r.unwrap(), 1);\n\
             }\n\
         }\n",
    );
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = UnwrapCheck.run(&ws);
    assert!(
        violations.is_empty(),
        ".unwrap() inside #[cfg(test)] should not be flagged; got: {:?}",
        violations
    );

    let _ = fs::remove_dir_all(&dir);
}

// ===========================================================================
// 7. Violation format string is correct
// ===========================================================================

#[test]
fn test_violation_format() {
    let v = PolicyViolation {
        crate_name: "my-crate".to_string(),
        file: PathBuf::from("/src/lib.rs"),
        line: 42,
        message: "some violation".to_string(),
        severity: Severity::Error,
    };
    let formatted = format_violation(&v);
    assert!(formatted.contains("[ERROR]"), "Should contain severity");
    assert!(
        formatted.contains("[my-crate]"),
        "Should contain crate name"
    );
    assert!(formatted.contains(":42"), "Should contain line number");
    assert!(
        formatted.contains("some violation"),
        "Should contain message"
    );
}

// ===========================================================================
// 8. JSON report is valid JSON
// ===========================================================================

#[test]
fn test_json_report_valid() {
    let violations = vec![
        PolicyViolation {
            crate_name: "crate-a".to_string(),
            file: PathBuf::from("/path/lib.rs"),
            line: 5,
            message: "test violation".to_string(),
            severity: Severity::Error,
        },
        PolicyViolation {
            crate_name: "crate-b".to_string(),
            file: PathBuf::from("/path/other.rs"),
            line: 10,
            message: "another violation".to_string(),
            severity: Severity::Warning,
        },
    ];

    let json = json_report(&violations);
    // Basic structural checks — not a full JSON parser
    assert!(
        json.trim_start().starts_with('['),
        "JSON should start with ["
    );
    assert!(json.trim_end().ends_with(']'), "JSON should end with ]");
    assert!(
        json.contains("\"crate_name\":\"crate-a\""),
        "Should contain crate-a"
    );
    assert!(
        json.contains("\"crate_name\":\"crate-b\""),
        "Should contain crate-b"
    );
    assert!(json.contains("\"line\":5"), "Should contain line 5");
    assert!(
        json.contains("\"severity\":\"ERROR\""),
        "Should contain ERROR severity"
    );
}

// ===========================================================================
// 9. exit_code returns 1 when there are Error violations
// ===========================================================================

#[test]
fn test_exit_code_errors() {
    let violations = vec![PolicyViolation {
        crate_name: "c".to_string(),
        file: PathBuf::from("/f.rs"),
        line: 1,
        message: "error".to_string(),
        severity: Severity::Error,
    }];
    assert_eq!(
        exit_code(&violations),
        1,
        "Should return 1 for Error violations"
    );
}

// ===========================================================================
// 10. exit_code returns 0 when there are no Error violations
// ===========================================================================

#[test]
fn test_no_violations_exit_0() {
    // Empty violations
    assert_eq!(exit_code(&[]), 0, "Empty violations → exit 0");

    // Only warnings — still exit 0
    let warnings = vec![PolicyViolation {
        crate_name: "c".to_string(),
        file: PathBuf::from("/f.rs"),
        line: 1,
        message: "warn".to_string(),
        severity: Severity::Warning,
    }];
    assert_eq!(exit_code(&warnings), 0, "Warnings only → exit 0");
}

// ===========================================================================
// Additional coverage
// ===========================================================================

#[test]
fn test_walk_rust_files_finds_nested_files() {
    let dir = temp_dir("walk_nested");
    let src = dir.join("src");
    let sub = src.join("sub");
    fs::create_dir_all(&sub).expect("sub dir");
    fs::write(src.join("lib.rs"), "").expect("lib.rs");
    fs::write(sub.join("helper.rs"), "").expect("helper.rs");

    let files = walk_rust_files(&dir);
    assert_eq!(files.len(), 2, "Should find both lib.rs and helper.rs");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_json_report_empty_array() {
    let json = json_report(&[]);
    assert_eq!(
        json.trim(),
        "[]",
        "Empty violations should produce empty JSON array"
    );
}

#[test]
fn test_print_report_empty_does_not_panic() {
    // Just ensure it does not panic
    print_report(&[]);
}

#[test]
fn test_print_report_with_violations_does_not_panic() {
    let violations = vec![PolicyViolation {
        crate_name: "crate-x".to_string(),
        file: PathBuf::from("/x/src/lib.rs"),
        line: 7,
        message: "some issue".to_string(),
        severity: Severity::Warning,
    }];
    print_report(&violations);
}

#[test]
fn test_banned_dep_bincode_workspace_form() {
    let dir = temp_dir("bincode_ws");
    write_cargo_toml(
        &dir,
        "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n\
         [dependencies]\nbincode.workspace = true\n",
    );
    let ws = single_crate_workspace(&dir, "my-crate", false);

    let violations = BannedDepCheck.run(&ws);
    assert!(
        violations.iter().any(|v| v.message.contains("bincode")),
        "Should detect bincode.workspace = true form"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_globally_banned_import_zip_in_any_crate() {
    let dir = temp_dir("zip_import");
    write_src(&dir, "lib.rs", "use zip::ZipArchive;\n");
    let ws = single_crate_workspace(&dir, "any-crate", false);

    let violations = BannedImportCheck.run(&ws);
    assert!(
        violations.iter().any(|v| v.message.contains("zip")),
        "Should detect 'use zip::' globally"
    );
    assert_eq!(
        violations
            .iter()
            .find(|v| v.message.contains("zip"))
            .unwrap()
            .severity,
        Severity::Error,
        "'use zip::' is globally banned — Error severity"
    );

    let _ = fs::remove_dir_all(&dir);
}
