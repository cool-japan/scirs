//! Check for banned `use` statements in Rust source files.
//!
//! ## Rules
//!
//! | Pattern | Allowed in | Severity |
//! |---------|-----------|---------|
//! | `use rand::` / `use ndarray::` / `use ndarray_rand::` | `scirs2-core` only | Warning |
//! | `use openblas` / `use bincode` / `use flate2` / `use zip` / `use brotli` / `use lz4` / `use snap` | nowhere | Error |
//!
//! Test files (`/tests/` directory) and example files (`/examples/` directory)
//! are excluded from all import scanning.

use crate::violation::{PolicyViolation, Severity};
use crate::workspace::WorkspaceInfo;
use std::path::Path;

// ---------------------------------------------------------------------------
// Banned import tables
// ---------------------------------------------------------------------------

/// Import prefixes banned in non-core crates (use scirs2-core re-exports instead).
const NON_CORE_BANNED: &[(&str, &str)] = &[
    (
        "use rand::",
        "use scirs2_core::random (or scirs2_core re-exports) instead of rand directly",
    ),
    (
        "use ndarray::",
        "use scirs2_core::ndarray re-export instead of ndarray directly",
    ),
    (
        "use ndarray_rand::",
        "use scirs2_core random utilities instead of ndarray_rand directly",
    ),
];

/// Import prefixes banned everywhere (no crate is allowed to use these).
const GLOBALLY_BANNED: &[(&str, &str)] = &[
    (
        "use openblas",
        "use oxiblas instead of openblas (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use bincode",
        "use oxicode instead of bincode (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use flate2",
        "use oxiarc-deflate/oxiarc-* instead of flate2 (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use zip::",
        "use oxiarc-archive instead of zip (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use brotli",
        "use oxiarc-brotli instead of brotli (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use lz4::",
        "use oxiarc-lz4 instead of lz4 (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "use snap::",
        "use oxiarc-snappy instead of snap (COOLJAPAN Pure Rust Policy)",
    ),
];

// ---------------------------------------------------------------------------
// Check struct
// ---------------------------------------------------------------------------

/// Check for banned `use` statements across the workspace.
pub struct BannedImportCheck;

impl BannedImportCheck {
    /// Run the check against the given workspace and return all violations.
    pub fn run(&self, workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        for crate_info in &workspace.crates {
            let rs_files = crate::workspace::walk_rust_files(&crate_info.path);

            for file in &rs_files {
                // Skip test and example directories
                if is_excluded_path(file) {
                    continue;
                }

                let content = match std::fs::read_to_string(file) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Check globally banned imports — all crates
                for (pattern, reason) in GLOBALLY_BANNED {
                    scan_file_for_import(
                        file,
                        &content,
                        pattern,
                        reason,
                        Severity::Error,
                        &crate_info.name,
                        &mut violations,
                    );
                }

                // Check non-core banned imports — skip scirs2-core
                if !crate_info.is_core {
                    for (pattern, reason) in NON_CORE_BANNED {
                        scan_file_for_import(
                            file,
                            &content,
                            pattern,
                            reason,
                            Severity::Warning,
                            &crate_info.name,
                            &mut violations,
                        );
                    }
                }
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scan a single file for a specific import pattern, emitting one violation
/// per matching line.
fn scan_file_for_import(
    file: &Path,
    content: &str,
    pattern: &str,
    reason: &str,
    severity: Severity,
    crate_name: &str,
    violations: &mut Vec<PolicyViolation>,
) {
    for (line_idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }
        // Check both top-level and nested (indented) use statements
        if trimmed.starts_with(pattern) || line.trim_start().starts_with(pattern) {
            violations.push(PolicyViolation {
                crate_name: crate_name.to_string(),
                file: file.to_path_buf(),
                line: line_idx + 1,
                message: format!(
                    "banned import '{}': {}",
                    pattern.trim_end_matches("::"),
                    reason
                ),
                severity: severity.clone(),
            });
        }
    }
}

/// Returns `true` if the file is in a directory that should be excluded
/// from import scanning (tests, examples, build scripts).
fn is_excluded_path(file: &Path) -> bool {
    let s = file.to_string_lossy();
    s.contains("/tests/")
        || s.contains("/examples/")
        || s.contains("\\tests\\")
        || s.contains("\\examples\\")
        || file.file_name().is_some_and(|n| n == "build.rs")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{CrateInfo, WorkspaceInfo};
    use std::fs;
    use std::path::{Path, PathBuf};

    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "bi_{}_{}_{}",
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

    fn workspace_with_crate(crate_path: &Path, name: &str, is_core: bool) -> WorkspaceInfo {
        WorkspaceInfo {
            root: crate_path.parent().unwrap_or(crate_path).to_path_buf(),
            crates: vec![CrateInfo {
                name: name.to_string(),
                path: crate_path.to_path_buf(),
                is_core,
            }],
        }
    }

    fn write_src_file(crate_dir: &Path, filename: &str, content: &str) {
        let src = crate_dir.join("src");
        fs::create_dir_all(&src).expect("src dir");
        fs::write(src.join(filename), content).expect("write file");
    }

    #[test]
    fn test_detect_banned_import_rand() {
        let dir = temp_dir("rand");
        write_src_file(&dir, "lib.rs", "use rand::Rng;\nfn foo() {}\n");
        let ws = workspace_with_crate(&dir, "my-crate", false);

        let violations = BannedImportCheck.run(&ws);
        assert!(!violations.is_empty(), "Should detect use rand:: violation");
        assert!(
            violations[0].message.contains("rand"),
            "Message should mention rand"
        );
        assert_eq!(violations[0].line, 1, "Should be on line 1");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_banned_import_ndarray() {
        let dir = temp_dir("ndarray");
        write_src_file(&dir, "lib.rs", "fn ignore() {}\nuse ndarray::Array2;\n");
        let ws = workspace_with_crate(&dir, "my-crate", false);

        let violations = BannedImportCheck.run(&ws);
        assert!(
            !violations.is_empty(),
            "Should detect use ndarray:: violation"
        );
        assert_eq!(violations[0].line, 2, "ndarray on line 2");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_globally_banned_flate2() {
        let dir = temp_dir("flate2");
        write_src_file(&dir, "lib.rs", "use flate2::write::GzEncoder;\n");
        let ws = workspace_with_crate(&dir, "any-crate", false);

        let violations = BannedImportCheck.run(&ws);
        let flate2_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("flate2"))
            .collect();
        assert!(!flate2_v.is_empty(), "Should detect flate2");
        assert_eq!(
            flate2_v[0].severity,
            Severity::Error,
            "flate2 is Error severity"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_globally_banned_in_core_still_caught() {
        // Even scirs2-core is checked for globally banned imports
        let dir = temp_dir("core_global");
        write_src_file(&dir, "lib.rs", "use bincode::serialize;\n");
        let ws = workspace_with_crate(&dir, "scirs2-core", true);

        let violations = BannedImportCheck.run(&ws);
        let bincode_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("bincode"))
            .collect();
        assert!(
            !bincode_v.is_empty(),
            "Even core is checked for globally banned imports"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_core_exempt_from_rand() {
        // scirs2-core may use rand directly
        let dir = temp_dir("core_rand_ok");
        write_src_file(&dir, "lib.rs", "use rand::Rng;\n");
        let ws = workspace_with_crate(&dir, "scirs2-core", true);

        let violations = BannedImportCheck.run(&ws);
        let rand_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("rand"))
            .collect();
        assert!(
            rand_v.is_empty(),
            "scirs2-core should be exempt from rand check"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tests_dir_excluded() {
        let dir = temp_dir("tests_excl");
        let tests = dir.join("tests");
        fs::create_dir_all(&tests).expect("tests dir");
        fs::write(tests.join("integration.rs"), "use rand::thread_rng;\n").expect("write");
        // Also need a src dir or the crate has no files
        write_src_file(&dir, "lib.rs", "// clean\n");
        let ws = workspace_with_crate(&dir, "my-crate", false);

        let violations = BannedImportCheck.run(&ws);
        let rand_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("rand"))
            .collect();
        assert!(rand_v.is_empty(), "tests/ should be excluded");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_comments_not_matched() {
        let dir = temp_dir("comments");
        write_src_file(
            &dir,
            "lib.rs",
            "// use rand::Rng; -- this is a comment\n// use flate2;\n",
        );
        let ws = workspace_with_crate(&dir, "my-crate", false);

        let violations = BannedImportCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "Commented-out imports should not trigger violations"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_clean_file_no_violations() {
        let dir = temp_dir("clean_import");
        write_src_file(
            &dir,
            "lib.rs",
            "use scirs2_core::ndarray::Array2;\nfn foo() {}\n",
        );
        let ws = workspace_with_crate(&dir, "my-crate", false);

        let violations = BannedImportCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "Clean imports should produce no violations"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
