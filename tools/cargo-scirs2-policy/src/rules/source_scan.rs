//! Rules SOURCE_SCAN_001 and SOURCE_SCAN_002: source code scanning.
//!
//! Detects direct usage of `rand::` and `ndarray::` in non-core crates.
//! Non-core crates should route through `scirs2_core::random` and
//! `scirs2_core::ndarray` instead.

use super::{PolicyRule, Severity, Violation};
use std::path::Path;

// ---------------------------------------------------------------------------
// Rule implementations
// ---------------------------------------------------------------------------

/// Rule SOURCE_SCAN_001: no `use rand::` in non-core Rust source files.
pub struct DirectRandUsageRule;

impl PolicyRule for DirectRandUsageRule {
    fn id(&self) -> &'static str {
        "SOURCE_SCAN_001"
    }

    fn description(&self) -> &'static str {
        "Non-core crates must not import rand:: directly; \
         use scirs2_core::random utilities instead"
    }

    fn check(&self, workspace: &Path) -> Vec<Violation> {
        scan_for_pattern(
            workspace,
            "use rand::",
            "scirs2-core",
            self.id(),
            "Direct `use rand::` import found; use scirs2_core::random instead",
            Severity::Warning,
        )
    }
}

/// Rule SOURCE_SCAN_002: no `use ndarray::` in non-core Rust source files.
pub struct DirectNdarrayUsageRule;

impl PolicyRule for DirectNdarrayUsageRule {
    fn id(&self) -> &'static str {
        "SOURCE_SCAN_002"
    }

    fn description(&self) -> &'static str {
        "Non-core crates must not use `use ndarray::` directly; \
         use scirs2_core::ndarray re-export instead"
    }

    fn check(&self, workspace: &Path) -> Vec<Violation> {
        scan_for_pattern(
            workspace,
            "use ndarray::",
            "scirs2-core",
            self.id(),
            "Direct `use ndarray::` import found; use scirs2_core::ndarray instead",
            Severity::Info,
        )
    }
}

// ---------------------------------------------------------------------------
// Shared scan logic
// ---------------------------------------------------------------------------

/// Walk `workspace`, scanning every `.rs` file that is not under the
/// `exempt_crate_dir` directory, and collect a [`Violation`] for each file
/// containing `pattern`.
///
/// Test files (paths containing `/tests/`) and example files (`/examples/`)
/// are excluded from scanning because they may legitimately use the raw crates.
pub(crate) fn scan_for_pattern(
    workspace: &Path,
    pattern: &str,
    exempt_crate_dir: &str,
    rule_id: &str,
    message: &str,
    severity: Severity,
) -> Vec<Violation> {
    let mut violations = Vec::new();

    let walker = walkdir::WalkDir::new(workspace)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && name != "target"
        });

    for entry in walker.flatten() {
        let path = entry.path();
        // Only process .rs files
        if path.extension().is_some_and(|e| e == "rs") {
            let path_str = path.to_string_lossy();
            // Exempt the core crate itself
            if path_str.contains(exempt_crate_dir) {
                continue;
            }
            // Exempt tests and examples
            if path_str.contains("/tests/") || path_str.contains("/examples/") {
                continue;
            }
            // Also skip this tool's own source
            if path_str.contains("cargo-scirs2-policy") {
                continue;
            }

            if let Ok(content) = std::fs::read_to_string(path) {
                if content.contains(pattern) {
                    violations.push(Violation {
                        rule_id: rule_id.to_string(),
                        message: message.to_string(),
                        file: Some(path.display().to_string()),
                        severity: severity.clone(),
                    });
                }
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(suffix: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "policy_scan_{}_{}_{}",
            std::process::id(),
            suffix,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&base).expect("create temp dir");
        base
    }

    #[test]
    fn test_rand_usage_found() {
        let dir = temp_dir("rand_found");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).expect("create src");
        fs::write(src_dir.join("lib.rs"), "use rand::Rng;\nfn foo() {}\n").expect("write lib.rs");

        let rule = DirectRandUsageRule;
        let violations = rule.check(&dir);
        assert!(!violations.is_empty(), "Should find rand violation");
        assert_eq!(violations[0].severity, Severity::Warning);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rand_usage_not_found_clean() {
        let dir = temp_dir("rand_clean");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).expect("create src");
        fs::write(
            src_dir.join("lib.rs"),
            "use scirs2_core::random::Rng;\nfn foo() {}\n",
        )
        .expect("write lib.rs");

        let rule = DirectRandUsageRule;
        let violations = rule.check(&dir);
        assert!(violations.is_empty(), "Should have no rand violations");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ndarray_usage_found() {
        let dir = temp_dir("ndarray_found");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).expect("create src");
        fs::write(
            src_dir.join("lib.rs"),
            "use ndarray::Array2;\nfn bar() {}\n",
        )
        .expect("write lib.rs");

        let rule = DirectNdarrayUsageRule;
        let violations = rule.check(&dir);
        assert!(!violations.is_empty(), "Should find ndarray violation");
        assert_eq!(violations[0].severity, Severity::Info);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_exempt_core_crate_skipped() {
        // A file whose path contains "scirs2-core" should be skipped
        let dir = temp_dir("core_exempt");
        let core_dir = dir.join("scirs2-core").join("src");
        fs::create_dir_all(&core_dir).expect("create core src");
        fs::write(
            core_dir.join("lib.rs"),
            "use rand::Rng;\npub fn core_fn() {}\n",
        )
        .expect("write lib.rs");

        let rule = DirectRandUsageRule;
        let violations = rule.check(&dir);
        assert!(
            violations.is_empty(),
            "scirs2-core source should be exempt from rand scan"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tests_dir_skipped() {
        let dir = temp_dir("tests_skip");
        let tests_dir = dir.join("tests");
        fs::create_dir_all(&tests_dir).expect("create tests dir");
        fs::write(
            tests_dir.join("integration_test.rs"),
            "use rand::thread_rng;\n",
        )
        .expect("write test file");

        let rule = DirectRandUsageRule;
        let violations = rule.check(&dir);
        assert!(
            violations.is_empty(),
            "Files under /tests/ should be exempt"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_examples_dir_skipped() {
        let dir = temp_dir("examples_skip");
        let ex_dir = dir.join("examples");
        fs::create_dir_all(&ex_dir).expect("create examples dir");
        fs::write(ex_dir.join("demo.rs"), "use ndarray::arr2;\n").expect("write example file");

        let rule = DirectNdarrayUsageRule;
        let violations = rule.check(&dir);
        assert!(
            violations.is_empty(),
            "Files under /examples/ should be exempt"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
