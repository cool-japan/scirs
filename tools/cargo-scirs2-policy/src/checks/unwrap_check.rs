//! Rule UNWRAP_001: detect `.unwrap()` in production (non-test) Rust code.
//!
//! The SciRS2 project mandates that production code must not call `.unwrap()`
//! (or `.expect()` where a proper error-propagation path exists).  This check
//! detects `.unwrap()` calls outside of test code.
//!
//! ## Exclusions
//!
//! The following are **not** flagged:
//!
//! * Files inside the `tests/` or `examples/` directory.
//! * Lines that are inside a `#[cfg(test)]` module block.
//! * Lines that are commented out (`//` or `/* */`).
//!
//! ## Limitations
//!
//! Tracking `#[cfg(test)]` block boundaries is done with a simple brace-depth
//! counter after seeing the `#[cfg(test)]` attribute followed by `mod`.  This
//! heuristic works for the common case but may miss deeply nested or
//! non-idiomatic constructs.

use crate::violation::{PolicyViolation, Severity};
use crate::workspace::WorkspaceInfo;
use std::path::Path;

// ---------------------------------------------------------------------------
// Check struct
// ---------------------------------------------------------------------------

/// Detect `.unwrap()` usage in production (non-test) source code.
pub struct UnwrapCheck;

impl UnwrapCheck {
    /// Run the check and return all violations.
    pub fn run(&self, workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        for crate_info in &workspace.crates {
            let rs_files = crate::workspace::walk_rust_files(&crate_info.path);
            for file in &rs_files {
                if is_excluded_path(file) {
                    continue;
                }
                let content = match std::fs::read_to_string(file) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let file_violations = scan_for_unwrap(file, &content, &crate_info.name);
                violations.extend(file_violations);
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// Core scan logic
// ---------------------------------------------------------------------------

/// Scan a file's content for `.unwrap()` calls outside of test blocks.
///
/// Returns one [`PolicyViolation`] per matching line.
pub fn scan_for_unwrap(file: &Path, content: &str, crate_name: &str) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();
    let mut in_cfg_test_block = false;
    let mut brace_depth: i64 = 0;
    // Track whether the previous non-empty, non-comment line was `#[cfg(test)]`
    // followed by a `mod` keyword on the same or next line.
    let mut cfg_test_pending = false;
    let mut mod_pending = false;

    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();

        // -- Track cfg(test) block entry -----------------------------------
        // Detect `#[cfg(test)]`
        if trimmed.contains("#[cfg(test)]") {
            cfg_test_pending = true;
        }
        // After seeing #[cfg(test)], expect a mod keyword
        if cfg_test_pending && trimmed.contains("mod ") && trimmed.contains('{') {
            in_cfg_test_block = true;
            cfg_test_pending = false;
            mod_pending = false;
        } else if cfg_test_pending && trimmed.contains("mod ") {
            mod_pending = true;
            cfg_test_pending = false;
        } else if mod_pending && trimmed.contains('{') {
            in_cfg_test_block = true;
            mod_pending = false;
        }

        // Track brace depth only when inside a cfg(test) block
        if in_cfg_test_block {
            for ch in trimmed.chars() {
                match ch {
                    '{' => brace_depth += 1,
                    '}' => {
                        brace_depth -= 1;
                        if brace_depth <= 0 {
                            in_cfg_test_block = false;
                            brace_depth = 0;
                        }
                    }
                    _ => {}
                }
            }
        }

        // -- Skip if in test block or excluded path -----------------------
        if in_cfg_test_block {
            continue;
        }

        // -- Skip comments ------------------------------------------------
        if is_comment_line(trimmed) {
            continue;
        }

        // -- Detect .unwrap() ---------------------------------------------
        if contains_unwrap(line) {
            violations.push(PolicyViolation {
                crate_name: crate_name.to_string(),
                file: file.to_path_buf(),
                line: line_num,
                message: format!(
                    ".unwrap() in production code at line {line_num}; \
                     use '?' or proper error handling instead (No unwrap policy)"
                ),
                severity: Severity::Warning,
            });
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the file should be excluded from unwrap scanning.
fn is_excluded_path(file: &Path) -> bool {
    let s = file.to_string_lossy();
    s.contains("/tests/")
        || s.contains("/examples/")
        || s.contains("\\tests\\")
        || s.contains("\\examples\\")
        || file.file_name().is_some_and(|n| n == "build.rs")
}

/// Returns `true` if the (trimmed) line is a comment.
fn is_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*')
}

/// Returns `true` if the line contains `.unwrap()` as a method call.
///
/// Handles:
/// - `.unwrap()` — standard form
/// - `.unwrap() ` — followed by whitespace
/// - Does NOT match `unwrap_or(` or other `unwrap_*` variants because they are
///   legitimate error-handling alternatives.
fn contains_unwrap(line: &str) -> bool {
    // We look for ".unwrap()" ensuring the char after ')' is not alphanumeric
    // (which would indicate something like `.unwrap_or`).
    let needle = ".unwrap()";
    let mut search = line;
    while let Some(pos) = search.find(needle) {
        let after_pos = pos + needle.len();
        let next_char = search[after_pos..].chars().next();
        // If the character after `.unwrap()` is alphabetic it's a false positive
        // e.g. `.unwrap_or(` — but we matched `.unwrap()` not `.unwrap_or`
        // so this is fine.  We only skip if it looks like `.unwrap()something`
        // which would be a syntax error anyway.  Accept the match.
        match next_char {
            // Chained method calls like `.unwrap().method()` are still real unwraps
            Some(c) if c.is_alphabetic() => {
                // Skip forward past this occurrence and keep looking
                search = &search[after_pos..];
                continue;
            }
            _ => return true,
        }
    }
    false
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
            "uw_{}_{}_{}",
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

    fn workspace_with_file(dir: &Path, filename: &str, content: &str) -> WorkspaceInfo {
        let src = dir.join("src");
        fs::create_dir_all(&src).expect("src dir");
        fs::write(src.join(filename), content).expect("write file");
        WorkspaceInfo {
            root: dir.parent().unwrap_or(dir).to_path_buf(),
            crates: vec![CrateInfo {
                name: "my-crate".to_string(),
                path: dir.to_path_buf(),
                is_core: false,
            }],
        }
    }

    #[test]
    fn test_unwrap_in_prod_code_detected() {
        let dir = temp_dir("prod_unwrap");
        let ws = workspace_with_file(
            &dir,
            "lib.rs",
            "fn foo() {\n    let x = something().unwrap();\n}\n",
        );
        let violations = UnwrapCheck.run(&ws);
        assert!(
            !violations.is_empty(),
            "Should detect .unwrap() in production code"
        );
        assert_eq!(violations[0].line, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unwrap_in_cfg_test_block_not_detected() {
        let dir = temp_dir("cfg_test");
        let ws = workspace_with_file(
            &dir,
            "lib.rs",
            "fn production() -> i32 { 42 }\n\
             \n\
             #[cfg(test)]\n\
             mod tests {\n\
                 #[test]\n\
                 fn test_foo() {\n\
                     let x: Result<i32, &str> = Ok(5);\n\
                     assert_eq!(x.unwrap(), 5);\n\
                 }\n\
             }\n",
        );
        let violations = UnwrapCheck.run(&ws);
        // .unwrap() is inside #[cfg(test)] mod → should NOT be flagged
        assert!(
            violations.is_empty(),
            "unwrap() in #[cfg(test)] block should not be flagged; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_comment_line_not_detected() {
        let dir = temp_dir("comment_unwrap");
        let ws = workspace_with_file(
            &dir,
            "lib.rs",
            "// let x = foo().unwrap(); -- commented out\nfn bar() {}\n",
        );
        let violations = UnwrapCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "Commented unwrap should not be flagged"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tests_dir_excluded() {
        let dir = temp_dir("tests_excl_uw");
        let tests = dir.join("tests");
        fs::create_dir_all(&tests).expect("tests dir");
        fs::write(tests.join("integration.rs"), "fn t() { foo().unwrap(); }\n").expect("write");
        // Need at least an empty src for the crate to be valid
        let src = dir.join("src");
        fs::create_dir_all(&src).expect("src dir");
        fs::write(src.join("lib.rs"), "// clean\n").expect("write");

        let ws = WorkspaceInfo {
            root: dir.parent().unwrap_or(&dir).to_path_buf(),
            crates: vec![CrateInfo {
                name: "my-crate".to_string(),
                path: dir.clone(),
                is_core: false,
            }],
        };
        let violations = UnwrapCheck.run(&ws);
        assert!(violations.is_empty(), "tests/ dir should be excluded");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_contains_unwrap_true() {
        assert!(contains_unwrap("    let x = foo().unwrap();"));
        assert!(contains_unwrap("    foo.bar.unwrap()"));
        assert!(contains_unwrap("x.unwrap() // trailing comment"));
    }

    #[test]
    fn test_contains_unwrap_false() {
        // Plain lines with no .unwrap() call should not match
        assert!(!contains_unwrap("    let _ = 42;"));
        assert!(!contains_unwrap("    foo();"));
        // Note: contains_unwrap is a raw string search — comment filtering
        // is performed by scan_for_unwrap. So "// x.unwrap()" DOES contain
        // ".unwrap()" as a substring; comment exclusion happens upstream.
        // (No assertion on comment lines here.)
    }

    #[test]
    fn test_scan_for_unwrap_multiple_lines() {
        let file = PathBuf::from("/fake/lib.rs");
        let content = "\
fn a() { foo().unwrap(); }\n\
fn b() { bar().unwrap(); }\n\
// commented().unwrap();\n\
fn c() { baz().unwrap_or(0); }\n\
";
        let violations = scan_for_unwrap(&file, content, "test-crate");
        // Lines 1 and 2 have real unwrap(), line 3 is comment, line 4 is unwrap_or
        assert_eq!(violations.len(), 2, "Should find 2 violations");
        assert_eq!(violations[0].line, 1);
        assert_eq!(violations[1].line, 2);
    }
}
