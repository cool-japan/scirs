//! Rules `IGNORE_AUDIT_001`..`IGNORE_AUDIT_004`: enforce the `#[ignore]`
//! reason taxonomy and outlaw a family of "fake-passing" test patterns that
//! the 0.6.5 workspace-wide ignore-legitimacy audit found hiding real defects
//! (a self-deadlock, two O(n^2) generator bugs, a ~75s unbounded TCP stall,
//! and several vacuous "Skipping"-only test bodies, among others).
//!
//! ## Rules
//!
//! | Rule | Detects |
//! |------|---------|
//! | `IGNORE_AUDIT_001` | a bare `#[ignore]` with no reason string |
//! | `IGNORE_AUDIT_002` | an `#[ignore = "..."]` reason not tagged with an approved category prefix |
//! | `IGNORE_AUDIT_003` | `assert!(true)` — a tautological, always-passing assertion |
//! | `IGNORE_AUDIT_004` | a `#[test]` fn whose `Err(_)`/`Err(e)` match arm body is *only* a `println!`/`eprintln!` call mentioning "skipping" |
//!
//! ## Approved `#[ignore = "..."]` reason prefixes
//!
//! Every `#[ignore = "..."]` reason must start with one of:
//!
//! * `requires-gpu:` — needs real GPU/CUDA/Metal/wgpu hardware not present in CI
//! * `requires-env:` — needs some other environment/feature not enabled by default
//! * `slow:` — correct, but too slow for the default test profile
//! * `bench:` — a benchmark entry point, not a correctness test
//! * `not-implemented:` — the feature under test is a documented, honest gap
//!
//! ## Scope
//!
//! Unlike [`crate::checks::unwrap_check`], this check scans a crate's
//! **entire** directory tree (`src/`, `tests/`, `benches/`, `examples/` —
//! see [`crate::workspace::walk_all_rust_files`]), not just `src/`, because
//! `#[ignore]`/`#[test]` predominantly live in `tests/` and in inline
//! `#[cfg(test)]` modules under `src/`. Unlike `unwrap_check`, it does
//! **not** exempt `#[cfg(test)]` blocks — those are precisely where this
//! check's subject matter lives.
//!
//! `cargo-scirs2-policy`'s own crate (`cargo-scirs2-policy`) is excluded
//! from this check (see the private `EXCLUDED_CRATES` constant in this
//! module): its own unit tests construct
//! literal fixture strings containing the very patterns this check looks
//! for (a bare `#[ignore]`, `assert!(true)`, an `Err(_) => println!("...
//! Skipping ...")` arm) in order to unit-test the detector itself. Scanning
//! those fixtures here would be self-referential noise, not a real hygiene
//! defect — the same class of exclusion [`crate::checks::banned_imports`]
//! already applies for `scirs2-core`.
//!
//! ## Limitations
//!
//! Detection is line/regex-based, matching the pragmatic style of the
//! sibling checks in this module: comments are skipped with the same
//! line-prefix heuristic as [`crate::checks::unwrap_check`] and
//! [`crate::checks::banned_imports`] (a block comment whose *interior*
//! lines don't themselves start with a comment marker is not fully
//! tracked). `IGNORE_AUDIT_004`'s match-arm body collection is a
//! brace-counting approximation, also in the same spirit as
//! `unwrap_check`'s `#[cfg(test)]` block tracking.

use crate::violation::{PolicyViolation, Severity};
use crate::workspace::WorkspaceInfo;
use std::path::Path;

// ---------------------------------------------------------------------------
// Taxonomy
// ---------------------------------------------------------------------------

/// Reason prefixes accepted for `#[ignore = "..."]` attributes.
const APPROVED_PREFIXES: &[&str] = &[
    "requires-gpu:",
    "requires-env:",
    "slow:",
    "bench:",
    "not-implemented:",
];

/// Crates excluded from this check entirely (see module docs: "Scope").
const EXCLUDED_CRATES: &[&str] = &["cargo-scirs2-policy"];

// ---------------------------------------------------------------------------
// Check struct
// ---------------------------------------------------------------------------

/// Enforce `#[ignore]` reason taxonomy and outlaw fake-passing test patterns.
pub struct IgnoreAuditCheck;

impl IgnoreAuditCheck {
    /// Run the check and return all violations.
    pub fn run(&self, workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        for crate_info in &workspace.crates {
            if EXCLUDED_CRATES.contains(&crate_info.name.as_str()) {
                continue;
            }
            let rs_files = crate::workspace::walk_all_rust_files(&crate_info.path);
            for file in &rs_files {
                let content = match std::fs::read_to_string(file) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                violations.extend(scan_file(file, &content, &crate_info.name));
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// Core scan logic
// ---------------------------------------------------------------------------

/// Run all four rules over a single file's content.
pub fn scan_file(file: &Path, content: &str, crate_name: &str) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();
    violations.extend(scan_ignore_attributes(file, content, crate_name));
    violations.extend(scan_assert_true(file, content, crate_name));
    violations.extend(scan_skipping_err_arms(file, content, crate_name));
    violations
}

/// `IGNORE_AUDIT_001` / `IGNORE_AUDIT_002`: bare `#[ignore]` or a reason
/// missing an approved category prefix.
fn scan_ignore_attributes(file: &Path, content: &str, crate_name: &str) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();
        if is_comment_line(trimmed) {
            continue;
        }
        let Some(after) = trimmed.strip_prefix("#[ignore") else {
            continue;
        };
        let after = after.trim_start();

        if after.starts_with(']') {
            // Rule (a): bare `#[ignore]`.
            violations.push(PolicyViolation {
                crate_name: crate_name.to_string(),
                file: file.to_path_buf(),
                line: line_num,
                message: format!(
                    "IGNORE_AUDIT_001: bare #[ignore] with no reason at line {line_num}; \
                     add #[ignore = \"<prefix>: <details>\"] with prefix one of \
                     requires-gpu:, requires-env:, slow:, bench:, not-implemented:"
                ),
                severity: Severity::Error,
            });
        } else if let Some(after_eq) = after.strip_prefix('=') {
            // Rule (b): `#[ignore = "..."]` — validate the reason's prefix.
            let after_eq = after_eq.trim_start();
            if let Some(reason) = after_eq.strip_prefix('"') {
                if !APPROVED_PREFIXES.iter().any(|p| reason.starts_with(p)) {
                    violations.push(PolicyViolation {
                        crate_name: crate_name.to_string(),
                        file: file.to_path_buf(),
                        line: line_num,
                        message: format!(
                            "IGNORE_AUDIT_002: #[ignore = \"...\"] reason at line {line_num} \
                             does not start with an approved prefix (requires-gpu:, \
                             requires-env:, slow:, bench:, not-implemented:); reason begins: {:?}",
                            first_n_chars(reason, 60)
                        ),
                        severity: Severity::Error,
                    });
                }
            }
            // A reason that isn't a plain string literal (unusual) is left
            // unflagged rather than risk a false positive.
        }
        // Any other shape after `#[ignore` (e.g. an unrelated attribute that
        // happens to share the prefix) is intentionally left unflagged.
    }

    violations
}

/// `IGNORE_AUDIT_003`: `assert!(true)` anywhere (a tautological assertion).
fn scan_assert_true(file: &Path, content: &str, crate_name: &str) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();
        if is_comment_line(trimmed) {
            continue;
        }
        if contains_assert_true(line) {
            violations.push(PolicyViolation {
                crate_name: crate_name.to_string(),
                file: file.to_path_buf(),
                line: line_num,
                message: format!(
                    "IGNORE_AUDIT_003: assert!(true) at line {line_num} is a tautological, \
                     always-passing assertion; assert the real expected behavior instead \
                     (or delete it if only compilation is being exercised)"
                ),
                severity: Severity::Error,
            });
        }
    }

    violations
}

/// `IGNORE_AUDIT_004`: a `#[test]` fn with an `Err(_)`/`Err(e)` match arm
/// whose entire body is a `println!`/`eprintln!` call mentioning "skipping".
fn scan_skipping_err_arms(file: &Path, content: &str, crate_name: &str) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    // Small state machine tracking "have we seen `#[test]`, are we waiting
    // for its `fn`'s opening brace, and — once inside — its brace depth" —
    // mirroring the `#[cfg(test)]` block tracking in `unwrap_check`, but for
    // an individual `#[test]` fn rather than a whole module.
    let mut in_test_fn = false;
    let mut test_attr_pending = false;
    let mut fn_brace_pending = false;
    let mut fn_brace_depth: i64 = 0;

    for (i, &raw_line) in lines.iter().enumerate() {
        let trimmed = raw_line.trim();
        let line_is_comment = is_comment_line(trimmed);

        if in_test_fn {
            if !line_is_comment {
                if let Some(arm_rest) = err_arm_rest(trimmed) {
                    if let Some(body) = collect_arm_body(&lines, i, arm_rest) {
                        if is_skipping_print_only_body(&body) {
                            violations.push(make_skipping_violation(file, crate_name, i + 1));
                        }
                    }
                }
            }
            fn_brace_depth += count_brace_delta(raw_line);
            if fn_brace_depth <= 0 {
                in_test_fn = false;
                fn_brace_depth = 0;
            }
            continue;
        }

        if line_is_comment {
            continue;
        }

        if fn_brace_pending {
            if raw_line.contains('{') {
                fn_brace_pending = false;
                let d = count_brace_delta(raw_line);
                if d > 0 {
                    in_test_fn = true;
                    fn_brace_depth = d;
                }
            }
            continue;
        }

        if test_attr_pending {
            if trimmed.starts_with("#[") {
                continue; // another stacked attribute; keep waiting for `fn`
            }
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
                test_attr_pending = false;
                if raw_line.contains('{') {
                    let d = count_brace_delta(raw_line);
                    if d > 0 {
                        in_test_fn = true;
                        fn_brace_depth = d;
                    }
                } else {
                    fn_brace_pending = true;
                }
            } else {
                test_attr_pending = false;
            }
            continue;
        }

        if trimmed.contains("#[test]") {
            // Handle both `#[test]` on its own line and the (rarer)
            // `#[test] fn foo() {` single-line style.
            if trimmed.starts_with("fn ")
                || trimmed.contains(" fn ")
                || trimmed.ends_with("#[test]")
            {
                if trimmed.ends_with("#[test]") {
                    test_attr_pending = true;
                } else if raw_line.contains('{') {
                    let d = count_brace_delta(raw_line);
                    if d > 0 {
                        in_test_fn = true;
                        fn_brace_depth = d;
                    }
                } else {
                    fn_brace_pending = true;
                }
            } else {
                test_attr_pending = true;
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_skipping_violation(file: &Path, crate_name: &str, line: usize) -> PolicyViolation {
    PolicyViolation {
        crate_name: crate_name.to_string(),
        file: file.to_path_buf(),
        line,
        message: format!(
            "IGNORE_AUDIT_004: #[test] fn has an Err(_)/Err(e) match arm at line {line} whose \
             body is only a println!/eprintln! mentioning \"skipping\" — this silently no-ops \
             the test on error instead of asserting real behavior; assert the expected outcome \
             (or panic!/return a real Result) instead"
        ),
        severity: Severity::Error,
    }
}

/// Returns `true` if the (trimmed) line is a comment — matches the same
/// heuristic used by the sibling checks in this module.
fn is_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*')
}

/// Truncate `s` to at most `n` `char`s (used to keep violation messages short).
fn first_n_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Returns `true` if `line` contains `assert!(true)` (allowing internal
/// whitespace, e.g. `assert! ( true )`), or `assert!(true, "message")` —
/// equally tautological since the condition is still the literal `true`.
///
/// Does *not* match `assert!(true_flag)` or similar identifiers — the
/// character immediately following `true` must be a non-identifier boundary.
fn contains_assert_true(line: &str) -> bool {
    let mut search = line;
    while let Some(pos) = search.find("assert!") {
        let after_kw = &search[pos + "assert!".len()..];
        let after_paren = after_kw.trim_start();
        if let Some(after_open) = after_paren.strip_prefix('(') {
            let after_open = after_open.trim_start();
            if let Some(after_true) = after_open.strip_prefix("true") {
                let next_char = after_true.chars().next();
                let at_boundary = !matches!(next_char, Some(c) if c.is_alphanumeric() || c == '_');
                if at_boundary {
                    let after_true = after_true.trim_start();
                    if after_true.starts_with(')') || after_true.starts_with(',') {
                        return true;
                    }
                }
            }
        }
        search = &search[pos + "assert!".len()..];
    }
    false
}

/// If `trimmed` starts with `Err(_) =>` or `Err(e) =>` (allowing arbitrary
/// whitespace before `=>`), returns the trimmed remainder of the line after
/// `=>`.
fn err_arm_rest(trimmed: &str) -> Option<&str> {
    for pat in ["Err(_)", "Err(e)"] {
        if let Some(rest) = trimmed.strip_prefix(pat) {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix("=>") {
                return Some(rest.trim_start());
            }
        }
    }
    None
}

/// Net change in brace depth contributed by a raw source line — a plain
/// character count, same approximation `unwrap_check` uses for its
/// `#[cfg(test)]` block tracking (braces inside string/format literals are
/// counted too, but in practice they are balanced within the line).
fn count_brace_delta(line: &str) -> i64 {
    let mut delta = 0i64;
    for ch in line.chars() {
        match ch {
            '{' => delta += 1,
            '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

/// Given the text right after an arm's opening `{` (already consumed),
/// find the byte offset of that block's matching `}` on the *same* line, if
/// present.
fn find_matching_close_same_line(s: &str) -> Option<usize> {
    let mut depth = 1i64;
    for (idx, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

/// Collect the full body text of an `Err(_) => ...` / `Err(e) => ...` match
/// arm starting at line `i`, where `arm_rest` is the trimmed content
/// immediately after `=>` on that line.
///
/// Returns `None` if a block-form arm's closing brace is never found before
/// EOF (rather than guessing).
fn collect_arm_body(lines: &[&str], i: usize, arm_rest: &str) -> Option<String> {
    let Some(after_open) = arm_rest.strip_prefix('{') else {
        // Non-block, single-expression arm: the whole rest of the line
        // (minus a trailing arm-separator comma) is the body. A non-block
        // match arm can only ever be one expression, so this is inherently
        // "the whole body" — no further collection needed.
        return Some(arm_rest.trim_end_matches(',').trim().to_string());
    };

    // Block-form arm: `Err(_) => { ... }`, possibly spanning multiple lines.
    if let Some(close_idx) = find_matching_close_same_line(after_open) {
        return Some(after_open[..close_idx].trim().to_string());
    }

    let mut depth: i64 = 1; // for the opening brace already consumed above
    let mut body = String::new();
    let first_seg = after_open.trim();
    if !first_seg.is_empty() {
        body.push_str(first_seg);
        body.push(' ');
    }

    let mut j = i;
    loop {
        j += 1;
        if j >= lines.len() {
            return None; // unterminated block — bail out rather than guess
        }
        let line = lines[j];
        let delta = count_brace_delta(line);
        if depth + delta <= 0 {
            if let Some(close_pos) = line.rfind('}') {
                let before = line[..close_pos].trim();
                if !before.is_empty() {
                    body.push_str(before);
                    body.push(' ');
                }
            }
            return Some(body.trim().to_string());
        }
        depth += delta;
        let t = line.trim();
        if !t.is_empty() && !is_comment_line(t) {
            body.push_str(t);
            body.push(' ');
        }
    }
}

/// Returns `true` when `body` (already whitespace-collapsed) is *exactly*
/// one `println!`/`eprintln!` call whose formatted text mentions "skipping"
/// (case-insensitive).
fn is_skipping_print_only_body(body: &str) -> bool {
    let body = body.trim();
    let body = body.strip_suffix(';').unwrap_or(body).trim_end();
    let body = body.strip_suffix(',').unwrap_or(body).trim_end();

    (body.starts_with("println!(") || body.starts_with("eprintln!("))
        && body.ends_with(')')
        && body.to_lowercase().contains("skipping")
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
            "ia_{}_{}_{}",
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

    fn workspace_with_file(dir: &Path, rel_path: &str, content: &str) -> WorkspaceInfo {
        let full = dir.join(rel_path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(&full, content).expect("write file");
        WorkspaceInfo {
            root: dir.parent().unwrap_or(dir).to_path_buf(),
            crates: vec![CrateInfo {
                name: "my-crate".to_string(),
                path: dir.to_path_buf(),
                is_core: false,
            }],
        }
    }

    // -- Rule (a): bare #[ignore] --------------------------------------

    #[test]
    fn test_bare_ignore_detected() {
        let dir = temp_dir("bare_ignore");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "#[cfg(test)]\nmod tests {\n    #[test]\n    #[ignore]\n    fn test_x() {}\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_001")),
            "Should detect bare #[ignore]; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bare_ignore_with_trailing_comment_still_bare() {
        // `#[ignore] // some prose` is still a *bare* attribute — the
        // structured `= "..."` reason form is what the taxonomy requires.
        let dir = temp_dir("bare_ignore_comment");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\n#[ignore] // Slow test - takes a while\nfn test_x() {}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_001")),
            "Trailing prose comment does not satisfy the reason taxonomy; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Rule (b): reason prefix taxonomy --------------------------------

    #[test]
    fn test_ignore_reason_missing_prefix_detected() {
        let dir = temp_dir("bad_prefix");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\n#[ignore = \"GPU availability varies by environment\"]\nfn test_x() {}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_002")),
            "Should detect an un-prefixed reason; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_all_five_approved_prefixes_accepted() {
        for prefix in [
            "requires-gpu",
            "requires-env",
            "slow",
            "bench",
            "not-implemented",
        ] {
            let dir = temp_dir(&format!("prefix_{prefix}"));
            let content = format!(
                "#[test]\n#[ignore = \"{prefix}: a legitimate reason\"]\nfn test_x() {{}}\n"
            );
            let ws = workspace_with_file(&dir, "tests/it.rs", &content);
            let violations = IgnoreAuditCheck.run(&ws);
            assert!(
                violations.is_empty(),
                "prefix {prefix}: should not be flagged; got: {:?}",
                violations
            );
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn test_ignore_reason_bad_prefix_similar_word_still_rejected() {
        // "slowly:" is not the approved "slow:" prefix.
        let dir = temp_dir("near_miss_prefix");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\n#[ignore = \"slowly going to fail\"]\nfn test_x() {}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_002")),
            "A near-miss prefix should still be rejected; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Rule (c): assert!(true) -----------------------------------------

    #[test]
    fn test_assert_true_detected() {
        let dir = temp_dir("assert_true");
        let ws = workspace_with_file(&dir, "src/lib.rs", "fn f() {\n    assert!(true);\n}\n");
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_003")),
            "Should detect assert!(true); got: {:?}",
            violations
        );
        assert_eq!(
            violations
                .iter()
                .find(|v| v.message.contains("IGNORE_AUDIT_003"))
                .unwrap()
                .line,
            2
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_assert_true_with_message_detected() {
        let dir = temp_dir("assert_true_msg");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "fn f() {\n    assert!(true, \"compiles\");\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_003")),
            "assert!(true, \"msg\") is equally tautological; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_assert_with_real_condition_not_flagged() {
        let dir = temp_dir("assert_real");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "fn f(x: i32) {\n    assert!(x > 0);\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_003")),
            "A real condition should not be flagged; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_assert_true_like_identifier_not_flagged() {
        // `true_flag` is an identifier, not the literal `true` — must not
        // be confused with the tautological form.
        let dir = temp_dir("assert_true_ident");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "fn f(true_flag: bool) {\n    assert!(true_flag);\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_003")),
            "assert!(true_flag) must not match the assert!(true) rule; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_commented_out_assert_true_not_flagged() {
        let dir = temp_dir("assert_true_comment");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "// assert!(true); -- left here for reference\nfn f() {}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "Commented-out assert!(true) should not be flagged; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Rule (d): Skipping-only Err arm inside a #[test] fn -------------

    #[test]
    fn test_skipping_err_arm_single_line_detected() {
        let dir = temp_dir("skip_single_line");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\nfn test_x() {\n    match compute() {\n        Ok(v) => assert_eq!(v, 1),\n        Err(_) => println!(\"Skipping: not available\"),\n    }\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_004")),
            "Should detect a single-line Skipping-only Err(_) arm; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_skipping_err_arm_block_form_detected() {
        let dir = temp_dir("skip_block");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\nfn test_x() {\n    match compute() {\n        Ok(v) => assert_eq!(v, 1),\n        Err(e) => {\n            println!(\"Skipping due to {}\", e);\n        }\n    }\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_004")),
            "Should detect a block-form Skipping-only Err(e) arm; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_err_arm_outside_test_fn_not_flagged() {
        // The identical Skipping-println pattern in a normal (non-#[test])
        // function is legitimate production error handling, not a vacuous
        // test — rule (d) only applies inside #[test] fns.
        let dir = temp_dir("skip_outside_test");
        let ws = workspace_with_file(
            &dir,
            "src/lib.rs",
            "fn load(path: &str) -> Option<Data> {\n    match read(path) {\n        Ok(d) => Some(d),\n        Err(_) => {\n            println!(\"Skipping unreadable file\");\n            None\n        }\n    }\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_004")),
            "Skipping-println outside a #[test] fn must not be flagged; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_err_arm_with_real_assertion_not_flagged() {
        let dir = temp_dir("err_real_assert");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\nfn test_x() {\n    match compute() {\n        Ok(v) => assert_eq!(v, 1),\n        Err(e) => panic!(\"unexpected error: {}\", e),\n    }\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_004")),
            "An Err(e) arm that panics is a real assertion, not a silent skip; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_err_arm_with_extra_statement_not_flagged() {
        // Body has more than *just* the println! call — not covered by rule (d).
        let dir = temp_dir("err_extra_stmt");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\nfn test_x() {\n    match compute() {\n        Ok(v) => assert_eq!(v, 1),\n        Err(_) => {\n            println!(\"Skipping\");\n            return;\n        }\n    }\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_004")),
            "A body with more than the println! call should not be flagged; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Scope: tests/benches/examples are scanned (unlike unwrap_check) --

    #[test]
    fn test_tests_dir_is_scanned_not_excluded() {
        let dir = temp_dir("tests_dir_scanned");
        let ws = workspace_with_file(
            &dir,
            "tests/integration.rs",
            "#[test]\n#[ignore]\nfn test_x() {}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            !violations.is_empty(),
            "tests/ must be scanned by ignore_audit (unlike unwrap_check); got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_examples_dir_is_scanned() {
        let dir = temp_dir("examples_dir_scanned");
        let ws = workspace_with_file(
            &dir,
            "examples/demo.rs",
            "fn main() {\n    assert!(true); // Compilation test\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("IGNORE_AUDIT_003")),
            "examples/ must be scanned; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Self-exemption for cargo-scirs2-policy --------------------------

    #[test]
    fn test_cargo_scirs2_policy_crate_is_excluded() {
        let dir = temp_dir("self_exempt");
        let src = dir.join("src");
        fs::create_dir_all(&src).expect("src dir");
        fs::write(
            src.join("lib.rs"),
            "#[cfg(test)]\nmod tests {\n    #[test]\n    #[ignore]\n    fn test_x() {\n        assert!(true);\n    }\n}\n",
        )
        .expect("write file");
        let ws = WorkspaceInfo {
            root: dir.parent().unwrap_or(&dir).to_path_buf(),
            crates: vec![CrateInfo {
                name: "cargo-scirs2-policy".to_string(),
                path: dir.clone(),
                is_core: false,
            }],
        };
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "cargo-scirs2-policy's own crate must be exempt; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Clean file → no violations ---------------------------------------

    #[test]
    fn test_clean_file_no_violations() {
        let dir = temp_dir("clean");
        let ws = workspace_with_file(
            &dir,
            "tests/it.rs",
            "#[test]\n#[ignore = \"slow: takes about 90s on CI\"]\nfn test_x() {\n    assert_eq!(2 + 2, 4);\n}\n",
        );
        let violations = IgnoreAuditCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "A clean, well-tagged test should have no violations; got: {:?}",
            violations
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Unit-level helper coverage ---------------------------------------

    #[test]
    fn test_contains_assert_true_variants() {
        assert!(contains_assert_true("assert!(true);"));
        assert!(contains_assert_true("    assert! ( true ) ;"));
        assert!(contains_assert_true("assert!(true, \"msg\");"));
        assert!(!contains_assert_true("assert!(true_val);"));
        assert!(!contains_assert_true("assert!(x == true);"));
        assert!(!contains_assert_true("assert_eq!(a, b);"));
    }

    #[test]
    fn test_is_skipping_print_only_body_variants() {
        assert!(is_skipping_print_only_body(
            "println!(\"Skipping: not available\")"
        ));
        assert!(is_skipping_print_only_body(
            "eprintln!(\"skipping due to {}\", e);"
        ));
        assert!(!is_skipping_print_only_body(
            "println!(\"Skipping\"); return;"
        ));
        assert!(!is_skipping_print_only_body("panic!(\"Skipping is bad\")"));
        assert!(!is_skipping_print_only_body(
            "println!(\"nothing relevant here\")"
        ));
    }
}
