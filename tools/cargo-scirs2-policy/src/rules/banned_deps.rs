//! Rule BANNED_DEP_001: banned direct dependencies in workspace members.
//!
//! Checks each `Cargo.toml` found under the workspace root for direct use of
//! packages that are prohibited by the COOLJAPAN Pure Rust Policy.

use super::{PolicyRule, Severity, Violation};
use std::path::Path;

/// Pairs of (crate_name, reason) for banned direct dependencies.
const BANNED_DEPS: &[(&str, &str)] = &[
    (
        "zip",
        "Use oxiarc-archive instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "flate2",
        "Use oxiarc-deflate/oxiarc-* instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "zstd",
        "Use oxiarc-zstd instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "bzip2",
        "Use oxiarc-bzip2 instead (COOLJAPAN Pure Rust Policy)",
    ),
    ("lz4", "Use oxiarc-lz4 instead (COOLJAPAN Pure Rust Policy)"),
    (
        "snap",
        "Use oxiarc-snappy instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "brotli",
        "Use oxiarc-brotli instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "miniz_oxide",
        "Use oxiarc-deflate instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "bincode",
        "Use oxicode instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "openblas-src",
        "Use oxiblas instead (COOLJAPAN Pure Rust Policy)",
    ),
    (
        "blas-src",
        "Use oxiblas instead (COOLJAPAN Pure Rust Policy)",
    ),
    ("z3", "Use oxiz instead (COOLJAPAN Pure Rust Policy)"),
    (
        "ndarray-npy",
        "Use custom binary format instead (removed to eliminate zip crate dependency)",
    ),
];

/// Rule: workspace members must not directly depend on banned packages.
pub struct BannedDirectDepRule;

impl PolicyRule for BannedDirectDepRule {
    fn id(&self) -> &'static str {
        "BANNED_DEP_001"
    }

    fn description(&self) -> &'static str {
        "Workspace members must not directly depend on banned packages \
         (zip, flate2, bincode, openblas-src, etc.)"
    }

    fn check(&self, workspace: &Path) -> Vec<Violation> {
        let mut violations = Vec::new();

        let walker = walkdir::WalkDir::new(workspace)
            .max_depth(4)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                // Skip target directory, hidden dirs, and the tool itself
                !name.starts_with('.') && name != "target"
            });

        for entry in walker.flatten() {
            let path = entry.path();
            if path.file_name().is_some_and(|n| n == "Cargo.toml") {
                // Skip the workspace root Cargo.toml — it may list banned crates
                // only in [workspace.dependencies] as reminders / with feature flags.
                if path == workspace.join("Cargo.toml") {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(path) {
                    for &(banned, reason) in BANNED_DEPS {
                        if is_direct_dep(&content, banned) {
                            violations.push(Violation {
                                rule_id: self.id().to_string(),
                                message: format!(
                                    "Banned dependency '{}' found: {}",
                                    banned, reason
                                ),
                                file: Some(path.display().to_string()),
                                severity: Severity::Error,
                            });
                        }
                    }
                }
            }
        }

        violations
    }
}

/// Returns `true` if `dep_name` appears as a direct dependency entry.
///
/// Looks for the following patterns (all common in `Cargo.toml`):
/// - `dep_name = "..."`
/// - `dep_name = { ... }`
/// - `dep_name.workspace = true`
///
/// Uses line-by-line matching with word-boundary checks to avoid false
/// positives where the banned name appears as a suffix of a longer crate
/// name (e.g. `oxiarc-lz4` must not match the `lz4` rule), or inside a
/// feature list string (e.g. `features = ["lz4"]`).
pub(crate) fn is_direct_dep(cargo_toml_content: &str, dep_name: &str) -> bool {
    for line in cargo_toml_content.lines() {
        let trimmed = line.trim();
        // Skip comments and empty lines
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        // Bare-identifier form: `lz4 = "..."` or `lz4 = {` or `lz4.workspace = true`
        // The character immediately following `dep_name` must be ' ', '=', or '.'
        // (but NOT '-' or any alphanumeric, which would indicate a longer name like
        // `lz4-something` or `lz4_binding`).
        if let Some(rest) = trimmed.strip_prefix(dep_name) {
            let next = rest.chars().next();
            match next {
                // `lz4 = ...` or `lz4= ...`
                Some(' ') | Some('=') => return true,
                // `lz4.workspace = true`
                Some('.') => return true,
                // anything else (e.g. '-', '_', alphanumeric) means it's a longer name
                _ => {}
            }
        }
        // Quoted form: `"lz4" = ...`
        let quoted = format!("\"{}\"", dep_name);
        if let Some(rest) = trimmed.strip_prefix(&quoted) {
            let next = rest.chars().next();
            if matches!(next, Some(' ') | Some('=')) {
                return true;
            }
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
    use std::fs;

    fn temp_dir(suffix: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "policy_banned_{}_{}_{}",
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
    fn test_is_direct_dep_simple_string() {
        assert!(is_direct_dep("zip = \"2.0\"", "zip"));
        assert!(!is_direct_dep("oxiarc-archive = \"1.0\"", "zip"));
    }

    #[test]
    fn test_is_direct_dep_table_form() {
        assert!(is_direct_dep(
            "flate2 = { version = \"1.0\", features = [\"zlib\"] }",
            "flate2"
        ));
    }

    #[test]
    fn test_is_direct_dep_workspace_form() {
        assert!(is_direct_dep("bincode.workspace = true", "bincode"));
    }

    #[test]
    fn test_is_direct_dep_no_partial_match() {
        // "zipfile" should not match the "zip" rule
        assert!(!is_direct_dep("zipfile = \"1.0\"", "zip"));
    }

    #[test]
    fn test_rule_finds_zip_violation() {
        let dir = temp_dir("zip");
        // Place the violating Cargo.toml in a sub-crate dir (not the workspace root)
        let crate_dir = dir.join("my-crate");
        fs::create_dir_all(&crate_dir).expect("create crate dir");
        fs::write(
            crate_dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.1.0\"\n\n[dependencies]\nzip = \"2.0\"\n",
        )
        .expect("write Cargo.toml");

        let rule = BannedDirectDepRule;
        let violations = rule.check(&dir);
        assert!(!violations.is_empty(), "Should detect zip violation");
        assert!(violations[0].message.contains("zip"));
        assert_eq!(violations[0].severity, Severity::Error);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rule_finds_bincode_violation() {
        let dir = temp_dir("bincode");
        // Place the violating Cargo.toml in a sub-crate dir (not the workspace root)
        let crate_dir = dir.join("my-crate");
        fs::create_dir_all(&crate_dir).expect("create crate dir");
        fs::write(
            crate_dir.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nbincode = \"1.3\"\n",
        )
        .expect("write Cargo.toml");

        let rule = BannedDirectDepRule;
        let violations = rule.check(&dir);
        assert!(
            violations.iter().any(|v| v.message.contains("bincode")),
            "Should detect bincode violation"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rule_clean_workspace_no_violations() {
        let dir = temp_dir("clean");
        let cargo_toml = dir.join("Cargo.toml");
        fs::write(
            &cargo_toml,
            "[package]\nname = \"test-clean\"\nversion = \"0.1.0\"\n\n[dependencies]\noxiarc-archive = \"1.0\"\n",
        )
        .expect("write Cargo.toml");

        let rule = BannedDirectDepRule;
        let violations = rule.check(&dir);
        assert!(
            violations.is_empty(),
            "Clean workspace should have no violations"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_is_direct_dep_no_false_positive_for_oxiarc_prefix() {
        // "oxiarc-lz4 = ..." should NOT match "lz4"
        assert!(!is_direct_dep("oxiarc-lz4 = { workspace = true }", "lz4"));
        assert!(!is_direct_dep("oxiarc-zstd = { workspace = true }", "zstd"));
        assert!(!is_direct_dep(
            "oxiarc-bzip2 = { workspace = true }",
            "bzip2"
        ));
        assert!(!is_direct_dep(
            "oxiarc-brotli = { workspace = true }",
            "brotli"
        ));
    }

    #[test]
    fn test_is_direct_dep_no_false_positive_for_feature_strings() {
        // parquet features = ["brotli", "lz4"] should NOT match
        let content = r#"parquet = { version = "58", features = ["brotli", "lz4", "snap"] }"#;
        assert!(!is_direct_dep(content, "lz4"));
        assert!(!is_direct_dep(content, "brotli"));
    }

    #[test]
    fn test_is_direct_dep_detects_actual_direct_dep() {
        assert!(is_direct_dep("lz4 = { version = \"1.24\" }", "lz4"));
        assert!(is_direct_dep("zstd = \"0.13\"", "zstd"));
        assert!(is_direct_dep("bzip2 = { workspace = true }", "bzip2"));
    }

    #[test]
    fn test_rule_skips_workspace_root() {
        // Root Cargo.toml listing banned deps in [workspace.dependencies] should not
        // trigger violations (it uses them only as reference / feature-gated).
        let dir = temp_dir("root_skip");
        let cargo_toml = dir.join("Cargo.toml");
        fs::write(
            &cargo_toml,
            "[workspace]\nmembers = []\n\n[workspace.dependencies]\nzip = { version = \"2.0\" }\n",
        )
        .expect("write Cargo.toml");

        let rule = BannedDirectDepRule;
        let violations = rule.check(&dir);
        // The root is skipped, so no violations expected
        assert!(
            violations.is_empty(),
            "Root Cargo.toml should be skipped; got: {:?}",
            violations
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
