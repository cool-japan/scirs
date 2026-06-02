//! Check `Cargo.toml` files for banned direct dependencies.
//!
//! ## Banned packages (COOLJAPAN Pure Rust Policy)
//!
//! | Package | Replacement |
//! |---------|-------------|
//! | `zip` | `oxiarc-archive` |
//! | `flate2` | `oxiarc-deflate` / `oxiarc-*` |
//! | `zstd` | `oxiarc-zstd` |
//! | `bzip2` | `oxiarc-bzip2` |
//! | `lz4` | `oxiarc-lz4` |
//! | `snap` | `oxiarc-snappy` |
//! | `brotli` | `oxiarc-brotli` |
//! | `miniz_oxide` | `oxiarc-deflate` |
//! | `bincode` | `oxicode` |
//! | `openblas-src` | `oxiblas` |
//! | `blas-src` | `oxiblas` |
//! | `cblas` | `oxiblas` |
//! | `lapack-src` | `oxiblas` |
//! | `rustfft` | `OxiFFT` |
//! | `z3` | `OxiZ` |
//! | `rand` | `scirs2-core` (in non-core crates) |
//!
//! A package is considered **allowed** if it appears only as an `optional = true`
//! dependency AND that optional dependency is NOT included in `default-features`
//! (i.e., it is feature-gated behind a non-default feature).  However, due to
//! the complexity of full TOML analysis, the current implementation flags
//! `optional = true` entries with a **Warning** rather than an **Error**, so
//! that teams can evaluate them individually.

use crate::violation::{PolicyViolation, Severity};
use crate::workspace::WorkspaceInfo;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Banned dependency table
// ---------------------------------------------------------------------------

/// `(crate_name, replacement_hint, severity_when_optional)`
const BANNED: &[(&str, &str, Severity)] = &[
    ("zip", "use oxiarc-archive instead", Severity::Error),
    (
        "flate2",
        "use oxiarc-deflate/oxiarc-* instead",
        Severity::Error,
    ),
    ("zstd", "use oxiarc-zstd instead", Severity::Error),
    ("bzip2", "use oxiarc-bzip2 instead", Severity::Error),
    ("lz4", "use oxiarc-lz4 instead", Severity::Error),
    ("snap", "use oxiarc-snappy instead", Severity::Error),
    ("brotli", "use oxiarc-brotli instead", Severity::Error),
    ("miniz_oxide", "use oxiarc-deflate instead", Severity::Error),
    ("bincode", "use oxicode instead", Severity::Error),
    ("openblas-src", "use oxiblas instead", Severity::Error),
    ("blas-src", "use oxiblas instead", Severity::Error),
    ("cblas", "use oxiblas instead", Severity::Error),
    ("lapack-src", "use oxiblas instead", Severity::Error),
    ("rustfft", "use OxiFFT instead", Severity::Error),
    ("z3", "use OxiZ instead", Severity::Error),
];

/// Packages that are banned in non-core crates only.
const NON_CORE_BANNED: &[(&str, &str)] = &[("rand", "use scirs2-core random utilities instead")];

// ---------------------------------------------------------------------------
// Check struct
// ---------------------------------------------------------------------------

/// Check `Cargo.toml` files for banned direct dependencies.
pub struct BannedDepCheck;

impl BannedDepCheck {
    /// Run the check against the given workspace and return all violations.
    pub fn run(&self, workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
        let mut violations = Vec::new();

        for crate_info in &workspace.crates {
            let cargo_toml = crate_info.path.join("Cargo.toml");
            if !cargo_toml.exists() {
                continue;
            }
            let content = match std::fs::read_to_string(&cargo_toml) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Check globally banned dependencies
            for (dep_name, hint, severity) in BANNED {
                if let Some((line_num, optional)) = find_dep(&content, dep_name) {
                    let effective_severity = if optional {
                        // Optional deps are warnings rather than errors —
                        // they may be legitimately feature-gated.
                        Severity::Warning
                    } else {
                        severity.clone()
                    };
                    violations.push(PolicyViolation {
                        crate_name: crate_info.name.clone(),
                        file: cargo_toml.clone(),
                        line: line_num,
                        message: format!(
                            "banned dependency '{}': {} (COOLJAPAN Pure Rust Policy){}",
                            dep_name,
                            hint,
                            if optional {
                                " [optional — verify feature gate]"
                            } else {
                                ""
                            },
                        ),
                        severity: effective_severity,
                    });
                }
            }

            // Check non-core banned dependencies
            if !crate_info.is_core {
                for (dep_name, hint) in NON_CORE_BANNED {
                    if let Some((line_num, optional)) = find_dep(&content, dep_name) {
                        let severity = Severity::Warning;
                        violations.push(PolicyViolation {
                            crate_name: crate_info.name.clone(),
                            file: cargo_toml.clone(),
                            line: line_num,
                            message: format!(
                                "banned dep '{}' in non-core crate: {}{}",
                                dep_name,
                                hint,
                                if optional { " [optional]" } else { "" },
                            ),
                            severity,
                        });
                    }
                }
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// TOML parsing helpers
// ---------------------------------------------------------------------------

/// Search `content` (a `Cargo.toml` file) for a dependency named `dep_name`.
///
/// Returns `Some((line_number, is_optional))` if found as a direct dependency
/// in `[dependencies]`, `[dev-dependencies]`, or `[build-dependencies]`
/// sections.  Returns `None` if not present.
///
/// `line_number` is 1-based.  `is_optional` is `true` when the entry includes
/// `optional = true`.
pub fn find_dep(content: &str, dep_name: &str) -> Option<(usize, bool)> {
    let mut in_deps_section = false;

    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();

        // Track which TOML section we are in
        if trimmed.starts_with('[') {
            in_deps_section = is_deps_section_header(trimmed);
            continue;
        }

        if !in_deps_section {
            continue;
        }

        // Skip comments and empty lines
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        // Check if this line declares `dep_name`
        if line_declares_dep(trimmed, dep_name) {
            let optional = trimmed.contains("optional = true") || trimmed.contains("optional=true");
            return Some((line_num, optional));
        }
    }
    None
}

/// Returns `true` if `header` is a `[dependencies]`-style section.
fn is_deps_section_header(header: &str) -> bool {
    let h = header.trim_matches(|c| c == '[' || c == ']');
    matches!(
        h,
        "dependencies" | "dev-dependencies" | "build-dependencies"
    )
}

/// Returns `true` if `line` (trimmed) declares `dep_name` as a dependency key.
///
/// Handles:
/// - `dep_name = "version"`
/// - `dep_name = { version = "..." }`
/// - `dep_name.workspace = true`
/// - `"dep_name" = ...`
///
/// Does NOT match longer names (e.g., `oxiarc-lz4` won't match `lz4`).
pub fn line_declares_dep(trimmed: &str, dep_name: &str) -> bool {
    // Bare form: `dep_name = ...` or `dep_name.xxx = ...`
    if let Some(rest) = trimmed.strip_prefix(dep_name) {
        let next = rest.chars().next();
        if matches!(next, Some(' ') | Some('=') | Some('.')) {
            return true;
        }
    }
    // Quoted form: `"dep_name" = ...`
    let quoted = format!("\"{}\"", dep_name);
    if let Some(rest) = trimmed.strip_prefix(&quoted) {
        let next = rest.chars().next();
        if matches!(next, Some(' ') | Some('=')) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// PathBuf helper (used in tests below)
// ---------------------------------------------------------------------------

fn _path(p: &str) -> PathBuf {
    PathBuf::from(p)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{CrateInfo, WorkspaceInfo};
    use std::fs;
    use std::path::Path;

    fn temp_dir(suffix: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "bd_{}_{}_{}",
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

    fn workspace_with_cargo_toml(
        dir: &Path,
        content: &str,
        name: &str,
        is_core: bool,
    ) -> WorkspaceInfo {
        fs::write(dir.join("Cargo.toml"), content).expect("write Cargo.toml");
        WorkspaceInfo {
            root: dir.parent().unwrap_or(dir).to_path_buf(),
            crates: vec![CrateInfo {
                name: name.to_string(),
                path: dir.to_path_buf(),
                is_core,
            }],
        }
    }

    #[test]
    fn test_detect_banned_dep_flate2() {
        let dir = temp_dir("flate2");
        let ws = workspace_with_cargo_toml(
            &dir,
            "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n[dependencies]\nflate2 = \"1.0\"\n",
            "my-crate",
            false,
        );
        let violations = BannedDepCheck.run(&ws);
        let flate2_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("flate2"))
            .collect();
        assert!(!flate2_v.is_empty(), "Should detect flate2");
        assert_eq!(
            flate2_v[0].severity,
            Severity::Error,
            "Non-optional flate2 is Error"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_allowed_optional_dep_is_warning() {
        let dir = temp_dir("optional");
        let ws = workspace_with_cargo_toml(
            &dir,
            "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n[dependencies]\nflate2 = { version = \"1.0\", optional = true }\n",
            "my-crate",
            false,
        );
        let violations = BannedDepCheck.run(&ws);
        let flate2_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("flate2"))
            .collect();
        assert!(!flate2_v.is_empty(), "Optional flate2 still flagged");
        assert_eq!(
            flate2_v[0].severity,
            Severity::Warning,
            "Optional flate2 should be Warning (not Error) — verify feature gate"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_clean_cargo_toml_no_violations() {
        let dir = temp_dir("clean_deps");
        let ws = workspace_with_cargo_toml(
            &dir,
            "[package]\nname = \"clean\"\nversion = \"0.1.0\"\n\n[dependencies]\noxiarc-archive = \"1.0\"\noxiblas = \"0.1\"\n",
            "clean",
            false,
        );
        let violations = BannedDepCheck.run(&ws);
        assert!(
            violations.is_empty(),
            "Clean Cargo.toml should have no violations"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rand_in_non_core_is_warning() {
        let dir = temp_dir("rand_dep");
        let ws = workspace_with_cargo_toml(
            &dir,
            "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n[dependencies]\nrand = \"0.8\"\n",
            "my-crate",
            false,
        );
        let violations = BannedDepCheck.run(&ws);
        let rand_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("rand"))
            .collect();
        assert!(!rand_v.is_empty(), "rand in non-core should be flagged");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rand_in_core_is_not_flagged() {
        let dir = temp_dir("rand_core");
        let ws = workspace_with_cargo_toml(
            &dir,
            "[package]\nname = \"scirs2-core\"\nversion = \"0.1.0\"\n\n[dependencies]\nrand = \"0.8\"\n",
            "scirs2-core",
            true,
        );
        let violations = BannedDepCheck.run(&ws);
        let rand_v: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("'rand'"))
            .collect();
        assert!(
            rand_v.is_empty(),
            "rand in scirs2-core should NOT be flagged"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_line_declares_dep_no_false_positive() {
        // "lz4-something" should not match "lz4"
        assert!(!line_declares_dep("lz4-sys = \"1.0\"", "lz4"));
        // "oxiarc-lz4 = ..." should not match "lz4"
        assert!(!line_declares_dep(
            "oxiarc-lz4 = { workspace = true }",
            "lz4"
        ));
        // Feature list strings should not match
        assert!(!line_declares_dep("features = [\"lz4\", \"snap\"]", "lz4"));
    }

    #[test]
    fn test_line_declares_dep_positive_cases() {
        assert!(line_declares_dep("lz4 = \"1.24\"", "lz4"));
        assert!(line_declares_dep("lz4 = { version = \"1.24\" }", "lz4"));
        assert!(line_declares_dep("lz4.workspace = true", "lz4"));
        assert!(line_declares_dep("\"lz4\" = \"1.24\"", "lz4"));
        assert!(line_declares_dep("zip = \"2.0\"", "zip"));
        assert!(line_declares_dep(
            "bincode = { version = \"1.3\", optional = true }",
            "bincode"
        ));
    }

    #[test]
    fn test_find_dep_only_matches_deps_section() {
        // A dep that appears only in [package] should not be found
        let content = "[package]\nname = \"zip\"\nversion = \"0.1.0\"\n\n[dependencies]\noxiarc-archive = \"1.0\"\n";
        assert!(
            find_dep(content, "zip").is_none(),
            "Package name 'zip' in [package] section should not match as a dependency"
        );
    }

    #[test]
    fn test_find_dep_detects_in_dev_dependencies() {
        let content = "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\nzip = \"2.0\"\n";
        let result = find_dep(content, "zip");
        assert!(result.is_some(), "Should find zip in [dev-dependencies]");
    }
}
