//! Dependency count auditing for cargo-scirs2-policy.
//!
//! This module provides utilities for auditing the dependency footprint of the
//! SciRS2 workspace:
//!
//! - Counting unique packages in `Cargo.lock` (total transitive closure)
//! - Counting direct workspace dependencies in `Cargo.toml`
//! - Flagging banned dependencies that should have been removed
//! - Producing a progress report against an optional baseline count
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use cargo_scirs2_policy::dep_audit::run_dep_audit;
//!
//! let result = run_dep_audit(Path::new("/path/to/workspace"), Some(850));
//! println!("{}", result.summary);
//! ```

use std::collections::HashSet;
use std::path::Path;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// The result of a dependency audit run.
#[derive(Debug)]
pub struct DepAuditResult {
    /// Number of direct dependencies declared in `[workspace.dependencies]`.
    pub total_direct_deps: usize,
    /// Number of unique package names in `Cargo.lock`.
    pub unique_package_names: usize,
    /// Banned dependency names found in `Cargo.lock`.
    pub flagged_banned: Vec<String>,
    /// Human-readable summary of the audit results.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Banned dependency list
// ---------------------------------------------------------------------------

/// Packages that must not appear in `Cargo.lock` per COOLJAPAN policy.
const BANNED_DEPS: &[&str] = &[
    "openblas-src",
    "bincode",
    "flate2",
    "zip",
    "zstd",
    "bzip2",
    "lz4",
    "snap",
    "brotli",
    "miniz_oxide",
    "rustfft",
    "z3",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse `Cargo.lock` and return the number of unique package names.
///
/// The count is based on `name = "..."` lines inside `[[package]]` sections.
/// Each distinct package name is counted once regardless of how many versions
/// appear.
///
/// # Errors
///
/// Returns an error string if `Cargo.lock` cannot be read.
pub fn count_unique_packages(cargo_lock_path: &Path) -> Result<usize, String> {
    let content = std::fs::read_to_string(cargo_lock_path)
        .map_err(|e| format!("Failed to read {}: {e}", cargo_lock_path.display()))?;

    let unique: HashSet<&str> = content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("name = \"") && trimmed.ends_with('"') {
                // Extract the name between the first and last quote
                let inner = trimmed.trim_start_matches("name = \"");
                Some(inner.trim_end_matches('"'))
            } else {
                None
            }
        })
        .collect();

    Ok(unique.len())
}

/// Count all `[[package]]` entries (including duplicates with different versions)
/// in `Cargo.lock`.
///
/// This is the total number of resolved packages, useful for tracking bloat over
/// time even when the same package appears in two versions.
///
/// # Errors
///
/// Returns an error string if the file cannot be read.
pub fn count_total_package_entries(cargo_lock_path: &Path) -> Result<usize, String> {
    let content = std::fs::read_to_string(cargo_lock_path)
        .map_err(|e| format!("Failed to read {}: {e}", cargo_lock_path.display()))?;

    let count = content
        .lines()
        .filter(|l| l.trim() == "[[package]]")
        .count();

    Ok(count)
}

/// Count direct workspace dependencies declared in `[workspace.dependencies]`
/// in the workspace root `Cargo.toml`.
///
/// Only non-empty, non-comment lines between the `[workspace.dependencies]`
/// header and the next `[...]` section are counted.
///
/// # Errors
///
/// Returns an error string if the file cannot be read.
pub fn count_workspace_direct_deps(workspace_cargo_toml: &Path) -> Result<usize, String> {
    let content = std::fs::read_to_string(workspace_cargo_toml)
        .map_err(|e| format!("Failed to read {}: {e}", workspace_cargo_toml.display()))?;

    let mut in_workspace_deps = false;
    let mut count = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.dependencies]" {
            in_workspace_deps = true;
            continue;
        }
        if in_workspace_deps {
            // Another section begins
            if trimmed.starts_with('[') {
                break;
            }
            // Skip blanks and comments
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                count += 1;
            }
        }
    }

    Ok(count)
}

/// Check which banned dependencies appear in `Cargo.lock`.
///
/// Returns the names of any banned packages found (in alphabetical order).
pub fn find_banned_in_lock(cargo_lock_path: &Path) -> Vec<String> {
    let content = match std::fs::read_to_string(cargo_lock_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut flagged: Vec<String> = BANNED_DEPS
        .iter()
        .filter(|dep| content.contains(&format!("name = \"{dep}\"")))
        .map(|dep| dep.to_string())
        .collect();

    flagged.sort();
    flagged
}

/// Run a complete dependency audit against a workspace root.
///
/// # Parameters
///
/// - `workspace_root`: path to the Cargo workspace root (must contain
///   `Cargo.toml` and `Cargo.lock`).
/// - `baseline_count`: optional previous unique-package count to compute
///   progress against.
///
/// # Returns
///
/// A [`DepAuditResult`] with all counts and a human-readable summary.
pub fn run_dep_audit(workspace_root: &Path, baseline_count: Option<usize>) -> DepAuditResult {
    let lock_path = workspace_root.join("Cargo.lock");
    let cargo_path = workspace_root.join("Cargo.toml");

    let unique_packages = count_unique_packages(&lock_path).unwrap_or(0);
    let total_entries = count_total_package_entries(&lock_path).unwrap_or(0);
    let direct_deps = count_workspace_direct_deps(&cargo_path).unwrap_or(0);
    let flagged = find_banned_in_lock(&lock_path);

    let progress_line = match baseline_count {
        Some(baseline) => {
            if unique_packages < baseline {
                format!(
                    "-{} packages ({:.1}% reduction vs baseline of {})",
                    baseline - unique_packages,
                    100.0 * (baseline - unique_packages) as f64 / baseline as f64,
                    baseline
                )
            } else if unique_packages > baseline {
                format!(
                    "+{} packages ({:.1}% increase vs baseline of {})",
                    unique_packages - baseline,
                    100.0 * (unique_packages - baseline) as f64 / baseline as f64,
                    baseline
                )
            } else {
                format!("unchanged vs baseline of {baseline}")
            }
        }
        None => String::from("no baseline provided"),
    };

    let banned_line = if flagged.is_empty() {
        String::from("none")
    } else {
        flagged.join(", ")
    };

    let summary = format!(
        "Unique package names:       {unique_packages}\n\
         Total package entries:      {total_entries}\n\
         Direct workspace deps:      {direct_deps}\n\
         Progress:                   {progress_line}\n\
         Flagged banned deps:        {banned_line}"
    );

    DepAuditResult {
        total_direct_deps: direct_deps,
        unique_package_names: unique_packages,
        flagged_banned: flagged,
        summary,
    }
}

/// Produce a verbose text audit report suitable for display on stdout.
///
/// Includes a banner, per-section breakdowns, and pass/fail indicators.
pub fn format_audit_report(result: &DepAuditResult) -> String {
    let mut out = String::new();

    out.push_str("=== SciRS2 Dependency Audit ===\n\n");

    out.push_str(&format!(
        "  Unique package names:    {}\n",
        result.unique_package_names
    ));
    out.push_str(&format!(
        "  Direct workspace deps:   {}\n",
        result.total_direct_deps
    ));

    out.push('\n');

    if result.flagged_banned.is_empty() {
        out.push_str("  [PASS] No banned dependencies found.\n");
    } else {
        out.push_str("  [FAIL] Banned dependencies present:\n");
        for dep in &result.flagged_banned {
            out.push_str(&format!("         - {dep}\n"));
        }
    }

    out.push('\n');
    out.push_str("--- Summary ---\n");
    out.push_str(&result.summary);
    out.push('\n');

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "dep_audit_{}_{}_{}",
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

    const SAMPLE_LOCK: &str = r#"version = 3

[[package]]
name = "serde"
version = "1.0.100"
source = "registry+..."

[[package]]
name = "serde"
version = "1.0.200"
source = "registry+..."

[[package]]
name = "tokio"
version = "1.0.0"
source = "registry+..."

[[package]]
name = "bincode"
version = "1.3.3"
source = "registry+..."
"#;

    const SAMPLE_CARGO_TOML: &str = r#"[workspace]
members = ["crate-a", "crate-b"]

[workspace.dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
# comment line
clap = { version = "4", features = ["derive"] }

[workspace.metadata]
foo = "bar"
"#;

    #[test]
    fn test_count_unique_packages() {
        let dir = temp_dir("unique");
        let lock = dir.join("Cargo.lock");
        fs::write(&lock, SAMPLE_LOCK).expect("write");

        let count = count_unique_packages(&lock).expect("count");
        // serde (2 versions = 1 unique), tokio (1), bincode (1) = 3 unique names
        assert_eq!(count, 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_count_total_package_entries() {
        let dir = temp_dir("total");
        let lock = dir.join("Cargo.lock");
        fs::write(&lock, SAMPLE_LOCK).expect("write");

        let count = count_total_package_entries(&lock).expect("count");
        assert_eq!(count, 4); // 2 serde + 1 tokio + 1 bincode

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_count_workspace_direct_deps() {
        let dir = temp_dir("direct");
        let toml = dir.join("Cargo.toml");
        fs::write(&toml, SAMPLE_CARGO_TOML).expect("write");

        let count = count_workspace_direct_deps(&toml).expect("count");
        assert_eq!(count, 3); // serde, tokio, clap (comment excluded)

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_banned_in_lock_detects_bincode() {
        let dir = temp_dir("banned");
        let lock = dir.join("Cargo.lock");
        fs::write(&lock, SAMPLE_LOCK).expect("write");

        let flagged = find_banned_in_lock(&lock);
        assert!(flagged.contains(&"bincode".to_string()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_banned_in_lock_clean() {
        let dir = temp_dir("clean");
        let lock = dir.join("Cargo.lock");
        fs::write(
            &lock,
            "version = 3\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n",
        )
        .expect("write");

        let flagged = find_banned_in_lock(&lock);
        assert!(flagged.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_full() {
        let dir = temp_dir("full_audit");
        let lock = dir.join("Cargo.lock");
        let cargo = dir.join("Cargo.toml");
        fs::write(&lock, SAMPLE_LOCK).expect("write lock");
        fs::write(&cargo, SAMPLE_CARGO_TOML).expect("write toml");

        let result = run_dep_audit(&dir, Some(10));
        assert_eq!(result.unique_package_names, 3);
        assert_eq!(result.total_direct_deps, 3);
        assert!(result.flagged_banned.contains(&"bincode".to_string()));
        assert!(result.summary.contains("Unique package names"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_baseline_improvement() {
        let dir = temp_dir("baseline_imp");
        let lock = dir.join("Cargo.lock");
        let cargo = dir.join("Cargo.toml");
        fs::write(&lock, SAMPLE_LOCK).expect("write lock");
        fs::write(&cargo, SAMPLE_CARGO_TOML).expect("write toml");

        let result = run_dep_audit(&dir, Some(6)); // 3 < 6 baseline
        assert!(result.summary.contains("reduction"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_baseline_regression() {
        let dir = temp_dir("baseline_reg");
        let lock = dir.join("Cargo.lock");
        let cargo = dir.join("Cargo.toml");
        fs::write(&lock, SAMPLE_LOCK).expect("write lock");
        fs::write(&cargo, SAMPLE_CARGO_TOML).expect("write toml");

        let result = run_dep_audit(&dir, Some(2)); // 3 > 2 baseline
        assert!(result.summary.contains("increase"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_no_baseline() {
        let dir = temp_dir("no_baseline");
        let lock = dir.join("Cargo.lock");
        let cargo = dir.join("Cargo.toml");
        fs::write(&lock, SAMPLE_LOCK).expect("write lock");
        fs::write(&cargo, SAMPLE_CARGO_TOML).expect("write toml");

        let result = run_dep_audit(&dir, None);
        assert!(result.summary.contains("no baseline"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_format_audit_report_pass() {
        let result = DepAuditResult {
            total_direct_deps: 50,
            unique_package_names: 200,
            flagged_banned: Vec::new(),
            summary: "summary".to_string(),
        };
        let report = format_audit_report(&result);
        assert!(report.contains("PASS"));
        assert!(!report.contains("FAIL"));
    }

    #[test]
    fn test_format_audit_report_fail() {
        let result = DepAuditResult {
            total_direct_deps: 50,
            unique_package_names: 200,
            flagged_banned: vec!["bincode".to_string()],
            summary: "summary".to_string(),
        };
        let report = format_audit_report(&result);
        assert!(report.contains("FAIL"));
        assert!(report.contains("bincode"));
    }

    #[test]
    fn test_count_unique_packages_missing_file() {
        let result = count_unique_packages(Path::new("/nonexistent/Cargo.lock"));
        assert!(result.is_err());
    }

    #[test]
    fn test_count_workspace_direct_deps_missing_file() {
        let result = count_workspace_direct_deps(Path::new("/nonexistent/Cargo.toml"));
        assert!(result.is_err());
    }
}
