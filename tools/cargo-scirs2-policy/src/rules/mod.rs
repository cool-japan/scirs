//! Policy rules for SciRS2 workspace compliance.
//!
//! Each rule implements [`PolicyRule`] and is registered in [`all_rules`].
//! Call [`check_workspace`] to run every registered rule and collect
//! [`Violation`] instances.

use std::path::Path;

pub mod banned_deps;
pub mod lock_analysis;
pub mod source_scan;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A policy violation found in the workspace.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Rule identifier (e.g., `"BANNED_DEP_001"`).
    pub rule_id: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// File where the violation was found, if applicable.
    pub file: Option<String>,
    /// Severity level.
    pub severity: Severity,
}

/// Severity of a policy violation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// Must be fixed before release.
    Error,
    /// Should be fixed but is not release-blocking.
    Warning,
    /// Informational note only.
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            Severity::Error => write!(f, "ERROR"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Info => write!(f, "INFO"),
            // Safety: #[non_exhaustive] — future variants handled gracefully
            _ => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// PolicyRule trait
// ---------------------------------------------------------------------------

/// A single compliance policy rule.
pub trait PolicyRule: Send + Sync {
    /// Unique identifier for this rule (e.g., `"BANNED_DEP_001"`).
    fn id(&self) -> &'static str;

    /// Human-readable description of what this rule checks.
    fn description(&self) -> &'static str;

    /// Run this rule against the given workspace root and return any violations.
    fn check(&self, workspace: &Path) -> Vec<Violation>;
}

// ---------------------------------------------------------------------------
// Rule registry
// ---------------------------------------------------------------------------

/// Returns all registered policy rules in check order.
pub fn all_rules() -> Vec<Box<dyn PolicyRule>> {
    vec![
        Box::new(banned_deps::BannedDirectDepRule),
        Box::new(source_scan::DirectRandUsageRule),
        Box::new(source_scan::DirectNdarrayUsageRule),
    ]
}

/// Run all registered rules against the workspace root and aggregate violations.
pub fn check_workspace(workspace: &Path) -> Vec<Violation> {
    all_rules()
        .into_iter()
        .flat_map(|rule| rule.check(workspace))
        .collect()
}

// ---------------------------------------------------------------------------
// Duplicate-version analysis (Cargo.lock)
// ---------------------------------------------------------------------------

/// Find packages that appear with multiple versions in `Cargo.lock`.
///
/// Returns a sorted list of `(package_name, [version, ...])` pairs where
/// `versions.len() > 1`.  The list is sorted alphabetically by package name.
///
/// This is informational: having multiple versions of a crate is not
/// necessarily a violation but may indicate dependency bloat.
pub fn find_duplicate_deps(lock_path: &Path) -> Vec<(String, Vec<String>)> {
    let content = match std::fs::read_to_string(lock_path) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Warning: could not read {}: {}", lock_path.display(), err);
            return Vec::new();
        }
    };

    let value: toml::Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Warning: failed to parse {}: {}", lock_path.display(), err);
            return Vec::new();
        }
    };

    let mut version_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    if let Some(packages) = value.get("package").and_then(|p| p.as_array()) {
        for pkg in packages {
            if let (Some(name), Some(version)) = (
                pkg.get("name").and_then(|n| n.as_str()),
                pkg.get("version").and_then(|v| v.as_str()),
            ) {
                version_map
                    .entry(name.to_string())
                    .or_default()
                    .push(version.to_string());
            }
        }
    }

    let mut duplicates: Vec<(String, Vec<String>)> = version_map
        .into_iter()
        .filter(|(_, versions)| versions.len() > 1)
        .collect();

    // Sort both the outer list and each version list for deterministic output
    for (_, versions) in &mut duplicates {
        versions.sort();
    }
    duplicates.sort_by(|a, b| a.0.cmp(&b.0));
    duplicates
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
            "policy_rules_{}_{}_{}",
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
    fn test_severity_display() {
        assert_eq!(Severity::Error.to_string(), "ERROR");
        assert_eq!(Severity::Warning.to_string(), "WARN");
        assert_eq!(Severity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_all_rules_returns_three_rules() {
        let rules = all_rules();
        assert_eq!(rules.len(), 3);
        let ids: Vec<&str> = rules.iter().map(|r| r.id()).collect();
        assert!(ids.contains(&"BANNED_DEP_001"));
        assert!(ids.contains(&"SOURCE_SCAN_001"));
        assert!(ids.contains(&"SOURCE_SCAN_002"));
    }

    #[test]
    fn test_find_duplicate_deps_parses_lock() {
        let dir = temp_dir("lock");
        let lock = dir.join("Cargo.lock");
        // Minimal Cargo.lock v3 with one dup (serde 1.0.100 and 1.0.200)
        fs::write(
            &lock,
            r#"version = 3

[[package]]
name = "serde"
version = "1.0.100"
source = "registry+..."
checksum = "aaa"

[[package]]
name = "serde"
version = "1.0.200"
source = "registry+..."
checksum = "bbb"

[[package]]
name = "tokio"
version = "1.0.0"
source = "registry+..."
checksum = "ccc"
"#,
        )
        .expect("write Cargo.lock");

        let dups = find_duplicate_deps(&lock);
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].0, "serde");
        assert_eq!(dups[0].1, vec!["1.0.100", "1.0.200"]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_duplicate_deps_no_dups() {
        let dir = temp_dir("no_dups");
        let lock = dir.join("Cargo.lock");
        fs::write(
            &lock,
            r#"version = 3

[[package]]
name = "serde"
version = "1.0.100"
source = "registry+..."
checksum = "aaa"
"#,
        )
        .expect("write Cargo.lock");

        let dups = find_duplicate_deps(&lock);
        assert!(dups.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_duplicate_deps_missing_file() {
        let path = std::path::Path::new("/nonexistent/Cargo.lock");
        let dups = find_duplicate_deps(path);
        assert!(dups.is_empty());
    }
}
