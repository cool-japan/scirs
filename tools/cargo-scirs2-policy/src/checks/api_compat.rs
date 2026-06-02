//! Public API surface tracking and compatibility checks.
//!
//! This module provides tools to snapshot the public API surface of a workspace
//! and compare it against a previous snapshot to detect backward-incompatible
//! removals.
//!
//! ## Rules
//!
//! | Rule ID | Severity | Description |
//! |---------|----------|-------------|
//! | `API_COMPAT_001` | ERROR | Public item removed without prior `#[deprecated]` |
//! | `API_COMPAT_002` | WARNING | Public item removed (was deprecated — ok if policy allows) |
//! | `API_COMPAT_003` | INFO | New public item added (informational) |

use crate::violation::{PolicyViolation, Severity};
use crate::workspace::{self, WorkspaceInfo};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public check type
// ---------------------------------------------------------------------------

/// Linter check that compares the current public API surface against a saved
/// snapshot and flags breaking changes.
pub struct ApiCompatCheck;

impl ApiCompatCheck {
    /// Run the API compatibility check.
    ///
    /// If the snapshot file at `snapshot_path` does not exist, this check is
    /// a no-op (returns empty violations).
    pub fn run(&self, workspace: &WorkspaceInfo, snapshot_path: &Path) -> Vec<PolicyViolation> {
        if !snapshot_path.exists() {
            return Vec::new();
        }
        check_api_compatibility(workspace, snapshot_path).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// API item representation
// ---------------------------------------------------------------------------

/// A single public API item (function, struct, enum, trait, type alias, etc.).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct ApiItem {
    /// The crate that owns this item.
    pub crate_name: String,
    /// Relative file path within the crate (e.g., `src/lib.rs`).
    pub file: String,
    /// 1-based line number where the item is declared.
    pub line: usize,
    /// The kind of item (e.g., `fn`, `struct`, `enum`, `trait`, `type`, `const`, `static`, `mod`).
    pub kind: String,
    /// The fully-qualified item name (e.g., `my_module::MyStruct`).
    pub name: String,
    /// Whether the item is currently marked `#[deprecated]`.
    pub is_deprecated: bool,
}

// ---------------------------------------------------------------------------
// API snapshot
// ---------------------------------------------------------------------------

/// A snapshot of the entire public API surface of a workspace.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ApiSnapshot {
    /// The version of the workspace at the time of snapshot.
    pub version: String,
    /// Timestamp (ISO 8601) when the snapshot was taken.
    pub timestamp: String,
    /// All public API items, sorted by (crate_name, file, name).
    pub items: Vec<ApiItem>,
}

impl ApiSnapshot {
    /// Returns the number of items in the snapshot.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the snapshot contains no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Save the snapshot to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialisation fails.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialise API snapshot: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write API snapshot to {}: {e}", path.display()))?;
        Ok(())
    }

    /// Load a snapshot from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or deserialisation fails.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read API snapshot from {}: {e}", path.display()))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse API snapshot: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Snapshot generation
// ---------------------------------------------------------------------------

/// Generate an API snapshot of the current workspace.
pub fn save_api_snapshot(workspace: &WorkspaceInfo, output_path: &Path) -> Result<(), String> {
    let items = collect_public_items(workspace);

    // Try to read the workspace version
    let version = read_workspace_version(&workspace.root).unwrap_or_else(|| "unknown".to_string());

    let snapshot = ApiSnapshot {
        version,
        timestamp: current_timestamp(),
        items,
    };

    snapshot.save(output_path)
}

/// Collect all public API items from the workspace.
pub fn collect_public_items(workspace: &WorkspaceInfo) -> Vec<ApiItem> {
    let mut items = Vec::new();

    for krate in &workspace.crates {
        let rs_files = workspace::walk_rust_files(&krate.path);

        for file in &rs_files {
            let content = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let relative = file
                .strip_prefix(&krate.path)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();

            let file_items = extract_public_items(&content, &krate.name, &relative);
            items.extend(file_items);
        }
    }

    items.sort();
    items
}

/// Extract public items from a single Rust source file.
///
/// Uses a simple line-based scanner (no syn dependency).
fn extract_public_items(content: &str, crate_name: &str, relative_file: &str) -> Vec<ApiItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut pending_deprecated = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Track #[deprecated] attributes
        if trimmed.starts_with("#[deprecated") {
            pending_deprecated = true;
            continue;
        }

        // Skip non-pub items
        if !trimmed.starts_with("pub ") && !trimmed.starts_with("pub(crate)") {
            // Reset deprecated flag on non-attribute, non-comment, non-empty lines
            if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("/*")
                && !trimmed.starts_with('*')
            {
                pending_deprecated = false;
            }
            continue;
        }

        // Skip pub(crate), pub(super), pub(in ...) — not truly public
        if trimmed.starts_with("pub(crate)")
            || trimmed.starts_with("pub(super)")
            || trimmed.starts_with("pub(in ")
        {
            pending_deprecated = false;
            continue;
        }

        // Extract the item kind and name
        if let Some((kind, name)) = parse_pub_item(trimmed) {
            items.push(ApiItem {
                crate_name: crate_name.to_string(),
                file: relative_file.to_string(),
                line: i + 1,
                kind,
                name,
                is_deprecated: pending_deprecated,
            });
        }

        pending_deprecated = false;
    }

    items
}

/// Parse a `pub ...` line to extract the item kind and name.
fn parse_pub_item(line: &str) -> Option<(String, String)> {
    // Remove `pub ` prefix
    let rest = line.strip_prefix("pub ")?;
    let rest = rest.trim_start();

    // Handle `pub async fn`
    let rest = rest.strip_prefix("async ").unwrap_or(rest);
    let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);

    let keywords: &[(&str, &str)] = &[
        ("fn ", "fn"),
        ("struct ", "struct"),
        ("enum ", "enum"),
        ("trait ", "trait"),
        ("type ", "type"),
        ("const ", "const"),
        ("static ", "static"),
        ("mod ", "mod"),
    ];

    for (prefix, kind) in keywords {
        if let Some(after) = rest.strip_prefix(prefix) {
            let name: String = after
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some((kind.to_string(), name));
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Compatibility check
// ---------------------------------------------------------------------------

/// Compare the current workspace API surface against a saved snapshot.
///
/// # Errors
///
/// Returns an error if the snapshot file cannot be loaded.
pub fn check_api_compatibility(
    workspace: &WorkspaceInfo,
    snapshot_path: &Path,
) -> Result<Vec<PolicyViolation>, String> {
    let old_snapshot = ApiSnapshot::load(snapshot_path)?;
    let current_items = collect_public_items(workspace);

    let mut violations = Vec::new();

    // Build lookup sets: (crate_name, kind, name)
    let old_set: BTreeSet<(&str, &str, &str)> = old_snapshot
        .items
        .iter()
        .map(|item| {
            (
                item.crate_name.as_str(),
                item.kind.as_str(),
                item.name.as_str(),
            )
        })
        .collect();

    let current_set: BTreeSet<(&str, &str, &str)> = current_items
        .iter()
        .map(|item| {
            (
                item.crate_name.as_str(),
                item.kind.as_str(),
                item.name.as_str(),
            )
        })
        .collect();

    // Build a map of old items for deprecated status lookup
    let old_deprecated: BTreeMap<(&str, &str, &str), bool> = old_snapshot
        .items
        .iter()
        .map(|item| {
            (
                (
                    item.crate_name.as_str(),
                    item.kind.as_str(),
                    item.name.as_str(),
                ),
                item.is_deprecated,
            )
        })
        .collect();

    // Find removed items (in old but not in current)
    for &key in &old_set {
        if !current_set.contains(&key) {
            let (crate_name, kind, name) = key;
            let was_deprecated = old_deprecated.get(&key).copied().unwrap_or(false);

            if was_deprecated {
                violations.push(PolicyViolation {
                    crate_name: crate_name.to_string(),
                    file: PathBuf::from("(API snapshot comparison)"),
                    line: 0,
                    message: format!(
                        "[API_COMPAT_002] pub {kind} `{name}` removed (was deprecated — verify deprecation window)"
                    ),
                    severity: Severity::Warning,
                });
            } else {
                violations.push(PolicyViolation {
                    crate_name: crate_name.to_string(),
                    file: PathBuf::from("(API snapshot comparison)"),
                    line: 0,
                    message: format!(
                        "[API_COMPAT_001] pub {kind} `{name}` removed without prior #[deprecated] — SemVer violation"
                    ),
                    severity: Severity::Error,
                });
            }
        }
    }

    // Find new items (in current but not in old) — informational
    for &key in &current_set {
        if !old_set.contains(&key) {
            let (crate_name, kind, name) = key;
            violations.push(PolicyViolation {
                crate_name: crate_name.to_string(),
                file: PathBuf::from("(API snapshot comparison)"),
                line: 0,
                message: format!("[API_COMPAT_003] pub {kind} `{name}` added (new API)"),
                severity: Severity::Info,
            });
        }
    }

    Ok(violations)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the workspace version from the root `Cargo.toml`.
fn read_workspace_version(root: &Path) -> Option<String> {
    let cargo_toml = root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).ok()?;

    let mut in_ws_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.package]" {
            in_ws_package = true;
            continue;
        }
        if trimmed.starts_with('[') && trimmed != "[workspace.package]" {
            in_ws_package = false;
        }
        if in_ws_package {
            if let Some(rest) = trimmed.strip_prefix("version") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim().trim_matches('"');
                    if !rest.is_empty() {
                        return Some(rest.to_string());
                    }
                }
            }
        }
    }

    // Fallback: check [package] section
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') && trimmed != "[package]" {
            in_package = false;
        }
        if in_package {
            if let Some(rest) = trimmed.strip_prefix("version") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim().trim_matches('"');
                    if !rest.is_empty() && !rest.contains("workspace") {
                        return Some(rest.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Get the current UTC timestamp in ISO 8601 format.
fn current_timestamp() -> String {
    // Simple epoch-based timestamp (no chrono dependency)
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{epoch}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{CrateInfo, WorkspaceInfo};
    use std::fs;

    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "api_compat_{}_{}_{}",
            std::process::id(),
            suffix,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&base).ok();
        base
    }

    fn make_workspace(dir: &Path, source: &str) -> WorkspaceInfo {
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(src_dir.join("lib.rs"), source).ok();

        WorkspaceInfo {
            root: dir.to_path_buf(),
            crates: vec![CrateInfo {
                name: "test-crate".to_string(),
                path: dir.to_path_buf(),
                is_core: false,
            }],
        }
    }

    #[test]
    fn test_extract_public_items_fn() {
        let content = "pub fn my_func(x: i32) -> i32 { x }\nfn private() {}\n";
        let items = extract_public_items(content, "my-crate", "src/lib.rs");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, "fn");
        assert_eq!(items[0].name, "my_func");
        assert!(!items[0].is_deprecated);
    }

    #[test]
    fn test_extract_public_items_struct_enum_trait() {
        let content = r#"
pub struct MyStruct;
pub enum MyEnum { A, B }
pub trait MyTrait {}
pub type MyAlias = i32;
pub const MY_CONST: i32 = 42;
"#;
        let items = extract_public_items(content, "my-crate", "src/lib.rs");
        assert_eq!(items.len(), 5);
        let kinds: Vec<&str> = items.iter().map(|i| i.kind.as_str()).collect();
        assert!(kinds.contains(&"struct"));
        assert!(kinds.contains(&"enum"));
        assert!(kinds.contains(&"trait"));
        assert!(kinds.contains(&"type"));
        assert!(kinds.contains(&"const"));
    }

    #[test]
    fn test_extract_public_items_deprecated() {
        let content = r#"
#[deprecated(since = "0.3.0")]
pub fn old_func() {}
pub fn new_func() {}
"#;
        let items = extract_public_items(content, "my-crate", "src/lib.rs");
        assert_eq!(items.len(), 2);
        let old = items.iter().find(|i| i.name == "old_func");
        assert!(old.is_some());
        assert!(old.map(|i| i.is_deprecated).unwrap_or(false));

        let new = items.iter().find(|i| i.name == "new_func");
        assert!(new.is_some());
        assert!(!new.map(|i| i.is_deprecated).unwrap_or(true));
    }

    #[test]
    fn test_extract_skips_pub_crate() {
        let content = "pub(crate) fn internal() {}\npub fn external() {}\n";
        let items = extract_public_items(content, "my-crate", "src/lib.rs");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "external");
    }

    #[test]
    fn test_api_snapshot_save_load() {
        let dir = temp_dir("snap_save_load");
        let snapshot = ApiSnapshot {
            version: "0.4.0".to_string(),
            timestamp: "12345".to_string(),
            items: vec![ApiItem {
                crate_name: "my-crate".to_string(),
                file: "src/lib.rs".to_string(),
                line: 1,
                kind: "fn".to_string(),
                name: "my_func".to_string(),
                is_deprecated: false,
            }],
        };

        let path = dir.join("snapshot.json");
        snapshot.save(&path).expect("save should succeed");

        let loaded = ApiSnapshot::load(&path).expect("load should succeed");
        assert_eq!(loaded.version, "0.4.0");
        assert_eq!(loaded.items.len(), 1);
        assert_eq!(loaded.items[0].name, "my_func");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_api_removal_without_deprecation_error() {
        let dir = temp_dir("api_removal");

        // Old snapshot has two items
        let old_snapshot = ApiSnapshot {
            version: "0.3.0".to_string(),
            timestamp: "1".to_string(),
            items: vec![
                ApiItem {
                    crate_name: "test-crate".to_string(),
                    file: "src/lib.rs".to_string(),
                    line: 1,
                    kind: "fn".to_string(),
                    name: "kept_func".to_string(),
                    is_deprecated: false,
                },
                ApiItem {
                    crate_name: "test-crate".to_string(),
                    file: "src/lib.rs".to_string(),
                    line: 5,
                    kind: "fn".to_string(),
                    name: "removed_func".to_string(),
                    is_deprecated: false,
                },
            ],
        };
        let snap_path = dir.join("old.json");
        old_snapshot.save(&snap_path).expect("save");

        // Current workspace only has kept_func
        let ws = make_workspace(&dir, "pub fn kept_func() {}\n");

        let violations = check_api_compatibility(&ws, &snap_path).expect("check should succeed");

        let removal_errors: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("API_COMPAT_001"))
            .collect();
        assert_eq!(removal_errors.len(), 1, "Should flag undeprecated removal");
        assert!(removal_errors[0].message.contains("removed_func"));
        assert_eq!(removal_errors[0].severity, Severity::Error);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_api_removal_with_deprecation_warning() {
        let dir = temp_dir("api_removal_dep");

        let old_snapshot = ApiSnapshot {
            version: "0.3.0".to_string(),
            timestamp: "1".to_string(),
            items: vec![ApiItem {
                crate_name: "test-crate".to_string(),
                file: "src/lib.rs".to_string(),
                line: 1,
                kind: "fn".to_string(),
                name: "removed_func".to_string(),
                is_deprecated: true, // was deprecated
            }],
        };
        let snap_path = dir.join("old.json");
        old_snapshot.save(&snap_path).expect("save");

        // Current workspace has nothing
        let ws = make_workspace(&dir, "// empty\n");

        let violations = check_api_compatibility(&ws, &snap_path).expect("check should succeed");

        let removal_warns: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("API_COMPAT_002"))
            .collect();
        assert_eq!(
            removal_warns.len(),
            1,
            "Should report deprecated removal as warning"
        );
        assert_eq!(removal_warns[0].severity, Severity::Warning);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_api_new_items_info() {
        let dir = temp_dir("api_new");

        // Empty old snapshot
        let old_snapshot = ApiSnapshot {
            version: "0.3.0".to_string(),
            timestamp: "1".to_string(),
            items: vec![],
        };
        let snap_path = dir.join("old.json");
        old_snapshot.save(&snap_path).expect("save");

        let ws = make_workspace(&dir, "pub fn new_func() {}\n");

        let violations = check_api_compatibility(&ws, &snap_path).expect("check should succeed");

        let new_infos: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("API_COMPAT_003"))
            .collect();
        assert_eq!(new_infos.len(), 1, "Should report new item as info");
        assert_eq!(new_infos[0].severity, Severity::Info);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_api_compat_check_no_snapshot_is_noop() {
        let dir = temp_dir("api_noop");
        let ws = make_workspace(&dir, "pub fn func() {}\n");

        let check = ApiCompatCheck;
        let violations = check.run(&ws, &dir.join("nonexistent.json"));
        assert!(
            violations.is_empty(),
            "No snapshot should mean no violations"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_full_check_pipeline() {
        let dir = temp_dir("full_pipeline");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            r#"
pub fn stable_func() {}

#[deprecated(since = "0.4.0", note = "use stable_func")]
pub fn deprecated_func() {}
"#,
        )
        .ok();

        let ws = WorkspaceInfo {
            root: dir.clone(),
            crates: vec![CrateInfo {
                name: "test-crate".to_string(),
                path: dir.clone(),
                is_core: false,
            }],
        };

        // 1. Save API snapshot
        let snap_path = dir.join("api_snapshot.json");
        save_api_snapshot(&ws, &snap_path).expect("save snapshot");

        // 2. Load and verify
        let snapshot = ApiSnapshot::load(&snap_path).expect("load snapshot");
        assert_eq!(snapshot.items.len(), 2);

        // 3. Check compatibility (same version — no removals)
        let violations = check_api_compatibility(&ws, &snap_path).expect("check compat");
        // No removals, no additions (same source)
        let removals: Vec<_> = violations
            .iter()
            .filter(|v| {
                v.message.contains("API_COMPAT_001") || v.message.contains("API_COMPAT_002")
            })
            .collect();
        assert!(removals.is_empty(), "No removals expected");

        let _ = fs::remove_dir_all(&dir);
    }
}
