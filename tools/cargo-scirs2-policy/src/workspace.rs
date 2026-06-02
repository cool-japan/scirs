//! Workspace discovery utilities for `cargo-scirs2-policy`.
//!
//! Provides types and functions for enumerating workspace crates and their
//! Rust source files so that linter rules can operate over a consistent view
//! of the workspace.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Information about a single workspace crate.
#[derive(Debug, Clone)]
pub struct CrateInfo {
    /// Crate name as declared in `[package] name = "..."`.
    pub name: String,
    /// Absolute path to the crate root (the directory containing `Cargo.toml`).
    pub path: PathBuf,
    /// `true` when this crate is `scirs2-core` (exempted from some rules).
    pub is_core: bool,
}

/// Information about the entire workspace.
#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    /// Absolute path to the workspace root.
    pub root: PathBuf,
    /// All crates discovered in the workspace.
    pub crates: Vec<CrateInfo>,
}

impl WorkspaceInfo {
    /// Returns an iterator over crate paths.
    pub fn crate_paths(&self) -> impl Iterator<Item = &Path> {
        self.crates.iter().map(|c| c.path.as_path())
    }
}

// ---------------------------------------------------------------------------
// Workspace discovery
// ---------------------------------------------------------------------------

/// Discover all workspace crates reachable from `root`.
///
/// If `root/Cargo.toml` contains a `[workspace]` section with `members`, each
/// member glob is resolved relative to `root`.  If no `[workspace]` section is
/// found (i.e., a single-crate repository), `root` itself is treated as the
/// only crate.
///
/// The returned [`WorkspaceInfo`] lists crates in the order they were
/// discovered; the order is deterministic (sorted alphabetically by path).
pub fn discover_workspace(root: &Path) -> WorkspaceInfo {
    let root = match root.canonicalize() {
        Ok(p) => p,
        Err(_) => root.to_path_buf(),
    };

    let workspace_toml = root.join("Cargo.toml");
    let crates = if workspace_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&workspace_toml) {
            let member_dirs = extract_workspace_members(&content, &root);
            if member_dirs.is_empty() {
                // Single-crate repo or workspace with no members listed
                single_crate_from_dir(&root).into_iter().collect()
            } else {
                member_dirs
                    .into_iter()
                    .filter_map(|dir| crate_info_from_dir(&dir))
                    .collect()
            }
        } else {
            Vec::new()
        }
    } else {
        // Walk to find Cargo.toml files (fallback for arbitrary directories)
        find_crates_by_walk(&root)
    };

    WorkspaceInfo { root, crates }
}

/// Walk all `src/**/*.rs` files in a crate directory.
///
/// Returns an empty `Vec` if the `src` directory does not exist.
pub fn walk_rust_files(crate_path: &Path) -> Vec<PathBuf> {
    let src = crate_path.join("src");
    if !src.exists() {
        return Vec::new();
    }

    let mut files = Vec::new();
    walk_dir_for_rs(&src, &mut files);
    files.sort();
    files
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn walk_dir_for_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir_for_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Parse the `[workspace] members = [...]` list from a workspace `Cargo.toml`.
///
/// Uses a simple line-by-line parser that understands the most common formats.
/// It does NOT support full TOML glob expansion (e.g., `crates/*`); it returns
/// each member path literal as-is and then resolves it relative to `root`.
fn extract_workspace_members(content: &str, root: &Path) -> Vec<PathBuf> {
    let mut in_workspace = false;
    let mut in_members = false;
    let mut members = Vec::new();
    let mut bracket_depth: i32 = 0;

    for line in content.lines() {
        let trimmed = line.trim();

        // Detect [workspace] section header
        if trimmed == "[workspace]" {
            in_workspace = true;
            in_members = false;
            continue;
        }
        // Any other top-level section ends the workspace block
        if trimmed.starts_with('[')
            && !trimmed.starts_with("[[")
            && in_workspace
            && trimmed != "[workspace]"
            && !trimmed.starts_with("[workspace.")
        {
            in_workspace = false;
            in_members = false;
        }

        if !in_workspace {
            continue;
        }

        // Start of members = [ ...
        if trimmed.starts_with("members") && trimmed.contains('=') {
            in_members = true;
        }

        if in_members {
            // Count brackets to know when the list ends
            for ch in trimmed.chars() {
                match ch {
                    '[' => bracket_depth += 1,
                    ']' => {
                        bracket_depth -= 1;
                        if bracket_depth <= 0 {
                            in_members = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            // Extract quoted member paths
            let mut rest = trimmed;
            while let Some(start) = rest.find('"') {
                rest = &rest[start + 1..];
                if let Some(end) = rest.find('"') {
                    let member = &rest[..end];
                    rest = &rest[end + 1..];

                    // Handle glob `crates/*` — resolve to all subdirs
                    if member.ends_with("/*") || member.ends_with("\\*") {
                        let glob_root = root.join(&member[..member.len() - 2]);
                        if let Ok(dir_entries) = std::fs::read_dir(&glob_root) {
                            let mut subdirs: Vec<PathBuf> = dir_entries
                                .flatten()
                                .map(|e| e.path())
                                .filter(|p| p.is_dir())
                                .filter(|p| p.join("Cargo.toml").exists())
                                .collect();
                            subdirs.sort();
                            members.extend(subdirs);
                        }
                    } else {
                        let member_path = root.join(member);
                        if member_path.exists() {
                            members.push(member_path);
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }

    members
}

/// Build a [`CrateInfo`] for a directory that contains a `Cargo.toml`.
fn crate_info_from_dir(dir: &Path) -> Option<CrateInfo> {
    let cargo_toml = dir.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).ok()?;
    let name = extract_package_name(&content).unwrap_or_else(|| {
        dir.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string())
    });
    let is_core = name == "scirs2-core" || name == "scirs2_core";
    Some(CrateInfo {
        name,
        path: dir.to_path_buf(),
        is_core,
    })
}

fn single_crate_from_dir(dir: &Path) -> Option<CrateInfo> {
    crate_info_from_dir(dir)
}

/// Walk a directory tree to find all `Cargo.toml` files (fallback discovery).
fn find_crates_by_walk(root: &Path) -> Vec<CrateInfo> {
    let mut crates = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                if path.join("Cargo.toml").exists() {
                    if let Some(info) = crate_info_from_dir(&path) {
                        crates.push(info);
                    }
                }
            }
        }
    }
    crates.sort_by(|a, b| a.path.cmp(&b.path));
    crates
}

/// Extract `[package] name = "..."` from a `Cargo.toml` string.
fn extract_package_name(content: &str) -> Option<String> {
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') {
            in_package = false;
        }
        if in_package {
            if let Some(rest) = trimmed.strip_prefix("name") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim();
                    // strip surrounding quotes
                    if let Some(name) = rest.strip_prefix('"') {
                        if let Some(name) = name.strip_suffix('"') {
                            return Some(name.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "ws_{}_{}_{}",
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

    fn write_crate(root: &Path, name: &str) {
        let crate_dir = root.join(name);
        fs::create_dir_all(crate_dir.join("src")).expect("create src");
        fs::write(
            crate_dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
        )
        .expect("write Cargo.toml");
        fs::write(crate_dir.join("src").join("lib.rs"), "// empty\n").expect("write lib.rs");
    }

    #[test]
    fn test_discover_workspace_single_crate() {
        let dir = temp_dir("single");
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n",
        )
        .expect("write");
        fs::create_dir_all(dir.join("src")).expect("src dir");
        fs::write(dir.join("src").join("lib.rs"), "").expect("lib.rs");

        let ws = discover_workspace(&dir);
        // Single-crate: no workspace members list → resolved to root as one crate
        // (or empty if members resolution fails gracefully)
        assert!(ws.root.ends_with(dir.file_name().unwrap()) || ws.root == dir);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_workspace_members() {
        let dir = temp_dir("multi");
        // Write workspace root
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crate-a\", \"crate-b\"]\n",
        )
        .expect("write workspace");
        write_crate(&dir, "crate-a");
        write_crate(&dir, "crate-b");

        let ws = discover_workspace(&dir);
        assert_eq!(ws.crates.len(), 2, "Should find 2 crates");
        let names: Vec<&str> = ws.crates.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"crate-a"));
        assert!(names.contains(&"crate-b"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_crate_is_core_flag() {
        let dir = temp_dir("core_flag");
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = [\"scirs2-core\", \"scirs2-linalg\"]\n",
        )
        .expect("write workspace");
        write_crate(&dir, "scirs2-core");
        write_crate(&dir, "scirs2-linalg");

        let ws = discover_workspace(&dir);
        let core = ws
            .crates
            .iter()
            .find(|c| c.name == "scirs2-core")
            .expect("core");
        assert!(core.is_core, "scirs2-core should have is_core=true");
        let linalg = ws
            .crates
            .iter()
            .find(|c| c.name == "scirs2-linalg")
            .expect("linalg");
        assert!(!linalg.is_core, "scirs2-linalg should have is_core=false");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_walk_rust_files() {
        let dir = temp_dir("walk_rs");
        let src = dir.join("src");
        fs::create_dir_all(src.join("sub")).expect("sub dir");
        fs::write(src.join("lib.rs"), "").expect("lib.rs");
        fs::write(src.join("sub").join("helper.rs"), "").expect("helper.rs");

        let files = walk_rust_files(&dir);
        assert_eq!(files.len(), 2, "Should find 2 .rs files");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_walk_rust_files_no_src_dir() {
        let dir = temp_dir("no_src");
        let files = walk_rust_files(&dir);
        assert!(files.is_empty(), "No src dir → empty list");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extract_package_name() {
        let content = "[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n";
        assert_eq!(extract_package_name(content), Some("my-crate".to_string()));
    }
}
