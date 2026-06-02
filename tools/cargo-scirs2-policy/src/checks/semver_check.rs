//! Deprecation policy enforcement check.
//!
//! Scans Rust source files for `#[deprecated]` attributes and validates them
//! against the project's deprecation timeline policy.
//!
//! ## Rules
//!
//! | Rule ID | Severity | Description |
//! |---------|----------|-------------|
//! | `DEPRECATION_001` | WARNING | `#[deprecated]` missing `since` version |
//! | `DEPRECATION_002` | WARNING | `#[deprecated]` missing `note` / migration guidance |
//! | `DEPRECATION_003` | INFO | Item deprecated 2+ minor versions ago — ready for removal |
//! | `DEPRECATION_004` | WARNING | `since` version is newer than the current crate version |

use crate::version_policy::{SemVer, VersionPolicy};
use crate::violation::{PolicyViolation, Severity};
use crate::workspace::{self, WorkspaceInfo};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public check type
// ---------------------------------------------------------------------------

/// Linter check that enforces the deprecation policy on `#[deprecated]` attrs.
pub struct SemVerCheck;

impl SemVerCheck {
    /// Run the deprecation policy check across the entire workspace.
    pub fn run(&self, workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
        let policy = VersionPolicy::default();
        check_deprecation_policy(workspace, &policy)
    }
}

// ---------------------------------------------------------------------------
// Deprecation entry
// ---------------------------------------------------------------------------

/// A single `#[deprecated]` annotation found in source code.
#[derive(Debug, Clone)]
pub struct DeprecationEntry {
    /// Absolute path to the file containing the attribute.
    pub file: PathBuf,
    /// 1-based line number of the `#[deprecated]` attribute.
    pub line: usize,
    /// Name of the item (function, struct, etc.) if extractable.
    pub item_name: String,
    /// Value of the `since` field, if present (e.g., `"0.4.0"`).
    pub since_version: Option<String>,
    /// Value of the `note` field, if present.
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Core check logic
// ---------------------------------------------------------------------------

/// Run the deprecation policy check against all crates in the workspace.
///
/// Returns a list of policy violations for any deprecation attributes that
/// do not conform to the configured [`VersionPolicy`].
pub fn check_deprecation_policy(
    workspace: &WorkspaceInfo,
    policy: &VersionPolicy,
) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();

    for krate in &workspace.crates {
        let crate_version = read_crate_version(&krate.path);
        let rs_files = workspace::walk_rust_files(&krate.path);

        for file in &rs_files {
            let content = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let entries = parse_deprecated_attributes(&content, file);

            for entry in &entries {
                // Check: missing `since` field
                if policy.require_since && entry.since_version.is_none() {
                    violations.push(PolicyViolation {
                        crate_name: krate.name.clone(),
                        file: entry.file.clone(),
                        line: entry.line,
                        message: format!(
                            "[DEPRECATION_001] #[deprecated] on `{}` is missing `since` version",
                            entry.item_name,
                        ),
                        severity: Severity::Warning,
                    });
                }

                // Check: missing `note` field
                if policy.require_note && entry.note.is_none() {
                    violations.push(PolicyViolation {
                        crate_name: krate.name.clone(),
                        file: entry.file.clone(),
                        line: entry.line,
                        message: format!(
                            "[DEPRECATION_002] #[deprecated] on `{}` is missing `note` (should explain replacement)",
                            entry.item_name,
                        ),
                        severity: Severity::Warning,
                    });
                }

                // Check: ready for removal (deprecated N+ minor versions ago)
                if let (Some(since_str), Some(ref current_str)) =
                    (&entry.since_version, &crate_version)
                {
                    if let (Some(since_ver), Some(current_ver)) =
                        (SemVer::parse(since_str), SemVer::parse(current_str))
                    {
                        let distance = since_ver.minor_distance(&current_ver);

                        // since_version newer than current? That's suspicious
                        if since_ver > current_ver {
                            violations.push(PolicyViolation {
                                crate_name: krate.name.clone(),
                                file: entry.file.clone(),
                                line: entry.line,
                                message: format!(
                                    "[DEPRECATION_004] #[deprecated(since = \"{}\")] on `{}` is newer than current version {}",
                                    since_str, entry.item_name, current_str,
                                ),
                                severity: Severity::Warning,
                            });
                        } else if distance >= policy.deprecation_window {
                            violations.push(PolicyViolation {
                                crate_name: krate.name.clone(),
                                file: entry.file.clone(),
                                line: entry.line,
                                message: format!(
                                    "[DEPRECATION_003] `{}` deprecated since {} ({} minor versions ago) — ready for removal",
                                    entry.item_name, since_str, distance,
                                ),
                                severity: Severity::Info,
                            });
                        }
                    }
                }
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Attribute parsing
// ---------------------------------------------------------------------------

/// Parse all `#[deprecated(...)]` attributes from a Rust source file.
///
/// Uses a simple line-based regex-like scanner (no syn/proc-macro dependency).
/// Handles single-line and multi-line attributes.
pub fn parse_deprecated_attributes(content: &str, file: &Path) -> Vec<DeprecationEntry> {
    let mut entries = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        // Look for #[deprecated or #[deprecated(
        if trimmed.starts_with("#[deprecated") || trimmed.starts_with("#![deprecated") {
            let attr_start_line = i + 1; // 1-based

            // Collect the full attribute text (may span multiple lines)
            let attr_text = collect_attribute_text(&lines, &mut i);

            // Parse since and note from the attribute text
            let since_version = extract_field(&attr_text, "since");
            let note = extract_field(&attr_text, "note");

            // Try to find the item name on the next non-attribute, non-comment,
            // non-empty line
            let item_name = find_item_name(&lines, i);

            entries.push(DeprecationEntry {
                file: file.to_path_buf(),
                line: attr_start_line,
                item_name,
                since_version,
                note,
            });
        }

        i += 1;
    }

    entries
}

/// Collect the full text of an attribute starting at `lines[*idx]`.
///
/// Handles multi-line attributes by tracking balanced parentheses.
/// Advances `*idx` past the last line of the attribute.
fn collect_attribute_text(lines: &[&str], idx: &mut usize) -> String {
    let mut text = String::new();
    let mut paren_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;
    let start = *idx;

    for (j, line) in lines.iter().enumerate().skip(start) {
        let line = line.trim();
        text.push_str(line);
        text.push(' ');

        for ch in line.chars() {
            match ch {
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
        }

        // Attribute is complete when all brackets and parens are closed
        if bracket_depth <= 0 && paren_depth <= 0 {
            *idx = j;
            break;
        }
    }

    text
}

/// Extract the value of a named field (e.g., `since` or `note`) from an
/// attribute text string.
///
/// Looks for patterns like `since = "0.4.0"` or `note = "use X instead"`.
fn extract_field(attr_text: &str, field_name: &str) -> Option<String> {
    // Find `field_name` followed by `=` and a quoted string
    let search = field_name.to_string();
    let mut pos = 0;

    while pos < attr_text.len() {
        let remaining = &attr_text[pos..];
        let field_pos = remaining.find(&search)?;

        let after_field = &remaining[field_pos + search.len()..];
        let after_field = after_field.trim_start();

        // Must be followed by '='
        if !after_field.starts_with('=') {
            pos += field_pos + search.len();
            continue;
        }

        let after_eq = after_field[1..].trim_start();

        // Extract the quoted string value
        if let Some(value) = extract_quoted_string(after_eq) {
            return Some(value);
        }

        pos += field_pos + search.len();
    }

    None
}

/// Extract a double-quoted string from the start of the input.
///
/// Returns the content between the first pair of unescaped double quotes,
/// or `None` if no valid quoted string is found.
fn extract_quoted_string(input: &str) -> Option<String> {
    let input = input.trim_start();
    if !input.starts_with('"') {
        return None;
    }

    let content = &input[1..];
    let mut result = String::new();
    let mut escaped = false;

    for ch in content.chars() {
        if escaped {
            result.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(result);
        } else {
            result.push(ch);
        }
    }

    None
}

/// Find the name of the item following a `#[deprecated]` attribute.
///
/// Scans forward from `start_line` (exclusive) for the next `pub fn`, `pub struct`,
/// `pub enum`, `pub trait`, `fn`, `struct`, `enum`, or `trait` declaration.
fn find_item_name(lines: &[&str], start_line: usize) -> String {
    for trimmed in lines
        .iter()
        .take(lines.len().min(start_line + 10))
        .skip(start_line + 1)
        .map(|l| l.trim())
    {
        // Skip other attributes, comments, and empty lines
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("//")
            || trimmed.starts_with("/*")
        {
            continue;
        }

        // Try to extract the item name from declaration keywords
        for keyword in &[
            "pub fn ",
            "fn ",
            "pub struct ",
            "struct ",
            "pub enum ",
            "enum ",
            "pub trait ",
            "trait ",
            "pub type ",
            "type ",
            "pub const ",
            "const ",
            "pub static ",
            "static ",
            "pub mod ",
            "mod ",
        ] {
            if let Some(rest) = find_after_keyword(trimmed, keyword) {
                // Name is the next identifier (up to '(' '<' ':' '{' ';' ' ')
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    return name;
                }
            }
        }

        // If we hit a non-attribute, non-comment line but couldn't parse it,
        // use the first word as the item name
        let first_word: String = trimmed
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !first_word.is_empty() {
            return first_word;
        }
    }

    "<unknown>".to_string()
}

/// Check if a line contains a keyword and return the text after it.
fn find_after_keyword<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
    // Handle visibility modifiers like `pub(crate) fn`
    let line_stripped = line
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub(super) ");

    if let Some(pos) = line.find(keyword) {
        Some(&line[pos + keyword.len()..])
    } else if let Some(pos) = line_stripped.find(keyword) {
        // Recalculate position in original line
        let offset = line.len() - line_stripped.len();
        Some(&line[offset + pos + keyword.len()..])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Cargo.toml version extraction
// ---------------------------------------------------------------------------

/// Read the version from a crate's `Cargo.toml`.
///
/// Handles both `version = "X.Y.Z"` and `version.workspace = true`.
/// For workspace-inherited versions, attempts to find the workspace root.
fn read_crate_version(crate_path: &Path) -> Option<String> {
    let cargo_toml = crate_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).ok()?;

    // Check for direct version first
    if let Some(version) = extract_toml_version(&content) {
        return Some(version);
    }

    // Check for workspace-inherited version
    if content.contains("version.workspace = true") || content.contains("version.workspace=true") {
        // Walk up to find the workspace root Cargo.toml
        let mut dir = crate_path.to_path_buf();
        while let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
            let ws_toml = dir.join("Cargo.toml");
            if ws_toml.exists() {
                if let Ok(ws_content) = std::fs::read_to_string(&ws_toml) {
                    if ws_content.contains("[workspace.package]")
                        || ws_content.contains("[workspace]")
                    {
                        if let Some(version) = extract_workspace_version(&ws_content) {
                            return Some(version);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Extract `version = "X.Y.Z"` from a `[package]` section.
fn extract_toml_version(content: &str) -> Option<String> {
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
                    let rest = rest.trim();
                    // Skip workspace = true
                    if rest.contains("workspace") {
                        return None;
                    }
                    return extract_quoted_string(rest);
                }
            }
        }
    }
    None
}

/// Extract version from `[workspace.package]` section.
fn extract_workspace_version(content: &str) -> Option<String> {
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
                    let rest = rest.trim();
                    return extract_quoted_string(rest);
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
    use crate::workspace::{CrateInfo, WorkspaceInfo};
    use std::fs;

    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "semver_check_{}_{}_{}",
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

    #[test]
    fn test_parse_deprecated_attribute_full() {
        let content = r#"
#[deprecated(since = "0.4.0", note = "use new_function instead")]
pub fn old_function() {}
"#;
        let entries = parse_deprecated_attributes(content, Path::new("/test.rs"));
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.since_version.as_deref(), Some("0.4.0"));
        assert_eq!(entry.note.as_deref(), Some("use new_function instead"));
        assert_eq!(entry.item_name, "old_function");
        assert_eq!(entry.line, 2);
    }

    #[test]
    fn test_parse_deprecated_attribute_no_fields() {
        let content = r#"
#[deprecated]
pub fn bare_deprecated() {}
"#;
        let entries = parse_deprecated_attributes(content, Path::new("/test.rs"));
        assert_eq!(entries.len(), 1);
        assert!(entries[0].since_version.is_none());
        assert!(entries[0].note.is_none());
        assert_eq!(entries[0].item_name, "bare_deprecated");
    }

    #[test]
    fn test_parse_deprecated_attribute_only_since() {
        let content = r#"
#[deprecated(since = "0.3.0")]
pub struct OldStruct;
"#;
        let entries = parse_deprecated_attributes(content, Path::new("/test.rs"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].since_version.as_deref(), Some("0.3.0"));
        assert!(entries[0].note.is_none());
        assert_eq!(entries[0].item_name, "OldStruct");
    }

    #[test]
    fn test_parse_deprecated_multiline() {
        let content = r#"
#[deprecated(
    since = "0.2.0",
    note = "use NewEnum instead"
)]
pub enum OldEnum {
    A,
    B,
}
"#;
        let entries = parse_deprecated_attributes(content, Path::new("/test.rs"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].since_version.as_deref(), Some("0.2.0"));
        assert_eq!(entries[0].note.as_deref(), Some("use NewEnum instead"));
        assert_eq!(entries[0].item_name, "OldEnum");
    }

    #[test]
    fn test_parse_multiple_deprecated() {
        let content = r#"
#[deprecated(since = "0.3.0", note = "use a")]
pub fn func_a() {}

#[deprecated(since = "0.4.0", note = "use b")]
pub fn func_b() {}
"#;
        let entries = parse_deprecated_attributes(content, Path::new("/test.rs"));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].item_name, "func_a");
        assert_eq!(entries[1].item_name, "func_b");
    }

    #[test]
    fn test_deprecation_current_version_ok() {
        let dir = temp_dir("dep_ok");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            "#[deprecated(since = \"0.4.0\", note = \"use new_fn\")]\npub fn old_fn() {}\n",
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

        let policy = VersionPolicy::default();
        let violations = check_deprecation_policy(&ws, &policy);

        // No warnings for current-version deprecation
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| {
                v.message.contains("DEPRECATION_001")
                    || v.message.contains("DEPRECATION_002")
                    || v.message.contains("DEPRECATION_004")
            })
            .collect();
        assert!(
            errors.is_empty(),
            "Current-version deprecation should have no issues: {errors:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_deprecation_two_versions_ago_flagged() {
        let dir = temp_dir("dep_old");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.6.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            "#[deprecated(since = \"0.4.0\", note = \"use new_fn\")]\npub fn old_fn() {}\n",
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

        let policy = VersionPolicy::default();
        let violations = check_deprecation_policy(&ws, &policy);

        let removal_ready: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("DEPRECATION_003"))
            .collect();
        assert_eq!(removal_ready.len(), 1, "Should flag item for removal");
        assert!(removal_ready[0].message.contains("ready for removal"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_deprecation_missing_since_warning() {
        let dir = temp_dir("dep_no_since");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            "#[deprecated(note = \"use new_fn\")]\npub fn old_fn() {}\n",
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

        let policy = VersionPolicy::default();
        let violations = check_deprecation_policy(&ws, &policy);

        let missing_since: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("DEPRECATION_001"))
            .collect();
        assert_eq!(missing_since.len(), 1, "Should warn about missing since");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_deprecation_missing_note_warning() {
        let dir = temp_dir("dep_no_note");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            "#[deprecated(since = \"0.4.0\")]\npub fn old_fn() {}\n",
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

        let policy = VersionPolicy::default();
        let violations = check_deprecation_policy(&ws, &policy);

        let missing_note: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("DEPRECATION_002"))
            .collect();
        assert_eq!(missing_note.len(), 1, "Should warn about missing note");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_deprecation_future_since_warning() {
        let dir = temp_dir("dep_future");
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).ok();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"test-crate\"\nversion = \"0.4.0\"\n",
        )
        .ok();
        fs::write(
            src_dir.join("lib.rs"),
            "#[deprecated(since = \"0.9.0\", note = \"future\")]\npub fn future_fn() {}\n",
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

        let policy = VersionPolicy::default();
        let violations = check_deprecation_policy(&ws, &policy);

        let future_warns: Vec<_> = violations
            .iter()
            .filter(|v| v.message.contains("DEPRECATION_004"))
            .collect();
        assert_eq!(
            future_warns.len(),
            1,
            "Should warn about future since version"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extract_field_since() {
        let attr = r#"#[deprecated(since = "0.4.0", note = "use X")] "#;
        assert_eq!(extract_field(attr, "since"), Some("0.4.0".to_string()));
        assert_eq!(extract_field(attr, "note"), Some("use X".to_string()));
    }

    #[test]
    fn test_extract_field_missing() {
        let attr = "#[deprecated] ";
        assert_eq!(extract_field(attr, "since"), None);
        assert_eq!(extract_field(attr, "note"), None);
    }

    #[test]
    fn test_extract_quoted_string() {
        assert_eq!(
            extract_quoted_string("\"hello world\""),
            Some("hello world".to_string())
        );
        assert_eq!(
            extract_quoted_string("\"with \\\"escaped\\\" quotes\""),
            Some("with \"escaped\" quotes".to_string())
        );
        assert_eq!(extract_quoted_string("no quotes"), None);
        assert_eq!(extract_quoted_string(""), None);
    }

    #[test]
    fn test_extract_toml_version() {
        let content = "[package]\nname = \"my-crate\"\nversion = \"1.2.3\"\n";
        assert_eq!(extract_toml_version(content), Some("1.2.3".to_string()));
    }

    #[test]
    fn test_extract_toml_version_workspace() {
        let content = "[package]\nname = \"my-crate\"\nversion.workspace = true\n";
        assert_eq!(extract_toml_version(content), None);
    }

    #[test]
    fn test_extract_workspace_version() {
        let content = "[workspace.package]\nversion = \"0.4.0\"\nedition = \"2021\"\n";
        assert_eq!(
            extract_workspace_version(content),
            Some("0.4.0".to_string())
        );
    }
}
