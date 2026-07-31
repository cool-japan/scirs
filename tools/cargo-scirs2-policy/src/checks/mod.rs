//! Fine-grained linter checks that emit [`PolicyViolation`] with per-line
//! source locations.
//!
//! These checks complement the existing [`crate::rules::PolicyRule`] trait
//! checks, providing more precise diagnostics (file + line number) and the
//! richer [`PolicyViolation`] type.
//!
//! # Available checks
//!
//! | Module | Check | Rule ID |
//! |--------|-------|---------|
//! | [`banned_imports`] | `use rand::` / `use ndarray::` in non-core crates | `SOURCE_SCAN_001/002` |
//! | [`banned_deps`] | Banned crates in `Cargo.toml` | `BANNED_DEP_001` |
//! | [`unwrap_check`] | `.unwrap()` outside `#[cfg(test)]` blocks | `UNWRAP_001` |
//! | [`semver_check`] | Deprecation policy enforcement | `DEPRECATION_001..004` |
//! | [`api_compat`] | Public API surface compatibility | `API_COMPAT_001..003` |
//! | [`ignore_audit`] | `#[ignore]` reason taxonomy + fake-passing test patterns | `IGNORE_AUDIT_001..004` |

pub mod api_compat;
pub mod banned_deps;
pub mod banned_imports;
pub mod ignore_audit;
pub mod semver_check;
pub mod unwrap_check;

use crate::violation::PolicyViolation;
use crate::workspace::WorkspaceInfo;

/// Run all checks in this module against a workspace and return the
/// aggregated violations.
///
/// Violations from all checks are concatenated; the caller is responsible
/// for sorting or grouping them as needed.
///
/// Note: the [`api_compat::ApiCompatCheck`] is not included here because
/// it requires an explicit snapshot path.  Use [`run_all_checks_with_snapshot`]
/// to include API compatibility checks.
pub fn run_all_checks(workspace: &WorkspaceInfo) -> Vec<PolicyViolation> {
    let mut out = Vec::new();
    out.extend(banned_imports::BannedImportCheck.run(workspace));
    out.extend(banned_deps::BannedDepCheck.run(workspace));
    out.extend(unwrap_check::UnwrapCheck.run(workspace));
    out.extend(semver_check::SemVerCheck.run(workspace));
    out.extend(ignore_audit::IgnoreAuditCheck.run(workspace));
    out
}

/// Run all checks including API compatibility against a saved snapshot.
///
/// If `snapshot_path` is `None`, the API compatibility check is skipped.
pub fn run_all_checks_with_snapshot(
    workspace: &WorkspaceInfo,
    snapshot_path: Option<&std::path::Path>,
) -> Vec<PolicyViolation> {
    let mut out = run_all_checks(workspace);
    if let Some(path) = snapshot_path {
        out.extend(api_compat::ApiCompatCheck.run(workspace, path));
    }
    out
}
