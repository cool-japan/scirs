//! `cargo-scirs2-policy` — policy compliance linter for the SciRS2 workspace.
//!
//! # Usage
//!
//! ```text
//! # Check workspace for policy violations
//! cargo scirs2-policy check --workspace /path/to/scirs
//!
//! # Emit JSON output
//! cargo scirs2-policy check --workspace . --format json
//!
//! # List all available rules
//! cargo scirs2-policy rules
//!
//! # Detect packages with multiple versions in Cargo.lock
//! cargo scirs2-policy duplicates --workspace .
//!
//! # Create a benchmark snapshot from criterion output
//! cargo scirs2-policy bench-snapshot --criterion-dir target/criterion --output baseline.json
//!
//! # Compare two benchmark snapshots and detect regressions
//! cargo scirs2-policy bench-diff --baseline baseline.json --current current.json --threshold 0.10
//!
//! # Audit workspace dependency footprint
//! cargo scirs2-policy dep-audit --workspace . --baseline-count 850
//! ```
//!
//! # Implemented rules
//!
//! | ID | Severity | Description |
//! |----|----------|-------------|
//! | `BANNED_DEP_001` | ERROR | Direct use of zip, flate2, bincode, openblas-src, … |
//! | `SOURCE_SCAN_001` | WARN | `use rand::` in non-core source files |
//! | `SOURCE_SCAN_002` | INFO | `use ndarray::` in non-core source files |

use cargo_scirs2_policy::{
    bench_regression, checks, dep_audit, report, rules, version_policy, violation, workspace,
};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Outer `cargo` wrapper so the binary can be invoked as `cargo scirs2-policy`.
#[derive(Parser)]
#[command(name = "cargo")]
#[command(bin_name = "cargo")]
enum Cargo {
    /// SciRS2 policy compliance linter.
    #[command(name = "scirs2-policy")]
    Policy(PolicyArgs),
}

/// Arguments for the `cargo scirs2-policy` subcommand.
#[derive(clap::Args)]
#[command(about = "SciRS2 policy compliance linter")]
#[command(version)]
struct PolicyArgs {
    #[command(subcommand)]
    command: PolicyCommand,
}

/// Available sub-commands.
#[derive(Subcommand)]
enum PolicyCommand {
    /// Check policy compliance across the workspace.
    Check {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Output format: `text` (default) or `json`.
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// List all available policy rules with their IDs and descriptions.
    Rules,
    /// Detect packages that appear with multiple versions in `Cargo.lock`.
    Duplicates {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
    /// Create a benchmark snapshot from a Criterion output directory.
    ///
    /// Walks `--criterion-dir` for `<name>/new/estimates.json` files and
    /// serialises the measurements to `--output` as a JSON snapshot.
    BenchSnapshot {
        /// Path to the Criterion output directory (default: `target/criterion`).
        #[arg(long, default_value = "target/criterion")]
        criterion_dir: PathBuf,
        /// Path to write the snapshot JSON (default: `<TMPDIR>/scirs2_bench_snapshot.json`).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two benchmark snapshots and report regressions.
    ///
    /// Exits with code `1` when any regression exceeds `--threshold`.
    BenchDiff {
        /// Path to the baseline snapshot JSON.
        #[arg(long)]
        baseline: PathBuf,
        /// Path to the current snapshot JSON.
        #[arg(long)]
        current: PathBuf,
        /// Relative regression threshold (default: 0.10 = 10%).
        #[arg(long, default_value = "0.10")]
        threshold: f64,
        /// Print the full diff report instead of only regressions.
        #[arg(long, default_value = "false")]
        full: bool,
    },
    /// Audit the workspace dependency footprint.
    ///
    /// Reports unique package count, direct deps, and flags banned packages.
    DepAudit {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Optional baseline unique-package count for progress tracking.
        #[arg(long)]
        baseline_count: Option<usize>,
        /// Exit with code `1` when banned packages are present.
        #[arg(long, default_value = "false")]
        strict: bool,
    },
    /// Check semantic versioning and deprecation policy compliance.
    ///
    /// Scans `#[deprecated]` attributes for `since`/`note` fields and
    /// flags items that are ready for removal.
    CheckSemver {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Optional path to an API snapshot JSON for compatibility checking.
        #[arg(long)]
        api_snapshot: Option<PathBuf>,
        /// Output format: `text` (default) or `json`.
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Save the current public API surface as a JSON snapshot.
    ///
    /// The snapshot can later be used with `check-semver --api-snapshot`
    /// to detect backward-incompatible changes.
    SaveApiSnapshot {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Path to write the snapshot JSON (default: `<TMPDIR>/scirs2_api_snapshot.json`).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Print the current SemVer commitment level and version policy.
    VersionPolicy {
        /// Path to the workspace root (defaults to the current directory).
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let Cargo::Policy(args) = Cargo::parse();

    let exit_code = match args.command {
        PolicyCommand::Check { workspace, format } => run_check(&workspace, &format as &str),
        PolicyCommand::Rules => {
            print_rules();
            0
        }
        PolicyCommand::Duplicates { workspace } => run_duplicates(&workspace),
        PolicyCommand::BenchSnapshot {
            criterion_dir,
            output,
        } => {
            let output_path =
                output.unwrap_or_else(|| std::env::temp_dir().join("scirs2_bench_snapshot.json"));
            run_bench_snapshot(&criterion_dir, &output_path)
        }
        PolicyCommand::BenchDiff {
            baseline,
            current,
            threshold,
            full,
        } => run_bench_diff(&baseline, &current, threshold, full),
        PolicyCommand::DepAudit {
            workspace,
            baseline_count,
            strict,
        } => run_dep_audit(&workspace, baseline_count, strict),
        PolicyCommand::CheckSemver {
            workspace,
            api_snapshot,
            format,
        } => run_check_semver(&workspace, api_snapshot.as_deref(), &format),
        PolicyCommand::SaveApiSnapshot { workspace, output } => {
            let output_path =
                output.unwrap_or_else(|| std::env::temp_dir().join("scirs2_api_snapshot.json"));
            run_save_api_snapshot(&workspace, &output_path)
        }
        PolicyCommand::VersionPolicy { workspace } => run_version_policy(&workspace),
    };

    std::process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

/// Run all policy rules against the workspace and print the results.
///
/// Returns `0` when no violations are found, `1` otherwise.
///
/// This runs both the legacy [`rules::PolicyRule`]-based checks and the newer
/// per-line [`checks`]-based checks (banned imports, unwrap detection, etc.).
fn run_check(workspace: &Path, output_format: &str) -> i32 {
    // Legacy rule-based checks (file-level granularity)
    let rule_violations = rules::check_workspace(workspace);

    // Fine-grained checks (per-line source locations)
    let ws_info = workspace::discover_workspace(workspace);
    let check_violations = checks::run_all_checks(&ws_info);

    // Convert legacy violations to PolicyViolation for unified reporting
    let mut all_policy: Vec<violation::PolicyViolation> = rule_violations
        .into_iter()
        .map(|v| {
            let msg = std::format!("[{}] {}", v.rule_id, v.message);
            violation::PolicyViolation {
                crate_name: String::new(),
                file: std::path::PathBuf::from(v.file.unwrap_or_default()),
                line: 0,
                message: msg,
                severity: match v.severity {
                    rules::Severity::Error => violation::Severity::Error,
                    rules::Severity::Warning => violation::Severity::Warning,
                    rules::Severity::Info => violation::Severity::Info,
                    // Safety: #[non_exhaustive] — future variants treated as Info
                    _ => violation::Severity::Info,
                },
            }
        })
        .collect();

    all_policy.extend(check_violations);

    match output_format {
        "json" => print!("{}", report::json_report(&all_policy)),
        _ => report::print_report(&all_policy),
    }

    violation::exit_code(&all_policy)
}

/// Print the list of registered policy rules to stdout.
fn print_rules() {
    println!("Available policy rules:");
    println!();
    for rule in rules::all_rules() {
        println!("  {:20} {}", rule.id(), rule.description());
    }
}

/// Parse `Cargo.lock` and print packages that have more than one version.
///
/// Returns `0` always — duplicate versions are informational, not an error.
fn run_duplicates(workspace: &Path) -> i32 {
    let lock_path = workspace.join("Cargo.lock");
    if !lock_path.exists() {
        eprintln!("Cargo.lock not found at {}", lock_path.display());
        return 1;
    }

    let duplicates = rules::find_duplicate_deps(&lock_path);
    if duplicates.is_empty() {
        println!("No duplicate dependency versions found.");
    } else {
        println!(
            "Packages with multiple versions ({} total):",
            duplicates.len()
        );
        println!();
        for (name, versions) in &duplicates {
            println!("  {:40} {}", name, versions.join(", "));
        }
    }
    0
}

/// Build a [`bench_regression::BenchmarkSnapshot`] from a Criterion output
/// directory and write it to `output`.
///
/// Returns `0` on success, `1` on any error.
fn run_bench_snapshot(criterion_dir: &Path, output: &Path) -> i32 {
    let snapshot = match bench_regression::BenchmarkSnapshot::from_criterion_dir(criterion_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading criterion directory: {e}");
            return 1;
        }
    };

    if snapshot.is_empty() {
        println!(
            "Warning: no measurements found in {}",
            criterion_dir.display()
        );
    } else {
        println!(
            "Collected {} benchmark measurement(s) from {}",
            snapshot.len(),
            criterion_dir.display()
        );
    }

    match snapshot.save(output) {
        Ok(()) => {
            println!("Snapshot written to {}", output.display());
            0
        }
        Err(e) => {
            eprintln!("Error writing snapshot: {e}");
            1
        }
    }
}

/// Load two snapshots and compare them.
///
/// Returns `0` when no regressions exceed `threshold`, `1` otherwise.
fn run_bench_diff(baseline_path: &Path, current_path: &Path, threshold: f64, full: bool) -> i32 {
    let baseline = match bench_regression::BenchmarkSnapshot::load(baseline_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading baseline snapshot: {e}");
            return 1;
        }
    };

    let current = match bench_regression::BenchmarkSnapshot::load(current_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading current snapshot: {e}");
            return 1;
        }
    };

    if full {
        let report = bench_regression::format_diff_report(&baseline, &current);
        print!("{report}");
    }

    let regressions = bench_regression::compare_snapshots(&baseline, &current, threshold);

    if !full {
        let report = bench_regression::format_regression_report(&regressions);
        print!("{report}");
    }

    if regressions.is_empty() {
        println!(
            "No regressions found (threshold: {:.1}%)",
            threshold * 100.0
        );
        0
    } else {
        eprintln!(
            "{} regression(s) detected above {:.1}% threshold.",
            regressions.len(),
            threshold * 100.0
        );
        1
    }
}

/// Run the dependency audit and print results.
///
/// Returns `0` on success.  When `strict` is `true`, returns `1` if any
/// banned dependencies are present.
fn run_dep_audit(workspace: &Path, baseline_count: Option<usize>, strict: bool) -> i32 {
    let result = dep_audit::run_dep_audit(workspace, baseline_count);
    let report = dep_audit::format_audit_report(&result);
    print!("{report}");

    if strict && !result.flagged_banned.is_empty() {
        eprintln!("Strict mode: banned dependencies present — failing.");
        return 1;
    }
    0
}

/// Run SemVer and deprecation policy checks.
///
/// Returns `0` when no errors are found, `1` otherwise.
fn run_check_semver(
    workspace_path: &Path,
    api_snapshot: Option<&Path>,
    output_format: &str,
) -> i32 {
    let ws_info = workspace::discover_workspace(workspace_path);
    let violations = checks::run_all_checks_with_snapshot(&ws_info, api_snapshot);

    // Filter to only semver/deprecation/api-compat violations
    let semver_violations: Vec<_> = violations
        .into_iter()
        .filter(|v| v.message.contains("DEPRECATION_") || v.message.contains("API_COMPAT_"))
        .collect();

    match output_format {
        "json" => print!("{}", report::json_report(&semver_violations)),
        _ => {
            if semver_violations.is_empty() {
                println!("SemVer/deprecation policy: all checks passed.");
            } else {
                report::print_report(&semver_violations);
            }
        }
    }

    violation::exit_code(&semver_violations)
}

/// Save the current public API surface as a snapshot.
///
/// Returns `0` on success, `1` on error.
fn run_save_api_snapshot(workspace_path: &Path, output: &Path) -> i32 {
    let ws_info = workspace::discover_workspace(workspace_path);
    let items = checks::api_compat::collect_public_items(&ws_info);

    match checks::api_compat::save_api_snapshot(&ws_info, output) {
        Ok(()) => {
            println!(
                "API snapshot saved to {} ({} public items)",
                output.display(),
                items.len(),
            );
            0
        }
        Err(e) => {
            eprintln!("Error saving API snapshot: {e}");
            1
        }
    }
}

/// Print the current version policy information.
fn run_version_policy(workspace_path: &Path) -> i32 {
    let cargo_toml = workspace_path.join("Cargo.toml");
    let version = if cargo_toml.exists() {
        std::fs::read_to_string(&cargo_toml)
            .ok()
            .and_then(|content| {
                // Try workspace.package first
                let mut in_ws_pkg = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed == "[workspace.package]" {
                        in_ws_pkg = true;
                        continue;
                    }
                    if trimmed.starts_with('[') && trimmed != "[workspace.package]" {
                        in_ws_pkg = false;
                    }
                    if in_ws_pkg {
                        if let Some(rest) = trimmed.strip_prefix("version") {
                            let rest = rest.trim_start();
                            if let Some(rest) = rest.strip_prefix('=') {
                                let s = rest.trim();
                                if let Some(inner) = s.strip_prefix('"') {
                                    if let Some(end) = inner.find('"') {
                                        let ver = &inner[..end];
                                        if !ver.is_empty() {
                                            return Some(ver.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                None
            })
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unknown".to_string()
    };

    let commitment = version_policy::current_commitment(&version)
        .unwrap_or(version_policy::SemVerCommitment::PreStable);
    let policy = version_policy::VersionPolicy::default();
    let lts = version_policy::LtsPolicy::default();

    println!("SciRS2 Version Policy");
    println!("=====================");
    println!();
    println!("Current version:        {version}");
    println!("SemVer commitment:      {commitment}");
    println!();
    println!("Deprecation Policy:");
    println!(
        "  Window:               {} minor versions before removal",
        policy.deprecation_window
    );
    println!("  Require since field:  {}", policy.require_since);
    println!("  Require note field:   {}", policy.require_note);
    println!();
    println!("LTS Policy:");
    if lts.branches.is_empty() {
        println!("  Branches:             (none configured)");
    } else {
        println!("  Branches:             {}", lts.branches.join(", "));
    }
    println!(
        "  Security patch window: {} months",
        lts.security_patch_window_months
    );
    println!();

    match commitment {
        version_policy::SemVerCommitment::PreStable => {
            println!("Note: Pre-stable (0.x) — breaking changes allowed per minor version.");
            println!("      Deprecation warnings are still required before removal.");
        }
        version_policy::SemVerCommitment::Stable1x => {
            println!("Note: Stable (1.x) — backward compatibility required within major version.");
            println!("      All removals must follow the full deprecation timeline.");
        }
        _ => {}
    }

    0
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
            "policy_main_{}_{}_{}",
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
    fn test_run_check_clean_workspace_returns_zero() {
        let dir = temp_dir("check_clean");
        let cargo_toml = dir.join("Cargo.toml");
        fs::write(
            &cargo_toml,
            "[package]\nname = \"clean\"\nversion = \"0.1.0\"\n\n[dependencies]\noxiarc-archive = \"1.0\"\n",
        )
        .expect("write Cargo.toml");

        let code = run_check(&dir, "text");
        assert_eq!(code, 0, "Clean workspace should return exit code 0");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_check_with_violation_returns_one() {
        let dir = temp_dir("check_violation");

        // Create a subdirectory so it doesn't get skipped as workspace root
        let sub = dir.join("mycrate");
        fs::create_dir_all(&sub).expect("create subcrate dir");
        let cargo_toml = sub.join("Cargo.toml");
        fs::write(
            &cargo_toml,
            "[package]\nname = \"bad\"\nversion = \"0.1.0\"\n\n[dependencies]\nzip = \"2.0\"\n",
        )
        .expect("write Cargo.toml");

        let code = run_check(&dir, "text");
        assert_eq!(
            code, 1,
            "Workspace with violations should return exit code 1"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_duplicates_missing_lock_returns_one() {
        let dir = temp_dir("dup_missing");
        let code = run_duplicates(&dir);
        assert_eq!(code, 1, "Missing Cargo.lock should return exit code 1");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_duplicates_no_dups_returns_zero() {
        let dir = temp_dir("dup_none");
        let lock = dir.join("Cargo.lock");
        fs::write(
            &lock,
            "version = 3\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n",
        )
        .expect("write Cargo.lock");

        let code = run_duplicates(&dir);
        assert_eq!(code, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_duplicates_with_dups_returns_zero() {
        // Duplicates are informational — should still return 0
        let dir = temp_dir("dup_found");
        let lock = dir.join("Cargo.lock");
        fs::write(
            &lock,
            "version = 3\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.1\"\n",
        )
        .expect("write Cargo.lock");

        let code = run_duplicates(&dir);
        assert_eq!(code, 0, "Duplicates are informational — should return 0");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_check_json_format_does_not_panic() {
        let dir = temp_dir("check_json");
        let code = run_check(&dir, "json");
        // Empty workspace has no violations
        assert_eq!(code, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // bench-snapshot tests
    // ------------------------------------------------------------------

    fn write_estimates(dir: &Path, bench_name: &str, mean_ns: f64) {
        let bench_dir = dir.join(bench_name).join("new");
        fs::create_dir_all(&bench_dir).expect("create bench dir");
        let estimates = serde_json::json!({
            "mean": { "point_estimate": mean_ns, "standard_error": mean_ns * 0.01 },
            "std_dev": { "point_estimate": mean_ns * 0.05, "standard_error": mean_ns * 0.005 }
        });
        fs::write(
            bench_dir.join("estimates.json"),
            serde_json::to_string_pretty(&estimates).expect("serialise"),
        )
        .expect("write estimates.json");
    }

    #[test]
    fn test_run_bench_snapshot_creates_output_file() {
        let crit_dir = temp_dir("snap_crit");
        let out_dir = temp_dir("snap_out");
        write_estimates(&crit_dir, "matmul", 1_234_567.0);
        write_estimates(&crit_dir, "fft", 987_654.0);

        let out_path = out_dir.join("snap.json");
        let code = run_bench_snapshot(&crit_dir, &out_path);
        assert_eq!(code, 0);
        assert!(out_path.exists());

        let snap = bench_regression::BenchmarkSnapshot::load(&out_path).expect("load");
        assert_eq!(snap.len(), 2);

        let _ = fs::remove_dir_all(&crit_dir);
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_run_bench_snapshot_missing_criterion_dir_returns_one() {
        let out = std::env::temp_dir().join("snap_missing_out.json");
        let code = run_bench_snapshot(Path::new("/nonexistent/criterion"), &out);
        assert_eq!(code, 1);
    }

    // ------------------------------------------------------------------
    // bench-diff tests
    // ------------------------------------------------------------------

    fn write_snapshot(path: &Path, measurements: &[(&str, f64)]) {
        let mut snap = bench_regression::BenchmarkSnapshot::new();
        for (name, mean) in measurements {
            snap.measurements
                .push(bench_regression::BenchmarkMeasurement {
                    name: name.to_string(),
                    mean_ns: *mean,
                    std_dev_ns: mean * 0.05,
                    std_err_ns: mean * 0.01,
                });
        }
        snap.save(path).expect("save snapshot");
    }

    #[test]
    fn test_run_bench_diff_no_regression_returns_zero() {
        let dir = temp_dir("diff_ok");
        let baseline = dir.join("baseline.json");
        let current = dir.join("current.json");

        write_snapshot(&baseline, &[("bench_a", 1_000.0), ("bench_b", 2_000.0)]);
        write_snapshot(&current, &[("bench_a", 1_020.0), ("bench_b", 2_010.0)]); // <2%

        let code = run_bench_diff(&baseline, &current, 0.10, false);
        assert_eq!(code, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_bench_diff_with_regression_returns_one() {
        let dir = temp_dir("diff_reg");
        let baseline = dir.join("baseline.json");
        let current = dir.join("current.json");

        write_snapshot(&baseline, &[("bench_a", 1_000.0)]);
        write_snapshot(&current, &[("bench_a", 1_500.0)]); // +50%

        let code = run_bench_diff(&baseline, &current, 0.10, false);
        assert_eq!(code, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_bench_diff_full_flag_does_not_panic() {
        let dir = temp_dir("diff_full");
        let baseline = dir.join("baseline.json");
        let current = dir.join("current.json");

        write_snapshot(&baseline, &[("bench_a", 1_000.0)]);
        write_snapshot(&current, &[("bench_a", 800.0)]); // improvement

        let code = run_bench_diff(&baseline, &current, 0.10, true);
        assert_eq!(code, 0); // improvement, no regression

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_bench_diff_missing_baseline_returns_one() {
        let dir = temp_dir("diff_no_base");
        let current = dir.join("current.json");
        write_snapshot(&current, &[("bench_a", 1_000.0)]);

        let code = run_bench_diff(
            Path::new("/nonexistent/baseline.json"),
            &current,
            0.10,
            false,
        );
        assert_eq!(code, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // dep-audit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_run_dep_audit_clean_returns_zero() {
        let dir = temp_dir("da_clean");
        fs::write(
            dir.join("Cargo.lock"),
            "version = 3\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n",
        )
        .expect("write lock");
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = []\n\n[workspace.dependencies]\nserde = \"1.0\"\n",
        )
        .expect("write toml");

        let code = run_dep_audit(&dir, None, false);
        assert_eq!(code, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_strict_with_banned_returns_one() {
        let dir = temp_dir("da_banned");
        fs::write(
            dir.join("Cargo.lock"),
            "version = 3\n\n[[package]]\nname = \"bincode\"\nversion = \"1.3.3\"\n",
        )
        .expect("write lock");
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = []\n\n[workspace.dependencies]\nbincode = \"1.0\"\n",
        )
        .expect("write toml");

        let code = run_dep_audit(&dir, None, true);
        assert_eq!(code, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_dep_audit_non_strict_with_banned_returns_zero() {
        let dir = temp_dir("da_banned_nonstrict");
        fs::write(
            dir.join("Cargo.lock"),
            "version = 3\n\n[[package]]\nname = \"bincode\"\nversion = \"1.3.3\"\n",
        )
        .expect("write lock");
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = []\n\n[workspace.dependencies]\nbincode = \"1.0\"\n",
        )
        .expect("write toml");

        let code = run_dep_audit(&dir, None, false);
        assert_eq!(code, 0); // Not strict — always 0 even with banned deps

        let _ = fs::remove_dir_all(&dir);
    }
}
