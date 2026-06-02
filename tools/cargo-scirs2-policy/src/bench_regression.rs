//! Benchmark regression detection for cargo-scirs2-policy.
//!
//! This module provides utilities for:
//! - Loading Criterion benchmark output (`estimates.json`) into a [`BenchmarkSnapshot`]
//! - Saving and loading snapshots as JSON for cross-run comparison
//! - Comparing two snapshots to detect performance regressions or improvements
//! - Formatting human-readable markdown reports of benchmark changes
//!
//! # Criterion output format
//!
//! Criterion writes results to `target/criterion/<bench_name>/new/estimates.json`:
//! ```json
//! {
//!   "mean": { "point_estimate": 123.45, "standard_error": 1.23 },
//!   "std_dev": { "point_estimate": 5.67, "standard_error": 0.45 }
//! }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use crate::bench_regression::{BenchmarkSnapshot, compare_snapshots, format_diff_report};
//!
//! let baseline = BenchmarkSnapshot::load(Path::new("baseline.json")).unwrap();
//! let current = BenchmarkSnapshot::from_criterion_dir(Path::new("target/criterion")).unwrap();
//! let regressions = compare_snapshots(&baseline, &current, 0.10);
//! println!("{}", format_diff_report(&baseline, &current));
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Criterion JSON structures
// ---------------------------------------------------------------------------

/// A single statistical estimate as written by Criterion.
#[derive(Debug, Deserialize)]
struct CriterionEstimate {
    point_estimate: f64,
    standard_error: f64,
}

/// The top-level `estimates.json` structure written by Criterion.
#[derive(Debug, Deserialize)]
struct CriterionEstimates {
    mean: CriterionEstimate,
    std_dev: CriterionEstimate,
}

// ---------------------------------------------------------------------------
// Public measurement types
// ---------------------------------------------------------------------------

/// A single benchmark measurement extracted from Criterion output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// Benchmark name (taken from the Criterion group/benchmark directory name).
    pub name: String,
    /// Mean execution time in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation of execution time in nanoseconds.
    pub std_dev_ns: f64,
    /// Standard error of the mean estimate.
    pub std_err_ns: f64,
}

impl BenchmarkMeasurement {
    /// Construct from raw Criterion estimates.
    fn from_criterion(name: String, estimates: &CriterionEstimates) -> Self {
        Self {
            name,
            mean_ns: estimates.mean.point_estimate,
            std_dev_ns: estimates.std_dev.point_estimate,
            std_err_ns: estimates.mean.standard_error,
        }
    }
}

// ---------------------------------------------------------------------------
// BenchmarkSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of all benchmark measurements at a point in time.
///
/// Snapshots can be serialised to / deserialised from JSON, making them
/// suitable as CI artefacts that persist across pipeline runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSnapshot {
    /// ISO-8601 timestamp when the snapshot was taken.
    pub timestamp: String,
    /// Git commit hash at snapshot time, if available.
    pub git_hash: Option<String>,
    /// All measurements included in this snapshot.
    pub measurements: Vec<BenchmarkMeasurement>,
}

impl Default for BenchmarkSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkSnapshot {
    /// Create an empty snapshot with the current timestamp.
    pub fn new() -> Self {
        Self {
            timestamp: current_timestamp(),
            git_hash: read_git_hash(),
            measurements: Vec::new(),
        }
    }

    /// Load a snapshot from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error string if the file cannot be read or the JSON is malformed.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse snapshot JSON: {e}"))
    }

    /// Save this snapshot to a JSON file.
    ///
    /// Creates parent directories if they do not exist.
    ///
    /// # Errors
    ///
    /// Returns an error string if serialisation or the write fails.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent dir {}: {}", parent.display(), e))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialise snapshot: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    }

    /// Build a snapshot by scanning a Criterion output directory.
    ///
    /// Walks `criterion_dir` looking for `<bench_name>/new/estimates.json`
    /// files and parses each one.  The benchmark name is derived from the
    /// immediate subdirectory name under `criterion_dir`.
    ///
    /// Missing or unreadable files are silently skipped so a partial Criterion
    /// run still produces a useful snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error only when `criterion_dir` itself cannot be read.
    pub fn from_criterion_dir(criterion_dir: &Path) -> Result<Self, String> {
        if !criterion_dir.exists() {
            return Err(format!(
                "Criterion output directory not found: {}",
                criterion_dir.display()
            ));
        }

        let mut snapshot = Self::new();

        let entries = std::fs::read_dir(criterion_dir).map_err(|e| {
            format!(
                "Failed to read criterion directory {}: {}",
                criterion_dir.display(),
                e
            )
        })?;

        let mut bench_paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.path())
            .collect();

        bench_paths.sort();

        for bench_dir in bench_paths {
            let bench_name = match bench_dir.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Each bench directory may itself contain sub-groups: look recursively
            // for `new/estimates.json` up to two levels deep.
            collect_estimates_recursive(&bench_dir, &bench_name, &mut snapshot.measurements, 0);
        }

        Ok(snapshot)
    }

    /// Find a measurement by exact name.
    pub fn find(&self, name: &str) -> Option<&BenchmarkMeasurement> {
        self.measurements.iter().find(|m| m.name == name)
    }

    /// Returns the number of measurements in this snapshot.
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns `true` when there are no measurements.
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Criterion directory traversal helpers
// ---------------------------------------------------------------------------

/// Recursively walk a criterion bench directory looking for `new/estimates.json`.
///
/// `depth` limits recursion to avoid runaway traversal; we stop at depth 3.
fn collect_estimates_recursive(
    dir: &Path,
    name_prefix: &str,
    measurements: &mut Vec<BenchmarkMeasurement>,
    depth: usize,
) {
    if depth > 3 {
        return;
    }

    let estimates_path = dir.join("new").join("estimates.json");
    if estimates_path.exists() {
        match load_estimates_file(&estimates_path) {
            Ok(est) => {
                measurements.push(BenchmarkMeasurement::from_criterion(
                    name_prefix.to_string(),
                    &est,
                ));
            }
            Err(e) => {
                eprintln!("Warning: skipping {}: {}", estimates_path.display(), e);
            }
        }
        return;
    }

    // No estimates.json at this level — look for sub-directories (benchmark groups)
    let sub_entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut sub_dirs: Vec<PathBuf> = sub_entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();
    sub_dirs.sort();

    for sub in sub_dirs {
        let sub_name = match sub.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let full_name = format!("{name_prefix}/{sub_name}");
        collect_estimates_recursive(&sub, &full_name, measurements, depth + 1);
    }
}

/// Parse a Criterion `estimates.json` file.
fn load_estimates_file(path: &Path) -> Result<CriterionEstimates, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("JSON parse error: {e}"))
}

// ---------------------------------------------------------------------------
// Regression analysis
// ---------------------------------------------------------------------------

/// A change (regression or improvement) found when comparing two snapshots.
#[derive(Debug, Clone)]
pub struct RegressionEntry {
    /// Benchmark name.
    pub name: String,
    /// Baseline mean execution time in nanoseconds.
    pub baseline_mean_ns: f64,
    /// Current mean execution time in nanoseconds.
    pub current_mean_ns: f64,
    /// Relative change: `(current - baseline) / baseline`.
    ///
    /// Positive values indicate a slowdown; negative values indicate
    /// a speedup (improvement).
    pub relative_change: f64,
    /// Standard error of the current measurement in nanoseconds.
    pub current_std_err_ns: f64,
}

impl RegressionEntry {
    /// Returns `true` when this entry represents a performance regression
    /// beyond the given threshold (e.g. `0.10` for 10%).
    pub fn is_regression(&self, threshold: f64) -> bool {
        self.relative_change > threshold
    }

    /// Returns `true` when this entry represents a significant improvement
    /// beyond the given threshold.
    pub fn is_improvement(&self, threshold: f64) -> bool {
        self.relative_change < -threshold
    }

    /// Formats the relative change as a percentage string, e.g. `"+12.3%"`.
    pub fn change_pct_str(&self) -> String {
        format!("{:+.1}%", self.relative_change * 100.0)
    }
}

/// Compare two snapshots and return all entries where the current measurement
/// differs from the baseline by more than `threshold` (e.g. `0.10` = 10%).
///
/// Only benchmarks present in *both* snapshots are compared.  Benchmarks
/// present only in the current snapshot (new benchmarks) are ignored.
///
/// The returned list is sorted by `relative_change` descending (worst
/// regressions first).
pub fn compare_snapshots(
    baseline: &BenchmarkSnapshot,
    current: &BenchmarkSnapshot,
    threshold: f64,
) -> Vec<RegressionEntry> {
    // Only slowdowns (positive relative change above threshold) are regressions.
    // Improvements (negative relative change) are excluded from this list;
    // use `all_changes` to retrieve both directions.
    let mut entries: Vec<RegressionEntry> = current
        .measurements
        .iter()
        .filter_map(|m| {
            let base = baseline.find(&m.name)?;
            if base.mean_ns == 0.0 {
                return None;
            }
            let rel = (m.mean_ns - base.mean_ns) / base.mean_ns;
            if rel > threshold {
                Some(RegressionEntry {
                    name: m.name.clone(),
                    baseline_mean_ns: base.mean_ns,
                    current_mean_ns: m.mean_ns,
                    relative_change: rel,
                    current_std_err_ns: m.std_err_ns,
                })
            } else {
                None
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        b.relative_change
            .partial_cmp(&a.relative_change)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries
}

/// Return all significant changes between two snapshots, split into
/// `(regressions, improvements)`.
///
/// - `regressions`: entries where `relative_change > threshold` (slowdowns)
/// - `improvements`: entries where `relative_change < -threshold` (speedups)
///
/// Benchmarks within `±threshold` of zero are considered stable and are
/// excluded from both lists.
pub fn all_changes(
    baseline: &BenchmarkSnapshot,
    current: &BenchmarkSnapshot,
    threshold: f64,
) -> (Vec<RegressionEntry>, Vec<RegressionEntry>) {
    // Build the complete list of relative-change entries for all shared benchmarks
    let all_entries: Vec<RegressionEntry> = current
        .measurements
        .iter()
        .filter_map(|m| {
            let base = baseline.find(&m.name)?;
            if base.mean_ns == 0.0 {
                return None;
            }
            let rel = (m.mean_ns - base.mean_ns) / base.mean_ns;
            Some(RegressionEntry {
                name: m.name.clone(),
                baseline_mean_ns: base.mean_ns,
                current_mean_ns: m.mean_ns,
                relative_change: rel,
                current_std_err_ns: m.std_err_ns,
            })
        })
        .collect();

    let regressions: Vec<RegressionEntry> = all_entries
        .iter()
        .filter(|e| e.relative_change > threshold)
        .cloned()
        .collect();

    let improvements: Vec<RegressionEntry> = all_entries
        .into_iter()
        .filter(|e| e.relative_change < -threshold)
        .collect();

    (regressions, improvements)
}

// ---------------------------------------------------------------------------
// Report formatters
// ---------------------------------------------------------------------------

/// Noise floor for the diff report (below this, changes are "stable").
const DIFF_NOISE_FLOOR: f64 = 0.02;

/// Format a markdown table showing only regression entries.
///
/// Each row includes the standard-error of the current measurement so readers
/// can gauge statistical significance.  Returns `"No regressions found.\n"`
/// when the slice is empty.
pub fn format_regression_report(entries: &[RegressionEntry]) -> String {
    // Use is_regression to filter out any improvements that might have crept
    // in (e.g., if the caller passed a mixed slice).
    let regressions: Vec<&RegressionEntry> =
        entries.iter().filter(|e| e.is_regression(0.0)).collect();

    if regressions.is_empty() {
        return String::from("No regressions found.\n");
    }

    let mut out = String::new();
    out.push_str("## Performance Regressions\n\n");
    out.push_str("| Benchmark | Baseline | Current (±err) | Change |\n");
    out.push_str("|-----------|----------|----------------|--------|\n");

    for e in regressions {
        out.push_str(&format!(
            "| {} | {} | {} ±{} | {} |\n",
            e.name,
            format_ns(e.baseline_mean_ns),
            format_ns(e.current_mean_ns),
            format_ns(e.current_std_err_ns),
            e.change_pct_str(),
        ));
    }

    out.push('\n');
    out
}

/// Format a full markdown diff report: regressions, improvements, stable, and
/// new benchmarks.
///
/// Uses [`all_changes`] with a 2% noise floor to classify each measurement.
pub fn format_diff_report(baseline: &BenchmarkSnapshot, current: &BenchmarkSnapshot) -> String {
    let (mut regressions, mut improvements) = all_changes(baseline, current, DIFF_NOISE_FLOOR);

    // Stable = present in both but not in either regressions/improvements list
    let mut stable: Vec<RegressionEntry> = current
        .measurements
        .iter()
        .filter_map(|m| {
            let base = baseline.find(&m.name)?;
            if base.mean_ns == 0.0 {
                return None;
            }
            let rel = (m.mean_ns - base.mean_ns) / base.mean_ns;
            // Within noise floor
            if rel.abs() <= DIFF_NOISE_FLOOR {
                Some(RegressionEntry {
                    name: m.name.clone(),
                    baseline_mean_ns: base.mean_ns,
                    current_mean_ns: m.mean_ns,
                    relative_change: rel,
                    current_std_err_ns: m.std_err_ns,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort: regressions worst-first, improvements best-first, stable by name
    regressions.sort_by(|a, b| {
        b.relative_change
            .partial_cmp(&a.relative_change)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    improvements.sort_by(|a, b| {
        a.relative_change
            .partial_cmp(&b.relative_change)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stable.sort_by(|a, b| a.name.cmp(&b.name));

    let mut out = String::new();

    // Header
    out.push_str("# Benchmark Diff Report\n\n");
    out.push_str(&format!("- Baseline: `{}`\n", baseline.timestamp));
    out.push_str(&format!("- Current:  `{}`\n", current.timestamp));
    out.push_str(&format!(
        "- Changes:  {} regression(s), {} improvement(s), {} stable\n\n",
        regressions.len(),
        improvements.len(),
        stable.len()
    ));

    // Regressions
    if !regressions.is_empty() {
        out.push_str("## Regressions\n\n");
        out.push_str("| Benchmark | Baseline | Current (±err) | Change |\n");
        out.push_str("|-----------|----------|----------------|--------|\n");
        for e in &regressions {
            out.push_str(&format!(
                "| {} | {} | {} ±{} | **{}** |\n",
                e.name,
                format_ns(e.baseline_mean_ns),
                format_ns(e.current_mean_ns),
                format_ns(e.current_std_err_ns),
                e.change_pct_str(),
            ));
        }
        out.push('\n');
    }

    // Improvements
    if !improvements.is_empty() {
        out.push_str("## Improvements\n\n");
        out.push_str("| Benchmark | Baseline | Current (±err) | Change |\n");
        out.push_str("|-----------|----------|----------------|--------|\n");
        for e in &improvements {
            // use is_improvement to confirm sign before formatting
            debug_assert!(e.is_improvement(0.0));
            out.push_str(&format!(
                "| {} | {} | {} ±{} | **{}** |\n",
                e.name,
                format_ns(e.baseline_mean_ns),
                format_ns(e.current_mean_ns),
                format_ns(e.current_std_err_ns),
                e.change_pct_str(),
            ));
        }
        out.push('\n');
    }

    // Stable
    if !stable.is_empty() {
        out.push_str("## Stable (within ±2%)\n\n");
        out.push_str("| Benchmark | Baseline | Current | Change |\n");
        out.push_str("|-----------|----------|---------|--------|\n");
        for e in &stable {
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                e.name,
                format_ns(e.baseline_mean_ns),
                format_ns(e.current_mean_ns),
                e.change_pct_str(),
            ));
        }
        out.push('\n');
    }

    // New benchmarks (present in current, not in baseline)
    let new_benches: Vec<&BenchmarkMeasurement> = current
        .measurements
        .iter()
        .filter(|m| baseline.find(&m.name).is_none())
        .collect();

    if !new_benches.is_empty() {
        out.push_str("## New Benchmarks\n\n");
        out.push_str("| Benchmark | Mean (±err) | Std Dev |\n");
        out.push_str("|-----------|-------------|----------|\n");
        for m in new_benches {
            out.push_str(&format!(
                "| {} | {} ±{} | {} |\n",
                m.name,
                format_ns(m.mean_ns),
                format_ns(m.std_err_ns),
                format_ns(m.std_dev_ns),
            ));
        }
        out.push('\n');
    }

    out
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Format a nanosecond duration into a human-readable string.
///
/// Uses ns / µs / ms / s suffixes based on magnitude.
fn format_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2}µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.3}ms", ns / 1_000_000.0)
    } else {
        format!("{:.3}s", ns / 1_000_000_000.0)
    }
}

/// Return the current UTC time as an ISO-8601 string.
///
/// Falls back to a Unix timestamp when system time is unavailable.
fn current_timestamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => {
            // Simple manual ISO-8601 UTC derivation (avoids chrono dep)
            let total_secs = d.as_secs();
            let (y, mo, day, h, min, sec) = unix_secs_to_ymd_hms(total_secs);
            format!("{y:04}-{mo:02}-{day:02}T{h:02}:{min:02}:{sec:02}Z")
        }
        Err(_) => String::from("1970-01-01T00:00:00Z"),
    }
}

/// Decompose a Unix timestamp (seconds since epoch) into (year, month, day, hour, min, sec).
///
/// Gregorian calendar approximation; accurate for dates 1970–2100.
fn unix_secs_to_ymd_hms(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let sec = secs % 60;
    let mins = secs / 60;
    let min = mins % 60;
    let hours = mins / 60;
    let hour = hours % 24;
    let mut days = hours / 24; // days since 1970-01-01

    // Shift epoch to 1 Mar 0000 for easier calculation
    days += 719_468;
    let era = days / 146_097;
    let doe = days % 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    (year, month, day, hour, min, sec)
}

/// Attempt to read the current Git commit hash from `.git/HEAD`.
fn read_git_hash() -> Option<String> {
    // Walk up from cwd looking for .git
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let head = dir.join(".git").join("HEAD");
        if head.exists() {
            let content = std::fs::read_to_string(&head).ok()?;
            let content = content.trim();
            // Resolve symbolic ref
            if let Some(stripped) = content.strip_prefix("ref: ") {
                let ref_path = dir
                    .join(".git")
                    .join(stripped.replace('/', std::path::MAIN_SEPARATOR_STR));
                let hash = std::fs::read_to_string(ref_path).ok()?;
                return Some(hash.trim().to_string());
            }
            // Detached HEAD — content is the hash itself
            if content.len() == 40 {
                return Some(content.to_string());
            }
            return None;
        }
        if !dir.pop() {
            break;
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
            "bench_regression_{}_{}_{}",
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

    fn write_estimates(dir: &Path, bench_name: &str, mean_ns: f64, std_dev_ns: f64) {
        let bench_dir = dir.join(bench_name).join("new");
        fs::create_dir_all(&bench_dir).expect("create bench dir");
        let estimates = serde_json::json!({
            "mean": { "point_estimate": mean_ns, "standard_error": mean_ns * 0.01 },
            "std_dev": { "point_estimate": std_dev_ns, "standard_error": std_dev_ns * 0.05 }
        });
        fs::write(
            bench_dir.join("estimates.json"),
            serde_json::to_string_pretty(&estimates).expect("serialise"),
        )
        .expect("write estimates.json");
    }

    #[test]
    fn test_snapshot_new_has_timestamp() {
        let s = BenchmarkSnapshot::new();
        assert!(!s.timestamp.is_empty());
        assert!(s.measurements.is_empty());
    }

    #[test]
    fn test_snapshot_save_load_roundtrip() {
        let dir = temp_dir("save_load");
        let path = dir.join("snap.json");

        let mut snap = BenchmarkSnapshot::new();
        snap.measurements.push(BenchmarkMeasurement {
            name: "matmul".to_string(),
            mean_ns: 12345.6,
            std_dev_ns: 100.0,
            std_err_ns: 10.0,
        });

        snap.save(&path).expect("save snapshot");
        let loaded = BenchmarkSnapshot::load(&path).expect("load snapshot");

        assert_eq!(loaded.measurements.len(), 1);
        assert_eq!(loaded.measurements[0].name, "matmul");
        assert!((loaded.measurements[0].mean_ns - 12345.6).abs() < 1e-6);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_from_criterion_dir_parses_estimates() {
        let dir = temp_dir("criterion");
        write_estimates(&dir, "matmul_1024", 1_234_567.0, 50_000.0);
        write_estimates(&dir, "fft_4096", 987_654.0, 20_000.0);

        let snap = BenchmarkSnapshot::from_criterion_dir(&dir).expect("parse criterion dir");
        assert_eq!(snap.measurements.len(), 2);

        let matmul = snap.find("matmul_1024").expect("find matmul");
        assert!((matmul.mean_ns - 1_234_567.0).abs() < 1.0);

        let fft = snap.find("fft_4096").expect("find fft");
        assert!((fft.mean_ns - 987_654.0).abs() < 1.0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_from_criterion_dir_missing_returns_err() {
        let result = BenchmarkSnapshot::from_criterion_dir(Path::new("/nonexistent/criterion/dir"));
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_snapshots_detects_regression() {
        let mut baseline = BenchmarkSnapshot::new();
        baseline.measurements.push(BenchmarkMeasurement {
            name: "bench_a".to_string(),
            mean_ns: 1_000.0,
            std_dev_ns: 10.0,
            std_err_ns: 1.0,
        });

        let mut current = BenchmarkSnapshot::new();
        current.measurements.push(BenchmarkMeasurement {
            name: "bench_a".to_string(),
            mean_ns: 1_200.0, // 20% slower
            std_dev_ns: 12.0,
            std_err_ns: 1.2,
        });

        let regressions = compare_snapshots(&baseline, &current, 0.10);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].name, "bench_a");
        assert!((regressions[0].relative_change - 0.2).abs() < 1e-6);
        assert!(regressions[0].is_regression(0.10));
        assert!(!regressions[0].is_improvement(0.10));
    }

    #[test]
    fn test_compare_snapshots_no_regression_within_threshold() {
        let mut baseline = BenchmarkSnapshot::new();
        baseline.measurements.push(BenchmarkMeasurement {
            name: "bench_b".to_string(),
            mean_ns: 1_000.0,
            std_dev_ns: 10.0,
            std_err_ns: 1.0,
        });

        let mut current = BenchmarkSnapshot::new();
        current.measurements.push(BenchmarkMeasurement {
            name: "bench_b".to_string(),
            mean_ns: 1_050.0, // 5% — within 10% threshold
            std_dev_ns: 10.0,
            std_err_ns: 1.0,
        });

        let regressions = compare_snapshots(&baseline, &current, 0.10);
        assert!(regressions.is_empty());
    }

    #[test]
    fn test_compare_snapshots_ignores_new_benchmarks() {
        let baseline = BenchmarkSnapshot::new(); // empty

        let mut current = BenchmarkSnapshot::new();
        current.measurements.push(BenchmarkMeasurement {
            name: "new_bench".to_string(),
            mean_ns: 5_000.0,
            std_dev_ns: 50.0,
            std_err_ns: 5.0,
        });

        let regressions = compare_snapshots(&baseline, &current, 0.10);
        assert!(regressions.is_empty(), "New benchmarks should be ignored");
    }

    #[test]
    fn test_regression_entry_is_improvement() {
        let entry = RegressionEntry {
            name: "fast".to_string(),
            baseline_mean_ns: 1_000.0,
            current_mean_ns: 800.0,
            relative_change: -0.20,
            current_std_err_ns: 8.0,
        };
        assert!(entry.is_improvement(0.10));
        assert!(!entry.is_regression(0.10));
        assert_eq!(entry.change_pct_str(), "-20.0%");
    }

    #[test]
    fn test_format_regression_report_empty() {
        let report = format_regression_report(&[]);
        assert!(report.contains("No regressions"));
    }

    #[test]
    fn test_format_regression_report_nonempty() {
        let entries = vec![RegressionEntry {
            name: "matmul".to_string(),
            baseline_mean_ns: 1_000_000.0,
            current_mean_ns: 1_500_000.0,
            relative_change: 0.50,
            current_std_err_ns: 15_000.0,
        }];
        let report = format_regression_report(&entries);
        assert!(report.contains("matmul"));
        assert!(report.contains("+50.0%"));
        assert!(report.contains("ms"));
    }

    #[test]
    fn test_format_diff_report_empty_snapshots() {
        let b = BenchmarkSnapshot::new();
        let c = BenchmarkSnapshot::new();
        let report = format_diff_report(&b, &c);
        assert!(report.contains("Benchmark Diff Report"));
    }

    #[test]
    fn test_format_ns_ranges() {
        assert!(format_ns(500.0).ends_with("ns"));
        assert!(format_ns(5_000.0).contains('µ'));
        assert!(format_ns(5_000_000.0).contains("ms"));
        assert!(format_ns(5_000_000_000.0).ends_with('s'));
    }

    #[test]
    fn test_unix_timestamp_conversion() {
        // 2026-03-22T00:00:00Z = 1774137600 seconds since epoch
        let (y, mo, d, h, m, s) = unix_secs_to_ymd_hms(1_774_137_600);
        assert_eq!(y, 2026);
        assert_eq!(mo, 3);
        assert_eq!(d, 22);
        assert_eq!(h, 0);
        assert_eq!(m, 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_snapshot_find_returns_none_for_missing() {
        let snap = BenchmarkSnapshot::new();
        assert!(snap.find("nonexistent").is_none());
    }

    #[test]
    fn test_snapshot_len_and_is_empty() {
        let mut snap = BenchmarkSnapshot::new();
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
        snap.measurements.push(BenchmarkMeasurement {
            name: "x".to_string(),
            mean_ns: 1.0,
            std_dev_ns: 0.1,
            std_err_ns: 0.01,
        });
        assert!(!snap.is_empty());
        assert_eq!(snap.len(), 1);
    }
}
