//! Reporting utilities for policy violations.
//!
//! Provides both human-readable (text) and machine-readable (JSON) output
//! for collections of [`PolicyViolation`].

use crate::violation::{format_violation, PolicyViolation, Severity};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

/// Print all violations to stdout, grouped by crate then by file.
///
/// Output example:
/// ```text
/// === scirs2-linalg ===
///   /path/to/src/lib.rs:
///     [ERROR] [scirs2-linalg] /path/to/src/lib.rs:42: banned import …
///
/// 3 violations in 2 crates (2 errors, 1 warning)
/// ```
pub fn print_report(violations: &[PolicyViolation]) {
    if violations.is_empty() {
        println!("No policy violations found.");
        return;
    }

    // Group: crate_name → file → violations
    let mut by_crate: BTreeMap<&str, BTreeMap<String, Vec<&PolicyViolation>>> = BTreeMap::new();
    for v in violations {
        by_crate
            .entry(v.crate_name.as_str())
            .or_default()
            .entry(v.file.display().to_string())
            .or_default()
            .push(v);
    }

    for (crate_name, files) in &by_crate {
        println!("=== {} ===", crate_name);
        for (file, vs) in files {
            println!("  {}:", file);
            for v in vs {
                println!("    {}", format_violation(v));
            }
        }
        println!();
    }

    // Summary
    let errors = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();
    let warnings = violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .count();
    let infos = violations
        .iter()
        .filter(|v| v.severity == Severity::Info)
        .count();
    let crate_count = by_crate.len();
    println!(
        "{} violation(s) in {} crate(s) ({} errors, {} warnings, {} info)",
        violations.len(),
        crate_count,
        errors,
        warnings,
        infos,
    );
}

// ---------------------------------------------------------------------------
// JSON report
// ---------------------------------------------------------------------------

/// Serialise violations to a JSON string.
///
/// Each violation becomes a JSON object with fields:
/// `crate_name`, `file`, `line`, `severity`, `message`.
///
/// The output is a JSON array; it is always valid JSON even when empty.
pub fn json_report(violations: &[PolicyViolation]) -> String {
    if violations.is_empty() {
        return "[]".to_string();
    }
    let mut out = String::from("[\n");
    for (i, v) in violations.iter().enumerate() {
        let comma = if i + 1 < violations.len() { "," } else { "" };
        let file_escaped = v
            .file
            .display()
            .to_string()
            .replace('\\', "/")
            .replace('"', "\\\"");
        let msg_escaped = v.message.replace('"', "\\\"");
        out.push_str(&format!(
            "  {{\"crate_name\":\"{}\",\"file\":\"{}\",\"line\":{},\"severity\":\"{}\",\"message\":\"{}\"}}{}",
            v.crate_name, file_escaped, v.line, v.severity, msg_escaped, comma
        ));
        out.push('\n');
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::violation::{PolicyViolation, Severity};
    use std::path::PathBuf;

    fn make_violation(
        crate_name: &str,
        file: &str,
        line: usize,
        severity: Severity,
    ) -> PolicyViolation {
        PolicyViolation {
            crate_name: crate_name.to_string(),
            file: PathBuf::from(file),
            line,
            message: "some message".to_string(),
            severity,
        }
    }

    #[test]
    fn test_print_report_empty_does_not_panic() {
        // Just verify it doesn't panic
        print_report(&[]);
    }

    #[test]
    fn test_print_report_nonempty_does_not_panic() {
        let violations = vec![
            make_violation("my-crate", "/src/lib.rs", 10, Severity::Error),
            make_violation("my-crate", "/src/lib.rs", 20, Severity::Warning),
            make_violation("other-crate", "/src/main.rs", 5, Severity::Info),
        ];
        print_report(&violations);
    }

    #[test]
    fn test_json_report_empty() {
        let json = json_report(&[]);
        // Empty input produces exactly "[]"
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_json_report_valid_structure() {
        let violations = vec![make_violation(
            "crate-a",
            "/path/lib.rs",
            7,
            Severity::Error,
        )];
        let json = json_report(&violations);
        assert!(json.starts_with('['), "Should start with [");
        assert!(json.trim_end().ends_with(']'), "Should end with ]");
        assert!(
            json.contains("\"crate_name\":\"crate-a\""),
            "Should contain crate name"
        );
        assert!(json.contains("\"line\":7"), "Should contain line number");
        assert!(
            json.contains("\"severity\":\"ERROR\""),
            "Should contain severity"
        );
    }

    #[test]
    fn test_json_report_multiple_violations() {
        let violations = vec![
            make_violation("crate-a", "/a/lib.rs", 1, Severity::Error),
            make_violation("crate-b", "/b/lib.rs", 2, Severity::Warning),
        ];
        let json = json_report(&violations);
        // Both entries should appear
        assert!(json.contains("\"crate_name\":\"crate-a\""));
        assert!(json.contains("\"crate_name\":\"crate-b\""));
        // Trailing comma only on first element
        assert!(
            json.contains("},\n  {"),
            "First entry should have trailing comma"
        );
    }

    #[test]
    fn test_json_report_escapes_quotes_in_message() {
        let mut v = make_violation("c", "/f.rs", 1, Severity::Info);
        v.message = r#"has "quotes" inside"#.to_string();
        let json = json_report(&[v]);
        assert!(json.contains(r#"\"quotes\""#), "Quotes should be escaped");
    }
}
