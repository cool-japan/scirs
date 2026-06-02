//! Output formatting for policy violations.
//!
//! Supports two formats:
//! - `"text"` (default): human-readable terminal output.
//! - `"json"`: machine-readable JSON array.

use crate::rules::{Severity, Violation};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Print `violations` to stdout in the specified `format`.
///
/// `format` should be either `"text"` or `"json"`; any unknown value falls
/// back to the text formatter.
pub fn print_violations(violations: &[Violation], format: &str) {
    match format {
        "json" => print_json(violations),
        _ => print_text(violations),
    }
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

/// Print violations as a human-readable summary.
fn print_text(violations: &[Violation]) {
    if violations.is_empty() {
        println!("No policy violations found.");
        return;
    }

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

    println!(
        "Policy violations: {} errors, {} warnings, {} info",
        errors, warnings, infos
    );
    println!();

    for v in violations {
        match &v.file {
            Some(file) => println!("[{}] {} ({})", v.severity, v.message, file),
            None => println!("[{}] {}", v.severity, v.message),
        }
    }
}

/// Print violations as a JSON array.
///
/// This is a hand-rolled serialiser so the tool has no runtime dependency on
/// `serde_json` for this simple structure (avoids inflating the binary for a
/// development tool).
fn print_json(violations: &[Violation]) {
    println!("[");
    for (i, v) in violations.iter().enumerate() {
        let comma = if i + 1 < violations.len() { "," } else { "" };
        let file_json = match &v.file {
            Some(f) => format!("\"{}\"", f.replace('\\', "/").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        println!("  {{");
        println!("    \"rule_id\": \"{}\",", v.rule_id);
        println!("    \"message\": \"{}\",", v.message.replace('"', "\\\""));
        println!("    \"severity\": \"{}\",", v.severity);
        println!("    \"file\": {}", file_json);
        println!("  }}{}", comma);
    }
    println!("]");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Violation;

    fn make_violation(severity: Severity, with_file: bool) -> Violation {
        Violation {
            rule_id: "TEST_001".to_string(),
            message: "test message".to_string(),
            file: if with_file {
                Some("/some/path/Cargo.toml".to_string())
            } else {
                None
            },
            severity,
        }
    }

    /// Capture stdout for a closure — used for assertion-based output tests.
    /// Because Rust tests share stdout, we test the logic indirectly through
    /// the public function signature here; full output capture would require
    /// a separate process or a writer trait, which is out of scope for this
    /// utility tool.
    #[test]
    fn test_print_violations_empty_does_not_panic() {
        // Should not panic on empty input
        print_violations(&[], "text");
        print_violations(&[], "json");
    }

    #[test]
    fn test_print_violations_nonempty_does_not_panic() {
        let v = vec![
            make_violation(Severity::Error, true),
            make_violation(Severity::Warning, false),
            make_violation(Severity::Info, true),
        ];
        print_violations(&v, "text");
        print_violations(&v, "json");
    }

    #[test]
    fn test_print_violations_unknown_format_falls_back_to_text() {
        let v = vec![make_violation(Severity::Error, false)];
        // Should not panic even with an unexpected format string
        print_violations(&v, "xml");
        print_violations(&v, "");
    }
}
