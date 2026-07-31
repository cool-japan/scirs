//! Cross-validation against reference implementations
//!
//! This module provides comprehensive validation of special functions
//! against multiple reference implementations including SciPy, GSL,
//! and high-precision arbitrary precision libraries.

use crate::error::SpecialResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;

/// Reference implementation sources
#[derive(Debug, Clone, Copy)]
pub enum ReferenceSource {
    SciPy,
    GSL,
    Mathematica,
    MPFR,
    Boost,
}

/// Test case for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub function: String,
    pub inputs: Vec<f64>,
    pub expected: f64,
    pub source: String,
    pub tolerance: f64,
}

/// Validation result for a single test
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub test_case: TestCase,
    pub computed: f64,
    pub error: f64,
    pub relative_error: f64,
    pub ulp_error: i64,
    pub passed: bool,
}

/// Summary of validation results
#[derive(Debug)]
pub struct ValidationSummary {
    pub function: String,
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub max_error: f64,
    pub mean_error: f64,
    pub max_ulp_error: i64,
    pub failed_cases: Vec<ValidationResult>,
}

/// Cross-validation framework
pub struct CrossValidator {
    test_cases: HashMap<String, Vec<TestCase>>,
    results: HashMap<String, Vec<ValidationResult>>,
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossValidator {
    pub fn new() -> Self {
        Self {
            test_cases: HashMap::new(),
            results: HashMap::new(),
        }
    }

    /// Load test cases from reference implementations
    pub fn load_test_cases(&mut self) -> SpecialResult<()> {
        // Load SciPy reference values
        self.load_scipy_references()?;

        // Load GSL reference values
        self.load_gsl_references()?;

        // Load high-precision reference values
        self.load_mpfr_references()?;

        Ok(())
    }

    /// Load reference values from SciPy
    fn load_scipy_references(&mut self) -> SpecialResult<()> {
        // This would typically read from a file or run a Python script
        // For now, we'll add some hardcoded test cases

        let gamma_tests = vec![
            TestCase {
                function: "gamma".to_string(),
                inputs: vec![0.5],
                expected: 1.7724538509055159, // sqrt(pi)
                source: "SciPy".to_string(),
                tolerance: 1e-15,
            },
            TestCase {
                function: "gamma".to_string(),
                inputs: vec![5.0],
                expected: 24.0,
                source: "SciPy".to_string(),
                tolerance: 1e-15,
            },
            TestCase {
                function: "gamma".to_string(),
                inputs: vec![10.5],
                expected: 1133278.3889487855,
                source: "SciPy".to_string(),
                tolerance: 1e-10,
            },
        ];

        self.test_cases.insert("gamma".to_string(), gamma_tests);

        let bessel_tests = vec![
            TestCase {
                function: "j0".to_string(),
                inputs: vec![1.0],
                expected: 0.7651976865579666,
                source: "SciPy".to_string(),
                tolerance: 1e-15,
            },
            TestCase {
                function: "j0".to_string(),
                inputs: vec![10.0],
                expected: -0.245_935_764_451_348_3,
                source: "SciPy".to_string(),
                tolerance: 1e-15,
            },
        ];

        self.test_cases
            .insert("bessel_j0".to_string(), bessel_tests);

        Ok(())
    }

    /// Load reference values from GSL
    fn load_gsl_references(&mut self) -> SpecialResult<()> {
        // Additional test cases from GNU Scientific Library
        let erf_tests = vec![
            TestCase {
                function: "erf".to_string(),
                inputs: vec![1.0],
                expected: 0.8427007929497149,
                source: "GSL".to_string(),
                tolerance: 1e-15,
            },
            TestCase {
                function: "erf".to_string(),
                inputs: vec![2.0],
                expected: 0.9953222650189527,
                source: "GSL".to_string(),
                tolerance: 1e-15,
            },
        ];

        self.test_cases
            .entry("erf".to_string())
            .or_default()
            .extend(erf_tests);

        Ok(())
    }

    /// Load high-precision reference values from MPFR
    fn load_mpfr_references(&mut self) -> SpecialResult<()> {
        // High-precision test cases for edge cases
        let edge_cases = vec![
            TestCase {
                function: "gamma".to_string(),
                inputs: vec![1e-10],
                expected: 9999999999.422784,
                source: "MPFR".to_string(),
                tolerance: 1e-6,
            },
            TestCase {
                function: "gamma".to_string(),
                inputs: vec![170.5],
                // Γ(170.5) ≈ 5.5620924145599996...e305 (verified against
                // Python's math.gamma/mpmath at 50 decimal digits). The
                // previous constant here, ~4.27e304, was simply wrong test
                // data -- off by a factor of ~13, and coincidentally close to
                // Γ(170.0) instead. It went unnoticed because the `gamma()`
                // implementation independently overflowed to `inf` for this
                // input (two separate bugs, both now fixed in gamma/core.rs
                // and gamma/approximations.rs).
                expected: 5.562_092_414_56e305,
                source: "MPFR".to_string(),
                tolerance: 1e-10,
            },
        ];

        self.test_cases
            .entry("gamma".to_string())
            .or_default()
            .extend(edge_cases);

        Ok(())
    }

    /// Run validation for a specific function
    pub fn validate_function<F>(&mut self, name: &str, func: F) -> ValidationSummary
    where
        F: Fn(&[f64]) -> f64,
    {
        let test_cases = self.test_cases.get(name).cloned().unwrap_or_default();
        let mut all_results = Vec::new();
        let mut errors = Vec::new();
        let mut ulp_errors = Vec::new();

        for test in test_cases {
            let computed = func(&test.inputs);
            let error = (computed - test.expected).abs();
            let relative_error = if test.expected != 0.0 {
                error / test.expected.abs()
            } else {
                error
            };

            let ulp_error = compute_ulp_error(computed, test.expected);
            // Authoritative per-case verdict: each test case's OWN
            // (deliberately tuned) relative tolerance, not a hardcoded
            // absolute threshold.
            let passed = relative_error <= test.tolerance;

            all_results.push(ValidationResult {
                test_case: test.clone(),
                computed,
                error,
                relative_error,
                ulp_error,
                passed,
            });

            errors.push(error);
            ulp_errors.push(ulp_error);
        }

        // Persist the FULL result set (not just failures) so
        // `generate_report` -- which reads `self.results` -- has real data
        // to report on, instead of always iterating zero entries.
        self.results.insert(name.to_string(), all_results.clone());

        let total = all_results.len();
        // Derive `passed`/`failed` from the SAME authoritative per-case
        // `passed` flag used for `failed_cases` below, instead of a
        // hardcoded absolute-error threshold that could (and did)
        // disagree with it for large-magnitude expected values where a
        // tiny, well-within-tolerance *relative* error is still an
        // enormous *absolute* error.
        let passed = all_results.iter().filter(|r| r.passed).count();
        let failed_cases: Vec<ValidationResult> =
            all_results.into_iter().filter(|r| !r.passed).collect();

        ValidationSummary {
            function: name.to_string(),
            total_tests: total,
            passed,
            failed: total - passed,
            max_error: errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mean_error: if total == 0 {
                0.0
            } else {
                errors.iter().sum::<f64>() / total as f64
            },
            max_ulp_error: ulp_errors.iter().cloned().max().unwrap_or(0),
            failed_cases,
        }
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("# Cross-Validation Report\n\n");

        for (function, results) in &self.results {
            report.push_str(&format!("## {function}\n\n"));

            // Summary statistics
            let total: usize = results.len();
            let passed = results.iter().filter(|r| r.passed).count();
            let failed = total - passed;

            report.push_str(&format!("- Total tests: {total}\n"));
            report.push_str(&format!(
                "- Passed: {passed} ({:.1}%)\n",
                100.0 * passed as f64 / total as f64
            ));
            report.push_str(&format!(
                "- Failed: {failed} ({:.1}%)\n",
                100.0 * failed as f64 / total as f64
            ));

            // Failed cases
            if failed > 0 {
                report.push_str("\n### Failed Cases\n\n");
                report.push_str(
                    "| Inputs | Expected | Computed | Rel Error | ULP Error | Source |\n",
                );
                report.push_str(
                    "|--------|----------|----------|-----------|-----------|--------|\n",
                );

                for result in results.iter().filter(|r| !r.passed).take(10) {
                    report.push_str(&format!(
                        "| {inputs:?} | {expected:.6e} | {computed:.6e} | {rel_error:.2e} | {ulp_error} | {source} |\n",
                        inputs = result.test_case.inputs,
                        expected = result.test_case.expected,
                        computed = result.computed,
                        rel_error = result.relative_error,
                        ulp_error = result.ulp_error,
                        source = result.test_case.source,
                    ));
                }

                if failed > 10 {
                    let more_failed = failed - 10;
                    report.push_str(&format!("\n... and {more_failed} more failed cases\n"));
                }
            }

            report.push('\n');
        }

        report
    }
}

/// Compute ULP (Units in Last Place) error
#[allow(dead_code)]
fn compute_ulp_error(a: f64, b: f64) -> i64 {
    if a == b {
        return 0;
    }

    let a_bits = a.to_bits();
    let b_bits = b.to_bits();

    // Use safe subtraction to avoid overflow
    if a_bits >= b_bits {
        (a_bits - b_bits) as i64
    } else {
        (b_bits - a_bits) as i64
    }
}

/// Python script runner for SciPy validation
pub struct PythonValidator {
    python_path: String,
}

impl Default for PythonValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonValidator {
    pub fn new() -> Self {
        Self {
            python_path: "python3".to_string(),
        }
    }

    /// Run Python script to compute reference values
    pub fn compute_reference(&self, function: &str, args: &[f64]) -> SpecialResult<f64> {
        let args_str = args
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let script = format!(
            r#"
import scipy.special as sp
import sys

result = sp.{function}({args_str})
print(result)
"#
        );

        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()
            .map_err(|e| crate::error::SpecialError::ComputationError(e.to_string()))?;

        if !output.status.success() {
            return Err(crate::error::SpecialError::ComputationError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        let result_str = String::from_utf8_lossy(&output.stdout);
        result_str
            .trim()
            .parse::<f64>()
            .map_err(|e| crate::error::SpecialError::ComputationError(e.to_string()))
    }
}

/// Automated test generation from reference implementations
#[allow(dead_code)]
pub fn generate_test_suite() -> SpecialResult<()> {
    let mut validator = CrossValidator::new();
    validator.load_test_cases()?;

    // Generate Rust test code
    let mut test_code = String::from("// Auto-generated cross-validation tests\n\n");
    test_code.push_str("#[cfg(test)]\nmod cross_validation_tests {\n");
    test_code.push_str("    use super::*;\n");
    test_code.push_str("    use approx::assert_relative_eq;\n\n");

    for (function, cases) in validator.test_cases {
        for (i, case) in cases.iter().enumerate() {
            let source_lower = case.source.to_lowercase();
            let input_str = case.inputs[0]
                .to_string()
                .replace('.', "_")
                .replace('-', "neg");
            let args_str = case
                .inputs
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            test_code.push_str(&format!(
                r#"
    #[test]
    fn test_{function}_{source_lower}_{i}_{input_str}() {{
        let result = {function}({args_str});
        assert_relative_eq!(result, {expected}, epsilon = {tolerance});
    }}
"#,
                expected = case.expected,
                tolerance = case.tolerance,
            ));
        }
    }

    test_code.push_str("}\n");

    std::fs::write("src/generated_cross_validation_tests.rs", test_code)
        .map_err(|e| crate::error::SpecialError::ComputationError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamma;

    #[test]
    fn test_cross_validator() {
        let mut validator = CrossValidator::new();
        validator.load_test_cases().expect("Operation failed");

        let summary = validator.validate_function("gamma", |args| gamma(args[0]));

        assert!(summary.total_tests > 0);
        assert!(summary.passed > 0);
        // `mean_error` is a mean of *absolute* errors (see `validate_function`),
        // and the loaded gamma test cases intentionally span ~1 to ~1e305 in
        // magnitude (including MPFR edge cases near the overflow boundary).
        // Even a tiny, fully-acceptable *relative* error (e.g. ~1e-14) on the
        // ~1e305-magnitude case translates into an enormous *absolute* error
        // that dominates the mean, so `mean_error < 1.0` is not a meaningful
        // bound here and must not be asserted directly (this is what the
        // original comment's "potential NaN/inf issues" was circling around --
        // in investigating it, two real overflow bugs were found and fixed in
        // `gamma()` itself, see gamma/core.rs and gamma/approximations.rs).
        // `mean_error` should still never be NaN, and each individual case is
        // checked against its own *relative*-error tolerance via `failed_cases`.
        assert!(
            summary.mean_error.is_finite(),
            "mean_error should never be NaN/inf for valid gamma inputs, got {}",
            summary.mean_error
        );
        assert!(
            summary.failed_cases.is_empty(),
            "some gamma test cases exceeded their tolerance: {:#?}",
            summary.failed_cases
        );

        // Regression guards for the summary/failed_cases mismatch bug:
        // `passed`/`failed` must be derived from the exact same per-case
        // verdicts as `failed_cases`, not a separate hardcoded threshold
        // that could disagree with it.
        assert_eq!(
            summary.failed,
            summary.failed_cases.len(),
            "summary.failed must equal the number of entries in failed_cases"
        );
        assert_eq!(
            summary.passed + summary.failed,
            summary.total_tests,
            "passed + failed must equal total_tests"
        );
    }

    /// Regression test for a specific, previously-real disagreement: the
    /// MPFR gamma(170.5) case (expected magnitude ~5.56e305, tolerance
    /// 1e-10 *relative*) can be comfortably within its relative tolerance
    /// while its *absolute* error is unavoidably far larger than 1e-10 --
    /// so a hardcoded `error <= 1e-10` absolute check would have counted
    /// it as failed while it was correctly absent from `failed_cases`
    /// (which uses the relative-tolerance-based `passed` flag). This test
    /// uses a synthetic function to isolate exactly that scenario without
    /// depending on the real `gamma()` implementation's current accuracy.
    #[test]
    fn test_validate_function_uses_relative_not_absolute_tolerance() {
        let mut validator = CrossValidator::new();
        validator.test_cases.insert(
            "huge_magnitude".to_string(),
            vec![TestCase {
                function: "huge_magnitude".to_string(),
                inputs: vec![1.0],
                expected: 5.0e305,
                source: "synthetic".to_string(),
                tolerance: 1e-10, // relative
            }],
        );

        // Computed value has relative error ~1e-12 (well within tolerance)
        // but absolute error ~5e293 (astronomically larger than 1e-10).
        let summary =
            validator.validate_function("huge_magnitude", |_| 5.0e305 * 1.000_000_000_001);

        assert_eq!(summary.total_tests, 1);
        assert_eq!(
            summary.passed, 1,
            "a tiny relative error at huge magnitude must count as passed"
        );
        assert_eq!(summary.failed, 0);
        assert!(
            summary.failed_cases.is_empty(),
            "must not be double-counted as failed via a hardcoded absolute threshold"
        );
    }

    /// Regression test for the dead `self.results` map: `generate_report`
    /// reads `self.results`, but `validate_function` used to never write
    /// to it, so the report always came out as just the two-line header
    /// regardless of how many functions were validated.
    #[test]
    fn test_generate_report_contains_validated_function() {
        let mut validator = CrossValidator::new();
        validator.load_test_cases().expect("Operation failed");
        validator.validate_function("gamma", |args| gamma(args[0]));

        let report = validator.generate_report();
        assert!(
            report.contains("## gamma"),
            "report must contain a section for the validated function, got:\n{report}"
        );
        assert!(report.contains("Total tests:"));
    }

    #[test]
    fn test_ulp_error() {
        assert_eq!(compute_ulp_error(1.0, 1.0), 0);
        assert!(compute_ulp_error(1.0, 1.0 + f64::EPSILON) <= 2);
    }
}
