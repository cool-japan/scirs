//! Python interoperability module for migration assistance
//!
//! This module provides helpers for users migrating from SciPy to SciRS2,
//! including compatibility layers and migration guides.

#![allow(dead_code)]

use crate::{bessel, erf, gamma, statistical};
use std::collections::HashMap;

/// Migration guide for common SciPy special functions to SciRS2
pub struct MigrationGuide {
    mappings: HashMap<String, FunctionMapping>,
    reverse_mappings: HashMap<String, String>, // SciRS2 -> SciPy
}

/// Represents a function mapping from SciPy to SciRS2
#[derive(Clone)]
pub struct FunctionMapping {
    pub scipy_name: String,
    pub scirs2_name: String,
    pub module_path: String,
    pub signature_changes: Vec<String>,
    pub notes: Vec<String>,
}

impl Default for MigrationGuide {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationGuide {
    /// Create a comprehensive migration guide
    pub fn new() -> Self {
        let mut mappings = HashMap::new();
        let mut reverse_mappings = HashMap::new();

        // Gamma functions
        mappings.insert(
            "scipy.special.gamma".to_string(),
            FunctionMapping {
                scipy_name: "gamma".to_string(),
                scirs2_name: "gamma".to_string(),
                module_path: "scirs2_special::gamma".to_string(),
                signature_changes: vec![],
                notes: vec![
                    "Direct replacement, same signature".to_string(),
                    "Returns NaN for negative integers (poles)".to_string(),
                ],
            },
        );

        mappings.insert(
            "scipy.special.gammaln".to_string(),
            FunctionMapping {
                scipy_name: "gammaln".to_string(),
                scirs2_name: "gammaln".to_string(),
                module_path: "scirs2_special::gamma".to_string(),
                signature_changes: vec![],
                notes: vec!["Direct replacement".to_string()],
            },
        );

        mappings.insert(
            "scipy.special.beta".to_string(),
            FunctionMapping {
                scipy_name: "beta".to_string(),
                scirs2_name: "beta".to_string(),
                module_path: "scirs2_special::gamma".to_string(),
                signature_changes: vec![],
                notes: vec!["Direct replacement".to_string()],
            },
        );

        // Bessel functions
        mappings.insert(
            "scipy.special.j0".to_string(),
            FunctionMapping {
                scipy_name: "j0".to_string(),
                scirs2_name: "j0".to_string(),
                module_path: "scirs2_special::bessel".to_string(),
                signature_changes: vec![],
                notes: vec!["Direct replacement".to_string()],
            },
        );

        mappings.insert(
            "scipy.special.jv".to_string(),
            FunctionMapping {
                scipy_name: "jv".to_string(),
                scirs2_name: "jv".to_string(),
                module_path: "scirs2_special::bessel".to_string(),
                signature_changes: vec!["Order parameter v must implement Float trait".to_string()],
                notes: vec!["Supports both integer and fractional orders".to_string()],
            },
        );

        // Error functions
        mappings.insert(
            "scipy.special.erf".to_string(),
            FunctionMapping {
                scipy_name: "erf".to_string(),
                scirs2_name: "erf".to_string(),
                module_path: "scirs2_special::erf".to_string(),
                signature_changes: vec![],
                notes: vec![
                    "Direct replacement".to_string(),
                    "Complex version available as erf_complex".to_string(),
                ],
            },
        );

        mappings.insert(
            "scipy.special.erfc".to_string(),
            FunctionMapping {
                scipy_name: "erfc".to_string(),
                scirs2_name: "erfc".to_string(),
                module_path: "scirs2_special::erf".to_string(),
                signature_changes: vec![],
                notes: vec!["Direct replacement".to_string()],
            },
        );

        // Statistical functions
        mappings.insert(
            "scipy.special.expit".to_string(),
            FunctionMapping {
                scipy_name: "expit".to_string(),
                scirs2_name: "logistic".to_string(),
                module_path: "scirs2_special::statistical".to_string(),
                signature_changes: vec![],
                notes: vec![
                    "Name change: expit -> logistic".to_string(),
                    "Same mathematical function: 1/(1+exp(-x))".to_string(),
                ],
            },
        );

        mappings.insert(
            "scipy.special.softmax".to_string(),
            FunctionMapping {
                scipy_name: "softmax".to_string(),
                scirs2_name: "softmax".to_string(),
                module_path: "scirs2_special::statistical".to_string(),
                signature_changes: vec![
                    "Takes &[f64] slice instead of numpy array".to_string(),
                    "Returns Vec<f64> instead of numpy array".to_string(),
                ],
                notes: vec!["Use ndarray for array operations".to_string()],
            },
        );

        // Orthogonal polynomials
        mappings.insert(
            "scipy.special.legendre".to_string(),
            FunctionMapping {
                scipy_name: "legendre".to_string(),
                scirs2_name: "legendre_p".to_string(),
                module_path: "scirs2_special::orthogonal".to_string(),
                signature_changes: vec![
                    "Returns polynomial value, not polynomial object".to_string(),
                    "Use legendre_p(n, x) for evaluation".to_string(),
                ],
                notes: vec![
                    "SciPy returns polynomial object, SciRS2 evaluates directly".to_string()
                ],
            },
        );

        // Additional mappings for comprehensive coverage

        // Airy functions
        mappings.insert(
            "scipy.special.airy".to_string(),
            FunctionMapping {
                scipy_name: "airy".to_string(),
                scirs2_name: "ai, bi, aip, bip".to_string(),
                module_path: "scirs2_special::airy".to_string(),
                signature_changes: vec![
                    "SciPy returns tuple (Ai, Aip, Bi, Bip)".to_string(),
                    "SciRS2 has separate functions for each".to_string(),
                ],
                notes: vec!["Use individual functions: ai(x), bi(x), aip(x), bip(x)".to_string()],
            },
        );

        // Elliptic functions
        mappings.insert(
            "scipy.special.ellipk".to_string(),
            FunctionMapping {
                scipy_name: "ellipk".to_string(),
                scirs2_name: "elliptic_k".to_string(),
                module_path: "scirs2_special::elliptic".to_string(),
                signature_changes: vec![],
                notes: vec!["Name change: ellipk -> elliptic_k".to_string()],
            },
        );

        // Hypergeometric functions
        mappings.insert(
            "scipy.special.hyp1f1".to_string(),
            FunctionMapping {
                scipy_name: "hyp1f1".to_string(),
                scirs2_name: "hyp1f1".to_string(),
                module_path: "scirs2_special::hypergeometric".to_string(),
                signature_changes: vec![],
                notes: vec!["Direct replacement".to_string()],
            },
        );

        // Spherical harmonics
        mappings.insert(
            "scipy.special.sph_harm".to_string(),
            FunctionMapping {
                scipy_name: "sph_harm".to_string(),
                scirs2_name: "sph_harm_complex".to_string(),
                module_path: "scirs2_special::spherical_harmonics".to_string(),
                signature_changes: vec![
                    "Parameter order: (m, n, theta, phi) in SciPy".to_string(),
                    "Parameter order: (n, m, theta, phi) in SciRS2".to_string(),
                ],
                notes: vec!["Watch out for parameter order change".to_string()],
            },
        );

        // Build reverse mappings
        for (scipy_name, mapping) in &mappings {
            reverse_mappings.insert(mapping.scirs2_name.clone(), scipy_name.clone());
        }

        MigrationGuide {
            mappings,
            reverse_mappings,
        }
    }

    /// Get mapping for a SciPy function
    pub fn get_mapping(&self, scipyfunc: &str) -> Option<&FunctionMapping> {
        self.mappings.get(scipyfunc)
    }

    /// Get reverse mapping (SciRS2 to SciPy)
    pub fn get_reverse_mapping(&self, scirs2func: &str) -> Option<&String> {
        self.reverse_mappings.get(scirs2func)
    }

    /// List all available mappings
    pub fn list_all_mappings(&self) -> Vec<(&String, &FunctionMapping)> {
        self.mappings.iter().collect()
    }

    /// Generate migration report for a list of SciPy functions
    pub fn generate_migration_report(&self, scipyfunctions: &[&str]) -> String {
        let mut report = String::from("SciPy to SciRS2 Migration Report\n");
        report.push_str("================================\n\n");

        for &func in scipyfunctions {
            if let Some(mapping) = self.get_mapping(func) {
                report.push_str(&format!("## {func}\n"));
                report.push_str(&format!("SciRS2 equivalent: `{}`\n", mapping.module_path));

                if !mapping.signature_changes.is_empty() {
                    report.push_str("\nSignature changes:\n");
                    for change in &mapping.signature_changes {
                        report.push_str(&format!("- {change}\n"));
                    }
                }

                if !mapping.notes.is_empty() {
                    report.push_str("\nNotes:\n");
                    for note in &mapping.notes {
                        report.push_str(&format!("- {note}\n"));
                    }
                }

                report.push('\n');
            } else {
                report.push_str(&format!("## {func}\n"));
                report.push_str(
                    "⚠️  No direct mapping found. May require custom implementation.\n\n",
                );
            }
        }

        report
    }
}

/// Compatibility layer providing SciPy-like function signatures
pub mod compat {
    use super::*;
    use scirs2_core::ndarray::{Array1, ArrayView1};

    /// SciPy-compatible gamma function for arrays
    pub fn gamma_array(x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(gamma::gamma)
    }

    /// SciPy-compatible erf function for arrays
    pub fn erf_array(x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(erf::erf)
    }

    /// SciPy-compatible j0 function for arrays
    pub fn j0_array(x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(bessel::j0)
    }

    /// SciPy-compatible softmax with axis parameter
    pub fn softmax_axis(x: &ArrayView1<f64>, _axis: Option<usize>) -> Vec<f64> {
        // Note: This is simplified for 1D arrays
        // Full implementation would handle multi-dimensional arrays
        match statistical::softmax(x.view()) {
            Ok(result) => result.to_vec(),
            Err(_) => vec![],
        }
    }
}

/// Code generation helpers for migration
pub mod codegen {
    use super::*;
    #[cfg(feature = "python-interop")]
    use regex::Regex;

    /// Generate Rust code equivalent to SciPy code
    pub fn generate_rust_equivalent(_scipycode: &str) -> Result<String, String> {
        #[cfg(feature = "python-interop")]
        {
            generate_rust_equivalent_regex(_scipycode)
        }

        #[cfg(not(feature = "python-interop"))]
        {
            generate_rust_equivalent_simple(_scipycode)
        }
    }

    #[cfg(feature = "python-interop")]
    fn generate_rust_equivalent_regex(_scipycode: &str) -> Result<String, String> {
        let guide = MigrationGuide::new();
        let mut rust_code = String::new();
        let mut imports = std::collections::HashSet::new();
        let mut code_lines = Vec::new();

        // Regex patterns for common SciPy function calls
        let patterns = vec![
            (r"scipy\.special\.(\w+)\s*\(", "scipy.special."),
            (r"from scipy\.special import (\w+(?:,\s*\w+)*)", ""),
            (r"special\.(\w+)\s*\(", "scipy.special."),
        ];

        // Extract function names
        let mut found_functions = Vec::new();
        for (pattern, prefix) in patterns {
            let re = Regex::new(pattern).map_err(|e| e.to_string())?;
            for cap in re.captures_iter(_scipycode) {
                if let Some(func_match) = cap.get(1) {
                    let funcs = func_match.as_str();
                    for func in funcs.split(',') {
                        let func = func.trim();
                        let full_name = format!("{prefix}{func}");
                        if let Some(mapping) = guide.get_mapping(&full_name) {
                            found_functions.push((func.to_string(), mapping.clone()));
                            imports.insert(mapping.module_path.clone());
                        }
                    }
                }
            }
        }

        // Generate imports
        for import in &imports {
            rust_code.push_str(&format!("use {import};\n"));
        }

        if !imports.is_empty() {
            rust_code.push('\n');
        }

        // Generate _code transformation hints
        let mut transformed = _scipycode.to_string();
        for (scipyfunc, mapping) in &found_functions {
            // Add transformation comments
            code_lines.push(format!("// {} -> {}", scipyfunc, mapping.scirs2_name));

            // Simple replacement (this is a simplified example)
            transformed =
                transformed.replace(&format!("scipy.special.{scipyfunc}"), &mapping.scirs2_name);
            transformed =
                transformed.replace(&format!("special.{scipyfunc}"), &mapping.scirs2_name);
        }

        // Add transformation notes
        if !found_functions.is_empty() {
            rust_code.push_str("// Transformed _code:\n");
            rust_code.push_str(&format!("// {transformed}\n"));

            rust_code.push_str("\n// Notes:\n");
            for (_, mapping) in &found_functions {
                for note in &mapping.notes {
                    rust_code.push_str(&format!("// - {note}\n"));
                }
            }
        }

        if rust_code.is_empty() {
            Err("No recognized SciPy functions found".to_string())
        } else {
            Ok(rust_code)
        }
    }

    fn generate_rust_equivalent_simple(_scipycode: &str) -> Result<String, String> {
        let guide = MigrationGuide::new();
        let mut rust_code = String::new();

        // Simple pattern matching for common cases without regex
        let known_functions = vec!["gamma", "erf", "j0", "j1", "beta", "gammaln"];

        for func in known_functions {
            let scipy_pattern = format!("scipy.special.{func}");
            if _scipycode.contains(&scipy_pattern) {
                let full_name = format!("scipy.special.{func}");
                if let Some(mapping) = guide.get_mapping(&full_name) {
                    let module_path = &mapping.module_path;
                    rust_code.push_str(&format!("use {module_path};\n"));
                    let scirs2_name = &mapping.scirs2_name;
                    rust_code.push_str(&format!("// Replace {scipy_pattern} with {scirs2_name}\n"));
                }
            }
        }

        if rust_code.is_empty() {
            Err("No recognized SciPy functions found".to_string())
        } else {
            Ok(rust_code)
        }
    }

    /// Generate import statements for common migrations
    pub fn generate_imports(_scipyimports: &[&str]) -> String {
        let mut _imports = String::from("// SciRS2 _imports\n");

        for &import in _scipyimports {
            match import {
                "gamma" | "gammaln" | "beta" => {
                    _imports.push_str("use scirs2_special::gamma::{gamma, gammaln, beta};\n");
                }
                "j0" | "j1" | "jv" => {
                    _imports.push_str("use scirs2_special::bessel::{j0, j1, jv};\n");
                }
                "erf" | "erfc" => {
                    _imports.push_str("use scirs2_special::erf::{erf, erfc};\n");
                }
                _ => {}
            }
        }

        _imports
    }
}

/// Performance comparison utilities
pub mod performance {
    /// Structure to hold performance comparison data
    pub struct PerformanceComparison {
        pub function_name: String,
        pub scipy_time_ms: f64,
        pub scirs2_time_ms: f64,
        pub speedup: f64,
        pub accuracy_difference: f64,
    }

    impl PerformanceComparison {
        /// Generate a performance report
        pub fn report(&self) -> String {
            format!(
                "{}: SciRS2 is {:.1}x {} (accuracy diff: {:.2e})",
                self.function_name,
                if self.speedup > 1.0 {
                    self.speedup
                } else {
                    1.0 / self.speedup
                },
                if self.speedup > 1.0 {
                    "faster"
                } else {
                    "slower"
                },
                self.accuracy_difference
            )
        }
    }
}

/// Migration examples
pub mod examples {
    /// Example: Migrating gamma function usage
    pub fn gamma_migration_example() -> String {
        r#"
// SciPy Python code:
// from scipy.special import gamma
// result = gamma(5.5)

// SciRS2 Rust equivalent:
use scirs2_special::gamma;

let result = gamma(5.5_f64);

// For array operations:
use scirs2_core::ndarray::Array1;

let x = Array1::linspace(0.1, 10.0, 100);
let gamma_values = x.mapv(gamma);
"#
        .to_string()
    }

    /// Example: Migrating Bessel function usage
    pub fn bessel_migration_example() -> String {
        r#"
// SciPy Python code:
// from scipy.special import j0, jv
// y1 = j0(2.5)
// y2 = jv(1.5, 3.0)

// SciRS2 Rust equivalent:
use scirs2_special::bessel::{j0, jv};

let y1 = j0(2.5_f64);
let y2 = jv(1.5_f64, 3.0_f64);
"#
        .to_string()
    }

    /// Example: Migrating statistical functions
    pub fn statistical_migration_example() -> String {
        r#"
// SciPy Python code:
// from scipy.special import expit, softmax
// sigmoid = expit(x)
// probs = softmax(logits)

// SciRS2 Rust equivalent:
use scirs2_special::statistical::{logistic, softmax};

let sigmoid = logistic(x);
let probs = softmax(&logits);  // Note: takes a slice

// For ndarray:
use scirs2_core::ndarray::Array1;

let x_array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let sigmoid_array = x_array.mapv(logistic);
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_guide() {
        let guide = MigrationGuide::new();

        // Test known mappings
        let gamma_mapping = guide.get_mapping("scipy.special.gamma").unwrap();
        assert_eq!(gamma_mapping.scirs2_name, "gamma");

        let expit_mapping = guide.get_mapping("scipy.special.expit").unwrap();
        assert_eq!(expit_mapping.scirs2_name, "logistic");
    }

    #[test]
    fn test_migration_report() {
        let guide = MigrationGuide::new();
        let functions = vec![
            "scipy.special.gamma",
            "scipy.special.erf",
            "scipy.special.unknown",
        ];
        let report = guide.generate_migration_report(&functions);

        assert!(report.contains("gamma"));
        assert!(report.contains("erf"));
        assert!(report.contains("No direct mapping found"));
    }

    #[test]
    fn test_codegen() {
        let _scipycode = "result = scipy.special.gamma(x)";
        let rust_code = codegen::generate_rust_equivalent(_scipycode).unwrap();

        assert!(rust_code.contains("use scirs2_special::gamma"));
    }

    #[test]
    fn test_performance_comparison() {
        let comparison = performance::PerformanceComparison {
            function_name: "gamma".to_string(),
            scipy_time_ms: 10.0,
            scirs2_time_ms: 2.0,
            speedup: 5.0,
            accuracy_difference: 1e-15,
        };

        let report = comparison.report();
        assert!(report.contains("5.0x faster"));
    }
}
