//! Comprehensive security auditor for critical optimization paths
//!
//! This module provides advanced security analysis for machine learning optimizers,
//! focusing on input validation, privacy guarantees, memory safety, and numerical stability.
//! The module has been refactored into focused submodules for better maintainability.

use crate::error::{OptimError, Result};
use std::time::{Duration, Instant};

// Re-export all public items from submodules
pub mod config;
pub mod types;
pub mod input_validation;
pub mod privacy;
pub mod memory_safety;
pub mod results;

pub use config::*;
pub use types::*;
pub use input_validation::*;
pub use privacy::*;
pub use memory_safety::*;
pub use results::*;

/// Comprehensive security auditor for optimizers
#[derive(Debug)]
pub struct SecurityAuditor {
    /// Configuration for security analysis
    config: SecurityAuditConfig,
    /// Input validation analyzer
    input_validator: InputValidationAnalyzer,
    /// Privacy guarantees analyzer
    privacy_analyzer: PrivacyGuaranteesAnalyzer,
    /// Memory safety analyzer
    memory_analyzer: MemorySafetyAnalyzer,
    /// Audit results storage
    audit_results: SecurityAuditResults,
}

impl SecurityAuditor {
    /// Create a new security auditor with default configuration
    pub fn new() -> Result<Self> {
        let config = SecurityAuditConfig::default();
        Self::with_config(config)
    }

    /// Create a security auditor with specific configuration
    pub fn with_config(config: SecurityAuditConfig) -> Result<Self> {
        config.validate().map_err(|e| OptimError::InvalidConfiguration(e))?;

        Ok(Self {
            config: config.clone(),
            input_validator: if config.enable_input_validation {
                InputValidationAnalyzer::with_builtin_tests()
            } else {
                InputValidationAnalyzer::new()
            },
            privacy_analyzer: if config.enable_privacy_analysis {
                PrivacyGuaranteesAnalyzer::with_builtin_tests()
            } else {
                PrivacyGuaranteesAnalyzer::new()
            },
            memory_analyzer: if config.enable_memory_safety {
                MemorySafetyAnalyzer::with_builtin_tests()
            } else {
                MemorySafetyAnalyzer::new()
            },
            audit_results: SecurityAuditResults::new(),
        })
    }

    /// Create a lightweight auditor for basic security checks
    pub fn lightweight() -> Result<Self> {
        Self::with_config(SecurityAuditConfig::lightweight())
    }

    /// Create a comprehensive auditor for thorough security analysis
    pub fn comprehensive() -> Result<Self> {
        Self::with_config(SecurityAuditConfig::comprehensive())
    }

    /// Run complete security audit
    pub fn run_complete_audit(&mut self) -> Result<&SecurityAuditResults> {
        let start_time = Instant::now();

        // Clear previous results
        self.audit_results = SecurityAuditResults::new();

        // Initialize execution summary
        let mut total_tests = 0;
        let mut total_passed = 0;
        let mut total_failed = 0;

        // Run input validation analysis if enabled
        if self.config.enable_input_validation {
            let validation_results = self.input_validator.run_all_tests()?;
            total_tests += validation_results.len();
            total_passed += validation_results.iter().filter(|r| r.status == TestStatus::Passed).count();
            total_failed += validation_results.iter().filter(|r| r.status == TestStatus::Failed).count();

            self.process_validation_results(&validation_results);
        }

        // Run privacy analysis if enabled
        if self.config.enable_privacy_analysis {
            let privacy_results = self.privacy_analyzer.run_all_tests()?;
            total_tests += privacy_results.len();
            total_passed += privacy_results.iter().filter(|r| r.status == TestStatus::Passed).count();
            total_failed += privacy_results.iter().filter(|r| r.status == TestStatus::Failed).count();

            self.process_privacy_results(&privacy_results);
        }

        // Run memory safety analysis if enabled
        if self.config.enable_memory_safety {
            let memory_results = self.memory_analyzer.run_all_tests()?;
            total_tests += memory_results.len();
            total_passed += memory_results.iter().filter(|r| r.status == TestStatus::Passed).count();
            total_failed += memory_results.iter().filter(|r| r.status == TestStatus::Failed).count();

            self.process_memory_results(&memory_results);
        }

        // Update execution summary
        let total_execution_time = start_time.elapsed();
        self.audit_results.execution_summary = ExecutionSummary {
            total_tests,
            tests_passed: total_passed,
            tests_failed: total_failed,
            tests_skipped: 0,
            total_execution_time,
            average_test_time: if total_tests > 0 {
                total_execution_time / total_tests as u32
            } else {
                Duration::from_secs(0)
            },
        };

        // Update vulnerability summary
        self.update_vulnerability_summary();

        // Calculate security score and risk assessment
        self.audit_results.calculate_security_score();
        self.update_risk_assessment();

        // Generate recommendations
        self.generate_recommendations();

        Ok(&self.audit_results)
    }

    /// Process input validation results into audit findings
    fn process_validation_results(&mut self, results: &[ValidationTestResult]) {
        for result in results {
            if let Some(vulnerability) = &result.vulnerability_detected {
                let finding = Finding::new(
                    format!("VAL-{}", self.audit_results.findings_by_category.len() + 1),
                    "Input Validation".to_string(),
                    result.severity.clone(),
                    result.test_name.clone(),
                    vulnerability.description.clone(),
                )
                .with_cvss_score(vulnerability.cvss_score);

                self.audit_results.add_finding(finding);
            }
        }

        // Update vulnerability statistics from input validation
        let stats = self.input_validator.get_statistics();
        self.audit_results.vulnerability_summary.total_vulnerabilities += stats.tests_failed;
    }

    /// Process privacy analysis results into audit findings
    fn process_privacy_results(&mut self, results: &[PrivacyTestResult]) {
        let mut privacy_violations = 0;
        let mut budget_violations = 0;
        let mut total_epsilon_violation = 0.0;
        let mut violation_count = 0;

        for result in results {
            for violation in &result.violations {
                privacy_violations += 1;
                violation_count += 1;
                total_epsilon_violation += violation.detected_params.violation_magnitude;

                if matches!(violation.violation_type, PrivacyViolationType::BudgetExceeded) {
                    budget_violations += 1;
                }

                let finding = Finding::new(
                    format!("PRIV-{}", self.audit_results.findings_by_category.len() + 1),
                    "Privacy".to_string(),
                    result.severity.clone(),
                    format!("Privacy Violation: {:?}", violation.violation_type),
                    format!("Privacy violation detected with confidence {:.2}", violation.confidence),
                );

                self.audit_results.add_finding(finding);
            }
        }

        // Store privacy analysis results
        self.audit_results.privacy_results = Some(PrivacyAnalysisResults {
            tests_executed: results.len(),
            violations_detected: privacy_violations,
            budget_violations,
            avg_epsilon_violation: if violation_count > 0 {
                total_epsilon_violation / violation_count as f64
            } else {
                0.0
            },
            most_common_violation: None, // Could be computed from violation types
        });

        self.audit_results.vulnerability_summary.total_vulnerabilities += privacy_violations;
    }

    /// Process memory safety results into audit findings
    fn process_memory_results(&mut self, results: &[MemoryTestResult]) {
        let mut issues_detected = 0;
        let mut leaks_detected = 0;
        let peak_memory = self.memory_analyzer.peak_memory_usage();

        for result in results {
            for issue in &result.issues {
                issues_detected += 1;

                if matches!(issue.issue_type, MemoryIssueType::Leak) {
                    leaks_detected += 1;
                }

                let finding = Finding::new(
                    format!("MEM-{}", self.audit_results.findings_by_category.len() + 1),
                    "Memory Safety".to_string(),
                    issue.severity.clone(),
                    format!("Memory Issue: {:?}", issue.issue_type),
                    issue.description.clone(),
                );

                self.audit_results.add_finding(finding);
            }
        }

        // Store memory analysis results
        self.audit_results.memory_results = Some(MemoryAnalysisResults {
            tests_executed: results.len(),
            issues_detected,
            peak_memory_usage: peak_memory,
            leaks_detected,
            most_common_issue: None, // Could be computed from issue types
        });

        self.audit_results.vulnerability_summary.total_vulnerabilities += issues_detected;
    }

    /// Update vulnerability summary across all analyzers
    fn update_vulnerability_summary(&mut self) {
        let mut critical = 0;
        let mut high = 0;
        let mut medium = 0;
        let mut low = 0;
        let mut total_cvss = 0.0;
        let mut cvss_count = 0;

        for findings in self.audit_results.findings_by_category.values() {
            for finding in findings {
                match finding.severity {
                    SeverityLevel::Critical => critical += 1,
                    SeverityLevel::High => high += 1,
                    SeverityLevel::Medium => medium += 1,
                    SeverityLevel::Low => low += 1,
                }

                if let Some(cvss) = finding.cvss_score {
                    total_cvss += cvss;
                    cvss_count += 1;
                }
            }
        }

        self.audit_results.vulnerability_summary.critical_count = critical;
        self.audit_results.vulnerability_summary.high_count = high;
        self.audit_results.vulnerability_summary.medium_count = medium;
        self.audit_results.vulnerability_summary.low_count = low;
        self.audit_results.vulnerability_summary.average_cvss = if cvss_count > 0 {
            total_cvss / cvss_count as f64
        } else {
            0.0
        };
    }

    /// Update risk assessment based on findings
    fn update_risk_assessment(&mut self) {
        let overall_risk = RiskLevel::from_security_score(self.audit_results.overall_security_score);

        // Generate risk factors
        let mut risk_factors = Vec::new();

        if self.audit_results.vulnerability_summary.critical_count > 0 {
            risk_factors.push(RiskFactor {
                name: "Critical Vulnerabilities".to_string(),
                impact: ImpactLevel::High,
                likelihood: 0.8,
                vulnerabilities: vec!["Critical security flaws require immediate attention".to_string()],
            });
        }

        if let Some(privacy) = &self.audit_results.privacy_results {
            if privacy.violations_detected > 0 {
                risk_factors.push(RiskFactor {
                    name: "Privacy Violations".to_string(),
                    impact: ImpactLevel::High,
                    likelihood: 0.6,
                    vulnerabilities: vec!["Privacy guarantees may be compromised".to_string()],
                });
            }
        }

        if let Some(memory) = &self.audit_results.memory_results {
            if memory.issues_detected > 0 {
                risk_factors.push(RiskFactor {
                    name: "Memory Safety Issues".to_string(),
                    impact: ImpactLevel::Medium,
                    likelihood: 0.4,
                    vulnerabilities: vec!["Memory safety issues may lead to crashes".to_string()],
                });
            }
        }

        // Generate priority actions
        let mut priority_actions = Vec::new();

        if self.audit_results.vulnerability_summary.critical_count > 0 {
            priority_actions.push(PriorityAction {
                action: "Address all critical vulnerabilities immediately".to_string(),
                priority: 1,
                effort: EffortLevel::High,
                impact: ImpactLevel::High,
                timeline: "Within 24 hours".to_string(),
            });
        }

        if self.audit_results.vulnerability_summary.high_count > 0 {
            priority_actions.push(PriorityAction {
                action: "Remediate high severity vulnerabilities".to_string(),
                priority: 2,
                effort: EffortLevel::Medium,
                impact: ImpactLevel::High,
                timeline: "Within 1 week".to_string(),
            });
        }

        self.audit_results.risk_assessment = RiskAssessment {
            overall_risk,
            risk_factors,
            compliance_status: ComplianceStatus {
                score: self.audit_results.overall_security_score,
                standards: vec!["OWASP".to_string(), "NIST".to_string()],
                failed_requirements: Vec::new(),
                compliance_recommendations: Vec::new(),
            },
            priority_actions,
        };
    }

    /// Generate comprehensive recommendations
    fn generate_recommendations(&mut self) {
        let mut recommendations = Vec::new();

        // General recommendations based on findings
        if self.audit_results.vulnerability_summary.total_vulnerabilities > 0 {
            recommendations.push("Implement comprehensive input validation for all user inputs".to_string());
            recommendations.push("Establish regular security testing and code review processes".to_string());
        }

        if self.audit_results.vulnerability_summary.critical_count > 0 {
            recommendations.push("Immediately address all critical vulnerabilities before production deployment".to_string());
        }

        if let Some(privacy) = &self.audit_results.privacy_results {
            if privacy.violations_detected > 0 {
                recommendations.push("Review and strengthen privacy guarantees and differential privacy parameters".to_string());
                recommendations.push("Implement privacy budget monitoring and enforcement".to_string());
            }
        }

        if let Some(memory) = &self.audit_results.memory_results {
            if memory.issues_detected > 0 {
                recommendations.push("Implement memory usage monitoring and limits".to_string());
                recommendations.push("Review memory management patterns and fix detected leaks".to_string());
            }
        }

        // Configuration-specific recommendations
        if !self.config.enable_crypto_analysis {
            recommendations.push("Consider enabling cryptographic security analysis for comprehensive coverage".to_string());
        }

        if !self.config.enable_numerical_analysis {
            recommendations.push("Enable numerical stability analysis to detect potential computation issues".to_string());
        }

        recommendations.sort();
        recommendations.dedup();
        self.audit_results.recommendations = recommendations;
    }

    /// Get the current audit configuration
    pub fn get_config(&self) -> &SecurityAuditConfig {
        &self.config
    }

    /// Update the audit configuration
    pub fn update_config(&mut self, config: SecurityAuditConfig) -> Result<()> {
        config.validate().map_err(|e| OptimError::InvalidConfiguration(e))?;
        self.config = config;
        Ok(())
    }

    /// Get the current audit results
    pub fn get_results(&self) -> &SecurityAuditResults {
        &self.audit_results
    }

    /// Get the input validation analyzer
    pub fn get_input_validator(&mut self) -> &mut InputValidationAnalyzer {
        &mut self.input_validator
    }

    /// Get the privacy analyzer
    pub fn get_privacy_analyzer(&mut self) -> &mut PrivacyGuaranteesAnalyzer {
        &mut self.privacy_analyzer
    }

    /// Get the memory safety analyzer
    pub fn get_memory_analyzer(&mut self) -> &mut MemorySafetyAnalyzer {
        &mut self.memory_analyzer
    }

    /// Clear all audit results and reset analyzers
    pub fn reset(&mut self) {
        self.audit_results = SecurityAuditResults::new();
        self.input_validator.clear_results();
        self.privacy_analyzer.clear_results();
        // Memory analyzer doesn't have a clear_results method in our current implementation
    }

    /// Generate a comprehensive security report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Security Audit Report\n\n");
        report.push_str(&format!("Generated: {:?}\n\n", self.audit_results.audit_timestamp));

        // Executive Summary
        report.push_str("## Executive Summary\n\n");
        report.push_str(&self.audit_results.generate_executive_summary());
        report.push_str("\n\n");

        // Detailed Findings
        if !self.audit_results.findings_by_category.is_empty() {
            report.push_str("## Detailed Findings\n\n");
            for (category, findings) in &self.audit_results.findings_by_category {
                report.push_str(&format!("### {}\n\n", category));
                for finding in findings {
                    report.push_str(&format!("**{}** ({})\n", finding.title, format!("{:?}", finding.severity)));
                    report.push_str(&format!("{}\n\n", finding.description));
                    if !finding.recommendations.is_empty() {
                        report.push_str("Recommendations:\n");
                        for rec in &finding.recommendations {
                            report.push_str(&format!("- {}\n", rec));
                        }
                        report.push_str("\n");
                    }
                }
            }
        }

        // Recommendations
        if !self.audit_results.recommendations.is_empty() {
            report.push_str("## Overall Recommendations\n\n");
            for (i, rec) in self.audit_results.recommendations.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, rec));
            }
            report.push_str("\n");
        }

        report
    }

    /// Export audit results to JSON
    pub fn export_json(&self) -> Result<String> {
        self.audit_results
            .to_json()
            .map_err(|e| OptimError::SerializationError(e.to_string()))
    }
}

impl Default for SecurityAuditor {
    fn default() -> Self {
        Self::new().expect("Failed to create default SecurityAuditor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_auditor() {
        let auditor = SecurityAuditor::new();
        assert!(auditor.is_ok());

        let auditor = auditor.unwrap();
        assert!(auditor.get_config().any_analysis_enabled());
    }

    #[test]
    fn test_lightweight_auditor() {
        let auditor = SecurityAuditor::lightweight();
        assert!(auditor.is_ok());

        let auditor = auditor.unwrap();
        let config = auditor.get_config();
        assert!(config.enable_input_validation);
        assert!(config.enable_memory_safety);
        assert!(!config.enable_privacy_analysis);
    }

    #[test]
    fn test_comprehensive_auditor() {
        let auditor = SecurityAuditor::comprehensive();
        assert!(auditor.is_ok());

        let auditor = auditor.unwrap();
        let config = auditor.get_config();
        assert!(config.enable_input_validation);
        assert!(config.enable_privacy_analysis);
        assert!(config.enable_memory_safety);
        assert!(config.enable_numerical_analysis);
        assert!(config.enable_access_control);
        assert!(config.enable_crypto_analysis);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = SecurityAuditConfig::default();
        config.max_test_iterations = 0;

        let auditor = SecurityAuditor::with_config(config);
        assert!(auditor.is_err());
    }

    #[test]
    fn test_complete_audit() {
        let mut auditor = SecurityAuditor::lightweight().unwrap();
        let results = auditor.run_complete_audit();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.execution_summary.total_tests > 0);
    }

    #[test]
    fn test_reset_auditor() {
        let mut auditor = SecurityAuditor::lightweight().unwrap();
        let _ = auditor.run_complete_audit().unwrap();

        // Should have some results
        assert!(auditor.get_results().execution_summary.total_tests > 0);

        auditor.reset();

        // Should be reset
        assert_eq!(auditor.get_results().execution_summary.total_tests, 0);
    }

    #[test]
    fn test_report_generation() {
        let mut auditor = SecurityAuditor::lightweight().unwrap();
        let _ = auditor.run_complete_audit().unwrap();

        let report = auditor.generate_report();
        assert!(report.contains("Security Audit Report"));
        assert!(report.contains("Executive Summary"));
    }

    #[test]
    fn test_json_export() {
        let mut auditor = SecurityAuditor::lightweight().unwrap();
        let _ = auditor.run_complete_audit().unwrap();

        let json_result = auditor.export_json();
        assert!(json_result.is_ok());

        let json = json_result.unwrap();
        assert!(json.contains("execution_summary"));
        assert!(json.contains("vulnerability_summary"));
    }
}