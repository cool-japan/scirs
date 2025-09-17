//! Security audit results aggregation and reporting
//!
//! This module provides comprehensive result collection, analysis, and reporting
//! capabilities for security audit operations across all security domains.

use std::collections::HashMap;
use std::time::Duration;

use super::types::*;

/// Comprehensive security audit results
#[derive(Debug, Clone)]
pub struct SecurityAuditResults {
    /// Test execution summary
    pub execution_summary: ExecutionSummary,
    /// Aggregated vulnerability statistics
    pub vulnerability_summary: VulnerabilitySummary,
    /// Privacy analysis results
    pub privacy_results: Option<PrivacyAnalysisResults>,
    /// Memory safety results
    pub memory_results: Option<MemoryAnalysisResults>,
    /// Detailed findings by category
    pub findings_by_category: HashMap<String, Vec<Finding>>,
    /// Overall security score (0-100)
    pub overall_security_score: f64,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Recommendations summary
    pub recommendations: Vec<String>,
    /// Timestamp of audit completion
    pub audit_timestamp: std::time::SystemTime,
}

/// Test execution summary
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Total tests executed
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Tests skipped
    pub tests_skipped: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average test execution time
    pub average_test_time: Duration,
}

/// Vulnerability summary across all analyzers
#[derive(Debug, Clone)]
pub struct VulnerabilitySummary {
    /// Total vulnerabilities found
    pub total_vulnerabilities: usize,
    /// Critical vulnerabilities
    pub critical_count: usize,
    /// High severity vulnerabilities
    pub high_count: usize,
    /// Medium severity vulnerabilities
    pub medium_count: usize,
    /// Low severity vulnerabilities
    pub low_count: usize,
    /// Vulnerabilities by type
    pub by_type: HashMap<String, usize>,
    /// Average CVSS score
    pub average_cvss: f64,
}

/// Privacy analysis results summary
#[derive(Debug, Clone)]
pub struct PrivacyAnalysisResults {
    /// Privacy tests executed
    pub tests_executed: usize,
    /// Privacy violations detected
    pub violations_detected: usize,
    /// Budget violations
    pub budget_violations: usize,
    /// Average epsilon violation
    pub avg_epsilon_violation: f64,
    /// Most common violation type
    pub most_common_violation: Option<String>,
}

/// Memory analysis results summary
#[derive(Debug, Clone)]
pub struct MemoryAnalysisResults {
    /// Memory tests executed
    pub tests_executed: usize,
    /// Memory issues detected
    pub issues_detected: usize,
    /// Peak memory usage during tests
    pub peak_memory_usage: usize,
    /// Memory leaks detected
    pub leaks_detected: usize,
    /// Most common memory issue type
    pub most_common_issue: Option<String>,
}

/// Individual security finding
#[derive(Debug, Clone)]
pub struct Finding {
    /// Finding identifier
    pub id: String,
    /// Category of finding
    pub category: String,
    /// Severity level
    pub severity: SeverityLevel,
    /// Title/summary
    pub title: String,
    /// Detailed description
    pub description: String,
    /// CVSS score if applicable
    pub cvss_score: Option<f64>,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Remediation recommendations
    pub recommendations: Vec<String>,
}

/// Overall risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    /// Risk factors identified
    pub risk_factors: Vec<RiskFactor>,
    /// Compliance status
    pub compliance_status: ComplianceStatus,
    /// Actionable recommendations
    pub priority_actions: Vec<PriorityAction>,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Individual risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    /// Risk factor name
    pub name: String,
    /// Impact level
    pub impact: ImpactLevel,
    /// Likelihood of exploitation
    pub likelihood: f64,
    /// Contributing vulnerabilities
    pub vulnerabilities: Vec<String>,
}

/// Compliance status
#[derive(Debug, Clone)]
pub struct ComplianceStatus {
    /// Overall compliance score (0-100)
    pub score: f64,
    /// Standards evaluated against
    pub standards: Vec<String>,
    /// Failed requirements
    pub failed_requirements: Vec<String>,
    /// Recommendations for compliance
    pub compliance_recommendations: Vec<String>,
}

/// Priority action for risk mitigation
#[derive(Debug, Clone)]
pub struct PriorityAction {
    /// Action description
    pub action: String,
    /// Priority level (1 = highest)
    pub priority: u8,
    /// Estimated effort
    pub effort: EffortLevel,
    /// Expected impact
    pub impact: ImpactLevel,
    /// Timeline recommendation
    pub timeline: String,
}

/// Effort levels for remediation actions
#[derive(Debug, Clone)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl SecurityAuditResults {
    /// Create a new empty audit results
    pub fn new() -> Self {
        Self {
            execution_summary: ExecutionSummary::default(),
            vulnerability_summary: VulnerabilitySummary::default(),
            privacy_results: None,
            memory_results: None,
            findings_by_category: HashMap::new(),
            overall_security_score: 0.0,
            risk_assessment: RiskAssessment::default(),
            recommendations: Vec::new(),
            audit_timestamp: std::time::SystemTime::now(),
        }
    }

    /// Calculate overall security score based on findings
    pub fn calculate_security_score(&mut self) {
        let total_possible = 100.0;
        let mut deductions = 0.0;

        // Deduct points based on vulnerability severity
        deductions += self.vulnerability_summary.critical_count as f64 * 25.0;
        deductions += self.vulnerability_summary.high_count as f64 * 15.0;
        deductions += self.vulnerability_summary.medium_count as f64 * 8.0;
        deductions += self.vulnerability_summary.low_count as f64 * 3.0;

        // Additional deductions for privacy and memory issues
        if let Some(privacy) = &self.privacy_results {
            deductions += privacy.violations_detected as f64 * 10.0;
        }

        if let Some(memory) = &self.memory_results {
            deductions += memory.issues_detected as f64 * 5.0;
        }

        self.overall_security_score = (total_possible - deductions).max(0.0);
    }

    /// Add a finding to the results
    pub fn add_finding(&mut self, finding: Finding) {
        let category = finding.category.clone();
        self.findings_by_category
            .entry(category)
            .or_insert_with(Vec::new)
            .push(finding);
    }

    /// Get findings by severity
    pub fn get_findings_by_severity(&self, severity: SeverityLevel) -> Vec<&Finding> {
        self.findings_by_category
            .values()
            .flatten()
            .filter(|f| f.severity == severity)
            .collect()
    }

    /// Get critical findings requiring immediate attention
    pub fn get_critical_findings(&self) -> Vec<&Finding> {
        self.get_findings_by_severity(SeverityLevel::Critical)
    }

    /// Generate executive summary
    pub fn generate_executive_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!(
            "Security Audit Summary\n====================\n\n"
        ));

        summary.push_str(&format!(
            "Overall Security Score: {:.1}/100\n",
            self.overall_security_score
        ));

        summary.push_str(&format!(
            "Risk Level: {:?}\n\n",
            self.risk_assessment.overall_risk
        ));

        summary.push_str(&format!(
            "Test Execution: {}/{} passed ({:.1}% success rate)\n",
            self.execution_summary.tests_passed,
            self.execution_summary.total_tests,
            (self.execution_summary.tests_passed as f64 / self.execution_summary.total_tests as f64) * 100.0
        ));

        summary.push_str(&format!(
            "Vulnerabilities Found: {} total ({} critical, {} high)\n\n",
            self.vulnerability_summary.total_vulnerabilities,
            self.vulnerability_summary.critical_count,
            self.vulnerability_summary.high_count
        ));

        if !self.risk_assessment.priority_actions.is_empty() {
            summary.push_str("Priority Actions:\n");
            for action in &self.risk_assessment.priority_actions {
                summary.push_str(&format!("- {}\n", action.action));
            }
        }

        summary
    }

    /// Export results to JSON format
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl Default for SecurityAuditResults {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ExecutionSummary {
    fn default() -> Self {
        Self {
            total_tests: 0,
            tests_passed: 0,
            tests_failed: 0,
            tests_skipped: 0,
            total_execution_time: Duration::from_secs(0),
            average_test_time: Duration::from_secs(0),
        }
    }
}

impl Default for VulnerabilitySummary {
    fn default() -> Self {
        Self {
            total_vulnerabilities: 0,
            critical_count: 0,
            high_count: 0,
            medium_count: 0,
            low_count: 0,
            by_type: HashMap::new(),
            average_cvss: 0.0,
        }
    }
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            overall_risk: RiskLevel::Low,
            risk_factors: Vec::new(),
            compliance_status: ComplianceStatus {
                score: 100.0,
                standards: Vec::new(),
                failed_requirements: Vec::new(),
                compliance_recommendations: Vec::new(),
            },
            priority_actions: Vec::new(),
        }
    }
}

impl RiskLevel {
    /// Convert risk level to numeric score for calculations
    pub fn to_score(&self) -> u8 {
        match self {
            RiskLevel::Low => 1,
            RiskLevel::Medium => 2,
            RiskLevel::High => 3,
            RiskLevel::Critical => 4,
        }
    }

    /// Determine risk level from security score
    pub fn from_security_score(score: f64) -> Self {
        match score {
            s if s >= 80.0 => RiskLevel::Low,
            s if s >= 60.0 => RiskLevel::Medium,
            s if s >= 40.0 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
}

impl Finding {
    /// Create a new security finding
    pub fn new(
        id: String,
        category: String,
        severity: SeverityLevel,
        title: String,
        description: String,
    ) -> Self {
        Self {
            id,
            category,
            severity,
            title,
            description,
            cvss_score: None,
            affected_components: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add a recommendation to the finding
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Add an affected component
    pub fn add_affected_component(&mut self, component: String) {
        self.affected_components.push(component);
    }

    /// Set CVSS score
    pub fn with_cvss_score(mut self, score: f64) -> Self {
        self.cvss_score = Some(score);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_audit_results() {
        let results = SecurityAuditResults::new();
        assert_eq!(results.overall_security_score, 0.0);
        assert_eq!(results.vulnerability_summary.total_vulnerabilities, 0);
    }

    #[test]
    fn test_security_score_calculation() {
        let mut results = SecurityAuditResults::new();
        results.vulnerability_summary.critical_count = 1;
        results.vulnerability_summary.high_count = 2;
        results.vulnerability_summary.medium_count = 3;
        results.vulnerability_summary.low_count = 4;

        results.calculate_security_score();

        // Should be 100 - (1*25 + 2*15 + 3*8 + 4*3) = 100 - (25 + 30 + 24 + 12) = 9.0
        assert_eq!(results.overall_security_score, 9.0);
    }

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_security_score(85.0), RiskLevel::Low);
        assert_eq!(RiskLevel::from_security_score(65.0), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_security_score(45.0), RiskLevel::High);
        assert_eq!(RiskLevel::from_security_score(25.0), RiskLevel::Critical);
    }

    #[test]
    fn test_finding_creation() {
        let mut finding = Finding::new(
            "VULN-001".to_string(),
            "Input Validation".to_string(),
            SeverityLevel::High,
            "Buffer Overflow Vulnerability".to_string(),
            "Application is vulnerable to buffer overflow attacks".to_string(),
        );

        finding.add_recommendation("Implement bounds checking".to_string());
        finding.add_affected_component("input_parser".to_string());

        assert_eq!(finding.recommendations.len(), 1);
        assert_eq!(finding.affected_components.len(), 1);
    }

    #[test]
    fn test_add_finding() {
        let mut results = SecurityAuditResults::new();
        let finding = Finding::new(
            "VULN-001".to_string(),
            "Memory Safety".to_string(),
            SeverityLevel::Medium,
            "Memory Leak".to_string(),
            "Potential memory leak detected".to_string(),
        );

        results.add_finding(finding);
        assert_eq!(results.findings_by_category.len(), 1);
        assert!(results.findings_by_category.contains_key("Memory Safety"));
    }

    #[test]
    fn test_executive_summary() {
        let mut results = SecurityAuditResults::new();
        results.execution_summary.total_tests = 10;
        results.execution_summary.tests_passed = 8;
        results.vulnerability_summary.total_vulnerabilities = 3;
        results.vulnerability_summary.critical_count = 1;
        results.vulnerability_summary.high_count = 1;
        results.overall_security_score = 75.0;

        let summary = results.generate_executive_summary();
        assert!(summary.contains("75.0/100"));
        assert!(summary.contains("8/10 passed"));
        assert!(summary.contains("3 total"));
    }
}