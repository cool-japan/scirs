//! Core data types and structures for security auditing
//!
//! This module provides comprehensive type definitions for security analysis,
//! including vulnerability classifications, test results, impact assessments,
//! and privacy-related structures.

use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// Core Security Types
// =============================================================================

/// Test execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    /// Test passed (no vulnerability)
    Passed,
    /// Test failed (vulnerability detected)
    Failed,
    /// Test timed out
    Timeout,
    /// Test error (couldn't execute)
    Error,
    /// Test skipped
    Skipped,
}

/// Severity levels for vulnerabilities
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeverityLevel {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Impact levels
#[derive(Debug, Clone)]
pub enum ImpactLevel {
    None,
    Low,
    Medium,
    High,
}

/// Complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
}

/// Privilege levels
#[derive(Debug, Clone)]
pub enum PrivilegeLevel {
    None,
    Low,
    High,
}

/// Accessibility levels
#[derive(Debug, Clone)]
pub enum AccessibilityLevel {
    Local,
    Adjacent,
    Network,
    Physical,
}

/// Types of vulnerabilities
#[derive(Debug, Clone)]
pub enum VulnerabilityType {
    /// Input validation bypass
    InputValidationBypass,
    /// Buffer overflow
    BufferOverflow,
    /// Privacy guarantee violation
    PrivacyViolation,
    /// Information disclosure
    InformationDisclosure,
    /// Denial of service
    DenialOfService,
    /// Memory corruption
    MemoryCorruption,
    /// Numerical instability
    NumericalInstability,
    /// Side-channel attack
    SideChannelAttack,
}

/// Detected vulnerability information
#[derive(Debug, Clone)]
pub struct Vulnerability {
    /// Vulnerability type
    pub vulnerability_type: VulnerabilityType,
    /// CVSS score (0-10)
    pub cvss_score: f64,
    /// Description
    pub description: String,
    /// Proof of concept
    pub proof_of_concept: String,
    /// Impact assessment
    pub impact: ImpactAssessment,
    /// Exploitability assessment
    pub exploitability: ExploitabilityAssessment,
}

/// Impact assessment for vulnerabilities
#[derive(Debug, Clone)]
pub struct ImpactAssessment {
    /// Confidentiality impact
    pub confidentiality: ImpactLevel,
    /// Integrity impact
    pub integrity: ImpactLevel,
    /// Availability impact
    pub availability: ImpactLevel,
    /// Privacy impact
    pub privacy: ImpactLevel,
}

/// Exploitability assessment
#[derive(Debug, Clone)]
pub struct ExploitabilityAssessment {
    /// Attack complexity
    pub attack_complexity: ComplexityLevel,
    /// Privileges required
    pub privileges_required: PrivilegeLevel,
    /// User interaction required
    pub user_interaction: bool,
    /// Attack vector accessibility
    pub attack_vector: AccessibilityLevel,
}

/// Statistics on detected vulnerabilities
#[derive(Debug, Clone)]
pub struct VulnerabilityStatistics {
    /// Total tests executed
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Vulnerabilities by severity
    pub vulnerabilities_by_severity: HashMap<SeverityLevel, usize>,
    /// Vulnerabilities by type
    pub vulnerabilities_by_type: HashMap<String, usize>,
    /// Average CVSS score
    pub average_cvss_score: f64,
    /// Time to detection
    pub average_detection_time: Duration,
}

// =============================================================================
// Input Validation Types
// =============================================================================

/// Categories of validation tests
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationCategory {
    /// Malformed input detection
    MalformedInput,
    /// Boundary condition testing
    BoundaryConditions,
    /// Type confusion attacks
    TypeConfusion,
    /// Buffer overflow attempts
    BufferOverflow,
    /// Injection attacks
    InjectionAttacks,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// Types of attack vectors
#[derive(Debug, Clone)]
pub enum AttackVector {
    /// NaN/Infinity injection
    NaNInjection,
    /// Extremely large values
    ExtremeValues,
    /// Dimension manipulation
    DimensionMismatch,
    /// Negative dimensions
    NegativeDimensions,
    /// Zero/empty arrays
    EmptyArrays,
    /// Malformed gradients
    MalformedGradients,
    /// Privacy parameter manipulation
    PrivacyParameterAttack,
    /// Memory exhaustion
    MemoryExhaustionAttack,
}

/// Expected behavior for security tests
#[derive(Debug, Clone)]
pub enum ExpectedBehavior {
    /// Should reject input with specific error
    RejectWithError(String),
    /// Should handle gracefully without crash
    HandleGracefully,
    /// Should sanitize input
    SanitizeInput,
    /// Should maintain security guarantees
    MaintainSecurityGuarantees,
}

/// Types of test payloads
#[derive(Debug, Clone)]
pub enum PayloadType {
    /// NaN values
    NaNPayload,
    /// Infinity values
    InfinityPayload,
    /// Extremely large numbers
    ExtremeValuePayload(f64),
    /// Zero-sized arrays
    ZeroSizedPayload,
    /// Mismatched dimensions
    DimensionMismatchPayload,
    /// Negative learning rates
    NegativeLearningRate,
    /// Invalid privacy parameters
    InvalidPrivacyParams,
}

/// Individual input validation test
#[derive(Debug, Clone)]
pub struct InputValidationTest {
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Test category
    pub category: ValidationCategory,
    /// Attack vector being tested
    pub attack_vector: AttackVector,
    /// Expected behavior
    pub expected_behavior: ExpectedBehavior,
    /// Test payload generator
    pub payload_generator: PayloadType,
}

/// Result of a validation test
#[derive(Debug, Clone)]
pub struct ValidationTestResult {
    /// Test name
    pub test_name: String,
    /// Test status
    pub status: TestStatus,
    /// Vulnerability detected
    pub vulnerability_detected: Option<Vulnerability>,
    /// Error message (if any)
    pub error_message: Option<String>,
    /// Execution time
    pub execution_time: Duration,
    /// Severity level
    pub severity: SeverityLevel,
    /// Recommendation
    pub recommendation: Option<String>,
}

// =============================================================================
// Privacy Types
// =============================================================================

/// Privacy mechanisms
#[derive(Debug, Clone)]
pub enum PrivacyMechanism {
    /// Differential privacy
    DifferentialPrivacy,
    /// Local differential privacy
    LocalDifferentialPrivacy,
    /// Federated learning privacy
    FederatedPrivacy,
    /// Secure multi-party computation
    SecureMultiParty,
}

/// Privacy attack scenarios
#[derive(Debug, Clone)]
pub enum PrivacyAttackScenario {
    /// Membership inference attack
    MembershipInference,
    /// Model inversion attack
    ModelInversion,
    /// Property inference attack
    PropertyInference,
    /// Reconstruction attack
    ReconstructionAttack,
    /// Budget exhaustion attack
    BudgetExhaustionAttack,
    /// Noise reduction attack
    NoiseReductionAttack,
}

/// Composition methods for privacy
#[derive(Debug, Clone)]
pub enum CompositionMethod {
    Basic,
    Advanced,
    Optimal,
    MomentsAccountant,
    RenyiDP,
}

/// Privacy constraints
#[derive(Debug, Clone)]
pub enum PrivacyConstraint {
    /// Maximum information leakage
    MaxInformationLeakage(f64),
    /// Minimum noise level
    MinNoiseLevel(f64),
    /// Maximum correlation
    MaxCorrelation(f64),
}

/// Privacy guarantee specifications
#[derive(Debug, Clone)]
pub struct PrivacyGuarantee {
    /// Epsilon parameter
    pub epsilon: f64,
    /// Delta parameter
    pub delta: f64,
    /// Composition method
    pub composition_method: CompositionMethod,
    /// Additional constraints
    pub constraints: Vec<PrivacyConstraint>,
}

/// Privacy security test
#[derive(Debug, Clone)]
pub struct PrivacyTest {
    /// Test name
    pub name: String,
    /// Privacy mechanism being tested
    pub mechanism: PrivacyMechanism,
    /// Attack scenario
    pub attack_scenario: PrivacyAttackScenario,
    /// Expected privacy guarantee
    pub expected_guarantee: PrivacyGuarantee,
}

/// Types of privacy violations
#[derive(Debug, Clone)]
pub enum PrivacyViolationType {
    /// Budget exceeded
    BudgetExceeded,
    /// Insufficient noise
    InsufficientNoise,
    /// Information leakage
    InformationLeakage,
    /// Correlation exposure
    CorrelationExposure,
    /// Membership disclosure
    MembershipDisclosure,
}

/// Privacy parameter violations
#[derive(Debug, Clone)]
pub struct PrivacyParameterViolation {
    /// Expected epsilon
    pub expected_epsilon: f64,
    /// Actual epsilon
    pub actual_epsilon: f64,
    /// Expected delta
    pub expected_delta: f64,
    /// Actual delta
    pub actual_delta: f64,
    /// Violation magnitude
    pub violation_magnitude: f64,
}

/// Privacy violation detection
#[derive(Debug, Clone)]
pub struct PrivacyViolation {
    /// Violation type
    pub violation_type: PrivacyViolationType,
    /// Detected parameters
    pub detected_params: PrivacyParameterViolation,
    /// Confidence level
    pub confidence: f64,
    /// Evidence
    pub evidence: Vec<String>,
}

/// Budget status
#[derive(Debug, Clone)]
pub enum BudgetStatus {
    Healthy,
    Warning,
    Critical,
    Exhausted,
}

/// Budget verification result
#[derive(Debug, Clone)]
pub struct BudgetVerificationResult {
    /// Test name
    pub test_name: String,
    /// Budget status
    pub budget_status: BudgetStatus,
    /// Remaining budget
    pub remaining_budget: f64,
    /// Projected exhaustion
    pub projected_exhaustion: Option<usize>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

// =============================================================================
// Memory Safety Types
// =============================================================================

/// Memory vulnerability types
#[derive(Debug, Clone)]
pub enum MemoryVulnerabilityType {
    /// Buffer overflow
    BufferOverflow,
    /// Use after free
    UseAfterFree,
    /// Memory leak
    MemoryLeak,
    /// Double free
    DoubleFree,
    /// Stack overflow
    StackOverflow,
    /// Heap corruption
    HeapCorruption,
}

/// Memory test scenarios
#[derive(Debug, Clone)]
pub enum MemoryTestScenario {
    /// Large array allocation
    LargeArrayAllocation,
    /// Rapid allocation/deallocation
    RapidAllocation,
    /// Deep recursion
    DeepRecursion,
    /// Circular references
    CircularReferences,
}

/// Memory safety test
#[derive(Debug, Clone)]
pub struct MemorySafetyTest {
    /// Test name
    pub name: String,
    /// Memory vulnerability type
    pub vulnerability_type: MemoryVulnerabilityType,
    /// Test scenario
    pub scenario: MemoryTestScenario,
}

/// Memory issue types
#[derive(Debug, Clone)]
pub enum MemoryIssueType {
    Leak,
    Corruption,
    OverAccess,
    UnderAccess,
    Fragmentation,
}

/// Memory location information
#[derive(Debug, Clone)]
pub struct MemoryLocation {
    /// Function name
    pub function: String,
    /// Line number
    pub line: usize,
    /// Memory address (if available)
    pub address: Option<usize>,
}

/// Memory issue detection
#[derive(Debug, Clone)]
pub struct MemoryIssue {
    /// Issue type
    pub issue_type: MemoryIssueType,
    /// Severity
    pub severity: SeverityLevel,
    /// Description
    pub description: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Memory location
    pub memory_location: Option<MemoryLocation>,
}

// =============================================================================
// Utility Implementations
// =============================================================================

impl Default for VulnerabilityStatistics {
    fn default() -> Self {
        Self {
            total_tests: 0,
            tests_passed: 0,
            tests_failed: 0,
            vulnerabilities_by_severity: HashMap::new(),
            vulnerabilities_by_type: HashMap::new(),
            average_cvss_score: 0.0,
            average_detection_time: Duration::from_secs(0),
        }
    }
}

impl VulnerabilityStatistics {
    /// Create a new empty vulnerability statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a new test result
    pub fn update_with_result(&mut self, result: &ValidationTestResult) {
        self.total_tests += 1;

        match result.status {
            TestStatus::Passed => self.tests_passed += 1,
            TestStatus::Failed => {
                self.tests_failed += 1;

                // Update severity statistics
                *self.vulnerabilities_by_severity.entry(result.severity.clone()).or_insert(0) += 1;

                // Update type statistics if vulnerability detected
                if let Some(vulnerability) = &result.vulnerability_detected {
                    let type_name = format!("{:?}", vulnerability.vulnerability_type);
                    *self.vulnerabilities_by_type.entry(type_name).or_insert(0) += 1;
                }
            }
            _ => {} // Other statuses don't affect pass/fail counts
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.tests_passed as f64 / self.total_tests as f64
        }
    }

    /// Get critical vulnerability count
    pub fn critical_vulnerabilities(&self) -> usize {
        self.vulnerabilities_by_severity.get(&SeverityLevel::Critical).copied().unwrap_or(0)
    }

    /// Get high severity vulnerability count
    pub fn high_vulnerabilities(&self) -> usize {
        self.vulnerabilities_by_severity.get(&SeverityLevel::High).copied().unwrap_or(0)
    }
}

impl SeverityLevel {
    /// Convert severity to numeric score for calculations
    pub fn to_score(&self) -> u8 {
        match self {
            SeverityLevel::Low => 1,
            SeverityLevel::Medium => 2,
            SeverityLevel::High => 3,
            SeverityLevel::Critical => 4,
        }
    }

    /// Convert from CVSS score to severity level
    pub fn from_cvss_score(score: f64) -> Self {
        match score {
            s if s >= 9.0 => SeverityLevel::Critical,
            s if s >= 7.0 => SeverityLevel::High,
            s if s >= 4.0 => SeverityLevel::Medium,
            _ => SeverityLevel::Low,
        }
    }
}

impl PrivacyGuarantee {
    /// Create a new privacy guarantee
    pub fn new(epsilon: f64, delta: f64, composition: CompositionMethod) -> Self {
        Self {
            epsilon,
            delta,
            composition_method: composition,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the privacy guarantee
    pub fn with_constraint(mut self, constraint: PrivacyConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Check if the guarantee is satisfied given actual parameters
    pub fn is_satisfied(&self, actual_epsilon: f64, actual_delta: f64) -> bool {
        actual_epsilon <= self.epsilon && actual_delta <= self.delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulnerability_statistics() {
        let mut stats = VulnerabilityStatistics::new();
        assert_eq!(stats.total_tests, 0);
        assert_eq!(stats.success_rate(), 0.0);

        let result = ValidationTestResult {
            test_name: "test1".to_string(),
            status: TestStatus::Passed,
            vulnerability_detected: None,
            error_message: None,
            execution_time: Duration::from_millis(100),
            severity: SeverityLevel::Low,
            recommendation: None,
        };

        stats.update_with_result(&result);
        assert_eq!(stats.total_tests, 1);
        assert_eq!(stats.tests_passed, 1);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn test_severity_level_conversions() {
        assert_eq!(SeverityLevel::from_cvss_score(9.5), SeverityLevel::Critical);
        assert_eq!(SeverityLevel::from_cvss_score(7.5), SeverityLevel::High);
        assert_eq!(SeverityLevel::from_cvss_score(5.0), SeverityLevel::Medium);
        assert_eq!(SeverityLevel::from_cvss_score(2.0), SeverityLevel::Low);
    }

    #[test]
    fn test_privacy_guarantee() {
        let guarantee = PrivacyGuarantee::new(1.0, 1e-5, CompositionMethod::Advanced)
            .with_constraint(PrivacyConstraint::MinNoiseLevel(0.1));

        assert!(guarantee.is_satisfied(0.8, 1e-6));
        assert!(!guarantee.is_satisfied(1.2, 1e-6));
        assert!(!guarantee.is_satisfied(0.8, 1e-4));
    }
}