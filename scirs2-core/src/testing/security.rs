//! # Security Testing Framework
//!
//! This module provides security testing capabilities for `SciRS2` Core,
//! focusing on input validation, bounds checking, and vulnerability discovery.
//! It includes:
//! - Input validation testing
//! - Buffer overflow detection
//! - Integer overflow testing
//! - Memory safety verification
//! - Denial of service attack simulation

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::testing::{TestConfig, TestResult};
use std::time::{Duration, Instant};

/// Security test configuration
#[derive(Debug, Clone)]
pub struct SecurityTestConfig {
    /// Maximum input size to test
    pub max_input_size: usize,
    /// Number of malicious input patterns to test
    pub malicious_patterns: usize,
    /// Enable bounds checking tests
    pub test_bounds_checking: bool,
    /// Enable integer overflow tests
    pub test_integer_overflow: bool,
    /// Enable memory safety tests
    pub test_memory_safety: bool,
    /// Enable DoS simulation tests
    pub test_dos_simulation: bool,
    /// Timeout for individual security tests
    pub test_timeout: Duration,
}

impl Default for SecurityTestConfig {
    fn default() -> Self {
        Self {
            max_input_size: 1024 * 1024, // 1MB
            malicious_patterns: 1000,
            test_bounds_checking: true,
            test_integer_overflow: true,
            test_memory_safety: true,
            test_dos_simulation: true,
            test_timeout: Duration::from_secs(10),
        }
    }
}

impl SecurityTestConfig {
    /// Create a new security test configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum input size
    pub fn with_max_input_size(mut self, size: usize) -> Self {
        self.max_input_size = size;
        self
    }

    /// Set the number of malicious patterns
    pub fn with_malicious_patterns(mut self, patterns: usize) -> Self {
        self.malicious_patterns = patterns;
        self
    }

    /// Enable or disable bounds checking tests
    pub fn with_bounds_checking(mut self, enabled: bool) -> Self {
        self.test_bounds_checking = enabled;
        self
    }

    /// Enable or disable integer overflow tests
    pub fn with_integer_overflow(mut self, enabled: bool) -> Self {
        self.test_integer_overflow = enabled;
        self
    }

    /// Enable or disable memory safety tests
    pub fn with_memory_safety(mut self, enabled: bool) -> Self {
        self.test_memory_safety = enabled;
        self
    }

    /// Enable or disable DoS simulation tests
    pub fn with_dos_simulation(mut self, enabled: bool) -> Self {
        self.test_dos_simulation = enabled;
        self
    }

    /// Set the test timeout
    pub fn with_test_timeout(mut self, timeout: Duration) -> Self {
        self.test_timeout = timeout;
        self
    }
}

/// Security test result
#[derive(Debug, Clone)]
pub struct SecurityTestResult {
    /// Test name
    pub test_name: String,
    /// Number of tests executed
    pub tests_executed: usize,
    /// Number of vulnerabilities found
    pub vulnerabilities_found: usize,
    /// Test execution time
    pub duration: Duration,
    /// Security issues discovered
    pub security_issues: Vec<SecurityIssue>,
    /// Overall security assessment
    pub security_level: SecurityLevel,
}

/// Security issue found during testing
#[derive(Debug, Clone)]
pub struct SecurityIssue {
    /// Issue severity
    pub severity: SecuritySeverity,
    /// Issue category
    pub category: SecurityCategory,
    /// Description of the issue
    pub description: String,
    /// Input that triggered the issue
    pub trigger_input: String,
    /// Recommended mitigation
    pub mitigation: Option<String>,
}

/// Security severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecuritySeverity {
    /// Critical security vulnerability
    Critical,
    /// High severity issue
    High,
    /// Medium severity issue
    Medium,
    /// Low severity issue
    Low,
    /// Informational finding
    Info,
}

/// Security issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityCategory {
    /// Buffer overflow or underflow
    BufferOverflow,
    /// Integer overflow or underflow
    IntegerOverflow,
    /// Out-of-bounds access
    OutOfBounds,
    /// Memory safety violation
    MemorySafety,
    /// Input validation bypass
    InputValidation,
    /// Denial of service vulnerability
    DenialOfService,
    /// Information disclosure
    InformationDisclosure,
}

/// Overall security assessment level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Severe security issues found
    Insecure,
    /// Some security concerns
    Vulnerable,
    /// Minor security issues
    Weak,
    /// Good security posture
    Secure,
    /// Excellent security
    Hardened,
}

/// Input validation security tester
pub struct InputValidationTester {
    config: SecurityTestConfig,
}

impl InputValidationTester {
    /// Create a new input validation tester
    pub fn new(config: SecurityTestConfig) -> Self {
        Self { config }
    }

    /// Test input validation with malicious patterns
    pub fn test_malicious_inputs<F>(&self, test_function: F) -> CoreResult<SecurityTestResult>
    where
        F: Fn(&[u8]) -> CoreResult<()>,
    {
        let start_time = Instant::now();
        let mut result = SecurityTestResult {
            test_name: "malicious_input_validation".to_string(),
            tests_executed: 0,
            vulnerabilities_found: 0,
            duration: Duration::from_secs(0),
            security_issues: Vec::new(),
            security_level: SecurityLevel::Secure,
        };

        // Test various malicious input patterns
        let patterns = self.generate_malicious_patterns();

        for (i, pattern) in patterns.iter().enumerate() {
            result.tests_executed += 1;

            // Test with timeout
            let test_start = Instant::now();
            let test_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_function(pattern)));

            let test_duration = test_start.elapsed();

            match test_result {
                Ok(Ok(())) => {
                    // Function handled input correctly
                }
                Ok(Err(_)) => {
                    // Function returned error - this is expected for malicious input
                }
                Err(_) => {
                    // Function panicked - potential vulnerability
                    result.vulnerabilities_found += 1;
                    result.security_issues.push(SecurityIssue {
                        severity: SecuritySeverity::High,
                        category: SecurityCategory::InputValidation,
                        description: "Function panicked on malicious input".to_string(),
                        trigger_input: format!("Pattern {}: {:?}", i, pattern),
                        mitigation: Some(
                            "Add proper input validation and error handling".to_string(),
                        ),
                    });
                }
            }

            // Check for potential DoS (excessive execution time)
            if test_duration > self.config.test_timeout {
                result.vulnerabilities_found += 1;
                result.security_issues.push(SecurityIssue {
                    severity: SecuritySeverity::Medium,
                    category: SecurityCategory::DenialOfService,
                    description: format!("Function took excessive time: {:?}", test_duration),
                    trigger_input: format!("Pattern {}: {:?}", i, pattern),
                    mitigation: Some("Add input size limits and timeouts".to_string()),
                });
            }
        }

        result.duration = start_time.elapsed();
        result.security_level = self.assess_security_level(&result.security_issues);

        Ok(result)
    }

    /// Test bounds checking with edge cases
    pub fn test_bounds_checking<F>(&self, test_function: F) -> CoreResult<SecurityTestResult>
    where
        F: Fn(usize, usize) -> CoreResult<()>,
    {
        let start_time = Instant::now();
        let mut result = SecurityTestResult {
            test_name: "bounds_checking".to_string(),
            tests_executed: 0,
            vulnerabilities_found: 0,
            duration: Duration::from_secs(0),
            security_issues: Vec::new(),
            security_level: SecurityLevel::Secure,
        };

        // Test boundary conditions
        let test_cases = vec![
            (0, 0),                   // Zero-zero case
            (0, 1),                   // Zero start
            (1, 0),                   // Zero length
            (usize::MAX, 1),          // Maximum start
            (1, usize::MAX),          // Maximum length
            (usize::MAX, usize::MAX), // Maximum both
            (usize::MAX - 1, 2),      // Overflow potential
        ];

        for (start, length) in test_cases {
            result.tests_executed += 1;

            let test_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                test_function(start, length)
            }));

            match test_result {
                Ok(Ok(())) => {
                    // Check if this should have been an error
                    if start.saturating_add(length) < start {
                        // Integer overflow occurred but wasn't caught
                        result.vulnerabilities_found += 1;
                        result.security_issues.push(SecurityIssue {
                            severity: SecuritySeverity::High,
                            category: SecurityCategory::IntegerOverflow,
                            description: "Integer overflow not detected".to_string(),
                            trigger_input: format!("start={}, length={}", start, length),
                            mitigation: Some(
                                "Add overflow checks in bounds validation".to_string(),
                            ),
                        });
                    }
                }
                Ok(Err(_)) => {
                    // Expected error for invalid bounds
                }
                Err(_) => {
                    // Panic indicates potential vulnerability
                    result.vulnerabilities_found += 1;
                    result.security_issues.push(SecurityIssue {
                        severity: SecuritySeverity::Critical,
                        category: SecurityCategory::OutOfBounds,
                        description: "Function panicked on bounds check".to_string(),
                        trigger_input: format!("start={}, length={}", start, length),
                        mitigation: Some("Implement safe bounds checking".to_string()),
                    });
                }
            }
        }

        result.duration = start_time.elapsed();
        result.security_level = self.assess_security_level(&result.security_issues);

        Ok(result)
    }

    /// Generate malicious input patterns
    #[allow(clippy::vec_init_then_push)]
    fn generate_malicious_patterns(&self) -> Vec<Vec<u8>> {
        let mut patterns = Vec::new();

        // Empty input
        patterns.push(vec![]);

        // Very large input
        patterns.push(vec![0xAA; self.config.max_input_size]);

        // Input with null bytes
        patterns.push(vec![0x00; 100]);

        // Input with high bytes
        patterns.push(vec![0xFF; 100]);

        // Patterns that might trigger buffer overflows
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            if size <= self.config.max_input_size {
                patterns.push(vec![0x41; size]); // 'A' pattern
                patterns.push(vec![0x90; size]); // NOP pattern
            }
        }

        // Format string patterns (might be relevant for some functions)
        patterns.push(b"%s%s%s%s%s%s%s%s%s%s".to_vec());
        patterns.push(b"%x%x%x%x%x%x%x%x%x%x".to_vec());
        patterns.push(b"%n%n%n%n%n%n%n%n%n%n".to_vec());

        // SQL injection patterns (might be relevant for database operations)
        patterns.push(b"' OR '1'='1".to_vec());
        patterns.push(b"'; DROP TABLE users; --".to_vec());

        // Path traversal patterns
        patterns.push(b"../../../etc/passwd".to_vec());
        patterns.push(b"..\\..\\..\\windows\\system32\\config\\sam".to_vec());

        // Unicode/encoding issues
        patterns.push(vec![0xC0, 0x80]); // Overlong encoding
        patterns.push(vec![0xED, 0xA0, 0x80]); // Surrogate
        patterns.push(vec![0xFF, 0xFE]); // BOM

        // Limit to requested number of patterns
        patterns.truncate(self.config.malicious_patterns);
        patterns
    }

    /// Assess overall security level based on issues found
    fn assess_security_level(&self, issues: &[SecurityIssue]) -> SecurityLevel {
        let critical_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::Critical)
            .count();
        let high_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::High)
            .count();
        let medium_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::Medium)
            .count();

        if critical_count > 0 {
            SecurityLevel::Insecure
        } else if high_count > 2 {
            SecurityLevel::Vulnerable
        } else if high_count > 0 || medium_count > 5 {
            SecurityLevel::Weak
        } else if medium_count > 0 {
            SecurityLevel::Secure
        } else {
            SecurityLevel::Hardened
        }
    }
}

/// Memory safety security tester
pub struct MemorySafetyTester {
    #[allow(dead_code)]
    config: SecurityTestConfig,
}

impl MemorySafetyTester {
    /// Create a new memory safety tester
    pub fn new(config: SecurityTestConfig) -> Self {
        Self { config }
    }

    /// Test for potential memory leaks
    pub fn test_memory_leaks<F>(&self, test_function: F) -> CoreResult<SecurityTestResult>
    where
        F: Fn() -> CoreResult<()>,
    {
        let start_time = Instant::now();
        let mut result = SecurityTestResult {
            test_name: "memory_leak_detection".to_string(),
            tests_executed: 0,
            vulnerabilities_found: 0,
            duration: Duration::from_secs(0),
            security_issues: Vec::new(),
            security_level: SecurityLevel::Secure,
        };

        // Get initial memory usage
        let initial_memory = self.get_memory_usage();

        // Run function multiple times to detect leaks
        for i in 0..100 {
            result.tests_executed += 1;

            match test_function() {
                Ok(()) => {}
                Err(_) => {
                    // Function errors are not necessarily security issues
                }
            }

            // Check memory usage periodically
            if i % 10 == 0 {
                if let Ok(current_memory) = self.get_memory_usage() {
                    if let Ok(initial) = initial_memory {
                        let memory_growth = current_memory.saturating_sub(initial);

                        // If memory has grown significantly, it might indicate a leak
                        if memory_growth > 10 * 1024 * 1024 {
                            // 10MB threshold
                            result.vulnerabilities_found += 1;
                            result.security_issues.push(SecurityIssue {
                                severity: SecuritySeverity::Medium,
                                category: SecurityCategory::MemorySafety,
                                description: format!(
                                    "Potential memory leak detected: {} MB growth",
                                    memory_growth / (1024 * 1024)
                                ),
                                trigger_input: format!("After {} iterations", i),
                                mitigation: Some(
                                    "Review memory management and cleanup".to_string(),
                                ),
                            });
                            break; // Stop testing once leak is detected
                        }
                    }
                }
            }
        }

        result.duration = start_time.elapsed();
        result.security_level = self.assess_security_level(&result.security_issues);

        Ok(result)
    }

    /// Test for use-after-free vulnerabilities (conceptual, as Rust prevents most of these)
    pub fn test_use_after_free(&self) -> CoreResult<SecurityTestResult> {
        let start_time = Instant::now();
        let result = SecurityTestResult {
            test_name: "use_after_free_detection".to_string(),
            tests_executed: 1,
            vulnerabilities_found: 0,
            duration: start_time.elapsed(),
            security_issues: vec![SecurityIssue {
                severity: SecuritySeverity::Info,
                category: SecurityCategory::MemorySafety,
                description: "Rust's ownership system prevents use-after-free vulnerabilities"
                    .to_string(),
                trigger_input: "N/A".to_string(),
                mitigation: Some("Continue using Rust's safe memory management".to_string()),
            }],
            security_level: SecurityLevel::Hardened,
        };

        Ok(result)
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> CoreResult<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to read memory status: {}",
                    e
                )))
            })?;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().map_err(|e| {
                            CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                                "Failed to parse memory: {}",
                                e
                            )))
                        })?;
                        return Ok(kb * 1024);
                    }
                }
            }
        }

        // Fallback for non-Linux systems
        Ok(0)
    }

    /// Assess security level
    fn assess_security_level(&self, issues: &[SecurityIssue]) -> SecurityLevel {
        let critical_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::Critical)
            .count();
        let high_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::High)
            .count();
        let medium_count = issues
            .iter()
            .filter(|i| i.severity == SecuritySeverity::Medium)
            .count();

        if critical_count > 0 {
            SecurityLevel::Insecure
        } else if high_count > 0 {
            SecurityLevel::Vulnerable
        } else if medium_count > 2 {
            SecurityLevel::Weak
        } else if medium_count > 0 {
            SecurityLevel::Secure
        } else {
            SecurityLevel::Hardened
        }
    }
}

/// High-level security testing utilities
pub struct SecurityTestUtils;

impl SecurityTestUtils {
    /// Create a comprehensive security test suite
    pub fn create_security_test_suite(name: &str, config: TestConfig) -> crate::testing::TestSuite {
        let mut suite = crate::testing::TestSuite::new(name, config);
        let security_config = SecurityTestConfig::default()
            .with_malicious_patterns(100)
            .with_max_input_size(1024);

        // Input validation tests
        let security_config_1 = security_config.clone();
        suite.add_test("malicious_input_validation", move |_runner| {
            let tester = InputValidationTester::new(security_config_1.clone());

            let result = tester.test_malicious_inputs(|input| {
                // Test a simple validation function
                if input.len() > 1000 {
                    Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                        "Input too large",
                    )))
                } else if input.is_empty() {
                    Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                        "Input cannot be empty",
                    )))
                } else {
                    Ok(())
                }
            })?;

            // Check if any critical vulnerabilities were found
            if result.security_level == SecurityLevel::Insecure {
                return Ok(TestResult::failure(
                    result.duration,
                    result.tests_executed,
                    format!(
                        "Critical security vulnerabilities found: {}",
                        result.vulnerabilities_found
                    ),
                ));
            }

            Ok(TestResult::success(result.duration, result.tests_executed))
        });

        // Bounds checking tests
        let security_config_2 = security_config.clone();
        suite.add_test("bounds_checking", move |_runner| {
            let tester = InputValidationTester::new(security_config_2.clone());

            let result = tester.test_bounds_checking(|start, length| {
                // Test bounds checking function
                let end = start.checked_add(length).ok_or_else(|| {
                    CoreError::ValidationError(crate::error::ErrorContext::new(
                        "Integer overflow in bounds calculation",
                    ))
                })?;

                if end > 1000 {
                    Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                        "Bounds exceed maximum allowed",
                    )))
                } else {
                    Ok(())
                }
            })?;

            if result.security_level == SecurityLevel::Insecure
                || result.security_level == SecurityLevel::Vulnerable
            {
                return Ok(TestResult::failure(
                    result.duration,
                    result.tests_executed,
                    format!(
                        "Bounds checking vulnerabilities found: {}",
                        result.vulnerabilities_found
                    ),
                ));
            }

            Ok(TestResult::success(result.duration, result.tests_executed))
        });

        // Memory safety tests
        let security_config_3 = security_config.clone();
        suite.add_test("memory_safety", move |_runner| {
            let tester = MemorySafetyTester::new(security_config_3.clone());

            let result = tester.test_memory_leaks(|| {
                // Test function that should not leak memory
                let _data = vec![0u8; 1000];
                Ok(())
            })?;

            if result.vulnerabilities_found > 0 {
                return Ok(TestResult::failure(
                    result.duration,
                    result.tests_executed,
                    format!(
                        "Memory safety issues found: {}",
                        result.vulnerabilities_found
                    ),
                ));
            }

            Ok(TestResult::success(result.duration, result.tests_executed))
        });

        // Use-after-free test (informational for Rust)
        let security_config_clone2 = security_config.clone();
        suite.add_test("use_after_free", move |_runner| {
            let tester = MemorySafetyTester::new(security_config_clone2.clone());
            let result = tester.test_use_after_free()?;

            // This should always pass in Rust
            Ok(TestResult::success(result.duration, result.tests_executed))
        });

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_config() {
        let config = SecurityTestConfig::new()
            .with_max_input_size(2048)
            .with_malicious_patterns(500)
            .with_bounds_checking(true)
            .with_memory_safety(false);

        assert_eq!(config.max_input_size, 2048);
        assert_eq!(config.malicious_patterns, 500);
        assert!(config.test_bounds_checking);
        assert!(!config.test_memory_safety);
    }

    #[test]
    fn test_security_severity() {
        assert_eq!(SecuritySeverity::Critical, SecuritySeverity::Critical);
        assert_ne!(SecuritySeverity::Critical, SecuritySeverity::High);
    }

    #[test]
    fn test_security_level_assessment() {
        let tester = InputValidationTester::new(SecurityTestConfig::default());

        // Test with no issues
        let level = tester.assess_security_level(&[]);
        assert_eq!(level, SecurityLevel::Hardened);

        // Test with critical issue
        let critical_issue = SecurityIssue {
            severity: SecuritySeverity::Critical,
            category: SecurityCategory::BufferOverflow,
            description: "Test issue".to_string(),
            trigger_input: "test".to_string(),
            mitigation: None,
        };
        let level = tester.assess_security_level(&[critical_issue]);
        assert_eq!(level, SecurityLevel::Insecure);
    }

    #[test]
    fn test_malicious_pattern_generation() {
        let tester =
            InputValidationTester::new(SecurityTestConfig::default().with_malicious_patterns(10));
        let patterns = tester.generate_malicious_patterns();

        assert_eq!(patterns.len(), 10);
        assert!(patterns.iter().any(|p| p.is_empty())); // Should include empty pattern
    }
}
