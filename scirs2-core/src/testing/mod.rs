//! # Testing Framework for `SciRS2` Core
//!
//! This module provides comprehensive testing infrastructure including:
//! - Property-based testing for mathematical properties
//! - Fuzzing tests for edge case discovery
//! - Stress testing for memory and performance limits
//! - Large-scale dataset testing
//! - Security audit utilities
//! - Integration testing with dependent modules
//! - Ecosystem integration testing for 1.0 release readiness
//!
//! ## Features
//!
//! - **Property-based testing**: Automatic generation of test cases to verify mathematical properties
//! - **Fuzzing**: Random input generation to discover edge cases and potential vulnerabilities
//! - **Stress testing**: Memory pressure and performance limit testing
//! - **Large-scale testing**: Multi-GB dataset handling and processing
//! - **Security auditing**: Input validation and bounds checking verification
//! - **Integration testing**: Cross-module compatibility and communication validation
//! - **Ecosystem integration**: Complete ecosystem validation for 1.0 release readiness

pub mod ecosystem_integration;
pub mod fuzzing;
pub mod integration;
pub mod large_scale;
pub mod propertybased;
pub mod security;
pub mod stress;

use crate::error::CoreResult;
#[cfg(target_os = "linux")]
use crate::error::{CoreError, ErrorContext};
use std::time::{Duration, Instant};

/// Test execution configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Maximum execution time for a single test
    pub timeout: Duration,
    /// Number of iterations for property-based tests
    pub iterations: usize,
    /// Memory limit for stress tests (in bytes)
    pub memory_limit: usize,
    /// Enable verbose logging during tests
    pub verbose: bool,
    /// Random seed for reproducible test runs
    pub seed: Option<u64>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            iterations: 1000,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            verbose: false,
            seed: None,
        }
    }
}

impl TestConfig {
    /// Create a new test configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the timeout for test execution
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the number of iterations for property-based tests
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the memory limit for stress tests
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set a random seed for reproducible tests
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Test result with performance metrics
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Whether the test passed
    pub passed: bool,
    /// Test execution time
    pub duration: Duration,
    /// Number of test cases executed
    pub cases_executed: usize,
    /// Memory usage during test (in bytes)
    pub memory_used: usize,
    /// Error information if test failed
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl TestResult {
    /// Create a successful test result
    pub fn success(duration: Duration, cases: usize) -> Self {
        Self {
            passed: true,
            duration,
            cases_executed: cases,
            memory_used: 0,
            error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a failed test result
    pub fn failure(duration: Duration, cases: usize, error: String) -> Self {
        Self {
            passed: false,
            duration,
            cases_executed: cases,
            memory_used: 0,
            error: Some(error),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add memory usage information
    pub fn with_memory_usage(mut self, memory: usize) -> Self {
        self.memory_used = memory;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Test runner that executes tests with timeout and resource monitoring
pub struct TestRunner {
    config: TestConfig,
}

impl TestRunner {
    /// Create a new test runner with the given configuration
    pub fn new(config: TestConfig) -> Self {
        Self { config }
    }

    /// Execute a test function with timeout and monitoring
    pub fn execute<F>(&self, test_name: &str, testfn: F) -> CoreResult<TestResult>
    where
        F: FnOnce() -> CoreResult<()>,
    {
        if self.config.verbose {
            println!("Executing test: {}", test_name);
        }

        let start_time = Instant::now();

        // Execute the test with timeout monitoring
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(testfn));

        let duration = start_time.elapsed();

        match result {
            Ok(Ok(())) => {
                if self.config.verbose {
                    println!("Test {} passed in {:?}", test_name, duration);
                }
                Ok(TestResult::success(duration, 1))
            }
            Ok(Err(e)) => {
                if self.config.verbose {
                    println!("Test {} failed: {:?}", test_name, e);
                }
                Ok(TestResult::failure(duration, 1, format!("{e:?}")))
            }
            Err(panic) => {
                let errormsg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "Unknown panic".to_string()
                };

                if self.config.verbose {
                    println!("Test {} panicked: {}", test_name, errormsg);
                }
                Ok(TestResult::failure(duration, 1, errormsg))
            }
        }
    }

    /// Execute multiple test iterations
    pub fn execute_iterations<F>(&self, test_name: &str, testfn: F) -> CoreResult<TestResult>
    where
        F: Fn(usize) -> CoreResult<()>,
    {
        if self.config.verbose {
            println!(
                "Executing {} iterations of test: {}",
                self.config.iterations, test_name
            );
        }

        let start_time = Instant::now();
        let mut cases_executed = 0;
        #[cfg(target_os = "linux")]
        let mut max_memory = 0;
        #[cfg(not(target_os = "linux"))]
        let max_memory = 0;

        for i in 0..self.config.iterations {
            // Check timeout
            if start_time.elapsed() > self.config.timeout {
                return Ok(TestResult::failure(
                    start_time.elapsed(),
                    cases_executed,
                    format!("Test timed out after {} iterations", cases_executed),
                ));
            }

            // Execute single iteration
            match testfn(i) {
                Ok(()) => {
                    cases_executed += 1;

                    // Monitor memory usage (simplified)
                    #[cfg(target_os = "linux")]
                    {
                        if let Ok(memory) = self.get_memory_usage() {
                            max_memory = max_memory.max(memory);

                            if memory > self.config.memory_limit {
                                return Ok(TestResult::failure(
                                    start_time.elapsed(),
                                    cases_executed,
                                    format!("Memory limit exceeded: {} bytes", memory),
                                )
                                .with_memory_usage(memory));
                            }
                        }
                    }
                }
                Err(e) => {
                    return Ok(TestResult::failure(
                        start_time.elapsed(),
                        cases_executed,
                        format!("Iteration {}: {:?}", i, e),
                    )
                    .with_memory_usage(max_memory));
                }
            }
        }

        let duration = start_time.elapsed();
        if self.config.verbose {
            println!(
                "Test {} completed {} iterations in {:?}",
                test_name, cases_executed, duration
            );
        }

        Ok(TestResult::success(duration, cases_executed).with_memory_usage(max_memory))
    }

    /// Get current memory usage (Linux-specific implementation)
    #[cfg(target_os = "linux")]
    #[allow(dead_code)]
    fn get_memory_usage(&self) -> CoreResult<usize> {
        use std::fs;

        let status = fs::read_to_string("/proc/self/status").map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to read /proc/self/status: {}",
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
                    return Ok(kb * 1024); // Convert KB to bytes
                }
            }
        }

        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Could not find VmRSS in /proc/self/status"),
        ))
    }

    /// Get current memory usage (fallback implementation)
    #[cfg(not(target_os = "linux"))]
    #[allow(dead_code)]
    fn get_memory_usage(&self) -> CoreResult<usize> {
        // Fallback: return 0 (no monitoring on non-Linux systems)
        Ok(0)
    }
}

/// Type alias for test functions
type TestFn = Box<dyn Fn(&TestRunner) -> CoreResult<TestResult> + Send + Sync>;

/// Test suite for organizing and running multiple tests
pub struct TestSuite {
    name: String,
    tests: Vec<TestFn>,
    config: TestConfig,
}

impl TestSuite {
    /// Create a new test suite
    pub fn new(name: &str, config: TestConfig) -> Self {
        Self {
            name: name.to_string(),
            tests: Vec::new(),
            config,
        }
    }

    /// Add a test to the suite
    pub fn add_test<F>(&mut self, test_name: &str, testfn: F)
    where
        F: Fn(&TestRunner) -> CoreResult<TestResult> + Send + Sync + 'static,
    {
        let name = test_name.to_string();
        self.tests.push(Box::new(move |runner| {
            println!("Running test: {}", name);
            testfn(runner)
        }));
    }

    /// Run all tests in the suite
    pub fn run(&self) -> CoreResult<Vec<TestResult>> {
        println!("Running test suite: {}", self.name);

        let runner = TestRunner::new(self.config.clone());
        let mut results = Vec::new();

        for (i, test) in self.tests.iter().enumerate() {
            println!("Test {}/{}", i + 1, self.tests.len());
            match test(&runner) {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    results.push(TestResult::failure(
                        Duration::from_secs(0),
                        0,
                        format!("{:?}", e),
                    ));
                }
            }
        }

        // Print summary
        let passed = results.iter().filter(|r| r.passed).count();
        let total = results.len();
        println!(
            "Test suite {} completed: {}/{} tests passed",
            self.name, passed, total
        );

        Ok(results)
    }
}
