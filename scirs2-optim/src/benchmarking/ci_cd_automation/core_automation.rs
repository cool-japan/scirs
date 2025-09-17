//! Core CI/CD Automation Engine
//!
//! This module provides the main automation engine that orchestrates all CI/CD automation
//! components including test execution, performance gates, reporting, artifact management,
//! and external integrations.

use crate::benchmarking::performance_regression_detector::{
    EnvironmentInfo, PerformanceRegressionDetector, RegressionConfig,
};
use crate::error::{OptimError, Result};
use std::path::Path;
use std::time::{Duration, SystemTime};

use super::config::CiCdAutomationConfig;
use super::test_execution::{
    PerformanceTestSuite, TestSuiteConfig, CiCdTestResult, TestSuiteStatistics,
    CiCdContext, TriggerEvent, GitInfo
};
use super::reporting::{ReportGenerator, GeneratedReport};
use super::artifact_management::ArtifactManager;
use super::integrations::{IntegrationManager, IntegrationNotification, NotificationType, NotificationPriority};
use super::performance_gates::{PerformanceGateEvaluator, GateResult};

/// Main CI/CD automation engine
#[derive(Debug)]
pub struct CiCdAutomation {
    /// Performance regression detector
    pub regression_detector: PerformanceRegressionDetector,
    /// CI/CD configuration
    pub config: CiCdAutomationConfig,
    /// Current environment information
    pub environment: EnvironmentInfo,
    /// Performance test suite
    pub test_suite: PerformanceTestSuite,
    /// Report generator
    pub report_generator: ReportGenerator,
    /// Artifact manager
    pub artifact_manager: ArtifactManager,
    /// Integration manager
    pub integration_manager: IntegrationManager,
    /// Performance gate evaluator
    pub gate_evaluator: PerformanceGateEvaluator,
    /// Automation statistics
    pub statistics: AutomationStatistics,
    /// Current execution context
    pub execution_context: Option<AutomationExecutionContext>,
}

/// Automation execution context
#[derive(Debug, Clone)]
pub struct AutomationExecutionContext {
    /// Execution ID
    pub execution_id: String,
    /// CI/CD context
    pub ci_context: CiCdContext,
    /// Git information
    pub git_info: Option<GitInfo>,
    /// Execution start time
    pub start_time: SystemTime,
    /// Execution trigger
    pub trigger_event: TriggerEvent,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Automation execution result
#[derive(Debug, Clone)]
pub struct AutomationExecutionResult {
    /// Execution context
    pub context: AutomationExecutionContext,
    /// Test execution results
    pub test_results: Vec<CiCdTestResult>,
    /// Test suite statistics
    pub statistics: TestSuiteStatistics,
    /// Gate evaluation result
    pub gate_result: GateResult,
    /// Generated reports
    pub reports: Vec<GeneratedReport>,
    /// Execution duration
    pub duration: Duration,
    /// Overall success status
    pub success: bool,
    /// Error messages (if any)
    pub errors: Vec<String>,
    /// Warnings (if any)
    pub warnings: Vec<String>,
}

/// Automation statistics
#[derive(Debug, Clone)]
pub struct AutomationStatistics {
    /// Total executions
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Total tests executed
    pub total_tests_executed: u64,
    /// Total test failures
    pub total_test_failures: u64,
    /// Total reports generated
    pub total_reports_generated: u64,
    /// Total artifacts stored
    pub total_artifacts_stored: u64,
    /// Total gate failures
    pub total_gate_failures: u64,
    /// Total integrations triggered
    pub total_integrations_triggered: u64,
    /// Average execution duration
    pub average_execution_duration: Duration,
    /// Last execution time
    pub last_execution_time: Option<SystemTime>,
}

/// Automation step result
#[derive(Debug, Clone)]
pub struct AutomationStepResult {
    /// Step name
    pub step_name: String,
    /// Step success status
    pub success: bool,
    /// Step duration
    pub duration: Duration,
    /// Step output/message
    pub message: String,
    /// Step metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl CiCdAutomation {
    /// Create a new CI/CD automation engine
    pub fn new(config: CiCdAutomationConfig) -> Result<Self> {
        let regression_detector = PerformanceRegressionDetector::new(RegressionConfig::default())?;
        let environment = Self::detect_environment()?;
        let test_suite = PerformanceTestSuite::new(TestSuiteConfig::default())?;
        let report_generator = ReportGenerator::new(config.reporting.clone())?;
        let artifact_manager = ArtifactManager::new(config.artifact_storage.clone())?;
        let integration_manager = IntegrationManager::new(config.integrations.clone())?;
        let gate_evaluator = PerformanceGateEvaluator::new(config.performance_gates.clone())?;

        Ok(Self {
            regression_detector,
            config,
            environment,
            test_suite,
            report_generator,
            artifact_manager,
            integration_manager,
            gate_evaluator,
            statistics: AutomationStatistics::default(),
            execution_context: None,
        })
    }

    /// Execute the complete CI/CD automation pipeline
    pub fn execute_automation(&mut self, trigger: TriggerEvent) -> Result<AutomationExecutionResult> {
        let execution_start = SystemTime::now();
        let execution_id = uuid::Uuid::new_v4().to_string();

        // Create execution context
        let context = AutomationExecutionContext {
            execution_id: execution_id.clone(),
            ci_context: self.detect_ci_context()?,
            git_info: self.gather_git_info().ok(),
            start_time: execution_start,
            trigger_event: trigger,
            metadata: std::collections::HashMap::new(),
        };

        self.execution_context = Some(context.clone());

        let mut step_results = Vec::new();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Step 1: Environment validation
        let env_result = self.execute_step("environment_validation", || {
            self.validate_environment()
        });
        step_results.push(env_result.clone());
        if !env_result.success {
            errors.push(format!("Environment validation failed: {}", env_result.message));
        }

        // Step 2: Test execution
        let test_result = self.execute_step("test_execution", || {
            self.execute_tests()
        });
        step_results.push(test_result.clone());

        let (test_results, test_statistics) = if test_result.success {
            // Extract test results from metadata (simplified)
            let results = self.test_suite.results.clone();
            let stats = self.test_suite.get_statistics();
            (results, stats)
        } else {
            errors.push(format!("Test execution failed: {}", test_result.message));
            (Vec::new(), TestSuiteStatistics {
                total_tests: 0,
                passed: 0,
                failed: 0,
                skipped: 0,
                errors: 0,
                total_duration: Duration::ZERO,
                success_rate: 0.0,
            })
        };

        // Step 3: Performance gate evaluation
        let gate_result = if !test_results.is_empty() {
            let gate_step = self.execute_step("gate_evaluation", || {
                self.evaluate_performance_gates(&test_results, &test_statistics)
            });
            step_results.push(gate_step.clone());

            if !gate_step.success {
                warnings.push(format!("Gate evaluation had issues: {}", gate_step.message));
            }

            // Extract gate result from evaluator
            self.gate_evaluator.evaluate_gates(&test_results, &test_statistics)?
        } else {
            GateResult {
                status: super::performance_gates::OverallGateStatus::Skipped,
                gate_results: Vec::new(),
                summary: super::performance_gates::GateEvaluationSummary {
                    total_gates: 0,
                    gates_passed: 0,
                    gates_failed: 0,
                    gates_warnings: 0,
                    gates_skipped: 0,
                    critical_failures: 0,
                    improvements_detected: 0,
                    regression_severity: super::performance_gates::RegressionSeverity::None,
                },
                timestamp: SystemTime::now(),
            }
        };

        // Step 4: Report generation
        let report_step = self.execute_step("report_generation", || {
            self.generate_reports(&test_results, &test_statistics)
        });
        step_results.push(report_step.clone());

        let reports = if report_step.success {
            // Get generated reports
            self.report_generator.generated_reports.clone()
        } else {
            warnings.push(format!("Report generation failed: {}", report_step.message));
            Vec::new()
        };

        // Step 5: Artifact storage
        let artifact_step = self.execute_step("artifact_storage", || {
            self.store_artifacts(&reports)
        });
        step_results.push(artifact_step.clone());
        if !artifact_step.success {
            warnings.push(format!("Artifact storage failed: {}", artifact_step.message));
        }

        // Step 6: Integration notifications
        let integration_step = self.execute_step("integration_notifications", || {
            self.send_notifications(&test_results, &test_statistics, &gate_result)
        });
        step_results.push(integration_step.clone());
        if !integration_step.success {
            warnings.push(format!("Integration notifications failed: {}", integration_step.message));
        }

        // Calculate overall success
        let success = errors.is_empty() && test_statistics.success_rate > 0.0;

        // Update statistics
        self.update_statistics(&test_results, &test_statistics, &reports, success);

        let duration = SystemTime::now().duration_since(execution_start).unwrap_or_default();

        Ok(AutomationExecutionResult {
            context,
            test_results,
            statistics: test_statistics,
            gate_result,
            reports,
            duration,
            success,
            errors,
            warnings,
        })
    }

    /// Execute a single automation step with error handling and timing
    fn execute_step<F, T>(&self, step_name: &str, step_fn: F) -> AutomationStepResult
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = SystemTime::now();

        match step_fn() {
            Ok(_) => AutomationStepResult {
                step_name: step_name.to_string(),
                success: true,
                duration: SystemTime::now().duration_since(start_time).unwrap_or_default(),
                message: format!("{} completed successfully", step_name),
                metadata: std::collections::HashMap::new(),
            },
            Err(e) => AutomationStepResult {
                step_name: step_name.to_string(),
                success: false,
                duration: SystemTime::now().duration_since(start_time).unwrap_or_default(),
                message: format!("{} failed: {}", step_name, e),
                metadata: std::collections::HashMap::new(),
            },
        }
    }

    /// Detect current environment information
    fn detect_environment() -> Result<EnvironmentInfo> {
        Ok(EnvironmentInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_count: num_cpus::get(),
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
            environment_vars: std::env::vars().collect(),
        })
    }

    /// Detect CI/CD context
    fn detect_ci_context(&self) -> Result<CiCdContext> {
        let platform_config = self.config.get_platform_config();

        let build_id = std::env::var(platform_config.env_vars.get("BUILD_ID").unwrap_or(&"BUILD_ID".to_string()))
            .unwrap_or_else(|_| "unknown".to_string());

        let build_number = std::env::var("BUILD_NUMBER")
            .ok()
            .and_then(|s| s.parse().ok());

        let trigger = if std::env::var("GITHUB_EVENT_NAME").is_ok() {
            match std::env::var("GITHUB_EVENT_NAME").as_deref() {
                Ok("push") => TriggerEvent::Push,
                Ok("pull_request") => TriggerEvent::PullRequest,
                Ok("schedule") => TriggerEvent::Schedule,
                _ => TriggerEvent::Manual,
            }
        } else {
            TriggerEvent::Manual
        };

        Ok(CiCdContext {
            platform: self.config.platform.clone(),
            build_id,
            build_number,
            trigger,
            environment_vars: std::env::vars().collect(),
            build_url: std::env::var("BUILD_URL").ok(),
            pull_request: None, // Would extract from environment
            triggered_by: std::env::var("GITHUB_ACTOR").ok(),
        })
    }

    /// Gather Git repository information
    fn gather_git_info(&self) -> Result<GitInfo> {
        use std::process::Command;

        let commit_hash = Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let branch = Command::new("git")
            .args(&["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let commit_message = Command::new("git")
            .args(&["log", "-1", "--pretty=%B"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .ok();

        let author = Command::new("git")
            .args(&["log", "-1", "--pretty=%an"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .ok();

        let is_clean = Command::new("git")
            .args(&["diff", "--quiet"])
            .status()
            .map(|status| status.success())
            .unwrap_or(false);

        Ok(GitInfo {
            commit_hash,
            branch,
            commit_message,
            author,
            commit_time: None, // Would parse from git log
            repository_url: std::env::var("GITHUB_REPOSITORY").ok()
                .map(|repo| format!("https://github.com/{}", repo)),
            is_clean,
        })
    }

    /// Validate execution environment
    fn validate_environment(&self) -> Result<()> {
        // Check if required tools are available
        let required_tools = vec!["git"];

        for tool in required_tools {
            if std::process::Command::new(tool)
                .arg("--version")
                .output()
                .is_err()
            {
                return Err(OptimError::InvalidConfig(
                    format!("Required tool not found: {}", tool)
                ));
            }
        }

        // Validate configuration
        self.config.validate().map_err(|e| OptimError::InvalidConfig(e))?;

        Ok(())
    }

    /// Execute performance tests
    fn execute_tests(&mut self) -> Result<()> {
        // Add sample test cases if none exist
        if self.test_suite.test_cases.is_empty() {
            self.add_default_test_cases()?;
        }

        // Set execution context
        if let Some(context) = &self.execution_context {
            self.test_suite.set_context(context.ci_context.clone());
        }

        // Execute test suite
        let _results = self.test_suite.execute()?;

        Ok(())
    }

    /// Add default test cases for demonstration
    fn add_default_test_cases(&mut self) -> Result<()> {
        use super::test_execution::{
            PerformanceTestCase, TestCategory, TestExecutor, EnvironmentRequirements
        };

        let test_case = PerformanceTestCase {
            name: "simple_benchmark".to_string(),
            category: TestCategory::Benchmark,
            executor: TestExecutor::Criterion,
            parameters: std::collections::HashMap::from([
                ("iterations".to_string(), "1000".to_string()),
            ]),
            baseline: None,
            timeout: Some(300), // 5 minutes
            iterations: 5,
            warmup_iterations: 2,
            dependencies: Vec::new(),
            tags: vec!["performance".to_string(), "benchmark".to_string()],
            environment_requirements: EnvironmentRequirements::default(),
            custom_config: std::collections::HashMap::new(),
        };

        self.test_suite.add_test_case(test_case);
        Ok(())
    }

    /// Evaluate performance gates
    fn evaluate_performance_gates(&mut self, test_results: &[CiCdTestResult], statistics: &TestSuiteStatistics) -> Result<()> {
        if self.config.performance_gates.enabled {
            let _gate_result = self.gate_evaluator.evaluate_gates(test_results, statistics)?;
            // Gate result is stored in the evaluator
        }
        Ok(())
    }

    /// Generate reports
    fn generate_reports(&mut self, test_results: &[CiCdTestResult], statistics: &TestSuiteStatistics) -> Result<()> {
        let output_dir = Path::new("./reports");
        let _reports = self.report_generator.generate_reports(test_results, statistics, output_dir)?;
        Ok(())
    }

    /// Store artifacts
    fn store_artifacts(&mut self, reports: &[GeneratedReport]) -> Result<()> {
        for report in reports {
            let tags = vec![
                format!("report_type:{:?}", report.report_type),
                "automated".to_string(),
            ];

            let _artifact_url = self.artifact_manager.upload_artifact(
                &report.file_path,
                &format!("reports/{}", report.file_path.file_name().unwrap().to_string_lossy()),
                tags,
            )?;
        }
        Ok(())
    }

    /// Send integration notifications
    fn send_notifications(
        &mut self,
        test_results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
        gate_result: &GateResult,
    ) -> Result<()> {
        // Handle test completion notifications
        self.integration_manager.handle_test_completion(test_results, statistics)?;

        // Handle performance gate notifications
        if gate_result.status != super::performance_gates::OverallGateStatus::AllPassed {
            let notification = IntegrationNotification {
                notification_type: NotificationType::PerformanceRegression,
                title: "Performance Gates Failed".to_string(),
                message: format!(
                    "Performance gate evaluation failed: {}/{} gates failed",
                    gate_result.summary.gates_failed,
                    gate_result.summary.total_gates
                ),
                priority: NotificationPriority::High,
                data: std::collections::HashMap::new(),
                timestamp: SystemTime::now(),
            };

            self.integration_manager.send_notification(&notification)?;
        }

        // Handle report generation notifications
        for report in &self.report_generator.generated_reports {
            self.integration_manager.handle_report_generated(report)?;
        }

        Ok(())
    }

    /// Update automation statistics
    fn update_statistics(
        &mut self,
        test_results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
        reports: &[GeneratedReport],
        success: bool,
    ) {
        self.statistics.total_executions += 1;

        if success {
            self.statistics.successful_executions += 1;
        } else {
            self.statistics.failed_executions += 1;
        }

        self.statistics.total_tests_executed += statistics.total_tests as u64;
        self.statistics.total_test_failures += statistics.failed as u64;
        self.statistics.total_reports_generated += reports.len() as u64;

        // Update average execution duration
        if let Some(context) = &self.execution_context {
            let duration = SystemTime::now().duration_since(context.start_time).unwrap_or_default();

            let total_duration = self.statistics.average_execution_duration * (self.statistics.total_executions - 1) as u32 + duration;
            self.statistics.average_execution_duration = total_duration / self.statistics.total_executions as u32;
        }

        self.statistics.last_execution_time = Some(SystemTime::now());
    }

    /// Get automation statistics
    pub fn get_statistics(&self) -> &AutomationStatistics {
        &self.statistics
    }

    /// Get current execution context
    pub fn get_execution_context(&self) -> Option<&AutomationExecutionContext> {
        self.execution_context.as_ref()
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = AutomationStatistics::default();
    }

    /// Validate automation configuration
    pub fn validate_configuration(&self) -> Result<()> {
        self.config.validate().map_err(|e| OptimError::InvalidConfig(e))
    }

    /// Get system health status
    pub fn get_health_status(&self) -> HealthStatus {
        let integration_health = self.integration_manager.get_health_status();

        HealthStatus {
            overall: if integration_health == super::integrations::IntegrationStatus::Healthy {
                ComponentHealth::Healthy
            } else {
                ComponentHealth::Warning
            },
            components: std::collections::HashMap::from([
                ("test_execution".to_string(), ComponentHealth::Healthy),
                ("reporting".to_string(), ComponentHealth::Healthy),
                ("artifact_management".to_string(), ComponentHealth::Healthy),
                ("performance_gates".to_string(), ComponentHealth::Healthy),
                ("integrations".to_string(), match integration_health {
                    super::integrations::IntegrationStatus::Healthy => ComponentHealth::Healthy,
                    super::integrations::IntegrationStatus::Warning => ComponentHealth::Warning,
                    super::integrations::IntegrationStatus::Error => ComponentHealth::Error,
                    _ => ComponentHealth::Unknown,
                }),
            ]),
            last_check: SystemTime::now(),
        }
    }
}

/// System health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Overall system health
    pub overall: ComponentHealth,
    /// Individual component health
    pub components: std::collections::HashMap<String, ComponentHealth>,
    /// Last health check time
    pub last_check: SystemTime,
}

/// Component health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentHealth {
    /// Component is healthy
    Healthy,
    /// Component has warnings
    Warning,
    /// Component has errors
    Error,
    /// Component status unknown
    Unknown,
}

// Default implementations

impl Default for AutomationStatistics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            total_tests_executed: 0,
            total_test_failures: 0,
            total_reports_generated: 0,
            total_artifacts_stored: 0,
            total_gate_failures: 0,
            total_integrations_triggered: 0,
            average_execution_duration: Duration::ZERO,
            last_execution_time: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_creation() {
        let config = CiCdAutomationConfig::default();
        let automation = CiCdAutomation::new(config);
        assert!(automation.is_ok());
    }

    #[test]
    fn test_statistics_default() {
        let stats = AutomationStatistics::default();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.successful_executions, 0);
        assert_eq!(stats.failed_executions, 0);
    }

    #[test]
    fn test_component_health() {
        assert_eq!(ComponentHealth::Healthy, ComponentHealth::Healthy);
        assert_ne!(ComponentHealth::Healthy, ComponentHealth::Warning);
    }

    #[test]
    fn test_automation_step_result() {
        let result = AutomationStepResult {
            step_name: "test_step".to_string(),
            success: true,
            duration: Duration::from_secs(1),
            message: "Success".to_string(),
            metadata: std::collections::HashMap::new(),
        };

        assert!(result.success);
        assert_eq!(result.step_name, "test_step");
    }

    #[test]
    fn test_environment_detection() {
        let env = CiCdAutomation::detect_environment();
        assert!(env.is_ok());

        let env_info = env.unwrap();
        assert!(!env_info.os.is_empty());
        assert!(!env_info.arch.is_empty());
        assert!(env_info.cpu_count > 0);
    }
}