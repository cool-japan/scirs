//! # CI/CD Automation System
//!
//! This module provides comprehensive CI/CD automation for performance testing,
//! benchmarking, and regression detection in the SciRS2 ecosystem.
//!
//! ## Architecture
//!
//! The CI/CD automation system is organized into 7 specialized modules:
//!
//! - **config**: Configuration management and platform settings
//! - **test_execution**: Test suite management and execution logic
//! - **reporting**: Report generation, templates, and formatting
//! - **artifact_management**: Storage providers and artifact handling
//! - **integrations**: External service integrations (GitHub, Slack, Email, Webhooks)
//! - **performance_gates**: Performance monitoring and gate evaluation
//! - **core_automation**: Main automation engine and orchestration
//!
//! ## Key Features
//!
//! - Multi-platform CI/CD support (GitHub Actions, GitLab CI, Jenkins, etc.)
//! - Performance regression detection with statistical analysis
//! - Comprehensive reporting with multiple output formats
//! - Artifact storage with multiple cloud providers
//! - Real-time integration with external services
//! - Performance gates with trend analysis and alerting
//! - Automated baseline management and historical tracking
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_optim::benchmarking::ci_cd_automation::{
//!     CiCdAutomation, CiCdAutomationConfig, CiCdPlatform
//! };
//!
//! // Create automation configuration
//! let config = CiCdAutomationConfig {
//!     enable_automation: true,
//!     platform: CiCdPlatform::GitHubActions,
//!     // ... other configuration
//! };
//!
//! // Initialize automation system
//! let automation = CiCdAutomation::new(config)?;
//!
//! // Run performance tests with CI/CD integration
//! let results = automation.run_automated_tests().await?;
//! ```

pub mod config;
pub mod test_execution;
pub mod reporting;
pub mod artifact_management;
pub mod integrations;
pub mod performance_gates;
pub mod core_automation;

// Re-export all public types and functions

// Configuration types and enums
pub use config::{
    CiCdAutomationConfig, CiCdPlatform, TestExecutionConfig, BaselineManagementConfig,
    ReportingConfig, ArtifactStorageConfig, IntegrationConfig, PerformanceGatesConfig,
    PlatformSpecificConfig, GitHubActionsConfig, GitLabCiConfig, JenkinsConfig,
    TeamCityConfig, CircleCiConfig, TravisCiConfig, AppVeyorConfig, AzureDevOpsConfig,
    TestEnvironmentConfig, TestSelectionConfig, ParallelismConfig, TimeoutConfig,
    RetryConfig, BaselineStrategy, BaselineUpdatePolicy, BaselineValidationConfig,
    ReportFormat, ReportTemplate, NotificationConfig, DistributionConfig,
    StorageProvider, S3Config, GcsConfig, AzureBlobConfig, FtpConfig, HttpConfig,
    CacheConfig, CompressionConfig, EncryptionConfig, GitHubIntegrationConfig,
    SlackIntegrationConfig, EmailIntegrationConfig, WebhookIntegrationConfig,
    GateEvaluationMode, RegressionThreshold, TrendAnalysisConfig, AlertConfig,
    EscalationConfig, validate_config, default_config, platform_specific_defaults,
};

// Test execution types and functionality
pub use test_execution::{
    PerformanceTestSuite, PerformanceTestCase, TestSuiteConfig, CiCdContext,
    CiCdTestResult, TestExecutor, TestCaseMetadata, TestDependency, TestResource,
    ResourceType, TestEnvironment, TestConfiguration, TestExecutionStrategy,
    ParallelTestExecutor, SequentialTestExecutor, TestResultCollector,
    TestMetricsCollector, TestFailureAnalyzer, TestRetryManager, TestTimeoutManager,
    TestResourceManager, ResourceAllocation, ResourceConstraints, DynamicResourceManager,
    TestOrchestrator, ExecutionPlan, TestBatch, BatchingStrategy, TestPrioritizer,
    PriorityLevel, TestFilterManager, FilterCriteria, ConditionalExecution,
    create_test_suite, configure_test_environment, execute_test_batch,
    collect_test_metrics, analyze_test_failures, manage_test_resources,
};

// Reporting types and functionality
pub use reporting::{
    ReportGenerator, TemplateEngine, GeneratedReport, ReportContent, ReportSection,
    ReportMetadata, ReportAsset, HtmlReportGenerator, JsonReportGenerator,
    MarkdownReportGenerator, PdfReportGenerator, JunitXmlReportGenerator,
    CustomReportGenerator, ReportTemplate as ReportTemplateStruct, TemplateVariable,
    TemplateContext, ReportFormatter, FormattingOptions, StyleConfiguration,
    ReportDistributor, DistributionChannel, NotificationManager, ReportNotification,
    NotificationTemplate, ReportArchiver, ArchivePolicy, ReportCompressor,
    CompressionSettings, ReportValidator, ValidationRule, ReportAnalyzer,
    AnalysisMetrics, TrendReporter, HistoricalComparison, generate_html_report,
    generate_json_report, generate_markdown_report, generate_pdf_report,
    generate_junit_xml_report, create_report_template, format_report_content,
    distribute_report, archive_report, validate_report_content,
};

// Artifact management types and functionality
pub use artifact_management::{
    ArtifactManager, ArtifactStorage, ArtifactRegistry, UploadManager, DownloadManager,
    RetentionManager, LocalStorage, S3Storage, GcsStorage, AzureBlobStorage,
    FtpStorage, HttpStorage, ArtifactMetadata, ArtifactInfo, ArtifactType,
    StorageLocation, UploadProgress, DownloadProgress, TransferStatistics,
    ArtifactCompressor, CompressionAlgorithm, ArtifactEncryptor, EncryptionKey,
    ArtifactValidator, ValidationResult, ArtifactIndexer, SearchCriteria,
    ArtifactCleaner, CleanupPolicy, RetentionPolicy, ArtifactMigrator,
    MigrationPlan, SyncManager, SyncPolicy, ArtifactMonitor, MonitoringMetrics,
    create_artifact_manager, upload_artifact, download_artifact, search_artifacts,
    cleanup_artifacts, migrate_artifacts, sync_artifacts, monitor_storage,
};

// Integration types and functionality
pub use integrations::{
    IntegrationManager, GitHubClient, SlackClient, EmailClient, WebhookClient,
    CustomIntegration, IntegrationStatistics, GitHubIntegration, SlackIntegration,
    EmailIntegration, WebhookIntegration, IntegrationEvent, EventPayload,
    NotificationPayload, MessageTemplate, MessageFormatter, DeliveryStatus,
    RetryPolicy as IntegrationRetryPolicy, RateLimiter, ApiCredentials,
    AuthenticationMethod, IntegrationHealth, HealthChecker, FailureRecovery,
    IntegrationMetrics, UsageStatistics, IntegrationLogger, LogLevel,
    IntegrationValidator, ValidationError, IntegrationTester, TestResult,
    create_integration_manager, setup_github_integration, setup_slack_integration,
    setup_email_integration, setup_webhook_integration, send_notification,
    check_integration_health, validate_integration_config, test_integration,
};

// Performance gates types and functionality
pub use performance_gates::{
    PerformanceGateEvaluator, BaselineMetric, GateEvaluationResult, GateState,
    PerformanceTrendAnalyzer, AlertManager, MetricType, GateConfiguration,
    ThresholdType, StatisticalMethod, TrendAnalysis, TrendDirection,
    AlertCondition, AlertSeverity, EscalationLevel, NotificationChannel,
    GatePolicy, PolicyRule, RuleAction, MetricCollector, MetricProcessor,
    MetricAggregator, StatisticalAnalyzer, RegressionDetector, AnomalyDetector,
    BaselineManager, BaselineCalculator, HistoricalDataManager, GateReporter,
    GateMetrics, GateStatistics, evaluate_performance_gates, analyze_trends,
    detect_regressions, detect_anomalies, update_baselines, generate_alerts,
    escalate_alerts, collect_gate_metrics, report_gate_status,
};

// Core automation types and functionality
pub use core_automation::{
    CiCdAutomation, PerformanceRegressionDetector, EnvironmentInfo, AutomationStatistics,
    AutomationExecutionContext, RegressionAnalysis, StatisticalTest, ConfidenceLevel,
    RegressionSeverity, AnalysisMethod, ComparisonStrategy, MetricComparison,
    RegressionReport, TrendAnalysisResult, PerformanceBaseline, AutomationEvent,
    ExecutionPhase, PhaseResult, AutomationMetrics, ExecutionSummary,
    PerformanceInsights, RecommendationEngine, ActionableInsight, InsightType,
    AutomationOrchestrator, WorkflowDefinition, WorkflowStep, StepDependency,
    AutomationScheduler, SchedulePolicy, TriggerCondition, AutomationLogger,
    LogEntry, LogContext, create_automation_engine, detect_performance_regression,
    analyze_performance_trends, generate_performance_insights, orchestrate_workflow,
    schedule_automation, log_automation_event,
};

// Convenience type aliases
pub type CiCdResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type AutomationResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Constants
pub const DEFAULT_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3600); // 1 hour
pub const DEFAULT_RETRY_ATTEMPTS: usize = 3;
pub const DEFAULT_PARALLEL_JOBS: usize = 4;
pub const DEFAULT_COMPRESSION_LEVEL: u8 = 6;
pub const DEFAULT_RETENTION_DAYS: u32 = 30;
pub const MAX_ARTIFACT_SIZE: u64 = 1024 * 1024 * 1024; // 1 GB
pub const MAX_REPORT_SIZE: usize = 100 * 1024 * 1024; // 100 MB
pub const DEFAULT_BASELINE_WINDOW: usize = 10;
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.05; // 5%

// Error types
pub use config::ConfigError;
pub use test_execution::TestExecutionError;
pub use reporting::ReportingError;
pub use artifact_management::ArtifactError;
pub use integrations::IntegrationError;
pub use performance_gates::GateError;
pub use core_automation::AutomationError;

// Utility functions
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn build_info() -> String {
    format!(
        "SciRS2 CI/CD Automation v{} ({})",
        version(),
        env!("CARGO_PKG_REPOSITORY")
    )
}