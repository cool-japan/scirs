//! CI/CD Automation Configuration Management
//!
//! This module provides comprehensive configuration management for CI/CD automation,
//! including platform settings, test execution parameters, baseline management,
//! reporting configuration, artifact storage, and integration settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// CI/CD automation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdAutomationConfig {
    /// Enable automated performance testing
    pub enable_automation: bool,
    /// CI/CD platform type
    pub platform: CiCdPlatform,
    /// Test execution configuration
    pub test_execution: TestExecutionConfig,
    /// Baseline management settings
    pub baseline_management: BaselineManagementConfig,
    /// Report generation settings
    pub reporting: ReportingConfig,
    /// Artifact storage settings
    pub artifact_storage: ArtifactStorageConfig,
    /// Integration settings
    pub integrations: IntegrationConfig,
    /// Performance gates configuration
    pub performance_gates: PerformanceGatesConfig,
}

/// Supported CI/CD platforms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CiCdPlatform {
    /// GitHub Actions workflow automation
    GitHubActions,
    /// GitLab CI/CD pipelines
    GitLabCI,
    /// Jenkins automation server
    Jenkins,
    /// Azure DevOps Services
    AzureDevOps,
    /// CircleCI continuous integration
    CircleCI,
    /// Travis CI (legacy support)
    TravisCI,
    /// JetBrains TeamCity
    TeamCity,
    /// Buildkite CI/CD platform
    Buildkite,
    /// Generic/custom CI/CD platform
    Generic,
}

/// Test execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionConfig {
    /// Run tests on every commit
    pub run_on_commit: bool,
    /// Run tests on pull requests
    pub run_on_pr: bool,
    /// Run tests on releases
    pub run_on_release: bool,
    /// Run tests on schedule
    pub run_on_schedule: Option<CronSchedule>,
    /// Test timeout in seconds
    pub test_timeout: u64,
    /// Number of test iterations
    pub test_iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Parallel test execution
    pub parallel_execution: bool,
    /// Test isolation level
    pub isolation_level: TestIsolationLevel,
    /// Maximum number of concurrent tests
    pub max_concurrent_tests: usize,
    /// Resource limits for test execution
    pub resource_limits: ResourceLimits,
    /// Test environment variables
    pub environment_variables: HashMap<String, String>,
    /// Custom test command overrides
    pub custom_commands: HashMap<String, String>,
}

/// Test isolation levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestIsolationLevel {
    /// Run tests in same process
    Process,
    /// Run each test in separate process
    ProcessIsolated,
    /// Run tests in Docker containers
    Container,
    /// Run tests in virtual machines
    VirtualMachine,
}

/// Resource limits for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: Option<f64>,
    /// Maximum execution time per test in seconds
    pub max_execution_time_sec: Option<u64>,
    /// Maximum disk usage in MB
    pub max_disk_mb: Option<usize>,
    /// Maximum network bandwidth in MB/s
    pub max_network_mbps: Option<f64>,
}

/// Cron schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronSchedule {
    /// Cron expression (e.g., "0 0 * * *" for daily at midnight)
    pub expression: String,
    /// Timezone for schedule execution
    pub timezone: String,
    /// Enable/disable scheduled execution
    pub enabled: bool,
}

/// Baseline management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineManagementConfig {
    /// Automatically update baseline on main branch
    pub auto_update_main: bool,
    /// Update baseline on release tags
    pub update_on_release: bool,
    /// Minimum improvement threshold for baseline updates
    pub min_improvement_threshold: f64,
    /// Maximum degradation allowed before failing
    pub max_degradation_threshold: f64,
    /// Baseline storage configuration
    pub storage: BaselineStorageConfig,
    /// Retention policy for baselines
    pub retention: BaselineRetentionPolicy,
    /// Baseline validation settings
    pub validation: BaselineValidationConfig,
}

/// Baseline storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStorageConfig {
    /// Storage provider type
    pub provider: BaselineStorageProvider,
    /// Storage location/path
    pub location: String,
    /// Encryption settings
    pub encryption: Option<EncryptionConfig>,
    /// Compression settings
    pub compression: Option<CompressionConfig>,
}

/// Baseline storage providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BaselineStorageProvider {
    /// Local filesystem storage
    Local,
    /// AWS S3 storage
    S3,
    /// Google Cloud Storage
    GCS,
    /// Azure Blob Storage
    AzureBlob,
    /// Git repository storage
    Git,
    /// Database storage
    Database,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key management configuration
    pub key_management: KeyManagementConfig,
    /// Enable encryption at rest
    pub encrypt_at_rest: bool,
    /// Enable encryption in transit
    pub encrypt_in_transit: bool,
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-256 encryption
    AES256,
    /// ChaCha20-Poly1305 encryption
    ChaCha20Poly1305,
    /// RSA encryption
    RSA,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key provider
    pub provider: KeyProvider,
    /// Key rotation interval in days
    pub rotation_interval_days: Option<u32>,
    /// Key derivation settings
    pub derivation: Option<KeyDerivationConfig>,
}

/// Key providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyProvider {
    /// Environment variables
    Environment,
    /// AWS KMS
    AWSKMS,
    /// Azure Key Vault
    AzureKeyVault,
    /// Google Cloud KMS
    GoogleCloudKMS,
    /// HashiCorp Vault
    Vault,
}

/// Key derivation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    /// Derivation function
    pub function: KeyDerivationFunction,
    /// Salt length in bytes
    pub salt_length: usize,
    /// Number of iterations
    pub iterations: u32,
}

/// Key derivation functions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyDerivationFunction {
    /// PBKDF2 with SHA-256
    PBKDF2SHA256,
    /// Argon2id
    Argon2id,
    /// scrypt
    Scrypt,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Enable compression for baselines
    pub compress_baselines: bool,
    /// Enable compression for reports
    pub compress_reports: bool,
    /// Minimum file size for compression (bytes)
    pub min_size_bytes: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// Gzip compression
    Gzip,
    /// Zstandard compression
    Zstd,
    /// LZ4 compression
    LZ4,
    /// Brotli compression
    Brotli,
}

/// Baseline retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRetentionPolicy {
    /// Keep baselines for N days
    pub retention_days: u32,
    /// Maximum number of baselines to keep
    pub max_baselines: usize,
    /// Keep all release baselines
    pub keep_release_baselines: bool,
    /// Keep milestone baselines
    pub keep_milestone_baselines: bool,
    /// Cleanup frequency in hours
    pub cleanup_frequency_hours: u32,
}

/// Baseline validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineValidationConfig {
    /// Enable baseline integrity checks
    pub enable_integrity_checks: bool,
    /// Enable statistical validation
    pub enable_statistical_validation: bool,
    /// Minimum sample size for validation
    pub min_sample_size: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Validation timeout in seconds
    pub validation_timeout_sec: u64,
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Generate HTML reports
    pub generate_html: bool,
    /// Generate JSON reports
    pub generate_json: bool,
    /// Generate JUnit XML reports
    pub generate_junit: bool,
    /// Generate Markdown reports
    pub generate_markdown: bool,
    /// Generate PDF reports
    pub generate_pdf: bool,
    /// Include detailed metrics
    pub include_detailed_metrics: bool,
    /// Include performance graphs
    pub include_graphs: bool,
    /// Include regression analysis
    pub include_regression_analysis: bool,
    /// Report templates configuration
    pub templates: ReportTemplateConfig,
    /// Report styling configuration
    pub styling: ReportStylingConfig,
    /// Report distribution configuration
    pub distribution: ReportDistributionConfig,
}

/// Report template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplateConfig {
    /// Custom HTML template path
    pub html_template_path: Option<PathBuf>,
    /// Custom Markdown template path
    pub markdown_template_path: Option<PathBuf>,
    /// Template variables
    pub template_variables: HashMap<String, String>,
    /// Enable template caching
    pub enable_caching: bool,
}

/// Report styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportStylingConfig {
    /// CSS style sheet path
    pub css_path: Option<PathBuf>,
    /// Color theme
    pub color_theme: ColorTheme,
    /// Font family
    pub font_family: String,
    /// Chart styling
    pub chart_style: ChartStyleConfig,
}

/// Color themes for reports
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColorTheme {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// High contrast theme
    HighContrast,
    /// Custom theme
    Custom,
}

/// Chart styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStyleConfig {
    /// Chart width in pixels
    pub width: u32,
    /// Chart height in pixels
    pub height: u32,
    /// Enable animations
    pub enable_animations: bool,
    /// Color palette
    pub color_palette: Vec<String>,
    /// Grid style
    pub grid_style: GridStyle,
}

/// Grid styles for charts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GridStyle {
    /// Solid grid lines
    Solid,
    /// Dashed grid lines
    Dashed,
    /// Dotted grid lines
    Dotted,
    /// No grid lines
    None,
}

/// Report distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistributionConfig {
    /// Email distribution settings
    pub email: Option<EmailDistributionConfig>,
    /// Slack distribution settings
    pub slack: Option<SlackDistributionConfig>,
    /// Webhook distribution settings
    pub webhooks: Vec<WebhookDistributionConfig>,
    /// File system distribution
    pub filesystem: Option<FilesystemDistributionConfig>,
}

/// Email distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailDistributionConfig {
    /// Recipients list
    pub recipients: Vec<String>,
    /// Subject template
    pub subject_template: String,
    /// Email body template
    pub body_template: String,
    /// Attach reports as files
    pub attach_reports: bool,
    /// Email priority
    pub priority: EmailPriority,
}

/// Email priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmailPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Urgent priority
    Urgent,
}

/// Slack distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackDistributionConfig {
    /// Slack channels
    pub channels: Vec<String>,
    /// Message template
    pub message_template: String,
    /// Include report attachments
    pub include_attachments: bool,
    /// Mention users on failures
    pub mention_on_failure: Vec<String>,
}

/// Webhook distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDistributionConfig {
    /// Webhook URL
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Headers
    pub headers: HashMap<String, String>,
    /// Payload template
    pub payload_template: String,
    /// Authentication
    pub auth: Option<WebhookAuth>,
    /// Timeout in seconds
    pub timeout_sec: u64,
    /// Retry configuration
    pub retry: WebhookRetryConfig,
}

/// HTTP methods for webhooks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    PATCH,
    DELETE,
}

/// Webhook authentication methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WebhookAuth {
    /// Bearer token authentication
    Bearer { token: String },
    /// Basic authentication
    Basic { username: String, password: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// Custom header authentication
    Custom { headers: HashMap<String, String> },
}

/// Webhook retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookRetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial delay in seconds
    pub initial_delay_sec: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay in seconds
    pub max_delay_sec: u64,
}

/// Filesystem distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemDistributionConfig {
    /// Output directory
    pub output_directory: PathBuf,
    /// File naming pattern
    pub file_naming_pattern: String,
    /// Create subdirectories by date
    pub create_date_subdirs: bool,
    /// File permissions (Unix-style)
    pub file_permissions: Option<u32>,
}

/// Artifact storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageConfig {
    /// Enable artifact storage
    pub enabled: bool,
    /// Storage provider
    pub provider: ArtifactStorageProvider,
    /// Provider-specific configuration
    pub storage_config: HashMap<String, String>,
    /// Retention policy
    pub retention: ArtifactRetentionPolicy,
    /// Upload configuration
    pub upload: ArtifactUploadConfig,
    /// Download configuration
    pub download: ArtifactDownloadConfig,
}

/// Artifact storage providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactStorageProvider {
    /// Local filesystem storage
    Local(PathBuf),
    /// AWS S3 storage
    S3 {
        bucket: String,
        region: String,
        prefix: Option<String>,
    },
    /// Google Cloud Storage
    GCS {
        bucket: String,
        prefix: Option<String>,
    },
    /// Azure Blob Storage
    AzureBlob {
        account: String,
        container: String,
        prefix: Option<String>,
    },
    /// FTP/SFTP storage
    FTP {
        host: String,
        port: u16,
        path: String,
        secure: bool,
    },
    /// HTTP/HTTPS storage
    HTTP {
        base_url: String,
        auth: Option<WebhookAuth>,
    },
}

/// Artifact retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    /// Maximum number of artifacts to keep
    pub max_artifacts: Option<usize>,
    /// Keep artifacts for releases
    pub keep_release_artifacts: bool,
    /// Keep artifacts for failed builds
    pub keep_failed_build_artifacts: bool,
    /// Cleanup schedule
    pub cleanup_schedule: Option<CronSchedule>,
}

/// Artifact upload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactUploadConfig {
    /// Enable compression during upload
    pub compress: bool,
    /// Compression level (1-9)
    pub compression_level: u8,
    /// Enable encryption during upload
    pub encrypt: bool,
    /// Upload timeout in seconds
    pub timeout_sec: u64,
    /// Maximum file size in MB
    pub max_file_size_mb: usize,
    /// Parallel upload settings
    pub parallel_uploads: ParallelUploadConfig,
}

/// Parallel upload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelUploadConfig {
    /// Enable parallel uploads
    pub enabled: bool,
    /// Maximum concurrent uploads
    pub max_concurrent: usize,
    /// Chunk size for multipart uploads (MB)
    pub chunk_size_mb: usize,
}

/// Artifact download configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDownloadConfig {
    /// Download timeout in seconds
    pub timeout_sec: u64,
    /// Enable download caching
    pub enable_caching: bool,
    /// Cache directory
    pub cache_directory: Option<PathBuf>,
    /// Cache size limit in MB
    pub cache_size_limit_mb: Option<usize>,
    /// Verify checksums on download
    pub verify_checksums: bool,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// GitHub integration settings
    pub github: Option<GitHubIntegration>,
    /// Slack integration settings
    pub slack: Option<SlackIntegration>,
    /// Email integration settings
    pub email: Option<EmailIntegration>,
    /// Webhook integrations
    pub webhooks: Vec<WebhookIntegration>,
    /// Custom integrations
    pub custom: HashMap<String, CustomIntegrationConfig>,
}

/// GitHub integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubIntegration {
    /// GitHub token for API access
    pub token: String,
    /// Repository owner
    pub owner: String,
    /// Repository name
    pub repository: String,
    /// Create status checks
    pub create_status_checks: bool,
    /// Create comments on PRs
    pub create_pr_comments: bool,
    /// Create issues for regressions
    pub create_regression_issues: bool,
    /// Label configuration
    pub labels: GitHubLabelConfig,
    /// Status check configuration
    pub status_checks: GitHubStatusCheckConfig,
}

/// GitHub label configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubLabelConfig {
    /// Performance regression label
    pub performance_regression: String,
    /// Performance improvement label
    pub performance_improvement: String,
    /// Test failure label
    pub test_failure: String,
    /// Automated label
    pub automated: String,
}

/// GitHub status check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubStatusCheckConfig {
    /// Status check context name
    pub context: String,
    /// Success description
    pub success_description: String,
    /// Failure description
    pub failure_description: String,
    /// Pending description
    pub pending_description: String,
}

/// Slack integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackIntegration {
    /// Slack webhook URL
    pub webhook_url: String,
    /// Default channel
    pub default_channel: String,
    /// Bot username
    pub username: Option<String>,
    /// Bot icon emoji
    pub icon_emoji: Option<String>,
    /// Notification settings
    pub notifications: SlackNotificationConfig,
}

/// Slack notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackNotificationConfig {
    /// Notify on test completion
    pub notify_on_completion: bool,
    /// Notify on regression detection
    pub notify_on_regression: bool,
    /// Notify on test failures
    pub notify_on_failure: bool,
    /// Notify on performance improvements
    pub notify_on_improvement: bool,
    /// Mention users for critical issues
    pub mention_users: Vec<String>,
}

/// Email integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIntegration {
    /// SMTP server configuration
    pub smtp: SmtpConfig,
    /// Default sender email
    pub from_email: String,
    /// Default recipient emails
    pub default_recipients: Vec<String>,
    /// Email templates
    pub templates: EmailTemplateConfig,
}

/// SMTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    /// SMTP server host
    pub host: String,
    /// SMTP server port
    pub port: u16,
    /// Use TLS encryption
    pub use_tls: bool,
    /// Username for authentication
    pub username: Option<String>,
    /// Password for authentication
    pub password: Option<String>,
    /// Connection timeout in seconds
    pub timeout_sec: u64,
}

/// Email template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailTemplateConfig {
    /// Subject template for success
    pub success_subject: String,
    /// Subject template for failure
    pub failure_subject: String,
    /// Body template for success
    pub success_body: String,
    /// Body template for failure
    pub failure_body: String,
    /// Include HTML formatting
    pub use_html: bool,
}

/// Webhook integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookIntegration {
    /// Webhook name/identifier
    pub name: String,
    /// Webhook URL
    pub url: String,
    /// HTTP method to use
    pub method: HttpMethod,
    /// HTTP headers
    pub headers: HashMap<String, String>,
    /// Authentication method
    pub auth: Option<WebhookAuth>,
    /// Trigger conditions
    pub triggers: WebhookTriggerConfig,
    /// Payload configuration
    pub payload: WebhookPayloadConfig,
}

/// Webhook trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookTriggerConfig {
    /// Trigger on test completion
    pub on_completion: bool,
    /// Trigger on regression detection
    pub on_regression: bool,
    /// Trigger on test failure
    pub on_failure: bool,
    /// Trigger on performance improvement
    pub on_improvement: bool,
    /// Custom trigger conditions
    pub custom_conditions: Vec<String>,
}

/// Webhook payload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayloadConfig {
    /// Payload format
    pub format: PayloadFormat,
    /// Include test results
    pub include_results: bool,
    /// Include metrics
    pub include_metrics: bool,
    /// Include environment info
    pub include_environment: bool,
    /// Custom payload template
    pub custom_template: Option<String>,
}

/// Payload formats for webhooks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PayloadFormat {
    /// JSON format
    JSON,
    /// XML format
    XML,
    /// Form data format
    FormData,
    /// Custom format
    Custom,
}

/// Custom integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIntegrationConfig {
    /// Integration type
    pub integration_type: String,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Enable/disable the integration
    pub enabled: bool,
}

/// Performance gates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGatesConfig {
    /// Enable performance gates
    pub enabled: bool,
    /// Metric gates configuration
    pub metric_gates: HashMap<MetricType, MetricGate>,
    /// Gate evaluation strategy
    pub evaluation_strategy: GateEvaluationStrategy,
    /// Failure handling configuration
    pub failure_handling: GateFailureHandling,
}

/// Metric types for performance gates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Execution time metric
    ExecutionTime,
    /// Memory usage metric
    MemoryUsage,
    /// CPU usage metric
    CpuUsage,
    /// Throughput metric
    Throughput,
    /// Latency metric
    Latency,
    /// Error rate metric
    ErrorRate,
    /// Custom metric
    Custom(String),
}

/// Metric gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricGate {
    /// Gate type
    pub gate_type: GateType,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Gate severity
    pub severity: GateSeverity,
    /// Enable/disable the gate
    pub enabled: bool,
}

/// Gate types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateType {
    /// Absolute threshold gate
    Absolute,
    /// Relative threshold gate (percentage change)
    Relative,
    /// Statistical threshold gate
    Statistical,
    /// Trend-based gate
    Trend,
}

/// Comparison operators for gates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOperator {
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Equal
    Equal,
    /// Not equal
    NotEqual,
}

/// Gate severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum GateSeverity {
    /// Informational (does not fail build)
    Info,
    /// Warning (logs warning but does not fail build)
    Warning,
    /// Error (fails build)
    Error,
    /// Critical (fails build and sends alerts)
    Critical,
}

/// Gate evaluation strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateEvaluationStrategy {
    /// Fail fast - stop on first gate failure
    FailFast,
    /// Evaluate all gates before failing
    EvaluateAll,
    /// Weighted evaluation based on severity
    Weighted,
}

/// Gate failure handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFailureHandling {
    /// Action to take on gate failure
    pub failure_action: GateFailureAction,
    /// Allow manual override of gate failures
    pub allow_manual_override: bool,
    /// Override timeout in hours
    pub override_timeout_hours: Option<u32>,
    /// Notification settings for failures
    pub notifications: GateFailureNotificationConfig,
}

/// Gate failure actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateFailureAction {
    /// Fail the build
    FailBuild,
    /// Mark build as unstable
    MarkUnstable,
    /// Log warning and continue
    LogWarning,
    /// Send notification only
    NotifyOnly,
}

/// Gate failure notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFailureNotificationConfig {
    /// Send email notifications
    pub send_email: bool,
    /// Send Slack notifications
    pub send_slack: bool,
    /// Send webhook notifications
    pub send_webhooks: bool,
    /// Create GitHub issues
    pub create_github_issues: bool,
    /// Escalation settings
    pub escalation: Option<NotificationEscalationConfig>,
}

/// Notification escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationEscalationConfig {
    /// Escalation delay in minutes
    pub delay_minutes: u32,
    /// Maximum escalation levels
    pub max_levels: u32,
    /// Escalation recipients by level
    pub recipients_by_level: HashMap<u32, Vec<String>>,
}

// Default implementations

impl Default for CiCdAutomationConfig {
    fn default() -> Self {
        Self {
            enable_automation: true,
            platform: CiCdPlatform::Generic,
            test_execution: TestExecutionConfig::default(),
            baseline_management: BaselineManagementConfig::default(),
            reporting: ReportingConfig::default(),
            artifact_storage: ArtifactStorageConfig::default(),
            integrations: IntegrationConfig::default(),
            performance_gates: PerformanceGatesConfig::default(),
        }
    }
}

impl Default for TestExecutionConfig {
    fn default() -> Self {
        Self {
            run_on_commit: true,
            run_on_pr: true,
            run_on_release: true,
            run_on_schedule: None,
            test_timeout: 3600, // 1 hour
            test_iterations: 5,
            warmup_iterations: 2,
            parallel_execution: true,
            isolation_level: TestIsolationLevel::Process,
            max_concurrent_tests: num_cpus::get(),
            resource_limits: ResourceLimits::default(),
            environment_variables: HashMap::new(),
            custom_commands: HashMap::new(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(4096),    // 4GB
            max_cpu_percent: Some(80.0),  // 80%
            max_execution_time_sec: Some(1800), // 30 minutes
            max_disk_mb: Some(10240),     // 10GB
            max_network_mbps: Some(100.0), // 100 MB/s
        }
    }
}

impl Default for BaselineManagementConfig {
    fn default() -> Self {
        Self {
            auto_update_main: true,
            update_on_release: true,
            min_improvement_threshold: 0.05, // 5%
            max_degradation_threshold: 0.10, // 10%
            storage: BaselineStorageConfig::default(),
            retention: BaselineRetentionPolicy::default(),
            validation: BaselineValidationConfig::default(),
        }
    }
}

impl Default for BaselineStorageConfig {
    fn default() -> Self {
        Self {
            provider: BaselineStorageProvider::Local,
            location: "./baselines".to_string(),
            encryption: None,
            compression: None,
        }
    }
}

impl Default for BaselineRetentionPolicy {
    fn default() -> Self {
        Self {
            retention_days: 90,
            max_baselines: 100,
            keep_release_baselines: true,
            keep_milestone_baselines: true,
            cleanup_frequency_hours: 24,
        }
    }
}

impl Default for BaselineValidationConfig {
    fn default() -> Self {
        Self {
            enable_integrity_checks: true,
            enable_statistical_validation: true,
            min_sample_size: 10,
            confidence_level: 0.95,
            validation_timeout_sec: 300,
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            generate_html: true,
            generate_json: true,
            generate_junit: false,
            generate_markdown: false,
            generate_pdf: false,
            include_detailed_metrics: true,
            include_graphs: true,
            include_regression_analysis: true,
            templates: ReportTemplateConfig::default(),
            styling: ReportStylingConfig::default(),
            distribution: ReportDistributionConfig::default(),
        }
    }
}

impl Default for ReportTemplateConfig {
    fn default() -> Self {
        Self {
            html_template_path: None,
            markdown_template_path: None,
            template_variables: HashMap::new(),
            enable_caching: true,
        }
    }
}

impl Default for ReportStylingConfig {
    fn default() -> Self {
        Self {
            css_path: None,
            color_theme: ColorTheme::Light,
            font_family: "Arial, sans-serif".to_string(),
            chart_style: ChartStyleConfig::default(),
        }
    }
}

impl Default for ChartStyleConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            enable_animations: true,
            color_palette: vec![
                "#1f77b4".to_string(),
                "#ff7f0e".to_string(),
                "#2ca02c".to_string(),
                "#d62728".to_string(),
                "#9467bd".to_string(),
            ],
            grid_style: GridStyle::Solid,
        }
    }
}

impl Default for ReportDistributionConfig {
    fn default() -> Self {
        Self {
            email: None,
            slack: None,
            webhooks: Vec::new(),
            filesystem: Some(FilesystemDistributionConfig::default()),
        }
    }
}

impl Default for FilesystemDistributionConfig {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("./reports"),
            file_naming_pattern: "{timestamp}_{test_name}_{format}".to_string(),
            create_date_subdirs: true,
            file_permissions: Some(0o644),
        }
    }
}

impl Default for ArtifactStorageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: ArtifactStorageProvider::Local(PathBuf::from("./artifacts")),
            storage_config: HashMap::new(),
            retention: ArtifactRetentionPolicy::default(),
            upload: ArtifactUploadConfig::default(),
            download: ArtifactDownloadConfig::default(),
        }
    }
}

impl Default for ArtifactRetentionPolicy {
    fn default() -> Self {
        Self {
            retention_days: 30,
            max_artifacts: Some(1000),
            keep_release_artifacts: true,
            keep_failed_build_artifacts: false,
            cleanup_schedule: None,
        }
    }
}

impl Default for ArtifactUploadConfig {
    fn default() -> Self {
        Self {
            compress: true,
            compression_level: 6,
            encrypt: false,
            timeout_sec: 300,
            max_file_size_mb: 1024, // 1GB
            parallel_uploads: ParallelUploadConfig::default(),
        }
    }
}

impl Default for ParallelUploadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent: 4,
            chunk_size_mb: 100, // 100MB chunks
        }
    }
}

impl Default for ArtifactDownloadConfig {
    fn default() -> Self {
        Self {
            timeout_sec: 300,
            enable_caching: true,
            cache_directory: Some(PathBuf::from("./cache")),
            cache_size_limit_mb: Some(5120), // 5GB
            verify_checksums: true,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            github: None,
            slack: None,
            email: None,
            webhooks: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

impl Default for PerformanceGatesConfig {
    fn default() -> Self {
        let mut metric_gates = HashMap::new();

        // Default execution time gate: fail if > 20% slower
        metric_gates.insert(
            MetricType::ExecutionTime,
            MetricGate {
                gate_type: GateType::Relative,
                threshold: 0.20, // 20% increase
                operator: ComparisonOperator::LessThanOrEqual,
                severity: GateSeverity::Error,
                enabled: true,
            }
        );

        // Default memory usage gate: fail if > 30% more memory
        metric_gates.insert(
            MetricType::MemoryUsage,
            MetricGate {
                gate_type: GateType::Relative,
                threshold: 0.30, // 30% increase
                operator: ComparisonOperator::LessThanOrEqual,
                severity: GateSeverity::Warning,
                enabled: true,
            }
        );

        Self {
            enabled: true,
            metric_gates,
            evaluation_strategy: GateEvaluationStrategy::EvaluateAll,
            failure_handling: GateFailureHandling::default(),
        }
    }
}

impl Default for GateFailureHandling {
    fn default() -> Self {
        Self {
            failure_action: GateFailureAction::FailBuild,
            allow_manual_override: true,
            override_timeout_hours: Some(24),
            notifications: GateFailureNotificationConfig::default(),
        }
    }
}

impl Default for GateFailureNotificationConfig {
    fn default() -> Self {
        Self {
            send_email: false,
            send_slack: false,
            send_webhooks: false,
            create_github_issues: false,
            escalation: None,
        }
    }
}

// Configuration validation and utilities

impl CiCdAutomationConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate test execution config
        self.test_execution.validate()?;

        // Validate baseline management config
        self.baseline_management.validate()?;

        // Validate reporting config
        self.reporting.validate()?;

        // Validate artifact storage config
        self.artifact_storage.validate()?;

        // Validate performance gates config
        self.performance_gates.validate()?;

        Ok(())
    }

    /// Get platform-specific configuration
    pub fn get_platform_config(&self) -> PlatformSpecificConfig {
        match self.platform {
            CiCdPlatform::GitHubActions => PlatformSpecificConfig::github_actions(),
            CiCdPlatform::GitLabCI => PlatformSpecificConfig::gitlab_ci(),
            CiCdPlatform::Jenkins => PlatformSpecificConfig::jenkins(),
            CiCdPlatform::AzureDevOps => PlatformSpecificConfig::azure_devops(),
            CiCdPlatform::CircleCI => PlatformSpecificConfig::circle_ci(),
            CiCdPlatform::TravisCI => PlatformSpecificConfig::travis_ci(),
            CiCdPlatform::TeamCity => PlatformSpecificConfig::team_city(),
            CiCdPlatform::Buildkite => PlatformSpecificConfig::buildkite(),
            CiCdPlatform::Generic => PlatformSpecificConfig::generic(),
        }
    }
}

impl TestExecutionConfig {
    fn validate(&self) -> Result<(), String> {
        if self.test_timeout == 0 {
            return Err("Test timeout cannot be zero".to_string());
        }

        if self.test_iterations == 0 {
            return Err("Test iterations cannot be zero".to_string());
        }

        if self.max_concurrent_tests == 0 {
            return Err("Max concurrent tests cannot be zero".to_string());
        }

        Ok(())
    }
}

impl BaselineManagementConfig {
    fn validate(&self) -> Result<(), String> {
        if self.min_improvement_threshold < 0.0 || self.min_improvement_threshold > 1.0 {
            return Err("Min improvement threshold must be between 0.0 and 1.0".to_string());
        }

        if self.max_degradation_threshold < 0.0 || self.max_degradation_threshold > 1.0 {
            return Err("Max degradation threshold must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

impl ReportingConfig {
    fn validate(&self) -> Result<(), String> {
        if !self.generate_html && !self.generate_json && !self.generate_junit && !self.generate_markdown && !self.generate_pdf {
            return Err("At least one report format must be enabled".to_string());
        }

        Ok(())
    }
}

impl ArtifactStorageConfig {
    fn validate(&self) -> Result<(), String> {
        if self.enabled {
            match &self.provider {
                ArtifactStorageProvider::Local(path) => {
                    if path.as_os_str().is_empty() {
                        return Err("Local storage path cannot be empty".to_string());
                    }
                }
                ArtifactStorageProvider::S3 { bucket, .. } => {
                    if bucket.is_empty() {
                        return Err("S3 bucket name cannot be empty".to_string());
                    }
                }
                ArtifactStorageProvider::GCS { bucket, .. } => {
                    if bucket.is_empty() {
                        return Err("GCS bucket name cannot be empty".to_string());
                    }
                }
                ArtifactStorageProvider::AzureBlob { account, container, .. } => {
                    if account.is_empty() || container.is_empty() {
                        return Err("Azure Blob account and container cannot be empty".to_string());
                    }
                }
                ArtifactStorageProvider::FTP { host, .. } => {
                    if host.is_empty() {
                        return Err("FTP host cannot be empty".to_string());
                    }
                }
                ArtifactStorageProvider::HTTP { base_url, .. } => {
                    if base_url.is_empty() {
                        return Err("HTTP base URL cannot be empty".to_string());
                    }
                }
            }
        }

        Ok(())
    }
}

impl PerformanceGatesConfig {
    fn validate(&self) -> Result<(), String> {
        if self.enabled && self.metric_gates.is_empty() {
            return Err("Performance gates are enabled but no gates are configured".to_string());
        }

        for (metric_type, gate) in &self.metric_gates {
            gate.validate(metric_type)?;
        }

        Ok(())
    }
}

impl MetricGate {
    fn validate(&self, _metric_type: &MetricType) -> Result<(), String> {
        if self.threshold < 0.0 {
            return Err("Gate threshold cannot be negative".to_string());
        }

        match self.gate_type {
            GateType::Relative => {
                if self.threshold > 10.0 { // 1000% change seems unreasonable
                    return Err("Relative gate threshold seems unreasonably high".to_string());
                }
            }
            _ => {}
        }

        Ok(())
    }
}

/// Platform-specific configuration settings
#[derive(Debug, Clone)]
pub struct PlatformSpecificConfig {
    /// Environment variable names for common values
    pub env_vars: HashMap<String, String>,
    /// Default artifact paths
    pub artifact_paths: Vec<String>,
    /// Platform-specific commands
    pub commands: HashMap<String, String>,
    /// Platform limitations
    pub limitations: PlatformLimitations,
}

/// Platform limitations and constraints
#[derive(Debug, Clone)]
pub struct PlatformLimitations {
    /// Maximum job runtime in minutes
    pub max_job_runtime_minutes: Option<u32>,
    /// Maximum artifact size in MB
    pub max_artifact_size_mb: Option<usize>,
    /// Maximum number of parallel jobs
    pub max_parallel_jobs: Option<usize>,
    /// Supported operating systems
    pub supported_os: Vec<String>,
}

impl PlatformSpecificConfig {
    fn github_actions() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "GITHUB_ACTIONS".to_string());
        env_vars.insert("BUILD_ID".to_string(), "GITHUB_RUN_ID".to_string());
        env_vars.insert("BRANCH".to_string(), "GITHUB_REF_NAME".to_string());
        env_vars.insert("COMMIT".to_string(), "GITHUB_SHA".to_string());

        let mut commands = HashMap::new();
        commands.insert("cache_key".to_string(), "echo \"::set-output name=cache-key::${{ hashFiles('**/Cargo.lock') }}\"".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands,
            limitations: PlatformLimitations {
                max_job_runtime_minutes: Some(360), // 6 hours
                max_artifact_size_mb: Some(2048),   // 2GB
                max_parallel_jobs: Some(20),
                supported_os: vec!["ubuntu-latest".to_string(), "windows-latest".to_string(), "macos-latest".to_string()],
            },
        }
    }

    fn gitlab_ci() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "GITLAB_CI".to_string());
        env_vars.insert("BUILD_ID".to_string(), "CI_JOB_ID".to_string());
        env_vars.insert("BRANCH".to_string(), "CI_COMMIT_REF_NAME".to_string());
        env_vars.insert("COMMIT".to_string(), "CI_COMMIT_SHA".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: None, // Configurable
                max_artifact_size_mb: Some(1024), // 1GB default
                max_parallel_jobs: None, // Depends on plan
                supported_os: vec!["linux".to_string(), "windows".to_string(), "macos".to_string()],
            },
        }
    }

    fn jenkins() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "JENKINS".to_string());
        env_vars.insert("BUILD_ID".to_string(), "BUILD_NUMBER".to_string());
        env_vars.insert("BRANCH".to_string(), "BRANCH_NAME".to_string());
        env_vars.insert("COMMIT".to_string(), "GIT_COMMIT".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: None, // Configurable
                max_artifact_size_mb: None,    // Configurable
                max_parallel_jobs: None,       // Configurable
                supported_os: vec!["linux".to_string(), "windows".to_string(), "macos".to_string()],
            },
        }
    }

    fn azure_devops() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "AZURE_DEVOPS".to_string());
        env_vars.insert("BUILD_ID".to_string(), "BUILD_BUILDID".to_string());
        env_vars.insert("BRANCH".to_string(), "BUILD_SOURCEBRANCHNAME".to_string());
        env_vars.insert("COMMIT".to_string(), "BUILD_SOURCEVERSION".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: Some(360), // 6 hours
                max_artifact_size_mb: Some(100),    // 100MB
                max_parallel_jobs: Some(10),
                supported_os: vec!["ubuntu-latest".to_string(), "windows-latest".to_string(), "macos-latest".to_string()],
            },
        }
    }

    fn circle_ci() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "CIRCLECI".to_string());
        env_vars.insert("BUILD_ID".to_string(), "CIRCLE_BUILD_NUM".to_string());
        env_vars.insert("BRANCH".to_string(), "CIRCLE_BRANCH".to_string());
        env_vars.insert("COMMIT".to_string(), "CIRCLE_SHA1".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: Some(300), // 5 hours
                max_artifact_size_mb: Some(3072),   // 3GB
                max_parallel_jobs: Some(16),
                supported_os: vec!["linux".to_string(), "macos".to_string(), "windows".to_string()],
            },
        }
    }

    fn travis_ci() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "TRAVIS".to_string());
        env_vars.insert("BUILD_ID".to_string(), "TRAVIS_BUILD_ID".to_string());
        env_vars.insert("BRANCH".to_string(), "TRAVIS_BRANCH".to_string());
        env_vars.insert("COMMIT".to_string(), "TRAVIS_COMMIT".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: Some(50), // 50 minutes
                max_artifact_size_mb: Some(100),   // 100MB
                max_parallel_jobs: Some(5),
                supported_os: vec!["linux".to_string(), "osx".to_string(), "windows".to_string()],
            },
        }
    }

    fn team_city() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "TEAMCITY".to_string());
        env_vars.insert("BUILD_ID".to_string(), "BUILD_NUMBER".to_string());
        env_vars.insert("BRANCH".to_string(), "teamcity.build.branch".to_string());
        env_vars.insert("COMMIT".to_string(), "BUILD_VCS_NUMBER".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: None, // Configurable
                max_artifact_size_mb: None,    // Configurable
                max_parallel_jobs: None,       // Configurable
                supported_os: vec!["linux".to_string(), "windows".to_string(), "macos".to_string()],
            },
        }
    }

    fn buildkite() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "BUILDKITE".to_string());
        env_vars.insert("BUILD_ID".to_string(), "BUILDKITE_BUILD_ID".to_string());
        env_vars.insert("BRANCH".to_string(), "BUILDKITE_BRANCH".to_string());
        env_vars.insert("COMMIT".to_string(), "BUILDKITE_COMMIT".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: None, // Configurable
                max_artifact_size_mb: None,    // Configurable
                max_parallel_jobs: None,       // Configurable
                supported_os: vec!["linux".to_string(), "windows".to_string(), "macos".to_string()],
            },
        }
    }

    fn generic() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("CI".to_string(), "true".to_string());
        env_vars.insert("BUILD_ID".to_string(), "BUILD_ID".to_string());
        env_vars.insert("BRANCH".to_string(), "BRANCH".to_string());
        env_vars.insert("COMMIT".to_string(), "COMMIT_SHA".to_string());

        Self {
            env_vars,
            artifact_paths: vec!["./target/criterion".to_string()],
            commands: HashMap::new(),
            limitations: PlatformLimitations {
                max_job_runtime_minutes: None,
                max_artifact_size_mb: None,
                max_parallel_jobs: None,
                supported_os: vec!["linux".to_string(), "windows".to_string(), "macos".to_string()],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = CiCdAutomationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_test_execution_config() {
        let mut config = TestExecutionConfig::default();
        config.test_timeout = 0;
        assert!(config.validate().is_err());

        config.test_timeout = 3600;
        config.test_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_platform_specific_configs() {
        let github_config = PlatformSpecificConfig::github_actions();
        assert!(github_config.env_vars.contains_key("CI"));
        assert_eq!(github_config.env_vars["CI"], "GITHUB_ACTIONS");

        let gitlab_config = PlatformSpecificConfig::gitlab_ci();
        assert_eq!(gitlab_config.env_vars["CI"], "GITLAB_CI");
    }

    #[test]
    fn test_metric_gate_validation() {
        let gate = MetricGate {
            gate_type: GateType::Relative,
            threshold: -1.0,
            operator: ComparisonOperator::LessThanOrEqual,
            severity: GateSeverity::Error,
            enabled: true,
        };

        assert!(gate.validate(&MetricType::ExecutionTime).is_err());
    }

    #[test]
    fn test_platform_enum_serialization() {
        let platform = CiCdPlatform::GitHubActions;
        let serialized = serde_json::to_string(&platform).unwrap();
        let deserialized: CiCdPlatform = serde_json::from_str(&serialized).unwrap();
        assert_eq!(platform, deserialized);
    }

    #[test]
    fn test_comprehensive_config() {
        let mut config = CiCdAutomationConfig::default();

        // Enable all features
        config.reporting.generate_html = true;
        config.reporting.generate_json = true;
        config.reporting.generate_markdown = true;
        config.reporting.generate_pdf = true;
        config.artifact_storage.enabled = true;
        config.performance_gates.enabled = true;

        assert!(config.validate().is_ok());
    }
}