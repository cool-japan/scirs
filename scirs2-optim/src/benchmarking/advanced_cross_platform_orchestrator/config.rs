//! Configuration structures for advanced cross-platform testing orchestrator
//!
//! This module provides comprehensive configuration management for cross-platform
//! testing including cloud providers, containers, resource limits, and CI/CD integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use super::types::*;

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Enable cloud-based testing
    pub enable_cloud_testing: bool,
    /// Enable container-based testing
    pub enable_container_testing: bool,
    /// Enable parallel platform testing
    pub enable_parallel_testing: bool,
    /// Maximum concurrent test jobs
    pub max_concurrent_jobs: usize,
    /// Test matrix configuration
    pub matrix_config: TestMatrixConfig,
    /// Cloud provider settings
    pub cloud_config: CloudConfig,
    /// Container configuration
    pub container_config: ContainerConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Reporting configuration
    pub reporting_config: ReportingConfig,
    /// CI/CD integration settings
    pub ci_cd_config: Option<CiCdIntegrationConfig>,
}

/// Test matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMatrixConfig {
    /// Target platforms
    pub platforms: Vec<PlatformSpec>,
    /// Rust versions to test
    pub rust_versions: Vec<String>,
    /// Feature combinations
    pub feature_combinations: Vec<FeatureCombination>,
    /// Optimization levels
    pub optimization_levels: Vec<OptimizationLevel>,
    /// Build profiles
    pub build_profiles: Vec<BuildProfile>,
    /// Test scenarios
    pub test_scenarios: Vec<TestScenario>,
}

/// Platform specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformSpec {
    /// Platform target
    pub target: PlatformTarget,
    /// Platform priority (1-10, higher = more important)
    pub priority: u8,
    /// Required for release
    pub required_for_release: bool,
    /// Performance baseline platform
    pub is_baseline: bool,
    /// Platform-specific configuration
    pub config: HashMap<String, String>,
    /// Resource requirements
    pub resource_requirements: PlatformResourceRequirements,
}

/// Platform resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformResourceRequirements {
    /// CPU cores required
    pub cpu_cores: usize,
    /// Memory required (MB)
    pub memory_mb: usize,
    /// Disk space required (MB)
    pub disk_mb: usize,
    /// GPU required
    pub gpu_required: bool,
    /// Network bandwidth (Mbps)
    pub network_bandwidth: Option<f64>,
}

/// Feature combination for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCombination {
    /// Combination name
    pub name: String,
    /// Enabled features
    pub enabled_features: Vec<String>,
    /// Disabled features
    pub disabled_features: Vec<String>,
    /// Test importance
    pub importance: FeatureImportance,
}

/// Build profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProfile {
    /// Profile name
    pub name: String,
    /// Cargo profile settings
    pub settings: HashMap<String, String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Compiler flags
    pub rustflags: Vec<String>,
}

/// Test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    /// Scenario name
    pub name: String,
    /// Test commands
    pub commands: Vec<String>,
    /// Test category
    pub category: TestCategory,
    /// Timeout
    pub timeout: Duration,
    /// Expected results
    pub expected_results: HashMap<String, String>,
}

/// Cloud provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// AWS configuration
    pub aws: Option<AwsConfig>,
    /// Azure configuration
    pub azure: Option<AzureConfig>,
    /// GCP configuration
    pub gcp: Option<GcpConfig>,
    /// GitHub Actions configuration
    pub github_actions: Option<GitHubActionsConfig>,
    /// Custom cloud providers
    pub custom_providers: Vec<CustomCloudConfig>,
    /// Default authentication
    pub default_auth: Option<CloudAuthConfig>,
    /// Cost optimization settings
    pub cost_optimization: CostOptimizationSettings,
}

/// AWS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsConfig {
    /// AWS region
    pub region: String,
    /// EC2 instance types
    pub instance_types: HashMap<PlatformTarget, String>,
    /// AMI mappings
    pub ami_mappings: HashMap<String, String>,
    /// VPC settings
    pub vpc_id: Option<String>,
    /// Subnet ID
    pub subnet_id: Option<String>,
    /// Security groups
    pub security_groups: Vec<String>,
    /// Key pair name
    pub key_pair: Option<String>,
    /// IAM role
    pub iam_role: Option<String>,
    /// Spot instance configuration
    pub use_spot_instances: bool,
    /// Maximum spot price
    pub max_spot_price: Option<f64>,
}

/// Azure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Azure subscription ID
    pub subscription_id: String,
    /// Resource group
    pub resource_group: String,
    /// VM size mappings
    pub vm_sizes: HashMap<PlatformTarget, String>,
    /// Image references
    pub images: HashMap<String, AzureImageRef>,
    /// Virtual network
    pub vnet: Option<String>,
    /// Subnet
    pub subnet: Option<String>,
    /// Network security group
    pub network_security_group: Option<String>,
}

/// Azure image reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureImageRef {
    pub publisher: String,
    pub offer: String,
    pub sku: String,
    pub version: String,
}

/// GCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpConfig {
    /// GCP project ID
    pub project_id: String,
    /// Zone
    pub zone: String,
    /// Machine types
    pub machine_types: HashMap<PlatformTarget, String>,
    /// Image families
    pub image_families: HashMap<String, String>,
    /// Network
    pub network: Option<String>,
    /// Subnetwork
    pub subnetwork: Option<String>,
    /// Service account
    pub service_account: Option<String>,
    /// Preemptible instances
    pub use_preemptible: bool,
}

/// GitHub Actions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubActionsConfig {
    /// GitHub repository
    pub repository: String,
    /// Workflow templates
    pub workflow_templates: HashMap<PlatformTarget, PathBuf>,
    /// Runner labels
    pub runner_labels: HashMap<PlatformTarget, Vec<String>>,
    /// Secrets to inject
    pub secrets: Vec<String>,
    /// Matrix strategy
    pub matrix_strategy: String,
}

/// Custom cloud provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCloudConfig {
    /// Provider name
    pub name: String,
    /// API endpoint
    pub endpoint: String,
    /// Authentication configuration
    pub auth: CloudAuthConfig,
    /// Platform mappings
    pub platform_mappings: Vec<CustomPlatformMapping>,
    /// Resource templates
    pub resource_templates: HashMap<String, String>,
}

/// Custom platform mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPlatformMapping {
    /// Platform target
    pub platform: PlatformTarget,
    /// Instance template
    pub instance_template: String,
    /// Configuration overrides
    pub config_overrides: HashMap<String, String>,
}

/// Cost optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationSettings {
    /// Enable cost optimization
    pub enabled: bool,
    /// Maximum cost per test run (USD)
    pub max_cost_per_run: Option<f64>,
    /// Prefer cheaper instances
    pub prefer_cheaper_instances: bool,
    /// Use spot/preemptible instances
    pub use_discounted_instances: bool,
    /// Auto-shutdown timeout
    pub auto_shutdown_timeout: Duration,
}

/// Container configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Container runtime
    pub runtime: ContainerRuntime,
    /// Registry configuration
    pub registry: RegistryConfig,
    /// Image tag strategy
    pub image_tag_strategy: ImageTagStrategy,
    /// Resource limits
    pub resource_limits: ContainerResourceLimits,
    /// Network configuration
    pub network: ContainerNetworkConfig,
    /// Volume mounts
    pub volumes: Vec<String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
}

/// Container registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Registry URL
    pub url: String,
    /// Username
    pub username: Option<String>,
    /// Authentication method
    pub auth: RegistryAuth,
    /// Default namespace
    pub namespace: String,
    /// Image prefix
    pub image_prefix: String,
}

/// Container resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResourceLimits {
    /// CPU limit (cores)
    pub cpu_limit: Option<f64>,
    /// Memory limit (MB)
    pub memory_limit: Option<usize>,
    /// Network bandwidth limit (Mbps)
    pub network_limit: Option<f64>,
    /// I/O bandwidth limit (MB/s)
    pub io_limit: Option<f64>,
    /// Process limit
    pub process_limit: Option<usize>,
}

/// Container network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerNetworkConfig {
    /// Network mode
    pub mode: NetworkMode,
    /// Port mappings
    pub port_mappings: Vec<(u16, u16)>,
    /// DNS servers
    pub dns_servers: Vec<String>,
    /// Extra hosts
    pub extra_hosts: HashMap<String, String>,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum test duration
    pub max_test_duration: Duration,
    /// Maximum memory usage (MB)
    pub max_memory_usage: usize,
    /// Maximum disk usage (MB)
    pub max_disk_usage: usize,
    /// Maximum network usage (MB)
    pub max_network_usage: usize,
    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,
    /// Budget limits
    pub budget_limits: BudgetLimits,
}

/// Budget limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetLimits {
    /// Daily budget (USD)
    pub daily_budget: Option<f64>,
    /// Monthly budget (USD)
    pub monthly_budget: Option<f64>,
    /// Per-test budget (USD)
    pub per_test_budget: Option<f64>,
    /// Alert threshold (percentage)
    pub alert_threshold: f64,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Report formats
    pub formats: Vec<ReportFormat>,
    /// Include detailed logs
    pub include_logs: bool,
    /// Include performance metrics
    pub include_metrics: bool,
    /// Include cost analysis
    pub include_costs: bool,
    /// Email notifications
    pub email_notifications: Vec<String>,
    /// Slack webhook
    pub slack_webhook: Option<String>,
}

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdIntegrationConfig {
    /// CI/CD platform
    pub platform: CiCdPlatform,
    /// Integration settings
    pub settings: HashMap<String, String>,
    /// Webhook URLs
    pub webhooks: Vec<String>,
    /// Status check configuration
    pub status_checks: StatusCheckConfig,
}

/// Status check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusCheckConfig {
    /// Required checks
    pub required_checks: Vec<String>,
    /// Optional checks
    pub optional_checks: Vec<String>,
    /// Failure threshold
    pub failure_threshold: f64,
    /// Timeout for checks
    pub check_timeout: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_cloud_testing: false,
            enable_container_testing: true,
            enable_parallel_testing: true,
            max_concurrent_jobs: 4,
            matrix_config: TestMatrixConfig::default(),
            cloud_config: CloudConfig::default(),
            container_config: ContainerConfig::default(),
            resource_limits: ResourceLimits::default(),
            reporting_config: ReportingConfig::default(),
            ci_cd_config: None,
        }
    }
}

impl Default for TestMatrixConfig {
    fn default() -> Self {
        Self {
            platforms: vec![
                PlatformSpec {
                    target: PlatformTarget::LinuxX86_64,
                    priority: 10,
                    required_for_release: true,
                    is_baseline: true,
                    config: HashMap::new(),
                    resource_requirements: PlatformResourceRequirements::default(),
                },
                PlatformSpec {
                    target: PlatformTarget::WindowsX86_64,
                    priority: 9,
                    required_for_release: true,
                    is_baseline: false,
                    config: HashMap::new(),
                    resource_requirements: PlatformResourceRequirements::default(),
                },
                PlatformSpec {
                    target: PlatformTarget::MacOSX86_64,
                    priority: 8,
                    required_for_release: true,
                    is_baseline: false,
                    config: HashMap::new(),
                    resource_requirements: PlatformResourceRequirements::default(),
                },
            ],
            rust_versions: vec!["stable".to_string(), "beta".to_string()],
            feature_combinations: vec![
                FeatureCombination {
                    name: "default".to_string(),
                    enabled_features: vec!["default".to_string()],
                    disabled_features: vec![],
                    importance: FeatureImportance::Critical,
                },
            ],
            optimization_levels: vec![
                OptimizationLevel::Debug,
                OptimizationLevel::Release,
            ],
            build_profiles: vec![
                BuildProfile {
                    name: "test".to_string(),
                    settings: HashMap::new(),
                    env_vars: HashMap::new(),
                    rustflags: vec![],
                },
            ],
            test_scenarios: vec![
                TestScenario {
                    name: "unit_tests".to_string(),
                    commands: vec!["cargo test".to_string()],
                    category: TestCategory::Unit,
                    timeout: Duration::from_secs(300),
                    expected_results: HashMap::new(),
                },
            ],
        }
    }
}

impl Default for PlatformResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 2,
            memory_mb: 4096,
            disk_mb: 10240,
            gpu_required: false,
            network_bandwidth: Some(100.0),
        }
    }
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            aws: None,
            azure: None,
            gcp: None,
            github_actions: None,
            custom_providers: vec![],
            default_auth: None,
            cost_optimization: CostOptimizationSettings::default(),
        }
    }
}

impl Default for CostOptimizationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cost_per_run: Some(10.0),
            prefer_cheaper_instances: true,
            use_discounted_instances: true,
            auto_shutdown_timeout: Duration::from_secs(3600),
        }
    }
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            runtime: ContainerRuntime::Docker,
            registry: RegistryConfig::default(),
            image_tag_strategy: ImageTagStrategy::GitHash,
            resource_limits: ContainerResourceLimits::default(),
            network: ContainerNetworkConfig::default(),
            volumes: vec![],
            env_vars: HashMap::new(),
        }
    }
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            url: "docker.io".to_string(),
            username: None,
            auth: RegistryAuth::None,
            namespace: "scirs2".to_string(),
            image_prefix: "test".to_string(),
        }
    }
}

impl Default for ContainerResourceLimits {
    fn default() -> Self {
        Self {
            cpu_limit: Some(2.0),
            memory_limit: Some(4096),
            network_limit: None,
            io_limit: None,
            process_limit: Some(1024),
        }
    }
}

impl Default for ContainerNetworkConfig {
    fn default() -> Self {
        Self {
            mode: NetworkMode::Bridge,
            port_mappings: vec![],
            dns_servers: vec![],
            extra_hosts: HashMap::new(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_test_duration: Duration::from_secs(3600),
            max_memory_usage: 8192,
            max_disk_usage: 20480,
            max_network_usage: 1024,
            max_concurrent_tests: 4,
            budget_limits: BudgetLimits::default(),
        }
    }
}

impl Default for BudgetLimits {
    fn default() -> Self {
        Self {
            daily_budget: Some(100.0),
            monthly_budget: Some(1000.0),
            per_test_budget: Some(5.0),
            alert_threshold: 80.0,
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            output_dir: "test_reports".into(),
            formats: vec![ReportFormat::Json, ReportFormat::Html],
            include_logs: true,
            include_metrics: true,
            include_costs: true,
            email_notifications: vec![],
            slack_webhook: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OrchestratorConfig::default();
        assert!(!config.enable_cloud_testing);
        assert!(config.enable_container_testing);
        assert_eq!(config.max_concurrent_jobs, 4);
    }

    #[test]
    fn test_platform_spec_creation() {
        let spec = PlatformSpec {
            target: PlatformTarget::LinuxX86_64,
            priority: 10,
            required_for_release: true,
            is_baseline: true,
            config: HashMap::new(),
            resource_requirements: PlatformResourceRequirements::default(),
        };

        assert_eq!(spec.priority, 10);
        assert!(spec.required_for_release);
        assert!(spec.is_baseline);
    }

    #[test]
    fn test_resource_requirements() {
        let reqs = PlatformResourceRequirements::default();
        assert_eq!(reqs.cpu_cores, 2);
        assert_eq!(reqs.memory_mb, 4096);
        assert!(!reqs.gpu_required);
    }
}