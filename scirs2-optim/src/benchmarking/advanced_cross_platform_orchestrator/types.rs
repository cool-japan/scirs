//! Core data types and enums for advanced cross-platform testing orchestrator
//!
//! This module provides comprehensive type definitions for cross-platform testing
//! including platforms, features, cloud providers, containers, and reporting structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

// Re-export types from cross_platform_tester
pub use crate::benchmarking::cross_platform_tester::{
    PlatformTarget, TestCategory, TestResult, TestStatus, PerformanceMetrics,
    CompatibilityIssue, PlatformRecommendation
};

/// Feature importance levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureImportance {
    Critical,
    High,
    Medium,
    Low,
}

/// Optimization levels for builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,
    Release,
    ReleaseLTO,
    MinSize,
    Custom(String),
}

/// Cloud authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudAuthConfig {
    ApiKey { key: String },
    OAuth { client_id: String, client_secret: String },
    ServiceAccount { key_file: PathBuf },
    None,
    Custom { config: HashMap<String, String> },
}

/// Container runtime options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerRuntime {
    Docker,
    Podman,
    Containerd,
    Custom(String),
}

/// Registry authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegistryAuth {
    None,
    UsernamePassword { username: String, password: String },
    Token { token: String },
    ServiceAccount { key_file: PathBuf },
}

/// Image tagging strategy for containers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageTagStrategy {
    GitHash,
    GitCommit,
    Timestamp,
    Sequential,
    SemVer,
    Custom(String),
}

/// Container network modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMode {
    Bridge,
    Host,
    None,
    Overlay,
    Custom(String),
}

/// Cloud instance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudInstanceStatus {
    Pending,
    Running,
    Stopping,
    Stopped,
    Terminated,
    Failed,
}

/// Port protocol types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PortProtocol {
    TCP,
    UDP,
    SCTP,
}

/// Container status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerStatus {
    Created,
    Running,
    Paused,
    Restarting,
    Removing,
    Exited,
    Dead,
}

/// Report output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Markdown,
    Pdf,
    Xml,
    Csv,
}

/// CI/CD platform types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CiCdPlatform {
    GitHubActions,
    GitLabCI,
    JenkinsCI,
    TeamCity,
    Azure,
    CircleCI,
    TravisCI,
    Custom(String),
}

/// Performance trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Resource type classifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Storage,
    Network,
    GPU,
}

/// Resource allocation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStatus {
    Available,
    Allocated,
    Overcommitted,
    Exhausted,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Test execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionContext {
    /// Test execution ID
    pub execution_id: String,
    /// Platform being tested
    pub platform: PlatformTarget,
    /// Rust version
    pub rust_version: String,
    /// Feature combination
    pub features: Vec<String>,
    /// Optimization level
    pub optimization: OptimizationLevel,
    /// Build profile
    pub build_profile: String,
    /// Test scenarios
    pub scenarios: Vec<String>,
    /// Start time
    pub start_time: SystemTime,
    /// Expected duration
    pub expected_duration: Duration,
}

/// Cloud instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInstance {
    /// Instance ID
    pub instance_id: String,
    /// Provider
    pub provider: String,
    /// Instance type
    pub instance_type: String,
    /// Platform target
    pub platform: PlatformTarget,
    /// Status
    pub status: CloudInstanceStatus,
    /// Public IP
    pub public_ip: Option<String>,
    /// Private IP
    pub private_ip: Option<String>,
    /// Launch time
    pub launch_time: SystemTime,
    /// Cost per hour
    pub cost_per_hour: f64,
    /// Configuration
    pub config: HashMap<String, String>,
}

/// Container information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    /// Container ID
    pub container_id: String,
    /// Container name
    pub name: String,
    /// Image
    pub image: String,
    /// Platform target
    pub platform: PlatformTarget,
    /// Status
    pub status: ContainerStatus,
    /// Port mappings
    pub ports: Vec<PortMapping>,
    /// Resource usage
    pub resource_usage: ContainerStats,
    /// Create time
    pub created_at: SystemTime,
    /// Start time
    pub started_at: Option<SystemTime>,
}

/// Port mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Host port
    pub host_port: u16,
    /// Container port
    pub container_port: u16,
    /// Protocol
    pub protocol: PortProtocol,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (percentage)
    pub cpu_usage: f64,
    /// Memory usage (MB)
    pub memory_usage: usize,
    /// Memory limit (MB)
    pub memory_limit: usize,
    /// Network I/O
    pub network_io: NetworkIO,
    /// Block I/O
    pub block_io: BlockIO,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Network I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIO {
    /// Bytes received
    pub rx_bytes: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Packets received
    pub rx_packets: u64,
    /// Packets transmitted
    pub tx_packets: u64,
}

/// Block I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockIO {
    /// Bytes read
    pub read_bytes: u64,
    /// Bytes written
    pub write_bytes: u64,
    /// Read operations
    pub read_ops: u64,
    /// Write operations
    pub write_ops: u64,
}

/// Cloud cost tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCosts {
    /// Total cost
    pub total_cost: f64,
    /// Cost by service
    pub cost_by_service: HashMap<String, f64>,
    /// Cost by region
    pub cost_by_region: HashMap<String, f64>,
    /// Cost by instance type
    pub cost_by_instance_type: HashMap<String, f64>,
    /// Currency
    pub currency: String,
    /// Billing period
    pub billing_period: (SystemTime, SystemTime),
}

/// Container statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStats {
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage (MB)
    pub memory_usage: usize,
    /// Memory limit (MB)
    pub memory_limit: usize,
    /// Network I/O
    pub network_io: NetworkIO,
    /// Block I/O
    pub block_io: BlockIO,
    /// Process count
    pub process_count: usize,
}

/// Compatibility matrix results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMatrix {
    /// Platform results
    pub platform_results: HashMap<PlatformTarget, Vec<TestResult>>,
    /// Feature compatibility
    pub feature_compatibility: HashMap<String, HashMap<PlatformTarget, bool>>,
    /// Performance comparison
    pub performance_comparison: HashMap<PlatformTarget, PerformanceMetrics>,
    /// Compatibility issues
    pub issues: Vec<CompatibilityIssue>,
    /// Overall compatibility score
    pub compatibility_score: f64,
}

/// Issue summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueSummary {
    /// Total issues
    pub total_issues: usize,
    /// Issues by severity
    pub issues_by_severity: HashMap<String, usize>,
    /// Issues by platform
    pub issues_by_platform: HashMap<PlatformTarget, usize>,
    /// Issues by category
    pub issues_by_category: HashMap<TestCategory, usize>,
    /// Blocking issues
    pub blocking_issues: usize,
}

/// Test matrix entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMatrixEntry {
    /// Entry ID
    pub id: String,
    /// Platform
    pub platform: PlatformTarget,
    /// Rust version
    pub rust_version: String,
    /// Features
    pub features: Vec<String>,
    /// Optimization level
    pub optimization: OptimizationLevel,
    /// Build profile
    pub build_profile: String,
    /// Test scenarios
    pub scenarios: Vec<String>,
    /// Priority
    pub priority: u8,
    /// Required for release
    pub required_for_release: bool,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Resource requirements
    pub resource_requirements: HashMap<ResourceType, f64>,
}

/// Orchestration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResult {
    /// Total tests executed
    pub total_tests: usize,
    /// Successful tests
    pub successful_tests: usize,
    /// Failed tests
    pub failed_tests: usize,
    /// Skipped tests
    pub skipped_tests: usize,
    /// Total execution time
    pub total_duration: Duration,
    /// Platform results
    pub platform_results: HashMap<PlatformTarget, Vec<TestResult>>,
    /// Compatibility matrix
    pub compatibility_matrix: CompatibilityMatrix,
    /// Issues summary
    pub issues_summary: IssueSummary,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Cloud costs
    pub cloud_costs: Option<CloudCosts>,
    /// Recommendations
    pub recommendations: Vec<PlatformRecommendation>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert name
    pub name: String,
    /// Alert level
    pub level: AlertLevel,
    /// Condition
    pub condition: String,
    /// Message template
    pub message_template: String,
    /// Notification channels
    pub channels: Vec<String>,
    /// Cooldown period
    pub cooldown: Duration,
}

impl Default for FeatureImportance {
    fn default() -> Self {
        FeatureImportance::Medium
    }
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Release
    }
}

impl Default for ContainerRuntime {
    fn default() -> Self {
        ContainerRuntime::Docker
    }
}

impl Default for RegistryAuth {
    fn default() -> Self {
        RegistryAuth::None
    }
}

impl Default for ImageTagStrategy {
    fn default() -> Self {
        ImageTagStrategy::GitHash
    }
}

impl Default for NetworkMode {
    fn default() -> Self {
        NetworkMode::Bridge
    }
}

impl Default for ReportFormat {
    fn default() -> Self {
        ReportFormat::Json
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_limit: 0,
            network_io: NetworkIO::default(),
            block_io: BlockIO::default(),
            timestamp: SystemTime::now(),
        }
    }
}

impl Default for CloudCosts {
    fn default() -> Self {
        Self {
            total_cost: 0.0,
            cost_by_service: HashMap::new(),
            cost_by_region: HashMap::new(),
            cost_by_instance_type: HashMap::new(),
            currency: "USD".to_string(),
            billing_period: (SystemTime::now(), SystemTime::now()),
        }
    }
}

impl Default for ContainerStats {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_limit: 0,
            network_io: NetworkIO::default(),
            block_io: BlockIO::default(),
            process_count: 0,
        }
    }
}

impl Default for NetworkIO {
    fn default() -> Self {
        Self {
            rx_bytes: 0,
            tx_bytes: 0,
            rx_packets: 0,
            tx_packets: 0,
        }
    }
}

impl Default for BlockIO {
    fn default() -> Self {
        Self {
            read_bytes: 0,
            write_bytes: 0,
            read_ops: 0,
            write_ops: 0,
        }
    }
}

impl Default for CompatibilityMatrix {
    fn default() -> Self {
        Self {
            platform_results: HashMap::new(),
            feature_compatibility: HashMap::new(),
            performance_comparison: HashMap::new(),
            issues: Vec::new(),
            compatibility_score: 0.0,
        }
    }
}

impl Default for IssueSummary {
    fn default() -> Self {
        Self {
            total_issues: 0,
            issues_by_severity: HashMap::new(),
            issues_by_platform: HashMap::new(),
            issues_by_category: HashMap::new(),
            blocking_issues: 0,
        }
    }
}

impl ToString for PlatformTarget {
    fn to_string(&self) -> String {
        match self {
            PlatformTarget::LinuxX86_64 => "linux-x86_64".to_string(),
            PlatformTarget::LinuxAarch64 => "linux-aarch64".to_string(),
            PlatformTarget::WindowsX86_64 => "windows-x86_64".to_string(),
            PlatformTarget::MacOSX86_64 => "macos-x86_64".to_string(),
            PlatformTarget::MacOSAarch64 => "macos-aarch64".to_string(),
            PlatformTarget::FreeBSDX86_64 => "freebsd-x86_64".to_string(),
            PlatformTarget::OpenBSDX86_64 => "openbsd-x86_64".to_string(),
            PlatformTarget::NetBSDX86_64 => "netbsd-x86_64".to_string(),
            PlatformTarget::SolarisX86_64 => "solaris-x86_64".to_string(),
            PlatformTarget::LinuxMips64 => "linux-mips64".to_string(),
            PlatformTarget::LinuxPowerPC64 => "linux-powerpc64".to_string(),
            PlatformTarget::LinuxS390X => "linux-s390x".to_string(),
            _ => format!("{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_importance_default() {
        assert_eq!(FeatureImportance::default(), FeatureImportance::Medium);
    }

    #[test]
    fn test_optimization_level_default() {
        matches!(OptimizationLevel::default(), OptimizationLevel::Release);
    }

    #[test]
    fn test_platform_target_to_string() {
        assert_eq!(PlatformTarget::LinuxX86_64.to_string(), "linux-x86_64");
        assert_eq!(PlatformTarget::WindowsX86_64.to_string(), "windows-x86_64");
        assert_eq!(PlatformTarget::MacOSX86_64.to_string(), "macos-x86_64");
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.cpu_usage, 0.0);
        assert_eq!(usage.memory_usage, 0);
    }

    #[test]
    fn test_container_runtime_default() {
        matches!(ContainerRuntime::default(), ContainerRuntime::Docker);
    }

    #[test]
    fn test_network_mode_default() {
        matches!(NetworkMode::default(), NetworkMode::Bridge);
    }
}