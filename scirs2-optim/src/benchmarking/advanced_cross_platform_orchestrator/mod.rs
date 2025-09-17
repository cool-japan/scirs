//! Advanced Cross-Platform Testing Orchestrator
//!
//! This module provides comprehensive cross-platform testing capabilities including:
//! - Multi-platform test execution (Linux, Windows, macOS, mobile, embedded)
//! - Cloud provider integration (AWS, Azure, GCP, GitHub Actions)
//! - Container runtime management (Docker, Podman)
//! - Test matrix generation and execution
//! - Resource allocation and cost tracking
//! - Result aggregation and compatibility analysis
//!
//! # Architecture
//!
//! The orchestrator is built with a modular architecture:
//!
//! - **config**: Configuration management for all components
//! - **types**: Core data types and platform definitions
//! - **orchestrator**: Main orchestration logic and workflow
//! - **matrix**: Test matrix generation and filtering
//! - **container**: Container runtime abstraction and management
//! - **cloud**: Cloud provider implementations and abstractions
//! - **resources**: Resource allocation, usage tracking, and cost management
//! - **aggregator**: Result collection and compatibility analysis
//!
//! # Usage Example
//!
//! ```rust
//! use scirs2_optim::benchmarking::advanced_cross_platform_orchestrator::{
//!     CrossPlatformOrchestrator, OrchestratorConfig, PlatformTarget
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = OrchestratorConfig::default();
//! let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
//!
//! // Execute cross-platform testing
//! let summary = orchestrator.execute_cross_platform_testing().await?;
//! println!("Testing completed with {} results", summary.total_tests);
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod types;
pub mod orchestrator;
pub mod matrix;
pub mod container;
pub mod cloud;
pub mod resources;
pub mod aggregator;

// Re-export core configuration types
pub use config::{
    OrchestratorConfig, TestMatrixConfig, ContainerConfig, CloudConfig,
    AwsConfig, AzureConfig, GcpConfig, GitHubActionsConfig, CustomCloudConfig,
    ResourceLimits, PlatformSpec, FeatureCombination, TestScenario,
    OptimizationLevel, BuildProfile, PlatformResourceRequirements,
    ContainerRegistryConfig, StorageConfig, NetworkConfig, SecurityConfig,
    MonitoringConfig, ComplianceConfig, MetricsConfig
};

// Re-export core data types
pub use types::{
    PlatformTarget, TestResult, TestStatus, PerformanceMetrics,
    CloudInstance, CloudInstanceStatus, ContainerInfo, ContainerStatus,
    ContainerStats, ResourceType, ResourceUsage, TestMatrixEntry,
    CrossPlatformTestingSummary, CompatibilityMatrix, CompatibilityIssue,
    TestExecutionResult, PlatformTestResult, CloudProviderType,
    ContainerRuntime, NetworkPort, PlatformCapabilities, HardwareInfo,
    SoftwareInfo, EnvironmentVariables, TestEnvironment,
    CrossPlatformMetrics, SecurityContext, ComplianceReport,
    Performance, BenchmarkResult, NumericalResult
};

// Re-export main orchestrator
pub use orchestrator::{
    CrossPlatformOrchestrator, ExecutionContext, TestExecutor,
    PlatformTestExecutor, CloudTestExecutor, ContainerTestExecutor
};

// Re-export test matrix functionality
pub use matrix::{
    TestMatrixGenerator, MatrixFilter, MatrixStatistics
};

// Re-export container management
pub use container::{
    ContainerManager, ContainerRuntimeTrait, DockerRuntime, PodmanRuntime
};

// Re-export cloud provider abstractions
pub use cloud::{
    CloudProvider, AwsProvider, AzureProvider, GcpProvider,
    GitHubActionsProvider, CustomProvider
};

// Re-export resource management
pub use resources::{
    PlatformResourceManager, ResourceUsageTracker, CostTracker, CostEntry
};

// Re-export result aggregation
pub use aggregator::{
    ResultAggregator, AggregationStatistics
};

/// Convenience function to create a default orchestrator
pub fn create_default_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let config = OrchestratorConfig::default();
    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create an orchestrator with cloud testing enabled
pub fn create_cloud_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_cloud_testing = true;
    config.cloud_config.enable_aws = true;
    config.cloud_config.enable_azure = true;
    config.cloud_config.enable_gcp = true;
    config.cloud_config.enable_github_actions = true;

    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create an orchestrator with container testing enabled
pub fn create_container_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_container_testing = true;
    config.container_config.runtime = ContainerRuntime::Docker;
    config.container_config.pull_latest_images = true;

    CrossPlatformOrchestrator::new(config)
}

/// Convenience function to create a minimal orchestrator for CI/CD
pub fn create_ci_orchestrator() -> crate::error::Result<CrossPlatformOrchestrator> {
    let mut config = OrchestratorConfig::default();
    config.enable_parallel_testing = true;
    config.max_concurrent_jobs = 4;
    config.enable_container_testing = true;
    config.enable_cloud_testing = false; // Disable cloud for CI to save costs

    // Configure for essential platforms only
    config.matrix_config.platforms = vec![
        PlatformSpec {
            target: PlatformTarget::LinuxX86_64,
            priority: 10,
            required_for_release: true,
            resource_requirements: PlatformResourceRequirements::default(),
        },
        PlatformSpec {
            target: PlatformTarget::WindowsX86_64,
            priority: 9,
            required_for_release: true,
            resource_requirements: PlatformResourceRequirements::default(),
        },
        PlatformSpec {
            target: PlatformTarget::MacOSX86_64,
            priority: 8,
            required_for_release: true,
            resource_requirements: PlatformResourceRequirements::default(),
        },
    ];

    CrossPlatformOrchestrator::new(config)
}

/// High-level async function to run complete cross-platform testing
pub async fn run_cross_platform_testing(
    config: Option<OrchestratorConfig>
) -> crate::error::Result<CrossPlatformTestingSummary> {
    let config = config.unwrap_or_default();
    let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
    orchestrator.execute_cross_platform_testing().await
}

/// High-level async function to run testing on specific platforms
pub async fn run_platform_specific_testing(
    platforms: Vec<PlatformTarget>,
    config: Option<OrchestratorConfig>
) -> crate::error::Result<CrossPlatformTestingSummary> {
    let mut config = config.unwrap_or_default();

    // Filter platforms to only include requested ones
    config.matrix_config.platforms = config.matrix_config.platforms
        .into_iter()
        .filter(|spec| platforms.contains(&spec.target))
        .collect();

    let mut orchestrator = CrossPlatformOrchestrator::new(config)?;
    orchestrator.execute_cross_platform_testing().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_orchestrator_creation() {
        let result = create_default_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cloud_orchestrator_creation() {
        let result = create_cloud_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_container_orchestrator_creation() {
        let result = create_container_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ci_orchestrator_creation() {
        let result = create_ci_orchestrator();
        assert!(result.is_ok());

        if let Ok(orchestrator) = result {
            // Verify CI-specific configuration
            assert!(orchestrator.config.enable_parallel_testing);
            assert_eq!(orchestrator.config.max_concurrent_jobs, 4);
            assert!(orchestrator.config.enable_container_testing);
            assert!(!orchestrator.config.enable_cloud_testing);
            assert_eq!(orchestrator.config.matrix_config.platforms.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_cross_platform_testing_execution() {
        let config = OrchestratorConfig::default();
        let result = run_cross_platform_testing(Some(config)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_platform_specific_testing() {
        let platforms = vec![PlatformTarget::LinuxX86_64, PlatformTarget::WindowsX86_64];
        let result = run_platform_specific_testing(platforms, None).await;
        assert!(result.is_ok());
    }
}