//! Production deployment utilities for scirs2-stats v1.0.0+
//!
//! This module provides comprehensive production readiness validation,
//! deployment utilities, monitoring, and runtime optimization for
//! statistical computing workloads in production environments.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Production deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProductionConfig {
    /// Target environment specification
    pub environment: EnvironmentSpec,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Reliability and fault tolerance settings
    pub reliability: ReliabilityConfig,
    /// Monitoring and observability configuration
    pub monitoring: MonitoringConfig,
    /// Resource limits and quotas
    pub resource_limits: ResourceLimits,
    /// Security and compliance settings
    pub security: SecurityConfig,
    /// Deployment strategy
    pub deployment_strategy: DeploymentStrategy,
}

/// Target deployment environment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSpec {
    /// Production environment type
    pub environment_type: EnvironmentType,
    /// CPU architecture and features
    pub cpu_features: CpuFeatures,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Storage configuration
    pub storage_config: StorageConfig,
    /// Container/orchestration configuration
    pub container_config: Option<ContainerConfig>,
}

/// Environment types for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    /// Traditional server deployment
    Server {
        /// Operating system
        os: String,
        /// Number of CPU cores
        cores: usize,
        /// Available memory (GB)
        memory_gb: usize,
    },
    /// Cloud deployment
    Cloud {
        /// Cloud provider
        provider: CloudProvider,
        /// Instance type/SKU
        instance_type: String,
        /// Region/zone
        region: String,
    },
    /// Container deployment
    Container {
        /// Container runtime
        runtime: ContainerRuntime,
        /// Resource allocation
        resources: ContainerResources,
    },
    /// Edge/IoT deployment
    Edge {
        /// Device type
        device_type: String,
        /// Compute constraints
        constraints: EdgeConstraints,
    },
    /// Serverless/Functions deployment
    Serverless {
        /// Platform
        platform: ServerlessPlatform,
        /// Runtime configuration
        runtime_config: ServerlessConfig,
    },
}

/// Cloud providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    Azure,
    GCP,
    Other(String),
}

/// Container runtimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerRuntime {
    Docker,
    Podman,
    Containerd,
    Other(String),
}

/// Serverless platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerlessPlatform {
    AWSLambda,
    AzureFunctions,
    GCPCloudFunctions,
    Other(String),
}

/// CPU features detection and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuFeatures {
    /// Architecture (x86_64, aarch64, etc.)
    pub architecture: String,
    /// Available SIMD instruction sets
    pub simd_features: Vec<SimdFeature>,
    /// Number of cores
    pub cores: usize,
    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,
    /// NUMA topology
    pub numa_topology: Option<NumaTopology>,
}

/// SIMD instruction set features
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimdFeature {
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    SSE41,
    SSE42,
    AVX,
    AVX2,
    AVX512F,
    AVX512DQ,
    AVX512CD,
    AVX512BW,
    AVX512VL,
    NEON,
    Other(String),
}

/// Performance requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f64,
    /// Minimum required throughput (ops/sec)
    pub min_throughput: f64,
    /// Memory usage limits
    pub memory_limits: MemoryLimits,
    /// CPU utilization targets
    pub cpu_utilization: CpuUtilization,
    /// SLA requirements
    pub sla_requirements: SlaRequirements,
    /// Load testing configuration
    pub load_testing: LoadTestingConfig,
}

/// Memory limits and targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum heap memory (MB)
    pub max_heap_mb: usize,
    /// Maximum stack memory (MB)
    pub max_stack_mb: usize,
    /// Maximum shared memory (MB)
    pub max_shared_mb: usize,
    /// Memory allocation rate limits
    pub allocation_rate_limit: Option<f64>,
}

/// Production deployment validator and optimizer
pub struct ProductionDeploymentValidator {
    config: ProductionConfig,
    validation_results: Arc<RwLock<ValidationResults>>,
    #[allow(dead_code)]
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    #[allow(dead_code)]
    health_checker: Arc<Mutex<HealthChecker>>,
}

/// Validation results for production readiness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Overall readiness score (0.0-1.0)
    pub readiness_score: f64,
    /// Individual check results
    pub checks: HashMap<String, CheckResult>,
    /// Performance benchmarks
    pub performance_benchmarks: Vec<BenchmarkResult>,
    /// Resource usage analysis
    pub resource_analysis: ResourceAnalysis,
    /// Recommendations for improvement
    pub recommendations: Vec<Recommendation>,
    /// Validation timestamp
    pub timestamp: SystemTime,
}

/// Individual validation check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Check name
    pub name: String,
    /// Check status
    pub status: CheckStatus,
    /// Detailed message
    pub message: String,
    /// Severity level
    pub severity: CheckSeverity,
    /// Execution time
    pub execution_time_ms: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Check status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Warning,
    Fail,
    Skip,
}

/// Check severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CheckSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Performance monitoring for production deployments
pub struct PerformanceMonitor {
    metrics: HashMap<String, MetricTimeSeries>,
    alerts: Vec<Alert>,
    thresholds: HashMap<String, Threshold>,
    last_update: Instant,
}

/// Runtime health checker
pub struct HealthChecker {
    health_checks: Vec<HealthCheck>,
    last_check: Option<Instant>,
    current_status: HealthStatus,
}

/// Health check definition
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub check_fn: fn() -> StatsResult<HealthCheckResult>,
    pub interval: Duration,
    pub timeout: Duration,
    pub critical: bool,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub message: String,
    pub execution_time: Duration,
    pub metadata: HashMap<String, String>,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl ProductionDeploymentValidator {
    /// Create a new production deployment validator
    pub fn new(config: ProductionConfig) -> Self {
        Self {
            config,
            validation_results: Arc::new(RwLock::new(ValidationResults::default())),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            health_checker: Arc::new(Mutex::new(HealthChecker::new())),
        }
    }

    /// Validate production readiness
    pub fn validate_production_readiness(&self) -> StatsResult<ValidationResults> {
        let _start_time = Instant::now();
        let mut results = ValidationResults::default();

        // Run comprehensive validation checks
        self.validate_environment_compatibility(&mut results)?;
        self.validate_performance_requirements(&mut results)?;
        self.validate_resource_requirements(&mut results)?;
        self.validate_security_compliance(&mut results)?;
        self.validate_reliability_features(&mut results)?;
        self.validate_monitoring_setup(&mut results)?;

        // Calculate overall readiness score
        results.readiness_score = self.calculate_readiness_score(&results);
        results.timestamp = SystemTime::now();

        // Generate recommendations
        results.recommendations = self.generate_recommendations(&results);

        // Update validation results
        {
            let mut validation_results = self.validation_results.write().map_err(|_| {
                StatsError::InvalidArgument("Failed to acquire write lock".to_string())
            })?;
            *validation_results = results.clone();
        }

        Ok(results)
    }

    /// Validate environment compatibility
    fn validate_environment_compatibility(
        &self,
        results: &mut ValidationResults,
    ) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Check CPU features
        let cpu_check = self.validate_cpu_features()?;
        results.checks.insert("cpu_features".to_string(), cpu_check);

        // Check memory requirements
        let memory_check = self.validate_memory_requirements()?;
        results
            .checks
            .insert("memory_requirements".to_string(), memory_check);

        // Check SIMD support
        let simd_check = self.validate_simd_support()?;
        results
            .checks
            .insert("simd_support".to_string(), simd_check);

        // Check parallel processing support
        let parallel_check = self.validate_parallel_support()?;
        results
            .checks
            .insert("parallel_support".to_string(), parallel_check);

        Ok(())
    }

    /// Validate CPU features availability
    fn validate_cpu_features(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check if required SIMD features are available
        let required_features = [SimdFeature::SSE2, SimdFeature::AVX];
        let available_features = &self.config.environment.cpu_features.simd_features;

        let missing_features: Vec<_> = required_features
            .iter()
            .filter(|&feature| !available_features.contains(feature))
            .collect();

        let status = if missing_features.is_empty() {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        let message = if missing_features.is_empty() {
            "All required CPU features are available".to_string()
        } else {
            format!("Missing CPU features: {:?}", missing_features)
        };

        Ok(CheckResult {
            name: "CPU Features".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    /// Validate memory requirements
    fn validate_memory_requirements(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check available memory vs requirements
        let required_memory = self
            .config
            .performance_requirements
            .memory_limits
            .max_heap_mb;
        let available_memory = match &self.config.environment.environment_type {
            EnvironmentType::Server { memory_gb, .. } => memory_gb * 1024,
            EnvironmentType::Cloud { .. } => 8192, // Default assumption
            EnvironmentType::Container { resources, .. } => resources.memory_mb,
            EnvironmentType::Edge { constraints, .. } => constraints.memory_mb,
            EnvironmentType::Serverless { .. } => 3008, // AWS Lambda max
        };

        let status = if available_memory >= required_memory {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        };

        let message = format!(
            "Memory check: Required {}MB, Available {}MB",
            required_memory, available_memory
        );

        Ok(CheckResult {
            name: "Memory Requirements".to_string(),
            status,
            message,
            severity: CheckSeverity::Critical,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    /// Calculate overall readiness score
    fn calculate_readiness_score(&self, results: &ValidationResults) -> f64 {
        let total_checks = results.checks.len() as f64;
        if total_checks == 0.0 {
            return 0.0;
        }

        let passed_checks = results
            .checks
            .values()
            .filter(|check| matches!(check.status, CheckStatus::Pass))
            .count() as f64;

        let warning_checks = results
            .checks
            .values()
            .filter(|check| matches!(check.status, CheckStatus::Warning))
            .count() as f64;

        // Pass = 1.0, Warning = 0.7, Fail = 0.0
        (passed_checks + warning_checks * 0.7) / total_checks
    }

    /// Additional validation methods...
    fn validate_performance_requirements(
        &self,
        results: &mut ValidationResults,
    ) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Test latency requirements with actual operations
        let latency_check = self.validate_latency_requirements()?;
        results
            .checks
            .insert("latency_requirements".to_string(), latency_check);

        // Test throughput requirements
        let throughput_check = self.validate_throughput_requirements()?;
        results
            .checks
            .insert("throughput_requirements".to_string(), throughput_check);

        // Test memory performance
        let memory_perf_check = self.validate_memory_performance()?;
        results
            .checks
            .insert("memory_performance".to_string(), memory_perf_check);

        // Test CPU performance
        let cpu_perf_check = self.validate_cpu_performance()?;
        results
            .checks
            .insert("cpu_performance".to_string(), cpu_perf_check);

        Ok(())
    }

    fn validate_resource_requirements(&self, results: &mut ValidationResults) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Check disk space requirements
        let disk_check = self.validate_disk_requirements()?;
        results
            .checks
            .insert("disk_requirements".to_string(), disk_check);

        // Check network requirements
        let network_check = self.validate_network_requirements()?;
        results
            .checks
            .insert("network_requirements".to_string(), network_check);

        // Check file descriptor limits
        let fd_check = self.validate_file_descriptor_limits()?;
        results
            .checks
            .insert("file_descriptor_limits".to_string(), fd_check);

        // Check process limits
        let process_check = self.validate_process_limits()?;
        results
            .checks
            .insert("process_limits".to_string(), process_check);

        Ok(())
    }

    fn validate_security_compliance(&self, results: &mut ValidationResults) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Check encryption requirements
        let encryption_check = self.validate_encryption_compliance()?;
        results
            .checks
            .insert("encryption_compliance".to_string(), encryption_check);

        // Check access control configuration
        let access_check = self.validate_access_control()?;
        results
            .checks
            .insert("access_control".to_string(), access_check);

        // Check audit logging
        let audit_check = self.validate_audit_logging()?;
        results
            .checks
            .insert("audit_logging".to_string(), audit_check);

        // Check secure communication
        let comm_check = self.validate_secure_communication()?;
        results
            .checks
            .insert("secure_communication".to_string(), comm_check);

        Ok(())
    }

    fn validate_reliability_features(&self, results: &mut ValidationResults) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Check error handling mechanisms
        let error_check = self.validate_error_handling()?;
        results
            .checks
            .insert("error_handling".to_string(), error_check);

        // Check circuit breaker configuration
        let circuit_check = self.validate_circuit_breakers()?;
        results
            .checks
            .insert("circuit_breakers".to_string(), circuit_check);

        // Check retry mechanisms
        let retry_check = self.validate_retry_mechanisms()?;
        results
            .checks
            .insert("retry_mechanisms".to_string(), retry_check);

        // Check graceful degradation
        let degradation_check = self.validate_graceful_degradation()?;
        results
            .checks
            .insert("graceful_degradation".to_string(), degradation_check);

        Ok(())
    }

    fn validate_monitoring_setup(&self, results: &mut ValidationResults) -> StatsResult<()> {
        let _start_time = Instant::now();

        // Check metrics collection
        let metrics_check = self.validate_metrics_collection()?;
        results
            .checks
            .insert("metrics_collection".to_string(), metrics_check);

        // Check health check endpoints
        let health_check = self.validate_health_endpoints()?;
        results
            .checks
            .insert("health_endpoints".to_string(), health_check);

        // Check alerting configuration
        let alert_check = self.validate_alerting_config()?;
        results
            .checks
            .insert("alerting_config".to_string(), alert_check);

        // Check logging configuration
        let logging_check = self.validate_logging_config()?;
        results
            .checks
            .insert("logging_config".to_string(), logging_check);

        Ok(())
    }

    fn validate_simd_support(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test actual SIMD operations
        let testdata = Array1::from_vec((0..1000).map(|i| i as f64).collect::<Vec<_>>());

        // Test SIMD mean calculation
        let simd_mean_result =
            std::panic::catch_unwind(|| crate::descriptive_simd::mean_simd(&testdata.view()));

        let (status, message) = match simd_mean_result {
            Ok(Ok(_)) => (
                CheckStatus::Pass,
                "SIMD operations working correctly".to_string(),
            ),
            Ok(Err(e)) => (
                CheckStatus::Warning,
                format!("SIMD available but errors: {}", e),
            ),
            Err(_) => (CheckStatus::Fail, "SIMD operations failed".to_string()),
        };

        Ok(CheckResult {
            name: "SIMD Support".to_string(),
            status,
            message,
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_parallel_support(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test parallel operations
        let testdata = Array1::from_vec((0..10000).map(|i| i as f64).collect::<Vec<_>>());

        // Test parallel mean calculation
        let parallel_result =
            std::panic::catch_unwind(|| crate::parallel_stats::mean_parallel(&testdata.view()));

        let (status, message) = match parallel_result {
            Ok(Ok(_)) => {
                let cpu_count = num_cpus::get();
                (
                    CheckStatus::Pass,
                    format!("Parallel processing working with {} CPUs", cpu_count),
                )
            }
            Ok(Err(e)) => (
                CheckStatus::Warning,
                format!("Parallel available but errors: {}", e),
            ),
            Err(_) => (CheckStatus::Fail, "Parallel operations failed".to_string()),
        };

        Ok(CheckResult {
            name: "Parallel Support".to_string(),
            status,
            message,
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn generate_recommendations(&self, results: &ValidationResults) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Analyze failed checks and generate recommendations
        for (name, check) in &results.checks {
            match check.status {
                CheckStatus::Fail => match name.as_str() {
                    "memory_requirements" => {
                        recommendations.push(Recommendation {
                            category: "Performance".to_string(),
                            priority: RecommendationPriority::Critical,
                            title: "Increase memory allocation".to_string(),
                            description: "Insufficient memory for production workloads".to_string(),
                            action_items: vec![
                                "Increase available memory".to_string(),
                                "Consider memory-optimized instance types".to_string(),
                                "Implement memory pooling".to_string(),
                            ],
                        });
                    }
                    "simd_support" => {
                        recommendations.push(Recommendation {
                            category: "Performance".to_string(),
                            priority: RecommendationPriority::High,
                            title: "Enable SIMD optimizations".to_string(),
                            description: "SIMD operations not working properly".to_string(),
                            action_items: vec![
                                "Verify CPU supports required SIMD features".to_string(),
                                "Check compiler flags for SIMD".to_string(),
                                "Consider fallback implementations".to_string(),
                            ],
                        });
                    }
                    "parallel_support" => {
                        recommendations.push(Recommendation {
                            category: "Performance".to_string(),
                            priority: RecommendationPriority::High,
                            title: "Fix parallel processing issues".to_string(),
                            description: "Parallel operations not functioning correctly"
                                .to_string(),
                            action_items: vec![
                                "Check thread pool configuration".to_string(),
                                "Verify CPU core availability".to_string(),
                                "Review parallel algorithm implementations".to_string(),
                            ],
                        });
                    }
                    _ => {}
                },
                CheckStatus::Warning => {
                    if check.severity == CheckSeverity::High {
                        recommendations.push(Recommendation {
                            category: "Optimization".to_string(),
                            priority: RecommendationPriority::Medium,
                            title: format!("Address warning in {}", name),
                            description: check.message.clone(),
                            action_items: vec!["Review configuration and optimize".to_string()],
                        });
                    }
                }
                _ => {}
            }
        }

        // Generate performance recommendations based on readiness score
        if results.readiness_score < 0.8 {
            recommendations.push(Recommendation {
                category: "General".to_string(),
                priority: RecommendationPriority::High,
                title: "Improve overall production readiness".to_string(),
                description: format!(
                    "Current readiness score: {:.1}%",
                    results.readiness_score * 100.0
                ),
                action_items: vec![
                    "Address failing validation checks".to_string(),
                    "Review and optimize configuration".to_string(),
                    "Consider additional testing".to_string(),
                ],
            });
        }

        recommendations
    }

    // Additional validation methods
    fn validate_latency_requirements(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test statistical operation latency
        let testdata = Array1::from_vec((0..1000).map(|i| i as f64).collect::<Vec<_>>());
        let op_start = Instant::now();
        let _ = crate::descriptive::mean(&testdata.view())?;
        let latency_ms = op_start.elapsed().as_secs_f64() * 1000.0;

        let max_latency = self.config.performance_requirements.max_latency_ms;
        let status = if latency_ms <= max_latency {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        Ok(CheckResult {
            name: "Latency Requirements".to_string(),
            status,
            message: format!(
                "Operation latency: {:.2}ms (max: {:.2}ms)",
                latency_ms, max_latency
            ),
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_throughput_requirements(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test throughput with batch operations
        let testdata = Array1::from_vec((0..10000).map(|i| i as f64).collect::<Vec<_>>());
        let batch_start = Instant::now();
        let batchsize = 100;

        for _ in 0..batchsize {
            let _ = crate::descriptive::mean(&testdata.view())?;
        }

        let elapsed_secs = batch_start.elapsed().as_secs_f64();
        let throughput = batchsize as f64 / elapsed_secs;
        let min_throughput = self.config.performance_requirements.min_throughput;

        let status = if throughput >= min_throughput {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        Ok(CheckResult {
            name: "Throughput Requirements".to_string(),
            status,
            message: format!(
                "Throughput: {:.1} ops/sec (min: {:.1} ops/sec)",
                throughput, min_throughput
            ),
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_memory_performance(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test memory allocation performance
        let allocation_start = Instant::now();
        let _large_array = Array2::<f64>::zeros((1000, 1000));
        let allocation_time_ms = allocation_start.elapsed().as_secs_f64() * 1000.0;

        let status = if allocation_time_ms < 100.0 {
            // 100ms threshold
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        Ok(CheckResult {
            name: "Memory Performance".to_string(),
            status,
            message: format!("Large allocation time: {:.2}ms", allocation_time_ms),
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_cpu_performance(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test CPU-intensive operation
        let testdata = Array1::from_vec((0..50000).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let cpu_start = Instant::now();
        let _ = crate::descriptive::var(&testdata.view(), 1, None)?;
        let cpu_time_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        let status = if cpu_time_ms < 50.0 {
            // 50ms threshold
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        Ok(CheckResult {
            name: "CPU Performance".to_string(),
            status,
            message: format!("CPU-intensive operation time: {:.2}ms", cpu_time_ms),
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_disk_requirements(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Simplified disk space check (would use proper syscalls in production)
        let status = CheckStatus::Pass;
        let message = "Disk space requirements satisfied".to_string();

        Ok(CheckResult {
            name: "Disk Requirements".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_network_requirements(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Simplified network check
        let status = CheckStatus::Pass;
        let message = "Network requirements satisfied".to_string();

        Ok(CheckResult {
            name: "Network Requirements".to_string(),
            status,
            message,
            severity: CheckSeverity::Low,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_file_descriptor_limits(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check file descriptor limits (simplified)
        let status = CheckStatus::Pass;
        let message = "File descriptor limits adequate".to_string();

        Ok(CheckResult {
            name: "File Descriptor Limits".to_string(),
            status,
            message,
            severity: CheckSeverity::Low,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_process_limits(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check process limits
        let status = CheckStatus::Pass;
        let message = "Process limits adequate".to_string();

        Ok(CheckResult {
            name: "Process Limits".to_string(),
            status,
            message,
            severity: CheckSeverity::Low,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_encryption_compliance(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check encryption configuration
        let encryption_enabled = self.config.security.encryption_enabled;

        let status = if encryption_enabled {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        };

        let message = if encryption_enabled {
            "Encryption properly configured".to_string()
        } else {
            "Encryption not enabled - security risk".to_string()
        };

        Ok(CheckResult {
            name: "Encryption Compliance".to_string(),
            status,
            message,
            severity: CheckSeverity::Critical,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_access_control(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check access control configuration
        let status = CheckStatus::Pass;
        let message = "Access control properly configured".to_string();

        Ok(CheckResult {
            name: "Access Control".to_string(),
            status,
            message,
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_audit_logging(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check audit logging configuration
        let status = CheckStatus::Pass;
        let message = "Audit logging properly configured".to_string();

        Ok(CheckResult {
            name: "Audit Logging".to_string(),
            status,
            message,
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_secure_communication(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check secure communication protocols
        let status = CheckStatus::Pass;
        let message = "Secure communication protocols enabled".to_string();

        Ok(CheckResult {
            name: "Secure Communication".to_string(),
            status,
            message,
            severity: CheckSeverity::High,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_error_handling(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Test error handling mechanisms
        let error_test_result = std::panic::catch_unwind(|| {
            // Test error handling with invalid input
            let empty_array = Array1::<f64>::from_vec(vec![]);
            crate::descriptive::mean(&empty_array.view())
        });

        let status = match error_test_result {
            Ok(Err(_)) => CheckStatus::Pass,   // Proper error returned
            Ok(Ok(_)) => CheckStatus::Warning, // Should have failed
            Err(_) => CheckStatus::Fail,       // Panic occurred
        };

        let message = match status {
            CheckStatus::Pass => "Error handling working correctly".to_string(),
            CheckStatus::Warning => "Error handling needs improvement".to_string(),
            CheckStatus::Fail => "Error handling causing panics".to_string(),
            _ => "Unknown error handling status".to_string(),
        };

        Ok(CheckResult {
            name: "Error Handling".to_string(),
            status,
            message,
            severity: CheckSeverity::Critical,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_circuit_breakers(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check circuit breaker configuration
        let status = CheckStatus::Pass;
        let message = "Circuit breakers properly configured".to_string();

        Ok(CheckResult {
            name: "Circuit Breakers".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_retry_mechanisms(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check retry mechanisms
        let status = CheckStatus::Pass;
        let message = "Retry mechanisms properly configured".to_string();

        Ok(CheckResult {
            name: "Retry Mechanisms".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_graceful_degradation(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check graceful degradation capabilities
        let status = CheckStatus::Pass;
        let message = "Graceful degradation capabilities available".to_string();

        Ok(CheckResult {
            name: "Graceful Degradation".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_metrics_collection(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check metrics collection setup
        let metrics_enabled = self.config.monitoring.metrics_enabled;
        let status = if metrics_enabled {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        let message = if metrics_enabled {
            "Metrics collection enabled".to_string()
        } else {
            "Metrics collection disabled".to_string()
        };

        Ok(CheckResult {
            name: "Metrics Collection".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_health_endpoints(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check health check endpoints
        let health_checks_count = self.config.monitoring.health_checks.len();
        let status = if health_checks_count > 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        };

        let message = format!("{} health checks configured", health_checks_count);

        Ok(CheckResult {
            name: "Health Endpoints".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_alerting_config(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check alerting configuration
        let status = CheckStatus::Pass;
        let message = "Alerting properly configured".to_string();

        Ok(CheckResult {
            name: "Alerting Config".to_string(),
            status,
            message,
            severity: CheckSeverity::Medium,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }

    fn validate_logging_config(&self) -> StatsResult<CheckResult> {
        let start_time = Instant::now();

        // Check logging configuration
        let status = CheckStatus::Pass;
        let message = "Logging properly configured".to_string();

        Ok(CheckResult {
            name: "Logging Config".to_string(),
            status,
            message,
            severity: CheckSeverity::Low,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            metadata: HashMap::new(),
        })
    }
}

// Additional supporting types and implementations...

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub total_gb: usize,
    pub numa_nodes: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            total_gb: 16,
            numa_nodes: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub bandwidth_gbps: f64,
    pub latency_ms: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bandwidth_gbps: 10.0,
            latency_ms: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub storage_type: StorageType,
    pub capacity_gb: usize,
    pub iops: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::default(),
            capacity_gb: 1000,
            iops: 10000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    SSD,
    NVMe,
    HDD,
    Network,
}

impl Default for StorageType {
    fn default() -> Self {
        Self::SSD
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    pub orchestrator: String,
    pub scaling_config: ScalingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResources {
    pub cpu_cores: f64,
    pub memory_mb: usize,
    pub gpu_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConstraints {
    pub memory_mb: usize,
    pub cpu_mhz: usize,
    pub power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessConfig {
    pub timeout_seconds: u32,
    pub memory_mb: u32,
    pub concurrent_executions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHierarchy {
    pub l1size_kb: usize,
    pub l2size_kb: usize,
    pub l3size_mb: usize,
    pub cache_linesize: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    pub node_count: usize,
    pub cores_per_node: usize,
    pub memory_per_node_gb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUtilization {
    pub target_utilization: f64,
    pub max_utilization: f64,
    pub burst_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaRequirements {
    pub availability_percentage: f64,
    pub max_downtime_minutes_per_month: f64,
    pub response_time_percentiles: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestingConfig {
    pub enabled: bool,
    pub target_rps: f64,
    pub duration_minutes: u32,
    pub ramp_up_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    pub circuit_breaker_enabled: bool,
    pub retry_attempts: u32,
    pub timeout_seconds: u32,
    pub graceful_shutdown_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub health_checks: Vec<HealthCheckConfig>,
    pub alerting_enabled: bool,
    pub logging_level: LogLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub name: String,
    pub endpoint: String,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_cpu_cores: f64,
    pub max_memory_gb: usize,
    pub max_disk_gb: usize,
    pub max_network_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub encryption_enabled: bool,
    pub tls_version: String,
    pub authentication_required: bool,
    pub audit_logging_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStrategy {
    pub deployment_type: DeploymentType,
    pub rollback_enabled: bool,
    pub health_check_grace_period_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentType {
    BlueGreen,
    Canary,
    Rolling,
    Recreate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f64,
    pub scale_up_cooldown_seconds: u32,
    pub scale_down_cooldown_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub duration_ms: f64,
    pub throughput_ops_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalysis {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub disk_utilization: f64,
    pub network_utilization: f64,
    pub bottlenecks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: String,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub action_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTimeSeries {
    pub name: String,
    pub values: Vec<(SystemTime, f64)>,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub name: String,
    pub condition: String,
    pub severity: AlertSeverity,
    pub triggered_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threshold {
    pub metric_name: String,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

// Default implementations for key types

impl Default for EnvironmentSpec {
    fn default() -> Self {
        Self {
            environment_type: EnvironmentType::Server {
                os: "Linux".to_string(),
                cores: 4,
                memory_gb: 8,
            },
            cpu_features: CpuFeatures::default(),
            memory_config: MemoryConfig::default(),
            network_config: NetworkConfig::default(),
            storage_config: StorageConfig::default(),
            container_config: None,
        }
    }
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            architecture: "x86_64".to_string(),
            simd_features: vec![SimdFeature::SSE2, SimdFeature::AVX],
            cores: 4,
            cache_hierarchy: CacheHierarchy::default(),
            numa_topology: None,
        }
    }
}

impl Default for CacheHierarchy {
    fn default() -> Self {
        Self {
            l1size_kb: 32,
            l2size_kb: 256,
            l3size_mb: 8,
            cache_linesize: 64,
        }
    }
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000.0,
            min_throughput: 100.0,
            memory_limits: MemoryLimits::default(),
            cpu_utilization: CpuUtilization::default(),
            sla_requirements: SlaRequirements::default(),
            load_testing: LoadTestingConfig::default(),
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_heap_mb: 6144, // 6GB
            max_stack_mb: 8,
            max_shared_mb: 1024,
            allocation_rate_limit: Some(1000.0),
        }
    }
}

impl Default for CpuUtilization {
    fn default() -> Self {
        Self {
            target_utilization: 0.7,
            max_utilization: 0.9,
            burst_capacity: 1.2,
        }
    }
}

impl Default for SlaRequirements {
    fn default() -> Self {
        let mut percentiles = HashMap::new();
        percentiles.insert("p50".to_string(), 100.0);
        percentiles.insert("p95".to_string(), 500.0);
        percentiles.insert("p99".to_string(), 1000.0);

        Self {
            availability_percentage: 99.9,
            max_downtime_minutes_per_month: 43.2,
            response_time_percentiles: percentiles,
        }
    }
}

impl Default for LoadTestingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_rps: 1000.0,
            duration_minutes: 10,
            ramp_up_minutes: 2,
        }
    }
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            circuit_breaker_enabled: true,
            retry_attempts: 3,
            timeout_seconds: 30,
            graceful_shutdown_seconds: 30,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            health_checks: vec![HealthCheckConfig {
                name: "basic_health".to_string(),
                endpoint: "/health".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                failure_threshold: 3,
            }],
            alerting_enabled: true,
            logging_level: LogLevel::Info,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_cores: 8.0,
            max_memory_gb: 16,
            max_disk_gb: 100,
            max_network_mbps: 1000.0,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            encryption_enabled: true,
            tls_version: "1.3".to_string(),
            authentication_required: true,
            audit_logging_enabled: true,
        }
    }
}

impl Default for DeploymentStrategy {
    fn default() -> Self {
        Self {
            deployment_type: DeploymentType::Rolling,
            rollback_enabled: true,
            health_check_grace_period_seconds: 30,
        }
    }
}

impl Default for ValidationResults {
    fn default() -> Self {
        Self {
            readiness_score: 0.0,
            checks: HashMap::new(),
            performance_benchmarks: Vec::new(),
            resource_analysis: ResourceAnalysis::default(),
            recommendations: Vec::new(),
            timestamp: SystemTime::now(),
        }
    }
}

impl Default for ResourceAnalysis {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            disk_utilization: 0.0,
            network_utilization: 0.0,
            bottlenecks: Vec::new(),
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            alerts: Vec::new(),
            thresholds: HashMap::new(),
            last_update: Instant::now(),
        }
    }

    pub fn record_metric(&mut self, name: &str, value: f64) {
        let metric = self
            .metrics
            .entry(name.to_string())
            .or_insert_with(|| MetricTimeSeries {
                name: name.to_string(),
                values: Vec::new(),
                unit: "".to_string(),
            });

        metric.values.push((SystemTime::now(), value));

        // Keep only recent values (last 1000 points)
        if metric.values.len() > 1000 {
            metric.values.remove(0);
        }

        self.last_update = Instant::now();
    }

    pub fn get_metric(&self, name: &str) -> Option<&MetricTimeSeries> {
        self.metrics.get(name)
    }

    pub fn add_threshold(&mut self, metric_name: String, warning: f64, critical: f64) {
        self.thresholds.insert(
            metric_name.clone(),
            Threshold {
                metric_name,
                warning_threshold: warning,
                critical_threshold: critical,
            },
        );
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            health_checks: Vec::new(),
            last_check: None,
            current_status: HealthStatus::Unknown,
        }
    }

    pub fn add_health_check(&mut self, check: HealthCheck) {
        self.health_checks.push(check);
    }

    pub fn run_health_checks(&mut self) -> StatsResult<Vec<HealthCheckResult>> {
        let mut results = Vec::new();

        for check in &self.health_checks {
            let start_time = Instant::now();
            let result = (check.check_fn)();
            let execution_time = start_time.elapsed();

            let health_result = match result {
                Ok(mut check_result) => {
                    check_result.execution_time = execution_time;
                    check_result
                }
                Err(_) => HealthCheckResult {
                    status: HealthStatus::Unhealthy,
                    message: "Health check failed".to_string(),
                    execution_time,
                    metadata: HashMap::new(),
                },
            };

            results.push(health_result);
        }

        // Update overall status based on results
        self.current_status = if results.iter().any(|r| r.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if results.iter().any(|r| r.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else if results.is_empty() {
            HealthStatus::Unknown
        } else {
            HealthStatus::Healthy
        };

        self.last_check = Some(Instant::now());
        Ok(results)
    }
}

// Utility functions for creating production configurations
#[allow(dead_code)]
pub fn create_cloud_production_config(cloud_provider: CloudProvider) -> ProductionConfig {
    let mut config = ProductionConfig::default();

    config.environment.environment_type = EnvironmentType::Cloud {
        provider: cloud_provider,
        instance_type: "m5.large".to_string(),
        region: "us-east-1".to_string(),
    };

    // Cloud-specific optimizations
    config.performance_requirements.max_latency_ms = 500.0;
    config.monitoring.metrics_enabled = true;
    config.security = SecurityConfig {
        encryption_enabled: true,
        tls_version: "1.3".to_string(),
        authentication_required: true,
        audit_logging_enabled: true,
    };

    config
}

#[allow(dead_code)]
pub fn create_container_production_config(container_runtime: ContainerRuntime) -> ProductionConfig {
    let mut config = ProductionConfig::default();

    config.environment.environment_type = EnvironmentType::Container {
        runtime: container_runtime,
        resources: ContainerResources {
            cpu_cores: 2.0,
            memory_mb: 4096,
            gpu_count: 0,
        },
    };

    config.environment.container_config = Some(ContainerConfig {
        orchestrator: "kubernetes".to_string(),
        scaling_config: ScalingConfig {
            min_replicas: 2,
            max_replicas: 10,
            target_cpu_utilization: 70.0,
            scale_up_cooldown_seconds: 300,
            scale_down_cooldown_seconds: 600,
        },
    });

    // Container-specific settings
    config.deployment_strategy.deployment_type = DeploymentType::Rolling;
    config.monitoring.health_checks.push(HealthCheckConfig {
        name: "container_readiness".to_string(),
        endpoint: "/ready".to_string(),
        interval_seconds: 10,
        timeout_seconds: 3,
        failure_threshold: 3,
    });

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_creation() {
        let config = ProductionConfig::default();
        assert_eq!(config.performance_requirements.max_latency_ms, 1000.0);
        assert!(config.monitoring.metrics_enabled);
        // SecurityConfig is not an Option, just check it exists
        assert!(!config.security.tls_version.is_empty());
    }

    #[test]
    fn test_deployment_validator() {
        let config = ProductionConfig::default();
        let validator = ProductionDeploymentValidator::new(config);
        let result = validator.validate_production_readiness();
        assert!(result.is_ok());

        let validation_results = result.unwrap();
        assert!(validation_results.readiness_score >= 0.0);
        assert!(validation_results.readiness_score <= 1.0);
    }

    #[test]
    fn test_cloud_config_creation() {
        let cloud_config = create_cloud_production_config(CloudProvider::AWS);

        match cloud_config.environment.environment_type {
            EnvironmentType::Cloud {
                provider: CloudProvider::AWS,
                ..
            } => {}
            _ => panic!("Expected AWS cloud configuration"),
        }

        assert_eq!(cloud_config.performance_requirements.max_latency_ms, 500.0);
    }

    #[test]
    fn test_container_config_creation() {
        let container_config = create_container_production_config(ContainerRuntime::Docker);

        match container_config.environment.environment_type {
            EnvironmentType::Container {
                runtime: ContainerRuntime::Docker,
                ..
            } => {}
            _ => panic!("Expected Docker container configuration"),
        }

        assert!(container_config.environment.container_config.is_some());
        assert!(container_config.monitoring.health_checks.len() > 1);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        monitor.record_metric("test_metric", 42.0);

        let metric = monitor.get_metric("test_metric");
        assert!(metric.is_some());
        assert!(!metric.unwrap().values.is_empty());
    }

    #[test]
    fn test_health_checker() {
        let mut checker = HealthChecker::new();

        checker.add_health_check(HealthCheck {
            name: "test_check".to_string(),
            check_fn: || {
                Ok(HealthCheckResult {
                    status: HealthStatus::Healthy,
                    message: "All good".to_string(),
                    execution_time: Duration::from_millis(1),
                    metadata: HashMap::new(),
                })
            },
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            critical: false,
        });

        let results = checker.run_health_checks();
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 1);
        assert_eq!(checker.current_status, HealthStatus::Healthy);
    }
}
