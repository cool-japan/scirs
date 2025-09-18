//! Core synchronization management for TPU pod coordination
//!
//! This module provides the refactored core synchronization management system,
//! breaking down the monolithic synchronization core into focused, maintainable modules.
//!
//! # Architecture
//!
//! The core synchronization system is organized into focused modules:
//!
//! - [`manager`]: Main synchronization manager and coordination
//! - [`config`]: Configuration structures and builders
//! - [`state`]: Global and device state management
//! - [`scheduler`]: Coordination scheduler and operation management
//! - [`monitoring`]: Performance monitoring and alerting
//! - [`optimization`]: Adaptive optimization and learning
//!
//! # Usage Example
//!
//! ```rust
//! use scirs2_optim::tpu::pod_coordination::synchronization::core::{
//!     SynchronizationManager, SynchronizationConfig,
//!     CoordinationScheduler, PerformanceMonitor,
//!     AdaptiveOptimizer
//! };
//!
//! // Create synchronization configuration
//! let config = SynchronizationConfig::default();
//!
//! // Initialize synchronization manager
//! let mut manager = SynchronizationManager::new(config)?;
//!
//! // Start synchronization
//! manager.start()?;
//!
//! // Perform global synchronization
//! manager.global_sync()?;
//! ```
//!
//! # Features
//!
//! ## Core Management
//! - Centralized synchronization coordination
//! - Device state management
//! - Health monitoring and recovery
//! - Configuration management
//!
//! ## Scheduling and Operations
//! - Priority-based operation scheduling
//! - Resource allocation and management
//! - Execution tracking and history
//! - Dependency management
//!
//! ## Performance Monitoring
//! - Real-time metrics collection
//! - Trend analysis and prediction
//! - Alerting and notifications
//! - Performance optimization
//!
//! ## Adaptive Optimization
//! - Machine learning-based parameter tuning
//! - Multi-objective optimization
//! - Active learning and exploration
//! - Meta-learning and transfer learning

pub mod manager;
pub mod config;
pub mod state;
pub mod scheduler;
pub mod monitoring;
pub mod optimization;

// Re-export core management types
pub use manager::{
    SynchronizationManager, SynchronizationManagerBuilder,
    SystemHealthStatus, HealthLevel, SyncSummary,
};

// Re-export configuration types
pub use config::{
    SynchronizationConfig, SynchronizationMode, SchedulerConfig, SchedulingAlgorithm,
    PrioritySettings, PriorityLevel, ResourceConfig, CPUAllocationConfig,
    MemoryAllocationConfig, NetworkAllocationConfig, ResourceLimits,
    MonitorConfig, MetricType, AlertConfig, AlertThresholds, NotificationConfig,
    OptimizerConfig, OptimizationObjective, LearningConfig, LearningAlgorithm,
    ConstraintConfig, RetrySettings, RetryCondition, BackoffStrategy,
};

// Re-export state management types
pub use state::{
    GlobalSynchronizationState, GlobalSyncStatus, GlobalQualityMetrics,
    DeviceSyncState, DeviceSyncStatus, DevicePerformanceMetrics,
    ResourceUtilization, GlobalBarrier, ResourcePool, AllocatedResources,
    ResourceUsageStatistics, UsageStatistics, PerformanceMetrics,
    LatencyMetrics, ThroughputMetrics, ErrorRateMetrics, SyncQualityMetrics,
    OptimizationState, ConvergenceStatus, SyncHealthSnapshot, HealthIssue,
    IssueSeverity, PerformanceIndicators, TrendDirection,
};

// Re-export scheduler types
pub use scheduler::{
    CoordinationScheduler, OperationId, ScheduledOperation, OperationType,
    OperationParameters, OperationStatus, QueuedOperation, ExecutionRecord,
    OperationResult, ExecutionMetrics, ResourceManager, ResourceRequirements,
    SchedulerState, SchedulerStatus, SchedulerStatistics,
};

// Re-export monitoring types
pub use monitoring::{
    PerformanceMonitor, MonitorState, MonitorStatus, DataCollector,
    CollectorType, CollectedData, MetricValue, MonitoringHistory,
    HistoricalMetric, MonitoringEvent, EventType, SeverityLevel,
    TrendAnalysis, Trend, PredictionModel, ModelType, Prediction,
    AnomalyResult, AnomalyType, AlertSystem, AlertId, Alert, AlertType,
    AlertRule, AlertCondition, NotificationSystem, NotificationRecord,
    NotificationStatus, AlertSummary,
};

// Re-export optimization types
pub use optimization::{
    AdaptiveOptimizer, ParameterSpace, ParameterDefinition, ParameterType,
    ObjectiveEvaluator, EvaluationMetric, ObjectiveEvaluation, OptimizationStrategy,
    StrategyType, StrategyContext, OptimizationHistory, OptimizationResult,
    LearningSystem, Model, TrainingData, ModelPerformance, ActiveLearning,
    MetaLearning, StrategyRecommendation,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;
use crate::error::{Result, OptimError};

/// Core synchronization result type
pub type CoreSyncResult<T> = Result<T>;

/// Core synchronization error types
#[derive(Debug, Clone)]
pub enum CoreSyncError {
    /// Manager error
    Manager(String),
    /// Configuration error
    Configuration(String),
    /// State error
    State(String),
    /// Scheduler error
    Scheduler(String),
    /// Monitoring error
    Monitoring(String),
    /// Optimization error
    Optimization(String),
    /// Resource error
    Resource(String),
    /// Network error
    Network(String),
    /// Timeout error
    Timeout,
}

impl std::fmt::Display for CoreSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreSyncError::Manager(msg) => write!(f, "Manager error: {}", msg),
            CoreSyncError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            CoreSyncError::State(msg) => write!(f, "State error: {}", msg),
            CoreSyncError::Scheduler(msg) => write!(f, "Scheduler error: {}", msg),
            CoreSyncError::Monitoring(msg) => write!(f, "Monitoring error: {}", msg),
            CoreSyncError::Optimization(msg) => write!(f, "Optimization error: {}", msg),
            CoreSyncError::Resource(msg) => write!(f, "Resource error: {}", msg),
            CoreSyncError::Network(msg) => write!(f, "Network error: {}", msg),
            CoreSyncError::Timeout => write!(f, "Timeout error"),
        }
    }
}

impl std::error::Error for CoreSyncError {}

/// Unified core synchronization facade
#[derive(Debug)]
pub struct UnifiedCoreSynchronization {
    /// Synchronization manager
    manager: SynchronizationManager,
    /// Performance monitor
    monitor: PerformanceMonitor,
    /// Adaptive optimizer
    optimizer: AdaptiveOptimizer,
    /// System statistics
    statistics: Arc<Mutex<UnifiedCoreStatistics>>,
    /// Last update time
    last_update: Arc<Mutex<Instant>>,
}

/// Unified core statistics
#[derive(Debug, Clone, Default)]
pub struct UnifiedCoreStatistics {
    /// Total operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average operation latency
    pub average_latency: Duration,
    /// Operations per second
    pub operations_per_second: f64,
    /// System health score
    pub health_score: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Optimization score
    pub optimization_score: f64,
}

impl UnifiedCoreSynchronization {
    /// Create new unified core synchronization
    pub fn new(config: SynchronizationConfig) -> CoreSyncResult<Self> {
        let manager = SynchronizationManager::new(config)
            .map_err(|e| CoreSyncError::Manager(e.to_string()))?;

        let monitor = PerformanceMonitor::new()
            .map_err(|e| CoreSyncError::Monitoring(e.to_string()))?;

        let optimizer = AdaptiveOptimizer::new()
            .map_err(|e| CoreSyncError::Optimization(e.to_string()))?;

        Ok(Self {
            manager,
            monitor,
            optimizer,
            statistics: Arc::new(Mutex::new(UnifiedCoreStatistics::default())),
            last_update: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// Initialize all core components
    pub fn initialize(&mut self) -> CoreSyncResult<()> {
        self.manager.start()
            .map_err(|e| CoreSyncError::Manager(e.to_string()))?;

        self.monitor.start()
            .map_err(|e| CoreSyncError::Monitoring(e.to_string()))?;

        self.optimizer.start()
            .map_err(|e| CoreSyncError::Optimization(e.to_string()))?;

        Ok(())
    }

    /// Shutdown all core components
    pub fn shutdown(&mut self) -> CoreSyncResult<()> {
        self.optimizer.stop()
            .map_err(|e| CoreSyncError::Optimization(e.to_string()))?;

        self.monitor.stop()
            .map_err(|e| CoreSyncError::Monitoring(e.to_string()))?;

        self.manager.stop()
            .map_err(|e| CoreSyncError::Manager(e.to_string()))?;

        Ok(())
    }

    /// Perform global synchronization
    pub fn global_sync(&mut self) -> CoreSyncResult<()> {
        self.manager.global_sync()
            .map_err(|e| CoreSyncError::Manager(e.to_string()))?;

        self.update_statistics()?;

        Ok(())
    }

    /// Add device to synchronization
    pub fn add_device(&mut self, device_id: DeviceId) -> CoreSyncResult<()> {
        self.manager.add_device(device_id)
            .map_err(|e| CoreSyncError::Manager(e.to_string()))
    }

    /// Remove device from synchronization
    pub fn remove_device(&mut self, device_id: DeviceId) -> CoreSyncResult<()> {
        self.manager.remove_device(device_id)
            .map_err(|e| CoreSyncError::Manager(e.to_string()))
    }

    /// Get system health status
    pub fn get_health_status(&self) -> SystemHealthStatus {
        self.manager.check_system_health()
    }

    /// Get unified statistics
    pub fn get_statistics(&self) -> UnifiedCoreStatistics {
        let stats = self.statistics.lock().unwrap();
        stats.clone()
    }

    /// Update statistics
    fn update_statistics(&self) -> CoreSyncResult<()> {
        let mut stats = self.statistics.lock().unwrap();

        // Get manager statistics
        let manager_stats = self.manager.get_statistics();
        stats.total_operations = manager_stats.get("total_operations").copied().unwrap_or(0.0) as u64;

        // Get performance metrics
        let perf_metrics = self.monitor.get_metrics();
        stats.average_latency = perf_metrics.latency.average;
        stats.operations_per_second = perf_metrics.throughput.ops_per_second;

        // Calculate health score
        let health_status = self.manager.check_system_health();
        stats.health_score = match health_status.overall_health {
            HealthLevel::Excellent => 1.0,
            HealthLevel::Good => 0.8,
            HealthLevel::Warning => 0.6,
            HealthLevel::Critical => 0.3,
        };

        // Update timestamp
        let mut last_update = self.last_update.lock().unwrap();
        *last_update = Instant::now();

        Ok(())
    }

    /// Trigger optimization
    pub fn trigger_optimization(&mut self) -> CoreSyncResult<()> {
        self.optimizer.optimize()
            .map_err(|e| CoreSyncError::Optimization(e.to_string()))
    }

    /// Collect metrics
    pub fn collect_metrics(&mut self) -> CoreSyncResult<()> {
        self.monitor.collect_metrics()
            .map_err(|e| CoreSyncError::Monitoring(e.to_string()))
    }

    /// Get alert summary
    pub fn get_alert_summary(&self) -> AlertSummary {
        self.monitor.get_alert_summary()
    }

    /// Schedule operation
    pub fn schedule_operation(&mut self, operation_type: OperationType, devices: Vec<DeviceId>) -> CoreSyncResult<OperationId> {
        let params = OperationParameters {
            timeout: Duration::from_secs(60),
            priority: 5,
            retry_settings: RetrySettings {
                max_attempts: 3,
                interval: Duration::from_secs(5),
                backoff: BackoffStrategy::Fixed,
                conditions: vec![RetryCondition::OnTimeout],
            },
            custom_params: HashMap::new(),
            resource_requirements: None,
            quality_requirements: None,
        };

        self.manager.schedule_operation(operation_type, devices, params)
            .map_err(|e| CoreSyncError::Scheduler(e.to_string()))
    }

    /// Force synchronization of specific devices
    pub fn force_sync_devices(&mut self, device_ids: &[DeviceId]) -> CoreSyncResult<()> {
        self.manager.force_sync_devices(device_ids)
            .map_err(|e| CoreSyncError::Manager(e.to_string()))
    }

    /// Create global barrier
    pub fn create_global_barrier(&mut self, barrier_id: String, participants: Vec<DeviceId>, timeout: Duration) -> CoreSyncResult<()> {
        let participants_set = participants.into_iter().collect();
        self.manager.create_global_barrier(barrier_id, participants_set, timeout)
            .map_err(|e| CoreSyncError::Manager(e.to_string()))
    }

    /// Get current epoch
    pub fn get_current_epoch(&self) -> u64 {
        self.manager.get_current_epoch()
    }

    /// Check if device is synchronized
    pub fn is_device_synchronized(&self, device_id: DeviceId) -> bool {
        self.manager.is_device_synchronized(device_id)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SynchronizationConfig) -> CoreSyncResult<()> {
        self.manager.update_config(config)
            .map_err(|e| CoreSyncError::Configuration(e.to_string()))
    }
}

/// Core synchronization builder
#[derive(Debug, Default)]
pub struct CoreSynchronizationBuilder {
    config: Option<SynchronizationConfig>,
    enable_monitoring: Option<bool>,
    enable_optimization: Option<bool>,
    monitoring_interval: Option<Duration>,
    optimization_frequency: Option<Duration>,
}

impl CoreSynchronizationBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set configuration
    pub fn config(mut self, config: SynchronizationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable monitoring
    pub fn enable_monitoring(mut self, enable: bool) -> Self {
        self.enable_monitoring = Some(enable);
        self
    }

    /// Enable optimization
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.enable_optimization = Some(enable);
        self
    }

    /// Set monitoring interval
    pub fn monitoring_interval(mut self, interval: Duration) -> Self {
        self.monitoring_interval = Some(interval);
        self
    }

    /// Set optimization frequency
    pub fn optimization_frequency(mut self, frequency: Duration) -> Self {
        self.optimization_frequency = Some(frequency);
        self
    }

    /// Build core synchronization
    pub fn build(self) -> CoreSyncResult<UnifiedCoreSynchronization> {
        let config = self.config.unwrap_or_default();
        let mut core = UnifiedCoreSynchronization::new(config)?;

        // Apply optional configurations
        if let Some(false) = self.enable_monitoring {
            // Monitoring would be disabled here
        }

        if let Some(false) = self.enable_optimization {
            // Optimization would be disabled here
        }

        core.initialize()?;

        Ok(core)
    }
}

/// Core synchronization utilities
pub mod utils {
    use super::*;

    /// Create default core synchronization
    pub fn create_default_core() -> CoreSyncResult<UnifiedCoreSynchronization> {
        CoreSynchronizationBuilder::new().build()
    }

    /// Create test core synchronization
    pub fn create_test_core() -> CoreSyncResult<UnifiedCoreSynchronization> {
        let config = SynchronizationConfig {
            sync_mode: SynchronizationMode::BulkSynchronous,
            global_timeout: Duration::from_secs(10),
            ..Default::default()
        };

        CoreSynchronizationBuilder::new()
            .config(config)
            .monitoring_interval(Duration::from_millis(100))
            .optimization_frequency(Duration::from_secs(5))
            .build()
    }

    /// Calculate overall system efficiency
    pub fn calculate_system_efficiency(core: &UnifiedCoreSynchronization) -> f64 {
        let stats = core.get_statistics();
        let health = core.get_health_status();

        let efficiency_factors = vec![
            stats.health_score,
            stats.optimization_score,
            match health.overall_health {
                HealthLevel::Excellent => 1.0,
                HealthLevel::Good => 0.8,
                HealthLevel::Warning => 0.6,
                HealthLevel::Critical => 0.3,
            },
        ];

        efficiency_factors.iter().sum::<f64>() / efficiency_factors.len() as f64
    }

    /// Get system status summary
    pub fn get_system_summary(core: &UnifiedCoreSynchronization) -> SystemSummary {
        let stats = core.get_statistics();
        let health = core.get_health_status();
        let alerts = core.get_alert_summary();

        SystemSummary {
            total_operations: stats.total_operations,
            health_score: stats.health_score,
            active_alerts: alerts.total_active,
            critical_alerts: alerts.critical_alerts,
            avg_latency: stats.average_latency,
            operations_per_second: stats.operations_per_second,
            system_efficiency: calculate_system_efficiency(core),
        }
    }

    /// Validate core configuration
    pub fn validate_core_config(config: &SynchronizationConfig) -> CoreSyncResult<()> {
        if config.global_timeout.as_secs() == 0 {
            return Err(CoreSyncError::Configuration("Global timeout cannot be zero".to_string()));
        }

        Ok(())
    }
}

/// System summary
#[derive(Debug, Clone)]
pub struct SystemSummary {
    /// Total operations performed
    pub total_operations: u64,
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Number of critical alerts
    pub critical_alerts: usize,
    /// Average operation latency
    pub avg_latency: Duration,
    /// Operations per second
    pub operations_per_second: f64,
    /// Overall system efficiency (0.0 to 1.0)
    pub system_efficiency: f64,
}

/// Core synchronization testing utilities
#[cfg(test)]
pub mod testing {
    use super::*;

    /// Create minimal test core
    pub fn create_minimal_test_core() -> CoreSyncResult<UnifiedCoreSynchronization> {
        let config = SynchronizationConfig {
            sync_mode: SynchronizationMode::BulkSynchronous,
            global_timeout: Duration::from_millis(100),
            ..Default::default()
        };

        CoreSynchronizationBuilder::new()
            .config(config)
            .enable_monitoring(false)
            .enable_optimization(false)
            .build()
    }

    /// Add test devices to core
    pub fn add_test_devices(core: &mut UnifiedCoreSynchronization, count: usize) -> CoreSyncResult<Vec<DeviceId>> {
        let mut device_ids = Vec::new();
        for i in 0..count {
            let device_id = DeviceId::from(i as u32);
            core.add_device(device_id)?;
            device_ids.push(device_id);
        }
        Ok(device_ids)
    }

    /// Simulate synchronization workload
    pub fn simulate_workload(core: &mut UnifiedCoreSynchronization, operations: usize) -> CoreSyncResult<()> {
        for i in 0..operations {
            if i % 10 == 0 {
                core.global_sync()?;
            } else {
                core.collect_metrics()?;
            }
        }
        Ok(())
    }

    /// Create test barrier
    pub fn create_test_barrier(core: &mut UnifiedCoreSynchronization, device_ids: &[DeviceId]) -> CoreSyncResult<String> {
        let barrier_id = format!("test_barrier_{}", rand::random::<u32>());
        core.create_global_barrier(barrier_id.clone(), device_ids.to_vec(), Duration::from_secs(5))?;
        Ok(barrier_id)
    }

    /// Assert system health
    pub fn assert_system_healthy(core: &UnifiedCoreSynchronization) {
        let health = core.get_health_status();
        assert!(matches!(health.overall_health, HealthLevel::Good | HealthLevel::Excellent));
    }

    /// Assert no critical alerts
    pub fn assert_no_critical_alerts(core: &UnifiedCoreSynchronization) {
        let alerts = core.get_alert_summary();
        assert_eq!(alerts.critical_alerts, 0);
    }
}

/// Integration with other synchronization components
pub mod integration {
    use super::*;

    /// Integrate with barrier synchronization
    pub fn integrate_with_barriers(core: &mut UnifiedCoreSynchronization) -> CoreSyncResult<()> {
        // Integration logic would go here
        Ok(())
    }

    /// Integrate with event synchronization
    pub fn integrate_with_events(core: &mut UnifiedCoreSynchronization) -> CoreSyncResult<()> {
        // Integration logic would go here
        Ok(())
    }

    /// Integrate with clock synchronization
    pub fn integrate_with_clocks(core: &mut UnifiedCoreSynchronization) -> CoreSyncResult<()> {
        // Integration logic would go here
        Ok(())
    }

    /// Integrate with consensus protocols
    pub fn integrate_with_consensus(core: &mut UnifiedCoreSynchronization) -> CoreSyncResult<()> {
        // Integration logic would go here
        Ok(())
    }
}

/// Performance analysis tools
pub mod analysis {
    use super::*;

    /// Analyze system performance
    pub fn analyze_performance(core: &UnifiedCoreSynchronization) -> PerformanceAnalysis {
        let stats = core.get_statistics();
        let health = core.get_health_status();

        PerformanceAnalysis {
            overall_score: stats.health_score,
            latency_score: calculate_latency_score(stats.average_latency),
            throughput_score: calculate_throughput_score(stats.operations_per_second),
            efficiency_score: stats.optimization_score,
            health_issues: health.issues.len(),
            recommendations: generate_recommendations(&stats, &health),
        }
    }

    /// Calculate latency performance score
    fn calculate_latency_score(latency: Duration) -> f64 {
        let latency_ms = latency.as_millis() as f64;
        if latency_ms < 10.0 {
            1.0
        } else if latency_ms < 50.0 {
            0.8
        } else if latency_ms < 100.0 {
            0.6
        } else {
            0.4
        }
    }

    /// Calculate throughput performance score
    fn calculate_throughput_score(ops_per_sec: f64) -> f64 {
        if ops_per_sec > 1000.0 {
            1.0
        } else if ops_per_sec > 500.0 {
            0.8
        } else if ops_per_sec > 100.0 {
            0.6
        } else {
            0.4
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(stats: &UnifiedCoreStatistics, health: &SystemHealthStatus) -> Vec<String> {
        let mut recommendations = Vec::new();

        if stats.health_score < 0.7 {
            recommendations.push("Consider reviewing system configuration".to_string());
        }

        if stats.average_latency > Duration::from_millis(100) {
            recommendations.push("High latency detected - optimize synchronization parameters".to_string());
        }

        if health.issues.len() > 5 {
            recommendations.push("Multiple issues detected - perform comprehensive health check".to_string());
        }

        recommendations
    }
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Overall performance score (0.0 to 1.0)
    pub overall_score: f64,
    /// Latency performance score
    pub latency_score: f64,
    /// Throughput performance score
    pub throughput_score: f64,
    /// Efficiency score
    pub efficiency_score: f64,
    /// Number of health issues
    pub health_issues: usize,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}