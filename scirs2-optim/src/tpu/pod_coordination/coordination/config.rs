//! Configuration Management for TPU Pod Coordination
//!
//! This module provides comprehensive configuration structures, validation,
//! and builders for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive configuration for TPU pod coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodCoordinationConfig {
    /// Number of TPU devices in the pod
    pub device_count: usize,
    /// Coordination strategy to use
    pub coordination_strategy: CoordinationStrategy,
    /// Communication pattern for inter-device communication
    pub communication_pattern: CommunicationPattern,
    /// Synchronization mode for coordination
    pub synchronization_mode: SynchronizationMode,
    /// Batch parallelization strategy
    pub batch_parallelization: BatchParallelizationStrategy,
    /// Gradient aggregation method
    pub gradient_aggregation: GradientAggregationMethod,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Memory management strategy
    pub memory_management: MemoryManagementStrategy,
    /// Maximum coordination timeout in seconds
    pub coordination_timeout: u64,
    /// Performance monitoring interval in milliseconds
    pub monitoring_interval: u64,
    /// Enable fault tolerance mechanisms
    pub enable_fault_tolerance: bool,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Quality of Service requirements
    pub qos_requirements: QoSRequirements,
}

/// Configuration builder for pod coordination
#[derive(Debug, Default)]
pub struct PodCoordinationConfigBuilder {
    device_count: Option<usize>,
    coordination_strategy: Option<CoordinationStrategy>,
    communication_pattern: Option<CommunicationPattern>,
    synchronization_mode: Option<SynchronizationMode>,
    batch_parallelization: Option<BatchParallelizationStrategy>,
    gradient_aggregation: Option<GradientAggregationMethod>,
    load_balancing: Option<LoadBalancingStrategy>,
    memory_management: Option<MemoryManagementStrategy>,
    coordination_timeout: Option<u64>,
    monitoring_interval: Option<u64>,
    enable_fault_tolerance: Option<bool>,
    enable_adaptive_optimization: Option<bool>,
    qos_requirements: Option<QoSRequirements>,
}

impl PodCoordinationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set device count
    pub fn device_count(mut self, count: usize) -> Self {
        self.device_count = Some(count);
        self
    }

    /// Set coordination strategy
    pub fn coordination_strategy(mut self, strategy: CoordinationStrategy) -> Self {
        self.coordination_strategy = Some(strategy);
        self
    }

    /// Set communication pattern
    pub fn communication_pattern(mut self, pattern: CommunicationPattern) -> Self {
        self.communication_pattern = Some(pattern);
        self
    }

    /// Set synchronization mode
    pub fn synchronization_mode(mut self, mode: SynchronizationMode) -> Self {
        self.synchronization_mode = Some(mode);
        self
    }

    /// Set batch parallelization strategy
    pub fn batch_parallelization(mut self, strategy: BatchParallelizationStrategy) -> Self {
        self.batch_parallelization = Some(strategy);
        self
    }

    /// Set gradient aggregation method
    pub fn gradient_aggregation(mut self, method: GradientAggregationMethod) -> Self {
        self.gradient_aggregation = Some(method);
        self
    }

    /// Set load balancing strategy
    pub fn load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.load_balancing = Some(strategy);
        self
    }

    /// Set memory management strategy
    pub fn memory_management(mut self, strategy: MemoryManagementStrategy) -> Self {
        self.memory_management = Some(strategy);
        self
    }

    /// Set coordination timeout
    pub fn coordination_timeout(mut self, timeout: u64) -> Self {
        self.coordination_timeout = Some(timeout);
        self
    }

    /// Set monitoring interval
    pub fn monitoring_interval(mut self, interval: u64) -> Self {
        self.monitoring_interval = Some(interval);
        self
    }

    /// Enable fault tolerance
    pub fn enable_fault_tolerance(mut self, enable: bool) -> Self {
        self.enable_fault_tolerance = Some(enable);
        self
    }

    /// Enable adaptive optimization
    pub fn enable_adaptive_optimization(mut self, enable: bool) -> Self {
        self.enable_adaptive_optimization = Some(enable);
        self
    }

    /// Set QoS requirements
    pub fn qos_requirements(mut self, qos: QoSRequirements) -> Self {
        self.qos_requirements = Some(qos);
        self
    }

    /// Build the configuration
    pub fn build(self) -> PodCoordinationConfig {
        PodCoordinationConfig {
            device_count: self.device_count.unwrap_or(8),
            coordination_strategy: self.coordination_strategy.unwrap_or(CoordinationStrategy::Adaptive),
            communication_pattern: self.communication_pattern.unwrap_or(CommunicationPattern::AllToAll),
            synchronization_mode: self.synchronization_mode.unwrap_or(SynchronizationMode::BulkSynchronous),
            batch_parallelization: self.batch_parallelization.unwrap_or(BatchParallelizationStrategy::DataParallel),
            gradient_aggregation: self.gradient_aggregation.unwrap_or(GradientAggregationMethod::AllReduce),
            load_balancing: self.load_balancing.unwrap_or(LoadBalancingStrategy::Adaptive),
            memory_management: self.memory_management.unwrap_or(MemoryManagementStrategy::Dynamic),
            coordination_timeout: self.coordination_timeout.unwrap_or(30),
            monitoring_interval: self.monitoring_interval.unwrap_or(1000),
            enable_fault_tolerance: self.enable_fault_tolerance.unwrap_or(true),
            enable_adaptive_optimization: self.enable_adaptive_optimization.unwrap_or(true),
            qos_requirements: self.qos_requirements.unwrap_or_default(),
        }
    }
}

/// Coordination strategies for pod management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Centralized coordination with master node
    Centralized,
    /// Decentralized coordination with peer-to-peer communication
    Decentralized,
    /// Hierarchical coordination with multiple levels
    Hierarchical,
    /// Adaptive coordination that switches based on workload
    Adaptive,
}

impl CoordinationStrategy {
    /// Get strategy description
    pub fn description(&self) -> &'static str {
        match self {
            CoordinationStrategy::Centralized => "Centralized coordination with master node",
            CoordinationStrategy::Decentralized => "Decentralized peer-to-peer coordination",
            CoordinationStrategy::Hierarchical => "Hierarchical multi-level coordination",
            CoordinationStrategy::Adaptive => "Adaptive strategy switching",
        }
    }

    /// Check if strategy supports fault tolerance
    pub fn supports_fault_tolerance(&self) -> bool {
        match self {
            CoordinationStrategy::Centralized => false,
            CoordinationStrategy::Decentralized => true,
            CoordinationStrategy::Hierarchical => true,
            CoordinationStrategy::Adaptive => true,
        }
    }
}

/// Communication patterns for inter-device communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationPattern {
    /// All-to-all communication pattern
    AllToAll,
    /// Ring communication pattern
    Ring,
    /// Tree communication pattern
    Tree,
    /// Mesh communication pattern
    Mesh,
    /// Butterfly communication pattern
    Butterfly,
    /// Hypercube communication pattern
    Hypercube,
    /// Custom pattern defined by user
    Custom(String),
}

impl CommunicationPattern {
    /// Get pattern description
    pub fn description(&self) -> String {
        match self {
            CommunicationPattern::AllToAll => "All-to-all communication".to_string(),
            CommunicationPattern::Ring => "Ring topology communication".to_string(),
            CommunicationPattern::Tree => "Tree topology communication".to_string(),
            CommunicationPattern::Mesh => "Mesh topology communication".to_string(),
            CommunicationPattern::Butterfly => "Butterfly topology communication".to_string(),
            CommunicationPattern::Hypercube => "Hypercube topology communication".to_string(),
            CommunicationPattern::Custom(name) => format!("Custom pattern: {}", name),
        }
    }

    /// Get communication complexity
    pub fn complexity(&self) -> CommunicationComplexity {
        match self {
            CommunicationPattern::AllToAll => CommunicationComplexity::High,
            CommunicationPattern::Ring => CommunicationComplexity::Low,
            CommunicationPattern::Tree => CommunicationComplexity::Medium,
            CommunicationPattern::Mesh => CommunicationComplexity::High,
            CommunicationPattern::Butterfly => CommunicationComplexity::Medium,
            CommunicationPattern::Hypercube => CommunicationComplexity::Medium,
            CommunicationPattern::Custom(_) => CommunicationComplexity::Unknown,
        }
    }
}

/// Communication complexity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunicationComplexity {
    Low,
    Medium,
    High,
    Unknown,
}

/// Synchronization modes for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationMode {
    /// Bulk synchronous parallel model
    BulkSynchronous,
    /// Asynchronous coordination
    Asynchronous,
    /// Bounded asynchronous with staleness bounds
    BoundedAsynchronous { staleness_bound: usize },
    /// Event-driven synchronization
    EventDriven,
}

impl SynchronizationMode {
    /// Get mode description
    pub fn description(&self) -> String {
        match self {
            SynchronizationMode::BulkSynchronous => "Bulk synchronous parallel model".to_string(),
            SynchronizationMode::Asynchronous => "Asynchronous coordination".to_string(),
            SynchronizationMode::BoundedAsynchronous { staleness_bound } => {
                format!("Bounded asynchronous (staleness: {})", staleness_bound)
            },
            SynchronizationMode::EventDriven => "Event-driven synchronization".to_string(),
        }
    }

    /// Check if mode requires global synchronization
    pub fn requires_global_sync(&self) -> bool {
        match self {
            SynchronizationMode::BulkSynchronous => true,
            SynchronizationMode::Asynchronous => false,
            SynchronizationMode::BoundedAsynchronous { .. } => false,
            SynchronizationMode::EventDriven => false,
        }
    }
}

/// Batch parallelization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchParallelizationStrategy {
    /// Data parallelism across devices
    DataParallel,
    /// Model parallelism across devices
    ModelParallel,
    /// Pipeline parallelism with staged execution
    PipelineParallel { stages: usize },
    /// Hybrid parallelism combining multiple strategies
    Hybrid {
        data_parallel_factor: usize,
        model_parallel_factor: usize,
    },
}

impl BatchParallelizationStrategy {
    /// Get strategy description
    pub fn description(&self) -> String {
        match self {
            BatchParallelizationStrategy::DataParallel => "Data parallelism".to_string(),
            BatchParallelizationStrategy::ModelParallel => "Model parallelism".to_string(),
            BatchParallelizationStrategy::PipelineParallel { stages } => {
                format!("Pipeline parallelism ({} stages)", stages)
            },
            BatchParallelizationStrategy::Hybrid { data_parallel_factor, model_parallel_factor } => {
                format!("Hybrid parallelism (data: {}, model: {})", data_parallel_factor, model_parallel_factor)
            },
        }
    }

    /// Get scaling efficiency
    pub fn scaling_efficiency(&self, device_count: usize) -> f64 {
        match self {
            BatchParallelizationStrategy::DataParallel => 0.9,
            BatchParallelizationStrategy::ModelParallel => 0.7,
            BatchParallelizationStrategy::PipelineParallel { stages } => {
                if device_count >= *stages {
                    0.85
                } else {
                    0.6
                }
            },
            BatchParallelizationStrategy::Hybrid { .. } => 0.8,
        }
    }
}

/// Gradient aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientAggregationMethod {
    /// Simple averaging of gradients
    Average,
    /// Weighted averaging based on batch sizes
    WeightedAverage,
    /// All-reduce aggregation
    AllReduce,
    /// Parameter server aggregation
    ParameterServer,
    /// Hierarchical aggregation
    Hierarchical,
    /// Compression-based aggregation
    Compressed { compression_ratio: f64 },
}

impl GradientAggregationMethod {
    /// Get method description
    pub fn description(&self) -> String {
        match self {
            GradientAggregationMethod::Average => "Simple gradient averaging".to_string(),
            GradientAggregationMethod::WeightedAverage => "Weighted gradient averaging".to_string(),
            GradientAggregationMethod::AllReduce => "All-reduce aggregation".to_string(),
            GradientAggregationMethod::ParameterServer => "Parameter server aggregation".to_string(),
            GradientAggregationMethod::Hierarchical => "Hierarchical aggregation".to_string(),
            GradientAggregationMethod::Compressed { compression_ratio } => {
                format!("Compressed aggregation (ratio: {:.2})", compression_ratio)
            },
        }
    }

    /// Get communication overhead
    pub fn communication_overhead(&self) -> f64 {
        match self {
            GradientAggregationMethod::Average => 1.0,
            GradientAggregationMethod::WeightedAverage => 1.1,
            GradientAggregationMethod::AllReduce => 0.8,
            GradientAggregationMethod::ParameterServer => 1.2,
            GradientAggregationMethod::Hierarchical => 0.9,
            GradientAggregationMethod::Compressed { compression_ratio } => *compression_ratio,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Load-aware assignment
    LoadAware,
    /// Performance-based assignment
    PerformanceBased,
    /// Adaptive assignment based on runtime metrics
    Adaptive,
    /// Custom load balancing function
    Custom(String),
}

impl LoadBalancingStrategy {
    /// Get strategy description
    pub fn description(&self) -> String {
        match self {
            LoadBalancingStrategy::RoundRobin => "Round-robin load balancing".to_string(),
            LoadBalancingStrategy::LoadAware => "Load-aware balancing".to_string(),
            LoadBalancingStrategy::PerformanceBased => "Performance-based balancing".to_string(),
            LoadBalancingStrategy::Adaptive => "Adaptive load balancing".to_string(),
            LoadBalancingStrategy::Custom(name) => format!("Custom strategy: {}", name),
        }
    }

    /// Check if strategy requires runtime monitoring
    pub fn requires_monitoring(&self) -> bool {
        match self {
            LoadBalancingStrategy::RoundRobin => false,
            LoadBalancingStrategy::LoadAware => true,
            LoadBalancingStrategy::PerformanceBased => true,
            LoadBalancingStrategy::Adaptive => true,
            LoadBalancingStrategy::Custom(_) => true,
        }
    }
}

/// Memory management strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryManagementStrategy {
    /// Static memory allocation
    Static,
    /// Dynamic memory allocation
    Dynamic,
    /// Memory pooling with reuse
    Pooled,
    /// Hierarchical memory management
    Hierarchical,
    /// Compressed memory storage
    Compressed,
}

impl MemoryManagementStrategy {
    /// Get strategy description
    pub fn description(&self) -> &'static str {
        match self {
            MemoryManagementStrategy::Static => "Static memory allocation",
            MemoryManagementStrategy::Dynamic => "Dynamic memory allocation",
            MemoryManagementStrategy::Pooled => "Memory pooling with reuse",
            MemoryManagementStrategy::Hierarchical => "Hierarchical memory management",
            MemoryManagementStrategy::Compressed => "Compressed memory storage",
        }
    }

    /// Get memory efficiency
    pub fn memory_efficiency(&self) -> f64 {
        match self {
            MemoryManagementStrategy::Static => 0.7,
            MemoryManagementStrategy::Dynamic => 0.8,
            MemoryManagementStrategy::Pooled => 0.9,
            MemoryManagementStrategy::Hierarchical => 0.85,
            MemoryManagementStrategy::Compressed => 0.95,
        }
    }
}

/// Quality of Service requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSRequirements {
    /// Maximum acceptable latency in milliseconds
    pub max_latency: f64,
    /// Minimum required throughput
    pub min_throughput: f64,
    /// Target accuracy for computations
    pub target_accuracy: f64,
    /// Reliability requirements (0.0 to 1.0)
    pub reliability: f64,
    /// Energy efficiency requirements
    pub energy_efficiency: f64,
}

impl QoSRequirements {
    /// Create new QoS requirements
    pub fn new(
        max_latency: f64,
        min_throughput: f64,
        target_accuracy: f64,
        reliability: f64,
        energy_efficiency: f64,
    ) -> Self {
        Self {
            max_latency,
            min_throughput,
            target_accuracy,
            reliability,
            energy_efficiency,
        }
    }

    /// Validate QoS requirements
    pub fn validate(&self) -> Result<(), String> {
        if self.max_latency <= 0.0 {
            return Err("Maximum latency must be positive".to_string());
        }
        if self.min_throughput <= 0.0 {
            return Err("Minimum throughput must be positive".to_string());
        }
        if self.target_accuracy < 0.0 || self.target_accuracy > 1.0 {
            return Err("Target accuracy must be between 0.0 and 1.0".to_string());
        }
        if self.reliability < 0.0 || self.reliability > 1.0 {
            return Err("Reliability must be between 0.0 and 1.0".to_string());
        }
        if self.energy_efficiency < 0.0 || self.energy_efficiency > 1.0 {
            return Err("Energy efficiency must be between 0.0 and 1.0".to_string());
        }
        Ok(())
    }

    /// Check if requirements are met
    pub fn are_met(
        &self,
        actual_latency: f64,
        actual_throughput: f64,
        actual_accuracy: f64,
        actual_reliability: f64,
        actual_efficiency: f64,
    ) -> bool {
        actual_latency <= self.max_latency
            && actual_throughput >= self.min_throughput
            && actual_accuracy >= self.target_accuracy
            && actual_reliability >= self.reliability
            && actual_efficiency >= self.energy_efficiency
    }
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub struct ConfigValidationError {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Configuration error in field '{}': {}", self.field, self.message)
    }
}

impl std::error::Error for ConfigValidationError {}

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate pod coordination configuration
    pub fn validate(config: &PodCoordinationConfig) -> Result<(), Vec<ConfigValidationError>> {
        let mut errors = Vec::new();

        // Validate device count
        if config.device_count == 0 {
            errors.push(ConfigValidationError {
                field: "device_count".to_string(),
                message: "Device count must be greater than 0".to_string(),
            });
        }

        // Validate coordination timeout
        if config.coordination_timeout == 0 {
            errors.push(ConfigValidationError {
                field: "coordination_timeout".to_string(),
                message: "Coordination timeout must be greater than 0".to_string(),
            });
        }

        // Validate monitoring interval
        if config.monitoring_interval == 0 {
            errors.push(ConfigValidationError {
                field: "monitoring_interval".to_string(),
                message: "Monitoring interval must be greater than 0".to_string(),
            });
        }

        // Validate QoS requirements
        if let Err(qos_error) = config.qos_requirements.validate() {
            errors.push(ConfigValidationError {
                field: "qos_requirements".to_string(),
                message: qos_error,
            });
        }

        // Validate strategy compatibility
        if !config.coordination_strategy.supports_fault_tolerance() && config.enable_fault_tolerance {
            errors.push(ConfigValidationError {
                field: "coordination_strategy".to_string(),
                message: "Selected coordination strategy does not support fault tolerance".to_string(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// Default implementations
impl Default for PodCoordinationConfig {
    fn default() -> Self {
        Self {
            device_count: 8,
            coordination_strategy: CoordinationStrategy::Adaptive,
            communication_pattern: CommunicationPattern::AllToAll,
            synchronization_mode: SynchronizationMode::BulkSynchronous,
            batch_parallelization: BatchParallelizationStrategy::DataParallel,
            gradient_aggregation: GradientAggregationMethod::AllReduce,
            load_balancing: LoadBalancingStrategy::Adaptive,
            memory_management: MemoryManagementStrategy::Dynamic,
            coordination_timeout: 30,
            monitoring_interval: 1000,
            enable_fault_tolerance: true,
            enable_adaptive_optimization: true,
            qos_requirements: QoSRequirements::default(),
        }
    }
}

impl Default for QoSRequirements {
    fn default() -> Self {
        Self {
            max_latency: 100.0,
            min_throughput: 1000.0,
            target_accuracy: 0.99,
            reliability: 0.999,
            energy_efficiency: 0.8,
        }
    }
}

/// Configuration profiles for common use cases
pub struct ConfigProfiles;

impl ConfigProfiles {
    /// High performance configuration
    pub fn high_performance() -> PodCoordinationConfig {
        PodCoordinationConfigBuilder::new()
            .device_count(16)
            .coordination_strategy(CoordinationStrategy::Hierarchical)
            .communication_pattern(CommunicationPattern::AllToAll)
            .synchronization_mode(SynchronizationMode::BulkSynchronous)
            .batch_parallelization(BatchParallelizationStrategy::Hybrid {
                data_parallel_factor: 4,
                model_parallel_factor: 4,
            })
            .gradient_aggregation(GradientAggregationMethod::AllReduce)
            .load_balancing(LoadBalancingStrategy::PerformanceBased)
            .memory_management(MemoryManagementStrategy::Pooled)
            .coordination_timeout(10)
            .monitoring_interval(500)
            .enable_fault_tolerance(true)
            .enable_adaptive_optimization(true)
            .qos_requirements(QoSRequirements::new(50.0, 5000.0, 0.995, 0.999, 0.9))
            .build()
    }

    /// Low latency configuration
    pub fn low_latency() -> PodCoordinationConfig {
        PodCoordinationConfigBuilder::new()
            .device_count(8)
            .coordination_strategy(CoordinationStrategy::Decentralized)
            .communication_pattern(CommunicationPattern::Ring)
            .synchronization_mode(SynchronizationMode::Asynchronous)
            .batch_parallelization(BatchParallelizationStrategy::DataParallel)
            .gradient_aggregation(GradientAggregationMethod::Compressed { compression_ratio: 0.5 })
            .load_balancing(LoadBalancingStrategy::LoadAware)
            .memory_management(MemoryManagementStrategy::Dynamic)
            .coordination_timeout(5)
            .monitoring_interval(100)
            .enable_fault_tolerance(false)
            .enable_adaptive_optimization(true)
            .qos_requirements(QoSRequirements::new(10.0, 2000.0, 0.98, 0.99, 0.7))
            .build()
    }

    /// Energy efficient configuration
    pub fn energy_efficient() -> PodCoordinationConfig {
        PodCoordinationConfigBuilder::new()
            .device_count(4)
            .coordination_strategy(CoordinationStrategy::Centralized)
            .communication_pattern(CommunicationPattern::Tree)
            .synchronization_mode(SynchronizationMode::BoundedAsynchronous { staleness_bound: 3 })
            .batch_parallelization(BatchParallelizationStrategy::DataParallel)
            .gradient_aggregation(GradientAggregationMethod::Hierarchical)
            .load_balancing(LoadBalancingStrategy::Adaptive)
            .memory_management(MemoryManagementStrategy::Compressed)
            .coordination_timeout(60)
            .monitoring_interval(2000)
            .enable_fault_tolerance(true)
            .enable_adaptive_optimization(true)
            .qos_requirements(QoSRequirements::new(200.0, 500.0, 0.97, 0.995, 0.95))
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = PodCoordinationConfigBuilder::new()
            .device_count(16)
            .coordination_strategy(CoordinationStrategy::Hierarchical)
            .enable_fault_tolerance(true)
            .build();

        assert_eq!(config.device_count, 16);
        assert!(matches!(config.coordination_strategy, CoordinationStrategy::Hierarchical));
        assert!(config.enable_fault_tolerance);
    }

    #[test]
    fn test_config_validation() {
        let mut config = PodCoordinationConfig::default();
        config.device_count = 0;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "device_count");
    }

    #[test]
    fn test_qos_validation() {
        let qos = QoSRequirements::new(100.0, 1000.0, 1.5, 0.99, 0.8);
        assert!(qos.validate().is_err());

        let qos = QoSRequirements::new(100.0, 1000.0, 0.99, 0.99, 0.8);
        assert!(qos.validate().is_ok());
    }

    #[test]
    fn test_config_profiles() {
        let hp_config = ConfigProfiles::high_performance();
        assert_eq!(hp_config.device_count, 16);
        assert!(matches!(hp_config.coordination_strategy, CoordinationStrategy::Hierarchical));

        let ll_config = ConfigProfiles::low_latency();
        assert!(matches!(ll_config.synchronization_mode, SynchronizationMode::Asynchronous));

        let ee_config = ConfigProfiles::energy_efficient();
        assert!(matches!(ee_config.memory_management, MemoryManagementStrategy::Compressed));
    }
}