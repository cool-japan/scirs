//! Deadlock Detection and Prevention Module
//!
//! This module provides comprehensive deadlock detection, prevention, and recovery
//! mechanisms for distributed TPU synchronization. It includes graph-based algorithms,
//! machine learning approaches, performance monitoring, and recovery strategies.
//!
//! # Components
//!
//! - **Types**: Core data structures and configuration types
//! - **Algorithms**: Detection algorithms and optimization strategies
//! - **Prevention**: Prevention strategies and policies
//! - **Graph**: Dependency graph management and analysis
//! - **Performance**: Performance monitoring and statistics
//! - **Recovery**: Recovery coordination and execution
//! - **ML**: Machine learning components for prediction

pub mod algorithms;
pub mod graph;
pub mod ml;
pub mod performance;
pub mod prevention;
pub mod recovery;
pub mod types;

// Re-export core types
pub use types::{
    AdvancedDeadlockConfig, AdvancedDiagnostics, AdaptiveSensitivity, DeadlockDetectionConfig,
    DeadlockDetector, DeadlockPerformanceConfig, DeadlockSensitivity, DetectionEvent,
    DetectionEventType, DetectionState, DetectionStatus, DeliveryMethod, ExportFormat,
    IntegrationSettings, NotificationConfig, NotificationType, PerformanceOptimization,
    ResourceLimits, SensitivityMetric,
};

// Re-export algorithm types
pub use algorithms::{
    BackoffStrategy, CacheOptimization, CachePolicy, CombinationStrategy, ConflictResolution,
    CycleDetectionMethod, DeadlockCriteria, DeadlockDetectionAlgorithm, ErrorHandling,
    GraphOptimization, GraphReductionMethod, ParallelProcessing, PrefetchingStrategy,
    PropagationStrategy, ResourceAllocationMethod, ResponseHandling, RetryPolicy,
    SafeStateMethod, SynchronizationMethod, TimestampOrdering, WorkDistribution,
};

// Re-export prevention types
pub use prevention::{
    AllocationPolicy, AvoidanceStrategy, BankersAlgorithmConfig, CircularWaitPrevention,
    ConservativeStrategy, DeadlockPrevention, DeadlockPreventionSystem, HoldAndWaitPrevention,
    MutualExclusionPrevention, NoPreemptionPrevention, OptimisticStrategy, OrderingStrategy,
    PreemptionPolicy, PreemptionStrategy, PreventionPolicy, PreventionStatistics,
    ResourceOrdering, TimeoutStrategy, ValidationStatistics, WoundWaitStrategy,
};

// Re-export machine learning types
pub use ml::{
    CombinationStrategy as MLCombinationStrategy, EnsembleMethod, FeatureExtraction,
    GraphFeature, MLModelType, ResourceFeature, TemporalFeature,
};

// Re-export graph types
pub use graph::{
    ChangeType, DependencyEdge, DependencyGraph, EdgeMetadata, EdgeType, GraphChange,
    GraphHistory, GraphMetadata, GraphNode, GraphOptimizationState, GraphProperties,
    GraphSnapshot, GraphStatistics, NodeMetadata, NodeState, NodeType, OptimizationOperation,
    OptimizationRecord, OptimizationStatistics, PerformanceImpact,
};

// Re-export performance types
pub use performance::{
    AllocationStrategy, AutoTuning, CachingStrategies, ComputationCaching, CpuLimits,
    CpuManagement, CpuMonitoring, DeadlockPerformanceConfig as PerformanceConfig,
    DeadlockStatistics, DetectionTimePercentiles, DetectionTimeStatistics, EvictionPolicy,
    GarbageCollection, GcStrategy, HealthChecking, HorizontalScaling, IoLimits, IoManagement,
    IoMonitoring, IoOptimization, IoScheduling, LoadBalancing, LoadBalancingMetric,
    LoadBalancingMonitoring, LoadBalancingStrategy, LoadBalancingThresholds, MemoryLimits,
    MemoryManagement, MemoryMonitoring, PerformanceMetric, PerformanceMonitoring,
    PerformanceOptimization as PerfOptimization, PerformanceTargets, PerformanceThresholds,
    ResultCaching, ScalabilityConfiguration, ScalingTrigger, SpawningStrategy,
    SystemImpactStatistics, ThreadManagement, UpdateStrategy, VerticalLimits, VerticalMetric,
    VerticalPerformanceMonitoring, VerticalScaling,
};

// Re-export recovery types
pub use recovery::{
    ActiveRecovery, CoordinatorSelection, CoordinatorState, DeadlockRecovery,
    DeadlockRecoverySystem, DeadlockSeverity, DetectedDeadlock, DistributedRecovery,
    DistributedRecoveryStrategy, ExecutionContext, ExecutionRecord, ExecutorCapabilities,
    ExecutorPerformance, PhaseRecord, PhaseResult, RecoveryAction, RecoveryCoordination,
    RecoveryCoordinator, RecoveryConstraint, RecoveryConstraintType, RecoveryExecutor,
    RecoveryExecutorStrategy, RecoveryObjective, RecoveryOptimization,
    RecoveryOptimizationAlgorithm, RecoveryPhase, RecoveryProgress, RecoveryRequest,
    RecoveryResult, RecoveryStatistics, RecoveryStrategy, RecoveryVerification,
    RecoveryVerificationMethod, RollbackMechanism, SelectionCriterion, StateSynchronization,
    StrategyStatistics, SynchronizationProtocol, SystemHealth, SystemState,
    VerificationSuccessCriteria, VictimSelection, VictimSelectionAlgorithm,
};

// Convenience type aliases for backward compatibility
pub type DeadlockConfig = DeadlockDetectionConfig;
pub type Statistics = DeadlockStatistics;
pub type PerformanceConfig = DeadlockPerformanceConfig;
pub type RecoveryConfig = DeadlockRecovery;

/// Create a new deadlock detector with default configuration
pub fn create_detector() -> crate::error::Result<DeadlockDetector> {
    DeadlockDetector::new()
}

/// Create a new deadlock detector with custom configuration
pub fn create_detector_with_config(config: DeadlockDetectionConfig) -> crate::error::Result<DeadlockDetector> {
    let mut detector = DeadlockDetector::new()?;
    detector.config = config;
    Ok(detector)
}

/// Create a new dependency graph
pub fn create_dependency_graph() -> crate::error::Result<DependencyGraph> {
    DependencyGraph::new()
}

/// Create a new recovery system
pub fn create_recovery_system(config: DeadlockRecovery) -> crate::error::Result<DeadlockRecoverySystem> {
    DeadlockRecoverySystem::new(config)
}

// Implementation from the original file that needs to be preserved
impl DeadlockDetector {
    /// Create a new deadlock detector
    pub fn new() -> crate::error::Result<Self> {
        Ok(Self {
            config: DeadlockDetectionConfig::default(),
            dependency_graph: DependencyGraph::new()?,
            detection_state: DetectionState {
                status: DetectionStatus::Idle,
                last_detection: std::time::Instant::now(),
                active_deadlocks: 0,
                history: Vec::new(),
            },
            statistics: DeadlockStatistics::default(),
            prevention_system: prevention::DeadlockPreventionSystem::new()?,
            recovery_system: recovery::DeadlockRecoverySystem::new(DeadlockRecovery::default())?,
        })
    }

    /// Detect deadlocks in the system
    pub fn detect_deadlocks(&mut self) -> crate::error::Result<Vec<String>> {
        self.detection_state.status = DetectionStatus::Running;
        self.detection_state.last_detection = std::time::Instant::now();

        // Use the graph's cycle detection
        if self.dependency_graph.has_cycle() {
            self.detection_state.status = DetectionStatus::DeadlockDetected;
            self.detection_state.active_deadlocks += 1;
            self.statistics.deadlocks_detected += 1;

            // Return detected deadlock IDs (simplified)
            Ok(vec!["deadlock_1".to_string()])
        } else {
            self.detection_state.status = DetectionStatus::Idle;
            Ok(Vec::new())
        }
    }

    /// Add a resource dependency
    pub fn add_dependency(&mut self, source: String, target: String) -> crate::error::Result<()> {
        use graph::{DependencyEdge, EdgeType, EdgeMetadata};

        // Add nodes if they don't exist
        if !self.dependency_graph.nodes.contains_key(&source) {
            let node = graph::GraphNode {
                id: source.clone(),
                node_type: graph::NodeType::Process,
                state: graph::NodeState::Active,
                metadata: graph::NodeMetadata::default(),
                timestamp: std::time::Instant::now(),
            };
            self.dependency_graph.add_node(node)?;
        }

        if !self.dependency_graph.nodes.contains_key(&target) {
            let node = graph::GraphNode {
                id: target.clone(),
                node_type: graph::NodeType::Resource,
                state: graph::NodeState::Active,
                metadata: graph::NodeMetadata::default(),
                timestamp: std::time::Instant::now(),
            };
            self.dependency_graph.add_node(node)?;
        }

        // Add the edge
        let edge = DependencyEdge {
            source: source.clone(),
            target: target.clone(),
            edge_type: EdgeType::WaitsFor,
            weight: 1.0,
            timestamp: std::time::Instant::now(),
            metadata: EdgeMetadata::default(),
        };

        self.dependency_graph.add_edge(edge)?;
        Ok(())
    }

    /// Remove a resource dependency
    pub fn remove_dependency(&mut self, source: &str, target: &str) -> crate::error::Result<()> {
        if let Some(edges) = self.dependency_graph.edges.get_mut(source) {
            edges.retain(|edge| edge.target != target);
            if edges.is_empty() {
                self.dependency_graph.edges.remove(source);
            }
        }
        Ok(())
    }

    /// Get current detection statistics
    pub fn get_statistics(&self) -> &DeadlockStatistics {
        &self.statistics
    }

    /// Update detector configuration
    pub fn update_config(&mut self, config: DeadlockDetectionConfig) {
        self.config = config;
    }
}