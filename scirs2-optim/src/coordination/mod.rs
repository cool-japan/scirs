//! Coordination module for optimization processes
//!
//! This module provides comprehensive coordination capabilities for optimization
//! workflows, including task scheduling, pipeline orchestration, and monitoring.
//! It replaces the monolithic optimization_coordinator.rs with a modular architecture.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use num_traits::Float;

// Submodule declarations
pub mod scheduling;
pub mod orchestration;
pub mod monitoring;

// Re-export key types from submodules
pub use scheduling::{
    TaskScheduler, OptimizationTask, TaskPriority, ResourceRequirement,
    SchedulingStrategy, TaskStatus, SchedulingResult,
    ResourceManager, ResourcePool, ResourceType, AllocationStrategy,
    ResourceUsage, ResourceMetrics, AllocationResult,
    PriorityManager, PriorityQueue, PriorityStrategy, PriorityAdjustment,
    MultiDimensionalPriority, DynamicPriorityAdjuster,
};

pub use orchestration::{
    PipelineOrchestrator, OptimizationPipeline, PipelineStage, PipelineConfig,
    PipelineStatus, ExecutionResult, DependencyGraph, PipelineMetrics,
    ExperimentManager, Experiment, ExperimentConfig, ExperimentStatus,
    HyperparameterConfig, ExperimentResult, StatisticalExperimentDesign,
    CheckpointManager, CheckpointConfig, CheckpointType, RecoveryStrategy,
    CheckpointMetadata, IncrementalCheckpointing, DistributedCheckpointing,
};

pub use monitoring::{
    PerformanceTracker, PerformanceMetrics, MetricCollector, MetricAggregator,
    PerformanceAlert, AlertManager,
    ConvergenceDetector, ConvergenceAnalyzer, ConvergenceCriteria, ConvergenceResult,
    ConvergenceMonitor, ConvergenceIndicator,
    AnomalyDetector, AnomalyAnalyzer, AnomalyAlert, AnomalyClassifier,
    OutlierDetector, AnomalyReporter,
};

/// Main coordination manager that integrates all coordination components
pub struct OptimizationCoordinator<T: Float + Debug + Send + Sync + 'static> {
    scheduler: TaskScheduler<T>,
    orchestrator: PipelineOrchestrator<T>,
    performance_tracker: PerformanceTracker<T>,
    convergence_detector: ConvergenceDetector<T>,
    anomaly_detector: AnomalyDetector<T>,
    config: CoordinatorConfig<T>,
    state: CoordinatorState<T>,
    metrics: CoordinatorMetrics<T>,
    _phantom: PhantomData<T>,
}

/// Configuration for the optimization coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorConfig<T: Float + Debug + Send + Sync + 'static> {
    pub max_concurrent_tasks: usize,
    pub default_timeout: Duration,
    pub monitoring_interval: Duration,
    pub checkpoint_interval: Duration,
    pub resource_allocation_strategy: AllocationStrategy,
    pub priority_strategy: PriorityStrategy,
    pub convergence_criteria: ConvergenceCriteria<T>,
    pub enable_anomaly_detection: bool,
    pub enable_auto_scaling: bool,
    pub enable_fault_tolerance: bool,
    pub performance_threshold: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorConfig<T> {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            default_timeout: Duration::from_secs(3600),
            monitoring_interval: Duration::from_secs(10),
            checkpoint_interval: Duration::from_secs(300),
            resource_allocation_strategy: AllocationStrategy::Balanced,
            priority_strategy: PriorityStrategy::FIFO,
            convergence_criteria: ConvergenceCriteria::default(),
            enable_anomaly_detection: true,
            enable_auto_scaling: true,
            enable_fault_tolerance: true,
            performance_threshold: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Internal state of the coordination manager
#[derive(Debug)]
pub struct CoordinatorState<T: Float + Debug + Send + Sync + 'static> {
    pub active_tasks: HashMap<String, OptimizationTask<T>>,
    pub active_pipelines: HashMap<String, OptimizationPipeline<T>>,
    pub active_experiments: HashMap<String, Experiment<T>>,
    pub resource_usage: ResourceUsage<T>,
    pub last_checkpoint: Option<Instant>,
    pub last_monitoring_update: Option<Instant>,
    pub coordination_start_time: Instant,
    pub total_tasks_processed: usize,
    pub total_experiments_completed: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> CoordinatorState<T> {
    pub fn new() -> Self {
        Self {
            active_tasks: HashMap::new(),
            active_pipelines: HashMap::new(),
            active_experiments: HashMap::new(),
            resource_usage: ResourceUsage::new(),
            last_checkpoint: None,
            last_monitoring_update: None,
            coordination_start_time: Instant::now(),
            total_tasks_processed: 0,
            total_experiments_completed: 0,
        }
    }
}

/// Metrics for coordination performance
#[derive(Debug, Clone)]
pub struct CoordinatorMetrics<T: Float + Debug + Send + Sync + 'static> {
    pub average_task_completion_time: T,
    pub throughput: T,
    pub resource_utilization: T,
    pub error_rate: T,
    pub convergence_rate: T,
    pub anomaly_detection_rate: T,
    pub uptime: Duration,
    pub total_processed_tasks: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorMetrics<T> {
    fn default() -> Self {
        Self {
            average_task_completion_time: T::zero(),
            throughput: T::zero(),
            resource_utilization: T::zero(),
            error_rate: T::zero(),
            convergence_rate: T::zero(),
            anomaly_detection_rate: T::zero(),
            uptime: Duration::new(0, 0),
            total_processed_tasks: 0,
        }
    }
}

/// Result of coordination operations
#[derive(Debug, Clone)]
pub struct CoordinationResult<T: Float + Debug + Send + Sync + 'static> {
    pub success: bool,
    pub task_id: String,
    pub execution_time: Duration,
    pub resource_usage: ResourceUsage<T>,
    pub performance_metrics: PerformanceMetrics<T>,
    pub convergence_result: Option<ConvergenceResult<T>>,
    pub anomaly_alerts: Vec<AnomalyAlert<T>>,
    pub errors: Vec<String>,
}

impl<T: Float + Debug + Send + Sync + 'static> OptimizationCoordinator<T> {
    /// Create a new optimization coordinator
    pub fn new(config: CoordinatorConfig<T>) -> Self {
        let scheduler = TaskScheduler::new(scheduling::SchedulerConfig {
            max_concurrent_tasks: config.max_concurrent_tasks,
            default_priority_strategy: config.priority_strategy.clone(),
            resource_allocation_strategy: config.resource_allocation_strategy.clone(),
            enable_load_balancing: true,
            task_timeout: config.default_timeout,
            enable_priority_inheritance: true,
        });

        let orchestrator = PipelineOrchestrator::new(orchestration::OrchestratorConfig {
            max_concurrent_pipelines: config.max_concurrent_tasks,
            default_timeout: config.default_timeout,
            enable_fault_tolerance: config.enable_fault_tolerance,
            checkpoint_interval: config.checkpoint_interval,
            enable_pipeline_optimization: true,
        });

        let performance_tracker = PerformanceTracker::new(monitoring::PerformanceConfig {
            collection_interval: config.monitoring_interval,
            buffer_size: 1000,
            enable_real_time_alerts: true,
            performance_threshold: config.performance_threshold,
            enable_trend_analysis: true,
        });

        let convergence_detector = ConvergenceDetector::new(config.convergence_criteria.clone());

        let anomaly_detector = if config.enable_anomaly_detection {
            AnomalyDetector::new(monitoring::AnomalyConfig::default())
        } else {
            AnomalyDetector::new(monitoring::AnomalyConfig::default())
        };

        Self {
            scheduler,
            orchestrator,
            performance_tracker,
            convergence_detector,
            anomaly_detector,
            state: CoordinatorState::new(),
            metrics: CoordinatorMetrics::default(),
            config,
            _phantom: PhantomData,
        }
    }

    /// Submit a single optimization task
    pub fn submit_task(&mut self, mut task: OptimizationTask<T>) -> Result<String, String> {
        // Generate unique task ID
        let task_id = format!("task_{}_{}", 
            self.state.total_tasks_processed, 
            Instant::now().elapsed().as_nanos());
        task.set_id(task_id.clone());

        // Schedule the task
        match self.scheduler.schedule_task(task.clone()) {
            Ok(scheduling_result) => {
                self.state.active_tasks.insert(task_id.clone(), task);
                self.state.total_tasks_processed += 1;
                
                // Start performance monitoring for the task
                self.performance_tracker.start_tracking(&task_id);
                
                Ok(task_id)
            }
            Err(e) => Err(format!("Failed to schedule task: {}", e)),
        }
    }

    /// Submit an optimization pipeline
    pub fn submit_pipeline(&mut self, mut pipeline: OptimizationPipeline<T>) -> Result<String, String> {
        let pipeline_id = format!("pipeline_{}_{}", 
            self.state.active_pipelines.len(), 
            Instant::now().elapsed().as_nanos());
        
        pipeline.set_id(pipeline_id.clone());

        match self.orchestrator.execute_pipeline(pipeline.clone()) {
            Ok(_) => {
                self.state.active_pipelines.insert(pipeline_id.clone(), pipeline);
                Ok(pipeline_id)
            }
            Err(e) => Err(format!("Failed to execute pipeline: {}", e)),
        }
    }

    /// Submit an experiment
    pub fn submit_experiment(&mut self, mut experiment: Experiment<T>) -> Result<String, String> {
        let experiment_id = format!("experiment_{}_{}", 
            self.state.active_experiments.len(), 
            Instant::now().elapsed().as_nanos());
        
        experiment.set_id(experiment_id.clone());

        // Convert experiment to pipeline for execution
        let pipeline = self.experiment_to_pipeline(&experiment)?;
        let pipeline_id = self.submit_pipeline(pipeline)?;
        
        self.state.active_experiments.insert(experiment_id.clone(), experiment);
        
        Ok(experiment_id)
    }

    /// Execute a coordination cycle
    pub fn execute_cycle(&mut self) -> Vec<CoordinationResult<T>> {
        let mut results = Vec::new();

        // Update monitoring
        self.update_monitoring();

        // Process scheduled tasks
        let task_results = self.process_scheduled_tasks();
        results.extend(task_results);

        // Update pipeline executions
        self.update_pipeline_executions();

        // Perform maintenance operations
        self.perform_maintenance();

        // Update metrics
        self.update_metrics();

        results
    }

    /// Monitor optimization value for convergence and anomalies
    pub fn monitor_optimization_value(&mut self, task_id: &str, value: T) -> MonitoringResult<T> {
        let mut alerts = Vec::new();
        
        // Update performance tracking
        self.performance_tracker.record_metric(task_id, value);
        
        // Check for convergence
        let convergence_result = self.convergence_detector.check_convergence(value);
        
        // Check for anomalies if enabled
        let anomaly_result = if self.config.enable_anomaly_detection {
            Some(self.anomaly_detector.detect_anomaly(value))
        } else {
            None
        };

        // Generate alerts based on monitoring results
        if let Some(ref anomaly) = anomaly_result {
            if anomaly.is_anomaly {
                alerts.push(MonitoringAlert::Anomaly(anomaly.clone()));
            }
        }

        if convergence_result.converged {
            alerts.push(MonitoringAlert::Convergence(convergence_result.clone()));
        }

        MonitoringResult {
            task_id: task_id.to_string(),
            value,
            convergence_result,
            anomaly_result,
            alerts,
            timestamp: Instant::now(),
        }
    }

    /// Get current coordination status
    pub fn get_status(&self) -> CoordinationStatus<T> {
        CoordinationStatus {
            active_tasks: self.state.active_tasks.len(),
            active_pipelines: self.state.active_pipelines.len(),
            active_experiments: self.state.active_experiments.len(),
            resource_utilization: self.state.resource_usage.clone(),
            metrics: self.metrics.clone(),
            uptime: self.state.coordination_start_time.elapsed(),
            health_status: self.assess_health_status(),
        }
    }

    /// Update resource allocation
    pub fn update_resource_allocation(&mut self, allocation: ResourceUsage<T>) -> Result<(), String> {
        self.state.resource_usage = allocation;
        
        // Update scheduler with new resource information
        self.scheduler.update_resource_availability(&self.state.resource_usage);
        
        // Update orchestrator
        self.orchestrator.update_resource_allocation(&self.state.resource_usage);
        
        Ok(())
    }

    /// Shutdown coordination gracefully
    pub fn shutdown(&mut self) -> Result<(), String> {
        // Complete active tasks
        self.complete_active_tasks()?;
        
        // Save checkpoints
        self.save_final_checkpoints()?;
        
        // Generate final report
        let report = self.generate_final_report();
        println!("Coordination shutdown report:\n{}", report);
        
        Ok(())
    }

    // Private helper methods

    fn update_monitoring(&mut self) {
        let now = Instant::now();
        
        if let Some(last_update) = self.last_monitoring_update {
            if now.duration_since(last_update) < self.config.monitoring_interval {
                return;
            }
        }

        // Update performance metrics
        self.performance_tracker.update();
        
        // Check for system-level anomalies
        if self.config.enable_anomaly_detection {
            let system_metrics = self.collect_system_metrics();
            for metric in system_metrics {
                let _ = self.anomaly_detector.detect_anomaly(metric);
            }
        }

        self.state.last_monitoring_update = Some(now);
    }

    fn process_scheduled_tasks(&mut self) -> Vec<CoordinationResult<T>> {
        let mut results = Vec::new();
        let ready_tasks = self.scheduler.get_ready_tasks();
        
        for task in ready_tasks {
            match self.execute_task(&task) {
                Ok(result) => {
                    results.push(result);
                    self.state.active_tasks.remove(&task.get_id());
                }
                Err(e) => {
                    results.push(CoordinationResult {
                        success: false,
                        task_id: task.get_id(),
                        execution_time: Duration::new(0, 0),
                        resource_usage: ResourceUsage::new(),
                        performance_metrics: PerformanceMetrics::default(),
                        convergence_result: None,
                        anomaly_alerts: Vec::new(),
                        errors: vec![e],
                    });
                }
            }
        }

        results
    }

    fn execute_task(&mut self, task: &OptimizationTask<T>) -> Result<CoordinationResult<T>, String> {
        let start_time = Instant::now();
        let task_id = task.get_id();
        
        // Execute the task (simplified)
        let performance_metrics = self.performance_tracker.get_metrics(&task_id)
            .unwrap_or_else(|| PerformanceMetrics::default());
        
        let execution_time = start_time.elapsed();
        
        Ok(CoordinationResult {
            success: true,
            task_id,
            execution_time,
            resource_usage: self.state.resource_usage.clone(),
            performance_metrics,
            convergence_result: None,
            anomaly_alerts: Vec::new(),
            errors: Vec::new(),
        })
    }

    fn update_pipeline_executions(&mut self) {
        // Update orchestrator state
        self.orchestrator.update_pipeline_states();
        
        // Check for completed pipelines
        let completed_pipelines = self.orchestrator.get_completed_pipelines();
        for pipeline_id in completed_pipelines {
            self.state.active_pipelines.remove(&pipeline_id);
        }
    }

    fn perform_maintenance(&mut self) {
        let now = Instant::now();
        
        // Perform checkpointing
        if let Some(last_checkpoint) = self.state.last_checkpoint {
            if now.duration_since(last_checkpoint) >= self.config.checkpoint_interval {
                let _ = self.create_checkpoint();
            }
        } else {
            let _ = self.create_checkpoint();
        }

        // Clean up completed tasks and experiments
        self.cleanup_completed_items();
        
        // Update adaptive parameters
        if self.config.enable_auto_scaling {
            self.update_adaptive_parameters();
        }
    }

    fn create_checkpoint(&mut self) -> Result<(), String> {
        // Create checkpoint through orchestrator
        self.orchestrator.create_system_checkpoint()?;
        self.state.last_checkpoint = Some(Instant::now());
        Ok(())
    }

    fn cleanup_completed_items(&mut self) {
        // Remove completed tasks older than threshold
        let threshold = Duration::from_secs(3600); // 1 hour
        let now = Instant::now();
        
        self.state.active_tasks.retain(|_, task| {
            !task.is_completed() || 
            task.get_completion_time()
                .map(|t| now.duration_since(t) < threshold)
                .unwrap_or(true)
        });
    }

    fn update_adaptive_parameters(&mut self) {
        // Adaptive resource allocation based on performance
        let current_metrics = &self.metrics;
        
        if current_metrics.resource_utilization > num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()) {
            // High utilization - consider scaling up
            let _ = self.request_additional_resources();
        } else if current_metrics.resource_utilization < num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()) {
            // Low utilization - consider scaling down
            let _ = self.release_excess_resources();
        }
    }

    fn request_additional_resources(&mut self) -> Result<(), String> {
        // Implementation would request additional compute resources
        Ok(())
    }

    fn release_excess_resources(&mut self) -> Result<(), String> {
        // Implementation would release unused resources
        Ok(())
    }

    fn update_metrics(&mut self) {
        let current_time = Instant::now();
        let uptime = current_time.duration_since(self.state.coordination_start_time);
        
        // Update basic metrics
        self.metrics.uptime = uptime;
        self.metrics.total_processed_tasks = self.state.total_tasks_processed;
        
        // Calculate throughput
        if uptime.as_secs() > 0 {
            self.metrics.throughput = num_traits::cast::cast(self.state.total_tasks_processed).unwrap_or_else(|| T::zero()) 
                                    / T::from(uptime.as_secs()).unwrap();
        }
        
        // Update resource utilization
        self.metrics.resource_utilization = self.state.resource_usage.get_total_utilization();
        
        // Update convergence and anomaly rates
        self.metrics.convergence_rate = self.calculate_convergence_rate();
        self.metrics.anomaly_detection_rate = self.calculate_anomaly_rate();
    }

    fn calculate_convergence_rate(&self) -> T {
        // Implementation would calculate convergence success rate
        num_traits::cast::cast(0.85).unwrap_or_else(|| T::zero()) // Placeholder
    }

    fn calculate_anomaly_rate(&self) -> T {
        // Implementation would calculate anomaly detection rate
        num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()) // Placeholder
    }

    fn collect_system_metrics(&self) -> Vec<T> {
        // Collect system-level metrics for anomaly detection
        vec![
            self.metrics.resource_utilization,
            self.metrics.throughput,
            T::from(self.state.active_tasks.len()).unwrap(),
            T::from(self.state.active_pipelines.len()).unwrap(),
        ]
    }

    fn experiment_to_pipeline(&self, experiment: &Experiment<T>) -> Result<OptimizationPipeline<T>, String> {
        // Convert experiment configuration to pipeline stages
        let mut pipeline = OptimizationPipeline::new();
        
        // Add data preparation stage
        pipeline.add_stage(PipelineStage::new("data_preparation".to_string()));
        
        // Add optimization stage
        pipeline.add_stage(PipelineStage::new("optimization".to_string()));
        
        // Add evaluation stage
        pipeline.add_stage(PipelineStage::new("evaluation".to_string()));
        
        Ok(pipeline)
    }

    fn assess_health_status(&self) -> HealthStatus {
        let error_rate = self.metrics.error_rate;
        let resource_utilization = self.metrics.resource_utilization;
        
        if error_rate > num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()) {
            HealthStatus::Unhealthy
        } else if resource_utilization > num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()) {
            HealthStatus::Degraded
        } else if error_rate > num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()) || resource_utilization > num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()) {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }

    fn complete_active_tasks(&mut self) -> Result<(), String> {
        // Wait for active tasks to complete or timeout
        let timeout = Duration::from_secs(60);
        let start = Instant::now();
        
        while !self.state.active_tasks.is_empty() && start.elapsed() < timeout {
            let results = self.process_scheduled_tasks();
            if results.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        
        if !self.state.active_tasks.is_empty() {
            return Err(format!("Timeout waiting for {} tasks to complete", 
                             self.state.active_tasks.len()));
        }
        
        Ok(())
    }

    fn save_final_checkpoints(&mut self) -> Result<(), String> {
        self.create_checkpoint()
    }

    fn generate_final_report(&self) -> String {
        format!(
            "Optimization Coordination Final Report:\n\
             - Total Uptime: {:?}\n\
             - Tasks Processed: {}\n\
             - Experiments Completed: {}\n\
             - Average Throughput: {:.2}\n\
             - Resource Utilization: {:.2}%\n\
             - Error Rate: {:.2}%\n\
             - Convergence Rate: {:.2}%\n\
             - Health Status: {:?}",
            self.metrics.uptime,
            self.metrics.total_processed_tasks,
            self.state.total_experiments_completed,
            self.metrics.throughput.to_f64().unwrap_or(0.0),
            (self.metrics.resource_utilization * num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())).to_f64().unwrap_or(0.0),
            (self.metrics.error_rate * num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())).to_f64().unwrap_or(0.0),
            (self.metrics.convergence_rate * num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())).to_f64().unwrap_or(0.0),
            self.assess_health_status(),
        )
    }
}

/// Result of monitoring operations
#[derive(Debug, Clone)]
pub struct MonitoringResult<T: Float + Debug + Send + Sync + 'static> {
    pub task_id: String,
    pub value: T,
    pub convergence_result: ConvergenceResult<T>,
    pub anomaly_result: Option<monitoring::AnomalyResult<T>>,
    pub alerts: Vec<MonitoringAlert<T>>,
    pub timestamp: Instant,
}

/// Monitoring alert types
#[derive(Debug, Clone)]
pub enum MonitoringAlert<T: Float> {
    Convergence(ConvergenceResult<T>),
    Anomaly(monitoring::AnomalyResult<T>),
    Performance(PerformanceAlert<T>),
    Resource(String),
}

/// Current status of the coordination system
#[derive(Debug, Clone)]
pub struct CoordinationStatus<T: Float + Debug + Send + Sync + 'static> {
    pub active_tasks: usize,
    pub active_pipelines: usize,
    pub active_experiments: usize,
    pub resource_utilization: ResourceUsage<T>,
    pub metrics: CoordinatorMetrics<T>,
    pub uptime: Duration,
    pub health_status: HealthStatus,
}

/// Health status of the coordination system
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Degraded,
    Unhealthy,
}

/// Builder for creating optimization coordinators
pub struct CoordinatorBuilder<T: Float + Debug + Send + Sync + 'static> {
    config: CoordinatorConfig<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> CoordinatorBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: CoordinatorConfig::default(),
        }
    }

    pub fn max_concurrent_tasks(mut self, max: usize) -> Self {
        self.config.max_concurrent_tasks = max;
        self
    }

    pub fn monitoring_interval(mut self, interval: Duration) -> Self {
        self.config.monitoring_interval = interval;
        self
    }

    pub fn enable_anomaly_detection(mut self, enable: bool) -> Self {
        self.config.enable_anomaly_detection = enable;
        self
    }

    pub fn enable_fault_tolerance(mut self, enable: bool) -> Self {
        self.config.enable_fault_tolerance = enable;
        self
    }

    pub fn convergence_criteria(mut self, criteria: ConvergenceCriteria<T>) -> Self {
        self.config.convergence_criteria = criteria;
        self
    }

    pub fn build(self) -> OptimizationCoordinator<T> {
        OptimizationCoordinator::new(self.config)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for CoordinatorBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = CoordinatorBuilder::<f64>::new()
            .max_concurrent_tasks(5)
            .enable_anomaly_detection(true)
            .build();

        let status = coordinator.get_status();
        assert_eq!(status.active_tasks, 0);
        assert_eq!(status.health_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_task_submission() {
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default());
        let task = OptimizationTask::new("test_task".to_string());
        
        let task_id = coordinator.submit_task(task).unwrap();
        assert!(!task_id.is_empty());
        
        let status = coordinator.get_status();
        assert!(status.active_tasks > 0);
    }

    #[test]
    fn test_monitoring() {
        let mut coordinator = OptimizationCoordinator::<f64>::new(CoordinatorConfig::default());
        let result = coordinator.monitor_optimization_value("test_task", 1.0);
        
        assert_eq!(result.task_id, "test_task");
        assert_eq!(result.value, 1.0);
    }
}