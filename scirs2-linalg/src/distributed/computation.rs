//! Distributed computation engine for linear algebra operations
//!
//! This module provides the core computation engine that orchestrates
//! distributed linear algebra operations, integrating SIMD acceleration,
//! load balancing, and fault tolerance.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::communication::DistributedCommunicator;
use super::coordination::DistributedCoordinator;
use super::distribution::LoadBalancer;
use super::matrix::DistributedMatrix;

/// Distributed computation engine
pub struct DistributedComputationEngine {
    /// Communicator for inter-node communication
    communicator: Arc<DistributedCommunicator>,
    /// Coordinator for synchronization
    coordinator: Arc<DistributedCoordinator>,
    /// Load balancer
    load_balancer: Arc<std::sync::Mutex<LoadBalancer>>,
    /// Configuration
    config: super::DistributedConfig,
    /// Performance metrics
    metrics: Arc<std::sync::Mutex<ComputationMetrics>>,
}

impl DistributedComputationEngine {
    /// Create a new distributed computation engine
    pub fn new(config: super::DistributedConfig) -> LinalgResult<Self> {
        let communicator = Arc::new(DistributedCommunicator::new(&config)?);
        let coordinator = Arc::new(DistributedCoordinator::new(&config)?);
        let load_balancer = Arc::new(std::sync::Mutex::new(LoadBalancer::new(&config)?));
        let metrics = Arc::new(std::sync::Mutex::new(ComputationMetrics::new()));
        
        Ok(Self {
            communicator,
            coordinator,
            load_balancer,
            config,
            metrics,
        })
    }
    
    /// Execute distributed matrix multiplication with optimization
    pub fn execute_matmul<T>(
        &self,
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
    {
        let start_time = Instant::now();
        
        // Pre-computation optimization
        self.optimize_computation_layout(a, b)?;
        
        // Execute the multiplication
        let result = if self.config.enable_simd {
            self.execute_simd_matmul(a, b)?
        } else {
            a.multiply(b)?
        };
        
        // Post-computation cleanup and metrics
        let elapsed = start_time.elapsed();
        self.record_computation_metrics("matmul", elapsed);
        
        Ok(result)
    }
    
    /// Execute distributed matrix operation with load balancing
    pub fn execute_with_load_balancing<T, F, R>(
        &self,
        operation: F,
        inputs: &[&DistributedMatrix<T>],
    ) -> LinalgResult<R>
    where
        T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
        F: Fn(&[&DistributedMatrix<T>]) -> LinalgResult<R> + Send + Sync,
        R: Send + Sync,
    {
        let start_time = Instant::now();
        
        // Check load balance
        let mut load_balancer = self.load_balancer.lock().unwrap();
        if let Some(plan) = load_balancer.suggest_redistribution() {
            // Implement redistribution if needed
            drop(load_balancer);
            self.implement_redistribution(plan)?;
        } else {
            drop(load_balancer);
        }
        
        // Execute operation
        let result = operation(inputs)?;
        
        // Record performance
        let elapsed = start_time.elapsed();
        let mut load_balancer = self.load_balancer.lock().unwrap();
        load_balancer.record_workload(self.config.node_rank, elapsed.as_millis() as f64);
        
        Ok(result)
    }
    
    /// Execute computation with fault tolerance
    pub fn execute_with_fault_tolerance<T, F, R>(
        &self,
        operation: F,
        max_retries: usize,
    ) -> LinalgResult<R>
    where
        F: Fn() -> LinalgResult<R> + Send + Sync,
        R: Send + Sync,
    {
        let mut retries = 0;
        
        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if retries >= max_retries {
                        return Err(e);
                    }
                    
                    // Handle potential node failure
                    if let LinalgError::CommunicationError(_) = e {
                        self.handle_communication_failure()?;
                    }
                    
                    retries += 1;
                    std::thread::sleep(Duration::from_millis(100 * retries as u64));
                }
            }
        }
    }
    
    /// Execute parallel computation across multiple operations
    pub fn execute_parallel<T, F, R>(
        &self,
        operations: Vec<F>,
    ) -> LinalgResult<Vec<R>>
    where
        F: Fn() -> LinalgResult<R> + Send + Sync + 'static,
        R: Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        use rayon::prelude::*;
        
        // Execute operations in parallel using rayon
        let results: Result<Vec<R>, LinalgError> = operations
            .into_par_iter()
            .map(|op| op())
            .collect();
        
        results
    }
    
    /// Optimize memory usage for computation
    pub fn optimize_memory_usage<T>(&self, matrices: &[&DistributedMatrix<T>]) -> LinalgResult<()>
    where
        T: Float + Send + Sync,
    {
        // Calculate memory requirements
        let total_memory: usize = matrices
            .iter()
            .map(|m| {
                let (rows, cols) = m.local_shape();
                rows * cols * std::mem::size_of::<T>()
            })
            .sum();
        
        // Check against memory limit
        if let Some(limit) = self.config.memory_limit_bytes {
            if total_memory > limit {
                return Err(LinalgError::MemoryError(format!(
                    "Memory usage {} exceeds limit {}",
                    total_memory, limit
                )));
            }
        }
        
        // Implement memory optimization strategies
        self.implement_memory_optimization(total_memory)?;
        
        Ok(())
    }
    
    /// Get computation performance metrics
    pub fn get_metrics(&self) -> ComputationMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = ComputationMetrics::new();
    }
    
    // Private helper methods
    
    fn optimize_computation_layout<T>(
        &self,
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<()>
    where
        T: Float,
    {
        // Analyze matrix shapes and distributions for optimization opportunities
        let (m, k) = a.global_shape();
        let (k2, n) = b.global_shape();
        
        // Check if redistribution would be beneficial
        let computation_cost = m * k * n;
        let communication_cost = (m * k + k * n) * self.config.num_nodes;
        
        if communication_cost < computation_cost / 10 {
            // Communication cost is low relative to computation, consider redistribution
            // This is a simplified heuristic - real implementation would be more sophisticated
        }
        
        Ok(())
    }
    
    fn execute_simd_matmul<T>(
        &self,
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: Float + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static,
    {
        // Use SIMD-accelerated GEMM for local computations
        a.gemm_simd(b, T::one(), T::zero())
    }
    
    fn implement_redistribution(
        &self,
        _plan: super::distribution::RedistributionPlan,
    ) -> LinalgResult<()> {
        // Implement data redistribution based on the plan
        // This would involve:
        // 1. Coordinating with other nodes
        // 2. Transferring data
        // 3. Updating local distributions
        
        self.coordinator.barrier()?;
        Ok(())
    }
    
    fn handle_communication_failure(&self) -> LinalgResult<()> {
        // Detect failed nodes
        // Implement recovery strategy
        // Update computation topology
        
        // For now, just synchronize remaining nodes
        self.coordinator.barrier()
    }
    
    fn implement_memory_optimization(&self, total_memory: usize) -> LinalgResult<()> {
        // Implement memory optimization strategies:
        // 1. Data compression
        // 2. Out-of-core computation
        // 3. Memory-efficient algorithms
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.peak_memory_usage = metrics.peak_memory_usage.max(total_memory);
        
        Ok(())
    }
    
    fn record_computation_metrics(&self, operation: &str, duration: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.operation_count += 1;
        metrics.total_computation_time += duration;
        metrics.operations.insert(operation.to_string(), 
            metrics.operations.get(operation).unwrap_or(&0) + 1);
    }
}

/// Performance metrics for distributed computations
#[derive(Debug, Clone)]
pub struct ComputationMetrics {
    /// Total number of operations performed
    pub operation_count: usize,
    /// Total computation time
    pub total_computation_time: Duration,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Operations by type
    pub operations: std::collections::HashMap<String, usize>,
    /// Load balancing efficiency
    pub load_balance_efficiency: f64,
    /// Communication overhead ratio
    pub communication_overhead: f64,
    /// SIMD utilization ratio
    pub simd_utilization: f64,
}

impl ComputationMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            operation_count: 0,
            total_computation_time: Duration::default(),
            peak_memory_usage: 0,
            operations: std::collections::HashMap::new(),
            load_balance_efficiency: 1.0,
            communication_overhead: 0.0,
            simd_utilization: 0.0,
        }
    }
    
    /// Calculate average operation time
    pub fn avg_operation_time(&self) -> Duration {
        if self.operation_count > 0 {
            self.total_computation_time / self.operation_count as u32
        } else {
            Duration::default()
        }
    }
    
    /// Calculate operations per second
    pub fn operations_per_second(&self) -> f64 {
        if self.total_computation_time.as_secs_f64() > 0.0 {
            self.operation_count as f64 / self.total_computation_time.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Get memory efficiency (operations per byte)
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_memory_usage > 0 {
            self.operation_count as f64 / self.peak_memory_usage as f64
        } else {
            0.0
        }
    }
}

impl Default for ComputationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Computation scheduling strategies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    /// First-come, first-served
    FCFS,
    /// Shortest job first
    SJF,
    /// Priority-based scheduling
    Priority,
    /// Load-balanced scheduling
    LoadBalanced,
}

/// Computation scheduler for managing distributed operations
pub struct ComputationScheduler {
    /// Scheduling strategy
    strategy: SchedulingStrategy,
    /// Operation queue
    operation_queue: std::sync::Mutex<std::collections::VecDeque<ScheduledOperation>>,
    /// Current node capabilities
    node_capabilities: std::collections::HashMap<usize, f64>,
}

impl ComputationScheduler {
    /// Create a new computation scheduler
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            operation_queue: std::sync::Mutex::new(std::collections::VecDeque::new()),
            node_capabilities: std::collections::HashMap::new(),
        }
    }
    
    /// Schedule an operation for execution
    pub fn schedule_operation(&self, operation: ScheduledOperation) {
        let mut queue = self.operation_queue.lock().unwrap();
        
        match self.strategy {
            SchedulingStrategy::FCFS => {
                queue.push_back(operation);
            }
            SchedulingStrategy::SJF => {
                // Insert in order of estimated duration
                let position = queue
                    .iter()
                    .position(|op| op.estimated_duration > operation.estimated_duration)
                    .unwrap_or(queue.len());
                queue.insert(position, operation);
            }
            SchedulingStrategy::Priority => {
                // Insert in order of priority
                let position = queue
                    .iter()
                    .position(|op| op.priority < operation.priority)
                    .unwrap_or(queue.len());
                queue.insert(position, operation);
            }
            SchedulingStrategy::LoadBalanced => {
                // Consider load balancing when scheduling
                queue.push_back(operation);
            }
        }
    }
    
    /// Get next operation to execute
    pub fn next_operation(&self) -> Option<ScheduledOperation> {
        let mut queue = self.operation_queue.lock().unwrap();
        queue.pop_front()
    }
    
    /// Update node capability
    pub fn update_capability(&mut self, node_rank: usize, capability: f64) {
        self.node_capabilities.insert(node_rank, capability);
    }
}

/// A scheduled operation
#[derive(Debug, Clone)]
pub struct ScheduledOperation {
    /// Operation identifier
    pub id: String,
    /// Operation type
    pub operation_type: String,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Priority level (higher = more important)
    pub priority: i32,
    /// Required memory
    pub memory_requirement: usize,
    /// Target nodes for execution
    pub target_nodes: Vec<usize>,
}

impl ScheduledOperation {
    /// Create a new scheduled operation
    pub fn new(
        id: String,
        operation_type: String,
        estimated_duration: Duration,
        priority: i32,
    ) -> Self {
        Self {
            id,
            operation_type,
            estimated_duration,
            priority,
            memory_requirement: 0,
            target_nodes: Vec::new(),
        }
    }
    
    /// Set memory requirement
    pub fn with_memory_requirement(mut self, bytes: usize) -> Self {
        self.memory_requirement = bytes;
        self
    }
    
    /// Set target nodes
    pub fn with_target_nodes(mut self, nodes: Vec<usize>) -> Self {
        self.target_nodes = nodes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{DistributedConfig, DistributionStrategy};
    
    #[test]
    fn test_computation_engine_creation() {
        let config = DistributedConfig::default()
            .with_num_nodes(2)
            .with_node_rank(0);
        
        let engine = DistributedComputationEngine::new(config).unwrap();
        let metrics = engine.get_metrics();
        
        assert_eq!(metrics.operation_count, 0);
        assert_eq!(metrics.total_computation_time, Duration::default());
    }
    
    #[test]
    fn test_computation_metrics() {
        let mut metrics = ComputationMetrics::new();
        
        metrics.operation_count = 10;
        metrics.total_computation_time = Duration::from_secs(5);
        metrics.peak_memory_usage = 1024;
        
        assert_eq!(metrics.avg_operation_time(), Duration::from_millis(500));
        assert_eq!(metrics.operations_per_second(), 2.0);
        assert!(metrics.memory_efficiency() > 0.0);
    }
    
    #[test]
    fn test_computation_scheduler() {
        let scheduler = ComputationScheduler::new(SchedulingStrategy::SJF);
        
        let op1 = ScheduledOperation::new(
            "op1".to_string(),
            "matmul".to_string(),
            Duration::from_secs(3),
            1,
        );
        
        let op2 = ScheduledOperation::new(
            "op2".to_string(),
            "transpose".to_string(),
            Duration::from_secs(1),
            2,
        );
        
        scheduler.schedule_operation(op1);
        scheduler.schedule_operation(op2);
        
        // With SJF, shorter operation should come first
        let next = scheduler.next_operation().unwrap();
        assert_eq!(next.id, "op2");
    }
    
    #[test]
    fn test_scheduled_operation_builder() {
        let op = ScheduledOperation::new(
            "test_op".to_string(),
            "matmul".to_string(),
            Duration::from_secs(2),
            5,
        )
        .with_memory_requirement(1024)
        .with_target_nodes(vec![0, 1, 2]);
        
        assert_eq!(op.memory_requirement, 1024);
        assert_eq!(op.target_nodes, vec![0, 1, 2]);
        assert_eq!(op.priority, 5);
    }
}