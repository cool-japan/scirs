//! CUDA execution pipeline management for optimizer kernels
//!
//! This module provides sophisticated pipeline management for CUDA kernel execution,
//! including asynchronous operations, multi-stream processing, dependency tracking,
//! and optimized resource utilization for machine learning optimizer workloads.

use crate::gpu::cuda_kernels::config::*;
use scirs2_core::error::{Result, ScirsMlError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream, CudaEvent, CudaFunction};

/// Pipeline execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStrategy {
    /// Sequential execution on single stream
    Sequential,
    /// Parallel execution on multiple streams
    MultiStream { stream_count: usize },
    /// Graph-based execution with dependency resolution
    GraphBased,
    /// Dynamic scheduling based on resource availability
    Adaptive,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Execution strategy
    pub strategy: PipelineStrategy,
    /// Maximum number of concurrent operations
    pub max_concurrent_ops: usize,
    /// Enable operation batching
    pub enable_batching: bool,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable kernel fusion
    pub enable_fusion: bool,
    /// Memory pool integration
    pub use_memory_pool: bool,
    /// Priority-based scheduling
    pub enable_priority_scheduling: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            strategy: PipelineStrategy::MultiStream { stream_count: 4 },
            max_concurrent_ops: 16,
            enable_batching: true,
            batch_timeout_ms: 10,
            max_batch_size: 32,
            enable_fusion: true,
            use_memory_pool: true,
            enable_priority_scheduling: true,
        }
    }
}

/// Operation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperationPriority {
    /// Low priority background operations
    Low = 0,
    /// Normal priority operations
    Normal = 1,
    /// High priority operations
    High = 2,
    /// Critical operations that must execute immediately
    Critical = 3,
}

impl Default for OperationPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Pipeline operation descriptor
#[derive(Debug, Clone)]
pub struct PipelineOperation {
    /// Unique operation ID
    pub id: u64,
    /// Operation type identifier
    pub op_type: String,
    /// Priority level
    pub priority: OperationPriority,
    /// Input dependencies (operation IDs)
    pub dependencies: Vec<u64>,
    /// Estimated execution time in microseconds
    pub estimated_time_us: u64,
    /// Memory requirements in bytes
    pub memory_requirement: usize,
    /// Kernel function to execute
    #[cfg(feature = "cuda")]
    pub kernel: Option<CudaFunction>,
    /// Kernel parameters
    pub parameters: Vec<KernelParameter>,
    /// Grid and block dimensions
    pub launch_config: LaunchConfig,
    /// Stream hint for execution
    pub stream_hint: Option<usize>,
    /// Completion callback
    pub completion_callback: Option<Arc<dyn Fn(Result<()>) + Send + Sync>>,
}

/// Kernel parameter wrapper
#[derive(Debug, Clone)]
pub enum KernelParameter {
    /// 32-bit integer parameter
    Int32(i32),
    /// 64-bit integer parameter
    Int64(i64),
    /// 32-bit float parameter
    Float32(f32),
    /// 64-bit float parameter
    Float64(f64),
    /// Raw pointer parameter
    Pointer(*const std::ffi::c_void),
    /// Mutable pointer parameter
    MutablePointer(*mut std::ffi::c_void),
}

unsafe impl Send for KernelParameter {}
unsafe impl Sync for KernelParameter {}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dims: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_memory: u32,
    /// Stream ID for execution
    pub stream_id: Option<usize>,
}

/// Operation execution state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationState {
    /// Operation is queued for execution
    Queued,
    /// Operation is ready to execute (dependencies satisfied)
    Ready,
    /// Operation is currently executing
    Executing,
    /// Operation completed successfully
    Completed,
    /// Operation failed with error
    Failed(String),
    /// Operation was cancelled
    Cancelled,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    /// Total operations processed
    pub total_operations: u64,
    /// Currently queued operations
    pub queued_operations: u64,
    /// Currently executing operations
    pub executing_operations: u64,
    /// Successfully completed operations
    pub completed_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average execution time per operation (microseconds)
    pub avg_execution_time_us: f64,
    /// Pipeline throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// Stream utilization per stream
    pub stream_utilization: HashMap<usize, f64>,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Average queue wait time
    pub avg_queue_wait_time_us: f64,
}

/// Asynchronous operation handle
pub struct OperationHandle {
    /// Operation ID
    operation_id: u64,
    /// Pipeline reference
    pipeline: Arc<CudaPipeline>,
    /// Completion waker
    waker: Arc<Mutex<Option<Waker>>>,
    /// Result storage
    result: Arc<Mutex<Option<Result<()>>>>,
}

impl Future for OperationHandle {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut waker = self.waker.lock().unwrap();
        let mut result = self.result.lock().unwrap();

        if let Some(res) = result.take() {
            Poll::Ready(res)
        } else {
            *waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// CUDA execution pipeline manager
pub struct CudaPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    /// CUDA streams for parallel execution
    #[cfg(feature = "cuda")]
    streams: Vec<Arc<CudaStream>>,
    /// Operation queue with priority ordering
    operation_queue: Arc<Mutex<VecDeque<PipelineOperation>>>,
    /// Currently executing operations
    executing_operations: Arc<RwLock<HashMap<u64, OperationState>>>,
    /// Dependency graph for operation scheduling
    dependency_graph: Arc<RwLock<HashMap<u64, Vec<u64>>>>,
    /// Pipeline statistics
    statistics: Arc<RwLock<PipelineStatistics>>,
    /// Next operation ID
    next_operation_id: Arc<Mutex<u64>>,
    /// Stream scheduler state
    stream_scheduler: Arc<Mutex<StreamScheduler>>,
    /// Operation completion handles
    completion_handles: Arc<Mutex<HashMap<u64, OperationHandle>>>,
    /// Pipeline shutdown flag
    shutdown: Arc<Mutex<bool>>,
    /// Worker thread handles
    worker_handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

/// Stream scheduler for managing CUDA stream allocation
struct StreamScheduler {
    /// Stream usage tracking (stream_id -> last_used_time)
    stream_usage: HashMap<usize, Instant>,
    /// Stream load estimation (stream_id -> estimated_remaining_work_us)
    stream_load: HashMap<usize, u64>,
    /// Round-robin counter for fair scheduling
    round_robin_counter: usize,
}

impl StreamScheduler {
    fn new(stream_count: usize) -> Self {
        let mut stream_usage = HashMap::new();
        let mut stream_load = HashMap::new();

        for i in 0..stream_count {
            stream_usage.insert(i, Instant::now());
            stream_load.insert(i, 0);
        }

        Self {
            stream_usage,
            stream_load,
            round_robin_counter: 0,
        }
    }

    /// Selects optimal stream for operation execution
    fn select_stream(&mut self, operation: &PipelineOperation) -> usize {
        // Check if operation has stream hint
        if let Some(hint) = operation.stream_hint {
            if hint < self.stream_usage.len() {
                return hint;
            }
        }

        // Find stream with minimum load
        let min_load_stream = self.stream_load.iter()
            .min_by_key(|(_, load)| *load)
            .map(|(stream_id, _)| *stream_id)
            .unwrap_or(0);

        // Update load estimation
        if let Some(load) = self.stream_load.get_mut(&min_load_stream) {
            *load += operation.estimated_time_us;
        }

        // Update usage timestamp
        self.stream_usage.insert(min_load_stream, Instant::now());

        min_load_stream
    }

    /// Updates stream load after operation completion
    fn update_stream_load(&mut self, stream_id: usize, actual_time_us: u64) {
        if let Some(load) = self.stream_load.get_mut(&stream_id) {
            *load = load.saturating_sub(actual_time_us);
        }
    }
}

impl CudaPipeline {
    /// Creates a new CUDA execution pipeline
    pub fn new(config: PipelineConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let (device, streams) = {
            let device = Arc::new(CudaDevice::new(0)?);
            let stream_count = match &config.strategy {
                PipelineStrategy::MultiStream { stream_count } => *stream_count,
                PipelineStrategy::Adaptive => 8, // Default for adaptive
                _ => 1,
            };

            let mut streams = Vec::new();
            for _ in 0..stream_count {
                streams.push(Arc::new(device.fork_default_stream()?));
            }

            (device, streams)
        };

        #[cfg(not(feature = "cuda"))]
        let streams = Vec::new();

        let stream_count = streams.len().max(1);
        let pipeline = Self {
            config,
            #[cfg(feature = "cuda")]
            device,
            #[cfg(feature = "cuda")]
            streams,
            operation_queue: Arc::new(Mutex::new(VecDeque::new())),
            executing_operations: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(PipelineStatistics::default())),
            next_operation_id: Arc::new(Mutex::new(1)),
            stream_scheduler: Arc::new(Mutex::new(StreamScheduler::new(stream_count))),
            completion_handles: Arc::new(Mutex::new(HashMap::new())),
            shutdown: Arc::new(Mutex::new(false)),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
        };

        // Start worker threads
        pipeline.start_workers()?;

        Ok(pipeline)
    }

    /// Starts worker threads for pipeline execution
    fn start_workers(&self) -> Result<()> {
        let worker_count = match &self.config.strategy {
            PipelineStrategy::Sequential => 1,
            PipelineStrategy::MultiStream { stream_count } => *stream_count,
            PipelineStrategy::GraphBased => 4,
            PipelineStrategy::Adaptive => 6,
        };

        let mut handles = self.worker_handles.lock().unwrap();

        for worker_id in 0..worker_count {
            let pipeline_clone = Arc::new(self.clone());
            let handle = std::thread::Builder::new()
                .name(format!("cuda-pipeline-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_thread(pipeline_clone, worker_id);
                })?;
            handles.push(handle);
        }

        Ok(())
    }

    /// Worker thread main loop
    fn worker_thread(pipeline: Arc<CudaPipeline>, worker_id: usize) {
        while !*pipeline.shutdown.lock().unwrap() {
            match pipeline.process_next_operation(worker_id) {
                Ok(processed) => {
                    if !processed {
                        // No operations available, sleep briefly
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
                Err(e) => {
                    eprintln!("Worker {} error: {}", worker_id, e);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Processes the next available operation
    fn process_next_operation(&self, worker_id: usize) -> Result<bool> {
        // Get next ready operation from queue
        let operation = {
            let mut queue = self.operation_queue.lock().unwrap();
            self.find_ready_operation(&mut queue)?
        };

        if let Some(mut op) = operation {
            // Mark operation as executing
            {
                let mut executing = self.executing_operations.write().unwrap();
                executing.insert(op.id, OperationState::Executing);
            }

            // Select stream for execution
            let stream_id = self.stream_scheduler.lock().unwrap()
                .select_stream(&op);

            // Execute operation
            let start_time = Instant::now();
            let result = self.execute_operation(&mut op, stream_id);
            let elapsed = start_time.elapsed();

            // Update stream load
            self.stream_scheduler.lock().unwrap()
                .update_stream_load(stream_id, elapsed.as_micros() as u64);

            // Complete operation
            self.complete_operation(op.id, result, elapsed)?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Finds the next ready operation in the queue
    fn find_ready_operation(&self, queue: &mut VecDeque<PipelineOperation>) -> Result<Option<PipelineOperation>> {
        // Sort queue by priority (highest first)
        let mut ops: Vec<PipelineOperation> = queue.drain(..).collect();
        ops.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Find first operation with satisfied dependencies
        let executing = self.executing_operations.read().unwrap();
        for (i, op) in ops.iter().enumerate() {
            if self.are_dependencies_satisfied(&op.dependencies, &executing)? {
                let ready_op = ops.remove(i);
                // Put remaining operations back in queue
                for remaining_op in ops {
                    queue.push_back(remaining_op);
                }
                return Ok(Some(ready_op));
            }
        }

        // Put all operations back in queue
        for op in ops {
            queue.push_back(op);
        }

        Ok(None)
    }

    /// Checks if operation dependencies are satisfied
    fn are_dependencies_satisfied(&self, dependencies: &[u64], executing: &HashMap<u64, OperationState>) -> Result<bool> {
        for &dep_id in dependencies {
            match executing.get(&dep_id) {
                Some(OperationState::Completed) => continue,
                Some(_) => return Ok(false), // Dependency not completed
                None => {
                    // Dependency not found - assume it was completed and cleaned up
                    continue;
                }
            }
        }
        Ok(true)
    }

    /// Executes a CUDA operation
    fn execute_operation(&self, operation: &mut PipelineOperation, stream_id: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let Some(kernel) = &operation.kernel {
                let stream = &self.streams[stream_id];

                // Convert parameters to CUDA format
                let cuda_params = self.convert_parameters(&operation.parameters)?;

                // Launch kernel
                unsafe {
                    kernel.launch_on_stream(
                        stream,
                        operation.launch_config.grid_dims,
                        operation.launch_config.block_dims,
                        operation.launch_config.shared_memory,
                        &cuda_params
                    )?;
                }

                // Synchronize stream if needed
                stream.synchronize()?;

                Ok(())
            } else {
                Err(ScirsMlError::InvalidArgument("No kernel provided for operation".into()))
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback - simulate work
            std::thread::sleep(Duration::from_micros(operation.estimated_time_us));
            Ok(())
        }
    }

    /// Converts kernel parameters to CUDA-compatible format
    #[cfg(feature = "cuda")]
    fn convert_parameters(&self, parameters: &[KernelParameter]) -> Result<Vec<Box<dyn std::any::Any>>> {
        let mut cuda_params = Vec::new();

        for param in parameters {
            match param {
                KernelParameter::Int32(val) => cuda_params.push(Box::new(*val) as Box<dyn std::any::Any>),
                KernelParameter::Int64(val) => cuda_params.push(Box::new(*val) as Box<dyn std::any::Any>),
                KernelParameter::Float32(val) => cuda_params.push(Box::new(*val) as Box<dyn std::any::Any>),
                KernelParameter::Float64(val) => cuda_params.push(Box::new(*val) as Box<dyn std::any::Any>),
                KernelParameter::Pointer(ptr) => cuda_params.push(Box::new(*ptr) as Box<dyn std::any::Any>),
                KernelParameter::MutablePointer(ptr) => cuda_params.push(Box::new(*ptr) as Box<dyn std::any::Any>),
            }
        }

        Ok(cuda_params)
    }

    /// Completes an operation and updates statistics
    fn complete_operation(&self, operation_id: u64, result: Result<()>, elapsed: Duration) -> Result<()> {
        // Update operation state
        {
            let mut executing = self.executing_operations.write().unwrap();
            let state = match result {
                Ok(()) => OperationState::Completed,
                Err(ref e) => OperationState::Failed(e.to_string()),
            };
            executing.insert(operation_id, state);
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_operations += 1;

            if result.is_ok() {
                stats.completed_operations += 1;
            } else {
                stats.failed_operations += 1;
            }

            // Update timing statistics
            let elapsed_us = elapsed.as_micros() as f64;
            let total_time = stats.avg_execution_time_us * (stats.total_operations - 1) as f64 + elapsed_us;
            stats.avg_execution_time_us = total_time / stats.total_operations as f64;

            // Update throughput
            stats.throughput_ops_per_sec = 1_000_000.0 / stats.avg_execution_time_us;
        }

        // Notify completion handle
        if let Some(handle) = self.completion_handles.lock().unwrap().get(&operation_id) {
            let mut result_storage = handle.result.lock().unwrap();
            *result_storage = Some(result.clone());

            if let Some(waker) = handle.waker.lock().unwrap().take() {
                waker.wake();
            }
        }

        // Execute completion callback if provided
        // Note: In real implementation, would need to store and retrieve callback

        Ok(())
    }

    /// Submits an operation to the pipeline
    pub fn submit_operation(&self, mut operation: PipelineOperation) -> Result<OperationHandle> {
        // Assign operation ID
        let operation_id = {
            let mut next_id = self.next_operation_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        operation.id = operation_id;

        // Create completion handle
        let handle = OperationHandle {
            operation_id,
            pipeline: Arc::new(self.clone()),
            waker: Arc::new(Mutex::new(None)),
            result: Arc::new(Mutex::new(None)),
        };

        // Store handle
        {
            let mut handles = self.completion_handles.lock().unwrap();
            handles.insert(operation_id, handle);
        }

        // Add to dependency graph
        if !operation.dependencies.is_empty() {
            let mut graph = self.dependency_graph.write().unwrap();
            graph.insert(operation_id, operation.dependencies.clone());
        }

        // Add to queue
        {
            let mut queue = self.operation_queue.lock().unwrap();

            // Insert in priority order
            let mut inserted = false;
            for (i, queued_op) in queue.iter().enumerate() {
                if operation.priority > queued_op.priority {
                    queue.insert(i, operation);
                    inserted = true;
                    break;
                }
            }

            if !inserted {
                queue.push_back(operation);
            }

            // Update statistics
            let mut stats = self.statistics.write().unwrap();
            stats.queued_operations += 1;
        }

        Ok(self.completion_handles.lock().unwrap().get(&operation_id).unwrap().clone())
    }

    /// Submits a batch of operations
    pub fn submit_batch(&self, operations: Vec<PipelineOperation>) -> Result<Vec<OperationHandle>> {
        let mut handles = Vec::new();

        for operation in operations {
            let handle = self.submit_operation(operation)?;
            handles.push(handle);
        }

        Ok(handles)
    }

    /// Cancels a pending operation
    pub fn cancel_operation(&self, operation_id: u64) -> Result<()> {
        // Remove from queue
        {
            let mut queue = self.operation_queue.lock().unwrap();
            if let Some(pos) = queue.iter().position(|op| op.id == operation_id) {
                queue.remove(pos);

                // Update statistics
                let mut stats = self.statistics.write().unwrap();
                stats.queued_operations = stats.queued_operations.saturating_sub(1);

                return Ok(());
            }
        }

        // Mark executing operation as cancelled
        {
            let mut executing = self.executing_operations.write().unwrap();
            if executing.contains_key(&operation_id) {
                executing.insert(operation_id, OperationState::Cancelled);
                return Ok(());
            }
        }

        Err(ScirsMlError::InvalidArgument("Operation not found".into()))
    }

    /// Gets current pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Flushes the pipeline and waits for all operations to complete
    pub fn flush(&self) -> Result<()> {
        // Wait for queue to empty and all operations to complete
        loop {
            let queue_size = self.operation_queue.lock().unwrap().len();
            let executing_count = self.executing_operations.read().unwrap().len();

            if queue_size == 0 && executing_count == 0 {
                break;
            }

            std::thread::sleep(Duration::from_millis(1));
        }

        Ok(())
    }

    /// Shuts down the pipeline gracefully
    pub fn shutdown(&self) -> Result<()> {
        // Set shutdown flag
        *self.shutdown.lock().unwrap() = true;

        // Wait for workers to finish
        let mut handles = self.worker_handles.lock().unwrap();
        while let Some(handle) = handles.pop() {
            handle.join().map_err(|_| ScirsMlError::RuntimeError("Failed to join worker thread".into()))?;
        }

        Ok(())
    }

    /// Generates pipeline performance report
    pub fn generate_report(&self) -> PipelineReport {
        let statistics = self.get_statistics();
        let queue_size = self.operation_queue.lock().unwrap().len();
        let executing_count = self.executing_operations.read().unwrap().len();

        PipelineReport {
            statistics,
            current_queue_size: queue_size,
            current_executing_count: executing_count,
            stream_count: self.streams.len(),
            recommendations: self.generate_recommendations(&statistics),
        }
    }

    /// Generates optimization recommendations
    fn generate_recommendations(&self, stats: &PipelineStatistics) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check throughput
        if stats.throughput_ops_per_sec < 100.0 {
            recommendations.push("Low throughput detected - consider increasing parallelism or optimizing kernels".to_string());
        }

        // Check failure rate
        let failure_rate = if stats.total_operations > 0 {
            (stats.failed_operations as f64 / stats.total_operations as f64) * 100.0
        } else {
            0.0
        };

        if failure_rate > 5.0 {
            recommendations.push(format!("High failure rate ({:.1}%) - investigate kernel errors", failure_rate));
        }

        // Check queue wait times
        if stats.avg_queue_wait_time_us > 10000.0 {
            recommendations.push("High queue wait times - consider increasing worker threads or stream count".to_string());
        }

        recommendations
    }
}

impl Clone for CudaPipeline {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            #[cfg(feature = "cuda")]
            device: Arc::clone(&self.device),
            #[cfg(feature = "cuda")]
            streams: self.streams.clone(),
            operation_queue: Arc::clone(&self.operation_queue),
            executing_operations: Arc::clone(&self.executing_operations),
            dependency_graph: Arc::clone(&self.dependency_graph),
            statistics: Arc::clone(&self.statistics),
            next_operation_id: Arc::clone(&self.next_operation_id),
            stream_scheduler: Arc::clone(&self.stream_scheduler),
            completion_handles: Arc::clone(&self.completion_handles),
            shutdown: Arc::clone(&self.shutdown),
            worker_handles: Arc::clone(&self.worker_handles),
        }
    }
}

impl Clone for OperationHandle {
    fn clone(&self) -> Self {
        Self {
            operation_id: self.operation_id,
            pipeline: Arc::clone(&self.pipeline),
            waker: Arc::clone(&self.waker),
            result: Arc::clone(&self.result),
        }
    }
}

/// Pipeline performance report
#[derive(Debug, Clone)]
pub struct PipelineReport {
    /// Current statistics
    pub statistics: PipelineStatistics,
    /// Current queue size
    pub current_queue_size: usize,
    /// Currently executing operations
    pub current_executing_count: usize,
    /// Number of streams
    pub stream_count: usize,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl PipelineReport {
    /// Formats the report as human-readable text
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== CUDA Pipeline Performance Report ===\n\n");

        // Current state
        report.push_str("Current State:\n");
        report.push_str(&format!("  Queue Size: {}\n", self.current_queue_size));
        report.push_str(&format!("  Executing: {}\n", self.current_executing_count));
        report.push_str(&format!("  Stream Count: {}\n", self.stream_count));
        report.push_str("\n");

        // Performance metrics
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!("  Total Operations: {}\n", self.statistics.total_operations));
        report.push_str(&format!("  Completed: {}\n", self.statistics.completed_operations));
        report.push_str(&format!("  Failed: {}\n", self.statistics.failed_operations));
        report.push_str(&format!("  Throughput: {:.2} ops/sec\n", self.statistics.throughput_ops_per_sec));
        report.push_str(&format!("  Avg Execution Time: {:.2} μs\n", self.statistics.avg_execution_time_us));
        report.push_str(&format!("  Avg Queue Wait: {:.2} μs\n", self.statistics.avg_queue_wait_time_us));
        report.push_str("\n");

        // Recommendations
        if !self.recommendations.is_empty() {
            report.push_str("Recommendations:\n");
            for recommendation in &self.recommendations {
                report.push_str(&format!("  • {}\n", recommendation));
            }
        }

        report
    }
}