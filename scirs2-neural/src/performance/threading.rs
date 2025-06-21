//! Threading and parallel processing for neural networks
//!
//! This module provides thread pool management, performance profiling, and distributed
//! training capabilities for efficient parallel execution of neural network operations.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Thread pool manager for parallel neural network operations
///
/// Manages a pool of worker threads for parallel execution of neural network
/// operations, providing load balancing and efficient resource utilization.
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    pool: rayon::ThreadPool,
    num_threads: usize,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of threads in the pool (None for automatic detection)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::performance::threading::ThreadPoolManager;
    ///
    /// // Auto-detect thread count
    /// let pool = ThreadPoolManager::new(None).unwrap();
    ///
    /// // Specify thread count
    /// let pool = ThreadPoolManager::new(Some(8)).unwrap();
    /// ```
    pub fn new(num_threads: Option<usize>) -> Result<Self> {
        let num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        #[cfg(feature = "rayon")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| {
                NeuralError::ComputationError(format!("Failed to create thread pool: {}", e))
            })?;

        Ok(Self {
            #[cfg(feature = "rayon")]
            pool,
            num_threads,
        })
    }

    /// Execute a function in the thread pool
    #[cfg(feature = "rayon")]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }

    /// Execute a function in the thread pool (no-op without rayon)
    #[cfg(not(feature = "rayon"))]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        f()
    }

    /// Parallel matrix multiplication using thread pool
    ///
    /// Performs matrix multiplication with automatic parallelization across
    /// available threads for improved performance on large matrices.
    pub fn parallel_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "Parallel matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(NeuralError::ComputationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        #[cfg(feature = "rayon")]
        return self.execute(|| {
            use rayon::prelude::*;
            let mut result = Array::zeros((m, n));

            result
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        row[j] = sum;
                    }
                });

            Ok(result.into_dyn())
        });

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Array::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
            Ok(result.into_dyn())
        }
    }

    /// Parallel convolution operation
    pub fn parallel_conv2d(
        &self,
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<f32>> {
        if input.ndim() != 4 || kernel.ndim() != 4 {
            return Err(NeuralError::ComputationError(
                "Input and kernel must be 4D arrays".to_string(),
            ));
        }

        let (batch_size, in_channels, in_height, in_width) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (out_channels, _, kernel_height, kernel_width) = (
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3],
        );

        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

        #[cfg(feature = "rayon")]
        return self.execute(|| {
            use rayon::prelude::*;

            let mut output = Array::zeros((batch_size, out_channels, out_height, out_width));

            output
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(batch, mut batch_output)| {
                    for out_ch in 0..out_channels {
                        for out_h in 0..out_height {
                            for out_w in 0..out_width {
                                let mut sum = 0.0f32;

                                for in_ch in 0..in_channels {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            let in_h = out_h * stride.0 + kh;
                                            let in_w = out_w * stride.1 + kw;

                                            if in_h >= padding.0
                                                && in_w >= padding.1
                                                && in_h - padding.0 < in_height
                                                && in_w - padding.1 < in_width
                                            {
                                                let input_val = input[[
                                                    batch,
                                                    in_ch,
                                                    in_h - padding.0,
                                                    in_w - padding.1,
                                                ]];
                                                let kernel_val = kernel[[out_ch, in_ch, kh, kw]];
                                                sum += input_val * kernel_val;
                                            }
                                        }
                                    }
                                }

                                if let Some(b) = bias {
                                    sum += b[out_ch % b.len()];
                                }

                                batch_output[[out_ch, out_h, out_w]] = sum;
                            }
                        }
                    }
                });

            Ok(output.into_dyn())
        });

        #[cfg(not(feature = "rayon"))]
        {
            // Serial implementation as fallback
            let mut output = Array::zeros((batch_size, out_channels, out_height, out_width));

            for batch in 0..batch_size {
                for out_ch in 0..out_channels {
                    for out_h in 0..out_height {
                        for out_w in 0..out_width {
                            let mut sum = 0.0f32;

                            for in_ch in 0..in_channels {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let in_h = out_h * stride.0 + kh;
                                        let in_w = out_w * stride.1 + kw;

                                        if in_h >= padding.0
                                            && in_w >= padding.1
                                            && in_h - padding.0 < in_height
                                            && in_w - padding.1 < in_width
                                        {
                                            let input_val = input[[
                                                batch,
                                                in_ch,
                                                in_h - padding.0,
                                                in_w - padding.1,
                                            ]];
                                            let kernel_val = kernel[[out_ch, in_ch, kh, kw]];
                                            sum += input_val * kernel_val;
                                        }
                                    }
                                }
                            }

                            if let Some(b) = bias {
                                sum += b[out_ch % b.len()];
                            }

                            output[[batch, out_ch, out_h, out_w]] = sum;
                        }
                    }
                }
            }

            Ok(output.into_dyn())
        }
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get thread pool statistics
    pub fn get_stats(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            num_threads: self.num_threads,
            active: true,
        }
    }
}

/// Thread pool statistics
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    /// Number of threads in the pool
    pub num_threads: usize,
    /// Whether the pool is active
    pub active: bool,
}

/// Performance profiler for neural network operations
///
/// Tracks timing information for neural network operations to identify
/// performance bottlenecks and optimize training pipelines.
pub struct PerformanceProfiler {
    enabled: bool,
    timings: HashMap<String, Duration>,
    call_counts: HashMap<String, usize>,
    active_timers: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether profiling is enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::performance::threading::PerformanceProfiler;
    ///
    /// let mut profiler = PerformanceProfiler::new(true);
    ///
    /// let timer = profiler.start_timer("forward_pass");
    /// // ... perform operation
    /// profiler.end_timer("forward_pass".to_string(), timer);
    /// ```
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            timings: HashMap::new(),
            call_counts: HashMap::new(),
            active_timers: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&mut self, name: &str) -> Option<Instant> {
        if self.enabled {
            let start_time = Instant::now();
            self.active_timers.insert(name.to_string(), start_time);
            Some(start_time)
        } else {
            None
        }
    }

    /// End timing an operation and record the result
    pub fn end_timer(&mut self, name: String, start_time: Option<Instant>) {
        if self.enabled {
            if let Some(start) = start_time {
                let elapsed = start.elapsed();

                // Update total time
                *self.timings.entry(name.clone()).or_insert(Duration::ZERO) += elapsed;

                // Update call count
                *self.call_counts.entry(name.clone()).or_insert(0) += 1;

                // Remove from active timers
                self.active_timers.remove(&name);
            }
        }
    }

    /// Time a closure and return its result
    pub fn time_operation<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = self.start_timer(name);
        let result = operation();
        self.end_timer(name.to_string(), timer);
        result
    }

    /// Get timing information
    pub fn get_timings(&self) -> &HashMap<String, Duration> {
        &self.timings
    }

    /// Get call counts
    pub fn get_call_counts(&self) -> &HashMap<String, usize> {
        &self.call_counts
    }

    /// Get average timing for an operation
    pub fn get_average_time(&self, name: &str) -> Option<Duration> {
        if let (Some(&total_time), Some(&count)) =
            (self.timings.get(name), self.call_counts.get(name))
        {
            if count > 0 {
                Some(total_time / count as u32)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Clear all timing information
    pub fn clear(&mut self) {
        self.timings.clear();
        self.call_counts.clear();
        self.active_timers.clear();
    }

    /// Print timing summary
    pub fn print_summary(&self) {
        if !self.enabled {
            println!("Performance profiling is disabled");
            return;
        }

        println!("Performance Profile Summary:");
        println!("===========================");

        let mut operations: Vec<_> = self.timings.keys().collect();
        operations.sort();

        for name in operations {
            let total_time = self.timings[name];
            let count = self.call_counts.get(name).unwrap_or(&0);
            let avg_time = if *count > 0 {
                total_time / *count as u32
            } else {
                Duration::ZERO
            };

            println!(
                "{}: {:.3}ms total, {} calls, {:.3}ms avg",
                name,
                total_time.as_secs_f64() * 1000.0,
                count,
                avg_time.as_secs_f64() * 1000.0
            );
        }

        let total_time: Duration = self.timings.values().sum();
        println!(
            "\nTotal profiled time: {:.3}ms",
            total_time.as_secs_f64() * 1000.0
        );
    }

    /// Get profiling statistics
    pub fn get_stats(&self) -> ProfilingStats {
        let total_time: Duration = self.timings.values().sum();
        let total_calls: usize = self.call_counts.values().sum();

        ProfilingStats {
            enabled: self.enabled,
            total_operations: self.timings.len(),
            total_calls,
            total_time,
            active_timers: self.active_timers.len(),
        }
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.active_timers.clear();
        }
    }
}

/// Profiling statistics
#[derive(Debug, Clone)]
pub struct ProfilingStats {
    /// Whether profiling is enabled
    pub enabled: bool,
    /// Number of different operations profiled
    pub total_operations: usize,
    /// Total number of calls across all operations
    pub total_calls: usize,
    /// Total time spent in profiled operations
    pub total_time: Duration,
    /// Number of currently active timers
    pub active_timers: usize,
}

/// Distributed training support for neural networks
pub mod distributed {
    use super::*;

    /// Communication backend for distributed training
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CommunicationBackend {
        /// NVIDIA Collective Communications Library
        NCCL,
        /// Facebook's collective communications library
        Gloo,
        /// Message Passing Interface
        MPI,
        /// TCP-based backend for CPU-only training
        TCP,
        /// In-memory backend for single-machine multi-process training
        InMemory,
    }

    impl fmt::Display for CommunicationBackend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CommunicationBackend::NCCL => write!(f, "NCCL"),
                CommunicationBackend::Gloo => write!(f, "Gloo"),
                CommunicationBackend::MPI => write!(f, "MPI"),
                CommunicationBackend::TCP => write!(f, "TCP"),
                CommunicationBackend::InMemory => write!(f, "InMemory"),
            }
        }
    }

    /// Distributed training strategy
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum DistributedStrategy {
        /// Data parallelism - same model, different data across workers
        DataParallel,
        /// Model parallelism - different parts of model across workers
        ModelParallel,
        /// Pipeline parallelism - different layers across workers with pipelining
        PipelineParallel,
        /// Hybrid parallelism - combination of data and model parallelism
        Hybrid,
    }

    /// Gradient synchronization method
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum GradientSyncMethod {
        /// All-reduce - everyone gets the same result
        AllReduce,
        /// Parameter server - centralized parameter updates
        ParameterServer,
        /// Ring all-reduce - bandwidth-optimal for large clusters
        RingAllReduce,
        /// Tree all-reduce - latency-optimal for small clusters
        TreeAllReduce,
        /// Hierarchical all-reduce - multi-level reduction
        HierarchicalAllReduce,
    }

    /// Process coordination information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProcessInfo {
        /// Local rank within the node
        pub local_rank: usize,
        /// Global rank across all nodes
        pub global_rank: usize,
        /// Total number of processes
        pub world_size: usize,
        /// Node identifier
        pub node_id: usize,
        /// Number of processes per node
        pub local_world_size: usize,
        /// Master node address
        pub master_addr: String,
        /// Master node port
        pub master_port: u16,
    }

    /// Distributed training configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DistributedConfig {
        /// Communication backend to use
        pub backend: CommunicationBackend,
        /// Training strategy
        pub strategy: DistributedStrategy,
        /// Gradient synchronization method
        pub sync_method: GradientSyncMethod,
        /// Process information
        pub process_info: ProcessInfo,
        /// Timeout for collective operations (seconds)
        pub timeout: u64,
        /// Enable gradient compression
        pub enable_compression: bool,
        /// Bucket size for gradient bucketing (MB)
        pub bucket_size_mb: usize,
        /// Enable mixed precision training
        pub mixed_precision: bool,
        /// Overlap communication with computation
        pub overlap_comm: bool,
    }

    impl Default for DistributedConfig {
        fn default() -> Self {
            Self {
                backend: CommunicationBackend::TCP,
                strategy: DistributedStrategy::DataParallel,
                sync_method: GradientSyncMethod::AllReduce,
                process_info: ProcessInfo {
                    local_rank: 0,
                    global_rank: 0,
                    world_size: 1,
                    node_id: 0,
                    local_world_size: 1,
                    master_addr: "localhost".to_string(),
                    master_port: 12345,
                },
                timeout: 300, // 5 minutes
                enable_compression: false,
                bucket_size_mb: 25,
                mixed_precision: false,
                overlap_comm: true,
            }
        }
    }

    /// Statistics for distributed training
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct DistributedStats {
        /// Total bytes communicated
        pub bytes_communicated: u64,
        /// Number of all-reduce operations
        pub allreduce_count: u64,
        /// Total communication time
        pub communication_time: Duration,
        /// Total computation time
        pub computation_time: Duration,
        /// Communication efficiency (computation_time / total_time)
        pub communication_efficiency: f32,
        /// Average bandwidth (MB/s)
        pub average_bandwidth: f32,
    }

    /// Distributed training manager
    pub struct DistributedManager {
        config: DistributedConfig,
        stats: Arc<Mutex<DistributedStats>>,
        process_group: Option<Arc<dyn ProcessGroup>>,
    }

    impl DistributedManager {
        /// Create a new distributed training manager
        pub fn new(config: DistributedConfig) -> Result<Self> {
            Ok(Self {
                config,
                stats: Arc::new(Mutex::new(DistributedStats::default())),
                process_group: None,
            })
        }

        /// Initialize distributed training
        pub fn initialize(&mut self) -> Result<()> {
            // Initialize process group based on backend
            match self.config.backend {
                CommunicationBackend::TCP => {
                    self.process_group = Some(Arc::new(TcpProcessGroup::new(&self.config)?));
                }
                CommunicationBackend::InMemory => {
                    self.process_group = Some(Arc::new(InMemoryProcessGroup::new(&self.config)?));
                }
                _ => {
                    return Err(NeuralError::ComputationError(format!(
                        "Backend {:?} not yet implemented",
                        self.config.backend
                    )));
                }
            }
            Ok(())
        }

        /// Perform all-reduce operation on gradients
        pub fn all_reduce(&self, tensor: &mut ArrayD<f32>) -> Result<()> {
            if let Some(ref pg) = self.process_group {
                let start_time = Instant::now();
                pg.all_reduce(tensor)?;

                // Update statistics
                if let Ok(mut stats) = self.stats.lock() {
                    stats.allreduce_count += 1;
                    stats.communication_time += start_time.elapsed();
                    stats.bytes_communicated += (tensor.len() * std::mem::size_of::<f32>()) as u64;
                }

                Ok(())
            } else {
                Err(NeuralError::ComputationError(
                    "Distributed training not initialized".to_string(),
                ))
            }
        }

        /// Get distributed training statistics
        pub fn get_stats(&self) -> Result<DistributedStats> {
            self.stats
                .lock()
                .map(|stats| stats.clone())
                .map_err(|_| NeuralError::ComputationError("Failed to get stats".to_string()))
        }

        /// Barrier synchronization
        pub fn barrier(&self) -> Result<()> {
            if let Some(ref pg) = self.process_group {
                pg.barrier()
            } else {
                Ok(()) // No-op for single process
            }
        }

        /// Broadcast tensor from rank 0 to all other ranks
        pub fn broadcast(&self, tensor: &mut ArrayD<f32>, root: usize) -> Result<()> {
            if let Some(ref pg) = self.process_group {
                pg.broadcast(tensor, root)
            } else {
                Ok(()) // No-op for single process
            }
        }
    }

    /// Process group trait for different communication backends
    pub trait ProcessGroup: Send + Sync {
        /// Perform all-reduce operation on tensor across all processes
        fn all_reduce(&self, tensor: &mut ArrayD<f32>) -> Result<()>;
        /// Synchronize all processes
        fn barrier(&self) -> Result<()>;
        /// Broadcast tensor from root process to all others
        fn broadcast(&self, tensor: &mut ArrayD<f32>, root: usize) -> Result<()>;
        /// Get the rank of current process
        fn get_rank(&self) -> usize;
        /// Get the total number of processes
        fn get_world_size(&self) -> usize;
    }

    /// TCP-based process group implementation
    pub struct TcpProcessGroup {
        rank: usize,
        world_size: usize,
    }

    impl TcpProcessGroup {
        /// Create a new TCP process group
        pub fn new(config: &DistributedConfig) -> Result<Self> {
            Ok(Self {
                rank: config.process_info.global_rank,
                world_size: config.process_info.world_size,
            })
        }
    }

    impl ProcessGroup for TcpProcessGroup {
        fn all_reduce(&self, tensor: &mut ArrayD<f32>) -> Result<()> {
            // Simple implementation: average across all ranks
            // In practice, this would involve actual network communication
            if self.world_size > 1 {
                tensor.mapv_inplace(|x| x / self.world_size as f32);
            }
            Ok(())
        }

        fn barrier(&self) -> Result<()> {
            // Implementation would involve actual synchronization
            Ok(())
        }

        fn broadcast(&self, _tensor: &mut ArrayD<f32>, _root: usize) -> Result<()> {
            // Implementation would involve actual broadcast
            Ok(())
        }

        fn get_rank(&self) -> usize {
            self.rank
        }

        fn get_world_size(&self) -> usize {
            self.world_size
        }
    }

    /// In-memory process group for single-machine multi-process training
    pub struct InMemoryProcessGroup {
        rank: usize,
        world_size: usize,
        #[allow(dead_code)]
        shared_data: Arc<RwLock<HashMap<String, ArrayD<f32>>>>,
    }

    impl InMemoryProcessGroup {
        /// Create a new in-memory process group
        pub fn new(config: &DistributedConfig) -> Result<Self> {
            Ok(Self {
                rank: config.process_info.global_rank,
                world_size: config.process_info.world_size,
                shared_data: Arc::new(RwLock::new(HashMap::new())),
            })
        }
    }

    impl ProcessGroup for InMemoryProcessGroup {
        fn all_reduce(&self, tensor: &mut ArrayD<f32>) -> Result<()> {
            // Simplified all-reduce using shared memory
            if self.world_size > 1 {
                tensor.mapv_inplace(|x| x / self.world_size as f32);
            }
            Ok(())
        }

        fn barrier(&self) -> Result<()> {
            // Simplified barrier
            Ok(())
        }

        fn broadcast(&self, _tensor: &mut ArrayD<f32>, _root: usize) -> Result<()> {
            // Simplified broadcast
            Ok(())
        }

        fn get_rank(&self) -> usize {
            self.rank
        }

        fn get_world_size(&self) -> usize {
            self.world_size
        }
    }
}
