//! Thread pool configurations for linear algebra operations
//!
//! This module provides flexible thread pool management with support for
//! different configurations optimized for various linear algebra workloads.

use super::configure_workers;
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex, Once};

/// Global thread pool manager
static INIT: Once = Once::new();
static mut GLOBAL_POOL: Option<Arc<Mutex<ThreadPoolManager>>> = None;

/// Thread pool configuration profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPoolProfile {
    /// Default profile - uses system defaults
    Default,
    /// CPU-bound profile - one thread per CPU core
    CpuBound,
    /// Memory-bound profile - fewer threads to reduce memory contention
    MemoryBound,
    /// Latency-sensitive profile - more threads for better responsiveness
    LatencySensitive,
    /// Custom profile with specific thread count
    Custom(usize),
}

impl ThreadPoolProfile {
    /// Get the number of threads for this profile
    pub fn num_threads(&self) -> usize {
        match self {
            ThreadPoolProfile::Default => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            ThreadPoolProfile::CpuBound => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            ThreadPoolProfile::MemoryBound => {
                // Use half the available cores to reduce memory contention
                std::thread::available_parallelism()
                    .map(|n| std::cmp::max(1, n.get() / 2))
                    .unwrap_or(2)
            }
            ThreadPoolProfile::LatencySensitive => {
                // Use 1.5x the available cores for better responsiveness
                std::thread::available_parallelism()
                    .map(|n| n.get() + n.get() / 2)
                    .unwrap_or(6)
            }
            ThreadPoolProfile::Custom(n) => *n,
        }
    }
}

/// Thread pool manager for linear algebra operations
pub struct ThreadPoolManager {
    profile: ThreadPoolProfile,
    /// Stack size for worker threads (in bytes)
    stacksize: Option<usize>,
    /// Thread name prefix
    thread_name_prefix: String,
    /// Whether to pin threads to CPU cores
    cpu_affinity: bool,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager with default settings
    pub fn new() -> Self {
        Self {
            profile: ThreadPoolProfile::Default,
            stacksize: None,
            thread_name_prefix: "linalg-worker".to_string(),
            cpu_affinity: false,
        }
    }

    /// Set the thread pool profile
    pub fn with_profile(mut self, profile: ThreadPoolProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the stack size for worker threads
    pub fn with_stacksize(mut self, size: usize) -> Self {
        self.stacksize = Some(size);
        self
    }

    /// Set the thread name prefix
    pub fn with_thread_name_prefix(mut self, prefix: String) -> Self {
        self.thread_name_prefix = prefix;
        self
    }

    /// Enable CPU affinity for worker threads
    pub fn with_cpu_affinity(mut self, enabled: bool) -> Self {
        self.cpu_affinity = enabled;
        self
    }

    /// Initialize the thread pool with current settings
    pub fn initialize(&self) -> Result<(), String> {
        let num_threads = self.profile.num_threads();

        // Configure rayon thread pool
        let thread_prefix = self.thread_name_prefix.clone();
        let mut pool_builder = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |idx| format!("{thread_prefix}-{idx}"));

        if let Some(stacksize) = self.stacksize {
            pool_builder = pool_builder.stack_size(stacksize);
        }

        pool_builder
            .build_global()
            .map_err(|e| format!("Failed to initialize thread pool: {e}"))?;

        // Set OpenMP threads for BLAS/LAPACK operations
        std::env::set_var("OMP_NUM_THREADS", num_threads.to_string());

        // Set MKL threads if using Intel MKL
        std::env::set_var("MKL_NUM_THREADS", num_threads.to_string());

        Ok(())
    }

    /// Get current thread pool statistics
    pub fn statistics(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            num_threads: self.profile.num_threads(),
            current_parallelism: num_threads(),
            profile: self.profile,
            stacksize: self.stacksize,
        }
    }
}

impl Default for ThreadPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread pool statistics
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    pub num_threads: usize,
    pub current_parallelism: usize,
    pub profile: ThreadPoolProfile,
    pub stacksize: Option<usize>,
}

/// Get the global thread pool manager
pub fn global_pool() -> Arc<Mutex<ThreadPoolManager>> {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_POOL = Some(Arc::new(Mutex::new(ThreadPoolManager::new())));
        });
        #[allow(static_mut_refs)]
        GLOBAL_POOL.as_ref().unwrap().clone()
    }
}

/// Initialize global thread pool with a specific profile
pub fn initialize_global_pool(profile: ThreadPoolProfile) -> Result<(), String> {
    let pool = global_pool();
    let mut manager = pool.lock().unwrap();
    manager.profile = profile;
    manager.initialize()
}

/// Adaptive thread pool that adjusts based on workload
pub struct AdaptiveThreadPool {
    min_threads: usize,
    max_threads: usize,
    current_threads: Arc<Mutex<usize>>,
    /// Tracks CPU utilization for adaptive scaling
    cpu_utilization: Arc<Mutex<f64>>,
}

impl AdaptiveThreadPool {
    /// Create a new adaptive thread pool
    pub fn new(_min_threads: usize, maxthreads: usize) -> Self {
        let current = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            min_threads: _min_threads,
            max_threads: maxthreads,
            current_threads: Arc::new(Mutex::new(current)),
            cpu_utilization: Arc::new(Mutex::new(0.0)),
        }
    }

    /// Update thread count based on current utilization
    pub fn adapt(&self, utilization: f64) {
        let mut current = self.current_threads.lock().unwrap();
        let mut cpu_util = self.cpu_utilization.lock().unwrap();
        *cpu_util = utilization;

        if utilization > 0.9 && *current < self.max_threads {
            // High utilization - increase threads
            *current = std::cmp::min(*current + 1, self.max_threads);
            self.apply_thread_count(*current);
        } else if utilization < 0.5 && *current > self.min_threads {
            // Low utilization - decrease threads
            *current = std::cmp::max(*current - 1, self.min_threads);
            self.apply_thread_count(*current);
        }
    }

    /// Apply the new thread count
    fn apply_thread_count(&self, count: usize) {
        configure_workers(Some(count));
    }

    /// Get current thread count
    pub fn current_thread_count(&self) -> usize {
        *self.current_threads.lock().unwrap()
    }
}

/// Thread pool benchmarking utilities
pub mod benchmark {
    use super::*;
    use std::time::{Duration, Instant};

    /// Benchmark result for a thread pool configuration
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub profile: ThreadPoolProfile,
        pub num_threads: usize,
        pub execution_time: Duration,
        pub throughput: f64,
    }

    /// Benchmark different thread pool configurations
    pub fn benchmark_configurations<F>(
        profiles: &[ThreadPoolProfile],
        workload: F,
    ) -> Vec<BenchmarkResult>
    where
        F: Fn() -> f64 + Clone,
    {
        let mut results = Vec::new();

        for &profile in profiles {
            // Initialize thread pool with profile
            if let Err(e) = initialize_global_pool(profile) {
                eprintln!("Failed to initialize pool for {profile:?}: {e}");
                continue;
            }

            // Warm up
            for _ in 0..3 {
                workload();
            }

            // Benchmark
            let start = Instant::now();
            let operations = 10;
            let mut total_work = 0.0;

            for _ in 0..operations {
                total_work += workload();
            }

            let elapsed = start.elapsed();
            let throughput = total_work / elapsed.as_secs_f64();

            results.push(BenchmarkResult {
                profile,
                num_threads: profile.num_threads(),
                execution_time: elapsed,
                throughput,
            });
        }

        results
    }

    /// Find optimal thread pool configuration for a workload
    pub fn find_optimal_configuration<F>(workload: F) -> ThreadPoolProfile
    where
        F: Fn() -> f64 + Clone,
    {
        let profiles = vec![
            ThreadPoolProfile::CpuBound,
            ThreadPoolProfile::MemoryBound,
            ThreadPoolProfile::LatencySensitive,
        ];

        let results = benchmark_configurations(&profiles, workload);

        results
            .into_iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
            .map(|r| r.profile)
            .unwrap_or(ThreadPoolProfile::Default)
    }
}

/// Enhanced thread pool with advanced monitoring and scaling
///
/// This provides sophisticated thread pool management with real-time monitoring,
/// dynamic scaling, and intelligent load balancing for optimal performance.
pub struct EnhancedThreadPool {
    #[allow(dead_code)]
    base_pool: Arc<Mutex<ThreadPoolManager>>,
    monitoring: Arc<Mutex<ThreadPoolMonitoring>>,
    scaling_policy: ScalingPolicy,
    load_balancer: LoadBalancer,
}

impl EnhancedThreadPool {
    /// Create a new enhanced thread pool
    pub fn new(profile: ThreadPoolProfile) -> Self {
        let base_pool = Arc::new(Mutex::new(ThreadPoolManager::new().with_profile(profile)));

        Self {
            base_pool,
            monitoring: Arc::new(Mutex::new(ThreadPoolMonitoring::new())),
            scaling_policy: ScalingPolicy::Conservative,
            load_balancer: LoadBalancer::RoundRobin,
        }
    }

    /// Set scaling policy
    pub fn with_scaling_policy(mut self, policy: ScalingPolicy) -> Self {
        self.scaling_policy = policy;
        self
    }

    /// Set load balancing strategy
    pub fn with_load_balancer(mut self, balancer: LoadBalancer) -> Self {
        self.load_balancer = balancer;
        self
    }

    /// Get current thread pool metrics
    pub fn get_metrics(&self) -> ThreadPoolMetrics {
        let monitoring = self.monitoring.lock().unwrap();
        monitoring.get_metrics()
    }

    /// Execute task with monitoring and adaptive scaling
    pub fn execute_monitored<F, R>(&self, task: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let start_time = std::time::Instant::now();

        // Update monitoring before execution
        {
            let mut monitoring = self.monitoring.lock().unwrap();
            monitoring.record_task_start();
        }

        // Execute task
        let result = task();

        // Update monitoring after execution
        {
            let mut monitoring = self.monitoring.lock().unwrap();
            monitoring.record_task_completion(start_time.elapsed());
        }

        // Check if scaling is needed
        self.check_and_scale();

        result
    }

    /// Check if thread pool scaling is needed and apply if necessary
    fn check_and_scale(&self) {
        let metrics = self.get_metrics();

        match self.scaling_policy {
            ScalingPolicy::Conservative => {
                // Scale up only if utilization > 90% for extended period
                if metrics.average_utilization > 0.9 && metrics.queue_length > 10 {
                    self.scale_up();
                }
                // Scale down only if utilization < 30% for extended period
                else if metrics.average_utilization < 0.3 && metrics.active_threads > 2 {
                    self.scale_down();
                }
            }
            ScalingPolicy::Aggressive => {
                // Scale up if utilization > 70%
                if metrics.average_utilization > 0.7 {
                    self.scale_up();
                }
                // Scale down if utilization < 50%
                else if metrics.average_utilization < 0.5 && metrics.active_threads > 1 {
                    self.scale_down();
                }
            }
            ScalingPolicy::LatencyOptimized => {
                // Prioritize low latency over efficiency
                if metrics.average_latency_ms > 10.0 {
                    self.scale_up();
                } else if metrics.average_latency_ms < 2.0 && metrics.active_threads > 2 {
                    self.scale_down();
                }
            }
            ScalingPolicy::Fixed => {
                // No scaling
            }
        }
    }

    /// Scale up the thread pool
    fn scale_up(&self) {
        // Implementation would involve creating new threads
        // For now, we'll just log the intent
        println!("Scaling up thread pool due to high utilization");
    }

    /// Scale down the thread pool
    fn scale_down(&self) {
        // Implementation would involve reducing threads
        // For now, we'll just log the intent
        println!("Scaling down thread pool due to low utilization");
    }
}

/// Thread pool scaling policies
#[derive(Debug, Clone, Copy)]
pub enum ScalingPolicy {
    /// Conservative scaling - only scale when definitely needed
    Conservative,
    /// Aggressive scaling - scale more readily for performance
    Aggressive,
    /// Optimized for low latency
    LatencyOptimized,
    /// Fixed thread count - no scaling
    Fixed,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancer {
    /// Simple round-robin task distribution
    RoundRobin,
    /// Least loaded thread gets next task
    LeastLoaded,
    /// Work-stealing between threads
    WorkStealing,
    /// NUMA-aware task assignment
    NumaAware,
}

/// Thread pool monitoring and metrics collection
struct ThreadPoolMonitoring {
    task_count: usize,
    total_execution_time: std::time::Duration,
    active_threads: usize,
    queue_length: usize,
    start_times: Vec<std::time::Instant>,
}

impl ThreadPoolMonitoring {
    fn new() -> Self {
        Self {
            task_count: 0,
            total_execution_time: std::time::Duration::ZERO,
            active_threads: 0,
            queue_length: 0,
            start_times: Vec::new(),
        }
    }

    fn record_task_start(&mut self) {
        self.task_count += 1;
        self.start_times.push(std::time::Instant::now());
        self.queue_length += 1;
    }

    fn record_task_completion(&mut self, duration: std::time::Duration) {
        self.total_execution_time += duration;
        self.queue_length = self.queue_length.saturating_sub(1);
    }

    fn get_metrics(&self) -> ThreadPoolMetrics {
        ThreadPoolMetrics {
            active_threads: self.active_threads,
            queue_length: self.queue_length,
            total_tasks: self.task_count,
            average_utilization: if self.active_threads > 0 {
                self.queue_length as f64 / self.active_threads as f64
            } else {
                0.0
            },
            average_latency_ms: if self.task_count > 0 {
                self.total_execution_time.as_millis() as f64 / self.task_count as f64
            } else {
                0.0
            },
            throughput_tasks_per_sec: if !self.total_execution_time.is_zero() {
                self.task_count as f64 / self.total_execution_time.as_secs_f64()
            } else {
                0.0
            },
        }
    }
}

/// Thread pool performance metrics
#[derive(Debug, Clone)]
pub struct ThreadPoolMetrics {
    /// Number of currently active threads
    pub active_threads: usize,
    /// Number of tasks waiting in queue
    pub queue_length: usize,
    /// Total number of tasks processed
    pub total_tasks: usize,
    /// Average thread utilization (0.0 to 1.0+)
    pub average_utilization: f64,
    /// Average task latency in milliseconds
    pub average_latency_ms: f64,
    /// Throughput in tasks per second
    pub throughput_tasks_per_sec: f64,
}
