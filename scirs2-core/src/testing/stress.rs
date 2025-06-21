//! # Stress Testing Framework
//!
//! This module provides stress testing capabilities for performance limits,
//! memory pressure, and system resource exhaustion scenarios. It includes:
//! - Memory pressure testing with large allocations
//! - CPU intensive workload testing
//! - Concurrent access stress testing
//! - Resource exhaustion simulation
//! - Performance degradation detection

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::testing::{TestConfig, TestResult};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Maximum memory to allocate (in bytes)
    pub max_memory: usize,
    /// Number of concurrent threads to use
    pub thread_count: usize,
    /// Duration for sustained stress testing
    pub duration: Duration,
    /// Memory allocation step size
    pub memory_step: usize,
    /// CPU workload intensity (iterations)
    pub cpu_intensity: usize,
    /// Enable memory leak detection
    pub detect_leaks: bool,
    /// Performance threshold (operations per second)
    pub performance_threshold: f64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            duration: Duration::from_secs(60),
            memory_step: 1024 * 1024, // 1MB
            cpu_intensity: 1000000,
            detect_leaks: true,
            performance_threshold: 1000.0,
        }
    }
}

impl StressTestConfig {
    /// Create a new stress test configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum memory allocation
    pub fn with_max_memory(mut self, memory: usize) -> Self {
        self.max_memory = memory;
        self
    }

    /// Set the thread count
    pub fn with_thread_count(mut self, count: usize) -> Self {
        self.thread_count = count;
        self
    }

    /// Set the test duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set the memory allocation step size
    pub fn with_memory_step(mut self, step: usize) -> Self {
        self.memory_step = step;
        self
    }

    /// Set the CPU intensity
    pub fn with_cpu_intensity(mut self, intensity: usize) -> Self {
        self.cpu_intensity = intensity;
        self
    }

    /// Enable or disable leak detection
    pub fn with_leak_detection(mut self, detect: bool) -> Self {
        self.detect_leaks = detect;
        self
    }

    /// Set the performance threshold
    pub fn with_performance_threshold(mut self, threshold: f64) -> Self {
        self.performance_threshold = threshold;
        self
    }
}

/// Result of stress testing
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Test name
    pub test_name: String,
    /// Peak memory usage observed
    pub peak_memory: usize,
    /// Average operations per second
    pub ops_per_second: f64,
    /// Total operations completed
    pub total_operations: usize,
    /// Test duration
    pub duration: Duration,
    /// Whether memory leaks were detected
    pub memory_leaks_detected: bool,
    /// Performance threshold met
    pub performance_threshold_met: bool,
    /// Error information if test failed
    pub error: Option<String>,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

impl StressTestResult {
    /// Create a new stress test result
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            peak_memory: 0,
            ops_per_second: 0.0,
            total_operations: 0,
            duration: Duration::from_secs(0),
            memory_leaks_detected: false,
            performance_threshold_met: false,
            error: None,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Set peak memory usage
    pub fn with_peak_memory(mut self, memory: usize) -> Self {
        self.peak_memory = memory;
        self
    }

    /// Set operations per second
    pub fn with_ops_per_second(mut self, ops: f64) -> Self {
        self.ops_per_second = ops;
        self
    }

    /// Set total operations
    pub fn with_total_operations(mut self, ops: usize) -> Self {
        self.total_operations = ops;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set memory leak detection result
    pub fn with_memory_leaks(mut self, detected: bool) -> Self {
        self.memory_leaks_detected = detected;
        self
    }

    /// Set performance threshold result
    pub fn with_performance_threshold(mut self, met: bool) -> Self {
        self.performance_threshold_met = met;
        self
    }

    /// Set error
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }

    /// Add a metric
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Memory pressure testing utilities
pub struct MemoryStressTester {
    config: StressTestConfig,
}

impl MemoryStressTester {
    /// Create a new memory stress tester
    pub fn new(config: StressTestConfig) -> Self {
        Self { config }
    }

    /// Test progressive memory allocation
    pub fn test_progressive_allocation(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("progressive_allocation".to_string());
        let mut allocations = Vec::new();
        let mut current_memory = 0;

        // Monitor initial memory
        let initial_memory = self.get_memory_usage()?;

        while current_memory < self.config.max_memory {
            // Allocate a chunk of memory
            let chunk_size = self
                .config
                .memory_step
                .min(self.config.max_memory - current_memory);

            match self.allocate_chunk(chunk_size) {
                Ok(chunk) => {
                    allocations.push(chunk);
                    current_memory += chunk_size;

                    // Update peak memory
                    if let Ok(memory) = self.get_memory_usage() {
                        result.peak_memory = result.peak_memory.max(memory - initial_memory);
                    }
                }
                Err(e) => {
                    result = result.with_error(format!(
                        "Allocation failed at {} bytes: {:?}",
                        current_memory, e
                    ));
                    break;
                }
            }

            // Check if we've exceeded time limit
            if start_time.elapsed() > self.config.duration {
                break;
            }
        }

        // Clean up allocations
        drop(allocations);

        // Check for memory leaks
        if self.config.detect_leaks {
            let final_memory = self.get_memory_usage()?;
            let leak_detected = final_memory > initial_memory + 1024 * 1024; // 1MB tolerance
            result = result.with_memory_leaks(leak_detected);
        }

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(current_memory / self.config.memory_step);

        Ok(result)
    }

    /// Test fragmented memory allocation
    pub fn test_fragmented_allocation(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("fragmented_allocation".to_string());
        let mut allocations = Vec::new();
        let mut operations = 0;

        // Allocate and deallocate in a fragmented pattern
        while start_time.elapsed() < self.config.duration {
            // Allocate several small chunks
            for _ in 0..10 {
                if let Ok(chunk) = self.allocate_chunk(1024) {
                    allocations.push(chunk);
                    operations += 1;
                }
            }

            // Deallocate every other chunk to create fragmentation
            let mut i = 0;
            allocations.retain(|_| {
                i += 1;
                i % 2 == 0
            });

            // Track peak memory
            if let Ok(memory) = self.get_memory_usage() {
                result.peak_memory = result.peak_memory.max(memory);
            }
        }

        // Clean up
        drop(allocations);

        let ops_per_second = operations as f64 / start_time.elapsed().as_secs_f64();
        let threshold_met = ops_per_second >= self.config.performance_threshold;

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(operations)
            .with_ops_per_second(ops_per_second)
            .with_performance_threshold(threshold_met);

        Ok(result)
    }

    /// Allocate a chunk of memory
    fn allocate_chunk(&self, size: usize) -> CoreResult<Vec<u8>> {
        let mut chunk = Vec::new();
        chunk.try_reserve(size).map_err(|e| {
            CoreError::MemoryError(ErrorContext::new(format!(
                "Failed to allocate {} bytes: {}",
                size, e
            )))
        })?;

        // Fill with data to ensure actual allocation
        chunk.resize(size, 42);
        Ok(chunk)
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> CoreResult<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to read memory status: {}",
                    e
                )))
            })?;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().map_err(|e| {
                            CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                                "Failed to parse memory: {}",
                                e
                            )))
                        })?;
                        return Ok(kb * 1024);
                    }
                }
            }
        }

        // Fallback for non-Linux systems
        Ok(0)
    }
}

/// CPU stress testing utilities
pub struct CpuStressTester {
    config: StressTestConfig,
}

impl CpuStressTester {
    /// Create a new CPU stress tester
    pub fn new(config: StressTestConfig) -> Self {
        Self { config }
    }

    /// Test CPU intensive workload
    pub fn test_cpu_intensive_workload(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("cpu_intensive_workload".to_string());
        let mut total_operations = 0;

        while start_time.elapsed() < self.config.duration {
            // Perform CPU-intensive computation
            let operations = self.cpu_intensive_computation(self.config.cpu_intensity)?;
            total_operations += operations;
        }

        let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64();
        let threshold_met = ops_per_second >= self.config.performance_threshold;

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(total_operations)
            .with_ops_per_second(ops_per_second)
            .with_performance_threshold(threshold_met);

        Ok(result)
    }

    /// Test concurrent CPU workload
    pub fn test_concurrent_cpu_workload(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("concurrent_cpu_workload".to_string());

        let config = Arc::new(self.config.clone());
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Spawn worker threads
        let mut handles = Vec::new();
        for thread_id in 0..self.config.thread_count {
            let config = Arc::clone(&config);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let mut operations = 0;
                while start_time.elapsed() < config.duration {
                    if let Ok(ops) =
                        Self::cpu_intensive_computation_static(config.cpu_intensity / 10)
                    {
                        operations += ops;
                    }
                }

                if let Ok(mut results) = results.lock() {
                    results.push((thread_id, operations));
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new("Thread join failed"))
            })?;
        }

        // Collect results
        let results_guard = results.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new("Failed to lock results"))
        })?;

        let total_operations: usize = results_guard.iter().map(|(_, ops)| ops).sum();
        let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64();
        let threshold_met = ops_per_second >= self.config.performance_threshold;

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(total_operations)
            .with_ops_per_second(ops_per_second)
            .with_performance_threshold(threshold_met)
            .with_metric("threads_used".to_string(), self.config.thread_count as f64);

        Ok(result)
    }

    /// Perform CPU-intensive computation
    fn cpu_intensive_computation(&self, iterations: usize) -> CoreResult<usize> {
        Self::cpu_intensive_computation_static(iterations)
    }

    /// Static version of CPU-intensive computation
    fn cpu_intensive_computation_static(iterations: usize) -> CoreResult<usize> {
        let mut sum = 0u64;
        let mut operations = 0;

        for i in 0..iterations {
            // Perform some non-trivial computation
            sum = sum.wrapping_add((i as u64).wrapping_mul(17).wrapping_add(23));
            sum = sum.wrapping_mul(31);
            operations += 1;
        }

        // Ensure the computation isn't optimized away
        if sum == 0 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Unexpected computation result"),
            ));
        }

        Ok(operations)
    }
}

/// Concurrent access stress testing
pub struct ConcurrencyStressTester {
    config: StressTestConfig,
}

impl ConcurrencyStressTester {
    /// Create a new concurrency stress tester
    pub fn new(config: StressTestConfig) -> Self {
        Self { config }
    }

    /// Test shared resource contention
    pub fn test_shared_resource_contention(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("shared_resource_contention".to_string());

        let shared_counter = Arc::new(std::sync::Mutex::new(0u64));
        let config = Arc::new(self.config.clone());
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Spawn worker threads that contend for shared resource
        let mut handles = Vec::new();
        for thread_id in 0..self.config.thread_count {
            let counter = Arc::clone(&shared_counter);
            let config = Arc::clone(&config);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let mut operations = 0;
                while start_time.elapsed() < config.duration {
                    // Contend for the shared resource
                    if let Ok(mut counter) = counter.lock() {
                        *counter += 1;
                        operations += 1;

                        // Simulate some work while holding the lock
                        for _ in 0..100 {
                            *counter = counter.wrapping_add(1).wrapping_sub(1);
                        }
                    }

                    // Brief yield to allow other threads to run
                    thread::yield_now();
                }

                if let Ok(mut results) = results.lock() {
                    results.push((thread_id, operations));
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new("Thread join failed"))
            })?;
        }

        // Collect results
        let results_guard = results.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new("Failed to lock results"))
        })?;

        let total_operations: usize = results_guard.iter().map(|(_, ops)| ops).sum();
        let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64();
        let threshold_met = ops_per_second >= self.config.performance_threshold;

        // Check final counter value
        let final_counter = *shared_counter.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new("Failed to lock counter"))
        })?;

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(total_operations)
            .with_ops_per_second(ops_per_second)
            .with_performance_threshold(threshold_met)
            .with_metric("final_counter_value".to_string(), final_counter as f64)
            .with_metric("threads_used".to_string(), self.config.thread_count as f64);

        Ok(result)
    }

    /// Test lock-free data structure performance under stress
    pub fn test_lock_free_performance(&self) -> CoreResult<StressTestResult> {
        let start_time = Instant::now();
        let mut result = StressTestResult::new("lock_free_performance".to_string());

        let atomic_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let config = Arc::new(self.config.clone());
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Spawn worker threads using atomic operations
        let mut handles = Vec::new();
        for thread_id in 0..self.config.thread_count {
            let counter = Arc::clone(&atomic_counter);
            let config = Arc::clone(&config);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let mut operations = 0;
                while start_time.elapsed() < config.duration {
                    // Perform atomic operations
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    operations += 1;

                    // Perform some additional atomic operations
                    let _old_value = counter.load(std::sync::atomic::Ordering::Relaxed);
                    counter.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    operations += 2;
                }

                if let Ok(mut results) = results.lock() {
                    results.push((thread_id, operations));
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new("Thread join failed"))
            })?;
        }

        // Collect results
        let results_guard = results.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new("Failed to lock results"))
        })?;

        let total_operations: usize = results_guard.iter().map(|(_, ops)| ops).sum();
        let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64();
        let threshold_met = ops_per_second >= self.config.performance_threshold;

        let final_counter = atomic_counter.load(std::sync::atomic::Ordering::Relaxed);

        result = result
            .with_duration(start_time.elapsed())
            .with_total_operations(total_operations)
            .with_ops_per_second(ops_per_second)
            .with_performance_threshold(threshold_met)
            .with_metric("final_atomic_value".to_string(), final_counter as f64)
            .with_metric("threads_used".to_string(), self.config.thread_count as f64);

        Ok(result)
    }
}

/// High-level stress testing utilities
pub struct StressTestUtils;

impl StressTestUtils {
    /// Create a comprehensive stress test suite
    pub fn create_stress_test_suite(name: &str, config: TestConfig) -> crate::testing::TestSuite {
        let mut suite = crate::testing::TestSuite::new(name, config);
        let stress_config = StressTestConfig::default()
            .with_duration(Duration::from_secs(10)) // Shorter duration for tests
            .with_max_memory(64 * 1024 * 1024); // 64MB for tests

        // Memory stress tests
        let stress_config_clone1 = stress_config.clone();
        suite.add_test("memory_progressive_allocation", move |_runner| {
            let tester = MemoryStressTester::new(stress_config_clone1.clone());
            let result = tester.test_progressive_allocation()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(
                TestResult::success(result.duration, result.total_operations)
                    .with_memory_usage(result.peak_memory),
            )
        });

        let stress_config_clone2 = stress_config.clone();
        suite.add_test("memory_fragmented_allocation", move |_runner| {
            let tester = MemoryStressTester::new(stress_config_clone2.clone());
            let result = tester.test_fragmented_allocation()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(
                TestResult::success(result.duration, result.total_operations)
                    .with_memory_usage(result.peak_memory),
            )
        });

        // CPU stress tests
        let stress_config_clone = stress_config.clone();
        suite.add_test("cpu_intensive_workload", move |_runner| {
            let tester = CpuStressTester::new(stress_config_clone.clone());
            let result = tester.test_cpu_intensive_workload()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(TestResult::success(
                result.duration,
                result.total_operations,
            ))
        });

        let stress_config_clone2 = stress_config.clone();
        suite.add_test("concurrent_cpu_workload", move |_runner| {
            let tester = CpuStressTester::new(stress_config_clone2.clone());
            let result = tester.test_concurrent_cpu_workload()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(TestResult::success(
                result.duration,
                result.total_operations,
            ))
        });

        // Concurrency stress tests
        let stress_config_clone3 = stress_config.clone();
        suite.add_test("shared_resource_contention", move |_runner| {
            let tester = ConcurrencyStressTester::new(stress_config_clone3.clone());
            let result = tester.test_shared_resource_contention()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(TestResult::success(
                result.duration,
                result.total_operations,
            ))
        });

        let stress_config_clone4 = stress_config.clone();
        suite.add_test("lock_free_performance", move |_runner| {
            let tester = ConcurrencyStressTester::new(stress_config_clone4.clone());
            let result = tester.test_lock_free_performance()?;

            if result.error.is_some() {
                return Ok(TestResult::failure(
                    result.duration,
                    result.total_operations,
                    result.error.unwrap(),
                ));
            }

            Ok(TestResult::success(
                result.duration,
                result.total_operations,
            ))
        });

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_config() {
        let config = StressTestConfig::new()
            .with_max_memory(512 * 1024 * 1024)
            .with_thread_count(8)
            .with_duration(Duration::from_secs(30))
            .with_cpu_intensity(500000);

        assert_eq!(config.max_memory, 512 * 1024 * 1024);
        assert_eq!(config.thread_count, 8);
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.cpu_intensity, 500000);
    }

    #[test]
    fn test_memory_stress_tester() {
        let config = StressTestConfig::default()
            .with_max_memory(1024 * 1024) // 1MB for test
            .with_duration(Duration::from_millis(100));

        let tester = MemoryStressTester::new(config);

        // This should complete without error
        let result = tester.test_progressive_allocation();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_stress_tester() {
        let config = StressTestConfig::default()
            .with_cpu_intensity(1000)
            .with_duration(Duration::from_millis(100));

        let tester = CpuStressTester::new(config);

        // This should complete without error
        let result = tester.test_cpu_intensive_workload();
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.total_operations > 0);
    }

    #[test]
    fn test_concurrency_stress_tester() {
        let config = StressTestConfig::default()
            .with_thread_count(2)
            .with_duration(Duration::from_millis(100));

        let tester = ConcurrencyStressTester::new(config);

        // This should complete without error
        let result = tester.test_shared_resource_contention();
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.total_operations > 0);
    }
}
