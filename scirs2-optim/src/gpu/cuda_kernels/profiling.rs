//! CUDA kernel profiling and performance monitoring
//!
//! This module provides comprehensive profiling capabilities for CUDA kernel execution,
//! including performance metrics collection, timing analysis, memory usage tracking,
//! and adaptive optimization based on runtime characteristics.

use crate::gpu::cuda_kernels::config::*;
use scirs2_core::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaEvent, CudaStream};

/// Performance metrics for CUDA kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of kernel launches
    pub total_launches: u64,
    /// Total execution time across all launches
    pub total_time: Duration,
    /// Average execution time per launch
    pub average_time: Duration,
    /// Minimum execution time observed
    pub min_time: Duration,
    /// Maximum execution time observed
    pub max_time: Duration,
    /// Memory throughput in GB/s
    pub memory_throughput: f64,
    /// Computational throughput in FLOPS
    pub compute_throughput: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Tensor Core utilization percentage (if applicable)
    pub tensor_core_utilization: f64,
    /// Error count during execution
    pub error_count: u64,
    /// Success rate percentage
    pub success_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_launches: 0,
            total_time: Duration::ZERO,
            average_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            memory_throughput: 0.0,
            compute_throughput: 0.0,
            gpu_utilization: 0.0,
            memory_utilization: 0.0,
            tensor_core_utilization: 0.0,
            error_count: 0,
            success_rate: 100.0,
        }
    }
}

/// Detailed timing information for a single kernel execution
#[derive(Debug, Clone)]
pub struct ExecutionTiming {
    /// Kernel launch timestamp
    pub launch_time: Instant,
    /// GPU execution start time
    pub gpu_start: Option<Instant>,
    /// GPU execution end time
    pub gpu_end: Option<Instant>,
    /// Memory transfer time (host to device)
    pub h2d_time: Duration,
    /// Memory transfer time (device to host)
    pub d2h_time: Duration,
    /// Kernel execution time on GPU
    pub kernel_time: Duration,
    /// Total end-to-end time
    pub total_time: Duration,
    /// Number of elements processed
    pub element_count: usize,
    /// Memory bytes transferred
    pub memory_transferred: usize,
    /// FLOP count estimate
    pub flop_count: u64,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Profiling sample with detailed metrics
#[derive(Debug, Clone)]
pub struct ProfilingSample {
    /// Sample timestamp
    pub timestamp: Instant,
    /// Kernel execution timing
    pub timing: ExecutionTiming,
    /// GPU memory usage at sample time
    pub memory_usage: MemoryUsage,
    /// GPU temperature if available
    pub gpu_temperature: Option<f32>,
    /// GPU power consumption if available
    pub power_consumption: Option<f32>,
    /// Clock frequencies
    pub clock_frequencies: ClockFrequencies,
}

/// GPU memory usage information
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Total GPU memory in bytes
    pub total_memory: u64,
    /// Used GPU memory in bytes
    pub used_memory: u64,
    /// Free GPU memory in bytes
    pub free_memory: u64,
    /// Memory utilization percentage
    pub utilization: f64,
    /// Peak memory usage during execution
    pub peak_usage: u64,
}

/// GPU clock frequency information
#[derive(Debug, Clone, Default)]
pub struct ClockFrequencies {
    /// Core clock frequency in MHz
    pub core_clock: u32,
    /// Memory clock frequency in MHz
    pub memory_clock: u32,
    /// Shader clock frequency in MHz
    pub shader_clock: u32,
}

/// CUDA kernel profiler for performance monitoring and optimization
pub struct KernelProfiler {
    /// Profiling configuration
    config: ProfilingConfig,
    /// Performance metrics accumulator
    metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Recent profiling samples (ring buffer)
    samples: Arc<Mutex<VecDeque<ProfilingSample>>>,
    /// Per-kernel metrics tracking
    kernel_metrics: Arc<Mutex<HashMap<String, PerformanceMetrics>>>,
    /// CUDA events for precise timing
    #[cfg(feature = "cuda")]
    cuda_events: Arc<Mutex<Vec<CudaEvent>>>,
    /// CUDA stream for profiling
    #[cfg(feature = "cuda")]
    profiling_stream: Arc<CudaStream>,
}

impl KernelProfiler {
    /// Creates a new kernel profiler
    pub fn new(config: ProfilingConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let (cuda_events, profiling_stream) = {
            let device = CudaDevice::new(0)?;
            let stream = device.fork_default_stream()?;
            let events = Vec::new();
            (Arc::new(Mutex::new(events)), Arc::new(stream))
        };

        Ok(Self {
            config,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            samples: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_samples))),
            kernel_metrics: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "cuda")]
            cuda_events,
            #[cfg(feature = "cuda")]
            profiling_stream,
        })
    }

    /// Starts profiling a kernel execution
    pub fn start_profiling(&self, kernel_name: &str) -> Result<ProfilingHandle> {
        if !self.config.enable_profiling {
            return Ok(ProfilingHandle::disabled());
        }

        let should_sample = fastrand::f32() < self.config.sample_rate;
        if !should_sample {
            return Ok(ProfilingHandle::disabled());
        }

        let launch_time = Instant::now();

        #[cfg(feature = "cuda")]
        let cuda_start_event = if self.config.enable_cuda_events {
            let mut events = self.cuda_events.lock().unwrap();
            let event = self.profiling_stream.as_ref().record_event()?;
            events.push(event.clone());
            Some(event)
        } else {
            None
        };

        Ok(ProfilingHandle {
            kernel_name: kernel_name.to_string(),
            launch_time,
            profiler: Some(self.clone()),
            #[cfg(feature = "cuda")]
            cuda_start_event,
            enabled: true,
        })
    }

    /// Records a completed kernel execution
    pub fn record_execution(&self, timing: ExecutionTiming, kernel_name: &str) -> Result<()> {
        if !self.config.enable_profiling {
            return Ok(());
        }

        // Update global metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            self.update_metrics(&mut metrics, &timing);
        }

        // Update per-kernel metrics
        {
            let mut kernel_metrics = self.kernel_metrics.lock().unwrap();
            let metrics = kernel_metrics.entry(kernel_name.to_string())
                .or_insert_with(PerformanceMetrics::default);
            self.update_metrics(metrics, &timing);
        }

        // Add profiling sample
        if self.config.enable_detailed_sampling {
            let sample = ProfilingSample {
                timestamp: timing.launch_time,
                timing,
                memory_usage: self.get_memory_usage()?,
                gpu_temperature: self.get_gpu_temperature(),
                power_consumption: self.get_power_consumption(),
                clock_frequencies: self.get_clock_frequencies(),
            };

            let mut samples = self.samples.lock().unwrap();
            if samples.len() >= self.config.max_samples {
                samples.pop_front();
            }
            samples.push_back(sample);
        }

        Ok(())
    }

    /// Updates performance metrics with new timing data
    fn update_metrics(&self, metrics: &mut PerformanceMetrics, timing: &ExecutionTiming) {
        metrics.total_launches += 1;
        metrics.total_time += timing.total_time;

        if timing.success {
            metrics.average_time = metrics.total_time / metrics.total_launches as u32;
            metrics.min_time = metrics.min_time.min(timing.total_time);
            metrics.max_time = metrics.max_time.max(timing.total_time);

            // Calculate throughput metrics
            if timing.total_time.as_secs_f64() > 0.0 {
                let time_secs = timing.total_time.as_secs_f64();
                metrics.memory_throughput = (timing.memory_transferred as f64) / (1e9 * time_secs);
                metrics.compute_throughput = (timing.flop_count as f64) / time_secs;
            }
        } else {
            metrics.error_count += 1;
        }

        metrics.success_rate = ((metrics.total_launches - metrics.error_count) as f64
            / metrics.total_launches as f64) * 100.0;
    }

    /// Gets current GPU memory usage
    fn get_memory_usage(&self) -> Result<MemoryUsage> {
        #[cfg(feature = "cuda")]
        {
            // Use CUDA Runtime API to get memory info
            // This is a simplified implementation - real implementation would use cudarc
            Ok(MemoryUsage::default())
        }
        #[cfg(not(feature = "cuda"))]
        Ok(MemoryUsage::default())
    }

    /// Gets current GPU temperature
    fn get_gpu_temperature(&self) -> Option<f32> {
        // Implementation would use NVML or similar
        None
    }

    /// Gets current GPU power consumption
    fn get_power_consumption(&self) -> Option<f32> {
        // Implementation would use NVML or similar
        None
    }

    /// Gets current GPU clock frequencies
    fn get_clock_frequencies(&self) -> ClockFrequencies {
        // Implementation would use NVML or similar
        ClockFrequencies::default()
    }

    /// Gets current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Gets performance metrics for a specific kernel
    pub fn get_kernel_metrics(&self, kernel_name: &str) -> Option<PerformanceMetrics> {
        self.kernel_metrics.lock().unwrap().get(kernel_name).cloned()
    }

    /// Gets recent profiling samples
    pub fn get_recent_samples(&self, count: usize) -> Vec<ProfilingSample> {
        let samples = self.samples.lock().unwrap();
        samples.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Generates a performance report
    pub fn generate_report(&self) -> ProfilingReport {
        let global_metrics = self.get_metrics();
        let kernel_metrics = self.kernel_metrics.lock().unwrap().clone();
        let recent_samples = self.get_recent_samples(100);

        ProfilingReport {
            global_metrics,
            kernel_metrics,
            sample_count: recent_samples.len(),
            report_timestamp: Instant::now(),
            recommendations: self.generate_recommendations(&global_metrics, &kernel_metrics),
        }
    }

    /// Generates optimization recommendations based on profiling data
    fn generate_recommendations(&self, global_metrics: &PerformanceMetrics,
                               kernel_metrics: &HashMap<String, PerformanceMetrics>) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check success rate
        if global_metrics.success_rate < 95.0 {
            recommendations.push(format!(
                "Low success rate ({:.1}%) - investigate error causes",
                global_metrics.success_rate
            ));
        }

        // Check memory throughput
        if global_metrics.memory_throughput > 0.0 && global_metrics.memory_throughput < 100.0 {
            recommendations.push("Low memory throughput - consider memory access optimization".to_string());
        }

        // Check GPU utilization
        if global_metrics.gpu_utilization < 80.0 {
            recommendations.push("Low GPU utilization - consider increasing batch size or parallelism".to_string());
        }

        // Check per-kernel performance
        for (kernel_name, metrics) in kernel_metrics {
            if metrics.error_count > 0 {
                recommendations.push(format!("Kernel '{}' has {} errors", kernel_name, metrics.error_count));
            }
        }

        recommendations
    }

    /// Resets all profiling metrics
    pub fn reset_metrics(&self) {
        *self.metrics.lock().unwrap() = PerformanceMetrics::default();
        self.kernel_metrics.lock().unwrap().clear();
        self.samples.lock().unwrap().clear();
    }
}

impl Clone for KernelProfiler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: Arc::clone(&self.metrics),
            samples: Arc::clone(&self.samples),
            kernel_metrics: Arc::clone(&self.kernel_metrics),
            #[cfg(feature = "cuda")]
            cuda_events: Arc::clone(&self.cuda_events),
            #[cfg(feature = "cuda")]
            profiling_stream: Arc::clone(&self.profiling_stream),
        }
    }
}

/// Handle for managing a single kernel profiling session
pub struct ProfilingHandle {
    kernel_name: String,
    launch_time: Instant,
    profiler: Option<KernelProfiler>,
    #[cfg(feature = "cuda")]
    cuda_start_event: Option<CudaEvent>,
    enabled: bool,
}

impl ProfilingHandle {
    /// Creates a disabled profiling handle
    fn disabled() -> Self {
        Self {
            kernel_name: String::new(),
            launch_time: Instant::now(),
            profiler: None,
            #[cfg(feature = "cuda")]
            cuda_start_event: None,
            enabled: false,
        }
    }

    /// Completes the profiling session with success
    pub fn complete_success(self, element_count: usize, memory_transferred: usize, flop_count: u64) -> Result<()> {
        if !self.enabled || self.profiler.is_none() {
            return Ok(());
        }

        let total_time = self.launch_time.elapsed();

        #[cfg(feature = "cuda")]
        let kernel_time = if let Some(ref event) = self.cuda_start_event {
            // Calculate precise GPU timing using CUDA events
            // This would require end event creation and synchronization
            total_time // Simplified
        } else {
            total_time
        };

        #[cfg(not(feature = "cuda"))]
        let kernel_time = total_time;

        let timing = ExecutionTiming {
            launch_time: self.launch_time,
            gpu_start: Some(self.launch_time),
            gpu_end: Some(self.launch_time + kernel_time),
            h2d_time: Duration::ZERO, // Would be measured separately
            d2h_time: Duration::ZERO, // Would be measured separately
            kernel_time,
            total_time,
            element_count,
            memory_transferred,
            flop_count,
            success: true,
            error_message: None,
        };

        if let Some(profiler) = &self.profiler {
            profiler.record_execution(timing, &self.kernel_name)?;
        }

        Ok(())
    }

    /// Completes the profiling session with error
    pub fn complete_error(self, error: &str) -> Result<()> {
        if !self.enabled || self.profiler.is_none() {
            return Ok(());
        }

        let total_time = self.launch_time.elapsed();

        let timing = ExecutionTiming {
            launch_time: self.launch_time,
            gpu_start: None,
            gpu_end: None,
            h2d_time: Duration::ZERO,
            d2h_time: Duration::ZERO,
            kernel_time: Duration::ZERO,
            total_time,
            element_count: 0,
            memory_transferred: 0,
            flop_count: 0,
            success: false,
            error_message: Some(error.to_string()),
        };

        if let Some(profiler) = &self.profiler {
            profiler.record_execution(timing, &self.kernel_name)?;
        }

        Ok(())
    }
}

/// Comprehensive profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Global performance metrics
    pub global_metrics: PerformanceMetrics,
    /// Per-kernel performance metrics
    pub kernel_metrics: HashMap<String, PerformanceMetrics>,
    /// Number of samples in the report
    pub sample_count: usize,
    /// Report generation timestamp
    pub report_timestamp: Instant,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl ProfilingReport {
    /// Formats the report as a human-readable string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== CUDA Kernel Profiling Report ===\n\n");

        // Global metrics
        report.push_str(&format!("Global Performance:\n"));
        report.push_str(&format!("  Total Launches: {}\n", self.global_metrics.total_launches));
        report.push_str(&format!("  Success Rate: {:.2}%\n", self.global_metrics.success_rate));
        report.push_str(&format!("  Average Time: {:.3}ms\n", self.global_metrics.average_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("  Memory Throughput: {:.2} GB/s\n", self.global_metrics.memory_throughput));
        report.push_str(&format!("  Compute Throughput: {:.2e} FLOPS\n", self.global_metrics.compute_throughput));
        report.push_str("\n");

        // Per-kernel metrics
        if !self.kernel_metrics.is_empty() {
            report.push_str("Per-Kernel Performance:\n");
            for (kernel, metrics) in &self.kernel_metrics {
                report.push_str(&format!("  {}:\n", kernel));
                report.push_str(&format!("    Launches: {}\n", metrics.total_launches));
                report.push_str(&format!("    Avg Time: {:.3}ms\n", metrics.average_time.as_secs_f64() * 1000.0));
                report.push_str(&format!("    Success Rate: {:.2}%\n", metrics.success_rate));
            }
            report.push_str("\n");
        }

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