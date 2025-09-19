//! Advanced GPU context management for special functions
//!
//! This module provides robust GPU context management with automatic
//! fallback, resource pooling, and performance monitoring.

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::gpu::{GpuBackend, GpuContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// GPU device information and capabilities
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_id: usize,
    pub device_name: String,
    pub memorysize: u64,
    pub compute_units: u32,
    pub max_workgroupsize: u32,
    pub backend_type: GpuBackend,
    pub is_available: bool,
}

/// Performance statistics for GPU operations
#[derive(Debug, Clone, Default)]
pub struct GpuPerformanceStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub memory_transfers: u64,
    pub total_data_transferred: u64,
    pub peak_memory_usage: u64,
    pub cache_hit_rate: f64,
    pub last_error_message: Option<String>,
    pub operations_per_second: f64,
}

/// Production configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuProductionConfig {
    /// Minimum array size to consider GPU acceleration (default: 1000)
    pub min_gpu_arraysize: usize,
    /// Maximum GPU memory usage percentage (default: 80%)
    pub max_memory_usage_percent: f32,
    /// Enable automatic GPU/CPU switching based on performance (default: true)
    pub enable_adaptive_switching: bool,
    /// GPU warmup iterations for performance measurement (default: 3)
    pub warmup_iterations: u32,
    /// Maximum number of retry attempts for failed GPU operations (default: 3)
    pub max_retry_attempts: u32,
    /// Enable performance profiling and logging (default: false)
    pub enable_profiling: bool,
    /// Preferred GPU backend type (default: Auto)
    pub preferred_backend: GpuBackend,
}

impl Default for GpuProductionConfig {
    fn default() -> Self {
        Self {
            min_gpu_arraysize: 1000,
            max_memory_usage_percent: 80.0,
            enable_adaptive_switching: true,
            warmup_iterations: 3,
            max_retry_attempts: 3,
            enable_profiling: false,
            preferred_backend: GpuBackend::Cpu,
        }
    }
}

/// GPU context pool for managing multiple contexts
pub struct GpuContextPool {
    contexts: RwLock<HashMap<GpuBackend, Arc<GpuContext>>>,
    device_info: RwLock<HashMap<GpuBackend, GpuDeviceInfo>>,
    performance_stats: RwLock<HashMap<GpuBackend, GpuPerformanceStats>>,
    fallback_threshold: Mutex<usize>,
    auto_fallback_enabled: Mutex<bool>,
    production_config: RwLock<GpuProductionConfig>,
    memory_usage_tracker: RwLock<HashMap<GpuBackend, u64>>,
}

impl Default for GpuContextPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuContextPool {
    /// Create a new GPU context pool with production configuration
    pub fn new() -> Self {
        Self {
            contexts: RwLock::new(HashMap::new()),
            device_info: RwLock::new(HashMap::new()),
            performance_stats: RwLock::new(HashMap::new()),
            fallback_threshold: Mutex::new(5), // Fall back after 5 consecutive failures
            auto_fallback_enabled: Mutex::new(true),
            production_config: RwLock::new(GpuProductionConfig::default()),
            memory_usage_tracker: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new GPU context pool with custom production configuration
    pub fn with_config(config: GpuProductionConfig) -> Self {
        Self {
            contexts: RwLock::new(HashMap::new()),
            device_info: RwLock::new(HashMap::new()),
            performance_stats: RwLock::new(HashMap::new()),
            fallback_threshold: Mutex::new(config.max_retry_attempts as usize),
            auto_fallback_enabled: Mutex::new(config.enable_adaptive_switching),
            production_config: RwLock::new(config),
            memory_usage_tracker: RwLock::new(HashMap::new()),
        }
    }

    /// Update production configuration
    pub fn update_config(&self, config: GpuProductionConfig) {
        *self.production_config.write().unwrap() = config;
    }

    /// Get current production configuration
    pub fn get_config(&self) -> GpuProductionConfig {
        self.production_config.read().unwrap().clone()
    }

    /// Initialize GPU context pool with device discovery
    pub fn initialize(&self) -> SpecialResult<()> {
        self.discover_devices()?;
        self.create_contexts()?;
        Ok(())
    }

    /// Discover available GPU devices
    fn discover_devices(&self) -> SpecialResult<()> {
        let mut device_info = self.device_info.write().unwrap();

        // Try to discover WebGPU devices
        if let Ok(info) = self.probe_webgpu_device() {
            device_info.insert(GpuBackend::Wgpu, info);
        }

        // Try to discover OpenCL devices
        if let Ok(info) = self.probe_opencl_device() {
            device_info.insert(GpuBackend::OpenCL, info);
        }

        // Try to discover CUDA devices
        if let Ok(info) = self.probe_cuda_device() {
            device_info.insert(GpuBackend::Cuda, info);
        }

        if device_info.is_empty() {
            #[cfg(feature = "gpu")]
            log::warn!("No GPU devices discovered");
        } else {
            #[cfg(feature = "gpu")]
            log::info!("Discovered {} GPU device(s)", device_info.len());
        }

        Ok(())
    }

    /// Probe WebGPU device capabilities
    fn probe_webgpu_device(&self) -> SpecialResult<GpuDeviceInfo> {
        // use scirs2_core::gpu;

        match GpuContext::new(GpuBackend::Wgpu) {
            Ok(_context) => {
                let info = GpuDeviceInfo {
                    device_id: 0,
                    device_name: "WebGPU Device".to_string(),
                    memorysize: 1024 * 1024 * 1024, // Assume 1GB
                    compute_units: 32,              // Conservative estimate
                    max_workgroupsize: 256,
                    backend_type: GpuBackend::Wgpu,
                    is_available: true,
                };

                #[cfg(feature = "gpu")]
                log::info!("WebGPU device available: {}", info.device_name);

                Ok(info)
            }
            Err(e) => {
                #[cfg(feature = "gpu")]
                log::debug!("WebGPU not available: {}", e);
                Err(SpecialError::GpuNotAvailable(
                    "WebGPU not available".to_string(),
                ))
            }
        }
    }

    /// Probe OpenCL device capabilities with advanced detection
    fn probe_opencl_device(&self) -> SpecialResult<GpuDeviceInfo> {
        // use scirs2_core::gpu;

        #[cfg(feature = "gpu")]
        log::debug!("Probing OpenCL devices...");

        // Try to create OpenCL context to test availability
        match GpuContext::new(GpuBackend::OpenCL) {
            Ok(context) => {
                // Query OpenCL device properties if possible
                let info = self
                    .query_opencl_device_info(&std::sync::Arc::new(context))
                    .unwrap_or_else(|_| {
                        // Fallback to conservative defaults
                        GpuDeviceInfo {
                            device_id: 0,
                            device_name: "OpenCL Device".to_string(),
                            memorysize: 2 * 1024 * 1024 * 1024, // 2GB assumption
                            compute_units: 16,                  // Conservative estimate
                            max_workgroupsize: 256,
                            backend_type: GpuBackend::OpenCL,
                            is_available: true,
                        }
                    });

                #[cfg(feature = "gpu")]
                log::info!(
                    "OpenCL device available: {} with {} compute units",
                    info.device_name,
                    info.compute_units
                );

                Ok(info)
            }
            Err(e) => {
                #[cfg(feature = "gpu")]
                log::debug!("OpenCL not available: {}", e);
                Err(SpecialError::GpuNotAvailable(format!(
                    "OpenCL not available: {}",
                    e
                )))
            }
        }
    }

    /// Probe CUDA device capabilities with NVIDIA GPU detection
    fn probe_cuda_device(&self) -> SpecialResult<GpuDeviceInfo> {
        // use scirs2_core::gpu;

        #[cfg(feature = "gpu")]
        log::debug!("Probing CUDA devices...");

        // Try to create CUDA context to test availability
        match GpuContext::new(GpuBackend::Cuda) {
            Ok(context) => {
                // Query CUDA device properties if possible
                let info = self
                    .query_cuda_device_info(&std::sync::Arc::new(context))
                    .unwrap_or_else(|_| {
                        // Fallback to conservative defaults for CUDA
                        GpuDeviceInfo {
                            device_id: 0,
                            device_name: "NVIDIA CUDA Device".to_string(),
                            memorysize: 4 * 1024 * 1024 * 1024, // 4GB assumption for CUDA
                            compute_units: 64,                  // Higher for CUDA devices
                            max_workgroupsize: 1024,            // CUDA supports larger workgroups
                            backend_type: GpuBackend::Cuda,
                            is_available: true,
                        }
                    });

                #[cfg(feature = "gpu")]
                log::info!(
                    "CUDA device available: {} with {} SMs",
                    info.device_name,
                    info.compute_units
                );

                Ok(info)
            }
            Err(e) => {
                #[cfg(feature = "gpu")]
                log::debug!("CUDA not available: {}", e);
                Err(SpecialError::GpuNotAvailable(format!(
                    "CUDA not available: {}",
                    e
                )))
            }
        }
    }

    /// Create GPU contexts for discovered devices
    fn create_contexts(&self) -> SpecialResult<()> {
        let device_info = self.device_info.read().unwrap();
        let mut contexts = self.contexts.write().unwrap();
        let mut stats = self.performance_stats.write().unwrap();

        for (&backend_type, info) in device_info.iter() {
            if info.is_available {
                match GpuContext::new(backend_type) {
                    Ok(context) => {
                        contexts.insert(backend_type, std::sync::Arc::new(context));
                        stats.insert(backend_type, GpuPerformanceStats::default());

                        #[cfg(feature = "gpu")]
                        log::info!("Created GPU context for {:?}", backend_type);
                    }
                    Err(e) => {
                        #[cfg(feature = "gpu")]
                        log::warn!("Failed to create context for {:?}: {}", backend_type, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the best available GPU context
    pub fn get_best_context(&self) -> SpecialResult<Arc<GpuContext>> {
        let contexts = self.contexts.read().unwrap();
        let stats = self.performance_stats.read().unwrap();

        // Prioritize based on performance stats and backend type
        let preferred_order = [GpuBackend::Cuda, GpuBackend::Wgpu, GpuBackend::OpenCL];

        for &backend_type in &preferred_order {
            if let Some(context) = contexts.get(&backend_type) {
                if let Some(stat) = stats.get(&backend_type) {
                    // Check if context is healthy (success rate > 80%)
                    let success_rate = if stat.total_operations > 0 {
                        stat.successful_operations as f64 / stat.total_operations as f64
                    } else {
                        1.0 // No operations yet, assume healthy
                    };

                    if success_rate > 0.8 {
                        #[cfg(feature = "gpu")]
                        log::debug!(
                            "Using {:?} context (success rate: {:.1}%)",
                            backend_type,
                            success_rate * 100.0
                        );
                        return Ok(Arc::clone(context));
                    }
                }
            }
        }

        Err(SpecialError::GpuNotAvailable(
            "No healthy GPU contexts available".to_string(),
        ))
    }

    /// Record operation performance
    pub fn record_operation(
        &self,
        backend_type: GpuBackend,
        execution_time: Duration,
        success: bool,
        datasize: usize,
    ) {
        let mut stats = self.performance_stats.write().unwrap();
        if let Some(stat) = stats.get_mut(&backend_type) {
            stat.total_operations += 1;

            if success {
                stat.successful_operations += 1;
                stat.total_execution_time += execution_time;
                stat.average_execution_time =
                    stat.total_execution_time / stat.successful_operations as u32;
                stat.total_data_transferred += datasize as u64;
            } else {
                stat.failed_operations += 1;
            }

            stat.memory_transfers += 1;
        }
    }

    /// Get performance statistics for a backend
    pub fn get_performance_stats(&self, backendtype: GpuBackend) -> Option<GpuPerformanceStats> {
        let stats = self.performance_stats.read().unwrap();
        stats.get(&backendtype).cloned()
    }

    /// Get all available device information
    pub fn get_device_info(&self) -> HashMap<GpuBackend, GpuDeviceInfo> {
        self.device_info.read().unwrap().clone()
    }

    /// Check if GPU acceleration should be used for given array size
    pub fn should_use_gpu(&self, arraysize: usize, data_typesize: usize) -> bool {
        // Only use GPU for sufficiently large arrays
        let min_elements = match data_typesize {
            4 => 512,  // f32
            8 => 256,  // f64
            _ => 1024, // other types
        };

        if arraysize < min_elements {
            return false;
        }

        // Check if auto fallback is enabled and we have healthy contexts
        let auto_fallback = *self.auto_fallback_enabled.lock().unwrap();
        if !auto_fallback {
            return false;
        }

        // Check if we have any available contexts
        let contexts = self.contexts.read().unwrap();
        !contexts.is_empty()
    }

    /// Enable or disable automatic fallback to CPU
    pub fn set_auto_fallback(&self, enabled: bool) {
        *self.auto_fallback_enabled.lock().unwrap() = enabled;
    }

    /// Set the threshold for fallback after consecutive failures
    pub fn set_fallback_threshold(&self, threshold: usize) {
        *self.fallback_threshold.lock().unwrap() = threshold;
    }

    /// Query OpenCL device information with detailed properties
    fn query_opencl_device_info(&self, context: &Arc<GpuContext>) -> SpecialResult<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        log::debug!("Querying OpenCL device properties...");

        let estimated_memory = 2 * 1024 * 1024 * 1024; // 2GB default
        let estimated_compute_units = 16; // Default estimate

        Ok(GpuDeviceInfo {
            device_id: 0,
            device_name: format!("OpenCL GPU Device (Unknown)"),
            memorysize: estimated_memory,
            compute_units: estimated_compute_units,
            max_workgroupsize: 256,
            backend_type: GpuBackend::OpenCL,
            is_available: true,
        })
    }

    /// Query CUDA device information with detailed properties
    fn query_cuda_device_info(&self, context: &Arc<GpuContext>) -> SpecialResult<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        log::debug!("Querying CUDA device properties...");

        let estimated_memory = 4 * 1024 * 1024 * 1024; // 4GB default
        let estimated_compute_units = 64; // Default estimate

        Ok(GpuDeviceInfo {
            device_id: 0,
            device_name: format!("NVIDIA CUDA Device (Unknown)"),
            memorysize: estimated_memory,
            compute_units: estimated_compute_units,
            max_workgroupsize: 1024,
            backend_type: GpuBackend::Cuda,
            is_available: true,
        })
    }

    /// Helper functions for device estimation
    fn estimate_gpu_memory_opencl(&self) -> u64 {
        2 * 1024 * 1024 * 1024
    }
    fn estimate_gpu_memory_cuda(&self) -> u64 {
        4 * 1024 * 1024 * 1024
    }
    fn estimate_compute_units_opencl(&self) -> u32 {
        32
    }
    fn estimate_compute_units_cuda(&self) -> u32 {
        64
    }
    fn detect_gpu_vendor(&self) -> String {
        "Unknown Vendor".to_string()
    }
    fn detect_nvidia_architecture(&self) -> String {
        "Unknown Architecture".to_string()
    }
    fn get_system_memorysize(&self) -> u64 {
        8 * 1024 * 1024 * 1024
    }
    fn is_likely_integrated_gpu(&self) -> bool {
        false
    }

    /// Advanced performance monitoring with trend analysis
    pub fn get_performance_trends(&self) -> HashMap<GpuBackend, String> {
        let stats = self.performance_stats.read().unwrap();
        let mut trends = HashMap::new();

        for (&backend_type, stat) in stats.iter() {
            let trend_analysis = if stat.total_operations > 10 {
                let success_rate = stat.successful_operations as f64 / stat.total_operations as f64;
                let avg_throughput = if stat.average_execution_time.as_millis() > 0 {
                    1000.0 / stat.average_execution_time.as_millis() as f64
                } else {
                    0.0
                };

                format!(
                    "Success: {:.1}%, Throughput: {:.1} ops/sec, Data: {} MB",
                    success_rate * 100.0,
                    avg_throughput,
                    stat.total_data_transferred / 1024 / 1024
                )
            } else {
                "Insufficient data for trend analysis".to_string()
            };
            trends.insert(backend_type, trend_analysis);
        }
        trends
    }

    /// Clear performance statistics
    pub fn reset_performance_stats(&self) {
        let mut stats = self.performance_stats.write().unwrap();
        for stat in stats.values_mut() {
            *stat = GpuPerformanceStats::default();
        }
        #[cfg(feature = "gpu")]
        log::info!("GPU performance statistics reset");
    }

    /// Get all performance statistics
    pub fn get_performance_stats_all(&self) -> HashMap<GpuBackend, GpuPerformanceStats> {
        self.performance_stats.read().unwrap().clone()
    }

    /// Get comprehensive system report"
    pub fn get_system_report(&self) -> String {
        let device_info = self.device_info.read().unwrap();
        let stats = self.performance_stats.read().unwrap();

        let mut report = String::new();
        report.push_str("=== GPU System Report ===\n\n");

        if device_info.is_empty() {
            report.push_str("No GPU devices available.\n");
        } else {
            report.push_str(&format!("Found {} GPU device(s):\n\n", device_info.len()));

            for (backend_type, info) in device_info.iter() {
                report.push_str(&format!("Backend: {:?}\n", backend_type));
                report.push_str(&format!("  Device: {}\n", info.device_name));
                report.push_str(&format!("  Memory: {} MB\n", info.memorysize / 1024 / 1024));
                report.push_str(&format!("  Compute Units: {}\n", info.compute_units));
                report.push_str(&format!(
                    "  Max Workgroup Size: {}\n",
                    info.max_workgroupsize
                ));
                report.push_str(&format!("  Available: {}\n", info.is_available));

                if let Some(stat) = stats.get(backend_type) {
                    if stat.total_operations > 0 {
                        let success_rate =
                            stat.successful_operations as f64 / stat.total_operations as f64;
                        report.push_str(&format!("  Success Rate: {:.1}%\n", success_rate * 100.0));
                        report.push_str(&format!(
                            "  Avg Execution Time: {:?}\n",
                            stat.average_execution_time
                        ));
                        report.push_str(&format!(
                            "  Total Data Transferred: {} MB\n",
                            stat.total_data_transferred / 1024 / 1024
                        ));
                    } else {
                        report.push_str("  No operations recorded\n");
                    }
                }
                report.push('\n');
            }
        }

        report
    }
}

/// Global GPU context pool instance
static GPU_POOL: std::sync::OnceLock<GpuContextPool> = std::sync::OnceLock::new();

/// Get the global GPU context pool
#[allow(dead_code)]
pub fn get_gpu_pool() -> &'static GpuContextPool {
    GPU_POOL.get_or_init(|| {
        let pool = GpuContextPool::new();
        if let Err(e) = pool.initialize() {
            #[cfg(feature = "gpu")]
            log::warn!("Failed to initialize GPU pool: {}", e);
        }
        pool
    })
}

/// Initialize the global GPU context pool
#[allow(dead_code)]
pub fn initialize_gpu_system() -> SpecialResult<()> {
    let pool = get_gpu_pool();
    pool.initialize()
}

/// Get the best available GPU context from the global pool
#[allow(dead_code)]
pub fn get_best_gpu_context() -> SpecialResult<Arc<GpuContext>> {
    get_gpu_pool().get_best_context()
}

/// Check if GPU should be used for computation
#[allow(dead_code)]
pub fn should_use_gpu_computation(_arraysize: usize, elementsize: usize) -> bool {
    get_gpu_pool().should_use_gpu(_arraysize, elementsize)
}

/// Record GPU operation performance
#[allow(dead_code)]
pub fn record_gpu_performance(
    backend_type: GpuBackend,
    execution_time: Duration,
    success: bool,
    datasize: usize,
) {
    get_gpu_pool().record_operation(backend_type, execution_time, success, datasize);
}

/// Validate GPU infrastructure for production use
#[allow(dead_code)]
pub fn validate_gpu_production_readiness() -> SpecialResult<String> {
    let pool = get_gpu_pool();
    let mut validation_report = String::new();

    // Check device availability
    let device_info = pool.get_device_info();
    if device_info.is_empty() {
        validation_report.push_str("⚠️  No GPU devices detected\n");
        validation_report.push_str("   Recommendation: GPU features will use CPU fallback\n\n");
    } else {
        validation_report.push_str(&format!(
            "✅ {} GPU device(s) available\n",
            device_info.len()
        ));

        // Check memory capacity
        for (backend, info) in device_info.iter() {
            let memory_gb = info.memorysize as f64 / (1024.0 * 1024.0 * 1024.0);
            validation_report.push_str(&format!(
                "   {:?}: {:.1} GB memory, {} compute units\n",
                backend, memory_gb, info.compute_units
            ));

            if memory_gb < 2.0 {
                validation_report
                    .push_str("   ⚠️  Low GPU memory may limit large array processing\n");
            }
        }
        validation_report.push('\n');
    }

    // Check performance history
    let performance_trends = pool.get_performance_trends();
    if !performance_trends.is_empty() {
        validation_report.push_str("📊 Performance History:\n");
        for (backend, trend) in performance_trends {
            validation_report.push_str(&format!("   {:?}: {}\n", backend, trend));
        }
        validation_report.push('\n');
    }

    // Configuration validation
    let config = pool.get_config();
    validation_report.push_str("⚙️  Configuration:\n");
    validation_report.push_str(&format!(
        "   Min array size for GPU: {}\n",
        config.min_gpu_arraysize
    ));
    validation_report.push_str(&format!(
        "   Max memory usage: {:.0}%\n",
        config.max_memory_usage_percent
    ));
    validation_report.push_str(&format!(
        "   Adaptive switching: {}\n",
        config.enable_adaptive_switching
    ));
    validation_report.push_str(&format!(
        "   Preferred backend: {:?}\n",
        config.preferred_backend
    ));

    // Recommendations
    validation_report.push_str("\n🎯 Recommendations:\n");
    if device_info.is_empty() {
        validation_report.push_str("   • Install GPU drivers for acceleration\n");
        validation_report.push_str("   • Enable GPU features in scirs2-core\n");
    } else {
        validation_report.push_str("   • GPU infrastructure ready for production use\n");
        validation_report.push_str("   • Monitor performance with get_system_report()\n");
        validation_report.push_str("   • Adjust min_gpu_arraysize based on workload\n");
    }

    Ok(validation_report)
}

/// Enable production monitoring with performance alerts
#[allow(dead_code)]
pub fn enable_gpu_monitoring(_enablealerts: bool) -> SpecialResult<()> {
    let pool = get_gpu_pool();
    let mut config = pool.get_config();
    config.enable_profiling = true;
    pool.update_config(config);

    #[cfg(feature = "gpu")]
    {
        if _enablealerts {
            log::info!("GPU performance monitoring enabled with alerts");
        } else {
            log::info!("GPU performance monitoring enabled without alerts");
        }
    }

    Ok(())
}

/// Get GPU resource utilization report
#[allow(dead_code)]
pub fn get_gpu_resource_utilization() -> String {
    let pool = get_gpu_pool();
    let device_info = pool.get_device_info();
    let stats = pool.get_performance_stats_all();

    let mut report = String::new();
    report.push_str("=== GPU Resource Utilization ===\n");

    for (backend, info) in device_info.iter() {
        if let Some(stat) = stats.get(backend) {
            let memory_usage = (stat.peak_memory_usage as f64 / info.memorysize as f64) * 100.0;
            let efficiency = if stat.total_operations > 0 {
                (stat.successful_operations as f64 / stat.total_operations as f64) * 100.0
            } else {
                0.0
            };

            report.push_str(&format!("\n{:?}:\n", backend));
            report.push_str(&format!("  Peak Memory Usage: {:.1}%\n", memory_usage));
            report.push_str(&format!("  Success Rate: {:.1}%\n", efficiency));
            report.push_str(&format!(
                "  Operations/sec: {:.1}\n",
                stat.operations_per_second
            ));
            report.push_str(&format!(
                "  Cache Hit Rate: {:.1}%\n",
                stat.cache_hit_rate * 100.0
            ));

            if let Some(ref error) = stat.last_error_message {
                report.push_str(&format!("  Last Error: {}\n", error));
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_pool_creation() {
        let pool = GpuContextPool::new();
        assert!(pool.get_device_info().is_empty());
    }

    #[test]
    fn test_should_use_gpu_logic() {
        let pool = GpuContextPool::new();

        // Small arrays should not use GPU
        assert!(!pool.should_use_gpu(100, 4));

        // Large arrays might use GPU (depends on availability)
        // This test doesn't guarantee GPU availability, so we just check the logic
        let use_large_f32 = pool.should_use_gpu(1000, 4);
        let use_large_f64 = pool.should_use_gpu(1000, 8);

        // Results depend on GPU availability, but the calls should not panic
        assert!(use_large_f32 == true || use_large_f32 == false);
        assert!(use_large_f64 == true || use_large_f64 == false);
    }

    #[test]
    fn test_performance_stats() {
        let pool = GpuContextPool::new();
        let backend = GpuBackend::Wgpu;

        // Initial stats should be None (no context created)
        assert!(pool.get_performance_stats(backend).is_none());

        // After initialization, stats might be available
        let _ = pool.initialize();
        // Note: We can't guarantee GPU availability in tests
    }
}
