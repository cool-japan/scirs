//! # Resource Discovery and Hardware Detection
//!
//! This module provides automatic hardware detection and resource discovery
//! capabilities for optimizing `SciRS2` Core performance based on available
//! system resources. It includes:
//! - CPU detection and optimization parameter tuning
//! - Memory hierarchy analysis
//! - GPU discovery and capability assessment
//! - Network and storage resource detection
//! - Dynamic optimization parameter adjustment

pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod network;
pub mod optimization;
pub mod storage;

use crate::error::{CoreError, CoreResult};
// Collections are used in the SystemResources struct
use std::time::Duration;

/// System resource information
#[derive(Debug, Clone)]
pub struct SystemResources {
    /// CPU information
    pub cpu: cpu::CpuInfo,
    /// Memory information
    pub memory: memory::MemoryInfo,
    /// GPU information (if available)
    pub gpu: Option<gpu::GpuInfo>,
    /// Network information
    pub network: network::NetworkInfo,
    /// Storage information
    pub storage: storage::StorageInfo,
    /// Optimization recommendations
    pub optimization_params: optimization::OptimizationParams,
}

impl SystemResources {
    /// Discover all system resources
    pub fn discover() -> CoreResult<Self> {
        let cpu = cpu::CpuInfo::detect()?;
        let memory = memory::MemoryInfo::detect()?;
        let gpu = gpu::GpuInfo::detect().ok();
        let network = network::NetworkInfo::detect()?;
        let storage = storage::StorageInfo::detect()?;

        let optimization_params = optimization::OptimizationParams::generate(
            &cpu,
            &memory,
            gpu.as_ref(),
            &network,
            &storage,
        )?;

        Ok(Self {
            cpu,
            memory,
            gpu,
            network,
            storage,
            optimization_params,
        })
    }

    /// Get recommended thread count for parallel operations
    pub fn recommended_thread_count(&self) -> usize {
        self.optimization_params.thread_count
    }

    /// Get recommended chunk size for memory operations
    pub fn recommended_chunk_size(&self) -> usize {
        self.optimization_params.chunk_size
    }

    /// Check if SIMD operations are supported
    pub fn supports_simd(&self) -> bool {
        self.cpu.simd_capabilities.avx2 || self.cpu.simd_capabilities.sse4_2
    }

    /// Check if GPU acceleration is available
    pub fn supports_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get total available memory in bytes
    pub fn total_memory(&self) -> usize {
        self.memory.total_memory
    }

    /// Get available memory in bytes
    pub fn available_memory(&self) -> usize {
        self.memory.available_memory
    }

    /// Get performance tier classification
    pub fn performance_tier(&self) -> PerformanceTier {
        let cpu_score = self.cpu.performance_score();
        let memory_score = self.memory.performance_score();
        let gpu_score = self
            .gpu
            .as_ref()
            .map(|g| g.performance_score())
            .unwrap_or(0.0);

        let combined_score = (cpu_score + memory_score + gpu_score) / 3.0;

        if combined_score >= 0.8 {
            PerformanceTier::High
        } else if combined_score >= 0.5 {
            PerformanceTier::Medium
        } else {
            PerformanceTier::Low
        }
    }

    /// Generate a system summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# System Resource Summary\n\n");

        // CPU information
        report.push_str("## CPU\n");
        report.push_str(&format!("- Model: {}\n", self.cpu.model));
        report.push_str(&format!(
            "- Cores: {} physical, {} logical\n",
            self.cpu.physical_cores, self.cpu.logical_cores
        ));
        report.push_str(&format!(
            "- Base frequency: {:.2} GHz\n",
            self.cpu.base_frequency_ghz
        ));
        report.push_str(&format!(
            "- Cache L1: {} KB, L2: {} KB, L3: {} KB\n",
            self.cpu.cache_l1_kb, self.cpu.cache_l2_kb, self.cpu.cache_l3_kb
        ));

        // SIMD capabilities
        report.push_str("- SIMD support:");
        if self.cpu.simd_capabilities.sse4_2 {
            report.push_str(" SSE4.2");
        }
        if self.cpu.simd_capabilities.avx2 {
            report.push_str(" AVX2");
        }
        if self.cpu.simd_capabilities.avx512 {
            report.push_str(" AVX512");
        }
        if self.cpu.simd_capabilities.neon {
            report.push_str(" NEON");
        }
        report.push('\n');

        // Memory information
        report.push_str("\n## Memory\n");
        report.push_str(&format!(
            "- Total: {:.2} GB\n",
            self.memory.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        report.push_str(&format!(
            "- Available: {:.2} GB\n",
            self.memory.available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        report.push_str(&format!(
            "- Page size: {} KB\n",
            self.memory.page_size / 1024
        ));

        // GPU information
        if let Some(ref gpu) = self.gpu {
            report.push_str("\n## GPU\n");
            report.push_str(&format!("- Model: {}\n", gpu.name));
            report.push_str(&format!(
                "- Memory: {:.2} GB\n",
                gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
            report.push_str(&format!("- Compute units: {}\n", gpu.compute_units));
        }

        // Optimization recommendations
        report.push_str("\n## Optimization Recommendations\n");
        report.push_str(&format!(
            "- Thread count: {}\n",
            self.optimization_params.thread_count
        ));
        report.push_str(&format!(
            "- Chunk size: {} KB\n",
            self.optimization_params.chunk_size / 1024
        ));
        report.push_str(&format!(
            "- SIMD enabled: {}\n",
            self.optimization_params.enable_simd
        ));
        report.push_str(&format!(
            "- GPU enabled: {}\n",
            self.optimization_params.enable_gpu
        ));

        // Performance tier
        report.push_str(&format!(
            "\n## Performance Tier: {:?}\n",
            self.performance_tier()
        ));

        report
    }
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTier {
    /// High-performance system (server, workstation)
    High,
    /// Medium-performance system (desktop, laptop)
    Medium,
    /// Low-performance system (embedded, older hardware)
    Low,
}

/// Resource discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Enable CPU detection
    pub detect_cpu: bool,
    /// Enable memory detection
    pub detect_memory: bool,
    /// Enable GPU detection
    pub detect_gpu: bool,
    /// Enable network detection
    pub detect_network: bool,
    /// Enable storage detection
    pub detect_storage: bool,
    /// Cache discovery results
    pub cache_results: bool,
    /// Cache duration
    pub cache_duration: Duration,
    /// Enable detailed detection (may be slower)
    pub detailed_detection: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            detect_cpu: true,
            detect_memory: true,
            detect_gpu: true,
            detect_network: true,
            detect_storage: true,
            cache_results: true,
            cache_duration: Duration::from_secs(300), // 5 minutes
            detailed_detection: false,
        }
    }
}

impl DiscoveryConfig {
    /// Create a new discovery configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable all detection
    pub fn detect_all(mut self) -> Self {
        self.detect_cpu = true;
        self.detect_memory = true;
        self.detect_gpu = true;
        self.detect_network = true;
        self.detect_storage = true;
        self
    }

    /// Disable all detection
    pub fn detect_none(mut self) -> Self {
        self.detect_cpu = false;
        self.detect_memory = false;
        self.detect_gpu = false;
        self.detect_network = false;
        self.detect_storage = false;
        self
    }

    /// Enable only essential detection (CPU and memory)
    pub fn detect_essential(mut self) -> Self {
        self.detect_cpu = true;
        self.detect_memory = true;
        self.detect_gpu = false;
        self.detect_network = false;
        self.detect_storage = false;
        self
    }

    /// Enable caching with custom duration
    pub fn with_cache_duration(mut self, duration: Duration) -> Self {
        self.cache_results = true;
        self.cache_duration = duration;
        self
    }

    /// Enable detailed detection
    pub fn with_detailed_detection(mut self, enabled: bool) -> Self {
        self.detailed_detection = enabled;
        self
    }
}

/// Resource discovery manager with caching
pub struct ResourceDiscovery {
    config: DiscoveryConfig,
    cached_resources: std::sync::Mutex<Option<(SystemResources, std::time::Instant)>>,
}

impl ResourceDiscovery {
    /// Create a new resource discovery manager
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            cached_resources: std::sync::Mutex::new(None),
        }
    }
}

impl Default for ResourceDiscovery {
    fn default() -> Self {
        Self::new(DiscoveryConfig::default())
    }
}

impl ResourceDiscovery {
    /// Discover system resources with caching
    pub fn discover(&self) -> CoreResult<SystemResources> {
        if self.config.cache_results {
            if let Ok(cache) = self.cached_resources.lock() {
                if let Some((ref resources, timestamp)) = *cache {
                    if timestamp.elapsed() < self.config.cache_duration {
                        return Ok(resources.clone());
                    }
                }
            }
        }

        // Perform discovery
        let resources = self.discover_fresh()?;

        // Update cache
        if self.config.cache_results {
            if let Ok(mut cache) = self.cached_resources.lock() {
                *cache = Some((resources.clone(), std::time::Instant::now()));
            }
        }

        Ok(resources)
    }

    /// Force fresh discovery without cache
    pub fn discover_fresh(&self) -> CoreResult<SystemResources> {
        let cpu = if self.config.detect_cpu {
            cpu::CpuInfo::detect()?
        } else {
            cpu::CpuInfo::default()
        };

        let memory = if self.config.detect_memory {
            memory::MemoryInfo::detect()?
        } else {
            memory::MemoryInfo::default()
        };

        let gpu = if self.config.detect_gpu {
            gpu::GpuInfo::detect().ok()
        } else {
            None
        };

        let network = if self.config.detect_network {
            network::NetworkInfo::detect()?
        } else {
            network::NetworkInfo::default()
        };

        let storage = if self.config.detect_storage {
            storage::StorageInfo::detect()?
        } else {
            storage::StorageInfo::default()
        };

        let optimization_params = optimization::OptimizationParams::generate(
            &cpu,
            &memory,
            gpu.as_ref(),
            &network,
            &storage,
        )?;

        Ok(SystemResources {
            cpu,
            memory,
            gpu,
            network,
            storage,
            optimization_params,
        })
    }

    /// Clear cache
    pub fn clear_cache(&self) -> CoreResult<()> {
        if let Ok(mut cache) = self.cached_resources.lock() {
            *cache = None;
            Ok(())
        } else {
            Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Failed to clear cache"),
            ))
        }
    }

    /// Get cache status
    pub fn cache_status(&self) -> CoreResult<Option<Duration>> {
        if let Ok(cache) = self.cached_resources.lock() {
            if let Some((_, timestamp)) = cache.as_ref() {
                Ok(Some(timestamp.elapsed()))
            } else {
                Ok(None)
            }
        } else {
            Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Failed to read cache status"),
            ))
        }
    }
}

/// Global resource discovery instance
static GLOBAL_RESOURCE_DISCOVERY: std::sync::LazyLock<ResourceDiscovery> =
    std::sync::LazyLock::new(ResourceDiscovery::default);

/// Get the global resource discovery instance
pub fn global_resource_discovery() -> &'static ResourceDiscovery {
    &GLOBAL_RESOURCE_DISCOVERY
}

/// Quick access functions for common operations
/// Get system resources using global discovery
pub fn get_system_resources() -> CoreResult<SystemResources> {
    global_resource_discovery().discover()
}

/// Get recommended thread count
pub fn get_recommended_thread_count() -> CoreResult<usize> {
    Ok(get_system_resources()?.recommended_thread_count())
}

/// Get recommended chunk size
pub fn get_recommended_chunk_size() -> CoreResult<usize> {
    Ok(get_system_resources()?.recommended_chunk_size())
}

/// Check if SIMD is supported
pub fn is_simd_supported() -> CoreResult<bool> {
    Ok(get_system_resources()?.supports_simd())
}

/// Check if GPU is available
pub fn is_gpu_available() -> CoreResult<bool> {
    Ok(get_system_resources()?.supports_gpu())
}

/// Get total system memory
pub fn get_total_memory() -> CoreResult<usize> {
    Ok(get_system_resources()?.total_memory())
}

/// Get available system memory
pub fn get_available_memory() -> CoreResult<usize> {
    Ok(get_system_resources()?.available_memory())
}

/// Get performance tier
pub fn get_performance_tier() -> CoreResult<PerformanceTier> {
    Ok(get_system_resources()?.performance_tier())
}

/// Resource monitoring for adaptive optimization
pub struct ResourceMonitor {
    discovery: ResourceDiscovery,
    monitoring_interval: Duration,
    last_update: std::sync::Mutex<std::time::Instant>,
    adaptive_params: std::sync::Mutex<optimization::OptimizationParams>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(config: DiscoveryConfig, monitoring_interval: Duration) -> Self {
        let discovery = ResourceDiscovery::new(config);

        Self {
            discovery,
            monitoring_interval,
            last_update: std::sync::Mutex::new(std::time::Instant::now()),
            adaptive_params: std::sync::Mutex::new(optimization::OptimizationParams::default()),
        }
    }

    /// Update optimization parameters based on current resource state
    pub fn update_optimization_params(&self) -> CoreResult<optimization::OptimizationParams> {
        let should_update = {
            if let Ok(last_update) = self.last_update.lock() {
                last_update.elapsed() >= self.monitoring_interval
            } else {
                true
            }
        };

        if should_update {
            let resources = self.discovery.discover_fresh()?;

            // Update cached parameters
            if let Ok(mut params) = self.adaptive_params.lock() {
                *params = resources.optimization_params.clone();
            }

            // Update timestamp
            if let Ok(mut last_update) = self.last_update.lock() {
                *last_update = std::time::Instant::now();
            }

            Ok(resources.optimization_params)
        } else {
            // Return cached parameters
            if let Ok(params) = self.adaptive_params.lock() {
                Ok(params.clone())
            } else {
                Err(CoreError::ComputationError(
                    crate::error::ErrorContext::new("Failed to read adaptive parameters"),
                ))
            }
        }
    }

    /// Get current optimization parameters
    pub fn current_params(&self) -> CoreResult<optimization::OptimizationParams> {
        if let Ok(params) = self.adaptive_params.lock() {
            Ok(params.clone())
        } else {
            Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Failed to read current parameters"),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_config() {
        let config = DiscoveryConfig::new()
            .detect_essential()
            .with_cache_duration(Duration::from_secs(60))
            .with_detailed_detection(true);

        assert!(config.detect_cpu);
        assert!(config.detect_memory);
        assert!(!config.detect_gpu);
        assert_eq!(config.cache_duration, Duration::from_secs(60));
        assert!(config.detailed_detection);
    }

    #[test]
    fn test_performance_tier() {
        assert_eq!(PerformanceTier::High, PerformanceTier::High);
        assert_ne!(PerformanceTier::High, PerformanceTier::Low);
    }

    #[test]
    fn test_resource_discovery() {
        let config = DiscoveryConfig::new().detect_essential();
        let discovery = ResourceDiscovery::new(config);

        // This should work on any system
        let resources = discovery.discover();
        assert!(resources.is_ok());
    }

    #[test]
    fn test_global_functions() {
        // These should work on any system
        let thread_count = get_recommended_thread_count();
        assert!(thread_count.is_ok());
        assert!(thread_count.unwrap() > 0);

        let chunk_size = get_recommended_chunk_size();
        assert!(chunk_size.is_ok());
        assert!(chunk_size.unwrap() > 0);
    }

    #[test]
    fn test_resource_monitor() {
        let config = DiscoveryConfig::new().detect_essential();
        let monitor = ResourceMonitor::new(config, Duration::from_secs(1));

        let params = monitor.update_optimization_params();
        assert!(params.is_ok());

        let current = monitor.current_params();
        assert!(current.is_ok());
    }
}
