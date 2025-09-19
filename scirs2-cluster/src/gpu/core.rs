//! Core GPU types and configurations
//!
//! This module provides the fundamental types for GPU acceleration support,
//! including backend selection, device information, and basic configuration.

use crate::error::{ClusteringError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPU acceleration backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// OpenCL backend (cross-platform)
    OpenCl,
    /// AMD ROCm backend
    Rocm,
    /// Intel OneAPI backend
    OneApi,
    /// Apple Metal Performance Shaders
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

impl Default for GpuBackend {
    fn default() -> Self {
        GpuBackend::CpuFallback
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::OpenCl => write!(f, "OpenCL"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::OneApi => write!(f, "Intel OneAPI"),
            GpuBackend::Metal => write!(f, "Apple Metal"),
            GpuBackend::CpuFallback => write!(f, "CPU Fallback"),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device ID
    pub device_id: u32,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability or equivalent
    pub compute_capability: String,
    /// Number of compute units
    pub compute_units: u32,
    /// Backend type
    pub backend: GpuBackend,
    /// Whether device supports double precision
    pub supports_double_precision: bool,
}

impl GpuDevice {
    /// Create a new GPU device
    pub fn new(
        device_id: u32,
        name: String,
        total_memory: usize,
        available_memory: usize,
        compute_capability: String,
        compute_units: u32,
        backend: GpuBackend,
        supports_double_precision: bool,
    ) -> Self {
        Self {
            device_id,
            name,
            total_memory,
            available_memory,
            compute_capability,
            compute_units,
            backend,
            supports_double_precision,
        }
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            100.0 * (1.0 - (self.available_memory as f64 / self.total_memory as f64))
        }
    }

    /// Check if device is suitable for double precision computations
    pub fn is_suitable_for_double_precision(&self) -> bool {
        self.supports_double_precision
    }

    /// Get device score for selection (higher is better)
    pub fn get_device_score(&self) -> f64 {
        let memory_score = self.available_memory as f64 / 1_000_000_000.0; // GB
        let compute_score = self.compute_units as f64;
        let precision_bonus = if self.supports_double_precision {
            1.5
        } else {
            1.0
        };

        (memory_score + compute_score) * precision_bonus
    }
}

/// Device selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelection {
    /// Use first available device
    First,
    /// Use device with most memory
    MostMemory,
    /// Use device with highest compute capability
    HighestCompute,
    /// Use specific device by ID
    Specific(u32),
    /// Automatic selection based on workload
    Auto,
    /// Use fastest device for current workload
    Fastest,
}

impl Default for DeviceSelection {
    fn default() -> Self {
        DeviceSelection::Auto
    }
}

/// GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Preferred backend (will fallback if not available)
    pub preferred_backend: GpuBackend,
    /// Device selection strategy
    pub device_selection: DeviceSelection,
    /// Enable automatic CPU fallback
    pub auto_fallback: bool,
    /// Memory pool size in bytes (None for automatic)
    pub memory_pool_size: Option<usize>,
    /// Enable memory management optimization
    pub optimize_memory: bool,
    /// Custom backend-specific options
    pub backend_options: HashMap<String, String>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: GpuBackend::CpuFallback,
            device_selection: DeviceSelection::Auto,
            auto_fallback: true,
            memory_pool_size: None,
            optimize_memory: true,
            backend_options: HashMap::new(),
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            preferred_backend: backend,
            ..Default::default()
        }
    }

    /// Set device selection strategy
    pub fn with_device_selection(mut self, strategy: DeviceSelection) -> Self {
        self.device_selection = strategy;
        self
    }

    /// Set memory pool size
    pub fn with_memory_pool_size(mut self, size: usize) -> Self {
        self.memory_pool_size = Some(size);
        self
    }

    /// Disable automatic CPU fallback
    pub fn without_fallback(mut self) -> Self {
        self.auto_fallback = false;
        self
    }

    /// Add backend-specific option
    pub fn with_backend_option(mut self, key: String, value: String) -> Self {
        self.backend_options.insert(key, value);
        self
    }

    /// Create CUDA configuration
    pub fn cuda() -> Self {
        Self::new(GpuBackend::Cuda)
    }

    /// Create OpenCL configuration
    pub fn opencl() -> Self {
        Self::new(GpuBackend::OpenCl)
    }

    /// Create ROCm configuration
    pub fn rocm() -> Self {
        Self::new(GpuBackend::Rocm)
    }

    /// Create Metal configuration
    pub fn metal() -> Self {
        Self::new(GpuBackend::Metal)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if let DeviceSelection::Specific(id) = self.device_selection {
            if id > 64 {
                return Err(ClusteringError::InvalidInput(
                    "Device ID too high".to_string(),
                ));
            }
        }

        if let Some(pool_size) = self.memory_pool_size {
            if pool_size < 1024 * 1024 {
                return Err(ClusteringError::InvalidInput(
                    "Memory pool size too small (minimum 1MB)".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// GPU context for clustering operations
#[derive(Debug)]
pub struct GpuContext {
    /// Current device
    pub device: GpuDevice,
    /// Configuration
    pub config: GpuConfig,
    /// Whether GPU is actually available
    pub gpu_available: bool,
    /// Backend-specific context data
    pub backend_context: BackendContext,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new(device: GpuDevice, config: GpuConfig) -> Result<Self> {
        config.validate()?;

        let gpu_available = Self::check_gpu_availability(&device, &config);
        let backend_context = BackendContext::new(&device.backend)?;

        Ok(Self {
            device,
            config,
            gpu_available,
            backend_context,
        })
    }

    /// Check if GPU acceleration is available
    fn check_gpu_availability(device: &GpuDevice, config: &GpuConfig) -> bool {
        // Simplified availability check
        match (device.backend, config.preferred_backend) {
            (GpuBackend::CpuFallback, _) => false,
            (backend1, backend2) if backend1 == backend2 => true,
            _ => config.auto_fallback,
        }
    }

    /// Get effective backend (considering fallback)
    pub fn effective_backend(&self) -> GpuBackend {
        if self.gpu_available {
            self.device.backend
        } else {
            GpuBackend::CpuFallback
        }
    }

    /// Check if using GPU acceleration
    pub fn is_gpu_accelerated(&self) -> bool {
        self.gpu_available && self.device.backend != GpuBackend::CpuFallback
    }

    /// Get memory information
    pub fn memory_info(&self) -> (usize, usize) {
        (self.device.total_memory, self.device.available_memory)
    }
}

/// Backend-specific context (placeholder for actual implementations)
#[derive(Debug)]
pub enum BackendContext {
    /// CUDA context placeholder
    Cuda {
        /// Device context handle
        context_handle: u64,
        /// Stream handle
        stream_handle: u64,
    },
    /// OpenCL context placeholder
    OpenCl {
        /// Context handle
        context_handle: u64,
        /// Command queue handle
        queue_handle: u64,
    },
    /// ROCm context placeholder
    Rocm {
        /// HIP context handle
        context_handle: u64,
    },
    /// OneAPI context placeholder
    OneApi {
        /// SYCL context handle
        context_handle: u64,
    },
    /// Metal context placeholder
    Metal {
        /// Metal device handle
        device_handle: u64,
        /// Command queue handle
        queue_handle: u64,
    },
    /// CPU fallback (no context needed)
    CpuFallback,
}

impl BackendContext {
    /// Create a new backend context
    pub fn new(backend: &GpuBackend) -> Result<Self> {
        match backend {
            GpuBackend::Cuda => Ok(BackendContext::Cuda {
                context_handle: 0, // Placeholder
                stream_handle: 0,
            }),
            GpuBackend::OpenCl => Ok(BackendContext::OpenCl {
                context_handle: 0, // Placeholder
                queue_handle: 0,
            }),
            GpuBackend::Rocm => Ok(BackendContext::Rocm {
                context_handle: 0, // Placeholder
            }),
            GpuBackend::OneApi => Ok(BackendContext::OneApi {
                context_handle: 0, // Placeholder
            }),
            GpuBackend::Metal => Ok(BackendContext::Metal {
                device_handle: 0, // Placeholder
                queue_handle: 0,
            }),
            GpuBackend::CpuFallback => Ok(BackendContext::CpuFallback),
        }
    }

    /// Check if context is valid
    pub fn is_valid(&self) -> bool {
        match self {
            BackendContext::CpuFallback => true,
            _ => true, // Simplified for stub implementation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice::new(
            0,
            "Test GPU".to_string(),
            8_000_000_000, // 8GB
            6_000_000_000, // 6GB available
            "7.5".to_string(),
            2048,
            GpuBackend::Cuda,
            true,
        );

        assert_eq!(device.device_id, 0);
        assert_eq!(device.name, "Test GPU");
        assert_eq!(device.memory_utilization(), 25.0); // 25% used
        assert!(device.is_suitable_for_double_precision());
    }

    #[test]
    fn test_gpu_config_validation() {
        let config = GpuConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = GpuConfig::default().with_memory_pool_size(1024); // Too small
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_device_selection_strategies() {
        assert_eq!(DeviceSelection::default(), DeviceSelection::Auto);

        let specific = DeviceSelection::Specific(0);
        if let DeviceSelection::Specific(id) = specific {
            assert_eq!(id, 0);
        }
    }

    #[test]
    fn test_backend_context_creation() {
        let cuda_context = BackendContext::new(&GpuBackend::Cuda).unwrap();
        assert!(cuda_context.is_valid());

        let cpu_context = BackendContext::new(&GpuBackend::CpuFallback).unwrap();
        assert!(cpu_context.is_valid());
    }
}
