//! GPU device detection and management
//!
//! This module handles the detection and management of GPU devices across
//! different backends (CUDA, OpenCL, Metal, ROCm) with automatic fallback to CPU.

use std::fmt::Debug;

use super::config::{GpuBackend, GpuCapabilities};
use crate::error::{Result, TimeSeriesError};

/// GPU device manager for detecting and managing GPU devices
#[derive(Debug)]
pub struct GpuDeviceManager {
    /// Available devices
    devices: Vec<GpuCapabilities>,
    /// Current device
    current_device: Option<usize>,
}

impl GpuDeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        // Detect actual GPU devices when dependencies are available
        let mut devices = Vec::new();

        // Try to detect CUDA devices
        if let Some(cuda_devices) = Self::detect_cuda_devices() {
            devices.extend(cuda_devices);
        }

        // Try to detect OpenCL devices
        if let Some(opencl_devices) = Self::detect_opencl_devices() {
            devices.extend(opencl_devices);
        }

        // Try to detect Metal devices (Apple Silicon)
        if let Some(metal_devices) = Self::detect_metal_devices() {
            devices.extend(metal_devices);
        }

        // Try to detect ROCm devices (AMD)
        if let Some(rocm_devices) = Self::detect_rocm_devices() {
            devices.extend(rocm_devices);
        }

        // Always provide CPU fallback if no GPU devices found
        if devices.is_empty() {
            devices.push(GpuCapabilities {
                backend: GpuBackend::CpuFallback,
                compute_capability: None,
                memory: Self::get_system_memory(),
                multiprocessors: Self::get_cpu_cores(),
                supports_fp16: false,
                supports_tensor_cores: false,
                max_threads_per_block: 1,
                tensor_cores_generation: None,
                memory_bandwidth: 100.0, // GB/s - rough estimate for system memory
                tensor_performance: None,
            });
        }

        Ok(Self {
            devices,
            current_device: Some(0), // Default to first device
        })
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GpuCapabilities] {
        &self.devices
    }

    /// Set current device
    pub fn set_device(&mut self, deviceid: usize) -> Result<()> {
        if deviceid >= self.devices.len() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Device {deviceid} not available"
            )));
        }
        self.current_device = Some(deviceid);
        Ok(())
    }

    /// Get current device capabilities
    pub fn current_device_capabilities(&self) -> Option<&GpuCapabilities> {
        self.current_device.map(|id| &self.devices[id])
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.devices
            .iter()
            .any(|dev| !matches!(dev.backend, GpuBackend::CpuFallback))
    }

    /// Detect CUDA devices.
    ///
    /// This crate does not link the CUDA Runtime/Driver API, so it can only
    /// detect the *presence* of an NVIDIA driver (via the device nodes under
    /// `/dev` and `/proc`). It deliberately does **not** fabricate the device's
    /// capabilities (compute capability, VRAM, tensor-core generation, etc.):
    /// those are reported as unknown because they cannot be queried without the
    /// runtime. Returns `None` when no driver is detected.
    fn detect_cuda_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/proc/driver/nvidia").exists()
            {
                return Some(vec![Self::detected_device_unknown_specs(GpuBackend::Cuda)]);
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, detecting CUDA would require querying nvml.dll or WMI,
            // which is not wired in here. Report no device rather than guessing.
        }

        None
    }

    /// Build a [`GpuCapabilities`] for a device whose backend has been detected
    /// but whose hardware specifications are unknown (because no vendor runtime
    /// is linked to query them).
    ///
    /// All capability fields are set to conservative "unknown" values: zero
    /// memory/bandwidth, no compute capability, no tensor-core support. This
    /// signals that a device of the given backend is present without claiming
    /// any specific (and potentially false) performance characteristics.
    fn detected_device_unknown_specs(backend: GpuBackend) -> GpuCapabilities {
        GpuCapabilities {
            backend,
            compute_capability: None, // unknown without the runtime
            memory: 0,                // unknown (not 40 GB, etc.)
            multiprocessors: 0,       // unknown
            supports_fp16: false,     // unknown -> conservatively false
            supports_tensor_cores: false,
            max_threads_per_block: 0, // unknown
            tensor_cores_generation: None,
            memory_bandwidth: 0.0,    // unknown
            tensor_performance: None, // unknown
        }
    }

    /// Detect OpenCL devices.
    ///
    /// Only the presence of OpenCL ICD/driver libraries is detected; the actual
    /// platform/device enumeration (and therefore the real capabilities) is not
    /// performed because no OpenCL bindings are linked. Capabilities are
    /// reported as unknown rather than fabricated. Returns `None` when no
    /// OpenCL driver is found.
    fn detect_opencl_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        {
            if Self::has_opencl_drivers() {
                return Some(vec![Self::detected_device_unknown_specs(
                    GpuBackend::OpenCL,
                )]);
            }
        }

        None
    }

    /// Detect Metal devices (Apple Silicon).
    ///
    /// Detects whether the process runs on Apple Silicon / has the Metal
    /// framework available, but does not query the actual GPU via Metal, so the
    /// device's real capabilities are reported as unknown rather than
    /// fabricated. Returns `None` when no Metal-capable device is detected.
    fn detect_metal_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "macos")]
        {
            if Self::is_apple_silicon() || Self::has_metal_gpu() {
                return Some(vec![Self::detected_device_unknown_specs(GpuBackend::Metal)]);
            }
        }

        None
    }

    /// Detect ROCm devices (AMD).
    ///
    /// Only the presence of a ROCm installation / KFD device node is detected;
    /// the device's real capabilities are not queried (no ROCm bindings are
    /// linked) and are therefore reported as unknown rather than fabricated.
    /// Returns `None` when no ROCm device is detected.
    fn detect_rocm_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/opt/rocm").exists()
                || std::path::Path::new("/dev/kfd").exists()
            {
                return Some(vec![Self::detected_device_unknown_specs(GpuBackend::Rocm)]);
            }
        }

        None
    }

    /// Check for OpenCL drivers
    fn has_opencl_drivers() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so").exists()
                || std::path::Path::new("/usr/lib64/libOpenCL.so").exists()
        }
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:/Windows/System32/OpenCL.dll").exists()
        }
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }

    /// Check if running on Apple Silicon
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        std::env::consts::ARCH == "aarch64"
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        false
    }

    /// Check for Metal GPU
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        false
    }

    /// Get system memory size
    fn get_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /proc/meminfo
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Default to 8GB if detection fails
        8 * 1024 * 1024 * 1024
    }

    /// Get number of CPU cores
    fn get_cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4) // Default to 4 cores
    }
}

impl Default for GpuDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            devices: vec![],
            current_device: None,
        })
    }
}
