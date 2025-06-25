//! GPU backend implementations and detection utilities
//!
//! This module contains backend-specific implementations for various GPU platforms
//! and utilities for detecting available GPU backends.

use crate::gpu::{GpuBackend, GpuError};
use std::process::Command;

#[cfg(target_os = "macos")]
use serde_json;

// Backend implementation modules
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_mps;

// Re-export backend implementations
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{MetalBufferOptions, MetalContext, MetalStorageMode};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_mps::{MPSContext, MPSDataType, MPSOperations};

/// Information about available GPU hardware
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// The GPU backend type
    pub backend: GpuBackend,
    /// Device name
    pub device_name: String,
    /// Available memory in bytes
    pub memory_bytes: Option<u64>,
    /// Compute capability or equivalent
    pub compute_capability: Option<String>,
    /// Whether the device supports tensor operations
    pub supports_tensors: bool,
}

/// Detection results for all available GPU backends
#[derive(Debug, Clone)]
pub struct GpuDetectionResult {
    /// Available GPU devices
    pub devices: Vec<GpuInfo>,
    /// Recommended backend for scientific computing
    pub recommended_backend: GpuBackend,
}

/// Detect available GPU backends and devices
pub fn detect_gpu_backends() -> GpuDetectionResult {
    let mut devices = Vec::new();

    // Skip GPU detection in test environment to avoid segfaults from external commands
    #[cfg(not(test))]
    {
        // Detect CUDA devices
        if let Ok(cuda_devices) = detect_cuda_devices() {
            devices.extend(cuda_devices);
        }

        // Detect ROCm devices
        if let Ok(rocm_devices) = detect_rocm_devices() {
            devices.extend(rocm_devices);
        }

        // Detect Metal devices (macOS)
        #[cfg(target_os = "macos")]
        if let Ok(metal_devices) = detect_metal_devices() {
            devices.extend(metal_devices);
        }

        // Detect OpenCL devices
        if let Ok(opencl_devices) = detect_opencl_devices() {
            devices.extend(opencl_devices);
        }
    }

    // Determine recommended backend
    let recommended_backend = if devices
        .iter()
        .any(|d: &GpuInfo| d.backend == GpuBackend::Cuda)
    {
        GpuBackend::Cuda
    } else if devices
        .iter()
        .any(|d: &GpuInfo| d.backend == GpuBackend::Rocm)
    {
        GpuBackend::Rocm
    } else if devices
        .iter()
        .any(|d: &GpuInfo| d.backend == GpuBackend::Metal)
    {
        GpuBackend::Metal
    } else if devices
        .iter()
        .any(|d: &GpuInfo| d.backend == GpuBackend::OpenCL)
    {
        GpuBackend::OpenCL
    } else {
        GpuBackend::Cpu
    };

    // Always add CPU fallback
    devices.push(GpuInfo {
        backend: GpuBackend::Cpu,
        device_name: "CPU".to_string(),
        memory_bytes: None,
        compute_capability: None,
        supports_tensors: false,
    });

    GpuDetectionResult {
        devices,
        recommended_backend,
    }
}

/// Detect ROCm devices using rocm-smi
fn detect_rocm_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to run rocm-smi to detect ROCm devices
    match Command::new("rocm-smi")
        .arg("--showproductname")
        .arg("--showmeminfo")
        .arg("vram")
        .arg("--csv")
        .output()
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            for line in output_str.lines().skip(1) {
                // Skip header line
                if line.trim().is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let device_name = parts[1].trim_matches('"').to_string();
                    let memory_str = parts[2].trim_matches('"');

                    // Parse memory (format might be like "16368 MB")
                    let memory_mb = memory_str
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0)
                        * 1024
                        * 1024; // Convert MB to bytes

                    devices.push(GpuInfo {
                        backend: GpuBackend::Rocm,
                        device_name,
                        memory_bytes: Some(memory_mb),
                        compute_capability: Some("RDNA/CDNA".to_string()),
                        supports_tensors: true, // Modern AMD GPUs support matrix operations
                    });
                }
            }
        }
        _ => {
            // rocm-smi not available or failed
            // In a real implementation, we could try other methods like:
            // - Direct HIP runtime API calls
            // - /sys/class/drm/cardX/ on Linux
            // - rocminfo command
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("ROCm".to_string()))
    } else {
        Ok(devices)
    }
}

/// Detect CUDA devices using nvidia-ml-py or nvidia-smi
fn detect_cuda_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to run nvidia-smi to detect CUDA devices
    match Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let device_name = parts[0].to_string();
                    let memory_mb = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // Convert MB to bytes
                    let compute_capability = parts[2].to_string();

                    // Parse compute capability to determine tensor core support
                    let supports_tensors =
                        if let Some(major_str) = compute_capability.split('.').next() {
                            major_str.parse::<u32>().unwrap_or(0) >= 7 // Tensor cores available on Volta+ (7.0+)
                        } else {
                            false
                        };

                    devices.push(GpuInfo {
                        backend: GpuBackend::Cuda,
                        device_name,
                        memory_bytes: Some(memory_mb),
                        compute_capability: Some(compute_capability),
                        supports_tensors,
                    });
                }
            }
        }
        _ => {
            // nvidia-smi not available or failed
            // In a real implementation, we could try other methods like:
            // - Direct CUDA runtime API calls
            // - nvidia-ml-py if available
            // - /proc/driver/nvidia/gpus/ on Linux
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("CUDA".to_string()))
    } else {
        Ok(devices)
    }
}

/// Detect Metal devices (macOS only)
#[cfg(target_os = "macos")]
fn detect_metal_devices() -> Result<Vec<GpuInfo>, GpuError> {
    use std::str::FromStr;

    let mut devices = Vec::new();

    // Try to detect Metal devices using system_profiler
    match Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .arg("-json")
        .output()
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            // Try to parse JSON output
            if let Ok(json_value) = serde_json::Value::from_str(&output_str) {
                if let Some(displays) = json_value
                    .get("SPDisplaysDataType")
                    .and_then(|v| v.as_array())
                {
                    for display in displays {
                        // Extract GPU information from each display
                        if let Some(model) = display.get("sppci_model").and_then(|v| v.as_str()) {
                            let mut gpu_info = GpuInfo {
                                backend: GpuBackend::Metal,
                                device_name: model.to_string(),
                                memory_bytes: None,
                                compute_capability: None,
                                supports_tensors: true,
                            };

                            // Try to extract VRAM if available
                            if let Some(vram_str) = display
                                .get("vram_pcie")
                                .and_then(|v| v.as_str())
                                .or_else(|| display.get("vram").and_then(|v| v.as_str()))
                            {
                                // Parse VRAM string like "8 GB" or "8192 MB"
                                if let Some(captures) = regex::Regex::new(r"(\d+)\s*(GB|MB)")
                                    .ok()
                                    .and_then(|re| re.captures(vram_str))
                                {
                                    if let (Some(value), Some(unit)) =
                                        (captures.get(1), captures.get(2))
                                    {
                                        if let Ok(num) = u64::from_str(value.as_str()) {
                                            gpu_info.memory_bytes = Some(match unit.as_str() {
                                                "GB" => num * 1024 * 1024 * 1024,
                                                "MB" => num * 1024 * 1024,
                                                _ => 0,
                                            });
                                        }
                                    }
                                }
                            }

                            // Extract Metal family support
                            if let Some(metal_family) =
                                display.get("sppci_metal_family").and_then(|v| v.as_str())
                            {
                                gpu_info.compute_capability = Some(metal_family.to_string());
                            }

                            devices.push(gpu_info);
                        }
                    }
                }
            }

            // If JSON parsing failed or no devices found, try to detect via Metal API
            if devices.is_empty() {
                // Check if Metal is available
                #[cfg(feature = "metal")]
                {
                    use metal::Device;
                    if let Some(device) = Device::system_default() {
                        let name = device.name().to_string();
                        let mut gpu_info = GpuInfo {
                            backend: GpuBackend::Metal,
                            device_name: name.clone(),
                            memory_bytes: None,
                            compute_capability: None,
                            supports_tensors: true,
                        };

                        // GPU family detection would go here
                        // Note: MTLGPUFamily is not exposed in the current metal crate
                        gpu_info.compute_capability = Some("Metal GPU".to_string());

                        devices.push(gpu_info);
                    }
                }

                // Fallback if Metal crate not available but we're on macOS
                #[cfg(not(feature = "metal"))]
                {
                    devices.push(GpuInfo {
                        backend: GpuBackend::Metal,
                        device_name: "Metal GPU".to_string(),
                        memory_bytes: None,
                        compute_capability: None,
                        supports_tensors: true,
                    });
                }
            }
        }
        _ => {
            // system_profiler failed, try Metal API directly
            #[cfg(feature = "metal")]
            {
                use metal::Device;
                if let Some(device) = Device::system_default() {
                    devices.push(GpuInfo {
                        backend: GpuBackend::Metal,
                        device_name: device.name().to_string(),
                        memory_bytes: None,
                        compute_capability: None,
                        supports_tensors: true,
                    });
                } else {
                    return Err(GpuError::BackendNotAvailable("Metal".to_string()));
                }
            }

            #[cfg(not(feature = "metal"))]
            {
                return Err(GpuError::BackendNotAvailable("Metal".to_string()));
            }
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("Metal".to_string()))
    } else {
        Ok(devices)
    }
}

/// Detect Metal devices (non-macOS - not available)
#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn detect_metal_devices() -> Result<Vec<GpuInfo>, GpuError> {
    Err(GpuError::BackendNotAvailable(
        "Metal (not macOS)".to_string(),
    ))
}

/// Detect OpenCL devices
fn detect_opencl_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to detect OpenCL devices using clinfo
    match Command::new("clinfo").arg("--list").output() {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            for line in output_str.lines() {
                if line.trim().starts_with("Platform") || line.trim().starts_with("Device") {
                    // In a real implementation, we would parse clinfo output properly
                    // For now, just add a generic OpenCL device
                    devices.push(GpuInfo {
                        backend: GpuBackend::OpenCL,
                        device_name: "OpenCL Device".to_string(),
                        memory_bytes: None,
                        compute_capability: None,
                        supports_tensors: false,
                    });
                    break; // Just add one for demo
                }
            }
        }
        _ => {
            return Err(GpuError::BackendNotAvailable("OpenCL".to_string()));
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("OpenCL".to_string()))
    } else {
        Ok(devices)
    }
}

/// Check if a specific backend is properly installed and functional
pub fn check_backend_installation(backend: GpuBackend) -> Result<bool, GpuError> {
    match backend {
        GpuBackend::Cuda => {
            // Check for CUDA installation
            match Command::new("nvcc").arg("--version").output() {
                Ok(output) if output.status.success() => Ok(true),
                _ => Ok(false),
            }
        }
        GpuBackend::Rocm => {
            // Check for ROCm installation
            match Command::new("hipcc").arg("--version").output() {
                Ok(output) if output.status.success() => Ok(true),
                _ => {
                    // Also try rocm-smi as an alternative check
                    match Command::new("rocm-smi").arg("--version").output() {
                        Ok(output) if output.status.success() => Ok(true),
                        _ => Ok(false),
                    }
                }
            }
        }
        GpuBackend::Metal => {
            #[cfg(target_os = "macos")]
            {
                // Metal is always available on macOS
                Ok(true)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Ok(false)
            }
        }
        GpuBackend::OpenCL => {
            // Check for OpenCL installation
            match Command::new("clinfo").output() {
                Ok(output) if output.status.success() => Ok(true),
                _ => Ok(false),
            }
        }
        GpuBackend::Wgpu => {
            // WebGPU is always available through wgpu crate
            Ok(true)
        }
        GpuBackend::Cpu => Ok(true),
    }
}

/// Get detailed information about a specific GPU device
pub fn get_device_info(backend: GpuBackend, device_id: usize) -> Result<GpuInfo, GpuError> {
    let detection_result = detect_gpu_backends();

    detection_result
        .devices
        .into_iter()
        .filter(|d| d.backend == backend)
        .nth(device_id)
        .ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "Device {} not found for backend {}",
                device_id, backend
            ))
        })
}

/// Initialize the optimal GPU backend for the current system
pub fn initialize_optimal_backend() -> Result<GpuBackend, GpuError> {
    let detection_result = detect_gpu_backends();

    // Try backends in order of preference for scientific computing
    let preference_order = [
        GpuBackend::Cuda,   // Best for scientific computing
        GpuBackend::Rocm,   // Second best for scientific computing (AMD)
        GpuBackend::Metal,  // Good on Apple hardware
        GpuBackend::OpenCL, // Widely compatible
        GpuBackend::Wgpu,   // Modern cross-platform
        GpuBackend::Cpu,    // Always available fallback
    ];

    for backend in preference_order.iter() {
        if detection_result
            .devices
            .iter()
            .any(|d: &GpuInfo| d.backend == *backend)
        {
            return Ok(*backend);
        }
    }

    // Should never reach here since CPU is always available
    Ok(GpuBackend::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info_creation() {
        let info = GpuInfo {
            backend: GpuBackend::Cuda,
            device_name: "NVIDIA GeForce RTX 3080".to_string(),
            memory_bytes: Some(10 * 1024 * 1024 * 1024), // 10GB
            compute_capability: Some("8.6".to_string()),
            supports_tensors: true,
        };

        assert_eq!(info.backend, GpuBackend::Cuda);
        assert_eq!(info.device_name, "NVIDIA GeForce RTX 3080");
        assert_eq!(info.memory_bytes, Some(10 * 1024 * 1024 * 1024));
        assert_eq!(info.compute_capability, Some("8.6".to_string()));
        assert!(info.supports_tensors);
    }

    #[test]
    fn test_gpu_detection_result_with_cpu_fallback() {
        let result = detect_gpu_backends();

        // Should always have at least CPU fallback
        assert!(!result.devices.is_empty());
        assert!(result
            .devices
            .iter()
            .any(|d: &GpuInfo| d.backend == GpuBackend::Cpu));

        // Should have a recommended backend
        match result.recommended_backend {
            GpuBackend::Cuda
            | GpuBackend::Rocm
            | GpuBackend::Metal
            | GpuBackend::OpenCL
            | GpuBackend::Cpu => {}
            _ => panic!("Unexpected recommended backend"),
        }
    }

    #[test]
    fn test_check_backend_installation_cpu() {
        // CPU should always be available
        let result = check_backend_installation(GpuBackend::Cpu).unwrap();
        assert!(result);
    }

    #[test]
    fn test_check_backend_installation_wgpu() {
        // WebGPU should always be available through wgpu crate
        let result = check_backend_installation(GpuBackend::Wgpu).unwrap();
        assert!(result);
    }

    #[test]
    fn test_check_backend_installation_metal() {
        let result = check_backend_installation(GpuBackend::Metal).unwrap();
        #[cfg(target_os = "macos")]
        assert!(result);
        #[cfg(not(target_os = "macos"))]
        assert!(!result);
    }

    #[test]
    fn test_initialize_optimal_backend() {
        let backend = initialize_optimal_backend().unwrap();

        // Should return a valid backend
        match backend {
            GpuBackend::Cuda
            | GpuBackend::Rocm
            | GpuBackend::Wgpu
            | GpuBackend::Metal
            | GpuBackend::OpenCL
            | GpuBackend::Cpu => {}
        }
    }

    #[test]
    fn test_get_device_info_invalid_device() {
        // Try to get info for a non-existent device
        let result = get_device_info(GpuBackend::Cpu, 100);

        assert!(result.is_err());
        match result {
            Err(GpuError::InvalidParameter(_)) => {}
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_get_device_info_cpu() {
        // CPU device should always be available
        let result = get_device_info(GpuBackend::Cpu, 0);

        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.backend, GpuBackend::Cpu);
        assert_eq!(info.device_name, "CPU");
        assert!(!info.supports_tensors);
    }

    #[test]
    fn test_detect_metal_devices_non_macos() {
        #[cfg(not(target_os = "macos"))]
        {
            let result = detect_metal_devices();
            assert!(result.is_err());
            match result {
                Err(GpuError::BackendNotAvailable(_)) => {}
                _ => panic!("Expected BackendNotAvailable error"),
            }
        }
    }

    #[test]
    fn test_gpu_info_clone() {
        let info = GpuInfo {
            backend: GpuBackend::Rocm,
            device_name: "AMD Radeon RX 6900 XT".to_string(),
            memory_bytes: Some(16 * 1024 * 1024 * 1024), // 16GB
            compute_capability: Some("RDNA2".to_string()),
            supports_tensors: true,
        };

        let cloned = info.clone();
        assert_eq!(info.backend, cloned.backend);
        assert_eq!(info.device_name, cloned.device_name);
        assert_eq!(info.memory_bytes, cloned.memory_bytes);
        assert_eq!(info.compute_capability, cloned.compute_capability);
        assert_eq!(info.supports_tensors, cloned.supports_tensors);
    }

    #[test]
    fn test_gpu_detection_result_clone() {
        let devices = vec![
            GpuInfo {
                backend: GpuBackend::Cuda,
                device_name: "NVIDIA A100".to_string(),
                memory_bytes: Some(40 * 1024 * 1024 * 1024),
                compute_capability: Some("8.0".to_string()),
                supports_tensors: true,
            },
            GpuInfo {
                backend: GpuBackend::Cpu,
                device_name: "CPU".to_string(),
                memory_bytes: None,
                compute_capability: None,
                supports_tensors: false,
            },
        ];

        let result = GpuDetectionResult {
            devices: devices.clone(),
            recommended_backend: GpuBackend::Cuda,
        };

        let cloned = result.clone();
        assert_eq!(result.devices.len(), cloned.devices.len());
        assert_eq!(result.recommended_backend, cloned.recommended_backend);
    }

    // Mock tests to verify error handling in detection functions
    #[test]
    fn test_detect_cuda_devices_error_handling() {
        // In the real implementation, detect_cuda_devices returns an error
        // when nvidia-smi is not available. We can't easily test this without
        // mocking the Command execution, but we can at least call the function
        let _ = detect_cuda_devices();
    }

    #[test]
    fn test_detect_rocm_devices_error_handling() {
        // Similar to CUDA test
        let _ = detect_rocm_devices();
    }

    #[test]
    fn test_detect_opencl_devices_error_handling() {
        // Similar to CUDA test
        let _ = detect_opencl_devices();
    }

    #[test]
    fn test_backend_preference_order() {
        // Test that initialize_optimal_backend respects the preference order
        let result = detect_gpu_backends();

        // If we have multiple backends, the recommended should follow preference
        if result
            .devices
            .iter()
            .any(|d: &GpuInfo| d.backend == GpuBackend::Cuda)
        {
            // If CUDA is available, it should be preferred
            let optimal = initialize_optimal_backend().unwrap();
            if result
                .devices
                .iter()
                .filter(|d| d.backend == GpuBackend::Cuda)
                .count()
                > 0
            {
                assert_eq!(optimal, GpuBackend::Cuda);
            }
        }
    }
}
