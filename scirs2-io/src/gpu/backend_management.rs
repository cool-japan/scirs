//! GPU backend management for I/O operations
//!
//! This module provides comprehensive GPU backend detection, validation,
//! and capability management for optimal I/O performance across different
//! GPU vendors and architectures.

use crate::error::{IoError, Result};
use scirs2_core::gpu::{GpuBackend, GpuDevice, GpuDeviceInfo, GpuError};
use scirs2_core::simd_ops::PlatformCapabilities;

/// Minimum total device memory (in bytes) a CUDA device must report to be
/// considered usable for I/O acceleration. A reported value of `0` means the
/// adapter memory size is unknown and this check is skipped (see
/// [`GpuDeviceInfo::total_memory`]).
const CUDA_MIN_TOTAL_MEMORY_BYTES: u64 = 512 * 1024 * 1024;

/// Minimum total device memory (in bytes) required for the Metal backend. A
/// reported value of `0` (unknown) skips the check.
const METAL_MIN_TOTAL_MEMORY_BYTES: u64 = 256 * 1024 * 1024;

/// Minimum total device memory (in bytes) required for the OpenCL backend. A
/// reported value of `0` (unknown) skips the check.
const OPENCL_MIN_TOTAL_MEMORY_BYTES: u64 = 128 * 1024 * 1024;

/// Minimum work-group size required by the CUDA backend (a single warp).
const CUDA_MIN_WORK_GROUP_SIZE: u32 = 32;

/// Minimum work-group size required by the Metal backend (a single SIMD-group).
const METAL_MIN_WORK_GROUP_SIZE: u32 = 32;

/// Minimum work-group size required by the OpenCL backend.
const OPENCL_MIN_WORK_GROUP_SIZE: u32 = 32;

/// Bytes per gigabyte, used to convert the queried byte counts into the
/// gigabyte figure exposed by [`BackendCapabilities`].
const BYTES_PER_GIB: f64 = (1024 * 1024 * 1024) as f64;

/// GPU-accelerated I/O processor with backend management
#[derive(Debug)]
pub struct GpuIoProcessor {
    /// GPU device handle
    pub device: GpuDevice,
    /// Platform capabilities
    pub capabilities: PlatformCapabilities,
}

impl GpuIoProcessor {
    /// Create a new GPU I/O processor with the preferred backend
    pub fn new() -> Result<Self> {
        let capabilities = PlatformCapabilities::detect();

        if !capabilities.gpu_available {
            return Err(IoError::Other("GPU acceleration not available. Please ensure GPU drivers are installed and properly configured.".to_string()));
        }

        // Try backends in order of preference with proper detection
        let backend = Self::detect_optimal_backend()
            .map_err(|e| IoError::Other(format!("Failed to detect optimal GPU backend: {}", e)))?;
        let device = GpuDevice::new(backend, 0);

        // Validate the selected device's reported capabilities against the
        // backend's minimum requirements before handing it back to the caller.
        // `detect_optimal_backend` already filters on `validate_backend`, but
        // re-checking here guards against a backend that becomes unusable
        // between detection and device construction and turns a soft "skip" in
        // detection into a hard error for an explicitly created processor.
        if backend != GpuBackend::Cpu && !Self::validate_device_info(&device)? {
            return Err(IoError::ValidationError(format!(
                "GPU backend {backend} did not meet minimum capability requirements",
            )));
        }

        Ok(Self {
            device,
            capabilities,
        })
    }

    /// Create a new GPU I/O processor with a specific backend
    pub fn with_backend(backend: GpuBackend) -> Result<Self> {
        if !Self::is_backend_available(backend) {
            return Err(IoError::Other(format!(
                "GPU backend {} is not available",
                backend
            )));
        }

        let device = GpuDevice::new(backend, 0);
        let capabilities = PlatformCapabilities::detect();

        Ok(Self {
            device,
            capabilities,
        })
    }

    /// Detect the optimal GPU backend for the current system
    pub fn detect_optimal_backend() -> Result<GpuBackend> {
        // Check each backend in order of preference
        let backends_to_try = [
            GpuBackend::Cuda,   // NVIDIA - best performance and feature support
            GpuBackend::Metal,  // Apple Silicon - excellent for Mac
            GpuBackend::OpenCL, // Cross-platform fallback
        ];

        for &backend in &backends_to_try {
            if Self::is_backend_available(backend) {
                // Additional validation - check if we can actually create a device
                match Self::validate_backend(backend) {
                    Ok(true) => return Ok(backend),
                    _ => continue,
                }
            }
        }
        // Fallback to CPU when no GPU backend is available
        Ok(GpuBackend::Cpu)
    }

    /// Validate that a backend is functional (not just available)
    pub fn validate_backend(backend: GpuBackend) -> Result<bool> {
        if !backend.is_available() {
            return Ok(false);
        }

        // Create a test device and validate its reported capabilities against
        // the backend's minimum requirements via a real `get_info()` query.
        let device = GpuDevice::new(backend, 0);
        Self::validate_device_info(&device)
    }

    /// Query a device's capabilities and dispatch to the backend-specific
    /// validator.
    ///
    /// This is the single point where [`GpuDevice::get_info`] is consulted, so
    /// every validation path observes a consistent [`GpuDeviceInfo`] snapshot.
    /// Backends other than CUDA / Metal / OpenCL (including the CPU fallback)
    /// are reported as not GPU-usable.
    ///
    /// # Errors
    ///
    /// Propagates any failure surfaced by [`GpuDevice::get_info`].
    fn validate_device_info(device: &GpuDevice) -> Result<bool> {
        let info = device
            .get_info()
            .map_err(|e| IoError::Other(format!("Failed to query GPU device info: {e}")))?;

        match device.backend() {
            GpuBackend::Cuda => Self::validate_cuda_backend(&info),
            GpuBackend::Metal => Self::validate_metal_backend(&info),
            GpuBackend::OpenCL => Self::validate_opencl_backend(&info),
            _ => Ok(false),
        }
    }

    /// Validate CUDA device capabilities against the backend minimums.
    ///
    /// CUDA is the preferred backend for double-precision scientific I/O, so a
    /// device that reports no `f64` support is rejected. Memory and work-group
    /// limits are checked only when the device reports a known (non-zero) value;
    /// a reported `0` is treated as "unknown" and that particular check is
    /// skipped rather than failing (see [`GpuDeviceInfo`]).
    fn validate_cuda_backend(info: &GpuDeviceInfo) -> Result<bool> {
        if info.backend != GpuBackend::Cuda {
            return Ok(false);
        }
        // CUDA scientific kernels rely on double precision; a device that
        // cannot provide it is not usable for this backend.
        if !info.supports_fp64 {
            return Ok(false);
        }
        Ok(
            Self::meets_memory_requirement(info, CUDA_MIN_TOTAL_MEMORY_BYTES)
                && Self::meets_work_group_requirement(info, CUDA_MIN_WORK_GROUP_SIZE),
        )
    }

    /// Validate Metal device capabilities against the backend minimums.
    ///
    /// Metal does not expose double precision in shaders, so `f64` is not
    /// required; instead half-precision support is required because the Metal
    /// I/O kernels rely on it. Memory and work-group checks are skipped when the
    /// reported value is unknown (`0`).
    fn validate_metal_backend(info: &GpuDeviceInfo) -> Result<bool> {
        if info.backend != GpuBackend::Metal {
            return Ok(false);
        }
        // Metal cannot offer f64 in shaders, so the half-precision path is the
        // supported precision and must be present.
        if !info.supports_fp16 {
            return Ok(false);
        }
        Ok(
            Self::meets_memory_requirement(info, METAL_MIN_TOTAL_MEMORY_BYTES)
                && Self::meets_work_group_requirement(info, METAL_MIN_WORK_GROUP_SIZE),
        )
    }

    /// Validate OpenCL device capabilities against the backend minimums.
    ///
    /// OpenCL `f64`/`f16` are optional extensions, so neither precision is
    /// required; the device only has to provide enough memory (when known) and
    /// a large enough work group. Unknown (`0`) values skip their checks.
    fn validate_opencl_backend(info: &GpuDeviceInfo) -> Result<bool> {
        if info.backend != GpuBackend::OpenCL {
            return Ok(false);
        }
        Ok(
            Self::meets_memory_requirement(info, OPENCL_MIN_TOTAL_MEMORY_BYTES)
                && Self::meets_work_group_requirement(info, OPENCL_MIN_WORK_GROUP_SIZE),
        )
    }

    /// Returns `true` when the device's reported total memory meets `minimum`,
    /// treating a reported `0` as "unknown" and therefore passing the check.
    fn meets_memory_requirement(info: &GpuDeviceInfo, minimum: u64) -> bool {
        info.total_memory == 0 || info.total_memory >= minimum
    }

    /// Returns `true` when the device's reported maximum work-group size meets
    /// `minimum`, treating a reported `0` as "unknown" and passing the check.
    fn meets_work_group_requirement(info: &GpuDeviceInfo, minimum: u32) -> bool {
        info.max_work_group_size == 0 || info.max_work_group_size >= minimum
    }

    /// Get detailed backend capabilities.
    ///
    /// Queries the underlying device with [`GpuDevice::get_info`] and translates
    /// the returned [`GpuDeviceInfo`] into a [`BackendCapabilities`]. Fields the
    /// device reports as "unknown" (memory `0`, work-group size `0`) fall back
    /// to conservative, backend-appropriate defaults so the returned struct is
    /// always well-formed and non-zero where downstream code expects positive
    /// values.
    ///
    /// # Errors
    ///
    /// Propagates any failure surfaced by [`GpuDevice::get_info`].
    pub fn get_backend_capabilities(&self) -> Result<BackendCapabilities> {
        let info = self
            .device
            .get_info()
            .map_err(|e| IoError::Other(format!("Failed to query GPU device info: {e}")))?;
        Ok(Self::capabilities_from_info(&info))
    }

    /// Translate a queried [`GpuDeviceInfo`] into [`BackendCapabilities`],
    /// substituting conservative defaults for any "unknown" (`0`) field.
    fn capabilities_from_info(info: &GpuDeviceInfo) -> BackendCapabilities {
        // Conservative per-backend fallbacks used only when the device reports
        // a field as unknown (`0`).
        let default_memory_gb = match info.backend {
            GpuBackend::Cuda | GpuBackend::Rocm => 4.0,
            GpuBackend::Metal => 8.0,
            GpuBackend::OpenCL | GpuBackend::Wgpu => 2.0,
            GpuBackend::Cpu => 1.0,
        };
        let default_work_group: usize = match info.backend {
            GpuBackend::Cuda | GpuBackend::Rocm | GpuBackend::Metal => 1024,
            GpuBackend::OpenCL | GpuBackend::Wgpu => 256,
            GpuBackend::Cpu => 1,
        };

        let memory_gb = if info.total_memory == 0 {
            default_memory_gb
        } else {
            info.total_memory as f64 / BYTES_PER_GIB
        };

        let max_work_group_size = if info.max_work_group_size == 0 {
            default_work_group
        } else {
            info.max_work_group_size as usize
        };

        // Derive a plausible maximum single-allocation size from the available
        // (preferred) or total memory, capping at the whole device when known.
        let known_memory_bytes = if info.available_memory != 0 {
            info.available_memory
        } else {
            info.total_memory
        };
        let max_allocation_size = if known_memory_bytes == 0 {
            1usize << 30 // 1 GiB when memory is unknown
        } else {
            usize::try_from(known_memory_bytes).unwrap_or(usize::MAX)
        };

        BackendCapabilities {
            backend: info.backend,
            memory_gb,
            max_work_group_size,
            supports_fp64: info.supports_fp64,
            supports_fp16: info.supports_fp16,
            // Compute-unit count is not exposed by GpuDeviceInfo; estimate from
            // the work-group size as a stable, positive proxy.
            compute_units: (max_work_group_size / 32).max(1),
            max_allocation_size,
            local_memory_size: 48 * 1024, // 48 KiB: typical shared-memory budget
        }
    }

    /// Get the current GPU backend
    pub fn backend(&self) -> GpuBackend {
        self.device.backend()
    }

    /// Check if a specific backend is available
    pub fn is_backend_available(backend: GpuBackend) -> bool {
        backend.is_available()
    }

    /// List all available backends on the system
    pub fn list_available_backends() -> Vec<GpuBackend> {
        let mut list: Vec<GpuBackend> = [GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::OpenCL]
            .iter()
            .filter(|&&backend| Self::is_backend_available(backend))
            .copied()
            .collect();
        // Ensure CPU is always present as a safe fallback
        if !list.contains(&GpuBackend::Cpu) {
            list.push(GpuBackend::Cpu);
        }
        list
    }

    /// Get optimal backend for specific workload type
    pub fn get_optimal_backend_for_workload(workload: GpuWorkloadType) -> Result<GpuBackend> {
        let available_backends = Self::list_available_backends();

        if available_backends.is_empty() {
            return Ok(GpuBackend::Cpu);
        }

        // Choose backend based on workload characteristics
        match workload {
            GpuWorkloadType::MachineLearning => {
                // CUDA is preferred for ML workloads
                if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
            GpuWorkloadType::ImageProcessing => {
                // Metal is excellent for image processing on Apple devices
                if available_backends.contains(&GpuBackend::Metal) {
                    Ok(GpuBackend::Metal)
                } else if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
            GpuWorkloadType::GeneralCompute => {
                // Use first available backend for general compute
                Ok(available_backends[0])
            }
            GpuWorkloadType::Compression => {
                // CUDA typically performs well for compression
                if available_backends.contains(&GpuBackend::Cuda) {
                    Ok(GpuBackend::Cuda)
                } else {
                    Ok(available_backends[0])
                }
            }
        }
    }
}

impl Default for GpuIoProcessor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to CPU backend if GPU creation fails
            GpuIoProcessor {
                device: GpuDevice::new(GpuBackend::Cpu, 0),
                capabilities: PlatformCapabilities::detect(),
            }
        })
    }
}

/// Detailed backend capabilities for optimization decisions
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// GPU backend type
    pub backend: GpuBackend,
    /// Total memory in gigabytes
    pub memory_gb: f64,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Whether FP64 operations are supported
    pub supports_fp64: bool,
    /// Whether FP16 operations are supported
    pub supports_fp16: bool,
    /// Number of compute units
    pub compute_units: usize,
    /// Maximum allocation size in bytes
    pub max_allocation_size: usize,
    /// Local memory size in bytes
    pub local_memory_size: usize,
}

impl BackendCapabilities {
    /// Check if backend supports high-precision computations
    pub fn supports_high_precision(&self) -> bool {
        self.supports_fp64
    }

    /// Check if backend supports half-precision optimizations
    pub fn supports_half_precision(&self) -> bool {
        self.supports_fp16
    }

    /// Get optimal work group size for given problem size
    pub fn get_optimal_work_group_size(&self, problem_size: usize) -> usize {
        let base_size = match self.backend {
            GpuBackend::Cuda => 256,   // CUDA warps are 32, good block size is 256
            GpuBackend::Metal => 64,   // Metal threadgroups typically 64
            GpuBackend::OpenCL => 128, // OpenCL flexible, 128 is safe
            _ => 64,
        };

        // Ensure we don't exceed device limits
        base_size.min(self.max_work_group_size).min(problem_size)
    }

    /// Estimate memory bandwidth in GB/s
    pub fn estimate_memory_bandwidth(&self) -> f64 {
        match self.backend {
            GpuBackend::Cuda => self.memory_gb * 0.8, // CUDA typically achieves ~80% peak
            GpuBackend::Metal => self.memory_gb * 0.7, // Metal varies more
            GpuBackend::OpenCL => self.memory_gb * 0.6, // OpenCL is more conservative
            _ => self.memory_gb * 0.4,
        }
    }
}

/// GPU workload types for optimal backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWorkloadType {
    /// Machine learning workloads
    MachineLearning,
    /// Image processing workloads
    ImageProcessing,
    /// General compute workloads
    GeneralCompute,
    /// Data compression workloads
    Compression,
}

/// Backend performance characteristics for optimization
#[derive(Debug, Clone)]
pub struct BackendPerformanceProfile {
    /// GPU backend type
    pub backend: GpuBackend,
    /// Throughput in GB/s
    pub throughput_gbps: f64,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Power efficiency score (0.0-1.0)
    pub power_efficiency: f64,
    /// Memory efficiency score (0.0-1.0)
    pub memory_efficiency: f64,
}

impl BackendPerformanceProfile {
    /// Create performance profile for a backend
    pub fn new(backend: GpuBackend, capabilities: &BackendCapabilities) -> Self {
        let (throughput, latency, power_eff, mem_eff) = match backend {
            GpuBackend::Cuda => (capabilities.memory_gb * 0.8, 0.1, 0.7, 0.9),
            GpuBackend::Metal => (capabilities.memory_gb * 0.7, 0.15, 0.9, 0.8),
            GpuBackend::OpenCL => (capabilities.memory_gb * 0.6, 0.2, 0.6, 0.7),
            _ => (capabilities.memory_gb * 0.4, 0.5, 0.8, 0.5),
        };

        Self {
            backend,
            throughput_gbps: throughput,
            latency_ms: latency,
            power_efficiency: power_eff,
            memory_efficiency: mem_eff,
        }
    }

    /// Calculate overall performance score
    pub fn performance_score(&self) -> f64 {
        self.throughput_gbps * 0.4
            + (1.0 / self.latency_ms) * 0.3
            + self.power_efficiency * 0.2
            + self.memory_efficiency * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_availability_detection() {
        let backends = GpuIoProcessor::list_available_backends();
        // At least CPU backend should always be available
        assert!(!backends.is_empty());
    }

    #[test]
    fn test_backend_capabilities() {
        if let Ok(processor) = GpuIoProcessor::new() {
            let capabilities = processor
                .get_backend_capabilities()
                .expect("Operation failed");
            assert!(capabilities.memory_gb > 0.0);
            assert!(capabilities.compute_units > 0);
        }
    }

    #[test]
    fn test_optimal_backend_for_workload() {
        let backend =
            GpuIoProcessor::get_optimal_backend_for_workload(GpuWorkloadType::MachineLearning);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_work_group_size_calculation() {
        let capabilities = BackendCapabilities {
            backend: GpuBackend::Cuda,
            memory_gb: 8.0,
            max_work_group_size: 1024,
            supports_fp64: true,
            supports_fp16: true,
            compute_units: 32,
            max_allocation_size: 1024 * 1024 * 1024,
            local_memory_size: 48 * 1024,
        };

        let work_group_size = capabilities.get_optimal_work_group_size(10000);
        assert_eq!(work_group_size, 256); // CUDA optimal

        let small_size = capabilities.get_optimal_work_group_size(100);
        assert_eq!(small_size, 100); // Limited by problem size
    }

    #[test]
    fn test_performance_profile_scoring() {
        let capabilities = BackendCapabilities {
            backend: GpuBackend::Cuda,
            memory_gb: 8.0,
            max_work_group_size: 1024,
            supports_fp64: true,
            supports_fp16: true,
            compute_units: 32,
            max_allocation_size: 1024 * 1024 * 1024,
            local_memory_size: 48 * 1024,
        };

        let profile = BackendPerformanceProfile::new(GpuBackend::Cuda, &capabilities);
        let score = profile.performance_score();
        assert!(score > 0.0);
    }

    /// The real validation path must route through `GpuDevice::get_info()` and
    /// return a sane boolean for each GPU backend without panicking.
    #[test]
    fn test_validate_device_info_per_backend() {
        for backend in [GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::OpenCL] {
            let device = GpuDevice::new(backend, 0);
            // Must not error: get_info() is infallible in the Pure-Rust build.
            let valid = GpuIoProcessor::validate_device_info(&device)
                .expect("validate_device_info should not fail in the default build");
            // With the deterministic placeholder info (memory/work-group "0" =>
            // skipped, and the precisions matched to each backend's contract),
            // every GPU backend validates as usable.
            assert!(
                valid,
                "expected backend {backend} to validate against its placeholder GpuDeviceInfo",
            );
        }
        // The CPU backend is never a usable GPU backend.
        let cpu = GpuDevice::new(GpuBackend::Cpu, 0);
        assert!(!GpuIoProcessor::validate_device_info(&cpu)
            .expect("validate_device_info should not fail for CPU"));
    }

    /// `validate_backend` is the public entry used during detection and must
    /// agree with the device-info validation for the GPU backends.
    #[test]
    fn test_validate_backend_returns_ok() {
        for backend in [GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::OpenCL] {
            let result = GpuIoProcessor::validate_backend(backend);
            assert!(
                result.is_ok(),
                "validate_backend({backend}) should return Ok, got {result:?}",
            );
        }
    }

    /// `get_backend_capabilities()` must build a well-formed, non-degenerate
    /// struct from the device's queried info for at least one backend.
    #[test]
    fn test_get_backend_capabilities_is_well_formed() {
        let processor = GpuIoProcessor::with_backend(GpuBackend::Cpu)
            .expect("CPU backend is always constructible");
        let caps = processor
            .get_backend_capabilities()
            .expect("get_backend_capabilities should query device info and succeed");

        // Every field must be populated with a sane, positive value even though
        // the placeholder GpuDeviceInfo reports memory/work-group size as "0".
        assert_eq!(caps.backend, GpuBackend::Cpu);
        assert!(
            caps.memory_gb > 0.0,
            "memory_gb fell back to a positive default"
        );
        assert!(caps.max_work_group_size >= 1);
        assert!(caps.compute_units >= 1);
        assert!(caps.max_allocation_size >= 1 << 20);
        assert!(caps.local_memory_size > 0);
    }

    /// Memory / work-group requirement helpers must treat a reported `0`
    /// (unknown) as "skip the check" rather than as a failure.
    #[test]
    fn test_unknown_capability_skips_check() {
        let mut info = GpuDeviceInfo::for_backend(GpuBackend::Cuda);
        // Placeholder reports 0 (unknown) for both -> checks are skipped.
        assert!(GpuIoProcessor::meets_memory_requirement(
            &info,
            CUDA_MIN_TOTAL_MEMORY_BYTES
        ));
        assert!(GpuIoProcessor::meets_work_group_requirement(
            &info,
            CUDA_MIN_WORK_GROUP_SIZE
        ));

        // A known-but-too-small memory value must now fail the check.
        info.total_memory = CUDA_MIN_TOTAL_MEMORY_BYTES - 1;
        assert!(!GpuIoProcessor::meets_memory_requirement(
            &info,
            CUDA_MIN_TOTAL_MEMORY_BYTES
        ));

        // A known, sufficient memory value passes.
        info.total_memory = CUDA_MIN_TOTAL_MEMORY_BYTES;
        assert!(GpuIoProcessor::meets_memory_requirement(
            &info,
            CUDA_MIN_TOTAL_MEMORY_BYTES
        ));
    }

    /// CUDA validation must reject a device that reports no double-precision
    /// support, since the CUDA I/O kernels depend on it.
    #[test]
    fn test_cuda_rejects_without_fp64() {
        let mut info = GpuDeviceInfo::for_backend(GpuBackend::Cuda);
        info.supports_fp64 = false;
        assert!(!GpuIoProcessor::validate_cuda_backend(&info)
            .expect("validate_cuda_backend should not fail"));
    }

    /// Capabilities built from a device that reports *known* memory must reflect
    /// the queried byte count rather than the conservative fallback.
    #[test]
    fn test_capabilities_uses_known_memory() {
        let mut info = GpuDeviceInfo::for_backend(GpuBackend::OpenCL);
        info.total_memory = 8 * 1024 * 1024 * 1024; // 8 GiB
        info.max_work_group_size = 512;
        let caps = GpuIoProcessor::capabilities_from_info(&info);
        assert!((caps.memory_gb - 8.0).abs() < 1e-9);
        assert_eq!(caps.max_work_group_size, 512);
    }
}
