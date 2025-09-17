//! GPU context management and backend selection
//!
//! This module provides GPU context initialization, backend selection,
//! and error handling for computer vision operations.

use crate::error::{Result, VisionError};
use scirs2_core::gpu::{GpuBackend, GpuContext};

/// GPU-accelerated vision context
pub struct GpuVisionContext {
    pub(crate) context: GpuContext,
    backend: GpuBackend,
}

impl GpuVisionContext {
    /// Create a new GPU vision context with the preferred backend
    pub fn new() -> Result<Self> {
        let preferred_backend = GpuBackend::preferred();

        // Try preferred backend first
        match GpuContext::new(preferred_backend) {
            Ok(context) => {
                eprintln!("Successfully created GPU context with backend: {preferred_backend:?}");
                Ok(Self {
                    context,
                    backend: preferred_backend,
                })
            }
            Err(preferred_error) => {
                eprintln!(
                    "Failed to create GPU context with preferred backend {preferred_backend:?}: {preferred_error}"
                );

                // Try fallback backends in order of preference
                let fallback_backends = [
                    GpuBackend::Cpu,    // Always available as final fallback
                    GpuBackend::Wgpu,   // Cross-platform
                    GpuBackend::OpenCL, // Widely supported
                    GpuBackend::Cuda,   // NVIDIA specific
                    GpuBackend::Metal,  // Apple specific
                ];

                for &fallback_backend in &fallback_backends {
                    if fallback_backend == preferred_backend {
                        continue; // Skip already tried backend
                    }

                    match GpuContext::new(fallback_backend) {
                        Ok(context) => {
                            eprintln!(
                                "Successfully created GPU context with fallback backend: {fallback_backend:?}"
                            );
                            return Ok(Self {
                                context,
                                backend: fallback_backend,
                            });
                        }
                        Err(fallback_error) => {
                            eprintln!(
                                "Fallback backend {fallback_backend:?} also failed: {fallback_error}"
                            );
                        }
                    }
                }

                // If all backends fail, return the original error with helpful context
                Err(VisionError::Other(format!(
                    "Failed to create GPU context with any backend. Preferred backend {preferred_backend:?} failed with: {preferred_error}. All fallback backends also failed. Check GPU drivers and compute capabilities."
                )))
            }
        }
    }

    /// Create a new GPU vision context with a specific backend
    pub fn with_backend(backend: GpuBackend) -> Result<Self> {
        match GpuContext::new(backend) {
            Ok(context) => {
                eprintln!("Successfully created GPU context with requested backend: {backend:?}");
                Ok(Self { context, backend })
            }
            Err(error) => {
                let detailed_error = match backend {
                    GpuBackend::Cuda => {
                        format!(
                            "CUDA backend failed: {error}. Ensure NVIDIA drivers are installed and CUDA-capable GPU is available."
                        )
                    }
                    GpuBackend::Metal => {
                        format!(
                            "Metal backend failed: {error}. Metal is only available on macOS with compatible hardware."
                        )
                    }
                    GpuBackend::OpenCL => {
                        format!(
                            "OpenCL backend failed: {error}. Check OpenCL runtime installation and driver support."
                        )
                    }
                    GpuBackend::Wgpu => {
                        format!(
                            "WebGPU backend failed: {error}. Check GPU drivers and WebGPU support."
                        )
                    }
                    GpuBackend::Cpu => {
                        format!(
                            "CPU backend failed: {error}. This should not happen as CPU backend should always be available."
                        )
                    }
                    GpuBackend::Rocm => {
                        format!(
                            "ROCm backend failed: {error}. Check ROCm installation and AMD GPU drivers."
                        )
                    }
                };

                eprintln!("GPU context creation failed: {detailed_error}");
                Err(VisionError::Other(detailed_error))
            }
        }
    }

    /// Get the backend being used
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get backend name as string
    pub fn backend_name(&self) -> &str {
        self.context.backend_name()
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.backend != GpuBackend::Cpu
    }

    /// Get available GPU memory
    pub fn available_memory(&self) -> Option<usize> {
        self.context.get_available_memory()
    }

    /// Get total GPU memory
    pub fn total_memory(&self) -> Option<usize> {
        self.context.get_total_memory()
    }
}

/// GPU memory usage statistics
pub struct GpuMemoryStats {
    /// Total GPU memory in bytes
    pub total_memory: usize,
    /// Available GPU memory in bytes
    pub available_memory: usize,
    /// Used GPU memory in bytes
    pub used_memory: usize,
    /// GPU memory utilization as percentage (0-100)
    pub utilization_percent: f32,
}

impl GpuVisionContext {
    /// Get current GPU memory statistics
    pub fn memory_stats(&self) -> Option<GpuMemoryStats> {
        let total = self.total_memory()?;
        let available = self.available_memory()?;
        let used = total.saturating_sub(available);
        let utilization = (used as f32 / total as f32) * 100.0;

        Some(GpuMemoryStats {
            total_memory: total,
            available_memory: available,
            used_memory: used,
            utilization_percent: utilization,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let result = GpuVisionContext::new();
        // Should succeed with at least CPU backend
        assert!(result.is_ok());

        let ctx = result.unwrap();
        println!("GPU backend: {}", ctx.backend_name());
    }

    #[test]
    fn test_gpu_memory_info() {
        if let Ok(ctx) = GpuVisionContext::new() {
            if let Some(stats) = ctx.memory_stats() {
                println!("GPU Memory Stats:");
                println!("  Total: {} MB", stats.total_memory / (1024 * 1024));
                println!("  Available: {} MB", stats.available_memory / (1024 * 1024));
                println!("  Used: {} MB", stats.used_memory / (1024 * 1024));
                println!("  Utilization: {:.1}%", stats.utilization_percent);
            }
        }
    }

    #[test]
    fn test_backend_selection() {
        // Test CPU backend explicitly
        let cpu_ctx = GpuVisionContext::with_backend(GpuBackend::Cpu);
        assert!(cpu_ctx.is_ok());

        let ctx = cpu_ctx.unwrap();
        assert_eq!(ctx.backend(), GpuBackend::Cpu);
        assert!(!ctx.is_gpu_available());
    }
}
