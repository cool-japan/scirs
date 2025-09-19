//! GPU backend implementations for different hardware platforms
//!
//! This module contains implementations for various GPU backends including
//! CUDA, OpenCL, ROCm, Metal, and others. Each backend provides a consistent
//! interface for GPU-accelerated linear algebra operations.
//!
//! ## Available Backends
//!
//! - **CUDA**: NVIDIA GPU acceleration with cuBLAS integration
//! - **OpenCL**: Cross-platform GPU acceleration with clBLAS integration
//! - **Metal**: Apple GPU acceleration for macOS/iOS devices
//! - **ROCm**: AMD GPU acceleration with HIP integration
//! - **CPU Fallback**: Software implementation using CPU operations
//!
//! ## Backend Selection
//!
//! Backends are conditionally compiled based on feature flags:
//! ```toml
//! [features]
//! cuda = ["dep:cuda-sys"]
//! opencl = ["dep:opencl-sys"]
//! metal = ["dep:metal"]
//! rocm = ["dep:hip-sys"]
//! ```

// Declare all backend modules
pub mod common;
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "rocm")]
pub mod rocm;

// Re-export common types and utilities
pub use common::*;

// Re-export all backend implementations for backward compatibility
pub use cpu::CpuFallbackBackend;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaContext, CudaPerformanceStats};

#[cfg(feature = "opencl")]
pub use opencl::{OpenClBackend, OpenClContext, OpenClPerformanceStats};

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

#[cfg(feature = "rocm")]
pub use rocm::RocmBackend;

// Note: When features are disabled, these backends are not available.
// The select_best_backend() function will fall back to CPU implementation.

/// Automatically select the best available GPU backend
pub fn select_best_backend() -> Box<dyn GpuBackend> {
    // Try backends in order of preference
    #[cfg(feature = "cuda")]
    if let Ok(backend) = CudaBackend::new() {
        if backend.is_available() {
            return Box::new(backend);
        }
    }

    #[cfg(feature = "opencl")]
    if let Ok(backend) = OpenClBackend::new() {
        if backend.is_available() {
            return Box::new(backend);
        }
    }

    #[cfg(feature = "metal")]
    if let Ok(backend) = MetalBackend::new() {
        if backend.is_available() {
            return Box::new(backend);
        }
    }

    #[cfg(feature = "rocm")]
    if let Ok(backend) = RocmBackend::new() {
        if backend.is_available() {
            return Box::new(backend);
        }
    }

    // Fallback to CPU
    Box::new(CpuFallbackBackend::new())
}

/// Get all available backends
pub fn available_backends() -> Vec<Box<dyn GpuBackend>> {
    let mut backends = Vec::new();

    // Always include CPU fallback
    backends.push(Box::new(CpuFallbackBackend::new()) as Box<dyn GpuBackend>);

    #[cfg(feature = "cuda")]
    if let Ok(backend) = CudaBackend::new() {
        if backend.is_available() {
            backends.push(Box::new(backend));
        }
    }

    #[cfg(feature = "opencl")]
    if let Ok(backend) = OpenClBackend::new() {
        if backend.is_available() {
            backends.push(Box::new(backend));
        }
    }

    #[cfg(feature = "metal")]
    if let Ok(backend) = MetalBackend::new() {
        if backend.is_available() {
            backends.push(Box::new(backend));
        }
    }

    #[cfg(feature = "rocm")]
    if let Ok(backend) = RocmBackend::new() {
        if backend.is_available() {
            backends.push(Box::new(backend));
        }
    }

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_always_available() {
        let backend = CpuFallbackBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "CPU Fallback");
    }

    #[test]
    fn test_select_best_backend() {
        let backend = select_best_backend();
        assert!(backend.is_available());
        // Should always have at least CPU fallback
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        assert!(!backends.is_empty()); // Should always have at least CPU fallback

        // Check that all returned backends are available
        for backend in backends {
            assert!(backend.is_available());
        }
    }
}
