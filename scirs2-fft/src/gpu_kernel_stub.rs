//! GPU kernel stub module for FFT operations
//!
//! This module provides a compatibility layer for GPU-accelerated FFT operations.
//! All GPU operations are redirected to the scirs2-core GPU module to comply with
//! the strict acceleration policy.
//!
//! IMPORTANT: Direct CUDA/OpenCL/Metal API calls are FORBIDDEN in individual modules.
//! All GPU operations MUST go through scirs2-core::gpu.

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;

/// Placeholder for GPU kernel implementations
///
/// All actual GPU operations should be registered with and executed through
/// the scirs2-core GPU kernel registry.
pub struct GpuFftKernel {
    name: String,
}

impl GpuFftKernel {
    /// Create a new GPU FFT kernel stub
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Execute the kernel (stub implementation)
    ///
    /// In the real implementation, this would dispatch to scirs2-core::gpu
    pub fn execute(&self, _input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
        Err(FFTError::NotImplementedError(format!(
            "GPU kernel '{}' not yet migrated to scirs2-core GPU system. \
             Direct GPU implementations are forbidden - all GPU operations must use scirs2-core::gpu",
            self.name
        )))
    }
}

/// Check if GPU is available through core
#[allow(dead_code)]
pub fn is_gpu_available() -> bool {
    // This should use scirs2-core::gpu capabilities
    false
}

/// Message for functions that need migration
pub const MIGRATION_MESSAGE: &str =
    "This GPU functionality needs to be migrated to use scirs2-core::gpu module. \
     Direct CUDA/OpenCL/Metal implementations are forbidden by the strict acceleration policy.";
