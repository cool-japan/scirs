//! GPU backend stub for FEM stiffness assembly via wgpu.
//!
//! This module is only compiled when the `gpu_fem` feature is enabled.
//! It provides a placeholder that always returns `GpuFemError::GpuNotAvailable`,
//! signalling the dispatcher to fall back to the CPU path.
//!
//! # Future Work
//!
//! A full WGSL compute-shader implementation would:
//! 1. Upload element data (node coordinates, D-matrix) to GPU buffers.
//! 2. Dispatch a compute shader where each workgroup handles one element.
//! 3. Use `atomicAdd` on a shared sparse-matrix buffer to scatter element
//!    stiffness contributions (shared-memory atomic scatter pattern).
//! 4. Readback and convert to the COO `StiffnessMatrix` format.

use crate::error::IntegrateError;
use crate::gpu_fem::dispatch::GpuFemError;
use crate::gpu_fem::stiffness::{Element2D, StiffnessMatrix};
use scirs2_core::ndarray::Array2;

/// Attempt GPU-accelerated stiffness assembly.
///
/// Currently always returns `GpuFemError::GpuNotAvailable`, causing the
/// dispatcher to fall back to the CPU Rayon path.
pub fn assemble_stiffness_gpu(
    _elements: &[Element2D],
    _d_matrix: &Array2<f64>,
    _n_nodes: usize,
) -> Result<StiffnessMatrix, GpuFemError> {
    Err(GpuFemError::GpuNotAvailable)
}
