//! wgpu linear-scan backend for GPU k-NN (stub, `gpu_kdtree` feature).
//!
//! This module provides the entry point for a real wgpu / WGSL parallel
//! linear-scan implementation.  In the current release the stub unconditionally
//! returns an error so the auto-dispatch layer falls back to the CPU k-d tree
//! path.  This matches the established pattern used in other GPU stubs across
//! the SciRS2 codebase.
//!
//! ## Future implementation
//!
//! A real implementation would:
//! 1. Pack all points into a `wgpu::Buffer` as `f32` (or `f64` via
//!    `shader_f64` extension where available).
//! 2. Dispatch a WGSL compute shader that, for each query thread, computes
//!    distances to all stored points and maintains a small local heap of size k.
//! 3. Download the result buffers back to the host and convert them to
//!    `KdQueryResult` values.
//!
//! Because wgpu initialization is async and allocation can fail, the stub
//! returns `Err` immediately to exercise the fallback path in tests.

use super::tree::{GpuKdTree, KdQueryResult};
use crate::error::InterpolateError;

/// Attempt a GPU linear-scan k-NN via wgpu.
///
/// # Current behavior
///
/// Returns `Err` unconditionally — the GPU backend is not yet implemented.
/// The caller (`knn_auto_dispatch`) silently falls back to the CPU path.
pub fn knn_wgpu(
    _tree: &GpuKdTree,
    _queries: &[Vec<f64>],
    _k: usize,
) -> Result<Vec<KdQueryResult>, InterpolateError> {
    Err(InterpolateError::NotImplemented(
        "GPU wgpu linear-scan backend is not yet implemented; \
         using CPU k-d tree path"
            .to_string(),
    ))
}
