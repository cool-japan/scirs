//! GPU-accelerated Conjugate Gradient direction update.
//!
//! This module provides `try_gpu_cg_update` which replaces the CPU beta/direction
//! computation in `minimize_conjugate_gradient_with_jacobian` when:
//! - The `gpu` feature is enabled (`scirs2-core/array_protocol_wgpu`).
//! - A wgpu adapter is available at runtime.
//! - `n_params >= threshold` (default `GPU_CG_THRESHOLD = 4096`).
//!
//! ## Precision note
//! CG uses `f64` internally; the GPU operates on `f32`.  Vectors are cast to `f32`
//! on upload and back to `f64` on download.  For high-precision problems
//! (tolerance ≤ 1e-12) users can disable GPU acceleration by setting
//! `options.use_gpu = false`.

use crate::error::OptimizeError;
use scirs2_core::ndarray::Array1;

/// Minimum number of parameters to trigger GPU dispatch for CG.
pub(crate) const GPU_CG_THRESHOLD: usize = 4096;

/// Result of attempting GPU-accelerated CG direction update.
pub(crate) enum GpuCgResult {
    /// GPU dispatch succeeded; contains `(g_new_norm_sq, beta, new_direction)`.
    Done(f64, f64, Array1<f64>),
    /// GPU not available/applicable; caller should use the CPU path.
    FallbackToCpu,
}

/// Attempt to compute the CG direction update on the GPU.
///
/// Computes:
/// - `g_new_norm_sq = g_new · g_new`  (via GPU dot)
/// - `g_norm_sq = g · g`              (via GPU dot)
/// - `beta = g_new_norm_sq / g_norm_sq`
/// - `new_p = -g_new + beta * p`      (via GPU subtract + scale)
///
/// Returns `GpuCgResult::Done(g_new_norm_sq, beta, new_p)` on success,
/// or `GpuCgResult::FallbackToCpu` when GPU is unavailable or disabled.
///
/// # Parameters
/// - `g`         — previous gradient (host f64).
/// - `g_new`     — new gradient (host f64).
/// - `p`         — previous search direction (host f64).
/// - `use_gpu`   — from `Options::use_gpu`.
/// - `threshold` — effective GPU threshold.
pub(crate) fn try_gpu_cg_update(
    g: &Array1<f64>,
    g_new: &Array1<f64>,
    p: &Array1<f64>,
    use_gpu: bool,
    threshold: usize,
) -> GpuCgResult {
    let n = g.len();

    if !use_gpu || n < threshold {
        return GpuCgResult::FallbackToCpu;
    }

    do_gpu_cg_dispatch(g, g_new, p)
}

/// Dispatch to GPU implementation; compiled only when `gpu` feature is active.
fn do_gpu_cg_dispatch(g: &Array1<f64>, g_new: &Array1<f64>, p: &Array1<f64>) -> GpuCgResult {
    #[cfg(feature = "gpu")]
    {
        use scirs2_core::array_protocol::gpu_ndarray::{global_context, is_gpu_available};

        if !is_gpu_available() {
            return GpuCgResult::FallbackToCpu;
        }
        let ctx = match global_context() {
            Some(c) => c,
            None => return GpuCgResult::FallbackToCpu,
        };
        match gpu_cg_update_inner(g, g_new, p, &ctx) {
            Ok(result) => result,
            // Silently fall back on any GPU error.
            Err(_) => GpuCgResult::FallbackToCpu,
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (g, g_new, p);
        GpuCgResult::FallbackToCpu
    }
}

/// Core GPU implementation — only compiled with the `gpu` feature.
#[cfg(feature = "gpu")]
fn gpu_cg_update_inner(
    g: &Array1<f64>,
    g_new: &Array1<f64>,
    p: &Array1<f64>,
    ctx: &std::sync::Arc<scirs2_core::gpu::backends::WebGPUContext>,
) -> Result<GpuCgResult, OptimizeError> {
    use scirs2_core::array_protocol::gpu_ndarray::GpuNdarray;

    let n = g.len();

    // --- upload helper: f64 Array1 → 1-D GpuNdarray<f32> ----------------
    let upload = |arr: &Array1<f64>| -> Result<GpuNdarray<f32>, OptimizeError> {
        let data_f32: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
        GpuNdarray::from_ndarray_data(&data_f32, vec![n], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU upload: {e}")))
    };

    // --- GPU dot: a · b = sum(a * b) → f64 ------------------------------
    let gpu_dot = |a: &GpuNdarray<f32>, b: &GpuNdarray<f32>| -> Result<f64, OptimizeError> {
        a.dot_gpu(b)
            .map(f64::from)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU dot: {e}")))
    };

    // Upload vectors.
    let g_gpu = upload(g)?;
    let g_new_gpu = upload(g_new)?;
    let p_gpu = upload(p)?;

    // Compute norm-squares.
    let g_new_norm_sq = gpu_dot(&g_new_gpu, &g_new_gpu)?;
    let g_norm_sq = gpu_dot(&g_gpu, &g_gpu)?;

    // Compute beta (Fletcher-Reeves).
    let beta = if g_norm_sq < 1e-10 {
        0.0_f64
    } else {
        g_new_norm_sq / g_norm_sq
    };

    // Compute new direction: new_p = -g_new + beta * p  (GPU).
    // Step 1: scaled_p = beta * p
    let scaled_p = p_gpu
        .multiply_by_scalar_f32(beta as f32)
        .map_err(|e| OptimizeError::ComputationError(format!("GPU scale p: {e}")))?;

    // Step 2: neg_g_new = g_new * (-1)
    let neg_g_new = g_new_gpu
        .multiply_by_scalar_f32(-1.0_f32)
        .map_err(|e| OptimizeError::ComputationError(format!("GPU negate g_new: {e}")))?;

    // Step 3: new_p = neg_g_new + scaled_p
    let new_p_gpu = neg_g_new
        .add(&scaled_p)
        .map_err(|e| OptimizeError::ComputationError(format!("GPU direction add: {e}")))?;

    // Download result.
    let new_p_host = new_p_gpu
        .to_vec()
        .map_err(|e| OptimizeError::ComputationError(format!("GPU download: {e}")))?;

    if new_p_host.len() != n {
        return Err(OptimizeError::ComputationError(format!(
            "GPU CG result length mismatch: got {}, expected {n}",
            new_p_host.len()
        )));
    }

    let new_p = Array1::from_iter(new_p_host.into_iter().map(f64::from));

    Ok(GpuCgResult::Done(g_new_norm_sq, beta, new_p))
}
