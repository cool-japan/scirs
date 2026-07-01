//! GPU-accelerated L-BFGS 2-loop recursion.
//!
//! This module provides a `try_gpu_two_loop_recursion` function that replaces
//! the CPU two-loop recursion in `minimize_lbfgs` when:
//! - The `gpu` feature is enabled (`scirs2-core/array_protocol_wgpu`).
//! - A wgpu adapter is available at runtime.
//! - `n_params >= threshold` (default `GPU_LBFGS_THRESHOLD = 4096`).
//!
//! ## Precision note
//! L-BFGS uses `f64` internally; the GPU operates on `f32`.  Vectors are cast
//! to `f32` on upload and back to `f64` on download.  For high-precision
//! problems (tolerance ≤ 1e-12) users can disable GPU acceleration by setting
//! `options.use_gpu = false`.

use crate::error::OptimizeError;
use scirs2_core::ndarray::Array1;

/// Minimum number of parameters to trigger GPU dispatch.
pub(crate) const GPU_LBFGS_THRESHOLD: usize = 4096;

/// Result of attempting GPU dispatch.
pub(crate) enum GpuDispatchResult {
    /// GPU dispatch succeeded; use this search direction (already negated for descent).
    Done(Array1<f64>),
    /// GPU not available/applicable; caller should use the CPU path.
    FallbackToCpu,
}

/// Attempt to compute the L-BFGS two-loop recursion on the GPU.
///
/// Returns `GpuDispatchResult::Done(dir)` on success (direction already negated),
/// or `GpuDispatchResult::FallbackToCpu` when GPU is unavailable or disabled.
///
/// # Parameters
/// - `s_vectors`  — curvature pairs s_k (host f64).
/// - `y_vectors`  — curvature pairs y_k (host f64).
/// - `rho_values` — precomputed 1/(y_k · s_k).
/// - `gradient`   — current gradient q₀ (f64, NOT yet negated).
/// - `use_gpu`    — from `Options::use_gpu`.
/// - `threshold`  — effective GPU threshold.
pub(crate) fn try_gpu_two_loop_recursion(
    s_vectors: &[Array1<f64>],
    y_vectors: &[Array1<f64>],
    rho_values: &[f64],
    gradient: &Array1<f64>,
    use_gpu: bool,
    threshold: usize,
) -> GpuDispatchResult {
    let n = gradient.len();

    if !use_gpu || n < threshold {
        return GpuDispatchResult::FallbackToCpu;
    }

    // The gpu feature gate is resolved at compile time inside the inner function.
    do_gpu_dispatch(s_vectors, y_vectors, rho_values, gradient)
}

/// Dispatch to GPU implementation; compiled only when `gpu` feature is active.
fn do_gpu_dispatch(
    s_vectors: &[Array1<f64>],
    y_vectors: &[Array1<f64>],
    rho_values: &[f64],
    gradient: &Array1<f64>,
) -> GpuDispatchResult {
    #[cfg(feature = "wgpu")]
    {
        use scirs2_core::array_protocol::gpu_ndarray::{global_context, is_gpu_available};

        if !is_gpu_available() {
            return GpuDispatchResult::FallbackToCpu;
        }
        let ctx = match global_context() {
            Some(c) => c,
            None => return GpuDispatchResult::FallbackToCpu,
        };
        match gpu_two_loop_inner(s_vectors, y_vectors, rho_values, gradient, &ctx) {
            Ok(dir) => GpuDispatchResult::Done(dir),
            // Silently fall back on any GPU error.
            Err(_) => GpuDispatchResult::FallbackToCpu,
        }
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (s_vectors, y_vectors, rho_values, gradient);
        GpuDispatchResult::FallbackToCpu
    }
}

/// Core GPU implementation — only compiled with the `gpu` feature.
#[cfg(feature = "wgpu")]
fn gpu_two_loop_inner(
    s_vectors: &[Array1<f64>],
    y_vectors: &[Array1<f64>],
    rho_values: &[f64],
    gradient: &Array1<f64>,
    ctx: &std::sync::Arc<scirs2_core::gpu::backends::WebGPUContext>,
) -> Result<Array1<f64>, OptimizeError> {
    use scirs2_core::array_protocol::gpu_ndarray::GpuNdarray;

    let n = gradient.len();
    let m = s_vectors.len();

    // --- helpers -----------------------------------------------------------

    // Upload a f64 slice as a 1-D f32 GpuNdarray.
    let upload = |data: &[f64]| -> Result<GpuNdarray<f32>, OptimizeError> {
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let len = data_f32.len();
        GpuNdarray::from_ndarray_data(&data_f32, vec![len], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU upload: {e}")))
    };

    // GPU dot product: a · b = sum(a * b).
    let dot_gpu = |a: &GpuNdarray<f32>, b: &GpuNdarray<f32>| -> Result<f64, OptimizeError> {
        a.dot_gpu(b)
            .map(|v| f64::from(v))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU dot: {e}")))
    };

    // --- Upload s and y vectors --------------------------------------------

    let mut s_gpu: Vec<GpuNdarray<f32>> = Vec::with_capacity(m);
    let mut y_gpu: Vec<GpuNdarray<f32>> = Vec::with_capacity(m);
    for sv in s_vectors {
        s_gpu.push(upload(sv.as_slice().ok_or_else(|| {
            OptimizeError::ComputationError("s_vector not contiguous".into())
        })?)?);
    }
    for yv in y_vectors {
        y_gpu.push(upload(yv.as_slice().ok_or_else(|| {
            OptimizeError::ComputationError("y_vector not contiguous".into())
        })?)?);
    }

    // Upload gradient as initial q.
    let grad_slice: Vec<f64> = gradient.iter().copied().collect();
    let mut q_gpu = upload(&grad_slice)?;

    // --- First loop (reverse, i = m-1 down to 0) ---------------------------
    // alpha_values[j] holds the alpha computed at the j-th push (j=0 corresponds to i=m-1).
    let mut alpha_values: Vec<f64> = Vec::with_capacity(m);

    for i in (0..m).rev() {
        let rho_i = rho_values[i];
        let alpha_i = rho_i * dot_gpu(&s_gpu[i], &q_gpu)?;
        alpha_values.push(alpha_i);

        // q ← q - alpha_i * y_i
        let scaled_y = y_gpu[i]
            .multiply_by_scalar_f32(alpha_i as f32)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU scale y: {e}")))?;
        q_gpu = q_gpu
            .subtract(&scaled_y)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU q subtract: {e}")))?;
    }

    // --- Scaling: r = gamma * q --------------------------------------------
    let mut r_gpu = if m > 0 {
        let ys = dot_gpu(&y_gpu[m - 1], &s_gpu[m - 1])?;
        let yy = dot_gpu(&y_gpu[m - 1], &y_gpu[m - 1])?;
        if ys > 0.0 && yy > 0.0 {
            let gamma = (ys / yy) as f32;
            q_gpu
                .multiply_by_scalar_f32(gamma)
                .map_err(|e| OptimizeError::ComputationError(format!("GPU gamma scale: {e}")))?
        } else {
            q_gpu
        }
    } else {
        q_gpu
    };

    // --- Second loop (forward, i = 0 up to m-1) ----------------------------
    // alpha_values was pushed in reverse order: alpha_values[m-1-i] is alpha for index i.
    for i in 0..m {
        let alpha_i = alpha_values[m - 1 - i];
        let rho_i = rho_values[i];
        let beta_i = rho_i * dot_gpu(&y_gpu[i], &r_gpu)?;
        let coeff = (alpha_i - beta_i) as f32;

        // r ← r + coeff * s_i
        let scaled_s = s_gpu[i]
            .multiply_by_scalar_f32(coeff)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU scale s: {e}")))?;
        r_gpu = r_gpu
            .add(&scaled_s)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU r add: {e}")))?;
    }

    // --- Download and negate (L-BFGS returns ascent; negate for descent) ---
    let r_host = r_gpu
        .to_vec()
        .map_err(|e| OptimizeError::ComputationError(format!("GPU download: {e}")))?;

    if r_host.len() != n {
        return Err(OptimizeError::ComputationError(format!(
            "GPU result length mismatch: got {}, expected {n}",
            r_host.len()
        )));
    }

    Ok(Array1::from_iter(
        r_host.into_iter().map(|v| -(f64::from(v))),
    ))
}
