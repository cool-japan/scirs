//! GPU-accelerated Newton-CG inner-loop (Hessian-vector products and dot products).
//!
//! This module provides `try_gpu_newton_cg_solve` which replaces `solve_newton_cg_system`
//! in `minimize_newton_cg` when:
//! - The `gpu` feature is enabled (`scirs2-core/array_protocol_wgpu`).
//! - A wgpu adapter is available at runtime.
//! - `n_params >= threshold` (default `GPU_NEWTON_THRESHOLD = 4096`).
//!
//! ## What is GPU-accelerated
//! The inner CG loop of the Newton-CG subsystem performs, per iteration:
//! - `hp = H * p`       — Hessian-vector product (matrix-vector matmul, O(n²))
//! - `php = p · hp`     — curvature scalar (dot product)
//! - `r · r`            — residual norm squared (dot product)
//!
//! The matrix-vector product is the dominant cost for large `n`.  The GPU
//! kernel uses a tiled 16×16 matmul (WGSL), so `p` is uploaded as `[n,1]`
//! and the result is reshaped back to a 1-D vector.
//!
//! ## Precision note
//! The GPU operates on `f32`.  For the inner CG convergence criterion
//! `cg_tol = min(0.1, r0_norm * gtol)` with `gtol = 1e-5`, f32 precision
//! (~7 significant digits) is borderline.  If the inner CG does not converge
//! to tolerance, the result is still usable (inexact Newton) but may require
//! more outer iterations.  Set `options.use_gpu = false` to force f64 paths.

use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2};

/// Minimum number of parameters to trigger GPU dispatch for Newton-CG.
pub(crate) const GPU_NEWTON_THRESHOLD: usize = 4096;

/// Result of attempting GPU-accelerated Newton-CG solve.
pub(crate) enum GpuNewtonResult {
    /// GPU dispatch succeeded; contains the Newton step direction.
    Done(Array1<f64>),
    /// GPU not available/applicable; caller should use the CPU path.
    FallbackToCpu,
}

/// Attempt to solve the Newton-CG system `Hx = -g` using GPU-accelerated inner CG.
///
/// Returns `GpuNewtonResult::Done(step)` on success, or `GpuNewtonResult::FallbackToCpu`
/// when GPU is unavailable or disabled.
///
/// # Parameters
/// - `g`         — current gradient (host f64).
/// - `hess`      — Hessian matrix (host f64, `n × n`).
/// - `tol`       — convergence tolerance (from `Options::gtol`).
/// - `use_gpu`   — from `Options::use_gpu`.
/// - `threshold` — effective GPU threshold.
pub(crate) fn try_gpu_newton_cg_solve(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    tol: f64,
    use_gpu: bool,
    threshold: usize,
) -> GpuNewtonResult {
    let n = g.len();

    if !use_gpu || n < threshold {
        return GpuNewtonResult::FallbackToCpu;
    }

    do_gpu_newton_dispatch(g, hess, tol)
}

/// Dispatch to GPU implementation; compiled only when `gpu` feature is active.
fn do_gpu_newton_dispatch(g: &Array1<f64>, hess: &Array2<f64>, tol: f64) -> GpuNewtonResult {
    #[cfg(feature = "gpu")]
    {
        use scirs2_core::array_protocol::gpu_ndarray::{global_context, is_gpu_available};

        if !is_gpu_available() {
            return GpuNewtonResult::FallbackToCpu;
        }
        let ctx = match global_context() {
            Some(c) => c,
            None => return GpuNewtonResult::FallbackToCpu,
        };
        match gpu_newton_cg_inner(g, hess, tol, &ctx) {
            Ok(result) => result,
            // Silently fall back on any GPU error.
            Err(_) => GpuNewtonResult::FallbackToCpu,
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (g, hess, tol);
        GpuNewtonResult::FallbackToCpu
    }
}

/// Core GPU implementation — only compiled with the `gpu` feature.
#[cfg(feature = "gpu")]
fn gpu_newton_cg_inner(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    tol: f64,
    ctx: &std::sync::Arc<scirs2_core::gpu::backends::WebGPUContext>,
) -> Result<GpuNewtonResult, OptimizeError> {
    use scirs2_core::array_protocol::gpu_ndarray::GpuNdarray;

    let n = g.len();

    // --- upload helpers ---------------------------------------------------

    // Upload a 1-D f64 Array1 as a 1-D f32 GpuNdarray (shape [n]).
    let upload_vec = |arr: &Array1<f64>| -> Result<GpuNdarray<f32>, OptimizeError> {
        let data_f32: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
        GpuNdarray::from_ndarray_data(&data_f32, vec![n], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU vec upload: {e}")))
    };

    // Upload a 1-D f64 vector as a 2-D column [n,1] f32 GpuNdarray.
    // Required because `dispatch_matmul` only accepts 2-D arrays.
    let upload_col = |arr: &Array1<f64>| -> Result<GpuNdarray<f32>, OptimizeError> {
        let data_f32: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
        GpuNdarray::from_ndarray_data(&data_f32, vec![n, 1], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU col upload: {e}")))
    };

    // Upload the Hessian matrix (n × n) as a 2-D f32 GpuNdarray.
    let upload_hess = |h: &Array2<f64>| -> Result<GpuNdarray<f32>, OptimizeError> {
        let data_f32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        GpuNdarray::from_ndarray_data(&data_f32, vec![n, n], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU hess upload: {e}")))
    };

    // GPU dot product of two 1-D arrays.
    let gpu_dot = |a: &GpuNdarray<f32>, b: &GpuNdarray<f32>| -> Result<f64, OptimizeError> {
        a.dot_gpu(b)
            .map(f64::from)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU dot: {e}")))
    };

    // GPU matmul for Hessian-vector product: H [n×n] × p_col [n×1] → result [n×1].
    // Returns the result as a 1-D GpuNdarray of shape [n].
    let gpu_hessian_vector = |h_gpu: &GpuNdarray<f32>,
                              p_col: &GpuNdarray<f32>|
     -> Result<GpuNdarray<f32>, OptimizeError> {
        // matmul returns [n,1].
        let result_2d = h_gpu
            .matmul(p_col)
            .map_err(|e| OptimizeError::ComputationError(format!("GPU matmul: {e}")))?;

        // Flatten [n,1] → [n] by downloading and re-uploading (cheapest available path).
        let flat = result_2d
            .to_vec()
            .map_err(|e| OptimizeError::ComputationError(format!("GPU download hp: {e}")))?;

        if flat.len() != n {
            return Err(OptimizeError::ComputationError(format!(
                "GPU matmul output length mismatch: got {}, expected {n}",
                flat.len()
            )));
        }

        GpuNdarray::from_ndarray_data(&flat, vec![n], std::sync::Arc::clone(ctx))
            .map_err(|e| OptimizeError::ComputationError(format!("GPU re-upload hp: {e}")))
    };

    // --- Upload Hessian once (stays on GPU for all CG iters) ---------------
    let h_gpu = upload_hess(hess)?;

    // --- CG initialization ------------------------------------------------
    // Solve: H x = -g  starting from x = 0.
    // r = -g - H*0 = -g;  p = r.

    let mut x = Array1::zeros(n); // kept on CPU as accumulator

    // If gradient is zero, return zero step.
    if g.dot(g) < 1e-10 {
        return Ok(GpuNewtonResult::Done(x));
    }

    let neg_g: Array1<f64> = -g; // -g
    let mut r = neg_g.clone(); // r = -g
    let mut p_vec = r.clone(); // p = r  (search direction)

    let r0_norm = r.dot(&r).sqrt();
    let cg_tol = f64::min(0.1, r0_norm * tol);
    let max_cg_iters = 2 * n;

    for _ in 0..max_cg_iters {
        // Upload p as a 2-D column for matmul.
        let p_col = upload_col(&p_vec)?;
        let hp_1d = gpu_hessian_vector(&h_gpu, &p_col)?;

        // p · (H p) — curvature.  Upload p as 1-D for dot.
        let p_1d = upload_vec(&p_vec)?;
        let php = gpu_dot(&p_1d, &hp_1d)?;

        if php <= 1e-10 {
            // Negative or negligible curvature — return current iterate.
            return Ok(GpuNewtonResult::Done(x));
        }

        // r · r (current residual).
        let r_1d = upload_vec(&r)?;
        let rr = gpu_dot(&r_1d, &r_1d)?;

        let alpha = rr / php;

        // x ← x + alpha * p  (CPU, x is small scalar update)
        x = &x + &(&p_vec * alpha);

        // r_new = r - alpha * H p  (download hp from GPU, do CPU update).
        let hp_host: Vec<f64> = hp_1d
            .to_vec()
            .map_err(|e| OptimizeError::ComputationError(format!("GPU download hp_host: {e}")))?
            .into_iter()
            .map(f64::from)
            .collect();
        let hp_arr = Array1::from_vec(hp_host);
        let r_new = &r - &(&hp_arr * alpha);

        // Check convergence.
        if r_new.dot(&r_new).sqrt() < cg_tol {
            // Update x for this final step and break.
            r = r_new;
            break;
        }

        // beta = (r_new · r_new) / (r · r).
        let r_new_norm_sq = r_new.dot(&r_new);
        let beta = r_new_norm_sq / rr;

        // p_new = r_new + beta * p.
        p_vec = &r_new + &(&p_vec * beta);
        r = r_new;
    }

    let _ = r; // suppress unused-variable warning
    Ok(GpuNewtonResult::Done(x))
}
