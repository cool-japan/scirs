//! Real `GpuNdarray<f32>` dispatch for the heavy linear-algebra steps of the
//! GPU dataset generators.
//!
//! This module mirrors the canonical pattern used by
//! `scirs2-optimize/src/unconstrained/lbfgs_gpu.rs`:
//!
//! 1. **Threshold-gate** — only dispatch to the GPU when the problem is large
//!    enough ([`GPU_DATASET_THRESHOLD`]); otherwise the caller uses CPU.
//! 2. **Probe availability** — `is_gpu_available()` / `global_context()` with a
//!    graceful fall back to CPU when no wgpu adapter is present.
//! 3. **Upload** host `f64` → GPU `f32` via
//!    [`GpuNdarray::from_ndarray_data`](scirs2_core::array_protocol::gpu_ndarray::GpuNdarray::from_ndarray_data).
//! 4. **Run ops** — `matmul` (regression target) / `add` +
//!    `multiply_by_scalar_f32` (classification / blobs broadcast offsets).
//! 5. **Read back** GPU `f32` → host `f64` via `.to_vec()`.
//! 6. **On ANY GPU error → fall back to CPU** (never panic).
//!
//! ## Precision note
//! The generators use `f64` internally; the GPU operates on `f32`.  Values are
//! cast to `f32` on upload and back to `f64` on download, so the GPU path is
//! numerically equivalent to the CPU path only within `f32` tolerance
//! (~`1e-4`).  Because the random draws are performed on the host *before*
//! upload, both paths consume identical RNG sequences and therefore agree up to
//! `f32` rounding of the arithmetic itself.

/// Minimum number of output elements required to trigger GPU dispatch.
///
/// Below this size the host↔device transfer overhead dominates and the CPU
/// path is faster, so [`try_*`](self) helpers return `FallbackToCpu`.
pub(crate) const GPU_DATASET_THRESHOLD: usize = 4096;

/// Outcome of an attempted GPU dispatch.
pub(crate) enum GpuDispatch<T> {
    /// GPU dispatch succeeded; carry the computed result.
    Done(T),
    /// GPU not available/applicable; caller should run the CPU path.
    FallbackToCpu,
}

/// Attempt the regression target computation `y = X[:, :k] · coef` on the GPU.
///
/// `data` is the row-major `[n_samples, n_features]` design matrix, `coef` the
/// length-`n_informative` coefficient vector.  Returns the *noise-free* targets
/// (length `n_samples`); the caller adds per-sample noise afterwards so the RNG
/// sequence matches the CPU path exactly.
///
/// Falls back to CPU when the output is below [`GPU_DATASET_THRESHOLD`], the
/// `gpu_wgpu` feature is off, or no adapter is available.
pub(crate) fn try_regression_targets_gpu(
    data: &[f64],
    coef: &[f64],
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
) -> GpuDispatch<Vec<f64>> {
    // Threshold on the matmul work (samples × informative columns).
    if n_samples.saturating_mul(n_informative) < GPU_DATASET_THRESHOLD {
        return GpuDispatch::FallbackToCpu;
    }
    regression_targets_gpu_inner(data, coef, n_samples, n_features, n_informative)
}

/// Attempt the classification informative-feature transform on the GPU.
///
/// Computes `informative = centroids + 0.3 * noise` elementwise, where both
/// `centroids` (per-sample broadcast centroid coordinates) and `noise` are
/// row-major `[n_samples, n_informative]` host buffers drawn on the CPU in the
/// canonical order.  Returns the flat `[n_samples, n_informative]` result.
///
/// Falls back to CPU below [`GPU_DATASET_THRESHOLD`] or when no adapter exists.
pub(crate) fn try_classification_informative_gpu(
    centroids: &[f64],
    noise: &[f64],
    n_samples: usize,
    n_informative: usize,
) -> GpuDispatch<Vec<f64>> {
    let n = n_samples.saturating_mul(n_informative);
    if n < GPU_DATASET_THRESHOLD {
        return GpuDispatch::FallbackToCpu;
    }
    affine_offset_gpu_inner(centroids, noise, 0.3, n)
}

/// Attempt the blob sample generation `sample = center + noise` on the GPU.
///
/// `center_broadcast` and `noise` are row-major `[n_samples_center, n_features]`
/// host buffers; the center row is broadcast across all samples on the CPU
/// before upload.  Returns the flat `[n_samples_center, n_features]` result.
///
/// Falls back to CPU below [`GPU_DATASET_THRESHOLD`] or when no adapter exists.
pub(crate) fn try_blobs_center_gpu(
    center_broadcast: &[f64],
    noise: &[f64],
    n_samples_center: usize,
    n_features: usize,
) -> GpuDispatch<Vec<f64>> {
    let n = n_samples_center.saturating_mul(n_features);
    if n < GPU_DATASET_THRESHOLD {
        return GpuDispatch::FallbackToCpu;
    }
    // sample = center + 1.0 * noise
    affine_offset_gpu_inner(center_broadcast, noise, 1.0, n)
}

// ──────────────────────────────────────────────────────────────────────
// Inner implementations — compiled with the GPU kernels only when the
// `gpu_wgpu` feature is active; otherwise they always fall back to CPU.
// ──────────────────────────────────────────────────────────────────────

/// `base + scale * delta`, elementwise, on the GPU (length `n`).
#[cfg(feature = "gpu_wgpu")]
fn affine_offset_gpu_inner(
    base: &[f64],
    delta: &[f64],
    scale: f32,
    n: usize,
) -> GpuDispatch<Vec<f64>> {
    use scirs2_core::array_protocol::gpu_ndarray::{global_context, is_gpu_available, GpuNdarray};
    use std::sync::Arc;

    if base.len() != n || delta.len() != n {
        return GpuDispatch::FallbackToCpu;
    }
    if !is_gpu_available() {
        return GpuDispatch::FallbackToCpu;
    }
    let ctx = match global_context() {
        Some(c) => c,
        None => return GpuDispatch::FallbackToCpu,
    };

    let run = || -> Result<Vec<f64>, scirs2_core::gpu::GpuError> {
        let base_f32: Vec<f32> = base.iter().map(|&v| v as f32).collect();
        let delta_f32: Vec<f32> = delta.iter().map(|&v| v as f32).collect();

        let base_gpu = GpuNdarray::<f32>::from_ndarray_data(&base_f32, vec![n], Arc::clone(&ctx))?;
        let delta_gpu =
            GpuNdarray::<f32>::from_ndarray_data(&delta_f32, vec![n], Arc::clone(&ctx))?;

        let scaled = delta_gpu.multiply_by_scalar_f32(scale)?;
        let summed = base_gpu.add(&scaled)?;
        let host = summed.to_vec()?;
        if host.len() != n {
            return Err(scirs2_core::gpu::GpuError::Other(format!(
                "GPU result length mismatch: got {}, expected {n}",
                host.len()
            )));
        }
        Ok(host.into_iter().map(f64::from).collect())
    };

    match run() {
        Ok(v) => GpuDispatch::Done(v),
        Err(_) => GpuDispatch::FallbackToCpu,
    }
}

/// CPU-only stub used when the `gpu_wgpu` feature is disabled.
#[cfg(not(feature = "gpu_wgpu"))]
fn affine_offset_gpu_inner(
    base: &[f64],
    delta: &[f64],
    scale: f32,
    n: usize,
) -> GpuDispatch<Vec<f64>> {
    let _ = (base, delta, scale, n);
    GpuDispatch::FallbackToCpu
}

/// Regression targets `y = X[:, :k] · coef` via a GPU `matmul`.
#[cfg(feature = "gpu_wgpu")]
fn regression_targets_gpu_inner(
    data: &[f64],
    coef: &[f64],
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
) -> GpuDispatch<Vec<f64>> {
    use scirs2_core::array_protocol::gpu_ndarray::{global_context, is_gpu_available, GpuNdarray};
    use std::sync::Arc;

    if data.len() != n_samples * n_features || coef.len() < n_informative {
        return GpuDispatch::FallbackToCpu;
    }
    if !is_gpu_available() {
        return GpuDispatch::FallbackToCpu;
    }
    let ctx = match global_context() {
        Some(c) => c,
        None => return GpuDispatch::FallbackToCpu,
    };

    let run = || -> Result<Vec<f64>, scirs2_core::gpu::GpuError> {
        // Pack the informative sub-matrix X[:, :k] contiguously as [n_samples, k].
        let mut x_inf = Vec::with_capacity(n_samples * n_informative);
        for i in 0..n_samples {
            let row = i * n_features;
            for j in 0..n_informative {
                x_inf.push(data[row + j] as f32);
            }
        }
        // coef as a [k, 1] column so matmul yields [n_samples, 1].
        let coef_f32: Vec<f32> = coef.iter().take(n_informative).map(|&v| v as f32).collect();

        let x_gpu = GpuNdarray::<f32>::from_ndarray_data(
            &x_inf,
            vec![n_samples, n_informative],
            Arc::clone(&ctx),
        )?;
        let coef_gpu = GpuNdarray::<f32>::from_ndarray_data(
            &coef_f32,
            vec![n_informative, 1],
            Arc::clone(&ctx),
        )?;

        let prod = x_gpu.matmul(&coef_gpu)?; // [n_samples, 1]
        let host = prod.to_vec()?;
        if host.len() != n_samples {
            return Err(scirs2_core::gpu::GpuError::Other(format!(
                "GPU matmul result length mismatch: got {}, expected {n_samples}",
                host.len()
            )));
        }
        Ok(host.into_iter().map(f64::from).collect())
    };

    match run() {
        Ok(v) => GpuDispatch::Done(v),
        Err(_) => GpuDispatch::FallbackToCpu,
    }
}

/// CPU-only stub used when the `gpu_wgpu` feature is disabled.
#[cfg(not(feature = "gpu_wgpu"))]
fn regression_targets_gpu_inner(
    data: &[f64],
    coef: &[f64],
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
) -> GpuDispatch<Vec<f64>> {
    let _ = (data, coef, n_samples, n_features, n_informative);
    GpuDispatch::FallbackToCpu
}
