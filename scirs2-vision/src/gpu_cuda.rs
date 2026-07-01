#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for 2D image
//! convolution, calling the pure-Rust `oxicuda-dnn` library crate directly.
//!
//! This module is ADDITIVE and entirely feature-gated behind the `cuda` feature
//! (off by default). It does NOT replace or modify the existing wgpu GPU
//! subsystem (`src/gpu_ops.rs`, `src/gpu_modules/`). The public entry point is
//! `f64`-native — the wgpu path is `f32`; this oxicuda CUDA path is its `f64`
//! analog, with no silent downcast at the boundary.
//!
//! [`cuda_is_available`] never panics, and [`cuda_convolve_2d`] returns a
//! [`VisionError::GpuError`] rather than aborting when no NVIDIA device is
//! present.
//!
//! # Numerical equivalence to `simd_convolve_2d` (see `src/simd_ops.rs`)
//!
//! `cuda_convolve_2d` is the `f64` analog of `crate::simd_ops::simd_convolve_2d`
//! and is built to reproduce its result EXACTLY. Reading that function:
//!
//! * It allocates `Array2::zeros((height, width))` (output is the SAME size as
//!   the input) and only writes the INTERIOR region
//!   `y in k_half_h..(height - k_half_h)`, `x in k_half_w..(width - k_half_w)`.
//!   The border ring (width `k_half`) is left as hard zeros — it is NOT a
//!   zero-padded 'same' convolution; it is a 'valid' convolution embedded in the
//!   interior of a zero-filled same-size array. The written interior extent is
//!   exactly `(height - k_height + 1) x (width - k_width + 1)` (kernel is odd).
//! * Its inner loop computes
//!   `out[y,x] = sum over ky,kx of image[y+ky-k_half_h, x+kx-k_half_w] * kernel[ky,kx]`
//!   — the kernel index `(ky,kx)` is aligned with the forward image offset, with
//!   NO 180-degree flip. That is CROSS-CORRELATION, not true convolution.
//!
//! `oxicuda-dnn`'s `conv_forward` ALSO computes cross-correlation (the cuDNN
//! convention — no `mode`/CONVOLUTION enum exists; its CPU reference and
//! implicit-GEMM kernel use `ih = oh*stride - pad + kr*dilation`, with no kernel
//! reversal). Both conventions match, so the kernel is uploaded AS-IS with NO
//! pre-flip.
//!
//! To reproduce the hard-zero borders exactly, we run a VALID convolution
//! (`pad = 0`) producing an `(out_h, out_w) = (H - k_h + 1, W - k_w + 1)` result
//! on the GPU and embed it into a zero-filled `(H, W)` array at offset
//! `(k_half_h, k_half_w)` — structurally identical to what `simd_convolve_2d`
//! does. (Using oxicuda 'same' padding would instead compute non-zero
//! zero-padded border values, which would NOT match `simd_convolve_2d`'s hard
//! zeros.) This makes correctness provable by construction even though there is
//! no NVIDIA device on the build host to runtime-verify it.

use crate::error::{Result, VisionError};
use oxicuda_dnn::conv::conv_forward;
use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut};
use oxicuda_dnn::{DnnError, DnnHandle};
use oxicuda_memory::DeviceBuffer;
use scirs2_core::ndarray::{s, Array2, ArrayView2};

/// Returns `true` iff the CUDA driver initializes and at least one NVIDIA device
/// is visible. Never panics. On non-NVIDIA platforms (e.g. macOS) the driver
/// fails to initialize or reports zero devices and this returns `false`.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Maps an `oxicuda-dnn` error onto the crate's `GpuError`.
fn dnn_err(e: DnnError) -> VisionError {
    VisionError::GpuError(format!("oxicuda-dnn: {e}"))
}

/// Maps an `oxicuda` CUDA driver/memory error onto `GpuError`.
fn cuda_err(e: oxicuda_driver::CudaError) -> VisionError {
    VisionError::GpuError(format!("oxicuda CUDA driver: {e}"))
}

/// Initializes the CUDA driver, selects device 0, and builds a shared context
/// wrapped in an `Arc` as required by the oxicuda-dnn handle constructor.
fn build_context() -> Result<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init().map_err(|e| VisionError::GpuError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| VisionError::GpuError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(VisionError::GpuError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

/// 2D image convolution on the GPU in `f64` via `oxicuda-dnn`'s `conv_forward`.
///
/// This is the `f64` analog of `crate::simd_ops::simd_convolve_2d` and
/// reproduces its result exactly: cross-correlation (no kernel flip), hard-zero
/// borders, and an output the same `H x W` size as the input. See the
/// module-level docs for the padding/flip equivalence proof.
///
/// Returns [`VisionError::InvalidInput`] for an even-sized kernel, an empty
/// image, or a kernel larger than the image (mirroring / hardening
/// `simd_convolve_2d`'s own validation), and [`VisionError::GpuError`] when no
/// NVIDIA device is present or a device operation fails.
pub fn cuda_convolve_2d(image: &ArrayView2<f64>, kernel: &ArrayView2<f64>) -> Result<Array2<f64>> {
    let (height, width) = image.dim();
    let (k_height, k_width) = kernel.dim();

    // Mirror simd_convolve_2d's validation: kernel must be odd-sized.
    if k_height % 2 == 0 || k_width % 2 == 0 {
        return Err(VisionError::InvalidInput(
            "Kernel must have odd dimensions".to_string(),
        ));
    }
    // Non-empty image (simd_convolve_2d would underflow on an empty image).
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidInput(
            "Image must be non-empty".to_string(),
        ));
    }
    // Kernel must fit so the 'valid' interior is non-empty (this is exactly the
    // region simd_convolve_2d writes; a larger kernel would underflow there).
    if height < k_height || width < k_width {
        return Err(VisionError::InvalidInput(format!(
            "Kernel ({k_height}x{k_width}) larger than image ({height}x{width})"
        )));
    }

    let k_half_h = k_height / 2;
    let k_half_w = k_width / 2;
    let out_h = height - k_height + 1; // 'valid' interior height
    let out_w = width - k_width + 1; // 'valid' interior width

    // Row-major contiguous host copies for the device upload.
    let image_std = image.as_standard_layout();
    let image_slice = image_std
        .as_slice()
        .ok_or_else(|| VisionError::GpuError("cuda_convolve_2d: image not contiguous".into()))?;
    let kernel_std = kernel.as_standard_layout();
    let kernel_slice = kernel_std
        .as_slice()
        .ok_or_else(|| VisionError::GpuError("cuda_convolve_2d: kernel not contiguous".into()))?;

    let ctx = build_context()?;
    let handle = DnnHandle::new(&ctx).map_err(dnn_err)?;

    // Upload image (NCHW: N=1, C=1, H, W) and kernel (NCHW: K=1, C=1, R, S);
    // no kernel flip, since both sides are cross-correlation.
    let d_input = DeviceBuffer::from_host(image_slice).map_err(cuda_err)?;
    let d_filter = DeviceBuffer::from_host(kernel_slice).map_err(cuda_err)?;
    let mut d_output = DeviceBuffer::<f64>::alloc(out_h * out_w).map_err(cuda_err)?;

    let input_desc =
        TensorDesc::<f64>::nchw(&d_input, 1, 1, height as u32, width as u32).map_err(dnn_err)?;
    let filter_desc = TensorDesc::<f64>::nchw(&d_filter, 1, 1, k_height as u32, k_width as u32)
        .map_err(dnn_err)?;
    let mut output_desc =
        TensorDescMut::<f64>::nchw(&mut d_output, 1, 1, out_h as u32, out_w as u32)
            .map_err(dnn_err)?;

    // 'valid' convolution: pad = 0, stride = 1, dilation = 1, groups = 1.
    let conv_desc = ConvolutionDescriptor::conv2d(0, 0, 1, 1, 1, 1, 1).map_err(dnn_err)?;

    // Run the forward pass. Direct / implicit-GEMM algorithms need no workspace
    // (None); if the engine selects an algorithm requiring scratch it returns
    // DnnError::WorkspaceRequired(bytes) — allocate the byte buffer and retry.
    match conv_forward::<f64>(
        &handle,
        &input_desc,
        &filter_desc,
        &mut output_desc,
        &conv_desc,
        None,
    ) {
        Ok(()) => {}
        Err(DnnError::WorkspaceRequired(bytes)) => {
            let mut workspace = DeviceBuffer::<u8>::alloc(bytes).map_err(cuda_err)?;
            conv_forward::<f64>(
                &handle,
                &input_desc,
                &filter_desc,
                &mut output_desc,
                &conv_desc,
                Some(&mut workspace),
            )
            .map_err(dnn_err)?;
        }
        Err(e) => return Err(dnn_err(e)),
    }

    // Download the 'valid' result and embed it into the interior of a
    // zero-filled (height, width) array — exactly as simd_convolve_2d does.
    let mut host_out = vec![0.0f64; out_h * out_w];
    d_output.copy_to_host(&mut host_out).map_err(cuda_err)?;
    let valid = Array2::from_shape_vec((out_h, out_w), host_out)
        .map_err(|e| VisionError::GpuError(format!("output reshape: {e}")))?;

    let mut output = Array2::zeros((height, width));
    output
        .slice_mut(s![k_half_h..k_half_h + out_h, k_half_w..k_half_w + out_w])
        .assign(&valid);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// CPU cross-correlation with hard-zero borders — a faithful mirror of
    /// `simd_convolve_2d` used to check the GPU result when a device is present.
    fn cpu_reference(image: &ArrayView2<f64>, kernel: &ArrayView2<f64>) -> Array2<f64> {
        let (height, width) = image.dim();
        let (k_height, k_width) = kernel.dim();
        let k_half_h = k_height / 2;
        let k_half_w = k_width / 2;
        let mut out = Array2::zeros((height, width));
        for y in k_half_h..(height - k_half_h) {
            for x in k_half_w..(width - k_half_w) {
                let mut acc = 0.0;
                for ky in 0..k_height {
                    for kx in 0..k_width {
                        acc += image[[y + ky - k_half_h, x + kx - k_half_w]] * kernel[[ky, kx]];
                    }
                }
                out[[y, x]] = acc;
            }
        }
        out
    }

    #[test]
    fn cuda_convolve_2d_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let image = Array2::from_shape_vec((5, 5), (1..=25).map(|v| v as f64).collect())
            .expect("valid 5x5 image");
        let kernel =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
                .expect("valid 3x3 kernel");
        let got = cuda_convolve_2d(&image.view(), &kernel.view()).expect("cuda_convolve_2d failed");
        let expected = cpu_reference(&image.view(), &kernel.view());
        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    #[test]
    fn cuda_convolve_2d_rejects_even_kernel() {
        // Runs without a GPU: validation rejects before any device call.
        let image = Array2::<f64>::zeros((5, 5));
        let kernel = Array2::<f64>::zeros((2, 2));
        assert!(cuda_convolve_2d(&image.view(), &kernel.view()).is_err());
    }

    #[test]
    fn cuda_convolve_2d_rejects_empty_image() {
        // Runs without a GPU: validation rejects before any device call.
        let image = Array2::<f64>::zeros((0, 0));
        let kernel = Array2::<f64>::zeros((3, 3));
        assert!(cuda_convolve_2d(&image.view(), &kernel.view()).is_err());
    }

    // ---- Non-symmetric kernel + non-square image + border check ----
    //
    // The existing `cuda_convolve_2d_or_skip` test uses a symmetric plus-kernel
    // on a square 5×5 image.  This test exercises:
    //
    //  • A fully non-symmetric 3×3 kernel [[1,2,3],[4,5,6],[7,8,9]]: any silent
    //    kernel flip (true-convolution vs cross-correlation) or row/col transposition
    //    in the device upload path would produce numerically wrong interior values.
    //
    //  • A non-square image (5 rows × 7 cols): any height/width swap in the
    //    NCHW tensor descriptor would cause an out-of-bounds access or wrong result.
    //
    //  • Hard-zero border ring: `cuda_convolve_2d` runs a VALID convolution on the
    //    GPU and embeds the (H-kH+1)×(W-kW+1) interior result into a zero-filled
    //    H×W array, so the border pixels at row 0, row H-1, col 0, col W-1 must
    //    be EXACTLY 0.0 — not merely small, but bitwise zero.
    //
    //  The GPU result is compared against the local `cpu_reference` cross-correlation
    //  (faithful mirror of `simd_convolve_2d`) within 1e-9.
    #[test]
    fn cuda_convolve_2d_asymmetric_kernel_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        // Non-square image: 5 rows × 7 cols, values 1..35.
        let image = Array2::from_shape_vec((5, 7), (1..=35).map(|v| v as f64).collect::<Vec<_>>())
            .expect("valid 5×7 image");

        // Fully non-symmetric 3×3 kernel: [[1,2,3],[4,5,6],[7,8,9]].
        // A true-convolution kernel flip would rotate this 180° → [[9,8,7],[6,5,4],[3,2,1]],
        // producing different interior values and catching any mode confusion.
        let kernel =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("valid 3×3 kernel");

        let got = cuda_convolve_2d(&image.view(), &kernel.view()).expect("cuda_convolve_2d failed");
        let expected = cpu_reference(&image.view(), &kernel.view());

        // ── Interior values: GPU must match CPU cross-correlation within 1e-9.
        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "non-symmetric kernel max abs diff {max_diff:.3e} exceeds 1e-9 \
             (image 5×7, kernel [[1..9]])"
        );

        // ── Border ring: all pixels in the k_half=1 border must be exactly 0.
        // For a 3×3 kernel on a 5×7 image: rows {0, 4} and cols {0, 6} form the
        // border.  These slots were never written by the VALID convolution; they
        // must be exactly 0 (not just small), matching simd_convolve_2d semantics.
        let (height, width) = got.dim(); // (5, 7)
        let k_half = 1_usize; // floor(3/2)

        // Top and bottom border rows.
        for row_idx in [0, height - 1] {
            for col_idx in 0..width {
                let val = got[[row_idx, col_idx]];
                assert!(
                    val == 0.0,
                    "border pixel [{row_idx},{col_idx}] is {val} (expected exactly 0.0)"
                );
            }
        }
        // Left and right border columns (excluding corners already checked above).
        for row_idx in k_half..(height - k_half) {
            for col_idx in [0, width - 1] {
                let val = got[[row_idx, col_idx]];
                assert!(
                    val == 0.0,
                    "border pixel [{row_idx},{col_idx}] is {val} (expected exactly 0.0)"
                );
            }
        }
    }
}
