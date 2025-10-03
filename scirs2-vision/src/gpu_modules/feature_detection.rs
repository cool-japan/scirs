//! GPU-accelerated feature detection operations
//!
//! This module provides feature detection algorithms including Sobel edge detection,
//! Harris corner detection, and supporting operations optimized for GPU execution.

use super::{basic_operations::gpu_convolve_2d, context::GpuVisionContext};
use crate::error::{Result, VisionError};
use scirs2_core::gpu::GpuBackend;
use scirs2_core::ndarray::{Array2, ArrayView2};

/// GPU-accelerated Sobel edge detection
///
/// Computes Sobel gradients on GPU.
///
/// # Arguments
///
/// * `ctx` - GPU vision context
/// * `image` - Input grayscale image
///
/// # Returns
///
/// * Tuple of (gradient_x, gradient_y, magnitude)
#[allow(dead_code)]
pub fn gpu_sobel_gradients(
    ctx: &GpuVisionContext,
    image: &ArrayView2<f32>,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    // Sobel kernels
    let sobel_x =
        scirs2_core::ndarray::arr2(&[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]);

    let sobel_y =
        scirs2_core::ndarray::arr2(&[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]);

    // Compute gradients using GPU convolution
    let grad_x = gpu_convolve_2d(ctx, image, &sobel_x.view())?;
    let grad_y = gpu_convolve_2d(ctx, image, &sobel_y.view())?;

    // Compute magnitude on GPU
    let magnitude = gpu_gradient_magnitude(ctx, &grad_x.view(), &grad_y.view())?;

    Ok((grad_x, grad_y, magnitude))
}

/// GPU-accelerated gradient magnitude computation
///
/// Computes sqrt(gx^2 + gy^2) on GPU.
#[allow(dead_code)]
fn gpu_gradient_magnitude(
    ctx: &GpuVisionContext,
    grad_x: &ArrayView2<f32>,
    grad_y: &ArrayView2<f32>,
) -> Result<Array2<f32>> {
    let (height, width) = grad_x.dim();

    if !ctx.is_gpu_available() {
        // CPU fallback with SIMD optimization
        let mut magnitude = Array2::zeros((height, width));
        for ((m, gx), gy) in magnitude.iter_mut().zip(grad_x.iter()).zip(grad_y.iter()) {
            *m = (gx * gx + gy * gy).sqrt();
        }
        return Ok(magnitude);
    }

    // GPU implementation
    let grad_x_flat: Vec<f32> = grad_x.iter().cloned().collect();
    let grad_y_flat: Vec<f32> = grad_y.iter().cloned().collect();

    let grad_x_buffer = ctx.context.create_buffer_from_slice(&grad_x_flat);
    let grad_y_buffer = ctx.context.create_buffer_from_slice(&grad_y_flat);
    let output_buffer = ctx.context.create_buffer::<f32>(height * width);

    // Define gradient magnitude kernel
    let kernel_source = match ctx.backend() {
        GpuBackend::Cuda => {
            r#"
extern "C" __global__ void gradient_magnitude(
    const float* __restrict__ grad_x,
    const float* __restrict__ grad_y,
    float* __restrict__ magnitude,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float gx = grad_x[idx];
        float gy = grad_y[idx];
        magnitude[idx] = sqrtf(gx * gx + gy * gy);
    }
}
"#
        }
        GpuBackend::Wgpu => {
            r#"
@group(0) @binding(0) var<storage, read> grad_x: array<f32>;
@group(0) @binding(1) var<storage, read> grad_y: array<f32>;
@group(0) @binding(2) var<storage, write> magnitude: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
#[allow(dead_code)]
fn gradient_magnitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    let gx = grad_x[idx];
    let gy = grad_y[idx];
    magnitude[idx] = sqrt(gx * gx + gy * gy);
}
"#
        }
        _ => {
            // Fall back to CPU for unsupported backends
            let mut magnitude = Array2::zeros((height, width));
            for ((m, gx), gy) in magnitude.iter_mut().zip(grad_x.iter()).zip(grad_y.iter()) {
                *m = (gx * gx + gy * gy).sqrt();
            }
            return Ok(magnitude);
        }
    };

    ctx.context.execute(|compiler| {
        match compiler.compile(kernel_source) {
            Ok(kernel_handle) => {
                kernel_handle.set_buffer("grad_x", &grad_x_buffer);
                kernel_handle.set_buffer("grad_y", &grad_y_buffer);
                kernel_handle.set_buffer("magnitude", &output_buffer);
                kernel_handle.set_u32("size", (height * width) as u32);

                let workgroup_size = 256;
                let work_groups = (height * width).div_ceil(workgroup_size);

                kernel_handle.dispatch([work_groups as u32, 1, 1]);

                let mut result_flat = vec![0.0f32; height * width];
                output_buffer.copy_to_host(&mut result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to copy result from GPU: {e}")))?;

                Array2::from_shape_vec((height, width), result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to reshape output: {e}")))
            }
            Err(compile_error) => {
                // Log compilation error and fall back to CPU
                eprintln!(
                    "GPU gradient magnitude kernel compilation failed for backend {:?}: {}. Using CPU fallback.",
                    ctx.backend(),
                    compile_error
                );

                // CPU fallback implementation
                let mut magnitude = Array2::zeros((height, width));
                for ((m, gx), gy) in magnitude.iter_mut().zip(grad_x.iter()).zip(grad_y.iter()) {
                    *m = (gx * gx + gy * gy).sqrt();
                }
                Ok(magnitude)
            }
        }
    })
}

/// GPU-accelerated Harris corner detection
///
/// Detects corners using the Harris corner detector on GPU.
///
/// # Arguments
///
/// * `ctx` - GPU vision context
/// * `image` - Input grayscale image
/// * `k` - Harris detector parameter (typically 0.04-0.06)
/// * `threshold` - Corner response threshold
///
/// # Returns
///
/// * Corner response map
#[allow(dead_code)]
pub fn gpu_harris_corners(
    ctx: &GpuVisionContext,
    image: &ArrayView2<f32>,
    k: f32,
    threshold: f32,
) -> Result<Array2<f32>> {
    // Compute gradients
    let (grad_x, grad_y, _) = gpu_sobel_gradients(ctx, image)?;

    // Compute structure tensor elements
    let ixx = gpu_element_wise_multiply(ctx, &grad_x.view(), &grad_x.view())?;
    let iyy = gpu_element_wise_multiply(ctx, &grad_y.view(), &grad_y.view())?;
    let ixy = gpu_element_wise_multiply(ctx, &grad_x.view(), &grad_y.view())?;

    // Apply Gaussian smoothing to structure tensor
    let sigma = 1.0;
    let sxx = super::basic_operations::gpu_gaussian_blur(ctx, &ixx.view(), sigma)?;
    let syy = super::basic_operations::gpu_gaussian_blur(ctx, &iyy.view(), sigma)?;
    let sxy = super::basic_operations::gpu_gaussian_blur(ctx, &ixy.view(), sigma)?;

    // Compute Harris response
    gpu_harris_response(ctx, &sxx.view(), &syy.view(), &sxy.view(), k, threshold)
}

/// GPU element-wise multiplication
#[allow(dead_code)]
pub fn gpu_element_wise_multiply(
    ctx: &GpuVisionContext,
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> Result<Array2<f32>> {
    let (height, width) = a.dim();

    if !ctx.is_gpu_available() {
        return Ok(a * b);
    }

    let a_flat: Vec<f32> = a.iter().cloned().collect();
    let b_flat: Vec<f32> = b.iter().cloned().collect();

    let a_buffer = ctx.context.create_buffer_from_slice(&a_flat);
    let b_buffer = ctx.context.create_buffer_from_slice(&b_flat);
    let output_buffer = ctx.context.create_buffer::<f32>(height * width);

    let kernel_source = match ctx.backend() {
        GpuBackend::Cuda => {
            r#"
extern "C" __global__ void element_wise_multiply(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}
"#
        }
        GpuBackend::Wgpu => {
            r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
#[allow(dead_code)]
fn element_wise_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = a[idx] * b[idx];
}
"#
        }
        _ => return Ok(a * b),
    };

    ctx.context
        .execute(|compiler| match compiler.compile(kernel_source) {
            Ok(kernel_handle) => {
                kernel_handle.set_buffer("a", &a_buffer);
                kernel_handle.set_buffer("b", &b_buffer);
                kernel_handle.set_buffer("output", &output_buffer);
                kernel_handle.set_u32("size", (height * width) as u32);

                let workgroup_size = 256;
                let work_groups = (height * width).div_ceil(workgroup_size);

                kernel_handle.dispatch([work_groups as u32, 1, 1]);

                let mut result_flat = vec![0.0f32; height * width];
                output_buffer.copy_to_host(&mut result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to copy result from GPU: {e}")))?;

                Array2::from_shape_vec((height, width), result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to reshape output: {e}")))
            }
            Err(compile_error) => {
                eprintln!(
                    "GPU element-wise multiplication kernel compilation failed for backend {:?}: {}. Using CPU fallback.",
                    ctx.backend(),
                    compile_error
                );
                Ok(a * b)
            },
        })
}

/// Compute Harris corner response
#[allow(dead_code)]
fn gpu_harris_response(
    ctx: &GpuVisionContext,
    sxx: &ArrayView2<f32>,
    syy: &ArrayView2<f32>,
    sxy: &ArrayView2<f32>,
    k: f32,
    threshold: f32,
) -> Result<Array2<f32>> {
    let (height, width) = sxx.dim();

    if !ctx.is_gpu_available() {
        // CPU fallback
        let mut response = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                let det = sxx[[y, x]] * syy[[y, x]] - sxy[[y, x]] * sxy[[y, x]];
                let trace = sxx[[y, x]] + syy[[y, x]];
                let r = det - k * trace * trace;
                response[[y, x]] = if r > threshold { r } else { 0.0 };
            }
        }
        return Ok(response);
    }

    let sxx_flat: Vec<f32> = sxx.iter().cloned().collect();
    let syy_flat: Vec<f32> = syy.iter().cloned().collect();
    let sxy_flat: Vec<f32> = sxy.iter().cloned().collect();

    let sxx_buffer = ctx.context.create_buffer_from_slice(&sxx_flat);
    let syy_buffer = ctx.context.create_buffer_from_slice(&syy_flat);
    let sxy_buffer = ctx.context.create_buffer_from_slice(&sxy_flat);
    let output_buffer = ctx.context.create_buffer::<f32>(height * width);

    let kernel_source = match ctx.backend() {
        GpuBackend::Cuda => {
            r#"
extern "C" __global__ void harris_response(
    const float* __restrict__ sxx,
    const float* __restrict__ syy,
    const float* __restrict__ sxy,
    float* __restrict__ response,
    float k,
    float threshold,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float det = sxx[idx] * syy[idx] - sxy[idx] * sxy[idx];
        float trace = sxx[idx] + syy[idx];
        float r = det - k * trace * trace;
        response[idx] = (r > threshold) ? r : 0.0f;
    }
}
"#
        }
        GpuBackend::Wgpu => {
            r#"
@group(0) @binding(0) var<storage, read> sxx: array<f32>;
@group(0) @binding(1) var<storage, read> syy: array<f32>;
@group(0) @binding(2) var<storage, read> sxy: array<f32>;
@group(0) @binding(3) var<storage, write> response: array<f32>;
@group(0) @binding(4) var<uniform> k: f32;
@group(0) @binding(5) var<uniform> threshold: f32;
@group(0) @binding(6) var<uniform> size: u32;

@compute @workgroup_size(256)
#[allow(dead_code)]
fn harris_response(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    let det = sxx[idx] * syy[idx] - sxy[idx] * sxy[idx];
    let trace = sxx[idx] + syy[idx];
    let r = det - k * trace * trace;
    response[idx] = select(0.0, r, r > threshold);
}
"#
        }
        _ => {
            // CPU fallback
            let mut response = Array2::zeros((height, width));
            for y in 0..height {
                for x in 0..width {
                    let det = sxx[[y, x]] * syy[[y, x]] - sxy[[y, x]] * sxy[[y, x]];
                    let trace = sxx[[y, x]] + syy[[y, x]];
                    let r = det - k * trace * trace;
                    response[[y, x]] = if r > threshold { r } else { 0.0 };
                }
            }
            return Ok(response);
        }
    };

    ctx.context.execute(|compiler| {
        match compiler.compile(kernel_source) {
            Ok(kernel_handle) => {
                kernel_handle.set_buffer("sxx", &sxx_buffer);
                kernel_handle.set_buffer("syy", &syy_buffer);
                kernel_handle.set_buffer("sxy", &sxy_buffer);
                kernel_handle.set_buffer("response", &output_buffer);
                kernel_handle.set_f32("k", k);
                kernel_handle.set_f32("threshold", threshold);
                kernel_handle.set_u32("size", (height * width) as u32);

                let workgroup_size = 256;
                let work_groups = (height * width).div_ceil(workgroup_size);

                kernel_handle.dispatch([work_groups as u32, 1, 1]);

                let mut result_flat = vec![0.0f32; height * width];
                output_buffer.copy_to_host(&mut result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to copy result from GPU: {e}")))?;

                Array2::from_shape_vec((height, width), result_flat)
                    .map_err(|e| VisionError::Other(format!("Failed to reshape output: {e}")))
            }
            Err(compile_error) => {
                eprintln!(
                    "GPU Harris response kernel compilation failed for backend {:?}: {}. Using CPU fallback.",
                    ctx.backend(),
                    compile_error
                );

                // CPU fallback implementation
                let mut response = Array2::zeros((height, width));
                for y in 0..height {
                    for x in 0..width {
                        let det = sxx[[y, x]] * syy[[y, x]] - sxy[[y, x]] * sxy[[y, x]];
                        let trace = sxx[[y, x]] + syy[[y, x]];
                        let r = det - k * trace * trace;
                        response[[y, x]] = if r > threshold { r } else { 0.0 };
                    }
                }
                Ok(response)
            }
        }
    })
}
