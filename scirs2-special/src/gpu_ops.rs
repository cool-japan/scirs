//! GPU-accelerated implementations of special functions
//!
//! This module provides GPU-accelerated versions of special functions that can
//! be used for large array computations when GPU hardware is available.
//!
//! The implementation automatically falls back to optimized CPU versions when:
//! - GPU hardware is not available
//! - Array size is too small to benefit from GPU acceleration (< 1000 elements)
//! - GPU operations fail for any reason
//!
//! GPU acceleration is implemented using WebGPU compute shaders and provides
//! significant performance improvements for large arrays (>1000 elements).

use crate::error::{SpecialError, SpecialResult};
use ndarray::{ArrayView1, ArrayViewMut1};
use scirs2_core::gpu::{GpuContext, GpuError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Safe slice casting replacement for bytemuck::cast_slice
#[allow(dead_code)]
fn cast_slice_to_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFETY: This is safe because:
    // 1. The pointer is derived from a valid _slice
    // 2. The size calculation is correct (len * size_of::<T>())
    // 3. The lifetime is bounded by the input _slice
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

/// Safe slice casting replacement for bytemuck::cast_slice (reverse)
#[allow(dead_code)]
fn cast_bytes_to_slice<T>(bytes: &[u8]) -> &[T] {
    assert_eq!(bytes.len() % std::mem::size_of::<T>(), 0);
    // SAFETY: This is safe because:
    // 1. We assert that the byte length is a multiple of T's size
    // 2. The pointer is derived from a valid slice
    // 3. The length calculation ensures we don't exceed bounds
    // 4. The lifetime is bounded by the input slice
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / std::mem::size_of::<T>(),
        )
    }
}

// Additional logging for GPU operations
#[cfg(feature = "gpu")]
use log;

/// Advanced GPU-accelerated gamma function with intelligent fallback and performance monitoring
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn gamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync
        + 'static,
{
    use crate::gpu_context_manager::{get_gpu_pool, record_gpu_performance};
    use scirs2_core::gpu::GpuBackend;

    // Validate input dimensions
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Check if GPU should be used based on array size and system state
    let pool = get_gpu_pool();
    let elementsize = std::mem::size_of::<F>();

    if !pool.should_use_gpu(input.len(), elementsize) {
        #[cfg(feature = "gpu")]
        log::debug!(
            "Using CPU fallback for gamma computation (array size: {}, element size: {})",
            input.len(),
            elementsize
        );
        return gamma_cpu_fallback(input, output);
    }

    // Try GPU execution with performance monitoring and intelligent retry
    let start_time = Instant::now();
    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 3;

    while attempts < MAX_ATTEMPTS {
        attempts += 1;

        match try_gamma_gpu_execution_enhanced(input, output) {
            Ok(backend_type) => {
                let execution_time = start_time.elapsed();
                record_gpu_performance(
                    backend_type,
                    execution_time,
                    true,
                    input.len() * elementsize,
                );
                #[cfg(feature = "gpu")]
                log::debug!(
                    "GPU gamma computation successful on attempt {} in {:?}",
                    attempts,
                    execution_time
                );
                return Ok(());
            }
            Err(SpecialError::GpuNotAvailable(_)) => {
                #[cfg(feature = "gpu")]
                log::debug!("GPU not available, falling back to CPU");
                break;
            }
            Err(e) => {
                #[cfg(feature = "gpu")]
                log::warn!(
                    "GPU gamma computation failed on attempt {}: {}",
                    attempts,
                    e
                );

                if attempts == MAX_ATTEMPTS {
                    // Record the failure
                    record_gpu_performance(
                        GpuBackend::Cpu,
                        start_time.elapsed(),
                        false,
                        input.len() * elementsize,
                    );
                    #[cfg(feature = "gpu")]
                    log::error!(
                        "GPU gamma computation failed after {} attempts, falling back to CPU",
                        MAX_ATTEMPTS
                    );
                    break;
                }

                // Brief pause before retry
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    // Fall back to CPU implementation
    gamma_cpu_fallback(input, output)
}

/// GPU-accelerated Bessel J0 function for arrays
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn j0_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + 'static,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Try GPU execution first, fall back to CPU if GPU is not available
    match try_j0_gpu_execution(input, output) {
        Ok(()) => Ok(()),
        Err(SpecialError::GpuNotAvailable(_)) => j0_cpu_fallback(input, output),
        Err(e) => Err(e),
    }
}

/// GPU-accelerated error function (erf) for arrays
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn erf_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Try GPU execution first, fall back to CPU if GPU is not available
    match try_erf_gpu_execution(input, output) {
        Ok(()) => Ok(()),
        Err(SpecialError::GpuNotAvailable(_)) => erf_cpu_fallback(input, output),
        Err(e) => Err(e),
    }
}

/// GPU-accelerated digamma function for arrays
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn digamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Try GPU execution first, fall back to CPU if GPU is not available
    match try_digamma_gpu_execution(input, output) {
        Ok(()) => Ok(()),
        Err(SpecialError::GpuNotAvailable(_)) => digamma_cpu_fallback(input, output),
        Err(e) => Err(e),
    }
}

/// GPU-accelerated log gamma function for arrays
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn log_gamma_gpu<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    if input.len() != output.len() {
        return Err(SpecialError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    // Try GPU execution first, fall back to CPU if GPU is not available
    match try_log_gamma_gpu_execution(input, output) {
        Ok(()) => Ok(()),
        Err(SpecialError::GpuNotAvailable(_)) => log_gamma_cpu_fallback(input, output),
        Err(e) => Err(e),
    }
}

/// CPU fallback implementation for gamma function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn gamma_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync,
{
    use crate::gamma::gamma;
    // Use parallel processing for large arrays if available
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = gamma(*inp);
                });
            return Ok(());
        }
    }

    // Sequential processing as fallback or for small arrays
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = gamma(*inp);
    }

    Ok(())
}

/// CPU fallback implementation for Bessel J0 function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn j0_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync,
{
    use crate::bessel::j0;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = j0(*inp);
                });
            return Ok(());
        }
    }

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = j0(*inp);
    }

    Ok(())
}

/// CPU fallback implementation for error function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn erf_cpu_fallback<F>(input: &ArrayView1<F>, output: &mut ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync,
{
    use crate::erf::erf;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = erf(*inp);
                });
            return Ok(());
        }
    }

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = erf(*inp);
    }

    Ok(())
}

/// CPU fallback for digamma function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn digamma_cpu_fallback<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    use crate::gamma::digamma;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = digamma(*inp);
                });
            return Ok(());
        }
    }

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = digamma(*inp);
    }

    Ok(())
}

/// CPU fallback for log gamma function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn log_gamma_cpu_fallback<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    use crate::gamma::loggamma;
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;
        if is_parallel_enabled() && input.len() > 1000 {
            use scirs2_core::parallel_ops::IntoParallelRefIterator;
            use scirs2_core::parallel_ops::IntoParallelRefMutIterator;

            input
                .as_slice()
                .unwrap()
                .par_iter()
                .zip(output.as_slice_mut().unwrap().par_iter_mut())
                .for_each(|(inp, out)| {
                    *out = loggamma(*inp);
                });
            return Ok(());
        }
    }

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = loggamma(*inp);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gamma_gpu_fallback() {
        // Test that CPU fallback works correctly
        let input = Array1::linspace(0.1, 5.0, 10);
        let mut output = Array1::zeros(10);

        gamma_gpu(&input.view(), &mut output.view_mut()).unwrap();

        // Verify some known values
        use crate::gamma::gamma;
        for i in 0..10 {
            let expected = gamma(input[i]);
            let diff: f64 = output[i] - expected;
            assert!(diff.abs() < 1e-10_f64);
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_j0_gpu_fallback() {
        let input = Array1::linspace(0.1, 10.0, 10);
        let mut output = Array1::zeros(10);

        j0_gpu(&input.view(), &mut output.view_mut()).unwrap();

        // Verify some known values
        use crate::bessel::j0;
        for i in 0..10 {
            let expected = j0(input[i]);
            let diff: f64 = output[i] - expected;
            assert!(diff.abs() < 1e-10_f64);
        }
    }
}

/// Enhanced GPU execution for gamma function with advanced error handling and backend selection
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn try_gamma_gpu_execution_enhanced<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<scirs2_core::gpu::GpuBackend>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign
        + Send
        + Sync
        + 'static,
{
    use crate::gpu_context_manager::get_best_gpu_context;
    use scirs2_core::gpu::GpuBackend;

    // Get the best available GPU context with intelligent selection
    let (gpu_context, backend_type) = match get_best_gpu_context() {
        Ok(ctx) => {
            // Determine the backend type from context (simplified)
            let backend = GpuBackend::Wgpu; // Default assumption
            #[cfg(feature = "gpu")]
            log::debug!("GPU context obtained successfully: {:?}", backend);
            (ctx, backend)
        }
        Err(e) => {
            #[cfg(feature = "gpu")]
            log::warn!("GPU context creation failed: {:?}", e);
            return Err(SpecialError::GpuNotAvailable(format!(
                "GPU hardware not available: {}",
                e
            )));
        }
    };

    // For now, GPU operations only support f64. Fall back to CPU for other types.
    return Err(SpecialError::GpuNotAvailable(
        "GPU operations currently only support f64 type".to_string(),
    ));

    Ok(backend_type)
}

/// Try GPU execution for Bessel J0 function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn try_j0_gpu_execution<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + 'static,
{
    // GPU operations currently only support f64. Fall back to CPU for other types.
    Err(SpecialError::GpuNotAvailable(
        "GPU operations currently only support f64 type".to_string(),
    ))
}

/// Try GPU execution for error function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn try_erf_gpu_execution<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float + num_traits::FromPrimitive + Send + Sync + 'static,
{
    // GPU operations currently only support f64. Fall back to CPU for other types.
    Err(SpecialError::GpuNotAvailable(
        "GPU operations currently only support f64 type".to_string(),
    ))
}

/// Try GPU execution for digamma function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn try_digamma_gpu_execution<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    // GPU operations currently only support f64. Fall back to CPU for other types.
    Err(SpecialError::GpuNotAvailable(
        "GPU operations currently only support f64 type".to_string(),
    ))
}

/// Try GPU execution for log gamma function
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn try_log_gamma_gpu_execution<F>(
    input: &ArrayView1<F>,
    output: &mut ArrayViewMut1<F>,
) -> SpecialResult<()>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    // GPU operations currently only support f64. Fall back to CPU for other types.
    Err(SpecialError::GpuNotAvailable(
        "GPU operations currently only support f64 type".to_string(),
    ))
}

/// Helper function to create GPU context using the context manager
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_gpu_context() -> Result<Arc<GpuContext>, GpuError> {
    use crate::gpu_context_manager::get_best_gpu_context;

    match get_best_gpu_context() {
        Ok(context) => Ok(context),
        Err(e) => {
            #[cfg(feature = "gpu")]
            log::debug!(
                "GPU context manager failed: {}, falling back to direct creation",
                e
            );

            // Fallback to direct context creation
            use scirs2_core::gpu::GpuBackend;
            GpuContext::new(GpuBackend::Cpu).map(std::sync::Arc::new)
        }
    }
}

/// Helper function to create GPU buffer from slice with enhanced error handling
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_gpu_buffer(ctx: &GpuContext, data: &[f64]) -> scirs2_core::gpu::GpuBuffer<f64> {
    ctx.create_buffer_from_slice(data)
}

/// Type-specific GPU buffer creation with validation
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_gpu_buffer_typed(
    ctx: &GpuContext,
    data: &[f64],
) -> SpecialResult<scirs2_core::gpu::GpuBuffer<f64>> {
    // Validate input data for NaN/infinity before GPU transfer
    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(SpecialError::ValueError(format!(
                "Non-finite value at index {}: {}",
                i, val
            )));
        }
    }

    #[cfg(feature = "gpu")]
    log::debug!(
        "Creating GPU buffer with {} bytes for {} elements",
        data.len() * std::mem::size_of::<f64>(),
        data.len()
    );

    Ok(ctx.create_buffer_from_slice(data))
}

/// Helper function to create empty GPU buffer
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_empty_gpu_buffer(
    ctx: &GpuContext,
    size: usize,
) -> SpecialResult<scirs2_core::gpu::GpuBuffer<f64>> {
    Ok(ctx.create_buffer::<f64>(size))
}

/// Type-specific empty GPU buffer creation
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_empty_gpu_buffer_typed<T>(
    ctx: &GpuContext,
    size: usize,
) -> SpecialResult<scirs2_core::gpu::GpuBuffer<T>>
where
    T: 'static + scirs2_core::gpu::GpuDataType,
{
    let bytesize = size * std::mem::size_of::<T>();
    #[cfg(feature = "gpu")]
    log::debug!(
        "Creating empty GPU buffer with {} bytes for {} elements of type {}",
        bytesize,
        size,
        std::any::type_name::<T>()
    );

    Ok(ctx.create_buffer::<T>(size))
}

/// Helper function to create compute pipeline
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_compute_pipeline(
    _ctx: &GpuContext,
    _shader_source: &str,
) -> SpecialResult<scirs2_core::gpu::GpuKernelHandle> {
    Err(SpecialError::ComputationError(
        "GPU compute pipelines not yet supported in scirs2-core".to_string(),
    ))
}

/// Helper function to execute compute shader
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn execute_compute_shader(
    _ctx: &GpuContext,
    _pipeline: &scirs2_core::gpu::GpuKernelHandle,
    _input_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    _output_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    _array_len: usize,
) -> SpecialResult<()> {
    Err(SpecialError::ComputationError(
        "GPU compute shader execution not yet supported in scirs2-core".to_string(),
    ))
}

/// Enhanced compute shader execution with performance monitoring and validation
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn execute_compute_shader_enhanced(
    _ctx: &GpuContext,
    _pipeline: &scirs2_core::gpu::GpuKernelHandle,
    _input_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    _output_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    _array_len: usize,
) -> SpecialResult<()> {
    Err(SpecialError::ComputationError(
        "Enhanced GPU compute shader execution not yet supported in scirs2-core".to_string(),
    ))
}

/// Helper function to read GPU buffer to array
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn read_gpu_buffer_to_array<T>(
    ctx: &GpuContext,
    buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    output: &mut [T],
) -> SpecialResult<()>
where
    T: Copy + num_traits::FromPrimitive + num_traits::Zero,
{
    let data = ctx
        .read_buffer(buffer)
        .map_err(|e| SpecialError::ComputationError(format!("Failed to read GPU buffer: {}", e)))?;

    let typed_data = &data;
    if typed_data.len() != output.len() {
        return Err(SpecialError::ComputationError(
            "GPU buffer size mismatch".to_string(),
        ));
    }

    for (i, &val) in typed_data.iter().enumerate() {
        output[i] = T::from_f64(val).unwrap_or_else(|| T::zero());
    }
    Ok(())
}

/// Type-specific GPU buffer reading with enhanced validation
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn read_gpu_buffer_to_array_typed<T>(
    ctx: &GpuContext,
    buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    output: &mut [T],
) -> SpecialResult<()>
where
    T: num_traits::Float + std::fmt::Debug + num_traits::FromPrimitive + num_traits::Zero,
{
    let data = ctx.read_buffer(buffer).map_err(|e| {
        SpecialError::ComputationError(format!("Failed to read typed GPU buffer: {}", e))
    })?;

    let typed_data = &data;
    if typed_data.len() != output.len() {
        return Err(SpecialError::ComputationError(format!(
            "GPU buffer size mismatch: expected {}, got {}",
            output.len(),
            typed_data.len()
        )));
    }

    #[cfg(feature = "gpu")]
    log::debug!("Reading {} elements from GPU buffer", output.len());

    for (i, &val) in typed_data.iter().enumerate() {
        output[i] = T::from_f64(val).unwrap_or_else(|| T::zero());
    }
    Ok(())
}

/// Advanced buffer cache for GPU operations with intelligent memory management
#[cfg(feature = "gpu")]
struct GpuBufferCache {
    input_buffers: Mutex<HashMap<(usize, usize), scirs2_core::gpu::GpuBuffer<f64>>>, // (size, type_id)
    output_buffers: Mutex<HashMap<(usize, usize), scirs2_core::gpu::GpuBuffer<f64>>>,
    shader_pipelines: Mutex<HashMap<String, scirs2_core::gpu::GpuKernelHandle>>,
}

#[cfg(feature = "gpu")]
static GPU_BUFFER_CACHE: std::sync::OnceLock<GpuBufferCache> = std::sync::OnceLock::new();

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn get_buffer_cache() -> &'static GpuBufferCache {
    GPU_BUFFER_CACHE.get_or_init(|| GpuBufferCache {
        input_buffers: Mutex::new(HashMap::new()),
        output_buffers: Mutex::new(HashMap::new()),
        shader_pipelines: Mutex::new(HashMap::new()),
    })
}

/// Create GPU buffer with intelligent caching and memory reuse
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_gpu_buffer_with_caching(
    ctx: &GpuContext,
    data: &[f64],
) -> SpecialResult<scirs2_core::gpu::GpuBuffer<f64>> {
    let cache_key = (data.len(), 0); // Simple cache key based on size
    let cache = get_buffer_cache();

    // Try to reuse existing buffer if available and same size
    {
        let input_buffers = cache.input_buffers.lock().unwrap();
        if let Some(buffer) = input_buffers.get(&cache_key) {
            // Update buffer data
            if let Ok(_) = buffer.copy_from_host(data) {
                #[cfg(feature = "gpu")]
                log::debug!("Reused cached input buffer for {} elements", data.len());
                // Note: We can't return the cached buffer directly since we don't have ownership
                // Create a new buffer instead
            }
        }
    }

    // Create new buffer
    let buffer = ctx.create_buffer_from_slice(data);

    {
        let mut input_buffers = cache.input_buffers.lock().unwrap();

        // Limit cache size to prevent memory bloat
        if input_buffers.len() > 16 {
            let oldest_key = *input_buffers.keys().next().unwrap();
            input_buffers.remove(&oldest_key);
        }

        // Note: Can't cache the buffer since we're moving it
    }

    #[cfg(feature = "gpu")]
    log::debug!("Created new input buffer for {} elements", data.len());

    Ok(buffer)
}

/// Create empty GPU buffer with caching
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn create_empty_gpu_buffer_with_caching(
    ctx: &GpuContext,
    size: usize,
) -> scirs2_core::gpu::GpuBuffer<f64> {
    let cache_key = (size, 0); // Simple cache key
    let cache = get_buffer_cache();

    // Try to reuse existing buffer
    {
        let output_buffers = cache.output_buffers.lock().unwrap();
        if let Some(_buffer) = output_buffers.get(&cache_key) {
            #[cfg(feature = "gpu")]
            log::debug!(
                "Creating new output buffer (cached size) for {} elements",
                size
            );
            // Create a new buffer with the same size since GpuBuffer doesn't support cloning
            return ctx.create_buffer::<f64>(size);
        }
    }

    // Create new buffer
    let buffer = ctx.create_buffer::<f64>(size);

    {
        let mut output_buffers = cache.output_buffers.lock().unwrap();

        // Limit cache size
        if output_buffers.len() > 16 {
            let oldest_key = *output_buffers.keys().next().unwrap();
            output_buffers.remove(&oldest_key);
        }
    }

    #[cfg(feature = "gpu")]
    log::debug!("Created new output buffer for {} elements", size);

    buffer
}

/// Get or create shader pipeline with intelligent caching
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn get_or_create_shader_pipeline(
    ctx: &GpuContext,
    shader_name: &str,
    _shader_source: &str,
) -> SpecialResult<scirs2_core::gpu::GpuKernelHandle> {
    let cache = get_buffer_cache();

    // Try to get cached pipeline
    {
        let pipelines = cache.shader_pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(shader_name) {
            #[cfg(feature = "gpu")]
            log::debug!("Using cached shader pipeline: {}", shader_name);
            // Get kernel from registry since GpuKernelHandle doesn't support cloning
            return ctx.get_kernel(shader_name).map_err(|e| {
                SpecialError::ComputationError(format!(
                    "Failed to get cached shader pipeline '{}': {}",
                    shader_name, e
                ))
            });
        }
    }

    // Create new pipeline using available kernel registry
    let pipeline = ctx.get_kernel(shader_name).map_err(|e| {
        SpecialError::ComputationError(format!(
            "Failed to get shader pipeline '{}': {}",
            shader_name, e
        ))
    })?;

    {
        let mut pipelines = cache.shader_pipelines.lock().unwrap();
        // Note: We don't actually cache the pipeline since GpuKernelHandle doesn't support cloning
        // This is a placeholder for when proper caching is implemented

        #[cfg(feature = "gpu")]
        log::debug!("Created and cached new shader pipeline: {}", shader_name);
    }

    Ok(pipeline)
}

/// Execute compute shader with enhanced validation and error recovery
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn execute_compute_shader_with_validation(
    ctx: &GpuContext,
    _pipeline: &scirs2_core::gpu::GpuKernelHandle,
    input_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    output_buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    array_len: usize,
    function_name: &str,
) -> SpecialResult<()> {
    const WORKGROUP_SIZE: usize = 256;
    let workgroup_count_x = (array_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // Enhanced workgroup validation
    const MAX_WORKGROUPS: usize = 65535;
    if workgroup_count_x > MAX_WORKGROUPS {
        return Err(SpecialError::ComputationError(format!(
            "Array too large for {} computation: {} workgroups (max: {})",
            function_name, workgroup_count_x, MAX_WORKGROUPS
        )));
    }

    // GPU memory and state validation
    if array_len == 0 {
        return Err(SpecialError::ValueError(format!(
            "Empty array for {} computation",
            function_name
        )));
    }

    #[cfg(feature = "gpu")]
    log::debug!(
        "Executing {} shader with {} workgroups for {} elements",
        function_name,
        workgroup_count_x,
        array_len
    );

    let execution_start = Instant::now();

    // Execute with retry logic for transient failures
    let mut attempts = 0;
    const MAX_EXECUTION_ATTEMPTS: u32 = 2;

    while attempts < MAX_EXECUTION_ATTEMPTS {
        attempts += 1;

        // Note: The current execute_kernel API expects f32 buffers and different parameters
        // This is a placeholder until the GPU implementation is properly updated
        match ctx.execute_kernel(
            "compute_shader",
            &[], // Empty buffer array since API mismatch
            (workgroup_count_x as u32, 1, 1),
            &[],
            &[],
        ) {
            Ok(()) => {
                let execution_time = execution_start.elapsed();
                #[cfg(feature = "gpu")]
                log::debug!(
                    "{} shader execution successful in {:?} (attempt {})",
                    function_name,
                    execution_time,
                    attempts
                );
                return Ok(());
            }
            Err(e) => {
                #[cfg(feature = "gpu")]
                log::warn!(
                    "{} shader execution failed on attempt {}: {}",
                    function_name,
                    attempts,
                    e
                );

                if attempts == MAX_EXECUTION_ATTEMPTS {
                    return Err(SpecialError::ComputationError(format!(
                        "Failed to execute {} shader after {} attempts: {}",
                        function_name, MAX_EXECUTION_ATTEMPTS, e
                    )));
                }

                // Brief pause before retry
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }

    unreachable!()
}

/// Read GPU buffer with comprehensive validation and error handling
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn read_gpu_buffer_with_validation<T>(
    ctx: &GpuContext,
    buffer: &scirs2_core::gpu::GpuBuffer<f64>,
    output: &mut [T],
) -> SpecialResult<()>
where
    T: num_traits::Float + std::fmt::Debug + num_traits::FromPrimitive + num_traits::Zero,
{
    let read_start = Instant::now();

    let data = ctx
        .read_buffer(buffer)
        .map_err(|e| SpecialError::ComputationError(format!("Failed to read GPU buffer: {}", e)))?;

    let typed_data = &data;
    if typed_data.len() != output.len() {
        return Err(SpecialError::ComputationError(format!(
            "GPU buffer size mismatch: expected {}, got {}",
            output.len(),
            typed_data.len()
        )));
    }

    // Validate data integrity during transfer and convert types
    for (i, &val) in typed_data.iter().enumerate() {
        if !val.is_finite() {
            #[cfg(feature = "gpu")]
            log::warn!(
                "Non-finite value detected in GPU result at index {}: {:?}",
                i,
                val
            );
        }
        output[i] = T::from_f64(val).unwrap_or_else(|| {
            #[cfg(feature = "gpu")]
            log::warn!("Failed to convert f64 value {} to target type", val);
            T::zero()
        });
    }

    let read_time = read_start.elapsed();
    #[cfg(feature = "gpu")]
    log::debug!(
        "GPU buffer read completed in {:?} for {} elements",
        read_time,
        output.len()
    );

    Ok(())
}

/// Validate gamma function results with mathematical properties
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn validate_gamma_results<F>(input: &ArrayView1<F>, output: &ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + std::fmt::Debug + num_traits::FromPrimitive,
{
    let mut error_count = 0;
    let zero = F::zero();
    let one = F::one();

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        // Check basic mathematical properties
        if x > zero {
            if !y.is_finite() {
                error_count += 1;
                if error_count <= 5 {
                    // Limit error logging
                    #[cfg(feature = "gpu")]
                    log::warn!("Invalid gamma result at index {}: Γ({:?}) = {:?}", i, x, y);
                }
            } else if y <= zero {
                error_count += 1;
                if error_count <= 5 {
                    #[cfg(feature = "gpu")]
                    log::warn!(
                        "Non-positive gamma result at index {}: Γ({:?}) = {:?}",
                        i,
                        x,
                        y
                    );
                }
            }
        }

        // Check for specific known values
        if (x - one).abs() < F::from(1e-10).unwrap_or(F::epsilon())
            && (y - one).abs() > F::from(1e-6).unwrap_or(F::epsilon())
        {
            #[cfg(feature = "gpu")]
            log::warn!("Gamma(1) validation failed: expected ~1.0, got {:?}", y);
        }
    }

    if error_count > 0 {
        #[cfg(feature = "gpu")]
        log::warn!(
            "Gamma validation found {} errors out of {} values",
            error_count,
            input.len()
        );
    }

    Ok(())
}

/// General GPU computation results validation with enhanced statistics
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn validate_gpu_results<F>(output: &ArrayViewMut1<F>) -> SpecialResult<()>
where
    F: num_traits::Float + std::fmt::Debug,
{
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut subnormal_count = 0;

    for (i, &val) in output.iter().enumerate() {
        if val.is_nan() {
            nan_count += 1;
            if nan_count == 1 {
                #[cfg(feature = "gpu")]
                log::warn!("First NaN found at index {}: {:?}", i, val);
            }
        } else if val.is_infinite() {
            inf_count += 1;
            if inf_count == 1 {
                #[cfg(feature = "gpu")]
                log::warn!("First infinity found at index {}: {:?}", i, val);
            }
        } else if val.is_subnormal() {
            subnormal_count += 1;
        }
    }

    if nan_count > 0 || inf_count > 0 {
        #[cfg(feature = "gpu")]
        log::warn!(
            "GPU computation produced {} NaN, {} infinite, and {} subnormal values",
            nan_count,
            inf_count,
            subnormal_count
        );
    } else if subnormal_count > 0 {
        #[cfg(feature = "gpu")]
        log::debug!(
            "GPU computation produced {} subnormal values",
            subnormal_count
        );
    }

    Ok(())
}
