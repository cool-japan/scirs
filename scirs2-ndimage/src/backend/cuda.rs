//! CUDA backend implementation for GPU acceleration
//!
//! This module provides CUDA-specific implementations for GPU-accelerated
//! image processing operations.
//!
//! # Pure-Rust / zero build-time CUDA SDK dependency (COOLJAPAN policy)
//!
//! No CUDA SDK, headers, or link stubs are required to **build** this crate.
//! The CUDA driver, device memory, and kernel-launch operations are provided by
//! the pure-Rust `oxicuda` ecosystem (`oxicuda-driver`, `oxicuda-memory`), which
//! loads `libcuda` at **runtime** via `dlopen`. There is no `#[link]`
//! attribute and no `build.rs`.
//!
//! CUDA-C kernel JIT compilation (the historical NVRTC path) is provided by the
//! pure-Rust `oxicuda-nvrtc` crate: [`compile_kernel`](CudaContext::compile_kernel)
//! resolves the NVRTC runtime library (`libnvrtc`) lazily at process runtime via
//! `dlopen` — zero build-time CUDA SDK, no `#[link]` attribute, no `build.rs`,
//! and no `-lnvrtc`. When neither a CUDA driver nor `libnvrtc` is present (as on
//! a CUDA-less host) every entry point degrades to a typed [`NdimageError`]
//! rather than panicking, including in every `Drop` (device resources are
//! RAII-managed by `oxicuda` and log rather than panic on failure).

use crate::backend::kernels::{GpuBuffer, GpuKernelExecutor, KernelInfo};
use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use oxicuda_driver::ffi::{CUdeviceptr, CUstream};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::{Context, Device, Function, Module};
use oxicuda_memory::DeviceBuffer;

/// GPU context trait for different GPU backends
pub trait GpuContext: Send + Sync {
    fn name(&self) -> &str;
    fn device_count(&self) -> usize;
    fn current_device(&self) -> usize;
    fn memory_info(&self) -> (usize, usize); // (used, total)
}

/// Map an `oxicuda` CUDA driver error into an [`NdimageError`].
fn cuda_err(context: &str, error: oxicuda_driver::CudaError) -> NdimageError {
    NdimageError::ComputationError(format!("{context}: {error}"))
}

/// Map an [`oxicuda_nvrtc::NvrtcError`] into an [`NdimageError`], preserving the
/// backend's historical error semantics: a missing NVRTC runtime is a typed
/// "not implemented on this host" condition, a CUDA-C compilation failure
/// carries the full NVRTC build log, and every other failure is reported as a
/// computation error.
fn nvrtc_err(error: oxicuda_nvrtc::NvrtcError) -> NdimageError {
    match error {
        oxicuda_nvrtc::NvrtcError::Unavailable { .. } => NdimageError::NotImplementedError(
            "CUDA kernel JIT compilation requires the NVRTC runtime library (libnvrtc), \
             which is not available on this system. oxicuda-ptx provides pure-Rust PTX \
             generation but not CUDA-C runtime compilation."
                .into(),
        ),
        oxicuda_nvrtc::NvrtcError::Compilation { code, msg, log } => {
            NdimageError::ComputationError(format!(
                "CUDA kernel compilation failed (nvrtc error {code}: {msg}):\n{log}"
            ))
        }
        other => {
            NdimageError::ComputationError(format!("CUDA kernel JIT compilation failed: {other}"))
        }
    }
}

/// CUDA-specific GPU buffer implementation.
///
/// Backed by an `oxicuda-memory` [`DeviceBuffer<u8>`], which owns its device
/// allocation and frees it on drop (logging rather than panicking on failure).
pub struct CudaBuffer<T>
where
    T: Send + Sync,
{
    buffer: DeviceBuffer<u8>,
    size: usize,
    phantom: PhantomData<T>,
}

impl<T: Send + Sync + 'static> CudaBuffer<T> {
    pub fn new(size: usize) -> NdimageResult<Self> {
        let byte_size = size
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| NdimageError::ComputationError("CUDA buffer size overflow".into()))?;

        let buffer = if byte_size == 0 {
            // Zero-length placeholder view: never dereferenced and non-owning,
            // so its drop is a no-op and requires no CUDA driver.
            // SAFETY: length is zero and the pointer is never dereferenced.
            unsafe { DeviceBuffer::<u8>::from_raw(0, 0) }
        } else {
            DeviceBuffer::<u8>::alloc(byte_size).map_err(|e| cuda_err("CUDA malloc failed", e))?
        };

        Ok(Self {
            buffer,
            size,
            phantom: PhantomData,
        })
    }

    pub fn from_host_data(data: &[T]) -> NdimageResult<Self> {
        let mut buffer = Self::new(data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }

    /// Raw device pointer of this buffer, for kernel-launch argument lists.
    fn device_ptr(&self) -> CUdeviceptr {
        self.buffer.as_device_ptr()
    }
}

impl<T: Send + Sync + 'static> GpuBuffer<T> for CudaBuffer<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_from_host(&mut self, data: &[T]) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }
        if self.size == 0 {
            return Ok(());
        }
        // Reinterpret the typed host slice as raw bytes for the device copy,
        // exactly as a `cudaMemcpy` of the underlying storage would.
        // SAFETY: `data` is valid for `size_of_val(data)` bytes and the device
        // buffer holds exactly that many bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.buffer
            .copy_from_host(bytes)
            .map_err(|e| cuda_err("CUDA memcpy failed", e))
    }

    fn copy_to_host(&self, data: &mut [T]) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }
        if self.size == 0 {
            return Ok(());
        }
        let byte_len = std::mem::size_of_val(data);
        // SAFETY: `data` is valid for `byte_len` bytes and the device buffer
        // holds exactly that many bytes.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, byte_len) };
        self.buffer
            .copy_to_host(bytes)
            .map_err(|e| cuda_err("CUDA memcpy failed", e))
    }
}

/// CUDA context implementation.
///
/// Owns an `oxicuda-driver` [`Context`] (kept alive for the lifetime of the
/// backend) that binds device `device_id` on the creating thread.
pub struct CudaContext {
    device_id: i32,
    compute_capability: (i32, i32),
    max_threads_per_block: i32,
    max_shared_memory: usize,
    context: Arc<Context>,
}

impl CudaContext {
    pub fn new(deviceid: Option<usize>) -> NdimageResult<Self> {
        let device_id = deviceid.unwrap_or(0) as i32;

        oxicuda_driver::init().map_err(|e| cuda_err("Failed to initialize CUDA driver", e))?;

        let device_count =
            Device::count().map_err(|e| cuda_err("Failed to get CUDA device count", e))?;
        if device_id >= device_count {
            return Err(NdimageError::InvalidInput(format!(
                "CUDA device {device_id} not found. Only {device_count} devices available"
            )));
        }

        let device =
            Device::get(device_id).map_err(|e| cuda_err("Failed to get CUDA device", e))?;
        let context = Arc::new(
            Context::new(&device).map_err(|e| cuda_err("Failed to create CUDA context", e))?,
        );

        // Query real device properties; fall back to sensible defaults if a
        // particular attribute cannot be read (never panics).
        let compute_capability = device.compute_capability().unwrap_or((7, 5));
        let max_threads_per_block = device.max_threads_per_block().unwrap_or(1024);
        let max_shared_memory = device
            .max_shared_memory_per_block()
            .map(|v| v as usize)
            .unwrap_or(49_152);

        Ok(Self {
            device_id,
            compute_capability,
            max_threads_per_block,
            max_shared_memory,
            context,
        })
    }

    /// Compute capability `(major, minor)` of the bound device.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Maximum threads per block reported by the bound device.
    pub fn max_threads_per_block(&self) -> i32 {
        self.max_threads_per_block
    }

    /// Maximum shared memory per block (bytes) reported by the bound device.
    pub fn max_shared_memory(&self) -> usize {
        self.max_shared_memory
    }

    /// Get optimal kernel compilation options based on compute capability.
    ///
    /// Plain `String`s are sufficient here: `oxicuda-nvrtc` performs the C-ABI
    /// conversion itself (rejecting interior NUL bytes before any FFI call).
    fn get_compilation_options(&self) -> Vec<String> {
        let mut options = vec![
            format!(
                "--gpu-architecture=compute_{}{}",
                self.compute_capability.0, self.compute_capability.1
            ),
            "--fmad=true".to_string(),
            "--use_fast_math".to_string(),
            "--restrict".to_string(),
        ];

        // Add optimization options based on compute capability
        if self.compute_capability >= (7, 0) {
            options.push("--extra-device-vectorization".to_string());
        }

        if self.compute_capability >= (8, 0) {
            options.push("--allow-unsupported-compiler".to_string());
        }

        options
    }

    pub fn compile_kernel(&self, source: &str, kernelname: &str) -> NdimageResult<CudaKernel> {
        // Check cache first
        {
            let cache = KERNEL_CACHE.lock().map_err(|_| {
                NdimageError::ComputationError("Failed to acquire kernel cache lock".into())
            })?;
            if let Some(kernel) = cache.get(kernelname) {
                return Ok(kernel.clone());
            }
        }

        // Convert OpenCL-style kernel to CUDA
        let cuda_source = convert_opencl_to_cuda(source);

        // JIT-compile CUDA-C to PTX through `oxicuda-nvrtc` (libnvrtc is
        // resolved lazily at runtime; no build-time dependency). A host without
        // the NVRTC runtime yields a typed `NotImplementedError` via
        // [`nvrtc_err`], never a panic.
        let options = self.get_compilation_options();
        let option_refs: Vec<&str> = options.iter().map(String::as_str).collect();
        let ptx = oxicuda_nvrtc::compile_to_ptx(&cuda_source, kernelname, &option_refs)
            .map_err(nvrtc_err)?;

        // Load the PTX module and look up the kernel through oxicuda-driver.
        // `Ptx::as_str` is the UTF-8 PTX text without its trailing NUL, exactly
        // what `Module::from_ptx` expects.
        let module = Module::from_ptx(ptx.as_str())
            .map_err(|e| cuda_err("Failed to load CUDA module", e))?;
        let function = module
            .get_function(kernelname)
            .map_err(|e| cuda_err(&format!("Failed to get CUDA function '{kernelname}'"), e))?;

        let kernel = CudaKernel {
            name: kernelname.to_string(),
            module: Arc::new(module),
            function,
            ptx_code: ptx.as_str().as_bytes().to_vec(),
        };

        // Cache the compiled kernel
        {
            let mut cache = KERNEL_CACHE.lock().map_err(|_| {
                NdimageError::ComputationError(
                    "Failed to acquire kernel cache lock for insertion".into(),
                )
            })?;
            cache.insert(kernelname.to_string(), kernel.clone());
        }

        Ok(kernel)
    }
}

impl GpuContext for CudaContext {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_count(&self) -> usize {
        Device::count().map(|c| c as usize).unwrap_or(0)
    }

    fn current_device(&self) -> usize {
        self.device_id as usize
    }

    fn memory_info(&self) -> (usize, usize) {
        oxicuda_memory::memory_info()
            .map(|m| (m.used(), m.total))
            .unwrap_or((0, 0))
    }
}

/// CUDA kernel handle.
///
/// Holds the owning [`Module`] via `Arc` so the module stays loaded for as long
/// as any clone of this kernel (including the cache entry) is alive; the raw
/// [`Function`] handle is a lightweight copy that borrows from that module.
#[derive(Clone)]
pub struct CudaKernel {
    name: String,
    module: Arc<Module>,
    function: Function,
    ptx_code: Vec<u8>,
}

// Kernel cache to avoid recompilation
lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Arc<Mutex<HashMap<String, CudaKernel>>> = Arc::new(Mutex::new(HashMap::new()));
}

/// CUDA kernel executor implementation.
///
/// Owns an `oxicuda-driver` [`Stream`](oxicuda_driver::Stream) bound to the
/// context; the stream is destroyed via RAII on drop (logging, never panicking).
pub struct CudaExecutor {
    context: Arc<CudaContext>,
    stream: oxicuda_driver::Stream,
}

impl CudaExecutor {
    pub fn new(context: Arc<CudaContext>) -> NdimageResult<Self> {
        let stream = oxicuda_driver::Stream::new(&context.context)
            .map_err(|e| cuda_err("Failed to create CUDA stream", e))?;
        Ok(Self { context, stream })
    }
}

impl<T> GpuKernelExecutor<T> for CudaExecutor
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    fn execute_kernel(
        &self,
        kernel: &KernelInfo,
        inputs: &[&dyn GpuBuffer<T>],
        outputs: &[&mut dyn GpuBuffer<T>],
        work_size: &[usize],
        params: &[T],
    ) -> NdimageResult<()> {
        // Compile kernel
        let cuda_kernel = self
            .context
            .compile_kernel(&kernel.source, &kernel.entry_point)?;

        // Calculate grid and block dimensions
        let (grid_dim, block_dim) = calculate_launch_config(work_size, kernel.work_dimensions);

        // Collect device pointers into a stable buffer that outlives the launch,
        // then build the pointer-to-argument list the driver expects.
        let mut dev_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(inputs.len() + outputs.len());
        for input in inputs {
            let cuda_buf = input
                .as_any()
                .downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            dev_ptrs.push(cuda_buf.device_ptr());
        }
        for output in outputs {
            let cuda_buf = output
                .as_any()
                .downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            dev_ptrs.push(cuda_buf.device_ptr());
        }

        let mut param_storage: Vec<T> = params.to_vec();

        // Each kernel argument entry is a pointer to the value being passed:
        // a `CUdeviceptr` for buffers, and a `T` for scalar parameters.
        let mut kernel_args: Vec<*mut c_void> =
            Vec::with_capacity(dev_ptrs.len() + param_storage.len());
        for dp in &mut dev_ptrs {
            kernel_args.push(dp as *mut CUdeviceptr as *mut c_void);
        }
        for param in &mut param_storage {
            kernel_args.push(param as *mut T as *mut c_void);
        }

        let api = try_driver().map_err(|e| cuda_err("CUDA driver unavailable", e))?;

        // SAFETY: `cuda_kernel.function` belongs to a live module (held via
        // `Arc` in `cuda_kernel`), `self.stream` is a live stream, and
        // `kernel_args` points to values that outlive this call.
        let launch_rc = unsafe {
            (api.cu_launch_kernel)(
                cuda_kernel.function.raw(),
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                0, // shared memory
                self.stream.raw(),
                kernel_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        oxicuda_driver::check(launch_rc).map_err(|e| cuda_err("CUDA kernel launch failed", e))?;

        // Synchronize stream (surfaces any asynchronous kernel error).
        // SAFETY: `self.stream` is a live stream handle.
        let sync_rc = unsafe { (api.cu_stream_synchronize)(self.stream.raw()) };
        oxicuda_driver::check(sync_rc).map_err(|e| cuda_err("CUDA stream sync failed", e))?;

        Ok(())
    }
}

/// High-level CUDA operations
pub struct CudaOperations {
    context: Arc<CudaContext>,
    executor: CudaExecutor,
}

impl CudaOperations {
    pub fn new(deviceid: Option<usize>) -> NdimageResult<Self> {
        let context = Arc::new(CudaContext::new(deviceid)?);
        let executor = CudaExecutor::new(context.clone())?;

        Ok(Self { context, executor })
    }

    /// Access the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// GPU-accelerated Gaussian filter
    pub fn gaussian_filter_2d<T>(
        &self,
        input: &ArrayView2<T>,
        sigma: [T; 2],
    ) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_gaussian_filter_2d(input, sigma, &self.executor)
    }

    /// GPU-accelerated convolution
    pub fn convolve_2d<T>(
        &self,
        input: &ArrayView2<T>,
        kernel: &ArrayView2<T>,
    ) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_convolve_2d(input, kernel, &self.executor)
    }

    /// GPU-accelerated median filter
    pub fn median_filter_2d<T>(
        &self,
        input: &ArrayView2<T>,
        size: [usize; 2],
    ) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_median_filter_2d(input, size, &self.executor)
    }

    /// GPU-accelerated morphological erosion
    pub fn erosion_2d<T>(
        &self,
        input: &ArrayView2<T>,
        structure: &ArrayView2<bool>,
    ) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix2>>
    where
        T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    {
        crate::backend::kernels::gpu_erosion_2d(input, structure, &self.executor)
    }
}

/// Helper function to allocate GPU buffer
#[allow(dead_code)]
pub fn allocate_gpu_buffer<T>(data: &[T]) -> NdimageResult<Box<dyn GpuBuffer<T>>>
where
    T: Send + Sync + 'static,
{
    Ok(Box::new(CudaBuffer::from_host_data(data)?))
}

/// Helper function to allocate empty GPU buffer
#[allow(dead_code)]
pub fn allocate_gpu_buffer_empty<T>(size: usize) -> NdimageResult<Box<dyn GpuBuffer<T>>>
where
    T: Send + Sync + 'static,
{
    Ok(Box::new(CudaBuffer::<T>::new(size)?))
}

/// Advanced CUDA memory manager with buffer pooling.
///
/// Pools hold raw `oxicuda-driver` device pointers ([`CUdeviceptr`]); the
/// `*mut c_void` values in the public API are the same device addresses viewed
/// as opaque pointers.
pub struct CudaMemoryManager {
    buffer_pools: HashMap<usize, Vec<CUdeviceptr>>,
    total_allocated: usize,
    max_pool_size: usize,
}

impl CudaMemoryManager {
    pub fn new(_max_poolsize: usize) -> Self {
        Self {
            buffer_pools: HashMap::new(),
            total_allocated: 0,
            max_pool_size: _max_poolsize,
        }
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocate_buffer(&mut self, size: usize) -> NdimageResult<*mut c_void> {
        // Try to reuse a buffer from the pool
        if let Some(pool) = self.buffer_pools.get_mut(&size) {
            if let Some(dptr) = pool.pop() {
                return Ok(dptr as usize as *mut c_void);
            }
        }

        // Allocate a new buffer via the CUDA driver.
        let api = try_driver().map_err(|e| cuda_err("CUDA driver unavailable", e))?;
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: `dptr` is a valid out-pointer; `cu_mem_alloc_v2` writes a
        // device pointer on success.
        let rc = unsafe { (api.cu_mem_alloc_v2)(&mut dptr, size) };
        oxicuda_driver::check(rc).map_err(|e| cuda_err("CUDA malloc failed", e))?;

        self.total_allocated += size;
        Ok(dptr as usize as *mut c_void)
    }

    /// Return a buffer to the pool for reuse
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn deallocate_buffer(&mut self, ptr: *mut c_void, size: usize) -> NdimageResult<()> {
        let dptr = ptr as usize as CUdeviceptr;
        let pool = self.buffer_pools.entry(size).or_default();

        if pool.len() < self.max_pool_size {
            pool.push(dptr);
        } else {
            // Pool is full, actually free the memory
            let api = try_driver().map_err(|e| cuda_err("CUDA driver unavailable", e))?;
            // SAFETY: `dptr` was allocated by `cu_mem_alloc_v2` and not yet freed.
            let rc = unsafe { (api.cu_mem_free_v2)(dptr) };
            oxicuda_driver::check(rc).map_err(|e| cuda_err("CUDA free failed", e))?;
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> (usize, usize) {
        let pooled_memory: usize = self
            .buffer_pools
            .iter()
            .map(|(size, pool)| size * pool.len())
            .sum();
        (self.total_allocated, pooled_memory)
    }

    /// Clear all pools and free memory
    pub fn clear_pools(&mut self) -> NdimageResult<()> {
        // If the driver is unavailable there is nothing to free at the device
        // level; simply drop the tracked pointers.
        let api = try_driver().ok();
        for (size, pool) in self.buffer_pools.drain() {
            for dptr in pool {
                if let Some(api) = api {
                    // SAFETY: `dptr` was allocated by `cu_mem_alloc_v2`.
                    let rc = unsafe { (api.cu_mem_free_v2)(dptr) };
                    oxicuda_driver::check(rc)
                        .map_err(|e| cuda_err("CUDA free failed during pool clear", e))?;
                }
                self.total_allocated = self.total_allocated.saturating_sub(size);
            }
        }
        Ok(())
    }
}

impl Drop for CudaMemoryManager {
    fn drop(&mut self) {
        // Best effort cleanup - ignore errors during drop.
        let _ = self.clear_pools();
    }
}

/// Advanced CUDA execution context with profiling and optimization
pub struct AdvancedCudaExecutor {
    context: Arc<CudaContext>,
    stream: oxicuda_driver::Stream,
    memory_manager: Mutex<CudaMemoryManager>,
    execution_stats: Mutex<ExecutionStats>,
}

#[derive(Default)]
struct ExecutionStats {
    kernel_launches: u64,
    total_execution_time: f64,
    memory_transfers: u64,
    total_transfer_time: f64,
}

impl AdvancedCudaExecutor {
    pub fn new(context: Arc<CudaContext>) -> NdimageResult<Self> {
        let stream = oxicuda_driver::Stream::new(&context.context)
            .map_err(|e| cuda_err("Failed to create CUDA stream", e))?;

        Ok(Self {
            context,
            stream,
            memory_manager: Mutex::new(CudaMemoryManager::new(10)), // Pool up to 10 buffers per size
            execution_stats: Mutex::new(ExecutionStats::default()),
        })
    }

    /// Access the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Raw stream handle (as an opaque pointer) for async buffer transfers.
    pub fn stream(&self) -> *mut c_void {
        self.stream.raw().0
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> NdimageResult<(u64, f64, u64, f64)> {
        let stats = self
            .execution_stats
            .lock()
            .map_err(|_| NdimageError::ComputationError("Failed to acquire stats lock".into()))?;
        Ok((
            stats.kernel_launches,
            stats.total_execution_time,
            stats.memory_transfers,
            stats.total_transfer_time,
        ))
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> NdimageResult<(usize, usize)> {
        let memory_manager = self.memory_manager.lock().map_err(|_| {
            NdimageError::ComputationError("Failed to acquire memory manager lock".into())
        })?;
        Ok(memory_manager.get_memory_stats())
    }

    /// Allocate a managed buffer
    pub fn allocate_managed_buffer<T>(&self, size: usize) -> NdimageResult<CudaManagedBuffer<T>> {
        let mut memory_manager = self.memory_manager.lock().map_err(|_| {
            NdimageError::ComputationError("Failed to acquire memory manager lock".into())
        })?;

        let byte_size = size * std::mem::size_of::<T>();
        let device_ptr = memory_manager.allocate_buffer(byte_size)?;

        Ok(CudaManagedBuffer {
            device_ptr,
            size,
            byte_size,
            phantom: PhantomData,
        })
    }
}

/// CUDA buffer with managed lifecycle
pub struct CudaManagedBuffer<T> {
    device_ptr: *mut c_void,
    size: usize,
    byte_size: usize,
    phantom: PhantomData<T>,
}

impl<T> CudaManagedBuffer<T> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn copy_from_host_async(&self, data: &[T], stream: *mut c_void) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }

        let api = try_driver().map_err(|e| cuda_err("CUDA driver unavailable", e))?;
        let dptr = self.device_ptr as usize as CUdeviceptr;
        // SAFETY: `dptr` addresses `self.byte_size` device bytes, `data` is a
        // valid host slice of the same byte size, and `stream` is a CUDA stream
        // handle supplied by the caller.
        let rc = unsafe {
            (api.cu_memcpy_htod_async_v2)(
                dptr,
                data.as_ptr() as *const c_void,
                self.byte_size,
                CUstream(stream),
            )
        };
        oxicuda_driver::check(rc).map_err(|e| cuda_err("CUDA async memcpy failed", e))
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn copy_to_host_async(&self, data: &mut [T], stream: *mut c_void) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }

        let api = try_driver().map_err(|e| cuda_err("CUDA driver unavailable", e))?;
        let dptr = self.device_ptr as usize as CUdeviceptr;
        // SAFETY: `dptr` addresses `self.byte_size` device bytes, `data` is a
        // valid, writable host slice of the same byte size, and `stream` is a
        // CUDA stream handle supplied by the caller.
        let rc = unsafe {
            (api.cu_memcpy_dtoh_async_v2)(
                data.as_mut_ptr() as *mut c_void,
                dptr,
                self.byte_size,
                CUstream(stream),
            )
        };
        oxicuda_driver::check(rc).map_err(|e| cuda_err("CUDA async memcpy failed", e))
    }
}

/// Convert OpenCL-style kernel to CUDA syntax
#[allow(dead_code)]
fn convert_opencl_to_cuda(source: &str) -> String {
    let mut cuda_source = source.to_string();

    // Handle kernel declaration
    cuda_source = cuda_source.replace("__kernel", "extern \"C\" __global__");

    // Handle address space qualifiers
    cuda_source = cuda_source.replace("__global ", "");
    cuda_source = cuda_source.replace("__local", "__shared__");
    cuda_source = cuda_source.replace("__constant", "__constant__");

    // Handle work item functions
    cuda_source = cuda_source.replace("get_global_id(0)", "blockIdx.x * blockDim.x + threadIdx.x");
    cuda_source = cuda_source.replace("get_global_id(1)", "blockIdx.y * blockDim.y + threadIdx.y");
    cuda_source = cuda_source.replace("get_global_id(2)", "blockIdx.z * blockDim.z + threadIdx.z");

    cuda_source = cuda_source.replace("get_local_id(0)", "threadIdx.x");
    cuda_source = cuda_source.replace("get_local_id(1)", "threadIdx.y");
    cuda_source = cuda_source.replace("get_local_id(2)", "threadIdx.z");

    cuda_source = cuda_source.replace("get_group_id(0)", "blockIdx.x");
    cuda_source = cuda_source.replace("get_group_id(1)", "blockIdx.y");
    cuda_source = cuda_source.replace("get_group_id(2)", "blockIdx.z");

    cuda_source = cuda_source.replace("get_local_size(0)", "blockDim.x");
    cuda_source = cuda_source.replace("get_local_size(1)", "blockDim.y");
    cuda_source = cuda_source.replace("get_local_size(2)", "blockDim.z");

    cuda_source = cuda_source.replace("get_global_size(0)", "gridDim.x * blockDim.x");
    cuda_source = cuda_source.replace("get_global_size(1)", "gridDim.y * blockDim.y");
    cuda_source = cuda_source.replace("get_global_size(2)", "gridDim.z * blockDim.z");

    // Handle synchronization
    cuda_source = cuda_source.replace("barrier(CLK_LOCAL_MEM_FENCE)", "__syncthreads()");
    cuda_source = cuda_source.replace("barrier(CLK_GLOBAL_MEM_FENCE)", "__threadfence()");

    // Handle math functions - some have different names in CUDA
    cuda_source = cuda_source.replace("clamp(", "fminf(fmaxf(");
    cuda_source = cuda_source.replace("mix(", "lerp(");
    cuda_source = cuda_source.replace("mad(", "fmaf(");

    // Handle atomic operations
    cuda_source = cuda_source.replace("atomic_add(", "atomicAdd(");
    cuda_source = cuda_source.replace("atomic_sub(", "atomicSub(");
    cuda_source = cuda_source.replace("atomic_inc(", "atomicInc(");
    cuda_source = cuda_source.replace("atomic_dec(", "atomicDec(");
    cuda_source = cuda_source.replace("atomic_min(", "atomicMin(");
    cuda_source = cuda_source.replace("atomic_max(", "atomicMax(");
    cuda_source = cuda_source.replace("atomic_and(", "atomicAnd(");
    cuda_source = cuda_source.replace("atomic_or(", "atomicOr(");
    cuda_source = cuda_source.replace("atomic_xor(", "atomicXor(");

    // Add common CUDA includes if not present
    if !cuda_source.contains("#include") {
        cuda_source = format!(
            "#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n\n{cuda_source}"
        );
    }

    cuda_source
}

/// Calculate optimal grid and block dimensions for kernel launch
#[allow(dead_code)]
fn calculate_launch_config(
    work_size: &[usize],
    dimensions: usize,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    calculate_launch_config_advanced(work_size, dimensions, 1024, (65535, 65535, 65535))
}

/// Advanced launch configuration calculation with device constraints
#[allow(dead_code)]
fn calculate_launch_config_advanced(
    work_size: &[usize],
    dimensions: usize,
    max_threads_per_block: usize,
    max_grid_size: (u32, u32, u32),
) -> ((u32, u32, u32), (u32, u32, u32)) {
    // Determine optimal _block _size based on dimensionality and constraints
    let block_size = match dimensions {
        1 => {
            // For 1D, use power-of-2 _block sizes for better occupancy
            let optimal_size = if work_size[0] < 128 {
                64
            } else if work_size[0] < 512 {
                128
            } else if work_size[0] < 2048 {
                256
            } else {
                512
            };
            (optimal_size.min(max_threads_per_block), 1, 1)
        }
        2 => {
            // For 2D, balance between x and y dimensions
            let total_threads = max_threads_per_block.min(1024);
            let aspect_ratio = work_size[0] as f64 / work_size[1] as f64;

            let (bx, by) = if aspect_ratio > 2.0 {
                (32, total_threads / 32) // Wide images
            } else if aspect_ratio < 0.5 {
                (total_threads / 32, 32) // Tall images
            } else {
                // Square-ish images - use square blocks
                let sqrt_threads = (total_threads as f64).sqrt() as usize;
                let power_of_2 = 1 << (sqrt_threads as f64).log2().floor() as usize;
                (power_of_2, total_threads / power_of_2)
            };
            (bx, by, 1)
        }
        3 => {
            // For 3D, distribute threads more evenly
            let total_threads = max_threads_per_block.min(512); // Use fewer threads for 3D
            let cube_root = (total_threads as f64).powf(1.0 / 3.0) as usize;
            let optimal_dim = 1 << (cube_root as f64).log2().floor() as usize;
            let remaining = total_threads / (optimal_dim * optimal_dim);
            (optimal_dim, optimal_dim, remaining.max(1))
        }
        _ => (256, 1, 1), // Default fallback
    };

    // Calculate grid _size ensuring we don't exceed device limits
    let grid_size = match dimensions {
        1 => {
            let blocks =
                ((work_size[0] + block_size.0 - 1) / block_size.0).min(max_grid_size.0 as usize);
            (blocks as u32, 1, 1)
        }
        2 => {
            let blocks_x =
                ((work_size[0] + block_size.0 - 1) / block_size.0).min(max_grid_size.0 as usize);
            let blocks_y =
                ((work_size[1] + block_size.1 - 1) / block_size.1).min(max_grid_size.1 as usize);
            (blocks_x as u32, blocks_y as u32, 1)
        }
        3 => {
            let blocks_x =
                ((work_size[0] + block_size.0 - 1) / block_size.0).min(max_grid_size.0 as usize);
            let blocks_y =
                ((work_size[1] + block_size.1 - 1) / block_size.1).min(max_grid_size.1 as usize);
            let blocks_z =
                ((work_size[2] + block_size.2 - 1) / block_size.2).min(max_grid_size.2 as usize);
            (blocks_x as u32, blocks_y as u32, blocks_z as u32)
        }
        _ => {
            let blocks =
                ((work_size[0] + block_size.0 - 1) / block_size.0).min(max_grid_size.0 as usize);
            (blocks as u32, 1, 1)
        }
    };

    (
        grid_size,
        (
            block_size.0 as u32,
            block_size.1 as u32,
            block_size.2 as u32,
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires-gpu: requires a CUDA-capable device"]
    fn test_cudacontext_creation() {
        let context = CudaContext::new(None);
        assert!(context.is_ok());

        if let Ok(ctx) = context {
            assert_eq!(ctx.device_id, 0);
            assert!(ctx.device_count() > 0);
        }
    }

    #[test]
    #[ignore = "requires-gpu: requires a CUDA-capable device"]
    fn test_cuda_buffer_allocation() {
        let buffer = CudaBuffer::<f32>::new(1024);
        assert!(buffer.is_ok());

        if let Ok(buf) = buffer {
            assert_eq!(buf.size(), 1024);
        }
    }

    #[test]
    fn convert_opencl_to_cuda_translates_qualifiers() {
        let src = "__kernel void k(__global float* a) { int i = get_global_id(0); a[i] = 0.0f; }";
        let out = convert_opencl_to_cuda(src);
        assert!(out.contains("extern \"C\" __global__"));
        assert!(out.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(out.contains("#include <cuda_runtime.h>"));
    }

    #[test]
    fn launch_config_covers_work_items() {
        let (grid, block) = calculate_launch_config(&[1024, 768], 2);
        assert!(block.0 >= 1 && block.1 >= 1 && block.2 == 1);
        assert!(grid.0 >= 1 && grid.1 >= 1);

        let (grid1, block1) = calculate_launch_config(&[4096], 1);
        assert_eq!(block1.1, 1);
        assert_eq!(block1.2, 1);
        assert!(grid1.0 >= 1);
    }

    #[test]
    fn nvrtc_probe_never_panics() {
        // Probing for libnvrtc must not panic whether or not it is installed.
        let _ = oxicuda_nvrtc::is_available();
    }

    #[test]
    fn cuda_context_creation_never_panics_without_gpu() {
        // On a machine without a CUDA driver this returns an error; on one with
        // a driver it may succeed. Either way it must not panic.
        let _ = CudaContext::new(None);
    }

    #[test]
    fn allocate_gpu_buffer_degrades_without_gpu() {
        // Allocation without a CUDA driver returns a typed error rather than
        // panicking.
        let _ = allocate_gpu_buffer::<f32>(&[1.0_f32, 2.0, 3.0]);
        let _ = allocate_gpu_buffer_empty::<f32>(16);
    }
}
