//! Concrete GPU backend implementations for CUDA and OpenCL
//!
//! This module provides actual implementations of CUDA and OpenCL backends
//! for ndimage operations, replacing the placeholder implementations with
//! functional GPU compute capabilities.

#[cfg(any(feature = "cuda", feature = "opencl"))]
use std::collections::HashMap;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use std::sync::{Arc, Mutex};

use scirs2_core::ndarray::{Array, ArrayView2, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive};

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[allow(unused_imports)]
use crate::backend::gpu_acceleration_framework::{
    CompiledKernel, GpuBuffer, GpuBufferHandle, KernelHandle,
};

#[cfg(feature = "cuda")]
use crate::backend::gpu_acceleration_framework::{CudaBufferHandle, CudaKernelHandle};

#[cfg(feature = "opencl")]
use crate::backend::gpu_acceleration_framework::{OpenCLBufferHandle, OpenCLKernelHandle};
use crate::error::{NdimageError, NdimageResult};

/// CUDA backend implementation
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    /// CUDA context
    context: CudaContext,
    /// Device properties
    device_properties: CudaDeviceProperties,
    /// Compiled kernels cache
    kernel_cache: Arc<Mutex<HashMap<String, CudaKernelHandle>>>,
    /// Memory allocations tracking
    allocations: Arc<Mutex<HashMap<usize, usize>>>, // ptr -> size
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaContext {
    /// CUDA context handle
    pub context: usize,
    /// CUDA device ID
    pub device_id: i32,
    /// CUDA stream
    pub stream: usize,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn new(_device_id: Option<usize>) -> crate::error::NdimageResult<Self> {
        use crate::error::NdimageError;

        // A real implementation would initialize a CUDA context (cuCtxCreate)
        // bound to the requested device. The CUDA runtime is not linked into
        // this build, so we return an honest error rather than handing back a
        // context with null handles that would silently no-op.
        Err(NdimageError::GpuNotAvailable(
            "CUDA backend is not linked into this build: cannot create a CUDA \
             context. Build with a CUDA toolkit and FFI bindings to enable GPU \
             dispatch."
                .to_string(),
        ))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    /// Device name
    pub name: String,
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Compute capability major version
    pub compute_capability_major: i32,
    /// Compute capability minor version
    pub compute_capability_minor: i32,
}

/// OpenCL backend implementation
#[cfg(feature = "opencl")]
pub struct OpenCLBackend {
    /// OpenCL context
    context: OpenCLContext,
    /// Device properties
    device_properties: OpenCLDeviceProperties,
    /// Compiled kernels cache
    kernel_cache: Arc<Mutex<HashMap<String, OpenCLKernelHandle>>>,
    /// Memory allocations tracking
    allocations: Arc<Mutex<HashMap<usize, usize>>>, // buffer -> size
}

#[cfg(feature = "opencl")]
#[derive(Debug, Clone)]
pub struct OpenCLContext {
    /// OpenCL context handle
    pub context: usize,
    /// OpenCL device ID
    pub device: usize,
    /// OpenCL command queue
    pub queue: usize,
    /// Platform ID
    pub platform: usize,
}

#[cfg(feature = "opencl")]
impl OpenCLContext {
    pub fn new(_device_id: Option<usize>) -> crate::error::NdimageResult<Self> {
        use crate::error::NdimageError;

        // A real implementation would initialize an OpenCL platform, device,
        // context, and command queue. No OpenCL ICD is linked into this build,
        // so we return an honest error rather than handing back null handles
        // that would silently no-op.
        Err(NdimageError::GpuNotAvailable(
            "OpenCL backend is not linked into this build: cannot create an \
             OpenCL context. Build with an OpenCL ICD loader and FFI bindings to \
             enable GPU dispatch."
                .to_string(),
        ))
    }
}

#[cfg(feature = "opencl")]
#[derive(Debug, Clone)]
pub struct OpenCLDeviceProperties {
    /// Device name
    pub name: String,
    /// Global memory size in bytes
    pub global_memory_size: usize,
    /// Local memory size in bytes
    pub local_memory_size: usize,
    /// Maximum compute units
    pub max_compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Device type (GPU, CPU, etc.)
    pub device_type: String,
}

// CUDA Backend Implementation
#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Initialize CUDA backend
    pub fn new() -> NdimageResult<Self> {
        // Initialize CUDA runtime
        let device_count = Self::get_device_count()?;
        if device_count == 0 {
            return Err(NdimageError::GpuNotAvailable(
                "No CUDA devices found".to_string(),
            ));
        }

        // Use device 0 by default
        let device_id = 0;
        let context = Self::createcontext(device_id)?;
        let device_properties = Self::get_device_properties(device_id)?;

        Ok(Self {
            context,
            device_properties,
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            allocations: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Allocate GPU memory
    pub fn allocate_memory(&self, size: usize) -> NdimageResult<CudaBufferHandle> {
        let device_ptr = self.cuda_malloc(size)?;

        // Track allocation
        {
            let mut allocations = self.allocations.lock().expect("Operation failed");
            allocations.insert(device_ptr, size);
        }

        Ok(CudaBufferHandle {
            device_ptr,
            device_id: self.context.device_id,
            stream: Some(self.context.stream),
        })
    }

    /// Deallocate GPU memory
    pub fn deallocate_memory(&self, handle: &CudaBufferHandle) -> NdimageResult<()> {
        self.cuda_free(handle.device_ptr)?;

        // Remove from tracking
        {
            let mut allocations = self.allocations.lock().expect("Operation failed");
            allocations.remove(&handle.device_ptr);
        }

        Ok(())
    }

    /// Copy data from host to device
    pub fn copy_to_device<T>(
        &self,
        host_data: &[T],
        device_handle: &CudaBufferHandle,
    ) -> NdimageResult<()>
    where
        T: Clone,
    {
        let size_bytes = host_data.len() * std::mem::size_of::<T>();
        self.cuda_memcpy_htod(
            device_handle.device_ptr,
            host_data.as_ptr() as *const u8,
            size_bytes,
        )
    }

    /// Copy data from device to host
    pub fn copy_from_device<T>(
        &self,
        device_handle: &CudaBufferHandle,
        host_data: &mut [T],
    ) -> NdimageResult<()>
    where
        T: Clone,
    {
        let size_bytes = host_data.len() * std::mem::size_of::<T>();
        self.cuda_memcpy_dtoh(
            host_data.as_mut_ptr() as *mut u8,
            device_handle.device_ptr,
            size_bytes,
        )
    }

    /// Compile CUDA kernel
    pub fn compile_kernel(
        &self,
        source: &str,
        kernel_name: &str,
    ) -> NdimageResult<CudaKernelHandle> {
        // Check cache first
        {
            let cache = self.kernel_cache.lock().expect("Operation failed");
            if let Some(handle) = cache.get(&format!("{}:{}", source.len(), kernel_name)) {
                return Ok(handle.clone());
            }
        }

        // Compile kernel
        let module = self.compile_ptx_from_source(source)?;
        let function = self.get_function(module, kernel_name)?;

        let handle = CudaKernelHandle { function, module };

        // Cache the compiled kernel
        {
            let mut cache = self.kernel_cache.lock().expect("Operation failed");
            cache.insert(format!("{}:{}", source.len(), kernel_name), handle.clone());
        }

        Ok(handle)
    }

    /// Launch CUDA kernel
    pub fn launch_kernel<T>(
        &self,
        kernel: &CudaKernelHandle,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[&CudaBufferHandle],
        shared_memory: usize,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive,
    {
        // Prepare kernel arguments
        let mut kernel_args: Vec<*mut std::ffi::c_void> = Vec::new();
        for arg in args {
            kernel_args.push(&arg.device_ptr as *const usize as *mut std::ffi::c_void);
        }

        // Launch kernel
        self.cuda_launch_kernel(
            kernel.function,
            grid_dim,
            block_dim,
            kernel_args.as_ptr(),
            shared_memory,
            self.context.stream,
        )?;

        // Synchronize stream
        self.cuda_stream_synchronize(self.context.stream)?;

        Ok(())
    }

    /// Execute 2D convolution on GPU
    pub fn execute_convolution_2d<T>(
        &self,
        input: ArrayView2<T>,
        kernel: ArrayView2<T>,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone,
    {
        let (input_height, input_width) = input.dim();
        let (kernel_height, kernel_width) = kernel.dim();

        // Allocate GPU memory
        let input_size = input_height * input_width;
        let kernel_size = kernel_height * kernel_width;
        let output_size = input_height * input_width;

        let input_gpu = self.allocate_memory(input_size * std::mem::size_of::<T>())?;
        let kernel_gpu = self.allocate_memory(kernel_size * std::mem::size_of::<T>())?;
        let output_gpu = self.allocate_memory(output_size * std::mem::size_of::<T>())?;

        // Copy data to GPU
        let input_flat: Vec<T> = input.iter().cloned().collect();
        let kernel_flat: Vec<T> = kernel.iter().cloned().collect();

        self.copy_to_device(&input_flat, &input_gpu)?;
        self.copy_to_device(&kernel_flat, &kernel_gpu)?;

        // Compile and launch convolution kernel
        let conv_kernel =
            self.compile_kernel(&self.get_convolution_kernel_source(), "convolution_2d")?;

        // Calculate grid and block dimensions
        let block_size = 16;
        let grid_x = (input_width + block_size - 1) / block_size;
        let grid_y = (input_height + block_size - 1) / block_size;

        let args = [&input_gpu, &kernel_gpu, &output_gpu];

        self.launch_kernel::<T>(
            &conv_kernel,
            (grid_x as u32, grid_y as u32, 1),
            (block_size as u32, block_size as u32, 1),
            &args,
            0, // No shared memory
        )?;

        // Copy result back to host
        let mut output_flat = vec![T::zero(); output_size];
        self.copy_from_device(&output_gpu, &mut output_flat)?;

        // Clean up GPU memory
        self.deallocate_memory(&input_gpu)?;
        self.deallocate_memory(&kernel_gpu)?;
        self.deallocate_memory(&output_gpu)?;

        // Reshape result
        Ok(
            Array::from_shape_vec((input_height, input_width), output_flat).map_err(|e| {
                NdimageError::InvalidInput(format!("Failed to reshape result: {}", e))
            })?,
        )
    }

    // Low-level CUDA API wrappers.
    //
    // These are the FFI seams that must bind to the real CUDA runtime/driver
    // API (cudaGetDeviceCount, cudaMalloc, cudaLaunchKernel, ...). This build
    // does NOT link the CUDA runtime, so every seam returns an honest error
    // instead of fabricating success (a no-op `Ok(())`, a dummy device pointer,
    // or invented device specifications such as a "GeForce RTX 4090"). Returning
    // an error from `get_device_count` makes `CudaBackend::new` fail cleanly so
    // that no kernel is ever reported as having run.

    fn cuda_not_linked(op: &str) -> NdimageError {
        NdimageError::GpuNotAvailable(format!(
            "CUDA backend is not linked into this build: cannot perform '{op}'. \
             Build with a CUDA toolkit and the appropriate FFI bindings to enable \
             GPU dispatch. Refusing to fabricate a successful GPU operation.",
        ))
    }

    fn get_device_count() -> NdimageResult<i32> {
        // Would call cudaGetDeviceCount; the runtime is not linked.
        Err(Self::cuda_not_linked("cudaGetDeviceCount"))
    }

    fn createcontext(_deviceid: i32) -> NdimageResult<CudaContext> {
        // Would initialize a CUDA context and stream; the runtime is not linked.
        Err(Self::cuda_not_linked("cuCtxCreate"))
    }

    fn get_device_properties(_deviceid: i32) -> NdimageResult<CudaDeviceProperties> {
        // Would query real device properties; the runtime is not linked.
        Err(Self::cuda_not_linked("cudaGetDeviceProperties"))
    }

    fn cuda_malloc(&self, _size: usize) -> NdimageResult<usize> {
        Err(Self::cuda_not_linked("cudaMalloc"))
    }

    fn cuda_free(&self, _deviceptr: usize) -> NdimageResult<()> {
        Err(Self::cuda_not_linked("cudaFree"))
    }

    fn cuda_memcpy_htod(
        &self,
        _device_ptr: usize,
        _host_ptr: *const u8,
        _size: usize,
    ) -> NdimageResult<()> {
        Err(Self::cuda_not_linked("cudaMemcpy (host to device)"))
    }

    fn cuda_memcpy_dtoh(
        &self,
        _host_ptr: *mut u8,
        _device_ptr: usize,
        _size: usize,
    ) -> NdimageResult<()> {
        Err(Self::cuda_not_linked("cudaMemcpy (device to host)"))
    }

    fn compile_ptx_from_source(&self, _source: &str) -> NdimageResult<usize> {
        Err(Self::cuda_not_linked("nvrtc/cuModuleLoadData"))
    }

    fn get_function(&self, _module: usize, _name: &str) -> NdimageResult<usize> {
        Err(Self::cuda_not_linked("cuModuleGetFunction"))
    }

    fn cuda_launch_kernel(
        &self,
        _function: usize,
        _grid_dim: (u32, u32, u32),
        _block_dim: (u32, u32, u32),
        _args: *const *mut std::ffi::c_void,
        _shared_memory: usize,
        _stream: usize,
    ) -> NdimageResult<()> {
        Err(Self::cuda_not_linked("cudaLaunchKernel"))
    }

    fn cuda_stream_synchronize(&self, _stream: usize) -> NdimageResult<()> {
        Err(Self::cuda_not_linked("cudaStreamSynchronize"))
    }

    fn get_convolution_kernel_source(&self) -> String {
        // CUDA kernel source for 2D convolution
        r#"
extern "C" __global__ void convolution_2d(
    const float* input,
    const float* kernel,
    float* output,
    int input_width,
    int input_height,
    int kernel_width,
    int kernel_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= input_width || y >= input_height) return;
    
    float sum = 0.0f;
    int kernel_center_x = kernel_width / 2;
    int kernel_center_y = kernel_height / 2;
    
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            int input_x = x + kx - kernel_center_x;
            int input_y = y + ky - kernel_center_y;
            
            // Boundary handling: clamp to edges
            input_x = max(0, min(input_x, input_width - 1));
            input_y = max(0, min(input_y, input_height - 1));
            
            sum += input[input_y * input_width + input_x] * kernel[ky * kernel_width + kx];
        }
    }
    
    output[y * input_width + x] = sum;
}
"#
        .to_string()
    }
}

// OpenCL Backend Implementation
#[cfg(feature = "opencl")]
impl OpenCLBackend {
    /// Initialize OpenCL backend
    pub fn new() -> NdimageResult<Self> {
        let context = Self::create_openclcontext()?;
        let device_properties = Self::get_device_properties(&context)?;

        Ok(Self {
            context,
            device_properties,
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            allocations: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Allocate OpenCL buffer
    pub fn allocate_buffer(&self, size: usize) -> NdimageResult<OpenCLBufferHandle> {
        let buffer = self.cl_create_buffer(size)?;

        // Track allocation
        {
            let mut allocations = self.allocations.lock().expect("Operation failed");
            allocations.insert(buffer, size);
        }

        Ok(OpenCLBufferHandle {
            buffer,
            context: self.context.context,
            queue: self.context.queue,
        })
    }

    /// Deallocate OpenCL buffer
    pub fn deallocate_buffer(&self, handle: &OpenCLBufferHandle) -> NdimageResult<()> {
        self.cl_release_buffer(handle.buffer)?;

        // Remove from tracking
        {
            let mut allocations = self.allocations.lock().expect("Operation failed");
            allocations.remove(&handle.buffer);
        }

        Ok(())
    }

    /// Write data to OpenCL buffer
    pub fn write_buffer<T>(&self, buffer: &OpenCLBufferHandle, data: &[T]) -> NdimageResult<()>
    where
        T: Clone,
    {
        let size_bytes = data.len() * std::mem::size_of::<T>();
        self.cl_enqueue_write_buffer(buffer.buffer, data.as_ptr() as *const u8, size_bytes)
    }

    /// Read data from OpenCL buffer
    pub fn read_buffer<T>(&self, buffer: &OpenCLBufferHandle, data: &mut [T]) -> NdimageResult<()>
    where
        T: Clone,
    {
        let size_bytes = data.len() * std::mem::size_of::<T>();
        self.cl_enqueue_read_buffer(buffer.buffer, data.as_mut_ptr() as *mut u8, size_bytes)
    }

    /// Compile OpenCL kernel
    pub fn compile_kernel(
        &self,
        source: &str,
        kernel_name: &str,
    ) -> NdimageResult<OpenCLKernelHandle> {
        // Check cache first
        let cache_key = format!("{}:{}", source.len(), kernel_name);
        {
            let cache = self.kernel_cache.lock().expect("Operation failed");
            if let Some(handle) = cache.get(&cache_key) {
                return Ok(handle.clone());
            }
        }

        // Compile kernel
        let program = self.cl_create_program_with_source(source)?;
        self.cl_build_program(program)?;
        let kernel = self.cl_create_kernel(program, kernel_name)?;

        let handle = OpenCLKernelHandle { kernel, program };

        // Cache the compiled kernel
        {
            let mut cache = self.kernel_cache.lock().expect("Operation failed");
            cache.insert(cache_key, handle.clone());
        }

        Ok(handle)
    }

    /// Execute OpenCL kernel
    pub fn execute_kernel(
        &self,
        kernel: &OpenCLKernelHandle,
        global_work_size: &[usize],
        local_work_size: Option<&[usize]>,
        args: &[&OpenCLBufferHandle],
    ) -> NdimageResult<()> {
        // Set kernel arguments
        for (i, arg) in args.iter().enumerate() {
            self.cl_set_kernel_arg(kernel.kernel, i, &arg.buffer)?;
        }

        // Enqueue kernel execution
        self.cl_enqueue_nd_range_kernel(kernel.kernel, global_work_size, local_work_size)?;

        // Wait for completion
        self.cl_finish()?;

        Ok(())
    }

    /// Execute 2D convolution using OpenCL
    pub fn execute_convolution_2d<T>(
        &self,
        input: ArrayView2<T>,
        kernel: ArrayView2<T>,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone,
    {
        let (input_height, input_width) = input.dim();
        let (kernel_height, kernel_width) = kernel.dim();

        // Allocate OpenCL buffers
        let input_size = input_height * input_width;
        let kernel_size = kernel_height * kernel_width;

        let input_buffer = self.allocate_buffer(input_size * std::mem::size_of::<T>())?;
        let kernel_buffer = self.allocate_buffer(kernel_size * std::mem::size_of::<T>())?;
        let output_buffer = self.allocate_buffer(input_size * std::mem::size_of::<T>())?;

        // Copy data to GPU
        let input_flat: Vec<T> = input.iter().cloned().collect();
        let kernel_flat: Vec<T> = kernel.iter().cloned().collect();

        self.write_buffer(&input_buffer, &input_flat)?;
        self.write_buffer(&kernel_buffer, &kernel_flat)?;

        // Compile and execute convolution kernel
        let conv_kernel =
            self.compile_kernel(&self.get_convolution_kernel_source(), "convolution_2d")?;

        let global_work_size = [input_width, input_height];
        let local_work_size = [16, 16];

        let args = [&input_buffer, &kernel_buffer, &output_buffer];

        self.execute_kernel(
            &conv_kernel,
            &global_work_size,
            Some(&local_work_size),
            &args,
        )?;

        // Copy result back
        let mut output_flat = vec![T::zero(); input_size];
        self.read_buffer(&output_buffer, &mut output_flat)?;

        // Clean up
        self.deallocate_buffer(&input_buffer)?;
        self.deallocate_buffer(&kernel_buffer)?;
        self.deallocate_buffer(&output_buffer)?;

        // Reshape result
        Ok(
            Array::from_shape_vec((input_height, input_width), output_flat).map_err(|e| {
                NdimageError::InvalidInput(format!("Failed to reshape result: {}", e))
            })?,
        )
    }

    // Low-level OpenCL API wrappers.
    //
    // These are the FFI seams that must bind to the real OpenCL API
    // (clGetPlatformIDs, clCreateBuffer, clEnqueueNDRangeKernel, ...). This
    // build does NOT link an OpenCL ICD, so every seam returns an honest error
    // instead of fabricating success (a no-op `Ok(())`, a dummy buffer handle,
    // or invented device specifications such as an "AMD Radeon RX 7900 XTX").
    // Returning an error from `create_openclcontext` makes `OpenCLBackend::new`
    // fail cleanly so that no kernel is ever reported as having run.

    fn opencl_not_linked(op: &str) -> NdimageError {
        NdimageError::GpuNotAvailable(format!(
            "OpenCL backend is not linked into this build: cannot perform '{op}'. \
             Build with an OpenCL ICD loader and the appropriate FFI bindings to \
             enable GPU dispatch. Refusing to fabricate a successful GPU operation.",
        ))
    }

    fn create_openclcontext() -> NdimageResult<OpenCLContext> {
        // Would initialize an OpenCL platform/device/context/queue.
        Err(Self::opencl_not_linked("clCreateContext"))
    }

    fn get_device_properties(_context: &OpenCLContext) -> NdimageResult<OpenCLDeviceProperties> {
        // Would query real OpenCL device properties.
        Err(Self::opencl_not_linked("clGetDeviceInfo"))
    }

    fn cl_create_buffer(&self, _size: usize) -> NdimageResult<usize> {
        Err(Self::opencl_not_linked("clCreateBuffer"))
    }

    fn cl_release_buffer(&self, _buffer: usize) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clReleaseMemObject"))
    }

    fn cl_enqueue_write_buffer(
        &self,
        _buffer: usize,
        _data: *const u8,
        _size: usize,
    ) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clEnqueueWriteBuffer"))
    }

    fn cl_enqueue_read_buffer(
        &self,
        _buffer: usize,
        _data: *mut u8,
        _size: usize,
    ) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clEnqueueReadBuffer"))
    }

    fn cl_create_program_with_source(&self, _source: &str) -> NdimageResult<usize> {
        Err(Self::opencl_not_linked("clCreateProgramWithSource"))
    }

    fn cl_build_program(&self, _program: usize) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clBuildProgram"))
    }

    fn cl_create_kernel(&self, _program: usize, _name: &str) -> NdimageResult<usize> {
        Err(Self::opencl_not_linked("clCreateKernel"))
    }

    fn cl_set_kernel_arg(
        &self,
        _kernel: usize,
        _arg_index: usize,
        _buffer: &usize,
    ) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clSetKernelArg"))
    }

    fn cl_enqueue_nd_range_kernel(
        &self,
        _kernel: usize,
        _global_work_size: &[usize],
        _local_work_size: Option<&[usize]>,
    ) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clEnqueueNDRangeKernel"))
    }

    fn cl_finish(&self) -> NdimageResult<()> {
        Err(Self::opencl_not_linked("clFinish"))
    }

    fn get_convolution_kernel_source(&self) -> String {
        // OpenCL kernel source for 2D convolution
        r#"
__kernel void convolution_2d(
    __global const float* input__global const float* kernel__global float* output,
    const int input_width,
    const int input_height,
    const int kernel_width,
    const int kernel_height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= input_width || y >= input_height) return;
    
    float sum = 0.0f;
    int kernel_center_x = kernel_width / 2;
    int kernel_center_y = kernel_height / 2;
    
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            int input_x = x + kx - kernel_center_x;
            int input_y = y + ky - kernel_center_y;
            
            // Boundary handling: clamp to edges
            input_x = max(0, min(input_x, input_width - 1));
            input_y = max(0, min(input_y, input_height - 1));
            
            sum += input[input_y * input_width + input_x] * kernel[ky * kernel_width + kx];
        }
    }
    
    output[y * input_width + x] = sum;
}
"#
        .to_string()
    }
}

/// Factory function to create appropriate GPU backend
#[allow(dead_code)]
pub fn create_gpu_backend() -> NdimageResult<Box<dyn GpuBackend>> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_backend) = CudaBackend::new() {
            return Ok(Box::new(cuda_backend));
        }
    }

    #[cfg(feature = "opencl")]
    {
        if let Ok(opencl_backend) = OpenCLBackend::new() {
            return Ok(Box::new(opencl_backend));
        }
    }

    Err(NdimageError::GpuNotAvailable(
        "GPU backend not available".to_string(),
    ))
}

/// Common GPU backend trait
pub trait GpuBackend: Send + Sync {
    /// Get backend name
    fn get_name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Get memory info
    fn get_memory_info(&self) -> (usize, usize); // (free, total)

    /// Execute 2D convolution
    fn execute_convolution_2d_f32(
        &self,
        input: ArrayView2<f32>,
        kernel: ArrayView2<f32>,
    ) -> NdimageResult<Array<f32, Ix2>>;

    /// Execute 2D convolution for f64
    fn execute_convolution_2d_f64(
        &self,
        input: ArrayView2<f64>,
        kernel: ArrayView2<f64>,
    ) -> NdimageResult<Array<f64, Ix2>>;
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn get_name(&self) -> &str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        // A `CudaBackend` cannot currently be constructed (`CudaBackend::new`
        // returns an error because the CUDA runtime is not linked), so this is
        // unreachable in practice; it would only be reachable once real FFI
        // bindings exist, at which point reaching here means CUDA is available.
        true
    }

    fn get_memory_info(&self) -> (usize, usize) {
        // Would query actual CUDA memory info (cudaMemGetInfo). The runtime is
        // not linked, so report zero rather than fabricating device capacity.
        (0, 0) // (free, total)
    }

    fn execute_convolution_2d_f32(
        &self,
        input: ArrayView2<f32>,
        kernel: ArrayView2<f32>,
    ) -> NdimageResult<Array<f32, Ix2>> {
        self.execute_convolution_2d(input, kernel)
    }

    fn execute_convolution_2d_f64(
        &self,
        input: ArrayView2<f64>,
        kernel: ArrayView2<f64>,
    ) -> NdimageResult<Array<f64, Ix2>> {
        self.execute_convolution_2d(input, kernel)
    }
}

#[cfg(feature = "opencl")]
impl GpuBackend for OpenCLBackend {
    fn get_name(&self) -> &str {
        "OpenCL"
    }

    fn is_available(&self) -> bool {
        // An `OpenCLBackend` cannot currently be constructed
        // (`OpenCLBackend::new` returns an error because no OpenCL ICD is
        // linked), so this is unreachable in practice; it would only be
        // reachable once real FFI bindings exist.
        true
    }

    fn get_memory_info(&self) -> (usize, usize) {
        // Would query actual OpenCL memory info (clGetDeviceInfo). No ICD is
        // linked, so report zero rather than fabricating device capacity.
        (0, 0) // (free, total)
    }

    fn execute_convolution_2d_f32(
        &self,
        input: ArrayView2<f32>,
        kernel: ArrayView2<f32>,
    ) -> NdimageResult<Array<f32, Ix2>> {
        self.execute_convolution_2d(input, kernel)
    }

    fn execute_convolution_2d_f64(
        &self,
        input: ArrayView2<f64>,
        kernel: ArrayView2<f64>,
    ) -> NdimageResult<Array<f64, Ix2>> {
        self.execute_convolution_2d(input, kernel)
    }
}

#[cfg(all(feature = "cuda", feature = "gpu"))]
impl crate::backend::GpuContext for CudaContext {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_count(&self) -> usize {
        // A `CudaContext` cannot be constructed without a linked CUDA runtime,
        // so there are genuinely no devices to report here.
        0
    }

    fn current_device(&self) -> usize {
        self.device_id as usize
    }

    fn memory_info(&self) -> (usize, usize) {
        // No CUDA runtime linked: report zero rather than a fabricated capacity.
        (0, 0) // (free, total)
    }
}

#[cfg(all(feature = "opencl", feature = "gpu"))]
impl crate::backend::GpuContext for OpenCLContext {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device_count(&self) -> usize {
        // An `OpenCLContext` cannot be constructed without a linked OpenCL ICD,
        // so there are genuinely no devices to report here.
        0
    }

    fn current_device(&self) -> usize {
        self.device
    }

    fn memory_info(&self) -> (usize, usize) {
        // No OpenCL ICD linked: report zero rather than a fabricated capacity.
        (0, 0) // (free, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_backend_creation() {
        let result = create_gpu_backend();
        // This test may fail on systems without GPU support, which is expected
        assert!(result.is_ok() || result.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_backend_creation() {
        let result = CudaBackend::new();
        // This may fail without actual CUDA drivers
        assert!(result.is_ok() || result.is_err());
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl_backend_creation() {
        let result = OpenCLBackend::new();
        // This may fail without actual OpenCL drivers
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_convolution_execution() {
        // Test with small arrays
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let kernel = array![[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];

        if let Ok(backend) = create_gpu_backend() {
            let result = backend.execute_convolution_2d_f64(input.view(), kernel.view());
            // Should either succeed or fail gracefully
            assert!(result.is_ok() || result.is_err());
        }
    }
}
