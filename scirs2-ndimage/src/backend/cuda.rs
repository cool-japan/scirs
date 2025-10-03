//! CUDA backend implementation for GPU acceleration
//!
//! This module provides CUDA-specific implementations for GPU-accelerated
//! image processing operations.

// Allow unused unsafe blocks for CUDA fallback functions
#![allow(unused_unsafe)]

use crate::backend::kernels::{GpuBuffer, GpuKernelExecutor, KernelInfo};
use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array, ArrayView2, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Debug;

/// GPU context trait for different GPU backends
pub trait GpuContext: Send + Sync {
    fn name(&self) -> &str;
    fn device_count(&self) -> usize;
    fn current_device(&self) -> usize;
    fn memory_info(&self) -> (usize, usize); // (used, total)
}

use std::ptr;
use std::sync::{Arc, Mutex};

// CUDA FFI bindings - only link on supported platforms with CUDA feature
#[cfg(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
))]
#[link(name = "cuda")]
#[link(name = "cudart")]
#[link(name = "nvrtc")]
extern "C" {
    // CUDA Runtime API
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;

    // CUDA Driver API for kernel launch
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const c_char) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;

    // NVRTC API for runtime compilation
    fn nvrtcCreateProgram(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn nvrtcDestroyProgram(prog: *mut *mut c_void) -> i32;
    fn nvrtcCompileProgram(
        prog: *mut c_void,
        num_options: i32,
        options: *const *const c_char,
    ) -> i32;
    fn nvrtcGetPTXSize(_prog: *mut c_void, ptxsize: *mut usize) -> i32;
    fn nvrtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> i32;
    fn nvrtcGetProgramLogSize(_prog: *mut c_void, logsize: *mut usize) -> i32;
    fn nvrtcGetProgramLog(prog: *mut c_void, log: *mut c_char) -> i32;
}

// Fallback function declarations for non-CUDA platforms
// These function names must match CUDA API exactly
#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaMalloc(_dev_ptr: *mut *mut c_void, _size: usize) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaFree(_dev_ptr: *mut c_void) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaMemcpyAsync(
    _dst: *mut c_void,
    _src: *const c_void,
    _count: usize,
    _kind: i32,
    _stream: *mut c_void,
) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaStreamCreate(_stream: *mut *mut c_void) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaGetLastError() -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaGetErrorString(_error: i32) -> *const c_char {
    b"No error (fallback)\0".as_ptr() as *const c_char
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaGetDeviceCount(_count: *mut i32) -> i32 {
    unsafe {
        *_count = 1; // Simulate 1 device
    }
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaMemGetInfo(_free: *mut usize, _total: *mut usize) -> i32 {
    unsafe {
        *_free = 1024 * 1024 * 1024; // 1GB free
        *_total = 2 * 1024 * 1024 * 1024; // 2GB total
    }
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaStreamDestroy(_stream: *mut c_void) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaStreamSynchronize(_stream: *mut c_void) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcCreateProgram(
    _prog: *mut *mut c_void,
    _src: *const c_char,
    _name: *const c_char,
    _num_headers: i32,
    _headers: *const *const c_char,
    _include_names: *const *const c_char,
) -> i32 {
    unsafe {
        *_prog = 0x1 as *mut c_void; // Dummy program pointer
    }
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcCompileProgram(
    _prog: *mut c_void,
    _num_options: i32,
    _options: *const *const c_char,
) -> i32 {
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcGetProgramLogSize(_prog: *mut c_void, logsize: *mut usize) -> i32 {
    unsafe {
        *logsize = 1; // Empty log
    }
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcGetProgramLog(_prog: *mut c_void, _log: *mut c_char) -> i32 {
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcDestroyProgram(_prog: *mut *mut c_void) -> i32 {
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcGetPTXSize(_prog: *mut c_void, ptxsize: *mut usize) -> i32 {
    unsafe {
        *ptxsize = 100; // Dummy PTX size
    }
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn nvrtcGetPTX(_prog: *mut c_void, _ptx: *mut c_char) -> i32 {
    NVRTC_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cuModuleLoadData(_module: *mut *mut c_void, _image: *const c_void) -> i32 {
    unsafe {
        *_module = 0x2 as *mut c_void; // Dummy module pointer
    }
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cuModuleGetFunction(_hfunc: *mut *mut c_void, _hmod: *mut c_void, _name: *const c_char) -> i32 {
    unsafe {
        *_hfunc = 0x3 as *mut c_void; // Dummy function pointer
    }
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cuLaunchKernel(
    _f: *mut c_void,
    _grid_dim_x: u32,
    _grid_dim_y: u32,
    _grid_dim_z: u32,
    _block_dim_x: u32,
    _block_dim_y: u32,
    _block_dim_z: u32,
    _shared_mem_bytes: u32,
    _stream: *mut c_void,
    _kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaMemcpy(_dst: *mut c_void, _src: *const c_void, _count: usize, _kind: i32) -> i32 {
    CUDA_SUCCESS
}

#[cfg(not(all(
    feature = "cuda",
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
#[allow(non_snake_case)]
fn cudaSetDevice(_device: i32) -> i32 {
    CUDA_SUCCESS
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// CUDA error codes
const CUDA_SUCCESS: i32 = 0;
const NVRTC_SUCCESS: i32 = 0;

// Helper function to get CUDA error string
#[allow(dead_code)]
fn cuda_error_string(error: i32) -> String {
    unsafe {
        let error_ptr = cudaGetErrorString(error);
        if error_ptr.is_null() {
            format!("Unknown CUDA error: {error}")
        } else {
            CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
        }
    }
}

/// CUDA-specific GPU buffer implementation
pub struct CudaBuffer<T>
where
    T: Send + Sync,
{
    device_ptr: *mut c_void,
    size: usize,
    phantom: std::marker::PhantomData<T>,
}

// CUDA device pointers are thread-safe as long as the CUDA context is properly managed
unsafe impl<T: Send + Sync> Send for CudaBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for CudaBuffer<T> {}

impl<T: Send + Sync + 'static> CudaBuffer<T> {
    pub fn new(size: usize) -> NdimageResult<Self> {
        let mut device_ptr: *mut c_void = ptr::null_mut();
        let byte_size = size * std::mem::size_of::<T>();

        unsafe {
            let result = cudaMalloc(&mut device_ptr, byte_size);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA malloc failed with error code: {result}"
                )));
            }
        }

        Ok(Self {
            device_ptr,
            size,
            phantom: std::marker::PhantomData,
        })
    }

    pub fn from_host_data(data: &[T]) -> NdimageResult<Self> {
        let mut buffer = Self::new(data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }
}

impl<T: Send + Sync> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.device_ptr.is_null() {
                cudaFree(self.device_ptr);
            }
        }
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

        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            let result = cudaMemcpy(
                self.device_ptr,
                data.as_ptr() as *const c_void,
                byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA memcpy failed with error code: {result}"
                )));
            }
        }

        Ok(())
    }

    fn copy_to_host(&self, data: &mut [T]) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }

        let byte_size = self.size * std::mem::size_of::<T>();
        unsafe {
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.device_ptr,
                byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );

            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA memcpy failed with error code: {result}"
                )));
            }
        }

        Ok(())
    }
}

/// CUDA context implementation
pub struct CudaContext {
    device_id: i32,
    compute_capability: (i32, i32),
    max_threads_per_block: i32,
    max_shared_memory: usize,
}

impl CudaContext {
    pub fn new(_deviceid: Option<usize>) -> NdimageResult<Self> {
        let device_id = _deviceid.unwrap_or(0) as i32;

        // Check if device exists
        let mut device_count: i32 = 0;
        unsafe {
            let result = cudaGetDeviceCount(&mut device_count);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to get CUDA device count: {result}"
                )));
            }

            if device_id >= device_count {
                return Err(NdimageError::InvalidInput(format!(
                    "CUDA device {device_id} not found. Only {device_count} devices available"
                )));
            }

            // Set the device
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to set CUDA device: {result}"
                )));
            }
        }

        // Get device properties
        let (compute_capability, max_threads_per_block, max_shared_memory) =
            Self::get_device_properties(device_id)?;

        Ok(Self {
            device_id,
            compute_capability,
            max_threads_per_block,
            max_shared_memory,
        })
    }

    /// Get device properties for the specified CUDA device
    fn get_device_properties(_deviceid: i32) -> NdimageResult<((i32, i32), i32, usize)> {
        // For now, return sensible defaults based on common GPU architectures
        // In a full implementation, this would query actual device properties
        let compute_capability = match _deviceid {
            0 => (7, 5), // Assume Turing architecture for first device
            1 => (8, 0), // Assume Ampere architecture for second device
            _ => (7, 0), // Default to Volta for others
        };

        let max_threads_per_block = match compute_capability {
            (8, _) => 1024, // Ampere
            (7, _) => 1024, // Turing/Volta
            _ => 512,       // Older architectures
        };

        let max_shared_memory = match compute_capability {
            (8, _) => 99328, // Ampere: 96KB + 3KB
            (7, 5) => 65536, // Turing: 64KB
            (7, _) => 49152, // Volta: 48KB
            _ => 32768,      // Older: 32KB
        };

        Ok((compute_capability, max_threads_per_block, max_shared_memory))
    }

    /// Get optimal kernel compilation options based on compute capability
    fn get_compilation_options(&self) -> NdimageResult<Vec<CString>> {
        let arch_option = format!(
            "--gpu-architecture=compute_{}{}",
            self.compute_capability.0, self.compute_capability.1
        );

        let mut options = vec![
            CString::new(arch_option).map_err(|_| {
                NdimageError::ComputationError(
                    "Failed to create compute architecture option".into(),
                )
            })?,
            CString::new("--fmad=true").map_err(|_| {
                NdimageError::ComputationError("Failed to create fmad option".into())
            })?,
            CString::new("--use_fast_math").map_err(|_| {
                NdimageError::ComputationError("Failed to create fast math option".into())
            })?,
            CString::new("--restrict").map_err(|_| {
                NdimageError::ComputationError("Failed to create restrict option".into())
            })?,
        ];

        // Add optimization options based on compute capability
        if self.compute_capability >= (7, 0) {
            options.push(CString::new("--extra-device-vectorization").map_err(|_| {
                NdimageError::ComputationError("Failed to create vectorization option".into())
            })?);
        }

        if self.compute_capability >= (8, 0) {
            options.push(CString::new("--allow-unsupported-compiler").map_err(|_| {
                NdimageError::ComputationError("Failed to create compiler option".into())
            })?);
        }

        Ok(options)
    }

    pub fn compile_kernel(&self, source: &str, kernelname: &str) -> NdimageResult<CudaKernel> {
        // Check cache first
        {
            let cache = KERNEL_CACHE.lock().map_err(|_| {
                NdimageError::ComputationError("Failed to acquire kernel cache lock".into())
            })?;
            if let Some(kernel) = cache.get(kernelname) {
                return Ok(CudaKernel {
                    name: kernel.name.clone(),
                    module: kernel.module,
                    function: kernel.function,
                    ptx_code: kernel.ptx_code.clone(),
                });
            }
        }

        // Convert OpenCL-style kernel to CUDA
        let cuda_source = convert_opencl_to_cuda(source);
        let c_source = CString::new(cuda_source).map_err(|_| {
            NdimageError::ComputationError("Failed to create C string for kernel source".into())
        })?;
        let c_name = CString::new(kernelname).map_err(|_| {
            NdimageError::ComputationError("Failed to create C string for kernel _name".into())
        })?;

        unsafe {
            // Create NVRTC program
            let mut prog: *mut c_void = ptr::null_mut();
            let result = nvrtcCreateProgram(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            );

            if result != NVRTC_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to create NVRTC program: {result}"
                )));
            }

            // Compile program with appropriate options based on compute capability
            let options = self.get_compilation_options()?;
            let option_ptrs: Vec<*const c_char> = options.iter().map(|s| s.as_ptr()).collect();

            let compile_result =
                nvrtcCompileProgram(prog, option_ptrs.len() as i32, option_ptrs.as_ptr());

            // Get compilation log
            if compile_result != NVRTC_SUCCESS {
                let mut log_size: usize = 0;
                nvrtcGetProgramLogSize(prog, &mut log_size);

                if log_size > 0 {
                    let mut log = vec![0u8; log_size];
                    nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
                    let log_str = String::from_utf8_lossy(&log[..log_size - 1]);

                    nvrtcDestroyProgram(&mut prog);
                    return Err(NdimageError::ComputationError(format!(
                        "CUDA compilation failed:\n{log_str}"
                    )));
                }
            }

            // Get PTX code
            let mut ptx_size: usize = 0;
            nvrtcGetPTXSize(prog, &mut ptx_size);

            let mut ptx_code = vec![0u8; ptx_size];
            nvrtcGetPTX(prog, ptx_code.as_mut_ptr() as *mut c_char);

            // Clean up NVRTC program
            nvrtcDestroyProgram(&mut prog);

            // Load PTX module
            let mut module: *mut c_void = ptr::null_mut();
            let load_result = cuModuleLoadData(&mut module, ptx_code.as_ptr() as *const c_void);

            if load_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to load CUDA module: {}",
                    cuda_error_string(load_result)
                )));
            }

            // Get function from module
            let mut function: *mut c_void = ptr::null_mut();
            let func_result = cuModuleGetFunction(&mut function, module, c_name.as_ptr());

            if func_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to get CUDA function: {}",
                    cuda_error_string(func_result)
                )));
            }

            let kernel = CudaKernel {
                name: kernelname.to_string(),
                module,
                function,
                ptx_code: ptx_code[..ptx_size - 1].to_vec(), // Remove null terminator
            };

            // Cache the compiled kernel
            {
                let mut cache = KERNEL_CACHE.lock().map_err(|_| {
                    NdimageError::ComputationError(
                        "Failed to acquire kernel cache lock for insertion".into(),
                    )
                })?;
                cache.insert(
                    kernelname.to_string(),
                    CudaKernel {
                        name: kernel.name.clone(),
                        module: kernel.module,
                        function: kernel.function,
                        ptx_code: kernel.ptx_code.clone(),
                    },
                );
            }

            Ok(kernel)
        }
    }
}

impl GpuContext for CudaContext {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_count(&self) -> usize {
        let mut count: i32 = 0;
        unsafe {
            cudaGetDeviceCount(&mut count);
        }
        count as usize
    }

    fn current_device(&self) -> usize {
        self.device_id as usize
    }

    fn memory_info(&self) -> (usize, usize) {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            cudaMemGetInfo(&mut free, &mut total);
        }
        (total - free, total)
    }
}

/// CUDA kernel handle
pub struct CudaKernel {
    name: String,
    module: *mut c_void,
    function: *mut c_void,
    ptx_code: Vec<u8>,
}

// SAFETY: CudaKernel's raw pointers are managed by CUDA runtime
// and the kernel cache is only accessed through proper synchronization
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

// SAFETY: CudaExecutor's raw pointers are managed by CUDA runtime
// and access is properly synchronized
unsafe impl Send for CudaExecutor {}
unsafe impl Sync for CudaExecutor {}

// Kernel cache to avoid recompilation
lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Arc<Mutex<HashMap<String, CudaKernel>>> = Arc::new(Mutex::new(HashMap::new()));
}

/// CUDA kernel executor implementation
pub struct CudaExecutor {
    context: Arc<CudaContext>,
    stream: *mut c_void,
}

impl CudaExecutor {
    pub fn new(context: Arc<CudaContext>) -> NdimageResult<Self> {
        let mut stream: *mut c_void = ptr::null_mut();
        unsafe {
            let result = cudaStreamCreate(&mut stream);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to create CUDA stream: {result}"
                )));
            }
        }

        Ok(Self { context, stream })
    }
}

impl Drop for CudaExecutor {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
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

        // Prepare kernel arguments
        let mut kernel_args: Vec<*mut c_void> = Vec::new();

        // Add input buffers
        for input in inputs {
            let cuda_buf = input
                .as_any()
                .downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            kernel_args.push(&cuda_buf.device_ptr as *const _ as *mut c_void);
        }

        // Add output buffers
        for output in outputs {
            let cuda_buf = output
                .as_any()
                .downcast_ref::<CudaBuffer<T>>()
                .ok_or_else(|| NdimageError::InvalidInput("Expected CUDA buffer".into()))?;
            kernel_args.push(&cuda_buf.device_ptr as *const _ as *mut c_void);
        }

        // Add scalar parameters
        let mut param_storage: Vec<T> = params.to_vec();
        for param in &mut param_storage {
            kernel_args.push(param as *mut T as *mut c_void);
        }

        // Launch kernel
        unsafe {
            let result = cuLaunchKernel(
                cuda_kernel.function,
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                0, // shared memory
                self.stream,
                kernel_args.as_mut_ptr(),
                ptr::null_mut(),
            );

            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA kernel launch failed: {}",
                    cuda_error_string(result)
                )));
            }

            // Synchronize stream
            let sync_result = cudaStreamSynchronize(self.stream);
            if sync_result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA stream sync failed: {}",
                    cuda_error_string(sync_result)
                )));
            }

            // Check for kernel errors
            let error = cudaGetLastError();
            if error != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA kernel execution error: {}",
                    cuda_error_string(error)
                )));
            }
        }

        Ok(())
    }
}

/// High-level CUDA operations
pub struct CudaOperations {
    context: Arc<CudaContext>,
    executor: CudaExecutor,
}

impl CudaOperations {
    pub fn new(_deviceid: Option<usize>) -> NdimageResult<Self> {
        let context = Arc::new(CudaContext::new(_deviceid)?);
        let executor = CudaExecutor::new(context.clone())?;

        Ok(Self { context, executor })
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

/// Advanced CUDA memory manager with buffer pooling
pub struct CudaMemoryManager {
    buffer_pools: std::collections::HashMap<usize, Vec<*mut c_void>>,
    total_allocated: usize,
    max_pool_size: usize,
}

impl CudaMemoryManager {
    pub fn new(_max_poolsize: usize) -> Self {
        Self {
            buffer_pools: std::collections::HashMap::new(),
            total_allocated: 0,
            max_pool_size: _max_poolsize,
        }
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocate_buffer(&mut self, size: usize) -> NdimageResult<*mut c_void> {
        // Try to reuse a buffer from the pool
        if let Some(pool) = self.buffer_pools.get_mut(&size) {
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }

        // Allocate a new buffer
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let result = cudaMalloc(&mut device_ptr, size);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA malloc failed: {}",
                    cuda_error_string(result)
                )));
            }
        }

        self.total_allocated += size;
        Ok(device_ptr)
    }

    /// Return a buffer to the pool for reuse
    pub fn deallocate_buffer(&mut self, ptr: *mut c_void, size: usize) -> NdimageResult<()> {
        let pool = self.buffer_pools.entry(size).or_insert_with(Vec::new);

        if pool.len() < self.max_pool_size {
            pool.push(ptr);
        } else {
            // Pool is full, actually free the memory
            unsafe {
                let result = cudaFree(ptr);
                if result != CUDA_SUCCESS {
                    return Err(NdimageError::ComputationError(format!(
                        "CUDA free failed: {}",
                        cuda_error_string(result)
                    )));
                }
            }
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
        for (size, pool) in self.buffer_pools.drain() {
            for ptr in pool {
                unsafe {
                    let result = cudaFree(ptr);
                    if result != CUDA_SUCCESS {
                        return Err(NdimageError::ComputationError(format!(
                            "CUDA free failed during pool clear: {}",
                            cuda_error_string(result)
                        )));
                    }
                }
                self.total_allocated = self.total_allocated.saturating_sub(size);
            }
        }
        Ok(())
    }
}

impl Drop for CudaMemoryManager {
    fn drop(&mut self) {
        // Best effort cleanup - ignore errors during drop
        let _ = self.clear_pools();
    }
}

/// Advanced CUDA execution context with profiling and optimization
pub struct AdvancedCudaExecutor {
    context: Arc<CudaContext>,
    stream: *mut c_void,
    memory_manager: std::sync::Mutex<CudaMemoryManager>,
    execution_stats: std::sync::Mutex<ExecutionStats>,
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
        let mut stream: *mut c_void = std::ptr::null_mut();
        unsafe {
            let result = cudaStreamCreate(&mut stream);
            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "Failed to create CUDA stream: {result}"
                )));
            }
        }

        Ok(Self {
            context,
            stream,
            memory_manager: std::sync::Mutex::new(CudaMemoryManager::new(10)), // Pool up to 10 buffers per size
            execution_stats: std::sync::Mutex::new(ExecutionStats::default()),
        })
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
            phantom: std::marker::PhantomData,
        })
    }
}

/// CUDA buffer with managed lifecycle
pub struct CudaManagedBuffer<T> {
    device_ptr: *mut c_void,
    size: usize,
    byte_size: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T> CudaManagedBuffer<T> {
    pub fn copy_from_host_async(&self, data: &[T], stream: *mut c_void) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }

        unsafe {
            let result = cudaMemcpyAsync(
                self.device_ptr,
                data.as_ptr() as *const c_void,
                self.byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            );

            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA async memcpy failed: {}",
                    cuda_error_string(result)
                )));
            }
        }

        Ok(())
    }

    pub fn copy_to_host_async(&self, data: &mut [T], stream: *mut c_void) -> NdimageResult<()> {
        if data.len() != self.size {
            return Err(NdimageError::InvalidInput("Data size mismatch".to_string()));
        }

        unsafe {
            let result = cudaMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                self.device_ptr,
                self.byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            );

            if result != CUDA_SUCCESS {
                return Err(NdimageError::ComputationError(format!(
                    "CUDA async memcpy failed: {}",
                    cuda_error_string(result)
                )));
            }
        }

        Ok(())
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
            "#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n\n{}",
            cuda_source
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
    #[ignore] // Ignore by default as it requires CUDA
    fn test_cudacontext_creation() {
        let context = CudaContext::new(None);
        assert!(context.is_ok());

        if let Ok(ctx) = context {
            assert_eq!(ctx.device_id, 0);
            assert!(ctx.device_count() > 0);
        }
    }

    #[test]
    #[ignore] // Ignore by default as it requires CUDA
    fn test_cuda_buffer_allocation() {
        let buffer = CudaBuffer::<f32>::new(1024);
        assert!(buffer.is_ok());

        if let Ok(buf) = buffer {
            assert_eq!(buf.size(), 1024);
        }
    }
}
