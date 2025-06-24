//! GPU acceleration module for scirs2-core
//!
//! This module provides hardware acceleration support across different GPU backends.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

pub mod async_execution;
pub mod auto_tuning;
pub mod backends;
pub mod benchmarks;
pub mod heterogeneous;
pub mod kernels;
pub mod tensor_cores;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm backend
    Rocm,
    /// WebGPU backend
    Wgpu,
    /// Apple Metal backend
    Metal,
    /// OpenCL backend
    OpenCL,
    /// CPU fallback
    Cpu,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::preferred()
    }
}

impl GpuBackend {
    /// Get the preferred GPU backend for the current system
    pub fn preferred() -> Self {
        // Use the backend detection system to find the optimal backend
        // This will properly detect available GPUs and fall back to CPU if needed
        match backends::initialize_optimal_backend() {
            Ok(backend) => {
                // If we get a non-CPU backend, verify it's actually usable
                if backend != GpuBackend::Cpu {
                    // Check if we can actually create a context with this backend
                    // For now, since implementations are stubs, fall back to CPU
                    #[cfg(not(test))]
                    {
                        // In non-test environments, we don't have real GPU implementations yet
                        return GpuBackend::Cpu;
                    }
                    #[cfg(test)]
                    {
                        // In tests, we can pretend the backend works
                        return backend;
                    }
                }
                backend
            }
            Err(_) => {
                // If detection fails entirely, use CPU
                GpuBackend::Cpu
            }
        }
    }

    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            // In a real implementation, we would check if the backend is available
            // For now, just return a default value based on feature flags
            GpuBackend::Cuda => cfg!(feature = "cuda"),
            GpuBackend::Rocm => cfg!(feature = "rocm"),
            GpuBackend::Wgpu => cfg!(feature = "wgpu"),
            GpuBackend::Metal => cfg!(all(feature = "metal", target_os = "macos")),
            GpuBackend::OpenCL => cfg!(feature = "opencl"),
            GpuBackend::Cpu => true,
        }
    }
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::Wgpu => write!(f, "WebGPU"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

use crate::error::{CoreError, ErrorContext, ErrorLocation};

/// Error type for GPU operations
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// Backend is not available
    #[error("GPU backend {0} is not available")]
    BackendNotAvailable(String),

    /// Backend is not supported
    #[error("GPU backend {0} is not supported")]
    UnsupportedBackend(GpuBackend),

    /// Backend is not supported for a kernel
    #[error("GPU backend {0:?} is not supported for this kernel")]
    BackendNotSupported(GpuBackend),

    /// Backend is not implemented yet
    #[error("GPU backend {0} is not implemented yet")]
    BackendNotImplemented(GpuBackend),

    /// Out of memory
    #[error("GPU out of memory: {0}")]
    OutOfMemory(String),

    /// Kernel compilation error
    #[error("Kernel compilation error: {0}")]
    KernelCompilationError(String),

    /// Kernel execution error
    #[error("Kernel execution error: {0}")]
    KernelExecutionError(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Kernel not found
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Specialization not supported
    #[error("Kernel specialization not supported")]
    SpecializationNotSupported,

    /// Unsupported data type
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(kernels::DataType),

    /// Other error
    #[error("{0}")]
    Other(String),
}

/// GPU device abstraction
pub struct GpuDevice {
    backend: GpuBackend,
    device_id: usize,
}

impl GpuDevice {
    /// Create a new GPU device
    pub fn new(backend: GpuBackend, device_id: usize) -> Self {
        Self { backend, device_id }
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get the device ID
    pub fn id(&self) -> usize {
        self.device_id
    }

    /// Compile a kernel from source
    pub fn compile_kernel(&self, _source: &str, entry_point: &str) -> Result<GpuKernel, GpuError> {
        // Placeholder implementation
        Ok(GpuKernel {
            backend: self.backend,
            entry_point: entry_point.to_string(),
        })
    }
}

/// GPU kernel abstraction
pub struct GpuKernel {
    backend: GpuBackend,
    entry_point: String,
}

impl GpuKernel {
    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get the entry point name
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

/// Convert GPU errors to core errors with semantic preservation
impl From<GpuError> for CoreError {
    fn from(err: GpuError) -> Self {
        match err {
            GpuError::BackendNotAvailable(backend) => CoreError::ComputationError(
                ErrorContext::new(format!("GPU backend {} is not available", backend))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::UnsupportedBackend(backend) => CoreError::NotImplementedError(
                ErrorContext::new(format!("GPU backend {} is not supported", backend))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::BackendNotSupported(backend) => CoreError::NotImplementedError(
                ErrorContext::new(format!(
                    "GPU backend {:?} is not supported for this kernel",
                    backend
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::BackendNotImplemented(backend) => CoreError::NotImplementedError(
                ErrorContext::new(format!("GPU backend {} is not implemented yet", backend))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::OutOfMemory(details) => CoreError::MemoryError(
                ErrorContext::new(format!("GPU out of memory: {}", details))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::KernelCompilationError(msg) => CoreError::ComputationError(
                ErrorContext::new(format!("Kernel compilation failed: {}", msg))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::KernelExecutionError(msg) => CoreError::ComputationError(
                ErrorContext::new(format!("Kernel execution failed: {}", msg))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::InvalidParameter(msg) => CoreError::InvalidArgument(
                ErrorContext::new(format!("Invalid GPU parameter: {}", msg))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::KernelNotFound(name) => CoreError::ComputationError(
                ErrorContext::new(format!("GPU kernel not found: {}", name))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::SpecializationNotSupported => CoreError::NotImplementedError(
                ErrorContext::new("Kernel specialization not supported".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::UnsupportedDataType(dtype) => CoreError::TypeError(
                ErrorContext::new(format!("Unsupported GPU data type: {:?}", dtype))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            GpuError::Other(msg) => CoreError::ComputationError(
                ErrorContext::new(msg).with_location(ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}

/// Trait for types that can be used with GPU operations
pub trait GpuDataType: Copy + Send + Sync + 'static {}

// Implement for common types
impl GpuDataType for f32 {}
impl GpuDataType for f64 {}
impl GpuDataType for i32 {}
impl GpuDataType for u32 {}
impl GpuDataType for u8 {}
impl GpuDataType for i8 {}
impl GpuDataType for u16 {}
impl GpuDataType for i16 {}
impl GpuDataType for u64 {}
impl GpuDataType for i64 {}

/// GPU buffer
pub struct GpuBuffer<T: GpuDataType> {
    inner: Arc<dyn GpuBufferImpl>,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T: GpuDataType> GpuBuffer<T> {
    /// Create a new buffer with the given size
    pub(crate) fn new(inner: Arc<dyn GpuBufferImpl>, size: usize) -> Self {
        Self {
            inner,
            size,
            _phantom: PhantomData,
        }
    }

    /// Get the size of the buffer in elements
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy data from the host to the device
    pub fn copy_from_host(&self, data: &[T]) {
        assert!(data.len() <= self.size, "Data size exceeds buffer size");
        unsafe {
            self.inner
                .copy_from_host(data.as_ptr() as *const u8, std::mem::size_of_val(data));
        }
    }

    /// Copy data from the device to the host
    pub fn copy_to_host(&self, data: &mut [T]) {
        assert!(data.len() <= self.size, "Data size exceeds buffer size");
        unsafe {
            self.inner
                .copy_to_host(data.as_mut_ptr() as *mut u8, std::mem::size_of_val(data));
        }
    }

    /// Convert the buffer contents to a vector
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = vec![unsafe { std::mem::zeroed() }; self.size];
        self.copy_to_host(&mut result);
        result
    }
}

/// GPU kernel handle
pub struct GpuKernelHandle {
    inner: Arc<dyn GpuKernelImpl>,
}

impl GpuKernelHandle {
    /// Create a new kernel handle
    pub(crate) fn new(inner: Arc<dyn GpuKernelImpl>) -> Self {
        Self { inner }
    }

    /// Set a buffer parameter
    pub fn set_buffer<T: GpuDataType>(&self, name: &str, buffer: &GpuBuffer<T>) {
        self.inner.set_buffer(name, &buffer.inner);
    }

    /// Set a u32 parameter
    pub fn set_u32(&self, name: &str, value: u32) {
        self.inner.set_u32(name, value);
    }

    /// Set an i32 parameter
    pub fn set_i32(&self, name: &str, value: i32) {
        self.inner.set_i32(name, value);
    }

    /// Set an f32 parameter
    pub fn set_f32(&self, name: &str, value: f32) {
        self.inner.set_f32(name, value);
    }

    /// Set an f64 parameter
    pub fn set_f64(&self, name: &str, value: f64) {
        self.inner.set_f64(name, value);
    }

    /// Dispatch the kernel with the given work group counts
    pub fn dispatch(&self, work_groups: [u32; 3]) {
        self.inner.dispatch(work_groups);
    }
}

/// GPU compiler for dynamic kernels
pub struct GpuCompiler {
    inner: Arc<dyn GpuCompilerImpl>,
}

impl GpuCompiler {
    /// Create a new compiler
    pub(crate) fn new(inner: Arc<dyn GpuCompilerImpl>) -> Self {
        Self { inner }
    }

    /// Compile a kernel from source
    pub fn compile(&self, source: &str) -> Result<GpuKernelHandle, GpuError> {
        let kernel = self.inner.compile(source)?;
        Ok(GpuKernelHandle::new(kernel))
    }

    /// Compile a kernel for the specified input and output types
    pub fn compile_kernel<I: GpuDataType, O: GpuDataType>(&self, name: &str) -> GpuKernelHandle {
        let kernel = self.inner.compile_typed(
            name,
            std::any::TypeId::of::<I>(),
            std::any::TypeId::of::<O>(),
        );
        GpuKernelHandle::new(kernel)
    }
}

/// GPU context for managing GPU resources and operations
pub struct GpuContext {
    inner: Arc<dyn GpuContextImpl>,
    backend: GpuBackend,
    kernel_registry: kernels::KernelRegistry,
}

impl GpuContext {
    /// Create a new GPU context with the specified backend
    pub fn new(backend: GpuBackend) -> Result<Self, GpuError> {
        // First check if the backend is available at compile time
        if !backend.is_available() {
            return Err(GpuError::BackendNotAvailable(backend.to_string()));
        }

        // For non-CPU backends, also check runtime availability
        if backend != GpuBackend::Cpu {
            let detection_result = backends::detect_gpu_backends();
            let backend_available = detection_result
                .devices
                .iter()
                .any(|d| d.backend == backend && d.backend != GpuBackend::Cpu);

            if !backend_available {
                return Err(GpuError::BackendNotAvailable(format!(
                    "{} (no devices detected at runtime)",
                    backend
                )));
            }
        }

        let inner = match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // This is just a stub - in a real implementation, we would use the cuda crate
                    // to create a context and return it
                    #[cfg(test)]
                    {
                        // For testing, we can use a mock implementation
                        Arc::new(CpuContext::new()) as Arc<dyn GpuContextImpl>
                    }
                    #[cfg(not(test))]
                    {
                        return Err(GpuError::BackendNotImplemented(backend));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(GpuError::UnsupportedBackend(backend));
                }
            }
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    // This is just a stub - in a real implementation, we would use the hip-sys crate
                    // to create a ROCm context and return it
                    #[cfg(test)]
                    {
                        // For testing, we can use a mock implementation
                        Arc::new(CpuContext::new()) as Arc<dyn GpuContextImpl>
                    }
                    #[cfg(not(test))]
                    {
                        return Err(GpuError::BackendNotImplemented(backend));
                    }
                }
                #[cfg(not(feature = "rocm"))]
                {
                    return Err(GpuError::UnsupportedBackend(backend));
                }
            }
            GpuBackend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    // This is just a stub - in a real implementation, we would use the wgpu crate
                    // to create a context and return it
                    return Err(GpuError::BackendNotImplemented(backend));
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    return Err(GpuError::UnsupportedBackend(backend));
                }
            }
            GpuBackend::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    use crate::gpu::backends::metal::MetalContext;
                    match MetalContext::new() {
                        Ok(ctx) => Arc::new(ctx) as Arc<dyn GpuContextImpl>,
                        Err(e) => return Err(e),
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    return Err(GpuError::UnsupportedBackend(backend));
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    // This is just a stub - in a real implementation, we would use the opencl crate
                    // to create a context and return it
                    return Err(GpuError::BackendNotImplemented(backend));
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(GpuError::UnsupportedBackend(backend));
                }
            }
            GpuBackend::Cpu => Arc::new(CpuContext::new()) as Arc<dyn GpuContextImpl>,
        };

        Ok(Self {
            inner,
            backend,
            kernel_registry: kernels::KernelRegistry::with_default_kernels(),
        })
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get the backend name
    pub const fn backend_name(&self) -> &str {
        match self.backend {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Rocm => "ROCm",
            GpuBackend::Wgpu => "WebGPU",
            GpuBackend::Metal => "Metal",
            GpuBackend::OpenCL => "OpenCL",
            GpuBackend::Cpu => "CPU",
        }
    }

    /// Create a buffer with the given size
    pub fn create_buffer<T: GpuDataType>(&self, size: usize) -> GpuBuffer<T> {
        let inner = self.inner.create_buffer(size * std::mem::size_of::<T>());
        GpuBuffer::new(inner, size)
    }

    /// Create a buffer from a slice
    pub fn create_buffer_from_slice<T: GpuDataType>(&self, data: &[T]) -> GpuBuffer<T> {
        let buffer = self.create_buffer::<T>(data.len());
        buffer.copy_from_host(data);
        buffer
    }

    /// Execute a function with a compiler
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&GpuCompiler) -> R,
    {
        let compiler = GpuCompiler::new(self.inner.create_compiler());
        f(&compiler)
    }

    /// Get a kernel from the registry
    pub fn get_kernel(&self, name: &str) -> Result<GpuKernelHandle, GpuError> {
        let kernel = self
            .kernel_registry
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;

        let kernel_source = kernel.source_for_backend(self.backend)?;
        let metadata = kernel.metadata();

        let handle = self.compile_kernel_with_metadata(&kernel_source, &metadata)?;
        Ok(handle)
    }

    /// Get a specialized kernel from the registry
    pub fn get_specialized_kernel(
        &self,
        name: &str,
        params: &kernels::KernelParams,
    ) -> Result<GpuKernelHandle, GpuError> {
        let specialized = self.kernel_registry.get_specialized(name, params)?;
        let kernel_source = specialized.source_for_backend(self.backend)?;
        let metadata = specialized.metadata();

        let handle = self.compile_kernel_with_metadata(&kernel_source, &metadata)?;
        Ok(handle)
    }

    /// Compile a kernel with metadata
    fn compile_kernel_with_metadata(
        &self,
        source: &str,
        _metadata: &kernels::KernelMetadata,
    ) -> Result<GpuKernelHandle, GpuError> {
        self.execute(|compiler| compiler.compile(source))
    }

    /// Get available memory on the device
    pub fn get_available_memory(&self) -> Option<usize> {
        // In a real implementation, this would query the device
        // For now, return a placeholder value
        Some(1024 * 1024 * 1024) // 1GB
    }

    /// Get total memory on the device
    pub fn get_total_memory(&self) -> Option<usize> {
        // In a real implementation, this would query the device
        // For now, return a placeholder value
        Some(4 * 1024 * 1024 * 1024) // 4GB
    }
}

// The following trait definitions would be implemented by backend-specific
// code in a real implementation

/// GPU buffer implementation trait
pub(crate) trait GpuBufferImpl: Send + Sync {
    /// Copy data from host to device
    unsafe fn copy_from_host(&self, data: *const u8, size: usize);

    /// Copy data from device to host
    unsafe fn copy_to_host(&self, data: *mut u8, size: usize);

    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// GPU kernel implementation trait
pub(crate) trait GpuKernelImpl: Send + Sync {
    /// Set a buffer parameter
    fn set_buffer(&self, name: &str, buffer: &Arc<dyn GpuBufferImpl>);

    /// Set a u32 parameter
    fn set_u32(&self, name: &str, value: u32);

    /// Set an i32 parameter
    fn set_i32(&self, name: &str, value: i32);

    /// Set an f32 parameter
    fn set_f32(&self, name: &str, value: f32);

    /// Set an f64 parameter
    fn set_f64(&self, name: &str, value: f64);

    /// Dispatch the kernel
    fn dispatch(&self, work_groups: [u32; 3]);
}

/// GPU compiler implementation trait
pub(crate) trait GpuCompilerImpl: Send + Sync {
    /// Compile a kernel from source
    fn compile(&self, source: &str) -> Result<Arc<dyn GpuKernelImpl>, GpuError>;

    /// Compile a typed kernel
    fn compile_typed(
        &self,
        name: &str,
        input_type: std::any::TypeId,
        output_type: std::any::TypeId,
    ) -> Arc<dyn GpuKernelImpl>;
}

/// GPU context implementation trait
pub(crate) trait GpuContextImpl: Send + Sync {
    /// Create a buffer
    fn create_buffer(&self, size: usize) -> Arc<dyn GpuBufferImpl>;

    /// Create a compiler
    fn create_compiler(&self) -> Arc<dyn GpuCompilerImpl>;
}

// CPU fallback implementation

/// CPU context implementation
struct CpuContext;

impl CpuContext {
    /// Create a new CPU context
    fn new() -> Self {
        Self
    }
}

impl GpuContextImpl for CpuContext {
    fn create_buffer(&self, size: usize) -> Arc<dyn GpuBufferImpl> {
        Arc::new(CpuBuffer::new(size))
    }

    fn create_compiler(&self) -> Arc<dyn GpuCompilerImpl> {
        Arc::new(CpuCompiler)
    }
}

/// CPU buffer implementation
struct CpuBuffer {
    data: Vec<u8>,
}

impl CpuBuffer {
    /// Create a new CPU buffer
    fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }
}

impl GpuBufferImpl for CpuBuffer {
    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        let mut_self = self as *const Self as *mut Self;
        let data_ptr = (*mut_self).data.as_mut_ptr();
        std::ptr::copy_nonoverlapping(data, data_ptr, size);
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        let data_ptr = self.data.as_ptr();
        std::ptr::copy_nonoverlapping(data_ptr, data, size);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// CPU compiler implementation
struct CpuCompiler;

impl GpuCompilerImpl for CpuCompiler {
    fn compile(&self, _source: &str) -> Result<Arc<dyn GpuKernelImpl>, GpuError> {
        // In a real implementation, we would parse and execute the kernel
        // For now, just return a dummy implementation
        Ok(Arc::new(CpuKernel))
    }

    fn compile_typed(
        &self,
        _name: &str,
        _input_type: std::any::TypeId,
        _output_type: std::any::TypeId,
    ) -> Arc<dyn GpuKernelImpl> {
        // In a real implementation, we would select an appropriate implementation
        // For now, just return a dummy implementation
        Arc::new(CpuKernel)
    }
}

/// CPU kernel implementation
struct CpuKernel;

impl GpuKernelImpl for CpuKernel {
    fn set_buffer(&self, _name: &str, _buffer: &Arc<dyn GpuBufferImpl>) {
        // In a real implementation, we would store the buffer
    }

    fn set_u32(&self, _name: &str, _value: u32) {
        // In a real implementation, we would store the value
    }

    fn set_i32(&self, _name: &str, _value: i32) {
        // In a real implementation, we would store the value
    }

    fn set_f32(&self, _name: &str, _value: f32) {
        // In a real implementation, we would store the value
    }

    fn set_f64(&self, _name: &str, _value: f64) {
        // In a real implementation, we would store the value
    }

    fn dispatch(&self, _work_groups: [u32; 3]) {
        // In a real implementation, we would execute the kernel
    }
}

// In a real implementation, we would have implementations for other backends
// such as CUDA, WebGPU, Metal, and OpenCL.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_preferred() {
        let backend = GpuBackend::preferred();
        // Should return a valid backend
        match backend {
            GpuBackend::Cuda
            | GpuBackend::Rocm
            | GpuBackend::Wgpu
            | GpuBackend::Metal
            | GpuBackend::OpenCL
            | GpuBackend::Cpu => {}
        }
    }

    #[test]
    fn test_gpu_backend_default() {
        let backend = GpuBackend::default();
        assert_eq!(backend, GpuBackend::preferred());
    }

    #[test]
    fn test_gpu_backend_is_available() {
        let backend = GpuBackend::Cpu;
        assert!(backend.is_available());

        // Test other backends based on feature flags
        #[cfg(feature = "cuda")]
        assert!(GpuBackend::Cuda.is_available());
        #[cfg(not(feature = "cuda"))]
        assert!(!GpuBackend::Cuda.is_available());

        #[cfg(feature = "rocm")]
        assert!(GpuBackend::Rocm.is_available());
        #[cfg(not(feature = "rocm"))]
        assert!(!GpuBackend::Rocm.is_available());

        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert!(GpuBackend::Metal.is_available());
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        assert!(!GpuBackend::Metal.is_available());
    }

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Cuda.to_string(), "CUDA");
        assert_eq!(GpuBackend::Rocm.to_string(), "ROCm");
        assert_eq!(GpuBackend::Wgpu.to_string(), "WebGPU");
        assert_eq!(GpuBackend::Metal.to_string(), "Metal");
        assert_eq!(GpuBackend::OpenCL.to_string(), "OpenCL");
        assert_eq!(GpuBackend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn test_gpu_error_from_conversion() {
        let gpu_error = GpuError::BackendNotAvailable("CUDA".to_string());
        let core_error: CoreError = gpu_error.into();
        match core_error {
            CoreError::ComputationError(_) => {}
            _ => panic!("Expected ComputationError"),
        }

        let gpu_error = GpuError::OutOfMemory("8GB required".to_string());
        let core_error: CoreError = gpu_error.into();
        match core_error {
            CoreError::MemoryError(_) => {}
            _ => panic!("Expected MemoryError"),
        }

        let gpu_error = GpuError::InvalidParameter("batch_size must be > 0".to_string());
        let core_error: CoreError = gpu_error.into();
        match core_error {
            CoreError::InvalidArgument(_) => {}
            _ => panic!("Expected InvalidArgument"),
        }

        let gpu_error = GpuError::UnsupportedDataType(kernels::DataType::Float16);
        let core_error: CoreError = gpu_error.into();
        match core_error {
            CoreError::TypeError(_) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_gpu_data_type_trait() {
        // Test that various types implement GpuDataType
        fn assert_gpu_data_type<T: GpuDataType>() {}

        assert_gpu_data_type::<f32>();
        assert_gpu_data_type::<f64>();
        assert_gpu_data_type::<i32>();
        assert_gpu_data_type::<u32>();
        assert_gpu_data_type::<u8>();
        assert_gpu_data_type::<i8>();
        assert_gpu_data_type::<u16>();
        assert_gpu_data_type::<i16>();
        assert_gpu_data_type::<u64>();
        assert_gpu_data_type::<i64>();
    }

    #[test]
    fn test_gpu_buffer_creation() {
        let inner = Arc::new(CpuBuffer::new(100));
        let buffer = GpuBuffer::<f32>::new(inner, 25);

        assert_eq!(buffer.len(), 25);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_gpu_buffer_empty() {
        let inner = Arc::new(CpuBuffer::new(0));
        let buffer = GpuBuffer::<f32>::new(inner, 0);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_gpu_buffer_copy_operations() {
        let inner = Arc::new(CpuBuffer::new(16));
        let buffer = GpuBuffer::<f32>::new(inner, 4);

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        buffer.copy_from_host(&data);

        let mut result = vec![0.0f32; 4];
        buffer.copy_to_host(&mut result);

        assert_eq!(result, data);
    }

    #[test]
    fn test_gpu_buffer_to_vec() {
        let inner = Arc::new(CpuBuffer::new(12));
        let buffer = GpuBuffer::<f32>::new(inner, 3);

        let data = vec![5.0f32, 6.0, 7.0];
        buffer.copy_from_host(&data);

        let result = buffer.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    #[should_panic(expected = "Data size exceeds buffer size")]
    fn test_gpu_buffer_copy_from_host_overflow() {
        let inner = Arc::new(CpuBuffer::new(8));
        let buffer = GpuBuffer::<f32>::new(inner, 2);

        let data = vec![1.0f32, 2.0, 3.0]; // 3 elements > 2 buffer size
        buffer.copy_from_host(&data);
    }

    #[test]
    #[should_panic(expected = "Data size exceeds buffer size")]
    fn test_gpu_buffer_copy_to_host_overflow() {
        let inner = Arc::new(CpuBuffer::new(8));
        let buffer = GpuBuffer::<f32>::new(inner, 2);

        let mut data = vec![0.0f32; 3]; // 3 elements > 2 buffer size
        buffer.copy_to_host(&mut data);
    }

    #[test]
    fn test_gpu_kernel_handle() {
        let kernel = Arc::new(CpuKernel);
        let handle = GpuKernelHandle::new(kernel);

        // Test setting various parameter types
        let buffer = GpuBuffer::<f32>::new(Arc::new(CpuBuffer::new(16)), 4);
        handle.set_buffer("input", &buffer);
        handle.set_u32("size", 100);
        handle.set_i32("offset", -5);
        handle.set_f32("scale", 2.5);
        handle.set_f64("precision", 0.0001);

        // Test dispatch
        handle.dispatch([16, 8, 1]);
    }

    #[test]
    fn test_gpu_context_cpu_backend() {
        let context = GpuContext::new(GpuBackend::Cpu).unwrap();
        assert_eq!(context.backend(), GpuBackend::Cpu);
        assert_eq!(context.backend_name(), "CPU");

        // Test memory query methods
        assert_eq!(context.get_available_memory(), Some(1024 * 1024 * 1024));
        assert_eq!(context.get_total_memory(), Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_gpu_context_buffer_creation() {
        let context = GpuContext::new(GpuBackend::Cpu).unwrap();

        let buffer = context.create_buffer::<f32>(100);
        assert_eq!(buffer.len(), 100);

        let data = vec![1.0f32; 50];
        let buffer_from_slice = context.create_buffer_from_slice(&data);
        assert_eq!(buffer_from_slice.len(), 50);

        let result = buffer_from_slice.to_vec();
        assert_eq!(result, data);
    }

    #[test]
    fn test_gpu_context_unsupported_backend() {
        // Test a backend that's not available
        #[cfg(not(feature = "cuda"))]
        {
            let result = GpuContext::new(GpuBackend::Cuda);
            assert!(result.is_err());
            match result {
                Err(GpuError::UnsupportedBackend(_)) => {}
                Err(GpuError::BackendNotAvailable(_)) => {} // Also accept this error
                Err(e) => panic!(
                    "Expected UnsupportedBackend or BackendNotAvailable error, got: {:?}",
                    e
                ),
                Ok(_) => panic!("Expected error, got Ok"),
            }
        }
    }

    #[test]
    fn test_gpu_compiler() {
        let compiler_impl = Arc::new(CpuCompiler);
        let compiler = GpuCompiler::new(compiler_impl);

        // Test compiling from source
        let kernel = compiler.compile("dummy kernel source").unwrap();
        kernel.dispatch([1, 1, 1]);

        // Test typed compilation
        let typed_kernel = compiler.compile_kernel::<f32, f32>("vector_add");
        typed_kernel.dispatch([32, 1, 1]);
    }

    #[test]
    fn test_gpu_context_execute() {
        let context = GpuContext::new(GpuBackend::Cpu).unwrap();

        let result = context.execute(|compiler| compiler.compile("test kernel").is_ok());

        assert!(result);
    }

    #[test]
    fn test_gpu_context_kernel_registry() {
        let context = GpuContext::new(GpuBackend::Cpu).unwrap();

        // Test getting a non-existent kernel
        let result = context.get_kernel("non_existent_kernel");
        assert!(result.is_err());
        match result {
            Err(GpuError::KernelNotFound(_)) => {}
            _ => panic!("Expected KernelNotFound error"),
        }
    }

    #[test]
    fn test_cpu_buffer_implementation() {
        let buffer = CpuBuffer::new(256);
        assert_eq!(buffer.data.len(), 256);

        // Test that initial data is zeroed
        assert!(buffer.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_gpu_error_display() {
        let error = GpuError::BackendNotAvailable("CUDA".to_string());
        assert_eq!(error.to_string(), "GPU backend CUDA is not available");

        let error = GpuError::OutOfMemory("allocation failed".to_string());
        assert_eq!(error.to_string(), "GPU out of memory: allocation failed");

        let error = GpuError::KernelCompilationError("syntax error".to_string());
        assert_eq!(error.to_string(), "Kernel compilation error: syntax error");

        let error = GpuError::KernelNotFound("gemm".to_string());
        assert_eq!(error.to_string(), "Kernel not found: gemm");
    }

    #[test]
    fn test_backend_equality() {
        assert_eq!(GpuBackend::Cuda, GpuBackend::Cuda);
        assert_ne!(GpuBackend::Cuda, GpuBackend::Rocm);

        // Test Clone and Copy
        let backend = GpuBackend::Metal;
        let cloned = backend;
        let copied = backend;
        assert_eq!(backend, cloned);
        assert_eq!(backend, copied);
    }

    #[test]
    fn test_backend_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(GpuBackend::Cuda);
        set.insert(GpuBackend::Rocm);
        set.insert(GpuBackend::Cuda); // Duplicate

        assert_eq!(set.len(), 2); // Should only have 2 unique entries
        assert!(set.contains(&GpuBackend::Cuda));
        assert!(set.contains(&GpuBackend::Rocm));
    }
}
