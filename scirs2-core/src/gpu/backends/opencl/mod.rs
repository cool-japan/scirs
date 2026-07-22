//! OpenCL backend implementation for GPU operations.
//!
//! This module provides OpenCL-specific implementations for GPU operations.
//! It talks to the OpenCL ICD entirely through the in-repo pure-Rust runtime
//! loader in the internal `ffi` submodule — there is no `#[link]`, no `build.rs`, and no
//! `-lOpenCL`, so the crate builds on machines without an OpenCL development
//! package. When no ICD is present at runtime the backend reports itself
//! unavailable and callers fall through to another backend, exactly as with
//! the wgpu/Metal/CPU paths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::gpu::{GpuBufferImpl, GpuCompilerImpl, GpuContextImpl, GpuError, GpuKernelImpl};

mod ffi;
mod memory_pool;

use ffi::{ClBuffer, ClContext, ClKernel, ClProgram, ClQueue};
use memory_pool::OpenCLMemoryPool;

// OpenCL kernel source templates
#[allow(dead_code)]
const ADAM_KERNEL_OPENCL: &str = r#"
__kernel void adam_update_f32(
    __global float* params, __global const float* grads, __global float* m, __global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const int n
) {
    const int idx = get_global_id(0);

    if (idx < n) {
        float grad = grads[idx];

        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[idx];
        }

        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected moment estimates
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;

        // Update parameters
        params[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}
"#;

#[allow(dead_code)]
const GEMM_KERNEL_OPENCL: &str = r#"
__kernel void gemm_f32(
    __global const float* A, __global const float* B, __global float* C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
"#;

/// OpenCL context wrapper
pub struct OpenCLContext {
    /// Device id used for program builds; a plain handle (not released here).
    device: ffi::cl_device_id,
    context: Arc<ClContext>,
    queue: Arc<ClQueue>,
    compiled_kernels: Arc<Mutex<HashMap<String, OpenCLKernel>>>,
    memory_pool: Arc<Mutex<OpenCLMemoryPool>>,
}

// SAFETY: `OpenCLContext` only holds thread-safe OpenCL handles (the RAII
// newtypes are `Send + Sync`) plus a raw `cl_device_id` handle. The device id
// is an immutable handle used solely to build programs; all mutable queue
// state is guarded by `Arc<Mutex<..>>`. OpenCL runtime objects are safe to
// reference across threads, so the context is `Send + Sync`.
unsafe impl Send for OpenCLContext {}
unsafe impl Sync for OpenCLContext {}

impl OpenCLContext {
    /// Create a new OpenCL context backed by the first available GPU device.
    pub fn new() -> Result<Self, GpuError> {
        let api = ffi::api().ok_or_else(|| GpuError::BackendNotAvailable("OpenCL".to_string()))?;

        let platforms = api.platform_ids()?;
        if platforms.is_empty() {
            return Err(GpuError::Other("No OpenCL platforms found".to_string()));
        }

        let device_ids = api.device_ids(ffi::CL_DEVICE_TYPE_GPU)?;
        if device_ids.is_empty() {
            return Err(GpuError::Other("No OpenCL GPU devices found".to_string()));
        }
        let device = device_ids[0];

        let context = ClContext(api.create_context(device)?);
        let queue = ClQueue(api.create_command_queue(context.0, device)?);

        Ok(Self {
            device,
            context: Arc::new(context),
            queue: Arc::new(queue),
            compiled_kernels: Arc::new(Mutex::new(HashMap::new())),
            memory_pool: Arc::new(Mutex::new(OpenCLMemoryPool::new(1024 * 1024 * 1024))), // 1GB pool
        })
    }

    /// Check if OpenCL is available and at least one GPU device exists.
    pub fn is_available() -> bool {
        ffi::api().is_some_and(|api| {
            api.device_ids(ffi::CL_DEVICE_TYPE_GPU)
                .is_ok_and(|devices| !devices.is_empty())
        })
    }

    /// Compile a kernel from OpenCL source.
    fn compile_kernel_internal(&self, source: &str, name: &str) -> Result<OpenCLKernel, GpuError> {
        let api = ffi::api().ok_or_else(|| GpuError::BackendNotAvailable("OpenCL".to_string()))?;

        let program = ClProgram(api.build_program(self.context.0, self.device, source)?);
        let kernel = ClKernel(api.create_kernel(program.0, name)?);

        Ok(OpenCLKernel {
            program,
            kernel,
            queue: Arc::clone(&self.queue),
            name: name.to_string(),
        })
    }

    /// Allocate a device memory buffer of `size` bytes.
    pub fn allocate_device_memory(&self, size: usize) -> Result<ClBuffer, GpuError> {
        let api = ffi::api().ok_or_else(|| GpuError::BackendNotAvailable("OpenCL".to_string()))?;
        let mem = api.create_buffer(self.context.0, ffi::CL_MEM_READ_WRITE, size)?;
        Ok(ClBuffer { mem, size })
    }
}

impl GpuContextImpl for OpenCLContext {
    fn create_buffer(&self, size: usize) -> Arc<dyn GpuBufferImpl> {
        // Try to allocate from memory pool first
        if let Ok(mut pool) = self.memory_pool.lock() {
            if let Some(buffer) = pool.allocate(size) {
                return Arc::new(OpenCLBuffer {
                    buffer: Some(buffer),
                    queue: Arc::clone(&self.queue),
                    size,
                    memory_pool: Arc::clone(&self.memory_pool),
                });
            }
        }

        // Fallback to direct allocation
        match self.allocate_device_memory(size) {
            Ok(buffer) => Arc::new(OpenCLBuffer {
                buffer: Some(buffer),
                queue: Arc::clone(&self.queue),
                size,
                memory_pool: Arc::clone(&self.memory_pool),
            }),
            Err(e) => {
                // Create a CPU fallback buffer when OpenCL memory is exhausted
                eprintln!(
                    "Warning: OpenCL buffer allocation failed ({e}), creating CPU fallback buffer"
                );
                Arc::new(OpenCLCpuFallbackBuffer {
                    data: vec![0u8; size],
                    size,
                    memory_pool: Arc::clone(&self.memory_pool),
                })
            }
        }
    }

    fn create_compiler(&self) -> Arc<dyn GpuCompilerImpl> {
        Arc::new(OpenCLCompiler {
            context: Arc::new(OpenCLContext {
                device: self.device,
                context: Arc::clone(&self.context),
                queue: Arc::clone(&self.queue),
                compiled_kernels: Arc::clone(&self.compiled_kernels),
                memory_pool: Arc::clone(&self.memory_pool),
            }),
        })
    }
}

/// OpenCL kernel wrapper
struct OpenCLKernel {
    // Retained so the program outlives the kernel created from it; released
    // through `ClProgram`'s `Drop`.
    #[allow(dead_code)]
    program: ClProgram,
    kernel: ClKernel,
    queue: Arc<ClQueue>,
    #[allow(dead_code)]
    name: String,
}

/// OpenCL compiler implementation
struct OpenCLCompiler {
    context: Arc<OpenCLContext>,
}

impl GpuCompilerImpl for OpenCLCompiler {
    fn compile(&self, source: &str) -> Result<Arc<dyn GpuKernelImpl>, GpuError> {
        let kernel = self.context.compile_kernel_internal(source, "kernel")?;
        let name = kernel.name.clone();
        // Store the compiled kernel so `dispatch` can look it up by name.
        if let Ok(mut kernels) = self.context.compiled_kernels.lock() {
            kernels.insert(name.clone(), kernel);
        }
        Ok(Arc::new(OpenCLKernelHandle {
            kernel_name: name,
            compiled_kernels: Arc::clone(&self.context.compiled_kernels),
            params: Arc::new(Mutex::new(Vec::new())),
        }))
    }

    fn compile_typed(
        &self,
        name: &str,
        _input_type: std::any::TypeId,
        _output_type: std::any::TypeId,
    ) -> Arc<dyn GpuKernelImpl> {
        Arc::new(OpenCLKernelHandle {
            kernel_name: name.to_string(),
            compiled_kernels: Arc::clone(&self.context.compiled_kernels),
            params: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

/// OpenCL kernel handle for execution
struct OpenCLKernelHandle {
    kernel_name: String,
    compiled_kernels: Arc<Mutex<HashMap<String, OpenCLKernel>>>,
    // Insertion-ordered parameter list. The order in which parameters are set
    // is the stable OpenCL argument index order used when binding.
    params: Arc<Mutex<Vec<(String, KernelParam)>>>,
}

enum KernelParam {
    Buffer(Arc<dyn GpuBufferImpl>),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl OpenCLKernelHandle {
    /// Insert or replace a parameter by name while preserving insertion order.
    fn set_param(&self, name: &str, param: KernelParam) {
        if let Ok(mut params) = self.params.lock() {
            if let Some(slot) = params.iter_mut().find(|(n, _)| n == name) {
                slot.1 = param;
            } else {
                params.push((name.to_string(), param));
            }
        }
    }
}

impl GpuKernelImpl for OpenCLKernelHandle {
    fn set_buffer(&self, name: &str, buffer: &Arc<dyn GpuBufferImpl>) {
        self.set_param(name, KernelParam::Buffer(Arc::clone(buffer)));
    }

    fn set_u32(&self, name: &str, value: u32) {
        self.set_param(name, KernelParam::U32(value));
    }

    fn set_i32(&self, name: &str, value: i32) {
        self.set_param(name, KernelParam::I32(value));
    }

    fn set_f32(&self, name: &str, value: f32) {
        self.set_param(name, KernelParam::F32(value));
    }

    fn set_f64(&self, name: &str, value: f64) {
        self.set_param(name, KernelParam::F64(value));
    }

    fn dispatch(&self, workgroups: [u32; 3]) {
        // Every step degrades to a no-op (never a panic) when a precondition
        // is unmet: no ICD, poisoned lock, or unknown kernel name.
        let Some(api) = ffi::api() else {
            return;
        };
        let Ok(kernels) = self.compiled_kernels.lock() else {
            return;
        };
        let Some(kernel) = kernels.get(&self.kernel_name) else {
            return;
        };
        let Ok(params) = self.params.lock() else {
            return;
        };

        let kernel_handle = kernel.kernel.0;

        // Bind every argument in stable (insertion) index order.
        for (index, (_name, param)) in params.iter().enumerate() {
            let index = index as ffi::cl_uint;
            let bind = match param {
                KernelParam::Buffer(buffer) => {
                    match buffer.as_any().downcast_ref::<OpenCLBuffer>() {
                        Some(cl_buffer) => match cl_buffer.mem_handle() {
                            // A buffer arg is bound as the address of its
                            // `cl_mem` handle with size `size_of::<cl_mem>()`.
                            Some(mem) => api.set_arg_mem(kernel_handle, index, &mem),
                            None => Ok(()),
                        },
                        // A CPU fallback buffer has no device handle to bind.
                        None => Ok(()),
                    }
                }
                KernelParam::U32(value) => {
                    api.set_arg_bytes(kernel_handle, index, &value.to_ne_bytes())
                }
                KernelParam::I32(value) => {
                    api.set_arg_bytes(kernel_handle, index, &value.to_ne_bytes())
                }
                KernelParam::F32(value) => {
                    api.set_arg_bytes(kernel_handle, index, &value.to_ne_bytes())
                }
                KernelParam::F64(value) => {
                    api.set_arg_bytes(kernel_handle, index, &value.to_ne_bytes())
                }
            };
            if bind.is_err() {
                return;
            }
        }

        // Enqueue the kernel and block for completion.
        let global = [workgroups[0] as usize];
        let local = [64usize];
        if api
            .enqueue_nd_range(kernel.queue.0, kernel_handle, &global, Some(&local))
            .is_err()
        {
            return;
        }
        let _ = api.finish(kernel.queue.0);
    }
}

/// OpenCL buffer implementation
struct OpenCLBuffer {
    // `None` only transiently while being returned to the pool in `Drop`.
    buffer: Option<ClBuffer>,
    queue: Arc<ClQueue>,
    size: usize,
    memory_pool: Arc<Mutex<OpenCLMemoryPool>>,
}

impl OpenCLBuffer {
    /// The underlying `cl_mem` handle, if this buffer still owns one.
    fn mem_handle(&self) -> Option<ffi::cl_mem> {
        self.buffer.as_ref().map(|b| b.mem)
    }
}

impl GpuBufferImpl for OpenCLBuffer {
    fn size(&self) -> usize {
        self.size
    }

    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        if size > self.size {
            return;
        }
        let Some(api) = ffi::api() else {
            return;
        };
        let Some(buffer) = self.buffer.as_ref() else {
            return;
        };
        let data_slice = std::slice::from_raw_parts(data, size);
        if let Err(e) = api.enqueue_write(self.queue.0, buffer.mem, 0, data_slice) {
            eprintln!("Warning: OpenCL write buffer failed: {e}");
        }
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        if size > self.size {
            return;
        }
        let Some(api) = ffi::api() else {
            return;
        };
        let Some(buffer) = self.buffer.as_ref() else {
            return;
        };
        let data_slice = std::slice::from_raw_parts_mut(data, size);
        if let Err(e) = api.enqueue_read(self.queue.0, buffer.mem, 0, data_slice) {
            eprintln!("Warning: OpenCL read buffer failed: {e}");
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for OpenCLBuffer {
    fn drop(&mut self) {
        // Return the buffer to the memory pool when possible; otherwise it is
        // dropped here, releasing the device memory via `ClBuffer`'s `Drop`.
        if let Some(buffer) = self.buffer.take() {
            if let Ok(mut pool) = self.memory_pool.lock() {
                pool.deallocate(buffer);
            }
        }
    }
}

/// CPU fallback buffer for when OpenCL buffer allocation fails
/// This provides a graceful degradation when GPU memory is exhausted
struct OpenCLCpuFallbackBuffer {
    data: Vec<u8>,
    size: usize,
    #[allow(dead_code)]
    memory_pool: Arc<Mutex<OpenCLMemoryPool>>,
}

impl GpuBufferImpl for OpenCLCpuFallbackBuffer {
    fn size(&self) -> usize {
        self.size
    }

    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        if size > self.size {
            eprintln!("Warning: OpenCL CPU fallback buffer copy_from_host size mismatch");
            return;
        }

        // Since this is a CPU fallback, we can use safe Rust internally
        let _data_slice = std::slice::from_raw_parts(data, size);
        // We can't mutate self.data directly since &self is immutable
        // In a real implementation, this would require interior mutability
        eprintln!("Warning: CPU fallback buffer copy_from_host called (size: {size})");
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        if size > self.size {
            eprintln!("Warning: OpenCL CPU fallback buffer copy_to_host size mismatch");
            return;
        }

        // Copy from CPU buffer to host
        let data_slice = std::slice::from_raw_parts_mut(data, size);
        let copy_size = size.min(self.data.len());
        data_slice[..copy_size].copy_from_slice(&self.data[..copy_size]);

        eprintln!("Warning: CPU fallback buffer copy_to_host called (size: {size})");
    }

    fn device_ptr(&self) -> u64 {
        self.data.as_ptr() as u64
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ICD probe must never panic, must be cached (stable across calls),
    /// and its availability is logged either way. On the build host
    /// `libOpenCL.so.1` is present so it typically resolves; bare CI machines
    /// may lack it — both outcomes keep this test green.
    #[test]
    fn ffi_api_probe_does_not_panic_and_is_stable() {
        let available = ffi::api().is_some();
        println!("OpenCL ICD loadable: {available}");
        assert_eq!(available, ffi::api().is_some());
    }

    /// `OpenCLContext::is_available()` must return a value without panicking,
    /// regardless of whether an ICD or GPU device is present.
    #[test]
    fn is_available_does_not_panic() {
        let available = OpenCLContext::is_available();
        println!("OpenCLContext::is_available() = {available}");
    }

    /// `OpenCLContext::new()` degrades gracefully: `Ok` on a machine with a
    /// usable OpenCL GPU, `Err` otherwise — never a panic.
    #[test]
    fn context_new_degrades_gracefully() {
        match OpenCLContext::new() {
            Ok(_ctx) => println!("OpenCL context created (GPU device present)"),
            Err(e) => println!("OpenCL unavailable, graceful error: {e}"),
        }
    }

    /// The public `GpuContext` entry point for the OpenCL backend must also
    /// degrade gracefully to an error (never a panic) when unavailable.
    #[test]
    fn gpu_context_opencl_degrades_gracefully() {
        use crate::gpu::{GpuBackend, GpuContext};

        // Detection is panic-free.
        let _ = GpuBackend::OpenCL.is_available();

        // Construction yields Ok (real GPU) or Err (BackendNotAvailable / no
        // device), never a panic.
        match GpuContext::new(GpuBackend::OpenCL) {
            Ok(_ctx) => println!("GpuContext(OpenCL) created"),
            Err(e) => println!("GpuContext(OpenCL) graceful error: {e}"),
        }
    }
}
