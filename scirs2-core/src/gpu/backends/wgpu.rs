//! WebGPU backend implementation for GPU operations
//!
//! This module provides WebGPU-specific implementations for cross-platform GPU operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::gpu::{GpuBufferImpl, GpuCompilerImpl, GpuContextImpl, GpuError, GpuKernelImpl};

#[cfg(feature = "wgpu_backend")]
#[allow(unused_imports)]
use wgpu::{
    util::DeviceExt, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, ComputePipeline, Device, DeviceDescriptor,
    Features, Instance, InstanceDescriptor, Limits, PowerPreference, Queue, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension,
};

// Fallback types for when WebGPU is not available
#[cfg(not(feature = "wgpu_backend"))]
type WgpuDevice = *mut std::ffi::c_void;
#[cfg(not(feature = "wgpu_backend"))]
type WgpuQueue = *mut std::ffi::c_void;
#[cfg(not(feature = "wgpu_backend"))]
type WgpuBuffer = *mut std::ffi::c_void;
#[cfg(not(feature = "wgpu_backend"))]
type WgpuComputePipeline = *mut std::ffi::c_void;

// WebGPU shader source templates
#[allow(dead_code)]
const ADAM_SHADER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

struct AdamUniforms {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    n: u32,
};

@group(0) @binding(4) var<uniform> uniforms: AdamUniforms;

@compute @workgroup_size(64)
#[allow(dead_code)]
fn adam_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= uniforms.n) {
        return;
    }
    
    var grad = grads[idx];
    
    // Apply weight decay
    if (uniforms.weight_decay > 0.0) {
        grad += uniforms.weight_decay * params[idx];
    }
    
    // Update biased first moment estimate
    m[idx] = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * grad;
    
    // Update biased second raw moment estimate
    v[idx] = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * grad * grad;
    
    // Compute bias-corrected moment estimates
    let m_hat = m[idx] / uniforms.bias_correction1;
    let v_hat = v[idx] / uniforms.bias_correction2;
    
    // Update parameters
    params[idx] -= uniforms.lr * m_hat / (sqrt(v_hat) + uniforms.eps);
}
"#;

#[allow(dead_code)]
const GEMM_SHADER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;

struct GemmUniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
};

@group(0) @binding(3) var<uniform> uniforms: GemmUniforms;

@compute @workgroup_size(8, 8)
#[allow(dead_code)]
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    var sum = 0.0;
    for (var k = 0u; k < uniforms.K; k++) {
        sum += matrix_a[row * uniforms.K + k] * matrix_b[k * uniforms.N + col];
    }
    
    let idx = row * uniforms.N + col;
    matrix_c[idx] = uniforms.alpha * sum + uniforms.beta * matrix_c[idx];
}
"#;

/// WebGPU context wrapper
pub struct WebGPUContext {
    #[cfg(feature = "wgpu_backend")]
    device: Arc<Device>,
    #[cfg(feature = "wgpu_backend")]
    queue: Arc<Queue>,
    #[cfg(not(feature = "wgpu_backend"))]
    device: Arc<WgpuDevice>,
    #[cfg(not(feature = "wgpu_backend"))]
    queue: Arc<WgpuQueue>,
    compiled_shaders: Arc<Mutex<HashMap<String, WebGPUShader>>>,
    memory_pool: Arc<Mutex<WebGPUMemoryPool>>,
}

// WebGPU handles are safe to send between threads when properly synchronized
unsafe impl Send for WebGPUContext {}
unsafe impl Sync for WebGPUContext {}

impl WebGPUContext {
    /// Create a new WebGPU context
    pub fn new() -> Result<Self, GpuError> {
        #[cfg(feature = "wgpu_backend")]
        {
            // Real WebGPU implementation
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });

            let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok_or_else(|| GpuError::Other("Failed to find WebGPU adapter".to_string()))?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &DeviceDescriptor {
                    label: Some("SciRS2 WebGPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            ))
            .map_err(|e| GpuError::Other(format!("{e}")))?;

            Ok(Self {
                device: Arc::new(device),
                queue: Arc::new(queue),
                compiled_shaders: Arc::new(Mutex::new(HashMap::new())),
                memory_pool: Arc::new(Mutex::new(WebGPUMemoryPool::new(1024 * 1024 * 1024))), // 1GB pool
            })
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback implementation
            let device = Self::initialize_webgpu()?;
            let queue = Self::create_queue(device)?;

            Ok(Self {
                device,
                queue,
                compiled_shaders: Arc::new(Mutex::new(HashMap::new())),
                memory_pool: Arc::new(Mutex::new(WebGPUMemoryPool::new(1024 * 1024 * 1024))), // 1GB pool
            })
        }
    }

    /// Check if WebGPU is available and working
    pub fn is_available() -> bool {
        #[cfg(feature = "wgpu_backend")]
        {
            // Real WebGPU implementation - try to create an instance and adapter
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });

            // Try to get an adapter (this is async, so we use a simple runtime check)
            pollster::block_on(async {
                instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::default(),
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .is_some()
            })
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback: return false since we don't have real WebGPU
            false
        }
    }

    /// Compile a shader from WGSL source
    fn compile_shader_internal(&self, source: &str, name: &str) -> Result<WebGPUShader, GpuError> {
        #[cfg(feature = "wgpu_backend")]
        {
            // Real WebGPU implementation
            let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(name),
                source: ShaderSource::Wgsl(source.into()),
            });

            // Extract entry point from source or use default
            let entry_point = Self::extract_entry_point(source).unwrap_or("main");

            // Create bind group layout for shader parameters
            let bind_group_layout = self.create_bind_group_layout_from_source(source, name)?;

            // Create pipeline layout with our bind group layout
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{}_layout", name)),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&format!("{}_pipeline", name)),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point,
                        compilation_options: Default::default(),
                        cache: None,
                    });

            Ok(WebGPUShader {
                pipeline: compute_pipeline,
                bind_group_layout,
                name: name.to_string(),
            })
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback implementation
            let pipeline = Self::compile_wgsl_source(source, name)?;

            Ok(WebGPUShader {
                pipeline,
                bind_group_layout: std::ptr::null_mut(),
                name: name.to_string(),
            })
        }
    }

    /// Create bind group layout from WGSL source analysis
    #[cfg(feature = "wgpu_backend")]
    fn create_bind_group_layout_from_source(
        &self,
        source: &str,
        name: &str,
    ) -> Result<BindGroupLayout, GpuError> {
        // Parse WGSL source to extract binding information
        let mut entries = Vec::new();
        let mut binding_index = 0;

        // Analyze shader source to determine bindings
        for line in source.lines() {
            let trimmed = line.trim();

            if trimmed.contains("@group(0) @binding(") {
                // Extract binding type from the line
                if trimmed.contains("var<storage, read_write>") || trimmed.contains("var<storage>")
                {
                    // Storage buffer (read-write)
                    entries.push(BindGroupLayoutEntry {
                        binding: binding_index,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                } else if trimmed.contains("var<storage, read>") {
                    // Storage buffer (read-only)
                    entries.push(BindGroupLayoutEntry {
                        binding: binding_index,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                } else if trimmed.contains("var<uniform>") {
                    // Uniform buffer
                    entries.push(BindGroupLayoutEntry {
                        binding: binding_index,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                binding_index += 1;
            }
        }

        // If no bindings found, create a simple layout for basic compute
        if entries.is_empty() {
            entries.push(BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("{}_bind_group_layout", name)),
                entries: &entries,
            });

        Ok(bind_group_layout)
    }

    /// Allocate device memory
    #[cfg(feature = "wgpu_backend")]
    pub fn allocate_device_memory(&self, size: usize) -> Result<Buffer, GpuError> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("SciRS2 Buffer"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(buffer)
    }

    /// Allocate device memory (fallback)
    #[cfg(not(feature = "wgpu_backend"))]
    pub fn allocate_device_memory_2(&self, size: usize) -> Result<WgpuBuffer, GpuError> {
        // Fallback implementation: return a simulated buffer handle
        Ok((0x1000 + size) as WgpuBuffer)
    }

    // Fallback methods for when WebGPU is not available
    #[cfg(not(feature = "wgpu_backend"))]
    fn initialize_webgpu() -> Result<WgpuDevice, GpuError> {
        // Stub implementation
        Ok(0x1 as WgpuDevice)
    }

    #[cfg(not(feature = "wgpu_backend"))]
    fn create_queue(device: WgpuDevice) -> Result<WgpuQueue, GpuError> {
        // Stub implementation
        Ok(0x2 as WgpuQueue)
    }

    #[cfg(not(feature = "wgpu_backend"))]
    fn compile_wgsl_source(source: &str, name: &str) -> Result<WgpuComputePipeline, GpuError> {
        // Stub implementation
        Ok(0x3 as WgpuComputePipeline)
    }

    /// Extract the entry point function name from WGSL source code
    fn extract_entry_point(source: &str) -> Option<&str> {
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check if this line contains @compute
            if trimmed.contains("@compute") {
                // The function might be on the same line or the next line
                let mut search_line = trimmed;
                let mut search_idx = 0;

                // If @compute and function are not on the same line, check next line
                if !search_line.contains("fn ") && search_idx + 1 < lines.len() {
                    search_idx += 1;
                    search_line = lines[search_idx].trim();
                }

                // Extract function name
                if let Some(start) = search_line.find("fn ") {
                    let remaining = &search_line[start + 3..];
                    if let Some(end) = remaining.find('(') {
                        let funcname = remaining[..end].trim();
                        return Some(funcname);
                    }
                }
            }
        }

        None
    }
}

impl GpuContextImpl for WebGPUContext {
    fn create_buffer(&self, size: usize) -> Arc<dyn GpuBufferImpl> {
        // Try to allocate from memory pool first
        if let Ok(mut pool) = self.memory_pool.lock() {
            if let Some(device_buffer) = pool.allocate(size) {
                return Arc::new(WebGPUBuffer {
                    device_buffer: Some(device_buffer),
                    #[cfg(feature = "wgpu_backend")]
                    queue: Arc::clone(&self.queue),
                    #[cfg(not(feature = "wgpu_backend"))]
                    queue: self.queue,
                    size,
                    memory_pool: Arc::clone(&self.memory_pool),
                });
            }
        }

        // Fallback to direct allocation
        let device_buffer = match self.allocate_device_memory(size) {
            Ok(buffer) => buffer,
            Err(e) => {
                // Log the WebGPU allocation failure and create a CPU fallback
                eprintln!(
                    "Warning: WebGPU buffer allocation failed ({}), creating CPU fallback buffer",
                    e
                );

                #[cfg(feature = "wgpu_backend")]
                {
                    // Create a CPU fallback buffer with minimal size for WebGPU compatibility
                    // This is a last resort when GPU memory is exhausted
                    return Arc::new(WebGPUCpuFallbackBuffer {
                        data: vec![0u8; size],
                        size,
                        memory_pool: Arc::clone(&self.memory_pool),
                    });
                }
                #[cfg(not(feature = "wgpu_backend"))]
                {
                    (0x2000 + size) as WgpuBuffer
                }
            }
        };

        Arc::new(WebGPUBuffer {
            device_buffer: Some(device_buffer),
            #[cfg(feature = "wgpu_backend")]
            queue: Arc::clone(&self.queue),
            #[cfg(not(feature = "wgpu_backend"))]
            queue: self.queue,
            size,
            memory_pool: Arc::clone(&self.memory_pool),
        })
    }

    fn create_compiler(&self) -> Arc<dyn GpuCompilerImpl> {
        Arc::new(WebGPUCompiler {
            context: Arc::new(WebGPUContext {
                memory_pool: Arc::clone(&self.memory_pool),
                compiled_shaders: Arc::clone(&self.compiled_shaders),
                #[cfg(feature = "wgpu_backend")]
                device: Arc::clone(&self.device),
                #[cfg(feature = "wgpu_backend")]
                queue: Arc::clone(&self.queue),
                #[cfg(not(feature = "wgpu_backend"))]
                device: Arc::clone(&self.device),
                #[cfg(not(feature = "wgpu_backend"))]
                queue: Arc::clone(&self.queue),
            }),
        })
    }
}

/// WebGPU shader wrapper
struct WebGPUShader {
    #[cfg(feature = "wgpu_backend")]
    pipeline: ComputePipeline,
    #[cfg(not(feature = "wgpu_backend"))]
    pipeline: WgpuComputePipeline,
    #[cfg(feature = "wgpu_backend")]
    #[allow(dead_code)]
    bind_group_layout: BindGroupLayout,
    #[cfg(not(feature = "wgpu_backend"))]
    #[allow(dead_code)]
    bind_group_layout: *mut std::ffi::c_void,
    #[allow(dead_code)]
    name: String,
}

// WebGPU shader handles are safe to send between threads when properly synchronized
unsafe impl Send for WebGPUShader {}
unsafe impl Sync for WebGPUShader {}

/// WebGPU compiler implementation
struct WebGPUCompiler {
    context: Arc<WebGPUContext>,
}

impl GpuCompilerImpl for WebGPUCompiler {
    fn compile(&self, source: &str) -> Result<Arc<dyn GpuKernelImpl>, GpuError> {
        let shader = self.context.compile_shader_internal(source, "shader")?;
        Ok(Arc::new(WebGPUKernelHandle {
            shader_name: shader.name.clone(),
            compiled_shaders: Arc::clone(&self.context.compiled_shaders),
            params: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "wgpu_backend")]
            device: Arc::clone(&self.context.device),
            #[cfg(feature = "wgpu_backend")]
            queue: Arc::clone(&self.context.queue),
            #[cfg(not(feature = "wgpu_backend"))]
            device: self.context.device,
            #[cfg(not(feature = "wgpu_backend"))]
            queue: self.context.queue,
        }))
    }

    fn compile_typed(
        &self,
        name: &str,
        _input_type: std::any::TypeId,
        _output_type: std::any::TypeId,
    ) -> Arc<dyn GpuKernelImpl> {
        Arc::new(WebGPUKernelHandle {
            shader_name: name.to_string(),
            compiled_shaders: Arc::clone(&self.context.compiled_shaders),
            params: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "wgpu_backend")]
            device: Arc::clone(&self.context.device),
            #[cfg(feature = "wgpu_backend")]
            queue: Arc::clone(&self.context.queue),
            #[cfg(not(feature = "wgpu_backend"))]
            device: self.context.device,
            #[cfg(not(feature = "wgpu_backend"))]
            queue: self.context.queue,
        })
    }
}

/// WebGPU kernel handle for execution
struct WebGPUKernelHandle {
    shader_name: String,
    compiled_shaders: Arc<Mutex<HashMap<String, WebGPUShader>>>,
    params: Arc<Mutex<HashMap<String, KernelParam>>>,
    #[cfg(feature = "wgpu_backend")]
    device: Arc<Device>,
    #[cfg(feature = "wgpu_backend")]
    queue: Arc<Queue>,
    #[cfg(not(feature = "wgpu_backend"))]
    device: WgpuDevice,
    #[cfg(not(feature = "wgpu_backend"))]
    queue: WgpuQueue,
}

enum KernelParam {
    #[allow(dead_code)]
    Buffer(Arc<dyn GpuBufferImpl>),
    #[allow(dead_code)]
    U32(u32),
    #[allow(dead_code)]
    I32(i32),
    #[allow(dead_code)]
    F32(f32),
    #[allow(dead_code)]
    F64(f64),
}

impl GpuKernelImpl for WebGPUKernelHandle {
    fn set_buffer(&self, name: &str, buffer: &Arc<dyn GpuBufferImpl>) {
        let mut params = self.params.lock().unwrap();
        params.insert(name.to_string(), KernelParam::Buffer(Arc::clone(buffer)));
    }

    fn set_u32(&self, name: &str, value: u32) {
        let mut params = self.params.lock().unwrap();
        params.insert(name.to_string(), KernelParam::U32(value));
    }

    fn set_i32(&self, name: &str, value: i32) {
        let mut params = self.params.lock().unwrap();
        params.insert(name.to_string(), KernelParam::I32(value));
    }

    fn set_f32(&self, name: &str, value: f32) {
        let mut params = self.params.lock().unwrap();
        params.insert(name.to_string(), KernelParam::F32(value));
    }

    fn set_f64(&self, name: &str, value: f64) {
        let mut params = self.params.lock().unwrap();
        params.insert(name.to_string(), KernelParam::F64(value));
    }

    fn dispatch(&self, workgroups: [u32; 3]) {
        #[cfg(feature = "wgpu_backend")]
        {
            // Real WebGPU compute dispatch
            let shaders = self.compiled_shaders.lock().unwrap();
            if let Some(shader) = shaders.get(&self.shader_name) {
                let params = self.params.lock().unwrap();

                // Create command encoder
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Compute Command Encoder"),
                        });

                // Begin compute pass
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });

                    // Set the compute pipeline
                    compute_pass.set_pipeline(&shader.pipeline);

                    // Create and set bind _groups with buffers and uniforms
                    // TODO: Fix create_bind_group_from_params method
                    // if let Ok(bind_group) =
                    //     self.create_bind_group_from_params(&shader.bind_group_layout, &params)
                    // {
                    //     compute_pass.set_bind_group(0, &bind_group, &[]);
                    // } else {
                    //     // Handle error by logging and continuing without bind group for now
                    //     eprintln!(
                    //         "Warning: Failed to create bind group for shader {}",
                    //         self.shader_name
                    //     );
                    // }

                    // Dispatch the compute shader
                    compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                }

                // Submit the command buffer
                let command_buffer = encoder.finish();
                self.queue.submit(std::iter::once(command_buffer));

                eprintln!(
                    "WebGPU compute shader {} dispatched with workgroups: {:?}",
                    self.shader_name, workgroups
                );
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback implementation - just log the execution
            eprintln!("Executing WebGPU shader {} (simulated)", self.shader_name);
            eprintln!("Work groups: {:?}", workgroups);
        }
    }
}

/// WebGPU buffer implementation
struct WebGPUBuffer {
    #[cfg(feature = "wgpu_backend")]
    device_buffer: Option<Buffer>,
    #[cfg(feature = "wgpu_backend")]
    queue: Arc<Queue>,
    #[cfg(not(feature = "wgpu_backend"))]
    device_buffer: Option<WgpuBuffer>,
    #[cfg(not(feature = "wgpu_backend"))]
    queue: WgpuQueue,
    size: usize,
    memory_pool: Arc<Mutex<WebGPUMemoryPool>>,
}

// WebGPU buffer handles are safe to send between threads when properly synchronized
// The real wgpu types (Buffer, Queue) are Send + Sync
// For fallback types (raw pointers), we assume proper synchronization is handled externally
unsafe impl Send for WebGPUBuffer {}
unsafe impl Sync for WebGPUBuffer {}

impl GpuBufferImpl for WebGPUBuffer {
    fn size(&self) -> usize {
        self.size
    }

    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        #[cfg(feature = "wgpu_backend")]
        {
            // Validate data size
            if size > self.size {
                // In unsafe context, we can't return an error, so we'll just log and return
                eprintln!(
                    "Warning: Data size {} exceeds buffer size {}",
                    size, self.size
                );
                return;
            }

            // Convert raw pointer to slice for WebGPU API
            let data_slice = std::slice::from_raw_parts(data, size);

            // Real WebGPU implementation - write data to buffer
            if let Some(ref buffer) = self.device_buffer {
                self.queue.write_buffer(buffer, 0, data_slice);
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback implementation - just validate
            if size > self.size {
                eprintln!(
                    "Warning: Data size {} exceeds buffer size {}",
                    size, self.size
                );
            }
            // In fallback mode, we just simulate the operation
        }
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        #[cfg(feature = "wgpu_backend")]
        {
            // Validate data size
            if size > self.size {
                eprintln!(
                    "Warning: Data size {} exceeds buffer size {}",
                    size, self.size
                );
                return;
            }

            // WebGPU buffer reading typically requires async operations and mapping
            // For now, we can't properly implement this in an unsafe synchronous context
            eprintln!(
                "Warning: WebGPU buffer reading requires async support - not yet implemented"
            );

            // Zero out the data as a placeholder
            let data_slice = std::slice::from_raw_parts_mut(data, size);
            data_slice.fill(0);
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            // Fallback implementation - just validate and zero out
            if size > self.size {
                eprintln!(
                    "Warning: Data size {} exceeds buffer size {}",
                    size, self.size
                );
            }

            // Zero out the data as a placeholder
            let data_slice = std::slice::from_raw_parts_mut(data, size);
            data_slice.fill(0);
        }
    }

    fn device_ptr(&self) -> u64 {
        #[cfg(feature = "wgpu_backend")]
        {
            // WebGPU doesn't expose raw device pointers, so we return a placeholder
            // In a real implementation, this might return a handle or ID
            &self.device_buffer as *const _ as u64
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            self.device_buffer as u64
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for WebGPUBuffer {
    fn drop(&mut self) {
        // Return buffer to memory pool if possible
        if let Ok(mut pool) = self.memory_pool.lock() {
            #[cfg(feature = "wgpu_backend")]
            {
                // In real implementation, would return buffer to pool
                if let Some(buffer) = self.device_buffer.take() {
                    pool.deallocate(buffer);
                }
            }
            #[cfg(not(feature = "wgpu_backend"))]
            {
                if let Some(buffer) = self.device_buffer.take() {
                    pool.deallocate(buffer);
                }
            }
        }
    }
}

/// CPU fallback buffer for when WebGPU buffer allocation fails
/// This provides a graceful degradation when GPU memory is exhausted
struct WebGPUCpuFallbackBuffer {
    data: Vec<u8>,
    size: usize,
    #[allow(dead_code)]
    memory_pool: Arc<Mutex<WebGPUMemoryPool>>,
}

impl GpuBufferImpl for WebGPUCpuFallbackBuffer {
    fn size(&self) -> usize {
        self.size
    }

    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        if size > self.size {
            eprintln!("Warning: WebGPU CPU fallback buffer copy_from_host size mismatch");
            return;
        }

        // Since this is a CPU fallback, we can use safe Rust internally
        let data_slice = std::slice::from_raw_parts(data, size);
        // We can't mutate self.data directly since &self is immutable
        // In a real implementation, this would require interior mutability
        eprintln!(
            "Warning: CPU fallback buffer copy_from_host called (size: {})",
            size
        );
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        if size > self.size {
            eprintln!("Warning: WebGPU CPU fallback buffer copy_to_host size mismatch");
            return;
        }

        // Copy from CPU buffer to host
        let data_slice = std::slice::from_raw_parts_mut(data, size);
        let copy_size = size.min(self.data.len());
        data_slice[..copy_size].copy_from_slice(&self.data[..copy_size]);

        eprintln!(
            "Warning: CPU fallback buffer copy_to_host called (size: {})",
            size
        );
    }

    fn device_ptr(&self) -> u64 {
        self.data.as_ptr() as u64
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Safety: WebGPUCpuFallbackBuffer is thread-safe since it only contains owned data
unsafe impl Send for WebGPUCpuFallbackBuffer {}
unsafe impl Sync for WebGPUCpuFallbackBuffer {}

/// WebGPU memory pool for efficient buffer management
struct WebGPUMemoryPool {
    #[cfg(feature = "wgpu_backend")]
    available_buffers: HashMap<usize, Vec<Buffer>>,
    #[cfg(not(feature = "wgpu_backend"))]
    available_buffers: HashMap<usize, Vec<WgpuBuffer>>,
    #[allow(dead_code)]
    total_size: usize,
    used_size: usize,
}

impl WebGPUMemoryPool {
    fn new(totalsize: usize) -> Self {
        Self {
            available_buffers: HashMap::new(),
            total_size: totalsize,
            used_size: 0,
        }
    }

    #[cfg(feature = "wgpu_backend")]
    fn allocate(&mut self, size: usize) -> Option<Buffer> {
        // Try to find a suitable buffer in the pool
        if let Some(buffers) = self.available_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                self.used_size += size;
                return Some(buffer);
            }
        }
        None
    }

    #[cfg(not(feature = "wgpu_backend"))]
    fn allocate(&mut self, size: usize) -> Option<WgpuBuffer> {
        // Try to find a suitable buffer in the pool
        if let Some(buffers) = self.available_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                self.used_size += size;
                return Some(buffer);
            }
        }
        None
    }

    #[cfg(feature = "wgpu_backend")]
    fn deallocate(&mut self, buffer: Buffer) {
        // Return buffer to pool
        let size = buffer.size() as usize;
        self.available_buffers
            .entry(size)
            .or_insert_with(Vec::new)
            .push(buffer);
        self.used_size = self.used_size.saturating_sub(size);
    }

    #[cfg(not(feature = "wgpu_backend"))]
    fn deallocate(&mut self, buffer: WgpuBuffer) {
        // Fallback implementation - track the buffer
        let size = 1024; // Placeholder size
        self.available_buffers
            .entry(size)
            .or_insert_with(Vec::new)
            .push(buffer);
        self.used_size = self.used_size.saturating_sub(size);
    }

    #[allow(dead_code)]
    fn get_memory_usage(&self) -> (usize, usize) {
        (self.used_size, self.total_size)
    }
}
