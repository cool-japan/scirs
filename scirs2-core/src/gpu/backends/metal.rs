//! Metal GPU backend implementation for macOS
//!
//! This module provides Metal-specific implementations for GPU operations,
//! utilizing Apple's Metal framework for high-performance computing on macOS.

#![cfg(all(feature = "metal", target_os = "macos"))]

use crate::gpu::{GpuBufferImpl, GpuCompilerImpl, GpuContextImpl, GpuError, GpuKernelImpl};
use metal::{
    Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Library,
    MTLCPUCacheMode, MTLHazardTrackingMode, MTLResourceOptions, MTLSize,
};
// Import Device directly from the re-export
pub use metal::Device;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::gpu::backends::metal_mps::MPSOperations;

/// Metal storage mode configuration
#[derive(Debug, Clone, Copy)]
pub enum MetalStorageMode {
    /// Shared between CPU and GPU (unified memory on Apple Silicon)
    Shared,
    /// GPU private memory
    Private,
    /// CPU-accessible, GPU reads cached through texture cache
    Managed,
}

/// Metal buffer options
#[derive(Debug, Clone)]
pub struct MetalBufferOptions {
    pub storage_mode: MetalStorageMode,
    pub cache_mode: MTLCPUCacheMode,
    pub hazard_tracking_mode: MTLHazardTrackingMode,
}

impl Default for MetalBufferOptions {
    fn default() -> Self {
        Self {
            storage_mode: MetalStorageMode::Shared,
            cache_mode: MTLCPUCacheMode::DefaultCache,
            hazard_tracking_mode: MTLHazardTrackingMode::Default,
        }
    }
}

/// Metal pipeline configuration
#[derive(Debug, Clone)]
pub struct MetalPipelineConfig {
    pub shader_source: String,
    pub entry_point: String,
    pub use_simd_groups: bool,
    pub threadgroup_memory_length: usize,
    pub max_total_threads_per_threadgroup: usize,
}

/// Metal context implementation
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    /// Cache of compiled libraries
    library_cache: Arc<RwLock<HashMap<String, Library>>>,
    /// Device capabilities
    capabilities: MetalDeviceCapabilities,
    /// Metal Performance Shaders operations (if available)
    // MPS operations are available when Metal feature is enabled
    mps_operations: Option<Arc<MPSOperations>>,
}

/// Metal device capabilities
#[derive(Debug, Clone)]
struct MetalDeviceCapabilities {
    max_threads_per_threadgroup: usize,
    max_buffer_length: usize,
    supports_family_mac2: bool,
    supports_family_apple7: bool,
    unified_memory: bool,
}

impl MetalContext {
    /// Create a new Metal context
    pub fn new() -> Result<Self, GpuError> {
        // Get the default Metal device
        let device = Device::system_default()
            .ok_or_else(|| GpuError::BackendNotAvailable("No Metal device found".to_string()))?;

        // Create command queue with maximum command buffer count
        let command_queue = device.new_command_queue_with_max_command_buffer_count(128);

        // Detect device capabilities
        let capabilities = MetalDeviceCapabilities {
            max_threads_per_threadgroup: 1024, // Conservative default
            max_buffer_length: device.max_buffer_length() as usize,
            supports_family_mac2: device.supports_family(metal::MTLGPUFamily::Mac2),
            supports_family_apple7: device.supports_family(metal::MTLGPUFamily::Apple7),
            unified_memory: device.has_unified_memory(),
        };

        // Initialize MPS operations if available
        // MPS operations are available when Metal feature is enabled
        let mps_operations = Some(Arc::new(MPSOperations::new(
            device.clone(),
            command_queue.clone(),
        )));

        Ok(Self {
            device,
            command_queue,
            library_cache: Arc::new(RwLock::new(HashMap::new())),
            capabilities,
            // MPS operations are available when Metal feature is enabled
            mps_operations,
        })
    }

    /// Get device name
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Check if the device supports unified memory (Apple Silicon)
    pub fn has_unified_memory(&self) -> bool {
        self.capabilities.unified_memory
    }

    /// Get MPS operations interface
    // MPS operations are available when Metal feature is enabled
    pub fn mps_operations(&self) -> Option<&Arc<MPSOperations>> {
        self.mps_operations.as_ref()
    }
}

impl GpuContextImpl for MetalContext {
    fn create_buffer(&self, size: usize) -> Arc<dyn GpuBufferImpl> {
        Arc::new(MetalBuffer::new(
            &self.device,
            size,
            MetalBufferOptions::default(),
        ))
    }

    fn create_compiler(&self) -> Arc<dyn GpuCompilerImpl> {
        Arc::new(MetalCompiler::new(
            self.device.clone(),
            self.command_queue.clone(),
            self.library_cache.clone(),
        ))
    }
}

impl MetalContext {
    /// Create a buffer with specific options
    pub fn create_buffer_with_options(
        &self,
        size: usize,
        options: MetalBufferOptions,
    ) -> Arc<MetalBuffer> {
        Arc::new(MetalBuffer::new(&self.device, size, options))
    }
}

/// Metal buffer implementation with unified memory support
pub struct MetalBuffer {
    buffer: Buffer,
    size: usize,
    options: MetalBufferOptions,
}

impl MetalBuffer {
    /// Create a new Metal buffer
    fn new(device: &Device, size: usize, options: MetalBufferOptions) -> Self {
        // Convert options to Metal resource options
        let mut resource_options = MTLResourceOptions::empty();

        // Set storage mode
        match options.storage_mode {
            MetalStorageMode::Shared => {
                resource_options |= MTLResourceOptions::StorageModeShared;
            }
            MetalStorageMode::Private => {
                resource_options |= MTLResourceOptions::StorageModePrivate;
            }
            MetalStorageMode::Managed => {
                resource_options |= MTLResourceOptions::StorageModeManaged;
            }
        }

        // Set CPU cache mode
        match options.cache_mode {
            MTLCPUCacheMode::DefaultCache => {
                resource_options |= MTLResourceOptions::CPUCacheModeDefaultCache;
            }
            MTLCPUCacheMode::WriteCombined => {
                resource_options |= MTLResourceOptions::CPUCacheModeWriteCombined;
            }
        }

        // Set hazard tracking mode
        match options.hazard_tracking_mode {
            MTLHazardTrackingMode::Default => {
                // Default mode, no specific flag
            }
            MTLHazardTrackingMode::Tracked => {
                resource_options |= MTLResourceOptions::HazardTrackingModeTracked;
            }
            MTLHazardTrackingMode::Untracked => {
                resource_options |= MTLResourceOptions::HazardTrackingModeUntracked;
            }
        }

        let buffer = device.new_buffer(size as u64, resource_options);

        Self {
            buffer,
            size,
            options,
        }
    }

    /// Get the underlying Metal buffer
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl GpuBufferImpl for MetalBuffer {
    unsafe fn copy_from_host(&self, data: *const u8, size: usize) {
        assert!(size <= self.size, "Data size exceeds buffer size");
        let contents = self.buffer.contents();
        std::ptr::copy_nonoverlapping(data, contents as *mut u8, size);
    }

    unsafe fn copy_to_host(&self, data: *mut u8, size: usize) {
        assert!(size <= self.size, "Data size exceeds buffer size");
        let contents = self.buffer.contents();
        std::ptr::copy_nonoverlapping(contents as *const u8, data, size);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Metal compiler implementation
pub struct MetalCompiler {
    device: Device,
    command_queue: CommandQueue,
    /// Cache of compiled pipelines
    pipeline_cache: Arc<RwLock<HashMap<String, Arc<ComputePipelineState>>>>,
    /// Shared library cache
    library_cache: Arc<RwLock<HashMap<String, Library>>>,
}

impl MetalCompiler {
    /// Create a new Metal compiler
    fn new(
        device: Device,
        command_queue: CommandQueue,
        library_cache: Arc<RwLock<HashMap<String, Library>>>,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipeline_cache: Arc::new(RwLock::new(HashMap::new())),
            library_cache,
        }
    }

    /// Compile Metal shader source into a compute pipeline
    fn compile_source(&self, source: &str) -> Result<Arc<ComputePipelineState>, GpuError> {
        // Check cache first
        let cache_key = source.to_string();
        {
            let cache = self.pipeline_cache.read().unwrap();
            if let Some(pipeline) = cache.get(&cache_key) {
                return Ok(pipeline.clone());
            }
        }

        // Compile the shader
        let library = self
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| GpuError::KernelCompilationError(e.to_string()))?;

        // Get the main compute function
        let function = library
            .get_function("main0", None)
            .map_err(|e| GpuError::KernelCompilationError(e))?;

        // Create compute pipeline
        let pipeline_descriptor = ComputePipelineDescriptor::new();
        pipeline_descriptor.set_compute_function(Some(&function));

        let pipeline = self
            .device
            .new_compute_pipeline_state(&pipeline_descriptor)
            .map_err(|e| GpuError::KernelCompilationError(e.to_string()))?;

        let pipeline = Arc::new(pipeline);

        // Cache the compiled pipeline
        {
            let mut cache = self.pipeline_cache.write().unwrap();
            cache.insert(cache_key, pipeline.clone());
        }

        Ok(pipeline)
    }
}

impl GpuCompilerImpl for MetalCompiler {
    fn compile(&self, source: &str) -> Result<Arc<dyn GpuKernelImpl>, GpuError> {
        let pipeline = self.compile_source(source)?;
        Ok(Arc::new(MetalKernel::new(
            self.device.clone(),
            self.command_queue.clone(),
            pipeline,
        )))
    }

    fn compile_typed(
        &self,
        name: &str,
        _input_type: std::any::TypeId,
        _output_type: std::any::TypeId,
    ) -> Arc<dyn GpuKernelImpl> {
        // For typed compilation, we would generate appropriate Metal shader code
        // based on the input/output types. For now, return a stub.
        Arc::new(MetalKernel::stub(
            self.device.clone(),
            self.command_queue.clone(),
            name.to_string(),
        ))
    }
}

/// Metal kernel implementation
pub struct MetalKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline: Option<Arc<ComputePipelineState>>,
    /// Parameters bound to the kernel
    parameters: Arc<Mutex<KernelParameters>>,
}

/// Kernel parameters storage
struct KernelParameters {
    buffers: HashMap<String, Arc<dyn GpuBufferImpl>>,
    scalars: HashMap<String, ScalarValue>,
}

/// Scalar parameter value
enum ScalarValue {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl MetalKernel {
    /// Create a new Metal kernel with a compiled pipeline
    fn new(
        device: Device,
        command_queue: CommandQueue,
        pipeline: Arc<ComputePipelineState>,
    ) -> Self {
        Self {
            device,
            command_queue,
            pipeline: Some(pipeline),
            parameters: Arc::new(Mutex::new(KernelParameters {
                buffers: HashMap::new(),
                scalars: HashMap::new(),
            })),
        }
    }

    /// Create a stub kernel for typed compilation
    fn stub(device: Device, command_queue: CommandQueue, _name: String) -> Self {
        Self {
            device,
            command_queue,
            pipeline: None,
            parameters: Arc::new(Mutex::new(KernelParameters {
                buffers: HashMap::new(),
                scalars: HashMap::new(),
            })),
        }
    }
}

impl GpuKernelImpl for MetalKernel {
    fn set_buffer(&self, name: &str, buffer: &Arc<dyn GpuBufferImpl>) {
        let mut params = self.parameters.lock().unwrap();
        params.buffers.insert(name.to_string(), buffer.clone());
    }

    fn set_u32(&self, name: &str, value: u32) {
        let mut params = self.parameters.lock().unwrap();
        params
            .scalars
            .insert(name.to_string(), ScalarValue::U32(value));
    }

    fn set_i32(&self, name: &str, value: i32) {
        let mut params = self.parameters.lock().unwrap();
        params
            .scalars
            .insert(name.to_string(), ScalarValue::I32(value));
    }

    fn set_f32(&self, name: &str, value: f32) {
        let mut params = self.parameters.lock().unwrap();
        params
            .scalars
            .insert(name.to_string(), ScalarValue::F32(value));
    }

    fn set_f64(&self, name: &str, value: f64) {
        let mut params = self.parameters.lock().unwrap();
        params
            .scalars
            .insert(name.to_string(), ScalarValue::F64(value));
    }

    fn dispatch(&self, work_groups: [u32; 3]) {
        let Some(pipeline) = &self.pipeline else {
            eprintln!("Warning: Attempting to dispatch stub kernel");
            return;
        };

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set the compute pipeline
        encoder.set_compute_pipeline_state(pipeline);

        // Bind parameters
        let params = self.parameters.lock().unwrap();

        // Bind buffers (assuming standard binding indices for now)
        let mut buffer_index = 0;
        for (_name, buffer) in &params.buffers {
            // Downcast to MetalBuffer to get the underlying Metal buffer
            if let Some(metal_buffer) = buffer.as_any().downcast_ref::<MetalBuffer>() {
                encoder.set_buffer(buffer_index, Some(metal_buffer.metal_buffer()), 0);
                buffer_index += 1;
            }
        }

        // For scalar parameters, we would typically pack them into a constant buffer
        // For now, we'll skip this part as it requires more sophisticated parameter layout

        // Dispatch the kernel
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(
            work_groups[0] as u64,
            work_groups[1] as u64,
            work_groups[2] as u64,
        );

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);

        // Finish encoding and commit
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_creation() {
        // This test will only pass on macOS with Metal support
        if !cfg!(target_os = "macos") {
            return;
        }

        match MetalContext::new() {
            Ok(_) => {
                // Successfully created Metal context
            }
            Err(e) => {
                // Metal might not be available in CI environment
                eprintln!("Metal context creation failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_metal_buffer_creation() {
        if !cfg!(target_os = "macos") {
            return;
        }

        let context = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => return, // Skip test if Metal not available
        };

        let buffer = context.create_buffer(1024);
        // Buffer should be created successfully
        assert!(Arc::strong_count(&buffer) == 1);
    }
}
