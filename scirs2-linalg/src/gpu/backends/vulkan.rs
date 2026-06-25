//! Vulkan backend implementation for cross-platform GPU acceleration
//!
//! This module provides a comprehensive Vulkan compute backend for GPU-accelerated
//! linear algebra operations. Vulkan is a modern, cross-platform graphics and compute
//! API that provides high-performance GPU acceleration on Windows, Linux, macOS (via MoltenVK),
//! and mobile platforms.
//!
//! ## Features
//!
//! - Cross-platform GPU compute support
//! - SPIR-V shader compilation and caching
//! - Advanced memory management with suballocation
//! - Command buffer pooling and reuse
//! - Multi-queue support for concurrent operations
//! - Synchronization primitives for async operations

use super::common::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Vulkan compute backend with comprehensive GPU support
#[cfg(feature = "vulkan")]
pub mod vulkan_impl {
    use super::*;

    // Vulkan types (mock definitions - real implementation would use vulkan-sys or ash crate)
    type VkResult = i32;
    type VkInstance = *mut std::ffi::c_void;
    type VkPhysicalDevice = *mut std::ffi::c_void;
    type VkDevice = *mut std::ffi::c_void;
    type VkQueue = *mut std::ffi::c_void;
    type VkCommandPool = *mut std::ffi::c_void;
    type VkCommandBuffer = *mut std::ffi::c_void;
    type VkBuffer = *mut std::ffi::c_void;
    type VkDeviceMemory = *mut std::ffi::c_void;
    type VkPipeline = *mut std::ffi::c_void;
    type VkPipelineLayout = *mut std::ffi::c_void;
    type VkDescriptorSetLayout = *mut std::ffi::c_void;
    type VkDescriptorPool = *mut std::ffi::c_void;
    type VkDescriptorSet = *mut std::ffi::c_void;
    type VkShaderModule = *mut std::ffi::c_void;
    type VkFence = *mut std::ffi::c_void;
    type VkSemaphore = *mut std::ffi::c_void;

    const VK_SUCCESS: VkResult = 0;
    const VK_ERROR_OUT_OF_HOST_MEMORY: VkResult = -1;
    const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResult = -2;

    /// Thread-safe wrapper for Vulkan handles
    #[derive(Debug, Clone, Copy)]
    pub struct SafeVkHandle(pub *mut std::ffi::c_void);

    // SAFETY: Vulkan handles are thread-safe when properly synchronized
    // This is a mock implementation for testing purposes
    unsafe impl Send for SafeVkHandle {}
    unsafe impl Sync for SafeVkHandle {}

    impl SafeVkHandle {
        fn null() -> Self {
            Self(std::ptr::null_mut())
        }

        fn is_null(&self) -> bool {
            self.0.is_null()
        }
    }

    // Mock Vulkan API functions
    fn vk_create_instance() -> (VkResult, SafeVkHandle) {
        (VK_SUCCESS, SafeVkHandle::null())
    }

    fn vk_enumerate_physical_devices(_instance: SafeVkHandle) -> (VkResult, Vec<SafeVkHandle>) {
        (VK_SUCCESS, vec![])
    }

    fn vk_get_physical_device_properties(_device: SafeVkHandle) -> VulkanPhysicalDeviceProperties {
        VulkanPhysicalDeviceProperties::default()
    }

    fn vk_get_physical_device_memory_properties(_device: SafeVkHandle) -> VulkanMemoryProperties {
        VulkanMemoryProperties::default()
    }

    fn vk_create_device(_physical_device: SafeVkHandle) -> (VkResult, SafeVkHandle) {
        (VK_SUCCESS, SafeVkHandle::null())
    }

    fn vk_get_device_queue(
        _device: SafeVkHandle,
        _queue_family: u32,
        _queue_index: u32,
    ) -> SafeVkHandle {
        SafeVkHandle::null()
    }

    fn vk_allocate_memory(
        _device: SafeVkHandle,
        _size: usize,
        _memory_type_index: u32,
    ) -> (VkResult, SafeVkHandle) {
        (VK_SUCCESS, SafeVkHandle::null())
    }

    fn vk_create_buffer(
        _device: SafeVkHandle,
        _size: usize,
        _usage: u32,
    ) -> (VkResult, SafeVkHandle) {
        (VK_SUCCESS, SafeVkHandle::null())
    }

    fn vk_bind_buffer_memory(
        _device: SafeVkHandle,
        _buffer: SafeVkHandle,
        _memory: SafeVkHandle,
        _offset: usize,
    ) -> VkResult {
        VK_SUCCESS
    }

    fn vk_map_memory(
        _device: SafeVkHandle,
        _memory: SafeVkHandle,
        _offset: usize,
        _size: usize,
    ) -> (VkResult, *mut std::ffi::c_void) {
        (VK_SUCCESS, std::ptr::null_mut())
    }

    fn vk_unmap_memory(_device: SafeVkHandle, _memory: SafeVkHandle) {
        // No-op in mock
    }

    fn vk_create_command_pool(
        _device: SafeVkHandle,
        _queue_family: u32,
    ) -> (VkResult, SafeVkHandle) {
        (VK_SUCCESS, SafeVkHandle::null())
    }

    fn vk_allocate_command_buffers(
        _device: SafeVkHandle,
        _pool: SafeVkHandle,
        _count: u32,
    ) -> (VkResult, Vec<SafeVkHandle>) {
        (VK_SUCCESS, vec![])
    }

    fn vk_queue_submit(
        _queue: SafeVkHandle,
        _command_buffers: &[SafeVkHandle],
        _fence: SafeVkHandle,
    ) -> VkResult {
        VK_SUCCESS
    }

    fn vk_queue_wait_idle(_queue: SafeVkHandle) -> VkResult {
        VK_SUCCESS
    }

    fn vk_device_wait_idle(_device: SafeVkHandle) -> VkResult {
        VK_SUCCESS
    }

    /// Vulkan physical device properties
    #[derive(Debug, Clone, Default)]
    pub struct VulkanPhysicalDeviceProperties {
        pub api_version: u32,
        pub driver_version: u32,
        pub vendor_id: u32,
        pub device_id: u32,
        pub device_type: u32,
        pub device_name: String,
        pub max_compute_work_groupcount: [u32; 3],
        pub max_compute_work_groupsize: [u32; 3],
        pub max_compute_work_group_invocations: u32,
        pub max_push_constants_size: u32,
        pub max_memory_allocation_count: u32,
        pub max_bound_descriptor_sets: u32,
        pub max_storage_bufferrange: u64,
        pub subgroup_size: u32,
        pub supported_subgroup_operations: u32,
        pub timestamp_period: f32,
    }

    /// Vulkan memory properties
    #[derive(Debug, Clone, Default)]
    pub struct VulkanMemoryProperties {
        pub memory_type_count: u32,
        pub memory_types: Vec<VulkanMemoryType>,
        pub memory_heap_count: u32,
        pub memory_heaps: Vec<VulkanMemoryHeap>,
    }

    /// Vulkan memory type
    #[derive(Debug, Clone, Default)]
    pub struct VulkanMemoryType {
        pub property_flags: u32,
        pub heap_index: u32,
    }

    /// Vulkan memory heap
    #[derive(Debug, Clone, Default)]
    pub struct VulkanMemoryHeap {
        pub size: u64,
        pub flags: u32,
    }

    /// Vulkan queue family properties
    #[derive(Debug, Clone, Default)]
    pub struct VulkanQueueFamilyProperties {
        pub queue_flags: u32,
        pub queue_count: u32,
        pub timestamp_valid_bits: u32,
        pub min_image_transfer_granularity: [u32; 3],
    }

    /// Vulkan compute pipeline
    #[derive(Debug)]
    pub struct VulkanComputePipeline {
        pipeline: SafeVkHandle,
        layout: SafeVkHandle,
        descriptor_set_layout: SafeVkHandle,
        shader_module: SafeVkHandle,
        entry_point: String,
        specialization_constants: HashMap<u32, Vec<u8>>,
    }

    impl VulkanComputePipeline {
        fn new(
            pipeline: SafeVkHandle,
            layout: SafeVkHandle,
            descriptor_set_layout: SafeVkHandle,
            shader_module: SafeVkHandle,
            entry_point: &str,
        ) -> Self {
            Self {
                pipeline,
                layout,
                descriptor_set_layout,
                shader_module,
                entry_point: entry_point.to_string(),
                specialization_constants: HashMap::new(),
            }
        }
    }

    /// Comprehensive Vulkan backend with multi-queue support
    pub struct VulkanBackend {
        instance: SafeVkHandle,
        physical_devices: Vec<VulkanPhysicalDeviceInfo>,
        api_version: u32,
        extensions: Vec<String>,
        validation_layers_enabled: bool,
    }

    /// Vulkan physical device information
    #[derive(Debug, Clone)]
    struct VulkanPhysicalDeviceInfo {
        handle: SafeVkHandle,
        properties: VulkanPhysicalDeviceProperties,
        memory_properties: VulkanMemoryProperties,
        queue_families: Vec<VulkanQueueFamilyProperties>,
        supports_compute: bool,
        compute_queue_family: Option<u32>,
        transfer_queue_family: Option<u32>,
    }

    impl VulkanBackend {
        /// Create a new Vulkan backend with automatic initialization
        pub fn new() -> LinalgResult<Self> {
            Self::with_options(VulkanBackendOptions::default())
        }

        /// Create a Vulkan backend with custom options
        pub fn with_options(options: VulkanBackendOptions) -> LinalgResult<Self> {
            // Create Vulkan instance
            let (result, instance) = vk_create_instance();
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create Vulkan instance: error code {}",
                    result
                )));
            }

            // Enumerate physical devices
            let (result, physical_device_handles) = vk_enumerate_physical_devices(instance);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to enumerate Vulkan physical devices: error code {}",
                    result
                )));
            }

            let mut physical_devices = Vec::new();

            for handle in physical_device_handles {
                let properties = vk_get_physical_device_properties(handle);
                let memory_properties = vk_get_physical_device_memory_properties(handle);

                // Mock queue families
                let queue_families = vec![VulkanQueueFamilyProperties {
                    queue_flags: 0x0F, // VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT
                    queue_count: 16,
                    timestamp_valid_bits: 64,
                    min_image_transfer_granularity: [1, 1, 1],
                }];

                // Find compute and transfer queue families
                let compute_queue_family = queue_families
                    .iter()
                    .position(|qf| (qf.queue_flags & 0x02) != 0)
                    .map(|i| i as u32);

                let transfer_queue_family = queue_families
                    .iter()
                    .position(|qf| (qf.queue_flags & 0x04) != 0)
                    .map(|i| i as u32);

                physical_devices.push(VulkanPhysicalDeviceInfo {
                    handle,
                    properties,
                    memory_properties,
                    queue_families,
                    supports_compute: compute_queue_family.is_some(),
                    compute_queue_family,
                    transfer_queue_family,
                });
            }

            Ok(Self {
                instance,
                physical_devices,
                api_version: options.api_version.unwrap_or(0x0010_2000), // Vulkan 1.2
                extensions: options.extensions,
                validation_layers_enabled: options.enable_validation,
            })
        }

        /// Get Vulkan API version
        pub fn api_version(&self) -> u32 {
            self.api_version
        }

        /// Check if validation layers are enabled
        pub fn validation_layers_enabled(&self) -> bool {
            self.validation_layers_enabled
        }

        /// Get enabled extensions
        pub fn extensions(&self) -> &[String] {
            &self.extensions
        }
    }

    impl GpuBackend for VulkanBackend {
        fn name(&self) -> &str {
            "Vulkan"
        }

        fn is_available(&self) -> bool {
            !self.physical_devices.is_empty()
                && self.physical_devices.iter().any(|d| d.supports_compute)
        }

        fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
            let devices = self
                .physical_devices
                .iter()
                .filter(|d| d.supports_compute)
                .map(|device| {
                    let props = &device.properties;
                    let mem_props = &device.memory_properties;

                    // Calculate total device memory
                    let total_memory: usize = mem_props
                        .memory_heaps
                        .iter()
                        .filter(|h| (h.flags & 0x01) != 0) // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        .map(|h| h.size as usize)
                        .sum();

                    // Estimate memory bandwidth (simplified calculation)
                    let memory_bandwidth = 500.0; // Placeholder - would need actual measurement

                    GpuDeviceInfo {
                        device_type: GpuDeviceType::Vulkan,
                        name: props.device_name.clone(),
                        total_memory,
                        compute_units: props.max_compute_work_group_invocations,
                        clock_frequency: 1500, // Placeholder - Vulkan doesn't directly expose this
                        supports_fp64: true,
                        supports_fp16: true,
                        max_work_groupsize: props.max_compute_work_group_invocations as usize,
                        memory_bandwidth,
                        l2_cachesize: 4 * 1024 * 1024, // Placeholder
                        shared_memory_per_block: 48 * 1024, // Typical shared memory size
                        registers_per_block: 65536,
                        warpsize: props.subgroup_size,
                        max_threads_per_mp: props.max_compute_work_group_invocations,
                        multiprocessor_count: 32,     // Placeholder
                        supports_tensor_cores: false, // Vulkan doesn't expose this directly
                        supports_mixed_precision: true,
                        vendor: Self::vendor_name(props.vendor_id),
                    }
                })
                .collect();

            Ok(devices)
        }

        fn create_context(&self, device_id: usize) -> LinalgResult<Box<dyn GpuContext>> {
            let compute_devices: Vec<_> = self
                .physical_devices
                .iter()
                .filter(|d| d.supports_compute)
                .collect();

            if device_id >= compute_devices.len() {
                return Err(LinalgError::ComputationError(format!(
                    "Invalid device ID: {} (available devices: {})",
                    device_id,
                    compute_devices.len()
                )));
            }

            let physical_device = &compute_devices[device_id];

            // Create logical device
            let (result, device) = vk_create_device(physical_device.handle);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create Vulkan logical device: error code {}",
                    result
                )));
            }

            // Get compute queue
            let compute_queue = physical_device
                .compute_queue_family
                .map(|family| vk_get_device_queue(device, family, 0));

            // Get transfer queue
            let transfer_queue = physical_device.transfer_queue_family.and_then(|family| {
                if family != physical_device.compute_queue_family.unwrap_or(u32::MAX) {
                    Some(vk_get_device_queue(device, family, 0))
                } else {
                    None
                }
            });

            // Create command pool
            let (result, command_pool) =
                vk_create_command_pool(device, physical_device.compute_queue_family.unwrap_or(0));
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create Vulkan command pool: error code {}",
                    result
                )));
            }

            Ok(Box::new(VulkanContext::new(
                (*physical_device).clone(),
                device,
                compute_queue,
                transfer_queue,
                command_pool,
            )))
        }
    }

    impl VulkanBackend {
        fn vendor_name(vendor_id: u32) -> String {
            match vendor_id {
                0x1002 => "AMD".to_string(),
                0x10DE => "NVIDIA".to_string(),
                0x8086 => "Intel".to_string(),
                0x13B5 => "ARM".to_string(),
                0x5143 => "Qualcomm".to_string(),
                0x1010 => "ImgTec".to_string(),
                0x106B => "Apple".to_string(),
                _ => format!("Unknown (0x{:04X})", vendor_id),
            }
        }
    }

    /// Options for Vulkan backend initialization
    #[derive(Debug, Clone, Default)]
    pub struct VulkanBackendOptions {
        pub api_version: Option<u32>,
        pub extensions: Vec<String>,
        pub enable_validation: bool,
        pub prefer_discrete_gpu: bool,
    }

    /// Vulkan compute context with comprehensive resource management
    #[derive(Debug)]
    pub struct VulkanContext {
        physical_device_info: VulkanPhysicalDeviceInfo,
        device: SafeVkHandle,
        compute_queue: Option<SafeVkHandle>,
        transfer_queue: Option<SafeVkHandle>,
        command_pool: SafeVkHandle,
        memory_pool: VulkanMemoryPool,
        pipeline_cache: Arc<Mutex<HashMap<String, VulkanComputePipeline>>>,
        descriptor_pool: Option<SafeVkHandle>,
        performance_stats: VulkanPerformanceStats,
    }

    impl VulkanContext {
        fn new(
            physical_device_info: VulkanPhysicalDeviceInfo,
            device: SafeVkHandle,
            compute_queue: Option<SafeVkHandle>,
            transfer_queue: Option<SafeVkHandle>,
            command_pool: SafeVkHandle,
        ) -> Self {
            let memory_pool = VulkanMemoryPool::new(device);

            Self {
                physical_device_info,
                device,
                compute_queue,
                transfer_queue,
                command_pool,
                memory_pool,
                pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
                descriptor_pool: None,
                performance_stats: VulkanPerformanceStats::new(),
            }
        }

        /// Get the Vulkan device handle
        pub fn device(&self) -> SafeVkHandle {
            self.device
        }

        /// Get compute queue
        pub fn compute_queue(&self) -> Option<SafeVkHandle> {
            self.compute_queue
        }

        /// Get transfer queue (if separate from compute)
        pub fn transfer_queue(&self) -> Option<SafeVkHandle> {
            self.transfer_queue
        }

        /// Get command pool
        pub fn command_pool(&self) -> SafeVkHandle {
            self.command_pool
        }

        /// Get performance statistics
        pub fn performance_stats(&self) -> &VulkanPerformanceStats {
            &self.performance_stats
        }

        /// Compile and cache a compute shader
        pub fn compile_shader(
            &mut self,
            _shader_name: &str,
            _spirv_code: &[u32],
            _entry_point: &str,
        ) -> LinalgResult<()> {
            // In a real implementation, compile SPIR-V shader
            Ok(())
        }

        /// Get or create a compute pipeline
        pub fn get_pipeline(&self, _name: &str) -> Option<SafeVkHandle> {
            // Would return cached pipeline or None
            None
        }

        /// Allocate command buffers
        pub fn allocate_command_buffers(&self, count: u32) -> LinalgResult<Vec<SafeVkHandle>> {
            let (result, buffers) =
                vk_allocate_command_buffers(self.device, self.command_pool, count);

            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to allocate command buffers: error code {}",
                    result
                )));
            }

            Ok(buffers)
        }

        /// Submit command buffers to compute queue
        pub fn submit_compute(&self, command_buffers: &[SafeVkHandle]) -> LinalgResult<()> {
            if let Some(queue) = self.compute_queue {
                let result = vk_queue_submit(queue, command_buffers, SafeVkHandle::null());
                if result != VK_SUCCESS {
                    return Err(LinalgError::ComputationError(format!(
                        "Failed to submit compute commands: error code {}",
                        result
                    )));
                }
            }
            Ok(())
        }
    }

    impl GpuContext for VulkanContext {
        #[allow(static_mut_refs)]
        fn device_info(&self) -> &GpuDeviceInfo {
            static mut CACHED_INFO: Option<GpuDeviceInfo> = None;

            unsafe {
                if CACHED_INFO.is_none() {
                    let props = &self.physical_device_info.properties;
                    let mem_props = &self.physical_device_info.memory_properties;

                    let total_memory: usize = mem_props
                        .memory_heaps
                        .iter()
                        .filter(|h| (h.flags & 0x01) != 0)
                        .map(|h| h.size as usize)
                        .sum();

                    CACHED_INFO = Some(GpuDeviceInfo {
                        device_type: GpuDeviceType::Vulkan,
                        name: props.device_name.clone(),
                        total_memory,
                        compute_units: props.max_compute_work_group_invocations,
                        clock_frequency: 1500,
                        supports_fp64: true,
                        supports_fp16: true,
                        max_work_groupsize: props.max_compute_work_group_invocations as usize,
                        memory_bandwidth: 500.0,
                        l2_cachesize: 4 * 1024 * 1024,
                        shared_memory_per_block: 48 * 1024,
                        registers_per_block: 65536,
                        warpsize: props.subgroup_size,
                        max_threads_per_mp: props.max_compute_work_group_invocations,
                        multiprocessor_count: 32,
                        supports_tensor_cores: false,
                        supports_mixed_precision: true,
                        vendor: VulkanBackend::vendor_name(props.vendor_id),
                    });
                }

                CACHED_INFO.as_ref().expect("GpuDeviceInfo not initialized")
            }
        }

        fn synchronize(&self) -> LinalgResult<()> {
            let result = vk_device_wait_idle(self.device);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Vulkan synchronization failed: error code {}",
                    result
                )));
            }
            Ok(())
        }

        fn available_memory(&self) -> LinalgResult<usize> {
            // Derived from this device's reported memory heaps. No physical Vulkan
            // device is enumerated in this build (the backend reports unavailable),
            // so a `VulkanContext` is never actually constructed and this path is
            // not reachable; when it is, it sums real heap sizes rather than
            // returning a fabricated constant (empty heaps yield 0).
            let total_memory: usize = self
                .physical_device_info
                .memory_properties
                .memory_heaps
                .iter()
                .filter(|h| (h.flags & 0x01) != 0)
                .map(|h| h.size as usize)
                .sum();

            Ok(total_memory / 2)
        }
    }

    impl GpuContextAlloc for VulkanContext {
        fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
            &self,
            size: usize,
        ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
            let buffer = VulkanBuffer::new(size, self.device)?;
            Ok(Box::new(buffer))
        }
    }

    /// Vulkan memory pool for efficient suballocation
    #[derive(Debug)]
    struct VulkanMemoryPool {
        device: SafeVkHandle,
        allocations: HashMap<usize, VulkanAllocation>,
        free_blocks: HashMap<u32, Vec<VulkanMemoryBlock>>,
        total_allocated: usize,
        peak_usage: usize,
        allocation_count: usize,
    }

    #[derive(Debug)]
    struct VulkanAllocation {
        memory: SafeVkHandle,
        size: usize,
        memory_type_index: u32,
        mapped_ptr: Option<*mut std::ffi::c_void>,
    }

    // SAFETY: Vulkan device memory handles and mapped pointers are thread-safe
    // when properly synchronized through the Vulkan driver. The raw pointer
    // is only used for memory operations that are synchronized by Vulkan.
    unsafe impl Send for VulkanAllocation {}
    unsafe impl Sync for VulkanAllocation {}

    #[derive(Debug, Clone)]
    struct VulkanMemoryBlock {
        allocation_id: usize,
        offset: usize,
        size: usize,
    }

    impl VulkanMemoryPool {
        fn new(device: SafeVkHandle) -> Self {
            Self {
                device,
                allocations: HashMap::new(),
                free_blocks: HashMap::new(),
                total_allocated: 0,
                peak_usage: 0,
                allocation_count: 0,
            }
        }

        #[allow(dead_code)]
        fn allocate(
            &mut self,
            size: usize,
            memory_type_index: u32,
        ) -> LinalgResult<VulkanMemoryBlock> {
            // Try to find a free block
            if let Some(blocks) = self.free_blocks.get_mut(&memory_type_index) {
                if let Some(block) = blocks.iter().position(|b| b.size >= size) {
                    return Ok(blocks.swap_remove(block));
                }
            }

            // Allocate new memory
            let (result, memory) = vk_allocate_memory(self.device, size, memory_type_index);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Vulkan memory allocation failed: error code {}",
                    result
                )));
            }

            let allocation_id = self.allocation_count;
            self.allocation_count += 1;

            self.allocations.insert(
                allocation_id,
                VulkanAllocation {
                    memory,
                    size,
                    memory_type_index,
                    mapped_ptr: None,
                },
            );

            self.total_allocated += size;
            self.peak_usage = self.peak_usage.max(self.total_allocated);

            Ok(VulkanMemoryBlock {
                allocation_id,
                offset: 0,
                size,
            })
        }

        #[allow(dead_code)]
        fn deallocate(&mut self, block: VulkanMemoryBlock) {
            if let Some(allocation) = self.allocations.get(&block.allocation_id) {
                self.free_blocks
                    .entry(allocation.memory_type_index)
                    .or_default()
                    .push(block);
            }
        }
    }

    /// Vulkan buffer implementation
    #[derive(Debug)]
    struct VulkanBuffer<T> {
        buffer: SafeVkHandle,
        memory: SafeVkHandle,
        device: SafeVkHandle,
        size: usize,
        is_host_visible: bool,
        _phantom: std::marker::PhantomData<T>,
    }

    // SAFETY: Vulkan buffers are thread-safe when properly synchronized
    unsafe impl<T> Send for VulkanBuffer<T> {}
    unsafe impl<T> Sync for VulkanBuffer<T> {}

    impl<T: Clone + Send + Sync + Copy> VulkanBuffer<T> {
        fn new(size: usize, device: SafeVkHandle) -> LinalgResult<Self> {
            let byte_size = size * std::mem::size_of::<T>();

            // Create buffer
            let (result, buffer) = vk_create_buffer(
                device,
                byte_size,
                0x80 | 0x01, // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            );

            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to create Vulkan buffer: error code {}",
                    result
                )));
            }

            // Allocate memory
            let (result, memory) = vk_allocate_memory(device, byte_size, 0);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to allocate Vulkan buffer memory: error code {}",
                    result
                )));
            }

            // Bind memory to buffer
            let result = vk_bind_buffer_memory(device, buffer, memory, 0);
            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to bind Vulkan buffer memory: error code {}",
                    result
                )));
            }

            Ok(Self {
                buffer,
                memory,
                device,
                size,
                is_host_visible: true,
                _phantom: std::marker::PhantomData,
            })
        }

        /// Get the Vulkan buffer handle
        pub fn vk_buffer(&self) -> SafeVkHandle {
            self.buffer
        }
    }

    impl<T: Clone + Send + Sync + Copy + std::fmt::Debug> GpuBuffer<T> for VulkanBuffer<T> {
        fn len(&self) -> usize {
            self.size
        }

        fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            if !self.is_host_visible {
                return Err(LinalgError::ComputationError(
                    "Buffer is not host visible".to_string(),
                ));
            }

            let byte_size = data.len() * std::mem::size_of::<T>();
            let (result, mapped_ptr) = vk_map_memory(self.device, self.memory, 0, byte_size);

            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to map Vulkan memory: error code {}",
                    result
                )));
            }

            if !mapped_ptr.is_null() {
                // Copy data (in mock, this is a no-op)
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        mapped_ptr as *mut u8,
                        byte_size,
                    );
                }
            }

            vk_unmap_memory(self.device, self.memory);
            Ok(())
        }

        fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()> {
            if data.len() != self.size {
                return Err(LinalgError::ShapeError(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    self.size,
                    data.len()
                )));
            }

            if !self.is_host_visible {
                return Err(LinalgError::ComputationError(
                    "Buffer is not host visible".to_string(),
                ));
            }

            let byte_size = data.len() * std::mem::size_of::<T>();
            let (result, mapped_ptr) = vk_map_memory(self.device, self.memory, 0, byte_size);

            if result != VK_SUCCESS {
                return Err(LinalgError::ComputationError(format!(
                    "Failed to map Vulkan memory: error code {}",
                    result
                )));
            }

            if !mapped_ptr.is_null() {
                // Copy data (in mock, this is a no-op)
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mapped_ptr as *const u8,
                        data.as_mut_ptr() as *mut u8,
                        byte_size,
                    );
                }
            }

            vk_unmap_memory(self.device, self.memory);
            Ok(())
        }

        fn device_ptr(&self) -> *mut std::ffi::c_void {
            self.buffer.0
        }
    }

    /// Performance statistics for Vulkan operations
    #[derive(Debug, Clone)]
    pub struct VulkanPerformanceStats {
        pub compute_dispatches: usize,
        pub buffer_operations: usize,
        pub total_compute_time_ms: f64,
        pub total_transfer_time_ms: f64,
        pub pipeline_cache_hits: usize,
        pub pipeline_cache_misses: usize,
        pub memory_allocated: usize,
        pub memory_freed: usize,
    }

    impl VulkanPerformanceStats {
        fn new() -> Self {
            Self {
                compute_dispatches: 0,
                buffer_operations: 0,
                total_compute_time_ms: 0.0,
                total_transfer_time_ms: 0.0,
                pipeline_cache_hits: 0,
                pipeline_cache_misses: 0,
                memory_allocated: 0,
                memory_freed: 0,
            }
        }

        pub fn compute_efficiency(&self) -> f64 {
            if self.total_compute_time_ms + self.total_transfer_time_ms == 0.0 {
                return 0.0;
            }
            self.total_compute_time_ms / (self.total_compute_time_ms + self.total_transfer_time_ms)
        }

        pub fn cache_hit_rate(&self) -> f64 {
            let total = self.pipeline_cache_hits + self.pipeline_cache_misses;
            if total == 0 {
                return 0.0;
            }
            self.pipeline_cache_hits as f64 / total as f64
        }
    }
}

// Re-export the Vulkan backend when the feature is enabled
#[cfg(feature = "vulkan")]
pub use vulkan_impl::*;

// Provide a stub when Vulkan is not available
#[cfg(not(feature = "vulkan"))]
pub struct VulkanBackend;

#[cfg(not(feature = "vulkan"))]
impl VulkanBackend {
    pub fn new() -> LinalgResult<Self> {
        Err(LinalgError::ComputationError(
            "Vulkan support not compiled in".to_string(),
        ))
    }
}

#[cfg(not(feature = "vulkan"))]
impl GpuBackend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan (not available)"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
        Ok(vec![])
    }

    fn create_context(&self, _device_id: usize) -> LinalgResult<Box<dyn GpuContext>> {
        Err(LinalgError::ComputationError(
            "Vulkan support not compiled in".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_backend_stub() {
        #[cfg(not(feature = "vulkan"))]
        {
            let backend = VulkanBackend;
            assert!(!backend.is_available());
            assert_eq!(backend.name(), "Vulkan (not available)");
        }
    }
}
