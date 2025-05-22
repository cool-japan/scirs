//! Cross-device memory management for efficient data transfer between CPU and GPU
//!
//! This module provides utilities for managing memory across different devices
//! (CPU, GPU, TPU) with efficient data transfer and synchronization. It includes:
//!
//! - Cross-device memory transfer with automatic format conversion
//! - Memory pools for efficient allocation
//! - Smart caching for frequently accessed data
//! - Efficient pinned memory for faster CPU-GPU transfers
//! - Asynchronous data transfer with event-based synchronization

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuError};
use ndarray::{Array, ArrayBase, Dimension, Ix1, Ix2, IxDyn, RawData, RawDataClone};
use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};

/// Device types supported by the cross-device memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (host) memory
    Cpu,
    /// Discrete GPU memory
    Gpu(GpuBackend),
    /// TPU (Tensor Processing Unit) memory
    Tpu,
}

impl DeviceType {
    /// Check if the device is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::Cpu => true,
            DeviceType::Gpu(backend) => backend.is_available(),
            DeviceType::Tpu => false, // TPU support not yet implemented
        }
    }

    /// Get the name of the device
    pub fn name(&self) -> String {
        match self {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Gpu(backend) => format!("GPU ({})", backend),
            DeviceType::Tpu => "TPU".to_string(),
        }
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Gpu(backend) => write!(f, "GPU ({})", backend),
            DeviceType::Tpu => write!(f, "TPU"),
        }
    }
}

/// Memory transfer direction between devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device transfer (e.g., CPU to GPU)
    HostToDevice,
    /// Device to host transfer (e.g., GPU to CPU)
    DeviceToHost,
    /// Device to device transfer (e.g., GPU to TPU)
    DeviceToDevice,
}

/// Transfer mode for cross-device operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous transfer (blocks until complete)
    Synchronous,
    /// Asynchronous transfer (returns immediately, track with events)
    Asynchronous,
}

/// Memory layout for device buffers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major layout (C-style)
    RowMajor,
    /// Column-major layout (Fortran-style)
    ColumnMajor,
    /// Strided layout (custom strides)
    Strided,
}

/// Options for cross-device memory transfers
#[derive(Debug, Clone)]
pub struct TransferOptions {
    /// Transfer mode (synchronous or asynchronous)
    pub mode: TransferMode,
    /// Memory layout for the transfer
    pub layout: MemoryLayout,
    /// Whether to use pinned memory for the transfer
    pub use_pinned_memory: bool,
    /// Whether to enable streaming transfers for large buffers
    pub enable_streaming: bool,
    /// Stream ID for asynchronous transfers
    pub stream_id: Option<usize>,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            mode: TransferMode::Synchronous,
            layout: MemoryLayout::RowMajor,
            use_pinned_memory: true,
            enable_streaming: true,
            stream_id: None,
        }
    }
}

/// Builder for transfer options
#[derive(Debug, Clone)]
pub struct TransferOptionsBuilder {
    options: TransferOptions,
}

impl TransferOptionsBuilder {
    /// Create a new transfer options builder with default values
    pub fn new() -> Self {
        Self {
            options: TransferOptions::default(),
        }
    }

    /// Set the transfer mode
    pub fn mode(mut self, mode: TransferMode) -> Self {
        self.options.mode = mode;
        self
    }

    /// Set the memory layout
    pub fn layout(mut self, layout: MemoryLayout) -> Self {
        self.options.layout = layout;
        self
    }

    /// Set whether to use pinned memory
    pub fn use_pinned_memory(mut self, use_pinned_memory: bool) -> Self {
        self.options.use_pinned_memory = use_pinned_memory;
        self
    }

    /// Set whether to enable streaming transfers
    pub fn enable_streaming(mut self, enable_streaming: bool) -> Self {
        self.options.enable_streaming = enable_streaming;
        self
    }

    /// Set the stream ID for asynchronous transfers
    pub fn stream_id(mut self, stream_id: Option<usize>) -> Self {
        self.options.stream_id = stream_id;
        self
    }

    /// Build the transfer options
    pub fn build(self) -> TransferOptions {
        self.options
    }
}

/// Cache key for the device memory cache
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheKey {
    /// Data identifier (usually the memory address of the host array)
    data_id: usize,
    /// Device type
    device: DeviceType,
    /// Element type ID
    type_id: TypeId,
    /// Size in elements
    size: usize,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data_id.hash(state);
        self.device.hash(state);
        self.type_id.hash(state);
        self.size.hash(state);
    }
}

/// Event for tracking asynchronous operations
#[derive(Debug)]
pub struct TransferEvent {
    /// Device associated with the event
    device: DeviceType,
    /// Internal event handle (implementation-specific)
    handle: Arc<Mutex<Box<dyn std::any::Any + Send + Sync>>>,
    /// Whether the event has been completed
    completed: Arc<std::sync::atomic::AtomicBool>,
}

impl TransferEvent {
    /// Create a new transfer event
    fn new(device: DeviceType, handle: Box<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            device,
            handle: Arc::new(Mutex::new(handle)),
            completed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Wait for the event to complete
    pub fn wait(&self) {
        // In a real implementation, this would block until the event is complete
        // For now, just set the completed flag for demonstration
        self.completed
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the event has completed
    pub fn is_complete(&self) -> bool {
        self.completed.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Cache entry for the device memory cache
struct CacheEntry<T: GpuDataType> {
    /// Buffer on the device
    buffer: DeviceBuffer<T>,
    /// Size in elements
    size: usize,
    /// Last access time
    last_access: std::time::Instant,
    /// Whether the buffer is dirty (modified on device)
    dirty: bool,
}

/// Device memory manager for cross-device operations
pub struct DeviceMemoryManager {
    /// GPU context for accessing GPU functionality
    gpu_context: Option<GpuContext>,
    /// Cache of device buffers
    cache: Mutex<HashMap<CacheKey, Box<dyn std::any::Any + Send + Sync>>>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size in bytes
    current_cache_size: std::sync::atomic::AtomicUsize,
    /// Whether the caching is enabled
    enable_caching: bool,
}

impl DeviceMemoryManager {
    /// Create a new device memory manager
    pub fn new(enable_caching: bool, max_cache_size: usize) -> Result<Self, CoreError> {
        // Try to create a GPU context if a GPU is available
        let gpu_context = match GpuBackend::preferred() {
            backend if backend.is_available() => match GpuContext::new(backend) {
                Ok(context) => Some(context),
                Err(_) => None,
            },
            _ => None,
        };

        Ok(Self {
            gpu_context,
            cache: Mutex::new(HashMap::new()),
            max_cache_size,
            current_cache_size: std::sync::atomic::AtomicUsize::new(0),
            enable_caching,
        })
    }

    /// Check if a device type is available
    pub fn is_device_available(&self, device: DeviceType) -> bool {
        match device {
            DeviceType::Cpu => true,
            DeviceType::Gpu(_) => self.gpu_context.is_some(),
            DeviceType::Tpu => false, // TPU not yet supported
        }
    }

    /// Get a list of available devices
    pub fn available_devices(&self) -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];

        if let Some(ref context) = self.gpu_context {
            devices.push(DeviceType::Gpu(context.backend()));
        }

        devices
    }

    /// Transfer data from host to device
    pub fn transfer_to_device<T, S, D>(
        &self,
        array: &ArrayBase<S, D>,
        device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        S: RawData<Elem = T>,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {} is not available", device))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // For CPU, just create a view of the array
        if device == DeviceType::Cpu {
            return Ok(DeviceArray::new_cpu(array.to_owned()));
        }

        // For GPU, create a GPU buffer
        if let DeviceType::Gpu(backend) = device {
            if let Some(ref context) = self.gpu_context {
                if context.backend() != backend {
                    return Err(CoreError::DeviceError(
                        ErrorContext::new(format!(
                            "GPU backend mismatch: requested {}, available {}",
                            backend,
                            context.backend()
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }

                // Create a flat view of the array data
                let flat_data = array.as_slice().ok_or_else(|| {
                    CoreError::DeviceError(
                        ErrorContext::new("Array is not contiguous".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                // Check if we have a cached buffer for this array
                let data_id = flat_data.as_ptr() as usize;
                let key = CacheKey {
                    data_id,
                    device,
                    type_id: TypeId::of::<T>(),
                    size: flat_data.len(),
                };

                let buffer = if self.enable_caching {
                    let mut cache = self.cache.lock().unwrap();
                    if let Some(entry) = cache.get_mut(&key) {
                        // We found a cached entry, cast it to the correct type
                        if let Some(entry) = entry.downcast_mut::<CacheEntry<T>>() {
                            // Update the last access time
                            entry.last_access = std::time::Instant::now();
                            entry.buffer.clone()
                        } else {
                            // This should never happen if our caching logic is correct
                            return Err(CoreError::DeviceError(
                                ErrorContext::new("Cache entry type mismatch".to_string())
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            ));
                        }
                    } else {
                        // No cached entry, create a new buffer
                        let gpu_buffer = context.create_buffer_from_slice(flat_data);
                        let buffer = DeviceBuffer::new_gpu(gpu_buffer);

                        // Add to cache
                        let entry = CacheEntry {
                            buffer: buffer.clone(),
                            size: flat_data.len(),
                            last_access: std::time::Instant::now(),
                            dirty: false,
                        };

                        let buffer_size = std::mem::size_of::<T>() * flat_data.len();
                        self.current_cache_size
                            .fetch_add(buffer_size, std::sync::atomic::Ordering::SeqCst);

                        // If we're over the cache size limit, evict old entries
                        self.evict_cache_entries_if_needed();

                        cache.insert(key, Box::new(entry));
                        buffer
                    }
                } else {
                    // Caching is disabled, just create a new buffer
                    let gpu_buffer = context.create_buffer_from_slice(flat_data);
                    DeviceBuffer::new_gpu(gpu_buffer)
                };

                return Ok(DeviceArray {
                    buffer,
                    shape: array.raw_dim(),
                    device,
                    _phantom: PhantomData,
                });
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!("Unsupported device type: {}", device))
                .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }

    /// Transfer data from device to host
    pub fn transfer_to_host<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        options: Option<TransferOptions>,
    ) -> CoreResult<Array<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // For CPU arrays, just clone the data
        if device_array.device == DeviceType::Cpu {
            if let Some(cpu_array) = device_array.buffer.get_cpu_array() {
                return Ok(cpu_array.clone());
            }
        }

        // For GPU arrays, copy the data back to the host
        if let DeviceType::Gpu(_) = device_array.device {
            if let Some(gpu_buffer) = device_array.buffer.get_gpu_buffer() {
                let size = device_array.size();
                let mut data = vec![unsafe { std::mem::zeroed() }; size];

                // Copy data from GPU to host
                gpu_buffer.copy_to_host(&mut data);

                // Reshape the data to match the original array shape
                return Ok(
                    Array::from_shape_vec(device_array.shape.clone(), data).map_err(|e| {
                        CoreError::DeviceError(
                            ErrorContext::new(format!("Failed to reshape array: {}", e))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?,
                );
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!(
                "Unsupported device type for transfer to host: {}",
                device_array.device
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }

    /// Transfer data between devices
    pub fn transfer_between_devices<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        target_device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // If the source and target devices are the same, just clone the array
        if device_array.device == target_device {
            return Ok(device_array.clone());
        }

        // For transfers to CPU, use transfer_to_host
        if target_device == DeviceType::Cpu {
            let host_array = self.transfer_to_host(device_array, Some(options))?;
            return Ok(DeviceArray::new_cpu(host_array));
        }

        // For transfers from CPU to another device, use transfer_to_device
        if device_array.device == DeviceType::Cpu {
            if let Some(cpu_array) = device_array.buffer.get_cpu_array() {
                return self.transfer_to_device(&cpu_array, target_device, Some(options));
            }
        }

        // For transfers between GPUs (or future TPU support)
        // In a real implementation, we would use peer-to-peer transfers if available,
        // or copy through host memory if not

        // For now, we'll transfer through host memory
        let host_array = self.transfer_to_host(device_array, Some(options))?;
        self.transfer_to_device(&host_array, target_device, Some(options))
    }

    /// Evict cache entries if the total size exceeds the limit
    fn evict_cache_entries_if_needed(&self) {
        let current_size = self
            .current_cache_size
            .load(std::sync::atomic::Ordering::SeqCst);
        if current_size <= self.max_cache_size {
            return;
        }

        let mut cache = self.cache.lock().unwrap();

        // Sort entries by last access time (oldest first)
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by(|a, b| {
            // This is unsafe because we're downcasting without checking,
            // but we should never have different types in the cache for the same key
            // Get last_access for the first entry
            let a_time = match a.1.downcast_ref::<CacheEntry<f32>>() {
                Some(entry) => entry.last_access,
                None => match a.1.downcast_ref::<CacheEntry<f64>>() {
                    Some(entry) => entry.last_access,
                    None => match a.1.downcast_ref::<CacheEntry<i32>>() {
                        Some(entry) => entry.last_access,
                        None => match a.1.downcast_ref::<CacheEntry<u32>>() {
                            Some(entry) => entry.last_access,
                            None => std::time::Instant::now(), // Fallback, shouldn't happen
                        },
                    },
                },
            };

            // Get last_access for the second entry
            let b_time = match b.1.downcast_ref::<CacheEntry<f32>>() {
                Some(entry) => entry.last_access,
                None => match b.1.downcast_ref::<CacheEntry<f64>>() {
                    Some(entry) => entry.last_access,
                    None => match b.1.downcast_ref::<CacheEntry<i32>>() {
                        Some(entry) => entry.last_access,
                        None => match b.1.downcast_ref::<CacheEntry<u32>>() {
                            Some(entry) => entry.last_access,
                            None => std::time::Instant::now(), // Fallback, shouldn't happen
                        },
                    },
                },
            };

            a_time.cmp(&b_time)
        });

        // Remove entries until we're under the limit
        let mut removed_size = 0;
        let target_size = current_size - self.max_cache_size / 2; // Remove enough to get below half the limit

        for (key, _) in entries {
            let key = key.clone();
            let entry = cache.remove(&key).unwrap();

            // Calculate the size of the entry based on its type
            let entry_size = match entry.downcast_ref::<CacheEntry<f32>>() {
                Some(entry) => entry.size * std::mem::size_of::<f32>(),
                None => match entry.downcast_ref::<CacheEntry<f64>>() {
                    Some(entry) => entry.size * std::mem::size_of::<f64>(),
                    None => match entry.downcast_ref::<CacheEntry<i32>>() {
                        Some(entry) => entry.size * std::mem::size_of::<i32>(),
                        None => match entry.downcast_ref::<CacheEntry<u32>>() {
                            Some(entry) => entry.size * std::mem::size_of::<u32>(),
                            None => 0, // Fallback, shouldn't happen
                        },
                    },
                },
            };

            removed_size += entry_size;

            if removed_size >= target_size {
                break;
            }
        }

        // Update the current cache size
        self.current_cache_size
            .fetch_sub(removed_size, std::sync::atomic::Ordering::SeqCst);
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        self.current_cache_size
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Execute a kernel on a device array
    pub fn execute_kernel<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        kernel_name: &str,
        params: HashMap<String, KernelParam>,
    ) -> CoreResult<()>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Only GPU devices support kernel execution
        if let DeviceType::Gpu(_) = device_array.device {
            if let Some(ref context) = self.gpu_context {
                // Get the kernel
                let mut kernel = context.get_kernel(kernel_name)?;

                // Set the input buffer parameter
                if let Some(gpu_buffer) = device_array.buffer.get_gpu_buffer() {
                    kernel.set_buffer("input", gpu_buffer);
                }

                // Set other parameters
                for (name, param) in params {
                    match param {
                        KernelParam::Buffer(buffer) => {
                            if let Some(gpu_buffer) = buffer.get_gpu_buffer() {
                                kernel.set_buffer(&name, gpu_buffer);
                            }
                        }
                        KernelParam::U32(value) => kernel.set_u32(&name, value),
                        KernelParam::I32(value) => kernel.set_i32(&name, value),
                        KernelParam::F32(value) => kernel.set_f32(&name, value),
                        KernelParam::F64(value) => kernel.set_f64(&name, value),
                    }
                }

                // Compute dispatch dimensions
                let total_elements = device_array.size();
                let work_group_size = 256; // A common CUDA/OpenCL work group size
                let num_groups = (total_elements + work_group_size - 1) / work_group_size;

                // Dispatch the kernel
                kernel.dispatch([num_groups as u32, 1, 1]);

                return Ok(());
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!(
                "Unsupported device type for kernel execution: {}",
                device_array.device
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Kernel parameter for GPU execution
#[derive(Debug, Clone)]
pub enum KernelParam {
    /// Buffer parameter
    Buffer(DeviceBuffer<f32>), // Note: In a real implementation, this would be generic
    /// U32 parameter
    U32(u32),
    /// I32 parameter
    I32(i32),
    /// F32 parameter
    F32(f32),
    /// F64 parameter
    F64(f64),
}

/// Buffer location (CPU or GPU)
#[derive(Debug, Clone)]
enum BufferLocation<T: GpuDataType> {
    /// CPU buffer
    Cpu(Arc<Array<T, IxDyn>>),
    /// GPU buffer
    Gpu(Arc<GpuBuffer<T>>),
}

/// Buffer for cross-device operations
#[derive(Debug, Clone)]
pub struct DeviceBuffer<T: GpuDataType> {
    /// Buffer data (CPU or GPU)
    location: BufferLocation<T>,
}

impl<T: GpuDataType> DeviceBuffer<T> {
    /// Create a new CPU buffer
    fn new_cpu<D: Dimension>(array: Array<T, D>) -> Self {
        Self {
            location: BufferLocation::Cpu(Arc::new(array.into_dyn())),
        }
    }

    /// Create a new GPU buffer
    fn new_gpu(buffer: GpuBuffer<T>) -> Self {
        Self {
            location: BufferLocation::Gpu(Arc::new(buffer)),
        }
    }

    /// Get the CPU array if available
    fn get_cpu_array(&self) -> Option<&Array<T, IxDyn>> {
        match self.location {
            BufferLocation::Cpu(ref array) => Some(array),
            _ => None,
        }
    }

    /// Get the GPU buffer if available
    fn get_gpu_buffer(&self) -> Option<&GpuBuffer<T>> {
        match self.location {
            BufferLocation::Gpu(ref buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Get the size of the buffer in elements
    fn size(&self) -> usize {
        match self.location {
            BufferLocation::Cpu(ref array) => array.len(),
            BufferLocation::Gpu(ref buffer) => buffer.len(),
        }
    }
}

/// Array residing on a specific device (CPU, GPU, TPU)
#[derive(Debug, Clone)]
pub struct DeviceArray<T: GpuDataType, D: Dimension> {
    /// Buffer containing the array data
    buffer: DeviceBuffer<T>,
    /// Shape of the array
    shape: D,
    /// Device where the array resides
    device: DeviceType,
    /// Phantom data for the element type
    _phantom: PhantomData<T>,
}

impl<T: GpuDataType, D: Dimension> DeviceArray<T, D> {
    /// Create a new CPU array
    fn new_cpu<S: RawData<Elem = T>>(array: ArrayBase<S, D>) -> Self {
        Self {
            buffer: DeviceBuffer::new_cpu(array.to_owned()),
            shape: array.raw_dim(),
            device: DeviceType::Cpu,
            _phantom: PhantomData,
        }
    }

    /// Get the device where the array resides
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &D {
        &self.shape
    }

    /// Get the size of the array in elements
    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Check if the array is on the CPU
    pub fn is_on_cpu(&self) -> bool {
        self.device == DeviceType::Cpu
    }

    /// Check if the array is on a GPU
    pub fn is_on_gpu(&self) -> bool {
        matches!(self.device, DeviceType::Gpu(_))
    }

    /// Get a reference to the underlying CPU array if available
    pub fn as_cpu_array(&self) -> Option<&Array<T, IxDyn>> {
        self.buffer.get_cpu_array()
    }

    /// Get a reference to the underlying GPU buffer if available
    pub fn as_gpu_buffer(&self) -> Option<&GpuBuffer<T>> {
        self.buffer.get_gpu_buffer()
    }
}

/// Stream for asynchronous operations
pub struct DeviceStream {
    /// Device associated with the stream
    device: DeviceType,
    /// Internal stream handle (implementation-specific)
    handle: Arc<Mutex<Box<dyn std::any::Any + Send + Sync>>>,
}

impl DeviceStream {
    /// Create a new device stream
    pub fn new(device: DeviceType) -> CoreResult<Self> {
        // In a real implementation, we would create a stream for the device
        // For now, just create a dummy stream
        Ok(Self {
            device,
            handle: Arc::new(Mutex::new(Box::new(()))),
        })
    }

    /// Synchronize the stream
    pub fn synchronize(&self) {
        // In a real implementation, this would wait for all operations to complete
    }
}

/// Memory pool for efficient allocation on a device
pub struct DeviceMemoryPool {
    /// Device associated with the pool
    device: DeviceType,
    /// List of free buffers by size
    free_buffers: Mutex<HashMap<usize, Vec<Box<dyn std::any::Any + Send + Sync>>>>,
    /// Maximum pool size in bytes
    max_pool_size: usize,
    /// Current pool size in bytes
    current_pool_size: std::sync::atomic::AtomicUsize,
}

impl DeviceMemoryPool {
    /// Create a new device memory pool
    pub fn new(device: DeviceType, max_pool_size: usize) -> Self {
        Self {
            device,
            free_buffers: Mutex::new(HashMap::new()),
            max_pool_size,
            current_pool_size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a buffer of the given size
    pub fn allocate<T: GpuDataType>(&self, size: usize) -> CoreResult<DeviceBuffer<T>> {
        // Check if we have a free buffer of the right size
        let mut free_buffers = self.free_buffers.lock().unwrap();
        if let Some(buffers) = free_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                // We found a free buffer, cast it to the correct type
                if let Ok(buffer) = buffer.downcast::<DeviceBuffer<T>>() {
                    return Ok(*buffer);
                }
            }
        }

        // No free buffer, allocate a new one
        match self.device {
            DeviceType::Cpu => {
                // Allocate CPU memory
                let array = Array::<T, _>::zeros(IxDyn(&[size]));
                Ok(DeviceBuffer::new_cpu(array))
            }
            DeviceType::Gpu(_) => {
                // Allocate GPU memory
                Err(CoreError::ImplementationError(
                    ErrorContext::new("GPU memory allocation not implemented".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
            DeviceType::Tpu => {
                // TPU not yet supported
                Err(CoreError::DeviceError(
                    ErrorContext::new("TPU not supported".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        }
    }

    /// Free a buffer (return it to the pool)
    pub fn free<T: GpuDataType>(&self, buffer: DeviceBuffer<T>) {
        let size = buffer.size();
        let buffer_size = size * std::mem::size_of::<T>();

        // Check if adding this buffer would exceed the pool size
        let current_size = self
            .current_pool_size
            .load(std::sync::atomic::Ordering::SeqCst);
        if current_size + buffer_size > self.max_pool_size {
            // Pool is full, just let the buffer be dropped
            return;
        }

        // Add the buffer to the pool
        let mut free_buffers = self.free_buffers.lock().unwrap();
        free_buffers
            .entry(size)
            .or_insert_with(Vec::new)
            .push(Box::new(buffer));

        // Update the pool size
        self.current_pool_size
            .fetch_add(buffer_size, std::sync::atomic::Ordering::SeqCst);
    }

    /// Clear the pool
    pub fn clear(&self) {
        let mut free_buffers = self.free_buffers.lock().unwrap();
        free_buffers.clear();
        self.current_pool_size
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Cross-device array operations
impl<T: GpuDataType, D: Dimension> DeviceArray<T, D> {
    /// Map the array elements using a function
    pub fn map<F>(&self, f: F, manager: &DeviceMemoryManager) -> CoreResult<DeviceArray<T, D>>
    where
        F: Fn(T) -> T + Send + Sync,
        D: Clone,
    {
        // For CPU arrays, use ndarray's map function
        if self.is_on_cpu() {
            if let Some(cpu_array) = self.as_cpu_array() {
                let mapped = cpu_array.map(|&x| f(x));
                return Ok(DeviceArray {
                    buffer: DeviceBuffer::new_cpu(mapped),
                    shape: self.shape.clone(),
                    device: DeviceType::Cpu,
                    _phantom: PhantomData,
                });
            }
        }

        // For GPU arrays, transfer to host, map, and transfer back
        // In a real implementation, we would use a GPU kernel
        let host_array = manager.transfer_to_host(self, None)?;
        let mapped = host_array.map(|&x| f(x));
        manager.transfer_to_device(&mapped, self.device, None)
    }

    /// Reduce the array using a binary operation
    pub fn reduce<F>(&self, f: F, manager: &DeviceMemoryManager) -> CoreResult<T>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Copy,
    {
        // For CPU arrays, use ndarray's fold function
        if self.is_on_cpu() {
            if let Some(cpu_array) = self.as_cpu_array() {
                if cpu_array.is_empty() {
                    return Err(CoreError::ValueError(
                        ErrorContext::new("Cannot reduce empty array".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }

                let first = cpu_array[0];
                let result = cpu_array.iter().skip(1).fold(first, |acc, &x| f(acc, x));
                return Ok(result);
            }
        }

        // For GPU arrays, transfer to host and reduce
        // In a real implementation, we would use a GPU reduction kernel
        let host_array = manager.transfer_to_host(self, None)?;
        if host_array.is_empty() {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot reduce empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let first = host_array[0];
        let result = host_array.iter().skip(1).fold(first, |acc, &x| f(acc, x));
        Ok(result)
    }
}

/// Cross-device manager for handling data transfers and operations
#[derive(Debug)]
pub struct CrossDeviceManager {
    /// Memory managers for each device
    memory_managers: HashMap<DeviceType, DeviceMemoryManager>,
    /// Memory pools for each device
    memory_pools: HashMap<DeviceType, DeviceMemoryPool>,
    /// Active data transfers
    active_transfers: Mutex<Vec<TransferEvent>>,
    /// Enable caching
    enable_caching: bool,
    /// Maximum cache size in bytes
    max_cache_size: usize,
}

impl CrossDeviceManager {
    /// Create a new cross-device manager
    pub fn new(enable_caching: bool, max_cache_size: usize) -> CoreResult<Self> {
        let mut memory_managers = HashMap::new();
        let mut memory_pools = HashMap::new();

        // Create CPU memory manager and pool
        let cpu_manager = DeviceMemoryManager::new(enable_caching, max_cache_size)?;
        memory_managers.insert(DeviceType::Cpu, cpu_manager);
        memory_pools.insert(
            DeviceType::Cpu,
            DeviceMemoryPool::new(DeviceType::Cpu, max_cache_size),
        );

        // Try to create GPU memory manager and pool
        let gpu_backend = GpuBackend::preferred();
        if gpu_backend.is_available() {
            let gpu_device = DeviceType::Gpu(gpu_backend);
            let gpu_manager = DeviceMemoryManager::new(enable_caching, max_cache_size)?;
            memory_managers.insert(gpu_device, gpu_manager);
            memory_pools.insert(
                gpu_device,
                DeviceMemoryPool::new(gpu_device, max_cache_size),
            );
        }

        Ok(Self {
            memory_managers,
            memory_pools,
            active_transfers: Mutex::new(Vec::new()),
            enable_caching,
            max_cache_size,
        })
    }

    /// Get a list of available devices
    pub fn available_devices(&self) -> Vec<DeviceType> {
        self.memory_managers.keys().cloned().collect()
    }

    /// Check if a device is available
    pub fn is_device_available(&self, device: DeviceType) -> bool {
        self.memory_managers.contains_key(&device)
    }

    /// Transfer data to a device
    pub fn to_device<T, S, D>(
        &self,
        array: &ArrayBase<S, D>,
        device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        S: RawData<Elem = T>,
        D: Dimension,
    {
        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {} is not available", device))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory manager for the device
        let manager = self.memory_managers.get(&device).unwrap();
        manager.transfer_to_device(array, device, options)
    }

    /// Transfer data from a device to the host
    pub fn to_host<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        options: Option<TransferOptions>,
    ) -> CoreResult<Array<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Get the memory manager for the device
        let manager = self
            .memory_managers
            .get(&device_array.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", device_array.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.transfer_to_host(device_array, options)
    }

    /// Transfer data between devices
    pub fn transfer<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        target_device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Check if the target device is available
        if !self.is_device_available(target_device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {} is not available", target_device))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory manager for the source device
        let manager = self
            .memory_managers
            .get(&device_array.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", device_array.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.transfer_between_devices(device_array, target_device, options)
    }

    /// Execute a kernel on a device array
    pub fn execute_kernel<T, D>(
        &self,
        device_array: &DeviceArray<T, D>,
        kernel_name: &str,
        params: HashMap<String, KernelParam>,
    ) -> CoreResult<()>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Get the memory manager for the device
        let manager = self
            .memory_managers
            .get(&device_array.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", device_array.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.execute_kernel(device_array, kernel_name, params)
    }

    /// Allocate memory on a device
    pub fn allocate<T: GpuDataType>(
        &self,
        size: usize,
        device: DeviceType,
    ) -> CoreResult<DeviceBuffer<T>> {
        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {} is not available", device))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory pool for the device
        let pool = self.memory_pools.get(&device).unwrap();
        pool.allocate(size)
    }

    /// Free memory on a device
    pub fn free<T: GpuDataType>(&self, buffer: DeviceBuffer<T>, device: DeviceType) {
        // Check if the device is available
        if !self.is_device_available(device) {
            return;
        }

        // Get the memory pool for the device
        let pool = self.memory_pools.get(&device).unwrap();
        pool.free(buffer);
    }

    /// Clear all caches and pools
    pub fn clear(&self) {
        // Clear all memory managers
        for (_, manager) in &self.memory_managers {
            manager.clear_cache();
        }

        // Clear all memory pools
        for (_, pool) in &self.memory_pools {
            pool.clear();
        }

        // Clear active transfers
        let mut active_transfers = self.active_transfers.lock().unwrap();
        active_transfers.clear();
    }

    /// Wait for all active transfers to complete
    pub fn synchronize(&self) {
        let mut active_transfers = self.active_transfers.lock().unwrap();
        for event in active_transfers.drain(..) {
            event.wait();
        }
    }
}

/// Create a cross-device manager with default settings
pub fn create_cross_device_manager() -> CoreResult<CrossDeviceManager> {
    CrossDeviceManager::new(true, 1024 * 1024 * 1024) // 1 GB cache by default
}

/// Extension trait for arrays to simplify device transfers
pub trait ToDevice<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    /// Transfer the array to a device
    fn to_device(
        &self,
        device: DeviceType,
        manager: &CrossDeviceManager,
    ) -> CoreResult<DeviceArray<T, D>>;
}

impl<T, S, D> ToDevice<T, D> for ArrayBase<S, D>
where
    T: GpuDataType,
    S: RawData<Elem = T>,
    D: Dimension,
{
    fn to_device(
        &self,
        device: DeviceType,
        manager: &CrossDeviceManager,
    ) -> CoreResult<DeviceArray<T, D>> {
        manager.to_device(self, device, None)
    }
}

/// Extension trait for device arrays to simplify host transfers
pub trait ToHost<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    /// Transfer the device array to the host
    fn to_host(&self, manager: &CrossDeviceManager) -> CoreResult<Array<T, D>>;
}

impl<T, D> ToHost<T, D> for DeviceArray<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    fn to_host(&self, manager: &CrossDeviceManager) -> CoreResult<Array<T, D>> {
        manager.to_host(self, None)
    }
}

// Convenience functions

/// Create a device array on the CPU
pub fn create_cpu_array<T, S, D>(array: &ArrayBase<S, D>) -> DeviceArray<T, D>
where
    T: GpuDataType,
    S: RawData<Elem = T>,
    D: Dimension,
{
    DeviceArray::new_cpu(array.to_owned())
}

/// Create a device array on the GPU
pub fn create_gpu_array<T, S, D>(
    array: &ArrayBase<S, D>,
    manager: &CrossDeviceManager,
) -> CoreResult<DeviceArray<T, D>>
where
    T: GpuDataType,
    S: RawData<Elem = T>,
    D: Dimension,
{
    // Find the first available GPU
    for device in manager.available_devices() {
        if let DeviceType::Gpu(_) = device {
            return manager.to_device(array, device, None);
        }
    }

    Err(CoreError::DeviceError(
        ErrorContext::new("No GPU device available".to_string())
            .with_location(ErrorLocation::new(file!(), line!())),
    ))
}

/// Transfer an array to the best available device
pub fn to_best_device<T, S, D>(
    array: &ArrayBase<S, D>,
    manager: &CrossDeviceManager,
) -> CoreResult<DeviceArray<T, D>>
where
    T: GpuDataType,
    S: RawData<Elem = T>,
    D: Dimension,
{
    // Try to find a GPU first
    for device in manager.available_devices() {
        if let DeviceType::Gpu(_) = device {
            return manager.to_device(array, device, None);
        }
    }

    // Fall back to CPU
    Ok(create_cpu_array(array))
}
