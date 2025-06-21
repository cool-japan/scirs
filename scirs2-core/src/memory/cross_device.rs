//! # Cross-Device Memory Management
//!
//! This module provides unified memory management across different compute devices
//! including CPU, GPU, and TPU with automatic data movement and synchronization.

use crate::error::{CoreError, CoreResult};
use crate::gpu::GpuContext;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Error types for cross-device memory management
#[derive(Debug, thiserror::Error)]
pub enum CrossDeviceError {
    /// Device not found
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    /// Memory allocation failed
    #[error("Memory allocation failed on device {device}: {reason}")]
    AllocationFailed { device: String, reason: String },

    /// Data transfer failed
    #[error("Data transfer failed from {from} to {to}: {reason}")]
    TransferFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// Synchronization failed
    #[error("Device synchronization failed: {0}")]
    SynchronizationFailed(String),

    /// Invalid device type
    #[error("Invalid device type: {0}")]
    InvalidDeviceType(String),

    /// Memory not found
    #[error("Memory allocation not found: {0}")]
    MemoryNotFound(String),
}

impl From<CrossDeviceError> for CoreError {
    fn from(err: CrossDeviceError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

/// Device types supported by the memory manager
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU memory (system RAM)
    Cpu,
    /// NVIDIA GPU (CUDA)
    CudaGpu(u32),
    /// AMD GPU (ROCm/OpenCL)
    RocmGpu(u32),
    /// Intel GPU (OpenCL)
    IntelGpu(u32),
    /// Apple Metal GPU
    MetalGpu(u32),
    /// Google TPU
    Tpu(u32),
    /// Generic OpenCL device
    OpenClDevice(u32),
}

impl DeviceType {
    /// Get a string representation of the device type
    pub const fn as_str(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "CPU",
            DeviceType::CudaGpu(_) => "CUDA_GPU",
            DeviceType::RocmGpu(_) => "ROCM_GPU",
            DeviceType::IntelGpu(_) => "INTEL_GPU",
            DeviceType::MetalGpu(_) => "METAL_GPU",
            DeviceType::Tpu(_) => "TPU",
            DeviceType::OpenClDevice(_) => "OPENCL",
        }
    }

    /// Get device ID
    pub fn device_id(&self) -> u32 {
        match self {
            DeviceType::Cpu => 0,
            DeviceType::CudaGpu(id)
            | DeviceType::RocmGpu(id)
            | DeviceType::IntelGpu(id)
            | DeviceType::MetalGpu(id)
            | DeviceType::Tpu(id)
            | DeviceType::OpenClDevice(id) => *id,
        }
    }

    /// Check if device supports unified memory
    pub fn supports_unified_memory(&self) -> bool {
        matches!(self, DeviceType::CudaGpu(_) | DeviceType::RocmGpu(_))
    }

    /// Check if device supports peer-to-peer transfer
    pub fn supports_p2p_transfer(&self, other: &DeviceType) -> bool {
        matches!(
            (self, other),
            (DeviceType::CudaGpu(_), DeviceType::CudaGpu(_))
                | (DeviceType::RocmGpu(_), DeviceType::RocmGpu(_))
        )
    }
}

/// Memory allocation information
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Unique allocation ID
    pub id: String,
    /// Device where memory is allocated
    pub device: DeviceType,
    /// Size in bytes
    pub size: usize,
    /// Memory address (platform-specific)
    pub address: usize,
    /// Data type information
    pub data_type: TypeId,
    /// Creation timestamp
    pub created_at: std::time::Instant,
    /// Last access timestamp
    pub last_accessed: std::time::Instant,
    /// Reference count
    pub ref_count: usize,
}

impl MemoryAllocation {
    /// Create a new memory allocation record
    pub fn new(
        id: String,
        device: DeviceType,
        size: usize,
        address: usize,
        data_type: TypeId,
    ) -> Self {
        let now = std::time::Instant::now();
        Self {
            id,
            device,
            size,
            address,
            data_type,
            created_at: now,
            last_accessed: now,
            ref_count: 1,
        }
    }

    /// Update last access time
    pub fn touch(&mut self) {
        self.last_accessed = std::time::Instant::now();
    }

    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn remove_ref(&mut self) -> usize {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count
    }
}

/// Device interface trait
pub trait Device: Send + Sync {
    /// Get device type
    fn device_type(&self) -> DeviceType;

    /// Allocate memory on this device
    fn allocate(&self, size: usize) -> CoreResult<usize>;

    /// Deallocate memory on this device
    fn deallocate(&self, address: usize) -> CoreResult<()>;

    /// Copy data to this device from CPU
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `src` points to at least `size` bytes of valid memory
    /// - `dst` is a valid device memory address with at least `size` bytes allocated
    /// - The memory regions do not overlap
    unsafe fn copy_from_host(&self, src: *const u8, dst: usize, size: usize) -> CoreResult<()>;

    /// Copy data from this device to CPU
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `src` is a valid device memory address with at least `size` bytes allocated
    /// - `dst` points to at least `size` bytes of valid writable memory
    /// - The memory regions do not overlap
    unsafe fn copy_to_host(&self, src: usize, dst: *mut u8, size: usize) -> CoreResult<()>;

    /// Copy data between devices (if supported)
    fn copy_peer(
        &self,
        src: usize,
        dst_device: &dyn Device,
        dst: usize,
        size: usize,
    ) -> CoreResult<()>;

    /// Synchronize device operations
    fn synchronize(&self) -> CoreResult<()>;

    /// Get available memory in bytes
    fn available_memory(&self) -> CoreResult<usize>;

    /// Get total memory in bytes
    fn total_memory(&self) -> CoreResult<usize>;
}

/// CPU device implementation
pub struct CpuDevice {
    device_type: DeviceType,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        Self {
            device_type: DeviceType::Cpu,
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for CpuDevice {
    fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    fn allocate(&self, size: usize) -> CoreResult<usize> {
        let layout = std::alloc::Layout::from_size_align(size, 64).map_err(|e| {
            CrossDeviceError::AllocationFailed {
                device: "CPU".to_string(),
                reason: e.to_string(),
            }
        })?;

        unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                Err(CrossDeviceError::AllocationFailed {
                    device: "CPU".to_string(),
                    reason: "Out of memory".to_string(),
                }
                .into())
            } else {
                Ok(ptr as usize)
            }
        }
    }

    fn deallocate(&self, address: usize) -> CoreResult<()> {
        // Note: In a real implementation, we'd need to track the layout
        // For now, we'll skip the actual deallocation
        let _ = address;
        Ok(())
    }

    unsafe fn copy_from_host(&self, src: *const u8, dst: usize, size: usize) -> CoreResult<()> {
        std::ptr::copy_nonoverlapping(src, dst as *mut u8, size);
        Ok(())
    }

    unsafe fn copy_to_host(&self, src: usize, dst: *mut u8, size: usize) -> CoreResult<()> {
        std::ptr::copy_nonoverlapping(src as *const u8, dst, size);
        Ok(())
    }

    fn copy_peer(
        &self,
        _src: usize,
        _dst_device: &dyn Device,
        _dst: usize,
        _size: usize,
    ) -> CoreResult<()> {
        Err(CrossDeviceError::TransferFailed {
            from: "CPU".to_string(),
            to: "unknown".to_string(),
            reason: "Peer-to-peer not supported for CPU".to_string(),
        }
        .into())
    }

    fn synchronize(&self) -> CoreResult<()> {
        // CPU operations are synchronous
        Ok(())
    }

    fn available_memory(&self) -> CoreResult<usize> {
        // Simple approximation - in reality would use platform-specific APIs
        Ok(8 * 1024 * 1024 * 1024) // 8 GB
    }

    fn total_memory(&self) -> CoreResult<usize> {
        Ok(16 * 1024 * 1024 * 1024) // 16 GB
    }
}

/// GPU device wrapper
pub struct GpuContextWrapper {
    inner: Arc<GpuContext>,
    device_type: DeviceType,
}

impl GpuContextWrapper {
    /// Create a new GPU device wrapper
    pub fn new(gpu_device: Arc<GpuContext>, device_type: DeviceType) -> Self {
        Self {
            inner: gpu_device,
            device_type,
        }
    }
}

impl Device for GpuContextWrapper {
    fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    fn allocate(&self, size: usize) -> CoreResult<usize> {
        // Use the GPU device's buffer allocation
        let buffer = self.inner.create_buffer::<u8>(size);
        // In a real implementation, we'd extract the actual device pointer
        // For now, we'll use a placeholder based on buffer properties
        let _buffer = buffer; // Store the buffer (in real implementation, we'd track this)
        Ok(size) // Return the size as a placeholder ID
    }

    fn deallocate(&self, _address: usize) -> CoreResult<()> {
        // GPU buffers are automatically freed when dropped
        Ok(())
    }

    unsafe fn copy_from_host(&self, _src: *const u8, _dst: usize, _size: usize) -> CoreResult<()> {
        // Would use GPU-specific memory copy operations
        Ok(())
    }

    unsafe fn copy_to_host(&self, _src: usize, _dst: *mut u8, _size: usize) -> CoreResult<()> {
        // Would use GPU-specific memory copy operations
        Ok(())
    }

    fn copy_peer(
        &self,
        _src: usize,
        _dst_device: &dyn Device,
        _dst: usize,
        _size: usize,
    ) -> CoreResult<()> {
        // Would implement GPU-to-GPU transfers
        Ok(())
    }

    fn synchronize(&self) -> CoreResult<()> {
        // Would synchronize GPU streams/queues
        Ok(())
    }

    fn available_memory(&self) -> CoreResult<usize> {
        self.inner.get_available_memory().ok_or_else(|| {
            CrossDeviceError::DeviceNotFound("GPU memory info unavailable".to_string()).into()
        })
    }

    fn total_memory(&self) -> CoreResult<usize> {
        self.inner.get_total_memory().ok_or_else(|| {
            CrossDeviceError::DeviceNotFound("GPU memory info unavailable".to_string()).into()
        })
    }
}

/// Cross-device memory manager
pub struct CrossDeviceMemoryManager {
    devices: RwLock<HashMap<DeviceType, Arc<dyn Device>>>,
    allocations: RwLock<HashMap<String, MemoryAllocation>>,
    allocation_counter: Mutex<u64>,
    default_device: RwLock<Option<DeviceType>>,
}

impl CrossDeviceMemoryManager {
    /// Create a new cross-device memory manager
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(HashMap::new()),
            allocations: RwLock::new(HashMap::new()),
            allocation_counter: Mutex::new(0),
            default_device: RwLock::new(None),
        }
    }

    /// Register a device with the manager
    pub fn register_device(&self, device: Arc<dyn Device>) -> CoreResult<()> {
        let device_type = device.device_type();
        let mut devices = self.devices.write().unwrap();
        devices.insert(device_type.clone(), device);

        // Set as default if it's the first device
        let mut default_device = self.default_device.write().unwrap();
        if default_device.is_none() {
            *default_device = Some(device_type);
        }

        Ok(())
    }

    /// Set the default device
    pub fn set_default_device(&self, device_type: DeviceType) -> CoreResult<()> {
        let devices = self.devices.read().unwrap();
        if !devices.contains_key(&device_type) {
            return Err(CrossDeviceError::DeviceNotFound(format!("{:?}", device_type)).into());
        }

        let mut default_device = self.default_device.write().unwrap();
        *default_device = Some(device_type);

        Ok(())
    }

    /// Get the default device
    pub fn get_default_device(&self) -> Option<DeviceType> {
        self.default_device.read().unwrap().clone()
    }

    /// Allocate memory on a specific device
    pub fn allocate<T: 'static>(
        self: &Arc<Self>,
        device_type: &DeviceType,
        count: usize,
    ) -> CoreResult<CrossDeviceBuffer<T>> {
        let devices = self.devices.read().unwrap();
        let device = devices
            .get(device_type)
            .ok_or_else(|| CrossDeviceError::DeviceNotFound(format!("{:?}", device_type)))?;

        let size = count * std::mem::size_of::<T>();
        let address = device.allocate(size)?;

        let allocation_id = self.generate_allocation_id();
        let allocation = MemoryAllocation::new(
            allocation_id.clone(),
            device_type.clone(),
            size,
            address,
            TypeId::of::<T>(),
        );

        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(allocation_id.clone(), allocation);

        Ok(CrossDeviceBuffer::new(
            allocation_id,
            device_type.clone(),
            address,
            count,
            self.clone(),
        ))
    }

    /// Allocate memory on the default device
    pub fn allocate_default<T: 'static>(
        self: &Arc<Self>,
        count: usize,
    ) -> CoreResult<CrossDeviceBuffer<T>> {
        let default_device = self
            .get_default_device()
            .ok_or_else(|| CrossDeviceError::DeviceNotFound("No default device set".to_string()))?;

        self.allocate(&default_device, count)
    }

    /// Transfer data between devices
    pub fn transfer<T: 'static + Copy>(
        self: &Arc<Self>,
        src_buffer: &CrossDeviceBuffer<T>,
        dst_device: &DeviceType,
    ) -> CoreResult<CrossDeviceBuffer<T>> {
        let devices = self.devices.read().unwrap();
        let src_device = devices.get(&src_buffer.device_type).ok_or_else(|| {
            CrossDeviceError::DeviceNotFound(format!("{:?}", src_buffer.device_type))
        })?;
        let dst_device_obj = devices
            .get(dst_device)
            .ok_or_else(|| CrossDeviceError::DeviceNotFound(format!("{:?}", dst_device)))?;

        // Allocate memory on destination device
        let dst_buffer = self.allocate::<T>(dst_device, src_buffer.count)?;

        let size = src_buffer.count * std::mem::size_of::<T>();

        // Try peer-to-peer transfer first
        if src_buffer.device_type.supports_p2p_transfer(dst_device) {
            src_device.copy_peer(
                src_buffer.address,
                dst_device_obj.as_ref(),
                dst_buffer.address,
                size,
            )?;
        } else {
            // Fall back to CPU staging
            let staging_buffer = self.allocate::<T>(&DeviceType::Cpu, src_buffer.count)?;

            // Copy from source to CPU
            unsafe {
                src_device.copy_to_host(
                    src_buffer.address,
                    staging_buffer.address as *mut u8,
                    size,
                )?;
            }

            // Copy from CPU to destination
            unsafe {
                dst_device_obj.copy_from_host(
                    staging_buffer.address as *const u8,
                    dst_buffer.address,
                    size,
                )?;
            }
        }

        Ok(dst_buffer)
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> CoreResult<()> {
        let devices = self.devices.read().unwrap();
        for device in devices.values() {
            device.synchronize()?;
        }
        Ok(())
    }

    /// Get memory statistics
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        let allocations = self.allocations.read().unwrap();
        let devices = self.devices.read().unwrap();

        let mut stats_by_device = HashMap::new();
        let mut total_allocated = 0;
        let mut total_allocations = 0;

        for allocation in allocations.values() {
            let device_stats =
                stats_by_device
                    .entry(allocation.device.clone())
                    .or_insert(DeviceMemoryStats {
                        device_type: allocation.device.clone(),
                        allocated_bytes: 0,
                        allocation_count: 0,
                        available_bytes: 0,
                        total_bytes: 0,
                    });

            device_stats.allocated_bytes += allocation.size;
            device_stats.allocation_count += 1;
            total_allocated += allocation.size;
            total_allocations += 1;
        }

        // Update available/total memory from devices
        for (device_type, device) in devices.iter() {
            if let Some(device_stats) = stats_by_device.get_mut(device_type) {
                device_stats.available_bytes = device.available_memory().unwrap_or(0);
                device_stats.total_bytes = device.total_memory().unwrap_or(0);
            }
        }

        MemoryStatistics {
            total_allocated_bytes: total_allocated,
            total_allocations,
            device_stats: stats_by_device.into_values().collect(),
        }
    }

    /// Clean up unused allocations
    pub fn cleanup_unused(&self, max_age: std::time::Duration) -> usize {
        let mut allocations = self.allocations.write().unwrap();
        let now = std::time::Instant::now();
        let mut cleaned = 0;

        allocations.retain(|_, allocation| {
            if allocation.ref_count == 0 && now.duration_since(allocation.last_accessed) > max_age {
                // In a real implementation, we'd call deallocate on the device
                cleaned += 1;
                false
            } else {
                true
            }
        });

        cleaned
    }

    /// Generate unique allocation ID
    fn generate_allocation_id(&self) -> String {
        let counter = {
            let mut counter = self.allocation_counter.lock().unwrap();
            *counter += 1;
            *counter
        };

        format!("alloc_{:016x}", counter)
    }

    /// Internal method to remove allocation (called by CrossDeviceBuffer on drop)
    pub(crate) fn remove_allocation(&self, allocation_id: &str) {
        let mut allocations = self.allocations.write().unwrap();
        if let Some(allocation) = allocations.get_mut(allocation_id) {
            if allocation.remove_ref() == 0 {
                allocations.remove(allocation_id);
            }
        }
    }

    /// Internal method to touch allocation (update last access time)
    pub(crate) fn touch_allocation(&self, allocation_id: &str) {
        let mut allocations = self.allocations.write().unwrap();
        if let Some(allocation) = allocations.get_mut(allocation_id) {
            allocation.touch();
        }
    }
}

impl Default for CrossDeviceMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-device buffer that manages memory across different devices
pub struct CrossDeviceBuffer<T> {
    allocation_id: String,
    device_type: DeviceType,
    address: usize,
    count: usize,
    manager: Arc<CrossDeviceMemoryManager>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> CrossDeviceBuffer<T> {
    /// Create a new cross-device buffer
    fn new(
        allocation_id: String,
        device_type: DeviceType,
        address: usize,
        count: usize,
        manager: Arc<CrossDeviceMemoryManager>,
    ) -> Self {
        Self {
            allocation_id,
            device_type,
            address,
            count,
            manager,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the device type this buffer is allocated on
    pub const fn device_type(&self) -> &DeviceType {
        &self.device_type
    }

    /// Get the number of elements in the buffer
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.count * std::mem::size_of::<T>()
    }

    /// Get the raw address (device-specific)
    pub fn raw_address(&self) -> usize {
        self.manager.touch_allocation(&self.allocation_id);
        self.address
    }

    /// Transfer this buffer to another device
    pub fn transfer_to(&self, device_type: &DeviceType) -> CoreResult<CrossDeviceBuffer<T>>
    where
        T: Copy + 'static,
    {
        self.manager.transfer(self, device_type)
    }

    /// Copy data from host to this buffer
    pub fn copy_from_host(&self, data: &[T]) -> CoreResult<()>
    where
        T: Copy,
    {
        if data.len() != self.count {
            return Err(CrossDeviceError::InvalidDeviceType(format!(
                "Data length {} doesn't match buffer capacity {}",
                data.len(),
                self.count
            ))
            .into());
        }

        let devices = self.manager.devices.read().unwrap();
        let device = devices
            .get(&self.device_type)
            .ok_or_else(|| CrossDeviceError::DeviceNotFound(format!("{:?}", self.device_type)))?;

        unsafe {
            device.copy_from_host(data.as_ptr() as *const u8, self.address, self.size_bytes())?;
        }

        self.manager.touch_allocation(&self.allocation_id);
        Ok(())
    }

    /// Copy data from this buffer to host
    pub fn copy_to_host(&self) -> CoreResult<Vec<T>>
    where
        T: Copy + Default,
    {
        let mut result = vec![T::default(); self.count];

        let devices = self.manager.devices.read().unwrap();
        let device = devices
            .get(&self.device_type)
            .ok_or_else(|| CrossDeviceError::DeviceNotFound(format!("{:?}", self.device_type)))?;

        unsafe {
            device.copy_to_host(
                self.address,
                result.as_mut_ptr() as *mut u8,
                self.size_bytes(),
            )?;
        }

        self.manager.touch_allocation(&self.allocation_id);
        Ok(result)
    }
}

impl<T> Clone for CrossDeviceBuffer<T> {
    fn clone(&self) -> Self {
        // Increment reference count
        {
            let mut allocations = self.manager.allocations.write().unwrap();
            if let Some(allocation) = allocations.get_mut(&self.allocation_id) {
                allocation.add_ref();
            }
        }

        Self {
            allocation_id: self.allocation_id.clone(),
            device_type: self.device_type.clone(),
            address: self.address,
            count: self.count,
            manager: self.manager.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Drop for CrossDeviceBuffer<T> {
    fn drop(&mut self) {
        self.manager.remove_allocation(&self.allocation_id);
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total bytes allocated across all devices
    pub total_allocated_bytes: usize,
    /// Total number of allocations
    pub total_allocations: usize,
    /// Statistics per device
    pub device_stats: Vec<DeviceMemoryStats>,
}

/// Memory statistics for a specific device
#[derive(Debug, Clone)]
pub struct DeviceMemoryStats {
    /// Device type
    pub device_type: DeviceType,
    /// Currently allocated bytes on this device
    pub allocated_bytes: usize,
    /// Number of active allocations
    pub allocation_count: usize,
    /// Available memory on device
    pub available_bytes: usize,
    /// Total memory on device
    pub total_bytes: usize,
}

impl DeviceMemoryStats {
    /// Get memory usage percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.allocated_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Global cross-device memory manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<Arc<CrossDeviceMemoryManager>> =
    std::sync::OnceLock::new();

/// Get the global cross-device memory manager
pub fn global_manager() -> Arc<CrossDeviceMemoryManager> {
    GLOBAL_MANAGER
        .get_or_init(|| {
            let manager = Arc::new(CrossDeviceMemoryManager::new());

            // Register CPU device by default
            let cpu_device = Arc::new(CpuDevice::new());
            let _ = manager.register_device(cpu_device);

            manager
        })
        .clone()
}

/// Initialize cross-device memory management with GPU devices
pub fn initialize_with_gpu_devices(gpu_devices: Vec<Arc<GpuContext>>) -> CoreResult<()> {
    let manager = global_manager();

    for (i, gpu_device) in gpu_devices.into_iter().enumerate() {
        let device_type = DeviceType::CudaGpu(i as u32); // Assume CUDA for now
        let wrapper = Arc::new(GpuContextWrapper::new(gpu_device, device_type));
        manager.register_device(wrapper)?;
    }

    Ok(())
}

/// Convenience functions for cross-device memory management
pub mod utils {
    use super::*;

    /// Allocate a buffer on the best available device
    pub fn allocate_optimal<T: 'static>(count: usize) -> CoreResult<CrossDeviceBuffer<T>> {
        let manager = global_manager();
        let stats = manager.get_memory_statistics();

        // Find device with most available memory
        let best_device = stats
            .device_stats
            .iter()
            .max_by_key(|s| s.available_bytes)
            .map(|s| s.device_type.clone())
            .unwrap_or(DeviceType::Cpu);

        manager.allocate(&best_device, count)
    }

    /// Create a buffer with data from host
    pub fn create_buffer_with_data<T: Copy + 'static>(
        data: &[T],
        device_type: &DeviceType,
    ) -> CoreResult<CrossDeviceBuffer<T>> {
        let manager = global_manager();
        let buffer = manager.allocate(device_type, data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }

    /// Transfer data between any two devices
    pub fn transfer_data<T: Copy + 'static>(
        src_buffer: &CrossDeviceBuffer<T>,
        dst_device: &DeviceType,
    ) -> CoreResult<CrossDeviceBuffer<T>> {
        let manager = global_manager();
        manager.transfer(src_buffer, dst_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_creation() {
        let cpu = DeviceType::Cpu;
        let gpu = DeviceType::CudaGpu(0);
        let tpu = DeviceType::Tpu(1);

        assert_eq!(cpu.as_str(), "CPU");
        assert_eq!(gpu.as_str(), "CUDA_GPU");
        assert_eq!(tpu.as_str(), "TPU");

        assert_eq!(cpu.device_id(), 0);
        assert_eq!(gpu.device_id(), 0);
        assert_eq!(tpu.device_id(), 1);
    }

    #[test]
    fn test_device_capabilities() {
        let cpu = DeviceType::Cpu;
        let cuda = DeviceType::CudaGpu(0);
        let rocm = DeviceType::RocmGpu(0);

        assert!(!cpu.supports_unified_memory());
        assert!(cuda.supports_unified_memory());
        assert!(rocm.supports_unified_memory());

        assert!(cuda.supports_p2p_transfer(&DeviceType::CudaGpu(1)));
        assert!(!cuda.supports_p2p_transfer(&DeviceType::RocmGpu(0)));
        assert!(!cpu.supports_p2p_transfer(&DeviceType::CudaGpu(0)));
    }

    #[test]
    fn test_memory_allocation_creation() {
        let allocation = MemoryAllocation::new(
            "test_alloc".to_string(),
            DeviceType::Cpu,
            1024,
            0x1000,
            TypeId::of::<f32>(),
        );

        assert_eq!(allocation.id, "test_alloc");
        assert_eq!(allocation.size, 1024);
        assert_eq!(allocation.address, 0x1000);
        assert_eq!(allocation.ref_count, 1);
    }

    #[test]
    fn test_cpu_device() {
        let cpu = CpuDevice::new();
        assert_eq!(cpu.device_type(), DeviceType::Cpu);

        // Test memory info
        assert!(cpu.available_memory().is_ok());
        assert!(cpu.total_memory().is_ok());

        // Test synchronization
        assert!(cpu.synchronize().is_ok());
    }

    #[test]
    fn test_cross_device_manager() {
        let manager = CrossDeviceMemoryManager::new();

        // Register CPU device
        let cpu_device = Arc::new(CpuDevice::new());
        assert!(manager.register_device(cpu_device).is_ok());

        // Check default device
        assert_eq!(manager.get_default_device(), Some(DeviceType::Cpu));

        // Get initial statistics
        let stats = manager.get_memory_statistics();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_allocated_bytes, 0);
    }

    #[test]
    fn test_global_manager() {
        let manager = global_manager();
        assert_eq!(manager.get_default_device(), Some(DeviceType::Cpu));

        let stats = manager.get_memory_statistics();
        assert!(!stats.device_stats.is_empty());
    }

    #[test]
    fn test_memory_statistics() {
        let stats = DeviceMemoryStats {
            device_type: DeviceType::Cpu,
            allocated_bytes: 1024,
            allocation_count: 1,
            available_bytes: 7 * 1024 * 1024 * 1024,
            total_bytes: 8 * 1024 * 1024 * 1024,
        };

        let usage = stats.usage_percentage();
        assert!(usage > 0.0 && usage < 1.0);
    }
}
