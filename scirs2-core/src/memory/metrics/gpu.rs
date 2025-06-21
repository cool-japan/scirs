//! Memory metrics integration with GPU operations
//!
//! This module provides memory tracking for GPU buffers and operations.

use crate::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuKernelHandle};
use crate::memory::metrics::{track_allocation, track_deallocation};

/// A tracked GPU buffer that monitors memory usage
pub struct TrackedGpuBuffer<T: GpuDataType> {
    /// Inner GPU buffer
    inner: GpuBuffer<T>,
    /// Component name for tracking
    component_name: String,
    /// Size in bytes
    size_bytes: usize,
    /// Flag to track if this is being actively tracked
    is_tracked: bool,
}

impl<T: GpuDataType> TrackedGpuBuffer<T> {
    /// Create a new tracked GPU buffer from an existing buffer
    pub fn new(buffer: GpuBuffer<T>, component_name: impl Into<String>) -> Self {
        let size_bytes = buffer.len() * std::mem::size_of::<T>();
        let component_name = component_name.into();

        // Track the initial allocation
        track_allocation(&component_name, size_bytes, &buffer as *const _ as usize);

        Self {
            inner: buffer,
            component_name,
            size_bytes,
            is_tracked: true,
        }
    }

    /// Get the size of the buffer in elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the size of the buffer in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the component name used for tracking
    pub fn component_name(&self) -> &str {
        &self.component_name
    }

    /// Copy data from the host to the device
    pub fn copy_from_host(&self, data: &[T]) {
        self.inner.copy_from_host(data);
    }

    /// Copy data from the device to the host
    pub fn copy_to_host(&self, data: &mut [T]) {
        self.inner.copy_to_host(data);
    }

    /// Convert the buffer contents to a vector
    pub fn to_vec(&self) -> Vec<T> {
        self.inner.to_vec()
    }

    /// Get the inner buffer
    pub const fn inner(&self) -> &GpuBuffer<T> {
        &self.inner
    }

    /// Stop tracking this buffer without deallocating it
    ///
    /// This is useful when passing ownership to another system
    pub fn stop_tracking(&mut self) {
        if self.is_tracked {
            track_deallocation(
                &self.component_name,
                self.size_bytes,
                &self.inner as *const _ as usize,
            );
            self.is_tracked = false;
        }
    }

    /// Resume tracking this buffer
    pub fn resume_tracking(&mut self) {
        if !self.is_tracked {
            track_allocation(
                &self.component_name,
                self.size_bytes,
                &self.inner as *const _ as usize,
            );
            self.is_tracked = true;
        }
    }
}

impl<T: GpuDataType> Drop for TrackedGpuBuffer<T> {
    fn drop(&mut self) {
        // Only track deallocation if we're still tracking
        if self.is_tracked {
            track_deallocation(
                &self.component_name,
                self.size_bytes,
                &self.inner as *const _ as usize,
            );
        }
    }
}

/// A tracked GPU context that creates tracked buffers
pub struct TrackedGpuContext {
    /// Inner GPU context
    inner: GpuContext,
    /// Component name for tracking
    component_name: String,
}

impl TrackedGpuContext {
    /// Create a new tracked GPU context
    pub fn new(inner: GpuContext, component_name: impl Into<String>) -> Self {
        Self {
            inner,
            component_name: component_name.into(),
        }
    }

    /// Create a tracked GPU context with the specified backend
    pub fn with_backend(
        backend: GpuBackend,
        component_name: impl Into<String>,
    ) -> Result<Self, crate::gpu::GpuError> {
        let context = GpuContext::new(backend)?;
        Ok(Self::new(context, component_name))
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.inner.backend()
    }

    /// Get the backend name
    pub const fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    /// Create a tracked buffer with the given size
    pub fn create_buffer<T: GpuDataType>(&self, size: usize) -> TrackedGpuBuffer<T> {
        let buffer = self.inner.create_buffer::<T>(size);
        let buffer_name = format!("{}::Buffer_{:p}", self.component_name, &buffer);
        TrackedGpuBuffer::new(buffer, buffer_name)
    }

    /// Create a tracked buffer from a slice
    pub fn create_buffer_from_slice<T: GpuDataType>(&self, data: &[T]) -> TrackedGpuBuffer<T> {
        let buffer = self.inner.create_buffer_from_slice(data);
        let buffer_name = format!("{}::Buffer_{:p}", self.component_name, &buffer);
        TrackedGpuBuffer::new(buffer, buffer_name)
    }

    /// Execute a function with a compiler
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&crate::gpu::GpuCompiler) -> R,
    {
        self.inner.execute(f)
    }

    /// Get a kernel from the registry
    pub fn get_kernel(&self, name: &str) -> Result<GpuKernelHandle, crate::gpu::GpuError> {
        self.inner.get_kernel(name)
    }

    /// Get a specialized kernel from the registry
    pub fn get_specialized_kernel(
        &self,
        name: &str,
        params: &crate::gpu::kernels::KernelParams,
    ) -> Result<GpuKernelHandle, crate::gpu::GpuError> {
        self.inner.get_specialized_kernel(name, params)
    }

    /// Get the inner context
    pub const fn inner(&self) -> &GpuContext {
        &self.inner
    }
}

/// Create memory allocation tracking hooks for GPU operations
///
/// This sets up tracking hooks for CUDA, WebGPU, Metal, and OpenCL backends
/// and reports memory allocations and deallocations to the memory metrics system.
pub fn setup_gpu_memory_tracking() {
    // In a real implementation, this would set up hooks to track GPU memory
    // This would be different for each backend

    // Here's a sketch of what it would do:
    // 1. Register memory allocation hooks with each GPU API
    // 2. Forward memory events to the metrics system
    // 3. Periodically poll for GPU memory usage stats

    // For now, this is just a placeholder since we can only track
    // what we create through our own API
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::metrics::{generate_memory_report, reset_memory_metrics};

    #[test]
    fn test_tracked_gpu_buffer() {
        // Only run this test if CPU backend is available
        let context = match GpuContext::new(GpuBackend::Cpu) {
            Ok(ctx) => ctx,
            Err(_) => return, // Skip test if no GPU backend is available
        };

        // Reset metrics
        reset_memory_metrics();

        // Create a tracked GPU context
        let tracked_ctx = TrackedGpuContext::new(context, "GpuTests");

        // Create a buffer
        let buffer_size = 1000;
        let element_size = std::mem::size_of::<f32>();
        let buffer = tracked_ctx.create_buffer::<f32>(buffer_size);

        // Check that allocation was tracked
        let report = generate_memory_report();
        assert!(report.total_current_usage > 0);
        assert_eq!(report.total_allocation_count, 1);

        // Get the buffer name from the report
        let buffer_component = report
            .component_stats
            .keys()
            .find(|name| name.starts_with("GpuTests::Buffer_"))
            .expect("Should have a buffer component");

        // Verify buffer size
        let component_stats = &report.component_stats[buffer_component];
        assert_eq!(component_stats.current_usage, buffer_size * element_size);

        // Drop the buffer
        drop(buffer);

        // Check that deallocation was tracked
        let report = generate_memory_report();
        assert_eq!(report.total_current_usage, 0);
    }
}
