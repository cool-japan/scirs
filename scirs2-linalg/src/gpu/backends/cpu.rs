//! CPU fallback backend implementation
//!
//! This module provides a CPU fallback backend that implements GPU traits using CPU operations.
//! This is useful when no GPU backends are available or for testing purposes.

use super::common::*;

/// CPU fallback backend that implements GPU traits using CPU operations
pub struct CpuFallbackBackend {
    device_info: GpuDeviceInfo,
}

impl Default for CpuFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFallbackBackend {
    pub fn new() -> Self {
        Self {
            device_info: GpuDeviceInfo {
                device_type: GpuDeviceType::OpenCl, // Use OpenCL as generic type
                name: "CPU Fallback".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB estimate
                compute_units: num_cpus::get() as u32,
                clock_frequency: 3000, // 3GHz estimate
                supports_fp64: true,
                supports_fp16: false,
                max_work_groupsize: 1024,
                memory_bandwidth: 100.0, // CPU memory bandwidth estimate
                l2_cachesize: 32 * 1024 * 1024, // 32MB L2 cache estimate
                shared_memory_per_block: 0, // No shared memory concept for CPU
                registers_per_block: 0,
                warpsize: 1, // No SIMD grouping for CPU
                max_threads_per_mp: 1,
                multiprocessor_count: num_cpus::get() as u32,
                supports_tensor_cores: false,
                supports_mixed_precision: false,
                vendor: "CPU".to_string(),
            },
        }
    }
}

impl GpuBackend for CpuFallbackBackend {
    fn name(&self) -> &str {
        "CPU Fallback"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn list_devices(&self) -> LinalgResult<Vec<GpuDeviceInfo>> {
        Ok(vec![self.device_info.clone()])
    }

    fn create_context(&self, device_id: usize) -> LinalgResult<Box<dyn GpuContext>> {
        if device_id != 0 {
            return Err(LinalgError::ComputationError(
                "CPU fallback only has one device".to_string(),
            ));
        }

        Ok(Box::new(CpuFallbackContext {
            device_info: self.device_info.clone(),
        }))
    }
}

/// CPU fallback context implementation
#[derive(Debug)]
struct CpuFallbackContext {
    device_info: GpuDeviceInfo,
}

impl GpuContext for CpuFallbackContext {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    fn synchronize(&self) -> LinalgResult<()> {
        // CPU operations are always synchronous
        Ok(())
    }

    fn available_memory(&self) -> LinalgResult<usize> {
        // Return a reasonable estimate for available system memory
        Ok(self.device_info.total_memory / 2)
    }
}

impl GpuContextAlloc for CpuFallbackContext {
    fn allocate_buffer<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug>(
        &self,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
        Ok(Box::new(CpuBuffer::new(size)))
    }
}

/// CPU buffer implementation that just wraps a Vec
#[derive(Debug)]
struct CpuBuffer<T> {
    data: Vec<T>,
}

impl<T: Clone + Send + Sync> CpuBuffer<T> {
    fn new(size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
        }
    }
}

impl<T: Clone + Send + Sync + Copy + std::fmt::Debug> GpuBuffer<T> for CpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn copy_from_host(&mut self, data: &[T]) -> LinalgResult<()> {
        self.data.clear();
        self.data.extend_from_slice(data);
        Ok(())
    }

    fn copy_to_host(&self, data: &mut [T]) -> LinalgResult<()> {
        if data.len() != self.data.len() {
            return Err(LinalgError::ShapeError("Buffer size mismatch".to_string()));
        }
        data.copy_from_slice(&self.data);
        Ok(())
    }

    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.data.as_ptr() as *mut std::ffi::c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_backend() {
        let backend = CpuFallbackBackend::new();
        assert_eq!(backend.name(), "CPU Fallback");
        assert!(backend.is_available());

        let devices = backend.list_devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "CPU Fallback");
    }

    #[test]
    fn test_cpu_fallback_context() {
        let backend = CpuFallbackBackend::new();
        let context = backend.create_context(0).unwrap();

        assert_eq!(context.device_info().name, "CPU Fallback");
        assert!(context.available_memory().unwrap() > 0);
        assert!(context.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_buffer() {
        let backend = CpuFallbackBackend::new();
        let device_info = backend.device_info.clone();

        // Create context directly to access allocate_buffer method
        let cpu_context = CpuFallbackContext { device_info };
        let mut buffer = cpu_context.allocate_buffer::<f32>(10).unwrap();
        assert_eq!(buffer.len(), 0); // Initially empty

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.copy_from_host(&data).unwrap();
        assert_eq!(buffer.len(), 5);

        let mut output = vec![0.0; 5];
        buffer.copy_to_host(&mut output).unwrap();
        assert_eq!(output, data);
    }
}
