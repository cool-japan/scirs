//! GPU-accelerated sparse FFT implementation using scirs2-core abstractions
//!
//! This module provides GPU implementations of sparse FFT algorithms through
//! the scirs2-core::gpu module. All direct GPU API calls are forbidden.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{
    SparseFFTAlgorithm, SparseFFTConfig, SparseFFTResult, SparsityEstimationMethod, WindowFunction,
};
use num_complex::Complex64;
use num_traits::NumCast;
use scirs2_core::gpu::{GpuBackend, GpuDevice};
use scirs2_core::simd_ops::PlatformCapabilities;
use std::fmt::Debug;
use std::time::Instant;

/// Placeholder for GPU buffer descriptor - to be implemented with core GPU abstractions
#[allow(dead_code)]
pub struct BufferDescriptor {
    size: usize,
    id: u64,
}

/// Placeholder for buffer location - to be implemented with core GPU abstractions  
pub enum BufferLocation {
    Device,
    Host,
}

/// Placeholder for buffer type - to be implemented with core GPU abstractions
pub enum BufferType {
    Input,
    Output,
    Work,
}

/// Placeholder for GPU stream - to be implemented with core GPU abstractions
#[allow(dead_code)]
pub struct GpuStream {
    id: u64,
}

impl GpuStream {
    pub fn new(_device_id: i32) -> FFTResult<Self> {
        Err(FFTError::NotImplementedError(
            "GPU streams need to be implemented with scirs2-core::gpu abstractions".to_string(),
        ))
    }
}

/// Placeholder memory manager - to be implemented with core GPU abstractions
pub struct GpuMemoryManager;

impl GpuMemoryManager {
    pub fn allocate(
        &self,
        _size: usize,
        _location: BufferLocation,
        _buffer_type: BufferType,
    ) -> FFTResult<BufferDescriptor> {
        Err(FFTError::NotImplementedError(
            "GPU memory management needs to be implemented with scirs2-core::gpu abstractions"
                .to_string(),
        ))
    }

    pub fn free(&self, _descriptor: BufferDescriptor) -> FFTResult<()> {
        Err(FFTError::NotImplementedError(
            "GPU memory management needs to be implemented with scirs2-core::gpu abstractions"
                .to_string(),
        ))
    }
}

/// Placeholder for global memory manager - to be implemented with core GPU abstractions
pub fn get_global_memory_manager() -> FFTResult<GpuMemoryManager> {
    Err(FFTError::NotImplementedError(
        "GPU memory management needs to be implemented with scirs2-core::gpu abstractions"
            .to_string(),
    ))
}

/// Check if GPU is available through core platform capabilities
pub fn ensure_gpu_available() -> FFTResult<bool> {
    let caps = PlatformCapabilities::detect();
    Ok(caps.cuda_available || caps.gpu_available)
}

/// GPU device information using core abstractions
pub struct GpuDeviceInfo {
    /// Device wrapped from core GPU module
    pub device: GpuDevice,
    /// Whether the device is initialized
    pub initialized: bool,
}

impl GpuDeviceInfo {
    /// Create GPU device info using core abstractions
    pub fn new(device_id: usize) -> FFTResult<Self> {
        let device = GpuDevice::new(GpuBackend::default(), device_id);
        Ok(Self {
            device,
            initialized: true,
        })
    }

    /// Check if device is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }
}

/// GPU context for FFT operations using core abstractions
#[allow(dead_code)]
pub struct GpuContext {
    /// Device ID
    device_id: i32,
    /// Device information
    device_info: GpuDeviceInfo,
    /// GPU stream
    stream: GpuStream,
    /// Whether the context is initialized
    initialized: bool,
}

impl GpuContext {
    /// Create a new CUDA context for the specified device
    pub fn new(device_id: i32) -> FFTResult<Self> {
        // In a real implementation, this would query the device and initialize CUDA

        // Create device info using core abstractions
        let device_info = GpuDeviceInfo::new(device_id as usize)?;

        // Create stream
        let stream = GpuStream::new(device_id)?;

        Ok(Self {
            device_id,
            device_info,
            stream,
            initialized: true,
        })
    }

    /// Get device information
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Get stream
    pub fn stream(&self) -> &GpuStream {
        &self.stream
    }

    /// Allocate device memory
    pub fn allocate(&self, size_bytes: usize) -> FFTResult<BufferDescriptor> {
        // In a real implementation, this would call cudaMalloc

        // Use the global memory manager to track allocations
        let manager = get_global_memory_manager()?;

        manager.allocate(size_bytes, BufferLocation::Device, BufferType::Work)
    }

    /// Free device memory
    pub fn free(&self, descriptor: BufferDescriptor) -> FFTResult<()> {
        // In a real implementation, this would call cudaFree

        // Use the global memory manager to track allocations
        let manager = get_global_memory_manager()?;

        manager.free(descriptor)
    }

    /// Copy data from host to device
    pub fn copy_host_to_device<T>(
        &self,
        host_data: &[T],
        device_buffer: &BufferDescriptor,
    ) -> FFTResult<()> {
        // In a real implementation, this would call cudaMemcpy

        // Check if sizes match
        let host_size_bytes = std::mem::size_of_val(host_data);
        let device_size_bytes = device_buffer.size;

        if host_size_bytes > device_size_bytes {
            return Err(FFTError::DimensionError(format!(
                "Host buffer size ({} bytes) exceeds device buffer size ({} bytes)",
                host_size_bytes, device_size_bytes
            )));
        }

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_device_to_host<T>(
        &self,
        device_buffer: &BufferDescriptor,
        host_data: &mut [T],
    ) -> FFTResult<()> {
        // In a real implementation, this would call cudaMemcpy

        // Check if sizes match
        let host_size_bytes = std::mem::size_of_val(host_data);
        let device_size_bytes = device_buffer.size;

        if device_size_bytes > host_size_bytes {
            return Err(FFTError::DimensionError(format!(
                "Device buffer size ({} bytes) exceeds host buffer size ({} bytes)",
                device_size_bytes, host_size_bytes
            )));
        }

        Ok(())
    }
}

/// CUDA-accelerated sparse FFT implementation
pub struct GpuSparseFFT {
    /// CUDA context
    context: GpuContext,
    /// Sparse FFT configuration
    config: SparseFFTConfig,
    /// Buffer for input signal on device
    input_buffer: Option<BufferDescriptor>,
    /// Buffer for output values on device
    output_values_buffer: Option<BufferDescriptor>,
    /// Buffer for output indices on device
    output_indices_buffer: Option<BufferDescriptor>,
}

impl GpuSparseFFT {
    /// Create a new CUDA-accelerated sparse FFT processor
    pub fn new(device_id: i32, config: SparseFFTConfig) -> FFTResult<Self> {
        // GPU device initialization handled by core GPU abstractions
        // TODO: Use scirs2-core::gpu device initialization

        // Initialize CUDA context
        let context = GpuContext::new(device_id)?;

        Ok(Self {
            context,
            config,
            input_buffer: None,
            output_values_buffer: None,
            output_indices_buffer: None,
        })
    }

    /// Initialize buffers for the given signal size
    fn initialize_buffers(&mut self, signal_size: usize) -> FFTResult<()> {
        // Free existing buffers if any
        self.free_buffers()?;

        // Get memory manager
        let memory_manager = get_global_memory_manager()?;

        // Allocate input buffer
        let input_buffer = memory_manager.allocate(
            signal_size * std::mem::size_of::<Complex64>(),
            BufferLocation::Device,
            BufferType::Input,
        )?;
        self.input_buffer = Some(input_buffer);

        // Allocate output buffers (assuming worst case: all components are significant)
        let max_components = self.config.sparsity.min(signal_size);

        let output_values_buffer = memory_manager.allocate(
            max_components * std::mem::size_of::<Complex64>(),
            BufferLocation::Device,
            BufferType::Output,
        )?;
        self.output_values_buffer = Some(output_values_buffer);

        let output_indices_buffer = memory_manager.allocate(
            max_components * std::mem::size_of::<usize>(),
            BufferLocation::Device,
            BufferType::Output,
        )?;
        self.output_indices_buffer = Some(output_indices_buffer);

        Ok(())
    }

    /// Free all buffers
    fn free_buffers(&mut self) -> FFTResult<()> {
        if let Ok(memory_manager) = get_global_memory_manager() {
            if let Some(buffer) = self.input_buffer.take() {
                memory_manager.free(buffer)?;
            }

            if let Some(buffer) = self.output_values_buffer.take() {
                memory_manager.free(buffer)?;
            }

            if let Some(buffer) = self.output_indices_buffer.take() {
                memory_manager.free(buffer)?;
            }
        }

        Ok(())
    }

    /// Perform sparse FFT on a signal
    pub fn sparse_fft<T>(&mut self, signal: &[T]) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        let start = Instant::now();

        // Initialize buffers
        self.initialize_buffers(signal.len())?;

        // Convert input to complex
        let signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        // Copy the input signal to the device
        if let Some(input_buffer) = &self.input_buffer {
            self.context
                .copy_host_to_device(&signal_complex, input_buffer)?;
        } else {
            return Err(FFTError::MemoryError(
                "Input buffer not initialized".to_string(),
            ));
        }

        // Use the appropriate kernel based on the algorithm
        let result = match self.config.algorithm {
            SparseFFTAlgorithm::Sublinear => crate::execute_cuda_sublinear_sparse_fft(
                &signal_complex,
                self.config.sparsity,
                self.config.algorithm,
            )?,
            SparseFFTAlgorithm::CompressedSensing => {
                crate::execute_cuda_compressed_sensing_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                )?
            }
            SparseFFTAlgorithm::Iterative => {
                crate::execute_cuda_iterative_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    100, // Default number of iterations
                )?
            }
            SparseFFTAlgorithm::FrequencyPruning => {
                crate::execute_cuda_frequency_pruning_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    0.01, // Default threshold
                )?
            }
            SparseFFTAlgorithm::SpectralFlatness => {
                crate::execute_cuda_spectral_flatness_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    self.config.flatness_threshold,
                )?
            }
            // For other algorithms, fall back to CPU implementation for now
            _ => {
                let mut cpu_processor = crate::sparse_fft::SparseFFT::new(self.config.clone());
                let mut cpu_result = cpu_processor.sparse_fft(&signal_complex)?;

                // Update the computation time and algorithm
                cpu_result.computation_time = start.elapsed();
                cpu_result.algorithm = self.config.algorithm;

                cpu_result
            }
        };

        Ok(result)
    }
}

impl Drop for GpuSparseFFT {
    fn drop(&mut self) {
        // Free all resources
        let _ = self.free_buffers();
    }
}

/// Perform CUDA-accelerated sparse FFT
///
/// This is a convenience function that creates a CUDA sparse FFT processor
/// and performs the computation.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `k` - Expected sparsity (number of significant frequency components)
/// * `device_id` - CUDA device ID (-1 for auto-select)
/// * `algorithm` - Sparse FFT algorithm variant
/// * `window_function` - Window function to apply before FFT
///
/// # Returns
///
/// * Sparse FFT result containing frequency components, indices, and timing information
#[allow(clippy::too_many_arguments)]
pub fn cuda_sparse_fft<T>(
    signal: &[T],
    k: usize,
    device_id: i32,
    algorithm: Option<SparseFFTAlgorithm>,
    window_function: Option<WindowFunction>,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Check if GPU is available
    if !ensure_gpu_available()? {
        return Err(FFTError::ComputationError(
            "GPU is not available. Either GPU features are not enabled or GPU hardware/drivers are not available.".to_string()
        ));
    }

    // Create a base configuration
    let config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity: k,
        algorithm: algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear),
        window_function: window_function.unwrap_or(WindowFunction::None),
        ..SparseFFTConfig::default()
    };

    // GPU memory manager initialization handled by core GPU abstractions
    // TODO: Use scirs2-core::gpu memory management initialization

    // Create processor and perform computation
    let mut processor = GpuSparseFFT::new(device_id, config)?;
    processor.sparse_fft(signal)
}

/// Perform batch CUDA-accelerated sparse FFT
///
/// Process multiple signals in batch mode for better GPU utilization.
///
/// # Arguments
///
/// * `signals` - List of input signals
/// * `k` - Expected sparsity
/// * `device_id` - CUDA device ID (-1 for auto-select)
/// * `algorithm` - Sparse FFT algorithm variant
/// * `window_function` - Window function to apply before FFT
///
/// # Returns
///
/// * List of sparse FFT results for each input signal
#[allow(clippy::too_many_arguments)]
pub fn cuda_batch_sparse_fft<T>(
    signals: &[Vec<T>],
    k: usize,
    device_id: i32,
    algorithm: Option<SparseFFTAlgorithm>,
    window_function: Option<WindowFunction>,
) -> FFTResult<Vec<SparseFFTResult>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Create a base configuration
    let config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity: k,
        algorithm: algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear),
        window_function: window_function.unwrap_or(WindowFunction::None),
        ..SparseFFTConfig::default()
    };

    // Create processor
    let mut processor = GpuSparseFFT::new(device_id, config)?;

    // Process each signal
    let mut results = Vec::with_capacity(signals.len());
    for signal in signals {
        results.push(processor.sparse_fft(signal)?);
    }

    Ok(results)
}

/// Initialize GPU subsystem and get available GPU devices
pub fn get_cuda_devices() -> FFTResult<Vec<GpuDeviceInfo>> {
    // In a real implementation, this would query all available GPU devices through scirs2-core

    // First check if GPU is available
    if !ensure_gpu_available().unwrap_or(false) {
        return Ok(Vec::new());
    }

    // For now, return dummy data until actual GPU implementation is complete
    let devices = vec![GpuDeviceInfo::new(0)?];

    Ok(devices)
}

// Note: is_cuda_available() is now provided by sparse_fft_gpu_memory module

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_fft_gpu_memory::AllocationStrategy;
    use std::f64::consts::PI;

    // Helper function to create a sparse signal
    fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
        let mut signal = vec![0.0; n];

        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            for &(freq, amp) in frequencies {
                signal[i] += amp * (freq as f64 * t).sin();
            }
        }

        signal
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_cuda_initialization() {
        // Initialize global memory manager
        let _ = crate::sparse_fft_gpu_memory::init_global_memory_manager(
            crate::sparse_fft_gpu::GPUBackend::CUDA,
            0,
            AllocationStrategy::CacheBySize,
            1024 * 1024 * 1024, // 1GB limit
        );

        // Get CUDA devices
        let devices = get_cuda_devices().unwrap();
        assert!(!devices.is_empty());

        // Create a CUDA context
        let context = GpuContext::new(0).unwrap();
        assert_eq!(context.device_id, 0);
        assert!(context.initialized);
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_cuda_sparse_fft() {
        // Create a signal with 3 frequency components
        let n = 256;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signal = create_sparse_signal(n, &frequencies);

        // Test with default parameters
        let result = cuda_sparse_fft(
            &signal,
            6,
            0,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();

        // Should find the frequency components
        assert!(!result.values.is_empty());
        assert_eq!(result.algorithm, SparseFFTAlgorithm::Sublinear);
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_cuda_batch_processing() {
        // Create multiple signals
        let n = 128;
        let signals = vec![
            create_sparse_signal(n, &[(3, 1.0), (7, 0.5)]),
            create_sparse_signal(n, &[(5, 1.0), (10, 0.7)]),
            create_sparse_signal(n, &[(2, 0.8), (12, 0.6)]),
        ];

        // Test batch processing
        let results =
            cuda_batch_sparse_fft(&signals, 4, 0, Some(SparseFFTAlgorithm::Sublinear), None)
                .unwrap();

        // Should return the same number of results as input signals
        assert_eq!(results.len(), signals.len());

        // Each result should have frequency components
        for result in results {
            assert!(!result.values.is_empty());
        }
    }
}

// Duplicate function removed
// See is_cuda_available() defined above
