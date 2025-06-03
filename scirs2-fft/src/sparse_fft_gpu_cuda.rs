//! CUDA-accelerated sparse FFT implementation
//!
//! This module provides CUDA-specific implementations of sparse FFT algorithms.
//! It requires the CUDA toolkit to be installed and the `cuda` feature to be enabled.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{
    SparseFFTAlgorithm, SparseFFTConfig, SparseFFTResult, SparsityEstimationMethod, WindowFunction,
};
// Import removed - unused
use crate::sparse_fft_gpu_memory::{
    get_global_memory_manager, init_global_memory_manager, BufferDescriptor, BufferLocation,
    BufferType,
};
use num_complex::Complex64;
use num_traits::NumCast;
use std::fmt::Debug;
use std::time::Instant;

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CUDADeviceInfo {
    /// Device ID
    pub device_id: i32,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Compute capability
    pub compute_capability: (i32, i32),
    /// Number of multiprocessors
    pub multiprocessor_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Maximum shared memory per block in bytes
    pub max_shared_memory_per_block: usize,
}

/// CUDA stream for asynchronous execution
#[derive(Debug, Clone)]
pub struct CUDAStream {
    /// Stream handle (would be CUstream/cudaStream_t in real implementation)
    #[allow(dead_code)]
    handle: usize,
    /// Device ID this stream is associated with
    #[allow(dead_code)]
    device_id: i32,
}

impl CUDAStream {
    /// Create a new CUDA stream
    pub fn new(device_id: i32) -> FFTResult<Self> {
        // In a real implementation, this would call cudaStreamCreate
        let handle = 1; // Dummy handle for now

        Ok(Self { handle, device_id })
    }

    /// Synchronize the stream
    pub fn synchronize(&self) -> FFTResult<()> {
        // In a real implementation, this would call cudaStreamSynchronize
        Ok(())
    }
}

impl Drop for CUDAStream {
    fn drop(&mut self) {
        // In a real implementation, this would call cudaStreamDestroy
    }
}

/// CUDA context for GPU operations
#[derive(Debug, Clone)]
pub struct CUDAContext {
    /// Device ID
    #[allow(dead_code)]
    device_id: i32,
    /// Device information
    device_info: CUDADeviceInfo,
    /// Primary stream for operations
    stream: CUDAStream,
    /// Whether the context is initialized
    #[allow(dead_code)]
    initialized: bool,
}

impl CUDAContext {
    /// Create a new CUDA context for the specified device
    pub fn new(device_id: i32) -> FFTResult<Self> {
        // In a real implementation, this would query the device and initialize CUDA

        // Create dummy device info or query actual device if CUDA is available
        let device_info = if cfg!(feature = "cuda") {
            // Query CUDA device info (in a real implementation)
            // For now, create dummy info
            CUDADeviceInfo {
                device_id,
                name: format!("CUDA Device {}", device_id),
                total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
                compute_capability: (8, 0),           // SM 8.0
                multiprocessor_count: 30,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 48 * 1024, // 48 KB
            }
        } else {
            // Create dummy device info
            CUDADeviceInfo {
                device_id,
                name: format!("CUDA Device {}", device_id),
                total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
                compute_capability: (8, 0),           // SM 8.0
                multiprocessor_count: 30,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 48 * 1024, // 48 KB
            }
        };

        // Create stream
        let stream = CUDAStream::new(device_id)?;

        Ok(Self {
            device_id,
            device_info,
            stream,
            initialized: true,
        })
    }

    /// Get device information
    pub fn device_info(&self) -> &CUDADeviceInfo {
        &self.device_info
    }

    /// Get stream
    pub fn stream(&self) -> &CUDAStream {
        &self.stream
    }

    /// Allocate device memory
    pub fn allocate(&self, size_bytes: usize) -> FFTResult<BufferDescriptor> {
        // In a real implementation, this would call cudaMalloc

        // Use the global memory manager to track allocations
        let manager = get_global_memory_manager()?;
        let mut manager = manager.lock().unwrap();

        manager.allocate_buffer(
            size_bytes,
            1, // Element size = 1 byte
            BufferLocation::Device,
            BufferType::Work,
        )
    }

    /// Free device memory
    pub fn free(&self, descriptor: BufferDescriptor) -> FFTResult<()> {
        // In a real implementation, this would call cudaFree

        // Use the global memory manager to track allocations
        let manager = get_global_memory_manager()?;
        let mut manager = manager.lock().unwrap();

        manager.release_buffer(descriptor)
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
        let device_size_bytes = device_buffer.size * device_buffer.element_size;

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
        let device_size_bytes = device_buffer.size * device_buffer.element_size;

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
pub struct CUDASparseFFT {
    /// CUDA context
    context: CUDAContext,
    /// Sparse FFT configuration
    config: SparseFFTConfig,
    /// Buffer for input signal on device
    input_buffer: Option<BufferDescriptor>,
    /// Buffer for output values on device
    output_values_buffer: Option<BufferDescriptor>,
    /// Buffer for output indices on device
    output_indices_buffer: Option<BufferDescriptor>,
}

impl CUDASparseFFT {
    /// Create a new CUDA-accelerated sparse FFT processor
    pub fn new(device_id: i32, config: SparseFFTConfig) -> FFTResult<Self> {
        // Initialize CUDA context
        let context = CUDAContext::new(device_id)?;

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

        // Allocate input buffer
        let input_size_bytes = signal_size * std::mem::size_of::<Complex64>();
        self.input_buffer = Some(self.context.allocate(input_size_bytes)?);

        // Allocate output buffers (assuming worst case: all components are significant)
        let max_components = self.config.sparsity.min(signal_size);
        let output_values_size_bytes = max_components * std::mem::size_of::<Complex64>();
        self.output_values_buffer = Some(self.context.allocate(output_values_size_bytes)?);

        let output_indices_size_bytes = max_components * std::mem::size_of::<usize>();
        self.output_indices_buffer = Some(self.context.allocate(output_indices_size_bytes)?);

        Ok(())
    }

    /// Free all buffers
    fn free_buffers(&mut self) -> FFTResult<()> {
        if let Some(buffer) = self.input_buffer.take() {
            self.context.free(buffer)?;
        }

        if let Some(buffer) = self.output_values_buffer.take() {
            self.context.free(buffer)?;
        }

        if let Some(buffer) = self.output_indices_buffer.take() {
            self.context.free(buffer)?;
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
                self.config.window_function,
                self.context.device_info().device_id,
            )?,
            SparseFFTAlgorithm::CompressedSensing => {
                crate::execute_cuda_compressed_sensing_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    self.config.window_function,
                    self.context.device_info().device_id,
                )?
            }
            SparseFFTAlgorithm::Iterative => {
                crate::execute_cuda_iterative_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    None, // Use default number of iterations
                    self.config.window_function,
                    self.context.device_info().device_id,
                )?
            }
            SparseFFTAlgorithm::FrequencyPruning => {
                crate::execute_cuda_frequency_pruning_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    None, // Use default number of bands
                    self.config.window_function,
                    self.context.device_info().device_id,
                )?
            }
            SparseFFTAlgorithm::SpectralFlatness => {
                crate::execute_cuda_spectral_flatness_sparse_fft(
                    &signal_complex,
                    self.config.sparsity,
                    Some(self.config.flatness_threshold),
                    Some(self.config.window_size),
                    self.config.window_function,
                    self.context.device_info().device_id,
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

impl Drop for CUDASparseFFT {
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
    // Create a base configuration
    let config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity: k,
        algorithm: algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear),
        window_function: window_function.unwrap_or(WindowFunction::None),
        ..SparseFFTConfig::default()
    };

    // Initialize memory manager if not already initialized
    init_global_memory_manager(
        crate::sparse_fft_gpu::GPUBackend::CUDA,
        device_id,
        crate::sparse_fft_gpu_memory::AllocationStrategy::CacheBySize,
        1024 * 1024 * 1024, // 1 GB limit
    )?;

    // Create processor and perform computation
    let mut processor = CUDASparseFFT::new(device_id, config)?;
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
    let mut processor = CUDASparseFFT::new(device_id, config)?;

    // Process each signal
    let mut results = Vec::with_capacity(signals.len());
    for signal in signals {
        results.push(processor.sparse_fft(signal)?);
    }

    Ok(results)
}

/// Initialize CUDA subsystem and get available CUDA devices
pub fn get_cuda_devices() -> FFTResult<Vec<CUDADeviceInfo>> {
    // In a real implementation, this would query all available CUDA devices

    // First check if CUDA is available
    if !is_cuda_available() {
        return Ok(Vec::new());
    }

    // For now, return dummy data until actual CUDA implementation is complete
    let devices = vec![CUDADeviceInfo {
        device_id: 0,
        name: "NVIDIA GeForce RTX 3080".to_string(),
        total_memory: 10 * 1024 * 1024 * 1024, // 10 GB
        compute_capability: (8, 6),            // SM 8.6
        multiprocessor_count: 68,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 48 * 1024, // 48 KB
    }];

    Ok(devices)
}

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    // In a real implementation, this would check if CUDA is available
    // For now, check if the CUDA feature is enabled
    cfg!(feature = "cuda")
}

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
        let context = CUDAContext::new(0).unwrap();
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
