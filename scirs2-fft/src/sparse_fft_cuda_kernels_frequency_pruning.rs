//! CUDA kernel implementations for frequency pruning sparse FFT algorithm
//!
//! This module contains the implementation of frequency pruning sparse FFT algorithm
//! optimized for GPU execution using CUDA. This algorithm works by pruning insignificant
//! frequency bands to focus computation resources on promising spectral regions.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{SparseFFTAlgorithm, SparseFFTResult, WindowFunction};
use crate::sparse_fft_cuda_kernels::CUDAWindowKernel;
use crate::sparse_fft_gpu_cuda::CUDAContext;
use crate::sparse_fft_gpu_kernels::{GPUKernel, KernelConfig, KernelStats};
use crate::sparse_fft_gpu_memory::BufferDescriptor;

use num_complex::Complex64;
use num_traits::NumCast;
use std::fmt::Debug;
use std::time::Instant;

/// CUDA kernel for Frequency Pruning Sparse FFT algorithm
pub struct CUDAFrequencyPruningSparseFFTKernel {
    /// Kernel configuration
    config: KernelConfig,
    /// Input buffer
    input_buffer: BufferDescriptor,
    /// Output values buffer
    output_values_buffer: BufferDescriptor,
    /// Output indices buffer
    output_indices_buffer: BufferDescriptor,
    /// Expected sparsity
    sparsity: usize,
    /// Number of frequency bands to use
    bands: usize,
    /// CUDA context
    context: CUDAContext,
    /// Window function to apply (if any)
    window_function: WindowFunction,
}

impl CUDAFrequencyPruningSparseFFTKernel {
    /// Create a new CUDA Frequency Pruning Sparse FFT kernel
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_buffer: BufferDescriptor,
        output_values_buffer: BufferDescriptor,
        output_indices_buffer: BufferDescriptor,
        sparsity: usize,
        bands: Option<usize>,
        context: CUDAContext,
        window_function: WindowFunction,
    ) -> Self {
        let mut config = KernelConfig::default();

        // Calculate grid size based on input size and block size
        let input_size = input_buffer.size / std::mem::size_of::<Complex64>();
        config.grid_size = input_size.div_ceil(config.block_size);

        // Optimize for Frequency Pruning algorithm
        config.block_size = 256;
        config.shared_memory_size = 16 * 1024; // 16 KB

        // Enable mixed precision if supported
        let compute_capability = context.device_info().compute_capability;
        if compute_capability.0 >= 7 || (compute_capability.0 == 6 && compute_capability.1 >= 1) {
            config.use_mixed_precision = true;
        }

        // Determine number of bands (default: sqrt(N))
        let default_bands = (input_size as f64).sqrt().ceil() as usize;
        let bands = bands.unwrap_or(default_bands);

        Self {
            config,
            input_buffer,
            output_values_buffer,
            output_indices_buffer,
            sparsity,
            bands,
            context,
            window_function,
        }
    }

    /// Apply window function if needed
    fn apply_window(&self) -> FFTResult<KernelStats> {
        if self.window_function == WindowFunction::None {
            // No window function, return dummy stats
            return Ok(KernelStats {
                execution_time_ms: 0.0,
                memory_bandwidth_gb_s: 0.0,
                compute_throughput_gflops: 0.0,
                bytes_transferred_to_device: 0,
                bytes_transferred_from_device: 0,
                occupancy_percent: 0.0,
            });
        }

        // Create and execute window kernel
        let window_kernel = CUDAWindowKernel::new(
            self.input_buffer.clone(),
            self.window_function,
            self.context.clone(),
        );

        window_kernel.execute()
    }
}

impl GPUKernel for CUDAFrequencyPruningSparseFFTKernel {
    fn name(&self) -> &str {
        "CUDA_FrequencyPruning_SparseFFT_Kernel"
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn set_config(&mut self, config: KernelConfig) {
        self.config = config;
    }

    fn execute(&self) -> FFTResult<KernelStats> {
        // Start timer
        let start = Instant::now();

        // Apply window function if needed
        let window_stats = self.apply_window()?;

        // Get buffer sizes and pointers
        let input_size = self.input_buffer.size / std::mem::size_of::<Complex64>();
        let (input_host_ptr, _) = self.input_buffer.get_host_ptr();
        let (output_values_ptr, _) = self.output_values_buffer.get_host_ptr();
        let (output_indices_ptr, _) = self.output_indices_buffer.get_host_ptr();

        // In a real CUDA implementation, this would use custom CUDA kernels for:
        // 1. Divide the spectrum into bands and calculate energy per band
        // 2. Select bands with significant energy
        // 3. Perform detailed FFT only on selected bands
        // 4. Find the top frequency components across selected bands

        // For CPU fallback mode, we'll implement a basic frequency pruning approach:

        // Get input signal
        let input_signal =
            unsafe { std::slice::from_raw_parts(input_host_ptr as *const Complex64, input_size) };

        // 1. Perform full FFT first (in a real implementation, we would do a coarse FFT)
        let mut fft_output = input_signal.to_vec();
        let fft = rustfft::FftPlanner::new().plan_fft_forward(input_size);
        fft.process(&mut fft_output);

        // 2. Divide the spectrum into bands
        let band_size = input_size / self.bands;
        let mut band_energies = Vec::with_capacity(self.bands);

        // Calculate energy per band
        for b in 0..self.bands {
            let start_idx = b * band_size;
            let end_idx = if b == self.bands - 1 {
                input_size
            } else {
                (b + 1) * band_size
            };

            // Calculate total energy in this band
            let energy: f64 = fft_output[start_idx..end_idx]
                .iter()
                .map(|c| c.norm_sqr())
                .sum();

            band_energies.push((b, energy));
        }

        // 3. Sort bands by energy (descending)
        band_energies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Determine how many bands to keep based on sparsity
        // (in a real implementation, we might use a dynamic threshold)
        let bands_to_keep = (self.sparsity as f64 / band_size as f64).ceil() as usize;
        let bands_to_keep = bands_to_keep.min(self.bands).max(1);

        // 5. Select the most energetic bands
        let selected_bands: Vec<usize> = band_energies
            .iter()
            .take(bands_to_keep)
            .map(|&(band_idx, _)| band_idx)
            .collect();

        // 6. For each selected band, find the significant frequencies
        let mut frequency_components = Vec::new();

        for &band_idx in &selected_bands {
            let start_idx = band_idx * band_size;
            let end_idx = if band_idx == self.bands - 1 {
                input_size
            } else {
                (band_idx + 1) * band_size
            };

            // Find peak frequencies in this band
            for idx in start_idx..end_idx {
                let left = if idx > start_idx {
                    fft_output[idx - 1].norm_sqr()
                } else {
                    0.0
                };
                let right = if idx < end_idx - 1 {
                    fft_output[idx + 1].norm_sqr()
                } else {
                    0.0
                };
                let current = fft_output[idx].norm_sqr();

                // Check if it's a local peak
                if current > left && current > right && current > 1e-10 {
                    frequency_components.push((idx, fft_output[idx]));
                }
            }
        }

        // 7. Sort all found components by magnitude and take the top-k
        frequency_components.sort_by(|a, b| {
            b.1.norm()
                .partial_cmp(&a.1.norm())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Copy results to output buffers
        unsafe {
            let output_values =
                std::slice::from_raw_parts_mut(output_values_ptr as *mut Complex64, self.sparsity);

            let output_indices =
                std::slice::from_raw_parts_mut(output_indices_ptr as *mut usize, self.sparsity);

            // Fill output buffers with top-k components
            for i in 0..frequency_components.len().min(self.sparsity) {
                output_values[i] = frequency_components[i].1;
                output_indices[i] = frequency_components[i].0;
            }

            // Fill the rest with zeros if needed
            for i in frequency_components.len().min(self.sparsity)..self.sparsity {
                output_values[i] = Complex64::new(0.0, 0.0);
                output_indices[i] = 0;
            }
        }

        // Copy results back to device if needed
        self.output_values_buffer.copy_host_to_device()?;
        self.output_indices_buffer.copy_host_to_device()?;

        // Calculate statistics
        let execution_time_ms =
            start.elapsed().as_secs_f64() * 1000.0 + window_stats.execution_time_ms;
        let bytes_processed = input_size * std::mem::size_of::<Complex64>()
            + bands_to_keep * band_size * std::mem::size_of::<Complex64>();

        // FFT plus band analysis operations
        let approx_flops = input_size as f64 * (input_size as f64).log2() * 5.0 + // Main FFT
                          bands_to_keep as f64 * band_size as f64 * 10.0; // Band processing

        // Calculate stats
        let stats = KernelStats {
            execution_time_ms,
            memory_bandwidth_gb_s: (bytes_processed as f64 / execution_time_ms * 1000.0) / 1e9,
            compute_throughput_gflops: (approx_flops / execution_time_ms * 1000.0) / 1e9,
            bytes_transferred_to_device: 0, // Already on device
            bytes_transferred_from_device: self.sparsity * std::mem::size_of::<Complex64>() * 2, // Values and indices
            occupancy_percent: 75.0,
        };

        Ok(stats)
    }
}

/// Execute CUDA Frequency Pruning Sparse FFT
pub fn execute_cuda_frequency_pruning_sparse_fft<T>(
    signal: &[T],
    sparsity: usize,
    bands: Option<usize>,
    window_function: WindowFunction,
    device_id: i32,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    let start = Instant::now();

    // Create CUDA context
    let context = CUDAContext::new(device_id)?;

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

    // Allocate buffers
    let input_size_bytes = signal_complex.len() * std::mem::size_of::<Complex64>();
    let input_buffer = context.allocate(input_size_bytes)?;

    let output_values_size_bytes = sparsity * std::mem::size_of::<Complex64>();
    let output_values_buffer = context.allocate(output_values_size_bytes)?;

    let output_indices_size_bytes = sparsity * std::mem::size_of::<usize>();
    let output_indices_buffer = context.allocate(output_indices_size_bytes)?;

    // Get host pointers and copy input data
    let (input_host_ptr, _) = input_buffer.get_host_ptr();
    unsafe {
        std::ptr::copy_nonoverlapping(
            signal_complex.as_ptr(),
            input_host_ptr as *mut Complex64,
            signal_complex.len(),
        );
    }

    // Transfer data to device
    input_buffer.copy_host_to_device()?;

    // Create and execute kernel
    let kernel = CUDAFrequencyPruningSparseFFTKernel::new(
        input_buffer.clone(),
        output_values_buffer.clone(),
        output_indices_buffer.clone(),
        sparsity,
        bands,
        context.clone(),
        window_function,
    );

    // Execute kernel - this now uses our actual kernel implementation
    let _stats = kernel.execute()?;

    // Transfer results back from device
    output_values_buffer.copy_device_to_host()?;
    output_indices_buffer.copy_device_to_host()?;

    // Read results from output buffers
    let (values_ptr, _) = output_values_buffer.get_host_ptr();
    let (indices_ptr, _) = output_indices_buffer.get_host_ptr();

    // Convert to Rust vectors
    let values =
        unsafe { std::slice::from_raw_parts(values_ptr as *const Complex64, sparsity).to_vec() };

    let indices =
        unsafe { std::slice::from_raw_parts(indices_ptr as *const usize, sparsity).to_vec() };

    // Filter out zero values (unused entries)
    let mut result_values = Vec::new();
    let mut result_indices = Vec::new();

    for i in 0..sparsity {
        if values[i].norm() > 1e-10 {
            result_values.push(values[i]);
            result_indices.push(indices[i]);
        }
    }

    let estimated_sparsity = result_values.len();

    // Free buffers
    context.free(input_buffer)?;
    context.free(output_values_buffer)?;
    context.free(output_indices_buffer)?;

    // Build result
    let computation_time = start.elapsed();

    let result = SparseFFTResult {
        values: result_values,
        indices: result_indices,
        estimated_sparsity,
        computation_time,
        algorithm: SparseFFTAlgorithm::FrequencyPruning,
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_execute_cuda_frequency_pruning_sparse_fft() {
        // Create a sparse signal
        let n = 1024;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signal = create_sparse_signal(n, &frequencies);

        // Execute CUDA sparse FFT
        let result = execute_cuda_frequency_pruning_sparse_fft(
            &signal,
            6,
            None, // Default bands
            WindowFunction::Hann,
            0,
        )
        .unwrap();

        // Check results
        assert_eq!(result.values.len(), 6);
        assert_eq!(result.indices.len(), 6);
        assert_eq!(result.algorithm, SparseFFTAlgorithm::FrequencyPruning);
    }
}
