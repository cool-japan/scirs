//! Multi-GPU Sparse FFT Implementation
//!
//! This module provides advanced multi-GPU support for sparse FFT operations,
//! allowing parallel processing across multiple GPU devices for maximum performance.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{SparseFFTAlgorithm, SparseFFTConfig, SparseFFTResult, WindowFunction};
use crate::sparse_fft_gpu::{GPUBackend, GPUSparseFFTConfig};
use crate::sparse_fft_gpu_memory::{
    init_cuda_device, init_hip_device, init_sycl_device, is_cuda_available, is_hip_available,
    is_sycl_available,
};
use num_complex::Complex64;
use num_traits::NumCast;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Information about an available GPU device
#[derive(Debug, Clone)]
pub struct GPUDeviceInfo {
    /// Device ID
    pub device_id: i32,
    /// Backend type
    pub backend: GPUBackend,
    /// Device name/description
    pub device_name: String,
    /// Available memory in bytes
    pub memory_total: usize,
    /// Free memory in bytes
    pub memory_free: usize,
    /// Compute capability or equivalent
    pub compute_capability: f32,
    /// Number of compute units/SMs
    pub compute_units: usize,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Is this device currently available
    pub is_available: bool,
}

impl Default for GPUDeviceInfo {
    fn default() -> Self {
        Self {
            device_id: -1,
            backend: GPUBackend::CPUFallback,
            device_name: "Unknown Device".to_string(),
            memory_total: 0,
            memory_free: 0,
            compute_capability: 0.0,
            compute_units: 0,
            max_threads_per_block: 0,
            is_available: false,
        }
    }
}

/// Multi-GPU workload distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadDistribution {
    /// Equal distribution across all devices
    Equal,
    /// Distribution based on device memory capacity
    MemoryBased,
    /// Distribution based on device compute capability
    ComputeBased,
    /// Manual distribution with specified ratios
    Manual,
    /// Adaptive distribution based on runtime performance
    Adaptive,
}

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGPUConfig {
    /// Base sparse FFT configuration
    pub base_config: SparseFFTConfig,
    /// Workload distribution strategy
    pub distribution: WorkloadDistribution,
    /// Manual distribution ratios (if using Manual distribution)
    pub manual_ratios: Vec<f32>,
    /// Maximum number of devices to use (0 = use all available)
    pub max_devices: usize,
    /// Minimum signal size to enable multi-GPU processing
    pub min_signal_size: usize,
    /// Overlap between chunks for boundary handling
    pub chunk_overlap: usize,
    /// Enable load balancing between devices
    pub enable_load_balancing: bool,
    /// Timeout for device operations in milliseconds
    pub device_timeout_ms: u64,
}

impl Default for MultiGPUConfig {
    fn default() -> Self {
        Self {
            base_config: SparseFFTConfig::default(),
            distribution: WorkloadDistribution::ComputeBased,
            manual_ratios: Vec::new(),
            max_devices: 0,        // Use all available
            min_signal_size: 4096, // Only use multi-GPU for larger signals
            chunk_overlap: 0,
            enable_load_balancing: true,
            device_timeout_ms: 5000,
        }
    }
}

/// Multi-GPU sparse FFT processor
pub struct MultiGPUSparseFFT {
    /// Configuration
    config: MultiGPUConfig,
    /// Available devices
    devices: Vec<GPUDeviceInfo>,
    /// Device selection for current operation
    selected_devices: Vec<usize>,
    /// Performance history for adaptive load balancing
    performance_history: Arc<Mutex<HashMap<i32, Vec<f64>>>>,
    /// Is multi-GPU initialized
    initialized: bool,
}

impl MultiGPUSparseFFT {
    /// Create a new multi-GPU sparse FFT processor
    pub fn new(config: MultiGPUConfig) -> Self {
        Self {
            config,
            devices: Vec::new(),
            selected_devices: Vec::new(),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            initialized: false,
        }
    }

    /// Initialize multi-GPU system and enumerate devices
    pub fn initialize(&mut self) -> FFTResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Enumerate all available GPU devices
        self.enumerate_devices()?;

        // Select devices based on configuration
        self.select_devices()?;

        self.initialized = true;
        Ok(())
    }

    /// Enumerate all available GPU devices
    fn enumerate_devices(&mut self) -> FFTResult<()> {
        self.devices.clear();

        // Enumerate CUDA devices
        if is_cuda_available() {
            self.enumerate_cuda_devices()?;
        }

        // Enumerate HIP devices
        if is_hip_available() {
            self.enumerate_hip_devices()?;
        }

        // Enumerate SYCL devices
        if is_sycl_available() {
            self.enumerate_sycl_devices()?;
        }

        // Add CPU fallback as last resort
        self.devices.push(GPUDeviceInfo {
            device_id: -1,
            backend: GPUBackend::CPUFallback,
            device_name: "CPU Fallback".to_string(),
            memory_total: 16 * 1024 * 1024 * 1024, // Assume 16GB RAM
            memory_free: 8 * 1024 * 1024 * 1024,   // Assume half available
            compute_capability: 1.0,
            compute_units: num_cpus::get(),
            max_threads_per_block: 1,
            is_available: true,
        });

        Ok(())
    }

    /// Enumerate CUDA devices
    fn enumerate_cuda_devices(&mut self) -> FFTResult<()> {
        // Initialize CUDA if available
        if init_cuda_device()? {
            // In a real implementation, this would query actual CUDA devices
            // For now, simulate one CUDA device
            self.devices.push(GPUDeviceInfo {
                device_id: 0,
                backend: GPUBackend::CUDA,
                device_name: "NVIDIA GPU (simulated)".to_string(),
                memory_total: 8 * 1024 * 1024 * 1024, // 8GB
                memory_free: 6 * 1024 * 1024 * 1024,  // 6GB free
                compute_capability: 8.6,
                compute_units: 68,
                max_threads_per_block: 1024,
                is_available: true,
            });
        }

        Ok(())
    }

    /// Enumerate HIP devices
    fn enumerate_hip_devices(&mut self) -> FFTResult<()> {
        // Initialize HIP if available
        if init_hip_device()? {
            // In a real implementation, this would query actual HIP devices
            // For now, simulate one HIP device
            self.devices.push(GPUDeviceInfo {
                device_id: 0,
                backend: GPUBackend::HIP,
                device_name: "AMD GPU (simulated)".to_string(),
                memory_total: 16 * 1024 * 1024 * 1024, // 16GB
                memory_free: 12 * 1024 * 1024 * 1024,  // 12GB free
                compute_capability: 10.3,              // GFX103x equivalent
                compute_units: 40,
                max_threads_per_block: 256,
                is_available: true,
            });
        }

        Ok(())
    }

    /// Enumerate SYCL devices
    fn enumerate_sycl_devices(&mut self) -> FFTResult<()> {
        // Initialize SYCL if available
        if init_sycl_device()? {
            // In a real implementation, this would query actual SYCL devices
            // For now, simulate one SYCL device
            self.devices.push(GPUDeviceInfo {
                device_id: 0,
                backend: GPUBackend::SYCL,
                device_name: "Intel GPU (simulated)".to_string(),
                memory_total: 4 * 1024 * 1024 * 1024, // 4GB
                memory_free: 3 * 1024 * 1024 * 1024,  // 3GB free
                compute_capability: 1.2,              // Intel GPU equivalent
                compute_units: 96,
                max_threads_per_block: 512,
                is_available: true,
            });
        }

        Ok(())
    }

    /// Select devices based on configuration
    fn select_devices(&mut self) -> FFTResult<()> {
        self.selected_devices.clear();

        // Filter available devices
        let available_devices: Vec<(usize, &GPUDeviceInfo)> = self
            .devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.is_available)
            .collect();

        if available_devices.is_empty() {
            return Err(FFTError::ComputationError(
                "No available GPU devices found".to_string(),
            ));
        }

        // Determine how many devices to use
        let max_devices = if self.config.max_devices == 0 {
            available_devices.len()
        } else {
            self.config.max_devices.min(available_devices.len())
        };

        // Select devices based on strategy
        match self.config.distribution {
            WorkloadDistribution::Equal => {
                // Use first N available devices
                for i in 0..max_devices {
                    self.selected_devices.push(available_devices[i].0);
                }
            }
            WorkloadDistribution::ComputeBased => {
                // Sort by compute capability and select top N
                let mut sorted_devices = available_devices;
                sorted_devices.sort_by(|a, b| {
                    b.1.compute_capability
                        .partial_cmp(&a.1.compute_capability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for i in 0..max_devices {
                    self.selected_devices.push(sorted_devices[i].0);
                }
            }
            WorkloadDistribution::MemoryBased => {
                // Sort by available memory and select top N
                let mut sorted_devices = available_devices;
                sorted_devices.sort_by(|a, b| b.1.memory_free.cmp(&a.1.memory_free));

                for i in 0..max_devices {
                    self.selected_devices.push(sorted_devices[i].0);
                }
            }
            WorkloadDistribution::Manual => {
                // Use manual selection (for now, just use first N devices)
                for i in 0..max_devices {
                    self.selected_devices.push(available_devices[i].0);
                }
            }
            WorkloadDistribution::Adaptive => {
                // Clone available devices to avoid borrow issues
                let available_devices_clone: Vec<(usize, GPUDeviceInfo)> = available_devices
                    .iter()
                    .map(|(idx, device)| (*idx, (*device).clone()))
                    .collect();

                // Use performance history to select best devices
                self.select_adaptive_devices_with_clone(available_devices_clone, max_devices)?;
            }
        }

        Ok(())
    }

    /// Select devices based on adaptive performance history
    fn select_adaptive_devices_with_clone(
        &mut self,
        available_devices: Vec<(usize, GPUDeviceInfo)>,
        max_devices: usize,
    ) -> FFTResult<()> {
        let performance_history = self.performance_history.lock().unwrap();

        // Calculate average performance for each device
        let mut device_scores: Vec<(usize, f64)> = available_devices
            .iter()
            .map(|(idx, device)| {
                let avg_performance = performance_history
                    .get(&device.device_id)
                    .map(|times| {
                        if times.is_empty() {
                            // Default score based on device capabilities
                            device.compute_capability as f64 * device.compute_units as f64
                        } else {
                            // Higher score for faster devices (lower execution times)
                            1.0 / (times.iter().sum::<f64>() / times.len() as f64)
                        }
                    })
                    .unwrap_or_else(|| {
                        // Default score for devices without history
                        device.compute_capability as f64 * device.compute_units as f64
                    });

                (*idx, avg_performance)
            })
            .collect();

        // Sort by performance score (descending)
        device_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top N devices
        for i in 0..max_devices {
            self.selected_devices.push(device_scores[i].0);
        }

        Ok(())
    }

    /// Get information about available devices
    pub fn get_devices(&self) -> &[GPUDeviceInfo] {
        &self.devices
    }

    /// Get information about selected devices
    pub fn get_selected_devices(&self) -> Vec<&GPUDeviceInfo> {
        self.selected_devices
            .iter()
            .map(|&idx| &self.devices[idx])
            .collect()
    }

    /// Perform multi-GPU sparse FFT
    pub fn sparse_fft<T>(&mut self, signal: &[T]) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug + Send + Sync + 'static,
    {
        if !self.initialized {
            self.initialize()?;
        }

        let signal_len = signal.len();

        // Check if signal is large enough for multi-GPU processing
        if signal_len < self.config.min_signal_size || self.selected_devices.len() <= 1 {
            // Fall back to single-device processing
            return self.single_device_sparse_fft(signal);
        }

        // Distribute workload across selected devices
        self.multi_device_sparse_fft(signal)
    }

    /// Single-device sparse FFT fallback
    fn single_device_sparse_fft<T>(&mut self, signal: &[T]) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Use the best available device
        let device_idx = self.selected_devices.first().copied().unwrap_or(0);
        let device = &self.devices[device_idx];

        // Create GPU configuration for the selected device
        let gpu_config = GPUSparseFFTConfig {
            base_config: self.config.base_config.clone(),
            backend: device.backend,
            device_id: device.device_id,
            ..GPUSparseFFTConfig::default()
        };

        // Create GPU processor and perform computation
        let mut processor = crate::sparse_fft_gpu::GPUSparseFFT::new(gpu_config);
        processor.sparse_fft(signal)
    }

    /// Multi-device sparse FFT implementation
    fn multi_device_sparse_fft<T>(&mut self, signal: &[T]) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug + Send + Sync + 'static,
    {
        let signal_len = signal.len();
        let num_devices = self.selected_devices.len();

        // Calculate chunk sizes based on distribution strategy
        let chunk_sizes = self.calculate_chunk_sizes(signal_len, num_devices)?;

        // Split signal into chunks
        let chunks = self.split_signal(signal, &chunk_sizes)?;

        // Process chunks in parallel across devices
        let chunk_results: Result<Vec<_>, _> = chunks
            .par_iter()
            .zip(self.selected_devices.par_iter())
            .map(|(chunk, &device_idx)| {
                let device = &self.devices[device_idx];
                let start_time = Instant::now();

                // Create GPU configuration for this device
                let gpu_config = GPUSparseFFTConfig {
                    base_config: self.config.base_config.clone(),
                    backend: device.backend,
                    device_id: device.device_id,
                    ..GPUSparseFFTConfig::default()
                };

                // Process chunk
                let mut processor = crate::sparse_fft_gpu::GPUSparseFFT::new(gpu_config);
                let result = processor.sparse_fft(chunk);

                // Record performance for adaptive selection
                if result.is_ok() {
                    let execution_time = start_time.elapsed().as_secs_f64();
                    if let Ok(mut history) = self.performance_history.try_lock() {
                        history
                            .entry(device.device_id)
                            .or_default()
                            .push(execution_time);

                        // Keep only recent history (last 10 measurements)
                        if let Some(times) = history.get_mut(&device.device_id) {
                            if times.len() > 10 {
                                times.drain(0..times.len() - 10);
                            }
                        }
                    }
                }

                result
            })
            .collect();

        let chunk_results = chunk_results?;

        // Combine results from all chunks
        self.combine_chunk_results(chunk_results)
    }

    /// Calculate chunk sizes for workload distribution
    fn calculate_chunk_sizes(
        &self,
        signal_len: usize,
        num_devices: usize,
    ) -> FFTResult<Vec<usize>> {
        let mut chunk_sizes = Vec::with_capacity(num_devices);

        match self.config.distribution {
            WorkloadDistribution::Equal => {
                let base_size = signal_len / num_devices;
                let remainder = signal_len % num_devices;

                for i in 0..num_devices {
                    let size = if i < remainder {
                        base_size + 1
                    } else {
                        base_size
                    };
                    chunk_sizes.push(size);
                }
            }
            WorkloadDistribution::ComputeBased => {
                // Distribute based on compute capability
                let total_compute: f32 = self
                    .selected_devices
                    .iter()
                    .map(|&idx| {
                        self.devices[idx].compute_capability
                            * self.devices[idx].compute_units as f32
                    })
                    .sum();

                let mut remaining = signal_len;
                for (i, &device_idx) in self.selected_devices.iter().enumerate() {
                    let device = &self.devices[device_idx];
                    let device_compute = device.compute_capability * device.compute_units as f32;
                    let ratio = device_compute / total_compute;

                    let size = if i == num_devices - 1 {
                        remaining // Give remainder to last device
                    } else {
                        let size = (signal_len as f32 * ratio) as usize;
                        remaining = remaining.saturating_sub(size);
                        size
                    };

                    chunk_sizes.push(size);
                }
            }
            WorkloadDistribution::MemoryBased => {
                // Distribute based on available memory
                let total_memory: usize = self
                    .selected_devices
                    .iter()
                    .map(|&idx| self.devices[idx].memory_free)
                    .sum();

                let mut remaining = signal_len;
                for (i, &device_idx) in self.selected_devices.iter().enumerate() {
                    let device = &self.devices[device_idx];
                    let ratio = device.memory_free as f32 / total_memory as f32;

                    let size = if i == num_devices - 1 {
                        remaining
                    } else {
                        let size = (signal_len as f32 * ratio) as usize;
                        remaining = remaining.saturating_sub(size);
                        size
                    };

                    chunk_sizes.push(size);
                }
            }
            WorkloadDistribution::Manual => {
                if self.config.manual_ratios.len() != num_devices {
                    return Err(FFTError::ValueError(
                        "Manual ratios length must match number of selected devices".to_string(),
                    ));
                }

                let total_ratio: f32 = self.config.manual_ratios.iter().sum();
                let mut remaining = signal_len;

                for (i, &ratio) in self.config.manual_ratios.iter().enumerate() {
                    let size = if i == num_devices - 1 {
                        remaining
                    } else {
                        let size = (signal_len as f32 * ratio / total_ratio) as usize;
                        remaining = remaining.saturating_sub(size);
                        size
                    };

                    chunk_sizes.push(size);
                }
            }
            WorkloadDistribution::Adaptive => {
                // Use performance history to determine optimal distribution
                // For now, fall back to compute-based distribution
                return self.calculate_chunk_sizes(signal_len, num_devices);
            }
        }

        Ok(chunk_sizes)
    }

    /// Split signal into chunks based on calculated sizes
    fn split_signal<T>(&self, signal: &[T], chunk_sizes: &[usize]) -> FFTResult<Vec<Vec<T>>>
    where
        T: Copy,
    {
        let mut chunks = Vec::new();
        let mut offset = 0;

        for &chunk_size in chunk_sizes {
            if offset + chunk_size > signal.len() {
                return Err(FFTError::ValueError(
                    "Chunk sizes exceed signal length".to_string(),
                ));
            }

            let chunk_end = offset + chunk_size;
            let chunk = signal[offset..chunk_end].to_vec();
            chunks.push(chunk);
            offset = chunk_end;
        }

        Ok(chunks)
    }

    /// Combine results from multiple chunks
    fn combine_chunk_results(
        &self,
        chunk_results: Vec<SparseFFTResult>,
    ) -> FFTResult<SparseFFTResult> {
        if chunk_results.is_empty() {
            return Err(FFTError::ComputationError(
                "No chunk results to combine".to_string(),
            ));
        }

        if chunk_results.len() == 1 {
            return Ok(chunk_results.into_iter().next().unwrap());
        }

        // Use the computation time from the slowest device
        let max_computation_time = chunk_results
            .iter()
            .map(|r| r.computation_time)
            .max()
            .unwrap_or_default();

        // Combine frequency components from all chunks
        let mut combined_values = Vec::new();
        let mut combined_indices = Vec::new();
        let mut index_offset = 0;

        for result in chunk_results {
            // Store the indices length before moving
            let indices_len = result.indices.len();

            // Add values from this chunk
            combined_values.extend(result.values);

            // Adjust indices to account for chunk offset
            let adjusted_indices: Vec<usize> = result
                .indices
                .into_iter()
                .map(|idx| idx + index_offset)
                .collect();
            combined_indices.extend(adjusted_indices);

            // Update offset for next chunk
            // This is a simplified approach - in practice, frequency domain combining is more complex
            index_offset += indices_len;
        }

        // Remove duplicates and sort
        let mut frequency_map: std::collections::HashMap<usize, Complex64> =
            std::collections::HashMap::new();

        for (idx, value) in combined_indices.iter().zip(combined_values.iter()) {
            frequency_map.insert(*idx, *value);
        }

        let mut sorted_entries: Vec<_> = frequency_map.into_iter().collect();
        sorted_entries.sort_by_key(|&(idx, _)| idx);

        let final_indices: Vec<usize> = sorted_entries.iter().map(|(idx, _)| *idx).collect();
        let final_values: Vec<Complex64> = sorted_entries.iter().map(|(_, val)| *val).collect();

        // Calculate combined sparsity
        let total_estimated_sparsity = final_values.len();

        Ok(SparseFFTResult {
            values: final_values,
            indices: final_indices,
            estimated_sparsity: total_estimated_sparsity,
            computation_time: max_computation_time,
            algorithm: self.config.base_config.algorithm,
        })
    }

    /// Get performance statistics for each device
    pub fn get_performance_stats(&self) -> HashMap<i32, Vec<f64>> {
        self.performance_history.lock().unwrap().clone()
    }

    /// Reset performance history
    pub fn reset_performance_history(&mut self) {
        self.performance_history.lock().unwrap().clear();
    }
}

/// Convenience function for multi-GPU sparse FFT with default configuration
pub fn multi_gpu_sparse_fft<T>(
    signal: &[T],
    k: usize,
    algorithm: Option<SparseFFTAlgorithm>,
    window_function: Option<WindowFunction>,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + Send + Sync + 'static,
{
    let base_config = SparseFFTConfig {
        sparsity: k,
        algorithm: algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear),
        window_function: window_function.unwrap_or(WindowFunction::None),
        ..SparseFFTConfig::default()
    };

    let config = MultiGPUConfig {
        base_config,
        ..MultiGPUConfig::default()
    };

    let mut processor = MultiGPUSparseFFT::new(config);
    processor.sparse_fft(signal)
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
    #[ignore = "Ignored for alpha-4 release - multi-GPU dependent test"]
    fn test_multi_gpu_initialization() {
        let mut processor = MultiGPUSparseFFT::new(MultiGPUConfig::default());
        let result = processor.initialize();

        // Should succeed even if no GPU devices available (CPU fallback)
        assert!(result.is_ok());
        assert!(!processor.get_devices().is_empty());
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - multi-GPU dependent test"]
    fn test_device_enumeration() {
        let mut processor = MultiGPUSparseFFT::new(MultiGPUConfig::default());
        processor.initialize().unwrap();

        let devices = processor.get_devices();
        assert!(!devices.is_empty());

        // Should have at least CPU fallback
        assert!(devices.iter().any(|d| d.backend == GPUBackend::CPUFallback));
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - multi-GPU dependent test"]
    fn test_multi_gpu_sparse_fft() {
        let n = 8192; // Large enough to trigger multi-GPU
        let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.25)];
        let signal = create_sparse_signal(n, &frequencies);

        let result = multi_gpu_sparse_fft(
            &signal,
            6,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.values.is_empty());
        assert_eq!(result.values.len(), result.indices.len());
    }

    #[test]
    fn test_chunk_size_calculation() {
        let config = MultiGPUConfig {
            distribution: WorkloadDistribution::Equal,
            ..MultiGPUConfig::default()
        };
        let mut processor = MultiGPUSparseFFT::new(config);

        // Simulate device setup
        processor.selected_devices = vec![0, 1, 2];

        let chunk_sizes = processor.calculate_chunk_sizes(1000, 3).unwrap();
        assert_eq!(chunk_sizes.len(), 3);
        assert_eq!(chunk_sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn test_signal_splitting() {
        let processor = MultiGPUSparseFFT::new(MultiGPUConfig::default());
        let signal = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunk_sizes = vec![3, 3, 4];

        let chunks = processor.split_signal(&signal, &chunk_sizes).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7, 8, 9, 10]);
    }
}
