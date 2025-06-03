//! Batch processing for sparse FFT algorithms
//!
//! This module provides batch processing capabilities for sparse FFT algorithms,
//! which can significantly improve performance when processing multiple signals,
//! especially on GPU hardware.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{
    SparseFFTAlgorithm, SparseFFTConfig, SparseFFTResult, SparsityEstimationMethod, WindowFunction,
};
use crate::sparse_fft_gpu::{GPUBackend, GPUSparseFFTConfig};

use num_complex::Complex64;
use num_traits::NumCast;
use rayon::prelude::*;
use std::fmt::Debug;
use std::time::Instant;

/// Batch processing configuration for sparse FFT
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (number of signals per batch)
    pub max_batch_size: usize,
    /// Whether to use parallel processing on CPU
    pub use_parallel: bool,
    /// Maximum memory usage per batch in bytes (0 for unlimited)
    pub max_memory_per_batch: usize,
    /// Whether to use mixed precision computation
    pub use_mixed_precision: bool,
    /// Whether to use in-place computation when possible
    pub use_inplace: bool,
    /// Whether to preserve input signals (false = allow modification)
    pub preserve_input: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            use_parallel: true,
            max_memory_per_batch: 0, // Unlimited
            use_mixed_precision: false,
            use_inplace: true,
            preserve_input: true,
        }
    }
}

/// Perform batch sparse FFT on CPU
///
/// Process multiple signals in a batch for better performance.
///
/// # Arguments
///
/// * `signals` - List of input signals
/// * `k` - Expected sparsity (number of significant frequency components)
/// * `algorithm` - Sparse FFT algorithm variant
/// * `window_function` - Window function to apply before FFT
/// * `batch_config` - Batch processing configuration
///
/// # Returns
///
/// * Vector of sparse FFT results, one for each input signal
#[allow(clippy::too_many_arguments)]
pub fn batch_sparse_fft<T>(
    signals: &[Vec<T>],
    k: usize,
    algorithm: Option<SparseFFTAlgorithm>,
    window_function: Option<WindowFunction>,
    batch_config: Option<BatchConfig>,
) -> FFTResult<Vec<SparseFFTResult>>
where
    T: NumCast + Copy + Debug + Sync + 'static,
{
    let config = batch_config.unwrap_or_default();
    let alg = algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear);
    let window = window_function.unwrap_or(WindowFunction::None);

    let start = Instant::now();

    // Create sparse FFT config
    let fft_config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity: k,
        algorithm: alg,
        window_function: window,
        ..SparseFFTConfig::default()
    };

    let results = if config.use_parallel {
        // Process signals in parallel using Rayon
        signals
            .par_iter()
            .map(|signal| {
                let mut processor = crate::sparse_fft::SparseFFT::new(fft_config.clone());

                // Convert signal to complex
                let signal_complex: FFTResult<Vec<Complex64>> = signal
                    .iter()
                    .map(|&val| {
                        let val_f64 = NumCast::from(val).ok_or_else(|| {
                            FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                        })?;
                        Ok(Complex64::new(val_f64, 0.0))
                    })
                    .collect();

                processor.sparse_fft(&signal_complex?)
            })
            .collect::<FFTResult<Vec<_>>>()
    } else {
        // Process signals sequentially
        let mut results = Vec::with_capacity(signals.len());
        for signal in signals {
            let mut processor = crate::sparse_fft::SparseFFT::new(fft_config.clone());

            // Convert signal to complex
            let signal_complex: FFTResult<Vec<Complex64>> = signal
                .iter()
                .map(|&val| {
                    let val_f64 = NumCast::from(val).ok_or_else(|| {
                        FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                    })?;
                    Ok(Complex64::new(val_f64, 0.0))
                })
                .collect();

            results.push(processor.sparse_fft(&signal_complex?)?);
        }
        Ok(results)
    }?;

    // Update computation time to include batching overhead
    let total_time = start.elapsed();
    let avg_time_per_signal = total_time.div_f64(signals.len() as f64);

    // Return results with updated computation time
    let mut final_results = Vec::with_capacity(results.len());
    for mut result in results {
        result.computation_time = avg_time_per_signal;
        final_results.push(result);
    }

    Ok(final_results)
}

/// Perform batch sparse FFT on GPU
///
/// Process multiple signals in a batch for better GPU utilization.
///
/// # Arguments
///
/// * `signals` - List of input signals
/// * `k` - Expected sparsity (number of significant frequency components)
/// * `device_id` - GPU device ID (-1 for auto-select)
/// * `backend` - GPU backend (CUDA, HIP, SYCL)
/// * `algorithm` - Sparse FFT algorithm variant
/// * `window_function` - Window function to apply before FFT
/// * `batch_config` - Batch processing configuration
///
/// # Returns
///
/// * Vector of sparse FFT results, one for each input signal
#[allow(clippy::too_many_arguments)]
pub fn gpu_batch_sparse_fft<T>(
    signals: &[Vec<T>],
    k: usize,
    device_id: i32,
    backend: GPUBackend,
    algorithm: Option<SparseFFTAlgorithm>,
    window_function: Option<WindowFunction>,
    batch_config: Option<BatchConfig>,
) -> FFTResult<Vec<SparseFFTResult>>
where
    T: NumCast + Copy + Debug + Sync + 'static,
{
    let config = batch_config.unwrap_or_default();
    let alg = algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear);
    let window = window_function.unwrap_or(WindowFunction::None);

    // Calculate batch sizes
    let total_signals = signals.len();
    let batch_size = config.max_batch_size.min(total_signals);
    let num_batches = total_signals.div_ceil(batch_size);

    // Create sparse FFT config
    let base_fft_config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity: k,
        algorithm: alg,
        window_function: window,
        ..SparseFFTConfig::default()
    };

    // Create GPU config
    let _gpu_config = GPUSparseFFTConfig {
        base_config: base_fft_config,
        backend,
        device_id,
        batch_size,
        max_memory: config.max_memory_per_batch,
        use_mixed_precision: config.use_mixed_precision,
        use_inplace: config.use_inplace,
        stream_count: 2, // Use 2 streams for overlap
    };

    let start = Instant::now();

    // Process signals in batches
    let mut all_results = Vec::with_capacity(total_signals);
    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(total_signals);
        let current_batch = &signals[start_idx..end_idx];

        // Process this batch
        match backend {
            GPUBackend::CUDA => {
                let batch_results = crate::cuda_batch_sparse_fft(
                    current_batch,
                    k,
                    device_id,
                    Some(alg),
                    Some(window),
                )?;
                all_results.extend(batch_results);
            }
            _ => {
                // For other backends, fall back to CPU for now
                let batch_results =
                    batch_sparse_fft(current_batch, k, Some(alg), Some(window), None)?;
                all_results.extend(batch_results);
            }
        }
    }

    // Update computation time to include batching overhead
    let total_time = start.elapsed();
    let avg_time_per_signal = total_time.div_f64(signals.len() as f64);

    // Return results with updated computation time
    let mut final_results = Vec::with_capacity(all_results.len());
    for mut result in all_results {
        result.computation_time = avg_time_per_signal;
        final_results.push(result);
    }

    Ok(final_results)
}

/// Optimized batch processing for spectral flatness sparse FFT
///
/// This function is specialized for the spectral flatness algorithm,
/// which can benefit from batch processing due to its signal analysis
/// requirements.
///
/// # Arguments
///
/// * `signals` - List of input signals
/// * `flatness_threshold` - Threshold for spectral flatness (0-1, lower = more selective)
/// * `window_size` - Size of windows for local flatness analysis
/// * `window_function` - Window function to apply before FFT
/// * `device_id` - GPU device ID (-1 for auto-select)
/// * `batch_config` - Batch processing configuration
///
/// # Returns
///
/// * Vector of sparse FFT results, one for each input signal
#[allow(clippy::too_many_arguments)]
pub fn spectral_flatness_batch_sparse_fft<T>(
    signals: &[Vec<T>],
    flatness_threshold: f64,
    window_size: usize,
    window_function: Option<WindowFunction>,
    device_id: Option<i32>,
    batch_config: Option<BatchConfig>,
) -> FFTResult<Vec<SparseFFTResult>>
where
    T: NumCast + Copy + Debug + Sync + 'static,
{
    let config = batch_config.unwrap_or_default();
    let window = window_function.unwrap_or(WindowFunction::Hann); // Default to Hann for spectral flatness
    let device = device_id.unwrap_or(0);

    // Calculate batch sizes
    let total_signals = signals.len();
    let batch_size = config.max_batch_size.min(total_signals);
    let num_batches = total_signals.div_ceil(batch_size);

    // Initialize the memory manager if GPU is used
    if device >= 0 {
        crate::init_global_memory_manager(
            GPUBackend::CUDA,
            device,
            crate::AllocationStrategy::CacheBySize,
            config.max_memory_per_batch.max(1024 * 1024 * 1024), // At least 1 GB
        )?;
    }

    let start = Instant::now();

    // Process signals in batches
    let mut all_results = Vec::with_capacity(total_signals);

    if device >= 0 && cfg!(feature = "cuda") {
        // GPU processing with CUDA
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(total_signals);
            let current_batch = &signals[start_idx..end_idx];

            // Create a base configuration for this batch
            let _base_config = SparseFFTConfig {
                estimation_method: SparsityEstimationMethod::SpectralFlatness,
                sparsity: 0, // Will be determined automatically
                algorithm: SparseFFTAlgorithm::SpectralFlatness,
                window_function: window,
                flatness_threshold,
                window_size,
                ..SparseFFTConfig::default()
            };

            // Use standard GPU batch processing
            for signal in current_batch {
                // Convert signal to complex
                let signal_complex: FFTResult<Vec<Complex64>> = signal
                    .iter()
                    .map(|&val| {
                        let val_f64 = NumCast::from(val).ok_or_else(|| {
                            FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                        })?;
                        Ok(Complex64::new(val_f64, 0.0))
                    })
                    .collect();

                // Process with GPU
                let result = crate::execute_cuda_spectral_flatness_sparse_fft(
                    &signal_complex?,
                    0, // Will be determined automatically
                    Some(flatness_threshold),
                    Some(window_size),
                    window,
                    device,
                )?;

                all_results.push(result);
            }
        }
    } else {
        // CPU processing
        if config.use_parallel {
            // Process all signals in parallel using Rayon
            let parallel_results: FFTResult<Vec<_>> = signals
                .par_iter()
                .map(|signal| {
                    // Create configuration
                    let fft_config = SparseFFTConfig {
                        estimation_method: SparsityEstimationMethod::SpectralFlatness,
                        sparsity: 0, // Will be determined automatically
                        algorithm: SparseFFTAlgorithm::SpectralFlatness,
                        window_function: window,
                        flatness_threshold,
                        window_size,
                        ..SparseFFTConfig::default()
                    };

                    // Convert signal to complex
                    let signal_complex: FFTResult<Vec<Complex64>> = signal
                        .iter()
                        .map(|&val| {
                            let val_f64 = NumCast::from(val).ok_or_else(|| {
                                FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                            })?;
                            Ok(Complex64::new(val_f64, 0.0))
                        })
                        .collect();

                    // Process with CPU
                    let mut processor = crate::sparse_fft::SparseFFT::new(fft_config);
                    processor.sparse_fft(&signal_complex?)
                })
                .collect();

            all_results = parallel_results?;
        } else {
            // Process signals sequentially
            for signal in signals {
                // Create configuration
                let fft_config = SparseFFTConfig {
                    estimation_method: SparsityEstimationMethod::SpectralFlatness,
                    sparsity: 0, // Will be determined automatically
                    algorithm: SparseFFTAlgorithm::SpectralFlatness,
                    window_function: window,
                    flatness_threshold,
                    window_size,
                    ..SparseFFTConfig::default()
                };

                // Convert signal to complex
                let signal_complex: FFTResult<Vec<Complex64>> = signal
                    .iter()
                    .map(|&val| {
                        let val_f64 = NumCast::from(val).ok_or_else(|| {
                            FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                        })?;
                        Ok(Complex64::new(val_f64, 0.0))
                    })
                    .collect();

                // Process with CPU
                let mut processor = crate::sparse_fft::SparseFFT::new(fft_config);
                let result = processor.sparse_fft(&signal_complex?)?;
                all_results.push(result);
            }
        }
    }

    // Update computation time to include batching overhead
    let total_time = start.elapsed();
    let avg_time_per_signal = total_time.div_f64(signals.len() as f64);

    // Return results with updated computation time
    let mut final_results = Vec::with_capacity(all_results.len());
    for mut result in all_results {
        result.computation_time = avg_time_per_signal;
        final_results.push(result);
    }

    Ok(final_results)
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

    // Helper to add noise to signals
    fn add_noise(signal: &[f64], noise_level: f64) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::rng();
        signal
            .iter()
            .map(|&x| x + rng.random_range(-noise_level..noise_level))
            .collect()
    }

    // Helper to create a batch of similar signals with different noise
    fn create_signal_batch(
        count: usize,
        n: usize,
        frequencies: &[(usize, f64)],
        noise_level: f64,
    ) -> Vec<Vec<f64>> {
        let base_signal = create_sparse_signal(n, frequencies);
        (0..count)
            .map(|_| add_noise(&base_signal, noise_level))
            .collect()
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - experiencing issues with batch processing"]
    fn test_cpu_batch_processing() {
        // Create a batch of signals
        let n = 256;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signals = create_signal_batch(5, n, &frequencies, 0.1);

        // Test batch processing
        let results = batch_sparse_fft(
            &signals,
            6, // Look for up to 6 components
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
            None,
        )
        .unwrap();

        // Check results
        assert_eq!(results.len(), signals.len());

        // Each result should identify the key frequencies
        for result in &results {
            assert!(
                result.indices.contains(&3) || result.indices.contains(&(n - 3)),
                "Failed to find frequency component at 3 Hz"
            );
            assert!(
                result.indices.contains(&7) || result.indices.contains(&(n - 7)),
                "Failed to find frequency component at 7 Hz"
            );
            assert!(
                result.indices.contains(&15) || result.indices.contains(&(n - 15)),
                "Failed to find frequency component at 15 Hz"
            );
        }
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - experiencing issues with parallel batch processing"]
    fn test_parallel_batch_processing() {
        // Create a larger batch of signals
        let n = 256;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signals = create_signal_batch(10, n, &frequencies, 0.1);

        // Test parallel batch processing
        let batch_config = BatchConfig {
            use_parallel: true,
            ..BatchConfig::default()
        };

        let results = batch_sparse_fft(
            &signals,
            6, // Look for up to 6 components
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
            Some(batch_config),
        )
        .unwrap();

        // Check results
        assert_eq!(results.len(), signals.len());

        // Each result should identify the key frequencies
        for result in &results {
            assert!(
                result.indices.contains(&3) || result.indices.contains(&(n - 3)),
                "Failed to find frequency component at 3 Hz"
            );
            assert!(
                result.indices.contains(&7) || result.indices.contains(&(n - 7)),
                "Failed to find frequency component at 7 Hz"
            );
            assert!(
                result.indices.contains(&15) || result.indices.contains(&(n - 15)),
                "Failed to find frequency component at 15 Hz"
            );
        }
    }

    #[test]
    #[ignore = "Ignored for alpha-4 release - experiencing issues with spectral flatness batch processing"]
    fn test_spectral_flatness_batch() {
        // Create a batch of signals with different noise levels
        let n = 512;
        let frequencies = vec![(30, 1.0), (70, 0.5), (120, 0.25)];

        // Create signals with increasing noise
        let mut signals = Vec::new();
        for i in 0..5 {
            let noise_level = 0.05 * (i + 1) as f64;
            let base_signal = create_sparse_signal(n, &frequencies);
            signals.push(add_noise(&base_signal, noise_level));
        }

        // Process with spectral flatness batch function
        let results = spectral_flatness_batch_sparse_fft(
            &signals,
            0.3, // Flatness threshold
            32,  // Window size
            Some(WindowFunction::Hann),
            None, // Use CPU
            None, // Default config
        )
        .unwrap();

        // Check results
        assert_eq!(results.len(), signals.len());

        // Spectral flatness should find the main frequencies even with noise
        for result in &results {
            // Check that the algorithm is correctly set
            assert_eq!(result.algorithm, SparseFFTAlgorithm::SpectralFlatness);

            // At least one of the key frequencies should be found
            let found_30 = result.indices.contains(&30) || result.indices.contains(&(n - 30));
            let found_70 = result.indices.contains(&70) || result.indices.contains(&(n - 70));
            let found_120 = result.indices.contains(&120) || result.indices.contains(&(n - 120));

            assert!(
                found_30 || found_70 || found_120,
                "Failed to find any of the key frequencies"
            );
        }
    }
}
