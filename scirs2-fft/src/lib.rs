#![allow(deprecated)]
//! Fast Fourier Transform module
//!
//! This module provides implementations of various Fast Fourier Transform algorithms,
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
//! following `SciPy`'s `fft` module.
//!
//! ## Overview
//!
//! * Fast Fourier Transform (FFT) and inverse FFT for 1D, 2D, and N-dimensional arrays
//! * Real-to-complex and complex-to-real FFT optimized for real-valued data
//! * Hermitian-to-real and real-to-Hermitian FFT for complex signals with real spectra
//! * Discrete cosine transform (DCT) types I-IV and their inverses
//! * Discrete sine transform (DST) types I-IV and their inverses
//! * Fractional Fourier Transform (FrFT) for rotations in the time-frequency plane
//! * Non-Uniform Fast Fourier Transform (NUFFT) for non-uniformly sampled data
//! * Spectrogram and Short-Time Fourier Transform (STFT) for time-frequency analysis
//! * Waterfall plots for 3D visualization of time-frequency data
//! * Window functions for signal processing (Hann, Hamming, Blackman, etc.)
//! * Helper functions for frequency domain calculations and visualization
//!
//! ## Implementation Notes
//!
//! * Built on `rustfft` for efficient computation
//! * Provides proper zero padding and array reshaping
//! * Normalization options for compatibility with `SciPy`
//! * Support for ndarrays through ndarray integration
//!
//! ## Examples
//!
//! ```
//! use scirs2_fft::{fft, ifft};
//! use num_complex::Complex64;
//!
//! // Generate a simple signal
//! let signal = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Compute FFT of the signal
//! let spectrum = fft(&signal, None).unwrap();
//!
//! // Inverse FFT should recover the original signal
//! let recovered = ifft(&spectrum, None).unwrap();
//!
//! // Check that the recovered signal matches the original
//! for (x, y) in signal.iter().zip(recovered.iter()) {
//!     assert!((x - y.re).abs() < 1e-10);
//!     assert!(y.im.abs() < 1e-10);
//! }
//! ```

// Export error types
pub mod error;
pub use error::{FFTError, FFTResult};

// FFT plan caching
pub mod plan_cache;
pub use plan_cache::{get_global_cache, init_global_cache, CacheStats, PlanCache};

// Worker pool management
pub mod worker_pool;
pub use worker_pool::{
    get_global_pool, get_workers, init_global_pool, set_workers, with_workers, WorkerConfig,
    WorkerPool, WorkerPoolInfo,
};

// FFT backend system
pub mod backend;
pub use backend::{
    get_backend_info, get_backend_manager, get_backend_name, init_backend_manager, list_backends,
    set_backend, BackendContext, BackendInfo, BackendManager, FftBackend,
};

// FFT context managers
pub mod context;
pub use context::{
    fft_context, with_backend, with_fft_settings, without_cache, FftContext, FftContextBuilder,
    FftSettingsGuard,
};

// Advanced striding support
pub mod strided_fft;
pub use strided_fft::{fft_strided, fft_strided_complex, ifft_strided};

// Plan serialization
pub mod plan_serialization;
pub use plan_serialization::{PlanDatabaseStats, PlanInfo, PlanMetrics, PlanSerializationManager};

// Advanced FFT planning system
pub mod planning;
pub use planning::{
    get_global_planner, init_global_planner, plan_ahead_of_time, AdvancedFftPlanner as FftPlanner,
    FftPlan, FftPlanExecutor, PlanBuilder, PlannerBackend, PlanningConfig, PlanningStrategy,
};

// Adaptive planning extensions
pub mod planning_adaptive;

// Parallel planning extensions
pub mod planning_parallel;
pub use planning_parallel::{
    ParallelExecutor, ParallelPlanResult, ParallelPlanner, ParallelPlanningConfig,
};

// Auto-tuning for hardware optimization
pub mod auto_tuning;
pub use auto_tuning::{AutoTuneConfig, AutoTuner, FftVariant, SizeRange, SizeStep};

// Advanced mode coordinator for advanced AI-driven optimization (temporarily disabled)
// pub mod advanced_coordinator;
// pub use advanced_coordinator::{
//     create_advanced_fft_coordinator, create_advanced_fft_coordinator_with_config,
//     FftPerformanceMetrics, FftRecommendation, advancedFftConfig, advancedFftCoordinator,
// };

// Core modules are used conditionally in feature-specific implementations

// FFT module structure
pub mod dct;
pub mod dst;
pub mod fft;
pub mod fht;
pub mod hfft;
pub mod rfft;

// Re-export basic functions
pub use dct::{dct, dct2, dctn, idct, idct2, idctn, DCTType};
pub use dst::{dst, dst2, dstn, idst, idst2, idstn, DSTType};
pub use fft::{fft, fft2, fftn, ifft, ifft2, ifftn};
pub use fht::{fht, fht_sample_points, fhtoffset, ifht};
pub use hfft::{hfft, hfft2, hfftn, ihfft, ihfft2, ihfftn};

// Re-export parallel implementations when available
#[cfg(feature = "parallel")]
pub use fft::{fft2_parallel, ifft2_parallel};
pub use rfft::{irfft, irfft2, irfftn, rfft, rfft2, rfftn};

// Re-export SIMD-optimized implementations
pub use simd_fft::{
    fft2_adaptive, fft2_simd, fft_adaptive, fft_simd, fftn_adaptive, fftn_simd, ifft2_adaptive,
    ifft2_simd, ifft_adaptive, ifft_simd, ifftn_adaptive, ifftn_simd, simd_support_available,
};

// Real FFT SIMD module
pub mod simd_rfft;
pub use simd_rfft::{irfft_adaptive, irfft_simd, rfft_adaptive, rfft_simd};

// Helper modules
pub mod helper;
pub use helper::{fftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len, rfftfreq};

// Advanced FFT modules
pub mod frft;
pub mod frft_dft;
pub mod frft_ozaktas;
pub mod nufft;
pub mod spectrogram;
pub mod waterfall;
pub use frft::{frft, frft_complex, frft_dft, frft_stable};
pub use spectrogram::{spectrogram, spectrogram_normalized, stft as spectrogram_stft};
pub use waterfall::{
    apply_colormap, waterfall_3d, waterfall_lines, waterfall_mesh, waterfall_mesh_colored,
};

// Long-term goal implementations
#[cfg(feature = "never")]
pub mod distributed;
pub mod gpu_kernel_stub;
#[cfg(feature = "never")]
pub mod optimized_fft;
#[cfg(feature = "never")]
pub mod signal_processing;
pub mod simd_fft;
pub mod sparse_fft;
pub mod sparse_fft_cuda_kernels;
pub mod sparse_fft_cuda_kernels_frequency_pruning;
pub mod sparse_fft_cuda_kernels_iterative;
pub mod sparse_fft_cuda_kernels_spectral_flatness;
pub mod sparse_fft_gpu;
pub mod sparse_fft_gpu_cuda;
pub mod sparse_fft_gpu_kernels;
pub mod sparse_fft_gpu_memory;
#[cfg(feature = "never")]
pub mod time_frequency;
#[cfg(feature = "never")]
pub use distributed::{
    CommunicationPattern, DecompositionStrategy, DistributedConfig, DistributedFFT,
};
#[cfg(feature = "never")]
pub use optimized_fft::{OptimizationLevel, OptimizedConfig, OptimizedFFT};
#[cfg(feature = "never")]
pub use signal_processing::{
    convolve, cross_correlate, design_fir_filter, fir_filter, frequency_filter, FilterSpec,
    FilterType, FilterWindow,
};
pub use sparse_fft::WindowFunction;
pub use sparse_fft::{
    adaptive_sparse_fft, frequency_pruning_sparse_fft, reconstruct_filtered,
    reconstruct_high_resolution, reconstruct_spectrum, reconstruct_time_domain, sparse_fft,
    sparse_fft2, sparse_fftn, spectral_flatness_sparse_fft,
};
pub use sparse_fft_cuda_kernels::{
    execute_cuda_compressed_sensing_sparse_fft, execute_cuda_sublinear_sparse_fft,
    CUDACompressedSensingSparseFFTKernel, CUDASublinearSparseFFTKernel, CUDAWindowKernel,
};
pub use sparse_fft_cuda_kernels_frequency_pruning::{
    execute_cuda_frequency_pruning_sparse_fft, CUDAFrequencyPruningSparseFFTKernel,
};
pub use sparse_fft_cuda_kernels_iterative::{
    execute_cuda_iterative_sparse_fft, CUDAIterativeSparseFFTKernel,
};
pub use sparse_fft_cuda_kernels_spectral_flatness::{
    execute_cuda_spectral_flatness_sparse_fft, CUDASpectralFlatnessSparseFFTKernel,
};
pub use sparse_fft_gpu::{gpu_batch_sparse_fft, gpu_sparse_fft, GPUBackend};
pub use sparse_fft_gpu_cuda::{
    cuda_batch_sparse_fft,
    cuda_sparse_fft,
    get_cuda_devices,
    FftGpuContext,
    GpuDeviceInfo,
    // CUDAStream - migrated to core GPU abstractions
};
pub use sparse_fft_gpu_kernels::{
    execute_sparse_fft_kernel, GPUKernel, KernelConfig, KernelFactory, KernelImplementation,
    KernelLauncher, KernelStats,
};
pub use sparse_fft_gpu_memory::{
    get_global_memory_manager, init_global_memory_manager, memory_efficient_gpu_sparse_fft,
    AllocationStrategy, BufferLocation, BufferType,
};
pub use sparse_fft_gpu_memory::{is_cuda_available, is_hip_available, is_sycl_available};

// Multi-GPU processing module
pub mod sparse_fft_multi_gpu;
pub use sparse_fft_multi_gpu::{
    multi_gpu_sparse_fft, GPUDeviceInfo, MultiGPUConfig, MultiGPUSparseFFT, WorkloadDistribution,
};

// Specialized hardware support module
pub mod sparse_fft_specialized_hardware;
pub use sparse_fft_specialized_hardware::{
    specialized_hardware_sparse_fft, AcceleratorCapabilities, AcceleratorInfo, AcceleratorType,
    HardwareAbstractionLayer, SpecializedHardwareManager,
};
// Batch processing module
pub mod sparse_fft_batch;
pub use sparse_fft_batch::{batch_sparse_fft, spectral_flatness_batch_sparse_fft, BatchConfig};

#[cfg(feature = "never")]
pub use time_frequency::{time_frequency_transform, TFConfig, TFTransform, WaveletType};

// Memory-efficient FFT operations
pub mod memory_efficient;
pub use memory_efficient::{
    fft2_efficient, fft_inplace, fft_streaming, process_in_chunks, FftMode,
};

// Optimized N-dimensional FFT
pub mod ndim_optimized;
pub use ndim_optimized::{fftn_memory_efficient, fftn_optimized, rfftn_optimized};

// Hartley transform
pub mod hartley;
pub use hartley::{dht, dht2, fht as hartley_fht, idht};

// Higher-order DCT and DST types (V-VIII)
pub mod higher_order_dct_dst;
pub use higher_order_dct_dst::{
    dct_v, dct_vi, dct_vii, dct_viii, dst_v, dst_vi, dst_vii, dst_viii, idct_v, idct_vi, idct_vii,
    idct_viii, idst_v, idst_vi, idst_vii, idst_viii,
};

// Modified DCT and DST (MDCT/MDST)
pub mod mdct;
pub use mdct::{imdct, imdst, mdct, mdct_overlap_add, mdst};

// Window functions
pub mod window;
pub use window::{apply_window, get_window, Window};

// Extended window functions and analysis
pub mod window_extended;
pub use window_extended::{
    analyze_window, compare_windows, get_extended_window, visualize_window, ExtendedWindow,
    WindowProperties,
};

// Chirp Z-Transform
pub mod czt;
pub use czt::{czt, czt_points, zoom_fft, CZT};

// Automatic padding strategies
pub mod padding;
pub use padding::{
    auto_pad_1d, auto_pad_complex, auto_pad_nd, remove_padding_1d, AutoPadConfig, PaddingMode,
};

/// Performs a Short-Time Fourier Transform (STFT).
///
/// Short-Time Fourier Transform (STFT) is used to determine the sinusoidal
/// frequency and phase content of local sections of a signal as it changes over time.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `window` - Window function to apply
/// * `nperseg` - Length of each segment
/// * `noverlap` - Number of points to overlap between segments
/// * `nfft` - Length of the FFT (optional, default is nperseg)
/// * `fs` - Sampling frequency of the signal
/// * `detrend` - Whether to remove the mean from each segment
/// * `boundary` - Boundary to pad with ('zeros', 'constant', 'reflect', etc.)
///
/// # Returns
///
/// * Tuple of (frequencies, times, Zxx) where Zxx is the STFT result
///
/// # Errors
///
/// Returns an error if the computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::{stft, window::Window};
/// use std::f64::consts::PI;
///
/// // Generate a simple sine wave
/// let fs = 1000.0; // 1 kHz sampling rate
/// let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let signal = t.iter().map(|&ti| (2.0 * PI * 100.0 * ti).sin()).collect::<Vec<_>>();
///
/// // Compute STFT
/// let (frequencies, times, result) = stft(
///     &signal,
///     Window::Hann,
///     256,
///     Some(128),
///     None,
///     Some(fs),
///     None,
///     None,
/// ).unwrap();
///
/// // Check dimensions
/// assert_eq!(frequencies.len(), result.shape()[0]);
/// assert_eq!(times.len(), result.shape()[1]);
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn stft<T>(
    x: &[T],
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    fs: Option<f64>,
    detrend: Option<bool>,
    boundary: Option<&str>,
) -> FFTResult<(Vec<f64>, Vec<f64>, ndarray::Array2<num_complex::Complex64>)>
where
    T: num_traits::NumCast + Copy + std::fmt::Debug,
{
    spectrogram::stft(
        x,
        window,
        nperseg,
        noverlap,
        nfft,
        fs,
        detrend,
        Some(true),
        boundary,
    )
}

/// Performs the Hilbert transform.
///
/// The Hilbert transform finds the analytical signal, which can be used to
/// determine instantaneous amplitude and frequency. It is defined by convolving
/// the signal with 1/(πt).
///
/// # Arguments
///
/// * `x` - Input signal (real-valued array)
///
/// # Returns
///
/// * A complex-valued array containing the analytic signal, where the real part
///   is the original signal and the imaginary part is the Hilbert transform.
///
/// # Errors
///
/// Returns an error if the computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert;
/// use std::f64::consts::PI;
///
/// // Generate a cosine signal
/// let n = 100;
/// let freq = 5.0; // Hz
/// let dt = 0.01;  // 100 Hz sampling
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).cos()).collect();
///
/// // Compute Hilbert transform
/// let analytic_signal = hilbert(&signal).unwrap();
///
/// // For a cosine wave, the analytical signal should have a magnitude of approximately 1
/// let mid_point = n / 2;
/// let magnitude = (analytic_signal[mid_point].re.powi(2) +
///                 analytic_signal[mid_point].im.powi(2)).sqrt();
/// assert!((magnitude - 1.0).abs() < 0.1);
/// ```
///
/// # References
///
/// * Marple, S. L. "Computing the Discrete-Time Analytic Signal via FFT."
///   IEEE Transactions on Signal Processing, Vol. 47, No. 9, 1999.

/// Helper function to try and extract a Complex value
#[allow(dead_code)]
fn try_as_complex<U: 'static + Copy>(val: U) -> Option<num_complex::Complex64> {
    use num_complex::Complex64;
    use std::any::Any;

    // Try to use runtime type checking with Any for _complex types
    if let Some(_complex) = (&val as &dyn Any).downcast_ref::<Complex64>() {
        return Some(*_complex);
    }

    // Try to handle f32 _complex numbers
    if let Some(complex32) = (&val as &dyn Any).downcast_ref::<num_complex::Complex<f32>>() {
        return Some(Complex64::new(complex32.re as f64, complex32.im as f64));
    }

    None
}

#[allow(dead_code)]
pub fn hilbert<T>(x: &[T]) -> FFTResult<Vec<num_complex::Complex64>>
where
    T: num_traits::NumCast + Copy + std::fmt::Debug + 'static,
{
    use num_complex::Complex64;

    // Input length
    let n = x.len();

    // Convert input to a vector of f64
    let signal: Vec<f64> = x
        .iter()
        .map(|&val| {
            // First, try to cast directly to f64
            if let Some(val_f64) = num_traits::cast::<T, f64>(val) {
                return Ok(val_f64);
            }

            // If direct casting fails, check if it's a Complex value
            // and use just the real part (for doctests which use Complex inputs)
            match try_as_complex(val) {
                Some(c) => Ok(c.re),
                None => Err(FFTError::ValueError(format!(
                    "Could not convert {val:?} to numeric type"
                ))),
            }
        })
        .collect::<FFTResult<Vec<_>>>()?;

    // Compute FFT of the input signal
    let spectrum = fft(&signal, None)?;

    // Create the frequency domain filter for the Hilbert transform
    // For a proper Hilbert transform, we need to:
    // 1. Set the DC component (0 frequency) to 1
    // 2. Double the positive frequencies and multiply by -i
    // 3. Zero out the negative frequencies
    let mut h = vec![Complex64::new(1.0, 0.0); n];

    if n % 2 == 0 {
        // Even length case
        h[0] = Complex64::new(1.0, 0.0); // DC component
        h[n / 2] = Complex64::new(1.0, 0.0); // Nyquist component

        // Positive frequencies (multiply by 2 and by -i)
        h.iter_mut().take(n / 2).skip(1).for_each(|val| {
            *val = Complex64::new(0.0, -2.0); // Equivalent to 2 * (-i)
        });

        // Negative frequencies (set to 0)
        h.iter_mut().skip(n / 2 + 1).for_each(|val| {
            *val = Complex64::new(0.0, 0.0);
        });
    } else {
        // Odd length case
        h[0] = Complex64::new(1.0, 0.0); // DC component

        // Positive frequencies (multiply by 2 and by -i)
        h.iter_mut().take(n.div_ceil(2)).skip(1).for_each(|val| {
            *val = Complex64::new(0.0, -2.0); // Equivalent to 2 * (-i)
        });

        // Negative frequencies (set to 0)
        h.iter_mut().skip(n.div_ceil(2)).for_each(|val| {
            *val = Complex64::new(0.0, 0.0);
        });
    }

    // Apply the filter in frequency domain
    let filtered_spectrum: Vec<Complex64> = spectrum
        .iter()
        .zip(h.iter())
        .map(|(&s, &h)| s * h)
        .collect();

    // Compute inverse FFT to get the analytic signal
    let analytic_signal = ifft(&filtered_spectrum, None)?;

    Ok(analytic_signal)
}

/// Returns the minimum and maximum values for each FFT dimension.
///
/// # Arguments
///
/// * `shape` - The shape of the FFT
///
/// # Returns
///
/// A vector of tuples (min, max) for each dimension of the FFT.
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft_bounds;
///
/// let bounds = fft_bounds(&[4, 4]);
/// assert_eq!(bounds, vec![(-2, 1), (-2, 1)]);
///
/// let bounds = fft_bounds(&[5, 3]);
/// assert_eq!(bounds, vec![(-2, 2), (-1, 1)]);
/// ```
#[must_use]
#[allow(dead_code)]
pub fn fft_bounds(shape: &[usize]) -> Vec<(i32, i32)> {
    shape
        .iter()
        .map(|&n| {
            // Cast with explicit handling for possible truncation/wrapping
            let n_i32 = i32::try_from(n).unwrap_or(i32::MAX);
            let min = -(n_i32 / 2);
            let max = n_i32 - 1 + min;
            (min, max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_bounds() {
        // Test even sizes
        let bounds = fft_bounds(&[4, 4]);
        assert_eq!(bounds, vec![(-2, 1), (-2, 1)]);

        // Test odd sizes
        let bounds = fft_bounds(&[5, 3]);
        assert_eq!(bounds, vec![(-2, 2), (-1, 1)]);

        // Test mixed sizes
        let bounds = fft_bounds(&[6, 7, 8]);
        assert_eq!(bounds, vec![(-3, 2), (-3, 3), (-4, 3)]);
    }

    #[test]
    fn test_hilbert_transform() {
        // Test on a cosine wave instead of sine wave to make the math easier
        let n = 1000;
        let freq = 5.0; // 5 Hz
        let sample_rate = 100.0; // 100 Hz
        let dt = 1.0 / sample_rate;

        // Create a cosine wave
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).cos()).collect();

        // Compute Hilbert transform
        let analytic = hilbert(&signal).unwrap();

        // The Hilbert transform of cos(x) is sin(x)
        // So the analytic signal should be cos(x) + i*sin(x) = e^(ix)
        // Check the envelope (magnitude) which should be approximately 1
        let start_idx = n / 4;
        let end_idx = 3 * n / 4;

        for i in start_idx..end_idx {
            let magnitude = (analytic[i].re.powi(2) + analytic[i].im.powi(2)).sqrt();
            assert_relative_eq!(magnitude, 1.0, epsilon = 0.1);

            // Also check if the phase is advancing correctly
            if i > start_idx {
                let phase_i = analytic[i].im.atan2(analytic[i].re);
                let phase_i_prev = analytic[i - 1].im.atan2(analytic[i - 1].re);

                // Check if phase is advancing in the right direction
                // We need to handle phase wrapping around ±π
                let mut phase_diff = phase_i - phase_i_prev;
                if phase_diff > PI {
                    phase_diff -= 2.0 * PI;
                } else if phase_diff < -PI {
                    phase_diff += 2.0 * PI;
                }

                // For positive frequency, phase should generally advance positively
                assert!(phase_diff > 0.0);
            }
        }
    }
}

// Include ARM-specific FFT tests
#[cfg(test)]
mod arm_fft_test;
