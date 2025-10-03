//! Stub implementations for all GPU kernel modules
//!
//! This file contains stub implementations for all GPU kernel modules.
//! All actual GPU operations must be migrated to use scirs2-core::gpu module.

use crate::error::{FFTError, FFTResult};
use crate::gpu_kernel_stub::MIGRATION_MESSAGE;
use crate::sparse_fft::SparseFFTResult;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

// ============ sparse_fft_cuda_kernels_frequency_pruning.rs stubs ============

/// CUDA frequency pruning sparse FFT kernel stub
pub struct CUDAFrequencyPruningSparseFFTKernel;

impl CUDAFrequencyPruningSparseFFTKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CUDAFrequencyPruningSparseFFTKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute CUDA frequency pruning sparse FFT (stub)
#[allow(dead_code)]
pub fn execute_cuda_frequency_pruning_sparse_fft<T>(
    _input: &[T],
    _k: usize,
    _threshold: f64,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug,
{
    Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
}

// ============ sparse_fft_cuda_kernels_iterative.rs stubs ============

/// CUDA iterative sparse FFT kernel stub
pub struct CUDAIterativeSparseFFTKernel;

impl CUDAIterativeSparseFFTKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CUDAIterativeSparseFFTKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute CUDA iterative sparse FFT (stub)
#[allow(dead_code)]
pub fn execute_cuda_iterative_sparse_fft<T>(
    _input: &[T],
    _k: usize,
    _max_iterations: usize,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug,
{
    Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
}

// ============ sparse_fft_cuda_kernels_spectral_flatness.rs stubs ============

/// CUDA spectral flatness sparse FFT kernel stub
pub struct CUDASpectralFlatnessSparseFFTKernel;

impl CUDASpectralFlatnessSparseFFTKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CUDASpectralFlatnessSparseFFTKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute CUDA spectral flatness sparse FFT (stub)
#[allow(dead_code)]
pub fn execute_cuda_spectral_flatness_sparse_fft<T>(
    _input: &[T],
    _k: usize,
    _flatness_threshold: f64,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug,
{
    Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
}
