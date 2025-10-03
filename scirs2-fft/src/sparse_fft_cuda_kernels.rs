//! GPU kernel stub for sparse FFT algorithms
//!
//! This module provides a compatibility layer for GPU-accelerated sparse FFT.
//! All GPU operations must be migrated to use scirs2-core::gpu module.
//! Direct CUDA implementations are FORBIDDEN by the strict acceleration policy.

use crate::error::{FFTError, FFTResult};
use crate::gpu_kernel_stub::MIGRATION_MESSAGE;
use crate::sparse_fft::{SparseFFTAlgorithm, SparseFFTResult, WindowFunction};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

/// CUDA window kernel stub
pub struct CUDAWindowKernel;

impl CUDAWindowKernel {
    pub fn new(_window: WindowFunction) -> Self {
        Self
    }

    pub fn apply(&self, _data: &mut [Complex64]) -> FFTResult<()> {
        Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
    }
}

/// CUDA sublinear sparse FFT kernel stub
pub struct CUDASublinearSparseFFTKernel;

impl CUDASublinearSparseFFTKernel {
    pub fn new(_algorithm: SparseFFTAlgorithm) -> Self {
        Self
    }

    pub fn execute<T>(&self, _input: &[T], _k: usize) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug,
    {
        Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
    }
}

/// Execute CUDA sublinear sparse FFT (stub)
#[allow(dead_code)]
pub fn execute_cuda_sublinear_sparse_fft<T>(
    _input: &[T],
    _k: usize,
    _algorithm: SparseFFTAlgorithm,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug,
{
    Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
}

/// CUDA compressed sensing sparse FFT kernel stub
pub struct CUDACompressedSensingSparseFFTKernel;

impl CUDACompressedSensingSparseFFTKernel {
    pub fn new() -> Self {
        Self
    }

    pub fn execute<T>(&self, _input: &[T], _k: usize) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug,
    {
        Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
    }
}

impl Default for CUDACompressedSensingSparseFFTKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute CUDA compressed sensing sparse FFT (stub)
#[allow(dead_code)]
pub fn execute_cuda_compressed_sensing_sparse_fft<T>(
    _input: &[T],
    _k: usize,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug,
{
    Err(FFTError::NotImplementedError(MIGRATION_MESSAGE.to_string()))
}
