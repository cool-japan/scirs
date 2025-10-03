//! Minimal SIMD-accelerated FFT operations stub
//!
//! This module provides minimal stubs for SIMD-accelerated FFT operations.
//! All actual SIMD operations are delegated to scirs2-core when available.

use crate::error::FFTResult;
use crate::fft;
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use scirs2_core::simd_ops::PlatformCapabilities;
use std::fmt::Debug;

/// Normalization mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    None,
    Backward,
    Ortho,
    Forward,
}

/// Check if SIMD support is available
#[allow(dead_code)]
pub fn simd_support_available() -> bool {
    let caps = PlatformCapabilities::detect();
    caps.simd_available
}

/// Apply SIMD normalization (stub - not used in current implementation)
#[allow(dead_code)]
pub fn apply_simd_normalization(data: &mut [Complex64], scale: f64) {
    for c in data.iter_mut() {
        *c *= scale;
    }
}

/// SIMD-accelerated 1D FFT
#[allow(dead_code)]
pub fn fft_simd<T>(x: &[T], _norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    fft::fft(x, None)
}

/// SIMD-accelerated 1D inverse FFT
#[allow(dead_code)]
pub fn ifft_simd<T>(x: &[T], _norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    fft::ifft(x, None)
}

/// SIMD-accelerated 2D FFT
#[allow(dead_code)]
pub fn fft2_simd<T>(
    x: &[T],
    shape: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // If no shape is provided, try to infer a square shape
    let (n_rows, n_cols) = if let Some(s) = shape {
        s
    } else {
        let len = x.len();
        let size = (len as f64).sqrt() as usize;
        if size * size != len {
            return Err(crate::error::FFTError::ValueError(
                "Cannot infer 2D shape from slice length".to_string(),
            ));
        }
        (size, size)
    };

    // Check that the slice has the right number of elements
    if x.len() != n_rows * n_cols {
        return Err(crate::error::FFTError::ValueError(format!(
            "Shape ({}, {}) requires {} elements, but slice has {}",
            n_rows,
            n_cols,
            n_rows * n_cols,
            x.len()
        )));
    }

    // Convert slice to 2D array
    let mut values = Vec::with_capacity(n_rows * n_cols);
    for &val in x.iter() {
        values.push(val);
    }
    let arr = Array2::from_shape_vec((n_rows, n_cols), values)
        .map_err(|e| crate::error::FFTError::DimensionError(e.to_string()))?;

    // Use the regular fft2 function
    crate::fft::fft2(&arr, None, None, norm)
}

/// SIMD-accelerated 2D inverse FFT
#[allow(dead_code)]
pub fn ifft2_simd<T>(
    _x: &[T],
    _shape: Option<(usize, usize)>,
    _norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "2D inverse FFT from slice not yet implemented".to_string(),
    ))
}

/// SIMD-accelerated N-dimensional FFT
#[allow(dead_code)]
pub fn fftn_simd<T>(
    x: &[T],
    shape: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Shape is required for N-dimensional FFT from slice
    let shape = shape.ok_or_else(|| {
        crate::error::FFTError::ValueError(
            "Shape is required for N-dimensional FFT from slice".to_string(),
        )
    })?;

    // Calculate total number of elements
    let total_elements: usize = shape.iter().product();

    // Check that the slice has the right number of elements
    if x.len() != total_elements {
        return Err(crate::error::FFTError::ValueError(format!(
            "Shape {:?} requires {} elements, but slice has {}",
            shape,
            total_elements,
            x.len()
        )));
    }

    // Convert slice to N-dimensional array
    let mut values = Vec::with_capacity(total_elements);
    for &val in x.iter() {
        values.push(val);
    }
    let arr = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|e| crate::error::FFTError::DimensionError(e.to_string()))?;

    // Use the regular fftn function
    crate::fft::fftn(&arr, None, axes.map(|a| a.to_vec()), norm, None, None)
}

/// SIMD-accelerated N-dimensional inverse FFT
#[allow(dead_code)]
pub fn ifftn_simd<T>(
    _x: &[T],
    _shape: Option<&[usize]>,
    _axes: Option<&[usize]>,
    _norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // For now, just create a simple error
    Err(crate::error::FFTError::NotImplementedError(
        "N-dimensional inverse FFT from slice not yet implemented".to_string(),
    ))
}

/// Adaptive FFT
#[allow(dead_code)]
pub fn fft_adaptive<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    fft_simd(x, norm)
}

/// Adaptive inverse FFT
#[allow(dead_code)]
pub fn ifft_adaptive<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    ifft_simd(x, norm)
}

/// Adaptive 2D FFT
#[allow(dead_code)]
pub fn fft2_adaptive<T>(
    _x: &[T],
    shape: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    fft2_simd(_x, shape, norm)
}

/// Adaptive 2D inverse FFT
#[allow(dead_code)]
pub fn ifft2_adaptive<T>(
    _x: &[T],
    shape: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    ifft2_simd(_x, shape, norm)
}

/// Adaptive N-dimensional FFT
#[allow(dead_code)]
pub fn fftn_adaptive<T>(
    _x: &[T],
    shape: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    fftn_simd(_x, shape, axes, norm)
}

/// Adaptive N-dimensional inverse FFT
#[allow(dead_code)]
pub fn ifftn_adaptive<T>(
    _x: &[T],
    shape: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    ifftn_simd(_x, shape, axes, norm)
}
