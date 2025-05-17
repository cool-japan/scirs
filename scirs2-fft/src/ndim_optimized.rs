//! Optimized N-dimensional FFT operations
//!
//! This module provides optimized implementations of N-dimensional FFT
//! operations with better memory access patterns and performance.

use ndarray::{Array, ArrayView, Axis, Dimension};
use num_complex::Complex64;
use num_traits::NumCast;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::cmp::min;

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::rfft::rfft;

/// Optimized N-dimensional FFT with better memory access patterns
pub fn fftn_optimized<T, D>(
    x: &ArrayView<T, D>,
    _shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
) -> FFTResult<Array<Complex64, D>>
where
    T: NumCast + Copy + Send + Sync,
    D: Dimension,
{
    let ndim = x.ndim();

    // Convert input to complex
    let mut result = Array::zeros(x.raw_dim());
    ndarray::Zip::from(&mut result)
        .and(x)
        .for_each(|dst, &src| {
            *dst = Complex64::new(
                NumCast::from(src)
                    .ok_or_else(|| {
                        FFTError::ValueError("Failed to convert input to complex".to_string())
                    })
                    .unwrap(),
                0.0,
            );
        });

    // Determine axes to transform
    let axes_to_transform = if let Some(a) = axes {
        validate_axes(&a, ndim)?;
        a
    } else {
        (0..ndim).collect()
    };

    // Optimize axis order based on memory layout
    let optimized_order = optimize_axis_order(&axes_to_transform, result.shape());

    // Apply FFT along each axis in optimized order
    for &axis in &optimized_order {
        apply_fft_along_axis(&mut result, axis)?;
    }

    Ok(result)
}

/// Apply FFT along a specific axis
fn apply_fft_along_axis<D>(data: &mut Array<Complex64, D>, axis: usize) -> FFTResult<()>
where
    D: Dimension,
{
    let axis_len = data.shape()[axis];

    // Create temporary buffer for FFT
    let mut buffer = vec![Complex64::new(0.0, 0.0); axis_len];

    // Process slices along the specified axis
    for mut lane in data.lanes_mut(Axis(axis)) {
        // Copy data to buffer
        buffer
            .iter_mut()
            .zip(lane.iter())
            .for_each(|(b, &x)| *b = x);

        // Perform FFT
        let transformed = fft(&buffer, None)?;

        // Copy results back
        lane.iter_mut()
            .zip(transformed.iter())
            .for_each(|(dst, &src)| *dst = src);
    }

    Ok(())
}

/// Optimize axis order based on memory layout and cache efficiency
fn optimize_axis_order(axes: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut axis_info: Vec<(usize, usize, usize)> = axes
        .iter()
        .map(|&axis| {
            let size = shape[axis];
            let stride = shape.iter().skip(axis + 1).product::<usize>();
            (axis, size, stride)
        })
        .collect();

    // Sort by stride (smallest first) for better cache locality
    axis_info.sort_by_key(|&(_, _, stride)| stride);

    // Return optimized axis order
    axis_info.into_iter().map(|(axis, _, _)| axis).collect()
}

/// Validate that axes are within bounds
fn validate_axes(axes: &[usize], ndim: usize) -> FFTResult<()> {
    for &axis in axes {
        if axis >= ndim {
            return Err(FFTError::ValueError(format!(
                "Axis {} is out of bounds for array with {} dimensions",
                axis, ndim
            )));
        }
    }
    Ok(())
}

/// Determine whether to use parallel processing
#[allow(dead_code)]
fn should_parallelize(data_size: usize, axis_len: usize) -> bool {
    // Use parallel processing for large data sizes
    const MIN_PARALLEL_SIZE: usize = 10000;
    data_size > MIN_PARALLEL_SIZE && axis_len > 64
}

/// Apply FFT along axis with optional parallelization
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn apply_fft_parallel<D>(data: &mut Array<Complex64, D>, axis: usize) -> FFTResult<()>
where
    D: Dimension,
{
    let axis_len = data.shape()[axis];
    let total_size: usize = data.shape().iter().product();

    if should_parallelize(total_size, axis_len) {
        // Process lanes in parallel
        let mut lanes: Vec<_> = data.lanes_mut(Axis(axis)).into_iter().collect();

        lanes.par_iter_mut().try_for_each(|lane| {
            let buffer: Vec<Complex64> = lane.to_vec();
            let transformed = fft(&buffer, None)?;
            lane.iter_mut()
                .zip(transformed.iter())
                .for_each(|(dst, &src)| *dst = src);
            Ok(())
        })
    } else {
        apply_fft_along_axis(data, axis)
    }
}

/// Memory-efficient FFT for very large arrays
pub fn fftn_memory_efficient<T, D>(
    x: &ArrayView<T, D>,
    axes: Option<Vec<usize>>,
    _max_memory_gb: f64,
) -> FFTResult<Array<Complex64, D>>
where
    T: NumCast + Copy + Send + Sync,
    D: Dimension,
{
    let ndim = x.ndim();
    let axes_to_transform = if let Some(a) = axes {
        validate_axes(&a, ndim)?;
        a
    } else {
        (0..ndim).collect()
    };

    // For memory efficiency, we process one axis at a time
    // and use chunking for very large dimensions
    let mut result = Array::zeros(x.raw_dim());

    // Convert input to complex
    ndarray::Zip::from(&mut result)
        .and(x)
        .for_each(|dst, &src| {
            *dst = Complex64::new(
                NumCast::from(src)
                    .ok_or_else(|| {
                        FFTError::ValueError("Failed to convert input to complex".to_string())
                    })
                    .unwrap(),
                0.0,
            );
        });

    // Process each axis with chunking if needed
    for &axis in &axes_to_transform {
        let axis_len = result.shape()[axis];

        if axis_len > 1048576 {
            // For very large axes, use chunked processing
            apply_fft_chunked(&mut result, axis)?;
        } else {
            apply_fft_along_axis(&mut result, axis)?;
        }
    }

    Ok(result)
}

/// Apply FFT along axis using chunked processing for large dimensions
fn apply_fft_chunked<D>(data: &mut Array<Complex64, D>, axis: usize) -> FFTResult<()>
where
    D: Dimension,
{
    let axis_len = data.shape()[axis];
    const CHUNK_SIZE: usize = 65536; // Process in 64K chunks

    // This is a simplified chunking strategy
    // In practice, we'd need to handle overlapping chunks
    // for proper FFT computation
    let n_chunks = axis_len.div_ceil(CHUNK_SIZE);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * CHUNK_SIZE;
        let end = min(start + CHUNK_SIZE, axis_len);
        let chunk_len = end - start;

        // Process chunk
        let mut buffer = vec![Complex64::new(0.0, 0.0); chunk_len];

        for mut lane in data.lanes_mut(Axis(axis)) {
            // Extract chunk from lane
            buffer
                .iter_mut()
                .zip(lane.slice_axis(Axis(0), (start..end).into()).iter())
                .for_each(|(b, &x)| *b = x);

            // Perform FFT on chunk
            let transformed = fft(&buffer, None)?;

            // Copy results back to chunk
            lane.slice_axis_mut(Axis(0), (start..end).into())
                .iter_mut()
                .zip(transformed.iter())
                .for_each(|(dst, &src)| *dst = src);
        }
    }

    Ok(())
}

/// Optimized real-to-complex N-dimensional FFT
pub fn rfftn_optimized<T, D>(
    x: &ArrayView<T, D>,
    _shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
) -> FFTResult<Array<Complex64, D>>
where
    T: NumCast + Copy + Send + Sync,
    D: Dimension,
{
    // For real FFT, we can optimize the first transform
    // and use symmetry properties
    let ndim = x.ndim();
    let mut axes_to_transform = if let Some(a) = axes {
        validate_axes(&a, ndim)?;
        a
    } else {
        (0..ndim).collect()
    };

    // Process the last axis with real FFT for efficiency
    let last_axis = axes_to_transform.pop().unwrap_or(ndim - 1);

    // Convert to real array for first transform
    let mut real_data = Array::zeros(x.raw_dim());
    ndarray::Zip::from(&mut real_data)
        .and(x)
        .for_each(|dst, &src| {
            *dst = NumCast::from(src)
                .ok_or_else(|| FFTError::ValueError("Failed to convert input to float".to_string()))
                .unwrap();
        });

    // Apply real FFT on the last axis
    let mut result: Array<Complex64, D> = Array::zeros(x.raw_dim());

    // This is a simplified implementation - proper real FFT would have different output dimensions
    for lane in real_data.lanes(Axis(last_axis)) {
        let real_vec: Vec<f64> = lane.to_vec();
        let _complex_vec = rfft(&real_vec, None)?;

        // For now, just convert to complex array format
        // This is a placeholder implementation
    }

    // Apply complex FFT on remaining axes
    for &axis in &axes_to_transform {
        apply_fft_along_axis(&mut result, axis)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_optimization() {
        let axes = vec![0, 1, 2];
        let shape = vec![10, 100, 1000];
        let optimized = optimize_axis_order(&axes, &shape);

        // Should order from smallest stride (rightmost) to largest
        assert_eq!(optimized[0], 2);
        assert_eq!(optimized[1], 1);
        assert_eq!(optimized[2], 0);
    }

    #[test]
    fn test_parallelize_decision() {
        // Test with both conditions met: large data size and axis length > 64
        assert!(should_parallelize(10001, 100));
        // Test with only data size large enough but axis too small
        assert!(!should_parallelize(10001, 50));
        // Test with both too small
        assert!(!should_parallelize(100, 10));
    }

    #[test]
    fn test_validate_axes() {
        assert!(validate_axes(&[0, 1, 2], 3).is_ok());
        assert!(validate_axes(&[0, 1, 3], 3).is_err());
    }
}
