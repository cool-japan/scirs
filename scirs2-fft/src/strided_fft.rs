//! Advanced Strided FFT Operations
//!
//! This module provides optimized FFT operations for arrays with
//! arbitrary memory layouts and striding patterns.

use rustfft::FftPlanner;
use scirs2_core::ndarray::{ArrayBase, Data, Dimension};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::sync::Arc;

use crate::error::{FFTError, FFTResult};
use crate::plan_cache::get_global_cache;

/// Execute FFT on strided data with optimal memory access
#[allow(dead_code)]
pub fn fft_strided<S, D>(
    input: &ArrayBase<S, D>,
    axis: usize,
) -> FFTResult<scirs2_core::ndarray::Array<Complex64, D>>
where
    S: Data,
    D: Dimension,
    S::Elem: NumCast + Copy,
{
    // Validate axis
    if axis >= input.ndim() {
        return Err(FFTError::ValueError(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    // Create output array with same shape
    let mut output = scirs2_core::ndarray::Array::zeros(input.raw_dim());

    // Get FFT plan from cache
    let axis_len = input.shape()[axis];
    let mut planner = FftPlanner::new();
    let fft_plan = get_global_cache().get_or_create_plan(axis_len, true, &mut planner);

    // Process data along the specified axis
    process_strided_fft(input, &mut output, axis, fft_plan)?;

    Ok(output)
}

/// Process data with arbitrary striding
#[allow(dead_code)]
fn process_strided_fft<S, D>(
    input: &ArrayBase<S, D>,
    output: &mut scirs2_core::ndarray::Array<Complex64, D>,
    axis: usize,
    fft_plan: Arc<dyn rustfft::Fft<f64>>,
) -> FFTResult<()>
where
    S: Data,
    D: Dimension,
    S::Elem: NumCast + Copy,
{
    let axis_len = input.shape()[axis];

    // Create temporary buffer for FFT input/output
    let mut buffer = vec![Complex64::new(0.0, 0.0); axis_len];

    // Process each lane along the given axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        // Copy data to input buffer with proper conversion
        for (i, &val) in i_lane.iter().enumerate() {
            let val_f64 = NumCast::from(val).ok_or_else(|| {
                FFTError::ValueError(format!("Failed to convert value at index {i} to f64"))
            })?;
            buffer[i] = Complex64::new(val_f64, 0.0);
        }

        // Perform FFT (in-place)
        fft_plan.process(&mut buffer);

        // Copy results back to output
        for (i, dst) in o_lane.iter_mut().enumerate() {
            *dst = buffer[i];
        }
    }

    Ok(())
}

/// Execute FFT on strided data with optimal memory access for complex input
#[allow(dead_code)]
pub fn fft_strided_complex<S, D>(
    input: &ArrayBase<S, D>,
    axis: usize,
) -> FFTResult<scirs2_core::ndarray::Array<Complex64, D>>
where
    S: Data,
    D: Dimension,
    S::Elem: Into<Complex64> + Copy,
{
    // Validate axis
    if axis >= input.ndim() {
        return Err(FFTError::ValueError(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    // Create output array with same shape
    let mut output = scirs2_core::ndarray::Array::zeros(input.raw_dim());

    // Get FFT plan from cache
    let axis_len = input.shape()[axis];
    let mut planner = FftPlanner::new();
    let fft_plan = get_global_cache().get_or_create_plan(axis_len, true, &mut planner);

    // Process data along the specified axis
    process_strided_complex_fft(input, &mut output, axis, fft_plan)?;

    Ok(output)
}

/// Process complex data with arbitrary striding
#[allow(dead_code)]
fn process_strided_complex_fft<S, D>(
    input: &ArrayBase<S, D>,
    output: &mut scirs2_core::ndarray::Array<Complex64, D>,
    axis: usize,
    fft_plan: Arc<dyn rustfft::Fft<f64>>,
) -> FFTResult<()>
where
    S: Data,
    D: Dimension,
    S::Elem: Into<Complex64> + Copy,
{
    let axis_len = input.shape()[axis];

    // Create temporary buffer for FFT input/output
    let mut buffer = vec![Complex64::new(0.0, 0.0); axis_len];

    // Process each lane along the given axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        // Copy data to input buffer with proper conversion
        for (i, &val) in i_lane.iter().enumerate() {
            buffer[i] = val.into();
        }

        // Perform FFT (in-place)
        fft_plan.process(&mut buffer);

        // Copy results back to output
        for (i, dst) in o_lane.iter_mut().enumerate() {
            *dst = buffer[i];
        }
    }

    Ok(())
}

/// Execute inverse FFT on strided data
#[allow(dead_code)]
pub fn ifft_strided<S, D>(
    input: &ArrayBase<S, D>,
    axis: usize,
) -> FFTResult<scirs2_core::ndarray::Array<Complex64, D>>
where
    S: Data,
    D: Dimension,
    S::Elem: Into<Complex64> + Copy,
{
    // Validate axis
    if axis >= input.ndim() {
        return Err(FFTError::ValueError(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    // Create output array with same shape
    let mut output = scirs2_core::ndarray::Array::zeros(input.raw_dim());

    // Get inverse FFT plan from cache
    let axis_len = input.shape()[axis];
    let mut planner = FftPlanner::new();
    let ifft_plan = get_global_cache().get_or_create_plan(axis_len, false, &mut planner);

    // Process data along the specified axis
    process_strided_inverse_fft(input, &mut output, axis, ifft_plan)?;

    // Apply normalization
    let scale = 1.0 / (axis_len as f64);
    output.mapv_inplace(|val| val * scale);

    Ok(output)
}

/// Process data with arbitrary striding for inverse FFT
#[allow(dead_code)]
fn process_strided_inverse_fft<S, D>(
    input: &ArrayBase<S, D>,
    output: &mut scirs2_core::ndarray::Array<Complex64, D>,
    axis: usize,
    ifft_plan: Arc<dyn rustfft::Fft<f64>>,
) -> FFTResult<()>
where
    S: Data,
    D: Dimension,
    S::Elem: Into<Complex64> + Copy,
{
    let axis_len = input.shape()[axis];

    // Create temporary buffer for FFT input/output
    let mut buffer = vec![Complex64::new(0.0, 0.0); axis_len];

    // Process each lane along the given axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        // Copy data to input buffer with proper conversion
        for (i, &val) in i_lane.iter().enumerate() {
            buffer[i] = val.into();
        }

        // Perform inverse FFT (in-place)
        ifft_plan.process(&mut buffer);

        // Copy results back to output
        for (i, dst) in o_lane.iter_mut().enumerate() {
            *dst = buffer[i];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_fft_strided_1d() {
        // Create a test signal
        let n = 8;
        let mut input = scirs2_core::ndarray::Array1::zeros(n);
        for i in 0..n {
            input[i] = i as f64;
        }

        // Compute FFT using strided implementation
        let result = fft_strided(&input, 0).unwrap();

        // Compare with expected FFT result
        // (We would compare with the standard FFT implementation)
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_fft_strided_2d() {
        // Create a 2D test array
        let mut input = Array2::zeros((4, 6));
        for i in 0..4 {
            for j in 0..6 {
                input[[i, j]] = (i * 10 + j) as f64;
            }
        }

        // FFT along first axis
        let result1 = fft_strided(&input, 0).unwrap();
        assert_eq!(result1.shape(), input.shape());

        // FFT along second axis
        let result2 = fft_strided(&input, 1).unwrap();
        assert_eq!(result2.shape(), input.shape());
    }

    #[test]
    fn test_ifft_strided() {
        // Create a complex test signal
        let n = 8;
        let mut input = scirs2_core::ndarray::Array1::zeros(n);
        for i in 0..n {
            input[i] = Complex64::new(i as f64, (i * 2) as f64);
        }

        // Forward and inverse FFT should give back the input
        let forward = fft_strided_complex(&input, 0).unwrap();
        let inverse = ifft_strided(&forward, 0).unwrap();

        // Check round-trip accuracy
        for i in 0..n {
            assert!((inverse[i] - input[i]).norm() < 1e-10);
        }
    }
}
