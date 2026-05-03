//! Advanced Strided FFT Operations
//!
//! This module provides optimized FFT operations for arrays with
//! arbitrary memory layouts and striding patterns.
//! Uses OxiFFT as the backend (COOLJAPAN Pure Rust policy).

#[cfg(feature = "oxifft")]
use crate::oxifft_plan_cache;
#[cfg(feature = "oxifft")]
use oxifft::{Complex as OxiComplex, Direction};
use scirs2_core::ndarray::{ArrayBase, Data, Dimension};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;

use crate::error::{FFTError, FFTResult};

/// Execute FFT on strided data with optimal memory access (OxiFFT backend)
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
    let axis_len = input.shape()[axis];

    // Process each lane along the specified axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        #[cfg(feature = "oxifft")]
        {
            let mut input_oxi: Vec<OxiComplex<f64>> = Vec::with_capacity(axis_len);
            for &val in i_lane.iter() {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError("Failed to convert value to f64".to_string())
                })?;
                input_oxi.push(OxiComplex::new(val_f64, 0.0));
            }

            let mut output_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); axis_len];

            oxifft_plan_cache::execute_c2c(&input_oxi, &mut output_oxi, Direction::Forward)?;

            for (i, &val) in output_oxi.iter().enumerate() {
                o_lane[i] = Complex64::new(val.re, val.im);
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            return Err(FFTError::ValueError(
                "No FFT backend available. Enable the 'oxifft' feature.".to_string(),
            ));
        }
    }

    Ok(output)
}

/// Execute FFT on strided data with optimal memory access for complex input (OxiFFT backend)
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
    let axis_len = input.shape()[axis];

    // Process each lane along the specified axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        #[cfg(feature = "oxifft")]
        {
            let input_oxi: Vec<OxiComplex<f64>> = i_lane
                .iter()
                .map(|&val| {
                    let c: Complex64 = val.into();
                    OxiComplex::new(c.re, c.im)
                })
                .collect();

            let mut output_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); axis_len];

            oxifft_plan_cache::execute_c2c(&input_oxi, &mut output_oxi, Direction::Forward)?;

            for (i, &val) in output_oxi.iter().enumerate() {
                o_lane[i] = Complex64::new(val.re, val.im);
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            return Err(FFTError::ValueError(
                "No FFT backend available. Enable the 'oxifft' feature.".to_string(),
            ));
        }
    }

    Ok(output)
}

/// Execute inverse FFT on strided data (OxiFFT backend)
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
    let axis_len = input.shape()[axis];

    // Process each lane along the specified axis
    for (i_lane, mut o_lane) in input
        .lanes(scirs2_core::ndarray::Axis(axis))
        .into_iter()
        .zip(output.lanes_mut(scirs2_core::ndarray::Axis(axis)))
    {
        #[cfg(feature = "oxifft")]
        {
            let input_oxi: Vec<OxiComplex<f64>> = i_lane
                .iter()
                .map(|&val| {
                    let c: Complex64 = val.into();
                    OxiComplex::new(c.re, c.im)
                })
                .collect();

            let mut output_oxi: Vec<OxiComplex<f64>> = vec![OxiComplex::new(0.0, 0.0); axis_len];

            oxifft_plan_cache::execute_c2c(&input_oxi, &mut output_oxi, Direction::Backward)?;

            let scale = 1.0 / (axis_len as f64);
            for (i, &val) in output_oxi.iter().enumerate() {
                o_lane[i] = Complex64::new(val.re * scale, val.im * scale);
            }
        }

        #[cfg(not(feature = "oxifft"))]
        {
            return Err(FFTError::ValueError(
                "No FFT backend available. Enable the 'oxifft' feature.".to_string(),
            ));
        }
    }

    Ok(output)
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
        let result = fft_strided(&input, 0).expect("Operation failed");

        // Shape must match
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
        let result1 = fft_strided(&input, 0).expect("Operation failed");
        assert_eq!(result1.shape(), input.shape());

        // FFT along second axis
        let result2 = fft_strided(&input, 1).expect("Operation failed");
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
        let forward = fft_strided_complex(&input, 0).expect("Operation failed");
        let inverse = ifft_strided(&forward, 0).expect("Operation failed");

        // Check round-trip accuracy
        for i in 0..n {
            assert!((inverse[i] - input[i]).norm() < 1e-10);
        }
    }
}
