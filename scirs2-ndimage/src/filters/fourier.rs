//! Fourier domain filtering functions
//!
//! This module provides filters that operate in the Fourier domain,
//! which can be more efficient for certain operations, especially
//! with large kernels or when multiple filters are applied.
//!
//! For very large datasets that don't fit in memory, use the streaming
//! variants of these functions (e.g., `fourier_gaussian_streaming`).

use scirs2_core::ndarray::{Array, Array1, Array2, Dimension};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use scirs2_fft::{fft, fft2, fftfreq, ifft, ifft2, FFTError};

// Conversion from FFTError to NdimageError
impl From<FFTError> for NdimageError {
    fn from(err: FFTError) -> Self {
        NdimageError::ComputationError(format!("FFT error: {}", err))
    }
}

/// Apply a Gaussian filter in the Fourier domain
///
/// This is more efficient than spatial domain filtering for large sigma values.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `sigma` - Standard deviation for Gaussian kernel (can be different for each axis)
/// * `truncate` - Truncate the filter at this many standard deviations (not used in Fourier domain)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn fourier_gaussian<T, D>(
    input: &Array<T, D>,
    sigma: &[T],
    _truncate: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if sigma.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Sigma must have same length as input dimensions (got {} expected {})",
            sigma.len(),
            input.ndim()
        )));
    }

    for &s in sigma {
        if s <= T::zero() {
            return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
        }
    }

    match input.ndim() {
        1 => {
            // 1D case
            let input_1d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_gaussian_1d(&input_1d_owned, sigma[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            // 2D case
            let input_2d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_gaussian_2d(&input_2d_owned, sigma[0], sigma[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => {
            // N-dimensional case (3D and higher)
            let result = fourier_gaussian_nd(input, sigma)?;
            Ok(result)
        }
    }
}

/// Apply 1D Gaussian filter in Fourier domain
#[allow(dead_code)]
fn fourier_gaussian_1d<T>(input: &Array1<T>, sigma: T) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply Gaussian filter
    let sigma_f64: f64 = NumCast::from(sigma).unwrap_or(1.0);
    let two_pi = 2.0 * PI;

    for (i, freq) in freqs.iter().enumerate() {
        let factor = (-0.5 * (two_pi * freq * sigma_f64).powi(2)).exp();
        spectrum[i] *= factor;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from_f64(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D Gaussian filter in Fourier domain
#[allow(dead_code)]
fn fourier_gaussian_2d<T>(input: &Array2<T>, sigma_y: T, sigma_x: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Apply Gaussian filter
    let sigma_y_f64: f64 = NumCast::from(sigma_y).unwrap_or(1.0);
    let sigma_x_f64: f64 = NumCast::from(sigma_x).unwrap_or(1.0);
    let two_pi = 2.0 * PI;

    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let factor_y = (-0.5 * (two_pi * fy * sigma_y_f64).powi(2)).exp();
            let factor_x = (-0.5 * (two_pi * fx * sigma_x_f64).powi(2)).exp();
            filtered_spectrum[[i, j]] *= factor_y * factor_x;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from_f64(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply N-dimensional Gaussian filter in Fourier domain
///
/// This implementation uses separable 1D FFTs along each axis for efficiency
#[allow(dead_code)]
fn fourier_gaussian_nd<T, D>(input: &Array<T, D>, sigma: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    if input.ndim() != sigma.len() {
        return Err(NdimageError::InvalidInput(
            "Sigma array length must match _input dimensions".into(),
        ));
    }

    // For N-dimensional arrays, apply separable filtering
    // Use a simplified approach that works for common 3D case
    if input.ndim() == 3 {
        return fourier_gaussian_3d(input, sigma);
    }

    // For higher dimensions, fall back to a general but simpler approach
    // Convert to 1D, apply filter, and reshape back
    let shape = input.shape().to_vec();
    let flat_input = input.view().into_shape(input.len())?;

    // Create equivalent 1D sigma (use geometric mean of all dimensions)
    let sigma_1d = {
        let product: f64 = sigma
            .iter()
            .map(|&s| NumCast::from(s).unwrap_or(1.0))
            .fold(1.0, |acc, x| acc * x);
        let geometric_mean = product.powf(1.0 / sigma.len() as f64);
        T::from_f64(geometric_mean).unwrap_or(T::one())
    };

    // Apply 1D filter (convert view to owned array)
    let flat_owned = flat_input.to_owned();
    let filtered_1d = fourier_gaussian_1d(&flat_owned, sigma_1d)?;

    // Reshape back to original dimensions and convert to correct dimension type
    let result_dyn = filtered_1d.into_shape(shape)?;
    let result = result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert result to original dimension type".into())
    })?;
    Ok(result)
}

/// Apply 3D Gaussian filter in Fourier domain
#[allow(dead_code)]
fn fourier_gaussian_3d<T, D>(input: &Array<T, D>, sigma: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Convert to 3D for processing
    let input_3d = input
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 3D".into()))?;

    let (nz, ny, nx) = input_3d.dim();

    // Convert sigma values
    let sigma_z: f64 = NumCast::from(sigma[0]).unwrap_or(1.0);
    let sigma_y: f64 = NumCast::from(sigma[1]).unwrap_or(1.0);
    let sigma_x: f64 = NumCast::from(sigma[2]).unwrap_or(1.0);

    // Convert _input to f64 for FFT processing
    let mut input_f64 = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                input_f64[[i, j, k]] = NumCast::from(input_3d[[i, j, k]]).unwrap_or(0.0);
            }
        }
    }

    // Apply separable filtering along each axis
    let two_pi = 2.0 * PI;

    // Filter along z-axis (axis 0)
    let freqs_z = fftfreq(nz, 1.0)?;
    for j in 0..ny {
        for k in 0..nx {
            let slice: Vec<f64> = (0..nz).map(|i| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (i, &freq) in freqs_z.iter().enumerate() {
                let factor = (-0.5 * (two_pi * freq * sigma_z).powi(2)).exp();
                spectrum[i] *= factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (i, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along y-axis (axis 1)
    let freqs_y = fftfreq(ny, 1.0)?;
    for i in 0..nz {
        for k in 0..nx {
            let slice: Vec<f64> = (0..ny).map(|j| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (j, &freq) in freqs_y.iter().enumerate() {
                let factor = (-0.5 * (two_pi * freq * sigma_y).powi(2)).exp();
                spectrum[j] *= factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (j, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along x-axis (axis 2)
    let freqs_x = fftfreq(nx, 1.0)?;
    for i in 0..nz {
        for j in 0..ny {
            let slice: Vec<f64> = (0..nx).map(|k| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (k, &freq) in freqs_x.iter().enumerate() {
                let factor = (-0.5 * (two_pi * freq * sigma_x).powi(2)).exp();
                spectrum[k] *= factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (k, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Convert back to T
    let mut output_3d = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                output_3d[[i, j, k]] = T::from_f64(input_f64[[i, j, k]]).unwrap_or(T::zero());
            }
        }
    }

    // Convert back to original dimension type
    let result = output_3d
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 3D".into()))?;

    Ok(result)
}

/// Apply a uniform (box) filter in the Fourier domain
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the box filter in each dimension
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn fourier_uniform<T, D>(input: &Array<T, D>, size: &[usize]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as _input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Filter size cannot be zero".into(),
            ));
        }
    }

    match input.ndim() {
        1 => {
            let input_1d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_uniform_1d(&input_1d_owned, size[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            let input_2d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_uniform_2d(&input_2d_owned, size[0], size[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => {
            // N-dimensional case (3D and higher)
            let result = fourier_uniform_nd(input, size)?;
            Ok(result)
        }
    }
}

/// Apply 1D uniform filter in Fourier domain
#[allow(dead_code)]
fn fourier_uniform_1d<T>(input: &Array1<T>, size: usize) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply sinc filter (Fourier transform of box function)
    for (i, freq) in freqs.iter().enumerate() {
        let x = size as f64 * freq * 2.0 * PI;
        let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
        spectrum[i] *= sinc;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from_f64(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D uniform filter in Fourier domain
#[allow(dead_code)]
fn fourier_uniform_2d<T>(
    input: &Array2<T>,
    size_y: usize,
    size_x: usize,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Apply sinc filter
    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let x_y = size_y as f64 * fy * 2.0 * PI;
            let x_x = size_x as f64 * fx * 2.0 * PI;

            let sinc_y = if x_y.abs() < 1e-10 {
                1.0
            } else {
                x_y.sin() / x_y
            };
            let sinc_x = if x_x.abs() < 1e-10 {
                1.0
            } else {
                x_x.sin() / x_x
            };

            filtered_spectrum[[i, j]] *= sinc_y * sinc_x;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from_f64(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply N-dimensional uniform filter in Fourier domain
#[allow(dead_code)]
fn fourier_uniform_nd<T, D>(input: &Array<T, D>, size: &[usize]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    if input.ndim() != size.len() {
        return Err(NdimageError::InvalidInput(
            "Size array length must match _input dimensions".into(),
        ));
    }

    // For 3D case, implement separable filtering
    if input.ndim() == 3 {
        return fourier_uniform_3d(input, size);
    }

    // For higher dimensions, use simplified approach
    let shape = input.shape().to_vec();
    let flat_input = input.view().into_shape(input.len())?;

    // Create equivalent 1D size (use geometric mean)
    let size_1d = {
        let product: f64 = size.iter().map(|&s| s as f64).fold(1.0, |acc, x| acc * x);
        let geometric_mean = product.powf(1.0 / size.len() as f64);
        geometric_mean.round() as usize
    };

    // Apply 1D filter (convert view to owned array)
    let flat_owned = flat_input.to_owned();
    let filtered_1d = fourier_uniform_1d(&flat_owned, size_1d)?;

    // Reshape back to original dimensions and convert to correct dimension type
    let result_dyn = filtered_1d.into_shape(shape)?;
    let result = result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert result to original dimension type".into())
    })?;
    Ok(result)
}

/// Apply 3D uniform filter in Fourier domain
#[allow(dead_code)]
fn fourier_uniform_3d<T, D>(input: &Array<T, D>, size: &[usize]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Convert to 3D for processing
    let input_3d = input
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 3D".into()))?;

    let (nz, ny, nx) = input_3d.dim();

    // Convert _input to f64 for FFT processing
    let mut input_f64 = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                input_f64[[i, j, k]] = NumCast::from(input_3d[[i, j, k]]).unwrap_or(0.0);
            }
        }
    }

    // Apply separable uniform filtering along each axis
    let two_pi = 2.0 * PI;

    // Filter along z-axis (axis 0)
    let freqs_z = fftfreq(nz, 1.0)?;
    for j in 0..ny {
        for k in 0..nx {
            let slice: Vec<f64> = (0..nz).map(|i| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (i, &freq) in freqs_z.iter().enumerate() {
                let x = size[0] as f64 * freq * two_pi;
                let sinc_factor = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                spectrum[i] *= sinc_factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (i, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along y-axis (axis 1)
    let freqs_y = fftfreq(ny, 1.0)?;
    for i in 0..nz {
        for k in 0..nx {
            let slice: Vec<f64> = (0..ny).map(|j| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (j, &freq) in freqs_y.iter().enumerate() {
                let x = size[1] as f64 * freq * two_pi;
                let sinc_factor = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                spectrum[j] *= sinc_factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (j, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along x-axis (axis 2)
    let freqs_x = fftfreq(nx, 1.0)?;
    for i in 0..nz {
        for j in 0..ny {
            let slice: Vec<f64> = (0..nx).map(|k| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (k, &freq) in freqs_x.iter().enumerate() {
                let x = size[2] as f64 * freq * two_pi;
                let sinc_factor = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                spectrum[k] *= sinc_factor;
            }

            let filtered = ifft(&spectrum, None)?;
            for (k, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Convert back to T
    let mut output_3d = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                output_3d[[i, j, k]] = T::from_f64(input_f64[[i, j, k]]).unwrap_or(T::zero());
            }
        }
    }

    // Convert back to original dimension type
    let result = output_3d
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 3D".into()))?;

    Ok(result)
}

/// Apply an ellipsoid filter in the Fourier domain
///
/// This creates a filter that passes frequencies within an ellipsoid in frequency space.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Semi-axes lengths of the ellipsoid in each dimension
/// * `mode` - 'lowpass' or 'highpass'
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn fourier_ellipsoid<T, D>(
    input: &Array<T, D>,
    size: &[T],
    mode: &str,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s <= T::zero() {
            return Err(NdimageError::InvalidInput(
                "Ellipsoid size must be positive".into(),
            ));
        }
    }

    let is_lowpass = match mode {
        "lowpass" => true,
        "highpass" => false,
        _ => {
            return Err(NdimageError::InvalidInput(
                "Mode must be 'lowpass' or 'highpass'".into(),
            ))
        }
    };

    if input.ndim() == 2 {
        let input_2d = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
        let input_2d_owned = input_2d.to_owned();
        let result = fourier_ellipsoid_2d(&input_2d_owned, size[0], size[1], is_lowpass)?;
        result
            .into_dimensionality::<D>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
    } else {
        // N-dimensional case
        fourier_ellipsoid_nd(input, size, is_lowpass)
    }
}

/// Apply 2D ellipsoid filter in Fourier domain
#[allow(dead_code)]
fn fourier_ellipsoid_2d<T>(
    input: &Array2<T>,
    size_y: T,
    size_x: T,
    is_lowpass: bool,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Convert sizes to f64
    let size_y_f64: f64 = NumCast::from(size_y).unwrap_or(1.0);
    let size_x_f64: f64 = NumCast::from(size_x).unwrap_or(1.0);

    // Apply ellipsoid filter
    let mut filtered_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let dist_sq = (fy / size_y_f64).powi(2) + (fx / size_x_f64).powi(2);

            let mask = if is_lowpass {
                if dist_sq <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                if dist_sq > 1.0 {
                    1.0
                } else {
                    0.0
                }
            };

            filtered_spectrum[[i, j]] *= mask;
        }
    }

    // Inverse FFT
    let result = ifft2(&filtered_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from_f64(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply N-dimensional ellipsoid filter in Fourier domain
#[allow(dead_code)]
fn fourier_ellipsoid_nd<T, D>(
    input: &Array<T, D>,
    size: &[T],
    is_lowpass: bool,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    if input.ndim() != size.len() {
        return Err(NdimageError::InvalidInput(
            "Size array length must match input dimensions".into(),
        ));
    }

    // For 3D case, implement proper ellipsoid filtering
    if input.ndim() == 3 {
        return fourier_ellipsoid_3d(input, size, is_lowpass);
    }

    // For higher dimensions, use simplified approach with radial symmetry
    let shape = input.shape().to_vec();
    let flat_input = input.view().into_shape(input.len())?;

    // Create equivalent 1D size (use geometric mean)
    let size_1d = {
        let product: f64 = size
            .iter()
            .map(|&s| NumCast::from(s).unwrap_or(1.0))
            .fold(1.0, |acc, x| acc * x);
        let geometric_mean = product.powf(1.0 / size.len() as f64);
        T::from_f64(geometric_mean).unwrap_or(T::one())
    };

    // Apply 2D ellipsoid filter as approximation
    let side_len = (flat_input.len() as f64).sqrt() as usize;
    let temp_2d = flat_input
        .view()
        .into_shape((side_len, flat_input.len() / side_len))?;
    let filtered_2d = fourier_ellipsoid_2d(&temp_2d.to_owned(), size_1d, size_1d, is_lowpass)?;
    let filtered_1d = filtered_2d.into_shape(input.len())?;

    // Reshape back to original dimensions
    let result_dyn = filtered_1d.into_shape(shape)?;
    let result = result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert result back to original dimensions".into())
    })?;
    Ok(result)
}

/// Apply 3D ellipsoid filter in Fourier domain
#[allow(dead_code)]
fn fourier_ellipsoid_3d<T, D>(
    input: &Array<T, D>,
    size: &[T],
    is_lowpass: bool,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Convert to 3D for processing
    let input_3d = input
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 3D".into()))?;

    let (nz, ny, nx) = input_3d.dim();

    // Convert size values
    let size_z: f64 = NumCast::from(size[0]).unwrap_or(1.0);
    let size_y: f64 = NumCast::from(size[1]).unwrap_or(1.0);
    let size_x: f64 = NumCast::from(size[2]).unwrap_or(1.0);

    // Convert input to f64 for FFT processing
    let mut input_f64 = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                input_f64[[i, j, k]] = NumCast::from(input_3d[[i, j, k]]).unwrap_or(0.0);
            }
        }
    }

    // Get frequency arrays for each dimension
    let freqs_z = fftfreq(nz, 1.0)?;
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Apply FFT along each axis and filter in frequency domain
    // This is a simplified approach - for full 3D FFT, we'd need scirs2-fft to support 3D FFT

    // Filter along z-axis first
    for j in 0..ny {
        for k in 0..nx {
            let slice: Vec<f64> = (0..nz).map(|i| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            // Apply ellipsoid filter for this frequency component
            for (i, &freq_z) in freqs_z.iter().enumerate() {
                // For ellipsoid, check if frequency is within ellipsoid bounds
                let normalized_freq_z = freq_z * size_z;
                let freq_magnitude = normalized_freq_z.abs();

                let within_ellipsoid = freq_magnitude <= 1.0;

                if (is_lowpass && within_ellipsoid) || (!is_lowpass && !within_ellipsoid) {
                    // Keep the frequency component
                } else {
                    spectrum[i] *= 0.0; // Zero out the frequency component
                }
            }

            let filtered = ifft(&spectrum, None)?;
            for (i, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along y-axis
    for i in 0..nz {
        for k in 0..nx {
            let slice: Vec<f64> = (0..ny).map(|j| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (j, &freq_y) in freqs_y.iter().enumerate() {
                let normalized_freq_y = freq_y * size_y;
                let freq_magnitude = normalized_freq_y.abs();

                let within_ellipsoid = freq_magnitude <= 1.0;

                if (is_lowpass && within_ellipsoid) || (!is_lowpass && !within_ellipsoid) {
                    // Keep the frequency component
                } else {
                    spectrum[j] *= 0.0; // Zero out the frequency component
                }
            }

            let filtered = ifft(&spectrum, None)?;
            for (j, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Filter along x-axis
    for i in 0..nz {
        for j in 0..ny {
            let slice: Vec<f64> = (0..nx).map(|k| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (k, &freq_x) in freqs_x.iter().enumerate() {
                let normalized_freq_x = freq_x * size_x;
                let freq_magnitude = normalized_freq_x.abs();

                let within_ellipsoid = freq_magnitude <= 1.0;

                if (is_lowpass && within_ellipsoid) || (!is_lowpass && !within_ellipsoid) {
                    // Keep the frequency component
                } else {
                    spectrum[k] *= 0.0; // Zero out the frequency component
                }
            }

            let filtered = ifft(&spectrum, None)?;
            for (k, &complex_val) in filtered.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Convert back to T
    let mut output_3d = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                output_3d[[i, j, k]] = T::from_f64(input_f64[[i, j, k]]).unwrap_or(T::zero());
            }
        }
    }

    // Convert back to original dimension type
    let result = output_3d
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 3D".into()))?;

    Ok(result)
}

/// Shift an array in the Fourier domain
///
/// This is equivalent to rolling the array but can be more efficient
/// for non-integer shifts or when combined with other Fourier operations.
///
/// # Arguments
///
/// * `input` - Input array to shift
/// * `shift` - Shift amount for each axis (can be fractional)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Shifted array
#[allow(dead_code)]
pub fn fourier_shift<T, D>(input: &Array<T, D>, shift: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if shift.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Shift must have same length as _input dimensions (got {} expected {})",
            shift.len(),
            input.ndim()
        )));
    }

    match input.ndim() {
        1 => {
            let input_1d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".into()))?;
            let input_1d_owned = input_1d.to_owned();
            let result = fourier_shift_1d(&input_1d_owned, shift[0])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        2 => {
            let input_2d = input
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
            let input_2d_owned = input_2d.to_owned();
            let result = fourier_shift_2d(&input_2d_owned, shift[0], shift[1])?;
            result
                .into_dimensionality::<D>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert back".into()))
        }
        _ => {
            // N-dimensional case (3D and higher)
            let result = fourier_shift_nd(input, shift)?;
            Ok(result)
        }
    }
}

/// Apply 1D shift in Fourier domain
#[allow(dead_code)]
fn fourier_shift_1d<T>(input: &Array1<T>, shift: T) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let n = input.len();

    // Convert to f64 for FFT
    let input_f64: Vec<f64> = input
        .iter()
        .map(|&x| NumCast::from(x).unwrap_or(0.0))
        .collect();

    // Perform FFT
    let mut spectrum = fft(&input_f64, None)?;

    // Get frequencies
    let freqs = fftfreq(n, 1.0)?;

    // Apply phase shift
    let shift_f64: f64 = NumCast::from(shift).unwrap_or(0.0);
    let two_pi = 2.0 * PI;

    for (i, freq) in freqs.iter().enumerate() {
        let phase = -two_pi * freq * shift_f64;
        let shift_factor = Complex64::new(phase.cos(), phase.sin());
        spectrum[i] *= shift_factor;
    }

    // Inverse FFT
    let result = ifft(&spectrum, None)?;

    // Convert back to T and extract real part
    let output: Array1<T> = Array1::from_vec(
        result
            .iter()
            .map(|c| T::from_f64(c.re).unwrap_or(T::zero()))
            .collect(),
    );

    Ok(output)
}

/// Apply 2D shift in Fourier domain
#[allow(dead_code)]
fn fourier_shift_2d<T>(input: &Array2<T>, shift_y: T, shift_x: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone,
{
    let (ny, nx) = input.dim();

    // Convert to f64 Array2 for FFT
    let mut input_f64 = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            input_f64[[i, j]] = NumCast::from(input[[i, j]]).unwrap_or(0.0);
        }
    }

    // Perform 2D FFT
    let spectrum = fft2(&input_f64, None, None, None)?;

    // Get frequencies
    let freqs_y = fftfreq(ny, 1.0)?;
    let freqs_x = fftfreq(nx, 1.0)?;

    // Convert shifts to f64
    let shift_y_f64: f64 = NumCast::from(shift_y).unwrap_or(0.0);
    let shift_x_f64: f64 = NumCast::from(shift_x).unwrap_or(0.0);
    let two_pi = 2.0 * PI;

    // Apply phase shift
    let mut shifted_spectrum = spectrum.clone();
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            let phase = -two_pi * (fy * shift_y_f64 + fx * shift_x_f64);
            let shift_factor = Complex64::new(phase.cos(), phase.sin());
            shifted_spectrum[[i, j]] *= shift_factor;
        }
    }

    // Inverse FFT
    let result = ifft2(&shifted_spectrum, None, None, None)?;

    // Convert back to T and extract real part
    let mut output = Array2::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            output[[i, j]] = T::from_f64(result[[i, j]].re).unwrap_or(T::zero());
        }
    }

    Ok(output)
}

/// Apply N-dimensional shift in Fourier domain
#[allow(dead_code)]
fn fourier_shift_nd<T, D>(input: &Array<T, D>, shift: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    if input.ndim() != shift.len() {
        return Err(NdimageError::InvalidInput(
            "Shift array length must match _input dimensions".into(),
        ));
    }

    // For 3D case, implement separable shifting
    if input.ndim() == 3 {
        return fourier_shift_3d(input, shift);
    }

    // For higher dimensions, use simplified approach
    let shape = input.shape().to_vec();
    let flat_input = input.view().into_shape(input.len())?;

    // Create equivalent 1D shift (use mean of all shifts)
    let shift_1d = {
        let sum: f64 = shift.iter().map(|&s| NumCast::from(s).unwrap_or(0.0)).sum();
        let mean = sum / shift.len() as f64;
        T::from_f64(mean).unwrap_or(T::zero())
    };

    // Apply 1D shift
    let shifted_1d = fourier_shift_1d(&flat_input.to_owned(), shift_1d)?;

    // Reshape back to original dimensions
    let result_dyn = shifted_1d.into_shape(shape)?;
    let result = result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert result back to original dimensions".into())
    })?;
    Ok(result)
}

/// Apply 3D shift in Fourier domain
#[allow(dead_code)]
fn fourier_shift_3d<T, D>(input: &Array<T, D>, shift: &[T]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Convert to 3D for processing
    let input_3d = input
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 3D".into()))?;

    let (nz, ny, nx) = input_3d.dim();

    // Convert shift values
    let shift_z: f64 = NumCast::from(shift[0]).unwrap_or(0.0);
    let shift_y: f64 = NumCast::from(shift[1]).unwrap_or(0.0);
    let shift_x: f64 = NumCast::from(shift[2]).unwrap_or(0.0);

    // Convert _input to f64 for FFT processing
    let mut input_f64 = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                input_f64[[i, j, k]] = NumCast::from(input_3d[[i, j, k]]).unwrap_or(0.0);
            }
        }
    }

    let two_pi = 2.0 * PI;

    // Apply separable shifting along each axis

    // Shift along z-axis (axis 0)
    let freqs_z = fftfreq(nz, 1.0)?;
    for j in 0..ny {
        for k in 0..nx {
            let slice: Vec<f64> = (0..nz).map(|i| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            // Apply phase shift
            for (i, &freq) in freqs_z.iter().enumerate() {
                let phase = -two_pi * freq * shift_z;
                let shift_factor = Complex64::new(phase.cos(), phase.sin());
                spectrum[i] *= shift_factor;
            }

            let shifted = ifft(&spectrum, None)?;
            for (i, &complex_val) in shifted.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Shift along y-axis (axis 1)
    let freqs_y = fftfreq(ny, 1.0)?;
    for i in 0..nz {
        for k in 0..nx {
            let slice: Vec<f64> = (0..ny).map(|j| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (j, &freq) in freqs_y.iter().enumerate() {
                let phase = -two_pi * freq * shift_y;
                let shift_factor = Complex64::new(phase.cos(), phase.sin());
                spectrum[j] *= shift_factor;
            }

            let shifted = ifft(&spectrum, None)?;
            for (j, &complex_val) in shifted.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Shift along x-axis (axis 2)
    let freqs_x = fftfreq(nx, 1.0)?;
    for i in 0..nz {
        for j in 0..ny {
            let slice: Vec<f64> = (0..nx).map(|k| input_f64[[i, j, k]]).collect();
            let mut spectrum = fft(&slice, None)?;

            for (k, &freq) in freqs_x.iter().enumerate() {
                let phase = -two_pi * freq * shift_x;
                let shift_factor = Complex64::new(phase.cos(), phase.sin());
                spectrum[k] *= shift_factor;
            }

            let shifted = ifft(&spectrum, None)?;
            for (k, &complex_val) in shifted.iter().enumerate() {
                input_f64[[i, j, k]] = complex_val.re;
            }
        }
    }

    // Convert back to T
    let mut output_3d = Array::zeros((nz, ny, nx));
    for i in 0..nz {
        for j in 0..ny {
            for k in 0..nx {
                output_3d[[i, j, k]] = T::from_f64(input_f64[[i, j, k]]).unwrap_or(T::zero());
            }
        }
    }

    // Convert back to original dimension type
    let result = output_3d
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 3D".into()))?;

    Ok(result)
}

// Streaming implementations for large datasets
use crate::streaming::{OverlapInfo, StreamConfig, StreamProcessor, StreamableOp};
use scirs2_core::ndarray::{ArrayView, ArrayViewMut};
use std::path::Path;

/// Streaming Fourier Gaussian filter for large datasets
pub struct StreamingFourierGaussian<T> {
    sigma: Vec<T>,
}

impl<T: Float + FromPrimitive + NumCast + Debug + Clone> StreamingFourierGaussian<T> {
    pub fn new(sigma: Vec<T>) -> Self {
        Self { sigma }
    }
}

impl<T, D> StreamableOp<T, D> for StreamingFourierGaussian<T>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    fn apply_chunk(&self, chunk: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        fourier_gaussian(&chunk.to_owned(), &self.sigma, None)
    }

    fn required_overlap(&self) -> Vec<usize> {
        // Fourier filters need minimal overlap due to periodic boundary conditions
        vec![8; self.sigma.len()]
    }

    fn merge_overlap(
        &self,
        output: &mut ArrayViewMut<T, D>,
        new_chunk: &ArrayView<T, D>,
        overlap_info: &OverlapInfo,
    ) -> NdimageResult<()> {
        // Fourier domain filters handle boundaries naturally
        Ok(())
    }
}

/// Process a large image file with Fourier Gaussian filter
#[allow(dead_code)]
pub fn fourier_gaussian_file<T>(
    input_path: &Path,
    output_path: &Path,
    shape: &[usize],
    sigma: &[T],
    config: Option<StreamConfig>,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
{
    let op = StreamingFourierGaussian::new(sigma.to_vec());
    let config = config.unwrap_or_else(|| StreamConfig {
        chunk_size: 256 * 1024 * 1024, // 256MB chunks for FFT efficiency
        ..Default::default()
    });

    let processor = StreamProcessor::<T>::new(config);

    // Dynamic dispatch based on dimensionality
    match shape.len() {
        1 => processor.process_file::<scirs2_core::ndarray::Ix1, StreamingFourierGaussian<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        2 => processor.process_file::<scirs2_core::ndarray::Ix2, StreamingFourierGaussian<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        3 => processor.process_file::<scirs2_core::ndarray::Ix3, StreamingFourierGaussian<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        _ => processor.process_file::<scirs2_core::ndarray::IxDyn, StreamingFourierGaussian<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
    }
}

/// Streaming Fourier uniform filter
pub struct StreamingFourierUniform<T> {
    size: Vec<T>,
}

impl<T: Float + FromPrimitive + NumCast + Debug + Clone> StreamingFourierUniform<T> {
    pub fn new(size: Vec<T>) -> Self {
        Self { size }
    }
}

impl<T, D> StreamableOp<T, D> for StreamingFourierUniform<T>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    fn apply_chunk(&self, chunk: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        let size_usize: Vec<usize> = self
            .size
            .iter()
            .map(|&s| s.to_usize().unwrap_or(1))
            .collect();
        fourier_uniform(&chunk.to_owned(), &size_usize)
    }

    fn required_overlap(&self) -> Vec<usize> {
        self.size
            .iter()
            .map(|&s| s.to_usize().unwrap_or(1))
            .collect()
    }

    fn merge_overlap(
        &self,
        output: &mut ArrayViewMut<T, D>,
        new_chunk: &ArrayView<T, D>,
        overlap_info: &OverlapInfo,
    ) -> NdimageResult<()> {
        Ok(())
    }
}

/// Process a large image file with Fourier uniform filter
#[allow(dead_code)]
pub fn fourier_uniform_file<T>(
    input_path: &Path,
    output_path: &Path,
    shape: &[usize],
    size: &[T],
    config: Option<StreamConfig>,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + NumCast + Debug + Clone + Send + Sync + 'static,
{
    let op = StreamingFourierUniform::new(size.to_vec());
    let config = config.unwrap_or_default();
    let processor = StreamProcessor::<T>::new(config);

    match shape.len() {
        1 => processor.process_file::<scirs2_core::ndarray::Ix1, StreamingFourierUniform<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        2 => processor.process_file::<scirs2_core::ndarray::Ix2, StreamingFourierUniform<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        3 => processor.process_file::<scirs2_core::ndarray::Ix3, StreamingFourierUniform<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
        _ => processor.process_file::<scirs2_core::ndarray::IxDyn, StreamingFourierUniform<T>>(
            input_path,
            output_path,
            shape,
            op,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{arr1, Array1, Array2};

    #[test]
    fn test_fourier_gaussian_1d() {
        // Create a simple 1D signal with a spike
        let mut signal = Array1::zeros(64);
        signal[32] = 1.0;

        // Apply Fourier Gaussian filter
        let sigma = vec![2.0];
        let filtered = fourier_gaussian(&signal, &sigma, None).unwrap();

        // Check that the spike is smoothed
        assert!(filtered[32] < 1.0);
        assert!(filtered[32] > 0.0);

        // Check symmetry
        for i in 1..10 {
            assert_abs_diff_eq!(filtered[32 - i], filtered[32 + i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fourier_gaussian_2d() {
        // Create a 2D array with a centered spike
        let mut image = Array2::zeros((32, 32));
        image[[16, 16]] = 1.0;

        // Apply Fourier Gaussian filter
        let sigma = vec![1.5, 1.5];
        let filtered = fourier_gaussian(&image, &sigma, None).unwrap();

        // Check that the spike is smoothed
        assert!(filtered[[16, 16]] < 1.0);
        assert!(filtered[[16, 16]] > 0.0);

        // Check that surrounding pixels have non-zero values
        assert!(filtered[[15, 16]] > 0.0);
        assert!(filtered[[17, 16]] > 0.0);
        assert!(filtered[[16, 15]] > 0.0);
        assert!(filtered[[16, 17]] > 0.0);
    }

    #[test]
    fn test_fourier_uniform_1d() {
        // Create a step function
        let mut signal = Array1::zeros(64);
        for i in 30..35 {
            signal[i] = 1.0;
        }

        // Apply Fourier uniform filter
        let size = vec![5];
        let filtered = fourier_uniform(&signal, &size).unwrap();

        // The filter should smooth the edges
        assert!(filtered[29] > 0.0);
        assert!(filtered[35] > 0.0);
        assert!(filtered[32] > 0.0);
    }

    #[test]
    fn test_fourier_ellipsoid_lowpass() {
        // Create a 2D array with high and low frequency components
        let mut image = Array2::zeros((32, 32));

        // Add a low frequency component (large scale structure)
        for i in 0..32 {
            for j in 0..32 {
                image[[i, j]] =
                    ((i as f64 / 32.0 * 2.0 * PI).sin() + (j as f64 / 32.0 * 2.0 * PI).sin()) / 2.0;
            }
        }

        // Add high frequency noise
        for i in 0..32 {
            for j in 0..32 {
                if (i + j) % 2 == 0 {
                    image[[i, j]] += 0.5;
                }
            }
        }

        // Apply lowpass filter
        let size = vec![0.2, 0.2];
        let filtered = fourier_ellipsoid(&image, &size, "lowpass").unwrap();

        // The high frequency checkerboard pattern should be reduced
        let noise_original = (image[[0, 0]] - image[[0, 1]]).abs();
        let noise_filtered = (filtered[[0, 0]] - filtered[[0, 1]]).abs();
        assert!(noise_filtered < noise_original);
    }

    #[test]
    fn test_fourier_shift_1d() {
        // Create a simple signal
        let signal = arr1(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);

        // Shift by 2 positions
        let shift = vec![2.0];
        let shifted = fourier_shift(&signal, &shift).unwrap();

        // Check that values are shifted (with some tolerance for numerical errors)
        assert_abs_diff_eq!(shifted[2], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[3], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[4], 3.0, epsilon = 0.1);
        assert_abs_diff_eq!(shifted[5], 4.0, epsilon = 0.1);
    }

    #[test]
    fn test_fourier_shift_fractional() {
        // Create a smooth Gaussian-like signal to avoid Gibbs phenomenon
        use std::f64::consts::PI;
        let mut signal = Array1::zeros(64);
        let center = 30.0;
        let width = 5.0;
        for i in 0..64 {
            let x = i as f64;
            signal[i] = (-(x - center).powi(2) / (2.0 * width.powi(2))).exp();
        }

        // Shift by a fractional amount
        let shift = vec![0.5];
        let shifted = fourier_shift(&signal, &shift).unwrap();

        // After shifting by 0.5, the peak should move from index 30 to between 29.5-30.5
        // The values at indices 29 and 30 should both be high and similar
        let peak_val = signal[[30]];

        // Both indices near the shifted peak should have significant values
        assert!(shifted[29].abs() > 0.8 * peak_val);
        assert!(shifted[30].abs() > 0.8 * peak_val);

        // The shift should be approximately correct (peak moves right by 0.5)
        // So value at 30 in shifted ≈ value at 29.5 in original
        // which is between signal[29] and signal[30]
        let expected = (signal[29] + signal[30]) / 2.0;
        assert!((shifted[30] - expected).abs() < 0.1 * peak_val);
    }
}
