//! Advanced filtering operations for n-dimensional arrays
//!
//! This module provides advanced filtering functions including Gabor filters,
//! steerable filters, and other specialized convolution operations that are
//! commonly used in computer vision and signal processing.

use scirs2_core::ndarray::{Array2, ArrayView2, Dimension, Zip};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{convolve, BorderMode};
use crate::utils::safe_f64_to_float;

/// Helper function for safe usize conversion
#[allow(dead_code)]
fn safe_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert _value to usize".to_string())
    })
}

/// Helper function for safe f64 conversion
#[allow(dead_code)]
fn safe_to_f64<T: Float>(value: T) -> NdimageResult<f64> {
    value
        .to_f64()
        .ok_or_else(|| NdimageError::ComputationError("Failed to convert value to f64".to_string()))
}

/// Helper function for safe isize conversion to float
#[allow(dead_code)]
fn safe_isize_to_float<T: Float + FromPrimitive>(value: isize) -> NdimageResult<T> {
    T::from_isize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert isize {} to float type", value))
    })
}

/// Helper function for safe usize conversion to float
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Gabor filter parameters
#[derive(Debug, Clone)]
pub struct GaborParams<T> {
    /// Wavelength of the sinusoidal component
    pub wavelength: T,
    /// Orientation of the Gabor filter in radians (0 = horizontal)
    pub orientation: T,
    /// Standard deviation of the Gaussian envelope in x direction
    pub sigma_x: T,
    /// Standard deviation of the Gaussian envelope in y direction  
    pub sigma_y: T,
    /// Phase offset of the sinusoidal component in radians
    pub phase: T,
    /// Aspect ratio (sigma_y / sigma_x)
    pub aspect_ratio: Option<T>,
}

impl<T> Default for GaborParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            wavelength: safe_f64_to_float::<T>(4.0)
                .unwrap_or_else(|_| T::from_f64(4.0).unwrap_or_else(|| T::one())),
            orientation: T::zero(),
            sigma_x: safe_f64_to_float::<T>(2.0)
                .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one())),
            sigma_y: safe_f64_to_float::<T>(2.0)
                .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one())),
            phase: T::zero(),
            aspect_ratio: None,
        }
    }
}

/// Apply a Gabor filter to a 2D array
///
/// Gabor filters are widely used in image processing for texture analysis,
/// edge detection, and feature extraction. They combine a Gaussian envelope
/// with a sinusoidal wave at a specific orientation and frequency.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `params` - Gabor filter parameters
/// * `kernel_size` - Size of the filter kernel (if None, automatically calculated)
/// * `mode` - Border handling mode
///
/// # Returns
///
/// * `Result<Array2<T>>` - Filtered array
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::filters::advanced::{gabor_filter, GaborParams};
/// use scirs2_ndimage::filters::BorderMode;
///
/// let image = Array2::from_elem((64, 64), 0.5);
/// let params = GaborParams {
///     wavelength: 8.0,
///     orientation: std::f64::consts::PI / 4.0, // 45 degrees
///     sigma_x: 3.0,
///     sigma_y: 3.0,
///     phase: 0.0,
///     aspect_ratio: None,
/// };
///
/// let result = gabor_filter(&image.view(), &params, None, Some(BorderMode::Reflect)).unwrap();
/// ```
#[allow(dead_code)]
pub fn gabor_filter<T>(
    input: &ArrayView2<T>,
    params: &GaborParams<T>,
    kernel_size: Option<usize>,
    mode: Option<BorderMode>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Determine kernel _size
    let sigma_max = if let Some(aspect) = params.aspect_ratio {
        params.sigma_x.max(params.sigma_x * aspect)
    } else {
        params.sigma_x.max(params.sigma_y)
    };

    let auto_size = (sigma_max
        * safe_f64_to_float::<T>(6.0)
            .unwrap_or_else(|_| T::from_f64(6.0).unwrap_or_else(|| T::one())))
    .to_usize()
    .unwrap_or(21);
    let _size = kernel_size.unwrap_or(auto_size);

    // Ensure odd kernel _size
    let _size = if _size % 2 == 0 { _size + 1 } else { _size };

    // Generate Gabor kernel
    let kernel = generate_gabor_kernel(_size, params)?;

    // Apply convolution
    convolve(&input.to_owned(), &kernel, Some(border_mode))
}

/// Generate a Gabor filter kernel
#[allow(dead_code)]
fn generate_gabor_kernel<T>(size: usize, params: &GaborParams<T>) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug,
{
    let mut kernel = Array2::zeros((size, size));
    let center = (size / 2) as isize;

    let cos_theta = params.orientation.cos();
    let sin_theta = params.orientation.sin();

    let sigma_x = params.sigma_x;
    let sigma_y = if let Some(aspect) = params.aspect_ratio {
        sigma_x * aspect
    } else {
        params.sigma_y
    };

    let lambda = params.wavelength;
    let phase = params.phase;

    // Pre-compute constants
    let sigma_x_sq = sigma_x * sigma_x;
    let sigma_y_sq = sigma_y * sigma_y;
    let two_pi = safe_f64_to_float::<T>(2.0 * std::f64::consts::PI)?;
    let two = safe_f64_to_float::<T>(2.0)?;

    for i in 0..size {
        for j in 0..size {
            let x: T = safe_isize_to_float(j as isize - center)?;
            let y: T = safe_isize_to_float(i as isize - center)?;

            // Rotate coordinates
            let x_rot = x * cos_theta + y * sin_theta;
            let y_rot = -x * sin_theta + y * cos_theta;

            // Gaussian envelope
            let gauss_exp =
                -(x_rot * x_rot / (two * sigma_x_sq) + y_rot * y_rot / (two * sigma_y_sq));
            let gauss = gauss_exp.exp();

            // Sinusoidal component
            let wave_arg = two_pi * x_rot / lambda + phase;
            let wave = wave_arg.cos();

            kernel[[i, j]] = gauss * wave;
        }
    }

    Ok(kernel)
}

/// Apply a bank of Gabor filters with different orientations
///
/// This function applies multiple Gabor filters with different orientations
/// to capture texture features at multiple angles.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `base_params` - Base Gabor parameters (orientation will be modified)
/// * `num_orientations` - Number of different orientations to use
/// * `kernel_size` - Size of the filter kernels
/// * `mode` - Border handling mode
///
/// # Returns
///
/// * `Result<Vec<Array2<T>>>` - Vector of filtered arrays, one for each orientation
#[allow(dead_code)]
pub fn gabor_filter_bank<T>(
    input: &ArrayView2<T>,
    base_params: &GaborParams<T>,
    num_orientations: usize,
    kernel_size: Option<usize>,
    mode: Option<BorderMode>,
) -> NdimageResult<Vec<Array2<T>>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let mut results = Vec::with_capacity(num_orientations);
    let pi = safe_f64_to_float::<T>(std::f64::consts::PI)?;
    let angle_step = pi / safe_usize_to_float::<T>(num_orientations)?;

    for i in 0..num_orientations {
        let mut params = base_params.clone();
        params.orientation = safe_usize_to_float::<T>(i)? * angle_step;

        let filtered = gabor_filter(input, &params, kernel_size, mode)?;
        results.push(filtered);
    }

    Ok(results)
}

/// Apply a Log-Gabor filter to a 2D array
///
/// Log-Gabor filters have advantages over traditional Gabor filters as they
/// have no DC component and can be constructed with arbitrary bandwidth.
/// This implementation uses proper FFT processing for accurate frequency domain filtering.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `center_freq` - Center frequency of the filter (normalized, 0.0-0.5)
/// * `bandwidth` - Bandwidth of the filter (in octaves)
/// * `orientation` - Orientation in radians (0 = horizontal)
/// * `angular_bandwidth` - Angular bandwidth in radians
///
/// # Returns
///
/// * `Result<Array2<T>>` - Filtered array
#[allow(dead_code)]
pub fn log_gabor_filter<T>(
    input: &ArrayView2<T>,
    center_freq: T,
    bandwidth: T,
    orientation: T,
    angular_bandwidth: T,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static + SimdUnifiedOps,
{
    let (height, width) = input.dim();

    // Validate inputs
    let center_freq_f64 = safe_to_f64(center_freq)?;
    if center_freq_f64 <= 0.0 || center_freq_f64 >= 0.5 {
        return Err(NdimageError::InvalidInput(
            "Center frequency must be between 0.0 and 0.5".into(),
        ));
    }

    // Convert input to f64 for FFT processing
    let input_f64: Array2<f64> = input.mapv(|x| safe_to_f64(x).unwrap_or(0.0));

    // Apply 2D FFT
    let fft_result = apply_2d_fft(&input_f64)?;

    // Create Log-Gabor filter in frequency domain
    let filter = create_log_gabor_frequency_filter(
        height,
        width,
        center_freq_f64,
        safe_to_f64(bandwidth)?,
        safe_to_f64(orientation)?,
        safe_to_f64(angular_bandwidth)?,
    )?;

    // Apply filter in frequency domain
    let filtered_fft = apply_frequency_filter(&fft_result, &filter)?;

    // Apply inverse FFT
    let result_f64 = apply_2d_ifft(&filtered_fft)?;

    // Convert back to original type and take real part
    let output: Array2<T> =
        result_f64.mapv(|x| safe_f64_to_float::<T>(x.re).unwrap_or_else(|_| T::zero()));

    Ok(output)
}

/// Complex number representation for FFT
#[derive(Debug, Clone, Copy)]
struct Complex64 {
    re: f64,
    im: f64,
}

impl Complex64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn magnitude_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Apply 2D FFT to input array
#[allow(dead_code)]
fn apply_2d_fft(input: &Array2<f64>) -> NdimageResult<Array2<Complex64>> {
    let (height, width) = input.dim();
    let mut result = Array2::from_elem((height, width), Complex64::zero());

    // Convert _input to complex
    for ((i, j), &val) in input.indexed_iter() {
        result[[i, j]] = Complex64::new(val, 0.0);
    }

    // Apply FFT to each row (simplified implementation)
    for i in 0..height {
        let mut row: Vec<Complex64> = (0..width).map(|j| result[[i, j]]).collect();
        fft_1d_radix2(&mut row);
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    // Apply FFT to each column
    for j in 0..width {
        let mut col: Vec<Complex64> = (0..height).map(|i| result[[i, j]]).collect();
        fft_1d_radix2(&mut col);
        for (i, val) in col.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// Apply inverse 2D FFT
#[allow(dead_code)]
fn apply_2d_ifft(input: &Array2<Complex64>) -> NdimageResult<Array2<Complex64>> {
    let (height, width) = input.dim();
    let mut result = input.clone();

    // Apply IFFT to each row
    for i in 0..height {
        let mut row: Vec<Complex64> = (0..width).map(|j| result[[i, j]]).collect();
        ifft_1d_radix2(&mut row);
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    // Apply IFFT to each column
    for j in 0..width {
        let mut col: Vec<Complex64> = (0..height).map(|i| result[[i, j]]).collect();
        ifft_1d_radix2(&mut col);
        for (i, val) in col.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// Simple radix-2 FFT implementation for 1D arrays
#[allow(dead_code)]
fn fft_1d_radix2(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey FFT
    let mut length = 2;
    while length <= n {
        let angle = -2.0 * std::f64::consts::PI / length as f64;
        let wlen = Complex64::new(angle.cos(), angle.sin());

        for i in (0..n).step_by(length) {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..length / 2 {
                let u = data[i + j];
                let v = data[i + j + length / 2].mul(w);
                data[i + j] = Complex64::new(u.re + v.re, u.im + v.im);
                data[i + j + length / 2] = Complex64::new(u.re - v.re, u.im - v.im);
                w = w.mul(wlen);
            }
        }
        length <<= 1;
    }
}

/// Simple radix-2 IFFT implementation for 1D arrays
#[allow(dead_code)]
fn ifft_1d_radix2(data: &mut [Complex64]) {
    let n = data.len();

    // Conjugate the complex numbers
    for item in data.iter_mut() {
        item.im = -item.im;
    }

    // Apply FFT
    fft_1d_radix2(data);

    // Conjugate again and scale
    for item in data.iter_mut() {
        item.im = -item.im;
        item.re /= n as f64;
        item.im /= n as f64;
    }
}

/// Create Log-Gabor filter in frequency domain
#[allow(dead_code)]
fn create_log_gabor_frequency_filter(
    height: usize,
    width: usize,
    center_freq: f64,
    bandwidth: f64,
    orientation: f64,
    angular_bandwidth: f64,
) -> NdimageResult<Array2<f64>> {
    let mut filter = Array2::zeros((height, width));

    let _center_y = height as f64 / 2.0;
    let _center_x = width as f64 / 2.0;

    for i in 0..height {
        for j in 0..width {
            // Calculate frequency coordinates (shifted to center)
            let fy = if i < height / 2 {
                i as f64 / height as f64
            } else {
                (i as f64 - height as f64) / height as f64
            };

            let fx = if j < width / 2 {
                j as f64 / width as f64
            } else {
                (j as f64 - width as f64) / width as f64
            };

            // Calculate radius and angle in frequency domain
            let radius = (fx * fx + fy * fy).sqrt();
            let angle = fy.atan2(fx);

            // Log-Gabor radial component
            let radial_response = if radius > 1e-10 {
                let log_radius = radius.log2();
                let log_center = center_freq.log2();
                let radial_diff = log_radius - log_center;
                (-radial_diff * radial_diff / (2.0 * bandwidth * bandwidth)).exp()
            } else {
                0.0 // No DC component
            };

            // Angular component (Gaussian)
            let angle_diff = (angle - orientation).abs();
            let angle_diff_wrapped = angle_diff.min(2.0 * std::f64::consts::PI - angle_diff);
            let angular_response = (-angle_diff_wrapped * angle_diff_wrapped
                / (2.0 * angular_bandwidth * angular_bandwidth))
                .exp();

            filter[[i, j]] = radial_response * angular_response;
        }
    }

    Ok(filter)
}

/// Apply frequency domain filter
#[allow(dead_code)]
fn apply_frequency_filter(
    fft_data: &Array2<Complex64>,
    filter: &Array2<f64>,
) -> NdimageResult<Array2<Complex64>> {
    if fft_data.dim() != filter.dim() {
        return Err(NdimageError::DimensionError(
            "FFT _data and filter must have the same dimensions".into(),
        ));
    }

    let result = Zip::from(fft_data)
        .and(filter)
        .map_collect(|&complex_val, &filter_val| {
            Complex64::new(complex_val.re * filter_val, complex_val.im * filter_val)
        });

    Ok(result)
}

/// Apply a steerable filter to a 2D array
///
/// Steerable filters allow efficient computation of filter responses at
/// arbitrary orientations by combining a small number of basis filters.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `filter_order` - Order of the steerable filter (1, 2, or 3)
/// * `orientation` - Desired orientation in radians
/// * `sigma` - Standard deviation of the Gaussian envelope
/// * `mode` - Border handling mode
///
/// # Returns
///
/// * `Result<Array2<T>>` - Filtered array
#[allow(dead_code)]
pub fn steerable_filter<T>(
    input: &ArrayView2<T>,
    filter_order: usize,
    orientation: T,
    sigma: T,
    mode: Option<BorderMode>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    if filter_order == 0 || filter_order > 3 {
        return Err(NdimageError::InvalidInput(
            "Steerable filter _order must be 1, 2, or 3".into(),
        ));
    }

    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Generate basis filters
    let basis_filters = generate_steerable_basis(filter_order, sigma)?;

    // Compute steering coefficients
    let coeffs = compute_steering_coefficients(filter_order, orientation);

    // Apply basis filters and combine with steering coefficients
    let mut result = Array2::zeros(input.dim());

    for (basis_filter, coeff) in basis_filters.iter().zip(coeffs.iter()) {
        let filtered = convolve(&input.to_owned(), basis_filter, Some(border_mode))?;
        result = &result + &(filtered.mapv(|x| x * *coeff));
    }

    Ok(result)
}

/// Generate basis filters for steerable filtering
#[allow(dead_code)]
fn generate_steerable_basis<T>(order: usize, sigma: T) -> NdimageResult<Vec<Array2<T>>>
where
    T: Float + FromPrimitive + Debug,
{
    let size = (sigma
        * safe_f64_to_float::<T>(6.0)
            .unwrap_or_else(|_| T::from_f64(6.0).unwrap_or_else(|| T::one())))
    .to_usize()
    .unwrap_or(15);
    let size = if size % 2 == 0 { size + 1 } else { size };
    let center = (size / 2) as isize;

    let mut basis_filters = Vec::new();

    match order {
        1 => {
            // First-_order steerable filter (G1 basis)
            for angle in [0.0, std::f64::consts::PI / 2.0] {
                let mut kernel = Array2::zeros((size, size));
                let cos_theta = safe_f64_to_float::<T>(angle.cos())?;
                let sin_theta = safe_f64_to_float::<T>(angle.sin())?;

                for i in 0..size {
                    for j in 0..size {
                        let x: T = safe_isize_to_float(j as isize - center)?;
                        let y: T = safe_isize_to_float(i as isize - center)?;

                        let r_sq: T = x * x + y * y;
                        let gauss = (-r_sq / (safe_f64_to_float::<T>(2.0)? * sigma * sigma)).exp();

                        // First derivative of Gaussian
                        let deriv = if angle == 0.0 { x } else { y };
                        kernel[[i, j]] = -deriv * gauss / (sigma * sigma);
                    }
                }
                basis_filters.push(kernel);
            }
        }
        2 => {
            // Second-_order steerable filter (G2 basis)
            for angle in [0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0] {
                let mut kernel = Array2::zeros((size, size));

                for i in 0..size {
                    for j in 0..size {
                        let x: T = safe_isize_to_float(j as isize - center)?;
                        let y: T = safe_isize_to_float(i as isize - center)?;

                        let r_sq: T = x * x + y * y;
                        let gauss = (-r_sq / (safe_f64_to_float::<T>(2.0)? * sigma * sigma)).exp();
                        let sigma_sq = sigma * sigma;

                        let value = match angle as i32 {
                            0 => (x * x - sigma_sq) * gauss / (sigma_sq * sigma_sq), // G2a
                            1 => {
                                (safe_f64_to_float::<T>(2.0)? * x * y) * gauss
                                    / (sigma_sq * sigma_sq)
                            } // G2b
                            _ => (y * y - sigma_sq) * gauss / (sigma_sq * sigma_sq), // G2c
                        };

                        kernel[[i, j]] = value;
                    }
                }
                basis_filters.push(kernel);
            }
        }
        3 => {
            // Third-_order steerable filter (G3 basis)
            // Third-_order derivatives of Gaussian in different orientations
            for angle_idx in 0..4 {
                let angle = angle_idx as f64 * std::f64::consts::PI / 4.0; // 0, π/4, π/2, 3π/4
                let mut kernel = Array2::zeros((size, size));

                for i in 0..size {
                    for j in 0..size {
                        let x: T = safe_isize_to_float(j as isize - center)?;
                        let y: T = safe_isize_to_float(i as isize - center)?;

                        let r_sq: T = x * x + y * y;
                        let gauss = (-r_sq / (safe_f64_to_float::<T>(2.0)? * sigma * sigma)).exp();
                        let sigma_cubed = sigma * sigma * sigma;
                        let sigma_sq = sigma * sigma;

                        // Third-_order derivatives of Gaussian
                        let value = match angle_idx {
                            0 => {
                                // d³/dx³
                                let x3 = x * x * x;
                                let term1 = x3 / (sigma_cubed * sigma_cubed);
                                let term2 = safe_f64_to_float::<T>(3.0)? * x / (sigma_cubed);
                                (term1 - term2) * gauss
                            }
                            1 => {
                                // d³/dx²dy (mixed derivative)
                                let x2y = x * x * y;
                                let term1 = x2y / (sigma_cubed * sigma_cubed);
                                let term2 = y / (sigma_cubed);
                                (term1 - term2) * gauss
                            }
                            2 => {
                                // d³/dxdy² (mixed derivative)
                                let xy2 = x * y * y;
                                let term1 = xy2 / (sigma_cubed * sigma_cubed);
                                let term2 = x / (sigma_cubed);
                                (term1 - term2) * gauss
                            }
                            _ => {
                                // d³/dy³
                                let y3 = y * y * y;
                                let term1 = y3 / (sigma_cubed * sigma_cubed);
                                let term2 = safe_f64_to_float::<T>(3.0)? * y / (sigma_cubed);
                                (term1 - term2) * gauss
                            }
                        };

                        kernel[[i, j]] = value;
                    }
                }
                basis_filters.push(kernel);
            }
        }
        _ => unreachable!(),
    }

    Ok(basis_filters)
}

/// Compute steering coefficients for a given orientation
#[allow(dead_code)]
fn compute_steering_coefficients<T>(order: usize, orientation: T) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    let cos_theta = orientation.cos();
    let sin_theta = orientation.sin();

    match order {
        1 => vec![cos_theta, sin_theta],
        2 => {
            let cos2 = cos_theta * cos_theta;
            let sin2 = sin_theta * sin_theta;
            let cos_sin = cos_theta * sin_theta;
            vec![
                cos2,
                safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()))
                    * cos_sin,
                sin2,
            ]
        }
        3 => {
            let cos3 = cos_theta * cos_theta * cos_theta;
            let sin3 = sin_theta * sin_theta * sin_theta;
            let cos2_sin = cos_theta * cos_theta * sin_theta;
            let cos_sin2 = cos_theta * sin_theta * sin_theta;
            vec![
                cos3,
                safe_f64_to_float::<T>(3.0)
                    .unwrap_or_else(|_| T::from_f64(3.0).unwrap_or_else(|| T::one()))
                    * cos2_sin,
                safe_f64_to_float::<T>(3.0)
                    .unwrap_or_else(|_| T::from_f64(3.0).unwrap_or_else(|| T::one()))
                    * cos_sin2,
                sin3,
            ]
        }
        _ => vec![T::one()], // Fallback
    }
}

/// Apply a bilateral edge-preserving gradient filter
///
/// This filter computes gradients while preserving edges using bilateral filtering principles.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `spatial_sigma` - Spatial standard deviation for bilateral component
/// * `range_sigma` - Range standard deviation for bilateral component
/// * `direction` - Gradient direction ('x', 'y', or 'magnitude')
///
/// # Returns
///
/// * `Result<Array2<T>>` - Gradient-filtered array
#[allow(dead_code)]
pub fn bilateral_gradient_filter<T>(
    input: &ArrayView2<T>,
    spatial_sigma: T,
    range_sigma: T,
    direction: char,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

    let window_size = (spatial_sigma
        * safe_f64_to_float::<T>(6.0)
            .unwrap_or_else(|_| T::from_f64(6.0).unwrap_or_else(|| T::one())))
    .to_usize()
    .unwrap_or(7);
    let window_size = if window_size % 2 == 0 {
        window_size + 1
    } else {
        window_size
    };
    let half_window = window_size / 2;

    let spatial_factor = -safe_f64_to_float::<T>(0.5)
        .unwrap_or_else(|_| T::from_f64(0.5).unwrap_or_else(|| T::one()))
        / (spatial_sigma * spatial_sigma);
    let range_factor = -safe_f64_to_float::<T>(0.5)
        .unwrap_or_else(|_| T::from_f64(0.5).unwrap_or_else(|| T::one()))
        / (range_sigma * range_sigma);

    // Process each pixel
    for i in half_window..height - half_window {
        for j in half_window..width - half_window {
            let center_val = input[[i, j]];
            let mut weighted_sum = T::zero();
            let mut weight_sum = T::zero();

            for di in 0..window_size {
                for dj in 0..window_size {
                    let ni = i - half_window + di;
                    let nj = j - half_window + dj;

                    let neighbor_val = input[[ni, nj]];

                    // Spatial weight
                    let dx = safe_isize_to_float((dj as isize) - (half_window as isize))
                        .unwrap_or_else(|_| T::zero());
                    let dy = safe_isize_to_float((di as isize) - (half_window as isize))
                        .unwrap_or_else(|_| T::zero());
                    let spatial_dist = dx * dx + dy * dy;
                    let spatial_weight = (spatial_dist * spatial_factor).exp();

                    // Range weight
                    let range_diff = neighbor_val - center_val;
                    let range_weight = (range_diff * range_diff * range_factor).exp();

                    let total_weight = spatial_weight * range_weight;

                    // Apply gradient computation based on direction
                    let gradient_val = match direction {
                        'x' => dx * neighbor_val,
                        'y' => dy * neighbor_val,
                        _ => (dx * dx + dy * dy).sqrt() * neighbor_val, // magnitude
                    };

                    weighted_sum = weighted_sum + total_weight * gradient_val;
                    weight_sum = weight_sum + total_weight;
                }
            }

            output[[i, j]] = if weight_sum > T::zero() {
                weighted_sum / weight_sum
            } else {
                T::zero()
            };
        }
    }

    Ok(output)
}

/// Apply anisotropic diffusion filtering for edge-preserving smoothing
///
/// Anisotropic diffusion is a technique for smoothing images while preserving edges.
/// It reduces noise while keeping important image features intact.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `num_iterations` - Number of diffusion iterations
/// * `kappa` - Diffusion constant (controls edge preservation)
/// * `gamma` - Step size for numerical integration
/// * `option` - Diffusion function option (1 or 2)
///
/// # Returns
///
/// * `Result<Array2<T>>` - Filtered array
#[allow(dead_code)]
pub fn anisotropic_diffusion<T>(
    input: &ArrayView2<T>,
    num_iterations: usize,
    kappa: T,
    gamma: T,
    option: usize,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    if option != 1 && option != 2 {
        return Err(NdimageError::InvalidInput(
            "Diffusion option must be 1 or 2".into(),
        ));
    }

    let (height, width) = input.dim();
    let mut image = input.to_owned();

    let gamma_quarter = gamma
        / safe_f64_to_float::<T>(4.0)
            .unwrap_or_else(|_| T::from_f64(4.0).unwrap_or_else(|| T::one()));

    for _ in 0..num_iterations {
        let mut newimage = image.clone();

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                // Compute gradients
                let grad_n = image[[i - 1, j]] - image[[i, j]];
                let grad_s = image[[i + 1, j]] - image[[i, j]];
                let grad_e = image[[i, j + 1]] - image[[i, j]];
                let grad_w = image[[i, j - 1]] - image[[i, j]];

                // Compute diffusion coefficients
                let c_n = diffusion_function(grad_n, kappa, option);
                let c_s = diffusion_function(grad_s, kappa, option);
                let c_e = diffusion_function(grad_e, kappa, option);
                let c_w = diffusion_function(grad_w, kappa, option);

                // Update pixel value
                let update =
                    gamma_quarter * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w);
                newimage[[i, j]] = image[[i, j]] + update;
            }
        }

        image = newimage;
    }

    Ok(image)
}

/// Diffusion function for anisotropic diffusion
#[allow(dead_code)]
fn diffusion_function<T>(gradient: T, kappa: T, option: usize) -> T
where
    T: Float + FromPrimitive,
{
    match option {
        1 => {
            // Exponential diffusion function
            let ratio = gradient / kappa;
            (-ratio * ratio).exp()
        }
        2 => {
            // Rational diffusion function
            let ratio = gradient / kappa;
            T::one() / (T::one() + ratio * ratio)
        }
        _ => T::one(),
    }
}

/// Apply non-local means denoising filter
///
/// Non-local means is a powerful denoising algorithm that preserves texture
/// and fine details by comparing patches throughout the image.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `h` - Filtering parameter (higher h removes more noise but also removes image details)
/// * `patch_size` - Size of patches for comparison
/// * `search_window` - Size of search window around each pixel
///
/// # Returns
///
/// * `Result<Array2<T>>` - Denoised array
#[allow(dead_code)]
pub fn non_local_means<T>(
    input: &ArrayView2<T>,
    h: T,
    patch_size: usize,
    search_window: usize,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

    let patch_radius = patch_size / 2;
    let search_radius = search_window / 2;
    let h_squared = h * h;
    let patch_area = safe_usize_to_float(patch_size * patch_size).unwrap_or_else(|_| T::one());

    for i in patch_radius..height - patch_radius {
        for j in patch_radius..width - patch_radius {
            let mut weighted_sum = T::zero();
            let mut weight_sum = T::zero();

            // Search _window bounds
            let search_i_min = (i.saturating_sub(search_radius)).max(patch_radius);
            let search_i_max = (i + search_radius + 1).min(height - patch_radius);
            let search_j_min = (j.saturating_sub(search_radius)).max(patch_radius);
            let search_j_max = (j + search_radius + 1).min(width - patch_radius);

            for search_i in search_i_min..search_i_max {
                for search_j in search_j_min..search_j_max {
                    // Compute patch distance
                    let mut patch_distance = T::zero();

                    for di in 0..patch_size {
                        for dj in 0..patch_size {
                            let pi = i - patch_radius + di;
                            let pj = j - patch_radius + dj;
                            let si = search_i - patch_radius + di;
                            let sj = search_j - patch_radius + dj;

                            if pi < height && pj < width && si < height && sj < width {
                                let diff = input[[pi, pj]] - input[[si, sj]];
                                patch_distance = patch_distance + diff * diff;
                            }
                        }
                    }

                    patch_distance = patch_distance / patch_area;

                    // Compute weight
                    let weight = (-patch_distance.max(T::zero()) / h_squared).exp();

                    weighted_sum = weighted_sum + weight * input[[search_i, search_j]];
                    weight_sum = weight_sum + weight;
                }
            }

            output[[i, j]] = if weight_sum > T::zero() {
                weighted_sum / weight_sum
            } else {
                input[[i, j]]
            };
        }
    }

    Ok(output)
}

/// Apply adaptive Wiener filtering for noise reduction
///
/// Adaptive Wiener filter adjusts its behavior based on local image statistics,
/// providing good noise reduction while preserving edges and details.
///
/// # Arguments
///
/// * `input` - Input 2D array (noisy image)
/// * `noise_variance` - Estimated noise variance
/// * `window_size` - Size of local analysis window
///
/// # Returns
///
/// * `Result<Array2<T>>` - Filtered array
#[allow(dead_code)]
pub fn adaptive_wiener_filter<T>(
    input: &ArrayView2<T>,
    noise_variance: T,
    window_size: usize,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

    let half_window = window_size / 2;

    for i in half_window..height - half_window {
        for j in half_window..width - half_window {
            // Extract local window
            let mut local_values = Vec::with_capacity(window_size * window_size);

            for di in 0..window_size {
                for dj in 0..window_size {
                    let y = i - half_window + di;
                    let x = j - half_window + dj;
                    local_values.push(input[[y, x]]);
                }
            }

            // Compute local statistics
            let local_mean = local_values
                .iter()
                .cloned()
                .fold(T::zero(), |acc, x| acc + x)
                / safe_usize_to_float(local_values.len()).unwrap_or_else(|_| T::one());

            let local_variance = local_values
                .iter()
                .map(|&x| (x - local_mean) * (x - local_mean))
                .fold(T::zero(), |acc, x| acc + x)
                / safe_usize_to_float(local_values.len()).unwrap_or_else(|_| T::one());

            // Wiener filter coefficient
            let signal_variance = (local_variance - noise_variance).max(T::zero());
            let wiener_gain = if local_variance > T::zero() {
                signal_variance / local_variance
            } else {
                T::zero()
            };

            // Apply filter
            let center_value = input[[i, j]];
            output[[i, j]] = local_mean + wiener_gain * (center_value - local_mean);
        }
    }

    Ok(output)
}

/// Apply shock filter for edge enhancement
///
/// Shock filters enhance edges by applying different smoothing/sharpening
/// based on the local image structure (edges vs. smooth regions).
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `num_iterations` - Number of iterations
/// * `dt` - Time step
/// * `sigma` - Gaussian smoothing parameter for structure tensor
///
/// # Returns
///
/// * `Result<Array2<T>>` - Enhanced array
#[allow(dead_code)]
pub fn shock_filter<T>(
    input: &ArrayView2<T>,
    num_iterations: usize,
    dt: T,
    sigma: T,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let mut image = input.to_owned();

    for _ in 0..num_iterations {
        let newimage = shock_filter_iteration(&image.view(), dt, sigma)?;
        image = newimage;
    }

    Ok(image)
}

/// Single iteration of shock filter
#[allow(dead_code)]
fn shock_filter_iteration<T>(input: &ArrayView2<T>, dt: T, sigma: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (height, width) = input.dim();
    let mut output = input.to_owned();

    // Compute smoothed version for structure analysis
    // Convert to f64 for gaussian_filter
    let input_f64 = input.mapv(|x| x.to_f64().unwrap_or(0.0));
    let sigma_f64 = sigma.to_f64().unwrap_or(1.0);
    let smoothed_f64 = crate::filters::gaussian_filter(&input_f64, sigma_f64, None, None)?;
    // Convert back to T
    let smoothed = smoothed_f64.mapv(|x| T::from_f64(x).unwrap_or_else(|| T::zero()));

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // Compute gradients on smoothed image
            let grad_x = (smoothed[[i, j + 1]] - smoothed[[i, j - 1]])
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));
            let grad_y = (smoothed[[i + 1, j]] - smoothed[[i - 1, j]])
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));
            let grad_magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();

            if grad_magnitude > T::zero() {
                // Compute directional derivatives
                let grad_x_norm = grad_x / grad_magnitude;
                let grad_y_norm = grad_y / grad_magnitude;

                // Second directional derivative (along gradient direction)
                let uxx = input[[i, j + 1]]
                    - safe_f64_to_float::<T>(2.0)
                        .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()))
                        * input[[i, j]]
                    + input[[i, j - 1]];
                let uyy = input[[i + 1, j]]
                    - safe_f64_to_float::<T>(2.0)
                        .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()))
                        * input[[i, j]]
                    + input[[i - 1, j]];
                let uxy = (input[[i + 1, j + 1]] - input[[i + 1, j - 1]] - input[[i - 1, j + 1]]
                    + input[[i - 1, j - 1]])
                    / safe_f64_to_float::<T>(4.0)
                        .unwrap_or_else(|_| T::from_f64(4.0).unwrap_or_else(|| T::one()));

                let directional_curvature = grad_x_norm * grad_x_norm * uxx
                    + safe_f64_to_float::<T>(2.0)
                        .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()))
                        * grad_x_norm
                        * grad_y_norm
                        * uxy
                    + grad_y_norm * grad_y_norm * uyy;

                // Shock filter update: enhance if curvature is negative (edge), smooth if positive
                let sign = if directional_curvature < T::zero() {
                    -T::one()
                } else {
                    T::one()
                };
                let update = dt * sign * grad_magnitude;

                output[[i, j]] = input[[i, j]] + update;
            }
        }
    }

    Ok(output)
}

/// Apply coherence enhancing diffusion for line-like structure enhancement
///
/// This filter enhances line-like structures (such as blood vessels, fibers)
/// while smoothing perpendicular to these structures.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `num_iterations` - Number of diffusion iterations
/// * `alpha` - Diffusion strength parameter
/// * `sigma` - Pre-smoothing parameter
/// * `c` - Contrast parameter
///
/// # Returns
///
/// * `Result<Array2<T>>` - Enhanced array
#[allow(dead_code)]
pub fn coherence_enhancing_diffusion<T>(
    input: &ArrayView2<T>,
    num_iterations: usize,
    alpha: T,
    sigma: T,
    c: T,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let mut image = input.to_owned();

    // Pre-smooth the image
    // Convert to f64 for gaussian_filter
    let image_f64 = image.mapv(|x| x.to_f64().unwrap_or(0.0));
    let sigma_f64 = sigma.to_f64().unwrap_or(1.0);
    let smoothed_f64 = crate::filters::gaussian_filter(&image_f64, sigma_f64, None, None)?;
    // Convert back to T
    image = smoothed_f64.mapv(|x| T::from_f64(x).unwrap_or_else(|| T::zero()));

    for _ in 0..num_iterations {
        let newimage = ced_iteration(&image.view(), alpha, c)?;
        image = newimage;
    }

    Ok(image)
}

/// Single iteration of coherence enhancing diffusion
#[allow(dead_code)]
fn ced_iteration<T>(input: &ArrayView2<T>, alpha: T, c: T) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone,
{
    let (height, width) = input.dim();
    let mut output = input.to_owned();

    let dt = safe_f64_to_float::<T>(0.25)
        .unwrap_or_else(|_| T::from_f64(0.25).unwrap_or_else(|| T::one())); // Time step

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // Compute structure tensor
            let grad_x = (input[[i, j + 1]] - input[[i, j - 1]])
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));
            let grad_y = (input[[i + 1, j]] - input[[i - 1, j]])
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));

            let j11 = grad_x * grad_x;
            let j12 = grad_x * grad_y;
            let j22 = grad_y * grad_y;

            // Compute eigenvalues of structure tensor
            let trace = j11 + j22;
            let det = j11 * j22 - j12 * j12;
            let discriminant = (trace * trace
                - safe_f64_to_float::<T>(4.0)
                    .unwrap_or_else(|_| T::from_f64(4.0).unwrap_or_else(|| T::one()))
                    * det)
                .max(T::zero())
                .sqrt();

            let lambda1 = (trace + discriminant)
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));
            let lambda2 = (trace - discriminant)
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));

            // Compute diffusivities
            let coherence = if lambda1 > T::zero() {
                (lambda1 - lambda2) / lambda1
            } else {
                T::zero()
            };

            let d1 = alpha;
            let d2 = alpha + (T::one() - alpha) * (-(coherence * coherence / (c * c))).exp();

            // Compute diffusion update (simplified)
            let laplacian =
                input[[i + 1, j]] + input[[i - 1, j]] + input[[i, j + 1]] + input[[i, j - 1]]
                    - safe_f64_to_float::<T>(4.0)
                        .unwrap_or_else(|_| T::from_f64(4.0).unwrap_or_else(|| T::one()))
                        * input[[i, j]];

            // Average diffusivity for simplification
            let avg_diffusivity = (d1 + d2)
                / safe_f64_to_float::<T>(2.0)
                    .unwrap_or_else(|_| T::from_f64(2.0).unwrap_or_else(|| T::one()));

            output[[i, j]] = input[[i, j]] + dt * avg_diffusivity * laplacian;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gabor_filter() {
        let input = array![
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ];

        let params = GaborParams {
            wavelength: 4.0,
            orientation: 0.0,
            sigma_x: 1.0,
            sigma_y: 1.0,
            phase: 0.0,
            aspect_ratio: None,
        };

        let result = gabor_filter(&input.view(), &params, Some(5), None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_gabor_filter_bank() {
        let input = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];

        let params = GaborParams::default();
        let results = gabor_filter_bank(&input.view(), &params, 4, Some(3), None).unwrap();

        assert_eq!(results.len(), 4);
        for result in results {
            assert_eq!(result.shape(), input.shape());
        }
    }

    #[test]
    fn test_steerable_filter() {
        let input = array![[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]];

        let result =
            steerable_filter(&input.view(), 1, std::f64::consts::PI / 4.0, 1.0, None).unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_bilateral_gradient_filter() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = bilateral_gradient_filter(&input.view(), 1.0, 2.0, 'x').unwrap();
        assert_eq!(result.shape(), input.shape());
    }
}
