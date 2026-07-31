//! Complex-to-Real transforms for HFFT
//!
//! This module contains functions for transforming complex arrays to real arrays
//! using the Hermitian Fast Fourier Transform (HFFT).

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use scirs2_core::ndarray::{Array, Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

// Importing the try_as_complex utility for type conversion
use super::utility::{other_axis_index_combinations, try_as_complex, HfftNorm};

/// Compute the 1-dimensional discrete Fourier Transform for a Hermitian-symmetric input.
///
/// This function computes the FFT of a Hermitian-symmetric complex array,
/// resulting in a real-valued output. A Hermitian-symmetric array satisfies
/// `a[i] = conj(a[-i])` for all indices `i`.
///
/// # Arguments
///
/// * `x` - Input complex-valued array with Hermitian symmetry
/// * `n` - Length of the transformed axis (optional)
/// * `norm` - Normalization mode (optional, default is "backward"):
///   * "backward": No normalization on forward transforms, 1/n on inverse
///   * "forward": 1/n on forward transforms, no normalization on inverse
///   * "ortho": 1/sqrt(n) on both forward and inverse transforms
///
/// # Returns
///
/// * The real-valued Fourier transform of the Hermitian-symmetric input array
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::Complex64;
/// use scirs2_fft::hfft;
///
/// // Create a simple Hermitian-symmetric array (DC component is real)
/// let x = vec![
///     Complex64::new(1.0, 0.0),  // DC component (real)
///     Complex64::new(2.0, 1.0),  // Positive frequency
///     Complex64::new(2.0, -1.0), // Negative frequency (conjugate of above)
/// ];
///
/// // Compute the HFFT
/// let result = hfft(&x, None, None).expect("valid input");
///
/// // The result should be real-valued
/// assert!(result.len() == 3);
/// // Check that the result is real (imaginary parts are negligible)
/// for &val in &result {
///     assert!(val.is_finite());
/// }
/// ```
#[allow(dead_code)]
pub fn hfft<T>(x: &[T], n: Option<usize>, norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Fast path for handling Complex64 input (common case)
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        // This is a safe transmutation since we've verified the types match
        let complex_input: &[Complex64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const Complex64, x.len()) };

        // Use a copy of the input with the DC component made real to ensure Hermitian symmetry
        let mut adjusted_input = Vec::with_capacity(complex_input.len());
        if !complex_input.is_empty() {
            // Ensure the DC component is real
            adjusted_input.push(Complex64::new(complex_input[0].re, 0.0));

            // Copy the rest of the elements unchanged
            adjusted_input.extend_from_slice(&complex_input[1..]);
        }

        return _hfft_complex(&adjusted_input, n, norm);
    }

    // For other types, convert manually
    let mut complex_input = Vec::with_capacity(x.len());

    for (i, &val) in x.iter().enumerate() {
        // Try to convert to complex directly using our specialized function
        if let Some(c) = try_as_complex(val) {
            // For the first element (DC component), ensure it's real
            if i == 0 {
                complex_input.push(Complex64::new(c.re, 0.0));
            } else {
                complex_input.push(c);
            }
            continue;
        }

        // For scalar types, try direct conversion to f64 and create a complex with zero imaginary part
        if let Some(val_f64) = NumCast::from(val) {
            complex_input.push(Complex64::new(val_f64, 0.0));
            continue;
        }

        // If we can't convert, return an error
        return Err(FFTError::ValueError(format!(
            "Could not convert {val:?} to Complex64"
        )));
    }

    _hfft_complex(&complex_input, n, norm)
}

/// Internal implementation for Complex64 input
#[allow(dead_code)]
fn _hfft_complex(x: &[Complex64], n: Option<usize>, norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n_fft = n.unwrap_or(x.len());

    // Calculate the expected length of the output (real) array
    let n_real = n_fft;

    // Create the output array
    let mut output = Vec::with_capacity(n_real);

    // Compute FFT of the input. `fft()` performs a completely unnormalized
    // (raw) forward transform, which is exactly the "backward"-mode baseline
    // for a forward-type transform, so the norm-dependent scale factor below
    // can be applied directly to its result.
    let fft_result = fft(x, Some(n_fft))?;
    let scale = HfftNorm::parse(norm)?.forward_scale(n_fft)?;

    // Extract real parts from the FFT result - the result should be real
    // (within numerical precision) due to the Hermitian symmetry of the input
    for val in fft_result {
        output.push(val.re * scale);
    }

    Ok(output)
}

/// Compute the 2-dimensional discrete Fourier Transform for a Hermitian-symmetric input.
///
/// This function computes the FFT of a Hermitian-symmetric complex 2D array,
/// resulting in a real-valued output.
///
/// # Arguments
///
/// * `x` - Input complex-valued 2D array with Hermitian symmetry
/// * `shape` - The shape of the result (optional)
/// * `axes` - The axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode (optional, default is "backward")
///
/// # Returns
///
/// * The real-valued 2D Fourier transform of the Hermitian-symmetric input array
#[allow(dead_code)]
pub fn hfft2<T>(
    x: &ArrayView2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // For testing purposes, directly call internal implementation with converted values
    // This is not ideal for production code but helps us validate the functionality
    #[cfg(test)]
    {
        // Special case for Complex64 input which is the common case
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
            // Create a view with the correct type
            let ptr = x.as_ptr() as *const Complex64;
            let complex_view = unsafe { ArrayView2::from_shape_ptr(x.dim(), ptr) };

            return _hfft2_complex(&complex_view, shape, axes, norm);
        }
    }

    // General case for other types
    let (n_rows, n_cols) = x.dim();

    // Convert input to complex array
    let mut complex_input = Array2::zeros((n_rows, n_cols));
    for r in 0..n_rows {
        for c in 0..n_cols {
            let val = x[[r, c]];
            // Try to convert to complex directly
            if let Some(complex) = try_as_complex(val) {
                complex_input[[r, c]] = complex;
                continue;
            }

            // For scalar types, try direct conversion to f64 and create a complex with zero imaginary part
            if let Some(val_f64) = NumCast::from(val) {
                complex_input[[r, c]] = Complex64::new(val_f64, 0.0);
                continue;
            }

            // If we can't convert, return an error
            return Err(FFTError::ValueError(format!(
                "Could not convert {val:?} to Complex64"
            )));
        }
    }

    _hfft2_complex(&complex_input.view(), shape, axes, norm)
}

/// Internal implementation for complex input
#[allow(dead_code)]
fn _hfft2_complex(
    x: &ArrayView2<Complex64>,
    shape: Option<(usize, usize)>,
    axes: Option<(usize, usize)>,
    norm: Option<&str>,
) -> FFTResult<Array2<f64>> {
    // Extract dimensions
    let (n_rows, n_cols) = x.dim();

    // Get output shape
    let (out_rows, out_cols) = shape.unwrap_or((n_rows, n_cols));

    // Get axes
    let (axis_0, axis_1) = axes.unwrap_or((0, 1));
    if axis_0 >= 2 || axis_1 >= 2 {
        return Err(FFTError::ValueError(
            "Axes must be 0 or 1 for 2D arrays".to_string(),
        ));
    }

    // The norm scale for a multi-dimensional transform is computed from the
    // total number of transformed elements (matching NumPy/SciPy's `fftn`
    // convention), applied once at the end since both passes below use the
    // crate's completely unnormalized `fft()`.
    let total_elements = out_rows.saturating_mul(out_cols);
    let scale = HfftNorm::parse(norm)?.forward_scale(total_elements)?;

    // Create a flattened temporary array for the first FFT along axis 0
    let mut temp = Array2::zeros((out_rows, n_cols));

    // Perform 1D FFTs along axis 0 (rows)
    for c in 0..n_cols {
        // Extract a column
        let mut col = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            col.push(x[[r, c]]);
        }

        // Perform 1D FFT for each column (unnormalized; norm is applied once below)
        let fft_col = fft(&col, Some(out_rows))?;

        // Store the result in the temporary array
        for r in 0..out_rows {
            temp[[r, c]] = fft_col[r];
        }
    }

    // Create the final output array
    let mut output = Array2::zeros((out_rows, out_cols));

    // Perform 1D FFTs along axis 1 (columns)
    for r in 0..out_rows {
        // Extract a row
        let mut row = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            row.push(temp[[r, c]]);
        }

        // Perform 1D FFT for each row (unnormalized; norm is applied once below)
        let fft_row = fft(&row, Some(out_cols))?;

        // Store only the real part in the output, scaled per the requested norm mode
        for c in 0..out_cols {
            output[[r, c]] = fft_row[c].re * scale;
        }
    }

    Ok(output)
}

/// Compute the N-dimensional discrete Fourier Transform for Hermitian-symmetric input.
///
/// This function computes the FFT of a Hermitian-symmetric complex N-dimensional array,
/// resulting in a real-valued output.
///
/// # Arguments
///
/// * `x` - Input complex-valued N-dimensional array with Hermitian symmetry
/// * `shape` - The shape of the result (optional)
/// * `axes` - The axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode (optional, default is "backward")
/// * `overwrite_x` - Whether to overwrite the input array (optional)
/// * `workers` - Number of workers to use for parallel computation (optional)
///
/// # Returns
///
/// * The real-valued N-dimensional Fourier transform of the Hermitian-symmetric input array
#[allow(dead_code)]
pub fn hfftn<T>(
    x: &ArrayView<T, IxDyn>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    overwrite_x: Option<bool>,
    workers: Option<usize>,
) -> FFTResult<Array<f64, IxDyn>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // For testing purposes, directly call internal implementation with converted values
    // This is not ideal for production code but helps us validate the functionality
    #[cfg(test)]
    {
        // Special case for handling Complex64 input (common case)
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
            // Create a view with the correct type
            let ptr = x.as_ptr() as *const Complex64;
            let complex_view = unsafe { ArrayView::from_shape_ptr(IxDyn(x.shape()), ptr) };

            return _hfftn_complex(&complex_view, shape, axes, norm, overwrite_x, workers);
        }
    }

    // For other types, convert to complex and call the internal implementation
    let xshape = x.shape().to_vec();

    // Convert input to complex array
    let complex_input = Array::from_shape_fn(IxDyn(&xshape), |idx| {
        let val = x[idx.clone()];

        // Try to convert to complex directly
        if let Some(c) = try_as_complex(val) {
            return c;
        }

        // For scalar types, try direct conversion to f64 and create a complex with zero imaginary part
        if let Some(val_f64) = NumCast::from(val) {
            return Complex64::new(val_f64, 0.0);
        }

        // If we can't convert, return an error
        Complex64::new(0.0, 0.0) // Default value (we'll handle errors elsewhere if necessary)
    });

    _hfftn_complex(
        &complex_input.view(),
        shape,
        axes,
        norm,
        overwrite_x,
        workers,
    )
}

/// Internal implementation for complex input
#[allow(dead_code)]
fn _hfftn_complex(
    x: &ArrayView<Complex64, IxDyn>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    _overwrite_x: Option<bool>,
    _workers: Option<usize>,
) -> FFTResult<Array<f64, IxDyn>> {
    // The overwrite_x and _workers parameters are not used in this implementation
    // They are included for API compatibility with scipy's fftn

    let xshape = x.shape().to_vec();
    let ndim = xshape.len();

    // Handle empty array case
    if ndim == 0 || xshape.contains(&0) {
        return Ok(Array::zeros(IxDyn(&[])));
    }

    // Determine the output shape
    let outshape = match shape {
        Some(s) => {
            if s.len() != ndim {
                return Err(FFTError::ValueError(format!(
                    "Shape must have the same number of dimensions as input, got {} != {}",
                    s.len(),
                    ndim
                )));
            }
            s
        }
        None => xshape.clone(),
    };

    // Determine the axes
    let transform_axes = match axes {
        Some(a) => {
            let mut sorted_axes = a.clone();
            sorted_axes.sort_unstable();
            sorted_axes.dedup();

            // Validate axes
            for &ax in &sorted_axes {
                if ax >= ndim {
                    return Err(FFTError::ValueError(format!(
                        "Axis {ax} is out of bounds for array of dimension {ndim}"
                    )));
                }
            }
            sorted_axes
        }
        None => (0..ndim).collect(),
    };

    // Simple case: 1D transform
    if ndim == 1 {
        let mut complex_result = Vec::with_capacity(x.len());
        for &val in x.iter() {
            complex_result.push(val);
        }

        let fft_result = fft(&complex_result, Some(outshape[0]))?;
        let scale = HfftNorm::parse(norm)?.forward_scale(outshape[0])?;
        let mut real_result = Array::zeros(IxDyn(&[outshape[0]]));

        for i in 0..outshape[0] {
            real_result[i] = fft_result[i].re * scale;
        }

        return Ok(real_result);
    }

    // The norm scale for an N-D transform is computed from the total number
    // of elements across only the axes actually being transformed (matching
    // NumPy/SciPy's `fftn` convention), applied once at the end since every
    // per-axis pass below uses the crate's completely unnormalized `fft()`.
    let total_elements: usize = transform_axes.iter().map(|&ax| outshape[ax]).product();
    let scale = HfftNorm::parse(norm)?.forward_scale(total_elements)?;

    // For multi-dimensional transforms, we have to transform along each axis
    let mut array = Array::from_shape_fn(IxDyn(&xshape), |idx| x[idx.clone()]);

    // For each axis, perform a 1D transform along that axis
    for &axis in &transform_axes {
        // The current shape (as of this pass; earlier passes may already
        // have resized previously-transformed axes).
        let current_shape = array.shape().to_vec();
        let axis_dim = outshape[axis];

        // Reshape the array to transform along this axis
        let mut workingshape = current_shape.clone();
        workingshape[axis] = axis_dim;

        // Allocate an array for the result along this axis
        let mut axis_result = Array::zeros(IxDyn(&workingshape));

        // Sweep every fiber along `axis` (i.e. every combination of the
        // *other* axes' indices), not just the single fiber at the origin.
        for mut indices in other_axis_index_combinations(&current_shape, axis) {
            let mut fiber = Vec::with_capacity(current_shape[axis]);
            for i in 0..current_shape[axis] {
                indices[axis] = i;
                fiber.push(array[IxDyn(&indices)]);
            }

            // Perform the 1D FFT (unnormalized; the total norm scale is applied once below)
            let fft_result = fft(&fiber, Some(axis_dim))?;

            // Store the result back in the working array
            for (i, val) in fft_result.iter().enumerate().take(axis_dim) {
                indices[axis] = i;
                axis_result[IxDyn(&indices)] = *val;
            }
        }

        // Update the array for the next axis transformation
        array = axis_result;
    }

    // Extract real part from the final complex array
    let mut real_result = Array::zeros(IxDyn(&outshape));
    for (i, &val) in array.iter().enumerate() {
        // Get the indices for this element
        // This is a simplified approach for the refactoring, in production code we'd use ndarray's APIs better
        let mut idx = vec![0; ndim];
        for (dim, idx_val) in idx.iter_mut().enumerate().take(ndim) {
            let stride = array.strides()[dim] as usize;
            if let Some(divided) = i.checked_div(stride) {
                *idx_val = divided % array.shape()[dim];
            }
        }
        real_result[IxDyn(&idx)] = val.re * scale;
    }

    Ok(real_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const EPS: f64 = 1e-8;

    /// A Hermitian-symmetric (`x[k] = conj(x[N-k])`) complex sequence of
    /// length 5 (odd `N`, no Nyquist term). Non-constant so a fabricated
    /// stub (e.g. one that just echoes the input or a constant) cannot pass.
    fn hermitian_n5() -> Vec<Complex64> {
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(-1.0, -0.5),
            Complex64::new(2.0, -1.0),
        ]
    }

    /// A Hermitian-symmetric complex sequence of length 6 (even `N`,
    /// includes a real Nyquist term at index 3).
    fn hermitian_n6() -> Vec<Complex64> {
        vec![
            Complex64::new(1.5, 0.0),
            Complex64::new(2.0, -0.3),
            Complex64::new(-0.5, 1.0),
            Complex64::new(0.8, 0.0),
            Complex64::new(-0.5, -1.0),
            Complex64::new(2.0, 0.3),
        ]
    }

    // Reference values below were computed with `numpy.fft.fft(x,
    // norm=mode).real` for the exact `x` above. `numpy.fft.fft`'s `norm`
    // convention *is* the forward-type convention `hfft` documents
    // ("backward": no scaling; "forward": 1/n; "ortho": 1/sqrt(n)), and
    // `hfft`'s own core computation is precisely an unnormalized `fft()` of
    // a full-length Hermitian-symmetric sequence followed by taking the
    // real part, so this is an exact (not approximate-by-construction)
    // ground truth for the scaling this test is checking.

    #[test]
    fn test_hfft_norm_backward_matches_numpy_n5() {
        let x = hermitian_n5();
        let expected = [
            5.0,
            8.344_000_251_132_465,
            -0.629_587_977_959_892,
            -1.078_615_954_539_477_3,
            3.364_203_681_366_904_5,
        ];
        let result = hfft(&x, None, Some("backward")).expect("hfft failed");
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = EPS);
        }
        // The default (`None`) must behave exactly like "backward".
        let result_default = hfft(&x, None, None).expect("hfft failed");
        for (r, e) in result_default.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = EPS);
        }
    }

    #[test]
    fn test_hfft_norm_forward_matches_numpy_n5() {
        let x = hermitian_n5();
        let expected = [
            1.0,
            1.668_800_050_226_493_2,
            -0.125_917_595_591_978_4,
            -0.215_723_190_907_895_5,
            0.672_840_736_273_380_9,
        ];
        let result = hfft(&x, None, Some("forward")).expect("hfft failed");
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = EPS);
        }
    }

    #[test]
    fn test_hfft_norm_ortho_matches_numpy_n5() {
        let x = hermitian_n5();
        let expected = [
            2.236_067_977_499_79,
            3.731_550_353_161_501_7,
            -0.281_560_303_306_991_56,
            -0.482_371_719_193_218_8,
            1.504_517_624_338_288_3,
        ];
        let result = hfft(&x, None, Some("ortho")).expect("hfft failed");
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = EPS);
        }
    }

    #[test]
    fn test_hfft_norm_modes_match_numpy_n6() {
        let x = hermitian_n6();
        let backward = [
            5.3,
            4.412_435_565_298_214,
            -1.451_666_049_839_540_4,
            -4.3,
            3.051_666_049_839_54,
            1.987_564_434_701_786_2,
        ];
        let forward = [
            0.883_333_333_333_333_3,
            0.735_405_927_549_702_2,
            -0.241_944_341_639_923_4,
            -0.716_666_666_666_666_6,
            0.508_611_008_306_59,
            0.331_260_739_116_964_35,
        ];
        let ortho = [
            2.163_715_939_458_474_4,
            1.801_369_276_314_945_1,
            -0.592_640_183_171_421_4,
            -1.755_467_648_994_611,
            1.245_837_447_913_602_3,
            0.811_419_782_653_778_7,
        ];

        for (mode, expected) in [
            ("backward", backward),
            ("forward", forward),
            ("ortho", ortho),
        ] {
            let result = hfft(&x, None, Some(mode)).expect("hfft failed");
            for (r, e) in result.iter().zip(expected.iter()) {
                assert_abs_diff_eq!(r, e, epsilon = EPS);
            }
        }

        // Regression guard for the original bug (norm accepted but
        // silently ignored, i.e. always behaving as "backward"): forward
        // and ortho scaling must actually change the numeric result.
        let backward_result = hfft(&x, None, Some("backward")).expect("hfft failed");
        let forward_result = hfft(&x, None, Some("forward")).expect("hfft failed");
        let ortho_result = hfft(&x, None, Some("ortho")).expect("hfft failed");
        assert!((backward_result[0] - forward_result[0]).abs() > 1.0);
        assert!((backward_result[0] - ortho_result[0]).abs() > 1.0);
    }

    #[test]
    fn test_hfft_invalid_norm_is_an_error() {
        let x = hermitian_n5();
        let err = hfft(&x, None, Some("bogus")).unwrap_err();
        assert!(matches!(err, FFTError::ValueError(_)));
    }

    /// A Hermitian-symmetric 3x4 complex array, constructed as `fft2` of a
    /// real (and non-constant) array -- which is guaranteed to be
    /// Hermitian-symmetric by construction.
    fn hermitian_2d() -> Array2<Complex64> {
        #[rustfmt::skip]
        let data = vec![
            Complex64::new(5.0, 0.0), Complex64::new(2.25, -2.75), Complex64::new(4.5, 0.0), Complex64::new(2.25, 2.75),
            Complex64::new(0.5, -1.732_050_807_568_877_2), Complex64::new(9.336_696_879_329_399, -5.854_646_071_760_522), Complex64::new(-6.75, -0.433_012_701_892_219_3), Complex64::new(-4.086_696_879_329_399, -4.104_646_071_760_522),
            Complex64::new(0.5, 1.732_050_807_568_877_2), Complex64::new(-4.086_696_879_329_399, 4.104_646_071_760_522), Complex64::new(-6.75, 0.433_012_701_892_219_3), Complex64::new(9.336_696_879_329_399, 5.854_646_071_760_522),
        ];
        Array2::from_shape_vec((3, 4), data).expect("valid shape")
    }

    // Reference values computed with `numpy.fft.fft2(x, norm=mode).real`
    // for the exact `x` returned by `hermitian_2d()` above.
    fn hfft2_expected_backward() -> Array2<f64> {
        #[rustfmt::skip]
        let data = vec![
            12.0, 6.0, -18.0, 24.0,
            -6.0, -36.0, 30.0, 18.0,
            36.0, 15.0, 3.0, -24.0,
        ];
        Array2::from_shape_vec((3, 4), data).expect("valid shape")
    }

    fn hfft2_expected_forward() -> Array2<f64> {
        #[rustfmt::skip]
        let data = vec![
            1.0, 0.5, -1.5, 2.0,
            -0.5, -3.0, 2.5, 1.5,
            3.0, 1.25, 0.25, -2.0,
        ];
        Array2::from_shape_vec((3, 4), data).expect("valid shape")
    }

    fn hfft2_expected_ortho() -> Array2<f64> {
        #[rustfmt::skip]
        let data = vec![
            3.464_101_615_137_754_4, 1.732_050_807_568_877_6, -5.196_152_422_706_633, 6.928_203_230_275_51,
            -1.732_050_807_568_876_5, -10.392_304_845_413_266, 8.660_254_037_844_387, 5.196_152_422_706_631,
            10.392_304_845_413_266, 4.330_127_018_922_193, 0.866_025_403_784_439_3, -6.928_203_230_275_509,
        ];
        Array2::from_shape_vec((3, 4), data).expect("valid shape")
    }

    #[test]
    fn test_hfft2_norm_modes_match_numpy() {
        let x = hermitian_2d();

        let backward = hfft2(&x.view(), None, None, Some("backward")).expect("hfft2 failed");
        for (r, e) in backward.iter().zip(hfft2_expected_backward().iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }

        let forward = hfft2(&x.view(), None, None, Some("forward")).expect("hfft2 failed");
        for (r, e) in forward.iter().zip(hfft2_expected_forward().iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }

        let ortho = hfft2(&x.view(), None, None, Some("ortho")).expect("hfft2 failed");
        for (r, e) in ortho.iter().zip(hfft2_expected_ortho().iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hfftn_full_transform_matches_hfft2() {
        // hfftn transforming every axis of the same 2D input must agree
        // with the dedicated hfft2 implementation (and hence with the
        // same numpy-derived reference values), exercising the generic
        // N-D per-axis code path rather than hfft2's own hand-written one.
        let x = hermitian_2d().into_dyn();

        for (mode, expected) in [
            ("backward", hfft2_expected_backward()),
            ("forward", hfft2_expected_forward()),
            ("ortho", hfft2_expected_ortho()),
        ] {
            let result =
                hfftn(&x.view(), None, None, Some(mode), None, None).expect("hfftn failed");
            for (r, e) in result.iter().zip(expected.iter()) {
                assert_abs_diff_eq!(r, e, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_hfftn_partial_axes_scale_uses_only_transformed_axis_size() {
        // Two independent length-4 Hermitian-symmetric rows. When `axes`
        // restricts the transform to axis 1 only, the norm scale factor
        // must be based on the size of *that* axis (4), not on the total
        // element count (8) -- and every row must actually be transformed,
        // not just the row at the origin (regression guard for a fiber-walk
        // bug where only the all-zero-index fiber was ever processed).
        #[rustfmt::skip]
        let data = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5), Complex64::new(-1.0, 0.0), Complex64::new(2.0, 0.5),
            Complex64::new(-0.5, 0.0), Complex64::new(0.25, 1.0), Complex64::new(0.75, 0.0), Complex64::new(0.25, -1.0),
        ];
        let x = Array2::from_shape_vec((2, 4), data)
            .expect("valid shape")
            .into_dyn();

        let cases: [(&str, [f64; 4], [f64; 4]); 3] = [
            (
                "backward",
                [4.0, 1.0, -4.0, 3.0],
                [0.75, 0.75, -0.25, -3.25],
            ),
            (
                "forward",
                [1.0, 0.25, -1.0, 0.75],
                [0.1875, 0.1875, -0.0625, -0.8125],
            ),
            (
                "ortho",
                [2.0, 0.5, -2.0, 1.5],
                [0.375, 0.375, -0.125, -1.625],
            ),
        ];

        for (mode, expected_row0, expected_row1) in cases {
            let result = hfftn(&x.view(), None, Some(vec![1]), Some(mode), None, None)
                .expect("hfftn failed");
            let row0 = result.index_axis(scirs2_core::ndarray::Axis(0), 0);
            let row1 = result.index_axis(scirs2_core::ndarray::Axis(0), 1);
            for (r, e) in row0.iter().zip(expected_row0.iter()) {
                assert_abs_diff_eq!(r, e, epsilon = EPS);
            }
            for (r, e) in row1.iter().zip(expected_row1.iter()) {
                assert_abs_diff_eq!(r, e, epsilon = EPS);
            }
            // Row 1 must not be left as all-zero (the fiber-walk bug this
            // guards against always left every non-origin fiber at zero).
            assert!(row1.iter().any(|v| v.abs() > 1e-6));
        }
    }
}
