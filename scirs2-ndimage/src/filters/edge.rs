//! Edge detection filters for n-dimensional arrays

use scirs2_core::ndarray::{array, Array, Array2, Dimension, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::{convolve, BorderMode};
use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe conversion of hardcoded constants
#[allow(dead_code)]
fn safe_i32_to_float<T: Float + FromPrimitive>(value: i32) -> NdimageResult<T> {
    T::from_i32(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert i32 {} to float type", value))
    })
}

/// Apply a Sobel filter to calculate gradients in an n-dimensional array
///
/// The Sobel operator computes an approximation of the gradient of the image intensity function.
/// It is based on convolving the image with a small, separable, and integer-valued filter in the
/// horizontal and vertical directions, which emphasizes edges.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `axis` - Axis along which to calculate the gradient
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array with gradient values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{sobel, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Calculate gradient along x-axis (axis=1)
/// let gradient_x = sobel(&image, 1, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn sobel<T, D>(
    input: &Array<T, D>,
    axis: usize,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for sobel filter".into(),
        ));
    }

    if axis >= input.ndim() {
        return Err(NdimageError::InvalidInput(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // For 2D arrays, we can implement the Sobel operator directly
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Apply the 2D Sobel filter
        let result = match axis {
            0 => sobel_2d_x(&input_2d, &border_mode)?, // Vertical edges (y-derivative)
            1 => sobel_2d_y(&input_2d, &border_mode)?, // Horizontal edges (x-derivative)
            _ => {
                return Err(NdimageError::InvalidInput(format!(
                    "Invalid axis {} for 2D array, must be 0 or 1",
                    axis
                )));
            }
        };

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, apply separable 1D filters
        sobel_nd(input, axis, &border_mode)
    }
}

/// Apply a Laplace filter to detect edges in an n-dimensional array
///
/// The Laplacian operator is a 2D isotropic measure of the 2nd spatial derivative
/// of an image. It is used for edge detection. The Laplacian of an image highlights
/// regions of rapid intensity change, which is useful for detecting edges.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `diagonal` - Whether to include diagonal neighbors in the Laplacian kernel (defaults to false)
///   When true, uses an 8-connected kernel [-1,-1,-1; -1,8,-1; -1,-1,-1]
///   When false, uses a 4-connected kernel [0,-1,0; -1,4,-1; 0,-1,0]
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array with Laplacian values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{laplace, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Apply 4-connected Laplacian filter
/// let edges = laplace(&image, None, None).unwrap();
///
/// // Apply 8-connected Laplacian filter
/// let edges_diagonal = laplace(&image, None, Some(true)).unwrap();
/// ```
#[allow(dead_code)]
pub fn laplace<T, D>(
    input: &Array<T, D>,
    mode: Option<BorderMode>,
    diagonal: Option<bool>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let use_diagonal = diagonal.unwrap_or(false);

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for laplace filter".into(),
        ));
    }

    // For 2D arrays, apply Laplacian filter directly
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Apply the 2D Laplacian filter
        let result = if use_diagonal {
            laplace_2d_8connected(&input_2d, &border_mode)?
        } else {
            laplace_2d_4connected(&input_2d, &border_mode)?
        };

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, we'll need a more general approach
        // For now, return a placeholder until we implement the full n-dimensional version
        Err(NdimageError::NotImplementedError(
            "Laplace filter not yet implemented for arrays with more than 2 dimensions".into(),
        ))
    }
}

/// Apply a Prewitt filter to calculate gradients in an n-dimensional array
///
/// The Prewitt operator computes an approximation of the gradient of the image intensity function,
/// similar to the Sobel operator but using a slightly different kernel.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `axis` - Axis along which to calculate the gradient
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array with gradient values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{prewitt, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Calculate gradient along x-axis (axis=1)
/// let gradient_x = prewitt(&image, 1, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn prewitt<T, D>(
    input: &Array<T, D>,
    axis: usize,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for prewitt filter".into(),
        ));
    }

    if axis >= input.ndim() {
        return Err(NdimageError::InvalidInput(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // For 2D arrays, we can implement the Prewitt operator directly
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Apply the 2D Prewitt filter
        let result = match axis {
            0 => prewitt_2d_x(&input_2d, &border_mode)?, // Vertical edges (y-derivative)
            1 => prewitt_2d_y(&input_2d, &border_mode)?, // Horizontal edges (x-derivative)
            _ => {
                return Err(NdimageError::InvalidInput(format!(
                    "Invalid axis {} for 2D array, must be 0 or 1",
                    axis
                )));
            }
        };

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, we'll need a more general approach
        // For now, return a placeholder until we implement the full n-dimensional version
        Err(NdimageError::NotImplementedError(
            "Prewitt filter not yet implemented for arrays with more than 2 dimensions".into(),
        ))
    }
}

/// Apply a Roberts Cross filter to detect edges in an n-dimensional array
///
/// The Roberts Cross operator computes a simple 2D approximation of the gradient.
/// It is one of the earliest edge detection operators and highlights regions of high
/// spatial frequency which often correspond to edges.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `axis` - Optional axis parameter (0 for vertical gradient, 1 for horizontal gradient).
///   If None, returns the combined gradient magnitude.
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array with Roberts Cross values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{roberts, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Apply Roberts Cross filter (combined magnitude)
/// let edges = roberts(&image, None, None).unwrap();
///
/// // Get horizontal component
/// let edges_x = roberts(&image, None, Some(1)).unwrap();
/// ```
#[allow(dead_code)]
pub fn roberts<T, D>(
    input: &Array<T, D>,
    mode: Option<BorderMode>,
    axis: Option<usize>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for Roberts filter".into(),
        ));
    }

    // For 2D arrays, calculate Roberts Cross
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Apply the 2D Roberts Cross filter
        let result = match axis {
            Some(0) => roberts_2d_x(&input_2d, &border_mode)?, // Vertical gradient
            Some(1) => roberts_2d_y(&input_2d, &border_mode)?, // Horizontal gradient
            Some(_) => {
                return Err(NdimageError::InvalidInput(
                    "Invalid axis for 2D array, must be 0 or 1".to_string(),
                ));
            }
            None => {
                // Calculate gradient magnitude
                let gradient_x = roberts_2d_x(&input_2d, &border_mode)?;
                let gradient_y = roberts_2d_y(&input_2d, &border_mode)?;

                // Calculate magnitude: sqrt(dx^2 + dy^2)
                let mut magnitude = gradient_x.to_owned();
                for i in 0..magnitude.nrows() {
                    for j in 0..magnitude.ncols() {
                        let gx = gradient_x[[i, j]];
                        let gy = gradient_y[[i, j]];
                        magnitude[[i, j]] = (gx * gx + gy * gy).sqrt();
                    }
                }
                magnitude
            }
        };

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, we'll need a more general approach
        Err(NdimageError::NotImplementedError(
            "Roberts filter not yet implemented for arrays with more than 2 dimensions".into(),
        ))
    }
}

/// Calculate the gradient magnitude of an n-dimensional array
///
/// Calculates the gradient magnitude using the Sobel operator by default
/// or using the specified operator.
///
/// # Arguments
///
/// * `input` - Input array
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `method` - Edge detection method to use for gradient calculation ("sobel", "prewitt", "roberts", or "scharr").
///   Default is "sobel".
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Array with gradient magnitude values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{gradient_magnitude, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Calculate gradient magnitude using default Sobel operator
/// let magnitude = gradient_magnitude(&image, None, None).unwrap();
///
/// // Calculate using Prewitt operator
/// let magnitude_prewitt = gradient_magnitude(&image, None, Some("prewitt")).unwrap();
///
/// // Calculate using Scharr operator (more accurate for diagonal edges)
/// let magnitude_scharr = gradient_magnitude(&image, None, Some("scharr")).unwrap();
/// ```
#[allow(dead_code)]
pub fn gradient_magnitude<T, D>(
    input: &Array<T, D>,
    mode: Option<BorderMode>,
    method: Option<&str>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let method_str = method.unwrap_or("sobel");

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for gradient magnitude".into(),
        ));
    }

    // For 2D arrays, we can calculate the gradient magnitude directly
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Calculate gradients based on the specified method
        let (gradient_x, gradient_y) = match method_str.to_lowercase().as_str() {
            "sobel" => {
                let gx = sobel_2d_y(&input_2d, &border_mode)?;
                let gy = sobel_2d_x(&input_2d, &border_mode)?;
                (gx, gy)
            }
            "prewitt" => {
                let gx = prewitt_2d_y(&input_2d, &border_mode)?;
                let gy = prewitt_2d_x(&input_2d, &border_mode)?;
                (gx, gy)
            }
            "roberts" => {
                let gx = roberts_2d_x(&input_2d, &border_mode)?;
                let gy = roberts_2d_y(&input_2d, &border_mode)?;
                (gx, gy)
            }
            "scharr" => {
                let gx = scharr_2d_y(&input_2d, &border_mode)?;
                let gy = scharr_2d_x(&input_2d, &border_mode)?;
                (gx, gy)
            }
            _ => {
                return Err(NdimageError::InvalidInput(format!(
                    "Invalid method: {}, must be one of: sobel, prewitt, roberts, scharr",
                    method_str
                )));
            }
        };

        // Calculate magnitude: sqrt(dx^2 + dy^2)
        let mut result = input_2d.to_owned();
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                let gx = gradient_x[[i, j]];
                let gy = gradient_y[[i, j]];
                let magnitude = (gx * gx + gy * gy).sqrt();
                result[[i, j]] = magnitude;
            }
        }

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, we'll need a more general approach
        Err(NdimageError::NotImplementedError(
            "Gradient magnitude not yet implemented for arrays with more than 2 dimensions".into(),
        ))
    }
}

// Helper function to apply Prewitt filter along y-axis (vertical gradient)
#[allow(dead_code)]
fn prewitt_2d_x<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Prewitt 3x3 kernel for y-derivative (vertical gradient)
    // [[ 1,  1,  1],
    //  [ 0,  0,  0],
    //  [-1, -1, -1]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = T::one();
    kernel[[0, 1]] = T::one();
    kernel[[0, 2]] = T::one();
    kernel[[2, 0]] = -T::one();
    kernel[[2, 1]] = -T::one();
    kernel[[2, 2]] = -T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Prewitt filter along x-axis (horizontal gradient)
#[allow(dead_code)]
fn prewitt_2d_y<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Prewitt 3x3 kernel for x-derivative (horizontal gradient)
    // [[-1,  0,  1],
    //  [-1,  0,  1],
    //  [-1,  0,  1]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = -T::one();
    kernel[[1, 0]] = -T::one();
    kernel[[2, 0]] = -T::one();
    kernel[[0, 2]] = T::one();
    kernel[[1, 2]] = T::one();
    kernel[[2, 2]] = T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Sobel filter along y-axis (vertical gradient)
#[allow(dead_code)]
fn sobel_2d_x<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Sobel 3x3 kernel for y-derivative (vertical gradient)
    // [[ 1,  2,  1],
    //  [ 0,  0,  0],
    //  [-1, -2, -1]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = T::one();
    kernel[[0, 1]] = safe_i32_to_float(2)?;
    kernel[[0, 2]] = T::one();
    kernel[[2, 0]] = -T::one();
    kernel[[2, 1]] = -safe_i32_to_float(2)?;
    kernel[[2, 2]] = -T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Sobel filter along x-axis (horizontal gradient)
#[allow(dead_code)]
fn sobel_2d_y<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Sobel 3x3 kernel for x-derivative (horizontal gradient)
    // [[-1,  0,  1],
    //  [-2,  0,  2],
    //  [-1,  0,  1]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = -T::one();
    kernel[[1, 0]] = -safe_i32_to_float(2)?;
    kernel[[2, 0]] = -T::one();
    kernel[[0, 2]] = T::one();
    kernel[[1, 2]] = safe_i32_to_float(2)?;
    kernel[[2, 2]] = T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Roberts Cross filter for the x-component
#[allow(dead_code)]
fn roberts_2d_x<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Roberts Cross kernel for x-component
    // [[ 1,  0],
    //  [ 0, -1]]
    let mut kernel = Array2::<T>::zeros((2, 2));
    kernel[[0, 0]] = T::one();
    kernel[[1, 1]] = -T::one();

    // Apply convolution - this applies the kernel at each position
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Roberts Cross filter for the y-component
#[allow(dead_code)]
fn roberts_2d_y<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Roberts Cross kernel for y-component
    // [[ 0,  1],
    //  [-1,  0]]
    let mut kernel = Array2::<T>::zeros((2, 2));
    kernel[[0, 1]] = T::one();
    kernel[[1, 0]] = -T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply 4-connected Laplace filter (for 2D arrays)
#[allow(dead_code)]
fn laplace_2d_4connected<T>(
    input: &Array<T, Ix2>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Laplacian 3x3 kernel (4-connected)
    // [[ 0,  1,  0],
    //  [ 1, -4,  1],
    //  [ 0,  1,  0]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 1]] = T::one();
    kernel[[1, 0]] = T::one();
    kernel[[1, 1]] = -safe_i32_to_float(4)?;
    kernel[[1, 2]] = T::one();
    kernel[[2, 1]] = T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply 8-connected Laplace filter (for 2D arrays)
#[allow(dead_code)]
fn laplace_2d_8connected<T>(
    input: &Array<T, Ix2>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Laplacian 3x3 kernel (8-connected)
    // [[-1, -1, -1],
    //  [-1,  8, -1],
    //  [-1, -1, -1]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = -T::one();
    kernel[[0, 1]] = -T::one();
    kernel[[0, 2]] = -T::one();
    kernel[[1, 0]] = -T::one();
    kernel[[1, 1]] = safe_i32_to_float(8)?;
    kernel[[1, 2]] = -T::one();
    kernel[[2, 0]] = -T::one();
    kernel[[2, 1]] = -T::one();
    kernel[[2, 2]] = -T::one();

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

/// Apply a Scharr filter to calculate gradients in an n-dimensional array
///
/// The Scharr operator is a more accurate approximation of the gradient than the Sobel operator.
/// It uses different coefficients that provide better rotational symmetry, which makes it more
/// accurate for calculating gradients in all directions.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `axis` - Axis along which to calculate the gradient
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array with gradient values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{scharr, BorderMode};
///
/// let image = array![
///     [0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0],
/// ];
///
/// // Calculate gradient along x-axis (axis=1)
/// let gradient_x = scharr(&image, 1, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn scharr<T, D>(
    input: &Array<T, D>,
    axis: usize,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() <= 1 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for scharr filter".into(),
        ));
    }

    if axis >= input.ndim() {
        return Err(NdimageError::InvalidInput(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // For 2D arrays, we can implement the Scharr operator directly
    if input.ndim() == 2 {
        // Convert to 2D array for processing
        let input_2d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        // Apply the 2D Scharr filter
        let result = match axis {
            0 => scharr_2d_x(&input_2d, &border_mode)?, // Vertical edges (y-derivative)
            1 => scharr_2d_y(&input_2d, &border_mode)?, // Horizontal edges (x-derivative)
            _ => {
                return Err(NdimageError::InvalidInput(format!(
                    "Invalid axis {} for 2D array, must be 0 or 1",
                    axis
                )));
            }
        };

        // Convert result back to original dimensionality
        result.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError(
                "Failed to convert result back to original dimensions".into(),
            )
        })
    } else {
        // For higher dimensions, we'll need a more general approach
        // For now, return a placeholder until we implement the full n-dimensional version
        Err(NdimageError::NotImplementedError(
            "Scharr filter not yet implemented for arrays with more than 2 dimensions".into(),
        ))
    }
}

// Helper function to apply Scharr filter along y-axis (vertical gradient)
#[allow(dead_code)]
fn scharr_2d_x<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Scharr 3x3 kernel for y-derivative (vertical gradient)
    // [[ 3,  10,  3],
    //  [ 0,   0,  0],
    //  [-3, -10, -3]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = safe_i32_to_float(3)?;
    kernel[[0, 1]] = safe_i32_to_float(10)?;
    kernel[[0, 2]] = safe_i32_to_float(3)?;
    kernel[[2, 0]] = safe_i32_to_float(-3)?;
    kernel[[2, 1]] = safe_i32_to_float(-10)?;
    kernel[[2, 2]] = safe_i32_to_float(-3)?;

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

// Helper function to apply Scharr filter along x-axis (horizontal gradient)
#[allow(dead_code)]
fn scharr_2d_y<T>(input: &Array<T, Ix2>, mode: &BorderMode) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Create the Scharr 3x3 kernel for x-derivative (horizontal gradient)
    // [[ -3,  0,  3],
    //  [-10,  0, 10],
    //  [ -3,  0,  3]]
    let mut kernel = Array2::<T>::zeros((3, 3));
    kernel[[0, 0]] = safe_i32_to_float(-3)?;
    kernel[[1, 0]] = safe_i32_to_float(-10)?;
    kernel[[2, 0]] = safe_i32_to_float(-3)?;
    kernel[[0, 2]] = safe_i32_to_float(3)?;
    kernel[[1, 2]] = safe_i32_to_float(10)?;
    kernel[[2, 2]] = safe_i32_to_float(3)?;

    // Apply convolution
    convolve(input, &kernel, Some(*mode))
}

/// N-dimensional Sobel filter implementation
#[allow(dead_code)]
fn sobel_nd<T, D>(input: &Array<T, D>, axis: usize, mode: &BorderMode) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
{
    use super::convolve::correlate1d;

    // First apply the derivative filter [-1, 0, 1] along the specified axis
    let deriv_kernel = array![-T::one(), T::zero(), T::one()];
    let mut result = correlate1d(input, &deriv_kernel, axis, Some(*mode), None)?;

    // Then apply the smoothing filter [1, 2, 1] along all other axes
    let smooth_kernel = array![T::one(), safe_i32_to_float(2)?, T::one()];

    for ax in 0..input.ndim() {
        if ax != axis {
            result = correlate1d(&result, &smooth_kernel, ax, Some(*mode), None)?;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_sobel() {
        // Create a simple test image with isolated point
        // This is better for testing since we can clearly predict the filter response
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply Sobel filter in both directions
        let sobel_x = sobel(&image, 0, None).expect("sobel filter should succeed"); // Vertical gradient (y direction)
        let sobel_y = sobel(&image, 1, None).expect("sobel filter should succeed"); // Horizontal gradient (x direction)

        // Check shape
        assert_eq!(sobel_x.shape(), image.shape());
        assert_eq!(sobel_y.shape(), image.shape());

        // For a point in the center, we expect:
        // - Positive response above the point and negative below in the y direction
        // - Positive response to the right of the point and negative to the left in the x direction
        assert!(sobel_x[[1, 2]] > 0.0); // Top of point (positive y gradient)
        assert!(sobel_x[[3, 2]] < 0.0); // Bottom of point (negative y gradient)

        assert!(sobel_y[[2, 3]] > 0.0); // Right of point (positive x gradient)
        assert!(sobel_y[[2, 1]] < 0.0); // Left of point (negative x gradient)
    }

    #[test]
    fn test_scharr() {
        // Create a simple test image with a point
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply Scharr filter in both directions
        let scharr_x = scharr(&image, 0, None).expect("scharr filter should succeed"); // Vertical gradient (y direction)
        let scharr_y = scharr(&image, 1, None).expect("scharr filter should succeed"); // Horizontal gradient (x direction)

        // Check shape
        assert_eq!(scharr_x.shape(), image.shape());
        assert_eq!(scharr_y.shape(), image.shape());

        // For a point in the center, we expect:
        // - Positive response above the point and negative below in the y direction
        // - Positive response to the right of the point and negative to the left in the x direction
        assert!(scharr_x[[1, 2]] > 0.0); // Top of point (positive y gradient)
        assert!(scharr_x[[3, 2]] < 0.0); // Bottom of point (negative y gradient)

        assert!(scharr_y[[2, 3]] > 0.0); // Right of point (positive x gradient)
        assert!(scharr_y[[2, 1]] < 0.0); // Left of point (negative x gradient)

        // Scharr should give stronger diagonal response than Sobel
        let sobel_x = sobel(&image, 0, None).expect("sobel filter for comparison should succeed");
        let sobel_y = sobel(&image, 1, None).expect("sobel filter for comparison should succeed");

        // Check diagonal values
        assert!(scharr_x[[1, 1]].abs() > sobel_x[[1, 1]].abs());
        assert!(scharr_y[[1, 1]].abs() > sobel_y[[1, 1]].abs());
    }

    #[test]
    fn test_laplace() {
        // Create a simple test image with a point
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply default 4-connected filter
        let result = laplace(&image, None, None).expect("laplace filter should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());

        // Point should create a distinctive Laplacian response
        // The center should be negative and surroundings positive
        assert!(result[[2, 2]] < 0.0); // Center is negative
        assert!(result[[1, 2]] > 0.0); // Surrounding pixels (up, down, left, right) are positive
        assert!(result[[2, 1]] > 0.0);
        assert!(result[[3, 2]] > 0.0);
        assert!(result[[2, 3]] > 0.0);

        // Diagonal neighbors should be zero in 4-connected Laplacian
        assert_eq!(result[[1, 1]], 0.0);
        assert_eq!(result[[1, 3]], 0.0);
        assert_eq!(result[[3, 1]], 0.0);
        assert_eq!(result[[3, 3]], 0.0);
    }

    #[test]
    fn test_laplace_8connected() {
        // Create a simple test image with a point
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply 8-connected filter
        let result =
            laplace(&image, None, Some(true)).expect("laplace 8-connected filter should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());

        // Point should create a distinctive Laplacian response
        // The center should be positive and all surroundings negative
        assert!(result[[2, 2]] > 0.0); // Center is positive (8)

        // All 8 neighbors should be negative
        assert!(result[[1, 1]] < 0.0); // Diagonal neighbors
        assert!(result[[1, 2]] < 0.0);
        assert!(result[[1, 3]] < 0.0);
        assert!(result[[2, 1]] < 0.0);
        assert!(result[[2, 3]] < 0.0);
        assert!(result[[3, 1]] < 0.0);
        assert!(result[[3, 2]] < 0.0);
        assert!(result[[3, 3]] < 0.0);

        // Check relative magnitudes - center should be 8x diagonal values
        let center_val = result[[2, 2]];
        let diagonal_val = -result[[1, 1]];
        let ratio = (center_val / diagonal_val).abs();
        assert!((ratio - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_prewitt() {
        // Create a simple test image with isolated point
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply Prewitt filter in both directions
        let prewitt_x = prewitt(&image, 0, None).expect("prewitt filter should succeed"); // Vertical gradient (y direction)
        let prewitt_y = prewitt(&image, 1, None).expect("prewitt filter should succeed"); // Horizontal gradient (x direction)

        // Check shape
        assert_eq!(prewitt_x.shape(), image.shape());
        assert_eq!(prewitt_y.shape(), image.shape());

        // For a point in the center, we expect similar responses to Sobel:
        // - Positive response above the point and negative below in the y direction
        // - Positive response to the right of the point and negative to the left in the x direction
        assert!(prewitt_x[[1, 2]] > 0.0); // Top of point (positive y gradient)
        assert!(prewitt_x[[3, 2]] < 0.0); // Bottom of point (negative y gradient)

        assert!(prewitt_y[[2, 3]] > 0.0); // Right of point (positive x gradient)
        assert!(prewitt_y[[2, 1]] < 0.0); // Left of point (negative x gradient)
    }

    #[test]
    fn test_roberts() {
        // For Roberts Cross, which uses a 2x2 kernel, we need a sharp
        // contrast in a diagonal pattern to get a strong response
        let mut image = Array2::<f64>::zeros((5, 5));
        // Create a 2x2 pattern in the center to ensure the kernel can be applied fully
        image[[1, 1]] = 0.0;
        image[[1, 2]] = 1.0;
        image[[2, 1]] = 1.0;
        image[[2, 2]] = 0.0;

        // Apply Roberts filter in both directions
        let roberts_x =
            roberts(&image, None, Some(0)).expect("roberts filter x-component should succeed"); // x-component
        let roberts_y =
            roberts(&image, None, Some(1)).expect("roberts filter y-component should succeed"); // y-component

        // Check shape
        assert_eq!(roberts_x.shape(), image.shape());
        assert_eq!(roberts_y.shape(), image.shape());

        // The combined magnitude should show the edges
        let roberts_mag =
            roberts(&image, None, None).expect("roberts magnitude filter should succeed");

        // The center 2x2 area is where we expect the response
        let center_region = roberts_mag.slice(scirs2_core::ndarray::s![1..3, 1..3]);

        // At least one value in the center region should be significantly positive
        let mut has_significant_response = false;
        for i in 0..2 {
            for j in 0..2 {
                if center_region[[i, j]] > 0.5 {
                    has_significant_response = true;
                    break;
                }
            }
            if has_significant_response {
                break;
            }
        }

        assert!(
            has_significant_response,
            "Roberts filter should have a significant response to the diagonal pattern"
        );

        // Test the relationship between components and magnitude
        for i in 1..3 {
            for j in 1..3 {
                // The magnitude at each point should equal the square root of the sum of squares
                let calculated = (roberts_x[[i, j]].powi(2) + roberts_y[[i, j]].powi(2)).sqrt();
                assert!(
                    (roberts_mag[[i, j]] - calculated).abs() < 1e-10,
                    "Magnitude should be sqrt(x² + y²) at position [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_gradient_magnitude() {
        // Create a simple test image with a vertical edge
        let image = array![
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ];

        // Calculate gradient magnitude with different methods
        let result_sobel = gradient_magnitude(&image, None, None)
            .expect("gradient_magnitude with sobel should succeed");
        let result_prewitt = gradient_magnitude(&image, None, Some("prewitt"))
            .expect("gradient_magnitude with prewitt should succeed");
        let result_roberts = gradient_magnitude(&image, None, Some("roberts"))
            .expect("gradient_magnitude with roberts should succeed");
        let result_scharr = gradient_magnitude(&image, None, Some("scharr"))
            .expect("gradient_magnitude with scharr should succeed");

        // Check shapes
        assert_eq!(result_sobel.shape(), image.shape());
        assert_eq!(result_prewitt.shape(), image.shape());
        assert_eq!(result_roberts.shape(), image.shape());
        assert_eq!(result_scharr.shape(), image.shape());

        // Edge should create a high gradient magnitude at the edge location
        for i in 0..5 {
            // For each method, the edge at x=2 should have highest gradient value in that row
            let mut max_sobel_idx = 0;
            let mut max_prewitt_idx = 0;
            let mut max_roberts_idx = 0;
            let mut max_scharr_idx = 0;

            for j in 0..5 {
                if result_sobel[[i, j]] > result_sobel[[i, max_sobel_idx]] {
                    max_sobel_idx = j;
                }
                if result_prewitt[[i, j]] > result_prewitt[[i, max_prewitt_idx]] {
                    max_prewitt_idx = j;
                }
                if result_roberts[[i, j]] > result_roberts[[i, max_roberts_idx]] {
                    max_roberts_idx = j;
                }
                if result_scharr[[i, j]] > result_scharr[[i, max_scharr_idx]] {
                    max_scharr_idx = j;
                }
            }

            // Edge is at position 1-2 or 2-3 for Sobel, Prewitt, and Scharr
            assert!(
                max_sobel_idx == 1 || max_sobel_idx == 2,
                "Row {}: max Sobel gradient not at expected position, found at {}",
                i,
                max_sobel_idx
            );
            assert!(
                max_prewitt_idx == 1 || max_prewitt_idx == 2,
                "Row {}: max Prewitt gradient not at expected position, found at {}",
                i,
                max_prewitt_idx
            );
            assert!(
                max_scharr_idx == 1 || max_scharr_idx == 2,
                "Row {}: max Scharr gradient not at expected position, found at {}",
                i,
                max_scharr_idx
            );
            // Roberts might detect the edge slightly differently due to its 2x2 kernel
            assert!(
                max_roberts_idx >= 1 && max_roberts_idx <= 3,
                "Row {}: max Roberts gradient not near edge, found at {}",
                i,
                max_roberts_idx
            );
        }

        // Test invalid method error
        let invalid_result = gradient_magnitude(&image, None, Some("invalid_method"));
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_sobel_3d() {
        use scirs2_core::ndarray::Array3;

        // Create a simple 3x3x3 test volume with a plane at z=1
        let mut volume = Array3::<f64>::zeros((3, 3, 3));
        for i in 0..3 {
            for j in 0..3 {
                volume[[i, j, 1]] = 1.0;
            }
        }

        // Test Sobel along axis 0 (x-axis)
        let result_x = sobel(&volume, 0, None).expect("sobel 3D x-axis should succeed");
        assert_eq!(result_x.shape(), volume.shape());

        // Test Sobel along axis 1 (y-axis)
        let result_y = sobel(&volume, 1, None).expect("sobel 3D y-axis should succeed");
        assert_eq!(result_y.shape(), volume.shape());

        // Test Sobel along axis 2 (z-axis)
        let result_z = sobel(&volume, 2, None).expect("sobel 3D z-axis should succeed");
        assert_eq!(result_z.shape(), volume.shape());

        // The gradient should be strongest along the z-axis
        let max_x = result_x.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let max_y = result_y.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let max_z = result_z.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        assert!(max_z > max_x, "Z gradient should be strongest");
        assert!(max_z > max_y, "Z gradient should be strongest");
    }
}
