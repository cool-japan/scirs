//! Reduction universal functions
//!
//! This module provides implementation of reduction operations
//! (sum, mean, etc.) as universal functions for efficient
//! array reductions along specified axes.

use ndarray::{Array, Array1, ArrayView, Axis, Dimension, Ix1, IxDyn, ShapeBuilder};
use crate::ufuncs::core::{UFunc, UFuncKind, apply_reduction, register_ufunc};
use std::sync::Once;

static INIT: Once = Once::new();

// Initialize the ufunc registry with reduction operations
fn init_reduction_ufuncs() {
    INIT.call_once(|| {
        // Register all the reduction ufuncs
        let _ = register_ufunc(Box::new(SumUFunc));
        let _ = register_ufunc(Box::new(ProductUFunc));
        let _ = register_ufunc(Box::new(MeanUFunc));
        let _ = register_ufunc(Box::new(StdUFunc));
        let _ = register_ufunc(Box::new(VarUFunc));
        let _ = register_ufunc(Box::new(MinUFunc));
        let _ = register_ufunc(Box::new(MaxUFunc));
    });
}

// Define the reduction ufuncs

/// Sum reduction universal function
pub struct SumUFunc;

impl UFunc for SumUFunc {
    fn name(&self) -> &str {
        "sum"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Sum requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply sum reduction along all dimensions
            let mut sum = 0.0;
            for &val in input_view.iter() {
                sum += val;
            }

            output1d[0] = sum;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Product reduction universal function
pub struct ProductUFunc;

impl UFunc for ProductUFunc {
    fn name(&self) -> &str {
        "product"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Product requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply product reduction along all dimensions
            let mut product = 1.0;
            for &val in input_view.iter() {
                product *= val;
            }

            output1d[0] = product;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Mean reduction universal function
pub struct MeanUFunc;

impl UFunc for MeanUFunc {
    fn name(&self) -> &str {
        "mean"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Mean requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply mean reduction along all dimensions
            let mut sum = 0.0;
            let count = input_view.len();

            if count == 0 {
                return Err("Cannot compute mean of empty array");
            }

            for &val in input_view.iter() {
                sum += val;
            }

            output1d[0] = sum / count as f64;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Standard deviation reduction universal function
pub struct StdUFunc;

impl UFunc for StdUFunc {
    fn name(&self) -> &str {
        "std"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Std requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply standard deviation reduction along all dimensions
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = input_view.len();

            if count <= 1 {
                return Err("Cannot compute standard deviation with less than 2 elements");
            }

            for &val in input_view.iter() {
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / count as f64;
            let variance = sum_sq / count as f64 - mean * mean;

            output1d[0] = variance.sqrt();
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Variance reduction universal function
pub struct VarUFunc;

impl UFunc for VarUFunc {
    fn name(&self) -> &str {
        "var"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Var requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply variance reduction along all dimensions
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = input_view.len();

            if count <= 1 {
                return Err("Cannot compute variance with less than 2 elements");
            }

            for &val in input_view.iter() {
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / count as f64;
            let variance = sum_sq / count as f64 - mean * mean;

            output1d[0] = variance;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Minimum reduction universal function
pub struct MinUFunc;

impl UFunc for MinUFunc {
    fn name(&self) -> &str {
        "min"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Min requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply minimum reduction along all dimensions
            if input_view.len() == 0 {
                return Err("Cannot compute minimum of empty array");
            }

            let mut min_val = f64::INFINITY;
            for &val in input_view.iter() {
                if val < min_val {
                    min_val = val;
                }
            }

            output1d[0] = min_val;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

/// Maximum reduction universal function
pub struct MaxUFunc;

impl UFunc for MaxUFunc {
    fn name(&self) -> &str {
        "max"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Reduction
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Max requires exactly one input array");
        }

        // Not a proper implementation of the reduction operation
        // Just a placeholder for the full implementation
        if let Some(output1d) = output.as_slice_mut() {
            let input_view = inputs[0];

            // Apply maximum reduction along all dimensions
            if input_view.len() == 0 {
                return Err("Cannot compute maximum of empty array");
            }

            let mut max_val = f64::NEG_INFINITY;
            for &val in input_view.iter() {
                if val > max_val {
                    max_val = val;
                }
            }

            output1d[0] = max_val;
            Ok(())
        } else {
            Err("Output array is not contiguous")
        }
    }
}

// Helper function to prepare the output array for reduction
fn prepare_reduction_output<D>(input: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> (Array<f64, Ix1>, Vec<usize>)
where
    D: Dimension,
{
    match axis {
        Some(ax) => {
            if ax >= input.ndim() {
                panic!("Axis index out of bounds");
            }

            // For reduction along a specific axis, the output shape is the input shape
            // with the specified axis removed
            let mut out_shape = Vec::with_capacity(input.ndim() - 1);
            let mut output_size = 1;

            for (i, &dim) in input.shape().iter().enumerate() {
                if i != ax {
                    out_shape.push(dim);
                    output_size *= dim;
                }
            }

            (Array::<f64, _>::zeros(output_size), out_shape)
        },
        None => {
            // For reduction over the entire array, the output shape is [1]
            (Array::<f64, _>::zeros(1), vec![1])
        }
    }
}

// Convenience functions for applying reduction ufuncs

/// Compute the sum of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the sum (None means sum over all elements)
///
/// # Returns
///
/// An array with the sum along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::sum;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Sum over all elements
/// let result = sum(&a, None);
/// assert_eq!(result, array![21.0]);
///
/// // Sum along axis 0 (columns)
/// let result = sum(&a, Some(0));
/// assert_eq!(result, array![5.0, 7.0, 9.0]);
///
/// // Sum along axis 1 (rows)
/// let result = sum(&a, Some(1));
/// assert_eq!(result, array![6.0, 15.0]);
/// ```
pub fn sum<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    // Prepare the output array
    let (mut output, _) = prepare_reduction_output(array, axis);

    // Apply the sum function along the specified axis
    match axis {
        Some(ax) => {
            apply_reduction(array, &mut output, Some(ax), Some(0.0),
                            |acc, &x| acc + x).unwrap();
        },
        None => {
            apply_reduction(array, &mut output, None, Some(0.0),
                            |acc, &x| acc + x).unwrap();
        }
    }

    output
}

/// Compute the product of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the product (None means product over all elements)
///
/// # Returns
///
/// An array with the product along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::product;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Product over all elements
/// let result = product(&a, None);
/// assert_eq!(result, array![720.0]);
///
/// // Product along axis 0 (columns)
/// let result = product(&a, Some(0));
/// assert_eq!(result, array![4.0, 10.0, 18.0]);
///
/// // Product along axis 1 (rows)
/// let result = product(&a, Some(1));
/// assert_eq!(result, array![6.0, 120.0]);
/// ```
pub fn product<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    // Prepare the output array
    let (mut output, _) = prepare_reduction_output(array, axis);

    // Apply the product function along the specified axis
    match axis {
        Some(ax) => {
            apply_reduction(array, &mut output, Some(ax), Some(1.0),
                            |acc, &x| acc * x).unwrap();
        },
        None => {
            apply_reduction(array, &mut output, None, Some(1.0),
                            |acc, &x| acc * x).unwrap();
        }
    }

    output
}

/// Compute the mean of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the mean (None means mean over all elements)
///
/// # Returns
///
/// An array with the mean along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::mean;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Mean over all elements
/// let result = mean(&a, None);
/// assert_eq!(result, array![3.5]);
///
/// // Mean along axis 0 (columns)
/// let result = mean(&a, Some(0));
/// assert_eq!(result, array![2.5, 3.5, 4.5]);
///
/// // Mean along axis 1 (rows)
/// let result = mean(&a, Some(1));
/// assert_eq!(result, array![2.0, 5.0]);
/// ```
pub fn mean<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    let sum_result = sum(array, axis.clone());

    match axis {
        Some(ax) => {
            // Divide by the length of the specified axis
            let axis_len = array.len_of(ndarray::Axis(ax)) as f64;
            sum_result.map(|&x| x / axis_len)
        },
        None => {
            // Divide by the total number of elements
            let total_elements = array.len() as f64;
            Array::from_vec(vec![sum_result[0] / total_elements])
        }
    }
}

/// Compute the standard deviation of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the standard deviation (None means std over all elements)
///
/// # Returns
///
/// An array with the standard deviation along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::std;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Standard deviation over all elements
/// let result = std(&a, None);
/// assert!((result[0] - 1.870829).abs() < 1e-6);
///
/// // Standard deviation along axis 0 (columns)
/// let result = std(&a, Some(0));
/// assert!((result[0] - 1.5).abs() < 1e-10);
/// assert!((result[1] - 1.5).abs() < 1e-10);
/// assert!((result[2] - 1.5).abs() < 1e-10);
/// ```
pub fn std<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    let var_result = var(array, axis);
    var_result.map(|&x| x.sqrt())
}

/// Compute the variance of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the variance (None means variance over all elements)
///
/// # Returns
///
/// An array with the variance along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::var;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Variance over all elements
/// let result = var(&a, None);
/// assert!((result[0] - 3.5).abs() < 1e-10);
///
/// // Variance along axis 0 (columns)
/// let result = var(&a, Some(0));
/// assert_eq!(result, array![2.25, 2.25, 2.25]);
/// ```
pub fn var<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    // Compute the mean
    let mean_result = mean(array, axis.clone());

    // Prepare the output array for variance
    let (mut output, _) = prepare_reduction_output(array, axis.clone());

    // Compute the variance
    match axis {
        Some(ax) => {
            // Calculate the squared deviations from the mean
            let axis_len = array.len_of(ndarray::Axis(ax)) as f64;

            // Use the apply_reduction function from the core module
            // This is a simplified implementation that only handles 2D arrays
            if array.ndim() == 2 {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                if ax == 0 {
                    // Reduce along rows
                    for j in 0..cols {
                        let mean_val = mean_result[j];
                        let mut sum_sq_diff = 0.0;

                        for i in 0..rows {
                            let diff = array[[i, j]] - mean_val;
                            sum_sq_diff += diff * diff;
                        }

                        output[j] = sum_sq_diff / axis_len;
                    }
                } else {
                    // Reduce along columns
                    for i in 0..rows {
                        let mean_val = mean_result[i];
                        let mut sum_sq_diff = 0.0;

                        for j in 0..cols {
                            let diff = array[[i, j]] - mean_val;
                            sum_sq_diff += diff * diff;
                        }

                        output[i] = sum_sq_diff / axis_len;
                    }
                }
            } else {
                // This simplified implementation only handles 2D arrays
                panic!("This simplified implementation only supports 2D arrays");
            }
        },
        None => {
            // Compute variance over the entire array
            let mean_val = mean_result[0];
            let total_elements = array.len() as f64;
            let mut sum_sq_diff = 0.0;

            for &val in array.iter() {
                let diff = val - mean_val;
                sum_sq_diff += diff * diff;
            }

            output[0] = sum_sq_diff / total_elements;
        }
    }

    output
}

/// Compute the minimum of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the minimum (None means min over all elements)
///
/// # Returns
///
/// An array with the minimum along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::min;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Minimum over all elements
/// let result = min(&a, None);
/// assert_eq!(result, array![1.0]);
///
/// // Minimum along axis 0 (columns)
/// let result = min(&a, Some(0));
/// assert_eq!(result, array![1.0, 2.0, 3.0]);
///
/// // Minimum along axis 1 (rows)
/// let result = min(&a, Some(1));
/// assert_eq!(result, array![1.0, 4.0]);
/// ```
pub fn min<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    // Prepare the output array
    let (mut output, _) = prepare_reduction_output(array, axis);

    // Compute the minimum
    match axis {
        Some(ax) => {
            // This is a simplified implementation that only handles 2D arrays
            if array.ndim() == 2 {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                if ax == 0 {
                    // Reduce along rows
                    for j in 0..cols {
                        let mut min_val = f64::INFINITY;

                        for i in 0..rows {
                            if array[[i, j]] < min_val {
                                min_val = array[[i, j]];
                            }
                        }

                        output[j] = min_val;
                    }
                } else {
                    // Reduce along columns
                    for i in 0..rows {
                        let mut min_val = f64::INFINITY;

                        for j in 0..cols {
                            if array[[i, j]] < min_val {
                                min_val = array[[i, j]];
                            }
                        }

                        output[i] = min_val;
                    }
                }
            } else {
                // This simplified implementation only handles 2D arrays
                panic!("This simplified implementation only supports 2D arrays");
            }
        },
        None => {
            // Find minimum over the entire array
            let mut min_val = f64::INFINITY;

            for &val in array.iter() {
                if val < min_val {
                    min_val = val;
                }
            }

            output[0] = min_val;
        }
    }

    output
}

/// Compute the maximum of array elements
///
/// # Arguments
///
/// * `array` - Input array
/// * `axis` - Optional axis along which to compute the maximum (None means max over all elements)
///
/// # Returns
///
/// An array with the maximum along the specified axis or over the entire array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::max;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Maximum over all elements
/// let result = max(&a, None);
/// assert_eq!(result, array![6.0]);
///
/// // Maximum along axis 0 (columns)
/// let result = max(&a, Some(0));
/// assert_eq!(result, array![4.0, 5.0, 6.0]);
///
/// // Maximum along axis 1 (rows)
/// let result = max(&a, Some(1));
/// assert_eq!(result, array![3.0, 6.0]);
/// ```
pub fn max<D>(array: &ndarray::ArrayBase<ndarray::Data, D>, axis: Option<usize>) -> Array<f64, Ix1>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_reduction_ufuncs();

    // Prepare the output array
    let (mut output, _) = prepare_reduction_output(array, axis);

    // Compute the maximum
    match axis {
        Some(ax) => {
            // This is a simplified implementation that only handles 2D arrays
            if array.ndim() == 2 {
                let (rows, cols) = (array.shape()[0], array.shape()[1]);

                if ax == 0 {
                    // Reduce along rows
                    for j in 0..cols {
                        let mut max_val = f64::NEG_INFINITY;

                        for i in 0..rows {
                            if array[[i, j]] > max_val {
                                max_val = array[[i, j]];
                            }
                        }

                        output[j] = max_val;
                    }
                } else {
                    // Reduce along columns
                    for i in 0..rows {
                        let mut max_val = f64::NEG_INFINITY;

                        for j in 0..cols {
                            if array[[i, j]] > max_val {
                                max_val = array[[i, j]];
                            }
                        }

                        output[i] = max_val;
                    }
                }
            } else {
                // This simplified implementation only handles 2D arrays
                panic!("This simplified implementation only supports 2D arrays");
            }
        },
        None => {
            // Find maximum over the entire array
            let mut max_val = f64::NEG_INFINITY;

            for &val in array.iter() {
                if val > max_val {
                    max_val = val;
                }
            }

            output[0] = max_val;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sum() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Sum over all elements
        let result = sum(&a, None);
        assert_eq!(result, array![21.0]);

        // Sum along axis 0 (columns)
        let result = sum(&a, Some(0));
        assert_eq!(result, array![5.0, 7.0, 9.0]);

        // Sum along axis 1 (rows)
        let result = sum(&a, Some(1));
        assert_eq!(result, array![6.0, 15.0]);
    }

    #[test]
    fn test_product() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Product over all elements
        let result = product(&a, None);
        assert_eq!(result, array![720.0]);

        // Product along axis 0 (columns)
        let result = product(&a, Some(0));
        assert_eq!(result, array![4.0, 10.0, 18.0]);

        // Product along axis 1 (rows)
        let result = product(&a, Some(1));
        assert_eq!(result, array![6.0, 120.0]);
    }

    #[test]
    fn test_mean() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Mean over all elements
        let result = mean(&a, None);
        assert_eq!(result, array![3.5]);

        // Mean along axis 0 (columns)
        let result = mean(&a, Some(0));
        assert_eq!(result, array![2.5, 3.5, 4.5]);

        // Mean along axis 1 (rows)
        let result = mean(&a, Some(1));
        assert_eq!(result, array![2.0, 5.0]);
    }

    #[test]
    fn test_std() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Standard deviation over all elements
        let result = std(&a, None);
        assert!((result[0] - 1.870829).abs() < 1e-6);

        // Standard deviation along axis 0 (columns)
        let result = std(&a, Some(0));
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);
        assert!((result[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_var() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Variance over all elements
        let result = var(&a, None);
        assert!((result[0] - 3.5).abs() < 1e-10);

        // Variance along axis 0 (columns)
        let result = var(&a, Some(0));
        assert_eq!(result, array![2.25, 2.25, 2.25]);
    }

    #[test]
    fn test_min() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Minimum over all elements
        let result = min(&a, None);
        assert_eq!(result, array![1.0]);

        // Minimum along axis 0 (columns)
        let result = min(&a, Some(0));
        assert_eq!(result, array![1.0, 2.0, 3.0]);

        // Minimum along axis 1 (rows)
        let result = min(&a, Some(1));
        assert_eq!(result, array![1.0, 4.0]);
    }

    #[test]
    fn test_max() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Maximum over all elements
        let result = max(&a, None);
        assert_eq!(result, array![6.0]);

        // Maximum along axis 0 (columns)
        let result = max(&a, Some(0));
        assert_eq!(result, array![4.0, 5.0, 6.0]);

        // Maximum along axis 1 (rows)
        let result = max(&a, Some(1));
        assert_eq!(result, array![3.0, 6.0]);
    }
}