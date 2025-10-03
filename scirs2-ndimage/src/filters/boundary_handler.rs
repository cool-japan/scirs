//! Optimized boundary handling for filters
//!
//! This module provides efficient boundary handling that avoids unnecessary array copies
//! by using virtual indexing and on-the-fly boundary value computation.

use scirs2_core::ndarray::{Array, ArrayView, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::BorderMode;
use crate::error::{NdimageError, NdimageResult};

/// A trait for efficient boundary handling without creating padded arrays
pub trait BoundaryHandler<T, D>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    /// Get a value at the given position, handling boundaries according to the mode
    fn get_value(&self, indices: &[isize]) -> T;

    /// Get the shape of the underlying array
    fn shape(&self) -> &[usize];

    /// Get the dimensionality
    fn ndim(&self) -> usize;
}

/// Virtual boundary handler that computes boundary values on-the-fly
pub struct VirtualBoundaryHandler<'a, T, D>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    array: &'a ArrayView<'a, T, D>,
    mode: BorderMode,
    constant_value: T,
}

impl<'a, T, D> VirtualBoundaryHandler<'a, T, D>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    /// Create a new virtual boundary handler
    pub fn new(
        array: &'a ArrayView<'a, T, D>,
        mode: BorderMode,
        constant_value: Option<T>,
    ) -> Self {
        Self {
            array,
            mode,
            constant_value: constant_value.unwrap_or_else(T::zero),
        }
    }

    /// Convert out-of-bounds indices to valid indices based on the boundary mode
    fn convert_indices(&self, indices: &[isize]) -> Option<Vec<usize>> {
        let shape = self.array.shape();
        let mut converted = Vec::with_capacity(indices.len());

        for (i, &idx) in indices.iter().enumerate() {
            if i >= shape.len() {
                return None;
            }

            let dim_size = shape[i] as isize;

            let valid_idx = match self.mode {
                BorderMode::Constant => {
                    if idx < 0 || idx >= dim_size {
                        return None; // Signal to use constant value
                    }
                    idx as usize
                }
                BorderMode::Nearest => {
                    if idx < 0 {
                        0
                    } else if idx >= dim_size {
                        (dim_size - 1) as usize
                    } else {
                        idx as usize
                    }
                }
                BorderMode::Reflect => {
                    // Reflect at boundaries: -1 -> 1, -2 -> 2, n -> n-2, n+1 -> n-3
                    let mut reflected = idx;

                    if reflected < 0 {
                        reflected = -reflected;
                    }

                    if reflected >= dim_size {
                        // Reflect from the end
                        let over = reflected - dim_size + 1;
                        reflected = dim_size - 1 - over;
                    }

                    // Ensure we're in bounds after reflection
                    reflected.clamp(0, dim_size - 1) as usize
                }
                BorderMode::Mirror => {
                    // Mirror at boundaries: -1 -> 0, -2 -> 1, n -> n-1, n+1 -> n-2
                    let mut mirrored = idx;

                    // Handle negative indices
                    while mirrored < 0 {
                        mirrored = -mirrored - 1;
                    }

                    // Handle indices beyond array
                    while mirrored >= dim_size {
                        mirrored = 2 * dim_size - mirrored - 1;
                    }

                    mirrored.clamp(0, dim_size - 1) as usize
                }
                BorderMode::Wrap => {
                    // Periodic wrapping
                    let wrapped = ((idx % dim_size) + dim_size) % dim_size;
                    wrapped as usize
                }
            };

            converted.push(valid_idx);
        }

        Some(converted)
    }
}

impl<'a, T, D> BoundaryHandler<T, D> for VirtualBoundaryHandler<'a, T, D>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    fn get_value(&self, indices: &[isize]) -> T {
        match self.convert_indices(indices) {
            Some(valid_indices) => {
                // Use dynamic indexing
                let array_dyn = self.array.view().into_dyn();
                let dyn_indices = scirs2_core::ndarray::IxDyn(&valid_indices);
                array_dyn[dyn_indices]
            }
            None => {
                // Out of bounds for constant mode
                self.constant_value
            }
        }
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn ndim(&self) -> usize {
        self.array.ndim()
    }
}

/// Apply a filter using virtual boundary handling
///
/// This function applies a filter kernel to an array using on-the-fly boundary
/// value computation, avoiding the need to create a padded array.
#[allow(dead_code)]
pub fn apply_filter_with_boundary<T, D, F>(
    input: &Array<T, D>,
    kernelshape: &[usize],
    mode: BorderMode,
    constant_value: Option<T>,
    mut filter_fn: F,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync,
    D: Dimension,
    F: FnMut(&VirtualBoundaryHandler<T, D>, &[usize]) -> T,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if kernelshape.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Kernel shape must have same length as input dimensions (got {} expected {})",
            kernelshape.len(),
            input.ndim()
        )));
    }

    // Create output array
    let mut output = Array::<T, D>::zeros(input.raw_dim());

    // Create boundary handler
    let input_view = input.view();
    let handler = VirtualBoundaryHandler::new(&input_view, mode, constant_value);

    // Apply filter at each position
    // For now, we'll use a simple nested loop approach
    // In the future, this could be parallelized
    let shape = input.shape();
    let mut indices = vec![0usize; input.ndim()];

    fn increment_indices(indices: &mut [usize], shape: &[usize]) -> bool {
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < shape[i] {
                return true;
            }
            indices[i] = 0;
        }
        false
    }

    loop {
        // Apply filter at current position
        let value = filter_fn(&handler, &indices);

        // Convert output to dynamic for assignment
        let mut output_dyn = output.view_mut().into_dyn();
        let dyn_indices = scirs2_core::ndarray::IxDyn(&indices);
        output_dyn[dyn_indices] = value;

        // Move to next position
        if !increment_indices(&mut indices, shape) {
            break;
        }
    }

    Ok(output)
}

/// Optimized convolution using virtual boundary handling
#[allow(dead_code)]
pub fn convolve_optimized<T, D, E>(
    input: &Array<T, D>,
    kernel: &Array<T, E>,
    mode: BorderMode,
    constant_value: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync,
    D: Dimension,
    E: Dimension,
{
    let kernelshape: Vec<usize> = kernel.shape().to_vec();
    let kernel_center: Vec<isize> = kernelshape.iter().map(|&s| (s / 2) as isize).collect();

    // Clone kernel for use in closure
    let kernel_clone = kernel.clone();

    apply_filter_with_boundary(
        input,
        &kernelshape,
        mode,
        constant_value,
        |handler, center_pos| {
            let mut sum = T::zero();
            let mut kernel_indices = vec![0usize; handler.ndim()];

            // Helper function to increment kernel indices
            fn increment_kernel_indices(indices: &mut [usize], shape: &[usize]) -> bool {
                for i in (0..indices.len()).rev() {
                    indices[i] += 1;
                    if indices[i] < shape[i] {
                        return true;
                    }
                    indices[i] = 0;
                }
                false
            }

            // Iterate over kernel positions
            loop {
                // Calculate input position for this kernel element
                let mut input_indices = vec![0isize; handler.ndim()];
                for i in 0..handler.ndim() {
                    input_indices[i] =
                        center_pos[i] as isize + kernel_indices[i] as isize - kernel_center[i];
                }

                // Get _value from input (with boundary handling)
                let input_val = handler.get_value(&input_indices);

                // Get kernel _value (flipped for convolution)
                let mut flipped_indices = vec![0usize; handler.ndim()];
                for i in 0..handler.ndim() {
                    flipped_indices[i] = kernelshape[i] - kernel_indices[i] - 1;
                }
                let kernel_dyn = kernel_clone.view().into_dyn();
                let kernel_val = kernel_dyn[scirs2_core::ndarray::IxDyn(&flipped_indices)];

                sum = sum + input_val * kernel_val;

                // Move to next kernel position
                if !increment_kernel_indices(&mut kernel_indices, &kernelshape) {
                    break;
                }
            }

            sum
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_virtual_boundary_handler_constant() {
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let view = array.view();
        let handler = VirtualBoundaryHandler::new(&view, BorderMode::Constant, Some(0.0));

        // Test in-bounds access
        assert_eq!(handler.get_value(&[0, 0]), 1.0);
        assert_eq!(handler.get_value(&[1, 1]), 4.0);

        // Test out-of-bounds access (should return constant)
        assert_eq!(handler.get_value(&[-1, 0]), 0.0);
        assert_eq!(handler.get_value(&[2, 0]), 0.0);
        assert_eq!(handler.get_value(&[0, -1]), 0.0);
        assert_eq!(handler.get_value(&[0, 2]), 0.0);
    }

    #[test]
    fn test_virtual_boundary_handler_nearest() {
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let view = array.view();
        let handler = VirtualBoundaryHandler::new(&view, BorderMode::Nearest, None);

        // Test in-bounds access
        assert_eq!(handler.get_value(&[0, 0]), 1.0);
        assert_eq!(handler.get_value(&[1, 1]), 4.0);

        // Test out-of-bounds access (should return nearest)
        assert_eq!(handler.get_value(&[-1, 0]), 1.0); // Top edge
        assert_eq!(handler.get_value(&[2, 0]), 3.0); // Bottom edge
        assert_eq!(handler.get_value(&[0, -1]), 1.0); // Left edge
        assert_eq!(handler.get_value(&[0, 2]), 2.0); // Right edge
        assert_eq!(handler.get_value(&[-1, -1]), 1.0); // Top-left corner
        assert_eq!(handler.get_value(&[2, 2]), 4.0); // Bottom-right corner
    }

    #[test]
    fn test_virtual_boundary_handler_wrap() {
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let view = array.view();
        let handler = VirtualBoundaryHandler::new(&view, BorderMode::Wrap, None);

        // Test in-bounds access
        assert_eq!(handler.get_value(&[0, 0]), 1.0);
        assert_eq!(handler.get_value(&[1, 1]), 4.0);

        // Test out-of-bounds access (should wrap)
        assert_eq!(handler.get_value(&[-1, 0]), 3.0); // Wraps to bottom
        assert_eq!(handler.get_value(&[2, 0]), 1.0); // Wraps to top
        assert_eq!(handler.get_value(&[0, -1]), 2.0); // Wraps to right
        assert_eq!(handler.get_value(&[0, 2]), 1.0); // Wraps to left
        assert_eq!(handler.get_value(&[-1, -1]), 4.0); // Wraps to bottom-right
    }

    #[test]
    fn test_convolution_with_boundary() {
        let input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let kernel = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Simple diagonal kernel

        // Test with constant boundary
        let result = convolve_optimized(&input, &kernel, BorderMode::Constant, Some(0.0)).unwrap();

        // The result should preserve the input shape
        assert_eq!(result.shape(), input.shape());

        // Test specific values
        // At position (0,0): input[0,0] * kernel[1,1] + input[-1,-1] * kernel[0,0] = 1*1 + 0*1 = 1
        assert_eq!(result[[0, 0]], 1.0);

        // At position (1,1): input[1,1] * kernel[1,1] + input[0,0] * kernel[0,0] = 5*1 + 1*1 = 6
        assert_eq!(result[[1, 1]], 6.0);
    }
}
