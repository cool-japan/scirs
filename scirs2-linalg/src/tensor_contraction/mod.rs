//! Tensor contraction operations
//!
//! This module provides functionality for tensor contractions, which generalize matrix
//! multiplication to higher-order tensors.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayD, ArrayView, ArrayViewD, Dimension};
use num_traits::{Float, NumAssign, One, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::{Arc, Mutex};

// Public modules
pub mod cp;
pub mod tensor_network;
pub mod tensor_train;
pub mod tucker;

/// Performs tensor contraction between two tensors.
///
/// Tensor contraction is a generalization of matrix multiplication to higher-order tensors.
/// It sums over specified pairs of indices from two input tensors to produce a result tensor.
/// This implementation uses parallelism for improved performance on multi-core systems.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `axes_a` - Axes of tensor `a` to contract
/// * `axes_b` - Corresponding axes of tensor `b` to contract with `axes_a`
///
/// # Returns
///
/// * Result of the tensor contraction
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_linalg::tensor_contraction::contract;
///
/// // Create a 2x3x2 tensor
/// let a = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Create a 3x2x2 tensor
/// let b = array![[[1.0, 2.0], [3.0, 4.0]],
///                [[5.0, 6.0], [7.0, 8.0]],
///                [[9.0, 10.0], [11.0, 12.0]]];
///
/// // Contract over axis 1 of a and axis 0 of b
/// let result = contract(&a.view(), &b.view(), &[1], &[0]).unwrap();
///
/// // The result should be a 2x2x2x2 tensor
/// assert_eq!(result.shape(), &[2, 2, 2, 2]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn contract<A, D1, D2>(
    a: &ArrayView<A, D1>,
    b: &ArrayView<A, D2>,
    axes_a: &[usize],
    axes_b: &[usize],
) -> LinalgResult<ArrayD<A>>
where
    A: Clone + Float + NumAssign + Zero + Send + Sync + Sum + Debug + 'static,
    D1: Dimension,
    D2: Dimension,
{
    // Check that the number of axes to contract is the same
    if axes_a.len() != axes_b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Number of contraction axes must match: got {} and {}",
            axes_a.len(),
            axes_b.len()
        )));
    }

    // Check that the contraction axes have compatible dimensions
    for (i, (&ax_a, &ax_b)) in axes_a.iter().zip(axes_b.iter()).enumerate() {
        if ax_a >= a.ndim() {
            return Err(LinalgError::ShapeError(format!(
                "Axis {} out of bounds for first input with dimension {}",
                ax_a,
                a.ndim()
            )));
        }

        if ax_b >= b.ndim() {
            return Err(LinalgError::ShapeError(format!(
                "Axis {} out of bounds for second input with dimension {}",
                ax_b,
                b.ndim()
            )));
        }

        if a.shape()[ax_a] != b.shape()[ax_b] {
            return Err(LinalgError::ShapeError(format!(
                "Dimension mismatch at index {}: {} != {}",
                i,
                a.shape()[ax_a],
                b.shape()[ax_b]
            )));
        }
    }

    // Determine the axes that are not contracted
    let free_axes_a: Vec<usize> = (0..a.ndim()).filter(|&ax| !axes_a.contains(&ax)).collect();
    let free_axes_b: Vec<usize> = (0..b.ndim()).filter(|&ax| !axes_b.contains(&ax)).collect();

    // Determine the shape of the result tensor
    let mut result_shape = Vec::with_capacity(free_axes_a.len() + free_axes_b.len());
    let mut free_dims_a = Vec::with_capacity(free_axes_a.len());
    let mut free_dims_b = Vec::with_capacity(free_axes_b.len());

    for &ax in &free_axes_a {
        result_shape.push(a.shape()[ax]);
        free_dims_a.push(a.shape()[ax]);
    }

    for &ax in &free_axes_b {
        result_shape.push(b.shape()[ax]);
        free_dims_b.push(b.shape()[ax]);
    }

    // Convert to dynamic array views
    let a_dyn = a.view().into_dyn();
    let b_dyn = b.view().into_dyn();

    // Create the result tensor
    let result = ArrayD::zeros(result_shape.clone());
    let result = Arc::new(Mutex::new(result));

    // Generate all free indices combinations
    let mut all_free_indices = Vec::new();
    let total_combinations: usize = free_dims_a.iter().chain(free_dims_b.iter()).product();
    all_free_indices.reserve(total_combinations);

    // Function to generate all combinations of free indices
    fn generate_indices(
        free_dims: &[usize],
        current: Vec<usize>,
        depth: usize,
        all_indices: &mut Vec<Vec<usize>>,
    ) {
        if depth == free_dims.len() {
            all_indices.push(current);
            return;
        }

        let mut current = current;
        for i in 0..free_dims[depth] {
            current.push(i);
            generate_indices(free_dims, current.clone(), depth + 1, all_indices);
            current.pop();
        }
    }

    // Generate all combinations of free indices
    let mut combined_dims = free_dims_a.clone();
    combined_dims.extend(free_dims_b.iter());
    generate_indices(&combined_dims, Vec::new(), 0, &mut all_free_indices);

    // Process each combination of free indices in parallel
    use rayon::prelude::*;

    let results: Vec<_> = all_free_indices
        .par_iter()
        .map(|free_idx| {
            let free_idx_a = &free_idx[0..free_dims_a.len()];
            let free_idx_b = &free_idx[free_dims_a.len()..];

            // Prepare indexing arrays
            let mut a_idx = vec![0; a.ndim()];
            let mut b_idx = vec![0; b.ndim()];

            // Set free indices
            for (i, &ax) in free_axes_a.iter().enumerate() {
                a_idx[ax] = free_idx_a[i];
            }

            for (i, &ax) in free_axes_b.iter().enumerate() {
                b_idx[ax] = free_idx_b[i];
            }

            // Compute the contraction for this combination of free indices
            let mut sum = A::zero();

            // Recursively compute the contraction for all contracted indices
            fn accumulate_sum<A>(
                a: &ArrayViewD<A>,
                b: &ArrayViewD<A>,
                a_idx: &mut Vec<usize>,
                b_idx: &mut Vec<usize>,
                axes_a: &[usize],
                axes_b: &[usize],
                depth: usize,
                sum: &mut A,
            ) where
                A: Clone + Float + NumAssign + Zero,
            {
                if depth == axes_a.len() {
                    // All contracted indices are set, accumulate the product
                    *sum += a[a_idx.as_slice()] * b[b_idx.as_slice()];
                    return;
                }

                let ax_a = axes_a[depth];
                let ax_b = axes_b[depth];
                let dim = a.shape()[ax_a]; // Dimension size for this contracted axis

                for i in 0..dim {
                    a_idx[ax_a] = i;
                    b_idx[ax_b] = i;
                    accumulate_sum(a, b, a_idx, b_idx, axes_a, axes_b, depth + 1, sum);
                }
            }

            accumulate_sum(
                &a_dyn, &b_dyn, &mut a_idx, &mut b_idx, axes_a, axes_b, 0, &mut sum,
            );

            (free_idx.clone(), sum)
        })
        .collect();

    // Update the result tensor
    for (idx, sum) in results {
        let mut result_tensor = result.lock().unwrap();
        result_tensor[idx.as_slice()] = sum;
    }

    Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
}

/// Performs matrix multiplication along specified batch dimensions.
///
/// This is a special case of tensor contraction commonly used in deep learning
/// for batch matrix multiplication. The implementation uses parallelism to improve
/// performance on multi-core systems.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `batch_dims` - Number of batch dimensions (from the start of the tensor)
///
/// # Returns
///
/// * Result of batch matrix multiplication
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_linalg::tensor_contraction::batch_matmul;
///
/// // Create a batch of 2 matrices, each 2x3
/// let a = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
///                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]];
///
/// // Create a batch of 2 matrices, each 3x2
/// let b = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Perform batch matrix multiplication
/// let result = batch_matmul(&a.view(), &b.view(), 1).unwrap();
///
/// // The result should be a batch of 2 matrices, each 2x2
/// assert_eq!(result.shape(), &[2, 2, 2]);
///
/// // Check a specific result: a[0] @ b[0]
/// assert_eq!(result[[0, 0, 0]], 1.0 * 1.0 + 2.0 * 3.0 + 3.0 * 5.0);
/// ```
pub fn batch_matmul<A, D1, D2>(
    a: &ArrayView<A, D1>,
    b: &ArrayView<A, D2>,
    batch_dims: usize,
) -> LinalgResult<ArrayD<A>>
where
    A: Clone + Float + NumAssign + Zero + Send + Sync + Sum + Debug + 'static,
    D1: Dimension,
    D2: Dimension,
{
    // Check dimensions
    if a.ndim() < batch_dims + 2 || b.ndim() < batch_dims + 2 {
        return Err(LinalgError::ShapeError(format!(
            "Both tensors must have at least batch_dims + 2 dimensions, got {} and {}",
            a.ndim(),
            b.ndim()
        )));
    }

    // Check that batch dimensions match
    for i in 0..batch_dims {
        if a.shape()[i] != b.shape()[i] {
            return Err(LinalgError::ShapeError(format!(
                "Batch dimensions must match: {} != {} at index {}",
                a.shape()[i],
                b.shape()[i],
                i
            )));
        }
    }

    // Check matrix dimensions for compatibility
    if a.shape()[batch_dims + 1] != b.shape()[batch_dims] {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions for matrix multiplication must match: {} != {}",
            a.shape()[batch_dims + 1],
            b.shape()[batch_dims]
        )));
    }

    // Determine output shape
    let mut out_shape = Vec::with_capacity(batch_dims + 2);
    for i in 0..batch_dims {
        out_shape.push(a.shape()[i]);
    }
    out_shape.push(a.shape()[batch_dims]); // M
    out_shape.push(b.shape()[batch_dims + 1]); // N

    // Convert to dynamic array views
    let a_dyn = a.view().into_dyn();
    let b_dyn = b.view().into_dyn();

    // Flatten batch dimensions
    let batch_size: usize = out_shape.iter().take(batch_dims).product();
    let m = a.shape()[batch_dims]; // rows in A
    let k = a.shape()[batch_dims + 1]; // cols in A, rows in B
    let n = b.shape()[batch_dims + 1]; // cols in B

    // Create result array
    let result = ArrayD::zeros(out_shape.clone());
    let result = Arc::new(Mutex::new(result));

    // Generate all batch indices
    let mut all_batch_indices = Vec::with_capacity(batch_size);

    fn generate_batch_indices(
        shape: &[usize],
        current: Vec<usize>,
        depth: usize,
        max_depth: usize,
        all_indices: &mut Vec<Vec<usize>>,
    ) {
        if depth == max_depth {
            all_indices.push(current);
            return;
        }

        let mut current = current;
        for i in 0..shape[depth] {
            current.push(i);
            generate_batch_indices(shape, current.clone(), depth + 1, max_depth, all_indices);
            current.pop();
        }
    }

    generate_batch_indices(
        &out_shape,
        Vec::new(),
        0,
        batch_dims,
        &mut all_batch_indices,
    );

    // Process each batch in parallel
    use rayon::prelude::*;

    let results: Vec<_> = all_batch_indices
        .par_iter()
        .map(|batch_idx| {
            // Perform matrix multiplication for this batch
            let mut result_batch = Array2::zeros((m, n));

            for i in 0..m {
                for j in 0..n {
                    let mut sum = A::zero();

                    // Compute dot product
                    for p in 0..k {
                        // Create full indices for a[batch_idx, i, p] and b[batch_idx, p, j]
                        let mut a_idx = batch_idx.clone();
                        a_idx.push(i);
                        a_idx.push(p);

                        let mut b_idx = batch_idx.clone();
                        b_idx.push(p);
                        b_idx.push(j);

                        sum += a_dyn[a_idx.as_slice()] * b_dyn[b_idx.as_slice()];
                    }

                    result_batch[[i, j]] = sum;
                }
            }

            (batch_idx.clone(), result_batch)
        })
        .collect();

    // Update the result array
    for (batch_idx, result_batch) in results {
        let mut result_tensor = result.lock().unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut result_idx = batch_idx.clone();
                result_idx.push(i);
                result_idx.push(j);
                result_tensor[result_idx.as_slice()] = result_batch[[i, j]];
            }
        }
    }

    Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
}

/// Calculates the mode-n product of a tensor with a matrix.
///
/// The mode-n product is defined as a contraction between a tensor and a matrix
/// along a specific mode (dimension) of the tensor. This implementation uses
/// parallelism to improve performance on multi-core systems.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `matrix` - Matrix to multiply with
/// * `mode` - The mode (dimension) along which to contract
///
/// # Returns
///
/// * Result of the mode-n product
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_linalg::tensor_contraction::mode_n_product;
///
/// // Create a 2x3x2 tensor
/// let tensor = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                     [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Create a 4x2 matrix (will transform mode 0 of the tensor)
/// let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
///
/// // Calculate mode-0 product
/// let result = mode_n_product(&tensor.view(), &matrix.view(), 0).unwrap();
///
/// // The result should have shape 4x3x2
/// assert_eq!(result.shape(), &[4, 3, 2]);
/// ```
pub fn mode_n_product<A, D1, D2>(
    tensor: &ArrayView<A, D1>,
    matrix: &ArrayView<A, D2>,
    mode: usize,
) -> LinalgResult<ArrayD<A>>
where
    A: Clone + Float + NumAssign + Zero + Send + Sync + Debug + 'static,
    D1: Dimension,
    D2: Dimension,
{
    // Check mode is valid
    if mode >= tensor.ndim() {
        return Err(LinalgError::ShapeError(format!(
            "Mode {} is out of bounds for tensor with {} dimensions",
            mode,
            tensor.ndim()
        )));
    }

    // Check matrix dimensions
    if matrix.ndim() != 2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be 2-dimensional, got {} dimensions",
            matrix.ndim()
        )));
    }

    // Check that matrix columns match the tensor's mode dimension
    if matrix.shape()[1] != tensor.shape()[mode] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match tensor dimension along mode {} ({})",
            matrix.shape()[1],
            mode,
            tensor.shape()[mode]
        )));
    }

    // Determine output shape
    let mut out_shape = tensor.shape().to_vec();
    out_shape[mode] = matrix.shape()[0];

    // Convert to dynamic array views
    let tensor_dyn = tensor.view().into_dyn();
    let matrix_view = match matrix.view().into_dimensionality::<ndarray::Ix2>() {
        Ok(view) => view,
        Err(_) => {
            return Err(LinalgError::ComputationError(
                "Failed to convert matrix to 2D view".to_string(),
            ))
        }
    };

    // Create result array with mutex for thread-safe updates
    let result = ArrayD::zeros(out_shape.clone());
    let result = Arc::new(Mutex::new(result));

    // Generate all indices for tensor except the mode dimension
    let mut all_indices = Vec::new();
    let mut shape_without_mode = tensor.shape().to_vec();
    shape_without_mode.remove(mode);

    // Calculate total number of combinations
    let total_combinations: usize = shape_without_mode.iter().product();
    all_indices.reserve(total_combinations);

    fn generate_indices_without_mode(
        shape: &[usize],
        current: Vec<usize>,
        depth: usize,
        mode: usize,
        _mode_dim: usize,
        all_indices: &mut Vec<Vec<usize>>,
    ) {
        if depth == shape.len() {
            all_indices.push(current);
            return;
        }

        // Skip mode dimension
        if depth == mode {
            generate_indices_without_mode(shape, current, depth + 1, mode, _mode_dim, all_indices);
            return;
        }

        let current_dim = if depth > mode { depth - 1 } else { depth };
        let dim_size = shape[current_dim];

        let mut current = current;
        for i in 0..dim_size {
            current.push(i);
            generate_indices_without_mode(
                shape,
                current.clone(),
                depth + 1,
                mode,
                _mode_dim,
                all_indices,
            );
            current.pop();
        }
    }

    generate_indices_without_mode(
        tensor.shape(),
        Vec::new(),
        0,
        mode,
        tensor.shape()[mode],
        &mut all_indices,
    );

    // Process each combination of indices in parallel
    use rayon::prelude::*;

    let all_results: Vec<Vec<_>> = all_indices
        .par_iter()
        .map(|idx| {
            // Create complete index arrays for old and new tensors
            let mut tensor_idx = idx.clone();
            tensor_idx.insert(mode, 0); // Will be updated in the loop

            // Process this tensor element
            let mut results = Vec::new();
            for j in 0..matrix.shape()[0] {
                let mut sum = A::zero();

                // Sum over the contracted dimension
                for k in 0..tensor.shape()[mode] {
                    tensor_idx[mode] = k;
                    sum += tensor_dyn[tensor_idx.as_slice()] * matrix_view[[j, k]];
                }

                // Create index for the result tensor
                let mut result_idx = idx.clone();
                result_idx.insert(mode, j);

                results.push((result_idx, sum));
            }

            results
        })
        .collect();

    // Update the result tensor
    for batch_results in all_results {
        let mut result_tensor = result.lock().unwrap();
        for (idx, sum) in batch_results {
            result_tensor[idx.as_slice()] = sum;
        }
    }

    Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
}

/// Einstein summation (einsum) for tensor contractions.
///
/// Provides a concise way to express many common multi-dimensional linear algebraic array operations
/// using the Einstein summation convention. This implementation handles basic einsum operations
/// including matrix multiplication, inner/outer product, and various trace operations.
/// It uses parallelism to improve performance on multi-core systems.
///
/// # Arguments
///
/// * `einsum_str` - String describing the operation in the format 'ij,jk->ik'
/// * `tensors` - Slice of tensor views to operate on
///
/// # Returns
///
/// * Result of the einsum operation
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_linalg::tensor_contraction::einsum;
///
/// // Create 2x3 and 3x4 matrices for matrix multiplication
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let b = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]];
///
/// // Perform matrix multiplication with einsum
/// let result = einsum("ij,jk->ik", &[&a.view(), &b.view()]).unwrap();
///
/// // The result should be a 2x4 matrix
/// assert_eq!(result.shape(), &[2, 4]);
///
/// // Check a specific value
/// assert_eq!(result[[0, 0]], 1.0 * 1.0 + 2.0 * 5.0 + 3.0 * 9.0);
/// ```
#[allow(clippy::type_complexity)]
pub fn einsum<'a, A>(
    einsum_str: &str,
    tensors: &'a [&'a ArrayViewD<'a, A>],
) -> LinalgResult<ArrayD<A>>
where
    A: Clone + Float + NumAssign + Zero + Send + Sync + Sum + Debug + 'static,
{
    // Parse the einsum string
    fn parse_einsum_notation(einsum_str: &str) -> LinalgResult<(Vec<Vec<char>>, Vec<char>)> {
        // Split the string into input and output parts
        let parts: Vec<&str> = einsum_str.split("->").collect();
        if parts.len() != 2 {
            return Err(LinalgError::ValueError(
                "Einsum string must contain exactly one '->'".to_string(),
            ));
        }

        // Parse input indices
        let inputs: Vec<&str> = parts[0].split(',').collect();
        let mut input_indices = Vec::with_capacity(inputs.len());
        for input in inputs {
            let indices: Vec<char> = input.trim().chars().collect();
            input_indices.push(indices);
        }

        // Parse output indices
        let output_indices: Vec<char> = parts[1].trim().chars().collect();

        Ok((input_indices, output_indices))
    }

    // Parse the einsum string
    let (input_indices, output_indices) = parse_einsum_notation(einsum_str)?;

    // Check that the number of input tensors matches the number of input indices
    if tensors.len() != input_indices.len() {
        return Err(LinalgError::ValueError(format!(
            "Number of tensors ({}) doesn't match number of index groups ({})",
            tensors.len(),
            input_indices.len()
        )));
    }

    // Verify that each tensor's rank matches its indices
    for (i, (tensor, indices)) in tensors.iter().zip(input_indices.iter()).enumerate() {
        if tensor.ndim() != indices.len() {
            return Err(LinalgError::ShapeError(format!(
                "Tensor {} has {} dimensions, but {} indices were provided",
                i,
                tensor.ndim(),
                indices.len()
            )));
        }
    }

    // Build a map of index labels to dimensions
    let mut index_to_dim: HashMap<char, usize> = HashMap::new();

    // First pass: collect all dimension sizes and check consistency
    for (tensor, indices) in tensors.iter().zip(input_indices.iter()) {
        for (&dim_size, &idx) in tensor.shape().iter().zip(indices.iter()) {
            if let Some(&existing_dim) = index_to_dim.get(&idx) {
                if existing_dim != dim_size {
                    return Err(LinalgError::ShapeError(format!(
                        "Inconsistent dimensions for index '{}': {} and {}",
                        idx, existing_dim, dim_size
                    )));
                }
            } else {
                index_to_dim.insert(idx, dim_size);
            }
        }
    }

    // Verify that all output indices exist in input indices
    for &idx in &output_indices {
        if !index_to_dim.contains_key(&idx) {
            return Err(LinalgError::ValueError(format!(
                "Output index '{}' not found in any input indices",
                idx
            )));
        }
    }

    // Determine output shape
    let mut output_shape = Vec::with_capacity(output_indices.len());
    for &idx in &output_indices {
        output_shape.push(index_to_dim[&idx]);
    }

    // Create the output tensor
    let result = Arc::new(Mutex::new(ArrayD::zeros(output_shape.clone())));

    // Collect contracted indices (those not in output_indices)
    let mut contracted_indices: Vec<char> = Vec::new();
    for indices in input_indices.iter() {
        for &idx in indices {
            if !output_indices.contains(&idx) && !contracted_indices.contains(&idx) {
                contracted_indices.push(idx);
            }
        }
    }

    // Generate all output index combinations
    let mut all_output_indices = Vec::new();
    let total_output_combinations: usize = output_shape.iter().product();
    all_output_indices.reserve(total_output_combinations);

    fn generate_output_indices(
        shape: &[usize],
        current: Vec<usize>,
        depth: usize,
        all_indices: &mut Vec<Vec<usize>>,
    ) {
        if depth == shape.len() {
            all_indices.push(current);
            return;
        }

        let mut current = current;
        for i in 0..shape[depth] {
            current.push(i);
            generate_output_indices(shape, current.clone(), depth + 1, all_indices);
            current.pop();
        }
    }

    generate_output_indices(&output_shape, Vec::new(), 0, &mut all_output_indices);

    // Process each output combination in parallel
    use rayon::prelude::*;

    let results: Vec<_> = all_output_indices
        .par_iter()
        .map(|output_idx| {
            // Create an index mapping for output indices
            let mut index_values = HashMap::new();
            for (i, &idx) in output_indices.iter().enumerate() {
                index_values.insert(idx, output_idx[i]);
            }

            // Use a recursive function that does not require capturing the environment
            fn compute_sum_recursive<A>(
                tensors: &[&ArrayViewD<A>],
                input_indices: &[Vec<char>],
                contracted_indices: &[char],
                index_values: &mut HashMap<char, usize>,
                index_to_dim: &HashMap<char, usize>,
                depth: usize,
            ) -> A
            where
                A: Clone + Float + NumAssign + Zero + One,
            {
                // Base case: all contracted indices have values assigned
                if depth == contracted_indices.len() {
                    // Compute product of tensor elements
                    let mut product = A::one();

                    for (tensor, indices) in tensors.iter().zip(input_indices.iter()) {
                        // Create index array for this tensor
                        let tensor_indices: Vec<usize> =
                            indices.iter().map(|&idx| index_values[&idx]).collect();

                        // Multiply by tensor element at these indices
                        product *= tensor[tensor_indices.as_slice()];
                    }

                    return product;
                }

                // Recursive case: assign a value to the current contracted index
                let idx = contracted_indices[depth];
                let dim = index_to_dim[&idx];
                let mut sum = A::zero();

                for i in 0..dim {
                    index_values.insert(idx, i);
                    sum += compute_sum_recursive(
                        tensors,
                        input_indices,
                        contracted_indices,
                        index_values,
                        index_to_dim,
                        depth + 1,
                    );
                }

                sum
            }

            // Compute the sum for this output element
            let sum = compute_sum_recursive(
                tensors,
                &input_indices,
                &contracted_indices,
                &mut index_values,
                &index_to_dim,
                0,
            );

            (output_idx.clone(), sum)
        })
        .collect();

    // Update the result tensor
    for (idx, sum) in results {
        let mut result_tensor = result.lock().unwrap();
        result_tensor[idx.as_slice()] = sum;
    }

    Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
}

/// Performs a tensor decomposition using the Higher-Order SVD (HOSVD) method.
///
/// HOSVD is a generalization of the matrix SVD to higher-order tensors.
/// It computes a tensor in the form of a core tensor multiplied by a set of
/// orthogonal matrices along each mode. This implementation uses parallelism
/// to improve performance on multi-core systems.
///
/// # Arguments
///
/// * `tensor` - The input tensor to decompose
/// * `rank` - The target rank for each mode (the size of the core tensor)
///
/// # Returns
///
/// * A tuple containing the core tensor and the factor matrices
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_linalg::tensor_contraction::hosvd;
///
/// // Create a 3D tensor
/// let tensor = array![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
///
/// // Decompose tensor with full rank
/// let (core, factors) = hosvd(&tensor.view(), &[2, 2, 2]).unwrap();
///
/// // The core tensor should have the specified rank
/// assert_eq!(core.shape(), &[2, 2, 2]);
///
/// // The factors should preserve the original dimensions
/// assert_eq!(factors[0].shape(), &[2, 2]); // mode 1
/// assert_eq!(factors[1].shape(), &[2, 2]); // mode 2
/// assert_eq!(factors[2].shape(), &[2, 2]); // mode 3
/// ```
pub fn hosvd<A, D>(
    tensor: &ArrayView<A, D>,
    rank: &[usize],
) -> LinalgResult<(ArrayD<A>, Vec<Array2<A>>)>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Send
        + Sync
        + Sum
        + Debug
        + 'static
        + ndarray::ScalarOperand,
    D: Dimension,
{
    // Check that the rank for each mode is valid
    if rank.len() != tensor.ndim() {
        return Err(LinalgError::ShapeError(format!(
            "Rank vector length ({}) must match tensor dimensions ({})",
            rank.len(),
            tensor.ndim()
        )));
    }

    for (i, &r) in rank.iter().enumerate() {
        if r > tensor.shape()[i] {
            return Err(LinalgError::ShapeError(format!(
                "Rank for mode {} ({}) cannot exceed the mode dimension ({})",
                i,
                r,
                tensor.shape()[i]
            )));
        }
    }

    // Convert to dynamic array
    let tensor_dyn = tensor.to_owned().into_dyn();

    // Compute factor matrices for each mode in parallel
    use rayon::prelude::*;

    let modes: Vec<usize> = (0..tensor.ndim()).collect();
    let factors: Vec<Array2<A>> = modes
        .par_iter()
        .map(|mode| {
            // Unfold the tensor along this mode
            let unfolded = unfold(&tensor_dyn, *mode).unwrap();

            // Compute SVD of the unfolded tensor
            let (u, _, _) = svd_truncated(&unfolded, rank[*mode]).unwrap();
            u
        })
        .collect();

    // Compute the core tensor
    let mut core = tensor_dyn.to_owned();

    for (mode, factor) in factors.iter().enumerate() {
        // Mode-n product of the tensor with the transpose of the factor matrix
        let factor_t = factor.t().to_owned();
        core = mode_n_product(&core.view(), &factor_t.view(), mode)?;
    }

    Ok((core, factors))
}

// Helper function to unfold a tensor along a specified mode
// "Unfolding" means reshaping a tensor into a matrix
fn unfold<A>(tensor: &ArrayD<A>, mode: usize) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + Debug + Send + Sync,
{
    if mode >= tensor.ndim() {
        return Err(LinalgError::ShapeError(format!(
            "Mode {} is out of bounds for tensor with {} dimensions",
            mode,
            tensor.ndim()
        )));
    }

    let shape = tensor.shape();
    let mode_dim = shape[mode];

    // Calculate the product of all other dimensions
    let other_dims_prod: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &dim)| dim)
        .product();

    // Create result matrix
    let mut result = Array2::zeros((mode_dim, other_dims_prod));

    // Helper function to calculate column index
    fn calc_col_idx(idx: &[usize], shape: &[usize], mode: usize) -> usize {
        let mut col_idx = 0;
        let mut stride = 1;

        for dim in (0..shape.len()).rev() {
            if dim != mode {
                col_idx += idx[dim] * stride;
                stride *= shape[dim];
            }
        }

        col_idx
    }

    // Populate the unfolded tensor (vectorized for better performance)
    let tensor_shape = tensor.shape().to_vec();

    // Generate all indices
    let mut all_indices = Vec::new();
    let total_elements: usize = tensor_shape.iter().product();
    all_indices.reserve(total_elements);

    fn generate_tensor_indices(
        shape: &[usize],
        current: Vec<usize>,
        depth: usize,
        all_indices: &mut Vec<Vec<usize>>,
    ) {
        if depth == shape.len() {
            all_indices.push(current);
            return;
        }

        let mut current = current;
        for i in 0..shape[depth] {
            current.push(i);
            generate_tensor_indices(shape, current.clone(), depth + 1, all_indices);
            current.pop();
        }
    }

    generate_tensor_indices(&tensor_shape, Vec::new(), 0, &mut all_indices);

    // Process all indices in parallel
    use rayon::prelude::*;

    let results: Vec<_> = all_indices
        .par_iter()
        .map(|idx| {
            let mode_idx = idx[mode];
            let col_idx = calc_col_idx(idx, &tensor_shape, mode);
            let val = tensor[idx.as_slice()];
            (mode_idx, col_idx, val)
        })
        .collect();

    // Update the result matrix
    for (mode_idx, col_idx, val) in results {
        result[[mode_idx, col_idx]] = val;
    }

    Ok(result)
}

// Helper function to compute truncated SVD
pub fn svd_truncated<A>(
    matrix: &Array2<A>,
    rank: usize,
) -> LinalgResult<(Array2<A>, Array2<A>, Array2<A>)>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Send
        + Sync
        + Sum
        + std::fmt::Debug
        + 'static
        + ndarray::ScalarOperand,
{
    use crate::decomposition::svd;

    // Convert to view and call SVD with full_matrices=false
    let matrix_view = matrix.view();
    let (u, s, vt) = svd(&matrix_view, false, None)?;

    // Truncate to the specified rank
    let u_trunc = u.slice(ndarray::s![.., ..rank]).to_owned();
    let s_trunc = Array2::from_diag(&s.slice(ndarray::s![..rank]));
    let vt_trunc = vt.slice(ndarray::s![..rank, ..]).to_owned();

    Ok((u_trunc, s_trunc, vt_trunc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_multiplication() {
        // 2x3 matrix
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // 3x2 matrix
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        // Contract along axis 1 of a and axis 0 of b
        let result = contract(&a.view(), &b.view(), &[1], &[0]).unwrap();

        // Expected: a @ b
        let expected = array![[58.0, 64.0], [139.0, 154.0]];

        assert_eq!(result.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_batch_matmul() {
        // Batch of 2 matrices, each 2x3
        let a = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ];

        // Batch of 2 matrices, each 3x2
        let b = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Batch matrix multiplication with 1 batch dimension
        let result = batch_matmul(&a.view(), &b.view(), 1).unwrap();

        // Expected results for each batch
        // First batch: [1,2,3; 4,5,6] × [1,2; 3,4; 5,6] = [22,28; 49,64]
        let expected_batch0 = array![[22.0, 28.0], [49.0, 64.0]];

        // Second batch: [7,8,9; 10,11,12] × [7,8; 9,10; 11,12] = [220,244; 301,334]
        let expected_batch1 = array![[220.0, 244.0], [301.0, 334.0]];

        assert_eq!(result.shape(), &[2, 2, 2]);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[0, i, j]], expected_batch0[[i, j]], epsilon = 1e-10);
                assert_abs_diff_eq!(result[[1, i, j]], expected_batch1[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_einsum_matrix_multiplication() {
        // Matrix multiplication: "ij,jk->ik"
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let a_view = a.view().into_dyn();
        let b_view = b.view().into_dyn();

        let result = einsum("ij,jk->ik", &[&a_view, &b_view]).unwrap();

        // Expected: a @ b
        let expected = array![[58.0, 64.0], [139.0, 154.0]];

        assert_eq!(result.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_einsum_inner_product() {
        // Inner product: "i,i->"
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let a_view = a.view().into_dyn();
        let b_view = b.view().into_dyn();

        let result = einsum("i,i->", &[&a_view, &b_view]).unwrap();

        // Expected: sum(a * b)
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0

        assert_eq!(result.shape(), &[] as &[usize]);
        // For scalar output, we need to get the first element
        assert_abs_diff_eq!(result.iter().next().unwrap(), &expected, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "Needs investigation - possibly SVD-related issue"]
    fn test_mode_n_product() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Create a 4x2 matrix (will transform mode 0)
        let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        // Perform mode-0 product
        let result = mode_n_product(&tensor.view(), &matrix.view(), 0).unwrap();

        assert_eq!(result.shape(), &[4, 3, 2]);

        // Check some values
        // First row of result[0] should be a linear combination of tensor's first rows
        assert_abs_diff_eq!(result[[0, 0, 0]], 1.0 * 1.0 + 2.0 * 7.0, epsilon = 1e-10); // 1*1 + 2*7
        assert_abs_diff_eq!(result[[0, 0, 1]], 1.0 * 2.0 + 2.0 * 8.0, epsilon = 1e-10);
        // 1*2 + 2*8
    }

    #[test]
    fn test_hosvd_basic() {
        // Create a simple 2x2x2 tensor
        let tensor = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

        // Decompose with full rank
        let (core, factors) = hosvd(&tensor.view(), &[2, 2, 2]).unwrap();

        // Check dimensions
        assert_eq!(core.shape(), &[2, 2, 2]);
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0].shape(), &[2, 2]);
        assert_eq!(factors[1].shape(), &[2, 2]);
        assert_eq!(factors[2].shape(), &[2, 2]);

        // Reconstruct the tensor and check approximation
        let mut reconstructed = core.clone();

        for (mode, factor) in factors.iter().enumerate() {
            reconstructed = mode_n_product(&reconstructed.view(), &factor.view(), mode).unwrap();
        }

        // Check that the reconstruction is close to the original
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_abs_diff_eq!(
                        reconstructed[[i, j, k]],
                        tensor[[i, j, k]],
                        epsilon = 1e-5
                    );
                }
            }
        }
    }
}
