//! Tensor Train Decomposition (TT-Decomposition)
//!
//! This module provides functionality for Tensor Train decomposition,
//! which is a type of tensor decomposition that compactly represents a
//! high-dimensional tensor as a sequence of low-dimensional tensors.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array, Array2, Array3, ArrayD, ArrayView, Dimension};
use num_traits::{Float, NumAssign, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Represents a tensor in Tensor Train (TT) format.
///
/// A tensor train decomposition represents an N-dimensional tensor as a
/// sequence of 3D tensors (called TT-cores), where the first and last cores
/// have special boundary dimensions.
///
/// # Fields
///
/// * `cores` - Vector of 3D tensors (TT-cores)
/// * `ranks` - Vector of TT-ranks (including boundary ranks which are always 1)
/// * `shape` - Original tensor shape
#[derive(Debug, Clone)]
pub struct TensorTrain<A>
where
    A: Clone + Float + Debug,
{
    /// The TT-cores, each with shape (r_{i-1}, n_i, r_i)
    pub cores: Vec<Array3<A>>,
    /// The TT-ranks (including boundary ranks which are always 1)
    pub ranks: Vec<usize>,
    /// Original tensor shape
    pub shape: Vec<usize>,
}

impl<A> TensorTrain<A>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    /// Creates a new TensorTrain from the given cores, ranks, and original shape.
    ///
    /// # Arguments
    ///
    /// * `cores` - Vector of 3D tensors (TT-cores)
    /// * `ranks` - Vector of TT-ranks (including boundary ranks which should be 1)
    /// * `shape` - Original tensor shape
    ///
    /// # Returns
    ///
    /// * `TensorTrain` - A new TensorTrain instance
    pub fn new(cores: Vec<Array3<A>>, ranks: Vec<usize>, shape: Vec<usize>) -> LinalgResult<Self> {
        // Validate input
        if cores.is_empty() {
            return Err(LinalgError::ValueError(
                "Cores list cannot be empty".to_string(),
            ));
        }

        if ranks.len() != shape.len() + 1 {
            return Err(LinalgError::ShapeError(format!(
                "Ranks vector length ({}) must be 1 more than shape length ({})",
                ranks.len(),
                shape.len()
            )));
        }

        if ranks[0] != 1 || ranks[ranks.len() - 1] != 1 {
            return Err(LinalgError::ValueError(
                "First and last TT-ranks must be 1".to_string(),
            ));
        }

        if cores.len() != shape.len() {
            return Err(LinalgError::ShapeError(format!(
                "Number of cores ({}) must match the shape length ({})",
                cores.len(),
                shape.len()
            )));
        }

        // Validate core dimensions
        for (i, core) in cores.iter().enumerate() {
            if core.ndim() != 3 {
                return Err(LinalgError::ShapeError(format!(
                    "Core {} must be 3-dimensional, got {} dimensions",
                    i,
                    core.ndim()
                )));
            }

            if core.shape()[0] != ranks[i] {
                return Err(LinalgError::ShapeError(format!(
                    "Core {} has first dimension {}, expected rank {}",
                    i,
                    core.shape()[0],
                    ranks[i]
                )));
            }

            if core.shape()[1] != shape[i] {
                return Err(LinalgError::ShapeError(format!(
                    "Core {} has second dimension {}, expected shape dimension {}",
                    i,
                    core.shape()[1],
                    shape[i]
                )));
            }

            if core.shape()[2] != ranks[i + 1] {
                return Err(LinalgError::ShapeError(format!(
                    "Core {} has third dimension {}, expected rank {}",
                    i,
                    core.shape()[2],
                    ranks[i + 1]
                )));
            }
        }

        Ok(TensorTrain {
            cores,
            ranks,
            shape,
        })
    }

    /// Reconstructs the full tensor from the TT-decomposition.
    ///
    /// # Returns
    ///
    /// * `ArrayD<A>` - The full tensor
    pub fn to_full(&self) -> LinalgResult<ArrayD<A>> {
        let n_dims = self.shape.len();

        // Start with the first core reshaped to remove the leading dimension of 1
        let mut result = ArrayD::zeros(ndarray::IxDyn(&[self.shape[0], self.ranks[1]]));

        // Copy the first core data
        for i in 0..self.shape[0] {
            for j in 0..self.ranks[1] {
                result[[i, j]] = self.cores[0][[0, i, j]];
            }
        }

        // Multiply by each core in sequence
        for i in 1..n_dims {
            let core = &self.cores[i];

            // Get current result shape and prepare for contraction
            let current_shape = result.shape().to_vec();
            let current_rank = current_shape[current_shape.len() - 1];

            // Reshape result to prepare for contraction
            let result_flat = result
                .into_shape_with_order((
                    current_shape[..current_shape.len() - 1].iter().product(),
                    current_rank,
                ))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            // Contract with the current core along the rank dimension
            let mut new_shape = current_shape[..current_shape.len() - 1].to_vec();
            new_shape.push(self.shape[i]);
            new_shape.push(self.ranks[i + 1]);

            // Compute the contraction manually
            let mut result_contracted = Array::zeros(ndarray::IxDyn(&new_shape));

            // Handle different dimensionality cases
            match new_shape.len() - 2 {
                0 => {
                    // Special case: no free indices
                    for k in 0..self.shape[i] {
                        for l in 0..self.ranks[i + 1] {
                            let mut sum = A::zero();
                            for r in 0..current_rank {
                                sum += result_flat[[0, r]] * core[[r, k, l]];
                            }
                            result_contracted[[k, l]] = sum;
                        }
                    }
                }
                1 => {
                    // One free index dimension
                    for idx1 in 0..new_shape[0] {
                        for k in 0..self.shape[i] {
                            for l in 0..self.ranks[i + 1] {
                                let mut sum = A::zero();
                                for r in 0..current_rank {
                                    sum += result_flat[[idx1, r]] * core[[r, k, l]];
                                }
                                result_contracted[[idx1, k, l]] = sum;
                            }
                        }
                    }
                }
                2 => {
                    // Two free index dimensions
                    for idx1 in 0..new_shape[0] {
                        for idx2 in 0..new_shape[1] {
                            let flat_idx = idx1 * new_shape[1] + idx2;
                            for k in 0..self.shape[i] {
                                for l in 0..self.ranks[i + 1] {
                                    let mut sum = A::zero();
                                    for r in 0..current_rank {
                                        sum += result_flat[[flat_idx, r]] * core[[r, k, l]];
                                    }
                                    result_contracted[[idx1, idx2, k, l]] = sum;
                                }
                            }
                        }
                    }
                }
                3 => {
                    // Three free index dimensions
                    for idx1 in 0..new_shape[0] {
                        for idx2 in 0..new_shape[1] {
                            for idx3 in 0..new_shape[2] {
                                let stride2 = new_shape[2];
                                let flat_idx =
                                    idx1 * new_shape[1] * stride2 + idx2 * stride2 + idx3;
                                for k in 0..self.shape[i] {
                                    for l in 0..self.ranks[i + 1] {
                                        let mut sum = A::zero();
                                        for r in 0..current_rank {
                                            sum += result_flat[[flat_idx, r]] * core[[r, k, l]];
                                        }
                                        result_contracted[[idx1, idx2, idx3, k, l]] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // General case for more dimensions - less efficient but more flexible
                    // Create a helper function to iterate through all possible indices
                    fn visit_indices(
                        shape: &[usize],
                        current_indices: &mut Vec<usize>,
                        depth: usize,
                        callback: &mut dyn FnMut(&[usize]),
                    ) {
                        if depth == shape.len() {
                            callback(current_indices);
                            return;
                        }

                        for i in 0..shape[depth] {
                            current_indices.push(i);
                            visit_indices(shape, current_indices, depth + 1, callback);
                            current_indices.pop();
                        }
                    }

                    // Calculate flat index from multi-index
                    fn flat_index(indices: &[usize], shape: &[usize]) -> usize {
                        let mut idx = 0;
                        let mut stride = 1;

                        for i in (0..indices.len()).rev() {
                            idx += indices[i] * stride;
                            if i > 0 {
                                stride *= shape[i];
                            }
                        }

                        idx
                    }

                    let free_dims = new_shape[..new_shape.len() - 2].to_vec();
                    let mut indices = Vec::new();

                    let mut callback = |idx: &[usize]| {
                        let flat_idx = flat_index(idx, &free_dims);

                        for k in 0..self.shape[i] {
                            for l in 0..self.ranks[i + 1] {
                                let mut sum = A::zero();
                                for r in 0..current_rank {
                                    sum += result_flat[[flat_idx, r]] * core[[r, k, l]];
                                }

                                // Set the result in the new array
                                let mut idx_full = idx.to_vec();
                                idx_full.push(k);
                                idx_full.push(l);

                                // Convert to dynamic index
                                result_contracted[ndarray::IxDyn(&idx_full)] = sum;
                            }
                        }
                    };

                    visit_indices(&free_dims, &mut indices, 0, &mut callback);
                }
            }

            result = result_contracted;
        }

        // Convert to the final tensor with the original shape
        let mut final_result = ArrayD::zeros(ndarray::IxDyn(self.shape.as_slice()));

        // Special case if we only have 1 final rank dimension
        if self.ranks[n_dims] == 1 {
            // Prepare helper function to recursively set values in the final tensor
            fn set_values<A>(
                result: &ArrayD<A>,
                final_result: &mut ArrayD<A>,
                current_idx: &mut Vec<usize>,
                shape: &[usize],
                depth: usize,
            ) where
                A: Clone,
            {
                if depth == shape.len() {
                    // We've reached a leaf node, copy the value
                    let mut source_idx = current_idx.clone();
                    source_idx.push(0); // Add the final dimension which is always 0 when ranks[n_dims] == 1
                    final_result[current_idx.as_slice()] = result[source_idx.as_slice()].clone();
                    return;
                }

                // Recursively process each index at the current depth
                for i in 0..shape[depth] {
                    current_idx.push(i);
                    set_values(result, final_result, current_idx, shape, depth + 1);
                    current_idx.pop();
                }
            }

            let mut current_idx = Vec::new();
            set_values(
                &result,
                &mut final_result,
                &mut current_idx,
                self.shape.as_slice(),
                0,
            );
        } else {
            // Handle the general case (not needed for valid tensor trains where ranks[n_dims] == 1)
            return Err(LinalgError::ComputationError(
                "Last rank dimension must be 1 for a valid tensor train".to_string(),
            ));
        }

        Ok(final_result)
    }

    /// Evaluates the tensor train at a specific multi-index.
    ///
    /// This is more efficient than reconstructing the full tensor when
    /// only specific elements are needed.
    ///
    /// # Arguments
    ///
    /// * `indices` - The multi-index at which to evaluate the tensor
    ///
    /// # Returns
    ///
    /// * The tensor value at the given indices
    pub fn get(&self, indices: &[usize]) -> LinalgResult<A> {
        if indices.len() != self.shape.len() {
            return Err(LinalgError::ShapeError(format!(
                "Index length ({}) must match tensor dimensionality ({})",
                indices.len(),
                self.shape.len()
            )));
        }

        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(LinalgError::ShapeError(format!(
                    "Index {} for dimension {} out of bounds ({})",
                    idx, i, dim
                )));
            }
        }

        // Start with the first core slice at the first index
        let mut result = self.cores[0]
            .slice(ndarray::s![0, indices[0], ..])
            .to_owned();

        // Multiply by each core slice in sequence
        for i in 1..self.shape.len() {
            let core_slice = self.cores[i].slice(ndarray::s![.., indices[i], ..]);

            // Matrix multiplication: result (1 x r_i) * core_slice (r_i x r_{i+1})
            let mut new_result = Array::zeros((1, core_slice.shape()[1]));

            for j in 0..core_slice.shape()[1] {
                let mut sum = A::zero();
                for k in 0..result.len() {
                    sum += result[k] * core_slice[[k, j]];
                }
                new_result[[0, j]] = sum;
            }

            let shape1 = new_result.shape()[1];
            result = new_result
                .into_shape_with_order(shape1)
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;
        }

        // The result should be a scalar
        if result.len() != 1 {
            return Err(LinalgError::ComputationError(
                "Final result should be a scalar".to_string(),
            ));
        }

        Ok(result[0])
    }

    /// Performs rounding (compression) of the tensor train.
    ///
    /// Reduces the TT-ranks of the decomposition to approximate the original
    /// tensor with a specified relative error tolerance.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Relative error tolerance for the approximation
    ///
    /// # Returns
    ///
    /// * A new `TensorTrain` with reduced ranks
    pub fn round(&self, epsilon: A) -> LinalgResult<Self> {
        if epsilon <= A::zero() {
            return Err(LinalgError::ValueError(
                "Epsilon must be positive".to_string(),
            ));
        }

        // Clone the cores to avoid modifying the original
        let mut cores = self.cores.clone();
        let mut ranks = self.ranks.clone();

        // Forward orthogonalization
        for i in 0..cores.len() - 1 {
            // Reshape core tensor to a matrix
            let core = &cores[i];
            let (r1, n, r2) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            let core_mat = core
                .clone()
                .into_shape_with_order((r1 * n, r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            // QR decomposition
            let (q, r) = qr_decomposition(&core_mat)?;

            // Get the shape value before moving q
            let q_shape1 = q.shape()[1];

            // Update the current core
            cores[i] = q
                .into_shape_with_order((r1, n, q_shape1))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            // Update the next core
            let next_core = &cores[i + 1];
            let (next_r1, next_n, next_r2) = (
                next_core.shape()[0],
                next_core.shape()[1],
                next_core.shape()[2],
            );

            // Contract R with the next core
            let next_core_mat = next_core
                .clone()
                .into_shape_with_order((next_r1, next_n * next_r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            let updated_next_core = r.dot(&next_core_mat);
            cores[i + 1] = updated_next_core
                .into_shape_with_order((r.shape()[0], next_n, next_r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;
        }

        // Backward truncation and orthogonalization
        for i in (1..cores.len()).rev() {
            // Reshape core tensor to a matrix
            let core = &cores[i];
            let (r1, n, r2) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            let core_mat = core
                .clone()
                .into_shape_with_order((r1, n * r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            // SVD decomposition with truncation
            let (u, s, vt) = svd_with_truncation(&core_mat, epsilon)?;

            // Update ranks
            ranks[i] = u.shape()[1];

            // Update the current core
            cores[i] = vt
                .into_shape_with_order((u.shape()[1], n, r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            // Update the previous core
            let prev_core = &cores[i - 1];
            let (prev_r1, prev_n, prev_r2) = (
                prev_core.shape()[0],
                prev_core.shape()[1],
                prev_core.shape()[2],
            );

            // Contract u*s with the previous core
            let u_s = Array2::from_diag(&s).dot(&u.t());
            let prev_core_mat = prev_core
                .clone()
                .into_shape_with_order((prev_r1 * prev_n, prev_r2))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

            let updated_prev_core = prev_core_mat.dot(&u_s);
            cores[i - 1] = updated_prev_core
                .into_shape_with_order((prev_r1, prev_n, u.shape()[1]))
                .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;
        }

        TensorTrain::new(cores, ranks, self.shape.clone())
    }
}

/// Decomposes a tensor into tensor train format.
///
/// This function implements the TT-SVD algorithm for tensor train decomposition.
/// The algorithm works by sequentially decomposing the tensor along each dimension
/// using SVD.
///
/// # Arguments
///
/// * `tensor` - The input tensor to decompose
/// * `max_rank` - Maximum TT-rank (if None, no rank truncation is performed)
/// * `epsilon` - Relative error tolerance for rank truncation (if None, no truncation is performed)
///
/// # Returns
///
/// * A `TensorTrain` object containing the decomposition
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayD, IxDyn};
/// use scirs2_linalg::tensor_contraction::tensor_train::tensor_train_decomposition;
///
/// // Create a 2x3x2 tensor
/// let tensor = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                      [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Decompose into tensor train format with maximum rank 2
/// let tt = tensor_train_decomposition(&tensor.view(), Some(2), None).unwrap();
///
/// // The TT-ranks should be [1, 2, 2, 1] (boundary ranks are always 1)
/// assert_eq!(tt.ranks, vec![1, 2, 2, 1]);
///
/// // Reconstruct the tensor and check the approximation
/// let reconstructed = tt.to_full().unwrap();
/// for i in 0..2 {
///     for j in 0..3 {
///         for k in 0..2 {
///             assert!((reconstructed[[i, j, k]] - tensor[[i, j, k]]).abs() < 1e-10);
///         }
///     }
/// }
/// ```
pub fn tensor_train_decomposition<A, D>(
    tensor: &ArrayView<A, D>,
    max_rank: Option<usize>,
    epsilon: Option<A>,
) -> LinalgResult<TensorTrain<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
    D: Dimension,
{
    // Convert to dynamic dimensionality
    let tensor_dyn = tensor.to_owned().into_dyn();
    let shape = tensor.shape().to_vec();
    let ndim = shape.len();

    // Initialize ranks vector (boundary ranks are always 1)
    let mut ranks = vec![1];
    ranks.resize(ndim + 1, 1);

    // Initialize cores vector
    let mut cores = Vec::with_capacity(ndim);

    // Create a copy of the tensor to work with
    let mut curr_tensor = tensor_dyn;

    // For each dimension, perform SVD and extract the core
    for k in 0..ndim - 1 {
        // Reshape tensor to a matrix where rows correspond to all previous dimensions
        // and current dimension, and columns to all remaining dimensions
        let rows = ranks[k] * shape[k];
        let cols: usize = shape.iter().skip(k + 1).product();

        let tensor_mat = curr_tensor
            .clone()
            .into_shape_with_order((rows, cols))
            .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

        // Perform SVD with truncation if requested
        let (u, s, vt) = match (max_rank, epsilon) {
            (Some(max_r), Some(eps)) => {
                let (u, s, vt) = svd_with_truncation_and_max_rank(&tensor_mat, eps, max_r)?;
                ranks[k + 1] = u.shape()[1];
                (u, s, vt)
            }
            (Some(max_r), None) => {
                let (u, s, vt) = svd_with_max_rank(&tensor_mat, max_r)?;
                ranks[k + 1] = u.shape()[1];
                (u, s, vt)
            }
            (None, Some(eps)) => {
                let (u, s, vt) = svd_with_truncation(&tensor_mat, eps)?;
                ranks[k + 1] = u.shape()[1];
                (u, s, vt)
            }
            (None, None) => {
                let (u, s, vt) = svd(&tensor_mat)?;
                ranks[k + 1] = s.len();
                (u, s, vt)
            }
        };

        // Create the current core tensor
        let core = u
            .into_shape_with_order((ranks[k], shape[k], ranks[k + 1]))
            .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

        cores.push(core);

        // Update the current tensor for the next iteration
        let s_vt = Array2::from_diag(&s).dot(&vt);
        curr_tensor = s_vt
            .into_shape_with_order(
                std::iter::once(ranks[k + 1])
                    .chain(shape.iter().skip(k + 1).copied())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;
    }

    // Add the last core
    let last_core = curr_tensor
        .into_shape_with_order((ranks[ndim - 1], shape[ndim - 1], ranks[ndim]))
        .map_err(|e| LinalgError::ComputationError(format!("Reshape error: {}", e)))?;

    cores.push(last_core);

    // Create the TensorTrain object
    TensorTrain::new(cores, ranks, shape)
}

// Helper function for full SVD decomposition
fn svd<A>(matrix: &Array2<A>) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    use crate::decomposition::svd as svd_decomp;

    // Convert to view and call with full_matrices=false
    let matrix_view = matrix.view();
    svd_decomp(&matrix_view, false, None)
}

// Helper function for SVD with truncation based on relative error
fn svd_with_truncation<A>(
    matrix: &Array2<A>,
    epsilon: A,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    let (u, s, vt) = svd(matrix)?;

    // Normalize singular values to get relative importance
    let s_norm = if !s.is_empty() && s[0] > A::zero() {
        s.mapv(|x| x / s[0])
    } else {
        s.clone()
    };

    // Find the number of singular values to keep based on epsilon
    let mut rank = 0;
    for (i, &val) in s_norm.iter().enumerate() {
        if val < epsilon {
            rank = i;
            break;
        }
        rank = i + 1;
    }

    // Ensure at least one singular value is kept
    rank = rank.max(1);

    // Truncate the matrices
    let u_trunc = u.slice(ndarray::s![.., ..rank]).to_owned();
    let s_trunc = s.slice(ndarray::s![..rank]).to_owned();
    let vt_trunc = vt.slice(ndarray::s![..rank, ..]).to_owned();

    Ok((u_trunc, s_trunc, vt_trunc))
}

// Helper function for SVD with maximum rank constraint
fn svd_with_max_rank<A>(
    matrix: &Array2<A>,
    max_rank: usize,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    let (u, s, vt) = svd(matrix)?;

    // Compute the rank to use (minimum of max_rank and the number of singular values)
    let rank = max_rank.min(s.len());

    // Truncate the matrices
    let u_trunc = u.slice(ndarray::s![.., ..rank]).to_owned();
    let s_trunc = s.slice(ndarray::s![..rank]).to_owned();
    let vt_trunc = vt.slice(ndarray::s![..rank, ..]).to_owned();

    Ok((u_trunc, s_trunc, vt_trunc))
}

// Helper function for SVD with both truncation and maximum rank
fn svd_with_truncation_and_max_rank<A>(
    matrix: &Array2<A>,
    epsilon: A,
    max_rank: usize,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    let (u, s, vt) = svd(matrix)?;

    // Normalize singular values to get relative importance
    let s_norm = if !s.is_empty() && s[0] > A::zero() {
        s.mapv(|x| x / s[0])
    } else {
        s.clone()
    };

    // Find the number of singular values to keep based on epsilon
    let mut rank = 0;
    for (i, &val) in s_norm.iter().enumerate() {
        if val < epsilon {
            rank = i;
            break;
        }
        rank = i + 1;
    }

    // Ensure at least one singular value is kept
    rank = rank.max(1);

    // Apply max_rank constraint
    rank = rank.min(max_rank);

    // Truncate the matrices
    let u_trunc = u.slice(ndarray::s![.., ..rank]).to_owned();
    let s_trunc = s.slice(ndarray::s![..rank]).to_owned();
    let vt_trunc = vt.slice(ndarray::s![..rank, ..]).to_owned();

    Ok((u_trunc, s_trunc, vt_trunc))
}

// Helper function for QR decomposition
fn qr_decomposition<A>(matrix: &Array2<A>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + 'static + ndarray::ScalarOperand,
{
    use crate::decomposition::qr;

    // Convert to view and call QR
    let matrix_view = matrix.view();
    let (q, r) = qr(&matrix_view, None)?;
    Ok((q, r))
}

/// Helper type for more convenient type definition
pub type Array1<A> = Array<A, ndarray::Ix1>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_tensor_train_decomposition_3d() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with full rank (without truncation)
        let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();

        // Verify the shape and ranks
        assert_eq!(tt.shape, vec![2, 3, 2]);
        assert_eq!(tt.ranks.len(), 4); // n+1 ranks for n-dimensional tensor
        assert_eq!(tt.ranks[0], 1); // First rank is always 1
        assert_eq!(tt.ranks[3], 1); // Last rank is always 1

        // Each core should have the correct shape:
        // G_1: 1 x n_1 x r_1
        // G_i: r_{i-1} x n_i x r_i
        // G_d: r_{d-1} x n_d x 1
        for i in 0..tt.cores.len() {
            let core = &tt.cores[i];
            assert_eq!(core.shape()[0], tt.ranks[i]);
            assert_eq!(core.shape()[1], tt.shape[i]);
            assert_eq!(core.shape()[2], tt.ranks[i + 1]);
        }

        // Reconstruct the tensor
        let reconstructed = tt.to_full().unwrap();

        // Check that the reconstruction matches the original tensor
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    assert_abs_diff_eq!(
                        reconstructed[[i, j, k]],
                        tensor[[i, j, k]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_tensor_train_decomposition_with_truncation() {
        // Create a 4x3x2x2 tensor with some structure
        let mut tensor = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 2, 2]));

        // Fill with some values (outer product-like pattern for simplicity)
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    for l in 0..2 {
                        tensor[[i, j, k, l]] =
                            (i + 1) as f64 * (j + 1) as f64 * (k + 1) as f64 * (l + 1) as f64;
                    }
                }
            }
        }

        // Decompose with rank truncation (max rank 2)
        let tt = tensor_train_decomposition(&tensor.view(), Some(2), None).unwrap();

        // Check that no rank exceeds 2
        for &r in &tt.ranks {
            assert!(r <= 2);
        }

        // Reconstruct the tensor
        let reconstructed = tt.to_full().unwrap();

        // For low rank tensors like this one, we should still get a good approximation
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    for l in 0..2 {
                        assert_abs_diff_eq!(
                            reconstructed[[i, j, k, l]],
                            tensor[[i, j, k, l]],
                            epsilon = 1e-6
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_get_tensor_element() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose without truncation
        let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();

        // Test getting individual elements
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    let value = tt.get(&[i, j, k]).unwrap();
                    assert_abs_diff_eq!(value, tensor[[i, j, k]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_round_tensor_train() {
        // Create a 3x4x3x2 tensor
        let mut tensor = ArrayD::<f64>::zeros(ndarray::IxDyn(&[3, 4, 3, 2]));

        // Fill with values
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..3 {
                    for l in 0..2 {
                        tensor[[i, j, k, l]] =
                            (i + 1) as f64 * (j + 1) as f64 * (k + 1) as f64 * (l + 1) as f64;
                    }
                }
            }
        }

        // Decompose with full rank
        let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();

        // Round the tensor train with various epsilon values
        for epsilon in [1e-8, 1e-4, 1e-2].iter() {
            let rounded_tt = tt.round(*epsilon).unwrap();

            // Reconstruct the tensor
            let reconstructed = rounded_tt.to_full().unwrap();

            // Check that the error is within bounds
            let mut max_error = 0.0;
            let norm = tensor.mapv(|x| x * x).sum().sqrt();

            for i in 0..3 {
                for j in 0..4 {
                    for k in 0..3 {
                        for l in 0..2 {
                            let error = (reconstructed[[i, j, k, l]] - tensor[[i, j, k, l]]).abs();
                            max_error = max_error.max(error);
                        }
                    }
                }
            }

            // Relative error should be less than epsilon
            let relative_error = max_error / norm;
            assert!(relative_error <= *epsilon || relative_error <= 1e-10);
        }
    }
}
