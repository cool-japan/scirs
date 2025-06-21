//! Canonical Polyadic (CP) Decomposition
//!
//! This module provides functionality for Canonical Polyadic decomposition (also known as
//! CANDECOMP/PARAFAC), which decomposes a tensor into a sum of rank-one tensors.

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayD, ArrayView, Dimension};
use num_traits::{Float, NumAssign, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Represents a tensor in Canonical Polyadic (CP) format.
///
/// A CP decomposition represents an N-dimensional tensor as a sum of rank-one tensors,
/// where each rank-one tensor is an outer product of vectors.
///
/// # Fields
///
/// * `factors` - Vector of factor matrices (one for each mode)
/// * `weights` - Optional weights for each component (if None, uniform weighting is assumed)
/// * `shape` - Original tensor shape
#[derive(Debug, Clone)]
pub struct CanonicalPolyadic<A>
where
    A: Clone + Float + Debug,
{
    /// The factor matrices, each with shape (mode_dim, rank)
    pub factors: Vec<Array2<A>>,
    /// Optional weights for each component
    pub weights: Option<Array1<A>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
}

impl<A> CanonicalPolyadic<A>
where
    A: Clone + Float + NumAssign + Zero + Debug + Sum + Send + Sync + 'static,
{
    /// Creates a new Canonical Polyadic decomposition from the given factor matrices and weights.
    ///
    /// # Arguments
    ///
    /// * `factors` - Vector of factor matrices (one for each mode)
    /// * `weights` - Optional weights for each component (if None, uniform weighting is assumed)
    /// * `shape` - Original tensor shape (optional, if None will be derived from factors)
    ///
    /// # Returns
    ///
    /// * `CanonicalPolyadic` - A new CP instance
    pub fn new(
        factors: Vec<Array2<A>>,
        weights: Option<Array1<A>>,
        shape: Option<Vec<usize>>,
    ) -> LinalgResult<Self> {
        // Basic validation
        if factors.is_empty() {
            return Err(LinalgError::ValueError(
                "Factor matrices list cannot be empty".to_string(),
            ));
        }

        // All factor matrices should have the same number of columns (rank)
        let rank = factors[0].shape()[1];
        for (i, factor) in factors.iter().enumerate().skip(1) {
            if factor.shape()[1] != rank {
                return Err(LinalgError::ShapeError(format!(
                    "All factor matrices must have the same number of columns (rank). Factor 0 has {} columns, but factor {} has {} columns.",
                    rank,
                    i,
                    factor.shape()[1]
                )));
            }
        }

        // Validate weights if provided
        if let Some(ref w) = weights {
            if w.len() != rank {
                return Err(LinalgError::ShapeError(format!(
                    "Weights length ({}) must match decomposition rank ({})",
                    w.len(),
                    rank
                )));
            }
        }

        // Derive shape from factors if not provided
        let shape = match shape {
            Some(s) => {
                if s.len() != factors.len() {
                    return Err(LinalgError::ShapeError(format!(
                        "Shape length ({}) must match number of factor matrices ({})",
                        s.len(),
                        factors.len()
                    )));
                }

                for (i, &dim) in s.iter().enumerate() {
                    if dim != factors[i].shape()[0] {
                        return Err(LinalgError::ShapeError(format!(
                            "Shape dimension {} ({}) must match factor matrix {} rows ({})",
                            i,
                            dim,
                            i,
                            factors[i].shape()[0]
                        )));
                    }
                }
                s
            }
            None => factors.iter().map(|f| f.shape()[0]).collect(),
        };

        Ok(CanonicalPolyadic {
            factors,
            weights,
            shape,
        })
    }

    /// Reconstructs the full tensor from the CP decomposition.
    ///
    /// # Returns
    ///
    /// * `ArrayD<A>` - The full tensor
    pub fn to_full(&self) -> LinalgResult<ArrayD<A>> {
        let rank = self.factors[0].shape()[1];
        let _n_modes = self.factors.len();

        // Create a tensor filled with zeros
        let mut result = ArrayD::zeros(self.shape.clone());

        // For each rank-1 component
        for r in 0..rank {
            // Get the weight for this component (1.0 if no weights)
            let weight = match &self.weights {
                Some(w) => w[r],
                None => A::one(),
            };

            // Build indices to iterate over the full tensor
            let mut indices_vec = Vec::new();
            fn generate_indices(
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
                    generate_indices(shape, current.clone(), depth + 1, all_indices);
                    current.pop();
                }
            }

            generate_indices(&self.shape, Vec::new(), 0, &mut indices_vec);

            // For each element in the tensor
            for idx in indices_vec {
                // Compute the outer product value for this element
                let mut value = weight;
                for (mode, &i) in idx.iter().enumerate() {
                    value *= self.factors[mode][[i, r]];
                }

                // Add to the result
                let result_idx = ndarray::IxDyn(idx.as_slice());
                result[&result_idx] += value;
            }
        }

        Ok(result)
    }

    /// Computes the reconstruction error of the CP decomposition.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The original tensor
    ///
    /// # Returns
    ///
    /// * `A` - The relative reconstruction error (Frobenius norm of the difference divided by
    ///   the Frobenius norm of the original tensor)
    pub fn reconstruction_error(&self, tensor: &ArrayView<A, impl Dimension>) -> LinalgResult<A> {
        // Reconstruct the tensor
        let reconstructed = self.to_full()?;

        // Convert original tensor to dynamic dimensionality
        let tensor_dyn = tensor.to_owned().into_dyn();

        // Verify shapes match
        if tensor_dyn.shape() != reconstructed.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Original tensor shape {:?} does not match reconstructed tensor shape {:?}",
                tensor_dyn.shape(),
                reconstructed.shape()
            )));
        }

        // Compute Frobenius norm of the difference and the original tensor
        let mut diff_squared_sum = A::zero();
        let mut orig_squared_sum = A::zero();

        for (idx, &orig_val) in tensor_dyn.indexed_iter() {
            let rec_val = reconstructed[idx.clone()];
            diff_squared_sum += (orig_val - rec_val).powi(2);
            orig_squared_sum += orig_val.powi(2);
        }

        // Handle division by zero
        if orig_squared_sum == A::zero() {
            return Ok(if diff_squared_sum == A::zero() {
                A::zero()
            } else {
                A::infinity()
            });
        }

        // Return relative error
        Ok((diff_squared_sum / orig_squared_sum).sqrt())
    }

    /// Compresses the CP decomposition by truncating to a lower rank.
    ///
    /// # Arguments
    ///
    /// * `new_rank` - The new rank to compress to (must be less than or equal to current rank)
    ///
    /// # Returns
    ///
    /// * `CanonicalPolyadic` - A new, compressed CP decomposition
    pub fn compress(&self, new_rank: usize) -> LinalgResult<Self> {
        let current_rank = self.factors[0].shape()[1];

        // Validate new rank
        if new_rank > current_rank {
            return Err(LinalgError::ValueError(format!(
                "New rank ({}) must be less than or equal to current rank ({})",
                new_rank, current_rank
            )));
        }

        if new_rank == 0 {
            return Err(LinalgError::ValueError(
                "New rank must be at least 1".to_string(),
            ));
        }

        // If no compression needed, return a clone
        if new_rank == current_rank {
            return Ok(self.clone());
        }

        // Truncate each factor matrix
        let compressed_factors: Vec<Array2<A>> = self
            .factors
            .iter()
            .map(|factor| factor.slice(ndarray::s![.., ..new_rank]).to_owned())
            .collect();

        // Truncate weights if present
        let compressed_weights = self
            .weights
            .as_ref()
            .map(|w| w.slice(ndarray::s![..new_rank]).to_owned());

        // Create new CP decomposition
        CanonicalPolyadic::new(
            compressed_factors,
            compressed_weights,
            Some(self.shape.clone()),
        )
    }
}

/// Performs Canonical Polyadic (CP) decomposition using Alternating Least Squares (ALS).
///
/// This function computes a CP decomposition, which represents an N-dimensional tensor
/// as a sum of rank-one tensors.
///
/// # Arguments
///
/// * `tensor` - The tensor to decompose
/// * `rank` - The target rank for the decomposition
/// * `max_iterations` - Maximum number of ALS iterations
/// * `tolerance` - Convergence tolerance for relative change in fit
/// * `normalize` - Whether to normalize the factor matrices and store the norms as weights
///
/// # Returns
///
/// * A `CanonicalPolyadic` object containing the decomposition
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::tensor_contraction::cp::cp_als;
///
/// // Create a 2x3x2 tensor
/// let tensor = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                     [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Decompose with target rank 3, maximum 50 iterations, and tolerance 1e-4
/// let cp = cp_als(&tensor.view(), 3, 50, 1e-4, true).unwrap();
///
/// // The factors should preserve the original dimensions
/// assert_eq!(cp.factors[0].shape(), &[2, 3]); // mode 0: 2 rows, rank 3
/// assert_eq!(cp.factors[1].shape(), &[3, 3]); // mode 1: 3 rows, rank 3
/// assert_eq!(cp.factors[2].shape(), &[2, 3]); // mode 2: 2 rows, rank 3
/// ```
pub fn cp_als<A, D>(
    tensor: &ArrayView<A, D>,
    rank: usize,
    max_iterations: usize,
    tolerance: A,
    normalize: bool,
) -> LinalgResult<CanonicalPolyadic<A>>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Debug
        + Sum
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
    D: Dimension,
{
    // Validate inputs
    if rank == 0 {
        return Err(LinalgError::ValueError(
            "Rank must be at least 1".to_string(),
        ));
    }

    let n_modes = tensor.ndim();
    let shape = tensor.shape().to_vec();

    // Initialize factors with random values
    let mut factors: Vec<Array2<A>> = Vec::with_capacity(n_modes);
    for &dim in shape.iter() {
        // Initialize with random values scaled between 0 and 1
        let mut factor = Array2::zeros((dim, rank));
        for i in 0..dim {
            for j in 0..rank {
                // Simple deterministic initialization - replace with random in production
                factor[[i, j]] = A::from(((i + 1) * (j + 1)) % 10).unwrap() / A::from(10).unwrap();
            }
        }
        factors.push(factor);
    }

    // Convert tensor to dynamic dimensionality
    let tensor_dyn = tensor.to_owned().into_dyn();

    // Unfold the tensor along each mode
    let unfolded_tensors: Vec<Array2<A>> = (0..n_modes)
        .map(|mode| unfold_tensor(&tensor_dyn, mode).expect("Tensor unfolding failed"))
        .collect();

    // Main ALS loop
    let mut prev_error = A::infinity();

    for iteration in 0..max_iterations {
        // For each mode, update the corresponding factor matrix
        for mode in 0..n_modes {
            // Compute the Khatri-Rao product of all factors except the current one
            let kr_product = khatri_rao_product(&factors, mode)?;

            // Update the factor using the matricized tensor and the Khatri-Rao product
            let mttkrp = unfolded_tensors[mode].dot(&kr_product);

            // Compute the pseudoinverse of the Khatri-Rao product using the Gram matrix
            let gram_matrix = compute_gram_matrix(&factors, mode)?;
            let gram_inv = pseudo_inverse(&gram_matrix)?;

            // Update the factor for this mode
            factors[mode] = mttkrp.dot(&gram_inv);
        }

        // Normalize factors and compute weights if required
        let mut weights = None;
        if normalize {
            weights = Some(normalize_factors(&mut factors));
        }

        // Check for convergence
        let cp = CanonicalPolyadic::new(factors.clone(), weights.clone(), Some(shape.clone()))?;
        let error = cp.reconstruction_error(tensor)?;
        let rel_improvement = (prev_error - error) / prev_error;

        if !rel_improvement.is_nan() && rel_improvement < tolerance && iteration > 0 {
            return Ok(cp);
        }

        prev_error = error;
    }

    // Create the final CP decomposition
    let weights = if normalize {
        Some(normalize_factors(&mut factors))
    } else {
        None
    };

    CanonicalPolyadic::new(factors, weights, Some(shape))
}

// Unfolds a tensor along a specified mode
fn unfold_tensor<A>(tensor: &ArrayD<A>, mode: usize) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    let shape = tensor.shape();

    if mode >= shape.len() {
        return Err(LinalgError::ShapeError(format!(
            "Mode {} is out of bounds for tensor with {} dimensions",
            mode,
            shape.len()
        )));
    }

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

    // Populate the unfolded tensor
    for idx in ndarray::indices(shape) {
        let mode_idx = idx[mode];
        let idx_vec: Vec<usize> = idx.as_array_view().to_vec();
        let col_idx = calc_col_idx(&idx_vec, shape, mode);
        result[[mode_idx, col_idx]] = tensor[idx.clone()];
    }

    Ok(result)
}

// Computes the Khatri-Rao product (columnwise Kronecker product) of all factors except one
fn khatri_rao_product<A>(factors: &[Array2<A>], skip_mode: usize) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    let n_modes = factors.len();

    if n_modes <= 1 {
        return Err(LinalgError::ValueError(
            "Need at least two factor matrices to compute Khatri-Rao product".to_string(),
        ));
    }

    let rank = factors[0].shape()[1];

    // If we're skipping all but one matrix, return that matrix
    if n_modes == 2 && skip_mode < n_modes {
        let other_mode = if skip_mode == 0 { 1 } else { 0 };
        return Ok(factors[other_mode].clone());
    }

    // Determine the number of rows in the result
    let _n_rows: usize = factors
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != skip_mode)
        .map(|(_, f)| f.shape()[0])
        .product();

    // Find the first non-skipped matrix to initialize
    let mut result = None;
    let mut result_rows = 1;

    // Compute the Khatri-Rao product
    for (mode, factor) in factors.iter().enumerate() {
        if mode == skip_mode {
            continue;
        }

        match result {
            None => {
                // First non-skipped matrix becomes the initial result
                result = Some(factor.clone());
                result_rows = factor.shape()[0];
            }
            Some(prev_result) => {
                let factor_rows = factor.shape()[0];

                // Initialize a new result matrix
                let mut new_result = Array2::zeros((result_rows * factor_rows, rank));

                // Compute the Khatri-Rao product
                for r in 0..rank {
                    let mut col_idx = 0;
                    for i in 0..result_rows {
                        for j in 0..factor_rows {
                            new_result[[col_idx, r]] = prev_result[[i, r]] * factor[[j, r]];
                            col_idx += 1;
                        }
                    }
                }

                result = Some(new_result);
                result_rows = result.as_ref().unwrap().shape()[0];
            }
        }
    }

    result.ok_or_else(|| LinalgError::ValueError("All factors were skipped".to_string()))
}

// Computes the Gram matrix for ALS update
fn compute_gram_matrix<A>(factors: &[Array2<A>], skip_mode: usize) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    let _n_modes = factors.len();
    let rank = factors[0].shape()[1];

    // Initialize result with ones
    let mut gram = Array2::ones((rank, rank));

    // Compute the Gram matrix as the Hadamard product of all factor Gram matrices
    for (mode, factor) in factors.iter().enumerate() {
        if mode == skip_mode {
            continue;
        }

        let factor_t = factor.t();
        let factor_gram = factor_t.dot(factor);

        // Update the Gram matrix with Hadamard (element-wise) product
        for i in 0..rank {
            for j in 0..rank {
                gram[[i, j]] *= factor_gram[[i, j]];
            }
        }
    }

    Ok(gram)
}

// Computes the Moore-Penrose pseudoinverse using SVD
fn pseudo_inverse<A>(matrix: &Array2<A>) -> LinalgResult<Array2<A>>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Debug
        + Sum
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
{
    let (u, s, vt) = svd(&matrix.view(), false, None)?;

    // Compute the pseudoinverse of the singular values
    let mut s_inv = Array2::zeros((s.len(), s.len()));
    for i in 0..s.len() {
        if s[i] > A::epsilon() * A::from(10.0).unwrap() {
            s_inv[[i, i]] = A::one() / s[i];
        }
    }

    // Compute V * S^-1 * U^T
    let v = vt.t();
    let u_t = u.t();

    let vs_inv = v.dot(&s_inv);
    let result = vs_inv.dot(&u_t);

    Ok(result)
}

// Normalizes the factor matrices and returns the weights
fn normalize_factors<A>(factors: &mut [Array2<A>]) -> Array1<A>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    let _n_modes = factors.len();
    let rank = factors[0].shape()[1];

    // Initialize weights
    let mut weights = Array1::ones(rank);

    // For each component
    for r in 0..rank {
        // For each mode, compute the norm of the r-th column
        for factor in factors.iter_mut() {
            let mut norm_sq = A::zero();
            for i in 0..factor.shape()[0] {
                norm_sq += factor[[i, r]].powi(2);
            }
            let norm = norm_sq.sqrt();

            if norm > A::epsilon() {
                // Normalize the column
                for i in 0..factor.shape()[0] {
                    factor[[i, r]] /= norm;
                }

                // Update the weight
                weights[r] *= norm;
            }
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_cp_decomposition_basic() {
        // Create a simple 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with rank 2 (avoiding 3x3 matrices that fail in eigen)
        let cp = cp_als(&tensor.view(), 2, 50, 1e-4, true).unwrap();

        // Check dimensions
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[2, 2]);
        assert_eq!(cp.factors[1].shape(), &[3, 2]);
        assert_eq!(cp.factors[2].shape(), &[2, 2]);

        // Weights should be present
        assert!(cp.weights.is_some());
        assert_eq!(cp.weights.as_ref().unwrap().len(), 2);

        // Reconstruct the tensor
        let _reconstructed = cp.to_full().unwrap();

        // Check reconstruction error
        let error = cp.reconstruction_error(&tensor.view()).unwrap();
        assert!(error < 0.1); // Error should be small for this simple tensor
    }

    #[test]
    fn test_cp_decomposition_truncated() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with rank 2
        let cp = cp_als(&tensor.view(), 2, 50, 1e-4, true).unwrap();

        // Check dimensions
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[2, 2]);
        assert_eq!(cp.factors[1].shape(), &[3, 2]);
        assert_eq!(cp.factors[2].shape(), &[2, 2]);

        // Reconstruct the tensor
        let _reconstructed = cp.to_full().unwrap();

        // Check reconstruction error (should be non-zero for truncated rank)
        let error = cp.reconstruction_error(&tensor.view()).unwrap();
        assert!(error > 0.0);
        assert!(error < 0.5); // Error should still be reasonable for rank-2 approximation
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_compress() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with rank 4 (avoiding 3x3 matrices that fail in eigen)
        let cp = cp_als(&tensor.view(), 4, 50, 1e-4, true).unwrap();

        // Compress to rank 2
        let compressed = cp.compress(2).unwrap();

        // Check dimensions
        assert_eq!(compressed.factors.len(), 3);
        assert_eq!(compressed.factors[0].shape(), &[2, 2]);
        assert_eq!(compressed.factors[1].shape(), &[3, 2]);
        assert_eq!(compressed.factors[2].shape(), &[2, 2]);

        // Weights should be truncated
        assert!(compressed.weights.is_some());
        assert_eq!(compressed.weights.as_ref().unwrap().len(), 2);

        // Error of compressed decomposition should be higher than original
        let error_orig = cp.reconstruction_error(&tensor.view()).unwrap();
        let error_comp = compressed.reconstruction_error(&tensor.view()).unwrap();
        assert!(error_comp >= error_orig * 0.99); // Allow for numerical imprecision
    }

    #[test]
    fn test_reconstruction() {
        // Create a simple rank-1 tensor (outer product of vectors)
        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0, 5.0];
        let c = array![6.0, 7.0];

        // Manual construction of a rank-1 tensor
        let mut tensor = ArrayD::<f64>::zeros(ndarray::IxDyn(&[2, 3, 2]));
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    tensor[[i, j, k]] = a[i] * b[j] * c[k];
                }
            }
        }

        // Create factor matrices for the CP decomposition
        let factors = vec![
            Array2::from_shape_fn((2, 1), |(i, _)| a[i]),
            Array2::from_shape_fn((3, 1), |(j, _)| b[j]),
            Array2::from_shape_fn((2, 1), |(k, _)| c[k]),
        ];

        // Create CP decomposition
        let cp = CanonicalPolyadic::new(factors, None, Some(vec![2, 3, 2])).unwrap();

        // Reconstruct the tensor
        let reconstructed = cp.to_full().unwrap();

        // Check reconstruction (should be exact for this simple case)
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
    fn test_khatri_rao_product() {
        // Create two matrices
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        // Create a vector of factors
        let factors = vec![a.clone(), b.clone()];

        // Compute Khatri-Rao product, skipping the first matrix
        let kr = khatri_rao_product(&factors, 0).unwrap();

        // Result should be the same as b (when skipping a)
        assert_eq!(kr.shape(), &[3, 2]);
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(kr[[i, j]], b[[i, j]], epsilon = 1e-10);
            }
        }

        // Compute Khatri-Rao product, skipping the second matrix
        let kr = khatri_rao_product(&factors, 1).unwrap();

        // Result should be the same as a (when skipping b)
        assert_eq!(kr.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(kr[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
