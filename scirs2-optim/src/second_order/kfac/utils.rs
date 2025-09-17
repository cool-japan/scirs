//! Utility functions for K-FAC optimization
//!
//! This module contains helper functions and utilities used throughout
//! the K-FAC implementation, including layer-specific operations and
//! mathematical utilities.

use crate::error::Result;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// K-FAC utilities for layer-specific operations
pub struct KFACUtils;

impl KFACUtils {
    /// Compute K-FAC update for convolutional layers
    pub fn conv_kfac_update<T: Float + 'static>(
        input_patches: &Array2<T>,
        output_gradients: &Array2<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Array2<T>> {
        // Simplified convolution K-FAC update
        // In practice, this would involve more complex patch extraction and reshaping
        let batch_size = input_patches.nrows();
        let input_dim = input_patches.ncols();
        let output_dim = output_gradients.ncols();

        // Create a placeholder update matrix
        let mut update = Array2::zeros((kernel_size.0 * kernel_size.1, output_dim));

        // Simple averaging across batch
        if batch_size > 0 {
            let scale = T::one() / T::from(batch_size).unwrap();
            for i in 0..update.nrows() {
                for j in 0..update.ncols() {
                    let input_idx = i % input_dim;
                    let output_idx = j % output_dim;

                    let mut sum = T::zero();
                    for b in 0..batch_size {
                        if input_idx < input_dim && output_idx < output_dim {
                            sum = sum + input_patches[[b, input_idx]] * output_gradients[[b, output_idx]];
                        }
                    }
                    update[[i, j]] = sum * scale;
                }
            }
        }

        Ok(update)
    }

    /// Compute batch normalization statistics for K-FAC
    pub fn batchnorm_statistics<T: Float + num_traits::FromPrimitive>(
        input: &Array2<T>,
        eps: T,
    ) -> Result<(Array1<T>, Array1<T>)> {
        let batch_size = input.nrows();
        let num_features = input.ncols();

        if batch_size == 0 {
            return Ok((Array1::zeros(num_features), Array1::ones(num_features)));
        }

        let batch_size_t = T::from(batch_size).unwrap();

        // Compute mean
        let mean = input.mean_axis(ndarray::Axis(0)).unwrap();

        // Compute variance
        let mut var = Array1::zeros(num_features);
        for i in 0..num_features {
            let mut sum_sq_diff = T::zero();
            for j in 0..batch_size {
                let diff = input[[j, i]] - mean[i];
                sum_sq_diff = sum_sq_diff + diff * diff;
            }
            var[i] = sum_sq_diff / batch_size_t + eps;
        }

        Ok((mean, var))
    }

    /// Compute K-FAC update for grouped convolution layers
    pub fn grouped_conv_kfac<T: Float + ndarray::ScalarOperand>(
        input: &Array2<T>,
        gradients: &Array2<T>,
        num_groups: usize,
    ) -> Result<Array2<T>> {
        let batch_size = input.nrows();
        let input_channels = input.ncols();
        let output_channels = gradients.ncols();

        if num_groups == 0 {
            return Err(crate::error::OptimError::InvalidParameter(
                "Number of groups must be positive".to_string(),
            ));
        }

        let input_per_group = input_channels / num_groups;
        let output_per_group = output_channels / num_groups;

        let mut result = Array2::zeros((input_channels, output_channels));

        // Process each group separately
        for group in 0..num_groups {
            let input_start = group * input_per_group;
            let input_end = input_start + input_per_group;
            let output_start = group * output_per_group;
            let output_end = output_start + output_per_group;

            // Extract group data
            let group_input = input.slice(ndarray::s![.., input_start..input_end]);
            let group_gradients = gradients.slice(ndarray::s![.., output_start..output_end]);

            // Compute group covariance
            let group_update = group_input.t().dot(&group_gradients);

            // Place back in result
            result
                .slice_mut(ndarray::s![input_start..input_end, output_start..output_end])
                .assign(&group_update);
        }

        // Normalize by batch size
        if batch_size > 0 {
            let scale = T::one() / T::from(batch_size).unwrap();
            result = result * scale;
        }

        Ok(result)
    }

    /// Compute eigenvalue-based regularization
    pub fn eigenvalue_regularization<T: Float>(
        matrix: &Array2<T>,
        min_eigenvalue: T,
    ) -> Array2<T> {
        let n = matrix.nrows();
        let mut regularized = matrix.clone();

        // Simple diagonal regularization (in practice, would use proper eigendecomposition)
        for i in 0..n {
            if regularized[[i, i]] < min_eigenvalue {
                regularized[[i, i]] = min_eigenvalue;
            }
        }

        regularized
    }

    /// Compute Kronecker product approximation for two matrices
    pub fn kronecker_product_approx<T: Float>(
        a: &Array2<T>,
        b: &Array2<T>,
    ) -> Array2<T> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));

        for i in 0..a_rows {
            for j in 0..a_cols {
                let a_val = a[[i, j]];
                for k in 0..b_rows {
                    for l in 0..b_cols {
                        result[[i * b_rows + k, j * b_cols + l]] = a_val * b[[k, l]];
                    }
                }
            }
        }

        result
    }

    /// Compute trace of a matrix
    pub fn trace<T: Float>(matrix: &Array2<T>) -> T {
        let n = matrix.nrows().min(matrix.ncols());
        let mut trace = T::zero();

        for i in 0..n {
            trace = trace + matrix[[i, i]];
        }

        trace
    }

    /// Compute Frobenius norm of a matrix
    pub fn frobenius_norm<T: Float + std::iter::Sum>(matrix: &Array2<T>) -> T {
        matrix.iter().map(|&x| x * x).sum::<T>().sqrt()
    }

    /// Check if two matrices are approximately equal
    pub fn matrices_approx_equal<T: Float>(
        a: &Array2<T>,
        b: &Array2<T>,
        tolerance: T,
    ) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            if (*a_val - *b_val).abs() > tolerance {
                return false;
            }
        }

        true
    }

    /// Compute running average with exponential decay
    pub fn exponential_moving_average<T: Float>(
        current_value: T,
        new_value: T,
        decay: T,
    ) -> T {
        decay * current_value + (T::one() - decay) * new_value
    }

    /// Clamp eigenvalues to prevent numerical instability
    pub fn clamp_eigenvalues<T: Float>(
        eigenvalues: &mut Array1<T>,
        min_val: T,
        max_val: T,
    ) {
        for eigenval in eigenvalues.iter_mut() {
            *eigenval = (*eigenval).max(min_val).min(max_val);
        }
    }

    /// Compute condition number using singular values (approximation)
    pub fn condition_number_svd_approx<T: Float>(matrix: &Array2<T>) -> T {
        // Simple approximation using diagonal elements
        let diag = matrix.diag();
        let max_diag = diag.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x.abs()));
        let min_diag = diag.iter().fold(T::infinity(), |acc, &x| acc.min(x.abs()));

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }

    /// Extract diagonal elements and create diagonal matrix
    pub fn diag_matrix<T: Float + Clone>(diagonal: &Array1<T>) -> Array2<T> {
        let n = diagonal.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            matrix[[i, i]] = diagonal[i].clone();
        }

        matrix
    }

    /// Symmetrize a matrix: (A + A^T) / 2
    pub fn symmetrize<T: Float>(matrix: &Array2<T>) -> Array2<T> {
        let n = matrix.nrows();
        let mut result = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) / T::from(2.0).unwrap();
            }
        }

        result
    }
}

/// Ordered float wrapper for comparison operations
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat<T: Float>(pub T);

impl<T: Float> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}

impl<T: Float> Eq for OrderedFloat<T> {}

impl<T: Float> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Float> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_computation() {
        let matrix = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let trace = KFACUtils::trace(&matrix);
        assert!((trace - 15.0).abs() < 1e-10); // 1 + 5 + 9 = 15
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = KFACUtils::frobenius_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_exponential_moving_average() {
        let current = 10.0;
        let new_val = 20.0;
        let decay = 0.9;

        let result = KFACUtils::exponential_moving_average(current, new_val, decay);
        let expected = 0.9 * 10.0 + 0.1 * 20.0; // 9.0 + 2.0 = 11.0
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matrices_approx_equal() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.001, 2.001, 3.001, 4.001]).unwrap();

        assert!(KFACUtils::matrices_approx_equal(&a, &b, 0.01));
        assert!(!KFACUtils::matrices_approx_equal(&a, &b, 0.0001));
    }

    #[test]
    fn test_symmetrize() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let symmetric = KFACUtils::symmetrize(&matrix);

        assert!((symmetric[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((symmetric[[0, 1]] - 2.5).abs() < 1e-10); // (2 + 3) / 2
        assert!((symmetric[[1, 0]] - 2.5).abs() < 1e-10); // (3 + 2) / 2
        assert!((symmetric[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diag_matrix() {
        let diagonal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let matrix = KFACUtils::diag_matrix(&diagonal);

        assert_eq!(matrix.dim(), (3, 3));
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((matrix[[1, 1]] - 2.0).abs() < 1e-10);
        assert!((matrix[[2, 2]] - 3.0).abs() < 1e-10);
        assert!((matrix[[0, 1]]).abs() < 1e-10); // Off-diagonal should be zero
    }

    #[test]
    fn test_ordered_float() {
        let a = OrderedFloat(1.5);
        let b = OrderedFloat(2.5);
        let c = OrderedFloat(1.5);

        assert!(a < b);
        assert!(a == c);
        assert!(b > a);
    }

    #[test]
    fn test_batchnorm_statistics() {
        let input = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]).unwrap();

        let (mean, var) = KFACUtils::batchnorm_statistics(&input, 1e-8).unwrap();

        // Expected mean: [4.0, 5.0] (column-wise average)
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 5.0).abs() < 1e-6);

        // Variance should be positive
        assert!(var[0] > 0.0);
        assert!(var[1] > 0.0);
    }
}