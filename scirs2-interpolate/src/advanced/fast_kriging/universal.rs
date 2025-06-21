//! Fast Universal Kriging Implementation
//!
//! This module contains the implementation of universal kriging for large datasets
//! using various approximation methods to improve computational efficiency.
//! Universal kriging allows for trend modeling using basis functions.

use crate::advanced::enhanced_kriging::{AnisotropicCovariance, TrendFunction};
use crate::advanced::fast_kriging::{
    FastKriging, FastKrigingBuilder, FastKrigingMethod, FastPredictionResult,
};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

// Import shared utility functions from parent module
use super::covariance::{
    compute_anisotropic_distance, compute_covariance, compute_low_rank_approximation,
    compute_tapered_covariance, find_nearest_neighbors, project_to_feature,
};

/// Create basis functions for the trend model
pub fn create_basis_functions<F: Float + FromPrimitive>(
    points: &ArrayView2<F>,
    trend_fn: TrendFunction,
) -> InterpolateResult<Array2<F>> {
    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];

    match trend_fn {
        TrendFunction::Constant => {
            // Just a constant term: X = [1, 1, ..., 1]
            let mut basis = Array2::zeros((n_points, 1));
            for i in 0..n_points {
                basis[[i, 0]] = F::one();
            }
            Ok(basis)
        }
        TrendFunction::Linear => {
            // Constant term + linear terms: X = [1, x1, x2, ..., xn]
            let n_basis = n_dims + 1;
            let mut basis = Array2::zeros((n_points, n_basis));

            // Constant term
            for i in 0..n_points {
                basis[[i, 0]] = F::one();
            }

            // Linear terms
            for i in 0..n_points {
                for j in 0..n_dims {
                    basis[[i, j + 1]] = points[[i, j]];
                }
            }

            Ok(basis)
        }
        TrendFunction::Quadratic => {
            // Constant + linear + quadratic terms
            // X = [1, x1, x2, ..., xn, x1^2, x1*x2, ..., xn^2]

            // Number of basis functions:
            // 1 (constant) + n_dims (linear) + n_dims*(n_dims+1)/2 (quadratic)
            let n_quad_terms = n_dims * (n_dims + 1) / 2;
            let n_basis = 1 + n_dims + n_quad_terms;

            let mut basis = Array2::zeros((n_points, n_basis));

            // Constant term
            for i in 0..n_points {
                basis[[i, 0]] = F::one();
            }

            // Linear terms
            for i in 0..n_points {
                for j in 0..n_dims {
                    basis[[i, j + 1]] = points[[i, j]];
                }
            }

            // Quadratic terms
            let mut term_idx = 1 + n_dims;
            for i in 0..n_points {
                for j in 0..n_dims {
                    for k in j..n_dims {
                        if j == k {
                            // x_j^2
                            basis[[i, term_idx]] = points[[i, j]] * points[[i, j]];
                        } else {
                            // x_j * x_k
                            basis[[i, term_idx]] = points[[i, j]] * points[[i, k]];
                        }
                        term_idx += 1;
                    }
                }
            }

            Ok(basis)
        }
        TrendFunction::Custom(_) => {
            // For custom basis functions, default to constant
            let mut basis = Array2::zeros((n_points, 1));
            for i in 0..n_points {
                basis[[i, 0]] = F::one();
            }
            Ok(basis)
        }
    }
}

/// Compute trend coefficients using least squares
pub fn compute_trend_coefficients<F: Float + FromPrimitive + 'static>(
    _points: &Array2<F>,
    values: &Array1<F>,
    basis_functions: &Array2<F>,
    _trend_fn: TrendFunction,
) -> InterpolateResult<Array1<F>> {
    // Basic least squares: β = (X'X)^(-1) X'y
    // Compute matrix products for least squares
    #[allow(unused_variables)]
    let xtx = basis_functions.t().dot(basis_functions);
    #[allow(unused_variables)]
    let xty = basis_functions.t().dot(values);

    #[cfg(feature = "linalg")]
    {
        use ndarray_linalg::Solve;
        // Convert to f64 for linear algebra
        let xtx_f64 = xtx.mapv(|x| x.to_f64().unwrap());
        let xty_f64 = xty.mapv(|x| x.to_f64().unwrap());
        match xtx_f64.solve(&xty_f64) {
            Ok(coeffs) => Ok(coeffs.mapv(|x| F::from_f64(x).unwrap())),
            Err(_) => {
                // Fallback to simple mean for constant trend
                let mut coeffs = Array1::zeros(basis_functions.shape()[1]);
                coeffs[0] = values.mean().unwrap_or(F::zero());
                Ok(coeffs)
            }
        }
    }

    #[cfg(not(feature = "linalg"))]
    {
        // Without linalg, just use mean as constant
        let mut coeffs = Array1::zeros(basis_functions.shape()[1]);
        coeffs[0] = values.mean().unwrap_or(F::zero());
        Ok(coeffs)
    }
}

impl<F> FastKriging<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Local kriging prediction with trend model
    pub fn predict_local_with_trend(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let n_query = query_points.shape()[0];
        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        // Compute basis functions for query points
        let query_basis = create_basis_functions(query_points, self.trend_fn)?;

        // For each query point
        for i in 0..n_query {
            let query_point = query_points.slice(ndarray::s![i, ..]);

            // Find nearest neighbors
            let (indices, _distances) = find_nearest_neighbors(
                &query_point,
                &self.points,
                self.max_neighbors,
                self.radius_multiplier,
            )?;

            // Skip if no neighbors found
            if indices.is_empty() {
                // Use global mean as fallback
                values[i] = self.values.mean().unwrap_or(F::zero());
                variances[i] = self.anisotropic_cov.sigma_sq;
                continue;
            }

            // Extract local neighborhood
            let n_neighbors = indices.len();
            let mut local_points = Array2::zeros((n_neighbors, query_point.len()));
            let mut local_values = Array1::zeros(n_neighbors);

            for (j, &idx) in indices.iter().enumerate() {
                local_points
                    .slice_mut(ndarray::s![j, ..])
                    .assign(&self.points.slice(ndarray::s![idx, ..]));
                local_values[j] = self.values[idx];
            }

            // Compute local covariance matrix
            let mut cov_matrix = Array2::zeros((n_neighbors, n_neighbors));
            for j in 0..n_neighbors {
                for k in 0..n_neighbors {
                    if j == k {
                        cov_matrix[[j, k]] =
                            self.anisotropic_cov.sigma_sq + self.anisotropic_cov.nugget;
                    } else {
                        let dist = compute_anisotropic_distance(
                            &local_points.slice(ndarray::s![j, ..]),
                            &local_points.slice(ndarray::s![k, ..]),
                            &self.anisotropic_cov,
                        )?;
                        cov_matrix[[j, k]] = compute_covariance(dist, &self.anisotropic_cov);
                    }
                }
            }

            // Compute local trend basis
            let local_basis = create_basis_functions(&local_points.view(), self.trend_fn)?;
            let n_basis = local_basis.shape()[1];

            // Augmented system for Universal Kriging
            let mut aug_matrix = Array2::zeros((n_neighbors + n_basis, n_neighbors + n_basis));

            // Fill covariance block
            for j in 0..n_neighbors {
                for k in 0..n_neighbors {
                    aug_matrix[[j, k]] = cov_matrix[[j, k]];
                }
            }

            // Fill basis function blocks
            for j in 0..n_neighbors {
                for k in 0..n_basis {
                    let idx = n_neighbors + k;
                    aug_matrix[[j, idx]] = local_basis[[j, k]];
                    aug_matrix[[idx, j]] = local_basis[[j, k]];
                }
            }

            // Zero block in lower right
            for j in 0..n_basis {
                for k in 0..n_basis {
                    let idx1 = n_neighbors + j;
                    let idx2 = n_neighbors + k;
                    aug_matrix[[idx1, idx2]] = F::zero();
                }
            }

            // Create right-hand side
            let mut rhs = Array1::zeros(n_neighbors + n_basis);
            for j in 0..n_neighbors {
                rhs[j] = local_values[j];
            }

            // Solve the system
            #[cfg(not(feature = "linalg"))]
            {
                // Fallback without linalg
                values[i] = local_values.mean().unwrap_or(F::zero());
                variances[i] = self.anisotropic_cov.sigma_sq;
                continue;
            }

            // Only gets here if linalg is enabled
            #[cfg(feature = "linalg")]
            {
                use ndarray_linalg::Solve;
                // Convert to f64 for linear algebra
                let aug_matrix_f64 = aug_matrix.mapv(|x| x.to_f64().unwrap());
                let rhs_f64 = rhs.mapv(|x| x.to_f64().unwrap());
                let solution = match aug_matrix_f64.solve(&rhs_f64) {
                    Ok(sol) => sol.mapv(|x| F::from_f64(x).unwrap()),
                    Err(_) => {
                        // Fallback to standard kriging if system can't be solved
                        let cov_matrix_f64 = cov_matrix.mapv(|x| x.to_f64().unwrap());
                        let local_values_f64 = local_values.mapv(|x| x.to_f64().unwrap());
                        let weights = match cov_matrix_f64.solve(&local_values_f64) {
                            Ok(w) => w.mapv(|x| F::from_f64(x).unwrap()),
                            Err(_) => {
                                // Return mean as last resort
                                values[i] = local_values.mean().unwrap_or(F::zero());
                                variances[i] = self.anisotropic_cov.sigma_sq;
                                continue;
                            }
                        };

                        // Use weights for prediction
                        let mut prediction = F::zero();
                        for j in 0..n_neighbors {
                            prediction += weights[j] * local_values[j];
                        }

                        // Return basic prediction
                        values[i] = prediction;
                        variances[i] = self.anisotropic_cov.sigma_sq; // Simplified variance
                        continue;
                    }
                };

                // Extract weights and trend coefficients
                let weights = solution.slice(ndarray::s![0..n_neighbors]).to_owned();
                let trend_coeffs = solution.slice(ndarray::s![n_neighbors..]).to_owned();

                // Compute prediction
                let mut trend = F::zero();
                for j in 0..n_basis {
                    trend += trend_coeffs[j] * query_basis[[i, j]];
                }

                let mut residual = F::zero();
                for j in 0..n_neighbors {
                    residual += weights[j] * local_values[j];
                }

                // Store prediction
                values[i] = trend + residual;

                // Compute approximate variance (simplified)
                let mut k_star = Array1::zeros(n_neighbors);
                for j in 0..n_neighbors {
                    let dist = compute_anisotropic_distance(
                        &query_point,
                        &local_points.slice(ndarray::s![j, ..]),
                        &self.anisotropic_cov,
                    )?;
                    k_star[j] = compute_covariance(dist, &self.anisotropic_cov);
                }

                let variance = self.anisotropic_cov.sigma_sq - k_star.dot(&weights);
                variances[i] = if variance < F::zero() {
                    F::zero()
                } else {
                    variance
                };
            }
        }

        Ok(FastPredictionResult {
            value: values,
            variance: variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// HODLR kriging prediction with hierarchical matrices (universal version)
    pub fn predict_hodlr(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        // Extract leaf size from method
        let leaf_size = match self.approx_method {
            FastKrigingMethod::HODLR(size) => size,
            _ => {
                return Err(InterpolateError::InvalidOperation(
                    "Invalid method type for HODLR prediction".to_string(),
                ));
            }
        };

        // For HODLR, we divide the dataset into a hierarchical tree of blocks
        // For this implementation, we'll use a simplified approach with recursion

        let n_query = query_points.shape()[0];
        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        // Compute basis functions for query points if needed for universal kriging
        let query_basis = create_basis_functions(query_points, self.trend_fn)?;

        // Create temporary trend coefficients if not pre-computed
        let trend_coeffs = match &self.trend_coeffs {
            Some(coeffs) => coeffs.clone(),
            None => {
                // We need to compute trend coefficients
                let basis_functions = match &self.basis_functions {
                    Some(basis) => basis,
                    None => {
                        // Create basis functions first
                        &create_basis_functions(&self.points.view(), self.trend_fn)?
                    }
                };

                compute_trend_coefficients(
                    &self.points,
                    &self.values,
                    basis_functions,
                    self.trend_fn,
                )?
            }
        };

        // Recursion helper: Start with the full dataset
        let n_train = self.points.shape()[0];
        let train_indices: Vec<usize> = (0..n_train).collect();

        // For each query point
        for i in 0..n_query {
            let query_point = query_points.slice(ndarray::s![i, ..]);

            // Compute trend component
            let mut trend = F::zero();
            for j in 0..query_basis.shape()[1] {
                trend += trend_coeffs[j] * query_basis[[i, j]];
            }

            // Compute prediction using HODLR approximation
            let residual =
                self.hodlr_predict_point(&query_point, &train_indices, leaf_size, trend)?;

            // Store results
            values[i] = trend + residual;

            // For variance, we use a simplified approximation
            // In a full implementation, this would involve traversing the hierarchy
            variances[i] = self.anisotropic_cov.sigma_sq * F::from_f64(0.1).unwrap();
        }

        Ok(FastPredictionResult {
            value: values,
            variance: variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// Helper function for HODLR prediction of a single point
    pub fn hodlr_predict_point(
        &self,
        query_point: &ArrayView1<F>,
        indices: &[usize],
        leaf_size: usize,
        trend: F,
    ) -> InterpolateResult<F> {
        // If we're at a leaf node or only have a few points, solve directly
        if indices.len() <= leaf_size {
            // Use direct solution for small blocks
            let n_points = indices.len();

            // Extract subset of points
            let mut block_points = Array2::zeros((n_points, query_point.len()));
            let mut block_values = Array1::zeros(n_points);

            for (i, &idx) in indices.iter().enumerate() {
                block_points
                    .slice_mut(ndarray::s![i, ..])
                    .assign(&self.points.slice(ndarray::s![idx, ..]));
                block_values[i] = self.values[idx] - trend; // Use residuals
            }

            // Compute covariance matrix for this block
            let mut cov_matrix = Array2::zeros((n_points, n_points));
            for i in 0..n_points {
                for j in 0..n_points {
                    if i == j {
                        cov_matrix[[i, j]] =
                            self.anisotropic_cov.sigma_sq + self.anisotropic_cov.nugget;
                    } else {
                        let dist = compute_anisotropic_distance(
                            &block_points.slice(ndarray::s![i, ..]),
                            &block_points.slice(ndarray::s![j, ..]),
                            &self.anisotropic_cov,
                        )?;
                        cov_matrix[[i, j]] = compute_covariance(dist, &self.anisotropic_cov);
                    }
                }
            }

            // Compute cross-covariance vector
            let mut k_star = Array1::zeros(n_points);
            for i in 0..n_points {
                let dist = compute_anisotropic_distance(
                    query_point,
                    &block_points.slice(ndarray::s![i, ..]),
                    &self.anisotropic_cov,
                )?;
                k_star[i] = compute_covariance(dist, &self.anisotropic_cov);
            }

            // Solve the system for weights
            #[cfg(feature = "linalg")]
            let weights = {
                use ndarray_linalg::Solve;
                // Convert to f64 for linear algebra
                let cov_matrix_f64 = cov_matrix.mapv(|x| x.to_f64().unwrap());
                let block_values_f64 = block_values.mapv(|x| x.to_f64().unwrap());
                match cov_matrix_f64.solve(&block_values_f64) {
                    Ok(w) => w.mapv(|x| F::from_f64(x).unwrap()),
                    Err(_) => {
                        // Fallback to diagonal approximation
                        let mut w = Array1::zeros(n_points);
                        for i in 0..n_points {
                            w[i] = block_values[i]
                                / (self.anisotropic_cov.sigma_sq + self.anisotropic_cov.nugget);
                        }
                        w
                    }
                }
            };

            #[cfg(not(feature = "linalg"))]
            let weights = {
                // Fallback without linalg
                let mut w = Array1::zeros(n_points);
                for i in 0..n_points {
                    w[i] = block_values[i]
                        / (self.anisotropic_cov.sigma_sq + self.anisotropic_cov.nugget);
                }
                w
            };

            // Compute prediction
            let prediction = k_star.dot(&weights);

            return Ok(prediction);
        }

        // Otherwise, partition the points into near and far sets
        // For simplicity, we split on the dimension with largest extent
        let n_dims = self.points.shape()[1];
        let mut max_extent = F::zero();
        let mut split_dim = 0;

        for d in 0..n_dims {
            let mut min_val = F::infinity();
            let mut max_val = F::neg_infinity();

            for &idx in indices {
                let val = self.points[[idx, d]];
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let extent = max_val - min_val;
            if extent > max_extent {
                max_extent = extent;
                split_dim = d;
            }
        }

        // Find median value for splitting
        let mut values_at_dim: Vec<F> = indices
            .iter()
            .map(|&idx| self.points[[idx, split_dim]])
            .collect();
        values_at_dim.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Partition into near and far sets
        let query_val = query_point[split_dim];
        let (near_indices, far_indices): (Vec<usize>, Vec<usize>) =
            indices.iter().copied().partition(|&idx| {
                let dist = (self.points[[idx, split_dim]] - query_val).abs();
                dist <= max_extent * F::from_f64(0.5).unwrap()
            });

        // Recursively compute prediction for near points
        let near_prediction = if !near_indices.is_empty() {
            self.hodlr_predict_point(query_point, &near_indices, leaf_size, trend)?
        } else {
            F::zero()
        };

        // For far points, use low-rank approximation based on a few sample points
        let far_prediction = if !far_indices.is_empty() {
            // Select a subsample of far points for low-rank approximation
            let n_samples =
                std::cmp::min(far_indices.len(), std::cmp::max(5, far_indices.len() / 10));

            let step = if n_samples >= far_indices.len() {
                1
            } else {
                far_indices.len() / n_samples
            };

            let mut sample_indices = Vec::with_capacity(n_samples);
            for i in (0..far_indices.len()).step_by(step) {
                if sample_indices.len() < n_samples {
                    sample_indices.push(far_indices[i]);
                } else {
                    break;
                }
            }

            // Compute a simplified low-rank approximation for the far points
            self.hodlr_predict_point(query_point, &sample_indices, leaf_size, trend)?
                * F::from_f64(far_indices.len() as f64 / sample_indices.len() as f64).unwrap()
        } else {
            F::zero()
        };

        // Combine results with appropriate weighting
        let total_points = near_indices.len() + far_indices.len();
        let near_weight = F::from_f64(near_indices.len() as f64 / total_points as f64).unwrap();
        let far_weight = F::from_f64(far_indices.len() as f64 / total_points as f64).unwrap();

        Ok(near_weight * near_prediction + far_weight * far_prediction)
    }
}

/// Builder extension methods for universal kriging
impl<F> FastKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Build a universal fast kriging model 
    pub fn build_universal(self) -> InterpolateResult<FastKriging<F>> {
        FastKriging::from_builder(self)
    }
}
