//! Fast Ordinary Kriging Implementation
//!
//! This module contains the implementation of ordinary kriging for large datasets
//! using various approximation methods to improve computational efficiency.
//! Ordinary kriging assumes a constant but unknown mean.

use crate::advanced::enhanced_kriging::AnisotropicCovariance;
use crate::advanced::fast_kriging::{
    FastKriging, FastKrigingBuilder, FastKrigingMethod, FastPredictionResult, SparseComponents,
};
use crate::error::{InterpolateError, InterpolateResult};
use crate::numerical_stability::{assess_matrix_condition, safe_reciprocal, StabilityLevel};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

// Import shared utility functions from parent module
use super::covariance::{
use statrs::statistics::Statistics;
    compute_anisotropic_distance, compute_covariance, compute_low_rank_approximation,
    compute_tapered_covariance, find_nearest_neighbors, project_to_feature,
};

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
    /// Predict values at new points using fast approximation with local kriging
    pub fn predict_local(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let n_query = query_points.shape()[0];
        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        // For each query point
        for i in 0..n_query {
            let query_point = query_points.slice(scirs2_core::ndarray::s![i, ..]);

            // Find nearest neighbors
            let (indices_distances) = find_nearest_neighbors(
                &query_point,
                &self._points,
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
                    .slice_mut(scirs2_core::ndarray::s![j, ..])
                    .assign(&self._points.slice(scirs2_core::ndarray::s![idx, ..]));
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
                            &local_points.slice(scirs2_core::ndarray::s![j, ..]),
                            &local_points.slice(scirs2_core::ndarray::s![k, ..]),
                            &self.anisotropic_cov,
                        )?;
                        cov_matrix[[j, k]] = compute_covariance(dist, &self.anisotropic_cov);
                    }
                }
            }

            // Simple Kriging with constant mean
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

                // Assess local covariance matrix condition before solving
                let condition_report = assess_matrix_condition(&cov_matrix.view());
                if let Ok(report) = condition_report {
                    match report.stability_level {
                        StabilityLevel::Poor => {
                            // Silently use regularized fallback for poorly conditioned matrices
                            let regularization = F::from_f64(1e-8).unwrap_or_else(|| F::epsilon());
                            let mut regularized_cov = cov_matrix.clone();
                            for j in 0..regularized_cov.nrows() {
                                regularized_cov[[j, j]] += regularization;
                            }

                            // Try regularized solve, otherwise fallback to mean
                            let cov_matrix_f64 = regularized_cov.mapv(|x| x.to_f64().unwrap());
                            let local_values_f64 = local_values.mapv(|x| x.to_f64().unwrap());

                            #[cfg(feature = "linalg")]
                            {
                                use ndarray_linalg::Solve;
                                if let Ok(weights_f64) = cov_matrix_f64.solve(&local_values_f64) {
                                    let weights = weights_f64.mapv(|x| F::from_f64(x).unwrap());
                                    let mut prediction = F::zero();
                                    for j in 0..n_neighbors {
                                        prediction += weights[j] * local_values[j];
                                    }
                                    values[i] = prediction;
                                    variances[i] =
                                        self.anisotropic_cov.sigma_sq * F::from_f64(0.1).unwrap();
                                    continue;
                                }
                            }

                            // Final fallback to mean
                            values[i] = local_values.mean().unwrap_or(F::zero());
                            variances[i] = self.anisotropic_cov.sigma_sq;
                            continue;
                        }
                        StabilityLevel::Marginal => {
                            // Apply light regularization for marginal conditioning
                            let regularization = F::from_f64(1e-10).unwrap_or_else(|| F::epsilon());
                            for j in 0..cov_matrix.nrows() {
                                cov_matrix[[j, j]] += regularization;
                            }
                        }
                        _ => {}
                    }
                }

                // Convert to f64 for linear algebra
                let cov_matrix_f64 = cov_matrix.mapv(|x| x.to_f64().unwrap());
                let local_values_f64 = local_values.mapv(|x| x.to_f64().unwrap());
                let weights = match cov_matrix_f64.solve(&local_values_f64) {
                    Ok(w) => w.mapv(|x| F::from_f64(x).unwrap()),
                    Err(_) => {
                        // Try iterative refinement with regularization
                        let regularization = F::from_f64(1e-6).unwrap_or_else(|| F::epsilon());
                        let mut regularized_cov = cov_matrix.clone();
                        for j in 0..regularized_cov.nrows() {
                            regularized_cov[[j, j]] += regularization;
                        }

                        let cov_matrix_f64 = regularized_cov.mapv(|x| x.to_f64().unwrap());
                        let local_values_f64 = local_values.mapv(|x| x.to_f64().unwrap());

                        #[cfg(feature = "linalg")]
                        {
                            use ndarray_linalg::Solve;
                            if let Ok(weights_f64) = cov_matrix_f64.solve(&local_values_f64) {
                                let weights = weights_f64.mapv(|x| F::from_f64(x).unwrap());
                                let mut prediction = F::zero();
                                for j in 0..n_neighbors {
                                    prediction += weights[j] * local_values[j];
                                }
                                values[i] = prediction;
                                variances[i] =
                                    self.anisotropic_cov.sigma_sq * F::from_f64(1.5).unwrap();
                                continue;
                            }
                        }

                        // Final fallback to mean
                        values[i] = local_values.mean().unwrap_or(F::zero());
                        variances[i] = self.anisotropic_cov.sigma_sq;
                        continue;
                    }
                };

                // Compute prediction
                let mut prediction = F::zero();
                for j in 0..n_neighbors {
                    prediction += weights[j] * local_values[j];
                }

                // Store prediction
                values[i] = prediction;

                // Compute approximate variance (simplified)
                let mut k_star = Array1::zeros(n_neighbors);
                for j in 0..n_neighbors {
                    let dist = compute_anisotropic_distance(
                        &query_point,
                        &local_points.slice(scirs2_core::ndarray::s![j, ..]),
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

    /// Fixed rank kriging prediction using low-rank approximation
    pub fn predict_fixed_rank(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        // Ensure we have the low-rank components
        let (u, s, v) = match &self.low_rank_components {
            Some(components) => components,
            None => {
                return Err(InterpolateError::InvalidOperation(
                    "Low-rank components not pre-computed for FixedRank method".to_string(),
                ));
            }
        };

        let n_query = query_points.shape()[0];
        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        // Compute cross-covariance matrix between query and training _points
        let rank = u.shape()[1];
        let mut query_features = Array2::zeros((n_query, rank));

        // Project query _points into low-rank feature space
        for i in 0..n_query {
            let query_point = query_points.slice(scirs2_core::ndarray::s![i, ..]);

            // Compute projection
            for j in 0..rank {
                let feature =
                    project_to_feature(&query_point, &self._points, j, &self.anisotropic_cov)?;
                query_features[[i, j]] = feature;
            }
        }

        // Compute predictions efficiently using low-rank structure
        for i in 0..n_query {
            // Compute residual using low-rank approximation
            let query_feature = "query_features".slice(scirs2_core::ndarray::s![i, ..]);

            // Create safe reciprocal array for singular values
            let mut s_inv = Array1::zeros(s.len());
            for (j, &sv) in s.iter().enumerate() {
                match safe_reciprocal(sv) {
                    Ok(inv_sv) => s_inv[j] = inv_sv,
                    Err(_) => {
                        // Silently handle small singular values by setting to zero
                        s_inv[j] = F::zero();
                    }
                }
            }

            let projected = u.dot(&(s_inv * v.t().dot(&self.values)));
            values[i] = query_feature.dot(&projected);

            // Approximate variance (simplified)
            let variance = self.anisotropic_cov.sigma_sq
                - query_feature.dot(&s_inv) * query_feature.dot(&s_inv);

            variances[i] = if variance < F::zero() {
                F::zero()
            } else {
                variance
            };
        }

        Ok(FastPredictionResult {
            value: values,
            variance: variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// Tapered kriging prediction using sparse matrices
    pub fn predict_tapered(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        // Ensure we have the sparse components
        let (indices, values_vec) = match &self.sparse_components {
            Some(components) => components,
            None => {
                return Err(InterpolateError::InvalidOperation(
                    "Sparse components not pre-computed for Tapering method".to_string(),
                ));
            }
        };

        let n_query = query_points.shape()[0];
        let mut pred_values = Array1::zeros(n_query);
        let mut pred_variances = Array1::zeros(n_query);

        // Extract taper range from method
        let taper_range = match self.approx_method {
            FastKrigingMethod::Tapering(range) => F::from_f64(range).unwrap(, _ => {
                return Err(InterpolateError::InvalidOperation(
                    "Invalid method type for tapered prediction".to_string(),
                ));
            }
        };

        // For each query point
        for i in 0..n_query {
            let query_point = query_points.slice(scirs2_core::ndarray::s![i, ..]);

            // Find training _points within taper range
            let n_train = self._points.shape()[0];
            let mut nonzero_indices = Vec::new();
            let mut k_star = Vec::new();

            for j in 0..n_train {
                let dist = compute_anisotropic_distance(
                    &query_point,
                    &self._points.slice(scirs2_core::ndarray::s![j, ..]),
                    &self.anisotropic_cov,
                )?;

                if dist <= taper_range {
                    nonzero_indices.push(j);
                    k_star.push(compute_covariance(dist, &self.anisotropic_cov));
                }
            }

            // If no _points within range, use global mean
            if nonzero_indices.is_empty() {
                pred_values[i] = self.values.mean().unwrap_or(F::zero());
                pred_variances[i] = self.anisotropic_cov.sigma_sq;
                continue;
            }

            // Create sparse vector for cross-covariance
            // This is a simplified sparse operation - a full implementation would
            // use a proper sparse matrix library
            let n_nonzero = nonzero_indices.len();
            let mut alpha = Array1::zeros(n_nonzero);

            // For each training point with non-zero covariance
            for (idx, &j) in nonzero_indices.iter().enumerate() {
                // Find corresponding value in the sparse representation
                let mut a_j = F::zero();
                for p in 0..n_train {
                    if p == j {
                        // This is the row we want
                        for (&(row, col), &val) in indices.iter().zip(values_vec.iter()) {
                            if row == p {
                                // Multiply sparse row by values vector
                                a_j += val * self.values[col];
                            }
                        }
                        break;
                    }
                }
                alpha[idx] = a_j;
            }

            // Compute prediction
            let mut prediction = F::zero();
            for idx in 0..n_nonzero {
                prediction += k_star[idx] * alpha[idx];
            }

            // Final prediction
            pred_values[i] = prediction;

            // Compute approximate variance (simplified for sparse case)
            let mut variance = self.anisotropic_cov.sigma_sq;
            for value in k_star.iter().take(n_nonzero) {
                variance -= *value * *value / self.anisotropic_cov.sigma_sq;
            }

            pred_variances[i] = if variance < F::zero() {
                F::zero()
            } else {
                variance
            };
        }

        Ok(FastPredictionResult {
            value: pred_values,
            variance: pred_variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// Predict values at new points using Hierarchical Off-Diagonal Low-Rank approximation
    pub fn predict_hodlr(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let n_query = query_points.shape()[0];
        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        // HODLR approximates the covariance matrix hierarchically
        // For this implementation, we'll use a simplified approach
        // that combines local kriging with global low-rank approximation

        let max_leaf_size = match self.approx_method {
            FastKrigingMethod::HODLR(leaf_size) => leaf_size_ => 32, // Default leaf size
        };

        // Divide the data into hierarchical blocks
        let n_points = self._points.nrows();
        let n_blocks = (n_points + max_leaf_size - 1) / max_leaf_size;

        for i in 0..n_query {
            let query_point = query_points.slice(scirs2_core::ndarray::s![i, ..]);

            // Find the most relevant blocks for this query point
            let mut block_contributions = Vec::new();
            let mut total_weight = F::zero();

            for block_idx in 0..n_blocks {
                let start_idx = block_idx * max_leaf_size;
                let end_idx = std::cmp::min(start_idx + max_leaf_size, n_points);

                if start_idx >= end_idx {
                    continue;
                }

                // Compute distance to block centroid
                let mut centroid = vec![F::zero(); query_point.len()];
                for j in start_idx..end_idx {
                    for d in 0..query_point.len() {
                        centroid[d] += self._points[[j, d]];
                    }
                }
                for d in 0..query_point.len() {
                    centroid[d] /= F::from_usize(end_idx - start_idx).unwrap();
                }

                // Compute distance to centroid
                let mut dist_sq = F::zero();
                for d in 0..query_point.len() {
                    let diff = query_point[d] - centroid[d];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();

                // Weight based on inverse distance
                let weight = if dist < F::from_f64(1e-10).unwrap() {
                    F::from_f64(1e10).unwrap() // Very close to centroid
                } else {
                    F::one() / (F::one() + dist)
                };

                // Compute block contribution
                if weight > F::from_f64(1e-6).unwrap() {
                    // Use local kriging within this block
                    let block_points = self._points.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]);
                    let block_values = self.values.slice(scirs2_core::ndarray::s![start_idx..end_idx]);

                    let local_prediction = self.predict_block_local(
                        &query_point,
                        &block_points.to_owned(),
                        &block_values.to_owned(),
                    )?;

                    block_contributions.push((local_prediction, weight));
                    total_weight += weight;
                }
            }

            // Combine block contributions with weighted average
            let mut prediction = F::zero();
            let mut variance = F::zero();

            if total_weight > F::zero() {
                for (local_pred, weight) in block_contributions {
                    let normalized_weight = weight / total_weight;
                    prediction += local_pred.0 * normalized_weight;
                    variance += local_pred.1 * normalized_weight * normalized_weight;
                }
            } else {
                // Fallback to global mean
                prediction = self.values.mean().unwrap_or(F::zero());
                variance = self.anisotropic_cov.sigma_sq;
            }

            values[i] = prediction;
            variances[i] = variance;
        }

        Ok(FastPredictionResult {
            value: values,
            variance: variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// Helper function for local block prediction in HODLR
    fn predict_block_local(
        &self,
        query_point: &ArrayView1<F>,
        block_points: &Array2<F>,
        block_values: &Array1<F>,
    ) -> InterpolateResult<(F, F)> {
        let n_block = block_points.nrows();

        if n_block == 0 {
            return Ok((F::zero(), self.anisotropic_cov.sigma_sq));
        }

        if n_block == 1 {
            // Single _point - return its value
            return Ok((block_values[0], self.anisotropic_cov.sigma_sq));
        }

        // Build local covariance matrix
        let mut cov_matrix = Array2::zeros((n_block, n_block));
        for j in 0..n_block {
            for k in 0..n_block {
                if j == k {
                    cov_matrix[[j, k]] =
                        self.anisotropic_cov.sigma_sq + self.anisotropic_cov.nugget;
                } else {
                    let dist = compute_anisotropic_distance(
                        &block_points.slice(scirs2_core::ndarray::s![j, ..]),
                        &block_points.slice(scirs2_core::ndarray::s![k, ..]),
                        &self.anisotropic_cov,
                    )?;
                    cov_matrix[[j, k]] = compute_covariance(dist, &self.anisotropic_cov);
                }
            }
        }

        // Compute cross-covariances with query _point
        let mut k_star = Array1::zeros(n_block);
        for j in 0..n_block {
            let dist = compute_anisotropic_distance(
                query_point,
                &block_points.slice(scirs2_core::ndarray::s![j, ..]),
                &self.anisotropic_cov,
            )?;
            k_star[j] = compute_covariance(dist, &self.anisotropic_cov);
        }

        // Solve for weights
        #[cfg(feature = "linalg")]
        {
            use ndarray_linalg::Solve;

            let cov_matrix_f64 = cov_matrix.mapv(|x| x.to_f64().unwrap());
            let block_values_f64 = block_values.mapv(|x| x.to_f64().unwrap());

            match cov_matrix_f64.solve(&block_values_f64) {
                Ok(weights_f64) => {
                    let weights = weights_f64.mapv(|x| F::from_f64(x).unwrap());
                    let prediction = k_star.dot(&weights);

                    // Compute variance
                    let k_star_f64 = k_star.mapv(|x| x.to_f64().unwrap());
                    let cov_inv_k_star = match cov_matrix_f64.solve(&k_star_f64) {
                        Ok(result) => result.mapv(|x| F::from_f64(x).unwrap()),
                        Err(_) => Array1::zeros(n_block),
                    };
                    let variance = self.anisotropic_cov.sigma_sq - k_star.dot(&cov_inv_k_star);

                    Ok((prediction, variance.max(F::zero())))
                }
                Err(_) => {
                    // Fallback to mean
                    let mean_val = block_values.mean().unwrap_or(F::zero());
                    Ok((mean_val, self.anisotropic_cov.sigma_sq))
                }
            }
        }

        #[cfg(not(feature = "linalg"))]
        {
            // Fallback without linalg
            let mean_val = block_values.mean().unwrap_or(F::zero());
            Ok((mean_val, self.anisotropic_cov.sigma_sq))
        }
    }
}

/// Builder extension methods for ordinary kriging
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
    /// Build an ordinary fast kriging model with default parameters
    pub fn build_ordinary(self) -> InterpolateResult<FastKriging<F>> {
        FastKriging::from_builder(self)
    }
}
