//! Covariance Functions for Fast Kriging
//!
//! This module provides implementations of covariance functions and related
//! utilities for fast kriging interpolation.

use crate::advanced::enhanced_kriging::AnisotropicCovariance;
use crate::advanced::kriging::CovarianceFunction;
use crate::error::InterpolateResult;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for sparse matrix representation
type SparseComponents<F> = (Vec<(usize, usize)>, Vec<F>);

/// Find the k nearest neighbors to a query point
pub fn find_nearest_neighbors<F: Float + FromPrimitive>(
    query_point: &ArrayView1<F>,
    points: &Array2<F>,
    max_neighbors: usize,
    radius_multiplier: F,
) -> InterpolateResult<(Vec<usize>, Vec<F>)> {
    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];

    // Calculate distances
    let mut distances = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let mut dist_sq = F::zero();
        for j in 0..n_dims {
            let diff = query_point[j] - points[[i, j]];
            dist_sq = dist_sq + diff * diff;
        }
        distances.push((i, dist_sq.sqrt()));
    }

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Determine search radius based on max_neighbors
    let radius = if distances.len() > max_neighbors {
        distances[max_neighbors - 1].1 * radius_multiplier
    } else {
        // If we have fewer points than max_neighbors, use all points
        F::infinity()
    };

    // Select points within radius, up to max_neighbors
    let mut indices = Vec::with_capacity(max_neighbors);
    let mut selected_distances = Vec::with_capacity(max_neighbors);

    for (idx, dist) in distances {
        if dist <= radius && indices.len() < max_neighbors {
            indices.push(idx);
            selected_distances.push(dist);
        }
    }

    Ok((indices, selected_distances))
}

/// Compute anisotropic distance between two points
pub fn compute_anisotropic_distance<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
>(
    p1: &ArrayView1<F>,
    p2: &ArrayView1<F>,
    anisotropic_cov: &AnisotropicCovariance<F>,
) -> InterpolateResult<F> {
    let n_dims = p1.len();

    if n_dims != anisotropic_cov.length_scales.len() {
        return Err(crate::error::InterpolateError::DimensionMismatch(
            "Number of length scales must match dimension of points".to_string(),
        ));
    }

    // Simple anisotropic distance (no rotation)
    let mut sum_sq = F::zero();
    for i in 0..n_dims {
        let diff = (p1[i] - p2[i]) / anisotropic_cov.length_scales[i];
        sum_sq += diff * diff;
    }

    Ok(sum_sq.sqrt())
}

/// Evaluate the covariance function
pub fn compute_covariance<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
>(
    r: F,
    anisotropic_cov: &AnisotropicCovariance<F>,
) -> F {
    match anisotropic_cov.cov_fn {
        CovarianceFunction::SquaredExponential => {
            // σ² exp(-r²)
            anisotropic_cov.sigma_sq * (-r * r).exp()
        }
        CovarianceFunction::Exponential => {
            // σ² exp(-r)
            anisotropic_cov.sigma_sq * (-r).exp()
        }
        CovarianceFunction::Matern32 => {
            // σ² (1 + √3r) exp(-√3r)
            let sqrt3_r = F::from_f64(3.0).unwrap().sqrt() * r;
            anisotropic_cov.sigma_sq * (F::one() + sqrt3_r) * (-sqrt3_r).exp()
        }
        CovarianceFunction::Matern52 => {
            // σ² (1 + √5r + 5r²/3) exp(-√5r)
            let sqrt5_r = F::from_f64(5.0).unwrap().sqrt() * r;
            let factor =
                F::one() + sqrt5_r + F::from_f64(5.0).unwrap() * r * r / F::from_f64(3.0).unwrap();
            anisotropic_cov.sigma_sq * factor * (-sqrt5_r).exp()
        }
        CovarianceFunction::RationalQuadratic => {
            // σ² (1 + r²/(2α))^(-α)
            let alpha = anisotropic_cov.extra_params;
            let r_sq_div_2a = r * r / (F::from_f64(2.0).unwrap() * alpha);
            anisotropic_cov.sigma_sq * (F::one() + r_sq_div_2a).powf(-alpha)
        }
    }
}

/// Compute low-rank approximation of the covariance matrix
pub fn compute_low_rank_approximation<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
>(
    points: &Array2<F>,
    anisotropic_cov: &AnisotropicCovariance<F>,
    rank: usize,
) -> InterpolateResult<(Array2<F>, Array1<F>, Array2<F>)> {
    // In a full implementation, you would use Nyström method or randomized SVD
    // For this simplified example, we'll compute a small sample covariance matrix

    let n_points = points.shape()[0];
    let max_sample = std::cmp::min(rank * 2, n_points);

    // Use a subset of points
    let mut sample_indices = Vec::with_capacity(max_sample);
    let step = n_points / max_sample;
    for i in 0..max_sample {
        sample_indices.push(i * step);
    }

    let mut sample_cov = Array2::zeros((max_sample, max_sample));

    // Compute sample covariance matrix
    for i in 0..max_sample {
        for j in 0..max_sample {
            let idx_i = sample_indices[i];
            let idx_j = sample_indices[j];

            if i == j {
                sample_cov[[i, j]] = anisotropic_cov.sigma_sq + anisotropic_cov.nugget;
            } else {
                let dist = compute_anisotropic_distance(
                    &points.slice(ndarray::s![idx_i, ..]),
                    &points.slice(ndarray::s![idx_j, ..]),
                    anisotropic_cov,
                )?;
                sample_cov[[i, j]] = compute_covariance(dist, anisotropic_cov);
            }
        }
    }

    // Compute SVD of sample covariance
    // SVD components with conditional compilation
    #[cfg(feature = "linalg")]
    let (u, s, vt) = {
        use ndarray_linalg::SVD;
        // Convert to f64 for SVD
        let sample_cov_f64 = sample_cov.mapv(|x| x.to_f64().unwrap());
        match sample_cov_f64.svd(true, true) {
            Ok((u_val, s_val, vt_val)) => {
                let u = u_val.map_or_else(
                    || Array2::eye(s_val.len()),
                    |u| u.mapv(|x| F::from_f64(x).unwrap()),
                );
                let s = s_val.mapv(|x| F::from_f64(x).unwrap());
                let vt = vt_val.map_or_else(
                    || Array2::eye(s_val.len()),
                    |vt| vt.mapv(|x| F::from_f64(x).unwrap()),
                );
                (u, s, vt)
            }
            Err(_) => {
                return Err(crate::error::InterpolateError::ComputationError(
                    "SVD computation failed for low-rank approximation".to_string(),
                ));
            }
        }
    };

    #[cfg(not(feature = "linalg"))]
    let (u, s, vt) = {
        // Fallback without SVD
        let identity = Array2::eye(max_sample);
        let values = Array1::from_elem(max_sample, anisotropic_cov.sigma_sq);
        (identity.clone(), values, identity)
    };

    // Truncate to desired rank
    let actual_rank = std::cmp::min(rank, s.len());
    let u_r = u.slice(ndarray::s![.., 0..actual_rank]).to_owned();
    let s_r = s.slice(ndarray::s![0..actual_rank]).to_owned();
    let v_r = vt.slice(ndarray::s![0..actual_rank, ..]).t().to_owned();

    Ok((u_r, s_r, v_r))
}

/// Compute tapered covariance representation
pub fn compute_tapered_covariance<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
>(
    points: &Array2<F>,
    anisotropic_cov: &AnisotropicCovariance<F>,
    taper_range: F,
) -> InterpolateResult<SparseComponents<F>> {
    let n_points = points.shape()[0];
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Compute sparse representation with tapering
    for i in 0..n_points {
        for j in 0..=i {
            // Lower triangular + diagonal
            if i == j {
                // Diagonal element
                let value = anisotropic_cov.sigma_sq + anisotropic_cov.nugget;
                indices.push((i, j));
                values.push(value);
            } else {
                // Off-diagonal element
                let dist = compute_anisotropic_distance(
                    &points.slice(ndarray::s![i, ..]),
                    &points.slice(ndarray::s![j, ..]),
                    anisotropic_cov,
                )?;

                // Apply tapering
                if dist <= taper_range {
                    let value = compute_covariance(dist, anisotropic_cov);
                    indices.push((i, j));
                    values.push(value);

                    // Add symmetric element
                    indices.push((j, i));
                    values.push(value);
                }
            }
        }
    }

    Ok((indices, values))
}

/// Project a point onto a feature in the feature space
pub fn project_to_feature<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
>(
    query_point: &ArrayView1<F>,
    points: &Array2<F>,
    feature_idx: usize,
    anisotropic_cov: &AnisotropicCovariance<F>,
) -> InterpolateResult<F> {
    let n_points = points.shape()[0];

    // In a full implementation, this would use the eigenvectors
    // For this simplified example, we'll use landmark points

    // Use a landmark point as basis for the feature
    let landmark_idx = feature_idx % n_points;
    let landmark = points.slice(ndarray::s![landmark_idx, ..]);

    // Project by computing covariance with landmark
    let dist = compute_anisotropic_distance(query_point, &landmark, anisotropic_cov)?;

    let projection = compute_covariance(dist, anisotropic_cov);

    Ok(projection)
}