//! Spatial statistics module for analyzing spatial patterns and relationships
//!
//! This module provides statistical measures commonly used in spatial analysis,
//! including measures of spatial autocorrelation, clustering, and pattern analysis.
//!
//! # Features
//!
//! * **Spatial Autocorrelation**: Moran's I, Geary's C
//! * **Local Indicators**: Local Moran's I (LISA)
//! * **Distance-based Statistics**: Getis-Ord statistics
//! * **Pattern Analysis**: Nearest neighbor analysis
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_spatial::spatial_stats::{morans_i, gearys_c};
//!
//! // Create spatial data (values at different locations)
//! let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
//!
//! // Define spatial weights matrix (adjacency-based)
//! let weights = array![
//!     [0.0, 1.0, 0.0, 0.0, 1.0],
//!     [1.0, 0.0, 1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0, 0.0, 1.0],
//!     [1.0, 0.0, 0.0, 1.0, 0.0],
//! ];
//!
//! // Calculate spatial autocorrelation
//! let moran = morans_i(&values.view(), &weights.view()).unwrap();
//! let geary = gearys_c(&values.view(), &weights.view()).unwrap();
//!
//! println!("Moran's I: {:.3}", moran);
//! println!("Geary's C: {:.3}", geary);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;

use crate::error::{SpatialError, SpatialResult};

/// Calculate Moran's I statistic for spatial autocorrelation
///
/// Moran's I measures the degree of spatial autocorrelation in a dataset.
/// Values range from -1 (perfect negative autocorrelation) to +1 (perfect positive autocorrelation).
/// A value of 0 indicates no spatial autocorrelation (random spatial pattern).
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix (typically binary adjacency or distance-based)
///
/// # Returns
///
/// * Moran's I statistic
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::morans_i;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let moran = morans_i(&values.view(), &weights.view()).unwrap();
/// println!("Moran's I: {:.3}", moran);
/// ```
#[allow(dead_code)]
pub fn morans_i<T: Float>(values: &ArrayView1<T>, weights: &ArrayView2<T>) -> SpatialResult<T> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate mean
    let mean = values.sum() / T::from(n).unwrap();

    // Calculate deviations from mean
    let deviations: Array1<T> = values.map(|&x| x - mean);

    // Calculate sum of weights
    let w_sum = weights.sum();

    if w_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Sum of weights cannot be zero".to_string(),
        ));
    }

    // Calculate numerator: sum of (w_ij * (x_i - mean) * (x_j - mean))
    let mut numerator = T::zero();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                numerator = numerator + weights[[i, j]] * deviations[i] * deviations[j];
            }
        }
    }

    // Calculate denominator: sum of (x_i - mean)^2
    let denominator: T = deviations.map(|&x| x * x).sum();

    if denominator.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    // Moran's I = (n / W) * (numerator / denominator)
    let morans_i = (T::from(n).unwrap() / w_sum) * (numerator / denominator);

    Ok(morans_i)
}

/// Calculate Geary's C statistic for spatial autocorrelation
///
/// Geary's C is another measure of spatial autocorrelation that ranges from 0 to 2.
/// Values close to 1 indicate no spatial autocorrelation, values < 1 indicate positive
/// autocorrelation, and values > 1 indicate negative autocorrelation.
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
///
/// # Returns
///
/// * Geary's C statistic
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::gearys_c;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let geary = gearys_c(&values.view(), &weights.view()).unwrap();
/// println!("Geary's C: {:.3}", geary);
/// ```
#[allow(dead_code)]
pub fn gearys_c<T: Float>(values: &ArrayView1<T>, weights: &ArrayView2<T>) -> SpatialResult<T> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate mean
    let mean = values.sum() / T::from(n).unwrap();

    // Calculate sum of weights
    let w_sum = weights.sum();

    if w_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Sum of weights cannot be zero".to_string(),
        ));
    }

    // Calculate numerator: sum of (w_ij * (x_i - x_j)^2)
    let mut numerator = T::zero();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = values[i] - values[j];
                numerator = numerator + weights[[i, j]] * diff * diff;
            }
        }
    }

    // Calculate denominator: 2 * W * sum of (x_i - mean)^2
    let variance_sum: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum();

    if variance_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let denominator = (T::one() + T::one()) * w_sum * variance_sum;

    // Geary's C = ((n-1) / 2W) * (numerator / variance_sum)
    let gearys_c = (T::from((n - 1) as i32).unwrap() / denominator) * numerator;

    Ok(gearys_c)
}

/// Calculate Local Indicators of Spatial Association (LISA) using Local Moran's I
///
/// Local Moran's I identifies clusters and outliers for each location individually.
/// Positive values indicate that a location is part of a cluster of similar values,
/// while negative values indicate spatial outliers.
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
///
/// # Returns
///
/// * Array of Local Moran's I values, one for each location
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::local_morans_i;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let local_i = local_morans_i(&values.view(), &weights.view()).unwrap();
/// println!("Local Moran's I values: {:?}", local_i);
/// ```
#[allow(dead_code)]
pub fn local_morans_i<T: Float>(
    values: &ArrayView1<T>,
    weights: &ArrayView2<T>,
) -> SpatialResult<Array1<T>> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate global mean
    let mean = values.sum() / T::from(n).unwrap();

    // Calculate global variance
    let variance: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum()
        / T::from(n).unwrap();

    if variance.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let mut local_i = Array1::zeros(n);

    for i in 0..n {
        let zi = (values[i] - mean) / variance.sqrt();

        // Calculate weighted sum of neighboring deviations
        let mut weighted_sum = T::zero();
        for j in 0..n {
            if i != j && weights[[i, j]] > T::zero() {
                let zj = (values[j] - mean) / variance.sqrt();
                weighted_sum = weighted_sum + weights[[i, j]] * zj;
            }
        }

        local_i[i] = zi * weighted_sum;
    }

    Ok(local_i)
}

/// Calculate Getis-Ord Gi statistic for hotspot analysis
///
/// The Getis-Ord Gi statistic identifies statistically significant spatial
/// clusters of high values (hotspots) and low values (coldspots).
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
/// * `include_self` - Whether to include the focal location in the calculation
///
/// # Returns
///
/// * Array of Gi statistics, one for each location
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::getis_ord_gi;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let gi_stats = getis_ord_gi(&values.view(), &weights.view(), false).unwrap();
/// println!("Gi statistics: {:?}", gi_stats);
/// ```
#[allow(dead_code)]
pub fn getis_ord_gi<T: Float>(
    values: &ArrayView1<T>,
    weights: &ArrayView2<T>,
    include_self: bool,
) -> SpatialResult<Array1<T>> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate global mean and variance
    let mean = values.sum() / T::from(n).unwrap();
    let variance: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum()
        / T::from(n).unwrap();

    if variance.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let _std_dev = variance.sqrt();
    let mut gi_stats = Array1::zeros(n);

    for i in 0..n {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut weight_sum_squared = T::zero();

        for j in 0..n {
            let use_weight = if include_self {
                weights[[i, j]]
            } else if i == j {
                T::zero()
            } else {
                weights[[i, j]]
            };

            if use_weight > T::zero() {
                weighted_sum = weighted_sum + use_weight * values[j];
                weight_sum = weight_sum + use_weight;
                weight_sum_squared = weight_sum_squared + use_weight * use_weight;
            }
        }

        if weight_sum > T::zero() {
            let n_f = T::from(n).unwrap();
            let expected = weight_sum * mean;

            // Calculate standard deviation of the sum
            let variance_of_sum =
                (n_f * weight_sum_squared - weight_sum * weight_sum) * variance / (n_f - T::one());

            if variance_of_sum > T::zero() {
                gi_stats[i] = (weighted_sum - expected) / variance_of_sum.sqrt();
            }
        }
    }

    Ok(gi_stats)
}

/// Calculate spatial weights matrix based on distance decay
///
/// Creates a spatial weights matrix where weights decay with distance according
/// to a specified function (inverse distance, exponential decay, etc.).
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each location
/// * `max_distance` - Maximum distance for neighbors (beyond this, weight = 0)
/// * `decay_function` - Function to apply distance decay ("inverse", "exponential", "gaussian")
/// * `bandwidth` - Parameter controlling the rate of decay
///
/// # Returns
///
/// * Spatial weights matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::distance_weights_matrix;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let weights = distance_weights_matrix(
///     &coords.view(),
///     2.0,
///     "inverse",
///     1.0
/// ).unwrap();
///
/// println!("Distance-based weights matrix: {:?}", weights);
/// ```
#[allow(dead_code)]
pub fn distance_weights_matrix<T: Float>(
    coordinates: &ArrayView2<T>,
    max_distance: T,
    decay_function: &str,
    bandwidth: T,
) -> SpatialResult<Array2<T>> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D (x, y)".to_string(),
        ));
    }

    let mut weights = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Calculate Euclidean _distance
                let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
                let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
                let _distance = (dx * dx + dy * dy).sqrt();

                if _distance <= max_distance {
                    let weight = match decay_function {
                        "inverse" => {
                            if _distance > T::zero() {
                                T::one() / (T::one() + _distance / bandwidth)
                            } else {
                                T::zero()
                            }
                        }
                        "exponential" => (-_distance / bandwidth).exp(),
                        "gaussian" => {
                            let exponent = -(_distance * _distance) / (bandwidth * bandwidth);
                            exponent.exp()
                        }
                        _ => {
                            return Err(SpatialError::ValueError(
                                "Unknown decay function. Use 'inverse', 'exponential', or 'gaussian'".to_string(),
                            ));
                        }
                    };

                    weights[[i, j]] = weight;
                }
            }
        }
    }

    Ok(weights)
}

/// Calculate Clark-Evans nearest neighbor index
///
/// The Clark-Evans index compares the average nearest neighbor distance
/// to the expected distance in a random point pattern. Values < 1 indicate
/// clustering, values > 1 indicate regularity, and values ≈ 1 indicate randomness.
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each point
/// * `study_area` - Area of the study region
///
/// # Returns
///
/// * Clark-Evans index (R)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::clark_evans_index;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let ce_index = clark_evans_index(&coords.view(), 4.0).unwrap();
/// println!("Clark-Evans index: {:.3}", ce_index);
/// ```
#[allow(dead_code)]
pub fn clark_evans_index<T: Float>(coordinates: &ArrayView2<T>, study_area: T) -> SpatialResult<T> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D (x, y)".to_string(),
        ));
    }

    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points to calculate nearest neighbor distances".to_string(),
        ));
    }

    // Calculate nearest neighbor distances
    let mut nn_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_distance = T::infinity();

        for j in 0..n {
            if i != j {
                let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
                let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
                let distance = (dx * dx + dy * dy).sqrt();

                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        nn_distances.push(min_distance);
    }

    // Calculate observed mean nearest neighbor distance
    let observed_mean =
        nn_distances.iter().fold(T::zero(), |acc, &d| acc + d) / T::from(n).unwrap();

    // Calculate expected mean nearest neighbor distance for random pattern
    let density = T::from(n).unwrap() / study_area;
    let expected_mean = T::one() / (T::from(2.0).unwrap() * density.sqrt());

    // Clark-Evans index
    let clark_evans = observed_mean / expected_mean;

    Ok(clark_evans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_morans_i() {
        // Test with a simple case where adjacent values are similar
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let moran = morans_i(&values.view(), &weights.view()).unwrap();

        // Should be positive due to spatial clustering
        assert!(moran > 0.0);
    }

    #[test]
    fn test_gearys_c() {
        // Test with clustered data
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let geary = gearys_c(&values.view(), &weights.view()).unwrap();

        // Should be less than 1 due to positive spatial autocorrelation
        assert!(geary < 1.0);
    }

    #[test]
    fn test_local_morans_i() {
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let local_i = local_morans_i(&values.view(), &weights.view()).unwrap();

        // Should have 5 values (one for each location)
        assert_eq!(local_i.len(), 5);
    }

    #[test]
    fn test_distance_weights_matrix() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0],];

        let weights = distance_weights_matrix(&coords.view(), 1.5, "inverse", 1.0).unwrap();

        // Check dimensions
        assert_eq!(weights.shape(), &[4, 4]);

        // Diagonal should be zero
        for i in 0..4 {
            assert_relative_eq!(weights[[i, i]], 0.0, epsilon = 1e-10);
        }

        // Points (0,0) and (1,0) should have positive weight (distance = 1)
        assert!(weights[[0, 1]] > 0.0);
        assert!(weights[[1, 0]] > 0.0);

        // Points (0,0) and (2,2) should have zero weight (distance > max_distance)
        assert_relative_eq!(weights[[0, 3]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clark_evans_index() {
        // Perfect grid pattern should have R > 1 (regular)
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let ce_index = clark_evans_index(&coords.view(), 4.0).unwrap();

        // Grid pattern should be regular (R > 1)
        assert!(ce_index > 1.0);
    }
}
