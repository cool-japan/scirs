//! Enhanced input validation utilities
//!
//! This module provides comprehensive input validation functions that are
//! compatible with SciPy's validation patterns and provide consistent
//! error messages across all clustering algorithms.

use ndarray::{ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Validation configuration for different algorithms
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum number of samples required
    pub min_samples: usize,
    /// Maximum number of samples before warning
    pub max_samples_warning: Option<usize>,
    /// Minimum number of features required
    pub min_features: usize,
    /// Whether to check for finite values
    pub check_finite: bool,
    /// Whether to allow empty data
    pub allow_empty: bool,
    /// Custom error message prefix
    pub error_prefix: Option<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_samples: 2,
            max_samples_warning: Some(10000),
            min_features: 1,
            check_finite: true,
            allow_empty: false,
            error_prefix: None,
        }
    }
}

impl ValidationConfig {
    /// Create validation config for K-means
    pub fn for_kmeans() -> Self {
        Self {
            min_samples: 1,
            max_samples_warning: Some(50000),
            min_features: 1,
            check_finite: true,
            allow_empty: false,
            error_prefix: Some("K-means".to_string()),
        }
    }

    /// Create validation config for hierarchical clustering
    pub fn for_hierarchical() -> Self {
        Self {
            min_samples: 2,
            max_samples_warning: Some(5000),
            min_features: 1,
            check_finite: true,
            allow_empty: false,
            error_prefix: Some("Hierarchical clustering".to_string()),
        }
    }

    /// Create validation config for DBSCAN
    pub fn for_dbscan() -> Self {
        Self {
            min_samples: 2,
            max_samples_warning: Some(20000),
            min_features: 1,
            check_finite: true,
            allow_empty: false,
            error_prefix: Some("DBSCAN".to_string()),
        }
    }

    /// Create validation config for spectral clustering
    pub fn for_spectral() -> Self {
        Self {
            min_samples: 2,
            max_samples_warning: Some(1000),
            min_features: 1,
            check_finite: true,
            allow_empty: false,
            error_prefix: Some("Spectral clustering".to_string()),
        }
    }
}

/// Comprehensive data validation for clustering algorithms
///
/// Validates input data according to the specified configuration and provides
/// SciPy-compatible error messages.
///
/// # Arguments
///
/// * `data` - Input data matrix (n_samples × n_features)
/// * `config` - Validation configuration
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, detailed error if invalid
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::input_validation::{validate_clustering_data, ValidationConfig};
///
/// let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
/// let config = ValidationConfig::for_kmeans();
///
/// assert!(validate_clustering_data(data.view(), &config).is_ok());
/// ```
pub fn validate_clustering_data<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    config: &ValidationConfig,
) -> Result<()> {
    let (n_samples, n_features) = data.dim();
    let prefix = config.error_prefix.as_deref().unwrap_or("Clustering");

    // Check empty data
    if n_samples == 0 && !config.allow_empty {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Input data cannot be empty",
            prefix
        )));
    }

    if n_features == 0 && !config.allow_empty {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Input data must have at least one feature",
            prefix
        )));
    }

    // Check minimum requirements
    if n_samples < config.min_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Need at least {} samples, got {}",
            prefix, config.min_samples, n_samples
        )));
    }

    if n_features < config.min_features {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Need at least {} features, got {}",
            prefix, config.min_features, n_features
        )));
    }

    // Check for size warnings
    if let Some(max_warn) = config.max_samples_warning {
        if n_samples > max_warn {
            eprintln!(
                "Warning: {} with {} samples may be slow. Consider using a subset or more efficient algorithm.",
                prefix, n_samples
            );
        }
    }

    // Check for finite values
    if config.check_finite {
        validate_finite_values(data, prefix)?;
    }

    Ok(())
}

/// Validate that all values in the data are finite
fn validate_finite_values<F: Float + Debug>(data: ArrayView2<F>, prefix: &str) -> Result<()> {
    for (i, row) in data.axis_iter(Axis(0)).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ClusteringError::InvalidInput(format!(
                    "{}: Non-finite value {:?} at position ({}, {})",
                    prefix, value, i, j
                )));
            }
        }
    }
    Ok(())
}

/// Validate cluster count parameter
///
/// Ensures the number of clusters is valid for the given dataset.
///
/// # Arguments
///
/// * `n_clusters` - Number of clusters requested
/// * `n_samples` - Number of samples in dataset
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_n_clusters(n_clusters: usize, n_samples: usize, algorithm: &str) -> Result<()> {
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Number of clusters must be positive, got 0",
            algorithm
        )));
    }

    if n_clusters > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Number of clusters ({}) cannot exceed number of samples ({})",
            algorithm, n_clusters, n_samples
        )));
    }

    Ok(())
}

/// Validate distance/similarity parameters
///
/// Checks that distance thresholds and similarity parameters are valid.
///
/// # Arguments
///
/// * `value` - Parameter value to validate
/// * `param_name` - Parameter name for error messages
/// * `min_value` - Minimum allowed value (inclusive)
/// * `max_value` - Maximum allowed value (inclusive), None for no limit
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_distance_parameter<F: Float + FromPrimitive + Debug + PartialOrd>(
    value: F,
    param_name: &str,
    min_value: Option<F>,
    max_value: Option<F>,
    algorithm: &str,
) -> Result<()> {
    if !value.is_finite() {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: {} must be finite, got {:?}",
            algorithm, param_name, value
        )));
    }

    if let Some(min_val) = min_value {
        if value < min_val {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: {} must be >= {:?}, got {:?}",
                algorithm, param_name, min_val, value
            )));
        }
    }

    if let Some(max_val) = max_value {
        if value > max_val {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: {} must be <= {:?}, got {:?}",
                algorithm, param_name, max_val, value
            )));
        }
    }

    Ok(())
}

/// Validate integer parameters with bounds
///
/// Ensures integer parameters are within valid ranges.
///
/// # Arguments
///
/// * `value` - Parameter value to validate
/// * `param_name` - Parameter name for error messages
/// * `min_value` - Minimum allowed value (inclusive)
/// * `max_value` - Maximum allowed value (inclusive), None for no limit
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_integer_parameter(
    value: usize,
    param_name: &str,
    min_value: Option<usize>,
    max_value: Option<usize>,
    algorithm: &str,
) -> Result<()> {
    if let Some(min_val) = min_value {
        if value < min_val {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: {} must be >= {}, got {}",
                algorithm, param_name, min_val, value
            )));
        }
    }

    if let Some(max_val) = max_value {
        if value > max_val {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: {} must be <= {}, got {}",
                algorithm, param_name, max_val, value
            )));
        }
    }

    Ok(())
}

/// Validate sample weights
///
/// Ensures sample weights are valid (non-negative, finite, and consistent with data size).
///
/// # Arguments
///
/// * `weights` - Sample weights array
/// * `n_samples` - Expected number of samples
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_sample_weights<F: Float + FromPrimitive + Debug + PartialOrd>(
    weights: ArrayView1<F>,
    n_samples: usize,
    algorithm: &str,
) -> Result<()> {
    if weights.len() != n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Sample weights length ({}) must match number of samples ({})",
            algorithm,
            weights.len(),
            n_samples
        )));
    }

    for (i, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: Sample weight at index {} is not finite: {:?}",
                algorithm, i, weight
            )));
        }

        if weight < F::zero() {
            return Err(ClusteringError::InvalidInput(format!(
                "{}: Sample weight at index {} must be non-negative, got {:?}",
                algorithm, i, weight
            )));
        }
    }

    // Check that not all weights are zero
    let sum_weights = weights.iter().fold(F::zero(), |acc, &w| acc + w);
    if sum_weights <= F::zero() {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Sum of sample weights must be positive",
            algorithm
        )));
    }

    Ok(())
}

/// Validate cluster initialization data
///
/// Validates initial cluster centers or assignments for clustering algorithms.
///
/// # Arguments
///
/// * `init_data` - Initial cluster centers (k × n_features)
/// * `n_clusters` - Expected number of clusters
/// * `n_features` - Expected number of features
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_cluster_initialization<F: Float + FromPrimitive + Debug + PartialOrd>(
    init_data: ArrayView2<F>,
    n_clusters: usize,
    n_features: usize,
    algorithm: &str,
) -> Result<()> {
    let (init_clusters, init_features) = init_data.dim();

    if init_clusters != n_clusters {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Initial cluster centers must have {} clusters, got {}",
            algorithm, n_clusters, init_clusters
        )));
    }

    if init_features != n_features {
        return Err(ClusteringError::InvalidInput(format!(
            "{}: Initial cluster centers must have {} features, got {}",
            algorithm, n_features, init_features
        )));
    }

    // Check for finite values
    for (i, row) in init_data.axis_iter(Axis(0)).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ClusteringError::InvalidInput(format!(
                    "{}: Non-finite value {:?} in initial cluster center at position ({}, {})",
                    algorithm, value, i, j
                )));
            }
        }
    }

    Ok(())
}

/// Validate convergence parameters
///
/// Validates convergence threshold and maximum iterations for iterative algorithms.
///
/// # Arguments
///
/// * `tolerance` - Convergence tolerance
/// * `max_iterations` - Maximum number of iterations
/// * `algorithm` - Algorithm name for error messages
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_convergence_parameters<F: Float + FromPrimitive + Debug + PartialOrd>(
    tolerance: Option<F>,
    max_iterations: Option<usize>,
    algorithm: &str,
) -> Result<()> {
    if let Some(tol) = tolerance {
        validate_distance_parameter(tol, "tolerance", Some(F::zero()), None, algorithm)?;
    }

    if let Some(max_iter) = max_iterations {
        validate_integer_parameter(max_iter, "max_iterations", Some(1), None, algorithm)?;
    }

    Ok(())
}

/// Check for duplicate data points
///
/// Identifies if the dataset contains duplicate points, which can cause issues
/// for some clustering algorithms.
///
/// # Arguments
///
/// * `data` - Input data matrix
/// * `tolerance` - Tolerance for considering points as duplicates
///
/// # Returns
///
/// * `Result<Vec<(usize, usize)>>` - List of duplicate point pairs
pub fn check_duplicate_points<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    tolerance: F,
) -> Result<Vec<(usize, usize)>> {
    let n_samples = data.shape()[0];
    let mut duplicates = Vec::new();

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut distance_squared = F::zero();
            for k in 0..data.shape()[1] {
                let diff = data[[i, k]] - data[[j, k]];
                distance_squared = distance_squared + diff * diff;
            }

            if distance_squared <= tolerance * tolerance {
                duplicates.push((i, j));
            }
        }
    }

    Ok(duplicates)
}

/// Validate and suggest appropriate clustering algorithm
///
/// Analyzes the dataset characteristics and suggests the most appropriate
/// clustering algorithm with explanations.
///
/// # Arguments
///
/// * `data` - Input data matrix
/// * `n_clusters` - Desired number of clusters (if known)
///
/// # Returns
///
/// * `Result<String>` - Recommendation message
pub fn suggest_clustering_algorithm<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    n_clusters: Option<usize>,
) -> Result<String> {
    let (n_samples, n_features) = data.dim();

    // Validate data first
    let config = ValidationConfig::default();
    validate_clustering_data(data, &config)?;

    let mut suggestions = Vec::new();

    // Analyze dataset characteristics
    if n_samples < 100 {
        suggestions
            .push("Small dataset: Consider hierarchical clustering for interpretable results");
    } else if n_samples > 10000 {
        suggestions.push("Large dataset: K-means or DBSCAN recommended for efficiency");
    }

    if n_features > 50 {
        suggestions.push(
            "High-dimensional data: Consider spectral clustering or dimensionality reduction",
        );
    }

    // Check for duplicates
    let duplicates = check_duplicate_points(data, F::from_f64(1e-10).unwrap())?;
    if !duplicates.is_empty() {
        suggestions.push("Duplicate points detected: DBSCAN may handle noise well");
    }

    // Algorithm-specific recommendations
    if let Some(k) = n_clusters {
        if k <= 10 {
            suggestions.push(
                "Small number of clusters: K-means with k-means++ initialization recommended",
            );
        } else {
            suggestions.push("Many clusters: Consider hierarchical clustering or DBSCAN");
        }
    } else {
        suggestions.push(
            "Unknown cluster count: DBSCAN or hierarchical clustering with automatic cut-off",
        );
    }

    // Performance considerations
    if n_samples > 5000 && n_features > 20 {
        suggestions.push("Performance consideration: Use parallel implementations when available");
    }

    let recommendation = if suggestions.is_empty() {
        "K-means with k-means++ initialization is a good general-purpose choice".to_string()
    } else {
        format!("Recommendations:\n{}", suggestions.join("\n- "))
    };

    Ok(recommendation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_validate_clustering_data() {
        // Valid data
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let config = ValidationConfig::default();
        assert!(validate_clustering_data(data.view(), &config).is_ok());

        // Too few samples
        let small_data = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(validate_clustering_data(small_data.view(), &config).is_err());

        // Non-finite values
        let invalid_data =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        assert!(validate_clustering_data(invalid_data.view(), &config).is_err());
    }

    #[test]
    fn test_validate_n_clusters() {
        assert!(validate_n_clusters(3, 10, "Test").is_ok());
        assert!(validate_n_clusters(0, 10, "Test").is_err()); // Zero clusters
        assert!(validate_n_clusters(15, 10, "Test").is_err()); // More clusters than samples
    }

    #[test]
    fn test_validate_distance_parameter() {
        assert!(validate_distance_parameter(1.0, "eps", Some(0.0), Some(10.0), "Test").is_ok());
        assert!(validate_distance_parameter(-1.0, "eps", Some(0.0), None, "Test").is_err());
        assert!(validate_distance_parameter(f64::NAN, "eps", None, None, "Test").is_err());
    }

    #[test]
    fn test_validate_sample_weights() {
        let weights = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(validate_sample_weights(weights.view(), 3, "Test").is_ok());

        let negative_weights = Array1::from_vec(vec![1.0, -2.0, 3.0]);
        assert!(validate_sample_weights(negative_weights.view(), 3, "Test").is_err());

        let wrong_size = Array1::from_vec(vec![1.0, 2.0]);
        assert!(validate_sample_weights(wrong_size.view(), 3, "Test").is_err());
    }

    #[test]
    fn test_check_duplicate_points() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let duplicates = check_duplicate_points(data.view(), 1e-10).unwrap();
        assert_eq!(duplicates.len(), 1); // Points 0 and 1 are identical
        assert_eq!(duplicates[0], (0, 1));
    }

    #[test]
    fn test_suggest_clustering_algorithm() {
        let data = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();
        let suggestion = suggest_clustering_algorithm(data.view(), Some(3)).unwrap();
        assert!(!suggestion.is_empty());
        assert!(suggestion.contains("K-means") || suggestion.contains("recommendation"));
    }

    #[test]
    fn test_validation_configs() {
        let kmeans_config = ValidationConfig::for_kmeans();
        assert_eq!(kmeans_config.min_samples, 1);

        let hierarchical_config = ValidationConfig::for_hierarchical();
        assert_eq!(hierarchical_config.min_samples, 2);

        let dbscan_config = ValidationConfig::for_dbscan();
        assert_eq!(dbscan_config.min_samples, 2);
    }
}
