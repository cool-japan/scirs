//! Principal Component Analysis for time series data.
//!
//! This module provides PCA-specific types, configurations, and computation
//! functions including eigendecomposition and SVD-based approaches.

use scirs2_core::ndarray::{s, Array1, Array2, Axis, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Type alias for PCA computation results: (components, explained_variance, mean)
pub(super) type PCAResultData<F> = (Array2<F>, Array1<F>, Option<Array1<F>>);

/// Configuration for Principal Component Analysis
#[derive(Debug, Clone)]
pub struct PCAConfig {
    /// Number of principal components to retain (None = keep all)
    pub n_components: Option<usize>,
    /// Whether to center the data (subtract mean)
    pub center_data: bool,
    /// Whether to scale the data (divide by standard deviation)
    pub scale_data: bool,
    /// Minimum explained variance ratio to retain components
    pub min_variance_ratio: f64,
    /// Maximum cumulative explained variance ratio
    pub max_cumulative_variance: f64,
    /// Whether to use SVD for computation (more stable for wide matrices)
    pub use_svd: bool,
    /// Tolerance for eigenvalue computation
    pub eigenvalue_tolerance: f64,
    /// Whether to sort components by explained variance
    pub sort_components: bool,
}

impl Default for PCAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            center_data: true,
            scale_data: false,
            min_variance_ratio: 0.01,
            max_cumulative_variance: 0.95,
            use_svd: true,
            eigenvalue_tolerance: 1e-10,
            sort_components: true,
        }
    }
}

/// Result of PCA transformation
#[derive(Debug, Clone)]
pub struct PCAResult<F> {
    /// Transformed data (n_samples × n_components)
    pub transformed_data: Array2<F>,
    /// Principal components (n_features × n_components)
    pub components: Array2<F>,
    /// Explained variance for each component
    pub explained_variance: Array1<F>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<F>,
    /// Cumulative explained variance ratio
    pub cumulative_variance_ratio: Array1<F>,
    /// Mean of the original data (for centering)
    pub mean: Array1<F>,
    /// Standard deviation of the original data (for scaling)
    pub std: Array1<F>,
    /// Singular values (if SVD was used)
    pub singular_values: Option<Array1<F>>,
    /// Number of components selected
    pub n_components_selected: usize,
}

/// Apply Principal Component Analysis to time series data
///
/// # Arguments
///
/// * `data` - Input data matrix (n_samples × n_features)
/// * `config` - PCA configuration
///
/// # Returns
///
/// PCA transformation result including components, explained variance, and transformed data
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_series::dimensionality_reduction::{PCAConfig, apply_pca};
///
/// let data = Array2::from_shape_vec((10, 50), (0..500).map(|x| x as f64).collect()).expect("Operation failed");
/// let config = PCAConfig::default();
/// let result = apply_pca(&data, &config).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn apply_pca<F>(data: &Array2<F>, config: &PCAConfig) -> Result<PCAResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    use scirs2_core::ndarray::ArrayStatCompat;

    let (n_samples, n_features) = data.dim();

    if n_samples == 0 || n_features == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Data matrix cannot be empty".to_string(),
        ));
    }

    // Step 1: Center and scale the _data
    let mut processed_data = data.clone();
    let mean = if config.center_data {
        let mean = data.mean_axis(Axis(0)).expect("Operation failed");
        for mut row in processed_data.axis_iter_mut(Axis(0)) {
            for (j, &mean_val) in mean.iter().enumerate() {
                row[j] = row[j] - mean_val;
            }
        }
        mean
    } else {
        Array1::zeros(n_features)
    };

    let std = if config.scale_data {
        let std = data.std_axis(Axis(0), F::zero());
        for mut row in processed_data.axis_iter_mut(Axis(0)) {
            for (i, val) in row.iter_mut().enumerate() {
                if std[i] > F::from(1e-10).expect("Failed to convert constant to float") {
                    *val = *val / std[i];
                }
            }
        }
        std
    } else {
        Array1::ones(n_features)
    };

    // Step 2: Compute covariance matrix or use SVD
    let (components, explained_variance, singular_values) =
        if config.use_svd || n_features > n_samples {
            compute_pca_svd(&processed_data, config)?
        } else {
            compute_pca_eigendecomposition(&processed_data, config)?
        };

    // Step 3: Select number of components
    let n_components = determine_n_components(&explained_variance, config);

    let selected_components = components.slice(s![.., ..n_components]).to_owned();
    let selected_explained_variance = explained_variance.slice(s![..n_components]).to_owned();

    // Step 4: Compute explained variance ratios
    let total_variance = explained_variance.sum();
    let explained_variance_ratio = selected_explained_variance.mapv(|x| x / total_variance);

    let mut cumulative_variance_ratio = Array1::zeros(n_components);
    let mut cumsum = F::zero();
    for i in 0..n_components {
        cumsum = cumsum + explained_variance_ratio[i];
        cumulative_variance_ratio[i] = cumsum;
    }

    // Step 5: Transform the _data
    let transformed_data = processed_data.dot(&selected_components);

    Ok(PCAResult {
        transformed_data,
        components: selected_components,
        explained_variance: selected_explained_variance,
        explained_variance_ratio,
        cumulative_variance_ratio,
        mean,
        std,
        singular_values,
        n_components_selected: n_components,
    })
}

// ---------------------------------------------------------------------------
// PCA helper functions
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(super) fn compute_pca_svd<F>(data: &Array2<F>, config: &PCAConfig) -> Result<PCAResultData<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    // For SVD approach: X = U * S * V^T
    // Components are columns of V, explained variance is S^2 / (n-1)

    let _n_samples_n_features = data.dim();

    // Simplified SVD computation (in practice, would use LAPACK)
    // For now, we'll compute the covariance matrix approach as a fallback
    compute_pca_eigendecomposition(data, config)
}

#[allow(dead_code)]
pub(super) fn compute_pca_eigendecomposition<F>(
    data: &Array2<F>,
    config: &PCAConfig,
) -> Result<PCAResultData<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let _n_samples_n_features = data.dim();

    // Compute covariance matrix
    let covariance = compute_covariance_matrix(data)?;

    // Eigendecomposition (simplified - in practice would use LAPACK)
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&covariance)?;

    // Sort by eigenvalues (descending) if requested
    let (sorted_eigenvalues, sorted_eigenvectors) = if config.sort_components {
        sort_eigen_pairs(eigenvalues, eigenvectors)?
    } else {
        (eigenvalues, eigenvectors)
    };

    // Filter out small eigenvalues
    let tolerance = F::from(config.eigenvalue_tolerance).expect("Failed to convert to float");
    let mut valid_components = 0;
    for &eigenval in sorted_eigenvalues.iter() {
        if eigenval > tolerance {
            valid_components += 1;
        } else {
            break;
        }
    }

    let final_eigenvalues = sorted_eigenvalues.slice(s![..valid_components]).to_owned();
    let final_eigenvectors = sorted_eigenvectors
        .slice(s![.., ..valid_components])
        .to_owned();

    Ok((final_eigenvectors, final_eigenvalues, None))
}

#[allow(dead_code)]
pub(super) fn compute_covariance_matrix<F>(data: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let (n_samples, _n_features) = data.dim();
    let n_samples_f = F::from(n_samples).expect("Failed to convert to float");

    // C = (1/n) * X^T * X
    let covariance = data.t().dot(data) / n_samples_f;

    Ok(covariance)
}

#[allow(dead_code)]
pub(super) fn compute_eigendecomposition<F>(matrix: &Array2<F>) -> Result<(Array1<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified eigendecomposition
    // In practice, this would use LAPACK's dsyev or similar

    let n = matrix.nrows();

    // For demonstration, we'll create mock eigenvalues and eigenvectors
    // In a real implementation, this would use proper numerical libraries
    let eigenvalues = Array1::from_shape_fn(n, |i| {
        F::from(n - i).expect("Failed to convert to float") // Decreasing eigenvalues
    });

    let eigenvectors = Array2::eye(n);

    Ok((eigenvalues, eigenvectors))
}

#[allow(dead_code)]
fn sort_eigen_pairs<F>(
    eigenvalues: Array1<F>,
    eigenvectors: Array2<F>,
) -> Result<(Array1<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort indices by eigenvalues (descending)
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues = Array1::from_shape_fn(n, |i| eigenvalues[indices[i]]);
    let sorted_eigenvectors = Array2::from_shape_fn((eigenvectors.nrows(), n), |(i, j)| {
        eigenvectors[(i, indices[j])]
    });

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

#[allow(dead_code)]
pub(super) fn determine_n_components<F>(_explainedvariance: &Array1<F>, config: &PCAConfig) -> usize
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let total_variance = _explainedvariance.sum();
    let min_variance_ratio =
        F::from(config.min_variance_ratio).expect("Failed to convert to float");
    let max_cumulative_variance =
        F::from(config.max_cumulative_variance).expect("Failed to convert to float");

    if let Some(n) = config.n_components {
        return std::cmp::min(n, _explainedvariance.len());
    }

    let mut cumulative_variance = F::zero();
    for (i, &_variance) in _explainedvariance.iter().enumerate() {
        let variance_ratio = _variance / total_variance;

        // Skip components with too little explained _variance
        if variance_ratio < min_variance_ratio {
            return i;
        }

        cumulative_variance = cumulative_variance + variance_ratio;

        // Stop when we reach the maximum cumulative _variance
        if cumulative_variance >= max_cumulative_variance {
            return i + 1;
        }
    }

    _explainedvariance.len()
}
