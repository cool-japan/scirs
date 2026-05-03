//! Functional Principal Component Analysis for time series data.
//!
//! This module provides functional PCA types, configurations, and computation
//! functions including basis function creation (B-splines, Fourier, polynomial,
//! wavelet) and smoothness regularization.

use scirs2_core::ndarray::{s, Array1, Array2, Axis, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::pca::{compute_covariance_matrix, compute_eigendecomposition};
use crate::error::{Result, TimeSeriesError};

/// Configuration for Functional Principal Component Analysis
#[derive(Debug, Clone)]
pub struct FunctionalPCAConfig {
    /// Number of functional principal components
    pub n_components: Option<usize>,
    /// Smoothing parameter for functional data
    pub smoothing_parameter: f64,
    /// Number of basis functions (e.g., B-splines)
    pub nbasis_functions: usize,
    /// Type of basis functions
    pub basis_type: BasisType,
    /// Whether to center functional data
    pub center_functions: bool,
    /// Whether to estimate derivatives
    pub estimate_derivatives: bool,
    /// Order of derivatives to estimate (0 = function values only)
    pub derivative_order: usize,
    /// Regularization parameter for smoothness
    pub regularization_parameter: f64,
}

impl Default for FunctionalPCAConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            smoothing_parameter: 0.01,
            nbasis_functions: 20,
            basis_type: BasisType::BSpline,
            center_functions: true,
            estimate_derivatives: false,
            derivative_order: 0,
            regularization_parameter: 1e-4,
        }
    }
}

/// Types of basis functions for functional PCA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasisType {
    /// B-spline basis functions
    BSpline,
    /// Fourier basis functions
    Fourier,
    /// Polynomial basis functions
    Polynomial,
    /// Wavelet basis functions
    Wavelet,
}

/// Result of Functional PCA
#[derive(Debug, Clone)]
pub struct FunctionalPCAResult<F> {
    /// Functional principal components (basis coefficients)
    pub functional_components: Array2<F>,
    /// Explained variance for each functional component
    pub explained_variance: Array1<F>,
    /// Explained variance ratio for each functional component
    pub explained_variance_ratio: Array1<F>,
    /// Mean function (coefficients)
    pub mean_function: Array1<F>,
    /// Basis functions evaluation points
    pub basis_evaluation: Array2<F>,
    /// Scores for each observation on functional components
    pub scores: Array2<F>,
    /// Reconstruction of original functions
    pub reconstructed_functions: Array2<F>,
    /// Smoothness measure for each component
    pub smoothness_measures: Array1<F>,
}

/// Apply Functional Principal Component Analysis to time series data
///
/// # Arguments
///
/// * `functional_data` - Input functional data matrix (n_functions × n_evaluation_points)
/// * `config` - Functional PCA configuration
///
/// # Returns
///
/// Functional PCA result including functional components and scores
#[allow(dead_code)]
pub fn apply_functional_pca<F>(
    functional_data: &Array2<F>,
    config: &FunctionalPCAConfig,
) -> Result<FunctionalPCAResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let (n_functions, n_points) = functional_data.dim();

    if n_functions == 0 || n_points == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Functional _data matrix cannot be empty".to_string(),
        ));
    }

    // Step 1: Create basis functions
    let basis_evaluation = createbasis_functions(n_points, config)?;
    let nbasis = basis_evaluation.ncols();

    // Step 2: Project _data onto basis functions
    let basis_coefficients = project_ontobasis(functional_data, &basis_evaluation)?;

    // Step 3: Center the coefficients if requested
    let centered_coefficients = if config.center_functions {
        let mean_function = basis_coefficients
            .mean_axis(Axis(0))
            .expect("Operation failed");
        let mut centered = basis_coefficients.clone();
        for mut row in centered.axis_iter_mut(Axis(0)) {
            for (j, &mean_val) in mean_function.iter().enumerate() {
                row[j] = row[j] - mean_val;
            }
        }
        (centered, mean_function)
    } else {
        let mean_function = Array1::zeros(nbasis);
        (basis_coefficients, mean_function)
    };

    // Step 4: Apply regularization for smoothness
    let regularized_covariance = apply_smoothness_regularization(
        &centered_coefficients.0,
        config.regularization_parameter,
        &basis_evaluation,
    )?;

    // Step 5: Eigendecomposition of regularized covariance
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&regularized_covariance)?;

    // Step 6: Select number of components
    let n_components = config
        .n_components
        .unwrap_or(std::cmp::min(n_functions.saturating_sub(1), nbasis));
    let n_components = std::cmp::min(n_components, eigenvalues.len());

    // Step 7: Extract functional components and compute scores
    let functional_components = eigenvectors.slice(s![.., ..n_components]).to_owned();
    let explained_variance = eigenvalues.slice(s![..n_components]).to_owned();

    let total_variance = eigenvalues.sum();
    let explained_variance_ratio = &explained_variance / total_variance;

    // Compute scores (projections onto functional components)
    let scores = centered_coefficients.0.dot(&functional_components);

    // Step 8: Reconstruct functions for validation
    let reconstructed_coefficients = scores.dot(&functional_components.t());
    let reconstructed_functions = reconstructed_coefficients.dot(&basis_evaluation.t());

    // Step 9: Compute smoothness measures
    let smoothness_measures =
        compute_smoothness_measures(&functional_components, &basis_evaluation)?;

    Ok(FunctionalPCAResult {
        functional_components,
        explained_variance,
        explained_variance_ratio,
        mean_function: centered_coefficients.1,
        basis_evaluation,
        scores,
        reconstructed_functions,
        smoothness_measures,
    })
}

// ---------------------------------------------------------------------------
// Functional PCA helper functions
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn createbasis_functions<F>(_npoints: usize, config: &FunctionalPCAConfig) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    match config.basis_type {
        BasisType::BSpline => create_bsplinebasis(_npoints, config.nbasis_functions),
        BasisType::Fourier => create_fourierbasis(_npoints, config.nbasis_functions),
        BasisType::Polynomial => create_polynomialbasis(_npoints, config.nbasis_functions),
        BasisType::Wavelet => create_waveletbasis(_npoints, config.nbasis_functions),
    }
}

#[allow(dead_code)]
fn create_bsplinebasis<F>(_n_points: usize, nbasis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified B-spline basis creation
    // In practice, this would use proper spline libraries

    let mut basis = Array2::zeros((_n_points, nbasis));

    for j in 0..nbasis {
        for i in 0.._n_points {
            let t = F::from(i).expect("Failed to convert to float")
                / F::from(_n_points - 1).expect("Failed to convert to float");
            let center = F::from(j).expect("Failed to convert to float")
                / F::from(nbasis - 1).expect("Failed to convert to float");
            let width = F::one() / F::from(nbasis).expect("Failed to convert to float");

            // Simple Gaussian-like basis function
            let diff = (t - center) / width;
            basis[(i, j)] = (-diff * diff).exp();
        }
    }

    Ok(basis)
}

#[allow(dead_code)]
fn create_fourierbasis<F>(_n_points: usize, nbasis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut basis = Array2::zeros((_n_points, nbasis));
    let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");

    for j in 0..nbasis {
        for i in 0.._n_points {
            let t = F::from(i).expect("Failed to convert to float")
                / F::from(_n_points - 1).expect("Failed to convert to float");
            let freq = F::from(j + 1).expect("Failed to convert to float");

            if j % 2 == 0 {
                // Cosine terms
                basis[(i, j)] =
                    (F::from(2.0).expect("Failed to convert constant to float") * pi * freq * t)
                        .cos();
            } else {
                // Sine terms
                basis[(i, j)] =
                    (F::from(2.0).expect("Failed to convert constant to float") * pi * freq * t)
                        .sin();
            }
        }
    }

    Ok(basis)
}

#[allow(dead_code)]
fn create_polynomialbasis<F>(_n_points: usize, nbasis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut basis = Array2::zeros((_n_points, nbasis));

    for j in 0..nbasis {
        for i in 0.._n_points {
            let t = F::from(i).expect("Failed to convert to float")
                / F::from(_n_points - 1).expect("Failed to convert to float");

            // Polynomial basis: t^j
            basis[(i, j)] = t.powf(F::from(j).expect("Failed to convert to float"));
        }
    }

    Ok(basis)
}

#[allow(dead_code)]
fn create_waveletbasis<F>(n_points: usize, nbasis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified wavelet basis (Haar wavelets)
    let mut basis = Array2::zeros((n_points, nbasis));

    // First basis function is constant
    for i in 0..n_points {
        basis[(i, 0)] = F::one()
            / F::from(n_points)
                .expect("Failed to convert to float")
                .sqrt();
    }

    // Additional basis functions are Haar wavelets at different scales
    for j in 1..nbasis {
        let scale = 1 << (j / 2); // Powers of 2
        let shift = j % scale;

        for i in 0..n_points {
            let t = F::from(i).expect("Failed to convert to float")
                / F::from(n_points - 1).expect("Failed to convert to float");
            let scaled_t = t * F::from(scale).expect("Failed to convert to float")
                - F::from(shift).expect("Failed to convert to float");

            if scaled_t >= F::zero() && scaled_t < F::one() {
                if scaled_t < F::from(0.5).expect("Failed to convert constant to float") {
                    basis[(i, j)] = F::one();
                } else {
                    basis[(i, j)] = -F::one();
                }
                basis[(i, j)] =
                    basis[(i, j)] / F::from(scale).expect("Failed to convert to float").sqrt();
            }
        }
    }

    Ok(basis)
}

#[allow(dead_code)]
fn project_ontobasis<F>(
    functional_data: &Array2<F>,
    basis_evaluation: &Array2<F>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Project functional _data onto basis functions
    // Coefficients = Data * Basis (assuming orthonormal basis)

    let coefficients = functional_data.dot(basis_evaluation);
    Ok(coefficients)
}

#[allow(dead_code)]
fn apply_smoothness_regularization<F>(
    coefficients: &Array2<F>,
    lambda: f64,
    _basis_evaluation: &Array2<F>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    // Apply smoothness penalty to covariance matrix
    // This is a simplified version - would compute roughness penalty matrix in practice

    let covariance = compute_covariance_matrix(coefficients)?;
    let lambda_f = F::from(lambda).expect("Failed to convert to float");
    let identity = Array2::eye(covariance.ncols());

    // Regularized covariance = Cov - lambda * I (simplified)
    let regularized = covariance - identity.mapv(|x: F| x * lambda_f);

    Ok(regularized)
}

#[allow(dead_code)]
fn compute_smoothness_measures<F>(
    components: &Array2<F>,
    _basis_evaluation: &Array2<F>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n_components = components.ncols();
    let mut smoothness = Array1::zeros(n_components);

    // Compute smoothness as second derivative norm (simplified)
    for j in 0..n_components {
        let component = components.column(j);

        // Simplified smoothness measure: sum of squared differences
        let mut roughness = F::zero();
        for i in 1..component.len() {
            let diff = component[i] - component[i - 1];
            roughness = roughness + diff * diff;
        }
        smoothness[j] = roughness;
    }

    Ok(smoothness)
}

// ---------------------------------------------------------------------------
// FunctionalPCA — high-level struct API
// ---------------------------------------------------------------------------

/// High-level functional PCA estimator with a scikit-learn-style API.
///
/// Wraps [`apply_functional_pca`] and exposes `new(n_components)` /
/// `fit_transform(data)` methods that the integration tests expect.
///
/// # Examples
///
/// ```rust
/// use scirs2_series::dimensionality_reduction::FunctionalPCA;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec(
///     (20, 5),
///     (0..100).map(|x| x as f64).collect(),
/// ).expect("shape ok");
///
/// let fpca = FunctionalPCA::new(2);
/// let reduced = fpca.fit_transform(&data).expect("should succeed");
/// assert_eq!(reduced.ncols(), 2);
/// assert_eq!(reduced.nrows(), 20);
/// ```
pub struct FunctionalPCA {
    n_components: usize,
}

impl FunctionalPCA {
    /// Create a new `FunctionalPCA` that retains `n_components` components.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit the model to `data` and return the scores matrix.
    ///
    /// `data` must have shape `(n_observations, n_features)`.
    /// The returned array has shape `(n_observations, n_components)`.
    pub fn fit_transform<F>(&self, data: &Array2<F>) -> crate::error::Result<Array2<F>>
    where
        F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
    {
        let config = FunctionalPCAConfig {
            n_components: Some(self.n_components),
            ..Default::default()
        };
        let result = apply_functional_pca(data, &config)?;
        Ok(result.scores)
    }
}
