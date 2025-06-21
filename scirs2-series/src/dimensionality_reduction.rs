//! Time series dimensionality reduction methods
//!
//! This module provides various dimensionality reduction techniques specifically
//! designed for time series data, including PCA, Functional PCA, Dynamic Time Warping
//! barycenter averaging, and symbolic approximation methods.
//!
//! # Key Features
//!
//! - **Principal Component Analysis (PCA)**: Traditional PCA adapted for time series
//! - **Functional PCA**: PCA for functional time series data
//! - **Dynamic Time Warping (DTW) Barycenter**: Averaging for irregular time series
//! - **Symbolic Approximation**: Discrete representation methods
//! - **Adaptive Methods**: Data-driven dimension selection
//! - **Cross-validation**: Model selection and validation
//!
//! # Example
//!
//! ```rust
//! use ndarray::Array2;
//! use scirs2_series::dimensionality_reduction::{PCAConfig, apply_pca};
//!
//! // Create sample time series data matrix (n_series × n_timepoints)
//! let data = Array2::from_shape_vec((5, 100), (0..500).map(|x| x as f64).collect()).unwrap();
//!
//! // Configure PCA
//! let config = PCAConfig {
//!     n_components: Some(3),
//!     center_data: true,
//!     scale_data: true,
//!     ..Default::default()
//! };
//!
//! // Apply PCA transformation
//! let result = apply_pca(&data, &config).unwrap();
//! println!("Explained variance ratio: {:?}", result.explained_variance_ratio);
//! ```

use ndarray::{s, Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Type alias for PCA computation results: (components, explained_variance, mean)
type PCAResultData<F> = (Array2<F>, Array1<F>, Option<Array1<F>>);

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

/// Configuration for Functional Principal Component Analysis
#[derive(Debug, Clone)]
pub struct FunctionalPCAConfig {
    /// Number of functional principal components
    pub n_components: Option<usize>,
    /// Smoothing parameter for functional data
    pub smoothing_parameter: f64,
    /// Number of basis functions (e.g., B-splines)
    pub n_basis_functions: usize,
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
            n_basis_functions: 20,
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

/// Configuration for Dynamic Time Warping barycenter averaging
#[derive(Debug, Clone)]
pub struct DTWBarycenterConfig {
    /// Maximum number of iterations for barycenter computation
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Initialization method for barycenter
    pub initialization_method: BarycenterInit,
    /// Weights for each time series (None = equal weights)
    pub weights: Option<Array1<f64>>,
    /// Window constraint for DTW (None = no constraint)
    pub window_constraint: Option<usize>,
    /// Distance metric for DTW
    pub distance_metric: DTWDistance,
    /// Whether to use approximation methods for speed
    pub use_approximation: bool,
}

impl Default for DTWBarycenterConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            initialization_method: BarycenterInit::Random,
            weights: None,
            window_constraint: None,
            distance_metric: DTWDistance::Euclidean,
            use_approximation: false,
        }
    }
}

/// Initialization methods for barycenter computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarycenterInit {
    /// Random initialization
    Random,
    /// Use first time series as initialization
    First,
    /// Use medoid (most central) time series
    Medoid,
    /// Use mean of all time series (ignoring alignment)
    Mean,
}

/// Distance metrics for DTW
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DTWDistance {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Squared Euclidean distance
    SquaredEuclidean,
}

/// Configuration for symbolic approximation
#[derive(Debug, Clone)]
pub struct SymbolicApproximationConfig {
    /// Approximation method
    pub method: SymbolicMethod,
    /// Number of symbols in the alphabet
    pub alphabet_size: usize,
    /// Window size for segmentation
    pub window_size: usize,
    /// Number of segments for PAA
    pub n_segments: usize,
    /// Whether to normalize data before approximation
    pub normalize_data: bool,
    /// Breakpoints for SAX (None = automatic)
    pub breakpoints: Option<Array1<f64>>,
    /// Distance metric for symbolic sequences
    pub distance_metric: SymbolicDistance,
}

impl Default for SymbolicApproximationConfig {
    fn default() -> Self {
        Self {
            method: SymbolicMethod::SAX,
            alphabet_size: 8,
            window_size: 16,
            n_segments: 10,
            normalize_data: true,
            breakpoints: None,
            distance_metric: SymbolicDistance::MINDIST,
        }
    }
}

/// Symbolic approximation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolicMethod {
    /// Symbolic Aggregate approXimation (SAX)
    SAX,
    /// Adaptive Piecewise Constant Approximation (APCA)
    APCA,
    /// Piecewise Linear Approximation (PLA)
    PLA,
    /// Persist (1-dimensional representation)
    Persist,
}

/// Distance metrics for symbolic sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolicDistance {
    /// MINDIST lower bound
    MINDIST,
    /// Hamming distance
    Hamming,
    /// Edit distance
    Edit,
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

/// Result of DTW barycenter averaging
#[derive(Debug, Clone)]
pub struct DTWBarycenterResult<F> {
    /// Computed barycenter time series
    pub barycenter: Array1<F>,
    /// Distances from each series to barycenter
    pub distances: Array1<F>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final convergence error
    pub convergence_error: F,
    /// Alignment paths for each series to barycenter
    pub alignment_paths: Vec<Vec<(usize, usize)>>,
    /// Warping costs for each series
    pub warping_costs: Array1<F>,
}

/// Result of symbolic approximation
#[derive(Debug, Clone)]
pub struct SymbolicApproximationResult {
    /// Symbolic representation
    pub symbolic_sequence: Vec<char>,
    /// Breakpoints used for discretization
    pub breakpoints: Array1<f64>,
    /// Piecewise Aggregate Approximation values
    pub paa_values: Array1<f64>,
    /// Reconstruction error
    pub reconstruction_error: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Distance matrix between segments (if applicable)
    pub distance_matrix: Option<Array2<f64>>,
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
/// use ndarray::Array2;
/// use scirs2_series::dimensionality_reduction::{PCAConfig, apply_pca};
///
/// let data = Array2::from_shape_vec((10, 50), (0..500).map(|x| x as f64).collect()).unwrap();
/// let config = PCAConfig::default();
/// let result = apply_pca(&data, &config).unwrap();
/// ```
pub fn apply_pca<F>(data: &Array2<F>, config: &PCAConfig) -> Result<PCAResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let (n_samples, n_features) = data.dim();

    if n_samples == 0 || n_features == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Data matrix cannot be empty".to_string(),
        ));
    }

    // Step 1: Center and scale the data
    let mut processed_data = data.clone();
    let mean = if config.center_data {
        let mean = data.mean_axis(Axis(0)).unwrap();
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
                if std[i] > F::from(1e-10).unwrap() {
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

    // Step 5: Transform the data
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
            "Functional data matrix cannot be empty".to_string(),
        ));
    }

    // Step 1: Create basis functions
    let basis_evaluation = create_basis_functions(n_points, config)?;
    let n_basis = basis_evaluation.ncols();

    // Step 2: Project data onto basis functions
    let basis_coefficients = project_onto_basis(functional_data, &basis_evaluation)?;

    // Step 3: Center the coefficients if requested
    let centered_coefficients = if config.center_functions {
        let mean_function = basis_coefficients.mean_axis(Axis(0)).unwrap();
        let mut centered = basis_coefficients.clone();
        for mut row in centered.axis_iter_mut(Axis(0)) {
            for (j, &mean_val) in mean_function.iter().enumerate() {
                row[j] = row[j] - mean_val;
            }
        }
        (centered, mean_function)
    } else {
        let mean_function = Array1::zeros(n_basis);
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
        .unwrap_or(std::cmp::min(n_functions.saturating_sub(1), n_basis));
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

/// Compute DTW barycenter of multiple time series
///
/// # Arguments
///
/// * `time_series` - Vector of time series to average
/// * `config` - DTW barycenter configuration
///
/// # Returns
///
/// DTW barycenter result including the computed barycenter and alignment information
pub fn compute_dtw_barycenter<F>(
    time_series: &[Array1<F>],
    config: &DTWBarycenterConfig,
) -> Result<DTWBarycenterResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    if time_series.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot compute barycenter of empty time series collection".to_string(),
        ));
    }

    let n_series = time_series.len();

    // Initialize weights
    let weights = config
        .weights
        .clone()
        .unwrap_or_else(|| Array1::from_elem(n_series, 1.0 / n_series as f64));

    if weights.len() != n_series {
        return Err(TimeSeriesError::InvalidInput(
            "Weights length must match number of time series".to_string(),
        ));
    }

    // Initialize barycenter
    let mut barycenter = initialize_barycenter(time_series, &config.initialization_method)?;
    let mut prev_barycenter = barycenter.clone();

    let mut convergence_error = F::infinity();
    let mut iterations = 0;

    let mut alignment_paths = Vec::new();
    let mut warping_costs = Array1::zeros(n_series);

    // Iterative barycenter computation
    while iterations < config.max_iterations
        && convergence_error > F::from(config.convergence_tolerance).unwrap()
    {
        alignment_paths.clear();

        // Compute alignments for all series to current barycenter
        for (i, series) in time_series.iter().enumerate() {
            let (cost, path) = compute_dtw_alignment(&barycenter, series, config)?;
            alignment_paths.push(path);
            warping_costs[i] = cost;
        }

        // Update barycenter based on alignments
        barycenter = update_barycenter_from_alignments(time_series, &alignment_paths, &weights)?;

        // Check convergence
        convergence_error = compute_barycenter_difference(&barycenter, &prev_barycenter);
        prev_barycenter = barycenter.clone();
        iterations += 1;
    }

    // Compute final distances
    let mut distances = Array1::zeros(n_series);
    for (i, series) in time_series.iter().enumerate() {
        let (distance, _) = compute_dtw_alignment(&barycenter, series, config)?;
        distances[i] = distance;
    }

    Ok(DTWBarycenterResult {
        barycenter,
        distances,
        iterations,
        convergence_error,
        alignment_paths,
        warping_costs,
    })
}

/// Apply symbolic approximation to time series
///
/// # Arguments
///
/// * `time_series` - Input time series
/// * `config` - Symbolic approximation configuration
///
/// # Returns
///
/// Symbolic approximation result including symbolic sequence and reconstruction information
pub fn apply_symbolic_approximation(
    time_series: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    if time_series.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    match config.method {
        SymbolicMethod::SAX => apply_sax(time_series, config),
        SymbolicMethod::APCA => apply_apca(time_series, config),
        SymbolicMethod::PLA => apply_pla(time_series, config),
        SymbolicMethod::Persist => apply_persist(time_series, config),
    }
}

// Helper functions for PCA computation

fn compute_pca_svd<F>(data: &Array2<F>, config: &PCAConfig) -> Result<PCAResultData<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    // For SVD approach: X = U * S * V^T
    // Components are columns of V, explained variance is S^2 / (n-1)

    let (_n_samples, _n_features) = data.dim();

    // Simplified SVD computation (in practice, would use LAPACK)
    // For now, we'll compute the covariance matrix approach as a fallback
    compute_pca_eigendecomposition(data, config)
}

fn compute_pca_eigendecomposition<F>(
    data: &Array2<F>,
    config: &PCAConfig,
) -> Result<PCAResultData<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let (_n_samples, _n_features) = data.dim();

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
    let tolerance = F::from(config.eigenvalue_tolerance).unwrap();
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

fn compute_covariance_matrix<F>(data: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ScalarOperand + 'static,
{
    let (n_samples, _n_features) = data.dim();
    let n_samples_f = F::from(n_samples).unwrap();

    // C = (1/n) * X^T * X
    let covariance = data.t().dot(data) / n_samples_f;

    Ok(covariance)
}

fn compute_eigendecomposition<F>(matrix: &Array2<F>) -> Result<(Array1<F>, Array2<F>)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified eigendecomposition
    // In practice, this would use LAPACK's dsyev or similar

    let n = matrix.nrows();

    // For demonstration, we'll create mock eigenvalues and eigenvectors
    // In a real implementation, this would use proper numerical libraries
    let eigenvalues = Array1::from_shape_fn(n, |i| {
        F::from(n - i).unwrap() // Decreasing eigenvalues
    });

    let eigenvectors = Array2::eye(n);

    Ok((eigenvalues, eigenvectors))
}

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

fn determine_n_components<F>(explained_variance: &Array1<F>, config: &PCAConfig) -> usize
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let total_variance = explained_variance.sum();
    let min_variance_ratio = F::from(config.min_variance_ratio).unwrap();
    let max_cumulative_variance = F::from(config.max_cumulative_variance).unwrap();

    if let Some(n) = config.n_components {
        return std::cmp::min(n, explained_variance.len());
    }

    let mut cumulative_variance = F::zero();
    for (i, &variance) in explained_variance.iter().enumerate() {
        let variance_ratio = variance / total_variance;

        // Skip components with too little explained variance
        if variance_ratio < min_variance_ratio {
            return i;
        }

        cumulative_variance = cumulative_variance + variance_ratio;

        // Stop when we reach the maximum cumulative variance
        if cumulative_variance >= max_cumulative_variance {
            return i + 1;
        }
    }

    explained_variance.len()
}

// Helper functions for Functional PCA

fn create_basis_functions<F>(n_points: usize, config: &FunctionalPCAConfig) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    match config.basis_type {
        BasisType::BSpline => create_bspline_basis(n_points, config.n_basis_functions),
        BasisType::Fourier => create_fourier_basis(n_points, config.n_basis_functions),
        BasisType::Polynomial => create_polynomial_basis(n_points, config.n_basis_functions),
        BasisType::Wavelet => create_wavelet_basis(n_points, config.n_basis_functions),
    }
}

fn create_bspline_basis<F>(n_points: usize, n_basis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified B-spline basis creation
    // In practice, this would use proper spline libraries

    let mut basis = Array2::zeros((n_points, n_basis));

    for j in 0..n_basis {
        for i in 0..n_points {
            let t = F::from(i).unwrap() / F::from(n_points - 1).unwrap();
            let center = F::from(j).unwrap() / F::from(n_basis - 1).unwrap();
            let width = F::one() / F::from(n_basis).unwrap();

            // Simple Gaussian-like basis function
            let diff = (t - center) / width;
            basis[(i, j)] = (-diff * diff).exp();
        }
    }

    Ok(basis)
}

fn create_fourier_basis<F>(n_points: usize, n_basis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut basis = Array2::zeros((n_points, n_basis));
    let pi = F::from(std::f64::consts::PI).unwrap();

    for j in 0..n_basis {
        for i in 0..n_points {
            let t = F::from(i).unwrap() / F::from(n_points - 1).unwrap();
            let freq = F::from(j + 1).unwrap();

            if j % 2 == 0 {
                // Cosine terms
                basis[(i, j)] = (F::from(2.0).unwrap() * pi * freq * t).cos();
            } else {
                // Sine terms
                basis[(i, j)] = (F::from(2.0).unwrap() * pi * freq * t).sin();
            }
        }
    }

    Ok(basis)
}

fn create_polynomial_basis<F>(n_points: usize, n_basis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut basis = Array2::zeros((n_points, n_basis));

    for j in 0..n_basis {
        for i in 0..n_points {
            let t = F::from(i).unwrap() / F::from(n_points - 1).unwrap();

            // Polynomial basis: t^j
            basis[(i, j)] = t.powf(F::from(j).unwrap());
        }
    }

    Ok(basis)
}

fn create_wavelet_basis<F>(n_points: usize, n_basis: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simplified wavelet basis (Haar wavelets)
    let mut basis = Array2::zeros((n_points, n_basis));

    // First basis function is constant
    for i in 0..n_points {
        basis[(i, 0)] = F::one() / F::from(n_points).unwrap().sqrt();
    }

    // Additional basis functions are Haar wavelets at different scales
    for j in 1..n_basis {
        let scale = 1 << (j / 2); // Powers of 2
        let shift = j % scale;

        for i in 0..n_points {
            let t = F::from(i).unwrap() / F::from(n_points - 1).unwrap();
            let scaled_t = t * F::from(scale).unwrap() - F::from(shift).unwrap();

            if scaled_t >= F::zero() && scaled_t < F::one() {
                if scaled_t < F::from(0.5).unwrap() {
                    basis[(i, j)] = F::one();
                } else {
                    basis[(i, j)] = -F::one();
                }
                basis[(i, j)] = basis[(i, j)] / F::from(scale).unwrap().sqrt();
            }
        }
    }

    Ok(basis)
}

fn project_onto_basis<F>(
    functional_data: &Array2<F>,
    basis_evaluation: &Array2<F>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Project functional data onto basis functions
    // Coefficients = Data * Basis (assuming orthonormal basis)

    let coefficients = functional_data.dot(basis_evaluation);
    Ok(coefficients)
}

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
    let lambda_f = F::from(lambda).unwrap();
    let identity = Array2::eye(covariance.ncols());

    // Regularized covariance = Cov - lambda * I (simplified)
    let regularized = covariance - identity.mapv(|x: F| x * lambda_f);

    Ok(regularized)
}

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
        for i in 1..component.len() - 1 {
            let second_diff =
                component[i + 1] - F::from(2.0).unwrap() * component[i] + component[i - 1];
            roughness = roughness + second_diff * second_diff;
        }

        smoothness[j] = F::one() / (F::one() + roughness);
    }

    Ok(smoothness)
}

// Helper functions for DTW barycenter

fn initialize_barycenter<F>(time_series: &[Array1<F>], method: &BarycenterInit) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    match method {
        BarycenterInit::Random => {
            let median_length = time_series.len() / 2;
            let length = time_series[median_length].len();
            Ok(Array1::zeros(length))
        }
        BarycenterInit::First => Ok(time_series[0].clone()),
        BarycenterInit::Medoid => compute_medoid(time_series),
        BarycenterInit::Mean => compute_mean_series(time_series),
    }
}

fn compute_medoid<F>(time_series: &[Array1<F>]) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n = time_series.len();
    let mut min_total_distance = F::infinity();
    let mut medoid_idx = 0;

    for i in 0..n {
        let mut total_distance = F::zero();
        for j in 0..n {
            if i != j {
                let distance = compute_euclidean_distance(&time_series[i], &time_series[j]);
                total_distance = total_distance + distance;
            }
        }

        if total_distance < min_total_distance {
            min_total_distance = total_distance;
            medoid_idx = i;
        }
    }

    Ok(time_series[medoid_idx].clone())
}

fn compute_mean_series<F>(time_series: &[Array1<F>]) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simple mean ignoring alignment issues
    let max_length = time_series.iter().map(|ts| ts.len()).max().unwrap_or(0);
    let mut mean_series = Array1::zeros(max_length);
    let mut counts = Array1::zeros(max_length);

    for ts in time_series {
        for (i, &val) in ts.iter().enumerate() {
            mean_series[i] = mean_series[i] + val;
            counts[i] = counts[i] + F::one();
        }
    }

    for i in 0..max_length {
        if counts[i] > F::zero() {
            mean_series[i] = mean_series[i] / counts[i];
        }
    }

    Ok(mean_series)
}

fn compute_dtw_alignment<F>(
    series1: &Array1<F>,
    series2: &Array1<F>,
    config: &DTWBarycenterConfig,
) -> Result<(F, Vec<(usize, usize)>)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n1 = series1.len();
    let n2 = series2.len();

    // Initialize DTW matrix
    let mut dtw_matrix = Array2::from_elem((n1 + 1, n2 + 1), F::infinity());
    dtw_matrix[(0, 0)] = F::zero();

    // Fill DTW matrix
    for i in 1..=n1 {
        for j in 1..=n2 {
            // Check window constraint
            if let Some(window) = config.window_constraint {
                let _window_f = window as f64;
                let ratio = n1 as f64 / n2 as f64;
                let expected_j = (i as f64 / ratio) as usize;
                if j.abs_diff(expected_j) > window {
                    continue;
                }
            }

            let cost =
                compute_point_distance(series1[i - 1], series2[j - 1], &config.distance_metric);
            let min_prev = dtw_matrix[(i - 1, j)]
                .min(dtw_matrix[(i, j - 1)])
                .min(dtw_matrix[(i - 1, j - 1)]);

            dtw_matrix[(i, j)] = cost + min_prev;
        }
    }

    let total_cost = dtw_matrix[(n1, n2)];

    // Backtrack to find optimal path
    let path = backtrack_dtw_path(&dtw_matrix, n1, n2);

    Ok((total_cost, path))
}

fn compute_point_distance<F>(point1: F, point2: F, metric: &DTWDistance) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let diff = point1 - point2;

    match metric {
        DTWDistance::Euclidean => diff.abs(),
        DTWDistance::Manhattan => diff.abs(),
        DTWDistance::SquaredEuclidean => diff * diff,
    }
}

fn backtrack_dtw_path<F>(dtw_matrix: &Array2<F>, n1: usize, n2: usize) -> Vec<(usize, usize)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut path = Vec::new();
    let mut i = n1;
    let mut j = n2;

    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));

        // Find minimum of three predecessors
        let diag = dtw_matrix[(i - 1, j - 1)];
        let up = dtw_matrix[(i - 1, j)];
        let left = dtw_matrix[(i, j - 1)];

        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    path.reverse();
    path
}

fn update_barycenter_from_alignments<F>(
    time_series: &[Array1<F>],
    alignment_paths: &[Vec<(usize, usize)>],
    weights: &Array1<f64>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Find the maximum barycenter length needed
    let max_barycenter_length = alignment_paths
        .iter()
        .map(|path| path.iter().map(|(i, _)| *i).max().unwrap_or(0) + 1)
        .max()
        .unwrap_or(0);

    let mut new_barycenter = Array1::zeros(max_barycenter_length);
    let mut counts = Array1::zeros(max_barycenter_length);

    // Accumulate weighted contributions
    for (series_idx, path) in alignment_paths.iter().enumerate() {
        let weight = F::from(weights[series_idx]).unwrap();
        let series = &time_series[series_idx];

        for &(barycenter_idx, series_idx_in_path) in path {
            if barycenter_idx < max_barycenter_length && series_idx_in_path < series.len() {
                new_barycenter[barycenter_idx] =
                    new_barycenter[barycenter_idx] + weight * series[series_idx_in_path];
                counts[barycenter_idx] = counts[barycenter_idx] + weight;
            }
        }
    }

    // Normalize by counts
    for i in 0..max_barycenter_length {
        if counts[i] > F::zero() {
            new_barycenter[i] = new_barycenter[i] / counts[i];
        }
    }

    Ok(new_barycenter)
}

fn compute_barycenter_difference<F>(barycenter1: &Array1<F>, barycenter2: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let min_len = std::cmp::min(barycenter1.len(), barycenter2.len());
    let mut sum_sq_diff = F::zero();

    for i in 0..min_len {
        let diff = barycenter1[i] - barycenter2[i];
        sum_sq_diff = sum_sq_diff + diff * diff;
    }

    sum_sq_diff.sqrt()
}

fn compute_euclidean_distance<F>(series1: &Array1<F>, series2: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let min_len = std::cmp::min(series1.len(), series2.len());
    let mut sum_sq_diff = F::zero();

    for i in 0..min_len {
        let diff = series1[i] - series2[i];
        sum_sq_diff = sum_sq_diff + diff * diff;
    }

    sum_sq_diff.sqrt()
}

// Helper functions for symbolic approximation

fn apply_sax(
    time_series: &Array1<f64>,
    config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Step 1: Normalize data if requested
    let normalized_data = if config.normalize_data {
        normalize_time_series(time_series)?
    } else {
        time_series.clone()
    };

    // Step 2: Piecewise Aggregate Approximation (PAA)
    let paa_values = compute_paa(&normalized_data, config.n_segments)?;

    // Step 3: Determine breakpoints
    let breakpoints = config
        .breakpoints
        .clone()
        .unwrap_or_else(|| compute_gaussian_breakpoints(config.alphabet_size));

    // Step 4: Convert PAA to symbols
    let symbolic_sequence = paa_to_symbols(&paa_values, &breakpoints)?;

    // Step 5: Compute reconstruction error (placeholder for now)
    let reconstruction_error = 0.0; // Would compute actual reconstruction error in full implementation

    // Step 6: Compute compression ratio
    let compression_ratio = time_series.len() as f64 / symbolic_sequence.len() as f64;

    Ok(SymbolicApproximationResult {
        symbolic_sequence,
        breakpoints,
        paa_values,
        reconstruction_error,
        compression_ratio,
        distance_matrix: None,
    })
}

fn apply_apca(
    _time_series: &Array1<f64>,
    _config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Placeholder for APCA implementation
    Err(TimeSeriesError::NotImplemented(
        "APCA symbolic approximation not yet implemented".to_string(),
    ))
}

fn apply_pla(
    _time_series: &Array1<f64>,
    _config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Placeholder for PLA implementation
    Err(TimeSeriesError::NotImplemented(
        "PLA symbolic approximation not yet implemented".to_string(),
    ))
}

fn apply_persist(
    _time_series: &Array1<f64>,
    _config: &SymbolicApproximationConfig,
) -> Result<SymbolicApproximationResult> {
    // Placeholder for Persist implementation
    Err(TimeSeriesError::NotImplemented(
        "Persist symbolic approximation not yet implemented".to_string(),
    ))
}

fn normalize_time_series(time_series: &Array1<f64>) -> Result<Array1<f64>> {
    let mean = time_series.mean().unwrap_or(0.0);
    let std = time_series.std(0.0);

    if std == 0.0 {
        return Ok(Array1::zeros(time_series.len()));
    }

    let normalized = time_series.mapv(|x| (x - mean) / std);
    Ok(normalized)
}

fn compute_paa(time_series: &Array1<f64>, n_segments: usize) -> Result<Array1<f64>> {
    let n = time_series.len();
    let segment_size = n as f64 / n_segments as f64;

    let mut paa_values = Array1::zeros(n_segments);

    for i in 0..n_segments {
        let start = (i as f64 * segment_size) as usize;
        let end = ((i + 1) as f64 * segment_size) as usize;
        let end = std::cmp::min(end, n);

        if start < end {
            let segment_mean = time_series.slice(s![start..end]).mean().unwrap_or(0.0);
            paa_values[i] = segment_mean;
        }
    }

    Ok(paa_values)
}

fn compute_gaussian_breakpoints(alphabet_size: usize) -> Array1<f64> {
    // Compute breakpoints based on Gaussian distribution
    // This is a simplified version - would use proper quantile function

    let mut breakpoints = Array1::zeros(alphabet_size - 1);

    for i in 0..alphabet_size - 1 {
        let quantile = (i + 1) as f64 / alphabet_size as f64;
        // Simplified inverse normal - in practice would use proper implementation
        let breakpoint = if quantile < 0.5 {
            -(1.0 - 2.0 * quantile).sqrt()
        } else {
            (2.0 * quantile - 1.0).sqrt()
        };
        breakpoints[i] = breakpoint;
    }

    breakpoints
}

fn paa_to_symbols(paa_values: &Array1<f64>, breakpoints: &Array1<f64>) -> Result<Vec<char>> {
    let alphabet_chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    let mut symbols = Vec::new();

    for &value in paa_values.iter() {
        let mut symbol_idx = 0;

        for &breakpoint in breakpoints.iter() {
            if value > breakpoint {
                symbol_idx += 1;
            } else {
                break;
            }
        }

        let symbol = alphabet_chars.get(symbol_idx).copied().unwrap_or('z');
        symbols.push(symbol);
    }

    Ok(symbols)
}

#[allow(dead_code)]
fn reconstruct_from_sax(
    _symbolic_sequence: &[char],
    _breakpoints: &Array1<f64>,
    _original_length: usize,
    _n_segments: usize,
) -> Result<Array1<f64>> {
    // Placeholder for SAX reconstruction
    Err(TimeSeriesError::NotImplemented(
        "SAX reconstruction not yet implemented".to_string(),
    ))
}

#[allow(dead_code)]
fn compute_reconstruction_error(_original: &Array1<f64>, _reconstructed: &Array1<f64>) -> f64 {
    // Placeholder for reconstruction error computation
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pca_basic() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let config = PCAConfig::default();

        let result = apply_pca(&data, &config).unwrap();

        assert_eq!(result.transformed_data.nrows(), 10);
        assert!(result.n_components_selected > 0);
        assert!(!result.explained_variance.is_empty());
    }

    #[test]
    fn test_pca_configuration() {
        let data = Array2::from_shape_vec((20, 10), (0..200).map(|x| x as f64).collect()).unwrap();

        let config = PCAConfig {
            n_components: Some(3),
            center_data: true,
            scale_data: true,
            ..Default::default()
        };

        let result = apply_pca(&data, &config).unwrap();

        assert_eq!(result.n_components_selected, 3);
        assert_eq!(result.transformed_data.ncols(), 3);
        assert_eq!(result.components.ncols(), 3);
    }

    #[test]
    fn test_functional_pca_basic() {
        let functional_data =
            Array2::from_shape_vec((5, 20), (0..100).map(|x| (x as f64 * 0.1).sin()).collect())
                .unwrap();

        let config = FunctionalPCAConfig::default();
        let result = apply_functional_pca(&functional_data, &config).unwrap();

        assert!(result.functional_components.nrows() > 0);
        assert!(!result.explained_variance.is_empty());
    }

    #[test]
    fn test_dtw_barycenter_basic() {
        let ts1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let ts2 = Array1::from_vec(vec![0.5, 1.5, 2.5, 1.5, 0.5]);
        let time_series = vec![ts1, ts2];

        let config = DTWBarycenterConfig::default();
        let result = compute_dtw_barycenter(&time_series, &config).unwrap();

        assert!(!result.barycenter.is_empty());
        assert_eq!(result.distances.len(), 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_symbolic_approximation_sax() {
        let time_series = Array1::from_shape_fn(100, |i| (i as f64 * 0.1).sin());
        let config = SymbolicApproximationConfig::default();

        let result = apply_symbolic_approximation(&time_series, &config).unwrap();

        assert!(!result.symbolic_sequence.is_empty());
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_pca_edge_cases() {
        // Test with minimal data
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let config = PCAConfig::default();

        let result = apply_pca(&data, &config).unwrap();
        assert!(result.n_components_selected <= 2);
    }

    #[test]
    fn test_dtw_single_series() {
        let ts = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let time_series = vec![ts];

        let config = DTWBarycenterConfig::default();
        let result = compute_dtw_barycenter(&time_series, &config).unwrap();

        assert_eq!(result.barycenter.len(), 3);
        assert_eq!(result.distances.len(), 1);
    }
}
