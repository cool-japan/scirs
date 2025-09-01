//! Fast random projection methods for dimensionality reduction
//!
//! This module provides algorithms for random projections, which are efficient
//! techniques for reducing dimensionality of data while approximately preserving
//! distances between points. These methods are particularly useful for large-scale
//! machine learning applications where computational efficiency is crucial.
//!
//! ## Available Methods
//!
//! * **Gaussian Random Projection**: Projects data using a random matrix with
//!   entries drawn from a Gaussian distribution
//! * **Sparse Random Projection**: Uses a sparse random matrix for projection,
//!   which is computationally more efficient
//! * **Very Sparse Random Projection**: An extremely sparse random projection
//!   that only uses values {-1, 0, 1}
//! * **Johnson-Lindenstrauss Transform**: Implements the Johnson-Lindenstrauss lemma
//!   for dimension reduction with theoretical guarantees

use ndarray::{Array2, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, Zero};
use rand::{self, Rng};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Generate a random projection matrix using Gaussian distribution
///
/// Creates a random matrix with entries drawn from a Gaussian distribution N(0, 1/k),
/// where k is the target dimension. This scaling ensures that the expected squared
/// Euclidean distance between points is preserved.
///
/// # Arguments
///
/// * `n_components` - Number of dimensions to project to
/// * `n_features` - Original number of features
///
/// # Returns
///
/// * A random projection matrix of shape (n_features, n_components)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_linalg::projection::gaussian_randommatrix;
///
/// // Generate a 1000x100 random projection matrix (projecting from 1000 to 100 dimensions)
/// let projectionmatrix = gaussian_randommatrix::<f64>(100, 1000).unwrap();
/// assert_eq!(projectionmatrix.shape(), &[1000, 100]);
/// ```
#[allow(dead_code)]
pub fn gaussian_randommatrix<F: Float + NumAssign + Zero + Sum + ScalarOperand>(
    n_components: usize,
    n_features: usize,
) -> LinalgResult<Array2<F>> {
    if n_components >= n_features {
        return Err(LinalgError::DimensionError(format!(
            "n_components must be less than n_features, got {n_components} >= {n_features}"
        )));
    }

    let mut rng = rand::rng();

    // Scaling factor for preserving distances
    let scale = F::from(1.0 / (n_components as f64).sqrt()).unwrap();

    // Generate Gaussian random matrix
    let mut _components = Array2::<F>::zeros((n_features, n_components));

    for i in 0..n_features {
        for j in 0..n_components {
            // Generate a standard normal using Box-Muller transform
            let u1: f64 = rng.random_range(0.0..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            let value = F::from(z).unwrap() * scale;
            _components[[i, j]] = value;
        }
    }

    Ok(_components)
}

/// Generate a sparse random projection matrix
///
/// Creates a sparse random matrix that has entries from {-√(s), 0, √(s)} with probabilities
/// {1/2s, 1-1/s, 1/2s}, where s determines the sparsity (typically set to √(n_features)).
/// This approach is computationally more efficient than Gaussian random projection while
/// maintaining similar theoretical guarantees.
///
/// # Arguments
///
/// * `n_components` - Number of dimensions to project to
/// * `n_features` - Original number of features
/// * `density` - Density of the random matrix (between 0 and 1, where smaller means sparser)
///
/// # Returns
///
/// * A sparse random projection matrix of shape (n_features, n_components)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_linalg::projection::sparse_randommatrix;
///
/// // Generate a 1000x100 sparse random projection matrix with density 0.1
/// let projectionmatrix = sparse_randommatrix::<f64>(100, 1000, 0.1).unwrap();
/// assert_eq!(projectionmatrix.shape(), &[1000, 100]);
/// ```
#[allow(dead_code)]
pub fn sparse_randommatrix<F: Float + NumAssign + Zero + Sum + ScalarOperand>(
    n_components: usize,
    n_features: usize,
    density: f64,
) -> LinalgResult<Array2<F>> {
    if n_components >= n_features {
        return Err(LinalgError::DimensionError(format!(
            "n_components must be less than n_features, got {n_components} >= {n_features}"
        )));
    }

    if density <= 0.0 || density > 1.0 {
        return Err(LinalgError::ValueError(format!(
            "density must be in (0, 1], got {density}"
        )));
    }

    let mut rng = rand::rng();

    // Scaling factor for preserving distances
    let scale = F::from(1.0 / (density * n_components as f64).sqrt()).unwrap();

    // Compute probability thresholds
    let prob_zero = 1.0 - density;
    let prob_neg = density / 2.0;

    // Generate sparse random matrix
    let mut _components = Array2::<F>::zeros((n_features, n_components));

    for i in 0..n_features {
        for j in 0..n_components {
            let prob = rng.random_range(0.0..1.0);
            let value = if prob < prob_neg {
                -scale
            } else if prob < prob_neg + prob_zero {
                F::zero()
            } else {
                scale
            };
            _components[[i, j]] = value;
        }
    }

    Ok(_components)
}

/// Generate a very sparse random projection matrix
///
/// Creates an extremely sparse random matrix with entries from {-1, 0, 1} with
/// optimized probability distribution. This is computationally even more efficient
/// than the sparse random projection while still preserving approximate distances.
///
/// # Arguments
///
/// * `n_components` - Number of dimensions to project to
/// * `n_features` - Original number of features
///
/// # Returns
///
/// * A very sparse random projection matrix of shape (n_features, n_components)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_linalg::projection::very_sparse_randommatrix;
///
/// // Generate a 1000x100 very sparse random projection matrix
/// let projectionmatrix = very_sparse_randommatrix::<f64>(100, 1000).unwrap();
/// assert_eq!(projectionmatrix.shape(), &[1000, 100]);
/// ```
#[allow(dead_code)]
pub fn very_sparse_randommatrix<F: Float + NumAssign + Zero + Sum + ScalarOperand>(
    n_components: usize,
    n_features: usize,
) -> LinalgResult<Array2<F>> {
    if n_components >= n_features {
        return Err(LinalgError::DimensionError(format!(
            "n_components must be less than n_features, got {n_components} >= {n_features}"
        )));
    }

    let mut rng = rand::rng();

    // Compute the sparsity parameter (sqrt(n_features))
    let s = (n_features as f64).sqrt();
    let prob_nonzero = 1.0 / s;
    let prob_neg = prob_nonzero / 2.0;

    // Scaling factor for preserving distances
    let scale = F::from((s / n_components as f64).sqrt()).unwrap();

    // Generate very sparse random matrix
    let mut _components = Array2::<F>::zeros((n_features, n_components));

    for i in 0..n_features {
        for j in 0..n_components {
            let prob = rng.random_range(0.0..1.0);
            let value = if prob < prob_neg {
                F::from(-1.0).unwrap() * scale
            } else if prob < prob_nonzero {
                F::from(1.0).unwrap() * scale
            } else {
                F::zero()
            };
            _components[[i, j]] = value;
        }
    }

    Ok(_components)
}

/// Project data using a random projection matrix
///
/// Performs dimensionality reduction by projecting data onto a lower-dimensional
/// space using a random projection matrix.
///
/// # Arguments
///
/// * `X` - Input data of shape (n_samples, n_features)
/// * `components` - Random projection matrix of shape (n_features, n_components)
///
/// # Returns
///
/// * Projected data of shape (n_samples, n_components)
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2};
/// use scirs2_linalg::projection::{gaussian_randommatrix, project};
///
/// // Generate sample data
/// let n_samples = 100;
/// let n_features = 1000;
/// let n_components = 50;
/// let X = Array2::<f64>::ones((n_samples, n_features));
///
/// // Generate random projection matrix
/// let components = gaussian_randommatrix::<f64>(n_components, n_features).unwrap();
///
/// // Project data
/// let X_projected = project(&X.view(), &components.view()).unwrap();
/// assert_eq!(X_projected.shape(), &[n_samples, n_components]);
/// ```
#[allow(dead_code)]
pub fn project<F: Float + NumAssign + Sum + ScalarOperand>(
    x: &ArrayView2<F>,
    components: &ArrayView2<F>,
) -> LinalgResult<Array2<F>> {
    let (_n_samples, n_features) = x.dim();
    let (n_components_features, _n_components) = components.dim();

    if n_features != n_components_features {
        return Err(LinalgError::DimensionError(format!(
            "Incompatible dimensions: x has {n_features} features but components has {n_components_features} features"
        )));
    }

    // Project the data using matrix multiplication
    let x_projected = x.dot(components);

    Ok(x_projected)
}

/// Project data using Johnson-Lindenstrauss transform
///
/// Performs dimensionality reduction using the Johnson-Lindenstrauss transform,
/// which provides theoretical guarantees on the preservation of pairwise distances.
/// This implementation automatically determines the optimal number of components
/// based on the number of samples and a specified error tolerance.
///
/// # Arguments
///
/// * `X` - Input data of shape (n_samples, n_features)
/// * `eps` - Maximum distortion of pairwise distances (between 0 and 1)
///
/// # Returns
///
/// * Projected data and the random projection matrix
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2};
/// use scirs2_linalg::projection::johnson_lindenstrauss_transform;
///
/// // Generate sample data
/// let n_samples = 1000;
/// let n_features = 10000;
/// let X = Array2::<f64>::ones((n_samples, n_features));
///
/// // Apply Johnson-Lindenstrauss transform with 10% error tolerance
/// let (X_projected, components) = johnson_lindenstrauss_transform(&X.view(), 0.1).unwrap();
///
/// // The number of dimensions is automatically determined
/// assert!(X_projected.shape()[1] < n_features);
/// ```
#[allow(dead_code)]
pub fn johnson_lindenstrauss_transform<F: Float + NumAssign + Zero + Sum + ScalarOperand>(
    x: &ArrayView2<F>,
    eps: f64,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    if eps <= 0.0 || eps >= 1.0 {
        return Err(LinalgError::ValueError(format!(
            "eps must be in (0, 1), got {eps}"
        )));
    }

    let (n_samples, n_features) = x.dim();

    // Calculate the minimum number of dimensions required by Johnson-Lindenstrauss lemma
    let n_components = johnson_lindenstrauss_min_dim(n_samples, eps)?;

    // Ensure we're actually reducing dimensions
    let n_components = n_components.min(n_features - 1);

    // Generate Gaussian random projection matrix
    let components = gaussian_randommatrix(n_components, n_features)?;

    // Project the data
    let x_projected = project(x, &components.view())?;

    Ok((x_projected, components))
}

/// Compute the minimum number of components needed for Johnson-Lindenstrauss transform
///
/// Calculates the minimum number of dimensions required to preserve pairwise distances
/// within a specified error tolerance (epsilon) according to the Johnson-Lindenstrauss lemma.
///
/// # Arguments
///
/// * `n_samples` - Number of samples in the dataset
/// * `eps` - Maximum distortion of pairwise distances (between 0 and 1)
///
/// # Returns
///
/// * Minimum number of dimensions required
///
/// # Examples
///
/// ```
/// use scirs2_linalg::projection::johnson_lindenstrauss_min_dim;
///
/// // For 10,000 samples and 10% error tolerance
/// let min_dim = johnson_lindenstrauss_min_dim(10000, 0.1).unwrap();
/// println!("Minimum dimensions needed: {}", min_dim);
/// ```
#[allow(dead_code)]
pub fn johnson_lindenstrauss_min_dim(_nsamples: usize, eps: f64) -> LinalgResult<usize> {
    if eps <= 0.0 || eps >= 1.0 {
        return Err(LinalgError::ValueError(format!(
            "eps must be in (0, 1), got {eps}"
        )));
    }

    // Calculate the minimum number of dimensions required by Johnson-Lindenstrauss lemma
    // The formula is: k >= 4 * ln(n) / (eps^2 / 2 - eps^3 / 3)
    let denominator = eps.powi(2) / 2.0 - eps.powi(3) / 3.0;
    let min_dim = (4.0 * (_nsamples as f64).ln() / denominator).ceil() as usize;

    Ok(min_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gaussian_randommatrix() {
        let n_components = 10;
        let n_features = 100;

        // Generate random projection matrix
        let components = gaussian_randommatrix::<f64>(n_components, n_features).unwrap();

        // Check dimensions
        assert_eq!(components.shape(), &[n_features, n_components]);

        // Check error for invalid dimensions
        let result = gaussian_randommatrix::<f64>(n_features, n_features);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_randommatrix() {
        let n_components = 10;
        let n_features = 100;
        let density = 0.1;

        // Generate sparse random projection matrix
        let components = sparse_randommatrix::<f64>(n_components, n_features, density).unwrap();

        // Check dimensions
        assert_eq!(components.shape(), &[n_features, n_components]);

        // Check sparsity
        let non_zeros = components.iter().filter(|&&x| x != 0.0).count();
        let total_elements = n_features * n_components;
        let actual_density = non_zeros as f64 / total_elements as f64;

        // Allow for some random variation around the target density
        assert!(actual_density > density * 0.5 && actual_density < density * 1.5);

        // Check error for invalid inputs
        let result = sparse_randommatrix::<f64>(n_features, n_features, density);
        assert!(result.is_err());

        let result = sparse_randommatrix::<f64>(n_components, n_features, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_very_sparse_randommatrix() {
        let n_components = 10;
        let n_features = 100;

        // Generate very sparse random projection matrix
        let components = very_sparse_randommatrix::<f64>(n_components, n_features).unwrap();

        // Check dimensions
        assert_eq!(components.shape(), &[n_features, n_components]);

        // Check sparsity - should be very sparse (around 1/sqrt(n_features) non-zeros)
        let non_zeros = components.iter().filter(|&&x| x != 0.0).count();
        let total_elements = n_features * n_components;
        let expected_density = 1.0 / (n_features as f64).sqrt();
        let actual_density = non_zeros as f64 / total_elements as f64;

        // Allow for some random variation around the target density
        assert!(actual_density > expected_density * 0.3 && actual_density < expected_density * 2.0);
    }

    #[test]
    fn test_project() {
        let n_samples = 5;
        let n_features = 20;
        let n_components = 3;

        // Create a simple input matrix
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = (i * n_features + j) as f64;
            }
        }

        // Create a simple projection matrix
        let components = gaussian_randommatrix::<f64>(n_components, n_features).unwrap();

        // Project the data
        let x_projected = project(&x.view(), &components.view()).unwrap();

        // Check dimensions
        assert_eq!(x_projected.shape(), &[n_samples, n_components]);

        // Check that projection is correct (X_proj = X * components)
        let expected_proj = x.dot(&components);
        assert_eq!(x_projected, expected_proj);
    }

    #[test]
    fn test_johnson_lindenstrauss_transform() {
        let n_samples = 50;
        let n_features = 100;
        let eps = 0.2;

        // Create a simple input matrix
        let x = Array2::<f64>::ones((n_samples, n_features));

        // Apply JL transform
        let (x_projected, components) = johnson_lindenstrauss_transform(&x.view(), eps).unwrap();

        // Check that dimensions are reduced
        assert!(x_projected.shape()[1] < n_features);

        // Check that the dimensions match theoretical bounds
        let min_dim = johnson_lindenstrauss_min_dim(n_samples, eps).unwrap();
        assert!(x_projected.shape()[1] <= min_dim);

        // Check that projection is correct
        let expected_proj = project(&x.view(), &components.view()).unwrap();
        assert_eq!(x_projected, expected_proj);
    }

    #[test]
    fn test_johnson_lindenstrauss_min_dim() {
        // Known values for specific inputs
        assert!(johnson_lindenstrauss_min_dim(10000, 0.1).unwrap() >= 500);

        // Error for invalid epsilon
        assert!(johnson_lindenstrauss_min_dim(1000, 0.0).is_err());
        assert!(johnson_lindenstrauss_min_dim(1000, 1.0).is_err());
    }
}
