//! GPU-accelerated transformations
//!
//! This module provides GPU-accelerated implementations of dimensionality reduction
//! and matrix operations. Currently provides basic stubs with CPU fallback.

use crate::error::{Result, TransformError};
use scirs2_core::gpu::{GpuBackend, GpuContext};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::validation::{check_not_empty, check_positive, checkarray_finite};

/// GPU-accelerated Principal Component Analysis
#[cfg(feature = "gpu")]
pub struct GpuPCA {
    /// Number of components to compute
    pub n_components: usize,
    /// Whether to center the data
    pub center: bool,
    /// Principal components (loading vectors)
    pub components: Option<Array2<f64>>,
    /// Explained variance for each component
    pub explained_variance: Option<Array1<f64>>,
    /// Mean values for centering
    pub mean: Option<Array1<f64>>,
    /// GPU context for GPU operations
    gpu_context: Option<GpuContext>,
}

#[cfg(feature = "gpu")]
impl GpuPCA {
    /// Create a new GPU PCA instance
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to compute
    ///
    /// # Returns
    ///
    /// Returns a new GpuPCA instance with GPU context initialized
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails or if n_components is 0
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::gpu::GpuPCA;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pca = GpuPCA::new(5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(n_components: usize) -> Result<Self> {
        check_positive(n_components, "n_components")?;

        let gpu_context = GpuContext::new(GpuBackend::preferred()).map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize GPU: {}", e))
        })?;

        Ok(GpuPCA {
            n_components,
            center: true,
            components: None,
            explained_variance: None,
            mean: None,
            gpu_context: Some(gpu_context),
        })
    }

    /// Fit the PCA model using a CPU SVD-based backend.
    ///
    /// Currently delegates to the CPU SVD-based PCA implementation.
    /// A wgpu-backed Jacobi SVD path is planned for v0.5.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid or if `n_components` exceeds
    /// `min(n_samples, n_features)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::gpu::GpuPCA;
    /// # use scirs2_core::ndarray::Array2;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut pca = GpuPCA::new(2)?;
    /// let data = Array2::from_shape_vec((5, 4), vec![
    ///     1.0, 2.0, 3.0, 4.0,
    ///     2.0, 3.0, 4.0, 5.0,
    ///     3.0, 4.0, 5.0, 6.0,
    ///     4.0, 5.0, 6.0, 7.0,
    ///     5.0, 6.0, 7.0, 8.0,
    /// ])?;
    /// pca.fit(&data.view())?;
    /// assert!(pca.components.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        check_not_empty(x, "x")?;
        checkarray_finite(x, "x")?;

        let (n_samples, n_features) = x.dim();
        if self.n_components > n_features.min(n_samples) {
            return Err(TransformError::InvalidInput(
                "n_components cannot be larger than min(n_samples, n_features)".to_string(),
            ));
        }

        // Delegate to CPU SVD-based PCA; GPU Jacobi SVD path planned for v0.5.
        let mut cpu_pca = crate::reduction::PCA::new(self.n_components, self.center, false);
        cpu_pca.fit(x)?;

        self.components = cpu_pca.components().cloned();
        self.mean = cpu_pca.mean().cloned();
        // Store the explained variance ratio directly; GpuPCA::explained_variance_ratio()
        // returns this field as-is (it is already normalised to sum ≈ 1).
        self.explained_variance = cpu_pca.explained_variance_ratio().cloned();

        Ok(())
    }

    /// Transform data using the fitted PCA model.
    ///
    /// Projects the input data onto the principal components computed during
    /// [`GpuPCA::fit`].  Currently runs on the CPU; a wgpu-backed path is
    /// planned for v0.5.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Transformed data matrix with shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// Returns `NotFitted` if [`GpuPCA::fit`] has not been called yet, or
    /// `InvalidInput` if the number of features does not match.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        check_not_empty(x, "x")?;
        checkarray_finite(x, "x")?;

        let components = self.components.as_ref().ok_or_else(|| {
            TransformError::NotFitted(
                "GpuPCA model has not been fitted; call fit() first".to_string(),
            )
        })?;

        let (n_samples, n_features) = x.dim();
        let n_comp_features = components.shape()[1];

        if n_features != n_comp_features {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but GpuPCA was fitted with {} features",
                n_features, n_comp_features,
            )));
        }

        // Center the data if the model was trained with centering.
        let x_centered: Array2<f64> = if self.center {
            if let Some(mean) = &self.mean {
                let mut centered = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    for j in 0..n_features {
                        centered[[i, j]] = x[[i, j]] - mean[j];
                    }
                }
                centered
            } else {
                // mean not available — use raw data
                x.to_owned()
            }
        } else {
            x.to_owned()
        };

        // Project onto principal components: result shape (n_samples, n_components).
        let transformed = x_centered.dot(&components.t());
        Ok(transformed)
    }

    /// Fit the PCA model and project data in one step.
    ///
    /// Equivalent to calling [`GpuPCA::fit`] followed by [`GpuPCA::transform`]
    /// on the same data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Transformed data matrix with shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// Returns any error produced by [`GpuPCA::fit`] or [`GpuPCA::transform`].
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get the explained variance ratio for each principal component.
    ///
    /// The returned array sums to approximately 1.0 after fitting.
    ///
    /// # Returns
    ///
    /// Array of explained variance ratios with length `n_components`
    ///
    /// # Errors
    ///
    /// Returns `NotFitted` if [`GpuPCA::fit`] has not been called yet.
    pub fn explained_variance_ratio(&self) -> Result<Array1<f64>> {
        self.explained_variance
            .as_ref()
            .cloned()
            .ok_or_else(|| TransformError::NotFitted("GpuPCA model not fitted".to_string()))
    }
}

/// GPU-accelerated matrix operations for transformations
#[cfg(feature = "gpu")]
pub struct GpuMatrixOps {
    #[allow(dead_code)]
    gpu_context: GpuContext,
}

#[cfg(feature = "gpu")]
impl GpuMatrixOps {
    /// Create new GPU matrix operations instance
    pub fn new() -> Result<Self> {
        let gpu_context = GpuContext::new(GpuBackend::preferred()).map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize GPU: {}", e))
        })?;

        Ok(GpuMatrixOps { gpu_context })
    }

    /// GPU-accelerated matrix multiplication (placeholder)
    pub fn matmul(self_a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<Array2<f64>> {
        Err(TransformError::NotImplemented(
            "GPU matrix multiplication is not yet implemented. Use CPU operations instead."
                .to_string(),
        ))
    }

    /// GPU-accelerated SVD decomposition (placeholder)
    pub fn svd(selfa: &ArrayView2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        Err(TransformError::NotImplemented(
            "GPU SVD is not yet implemented. Use CPU operations instead.".to_string(),
        ))
    }

    /// GPU-accelerated eigendecomposition (placeholder)
    pub fn eigh(selfa: &ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        Err(TransformError::NotImplemented(
            "GPU eigendecomposition is not yet implemented. Use CPU operations instead."
                .to_string(),
        ))
    }
}

/// GPU-accelerated t-SNE implementation
#[cfg(feature = "gpu")]
pub struct GpuTSNE {
    /// Number of dimensions for the embedding
    pub n_components: usize,
    /// Perplexity parameter
    pub perplexity: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// GPU context
    #[allow(dead_code)]
    gpu_context: GpuContext,
}

#[cfg(feature = "gpu")]
impl GpuTSNE {
    /// Create new GPU t-SNE instance
    pub fn new(n_components: usize) -> Result<Self> {
        check_positive(n_components, "n_components")?;

        let gpu_context = GpuContext::new(GpuBackend::preferred()).map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize GPU: {}", e))
        })?;

        Ok(GpuTSNE {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            max_iter: 1000,
            gpu_context,
        })
    }

    /// Set perplexity parameter
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit and transform data using GPU-accelerated t-SNE (placeholder)
    pub fn fit_transform(selfx: &ArrayView2<f64>) -> Result<Array2<f64>> {
        Err(TransformError::NotImplemented(
            "GPU t-SNE is not yet implemented. Use CPU t-SNE instead.".to_string(),
        ))
    }
}

// Stub implementations when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub struct GpuPCA;

#[cfg(not(feature = "gpu"))]
pub struct GpuMatrixOps;

#[cfg(not(feature = "gpu"))]
pub struct GpuTSNE;

#[cfg(not(feature = "gpu"))]
impl GpuPCA {
    pub fn new(_ncomponents: usize) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuMatrixOps {
    pub fn new() -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuTSNE {
    pub fn new(_ncomponents: usize) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ------------------------------------------------------------------
    // Helper: construct a simple (5 × 4) test matrix with known structure.
    // Each row is an arithmetic progression, so the data lies close to a
    // 1-dimensional subspace; PCA with 2 components should capture ≥ 99 %
    // of the variance.
    // ------------------------------------------------------------------
    #[cfg(feature = "gpu")]
    fn sample_data_5x4() -> Array2<f64> {
        Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0,
                5.0, 6.0, 7.0, 8.0,
            ],
        )
        .expect("shape vec must be consistent")
    }

    // ------------------------------------------------------------------
    // Construction tests
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_creation() {
        let pca = GpuPCA::new(3);
        assert!(pca.is_ok());
        let pca = pca.expect("GpuPCA::new should succeed");
        assert_eq!(pca.n_components, 3);
        assert!(pca.center);
        assert!(pca.components.is_none());
        assert!(pca.explained_variance.is_none());
        assert!(pca.mean.is_none());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_invalid_components() {
        let result = GpuPCA::new(0);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // fit() tests — verify that fit populates all model fields correctly.
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_populates_fields() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();

        pca.fit(&data.view()).expect("GpuPCA::fit should succeed");

        // All model fields should now be populated.
        assert!(pca.components.is_some(), "components must be set after fit");
        assert!(
            pca.explained_variance.is_some(),
            "explained_variance must be set after fit"
        );
        assert!(pca.mean.is_some(), "mean must be set after fit");
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_components_shape() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        // components: (n_components, n_features) = (2, 4)
        let comp = pca.components.as_ref().expect("components present");
        assert_eq!(comp.shape(), &[2, 4]);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_mean_shape() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        let mean = pca.mean.as_ref().expect("mean present");
        assert_eq!(mean.len(), 4, "mean must have one entry per feature");
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_explained_variance_length() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        let ev = pca
            .explained_variance
            .as_ref()
            .expect("explained_variance present");
        assert_eq!(
            ev.len(),
            2,
            "explained_variance must have n_components entries"
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_explained_variance_non_increasing() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        let ev = pca
            .explained_variance
            .as_ref()
            .expect("explained_variance present");
        for i in 0..ev.len() - 1 {
            assert!(
                ev[i] >= ev[i + 1] - 1e-10,
                "explained variance must be non-increasing: ev[{}]={} < ev[{}]={}",
                i,
                ev[i],
                i + 1,
                ev[i + 1]
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_n_components_too_large() {
        // n_components > min(n_samples, n_features) must fail
        let mut pca = GpuPCA::new(10).expect("GpuPCA::new(10) should succeed");
        let data = sample_data_5x4(); // 5 × 4 → max 4 components
        let result = pca.fit(&data.view());
        assert!(
            result.is_err(),
            "should fail when n_components > min(n_samples, n_features)"
        );
    }

    // ------------------------------------------------------------------
    // transform() tests
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_transform_shape() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        let transformed = pca
            .transform(&data.view())
            .expect("transform must succeed after fit");

        // Output shape: (n_samples, n_components) = (5, 2)
        assert_eq!(transformed.shape(), &[5, 2]);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_transform_all_finite() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");
        let transformed = pca.transform(&data.view()).expect("transform must succeed");

        assert!(
            transformed.iter().all(|v| v.is_finite()),
            "all transformed values must be finite"
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_transform_before_fit_returns_error() {
        let pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        let result = pca.transform(&data.view());
        assert!(result.is_err(), "transform before fit must return an error");
    }

    // ------------------------------------------------------------------
    // fit_transform() tests
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_transform_shape() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        let ft = pca
            .fit_transform(&data.view())
            .expect("fit_transform must succeed");
        assert_eq!(ft.shape(), &[5, 2]);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_fit_transform_matches_fit_then_transform() {
        let data = sample_data_5x4();

        let mut pca1 = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let ft = pca1
            .fit_transform(&data.view())
            .expect("fit_transform must succeed");

        let mut pca2 = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        pca2.fit(&data.view()).expect("fit must succeed");
        let t = pca2
            .transform(&data.view())
            .expect("transform must succeed");

        assert_eq!(ft.shape(), t.shape());
        for (a, b) in ft.iter().zip(t.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "fit_transform and fit+transform must agree: {} vs {}",
                a,
                b
            );
        }
    }

    // ------------------------------------------------------------------
    // explained_variance_ratio() method
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_explained_variance_ratio_before_fit_returns_error() {
        let pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        assert!(pca.explained_variance_ratio().is_err());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_pca_explained_variance_ratio_after_fit() {
        let mut pca = GpuPCA::new(2).expect("GpuPCA::new should succeed");
        let data = sample_data_5x4();
        pca.fit(&data.view()).expect("fit must succeed");

        let ratio = pca
            .explained_variance_ratio()
            .expect("explained_variance_ratio must succeed after fit");
        assert_eq!(ratio.len(), 2);
        // All ratios must be non-negative
        assert!(ratio.iter().all(|&v| v >= 0.0));
    }

    // ------------------------------------------------------------------
    // GpuMatrixOps and GpuTSNE smoke tests (unchanged behaviour)
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_matrix_ops_creation() {
        let ops = GpuMatrixOps::new();
        assert!(ops.is_ok());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_tsne_creation() {
        let tsne = GpuTSNE::new(2);
        assert!(tsne.is_ok());
        let tsne = tsne.expect("GpuTSNE::new should succeed");
        assert_eq!(tsne.n_components, 2);
        assert_eq!(tsne.perplexity, 30.0);
        assert_eq!(tsne.learning_rate, 200.0);
        assert_eq!(tsne.max_iter, 1000);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_tsne_with_params() {
        let tsne = GpuTSNE::new(3)
            .expect("GpuTSNE::new should succeed")
            .with_perplexity(50.0)
            .with_learning_rate(100.0)
            .with_max_iter(500);

        assert_eq!(tsne.n_components, 3);
        assert_eq!(tsne.perplexity, 50.0);
        assert_eq!(tsne.learning_rate, 100.0);
        assert_eq!(tsne.max_iter, 500);
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn test_gpu_features_disabled() {
        assert!(GpuPCA::new(2).is_err());
        assert!(GpuMatrixOps::new().is_err());
        assert!(GpuTSNE::new(2).is_err());
    }
}
