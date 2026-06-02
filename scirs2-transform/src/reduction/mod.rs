//! Dimensionality reduction techniques
//!
//! This module provides algorithms for reducing the dimensionality of data,
//! which is useful for visualization, feature extraction, and reducing
//! computational complexity.

/// Factor Analysis module
pub mod factor_analysis;
mod isomap;
mod lle;
mod spectral_embedding;
mod tsne;
mod umap;

/// Laplacian Eigenmaps for manifold learning
pub mod laplacian_eigenmaps;

/// Diffusion Maps for nonlinear dimensionality reduction
pub mod diffusion_maps;

pub use crate::reduction::diffusion_maps::DiffusionMaps;
pub use crate::reduction::factor_analysis::{
    factor_analysis, scree_plot_data, FactorAnalysis, FactorAnalysisResult, RotationMethod,
    ScreePlotData,
};
pub use crate::reduction::isomap::Isomap;
pub use crate::reduction::laplacian_eigenmaps::{
    GraphMethod, LaplacianEigenmaps, LaplacianType as LELaplacianType,
};
pub use crate::reduction::lle::LLE;
pub use crate::reduction::spectral_embedding::{AffinityMethod, SpectralEmbedding};
pub use crate::reduction::tsne::{trustworthiness, TSNE};
pub use crate::reduction::umap::UMAP;

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_linalg::svd;

use crate::error::{Result, TransformError};

// Define a small value to use for comparison with zero
const EPSILON: f64 = 1e-10;

/// Principal Component Analysis (PCA) dimensionality reduction
///
/// PCA finds the directions of maximum variance in the data and
/// projects the data onto a lower dimensional space.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of components to keep
    n_components: usize,
    /// Whether to center the data before computing the SVD
    center: bool,
    /// Whether to scale the data before computing the SVD
    scale: bool,
    /// The principal components
    components: Option<Array2<f64>>,
    /// The mean of the training data
    mean: Option<Array1<f64>>,
    /// The standard deviation of the training data
    std: Option<Array1<f64>>,
    /// The singular values of the centered training data
    singular_values: Option<Array1<f64>>,
    /// The explained variance ratio
    explained_variance_ratio: Option<Array1<f64>>,
}

impl PCA {
    /// Creates a new PCA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of components to keep
    /// * `center` - Whether to center the data before computing the SVD
    /// * `scale` - Whether to scale the data before computing the SVD
    ///
    /// # Returns
    /// * A new PCA instance
    pub fn new(ncomponents: usize, center: bool, scale: bool) -> Self {
        PCA {
            n_components: ncomponents,
            center,
            scale,
            components: None,
            mean: None,
            std: None,
            singular_values: None,
            explained_variance_ratio: None,
        }
    }

    /// Fits the PCA model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        // PCA can return at most min(n_samples, n_features) non-trivial components.
        // Validate against both: n_components must not exceed either dimension.
        // (Previously only n_features was checked; on n << p wide data this allowed
        // n_components > rank(X) which made the legacy full_matrices SVD path
        // collapse to zero singular values for the trailing components.)
        let max_components = n_samples.min(n_features);
        if self.n_components > max_components {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= min(n_samples, n_features)={}",
                self.n_components, max_components
            )));
        }

        // Center and scale data if requested
        let mut x_processed = Array2::zeros((n_samples, n_features));
        let mut mean = Array1::zeros(n_features);
        let mut std = Array1::ones(n_features);

        if self.center {
            for j in 0..n_features {
                let col_mean = x_f64.column(j).sum() / n_samples as f64;
                mean[j] = col_mean;

                for i in 0..n_samples {
                    x_processed[[i, j]] = x_f64[[i, j]] - col_mean;
                }
            }
        } else {
            x_processed.assign(&x_f64);
        }

        if self.scale {
            for j in 0..n_features {
                let col_std =
                    (x_processed.column(j).mapv(|x| x * x).sum() / n_samples as f64).sqrt();
                if col_std > f64::EPSILON {
                    std[j] = col_std;

                    for i in 0..n_samples {
                        x_processed[[i, j]] /= col_std;
                    }
                }
            }
        }

        // Perform thin SVD: full_matrices=false returns U (n x k), S (k), Vᵀ (k x m)
        // where k = min(n_samples, n_features). On wide data (n << p) this is critical:
        // full_matrices=true would build a p×p Vᵀ via Gram–Schmidt orthogonal extension,
        // which is O(p³) work — catastrophic for typical n << p problems (issue #124).
        let (_u, s, vt) = match svd::<f64>(&x_processed.view(), false, None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        // Extract components and singular values
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut singular_values = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            singular_values[i] = s[i];
            for j in 0..n_features {
                components[[i, j]] = vt[[i, j]];
            }
        }

        // Compute explained variance ratio.
        // With thin SVD we have only min(n,m) singular values, but for centred data the
        // trace of the covariance equals sum_i sigma_i^2 over those same min(n,m) values
        // (the trailing singular values from full_matrices=true would be zero), so this
        // ratio is identical to the previous behaviour on well-posed inputs.
        let total_variance = s.mapv(|s| s * s).sum();
        let explained_variance_ratio = if total_variance > EPSILON {
            singular_values.mapv(|s| s * s / total_variance)
        } else {
            Array1::zeros(self.n_components)
        };

        self.components = Some(components);
        self.mean = Some(mean);
        self.std = Some(std);
        self.singular_values = Some(singular_values);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Transforms the input data using the fitted PCA model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "PCA model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().expect("Operation failed");
        let mean = self.mean.as_ref().expect("Operation failed");
        let std = self.std.as_ref().expect("Operation failed");

        if n_features != components.shape()[1] {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but PCA was fitted with {} features",
                n_features,
                components.shape()[1]
            )));
        }

        // Center and scale data if the model was fitted with centering/scaling
        let mut x_processed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut value = x_f64[[i, j]];

                if self.center {
                    value -= mean[j];
                }

                if self.scale {
                    value /= std[j];
                }

                x_processed[[i, j]] = value;
            }
        }

        // Project data onto principal components
        let mut transformed = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut dot_product = 0.0;
                for k in 0..n_features {
                    dot_product += x_processed[[i, k]] * components[[j, k]];
                }
                transformed[[i, j]] = dot_product;
            }
        }

        Ok(transformed)
    }

    /// Fits the PCA model to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the principal components
    ///
    /// # Returns
    /// * `Option<&Array2<f64>>` - The principal components, shape (n_components, n_features)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Returns the mean of the training data (used for centering)
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The per-feature mean, shape (n_features,)
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Returns the explained variance ratio
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }
}

/// Truncated Singular Value Decomposition (SVD) for dimensionality reduction
///
/// This transformer performs linear dimensionality reduction by means of
/// truncated singular value decomposition (SVD). It works on any data and
/// not just sparse matrices.
#[derive(Debug, Clone)]
pub struct TruncatedSVD {
    /// Number of components to keep
    n_components: usize,
    /// The singular values of the training data
    singular_values: Option<Array1<f64>>,
    /// The right singular vectors
    components: Option<Array2<f64>>,
    /// The explained variance ratio
    explained_variance_ratio: Option<Array1<f64>>,
}

impl TruncatedSVD {
    /// Creates a new TruncatedSVD instance
    ///
    /// # Arguments
    /// * `n_components` - Number of components to keep
    ///
    /// # Returns
    /// * A new TruncatedSVD instance
    pub fn new(ncomponents: usize) -> Self {
        TruncatedSVD {
            n_components: ncomponents,
            singular_values: None,
            components: None,
            explained_variance_ratio: None,
        }
    }

    /// Fits the TruncatedSVD model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_features={}",
                self.n_components, n_features
            )));
        }

        // Perform SVD
        let (_u, s, vt) = match svd::<f64>(&x_f64.view(), true, None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        // Extract components and singular values
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut singular_values = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            singular_values[i] = s[i];
            for j in 0..n_features {
                components[[i, j]] = vt[[i, j]];
            }
        }

        // Compute explained variance ratio
        let total_variance =
            (x_f64.map_axis(Axis(1), |row| row.dot(&row)).sum()) / n_samples as f64;
        let explained_variance = singular_values.mapv(|s| s * s / n_samples as f64);
        let explained_variance_ratio = explained_variance.mapv(|v| v / total_variance);

        self.singular_values = Some(singular_values);
        self.components = Some(components);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Transforms the input data using the fitted TruncatedSVD model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "TruncatedSVD model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().expect("Operation failed");

        if n_features != components.shape()[1] {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but TruncatedSVD was fitted with {} features",
                n_features,
                components.shape()[1]
            )));
        }

        // Project data onto components
        let mut transformed = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut dot_product = 0.0;
                for k in 0..n_features {
                    dot_product += x_f64[[i, k]] * components[[j, k]];
                }
                transformed[[i, j]] = dot_product;
            }
        }

        Ok(transformed)
    }

    /// Fits the TruncatedSVD model to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the components (right singular vectors)
    ///
    /// # Returns
    /// * `Option<&Array2<f64>>` - The components, shape (n_components, n_features)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Returns the singular values
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The singular values
    pub fn singular_values(&self) -> Option<&Array1<f64>> {
        self.singular_values.as_ref()
    }

    /// Returns the explained variance ratio
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }
}

/// Linear Discriminant Analysis (LDA) for dimensionality reduction
///
/// LDA finds the directions that maximize the separation between classes.
#[derive(Debug, Clone)]
pub struct LDA {
    /// Number of components to keep
    n_components: usize,
    /// Whether to use Singular Value Decomposition
    solver: String,
    /// The LDA components
    components: Option<Array2<f64>>,
    /// The class means
    means: Option<Array2<f64>>,
    /// The explained variance ratio
    explained_variance_ratio: Option<Array1<f64>>,
}

impl LDA {
    /// Creates a new LDA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of components to keep
    /// * `solver` - The solver to use ('svd' or 'eigen')
    ///
    /// # Returns
    /// * A new LDA instance
    pub fn new(ncomponents: usize, solver: &str) -> Result<Self> {
        if solver != "svd" && solver != "eigen" {
            return Err(TransformError::InvalidInput(
                "solver must be 'svd' or 'eigen'".to_string(),
            ));
        }

        Ok(LDA {
            n_components: ncomponents,
            solver: solver.to_string(),
            components: None,
            means: None,
            explained_variance_ratio: None,
        })
    }

    /// Fits the LDA model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    /// * `y` - The target labels, shape (n_samples,)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S1, S2>(&mut self, x: &ArrayBase<S1, Ix2>, y: &ArrayBase<S2, Ix1>) -> Result<()>
    where
        S1: Data,
        S2: Data,
        S1::Elem: Float + NumCast,
        S2::Elem: Copy + NumCast + Eq + std::hash::Hash,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "x and y have incompatible shapes: x has {} samples, y has {} elements",
                n_samples,
                y.len()
            )));
        }

        // Convert y to class indices
        let mut class_indices = vec![];
        let mut class_map = std::collections::HashMap::new();
        let mut next_class_idx = 0;

        for &label in y.iter() {
            let label_u64 = NumCast::from(label).unwrap_or(0);

            if let std::collections::hash_map::Entry::Vacant(e) = class_map.entry(label_u64) {
                e.insert(next_class_idx);
                next_class_idx += 1;
            }

            class_indices.push(class_map[&label_u64]);
        }

        let n_classes = class_map.len();

        if n_classes <= 1 {
            return Err(TransformError::InvalidInput(
                "y has less than 2 classes, LDA requires at least 2 classes".to_string(),
            ));
        }

        let maxn_components = n_classes - 1;
        if self.n_components > maxn_components {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_classes-1={}",
                self.n_components, maxn_components
            )));
        }

        // Compute class means
        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_counts = vec![0; n_classes];

        for i in 0..n_samples {
            let class_idx = class_indices[i];
            class_counts[class_idx] += 1;

            for j in 0..n_features {
                class_means[[class_idx, j]] += x_f64[[i, j]];
            }
        }

        for i in 0..n_classes {
            if class_counts[i] > 0 {
                for j in 0..n_features {
                    class_means[[i, j]] /= class_counts[i] as f64;
                }
            }
        }

        // Compute global mean
        let mut global_mean = Array1::<f64>::zeros(n_features);
        for i in 0..n_samples {
            for j in 0..n_features {
                global_mean[j] += x_f64[[i, j]];
            }
        }
        global_mean.mapv_inplace(|x: f64| x / n_samples as f64);

        // Fast path for the n_samples < n_features (wide / n << p) regime with the
        // svd solver: form only n×p centred-within-class and c×p between-class matrices
        // and SVD those instead of the p×p scatter matrices. This is the textbook
        // small-side algorithm — equivalent to the existing p×p formulation but
        // O(n·p·rank) instead of O(p⁴). Issue #124.
        //
        // We deliberately keep the wide-old eigen path untouched: the eigen branch
        // requires Sw^{-1}·Sb explicitly, which the small-side trick does not produce.
        if self.solver == "svd" && n_samples < n_features {
            let (components, eigenvalues) = self.fit_svd_small_side(
                &x_f64,
                &class_means,
                &global_mean,
                &class_indices,
                &class_counts,
                n_samples,
                n_features,
                n_classes,
            )?;
            let total_eig = eigenvalues.iter().sum::<f64>();
            let explained_variance_ratio = if total_eig > EPSILON {
                eigenvalues.mapv(|e| e / total_eig)
            } else {
                Array1::from_elem(self.n_components, 1.0 / self.n_components as f64)
            };
            self.components = Some(components);
            self.means = Some(class_means);
            self.explained_variance_ratio = Some(explained_variance_ratio);
            return Ok(());
        }

        // Compute within-class scatter matrix
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_samples {
            let class_idx = class_indices[i];
            let mut x_centered = Array1::<f64>::zeros(n_features);

            for j in 0..n_features {
                x_centered[j] = x_f64[[i, j]] - class_means[[class_idx, j]];
            }

            for j in 0..n_features {
                for k in 0..n_features {
                    sw[[j, k]] += x_centered[j] * x_centered[k];
                }
            }
        }

        // Compute between-class scatter matrix
        let mut sb = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_classes {
            let mut mean_diff = Array1::<f64>::zeros(n_features);
            for j in 0..n_features {
                mean_diff[j] = class_means[[i, j]] - global_mean[j];
            }

            for j in 0..n_features {
                for k in 0..n_features {
                    sb[[j, k]] += class_counts[i] as f64 * mean_diff[j] * mean_diff[k];
                }
            }
        }

        // Solve the generalized eigenvalue problem
        let mut components = Array2::<f64>::zeros((self.n_components, n_features));
        let mut eigenvalues = Array1::<f64>::zeros(self.n_components);

        if self.solver == "svd" {
            // SVD-based solver

            // Decompose the within-class scatter matrix
            let (u_sw, s_sw, vt_sw) = match svd::<f64>(&sw.view(), true, None) {
                Ok(result) => result,
                Err(e) => return Err(TransformError::LinalgError(e)),
            };

            // Compute the pseudoinverse of sw^(1/2)
            let mut sw_sqrt_inv = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                if s_sw[i] > EPSILON {
                    for j in 0..n_features {
                        for k in 0..n_features {
                            let s_inv_sqrt = 1.0 / s_sw[i].sqrt();
                            sw_sqrt_inv[[j, k]] += u_sw[[j, i]] * s_inv_sqrt * vt_sw[[i, k]];
                        }
                    }
                }
            }

            // Transform the between-class scatter matrix
            let mut sb_transformed = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                for j in 0..n_features {
                    for k in 0..n_features {
                        for l in 0..n_features {
                            sb_transformed[[i, j]] +=
                                sw_sqrt_inv[[i, k]] * sb[[k, l]] * sw_sqrt_inv[[l, j]];
                        }
                    }
                }
            }

            // Perform SVD on the transformed between-class scatter matrix
            let (u_sb, s_sb, vt_sb) = match svd::<f64>(&sb_transformed.view(), true, None) {
                Ok(result) => result,
                Err(e) => return Err(TransformError::LinalgError(e)),
            };

            // Compute the LDA components
            for i in 0..self.n_components {
                eigenvalues[i] = s_sb[i];

                for j in 0..n_features {
                    for k in 0..n_features {
                        components[[i, j]] += sw_sqrt_inv[[k, j]] * u_sb[[k, i]];
                    }
                }
            }
        } else {
            // Eigen-based solver - proper generalized eigenvalue problem
            // Solve: Sb * v = λ * Sw * v

            // Step 1: Regularize Sw to ensure it's invertible
            let mut sw_reg = sw.clone();
            for i in 0..n_features {
                sw_reg[[i, i]] += EPSILON; // Add small regularization to diagonal
            }

            // Step 2: Compute Cholesky decomposition of regularized Sw
            // We'll use a simpler approach: Sw^(-1) * Sb
            let (u_sw, s_sw, vt_sw) = match svd::<f64>(&sw_reg.view(), true, None) {
                Ok(result) => result,
                Err(e) => return Err(TransformError::LinalgError(e)),
            };

            // Compute pseudoinverse of Sw
            let mut sw_inv = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                if s_sw[i] > EPSILON {
                    for j in 0..n_features {
                        for k in 0..n_features {
                            sw_inv[[j, k]] += u_sw[[j, i]] * (1.0 / s_sw[i]) * vt_sw[[i, k]];
                        }
                    }
                }
            }

            // Step 3: Compute Sw^(-1) * Sb
            let mut sw_inv_sb = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                for j in 0..n_features {
                    for k in 0..n_features {
                        sw_inv_sb[[i, j]] += sw_inv[[i, k]] * sb[[k, j]];
                    }
                }
            }

            // Step 4: Compute eigendecomposition of Sw^(-1) * Sb
            // Since this matrix may not be symmetric, we use the approach where we
            // symmetrize it by computing (Sw^(-1) * Sb + (Sw^(-1) * Sb)^T) / 2
            let mut sym_matrix = Array2::<f64>::zeros((n_features, n_features));
            for i in 0..n_features {
                for j in 0..n_features {
                    sym_matrix[[i, j]] = (sw_inv_sb[[i, j]] + sw_inv_sb[[j, i]]) / 2.0;
                }
            }

            // Perform eigendecomposition on the symmetrized matrix
            let (eig_vals, eig_vecs) = match scirs2_linalg::eigh::<f64>(&sym_matrix.view(), None) {
                Ok(result) => result,
                Err(_) => {
                    // Fallback to SVD if eigendecomposition fails
                    let (u, s, vt) = match svd::<f64>(&sw_inv_sb.view(), true, None) {
                        Ok(result) => result,
                        Err(e) => return Err(TransformError::LinalgError(e)),
                    };
                    (s, u)
                }
            };

            // Sort eigenvalues and eigenvectors in descending order
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&i, &j| {
                eig_vals[j]
                    .partial_cmp(&eig_vals[i])
                    .expect("Operation failed")
            });

            // Select top n_components eigenvectors
            for i in 0..self.n_components {
                let idx = indices[i];
                eigenvalues[i] = eig_vals[idx].max(0.0); // Ensure non-negative

                for j in 0..n_features {
                    components[[i, j]] = eig_vecs[[j, idx]];
                }
            }

            // Normalize components
            for i in 0..self.n_components {
                let mut norm = 0.0;
                for j in 0..n_features {
                    norm += components[[i, j]] * components[[i, j]];
                }
                norm = norm.sqrt();

                if norm > EPSILON {
                    for j in 0..n_features {
                        components[[i, j]] /= norm;
                    }
                }
            }
        }

        // Compute explained variance ratio
        let total_eigenvalues = eigenvalues.iter().sum::<f64>();
        let explained_variance_ratio = eigenvalues.mapv(|e| e / total_eigenvalues);

        self.components = Some(components);
        self.means = Some(class_means);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Small-side SVD solver for the n_samples < n_features ("wide" / n << p) regime.
    ///
    /// Replaces the textbook p×p Sw / Sb formulation with an n×p centred within-class
    /// matrix and a c×p between-class matrix. Avoids materialising p×p scatters at all,
    /// so cost drops from O(p^4) (the dense formulation's quartic four-deep loop) to
    /// O(n·p·rank). For n=10, p=8319 this is a ~10^9× reduction.
    ///
    /// Algorithm (analogous to scikit-learn's `LinearDiscriminantAnalysis(solver="svd")`):
    /// 1. Stack centred-within-class rows: Xw_i = X_i - mu_{class(i)}.  Shape (n, p).
    /// 2. Thin SVD of Xw = U_w · diag(S_w) · Vt_w with full_matrices=false.
    ///    Truncate to rank_w = #{ S_w > tol }.  (rank_w <= n - c.)
    /// 3. Whitening matrix W = Vt_w[:rank_w].T / S_w[:rank_w]   (shape (p, rank_w)).
    /// 4. Weighted between-class deviations: Xb_c = sqrt(N_c) · (mu_c - mu).
    ///    Shape (c, p), rank <= c - 1.
    /// 5. Project Xb through whitening: M = Xb · W                (shape (c, rank_w)).
    /// 6. Thin SVD of M = U_b · diag(S_b) · Vt_b.
    /// 7. Final scalings = W · Vt_b.T                              (shape (p, rank_w)),
    ///    take its first n_components columns transposed into row-major components.
    /// 8. eigenvalues are S_b[:n_components]^2 (Fisher discriminant ratios).
    #[allow(clippy::too_many_arguments)]
    fn fit_svd_small_side(
        &self,
        x_f64: &Array2<f64>,
        class_means: &Array2<f64>,
        global_mean: &Array1<f64>,
        class_indices: &[usize],
        class_counts: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Step 1: build n×p centred within-class matrix
        let mut xw = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let c = class_indices[i];
            for j in 0..n_features {
                xw[[i, j]] = x_f64[[i, j]] - class_means[[c, j]];
            }
        }

        // Step 2: thin SVD of Xw.  Vt_w shape (rank, p).
        let (_u_w, s_w, vt_w) =
            svd::<f64>(&xw.view(), false, None).map_err(TransformError::LinalgError)?;

        // Determine numerical rank
        let tol = s_w.iter().cloned().fold(0.0_f64, f64::max)
            * (n_samples.max(n_features) as f64)
            * f64::EPSILON;
        let rank_w = s_w.iter().filter(|&&v| v > tol).count();
        if rank_w == 0 {
            return Err(TransformError::ComputationError(
                "Within-class scatter has zero rank (all samples in same point per class)"
                    .to_string(),
            ));
        }

        // Step 3: whitening matrix W = Vt_w[:rank_w].T * diag(1/S_w[:rank_w])  shape (p, rank_w)
        let mut w = Array2::<f64>::zeros((n_features, rank_w));
        for k in 0..rank_w {
            let inv_s = 1.0 / s_w[k];
            for j in 0..n_features {
                w[[j, k]] = vt_w[[k, j]] * inv_s;
            }
        }

        // Step 4: weighted between-class deviations  Xb shape (c, p)
        let mut xb = Array2::<f64>::zeros((n_classes, n_features));
        for c in 0..n_classes {
            let sqrt_nc = (class_counts[c] as f64).sqrt();
            for j in 0..n_features {
                xb[[c, j]] = sqrt_nc * (class_means[[c, j]] - global_mean[j]);
            }
        }

        // Step 5: project Xb through whitening:  M = Xb · W   shape (c, rank_w)
        let m = xb.dot(&w);

        // Step 6: thin SVD of M.  Vt_b shape (rank_b, rank_w).
        let (_u_b, s_b, vt_b) =
            svd::<f64>(&m.view(), false, None).map_err(TransformError::LinalgError)?;

        let rank_b = s_b.len().min(vt_b.nrows());
        if self.n_components > rank_b {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} exceeds the rank of the between-class scatter ({})",
                self.n_components, rank_b
            )));
        }

        // Step 7: components = (W · Vt_b.T)[:, :n_components].T   shape (n_components, p)
        let mut components = Array2::<f64>::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            for j in 0..n_features {
                let mut acc = 0.0;
                for r in 0..rank_w {
                    acc += w[[j, r]] * vt_b[[k, r]];
                }
                components[[k, j]] = acc;
            }
        }

        // Step 8: eigenvalues = S_b^2 (discriminant ratios; sum is total separability)
        let mut eigenvalues = Array1::<f64>::zeros(self.n_components);
        for k in 0..self.n_components {
            eigenvalues[k] = s_b[k] * s_b[k];
        }

        Ok((components, eigenvalues))
    }

    /// Transforms the input data using the fitted LDA model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "LDA model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().expect("Operation failed");

        if n_features != components.shape()[1] {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but LDA was fitted with {} features",
                n_features,
                components.shape()[1]
            )));
        }

        // Project data onto LDA components
        let mut transformed = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut dot_product = 0.0;
                for k in 0..n_features {
                    dot_product += x_f64[[i, k]] * components[[j, k]];
                }
                transformed[[i, j]] = dot_product;
            }
        }

        Ok(transformed)
    }

    /// Fits the LDA model to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    /// * `y` - The target labels, shape (n_samples,)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<Array2<f64>>
    where
        S1: Data,
        S2: Data,
        S1::Elem: Float + NumCast,
        S2::Elem: Copy + NumCast + Eq + std::hash::Hash,
    {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Returns the LDA components
    ///
    /// # Returns
    /// * `Option<&Array2<f64>>` - The LDA components, shape (n_components, n_features)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Returns the explained variance ratio
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_pca_transform() {
        // Create a simple dataset
        let x = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("Operation failed");

        // Initialize and fit PCA with 2 components
        let mut pca = PCA::new(2, true, false);
        let x_transformed = pca.fit_transform(&x).expect("Operation failed");

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[4, 2]);

        // Check that we have the correct number of explained variance components
        let explained_variance = pca.explained_variance_ratio().expect("Operation failed");
        assert_eq!(explained_variance.len(), 2);

        // Check that the sum is a valid number (we don't need to enforce sum = 1)
        assert!(explained_variance.sum() > 0.0 && explained_variance.sum().is_finite());
    }

    #[test]
    fn test_truncated_svd() {
        // Create a simple dataset
        let x = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("Operation failed");

        // Initialize and fit TruncatedSVD with 2 components
        let mut svd = TruncatedSVD::new(2);
        let x_transformed = svd.fit_transform(&x).expect("Operation failed");

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[4, 2]);

        // Check that we have the correct number of explained variance components
        let explained_variance = svd.explained_variance_ratio().expect("Operation failed");
        assert_eq!(explained_variance.len(), 2);

        // Check that the sum is a valid number (we don't need to enforce sum = 1)
        assert!(explained_variance.sum() > 0.0 && explained_variance.sum().is_finite());
    }

    #[test]
    fn test_lda() {
        // Create a simple dataset with 2 classes
        let x = Array::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 4.0],
        )
        .expect("Operation failed");

        let y = Array::from_vec(vec![0, 0, 0, 1, 1, 1]);

        // Initialize and fit LDA with 1 component (max for 2 classes)
        let mut lda = LDA::new(1, "svd").expect("Operation failed");
        let x_transformed = lda.fit_transform(&x, &y).expect("Operation failed");

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[6, 1]);

        // Check that the explained variance ratio is 1.0 for a single component
        let explained_variance = lda.explained_variance_ratio().expect("Operation failed");
        assert_abs_diff_eq!(explained_variance[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lda_eigen_solver() {
        // Create a simple dataset with 3 classes
        let x = Array::from_shape_vec(
            (9, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, // Class 0
                5.0, 4.0, 6.0, 5.0, 7.0, 4.0, // Class 1
                9.0, 8.0, 10.0, 9.0, 11.0, 10.0, // Class 2
            ],
        )
        .expect("Operation failed");

        let y = Array::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

        // Test eigen solver
        let mut lda_eigen = LDA::new(2, "eigen").expect("Operation failed"); // 2 components for 3 classes
        let x_transformed_eigen = lda_eigen.fit_transform(&x, &y).expect("Operation failed");

        // Test SVD solver for comparison
        let mut lda_svd = LDA::new(2, "svd").expect("Operation failed");
        let x_transformed_svd = lda_svd.fit_transform(&x, &y).expect("Operation failed");

        // Check that both transformations have correct shape
        assert_eq!(x_transformed_eigen.shape(), &[9, 2]);
        assert_eq!(x_transformed_svd.shape(), &[9, 2]);

        // Check that both produce valid results
        assert!(x_transformed_eigen.iter().all(|&x| x.is_finite()));
        assert!(x_transformed_svd.iter().all(|&x| x.is_finite()));

        // Check that explained variance ratios are valid for both solvers
        let explained_variance_eigen = lda_eigen
            .explained_variance_ratio()
            .expect("Operation failed");
        let explained_variance_svd = lda_svd
            .explained_variance_ratio()
            .expect("Operation failed");

        assert_eq!(explained_variance_eigen.len(), 2);
        assert_eq!(explained_variance_svd.len(), 2);

        // Both should sum to approximately 1.0
        assert_abs_diff_eq!(explained_variance_eigen.sum(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(explained_variance_svd.sum(), 1.0, epsilon = 1e-10);

        // Eigenvalues should be non-negative
        assert!(explained_variance_eigen.iter().all(|&x| x >= 0.0));
        assert!(explained_variance_svd.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_lda_invalid_solver() {
        let result = LDA::new(1, "invalid");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("solver must be 'svd' or 'eigen'"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Regression tests for issue #124
    //
    // PCA and LDA on wide data (n_samples << n_features) used to be O(p^3) /
    // O(p^4) because the full-matrices SVD path extended Vᵀ to p×p via Gram–
    // Schmidt, and the LDA `svd` solver materialised p×p Sw / Sb scatters. The
    // fixes (thin SVD in PCA, small-side algorithm in LDA::fit) bring this to
    // O(n·p·rank). We verify both correctness and that the wide path completes
    // quickly enough for tests; the timing budget is generous so it isn't flaky.
    // ─────────────────────────────────────────────────────────────────────────

    fn issue_124_synthetic_data(
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
    ) -> (Array2<f64>, scirs2_core::ndarray::Array1<i32>) {
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            ((i * 7919 + j) % 97) as f64 / 97.0
        });
        let y = scirs2_core::ndarray::Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);
        (x, y)
    }

    #[test]
    fn test_issue_124_pca_wide_data_completes_quickly() {
        // n_samples << n_features regime: 12 samples × 400 features. Pre-fix, this
        // path runs the full-matrices SVD that extends Vᵀ to 400×400 via Gram–Schmidt
        // — measurable seconds. Post-fix, it should be <100 ms.
        let (x, _) = issue_124_synthetic_data(12, 400, 3);
        let start = std::time::Instant::now();
        let mut pca = PCA::new(2, true, false);
        let result = pca
            .fit_transform(&x)
            .expect("PCA on wide data must succeed");
        let elapsed = start.elapsed();

        assert_eq!(result.shape(), &[12, 2]);
        assert!(
            result.iter().all(|v| v.is_finite()),
            "PCA wide output must be finite"
        );
        // Generous bound to keep CI stable: pre-fix this took seconds, post-fix < 1s
        // is the expected behaviour on any reasonable machine. 5 s gives ample margin.
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "PCA on 12x400 wide data took {:.3}s — far slower than expected; \
             possible regression of issue #124",
            elapsed.as_secs_f64()
        );

        // Reconstruction sanity: shape is correct and singular values are sorted.
        let sv = pca
            .singular_values
            .as_ref()
            .expect("singular values present");
        assert!(sv.len() == 2);
        assert!(sv[0] >= sv[1]);
    }

    #[test]
    fn test_issue_124_pca_validates_n_components_against_min_dim() {
        // n_components > min(n_samples, n_features) must now error rather than
        // silently zero-pad the trailing components (legacy behaviour).
        let (x, _) = issue_124_synthetic_data(5, 100, 2);
        // min(5, 100) = 5; 6 components is invalid
        let mut pca_bad = PCA::new(6, true, false);
        let err = pca_bad
            .fit(&x)
            .expect_err("PCA with n_components > min(n,p) must fail");
        assert!(
            err.to_string().contains("min(n_samples, n_features)"),
            "error message should reference min(n_samples, n_features): {err}"
        );

        // The boundary case (n_components == min) must still succeed.
        let mut pca_ok = PCA::new(5, true, false);
        let r = pca_ok
            .fit_transform(&x)
            .expect("PCA with n_components == min(n,p) must succeed");
        assert_eq!(r.shape(), &[5, 5]);
    }

    #[test]
    fn test_issue_124_lda_wide_data_svd_solver_completes_quickly() {
        // 12 samples × 400 features × 3 classes — the wide regime that the small-side
        // svd solver now handles in O(n·p·rank). Pre-fix, this path materialised two
        // 400×400 scatter matrices and a four-deep loop ⇒ O(p^4).
        let (x, y) = issue_124_synthetic_data(12, 400, 3);

        let start = std::time::Instant::now();
        let mut lda = LDA::new(2, "svd").expect("LDA::new");
        let z = lda
            .fit_transform(&x, &y)
            .expect("LDA svd on wide data must succeed");
        let elapsed = start.elapsed();

        assert_eq!(z.shape(), &[12, 2]);
        assert!(z.iter().all(|v| v.is_finite()));
        // Pre-fix this was many seconds; post-fix < 1 s. 5 s gives ample CI margin.
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "LDA(svd) on 12x400 wide data took {:.3}s — possible regression of issue #124",
            elapsed.as_secs_f64()
        );

        // Components should have unit-ish norm (small-side variant returns whitened
        // projections, not strictly unit-norm, but each row should be finite & non-zero).
        let components = lda.components().expect("components present");
        assert_eq!(components.shape(), &[2, 400]);
        for row in components.outer_iter() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm.is_finite() && norm > 0.0, "component norm = {norm}");
        }
    }

    #[test]
    fn test_issue_124_lda_svd_small_side_separates_classes() {
        // Correctness: in the wide regime, the small-side SVD solver must still
        // produce projections that separate classes. Use 3 well-separated Gaussian-
        // ish clusters in 200-dim space with n=9 samples.
        let mut data = vec![0.0; 9 * 200];
        for i in 0..9 {
            let class = i / 3;
            let center = match class {
                0 => 0.0,
                1 => 10.0,
                _ => -10.0,
            };
            for j in 0..200 {
                // small per-sample jitter + per-class centre offset on the first 10 dims
                let jitter = ((i * 31 + j) % 7) as f64 * 0.01;
                let centre = if j < 10 { center } else { 0.0 };
                data[i * 200 + j] = centre + jitter;
            }
        }
        let x = Array2::from_shape_vec((9, 200), data).expect("shape ok");
        let y = scirs2_core::ndarray::Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

        let mut lda = LDA::new(2, "svd").expect("LDA::new");
        let z = lda
            .fit_transform(&x, &y)
            .expect("LDA svd on wide data must succeed");
        assert_eq!(z.shape(), &[9, 2]);

        // Within-class spread should be much smaller than between-class spread on
        // the leading discriminant axis.
        let class_means: Vec<f64> = (0..3)
            .map(|c| {
                let idxs: Vec<usize> = (0..9).filter(|&i| (i / 3) == c).collect();
                idxs.iter().map(|&i| z[[i, 0]]).sum::<f64>() / idxs.len() as f64
            })
            .collect();
        let mut between: f64 = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                between += (class_means[a] - class_means[b]).powi(2);
            }
        }
        let between = between.sqrt();
        let within: f64 = (0..9)
            .map(|i| (z[[i, 0]] - class_means[i / 3]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            between > 5.0 * within,
            "between/within ratio = {} (between={}, within={}) — class separation too weak",
            between / within.max(1e-12),
            between,
            within,
        );
    }
}
