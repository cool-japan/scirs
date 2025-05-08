//! Dimensionality reduction techniques
//!
//! This module provides algorithms for reducing the dimensionality of data,
//! which is useful for visualization, feature extraction, and reducing
//! computational complexity.

mod tsne;

pub use crate::reduction::tsne::{trustworthiness, TSNE};

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
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
    pub fn new(n_components: usize, center: bool, scale: bool) -> Self {
        PCA {
            n_components,
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

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

        // Perform SVD
        let (_u, s, vt) = match svd::<f64>(&x_processed.view(), true) {
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
        let total_variance = s.mapv(|s| s * s).sum();
        let explained_variance_ratio = singular_values.mapv(|s| s * s / total_variance);

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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "PCA model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

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
    pub fn new(n_components: usize) -> Self {
        TruncatedSVD {
            n_components,
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

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
        let (_u, s, vt) = match svd::<f64>(&x_f64.view(), true) {
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "TruncatedSVD model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().unwrap();

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
    pub fn new(n_components: usize, solver: &str) -> Result<Self> {
        if solver != "svd" && solver != "eigen" {
            return Err(TransformError::InvalidInput(
                "solver must be 'svd' or 'eigen'".to_string(),
            ));
        }

        Ok(LDA {
            n_components,
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S1::Elem, f64>(x).unwrap_or(0.0));

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
            let label_u64 = num_traits::cast::<S2::Elem, u64>(label).unwrap_or(0);

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

        let max_n_components = n_classes - 1;
        if self.n_components > max_n_components {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_classes-1={}",
                self.n_components, max_n_components
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
            let (u_sw, s_sw, vt_sw) = match svd::<f64>(&sw.view(), true) {
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
            let (u_sb, s_sb, _vt_sb) = match svd::<f64>(&sb_transformed.view(), true) {
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
            // Eigen-based solver
            // TODO: Implement proper generalized eigenvalue solver
            // For now, we'll use the SVD-based approach for all solvers

            // Decompose the within-class scatter matrix
            let (u_sw, s_sw, vt_sw) = match svd::<f64>(&sw.view(), true) {
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
            let (u_sb, s_sb, _vt_sb) = match svd::<f64>(&sb_transformed.view(), true) {
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
        }

        // Compute explained variance ratio
        let total_eigenvalues = eigenvalues.iter().sum::<f64>();
        let explained_variance_ratio = eigenvalues.mapv(|e| e / total_eigenvalues);

        self.components = Some(components);
        self.means = Some(class_means);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "LDA model has not been fitted".to_string(),
            ));
        }

        let components = self.components.as_ref().unwrap();

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
    use ndarray::Array;

    #[test]
    fn test_pca_transform() {
        // Create a simple dataset
        let x = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        // Initialize and fit PCA with 2 components
        let mut pca = PCA::new(2, true, false);
        let x_transformed = pca.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[4, 2]);

        // Check that we have the correct number of explained variance components
        let explained_variance = pca.explained_variance_ratio().unwrap();
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
        .unwrap();

        // Initialize and fit TruncatedSVD with 2 components
        let mut svd = TruncatedSVD::new(2);
        let x_transformed = svd.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[4, 2]);

        // Check that we have the correct number of explained variance components
        let explained_variance = svd.explained_variance_ratio().unwrap();
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
        .unwrap();

        let y = Array::from_vec(vec![0, 0, 0, 1, 1, 1]);

        // Initialize and fit LDA with 1 component (max for 2 classes)
        let mut lda = LDA::new(1, "svd").unwrap();
        let x_transformed = lda.fit_transform(&x, &y).unwrap();

        // Check that the shape is correct
        assert_eq!(x_transformed.shape(), &[6, 1]);

        // Check that the explained variance ratio is 1.0 for a single component
        let explained_variance = lda.explained_variance_ratio().unwrap();
        assert_abs_diff_eq!(explained_variance[0], 1.0, epsilon = 1e-10);
    }
}
