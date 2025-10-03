//! Discriminant Analysis
//!
//! This module provides implementations of Linear Discriminant Analysis (LDA) and
//! Quadratic Discriminant Analysis (QDA) for classification and dimensionality reduction.

use crate::error::{StatsError, StatsResult as Result};
use crate::error_handling_v2::ErrorCode;
use crate::{unified_error_handling::global_error_handler, validate_or_error};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Linear Discriminant Analysis (LDA)
///
/// LDA is a dimensionality reduction technique that finds linear combinations of features
/// that best separate different classes. It assumes that all classes have the same
/// covariance structure.
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Solver type for eigenvalue decomposition
    pub solver: LDASolver,
    /// Whether to shrink the covariance estimate
    pub shrinkage: Option<f64>,
    /// Number of components to keep (None = automatic)
    pub n_components: Option<usize>,
    /// Prior probabilities for each class (None = empirical)
    pub priors: Option<Array1<f64>>,
    /// Store training fit results
    pub store_covariance: bool,
}

/// Solver methods for LDA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LDASolver {
    /// SVD-based solver (most stable)
    Svd,
    /// Eigenvalue decomposition (faster for small problems)
    Eigen,
}

/// Result of Linear Discriminant Analysis
#[derive(Debug, Clone)]
pub struct LDAResult {
    /// Linear discriminant coefficients (scalings)
    pub scalings: Array2<f64>,
    /// Intercepts for each class
    pub intercept: Array1<f64>,
    /// Pooled covariance matrix
    pub covariance: Option<Array2<f64>>,
    /// Class means
    pub means: Array2<f64>,
    /// Prior probabilities for each class
    pub priors: Array1<f64>,
    /// Class labels
    pub classes: Array1<i32>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<f64>,
    /// Number of features used for training
    pub n_features: usize,
}

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self {
        Self {
            solver: LDASolver::Svd,
            shrinkage: None,
            n_components: None,
            priors: None,
            store_covariance: true,
        }
    }
}

impl LinearDiscriminantAnalysis {
    /// Create a new LDA instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the solver type
    pub fn with_solver(mut self, solver: LDASolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set shrinkage parameter for covariance regularization
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = Some(shrinkage);
        self
    }

    /// Set number of components to keep
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set prior probabilities
    pub fn with_priors(mut self, priors: Array1<f64>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Set whether to store covariance matrix
    pub fn with_store_covariance(mut self, store: bool) -> Self {
        self.store_covariance = store;
        self
    }

    /// Fit the LDA model
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<LDAResult> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "LDA fit");

        let (n_samples, n_features) = x.dim();
        let n_targets = y.len();

        if n_samples != n_targets {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "LDA fit",
                    "samplesize_mismatch",
                    format!("x: {}, y: {}", n_samples, n_targets),
                    "Number of samples in X and y must be equal",
                )
                .error);
        }

        if n_samples < 2 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2003,
                    "LDA fit",
                    "n_samples",
                    n_samples,
                    "LDA requires at least 2 samples",
                )
                .error);
        }

        // Get unique classes and validate
        let unique_classes = self.get_unique_classes(y)?;
        let n_classes = unique_classes.len();

        if n_classes < 2 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "LDA fit",
                    "n_classes",
                    n_classes,
                    "LDA requires at least 2 classes",
                )
                .error);
        }

        if n_features >= n_samples && self.solver == LDASolver::Eigen {
            return Err(handler
                .create_error(
                    ErrorCode::E1001,
                    "LDA fit",
                    "Use SVD solver when n_features >= n_samples for numerical stability",
                )
                .error);
        }

        // Compute class statistics
        let (class_means, class_priors, class_counts) =
            self.compute_class_statistics(x, y, &unique_classes)?;

        // Compute within-class and between-class scatter matrices
        let (sw, sb) = self.compute_scatter_matrices(x, y, &unique_classes, &class_means)?;

        // Apply shrinkage if specified
        let sw_regularized = if let Some(shrinkage) = self.shrinkage {
            self.apply_shrinkage(&sw, shrinkage)?
        } else {
            sw
        };

        // Solve generalized eigenvalue problem
        let (scalings, explained_variance_ratio) =
            self.solve_eigenvalue_problem(&sw_regularized, &sb)?;

        // Limit number of components
        let n_components = self
            .n_components
            .unwrap_or(n_classes - 1)
            .min(n_classes - 1)
            .min(n_features);

        let final_scalings = scalings
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();
        let final_explained_variance = explained_variance_ratio
            .slice(scirs2_core::ndarray::s![..n_components])
            .to_owned();

        // Compute intercept
        let intercept = self.compute_intercept(&class_means, &final_scalings, &class_priors)?;

        Ok(LDAResult {
            scalings: final_scalings,
            intercept,
            covariance: if self.store_covariance {
                Some(sw_regularized)
            } else {
                None
            },
            means: class_means,
            priors: class_priors,
            classes: unique_classes,
            explained_variance_ratio: final_explained_variance,
            n_features,
        })
    }

    /// Get unique classes from target array
    fn get_unique_classes(&self, y: ArrayView1<i32>) -> Result<Array1<i32>> {
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        Ok(Array1::from_vec(classes))
    }

    /// Compute class means, priors, and counts
    fn compute_class_statistics(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        classes: &Array1<i32>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<usize>)> {
        let (n_samples, n_features) = x.dim();
        let n_classes = classes.len();

        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::zeros(n_classes);

        // Compute class means and counts
        for (i, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<_> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            if class_indices.is_empty() {
                return Err(StatsError::InvalidArgument(format!(
                    "Class {} has no samples",
                    class_label
                )));
            }

            class_counts[i] = class_indices.len();

            // Compute mean for this class
            let mut sum = Array1::zeros(n_features);
            for &idx in &class_indices {
                sum += &x.row(idx);
            }
            class_means
                .row_mut(i)
                .assign(&(sum / class_indices.len() as f64));
        }

        // Compute priors
        let class_priors = if let Some(ref priors) = self.priors {
            if priors.len() != n_classes {
                return Err(StatsError::InvalidArgument(format!(
                    "Priors length ({}) must equal number of classes ({})",
                    priors.len(),
                    n_classes
                )));
            }
            priors.clone()
        } else {
            // Empirical priors
            class_counts.mapv(|count| count as f64 / n_samples as f64)
        };

        Ok((class_means, class_priors, class_counts.mapv(|x| x)))
    }

    /// Compute within-class (Sw) and between-class (Sb) scatter matrices
    fn compute_scatter_matrices(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        classes: &Array1<i32>,
        class_means: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (_n_samples, n_features) = x.dim();
        let _n_classes = classes.len();

        // Overall mean
        let overall_mean = x.mean_axis(Axis(0)).unwrap();

        // Initialize scatter matrices
        let mut sw = Array2::zeros((n_features, n_features));
        let mut sb = Array2::zeros((n_features, n_features));

        // Compute within-class scatter
        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mean = class_means.row(class_idx);

            for (sample_idx, &sample_label) in y.iter().enumerate() {
                if sample_label == class_label {
                    let sample = x.row(sample_idx);
                    let diff = &sample - &class_mean;

                    // Outer product: diff^T * diff
                    for i in 0..n_features {
                        for j in 0..n_features {
                            sw[[i, j]] += diff[i] * diff[j];
                        }
                    }
                }
            }
        }

        // Compute between-class scatter
        for (class_idx, _) in classes.iter().enumerate() {
            let class_mean = class_means.row(class_idx);
            let class_count = y
                .iter()
                .filter(|&&label| label == classes[class_idx])
                .count() as f64;
            let diff = &class_mean - &overall_mean;

            // Weighted outer product
            for i in 0..n_features {
                for j in 0..n_features {
                    sb[[i, j]] += class_count * diff[i] * diff[j];
                }
            }
        }

        Ok((sw, sb))
    }

    /// Apply shrinkage regularization to covariance matrix
    fn apply_shrinkage(&self, sw: &Array2<f64>, shrinkage: f64) -> Result<Array2<f64>> {
        let n_features = sw.nrows();
        let trace = (0..n_features).map(|i| sw[[i, i]]).sum::<f64>();
        let scaled_identity = Array2::eye(n_features) * (trace / n_features as f64);

        Ok((1.0 - shrinkage) * sw + shrinkage * scaled_identity)
    }

    /// Solve the generalized eigenvalue problem Sb * v = λ * Sw * v
    fn solve_eigenvalue_problem(
        &self,
        sw: &Array2<f64>,
        sb: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        match self.solver {
            LDASolver::Svd => self.solve_svd(sw, sb),
            LDASolver::Eigen => self.solve_eigen(sw, sb),
        }
    }

    /// SVD-based solver (more numerically stable)
    fn solve_svd(&self, sw: &Array2<f64>, sb: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        // Cholesky decomposition of Sw = L * L^T
        let l = scirs2_linalg::cholesky(&sw.view(), None).map_err(|e| {
            StatsError::ComputationError(format!(
                "Cholesky decomposition failed: {}. Try using shrinkage.",
                e
            ))
        })?;

        // Solve L * M = Sb for M
        let l_inv = scirs2_linalg::inv(&l.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert Cholesky factor: {}", e))
        })?;

        let m = l_inv.dot(sb).dot(&l_inv.t());

        // SVD of M
        let (u, s, vt) = m
            .svd(true, false)
            .map_err(|e| StatsError::ComputationError(format!("SVD failed: {}", e)))?;

        let u = u.unwrap();
        let s = s;

        // Transform back: scalings = L^{-T} * U
        let scalings = l_inv.t().dot(&u);

        // Sort by eigenvalues (singular values in descending order)
        let mut eigen_pairs: Vec<_> = s.iter().cloned().zip(scalings.columns()).collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val_, _)| *val_).collect();
        let eigenvectors: Array2<f64> = Array2::from_shape_vec(
            (scalings.nrows(), eigenvalues.len()),
            eigen_pairs
                .iter()
                .flat_map(|(_, vec)| vec.iter().cloned())
                .collect(),
        )
        .map_err(|e| {
            StatsError::ComputationError(format!("Failed to construct eigenvector matrix: {}", e))
        })?;

        // Compute explained variance ratio
        let total_variance: f64 = eigenvalues.iter().sum();
        let explained_variance_ratio = if total_variance > 1e-10 {
            Array1::from_vec(
                eigenvalues
                    .iter()
                    .map(|&val| val / total_variance)
                    .collect(),
            )
        } else {
            Array1::zeros(eigenvalues.len())
        };

        Ok((eigenvectors, explained_variance_ratio))
    }

    /// Eigenvalue-based solver
    fn solve_eigen(
        &self,
        sw: &Array2<f64>,
        sb: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::Eigh;

        // Compute Sw^{-1} * Sb
        let sw_inv = scirs2_linalg::inv(&sw.view(), None).map_err(|e| {
            StatsError::ComputationError(format!(
                "Failed to invert within-class scatter matrix: {}. Try using shrinkage.",
                e
            ))
        })?;

        let a = sw_inv.dot(sb);

        // Eigenvalue decomposition
        let (eigenvalues, eigenvectors) = a
            .eigh(scirs2_core::ndarray::ndarray_linalg::UPLO::Upper)
            .map_err(|e| {
                StatsError::ComputationError(format!("Eigenvalue decomposition failed: {}", e))
            })?;

        // Sort in descending order
        let mut eigen_pairs: Vec<_> = eigenvalues
            .iter()
            .cloned()
            .zip(eigenvectors.columns())
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let sorted_eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val_, _)| *val_).collect();
        let sorted_eigenvectors: Array2<f64> = Array2::from_shape_vec(
            (eigenvectors.nrows(), sorted_eigenvalues.len()),
            eigen_pairs
                .iter()
                .flat_map(|(_, vec)| vec.iter().cloned())
                .collect(),
        )
        .map_err(|e| {
            StatsError::ComputationError(format!("Failed to construct eigenvector matrix: {}", e))
        })?;

        // Compute explained variance ratio
        let total_variance: f64 = sorted_eigenvalues.iter().filter(|&&val| val > 0.0).sum();
        let explained_variance_ratio = if total_variance > 1e-10 {
            Array1::from_vec(
                sorted_eigenvalues
                    .iter()
                    .map(|&val| if val > 0.0 { val / total_variance } else { 0.0 })
                    .collect(),
            )
        } else {
            Array1::zeros(sorted_eigenvalues.len())
        };

        Ok((sorted_eigenvectors, explained_variance_ratio))
    }

    /// Compute intercept for decision function
    fn compute_intercept(
        &self,
        class_means: &Array2<f64>,
        scalings: &Array2<f64>,
        priors: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n_classes = class_means.nrows();
        let mut intercept = Array1::zeros(n_classes);

        for i in 0..n_classes {
            let class_mean = class_means.row(i);
            let projected_mean = scalings.t().dot(&class_mean.to_owned());
            let prior_term = priors[i].ln();

            // Intercept = log(prior) - 0.5 * mean^T * Sigma^{-1} * mean
            intercept[i] = prior_term - 0.5 * projected_mean.dot(&projected_mean);
        }

        Ok(intercept)
    }

    /// Transform data to discriminant space
    pub fn transform(&self, x: ArrayView2<f64>, result: &LDAResult) -> Result<Array2<f64>> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "LDA transform");

        if x.ncols() != result.n_features {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "LDA transform",
                    "n_features",
                    format!("input: {}, expected: {}", x.ncols(), result.n_features),
                    "Number of features must match training data",
                )
                .error);
        }

        Ok(x.dot(&result.scalings))
    }

    /// Predict class labels
    pub fn predict(&self, x: ArrayView2<f64>, result: &LDAResult) -> Result<Array1<i32>> {
        let scores = self.decision_function(x, result)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in scores.rows().into_iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            predictions[i] = result.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Compute decision function scores
    pub fn decision_function(&self, x: ArrayView2<f64>, result: &LDAResult) -> Result<Array2<f64>> {
        let projected = self.transform(x, result)?;
        let n_samples = projected.nrows();
        let n_classes = result.classes.len();

        let mut scores = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let sample = projected.row(i);
            for j in 0..n_classes {
                let class_mean = result.means.row(j);
                let projected_class_mean = result.scalings.t().dot(&class_mean.to_owned());

                // Linear discriminant function
                scores[[i, j]] = sample.dot(&projected_class_mean) + result.intercept[j];
            }
        }

        Ok(scores)
    }

    /// Compute prediction probabilities using softmax
    pub fn predict_proba(&self, x: ArrayView2<f64>, result: &LDAResult) -> Result<Array2<f64>> {
        let scores = self.decision_function(x, result)?;
        let mut probabilities = Array2::zeros(scores.dim());

        for (i, mut row) in probabilities.rows_mut().into_iter().enumerate() {
            let score_row = scores.row(i);
            let max_score = score_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute softmax (numerically stable)
            let mut sum_exp = 0.0;
            for (j, &score) in score_row.iter().enumerate() {
                let exp_score = (score - max_score).exp();
                row[j] = exp_score;
                sum_exp += exp_score;
            }

            // Normalize
            if sum_exp > 1e-10 {
                row /= sum_exp;
            } else {
                // Uniform distribution if all scores are very negative
                row.fill(1.0 / row.len() as f64);
            }
        }

        Ok(probabilities)
    }
}

/// Quadratic Discriminant Analysis (QDA)
///
/// QDA is similar to LDA but allows different covariance matrices for each class.
/// This makes it more flexible but requires more parameters.
#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    /// Prior probabilities for each class (None = empirical)
    pub priors: Option<Array1<f64>>,
    /// Regularization parameter for covariance matrices
    pub reg_param: f64,
    /// Store covariances during training
    pub store_covariance: bool,
}

/// Result of Quadratic Discriminant Analysis
#[derive(Debug, Clone)]
pub struct QDAResult {
    /// Covariance matrices for each class
    pub covariances: Option<Vec<Array2<f64>>>,
    /// Class means
    pub means: Array2<f64>,
    /// Prior probabilities for each class
    pub priors: Array1<f64>,
    /// Class labels
    pub classes: Array1<i32>,
    /// Number of features used for training
    pub n_features: usize,
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self {
            priors: None,
            reg_param: 0.0,
            store_covariance: true,
        }
    }
}

impl QuadraticDiscriminantAnalysis {
    /// Create a new QDA instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set prior probabilities
    pub fn with_priors(mut self, priors: Array1<f64>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Set regularization parameter
    pub fn with_reg_param(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set whether to store covariance matrices
    pub fn with_store_covariance(mut self, store: bool) -> Self {
        self.store_covariance = store;
        self
    }

    /// Fit the QDA model
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<i32>) -> Result<QDAResult> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "QDA fit");

        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "QDA fit",
                    "samplesize_mismatch",
                    format!("x: {}, y: {}", n_samples, y.len()),
                    "Number of samples in X and y must be equal",
                )
                .error);
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let unique_classes = Array1::from_vec(classes);
        let n_classes = unique_classes.len();

        if n_classes < 2 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "QDA fit",
                    "n_classes",
                    n_classes,
                    "QDA requires at least 2 classes",
                )
                .error);
        }

        // Compute class statistics
        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_covariances = Vec::with_capacity(n_classes);
        let mut class_counts = Array1::zeros(n_classes);

        for (class_idx, &class_label) in unique_classes.iter().enumerate() {
            let class_indices: Vec<_> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            let classsize = class_indices.len();
            if classsize < 2 {
                return Err(handler
                    .create_validation_error(
                        ErrorCode::E2003,
                        "QDA fit",
                        "classsize",
                        classsize,
                        "Each class must have at least 2 samples for covariance estimation",
                    )
                    .error);
            }

            class_counts[class_idx] = classsize;

            // Compute class mean
            let mut classdata = Array2::zeros((classsize, n_features));
            for (i, &sample_idx) in class_indices.iter().enumerate() {
                classdata.row_mut(i).assign(&x.row(sample_idx));
            }

            let class_mean = classdata.mean_axis(Axis(0)).unwrap();
            class_means.row_mut(class_idx).assign(&class_mean);

            // Compute class covariance
            let mut centered = classdata;
            for mut row in centered.rows_mut() {
                row -= &class_mean;
            }

            let mut cov = centered.t().dot(&centered) / (classsize - 1) as f64;

            // Apply regularization
            if self.reg_param > 0.0 {
                let trace = (0..n_features).map(|i| cov[[i, i]]).sum::<f64>();
                let identity_term: Array2<f64> =
                    Array2::eye(n_features) * (self.reg_param * trace / n_features as f64);
                cov = cov + identity_term;
            }

            class_covariances.push(cov);
        }

        // Compute priors
        let class_priors = if let Some(ref priors) = self.priors {
            if priors.len() != n_classes {
                return Err(StatsError::InvalidArgument(format!(
                    "Priors length ({}) must equal number of classes ({})",
                    priors.len(),
                    n_classes
                )));
            }
            priors.clone()
        } else {
            class_counts.mapv(|count| count as f64 / n_samples as f64)
        };

        Ok(QDAResult {
            covariances: if self.store_covariance {
                Some(class_covariances)
            } else {
                None
            },
            means: class_means,
            priors: class_priors,
            classes: unique_classes,
            n_features,
        })
    }

    /// Predict class labels
    pub fn predict(&self, x: ArrayView2<f64>, result: &QDAResult) -> Result<Array1<i32>> {
        let scores = self.decision_function(x, result)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in scores.rows().into_iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            predictions[i] = result.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Compute decision function scores
    pub fn decision_function(&self, x: ArrayView2<f64>, result: &QDAResult) -> Result<Array2<f64>> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "QDA decision_function");

        if x.ncols() != result.n_features {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "QDA decision_function",
                    "n_features",
                    format!("input: {}, expected: {}", x.ncols(), result.n_features),
                    "Number of features must match training data",
                )
                .error);
        }

        if result.covariances.is_none() {
            return Err(StatsError::InvalidArgument(
                "Covariances not stored during training. Set store_covariance=true.".to_string(),
            ));
        }

        let covariances = result.covariances.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = result.classes.len();
        let mut scores = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let class_mean = result.means.row(class_idx);
            let class_cov = &covariances[class_idx];

            // Compute inverse and determinant
            let cov_inv = scirs2_linalg::inv(&class_cov.view(), None).map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to invert covariance matrix for class {}: {}",
                    class_idx, e
                ))
            })?;

            let det_cov = scirs2_linalg::det(&class_cov.view(), None).map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to compute determinant for class {}: {}",
                    class_idx, e
                ))
            })?;

            if det_cov <= 0.0 {
                return Err(StatsError::ComputationError(format!(
                    "Covariance matrix for class {} is not positive definite",
                    class_idx
                )));
            }

            let log_det_term = -0.5 * det_cov.ln();
            let prior_term = result.priors[class_idx].ln();

            for sample_idx in 0..n_samples {
                let sample = x.row(sample_idx);
                let diff = &sample - &class_mean;

                // Quadratic form: (x - μ)^T Σ^{-1} (x - μ)
                let quad_form = diff.dot(&cov_inv.dot(&diff.to_owned()));

                scores[[sample_idx, class_idx]] = prior_term + log_det_term - 0.5 * quad_form;
            }
        }

        Ok(scores)
    }

    /// Compute prediction probabilities
    pub fn predict_proba(&self, x: ArrayView2<f64>, result: &QDAResult) -> Result<Array2<f64>> {
        let scores = self.decision_function(x, result)?;
        let mut probabilities = Array2::zeros(scores.dim());

        for (i, mut row) in probabilities.rows_mut().into_iter().enumerate() {
            let score_row = scores.row(i);
            let max_score = score_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute softmax (numerically stable)
            let mut sum_exp = 0.0;
            for (j, &score) in score_row.iter().enumerate() {
                let exp_score = (score - max_score).exp();
                row[j] = exp_score;
                sum_exp += exp_score;
            }

            // Normalize
            if sum_exp > 1e-10 {
                row /= sum_exp;
            } else {
                row.fill(1.0 / row.len() as f64);
            }
        }

        Ok(probabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lda_basic() {
        // Create non-degenerate data with proper variance in multiple dimensions
        let x = array![
            [1.0, 2.5],
            [2.1, 3.2],
            [2.8, 4.1],
            [6.2, 7.1],
            [7.3, 8.5],
            [8.1, 9.3],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let lda = LinearDiscriminantAnalysis::new();
        let result = lda.fit(x.view(), y.view()).unwrap();

        assert_eq!(result.classes, array![0, 1]);
        assert_eq!(result.means.nrows(), 2);
        assert_eq!(result.means.ncols(), 2);

        // Test prediction
        let predictions = lda.predict(x.view(), &result).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_qda_basic() {
        // Create non-degenerate data with different covariance structures for each class
        let x = array![
            [1.0, 2.5],
            [2.1, 3.2],
            [2.8, 4.1],
            [6.2, 7.1],
            [7.3, 8.5],
            [8.1, 9.3],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let qda = QuadraticDiscriminantAnalysis::new();
        let result = qda.fit(x.view(), y.view()).unwrap();

        assert_eq!(result.classes, array![0, 1]);
        assert_eq!(result.means.nrows(), 2);
        assert_eq!(result.means.ncols(), 2);

        // Test prediction
        let predictions = qda.predict(x.view(), &result).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_lda_transform() {
        // Create non-degenerate 3D data with independent variance in each dimension
        let x = array![
            [1.2, 2.8, 3.1],
            [2.1, 3.5, 4.2],
            [2.9, 4.1, 5.3],
            [6.1, 7.2, 8.5],
            [7.2, 8.3, 9.1],
            [8.3, 9.1, 10.2],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let lda = LinearDiscriminantAnalysis::new();
        let result = lda.fit(x.view(), y.view()).unwrap();

        let transformed = lda.transform(x.view(), &result).unwrap();
        assert_eq!(transformed.nrows(), 6);
        assert!(transformed.ncols() <= result.classes.len() - 1);
    }
}
