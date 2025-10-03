//! Canonical Correlation Analysis (CCA)
//!
//! CCA finds linear combinations of two sets of variables that are maximally correlated.
//! It's useful for understanding relationships between two multivariate datasets.

use crate::error::{StatsError, StatsResult as Result};
use crate::error_handling_v2::ErrorCode;
use crate::{unified_error_handling::global_error_handler, validate_or_error};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use statrs::statistics::Statistics;

/// Canonical Correlation Analysis
///
/// CCA finds linear combinations of variables in two datasets that have maximum correlation.
/// This is useful for exploring relationships between two multivariate datasets.
#[derive(Debug, Clone)]
pub struct CanonicalCorrelationAnalysis {
    /// Number of canonical components to compute
    pub n_components: Option<usize>,
    /// Whether to scale the data
    pub scale: bool,
    /// Regularization parameter for numerical stability
    pub reg_param: f64,
    /// Maximum number of iterations for iterative algorithms
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

/// Result of Canonical Correlation Analysis
#[derive(Debug, Clone)]
pub struct CCAResult {
    /// Canonical coefficients for first dataset (X)
    pub x_weights: Array2<f64>,
    /// Canonical coefficients for second dataset (Y)
    pub y_weights: Array2<f64>,
    /// Canonical correlations
    pub correlations: Array1<f64>,
    /// Canonical loadings for X (correlations between X variables and X canonical variates)
    pub x_loadings: Array2<f64>,
    /// Canonical loadings for Y (correlations between Y variables and Y canonical variates)
    pub y_loadings: Array2<f64>,
    /// Cross-loadings for X (correlations between X variables and Y canonical variates)
    pub x_cross_loadings: Array2<f64>,
    /// Cross-loadings for Y (correlations between Y variables and X canonical variates)
    pub y_cross_loadings: Array2<f64>,
    /// Means of X variables
    pub x_mean: Array1<f64>,
    /// Means of Y variables
    pub y_mean: Array1<f64>,
    /// Standard deviations of X variables (if scaled)
    pub x_std: Option<Array1<f64>>,
    /// Standard deviations of Y variables (if scaled)
    pub y_std: Option<Array1<f64>>,
    /// Number of components computed
    pub n_components: usize,
    /// Proportion of variance explained in X by each canonical component
    pub x_explained_variance_ratio: Array1<f64>,
    /// Proportion of variance explained in Y by each canonical component
    pub y_explained_variance_ratio: Array1<f64>,
}

impl Default for CanonicalCorrelationAnalysis {
    fn default() -> Self {
        Self {
            n_components: None,
            scale: true,
            reg_param: 1e-6,
            max_iter: 500,
            tol: 1e-8,
        }
    }
}

impl CanonicalCorrelationAnalysis {
    /// Create a new CCA instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of components to compute
    pub fn with_n_components(mut self, ncomponents: usize) -> Self {
        self.n_components = Some(ncomponents);
        self
    }

    /// Set whether to scale the data
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set regularization parameter
    pub fn with_reg_param(mut self, regparam: f64) -> Self {
        self.reg_param = regparam;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, maxiter: usize) -> Self {
        self.max_iter = maxiter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the CCA model
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<CCAResult> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "CCA fit");
        validate_or_error!(finite: y.as_slice().unwrap(), "y", "CCA fit");

        let (n_samples_x, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples_x != n_samples_y {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "CCA fit",
                    "samplesize_mismatch",
                    format!("x: {}, y: {}", n_samples_x, n_samples_y),
                    "X and Y must have the same number of samples",
                )
                .error);
        }

        let n_samples_ = n_samples_x;
        if n_samples_ < 2 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2003,
                    "CCA fit",
                    "n_samples_",
                    n_samples_,
                    "CCA requires at least 2 samples",
                )
                .error);
        }

        if n_features_x == 0 || n_features_y == 0 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2004,
                    "CCA fit",
                    "n_features",
                    format!("x: {}, y: {}", n_features_x, n_features_y),
                    "Both X and Y must have at least one feature",
                )
                .error);
        }

        // Determine number of components
        let max_components = n_features_x.min(n_features_y).min(n_samples_ - 1);
        let n_components = self
            .n_components
            .unwrap_or(max_components)
            .min(max_components);

        if n_components == 0 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "CCA fit",
                    "n_components",
                    n_components,
                    "Number of components must be positive",
                )
                .error);
        }

        // Center and optionally scale the data
        let (x_centered, x_mean, x_std) = self.center_and_scale(x)?;
        let (y_centered, y_mean, y_std) = self.center_and_scale(y)?;

        // Compute cross-covariance and auto-covariance matrices
        let (cxx, cyy, cxy) = self.compute_covariance_matrices(&x_centered, &y_centered)?;

        // Solve the generalized eigenvalue problem
        let (x_weights, y_weights, correlations) =
            self.solve_cca_eigenvalue_problem(&cxx, &cyy, &cxy, n_components)?;

        // Compute loadings and cross-loadings
        let x_canonical = x_centered.dot(&x_weights);
        let y_canonical = y_centered.dot(&y_weights);

        let x_loadings = self.compute_loadings(&x_centered, &x_canonical)?;
        let y_loadings = self.compute_loadings(&y_centered, &y_canonical)?;
        let x_cross_loadings = self.compute_loadings(&x_centered, &y_canonical)?;
        let y_cross_loadings = self.compute_loadings(&y_centered, &x_canonical)?;

        // Compute explained variance ratios
        let x_explained_variance_ratio =
            self.compute_explained_variance(&x_centered, &x_canonical)?;
        let y_explained_variance_ratio =
            self.compute_explained_variance(&y_centered, &y_canonical)?;

        Ok(CCAResult {
            x_weights,
            y_weights,
            correlations,
            x_loadings,
            y_loadings,
            x_cross_loadings,
            y_cross_loadings,
            x_mean,
            y_mean,
            x_std,
            y_std,
            n_components,
            x_explained_variance_ratio,
            y_explained_variance_ratio,
        })
    }

    /// Center and optionally scale data
    fn center_and_scale(
        &self,
        data: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Option<Array1<f64>>)> {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.to_owned();

        // Center data
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        if self.scale {
            // Compute standard deviations
            let mut std_dev = Array1::zeros(data.ncols());
            for j in 0..data.ncols() {
                let col = centered.column(j);
                let variance = col.mapv(|x| x * x).mean();
                std_dev[j] = variance.sqrt().max(1e-10); // Avoid division by zero
            }

            // Scale data
            for mut row in centered.rows_mut() {
                for j in 0..row.len() {
                    row[j] /= std_dev[j];
                }
            }

            Ok((centered, mean, Some(std_dev)))
        } else {
            Ok((centered, mean, None))
        }
    }

    /// Compute covariance matrices
    fn compute_covariance_matrices(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let n_samples_ = x.nrows() as f64;

        // Auto-covariance matrices
        let cxx = x.t().dot(x) / (n_samples_ - 1.0);
        let cyy = y.t().dot(y) / (n_samples_ - 1.0);

        // Cross-covariance matrix
        let cxy = x.t().dot(y) / (n_samples_ - 1.0);

        Ok((cxx, cyy, cxy))
    }

    /// Solve the CCA eigenvalue problem
    fn solve_cca_eigenvalue_problem(
        &self,
        cxx: &Array2<f64>,
        cyy: &Array2<f64>,
        cxy: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        // Regularized versions of covariance matrices
        let cxx_reg = self.regularize_covariance(cxx)?;
        let cyy_reg = self.regularize_covariance(cyy)?;

        // Compute inverse square roots
        let cxx_inv_sqrt = self.compute_inverse_sqrt(&cxx_reg)?;
        let cyy_inv_sqrt = self.compute_inverse_sqrt(&cyy_reg)?;

        // Form the matrix for SVD: Cxx^{-1/2} * Cxy * Cyy^{-1/2}
        let k = cxx_inv_sqrt.dot(cxy).dot(&cyy_inv_sqrt);

        // SVD of K
        let (u, s, vt) = k
            .svd(true, true)
            .map_err(|e| StatsError::ComputationError(format!("SVD failed in CCA: {}", e)))?;

        let u = u.unwrap();
        let vt = vt.unwrap();

        // Extract the desired number of _components
        let n_comp = n_components.min(s.len());
        let correlations = s.slice(scirs2_core::ndarray::s![..n_comp]).to_owned();
        let u_comp = u.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
        let v_comp = vt
            .slice(scirs2_core::ndarray::s![..n_comp, ..])
            .t()
            .to_owned();

        // Transform back to original space
        let x_weights = cxx_inv_sqrt.dot(&u_comp);
        let y_weights = cyy_inv_sqrt.dot(&v_comp);

        Ok((x_weights, y_weights, correlations))
    }

    /// Regularize covariance matrix for numerical stability
    fn regularize_covariance(&self, cov: &Array2<f64>) -> Result<Array2<f64>> {
        if self.reg_param <= 0.0 {
            return Ok(cov.clone());
        }

        let n = cov.nrows();
        let trace = (0..n).map(|i| cov[[i, i]]).sum::<f64>();
        let reg_term: Array2<f64> = Array2::eye(n) * (self.reg_param * trace / n as f64);

        Ok(cov + &reg_term)
    }

    /// Compute inverse square root of a symmetric positive definite matrix
    fn compute_inverse_sqrt(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        use scirs2_core::ndarray::ndarray_linalg::Eigh;

        let (eigenvalues, eigenvectors) = matrix
            .eigh(scirs2_core::ndarray::ndarray_linalg::UPLO::Upper)
            .map_err(|e| {
                StatsError::ComputationError(format!("Eigenvalue decomposition failed: {}", e))
            })?;

        // Check for positive definiteness
        let min_eigenvalue = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_eigenvalue <= 1e-10 {
            return Err(StatsError::ComputationError(format!(
                "Matrix is not positive definite (min eigenvalue: {})",
                min_eigenvalue
            )));
        }

        // Compute inverse square root
        let inv_sqrt_eigenvalues = eigenvalues.mapv(|x| x.sqrt().recip());
        let mut inv_sqrt = Array2::zeros(matrix.dim());

        for i in 0..eigenvalues.len() {
            let eigenvec = eigenvectors.column(i);
            let lambda_inv_sqrt = inv_sqrt_eigenvalues[i];

            for j in 0..matrix.nrows() {
                for k in 0..matrix.ncols() {
                    inv_sqrt[[j, k]] += lambda_inv_sqrt * eigenvec[j] * eigenvec[k];
                }
            }
        }

        Ok(inv_sqrt)
    }

    /// Compute loadings (correlations between original variables and canonical variates)
    fn compute_loadings(
        &self,
        original: &Array2<f64>,
        canonical: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples_ = original.nrows() as f64;
        let n_original = original.ncols();
        let n_canonical = canonical.ncols();

        let mut loadings = Array2::zeros((n_original, n_canonical));

        for i in 0..n_original {
            let orig_var = original.column(i);
            let orig_var_std = (orig_var.mapv(|x| x * x).sum() / (n_samples_ - 1.0)).sqrt();

            for j in 0..n_canonical {
                let canon_var = canonical.column(j);
                let canon_var_std = (canon_var.mapv(|x| x * x).sum() / (n_samples_ - 1.0)).sqrt();

                if orig_var_std > 1e-10 && canon_var_std > 1e-10 {
                    let covariance = orig_var.dot(&canon_var) / (n_samples_ - 1.0);
                    let correlation = covariance / (orig_var_std * canon_var_std);
                    loadings[[i, j]] = correlation;
                }
            }
        }

        Ok(loadings)
    }

    /// Compute explained variance ratio
    fn compute_explained_variance(
        &self,
        original: &Array2<f64>,
        canonical: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n_samples_ = original.nrows() as f64;
        let n_canonical = canonical.ncols();

        // Total variance in original variables
        let total_variance = (0..original.ncols())
            .map(|i| {
                let col = original.column(i);
                col.mapv(|x| x * x).sum() / (n_samples_ - 1.0)
            })
            .sum::<f64>();

        if total_variance <= 1e-10 {
            return Ok(Array1::zeros(n_canonical));
        }

        // Variance explained by each canonical component
        let mut explained_variance = Array1::zeros(n_canonical);
        for j in 0..n_canonical {
            let canon_var = canonical.column(j);
            let canon_variance = canon_var.mapv(|x| x * x).sum() / (n_samples_ - 1.0);
            explained_variance[j] = canon_variance / total_variance;
        }

        Ok(explained_variance)
    }

    /// Transform new data using fitted CCA model
    pub fn transform(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        result: &CCAResult,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "CCA transform");
        validate_or_error!(finite: y.as_slice().unwrap(), "y", "CCA transform");

        if x.ncols() != result.x_mean.len() {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "CCA transform",
                    "x_features",
                    format!("input: {}, expected: {}", x.ncols(), result.x_mean.len()),
                    "X must have the same number of features as training data",
                )
                .error);
        }

        if y.ncols() != result.y_mean.len() {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "CCA transform",
                    "y_features",
                    format!("input: {}, expected: {}", y.ncols(), result.y_mean.len()),
                    "Y must have the same number of features as training data",
                )
                .error);
        }

        // Center and scale X
        let mut x_processed = x.to_owned();
        for mut row in x_processed.rows_mut() {
            row -= &result.x_mean;
        }

        if let Some(ref x_std) = result.x_std {
            for mut row in x_processed.rows_mut() {
                for j in 0..row.len() {
                    row[j] /= x_std[j];
                }
            }
        }

        // Center and scale Y
        let mut y_processed = y.to_owned();
        for mut row in y_processed.rows_mut() {
            row -= &result.y_mean;
        }

        if let Some(ref y_std) = result.y_std {
            for mut row in y_processed.rows_mut() {
                for j in 0..row.len() {
                    row[j] /= y_std[j];
                }
            }
        }

        // Transform to canonical space
        let x_canonical = x_processed.dot(&result.x_weights);
        let y_canonical = y_processed.dot(&result.y_weights);

        Ok((x_canonical, y_canonical))
    }

    /// Compute canonical correlations for new data
    pub fn score(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        result: &CCAResult,
    ) -> Result<Array1<f64>> {
        let (x_canonical, y_canonical) = self.transform(x, y, result)?;
        let n_samples_ = x_canonical.nrows() as f64;
        let n_components = result.n_components;

        let mut correlations = Array1::zeros(n_components);
        for i in 0..n_components {
            let x_comp = x_canonical.column(i);
            let y_comp = y_canonical.column(i);

            let x_std = (x_comp.mapv(|x| x * x).sum() / (n_samples_ - 1.0)).sqrt();
            let y_std = (y_comp.mapv(|x| x * x).sum() / (n_samples_ - 1.0)).sqrt();

            if x_std > 1e-10 && y_std > 1e-10 {
                let covariance = x_comp.dot(&y_comp) / (n_samples_ - 1.0);
                correlations[i] = covariance / (x_std * y_std);
            }
        }

        Ok(correlations)
    }
}

/// Partial Least Squares (PLS) regression variant of CCA
///
/// PLS is similar to CCA but optimized for prediction rather than just correlation.
#[derive(Debug, Clone)]
pub struct PLSCanonical {
    /// Number of components
    pub n_components: usize,
    /// Whether to scale the data
    pub scale: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

/// Result of PLS Canonical analysis
#[derive(Debug, Clone)]
pub struct PLSResult {
    /// X weights
    pub x_weights: Array2<f64>,
    /// Y weights
    pub y_weights: Array2<f64>,
    /// X loadings
    pub x_loadings: Array2<f64>,
    /// Y loadings
    pub y_loadings: Array2<f64>,
    /// X scores
    pub x_scores: Array2<f64>,
    /// Y scores
    pub y_scores: Array2<f64>,
    /// X rotation matrix
    pub x_rotations: Array2<f64>,
    /// Y rotation matrix
    pub y_rotations: Array2<f64>,
    /// Means
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    /// Standard deviations (if scaled)
    pub x_std: Option<Array1<f64>>,
    pub y_std: Option<Array1<f64>>,
}

impl Default for PLSCanonical {
    fn default() -> Self {
        Self {
            n_components: 2,
            scale: true,
            max_iter: 500,
            tol: 1e-6,
        }
    }
}

impl PLSCanonical {
    /// Create new PLS instance
    pub fn new(_ncomponents: usize) -> Self {
        Self {
            n_components: _ncomponents,
            ..Default::default()
        }
    }

    /// Fit PLS model using NIPALS algorithm
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<PLSResult> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "PLS fit");
        validate_or_error!(finite: y.as_slice().unwrap(), "y", "PLS fit");

        let (n_samples_, n_x_features) = x.dim();
        let (n_samples_y, n_y_features) = y.dim();

        if n_samples_ != n_samples_y {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "PLS fit",
                    "samplesize_mismatch",
                    format!("x: {}, y: {}", n_samples_, n_samples_y),
                    "X and Y must have the same number of samples",
                )
                .error);
        }

        // Center and scale data
        let cca = CanonicalCorrelationAnalysis {
            scale: self.scale,
            ..Default::default()
        };
        let (mut x_current, x_mean, x_std) = cca.center_and_scale(x)?;
        let (mut y_current, y_mean, y_std) = cca.center_and_scale(y)?;

        // Initialize result matrices
        let mut x_weights = Array2::zeros((n_x_features, self.n_components));
        let mut y_weights = Array2::zeros((n_y_features, self.n_components));
        let mut x_loadings = Array2::zeros((n_x_features, self.n_components));
        let mut y_loadings = Array2::zeros((n_y_features, self.n_components));
        let mut x_scores = Array2::zeros((n_samples_, self.n_components));
        let mut y_scores = Array2::zeros((n_samples_, self.n_components));

        // NIPALS algorithm
        let mut actual_components = 0;
        for comp in 0..self.n_components {
            // Check if there's sufficient variance left for another component
            let x_var = x_current.iter().map(|&x| x * x).sum::<f64>();
            let y_var = y_current.iter().map(|&y| y * y).sum::<f64>();

            if x_var < 1e-12 || y_var < 1e-12 {
                // Not enough variance left, stop here
                break;
            }

            // Initialize weights with first column of Y
            let mut u = y_current.column(0).to_owned();
            let mut w_old = Array1::zeros(n_x_features);

            let mut converged_inner = false;
            for _iter in 0..self.max_iter {
                // X weights
                let w = x_current.t().dot(&u);
                let w_norm = (w.dot(&w)).sqrt();
                if w_norm < 1e-10 {
                    // No more meaningful components can be extracted
                    converged_inner = false;
                    break;
                }
                let w = w / w_norm;

                // X scores
                let t = x_current.dot(&w);

                // Y weights
                let c = y_current.t().dot(&t);
                let c_norm = (c.dot(&c)).sqrt();
                if c_norm < 1e-10 {
                    return Err(StatsError::ComputationError(
                        "Y weights became zero".to_string(),
                    ));
                }
                let c = c / c_norm;

                // Y scores
                u = y_current.dot(&c);

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    converged_inner = true;
                    break;
                }
                w_old = w.clone();
            }

            // If inner loop didn't converge, skip this component
            if !converged_inner {
                break;
            }

            // Compute loadings
            let w = x_current.t().dot(&u);
            let w_norm = (w.dot(&w)).sqrt();
            if w_norm < 1e-10 {
                break; // Can't extract this component
            }
            let w = w.clone() / w_norm;
            let t = x_current.dot(&w);
            let c = y_current.t().dot(&t);
            let c_norm = (c.dot(&c)).sqrt();
            if c_norm < 1e-10 {
                break; // Can't extract this component
            }
            let c = c.clone() / c_norm;
            let u = y_current.dot(&c);

            let t_dot_t = t.dot(&t);
            let u_dot_u = u.dot(&u);
            if t_dot_t < 1e-10 || u_dot_u < 1e-10 {
                break; // Can't extract this component
            }

            let p = x_current.t().dot(&t) / t_dot_t;
            let q = y_current.t().dot(&u) / u_dot_u;

            // Store results
            x_weights.column_mut(comp).assign(&w);
            y_weights.column_mut(comp).assign(&c);
            x_loadings.column_mut(comp).assign(&p);
            y_loadings.column_mut(comp).assign(&q);
            x_scores.column_mut(comp).assign(&t);
            y_scores.column_mut(comp).assign(&u);

            actual_components += 1;

            // Deflate matrices
            let _tt = Array1::from_vec(vec![t.dot(&t)]);
            let outer_product = &t
                .view()
                .insert_axis(Axis(1))
                .dot(&p.view().insert_axis(Axis(0)));
            x_current -= outer_product;

            let _uu = Array1::from_vec(vec![u.dot(&u)]);
            let outer_product_y = &u
                .view()
                .insert_axis(Axis(1))
                .dot(&q.view().insert_axis(Axis(0)));
            y_current -= outer_product_y;
        }

        // Slice matrices to actual components extracted
        let x_weights = x_weights.slice(s![.., ..actual_components]).to_owned();
        let y_weights = y_weights.slice(s![.., ..actual_components]).to_owned();
        let x_loadings = x_loadings.slice(s![.., ..actual_components]).to_owned();
        let y_loadings = y_loadings.slice(s![.., ..actual_components]).to_owned();
        let x_scores = x_scores.slice(s![.., ..actual_components]).to_owned();
        let y_scores = y_scores.slice(s![.., ..actual_components]).to_owned();

        // Compute rotation matrices only if we have components
        let (x_rotations, y_rotations) = if actual_components > 0 {
            let x_rot = x_weights.dot(
                &scirs2_linalg::inv(&(x_loadings.t().dot(&x_weights)).view(), None).map_err(
                    |e| {
                        StatsError::ComputationError(format!(
                            "Failed to compute X rotations: {}",
                            e
                        ))
                    },
                )?,
            );

            let y_rot = y_weights.dot(
                &scirs2_linalg::inv(&(y_loadings.t().dot(&y_weights)).view(), None).map_err(
                    |e| {
                        StatsError::ComputationError(format!(
                            "Failed to compute Y rotations: {}",
                            e
                        ))
                    },
                )?,
            );
            (x_rot, y_rot)
        } else {
            (
                Array2::zeros((n_x_features, 0)),
                Array2::zeros((n_y_features, 0)),
            )
        };

        Ok(PLSResult {
            x_weights,
            y_weights,
            x_loadings,
            y_loadings,
            x_scores,
            y_scores,
            x_rotations,
            y_rotations,
            x_mean,
            y_mean,
            x_std,
            y_std,
        })
    }

    /// Transform new data
    pub fn transform(&self, x: ArrayView2<f64>, result: &PLSResult) -> Result<Array2<f64>> {
        let handler = global_error_handler();
        validate_or_error!(finite: x.as_slice().unwrap(), "x", "PLS transform");

        if x.ncols() != result.x_mean.len() {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E2001,
                    "PLS transform",
                    "n_features",
                    format!("input: {}, expected: {}", x.ncols(), result.x_mean.len()),
                    "Number of features must match training data",
                )
                .error);
        }

        // Center and scale
        let mut x_processed = x.to_owned();
        for mut row in x_processed.rows_mut() {
            row -= &result.x_mean;
        }

        if let Some(ref x_std) = result.x_std {
            for mut row in x_processed.rows_mut() {
                for j in 0..row.len() {
                    row[j] /= x_std[j];
                }
            }
        }

        Ok(x_processed.dot(&result.x_rotations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cca_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let y = array![
            [2.0, 4.0],
            [4.0, 6.0],
            [6.0, 8.0],
            [8.0, 10.0],
            [10.0, 12.0],
        ];

        let cca = CanonicalCorrelationAnalysis::new().with_n_components(2);
        let result = cca.fit(x.view(), y.view()).unwrap();

        assert_eq!(result.n_components, 2);
        assert_eq!(result.x_weights.ncols(), 2);
        assert_eq!(result.y_weights.ncols(), 2);
        assert_eq!(result.correlations.len(), 2);

        // Test transformation
        let (x_canonical, y_canonical) = cca.transform(x.view(), y.view(), &result).unwrap();
        assert_eq!(x_canonical.nrows(), 5);
        assert_eq!(y_canonical.nrows(), 5);
        assert_eq!(x_canonical.ncols(), 2);
        assert_eq!(y_canonical.ncols(), 2);
    }

    #[test]
    fn test_pls_basic() {
        // Create data with more independent variation to support 2 components
        let x = array![[1.0, 3.0], [2.0, 1.0], [3.0, 4.0], [4.0, 2.0], [5.0, 5.0],];

        let y = array![[2.0, 6.0], [4.0, 2.0], [6.0, 8.0], [8.0, 4.0], [10.0, 10.0],];

        let pls = PLSCanonical::new(2);
        let result = pls.fit(x.view(), y.view()).unwrap();

        assert_eq!(result.x_weights.ncols(), 2);
        assert_eq!(result.y_weights.ncols(), 2);
        assert_eq!(result.x_scores.nrows(), 5);
        assert_eq!(result.y_scores.nrows(), 5);

        // Test transformation
        let transformed = pls.transform(x.view(), &result).unwrap();
        assert_eq!(transformed.nrows(), 5);
        assert_eq!(transformed.ncols(), 2);
    }
}
