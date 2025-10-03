//! Principal Component Analysis (PCA)
//!
//! PCA is a dimensionality reduction technique that finds the directions of maximum variance
//! in high-dimensional data and projects the data onto a lower-dimensional subspace.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::validation::*;

/// Principal Component Analysis
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of components to keep
    pub n_components: Option<usize>,
    /// Whether to use SVD instead of eigendecomposition  
    pub svd_solver: SvdSolver,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data to unit variance
    pub scale: bool,
    /// Random state for randomized solver
    pub random_state: Option<u64>,
}

/// SVD solver type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SvdSolver {
    /// Full SVD
    Full,
    /// Randomized SVD (for large datasets)
    Randomized,
    /// Automatically choose based on data size
    Auto,
}

/// Result of PCA fit
#[derive(Debug, Clone)]
pub struct PCAResult {
    /// Principal components (eigenvectors)
    pub components: Array2<f64>,
    /// Explained variance for each component
    pub explained_variance: Array1<f64>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<f64>,
    /// Singular values corresponding to each component
    pub singular_values: Array1<f64>,
    /// Mean of the training data
    pub mean: Array1<f64>,
    /// Standard deviation of the training data (if scaling was used)
    pub scale: Option<Array1<f64>>,
    /// Number of samples used for fitting
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
}

impl Default for PCA {
    fn default() -> Self {
        Self {
            n_components: None,
            svd_solver: SvdSolver::Auto,
            center: true,
            scale: false,
            random_state: None,
        }
    }
}

impl PCA {
    /// Create a new PCA instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of components to keep
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set the SVD solver
    pub fn with_svd_solver(mut self, solver: SvdSolver) -> Self {
        self.svd_solver = solver;
        self
    }

    /// Enable or disable centering
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Enable or disable scaling
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the PCA model to the data
    pub fn fit(&self, data: ArrayView2<f64>) -> Result<PCAResult> {
        checkarray_finite(&data, "data")?;
        let (n_samples, n_features) = data.dim();
        if n_samples < 2 {
            return Err(StatsError::InvalidArgument(
                "n_samples must be at least 2".to_string(),
            ));
        }
        if n_features < 1 {
            return Err(StatsError::InvalidArgument(
                "n_features must be at least 1".to_string(),
            ));
        }

        // Determine number of components
        let max_components = n_samples.min(n_features);
        let n_components = match self.n_components {
            Some(k) => {
                check_positive(k, "n_components")?;
                if k > max_components {
                    return Err(StatsError::InvalidArgument(format!(
                        "n_components ({}) cannot be larger than min(n_samples, n_features) = {}",
                        k, max_components
                    )));
                }
                k
            }
            None => max_components,
        };

        // Center the data
        let mean = if self.center {
            data.mean_axis(Axis(0)).unwrap()
        } else {
            Array1::zeros(n_features)
        };

        let mut centereddata = data.to_owned();
        if self.center {
            for mut row in centereddata.rows_mut() {
                row -= &mean;
            }
        }

        // Scale the data
        let scale = if self.scale {
            let std = centereddata.std_axis(Axis(0), 1.0);
            // Avoid division by zero
            let std = std.mapv(|s| if s > 1e-10 { s } else { 1.0 });

            for (mut col, &s) in centereddata.columns_mut().into_iter().zip(std.iter()) {
                col /= s;
            }
            Some(std)
        } else {
            None
        };

        // Choose solver
        let solver = match self.svd_solver {
            SvdSolver::Auto => {
                if n_samples >= 500 && n_features >= 500 && n_components < max_components / 2 {
                    SvdSolver::Randomized
                } else {
                    SvdSolver::Full
                }
            }
            solver => solver,
        };

        // Perform PCA
        let result = match solver {
            SvdSolver::Full => self.pca_svd(&centereddata, n_components, n_samples)?,
            SvdSolver::Randomized => self.pca_randomized(&centereddata, n_components, n_samples)?,
            _ => unreachable!(),
        };

        Ok(PCAResult {
            components: result.0,
            explained_variance: result.1,
            explained_variance_ratio: result.2,
            singular_values: result.3,
            mean,
            scale,
            n_samples_: n_samples,
            n_features,
        })
    }

    /// Perform PCA using SVD
    fn pca_svd(
        &self,
        data: &Array2<f64>,
        n_components: usize,
        n_samples: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        // Perform SVD: X = U * S * V^T
        let (_u, s, vt) = data
            .svd(true, true)
            .map_err(|e| StatsError::ComputationError(format!("SVD failed: {}", e)))?;
        let v = vt.unwrap().t().to_owned();

        // Extract _components
        let components = v
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();

        // Compute explained variance
        let singular_values = s.slice(scirs2_core::ndarray::s![..n_components]).to_owned();
        let explained_variance = &singular_values * &singular_values / (n_samples - 1) as f64;

        // Compute explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = &explained_variance / total_variance;

        Ok((
            components.t().to_owned(),
            explained_variance,
            explained_variance_ratio,
            singular_values,
        ))
    }

    /// Perform PCA using randomized SVD
    fn pca_randomized(
        &self,
        data: &Array2<f64>,
        n_components: usize,
        n_samples: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
        use scirs2_core::ndarray::ndarray_linalg::{QR, SVD};
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        use scirs2_core::random::{Distribution, Normal};

        let n_features = data.ncols();
        let n_oversamples = 10.min((n_features - n_components) / 2);
        let n_random = n_components + n_oversamples;

        // Initialize RNG
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                // Use a simple fallback seed based on current time or a fixed seed
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(seed)
            }
        };

        // Generate random matrix
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
        })?;
        let omega = Array2::from_shape_fn((n_features, n_random), |_| normal.sample(&mut rng));

        // Power iterations for better approximation
        let n_iter = 4;
        let mut q = data.dot(&omega);

        for _ in 0..n_iter {
            // QR decomposition
            let (q_mat, r) = q.qr().map_err(|e| {
                StatsError::ComputationError(format!("QR decomposition failed: {}", e))
            })?;
            q = q_mat;

            // Project back
            let z = data.t().dot(&q);
            let (q_mat, r) = z.qr().map_err(|e| {
                StatsError::ComputationError(format!("QR decomposition failed: {}", e))
            })?;
            q = data.dot(&q_mat);
        }

        // Final QR decomposition
        let (q_final, r) = q.qr().map_err(|e| {
            StatsError::ComputationError(format!("Final QR decomposition failed: {}", e))
        })?;

        // Project data onto subspace
        let b = q_final.t().dot(data);

        // SVD of small matrix B
        let (_u_small, s, vt) = b.svd(true, true).map_err(|e| {
            StatsError::ComputationError(format!("SVD of projected matrix failed: {}", e))
        })?;

        let v = vt.unwrap().t().to_owned();

        // Extract _components
        let components = v
            .slice(scirs2_core::ndarray::s![.., ..n_components])
            .to_owned();

        // Compute explained variance
        let singular_values = s.slice(scirs2_core::ndarray::s![..n_components]).to_owned();
        let explained_variance = &singular_values * &singular_values / (n_samples - 1) as f64;

        // Compute explained variance ratio
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = &explained_variance / total_variance;

        Ok((
            components.t().to_owned(),
            explained_variance,
            explained_variance_ratio,
            singular_values,
        ))
    }

    /// Transform data using the fitted PCA model
    pub fn transform(&self, data: ArrayView2<f64>, result: &PCAResult) -> Result<Array2<f64>> {
        checkarray_finite(&data, "data")?;
        if data.ncols() != result.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "data has {} features, expected {}",
                data.ncols(),
                result.n_features
            )));
        }

        let mut transformed = data.to_owned();

        // Center
        if self.center {
            for mut row in transformed.rows_mut() {
                row -= &result.mean;
            }
        }

        // Scale
        if let Some(ref scale) = result.scale {
            for (mut col, &s) in transformed.columns_mut().into_iter().zip(scale.iter()) {
                col /= s;
            }
        }

        // Project onto components
        Ok(transformed.dot(&result.components.t()))
    }

    /// Inverse transform from component space back to original space
    pub fn inverse_transform(
        &self,
        data: ArrayView2<f64>,
        result: &PCAResult,
    ) -> Result<Array2<f64>> {
        checkarray_finite(&data, "data")?;
        let n_components = result.components.nrows();
        if data.ncols() != n_components {
            return Err(StatsError::DimensionMismatch(format!(
                "data has {} components, expected {}",
                data.ncols(),
                n_components
            )));
        }

        // Project back to original space
        let mut reconstructed = data.dot(&result.components);

        // Inverse scale
        if let Some(ref scale) = result.scale {
            for (mut col, &s) in reconstructed.columns_mut().into_iter().zip(scale.iter()) {
                col *= s;
            }
        }

        // Add mean back
        if self.center {
            for mut row in reconstructed.rows_mut() {
                row += &result.mean;
            }
        }

        Ok(reconstructed)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&self, data: ArrayView2<f64>) -> Result<(Array2<f64>, PCAResult)> {
        let result = self.fit(data)?;
        let transformed = self.transform(data, &result)?;
        Ok((transformed, result))
    }
}

/// Compute the optimal number of components using Minka's MLE
#[allow(dead_code)]
pub fn mle_components(data: ArrayView2<f64>, maxcomponents: Option<usize>) -> Result<usize> {
    checkarray_finite(&data, "data")?;
    let (n_samples, n_features) = data.dim();

    let pca = PCA::new().with_n_components(maxcomponents.unwrap_or(n_features.min(n_samples)));
    let result = pca.fit(data)?;

    let eigenvalues = &result.explained_variance;
    let n = n_samples as f64;
    let p = n_features as f64;

    // Minka's MLE for PCA
    let mut best_k = 0;
    let mut best_ll = f64::NEG_INFINITY;

    for k in 0..eigenvalues.len() {
        let k_f64 = k as f64;

        // Average of remaining eigenvalues
        let sigma2 = if k < eigenvalues.len() - 1 {
            eigenvalues.slice(scirs2_core::ndarray::s![k + 1..]).sum() / (p - k_f64 - 1.0)
        } else {
            1e-10
        };

        // Log-likelihood
        let ll = -n / 2.0
            * (eigenvalues
                .slice(scirs2_core::ndarray::s![..=k])
                .mapv(f64::ln)
                .sum()
                + (p - k_f64 - 1.0) * sigma2.ln()
                + p * (2.0 * std::f64::consts::PI).ln());

        // AIC penalty
        let aic_penalty = k_f64 * (2.0 * p - k_f64 - 1.0);
        let aic = ll - aic_penalty;

        if aic > best_ll {
            best_ll = aic;
            best_k = k + 1;
        }
    }

    Ok(best_k)
}

/// Incremental PCA for large datasets that don't fit in memory
#[derive(Debug, Clone)]
pub struct IncrementalPCA {
    /// Base PCA configuration
    pub pca: PCA,
    /// Batch size for incremental updates
    pub batchsize: usize,
    /// Running mean
    mean: Option<Array1<f64>>,
    /// Running components
    components: Option<Array2<f64>>,
    /// Singular values
    singular_values: Option<Array1<f64>>,
    /// Number of samples seen
    n_samples_seen: usize,
    /// Incremental SVD state
    svd_u: Option<Array2<f64>>,
    svd_s: Option<Array1<f64>>,
    svd_v: Option<Array2<f64>>,
}

impl IncrementalPCA {
    /// Create a new incremental PCA instance
    pub fn new(n_components: usize, batchsize: usize) -> Result<Self> {
        check_positive(n_components, "n_components")?;
        check_positive(batchsize, "batchsize")?;

        Ok(Self {
            pca: PCA::new().with_n_components(n_components),
            batchsize,
            mean: None,
            components: None,
            singular_values: None,
            n_samples_seen: 0,
            svd_u: None,
            svd_s: None,
            svd_v: None,
        })
    }

    /// Partial fit on a batch of data
    pub fn partial_fit(&mut self, batch: ArrayView2<f64>) -> Result<()> {
        checkarray_finite(&batch, "batch")?;
        let (batchsize, n_features) = batch.dim();

        // Update mean incrementally
        let batch_mean = batch.mean_axis(Axis(0)).unwrap();
        let old_n = self.n_samples_seen;
        self.n_samples_seen += batchsize;

        self.mean = match &self.mean {
            None => Some(batch_mean.clone()),
            Some(mean) => {
                let updated = (mean * old_n as f64 + &batch_mean * batchsize as f64)
                    / self.n_samples_seen as f64;
                Some(updated)
            }
        };

        // Center the batch
        let mut centered_batch = batch.to_owned();
        for mut row in centered_batch.rows_mut() {
            row -= &batch_mean;
        }

        // Incremental SVD update using Brand's algorithm
        let n_components = self
            .pca
            .n_components
            .unwrap_or(n_features.min(self.n_samples_seen));

        if self.svd_u.is_none() {
            // First batch - initialize with standard SVD
            use scirs2_core::ndarray::ndarray_linalg::SVD;
            let (u, s, vt) = centered_batch
                .svd(true, true)
                .map_err(|e| StatsError::ComputationError(format!("Initial SVD failed: {}", e)))?;

            let u = u.unwrap();
            let vt = vt.unwrap();

            // Keep only n_components
            self.svd_u = Some(
                u.slice(scirs2_core::ndarray::s![.., ..n_components])
                    .to_owned(),
            );
            self.svd_s = Some(s.slice(scirs2_core::ndarray::s![..n_components]).to_owned());
            self.svd_v = Some(
                vt.slice(scirs2_core::ndarray::s![..n_components, ..])
                    .t()
                    .to_owned(),
            );

            self.components = Some(self.svd_v.as_ref().unwrap().t().to_owned());
            self.singular_values = Some(self.svd_s.as_ref().unwrap().clone());
        } else {
            // Incremental update
            let u_old = self.svd_u.as_ref().unwrap();
            let s_old = self.svd_s.as_ref().unwrap();
            let v_old = self.svd_v.as_ref().unwrap();

            // Project new data onto existing components
            let projection = centered_batch.dot(v_old);
            let residual = &centered_batch - &projection.dot(&v_old.t());

            // QR decomposition of residual
            use scirs2_core::ndarray::ndarray_linalg::QR;
            let (q_res, r_res) = residual.qr().map_err(|e| {
                StatsError::ComputationError(format!("QR decomposition failed: {}", e))
            })?;

            // Build augmented matrix
            let k = s_old.len();
            let p = r_res.ncols();

            // Create block matrix [diag(s_old), projection^T; 0, r_res]
            let mut augmented = Array2::zeros((k + p, k + p));
            for i in 0..k {
                augmented[[i, i]] = s_old[i];
            }
            for i in 0..projection.nrows() {
                for j in 0..k {
                    augmented[[j, k + i]] = projection[[i, j]];
                }
            }
            for i in 0..p {
                for j in 0..p {
                    augmented[[k + i, k + j]] = r_res[[i, j]];
                }
            }

            // SVD of augmented matrix
            use scirs2_core::ndarray::ndarray_linalg::SVD;
            let (u_aug, s_aug, vt_aug) = augmented.svd(true, true).map_err(|e| {
                StatsError::ComputationError(format!("Augmented SVD failed: {}", e))
            })?;

            let u_aug = u_aug.unwrap();
            let vt_aug = vt_aug.unwrap();

            // Update U
            let mut u_new = Array2::zeros((old_n + batchsize, n_components));
            let u_aug_slice = u_aug.slice(scirs2_core::ndarray::s![..n_components, ..n_components]);

            // Update old samples part
            let u_old_part = u_old.dot(&u_aug_slice.t());
            u_new
                .slice_mut(scirs2_core::ndarray::s![..old_n, ..])
                .assign(&u_old_part);

            // Update new samples part
            let u_batch_part =
                projection.dot(&u_aug_slice.slice(scirs2_core::ndarray::s![.., ..k]).t());
            let u_res_part = q_res.dot(&u_aug_slice.slice(scirs2_core::ndarray::s![.., k..]).t());
            u_new
                .slice_mut(scirs2_core::ndarray::s![old_n.., ..])
                .assign(&(&u_batch_part + &u_res_part));

            // Update singular values
            self.svd_s = Some(
                s_aug
                    .slice(scirs2_core::ndarray::s![..n_components])
                    .to_owned(),
            );

            // Update V
            let v_aug_slice =
                vt_aug.slice(scirs2_core::ndarray::s![..n_components, ..n_components]);
            let mut v_new = Array2::zeros((n_features, n_components));

            let v_old_part = v_old.dot(&v_aug_slice.slice(scirs2_core::ndarray::s![.., ..k]).t());
            let v_res_part = q_res
                .t()
                .dot(&centered_batch)
                .t()
                .dot(&v_aug_slice.slice(scirs2_core::ndarray::s![.., k..]).t());
            v_new.assign(&(&v_old_part + &v_res_part));

            self.svd_u = Some(u_new);
            self.svd_v = Some(v_new.clone());
            self.components = Some(v_new.t().to_owned());
            self.singular_values = Some(self.svd_s.as_ref().unwrap().clone());
        }

        Ok(())
    }

    /// Transform new data
    pub fn transform(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.components.is_none() || self.mean.is_none() {
            return Err(StatsError::ComputationError(
                "IncrementalPCA must be fitted before transform".to_string(),
            ));
        }

        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= self.mean.as_ref().unwrap();
        }

        Ok(centered.dot(&self.components.as_ref().unwrap().t()))
    }
}
