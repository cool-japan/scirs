//! Factor Analysis
//!
//! Factor analysis is a dimensionality reduction technique that identifies latent factors
//! that explain the correlations among observed variables.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::validation::*;

/// Factor Analysis model
#[derive(Debug, Clone)]
pub struct FactorAnalysis {
    /// Number of factors to extract
    pub n_factors: usize,
    /// Maximum number of iterations for EM algorithm
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to perform varimax rotation
    pub rotation: RotationType,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Type of factor rotation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotationType {
    /// No rotation
    None,
    /// Varimax rotation (orthogonal)
    Varimax,
    /// Promax rotation (oblique)
    Promax,
}

/// Result of factor analysis
#[derive(Debug, Clone)]
pub struct FactorAnalysisResult {
    /// Factor loadings matrix (p x k)
    pub loadings: Array2<f64>,
    /// Specific variances (unique factors)
    pub noise_variance: Array1<f64>,
    /// Factors scores for training data (n x k)
    pub scores: Array2<f64>,
    /// Mean of training data
    pub mean: Array1<f64>,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Proportion of variance explained by each factor
    pub explained_variance_ratio: Array1<f64>,
    /// Communalities (proportion of variance in each variable explained by factors)
    pub communalities: Array1<f64>,
}

impl Default for FactorAnalysis {
    fn default() -> Self {
        Self {
            n_factors: 2,
            max_iter: 1000,
            tol: 1e-6,
            rotation: RotationType::Varimax,
            random_state: None,
        }
    }
}

impl FactorAnalysis {
    /// Create a new factor analysis instance
    pub fn new(n_factors: usize) -> Result<Self> {
        check_positive(n_factors, "n_factors")?;
        Ok(Self {
            n_factors,
            ..Default::default()
        })
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set rotation type
    pub fn with_rotation(mut self, rotation: RotationType) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the factor analysis model
    pub fn fit(&self, data: ArrayView2<f64>) -> Result<FactorAnalysisResult> {
        checkarray_finite(&data, "data")?;
        let (n_samples, n_features) = data.dim();

        if n_samples < 2 {
            return Err(StatsError::InvalidArgument(
                "n_samples must be at least 2".to_string(),
            ));
        }

        if self.n_factors >= n_features {
            return Err(StatsError::InvalidArgument(format!(
                "n_factors ({}) must be less than n_features ({})",
                self.n_factors, n_features
            )));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centereddata = data.to_owned();
        for mut row in centereddata.rows_mut() {
            row -= &mean;
        }

        // Initialize parameters
        let (mut loadings, mut psi) = self.initialize_parameters(&centereddata)?;

        let mut prev_log_likelihood = f64::NEG_INFINITY;
        let mut n_iter = 0;

        // EM algorithm
        for iteration in 0..self.max_iter {
            // E-step: compute expected sufficient statistics
            let (e_h, e_hht) = self.e_step(&centereddata, &loadings, &psi)?;

            // M-step: update parameters
            let (new_loadings, new_psi) = self.m_step(&centereddata, &e_h, &e_hht)?;

            // Compute log-likelihood
            let log_likelihood =
                self.compute_log_likelihood(&centereddata, &new_loadings, &new_psi)?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                loadings = new_loadings;
                psi = new_psi;
                n_iter = iteration + 1;
                break;
            }

            loadings = new_loadings;
            psi = new_psi;
            prev_log_likelihood = log_likelihood;
            n_iter = iteration + 1;
        }

        if n_iter == self.max_iter {
            return Err(StatsError::ConvergenceError(format!(
                "EM algorithm failed to converge after {} iterations",
                self.max_iter
            )));
        }

        // Apply rotation if specified
        let rotated_loadings = match self.rotation {
            RotationType::None => loadings,
            RotationType::Varimax => self.varimax_rotation(&loadings)?,
            RotationType::Promax => self.promax_rotation(&loadings)?,
        };

        // Compute factor scores
        let scores = self.compute_factor_scores(&centereddata, &rotated_loadings, &psi)?;

        // Compute explained variance and communalities
        let explained_variance_ratio = self.compute_explained_variance(&rotated_loadings);
        let communalities = self.compute_communalities(&rotated_loadings);

        // Final log-likelihood
        let final_log_likelihood =
            self.compute_log_likelihood(&centereddata, &rotated_loadings, &psi)?;

        Ok(FactorAnalysisResult {
            loadings: rotated_loadings,
            noise_variance: psi,
            scores,
            mean,
            log_likelihood: final_log_likelihood,
            n_iter,
            explained_variance_ratio,
            communalities,
        })
    }

    /// Initialize factor loadings and specific variances
    fn initialize_parameters(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>)> {
        let (n_samples, n_features) = data.dim();

        // Initialize using SVD of data
        use scirs2_core::ndarray::ndarray_linalg::SVD;
        let (u, s, vt) = data.svd(false, true).map_err(|e| {
            StatsError::ComputationError(format!("SVD initialization failed: {}", e))
        })?;

        let v = vt.unwrap().t().to_owned();

        // Initial loadings from first k components
        let mut loadings = Array2::zeros((n_features, self.n_factors));
        for i in 0..self.n_factors {
            let scale = (s[i] / (n_samples as f64).sqrt()).max(1e-6);
            for j in 0..n_features {
                loadings[[j, i]] = v[[j, i]] * scale;
            }
        }

        // Initialize specific variances
        let mut psi = Array1::ones(n_features);
        for i in 0..n_features {
            let communality = loadings.row(i).dot(&loadings.row(i));
            psi[i] = (1.0 - communality).max(0.01); // Ensure positive
        }

        Ok((loadings, psi))
    }

    /// E-step of EM algorithm
    fn e_step(
        &self,
        data: &Array2<f64>,
        loadings: &Array2<f64>,
        psi: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = data.dim();

        // Construct precision matrix: Psi^{-1}
        let mut psi_inv = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            if psi[i] <= 0.0 {
                return Err(StatsError::ComputationError(
                    "Specific variances must be positive".to_string(),
                ));
            }
            psi_inv[[i, i]] = 1.0 / psi[i];
        }

        // Compute M = I + L^T Psi^{-1} L
        let lt_psi_inv = loadings.t().dot(&psi_inv);
        let m = Array2::eye(self.n_factors) + lt_psi_inv.dot(loadings);

        // Invert M
        let m_inv = scirs2_linalg::inv(&m.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert M matrix: {}", e))
        })?;

        // Compute conditional expectations
        let mut e_h = Array2::zeros((n_samples, self.n_factors));
        let e_hht = m_inv.clone(); // This is E[h h^T | x]

        for i in 0..n_samples {
            let x = data.row(i);
            let e_h_i = m_inv.dot(&lt_psi_inv.dot(&x.to_owned()));
            e_h.row_mut(i).assign(&e_h_i);
        }

        Ok((e_h, e_hht))
    }

    /// M-step of EM algorithm
    fn m_step(
        &self,
        data: &Array2<f64>,
        e_h: &Array2<f64>,
        e_hht: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let (n_samples, n_features) = data.dim();

        // Update loadings: L = (X^T E[H]) (E[H^T H])^{-1}
        let xte_h = data.t().dot(e_h);
        let sum_e_hht = e_hht * n_samples as f64; // Sum over samples

        let sum_e_hht_inv = scirs2_linalg::inv(&sum_e_hht.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert sum E[HH^T]: {}", e))
        })?;

        let new_loadings = xte_h.dot(&sum_e_hht_inv);

        // Update specific variances
        let mut new_psi = Array1::zeros(n_features);

        for j in 0..n_features {
            let x_j = data.column(j);
            let l_j = new_loadings.row(j);

            let mut sum_var = 0.0;
            for i in 0..n_samples {
                let x_ij = x_j[i];
                let e_h_i = e_h.row(i);
                let residual = x_ij - l_j.dot(&e_h_i.to_owned());
                sum_var += residual * residual;

                // Add E[h h^T] term
                let quad_form = l_j.dot(&e_hht.dot(&l_j.to_owned()));
                sum_var += quad_form;
            }

            new_psi[j] = (sum_var / n_samples as f64).max(1e-6); // Ensure positive
        }

        Ok((new_loadings, new_psi))
    }

    /// Compute log-likelihood
    fn compute_log_likelihood(
        &self,
        data: &Array2<f64>,
        loadings: &Array2<f64>,
        psi: &Array1<f64>,
    ) -> Result<f64> {
        let (n_samples, n_features) = data.dim();

        // Construct covariance matrix: Sigma = L L^T + Psi
        let ll_t = loadings.dot(&loadings.t());
        let mut sigma = ll_t;
        for i in 0..n_features {
            sigma[[i, i]] += psi[i];
        }

        // Compute determinant and inverse
        let det_sigma = scirs2_linalg::det(&sigma.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        if det_sigma <= 0.0 {
            return Err(StatsError::ComputationError(
                "Covariance matrix must be positive definite".to_string(),
            ));
        }

        let sigma_inv = scirs2_linalg::inv(&sigma.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert covariance: {}", e))
        })?;

        // Compute log-likelihood
        let mut log_likelihood = 0.0;
        let log_det_term =
            -0.5 * n_features as f64 * (2.0 * std::f64::consts::PI).ln() - 0.5 * det_sigma.ln();

        for i in 0..n_samples {
            let x = data.row(i);
            let quad_form = x.dot(&sigma_inv.dot(&x.to_owned()));
            log_likelihood += log_det_term - 0.5 * quad_form;
        }

        Ok(log_likelihood)
    }

    /// Varimax rotation
    fn varimax_rotation(&self, loadings: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_features, n_factors) = loadings.dim();
        let mut rotated = loadings.clone();

        let max_iter = 30;
        let tol = 1e-6;

        for _ in 0..max_iter {
            let rotation_matrix = Array2::<f64>::eye(n_factors);
            let mut converged = true;

            // Rotate each pair of factors
            for i in 0..n_factors {
                for j in (i + 1)..n_factors {
                    let col_i = rotated.column(i).to_owned();
                    let col_j = rotated.column(j).to_owned();

                    // Compute rotation angle
                    let u = &col_i * &col_i - &col_j * &col_j;
                    let v = 2.0 * &col_i * &col_j;

                    let a = u.sum();
                    let b = v.sum();
                    let c = (&u * &u - &v * &v).sum();
                    let d = 2.0 * (&u * &v).sum();

                    let num = d - 2.0 * a * b / n_features as f64;
                    let den = c - (a * a - b * b) / n_features as f64;

                    if den.abs() < 1e-10 {
                        continue;
                    }

                    let phi = 0.25 * (num / den).atan();

                    if phi.abs() > tol {
                        converged = false;

                        // Apply rotation
                        let cos_phi = phi.cos();
                        let sin_phi = phi.sin();

                        let new_col_i = cos_phi * &col_i - sin_phi * &col_j;
                        let new_col_j = sin_phi * &col_i + cos_phi * &col_j;

                        rotated.column_mut(i).assign(&new_col_i);
                        rotated.column_mut(j).assign(&new_col_j);
                    }
                }
            }

            if converged {
                break;
            }
        }

        Ok(rotated)
    }

    /// Promax rotation (oblique)
    fn promax_rotation(&self, loadings: &Array2<f64>) -> Result<Array2<f64>> {
        // First apply varimax rotation
        let varimax_rotated = self.varimax_rotation(loadings)?;

        // Then apply promax transformation
        let kappa = 4.0; // Power parameter
        let (n_features, n_factors) = varimax_rotated.dim();

        // Compute target matrix by raising loadings to power kappa
        let mut target = Array2::zeros((n_features, n_factors));
        for i in 0..n_features {
            for j in 0..n_factors {
                let val = varimax_rotated[[i, j]];
                target[[i, j]] = val.abs().powf(kappa) * val.signum();
            }
        }

        // Solve for transformation matrix using least squares
        // T = (L^T L)^{-1} L^T P where P is target
        let ltl = varimax_rotated.t().dot(&varimax_rotated);
        let ltl_inv = scirs2_linalg::inv(&ltl.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Failed to invert L^T L: {}", e)))?;

        let ltp = varimax_rotated.t().dot(&target);
        let transform = ltl_inv.dot(&ltp);

        // Apply transformation
        let rotated = varimax_rotated.dot(&transform);

        Ok(rotated)
    }

    /// Compute factor scores using regression method
    fn compute_factor_scores(
        &self,
        data: &Array2<f64>,
        loadings: &Array2<f64>,
        psi: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n_features = loadings.nrows();

        // Construct precision matrix
        let mut psi_inv = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            psi_inv[[i, i]] = 1.0 / psi[i];
        }

        // Compute factor score coefficient matrix: (L^T Psi^{-1} L)^{-1} L^T Psi^{-1}
        let lt_psi_inv = loadings.t().dot(&psi_inv);
        let lt_psi_inv_l = lt_psi_inv.dot(loadings);

        let lt_psi_inv_l_inv = scirs2_linalg::inv(&lt_psi_inv_l.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute factor score weights: {}", e))
        })?;

        let score_weights = lt_psi_inv_l_inv.dot(&lt_psi_inv);

        // Compute scores
        let scores = data.dot(&score_weights.t());

        Ok(scores)
    }

    /// Compute explained variance ratio for each factor
    fn compute_explained_variance(&self, loadings: &Array2<f64>) -> Array1<f64> {
        let factor_variances = loadings
            .axis_iter(Axis(1))
            .map(|col| col.dot(&col))
            .collect::<Vec<_>>();

        let total_variance: f64 = factor_variances.iter().sum();

        Array1::from_vec(factor_variances).mapv(|v| v / total_variance)
    }

    /// Compute communalities (proportion of variance explained for each variable)
    fn compute_communalities(&self, loadings: &Array2<f64>) -> Array1<f64> {
        let mut communalities = Array1::zeros(loadings.nrows());

        for i in 0..loadings.nrows() {
            communalities[i] = loadings.row(i).dot(&loadings.row(i));
        }

        communalities
    }

    /// Transform new data to factor space
    pub fn transform(
        &self,
        data: ArrayView2<f64>,
        result: &FactorAnalysisResult,
    ) -> Result<Array2<f64>> {
        checkarray_finite(&data, "data")?;

        if data.ncols() != result.mean.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "data has {} features, expected {}",
                data.ncols(),
                result.mean.len()
            )));
        }

        // Center the data
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &result.mean;
        }

        // Compute factor scores
        self.compute_factor_scores(&centered, &result.loadings, &result.noise_variance)
    }
}

/// Exploratory Factor Analysis (EFA) utilities
pub mod efa {
    use super::*;

    /// Determine optimal number of factors using parallel analysis
    pub fn parallel_analysis(
        data: ArrayView2<f64>,
        n_simulations: usize,
        percentile: f64,
        seed: Option<u64>,
    ) -> Result<usize> {
        checkarray_finite(&data, "data")?;
        check_positive(n_simulations, "n_simulations")?;

        if percentile <= 0.0 || percentile >= 100.0 {
            return Err(StatsError::InvalidArgument(
                "percentile must be between 0 and 100".to_string(),
            ));
        }

        let (n_samples, n_features) = data.dim();

        // Compute eigenvalues of real data correlation matrix
        let real_eigenvalues = compute_correlation_eigenvalues(data)?;

        // Initialize RNG
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let s = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(s)
            }
        };

        // Generate random data and compute eigenvalues
        let mut simulated_eigenvalues = Vec::with_capacity(n_simulations);

        for _ in 0..n_simulations {
            // Generate random normal data with same dimensions
            let mut randomdata = Array2::zeros((n_samples, n_features));
            use scirs2_core::random::{Distribution, Normal};
            let normal = Normal::new(0.0, 1.0).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
            })?;

            for i in 0..n_samples {
                for j in 0..n_features {
                    randomdata[[i, j]] = normal.sample(&mut rng);
                }
            }

            let eigenvalues = compute_correlation_eigenvalues(randomdata.view())?;
            simulated_eigenvalues.push(eigenvalues);
        }

        // Compute percentile thresholds
        let mut thresholds = Array1::zeros(n_features);
        for i in 0..n_features {
            let mut values: Vec<f64> = simulated_eigenvalues.iter().map(|ev| ev[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let index = ((percentile / 100.0) * (n_simulations - 1) as f64).round() as usize;
            thresholds[i] = values[index.min(n_simulations - 1)];
        }

        // Count factors where real eigenvalue > threshold
        let mut n_factors = 0;
        for i in 0..n_features {
            if real_eigenvalues[i] > thresholds[i] {
                n_factors += 1;
            } else {
                break;
            }
        }

        Ok(n_factors.max(1)) // At least 1 factor
    }

    /// Compute eigenvalues of correlation matrix
    fn compute_correlation_eigenvalues(data: ArrayView2<f64>) -> Result<Array1<f64>> {
        // Center data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // Compute correlation matrix
        let cov = centered.t().dot(&centered) / (data.nrows() - 1) as f64;

        // Standardize to correlation
        let mut corr = cov.clone();
        for i in 0..corr.nrows() {
            for j in 0..corr.ncols() {
                let std_i = cov[[i, i]].sqrt();
                let std_j = cov[[j, j]].sqrt();
                if std_i > 1e-10 && std_j > 1e-10 {
                    corr[[i, j]] = cov[[i, j]] / (std_i * std_j);
                }
            }
        }

        // Compute eigenvalues
        use scirs2_core::ndarray::ndarray_linalg::Eigh;
        let eigenvalues = corr
            .eigh(scirs2_core::ndarray::ndarray_linalg::UPLO::Upper)
            .map_err(|e| {
                StatsError::ComputationError(format!("Eigenvalue decomposition failed: {}", e))
            })?
            .0;

        // Sort in descending order
        let mut sorted_eigenvalues = eigenvalues.to_vec();
        sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

        Ok(Array1::from_vec(sorted_eigenvalues))
    }

    /// Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy
    pub fn kmo_test(data: ArrayView2<f64>) -> Result<f64> {
        checkarray_finite(&data, "data")?;

        // Compute correlation matrix
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        let cov = centered.t().dot(&centered) / (data.nrows() - 1) as f64;
        let n = cov.nrows();

        // Standardize to correlation
        let mut corr = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let std_i = cov[[i, i]].sqrt();
                let std_j = cov[[j, j]].sqrt();
                if std_i > 1e-10 && std_j > 1e-10 {
                    corr[[i, j]] = cov[[i, j]] / (std_i * std_j);
                } else if i == j {
                    corr[[i, j]] = 1.0;
                }
            }
        }

        // Compute anti-image correlation matrix
        let corr_inv = scirs2_linalg::inv(&corr.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert correlation matrix: {}", e))
        })?;

        // Compute KMO statistic
        let mut sum_squared_corr = 0.0;
        let mut sum_squared_partial = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum_squared_corr += corr[[i, j]] * corr[[i, j]];

                    // Partial correlation
                    let partial = -corr_inv[[i, j]] / (corr_inv[[i, i]] * corr_inv[[j, j]]).sqrt();
                    sum_squared_partial += partial * partial;
                }
            }
        }

        let kmo = sum_squared_corr / (sum_squared_corr + sum_squared_partial);
        Ok(kmo)
    }

    /// Bartlett's test of sphericity
    pub fn bartlett_test(data: ArrayView2<f64>) -> Result<(f64, f64)> {
        checkarray_finite(&data, "data")?;
        let (n, p) = data.dim();

        if n <= p {
            return Err(StatsError::InvalidArgument(
                "Number of samples must exceed number of variables".to_string(),
            ));
        }

        // Compute correlation matrix
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        let cov = centered.t().dot(&centered) / (n - 1) as f64;

        // Standardize to correlation
        let mut corr = Array2::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let std_i = cov[[i, i]].sqrt();
                let std_j = cov[[j, j]].sqrt();
                if std_i > 1e-10 && std_j > 1e-10 {
                    corr[[i, j]] = cov[[i, j]] / (std_i * std_j);
                } else if i == j {
                    corr[[i, j]] = 1.0;
                }
            }
        }

        // Compute test statistic
        let det_corr = scirs2_linalg::det(&corr.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        if det_corr <= 0.0 {
            return Err(StatsError::ComputationError(
                "Correlation matrix must be positive definite".to_string(),
            ));
        }

        let chi2 = -(n as f64 - 1.0 - (2.0 * p as f64 + 5.0) / 6.0) * det_corr.ln();
        let df = p * (p - 1) / 2;

        // Approximate p-value using chi-square distribution
        let p_value = chi2_survival(chi2, df as f64);

        Ok((chi2, p_value))
    }
}

/// Approximate survival function for chi-square distribution
#[allow(dead_code)]
fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    // Very rough approximation - in practice use proper chi-square CDF
    let mean = df;
    let var = 2.0 * df;
    let std = var.sqrt();

    // Normal approximation for large df
    if df > 30.0 {
        let z = (x - mean) / std;
        return 0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2));
    }

    // Simple exponential approximation for small df
    (-x / mean).exp()
}

/// Error function approximation
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
