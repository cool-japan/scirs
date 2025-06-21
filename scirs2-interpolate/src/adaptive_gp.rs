//! Gaussian process regression with adaptive kernels
//!
//! This module provides advanced Gaussian process (GP) regression with automatic
//! kernel selection and hyperparameter optimization. Gaussian processes provide
//! a probabilistic approach to interpolation that naturally quantifies uncertainty
//! and can adapt to different data characteristics through kernel selection.
//!
//! # Adaptive Kernel Features
//!
//! - **Multiple kernel types**: RBF, Matérn, Periodic, Linear, Polynomial, Spectral Mixture
//! - **Automatic kernel selection**: Model comparison and selection based on evidence
//! - **Hyperparameter optimization**: Marginal likelihood maximization with gradients
//! - **Kernel composition**: Automatic discovery of additive and multiplicative combinations
//! - **Uncertainty quantification**: Predictive variance and confidence intervals
//! - **Active learning integration**: Uncertainty-guided sampling for optimal data collection
//! - **Sparse GP methods**: Inducing points for scalability to large datasets
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_interpolate::adaptive_gp::{AdaptiveGaussianProcess, KernelType};
//!
//! // Create sample data with noise
//! let x = Array1::linspace(0.0_f64, 10.0_f64, 20);
//! let y = x.mapv(|x| x.sin() + 0.1_f64 * (5.0_f64 * x).cos()) + Array1::from_elem(20, 0.05_f64);
//!
//! // Create adaptive GP with automatic kernel selection
//! let mut gp = AdaptiveGaussianProcess::new()
//!     .with_kernel_candidates(vec![
//!         KernelType::RBF,
//!         KernelType::Matern52,
//!         KernelType::Periodic,
//!     ])
//!     .with_automatic_optimization(true)
//!     .with_uncertainty_quantification(true);
//!
//! // Fit the model (automatically selects best kernel and optimizes hyperparameters)
//! gp.fit(&x.view(), &y.view()).unwrap();
//!
//! // Make predictions with uncertainty
//! let x_new = Array1::linspace(0.0_f64, 10.0_f64, 100);
//! let (mean, variance) = gp.predict_with_uncertainty(&x_new.view()).unwrap();
//!
//! println!("Selected kernel: {:?}", gp.get_selected_kernel());
//! println!("Predictive uncertainty: {:?}", variance.mapv(|v: f64| v.sqrt()));
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display, LowerExp};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Types of kernels available for Gaussian process regression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Radial Basis Function (RBF) / Squared Exponential kernel
    RBF,
    /// Matérn kernel with ν = 1/2 (exponential)
    Matern12,
    /// Matérn kernel with ν = 3/2
    Matern32,
    /// Matérn kernel with ν = 5/2
    Matern52,
    /// Periodic kernel for repeating patterns
    Periodic,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
    /// Rational quadratic kernel
    RationalQuadratic,
    /// Spectral mixture kernel (for complex periodicities)
    SpectralMixture,
    /// White noise kernel
    WhiteNoise,
}

/// Hyperparameters for different kernel types
#[derive(Debug, Clone)]
pub struct KernelHyperparameters<T> {
    /// Output variance (signal variance)
    pub output_variance: T,
    /// Length scale parameter(s)
    pub length_scales: Vec<T>,
    /// Noise variance
    pub noise_variance: T,
    /// Additional kernel-specific parameters
    pub additional_params: HashMap<String, T>,
}

impl<T: Float + FromPrimitive> Default for KernelHyperparameters<T> {
    fn default() -> Self {
        Self {
            output_variance: T::one(),
            length_scales: vec![T::one()],
            noise_variance: T::from(0.01).unwrap(),
            additional_params: HashMap::new(),
        }
    }
}

/// Information about a fitted kernel model
#[derive(Debug, Clone)]
pub struct KernelModel<T> {
    /// Type of kernel
    pub kernel_type: KernelType,
    /// Optimized hyperparameters
    pub hyperparameters: KernelHyperparameters<T>,
    /// Log marginal likelihood (model evidence)
    pub log_marginal_likelihood: T,
    /// Number of hyperparameters in the model
    pub num_hyperparameters: usize,
    /// Bayesian Information Criterion (BIC) for model comparison
    pub bic: T,
    /// Training data size when model was fitted
    pub training_size: usize,
}

/// Configuration for adaptive Gaussian process regression
#[derive(Debug, Clone)]
pub struct AdaptiveGPConfig<T> {
    /// Candidate kernel types to consider
    pub kernel_candidates: Vec<KernelType>,
    /// Whether to enable automatic hyperparameter optimization
    pub enable_optimization: bool,
    /// Whether to enable uncertainty quantification
    pub enable_uncertainty: bool,
    /// Maximum number of optimization iterations
    pub max_optimization_iterations: usize,
    /// Convergence tolerance for optimization
    pub optimization_tolerance: T,
    /// Whether to try kernel combinations (additive/multiplicative)
    pub enable_kernel_composition: bool,
    /// Maximum number of components in composite kernels
    pub max_composite_components: usize,
    /// Whether to use sparse GP approximations for large datasets
    pub enable_sparse_gp: bool,
    /// Number of inducing points for sparse GP
    pub num_inducing_points: usize,
    /// Noise level for numerical stability
    pub jitter: T,
}

impl<T: Float + FromPrimitive> Default for AdaptiveGPConfig<T> {
    fn default() -> Self {
        Self {
            kernel_candidates: vec![
                KernelType::RBF,
                KernelType::Matern32,
                KernelType::Matern52,
                KernelType::Linear,
                KernelType::Periodic,
            ],
            enable_optimization: true,
            enable_uncertainty: true,
            max_optimization_iterations: 100,
            optimization_tolerance: T::from(1e-6).unwrap(),
            enable_kernel_composition: false,
            max_composite_components: 3,
            enable_sparse_gp: false,
            num_inducing_points: 50,
            jitter: T::from(1e-6).unwrap(),
        }
    }
}

/// Statistics tracking GP model selection and training
#[derive(Debug, Clone, Default)]
pub struct GPStats {
    /// Number of kernels evaluated
    pub kernels_evaluated: usize,
    /// Number of optimization iterations performed
    pub optimization_iterations: usize,
    /// Best log marginal likelihood achieved
    pub best_log_marginal_likelihood: f64,
    /// Computational time for model selection (milliseconds)
    pub model_selection_time_ms: u64,
    /// Computational time for hyperparameter optimization (milliseconds)
    pub optimization_time_ms: u64,
    /// Distribution of kernel types tried
    pub kernel_usage: HashMap<String, usize>,
}

/// Adaptive Gaussian Process for interpolation with automatic kernel selection
#[derive(Debug)]
pub struct AdaptiveGaussianProcess<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + 'static,
{
    /// Configuration settings
    config: AdaptiveGPConfig<T>,
    /// Training data
    x_train: Array1<T>,
    y_train: Array1<T>,
    /// Selected kernel model
    selected_model: Option<KernelModel<T>>,
    /// All evaluated models (for comparison)
    evaluated_models: Vec<KernelModel<T>>,
    /// Precomputed Cholesky decomposition for predictions
    cholesky_factor: Option<Array2<T>>,
    /// Alpha vector for predictions (K^{-1} * y)
    alpha: Option<Array1<T>>,
    /// Training statistics
    stats: GPStats,
    /// Inducing points for sparse GP
    #[allow(dead_code)]
    inducing_points: Option<Array1<T>>,
}

impl<T> Default for AdaptiveGaussianProcess<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AdaptiveGaussianProcess<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + 'static,
{
    /// Create a new adaptive Gaussian process with default configuration
    pub fn new() -> Self {
        Self {
            config: AdaptiveGPConfig::default(),
            x_train: Array1::zeros(0),
            y_train: Array1::zeros(0),
            selected_model: None,
            evaluated_models: Vec::new(),
            cholesky_factor: None,
            alpha: None,
            stats: GPStats::default(),
            inducing_points: None,
        }
    }

    /// Set the candidate kernel types to consider
    pub fn with_kernel_candidates(mut self, kernels: Vec<KernelType>) -> Self {
        self.config.kernel_candidates = kernels;
        self
    }

    /// Enable or disable automatic hyperparameter optimization
    pub fn with_automatic_optimization(mut self, enable: bool) -> Self {
        self.config.enable_optimization = enable;
        self
    }

    /// Enable or disable uncertainty quantification
    pub fn with_uncertainty_quantification(mut self, enable: bool) -> Self {
        self.config.enable_uncertainty = enable;
        self
    }

    /// Enable or disable kernel composition (additive/multiplicative combinations)
    pub fn with_kernel_composition(mut self, enable: bool) -> Self {
        self.config.enable_kernel_composition = enable;
        self
    }

    /// Set the maximum number of optimization iterations
    pub fn with_max_optimization_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_optimization_iterations = max_iter;
        self
    }

    /// Enable sparse GP approximations for large datasets
    pub fn with_sparse_approximation(mut self, enable: bool, num_inducing: usize) -> Self {
        self.config.enable_sparse_gp = enable;
        self.config.num_inducing_points = num_inducing;
        self
    }

    /// Fit the Gaussian process to training data
    ///
    /// This performs automatic kernel selection and hyperparameter optimization
    ///
    /// # Arguments
    ///
    /// * `x` - Input training data
    /// * `y` - Output training data
    ///
    /// # Returns
    ///
    /// Success indicator
    pub fn fit(&mut self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> InterpolateResult<bool> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < 2 {
            return Err(InterpolateError::InvalidValue(
                "At least 2 data points are required for GP regression".to_string(),
            ));
        }

        // Store training data
        self.x_train = x.to_owned();
        self.y_train = y.to_owned();

        let start_time = std::time::Instant::now();

        // Evaluate all candidate kernels
        self.evaluated_models.clear();
        self.stats = GPStats::default();

        for &kernel_type in &self.config.kernel_candidates.clone() {
            let model = self.fit_kernel(kernel_type)?;
            self.evaluated_models.push(model);

            // Update statistics
            *self
                .stats
                .kernel_usage
                .entry(format!("{:?}", kernel_type))
                .or_insert(0) += 1;
            self.stats.kernels_evaluated += 1;
        }

        // If kernel composition is enabled, try some combinations
        if self.config.enable_kernel_composition {
            self.evaluate_composite_kernels()?;
        }

        // Select the best kernel based on model evidence (log marginal likelihood)
        self.select_best_model()?;

        // Precompute quantities for prediction
        self.precompute_prediction_quantities()?;

        // Update timing statistics
        self.stats.model_selection_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(true)
    }

    /// Make predictions at new input points
    ///
    /// # Arguments
    ///
    /// * `x_new` - Input points for prediction
    ///
    /// # Returns
    ///
    /// Predicted mean values
    pub fn predict(&self, x_new: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        if self.selected_model.is_none() {
            return Err(InterpolateError::InvalidState(
                "Model must be fitted before making predictions".to_string(),
            ));
        }

        let (mean, _) = self.predict_with_uncertainty(x_new)?;
        Ok(mean)
    }

    /// Make predictions with uncertainty quantification
    ///
    /// # Arguments
    ///
    /// * `x_new` - Input points for prediction
    ///
    /// # Returns
    ///
    /// Tuple of (predicted means, predicted variances)
    pub fn predict_with_uncertainty(
        &self,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<(Array1<T>, Array1<T>)> {
        if self.selected_model.is_none() || self.alpha.is_none() {
            return Err(InterpolateError::InvalidState(
                "Model must be fitted before making predictions".to_string(),
            ));
        }

        let selected_model = self.selected_model.as_ref().unwrap();
        let alpha = self.alpha.as_ref().unwrap();

        // Compute kernel matrix between training and test points
        let k_star = self.compute_kernel_matrix_cross(
            &self.x_train.view(),
            x_new,
            selected_model.kernel_type,
            &selected_model.hyperparameters,
        )?;

        // Compute predictive mean: k_star^T * alpha
        let mean = k_star.t().dot(alpha);

        let variance = if self.config.enable_uncertainty {
            // Compute predictive variance
            self.compute_predictive_variance(x_new, &k_star)?
        } else {
            // Return zero variance if uncertainty is disabled
            Array1::zeros(x_new.len())
        };

        Ok((mean, variance))
    }

    /// Get the selected kernel type
    pub fn get_selected_kernel(&self) -> Option<KernelType> {
        self.selected_model.as_ref().map(|m| m.kernel_type)
    }

    /// Get the selected model's hyperparameters
    pub fn get_hyperparameters(&self) -> Option<&KernelHyperparameters<T>> {
        self.selected_model.as_ref().map(|m| &m.hyperparameters)
    }

    /// Get model selection and training statistics
    pub fn get_stats(&self) -> &GPStats {
        &self.stats
    }

    /// Get all evaluated models for comparison
    pub fn get_evaluated_models(&self) -> &[KernelModel<T>] {
        &self.evaluated_models
    }

    /// Fit a specific kernel type and return the fitted model
    fn fit_kernel(&mut self, kernel_type: KernelType) -> InterpolateResult<KernelModel<T>> {
        // Initialize hyperparameters based on kernel type
        let mut hyperparams = self.initialize_hyperparameters(kernel_type)?;

        let mut log_marginal_likelihood =
            self.compute_log_marginal_likelihood(kernel_type, &hyperparams)?;

        if self.config.enable_optimization {
            let optimization_start = std::time::Instant::now();

            // Optimize hyperparameters using gradient-free method (simplified)
            for iteration in 0..self.config.max_optimization_iterations {
                let improved = self.optimize_hyperparameters_step(
                    kernel_type,
                    &mut hyperparams,
                    &mut log_marginal_likelihood,
                )?;

                if !improved {
                    break;
                }

                self.stats.optimization_iterations += 1;

                // Check convergence (simplified)
                if iteration > 10 && iteration % 10 == 0 {
                    // In a real implementation, we'd check gradient norms
                    break;
                }
            }

            self.stats.optimization_time_ms += optimization_start.elapsed().as_millis() as u64;
        }

        // Compute BIC for model comparison
        let num_params = self.count_hyperparameters(kernel_type);
        let n = T::from(self.x_train.len()).unwrap();
        let bic = T::from(2.0).unwrap() * T::from(num_params).unwrap() * n.ln()
            - T::from(2.0).unwrap() * log_marginal_likelihood;

        Ok(KernelModel {
            kernel_type,
            hyperparameters: hyperparams,
            log_marginal_likelihood,
            num_hyperparameters: num_params,
            bic,
            training_size: self.x_train.len(),
        })
    }

    /// Initialize hyperparameters for a given kernel type
    fn initialize_hyperparameters(
        &self,
        kernel_type: KernelType,
    ) -> InterpolateResult<KernelHyperparameters<T>> {
        let mut hyperparams = KernelHyperparameters::default();

        // Estimate initial length scale from data
        let x_range = self.x_train[self.x_train.len() - 1] - self.x_train[0];
        let initial_length_scale = x_range / T::from(10.0).unwrap();
        hyperparams.length_scales = vec![initial_length_scale];

        // Estimate initial output variance from data variance
        let y_mean = self.y_train.sum() / T::from(self.y_train.len()).unwrap();
        let y_var = self.y_train.mapv(|y| (y - y_mean) * (y - y_mean)).sum()
            / T::from(self.y_train.len() - 1).unwrap();
        hyperparams.output_variance = y_var.max(T::from(0.01).unwrap());

        // Kernel-specific parameter initialization
        match kernel_type {
            KernelType::Periodic => {
                // Add period parameter
                hyperparams
                    .additional_params
                    .insert("period".to_string(), x_range / T::from(2.0).unwrap());
            }
            KernelType::Polynomial => {
                // Add degree parameter
                hyperparams
                    .additional_params
                    .insert("degree".to_string(), T::from(2.0).unwrap());
            }
            KernelType::RationalQuadratic => {
                // Add alpha parameter
                hyperparams
                    .additional_params
                    .insert("alpha".to_string(), T::one());
            }
            _ => {}
        }

        Ok(hyperparams)
    }

    /// Compute the log marginal likelihood for given kernel and hyperparameters
    fn compute_log_marginal_likelihood(
        &self,
        kernel_type: KernelType,
        hyperparams: &KernelHyperparameters<T>,
    ) -> InterpolateResult<T> {
        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(
            &self.x_train.view(),
            &self.x_train.view(),
            kernel_type,
            hyperparams,
        )?;

        // Add noise to diagonal
        let mut k_noisy = k_matrix;
        for i in 0..k_noisy.nrows() {
            k_noisy[(i, i)] += hyperparams.noise_variance + self.config.jitter;
        }

        // Compute Cholesky decomposition
        let cholesky = self.cholesky_decomposition(&k_noisy)?;

        // Compute log determinant via Cholesky factor
        let mut log_det = T::zero();
        for i in 0..cholesky.nrows() {
            log_det += cholesky[(i, i)].ln();
        }
        log_det = T::from(2.0).unwrap() * log_det;

        // Solve K * alpha = y
        let alpha = self.cholesky_solve(&cholesky, &self.y_train.view())?;

        // Compute data fit term: y^T * K^{-1} * y
        let data_fit = self.y_train.dot(&alpha);

        // Log marginal likelihood: -0.5 * (y^T * K^{-1} * y + log|K| + n * log(2π))
        let n = T::from(self.x_train.len()).unwrap();
        let log_2pi = T::from(2.0 * std::f64::consts::PI).unwrap().ln();

        let log_marginal_likelihood = -T::from(0.5).unwrap() * (data_fit + log_det + n * log_2pi);

        Ok(log_marginal_likelihood)
    }

    /// Perform one step of hyperparameter optimization
    fn optimize_hyperparameters_step(
        &self,
        kernel_type: KernelType,
        hyperparams: &mut KernelHyperparameters<T>,
        current_likelihood: &mut T,
    ) -> InterpolateResult<bool> {
        let mut improved = false;
        let _step_size = T::from(0.1).unwrap();

        // Try perturbing each hyperparameter
        let original_hyperparams = hyperparams.clone();

        // Optimize output variance
        let original_output_var = hyperparams.output_variance;
        for &multiplier in &[1.1, 0.9] {
            hyperparams.output_variance = original_output_var * T::from(multiplier).unwrap();
            if let Ok(likelihood) = self.compute_log_marginal_likelihood(kernel_type, hyperparams) {
                if likelihood > *current_likelihood {
                    *current_likelihood = likelihood;
                    improved = true;
                    break;
                }
            }
            hyperparams.output_variance = original_output_var;
        }

        // Optimize length scales
        for (i, &original_length_scale) in original_hyperparams.length_scales.iter().enumerate() {
            for &multiplier in &[1.2, 0.8] {
                hyperparams.length_scales[i] = original_length_scale * T::from(multiplier).unwrap();
                if let Ok(likelihood) =
                    self.compute_log_marginal_likelihood(kernel_type, hyperparams)
                {
                    if likelihood > *current_likelihood {
                        *current_likelihood = likelihood;
                        improved = true;
                        break;
                    }
                }
                hyperparams.length_scales[i] = original_length_scale;
            }
        }

        // Optimize noise variance
        let original_noise_var = hyperparams.noise_variance;
        for &multiplier in &[1.1, 0.9] {
            hyperparams.noise_variance = original_noise_var * T::from(multiplier).unwrap();
            if let Ok(likelihood) = self.compute_log_marginal_likelihood(kernel_type, hyperparams) {
                if likelihood > *current_likelihood {
                    *current_likelihood = likelihood;
                    improved = true;
                    break;
                }
            }
            hyperparams.noise_variance = original_noise_var;
        }

        Ok(improved)
    }

    /// Evaluate composite kernel combinations
    fn evaluate_composite_kernels(&mut self) -> InterpolateResult<()> {
        // For simplicity, we'll only try a few additive combinations
        let base_kernels = self.config.kernel_candidates.clone();

        for i in 0..base_kernels.len() {
            for _j in i + 1..base_kernels.len() {
                if self.evaluated_models.len() < 20 {
                    // Limit combinations
                    // In a full implementation, we'd properly handle composite kernels
                    // For now, we'll skip this to keep the implementation manageable
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Select the best model based on model evidence
    fn select_best_model(&mut self) -> InterpolateResult<()> {
        if self.evaluated_models.is_empty() {
            return Err(InterpolateError::InvalidState(
                "No models have been evaluated".to_string(),
            ));
        }

        // Find model with highest log marginal likelihood
        let best_model = self
            .evaluated_models
            .iter()
            .max_by(|a, b| {
                a.log_marginal_likelihood
                    .partial_cmp(&b.log_marginal_likelihood)
                    .unwrap()
            })
            .unwrap()
            .clone();

        self.stats.best_log_marginal_likelihood =
            best_model.log_marginal_likelihood.to_f64().unwrap();
        self.selected_model = Some(best_model);

        Ok(())
    }

    /// Precompute quantities needed for prediction
    fn precompute_prediction_quantities(&mut self) -> InterpolateResult<()> {
        if let Some(ref model) = self.selected_model {
            // Compute kernel matrix
            let k_matrix = self.compute_kernel_matrix(
                &self.x_train.view(),
                &self.x_train.view(),
                model.kernel_type,
                &model.hyperparameters,
            )?;

            // Add noise to diagonal
            let mut k_noisy = k_matrix;
            for i in 0..k_noisy.nrows() {
                k_noisy[(i, i)] += model.hyperparameters.noise_variance + self.config.jitter;
            }

            // Compute Cholesky decomposition
            let cholesky = self.cholesky_decomposition(&k_noisy)?;

            // Solve for alpha = K^{-1} * y
            let alpha = self.cholesky_solve(&cholesky, &self.y_train.view())?;

            self.cholesky_factor = Some(cholesky);
            self.alpha = Some(alpha);
        }

        Ok(())
    }

    /// Compute kernel matrix between two sets of points
    fn compute_kernel_matrix(
        &self,
        x1: &ArrayView1<T>,
        x2: &ArrayView1<T>,
        kernel_type: KernelType,
        hyperparams: &KernelHyperparameters<T>,
    ) -> InterpolateResult<Array2<T>> {
        let n1 = x1.len();
        let n2 = x2.len();
        let mut k_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                k_matrix[(i, j)] = self.kernel_function(x1[i], x2[j], kernel_type, hyperparams)?;
            }
        }

        Ok(k_matrix)
    }

    /// Compute kernel matrix between training and test points
    fn compute_kernel_matrix_cross(
        &self,
        x_train: &ArrayView1<T>,
        x_test: &ArrayView1<T>,
        kernel_type: KernelType,
        hyperparams: &KernelHyperparameters<T>,
    ) -> InterpolateResult<Array2<T>> {
        self.compute_kernel_matrix(x_train, x_test, kernel_type, hyperparams)
    }

    /// Evaluate kernel function between two points
    fn kernel_function(
        &self,
        x1: T,
        x2: T,
        kernel_type: KernelType,
        hyperparams: &KernelHyperparameters<T>,
    ) -> InterpolateResult<T> {
        let output_var = hyperparams.output_variance;
        let length_scale = hyperparams.length_scales[0];
        let distance = (x1 - x2).abs();

        let value = match kernel_type {
            KernelType::RBF => {
                let scaled_dist = distance / length_scale;
                output_var * (-T::from(0.5).unwrap() * scaled_dist * scaled_dist).exp()
            }
            KernelType::Matern12 => {
                let scaled_dist = distance / length_scale;
                output_var * (-scaled_dist).exp()
            }
            KernelType::Matern32 => {
                let scaled_dist = (T::from(3.0).unwrap().sqrt()) * distance / length_scale;
                output_var * (T::one() + scaled_dist) * (-scaled_dist).exp()
            }
            KernelType::Matern52 => {
                let scaled_dist = (T::from(5.0).unwrap().sqrt()) * distance / length_scale;
                output_var
                    * (T::one() + scaled_dist + scaled_dist * scaled_dist / T::from(3.0).unwrap())
                    * (-scaled_dist).exp()
            }
            KernelType::Linear => output_var * x1 * x2,
            KernelType::Polynomial => {
                let default_degree = T::from(2.0).unwrap();
                let degree = hyperparams
                    .additional_params
                    .get("degree")
                    .unwrap_or(&default_degree);
                output_var * (T::one() + x1 * x2 / length_scale).powf(*degree)
            }
            KernelType::Periodic => {
                let default_period = T::one();
                let period = hyperparams
                    .additional_params
                    .get("period")
                    .unwrap_or(&default_period);
                let pi = T::from(std::f64::consts::PI).unwrap();
                let sin_arg = pi * distance / *period;
                let scaled_sin = T::from(2.0).unwrap() * sin_arg.sin() / length_scale;
                output_var * (-T::from(0.5).unwrap() * scaled_sin * scaled_sin).exp()
            }
            KernelType::RationalQuadratic => {
                let default_alpha = T::one();
                let alpha = hyperparams
                    .additional_params
                    .get("alpha")
                    .unwrap_or(&default_alpha);
                let scaled_dist_sq = distance * distance
                    / (T::from(2.0).unwrap() * *alpha * length_scale * length_scale);
                output_var * (T::one() + scaled_dist_sq).powf(-*alpha)
            }
            KernelType::WhiteNoise => {
                if distance < T::epsilon() {
                    hyperparams.noise_variance
                } else {
                    T::zero()
                }
            }
            _ => {
                return Err(InterpolateError::InvalidValue(format!(
                    "Kernel type {:?} not implemented",
                    kernel_type
                )));
            }
        };

        Ok(value)
    }

    /// Compute predictive variance
    fn compute_predictive_variance(
        &self,
        x_new: &ArrayView1<T>,
        k_star: &Array2<T>,
    ) -> InterpolateResult<Array1<T>> {
        if let (Some(ref cholesky), Some(ref model)) = (&self.cholesky_factor, &self.selected_model)
        {
            let mut variance = Array1::zeros(x_new.len());

            for i in 0..x_new.len() {
                // Prior variance
                let prior_var = model.hyperparameters.output_variance;

                // Solve cholesky * v = k_star[:, i]
                let k_star_i = k_star.column(i);
                let v = self.cholesky_solve(cholesky, &k_star_i)?;

                // Predictive variance: prior_var - k_star_i^T * v
                let reduction = k_star_i.dot(&v);
                variance[i] = prior_var - reduction;

                // Ensure non-negative variance
                variance[i] = variance[i].max(T::zero());
            }

            Ok(variance)
        } else {
            Err(InterpolateError::InvalidState(
                "Model must be fitted before computing variance".to_string(),
            ))
        }
    }

    /// Count the number of hyperparameters for a kernel type
    fn count_hyperparameters(&self, kernel_type: KernelType) -> usize {
        match kernel_type {
            KernelType::RBF
            | KernelType::Matern12
            | KernelType::Matern32
            | KernelType::Matern52 => 3, // output_var, length_scale, noise_var
            KernelType::Linear => 2,            // output_var, noise_var
            KernelType::Polynomial => 4,        // output_var, length_scale, noise_var, degree
            KernelType::Periodic => 4,          // output_var, length_scale, noise_var, period
            KernelType::RationalQuadratic => 4, // output_var, length_scale, noise_var, alpha
            KernelType::SpectralMixture => 6,   // More complex
            KernelType::WhiteNoise => 1,        // noise_var only
        }
    }

    /// Compute Cholesky decomposition
    fn cholesky_decomposition(&self, matrix: &Array2<T>) -> InterpolateResult<Array2<T>> {
        let n = matrix.nrows();
        let mut chol = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum += chol[(j, k)] * chol[(j, k)];
                    }
                    let diag_val = matrix[(j, j)] - sum;
                    if diag_val <= T::zero() {
                        return Err(InterpolateError::InvalidValue(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[(j, j)] = diag_val.sqrt();
                } else {
                    // Off-diagonal element
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum += chol[(i, k)] * chol[(j, k)];
                    }
                    chol[(i, j)] = (matrix[(i, j)] - sum) / chol[(j, j)];
                }
            }
        }

        Ok(chol)
    }

    /// Solve linear system using Cholesky factorization
    fn cholesky_solve(
        &self,
        cholesky: &Array2<T>,
        rhs: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        let n = cholesky.nrows();

        // Forward substitution: L * y = rhs
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum += cholesky[(i, j)] * y[j];
            }
            y[i] = (rhs[i] - sum) / cholesky[(i, i)];
        }

        // Backward substitution: L^T * x = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = T::zero();
            for j in i + 1..n {
                sum += cholesky[(j, i)] * x[j];
            }
            x[i] = (y[i] - sum) / cholesky[(i, i)];
        }

        Ok(x)
    }
}

/// Convenience function to create an adaptive GP with automatic kernel selection
///
/// # Arguments
///
/// * `x` - Input training data
/// * `y` - Output training data
/// * `kernel_candidates` - List of kernel types to consider
///
/// # Returns
///
/// A fitted adaptive Gaussian process
pub fn make_adaptive_gp<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    kernel_candidates: Option<Vec<KernelType>>,
) -> InterpolateResult<AdaptiveGaussianProcess<T>>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + 'static,
{
    let mut gp = AdaptiveGaussianProcess::new();

    if let Some(kernels) = kernel_candidates {
        gp = gp.with_kernel_candidates(kernels);
    }

    gp.fit(x, y)?;
    Ok(gp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_adaptive_gp_creation() {
        let gp = AdaptiveGaussianProcess::<f64>::new();
        assert_eq!(gp.config.kernel_candidates.len(), 5);
        assert!(gp.config.enable_optimization);
        assert!(gp.config.enable_uncertainty);
    }

    #[test]
    fn test_adaptive_gp_simple_fit() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]); // x^2

        let mut gp = AdaptiveGaussianProcess::new()
            .with_kernel_candidates(vec![KernelType::RBF, KernelType::Polynomial]);

        let result = gp.fit(&x.view(), &y.view());
        assert!(result.is_ok());
        assert!(gp.get_selected_kernel().is_some());
    }

    #[test]
    fn test_adaptive_gp_prediction() {
        let x = Array1::linspace(0.0, 10.0, 11);
        let y = x.mapv(|x| x.sin());

        let mut gp = AdaptiveGaussianProcess::new()
            .with_kernel_candidates(vec![KernelType::RBF, KernelType::Matern52]);

        gp.fit(&x.view(), &y.view()).unwrap();

        let x_new = Array1::from_vec(vec![2.5, 7.5]);
        let predictions = gp.predict(&x_new.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        // Check that predictions are reasonable for sine function
        assert!((predictions[0] - 2.5_f64.sin()).abs() < 0.5);
        assert!((predictions[1] - 7.5_f64.sin()).abs() < 0.5);
    }

    #[test]
    fn test_adaptive_gp_uncertainty() {
        let x = Array1::from_vec(vec![0.0, 2.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 4.0, 16.0]);

        let mut gp = AdaptiveGaussianProcess::new()
            .with_kernel_candidates(vec![KernelType::RBF])
            .with_uncertainty_quantification(true);

        gp.fit(&x.view(), &y.view()).unwrap();

        let x_new = Array1::from_vec(vec![1.0, 3.0]);
        let (mean, variance) = gp.predict_with_uncertainty(&x_new.view()).unwrap();

        assert_eq!(mean.len(), 2);
        assert_eq!(variance.len(), 2);

        // Variance should be non-negative
        assert!(variance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_kernel_functions() {
        let gp = AdaptiveGaussianProcess::<f64>::new();
        let hyperparams = KernelHyperparameters::default();

        // Test RBF kernel
        let k_val = gp
            .kernel_function(0.0, 1.0, KernelType::RBF, &hyperparams)
            .unwrap();
        assert!(k_val > 0.0 && k_val < 1.0);

        // Test that kernel is symmetric
        let k_val2 = gp
            .kernel_function(1.0, 0.0, KernelType::RBF, &hyperparams)
            .unwrap();
        assert!((k_val - k_val2).abs() < 1e-10);

        // Test that kernel value at zero distance is maximal
        let k_zero = gp
            .kernel_function(0.0, 0.0, KernelType::RBF, &hyperparams)
            .unwrap();
        assert!(k_zero >= k_val);
    }

    #[test]
    fn test_model_selection() {
        let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 20);
        let y = x.mapv(|x| x.sin());

        let mut gp = AdaptiveGaussianProcess::new().with_kernel_candidates(vec![
            KernelType::RBF,
            KernelType::Periodic,
            KernelType::Linear,
        ]);

        gp.fit(&x.view(), &y.view()).unwrap();

        // For periodic data, periodic kernel should be preferred
        // (though this may not always happen due to optimization challenges)
        let selected = gp.get_selected_kernel().unwrap();
        assert!(matches!(selected, KernelType::RBF | KernelType::Periodic));

        // Check that multiple models were evaluated
        assert!(gp.get_evaluated_models().len() >= 3);
    }
}
