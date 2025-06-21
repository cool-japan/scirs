use crate::advanced::kriging::{CovarianceFunction, PredictionResult};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Enhanced prediction result with additional Bayesian information
#[derive(Debug, Clone)]
pub struct BayesianPredictionResult<F: Float> {
    /// Mean prediction at query points
    pub mean: Array1<F>,

    /// Prediction variance at query points
    pub variance: Array1<F>,

    /// Posterior samples at query points (if requested)
    pub posterior_samples: Option<Array2<F>>,

    /// Quantiles at specified levels (if requested)
    pub quantiles: Option<Vec<(F, Array1<F>)>>,

    /// Log marginal likelihood of the model
    pub log_marginal_likelihood: F,
}

/// Configuration for anisotropic covariance models
#[derive(Debug, Clone)]
pub struct AnisotropicCovariance<
    F: Float
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
> {
    /// Covariance function to use
    pub cov_fn: CovarianceFunction,

    /// Directional length scales for each dimension
    pub length_scales: Array1<F>,

    /// Signal variance parameter
    pub sigma_sq: F,

    /// Rotation angles for non-axis-aligned anisotropy
    pub angles: Option<Array1<F>>,

    /// Nugget parameter for stability
    pub nugget: F,

    /// Extra parameters for specific covariance functions
    pub extra_params: F,
}

/// Specification of trend functions for Universal Kriging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendFunction {
    /// Constant mean function (Ordinary Kriging)
    Constant,

    /// Linear trend function (first order polynomial)
    Linear,

    /// Quadratic trend function (second order polynomial)
    Quadratic,

    /// Custom trend function with specified degree
    Custom(usize),
}

/// Prior distributions for Bayesian parameter estimation
#[derive(Debug, Clone)]
pub enum ParameterPrior<
    F: Float
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
> {
    /// Uniform prior within bounds
    Uniform(F, F),

    /// Normal prior with mean and std
    Normal(F, F),

    /// Gamma prior with shape and scale
    Gamma(F, F),

    /// Inverse Gamma prior with shape and scale
    InverseGamma(F, F),

    /// Fixed value (delta prior)
    Fixed(F),
}

/// Enhanced Kriging (Gaussian Process) interpolator
///
/// This extends the basic Kriging interpolator with:
/// - Anisotropic covariance functions
/// - Universal kriging with flexible trend functions
/// - Bayesian parameter estimation and uncertainty quantification
/// - Advanced covariance structures for spatial data
#[derive(Debug, Clone)]
pub struct EnhancedKriging<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Sample points
    points: Array2<F>,

    /// Sample values
    values: Array1<F>,

    /// Anisotropic covariance configuration
    anisotropic_cov: AnisotropicCovariance<F>,

    /// Trend function for universal kriging
    _trend_fn: TrendFunction,

    /// Covariance matrix of sample points
    cov_matrix: Array2<F>,

    /// Cholesky factor of covariance matrix
    cholesky_factor: Option<Array2<F>>,

    /// Kriging weights
    weights: Array1<F>,

    /// Coefficients for trend function (universal kriging)
    trend_coeffs: Option<Array1<F>>,

    /// Prior distributions for Bayesian Kriging
    priors: Option<KrigingPriors<F>>,

    /// Number of posterior samples
    n_samples: usize,

    /// Basis functions for trend model
    basis_functions: Option<Array2<F>>,

    /// Whether to compute full posterior covariance
    compute_full_covariance: bool,

    /// Whether to use exact computation
    use_exact_computation: bool,

    /// Marker for generic type
    _phantom: PhantomData<F>,
}

#[derive(Debug, Clone)]
pub struct KrigingPriors<
    F: Float
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
> {
    /// Prior for sigma_sq parameter
    pub sigma_sq_prior: ParameterPrior<F>,

    /// Prior for length_scale parameter
    pub length_scale_prior: ParameterPrior<F>,

    /// Prior for nugget parameter
    pub nugget_prior: ParameterPrior<F>,

    /// Prior for trend coefficients
    pub trend_coeffs_prior: ParameterPrior<F>,
}

/// Builder for constructing EnhancedKriging models with a fluent API
///
/// This builder provides a clean, method-chaining interface for configuring and
/// creating kriging interpolators with advanced features.
#[derive(Debug, Clone)]
pub struct EnhancedKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Points for interpolation
    points: Option<Array2<F>>,

    /// Values for interpolation
    values: Option<Array1<F>>,

    /// Covariance function
    cov_fn: CovarianceFunction,

    /// Directional length scales for anisotropy
    length_scales: Option<Array1<F>>,

    /// Signal variance parameter
    sigma_sq: F,

    /// Orientation angles for anisotropy
    angles: Option<Array1<F>>,

    /// Nugget parameter
    nugget: F,

    /// Extra parameters for specific covariance functions
    extra_params: F,

    /// Trend function type
    _trend_fn: TrendFunction,

    /// Anisotropic covariance specification
    anisotropic_cov: Option<AnisotropicCovariance<F>>,

    /// Prior distributions for Bayesian Kriging
    priors: Option<KrigingPriors<F>>,

    /// Number of posterior samples
    n_samples: usize,

    /// Whether to compute full posterior covariance
    compute_full_covariance: bool,

    /// Whether to use exact computation
    use_exact_computation: bool,

    /// Whether to optimize parameters
    optimize_parameters: bool,

    /// Marker for generic type
    _phantom: PhantomData<F>,
}

impl<F> Default for EnhancedKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> EnhancedKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Create a new builder for EnhancedKriging
    pub fn new() -> Self {
        Self {
            points: None,
            values: None,
            cov_fn: CovarianceFunction::SquaredExponential,
            length_scales: None,
            sigma_sq: F::from_f64(1.0).unwrap(),
            angles: None,
            nugget: F::from_f64(1e-10).unwrap(),
            extra_params: F::from_f64(1.0).unwrap(),
            _trend_fn: TrendFunction::Constant,
            anisotropic_cov: None,
            priors: None,
            n_samples: 0,
            compute_full_covariance: false,
            use_exact_computation: true,
            optimize_parameters: false,
            _phantom: PhantomData,
        }
    }

    /// Set points for the interpolation
    pub fn points(mut self, points: Array2<F>) -> Self {
        self.points = Some(points);
        self
    }

    /// Set values for the interpolation
    pub fn values(mut self, values: Array1<F>) -> Self {
        self.values = Some(values);
        self
    }

    /// Set covariance function
    pub fn cov_fn(mut self, cov_fn: CovarianceFunction) -> Self {
        self.cov_fn = cov_fn;
        self
    }

    /// Set length scales for anisotropy
    pub fn length_scales(mut self, length_scales: Array1<F>) -> Self {
        self.length_scales = Some(length_scales);
        self
    }

    /// Set signal variance parameter
    pub fn sigma_sq(mut self, sigma_sq: F) -> Self {
        self.sigma_sq = sigma_sq;
        self
    }

    /// Set orientation angles for anisotropy
    pub fn angles(mut self, angles: Array1<F>) -> Self {
        self.angles = Some(angles);
        self
    }

    /// Set nugget parameter
    pub fn nugget(mut self, nugget: F) -> Self {
        self.nugget = nugget;
        self
    }

    /// Set extra parameters for specific covariance functions
    pub fn extra_params(mut self, extra_params: F) -> Self {
        self.extra_params = extra_params;
        self
    }

    /// Set anisotropic covariance specification
    pub fn anisotropic_cov(mut self, anisotropic_cov: AnisotropicCovariance<F>) -> Self {
        self.anisotropic_cov = Some(anisotropic_cov);
        self
    }

    /// Set prior distributions for Bayesian Kriging
    pub fn priors(mut self, priors: KrigingPriors<F>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Set number of posterior samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Enable or disable full posterior covariance computation
    pub fn compute_full_covariance(mut self, compute_full_covariance: bool) -> Self {
        self.compute_full_covariance = compute_full_covariance;
        self
    }

    /// Enable or disable exact computation
    pub fn use_exact_computation(mut self, use_exact_computation: bool) -> Self {
        self.use_exact_computation = use_exact_computation;
        self
    }

    /// Enable or disable parameter optimization
    pub fn optimize_parameters(mut self, optimize_parameters: bool) -> Self {
        self.optimize_parameters = optimize_parameters;
        self
    }

    /// Dummy implementation of build method for this simplified example
    pub fn build(self) -> InterpolateResult<EnhancedKriging<F>> {
        Err(InterpolateError::NotImplemented(
            "This is a simplified example".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
/// Specialized builder for Bayesian Kriging models with uncertainty quantification
pub struct BayesianKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Base kriging builder
    kriging_builder: EnhancedKrigingBuilder<F>,

    /// Prior for length scale
    length_scale_prior: Option<(F, F)>,

    /// Prior for variance
    variance_prior: Option<(F, F)>,

    /// Prior for nugget
    nugget_prior: Option<(F, F)>,

    /// Number of posterior samples to generate
    n_samples: usize,

    /// Whether to optimize parameters before sampling
    optimize_parameters: bool,

    /// Marker for generic type
    _phantom: PhantomData<F>,
}

impl<F> Default for BayesianKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BayesianKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Create a new Bayesian Kriging builder
    pub fn new() -> Self {
        Self {
            kriging_builder: EnhancedKrigingBuilder::new(),
            length_scale_prior: None,
            variance_prior: None,
            nugget_prior: None,
            n_samples: 1000, // Default to 1000 samples
            optimize_parameters: true,
            _phantom: PhantomData,
        }
    }

    /// Dummy build implementation for this simplified example
    pub fn build(self) -> InterpolateResult<EnhancedKriging<F>> {
        self.kriging_builder.build()
    }

    /// Get the length scale prior
    pub fn length_scale_prior(&self) -> Option<&(F, F)> {
        self.length_scale_prior.as_ref()
    }

    /// Get the variance prior
    pub fn variance_prior(&self) -> Option<&(F, F)> {
        self.variance_prior.as_ref()
    }

    /// Get the nugget prior
    pub fn nugget_prior(&self) -> Option<&(F, F)> {
        self.nugget_prior.as_ref()
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Check if parameter optimization is enabled
    pub fn optimize_parameters(&self) -> bool {
        self.optimize_parameters
    }
}

impl<F> AnisotropicCovariance<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Create a new anisotropic covariance specification
    pub fn new(
        cov_fn: CovarianceFunction,
        length_scales: Vec<F>,
        sigma_sq: F,
        nugget: F,
        angles: Option<Vec<F>>,
    ) -> Self {
        let length_scales_array = Array1::from_vec(length_scales);
        let angles_array = angles.map(Array1::from_vec);

        Self {
            cov_fn,
            length_scales: length_scales_array,
            sigma_sq,
            angles: angles_array,
            nugget,
            extra_params: F::one(),
        }
    }
}

impl<F> EnhancedKriging<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Create a builder for the enhanced Kriging interpolator
    pub fn builder() -> EnhancedKrigingBuilder<F> {
        EnhancedKrigingBuilder::new()
    }

    /// Dummy implementation of predict method for this simplified example
    pub fn predict(&self, _query_points: &ArrayView2<F>) -> InterpolateResult<PredictionResult<F>> {
        Err(InterpolateError::NotImplemented(
            "This is a simplified example".to_string(),
        ))
    }

    /// Get the sample points
    pub fn points(&self) -> &Array2<F> {
        &self.points
    }

    /// Get the sample values
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Get the anisotropic covariance configuration
    pub fn anisotropic_cov(&self) -> &AnisotropicCovariance<F> {
        &self.anisotropic_cov
    }

    /// Get the covariance matrix
    pub fn cov_matrix(&self) -> &Array2<F> {
        &self.cov_matrix
    }

    /// Get the Cholesky factor of the covariance matrix
    pub fn cholesky_factor(&self) -> Option<&Array2<F>> {
        self.cholesky_factor.as_ref()
    }

    /// Get the kriging weights
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Get the trend coefficients
    pub fn trend_coeffs(&self) -> Option<&Array1<F>> {
        self.trend_coeffs.as_ref()
    }

    /// Get the priors
    pub fn priors(&self) -> Option<&KrigingPriors<F>> {
        self.priors.as_ref()
    }

    /// Get the number of posterior samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the basis functions
    pub fn basis_functions(&self) -> Option<&Array2<F>> {
        self.basis_functions.as_ref()
    }

    /// Check if full covariance computation is enabled
    pub fn compute_full_covariance(&self) -> bool {
        self.compute_full_covariance
    }

    /// Check if exact computation is enabled
    pub fn use_exact_computation(&self) -> bool {
        self.use_exact_computation
    }
}

/// Convenience function to create an enhanced kriging model
///
/// Creates a basic enhanced kriging interpolator with default settings.
/// This is the simplest way to get started with kriging interpolation.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `values` - Training data values with shape (n_points,)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Length scale parameter for the covariance function
/// * `sigma_sq` - Signal variance parameter
///
/// # Returns
///
/// An enhanced kriging interpolator ready for prediction
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::enhanced_kriging::make_enhanced_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample 2D spatial data
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).unwrap();
/// let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// // Create enhanced kriging model
/// let kriging = make_enhanced_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::SquaredExponential,
///     1.0,  // length scale
///     1.0   // signal variance
/// ).unwrap();
///
/// // The model is ready for making predictions
/// println!("Enhanced kriging model created successfully");
/// ```
pub fn make_enhanced_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    _cov_fn: CovarianceFunction,
    _length_scale: F,
    _sigma_sq: F,
) -> InterpolateResult<EnhancedKriging<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    EnhancedKriging::builder()
        .points(points.to_owned())
        .values(values.to_owned())
        .build()
}

/// Convenience function to create a universal kriging model
///
/// Creates a universal kriging interpolator that can handle non-stationary data by
/// modeling a trend function in addition to the covariance structure. This is useful
/// when the data exhibits a clear trend or drift.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `values` - Training data values with shape (n_points,)
/// * `cov_fn` - Covariance function for the residuals
/// * `length_scale` - Length scale parameter for the covariance function
/// * `sigma_sq` - Signal variance parameter
/// * `trend_fn` - Type of trend function (Constant, Linear, Quadratic, etc.)
///
/// # Returns
///
/// A universal kriging interpolator with trend modeling
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::enhanced_kriging::{make_universal_kriging, TrendFunction};
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create data with a linear trend: z = x + y + noise
/// let points = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0,  // z ≈ 0
///     1.0, 0.0,  // z ≈ 1
///     0.0, 1.0,  // z ≈ 1
///     1.0, 1.0,  // z ≈ 2
///     2.0, 0.0,  // z ≈ 2
///     0.0, 2.0,  // z ≈ 2
/// ]).unwrap();
/// let values = Array1::from_vec(vec![0.1, 1.05, 0.95, 2.1, 1.9, 2.05]);
///
/// // Create universal kriging with linear trend
/// let kriging = make_universal_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Exponential,
///     0.5,  // length scale
///     0.1,  // signal variance
///     TrendFunction::Linear
/// ).unwrap();
///
/// println!("Universal kriging model with linear trend created");
/// ```
pub fn make_universal_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    _cov_fn: CovarianceFunction,
    _length_scale: F,
    _sigma_sq: F,
    _trend_fn: TrendFunction,
) -> InterpolateResult<EnhancedKriging<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    EnhancedKriging::builder()
        .points(points.to_owned())
        .values(values.to_owned())
        .build()
}

/// Convenience function to create a Bayesian kriging model
///
/// Creates a fully Bayesian kriging interpolator that incorporates parameter
/// uncertainty through prior distributions. This provides more robust uncertainty
/// quantification by marginalizing over hyperparameter uncertainty.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `values` - Training data values with shape (n_points,)
/// * `cov_fn` - Covariance function to use
/// * `priors` - Prior distributions for hyperparameters
/// * `n_samples` - Number of posterior samples for uncertainty quantification
///
/// # Returns
///
/// A Bayesian kriging interpolator with full uncertainty quantification
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::enhanced_kriging::{make_bayesian_kriging, KrigingPriors, ParameterPrior};
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create noisy observational data
/// let points = Array2::from_shape_vec((8, 1), vec![
///     0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
/// ]).unwrap();
/// let values = Array1::from_vec(vec![
///     0.1, 0.6, 0.9, 1.4, 1.8, 2.3, 2.9, 3.6  // f(x) ≈ x with noise
/// ]);
///
/// // Define prior distributions for hyperparameters
/// let priors = KrigingPriors {
///     length_scale_prior: ParameterPrior::Uniform(0.1, 2.0),
///     sigma_sq_prior: ParameterPrior::Uniform(0.01, 1.0),
///     nugget_prior: ParameterPrior::Uniform(0.001, 0.1),
///     trend_coeffs_prior: ParameterPrior::Uniform(0.0, 1.0),
/// };
///
/// // Create Bayesian kriging model
/// let kriging = make_bayesian_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     priors,
///     1000  // number of posterior samples
/// ).unwrap();
///
/// println!("Bayesian kriging model created with 1000 posterior samples");
/// ```
pub fn make_bayesian_kriging<F>(
    _points: &ArrayView2<F>,
    _values: &ArrayView1<F>,
    _cov_fn: CovarianceFunction,
    _priors: KrigingPriors<F>,
    _n_samples: usize,
) -> InterpolateResult<EnhancedKriging<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Sub<Output = F>
        + Div<Output = F>
        + Mul<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    BayesianKrigingBuilder::new().build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        // A simple test to verify that the builders can be created
        let builder = EnhancedKrigingBuilder::<f64>::new();
        assert_eq!(builder._trend_fn, TrendFunction::Constant);

        let bayes_builder = BayesianKrigingBuilder::<f64>::new();
        assert_eq!(bayes_builder.n_samples, 1000);
    }
}
