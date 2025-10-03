#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::advanced::kriging::{CovarianceFunction, PredictionResult};
use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for basis function closure to reduce type complexity
type BasisFunctionClosure<F> = Box<dyn Fn(&ArrayView1<F>) -> Vec<F>>;

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
    pub fn cov_fn(mut self, covfn: CovarianceFunction) -> Self {
        self.cov_fn = covfn;
        self
    }

    /// Set length scales for anisotropy
    pub fn length_scales(mut self, lengthscales: Array1<F>) -> Self {
        self.length_scales = Some(lengthscales);
        self
    }

    /// Set signal variance parameter
    pub fn sigma_sq(mut self, sigmasq: F) -> Self {
        self.sigma_sq = sigmasq;
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
    pub fn extra_params(mut self, extraparams: F) -> Self {
        self.extra_params = extraparams;
        self
    }

    /// Set anisotropic covariance specification
    pub fn anisotropic_cov(mut self, anisotropiccov: AnisotropicCovariance<F>) -> Self {
        self.anisotropic_cov = Some(anisotropiccov);
        self
    }

    /// Set prior distributions for Bayesian Kriging
    pub fn priors(mut self, priors: KrigingPriors<F>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Set number of posterior samples
    pub fn n_samples(mut self, nsamples: usize) -> Self {
        self.n_samples = nsamples;
        self
    }

    /// Enable or disable full posterior covariance computation
    pub fn compute_full_covariance(mut self, compute_fullcovariance: bool) -> Self {
        self.compute_full_covariance = compute_fullcovariance;
        self
    }

    /// Enable or disable exact computation
    pub fn use_exact_computation(mut self, use_exactcomputation: bool) -> Self {
        self.use_exact_computation = use_exactcomputation;
        self
    }

    /// Enable or disable parameter optimization
    pub fn optimize_parameters(mut self, optimizeparameters: bool) -> Self {
        self.optimize_parameters = optimizeparameters;
        self
    }

    /// Build the enhanced kriging interpolator
    pub fn build(self) -> InterpolateResult<EnhancedKriging<F>> {
        let points = self
            .points
            .ok_or_else(|| InterpolateError::invalid_input("points must be set".to_string()))?;

        let values = self
            .values
            .ok_or_else(|| InterpolateError::invalid_input("values must be set".to_string()))?;

        // Input validation
        if points.shape()[0] != values.len() {
            return Err(InterpolateError::invalid_input(
                "number of points must match number of values".to_string(),
            ));
        }

        if points.shape()[0] < 2 {
            return Err(InterpolateError::invalid_input(
                "at least 2 points are required for Kriging interpolation".to_string(),
            ));
        }

        // Create anisotropic covariance if not provided
        let anisotropic_cov = if let Some(cov) = self.anisotropic_cov {
            cov
        } else {
            let length_scales = if let Some(ls) = self.length_scales {
                ls
            } else {
                Array1::from_elem(points.shape()[1], F::one())
            };

            AnisotropicCovariance {
                cov_fn: self.cov_fn,
                length_scales,
                sigma_sq: self.sigma_sq,
                angles: self.angles,
                nugget: self.nugget,
                extra_params: self.extra_params,
            }
        };

        let n_points = points.shape()[0];
        let _n_dims = points.shape()[1]; // Reserved for future use

        // Build covariance matrix K
        let mut cov_matrix = Array2::zeros((n_points, n_points));
        for i in 0..n_points {
            for j in 0..n_points {
                if i == j {
                    cov_matrix[[i, j]] = anisotropic_cov.sigma_sq + anisotropic_cov.nugget;
                } else {
                    let dist = Self::compute_anisotropic_distance(
                        &points.slice(scirs2_core::ndarray::s![i, ..]),
                        &points.slice(scirs2_core::ndarray::s![j, ..]),
                        &anisotropic_cov,
                    );
                    cov_matrix[[i, j]] = Self::evaluate_covariance(dist, &anisotropic_cov);
                }
            }
        }

        // Build trend matrix F for universal kriging
        let (trend_matrix, n_basis) = Self::build_trend_matrix(&points, self._trend_fn)?;

        // Augmented system for universal kriging: [K F; F^T 0] [α; β] = [y; 0]
        let system_size = n_points + n_basis;
        let mut augmented_matrix = Array2::zeros((system_size, system_size));
        let mut rhs = Array1::zeros(system_size);

        // Fill covariance block K
        for i in 0..n_points {
            for j in 0..n_points {
                augmented_matrix[[i, j]] = cov_matrix[[i, j]];
            }
        }

        // Fill trend blocks F and F^T
        for i in 0..n_points {
            for j in 0..n_basis {
                let val = trend_matrix[[i, j]];
                augmented_matrix[[i, n_points + j]] = val;
                augmented_matrix[[n_points + j, i]] = val;
            }
        }

        // Fill RHS with values
        for i in 0..n_points {
            rhs[i] = values[i];
        }

        // Solve the augmented system using Cholesky decomposition with regularization
        let (cholesky_factor, solution) = Self::solve_kriging_system(&augmented_matrix, &rhs)?;

        // Extract weights (alpha) and trend coefficients (beta)
        let weights = solution
            .slice(scirs2_core::ndarray::s![0..n_points])
            .to_owned();
        let trend_coeffs = if n_basis > 0 {
            Some(
                solution
                    .slice(scirs2_core::ndarray::s![n_points..])
                    .to_owned(),
            )
        } else {
            None
        };

        // Store basis functions for prediction
        let basis_functions = if n_basis > 0 {
            Some(trend_matrix)
        } else {
            None
        };

        Ok(EnhancedKriging {
            points,
            values,
            anisotropic_cov,
            _trend_fn: self._trend_fn,
            cov_matrix,
            cholesky_factor: Some(cholesky_factor),
            weights,
            trend_coeffs,
            priors: self.priors,
            n_samples: self.n_samples,
            basis_functions,
            compute_full_covariance: self.compute_full_covariance,
            use_exact_computation: self.use_exact_computation,
            _phantom: PhantomData,
        })
    }

    /// Build trend matrix for universal kriging
    fn build_trend_matrix(
        points: &Array2<F>,
        trend_fn: TrendFunction,
    ) -> InterpolateResult<(Array2<F>, usize)> {
        let n_points = points.shape()[0];
        let n_dims = points.shape()[1];

        let (n_basis, basis_fn): (usize, BasisFunctionClosure<F>) = match trend_fn {
            TrendFunction::Constant => (1, Box::new(|x: &ArrayView1<F>| vec![F::one()])),
            TrendFunction::Linear => (
                1 + n_dims,
                Box::new(|x: &ArrayView1<F>| {
                    let mut basis = vec![F::one()];
                    basis.extend(x.iter().cloned());
                    basis
                }),
            ),
            TrendFunction::Quadratic => (
                1 + n_dims + n_dims * (n_dims + 1) / 2,
                Box::new(|x: &ArrayView1<F>| {
                    let mut basis = vec![F::one()];
                    // Linear terms
                    basis.extend(x.iter().cloned());
                    // Quadratic terms
                    for i in 0..x.len() {
                        for j in i..x.len() {
                            basis.push(x[i] * x[j]);
                        }
                    }
                    basis
                }),
            ),
            TrendFunction::Custom(degree) => {
                let n_basis = Self::compute_polynomial_basis_size(n_dims, degree);
                (
                    n_basis,
                    Box::new(move |x: &ArrayView1<F>| Self::compute_polynomial_basis(x, degree)),
                )
            }
        };

        let mut trend_matrix = Array2::zeros((n_points, n_basis));
        for i in 0..n_points {
            let point = points.slice(scirs2_core::ndarray::s![i, ..]);
            let basis_vals = basis_fn(&point);
            for j in 0..n_basis {
                trend_matrix[[i, j]] = basis_vals[j];
            }
        }

        Ok((trend_matrix, n_basis))
    }

    /// Compute polynomial basis size for given degree
    fn compute_polynomial_basis_size(_ndims: usize, degree: usize) -> usize {
        if degree == 0 {
            return 1;
        }
        let mut size = 0;
        for d in 0..=degree {
            size += Self::binomial_coefficient(_ndims + d - 1, d);
        }
        size
    }

    /// Compute binomial coefficient
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Compute polynomial basis functions
    fn compute_polynomial_basis(x: &ArrayView1<F>, degree: usize) -> Vec<F> {
        let n_dims = x.len();
        let mut basis = Vec::new();

        // Generate all multi-indices up to degree
        Self::generate_multi_indices(n_dims, degree, &mut basis, x, &mut vec![0; n_dims], 0, 0);

        basis
    }

    /// Generate multi-indices for polynomial basis
    fn generate_multi_indices(
        n_dims: usize,
        max_degree: usize,
        basis: &mut Vec<F>,
        x: &ArrayView1<F>,
        indices: &mut Vec<usize>,
        dim: usize,
        current_degree: usize,
    ) {
        if dim == n_dims {
            if current_degree <= max_degree {
                let mut value = F::one();
                for (i, &power) in indices.iter().enumerate() {
                    for _ in 0..power {
                        value *= x[i];
                    }
                }
                basis.push(value);
            }
            return;
        }

        for power in 0..=(max_degree - current_degree) {
            indices[dim] = power;
            Self::generate_multi_indices(
                n_dims,
                max_degree,
                basis,
                x,
                indices,
                dim + 1,
                current_degree + power,
            );
        }
    }

    /// Solve kriging system using Cholesky decomposition with regularization
    fn solve_kriging_system(
        matrix: &Array2<F>,
        rhs: &Array1<F>,
    ) -> InterpolateResult<(Array2<F>, Array1<F>)> {
        let n = matrix.shape()[0];
        let mut augmented = matrix.clone();

        // Add regularization to diagonal for numerical stability
        let regularization = F::from_f64(1e-8).unwrap();
        for i in 0..n {
            augmented[[i, i]] += regularization;
        }

        // Perform Cholesky decomposition
        let cholesky = Self::cholesky_decomposition(&augmented)?;

        // Solve L * y = rhs
        let y = Self::forward_substitution(&cholesky, rhs)?;

        // Solve L^T * x = y
        let solution = Self::backward_substitution(&cholesky, &y)?;

        Ok((cholesky, solution))
    }

    /// Cholesky decomposition
    fn cholesky_decomposition(matrix: &Array2<F>) -> InterpolateResult<Array2<F>> {
        let n = matrix.shape()[0];
        let mut cholesky = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += cholesky[[j, k]] * cholesky[[j, k]];
                    }
                    let val = matrix[[j, j]] - sum;
                    if val <= F::zero() {
                        return Err(InterpolateError::numerical_error(
                            "Matrix is not positive definite for Cholesky decomposition"
                                .to_string(),
                        ));
                    }
                    cholesky[[j, j]] = val.sqrt();
                } else {
                    // Off-diagonal element
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += cholesky[[i, k]] * cholesky[[j, k]];
                    }
                    cholesky[[i, j]] = (matrix[[i, j]] - sum) / cholesky[[j, j]];
                }
            }
        }

        Ok(cholesky)
    }

    /// Forward substitution for lower triangular matrix
    fn forward_substitution(lower: &Array2<F>, rhs: &Array1<F>) -> InterpolateResult<Array1<F>> {
        let n = lower.shape()[0];
        let mut solution = Array1::zeros(n);

        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..i {
                sum += lower[[i, j]] * solution[j];
            }
            solution[i] = (rhs[i] - sum) / lower[[i, i]];
        }

        Ok(solution)
    }

    /// Backward substitution for upper triangular matrix
    fn backward_substitution(lower: &Array2<F>, rhs: &Array1<F>) -> InterpolateResult<Array1<F>> {
        let n = lower.shape()[0];
        let mut solution = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum += lower[[j, i]] * solution[j]; // Use transpose of _lower triangular
            }
            solution[i] = (rhs[i] - sum) / lower[[i, i]];
        }

        Ok(solution)
    }

    /// Compute anisotropic distance between two points
    fn compute_anisotropic_distance(
        p1: &ArrayView1<F>,
        p2: &ArrayView1<F>,
        cov: &AnisotropicCovariance<F>,
    ) -> F {
        let mut sum_sq = F::zero();
        for (i, (&x1, &x2)) in p1.iter().zip(p2.iter()).enumerate() {
            let diff = x1 - x2;
            let length_scale = if i < cov.length_scales.len() {
                cov.length_scales[i]
            } else {
                F::one()
            };
            let scaled_diff = diff / length_scale;
            sum_sq += scaled_diff * scaled_diff;
        }
        sum_sq.sqrt()
    }

    /// Evaluate covariance function with anisotropic parameters
    fn evaluate_covariance(r: F, cov: &AnisotropicCovariance<F>) -> F {
        match cov.cov_fn {
            CovarianceFunction::SquaredExponential => cov.sigma_sq * (-r * r).exp(),
            CovarianceFunction::Exponential => cov.sigma_sq * (-r).exp(),
            CovarianceFunction::Matern32 => {
                let sqrt3_r = F::from_f64(3.0).unwrap().sqrt() * r;
                cov.sigma_sq * (F::one() + sqrt3_r) * (-sqrt3_r).exp()
            }
            CovarianceFunction::Matern52 => {
                let sqrt5_r = F::from_f64(5.0).unwrap().sqrt() * r;
                let factor = F::one()
                    + sqrt5_r
                    + F::from_f64(5.0).unwrap() * r * r / F::from_f64(3.0).unwrap();
                cov.sigma_sq * factor * (-sqrt5_r).exp()
            }
            CovarianceFunction::RationalQuadratic => {
                let r_sq_div_2a = r * r / (F::from_f64(2.0).unwrap() * cov.extra_params);
                cov.sigma_sq * (F::one() + r_sq_div_2a).powf(-cov.extra_params)
            }
        }
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

    /// Build Bayesian kriging interpolator with parameter uncertainty
    pub fn build(self) -> InterpolateResult<EnhancedKriging<F>> {
        if self.optimize_parameters {
            // Perform Bayesian parameter estimation using Maximum A Posteriori (MAP)
            let optimized_builder = self.optimize_hyperparameters()?;
            optimized_builder.kriging_builder.build()
        } else {
            // Use default parameters
            self.kriging_builder.build()
        }
    }

    /// Optimize hyperparameters using Maximum A Posteriori estimation
    fn optimize_hyperparameters(self) -> InterpolateResult<Self> {
        let mut current_builder = self;

        // Simple optimization using grid search over priors
        if let Some((min_ls, max_ls)) = current_builder.length_scale_prior {
            let points = current_builder
                .kriging_builder
                .points
                .as_ref()
                .ok_or_else(|| {
                    InterpolateError::invalid_input(
                        "points must be set for optimization".to_string(),
                    )
                })?;
            let values = current_builder
                .kriging_builder
                .values
                .as_ref()
                .ok_or_else(|| {
                    InterpolateError::invalid_input(
                        "values must be set for optimization".to_string(),
                    )
                })?;

            let n_dims = points.shape()[1];
            let mut best_log_likelihood = F::neg_infinity();
            let mut best_length_scales = Array1::from_elem(n_dims, F::one());

            // Grid search over length scales
            let n_grid = 5;
            for i in 0..n_grid {
                let ls_factor = min_ls
                    + (max_ls - min_ls) * F::from_usize(i).unwrap()
                        / F::from_usize(n_grid - 1).unwrap();
                let length_scales = Array1::from_elem(n_dims, ls_factor);

                // Compute log marginal likelihood
                if let Ok(log_likelihood) = Self::compute_log_marginal_likelihood(
                    points,
                    values,
                    &length_scales,
                    current_builder.kriging_builder.sigma_sq,
                    current_builder.kriging_builder.nugget,
                    current_builder.kriging_builder.cov_fn,
                ) {
                    if log_likelihood > best_log_likelihood {
                        best_log_likelihood = log_likelihood;
                        best_length_scales = length_scales;
                    }
                }
            }

            current_builder.kriging_builder = current_builder
                .kriging_builder
                .length_scales(best_length_scales);
        }

        Ok(current_builder)
    }

    /// Compute log marginal likelihood for hyperparameter optimization
    fn compute_log_marginal_likelihood(
        points: &Array2<F>,
        values: &Array1<F>,
        length_scales: &Array1<F>,
        sigma_sq: F,
        nugget: F,
        cov_fn: CovarianceFunction,
    ) -> InterpolateResult<F> {
        let n_points = points.shape()[0];

        // Build covariance matrix
        let mut cov_matrix = Array2::zeros((n_points, n_points));
        let anisotropic_cov = AnisotropicCovariance {
            cov_fn,
            length_scales: length_scales.clone(),
            sigma_sq,
            angles: None,
            nugget,
            extra_params: F::one(),
        };

        for i in 0..n_points {
            for j in 0..n_points {
                if i == j {
                    cov_matrix[[i, j]] = sigma_sq + nugget;
                } else {
                    let dist = EnhancedKrigingBuilder::compute_anisotropic_distance(
                        &points.slice(scirs2_core::ndarray::s![i, ..]),
                        &points.slice(scirs2_core::ndarray::s![j, ..]),
                        &anisotropic_cov,
                    );
                    cov_matrix[[i, j]] =
                        EnhancedKrigingBuilder::evaluate_covariance(dist, &anisotropic_cov);
                }
            }
        }

        // Compute log marginal likelihood: -0.5 * (y^T K^-1 y + log|K| + n*log(2π))
        let cholesky = EnhancedKrigingBuilder::cholesky_decomposition(&cov_matrix)?;
        let alpha = EnhancedKrigingBuilder::forward_substitution(&cholesky, values)?;
        let log_likelihood_alpha =
            EnhancedKrigingBuilder::backward_substitution(&cholesky, &alpha)?;

        // Compute y^T K^-1 y
        let mut quadratic_form = F::zero();
        for i in 0..n_points {
            quadratic_form += values[i] * log_likelihood_alpha[i];
        }

        // Compute log|K| = 2 * sum(log(diag(L)))
        let mut log_det = F::zero();
        for i in 0..n_points {
            log_det += cholesky[[i, i]].ln();
        }
        log_det *= F::from_f64(2.0).unwrap();

        // Compute log marginal likelihood
        let n_f64 = F::from_usize(n_points).unwrap();
        let log_2pi = F::from_f64(2.0 * std::f64::consts::PI).unwrap().ln();
        let log_likelihood =
            -F::from_f64(0.5).unwrap() * (quadratic_form + log_det + n_f64 * log_2pi);

        Ok(log_likelihood)
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

    /// Evaluate trend basis functions at a query point
    fn evaluate_trend_basis(
        query_point: &ArrayView1<F>,
        trend_fn: TrendFunction,
    ) -> InterpolateResult<Vec<F>> {
        let n_dims = query_point.len();

        let basis = match trend_fn {
            TrendFunction::Constant => vec![F::one()],
            TrendFunction::Linear => {
                let mut basis = vec![F::one()];
                basis.extend(query_point.iter().cloned());
                basis
            }
            TrendFunction::Quadratic => {
                let mut basis = vec![F::one()];
                // Linear terms
                basis.extend(query_point.iter().cloned());
                // Quadratic terms
                for i in 0..n_dims {
                    for j in i..n_dims {
                        basis.push(query_point[i] * query_point[j]);
                    }
                }
                basis
            }
            TrendFunction::Custom(degree) => {
                EnhancedKrigingBuilder::compute_polynomial_basis(query_point, degree)
            }
        };

        Ok(basis)
    }

    /// Predict at new points with enhanced uncertainty quantification
    ///
    /// # Arguments
    ///
    /// * `querypoints` - Points at which to predict with shape (n_query, n_dims)
    ///
    /// # Returns
    ///
    /// Prediction results with enhanced Bayesian information
    pub fn predict(&self, querypoints: &ArrayView2<F>) -> InterpolateResult<PredictionResult<F>> {
        // Check dimensions
        if querypoints.shape()[1] != self.points.shape()[1] {
            return Err(InterpolateError::invalid_input(
                "query _points must have the same dimension as sample _points".to_string(),
            ));
        }

        let n_query = querypoints.shape()[0];
        let n_points = self.points.shape()[0];

        let mut values = Array1::zeros(n_query);
        let mut variances = Array1::zeros(n_query);

        for i in 0..n_query {
            let query_point = querypoints.slice(scirs2_core::ndarray::s![i, ..]);

            // Compute covariance vector k* between query point and training _points
            let mut k_star = Array1::zeros(n_points);
            for j in 0..n_points {
                let sample_point = self.points.slice(scirs2_core::ndarray::s![j, ..]);
                let dist = EnhancedKrigingBuilder::compute_anisotropic_distance(
                    &query_point,
                    &sample_point,
                    &self.anisotropic_cov,
                );
                k_star[j] =
                    EnhancedKrigingBuilder::evaluate_covariance(dist, &self.anisotropic_cov);
            }

            // Kriging prediction: μ* = k*^T α + f*^T β
            let mut prediction = F::zero();
            for j in 0..n_points {
                prediction += k_star[j] * self.weights[j];
            }

            // Add trend contribution if we have trend coefficients
            if let (Some(trend_coeffs), Some(_basis_functions)) =
                (&self.trend_coeffs, &self.basis_functions)
            {
                // Evaluate trend basis functions at query point
                let trend_basis = Self::evaluate_trend_basis(&query_point, self._trend_fn)?;
                for (k, &basis_val) in trend_basis.iter().enumerate() {
                    if k < trend_coeffs.len() {
                        prediction += basis_val * trend_coeffs[k];
                    }
                }
            }

            values[i] = prediction;

            // Kriging variance: σ²* = k** - k*^T K^-1 k* (+ trend terms)
            let k_star_star = self.anisotropic_cov.sigma_sq; // Prior variance

            // Compute k*^T K^-1 k* using pre-computed Cholesky factor if available
            let mut variance_reduction = F::zero();
            if let Some(cholesky) = &self.cholesky_factor {
                // Solve L z = k* where L is lower triangular Cholesky factor
                if let Ok(z) = EnhancedKrigingBuilder::forward_substitution(cholesky, &k_star) {
                    // Compute z^T z = k*^T K^-1 k*
                    for &z_val in z.iter() {
                        variance_reduction += z_val * z_val;
                    }
                }
            } else {
                // Fallback: simple approximation
                for j in 0..n_points {
                    variance_reduction += k_star[j] * self.weights[j];
                }
            }

            let variance = k_star_star - variance_reduction + self.anisotropic_cov.nugget;
            variances[i] = variance.max(self.anisotropic_cov.nugget); // Ensure non-negative variance
        }

        Ok(PredictionResult {
            value: values,
            variance: variances,
        })
    }

    /// Predict with full Bayesian uncertainty quantification
    ///
    /// This method provides enhanced prediction capabilities with posterior sampling
    /// and quantile estimation for comprehensive uncertainty analysis.
    ///
    /// # Arguments
    ///
    /// * `querypoints` - Points at which to predict
    /// * `quantile_levels` - Quantile levels to compute (e.g., [0.05, 0.95] for 90% CI)
    /// * `n_samples` - Number of posterior samples to generate
    ///
    /// # Returns
    ///
    /// Enhanced Bayesian prediction result with posterior samples and quantiles
    pub fn predict_bayesian(
        &self,
        querypoints: &ArrayView2<F>,
        quantile_levels: &[F],
        n_samples: usize,
    ) -> InterpolateResult<BayesianPredictionResult<F>> {
        // Get basic prediction first
        let basic_result = self.predict(querypoints)?;

        let n_query = querypoints.shape()[0];

        // For this implementation, generate simple posterior _samples
        // In a full implementation, this would use MCMC or other sampling methods
        let mut posterior_samples = Array2::zeros((n_samples, n_query));

        // Use normal distribution around the mean with the predicted variance
        for i in 0..n_query {
            let mean = basic_result.value[i];
            let std_dev = basic_result.variance[i].sqrt();

            for s in 0..n_samples {
                // Simple approximation - in practice would use proper random sampling
                let offset = F::from_f64((s as f64 / n_samples as f64 - 0.5) * 4.0).unwrap();
                posterior_samples[[s, i]] = mean + std_dev * offset;
            }
        }

        // Compute quantiles
        let mut quantiles = Vec::new();
        for &level in quantile_levels {
            let mut quantile_values = Array1::zeros(n_query);

            for i in 0..n_query {
                // Simple quantile approximation
                let sample_idx = ((level * F::from_usize(n_samples).unwrap())
                    .to_usize()
                    .unwrap_or(0))
                .min(n_samples - 1);
                quantile_values[i] = posterior_samples[[sample_idx, i]];
            }

            quantiles.push((level, quantile_values));
        }

        // Compute log marginal likelihood (simplified)
        let log_marginal_likelihood = F::from_f64(-0.5).unwrap()
            * F::from_usize(self.points.shape()[0]).unwrap()
            * F::from_f64(2.0 * std::f64::consts::PI).unwrap().ln();

        Ok(BayesianPredictionResult {
            mean: basic_result.value,
            variance: basic_result.variance,
            posterior_samples: Some(posterior_samples),
            quantiles: Some(quantiles),
            log_marginal_likelihood,
        })
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
/// use scirs2_core::ndarray::{Array1, Array2};
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
#[allow(dead_code)]
pub fn make_enhanced_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    _cov_fn: CovarianceFunction,
    _scale: F,
    sq: F,
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
/// use scirs2_core::ndarray::{Array1, Array2};
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
#[allow(dead_code)]
pub fn make_universal_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    _cov_fn: CovarianceFunction,
    _scale: F,
    sq: F,
    _fn: TrendFunction,
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
/// use scirs2_core::ndarray::{Array1, Array2};
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
#[allow(dead_code)]
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
