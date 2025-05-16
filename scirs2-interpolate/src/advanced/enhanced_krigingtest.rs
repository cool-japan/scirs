// Test file with minimal implementation to verify compile-time fix
// This file just mocks the necessary types and implementations
// to verify our fix approach

use thiserror::Error;
use num_traits::{Float, FromPrimitive};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::marker::PhantomData;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub, AddAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceFunction {
    SquaredExponential,
    Exponential,
    Matern32,
    Matern52,
    RationalQuadratic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendFunction {
    Constant,
    Linear,
    Quadratic,
    Custom(usize),
}

#[derive(Debug, Clone)]
pub struct KrigingPriors<F> {
    pub length_scales_prior: ParameterPrior<F>,
    pub variance_prior: ParameterPrior<F>,
    pub nugget_prior: ParameterPrior<F>,
    pub extra_params_prior: ParameterPrior<F>,
    pub trend_coeffs_prior: ParameterPrior<F>,
}

#[derive(Debug, Clone)]
pub enum ParameterPrior<F> {
    Normal(F, F),
    LogNormal(F, F),
    Uniform(F, F),
    InverseGamma(F, F),
    Fixed(F),
}

#[derive(Debug, Clone)]
pub struct AnisotropicCovariance<F> {
    pub cov_fn: CovarianceFunction,
    pub length_scales: Vec<F>,
    pub sigma_sq: F,
    pub nugget: F,
    pub angles: Option<Vec<F>>,
    pub extra_params: F,
    pub _phantom: PhantomData<F>,
}

#[derive(Debug, Clone)]
pub struct EnhancedKriging<F> {
    points: Array2<F>,
    values: Array1<F>,
    anisotropic_cov: AnisotropicCovariance<F>,
    cholesky_factor: Option<Array2<F>>,
    weights: Array1<F>,
    trend_coeffs: Option<Array1<F>>,
    priors: Option<KrigingPriors<F>>,
    n_samples: usize,
    basis_functions: Option<Array2<F>>,
    compute_full_covariance: bool,
    use_exact_computation: bool,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub type InterpolateResult<T> = Result<T, InterpolateError>;

#[derive(Debug, Clone)]
pub struct EnhancedKrigingBuilder<F>
where
    F: Float + FromPrimitive + Debug,
{
    points: Option<Array2<F>>,
    values: Option<Array1<F>>,
    cov_fn: CovarianceFunction,
    length_scales: Option<Array1<F>>,
    sigma_sq: F,
    angles: Option<Array1<F>>,
    nugget: F,
    extra_params: F,
    trend_fn: TrendFunction,
    anisotropic_cov: Option<AnisotropicCovariance<F>>,
    priors: Option<KrigingPriors<F>>,
    n_samples: usize,
    compute_full_covariance: bool,
    use_exact_computation: bool,
    optimize_parameters: bool,
    _phantom: PhantomData<F>,
}

// Helper functions
fn create_basis_functions<F: Float + FromPrimitive>(
    points: &ArrayView2<F>,
    trend_fn: TrendFunction
) -> InterpolateResult<Array2<F>> {
    let n_points = points.shape()[0];
    Ok(Array2::ones((n_points, 1)))
}

fn anisotropic_distance<F: Float + FromPrimitive>(
    p1: &ArrayView1<F>,
    p2: &ArrayView1<F>,
    anisotropic_cov: &AnisotropicCovariance<F>
) -> InterpolateResult<F> {
    Ok(F::one())
}

fn covariance<F: Float + FromPrimitive>(
    r: F,
    anisotropic_cov: &AnisotropicCovariance<F>
) -> F {
    anisotropic_cov.sigma_sq
}

impl<F> AnisotropicCovariance<F>
where
    F: Float + FromPrimitive + Debug,
{
    pub fn new(
        cov_fn: CovarianceFunction,
        length_scales: Vec<F>,
        sigma_sq: F,
        nugget: F,
        angles: Option<Vec<F>>,
    ) -> Self {
        Self {
            cov_fn,
            length_scales,
            sigma_sq,
            nugget,
            angles,
            extra_params: F::one(),
            _phantom: PhantomData,
        }
    }
}

impl<F> EnhancedKrigingBuilder<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + Sub<Output = F> + Div<Output = F> + Mul<Output = F> + Add<Output = F>,
{
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
            trend_fn: TrendFunction::Constant,
            anisotropic_cov: None,
            priors: None,
            n_samples: 0,
            compute_full_covariance: false,
            use_exact_computation: true,
            optimize_parameters: false,
            _phantom: PhantomData,
        }
    }
    
    pub fn points(mut self, points: Array2<F>) -> Self {
        self.points = Some(points);
        self
    }
    
    pub fn values(mut self, values: Array1<F>) -> Self {
        self.values = Some(values);
        self
    }
    
    pub fn covariance_function(mut self, cov_fn: CovarianceFunction) -> Self {
        self.cov_fn = cov_fn;
        self
    }
    
    pub fn anisotropic_covariance(mut self, cov: AnisotropicCovariance<F>) -> Self {
        self.anisotropic_cov = Some(cov);
        self
    }
    
    pub fn optimize_parameters(mut self, optimize: bool) -> Self {
        self.optimize_parameters = optimize;
        self
    }
    
    // FIX BEGINS HERE - These methods were previously outside the impl block
    /// Set the covariance function
    pub fn with_covariance_function(mut self, cov_fn: CovarianceFunction) -> Self {
        self.cov_fn = cov_fn;
        self
    }

    /// Set anisotropic length scales (one per dimension)
    pub fn with_length_scales(mut self, length_scales: Array1<F>) -> Self {
        if length_scales.iter().any(|&l| l <= F::zero()) {
            panic!("Length scales must be positive");
        }
        self.length_scales = Some(length_scales);
        self
    }

    /// Set a single isotropic length scale
    pub fn with_length_scale(mut self, length_scale: F) -> Self {
        if length_scale <= F::zero() {
            panic!("Length scale must be positive");
        }
        self.length_scales = None; // Will be expanded in build
        self.sigma_sq = length_scale;
        self
    }

    /// Set the signal variance
    pub fn with_sigma_sq(mut self, sigma_sq: F) -> Self {
        if sigma_sq <= F::zero() {
            panic!("Signal variance must be positive");
        }
        self.sigma_sq = sigma_sq;
        self
    }

    /// Set anisotropy angles (rotation angles in radians)
    pub fn with_angles(mut self, angles: Array1<F>) -> Self {
        self.angles = Some(angles);
        self
    }

    /// Set the nugget parameter
    pub fn with_nugget(mut self, nugget: F) -> Self {
        if nugget < F::zero() {
            panic!("Nugget must be non-negative");
        }
        self.nugget = nugget;
        self
    }

    /// Set extra parameters for specific covariance functions
    pub fn with_extra_params(mut self, extra_params: F) -> Self {
        self.extra_params = extra_params;
        self
    }

    /// Set the trend function
    pub fn with_trend_function(mut self, trend_fn: TrendFunction) -> Self {
        self.trend_fn = trend_fn;
        self
    }

    /// Set priors for Bayesian Kriging
    pub fn with_priors(mut self, priors: KrigingPriors<F>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Set the number of posterior samples to generate
    pub fn with_posterior_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set whether to compute full posterior covariance
    pub fn with_full_covariance(mut self, compute_full_covariance: bool) -> Self {
        self.compute_full_covariance = compute_full_covariance;
        self
    }

    /// Set whether to use exact computation methods (slower but more accurate)
    pub fn with_exact_computation(mut self, use_exact_computation: bool) -> Self {
        self.use_exact_computation = use_exact_computation;
        self
    }
    
    // Simplified build function for testing
    pub fn build(self) -> InterpolateResult<EnhancedKriging<F>> {
        // Basic validation
        let points = match self.points {
            Some(p) => p,
            None => return Err(InterpolateError::InvalidValue(
                "Points must be provided".to_string()
            )),
        };
        
        let values = match self.values {
            Some(v) => v,
            None => return Err(InterpolateError::InvalidValue(
                "Values must be provided".to_string()
            )),
        };
        
        // Simplified build to verify syntax
        let kriging = EnhancedKriging {
            points,
            values,
            anisotropic_cov: match &self.anisotropic_cov {
                Some(cov) => cov.clone(),
                None => AnisotropicCovariance::new(
                    self.cov_fn,
                    vec![F::one()],
                    self.sigma_sq,
                    self.nugget,
                    None,
                ),
            },
            cholesky_factor: None,
            weights: Array1::zeros(0),
            trend_coeffs: None,
            priors: self.priors,
            n_samples: self.n_samples,
            basis_functions: None,
            compute_full_covariance: self.compute_full_covariance,
            use_exact_computation: self.use_exact_computation,
            _phantom: PhantomData,
        };
        
        Ok(kriging)
    }
}

impl<F> EnhancedKriging<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + Sub<Output = F> + Div<Output = F> + Mul<Output = F> + Add<Output = F>,
{
    pub fn builder() -> EnhancedKrigingBuilder<F> {
        EnhancedKrigingBuilder::new()
    }
    
    pub fn optimize_hyperparameters(&mut self) -> InterpolateResult<()> {
        Ok(())
    }
}

fn main() {
    println!("Enhanced Kriging Builder Test");
    
    // This verifies that our fix for the EnhancedKrigingBuilder methods works
    let _builder = EnhancedKrigingBuilder::<f64>::new()
        .with_covariance_function(CovarianceFunction::Matern52)
        .with_length_scale(1.0)
        .with_nugget(0.001);
        
    println!("All methods compile correctly!");
}