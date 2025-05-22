//! Statistical distributions
//!
//! This module provides statistical distribution functions and objects,
//! following SciPy's `stats.distributions` module.

use crate::error::StatsResult;

// Export distributions
pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi_square;
// pub mod circular; // Temporarily disabled
pub mod exponential;
pub mod f;
pub mod gamma;
pub mod geometric;
pub mod hypergeometric;
pub mod laplace;
pub mod logistic;
pub mod lognormal;
pub mod multivariate;
pub mod negative_binomial;
pub mod normal;
pub mod pareto;
pub mod poisson;
pub mod student_t;
pub mod uniform;
pub mod weibull;

// Re-export distribution functions
pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use binomial::Binomial;
pub use cauchy::Cauchy;
pub use chi_square::ChiSquare;
// pub use circular::{VonMises, WrappedCauchy}; // Temporarily disabled
pub use exponential::Exponential;
pub use f::F;
pub use gamma::Gamma;
pub use geometric::Geometric;
pub use hypergeometric::Hypergeometric;
pub use laplace::Laplace;
pub use logistic::Logistic;
pub use lognormal::Lognormal;
pub use multivariate::{
    Dirichlet, InverseWishart, Multinomial, MultivariateLognormal, MultivariateNormal,
    MultivariateT, Wishart,
};
pub use negative_binomial::NegativeBinomial;
pub use normal::Normal;
pub use pareto::Pareto;
pub use poisson::Poisson;
pub use student_t::StudentT;
pub use uniform::Uniform;
pub use weibull::Weibull;

/// Create a normal distribution with the given parameters.
///
/// This is a convenience function to create a normal distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (mean)
/// * `scale` - Scale parameter (standard deviation)
///
/// # Returns
///
/// * A normal distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let normal = distributions::norm(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = normal.pdf(0.0);
/// assert!((pdf_at_zero - 0.3989423).abs() < 1e-7);
/// ```
pub fn norm<F>(loc: F, scale: F) -> StatsResult<Normal<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Normal::new(loc, scale)
}

/// Create a uniform distribution with the given parameters.
///
/// This is a convenience function to create a uniform distribution with
/// the given lower and upper bounds.
///
/// # Arguments
///
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
///
/// # Returns
///
/// * A uniform distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let unif = distributions::uniform(0.0f64, 1.0).unwrap();
/// let pdf_at_half = unif.pdf(0.5);
/// assert!((pdf_at_half - 1.0).abs() < 1e-10);
/// ```
pub fn uniform<F>(low: F, high: F) -> StatsResult<Uniform<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Uniform::new(low, high)
}

/// Create a Student's t distribution with the given parameters.
///
/// This is a convenience function to create a Student's t distribution with
/// the given degrees of freedom, location, and scale parameters.
///
/// # Arguments
///
/// * `df` - Degrees of freedom (> 0)
/// * `loc` - Location parameter (default: 0)
/// * `scale` - Scale parameter (default: 1, must be > 0)
///
/// # Returns
///
/// * A Student's t distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let t = distributions::t(5.0f64, 0.0, 1.0).unwrap();
/// let pdf_at_zero = t.pdf(0.0);
/// assert!((pdf_at_zero - 0.3796).abs() < 1e-4);
/// ```
pub fn t<F>(df: F, loc: F, scale: F) -> StatsResult<StudentT<F>>
where
    F: num_traits::Float + num_traits::NumCast + std::marker::Send + std::marker::Sync + 'static,
{
    StudentT::new(df, loc, scale)
}

/// Create a Chi-square distribution with the given parameters.
///
/// This is a convenience function to create a Chi-square distribution with
/// the given degrees of freedom, location, and scale parameters.
///
/// # Arguments
///
/// * `df` - Degrees of freedom (> 0)
/// * `loc` - Location parameter (default: 0)
/// * `scale` - Scale parameter (default: 1, must be > 0)
///
/// # Returns
///
/// * A Chi-square distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let chi2 = distributions::chi2(2.0f64, 0.0, 1.0).unwrap();
/// let pdf_at_one = chi2.pdf(1.0);
/// assert!((pdf_at_one - 0.303).abs() < 1e-3);
/// ```
pub fn chi2<F>(df: F, loc: F, scale: F) -> StatsResult<ChiSquare<F>>
where
    F: num_traits::Float + num_traits::NumCast + std::marker::Send + std::marker::Sync + 'static,
{
    ChiSquare::new(df, loc, scale)
}

/// Create an F distribution with the given parameters.
///
/// This is a convenience function to create an F distribution with
/// the given degrees of freedom, location, and scale parameters.
///
/// # Arguments
///
/// * `dfn` - Numerator degrees of freedom (> 0)
/// * `dfd` - Denominator degrees of freedom (> 0)
/// * `loc` - Location parameter (default: 0)
/// * `scale` - Scale parameter (default: 1, must be > 0)
///
/// # Returns
///
/// * An F distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let f_dist = distributions::f(2.0f64, 10.0, 0.0, 1.0).unwrap();
/// let pdf_at_one = f_dist.pdf(1.0);
/// assert!((pdf_at_one - 0.335).abs() < 1e-3);
/// ```
pub fn f<T>(dfn: T, dfd: T, loc: T, scale: T) -> StatsResult<F<T>>
where
    T: num_traits::Float + num_traits::NumCast,
{
    F::new(dfn, dfd, loc, scale)
}

/// Create a Poisson distribution with the given parameters.
///
/// This is a convenience function to create a Poisson distribution with
/// the given rate (mean) and location parameters.
///
/// # Arguments
///
/// * `mu` - Rate parameter (mean) > 0
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Poisson distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let poisson = distributions::poisson(3.0f64, 0.0).unwrap();
/// let pmf_at_two = poisson.pmf(2.0);
/// assert!((pmf_at_two - 0.224).abs() < 1e-3);
/// ```
pub fn poisson<F>(mu: F, loc: F) -> StatsResult<Poisson<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Poisson::new(mu, loc)
}

/// Create a Gamma distribution with the given parameters.
///
/// This is a convenience function to create a Gamma distribution with
/// the given shape, scale, and location parameters.
///
/// # Arguments
///
/// * `shape` - Shape parameter (k or α) > 0
/// * `scale` - Scale parameter (θ) > 0
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Gamma distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let gamma = distributions::gamma(2.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_one = gamma.pdf(1.0);
/// assert!((pdf_at_one - 0.3678794).abs() < 1e-6);
/// ```
pub fn gamma<F>(shape: F, scale: F, loc: F) -> StatsResult<Gamma<F>>
where
    F: num_traits::Float
        + num_traits::NumCast
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static,
{
    Gamma::new(shape, scale, loc)
}

/// Create a Beta distribution with the given parameters.
///
/// This is a convenience function to create a Beta distribution with
/// the given alpha, beta, location, and scale parameters.
///
/// # Arguments
///
/// * `alpha` - Shape parameter α > 0
/// * `beta` - Shape parameter β > 0
/// * `loc` - Location parameter (default: 0)
/// * `scale` - Scale parameter (default: 1, must be > 0)
///
/// # Returns
///
/// * A Beta distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// // Special case: beta(2,3)
/// let beta = distributions::beta(2.0f64, 3.0, 0.0, 1.0).unwrap();
/// // This should be around 1.875 (exact: 15/8 = 1.875)
/// assert!((beta.pdf(0.5) - 1.875).abs() < 1e-3);
/// ```
pub fn beta<F>(alpha: F, beta: F, loc: F, scale: F) -> StatsResult<Beta<F>>
where
    F: num_traits::Float + num_traits::NumCast + std::fmt::Debug,
{
    Beta::new(alpha, beta, loc, scale)
}

/// Create an Exponential distribution with the given parameters.
///
/// This is a convenience function to create an Exponential distribution with
/// the given rate and location parameters.
///
/// # Arguments
///
/// * `rate` - Rate parameter λ > 0 (inverse of scale)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * An Exponential distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let exp = distributions::expon(1.0f64, 0.0).unwrap();
/// let pdf_at_one = exp.pdf(1.0);
/// assert!((pdf_at_one - 0.36787944).abs() < 1e-7);
/// ```
pub fn expon<F>(rate: F, loc: F) -> StatsResult<Exponential<F>>
where
    F: num_traits::Float + num_traits::NumCast + std::fmt::Debug,
{
    Exponential::new(rate, loc)
}

/// Create a Lognormal distribution with the given parameters.
///
/// This is a convenience function to create a Lognormal distribution with
/// the given mu (mean of the underlying normal distribution), sigma (standard deviation
/// of the underlying normal distribution), and location parameters.
///
/// # Arguments
///
/// * `mu` - Mean of the underlying normal distribution
/// * `sigma` - Standard deviation of the underlying normal distribution
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Lognormal distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let lognorm = distributions::lognorm(0.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_one = lognorm.pdf(1.0);
/// assert!((pdf_at_one - 0.3989423).abs() < 1e-7);
/// ```
pub fn lognorm<F>(mu: F, sigma: F, loc: F) -> StatsResult<Lognormal<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Lognormal::new(mu, sigma, loc)
}

/// Create a Weibull distribution with the given parameters.
///
/// This is a convenience function to create a Weibull distribution with
/// the given shape, scale, and location parameters.
///
/// # Arguments
///
/// * `shape` - Shape parameter (k > 0)
/// * `scale` - Scale parameter (lambda > 0)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Weibull distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let w = distributions::weibull(2.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_one = w.pdf(1.0);
/// assert!((pdf_at_one - 0.73575888).abs() < 1e-7);
/// ```
pub fn weibull<F>(shape: F, scale: F, loc: F) -> StatsResult<Weibull<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Weibull::new(shape, scale, loc)
}

/// Create a Pareto distribution with the given parameters.
///
/// This is a convenience function to create a Pareto distribution with
/// the given shape, scale, and location parameters.
///
/// # Arguments
///
/// * `shape` - Shape parameter (alpha > 0)
/// * `scale` - Scale parameter (x_m > 0)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Pareto distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let p = distributions::pareto(3.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_two = p.pdf(2.0);
/// assert!((pdf_at_two - 0.1875).abs() < 1e-7);
/// ```
pub fn pareto<F>(shape: F, scale: F, loc: F) -> StatsResult<Pareto<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Pareto::new(shape, scale, loc)
}

/// Create a Cauchy distribution with the given parameters.
///
/// This is a convenience function to create a Cauchy distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (median of the distribution)
/// * `scale` - Scale parameter (half-width at half-maximum) > 0
///
/// # Returns
///
/// * A Cauchy distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let c = distributions::cauchy(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = c.pdf(0.0);
/// assert!((pdf_at_zero - 0.3183099).abs() < 1e-7);
/// ```
pub fn cauchy<F>(loc: F, scale: F) -> StatsResult<Cauchy<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Cauchy::new(loc, scale)
}

/// Create a Laplace distribution with the given parameters.
///
/// This is a convenience function to create a Laplace distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (mean, median, and mode of the distribution)
/// * `scale` - Scale parameter (diversity) > 0
///
/// # Returns
///
/// * A Laplace distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let l = distributions::laplace(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = l.pdf(0.0);
/// assert!((pdf_at_zero - 0.5).abs() < 1e-7);
/// ```
pub fn laplace<F>(loc: F, scale: F) -> StatsResult<Laplace<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Laplace::new(loc, scale)
}

/// Create a Logistic distribution with the given parameters.
///
/// This is a convenience function to create a Logistic distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (mean, median, and mode of the distribution)
/// * `scale` - Scale parameter (diversity) > 0
///
/// # Returns
///
/// * A Logistic distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let l = distributions::logistic(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = l.pdf(0.0);
/// assert!((pdf_at_zero - 0.25).abs() < 1e-7);
/// ```
pub fn logistic<F>(loc: F, scale: F) -> StatsResult<Logistic<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Logistic::new(loc, scale)
}

/// Create a Bernoulli distribution with the given parameter.
///
/// This is a convenience function to create a Bernoulli distribution with
/// the given success probability.
///
/// # Arguments
///
/// * `p` - Success probability (0 ≤ p ≤ 1)
///
/// # Returns
///
/// * A Bernoulli distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let b = distributions::bernoulli(0.3f64).unwrap();
/// let pmf_at_one = b.pmf(1.0);
/// assert!((pmf_at_one - 0.3).abs() < 1e-7);
/// ```
pub fn bernoulli<F>(p: F) -> StatsResult<Bernoulli<F>>
where
    F: num_traits::Float + num_traits::NumCast + std::fmt::Display,
{
    Bernoulli::new(p)
}

/// Create a Binomial distribution with the given parameters.
///
/// This is a convenience function to create a Binomial distribution with
/// the given number of trials and success probability.
///
/// # Arguments
///
/// * `n` - Number of trials (n ≥ 0)
/// * `p` - Success probability (0 ≤ p ≤ 1)
///
/// # Returns
///
/// * A Binomial distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let b = distributions::binom(10, 0.5f64).unwrap();
/// let pmf_at_5 = b.pmf(5.0);
/// assert!((pmf_at_5 - 0.24609375).abs() < 1e-7);
/// ```
pub fn binom<F>(n: usize, p: F) -> StatsResult<Binomial<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Binomial::new(n, p)
}

/// Create a Geometric distribution with the given parameter.
///
/// This is a convenience function to create a Geometric distribution with
/// the given success probability.
///
/// # Arguments
///
/// * `p` - Success probability (0 < p ≤ 1)
///
/// # Returns
///
/// * A Geometric distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let g = distributions::geom(0.3f64).unwrap();
/// let pmf_at_2 = g.pmf(2.0);
/// assert!((pmf_at_2 - 0.147).abs() < 1e-3);
/// ```
pub fn geom<F>(p: F) -> StatsResult<Geometric<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    Geometric::new(p)
}

/// Create a Negative Binomial distribution with the given parameters.
///
/// This is a convenience function to create a Negative Binomial distribution with
/// the given number of successes and success probability.
///
/// # Arguments
///
/// * `r` - Number of successes to achieve (r > 0)
/// * `p` - Success probability (0 < p ≤ 1)
///
/// # Returns
///
/// * A Negative Binomial distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let nb = distributions::nbinom(5.0f64, 0.3).unwrap();
/// let pmf_at_7 = nb.pmf(7.0);
/// assert!((pmf_at_7 - 0.1311) < 1e-4);
/// ```
pub fn nbinom<F>(r: F, p: F) -> StatsResult<NegativeBinomial<F>>
where
    F: num_traits::Float + num_traits::NumCast,
{
    NegativeBinomial::new(r, p)
}

/// Create a Hypergeometric distribution with the given parameters.
///
/// This is a convenience function to create a Hypergeometric distribution with
/// the given population size, number of success states, number of draws, and location parameter.
///
/// # Arguments
///
/// * `n_population` - Population size (N > 0)
/// * `n_success` - Number of success states in the population (K <= N)
/// * `n_draws` - Number of draws (n <= N)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Hypergeometric distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// // Create a hypergeometric distribution
/// // N = 20 (population size)
/// // K = 7 (number of success states in the population)
/// // n = 12 (number of draws)
/// let hyper = distributions::hypergeom(20, 7, 12, 0.0f64).unwrap();
///
/// // Calculate PMF at different points
/// let pmf_3 = hyper.pmf(3.0); // Probability of exactly 3 successes
///
/// // Calculate mean
/// let mean = hyper.mean(); // Should be around 4.2
/// ```
pub fn hypergeom<F>(
    n_population: usize,
    n_success: usize,
    n_draws: usize,
    loc: F,
) -> StatsResult<Hypergeometric<F>>
where
    F: num_traits::Float + num_traits::NumCast + num_traits::FloatConst,
{
    Hypergeometric::new(n_population, n_success, n_draws, loc)
}

// Circular distribution functions temporarily disabled
