//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use scirs2_stats::distributions::bernoulli::Bernoulli as RustBernoulli;
use scirs2_stats::distributions::beta::Beta as RustBeta;
use scirs2_stats::distributions::binomial::Binomial as RustBinomial;
use scirs2_stats::distributions::cauchy::Cauchy as RustCauchy;
use scirs2_stats::distributions::chi_square::ChiSquare as RustChiSquare;
use scirs2_stats::distributions::exponential::Exponential as RustExponential;
use scirs2_stats::distributions::f::F as RustF;
use scirs2_stats::distributions::gamma::Gamma as RustGamma;
use scirs2_stats::distributions::geometric::Geometric as RustGeometric;
use scirs2_stats::distributions::hypergeometric::Hypergeometric as RustHypergeometric;
use scirs2_stats::distributions::laplace::Laplace as RustLaplace;
use scirs2_stats::distributions::logistic::Logistic as RustLogistic;
use scirs2_stats::distributions::lognormal::Lognormal as RustLognormal;
use scirs2_stats::distributions::negative_binomial::NegativeBinomial as RustNegativeBinomial;
use scirs2_stats::distributions::normal::Normal as RustNormal;
use scirs2_stats::distributions::pareto::Pareto as RustPareto;
use scirs2_stats::distributions::poisson::Poisson as RustPoisson;
use scirs2_stats::distributions::student_t::StudentT as RustStudentT;
use scirs2_stats::distributions::uniform::Uniform as RustUniform;
use scirs2_stats::distributions::weibull::Weibull as RustWeibull;
use scirs2_stats::{ContinuousDistribution, DiscreteDistribution};

/// Chi-square distribution
#[pyclass(name = "chi2")]
pub struct PyChiSquare {
    dist: RustChiSquare<f64>,
}
#[pymethods]
impl PyChiSquare {
    /// Create a new Chi-square distribution
    ///
    /// Parameters:
    /// - df: Degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (df, loc = 0.0, scale = 1.0))]
    fn new(df: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustChiSquare::new(df, loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Chi-square distribution creation failed: {}", e))
        })?;
        Ok(PyChiSquare { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Binomial distribution
#[pyclass(name = "binom")]
pub struct PyBinomial {
    dist: RustBinomial<f64>,
}
#[pymethods]
impl PyBinomial {
    /// Create a new Binomial distribution
    ///
    /// Parameters:
    /// - n: Number of trials
    /// - p: Probability of success
    #[new]
    fn new(n: usize, p: f64) -> PyResult<Self> {
        let dist = RustBinomial::new(n, p).map_err(|e| {
            PyRuntimeError::new_err(format!("Binomial distribution creation failed: {}", e))
        })?;
        Ok(PyBinomial { dist })
    }
    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }
    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Geometric distribution
#[pyclass(name = "geom")]
pub struct PyGeometric {
    dist: RustGeometric<f64>,
}
#[pymethods]
impl PyGeometric {
    /// Create a new Geometric distribution
    ///
    /// Parameters:
    /// - p: Success probability, 0 < p <= 1
    #[new]
    fn new(p: f64) -> PyResult<Self> {
        let dist = RustGeometric::new(p).map_err(|e| {
            PyRuntimeError::new_err(format!("Geometric distribution creation failed: {}", e))
        })?;
        Ok(PyGeometric { dist })
    }
    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }
    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Lognormal distribution
#[pyclass(name = "lognorm")]
pub struct PyLognormal {
    dist: RustLognormal<f64>,
}
#[pymethods]
impl PyLognormal {
    /// Create a new Lognormal distribution
    ///
    /// Parameters:
    /// - mu: Mean of underlying normal distribution (default: 0.0)
    /// - sigma: Standard deviation of underlying normal distribution (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (mu = 0.0, sigma = 1.0, loc = 0.0))]
    fn new(mu: f64, sigma: f64, loc: f64) -> PyResult<Self> {
        let dist = RustLognormal::new(mu, sigma, loc).map_err(|e| {
            PyRuntimeError::new_err(format!("Lognormal distribution creation failed: {}", e))
        })?;
        Ok(PyLognormal { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Weibull distribution
#[pyclass(name = "weibull_min")]
pub struct PyWeibull {
    dist: RustWeibull<f64>,
}
#[pymethods]
impl PyWeibull {
    /// Create a new Weibull distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter k > 0
    /// - scale: Scale parameter lambda > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale = 1.0, loc = 0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustWeibull::new(shape, scale, loc).map_err(|e| {
            PyRuntimeError::new_err(format!("Weibull distribution creation failed: {}", e))
        })?;
        Ok(PyWeibull { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Gamma distribution
#[pyclass(name = "gamma")]
pub struct PyGamma {
    dist: RustGamma<f64>,
}
#[pymethods]
impl PyGamma {
    /// Create a new Gamma distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter k > 0
    /// - scale: Scale parameter theta > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale = 1.0, loc = 0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustGamma::new(shape, scale, loc).map_err(|e| {
            PyRuntimeError::new_err(format!("Gamma distribution creation failed: {}", e))
        })?;
        Ok(PyGamma { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Logistic distribution
#[pyclass(name = "logistic")]
pub struct PyLogistic {
    dist: RustLogistic<f64>,
}
#[pymethods]
impl PyLogistic {
    /// Create a new Logistic distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc = 0.0, scale = 1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustLogistic::new(loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Logistic distribution creation failed: {}", e))
        })?;
        Ok(PyLogistic { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Poisson distribution
#[pyclass(name = "poisson")]
pub struct PyPoisson {
    dist: RustPoisson<f64>,
}
#[pymethods]
impl PyPoisson {
    /// Create a new Poisson distribution
    ///
    /// Parameters:
    /// - mu: Expected number of events (lambda parameter)
    #[new]
    fn new(mu: f64) -> PyResult<Self> {
        let dist = RustPoisson::new(mu, 0.0).map_err(|e| {
            PyRuntimeError::new_err(format!("Poisson distribution creation failed: {}", e))
        })?;
        Ok(PyPoisson { dist })
    }
    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }
    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Student's t distribution
#[pyclass(name = "t")]
pub struct PyStudentT {
    dist: RustStudentT<f64>,
}
#[pymethods]
impl PyStudentT {
    /// Create a new Student's t distribution
    ///
    /// Parameters:
    /// - df: Degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (df, loc = 0.0, scale = 1.0))]
    fn new(df: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustStudentT::new(df, loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Student's t distribution creation failed: {}", e))
        })?;
        Ok(PyStudentT { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Pareto distribution
#[pyclass(name = "pareto")]
pub struct PyPareto {
    dist: RustPareto<f64>,
}
#[pymethods]
impl PyPareto {
    /// Create a new Pareto distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter alpha > 0
    /// - scale: Scale parameter x_m > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale = 1.0, loc = 0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustPareto::new(shape, scale, loc).map_err(|e| {
            PyRuntimeError::new_err(format!("Pareto distribution creation failed: {}", e))
        })?;
        Ok(PyPareto { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Exponential distribution
#[pyclass(name = "expon")]
pub struct PyExponential {
    dist: RustExponential<f64>,
}
#[pymethods]
impl PyExponential {
    /// Create a new Exponential distribution
    ///
    /// Parameters:
    /// - scale: Scale parameter (1/lambda) (default: 1.0)
    #[new]
    #[pyo3(signature = (scale = 1.0))]
    fn new(scale: f64) -> PyResult<Self> {
        let rate = 1.0 / scale;
        let dist = RustExponential::new(rate, 0.0).map_err(|e| {
            PyRuntimeError::new_err(format!("Exponential distribution creation failed: {}", e))
        })?;
        Ok(PyExponential { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Beta distribution
#[pyclass(name = "beta")]
pub struct PyBeta {
    dist: RustBeta<f64>,
}
#[pymethods]
impl PyBeta {
    /// Create a new Beta distribution
    ///
    /// Parameters:
    /// - alpha: Shape parameter alpha > 0
    /// - beta: Shape parameter beta > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (alpha, beta, loc = 0.0, scale = 1.0))]
    fn new(alpha: f64, beta: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustBeta::new(alpha, beta, loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Beta distribution creation failed: {}", e))
        })?;
        Ok(PyBeta { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Normal (Gaussian) distribution
#[pyclass(name = "norm")]
pub struct PyNormal {
    dist: RustNormal<f64>,
}
#[pymethods]
impl PyNormal {
    /// Create a new Normal distribution
    ///
    /// Parameters:
    /// - loc: Mean (location) parameter (default: 0.0)
    /// - scale: Standard deviation (scale) parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (loc = 0.0, scale = 1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustNormal::new(loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Normal distribution creation failed: {}", e))
        })?;
        Ok(PyNormal { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Laplace distribution
#[pyclass(name = "laplace")]
pub struct PyLaplace {
    dist: RustLaplace<f64>,
}
#[pymethods]
impl PyLaplace {
    /// Create a new Laplace distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc = 0.0, scale = 1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustLaplace::new(loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Laplace distribution creation failed: {}", e))
        })?;
        Ok(PyLaplace { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Cauchy distribution
#[pyclass(name = "cauchy")]
pub struct PyCauchy {
    dist: RustCauchy<f64>,
}
#[pymethods]
impl PyCauchy {
    /// Create a new Cauchy distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc = 0.0, scale = 1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustCauchy::new(loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Cauchy distribution creation failed: {}", e))
        })?;
        Ok(PyCauchy { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// F distribution
#[pyclass(name = "f")]
pub struct PyF {
    dist: RustF<f64>,
}
#[pymethods]
impl PyF {
    /// Create a new F distribution
    ///
    /// Parameters:
    /// - dfn: Numerator degrees of freedom > 0
    /// - dfd: Denominator degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (dfn, dfd, loc = 0.0, scale = 1.0))]
    fn new(dfn: f64, dfd: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustF::new(dfn, dfd, loc, scale).map_err(|e| {
            PyRuntimeError::new_err(format!("F distribution creation failed: {}", e))
        })?;
        Ok(PyF { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
/// Uniform distribution
#[pyclass(name = "uniform")]
pub struct PyUniform {
    dist: RustUniform<f64>,
}
#[pymethods]
impl PyUniform {
    /// Create a new Uniform distribution
    ///
    /// Parameters:
    /// - loc: Lower bound (default: 0.0)
    /// - scale: Width (upper - lower) (default: 1.0)
    #[new]
    #[pyo3(signature = (loc = 0.0, scale = 1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustUniform::new(loc, loc + scale).map_err(|e| {
            PyRuntimeError::new_err(format!("Uniform distribution creation failed: {}", e))
        })?;
        Ok(PyUniform { dist })
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist
            .ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Bernoulli distribution
#[pyclass(name = "bernoulli")]
pub struct PyBernoulli {
    dist: RustBernoulli<f64>,
}
#[pymethods]
impl PyBernoulli {
    /// Create a new Bernoulli distribution
    ///
    /// Parameters:
    /// - p: Success probability, 0 <= p <= 1
    #[new]
    fn new(p: f64) -> PyResult<Self> {
        let dist = RustBernoulli::new(p).map_err(|e| {
            PyRuntimeError::new_err(format!("Bernoulli distribution creation failed: {}", e))
        })?;
        Ok(PyBernoulli { dist })
    }
    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }
    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, p_val: f64) -> PyResult<f64> {
        self.dist
            .ppf(p_val)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        self.dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))
    }
}

/// Negative Binomial distribution
#[pyclass(name = "nbinom")]
pub struct PyNegativeBinomial {
    dist: RustNegativeBinomial<f64>,
}
#[pymethods]
impl PyNegativeBinomial {
    /// Create a new Negative Binomial distribution
    ///
    /// Parameters:
    /// - n: Number of successes (r parameter), r > 0
    /// - p: Success probability, 0 < p <= 1
    #[new]
    fn new(n: f64, p: f64) -> PyResult<Self> {
        let dist = RustNegativeBinomial::new(n, p).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Negative Binomial distribution creation failed: {}",
                e
            ))
        })?;
        Ok(PyNegativeBinomial { dist })
    }
    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }
    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }
    /// Percent point function (inverse CDF)
    fn ppf(&self, pval: f64) -> PyResult<f64> {
        self.dist
            .ppf(pval)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        self.dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))
    }
}

/// Hypergeometric distribution
#[pyclass(name = "hypergeom")]
pub struct PyHypergeometric {
    dist: RustHypergeometric<f64>,
}
#[pymethods]
impl PyHypergeometric {
    /// Create a new Hypergeometric distribution
    ///
    /// Parameters:
    /// - m: Total population size
    /// - n: Number of success states in population
    /// - k: Number of draws
    #[new]
    fn new(m: usize, n: usize, k: usize) -> PyResult<Self> {
        let dist = RustHypergeometric::new(m, n, k, 0.0_f64).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Hypergeometric distribution creation failed: {}",
                e
            ))
        })?;
        Ok(PyHypergeometric { dist })
    }
    /// Probability mass function
    fn pmf(&self, x: f64) -> f64 {
        self.dist.pmf(x)
    }
    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }
    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self
            .dist
            .rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}
