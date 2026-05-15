//! Statistical distribution traits
//!
//! This module defines the core traits for statistical distributions,
//! including standard distributions and specialized circular distributions.

use crate::error::StatsResult;
use scirs2_core::ndarray::{Array, Array1, ArrayD, ArrayView1, IxDyn};
use scirs2_core::numeric::Float;

/// Base trait for all statistical distributions
pub trait Distribution<F: Float> {
    /// Mean (expected value) of the distribution
    fn mean(&self) -> F;

    /// Variance of the distribution
    fn var(&self) -> F;

    /// Standard deviation of the distribution
    fn std(&self) -> F;

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// An array of random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Array1<F>>;

    /// Generate random samples with a user-specified output shape.
    ///
    /// This is the SciPy-style ergonomic where `dist.rvs_array(&[8, 6])` returns
    /// an `8 × 6` array of samples drawn IID from this distribution. The default
    /// implementation draws `prod(shape)` scalar samples through `rvs(...)` and
    /// reshapes them into a dynamically-dimensioned array.
    ///
    /// # Arguments
    ///
    /// * `shape` - Desired output shape. An empty shape produces a 0-D scalar array.
    ///
    /// # Returns
    ///
    /// A dynamic-rank array of IID samples shaped exactly as requested.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `rvs` call fails or if the requested
    /// shape's element count overflows `usize`.
    fn rvs_array(&self, shape: &[usize]) -> StatsResult<ArrayD<F>> {
        // Compute total element count, guarding against multiplication overflow.
        let mut total: usize = 1;
        for &dim in shape {
            total = total.checked_mul(dim).ok_or_else(|| {
                crate::error::StatsError::InvalidArgument(
                    "Requested rvs_array shape overflows usize".to_string(),
                )
            })?;
        }

        // Empty shape ⇒ 0-D array containing a single sample.
        // Zero total ⇒ build an empty array with the requested shape (e.g. shape=[3, 0]).
        if total == 0 {
            return Array::from_shape_vec(IxDyn(shape), Vec::<F>::new()).map_err(|e| {
                crate::error::StatsError::DimensionMismatch(format!(
                    "Failed to construct empty rvs_array with the requested shape: {e}"
                ))
            });
        }

        let flat = self.rvs(total)?;
        Array::from_shape_vec(IxDyn(shape), flat.to_vec()).map_err(|e| {
            crate::error::StatsError::DimensionMismatch(format!(
                "Failed to reshape rvs samples into requested shape: {e}"
            ))
        })
    }

    /// Entropy of the distribution
    fn entropy(&self) -> F;
}

/// Trait for continuous distributions
pub trait ContinuousDistribution<F: Float>: Distribution<F> {
    /// Probability density function (PDF)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// The probability density at x
    fn pdf(&self, x: F) -> F;

    /// Cumulative distribution function (CDF)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// The cumulative probability up to x
    fn cdf(&self, x: F) -> F;

    /// Percent point function (inverse CDF)
    ///
    /// # Arguments
    ///
    /// * `q` - Quantile (probability) in [0, 1]
    ///
    /// # Returns
    ///
    /// The value x such that CDF(x) = q
    fn ppf(&self, q: F) -> StatsResult<F> {
        // Default implementation using binary search
        // This can be overridden for distributions with analytical ppf
        if q < F::zero() || q > F::one() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Quantile must be in [0, 1]".to_string(),
            ));
        }

        // Use binary search to find the inverse
        let mut low = F::from(-10.0).expect("Failed to convert constant to float");
        let mut high = F::from(10.0).expect("Failed to convert constant to float");
        let eps = F::from(1e-12).expect("Failed to convert constant to float");

        // Find a reasonable search range
        while self.cdf(low) > q {
            low = low * F::from(2.0).expect("Failed to convert constant to float");
        }
        while self.cdf(high) < q {
            high = high * F::from(2.0).expect("Failed to convert constant to float");
        }

        // Binary search
        for _ in 0..100 {
            let mid = (low + high) / F::from(2.0).expect("Failed to convert constant to float");
            let cdf_mid = self.cdf(mid);

            if (cdf_mid - q).abs() < eps {
                return Ok(mid);
            }

            if cdf_mid < q {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok((low + high) / F::from(2.0).expect("Failed to convert constant to float"))
    }

    /// Vectorised PDF: evaluate `pdf` at every element of `x`.
    ///
    /// SciPy-style ergonomic — accepts a 1-D `ArrayView1` and returns an owned
    /// `Array1` with the same length. The default implementation broadcasts
    /// the scalar `pdf` via `mapv`. Distributions that can compute the PDF
    /// faster on a batch are free to override this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::Normal;
    /// use scirs2_stats::traits::ContinuousDistribution;
    ///
    /// let n = Normal::new(0.0_f64, 1.0).expect("normal");
    /// let xs = array![-1.0, 0.0, 1.0];
    /// let pdfs = n.pdf_array(&xs.view());
    /// assert_eq!(pdfs.len(), 3);
    /// ```
    fn pdf_array(&self, x: &ArrayView1<F>) -> Array1<F> {
        x.mapv(|xi| self.pdf(xi))
    }

    /// Vectorised CDF: evaluate `cdf` at every element of `x`.
    ///
    /// SciPy-style ergonomic. The default implementation broadcasts the scalar
    /// `cdf` via `mapv`; distributions with a fast batch CDF should override.
    fn cdf_array(&self, x: &ArrayView1<F>) -> Array1<F> {
        x.mapv(|xi| self.cdf(xi))
    }

    /// Vectorised PPF: evaluate `ppf` at every element of `q`.
    ///
    /// Returns an error on the first element whose `ppf` call fails (typically
    /// out-of-range quantiles). On success the output array has the same length
    /// as the input view.
    fn ppf_array(&self, q: &ArrayView1<F>) -> StatsResult<Array1<F>> {
        let mut out = Array1::<F>::zeros(q.len());
        for (i, &qi) in q.iter().enumerate() {
            out[i] = self.ppf(qi)?;
        }
        Ok(out)
    }
}

/// Trait for discrete distributions
pub trait DiscreteDistribution<F: Float>: Distribution<F> {
    /// Probability mass function (PMF)
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the PMF
    ///
    /// # Returns
    ///
    /// The probability mass at k
    fn pmf(&self, k: F) -> F;

    /// Cumulative distribution function (CDF)
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// The cumulative probability up to k
    fn cdf(&self, k: F) -> F;

    /// Support of the distribution (range of possible values)
    fn support(&self) -> (Option<F>, Option<F>) {
        (None, None) // Default: unbounded support
    }

    /// Percent point function (inverse CDF)
    ///
    /// Returns the smallest integer `k` in the support such that `CDF(k) >= p`.
    /// This default implementation uses exponential expansion + binary search on
    /// the CDF, so it works for any discrete distribution that provides `cdf()`,
    /// `mean()`, `std()`, and `support()`.
    ///
    /// Distributions with a closed-form quantile function should override this.
    fn ppf(&self, p: F) -> StatsResult<F> {
        // --- Validate input ---
        if p < F::zero() || p > F::one() {
            return Err(crate::error::StatsError::DomainError(
                "Probability must be in [0, 1]".to_string(),
            ));
        }

        // Determine support bounds (integer-valued).
        let (sup_lo, sup_hi) = self.support();
        let lo_bound: i64 = match sup_lo {
            Some(v) => {
                let raw = v.to_f64().unwrap_or(0.0);
                raw.ceil() as i64
            }
            None => i64::MIN / 2,
        };
        let hi_bound: i64 = match sup_hi {
            Some(v) => {
                let raw = v.to_f64().unwrap_or(f64::MAX);
                raw.floor() as i64
            }
            None => i64::MAX / 2,
        };

        // p == 0 → return support minimum.
        if p <= F::zero() {
            let lo_f = F::from(lo_bound.max(0)).ok_or_else(|| {
                crate::error::StatsError::ComputationError(
                    "Cannot convert support lower bound to F".to_string(),
                )
            })?;
            return Ok(lo_f);
        }

        // p == 1 → return support maximum (or +∞ for infinite support).
        if p >= F::one() {
            return match sup_hi {
                Some(v) => Ok(v),
                None => Ok(F::infinity()),
            };
        }

        // --- Derive initial search range from mean ± 10·std ---
        let mean_f64 = self.mean().to_f64().unwrap_or(0.0);
        let std_f64 = self.std().to_f64().unwrap_or(1.0).max(1.0);

        let mut lo: i64 = ((mean_f64 - 10.0 * std_f64).floor() as i64).max(lo_bound);
        let mut hi: i64 = ((mean_f64 + 10.0 * std_f64).ceil() as i64).min(hi_bound);

        // Clamp lo so it is at least support minimum.
        lo = lo.max(lo_bound);
        // Ensure hi > lo.
        if hi <= lo {
            hi = lo + 1;
        }

        // --- Expand lo leftward until CDF(lo) can be < p ---
        // (For distributions starting at 0 this loop usually doesn't run.)
        {
            let mut step: i64 = (std_f64 * 10.0).ceil() as i64 + 1;
            loop {
                let lo_f = F::from(lo).ok_or_else(|| {
                    crate::error::StatsError::ComputationError(
                        "Overflow expanding lower bound in ppf".to_string(),
                    )
                })?;
                if self.cdf(lo_f) < p || lo <= lo_bound {
                    break;
                }
                lo = (lo - step).max(lo_bound);
                step = step.saturating_mul(2);
            }
            lo = lo.max(lo_bound);
        }

        // --- Expand hi rightward until CDF(hi) >= p ---
        {
            let cap: i64 = 1_000_000_000i64.min(hi_bound);
            let mut step: i64 = (std_f64 * 10.0).ceil() as i64 + 1;
            let mut iters = 0usize;
            loop {
                let hi_f = F::from(hi).ok_or_else(|| {
                    crate::error::StatsError::ComputationError(
                        "Overflow expanding upper bound in ppf".to_string(),
                    )
                })?;
                if self.cdf(hi_f) >= p || hi >= cap {
                    break;
                }
                hi = (hi + step).min(cap);
                step = step.saturating_mul(2);
                iters += 1;
                if iters > 100 {
                    break;
                }
            }
        }

        // --- Binary search: smallest integer k in [lo, hi] with CDF(k) >= p ---
        // Invariant: CDF(hi) >= p (if not, hi is the best we have).
        let mut left: i64 = lo;
        let mut right: i64 = hi;

        // Check if the right bound actually satisfies the condition.
        let right_f = F::from(right).ok_or_else(|| {
            crate::error::StatsError::ComputationError(
                "Overflow converting hi bound in ppf".to_string(),
            )
        })?;
        if self.cdf(right_f) < p {
            // Support is exhausted; return hi.
            return Ok(right_f);
        }

        while right - left > 1 {
            let mid: i64 = left + (right - left) / 2;
            let mid_f = F::from(mid).ok_or_else(|| {
                crate::error::StatsError::ComputationError(
                    "Overflow converting mid in ppf binary search".to_string(),
                )
            })?;
            if self.cdf(mid_f) >= p {
                right = mid;
            } else {
                left = mid;
            }
        }

        // After the loop, `right` satisfies cdf(right) >= p.
        // Check if `left` also satisfies cdf(left) >= p — if so it is the
        // *smaller* integer and is the correct answer.
        let left_f = F::from(left).ok_or_else(|| {
            crate::error::StatsError::ComputationError(
                "Overflow converting left in ppf".to_string(),
            )
        })?;
        if self.cdf(left_f) >= p {
            return Ok(left_f);
        }

        F::from(right).ok_or_else(|| {
            crate::error::StatsError::ComputationError(
                "Overflow converting result in ppf".to_string(),
            )
        })
    }

    /// Log probability mass function
    fn logpmf(&self, x: F) -> F {
        self.pmf(x).ln()
    }
}

/// Trait for circular distributions (distributions on the unit circle)
pub trait CircularDistribution<F: Float>: Distribution<F> {
    /// Probability density function for circular distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Angle in radians
    ///
    /// # Returns
    ///
    /// The probability density at angle x
    fn pdf(&self, x: F) -> F;

    /// Cumulative distribution function for circular distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Angle in radians
    ///
    /// # Returns
    ///
    /// The cumulative probability up to angle x
    fn cdf(&self, x: F) -> F;

    /// Generate a single random sample
    ///
    /// # Returns
    ///
    /// A single random sample from the distribution
    fn rvs_single(&self) -> StatsResult<F>;

    /// Circular mean (mean direction)
    ///
    /// # Returns
    ///
    /// The mean direction in radians
    fn circular_mean(&self) -> F;

    /// Circular variance
    ///
    /// # Returns
    ///
    /// The circular variance (1 - mean resultant length)
    fn circular_variance(&self) -> F;

    /// Circular standard deviation
    ///
    /// # Returns
    ///
    /// The circular standard deviation in radians
    fn circular_std(&self) -> F;

    /// Mean resultant length
    ///
    /// # Returns
    ///
    /// The mean resultant length (measure of concentration)
    fn mean_resultant_length(&self) -> F;

    /// Concentration parameter
    ///
    /// # Returns
    ///
    /// The concentration parameter of the distribution
    fn concentration(&self) -> F;
}

/// Trait for multivariate distributions
pub trait MultivariateDistribution<F: Float> {
    /// Probability density function for multivariate distributions
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// The probability density at x
    fn pdf(&self, x: &Array1<F>) -> F;

    /// Generate random samples from the multivariate distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// A matrix where each row is a sample
    fn rvs(&self, size: usize) -> StatsResult<scirs2_core::ndarray::Array2<F>>;

    /// Mean vector of the distribution
    fn mean(&self) -> Array1<F>;

    /// Covariance matrix of the distribution
    fn cov(&self) -> scirs2_core::ndarray::Array2<F>;

    /// Dimensionality of the distribution
    fn dim(&self) -> usize;

    /// Log probability density function for multivariate distributions
    fn logpdf(&self, x: &Array1<F>) -> F {
        self.pdf(x).ln()
    }

    /// Generate a single random sample from the multivariate distribution
    fn rvs_single(&self) -> StatsResult<Vec<F>> {
        let samples = self.rvs(1)?;
        Ok(samples.row(0).to_vec())
    }
}

/// Trait for distributions that support fitting to data
pub trait Fittable<F: Float> {
    /// Fit the distribution to observed data
    ///
    /// # Arguments
    ///
    /// * `data` - Observed data points
    ///
    /// # Returns
    ///
    /// A fitted distribution instance
    fn fit(data: &Array1<F>) -> StatsResult<Self>
    where
        Self: Sized;

    /// Maximum likelihood estimation of parameters
    ///
    /// # Arguments
    ///
    /// * `data` - Observed data points
    ///
    /// # Returns
    ///
    /// A tuple of estimated parameters
    fn mle(data: &Array1<F>) -> StatsResult<Vec<F>>;
}

/// Trait for distributions that can be truncated
pub trait Truncatable<F: Float>: Distribution<F> {
    /// Create a truncated version of the distribution
    ///
    /// # Arguments
    ///
    /// * `lower` - Lower bound (None for no lower bound)
    /// * `upper` - Upper bound (None for no upper bound)
    ///
    /// # Returns
    ///
    /// A truncated version of the distribution
    fn truncate(&self, lower: Option<F>, upper: Option<F>)
        -> StatsResult<Box<dyn Distribution<F>>>;
}

/// Trait for continuous distributions that support CDF-related functions
pub trait ContinuousCDF<F: Float>: ContinuousDistribution<F> {
    /// Survival function (1 - CDF)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the survival function
    ///
    /// # Returns
    ///
    /// The survival probability at x (1 - CDF(x))
    fn sf(&self, x: F) -> F {
        F::one() - self.cdf(x)
    }

    /// Hazard function (PDF / (1 - CDF))
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the hazard function
    ///
    /// # Returns
    ///
    /// The hazard rate at x
    fn hazard(&self, x: F) -> F {
        let sf_val = self.sf(x);
        if sf_val == F::zero() {
            F::infinity()
        } else {
            self.pdf(x) / sf_val
        }
    }

    /// Cumulative hazard function (-ln(survival function))
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the cumulative hazard function
    ///
    /// # Returns
    ///
    /// The cumulative hazard at x (-ln(1 - CDF(x)))
    fn cumhazard(&self, x: F) -> F {
        let sf_val = self.sf(x);
        if sf_val <= F::zero() {
            F::infinity()
        } else {
            -sf_val.ln()
        }
    }

    /// Inverse survival function (PPF(1 - q))
    ///
    /// # Arguments
    ///
    /// * `q` - Probability in [0, 1]
    ///
    /// # Returns
    ///
    /// The value x such that SF(x) = q (equivalent to PPF(1 - q))
    fn isf(&self, q: F) -> StatsResult<F> {
        if q < F::zero() || q > F::one() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Probability must be in [0, 1]".to_string(),
            ));
        }
        self.ppf(F::one() - q)
    }
}

/// Trait for discrete distributions that support CDF-related functions
pub trait DiscreteCDF<F: Float>: DiscreteDistribution<F> {
    /// Survival function (1 - CDF)
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the survival function
    ///
    /// # Returns
    ///
    /// The survival probability at k (1 - CDF(k))
    fn sf(&self, k: F) -> F {
        F::one() - self.cdf(k)
    }

    /// Hazard function (PMF / (1 - CDF))
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the hazard function
    ///
    /// # Returns
    ///
    /// The hazard rate at k
    fn hazard(&self, k: F) -> F {
        let sf_val = self.sf(k);
        if sf_val == F::zero() {
            F::infinity()
        } else {
            self.pmf(k) / sf_val
        }
    }

    /// Cumulative hazard function (-ln(survival function))
    ///
    /// # Arguments
    ///
    /// * `k` - Point at which to evaluate the cumulative hazard function
    ///
    /// # Returns
    ///
    /// The cumulative hazard at k (-ln(1 - CDF(k)))
    fn cumhazard(&self, k: F) -> F {
        let sf_val = self.sf(k);
        if sf_val <= F::zero() {
            F::infinity()
        } else {
            -sf_val.ln()
        }
    }

    /// Inverse survival function (PPF(1 - q))
    ///
    /// # Arguments
    ///
    /// * `q` - Probability in [0, 1]
    ///
    /// # Returns
    ///
    /// The value k such that SF(k) = q (equivalent to PPF(1 - q))
    fn isf(&self, q: F) -> StatsResult<F> {
        if q < F::zero() || q > F::one() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Probability must be in [0, 1]".to_string(),
            ));
        }
        self.ppf(F::one() - q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatsResult;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::numeric::Float;

    // -----------------------------------------------------------------------
    // Test fixture: Bernoulli(p) implemented via the default ppf
    // -----------------------------------------------------------------------
    struct TestBernoulli {
        p: f64,
    }

    impl Distribution<f64> for TestBernoulli {
        fn mean(&self) -> f64 {
            self.p
        }
        fn var(&self) -> f64 {
            self.p * (1.0 - self.p)
        }
        fn std(&self) -> f64 {
            self.var().sqrt()
        }
        fn rvs(&self, size: usize) -> StatsResult<Array1<f64>> {
            Ok(Array1::zeros(size))
        }
        fn entropy(&self) -> f64 {
            0.0
        }
    }

    impl DiscreteDistribution<f64> for TestBernoulli {
        fn pmf(&self, k: f64) -> f64 {
            if (k - 0.0).abs() < 1e-9 {
                1.0 - self.p
            } else if (k - 1.0).abs() < 1e-9 {
                self.p
            } else {
                0.0
            }
        }
        fn cdf(&self, k: f64) -> f64 {
            if k < 0.0 {
                0.0
            } else if k < 1.0 {
                1.0 - self.p
            } else {
                1.0
            }
        }
        fn support(&self) -> (Option<f64>, Option<f64>) {
            (Some(0.0), Some(1.0))
        }
        // ppf deliberately NOT overridden → exercises the default
    }

    // -----------------------------------------------------------------------
    // Test fixture: Poisson-like (mu) via default ppf
    // -----------------------------------------------------------------------
    struct TestPoisson {
        mu: f64,
    }

    impl Distribution<f64> for TestPoisson {
        fn mean(&self) -> f64 {
            self.mu
        }
        fn var(&self) -> f64 {
            self.mu
        }
        fn std(&self) -> f64 {
            self.mu.sqrt()
        }
        fn rvs(&self, size: usize) -> StatsResult<Array1<f64>> {
            Ok(Array1::zeros(size))
        }
        fn entropy(&self) -> f64 {
            0.0
        }
    }

    impl DiscreteDistribution<f64> for TestPoisson {
        fn pmf(&self, k: f64) -> f64 {
            if k < 0.0 || (k - k.floor()).abs() > 1e-9 {
                return 0.0;
            }
            let ki = k.round() as u64;
            let mut log_pmf = -(self.mu) + (ki as f64) * self.mu.ln();
            for i in 1..=ki {
                log_pmf -= (i as f64).ln();
            }
            log_pmf.exp()
        }
        fn cdf(&self, k: f64) -> f64 {
            if k < 0.0 {
                return 0.0;
            }
            let ki = k.floor() as u64;
            (0..=ki).map(|i| self.pmf(i as f64)).sum::<f64>()
        }
        fn support(&self) -> (Option<f64>, Option<f64>) {
            (Some(0.0), None)
        }
        // ppf deliberately NOT overridden
    }

    // -----------------------------------------------------------------------
    // Test fixture: Geometric(p) (# failures before first success, starting at 0)
    // CDF(k) = 1 - (1-p)^(k+1)
    // -----------------------------------------------------------------------
    struct TestGeometric {
        p: f64,
    }

    impl Distribution<f64> for TestGeometric {
        fn mean(&self) -> f64 {
            (1.0 - self.p) / self.p
        }
        fn var(&self) -> f64 {
            (1.0 - self.p) / (self.p * self.p)
        }
        fn std(&self) -> f64 {
            self.var().sqrt()
        }
        fn rvs(&self, size: usize) -> StatsResult<Array1<f64>> {
            Ok(Array1::zeros(size))
        }
        fn entropy(&self) -> f64 {
            0.0
        }
    }

    impl DiscreteDistribution<f64> for TestGeometric {
        fn pmf(&self, k: f64) -> f64 {
            if k < 0.0 || (k - k.floor()).abs() > 1e-9 {
                return 0.0;
            }
            let ki = k.round() as u64;
            self.p * (1.0 - self.p).powi(ki as i32)
        }
        fn cdf(&self, k: f64) -> f64 {
            if k < 0.0 {
                return 0.0;
            }
            let ki = k.floor() as u64;
            1.0 - (1.0 - self.p).powi(ki as i32 + 1)
        }
        fn support(&self) -> (Option<f64>, Option<f64>) {
            (Some(0.0), None)
        }
        // ppf deliberately NOT overridden
    }

    // -----------------------------------------------------------------------
    // Helper: verify cdf(ppf(p)) >= p  (fundamental discrete PPF invariant)
    // Also verifies ppf(p) is the SMALLEST such k.
    // -----------------------------------------------------------------------
    fn check_ppf_invariant<D: DiscreteDistribution<f64>>(dist: &D, p: f64) {
        let k = dist
            .ppf(p)
            .unwrap_or_else(|e| panic!("ppf({p}) failed: {e}"));
        let cdf_k = dist.cdf(k);
        assert!(
            cdf_k >= p - 1e-12,
            "CDF({k}) = {cdf_k} < p = {p}: invariant cdf(ppf(p)) >= p violated"
        );
        // Verify that k-1 does NOT satisfy the condition (k is smallest).
        if k >= 1.0 {
            let cdf_km1 = dist.cdf(k - 1.0);
            assert!(
                cdf_km1 < p + 1e-12,
                "CDF({}) = {} >= p = {p}: k={k} is not the SMALLEST such integer",
                k - 1.0,
                cdf_km1
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bernoulli(p=0.3) tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_ppf_bernoulli_invariants() {
        let b = TestBernoulli { p: 0.3 };
        for &p in &[0.1f64, 0.25, 0.5, 0.7, 0.9] {
            check_ppf_invariant(&b, p);
        }
    }

    #[test]
    fn test_default_ppf_bernoulli_known_values() {
        let b = TestBernoulli { p: 0.3 };
        // CDF(0) = 0.7, so ppf(p) = 0 for p <= 0.7 and 1 for p > 0.7
        assert_eq!(b.ppf(0.5).expect("ppf(0.5)"), 0.0);
        assert_eq!(b.ppf(0.7).expect("ppf(0.7)"), 0.0);
        assert_eq!(b.ppf(0.8).expect("ppf(0.8)"), 1.0);
        assert_eq!(b.ppf(1.0).expect("ppf(1.0)"), 1.0);
    }

    #[test]
    fn test_default_ppf_bernoulli_p0_returns_support_lo() {
        let b = TestBernoulli { p: 0.7 };
        // p == 0 → support minimum = 0
        assert_eq!(b.ppf(0.0).expect("ppf(0.0)"), 0.0);
    }

    #[test]
    fn test_default_ppf_bernoulli_p1_returns_support_hi() {
        let b = TestBernoulli { p: 0.7 };
        // p == 1 → support maximum = 1
        assert_eq!(b.ppf(1.0).expect("ppf(1.0)"), 1.0);
    }

    #[test]
    fn test_default_ppf_bernoulli_out_of_range() {
        let b = TestBernoulli { p: 0.5 };
        assert!(b.ppf(-0.1).is_err());
        assert!(b.ppf(1.1).is_err());
    }

    // -----------------------------------------------------------------------
    // Poisson(mu=3) tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_ppf_poisson_invariants() {
        let po = TestPoisson { mu: 3.0 };
        for &p in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
            check_ppf_invariant(&po, p);
        }
    }

    #[test]
    fn test_default_ppf_poisson_median() {
        let po = TestPoisson { mu: 3.0 };
        // Median of Poisson(3) is 3
        let med = po.ppf(0.5).expect("ppf(0.5)");
        assert!(
            (med - 3.0).abs() <= 1.0,
            "Expected median near 3, got {med}"
        );
    }

    #[test]
    fn test_default_ppf_poisson_p0_returns_zero() {
        let po = TestPoisson { mu: 5.0 };
        assert_eq!(po.ppf(0.0).expect("ppf(0.0)"), 0.0);
    }

    #[test]
    fn test_default_ppf_poisson_p1_returns_infinity() {
        let po = TestPoisson { mu: 5.0 };
        // Poisson has no finite upper bound → +∞
        let v = po.ppf(1.0).expect("ppf(1.0)");
        assert!(v.is_infinite() && v > 0.0, "Expected +∞, got {v}");
    }

    #[test]
    fn test_default_ppf_poisson_large_mu_invariants() {
        let po = TestPoisson { mu: 50.0 };
        for &p in &[0.1f64, 0.5, 0.9] {
            check_ppf_invariant(&po, p);
        }
    }

    // -----------------------------------------------------------------------
    // Geometric(p=0.4) tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_ppf_geometric_invariants() {
        let g = TestGeometric { p: 0.4 };
        for &p in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
            check_ppf_invariant(&g, p);
        }
    }

    #[test]
    fn test_default_ppf_geometric_p0_returns_zero() {
        let g = TestGeometric { p: 0.3 };
        assert_eq!(g.ppf(0.0).expect("ppf(0.0)"), 0.0);
    }

    #[test]
    fn test_default_ppf_geometric_p1_returns_infinity() {
        let g = TestGeometric { p: 0.3 };
        let v = g.ppf(1.0).expect("ppf(1.0)");
        assert!(v.is_infinite() && v > 0.0, "Expected +∞, got {v}");
    }

    #[test]
    fn test_default_ppf_geometric_heavy_tail() {
        // Small p → heavy tail, exercises exponential expansion of hi
        let g = TestGeometric { p: 0.05 };
        for &p in &[0.5f64, 0.9, 0.99] {
            check_ppf_invariant(&g, p);
        }
    }
}
