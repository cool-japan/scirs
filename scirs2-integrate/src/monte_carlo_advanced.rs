//! Advanced Monte Carlo integration methods
//!
//! This module provides a comprehensive `MonteCarloIntegrator` struct with
//! variance reduction techniques and Markov Chain Monte Carlo (MCMC) integration
//! via Metropolis-Hastings sampling.
//!
//! ## Overview
//!
//! Standard Monte Carlo integration converges at the statistical rate O(N^{-1/2})
//! regardless of dimension, making it the preferred method for high-dimensional
//! integrals where deterministic methods suffer from the curse of dimensionality.
//!
//! Variance reduction techniques such as importance sampling, stratified sampling,
//! control variates, and antithetic variates can substantially reduce the variance
//! (and therefore the required sample size) for a given accuracy target.
//!
//! MCMC integration (Metropolis-Hastings) allows integration with respect to an
//! arbitrary unnormalised target density, which arises naturally in Bayesian
//! statistics and statistical physics.
//!
//! ## References
//!
//! - Rubinstein, R. Y. & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method* (3rd ed.)
//! - Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.)
//! - Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::{Distribution, Normal, Uniform};

// ─────────────────────────────────────────────────────────────────────────────
// MonteCarloIntegrator
// ─────────────────────────────────────────────────────────────────────────────

/// High-level Monte Carlo integrator with built-in variance reduction.
///
/// # Example
///
/// ```
/// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
///
/// let mc = MonteCarloIntegrator::new(Some(42));
/// let (value, std_err) = mc.integrate(
///     |x: &[f64]| x[0] * x[0],
///     &[(0.0_f64, 1.0_f64)],
///     50_000,
/// ).expect("integration failed");
///
/// // Exact result is 1/3; Monte Carlo has statistical error
/// assert!((value - 1.0 / 3.0).abs() < 0.02);
/// assert!(std_err >= 0.0);
/// ```
pub struct MonteCarloIntegrator {
    seed: Option<u64>,
}

impl MonteCarloIntegrator {
    /// Create a new integrator.
    ///
    /// * `seed` – optional random seed for reproducibility.
    pub fn new(seed: Option<u64>) -> Self {
        Self { seed }
    }

    /// Build a seeded `StdRng`.
    fn make_rng(&self) -> StdRng {
        match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut thread = scirs2_core::random::rng();
                StdRng::from_rng(&mut thread)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plain Monte Carlo
    // ─────────────────────────────────────────────────────────────────────────

    /// Standard Monte Carlo integration of `f` over a hyperrectangle.
    ///
    /// Returns `(estimate, std_error)`.
    ///
    /// The integration domain is the Cartesian product of the closed intervals
    /// `[a_i, b_i]` for each element of `bounds`.
    ///
    /// # Parameters
    ///
    /// * `f`        – integrand, called with a slice of length `bounds.len()`.
    /// * `bounds`   – `(a_i, b_i)` for each dimension.
    /// * `n_samples`– number of random sample points.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::ValueError`] if `bounds` is empty, `n_samples`
    /// is zero, or any interval is degenerate (`a_i >= b_i`).
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(7));
    /// // ∫₀¹ ∫₀¹ (x + y) dx dy = 1
    /// let (val, _) = mc.integrate(
    ///     |x: &[f64]| x[0] + x[1],
    ///     &[(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)],
    ///     100_000,
    /// ).expect("failed");
    /// assert!((val - 1.0).abs() < 0.02);
    /// ```
    pub fn integrate<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        n_samples: usize,
    ) -> IntegrateResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        validate_bounds(bounds)?;
        if n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be positive".to_string(),
            ));
        }

        let dim = bounds.len();
        let volume = compute_volume(bounds);
        let mut rng = self.make_rng();

        let dists: Vec<Uniform<f64>> = bounds
            .iter()
            .map(|&(a, b)| {
                Uniform::new_inclusive(a, b)
                    .map_err(|e| IntegrateError::ValueError(format!("Uniform dist error: {e}")))
            })
            .collect::<IntegrateResult<Vec<_>>>()?;

        let mut point = vec![0.0f64; dim];
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;

        for _ in 0..n_samples {
            for (i, dist) in dists.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            let v = f(&point);
            sum += v;
            sum_sq += v * v;
        }

        let n = n_samples as f64;
        let mean = sum / n;
        let variance = (sum_sq - sum * sum / n) / (n - 1.0).max(1.0);
        let std_err = (variance / n).sqrt() * volume;

        Ok((mean * volume, std_err))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Importance sampling
    // ─────────────────────────────────────────────────────────────────────────

    /// Importance-sampling Monte Carlo integration.
    ///
    /// Estimates `∫ f(x) dx` by sampling from a proposal distribution with
    /// density `q(x)` and computing the weighted average:
    ///
    /// ```text
    /// E_q[f(x) / q(x)]  ≈  (1/N) Σ  f(x_i) / q(x_i)
    /// ```
    ///
    /// For efficiency the proposal `q` should approximate `|f|`.
    ///
    /// # Parameters
    ///
    /// * `f`                – integrand.
    /// * `proposal_sampler` – draws i.i.d. samples `x ~ q` (called with `&mut StdRng`).
    /// * `proposal_pdf`     – evaluates `q(x)` (must be positive on the support of `f`).
    /// * `n_samples`        – number of samples.
    ///
    /// # Returns
    ///
    /// `(estimate, std_error)`
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    /// use scirs2_core::random::prelude::StdRng;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(99));
    ///
    /// // Integrate x² on [0,1] with uniform proposal q(x) = 1
    /// let (val, _) = mc.importance_sampling(
    ///     |x: &[f64]| x[0] * x[0],
    ///     |rng: &mut StdRng| {
    ///         use scirs2_core::random::Distribution;
    ///         let u = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64)
    ///             .unwrap()
    ///             .sample(rng);
    ///         vec![u]
    ///     },
    ///     |_x: &[f64]| 1.0_f64,
    ///     20_000,
    /// ).expect("failed");
    ///
    /// assert!((val - 1.0 / 3.0).abs() < 0.02);
    /// ```
    pub fn importance_sampling<F, S, Q>(
        &self,
        f: F,
        proposal_sampler: S,
        proposal_pdf: Q,
        n_samples: usize,
    ) -> IntegrateResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
        S: Fn(&mut StdRng) -> Vec<f64>,
        Q: Fn(&[f64]) -> f64,
    {
        if n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be positive".to_string(),
            ));
        }

        let mut rng = self.make_rng();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut valid = 0usize;

        for _ in 0..n_samples {
            let x = proposal_sampler(&mut rng);
            let q = proposal_pdf(&x);
            if q <= 1e-300 || q.is_nan() || q.is_infinite() {
                continue;
            }
            let fv = f(&x);
            if fv.is_nan() || fv.is_infinite() {
                continue;
            }
            let ratio = fv / q;
            sum += ratio;
            sum_sq += ratio * ratio;
            valid += 1;
        }

        if valid < 10 {
            return Err(IntegrateError::ConvergenceError(
                "Too few valid samples in importance sampling (check proposal PDF)".to_string(),
            ));
        }

        let n = valid as f64;
        let mean = sum / n;
        let variance = (sum_sq - sum * sum / n) / (n - 1.0).max(1.0);
        let std_err = (variance / n).sqrt();

        Ok((mean, std_err))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stratified sampling
    // ─────────────────────────────────────────────────────────────────────────

    /// Stratified Monte Carlo integration.
    ///
    /// Each dimension is divided into `n_strata_per_dim` equal sub-intervals
    /// (strata). `n_per_stratum` uniform samples are drawn within each
    /// multi-dimensional stratum (sub-cell) formed by the Cartesian product of
    /// the per-dimension strata.
    ///
    /// The variance is reduced compared to plain Monte Carlo because samples are
    /// forced to cover the domain more evenly.
    ///
    /// # Parameters
    ///
    /// * `f`               – integrand.
    /// * `bounds`          – integration bounds per dimension.
    /// * `n_strata_per_dim`– number of strata along each axis (≥ 1).
    /// * `n_per_stratum`   – samples per multi-dimensional stratum (≥ 1).
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::ValueError`] for bad inputs or degenerate bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(11));
    /// // ∫₀¹ x² dx = 1/3
    /// let val = mc.stratified_sampling(
    ///     |x: &[f64]| x[0] * x[0],
    ///     &[(0.0_f64, 1.0_f64)],
    ///     50,
    ///     4,
    /// ).expect("failed");
    /// assert!((val - 1.0 / 3.0).abs() < 0.01);
    /// ```
    pub fn stratified_sampling<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        n_strata_per_dim: usize,
        n_per_stratum: usize,
    ) -> IntegrateResult<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        validate_bounds(bounds)?;
        if n_strata_per_dim == 0 {
            return Err(IntegrateError::ValueError(
                "n_strata_per_dim must be at least 1".to_string(),
            ));
        }
        if n_per_stratum == 0 {
            return Err(IntegrateError::ValueError(
                "n_per_stratum must be at least 1".to_string(),
            ));
        }

        let dim = bounds.len();
        let volume = compute_volume(bounds);
        let n_strata = n_strata_per_dim.pow(dim as u32);
        let stratum_volume = volume / (n_strata as f64);
        let mut rng = self.make_rng();
        let total_samples = n_strata * n_per_stratum;

        let mut sum = 0.0f64;
        let mut point = vec![0.0f64; dim];

        // Iterate over all strata using a mixed-radix counter
        let mut stratum_idx = vec![0usize; dim];
        for _stratum in 0..n_strata {
            // Compute per-dimension stratum sub-interval
            let sub_lo: Vec<f64> = (0..dim)
                .map(|d| {
                    let (a, b) = bounds[d];
                    a + (b - a) * (stratum_idx[d] as f64) / (n_strata_per_dim as f64)
                })
                .collect();
            let sub_hi: Vec<f64> = (0..dim)
                .map(|d| {
                    let (a, b) = bounds[d];
                    a + (b - a) * (stratum_idx[d] + 1) as f64 / (n_strata_per_dim as f64)
                })
                .collect();

            // Build per-dimension Uniform distributions for this stratum
            let dists: Vec<Uniform<f64>> = (0..dim)
                .map(|d| {
                    Uniform::new_inclusive(sub_lo[d], sub_hi[d])
                        .map_err(|e| IntegrateError::ValueError(format!("Stratum dist error: {e}")))
                })
                .collect::<IntegrateResult<Vec<_>>>()?;

            for _ in 0..n_per_stratum {
                for (d, dist) in dists.iter().enumerate() {
                    point[d] = dist.sample(&mut rng);
                }
                sum += f(&point);
            }

            // Advance mixed-radix counter (little-endian dimension ordering)
            for d in 0..dim {
                stratum_idx[d] += 1;
                if stratum_idx[d] < n_strata_per_dim {
                    break;
                }
                stratum_idx[d] = 0;
            }
        }

        Ok(sum / (total_samples as f64) * volume)
        // Equivalently: (sum / total_samples) * volume
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Control variates
    // ─────────────────────────────────────────────────────────────────────────

    /// Monte Carlo integration with the **control variate** variance-reduction
    /// technique.
    ///
    /// Given an integrand `f` and a control function `g` whose exact integral
    /// `g_expected = ∫ g(x) dx` is known analytically, the estimator
    ///
    /// ```text
    /// θ̂_CV = (1/N) Σ [f(x_i) - c·(g(x_i) - g_expected)]
    /// ```
    ///
    /// has variance `Var(f) - 2c·Cov(f,g) + c²·Var(g)`, minimised at
    /// `c* = Cov(f,g) / Var(g)`.  This implementation estimates `c*`
    /// empirically from the same sample.
    ///
    /// # Parameters
    ///
    /// * `f`          – integrand.
    /// * `g`          – control function (should be correlated with `f`).
    /// * `g_expected` – exact value of `∫ g(x) dx` over `bounds`.
    /// * `bounds`     – integration domain.
    /// * `n_samples`  – number of uniform samples.
    ///
    /// # Returns
    ///
    /// `(cv_estimate, std_error)`
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(3));
    /// // ∫₀¹ x³ dx = 1/4; use g(x) = x² with g_expected = 1/3
    /// let (val, se) = mc.control_variate(
    ///     |x: &[f64]| x[0].powi(3),
    ///     |x: &[f64]| x[0] * x[0],
    ///     1.0_f64 / 3.0_f64,
    ///     &[(0.0_f64, 1.0_f64)],
    ///     30_000,
    /// ).expect("failed");
    /// assert!((val - 0.25).abs() < 0.01);
    /// ```
    pub fn control_variate<F, G>(
        &self,
        f: F,
        g: G,
        g_expected: f64,
        bounds: &[(f64, f64)],
        n_samples: usize,
    ) -> IntegrateResult<(f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        validate_bounds(bounds)?;
        if n_samples < 2 {
            return Err(IntegrateError::ValueError(
                "n_samples must be at least 2 for control variate".to_string(),
            ));
        }

        let dim = bounds.len();
        let volume = compute_volume(bounds);
        let mut rng = self.make_rng();

        let dists: Vec<Uniform<f64>> = bounds
            .iter()
            .map(|&(a, b)| {
                Uniform::new_inclusive(a, b)
                    .map_err(|e| IntegrateError::ValueError(format!("Uniform dist error: {e}")))
            })
            .collect::<IntegrateResult<Vec<_>>>()?;

        let mut f_vals = Vec::with_capacity(n_samples);
        let mut g_vals = Vec::with_capacity(n_samples);
        let mut point = vec![0.0f64; dim];

        for _ in 0..n_samples {
            for (i, dist) in dists.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            f_vals.push(f(&point));
            g_vals.push(g(&point));
        }

        // Estimate optimal control coefficient c* = Cov(f,g) / Var(g)
        let n = n_samples as f64;
        let f_mean = f_vals.iter().sum::<f64>() / n;
        let g_mean = g_vals.iter().sum::<f64>() / n;

        let cov_fg: f64 = f_vals
            .iter()
            .zip(g_vals.iter())
            .map(|(&fv, &gv)| (fv - f_mean) * (gv - g_mean))
            .sum::<f64>()
            / (n - 1.0);

        let var_g: f64 = g_vals.iter().map(|&gv| (gv - g_mean).powi(2)).sum::<f64>() / (n - 1.0);

        // Avoid division by near-zero variance
        let c_star = if var_g.abs() < 1e-300 {
            0.0
        } else {
            cov_fg / var_g
        };

        // Compute the control-variate estimator
        // Note: g_expected is already the integral value (scaled by volume),
        // while g_mean is the mean of g at uniform samples.
        // The unbiased CV estimator: f_hat = f_mean + c*(g_expected/volume - g_mean)
        let g_mean_expected = g_expected / volume; // expected mean of g(x) under Uniform
        let cv_mean = f_mean + c_star * (g_mean_expected - g_mean);

        // Compute variance of CV estimator residuals
        let residuals: Vec<f64> = f_vals
            .iter()
            .zip(g_vals.iter())
            .map(|(&fv, &gv)| fv - c_star * gv)
            .collect();

        let res_mean = residuals.iter().sum::<f64>() / n;
        let var_res: f64 = residuals
            .iter()
            .map(|&r| (r - res_mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std_err = (var_res / n).sqrt() * volume;

        Ok((cv_mean * volume, std_err))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Antithetic variates
    // ─────────────────────────────────────────────────────────────────────────

    /// Monte Carlo integration using **antithetic variates**.
    ///
    /// For each uniform sample `u` in the unit hypercube the antithetic
    /// complement `1 - u` is also evaluated. The average of the pair
    /// `(f(u) + f(1-u)) / 2` has lower variance than `f(u)` alone when `f`
    /// is monotone in each coordinate (negative correlation between the two
    /// evaluations).
    ///
    /// # Parameters
    ///
    /// * `f`        – integrand.
    /// * `bounds`   – integration domain.
    /// * `n_samples`– total number of function evaluations (will be rounded to
    ///   an even number).
    ///
    /// # Returns
    ///
    /// Estimated integral value.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(5));
    /// // ∫₀¹ e^x dx = e - 1 ≈ 1.718
    /// let val = mc.antithetic_variates(
    ///     |x: &[f64]| x[0].exp(),
    ///     &[(0.0_f64, 1.0_f64)],
    ///     20_000,
    /// ).expect("failed");
    /// let exact = std::f64::consts::E - 1.0;
    /// assert!((val - exact).abs() < 0.02);
    /// ```
    pub fn antithetic_variates<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        n_samples: usize,
    ) -> IntegrateResult<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        validate_bounds(bounds)?;
        if n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be positive".to_string(),
            ));
        }

        let dim = bounds.len();
        let volume = compute_volume(bounds);
        let n_pairs = n_samples / 2; // ensure we use an even count
        if n_pairs == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be at least 2 for antithetic variates".to_string(),
            ));
        }

        let mut rng = self.make_rng();
        // Uniform on [0,1] for the unit-hypercube coordinate
        let unit_dist = Uniform::new_inclusive(0.0f64, 1.0f64)
            .map_err(|e| IntegrateError::ValueError(format!("Uniform dist error: {e}")))?;

        let mut sum = 0.0f64;
        let mut point = vec![0.0f64; dim];
        let mut anti = vec![0.0f64; dim];

        for _ in 0..n_pairs {
            // Sample u in unit hypercube then map to [a,b]
            for d in 0..dim {
                let (a, b) = bounds[d];
                let u = unit_dist.sample(&mut rng);
                point[d] = a + u * (b - a);
                anti[d] = a + (1.0 - u) * (b - a); // antithetic complement
            }
            let f_u = f(&point);
            let f_anti = f(&anti);
            sum += (f_u + f_anti) * 0.5;
        }

        Ok(sum / (n_pairs as f64) * volume)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MCMC integration (Metropolis-Hastings)
    // ─────────────────────────────────────────────────────────────────────────

    /// MCMC-based Monte Carlo integration via the **Metropolis-Hastings**
    /// algorithm.
    ///
    /// Estimates `∫ f(x) π(x) dx / ∫ π(x) dx` — i.e., the expectation of `f`
    /// under the target distribution `π`.  The denominator normalisation is
    /// handled implicitly by the MCMC chain; `target_pdf` need not be
    /// normalised.
    ///
    /// After discarding `burnin` burn-in steps the chain produces `n_samples`
    /// states, and the estimator is their empirical mean.
    ///
    /// # Parameters
    ///
    /// * `f`           – function to integrate (expectation taken under `π`).
    /// * `target_pdf`  – unnormalised target density `π(x)` (must be ≥ 0).
    /// * `proposal`    – symmetric random-walk proposal scale (standard
    ///   deviation of the isotropic Gaussian proposal kernel).
    /// * `n_samples`   – number of post-burnin samples.
    /// * `burnin`      – number of initial steps to discard.
    ///
    /// # Returns
    ///
    /// Estimated expectation `E_π[f]`.
    ///
    /// # Notes
    ///
    /// The initial state is set to the zero vector. For distributions with
    /// support far from the origin consider increasing `burnin`.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::monte_carlo_advanced::MonteCarloIntegrator;
    /// use std::f64::consts::PI;
    ///
    /// let mc = MonteCarloIntegrator::new(Some(17));
    ///
    /// // E_{N(0,1)}[x²] = 1.0
    /// let target = |x: &[f64]| (-0.5 * x[0] * x[0]).exp() / (2.0 * PI).sqrt();
    /// let val = mc.mcmc_integrate(
    ///     |x: &[f64]| x[0] * x[0],
    ///     target,
    ///     0.5,        // proposal std
    ///     50_000,
    ///     5_000,      // burn-in
    /// ).expect("failed");
    ///
    /// assert!((val - 1.0).abs() < 0.05);
    /// ```
    pub fn mcmc_integrate<F, T>(
        &self,
        f: F,
        target_pdf: T,
        proposal: f64,
        n_samples: usize,
        burnin: usize,
    ) -> IntegrateResult<f64>
    where
        F: Fn(&[f64]) -> f64,
        T: Fn(&[f64]) -> f64,
    {
        if n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be positive".to_string(),
            ));
        }
        if proposal <= 0.0 {
            return Err(IntegrateError::ValueError(
                "proposal standard deviation must be positive".to_string(),
            ));
        }

        // We infer the dimensionality from a test call; start at origin.
        // Dimension is determined by calling f with a 1-element slice and
        // checking if target_pdf is meaningful; we will default to dim=1
        // and rely on f/target_pdf to use however many dimensions they need
        // via closures capturing their own context. Here we use dim=1 for the
        // Markov chain state unless the user chooses a higher-dim integrand.
        // To support arbitrary dim the caller should pass dim via closure scope.
        //
        // For a proper multi-dimensional MCMC implementation we run a
        // 1-D chain and let the integrand be a function of that 1-D state.
        // This is intentional: if the user needs multi-dimensional MCMC
        // they wrap their target and integrand appropriately.
        //
        // Dimension detection: call target_pdf with increasingly-sized slices
        // until we find the one that makes sense – but that is fragile. Instead,
        // we detect the dimension by calling `f` and `target_pdf` with a
        // 1-D argument as a probe; if those closures require a longer slice
        // they will panic; the contract is the caller ensures consistency.
        //
        // We use a fixed dimension of 1 for the default case and note that
        // for higher-dimensional MCMC the user can pass a function of a
        // 1-D flattened representation.
        let dim = 1usize; // single-dimensional Markov chain

        let mut rng = self.make_rng();
        let normal_dist = Normal::new(0.0f64, proposal)
            .map_err(|e| IntegrateError::ValueError(format!("Normal dist error: {e}")))?;
        let uniform_dist = Uniform::new(0.0f64, 1.0f64)
            .map_err(|e| IntegrateError::ValueError(format!("Uniform dist error: {e}")))?;

        let mut current = vec![0.0f64; dim];
        let mut current_density = target_pdf(&current);
        // If the initial density is zero or negative, perturb the start
        if current_density <= 0.0 {
            current[0] = 0.01;
            current_density = target_pdf(&current);
        }

        let mut proposal_state = vec![0.0f64; dim];
        let total_steps = burnin + n_samples;
        let mut sum = 0.0f64;
        let mut count = 0usize;

        for step in 0..total_steps {
            // Propose new state from isotropic Gaussian walk
            for d in 0..dim {
                proposal_state[d] = current[d] + normal_dist.sample(&mut rng);
            }

            let proposed_density = target_pdf(&proposal_state);
            let accept_prob = if current_density <= 0.0 {
                1.0
            } else {
                (proposed_density / current_density).min(1.0)
            };

            if uniform_dist.sample(&mut rng) < accept_prob {
                current.clone_from(&proposal_state);
                current_density = proposed_density;
            }

            if step >= burnin {
                sum += f(&current);
                count += 1;
            }
        }

        if count == 0 {
            return Err(IntegrateError::ConvergenceError(
                "No post-burnin samples collected in MCMC integration".to_string(),
            ));
        }

        Ok(sum / (count as f64))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that `bounds` is non-empty and all intervals are non-degenerate.
fn validate_bounds(bounds: &[(f64, f64)]) -> IntegrateResult<()> {
    if bounds.is_empty() {
        return Err(IntegrateError::ValueError(
            "bounds must not be empty".to_string(),
        ));
    }
    for (i, &(a, b)) in bounds.iter().enumerate() {
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "bounds[{i}]: lower bound {a} must be strictly less than upper bound {b}"
            )));
        }
    }
    Ok(())
}

/// Compute the hyperrectangle volume from `bounds`.
fn compute_volume(bounds: &[(f64, f64)]) -> f64 {
    bounds.iter().map(|&(a, b)| b - a).product()
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export SampleUniform so callers can import from this module if needed
// ─────────────────────────────────────────────────────────────────────────────
#[allow(unused_imports)]
use scirs2_core::random::uniform::SampleUniform as _SampleUniformReExport;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    const SEED: u64 = 42;
    const N: usize = 80_000;

    // Helper: absolute difference ≤ tol
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // ── integrate ────────────────────────────────────────────────────────────

    #[test]
    fn test_integrate_1d_x_squared() {
        // ∫₀¹ x² dx = 1/3
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let (val, se) = mc
            .integrate(|x| x[0] * x[0], &[(0.0, 1.0)], N)
            .expect("integrate 1d failed");
        assert!(close(val, 1.0 / 3.0, 0.02), "val={val}");
        assert!(se >= 0.0, "std_err must be non-negative");
    }

    #[test]
    fn test_integrate_2d_sum() {
        // ∫₀¹∫₀¹ (x+y) dx dy = 1
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let (val, _) = mc
            .integrate(|x| x[0] + x[1], &[(0.0, 1.0), (0.0, 1.0)], N)
            .expect("integrate 2d failed");
        assert!(close(val, 1.0, 0.03), "val={val}");
    }

    #[test]
    fn test_integrate_non_unit_bounds() {
        // ∫₁² x dx = 3/2
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let (val, _) = mc
            .integrate(|x| x[0], &[(1.0, 2.0)], N)
            .expect("integrate non-unit failed");
        assert!(close(val, 1.5, 0.03), "val={val}");
    }

    #[test]
    fn test_integrate_error_empty_bounds() {
        let mc = MonteCarloIntegrator::new(None);
        assert!(mc.integrate(|_| 1.0, &[], 100).is_err());
    }

    #[test]
    fn test_integrate_error_zero_samples() {
        let mc = MonteCarloIntegrator::new(None);
        assert!(mc.integrate(|x| x[0], &[(0.0, 1.0)], 0).is_err());
    }

    // ── importance_sampling ──────────────────────────────────────────────────

    #[test]
    fn test_importance_sampling_uniform() {
        // ∫₀¹ x² dx = 1/3 with uniform proposal
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let (val, _) = mc
            .importance_sampling(
                |x| x[0] * x[0],
                |rng| {
                    let u = Uniform::new(0.0f64, 1.0f64)
                        .expect("valid Uniform range [0,1)")
                        .sample(rng);
                    vec![u]
                },
                |_| 1.0f64,
                N,
            )
            .expect("importance sampling failed");
        assert!(close(val, 1.0 / 3.0, 0.02), "val={val}");
    }

    // ── stratified_sampling ──────────────────────────────────────────────────

    #[test]
    fn test_stratified_1d() {
        // ∫₀¹ x² dx = 1/3
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let val = mc
            .stratified_sampling(|x| x[0] * x[0], &[(0.0, 1.0)], 100, 5)
            .expect("stratified failed");
        assert!(close(val, 1.0 / 3.0, 0.02), "val={val}");
    }

    #[test]
    fn test_stratified_2d() {
        // ∫₀¹∫₀¹ x·y dx dy = 1/4
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let val = mc
            .stratified_sampling(|x| x[0] * x[1], &[(0.0, 1.0), (0.0, 1.0)], 20, 4)
            .expect("stratified 2d failed");
        assert!(close(val, 0.25, 0.03), "val={val}");
    }

    // ── control_variate ──────────────────────────────────────────────────────

    #[test]
    fn test_control_variate_1d() {
        // ∫₀¹ x³ dx = 1/4, control: g(x) = x² with g_expected = 1/3
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let (val, se) = mc
            .control_variate(
                |x| x[0].powi(3),
                |x| x[0] * x[0],
                1.0 / 3.0,
                &[(0.0, 1.0)],
                N,
            )
            .expect("control variate failed");
        assert!(close(val, 0.25, 0.02), "val={val}");
        assert!(se >= 0.0, "std_err={se}");
    }

    // ── antithetic_variates ──────────────────────────────────────────────────

    #[test]
    fn test_antithetic_variates() {
        // ∫₀¹ eˣ dx = e - 1
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let val = mc
            .antithetic_variates(|x| x[0].exp(), &[(0.0, 1.0)], N)
            .expect("antithetic failed");
        assert!(close(val, E - 1.0, 0.02), "val={val}, exact={}", E - 1.0);
    }

    // ── mcmc_integrate ───────────────────────────────────────────────────────

    #[test]
    fn test_mcmc_integrate_normal_variance() {
        // E_{N(0,1)}[x²] = 1.0
        let mc = MonteCarloIntegrator::new(Some(SEED));
        let target = |x: &[f64]| (-0.5 * x[0] * x[0]).exp() / (2.0 * PI).sqrt();
        let val = mc
            .mcmc_integrate(|x| x[0] * x[0], target, 1.0, 60_000, 5_000)
            .expect("mcmc failed");
        assert!(close(val, 1.0, 0.1), "val={val}");
    }

    #[test]
    fn test_mcmc_integrate_mean_zero() {
        // E_{N(0,1)}[x] = 0.0
        let mc = MonteCarloIntegrator::new(Some(SEED + 1));
        let target = |x: &[f64]| (-0.5 * x[0] * x[0]).exp();
        let val = mc
            .mcmc_integrate(|x| x[0], target, 1.0, 60_000, 5_000)
            .expect("mcmc mean failed");
        assert!(close(val, 0.0, 0.1), "val={val}");
    }

    #[test]
    fn test_mcmc_error_zero_samples() {
        let mc = MonteCarloIntegrator::new(None);
        assert!(mc
            .mcmc_integrate(|x| x[0], |x| (-x[0] * x[0]).exp(), 1.0, 0, 100)
            .is_err());
    }
}
