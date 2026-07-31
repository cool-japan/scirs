//! Particle Filter (Sequential Monte Carlo) for nonlinear / non-Gaussian systems.
//!
//! The particle filter approximates the posterior distribution `p(x_k | z_{1:k})`
//! by a weighted set of `N` random "particles" (state hypotheses).  Each step:
//!
//! 1. **Predict** – propagate particles through the (possibly nonlinear, non-Gaussian)
//!    transition model and add process noise.
//! 2. **Update** – weight each particle by its likelihood under the current measurement.
//! 3. **Resample** – draw a new particle set from the categorical distribution defined
//!    by the normalized weights (systematic resampling used here for low variance).
//!
//! Two built-in likelihood models are provided:
//! - [`GaussianLikelihood`] – standard multivariate Gaussian observation noise.
//! - [`StudentTLikelihood`] – heavier-tailed Student-t for robustness to outliers.
//!
//! Users may supply any custom likelihood via the [`Likelihood`] trait.
//!
//! # References
//!
//! * Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993).
//!   "Novel approach to nonlinear/non-Gaussian Bayesian state estimation".
//!   *IEE Proceedings F*, 140(2), 107–113.
//! * Arulampalam, M.S. et al. (2002).
//!   "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking".
//!   *IEEE Transactions on Signal Processing*, 50(2), 174–188.
//! * Douc, R. & Cappe, O. (2005).
//!   "Comparison of resampling schemes for particle filtering".
//!   *ISPA 2005*, 64–69.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Likelihood trait
// ---------------------------------------------------------------------------

/// Trait for computing the observation likelihood `p(z | x)`.
///
/// Implement this trait to plug a custom noise model into [`ParticleFilter`].
pub trait Likelihood: Send + Sync {
    /// Return the (unnormalised) likelihood of observation `z` given state `x`.
    ///
    /// Values must be non-negative; only relative magnitudes matter because the
    /// weights are normalised after all particles are evaluated.
    fn likelihood(&self, z: &[f64], x: &[f64]) -> f64;
}

// ---------------------------------------------------------------------------
// Built-in likelihood models
// ---------------------------------------------------------------------------

/// Multivariate Gaussian (diagonal covariance) observation likelihood.
///
/// Computes `exp(-0.5 · ∑_i (z_i - h(x)_i)² / σ_i²)` where `h(x)` is the
/// linear observation matrix `H · x`.
#[derive(Debug, Clone)]
pub struct GaussianLikelihood {
    /// Diagonal measurement noise standard deviations (length = dim_z).
    pub std_devs: Vec<f64>,
    /// Observation matrix H (dim_z × dim_x), row-major.
    pub h: Vec<Vec<f64>>,
}

impl GaussianLikelihood {
    /// Create a new Gaussian likelihood model.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] when `std_devs` is empty or any
    /// standard deviation is non-positive, or when `h` has wrong dimensions.
    pub fn new(h: Vec<Vec<f64>>, std_devs: Vec<f64>) -> SignalResult<Self> {
        if std_devs.is_empty() {
            return Err(SignalError::ValueError(
                "std_devs must not be empty".to_string(),
            ));
        }
        for &s in &std_devs {
            if s <= 0.0 {
                return Err(SignalError::ValueError(format!(
                    "all std_devs must be positive, got {s}"
                )));
            }
        }
        if h.len() != std_devs.len() {
            return Err(SignalError::ValueError(format!(
                "H has {} rows but std_devs has {} elements",
                h.len(),
                std_devs.len()
            )));
        }
        Ok(Self { std_devs, h })
    }

    /// Apply observation matrix H to state x.
    fn observe(&self, x: &[f64]) -> Vec<f64> {
        self.h
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(h, xi)| h * xi).sum::<f64>())
            .collect()
    }
}

impl Likelihood for GaussianLikelihood {
    fn likelihood(&self, z: &[f64], x: &[f64]) -> f64 {
        let hx = self.observe(x);
        let log_l: f64 = z
            .iter()
            .zip(hx.iter())
            .zip(self.std_devs.iter())
            .map(|((zi, hxi), si)| {
                let err = zi - hxi;
                -0.5 * (err / si).powi(2)
            })
            .sum();
        log_l.exp()
    }
}

/// Student-t observation likelihood (robust to outliers).
///
/// Uses a diagonal Student-t model with degrees of freedom `nu` and scale `sigma`.
/// Heavier tails than Gaussian reduce the influence of outlier measurements.
#[derive(Debug, Clone)]
pub struct StudentTLikelihood {
    /// Degrees of freedom (> 0; values ≤ 2 give infinite variance).
    pub nu: f64,
    /// Scale parameters (one per measurement dimension).
    pub scales: Vec<f64>,
    /// Observation matrix H.
    pub h: Vec<Vec<f64>>,
}

impl StudentTLikelihood {
    /// Create a new Student-t likelihood model.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `nu` ≤ 0 or any scale ≤ 0.
    pub fn new(h: Vec<Vec<f64>>, scales: Vec<f64>, nu: f64) -> SignalResult<Self> {
        if nu <= 0.0 {
            return Err(SignalError::ValueError(format!("nu must be > 0, got {nu}")));
        }
        for &s in &scales {
            if s <= 0.0 {
                return Err(SignalError::ValueError(format!(
                    "all scales must be positive, got {s}"
                )));
            }
        }
        if h.len() != scales.len() {
            return Err(SignalError::ValueError(format!(
                "H has {} rows but scales has {} elements",
                h.len(),
                scales.len()
            )));
        }
        Ok(Self { nu, scales, h })
    }

    fn observe(&self, x: &[f64]) -> Vec<f64> {
        self.h
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(h, xi)| h * xi).sum::<f64>())
            .collect()
    }

    /// Univariate Student-t log-density (unnormalised constant dropped).
    fn log_student_t_density(err: f64, scale: f64, nu: f64) -> f64 {
        let t = err / scale;
        -(nu + 1.0) / 2.0 * (1.0 + t * t / nu).ln()
    }
}

impl Likelihood for StudentTLikelihood {
    fn likelihood(&self, z: &[f64], x: &[f64]) -> f64 {
        let hx = self.observe(x);
        let log_l: f64 = z
            .iter()
            .zip(hx.iter())
            .zip(self.scales.iter())
            .map(|((zi, hxi), si)| Self::log_student_t_density(zi - hxi, *si, self.nu))
            .sum();
        log_l.exp()
    }
}

// ---------------------------------------------------------------------------
// Resampling strategies
// ---------------------------------------------------------------------------

/// Resampling strategy for the particle filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingStrategy {
    /// Multinomial resampling (simplest but highest variance).
    Multinomial,
    /// Systematic resampling (lower variance, O(N) cost).  Recommended default.
    Systematic,
    /// Stratified resampling (similar to systematic, slightly lower variance).
    Stratified,
}

// ---------------------------------------------------------------------------
// Effective sample size
// ---------------------------------------------------------------------------

/// Compute the Effective Number of Particles (ENP / Neff).
///
/// `Neff = 1 / ∑_i w_i²`  where `w_i` are normalised weights.
/// A value close to `N` means all particles contribute equally;
/// a value close to 1 means the distribution has collapsed to a single particle.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
    if sum_sq <= 0.0 {
        0.0
    } else {
        1.0 / sum_sq
    }
}

// ---------------------------------------------------------------------------
// ParticleFilter struct
// ---------------------------------------------------------------------------

/// Bootstrap Particle Filter (Sequential Importance Resampling).
///
/// The state transition and observation models are provided as boxed closures,
/// making the filter compatible with any nonlinear, non-Gaussian system.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::particle::{ParticleFilter, GaussianLikelihood, ResamplingStrategy};
///
/// // 1-D constant-velocity model, position + velocity state
/// let f = |x: &[f64], rng: &mut dyn FnMut() -> f64| -> Vec<f64> {
///     let dt = 1.0_f64;
///     vec![
///         x[0] + x[1] * dt + 0.1 * rng(),
///         x[1] + 0.05 * rng(),
///     ]
/// };
///
/// let h = vec![vec![1.0_f64, 0.0]]; // observe position only
/// let likelihood = GaussianLikelihood::new(h, vec![0.5]).expect("operation should succeed");
///
/// let mut pf = ParticleFilter::new(200, 2, Box::new(likelihood), ResamplingStrategy::Systematic);
/// pf.initialize_uniform(&[-5.0, -1.0], &[5.0, 1.0]).expect("operation should succeed");
/// pf.set_transition(Box::new(f));
///
/// // Feed a measurement of position = 1.0
/// pf.step(&[1.0]).expect("operation should succeed");
/// let mean = pf.mean_state();
/// assert_eq!(mean.len(), 2);
/// ```
pub struct ParticleFilter {
    /// Number of particles.
    n_particles: usize,
    /// State dimension.
    dim_x: usize,
    /// Particle states: shape [n_particles × dim_x].
    particles: Vec<Vec<f64>>,
    /// Particle weights (log domain internally, normalised on access).
    log_weights: Vec<f64>,
    /// Resampling strategy.
    resampling: ResamplingStrategy,
    /// Effective sample size threshold as fraction of N (resampling triggered when Neff < threshold × N).
    resample_threshold: f64,
    /// Observation likelihood model.
    likelihood: Box<dyn Likelihood>,
    /// State transition function: `f(x, noise_fn)`.
    /// The second argument is a `dyn FnMut() -> f64` that draws N(0,1) samples.
    transition: Option<Box<dyn Fn(&[f64], &mut dyn FnMut() -> f64) -> Vec<f64>>>,
    /// Internal LCG state for reproducible noise.
    rng_state: u64,
}

impl ParticleFilter {
    /// Create a new particle filter.
    ///
    /// # Arguments
    ///
    /// * `n_particles`   - Number of particles `N`.
    /// * `dim_x`         - State space dimension.
    /// * `likelihood`    - Boxed [`Likelihood`] model.
    /// * `resampling`    - [`ResamplingStrategy`] to use.
    pub fn new(
        n_particles: usize,
        dim_x: usize,
        likelihood: Box<dyn Likelihood>,
        resampling: ResamplingStrategy,
    ) -> Self {
        let uniform_log_weight = -(n_particles as f64).ln();
        ParticleFilter {
            n_particles,
            dim_x,
            particles: vec![vec![0.0; dim_x]; n_particles],
            log_weights: vec![uniform_log_weight; n_particles],
            resampling,
            resample_threshold: 0.5, // resample when Neff < 0.5 N
            likelihood,
            transition: None,
            rng_state: 0x1234_5678_abcd_ef01,
        }
    }

    /// Set a custom resampling threshold (fraction of N).
    ///
    /// Resampling occurs when `Neff < threshold × N`.
    /// Set to 0.0 to disable adaptive resampling (always resample after update).
    /// Set to 1.0 to always resample.
    pub fn set_resample_threshold(&mut self, threshold: f64) {
        self.resample_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Seed the internal LCG random number generator.
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Set the state transition function.
    ///
    /// The closure takes the current particle state `x: &[f64]` and a normal
    /// random sample generator `rng: &mut dyn FnMut() -> f64`, and returns the
    /// propagated state.
    pub fn set_transition(&mut self, f: Box<dyn Fn(&[f64], &mut dyn FnMut() -> f64) -> Vec<f64>>) {
        self.transition = Some(f);
    }

    /// Initialise particles from a uniform distribution over the box `[lo, hi]`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if slice lengths do not match `dim_x`.
    pub fn initialize_uniform(&mut self, lo: &[f64], hi: &[f64]) -> SignalResult<()> {
        if lo.len() != self.dim_x || hi.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "lo and hi must have length {}, got {} and {}",
                self.dim_x,
                lo.len(),
                hi.len()
            )));
        }
        for i in 0..self.dim_x {
            if lo[i] > hi[i] {
                return Err(SignalError::ValueError(format!(
                    "lo[{i}] = {} > hi[{i}] = {}",
                    lo[i], hi[i]
                )));
            }
        }

        // Pre-generate random values to avoid double mutable borrow on self
        let random_values: Vec<Vec<f64>> = (0..self.n_particles)
            .map(|_| (0..self.dim_x).map(|_| self.lcg_uniform()).collect())
            .collect();
        for (p, rand_vals) in self.particles.iter_mut().zip(random_values.iter()) {
            for (d, state) in p.iter_mut().enumerate() {
                *state = lo[d] + (hi[d] - lo[d]) * rand_vals[d];
            }
        }
        let uniform_log = -(self.n_particles as f64).ln();
        self.log_weights.fill(uniform_log);
        Ok(())
    }

    /// Initialise particles from a Gaussian prior.
    ///
    /// # Arguments
    ///
    /// * `mean` - Prior mean (length `dim_x`).
    /// * `std`  - Prior standard deviation per dimension (length `dim_x`).
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] on length mismatch or non-positive std.
    pub fn initialize_gaussian(&mut self, mean: &[f64], std: &[f64]) -> SignalResult<()> {
        if mean.len() != self.dim_x || std.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "mean and std must have length {}, got {} and {}",
                self.dim_x,
                mean.len(),
                std.len()
            )));
        }
        for &s in std {
            if s <= 0.0 {
                return Err(SignalError::ValueError(format!(
                    "all std deviations must be positive, got {s}"
                )));
            }
        }

        // Pre-generate random values to avoid double mutable borrow on self
        let random_values: Vec<Vec<f64>> = (0..self.n_particles)
            .map(|_| (0..self.dim_x).map(|_| self.lcg_normal()).collect())
            .collect();
        for (p, rand_vals) in self.particles.iter_mut().zip(random_values.iter()) {
            for (d, state) in p.iter_mut().enumerate() {
                *state = mean[d] + std[d] * rand_vals[d];
            }
        }
        let uniform_log = -(self.n_particles as f64).ln();
        self.log_weights.fill(uniform_log);
        Ok(())
    }

    /// Perform one predict–update–resample cycle.
    ///
    /// # Arguments
    ///
    /// * `measurement` - Observed measurement vector `z_k`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ComputationError`] when no transition function has been set
    /// via [`Self::set_transition`], or [`SignalError::ValueError`] if weights collapse to zero.
    pub fn step(&mut self, measurement: &[f64]) -> SignalResult<()> {
        self.predict()?;
        self.update(measurement)?;
        // Adaptive resampling: only resample when Neff drops below threshold
        let weights = self.weights();
        let neff = effective_sample_size(&weights);
        if neff < self.resample_threshold * self.n_particles as f64 {
            self.resample()?;
        }
        Ok(())
    }

    /// Predict step: propagate each particle through the transition model.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ComputationError`] if no transition function has been set.
    pub fn predict(&mut self) -> SignalResult<()> {
        let f = self.transition.as_ref().ok_or_else(|| {
            SignalError::ComputationError(
                "transition function not set; call set_transition() first".to_string(),
            )
        })?;

        // We need mutable access to rng_state while calling f.
        // Work around the borrow checker by temporarily taking ownership.
        let mut rng_state = self.rng_state;
        for p in self.particles.iter_mut() {
            let mut noise_fn = {
                let state_ref = &mut rng_state;
                move || lcg_normal_with_state(state_ref)
            };
            let new_p = f(p, &mut noise_fn);
            *p = new_p;
        }
        self.rng_state = rng_state;
        Ok(())
    }

    /// Update step: reweight particles by the observation likelihood.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if all weights are zero (weight collapse).
    pub fn update(&mut self, measurement: &[f64]) -> SignalResult<()> {
        // Compute log-likelihoods and add to current log weights
        for (lw, p) in self.log_weights.iter_mut().zip(self.particles.iter()) {
            let l = self.likelihood.likelihood(measurement, p);
            // Guard against -inf from zero likelihood (numerically unstable particle)
            let log_l = if l > 0.0 { l.ln() } else { f64::NEG_INFINITY };
            *lw += log_l;
        }

        // Numerically stable log-sum-exp normalisation
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .filter(|lw| lw.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_lw.is_infinite() {
            return Err(SignalError::ValueError(
                "all particle weights collapsed to zero; try increasing particle count \
                 or widening the prior"
                    .to_string(),
            ));
        }

        let sum_exp: f64 = self.log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();

        let log_sum = max_lw + sum_exp.ln();
        for lw in self.log_weights.iter_mut() {
            *lw -= log_sum;
        }
        Ok(())
    }

    /// Resample particles using the configured resampling strategy.
    ///
    /// After resampling, all weights are reset to uniform `1/N`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] on numerical failure.
    pub fn resample(&mut self) -> SignalResult<()> {
        let weights = self.weights();
        let indices = match self.resampling {
            ResamplingStrategy::Multinomial => self.multinomial_resample(&weights),
            ResamplingStrategy::Systematic => self.systematic_resample(&weights),
            ResamplingStrategy::Stratified => self.stratified_resample(&weights),
        };
        let new_particles: Vec<Vec<f64>> =
            indices.iter().map(|&i| self.particles[i].clone()).collect();
        self.particles = new_particles;
        let uniform_log = -(self.n_particles as f64).ln();
        self.log_weights.fill(uniform_log);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Resampling implementations
    // -----------------------------------------------------------------------

    fn multinomial_resample(&mut self, weights: &[f64]) -> Vec<usize> {
        // Build CDF
        let cdf = cumsum(weights);
        let mut indices = Vec::with_capacity(self.n_particles);
        for _ in 0..self.n_particles {
            let u = self.lcg_uniform();
            let idx = cdf.partition_point(|&c| c < u).min(self.n_particles - 1);
            indices.push(idx);
        }
        indices
    }

    fn systematic_resample(&mut self, weights: &[f64]) -> Vec<usize> {
        let cdf = cumsum(weights);
        let u0 = self.lcg_uniform() / self.n_particles as f64;
        let mut indices = Vec::with_capacity(self.n_particles);
        let mut j = 0_usize;
        for i in 0..self.n_particles {
            let target = u0 + i as f64 / self.n_particles as f64;
            while j < self.n_particles - 1 && cdf[j] < target {
                j += 1;
            }
            indices.push(j);
        }
        indices
    }

    fn stratified_resample(&mut self, weights: &[f64]) -> Vec<usize> {
        let cdf = cumsum(weights);
        let mut indices = Vec::with_capacity(self.n_particles);
        let mut j = 0_usize;
        let n = self.n_particles as f64;
        for i in 0..self.n_particles {
            let u = (i as f64 + self.lcg_uniform()) / n;
            while j < self.n_particles - 1 && cdf[j] < u {
                j += 1;
            }
            indices.push(j);
        }
        indices
    }

    // -----------------------------------------------------------------------
    // State estimates
    // -----------------------------------------------------------------------

    /// Return normalised particle weights `w_i = exp(log_w_i)`.
    pub fn weights(&self) -> Vec<f64> {
        self.log_weights.iter().map(|&lw| lw.exp()).collect()
    }

    /// Weighted mean state estimate.
    pub fn mean_state(&self) -> Vec<f64> {
        let weights = self.weights();
        let mut mean = vec![0.0_f64; self.dim_x];
        for (p, &w) in self.particles.iter().zip(weights.iter()) {
            for (m, &pi) in mean.iter_mut().zip(p.iter()) {
                *m += w * pi;
            }
        }
        mean
    }

    /// Weighted variance (diagonal covariance) of the state estimate.
    pub fn variance_state(&self) -> Vec<f64> {
        let weights = self.weights();
        let mean = self.mean_state();
        let mut var = vec![0.0_f64; self.dim_x];
        for (p, &w) in self.particles.iter().zip(weights.iter()) {
            for (v, (&pi, &mi)) in var.iter_mut().zip(p.iter().zip(mean.iter())) {
                *v += w * (pi - mi).powi(2);
            }
        }
        var
    }

    /// Return the MAP particle (highest-weight particle).
    pub fn map_state(&self) -> &[f64] {
        let max_idx = self
            .log_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.particles[max_idx]
    }

    /// Number of particles.
    pub fn n_particles(&self) -> usize {
        self.n_particles
    }

    /// State dimension.
    pub fn dim_x(&self) -> usize {
        self.dim_x
    }

    /// Effective number of particles (Neff).
    pub fn neff(&self) -> f64 {
        effective_sample_size(&self.weights())
    }

    // -----------------------------------------------------------------------
    // RNG helpers (LCG + Box-Muller, consistent with ensemble.rs)
    // -----------------------------------------------------------------------

    /// Draw a U(0,1) sample.
    fn lcg_uniform(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draw a N(0,1) sample using Box-Muller.
    fn lcg_normal(&mut self) -> f64 {
        let u1 = self.lcg_uniform().max(1e-300);
        let u2 = self.lcg_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Standalone LCG normal sample for use in predict() without self-borrow conflict.
fn lcg_normal_with_state(rng_state: &mut u64) -> f64 {
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u1 = ((*rng_state >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u2 = (*rng_state >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Compute cumulative sum (prefix sum) of a weight vector.
fn cumsum(weights: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0_f64;
    for &w in weights {
        acc += w;
        cdf.push(acc);
    }
    cdf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: constant-velocity 1-D process model with Gaussian noise.
    fn cv_model(x: &[f64], rng: &mut dyn FnMut() -> f64) -> Vec<f64> {
        let dt = 1.0_f64;
        let proc_noise = 0.1;
        vec![x[0] + x[1] * dt + proc_noise * rng(), x[1] + 0.05 * rng()]
    }

    fn make_gaussian_likelihood() -> Box<dyn Likelihood> {
        let h = vec![vec![1.0_f64, 0.0]];
        Box::new(GaussianLikelihood::new(h, vec![0.5]).expect("likelihood"))
    }

    #[test]
    fn test_initialize_uniform_dimension() {
        let mut pf = ParticleFilter::new(
            100,
            2,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        pf.initialize_uniform(&[-5.0, -1.0], &[5.0, 1.0])
            .expect("should succeed in test");
        assert_eq!(pf.particles.len(), 100);
        assert!(pf.particles.iter().all(|p| p.len() == 2));
    }

    #[test]
    fn test_initialize_uniform_bounds() {
        let mut pf = ParticleFilter::new(
            200,
            2,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        pf.seed(42);
        pf.initialize_uniform(&[0.0, 0.0], &[1.0, 1.0])
            .expect("should succeed in test");
        for p in &pf.particles {
            assert!(p[0] >= 0.0 && p[0] <= 1.0, "p[0]={}", p[0]);
            assert!(p[1] >= 0.0 && p[1] <= 1.0, "p[1]={}", p[1]);
        }
    }

    #[test]
    fn test_initialize_gaussian() {
        let mut pf = ParticleFilter::new(
            500,
            2,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        pf.seed(0xdeadbeef);
        pf.initialize_gaussian(&[0.0, 1.0], &[0.5, 0.2])
            .expect("should succeed in test");
        assert_eq!(pf.particles.len(), 500);
    }

    #[test]
    fn test_step_updates_mean_toward_measurement() {
        let mut pf = ParticleFilter::new(
            500,
            2,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        pf.seed(0x1234);
        pf.initialize_gaussian(&[0.0, 0.0], &[3.0, 0.5])
            .expect("should succeed in test");
        pf.set_transition(Box::new(cv_model));

        // Feed measurement z = 5.0 repeatedly to drive the estimate up
        for _ in 0..10 {
            pf.step(&[5.0]).expect("step failed");
        }
        let mean = pf.mean_state();
        // Position estimate should shift significantly toward 5.0
        assert!(
            mean[0] > 1.0,
            "mean position {} should be > 1.0 after 10 updates toward z=5",
            mean[0]
        );
    }

    #[test]
    fn test_weight_collapse_error() {
        // All likelihoods = 0 => weights collapse
        struct ZeroLikelihood;
        impl Likelihood for ZeroLikelihood {
            fn likelihood(&self, _z: &[f64], _x: &[f64]) -> f64 {
                0.0
            }
        }
        let mut pf = ParticleFilter::new(
            50,
            1,
            Box::new(ZeroLikelihood),
            ResamplingStrategy::Systematic,
        );
        pf.initialize_uniform(&[-1.0], &[1.0])
            .expect("should succeed in test");
        pf.set_transition(Box::new(|x: &[f64], _rng: &mut dyn FnMut() -> f64| {
            x.to_vec()
        }));
        let result = pf.step(&[0.0]);
        assert!(result.is_err(), "should return Err on weight collapse");
    }

    #[test]
    fn test_effective_sample_size_uniform() {
        let n = 100_usize;
        let weights = vec![1.0 / n as f64; n];
        let neff = effective_sample_size(&weights);
        // For uniform weights Neff = N
        assert!((neff - n as f64).abs() < 1e-8, "Neff={neff}");
    }

    #[test]
    fn test_effective_sample_size_degenerate() {
        let mut weights = vec![0.0_f64; 100];
        weights[0] = 1.0;
        let neff = effective_sample_size(&weights);
        assert!((neff - 1.0).abs() < 1e-8, "Neff={neff}");
    }

    #[test]
    fn test_systematic_resample_preserves_distribution() {
        // After resampling a strongly-peaked distribution, most particles
        // should cluster around the high-weight region.
        let mut pf = ParticleFilter::new(
            200,
            1,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        pf.seed(99);
        pf.initialize_uniform(&[-10.0], &[10.0])
            .expect("should succeed in test");

        // Manually set weights so that particles near 0 get almost all weight
        let gaussian_like = |x: f64| (-(x * x) / 0.5).exp();
        let raw_weights: Vec<f64> = pf.particles.iter().map(|p| gaussian_like(p[0])).collect();
        let sum: f64 = raw_weights.iter().sum();
        pf.log_weights = raw_weights.iter().map(|w| (w / sum).ln()).collect();

        pf.resample().expect("should succeed in test");
        // After resampling, weights are uniform
        let new_weights = pf.weights();
        let target = 1.0 / 200.0;
        for &w in &new_weights {
            assert!((w - target).abs() < 1e-12, "post-resample weight {w}");
        }
        // Particles should be concentrated near 0
        let positions: Vec<f64> = pf.particles.iter().map(|p| p[0]).collect();
        let near_zero = positions.iter().filter(|&&x| x.abs() < 2.0).count();
        assert!(near_zero > 150, "only {near_zero}/200 particles near zero");
    }

    #[test]
    fn test_student_t_likelihood() {
        let h = vec![vec![1.0_f64]];
        let lik = StudentTLikelihood::new(h, vec![1.0], 3.0).expect("should succeed in test");
        // At zero residual, likelihood should be maximal
        let l_zero = lik.likelihood(&[0.0], &[0.0]);
        let l_far = lik.likelihood(&[10.0], &[0.0]);
        assert!(l_zero > l_far, "l_zero={l_zero}, l_far={l_far}");
    }

    #[test]
    fn test_map_state() {
        let mut pf = ParticleFilter::new(
            10,
            1,
            make_gaussian_likelihood(),
            ResamplingStrategy::Systematic,
        );
        // Manually set a known high-weight particle
        pf.particles[5] = vec![42.0];
        pf.log_weights[5] = 0.0; // highest (exp(0)=1)
        let map = pf.map_state();
        assert!((map[0] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_resampling() {
        let mut pf = ParticleFilter::new(
            100,
            1,
            make_gaussian_likelihood(),
            ResamplingStrategy::Multinomial,
        );
        pf.seed(77);
        pf.initialize_uniform(&[0.0], &[1.0])
            .expect("should succeed in test");
        // Just check it runs and produces correct particle count
        pf.resample().expect("should succeed in test");
        assert_eq!(pf.particles.len(), 100);
    }

    #[test]
    fn test_stratified_resampling() {
        let mut pf = ParticleFilter::new(
            100,
            1,
            make_gaussian_likelihood(),
            ResamplingStrategy::Stratified,
        );
        pf.seed(55);
        pf.initialize_uniform(&[0.0], &[1.0])
            .expect("should succeed in test");
        pf.resample().expect("should succeed in test");
        assert_eq!(pf.particles.len(), 100);
    }
}
