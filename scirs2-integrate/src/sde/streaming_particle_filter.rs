//! Streaming (real-time) SIR particle filter for sequential Bayesian state estimation.
//!
//! Implements the Bootstrap / SIR particle filter with adaptive systematic resampling
//! (Carpenter, Clifford & Fearnhead, 1999).  Observations are processed one at a time
//! via [`StreamingParticleFilter::step`], making this suitable for real-time applications.
//!
//! ## Algorithm
//!
//! At each time step `t`:
//!
//! 1. **Propagate**: draw `x_t^i ~ q(x_t | x_{t-1}^i)` for each particle.
//! 2. **Weight**: set `log w_t^i += log p(y_t | x_t^i)`.
//! 3. **Normalize**: use log-sum-exp for numerical stability.
//! 4. **Resample** (if ESS < N × ess_threshold): systematic resampling in O(N).
//!
//! ## Memory
//!
//! All particle arrays are pre-allocated; no per-step heap growth occurs.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::ndarray::{arr1, Array1};
//! use scirs2_integrate::sde::streaming_particle_filter::{
//!     StreamingParticleFilterBuilder, SimpleRng,
//! };
//!
//! let mut filter = StreamingParticleFilterBuilder::new(200)
//!     .transition(|x, seed| {
//!         let mut rng = SimpleRng::new(seed);
//!         arr1(&[x[0] + rng.normal() * 0.1])
//!     })
//!     .log_likelihood(|x, obs: &Array1<f64>| {
//!         let diff = x[0] - obs[0];
//!         -0.5 * diff * diff / 0.25
//!     })
//!     .ess_threshold(0.5)
//!     .seed(42)
//!     .build()
//!     .expect("build should succeed");
//!
//! let obs = arr1(&[0.1_f64]);
//! let est = filter.step(&obs);
//! assert!(est.effective_sample_size > 0.0);
//! ```

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ---------------------------------------------------------------------------
// Minimal deterministic PRNG — no external rand dependency
// ---------------------------------------------------------------------------

/// A minimal deterministic PRNG (PCG-XSH-RR 64-bit) suitable for particle noise.
///
/// Exposes `uniform`, `normal` (Box-Muller), and `next_u64`.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::streaming_particle_filter::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let u = rng.uniform();
/// assert!(u >= 0.0 && u < 1.0);
/// let n = rng.normal();
/// assert!(n.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
    inc: u64,
}

impl SimpleRng {
    /// Create a new `SimpleRng` seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        // PCG initialisation sequence
        let inc = seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1;
        let mut state = seed.wrapping_add(inc);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(inc);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(inc);
        Self { state, inc }
    }

    /// Advance the PRNG and return a 64-bit random integer.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = ((self.state >> 18) ^ self.state) >> 27;
        let rot = (self.state >> 59) as u32;
        let output = (xorshifted as u32).rotate_right(rot) as u64;
        // Combine two 32-bit outputs into a 64-bit value
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted2 = ((self.state >> 18) ^ self.state) >> 27;
        let rot2 = (self.state >> 59) as u32;
        let output2 = (xorshifted2 as u32).rotate_right(rot2) as u64;
        (output << 32) | output2
    }

    /// Return a sample uniform on `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53-bit mantissa
        bits as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Return a standard Normal sample via Box-Muller transform.
    ///
    /// Uses a single pair of uniform samples per call; the second variate is discarded.
    pub fn normal(&mut self) -> f64 {
        // Guard against log(0) by clamping u1 away from zero
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// FilterEstimate
// ---------------------------------------------------------------------------

/// Summary statistics returned by one call to [`StreamingParticleFilter::step`].
#[derive(Debug, Clone)]
pub struct FilterEstimate {
    /// Particle-weighted mean of the state vector.
    pub mean: Array1<f64>,
    /// Particle-weighted covariance matrix of the state.
    pub covariance: Array2<f64>,
    /// Effective sample size ∈ `(0, N]`.
    pub effective_sample_size: f64,
    /// Log marginal likelihood accumulated over all steps so far.
    pub log_marginal_likelihood: f64,
    /// Total number of resampling events triggered so far.
    pub resample_count: usize,
}

// ---------------------------------------------------------------------------
// StreamingParticleFilter
// ---------------------------------------------------------------------------

/// Streaming SIR particle filter with adaptive systematic resampling.
///
/// Processes observations sequentially in constant memory.  The particle
/// array and weight vector are allocated once at construction time and reused
/// across [`step`](Self::step) calls.
///
/// # Type constraints
///
/// The transition and log-likelihood closures must be `Send` so that the
/// filter itself can be sent across threads.
pub struct StreamingParticleFilter {
    /// Particle states: shape `(n_particles, state_dim)`.
    particles: Array2<f64>,
    /// Unnormalized log-weights, length `n_particles`.
    log_weights: Array1<f64>,
    /// Transition: `(state_view, seed) -> new_state`
    transition: Box<dyn Fn(&ArrayView1<f64>, u64) -> Array1<f64> + Send>,
    /// Log-likelihood: `(state_view, observation) -> log_p`
    log_likelihood: Box<dyn Fn(&ArrayView1<f64>, &Array1<f64>) -> f64 + Send>,
    /// Resample when ESS < n_particles * ess_threshold.
    ess_threshold: f64,
    /// Accumulated log marginal likelihood (log evidence).
    log_evidence: f64,
    /// Total resampling events triggered.
    resample_count: usize,
    /// Internal PRNG state.
    rng: SimpleRng,
    /// Number of filter steps executed.
    step_count: usize,
    /// Dimension of each particle state.
    state_dim: usize,
    /// Scratch buffer for propagated particles (avoids per-step allocation).
    scratch: Array2<f64>,
}

impl StreamingParticleFilter {
    // ---- Public accessors ----

    /// Number of particles `N`.
    #[inline]
    pub fn n_particles(&self) -> usize {
        self.particles.nrows()
    }

    /// Dimension of each particle state.
    #[inline]
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Total number of [`step`](Self::step) calls executed so far.
    #[inline]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Total number of resampling events triggered so far.
    #[inline]
    pub fn resample_count(&self) -> usize {
        self.resample_count
    }

    /// Current effective sample size ∈ `(0, N]`.
    pub fn effective_sample_size(&self) -> f64 {
        compute_ess(&self.log_weights)
    }

    /// Particle-weighted mean of the current state distribution.
    pub fn mean(&self) -> Array1<f64> {
        weighted_mean(&self.particles, &self.log_weights)
    }

    /// Particle-weighted covariance matrix of the current state distribution.
    pub fn covariance(&self) -> Array2<f64> {
        weighted_covariance(&self.particles, &self.log_weights)
    }

    /// Accumulated log marginal likelihood (log evidence).
    #[inline]
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_evidence
    }

    // ---- Core algorithm ----

    /// Perform one SIR filter step given a new `observation`.
    ///
    /// # Sequence
    ///
    /// 1. **Propagate** each particle through the transition function.
    /// 2. **Update** log-weights with the observation log-likelihood.
    /// 3. **Accumulate** log marginal likelihood increment.
    /// 4. **Normalize** log-weights (log-sum-exp).
    /// 5. **Resample** if ESS < threshold.
    ///
    /// # Arguments
    ///
    /// * `observation` – the current observation vector `y_t`.
    ///
    /// # Returns
    ///
    /// A [`FilterEstimate`] containing mean, covariance, ESS, accumulated log
    /// evidence, and resample count.
    pub fn step(&mut self, observation: &Array1<f64>) -> FilterEstimate {
        let n = self.n_particles();
        let state_dim = self.state_dim;

        // ---- 1. Propagate particles ----
        for i in 0..n {
            let seed = self.rng.next_u64();
            let row = self.particles.row(i);
            let new_state = (self.transition)(&row, seed);
            // Copy into scratch, padding/truncating to state_dim if needed
            for d in 0..state_dim {
                self.scratch[[i, d]] = if d < new_state.len() {
                    new_state[d]
                } else {
                    0.0
                };
            }
        }
        // Swap scratch into particles (zero-copy swap of underlying data)
        std::mem::swap(&mut self.particles, &mut self.scratch);

        // ---- 2. Update log-weights with observation log-likelihood ----
        // Capture the pre-update normalisation constant (should be 0 for
        // normalized weights, but we track it explicitly for correctness
        // immediately after initialization or any other corner case).
        let log_z_prior = logsumexp(&self.log_weights);
        for i in 0..n {
            let row = self.particles.row(i);
            let ll = (self.log_likelihood)(&row, observation);
            self.log_weights[i] += ll;
        }

        // ---- 3. Accumulate log marginal likelihood increment ----
        // The incremental log evidence is:
        //   log p(y_t | y_{1:t-1}) = logsumexp(w_after) - logsumexp(w_before)
        // where log_z_prior = logsumexp(w_before) ≈ 0 for normalized weights.
        let log_z = logsumexp(&self.log_weights);
        if log_z.is_finite() {
            let increment = log_z - log_z_prior;
            self.log_evidence += increment;
        }

        // ---- 4. Normalize log-weights ----
        self.log_weights.mapv_inplace(|lw| lw - log_z);

        // ---- 5. Adaptive systematic resampling ----
        let ess = compute_ess(&self.log_weights);
        let threshold = self.ess_threshold * n as f64;
        if ess < threshold {
            self.systematic_resample();
            self.resample_count += 1;
        }

        // ---- 6. Build estimate ----
        let ess_after = compute_ess(&self.log_weights);
        self.step_count += 1;

        FilterEstimate {
            mean: weighted_mean(&self.particles, &self.log_weights),
            covariance: weighted_covariance(&self.particles, &self.log_weights),
            effective_sample_size: ess_after,
            log_marginal_likelihood: self.log_evidence,
            resample_count: self.resample_count,
        }
    }

    // ---- Private helpers ----

    /// Systematic resampling (Carpenter–Clifford–Fearnhead 1999).
    ///
    /// Complexity O(N).  Resets all log-weights to `log(1/N)` afterwards.
    fn systematic_resample(&mut self) {
        let n = self.n_particles();
        let log_uniform = -(n as f64).ln(); // log(1/N)

        // Compute cumulative normalized weights in linear space
        let mut cumsum = vec![0.0_f64; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + self.log_weights[i].exp();
        }
        // Clamp last entry to exactly 1.0 to avoid floating-point overshoot
        cumsum[n] = 1.0;

        // Draw U ~ Uniform(0, 1/N)
        let u0 = self.rng.uniform() / n as f64;

        // Build index list by scanning the CDF once
        let mut indices = Vec::with_capacity(n);
        let mut j = 0_usize;
        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n - 1 && cumsum[j + 1] < u {
                j += 1;
            }
            indices.push(j);
        }

        // Resample particles into scratch buffer
        let state_dim = self.state_dim;
        for (i, &src) in indices.iter().enumerate() {
            for d in 0..state_dim {
                self.scratch[[i, d]] = self.particles[[src, d]];
            }
        }
        std::mem::swap(&mut self.particles, &mut self.scratch);

        // Reset weights to uniform
        self.log_weights.fill(log_uniform);
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Log-sum-exp of a slice (numerically stable).
fn logsumexp(log_w: &Array1<f64>) -> f64 {
    let max_lw = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = log_w.iter().map(|&lw| (lw - max_lw).exp()).sum();
    max_lw + sum_exp.ln()
}

/// Effective sample size: `1 / Σ (w^i)²` where `w^i = exp(log_w^i)`.
fn compute_ess(log_w: &Array1<f64>) -> f64 {
    let sum_sq: f64 = log_w
        .iter()
        .map(|&lw| {
            let w = lw.exp();
            w * w
        })
        .sum();
    if sum_sq <= 0.0 {
        log_w.len() as f64
    } else {
        1.0 / sum_sq
    }
}

/// Particle-weighted mean.  `particles` shape: `(N, D)`.
fn weighted_mean(particles: &Array2<f64>, log_w: &Array1<f64>) -> Array1<f64> {
    let n = particles.nrows();
    let d = particles.ncols();
    let mut mean = Array1::zeros(d);
    for i in 0..n {
        let w = log_w[i].exp();
        for k in 0..d {
            mean[k] += w * particles[[i, k]];
        }
    }
    mean
}

/// Particle-weighted covariance matrix.  `particles` shape: `(N, D)`.
fn weighted_covariance(particles: &Array2<f64>, log_w: &Array1<f64>) -> Array2<f64> {
    let n = particles.nrows();
    let d = particles.ncols();
    let mean = weighted_mean(particles, log_w);
    let mut cov = Array2::zeros((d, d));
    for i in 0..n {
        let w = log_w[i].exp();
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] += w * (particles[[i, j]] - mean[j]) * (particles[[i, k]] - mean[k]);
            }
        }
    }
    cov
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`StreamingParticleFilter`].
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::{arr1, Array1};
/// use scirs2_integrate::sde::streaming_particle_filter::{
///     StreamingParticleFilterBuilder, SimpleRng,
/// };
///
/// let filter = StreamingParticleFilterBuilder::new(100)
///     .transition(|x, seed| {
///         let mut rng = SimpleRng::new(seed);
///         arr1(&[x[0] + rng.normal() * 0.05])
///     })
///     .log_likelihood(|x, obs: &Array1<f64>| {
///         let diff = x[0] - obs[0];
///         -0.5 * diff * diff
///     })
///     .ess_threshold(0.5)
///     .seed(0)
///     .initial_state(arr1(&[0.0]))
///     .initial_spread(0.5)
///     .build()
///     .expect("build should succeed");
///
/// assert_eq!(filter.n_particles(), 100);
/// ```
pub struct StreamingParticleFilterBuilder {
    n_particles: usize,
    transition: Option<Box<dyn Fn(&ArrayView1<f64>, u64) -> Array1<f64> + Send>>,
    log_likelihood: Option<Box<dyn Fn(&ArrayView1<f64>, &Array1<f64>) -> f64 + Send>>,
    ess_threshold: f64,
    seed: u64,
    initial_state: Option<Array1<f64>>,
    initial_spread: f64,
}

impl StreamingParticleFilterBuilder {
    /// Start building a filter with `n_particles` particles.
    pub fn new(n_particles: usize) -> Self {
        Self {
            n_particles,
            transition: None,
            log_likelihood: None,
            ess_threshold: 0.5,
            seed: 12345,
            initial_state: None,
            initial_spread: 1.0,
        }
    }

    /// Set the transition function `(state, seed) -> new_state`.
    ///
    /// The `seed` argument is a fresh 64-bit value derived from the filter's
    /// internal PRNG; pass it to [`SimpleRng::new`] to obtain reproducible noise.
    pub fn transition<F>(mut self, f: F) -> Self
    where
        F: Fn(&ArrayView1<f64>, u64) -> Array1<f64> + Send + 'static,
    {
        self.transition = Some(Box::new(f));
        self
    }

    /// Set the log-likelihood function `(state, observation) -> log_p`.
    pub fn log_likelihood<F>(mut self, f: F) -> Self
    where
        F: Fn(&ArrayView1<f64>, &Array1<f64>) -> f64 + Send + 'static,
    {
        self.log_likelihood = Some(Box::new(f));
        self
    }

    /// Resample when ESS < `n_particles * threshold`.  Default `0.5`.
    pub fn ess_threshold(mut self, threshold: f64) -> Self {
        self.ess_threshold = threshold;
        self
    }

    /// Deterministic RNG seed.  Default `12345`.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Initial state (centre of particle cloud).  Default: origin.
    pub fn initial_state(mut self, state: Array1<f64>) -> Self {
        self.initial_state = Some(state);
        self
    }

    /// Standard deviation used to scatter initial particles around [`initial_state`].
    /// Default `1.0`.
    pub fn initial_spread(mut self, spread: f64) -> Self {
        self.initial_spread = spread;
        self
    }

    /// Build the [`StreamingParticleFilter`].
    ///
    /// # Errors
    ///
    /// * [`IntegrateError::ValueError`] if `n_particles == 0`.
    /// * [`IntegrateError::ValueError`] if `ess_threshold` is not in `(0, 1]`.
    /// * [`IntegrateError::InvalidInput`] if `transition` or `log_likelihood`
    ///   were not provided.
    pub fn build(self) -> IntegrateResult<StreamingParticleFilter> {
        if self.n_particles == 0 {
            return Err(IntegrateError::ValueError("n_particles must be > 0".into()));
        }
        if self.ess_threshold <= 0.0 || self.ess_threshold > 1.0 {
            return Err(IntegrateError::ValueError(format!(
                "ess_threshold must be in (0, 1], got {}",
                self.ess_threshold
            )));
        }
        let transition = self.transition.ok_or_else(|| {
            IntegrateError::InvalidInput("transition function not provided".into())
        })?;
        let log_likelihood = self.log_likelihood.ok_or_else(|| {
            IntegrateError::InvalidInput("log_likelihood function not provided".into())
        })?;

        let n = self.n_particles;
        let mut rng = SimpleRng::new(self.seed);

        // Determine state dimension from initial_state or default to 1
        let state_dim = self.initial_state.as_ref().map(|s| s.len()).unwrap_or(1);
        let center = self
            .initial_state
            .unwrap_or_else(|| Array1::zeros(state_dim));
        let spread = self.initial_spread;

        // Initialise particles with Gaussian noise around center
        let mut particles = Array2::zeros((n, state_dim));
        for i in 0..n {
            for d in 0..state_dim {
                particles[[i, d]] = center[d] + spread * rng.normal();
            }
        }
        let scratch = Array2::zeros((n, state_dim));

        // Uniform initial log-weights
        let log_w0 = -(n as f64).ln();
        let log_weights = Array1::from_elem(n, log_w0);

        Ok(StreamingParticleFilter {
            particles,
            log_weights,
            transition,
            log_likelihood,
            ess_threshold: self.ess_threshold,
            log_evidence: 0.0,
            resample_count: 0,
            rng,
            step_count: 0,
            state_dim,
            scratch,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn simple_rng_uniform_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
        }
    }

    #[test]
    fn simple_rng_normal_finite() {
        let mut rng = SimpleRng::new(0);
        for _ in 0..500 {
            let n = rng.normal();
            assert!(n.is_finite(), "normal produced non-finite: {n}");
        }
    }

    #[test]
    fn filter_builds_with_defaults() {
        let f = StreamingParticleFilterBuilder::new(100)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build();
        assert!(f.is_ok(), "build should succeed");
        let filter = f.expect("build failed");
        assert_eq!(filter.n_particles(), 100);
        assert_eq!(filter.step_count(), 0);
    }

    #[test]
    fn filter_zero_particles_error() {
        let res = StreamingParticleFilterBuilder::new(0)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build();
        assert!(res.is_err());
    }

    #[test]
    fn filter_missing_transition_error() {
        let res = StreamingParticleFilterBuilder::new(10)
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build();
        assert!(res.is_err());
    }

    #[test]
    fn step_count_increments() {
        let mut filter = StreamingParticleFilterBuilder::new(50)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build()
            .expect("build");
        let obs = arr1(&[0.0_f64]);
        filter.step(&obs);
        filter.step(&obs);
        assert_eq!(filter.step_count(), 2);
    }

    #[test]
    fn ess_full_with_uniform_weights() {
        let filter = StreamingParticleFilterBuilder::new(200)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build()
            .expect("build");
        let ess = filter.effective_sample_size();
        // With uniform weights ESS ≈ N
        assert!((ess - 200.0).abs() < 1.0, "ESS should be ~N, got {ess}");
    }

    #[test]
    fn mean_near_zero_after_flat_likelihood() {
        let mut filter = StreamingParticleFilterBuilder::new(300)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .initial_state(arr1(&[0.0]))
            .initial_spread(0.01)
            .seed(1)
            .build()
            .expect("build");
        let obs = arr1(&[0.0_f64]);
        let est = filter.step(&obs);
        assert!(est.mean[0].abs() < 0.1, "mean near 0, got {}", est.mean[0]);
    }

    #[test]
    fn log_evidence_accumulates() {
        let mut filter = StreamingParticleFilterBuilder::new(100)
            .transition(|x, _| x.to_owned())
            .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
            .build()
            .expect("build");
        let obs = arr1(&[0.0_f64]);
        let e1 = filter.step(&obs);
        let e2 = filter.step(&obs);
        // Each step with flat log-likelihood (ll=0) increments evidence by 0;
        // but the function still returns a finite accumulated value.
        assert!(e1.log_marginal_likelihood.is_finite());
        assert!(e2.log_marginal_likelihood.is_finite());
    }
}
