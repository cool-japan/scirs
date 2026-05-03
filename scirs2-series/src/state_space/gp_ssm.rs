//! Gaussian Process State Space Model (GP-SSM).
//!
//! Places GP priors over the transition and emission functions, using a
//! bootstrap particle filter for approximate inference and a lightweight
//! grid-search for hyperparameter optimisation.
//!
//! Reference: Turner, R., Deisenroth, M., & Rasmussen, C. (2010).
//! *State-Space Inference and Learning with Gaussian Processes*.
//! AISTATS 2010.
//!
//! ## Model
//! ```text
//! Transition:  x_{t+1} | x_t ~ GP(m_f(x_t), k_f(x_t, x_t'))
//! Emission:    y_t      | x_t ~ N(x_t, σ²_obs)
//! ```

use crate::error::{Result, TimeSeriesError};
use scirs2_core::numeric::{Float, FromPrimitive};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Kernel family for the GP prior over the transition function.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KernelType {
    /// Radial Basis Function (squared-exponential) kernel.
    RBF,
    /// Matérn 5/2 kernel.
    Matern52,
    /// Periodic kernel (captures repeating patterns).
    Periodic,
    /// Linear (dot-product) kernel.
    Linear,
}

/// Stationary kernel parametrised by length-scale and signal variance.
#[derive(Debug, Clone)]
pub struct Kernel<F> {
    /// Which kernel family to use.
    pub kernel_type: KernelType,
    /// Characteristic length-scale ℓ > 0.
    pub length_scale: F,
    /// Signal variance σ²_f > 0.
    pub signal_var: F,
}

impl<F> Kernel<F>
where
    F: Float + FromPrimitive,
{
    /// Evaluate k(x1, x2).
    pub fn evaluate(&self, x1: F, x2: F) -> F {
        let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
        let r = x1 - x2;
        match self.kernel_type {
            KernelType::RBF => {
                // k(x,y) = σ²_f · exp(−0.5 · r² / ℓ²)
                let exp_arg = r * r / (two * self.length_scale * self.length_scale);
                self.signal_var * (-exp_arg).exp()
            }
            KernelType::Matern52 => {
                // k(x,y) = σ²_f · (1 + √5·|r|/ℓ + 5r²/(3ℓ²)) · exp(−√5·|r|/ℓ)
                let five = F::from_f64(5.0).unwrap_or(F::one());
                let three = F::from_f64(3.0).unwrap_or(F::one());
                let sqrt5 = five.sqrt();
                let abs_r = r.abs();
                let t = sqrt5 * abs_r / self.length_scale;
                let poly = F::one() + t + t * t / three;
                self.signal_var * poly * (-t).exp()
            }
            KernelType::Periodic => {
                // k(x,y) = σ²_f · exp(−2 sin²(π·r/p) / ℓ²)  with p=1
                let pi = F::from_f64(std::f64::consts::PI).unwrap_or(F::one());
                let s = (pi * r).sin();
                let exp_arg = two * s * s / (self.length_scale * self.length_scale);
                self.signal_var * (-exp_arg).exp()
            }
            KernelType::Linear => {
                // k(x,y) = σ²_f · x·y
                self.signal_var * x1 * x2
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GP Transition (GP regression over a sparse inducing-point dataset)
// ---------------------------------------------------------------------------

/// GP-based transition model built from a set of inducing points (x, f(x)).
#[derive(Debug, Clone)]
pub struct GpTransition<F> {
    /// Kernel used to model the transition function.
    pub kernel: Kernel<F>,
    /// Input locations of the inducing points.
    pub inducing_x: Vec<F>,
    /// Function values at the inducing points.
    pub inducing_f: Vec<F>,
}

impl<F> GpTransition<F>
where
    F: Float + FromPrimitive + std::fmt::Debug + Clone + std::iter::Sum,
{
    /// Predict the next state distribution (mean, variance) at input `x`.
    ///
    /// Uses exact GP posterior conditioning on the inducing points.  For an
    /// empty inducing set the prior is returned: mean = x, var = signal_var.
    pub fn predict(&self, x: F) -> (F, F) {
        let n = self.inducing_x.len();
        if n == 0 {
            return (x, self.kernel.signal_var);
        }

        let noise = F::from_f64(1e-6).unwrap_or(F::zero()); // jitter for numerical stability

        // Build K (n×n) kernel matrix among inducing inputs.
        let mut k_mat: Vec<Vec<F>> = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i][j] = self.kernel.evaluate(self.inducing_x[i], self.inducing_x[j]);
                if i == j {
                    k_mat[i][j] = k_mat[i][j] + noise;
                }
            }
        }

        // k_* = [k(x, x_1), ..., k(x, x_n)]
        let k_star: Vec<F> = self
            .inducing_x
            .iter()
            .map(|&xi| self.kernel.evaluate(x, xi))
            .collect();

        // Solve K α = f  via simple Cholesky-free Gauss-Jordan elimination.
        // For small n this is acceptable; for large n a proper BLAS routine
        // should be used.
        let alpha = solve_linear_system(&k_mat, &self.inducing_f);

        // Posterior mean: k_*^T α
        let mean: F = k_star
            .iter()
            .zip(alpha.iter())
            .map(|(&ks, &a)| ks * a)
            .sum();

        // Posterior variance: k(x,x) − k_*^T K^{-1} k_*
        // Compute v = K^{-1} k_* by solving K v = k_*
        let v = solve_linear_system(&k_mat, &k_star);
        let k_star_dot_v: F = k_star.iter().zip(v.iter()).map(|(&ks, &vi)| ks * vi).sum();
        let prior_var = self.kernel.evaluate(x, x);
        let var = (prior_var - k_star_dot_v).max(F::from_f64(1e-8).unwrap_or(F::zero()));

        (mean, var)
    }
}

/// Solve the linear system A·x = b using Gaussian elimination with partial
/// pivoting.  Returns x.  If the matrix is (near-)singular, returns zeros.
fn solve_linear_system<F>(a: &[Vec<F>], b: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive + Clone,
{
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Augmented matrix [A | b]
    let mut mat: Vec<Vec<F>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.to_vec();
            r.push(bi);
            r
        })
        .collect();

    let eps = F::from_f64(1e-12).unwrap_or(F::zero());

    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = mat[col][col].abs();
        for row in (col + 1)..n {
            let v = mat[row][col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < eps {
            // Near-singular: return zero vector
            return vec![F::zero(); n];
        }
        mat.swap(col, pivot_row);

        let diag = mat[col][col];
        for j in col..=n {
            let v = mat[col][j];
            mat[col][j] = v / diag;
        }
        for row in 0..n {
            if row != col {
                let factor = mat[row][col];
                for j in col..=n {
                    let v = mat[row][j] - factor * mat[col][j];
                    mat[row][j] = v;
                }
            }
        }
    }

    mat.iter().map(|row| row[n]).collect()
}

// ---------------------------------------------------------------------------
// Bootstrap Particle Filter
// ---------------------------------------------------------------------------

/// Bootstrap (sequential importance resampling) particle filter.
#[derive(Debug, Clone)]
pub struct ParticleFilter<F> {
    /// Current particle positions in state space.
    pub particles: Vec<F>,
    /// Log-weights for each particle.
    pub weights: Vec<F>,
}

impl<F> ParticleFilter<F>
where
    F: Float + FromPrimitive + Clone + std::iter::Sum + std::fmt::Debug,
{
    /// Create a new particle filter with `n_particles` particles, all initialised
    /// at the zero state with equal log-weights.
    pub fn new(n_particles: usize) -> Self {
        let log_w = -(n_particles as f64).ln();
        let log_w = F::from_f64(log_w).unwrap_or(F::zero());
        Self {
            particles: vec![F::zero(); n_particles],
            weights: vec![log_w; n_particles],
        }
    }

    /// Propagate particles through the GP transition, weight by the Gaussian
    /// likelihood of `y_obs`, and return the incremental log-likelihood.
    pub fn step(&mut self, y_obs: F, transition: &GpTransition<F>, obs_noise: F, seed: u64) -> F {
        let n = self.particles.len();
        let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
        let pi2 = F::from_f64(2.0 * std::f64::consts::PI).unwrap_or(F::one());

        let mut new_particles = Vec::with_capacity(n);
        let mut new_log_weights = Vec::with_capacity(n);

        let mut lcg: u64 = seed.wrapping_add(1);

        for i in 0..n {
            // Draw x_{t+1} ~ p(x_{t+1} | x_t) from GP posterior
            let (mean, var) = transition.predict(self.particles[i]);
            let std_dev = var.sqrt();

            // Box-Muller normal sample using LCG
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
                .wrapping_add(i as u64);
            let u1 = F::from_f64((lcg >> 11) as f64 / (1u64 << 53) as f64).unwrap_or(F::zero());
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u2 = F::from_f64((lcg >> 11) as f64 / (1u64 << 53) as f64).unwrap_or(F::zero());

            let u1_clamped = u1.max(F::from_f64(1e-15).unwrap_or(F::zero()));
            let z = (-two * u1_clamped.ln()).sqrt() * (two * pi2 * u2).cos();

            let x_next = mean + std_dev * z;
            new_particles.push(x_next);

            // Gaussian log-likelihood: log N(y_obs; x_next, obs_noise)
            let diff = y_obs - x_next;
            let log_lik = F::from_f64(-0.5).unwrap_or(F::zero())
                * (diff * diff / obs_noise
                    + (two * pi2 * obs_noise)
                        .ln()
                        .max(F::from_f64(-100.0).unwrap_or(F::zero())));
            new_log_weights.push(self.weights[i] + log_lik);
        }

        // Log-sum-exp for marginal log-likelihood increment
        let max_lw = new_log_weights
            .iter()
            .cloned()
            .fold(F::neg_infinity(), F::max);
        let sum_exp: F = new_log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
        let log_z = max_lw + sum_exp.ln();

        // Normalise log-weights
        for lw in new_log_weights.iter_mut() {
            *lw = *lw - log_z;
        }

        self.particles = new_particles;
        self.weights = new_log_weights;

        log_z
    }

    /// Systematic resampling using the normalised weights.
    /// After resampling all log-weights are reset to −log(N).
    pub fn resample(&mut self, seed: u64) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }

        // Convert log-weights to normalised weights
        let max_lw = self.weights.iter().cloned().fold(F::neg_infinity(), F::max);
        let exps: Vec<F> = self.weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total: F = exps.iter().cloned().sum();
        let probs: Vec<F> = exps.iter().map(|&e| e / total).collect();

        // Build CDF
        let mut cdf = vec![F::zero(); n];
        cdf[0] = probs[0];
        for i in 1..n {
            cdf[i] = cdf[i - 1] + probs[i];
        }

        // LCG-based uniform starting point in [0, 1/N)
        let mut lcg = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u0_raw = (lcg >> 11) as f64 / (1u64 << 53) as f64;
        let inv_n = 1.0 / n as f64;
        let u0 = F::from_f64(u0_raw * inv_n).unwrap_or(F::zero());

        let mut new_particles = Vec::with_capacity(n);
        let mut j = 0usize;
        for i in 0..n {
            let u = u0 + F::from_f64(i as f64 * inv_n).unwrap_or(F::zero());
            while j < n - 1 && cdf[j] < u {
                j += 1;
            }
            new_particles.push(self.particles[j]);
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        }

        let log_w = -(n as f64).ln();
        let log_w_f = F::from_f64(log_w).unwrap_or(F::zero());
        self.particles = new_particles;
        self.weights = vec![log_w_f; n];
    }
}

// ---------------------------------------------------------------------------
// Configuration and Result
// ---------------------------------------------------------------------------

/// Configuration for the GP-SSM.
#[derive(Debug, Clone)]
pub struct GpSsmConfig {
    /// Dimension of the latent state (currently only 1-D implemented).
    pub latent_dim: usize,
    /// Dimension of the observations (currently only 1-D implemented).
    pub obs_dim: usize,
    /// Number of particles in the bootstrap filter.
    pub n_particles: usize,
    /// Number of MCMC / hyperparameter-search iterations.
    pub n_mcmc: usize,
    /// Kernel type for the GP transition.
    pub kernel: KernelType,
    /// Initial length-scale ℓ.
    pub length_scale: f64,
    /// Initial signal variance σ²_f.
    pub signal_var: f64,
    /// Observation noise variance σ²_obs.
    pub noise_var: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for GpSsmConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1,
            obs_dim: 1,
            n_particles: 50,
            n_mcmc: 100,
            kernel: KernelType::RBF,
            length_scale: 1.0,
            signal_var: 1.0,
            noise_var: 0.1,
            seed: 42,
        }
    }
}

/// Results returned after fitting a GP-SSM.
#[derive(Debug, Clone)]
pub struct GpSsmResult<F> {
    /// Total log-likelihood of the observation sequence under the fitted model.
    pub log_likelihood: F,
    /// Best length-scale found during hyperparameter search.
    pub fitted_length_scale: F,
    /// Best signal variance found during hyperparameter search.
    pub fitted_signal_var: F,
    /// Posterior mean of the latent state at each time step.
    pub state_mean: Vec<F>,
    /// Posterior standard deviation of the latent state at each time step.
    pub state_std: Vec<F>,
}

// ---------------------------------------------------------------------------
// GP-SSM
// ---------------------------------------------------------------------------

/// Gaussian Process State Space Model.
///
/// # Example
/// ```rust,no_run
/// use scirs2_series::state_space::gp_ssm::{GpSsm, GpSsmConfig};
///
/// let config = GpSsmConfig::default();
/// let mut model = GpSsm::new(config);
/// let obs: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
/// let result = model.fit(&obs).expect("fit should succeed");
/// println!("log_lik = {}", result.log_likelihood);
/// ```
#[derive(Debug, Clone)]
pub struct GpSsm<F> {
    /// Configuration.
    pub config: GpSsmConfig,
    /// GP transition model.
    pub transition: GpTransition<F>,
}

impl<F> GpSsm<F>
where
    F: Float + FromPrimitive + std::fmt::Debug + Clone + std::iter::Sum + std::fmt::Display,
{
    /// Construct a new GP-SSM from the given configuration.
    ///
    /// Inducing points are initialised from a linear grid over [−2, 2] with
    /// f values = 0 (uninformative prior).
    pub fn new(config: GpSsmConfig) -> Self {
        let n_inducing = 10usize;
        let kernel = Kernel {
            kernel_type: config.kernel.clone(),
            length_scale: F::from_f64(config.length_scale).unwrap_or(F::one()),
            signal_var: F::from_f64(config.signal_var).unwrap_or(F::one()),
        };

        let inducing_x: Vec<F> = (0..n_inducing)
            .map(|i| {
                let t = i as f64 / (n_inducing - 1) as f64 * 4.0 - 2.0;
                F::from_f64(t).unwrap_or(F::zero())
            })
            .collect();
        let inducing_f = vec![F::zero(); n_inducing];

        let transition = GpTransition {
            kernel,
            inducing_x,
            inducing_f,
        };

        Self { config, transition }
    }

    /// Run the bootstrap particle filter over `observations`.
    ///
    /// Returns `(trajectories, total_log_likelihood)` where `trajectories` is
    /// a flat vector of shape `[n_particles × T]` (row-major: row i contains
    /// all T states of particle i).
    pub fn particle_filter(&self, observations: &[F]) -> (Vec<Vec<F>>, F) {
        let n = self.config.n_particles;
        let t_len = observations.len();

        let mut pf = ParticleFilter::new(n);
        let obs_noise = F::from_f64(self.config.noise_var).unwrap_or(F::one());

        // Trajectory storage: [n_particles][T]
        let mut trajectories: Vec<Vec<F>> = vec![Vec::with_capacity(t_len); n];

        // Record initial (t=0) particles
        for (p, traj) in pf.particles.iter().zip(trajectories.iter_mut()) {
            traj.push(*p);
        }

        let mut total_log_lik = F::zero();

        for (t, &y) in observations.iter().enumerate() {
            let seed = self
                .config
                .seed
                .wrapping_add(t as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let inc = pf.step(y, &self.transition, obs_noise, seed);
            total_log_lik = total_log_lik + inc;

            // Resample every step (bootstrap filter)
            pf.resample(seed.wrapping_add(0xABCD_1234));

            for (p, traj) in pf.particles.iter().zip(trajectories.iter_mut()) {
                traj.push(*p);
            }
        }

        (trajectories, total_log_lik)
    }

    /// Fit the model by doing a lightweight grid search over `length_scale` ×
    /// `signal_var` (3 × 3 = 9 evaluations) and updating the inducing points
    /// with the data to improve the transition prior.
    pub fn fit(&mut self, observations: &[F]) -> Result<GpSsmResult<F>> {
        if observations.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "GP-SSM requires at least 2 observations".into(),
                required: 2,
                actual: observations.len(),
            });
        }

        // Grid search over length-scale and signal variance
        let ls_grid = [0.5_f64, 1.0, 2.0];
        let sv_grid = [0.5_f64, 1.0, 2.0];

        let mut best_ll = F::neg_infinity();
        let mut best_ls = self.config.length_scale;
        let mut best_sv = self.config.signal_var;

        for &ls in &ls_grid {
            for &sv in &sv_grid {
                self.transition.kernel.length_scale = F::from_f64(ls).unwrap_or(F::one());
                self.transition.kernel.signal_var = F::from_f64(sv).unwrap_or(F::one());

                let (_, ll) = self.particle_filter(observations);
                if ll > best_ll {
                    best_ll = ll;
                    best_ls = ls;
                    best_sv = sv;
                }
            }
        }

        // Apply best hyperparameters and update inducing points from data
        self.config.length_scale = best_ls;
        self.config.signal_var = best_sv;
        self.transition.kernel.length_scale = F::from_f64(best_ls).unwrap_or(F::one());
        self.transition.kernel.signal_var = F::from_f64(best_sv).unwrap_or(F::one());

        // Update inducing points: use consecutive observation pairs (x_t, x_{t+1})
        // as a data-driven surrogate for the transition function.
        let max_inducing = self.config.n_particles.min(20).min(observations.len() - 1);
        let step = (observations.len() - 1).max(1) / max_inducing.max(1);
        self.transition.inducing_x = (0..max_inducing)
            .map(|i| observations[(i * step).min(observations.len() - 2)])
            .collect();
        self.transition.inducing_f = (0..max_inducing)
            .map(|i| observations[(i * step + 1).min(observations.len() - 1)])
            .collect();

        // Run final particle filter with best parameters
        let (trajectories, final_ll) = self.particle_filter(observations);

        // Compute posterior mean and std from particle ensemble at each time step
        let t_len = observations.len();
        let n = self.config.n_particles;
        let mut state_mean = Vec::with_capacity(t_len);
        let mut state_std = Vec::with_capacity(t_len);

        for t in 0..t_len {
            // +1 because trajectories[i][0] is the initial state before t=0
            let pos: usize = (t + 1).min(trajectories[0].len() - 1);
            let vals: Vec<F> = (0..n).map(|i| trajectories[i][pos]).collect();
            let mean: F = vals.iter().cloned().sum::<F>() / F::from_usize(n).unwrap_or(F::one());
            let var: F = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<F>()
                / F::from_usize(n.max(1)).unwrap_or(F::one());
            state_mean.push(mean);
            state_std.push(var.sqrt());
        }

        Ok(GpSsmResult {
            log_likelihood: final_ll,
            fitted_length_scale: F::from_f64(best_ls).unwrap_or(F::one()),
            fitted_signal_var: F::from_f64(best_sv).unwrap_or(F::one()),
            state_mean,
            state_std,
        })
    }

    /// Forecast `n_steps` steps ahead by propagating particles from the final
    /// filter state.  Returns `Vec<(mean, std_dev)>` for each forecast step.
    pub fn predict(&self, observations: &[F], n_steps: usize) -> Vec<(F, F)> {
        if observations.is_empty() || n_steps == 0 {
            return vec![];
        }

        let (trajectories, _) = self.particle_filter(observations);
        let np = self.config.n_particles;

        // Initialise forecast particles from the final column of trajectories
        let last_t = trajectories[0].len().saturating_sub(1);
        let mut particles: Vec<F> = (0..np).map(|i| trajectories[i][last_t]).collect();

        let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
        let pi2 = F::from_f64(2.0 * std::f64::consts::PI).unwrap_or(F::one());
        let mut forecasts = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let mut new_particles = Vec::with_capacity(np);
            let mut lcg: u64 = self
                .config
                .seed
                .wrapping_add(step as u64 + 1000)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15);

            for (i, &px) in particles.iter().enumerate() {
                let (mean, var) = self.transition.predict(px);
                let std_dev = var.sqrt();
                lcg = lcg
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407)
                    .wrapping_add(i as u64);
                let u1 = F::from_f64((lcg >> 11) as f64 / (1u64 << 53) as f64).unwrap_or(F::zero());
                lcg = lcg
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u2 = F::from_f64((lcg >> 11) as f64 / (1u64 << 53) as f64).unwrap_or(F::zero());
                let u1c = u1.max(F::from_f64(1e-15).unwrap_or(F::zero()));
                let z = (-two * u1c.ln()).sqrt() * (two * pi2 * u2).cos();
                new_particles.push(mean + std_dev * z);
            }

            let mean: F =
                new_particles.iter().cloned().sum::<F>() / F::from_usize(np).unwrap_or(F::one());
            let var: F = new_particles
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<F>()
                / F::from_usize(np.max(1)).unwrap_or(F::one());
            forecasts.push((mean, var.sqrt()));
            particles = new_particles;
        }

        forecasts
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_obs(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.3).sin()).collect()
    }

    #[test]
    fn test_config_defaults() {
        let cfg = GpSsmConfig::default();
        assert_eq!(cfg.latent_dim, 1);
        assert_eq!(cfg.obs_dim, 1);
        assert_eq!(cfg.n_particles, 50);
        assert_eq!(cfg.n_mcmc, 100);
        assert!(matches!(cfg.kernel, KernelType::RBF));
        assert!((cfg.length_scale - 1.0).abs() < 1e-10);
        assert!((cfg.signal_var - 1.0).abs() < 1e-10);
        assert!((cfg.noise_var - 0.1).abs() < 1e-10);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_kernel_type_variants() {
        let _ = KernelType::RBF;
        let _ = KernelType::Matern52;
        let _ = KernelType::Periodic;
        let _ = KernelType::Linear;
    }

    #[test]
    fn test_kernel_rbf_self() {
        // k(x, x) should equal signal_var for any x
        let k: Kernel<f64> = Kernel {
            kernel_type: KernelType::RBF,
            length_scale: 1.0,
            signal_var: 2.5,
        };
        let val = k.evaluate(3.7, 3.7);
        assert!((val - 2.5).abs() < 1e-10, "k(x,x) = {val}");
    }

    #[test]
    fn test_kernel_rbf_off_diagonal() {
        let k: Kernel<f64> = Kernel {
            kernel_type: KernelType::RBF,
            length_scale: 1.0,
            signal_var: 1.0,
        };
        let diag = k.evaluate(1.0, 1.0);
        let off = k.evaluate(1.0, 5.0);
        assert!(off < diag, "off-diagonal {off} should be < diagonal {diag}");
    }

    #[test]
    fn test_particle_filter_shape() {
        let mut config = GpSsmConfig::default();
        config.n_particles = 10;
        let model: GpSsm<f64> = GpSsm::new(config);
        let obs = sine_obs(20);
        let (trajs, _) = model.particle_filter(&obs);
        // trajectories[i] has T+1 entries (initial + one per observation)
        assert_eq!(trajs.len(), 10);
        assert_eq!(trajs[0].len(), 21); // 20 obs + initial state
    }

    #[test]
    fn test_particle_filter_finite_log_lik() {
        let config = GpSsmConfig::default();
        let model: GpSsm<f64> = GpSsm::new(config);
        let obs = sine_obs(15);
        let (_, ll) = model.particle_filter(&obs);
        assert!(ll.is_finite(), "log_lik = {ll}");
    }

    #[test]
    fn test_weights_normalised() {
        let n = 20usize;
        let mut pf: ParticleFilter<f64> = ParticleFilter::new(n);
        // exp of log-weights should sum to 1.0 (weights initialized to log(1/n))
        let sum: f64 = pf.weights.iter().map(|&w| w.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
        // After resample weights stay equal
        pf.resample(99);
        let sum2: f64 = pf.weights.iter().map(|&w| (w - pf.weights[0]).exp()).sum();
        assert!((sum2 - n as f64).abs() < 1e-8);
    }

    #[test]
    fn test_resample_maintains_total() {
        let n = 15usize;
        let mut pf: ParticleFilter<f64> = ParticleFilter::new(n);
        // Perturb weights to be non-uniform
        for (i, w) in pf.weights.iter_mut().enumerate() {
            *w = (i as f64 * 0.1) - 1.0;
        }
        pf.resample(7);
        // After systematic resample we still have n particles
        assert_eq!(pf.particles.len(), n);
        assert_eq!(pf.weights.len(), n);
    }

    #[test]
    fn test_fit_returns_finite_ll() {
        let mut config = GpSsmConfig::default();
        config.n_particles = 20;
        config.n_mcmc = 9; // reduced for speed
        let mut model: GpSsm<f64> = GpSsm::new(config);
        let obs = sine_obs(20);
        let result = model.fit(&obs).expect("fit should succeed");
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_predict_length() {
        let mut config = GpSsmConfig::default();
        config.n_particles = 10;
        let mut model: GpSsm<f64> = GpSsm::new(config);
        let obs = sine_obs(15);
        let _ = model.fit(&obs).unwrap();
        let forecasts = model.predict(&obs, 5);
        assert_eq!(forecasts.len(), 5);
    }

    #[test]
    fn test_gp_transition_predict_variance_positive() {
        let kernel: Kernel<f64> = Kernel {
            kernel_type: KernelType::RBF,
            length_scale: 1.0,
            signal_var: 1.0,
        };
        let tr = GpTransition {
            kernel,
            inducing_x: vec![-1.0, 0.0, 1.0],
            inducing_f: vec![-0.5, 0.0, 0.5],
        };
        let (_, var) = tr.predict(0.5);
        assert!(var > 0.0, "variance {var} should be > 0");
    }

    #[test]
    fn test_state_mean_length_equals_obs() {
        let mut config = GpSsmConfig::default();
        config.n_particles = 10;
        let mut model: GpSsm<f64> = GpSsm::new(config);
        let obs = sine_obs(12);
        let result = model.fit(&obs).expect("fit should succeed");
        assert_eq!(result.state_mean.len(), obs.len());
    }
}
