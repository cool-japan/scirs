//! Stein Variational Gradient Descent (SVGD)
//!
//! Implements SVGD (Liu & Wang 2016) — a particle-based variational inference method
//! that maintains a set of particles and iteratively transports them to approximate
//! the target posterior distribution.
//!
//! The update rule uses a kernelized Stein operator:
//! ```text
//! theta_i <- theta_i + epsilon * phi*(theta_i)
//! phi*(theta) = (1/n) sum_j [k(theta_j, theta) * grad_theta_j log p(theta_j | x)
//!              + grad_theta_j k(theta_j, theta)]
//! ```
//! - The first term drives particles toward high-probability regions
//! - The second term acts as a repulsive force to maintain diversity
//!
//! Uses RBF kernel with median bandwidth heuristic and Adam optimizer for
//! adaptive step sizes.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

use super::{PosteriorResult, VariationalInference};

// ============================================================================
// RBF Kernel
// ============================================================================

/// Radial Basis Function (RBF) kernel: k(x, y) = exp(-||x - y||^2 / (2 h^2))
#[derive(Debug, Clone)]
pub struct RbfKernel {
    /// Bandwidth parameter h; if None, use median heuristic
    pub bandwidth: Option<f64>,
}

impl RbfKernel {
    /// Compute the median heuristic bandwidth from pairwise distances.
    /// h^2 = median(||x_i - x_j||^2) / log(n) for n particles.
    fn median_bandwidth(particles: &[Array1<f64>]) -> f64 {
        let n = particles.len();
        if n <= 1 {
            return 1.0;
        }

        let mut dists_sq: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = &particles[i] - &particles[j];
                dists_sq.push(diff.dot(&diff));
            }
        }

        if dists_sq.is_empty() {
            return 1.0;
        }

        dists_sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_sq = dists_sq[dists_sq.len() / 2];

        // h^2 = median / log(n), with floor to avoid degenerate bandwidth
        let log_n = (n as f64).ln().max(1.0);
        let h_sq = median_sq / log_n;
        h_sq.max(1e-6).sqrt()
    }

    /// Evaluate kernel k(x, y) and its gradient w.r.t. x.
    ///
    /// Returns (k_val, grad_x_k) where:
    /// - k_val = exp(-||x - y||^2 / (2 h^2))
    /// - grad_x_k = -k_val * (x - y) / h^2
    fn eval_with_grad(&self, x: &Array1<f64>, y: &Array1<f64>, h: f64) -> (f64, Array1<f64>) {
        let diff = x - y;
        let dist_sq = diff.dot(&diff);
        let h_sq = h * h;
        let k_val = (-dist_sq / (2.0 * h_sq)).exp();
        let grad_x = &diff * (-k_val / h_sq);
        (k_val, grad_x)
    }
}

// ============================================================================
// Adam Optimizer for SVGD (per-particle)
// ============================================================================

#[derive(Debug, Clone)]
struct SvgdAdamState {
    m: Vec<Array1<f64>>,
    v: Vec<Array1<f64>>,
    t: usize,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl SvgdAdamState {
    fn new(n_particles: usize, dim: usize) -> Self {
        Self {
            m: vec![Array1::zeros(dim); n_particles],
            v: vec![Array1::zeros(dim); n_particles],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Compute Adam update for each particle's gradient
    fn update(&mut self, grads: &[Array1<f64>]) -> Vec<Array1<f64>> {
        self.t += 1;
        let n = grads.len();
        let mut directions = Vec::with_capacity(n);

        for i in 0..n {
            let dim = grads[i].len();
            let mut dir = Array1::zeros(dim);
            for j in 0..dim {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grads[i][j];
                self.v[i][j] =
                    self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grads[i][j] * grads[i][j];
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                dir[j] = m_hat / (v_hat.sqrt() + self.epsilon);
            }
            directions.push(dir);
        }

        directions
    }
}

// ============================================================================
// SVGD Configuration
// ============================================================================

/// Configuration for Stein Variational Gradient Descent
#[derive(Debug, Clone)]
pub struct SvgdConfig {
    /// Number of particles
    pub num_particles: usize,
    /// Step size (learning rate)
    pub step_size: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance on average update norm
    pub tolerance: f64,
    /// Kernel bandwidth; None = median heuristic (recommended)
    pub kernel_bandwidth: Option<f64>,
    /// Random seed for particle initialization
    pub seed: u64,
    /// Initial particle spread (std of initialization distribution)
    pub init_spread: f64,
    /// Whether to use Adam optimizer for adaptive step sizes
    pub use_adam: bool,
}

impl Default for SvgdConfig {
    fn default() -> Self {
        Self {
            num_particles: 100,
            step_size: 0.1,
            max_iterations: 1000,
            tolerance: 1e-4,
            kernel_bandwidth: None,
            seed: 42,
            init_spread: 1.0,
            use_adam: true,
        }
    }
}

// ============================================================================
// SVGD Struct
// ============================================================================

/// Stein Variational Gradient Descent
///
/// A particle-based method that maintains a set of particles {theta_i} and
/// iteratively transports them to approximate the target posterior.
///
/// # Example
/// ```no_run
/// use scirs2_stats::variational::{Svgd, SvgdConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let config = SvgdConfig {
///     num_particles: 50,
///     step_size: 0.1,
///     max_iterations: 500,
///     ..Default::default()
/// };
///
/// let mut svgd = Svgd::new(config);
/// ```
#[derive(Debug, Clone)]
pub struct Svgd {
    /// Configuration
    pub config: SvgdConfig,
    /// Kernel
    kernel: RbfKernel,
}

impl Svgd {
    /// Create a new SVGD instance
    pub fn new(config: SvgdConfig) -> Self {
        let kernel = RbfKernel {
            bandwidth: config.kernel_bandwidth,
        };
        Self { config, kernel }
    }

    /// Initialize particles using quasi-random sequences
    fn init_particles(&self, dim: usize) -> Vec<Array1<f64>> {
        let n = self.config.num_particles;
        let golden = 1.618033988749895_f64;
        let plastic = 1.324717957244746_f64;

        (0..n)
            .map(|i| {
                Array1::from_shape_fn(dim, |d| {
                    let seed = self.config.seed.wrapping_add(i as u64 * 1000 + d as u64);
                    let u1 = ((seed as f64 * golden + d as f64 * plastic) % 1.0).abs();
                    let u2 = ((seed as f64 * plastic + d as f64 * golden + 0.5) % 1.0).abs();
                    let u1 = u1.max(1e-10).min(1.0 - 1e-10);
                    let u2 = u2.max(1e-10).min(1.0 - 1e-10);
                    let r = (-2.0 * u1.ln()).sqrt();
                    r * (2.0 * PI * u2).cos() * self.config.init_spread
                })
            })
            .collect()
    }

    /// Compute the SVGD update direction for all particles.
    ///
    /// phi*(theta_i) = (1/n) sum_j [k(theta_j, theta_i) * grad log p(theta_j)
    ///                              + grad_theta_j k(theta_j, theta_i)]
    fn compute_phi_star<F>(
        &self,
        particles: &[Array1<f64>],
        log_joint: &F,
        bandwidth: f64,
    ) -> StatsResult<Vec<Array1<f64>>>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        let n = particles.len();
        let dim = particles[0].len();

        // Compute gradients for all particles
        let mut grad_log_p: Vec<Array1<f64>> = Vec::with_capacity(n);
        for particle in particles {
            let (_log_p, grad) = log_joint(particle)?;
            grad_log_p.push(grad);
        }

        // Compute phi* for each particle
        let mut phi_star: Vec<Array1<f64>> = vec![Array1::zeros(dim); n];

        for i in 0..n {
            for j in 0..n {
                let (k_val, grad_k_j) =
                    self.kernel
                        .eval_with_grad(&particles[j], &particles[i], bandwidth);

                // Attractive term: k(theta_j, theta_i) * grad log p(theta_j)
                for d in 0..dim {
                    phi_star[i][d] += k_val * grad_log_p[j][d];
                }

                // Repulsive term: grad_theta_j k(theta_j, theta_i)
                // Note: grad_k_j = d k(theta_j, theta_i) / d theta_j
                for d in 0..dim {
                    phi_star[i][d] += grad_k_j[d];
                }
            }

            // Average over particles
            phi_star[i] /= n as f64;
        }

        Ok(phi_star)
    }

    /// Compute a proxy ELBO estimate for monitoring convergence.
    /// Uses the kernel density estimate of the entropy plus the average log joint.
    fn estimate_elbo<F>(
        &self,
        particles: &[Array1<f64>],
        log_joint: &F,
        bandwidth: f64,
    ) -> StatsResult<f64>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        let n = particles.len();
        let dim = particles[0].len();

        // Average log p(theta_i)
        let mut avg_log_p = 0.0;
        for particle in particles {
            let (log_p, _) = log_joint(particle)?;
            avg_log_p += log_p;
        }
        avg_log_p /= n as f64;

        // Kernel density entropy estimate:
        // H approx -1/n sum_i log(1/n sum_j k(theta_i, theta_j))
        let mut entropy_est = 0.0;
        for i in 0..n {
            let mut kde_sum = 0.0;
            for j in 0..n {
                let diff = &particles[i] - &particles[j];
                let dist_sq = diff.dot(&diff);
                kde_sum += (-dist_sq / (2.0 * bandwidth * bandwidth)).exp();
            }
            let norm_const = (2.0 * PI * bandwidth * bandwidth).powf(dim as f64 / 2.0);
            let density = kde_sum / (n as f64 * norm_const);
            if density > 1e-300 {
                entropy_est -= density.ln();
            }
        }
        entropy_est /= n as f64;

        Ok(avg_log_p + entropy_est)
    }
}

impl VariationalInference for Svgd {
    fn fit<F>(&mut self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if self.config.num_particles < 2 {
            return Err(StatsError::InvalidArgument(
                "num_particles must be at least 2".to_string(),
            ));
        }
        if self.config.step_size <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "step_size must be positive".to_string(),
            ));
        }

        let n = self.config.num_particles;
        let mut particles = self.init_particles(dim);
        let mut elbo_history = Vec::with_capacity(self.config.max_iterations);
        let mut converged = false;

        let mut adam = if self.config.use_adam {
            Some(SvgdAdamState::new(n, dim))
        } else {
            None
        };

        for _iter in 0..self.config.max_iterations {
            // Determine bandwidth
            let bandwidth = self
                .config
                .kernel_bandwidth
                .unwrap_or_else(|| RbfKernel::median_bandwidth(&particles));

            // Compute SVGD update directions
            let phi_star = self.compute_phi_star(&particles, &log_joint, bandwidth)?;

            // Update particles
            let updates: Vec<Array1<f64>> = if let Some(ref mut adam_state) = adam {
                let directions = adam_state.update(&phi_star);
                directions
                    .into_iter()
                    .map(|d| &d * self.config.step_size)
                    .collect()
            } else {
                phi_star
                    .iter()
                    .map(|phi| phi * self.config.step_size)
                    .collect()
            };

            // Compute average update norm for convergence check
            let avg_update_norm: f64 =
                updates.iter().map(|u| u.dot(u).sqrt()).sum::<f64>() / n as f64;

            for i in 0..n {
                particles[i] = &particles[i] + &updates[i];
            }

            // Estimate ELBO periodically (every 10 iterations to save computation)
            if _iter % 10 == 0 || _iter == self.config.max_iterations - 1 {
                let elbo = self.estimate_elbo(&particles, &log_joint, bandwidth)?;
                elbo_history.push(elbo);
            }

            // Check convergence
            if avg_update_norm < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute posterior statistics from particles
        let mut mean = Array1::zeros(dim);
        for p in &particles {
            mean = &mean + p;
        }
        mean /= n as f64;

        let mut var = Array1::zeros(dim);
        for p in &particles {
            let diff = p - &mean;
            var = &var + &(&diff * &diff);
        }
        var /= (n - 1) as f64;
        let std_devs = var.mapv(f64::sqrt);

        Ok(PosteriorResult {
            means: mean,
            std_devs,
            elbo_history,
            iterations: self.config.max_iterations,
            converged,
            samples: Some(particles),
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: SVGD particles converge to a known unimodal Gaussian posterior
    #[test]
    fn test_svgd_gaussian_convergence() {
        let target_mean = 2.0_f64;
        let target_var = 0.5_f64;

        let config = SvgdConfig {
            num_particles: 50,
            step_size: 0.1,
            max_iterations: 500,
            tolerance: 1e-5,
            seed: 42,
            init_spread: 2.0,
            use_adam: true,
            ..Default::default()
        };

        let mut svgd = Svgd::new(config);
        let result = svgd
            .fit(
                move |theta: &Array1<f64>| {
                    let x = theta[0];
                    let log_p = -0.5 * (x - target_mean).powi(2) / target_var;
                    let grad = Array1::from_vec(vec![-(x - target_mean) / target_var]);
                    Ok((log_p, grad))
                },
                1,
            )
            .expect("SVGD should succeed");

        assert!(
            (result.means[0] - target_mean).abs() < 0.5,
            "Mean should be near {}, got {}",
            target_mean,
            result.means[0]
        );
        assert!(
            result.samples.is_some(),
            "SVGD should return posterior samples"
        );
    }

    /// Test: SVGD on a bimodal target — particles should spread across both modes
    #[ignore = "slow: measured ~107.5s (100 particles x 1000 iterations, O(particles^2) \
                kernel/phi_star updates per iteration, no early-exit) -- razor-thin under \
                the workspace's 120s default nextest hard-kill, so left ignored rather than \
                risking flaky CI timeouts; 0.6.5 ignore audit"]
    #[test]
    fn test_svgd_bimodal() {
        // Bimodal: 0.5 * N(-3, 0.5) + 0.5 * N(3, 0.5)
        let config = SvgdConfig {
            num_particles: 100,
            step_size: 0.05,
            max_iterations: 1000,
            tolerance: 1e-6,
            seed: 123,
            init_spread: 5.0,
            use_adam: true,
            ..Default::default()
        };

        let mut svgd = Svgd::new(config);
        let result = svgd
            .fit(
                |theta: &Array1<f64>| {
                    let x = theta[0];
                    let var = 0.5;
                    // log of mixture: log(0.5 * N(x; -3, 0.5) + 0.5 * N(x; 3, 0.5))
                    let log_comp1 = -0.5 * (x + 3.0).powi(2) / var;
                    let log_comp2 = -0.5 * (x - 3.0).powi(2) / var;
                    let max_log = log_comp1.max(log_comp2);
                    let log_p =
                        max_log + ((log_comp1 - max_log).exp() + (log_comp2 - max_log).exp()).ln();

                    // Gradient of log mixture
                    let w1 = (log_comp1 - max_log).exp();
                    let w2 = (log_comp2 - max_log).exp();
                    let total = w1 + w2;
                    let grad_x = (w1 * (-(x + 3.0) / var) + w2 * (-(x - 3.0) / var)) / total;
                    Ok((log_p, Array1::from_vec(vec![grad_x])))
                },
                1,
            )
            .expect("SVGD should succeed");

        let samples = result.samples.as_ref().expect("should have samples");

        // Check that particles exist in both modes
        let left_count = samples.iter().filter(|p| p[0] < 0.0).count();
        let right_count = samples.iter().filter(|p| p[0] >= 0.0).count();
        assert!(
            left_count > 5 && right_count > 5,
            "Particles should spread across both modes: left={}, right={}",
            left_count,
            right_count
        );
    }

    /// Test: Repulsive kernel prevents particle collapse — particles should
    /// not all collapse to the same point even for a peaked target
    #[test]
    fn test_svgd_repulsive_prevents_collapse() {
        let config = SvgdConfig {
            num_particles: 30,
            step_size: 0.05,
            max_iterations: 200,
            tolerance: 1e-8,
            seed: 77,
            init_spread: 2.0,
            use_adam: true,
            ..Default::default()
        };

        let mut svgd = Svgd::new(config);
        let result = svgd
            .fit(
                |theta: &Array1<f64>| {
                    // Very peaked Gaussian: N(0, 0.01)
                    let x = theta[0];
                    let var = 0.01;
                    let log_p = -0.5 * x * x / var;
                    let grad = Array1::from_vec(vec![-x / var]);
                    Ok((log_p, grad))
                },
                1,
            )
            .expect("SVGD should succeed");

        let samples = result.samples.as_ref().expect("should have samples");

        // Compute variance of particle positions
        let mean = result.means[0];
        let var: f64 =
            samples.iter().map(|p| (p[0] - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Particles should NOT all collapse to exactly the same point
        assert!(
            var > 1e-10,
            "Particle variance {} should be nonzero (repulsion prevents collapse)",
            var
        );
    }

    /// Test: SVGD 2D Gaussian — mean and std should be reasonable
    #[test]
    fn test_svgd_2d_gaussian() {
        let config = SvgdConfig {
            num_particles: 80,
            step_size: 0.1,
            max_iterations: 500,
            tolerance: 1e-5,
            seed: 55,
            init_spread: 3.0,
            use_adam: true,
            ..Default::default()
        };

        let mut svgd = Svgd::new(config);
        let result = svgd
            .fit(
                |theta: &Array1<f64>| {
                    // N([1, -1], I)
                    let d0 = theta[0] - 1.0;
                    let d1 = theta[1] + 1.0;
                    let log_p = -0.5 * (d0 * d0 + d1 * d1);
                    let grad = Array1::from_vec(vec![-d0, -d1]);
                    Ok((log_p, grad))
                },
                2,
            )
            .expect("SVGD should succeed");

        assert!(
            (result.means[0] - 1.0).abs() < 1.0,
            "Mean[0] should be near 1.0, got {}",
            result.means[0]
        );
        assert!(
            (result.means[1] - (-1.0)).abs() < 1.0,
            "Mean[1] should be near -1.0, got {}",
            result.means[1]
        );
    }

    /// Test: validation errors
    #[test]
    fn test_svgd_validation() {
        let mut svgd = Svgd::new(SvgdConfig {
            num_particles: 1, // too few
            ..Default::default()
        });
        let result = svgd.fit(|_: &Array1<f64>| Ok((0.0, Array1::zeros(1))), 1);
        assert!(result.is_err());
    }

    /// Test: median bandwidth heuristic produces reasonable values
    #[test]
    fn test_median_bandwidth() {
        let particles = vec![
            Array1::from_vec(vec![0.0]),
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![3.0]),
            Array1::from_vec(vec![4.0]),
        ];
        let h = RbfKernel::median_bandwidth(&particles);
        assert!(h > 0.0, "Bandwidth should be positive");
        assert!(h < 10.0, "Bandwidth should be reasonable, got {}", h);
    }
}
