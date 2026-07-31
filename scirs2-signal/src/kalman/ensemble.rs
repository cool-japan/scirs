//! Ensemble Kalman Filter (EnKF) for high-dimensional systems.
//!
//! The EnKF represents the probability distribution by an ensemble of state vectors
//! and updates each member based on perturbed observations. This Monte Carlo
//! approach avoids explicit matrix inversions of the full covariance and scales
//! well to high-dimensional state spaces.
//!
//! # Algorithm
//!
//! 1. **Predict**: propagate each ensemble member through the model, add process noise.
//! 2. **Update**: compute ensemble mean/covariance, apply Kalman-like update to each member.
//!
//! # References
//!
//! * Evensen, G. (1994). "Sequential data assimilation with a nonlinear quasi-geostrophic model
//!   using Monte Carlo methods to forecast error statistics".
//!   *Journal of Geophysical Research*, 99(C5), 10143–10162.
//! * Burgers, G., van Leeuwen, P.J. & Evensen, G. (1998). "Analysis scheme in the ensemble
//!   Kalman filter". *Monthly Weather Review*, 126(6), 1719–1724.

use super::matrix_utils::{
    mat_add, mat_eye, mat_inv, mat_mul, mat_scale, mat_transpose, mat_vec_mul, outer_product,
};
use crate::error::{SignalError, SignalResult};

/// Ensemble Kalman Filter for data assimilation.
///
/// Uses N ensemble members to represent the state distribution. Each member
/// is an independent sample from the prior, updated with perturbed observations.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::EnsembleKalmanFilter;
///
/// let mut enkf = EnsembleKalmanFilter::new(2, 1, 50);
/// enkf.initialize_from_prior(&[0.0, 1.0], 0.1).expect("operation should succeed");
///
/// let f = |x: &[f64]| -> Vec<f64> { vec![x[0] + x[1], x[1]] };
/// let h = |x: &[f64]| -> Vec<f64> { vec![x[0]] };
/// enkf.set_model(Box::new(f));
/// enkf.set_observation(Box::new(h));
/// enkf.set_R(vec![vec![0.1]]).expect("operation should succeed");
///
/// enkf.predict(0.01).expect("operation should succeed");
/// enkf.update(&[1.0]).expect("operation should succeed");
/// let mean = enkf.mean_state();
/// assert_eq!(mean.len(), 2);
/// ```
pub struct EnsembleKalmanFilter {
    /// Ensemble members: N vectors of length dim_x
    ensemble: Vec<Vec<f64>>,
    /// State dimension
    dim_x: usize,
    /// Measurement dimension
    dim_z: usize,
    /// Number of ensemble members
    n_ensemble: usize,
    /// Nonlinear model (or linear proxy) f(x)
    model: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Observation operator h(x)
    observation: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Measurement noise covariance R (dim_z × dim_z)
    r: Vec<Vec<f64>>,
    /// Random state for reproducible perturbations (LCG)
    rng_state: u64,
}

impl EnsembleKalmanFilter {
    /// Create a new Ensemble Kalman Filter.
    ///
    /// # Arguments
    ///
    /// * `dim_x`      - State dimension
    /// * `dim_z`      - Measurement dimension
    /// * `n_ensemble` - Number of ensemble members (typically 20–200)
    pub fn new(dim_x: usize, dim_z: usize, n_ensemble: usize) -> Self {
        let identity_model = {
            let d = dim_x;
            move |x: &[f64]| x[..d].to_vec()
        };
        let identity_obs = {
            let d = dim_z.min(dim_x);
            move |x: &[f64]| x[..d].to_vec()
        };
        EnsembleKalmanFilter {
            ensemble: vec![vec![0.0_f64; dim_x]; n_ensemble],
            dim_x,
            dim_z,
            n_ensemble,
            model: Box::new(identity_model),
            observation: Box::new(identity_obs),
            r: mat_eye(dim_z),
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    /// Set the propagation model f(x).
    pub fn set_model(&mut self, f: Box<dyn Fn(&[f64]) -> Vec<f64>>) {
        self.model = f;
    }

    /// Set the observation operator h(x).
    pub fn set_observation(&mut self, h: Box<dyn Fn(&[f64]) -> Vec<f64>>) {
        self.observation = h;
    }

    /// Set the measurement noise covariance R.
    #[allow(non_snake_case)]
    pub fn set_R(&mut self, r: Vec<Vec<f64>>) -> SignalResult<()> {
        if r.len() != self.dim_z || r.iter().any(|row| row.len() != self.dim_z) {
            return Err(SignalError::ValueError(
                "R must be dim_z × dim_z".to_string(),
            ));
        }
        self.r = r;
        Ok(())
    }

    /// Initialize ensemble from a Gaussian prior N(mean, sigma²I).
    ///
    /// # Arguments
    ///
    /// * `mean`  - Prior mean state vector
    /// * `sigma` - Standard deviation for each state component
    pub fn initialize_from_prior(&mut self, mean: &[f64], sigma: f64) -> SignalResult<()> {
        if mean.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "Mean length {} != dim_x {}",
                mean.len(),
                self.dim_x
            )));
        }
        // Pre-generate all random values to avoid double mutable borrow
        let random_values: Vec<Vec<f64>> = (0..self.n_ensemble)
            .map(|_| (0..self.dim_x).map(|_| self.lcg_normal()).collect())
            .collect();
        for (member, rand_vals) in self.ensemble.iter_mut().zip(random_values.iter()) {
            for ((j, m), rv) in member.iter_mut().zip(mean.iter()).zip(rand_vals.iter()) {
                *j = m + sigma * rv;
            }
        }
        Ok(())
    }

    /// Initialize ensemble directly from a matrix of ensemble members.
    pub fn set_ensemble(&mut self, ensemble: Vec<Vec<f64>>) -> SignalResult<()> {
        if ensemble.len() != self.n_ensemble {
            return Err(SignalError::ValueError(format!(
                "Ensemble size {} != n_ensemble {}",
                ensemble.len(),
                self.n_ensemble
            )));
        }
        for (i, member) in ensemble.iter().enumerate() {
            if member.len() != self.dim_x {
                return Err(SignalError::ValueError(format!(
                    "Ensemble member {} has length {} != dim_x {}",
                    i,
                    member.len(),
                    self.dim_x
                )));
            }
        }
        self.ensemble = ensemble;
        Ok(())
    }

    /// Predict step: propagate each ensemble member through f and add process noise.
    ///
    /// # Arguments
    ///
    /// * `noise_scale` - Standard deviation of additive process noise
    pub fn predict(&mut self, noise_scale: f64) -> SignalResult<()> {
        // Pre-generate noise for all members to avoid borrow conflict on self inside loop
        let noise_vecs: Vec<Vec<f64>> = (0..self.n_ensemble)
            .map(|_| {
                (0..self.dim_x)
                    .map(|_| noise_scale * self.lcg_normal())
                    .collect()
            })
            .collect();
        for (member, noise_vec) in self.ensemble.iter_mut().zip(noise_vecs.iter()) {
            // Propagate through model
            let propagated = (self.model)(member);
            // Add process noise
            for (j, val) in propagated.into_iter().enumerate() {
                member[j] = val + noise_vec[j];
            }
        }
        Ok(())
    }

    /// Update step using perturbed observations (Burgers et al., 1998).
    ///
    /// Each ensemble member is updated with a perturbed observation:
    /// ```text
    /// y_i   = z + v_i,   v_i ~ N(0, R)
    /// K     = P_e * H^T * (H * P_e * H^T + R)^{-1}
    /// x_i   = x_i^f + K * (y_i - h(x_i^f))
    /// ```
    /// where P_e is the ensemble covariance.
    ///
    /// # Arguments
    ///
    /// * `z` - Observation vector of length dim_z
    pub fn update(&mut self, z: &[f64]) -> SignalResult<()> {
        if z.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "Measurement length {} != dim_z {}",
                z.len(),
                self.dim_z
            )));
        }

        // Compute ensemble mean
        let mean = self.mean_state();

        // Compute H applied to each ensemble member
        let h_members: Vec<Vec<f64>> = self
            .ensemble
            .iter()
            .map(|m| (self.observation)(m))
            .collect();

        // Ensemble mean in observation space
        let mut hz_mean = vec![0.0_f64; self.dim_z];
        for hm in &h_members {
            for (j, &v) in hm.iter().enumerate() {
                hz_mean[j] += v / self.n_ensemble as f64;
            }
        }

        // Innovation covariance: S = (1/(N-1)) * sum dz_i*dz_i^T + R
        // where dz_i = H*x_i - H*x_bar
        let n = self.n_ensemble as f64;
        let mut s = mat_scale(&self.r, 1.0); // S = R
        for hm in &h_members {
            let dz: Vec<f64> = hm.iter().zip(hz_mean.iter()).map(|(a, b)| a - b).collect();
            let outer = outer_product(&dz, &dz);
            for row in 0..self.dim_z {
                for col in 0..self.dim_z {
                    s[row][col] += outer[row][col] / (n - 1.0);
                }
            }
        }

        // Cross-covariance: T = (1/(N-1)) * sum dx_i * dz_i^T  (dim_x × dim_z)
        let mut t_cross = vec![vec![0.0_f64; self.dim_z]; self.dim_x];
        for (member, hm) in self.ensemble.iter().zip(h_members.iter()) {
            let dx: Vec<f64> = member.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();
            let dz: Vec<f64> = hm.iter().zip(hz_mean.iter()).map(|(a, b)| a - b).collect();
            for row in 0..self.dim_x {
                for col in 0..self.dim_z {
                    t_cross[row][col] += dx[row] * dz[col] / (n - 1.0);
                }
            }
        }

        // Kalman gain K = T * S^{-1}
        let s_inv = mat_inv(&s)?;
        let k = mat_mul(&t_cross, &s_inv)?;

        // Update each ensemble member with perturbed observation
        // Generate all perturbations using Cholesky of R
        let r_chol = super::matrix_utils::cholesky_decomp(&self.r)?;

        // Pre-generate all perturbation noise vectors before mutably iterating over ensemble
        let perturbed_observations: Vec<Vec<f64>> = (0..self.n_ensemble)
            .map(|_| {
                let noise: Vec<f64> = (0..self.dim_z).map(|_| self.lcg_normal()).collect();
                let perturbed_noise = mat_vec_mul(&r_chol, &noise)
                    .unwrap_or_else(|_| noise.iter().map(|&v| v * 0.01).collect());
                z.iter()
                    .zip(perturbed_noise.iter())
                    .map(|(zi, vi)| zi + vi)
                    .collect()
            })
            .collect();

        let h_members_clone = h_members.clone();
        let ensemble_snapshot: Vec<Vec<f64>> = self.ensemble.clone();

        for (i, member) in self.ensemble.iter_mut().enumerate() {
            // Innovation y_i - h(x_i)
            let innovation: Vec<f64> = perturbed_observations[i]
                .iter()
                .zip(h_members_clone[i].iter())
                .map(|(y, hx)| y - hx)
                .collect();

            // Update: x_i = x_i + K * innovation
            let k_innov = mat_vec_mul(&k, &innovation).unwrap_or_else(|_| vec![0.0; self.dim_x]);
            for j in 0..self.dim_x {
                member[j] = ensemble_snapshot[i][j] + k_innov[j];
            }
        }

        Ok(())
    }

    /// Compute the ensemble mean state.
    pub fn mean_state(&self) -> Vec<f64> {
        let n = self.n_ensemble as f64;
        let mut mean = vec![0.0_f64; self.dim_x];
        for member in &self.ensemble {
            for (j, &v) in member.iter().enumerate() {
                mean[j] += v / n;
            }
        }
        mean
    }

    /// Compute the sample ensemble covariance matrix.
    pub fn ensemble_covariance(&self) -> Vec<Vec<f64>> {
        let mean = self.mean_state();
        self.ensemble_covariance_internal(&mean)
    }

    fn ensemble_covariance_internal(&self, mean: &[f64]) -> Vec<Vec<f64>> {
        let n = (self.n_ensemble - 1) as f64;
        let mut cov = vec![vec![0.0_f64; self.dim_x]; self.dim_x];
        for member in &self.ensemble {
            let dx: Vec<f64> = member.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();
            let outer = outer_product(&dx, &dx);
            for row in 0..self.dim_x {
                for col in 0..self.dim_x {
                    cov[row][col] += outer[row][col] / n;
                }
            }
        }
        cov
    }

    /// Access the raw ensemble members.
    pub fn ensemble(&self) -> &[Vec<f64>] {
        &self.ensemble
    }

    /// Return the number of ensemble members.
    pub fn n_ensemble(&self) -> usize {
        self.n_ensemble
    }

    /// Simple LCG-based Box-Muller normal random number generator
    /// (avoids external RNG dependencies, for reproducibility).
    fn lcg_normal(&mut self) -> f64 {
        // LCG: Knuth parameters
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (self.rng_state >> 33) as f64 / (u32::MAX as f64 + 1.0);
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (self.rng_state >> 33) as f64 / (u32::MAX as f64 + 1.0);
        let u1 = u1.max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test ensemble Kalman filter on a linear 1D data assimilation problem.
    #[test]
    fn test_enkf_1d_tracking() {
        let mut enkf = EnsembleKalmanFilter::new(1, 1, 100);
        enkf.initialize_from_prior(&[0.0], 1.0).expect("init prior");
        enkf.set_model(Box::new(|x: &[f64]| vec![x[0]]));
        enkf.set_observation(Box::new(|x: &[f64]| vec![x[0]]));
        enkf.set_R(vec![vec![0.1]]).expect("set R");

        // True state = 5.0, observe it for 50 steps
        let true_state = 5.0_f64;
        for step in 0..50 {
            enkf.predict(0.01).expect("predict");
            let noise = (step as f64 * 1.23456).sin() * 0.3;
            enkf.update(&[true_state + noise]).expect("update");
        }

        let mean = enkf.mean_state();
        assert_eq!(mean.len(), 1);
        assert!(
            (mean[0] - true_state).abs() < 1.0,
            "EnKF mean {:.3} should be near true state {:.3}",
            mean[0],
            true_state
        );
    }

    #[test]
    fn test_enkf_covariance_shrinks() {
        let mut enkf = EnsembleKalmanFilter::new(2, 2, 50);
        enkf.initialize_from_prior(&[0.0, 0.0], 5.0)
            .expect("init prior");
        enkf.set_model(Box::new(|x: &[f64]| x.to_vec()));
        enkf.set_observation(Box::new(|x: &[f64]| x.to_vec()));
        enkf.set_R(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set R");

        let init_cov = enkf.ensemble_covariance();
        let init_trace = init_cov[0][0] + init_cov[1][1];

        // Feed many observations
        for _ in 0..50 {
            enkf.predict(0.001).expect("predict");
            enkf.update(&[1.0, 2.0]).expect("update");
        }

        let final_cov = enkf.ensemble_covariance();
        let final_trace = final_cov[0][0] + final_cov[1][1];

        assert!(
            final_trace < init_trace,
            "Covariance trace should decrease: {:.4} -> {:.4}",
            init_trace,
            final_trace
        );
    }

    #[test]
    fn test_enkf_nonlinear_update() {
        // Nonlinear observation: observe x^2
        let mut enkf = EnsembleKalmanFilter::new(1, 1, 200);
        enkf.initialize_from_prior(&[2.0], 0.5).expect("init prior");
        enkf.set_model(Box::new(|x: &[f64]| x.to_vec()));
        enkf.set_observation(Box::new(|x: &[f64]| vec![x[0] * x[0]]));
        enkf.set_R(vec![vec![0.1]]).expect("set R");

        // True state = 3.0, observe 9.0 for several steps
        for _ in 0..20 {
            enkf.predict(0.01).expect("predict");
            enkf.update(&[9.0]).expect("update");
        }

        let mean = enkf.mean_state();
        // Mean should shift toward sqrt(9) = 3.0
        assert!(
            mean[0] > 1.5,
            "EnKF mean {:.3} should have shifted toward true state 3.0",
            mean[0]
        );
    }

    #[test]
    fn test_ensemble_initialization() {
        let mut enkf = EnsembleKalmanFilter::new(3, 1, 50);
        enkf.initialize_from_prior(&[1.0, 2.0, 3.0], 0.1)
            .expect("init prior");
        let mean = enkf.mean_state();
        // Mean should be close to prior mean
        assert!((mean[0] - 1.0).abs() < 0.5, "Mean[0] ≈ 1.0");
        assert!((mean[1] - 2.0).abs() < 0.5, "Mean[1] ≈ 2.0");
        assert!((mean[2] - 3.0).abs() < 0.5, "Mean[2] ≈ 3.0");
    }
}
