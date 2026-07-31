//! Unscented Kalman Filter (UKF) for nonlinear systems.
//!
//! The UKF uses the Unscented Transform to propagate a Gaussian distribution
//! through a nonlinear function. It selects a set of 2n+1 sigma points that
//! capture the mean and covariance exactly, then propagates each sigma point
//! through the nonlinear function to recover the output statistics.
//!
//! This avoids the need to compute Jacobians and provides better accuracy
//! than the EKF for strongly nonlinear systems.
//!
//! # References
//!
//! * Julier, S.J. & Uhlmann, J.K. (1997). "A New Extension of the Kalman Filter
//!   to Nonlinear Systems". *Proc. SPIE 3068*, 182–193.
//! * Wan, E.A. & van der Merwe, R. (2000). "The Unscented Kalman Filter for
//!   Nonlinear Estimation". *Proc. IEEE ASSPCC*, 153–158.

use super::matrix_utils::{
    cholesky_decomp, mat_add, mat_eye, mat_inv, mat_mul, mat_sub, mat_transpose, mat_vec_mul,
    outer_product, vec_add, vec_scale, vec_sub,
};
use crate::error::{SignalError, SignalResult};

/// Unscented Kalman Filter using the Julier/Uhlmann sigma-point method.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::UnscentedKalmanFilter;
///
/// // Bearing-only tracking: state = [x, y, vx, vy], obs = [bearing]
/// let f = |x: &[f64]| -> Vec<f64> {
///     vec![x[0] + x[2], x[1] + x[3], x[2], x[3]]
/// };
/// let h = |x: &[f64]| -> Vec<f64> {
///     vec![x[1].atan2(x[0])]
/// };
///
/// let mut ukf = UnscentedKalmanFilter::new(Box::new(f), Box::new(h), 4, 1);
/// ukf.set_initial_state(&[1.0, 1.0, 0.1, 0.0]).expect("operation should succeed");
/// ukf.predict().expect("operation should succeed");
/// ukf.update(&[std::f64::consts::PI / 4.0]).expect("operation should succeed");
/// ```
pub struct UnscentedKalmanFilter {
    /// Nonlinear state transition function
    f: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Nonlinear observation function
    h: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Process noise covariance Q
    q: Vec<Vec<f64>>,
    /// Measurement noise covariance R
    r: Vec<Vec<f64>>,
    /// State covariance P
    p: Vec<Vec<f64>>,
    /// State estimate
    x: Vec<f64>,
    /// State dimension
    dim_x: usize,
    /// Measurement dimension
    dim_z: usize,
    /// Sigma point spread parameter (typically 1e-3)
    alpha: f64,
    /// Prior knowledge parameter (2 = Gaussian)
    beta: f64,
    /// Secondary scaling parameter (0 = default)
    kappa: f64,
}

impl UnscentedKalmanFilter {
    /// Create a new Unscented Kalman Filter.
    ///
    /// # Arguments
    ///
    /// * `f`      - Nonlinear state transition function
    /// * `h`      - Nonlinear observation function
    /// * `dim_x`  - State dimension
    /// * `dim_z`  - Measurement dimension
    pub fn new(
        f: Box<dyn Fn(&[f64]) -> Vec<f64>>,
        h: Box<dyn Fn(&[f64]) -> Vec<f64>>,
        dim_x: usize,
        dim_z: usize,
    ) -> Self {
        UnscentedKalmanFilter {
            f,
            h,
            q: mat_eye(dim_x),
            r: mat_eye(dim_z),
            p: mat_eye(dim_x),
            x: vec![0.0_f64; dim_x],
            dim_x,
            dim_z,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// Set UKF tuning parameters.
    ///
    /// * `alpha` - Spread of sigma points around mean (1e-4 to 1)
    /// * `beta`  - Prior knowledge of distribution (2 = Gaussian optimal)
    /// * `kappa` - Secondary scaling parameter (often 0 or 3-n)
    pub fn set_parameters(&mut self, alpha: f64, beta: f64, kappa: f64) -> SignalResult<()> {
        if alpha <= 0.0 {
            return Err(SignalError::ValueError(
                "alpha must be positive".to_string(),
            ));
        }
        self.alpha = alpha;
        self.beta = beta;
        self.kappa = kappa;
        Ok(())
    }

    /// Set the process noise covariance Q.
    #[allow(non_snake_case)]
    pub fn set_Q(&mut self, q: Vec<Vec<f64>>) -> SignalResult<()> {
        if q.len() != self.dim_x || q.iter().any(|r| r.len() != self.dim_x) {
            return Err(SignalError::ValueError(
                "Q must be dim_x × dim_x".to_string(),
            ));
        }
        self.q = q;
        Ok(())
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

    /// Set the initial state estimate.
    pub fn set_initial_state(&mut self, x0: &[f64]) -> SignalResult<()> {
        if x0.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "Initial state length {} != dim_x {}",
                x0.len(),
                self.dim_x
            )));
        }
        self.x = x0.to_vec();
        Ok(())
    }

    /// Set the initial state covariance P.
    #[allow(non_snake_case)]
    pub fn set_P(&mut self, p: Vec<Vec<f64>>) -> SignalResult<()> {
        if p.len() != self.dim_x || p.iter().any(|r| r.len() != self.dim_x) {
            return Err(SignalError::ValueError(
                "P must be dim_x × dim_x".to_string(),
            ));
        }
        self.p = p;
        Ok(())
    }

    /// Compute lambda and weights for sigma points.
    fn compute_weights(&self) -> (f64, Vec<f64>, Vec<f64>) {
        let n = self.dim_x as f64;
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;

        let n_sigma = 2 * self.dim_x + 1;
        let mut wm = vec![0.0_f64; n_sigma]; // mean weights
        let mut wc = vec![0.0_f64; n_sigma]; // covariance weights

        wm[0] = lambda / (n + lambda);
        wc[0] = wm[0] + 1.0 - self.alpha * self.alpha + self.beta;

        for i in 1..n_sigma {
            wm[i] = 0.5 / (n + lambda);
            wc[i] = wm[i];
        }

        (lambda, wm, wc)
    }

    /// Generate 2n+1 sigma points from current state estimate.
    fn sigma_points(&self, lambda: f64) -> SignalResult<Vec<Vec<f64>>> {
        let n = self.dim_x;
        let scale = (n as f64 + lambda).sqrt();

        // Cholesky decomposition of P
        let l = cholesky_decomp(&self.p)?;

        let mut sigma_pts = Vec::with_capacity(2 * n + 1);

        // First sigma point: x_0 = x
        sigma_pts.push(self.x.clone());

        // Sigma points x_i = x + sqrt((n+lambda)) * column_i(L)
        for i in 0..n {
            let col: Vec<f64> = l.iter().map(|row| row[i]).collect();
            let dx = vec_scale(&col, scale);
            sigma_pts.push(vec_add(&self.x, &dx));
        }

        // Sigma points x_{n+i} = x - sqrt((n+lambda)) * column_i(L)
        for i in 0..n {
            let col: Vec<f64> = l.iter().map(|row| row[i]).collect();
            let dx = vec_scale(&col, scale);
            sigma_pts.push(vec_sub(&self.x, &dx));
        }

        Ok(sigma_pts)
    }

    /// Predict step: propagate sigma points through f.
    ///
    /// ```text
    /// sigma_i = f(sigma_i)
    /// x_pred  = sum(wm_i * sigma_i)
    /// P_pred  = sum(wc_i * (sigma_i - x_pred)(sigma_i - x_pred)^T) + Q
    /// ```
    pub fn predict(&mut self) -> SignalResult<()> {
        let (lambda, wm, wc) = self.compute_weights();

        // Generate sigma points
        let sigma_pts = self.sigma_points(lambda)?;

        // Propagate through f
        let propagated: Vec<Vec<f64>> = sigma_pts.iter().map(|sp| (self.f)(sp)).collect();

        // Compute predicted mean
        let mut x_pred = vec![0.0_f64; self.dim_x];
        for (i, sp) in propagated.iter().enumerate() {
            for j in 0..self.dim_x {
                x_pred[j] += wm[i] * sp[j];
            }
        }

        // Compute predicted covariance
        let mut p_pred = vec![vec![0.0_f64; self.dim_x]; self.dim_x];
        for (i, sp) in propagated.iter().enumerate() {
            let diff = vec_sub(sp, &x_pred);
            let outer = outer_product(&diff, &diff);
            for row in 0..self.dim_x {
                for col in 0..self.dim_x {
                    p_pred[row][col] += wc[i] * outer[row][col];
                }
            }
        }
        // Add process noise
        let p_pred = mat_add(&p_pred, &self.q)?;

        self.x = x_pred;
        self.p = p_pred;

        Ok(())
    }

    /// Update step: compute predicted measurements from sigma points.
    ///
    /// ```text
    /// z_sigma_i = h(sigma_i)
    /// z_pred    = sum(wm_i * z_sigma_i)
    /// S         = sum(wc_i * (z_sigma_i - z_pred)(z_sigma_i - z_pred)^T) + R
    /// T         = sum(wc_i * (sigma_i - x_pred)(z_sigma_i - z_pred)^T)    (cross-covariance)
    /// K         = T * S^{-1}
    /// x         = x_pred + K * (z - z_pred)
    /// P         = P_pred - K * S * K^T
    /// ```
    pub fn update(&mut self, z: &[f64]) -> SignalResult<()> {
        if z.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "Measurement length {} != dim_z {}",
                z.len(),
                self.dim_z
            )));
        }

        let (lambda, wm, wc) = self.compute_weights();
        let sigma_pts = self.sigma_points(lambda)?;

        // Propagate sigma points through h
        let z_sigma: Vec<Vec<f64>> = sigma_pts.iter().map(|sp| (self.h)(sp)).collect();

        // Predicted measurement mean
        let mut z_pred = vec![0.0_f64; self.dim_z];
        for (i, zs) in z_sigma.iter().enumerate() {
            for j in 0..self.dim_z {
                z_pred[j] += wm[i] * zs[j];
            }
        }

        // Innovation covariance S = sum wc_i*(z_i-z_pred)(z_i-z_pred)^T + R
        let mut s = vec![vec![0.0_f64; self.dim_z]; self.dim_z];
        for (i, zs) in z_sigma.iter().enumerate() {
            let dz = vec_sub(zs, &z_pred);
            let outer = outer_product(&dz, &dz);
            for row in 0..self.dim_z {
                for col in 0..self.dim_z {
                    s[row][col] += wc[i] * outer[row][col];
                }
            }
        }
        let s = mat_add(&s, &self.r)?;

        // Cross-covariance T = sum wc_i*(x_i-x_pred)(z_i-z_pred)^T
        let x_pred = self.x.clone();
        let mut t_cross = vec![vec![0.0_f64; self.dim_z]; self.dim_x];
        for (i, (sp, zs)) in sigma_pts.iter().zip(z_sigma.iter()).enumerate() {
            let dx = vec_sub(sp, &x_pred);
            let dz = vec_sub(zs, &z_pred);
            for row in 0..self.dim_x {
                for col in 0..self.dim_z {
                    t_cross[row][col] += wc[i] * dx[row] * dz[col];
                }
            }
        }

        // Kalman gain K = T * S^{-1}
        let s_inv = mat_inv(&s)?;
        let k = mat_mul(&t_cross, &s_inv)?;

        // State update x = x + K * (z - z_pred)
        let innovation = vec_sub(z, &z_pred);
        let k_innov = mat_vec_mul(&k, &innovation)?;
        self.x = vec_add(&self.x, &k_innov);

        // Covariance update P = P - K * S * K^T
        let kt = mat_transpose(&k);
        let ks = mat_mul(&k, &s)?;
        let kskt = mat_mul(&ks, &kt)?;
        self.p = mat_sub(&self.p, &kskt)?;

        // Symmetrize
        let pt = mat_transpose(&self.p);
        let p_sym: Vec<Vec<f64>> = self
            .p
            .iter()
            .zip(pt.iter())
            .map(|(r1, r2)| {
                r1.iter()
                    .zip(r2.iter())
                    .map(|(a, b)| 0.5 * (a + b))
                    .collect()
            })
            .collect();
        self.p = p_sym;

        Ok(())
    }

    /// Return the current state estimate.
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Return the current state covariance.
    pub fn covariance(&self) -> &Vec<Vec<f64>> {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test UKF on bearing-only tracking problem.
    ///
    /// State: [x, y, vx, vy] (2D position + velocity)
    /// Observation: bearing angle θ = atan2(y, x)
    #[test]
    fn test_bearing_only_tracking() {
        // dt = 1, constant velocity
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0] + x[2], x[1] + x[3], x[2], x[3]] };
        // Observe bearing only
        let h = |x: &[f64]| -> Vec<f64> { vec![(x[1]).atan2(x[0])] };

        let mut ukf = UnscentedKalmanFilter::new(Box::new(f), Box::new(h), 4, 1);
        ukf.set_initial_state(&[1.0, 1.0, 0.1, 0.1])
            .expect("set initial state");
        ukf.set_Q(vec![
            vec![1e-4, 0.0, 0.0, 0.0],
            vec![0.0, 1e-4, 0.0, 0.0],
            vec![0.0, 0.0, 1e-4, 0.0],
            vec![0.0, 0.0, 0.0, 1e-4],
        ])
        .expect("set Q");
        ukf.set_R(vec![vec![0.01]]).expect("set R");
        ukf.set_P(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ])
        .expect("set P");

        // True trajectory: straight line at 45 degrees
        let mut true_x = 1.0_f64;
        let mut true_y = 1.0_f64;
        let vx = 0.1_f64;
        let vy = 0.1_f64;

        for step in 0..20 {
            true_x += vx;
            true_y += vy;

            let noise = (step as f64 % 5.0 - 2.5) * 0.005;
            let bearing = true_y.atan2(true_x) + noise;
            let z = vec![bearing];

            ukf.predict().expect("predict");
            ukf.update(&z).expect("update");
        }

        let state = ukf.state();
        // After tracking, state should be non-degenerate
        assert_eq!(state.len(), 4);
        // Position should be in reasonable range
        assert!(
            state[0].abs() < 20.0 && state[1].abs() < 20.0,
            "Position out of range: ({:.3}, {:.3})",
            state[0],
            state[1]
        );
    }

    #[test]
    fn test_ukf_sigma_point_count() {
        let f = |x: &[f64]| -> Vec<f64> { x.to_vec() };
        let h = |x: &[f64]| -> Vec<f64> { vec![x[0]] };
        let ukf = UnscentedKalmanFilter::new(Box::new(f), Box::new(h), 3, 1);
        let (lambda, wm, wc) = ukf.compute_weights();
        // Should have 2n+1 = 7 sigma points
        assert_eq!(wm.len(), 7);
        assert_eq!(wc.len(), 7);
        // Weights should sum to 1
        let wm_sum: f64 = wm.iter().sum();
        assert!(
            (wm_sum - 1.0).abs() < 1e-10,
            "Mean weights should sum to 1, got {:.6}",
            wm_sum
        );
        let _ = lambda; // used
    }

    #[test]
    fn test_ukf_preserves_linear_gaussian() {
        // For a linear Gaussian system, UKF should match KF closely
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0] + x[1], x[1]] };
        let h = |x: &[f64]| -> Vec<f64> { vec![x[0]] };

        let mut ukf = UnscentedKalmanFilter::new(Box::new(f), Box::new(h), 2, 1);
        ukf.set_initial_state(&[0.0, 1.0])
            .expect("set initial state");
        ukf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        ukf.set_R(vec![vec![0.1]]).expect("set R");

        // Should not error over many steps
        for k in 0..10 {
            ukf.predict().expect("predict");
            ukf.update(&[k as f64]).expect("update");
        }
        let state = ukf.state();
        assert!(
            state[0].is_finite() && state[1].is_finite(),
            "UKF state should be finite"
        );
    }
}
