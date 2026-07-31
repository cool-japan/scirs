//! Standard (linear) Kalman filter implementation.
//!
//! The Kalman filter is an optimal recursive linear estimator for linear
//! Gaussian systems. It maintains a state estimate and its covariance,
//! performing predict and update steps alternately.
//!
//! # System Model
//!
//! ```text
//! x_k = F * x_{k-1} + w_k,  w_k ~ N(0, Q)
//! z_k = H * x_k    + v_k,  v_k ~ N(0, R)
//! ```
//!
//! # References
//!
//! * Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems".
//!   *Journal of Basic Engineering*, 82(1), 35–45.

use super::matrix_utils::{
    dot, mat_add, mat_eye, mat_inv, mat_mul, mat_sub, mat_transpose, mat_vec_mul, outer_product,
    vec_add, vec_sub,
};
use crate::error::{SignalError, SignalResult};

/// Standard Kalman filter for linear Gaussian systems.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::KalmanFilter;
///
/// // 2-state (position, velocity), 1 measurement
/// let mut kf = KalmanFilter::new(2, 1);
/// // Constant-velocity transition
/// kf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]]).expect("operation should succeed");
/// // Observe only position
/// kf.set_H(vec![vec![1.0, 0.0]]).expect("operation should succeed");
/// kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]]).expect("operation should succeed");
/// kf.set_R(vec![vec![0.1]]).expect("operation should succeed");
/// kf.set_initial_state(&[0.0, 1.0]).expect("operation should succeed");
///
/// kf.predict().expect("operation should succeed");
/// kf.update(&[1.05]).expect("operation should succeed");
/// let state = kf.state();
/// assert_eq!(state.len(), 2);
/// ```
pub struct KalmanFilter {
    /// State dimension
    dim_x: usize,
    /// Measurement dimension
    dim_z: usize,
    /// State transition matrix (dim_x × dim_x)
    f: Vec<Vec<f64>>,
    /// Observation matrix (dim_z × dim_x)
    h: Vec<Vec<f64>>,
    /// Process noise covariance (dim_x × dim_x)
    q: Vec<Vec<f64>>,
    /// Measurement noise covariance (dim_z × dim_z)
    r: Vec<Vec<f64>>,
    /// State estimate covariance (dim_x × dim_x)
    p: Vec<Vec<f64>>,
    /// State estimate vector (dim_x)
    x: Vec<f64>,
    /// Whether a valid state estimate exists
    initialized: bool,
}

impl KalmanFilter {
    /// Create a new Kalman filter with default identity matrices.
    ///
    /// # Arguments
    ///
    /// * `dim_x` - Dimension of the state vector
    /// * `dim_z` - Dimension of the measurement vector
    ///
    /// # Panics
    ///
    /// Never panics; use `new` always returns `Ok`.
    pub fn new(dim_x: usize, dim_z: usize) -> Self {
        KalmanFilter {
            dim_x,
            dim_z,
            f: mat_eye(dim_x),
            h: {
                // Default H: first dim_z rows of identity-like matrix
                let rows = dim_z.min(dim_x);
                let mut h = vec![vec![0.0_f64; dim_x]; dim_z];
                for i in 0..rows {
                    h[i][i] = 1.0;
                }
                h
            },
            q: mat_eye(dim_x),
            r: mat_eye(dim_z),
            p: mat_eye(dim_x),
            x: vec![0.0_f64; dim_x],
            initialized: false,
        }
    }

    /// Set the state transition matrix F (dim_x × dim_x).
    #[allow(non_snake_case)]
    pub fn set_F(&mut self, f: Vec<Vec<f64>>) -> SignalResult<()> {
        if f.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "F must be {}×{}, got {}×?",
                self.dim_x,
                self.dim_x,
                f.len()
            )));
        }
        for (i, row) in f.iter().enumerate() {
            if row.len() != self.dim_x {
                return Err(SignalError::ValueError(format!(
                    "F row {} must have {} columns",
                    i, self.dim_x
                )));
            }
        }
        self.f = f;
        Ok(())
    }

    /// Set the observation matrix H (dim_z × dim_x).
    #[allow(non_snake_case)]
    pub fn set_H(&mut self, h: Vec<Vec<f64>>) -> SignalResult<()> {
        if h.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "H must have {} rows, got {}",
                self.dim_z,
                h.len()
            )));
        }
        for (i, row) in h.iter().enumerate() {
            if row.len() != self.dim_x {
                return Err(SignalError::ValueError(format!(
                    "H row {} must have {} columns",
                    i, self.dim_x
                )));
            }
        }
        self.h = h;
        Ok(())
    }

    /// Set the process noise covariance matrix Q (dim_x × dim_x).
    #[allow(non_snake_case)]
    pub fn set_Q(&mut self, q: Vec<Vec<f64>>) -> SignalResult<()> {
        if q.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "Q must be {}×{}, got {}×?",
                self.dim_x,
                self.dim_x,
                q.len()
            )));
        }
        for row in &q {
            if row.len() != self.dim_x {
                return Err(SignalError::ValueError(
                    "Q rows must match dim_x".to_string(),
                ));
            }
        }
        self.q = q;
        Ok(())
    }

    /// Set the measurement noise covariance matrix R (dim_z × dim_z).
    #[allow(non_snake_case)]
    pub fn set_R(&mut self, r: Vec<Vec<f64>>) -> SignalResult<()> {
        if r.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "R must be {}×{}, got {}×?",
                self.dim_z,
                self.dim_z,
                r.len()
            )));
        }
        for row in &r {
            if row.len() != self.dim_z {
                return Err(SignalError::ValueError(
                    "R rows must match dim_z".to_string(),
                ));
            }
        }
        self.r = r;
        Ok(())
    }

    /// Set the initial state estimate.
    pub fn set_initial_state(&mut self, x0: &[f64]) -> SignalResult<()> {
        if x0.len() != self.dim_x {
            return Err(SignalError::ValueError(format!(
                "Initial state must have {} elements, got {}",
                self.dim_x,
                x0.len()
            )));
        }
        self.x = x0.to_vec();
        self.initialized = true;
        Ok(())
    }

    /// Set the initial state covariance matrix P (dim_x × dim_x).
    #[allow(non_snake_case)]
    pub fn set_P(&mut self, p: Vec<Vec<f64>>) -> SignalResult<()> {
        if p.len() != self.dim_x {
            return Err(SignalError::ValueError(
                "P must be dim_x × dim_x".to_string(),
            ));
        }
        for row in &p {
            if row.len() != self.dim_x {
                return Err(SignalError::ValueError(
                    "P rows must match dim_x".to_string(),
                ));
            }
        }
        self.p = p;
        Ok(())
    }

    /// Perform the prediction step:
    ///
    /// ```text
    /// x_k|k-1 = F * x_{k-1|k-1}
    /// P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
    /// ```
    pub fn predict(&mut self) -> SignalResult<()> {
        // x = F * x
        let x_pred = mat_vec_mul(&self.f, &self.x)?;
        self.x = x_pred;

        // P = F * P * F^T + Q
        let fp = mat_mul(&self.f, &self.p)?;
        let ft = mat_transpose(&self.f);
        let fpft = mat_mul(&fp, &ft)?;
        self.p = mat_add(&fpft, &self.q)?;

        Ok(())
    }

    /// Perform the update (measurement) step:
    ///
    /// ```text
    /// y   = z - H * x_{k|k-1}          (innovation)
    /// S   = H * P_{k|k-1} * H^T + R    (innovation covariance)
    /// K   = P_{k|k-1} * H^T * S^{-1}   (Kalman gain)
    /// x   = x_{k|k-1} + K * y
    /// P   = (I - K * H) * P_{k|k-1}
    /// ```
    ///
    /// # Arguments
    ///
    /// * `z` - Measurement vector of length `dim_z`
    pub fn update(&mut self, z: &[f64]) -> SignalResult<()> {
        if z.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "Measurement must have {} elements, got {}",
                self.dim_z,
                z.len()
            )));
        }

        // Innovation y = z - H * x
        let hx = mat_vec_mul(&self.h, &self.x)?;
        let y = vec_sub(z, &hx);

        // Innovation covariance S = H * P * H^T + R
        let hp = mat_mul(&self.h, &self.p)?;
        let ht = mat_transpose(&self.h);
        let hpht = mat_mul(&hp, &ht)?;
        let s = mat_add(&hpht, &self.r)?;

        // Kalman gain K = P * H^T * S^{-1}
        let s_inv = mat_inv(&s)?;
        let pht = mat_mul(&self.p, &ht)?;
        let k = mat_mul(&pht, &s_inv)?;

        // Updated state x = x + K * y
        let ky = mat_vec_mul(&k, &y)?;
        self.x = vec_add(&self.x, &ky);

        // Updated covariance P = (I - K * H) * P
        let kh = mat_mul(&k, &self.h)?;
        let i_minus_kh = mat_sub(&mat_eye(self.dim_x), &kh)?;
        self.p = mat_mul(&i_minus_kh, &self.p)?;

        // Symmetrize P to prevent numerical drift
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

        self.initialized = true;
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

    /// Return the state dimension.
    pub fn dim_x(&self) -> usize {
        self.dim_x
    }

    /// Return the measurement dimension.
    pub fn dim_z(&self) -> usize {
        self.dim_z
    }

    /// Process a sequence of measurements and return the state history.
    ///
    /// For each measurement, performs predict followed by update.
    ///
    /// # Arguments
    ///
    /// * `measurements` - Sequence of measurement vectors
    ///
    /// # Returns
    ///
    /// Vector of state estimates, one per measurement.
    pub fn filter_sequence(&mut self, measurements: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
        let mut states = Vec::with_capacity(measurements.len());
        for z in measurements {
            self.predict()?;
            self.update(z)?;
            states.push(self.x.clone());
        }
        Ok(states)
    }

    /// Compute the log-likelihood of a measurement given current state.
    pub fn log_likelihood(&self, z: &[f64]) -> SignalResult<f64> {
        let hx = mat_vec_mul(&self.h, &self.x)?;
        let hp = mat_mul(&self.h, &self.p)?;
        let ht = mat_transpose(&self.h);
        let hpht = mat_mul(&hp, &ht)?;
        let s = mat_add(&hpht, &self.r)?;
        let s_inv = mat_inv(&s)?;
        let y: Vec<f64> = z.iter().zip(hx.iter()).map(|(zi, hi)| zi - hi).collect();
        let sy = mat_vec_mul(&s_inv, &y)?;
        let mahal: f64 = dot(&y, &sy);

        // Compute log|S| via Cholesky: log|S| = 2 * sum(log(diag(L)))
        use super::matrix_utils::cholesky_decomp;
        let l = cholesky_decomp(&s)?;
        let log_det_s: f64 = l
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].ln())
            .sum::<f64>()
            * 2.0;

        let k = self.dim_z as f64;
        let ll =
            -0.5 * (mahal + log_det_s + k * (std::f64::consts::LN_2 + std::f64::consts::PI.ln()));
        Ok(ll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Track a 1D constant-velocity object.
    ///
    /// State = [position, velocity], measurements are noisy positions.
    #[test]
    fn test_constant_velocity_tracking() {
        let mut kf = KalmanFilter::new(2, 1);
        // x_{k+1} = x_k + v_k * dt, v_{k+1} = v_k
        let dt = 1.0;
        kf.set_F(vec![vec![1.0, dt], vec![0.0, 1.0]])
            .expect("set F");
        // Observe position only
        kf.set_H(vec![vec![1.0, 0.0]]).expect("set H");
        // Small process noise
        kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        // Measurement noise variance
        kf.set_R(vec![vec![1.0]]).expect("set R");
        // Initial state: position=0, velocity=1
        kf.set_initial_state(&[0.0, 1.0])
            .expect("set initial state");

        // True trajectory: position = k, velocity = 1.0
        let true_positions: Vec<f64> = (0..20).map(|k| k as f64).collect();
        // Noisy measurements (deterministic for reproducibility)
        let measurements: Vec<Vec<f64>> = true_positions
            .iter()
            .enumerate()
            .map(|(i, &p)| vec![p + (i as f64 % 3.0 - 1.0) * 0.5])
            .collect();

        let states = kf
            .filter_sequence(&measurements)
            .expect("filter should succeed");

        assert_eq!(states.len(), 20);

        // After convergence the velocity estimate should be close to 1.0
        let final_velocity = states[19][1];
        assert!(
            (final_velocity - 1.0).abs() < 0.5,
            "Velocity estimate {:.3} should be near 1.0",
            final_velocity
        );
    }

    #[test]
    fn test_predict_update_cycle() {
        let mut kf = KalmanFilter::new(2, 2);
        kf.set_initial_state(&[1.0, 2.0])
            .expect("set initial state");
        kf.predict().expect("predict");
        kf.update(&[1.1, 2.1]).expect("update");
        let s = kf.state();
        assert_eq!(s.len(), 2);
        // State should be influenced by both prediction and measurement
        assert!((s[0] - 1.0).abs() < 0.5);
        assert!((s[1] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_covariance_decreases_after_updates() {
        let mut kf = KalmanFilter::new(1, 1);
        kf.set_F(vec![vec![1.0]]).expect("set F");
        kf.set_H(vec![vec![1.0]]).expect("set H");
        kf.set_Q(vec![vec![0.001]]).expect("set Q");
        kf.set_R(vec![vec![0.1]]).expect("set R");
        kf.set_initial_state(&[0.0]).expect("set initial state");
        kf.set_P(vec![vec![10.0]]).expect("set P");

        let initial_p = kf.covariance()[0][0];
        for _ in 0..20 {
            kf.predict().expect("predict");
            kf.update(&[0.0]).expect("update");
        }
        let final_p = kf.covariance()[0][0];
        assert!(
            final_p < initial_p,
            "Covariance should decrease: initial={:.4}, final={:.4}",
            initial_p,
            final_p
        );
    }

    #[test]
    fn test_dimension_validation() {
        let mut kf = KalmanFilter::new(2, 1);
        assert!(kf.set_F(vec![vec![1.0]]).is_err());
        assert!(kf.update(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_log_likelihood() {
        let mut kf = KalmanFilter::new(1, 1);
        kf.set_F(vec![vec![1.0]]).expect("set F");
        kf.set_H(vec![vec![1.0]]).expect("set H");
        kf.set_Q(vec![vec![0.01]]).expect("set Q");
        kf.set_R(vec![vec![0.1]]).expect("set R");
        kf.set_initial_state(&[5.0]).expect("set initial state");
        // A measurement near the predicted state should have higher log-likelihood
        let ll_near = kf.log_likelihood(&[5.0]).expect("ll_near");
        let ll_far = kf.log_likelihood(&[100.0]).expect("ll_far");
        assert!(ll_near > ll_far, "Near measurement should have higher ll");
    }

    #[test]
    fn test_outer_product_in_covariance_update() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![3.0, 4.0];
        let op = outer_product(&v1, &v2);
        assert!((op[0][0] - 3.0).abs() < 1e-12);
        assert!((op[0][1] - 4.0).abs() < 1e-12);
        assert!((op[1][0] - 6.0).abs() < 1e-12);
        assert!((op[1][1] - 8.0).abs() < 1e-12);
    }
}
