//! Information Filter — the dual form of the Kalman filter.
//!
//! Instead of storing state x and covariance P, the information filter
//! maintains:
//! - **Information matrix**: Ω = P⁻¹
//! - **Information vector**: ξ = P⁻¹ · x = Ω · x
//!
//! This representation is numerically advantageous when the initial uncertainty
//! is very large (P → ∞ → Ω → 0) and when fusing many measurements, as each
//! measurement simply adds to the information.
//!
//! # System Model
//!
//! Same linear Gaussian model as the standard Kalman filter:
//! ```text
//! x_k  = F * x_{k-1} + w_k,  w_k ~ N(0, Q)
//! z_k  = H * x_k     + v_k,  v_k ~ N(0, R)
//! ```
//!
//! # References
//!
//! * Maybeck, P.S. (1979). *Stochastic Models, Estimation and Control*, Vol. 1.
//! * Thrun, S., Burgard, W. & Fox, D. (2005). *Probabilistic Robotics*, Ch. 3.

use super::matrix_utils::{
    mat_add, mat_eye, mat_inv, mat_mul, mat_scale, mat_transpose, mat_vec_mul,
};
use crate::error::{SignalError, SignalResult};

/// Information Filter — dual representation of the Kalman filter.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::InformationFilter;
///
/// let mut inf = InformationFilter::new(2, 1);
/// inf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]]).expect("operation should succeed");
/// inf.set_H(vec![vec![1.0, 0.0]]).expect("operation should succeed");
/// inf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]]).expect("operation should succeed");
/// inf.set_R(vec![vec![0.1]]).expect("operation should succeed");
/// // Zero initial information = totally uncertain
/// inf.predict().expect("operation should succeed");
/// inf.update(&[1.0]).expect("operation should succeed");
/// let state = inf.state().expect("operation should succeed");
/// assert_eq!(state.len(), 2);
/// ```
pub struct InformationFilter {
    /// State transition matrix F (dim_x × dim_x)
    f: Vec<Vec<f64>>,
    /// Observation matrix H (dim_z × dim_x)
    h: Vec<Vec<f64>>,
    /// Process noise covariance Q (dim_x × dim_x)
    q: Vec<Vec<f64>>,
    /// Measurement noise covariance R (dim_z × dim_z)
    r: Vec<Vec<f64>>,
    /// Information matrix Ω = P⁻¹ (dim_x × dim_x)
    omega: Vec<Vec<f64>>,
    /// Information vector ξ = Ω·x (dim_x)
    xi: Vec<f64>,
    /// State dimension
    dim_x: usize,
    /// Measurement dimension
    dim_z: usize,
}

impl InformationFilter {
    /// Create a new Information Filter with zero initial information
    /// (equivalent to infinite covariance / totally uninformative prior).
    ///
    /// # Arguments
    ///
    /// * `dim_x` - State dimension
    /// * `dim_z` - Measurement dimension
    pub fn new(dim_x: usize, dim_z: usize) -> Self {
        InformationFilter {
            f: mat_eye(dim_x),
            h: {
                let rows = dim_z.min(dim_x);
                let mut h = vec![vec![0.0_f64; dim_x]; dim_z];
                for i in 0..rows {
                    h[i][i] = 1.0;
                }
                h
            },
            q: mat_eye(dim_x),
            r: mat_eye(dim_z),
            // Zero information = totally uncertain (P = ∞)
            omega: vec![vec![0.0_f64; dim_x]; dim_x],
            xi: vec![0.0_f64; dim_x],
            dim_x,
            dim_z,
        }
    }

    /// Initialize from a known state and covariance.
    pub fn initialize_from_state(&mut self, x: &[f64], p: &[Vec<f64>]) -> SignalResult<()> {
        if x.len() != self.dim_x {
            return Err(SignalError::ValueError("State length mismatch".to_string()));
        }
        self.omega = mat_inv(p)?;
        self.xi = mat_vec_mul(&self.omega, x)?;
        Ok(())
    }

    /// Set the state transition matrix F.
    #[allow(non_snake_case)]
    pub fn set_F(&mut self, f: Vec<Vec<f64>>) -> SignalResult<()> {
        if f.len() != self.dim_x || f.iter().any(|r| r.len() != self.dim_x) {
            return Err(SignalError::ValueError(
                "F must be dim_x × dim_x".to_string(),
            ));
        }
        self.f = f;
        Ok(())
    }

    /// Set the observation matrix H.
    #[allow(non_snake_case)]
    pub fn set_H(&mut self, h: Vec<Vec<f64>>) -> SignalResult<()> {
        if h.len() != self.dim_z || h.iter().any(|r| r.len() != self.dim_x) {
            return Err(SignalError::ValueError(
                "H must be dim_z × dim_x".to_string(),
            ));
        }
        self.h = h;
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
        if r.len() != self.dim_z || r.iter().any(|r| r.len() != self.dim_z) {
            return Err(SignalError::ValueError(
                "R must be dim_z × dim_z".to_string(),
            ));
        }
        self.r = r;
        Ok(())
    }

    /// Prediction step in information form.
    ///
    /// Using the Woodbury identity to avoid explicit P inversion:
    ///
    /// ```text
    /// M    = F^{-T} * Ω               (information propagated forward)
    /// Ω̄    = (F * Ω^{-1} * F^T + Q)^{-1}
    /// ξ̄    = Ω̄ * F^{-T} * ξ
    /// ```
    ///
    /// Note: when Ω is singular (initially zero), we handle this by
    /// working in covariance form for the predict step.
    pub fn predict(&mut self) -> SignalResult<()> {
        // Convert to covariance form for predict (handles zero Ω gracefully)
        let p_pred = if self.is_zero_omega() {
            // Ω = 0 → P = ∞. After predict with Q: P = F*∞*F^T + Q → still ∞
            // Approximate as very large P
            mat_scale(&mat_eye(self.dim_x), 1e10)
        } else {
            let p = mat_inv(&self.omega)?;
            let fp = mat_mul(&self.f, &p)?;
            let ft = mat_transpose(&self.f);
            let fpft = mat_mul(&fp, &ft)?;
            mat_add(&fpft, &self.q)?
        };

        // Updated state in covariance form
        let x_pred = if self.is_zero_omega() {
            vec![0.0_f64; self.dim_x]
        } else {
            let p = mat_inv(&self.omega)?;
            let x = mat_vec_mul(&p, &self.xi)?;
            mat_vec_mul(&self.f, &x)?
        };

        // Convert back to information form
        self.omega = mat_inv(&p_pred)?;
        self.xi = mat_vec_mul(&self.omega, &x_pred)?;

        Ok(())
    }

    /// Update step in information form (additive, numerically cheap).
    ///
    /// ```text
    /// R_inv = R^{-1}
    /// Ω     = Ω̄ + H^T * R^{-1} * H
    /// ξ     = ξ̄ + H^T * R^{-1} * z
    /// ```
    pub fn update(&mut self, z: &[f64]) -> SignalResult<()> {
        if z.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "Measurement length {} != dim_z {}",
                z.len(),
                self.dim_z
            )));
        }

        let r_inv = mat_inv(&self.r)?;
        let ht = mat_transpose(&self.h);

        // Information matrix update: Ω += H^T * R^{-1} * H
        let ht_r_inv = mat_mul(&ht, &r_inv)?;
        let info_update = mat_mul(&ht_r_inv, &self.h)?;
        self.omega = mat_add(&self.omega, &info_update)?;

        // Information vector update: ξ += H^T * R^{-1} * z
        let r_inv_z = mat_vec_mul(&r_inv, z)?;
        let xi_update = mat_vec_mul(&ht, &r_inv_z)?;
        for (a, b) in self.xi.iter_mut().zip(xi_update.iter()) {
            *a += b;
        }

        Ok(())
    }

    /// Extract the current state estimate: x = Ω^{-1} * ξ.
    ///
    /// # Returns
    ///
    /// `Ok(x)` if information matrix is invertible, `Err` if system is still uninformative.
    pub fn state(&self) -> SignalResult<Vec<f64>> {
        if self.is_zero_omega() {
            return Err(SignalError::ComputationError(
                "Information matrix is zero: system has not received enough observations"
                    .to_string(),
            ));
        }
        let p = mat_inv(&self.omega)?;
        mat_vec_mul(&p, &self.xi)
    }

    /// Extract the current state covariance: P = Ω^{-1}.
    pub fn covariance(&self) -> SignalResult<Vec<Vec<f64>>> {
        if self.is_zero_omega() {
            return Err(SignalError::ComputationError(
                "Information matrix is zero: covariance is infinite".to_string(),
            ));
        }
        mat_inv(&self.omega)
    }

    /// Access the information matrix Ω directly.
    pub fn information_matrix(&self) -> &Vec<Vec<f64>> {
        &self.omega
    }

    /// Access the information vector ξ directly.
    pub fn information_vector(&self) -> &[f64] {
        &self.xi
    }

    /// Fuse another information filter's information (sensor fusion shortcut).
    ///
    /// Since information is additive, multiple filters can be fused simply by
    /// adding their information matrices and vectors.
    pub fn fuse(&mut self, other: &InformationFilter) -> SignalResult<()> {
        if other.dim_x != self.dim_x {
            return Err(SignalError::ValueError(
                "Cannot fuse filters with different state dimensions".to_string(),
            ));
        }
        self.omega = mat_add(&self.omega, other.information_matrix())?;
        for (a, b) in self.xi.iter_mut().zip(other.information_vector().iter()) {
            *a += b;
        }
        Ok(())
    }

    fn is_zero_omega(&self) -> bool {
        self.omega
            .iter()
            .all(|row| row.iter().all(|&v| v.abs() < 1e-14))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare information filter with standard Kalman filter on a linear system.
    #[test]
    fn test_information_filter_matches_kalman() {
        use super::super::standard::KalmanFilter;

        let mut inf = InformationFilter::new(2, 1);
        inf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]])
            .expect("set F");
        inf.set_H(vec![vec![1.0, 0.0]]).expect("set H");
        inf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        inf.set_R(vec![vec![0.1]]).expect("set R");

        let p0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x0 = vec![0.0, 1.0];
        inf.initialize_from_state(&x0, &p0).expect("initialize");

        let mut kf = KalmanFilter::new(2, 1);
        kf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]])
            .expect("set F");
        kf.set_H(vec![vec![1.0, 0.0]]).expect("set H");
        kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        kf.set_R(vec![vec![0.1]]).expect("set R");
        kf.set_initial_state(&x0).expect("set initial state");
        kf.set_P(p0.clone()).expect("set P");

        let measurements = vec![vec![1.0], vec![2.1], vec![2.9], vec![4.0], vec![5.1]];

        for z in &measurements {
            inf.predict().expect("inf predict");
            inf.update(z).expect("inf update");
            kf.predict().expect("kf predict");
            kf.update(z).expect("kf update");
        }

        let inf_state = inf.state().expect("get inf state");
        let kf_state = kf.state();

        for i in 0..2 {
            assert!(
                (inf_state[i] - kf_state[i]).abs() < 1e-6,
                "IF and KF disagree at index {}: IF={:.6}, KF={:.6}",
                i,
                inf_state[i],
                kf_state[i]
            );
        }
    }

    #[test]
    fn test_information_update_is_additive() {
        // Starting from zero info, two identical measurements should give twice the info
        let mut inf1 = InformationFilter::new(1, 1);
        inf1.set_F(vec![vec![1.0]]).expect("set F");
        inf1.set_H(vec![vec![1.0]]).expect("set H");
        inf1.set_Q(vec![vec![0.0]]).expect("set Q"); // no process noise
        inf1.set_R(vec![vec![0.5]]).expect("set R");

        let mut inf2 = inf1.clone_structure();

        // Fuse two identical sensors
        inf1.update(&[3.0]).expect("update 1");
        inf2.update(&[3.0]).expect("update 2");
        inf1.fuse(&inf2).expect("fuse");

        // The fused information matrix should be 2 * R^{-1} * H^T * H = 2/0.5 = 4
        let omega = inf1.information_matrix();
        assert!(
            (omega[0][0] - 4.0).abs() < 1e-10,
            "Omega should be 4.0, got {:.6}",
            omega[0][0]
        );

        // State should still be 3.0
        let state = inf1.state().expect("get state");
        assert!(
            (state[0] - 3.0).abs() < 1e-10,
            "State should be 3.0, got {:.6}",
            state[0]
        );
    }

    #[test]
    fn test_uninformative_prior_works() {
        // Start with zero information (no prior)
        let mut inf = InformationFilter::new(1, 1);
        inf.set_H(vec![vec![1.0]]).expect("set H");
        inf.set_R(vec![vec![0.1]]).expect("set R");
        inf.set_Q(vec![vec![0.0]]).expect("set Q");
        inf.set_F(vec![vec![1.0]]).expect("set F");

        // Should fail before any update
        assert!(inf.state().is_err(), "Should fail with zero information");

        // After predict + update it should be OK
        inf.predict().expect("predict");
        inf.update(&[5.0]).expect("update");
        let state = inf.state().expect("state after update");
        assert!(
            (state[0] - 5.0).abs() < 1e-6,
            "State should be 5.0, got {:.6}",
            state[0]
        );
    }
}

#[cfg(test)]
impl InformationFilter {
    /// Helper to clone the structural parameters (F, H, Q, R) for testing.
    fn clone_structure(&self) -> Self {
        InformationFilter {
            f: self.f.clone(),
            h: self.h.clone(),
            q: self.q.clone(),
            r: self.r.clone(),
            omega: vec![vec![0.0_f64; self.dim_x]; self.dim_x],
            xi: vec![0.0_f64; self.dim_x],
            dim_x: self.dim_x,
            dim_z: self.dim_z,
        }
    }
}
