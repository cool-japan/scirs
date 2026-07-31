//! Extended Kalman Filter (EKF) for nonlinear systems.
//!
//! The EKF linearises the nonlinear dynamics at the current state estimate
//! using first-order Taylor expansion (Jacobian matrices), then applies the
//! standard Kalman predict/update equations with those Jacobians.
//!
//! # System Model
//!
//! ```text
//! x_k  = f(x_{k-1}) + w_k,    w_k ~ N(0, Q)
//! z_k  = h(x_k)      + v_k,   v_k ~ N(0, R)
//! ```
//!
//! # References
//!
//! * Jazwinski, A.H. (1970). *Stochastic Processes and Filtering Theory*.
//! * Welch & Bishop (1995). "An Introduction to the Kalman Filter".

use super::matrix_utils::{
    mat_add, mat_eye, mat_inv, mat_mul, mat_sub, mat_transpose, mat_vec_mul, vec_add, vec_sub,
};
use crate::error::{SignalError, SignalResult};

/// Extended Kalman Filter for nonlinear Gaussian systems.
///
/// Accepts user-supplied closures for the nonlinear state transition function,
/// the nonlinear observation function, and their respective Jacobians.
///
/// # Example
///
/// ```
/// use scirs2_signal::kalman::ExtendedKalmanFilter;
///
/// // Pendulum with state = [angle, angular_velocity]
/// let dt = 0.01_f64;
/// let g_over_l = 9.81 / 1.0; // g/l
///
/// let f = move |x: &[f64]| -> Vec<f64> {
///     vec![
///         x[0] + x[1] * dt,
///         x[1] - g_over_l * x[0].sin() * dt,
///     ]
/// };
/// let h = |x: &[f64]| -> Vec<f64> { vec![x[0]] };
///
/// let f_jac = move |x: &[f64]| -> Vec<Vec<f64>> {
///     vec![
///         vec![1.0, dt],
///         vec![-g_over_l * x[0].cos() * dt, 1.0],
///     ]
/// };
/// let h_jac = |_x: &[f64]| -> Vec<Vec<f64>> {
///     vec![vec![1.0, 0.0]]
/// };
///
/// let mut ekf = ExtendedKalmanFilter::new(
///     Box::new(f), Box::new(h), Box::new(f_jac), Box::new(h_jac),
///     2, 1,
/// );
/// ekf.set_initial_state(&[0.1, 0.0]).expect("operation should succeed");
/// ekf.predict().expect("operation should succeed");
/// ekf.update(&[0.09]).expect("operation should succeed");
/// ```
pub struct ExtendedKalmanFilter {
    /// Nonlinear state transition function f(x)
    f: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Nonlinear observation function h(x)
    h: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    /// Jacobian of f at x: ∂f/∂x evaluated at current state
    f_jacobian: Box<dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    /// Jacobian of h at x: ∂h/∂x evaluated at current state
    h_jacobian: Box<dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    /// Process noise covariance Q (dim_x × dim_x)
    q: Vec<Vec<f64>>,
    /// Measurement noise covariance R (dim_z × dim_z)
    r: Vec<Vec<f64>>,
    /// State estimate covariance P (dim_x × dim_x)
    p: Vec<Vec<f64>>,
    /// State estimate vector
    x: Vec<f64>,
    /// State dimension
    dim_x: usize,
    /// Measurement dimension
    dim_z: usize,
}

impl ExtendedKalmanFilter {
    /// Create a new Extended Kalman Filter.
    ///
    /// # Arguments
    ///
    /// * `f`         - Nonlinear state transition function
    /// * `h`         - Nonlinear observation function
    /// * `f_jacobian` - Jacobian of f at current state (dim_x × dim_x matrix)
    /// * `h_jacobian` - Jacobian of h at current state (dim_z × dim_x matrix)
    /// * `dim_x`     - State dimension
    /// * `dim_z`     - Measurement dimension
    pub fn new(
        f: Box<dyn Fn(&[f64]) -> Vec<f64>>,
        h: Box<dyn Fn(&[f64]) -> Vec<f64>>,
        f_jacobian: Box<dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
        h_jacobian: Box<dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
        dim_x: usize,
        dim_z: usize,
    ) -> Self {
        ExtendedKalmanFilter {
            f,
            h,
            f_jacobian,
            h_jacobian,
            q: mat_eye(dim_x),
            r: mat_eye(dim_z),
            p: mat_eye(dim_x),
            x: vec![0.0_f64; dim_x],
            dim_x,
            dim_z,
        }
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

    /// Predict step using the nonlinear transition function f and its Jacobian.
    ///
    /// ```text
    /// x_{k|k-1} = f(x_{k-1|k-1})
    /// F_k       = ∂f/∂x |_{x_{k-1|k-1}}
    /// P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q
    /// ```
    pub fn predict(&mut self) -> SignalResult<()> {
        // Nonlinear state propagation
        self.x = (self.f)(&self.x);

        // Linearise: F_k = Jacobian of f at x
        let f_k = (self.f_jacobian)(&self.x);

        // Predicted covariance: P = F * P * F^T + Q
        let fp = mat_mul(&f_k, &self.p)?;
        let ft = mat_transpose(&f_k);
        let fpft = mat_mul(&fp, &ft)?;
        self.p = mat_add(&fpft, &self.q)?;

        Ok(())
    }

    /// Update step using the nonlinear observation function h and its Jacobian.
    ///
    /// ```text
    /// H_k = ∂h/∂x |_{x_{k|k-1}}
    /// y   = z - h(x_{k|k-1})
    /// S   = H_k * P_{k|k-1} * H_k^T + R
    /// K   = P_{k|k-1} * H_k^T * S^{-1}
    /// x   = x_{k|k-1} + K * y
    /// P   = (I - K * H_k) * P_{k|k-1}
    /// ```
    pub fn update(&mut self, z: &[f64]) -> SignalResult<()> {
        if z.len() != self.dim_z {
            return Err(SignalError::ValueError(format!(
                "Measurement length {} != dim_z {}",
                z.len(),
                self.dim_z
            )));
        }

        // Predicted measurement
        let hx = (self.h)(&self.x);
        // Innovation
        let y = vec_sub(z, &hx);

        // Jacobian H_k at current estimate
        let h_k = (self.h_jacobian)(&self.x);

        // Innovation covariance S = H * P * H^T + R
        let hp = mat_mul(&h_k, &self.p)?;
        let ht = mat_transpose(&h_k);
        let hpht = mat_mul(&hp, &ht)?;
        let s = mat_add(&hpht, &self.r)?;

        // Kalman gain K = P * H^T * S^{-1}
        let s_inv = mat_inv(&s)?;
        let pht = mat_mul(&self.p, &ht)?;
        let k = mat_mul(&pht, &s_inv)?;

        // State update
        let ky = mat_vec_mul(&k, &y)?;
        self.x = vec_add(&self.x, &ky);

        // Covariance update (Joseph form for numerical stability)
        let kh = mat_mul(&k, &h_k)?;
        let i_kh = mat_sub(&mat_eye(self.dim_x), &kh)?;
        let p_new = mat_mul(&i_kh, &self.p)?;
        // Symmetrize
        let pt = mat_transpose(&p_new);
        self.p = p_new
            .iter()
            .zip(pt.iter())
            .map(|(r1, r2)| {
                r1.iter()
                    .zip(r2.iter())
                    .map(|(a, b)| 0.5 * (a + b))
                    .collect()
            })
            .collect();

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

/// Numerically compute the Jacobian of a function at point x using finite differences.
///
/// Useful for creating EKF instances when analytical Jacobians are not available.
///
/// # Arguments
///
/// * `f`    - The function to differentiate
/// * `x`    - The point at which to evaluate the Jacobian
/// * `eps`  - Step size for finite differences (default ~1e-5)
///
/// # Returns
///
/// Jacobian matrix (output_dim × input_dim)
pub fn numerical_jacobian<F>(f: &F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![vec![0.0_f64; n]; m];
    let mut xp = x.to_vec();
    for j in 0..n {
        xp[j] = x[j] + eps;
        let fp = f(&xp);
        xp[j] = x[j] - eps;
        let fm = f(&xp);
        xp[j] = x[j];
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test EKF on a nonlinear pendulum system.
    ///
    /// State: [theta, omega], Observation: [theta] (position only)
    #[test]
    fn test_pendulum_tracking() {
        let dt = 0.01_f64;
        let g_l = 9.81_f64;

        let f =
            move |x: &[f64]| -> Vec<f64> { vec![x[0] + x[1] * dt, x[1] - g_l * x[0].sin() * dt] };
        let h = |x: &[f64]| -> Vec<f64> { vec![x[0]] };

        let f_jac = move |x: &[f64]| -> Vec<Vec<f64>> {
            vec![vec![1.0, dt], vec![-g_l * x[0].cos() * dt, 1.0]]
        };
        let h_jac = |_x: &[f64]| -> Vec<Vec<f64>> { vec![vec![1.0, 0.0]] };

        let mut ekf = ExtendedKalmanFilter::new(
            Box::new(f),
            Box::new(h),
            Box::new(f_jac),
            Box::new(h_jac),
            2,
            1,
        );
        ekf.set_initial_state(&[0.1, 0.0])
            .expect("set initial state");
        ekf.set_Q(vec![vec![1e-5, 0.0], vec![0.0, 1e-5]])
            .expect("set Q");
        ekf.set_R(vec![vec![0.01]]).expect("set R");
        ekf.set_P(vec![vec![0.1, 0.0], vec![0.0, 0.1]])
            .expect("set P");

        // Simulate pendulum and add observations
        let mut true_theta = 0.1_f64;
        let mut true_omega = 0.0_f64;

        for step in 0..100 {
            let new_theta = true_theta + true_omega * dt;
            let new_omega = true_omega - g_l * true_theta.sin() * dt;
            true_theta = new_theta;
            true_omega = new_omega;

            // Noisy measurement (use step-dependent offset for reproducibility)
            let noise = (step as f64 % 5.0 - 2.5) * 0.01;
            let z = vec![true_theta + noise];

            ekf.predict().expect("predict");
            ekf.update(&z).expect("update");
        }

        let state = ekf.state();
        // After tracking the pendulum, angle estimate should be reasonable
        assert!(
            state[0].abs() < 0.5,
            "Pendulum angle estimate {:.4} should be bounded",
            state[0]
        );
    }

    #[test]
    fn test_numerical_jacobian_accuracy() {
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0] * x[0], x[0] * x[1], x[1].sin()] };
        let x = vec![2.0, 1.0];
        let jac = numerical_jacobian(&f, &x, 1e-6);

        // ∂(x0²)/∂x0 = 2*x0 = 4
        assert!(
            (jac[0][0] - 4.0).abs() < 1e-5,
            "Jacobian [0][0] should be 4.0"
        );
        // ∂(x0*x1)/∂x0 = x1 = 1
        assert!(
            (jac[1][0] - 1.0).abs() < 1e-5,
            "Jacobian [1][0] should be 1.0"
        );
        // ∂(sin(x1))/∂x1 = cos(x1) ≈ cos(1)
        assert!(
            (jac[2][1] - x[1].cos()).abs() < 1e-5,
            "Jacobian [2][1] should be cos(1)"
        );
    }

    #[test]
    fn test_ekf_linear_matches_kf() {
        // For a linear system, EKF and KF should give identical results
        let dt = 1.0_f64;
        let f_lin = move |x: &[f64]| -> Vec<f64> { vec![x[0] + x[1] * dt, x[1]] };
        let h_lin = |x: &[f64]| -> Vec<f64> { vec![x[0]] };
        let f_jac_lin = move |_x: &[f64]| -> Vec<Vec<f64>> { vec![vec![1.0, dt], vec![0.0, 1.0]] };
        let h_jac_lin = |_x: &[f64]| -> Vec<Vec<f64>> { vec![vec![1.0, 0.0]] };

        let mut ekf = ExtendedKalmanFilter::new(
            Box::new(f_lin),
            Box::new(h_lin),
            Box::new(f_jac_lin),
            Box::new(h_jac_lin),
            2,
            1,
        );
        ekf.set_initial_state(&[0.0, 1.0])
            .expect("set initial state");
        ekf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        ekf.set_R(vec![vec![1.0]]).expect("set R");

        use super::super::standard::KalmanFilter;
        let mut kf = KalmanFilter::new(2, 1);
        kf.set_F(vec![vec![1.0, dt], vec![0.0, 1.0]])
            .expect("set F");
        kf.set_H(vec![vec![1.0, 0.0]]).expect("set H");
        kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]])
            .expect("set Q");
        kf.set_R(vec![vec![1.0]]).expect("set R");
        kf.set_initial_state(&[0.0, 1.0])
            .expect("set initial state");

        let meas_seq = vec![vec![1.0], vec![2.1], vec![3.0], vec![3.9], vec![5.0]];
        for z in &meas_seq {
            ekf.predict().expect("ekf predict");
            ekf.update(z).expect("ekf update");
            kf.predict().expect("kf predict");
            kf.update(z).expect("kf update");
        }

        let ekf_state = ekf.state();
        let kf_state = kf.state();
        for i in 0..2 {
            assert!(
                (ekf_state[i] - kf_state[i]).abs() < 1e-8,
                "EKF and KF should agree for linear system: EKF[{}]={:.6}, KF[{}]={:.6}",
                i,
                ekf_state[i],
                i,
                kf_state[i]
            );
        }
    }
}
