// Kalman filtering module
//
// This module implements various Kalman filtering techniques for signal processing,
// including standard Kalman filter, extended Kalman filter, unscented Kalman filter,
// and ensemble Kalman filter.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, Array3};
use rand::Rng;
use scirs2_linalg::{cholesky, inv};

/// Configuration for Kalman filter
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// Initial state covariance matrix
    pub initial_p: Option<Array2<f64>>,
    /// Process noise covariance matrix
    pub q: Option<Array2<f64>>,
    /// Measurement noise covariance matrix
    pub r: Option<Array2<f64>>,
    /// Control input matrix (optional)
    pub b: Option<Array2<f64>>,
    /// Process noise scaling
    pub process_noise_scale: f64,
    /// Measurement noise scaling
    pub measurement_noise_scale: f64,
    /// Whether to adapt Q and R matrices during filtering
    pub adaptive: bool,
    /// Window size for adaptive estimation
    pub adaptive_window: usize,
    /// Forgetting factor for adaptive estimation (0 < factor ≤ 1)
    pub forgetting_factor: f64,
}

/// Configuration for Unscented Kalman Filter
#[derive(Debug, Clone)]
pub struct UkfConfig {
    /// Spread parameter
    pub alpha: f64,
    /// Prior knowledge parameter (2 is optimal for Gaussian distributions)
    pub beta: f64,
    /// Secondary scaling parameter
    pub kappa: f64,
}

impl Default for UkfConfig {
    fn default() -> Self {
        Self {
            alpha: 1e-3, // Small alpha for stability
            beta: 2.0,   // Optimal for Gaussian distributions
            kappa: 0.0,  // Secondary scaling parameter
        }
    }
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            initial_p: None,
            q: None,
            r: None,
            b: None,
            process_noise_scale: 1e-4,
            measurement_noise_scale: 1e-2,
            adaptive: false,
            adaptive_window: 10,
            forgetting_factor: 0.95,
        }
    }
}

/// Kalman filter for state estimation
///
/// # Arguments
///
/// * `z` - Measurement signal
/// * `f` - State transition matrix
/// * `h` - Measurement matrix
/// * `initial_x` - Initial state estimate (optional)
/// * `config` - Kalman filter configuration
///
/// # Returns
///
/// * Estimated state at each time step
pub fn kalman_filter(
    z: &Array1<f64>,
    f: &Array2<f64>,
    h: &Array2<f64>,
    initial_x: Option<Array1<f64>>,
    config: &KalmanConfig,
) -> SignalResult<Array2<f64>> {
    let n_samples = z.len();
    let n_states = f.shape()[0];

    // Validate input dimensions
    if f.shape()[0] != f.shape()[1] {
        return Err(SignalError::DimensionMismatch(
            "State transition matrix F must be square".to_string(),
        ));
    }

    if h.shape()[1] != n_states {
        return Err(SignalError::DimensionMismatch(
            "Measurement matrix H must have same number of columns as states".to_string(),
        ));
    }

    // Initialize state estimate
    let mut x = match initial_x {
        Some(x0) => {
            if x0.len() != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial state dimension mismatch".to_string(),
                ));
            }
            x0
        }
        None => Array1::<f64>::zeros(n_states),
    };

    // Initialize state covariance
    let mut p = match &config.initial_p {
        Some(p0) => {
            if p0.shape()[0] != n_states || p0.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial covariance dimension mismatch".to_string(),
                ));
            }
            p0.clone()
        }
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Process noise covariance
    let q = match &config.q {
        Some(q) => {
            if q.shape()[0] != n_states || q.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Process noise covariance dimension mismatch".to_string(),
                ));
            }
            q.clone()
        }
        None => Array2::<f64>::eye(n_states) * config.process_noise_scale,
    };

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => {
            if r.shape()[0] != h.shape()[0] || r.shape()[1] != h.shape()[0] {
                return Err(SignalError::DimensionMismatch(
                    "Measurement noise covariance dimension mismatch".to_string(),
                ));
            }
            r.clone()
        }
        None => Array2::<f64>::eye(h.shape()[0]) * config.measurement_noise_scale,
    };

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n_samples, n_states));

    // Adaptive filtering variables
    let mut adaptive_q = q.clone();
    let mut adaptive_r = r.clone();
    let mut innovation_history = Vec::with_capacity(config.adaptive_window);

    // Run Kalman filter
    for i in 0..n_samples {
        // Predict step
        let x_pred = f.dot(&x);
        let p_pred = f.dot(&p).dot(&f.t()) + &adaptive_q;

        // Update step
        let z_pred = h.dot(&x_pred);
        let z_value = z.slice(s![i]).to_owned();
        let innovation = &z_value - &z_pred;
        let innovation_cov = h.dot(&p_pred).dot(&h.t()) + &adaptive_r;

        // Kalman gain
        let k = match inv(&innovation_cov.view(), None) {
            Ok(inn_cov_inv) => p_pred.dot(&h.t()).dot(&inn_cov_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // State and covariance update
        x = &x_pred + &k.dot(&innovation);
        p = &p_pred - &k.dot(&innovation_cov).dot(&k.t());

        // Store state
        for j in 0..n_states {
            x_history[[i, j]] = x[j];
        }

        // Adaptive filtering (if enabled)
        if config.adaptive {
            // Store innovation for adaptive estimation
            innovation_history.push(innovation.to_owned());
            if innovation_history.len() > config.adaptive_window {
                innovation_history.remove(0);
            }

            if innovation_history.len() >= 3 {
                // Adaptive estimation of R
                let mut innovation_sum = Array1::<f64>::zeros(innovation.len());
                for inn in &innovation_history {
                    innovation_sum += inn;
                }
                let innovation_mean = innovation_sum / (innovation_history.len() as f64);

                // Compute sample covariance
                let mut r_estimate = Array2::<f64>::zeros(adaptive_r.dim());
                for inn in &innovation_history {
                    let centered = inn - &innovation_mean;
                    let centered_col = centered
                        .clone()
                        .into_shape_with_order((centered.len(), 1))
                        .unwrap();
                    let centered_row = centered
                        .clone()
                        .into_shape_with_order((1, centered.len()))
                        .unwrap();
                    r_estimate += &centered_col.dot(&centered_row);
                }
                r_estimate /= innovation_history.len() as f64;

                // Update R with forgetting factor
                adaptive_r = &adaptive_r * (1.0 - config.forgetting_factor)
                    + &r_estimate * config.forgetting_factor;

                // Adaptive estimation of Q
                // This is a simplified approach - in practice, estimating Q is more complex
                // than estimating R and might require more sophisticated techniques
                let pred_err = &x - &x_pred;
                let pred_err_col = pred_err
                    .clone()
                    .into_shape_with_order((pred_err.len(), 1))
                    .unwrap();
                let pred_err_row = pred_err
                    .clone()
                    .into_shape_with_order((1, pred_err.len()))
                    .unwrap();
                let q_update = pred_err_col.dot(&pred_err_row);
                adaptive_q = &adaptive_q * (1.0 - config.forgetting_factor)
                    + &q_update * config.forgetting_factor;
            }
        }
    }

    Ok(x_history)
}

/// Extended Kalman filter for nonlinear systems
///
/// # Arguments
///
/// * `z` - Measurement signal
/// * `f_func` - State transition function
/// * `h_func` - Measurement function
/// * `f_jacobian` - Jacobian of state transition function
/// * `h_jacobian` - Jacobian of measurement function
/// * `initial_x` - Initial state estimate
/// * `config` - Kalman filter configuration
///
/// # Returns
///
/// * Estimated state at each time step
pub fn extended_kalman_filter<F, H, FJ, HJ>(
    z: &Array2<f64>,
    f_func: F,
    h_func: H,
    f_jacobian: FJ,
    h_jacobian: HJ,
    initial_x: Array1<f64>,
    config: &KalmanConfig,
) -> SignalResult<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    H: Fn(&Array1<f64>) -> Array1<f64>,
    FJ: Fn(&Array1<f64>) -> Array2<f64>,
    HJ: Fn(&Array1<f64>) -> Array2<f64>,
{
    let n_samples = z.shape()[0];
    let n_states = initial_x.len();
    let n_measurements = z.shape()[1];

    // Initialize state estimate
    let mut x = initial_x;

    // Initialize state covariance
    let mut p = match &config.initial_p {
        Some(p0) => {
            if p0.shape()[0] != n_states || p0.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial covariance dimension mismatch".to_string(),
                ));
            }
            p0.clone()
        }
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Process noise covariance
    let q = match &config.q {
        Some(q) => {
            if q.shape()[0] != n_states || q.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Process noise covariance dimension mismatch".to_string(),
                ));
            }
            q.clone()
        }
        None => Array2::<f64>::eye(n_states) * config.process_noise_scale,
    };

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => {
            if r.shape()[0] != n_measurements || r.shape()[1] != n_measurements {
                return Err(SignalError::DimensionMismatch(
                    "Measurement noise covariance dimension mismatch".to_string(),
                ));
            }
            r.clone()
        }
        None => Array2::<f64>::eye(n_measurements) * config.measurement_noise_scale,
    };

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n_samples, n_states));

    // Run Extended Kalman filter
    for i in 0..n_samples {
        // Predict step
        let x_pred = f_func(&x);
        let f_jac = f_jacobian(&x);
        let p_pred = f_jac.dot(&p).dot(&f_jac.t()) + &q;

        // Update step
        let z_pred = h_func(&x_pred);
        let h_jac = h_jacobian(&x_pred);
        let innovation = z.slice(s![i, ..]).to_owned() - z_pred;
        let innovation_cov = h_jac.dot(&p_pred).dot(&h_jac.t()) + &r;

        // Kalman gain
        let k = match inv(&innovation_cov.view(), None) {
            Ok(inn_cov_inv) => p_pred.dot(&h_jac.t()).dot(&inn_cov_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // State and covariance update
        x = &x_pred + &k.dot(&innovation);
        p = &p_pred - &k.dot(&innovation_cov).dot(&k.t());

        // Store state
        for j in 0..n_states {
            x_history[[i, j]] = x[j];
        }
    }

    Ok(x_history)
}

/// Unscented Kalman filter for nonlinear systems
///
/// # Arguments
///
/// * `z` - Measurement signal
/// * `f_func` - State transition function
/// * `h_func` - Measurement function
/// * `initial_x` - Initial state estimate
/// * `config` - Kalman filter configuration
/// * `alpha` - Spread parameter
/// * `beta` - Prior knowledge parameter (2 is optimal for Gaussian distributions)
/// * `kappa` - Secondary scaling parameter
///
/// # Returns
///
/// * Estimated state at each time step
pub fn unscented_kalman_filter<F, H>(
    z: &Array2<f64>,
    f_func: F,
    h_func: H,
    initial_x: Array1<f64>,
    config: &KalmanConfig,
    ukf_config: Option<UkfConfig>,
) -> SignalResult<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    H: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n_samples = z.shape()[0];
    let n_states = initial_x.len();
    let n_measurements = z.shape()[1];

    // Get UKF parameters
    let ukf_params = ukf_config.unwrap_or_default();
    let alpha = ukf_params.alpha;
    let beta = ukf_params.beta;
    let kappa = ukf_params.kappa;

    // Initialize state estimate
    let mut x = initial_x;

    // Initialize state covariance
    let mut p = match &config.initial_p {
        Some(p0) => {
            if p0.shape()[0] != n_states || p0.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial covariance dimension mismatch".to_string(),
                ));
            }
            p0.clone()
        }
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Process noise covariance
    let q = match &config.q {
        Some(q) => {
            if q.shape()[0] != n_states || q.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Process noise covariance dimension mismatch".to_string(),
                ));
            }
            q.clone()
        }
        None => Array2::<f64>::eye(n_states) * config.process_noise_scale,
    };

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => {
            if r.shape()[0] != n_measurements || r.shape()[1] != n_measurements {
                return Err(SignalError::DimensionMismatch(
                    "Measurement noise covariance dimension mismatch".to_string(),
                ));
            }
            r.clone()
        }
        None => Array2::<f64>::eye(n_measurements) * config.measurement_noise_scale,
    };

    // UKF parameters
    let lambda = alpha * alpha * (n_states as f64 + kappa) - n_states as f64;
    let gamma = (n_states as f64 + lambda).sqrt();

    // Sigma point weights
    let w_m = vec![lambda / (n_states as f64 + lambda)];
    let mut w_m_i = vec![1.0 / (2.0 * (n_states as f64 + lambda)); 2 * n_states];
    let mut weights_mean = w_m.clone();
    weights_mean.append(&mut w_m_i);

    let w_c = vec![lambda / (n_states as f64 + lambda) + (1.0 - alpha * alpha + beta)];
    let mut w_c_i = vec![1.0 / (2.0 * (n_states as f64 + lambda)); 2 * n_states];
    let mut weights_cov = w_c.clone();
    weights_cov.append(&mut w_c_i);

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n_samples, n_states));

    // Run Unscented Kalman filter
    for i in 0..n_samples {
        // Generate sigma points
        let sigma_points = generate_sigma_points(&x, &p, gamma)?;

        // Predict sigma points through state transition function
        let mut predicted_sigmas = Vec::with_capacity(2 * n_states + 1);
        for sp in &sigma_points {
            predicted_sigmas.push(f_func(sp));
        }

        // Calculate predicted state and covariance
        let mut x_pred = Array1::<f64>::zeros(n_states);
        for j in 0..predicted_sigmas.len() {
            x_pred = &x_pred + &(&predicted_sigmas[j] * weights_mean[j]);
        }

        let mut p_pred = Array2::<f64>::zeros((n_states, n_states));
        for j in 0..predicted_sigmas.len() {
            let diff = &predicted_sigmas[j] - &x_pred;
            let diff_col = diff.clone().into_shape_with_order((diff.len(), 1)).unwrap();
            let diff_row = diff.clone().into_shape_with_order((1, diff.len())).unwrap();
            p_pred = &p_pred + &(weights_cov[j] * diff_col.dot(&diff_row));
        }
        p_pred = &p_pred + &q;

        // Predict measurement sigma points
        let mut measurement_sigmas = Vec::with_capacity(2 * n_states + 1);
        for sp in &predicted_sigmas {
            measurement_sigmas.push(h_func(sp));
        }

        // Calculate predicted measurement
        let mut z_pred = Array1::<f64>::zeros(n_measurements);
        for j in 0..measurement_sigmas.len() {
            z_pred = &z_pred + &(&measurement_sigmas[j] * weights_mean[j]);
        }

        // Calculate innovation covariance
        let mut s = Array2::<f64>::zeros((n_measurements, n_measurements));
        for j in 0..measurement_sigmas.len() {
            let diff = &measurement_sigmas[j] - &z_pred;
            let diff_col = diff.clone().into_shape_with_order((diff.len(), 1)).unwrap();
            let diff_row = diff.clone().into_shape_with_order((1, diff.len())).unwrap();
            s = &s + &(weights_cov[j] * diff_col.dot(&diff_row));
        }
        s = &s + &r;

        // Calculate cross-correlation
        let mut c = Array2::<f64>::zeros((n_states, n_measurements));
        for j in 0..predicted_sigmas.len() {
            let diff_x = &predicted_sigmas[j] - &x_pred;
            let diff_z = &measurement_sigmas[j] - &z_pred;
            let diff_x_col = diff_x
                .clone()
                .into_shape_with_order((diff_x.len(), 1))
                .unwrap();
            let diff_z_row = diff_z
                .clone()
                .into_shape_with_order((1, diff_z.len()))
                .unwrap();
            c = &c + &(weights_cov[j] * diff_x_col.dot(&diff_z_row));
        }

        // Kalman gain
        let k = match inv(&s.view(), None) {
            Ok(s_inv) => c.dot(&s_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // Update state and covariance
        let innovation = z.slice(s![i, ..]).to_owned() - z_pred;
        x = &x_pred + &k.dot(&innovation);
        p = &p_pred - &k.dot(&s).dot(&k.t());

        // Store state
        for j in 0..n_states {
            x_history[[i, j]] = x[j];
        }
    }

    Ok(x_history)
}

/// Generate sigma points for the unscented transform
///
/// # Arguments
///
/// * `x` - Current state estimate
/// * `p` - Current state covariance
/// * `gamma` - Scaling parameter
///
/// # Returns
///
/// * Vector of sigma points
fn generate_sigma_points(
    x: &Array1<f64>,
    p: &Array2<f64>,
    gamma: f64,
) -> SignalResult<Vec<Array1<f64>>> {
    let n = x.len();
    let mut sigma_points = Vec::with_capacity(2 * n + 1);

    // Add the mean as the first sigma point
    sigma_points.push(x.clone());

    // Calculate square root of covariance matrix using Cholesky decomposition
    let sqrt_p = match cholesky(&p.view(), None) {
        Ok(l) => l,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute Cholesky decomposition of covariance matrix".to_string(),
            ));
        }
    };

    // Generate remaining sigma points
    for j in 0..n {
        let col = sqrt_p.column(j).to_owned() * gamma;

        // x + sqrt(P) * gamma
        let sigma_plus = x + &col;
        sigma_points.push(sigma_plus);

        // x - sqrt(P) * gamma
        let sigma_minus = x - &col;
        sigma_points.push(sigma_minus);
    }

    Ok(sigma_points)
}

/// Ensemble Kalman filter (EnKF) for nonlinear systems
///
/// # Arguments
///
/// * `z` - Measurement signal
/// * `f_func` - State transition function (potentially stochastic)
/// * `h_func` - Measurement function
/// * `initial_x` - Initial state estimate
/// * `n_ensemble` - Number of ensemble members
/// * `config` - Kalman filter configuration
///
/// # Returns
///
/// * Estimated state at each time step
pub fn ensemble_kalman_filter<F, H>(
    z: &Array2<f64>,
    f_func: F,
    h_func: H,
    initial_x: Array1<f64>,
    n_ensemble: usize,
    config: &KalmanConfig,
) -> SignalResult<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    H: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n_samples = z.shape()[0];
    let n_states = initial_x.len();
    let n_measurements = z.shape()[1];

    // Initialize state covariance
    let initial_p = match &config.initial_p {
        Some(p0) => {
            if p0.shape()[0] != n_states || p0.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial covariance dimension mismatch".to_string(),
                ));
            }
            p0.clone()
        }
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Initialize ensemble
    let mut ensemble = Vec::with_capacity(n_ensemble);
    let mut rng = rand::rng();

    for _ in 0..n_ensemble {
        // Generate random perturbation based on initial covariance
        let mut perturbation = Array1::<f64>::zeros(n_states);
        for j in 0..n_states {
            perturbation[j] = rng.sample(rand_distr::StandardNormal);
        }

        // Apply Cholesky decomposition to ensure correct covariance structure
        let sqrt_p = match cholesky(&initial_p.view(), None) {
            Ok(l) => l,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute Cholesky decomposition of initial covariance".to_string(),
                ));
            }
        };

        let x_perturbed = &initial_x + &sqrt_p.dot(&perturbation);
        ensemble.push(x_perturbed);
    }

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => {
            if r.shape()[0] != n_measurements || r.shape()[1] != n_measurements {
                return Err(SignalError::DimensionMismatch(
                    "Measurement noise covariance dimension mismatch".to_string(),
                ));
            }
            r.clone()
        }
        None => Array2::<f64>::eye(n_measurements) * config.measurement_noise_scale,
    };

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n_samples, n_states));

    // Run Ensemble Kalman filter
    for i in 0..n_samples {
        // Forecast step
        for ensemble_item in ensemble.iter_mut().take(n_ensemble) {
            *ensemble_item = f_func(ensemble_item);
        }

        // Calculate ensemble mean
        let mut x_mean = Array1::<f64>::zeros(n_states);
        for e in &ensemble {
            x_mean = &x_mean + e;
        }
        x_mean /= n_ensemble as f64;

        // Calculate measured ensemble
        let mut measured_ensemble = Vec::with_capacity(n_ensemble);
        for e in &ensemble {
            measured_ensemble.push(h_func(e));
        }

        // Calculate measured ensemble mean
        let mut z_mean = Array1::<f64>::zeros(n_measurements);
        for me in &measured_ensemble {
            z_mean = &z_mean + me;
        }
        z_mean /= n_ensemble as f64;

        // Calculate covariance matrices
        let mut pxz = Array2::<f64>::zeros((n_states, n_measurements));
        let mut pzz = Array2::<f64>::zeros((n_measurements, n_measurements));

        for j in 0..n_ensemble {
            let x_diff = &ensemble[j] - &x_mean;
            let z_diff = &measured_ensemble[j] - &z_mean;

            let x_diff_col = x_diff
                .clone()
                .into_shape_with_order((x_diff.len(), 1))
                .unwrap();
            let z_diff_row = z_diff
                .clone()
                .into_shape_with_order((1, z_diff.len()))
                .unwrap();
            pxz = &pxz + &x_diff_col.dot(&z_diff_row);
            let z_diff_col = z_diff
                .clone()
                .into_shape_with_order((z_diff.len(), 1))
                .unwrap();
            let z_diff_row = z_diff
                .clone()
                .into_shape_with_order((1, z_diff.len()))
                .unwrap();
            pzz = &pzz + &z_diff_col.dot(&z_diff_row);
        }

        pxz /= (n_ensemble - 1) as f64;
        pzz /= (n_ensemble - 1) as f64;
        pzz = &pzz + &r;

        // Kalman gain
        let k = match inv(&pzz.view(), None) {
            Ok(pzz_inv) => pxz.dot(&pzz_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // Update ensemble members
        for j in 0..n_ensemble {
            // Generate perturbed observation
            let perturbed_z = &z.slice(s![i, ..]).to_owned()
                + &{
                    let std_dev = config.measurement_noise_scale.sqrt();
                    let mut noise = Array1::<f64>::zeros(n_measurements);
                    for k in 0..n_measurements {
                        noise[k] = rng.sample(rand_distr::Normal::new(0.0, std_dev).unwrap());
                    }
                    noise
                };

            // Update ensemble member
            let innovation = &perturbed_z - &measured_ensemble[j];
            ensemble[j] = &ensemble[j] + &k.dot(&innovation);
        }

        // Recalculate ensemble mean as state estimate
        let mut x_updated = Array1::<f64>::zeros(n_states);
        for e in &ensemble {
            x_updated = &x_updated + e;
        }
        x_updated /= n_ensemble as f64;

        // Store state
        for j in 0..n_states {
            x_history[[i, j]] = x_updated[j];
        }
    }

    Ok(x_history)
}

/// Apply Kalman filter to a 1D signal for denoising
///
/// # Arguments
///
/// * `signal` - Input signal to denoise
/// * `process_variance` - Process noise variance (default: 1e-5)
/// * `measurement_variance` - Measurement noise variance (default: 1e-1)
///
/// # Returns
///
/// * Denoised signal
pub fn kalman_denoise_1d(
    signal: &Array1<f64>,
    process_variance: Option<f64>,
    measurement_variance: Option<f64>,
) -> SignalResult<Array1<f64>> {
    let _n = signal.len();

    // Define a simple constant-velocity model
    let f = match Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]) {
        Ok(arr) => arr,
        Err(e) => {
            return Err(SignalError::InvalidArgument(format!(
                "Invalid shape: {}",
                e
            )))
        }
    };
    let h = match Array2::from_shape_vec((1, 2), vec![1.0, 0.0]) {
        Ok(arr) => arr,
        Err(e) => {
            return Err(SignalError::InvalidArgument(format!(
                "Invalid shape: {}",
                e
            )))
        }
    };

    // Configuration
    let config = KalmanConfig {
        process_noise_scale: process_variance.unwrap_or(1e-5),
        measurement_noise_scale: measurement_variance.unwrap_or(1e-1),
        ..KalmanConfig::default()
    };

    // Initial state (position and velocity)
    let initial_x = Array1::from_vec(vec![signal[0], 0.0]);

    // Apply Kalman filter
    let state_history = kalman_filter(signal, &f, &h, Some(initial_x), &config)?;

    // Extract position component (first state variable)
    let denoised = state_history.slice(s![.., 0]).to_owned();

    Ok(denoised)
}

/// Apply Kalman filter to a 2D signal (image) for denoising
///
/// # Arguments
///
/// * `image` - Input image to denoise
/// * `process_variance` - Process noise variance (default: 1e-5)
/// * `measurement_variance` - Measurement noise variance (default: 1e-1)
///
/// # Returns
///
/// * Denoised image
pub fn kalman_denoise_2d(
    image: &Array2<f64>,
    process_variance: Option<f64>,
    measurement_variance: Option<f64>,
) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();
    let mut denoised = Array2::<f64>::zeros((n_rows, n_cols));

    // Process each row independently
    for i in 0..n_rows {
        let row = image.slice(s![i, ..]).to_owned();
        let denoised_row = kalman_denoise_1d(&row, process_variance, measurement_variance)?;
        denoised.slice_mut(s![i, ..]).assign(&denoised_row);
    }

    // Process each column independently
    let mut column_denoised = Array2::<f64>::zeros((n_rows, n_cols));
    for j in 0..n_cols {
        let col = denoised.slice(s![.., j]).to_owned();
        let denoised_col = kalman_denoise_1d(&col, process_variance, measurement_variance)?;
        column_denoised.slice_mut(s![.., j]).assign(&denoised_col);
    }

    // Average the results from row and column processing
    let output = (&denoised + &column_denoised) / 2.0;

    Ok(output)
}

/// Apply Kalman filter to a 3D signal (color image) for denoising
///
/// # Arguments
///
/// * `image` - Input color image (shape: [height, width, channels])
/// * `process_variance` - Process noise variance (default: 1e-5)
/// * `measurement_variance` - Measurement noise variance (default: 1e-1)
/// * `joint_channels` - Whether to process color channels jointly (default: false)
///
/// # Returns
///
/// * Denoised color image
pub fn kalman_denoise_color(
    image: &Array3<f64>,
    process_variance: Option<f64>,
    measurement_variance: Option<f64>,
    joint_channels: bool,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();
    let mut output = Array3::<f64>::zeros((height, width, channels));

    if joint_channels {
        // Process each pixel as a vector measurement
        let f = Array2::<f64>::eye(channels) * 1.0;
        let h = Array2::<f64>::eye(channels) * 1.0;

        let config = KalmanConfig {
            process_noise_scale: process_variance.unwrap_or(1e-5),
            measurement_noise_scale: measurement_variance.unwrap_or(1e-1),
            ..KalmanConfig::default()
        };

        // Process each row
        for i in 0..height {
            for j in 0..width {
                // Create a time series of neighboring pixels horizontally
                let mut neighbors = Vec::new();
                for k in j.saturating_sub(2)..std::cmp::min(j + 3, width) {
                    neighbors.push(image.slice(s![i, k, ..]).to_owned());
                }

                // Create a measurement array
                let n_neighbors = neighbors.len();
                let mut z = Array2::<f64>::zeros((n_neighbors, channels));
                for (idx, neighbor) in neighbors.iter().enumerate() {
                    z.slice_mut(s![idx, ..]).assign(neighbor);
                }

                // Apply Kalman filter
                let initial_x = image.slice(s![i, j, ..]).to_owned();
                let state_history =
                    kalman_filter_vector_measurement(&z, &f, &h, Some(initial_x), &config)?;

                // Use the smoothed estimate for the center pixel
                let center_idx = std::cmp::min(2, n_neighbors - 1);
                output
                    .slice_mut(s![i, j, ..])
                    .assign(&state_history.slice(s![center_idx, ..]));
            }
        }

        // Process each column to further smooth the result
        let mut column_output = Array3::<f64>::zeros((height, width, channels));
        for j in 0..width {
            for i in 0..height {
                // Create a time series of neighboring pixels vertically
                let mut neighbors = Vec::new();
                for k in i.saturating_sub(2)..std::cmp::min(i + 3, height) {
                    neighbors.push(output.slice(s![k, j, ..]).to_owned());
                }

                // Create a measurement array
                let n_neighbors = neighbors.len();
                let mut z = Array2::<f64>::zeros((n_neighbors, channels));
                for (idx, neighbor) in neighbors.iter().enumerate() {
                    z.slice_mut(s![idx, ..]).assign(neighbor);
                }

                // Apply Kalman filter
                let initial_x = output.slice(s![i, j, ..]).to_owned();
                let state_history =
                    kalman_filter_vector_measurement(&z, &f, &h, Some(initial_x), &config)?;

                // Use the smoothed estimate for the center pixel
                let center_idx = std::cmp::min(2, n_neighbors - 1);
                column_output
                    .slice_mut(s![i, j, ..])
                    .assign(&state_history.slice(s![center_idx, ..]));
            }
        }

        // Average the results from horizontal and vertical processing
        output = (&output + &column_output) / 2.0;
    } else {
        // Process each channel independently
        for c in 0..channels {
            let channel = image.slice(s![.., .., c]).to_owned();
            let denoised_channel =
                kalman_denoise_2d(&channel, process_variance, measurement_variance)?;
            output.slice_mut(s![.., .., c]).assign(&denoised_channel);
        }
    }

    Ok(output)
}

/// Apply Kalman filter to a vector measurement
///
/// # Arguments
///
/// * `z` - Measurements (vector time series)
/// * `f` - State transition matrix
/// * `h` - Measurement matrix
/// * `initial_x` - Initial state (optional)
/// * `config` - Kalman filter configuration
///
/// # Returns
///
/// * Filtered state history
fn kalman_filter_vector_measurement(
    z: &Array2<f64>,
    f: &Array2<f64>,
    h: &Array2<f64>,
    initial_x: Option<Array1<f64>>,
    config: &KalmanConfig,
) -> SignalResult<Array2<f64>> {
    let n_samples = z.shape()[0];
    let n_states = f.shape()[0];
    let n_measurements = z.shape()[1];

    // Validate input dimensions
    if f.shape()[0] != f.shape()[1] {
        return Err(SignalError::DimensionMismatch(
            "State transition matrix F must be square".to_string(),
        ));
    }

    if h.shape()[0] != n_measurements || h.shape()[1] != n_states {
        return Err(SignalError::DimensionMismatch(
            "Measurement matrix H dimensions mismatch".to_string(),
        ));
    }

    // Initialize state estimate
    let mut x = match initial_x {
        Some(x0) => {
            if x0.len() != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial state dimension mismatch".to_string(),
                ));
            }
            x0
        }
        None => Array1::<f64>::zeros(n_states),
    };

    // Initialize state covariance
    let mut p = match &config.initial_p {
        Some(p0) => {
            if p0.shape()[0] != n_states || p0.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Initial covariance dimension mismatch".to_string(),
                ));
            }
            p0.clone()
        }
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Process noise covariance
    let q = match &config.q {
        Some(q) => {
            if q.shape()[0] != n_states || q.shape()[1] != n_states {
                return Err(SignalError::DimensionMismatch(
                    "Process noise covariance dimension mismatch".to_string(),
                ));
            }
            q.clone()
        }
        None => Array2::<f64>::eye(n_states) * config.process_noise_scale,
    };

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => {
            if r.shape()[0] != n_measurements || r.shape()[1] != n_measurements {
                return Err(SignalError::DimensionMismatch(
                    "Measurement noise covariance dimension mismatch".to_string(),
                ));
            }
            r.clone()
        }
        None => Array2::<f64>::eye(n_measurements) * config.measurement_noise_scale,
    };

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n_samples, n_states));

    // Run Kalman filter
    for i in 0..n_samples {
        // Predict step
        let x_pred = f.dot(&x);
        let p_pred = f.dot(&p).dot(&f.t()) + &q;

        // Update step
        let z_pred = h.dot(&x_pred);
        let innovation = z.slice(s![i, ..]).to_owned() - z_pred;
        let innovation_cov = h.dot(&p_pred).dot(&h.t()) + &r;

        // Kalman gain
        let k = match inv(&innovation_cov.view(), None) {
            Ok(inn_cov_inv) => p_pred.dot(&h.t()).dot(&inn_cov_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // State and covariance update
        x = &x_pred + &k.dot(&innovation);
        p = &p_pred - &k.dot(&innovation_cov).dot(&k.t());

        // Store state
        x_history.slice_mut(s![i, ..]).assign(&x);
    }

    Ok(x_history)
}

/// Estimate noise parameters from signal
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the window for local variance estimation
///
/// # Returns
///
/// * (process_variance, measurement_variance)
pub fn estimate_noise_parameters(
    signal: &Array1<f64>,
    window_size: usize,
) -> SignalResult<(f64, f64)> {
    let n = signal.len();

    if window_size >= n {
        return Err(SignalError::InvalidArgument(
            "Window size must be smaller than signal length".to_string(),
        ));
    }

    // Estimate measurement noise from high-frequency components
    let mut local_variances = Vec::new();

    for i in 0..n - window_size {
        let window = signal.slice(s![i..i + window_size]);
        let window_mean = window.mean().unwrap();
        let window_var = window
            .iter()
            .fold(0.0, |acc, &x| acc + (x - window_mean).powi(2))
            / window_size as f64;
        local_variances.push(window_var);
    }

    // Sort local variances and take the median as a robust estimate
    local_variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let measurement_var = local_variances[local_variances.len() / 2];

    // Estimate process noise from the signal's first difference (approximation of derivative)
    let mut diff = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        diff.push(signal[i + 1] - signal[i]);
    }

    let diff_mean = diff.iter().sum::<f64>() / diff.len() as f64;
    let process_var = diff
        .iter()
        .fold(0.0, |acc, &x| acc + (x - diff_mean).powi(2))
        / diff.len() as f64;

    // Heuristic adjustment: process variance is typically much smaller than measurement variance
    let adjusted_process_var = process_var.min(measurement_var) / 10.0;

    Ok((adjusted_process_var, measurement_var))
}

/// Apply an optimal-smoothing Kalman filter (RTS smoother)
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `f` - State transition matrix
/// * `h` - Measurement matrix
/// * `config` - Kalman filter configuration
///
/// # Returns
///
/// * Smoothed signal
pub fn kalman_smooth(
    signal: &Array1<f64>,
    f: &Array2<f64>,
    h: &Array2<f64>,
    config: &KalmanConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let n_states = f.shape()[0];

    // Forward pass (Kalman filter)
    let filtered_states = kalman_filter(signal, f, h, None, config)?;

    // Initialize smoothed states with the filtered states
    let mut smoothed_states = filtered_states.clone();

    // Extract state covariances from forward pass
    let mut filtered_covs = Vec::with_capacity(n);

    // Initialize state covariance
    let mut p = match &config.initial_p {
        Some(p0) => p0.clone(),
        None => Array2::<f64>::eye(n_states) * 1.0,
    };

    // Process noise covariance
    let q = match &config.q {
        Some(q) => q.clone(),
        None => Array2::<f64>::eye(n_states) * config.process_noise_scale,
    };

    // Measurement noise covariance
    let r = match &config.r {
        Some(r) => r.clone(),
        None => Array2::<f64>::eye(h.shape()[0]) * config.measurement_noise_scale,
    };

    // Recompute filtered covariances (we need them for smoothing)
    for i in 0..n {
        // Store the current covariance
        filtered_covs.push(p.clone());

        // Predict
        let x_pred = if i > 0 {
            f.dot(&filtered_states.slice(s![i - 1, ..]))
        } else {
            Array1::<f64>::zeros(n_states)
        };

        let p_pred = f.dot(&p).dot(&f.t()) + &q;

        // Update
        let z_pred = h.dot(&x_pred);
        let signal_value = signal.slice(s![i]).to_owned();
        let _innovation = &signal_value - &z_pred;
        let innovation_cov = h.dot(&p_pred).dot(&h.t()) + &r;

        // Kalman gain
        let k = match inv(&innovation_cov.view(), None) {
            Ok(inn_cov_inv) => p_pred.dot(&h.t()).dot(&inn_cov_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        // Update covariance
        p = &p_pred - &k.dot(&innovation_cov).dot(&k.t());
    }

    // Backward pass (RTS smoother)
    for i in (0..n - 1).rev() {
        // Predict next state and covariance
        let x_pred = f.dot(&filtered_states.slice(s![i, ..]));
        let p_pred = f.dot(&filtered_covs[i]).dot(&f.t()) + &q;

        // Smoother gain
        let g = match inv(&p_pred.view(), None) {
            Ok(p_pred_inv) => filtered_covs[i].dot(&f.t()).dot(&p_pred_inv),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert predicted covariance matrix".to_string(),
                ));
            }
        };

        // Smooth state
        let state_diff = &smoothed_states.slice(s![i + 1, ..]).to_owned() - &x_pred;
        smoothed_states
            .slice_mut(s![i, ..])
            .assign(&(&filtered_states.slice(s![i, ..]).to_owned() + &g.dot(&state_diff)));
    }

    // Extract the first component (assuming it's the actual signal)
    let smoothed_signal = smoothed_states.slice(s![.., 0]).to_owned();

    Ok(smoothed_signal)
}

/// Adaptive Kalman filtering for signals with time-varying noise characteristics
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `adaptive_window` - Window size for adaptive estimation
/// * `forgetting_factor` - Forgetting factor for exponential weighting (0 < factor ≤ 1)
///
/// # Returns
///
/// * Filtered signal
pub fn adaptive_kalman_filter(
    signal: &Array1<f64>,
    adaptive_window: usize,
    forgetting_factor: f64,
) -> SignalResult<Array1<f64>> {
    let _n = signal.len();

    // Simple constant position model
    let f = Array2::<f64>::eye(1) * 1.0;
    let h = Array2::<f64>::eye(1) * 1.0;

    // Configuration with adaptive settings
    let config = KalmanConfig {
        adaptive: true,
        adaptive_window,
        forgetting_factor,
        ..KalmanConfig::default()
    };

    // Apply adaptive Kalman filter
    let state_history = kalman_filter(signal, &f, &h, None, &config)?;

    // Extract filtered signal
    Ok(state_history.slice(s![.., 0]).to_owned())
}

/// Apply robust Kalman filter that can handle outliers
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `outlier_threshold` - Threshold for outlier detection (in sigma units)
///
/// # Returns
///
/// * Filtered signal
pub fn robust_kalman_filter(
    signal: &Array1<f64>,
    outlier_threshold: f64,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let n_states = 1;

    // Simple model
    let f = Array2::<f64>::eye(n_states) * 1.0;
    let h = Array2::<f64>::eye(n_states) * 1.0;

    // Initialize state and covariance
    let mut x = Array1::from_vec(vec![signal[0]]);
    let mut p = Array2::<f64>::eye(n_states) * 1.0;

    // Process and measurement noise
    let q = Array2::<f64>::eye(n_states) * 1e-4;
    let r = Array2::<f64>::eye(n_states) * 1e-2;

    // Prepare results
    let mut x_history = Array2::<f64>::zeros((n, n_states));

    // Run robust Kalman filter
    for i in 0..n {
        // Predict step
        let x_pred = f.dot(&x);
        let p_pred = f.dot(&p).dot(&f.t()) + &q;

        // Calculate innovation and innovation covariance
        let z_pred = h.dot(&x_pred);
        let signal_value = signal.slice(s![i]).to_owned();
        let innovation = &signal_value - &z_pred;
        let innovation_cov = h.dot(&p_pred).dot(&h.t()) + &r;

        // Calculate normalized innovation squared
        let s_inv = match inv(&innovation_cov.view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to invert innovation covariance matrix".to_string(),
                ));
            }
        };

        let nis = innovation.dot(&s_inv).dot(&innovation);

        // Check for outliers
        if nis > outlier_threshold * outlier_threshold {
            // Skip update for outliers
            x = x_pred;
        } else {
            // Regular Kalman update for inliers
            let k = p_pred.dot(&h.t()).dot(&s_inv);
            x = &x_pred + &k.dot(&innovation);
            p = &p_pred - &k.dot(&innovation_cov).dot(&k.t());
        }

        // Store state
        x_history.slice_mut(s![i, ..]).assign(&x);
    }

    // Extract filtered signal
    Ok(x_history.slice(s![.., 0]).to_owned())
}
